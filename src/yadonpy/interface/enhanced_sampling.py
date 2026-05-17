"""Enhanced-sampling preparation helpers for layer-stack interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..gmx.index import _write_ndx
from ..gmx.topology import SystemTopology, parse_system_top
from .postprocess import read_ndx_groups


@dataclass(frozen=True)
class SolvatedIonPullSpec:
    """Prepare a PLUMED pulling input for a solvated ion entering a target layer.

    The default is tailored to the Example 08 CMC-Na/carbonate/LiPF6 interface:
    select a Li+ whose solvent-oxygen coordination is closest to four, pull its
    z position toward the CMCNA layer COM, and monitor how solvent-O, CMC-O, and
    anion-F coordination numbers change along the path.
    """

    center_moltypes: Sequence[str] = ("Li", "LI")
    center_atom_names: Sequence[str] = ()
    target_group: str = "CMCNA"
    solvent_moltypes: Sequence[str] = ("EC", "EMC", "DEC")
    anion_moltypes: Sequence[str] = ("PF6",)
    solvent_ligand_elements: Sequence[str] = ("O",)
    target_ligand_elements: Sequence[str] = ("O",)
    anion_ligand_elements: Sequence[str] = ("F",)
    target_coordination_number: int = 4
    initial_coordination_cutoff_nm: float = 0.30
    cn_switch_r0_nm: float = 0.28
    cn_switch_nn: int = 6
    cn_switch_mm: int = 12
    pull_axis: Literal["x", "y", "z"] = "z"
    target_offset_nm: float = 0.0
    step0: int = 0
    step1: int = 500_000
    kappa0_kj_mol_nm2: float = 0.0
    kappa1_kj_mol_nm2: float = 1000.0
    print_stride: int = 100
    colvar_file: str = "COLVAR"


@dataclass(frozen=True)
class EnhancedSamplingPlan:
    """Files and metadata produced for a future enhanced-sampling run."""

    work_dir: Path
    plumed_dat: Path
    ndx_path: Path
    manifest_path: Path
    selected_center_atom: int
    target_group: str
    mdrun_extra_args: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SolvatedIonUmbrellaSpec(SolvatedIonPullSpec):
    """Prepare and run umbrella windows for a solvated ion entering CMC-Na."""

    print_stride: int = 1000
    window_count: int = 31
    steering_ns: float = 0.50
    window_equilibration_ns: float = 0.20
    window_production_ns: float = 1.00
    temperature_K: float = 318.15
    dt_ps: float = 0.001
    constraints: str = "none"
    umbrella_k_kj_mol_nm2: float = 1000.0
    pull_stride_ps: float = 1.0
    trajectory_stride_ps: float = 10.0
    energy_stride_ps: float = 1.0
    log_stride_ps: float = 10.0
    wham_skip_ps: float = 200.0
    wham_bins: int = 200
    pbc: str = "xyz"
    periodic_molecules: bool = True


@dataclass(frozen=True)
class UmbrellaSamplingPlan:
    """Prepared GROMACS/PLUMED umbrella PMF directory layout."""

    work_dir: Path
    selection_dir: Path
    steering_dir: Path
    windows_dir: Path
    wham_dir: Path
    postprocess_dir: Path
    source_gro: Path
    top_path: Path
    ndx_path: Path
    umbrella_ndx_path: Path
    manifest_path: Path
    selected_center_atom: int
    target_group: str
    initial_offset_nm: float
    target_offset_nm: float
    window_centers_nm: tuple[float, ...]
    windows: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class UmbrellaPmfResult:
    """PMF and coordination-analysis artifacts from umbrella sampling."""

    work_dir: Path
    summary_path: Path
    pmf_xvg: Path | None
    pmf_csv: Path | None
    histogram_xvg: Path | None
    histogram_csv: Path | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _AtomRecord:
    index: int
    moltype: str
    molecule_index: int
    atom_index_in_mol: int
    atomname: str
    atomtype: str
    charge: float
    mass: float
    coord_nm: tuple[float, float, float]


@dataclass(frozen=True)
class _SolvatedIonContext:
    system_dir: Path
    gro_path: Path
    top_path: Path
    ndx_path: Path
    manifest_path: Path | None
    box_nm: tuple[float, float, float]
    by_index: dict[int, _AtomRecord]
    selected: _AtomRecord
    target_group_key: str
    target_atoms: list[int]
    solvent_ligand_records: list[_AtomRecord]
    target_ligand_records: list[_AtomRecord]
    anion_ligand_records: list[_AtomRecord]
    center_rows: list[dict[str, Any]]
    target_com_nm: tuple[float, float, float]
    initial_axis_offset_nm: float


def _base_moltype(name: str) -> str:
    text = str(name).strip()
    text = re.sub(r"_\d{4}$", "", text)
    return text


def _token_matches(value: str, hints: Sequence[str]) -> bool:
    v = str(value).strip().lower()
    vb = _base_moltype(v).lower()
    for hint in hints or ():
        h = str(hint).strip().lower()
        if not h:
            continue
        if v == h or vb == h or v.startswith(h + "_") or vb.startswith(h + "_"):
            return True
    return False


def _element_symbol(record: _AtomRecord) -> str:
    for raw in (record.atomname, record.atomtype):
        text = re.sub(r"[^A-Za-z]", "", str(raw))
        if not text:
            continue
        if len(text) >= 2 and text[:2].lower() in {"li", "na", "cl", "br"}:
            return text[:2].title()
        return text[:1].upper()
    return ""


def _element_matches(record: _AtomRecord, elements: Sequence[str]) -> bool:
    wanted = {str(v).strip().title() for v in elements or () if str(v).strip()}
    return (not wanted) or _element_symbol(record) in wanted


def _read_gro_coords(gro_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {gro_path}")
    nat = int(lines[1].strip())
    coords = np.zeros((nat, 3), dtype=float)
    for i in range(nat):
        line = lines[2 + i]
        coords[i, 0] = float(line[20:28])
        coords[i, 1] = float(line[28:36])
        coords[i, 2] = float(line[36:44])
    box_parts = lines[2 + nat].split()
    box = tuple(float(v) for v in box_parts[:3])
    if len(box) != 3:
        raise ValueError(f"Could not read orthorhombic box from GRO file: {gro_path}")
    return coords, (float(box[0]), float(box[1]), float(box[2]))


def _expand_topology_atoms(topology: SystemTopology, coords: np.ndarray) -> list[_AtomRecord]:
    records: list[_AtomRecord] = []
    cursor = 0
    molecule_instance = 0
    for moltype, count in topology.molecules:
        mt = topology.moleculetypes.get(moltype)
        if mt is None:
            continue
        for _ in range(int(count)):
            molecule_instance += 1
            for local_idx in range(mt.natoms):
                if cursor >= coords.shape[0]:
                    raise ValueError("Topology atom count exceeds coordinate atom count.")
                records.append(
                    _AtomRecord(
                        index=cursor + 1,
                        moltype=str(moltype),
                        molecule_index=int(molecule_instance),
                        atom_index_in_mol=int(local_idx + 1),
                        atomname=str(mt.atomnames[local_idx]),
                        atomtype=str(mt.atomtypes[local_idx]),
                        charge=float(mt.charges[local_idx]),
                        mass=float(mt.masses[local_idx]),
                        coord_nm=tuple(float(v) for v in coords[cursor]),
                    )
                )
                cursor += 1
    if cursor != coords.shape[0]:
        raise ValueError(f"Topology/coordinate atom count mismatch: topology={cursor}, gro={coords.shape[0]}.")
    return records


def _minimum_image_distance_nm(a: _AtomRecord, b: _AtomRecord, box_nm: tuple[float, float, float]) -> float:
    delta = np.asarray(a.coord_nm, dtype=float) - np.asarray(b.coord_nm, dtype=float)
    box = np.asarray(box_nm, dtype=float)
    mask = box > 1.0e-12
    delta[mask] -= box[mask] * np.round(delta[mask] / box[mask])
    return float(np.linalg.norm(delta))


def _coordination_count(
    center: _AtomRecord,
    ligands: Sequence[_AtomRecord],
    *,
    cutoff_nm: float,
    box_nm: tuple[float, float, float],
) -> int:
    return int(
        sum(
            1
            for ligand in ligands
            if int(ligand.index) != int(center.index)
            and _minimum_image_distance_nm(center, ligand, box_nm) <= float(cutoff_nm)
        )
    )


def _center_score(row: Mapping[str, Any], target_cn: int) -> tuple[float, int, int, int]:
    solvent = int(row.get("solvent_ligand_count", 0))
    cmc = int(row.get("target_ligand_count", 0))
    anion = int(row.get("anion_ligand_count", 0))
    return (abs(solvent - int(target_cn)), -solvent, cmc, anion)


def _format_atoms(atoms: Sequence[int], *, per_line: int = 24) -> str:
    vals = [str(int(v)) for v in atoms]
    if len(vals) <= per_line:
        return ",".join(vals)
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append(",".join(vals[i : i + per_line]))
    return ",\\\n  ".join(lines)


def _prepare_solvated_ion_context(
    *,
    system_dir: Path,
    gro_path: Path,
    top_path: Path,
    ndx_path: Path,
    manifest_path: Path | None,
    spec: SolvatedIonPullSpec,
) -> _SolvatedIonContext:
    coords, box_nm = _read_gro_coords(gro_path)
    topology = parse_system_top(top_path)
    records = _expand_topology_atoms(topology, coords)
    by_index = {int(record.index): record for record in records}
    ndx_groups = read_ndx_groups(ndx_path) if ndx_path.is_file() else {}
    target_group_key = None
    for key in ndx_groups:
        if str(key).strip().upper() == str(spec.target_group).strip().upper():
            target_group_key = key
            break
    if target_group_key is None:
        raise ValueError(f"Target group {spec.target_group!r} was not found in {ndx_path}.")
    target_atoms = [int(v) for v in ndx_groups[target_group_key] if int(v) in by_index]
    if not target_atoms:
        raise ValueError(f"Target group {spec.target_group!r} contains no atoms present in the topology.")

    centers = [
        record
        for record in records
        if _token_matches(record.moltype, spec.center_moltypes)
        and (not spec.center_atom_names or _token_matches(record.atomname, spec.center_atom_names))
    ]
    if not centers:
        raise ValueError(f"No center atoms matched center_moltypes={tuple(spec.center_moltypes)!r}.")
    solvent_ligand_records = [
        record
        for record in records
        if _token_matches(record.moltype, spec.solvent_moltypes)
        and _element_matches(record, spec.solvent_ligand_elements)
    ]
    target_ligand_records = [
        by_index[idx]
        for idx in target_atoms
        if idx in by_index and _element_matches(by_index[idx], spec.target_ligand_elements)
    ]
    anion_ligand_records = [
        record
        for record in records
        if _token_matches(record.moltype, spec.anion_moltypes)
        and _element_matches(record, spec.anion_ligand_elements)
    ]

    center_rows: list[dict[str, Any]] = []
    for center in centers:
        center_rows.append(
            {
                "atom_index": int(center.index),
                "moltype": center.moltype,
                "molecule_index": int(center.molecule_index),
                "atomname": center.atomname,
                "z_nm": float(center.coord_nm[2]),
                "solvent_ligand_count": _coordination_count(
                    center,
                    solvent_ligand_records,
                    cutoff_nm=float(spec.initial_coordination_cutoff_nm),
                    box_nm=box_nm,
                ),
                "target_ligand_count": _coordination_count(
                    center,
                    target_ligand_records,
                    cutoff_nm=float(spec.initial_coordination_cutoff_nm),
                    box_nm=box_nm,
                ),
                "anion_ligand_count": _coordination_count(
                    center,
                    anion_ligand_records,
                    cutoff_nm=float(spec.initial_coordination_cutoff_nm),
                    box_nm=box_nm,
                ),
            }
        )
    center_rows = sorted(center_rows, key=lambda row: _center_score(row, int(spec.target_coordination_number)))
    selected = by_index[int(center_rows[0]["atom_index"])]
    target_coords = np.asarray([by_index[idx].coord_nm for idx in target_atoms if idx in by_index], dtype=float)
    target_com = np.mean(target_coords, axis=0)
    axis_i = {"x": 0, "y": 1, "z": 2}[str(spec.pull_axis).lower()]
    initial_axis_offset = float(np.asarray(selected.coord_nm, dtype=float)[axis_i] - target_com[axis_i])
    return _SolvatedIonContext(
        system_dir=system_dir,
        gro_path=gro_path,
        top_path=top_path,
        ndx_path=ndx_path,
        manifest_path=manifest_path,
        box_nm=box_nm,
        by_index=by_index,
        selected=selected,
        target_group_key=str(target_group_key),
        target_atoms=target_atoms,
        solvent_ligand_records=solvent_ligand_records,
        target_ligand_records=target_ligand_records,
        anion_ligand_records=anion_ligand_records,
        center_rows=center_rows,
        target_com_nm=tuple(float(v) for v in target_com),
        initial_axis_offset_nm=float(initial_axis_offset),
    )


def _write_plumed_cv_dat(
    path: Path,
    *,
    spec: SolvatedIonPullSpec,
    selected: _AtomRecord,
    target_atoms: Sequence[int],
    solvent_ligands: Sequence[int],
    target_ligands: Sequence[int],
    anion_ligands: Sequence[int],
) -> None:
    axis_component = {"x": "x", "y": "y", "z": "z"}[str(spec.pull_axis).lower()]
    args = [f"d_ion_target.{axis_component}"]
    lines = [
        "# Generated by YadonPy for CV recording during umbrella sampling.",
        "# Bias is applied by the GROMACS pull code, not by PLUMED.",
        "UNITS LENGTH=nm TIME=ps ENERGY=kj/mol",
        "",
        f"ion: GROUP ATOMS={int(selected.index)}",
        f"target_layer: GROUP ATOMS={_format_atoms(target_atoms)}",
        f"ion_com: COM ATOMS={int(selected.index)}",
        f"target_com: COM ATOMS={_format_atoms(target_atoms)}",
        "d_ion_target: DISTANCE ATOMS=ion_com,target_com COMPONENTS",
        "",
    ]
    if solvent_ligands:
        args.append("cn_solvent")
        lines.extend(
            [
                f"solvent_ligands: GROUP ATOMS={_format_atoms(solvent_ligands)}",
                "cn_solvent: COORDINATION "
                f"GROUPA=ion GROUPB=solvent_ligands R_0={float(spec.cn_switch_r0_nm):.4f} "
                f"NN={int(spec.cn_switch_nn)} MM={int(spec.cn_switch_mm)}",
                "",
            ]
        )
    if target_ligands:
        args.append("cn_target")
        lines.extend(
            [
                f"target_ligands: GROUP ATOMS={_format_atoms(target_ligands)}",
                "cn_target: COORDINATION "
                f"GROUPA=ion GROUPB=target_ligands R_0={float(spec.cn_switch_r0_nm):.4f} "
                f"NN={int(spec.cn_switch_nn)} MM={int(spec.cn_switch_mm)}",
                "",
            ]
        )
    if anion_ligands:
        args.append("cn_anion")
        lines.extend(
            [
                f"anion_ligands: GROUP ATOMS={_format_atoms(anion_ligands)}",
                "cn_anion: COORDINATION "
                f"GROUPA=ion GROUPB=anion_ligands R_0={float(spec.cn_switch_r0_nm):.4f} "
                f"NN={int(spec.cn_switch_nn)} MM={int(spec.cn_switch_mm)}",
                "",
            ]
        )
    lines.extend([f"PRINT STRIDE={int(spec.print_stride)} ARG={','.join(args)} FILE={spec.colvar_file}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plumed_dat(
    path: Path,
    *,
    spec: SolvatedIonPullSpec,
    selected: _AtomRecord,
    target_atoms: Sequence[int],
    solvent_ligands: Sequence[int],
    target_ligands: Sequence[int],
    anion_ligands: Sequence[int],
    initial_axis_offset_nm: float,
) -> None:
    axis_component = {"x": "x", "y": "y", "z": "z"}[str(spec.pull_axis).lower()]
    args = [f"d_ion_target.{axis_component}"]
    lines = [
        "# Generated by YadonPy for a solvated-ion pull into an interface layer.",
        "# Use with: gmx mdrun -deffnm md -plumed plumed.dat",
        "UNITS LENGTH=nm TIME=ps ENERGY=kj/mol",
        "",
        f"ion: GROUP ATOMS={int(selected.index)}",
        f"target_layer: GROUP ATOMS={_format_atoms(target_atoms)}",
        f"ion_com: COM ATOMS={int(selected.index)}",
        f"target_com: COM ATOMS={_format_atoms(target_atoms)}",
        "d_ion_target: DISTANCE ATOMS=ion_com,target_com COMPONENTS",
        "",
    ]
    if solvent_ligands:
        args.append("cn_solvent")
        lines.extend(
            [
                f"solvent_ligands: GROUP ATOMS={_format_atoms(solvent_ligands)}",
                "cn_solvent: COORDINATION "
                f"GROUPA=ion GROUPB=solvent_ligands R_0={float(spec.cn_switch_r0_nm):.4f} "
                f"NN={int(spec.cn_switch_nn)} MM={int(spec.cn_switch_mm)}",
                "",
            ]
        )
    if target_ligands:
        args.append("cn_target")
        lines.extend(
            [
                f"target_ligands: GROUP ATOMS={_format_atoms(target_ligands)}",
                "cn_target: COORDINATION "
                f"GROUPA=ion GROUPB=target_ligands R_0={float(spec.cn_switch_r0_nm):.4f} "
                f"NN={int(spec.cn_switch_nn)} MM={int(spec.cn_switch_mm)}",
                "",
            ]
        )
    if anion_ligands:
        args.append("cn_anion")
        lines.extend(
            [
                f"anion_ligands: GROUP ATOMS={_format_atoms(anion_ligands)}",
                "cn_anion: COORDINATION "
                f"GROUPA=ion GROUPB=anion_ligands R_0={float(spec.cn_switch_r0_nm):.4f} "
                f"NN={int(spec.cn_switch_nn)} MM={int(spec.cn_switch_mm)}",
                "",
            ]
        )
    lines.extend(
        [
            "pull_to_target: MOVINGRESTRAINT ...",
            f"  ARG=d_ion_target.{axis_component}",
            f"  STEP0={int(spec.step0)} AT0={float(initial_axis_offset_nm):.6f} KAPPA0={float(spec.kappa0_kj_mol_nm2):.6f}",
            f"  STEP1={int(spec.step1)} AT1={float(spec.target_offset_nm):.6f} KAPPA1={float(spec.kappa1_kj_mol_nm2):.6f}",
            "... MOVINGRESTRAINT",
            "",
            f"PRINT STRIDE={int(spec.print_stride)} ARG={','.join(args)} FILE={spec.colvar_file}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_solvated_ion_pull(
    *,
    system_dir: str | Path,
    gro_path: str | Path | None = None,
    top_path: str | Path | None = None,
    ndx_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    spec: SolvatedIonPullSpec | None = None,
) -> EnhancedSamplingPlan:
    """Prepare PLUMED input for pulling a solvated ion into a target layer.

    The helper is intentionally non-invasive: it does not launch MD.  It writes
    a PLUMED file, an audit index file, and a JSON manifest that records the
    selected ion and its initial coordination state.  Use the returned
    ``mdrun_extra_args`` with existing GROMACS workflow hooks when running the
    biased segment.
    """

    spec = spec or SolvatedIonPullSpec()
    system_dir = Path(system_dir).expanduser().resolve()
    gro_path = Path(gro_path).expanduser().resolve() if gro_path is not None else system_dir / "system.gro"
    top_path = Path(top_path).expanduser().resolve() if top_path is not None else system_dir / "system.top"
    ndx_path = Path(ndx_path).expanduser().resolve() if ndx_path is not None else system_dir / "system.ndx"
    if out_dir is None:
        out_dir = system_dir.parent / "07_enhanced_sampling" / "solvated_ion_pull"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    coords, box_nm = _read_gro_coords(gro_path)
    topology = parse_system_top(top_path)
    records = _expand_topology_atoms(topology, coords)
    by_index = {int(record.index): record for record in records}
    ndx_groups = read_ndx_groups(ndx_path) if ndx_path.is_file() else {}
    target_group_key = None
    for key in ndx_groups:
        if str(key).strip().upper() == str(spec.target_group).strip().upper():
            target_group_key = key
            break
    if target_group_key is None:
        raise ValueError(f"Target group {spec.target_group!r} was not found in {ndx_path}.")
    target_atoms = [int(v) for v in ndx_groups[target_group_key] if int(v) in by_index]
    if not target_atoms:
        raise ValueError(f"Target group {spec.target_group!r} contains no atoms present in the topology.")

    centers = [
        record
        for record in records
        if _token_matches(record.moltype, spec.center_moltypes)
        and (not spec.center_atom_names or _token_matches(record.atomname, spec.center_atom_names))
    ]
    if not centers:
        raise ValueError(f"No center atoms matched center_moltypes={tuple(spec.center_moltypes)!r}.")
    solvent_ligand_records = [
        record
        for record in records
        if _token_matches(record.moltype, spec.solvent_moltypes)
        and _element_matches(record, spec.solvent_ligand_elements)
    ]
    target_ligand_records = [
        by_index[idx]
        for idx in target_atoms
        if idx in by_index and _element_matches(by_index[idx], spec.target_ligand_elements)
    ]
    anion_ligand_records = [
        record
        for record in records
        if _token_matches(record.moltype, spec.anion_moltypes)
        and _element_matches(record, spec.anion_ligand_elements)
    ]

    center_rows: list[dict[str, Any]] = []
    for center in centers:
        solvent_count = _coordination_count(
            center,
            solvent_ligand_records,
            cutoff_nm=float(spec.initial_coordination_cutoff_nm),
            box_nm=box_nm,
        )
        target_count = _coordination_count(
            center,
            target_ligand_records,
            cutoff_nm=float(spec.initial_coordination_cutoff_nm),
            box_nm=box_nm,
        )
        anion_count = _coordination_count(
            center,
            anion_ligand_records,
            cutoff_nm=float(spec.initial_coordination_cutoff_nm),
            box_nm=box_nm,
        )
        center_rows.append(
            {
                "atom_index": int(center.index),
                "moltype": center.moltype,
                "molecule_index": int(center.molecule_index),
                "atomname": center.atomname,
                "z_nm": float(center.coord_nm[2]),
                "solvent_ligand_count": int(solvent_count),
                "target_ligand_count": int(target_count),
                "anion_ligand_count": int(anion_count),
            }
        )
    center_rows = sorted(center_rows, key=lambda row: _center_score(row, int(spec.target_coordination_number)))
    selected_index = int(center_rows[0]["atom_index"])
    selected = by_index[selected_index]
    target_coords = np.asarray([by_index[idx].coord_nm for idx in target_atoms if idx in by_index], dtype=float)
    target_com = np.mean(target_coords, axis=0)
    axis_i = {"x": 0, "y": 1, "z": 2}[str(spec.pull_axis).lower()]
    initial_axis_offset = float(np.asarray(selected.coord_nm, dtype=float)[axis_i] - target_com[axis_i])

    plumed_dat = out_dir / "plumed.dat"
    enhanced_ndx = out_dir / "enhanced_sampling.ndx"
    manifest_out = out_dir / "enhanced_sampling_manifest.json"
    _write_plumed_dat(
        plumed_dat,
        spec=spec,
        selected=selected,
        target_atoms=target_atoms,
        solvent_ligands=[record.index for record in solvent_ligand_records],
        target_ligands=[record.index for record in target_ligand_records],
        anion_ligands=[record.index for record in anion_ligand_records],
        initial_axis_offset_nm=initial_axis_offset,
    )
    _write_ndx(
        enhanced_ndx,
        [
            ("PULL_CENTER_ION", [int(selected.index)]),
            ("PULL_TARGET_LAYER", target_atoms),
            ("PULL_SOLVENT_LIGANDS", [record.index for record in solvent_ligand_records]),
            ("PULL_TARGET_LIGANDS", [record.index for record in target_ligand_records]),
            ("PULL_ANION_LIGANDS", [record.index for record in anion_ligand_records]),
        ],
    )

    metadata = {
        "schema_version": 1,
        "builder": "yadonpy.interface.enhanced_sampling.prepare_solvated_ion_pull",
        "inputs": {
            "system_dir": str(system_dir),
            "gro_path": str(gro_path),
            "top_path": str(top_path),
            "ndx_path": str(ndx_path),
            "manifest_path": None if manifest_path is None else str(Path(manifest_path).expanduser().resolve()),
        },
        "spec": asdict(spec),
        "selected_center": center_rows[0],
        "candidate_centers": center_rows,
        "target_group": str(target_group_key),
        "target_atom_count": int(len(target_atoms)),
        "ligand_atom_counts": {
            "solvent": int(len(solvent_ligand_records)),
            "target": int(len(target_ligand_records)),
            "anion": int(len(anion_ligand_records)),
        },
        "initial_axis_offset_nm": float(initial_axis_offset),
        "target_axis_offset_nm": float(spec.target_offset_nm),
        "plumed_dat": str(plumed_dat),
        "ndx_path": str(enhanced_ndx),
        "mdrun_extra_args": ["-plumed", str(plumed_dat)],
        "interpretation": {
            "selected_center_rule": "closest_initial_solvent_ligand_count_to_target_coordination_number",
            "coordination_cutoff_nm": float(spec.initial_coordination_cutoff_nm),
            "coordination_cvs": {
                "cn_solvent": "Li coordination to solvent ligand atoms",
                "cn_target": f"Li coordination to {target_group_key} ligand atoms",
                "cn_anion": "Li coordination to anion ligand atoms",
            },
            "pull_cv": f"d_ion_target.{spec.pull_axis}",
        },
    }
    manifest_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return EnhancedSamplingPlan(
        work_dir=out_dir,
        plumed_dat=plumed_dat,
        ndx_path=enhanced_ndx,
        manifest_path=manifest_out,
        selected_center_atom=int(selected.index),
        target_group=str(target_group_key),
        mdrun_extra_args=("-plumed", str(plumed_dat)),
        metadata=metadata,
    )


def _steps_from_ns(time_ns: float, dt_ps: float) -> int:
    return max(1, int(round(float(time_ns) * 1000.0 / max(float(dt_ps), 1.0e-12))))


def _nst_from_ps(interval_ps: float, dt_ps: float) -> int:
    return max(1, int(round(float(interval_ps) / max(float(dt_ps), 1.0e-12))))


def _axis_dim_vec(axis: str) -> tuple[str, str]:
    token = str(axis).strip().lower()
    if token == "x":
        return "Y N N", "1 0 0"
    if token == "y":
        return "N Y N", "0 1 0"
    if token == "z":
        return "N N Y", "0 0 1"
    raise ValueError("pull_axis must be x, y, or z.")


def _pull_mdp_block(
    *,
    spec: SolvatedIonUmbrellaSpec,
    init_nm: float,
    rate_nm_ps: float = 0.0,
) -> str:
    dim, vec = _axis_dim_vec(spec.pull_axis)
    stride = _nst_from_ps(float(spec.pull_stride_ps), float(spec.dt_ps))
    return "\n".join(
        [
            "; GROMACS pull-code umbrella coordinate.",
            "pull                      = yes",
            "pull-ngroups              = 2",
            "pull-ncoords              = 1",
            "pull-group1-name          = PULL_REFERENCE_CMCNA",
            "pull-group2-name          = PULL_CENTER_ION",
            "pull-coord1-type          = umbrella",
            "pull-coord1-geometry      = direction",
            "pull-coord1-groups        = 1 2",
            f"pull-coord1-dim           = {dim}",
            f"pull-coord1-vec           = {vec}",
            "pull-coord1-start         = no",
            f"pull-coord1-init          = {float(init_nm):.6f}",
            f"pull-coord1-rate          = {float(rate_nm_ps):.8f}",
            f"pull-coord1-k             = {float(spec.umbrella_k_kj_mol_nm2):.6f}",
            f"pull-nstxout              = {int(stride)}",
            f"pull-nstfout              = {int(stride)}",
            "",
        ]
    )


def _umbrella_mdp_text(
    *,
    spec: SolvatedIonUmbrellaSpec,
    time_ns: float,
    init_nm: float,
    rate_nm_ps: float = 0.0,
    gen_vel: str,
    continuation: str,
) -> str:
    from ..gmx.mdp_templates import MdpSpec, NVT_MDP, NVT_NO_CONSTRAINTS_MDP, default_mdp_params

    p = default_mdp_params()
    dt = float(spec.dt_ps)
    constraints = str(spec.constraints)
    p.update(
        {
            "nsteps": _steps_from_ns(time_ns, dt),
            "dt": dt,
            "ref_t": float(spec.temperature_K),
            "gen_temp": float(spec.temperature_K),
            "constraints": constraints,
            "pbc": str(spec.pbc),
            "periodic_molecules": "yes" if bool(spec.periodic_molecules) else "no",
            "gen_vel": str(gen_vel),
            "continuation": str(continuation),
            "nstxout": _nst_from_ps(float(spec.trajectory_stride_ps), dt),
            "nstxout_trr": 0,
            "nstvout": 0,
            "nstenergy": _nst_from_ps(float(spec.energy_stride_ps), dt),
            "nstlog": _nst_from_ps(float(spec.log_stride_ps), dt),
            "extra_mdp": _pull_mdp_block(spec=spec, init_nm=float(init_nm), rate_nm_ps=float(rate_nm_ps)),
        }
    )
    template = NVT_NO_CONSTRAINTS_MDP if constraints.strip().lower() == "none" else NVT_MDP
    return MdpSpec(template, p).render()


def _write_umbrella_lists(wham_dir: Path, windows: Sequence[dict[str, Any]]) -> dict[str, Path]:
    wham_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "tpr": wham_dir / "tpr-files.dat",
        "pullx": wham_dir / "pullx-files.dat",
        "pullf": wham_dir / "pullf-files.dat",
    }
    for key, out_path in files.items():
        lines = [str(Path(win[f"production_{key}"])) for win in windows]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return files


def _write_umbrella_ndx(path: Path, *, ctx: _SolvatedIonContext) -> None:
    system_atoms = sorted(int(idx) for idx in ctx.by_index)
    _write_ndx(
        path,
        [
            ("System", system_atoms),
            ("PULL_CENTER_ION", [int(ctx.selected.index)]),
            ("PULL_REFERENCE_CMCNA", ctx.target_atoms),
            ("PULL_TARGET_LAYER", ctx.target_atoms),
            ("PULL_SOLVENT_LIGANDS", [record.index for record in ctx.solvent_ligand_records]),
            ("PULL_TARGET_LIGANDS", [record.index for record in ctx.target_ligand_records]),
            ("PULL_ANION_LIGANDS", [record.index for record in ctx.anion_ligand_records]),
        ],
    )


def prepare_solvated_ion_umbrella(
    *,
    system_dir: str | Path,
    gro_path: str | Path | None = None,
    top_path: str | Path | None = None,
    ndx_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    spec: SolvatedIonUmbrellaSpec | None = None,
) -> UmbrellaSamplingPlan:
    """Prepare GROMACS pull-code umbrella windows and PLUMED CV inputs."""

    spec = spec or SolvatedIonUmbrellaSpec()
    if int(spec.window_count) < 2:
        raise ValueError("SolvatedIonUmbrellaSpec.window_count must be at least 2.")
    system_dir = Path(system_dir).expanduser().resolve()
    gro_path = Path(gro_path).expanduser().resolve() if gro_path is not None else system_dir / "system.gro"
    top_path = Path(top_path).expanduser().resolve() if top_path is not None else system_dir / "system.top"
    ndx_path = Path(ndx_path).expanduser().resolve() if ndx_path is not None else system_dir / "system.ndx"
    manifest_path_resolved = Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
    if out_dir is None:
        out_dir = system_dir.parent / "08_umbrella_sampling_pmf"
    work_dir = Path(out_dir).expanduser().resolve()
    selection_dir = work_dir / "02_solvated_li_selection"
    steering_dir = work_dir / "03_steering_pull"
    windows_dir = work_dir / "04_umbrella_windows"
    wham_dir = work_dir / "05_wham_pmf"
    postprocess_dir = work_dir / "06_postprocess"
    for directory in (selection_dir, steering_dir, windows_dir, wham_dir, postprocess_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ctx = _prepare_solvated_ion_context(
        system_dir=system_dir,
        gro_path=gro_path,
        top_path=top_path,
        ndx_path=ndx_path,
        manifest_path=manifest_path_resolved,
        spec=spec,
    )
    initial_offset = float(ctx.initial_axis_offset_nm)
    target_offset = float(spec.target_offset_nm)
    if abs(target_offset - initial_offset) < 1.0e-6:
        raise ValueError("Umbrella target_offset_nm is indistinguishable from the selected Li initial offset.")
    window_centers = np.linspace(initial_offset, target_offset, int(spec.window_count))

    umbrella_ndx = selection_dir / "umbrella_sampling.ndx"
    _write_umbrella_ndx(umbrella_ndx, ctx=ctx)
    selection_manifest = {
        "schema_version": 1,
        "selected_center": ctx.center_rows[0],
        "candidate_centers": ctx.center_rows,
        "target_group": ctx.target_group_key,
        "target_atom_count": int(len(ctx.target_atoms)),
        "target_com_nm": list(ctx.target_com_nm),
        "ligand_atom_counts": {
            "solvent": int(len(ctx.solvent_ligand_records)),
            "target": int(len(ctx.target_ligand_records)),
            "anion": int(len(ctx.anion_ligand_records)),
        },
        "initial_axis_offset_nm": initial_offset,
        "target_axis_offset_nm": target_offset,
    }
    (selection_dir / "solvated_li_selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    steering_rate = (target_offset - initial_offset) / max(float(spec.steering_ns) * 1000.0, 1.0e-12)
    (steering_dir / "steering.mdp").write_text(
        _umbrella_mdp_text(
            spec=spec,
            time_ns=float(spec.steering_ns),
            init_nm=initial_offset,
            rate_nm_ps=steering_rate,
            gen_vel="yes",
            continuation="no",
        ),
        encoding="utf-8",
    )
    _write_plumed_cv_dat(
        steering_dir / "plumed.dat",
        spec=spec,
        selected=ctx.selected,
        target_atoms=ctx.target_atoms,
        solvent_ligands=[record.index for record in ctx.solvent_ligand_records],
        target_ligands=[record.index for record in ctx.target_ligand_records],
        anion_ligands=[record.index for record in ctx.anion_ligand_records],
    )

    windows: list[dict[str, Any]] = []
    span = target_offset - initial_offset
    for idx, center in enumerate(window_centers.tolist()):
        win_dir = windows_dir / f"window_{idx:03d}"
        eq_dir = win_dir / "01_equilibration"
        prod_dir = win_dir / "02_production"
        eq_dir.mkdir(parents=True, exist_ok=True)
        prod_dir.mkdir(parents=True, exist_ok=True)
        window_plumed = prod_dir / "plumed.dat"
        _write_plumed_cv_dat(
            window_plumed,
            spec=spec,
            selected=ctx.selected,
            target_atoms=ctx.target_atoms,
            solvent_ligands=[record.index for record in ctx.solvent_ligand_records],
            target_ligands=[record.index for record in ctx.target_ligand_records],
            anion_ligands=[record.index for record in ctx.anion_ligand_records],
        )
        equil_mdp = win_dir / "equilibration.mdp"
        production_mdp = win_dir / "production.mdp"
        equil_mdp.write_text(
            _umbrella_mdp_text(
                spec=spec,
                time_ns=float(spec.window_equilibration_ns),
                init_nm=float(center),
                rate_nm_ps=0.0,
                gen_vel="yes",
                continuation="no",
            ),
            encoding="utf-8",
        )
        production_mdp.write_text(
            _umbrella_mdp_text(
                spec=spec,
                time_ns=float(spec.window_production_ns),
                init_nm=float(center),
                rate_nm_ps=0.0,
                gen_vel="no",
                continuation="yes",
            ),
            encoding="utf-8",
        )
        fraction = float((float(center) - initial_offset) / span) if abs(span) > 1.0e-12 else 0.0
        steering_time_ps = float(np.clip(fraction, 0.0, 1.0) * float(spec.steering_ns) * 1000.0)
        windows.append(
            {
                "index": int(idx),
                "center_nm": float(center),
                "steering_time_ps": steering_time_ps,
                "directory": str(win_dir),
                "start_gro": str(win_dir / "start.gro"),
                "equilibration_mdp": str(equil_mdp),
                "equilibration_dir": str(eq_dir),
                "equilibration_tpr": str(eq_dir / "md.tpr"),
                "production_mdp": str(production_mdp),
                "production_dir": str(prod_dir),
                "production_tpr": str(prod_dir / "md.tpr"),
                "production_pullx": str(prod_dir / "md_pullx.xvg"),
                "production_pullf": str(prod_dir / "md_pullf.xvg"),
                "production_colvar": str(prod_dir / str(spec.colvar_file)),
                "plumed_dat": str(window_plumed),
            }
        )

    wham_lists = _write_umbrella_lists(wham_dir, windows)
    window_table = selection_dir / "window_table.csv"
    with window_table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "center_nm", "steering_time_ps", "directory"])
        writer.writeheader()
        for win in windows:
            writer.writerow({k: win[k] for k in ("index", "center_nm", "steering_time_ps", "directory")})

    metadata = {
        "schema_version": 1,
        "builder": "yadonpy.interface.enhanced_sampling.prepare_solvated_ion_umbrella",
        "inputs": {
            "system_dir": str(system_dir),
            "gro_path": str(gro_path),
            "top_path": str(top_path),
            "ndx_path": str(ndx_path),
            "manifest_path": None if manifest_path_resolved is None else str(manifest_path_resolved),
        },
        "spec": asdict(spec),
        "selected_center": ctx.center_rows[0],
        "target_group": ctx.target_group_key,
        "initial_axis_offset_nm": initial_offset,
        "target_axis_offset_nm": target_offset,
        "window_centers_nm": [float(v) for v in window_centers.tolist()],
        "reaction_coordinate": {
            "definition": f"selected Li COM minus {ctx.target_group_key} COM along {spec.pull_axis}",
            "unit": "nm",
        },
        "paths": {
            "work_dir": str(work_dir),
            "selection_dir": str(selection_dir),
            "steering_dir": str(steering_dir),
            "windows_dir": str(windows_dir),
            "wham_dir": str(wham_dir),
            "postprocess_dir": str(postprocess_dir),
            "umbrella_ndx": str(umbrella_ndx),
            "window_table_csv": str(window_table),
            "wham_tpr_list": str(wham_lists["tpr"]),
            "wham_pullx_list": str(wham_lists["pullx"]),
            "wham_pullf_list": str(wham_lists["pullf"]),
        },
        "windows": windows,
        "notes": {
            "umbrella_bias": "GROMACS pull code",
            "plumed_role": "coordination CV recording only",
        },
    }
    manifest_out = selection_dir / "umbrella_sampling_manifest.json"
    manifest_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return UmbrellaSamplingPlan(
        work_dir=work_dir,
        selection_dir=selection_dir,
        steering_dir=steering_dir,
        windows_dir=windows_dir,
        wham_dir=wham_dir,
        postprocess_dir=postprocess_dir,
        source_gro=gro_path,
        top_path=top_path,
        ndx_path=ndx_path,
        umbrella_ndx_path=umbrella_ndx,
        manifest_path=manifest_out,
        selected_center_atom=int(ctx.selected.index),
        target_group=str(ctx.target_group_key),
        initial_offset_nm=initial_offset,
        target_offset_nm=target_offset,
        window_centers_nm=tuple(float(v) for v in window_centers.tolist()),
        windows=tuple(windows),
        metadata=metadata,
    )


def _load_umbrella_plan(value: UmbrellaSamplingPlan | str | Path) -> UmbrellaSamplingPlan:
    if isinstance(value, UmbrellaSamplingPlan):
        return value
    path = Path(value).expanduser().resolve()
    manifest = path if path.is_file() else path / "02_solvated_li_selection" / "umbrella_sampling_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    paths = data.get("paths") or {}
    windows = tuple(dict(win) for win in data.get("windows") or ())
    return UmbrellaSamplingPlan(
        work_dir=Path(paths["work_dir"]),
        selection_dir=Path(paths["selection_dir"]),
        steering_dir=Path(paths["steering_dir"]),
        windows_dir=Path(paths["windows_dir"]),
        wham_dir=Path(paths["wham_dir"]),
        postprocess_dir=Path(paths["postprocess_dir"]),
        source_gro=Path(data["inputs"]["gro_path"]),
        top_path=Path(data["inputs"]["top_path"]),
        ndx_path=Path(data["inputs"]["ndx_path"]),
        umbrella_ndx_path=Path(paths["umbrella_ndx"]),
        manifest_path=manifest,
        selected_center_atom=int((data.get("selected_center") or {}).get("atom_index")),
        target_group=str(data.get("target_group")),
        initial_offset_nm=float(data.get("initial_axis_offset_nm")),
        target_offset_nm=float(data.get("target_axis_offset_nm")),
        window_centers_nm=tuple(float(v) for v in data.get("window_centers_nm") or ()),
        windows=windows,
        metadata=data,
    )


def _path_exists_nonempty(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and p.stat().st_size > 0


def run_solvated_ion_umbrella(
    plan: UmbrellaSamplingPlan | str | Path,
    *,
    runner: Any | None = None,
    mpi: int = 1,
    omp: int = 14,
    gpu: int = 1,
    gpu_id: int | str | None = 0,
    restart: bool = True,
    run_steering: bool = True,
    run_windows: bool = True,
    run_wham: bool = True,
    analyze: bool = True,
) -> UmbrellaPmfResult | dict[str, Any]:
    """Run steering, umbrella windows, WHAM, and optional PMF post-processing."""

    from ..gmx.engine import GromacsRunner

    plan_obj = _load_umbrella_plan(plan)
    spec = SolvatedIonUmbrellaSpec(**(plan_obj.metadata.get("spec") or {}))
    runner = runner or GromacsRunner()
    run_log: dict[str, Any] = {"steering": {}, "windows": [], "wham": {}}
    use_gpu = bool(int(gpu))
    gpu_text = None if gpu_id is None else str(gpu_id)

    steering_tpr = plan_obj.steering_dir / "md.tpr"
    steering_gro = plan_obj.steering_dir / "md.gro"
    if run_steering:
        if not (restart and _path_exists_nonempty(steering_gro)):
            runner.grompp(
                mdp=plan_obj.steering_dir / "steering.mdp",
                gro=plan_obj.source_gro,
                top=plan_obj.top_path,
                ndx=plan_obj.umbrella_ndx_path,
                out_tpr=steering_tpr,
                cwd=plan_obj.steering_dir,
            )
            runner.mdrun(
                tpr=steering_tpr,
                deffnm="md",
                cwd=plan_obj.steering_dir,
                ntomp=int(omp),
                ntmpi=int(mpi),
                use_gpu=use_gpu,
                gpu_id=gpu_text,
                mdrun_extra_args=["-plumed", str(plan_obj.steering_dir / "plumed.dat")],
            )
        run_log["steering"] = {"tpr": str(steering_tpr), "gro": str(steering_gro)}

    steering_traj = plan_obj.steering_dir / "md.xtc"
    if not steering_traj.is_file():
        alt = plan_obj.steering_dir / "md.trr"
        steering_traj = alt if alt.is_file() else steering_traj

    for win in plan_obj.windows:
        win_dir = Path(win["directory"])
        start_gro = Path(win["start_gro"])
        if run_windows and not (restart and _path_exists_nonempty(start_gro)):
            if steering_traj.is_file() and steering_tpr.is_file():
                runner.run(
                    [
                        "trjconv",
                        "-s",
                        str(steering_tpr),
                        "-f",
                        str(steering_traj),
                        "-o",
                        str(start_gro),
                        "-dump",
                        f"{float(win['steering_time_ps']):.6f}",
                    ],
                    cwd=win_dir,
                    stdin_text="System\n",
                )
            else:
                shutil.copyfile(plan_obj.source_gro, start_gro)
        if not run_windows:
            continue
        eq_dir = Path(win["equilibration_dir"])
        prod_dir = Path(win["production_dir"])
        eq_gro = eq_dir / "md.gro"
        if not (restart and _path_exists_nonempty(eq_gro)):
            runner.grompp(
                mdp=Path(win["equilibration_mdp"]),
                gro=start_gro,
                top=plan_obj.top_path,
                ndx=plan_obj.umbrella_ndx_path,
                out_tpr=Path(win["equilibration_tpr"]),
                cwd=eq_dir,
            )
            runner.mdrun(
                tpr=Path(win["equilibration_tpr"]),
                deffnm="md",
                cwd=eq_dir,
                ntomp=int(omp),
                ntmpi=int(mpi),
                use_gpu=use_gpu,
                gpu_id=gpu_text,
            )
        prod_gro = prod_dir / "md.gro"
        if not (restart and _path_exists_nonempty(prod_gro)):
            runner.grompp(
                mdp=Path(win["production_mdp"]),
                gro=eq_gro,
                top=plan_obj.top_path,
                ndx=plan_obj.umbrella_ndx_path,
                out_tpr=Path(win["production_tpr"]),
                cpt=eq_dir / "md.cpt",
                cwd=prod_dir,
            )
            runner.mdrun(
                tpr=Path(win["production_tpr"]),
                deffnm="md",
                cwd=prod_dir,
                ntomp=int(omp),
                ntmpi=int(mpi),
                use_gpu=use_gpu,
                gpu_id=gpu_text,
                mdrun_extra_args=["-plumed", str(Path(win["plumed_dat"]))],
            )
        run_log["windows"].append({"index": int(win["index"]), "production_dir": str(prod_dir)})

    if run_wham:
        pmf_xvg = plan_obj.wham_dir / "pmf.xvg"
        hist_xvg = plan_obj.wham_dir / "histogram.xvg"
        if not (restart and _path_exists_nonempty(pmf_xvg) and _path_exists_nonempty(hist_xvg)):
            runner.run(
                [
                    "wham",
                    "-it",
                    str(plan_obj.wham_dir / "tpr-files.dat"),
                    "-ix",
                    str(plan_obj.wham_dir / "pullx-files.dat"),
                    "-o",
                    str(pmf_xvg),
                    "-hist",
                    str(hist_xvg),
                    "-unit",
                    "kJ",
                    "-b",
                    f"{float(spec.wham_skip_ps):.6f}",
                    "-bins",
                    str(int(spec.wham_bins)),
                ],
                cwd=plan_obj.wham_dir,
            )
        run_log["wham"] = {
            "pmf_xvg": str(pmf_xvg),
            "histogram_xvg": str(hist_xvg),
            "input_mode": "pullx",
        }

    run_log_path = plan_obj.work_dir / "umbrella_run_summary.json"
    run_log_path.write_text(json.dumps(run_log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if analyze:
        return analyze_umbrella_pmf(plan_obj)
    return run_log


def _xvg_to_csv(xvg_path: Path, csv_path: Path) -> tuple[Any | None, str | None]:
    if not Path(xvg_path).is_file():
        return None, "missing"
    try:
        from ..gmx.analysis.xvg import read_xvg

        data = read_xvg(Path(xvg_path))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        data.df.to_csv(csv_path, index=False)
        return data.df, None
    except Exception as exc:
        return None, str(exc)


def _read_colvar(path: Path) -> list[dict[str, float]]:
    if not Path(path).is_file():
        return []
    fields: list[str] | None = None
    rows: list[dict[str, float]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#!"):
            parts = line.split()
            if len(parts) >= 3 and parts[1].upper() == "FIELDS":
                fields = parts[2:]
            continue
        if line.startswith("#"):
            continue
        vals = line.split()
        if fields is None:
            fields = ["time", *[f"cv{i}" for i in range(1, len(vals))]]
        if len(vals) < len(fields):
            continue
        try:
            rows.append({name: float(vals[i]) for i, name in enumerate(fields)})
        except Exception:
            continue
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _plot_umbrella_outputs(
    *,
    pmf_df: Any | None,
    hist_df: Any | None,
    coord_rows: Sequence[Mapping[str, Any]],
    post_dir: Path,
) -> dict[str, str | None]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: dict[str, str | None] = {}
    if pmf_df is not None and len(pmf_df.columns) >= 2:
        x = pmf_df[pmf_df.columns[0]].to_numpy(dtype=float)
        y = pmf_df[pmf_df.columns[1]].to_numpy(dtype=float)
        out = post_dir / "pmf.svg"
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.plot(x, y, lw=2.0)
        ax.set_xlabel("Li-CMCNA COM z offset / nm")
        ax.set_ylabel("PMF / kJ mol$^{-1}$")
        ax.set_title("Umbrella PMF")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        outputs["pmf_svg"] = str(out)
    if hist_df is not None and len(hist_df.columns) >= 2:
        x = hist_df[hist_df.columns[0]].to_numpy(dtype=float)
        out = post_dir / "histogram_overlap.svg"
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for col in hist_df.columns[1:]:
            ax.plot(x, hist_df[col].to_numpy(dtype=float), lw=1.0, alpha=0.75)
        ax.set_xlabel("Li-CMCNA COM z offset / nm")
        ax.set_ylabel("umbrella histogram")
        ax.set_title("Umbrella window overlap")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        outputs["histogram_overlap_svg"] = str(out)
    if coord_rows:
        import pandas as pd

        df = pd.DataFrame(coord_rows)
        out = post_dir / "coordination_vs_reaction_coordinate.svg"
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for col, label in (("cn_solvent_mean", "solvent O"), ("cn_target_mean", "CMC O"), ("cn_anion_mean", "anion F")):
            if col in df:
                ax.plot(df["window_center_nm"], df[col], marker="o", ms=3.0, lw=1.5, label=label)
        ax.set_xlabel("Li-CMCNA COM z offset / nm")
        ax.set_ylabel("coordination number")
        ax.set_title("Li coordination along umbrella coordinate")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        outputs["coordination_svg"] = str(out)
        if pmf_df is not None and len(pmf_df.columns) >= 2:
            out2 = post_dir / "pmf_coordination_overlay.svg"
            fig, ax1 = plt.subplots(figsize=(6.6, 4.0))
            ax1.plot(pmf_df[pmf_df.columns[0]], pmf_df[pmf_df.columns[1]], color="black", lw=2.0, label="PMF")
            ax1.set_xlabel("Li-CMCNA COM z offset / nm")
            ax1.set_ylabel("PMF / kJ mol$^{-1}$")
            ax2 = ax1.twinx()
            for col, label in (("cn_solvent_mean", "solvent O"), ("cn_target_mean", "CMC O"), ("cn_anion_mean", "anion F")):
                if col in df:
                    ax2.plot(df["window_center_nm"], df[col], marker="o", ms=3.0, lw=1.2, label=label)
            ax2.set_ylabel("coordination number")
            ax1.grid(True, alpha=0.3)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize="small")
            fig.tight_layout()
            fig.savefig(out2)
            plt.close(fig)
            outputs["pmf_coordination_overlay_svg"] = str(out2)
    return outputs


def _histogram_overlap_rows(hist_df: Any | None) -> list[dict[str, Any]]:
    if hist_df is None or len(hist_df.columns) < 3:
        return []
    x = hist_df[hist_df.columns[0]].to_numpy(dtype=float)
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    rows: list[dict[str, Any]] = []
    cols = list(hist_df.columns[1:])
    normalized: list[np.ndarray] = []
    for col in cols:
        y = np.maximum(hist_df[col].to_numpy(dtype=float), 0.0)
        area = float(np.sum(y) * dx)
        normalized.append(y / max(area, 1.0e-12))
    for i in range(len(cols) - 1):
        overlap = float(np.sum(np.minimum(normalized[i], normalized[i + 1])) * dx)
        rows.append({"left_window": i, "right_window": i + 1, "overlap_fraction": overlap})
    return rows


def _write_umbrella_animation(
    *,
    pmf_df: Any | None,
    hist_df: Any | None,
    coord_rows: Sequence[Mapping[str, Any]],
    out_mp4: Path,
    fps: float = 1.0,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import animation
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as exc:
        return {"available": False, "mp4": None, "reason": f"matplotlib_unavailable:{exc.__class__.__name__}"}
    try:
        if not animation.writers.is_available("ffmpeg"):
            return {"available": False, "mp4": None, "reason": "ffmpeg_writer_unavailable"}
        writer = animation.FFMpegWriter(fps=max(1, int(round(float(fps)))), bitrate=1800)
    except Exception as exc:
        return {"available": False, "mp4": None, "reason": f"ffmpeg_writer_unavailable:{exc.__class__.__name__}"}
    try:
        import pandas as pd

        coord_df = pd.DataFrame(coord_rows)
    except Exception:
        coord_df = None
    frame_count = max(1, len(coord_rows), 1)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    def _update(frame_idx: int):
        ax_left.clear()
        ax_right.clear()
        if hist_df is not None and len(hist_df.columns) >= 2:
            x = hist_df[hist_df.columns[0]].to_numpy(dtype=float)
            upto = min(frame_idx + 1, len(hist_df.columns) - 1)
            for col in hist_df.columns[1 : 1 + upto]:
                ax_left.plot(x, hist_df[col].to_numpy(dtype=float), lw=0.9, alpha=0.55)
        if pmf_df is not None and len(pmf_df.columns) >= 2:
            ax_left.plot(
                pmf_df[pmf_df.columns[0]].to_numpy(dtype=float),
                pmf_df[pmf_df.columns[1]].to_numpy(dtype=float),
                color="black",
                lw=2.0,
                label="PMF",
            )
        ax_left.set_xlabel("Li-CMCNA COM z offset / nm")
        ax_left.set_ylabel("PMF / histogram")
        ax_left.set_title(f"Umbrella PMF and histograms ({frame_idx + 1}/{frame_count})")
        ax_left.grid(True, alpha=0.25)
        if coord_df is not None and not coord_df.empty:
            sub = coord_df.iloc[: frame_idx + 1]
            for col, label in (("cn_solvent_mean", "solvent O"), ("cn_target_mean", "CMC O"), ("cn_anion_mean", "anion F")):
                if col in sub:
                    ax_right.plot(sub["window_center_nm"], sub[col], marker="o", ms=3.0, lw=1.5, label=label)
        ax_right.set_xlabel("Li-CMCNA COM z offset / nm")
        ax_right.set_ylabel("coordination number")
        ax_right.set_title("Li coordination evolution")
        ax_right.grid(True, alpha=0.25)
        ax_right.legend(loc="best", fontsize="small")
        fig.tight_layout()

    ani = FuncAnimation(fig, _update, frames=frame_count, interval=1000)
    try:
        ani.save(out_mp4, writer=writer)
    except Exception as exc:
        plt.close(fig)
        return {"available": False, "mp4": None, "reason": f"mp4_write_failed:{exc.__class__.__name__}"}
    plt.close(fig)
    return {"available": True, "mp4": str(out_mp4)}


def analyze_umbrella_pmf(
    plan: UmbrellaSamplingPlan | str | Path,
    *,
    postprocess_dir: str | Path | None = None,
    fps: float = 1.0,
) -> UmbrellaPmfResult:
    """Parse WHAM PMF and PLUMED COLVAR files into standard PMF artifacts."""

    plan_obj = _load_umbrella_plan(plan)
    post_dir = Path(postprocess_dir).expanduser().resolve() if postprocess_dir is not None else plan_obj.postprocess_dir
    post_dir.mkdir(parents=True, exist_ok=True)
    pmf_xvg = plan_obj.wham_dir / "pmf.xvg"
    hist_xvg = plan_obj.wham_dir / "histogram.xvg"
    pmf_csv = post_dir / "pmf.csv"
    hist_csv = post_dir / "histogram.csv"
    pmf_df, pmf_error = _xvg_to_csv(pmf_xvg, pmf_csv)
    hist_df, hist_error = _xvg_to_csv(hist_xvg, hist_csv)

    merged_rows: list[dict[str, Any]] = []
    coord_rows: list[dict[str, Any]] = []
    for win in plan_obj.windows:
        colvar_path = Path(win["production_colvar"])
        rows = _read_colvar(colvar_path)
        for row in rows:
            payload = dict(row)
            payload["window_index"] = int(win["index"])
            payload["window_center_nm"] = float(win["center_nm"])
            merged_rows.append(payload)
        if rows:
            summary: dict[str, Any] = {
                "window_index": int(win["index"]),
                "window_center_nm": float(win["center_nm"]),
                "samples": int(len(rows)),
            }
            for key in ("d_ion_target.z", "d_ion_target.x", "d_ion_target.y", "cn_solvent", "cn_target", "cn_anion"):
                vals = [float(row[key]) for row in rows if key in row]
                if vals:
                    summary[f"{key.replace('.', '_')}_mean"] = float(np.mean(vals))
                    summary[f"{key.replace('.', '_')}_std"] = float(np.std(vals))
            coord_rows.append(summary)
        else:
            coord_rows.append(
                {
                    "window_index": int(win["index"]),
                    "window_center_nm": float(win["center_nm"]),
                    "samples": 0,
                }
            )

    merged_colvar_csv = post_dir / "merged_colvar.csv"
    coord_csv = post_dir / "coordination_by_window.csv"
    _write_rows(merged_colvar_csv, merged_rows)
    _write_rows(coord_csv, coord_rows)
    overlap_rows = _histogram_overlap_rows(hist_df)
    overlap_csv = post_dir / "histogram_overlap.csv"
    _write_rows(overlap_csv, overlap_rows)
    plot_outputs = _plot_umbrella_outputs(pmf_df=pmf_df, hist_df=hist_df, coord_rows=coord_rows, post_dir=post_dir)
    mp4 = _write_umbrella_animation(
        pmf_df=pmf_df,
        hist_df=hist_df,
        coord_rows=coord_rows,
        out_mp4=post_dir / "umbrella_pmf_timeseries.mp4",
        fps=float(fps),
    )
    summary = {
        "schema_version": 1,
        "workflow": "solvated_ion_umbrella_pmf",
        "plan_manifest": str(plan_obj.manifest_path),
        "selected_center_atom": int(plan_obj.selected_center_atom),
        "target_group": str(plan_obj.target_group),
        "window_count": int(len(plan_obj.windows)),
        "pmf": {
            "xvg": str(pmf_xvg) if pmf_xvg.is_file() else None,
            "csv": str(pmf_csv) if pmf_csv.is_file() else None,
            "available": pmf_df is not None,
            "error": pmf_error,
        },
        "histogram": {
            "xvg": str(hist_xvg) if hist_xvg.is_file() else None,
            "csv": str(hist_csv) if hist_csv.is_file() else None,
            "available": hist_df is not None,
            "error": hist_error,
            "overlap_csv": str(overlap_csv),
            "min_adjacent_overlap_fraction": (
                min(float(row["overlap_fraction"]) for row in overlap_rows) if overlap_rows else None
            ),
        },
        "coordination": {
            "merged_colvar_csv": str(merged_colvar_csv),
            "coordination_by_window_csv": str(coord_csv),
            "available": bool(merged_rows),
        },
        "plots": plot_outputs,
        "mp4": mp4,
        "artifacts": {
            "postprocess_dir": str(post_dir),
            "window_table_csv": str(plan_obj.selection_dir / "window_table.csv"),
            "summary_json": str(post_dir / "umbrella_pmf_summary.json"),
        },
    }
    summary_path = post_dir / "umbrella_pmf_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return UmbrellaPmfResult(
        work_dir=plan_obj.work_dir,
        summary_path=summary_path,
        pmf_xvg=pmf_xvg if pmf_xvg.is_file() else None,
        pmf_csv=pmf_csv if pmf_csv.is_file() else None,
        histogram_xvg=hist_xvg if hist_xvg.is_file() else None,
        histogram_csv=hist_csv if hist_csv.is_file() else None,
        metadata=summary,
    )


__all__ = [
    "EnhancedSamplingPlan",
    "SolvatedIonPullSpec",
    "SolvatedIonUmbrellaSpec",
    "UmbrellaPmfResult",
    "UmbrellaSamplingPlan",
    "analyze_umbrella_pmf",
    "prepare_solvated_ion_pull",
    "prepare_solvated_ion_umbrella",
    "run_solvated_ion_umbrella",
]
