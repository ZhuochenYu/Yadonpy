"""Enhanced-sampling preparation helpers for layer-stack interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
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
        "UNITS LENGTH=nm TIME=ps ENERGY=kJ/mol",
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


__all__ = [
    "EnhancedSamplingPlan",
    "SolvatedIonPullSpec",
    "prepare_solvated_ion_pull",
]
