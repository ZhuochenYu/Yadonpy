from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Geometry as Geom

from ..core import poly, utils
from ..core.graphite import GraphiteBuildResult, build_graphite, register_cell_species_metadata, stack_cell_blocks
from ..core.molspec import molecular_weight
from ..core.naming import get_name
from ..core.polyelectrolyte import detect_charged_groups
from ..core.workdir import workdir
from ..gmx.index import _write_ndx
from ..gmx.mdp_templates import MINIM_STEEP_MDP, NPT_MDP, NPT_NO_CONSTRAINTS_MDP, NVT_MDP, NVT_NO_CONSTRAINTS_MDP, MdpSpec, default_mdp_params
from ..gmx.workflows._util import RunResources
from ..gmx.workflows.eq import EqStage, EquilibrationJob
from ..io.gromacs_system import SystemExportResult, export_system_from_cell_meta
from ..io.mol2 import read_mol2_with_charges, write_mol2_from_top_gro_parmed
from ..runtime import resolve_restart
from ..sim import qm
from ..sim.preset.eq import _find_latest_equilibrated_gro
from .bulk_resize import build_bulk_equilibrium_profile, fixed_xy_semiisotropic_npt_overrides, read_equilibrated_box_nm
from . import builder as interface_builder
from .postprocess import read_ndx_groups
from .prep import equilibrate_bulk_with_eq21, plan_fixed_xy_direct_electrolyte_preparation


_AVOGADRO = 6.02214076e23


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    smiles: str
    charge_method: str = "RESP"
    bonded: str | None = None
    prefer_db: bool = False
    require_ready: bool = False
    use_ion_ff: bool = False
    charge_scale: float = 1.0


@dataclass(frozen=True)
class GraphiteSubstrateSpec:
    nx: int = 4
    ny: int = 4
    n_layers: int = 3
    edge_cap: str = "H"
    orientation: str = "basal"
    name: str = "GRAPH"
    top_padding_ang: float = 15.0


@dataclass(frozen=True)
class PolymerSlabSpec:
    name: str = "PEO"
    monomer_smiles: str = "*CCO*"
    monomers: tuple[MoleculeSpec, ...] = ()
    monomer_ratio: tuple[float, ...] = (1.0,)
    terminal_smiles: str = "[H][*]"
    terminal: MoleculeSpec | None = None
    chain_target_atoms: int = 280
    dp: int | None = None
    chain_count: int | None = None
    counterion: MoleculeSpec | None = None
    target_density_g_cm3: float = 1.10
    slab_z_nm: float = 3.6
    min_chain_count: int = 2
    tacticity: str = "atactic"
    charge_scale: float = 1.0
    initial_pack_z_scale: float = 1.18
    pack_retry: int = 30
    pack_retry_step: int = 2400
    pack_threshold_ang: float = 1.55
    pack_dec_rate: float = 0.72


@dataclass(frozen=True)
class ElectrolyteSlabSpec:
    solvents: tuple[MoleculeSpec, ...] = field(
        default_factory=lambda: (
            MoleculeSpec(name="DME", smiles="COCCOC"),
        )
    )
    salt_cation: MoleculeSpec = field(
        default_factory=lambda: MoleculeSpec(name="Li", smiles="[Li+]", use_ion_ff=True, charge_scale=0.8)
    )
    salt_anion: MoleculeSpec = field(
        default_factory=lambda: MoleculeSpec(
            name="FSI",
            smiles="FS(=O)(=O)[N-]S(=O)(=O)F",
            charge_scale=0.8,
        )
    )
    solvent_mass_ratio: tuple[float, ...] = (1.0,)
    target_density_g_cm3: float = 1.18
    slab_z_nm: float = 4.0
    salt_molarity_M: float = 1.0
    min_salt_pairs: int = 3
    initial_pack_density_g_cm3: float | None = None
    pack_retry: int = 30
    pack_retry_step: int = 2400
    pack_threshold_ang: float = 1.55
    pack_dec_rate: float = 0.72


@dataclass(frozen=True)
class SandwichRelaxationSpec:
    temperature_k: float = 300.0
    pressure_bar: float = 1.0
    mpi: int = 1
    omp: int = 8
    gpu: int = 1
    gpu_id: int | None = 0
    psi4_omp: int = 8
    psi4_memory_mb: int = 16000
    bulk_eq21_final_ns: float = 0.10
    bulk_additional_loops: int = 1
    bulk_eq21_exec_kwargs: dict[str, float] = field(default_factory=dict)
    graphite_to_polymer_gap_ang: float = 3.8
    polymer_to_electrolyte_gap_ang: float = 4.2
    top_padding_ang: float = 12.0
    stacked_pre_nvt_ps: float = 20.0
    stacked_z_relax_ps: float = 80.0
    stacked_exchange_ps: float = 120.0


@dataclass(frozen=True)
class SandwichPhaseReport:
    label: str
    box_nm: tuple[float, float, float]
    density_g_cm3: float
    species_names: tuple[str, ...]
    counts: tuple[int, ...]
    target_density_g_cm3: float | None = None


@dataclass(frozen=True)
class GraphitePolymerElectrolyteSandwichResult:
    graphite: GraphiteBuildResult
    polymer_phase: SandwichPhaseReport
    electrolyte_phase: SandwichPhaseReport
    stack_export: SystemExportResult
    relaxed_gro: Path
    manifest_path: Path
    stack_checks: dict[str, object] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ConfinedPhaseResult:
    label: str
    relaxed_block: object
    report: SandwichPhaseReport
    summary: dict[str, object]
    summary_path: Path
    top_path: Path
    gro_path: Path


def default_peo_polymer_spec(**kwargs) -> PolymerSlabSpec:
    return PolymerSlabSpec(**kwargs)


def default_peo_electrolyte_spec(**kwargs) -> ElectrolyteSlabSpec:
    return ElectrolyteSlabSpec(**kwargs)


def default_cmcna_polymer_spec(**kwargs) -> PolymerSlabSpec:
    base = PolymerSlabSpec(
        name="CMC",
        monomers=(
            MoleculeSpec(name="glucose_0", smiles="*OC1OC(CO)C(*)C(O)C1O"),
            MoleculeSpec(name="glucose_2", smiles="*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"),
            MoleculeSpec(name="glucose_3", smiles="*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"),
            MoleculeSpec(name="glucose_6", smiles="*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"),
        ),
        monomer_ratio=(12.0, 26.0, 27.0, 35.0),
        terminal=MoleculeSpec(name="CMC_terminal", smiles="[H][*]", require_ready=False, prefer_db=False),
        dp=60,
        target_density_g_cm3=1.45,
        slab_z_nm=4.2,
        min_chain_count=2,
        charge_scale=1.0,
        initial_pack_z_scale=1.24,
        pack_retry=60,
        pack_retry_step=3200,
        pack_threshold_ang=1.58,
        pack_dec_rate=0.70,
        counterion=MoleculeSpec(name="Na", smiles="[Na+]", use_ion_ff=True, charge_scale=1.0),
    )
    return replace(base, **kwargs)


def default_carbonate_lipf6_electrolyte_spec(**kwargs) -> ElectrolyteSlabSpec:
    base = ElectrolyteSlabSpec(
        solvents=(
            MoleculeSpec(name="EC", smiles="O=C1OCCO1"),
            MoleculeSpec(name="DEC", smiles="CCOC(=O)OCC"),
            MoleculeSpec(name="EMC", smiles="CCOC(=O)OC"),
        ),
        salt_cation=MoleculeSpec(name="Li", smiles="[Li+]", use_ion_ff=True, charge_scale=0.8),
        salt_anion=MoleculeSpec(name="PF6", smiles="F[P-](F)(F)(F)(F)F", bonded="DRIH", charge_scale=0.8, prefer_db=True, require_ready=True),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        target_density_g_cm3=1.32,
        slab_z_nm=4.8,
        salt_molarity_M=1.0,
        min_salt_pairs=6,
        initial_pack_density_g_cm3=0.88,
        pack_retry=40,
        pack_retry_step=2600,
        pack_threshold_ang=1.55,
        pack_dec_rate=0.70,
    )
    return replace(base, **kwargs)


def _estimate_chain_count(*, chain_mw: float, target_density_g_cm3: float, box_nm: tuple[float, float, float], minimum: int) -> int:
    volume_cm3 = float(box_nm[0] * box_nm[1] * box_nm[2]) * 1.0e-21
    target_mass_g = float(target_density_g_cm3) * volume_cm3
    target_mass_amu = target_mass_g * _AVOGADRO
    if chain_mw <= 0.0:
        return int(max(1, minimum))
    estimate = int(max(1, round(target_mass_amu / float(chain_mw))))
    return int(max(int(minimum), estimate))


def _smiles_formal_charge(smiles: str) -> int:
    mol = utils.mol_from_smiles(smiles, coord=False)
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def _is_polyelectrolyte_spec(spec: MoleculeSpec) -> bool:
    if "*" not in str(spec.smiles):
        return False
    try:
        mol = utils.mol_from_smiles(spec.smiles, coord=False)
    except Exception:
        return False
    summary = detect_charged_groups(mol, detection="auto")
    return bool(summary.get("groups"))


def _prepare_small_molecule(spec: MoleculeSpec, *, ff, ion_ff):
    if bool(spec.use_ion_ff):
        mol = ion_ff.mol(spec.smiles, name=spec.name)
        assigned = ion_ff.ff_assign(mol, report=False)
    else:
        mol = ff.mol(
            spec.smiles,
            name=spec.name,
            charge=spec.charge_method,
            require_ready=spec.require_ready,
            prefer_db=spec.prefer_db,
        )
        assigned = ff.ff_assign(mol, report=False, bonded=spec.bonded)
    if not assigned:
        raise RuntimeError(f"Cannot assign force field parameters for {spec.name} ({spec.smiles}).")
    return assigned


def _build_polymer_chain(*, ff, polymer: PolymerSlabSpec, relax: SandwichRelaxationSpec, chain_dir: Path):
    if polymer.monomers:
        monomer_specs = tuple(polymer.monomers)
    else:
        monomer_specs = (
            MoleculeSpec(
                name=f"{polymer.name}_monomer",
                smiles=polymer.monomer_smiles,
                require_ready=False,
                prefer_db=False,
            ),
        )
    if polymer.terminal is not None:
        terminal_spec = polymer.terminal
    else:
        terminal_spec = MoleculeSpec(
            name=f"{polymer.name}_terminal",
            smiles=polymer.terminal_smiles,
            require_ready=False,
            prefer_db=False,
        )

    monomers = []
    for spec in monomer_specs:
        monomer = ff.ff_assign(
            ff.mol(
                spec.smiles,
                name=spec.name,
                charge=spec.charge_method,
                require_ready=spec.require_ready,
                prefer_db=spec.prefer_db,
            ),
            report=False,
            bonded=spec.bonded,
            polyelectrolyte_mode=_is_polyelectrolyte_spec(spec),
        )
        if not monomer:
            raise RuntimeError(f"Cannot assign force field parameters for polymer monomer {spec.smiles}.")
        monomers.append(monomer)

    terminal = utils.mol_from_smiles(terminal_spec.smiles, name=terminal_spec.name)
    qm.assign_charges(
        terminal,
        charge=terminal_spec.charge_method,
        opt=True,
        work_dir=chain_dir,
        omp=int(relax.psi4_omp),
        memory=int(relax.psi4_memory_mb),
        log_name=None,
    )

    if polymer.dp is not None:
        dp = max(1, int(polymer.dp))
    else:
        if len(monomers) == 1:
            dp = max(1, int(poly.calc_n_from_num_atoms(monomers[0], int(polymer.chain_target_atoms), terminal1=terminal)))
        else:
            ratio = tuple(float(x) for x in polymer.monomer_ratio)
            dp = max(1, int(poly.calc_n_from_num_atoms(monomers, int(polymer.chain_target_atoms), ratio=ratio, terminal1=terminal)))

    rw_dir = chain_dir / "00_rw"
    term_dir = chain_dir / "01_term"
    if len(monomers) == 1:
        chain = poly.polymerize_rw(monomers[0], dp, tacticity=polymer.tacticity, work_dir=rw_dir)
    else:
        ratio = tuple(float(x) for x in polymer.monomer_ratio)
        if len(ratio) != len(monomers):
            raise ValueError("polymer.monomer_ratio must match polymer.monomers length")
        chain = poly.random_copolymerize_rw(
            monomers,
            dp,
            ratio=ratio,
            tacticity=polymer.tacticity,
            name=polymer.name,
            work_dir=rw_dir,
        )
    chain = poly.terminate_rw(chain, terminal, name=polymer.name, work_dir=term_dir)
    chain = ff.ff_assign(chain, report=False)
    if not chain:
        raise RuntimeError(f"Cannot assign force field parameters for polymer chain {polymer.name}.")
    return chain, dp


def _polymer_chain_formal_charge(mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def _prepare_polymer_phase_species(*, ff, ion_ff, polymer: PolymerSlabSpec, relax: SandwichRelaxationSpec, chain_dir: Path, box_nm: tuple[float, float, float]):
    polymer_chain, polymer_dp = _build_polymer_chain(ff=ff, polymer=polymer, relax=relax, chain_dir=chain_dir)
    if polymer.chain_count is not None:
        chain_count = max(1, int(polymer.chain_count))
    else:
        chain_count = _estimate_chain_count(
            chain_mw=float(molecular_weight(polymer_chain, strict=True)),
            target_density_g_cm3=float(polymer.target_density_g_cm3),
            box_nm=box_nm,
            minimum=int(polymer.min_chain_count),
        )

    species = [polymer_chain]
    counts = [int(chain_count)]
    charge_scale = [float(polymer.charge_scale)]
    notes: list[str] = []

    chain_formal_charge = int(_polymer_chain_formal_charge(polymer_chain))
    if chain_formal_charge != 0:
        if polymer.counterion is None:
            raise RuntimeError(
                f"Polymer {polymer.name} carries formal charge {chain_formal_charge} per chain but no counterion was configured."
            )
        counterion = _prepare_small_molecule(polymer.counterion, ff=ff, ion_ff=ion_ff)
        ion_charge = int(_smiles_formal_charge(polymer.counterion.smiles))
        if ion_charge == 0:
            raise RuntimeError(f"Configured polymer counterion {polymer.counterion.name} is neutral.")
        total_polymer_charge = int(chain_formal_charge * chain_count)
        if total_polymer_charge * ion_charge > 0:
            raise RuntimeError(
                f"Configured counterion {polymer.counterion.name} has the same charge sign as polymer {polymer.name}."
            )
        if abs(total_polymer_charge) % abs(ion_charge) != 0:
            raise RuntimeError(
                f"Polymer charge {total_polymer_charge} is not divisible by counterion charge {ion_charge} for {polymer.counterion.name}."
            )
        counterion_count = int(abs(total_polymer_charge) // abs(ion_charge))
        if counterion_count > 0:
            species.append(counterion)
            counts.append(counterion_count)
            charge_scale.append(float(polymer.counterion.charge_scale))
            notes.append(
                f"polymer formal charge per chain={chain_formal_charge}; added {counterion_count} {polymer.counterion.name} counterions to neutralize the slab"
            )

    return {
        "chain": polymer_chain,
        "dp": int(polymer_dp),
        "chain_count": int(chain_count),
        "species": species,
        "counts": counts,
        "charge_scale": charge_scale,
        "notes": tuple(notes),
    }


def _read_gro_z_coords(gro_path: Path) -> list[float]:
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid .gro file: {gro_path}")
    nat = int(lines[1].strip())
    z: list[float] = []
    for i in range(nat):
        raw = lines[2 + i]
        try:
            z.append(float(raw[36:44]))
        except Exception:
            z.append(float(raw[-8:]))
    return z


def _read_gro_box_nm(gro_path: Path) -> tuple[float, float, float]:
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid .gro file: {gro_path}")
    parts = lines[-1].split()
    if len(parts) < 3:
        raise ValueError(f"Invalid .gro box line: {gro_path}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _unwrap_phase_z(values: Sequence[float], *, box_z_nm: float) -> list[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or box_z_nm <= 0.0:
        return [float(x) for x in arr]
    if float(np.max(arr) - np.min(arr)) <= 0.5 * float(box_z_nm):
        return [float(x) for x in arr]
    ordered = np.sort(arr)
    cyclic = np.concatenate([ordered, ordered[:1] + float(box_z_nm)])
    gaps = np.diff(cyclic)
    split = int(np.argmax(gaps))
    if float(gaps[split]) <= 0.5 * float(box_z_nm):
        return [float(x) for x in arr]
    threshold = float(ordered[split])
    arr = np.where(arr <= threshold, arr + float(box_z_nm), arr)
    return [float(x) for x in arr]


def _build_stack_checks(*, gro_path: Path, ndx_groups: dict[str, list[int]]) -> dict[str, object]:
    z_coords = _read_gro_z_coords(gro_path)
    _box_x_nm, _box_y_nm, box_z_nm = _read_gro_box_nm(gro_path)
    payload: dict[str, object] = {"gro_path": str(gro_path)}
    phase_stats: dict[str, dict[str, float]] = {}
    for name in ("GRAPHITE", "POLYMER", "ELECTROLYTE"):
        members = [int(idx) for idx in ndx_groups.get(name, []) if 1 <= int(idx) <= len(z_coords)]
        if not members:
            continue
        values = _unwrap_phase_z([float(z_coords[idx - 1]) for idx in members], box_z_nm=float(box_z_nm))
        phase_stats[name] = {
            "min_z_nm": min(values),
            "mean_z_nm": sum(values) / float(len(values)),
            "max_z_nm": max(values),
        }
    payload["phases"] = phase_stats
    if len(phase_stats) == 3:
        observed = [name for name, _mean in sorted(((name, data["mean_z_nm"]) for name, data in phase_stats.items()), key=lambda item: item[1])]
        payload["observed_order"] = observed
        payload["expected_order"] = ["GRAPHITE", "POLYMER", "ELECTROLYTE"]
        payload["is_expected_order"] = observed == ["GRAPHITE", "POLYMER", "ELECTROLYTE"]
        payload["graphite_polymer_gap_nm"] = float(phase_stats["POLYMER"]["min_z_nm"] - phase_stats["GRAPHITE"]["max_z_nm"])
        payload["polymer_electrolyte_gap_nm"] = float(phase_stats["ELECTROLYTE"]["min_z_nm"] - phase_stats["POLYMER"]["max_z_nm"])
    return payload


def _append_group(groups: list[tuple[str, list[int]]], existing: dict[str, list[int]], name: str, members: Sequence[str]) -> None:
    merged: list[int] = []
    for member in members:
        merged.extend(int(idx) for idx in existing.get(str(member), []))
    merged_sorted = sorted(set(merged))
    if merged_sorted:
        groups.append((name, merged_sorted))


def _augment_sandwich_ndx(
    *,
    ndx_path: Path,
    graphite_name: str,
    polymer_name: str,
    electrolyte_names: Sequence[str],
) -> dict[str, list[int]]:
    existing = read_ndx_groups(ndx_path)
    merged_groups: list[tuple[str, list[int]]] = [(name, list(idxs)) for name, idxs in existing.items()]

    if "System" not in existing:
        system_atoms = sorted({int(idx) for idxs in existing.values() for idx in idxs})
        if system_atoms:
            merged_groups.insert(0, ("System", system_atoms))

    _append_group(merged_groups, existing, "GRAPHITE", [graphite_name, f"MOL_{graphite_name}"])
    _append_group(merged_groups, existing, "POLYMER", [polymer_name, f"MOL_{polymer_name}"])
    expanded = []
    for name in electrolyte_names:
        expanded.extend([name, f"MOL_{name}"])
    _append_group(merged_groups, existing, "ELECTROLYTE", expanded)
    _append_group(merged_groups, existing, "MOBILE", ["POLYMER", "ELECTROLYTE"])

    normalized = {name: list(idxs) for name, idxs in merged_groups}
    # MOBILE depends on the just-added phase groups.
    mobile = sorted(set(normalized.get("POLYMER", []) + normalized.get("ELECTROLYTE", [])))
    if mobile:
        normalized["MOBILE"] = mobile
        merged_groups = [(name, normalized[name]) for name in normalized]
    _write_ndx(ndx_path, merged_groups)
    return normalized


def _freeze_block(group_name: str) -> str:
    return "\n".join(
        (
            "; keep the graphite substrate frozen while polymer/electrolyte phases relax against it",
            f"freezegrps               = {group_name}",
            "freezedim                = Y Y Y",
        )
    )


def _sandwich_relaxation_stages(*, relax: SandwichRelaxationSpec, freeze_group: str) -> list[EqStage]:
    base = default_mdp_params()
    freeze = _freeze_block(freeze_group)
    fixed_xy = fixed_xy_semiisotropic_npt_overrides(pressure_bar=float(relax.pressure_bar))
    return [
        EqStage(
            "01_em",
            "minim",
            MdpSpec(
                MINIM_STEEP_MDP,
                {
                    **base,
                    "nsteps": 40000,
                    "emtol": 500.0,
                    "emstep": 0.001,
                    "extra_mdp": freeze,
                },
            ),
        ),
        EqStage(
            "02_pre_nvt",
            "nvt",
            MdpSpec(
                NVT_MDP,
                {
                    **base,
                    "dt": 0.001,
                    "nsteps": max(int(round(float(relax.stacked_pre_nvt_ps) / 0.001)), 1000),
                    "ref_t": float(relax.temperature_k),
                    "gen_temp": float(relax.temperature_k),
                    "gen_vel": "yes",
                    "extra_mdp": freeze,
                },
            ),
        ),
        EqStage(
            "03_z_relax",
            "npt",
            MdpSpec(
                NPT_MDP,
                {
                    **base,
                    **fixed_xy,
                    "dt": 0.001,
                    "nsteps": max(int(round(float(relax.stacked_z_relax_ps) / 0.001)), 1000),
                    "ref_t": float(relax.temperature_k),
                    "gen_temp": float(relax.temperature_k),
                    "gen_vel": "no",
                    "ref_p": fixed_xy["ref_p"],
                    "compressibility": fixed_xy["compressibility"],
                    "pcoupltype": fixed_xy["pcoupltype"],
                    "extra_mdp": freeze,
                },
            ),
        ),
        EqStage(
            "04_exchange",
            "npt",
            MdpSpec(
                NPT_MDP,
                {
                    **base,
                    **fixed_xy,
                    "dt": 0.002,
                    "nsteps": max(int(round(float(relax.stacked_exchange_ps) / 0.002)), 1000),
                    "ref_t": float(relax.temperature_k),
                    "gen_temp": float(relax.temperature_k),
                    "gen_vel": "no",
                    "ref_p": fixed_xy["ref_p"],
                    "compressibility": fixed_xy["compressibility"],
                    "pcoupltype": fixed_xy["pcoupltype"],
                    "extra_mdp": freeze,
                },
            ),
        ),
    ]


def _run_stacked_relaxation(
    *,
    export: SystemExportResult,
    work_dir: Path,
    relax: SandwichRelaxationSpec,
    freeze_group: str = "GRAPHITE",
    restart: bool | None = None,
) -> Path:
    stages = _sandwich_relaxation_stages(relax=relax, freeze_group=freeze_group)
    resources = RunResources(
        ntmpi=int(relax.mpi),
        ntomp=int(relax.omp),
        use_gpu=bool(relax.gpu),
        gpu_id=(str(relax.gpu_id) if relax.gpu_id is not None else None),
    )
    job = EquilibrationJob(
        gro=export.system_gro,
        top=export.system_top,
        ndx=export.system_ndx,
        provenance_ndx=export.system_ndx,
        out_dir=work_dir,
        stages=stages,
        resources=resources,
    )
    job.run(restart=bool(resolve_restart(restart)))
    return work_dir / stages[-1].name / "md.gro"


def _phase_report(*, label: str, counts: Sequence[int], mols: Sequence, work_dir: Path, target_density_g_cm3: float | None) -> SandwichPhaseReport:
    names = tuple(str(get_name(mol, default=f"{label}_{idx + 1}")) for idx, mol in enumerate(mols))
    profile = build_bulk_equilibrium_profile(
        counts=counts,
        mol_weights=[molecular_weight(mol, strict=True) for mol in mols],
        species_names=names,
        work_dir=work_dir,
    )
    return SandwichPhaseReport(
        label=str(label),
        box_nm=tuple(float(x) for x in profile.box_nm),
        density_g_cm3=float(profile.density_g_cm3),
        species_names=tuple(profile.species_names),
        counts=tuple(profile.counts),
        target_density_g_cm3=(None if target_density_g_cm3 is None else float(target_density_g_cm3)),
    )


def _covered_lateral_replicas(
    *,
    source_box_nm: tuple[float, float, float],
    target_lengths_nm: tuple[float, float],
    max_lateral_strain: float = 0.12,
) -> tuple[int, int]:
    reps: list[int] = []
    for src, target in zip(source_box_nm[:2], target_lengths_nm):
        src_len = max(float(src), 1.0e-9)
        target_len = max(float(target), 0.0)
        ceil_rep = max(1, int(math.ceil(target_len / src_len)))
        best_rep = ceil_rep
        for rep in range(1, ceil_rep + 1):
            strain = abs(target_len / (src_len * float(rep)) - 1.0)
            if strain <= float(max_lateral_strain):
                best_rep = rep
                break
        reps.append(int(best_rep))
    return int(reps[0]), int(reps[1])


def _initial_bulk_pack_density(
    *,
    target_density_g_cm3: float,
    phase: str,
    requested_density_g_cm3: float | None = None,
) -> float:
    if requested_density_g_cm3 is not None and float(requested_density_g_cm3) > 0.0:
        return float(requested_density_g_cm3)
    phase_key = str(phase).strip().lower()
    target = float(target_density_g_cm3)
    if phase_key == "polymer":
        return max(0.50, min(0.75, target * 0.60))
    return max(0.65, min(0.90, target * 0.80))


def _prepare_slab_from_equilibrated_bulk(
    *,
    label: str,
    bulk_work_dir: Path,
    target_lengths_nm: tuple[float, float],
    target_thickness_nm: float,
    out_dir: Path,
    restart: bool | None = None,
    surface_shell_nm: float = 0.80,
    core_guard_nm: float = 0.50,
    max_lateral_strain: float = 0.12,
):
    out_dir = Path(out_dir)
    snapshot_builder = interface_builder.InterfaceBuilder(work_dir=out_dir / "00_snapshot", restart=restart)
    source = snapshot_builder.bulk_source(name=label, work_dir=bulk_work_dir)
    source_box_nm = read_equilibrated_box_nm(gro_path=source.representative_gro)
    replicas_xy = _covered_lateral_replicas(
        source_box_nm=source_box_nm,
        target_lengths_nm=target_lengths_nm,
        max_lateral_strain=float(max_lateral_strain),
    )
    spec = interface_builder.SlabBuildSpec(
        axis="Z",
        target_thickness_nm=float(target_thickness_nm),
        surface_shell_nm=float(surface_shell_nm),
        core_guard_nm=float(core_guard_nm),
        prefer_densest_window=True,
        lateral_recentering=True,
    )
    prepared = interface_builder._prepare_slab(
        source=source,
        spec=spec,
        route="route_a",
        name=str(label),
        out_dir=out_dir / "01_slab",
        target_lengths_nm=(float(target_lengths_nm[0]), float(target_lengths_nm[1])),
        replicas_xy=replicas_xy,
        target_thickness_nm=float(target_thickness_nm),
        area_policy=interface_builder.AreaMismatchPolicy(
            reference_side="bottom",
            max_lateral_strain=float(max_lateral_strain),
        ),
    )
    note = (
        f"{label} slab was cut from equilibrated bulk snapshot {source.representative_gro.name} "
        f"with replicas_xy={replicas_xy} to match target footprint "
        f"({float(target_lengths_nm[0]):.3f}, {float(target_lengths_nm[1]):.3f}) nm"
    )
    return prepared, note


def _molecule_atom_blocks(*, species: Sequence, counts: Sequence[int]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    cursor = 0
    for mol, count in zip(species, counts):
        nat = int(mol.GetNumAtoms())
        for _ in range(int(count)):
            blocks.append((cursor, cursor + nat))
            cursor += nat
    return blocks


def _compact_packed_cell_z_by_molecule_centers(
    *,
    cell,
    species: Sequence,
    counts: Sequence[int],
    target_box_nm: tuple[float, float, float],
) -> tuple[object, str | None]:
    if cell is None or int(getattr(cell, "GetNumConformers", lambda: 0)()) <= 0:
        return cell, None

    blocks = _molecule_atom_blocks(species=species, counts=counts)
    if not blocks or int(blocks[-1][1]) != int(cell.GetNumAtoms()):
        return cell, None

    conf = cell.GetConformer(0)
    coords = np.asarray(conf.GetPositions(), dtype=float).copy()
    if coords.size == 0:
        return cell, None

    current_box = getattr(cell, "cell", None)
    if current_box is None:
        return cell, None

    old_zlo = float(current_box.zlo)
    old_zhi = float(current_box.zhi)
    old_z_len = max(float(old_zhi - old_zlo), 1.0e-6)
    target_x_ang = float(target_box_nm[0]) * 10.0
    target_y_ang = float(target_box_nm[1]) * 10.0
    target_z_ang = float(target_box_nm[2]) * 10.0
    fragment_spans = [float(np.max(coords[start:stop, 2]) - np.min(coords[start:stop, 2])) for start, stop in blocks]
    max_fragment_span = max(fragment_spans) if fragment_spans else 0.0
    prefit_z_ang = max(target_z_ang * 1.5, target_z_ang + max_fragment_span * 0.8)
    prefit_z_ang = min(old_z_len, prefit_z_ang)
    if target_z_ang <= 0.0 or old_z_len <= prefit_z_ang * 1.10:
        return cell, None

    old_center = 0.5 * (old_zlo + old_zhi)
    new_center = 0.5 * prefit_z_ang
    scale = prefit_z_ang / old_z_len
    if scale >= 0.999:
        return cell, None

    remapped = coords.copy()
    for start, stop in blocks:
        frag = coords[start:stop, 2]
        frag_center = float(np.mean(frag))
        new_frag_center = (frag_center - old_center) * scale + new_center
        remapped[start:stop, 2] = frag + (new_frag_center - frag_center)

    new_min = float(np.min(remapped[:, 2]))
    new_max = float(np.max(remapped[:, 2]))
    span = max(new_max - new_min, 1.0e-6)
    if span > prefit_z_ang:
        anchor = float(np.min(coords[:, 2]))
        atom_scale = prefit_z_ang / span
        remapped[:, 2] = (remapped[:, 2] - anchor) * atom_scale
        new_min = float(np.min(remapped[:, 2]))
        new_max = float(np.max(remapped[:, 2]))

    shift = new_center - 0.5 * (new_min + new_max)
    remapped[:, 2] += shift
    new_min = float(np.min(remapped[:, 2]))
    new_max = float(np.max(remapped[:, 2]))
    if new_min < 0.0:
        remapped[:, 2] -= new_min
    if new_max > prefit_z_ang:
        remapped[:, 2] -= (new_max - prefit_z_ang)

    for idx, xyz in enumerate(remapped):
        conf.SetAtomPosition(idx, Geom.Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    setattr(cell, "cell", utils.Cell(target_x_ang, 0.0, target_y_ang, 0.0, prefit_z_ang, 0.0))
    poly.set_cell_param_conf(cell, 0, target_x_ang, 0.0, target_y_ang, 0.0, prefit_z_ang, 0.0)
    note = (
        "polymer slab pack expanded along z during placement; remapped molecule centers "
        f"from {old_z_len / 10.0:.3f} nm to a pre-relaxation {prefit_z_ang / 10.0:.3f} nm "
        f"before EQ21 (target slab thickness {target_box_nm[2]:.3f} nm)"
    )
    return cell, note


def _confined_phase_durations_ps(relax: SandwichRelaxationSpec) -> tuple[float, float]:
    pre_nvt_ps = max(4.0, 0.5 * float(relax.stacked_pre_nvt_ps))
    density_relax_ps = max(12.0, 0.25 * float(relax.stacked_z_relax_ps))
    return float(pre_nvt_ps), float(density_relax_ps)


def _phase_wall_block(*, wall_atomtype: str, wall_mode: str = "12-6", wall_r_linpot_nm: float = 0.05) -> str:
    return "\n".join(
        (
            "nwall                    = 2",
            f"wall_type                = {str(wall_mode)}",
            f"wall_atomtype            = {str(wall_atomtype)} {str(wall_atomtype)}",
            "ewald-geometry           = 3dc",
            f"wall-r-linpot            = {float(wall_r_linpot_nm):.6g}",
        )
    )


def _phase_confined_relaxation_stages(
    *,
    relax: SandwichRelaxationSpec,
    wall_atomtype: str,
) -> list[EqStage]:
    base = default_mdp_params()
    pre_nvt_ps, density_relax_ps = _confined_phase_durations_ps(relax)
    fixed_xy = fixed_xy_semiisotropic_npt_overrides(pressure_bar=float(relax.pressure_bar))
    wall_mdp = _phase_wall_block(wall_atomtype=wall_atomtype)
    common = {
        **base,
        "pbc": "xy",
        "periodic_molecules": "yes",
        "periodic-molecules": "yes",
        "wall_mdp": wall_mdp,
    }
    return [
        EqStage(
            "01_em",
            "minim",
            MdpSpec(
                MINIM_STEEP_MDP,
                {
                    **common,
                    "nsteps": 60000,
                    "emtol": 500.0,
                    "emstep": 0.001,
                    "extra_mdp": "",
                },
            ),
        ),
        EqStage(
            "02_pre_nvt",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                {
                    **common,
                    "dt": 0.001,
                    "nsteps": max(int(round(float(pre_nvt_ps) / 0.001)), 2000),
                    "ref_t": float(relax.temperature_k),
                    "gen_temp": float(relax.temperature_k),
                    "gen_vel": "yes",
                    "extra_mdp": "",
                },
            ),
        ),
        EqStage(
            "03_density_relax",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                {
                    **common,
                    **fixed_xy,
                    "dt": 0.001,
                    "nsteps": max(int(round(float(density_relax_ps) / 0.001)), 4000),
                    "ref_t": float(relax.temperature_k),
                    "gen_temp": float(relax.temperature_k),
                    "gen_vel": "no",
                    "ref_p": fixed_xy["ref_p"],
                    "compressibility": fixed_xy["compressibility"],
                    "pcoupltype": fixed_xy["pcoupltype"],
                    "extra_mdp": "",
                },
            ),
        ),
    ]


def _gro_positions_nm(gro_path: Path) -> list[tuple[float, float, float]]:
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid .gro file: {gro_path}")
    nat = int(lines[1].strip())
    out: list[tuple[float, float, float]] = []
    for i in range(nat):
        raw = lines[2 + i]
        out.append((float(raw[20:28]), float(raw[28:36]), float(raw[36:44])))
    return out


def _rebox_block_for_phase_confinement(
    *,
    block,
    target_xy_nm: tuple[float, float],
    target_thickness_nm: float,
    vacuum_padding_ang: float,
    species: Sequence | None = None,
    counts: Sequence[int] | None = None,
):
    confined = utils.deepcopy_mol(block)
    conf = confined.GetConformer(0)
    coords = np.asarray(conf.GetPositions(), dtype=float).copy()
    if coords.size == 0:
        raise RuntimeError("Cannot confine an empty slab block.")
    periodic_lateral_wrap_applied = False

    if species is not None and counts is not None:
        try:
            blocks = _molecule_atom_blocks(species=species, counts=counts)
        except Exception:
            blocks = []
        if blocks and int(blocks[-1][1]) == int(confined.GetNumAtoms()):
            box_x_ang = float(target_xy_nm[0]) * 10.0
            box_y_ang = float(target_xy_nm[1]) * 10.0
            for start, stop in blocks:
                frag = coords[start:stop]
                if frag.size == 0:
                    continue
                center_x = float(np.mean(frag[:, 0]))
                center_y = float(np.mean(frag[:, 1]))
                if box_x_ang > 0.0:
                    frag[:, 0] += box_x_ang * math.floor((0.5 * box_x_ang - center_x) / box_x_ang)
                if box_y_ang > 0.0:
                    frag[:, 1] += box_y_ang * math.floor((0.5 * box_y_ang - center_y) / box_y_ang)
                coords[start:stop] = frag

    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    spans = maxs - mins
    target_x_ang = float(target_xy_nm[0]) * 10.0
    target_y_ang = float(target_xy_nm[1]) * 10.0
    slot_z_ang = max(float(target_thickness_nm) * 10.0, float(spans[2]))
    box_z_ang = slot_z_ang + 2.0 * float(vacuum_padding_ang)

    if float(spans[0]) > target_x_ang + 1.0e-6 or float(spans[1]) > target_y_ang + 1.0e-6:
        if target_x_ang > 0.0:
            coords[:, 0] = np.mod(coords[:, 0], target_x_ang)
        if target_y_ang > 0.0:
            coords[:, 1] = np.mod(coords[:, 1], target_y_ang)
        periodic_lateral_wrap_applied = True
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        spans = maxs - mins
        slot_z_ang = max(float(target_thickness_nm) * 10.0, float(spans[2]))
        box_z_ang = slot_z_ang + 2.0 * float(vacuum_padding_ang)

    if float(spans[0]) > target_x_ang + 1.0e-6 or float(spans[1]) > target_y_ang + 1.0e-6:
        raise RuntimeError(
            "Prepared slab is laterally larger than the graphite-matched target footprint "
            f"({float(spans[0]) / 10.0:.4f}, {float(spans[1]) / 10.0:.4f}) nm vs "
            f"({float(target_xy_nm[0]):.4f}, {float(target_xy_nm[1]):.4f}) nm."
        )

    lateral_shift = np.array(
        [
            0.5 * target_x_ang - 0.5 * float(mins[0] + maxs[0]),
            0.5 * target_y_ang - 0.5 * float(mins[1] + maxs[1]),
            float(vacuum_padding_ang) + 0.5 * (slot_z_ang - float(spans[2])) - float(mins[2]),
        ],
        dtype=float,
    )
    coords += lateral_shift
    for idx, xyz in enumerate(coords):
        conf.SetAtomPosition(idx, Geom.Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    setattr(confined, "cell", utils.Cell(target_x_ang, 0.0, target_y_ang, 0.0, box_z_ang, 0.0))
    poly.set_cell_param_conf(confined, 0, target_x_ang, 0.0, target_y_ang, 0.0, box_z_ang, 0.0)
    summary = {
        "target_xy_nm": [float(target_xy_nm[0]), float(target_xy_nm[1])],
        "target_thickness_nm": float(target_thickness_nm),
        "occupied_thickness_nm": float(spans[2]) / 10.0,
        "confined_box_nm": [target_x_ang / 10.0, target_y_ang / 10.0, box_z_ang / 10.0],
        "vacuum_padding_ang": float(vacuum_padding_ang),
        "periodic_lateral_wrap_applied": bool(periodic_lateral_wrap_applied),
    }
    note = "reboxed the prepared slab onto the graphite master footprint"
    if periodic_lateral_wrap_applied:
        note += " and restored lateral periodic coordinates"
    note += f" and inserted {float(vacuum_padding_ang) / 10.0:.3f} nm top/bottom vacuum before confined slab relaxation"
    return confined, summary, note


def _phase_local_density_summary(
    *,
    gro_path: Path,
    species: Sequence,
    counts: Sequence[int],
    center_window_fraction: float = 0.5,
) -> dict[str, object]:
    positions = _gro_positions_nm(gro_path)
    box_nm = _read_gro_box_nm(gro_path)
    if not positions:
        return {
            "box_nm": [float(x) for x in box_nm],
            "occupied_density_g_cm3": 0.0,
            "center_bulk_like_density_g_cm3": 0.0,
            "wrapped_across_z_boundary": False,
        }

    blocks = _molecule_atom_blocks(species=species, counts=counts)
    total_atoms = sum(stop - start for start, stop in blocks)
    if total_atoms != len(positions):
        return {
            "box_nm": [float(x) for x in box_nm],
            "occupied_density_g_cm3": 0.0,
            "center_bulk_like_density_g_cm3": 0.0,
            "wrapped_across_z_boundary": False,
            "block_alignment_ok": False,
        }

    z_all = [float(pos[2]) for pos in positions]
    occupied_min = float(min(z_all))
    occupied_max = float(max(z_all))
    occupied_thickness = max(occupied_max - occupied_min, 1.0e-6)
    total_mass_amu = sum(float(molecular_weight(mol, strict=True)) * int(count) for mol, count in zip(species, counts))
    occupied_volume_cm3 = float(box_nm[0] * box_nm[1] * occupied_thickness) * 1.0e-21
    occupied_density = 0.0 if occupied_volume_cm3 <= 0.0 else float(total_mass_amu / _AVOGADRO / occupied_volume_cm3)

    center_window = max(occupied_thickness * float(center_window_fraction), 1.0e-6)
    center_mid = 0.5 * (occupied_min + occupied_max)
    center_lo = center_mid - 0.5 * center_window
    center_hi = center_mid + 0.5 * center_window
    center_mass_amu = 0.0
    positions_arr = np.asarray(positions, dtype=float)
    for mol, (start, stop) in zip(
        [mol for mol, count in zip(species, counts) for _ in range(int(count))],
        blocks,
    ):
        masses = np.asarray([float(atom.GetMass()) for atom in mol.GetAtoms()], dtype=float)
        z = positions_arr[start:stop, 2]
        if z.size != masses.size or z.size == 0:
            continue
        com_z = float(np.average(z, weights=masses))
        if center_lo <= com_z <= center_hi:
            center_mass_amu += float(np.sum(masses))
    center_volume_cm3 = float(box_nm[0] * box_nm[1] * center_window) * 1.0e-21
    center_density = 0.0 if center_volume_cm3 <= 0.0 else float(center_mass_amu / _AVOGADRO / center_volume_cm3)
    return {
        "box_nm": [float(x) for x in box_nm],
        "occupied_z_range_nm": [occupied_min, occupied_max],
        "occupied_thickness_nm": occupied_thickness,
        "occupied_density_g_cm3": occupied_density,
        "center_bulk_like_window_nm": [center_lo, center_hi],
        "center_bulk_like_density_g_cm3": center_density,
        "wrapped_across_z_boundary": bool((occupied_max - occupied_min) > 0.90 * float(box_nm[2])),
        "block_alignment_ok": True,
    }


def _confined_phase_report(
    *,
    label: str,
    species_names: Sequence[str],
    counts: Sequence[int],
    target_density_g_cm3: float | None,
    summary: dict[str, object],
) -> SandwichPhaseReport:
    confined_box = tuple(float(x) for x in summary.get("box_nm", (0.0, 0.0, 0.0)))
    occupied_thickness = float(summary.get("occupied_thickness_nm", confined_box[2]))
    occupied_box = (float(confined_box[0]), float(confined_box[1]), float(occupied_thickness))
    return SandwichPhaseReport(
        label=str(label),
        box_nm=occupied_box,
        density_g_cm3=float(summary.get("occupied_density_g_cm3", 0.0)),
        species_names=tuple(str(x) for x in species_names),
        counts=tuple(int(x) for x in counts),
        target_density_g_cm3=(None if target_density_g_cm3 is None else float(target_density_g_cm3)),
    )


def _run_confined_phase_relaxation(
    *,
    label: str,
    prepared_slab,
    species: Sequence,
    counts: Sequence[int],
    charge_scale: Sequence[float],
    target_xy_nm: tuple[float, float],
    target_density_g_cm3: float,
    target_thickness_nm: float,
    ff_name: str,
    relax: SandwichRelaxationSpec,
    work_dir: Path,
    restart: bool | None = None,
) -> _ConfinedPhaseResult:
    from .protocol import _resolve_route_b_wall_atomtype

    base_block = _load_block_from_top_gro(
        top_path=prepared_slab.top_path,
        gro_path=prepared_slab.gro_path,
        fallback_cell=None,
    )
    if base_block is None:
        raise RuntimeError(f"Cannot load prepared slab geometry for confined {label} relaxation.")

    confined_block, rebox_summary, rebox_note = _rebox_block_for_phase_confinement(
        block=base_block,
        target_xy_nm=target_xy_nm,
        target_thickness_nm=float(target_thickness_nm),
        vacuum_padding_ang=max(12.0, float(relax.top_padding_ang)),
        species=species,
        counts=counts,
    )
    register_cell_species_metadata(
        confined_block,
        list(species),
        list(counts),
        charge_scale=list(charge_scale),
        pack_mode=f"{label}_confined_slab",
    )

    export = export_system_from_cell_meta(
        cell_mol=confined_block,
        out_dir=work_dir / "00_export",
        ff_name=str(ff_name),
        charge_method="RESP",
        write_system_mol2=False,
    )
    wall_atomtype, _available = _resolve_route_b_wall_atomtype(export.system_top, None)
    if wall_atomtype is None:
        raise RuntimeError(f"Could not resolve a valid wall atomtype for confined {label} slab relaxation.")

    stages = _phase_confined_relaxation_stages(relax=relax, wall_atomtype=wall_atomtype)
    resources = RunResources(
        ntmpi=int(relax.mpi),
        ntomp=int(relax.omp),
        use_gpu=bool(relax.gpu),
        gpu_id=(str(relax.gpu_id) if relax.gpu_id is not None else None),
    )
    job = EquilibrationJob(
        gro=export.system_gro,
        top=export.system_top,
        ndx=export.system_ndx,
        provenance_ndx=export.system_ndx,
        out_dir=work_dir / "01_relax",
        stages=stages,
        resources=resources,
    )
    job.run(restart=bool(resolve_restart(restart)))
    relaxed_gro = work_dir / "01_relax" / stages[-1].name / "md.gro"
    relaxed_block = _load_block_from_top_gro(
        top_path=export.system_top,
        gro_path=relaxed_gro,
        fallback_cell=confined_block,
    )
    if relaxed_block is None:
        raise RuntimeError(f"Could not load relaxed confined {label} slab from {relaxed_gro}.")

    density_summary = _phase_local_density_summary(gro_path=relaxed_gro, species=species, counts=counts)
    summary = {
        "label": str(label),
        "note": str(rebox_note),
        "target_density_g_cm3": float(target_density_g_cm3),
        "target_xy_nm": [float(target_xy_nm[0]), float(target_xy_nm[1])],
        "target_thickness_nm": float(target_thickness_nm),
        "wall_atomtype": str(wall_atomtype),
        **rebox_summary,
        **density_summary,
        "relaxed_gro": str(relaxed_gro),
        "top_path": str(export.system_top),
    }
    summary_path = work_dir / f"{label}_phase_confined_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = _confined_phase_report(
        label=label,
        species_names=[str(get_name(mol, default=f"{label}_{idx + 1}")) for idx, mol in enumerate(species)],
        counts=counts,
        target_density_g_cm3=float(target_density_g_cm3),
        summary=summary,
    )
    return _ConfinedPhaseResult(
        label=str(label),
        relaxed_block=relaxed_block,
        report=report,
        summary=summary,
        summary_path=summary_path,
        top_path=export.system_top,
        gro_path=relaxed_gro,
    )


def _load_relaxed_block(*, work_dir: Path, fallback_cell):
    gro_path = _find_latest_equilibrated_gro(Path(work_dir))
    top_path = Path(work_dir) / "02_system" / "system.top"
    if gro_path is None or not Path(gro_path).exists() or not top_path.exists():
        return fallback_cell
    return _load_block_from_top_gro(top_path=top_path, gro_path=Path(gro_path), fallback_cell=fallback_cell)


def _load_block_from_top_gro(*, top_path: Path, gro_path: Path, fallback_cell=None):
    mol2_path = Path(gro_path).with_suffix(".mol2")
    if not mol2_path.exists():
        try:
            write_mol2_from_top_gro_parmed(
                top_path=top_path,
                gro_path=Path(gro_path),
                out_mol2=mol2_path,
                overwrite=True,
            )
        except Exception:
            return fallback_cell

    try:
        relaxed = read_mol2_with_charges(mol2_path, sanitize=False, removeHs=False)
    except Exception:
        return fallback_cell

    try:
        box_nm = read_equilibrated_box_nm(gro_path=Path(gro_path))
        setattr(
            relaxed,
            "cell",
            utils.Cell(
                float(box_nm[0]) * 10.0,
                0.0,
                float(box_nm[1]) * 10.0,
                0.0,
                float(box_nm[2]) * 10.0,
                0.0,
            ),
        )
        poly.set_cell_param_conf(
            relaxed,
            0,
            float(box_nm[0]) * 10.0,
            0.0,
            float(box_nm[1]) * 10.0,
            0.0,
            float(box_nm[2]) * 10.0,
            0.0,
        )
    except Exception:
        return fallback_cell

    try:
        if fallback_cell is not None and fallback_cell.HasProp("_yadonpy_cell_meta"):
            relaxed.SetProp("_yadonpy_cell_meta", str(fallback_cell.GetProp("_yadonpy_cell_meta")))
    except Exception:
        pass
    return relaxed


def _prepared_slab_phase_report(
    *,
    label: str,
    prepared_slab,
    species_names: Sequence[str],
    target_density_g_cm3: float | None,
) -> SandwichPhaseReport:
    payload = json.loads(Path(prepared_slab.meta_path).read_text(encoding="utf-8"))
    counts_map = {str(name): int(count) for name, count in dict(payload.get("molecule_counts") or {}).items()}
    ordered_names: list[str] = []
    ordered_counts: list[int] = []
    for name in species_names:
        count = int(counts_map.get(str(name), 0))
        if count > 0:
            ordered_names.append(str(name))
            ordered_counts.append(int(count))
    if not ordered_names:
        for name, count in counts_map.items():
            if int(count) > 0:
                ordered_names.append(str(name))
                ordered_counts.append(int(count))
    density = payload.get("density_g_cm3")
    return SandwichPhaseReport(
        label=str(label),
        box_nm=tuple(float(x) for x in payload.get("box_nm", prepared_slab.box_nm)),
        density_g_cm3=(0.0 if density is None else float(density)),
        species_names=tuple(ordered_names),
        counts=tuple(ordered_counts),
        target_density_g_cm3=(None if target_density_g_cm3 is None else float(target_density_g_cm3)),
    )


def build_graphite_polymer_electrolyte_sandwich(
    *,
    work_dir,
    ff,
    ion_ff,
    graphite: GraphiteSubstrateSpec,
    polymer: PolymerSlabSpec,
    electrolyte: ElectrolyteSlabSpec,
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    restart: bool | None = None,
) -> GraphitePolymerElectrolyteSandwichResult:
    if len(electrolyte.solvents) != len(electrolyte.solvent_mass_ratio):
        raise ValueError("electrolyte.solvents and electrolyte.solvent_mass_ratio must have the same length")

    wd = workdir(work_dir, restart=restart)
    graphite_dir = Path(wd) / "01_graphite"
    polymer_chain_dir = Path(wd) / "02_polymer_chain"
    polymer_build_dir = Path(wd) / "03_polymer_slab" / "00_build"
    polymer_eq_dir = Path(wd) / "03_polymer_slab"
    electrolyte_build_dir = Path(wd) / "04_electrolyte_slab" / "00_build"
    electrolyte_eq_dir = Path(wd) / "04_electrolyte_slab"
    stack_dir = Path(wd) / "05_sandwich"
    relax_dir = Path(wd) / "06_relax"
    stack_dir.mkdir(parents=True, exist_ok=True)
    relax_dir.mkdir(parents=True, exist_ok=True)
    graphite_dir.mkdir(parents=True, exist_ok=True)
    polymer_chain_dir.mkdir(parents=True, exist_ok=True)

    graphite_result = build_graphite(
        nx=int(graphite.nx),
        ny=int(graphite.ny),
        n_layers=int(graphite.n_layers),
        orientation=graphite.orientation,
        edge_cap=graphite.edge_cap,
        ff=ff,
        name=graphite.name,
        top_padding_ang=float(graphite.top_padding_ang),
    )

    polymer_target_box_nm = (
        float(graphite_result.box_nm[0]),
        float(graphite_result.box_nm[1]),
        float(polymer.slab_z_nm),
    )
    polymer_phase_build = _prepare_polymer_phase_species(
        ff=ff,
        ion_ff=ion_ff,
        polymer=polymer,
        relax=relax,
        chain_dir=polymer_chain_dir,
        box_nm=polymer_target_box_nm,
    )
    polymer_chain = polymer_phase_build["chain"]
    polymer_dp = int(polymer_phase_build["dp"])
    chain_count = int(polymer_phase_build["chain_count"])
    polymer_build_density = _initial_bulk_pack_density(
        target_density_g_cm3=float(polymer.target_density_g_cm3),
        phase="polymer",
    )
    polymer_bulk = poly.amorphous_cell(
        list(polymer_phase_build["species"]),
        list(polymer_phase_build["counts"]),
        density=float(polymer_build_density),
        neutralize=False,
        charge_scale=list(polymer_phase_build["charge_scale"]),
        work_dir=polymer_build_dir,
        retry=int(polymer.pack_retry),
        retry_step=int(polymer.pack_retry_step),
        threshold=float(polymer.pack_threshold_ang),
        dec_rate=float(polymer.pack_dec_rate),
    )
    register_cell_species_metadata(
        polymer_bulk,
        list(polymer_phase_build["species"]),
        list(polymer_phase_build["counts"]),
        charge_scale=list(polymer_phase_build["charge_scale"]),
        pack_mode="sandwich_polymer_bulk",
    )
    bulk_eq21_exec_kwargs = {
        "time": float(relax.bulk_eq21_final_ns),
        "eq21_pre_nvt_ps": 5.0,
        "eq21_tmax": max(float(relax.temperature_k), 650.0),
        "eq21_pmax": 5000.0,
        "eq21_npt_time_scale": 0.4,
        **{str(k): float(v) for k, v in dict(relax.bulk_eq21_exec_kwargs).items()},
    }
    _ = equilibrate_bulk_with_eq21(
        label="Polymer bulk",
        ac=polymer_bulk,
        work_dir=polymer_eq_dir,
        temp=float(relax.temperature_k),
        press=float(relax.pressure_bar),
        mpi=int(relax.mpi),
        omp=int(relax.omp),
        gpu=int(relax.gpu),
        gpu_id=(0 if relax.gpu_id is None else int(relax.gpu_id)),
        additional_loops=int(relax.bulk_additional_loops),
        final_npt_ns=float(relax.bulk_eq21_final_ns),
        eq21_exec_kwargs=bulk_eq21_exec_kwargs,
    )
    polymer_slab, polymer_slab_note = _prepare_slab_from_equilibrated_bulk(
        label="polymer",
        bulk_work_dir=polymer_eq_dir,
        target_lengths_nm=(float(graphite_result.box_nm[0]), float(graphite_result.box_nm[1])),
        target_thickness_nm=float(polymer.slab_z_nm),
        out_dir=polymer_eq_dir / "05_prepare_slab",
        restart=restart,
    )
    polymer_species_names = [str(get_name(mol, default=f"POLY_{idx + 1}")) for idx, mol in enumerate(polymer_phase_build["species"])]
    polymer_prepared_report = _prepared_slab_phase_report(
        label="polymer",
        prepared_slab=polymer_slab,
        species_names=polymer_species_names,
        target_density_g_cm3=float(polymer.target_density_g_cm3),
    )
    polymer_count_map = {str(name): int(count) for name, count in zip(polymer_prepared_report.species_names, polymer_prepared_report.counts)}
    polymer_selected_counts = [int(polymer_count_map.get(name, 0)) for name in polymer_species_names]
    polymer_confined = _run_confined_phase_relaxation(
        label="polymer",
        prepared_slab=polymer_slab,
        species=list(polymer_phase_build["species"]),
        counts=list(polymer_selected_counts),
        charge_scale=list(polymer_phase_build["charge_scale"]),
        target_xy_nm=(float(graphite_result.box_nm[0]), float(graphite_result.box_nm[1])),
        target_density_g_cm3=float(polymer.target_density_g_cm3),
        target_thickness_nm=float(polymer.slab_z_nm),
        ff_name=str(ff.name),
        relax=relax,
        work_dir=polymer_eq_dir / "06_confined_slab",
        restart=restart,
    )
    polymer_report = polymer_confined.report
    polymer_relaxed_block = polymer_confined.relaxed_block

    solvent_mols = tuple(_prepare_small_molecule(spec, ff=ff, ion_ff=ion_ff) for spec in electrolyte.solvents)
    salt_cation = _prepare_small_molecule(electrolyte.salt_cation, ff=ff, ion_ff=ion_ff)
    salt_anion = _prepare_small_molecule(electrolyte.salt_anion, ff=ff, ion_ff=ion_ff)
    electrolyte_prep = plan_fixed_xy_direct_electrolyte_preparation(
        reference_box_nm=(
            float(graphite_result.box_nm[0]),
            float(graphite_result.box_nm[1]),
            float(electrolyte.slab_z_nm),
        ),
        target_density_g_cm3=float(electrolyte.target_density_g_cm3),
        solvent_mol_weights=[molecular_weight(mol, strict=True) for mol in solvent_mols],
        solvent_mass_ratio=list(float(x) for x in electrolyte.solvent_mass_ratio),
        salt_mol_weights=[molecular_weight(salt_cation, strict=True), molecular_weight(salt_anion, strict=True)],
        salt_molarity_M=float(electrolyte.salt_molarity_M),
        min_salt_pairs=int(electrolyte.min_salt_pairs),
        solvent_species_names=[spec.name for spec in electrolyte.solvents],
        salt_species_names=[electrolyte.salt_cation.name, electrolyte.salt_anion.name],
        initial_pack_density_g_cm3=electrolyte.initial_pack_density_g_cm3,
        pressure_bar=float(relax.pressure_bar),
    )
    electrolyte_mols = list(solvent_mols) + [salt_cation, salt_anion]
    electrolyte_charge_scale = [float(spec.charge_scale) for spec in electrolyte.solvents] + [
        float(electrolyte.salt_cation.charge_scale),
        float(electrolyte.salt_anion.charge_scale),
    ]
    electrolyte_build_density = _initial_bulk_pack_density(
        target_density_g_cm3=float(electrolyte.target_density_g_cm3),
        phase="electrolyte",
        requested_density_g_cm3=electrolyte.initial_pack_density_g_cm3,
    )
    electrolyte_bulk = poly.amorphous_cell(
        electrolyte_mols,
        list(electrolyte_prep.direct_plan.target_counts),
        density=float(electrolyte_build_density),
        neutralize=False,
        charge_scale=electrolyte_charge_scale,
        work_dir=electrolyte_build_dir,
        retry=int(electrolyte.pack_retry),
        retry_step=int(electrolyte.pack_retry_step),
        threshold=float(electrolyte.pack_threshold_ang),
        dec_rate=float(electrolyte.pack_dec_rate),
    )
    register_cell_species_metadata(
        electrolyte_bulk,
        electrolyte_mols,
        list(electrolyte_prep.direct_plan.target_counts),
        charge_scale=electrolyte_charge_scale,
        pack_mode="sandwich_electrolyte_bulk",
    )
    _ = equilibrate_bulk_with_eq21(
        label="Electrolyte bulk",
        ac=electrolyte_bulk,
        work_dir=electrolyte_eq_dir,
        temp=float(relax.temperature_k),
        press=float(relax.pressure_bar),
        mpi=int(relax.mpi),
        omp=int(relax.omp),
        gpu=int(relax.gpu),
        gpu_id=(0 if relax.gpu_id is None else int(relax.gpu_id)),
        additional_loops=int(relax.bulk_additional_loops),
        final_npt_ns=float(relax.bulk_eq21_final_ns),
        eq21_exec_kwargs=bulk_eq21_exec_kwargs,
    )
    electrolyte_slab, electrolyte_slab_note = _prepare_slab_from_equilibrated_bulk(
        label="electrolyte",
        bulk_work_dir=electrolyte_eq_dir,
        target_lengths_nm=(float(graphite_result.box_nm[0]), float(graphite_result.box_nm[1])),
        target_thickness_nm=float(electrolyte.slab_z_nm),
        out_dir=electrolyte_eq_dir / "05_prepare_slab",
        restart=restart,
    )
    electrolyte_species_names = [str(get_name(mol, default=f"EL_{idx + 1}")) for idx, mol in enumerate(electrolyte_mols)]
    electrolyte_prepared_report = _prepared_slab_phase_report(
        label="electrolyte",
        prepared_slab=electrolyte_slab,
        species_names=electrolyte_species_names,
        target_density_g_cm3=float(electrolyte.target_density_g_cm3),
    )
    electrolyte_count_map = {str(name): int(count) for name, count in zip(electrolyte_prepared_report.species_names, electrolyte_prepared_report.counts)}
    electrolyte_selected_counts = [int(electrolyte_count_map.get(name, 0)) for name in electrolyte_species_names]
    electrolyte_confined = _run_confined_phase_relaxation(
        label="electrolyte",
        prepared_slab=electrolyte_slab,
        species=list(electrolyte_mols),
        counts=list(electrolyte_selected_counts),
        charge_scale=list(electrolyte_charge_scale),
        target_xy_nm=(float(graphite_result.box_nm[0]), float(graphite_result.box_nm[1])),
        target_density_g_cm3=float(electrolyte.target_density_g_cm3),
        target_thickness_nm=float(electrolyte.slab_z_nm),
        ff_name=str(ff.name),
        relax=relax,
        work_dir=electrolyte_eq_dir / "06_confined_slab",
        restart=restart,
    )
    electrolyte_report = electrolyte_confined.report
    electrolyte_relaxed_block = electrolyte_confined.relaxed_block

    stacked = stack_cell_blocks(
        [graphite_result.cell, polymer_relaxed_block, electrolyte_relaxed_block],
        z_gaps_ang=[float(relax.graphite_to_polymer_gap_ang), float(relax.polymer_to_electrolyte_gap_ang)],
        top_padding_ang=float(relax.top_padding_ang),
    )
    stacked_mols = [graphite_result.layer_mol]
    stacked_counts = [int(graphite_result.layer_count)]
    stacked_charge_scale = [1.0]
    for mol, count, scale in zip(polymer_phase_build["species"], polymer_selected_counts, polymer_phase_build["charge_scale"]):
        if int(count) <= 0:
            continue
        stacked_mols.append(mol)
        stacked_counts.append(int(count))
        stacked_charge_scale.append(float(scale))
    for mol, count, scale in zip(electrolyte_mols, electrolyte_selected_counts, electrolyte_charge_scale):
        if int(count) <= 0:
            continue
        stacked_mols.append(mol)
        stacked_counts.append(int(count))
        stacked_charge_scale.append(float(scale))
    register_cell_species_metadata(
        stacked.cell,
        stacked_mols,
        stacked_counts,
        charge_scale=stacked_charge_scale,
        pack_mode="graphite_polymer_electrolyte_sandwich",
    )
    export = export_system_from_cell_meta(
        cell_mol=stacked.cell,
        out_dir=stack_dir,
        ff_name=str(ff.name),
        charge_method="RESP",
        write_system_mol2=False,
    )

    graphite_group_name = str(get_name(graphite_result.layer_mol, default=graphite.name))
    polymer_group_name = str(get_name(polymer_chain, default=polymer.name))
    electrolyte_group_names = [str(get_name(mol, default=f"EL{idx + 1}")) for idx, mol in enumerate(electrolyte_mols)]
    ndx_groups = _augment_sandwich_ndx(
        ndx_path=export.system_ndx,
        graphite_name=graphite_group_name,
        polymer_name=polymer_group_name,
        electrolyte_names=electrolyte_group_names,
    )
    relaxed_gro = _run_stacked_relaxation(
        export=export,
        work_dir=relax_dir,
        relax=relax,
        freeze_group="GRAPHITE",
        restart=restart,
    )
    stack_checks = _build_stack_checks(gro_path=relaxed_gro, ndx_groups=ndx_groups)

    manifest_path = stack_dir / "sandwich_manifest.json"
    progress_path = stack_dir / "sandwich_progress.json"
    notes = (
        "polymer and electrolyte were first equilibrated as standalone bulk phases, then graphite-matched slabs were cut from dense equilibrium windows before three-phase stacking",
        "each dense slab then underwent a separate fixed-XY confined pre-relaxation with z walls and explicit vacuum so the final stack no longer relies on z-periodic healing",
        "graphite stays frozen during the stacked relaxation so the liquid and polymer phases can relax density mainly along the surface normal",
        f"polymer chain target atoms={int(polymer.chain_target_atoms)} -> built DP={int(polymer_dp)} and chain_count={int(chain_count)}",
        f"polymer bulk initial pack density={float(polymer_build_density):.4f} g/cm^3 -> target equilibrium density={float(polymer.target_density_g_cm3):.4f} g/cm^3",
        f"electrolyte bulk initial pack density={float(electrolyte_build_density):.4f} g/cm^3 -> target equilibrium density={float(electrolyte.target_density_g_cm3):.4f} g/cm^3",
        *tuple(str(x) for x in polymer_phase_build["notes"]),
        str(polymer_slab_note),
        str(electrolyte_slab_note),
        f"polymer confined summary={polymer_confined.summary_path}",
        f"electrolyte confined summary={electrolyte_confined.summary_path}",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "graphite": asdict(graphite),
                "polymer": asdict(polymer),
                "electrolyte": asdict(electrolyte),
                "relax": asdict(relax),
                "graphite_box_nm": [float(x) for x in graphite_result.box_nm],
                "polymer_phase": asdict(polymer_report),
                "electrolyte_phase": asdict(electrolyte_report),
                "polymer_phase_confined": polymer_confined.summary,
                "electrolyte_phase_confined": electrolyte_confined.summary,
                "polymer_phase_confined_summary": str(polymer_confined.summary_path),
                "electrolyte_phase_confined_summary": str(electrolyte_confined.summary_path),
                "stack_box_nm": [float(x) for x in stacked.box_nm],
                "stack_export_dir": str(stack_dir),
                "relaxed_gro": str(relaxed_gro),
                "ndx_groups": {name: len(idxs) for name, idxs in ndx_groups.items()},
                "stack_checks": stack_checks,
                "notes": list(notes),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    progress_path.write_text(
        json.dumps(
            {
                "stage": "completed",
                "polymer_phase_confined_summary": str(polymer_confined.summary_path),
                "electrolyte_phase_confined_summary": str(electrolyte_confined.summary_path),
                "manifest_path": str(manifest_path),
                "relaxed_gro": str(relaxed_gro),
                "stack_checks": stack_checks,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return GraphitePolymerElectrolyteSandwichResult(
        graphite=graphite_result,
        polymer_phase=polymer_report,
        electrolyte_phase=electrolyte_report,
        stack_export=export,
        relaxed_gro=relaxed_gro,
        manifest_path=manifest_path,
        stack_checks=stack_checks,
        notes=notes,
    )


def build_graphite_peo_electrolyte_sandwich(
    *,
    work_dir,
    ff,
    ion_ff,
    graphite: GraphiteSubstrateSpec = GraphiteSubstrateSpec(),
    polymer: PolymerSlabSpec = PolymerSlabSpec(),
    electrolyte: ElectrolyteSlabSpec = ElectrolyteSlabSpec(),
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    restart: bool | None = None,
) -> GraphitePolymerElectrolyteSandwichResult:
    return build_graphite_polymer_electrolyte_sandwich(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite,
        polymer=polymer,
        electrolyte=electrolyte,
        relax=relax,
        restart=restart,
    )


def build_graphite_cmcna_electrolyte_sandwich(
    *,
    work_dir,
    ff,
    ion_ff,
    graphite: GraphiteSubstrateSpec = GraphiteSubstrateSpec(),
    polymer: PolymerSlabSpec | None = None,
    electrolyte: ElectrolyteSlabSpec | None = None,
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    restart: bool | None = None,
) -> GraphitePolymerElectrolyteSandwichResult:
    return build_graphite_polymer_electrolyte_sandwich(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite,
        polymer=(polymer if polymer is not None else default_cmcna_polymer_spec()),
        electrolyte=(electrolyte if electrolyte is not None else default_carbonate_lipf6_electrolyte_spec()),
        relax=relax,
        restart=restart,
    )


__all__ = [
    "ElectrolyteSlabSpec",
    "GraphitePolymerElectrolyteSandwichResult",
    "GraphiteSubstrateSpec",
    "MoleculeSpec",
    "PolymerSlabSpec",
    "SandwichPhaseReport",
    "SandwichRelaxationSpec",
    "build_graphite_cmcna_electrolyte_sandwich",
    "build_graphite_peo_electrolyte_sandwich",
    "build_graphite_polymer_electrolyte_sandwich",
    "default_carbonate_lipf6_electrolyte_spec",
    "default_cmcna_polymer_spec",
    "default_peo_electrolyte_spec",
    "default_peo_polymer_spec",
]
