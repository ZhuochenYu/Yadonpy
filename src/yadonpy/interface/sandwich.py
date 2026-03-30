from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Sequence

from ..core import poly, utils
from ..core.graphite import GraphiteBuildResult, build_graphite, register_cell_species_metadata, stack_cell_blocks
from ..core.molspec import molecular_weight
from ..core.naming import get_name
from ..core.polyelectrolyte import detect_charged_groups
from ..core.workdir import workdir
from ..gmx.index import _write_ndx
from ..gmx.mdp_templates import MINIM_STEEP_MDP, NPT_MDP, NVT_MDP, MdpSpec, default_mdp_params
from ..gmx.workflows._util import RunResources
from ..gmx.workflows.eq import EqStage, EquilibrationJob
from ..io.gromacs_system import SystemExportResult, export_system_from_cell_meta
from ..io.mol2 import read_mol2_with_charges, write_mol2_from_top_gro_parmed
from ..runtime import resolve_restart
from ..sim import qm
from ..sim.preset.eq import _find_latest_equilibrated_gro
from .bulk_resize import build_bulk_equilibrium_profile, fixed_xy_semiisotropic_npt_overrides, read_equilibrated_box_nm
from .postprocess import read_ndx_groups
from .prep import equilibrate_bulk_with_eq21, make_orthorhombic_pack_cell, plan_fixed_xy_direct_electrolyte_preparation


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
        ratio = tuple(float(x) for x in polymer.monomer_ratio) if polymer.monomers else None
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


def _build_stack_checks(*, gro_path: Path, ndx_groups: dict[str, list[int]]) -> dict[str, object]:
    z_coords = _read_gro_z_coords(gro_path)
    payload: dict[str, object] = {"gro_path": str(gro_path)}
    phase_stats: dict[str, dict[str, float]] = {}
    for name in ("GRAPHITE", "POLYMER", "ELECTROLYTE"):
        members = [int(idx) for idx in ndx_groups.get(name, []) if 1 <= int(idx) <= len(z_coords)]
        if not members:
            continue
        values = [float(z_coords[idx - 1]) for idx in members]
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


def _load_relaxed_block(*, work_dir: Path, fallback_cell):
    gro_path = _find_latest_equilibrated_gro(Path(work_dir))
    if gro_path is None or not Path(gro_path).exists():
        return fallback_cell

    mol2_path = Path(gro_path).with_suffix(".mol2")
    if not mol2_path.exists():
        top_path = Path(work_dir) / "02_system" / "system.top"
        if top_path.exists():
            try:
                write_mol2_from_top_gro_parmed(
                    top_path=top_path,
                    gro_path=Path(gro_path),
                    out_mol2=mol2_path,
                    overwrite=True,
                )
            except Exception:
                return fallback_cell
        else:
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
        if fallback_cell.HasProp("_yadonpy_cell_meta"):
            relaxed.SetProp("_yadonpy_cell_meta", str(fallback_cell.GetProp("_yadonpy_cell_meta")))
    except Exception:
        pass
    return relaxed


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
    polymer_build_box_nm = (
        float(polymer_target_box_nm[0]),
        float(polymer_target_box_nm[1]),
        float(polymer_target_box_nm[2]) * float(polymer.initial_pack_z_scale),
    )
    polymer_slab = poly.amorphous_cell(
        list(polymer_phase_build["species"]),
        list(polymer_phase_build["counts"]),
        cell=make_orthorhombic_pack_cell(polymer_build_box_nm),
        density=None,
        neutralize=False,
        charge_scale=list(polymer_phase_build["charge_scale"]),
        work_dir=polymer_build_dir,
        retry=int(polymer.pack_retry),
        retry_step=int(polymer.pack_retry_step),
        threshold=float(polymer.pack_threshold_ang),
        dec_rate=float(polymer.pack_dec_rate),
    )
    register_cell_species_metadata(
        polymer_slab,
        list(polymer_phase_build["species"]),
        list(polymer_phase_build["counts"]),
        charge_scale=list(polymer_phase_build["charge_scale"]),
        pack_mode="sandwich_polymer_slab",
    )
    fixed_xy = fixed_xy_semiisotropic_npt_overrides(pressure_bar=float(relax.pressure_bar))
    _ = equilibrate_bulk_with_eq21(
        label="Polymer slab",
        ac=polymer_slab,
        work_dir=polymer_eq_dir,
        temp=float(relax.temperature_k),
        press=float(relax.pressure_bar),
        mpi=int(relax.mpi),
        omp=int(relax.omp),
        gpu=int(relax.gpu),
        gpu_id=(0 if relax.gpu_id is None else int(relax.gpu_id)),
        additional_loops=int(relax.bulk_additional_loops),
        eq21_npt_mdp_overrides=fixed_xy,
        additional_mdp_overrides=fixed_xy,
        final_npt_ns=float(relax.bulk_eq21_final_ns),
        final_npt_mdp_overrides=fixed_xy,
        eq21_exec_kwargs={
            "time": float(relax.bulk_eq21_final_ns),
            "eq21_pre_nvt_ps": 5.0,
            "eq21_tmax": max(float(relax.temperature_k), 650.0),
            "eq21_pmax": 5000.0,
            "eq21_npt_time_scale": 0.4,
        },
    )
    polymer_report = _phase_report(
        label="polymer",
        counts=list(polymer_phase_build["counts"]),
        mols=list(polymer_phase_build["species"]),
        work_dir=polymer_eq_dir,
        target_density_g_cm3=float(polymer.target_density_g_cm3),
    )
    polymer_relaxed_block = _load_relaxed_block(work_dir=polymer_eq_dir, fallback_cell=polymer_slab)

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
    electrolyte_slab = poly.amorphous_cell(
        electrolyte_mols,
        list(electrolyte_prep.direct_plan.target_counts),
        cell=make_orthorhombic_pack_cell(electrolyte_prep.pack_plan.initial_pack_box_nm),
        density=None,
        neutralize=False,
        charge_scale=electrolyte_charge_scale,
        work_dir=electrolyte_build_dir,
        retry=int(electrolyte.pack_retry),
        retry_step=int(electrolyte.pack_retry_step),
        threshold=float(electrolyte.pack_threshold_ang),
        dec_rate=float(electrolyte.pack_dec_rate),
    )
    register_cell_species_metadata(
        electrolyte_slab,
        electrolyte_mols,
        list(electrolyte_prep.direct_plan.target_counts),
        charge_scale=electrolyte_charge_scale,
        pack_mode="sandwich_electrolyte_slab",
    )
    _ = equilibrate_bulk_with_eq21(
        label="Electrolyte slab",
        ac=electrolyte_slab,
        work_dir=electrolyte_eq_dir,
        temp=float(relax.temperature_k),
        press=float(relax.pressure_bar),
        mpi=int(relax.mpi),
        omp=int(relax.omp),
        gpu=int(relax.gpu),
        gpu_id=(0 if relax.gpu_id is None else int(relax.gpu_id)),
        additional_loops=int(relax.bulk_additional_loops),
        eq21_npt_mdp_overrides=electrolyte_prep.relax_mdp_overrides,
        additional_mdp_overrides=electrolyte_prep.relax_mdp_overrides,
        final_npt_ns=float(relax.bulk_eq21_final_ns),
        final_npt_mdp_overrides=electrolyte_prep.relax_mdp_overrides,
        eq21_exec_kwargs={
            "time": float(relax.bulk_eq21_final_ns),
            "eq21_pre_nvt_ps": 5.0,
            "eq21_tmax": max(float(relax.temperature_k), 650.0),
            "eq21_pmax": 5000.0,
            "eq21_npt_time_scale": 0.4,
        },
    )
    electrolyte_report = _phase_report(
        label="electrolyte",
        counts=list(electrolyte_prep.direct_plan.target_counts),
        mols=electrolyte_mols,
        work_dir=electrolyte_eq_dir,
        target_density_g_cm3=float(electrolyte.target_density_g_cm3),
    )
    electrolyte_relaxed_block = _load_relaxed_block(work_dir=electrolyte_eq_dir, fallback_cell=electrolyte_slab)

    stacked = stack_cell_blocks(
        [graphite_result.cell, polymer_relaxed_block, electrolyte_relaxed_block],
        z_gaps_ang=[float(relax.graphite_to_polymer_gap_ang), float(relax.polymer_to_electrolyte_gap_ang)],
        top_padding_ang=float(relax.top_padding_ang),
    )
    stacked_counts = [int(graphite_result.layer_count)] + list(polymer_phase_build["counts"]) + list(electrolyte_prep.direct_plan.target_counts)
    stacked_mols = [graphite_result.layer_mol] + list(polymer_phase_build["species"]) + electrolyte_mols
    stacked_charge_scale = [1.0] + list(polymer_phase_build["charge_scale"]) + electrolyte_charge_scale
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
    notes = (
        "polymer and electrolyte were first equilibrated independently with XY locked to the graphite footprint, then stacked explicitly into a three-phase sandwich",
        "graphite stays frozen during the stacked relaxation so the liquid and polymer phases can relax density mainly along the surface normal",
        f"polymer chain target atoms={int(polymer.chain_target_atoms)} -> built DP={int(polymer_dp)} and chain_count={int(chain_count)}",
        *tuple(str(x) for x in polymer_phase_build["notes"]),
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
