from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Geometry as Geom

from ..core import poly, utils
from ..core.chem_utils import select_best_charge_property
from ..core.graphite import GraphiteBuildResult, build_graphite, register_cell_species_metadata, stack_cell_blocks
from ..core.molspec import as_rdkit_mol, molecular_weight
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
from .sandwich_metrics import (
    build_sandwich_acceptance as _build_sandwich_acceptance,
    build_stack_checks as _build_stack_checks,
    confined_phase_report as _confined_phase_report,
    confined_summary_score as _confined_summary_score,
    needs_confined_rescue as _needs_confined_rescue,
    phase_gap_penalty_nm as _phase_gap_penalty_nm,
    phase_local_density_summary as _phase_local_density_summary,
    representative_phase_density as _representative_phase_density,
)
from .sandwich_packing import (
    PackBackoffResult as _PackBackoffResult,
    build_pack_density_ladder as _build_pack_density_ladder,
    initial_bulk_pack_density as _initial_bulk_pack_density,
    run_amorphous_cell_with_density_backoff as _run_amorphous_cell_with_density_backoff,
)
from .sandwich_specs import (
    ElectrolyteSlabSpec,
    GraphitePolymerElectrolyteSandwichResult,
    GraphiteSubstrateSpec,
    MoleculeSpec,
    PolymerSlabSpec,
    SandwichPhaseReport,
    SandwichRelaxationSpec,
    default_carbonate_lipf6_electrolyte_spec,
    default_cmcna_polymer_spec,
    default_peo_electrolyte_spec,
    default_peo_polymer_spec,
)


_AVOGADRO = 6.02214076e23


@dataclass(frozen=True)
class _ConfinedPhaseResult:
    label: str
    relaxed_block: object
    report: SandwichPhaseReport
    summary: dict[str, object]
    summary_path: Path
    top_path: Path
    gro_path: Path


def _write_sandwich_progress(progress_path: Path, payload: dict[str, object]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_graphite_cmcna_glucose6_periodic_case(
    *,
    work_dir: str | Path,
    ff,
    ion_ff=None,
    profile: str = "full",
    graphite: GraphiteSubstrateSpec | None = None,
    polymer: PolymerSlabSpec | None = None,
    electrolyte: ElectrolyteSlabSpec | None = None,
    relax: SandwichRelaxationSpec | None = None,
    restart: bool | None = None,
):
    """Thin wrapper around the Example 08 preset entrypoint."""
    from .sandwich_examples import build_graphite_cmcna_glucose6_periodic_case as _build_case

    return _build_case(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        profile=profile,
        graphite=graphite,
        polymer=polymer,
        electrolyte=electrolyte,
        relax=relax,
        restart=restart,
    )


def _estimate_chain_count(*, chain_mw: float, target_density_g_cm3: float, box_nm: tuple[float, float, float], minimum: int) -> int:
    volume_cm3 = float(box_nm[0] * box_nm[1] * box_nm[2]) * 1.0e-21
    target_mass_g = float(target_density_g_cm3) * volume_cm3
    target_mass_amu = target_mass_g * _AVOGADRO
    if chain_mw <= 0.0:
        return int(max(1, minimum))
    estimate = int(max(1, round(target_mass_amu / float(chain_mw))))
    return int(max(int(minimum), estimate))


def _phase_total_mass_amu(*, species: Sequence, counts: Sequence[int]) -> float:
    return float(sum(float(molecular_weight(mol, strict=True)) * int(count) for mol, count in zip(species, counts)))


def _phase_required_area_nm2(*, total_mass_amu: float, target_density_g_cm3: float, target_thickness_nm: float) -> float:
    density = float(target_density_g_cm3)
    thickness_nm = float(target_thickness_nm)
    if density <= 0.0 or thickness_nm <= 0.0:
        return 0.0
    mass_g = float(total_mass_amu) / float(_AVOGADRO)
    volume_cm3 = mass_g / density
    return float(volume_cm3 / (thickness_nm * 1.0e-21))


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
    formal_charge = int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))
    if formal_charge != 0:
        return True
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
            basis_set=spec.basis_set,
            method=spec.method,
            require_ready=spec.require_ready,
            prefer_db=spec.prefer_db,
            polyelectrolyte_mode=spec.polyelectrolyte_mode,
            polyelectrolyte_detection=spec.polyelectrolyte_detection,
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
        pe_mode = spec.polyelectrolyte_mode if spec.polyelectrolyte_mode is not None else _is_polyelectrolyte_spec(spec)
        monomer_handle = ff.mol(
            spec.smiles,
            name=spec.name,
            charge=spec.charge_method,
            basis_set=spec.basis_set,
            method=spec.method,
            require_ready=spec.require_ready,
            prefer_db=spec.prefer_db,
            polyelectrolyte_mode=pe_mode,
            polyelectrolyte_detection=spec.polyelectrolyte_detection,
        )
        monomer = ff.ff_assign(
            monomer_handle,
            report=False,
            bonded=spec.bonded,
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


def _polymer_chain_effective_charge(mol) -> tuple[int, dict[str, object]]:
    formal_charge = int(_polymer_chain_formal_charge(mol))
    prop, values = select_best_charge_property(mol)
    if prop and values:
        total_charge = float(sum(float(v) for v in values))
        rounded_charge = int(round(total_charge))
        if abs(total_charge - float(rounded_charge)) <= 0.25:
            return rounded_charge, {
                "source": str(prop),
                "raw_total_charge": float(total_charge),
                "rounded_charge": int(rounded_charge),
                "formal_charge": int(formal_charge),
            }
    return formal_charge, {
        "source": "formal_charge",
        "raw_total_charge": float(formal_charge),
        "rounded_charge": int(formal_charge),
        "formal_charge": int(formal_charge),
    }


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

    chain_charge, chain_charge_meta = _polymer_chain_effective_charge(polymer_chain)
    if chain_charge != 0:
        if polymer.counterion is None:
            raise RuntimeError(
                f"Polymer {polymer.name} carries charge {chain_charge} per chain but no counterion was configured."
            )
        counterion = _prepare_small_molecule(polymer.counterion, ff=ff, ion_ff=ion_ff)
        ion_charge = int(_smiles_formal_charge(polymer.counterion.smiles))
        if ion_charge == 0:
            raise RuntimeError(f"Configured polymer counterion {polymer.counterion.name} is neutral.")
        total_polymer_charge = int(chain_charge * chain_count)
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
                f"polymer effective charge per chain={chain_charge} via {chain_charge_meta['source']}; "
                f"added {counterion_count} {polymer.counterion.name} counterions to neutralize the slab"
            )

    if polymer.chain_count is not None:
        phase_mass_amu = sum(float(molecular_weight(mol, strict=True)) * int(count) for mol, count in zip(species, counts))
        phase_volume_cm3 = float(box_nm[0] * box_nm[1] * box_nm[2]) * 1.0e-21
        projected_density = 0.0 if phase_volume_cm3 <= 0.0 else float(phase_mass_amu / _AVOGADRO / phase_volume_cm3)
        notes.append(
            f"explicit polymer chain_count={int(chain_count)} projects bulk-like phase density≈{projected_density:.3f} g/cm^3 at the requested graphite footprint"
        )
        if projected_density < 0.90 * float(polymer.target_density_g_cm3):
            notes.append(
                f"warning: explicit chain_count={int(chain_count)} underfills the target density {float(polymer.target_density_g_cm3):.3f} g/cm^3; increase chain_count or allow automatic estimation for a denser slab"
            )

    return {
        "chain": polymer_chain,
        "dp": int(polymer_dp),
        "chain_count": int(chain_count),
        "species": species,
        "counts": counts,
        "charge_scale": charge_scale,
        "charged_phase": bool(chain_charge != 0),
        "notes": tuple(notes),
    }


def _preflight_graphite_footprint_from_phase_targets(
    *,
    graphite: GraphiteSubstrateSpec,
    graphite_result: GraphiteBuildResult,
    ff,
    ion_ff,
    polymer: PolymerSlabSpec,
    electrolyte: ElectrolyteSlabSpec,
    relax: SandwichRelaxationSpec,
    chain_dir: Path,
    area_margin: float = 1.05,
    max_rounds: int = 3,
) -> tuple[GraphiteSubstrateSpec, GraphiteBuildResult, list[dict[str, object]]]:
    current_graphite = graphite
    current_result = graphite_result
    negotiations: list[dict[str, object]] = []

    for preflight_round in range(max(1, int(max_rounds))):
        polymer_phase_build = _prepare_polymer_phase_species(
            ff=ff,
            ion_ff=ion_ff,
            polymer=polymer,
            relax=relax,
            chain_dir=chain_dir,
            box_nm=(
                float(current_result.box_nm[0]),
                float(current_result.box_nm[1]),
                float(polymer.slab_z_nm),
            ),
        )
        electrolyte_inputs = _prepare_electrolyte_phase_inputs(
            ff=ff,
            ion_ff=ion_ff,
            electrolyte=electrolyte,
            graphite_box_nm=tuple(float(x) for x in current_result.box_nm),
            relax=relax,
        )
        polymer_area_nm2 = _phase_required_area_nm2(
            total_mass_amu=_phase_total_mass_amu(
                species=list(polymer_phase_build["species"]),
                counts=list(polymer_phase_build["counts"]),
            ),
            target_density_g_cm3=float(polymer.target_density_g_cm3),
            target_thickness_nm=float(polymer.slab_z_nm),
        )
        electrolyte_area_nm2 = _phase_required_area_nm2(
            total_mass_amu=_phase_total_mass_amu(
                species=list(electrolyte_inputs["mols"]),
                counts=list(electrolyte_inputs["prep"].direct_plan.target_counts),
            ),
            target_density_g_cm3=float(electrolyte.target_density_g_cm3),
            target_thickness_nm=float(electrolyte.slab_z_nm),
        )
        current_area_nm2 = float(current_result.box_nm[0]) * float(current_result.box_nm[1])
        polymer_required_xy_nm = _preflight_required_xy_nm_from_target_area(
            current_box_nm=tuple(float(x) for x in current_result.box_nm),
            target_area_nm2=float(polymer_area_nm2) * float(area_margin),
            linear_headroom_xy=_preflight_linear_headroom_xy(label="polymer"),
        )
        electrolyte_required_xy_nm = _preflight_required_xy_nm_from_target_area(
            current_box_nm=tuple(float(x) for x in current_result.box_nm),
            target_area_nm2=float(electrolyte_area_nm2) * float(area_margin),
            linear_headroom_xy=_preflight_linear_headroom_xy(label="electrolyte"),
        )
        required_xy_nm = (
            max(float(current_result.box_nm[0]), float(polymer_required_xy_nm[0]), float(electrolyte_required_xy_nm[0])),
            max(float(current_result.box_nm[1]), float(polymer_required_xy_nm[1]), float(electrolyte_required_xy_nm[1])),
        )
        target_area_nm2 = max(
            float(current_area_nm2),
            float(required_xy_nm[0]) * float(required_xy_nm[1]),
        )
        target_nx, target_ny = _graphite_counts_for_required_xy(
            graphite=current_graphite,
            current_box_nm=tuple(float(x) for x in current_result.box_nm),
            required_xy_nm=required_xy_nm,
        )
        if target_nx == int(current_graphite.nx) and target_ny == int(current_graphite.ny):
            break
        next_graphite = replace(current_graphite, nx=int(target_nx), ny=int(target_ny))
        seed_result = build_graphite(
            nx=int(next_graphite.nx),
            ny=int(next_graphite.ny),
            n_layers=int(next_graphite.n_layers),
            orientation=next_graphite.orientation,
            edge_cap=next_graphite.edge_cap,
            ff=ff,
            name=next_graphite.name,
            top_padding_ang=float(next_graphite.top_padding_ang),
        )
        next_graphite, next_result = _expand_graphite_to_meet_required_xy(
            graphite=next_graphite,
            graphite_result=seed_result,
            ff=ff,
            required_xy_nm=required_xy_nm,
        )
        negotiations.append(
            {
                "stage": "preflight",
                "preflight_round": int(preflight_round + 1),
                "graphite_counts_before_xy": [int(current_graphite.nx), int(current_graphite.ny)],
                "graphite_counts_after_xy": [int(next_graphite.nx), int(next_graphite.ny)],
                "graphite_count_scale_xy": [
                    float(next_graphite.nx) / max(float(current_graphite.nx), 1.0),
                    float(next_graphite.ny) / max(float(current_graphite.ny), 1.0),
                ],
                "graphite_box_before_nm": [float(x) for x in current_result.box_nm],
                "graphite_box_after_nm": [float(x) for x in next_result.box_nm],
                "polymer_target_area_nm2": float(polymer_area_nm2),
                "electrolyte_target_area_nm2": float(electrolyte_area_nm2),
                "polymer_preflight_required_xy_nm": [float(polymer_required_xy_nm[0]), float(polymer_required_xy_nm[1])],
                "electrolyte_preflight_required_xy_nm": [float(electrolyte_required_xy_nm[0]), float(electrolyte_required_xy_nm[1])],
                "current_area_nm2": float(current_area_nm2),
                "target_area_nm2": float(target_area_nm2),
                "area_margin": float(area_margin),
                "required_xy_nm": [float(required_xy_nm[0]), float(required_xy_nm[1])],
            }
        )
        current_graphite = next_graphite
        current_result = next_result

    return current_graphite, current_result, negotiations


def _preflight_linear_headroom_xy(*, label: str) -> tuple[float, float]:
    phase = str(label).strip().lower()
    if phase != "polymer":
        return (1.0, 1.0)
    min_scale_xy = _phase_confined_min_scale_xy(label=phase)
    return (
        1.0 / max(float(min_scale_xy[0]), 1.0e-9),
        1.0 / max(float(min_scale_xy[1]), 1.0e-9),
    )


def _preflight_required_xy_nm_from_target_area(
    *,
    current_box_nm: tuple[float, float, float],
    target_area_nm2: float,
    linear_headroom_xy: tuple[float, float] = (1.0, 1.0),
) -> tuple[float, float]:
    current_area_nm2 = max(float(current_box_nm[0]) * float(current_box_nm[1]), 1.0e-12)
    scale = math.sqrt(max(float(target_area_nm2), 1.0e-12) / current_area_nm2)
    return (
        float(current_box_nm[0]) * float(scale) * float(linear_headroom_xy[0]),
        float(current_box_nm[1]) * float(scale) * float(linear_headroom_xy[1]),
    )


def _phase_round_dir(base_dir: Path, round_index: int) -> Path:
    return Path(base_dir) if int(round_index) == 0 else Path(base_dir) / f"round_{int(round_index):02d}"


def _run_polymer_phase_round(
    *,
    ff,
    ion_ff,
    graphite_box_nm: tuple[float, float, float],
    polymer: PolymerSlabSpec,
    relax: SandwichRelaxationSpec,
    chain_dir: Path,
    base_phase_dir: Path,
    restart: bool | None,
) -> dict[str, object]:
    round_dir = Path(base_phase_dir)
    build_dir = round_dir / "00_build"
    target_box_nm = (
        float(graphite_box_nm[0]),
        float(graphite_box_nm[1]),
        float(polymer.slab_z_nm),
    )
    phase_build = _prepare_polymer_phase_species(
        ff=ff,
        ion_ff=ion_ff,
        polymer=polymer,
        relax=relax,
        chain_dir=chain_dir,
        box_nm=target_box_nm,
    )
    pack_result = _run_amorphous_cell_with_density_backoff(
        label="polymer",
        pack_fn=poly.amorphous_cell,
        mols=list(phase_build["species"]),
        counts=list(phase_build["counts"]),
        charge_scale=list(phase_build["charge_scale"]),
        phase="polymer",
        target_density_g_cm3=float(polymer.target_density_g_cm3),
        z_scale=float(polymer.initial_pack_z_scale),
        charged=bool(phase_build.get("charged_phase", False)),
        work_dir=build_dir,
        retry=int(polymer.pack_retry),
        retry_step=int(polymer.pack_retry_step),
        threshold=float(polymer.pack_threshold_ang),
        dec_rate=float(polymer.pack_dec_rate),
    )
    bulk = pack_result.cell
    register_cell_species_metadata(
        bulk,
        list(phase_build["species"]),
        list(phase_build["counts"]),
        charge_scale=list(phase_build["charge_scale"]),
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
        ac=bulk,
        work_dir=round_dir,
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
    prepared_slab, slab_note = _prepare_slab_from_equilibrated_bulk(
        label="polymer",
        bulk_work_dir=round_dir,
        target_lengths_nm=(float(graphite_box_nm[0]), float(graphite_box_nm[1])),
        target_thickness_nm=float(polymer.slab_z_nm),
        out_dir=round_dir / "05_prepare_slab",
        restart=restart,
    )
    species_names = [str(get_name(mol, default=f"POLY_{idx + 1}")) for idx, mol in enumerate(phase_build["species"])]
    prepared_report = _prepared_slab_phase_report(
        label="polymer",
        prepared_slab=prepared_slab,
        species_names=species_names,
        target_density_g_cm3=float(polymer.target_density_g_cm3),
    )
    count_map = {str(name): int(count) for name, count in zip(prepared_report.species_names, prepared_report.counts)}
    selected_counts = [int(count_map.get(name, 0)) for name in species_names]
    return {
        "round_dir": round_dir,
        "phase_build": phase_build,
        "pack": pack_result,
        "bulk": bulk,
        "prepared_slab": prepared_slab,
        "prepared_report": prepared_report,
        "selected_counts": selected_counts,
        "slab_note": slab_note,
    }


def _prepare_electrolyte_phase_inputs(
    *,
    ff,
    ion_ff,
    electrolyte: ElectrolyteSlabSpec,
    graphite_box_nm: tuple[float, float, float],
    relax: SandwichRelaxationSpec,
):
    solvent_mols = tuple(_prepare_small_molecule(spec, ff=ff, ion_ff=ion_ff) for spec in electrolyte.solvents)
    salt_cation = _prepare_small_molecule(electrolyte.salt_cation, ff=ff, ion_ff=ion_ff)
    salt_anion = _prepare_small_molecule(electrolyte.salt_anion, ff=ff, ion_ff=ion_ff)
    prep = plan_fixed_xy_direct_electrolyte_preparation(
        reference_box_nm=(
            float(graphite_box_nm[0]),
            float(graphite_box_nm[1]),
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
    mols = list(solvent_mols) + [salt_cation, salt_anion]
    charge_scale = [float(spec.charge_scale) for spec in electrolyte.solvents] + [
        float(electrolyte.salt_cation.charge_scale),
        float(electrolyte.salt_anion.charge_scale),
    ]
    return {
        "mols": mols,
        "charge_scale": charge_scale,
        "prep": prep,
    }


def _run_electrolyte_phase_round(
    *,
    ff,
    ion_ff,
    graphite_box_nm: tuple[float, float, float],
    electrolyte: ElectrolyteSlabSpec,
    relax: SandwichRelaxationSpec,
    base_phase_dir: Path,
    restart: bool | None,
) -> dict[str, object]:
    round_dir = Path(base_phase_dir)
    inputs = _prepare_electrolyte_phase_inputs(
        ff=ff,
        ion_ff=ion_ff,
        electrolyte=electrolyte,
        graphite_box_nm=graphite_box_nm,
        relax=relax,
    )
    pack_result = _run_amorphous_cell_with_density_backoff(
        label="electrolyte",
        pack_fn=poly.amorphous_cell,
        mols=list(inputs["mols"]),
        counts=list(inputs["prep"].direct_plan.target_counts),
        charge_scale=list(inputs["charge_scale"]),
        phase="electrolyte",
        target_density_g_cm3=float(electrolyte.target_density_g_cm3),
        requested_density_g_cm3=electrolyte.initial_pack_density_g_cm3,
        work_dir=round_dir / "00_build",
        retry=int(electrolyte.pack_retry),
        retry_step=int(electrolyte.pack_retry_step),
        threshold=float(electrolyte.pack_threshold_ang),
        dec_rate=float(electrolyte.pack_dec_rate),
    )
    bulk = pack_result.cell
    register_cell_species_metadata(
        bulk,
        list(inputs["mols"]),
        list(inputs["prep"].direct_plan.target_counts),
        charge_scale=list(inputs["charge_scale"]),
        pack_mode="sandwich_electrolyte_bulk",
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
        label="Electrolyte bulk",
        ac=bulk,
        work_dir=round_dir,
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
    prepared_slab, slab_note = _prepare_slab_from_equilibrated_bulk(
        label="electrolyte",
        bulk_work_dir=round_dir,
        target_lengths_nm=(float(graphite_box_nm[0]), float(graphite_box_nm[1])),
        target_thickness_nm=float(electrolyte.slab_z_nm),
        out_dir=round_dir / "05_prepare_slab",
        restart=restart,
    )
    species_names = [str(get_name(mol, default=f"EL_{idx + 1}")) for idx, mol in enumerate(inputs["mols"])]
    prepared_report = _prepared_slab_phase_report(
        label="electrolyte",
        prepared_slab=prepared_slab,
        species_names=species_names,
        target_density_g_cm3=float(electrolyte.target_density_g_cm3),
    )
    count_map = {str(name): int(count) for name, count in zip(prepared_report.species_names, prepared_report.counts)}
    selected_counts = [int(count_map.get(name, 0)) for name in species_names]
    return {
        "round_dir": round_dir,
        "inputs": inputs,
        "pack": pack_result,
        "bulk": bulk,
        "prepared_slab": prepared_slab,
        "prepared_report": prepared_report,
        "selected_counts": selected_counts,
        "slab_note": slab_note,
    }


def _append_group(groups: list[tuple[str, list[int]]], existing: dict[str, list[int]], name: str, members: Sequence[str]) -> None:
    merged: list[int] = []
    for member in members:
        merged.extend(int(idx) for idx in existing.get(str(member), []))
    merged_sorted = sorted(set(merged))
    if merged_sorted:
        groups.append((name, merged_sorted))


def _ensure_system_group_in_ndx(ndx_path: Path) -> dict[str, list[int]]:
    existing = read_ndx_groups(ndx_path)
    if "System" in existing:
        return {str(name): list(idxs) for name, idxs in existing.items()}

    candidates = (
        existing.get("SYSTEM")
        or existing.get("system")
        or sorted({int(idx) for idxs in existing.values() for idx in idxs})
    )
    merged_groups: list[tuple[str, list[int]]] = [("System", list(candidates))]
    for name, idxs in existing.items():
        if str(name) == "System":
            continue
        merged_groups.append((str(name), list(idxs)))
    _write_ndx(ndx_path, merged_groups)
    return {str(name): list(idxs) for name, idxs in merged_groups}


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
        occupied_density_g_cm3=float(profile.density_g_cm3),
        bulk_like_density_g_cm3=float(profile.density_g_cm3),
    )


def _phase_round_progress_snapshot(
    *,
    round_result: dict[str, object],
    prepared_label: str,
) -> dict[str, object]:
    prepared_report = round_result.get("prepared_report")
    pack = round_result.get("pack")
    snapshot: dict[str, object] = {
        "round_dir": str(round_result["round_dir"]),
        "selected_counts": [int(x) for x in round_result.get("selected_counts", ())],
        "prepared_slab_meta": _prepared_slab_payload(round_result["prepared_slab"]),
        "slab_note": str(round_result.get("slab_note", "")),
    }
    if pack is not None:
        snapshot["bulk_pack_summary"] = str(pack.summary_path)
        snapshot["bulk_pack"] = dict(pack.summary)
    if prepared_report is not None:
        snapshot["prepared_report"] = asdict(prepared_report)
    snapshot["label"] = str(prepared_label)
    return snapshot


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


def _prepared_slab_payload(prepared_slab) -> dict[str, object]:
    meta_path = Path(prepared_slab.meta_path)
    if not meta_path.exists():
        return {
            "box_nm": [float(prepared_slab.box_nm[0]), float(prepared_slab.box_nm[1]), float(prepared_slab.box_nm[2])],
        }
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _prepared_slab_required_xy_nm(prepared_slab) -> tuple[float, float]:
    payload = _prepared_slab_payload(prepared_slab)
    box_nm = payload.get("box_nm", prepared_slab.box_nm)
    if not isinstance(box_nm, (list, tuple)) or len(box_nm) < 2:
        return float(prepared_slab.box_nm[0]), float(prepared_slab.box_nm[1])
    return float(box_nm[0]), float(box_nm[1])


def _prepared_slab_lateral_span_nm(
    *,
    prepared_slab,
    species: Sequence,
    counts: Sequence[int],
) -> tuple[float, float]:
    fallback = _prepared_slab_required_xy_nm(prepared_slab)
    block = _load_block_from_top_gro(
        top_path=Path(prepared_slab.top_path),
        gro_path=Path(prepared_slab.gro_path),
        fallback_cell=None,
    )
    if block is None:
        return fallback
    conf = block.GetConformer(0)
    coords = np.asarray(conf.GetPositions(), dtype=float).copy()
    if coords.size == 0:
        return fallback

    current_box = getattr(block, "cell", None)
    if current_box is not None:
        box_x_ang = max(float(current_box.xhi - current_box.xlo), 1.0e-9)
        box_y_ang = max(float(current_box.yhi - current_box.ylo), 1.0e-9)
    else:
        box_x_ang = max(float(fallback[0]) * 10.0, 1.0e-9)
        box_y_ang = max(float(fallback[1]) * 10.0, 1.0e-9)

    try:
        blocks = _molecule_atom_blocks(species=species, counts=counts)
        block_specs = _molecule_block_specs(species=species, counts=counts)
    except Exception:
        return fallback
    if not blocks or not block_specs or int(blocks[-1][1]) != int(block.GetNumAtoms()):
        return fallback

    coords, _changed = _restore_bonded_periodic_fragment_coordinates(
        coords,
        block_specs=block_specs,
        box_x_ang=box_x_ang,
        box_y_ang=box_y_ang,
    )
    coords, _ = _wrap_fragment_centers_into_box(coords, blocks=blocks, axis=0, box_len_ang=box_x_ang)
    coords, _ = _minimize_fragment_periodic_axis_span(coords, blocks=blocks, axis=0, box_len_ang=box_x_ang)
    coords, _ = _wrap_fragment_centers_into_box(coords, blocks=blocks, axis=1, box_len_ang=box_y_ang)
    coords, _ = _minimize_fragment_periodic_axis_span(coords, blocks=blocks, axis=1, box_len_ang=box_y_ang)
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    spans = maxs - mins
    span_x_nm = float(spans[0]) / 10.0
    span_y_nm = float(spans[1]) / 10.0
    # Prepared slabs are periodic XY blocks. If a read-back span still exceeds the slab box,
    # treat the excess as a coordinate imaging artifact rather than expanding graphite around it.
    span_x_nm = min(span_x_nm, float(fallback[0]))
    span_y_nm = min(span_y_nm, float(fallback[1]))
    return span_x_nm, span_y_nm


def _required_xy_with_lateral_compression_floor(
    *,
    raw_xy_nm: tuple[float, float],
    target_xy_nm: tuple[float, float],
    min_scale_xy: tuple[float, float],
) -> tuple[float, float]:
    req_x = max(float(target_xy_nm[0]), float(raw_xy_nm[0]) * float(min_scale_xy[0]))
    req_y = max(float(target_xy_nm[1]), float(raw_xy_nm[1]) * float(min_scale_xy[1]))
    return float(req_x), float(req_y)


def _phase_confined_min_scale_xy(*, label: str) -> tuple[float, float]:
    phase = str(label).strip().lower()
    if phase == "electrolyte":
        return (0.60, 0.60)
    return (0.82, 0.82)


def _graphite_repeat_factors_for_required_xy(
    *,
    current_box_nm: tuple[float, float, float],
    required_xy_nm: tuple[float, float],
) -> tuple[int, int]:
    box_x = max(float(current_box_nm[0]), 1.0e-9)
    box_y = max(float(current_box_nm[1]), 1.0e-9)
    req_x = max(float(required_xy_nm[0]), 0.0)
    req_y = max(float(required_xy_nm[1]), 0.0)
    return (
        max(1, int(math.ceil(req_x / box_x))),
        max(1, int(math.ceil(req_y / box_y))),
    )


def _graphite_counts_for_required_xy(
    *,
    graphite: GraphiteSubstrateSpec,
    current_box_nm: tuple[float, float, float],
    required_xy_nm: tuple[float, float],
) -> tuple[int, int]:
    current_nx = max(1, int(graphite.nx))
    current_ny = max(1, int(graphite.ny))
    pitch_x_nm = max(float(current_box_nm[0]) / float(current_nx), 1.0e-9)
    pitch_y_nm = max(float(current_box_nm[1]) / float(current_ny), 1.0e-9)
    req_x = max(float(required_xy_nm[0]), 0.0)
    req_y = max(float(required_xy_nm[1]), 0.0)
    target_nx = max(current_nx, int(math.ceil(req_x / pitch_x_nm)))
    target_ny = max(current_ny, int(math.ceil(req_y / pitch_y_nm)))
    return int(target_nx), int(target_ny)


def _expand_graphite_to_meet_required_xy(
    *,
    graphite: GraphiteSubstrateSpec,
    graphite_result: GraphiteBuildResult,
    ff,
    required_xy_nm: tuple[float, float],
    max_adjust_rounds: int = 6,
) -> tuple[GraphiteSubstrateSpec, GraphiteBuildResult]:
    required_x = max(float(required_xy_nm[0]), float(graphite_result.box_nm[0]))
    required_y = max(float(required_xy_nm[1]), float(graphite_result.box_nm[1]))
    current_graphite = graphite
    current_result = graphite_result

    for _ in range(max(1, int(max_adjust_rounds))):
        if (
            float(current_result.box_nm[0]) >= required_x - 1.0e-6
            and float(current_result.box_nm[1]) >= required_y - 1.0e-6
        ):
            break
        next_nx = max(
            int(current_graphite.nx) + 1,
            int(math.ceil(int(current_graphite.nx) * required_x / max(float(current_result.box_nm[0]), 1.0e-9))),
        )
        next_ny = max(
            int(current_graphite.ny) + 1,
            int(math.ceil(int(current_graphite.ny) * required_y / max(float(current_result.box_nm[1]), 1.0e-9))),
        )
        if next_nx == int(current_graphite.nx) and next_ny == int(current_graphite.ny):
            break
        current_graphite = replace(current_graphite, nx=int(next_nx), ny=int(next_ny))
        current_result = build_graphite(
            nx=int(current_graphite.nx),
            ny=int(current_graphite.ny),
            n_layers=int(current_graphite.n_layers),
            orientation=current_graphite.orientation,
            edge_cap=current_graphite.edge_cap,
            ff=ff,
            name=current_graphite.name,
            top_padding_ang=float(current_graphite.top_padding_ang),
        )

    return current_graphite, current_result


def _molecule_atom_blocks(*, species: Sequence, counts: Sequence[int]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    cursor = 0
    for mol, count in zip(species, counts):
        rdmol = as_rdkit_mol(mol, strict=False)
        if rdmol is None:
            raise TypeError("species entry does not expose RDKit atom topology for block partitioning")
        nat = int(rdmol.GetNumAtoms())
        for _ in range(int(count)):
            blocks.append((cursor, cursor + nat))
            cursor += nat
    return blocks


def _molecule_block_specs(*, species: Sequence, counts: Sequence[int]) -> list[tuple[int, int, object]]:
    specs: list[tuple[int, int, object]] = []
    cursor = 0
    for mol, count in zip(species, counts):
        rdmol = as_rdkit_mol(mol, strict=False)
        if rdmol is None:
            raise TypeError("species entry does not expose RDKit atom topology for block partitioning")
        nat = int(rdmol.GetNumAtoms())
        for _ in range(int(count)):
            specs.append((cursor, cursor + nat, rdmol))
            cursor += nat
    return specs


def _unwrap_fragment_bonded_periodic_axis(fragment: np.ndarray, mol, *, axis: int, box_len_ang: float) -> tuple[np.ndarray, bool]:
    mol = as_rdkit_mol(mol, strict=False)
    out = np.asarray(fragment, dtype=float).copy()
    if mol is None:
        return out, False
    if out.size == 0 or box_len_ang <= 0.0 or int(mol.GetNumAtoms()) != int(out.shape[0]) or int(mol.GetNumBonds()) <= 0:
        return out, False

    adjacency: list[list[int]] = [[] for _ in range(int(mol.GetNumAtoms()))]
    for bond in mol.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        adjacency[a].append(b)
        adjacency[b].append(a)

    applied = False
    visited = [False] * int(mol.GetNumAtoms())
    for root in range(int(mol.GetNumAtoms())):
        if visited[root]:
            continue
        visited[root] = True
        stack = [root]
        while stack:
            atom_idx = stack.pop()
            anchor = float(out[atom_idx, axis])
            for neigh in adjacency[atom_idx]:
                if visited[neigh]:
                    continue
                delta = float(out[neigh, axis] - anchor)
                shift = float(box_len_ang) * round(delta / float(box_len_ang))
                if abs(shift) > 1.0e-9:
                    out[neigh, axis] -= shift
                    applied = True
                visited[neigh] = True
                stack.append(neigh)
    return out, applied


def _restore_bonded_periodic_fragment_coordinates(
    coords: np.ndarray,
    *,
    block_specs: Sequence[tuple[int, int, object]],
    box_x_ang: float,
    box_y_ang: float,
) -> tuple[np.ndarray, bool]:
    restored = np.asarray(coords, dtype=float).copy()
    applied = False
    for start, stop, mol in block_specs:
        frag = restored[start:stop]
        frag, changed_x = _unwrap_fragment_bonded_periodic_axis(frag, mol, axis=0, box_len_ang=float(box_x_ang))
        frag, changed_y = _unwrap_fragment_bonded_periodic_axis(frag, mol, axis=1, box_len_ang=float(box_y_ang))
        restored[start:stop] = frag
        applied = applied or bool(changed_x) or bool(changed_y)
    return restored, applied


def _wrap_fragment_centers_into_box(
    coords: np.ndarray,
    *,
    blocks: Sequence[tuple[int, int]],
    axis: int,
    box_len_ang: float,
) -> tuple[np.ndarray, bool]:
    wrapped = np.asarray(coords, dtype=float).copy()
    if wrapped.size == 0 or box_len_ang <= 0.0:
        return wrapped, False
    applied = False
    for start, stop in blocks:
        frag = wrapped[start:stop]
        if frag.size == 0:
            continue
        center = float(np.mean(frag[:, axis]))
        shift = float(box_len_ang) * math.floor(center / float(box_len_ang))
        if abs(shift) > 1.0e-9:
            frag[:, axis] -= shift
            wrapped[start:stop] = frag
            applied = True
    return wrapped, applied


def _minimize_fragment_periodic_axis_span(
    coords: np.ndarray,
    *,
    blocks: Sequence[tuple[int, int]],
    axis: int,
    box_len_ang: float,
) -> tuple[np.ndarray, bool]:
    minimized = np.asarray(coords, dtype=float).copy()
    if minimized.size == 0 or box_len_ang <= 0.0 or not blocks:
        return minimized, False

    original_span = float(np.max(minimized[:, axis]) - np.min(minimized[:, axis]))
    best = minimized
    best_span = original_span
    centers = np.asarray([float(np.mean(minimized[start:stop, axis])) % float(box_len_ang) for start, stop in blocks], dtype=float)
    order = np.argsort(centers)
    for split in range(len(order)):
        trial = minimized.copy()
        for frag_idx in order[: split + 1]:
            start, stop = blocks[int(frag_idx)]
            trial[start:stop, axis] += float(box_len_ang)
        trial_span = float(np.max(trial[:, axis]) - np.min(trial[:, axis]))
        if trial_span + 1.0e-9 < best_span:
            best = trial
            best_span = trial_span
    return best, bool(best_span + 1.0e-9 < original_span)


def _scale_block_lateral_to_target(
    coords: np.ndarray,
    *,
    target_x_ang: float,
    target_y_ang: float,
    min_scale_xy: tuple[float, float] = (0.95, 0.95),
) -> tuple[np.ndarray, tuple[float, float], bool]:
    scaled = np.asarray(coords, dtype=float).copy()
    if scaled.size == 0:
        return scaled, (1.0, 1.0), False

    mins = np.min(scaled, axis=0)
    maxs = np.max(scaled, axis=0)
    spans = maxs - mins
    scale_x = 1.0 if float(spans[0]) <= float(target_x_ang) + 1.0e-9 else float(target_x_ang) / max(float(spans[0]), 1.0e-9)
    scale_y = 1.0 if float(spans[1]) <= float(target_y_ang) + 1.0e-9 else float(target_y_ang) / max(float(spans[1]), 1.0e-9)
    if scale_x + 1.0e-9 < float(min_scale_xy[0]) or scale_y + 1.0e-9 < float(min_scale_xy[1]):
        raise RuntimeError(
            "Prepared slab requires excessive lateral compression to match the graphite footprint "
            f"(scale_x={float(scale_x):.3f}, scale_y={float(scale_y):.3f}). "
            "Expand the graphite master footprint instead of forcing the soft slab into a much smaller XY box."
        )
    if scale_x >= 1.0 - 1.0e-9 and scale_y >= 1.0 - 1.0e-9:
        return scaled, (1.0, 1.0), False

    center_x = 0.5 * float(mins[0] + maxs[0])
    center_y = 0.5 * float(mins[1] + maxs[1])
    scaled[:, 0] = center_x + (scaled[:, 0] - center_x) * float(scale_x)
    scaled[:, 1] = center_y + (scaled[:, 1] - center_y) * float(scale_y)
    return scaled, (float(scale_x), float(scale_y)), True


def _soften_catastrophic_xy_overlaps(
    coords: np.ndarray,
    *,
    box_x_ang: float,
    box_y_ang: float,
    min_sep_ang: float = 0.075,
    max_rounds: int = 4,
) -> tuple[np.ndarray, dict[str, object]]:
    softened = np.asarray(coords, dtype=float).copy()
    if softened.size == 0 or float(min_sep_ang) <= 0.0:
        return softened, {"overlap_softening_applied": False, "overlap_pairs_softened": 0}

    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        return softened, {"overlap_softening_applied": False, "overlap_pairs_softened": 0}

    offsets = (
        (0.0, 0.0),
        (-float(box_x_ang), 0.0),
        (float(box_x_ang), 0.0),
        (0.0, -float(box_y_ang)),
        (0.0, float(box_y_ang)),
        (-float(box_x_ang), -float(box_y_ang)),
        (-float(box_x_ang), float(box_y_ang)),
        (float(box_x_ang), -float(box_y_ang)),
        (float(box_x_ang), float(box_y_ang)),
    )
    total_pairs = 0
    for _ in range(max(int(max_rounds), 1)):
        tiled: list[np.ndarray] = []
        tiled_index: list[int] = []
        for ox, oy in offsets:
            pts = softened.copy()
            pts[:, 0] += float(ox)
            pts[:, 1] += float(oy)
            tiled.append(pts)
            tiled_index.extend(range(int(len(softened))))
        cloud = np.vstack(tiled)
        tree = cKDTree(cloud)
        raw_pairs = tree.query_pairs(r=float(min_sep_ang), output_type="set")
        if not raw_pairs:
            break
        corrections = np.zeros_like(softened, dtype=float)
        unique_pairs: set[tuple[int, int]] = set()
        for i, j in raw_pairs:
            ai = int(tiled_index[i])
            aj = int(tiled_index[j])
            if ai == aj:
                continue
            key = (min(ai, aj), max(ai, aj))
            if key in unique_pairs:
                continue
            unique_pairs.add(key)
            vec = np.asarray(cloud[j] - cloud[i], dtype=float)
            dist = float(np.linalg.norm(vec))
            if dist >= float(min_sep_ang):
                continue
            if dist <= 1.0e-9:
                vec = np.asarray([1.0, 0.0, 0.0], dtype=float)
                dist = 1.0e-9
            push = 0.55 * (float(min_sep_ang) - dist)
            unit = vec / dist
            corrections[ai] -= 0.5 * push * unit
            corrections[aj] += 0.5 * push * unit
        if not unique_pairs:
            break
        softened += corrections
        total_pairs += len(unique_pairs)
    return softened, {
        "overlap_softening_applied": bool(total_pairs > 0),
        "overlap_pairs_softened": int(total_pairs),
    }


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
    pre_nvt_ps = max(6.0, 0.6 * float(relax.stacked_pre_nvt_ps))
    density_relax_ps = max(20.0, 0.50 * float(relax.stacked_z_relax_ps))
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
    bonded_lateral_unwrap_applied = False
    lateral_scale_xy = (1.0, 1.0)
    blocks: list[tuple[int, int]] = []
    block_specs: list[tuple[int, int, object]] = []

    if species is not None and counts is not None:
        try:
            blocks = _molecule_atom_blocks(species=species, counts=counts)
            block_specs = _molecule_block_specs(species=species, counts=counts)
        except Exception:
            blocks = []
            block_specs = []
        if block_specs:
            box_x_ang = float(target_xy_nm[0]) * 10.0
            box_y_ang = float(target_xy_nm[1]) * 10.0
            coords, bonded_lateral_unwrap_applied = _restore_bonded_periodic_fragment_coordinates(
                coords,
                block_specs=block_specs,
                box_x_ang=box_x_ang,
                box_y_ang=box_y_ang,
            )
            coords, wrapped_x = _wrap_fragment_centers_into_box(coords, blocks=blocks, axis=0, box_len_ang=box_x_ang)
            coords, minimized_x = _minimize_fragment_periodic_axis_span(coords, blocks=blocks, axis=0, box_len_ang=box_x_ang)
            coords, wrapped_y = _wrap_fragment_centers_into_box(coords, blocks=blocks, axis=1, box_len_ang=box_y_ang)
            coords, minimized_y = _minimize_fragment_periodic_axis_span(coords, blocks=blocks, axis=1, box_len_ang=box_y_ang)
            periodic_lateral_wrap_applied = bool(
                bonded_lateral_unwrap_applied or wrapped_x or minimized_x or wrapped_y or minimized_y
            )

    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    spans = maxs - mins
    target_x_ang = float(target_xy_nm[0]) * 10.0
    target_y_ang = float(target_xy_nm[1]) * 10.0
    slot_z_ang = max(float(target_thickness_nm) * 10.0, float(spans[2]))
    box_z_ang = slot_z_ang + 2.0 * float(vacuum_padding_ang)

    if float(spans[0]) > target_x_ang + 1.0e-6 or float(spans[1]) > target_y_ang + 1.0e-6:
        coords, lateral_scale_xy, scaled_to_target = _scale_block_lateral_to_target(
            coords,
            target_x_ang=target_x_ang,
            target_y_ang=target_y_ang,
            min_scale_xy=(0.95, 0.95),
        )
        if scaled_to_target:
            mins = np.min(coords, axis=0)
            maxs = np.max(coords, axis=0)
            spans = maxs - mins
            slot_z_ang = max(float(target_thickness_nm) * 10.0, float(spans[2]))
            box_z_ang = slot_z_ang + 2.0 * float(vacuum_padding_ang)

    if (not periodic_lateral_wrap_applied) and (
        float(spans[0]) > target_x_ang + 1.0e-6 or float(spans[1]) > target_y_ang + 1.0e-6
    ):
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
    coords, overlap_summary = _soften_catastrophic_xy_overlaps(
        coords,
        box_x_ang=target_x_ang,
        box_y_ang=target_y_ang,
    )
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
        "bonded_lateral_unwrap_applied": bool(bonded_lateral_unwrap_applied),
        "lateral_scale_xy": [float(lateral_scale_xy[0]), float(lateral_scale_xy[1])],
        **overlap_summary,
    }
    note = "reboxed the prepared slab onto the graphite master footprint"
    if bonded_lateral_unwrap_applied:
        note += " and restored bonded lateral periodic coordinates"
    if periodic_lateral_wrap_applied and not bonded_lateral_unwrap_applied:
        note += " and restored lateral periodic coordinates"
    if lateral_scale_xy[0] < 1.0 - 1.0e-9 or lateral_scale_xy[1] < 1.0 - 1.0e-9:
        note += (
            " and anisotropically compressed the soft slab onto the graphite XY footprint"
            f" (scale_x={float(lateral_scale_xy[0]):.3f}, scale_y={float(lateral_scale_xy[1]):.3f})"
        )
    if bool(overlap_summary.get("overlap_softening_applied", False)):
        note += f" and softened {int(overlap_summary.get('overlap_pairs_softened', 0))} catastrophic xy-overlap pairs before confined relaxation"
    note += f" and inserted {float(vacuum_padding_ang) / 10.0:.3f} nm top/bottom vacuum before confined slab relaxation"
    return confined, summary, note


def _window_size_nm(window: object) -> float:
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        return 0.0
    try:
        lo = float(window[0])
        hi = float(window[1])
    except Exception:
        return 0.0
    return max(0.0, hi - lo)


def _phase_surface_shell_nm(summary: dict[str, object]) -> float:
    try:
        occupied = float(summary.get("occupied_thickness_nm", 0.0))
    except Exception:
        occupied = 0.0
    core = _window_size_nm(summary.get("center_bulk_like_window_nm"))
    if occupied <= 0.0:
        return 0.0
    if core <= 0.0 or core > occupied:
        return 0.5 * occupied
    return 0.5 * max(0.0, occupied - core)


def _adaptive_stack_gaps_ang(
    *,
    relax: SandwichRelaxationSpec,
    polymer_summary: dict[str, object],
    polymer_target_density_g_cm3: float | None,
    electrolyte_summary: dict[str, object],
    electrolyte_target_density_g_cm3: float | None,
) -> tuple[float, float]:
    polymer_shell_nm = _phase_surface_shell_nm(polymer_summary)
    electrolyte_shell_nm = _phase_surface_shell_nm(electrolyte_summary)
    graphite_polymer_gap_nm = (float(relax.graphite_to_polymer_gap_ang) / 10.0) + 0.35 * polymer_shell_nm
    graphite_polymer_gap_nm += _phase_gap_penalty_nm(
        polymer_summary,
        target_density_g_cm3=polymer_target_density_g_cm3,
    )
    polymer_electrolyte_gap_nm = (float(relax.polymer_to_electrolyte_gap_ang) / 10.0) + 0.35 * (
        polymer_shell_nm + electrolyte_shell_nm
    )
    polymer_electrolyte_gap_nm += _phase_gap_penalty_nm(
        polymer_summary,
        target_density_g_cm3=polymer_target_density_g_cm3,
    )
    polymer_electrolyte_gap_nm += _phase_gap_penalty_nm(
        electrolyte_summary,
        target_density_g_cm3=electrolyte_target_density_g_cm3,
    )
    return (10.0 * graphite_polymer_gap_nm, 10.0 * polymer_electrolyte_gap_nm)


def _stack_master_xy_nm(*, graphite: GraphiteSubstrateSpec, graphite_box_nm: tuple[float, float, float]) -> tuple[float, float]:
    edge_cap = str(getattr(graphite, "edge_cap", "")).strip().lower()
    seam_clearance_nm = 0.18 if edge_cap == "periodic" else 0.0
    return (
        float(graphite_box_nm[0]) + float(seam_clearance_nm),
        float(graphite_box_nm[1]) + float(seam_clearance_nm),
    )


def _maybe_expand_graphite_for_phase_footprint(
    *,
    graphite: GraphiteSubstrateSpec,
    graphite_result: GraphiteBuildResult,
    ff,
    polymer_slab,
    polymer_species: Sequence,
    polymer_counts: Sequence[int],
    electrolyte_slab,
    electrolyte_species: Sequence,
    electrolyte_counts: Sequence[int],
) -> tuple[GraphiteSubstrateSpec, GraphiteBuildResult, dict[str, object] | None]:
    polymer_xy_nm = _prepared_slab_lateral_span_nm(
        prepared_slab=polymer_slab,
        species=polymer_species,
        counts=polymer_counts,
    )
    electrolyte_xy_nm = _prepared_slab_lateral_span_nm(
        prepared_slab=electrolyte_slab,
        species=electrolyte_species,
        counts=electrolyte_counts,
    )
    current_xy_nm = (float(graphite_result.box_nm[0]), float(graphite_result.box_nm[1]))
    polymer_required_xy_nm = _required_xy_with_lateral_compression_floor(
        raw_xy_nm=polymer_xy_nm,
        target_xy_nm=current_xy_nm,
        min_scale_xy=_phase_confined_min_scale_xy(label="polymer"),
    )
    electrolyte_required_xy_nm = _required_xy_with_lateral_compression_floor(
        raw_xy_nm=electrolyte_xy_nm,
        target_xy_nm=current_xy_nm,
        min_scale_xy=_phase_confined_min_scale_xy(label="electrolyte"),
    )
    required_xy_nm = (
        max(float(graphite_result.box_nm[0]), float(polymer_required_xy_nm[0]), float(electrolyte_required_xy_nm[0])),
        max(float(graphite_result.box_nm[1]), float(polymer_required_xy_nm[1]), float(electrolyte_required_xy_nm[1])),
    )
    target_nx, target_ny = _graphite_counts_for_required_xy(
        graphite=graphite,
        current_box_nm=graphite_result.box_nm,
        required_xy_nm=required_xy_nm,
    )
    if target_nx == int(graphite.nx) and target_ny == int(graphite.ny):
        return graphite, graphite_result, None

    expanded_graphite = replace(
        graphite,
        nx=int(target_nx),
        ny=int(target_ny),
    )
    seed_result = build_graphite(
        nx=int(expanded_graphite.nx),
        ny=int(expanded_graphite.ny),
        n_layers=int(expanded_graphite.n_layers),
        orientation=expanded_graphite.orientation,
        edge_cap=expanded_graphite.edge_cap,
        ff=ff,
        name=expanded_graphite.name,
        top_padding_ang=float(expanded_graphite.top_padding_ang),
    )
    expanded_graphite, expanded_result = _expand_graphite_to_meet_required_xy(
        graphite=expanded_graphite,
        graphite_result=seed_result,
        ff=ff,
        required_xy_nm=required_xy_nm,
    )
    return expanded_graphite, expanded_result, {
        "polymer_required_xy_nm": [float(polymer_xy_nm[0]), float(polymer_xy_nm[1])],
        "electrolyte_required_xy_nm": [float(electrolyte_xy_nm[0]), float(electrolyte_xy_nm[1])],
        "polymer_compression_aware_required_xy_nm": [float(polymer_required_xy_nm[0]), float(polymer_required_xy_nm[1])],
        "electrolyte_compression_aware_required_xy_nm": [float(electrolyte_required_xy_nm[0]), float(electrolyte_required_xy_nm[1])],
        "required_xy_nm": [float(required_xy_nm[0]), float(required_xy_nm[1])],
        "graphite_counts_before_xy": [int(graphite.nx), int(graphite.ny)],
        "graphite_counts_after_xy": [int(expanded_graphite.nx), int(expanded_graphite.ny)],
        "graphite_count_scale_xy": [
            float(expanded_graphite.nx) / max(float(graphite.nx), 1.0),
            float(expanded_graphite.ny) / max(float(graphite.ny), 1.0),
        ],
        "graphite_box_before_nm": [float(x) for x in graphite_result.box_nm],
        "graphite_box_after_nm": [float(x) for x in expanded_result.box_nm],
        "polymer_min_scale_xy": [float(x) for x in _phase_confined_min_scale_xy(label="polymer")],
        "electrolyte_min_scale_xy": [float(x) for x in _phase_confined_min_scale_xy(label="electrolyte")],
    }


def _compress_phase_block_z_to_target_thickness(
    *,
    block,
    target_thickness_nm: float,
    species: Sequence | None = None,
    counts: Sequence[int] | None = None,
):
    compressed = utils.deepcopy_mol(block)
    conf = compressed.GetConformer(0)
    coords = np.asarray(conf.GetPositions(), dtype=float).copy()
    if coords.size == 0:
        return compressed, {"z_compression_applied": False, "z_compression_scale": 1.0}

    target_thickness_ang = max(float(target_thickness_nm) * 10.0, 1.0e-6)
    z_min = float(np.min(coords[:, 2]))
    z_max = float(np.max(coords[:, 2]))
    current_thickness_ang = max(z_max - z_min, 1.0e-6)
    if current_thickness_ang <= target_thickness_ang * 1.02:
        return compressed, {"z_compression_applied": False, "z_compression_scale": 1.0}

    center_z = 0.5 * (z_min + z_max)
    scale = max(0.55, min(1.0, target_thickness_ang / current_thickness_ang))
    applied = False

    blocks: list[tuple[int, int]] = []
    if species is not None and counts is not None:
        try:
            blocks = _molecule_atom_blocks(species=species, counts=counts)
        except Exception:
            blocks = []
    if blocks and int(blocks[-1][1]) == int(compressed.GetNumAtoms()):
        for start, stop in blocks:
            frag = coords[start:stop]
            if frag.size == 0:
                continue
            frag_center = float(np.mean(frag[:, 2]))
            new_center = center_z + (frag_center - center_z) * scale
            frag[:, 2] += new_center - frag_center
            coords[start:stop] = frag
        applied = True
    else:
        coords[:, 2] = center_z + (coords[:, 2] - center_z) * scale
        applied = True

    for idx, xyz in enumerate(coords):
        conf.SetAtomPosition(idx, Geom.Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return compressed, {"z_compression_applied": bool(applied), "z_compression_scale": float(scale)}


def _normalize_confined_block_for_stack(
    *,
    block,
    target_xy_nm: tuple[float, float],
    occupied_thickness_nm: float,
    species: Sequence,
    counts: Sequence[int],
):
    normalized, _summary, _note = _rebox_block_for_phase_confinement(
        block=block,
        target_xy_nm=target_xy_nm,
        target_thickness_nm=max(float(occupied_thickness_nm), 1.0e-6),
        vacuum_padding_ang=0.0,
        species=species,
        counts=counts,
    )
    return normalized


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

    resources = RunResources(
        ntmpi=int(relax.mpi),
        ntomp=int(relax.omp),
        use_gpu=bool(relax.gpu),
        gpu_id=(str(relax.gpu_id) if relax.gpu_id is not None else None),
    )

    def _run_confined_round(
        *,
        round_label: str,
        source_block,
        round_relax: SandwichRelaxationSpec,
        vacuum_padding_ang: float,
        compress_to_target: bool,
        export_dir: Path,
        relax_dir: Path,
    ) -> tuple[object, dict[str, object], Path, Path]:
        confined_block, rebox_summary, rebox_note = _rebox_block_for_phase_confinement(
            block=source_block,
            target_xy_nm=target_xy_nm,
            target_thickness_nm=float(target_thickness_nm),
            vacuum_padding_ang=float(vacuum_padding_ang),
            species=species,
            counts=counts,
        )
        compression_summary = {"z_compression_applied": False, "z_compression_scale": 1.0}
        if bool(compress_to_target):
            confined_block, compression_summary = _compress_phase_block_z_to_target_thickness(
                block=confined_block,
                target_thickness_nm=float(target_thickness_nm),
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
            out_dir=export_dir,
            ff_name=str(ff_name),
            charge_method="RESP",
            write_system_mol2=False,
        )
        _ensure_system_group_in_ndx(export.system_ndx)
        wall_atomtype, _available = _resolve_route_b_wall_atomtype(export.system_top, None)
        if wall_atomtype is None:
            raise RuntimeError(f"Could not resolve a valid wall atomtype for confined {label} slab relaxation.")
        stages = _phase_confined_relaxation_stages(relax=round_relax, wall_atomtype=wall_atomtype)
        job = EquilibrationJob(
            gro=export.system_gro,
            top=export.system_top,
            ndx=export.system_ndx,
            provenance_ndx=export.system_ndx,
            out_dir=relax_dir,
            stages=stages,
            resources=resources,
        )
        job.run(restart=bool(resolve_restart(restart)))
        relaxed_gro = relax_dir / stages[-1].name / "md.gro"
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
            "round_label": str(round_label),
            "note": str(rebox_note),
            "target_density_g_cm3": float(target_density_g_cm3),
            "target_xy_nm": [float(target_xy_nm[0]), float(target_xy_nm[1])],
            "target_thickness_nm": float(target_thickness_nm),
            "wall_atomtype": str(wall_atomtype),
            **rebox_summary,
            **compression_summary,
            **density_summary,
            "relaxed_gro": str(relaxed_gro),
            "top_path": str(export.system_top),
        }
        return relaxed_block, summary, export.system_top, relaxed_gro

    selected_block, selected_summary, selected_top_path, selected_gro = _run_confined_round(
        round_label="round_00",
        source_block=base_block,
        round_relax=relax,
        vacuum_padding_ang=max(12.0, float(relax.top_padding_ang)),
        compress_to_target=False,
        export_dir=work_dir / "00_export",
        relax_dir=work_dir / "01_relax",
    )
    round_scores = {
        "round_00": _confined_summary_score(
            summary=selected_summary,
            target_density_g_cm3=float(target_density_g_cm3),
            target_thickness_nm=float(target_thickness_nm),
        )
    }
    rescue_applied = False
    if _needs_confined_rescue(
        summary=selected_summary,
        target_density_g_cm3=float(target_density_g_cm3),
        target_thickness_nm=float(target_thickness_nm),
    ):
        rescue_relax = replace(
            relax,
            stacked_pre_nvt_ps=max(float(relax.stacked_pre_nvt_ps) + 6.0, float(relax.stacked_pre_nvt_ps) * 1.4),
            stacked_z_relax_ps=max(float(relax.stacked_z_relax_ps) + 20.0, float(relax.stacked_z_relax_ps) * 1.5),
        )
        rescue_block, rescue_summary, rescue_top_path, rescue_gro = _run_confined_round(
            round_label="round_01_rescue",
            source_block=selected_block,
            round_relax=rescue_relax,
            vacuum_padding_ang=max(10.0, 0.85 * float(relax.top_padding_ang)),
            compress_to_target=True,
            export_dir=work_dir / "02_rescue_export",
            relax_dir=work_dir / "02_rescue_relax",
        )
        round_scores["round_01_rescue"] = _confined_summary_score(
            summary=rescue_summary,
            target_density_g_cm3=float(target_density_g_cm3),
            target_thickness_nm=float(target_thickness_nm),
        )
        if round_scores["round_01_rescue"] <= round_scores["round_00"]:
            selected_block = rescue_block
            selected_summary = rescue_summary
            selected_top_path = rescue_top_path
            selected_gro = rescue_gro
            rescue_applied = True

    summary = {
        **selected_summary,
        "rescue_applied": bool(rescue_applied),
        "round_scores": {str(k): float(v) for k, v in round_scores.items()},
        "selected_round": str(selected_summary.get("round_label", "round_00")),
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
        relaxed_block=selected_block,
        report=report,
        summary=summary,
        summary_path=summary_path,
        top_path=selected_top_path,
        gro_path=selected_gro,
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
        occupied_density_g_cm3=(0.0 if density is None else float(density)),
        bulk_like_density_g_cm3=(0.0 if density is None else float(density)),
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
    progress_path = stack_dir / "sandwich_progress.json"

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
    graphite, graphite_result, preflight_graphite_negotiations = _preflight_graphite_footprint_from_phase_targets(
        graphite=graphite,
        graphite_result=graphite_result,
        ff=ff,
        ion_ff=ion_ff,
        polymer=polymer,
        electrolyte=electrolyte,
        relax=relax,
        chain_dir=polymer_chain_dir,
    )
    _write_sandwich_progress(
        progress_path,
        {
            "stage": "graphite_built",
            "graphite_box_nm": [float(x) for x in graphite_result.box_nm],
            "graphite_spec": asdict(graphite),
            "phase_preparation_rounds": 0,
            "graphite_footprint_negotiations": preflight_graphite_negotiations,
        },
    )

    graphite_negotiations: list[dict[str, object]] = list(preflight_graphite_negotiations)
    preparation_round_count = 1
    max_preparation_rounds = 3
    while True:
        round_index = int(preparation_round_count - 1)
        polymer_round_dir = _phase_round_dir(polymer_eq_dir, round_index)
        electrolyte_round_dir = _phase_round_dir(electrolyte_eq_dir, round_index)

        polymer_round = _run_polymer_phase_round(
            ff=ff,
            ion_ff=ion_ff,
            graphite_box_nm=tuple(float(x) for x in graphite_result.box_nm),
            polymer=polymer,
            relax=relax,
            chain_dir=polymer_chain_dir,
            base_phase_dir=polymer_round_dir,
            restart=restart,
        )
        electrolyte_round = _run_electrolyte_phase_round(
            ff=ff,
            ion_ff=ion_ff,
            graphite_box_nm=tuple(float(x) for x in graphite_result.box_nm),
            electrolyte=electrolyte,
            relax=relax,
            base_phase_dir=electrolyte_round_dir,
            restart=restart,
        )
        _write_sandwich_progress(
            progress_path,
            {
                "stage": "phase_preparation_round",
                "current_round_index": int(round_index + 1),
                "phase_preparation_rounds": int(preparation_round_count),
                "graphite_box_nm": [float(x) for x in graphite_result.box_nm],
                "graphite_spec": asdict(graphite),
                "graphite_footprint_negotiations": graphite_negotiations,
                "polymer_round": _phase_round_progress_snapshot(
                    round_result=polymer_round,
                    prepared_label="polymer",
                ),
                "electrolyte_round": _phase_round_progress_snapshot(
                    round_result=electrolyte_round,
                    prepared_label="electrolyte",
                ),
            },
        )

        expanded_graphite, expanded_graphite_result, graphite_negotiation = _maybe_expand_graphite_for_phase_footprint(
            graphite=graphite,
            graphite_result=graphite_result,
            ff=ff,
            polymer_slab=polymer_round["prepared_slab"],
            polymer_species=list(polymer_round["phase_build"]["species"]),
            polymer_counts=list(polymer_round["selected_counts"]),
            electrolyte_slab=electrolyte_round["prepared_slab"],
            electrolyte_species=list(electrolyte_round["inputs"]["mols"]),
            electrolyte_counts=list(electrolyte_round["selected_counts"]),
        )
        if graphite_negotiation is None:
            break
        graphite_counts_before_log = graphite_negotiation.get("graphite_counts_before_xy")
        graphite_counts_after_log = graphite_negotiation.get("graphite_counts_after_xy")
        graphite_scale_xy_log = graphite_negotiation.get("graphite_count_scale_xy")
        polymer_required_xy_log = graphite_negotiation.get("polymer_required_xy_nm")
        electrolyte_required_xy_log = graphite_negotiation.get("electrolyte_required_xy_nm")
        polymer_required_comp_log = graphite_negotiation.get("polymer_compression_aware_required_xy_nm")
        electrolyte_required_comp_log = graphite_negotiation.get("electrolyte_compression_aware_required_xy_nm")
        utils.radon_print(
            "[INFO] graphite footprint expansion requested | "
            f"graphite_counts_before_xy={None if graphite_counts_before_log is None else tuple(int(x) for x in graphite_counts_before_log)} | "
            f"graphite_counts_after_xy={None if graphite_counts_after_log is None else tuple(int(x) for x in graphite_counts_after_log)} | "
            f"graphite_count_scale_xy={None if graphite_scale_xy_log is None else tuple(float(x) for x in graphite_scale_xy_log)} | "
            f"before_nm={tuple(float(x) for x in graphite_negotiation['graphite_box_before_nm'])} | "
            f"after_nm={tuple(float(x) for x in graphite_negotiation['graphite_box_after_nm'])} | "
            f"polymer_required_xy_nm={None if polymer_required_xy_log is None else tuple(float(x) for x in polymer_required_xy_log)} | "
            f"electrolyte_required_xy_nm={None if electrolyte_required_xy_log is None else tuple(float(x) for x in electrolyte_required_xy_log)} | "
            f"polymer_compression_aware_required_xy_nm={None if polymer_required_comp_log is None else tuple(float(x) for x in polymer_required_comp_log)} | "
            f"electrolyte_compression_aware_required_xy_nm={None if electrolyte_required_comp_log is None else tuple(float(x) for x in electrolyte_required_comp_log)}",
            level=1,
        )
        graphite_negotiations.append(graphite_negotiation)
        _write_sandwich_progress(
            progress_path,
            {
                "stage": "graphite_footprint_expansion",
                "current_round_index": int(round_index + 1),
                "next_round_index": int(round_index + 2),
                "phase_preparation_rounds": int(preparation_round_count),
                "graphite_box_nm": [float(x) for x in graphite_result.box_nm],
                "graphite_spec": asdict(graphite),
                "latest_graphite_footprint_negotiation": graphite_negotiation,
                "graphite_footprint_negotiations": graphite_negotiations,
                "polymer_round": _phase_round_progress_snapshot(
                    round_result=polymer_round,
                    prepared_label="polymer",
                ),
                "electrolyte_round": _phase_round_progress_snapshot(
                    round_result=electrolyte_round,
                    prepared_label="electrolyte",
                ),
            },
        )
        if preparation_round_count >= max_preparation_rounds:
            raise RuntimeError(
                f"Could not converge graphite master footprint negotiation within {max_preparation_rounds} preparation rounds."
            )
        graphite = expanded_graphite
        graphite_result = expanded_graphite_result
        preparation_round_count += 1

    polymer_phase_build = polymer_round["phase_build"]
    polymer_chain = polymer_phase_build["chain"]
    polymer_dp = int(polymer_phase_build["dp"])
    chain_count = int(polymer_phase_build["chain_count"])
    polymer_pack = polymer_round["pack"]
    polymer_build_density = float(polymer_pack.selected_density_g_cm3)
    polymer_slab = polymer_round["prepared_slab"]
    polymer_slab_note = polymer_round["slab_note"]
    polymer_selected_counts = list(polymer_round["selected_counts"])

    electrolyte_inputs = electrolyte_round["inputs"]
    electrolyte_mols = list(electrolyte_inputs["mols"])
    electrolyte_charge_scale = list(electrolyte_inputs["charge_scale"])
    electrolyte_pack = electrolyte_round["pack"]
    electrolyte_build_density = float(electrolyte_pack.selected_density_g_cm3)
    electrolyte_slab = electrolyte_round["prepared_slab"]
    electrolyte_slab_note = electrolyte_round["slab_note"]
    electrolyte_selected_counts = list(electrolyte_round["selected_counts"])

    graphite_negotiation = graphite_negotiations[-1] if graphite_negotiations else None
    _write_sandwich_progress(
        progress_path,
        {
            "stage": "confined_phase_relaxation_start",
            "current_round_index": int(preparation_round_count),
            "phase_preparation_rounds": int(preparation_round_count),
            "graphite_box_nm": [float(x) for x in graphite_result.box_nm],
            "graphite_spec": asdict(graphite),
            "graphite_footprint_negotiations": graphite_negotiations,
            "latest_graphite_footprint_negotiation": graphite_negotiation,
            "polymer_round": _phase_round_progress_snapshot(
                round_result=polymer_round,
                prepared_label="polymer",
            ),
            "electrolyte_round": _phase_round_progress_snapshot(
                round_result=electrolyte_round,
                prepared_label="electrolyte",
            ),
        },
    )
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

    stack_master_xy_nm = _stack_master_xy_nm(
        graphite=graphite,
        graphite_box_nm=(float(graphite_result.box_nm[0]), float(graphite_result.box_nm[1]), float(graphite_result.box_nm[2])),
    )
    polymer_stack_block = _normalize_confined_block_for_stack(
        block=polymer_relaxed_block,
        target_xy_nm=stack_master_xy_nm,
        occupied_thickness_nm=float(polymer_confined.summary.get("occupied_thickness_nm", polymer.slab_z_nm)),
        species=list(polymer_phase_build["species"]),
        counts=list(polymer_selected_counts),
    )
    electrolyte_stack_block = _normalize_confined_block_for_stack(
        block=electrolyte_relaxed_block,
        target_xy_nm=stack_master_xy_nm,
        occupied_thickness_nm=float(electrolyte_confined.summary.get("occupied_thickness_nm", electrolyte.slab_z_nm)),
        species=list(electrolyte_mols),
        counts=list(electrolyte_selected_counts),
    )
    graphite_polymer_gap_ang, polymer_electrolyte_gap_ang = _adaptive_stack_gaps_ang(
        relax=relax,
        polymer_summary=polymer_confined.summary,
        polymer_target_density_g_cm3=float(polymer.target_density_g_cm3),
        electrolyte_summary=electrolyte_confined.summary,
        electrolyte_target_density_g_cm3=float(electrolyte.target_density_g_cm3),
    )
    stacked = stack_cell_blocks(
        [graphite_result.cell, polymer_stack_block, electrolyte_stack_block],
        z_gaps_ang=[float(graphite_polymer_gap_ang), float(polymer_electrolyte_gap_ang)],
        top_padding_ang=float(relax.top_padding_ang),
        fixed_xy_ang=(float(stack_master_xy_nm[0]) * 10.0, float(stack_master_xy_nm[1]) * 10.0),
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
    acceptance = _build_sandwich_acceptance(
        polymer_summary=polymer_confined.summary,
        electrolyte_summary=electrolyte_confined.summary,
        stack_checks=stack_checks,
    )

    manifest_path = stack_dir / "sandwich_manifest.json"
    notes = (
        "polymer and electrolyte were first equilibrated as standalone bulk phases, then graphite-matched slabs were cut from dense equilibrium windows before three-phase stacking",
        "each dense slab then underwent a separate fixed-XY confined pre-relaxation with z walls and explicit vacuum so the final stack no longer relies on z-periodic healing",
        "graphite stays frozen during the stacked relaxation so the liquid and polymer phases can relax density mainly along the surface normal",
        f"phase preparation rounds={int(preparation_round_count)}",
        *(
            ()
            if graphite_negotiation is None
            else (
                "graphite master footprint was automatically expanded to cover the prepared soft-phase slabs before confined relaxation",
                f"graphite footprint expansion counts={None if graphite_negotiation.get('graphite_counts_after_xy') is None else tuple(int(x) for x in graphite_negotiation['graphite_counts_after_xy'])} "
                f"from_counts={None if graphite_negotiation.get('graphite_counts_before_xy') is None else tuple(int(x) for x in graphite_negotiation['graphite_counts_before_xy'])} "
                f"from {tuple(float(x) for x in graphite_negotiation['graphite_box_before_nm'])} nm "
                f"to {tuple(float(x) for x in graphite_negotiation['graphite_box_after_nm'])} nm",
            )
        ),
        f"polymer chain target atoms={int(polymer.chain_target_atoms)} -> built DP={int(polymer_dp)} and chain_count={int(chain_count)}",
        f"polymer bulk initial pack density={float(polymer_build_density):.4f} g/cm^3 -> target equilibrium density={float(polymer.target_density_g_cm3):.4f} g/cm^3",
        f"electrolyte bulk initial pack density={float(electrolyte_build_density):.4f} g/cm^3 -> target equilibrium density={float(electrolyte.target_density_g_cm3):.4f} g/cm^3",
        f"polymer bulk pack backoff summary={polymer_pack.summary_path}",
        f"electrolyte bulk pack backoff summary={electrolyte_pack.summary_path}",
        f"stack master footprint={float(stack_master_xy_nm[0]):.4f} x {float(stack_master_xy_nm[1]):.4f} nm",
        f"adaptive stack gaps: graphite/polymer={float(graphite_polymer_gap_ang) / 10.0:.3f} nm, polymer/electrolyte={float(polymer_electrolyte_gap_ang) / 10.0:.3f} nm",
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
                "phase_preparation_rounds": int(preparation_round_count),
                "graphite_footprint_negotiation": graphite_negotiation,
                "graphite_footprint_negotiations": graphite_negotiations,
                "polymer_phase": asdict(polymer_report),
                "electrolyte_phase": asdict(electrolyte_report),
                "polymer_bulk_pack": polymer_pack.summary,
                "electrolyte_bulk_pack": electrolyte_pack.summary,
                "polymer_bulk_pack_summary": str(polymer_pack.summary_path),
                "electrolyte_bulk_pack_summary": str(electrolyte_pack.summary_path),
                "polymer_phase_confined": polymer_confined.summary,
                "electrolyte_phase_confined": electrolyte_confined.summary,
                "polymer_phase_confined_summary": str(polymer_confined.summary_path),
                "electrolyte_phase_confined_summary": str(electrolyte_confined.summary_path),
                "stack_master_xy_nm": [float(stack_master_xy_nm[0]), float(stack_master_xy_nm[1])],
                "stack_gap_ang": {
                    "graphite_to_polymer": float(graphite_polymer_gap_ang),
                    "polymer_to_electrolyte": float(polymer_electrolyte_gap_ang),
                },
                "stack_box_nm": [float(x) for x in stacked.box_nm],
                "stack_export_dir": str(stack_dir),
                "relaxed_gro": str(relaxed_gro),
                "ndx_groups": {name: len(idxs) for name, idxs in ndx_groups.items()},
                "stack_checks": stack_checks,
                "acceptance": acceptance,
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
                "phase_preparation_rounds": int(preparation_round_count),
                "graphite_footprint_negotiations": graphite_negotiations,
                "latest_graphite_footprint_negotiation": graphite_negotiation,
                "polymer_bulk_pack_summary": str(polymer_pack.summary_path),
                "electrolyte_bulk_pack_summary": str(electrolyte_pack.summary_path),
                "polymer_phase_confined_summary": str(polymer_confined.summary_path),
                "electrolyte_phase_confined_summary": str(electrolyte_confined.summary_path),
                "manifest_path": str(manifest_path),
                "relaxed_gro": str(relaxed_gro),
                "stack_checks": stack_checks,
                "acceptance": acceptance,
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
        acceptance=acceptance,
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
    "build_graphite_cmcna_glucose6_periodic_case",
    "build_graphite_cmcna_electrolyte_sandwich",
    "build_graphite_peo_electrolyte_sandwich",
    "build_graphite_polymer_electrolyte_sandwich",
    "default_carbonate_lipf6_electrolyte_spec",
    "default_cmcna_polymer_spec",
    "default_peo_electrolyte_spec",
    "default_peo_polymer_spec",
]
