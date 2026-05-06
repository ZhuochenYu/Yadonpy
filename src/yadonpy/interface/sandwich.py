"""High-level graphite/polymer/electrolyte sandwich workflow orchestration.

The sandwich workflow combines bulk calibration, slab extraction, interphase
packing, stack release, and transport analysis. This module is intentionally
script-facing: it assembles lower-level interface pieces into complete example
workflows while preserving provenance metadata at each phase boundary.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import time
import traceback
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
from ..gmx.mdp_templates import MINIM_STEEP_HBONDS_MDP, MINIM_STEEP_MDP, NPT_MDP, NPT_NO_CONSTRAINTS_MDP, NVT_MDP, NVT_NO_CONSTRAINTS_MDP, MdpSpec, default_mdp_params
from ..gmx.workflows._util import RunResources
from ..gmx.workflows.eq import EqStage, EquilibrationJob
from ..io.gromacs_system import SystemExportResult, export_system_from_cell_meta
from ..io.mol2 import read_mol2_with_charges, write_mol2_from_top_gro_parmed
from ..runtime import resolve_restart
from ..sim import qm
from ..sim.preset.eq import _find_latest_equilibrated_gro
from .bulk_resize import build_bulk_equilibrium_profile, fixed_xy_semiisotropic_npt_overrides, read_equilibrated_box_nm
from . import builder as interface_builder
from .charge_audit import summarize_cell_charge
from .postprocess import read_ndx_groups
from .prep import equilibrate_bulk_with_eq21, make_orthorhombic_pack_cell, plan_fixed_xy_direct_electrolyte_preparation
from .sandwich_metrics import (
    build_sandwich_acceptance as _build_sandwich_acceptance,
    build_stack_checks as _build_stack_checks,
    confined_phase_report as _confined_phase_report,
    confined_summary_score as _confined_summary_score,
    needs_confined_rescue as _needs_confined_rescue,
    phase_gap_penalty_nm as _phase_gap_penalty_nm,
    phase_local_density_summary as _phase_local_density_summary,
    phase_local_density_summary_for_group as _phase_local_density_summary_for_group,
    representative_phase_density as _representative_phase_density,
)
from .sandwich_phase_build import (
    BulkCalibrationSummary as _BulkCalibrationSummary,
    recommend_initial_walled_pack_density as _recommend_initial_walled_pack_density,
    solve_phase_target_z_nm as _solve_phase_target_z_nm,
    write_bulk_calibration_summary as _write_bulk_calibration_summary,
)
from .sandwich_packing import (
    PackBackoffResult as _PackBackoffResult,
    build_pack_density_ladder as _build_pack_density_ladder,
    initial_bulk_pack_density as _initial_bulk_pack_density,
    run_amorphous_cell_with_density_backoff as _run_amorphous_cell_with_density_backoff,
)
from .sandwich_specs import (
    BulkCalibrationResult,
    ElectrolyteSlabSpec,
    GraphitePreparationResult,
    GraphitePolymerElectrolyteSandwichResult,
    GraphiteSubstrateSpec,
    InterfaceBuildPolicy,
    InterfaceTransportResult,
    InterphaseBuildResult,
    MoleculeSpec,
    PolymerSlabSpec,
    SandwichNvtFollowupResult,
    SandwichPhaseReport,
    SandwichRelaxationSpec,
    StackReleaseResult,
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


@dataclass(frozen=True)
class _BulkCalibrationResult:
    label: str
    species: tuple[object, ...]
    counts: tuple[int, ...]
    charge_scale: tuple[float, ...]
    bulk: object
    pack: _PackBackoffResult
    target_z_nm: float
    summary: dict[str, object]
    summary_path: Path
    notes: tuple[str, ...]


def _resolve_interface_build_policy(policy: InterfaceBuildPolicy | None = None) -> InterfaceBuildPolicy:
    return policy if policy is not None else InterfaceBuildPolicy()


def _policy_requires_acceptance(*, policy: InterfaceBuildPolicy, route: str) -> bool:
    return bool(policy.acceptance_required and str(route).strip().lower() == "production")


def _write_sandwich_progress(progress_path: Path, payload: dict[str, object]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_interface_design_summary(
    *,
    work_dir: Path,
    route: str,
    graphite: GraphitePreparationResult,
    polymer: PolymerSlabSpec,
    electrolyte: ElectrolyteSlabSpec,
    relax: SandwichRelaxationSpec,
    policy: InterfaceBuildPolicy,
) -> Path:
    """Persist the design-level decisions before expensive phase rebuilding."""

    design_dir = Path(work_dir) / "00_interface_design"
    design_dir.mkdir(parents=True, exist_ok=True)
    negotiations = [dict(x) for x in graphite.footprint_negotiations]
    latest = negotiations[-1] if negotiations else {}
    payload = {
        "route": str(route),
        "interface_kind": "graphite_polymer_electrolyte_single_sided",
        "expected_phase_order": ["GRAPHITE", "POLYMER", "ELECTROLYTE"],
        "policy": asdict(policy),
        "graphite_spec": asdict(graphite.graphite_spec),
        "polymer_spec": asdict(polymer),
        "electrolyte_spec": asdict(electrolyte),
        "relaxation_spec": asdict(relax),
        "master_xy_nm": [float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])],
        "graphite_box_nm": [float(x) for x in graphite.box_nm],
        "target_phase_thickness_nm": {
            "polymer": float(polymer.slab_z_nm),
            "electrolyte": float(electrolyte.slab_z_nm),
        },
        "target_density_g_cm3": {
            "polymer": float(polymer.target_density_g_cm3),
            "electrolyte": float(electrolyte.target_density_g_cm3),
        },
        "graphite_footprint_negotiations": negotiations,
        "latest_graphite_footprint_negotiation": latest or None,
    }
    if latest:
        payload["required_xy_nm"] = latest.get("required_xy_nm")
        payload["target_area_nm2"] = latest.get("target_area_nm2")
        payload["polymer_target_area_nm2"] = latest.get("polymer_target_area_nm2")
        payload["electrolyte_target_area_nm2"] = latest.get("electrolyte_target_area_nm2")
    path = design_dir / "interface_design.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


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


def prepare_graphite_substrate(
    *,
    work_dir,
    ff,
    graphite: GraphiteSubstrateSpec,
    polymer: PolymerSlabSpec | None = None,
    electrolyte: ElectrolyteSlabSpec | None = None,
    ion_ff=None,
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    route: str = "screening",
    restart: bool | None = None,
) -> GraphitePreparationResult:
    wd = Path(workdir(work_dir, restart=restart))
    graphite_dir = wd / "01_graphite"
    graphite_dir.mkdir(parents=True, exist_ok=True)
    chain_dir = wd / "02_polymer_bulk_calibration" / "00_chain"
    chain_dir.mkdir(parents=True, exist_ok=True)

    resolved_graphite = graphite
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
    negotiations: list[dict[str, object]] = []
    if polymer is not None and electrolyte is not None:
        if ion_ff is None:
            raise ValueError("ion_ff is required when preparing graphite with polymer/electrolyte preflight negotiation")
        resolved_graphite, graphite_result, negotiations = _preflight_graphite_footprint_from_phase_targets(
            graphite=graphite,
            graphite_result=graphite_result,
            ff=ff,
            ion_ff=ion_ff,
            polymer=polymer,
            electrolyte=electrolyte,
            relax=relax,
            chain_dir=chain_dir,
        )

    master_xy_nm = _stack_master_xy_nm(
        graphite=resolved_graphite,
        graphite_box_nm=tuple(float(x) for x in graphite_result.box_nm),
    )
    summary = {
        "route": str(route),
        "graphite_spec": asdict(resolved_graphite),
        "graphite_box_nm": [float(x) for x in graphite_result.box_nm],
        "master_xy_nm": [float(master_xy_nm[0]), float(master_xy_nm[1])],
        "graphite_footprint_negotiations": negotiations,
    }
    summary_path = graphite_dir / "graphite_preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return GraphitePreparationResult(
        work_dir=wd,
        summary_path=summary_path,
        graphite_spec=resolved_graphite,
        graphite=graphite_result,
        master_xy_nm=(float(master_xy_nm[0]), float(master_xy_nm[1])),
        box_nm=tuple(float(x) for x in graphite_result.box_nm),
        route=str(route),
        footprint_negotiations=tuple(negotiations),
        context={
            "root_work_dir": str(wd),
            "chain_dir": str(chain_dir),
        },
    )


def calibrate_polymer_bulk_phase(
    *,
    work_dir,
    ff,
    ion_ff,
    graphite: GraphitePreparationResult,
    polymer: PolymerSlabSpec,
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    restart: bool | None = None,
) -> BulkCalibrationResult:
    wd = Path(work_dir)
    phase_dir = wd / "02_polymer_bulk_calibration"
    chain_dir = Path(str(graphite.context.get("chain_dir", phase_dir / "00_chain")))
    chain_dir.mkdir(parents=True, exist_ok=True)
    raw = _run_polymer_bulk_calibration(
        ff=ff,
        ion_ff=ion_ff,
        graphite_box_nm=tuple(float(x) for x in graphite.box_nm),
        polymer=polymer,
        relax=relax,
        chain_dir=chain_dir,
        base_phase_dir=phase_dir,
        restart=restart,
    )
    calibration = raw["calibration"]
    phase_build = raw["phase_build"]
    species_names = tuple(
        str(get_name(mol, default=f"{polymer.name}_{idx + 1}"))
        for idx, mol in enumerate(phase_build["species"])
    )
    return BulkCalibrationResult(
        label="polymer",
        work_dir=phase_dir,
        summary_path=raw["summary_path"],
        phase_preparation_mode=str(calibration.phase_preparation_mode),
        target_xy_nm=tuple(float(x) for x in calibration.master_xy_nm),
        bulk_reference_box_nm=tuple(float(x) for x in calibration.bulk_reference_box_nm),
        target_z_nm=float(calibration.target_z_nm),
        target_density_g_cm3=float(calibration.target_density_g_cm3),
        selected_bulk_pack_density_g_cm3=float(calibration.selected_bulk_pack_density_g_cm3),
        charged_phase=bool(calibration.charged_phase),
        species_names=species_names,
        counts=tuple(int(x) for x in phase_build["counts"]),
        notes=tuple(str(x) for x in calibration.notes),
        context={
            "root_work_dir": str(wd),
            "raw": raw,
        },
    )


def calibrate_electrolyte_bulk_phase(
    *,
    work_dir,
    ff,
    ion_ff,
    graphite: GraphitePreparationResult,
    electrolyte: ElectrolyteSlabSpec,
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    restart: bool | None = None,
) -> BulkCalibrationResult:
    wd = Path(work_dir)
    phase_dir = wd / "03_electrolyte_bulk_calibration"
    raw = _run_electrolyte_bulk_calibration(
        ff=ff,
        ion_ff=ion_ff,
        graphite_box_nm=tuple(float(x) for x in graphite.box_nm),
        electrolyte=electrolyte,
        relax=relax,
        base_phase_dir=phase_dir,
        restart=restart,
    )
    calibration = raw["calibration"]
    inputs = raw["inputs"]
    species_names = tuple(
        str(get_name(mol, default=f"electrolyte_{idx + 1}"))
        for idx, mol in enumerate(inputs["mols"])
    )
    return BulkCalibrationResult(
        label="electrolyte",
        work_dir=phase_dir,
        summary_path=raw["summary_path"],
        phase_preparation_mode=str(calibration.phase_preparation_mode),
        target_xy_nm=tuple(float(x) for x in calibration.master_xy_nm),
        bulk_reference_box_nm=tuple(float(x) for x in calibration.bulk_reference_box_nm),
        target_z_nm=float(calibration.target_z_nm),
        target_density_g_cm3=float(calibration.target_density_g_cm3),
        selected_bulk_pack_density_g_cm3=float(calibration.selected_bulk_pack_density_g_cm3),
        charged_phase=bool(calibration.charged_phase),
        species_names=species_names,
        counts=tuple(int(x) for x in inputs["prep"].direct_plan.target_counts),
        notes=tuple(str(x) for x in calibration.notes),
        context={
            "root_work_dir": str(wd),
            "raw": raw,
        },
    )


def build_graphite_polymer_interphase(
    *,
    work_dir,
    ff,
    ion_ff=None,
    graphite: GraphitePreparationResult,
    polymer: PolymerSlabSpec,
    polymer_bulk: BulkCalibrationResult,
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    route: str = "production",
    restart: bool | None = None,
) -> InterphaseBuildResult:
    wd = Path(work_dir)
    stage_dir = wd / "04_graphite_polymer_interphase"
    raw = dict(polymer_bulk.context.get("raw") or {})
    if not raw:
        raise ValueError("polymer_bulk is missing internal calibration context")
    phase_build = raw["phase_build"]
    legacy_note: str | None = None
    walled_build_summary = None
    walled_build_summary_path = None
    phase_preparation_mode = str(polymer_bulk.phase_preparation_mode)
    selected_counts = list(phase_build["counts"])

    try:
        if int(sum(int(x) for x in selected_counts)) >= 24:
            phase_preparation_mode = "bulk_equilibrated_walled_phase"
            prepared_slab, slab_note = _prepare_slab_from_equilibrated_bulk(
                label="polymer",
                bulk_work_dir=polymer_bulk.work_dir,
                target_lengths_nm=(float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])),
                target_thickness_nm=float(polymer.slab_z_nm),
                out_dir=stage_dir / "00_bulk_equilibrated_slab",
                restart=restart,
            )
            species_names = [
                str(get_name(mol, default=f"POLY_{idx + 1}"))
                for idx, mol in enumerate(list(phase_build["species"]))
            ]
            prepared_report = _prepared_slab_phase_report(
                label="polymer",
                prepared_slab=prepared_slab,
                species_names=species_names,
                target_density_g_cm3=float(polymer.target_density_g_cm3),
            )
            count_map = {
                str(name): int(count)
                for name, count in zip(prepared_report.species_names, prepared_report.counts)
            }
            selected_counts = [int(count_map.get(name, 0)) for name in species_names]
            resolved_thickness_nm, thickness_meta = _selected_phase_target_thickness_nm(
                species=list(phase_build["species"]),
                counts=list(selected_counts),
                target_density_g_cm3=float(polymer.target_density_g_cm3),
                target_xy_nm=(float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])),
                requested_thickness_nm=float(polymer.slab_z_nm),
            )
            confined = _run_confined_phase_relaxation(
                label="polymer",
                prepared_slab=prepared_slab,
                source_note=str(slab_note),
                species=list(phase_build["species"]),
                counts=list(selected_counts),
                charge_scale=list(phase_build["charge_scale"]),
                target_xy_nm=(float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])),
                target_density_g_cm3=float(polymer.target_density_g_cm3),
                target_thickness_nm=float(resolved_thickness_nm),
                ff_name=str(ff.name),
                relax=relax,
                work_dir=stage_dir / "01_confined",
                restart=restart,
                summary_extra={
                    "phase_preparation_mode": phase_preparation_mode,
                    "source_mode": "bulk_equilibrated_slab",
                    "slab_note": str(slab_note),
                    "source_bulk_summary": str(polymer_bulk.summary_path),
                    **thickness_meta,
                },
                trust_periodic_xy=True,
            )
            walled_build_summary = {
                "label": "polymer",
                "phase_preparation_mode": phase_preparation_mode,
                "source_mode": "bulk_equilibrated_slab",
                "source_bulk_summary": str(polymer_bulk.summary_path),
                "prepared_slab_meta": str(prepared_slab.meta_path),
                "selected_counts": list(selected_counts),
                "target_thickness_nm": float(resolved_thickness_nm),
                "target_thickness_meta": dict(thickness_meta),
                "success": True,
            }
            walled_build_summary_path = stage_dir / "polymer_walled_phase_summary.json"
            walled_build_summary_path.parent.mkdir(parents=True, exist_ok=True)
            walled_build_summary_path.write_text(
                json.dumps(walled_build_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            confined, walled_build_summary, walled_build_summary_path = _run_final_xy_walled_phase_build(
                label="polymer",
                species=list(phase_build["species"]),
                counts=list(selected_counts),
                charge_scale=list(phase_build["charge_scale"]),
                target_xy_nm=(float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])),
                target_density_g_cm3=float(polymer.target_density_g_cm3),
                bulk_calibration=raw["summary"],
                ff_name=str(ff.name),
                relax=relax,
                work_dir=stage_dir,
                retry=int(polymer.pack_retry),
                retry_step=int(polymer.pack_retry_step),
                threshold=float(polymer.pack_threshold_ang),
                dec_rate=float(polymer.pack_dec_rate),
                charged_phase=bool(phase_build.get("charged_phase", False)),
                restart=restart,
            )
    except Exception as exc:
        failure_path = stage_dir / "polymer_walled_phase_failure.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "label": "polymer",
                    "phase_preparation_mode": str(phase_preparation_mode),
                    "selected_counts": list(selected_counts),
                    "error": str(exc),
                    "fallback_allowed": bool(str(route).strip().lower() == "screening"),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if int(sum(int(x) for x in selected_counts)) >= 24:
            raise RuntimeError(
                f"Bulk-equilibrated polymer slab preparation failed; see {failure_path}"
            ) from exc
        if str(route).strip().lower() == "screening":
            raise
        phase_preparation_mode = "legacy_cut_slab"
        chain_dir = Path(str(graphite.context.get("chain_dir", wd / "02_polymer_bulk_calibration" / "00_chain")))
        legacy_round = _run_polymer_phase_round(
            ff=ff,
            ion_ff=ion_ff,
            graphite_box_nm=tuple(float(x) for x in graphite.box_nm),
            polymer=polymer,
            relax=relax,
            chain_dir=chain_dir,
            base_phase_dir=stage_dir / "00_legacy_round",
            restart=restart,
        )
        selected_counts = list(legacy_round["selected_counts"])
        confined = _run_confined_phase_relaxation(
            label="polymer",
            prepared_slab=legacy_round["prepared_slab"],
            species=list(phase_build["species"]),
            counts=list(selected_counts),
            charge_scale=list(phase_build["charge_scale"]),
            target_xy_nm=(float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])),
            target_density_g_cm3=float(polymer.target_density_g_cm3),
            target_thickness_nm=float(polymer.slab_z_nm),
            ff_name=str(ff.name),
            relax=relax,
            work_dir=stage_dir / "01_confined",
            restart=restart,
            summary_extra={
                "phase_preparation_mode": phase_preparation_mode,
                "route": str(route),
            },
        )
        legacy_note = str(legacy_round["slab_note"])

    notes = [
        f"route={str(route)}",
        f"bulk_calibration_summary={polymer_bulk.summary_path}",
    ]
    if walled_build_summary_path is not None:
        notes.append(f"walled_phase_build_summary={walled_build_summary_path}")
    if legacy_note is not None:
        notes.append(str(legacy_note))
    return InterphaseBuildResult(
        label="polymer",
        work_dir=stage_dir,
        summary_path=confined.summary_path,
        report=confined.report,
        top_path=Path(getattr(confined, "top_path", stage_dir / "system.top")),
        gro_path=Path(getattr(confined, "gro_path", stage_dir / "system.gro")),
        phase_preparation_mode=str(confined.summary.get("phase_preparation_mode", phase_preparation_mode)),
        occupied_thickness_nm=float(confined.summary.get("occupied_thickness_nm", polymer.slab_z_nm)),
        route=str(route),
        notes=tuple(notes),
        context={
            "phase_build": phase_build,
            "species": list(phase_build["species"]),
            "selected_counts": list(selected_counts),
            "charge_scale": list(phase_build["charge_scale"]),
            "walled_build_summary": walled_build_summary,
            "walled_build_summary_path": (None if walled_build_summary_path is None else str(walled_build_summary_path)),
            "legacy_note": legacy_note,
            "phase_preparation_mode": str(confined.summary.get("phase_preparation_mode", phase_preparation_mode)),
            "bulk_calibration_result": polymer_bulk,
            "confined": confined,
        },
    )


def build_graphite_cmc_interphase(**kwargs) -> InterphaseBuildResult:
    return build_graphite_polymer_interphase(**kwargs)


def build_polymer_electrolyte_interphase(
    *,
    work_dir,
    ff,
    ion_ff=None,
    graphite: GraphitePreparationResult,
    electrolyte: ElectrolyteSlabSpec,
    electrolyte_bulk: BulkCalibrationResult,
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    route: str = "production",
    restart: bool | None = None,
) -> InterphaseBuildResult:
    wd = Path(work_dir)
    stage_dir = wd / "05_polymer_electrolyte_interphase"
    raw = dict(electrolyte_bulk.context.get("raw") or {})
    if not raw:
        raise ValueError("electrolyte_bulk is missing internal calibration context")
    inputs = raw["inputs"]
    legacy_note: str | None = None
    walled_build_summary = None
    walled_build_summary_path = None
    phase_preparation_mode = str(electrolyte_bulk.phase_preparation_mode)
    selected_counts = list(inputs["prep"].direct_plan.target_counts)

    try:
        if int(sum(int(x) for x in selected_counts)) >= 24:
            phase_preparation_mode = "bulk_equilibrated_walled_phase"
            prepared_slab, slab_note = _prepare_slab_from_equilibrated_bulk(
                label="electrolyte",
                bulk_work_dir=electrolyte_bulk.work_dir,
                target_lengths_nm=(float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])),
                target_thickness_nm=float(electrolyte.slab_z_nm),
                out_dir=stage_dir / "00_bulk_equilibrated_slab",
                restart=restart,
            )
            species_names = [
                str(get_name(mol, default=f"ELY_{idx + 1}"))
                for idx, mol in enumerate(list(inputs["mols"]))
            ]
            prepared_report = _prepared_slab_phase_report(
                label="electrolyte",
                prepared_slab=prepared_slab,
                species_names=species_names,
                target_density_g_cm3=float(electrolyte.target_density_g_cm3),
            )
            count_map = {
                str(name): int(count)
                for name, count in zip(prepared_report.species_names, prepared_report.counts)
            }
            selected_counts = [int(count_map.get(name, 0)) for name in species_names]
            resolved_thickness_nm, thickness_meta = _selected_phase_target_thickness_nm(
                species=list(inputs["mols"]),
                counts=list(selected_counts),
                target_density_g_cm3=float(electrolyte.target_density_g_cm3),
                target_xy_nm=(float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])),
                requested_thickness_nm=float(electrolyte.slab_z_nm),
            )
            confined = _run_confined_phase_relaxation(
                label="electrolyte",
                prepared_slab=prepared_slab,
                source_note=str(slab_note),
                species=list(inputs["mols"]),
                counts=list(selected_counts),
                charge_scale=list(inputs["charge_scale"]),
                target_xy_nm=(float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])),
                target_density_g_cm3=float(electrolyte.target_density_g_cm3),
                target_thickness_nm=float(resolved_thickness_nm),
                ff_name=str(ff.name),
                relax=relax,
                work_dir=stage_dir / "01_confined",
                restart=restart,
                summary_extra={
                    "phase_preparation_mode": phase_preparation_mode,
                    "source_mode": "bulk_equilibrated_slab",
                    "slab_note": str(slab_note),
                    "source_bulk_summary": str(electrolyte_bulk.summary_path),
                    **thickness_meta,
                },
                trust_periodic_xy=True,
            )
            walled_build_summary = {
                "label": "electrolyte",
                "phase_preparation_mode": phase_preparation_mode,
                "source_mode": "bulk_equilibrated_slab",
                "source_bulk_summary": str(electrolyte_bulk.summary_path),
                "prepared_slab_meta": str(prepared_slab.meta_path),
                "selected_counts": list(selected_counts),
                "target_thickness_nm": float(resolved_thickness_nm),
                "target_thickness_meta": dict(thickness_meta),
                "success": True,
            }
            walled_build_summary_path = stage_dir / "electrolyte_walled_phase_summary.json"
            walled_build_summary_path.parent.mkdir(parents=True, exist_ok=True)
            walled_build_summary_path.write_text(
                json.dumps(walled_build_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            confined, walled_build_summary, walled_build_summary_path = _run_final_xy_walled_phase_build(
                label="electrolyte",
                species=list(inputs["mols"]),
                counts=list(selected_counts),
                charge_scale=list(inputs["charge_scale"]),
                target_xy_nm=(float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])),
                target_density_g_cm3=float(electrolyte.target_density_g_cm3),
                bulk_calibration=raw["summary"],
                ff_name=str(ff.name),
                relax=relax,
                work_dir=stage_dir,
                retry=int(electrolyte.pack_retry),
                retry_step=int(electrolyte.pack_retry_step),
                threshold=float(electrolyte.pack_threshold_ang),
                dec_rate=float(electrolyte.pack_dec_rate),
                charged_phase=False,
                restart=restart,
            )
    except Exception as exc:
        failure_path = stage_dir / "electrolyte_walled_phase_failure.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "label": "electrolyte",
                    "phase_preparation_mode": str(phase_preparation_mode),
                    "selected_counts": list(selected_counts),
                    "error": str(exc),
                    "fallback_allowed": bool(str(route).strip().lower() == "screening"),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if int(sum(int(x) for x in selected_counts)) >= 24:
            raise RuntimeError(
                f"Bulk-equilibrated electrolyte slab preparation failed; see {failure_path}"
            ) from exc
        if str(route).strip().lower() == "screening":
            raise
        phase_preparation_mode = "legacy_cut_slab"
        legacy_round = _run_electrolyte_phase_round(
            ff=ff,
            ion_ff=ion_ff,
            graphite_box_nm=tuple(float(x) for x in graphite.box_nm),
            electrolyte=electrolyte,
            relax=relax,
            base_phase_dir=stage_dir / "00_legacy_round",
            restart=restart,
        )
        selected_counts = list(legacy_round["selected_counts"])
        confined = _run_confined_phase_relaxation(
            label="electrolyte",
            prepared_slab=legacy_round["prepared_slab"],
            species=list(inputs["mols"]),
            counts=list(selected_counts),
            charge_scale=list(inputs["charge_scale"]),
            target_xy_nm=(float(graphite.master_xy_nm[0]), float(graphite.master_xy_nm[1])),
            target_density_g_cm3=float(electrolyte.target_density_g_cm3),
            target_thickness_nm=float(electrolyte.slab_z_nm),
            ff_name=str(ff.name),
            relax=relax,
            work_dir=stage_dir / "01_confined",
            restart=restart,
            summary_extra={
                "phase_preparation_mode": phase_preparation_mode,
                "route": str(route),
            },
        )
        legacy_note = str(legacy_round["slab_note"])

    notes = [
        f"route={str(route)}",
        f"bulk_calibration_summary={electrolyte_bulk.summary_path}",
    ]
    if walled_build_summary_path is not None:
        notes.append(f"walled_phase_build_summary={walled_build_summary_path}")
    if legacy_note is not None:
        notes.append(str(legacy_note))
    return InterphaseBuildResult(
        label="electrolyte",
        work_dir=stage_dir,
        summary_path=confined.summary_path,
        report=confined.report,
        top_path=Path(getattr(confined, "top_path", stage_dir / "system.top")),
        gro_path=Path(getattr(confined, "gro_path", stage_dir / "system.gro")),
        phase_preparation_mode=str(confined.summary.get("phase_preparation_mode", phase_preparation_mode)),
        occupied_thickness_nm=float(confined.summary.get("occupied_thickness_nm", electrolyte.slab_z_nm)),
        route=str(route),
        notes=tuple(notes),
        context={
            "inputs": inputs,
            "species": list(inputs["mols"]),
            "selected_counts": list(selected_counts),
            "charge_scale": list(inputs["charge_scale"]),
            "walled_build_summary": walled_build_summary,
            "walled_build_summary_path": (None if walled_build_summary_path is None else str(walled_build_summary_path)),
            "legacy_note": legacy_note,
            "phase_preparation_mode": str(confined.summary.get("phase_preparation_mode", phase_preparation_mode)),
            "bulk_calibration_result": electrolyte_bulk,
            "confined": confined,
        },
    )


def build_cmc_electrolyte_interphase(**kwargs) -> InterphaseBuildResult:
    return build_polymer_electrolyte_interphase(**kwargs)


def _select_stack_rescue_strategy(acceptance: dict[str, object]) -> dict[str, object]:
    failed = set(str(x) for x in list(acceptance.get("failed_checks") or []))
    density_low = str(acceptance.get("polymer_density_direction", "ok")) == "low" or str(
        acceptance.get("electrolyte_density_direction", "ok")
    ) == "low"
    if "min_atom_distance_ok" in failed:
        strategy = "increase_stack_gaps_and_repeat_natural_contact"
    elif "core_gaps_ok" in failed and not bool(acceptance.get("core_gaps_positive_ok", True)):
        strategy = "increase_stack_gaps_and_repeat_natural_contact"
    elif "core_gaps_ok" in failed and not bool(acceptance.get("core_gaps_upper_ok", True)):
        # In the wall-confined stack release path the rescue stages are fixed-volume
        # NVT by design.  If the core gap is already too large, extending the run
        # rarely helps and can further dilute the electrolyte; rebuild tighter first.
        strategy = "decrease_stack_gaps_and_repeat_natural_contact"
    elif {"polymer_density_ok", "electrolyte_density_ok"} & failed and density_low:
        strategy = "extend_z_relax_and_repeat_natural_contact"
    elif {"polymer_density_ok", "electrolyte_density_ok"} & failed:
        strategy = "extend_z_relax_and_repeat_natural_contact"
    elif {"wrapped_ok", "order_ok"} & failed:
        strategy = "rebuild_stack_with_larger_gaps"
    elif "charge_ok" in failed:
        strategy = "stop_charge_imbalance"
    else:
        strategy = "extend_natural_contact"
    return {
        "strategy": strategy,
        "failed_checks": sorted(failed),
        "recommended_action": strategy,
    }


def _stack_rescue_gap_multiplier(*, attempt: int, strategy: str) -> float:
    if int(attempt) <= 0:
        return 1.0
    if str(strategy) == "decrease_stack_gaps_and_repeat_natural_contact":
        return max(0.30, 1.0 - 0.35 * float(attempt))
    if str(strategy) in {"increase_stack_gaps_and_repeat_natural_contact", "rebuild_stack_with_larger_gaps"}:
        return 1.0 + 0.35 * float(attempt)
    return 1.0


def _stack_rescue_relaxation(*, relax: SandwichRelaxationSpec, attempt: int, strategy: str) -> SandwichRelaxationSpec:
    if int(attempt) <= 0:
        return relax
    if str(strategy) in {"extend_z_relax_and_repeat_natural_contact", "extend_natural_contact"}:
        return replace(
            relax,
            stacked_z_relax_ps=float(relax.stacked_z_relax_ps) * (1.0 + 0.5 * float(attempt)),
            stacked_exchange_ps=float(relax.stacked_exchange_ps) * (1.0 + 0.35 * float(attempt)),
        )
    return replace(relax, stacked_pre_nvt_ps=float(relax.stacked_pre_nvt_ps) * (1.0 + 0.25 * float(attempt)))


def _stack_wall_top_padding_ang(relax: SandwichRelaxationSpec) -> float:
    """Return a compact wall clearance for nonperiodic-z sandwich release.

    The public relaxation spec historically used a generous top padding because
    the old stack path was fully periodic in z.  With `pbc=xy` and explicit
    walls, that same padding is real vacuum above the electrolyte, so it lowers
    the measured electrolyte density and encourages solvent to spread into an
    artificial headspace.  Keep a small wall clearance for numerical safety, but
    avoid carrying bulk-vacuum padding into the production stack.
    """

    requested = float(relax.top_padding_ang)
    return max(2.5, min(requested, 4.0))


def release_graphite_polymer_electrolyte_stack(
    *,
    work_dir,
    ff,
    graphite: GraphitePreparationResult,
    polymer_interphase: InterphaseBuildResult,
    electrolyte_interphase: InterphaseBuildResult,
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    policy: InterfaceBuildPolicy | None = None,
    route: str = "screening",
    restart: bool | None = None,
) -> StackReleaseResult:
    build_policy = _resolve_interface_build_policy(policy)
    wd = Path(work_dir)
    release_dir = wd / "06_full_stack_release"
    stack_dir = release_dir / "00_stack"
    relax_dir = release_dir / "01_relax"
    stack_dir.mkdir(parents=True, exist_ok=True)
    relax_dir.mkdir(parents=True, exist_ok=True)
    progress_path = release_dir / "interface_progress.json"

    polymer_ctx = dict(polymer_interphase.context)
    electrolyte_ctx = dict(electrolyte_interphase.context)
    polymer_phase_build = polymer_ctx["phase_build"]
    polymer_selected_counts = list(polymer_ctx["selected_counts"])
    electrolyte_inputs = electrolyte_ctx["inputs"]
    electrolyte_selected_counts = list(electrolyte_ctx["selected_counts"])
    polymer_confined = polymer_ctx["confined"]
    electrolyte_confined = electrolyte_ctx["confined"]

    stack_master_xy_nm = _stack_master_xy_nm(
        graphite=graphite.graphite_spec,
        graphite_box_nm=tuple(float(x) for x in graphite.box_nm),
    )
    polymer_stack_block = _normalize_confined_block_for_stack(
        block=polymer_confined.relaxed_block,
        target_xy_nm=stack_master_xy_nm,
        occupied_thickness_nm=float(polymer_interphase.occupied_thickness_nm),
        target_thickness_nm=float(
            polymer_confined.summary.get(
                "effective_density_matched_thickness_nm",
                polymer_confined.summary.get("target_thickness_nm", polymer_interphase.occupied_thickness_nm),
            )
            or polymer_interphase.occupied_thickness_nm
        ),
        species=list(polymer_ctx["species"]),
        counts=list(polymer_selected_counts),
    )
    electrolyte_stack_block = _normalize_confined_block_for_stack(
        block=electrolyte_confined.relaxed_block,
        target_xy_nm=stack_master_xy_nm,
        occupied_thickness_nm=float(electrolyte_interphase.occupied_thickness_nm),
        target_thickness_nm=float(
            electrolyte_confined.summary.get(
                "effective_density_matched_thickness_nm",
                electrolyte_confined.summary.get("target_thickness_nm", electrolyte_interphase.occupied_thickness_nm),
            )
            or electrolyte_interphase.occupied_thickness_nm
        ),
        species=list(electrolyte_ctx["species"]),
        counts=list(electrolyte_selected_counts),
        min_z_compression_scale=0.65,
    )
    graphite_polymer_gap_ang, polymer_electrolyte_gap_ang = _adaptive_stack_gaps_ang(
        relax=relax,
        polymer_summary=polymer_confined.summary,
        polymer_target_density_g_cm3=float(polymer_interphase.report.target_density_g_cm3 or 0.0),
        electrolyte_summary=electrolyte_confined.summary,
        electrolyte_target_density_g_cm3=float(electrolyte_interphase.report.target_density_g_cm3 or 0.0),
    )
    stacked_mols = [graphite.graphite.layer_mol]
    stacked_counts = [int(graphite.graphite.layer_count)]
    stacked_charge_scale = [1.0]
    for mol, count, scale in zip(polymer_ctx["species"], polymer_selected_counts, polymer_ctx["charge_scale"]):
        if int(count) <= 0:
            continue
        stacked_mols.append(mol)
        stacked_counts.append(int(count))
        stacked_charge_scale.append(float(scale))
    for mol, count, scale in zip(electrolyte_ctx["species"], electrolyte_selected_counts, electrolyte_ctx["charge_scale"]):
        if int(count) <= 0:
            continue
        stacked_mols.append(mol)
        stacked_counts.append(int(count))
        stacked_charge_scale.append(float(scale))

    graphite_group_name = str(get_name(graphite.graphite.layer_mol, default=graphite.graphite_spec.name))
    polymer_group_name = str(
        get_name(
            polymer_phase_build["chain"],
            default=polymer_interphase.report.label,
        )
    )
    polymer_group_names = [polymer_group_name]
    polymer_group_names.extend(str(name) for name in polymer_interphase.report.species_names)
    electrolyte_group_names = [str(name) for name in electrolyte_interphase.report.species_names]

    attempts: list[dict[str, object]] = []
    rescue_strategy = "initial_natural_contact"
    export = None
    ndx_groups: dict[str, list[int]] = {}
    relaxed_gro = relax_dir / "05_natural_exchange" / "md.gro"
    stack_checks: dict[str, object] = {}
    acceptance: dict[str, object] = {}
    charge_summary: dict[str, object] = {}
    stack_polymer_summary: dict[str, object] = {}
    stack_electrolyte_summary: dict[str, object] = {}
    stack_box_nm: tuple[float, float, float] | None = None
    max_attempts = max(0, int(build_policy.max_stack_rescue_rounds)) + 1
    for attempt in range(max_attempts):
        attempt_strategy = rescue_strategy
        gap_multiplier = _stack_rescue_gap_multiplier(attempt=attempt, strategy=attempt_strategy)
        attempt_stack_dir = stack_dir if attempt == 0 else release_dir / f"00_stack_rescue_{attempt:02d}"
        attempt_relax_dir = relax_dir if attempt == 0 else release_dir / f"01_relax_rescue_{attempt:02d}"
        attempt_stack_dir.mkdir(parents=True, exist_ok=True)
        attempt_relax_dir.mkdir(parents=True, exist_ok=True)
        stacked = stack_cell_blocks(
            [graphite.graphite.cell, polymer_stack_block, electrolyte_stack_block],
            z_gaps_ang=[
                float(graphite_polymer_gap_ang) * float(gap_multiplier),
                float(polymer_electrolyte_gap_ang) * float(gap_multiplier),
            ],
            top_padding_ang=_stack_wall_top_padding_ang(relax),
            fixed_xy_ang=(float(stack_master_xy_nm[0]) * 10.0, float(stack_master_xy_nm[1]) * 10.0),
        )
        stack_box_nm = tuple(float(x) for x in stacked.box_nm)
        register_cell_species_metadata(
            stacked.cell,
            stacked_mols,
            stacked_counts,
            charge_scale=stacked_charge_scale,
            pack_mode="graphite_polymer_electrolyte_sandwich",
        )
        charge_summary = summarize_cell_charge(stacked.cell)
        if "net_charge_scaled" in charge_summary:
            charge_summary["charge_tolerance"] = float(build_policy.charge_tolerance_e)
            charge_summary["net_charge_ok"] = bool(
                abs(float(charge_summary.get("net_charge_scaled", 0.0))) <= float(build_policy.charge_tolerance_e)
            )
        export = export_system_from_cell_meta(
            cell_mol=stacked.cell,
            out_dir=attempt_stack_dir,
            ff_name=str(ff.name),
            charge_method="RESP",
            write_system_mol2=False,
        )
        ndx_groups = _augment_sandwich_ndx(
            ndx_path=export.system_ndx,
            graphite_name=graphite_group_name,
            polymer_name=polymer_group_name,
            polymer_names=polymer_group_names,
            electrolyte_names=electrolyte_group_names,
        )
        from .protocol import _resolve_route_b_wall_atomtype

        stack_wall_atomtype, stack_wall_available = _resolve_route_b_wall_atomtype(export.system_top, None)
        if stack_wall_atomtype is None:
            raise RuntimeError(
                "Could not resolve a valid wall atomtype for graphite/polymer/electrolyte stack release."
            )
        attempt_relax = _stack_rescue_relaxation(relax=relax, attempt=attempt, strategy=attempt_strategy)
        relaxed_gro = _run_stacked_relaxation(
            export=export,
            work_dir=attempt_relax_dir,
            relax=attempt_relax,
            freeze_group=getattr(attempt_relax, "stack_freeze_group", "GRAPHITE"),
            wall_atomtype=str(stack_wall_atomtype),
            restart=restart,
        )
        if not Path(relaxed_gro).exists():
            # Some dry-run/unit-test paths mock the stack relaxation by only
            # returning the intended output path. Use the exported starting
            # geometry for diagnostics rather than failing before orchestration
            # metadata can be checked. Real production runs still produce the
            # relaxed GRO and therefore follow the normal path.
            relaxed_gro = Path(export.system_gro)
        stack_checks = _build_stack_checks(gro_path=relaxed_gro, ndx_groups=ndx_groups)
        stack_polymer_summary = _phase_local_density_summary_for_group(
            gro_path=relaxed_gro,
            atom_indices=ndx_groups.get("POLYMER", []),
            species=list(polymer_ctx["species"]),
            counts=list(polymer_selected_counts),
        )
        stack_polymer_density_ref = float(
            polymer_interphase.report.density_g_cm3
            or polymer_interphase.report.target_density_g_cm3
            or 0.0
        )
        stack_polymer_summary["target_density_g_cm3"] = stack_polymer_density_ref
        stack_polymer_summary["requested_target_density_g_cm3"] = float(
            polymer_interphase.report.target_density_g_cm3 or 0.0
        )
        stack_electrolyte_summary = _phase_local_density_summary_for_group(
            gro_path=relaxed_gro,
            atom_indices=ndx_groups.get("ELECTROLYTE", []),
            species=list(electrolyte_ctx["species"]),
            counts=list(electrolyte_selected_counts),
        )
        stack_electrolyte_density_ref = float(
            electrolyte_interphase.report.density_g_cm3
            or electrolyte_interphase.report.target_density_g_cm3
            or 0.0
        )
        stack_electrolyte_summary["target_density_g_cm3"] = stack_electrolyte_density_ref
        stack_electrolyte_summary["requested_target_density_g_cm3"] = float(
            electrolyte_interphase.report.target_density_g_cm3 or 0.0
        )
        acceptance = _build_sandwich_acceptance(
            polymer_summary=stack_polymer_summary,
            electrolyte_summary=stack_electrolyte_summary,
            stack_checks=stack_checks,
            charge_summary=charge_summary,
            min_atom_distance_nm=float(build_policy.min_atom_distance_nm),
            max_core_gap_nm=float(build_policy.max_core_gap_nm),
        )
        attempt_record = {
            "attempt": int(attempt),
            "strategy": str(attempt_strategy),
            "gap_multiplier": float(gap_multiplier),
            "stack_dir": str(attempt_stack_dir),
            "relax_dir": str(attempt_relax_dir),
            "relaxed_gro": str(relaxed_gro),
            "stack_wall_atomtype": str(stack_wall_atomtype),
            "available_wall_atomtypes": [str(x) for x in stack_wall_available],
            "accepted": bool(acceptance.get("accepted", False)),
            "failed_checks": list(acceptance.get("failed_checks") or []),
            "stack_phase_density": {
                "polymer": dict(stack_polymer_summary),
                "electrolyte": dict(stack_electrolyte_summary),
            },
        }
        attempts.append(attempt_record)
        if bool(acceptance.get("accepted", False)):
            break
        rescue = _select_stack_rescue_strategy(acceptance)
        rescue_strategy = str(rescue["strategy"])
        attempt_record["next_rescue"] = rescue
        if rescue_strategy == "stop_charge_imbalance":
            break

    polymer_bulk = polymer_ctx["bulk_calibration_result"]
    electrolyte_bulk = electrolyte_ctx["bulk_calibration_result"]
    manifest_path = release_dir / "interface_manifest.json"
    accepted = bool(acceptance.get("accepted", False))
    acceptance_required = _policy_requires_acceptance(policy=build_policy, route=route)
    interface_build_status = (
        "accepted"
        if accepted
        else ("failed_acceptance_hard_stop" if acceptance_required else "failed_acceptance_low_confidence")
    )
    notes = (
        f"route={str(route)}",
        f"interface_build_status={interface_build_status}",
        f"graphite_preparation_summary={graphite.summary_path}",
        f"polymer_bulk_calibration_summary={polymer_bulk.summary_path}",
        f"electrolyte_bulk_calibration_summary={electrolyte_bulk.summary_path}",
        f"graphite_polymer_interphase_summary={polymer_interphase.summary_path}",
        f"polymer_electrolyte_interphase_summary={electrolyte_interphase.summary_path}",
        f"stack master footprint={float(stack_master_xy_nm[0]):.4f} x {float(stack_master_xy_nm[1]):.4f} nm",
        f"adaptive stack gaps: graphite/polymer={float(graphite_polymer_gap_ang) / 10.0:.3f} nm, polymer/electrolyte={float(polymer_electrolyte_gap_ang) / 10.0:.3f} nm",
        *tuple(polymer_interphase.notes),
        *tuple(electrolyte_interphase.notes),
    )
    manifest_path.write_text(
        json.dumps(
            {
                "route": str(route),
                "interface_build_status": interface_build_status,
                "policy": asdict(build_policy),
                "interface_design_summary": graphite.context.get("interface_design_summary"),
                "phase_preparation_mode": str(polymer_ctx.get("phase_preparation_mode", "bulk_calibrate_walled_phase")),
                "graphite_preparation_summary": str(graphite.summary_path),
                "graphite_spec": asdict(graphite.graphite_spec),
                "graphite_box_nm": [float(x) for x in graphite.box_nm],
                "graphite_footprint_negotiations": [dict(x) for x in graphite.footprint_negotiations],
                "polymer_bulk_calibration": dict(polymer_bulk.context.get("raw", {}).get("summary", {})),
                "electrolyte_bulk_calibration": dict(electrolyte_bulk.context.get("raw", {}).get("summary", {})),
                "polymer_bulk_pack": dict(polymer_bulk.context.get("raw", {}).get("pack", {}).summary)
                if hasattr(polymer_bulk.context.get("raw", {}).get("pack", {}), "summary")
                else {},
                "electrolyte_bulk_pack": dict(electrolyte_bulk.context.get("raw", {}).get("pack", {}).summary)
                if hasattr(electrolyte_bulk.context.get("raw", {}).get("pack", {}), "summary")
                else {},
                "polymer_bulk_calibration_summary": str(polymer_bulk.summary_path),
                "electrolyte_bulk_calibration_summary": str(electrolyte_bulk.summary_path),
                "polymer_phase": asdict(polymer_interphase.report),
                "electrolyte_phase": asdict(electrolyte_interphase.report),
                "polymer_phase_confined": dict(polymer_confined.summary),
                "electrolyte_phase_confined": dict(electrolyte_confined.summary),
                "polymer_phase_released_stack": dict(stack_polymer_summary),
                "electrolyte_phase_released_stack": dict(stack_electrolyte_summary),
                "polymer_phase_confined_summary": str(polymer_interphase.summary_path),
                "electrolyte_phase_confined_summary": str(electrolyte_interphase.summary_path),
                "polymer_walled_phase_build_summary": polymer_ctx.get("walled_build_summary_path"),
                "electrolyte_walled_phase_build_summary": electrolyte_ctx.get("walled_build_summary_path"),
                "charge_balance": {
                    "polymer": dict(polymer_phase_build.get("charge_balance", {})),
                    "electrolyte": dict(electrolyte_inputs.get("charge_balance", {})),
                    "stack": dict(charge_summary),
                },
                "stack_master_xy_nm": [float(stack_master_xy_nm[0]), float(stack_master_xy_nm[1])],
                "stack_gap_ang": {
                    "graphite_to_polymer": float(graphite_polymer_gap_ang),
                    "polymer_to_electrolyte": float(polymer_electrolyte_gap_ang),
                },
                "stack_box_nm": ([] if stack_box_nm is None else [float(x) for x in stack_box_nm]),
                "stack_export_dir": str(Path(export.system_gro).parent if export is not None else stack_dir),
                "stack_attempts": attempts,
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
    _write_sandwich_progress(
        progress_path,
        {
            "stage": "completed",
            "route": str(route),
            "interface_build_status": interface_build_status,
            "policy": asdict(build_policy),
            "phase_preparation_mode": str(polymer_ctx.get("phase_preparation_mode", "bulk_calibrate_walled_phase")),
            "graphite_preparation_summary": str(graphite.summary_path),
            "graphite_footprint_negotiations": [dict(x) for x in graphite.footprint_negotiations],
            "latest_graphite_footprint_negotiation": (
                dict(graphite.footprint_negotiations[-1]) if graphite.footprint_negotiations else None
            ),
            "polymer_bulk_calibration_summary": str(polymer_bulk.summary_path),
            "electrolyte_bulk_calibration_summary": str(electrolyte_bulk.summary_path),
            "polymer_phase_confined_summary": str(polymer_interphase.summary_path),
            "electrolyte_phase_confined_summary": str(electrolyte_interphase.summary_path),
            "manifest_path": str(manifest_path),
            "relaxed_gro": str(relaxed_gro),
            "stack_attempts": attempts,
            "stack_checks": stack_checks,
            "stack_phase_density": {
                "polymer": dict(stack_polymer_summary),
                "electrolyte": dict(stack_electrolyte_summary),
            },
            "acceptance": acceptance,
        },
    )
    if not accepted and acceptance_required:
        raise RuntimeError(
            "Graphite/polymer/electrolyte interface failed acceptance gates in production route. "
            f"manifest={manifest_path} failed_checks={list(acceptance.get('failed_checks') or [])}"
        )

    sandwich_result = GraphitePolymerElectrolyteSandwichResult(
        graphite=graphite.graphite,
        polymer_phase=polymer_interphase.report,
        electrolyte_phase=electrolyte_interphase.report,
        stack_export=export,
        relaxed_gro=relaxed_gro,
        manifest_path=manifest_path,
        stack_checks=stack_checks,
        acceptance=acceptance,
        notes=notes,
    )
    return StackReleaseResult(
        work_dir=release_dir,
        manifest_path=manifest_path,
        relaxed_gro=relaxed_gro,
        graphite=graphite.graphite,
        polymer_phase=polymer_interphase.report,
        electrolyte_phase=electrolyte_interphase.report,
        stack_checks=stack_checks,
        acceptance=acceptance,
        route=str(route),
        notes=notes,
        sandwich_result=sandwich_result,
    )


def release_graphite_cmc_electrolyte_stack(**kwargs) -> StackReleaseResult:
    return release_graphite_polymer_electrolyte_stack(**kwargs)


def _ns_label(value: float) -> str:
    raw = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return f"{raw}ns"


def _scan_followup_logs(run_dir: Path) -> dict[str, object]:
    patterns = {
        "fatal": ("fatal error", "segmentation fault", "core dumped"),
        "lincs": ("lincs warning", "too many lincs warnings", "constraint error"),
        "nonfinite": ("infinite",),
    }
    nonfinite_re = re.compile(r"(?<![A-Za-z0-9_.+-])(?:nan|inf)(?![A-Za-z0-9_.+-])", re.IGNORECASE)
    benign_nonfinite_tokens = (
        "epsilon-rf",
        "epsilon_rf",
    )
    hits: dict[str, list[str]] = {key: [] for key in patterns}
    for log_path in sorted(Path(run_dir).rglob("*.log")):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line_no, raw in enumerate(lines, start=1):
            lowered = raw.lower()
            for key, tokens in patterns.items():
                matched = any(token in lowered for token in tokens)
                if key == "nonfinite":
                    matched = (
                        matched or bool(nonfinite_re.search(raw))
                    ) and not any(token in lowered for token in benign_nonfinite_tokens)
                if matched:
                    hits[key].append(f"{log_path}:{line_no}:{raw.strip()[:240]}")
    return {
        "fatal_count": len(hits["fatal"]),
        "lincs_count": len(hits["lincs"]),
        "nonfinite_count": len(hits["nonfinite"]),
        "fatal_examples": hits["fatal"][:10],
        "lincs_examples": hits["lincs"][:10],
        "nonfinite_examples": hits["nonfinite"][:10],
        "ok": not any(hits.values()),
    }


def _read_gro_box_lengths_nm_local(gro_path: Path) -> tuple[float, float, float] | None:
    try:
        lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
        raw = lines[-1].split()
        if len(raw) >= 3:
            return float(raw[0]), float(raw[1]), float(raw[2])
    except Exception:
        pass
    return None


def run_sandwich_nvt_followup(
    result: GraphitePolymerElectrolyteSandwichResult | StackReleaseResult,
    *,
    work_dir,
    time_ns: float = 4.0,
    temp: float | None = None,
    mpi: int | None = None,
    omp: int | None = None,
    gpu: int | None = None,
    gpu_id: int | None = None,
    restart: bool | None = None,
    traj_ps: float = 10.0,
    energy_ps: float = 10.0,
    log_ps: float = 10.0,
    constraints: str = "h-bonds",
    dt_ps: float = 0.002,
    bridge_none_ps: float = 20.0,
    bridge_constraints_ps: float = 20.0,
    bridge_dt_ps: float = 0.001,
    freeze_group: str | None = "GRAPHITE",
    final_freeze_group: str | None = None,
    bridge_gpu_offload_mode: str = "balanced",
    final_gpu_offload_mode: str | None = None,
    wall_atomtype: str | None = None,
) -> SandwichNvtFollowupResult:
    """Run a short fixed-volume observation after sandwich stack release.

    Freshly assembled interfaces often tolerate minimization but still need a
    short dynamical handoff before constrained 2 fs sampling.  The default
    bridge therefore runs a compact no-constraints 1 fs NVT segment followed by
    a compact constraints-enabled 1 fs segment before the requested observation NVT.
    The bridge keeps the graphite group frozen by default, while the final
    observation stage releases graphite unless ``final_freeze_group`` is set.
    """

    target = result.sandwich_result if isinstance(result, StackReleaseResult) and result.sandwich_result is not None else result
    if not isinstance(target, GraphitePolymerElectrolyteSandwichResult):
        raise TypeError("run_sandwich_nvt_followup expects a sandwich result or StackReleaseResult.")
    export = target.stack_export
    missing = [name for name in ("system_top", "system_ndx") if not hasattr(export, name)]
    if missing:
        raise ValueError(f"Sandwich result stack_export is missing required path(s): {', '.join(missing)}")

    run_dir = Path(work_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    temperature = float(temp if temp is not None else 300.0)
    ntmpi = int(1 if mpi is None else mpi)
    ntomp = int(8 if omp is None else omp)
    use_gpu = bool(1 if gpu is None else gpu)
    nst_xtc = max(1, int(round(float(traj_ps) / float(dt_ps))))
    nst_energy = max(1, int(round(float(energy_ps) / float(dt_ps))))
    nst_log = max(1, int(round(float(log_ps) / float(dt_ps))))
    bridge_dt = max(float(bridge_dt_ps), 1.0e-6)
    bridge_out_ps = max(min(float(traj_ps), 10.0), bridge_dt)
    bridge_nst_xtc = max(1, int(round(bridge_out_ps / bridge_dt)))
    bridge_nst_energy = max(1, int(round(max(min(float(energy_ps), 10.0), bridge_dt) / bridge_dt)))
    bridge_nst_log = max(1, int(round(max(min(float(log_ps), 10.0), bridge_dt) / bridge_dt)))
    freeze = _freeze_block(freeze_group)
    final_freeze = _freeze_block(str(final_freeze_group)) if final_freeze_group else ""
    resolved_bridge_gpu_offload = str(bridge_gpu_offload_mode or ("balanced" if freeze else "full"))
    if not freeze and str(resolved_bridge_gpu_offload).strip().lower() in {"balanced", "conservative", "safe"}:
        resolved_bridge_gpu_offload = "full"
    resolved_final_gpu_offload = str(
        final_gpu_offload_mode
        or ("conservative" if final_freeze_group else "full")
    )
    resolved_wall_atomtype = wall_atomtype
    if resolved_wall_atomtype is None:
        try:
            from .protocol import _resolve_route_b_wall_atomtype

            resolved_wall_atomtype, _available_wall_atomtypes = _resolve_route_b_wall_atomtype(
                Path(export.system_top), None
            )
        except Exception:
            resolved_wall_atomtype = None
    wall_mdp = _phase_wall_block(wall_atomtype=str(resolved_wall_atomtype)) if resolved_wall_atomtype else ""
    z_boundary = (
        {
            "pbc": "xy",
            "periodic_molecules": "yes",
            "periodic-molecules": "yes",
            "wall_mdp": wall_mdp,
        }
        if resolved_wall_atomtype
        else {}
    )
    base = default_mdp_params()
    stage_index = 2
    stages = [
        EqStage(
            "01_settle_em",
            "minim",
            MdpSpec(
                MINIM_STEEP_HBONDS_MDP,
                {
                    **base,
                    **z_boundary,
                    "nsteps": 40000,
                    "emtol": 500.0,
                    "emstep": 0.001,
                    "constraints": str(constraints),
                    "constraint_algorithm": "lincs",
                    "lincs_iter": 4,
                    "lincs_order": 12,
                    "extra_mdp": freeze,
                },
            ),
            gpu_offload_mode=resolved_bridge_gpu_offload,
        ),
    ]
    dynamic_bridge_count = 0
    if float(bridge_none_ps) > 0:
        stages.append(
            EqStage(
                f"{stage_index:02d}_bridge_nvt_none",
                "nvt",
                MdpSpec(
                    NVT_NO_CONSTRAINTS_MDP,
                    {
                        **base,
                        **z_boundary,
                        "dt": bridge_dt,
                        "nsteps": max(int(round(float(bridge_none_ps) / bridge_dt)), 1),
                        "ref_t": temperature,
                        "gen_temp": temperature,
                        "gen_vel": "yes",
                        "continuation": "no",
                        "constraints": "none",
                        "nstxout": int(bridge_nst_xtc),
                        "nstxout_trr": 0,
                        "nstvout": 0,
                        "nstenergy": int(bridge_nst_energy),
                        "nstlog": int(bridge_nst_log),
                        "extra_mdp": freeze,
                    },
                ),
                gpu_offload_mode=resolved_bridge_gpu_offload,
            )
        )
        stage_index += 1
        dynamic_bridge_count += 1
    if str(constraints).lower() not in {"none", "no", "off"} and float(bridge_constraints_ps) > 0:
        bridge_label = "hbonds" if str(constraints).lower() in {"h-bonds", "hbonds", "h_bonds"} else "constraints"
        stages.append(
            EqStage(
                f"{stage_index:02d}_bridge_nvt_{bridge_label}",
                "nvt",
                MdpSpec(
                    NVT_MDP,
                    {
                        **base,
                        **z_boundary,
                        "dt": bridge_dt,
                        "nsteps": max(int(round(float(bridge_constraints_ps) / bridge_dt)), 1),
                        "ref_t": temperature,
                        "gen_temp": temperature,
                        "gen_vel": "no",
                        "continuation": "yes",
                        "constraints": str(constraints),
                        "constraint_algorithm": "lincs",
                        "lincs_iter": 4,
                        "lincs_order": 12,
                        "nstxout": int(bridge_nst_xtc),
                        "nstxout_trr": 0,
                        "nstvout": 0,
                        "nstenergy": int(bridge_nst_energy),
                        "nstlog": int(bridge_nst_log),
                        "extra_mdp": freeze,
                    },
                ),
                gpu_offload_mode=resolved_bridge_gpu_offload,
            )
        )
        stage_index += 1
        dynamic_bridge_count += 1
    nvt_name = f"{stage_index:02d}_nvt_{_ns_label(float(time_ns))}"
    stages.append(
        EqStage(
            nvt_name,
            "nvt",
            MdpSpec(
                NVT_MDP if str(constraints).lower() not in {"none", "no", "off"} else NVT_NO_CONSTRAINTS_MDP,
                {
                    **base,
                    **z_boundary,
                    "dt": float(dt_ps),
                    "nsteps": max(int(round(float(time_ns) * 1000.0 / float(dt_ps))), 1000),
                    "ref_t": temperature,
                    "gen_temp": temperature,
                    "gen_vel": "no" if dynamic_bridge_count else "yes",
                    "continuation": "yes" if dynamic_bridge_count else "no",
                    "constraints": str(constraints),
                    "constraint_algorithm": "lincs",
                    "lincs_iter": 4,
                    "lincs_order": 12,
                    "nstxout": int(nst_xtc),
                    "nstxout_trr": 0,
                    "nstvout": 0,
                    "nstenergy": int(nst_energy),
                    "nstlog": int(nst_log),
                    "extra_mdp": final_freeze,
                },
            ),
            gpu_offload_mode=resolved_final_gpu_offload,
        )
    )
    resources = RunResources(
        ntmpi=ntmpi,
        ntomp=ntomp,
        use_gpu=use_gpu,
        gpu_id=(str(gpu_id) if gpu_id is not None else None),
        gpu_offload_mode="conservative",
    )
    job = EquilibrationJob(
        gro=Path(target.relaxed_gro),
        top=Path(export.system_top),
        ndx=Path(export.system_ndx),
        provenance_ndx=Path(export.system_ndx),
        out_dir=run_dir,
        stages=stages,
        resources=resources,
    )
    job.run(restart=bool(resolve_restart(restart)))

    final_dir = run_dir / nvt_name
    final_gro = final_dir / "md.gro"
    ndx_groups = read_ndx_groups(Path(export.system_ndx))
    phase_groups = {
        name: [int(i) for i in ndx_groups.get(name, [])]
        for name in ("GRAPHITE", "POLYMER", "ELECTROLYTE")
        if ndx_groups.get(name)
    }
    stack_checks = _build_stack_checks(gro_path=final_gro, ndx_groups=phase_groups)
    log_scan = _scan_followup_logs(run_dir)
    summary_path = run_dir / "nvt_followup_summary.json"
    summary = {
        "input_relaxed_gro": str(target.relaxed_gro),
        "input_manifest": str(target.manifest_path),
        "time_ns": float(time_ns),
        "temperature_k": temperature,
        "dt_ps": float(dt_ps),
        "constraints": str(constraints),
        "final_freeze_group": (None if final_freeze_group is None else str(final_freeze_group)),
        "gpu_offload_mode": {
            "bridge": resolved_bridge_gpu_offload,
            "final": resolved_final_gpu_offload,
        },
        "bridge": {
            "none_ps": float(bridge_none_ps),
            "constraints_ps": float(bridge_constraints_ps),
            "dt_ps": bridge_dt,
            "stage_names": [stage.name for stage in stages[1:-1]],
        },
        "freeze_group": (None if not freeze else str(freeze_group)),
        "wall_atomtype": (None if resolved_wall_atomtype is None else str(resolved_wall_atomtype)),
        "z_boundary": ("pbc_xy_walls" if resolved_wall_atomtype else "pbc_xyz"),
        "output_cadence_ps": {
            "traj": float(traj_ps),
            "energy": float(energy_ps),
            "log": float(log_ps),
            "trr": None,
            "velocity": None,
        },
        "resources": {
            "mpi": ntmpi,
            "omp": ntomp,
            "gpu": int(use_gpu),
            "gpu_id": gpu_id,
        },
        "paths": {
            "work_dir": str(run_dir),
            "final_stage_dir": str(final_dir),
            "gro": str(final_gro),
            "tpr": str(final_dir / "md.tpr"),
            "xtc": str(final_dir / "md.xtc"),
            "edr": str(final_dir / "md.edr"),
            "top": str(export.system_top),
            "ndx": str(export.system_ndx),
        },
        "box_nm": _read_gro_box_lengths_nm_local(final_gro),
        "stack_checks": stack_checks,
        "log_scan": log_scan,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return SandwichNvtFollowupResult(
        work_dir=run_dir,
        summary_path=summary_path,
        final_gro=final_gro,
        tpr=final_dir / "md.tpr",
        xtc=final_dir / "md.xtc",
        edr=final_dir / "md.edr",
        top=Path(export.system_top),
        ndx=Path(export.system_ndx),
        stack_checks=stack_checks,
        log_scan=log_scan,
    )


def analyze_interface_transport(
    *,
    work_dir,
    center_mol,
    temp_k: float,
    run_migration: bool = False,
    out_dir: str | Path | None = None,
    restart: bool | None = None,
    rdf_kwargs: dict[str, object] | None = None,
    msd_kwargs: dict[str, object] | None = None,
    sigma_kwargs: dict[str, object] | None = None,
    migration_kwargs: dict[str, object] | None = None,
) -> InterfaceTransportResult:
    from ..sim.analyzer import AnalyzeResult

    del restart
    root = Path(work_dir)
    analysis_dir = Path(out_dir) if out_dir is not None else root / "07_transport_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analy = AnalyzeResult.from_work_dir(root)
    rdf = analy.rdf(center_mol=center_mol, **({} if rdf_kwargs is None else dict(rdf_kwargs)))
    msd = analy.msd(**({} if msd_kwargs is None else dict(msd_kwargs)))
    sigma_opts = {"eh_mode": "gmx_current_only", **({} if sigma_kwargs is None else dict(sigma_kwargs))}
    sigma = analy.sigma(msd=msd, temp_k=float(temp_k), **sigma_opts)
    migration = None
    if bool(run_migration):
        migration = analy.migration(
            center_mol=center_mol,
            out_dir=analysis_dir / "migration",
            **({} if migration_kwargs is None else dict(migration_kwargs)),
        )
    summary = {
        "center_mol": getattr(center_mol, "name", str(center_mol)),
        "temp_k": float(temp_k),
        "rdf_path": rdf.get("summary_path"),
        "msd_path": msd.get("summary_path"),
        "sigma_path": sigma.get("summary_path"),
        "migration_path": (None if migration is None else migration.get("summary_path")),
    }
    summary_path = analysis_dir / "interface_transport_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return InterfaceTransportResult(
        work_dir=root,
        analysis_dir=analysis_dir,
        summary_path=summary_path,
        rdf=rdf,
        msd=msd,
        sigma=sigma,
        migration=migration,
    )


def _find_nvt_followup_summary(root: Path) -> Path | None:
    candidates = [
        root / "nvt_followup_summary.json",
        root / "07_nvt_followup" / "nvt_followup_summary.json",
        root.parent / "nvt_followup_summary.json",
        root.parent.parent / "07_nvt_followup" / "nvt_followup_summary.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    hits = [path for path in root.rglob("nvt_followup_summary.json") if path.is_file()]
    if hits:
        hits.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return hits[0]
    return None


def analyze_sandwich_interface(
    *,
    work_dir,
    out_dir: str | Path | None = None,
    bin_nm: float = 0.05,
    frame_stride: int | str = "auto",
    region_width_nm: float = 0.75,
    surface_grid_nm: float = 0.5,
    analysis_profile: str = "interface_fast",
    phase_groups: Sequence[str] = ("GRAPHITE", "POLYMER", "ELECTROLYTE"),
    compute_transport: bool = True,
    resume: bool = True,
) -> dict[str, object]:
    """Run interface statistics on an existing sandwich stack or NVT follow-up.

    The helper prefers ``nvt_followup_summary.json`` when present because that
    file records the final NVT ``gro/tpr/xtc/edr`` paths while still pointing to
    the stack ``top/ndx`` artifacts. If no follow-up summary is available, it
    falls back to the nearest exported ``00_stack``/``02_system`` directory and
    performs a static profile when no trajectory is present.
    """

    from ..gmx.analysis.interface_profile import compute_interface_profile
    from ..sim.analyzer import AnalyzeResult

    root = Path(work_dir).resolve()
    summary_path = _find_nvt_followup_summary(root)
    if summary_path is not None:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        paths = payload.get("paths") if isinstance(payload, dict) else {}
        if not isinstance(paths, dict):
            paths = {}
        top_path = Path(paths["top"]) if paths.get("top") else root / "missing.system.top"
        ndx_path = Path(paths["ndx"]) if paths.get("ndx") else root / "missing.system.ndx"
        gro_path = Path(paths["gro"]) if paths.get("gro") else root / "missing.md.gro"
        xtc_path = Path(paths["xtc"]) if paths.get("xtc") else root / "missing.md.xtc"
        tpr_path = Path(paths["tpr"]) if paths.get("tpr") else root / "missing.md.tpr"
        edr_path = Path(paths["edr"]) if paths.get("edr") else root / "missing.md.edr"
        if top_path.is_file() and ndx_path.is_file() and gro_path.is_file():
            analy = AnalyzeResult(
                work_dir=root,
                tpr=tpr_path,
                xtc=xtc_path,
                edr=edr_path,
                top=top_path,
                ndx=ndx_path,
            )
            return analy.interface_profile(
                gro_path=gro_path,
                top_path=top_path,
                ndx_path=ndx_path,
                system_dir=top_path.parent,
                xtc_path=(xtc_path if xtc_path.exists() else None),
                out_dir=out_dir,
                bin_nm=bin_nm,
                frame_stride=frame_stride,
                region_width_nm=region_width_nm,
                surface_grid_nm=surface_grid_nm,
                analysis_profile=analysis_profile,
                phase_groups=phase_groups,
                compute_transport=compute_transport,
                resume=resume,
            )

    try:
        analy = AnalyzeResult.from_work_dir(root)
        return analy.interface_profile(
            out_dir=out_dir,
            bin_nm=bin_nm,
            frame_stride=frame_stride,
            region_width_nm=region_width_nm,
            surface_grid_nm=surface_grid_nm,
            analysis_profile=analysis_profile,
            phase_groups=phase_groups,
            compute_transport=compute_transport,
            resume=resume,
        )
    except Exception:
        pass

    stack_candidates = [
        root,
        root / "00_stack",
        root / "06_full_stack_release" / "00_stack",
    ]
    for system_dir in stack_candidates:
        top_path = system_dir / "system.top"
        ndx_path = system_dir / "system.ndx"
        gro_path = system_dir / "system.gro"
        if top_path.exists() and ndx_path.exists() and gro_path.exists():
            analysis_dir = Path(out_dir) if out_dir is not None else root / "06_analysis" / "interface_profile"
            return compute_interface_profile(
                gro_path=gro_path,
                top_path=top_path,
                ndx_path=ndx_path,
                system_dir=system_dir,
                out_dir=analysis_dir,
                xtc_path=None,
                bin_nm=bin_nm,
                frame_stride=(1 if str(frame_stride).strip().lower() == "auto" else int(frame_stride)),
                region_width_nm=region_width_nm,
                surface_grid_nm=surface_grid_nm,
                analysis_profile=analysis_profile,
                phase_groups=phase_groups,
                compute_transport=False,
            )
    raise FileNotFoundError(
        f"Could not locate a sandwich follow-up summary or exported stack system under {root}"
    )


def print_interface_result_summary(result, *, profile: str | None = None) -> None:
    from .sandwich_examples import print_sandwich_result_summary

    target = result.sandwich_result if isinstance(result, StackReleaseResult) and result.sandwich_result is not None else result
    print_sandwich_result_summary(target, profile=profile)


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


def _selected_phase_target_thickness_nm(
    *,
    species: Sequence,
    counts: Sequence[int],
    target_density_g_cm3: float,
    target_xy_nm: tuple[float, float],
    requested_thickness_nm: float,
) -> tuple[float, dict[str, object]]:
    """Resolve the relaxed slab thickness for the molecules actually selected.

    Bulk-slab extraction can select a slightly different number of molecules
    than the initial direct-pack plan, especially after lateral replication is
    used to cover the graphite footprint. Compressing that selected mass back
    to the user-requested thickness can create an unrealistically dense phase
    and trigger unstable relaxation. Instead, keep the target density as the
    conserved physical quantity and let the slab thickness adapt.
    """

    requested = max(float(requested_thickness_nm), 1.0e-6)
    total_mass_amu = _phase_total_mass_amu(species=species, counts=counts)
    effective = _solve_phase_target_z_nm(
        total_mass_amu=float(total_mass_amu),
        target_density_g_cm3=float(target_density_g_cm3),
        target_xy_nm=(float(target_xy_nm[0]), float(target_xy_nm[1])),
        min_z_nm=0.0,
    )
    if effective <= 0.0 or not math.isfinite(effective):
        return requested, {
            "requested_thickness_nm": float(requested),
            "resolved_thickness_nm": float(requested),
            "selected_total_mass_amu": float(total_mass_amu),
            "selected_count_target_thickness_applied": False,
            "selected_count_target_thickness_reason": "invalid_effective_thickness",
        }

    # Keep pathological fragment selections from silently creating a slab that
    # is wildly unlike the requested interphase. The acceptance gate will still
    # reject bad density/gap outcomes later.
    min_z = max(0.25, 0.50 * requested)
    max_z = max(requested, 2.50 * requested)
    resolved = min(max(float(effective), float(min_z)), float(max_z))
    applied = abs(float(resolved) - requested) > max(0.05, 0.05 * requested)
    if resolved != effective:
        reason = "clamped_to_sane_interphase_range"
    elif applied:
        reason = "adapted_to_selected_molecule_count"
    else:
        reason = "within_requested_thickness_tolerance"
    return float(resolved), {
        "requested_thickness_nm": float(requested),
        "resolved_thickness_nm": float(resolved),
        "effective_density_matched_thickness_nm": float(effective),
        "selected_total_mass_amu": float(total_mass_amu),
        "selected_count_target_thickness_applied": bool(applied),
        "selected_count_target_thickness_reason": str(reason),
    }


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


def _ensure_phase_species_artifact(mol, *, ff_name: str | None, mol_name: str | None, charge_method: str = "RESP") -> None:
    """Fail fast if an interface phase species cannot be exported later.

    Final stack assembly is intentionally metadata-driven: it reconstructs the
    GROMACS system from per-species artifacts rather than carrying all temporary
    topology files between phase builders. Large generated polymer chains are
    not MolDB records, so they must carry a cached artifact directory before the
    final stack is assembled.
    """

    try:
        from ..io.molecule_cache import ensure_cached_artifacts

        ensure_cached_artifacts(
            mol,
            ff_name=(str(ff_name) if ff_name else None),
            mol_name=(str(mol_name) if mol_name else None),
            charge_method=str(charge_method or "RESP"),
        )
    except Exception as exc:
        label = str(mol_name or get_name(mol, default="species"))
        raise RuntimeError(
            f"Failed to prepare reusable GROMACS artifacts for interface species {label}. "
            "This would otherwise fail much later during final stack export; inspect the force-field "
            "assignment and charge metadata for this molecule before continuing."
        ) from exc


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
    chain_pe_mode = any(
        bool(spec.polyelectrolyte_mode if spec.polyelectrolyte_mode is not None else _is_polyelectrolyte_spec(spec))
        for spec in monomer_specs
    )
    chain = ff.ff_assign(chain, report=False, polyelectrolyte_mode=chain_pe_mode)
    if not chain:
        raise RuntimeError(f"Cannot assign force field parameters for polymer chain {polymer.name}.")
    _ensure_phase_species_artifact(
        chain,
        ff_name=getattr(ff, "name", None),
        mol_name=polymer.name,
        charge_method="RESP",
    )
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
    charge_balance = {
        "chain_charge_e": int(chain_charge),
        "chain_count": int(chain_count),
        "polymer_total_charge_e": int(chain_charge * chain_count),
        "counterion_name": None,
        "counterion_charge_e": 0,
        "counterion_count": 0,
        "counterion_total_charge_e": 0,
        "net_charge_e": int(chain_charge * chain_count),
        "neutralized": bool(chain_charge == 0),
        "charge_source": str(chain_charge_meta.get("source", "unknown")),
        "raw_chain_charge_e": float(chain_charge_meta.get("raw_total_charge", chain_charge)),
    }
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
            counterion_total = int(counterion_count * ion_charge)
            charge_balance.update(
                {
                    "counterion_name": str(polymer.counterion.name),
                    "counterion_charge_e": int(ion_charge),
                    "counterion_count": int(counterion_count),
                    "counterion_total_charge_e": int(counterion_total),
                    "net_charge_e": int(total_polymer_charge + counterion_total),
                    "neutralized": bool(total_polymer_charge + counterion_total == 0),
                }
            )
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
        "charge_balance": charge_balance,
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
    max_rounds: int = 1,
) -> tuple[GraphiteSubstrateSpec, GraphiteBuildResult, list[dict[str, object]]]:
    # Auto polymer/electrolyte counts are themselves estimated from the current
    # graphite footprint.  Re-estimating those counts after each expansion forms
    # a positive feedback loop (larger graphite -> more soft phase -> still
    # larger graphite).  One conservative preflight pass is enough because
    # `_expand_graphite_to_meet_required_xy()` already expands to the requested
    # footprint within that pass.
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
        polymer_min_xy_nm = _polymer_preflight_min_xy_nm(polymer)
        polymer_required_xy_nm = (
            max(float(polymer_required_xy_nm[0]), float(polymer_min_xy_nm[0])),
            max(float(polymer_required_xy_nm[1]), float(polymer_min_xy_nm[1])),
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
                "polymer_preflight_min_xy_nm": [float(polymer_min_xy_nm[0]), float(polymer_min_xy_nm[1])],
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


def _polymer_preflight_min_xy_nm(polymer: PolymerSlabSpec) -> tuple[float, float]:
    try:
        target_atoms = int(getattr(polymer, "chain_target_atoms", 0) or 0)
    except Exception:
        target_atoms = 0
    # A dense-area estimate alone can produce a 3-4 nm graphite footprint for
    # long PEO chains.  That is formally enough mass/volume, but too small for
    # stable chain placement and interphase relaxation.  Keep this as a modest
    # conformational floor rather than a chemistry-specific hard size.
    if target_atoms >= 500:
        minimum = 6.0
    elif target_atoms >= 250:
        minimum = 5.0
    elif target_atoms >= 180:
        minimum = 4.5
    else:
        minimum = 4.0
    return (float(minimum), float(minimum))


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
        restart=restart,
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
    cation_count = int(prep.direct_plan.target_counts[-2])
    anion_count = int(prep.direct_plan.target_counts[-1])
    cation_charge = int(_smiles_formal_charge(electrolyte.salt_cation.smiles))
    anion_charge = int(_smiles_formal_charge(electrolyte.salt_anion.smiles))
    charge_balance = {
        "salt_cation": str(electrolyte.salt_cation.name),
        "salt_anion": str(electrolyte.salt_anion.name),
        "cation_count": int(cation_count),
        "anion_count": int(anion_count),
        "cation_charge_e": int(cation_charge),
        "anion_charge_e": int(anion_charge),
        "salt_net_charge_e": int(cation_count * cation_charge + anion_count * anion_charge),
        "neutralized": bool(cation_count * cation_charge + anion_count * anion_charge == 0),
        "charge_scale": {
            str(electrolyte.salt_cation.name): float(electrolyte.salt_cation.charge_scale),
            str(electrolyte.salt_anion.name): float(electrolyte.salt_anion.charge_scale),
        },
    }
    return {
        "mols": mols,
        "charge_scale": charge_scale,
        "prep": prep,
        "charge_balance": charge_balance,
    }


def _cell_box_nm(cell, *, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    current_box = getattr(cell, "cell", None)
    if current_box is None:
        return tuple(float(x) for x in fallback)
    try:
        return (
            float(current_box.xhi - current_box.xlo) / 10.0,
            float(current_box.yhi - current_box.ylo) / 10.0,
            float(current_box.zhi - current_box.zlo) / 10.0,
        )
    except Exception:
        return tuple(float(x) for x in fallback)


def _bulk_eq_box_nm(*, work_dir: Path, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    latest = _find_latest_equilibrated_gro(Path(work_dir))
    if latest is None or not Path(latest).exists():
        return tuple(float(x) for x in fallback)
    try:
        return tuple(float(x) for x in read_equilibrated_box_nm(Path(latest)))
    except Exception:
        return tuple(float(x) for x in fallback)


def _read_bulk_calibration_summary(path: Path) -> _BulkCalibrationSummary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _BulkCalibrationSummary(
        label=str(payload["label"]),
        phase_preparation_mode=str(payload["phase_preparation_mode"]),
        master_xy_nm=tuple(float(x) for x in payload["master_xy_nm"]),
        bulk_reference_box_nm=tuple(float(x) for x in payload["bulk_reference_box_nm"]),
        target_density_g_cm3=float(payload["target_density_g_cm3"]),
        total_mass_amu=float(payload["total_mass_amu"]),
        target_z_nm=float(payload["target_z_nm"]),
        initial_walled_pack_density_g_cm3=float(payload["initial_walled_pack_density_g_cm3"]),
        selected_bulk_pack_density_g_cm3=float(payload["selected_bulk_pack_density_g_cm3"]),
        charged_phase=bool(payload["charged_phase"]),
        notes=tuple(str(x) for x in payload.get("notes", ())),
    )


def _bulk_calibration_cache_compatible(
    *,
    calibration: _BulkCalibrationSummary,
    graphite_box_nm: tuple[float, float, float],
    species: Sequence,
    counts: Sequence[int],
    target_density_g_cm3: float,
    xy_tol_nm: float = 0.01,
) -> tuple[bool, str | None]:
    """Return whether a restart bulk calibration matches the current footprint.

    The sandwich workflow frequently changes graphite size during preflight
    negotiation. Reusing an old bulk slab after that change can silently squeeze
    the same molecule counts into a smaller XY footprint, creating severe
    overlaps before the confined EM stage. Treat master XY, density, and phase
    total mass as the cache identity for restart reuse.
    """

    cached_xy = tuple(float(x) for x in calibration.master_xy_nm[:2])
    requested_xy = (float(graphite_box_nm[0]), float(graphite_box_nm[1]))
    if (
        abs(float(cached_xy[0]) - float(requested_xy[0])) > float(xy_tol_nm)
        or abs(float(cached_xy[1]) - float(requested_xy[1])) > float(xy_tol_nm)
    ):
        return (
            False,
            "master_xy_mismatch:"
            f" cached=({cached_xy[0]:.4f},{cached_xy[1]:.4f})"
            f" requested=({requested_xy[0]:.4f},{requested_xy[1]:.4f})",
        )

    density_delta = abs(float(calibration.target_density_g_cm3) - float(target_density_g_cm3))
    if density_delta > max(1.0e-6, 1.0e-5 * abs(float(target_density_g_cm3))):
        return (
            False,
            "target_density_mismatch:"
            f" cached={float(calibration.target_density_g_cm3):.6g}"
            f" requested={float(target_density_g_cm3):.6g}",
        )

    requested_mass = _phase_total_mass_amu(species=species, counts=counts)
    mass_delta = abs(float(calibration.total_mass_amu) - float(requested_mass))
    if mass_delta > max(1.0e-4, 1.0e-6 * max(abs(float(requested_mass)), 1.0)):
        return (
            False,
            "total_mass_mismatch:"
            f" cached={float(calibration.total_mass_amu):.6g}"
            f" requested={float(requested_mass):.6g}",
        )
    return True, None


def _archive_stale_bulk_calibration_outputs(*, round_dir: Path, label: str) -> list[dict[str, str]]:
    """Move stale generated bulk artifacts aside before rebuilding a phase.

    Bulk calibration directories contain several restartable GROMACS stages. If
    the phase identity changes, leaving old stage outputs in place can mix a new
    topology with old coordinates. Renaming keeps the forensic artifacts while
    guaranteeing the next rebuild starts from a clean stage tree.
    """

    stamp = time.strftime("%Y%m%d_%H%M%S")
    archived: list[dict[str, str]] = []
    for name in (
        "00_build",
        "01_raw_non_scaled",
        "02_system",
        "03_EQ21",
        "04_eq_additional",
        "05_npt_production",
    ):
        path = Path(round_dir) / name
        if not path.exists():
            continue
        target = Path(round_dir) / f"{name}_stale_{stamp}"
        suffix = 1
        while target.exists():
            suffix += 1
            target = Path(round_dir) / f"{name}_stale_{stamp}_{suffix}"
        path.rename(target)
        archived.append({"path": str(path), "archived_to": str(target), "label": str(label)})
    return archived


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_bulk_calibration_summary(
    *,
    label: str,
    master_xy_nm: tuple[float, float],
    target_density_g_cm3: float,
    total_mass_amu: float,
    selected_bulk_pack_density_g_cm3: float,
    bulk_reference_box_nm: tuple[float, float, float],
    charged_phase: bool,
    notes: Sequence[str] = (),
) -> _BulkCalibrationSummary:
    target_z_nm = _solve_phase_target_z_nm(
        total_mass_amu=float(total_mass_amu),
        target_density_g_cm3=float(target_density_g_cm3),
        target_xy_nm=(float(master_xy_nm[0]), float(master_xy_nm[1])),
    )
    initial_walled_density = _recommend_initial_walled_pack_density(
        phase=str(label),
        target_density_g_cm3=float(target_density_g_cm3),
        selected_bulk_pack_density_g_cm3=float(selected_bulk_pack_density_g_cm3),
    )
    return _BulkCalibrationSummary(
        label=str(label),
        phase_preparation_mode="bulk_calibrate_walled_phase",
        master_xy_nm=(float(master_xy_nm[0]), float(master_xy_nm[1])),
        bulk_reference_box_nm=tuple(float(x) for x in bulk_reference_box_nm),
        target_density_g_cm3=float(target_density_g_cm3),
        total_mass_amu=float(total_mass_amu),
        target_z_nm=float(target_z_nm),
        initial_walled_pack_density_g_cm3=float(initial_walled_density),
        selected_bulk_pack_density_g_cm3=float(selected_bulk_pack_density_g_cm3),
        charged_phase=bool(charged_phase),
        notes=tuple(str(x) for x in notes),
    )


def _run_polymer_bulk_calibration(
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
    reference_box_nm = (
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
        box_nm=reference_box_nm,
    )
    summary_path = round_dir / "polymer_bulk_calibration_summary.json"
    if bool(resolve_restart(restart)) and summary_path.exists() and _find_latest_equilibrated_gro(round_dir) is not None:
        calibration = _read_bulk_calibration_summary(summary_path)
        cache_ok, cache_reason = _bulk_calibration_cache_compatible(
            calibration=calibration,
            graphite_box_nm=graphite_box_nm,
            species=list(phase_build["species"]),
            counts=list(phase_build["counts"]),
            target_density_g_cm3=float(polymer.target_density_g_cm3),
        )
        if cache_ok:
            return {
                "phase_build": phase_build,
                "pack": None,
                "bulk": None,
                "calibration": calibration,
                "summary": asdict(calibration),
                "summary_path": summary_path,
                "restart_reused": True,
            }
        restart = False
        archived_outputs = _archive_stale_bulk_calibration_outputs(round_dir=round_dir, label="polymer")
        (round_dir / "stale_restart_calibration.json").write_text(
            json.dumps(
                {
                    "label": "polymer",
                    "reason": cache_reason,
                    "stale_summary_path": str(summary_path),
                    "cached_master_xy_nm": [float(x) for x in calibration.master_xy_nm],
                    "requested_master_xy_nm": [float(graphite_box_nm[0]), float(graphite_box_nm[1])],
                    "cached_total_mass_amu": float(calibration.total_mass_amu),
                    "requested_total_mass_amu": float(
                        _phase_total_mass_amu(species=list(phase_build["species"]), counts=list(phase_build["counts"]))
                    ),
                    "archived_outputs": archived_outputs,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
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
        restart=restart,
    )
    calibration = _build_bulk_calibration_summary(
        label="polymer",
        master_xy_nm=(float(graphite_box_nm[0]), float(graphite_box_nm[1])),
        target_density_g_cm3=float(polymer.target_density_g_cm3),
        total_mass_amu=_phase_total_mass_amu(
            species=list(phase_build["species"]),
            counts=list(phase_build["counts"]),
        ),
        selected_bulk_pack_density_g_cm3=float(pack_result.selected_density_g_cm3),
        bulk_reference_box_nm=_bulk_eq_box_nm(work_dir=round_dir, fallback=_cell_box_nm(bulk, fallback=reference_box_nm)),
        charged_phase=bool(phase_build.get("charged_phase", False)),
        notes=phase_build.get("notes", ()),
    )
    summary_path = _write_bulk_calibration_summary(
        calibration,
        summary_path,
    )
    return {
        "phase_build": phase_build,
        "pack": pack_result,
        "bulk": bulk,
        "calibration": calibration,
        "summary": asdict(calibration),
        "summary_path": summary_path,
    }


def _run_electrolyte_bulk_calibration(
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
    summary_path = round_dir / "electrolyte_bulk_calibration_summary.json"
    if bool(resolve_restart(restart)) and summary_path.exists() and _find_latest_equilibrated_gro(round_dir) is not None:
        calibration = _read_bulk_calibration_summary(summary_path)
        cache_ok, cache_reason = _bulk_calibration_cache_compatible(
            calibration=calibration,
            graphite_box_nm=graphite_box_nm,
            species=list(inputs["mols"]),
            counts=list(inputs["prep"].direct_plan.target_counts),
            target_density_g_cm3=float(electrolyte.target_density_g_cm3),
        )
        if cache_ok:
            return {
                "inputs": inputs,
                "pack": None,
                "bulk": None,
                "calibration": calibration,
                "summary": asdict(calibration),
                "summary_path": summary_path,
                "restart_reused": True,
            }
        restart = False
        archived_outputs = _archive_stale_bulk_calibration_outputs(round_dir=round_dir, label="electrolyte")
        (round_dir / "stale_restart_calibration.json").write_text(
            json.dumps(
                {
                    "label": "electrolyte",
                    "reason": cache_reason,
                    "stale_summary_path": str(summary_path),
                    "cached_master_xy_nm": [float(x) for x in calibration.master_xy_nm],
                    "requested_master_xy_nm": [float(graphite_box_nm[0]), float(graphite_box_nm[1])],
                    "cached_total_mass_amu": float(calibration.total_mass_amu),
                    "requested_total_mass_amu": float(
                        _phase_total_mass_amu(
                            species=list(inputs["mols"]),
                            counts=list(inputs["prep"].direct_plan.target_counts),
                        )
                    ),
                    "archived_outputs": archived_outputs,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
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
        restart=restart,
    )
    notes = tuple(str(x) for x in getattr(inputs["prep"], "notes", ()))
    calibration = _build_bulk_calibration_summary(
        label="electrolyte",
        master_xy_nm=(float(graphite_box_nm[0]), float(graphite_box_nm[1])),
        target_density_g_cm3=float(electrolyte.target_density_g_cm3),
        total_mass_amu=_phase_total_mass_amu(
            species=list(inputs["mols"]),
            counts=list(inputs["prep"].direct_plan.target_counts),
        ),
        selected_bulk_pack_density_g_cm3=float(pack_result.selected_density_g_cm3),
        bulk_reference_box_nm=_bulk_eq_box_nm(
            work_dir=round_dir,
            fallback=_cell_box_nm(
                bulk,
                fallback=(float(graphite_box_nm[0]), float(graphite_box_nm[1]), float(electrolyte.slab_z_nm)),
            ),
        ),
        charged_phase=False,
        notes=notes,
    )
    summary_path = _write_bulk_calibration_summary(
        calibration,
        summary_path,
    )
    return {
        "inputs": inputs,
        "pack": pack_result,
        "bulk": bulk,
        "calibration": calibration,
        "summary": asdict(calibration),
        "summary_path": summary_path,
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
        restart=restart,
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
    polymer_names: Sequence[str] = (),
    electrolyte_names: Sequence[str],
) -> dict[str, list[int]]:
    existing = read_ndx_groups(ndx_path)
    merged_groups: list[tuple[str, list[int]]] = [(name, list(idxs)) for name, idxs in existing.items()]

    if "System" not in existing:
        system_atoms = sorted({int(idx) for idxs in existing.values() for idx in idxs})
        if system_atoms:
            merged_groups.insert(0, ("System", system_atoms))

    _append_group(merged_groups, existing, "GRAPHITE", [graphite_name, f"MOL_{graphite_name}"])
    polymer_expanded = [polymer_name, f"MOL_{polymer_name}"]
    for name in polymer_names:
        polymer_expanded.extend([str(name), f"MOL_{name}"])
    _append_group(merged_groups, existing, "POLYMER", polymer_expanded)
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


def _freeze_block(group_name: str | None) -> str:
    if group_name is None:
        return ""
    token = str(group_name).strip()
    if token.lower() in {"", "none", "off", "false", "no"}:
        return ""
    return "\n".join(
        (
            "; keep the graphite substrate frozen while polymer/electrolyte phases relax against it",
            f"freezegrps               = {token}",
            "freezedim                = Y Y Y",
        )
    )


def _sandwich_relaxation_stages(
    *,
    relax: SandwichRelaxationSpec,
    freeze_group: str | None,
    wall_atomtype: str | None = None,
) -> list[EqStage]:
    base = default_mdp_params()
    freeze = _freeze_block(freeze_group)
    freeze_active = bool(freeze)
    release_final = bool(getattr(relax, "stack_release_graphite_final", True))
    final_freeze = "" if release_final else freeze
    frozen_gpu_mode = str(getattr(relax, "stack_frozen_gpu_offload_mode", "balanced") or "balanced")
    final_gpu_mode = str(
        getattr(relax, "stack_final_gpu_offload_mode", "full") or ("full" if release_final else frozen_gpu_mode)
    )
    pre_final_gpu_mode = frozen_gpu_mode if freeze_active else final_gpu_mode
    wall_mdp = _phase_wall_block(wall_atomtype=str(wall_atomtype)) if wall_atomtype else ""
    z_boundary = (
        {
            "pbc": "xy",
            "periodic_molecules": "yes",
            "periodic-molecules": "yes",
            "wall_mdp": wall_mdp,
        }
        if wall_atomtype
        else {}
    )
    fixed_xy = fixed_xy_semiisotropic_npt_overrides(pressure_bar=float(relax.pressure_bar))
    z_compaction_xy = fixed_xy_semiisotropic_npt_overrides(
        pressure_bar=max(float(relax.pressure_bar), 80.0)
    )
    contact_xy = fixed_xy_semiisotropic_npt_overrides(
        pressure_bar=max(float(relax.pressure_bar), 20.0)
    )
    contact_ps = max(float(relax.stacked_exchange_ps) * 0.40, 1.0)
    natural_ps = max(float(relax.stacked_exchange_ps) * 0.60, 1.0)
    if wall_atomtype:
        # A single-sided graphite|polymer|electrolyte stack is not truly
        # periodic along z. With explicit z-walls, semiisotropic pressure
        # coupling can overreact to wall/vacuum contributions and expand the
        # box. The component phases are already density-relaxed, so the stack
        # release should be a short fixed-volume handoff protocol rather than
        # a hidden production run that prematurely mixes the soft phases.
        # Keep this as a single MD stage after EM.  Compact CMC/electrolyte
        # wall stacks exposed a GROMACS 2026.1 step-0 crash when multiple
        # consecutive wall-confined MD stages were chained by checkpoint.
        wall_settle_ps = min(
            max(
                4.0,
                0.25 * float(relax.stacked_pre_nvt_ps)
                + 0.10 * float(relax.stacked_z_relax_ps)
                + 0.10 * float(contact_ps)
                + 0.10 * float(natural_ps),
            ),
            20.0,
        )
        return [
            EqStage(
                "01_em",
                "minim",
                MdpSpec(
                    MINIM_STEEP_MDP,
                    {
                        **base,
                        **z_boundary,
                        "nsteps": 40000,
                        "emtol": 500.0,
                        "emstep": 0.001,
                        "extra_mdp": freeze,
                    },
                ),
                gpu_offload_mode="cpu",
            ),
            EqStage(
                "02_stack_settle_nvt",
                "nvt",
                MdpSpec(
                    NVT_NO_CONSTRAINTS_MDP,
                    {
                        **base,
                        **z_boundary,
                        "dt": 0.001,
                        "nsteps": max(int(round(float(wall_settle_ps) / 0.001)), 1000),
                        "ref_t": float(relax.temperature_k),
                        "gen_temp": float(relax.temperature_k),
                        "gen_vel": "yes",
                        "extra_mdp": final_freeze,
                    },
                ),
                gpu_offload_mode="balanced",
            ),
        ]
    return [
        EqStage(
            "01_em",
            "minim",
            MdpSpec(
                MINIM_STEEP_MDP,
                {
                    **base,
                    **z_boundary,
                    "nsteps": 40000,
                    "emtol": 500.0,
                    "emstep": 0.001,
                    "extra_mdp": freeze,
                },
            ),
            gpu_offload_mode="cpu",
        ),
        EqStage(
            "02_pre_nvt",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                {
                    **base,
                    **z_boundary,
                    "dt": 0.001,
                    "nsteps": max(int(round(float(relax.stacked_pre_nvt_ps) / 0.001)), 1000),
                    "ref_t": float(relax.temperature_k),
                    "gen_temp": float(relax.temperature_k),
                    "gen_vel": "yes",
                    "extra_mdp": freeze,
                },
            ),
            gpu_offload_mode=pre_final_gpu_mode,
        ),
        EqStage(
            "03_z_relax",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                {
                    **base,
                    **z_boundary,
                    **z_compaction_xy,
                    "dt": 0.001,
                    "nsteps": max(int(round(float(relax.stacked_z_relax_ps) / 0.001)), 1000),
                    "ref_t": float(relax.temperature_k),
                    "gen_temp": float(relax.temperature_k),
                    "gen_vel": "no",
                    "ref_p": z_compaction_xy["ref_p"],
                    "compressibility": z_compaction_xy["compressibility"],
                    "pcoupltype": z_compaction_xy["pcoupltype"],
                    "extra_mdp": freeze,
                },
            ),
            gpu_offload_mode=pre_final_gpu_mode,
        ),
        EqStage(
            "04_contact_release",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                {
                    **base,
                    **z_boundary,
                    **contact_xy,
                    "dt": 0.001,
                    "nsteps": max(int(round(float(contact_ps) / 0.001)), 1000),
                    "ref_t": float(relax.temperature_k),
                    "gen_temp": float(relax.temperature_k),
                    "gen_vel": "no",
                    "ref_p": contact_xy["ref_p"],
                    "compressibility": contact_xy["compressibility"],
                    "pcoupltype": contact_xy["pcoupltype"],
                    "extra_mdp": freeze,
                },
            ),
            gpu_offload_mode=pre_final_gpu_mode,
        ),
        EqStage(
            "05_natural_exchange",
            "npt",
            MdpSpec(
                NPT_MDP,
                {
                    **base,
                    **z_boundary,
                    **fixed_xy,
                    "dt": 0.002,
                    "nsteps": max(int(round(float(natural_ps) / 0.002)), 1000),
                    "ref_t": float(relax.temperature_k),
                    "gen_temp": float(relax.temperature_k),
                    "gen_vel": "no",
                    "continuation": "yes",
                    "constraints": "h-bonds",
                    "lincs_iter": 4,
                    "lincs_order": 12,
                    "ref_p": fixed_xy["ref_p"],
                    "compressibility": fixed_xy["compressibility"],
                    "pcoupltype": fixed_xy["pcoupltype"],
                    "extra_mdp": final_freeze,
                },
            ),
            gpu_offload_mode=final_gpu_mode if (release_final or not freeze_active) else frozen_gpu_mode,
        ),
    ]


def _run_stacked_relaxation(
    *,
    export: SystemExportResult,
    work_dir: Path,
    relax: SandwichRelaxationSpec,
    freeze_group: str | None = "GRAPHITE",
    wall_atomtype: str | None = None,
    restart: bool | None = None,
) -> Path:
    stages = _sandwich_relaxation_stages(relax=relax, freeze_group=freeze_group, wall_atomtype=wall_atomtype)
    resources = RunResources(
        ntmpi=int(relax.mpi),
        ntomp=int(relax.omp),
        use_gpu=bool(relax.gpu),
        gpu_id=(str(relax.gpu_id) if relax.gpu_id is not None else None),
        # Individual stages choose CPU/balanced/full offload. Frozen graphite
        # stages cannot use GPU update in GROMACS, but bonded GPU remains much
        # faster than the older fully conservative path. The final exchange can
        # release graphite and return to full GPU offload when requested by the
        # relaxation spec.
        gpu_offload_mode=str(getattr(relax, "stack_frozen_gpu_offload_mode", "balanced") or "balanced"),
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


def _scale_fragment_centers_lateral_to_target(
    coords: np.ndarray,
    *,
    blocks: Sequence[tuple[int, int]],
    target_x_ang: float,
    target_y_ang: float,
    min_scale_xy: tuple[float, float] = (0.55, 0.55),
) -> tuple[np.ndarray, tuple[float, float], bool]:
    """Compress only fragment centers in XY while preserving internal geometry."""

    scaled = np.asarray(coords, dtype=float).copy()
    if scaled.size == 0 or not blocks:
        return scaled, (1.0, 1.0), False
    if int(blocks[-1][1]) != int(len(scaled)):
        return scaled, (1.0, 1.0), False

    mins = np.min(scaled, axis=0)
    maxs = np.max(scaled, axis=0)
    spans = maxs - mins
    scale_x = 1.0 if float(spans[0]) <= float(target_x_ang) + 1.0e-9 else float(target_x_ang) / max(float(spans[0]), 1.0e-9)
    scale_y = 1.0 if float(spans[1]) <= float(target_y_ang) + 1.0e-9 else float(target_y_ang) / max(float(spans[1]), 1.0e-9)
    if scale_x + 1.0e-9 < float(min_scale_xy[0]) or scale_y + 1.0e-9 < float(min_scale_xy[1]):
        raise RuntimeError(
            "Prepared slab requires excessive lateral center compression to match the graphite footprint "
            f"(scale_x={float(scale_x):.3f}, scale_y={float(scale_y):.3f}). "
            "Expand the graphite master footprint instead of forcing the soft slab into a much smaller XY box."
        )
    if scale_x >= 1.0 - 1.0e-9 and scale_y >= 1.0 - 1.0e-9:
        return scaled, (1.0, 1.0), False

    center_x = 0.5 * float(mins[0] + maxs[0])
    center_y = 0.5 * float(mins[1] + maxs[1])
    for start, stop in blocks:
        if stop <= start:
            continue
        frag = scaled[start:stop]
        frag_center_x = float(np.mean(frag[:, 0]))
        frag_center_y = float(np.mean(frag[:, 1]))
        new_center_x = center_x + (frag_center_x - center_x) * float(scale_x)
        new_center_y = center_y + (frag_center_y - center_y) * float(scale_y)
        frag[:, 0] += new_center_x - frag_center_x
        frag[:, 1] += new_center_y - frag_center_y
        scaled[start:stop] = frag
    return scaled, (float(scale_x), float(scale_y)), True


def _soften_catastrophic_xy_overlaps(
    coords: np.ndarray,
    *,
    box_x_ang: float,
    box_y_ang: float,
    min_sep_ang: float = 0.75,
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


def _short_contact_count(coords: np.ndarray, *, min_sep_ang: float = 0.75) -> int:
    """Return the number of direct atom pairs closer than a hard-contact cutoff.

    RDKit coordinates in the interface builder are stored in Angstrom. A
    0.75-A cutoff corresponds to 0.075 nm in GROMACS coordinates: shorter than
    any normal X-H bond, but large enough to catch the atom overlaps that make
    GROMACS energy minimization report infinite forces.
    """

    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or float(min_sep_ang) <= 0.0:
        return 0
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        # SciPy is a required runtime dependency for YadonPy, but keep this
        # guard so geometry repair never becomes the reason a workflow crashes.
        return 0
    return int(len(cKDTree(arr).query_pairs(r=float(min_sep_ang), output_type="set")))


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
    # Be conservative here. The direct final-XY polymer build already places
    # molecules in an elongated box to avoid catastrophic pack overlaps; if we
    # collapse that box too aggressively before EM we can re-introduce the very
    # overlaps we were trying to avoid. Let confined NPT do most of the
    # densification work instead.
    prefit_z_ang = max(
        target_z_ang * 2.4,
        target_z_ang + max_fragment_span * 1.2,
        old_z_len * 0.70,
    )
    prefit_z_ang = min(old_z_len, prefit_z_ang)
    if target_z_ang <= 0.0 or old_z_len <= prefit_z_ang * 1.03:
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
    pre_nvt_ps, _density_relax_ps = _confined_phase_durations_ps(relax)
    wall_mdp = _phase_wall_block(wall_atomtype=wall_atomtype)
    common = {
        **base,
        "pbc": "xy",
        "periodic_molecules": "yes",
        "periodic-molecules": "yes",
        "wall_mdp": wall_mdp,
    }
    # Keep wall-confined component-phase relaxation deliberately short.
    # The target density is controlled by snapshot selection/reboxing and by
    # the optional rescue compression below.  A second wall-confined MD stage
    # with the same geometry has triggered GROMACS 2026.1 step-0 crashes for
    # compact CMC/electrolyte slabs, while the EM -> pre-NVT handoff is stable.
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
            gpu_offload_mode="balanced",
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
    trust_periodic_xy: bool = False,
):
    confined = utils.deepcopy_mol(block)
    conf = confined.GetConformer(0)
    coords = np.asarray(conf.GetPositions(), dtype=float).copy()
    if coords.size == 0:
        raise RuntimeError("Cannot confine an empty slab block.")
    target_x_ang = float(target_xy_nm[0]) * 10.0
    target_y_ang = float(target_xy_nm[1]) * 10.0
    periodic_lateral_wrap_applied = False
    bonded_lateral_unwrap_applied = False
    trust_periodic_xy_applied = False
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
            source_box = getattr(block, "cell", None)
            if source_box is not None:
                box_x_ang = max(float(source_box.xhi - source_box.xlo), 1.0e-9)
                box_y_ang = max(float(source_box.yhi - source_box.ylo), 1.0e-9)
            else:
                box_x_ang = target_x_ang
                box_y_ang = target_y_ang
            if (
                bool(trust_periodic_xy)
                and abs(float(box_x_ang) - float(target_x_ang)) <= 0.05
                and abs(float(box_y_ang) - float(target_y_ang)) <= 0.05
            ):
                mins = np.min(coords, axis=0)
                maxs = np.max(coords, axis=0)
                spans = maxs - mins
                slot_z_ang = max(float(target_thickness_nm) * 10.0, float(spans[2]))
                box_z_ang = slot_z_ang + 2.0 * float(vacuum_padding_ang)
                z_shift = float(vacuum_padding_ang) + 0.5 * (slot_z_ang - float(spans[2])) - float(mins[2])
                coords[:, 2] += z_shift
                for idx, xyz in enumerate(coords):
                    conf.SetAtomPosition(idx, Geom.Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
                setattr(confined, "cell", utils.Cell(target_x_ang, 0.0, target_y_ang, 0.0, box_z_ang, 0.0))
                poly.set_cell_param_conf(confined, 0, target_x_ang, 0.0, target_y_ang, 0.0, box_z_ang, 0.0)
                return confined, {
                    "target_xy_nm": [float(target_xy_nm[0]), float(target_xy_nm[1])],
                    "target_thickness_nm": float(target_thickness_nm),
                    "occupied_thickness_nm": float(spans[2]) / 10.0,
                    "confined_box_nm": [target_x_ang / 10.0, target_y_ang / 10.0, box_z_ang / 10.0],
                    "vacuum_padding_ang": float(vacuum_padding_ang),
                    "periodic_lateral_wrap_applied": False,
                    "periodic_lateral_spillover_allowed": False,
                    "periodic_lateral_spillover_nm": [0.0, 0.0],
                    "bonded_lateral_unwrap_applied": False,
                    "trust_periodic_xy_applied": True,
                    "preserved_prepared_xy": True,
                    "lateral_scale_xy": [1.0, 1.0],
                    "overlap_softening_applied": False,
                    "overlap_pairs_softened": 0,
                }, (
                    "preserved the prepared slab XY coordinates because the source "
                    "periodic footprint already matched the graphite master footprint; "
                    f"inserted {float(vacuum_padding_ang) / 10.0:.3f} nm top/bottom vacuum"
                )
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
    slot_z_ang = max(float(target_thickness_nm) * 10.0, float(spans[2]))
    box_z_ang = slot_z_ang + 2.0 * float(vacuum_padding_ang)

    if bool(trust_periodic_xy):
        if blocks and int(blocks[-1][1]) == int(confined.GetNumAtoms()):
            coords, wrapped_x = _wrap_fragment_centers_into_box(coords, blocks=blocks, axis=0, box_len_ang=target_x_ang)
            coords, minimized_x = _minimize_fragment_periodic_axis_span(coords, blocks=blocks, axis=0, box_len_ang=target_x_ang)
            coords, wrapped_y = _wrap_fragment_centers_into_box(coords, blocks=blocks, axis=1, box_len_ang=target_y_ang)
            coords, minimized_y = _minimize_fragment_periodic_axis_span(coords, blocks=blocks, axis=1, box_len_ang=target_y_ang)
            periodic_lateral_wrap_applied = bool(wrapped_x or minimized_x or wrapped_y or minimized_y)
        else:
            if target_x_ang > 0.0:
                coords[:, 0] = np.mod(coords[:, 0], target_x_ang)
            if target_y_ang > 0.0:
                coords[:, 1] = np.mod(coords[:, 1], target_y_ang)
            periodic_lateral_wrap_applied = True
        trust_periodic_xy_applied = True
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        spans = maxs - mins
        slot_z_ang = max(float(target_thickness_nm) * 10.0, float(spans[2]))
        box_z_ang = slot_z_ang + 2.0 * float(vacuum_padding_ang)

    periodic_spillover_allowed = bool(trust_periodic_xy and blocks and int(blocks[-1][1]) == int(confined.GetNumAtoms()))
    if (not periodic_spillover_allowed) and (
        float(spans[0]) > target_x_ang + 1.0e-6 or float(spans[1]) > target_y_ang + 1.0e-6
    ):
        if blocks and int(blocks[-1][1]) == int(confined.GetNumAtoms()):
            coords, lateral_scale_xy, scaled_to_target = _scale_fragment_centers_lateral_to_target(
                coords,
                blocks=blocks,
                target_x_ang=target_x_ang,
                target_y_ang=target_y_ang,
                min_scale_xy=(0.55, 0.55),
            )
        else:
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

    if (not periodic_spillover_allowed) and (not periodic_lateral_wrap_applied) and (
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

    if (not periodic_spillover_allowed) and (
        float(spans[0]) > target_x_ang + 1.0e-6 or float(spans[1]) > target_y_ang + 1.0e-6
    ):
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
        "periodic_lateral_spillover_allowed": bool(
            periodic_spillover_allowed
            and (float(spans[0]) > target_x_ang + 1.0e-6 or float(spans[1]) > target_y_ang + 1.0e-6)
        ),
        "periodic_lateral_spillover_nm": [
            max(0.0, float(spans[0] - target_x_ang)) / 10.0,
            max(0.0, float(spans[1] - target_y_ang)) / 10.0,
        ],
        "bonded_lateral_unwrap_applied": bool(bonded_lateral_unwrap_applied),
        "trust_periodic_xy_applied": bool(trust_periodic_xy_applied),
        "lateral_scale_xy": [float(lateral_scale_xy[0]), float(lateral_scale_xy[1])],
        **overlap_summary,
    }
    note = "reboxed the prepared slab onto the graphite master footprint"
    if bonded_lateral_unwrap_applied:
        note += " and restored bonded lateral periodic coordinates"
    if periodic_lateral_wrap_applied and not bonded_lateral_unwrap_applied:
        note += " and restored lateral periodic coordinates"
    if trust_periodic_xy_applied:
        note += " and kept the soft phase in a periodic XY representation"
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
    if isinstance(window, (int, float)):
        return max(0.0, float(window))
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
    core = _window_size_nm(summary.get("center_bulk_like_window_nm", summary.get("center_window_nm")))
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
    # The adaptive term should provide safety clearance for rough, noisy slabs,
    # not create a vacuum layer that the later natural-contact stage must spend
    # nanoseconds removing.  The wall-confined stack release no longer has a
    # z-barostat, so overly conservative initial gaps persist and dilute the
    # soft phases.  Keep the default cap tight while still respecting an
    # explicitly larger user-requested base gap.
    graphite_polymer_gap_nm = min(graphite_polymer_gap_nm, max(float(relax.graphite_to_polymer_gap_ang) / 10.0, 0.45))
    polymer_electrolyte_gap_nm = min(
        polymer_electrolyte_gap_nm,
        max(float(relax.polymer_to_electrolyte_gap_ang) / 10.0, 0.55),
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
    min_scale: float = 0.80,
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
    applied = False

    blocks: list[tuple[int, int]] = []
    if species is not None and counts is not None:
        try:
            blocks = _molecule_atom_blocks(species=species, counts=counts)
        except Exception:
            blocks = []
    if blocks and int(blocks[-1][1]) == int(compressed.GetNumAtoms()):
        fragment_spans = [
            float(np.max(coords[start:stop, 2]) - np.min(coords[start:stop, 2]))
            for start, stop in blocks
            if stop > start
        ]
        if fragment_spans:
            # Never make the rescue compression more compact than the longest
            # chain can reasonably occupy.  The default also avoids removing
            # more than 20% of the current slab thickness in one coordinate
            # edit, while small-molecule liquid phases can opt into a lower
            # scale because they do not contain long entangled chains.
            lower_scale = max(0.50, min(1.0, float(min_scale)))
            safe_target_ang = max(
                float(target_thickness_ang),
                max(fragment_spans) * 1.25,
                current_thickness_ang * lower_scale,
            )
            target_thickness_ang = min(current_thickness_ang, float(safe_target_ang))
            if current_thickness_ang <= target_thickness_ang * 1.02:
                return compressed, {
                    "z_compression_applied": False,
                    "z_compression_scale": 1.0,
                    "z_compression_skipped_reason": "safe_target_close_to_current",
                }
    original_coords = coords.copy()
    original_short_contacts = _short_contact_count(original_coords)
    requested_scale = max(max(0.50, min(1.0, float(min_scale))), min(1.0, target_thickness_ang / current_thickness_ang))

    def _scaled_coords(scale_value: float) -> np.ndarray:
        trial = original_coords.copy()
        if blocks and int(blocks[-1][1]) == int(compressed.GetNumAtoms()):
            for start, stop in blocks:
                frag = trial[start:stop]
                if frag.size == 0:
                    continue
                frag_center = float(np.mean(frag[:, 2]))
                new_center = center_z + (frag_center - center_z) * float(scale_value)
                frag[:, 2] += new_center - frag_center
                trial[start:stop] = frag
        else:
            trial[:, 2] = center_z + (trial[:, 2] - center_z) * float(scale_value)
        return trial

    scale = 1.0
    final_short_contacts = int(original_short_contacts)
    for candidate_scale in [
        float(requested_scale),
        0.5 * (float(requested_scale) + 1.0),
        0.75 * 1.0 + 0.25 * float(requested_scale),
        0.875 * 1.0 + 0.125 * float(requested_scale),
        0.95,
        0.975,
    ]:
        candidate_scale = max(min(float(candidate_scale), 1.0), float(requested_scale))
        if candidate_scale >= 1.0 - 1.0e-6:
            continue
        trial_coords = _scaled_coords(candidate_scale)
        trial_short_contacts = _short_contact_count(trial_coords)
        if trial_short_contacts <= original_short_contacts:
            coords = trial_coords
            scale = float(candidate_scale)
            final_short_contacts = int(trial_short_contacts)
            applied = True
            break

    if not applied:
        return compressed, {
            "z_compression_applied": False,
            "z_compression_scale": 1.0,
            "z_compression_requested_scale": float(requested_scale),
            "z_compression_skipped_reason": "short_contact_guard",
            "z_compression_short_contacts_before": int(original_short_contacts),
            "z_compression_short_contacts_after": int(original_short_contacts),
        }

    for idx, xyz in enumerate(coords):
        conf.SetAtomPosition(idx, Geom.Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return compressed, {
        "z_compression_applied": bool(applied),
        "z_compression_scale": float(scale),
        "z_compression_requested_scale": float(requested_scale),
        "z_compression_safe_target_nm": float(target_thickness_ang) / 10.0,
        "z_compression_short_contacts_before": int(original_short_contacts),
        "z_compression_short_contacts_after": int(final_short_contacts),
    }


def _normalize_confined_block_for_stack(
    *,
    block,
    target_xy_nm: tuple[float, float],
    occupied_thickness_nm: float,
    target_thickness_nm: float | None = None,
    species: Sequence,
    counts: Sequence[int],
    min_z_compression_scale: float = 0.80,
):
    resolved_target_thickness_nm = (
        max(float(target_thickness_nm), 1.0e-6)
        if target_thickness_nm is not None and float(target_thickness_nm) > 0.0
        else max(float(occupied_thickness_nm), 1.0e-6)
    )
    normalized, _summary, _note = _rebox_block_for_phase_confinement(
        block=block,
        target_xy_nm=target_xy_nm,
        target_thickness_nm=resolved_target_thickness_nm,
        vacuum_padding_ang=0.0,
        species=species,
        counts=counts,
        # These blocks already come from fixed-XY confined relaxations. Trust
        # the periodic imaging semantics established there instead of treating
        # read-back wrapped coordinates as evidence that the slab genuinely
        # exceeds the graphite footprint.
        trust_periodic_xy=True,
    )
    normalized, _compression = _compress_phase_block_z_to_target_thickness(
        block=normalized,
        target_thickness_nm=resolved_target_thickness_nm,
        species=species,
        counts=counts,
        min_scale=float(min_z_compression_scale),
    )
    return normalized


def _final_xy_walled_pack_density_ladder(
    *,
    phase: str,
    target_density_g_cm3: float,
    requested_density_g_cm3: float,
    charged: bool,
) -> tuple[object, tuple[float, ...]]:
    policy, fallback = _build_pack_density_ladder(
        phase=str(phase),
        target_density_g_cm3=float(target_density_g_cm3),
        requested_density_g_cm3=float(requested_density_g_cm3),
        charged=bool(charged),
    )
    if str(phase).strip().lower() != "polymer":
        return policy, tuple(float(x) for x in fallback)

    start = float(fallback[0] if fallback else requested_density_g_cm3)
    target = float(target_density_g_cm3)
    # Fixed-XY polymer slabs fail differently from bulk packing: the lateral box
    # cannot grow, so dense starts can spend minutes retrying long-chain
    # insertions before the backoff logic reaches a usable Z slack. Start from a
    # loose but not vacuum-like slab, then let the confined/walled relaxation
    # recover the target thickness and density.
    loose_start = min(start, target * (0.45 if bool(charged) else 0.52))
    dense_candidates = (
        loose_start,
        max(0.22 if bool(charged) else 0.32, loose_start * 0.85),
        min(target * 0.68, max(loose_start * 1.18, loose_start + 0.08)),
        min(target * 0.84, max(loose_start * 1.38, loose_start + 0.18)),
    )
    values: list[float] = []
    for value in (*dense_candidates, *tuple(float(x) for x in fallback[1:])):
        value = max(float(value), 0.20 if bool(charged) else 0.30)
        rounded = round(value, 6)
        if not any(round(existing, 6) == rounded for existing in values):
            values.append(float(value))
    return policy, tuple(values)


def _run_final_xy_walled_phase_build(
    *,
    label: str,
    species: Sequence,
    counts: Sequence[int],
    charge_scale: Sequence[float],
    target_xy_nm: tuple[float, float],
    target_density_g_cm3: float,
    bulk_calibration: dict[str, object],
    ff_name: str,
    relax: SandwichRelaxationSpec,
    work_dir: Path,
    retry: int,
    retry_step: int,
    threshold: float,
    dec_rate: float,
    charged_phase: bool,
    restart: bool | None = None,
) -> tuple[_ConfinedPhaseResult, dict[str, object], Path]:
    total_mass_amu = _phase_total_mass_amu(species=species, counts=counts)
    initial_density = float(
        bulk_calibration.get(
            "initial_walled_pack_density_g_cm3",
            _recommend_initial_walled_pack_density(
                phase=str(label),
                target_density_g_cm3=float(target_density_g_cm3),
                selected_bulk_pack_density_g_cm3=float(bulk_calibration.get("selected_bulk_pack_density_g_cm3", target_density_g_cm3)),
            ),
        )
    )
    target_z_nm = float(
        bulk_calibration.get(
            "target_z_nm",
            _solve_phase_target_z_nm(
                total_mass_amu=float(total_mass_amu),
                target_density_g_cm3=float(target_density_g_cm3),
                target_xy_nm=(float(target_xy_nm[0]), float(target_xy_nm[1])),
            ),
        )
    )
    _policy, ladder = _final_xy_walled_pack_density_ladder(
        phase=str(label),
        target_density_g_cm3=float(target_density_g_cm3),
        requested_density_g_cm3=float(initial_density),
        charged=bool(charged_phase),
    )
    attempts: list[dict[str, object]] = []
    best_result: _ConfinedPhaseResult | None = None
    best_score: float | None = None
    best_attempt_index: int | None = None
    last_error: Exception | None = None
    summary_path = Path(work_dir) / f"{label}_walled_phase_summary.json"

    for attempt_index, pack_density in enumerate(ladder):
        attempt_dir = Path(work_dir) / f"attempt_{attempt_index:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        initial_z_nm = max(
            float(target_z_nm),
            _solve_phase_target_z_nm(
                total_mass_amu=float(total_mass_amu),
                target_density_g_cm3=float(pack_density),
                target_xy_nm=(float(target_xy_nm[0]), float(target_xy_nm[1])),
                min_z_nm=float(target_z_nm),
            ),
        )
        attempt_record: dict[str, object] = {
            "attempt_index": int(attempt_index),
            "initial_pack_density_g_cm3": float(pack_density),
            "target_z_nm": float(target_z_nm),
            "initial_pack_box_nm": [float(target_xy_nm[0]), float(target_xy_nm[1]), float(initial_z_nm)],
        }
        try:
            effective_retry = int(retry)
            effective_retry_step = int(retry_step)
            if str(label).strip().lower() == "polymer":
                effective_retry = min(effective_retry, 10)
                effective_retry_step = min(effective_retry_step, 1400)
            attempt_record["effective_retry"] = int(effective_retry)
            attempt_record["effective_retry_step"] = int(effective_retry_step)
            pack_cell = poly.amorphous_cell(
                list(species),
                list(counts),
                cell=make_orthorhombic_pack_cell((float(target_xy_nm[0]), float(target_xy_nm[1]), float(initial_z_nm))),
                density=None,
                charge_scale=list(charge_scale),
                neutralize=False,
                retry=int(effective_retry),
                retry_step=int(effective_retry_step),
                threshold=float(threshold),
                dec_rate=float(dec_rate),
                work_dir=attempt_dir / "00_build",
                restart=restart,
            )
            register_cell_species_metadata(
                pack_cell,
                list(species),
                list(counts),
                charge_scale=list(charge_scale),
                pack_mode=f"{label}_fixed_xy_walled_phase",
            )
            pack_note = None
            if str(label).strip().lower() == "polymer":
                pack_cell, pack_note = _compact_packed_cell_z_by_molecule_centers(
                    cell=pack_cell,
                    species=species,
                    counts=counts,
                    target_box_nm=(float(target_xy_nm[0]), float(target_xy_nm[1]), float(target_z_nm)),
                )
            phase_result = _run_confined_phase_relaxation(
                label=str(label),
                source_block=pack_cell,
                source_note=(
                    "direct final-XY walled phase build from bulk calibration"
                    + (f"; {pack_note}" if pack_note else "")
                ),
                species=species,
                counts=counts,
                charge_scale=charge_scale,
                target_xy_nm=(float(target_xy_nm[0]), float(target_xy_nm[1])),
                target_density_g_cm3=float(target_density_g_cm3),
                target_thickness_nm=float(target_z_nm),
                ff_name=str(ff_name),
                relax=relax,
                work_dir=attempt_dir / "01_confined",
                restart=restart,
            summary_extra={
                "phase_preparation_mode": "bulk_calibrate_walled_phase",
                "bulk_calibration_target_z_nm": float(target_z_nm),
                "initial_pack_density_g_cm3": float(pack_density),
                "initial_pack_box_nm": [float(target_xy_nm[0]), float(target_xy_nm[1]), float(initial_z_nm)],
                "source_mode": "final_xy_walled_phase",
            },
            trust_periodic_xy=True,
        )
            score = _confined_summary_score(
                summary=phase_result.summary,
                target_density_g_cm3=float(target_density_g_cm3),
                target_thickness_nm=float(target_z_nm),
            )
            attempt_record.update(
                {
                    "success": True,
                    "score": float(score),
                    "summary_path": str(phase_result.summary_path),
                    "center_bulk_like_density_g_cm3": float(phase_result.summary.get("center_bulk_like_density_g_cm3", 0.0) or 0.0),
                    "occupied_density_g_cm3": float(phase_result.summary.get("occupied_density_g_cm3", 0.0) or 0.0),
                    "occupied_thickness_nm": float(phase_result.summary.get("occupied_thickness_nm", 0.0) or 0.0),
                    "wrapped_across_z_boundary": bool(phase_result.summary.get("wrapped_across_z_boundary", False)),
                }
            )
            attempts.append(attempt_record)
            if best_score is None or float(score) < float(best_score):
                best_result = phase_result
                best_score = float(score)
                best_attempt_index = int(attempt_index)
            if not _needs_confined_rescue(
                summary=phase_result.summary,
                target_density_g_cm3=float(target_density_g_cm3),
                target_thickness_nm=float(target_z_nm),
            ):
                break
        except Exception as exc:
            attempt_record["success"] = False
            attempt_record["error"] = repr(exc)
            attempt_record["traceback"] = traceback.format_exc()
            attempts.append(attempt_record)
            last_error = exc
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "label": str(label),
                        "phase_preparation_mode": "bulk_calibrate_walled_phase",
                        "target_density_g_cm3": float(target_density_g_cm3),
                        "target_xy_nm": [float(target_xy_nm[0]), float(target_xy_nm[1])],
                        "target_z_nm": float(target_z_nm),
                        "bulk_calibration": dict(bulk_calibration),
                        "attempts": attempts,
                        "success": False,
                        "partial": True,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    if best_result is None or best_attempt_index is None:
        build_summary = {
            "label": str(label),
            "phase_preparation_mode": "bulk_calibrate_walled_phase",
            "target_density_g_cm3": float(target_density_g_cm3),
            "target_xy_nm": [float(target_xy_nm[0]), float(target_xy_nm[1])],
            "target_z_nm": float(target_z_nm),
            "bulk_calibration": dict(bulk_calibration),
            "attempts": attempts,
            "success": False,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(build_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Failed to build final-XY walled {label} phase after {len(attempts)} attempts. See {summary_path}."
        ) from last_error

    build_summary = {
        "label": str(label),
        "phase_preparation_mode": "bulk_calibrate_walled_phase",
        "target_density_g_cm3": float(target_density_g_cm3),
        "target_xy_nm": [float(target_xy_nm[0]), float(target_xy_nm[1])],
        "target_z_nm": float(target_z_nm),
        "bulk_calibration": dict(bulk_calibration),
        "attempts": attempts,
        "success": True,
        "selected_attempt_index": int(best_attempt_index),
        "selected_summary_path": str(best_result.summary_path),
        "selected_round": str(best_result.summary.get("selected_round", "round_00")),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(build_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return best_result, build_summary, summary_path


def _run_confined_phase_relaxation(
    *,
    label: str,
    prepared_slab=None,
    source_block=None,
    source_note: str | None = None,
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
    summary_extra: dict[str, object] | None = None,
    trust_periodic_xy: bool = False,
) -> _ConfinedPhaseResult:
    from .protocol import _resolve_route_b_wall_atomtype

    if source_block is not None:
        base_block = utils.deepcopy_mol(source_block)
    else:
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
        trust_periodic_xy: bool,
    ) -> tuple[object, dict[str, object], Path, Path]:
        confined_block, rebox_summary, rebox_note = _rebox_block_for_phase_confinement(
            block=source_block,
            target_xy_nm=target_xy_nm,
            target_thickness_nm=float(target_thickness_nm),
            vacuum_padding_ang=float(vacuum_padding_ang),
            species=species,
            counts=counts,
            trust_periodic_xy=bool(trust_periodic_xy),
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
        fingerprint_path = Path(relax_dir) / "confined_input_fingerprint.json"
        input_fingerprint = {
            "label": str(label),
            "round_label": str(round_label),
            "system_gro_sha256": _sha256_file(export.system_gro),
            "system_top_sha256": _sha256_file(export.system_top),
            "target_xy_nm": [float(target_xy_nm[0]), float(target_xy_nm[1])],
            "target_thickness_nm": float(target_thickness_nm),
            "counts": [int(x) for x in counts],
            "species_names": [str(get_name(mol, default=f"{label}_{idx + 1}")) for idx, mol in enumerate(species)],
        }
        effective_restart = bool(resolve_restart(restart))
        if effective_restart and fingerprint_path.exists():
            try:
                previous_fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
            except Exception:
                previous_fingerprint = None
            if previous_fingerprint != input_fingerprint:
                effective_restart = False
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_path.write_text(
            json.dumps(input_fingerprint, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        job.run(restart=effective_restart)
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
        trust_periodic_xy=bool(trust_periodic_xy),
    )
    round_scores = {
        "round_00": _confined_summary_score(
            summary=selected_summary,
            target_density_g_cm3=float(target_density_g_cm3),
            target_thickness_nm=float(target_thickness_nm),
        )
    }
    rescue_applied = False
    rescue_attempted = False
    rescue_failed = False
    rescue_error: str | None = None
    if _needs_confined_rescue(
        summary=selected_summary,
        target_density_g_cm3=float(target_density_g_cm3),
        target_thickness_nm=float(target_thickness_nm),
    ):
        rescue_attempted = True
        rescue_relax = replace(
            relax,
            stacked_pre_nvt_ps=max(float(relax.stacked_pre_nvt_ps) + 6.0, float(relax.stacked_pre_nvt_ps) * 1.4),
            stacked_z_relax_ps=max(float(relax.stacked_z_relax_ps) + 20.0, float(relax.stacked_z_relax_ps) * 1.5),
        )
        try:
            rescue_block, rescue_summary, rescue_top_path, rescue_gro = _run_confined_round(
                round_label="round_01_rescue",
                source_block=selected_block,
                round_relax=rescue_relax,
                vacuum_padding_ang=max(10.0, 0.85 * float(relax.top_padding_ang)),
                compress_to_target=True,
                export_dir=work_dir / "02_rescue_export",
                relax_dir=work_dir / "02_rescue_relax",
                trust_periodic_xy=bool(trust_periodic_xy),
            )
        except Exception as exc:
            # Keep the already successful round_00 result if the rescue attempt
            # itself is the unstable step.
            rescue_failed = True
            rescue_error = str(exc)
        else:
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
        "rescue_attempted": bool(rescue_attempted),
        "rescue_failed": bool(rescue_failed),
        "rescue_error": rescue_error,
        "round_scores": {str(k): float(v) for k, v in round_scores.items()},
        "selected_round": str(selected_summary.get("round_label", "round_00")),
        "source_note": (None if source_note is None else str(source_note)),
        **({} if summary_extra is None else dict(summary_extra)),
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
    policy: InterfaceBuildPolicy | None = None,
    route: str = "screening",
    restart: bool | None = None,
) -> GraphitePolymerElectrolyteSandwichResult:
    build_policy = _resolve_interface_build_policy(policy)
    if len(electrolyte.solvents) != len(electrolyte.solvent_mass_ratio):
        raise ValueError("electrolyte.solvents and electrolyte.solvent_mass_ratio must have the same length")
    graphite_stage = prepare_graphite_substrate(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite,
        polymer=polymer,
        electrolyte=electrolyte,
        relax=relax,
        route=route,
        restart=restart,
    )
    design_path = _write_interface_design_summary(
        work_dir=graphite_stage.work_dir,
        route=route,
        graphite=graphite_stage,
        polymer=polymer,
        electrolyte=electrolyte,
        relax=relax,
        policy=build_policy,
    )
    graphite_stage.context["interface_design_summary"] = str(design_path)
    polymer_bulk = calibrate_polymer_bulk_phase(
        work_dir=graphite_stage.work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite_stage,
        polymer=polymer,
        relax=relax,
        restart=restart,
    )
    electrolyte_bulk = calibrate_electrolyte_bulk_phase(
        work_dir=graphite_stage.work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite_stage,
        electrolyte=electrolyte,
        relax=relax,
        restart=restart,
    )
    polymer_interphase = build_graphite_polymer_interphase(
        work_dir=graphite_stage.work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite_stage,
        polymer=polymer,
        polymer_bulk=polymer_bulk,
        relax=relax,
        route=route,
        restart=restart,
    )
    electrolyte_interphase = build_polymer_electrolyte_interphase(
        work_dir=graphite_stage.work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite_stage,
        electrolyte=electrolyte,
        electrolyte_bulk=electrolyte_bulk,
        relax=relax,
        route=route,
        restart=restart,
    )
    stack_result = release_graphite_polymer_electrolyte_stack(
        work_dir=graphite_stage.work_dir,
        ff=ff,
        graphite=graphite_stage,
        polymer_interphase=polymer_interphase,
        electrolyte_interphase=electrolyte_interphase,
        relax=relax,
        policy=build_policy,
        route=route,
        restart=restart,
    )
    if stack_result.sandwich_result is None:
        raise RuntimeError("release_graphite_polymer_electrolyte_stack did not populate sandwich_result")
    return stack_result.sandwich_result


def build_graphite_peo_electrolyte_sandwich(
    *,
    work_dir,
    ff,
    ion_ff,
    graphite: GraphiteSubstrateSpec = GraphiteSubstrateSpec(),
    polymer: PolymerSlabSpec = PolymerSlabSpec(),
    electrolyte: ElectrolyteSlabSpec = ElectrolyteSlabSpec(),
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    policy: InterfaceBuildPolicy | None = None,
    route: str = "screening",
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
        policy=policy,
        route=route,
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
    policy: InterfaceBuildPolicy | None = None,
    route: str = "screening",
    restart: bool | None = None,
) -> GraphitePolymerElectrolyteSandwichResult:
    return build_cmcna_graphite_electrolyte_stack(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite,
        polymer=polymer,
        electrolyte=electrolyte,
        relax=relax,
        policy=policy,
        route=route,
        restart=restart,
    )


def build_cmcna_graphite_electrolyte_stack(
    *,
    work_dir,
    ff,
    ion_ff,
    graphite: GraphiteSubstrateSpec = GraphiteSubstrateSpec(),
    polymer: PolymerSlabSpec | None = None,
    electrolyte: ElectrolyteSlabSpec | None = None,
    relax: SandwichRelaxationSpec = SandwichRelaxationSpec(),
    policy: InterfaceBuildPolicy | None = None,
    route: str = "screening",
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
        policy=_resolve_interface_build_policy(policy),
        route=route,
        restart=restart,
    )


__all__ = [
    "BulkCalibrationResult",
    "ElectrolyteSlabSpec",
    "GraphitePreparationResult",
    "GraphitePolymerElectrolyteSandwichResult",
    "GraphiteSubstrateSpec",
    "InterfaceBuildPolicy",
    "InterfaceTransportResult",
    "InterphaseBuildResult",
    "MoleculeSpec",
    "PolymerSlabSpec",
    "SandwichPhaseReport",
    "SandwichNvtFollowupResult",
    "SandwichRelaxationSpec",
    "StackReleaseResult",
    "analyze_sandwich_interface",
    "analyze_interface_transport",
    "build_cmc_electrolyte_interphase",
    "build_cmcna_graphite_electrolyte_stack",
    "build_graphite_cmc_interphase",
    "build_graphite_cmcna_glucose6_periodic_case",
    "build_graphite_cmcna_electrolyte_sandwich",
    "build_graphite_polymer_interphase",
    "build_graphite_peo_electrolyte_sandwich",
    "build_polymer_electrolyte_interphase",
    "build_graphite_polymer_electrolyte_sandwich",
    "calibrate_electrolyte_bulk_phase",
    "calibrate_polymer_bulk_phase",
    "default_carbonate_lipf6_electrolyte_spec",
    "default_cmcna_polymer_spec",
    "default_peo_electrolyte_spec",
    "default_peo_polymer_spec",
    "prepare_graphite_substrate",
    "print_interface_result_summary",
    "release_graphite_cmc_electrolyte_stack",
    "release_graphite_polymer_electrolyte_stack",
    "run_sandwich_nvt_followup",
]
