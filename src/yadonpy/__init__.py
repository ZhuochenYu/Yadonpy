"""YadonPy package root."""

from __future__ import annotations

import os

from ._version import __version__
from .core.data_dir import ensure_initialized as _ensure_initialized


def _auto_initialize_user_data() -> None:
    flag = (os.environ.get("YADONPY_AUTO_INIT") or "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return
    try:
        _ensure_initialized()
    except Exception:
        # Importing yadonpy should stay lightweight and not fail solely because
        # the user's data directory cannot be initialized at import time.
        pass


_auto_initialize_user_data()

from .sim import qm  # noqa: F401
from .sim.analyzer import AnalyzeResult  # noqa: F401
from .diagnostics import doctor  # noqa: F401
from .interface import BulkCalibrationResult, ElectrolyteSlabSpec, GraphitePreparationResult, GraphiteSubstrateSpec, InterfaceBuilder, InterfaceDynamics, InterfaceProtocol, InterfaceRouteSpec, InterfaceTransportResult, InterphaseBuildResult, MoleculeSpec, PolymerSlabSpec, SandwichRelaxationSpec, StackReleaseResult, analyze_interface_transport, build_cmc_electrolyte_interphase, build_graphite_cmc_interphase, build_graphite_cmcna_example_case, build_graphite_cmcna_glucose6_periodic_case, build_graphite_cmcna_electrolyte_sandwich, build_graphite_peo_electrolyte_sandwich, build_graphite_peo_example_case, build_graphite_polymer_electrolyte_sandwich, build_graphite_polymer_interphase, build_interface, build_interface_from_workdirs, build_polymer_electrolyte_interphase, calibrate_electrolyte_bulk_phase, calibrate_polymer_bulk_phase, default_carbonate_lipf6_electrolyte_spec, default_cmcna_polymer_spec, default_peo_electrolyte_spec, default_peo_polymer_spec, prepare_graphite_substrate, print_interface_result_summary, print_sandwich_result_summary, release_graphite_cmc_electrolyte_stack, release_graphite_polymer_electrolyte_stack  # noqa: F401
from .runtime import get_run_options, set_run_options, run_options  # noqa: F401
from .api import (  # noqa: F401
    audit_default_moldb_sync,
    analyze_interface_transport,
    assign_charges,
    assign_forcefield,
    build_graphite,
    build_cmc_electrolyte_interphase,
    build_graphite_cmc_interphase,
    build_graphite_cmcna_example_case,
    build_graphite_cmcna_glucose6_periodic_case,
    build_graphite_cmcna_electrolyte_sandwich,
    build_graphite_peo_example_case,
    build_graphite_peo_electrolyte_sandwich,
    build_graphite_polymer_interphase,
    build_graphite_polymer_electrolyte_sandwich,
    build_polymer_electrolyte_interphase,
    calibrate_electrolyte_bulk_phase,
    calibrate_polymer_bulk_phase,
    conformation_search,
    default_carbonate_lipf6_electrolyte_spec,
    default_cmcna_polymer_spec,
    default_peo_electrolyte_spec,
    default_peo_polymer_spec,
    format_mechanics_result_summary,
    get_ff,
    list_charge_methods,
    list_forcefields,
    load_from_moldb,
    mol_from_smiles,
    parameterize_smiles,
    prepare_graphite_substrate,
    print_interface_result_summary,
    print_mechanics_result_summary,
    release_graphite_cmc_electrolyte_stack,
    release_graphite_polymer_electrolyte_stack,
    resolve_prepared_system,
    run_elongation_gmx,
    run_tg_scan_gmx,
)

__all__ = [
    '__version__',
    'assign_charges',
    'assign_forcefield',
    'AnalyzeResult',
    'audit_default_moldb_sync',
    'analyze_interface_transport',
    'build_graphite',
    'build_cmc_electrolyte_interphase',
    'build_graphite_cmc_interphase',
    'build_graphite_cmcna_example_case',
    'build_graphite_cmcna_glucose6_periodic_case',
    'build_graphite_cmcna_electrolyte_sandwich',
    'build_graphite_peo_example_case',
    'build_graphite_peo_electrolyte_sandwich',
    'build_graphite_polymer_interphase',
    'build_graphite_polymer_electrolyte_sandwich',
    'build_polymer_electrolyte_interphase',
    'BulkCalibrationResult',
    'calibrate_electrolyte_bulk_phase',
    'calibrate_polymer_bulk_phase',
    'conformation_search',
    'doctor',
    'default_carbonate_lipf6_electrolyte_spec',
    'default_cmcna_polymer_spec',
    'default_peo_electrolyte_spec',
    'default_peo_polymer_spec',
    'ElectrolyteSlabSpec',
    'GraphitePreparationResult',
    'GraphiteSubstrateSpec',
    'InterfaceBuilder',
    'InterfaceDynamics',
    'InterfaceProtocol',
    'InterfaceRouteSpec',
    'InterfaceTransportResult',
    'InterphaseBuildResult',
    'get_ff',
    'get_run_options',
    'list_charge_methods',
    'list_forcefields',
    'build_interface',
    'build_interface_from_workdirs',
    'load_from_moldb',
    'mol_from_smiles',
    'MoleculeSpec',
    'parameterize_smiles',
    'PolymerSlabSpec',
    'prepare_graphite_substrate',
    'print_interface_result_summary',
    'print_sandwich_result_summary',
    'print_mechanics_result_summary',
    'qm',
    'release_graphite_cmc_electrolyte_stack',
    'release_graphite_polymer_electrolyte_stack',
    'resolve_prepared_system',
    'run_elongation_gmx',
    'run_options',
    'run_tg_scan_gmx',
    'SandwichRelaxationSpec',
    'set_run_options',
    'StackReleaseResult',
    'format_mechanics_result_summary',
]
