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
from .sim.cleanup import CleanupResult, clean_md_trajectory_files  # noqa: F401
from .sim.interface_analysis import InterfaceAnalysis  # noqa: F401
from .sim.parallel_postprocess import InterfaceAnalysisBatchResult, InterfaceAnalysisTask, InterfaceAnalysisTaskResult, run_interface_analyses_parallel  # noqa: F401
from .sim.performance import IOAnalysisPolicy, resolve_io_analysis_policy  # noqa: F401
from .sim.preset.eq import XYSlabEquilibrationSpec  # noqa: F401
from .diagnostics import doctor  # noqa: F401
from .interface import InterfaceBuilder, InterfaceDynamics, InterfaceProtocol, InterfaceRouteSpec, build_interface, build_interface_from_workdirs  # noqa: F401
from .interface import CMCNAXYBulkSlabResult, CMCNAXYSlabRelaxationSpec, ElectrodeChargeSpec, EnhancedSamplingPlan, FixedChargeRegionSpec, GraphiteLayerSpec, GraphiteRestraintSpec, InterdiffusionStartSpec, LayerStackNvtResult, LayerStackRelaxationResult, LayerStackRelaxationSpec, LayerStackResult, LayerStackSpec, MolecularLayerSpec, SolvatedIonPullSpec, SolvatedIonUmbrellaSpec, UmbrellaPmfResult, UmbrellaSamplingPlan, VacuumLayerSpec, ZCompressionAnnealSpec, analyze_layer_stack_interface, analyze_umbrella_pmf, build_layer_stack, prepare_cmcna_xy_bulk_slab, prepare_solvated_ion_pull, prepare_solvated_ion_umbrella, run_layer_stack_nvt, run_layer_stack_relaxation, run_solvated_ion_umbrella  # noqa: F401
from .runtime import get_run_options, set_run_options, run_options  # noqa: F401
from .api import (  # noqa: F401
    audit_default_moldb_sync,
    audit_bundled_oplsaa_parameter_sanity,
    audit_oplsaa_assignment,
    audit_oplsaa_reference,
    analyze_layer_stack_interface,
    analyze_umbrella_pmf,
    assign_charges,
    assign_forcefield,
    build_graphite,
    build_layer_stack,
    clean_md_trajectory_files,
    conformation_search,
    format_mechanics_result_summary,
    get_ff,
    list_charge_methods,
    list_forcefields,
    load_from_moldb,
    mol_from_smiles,
    oplsaa_stability_preflight,
    parameterize_smiles,
    print_mechanics_result_summary,
    prepare_solvated_ion_pull,
    prepare_solvated_ion_umbrella,
    prepare_cmcna_xy_bulk_slab,
    resolve_prepared_system,
    run_elongation_gmx,
    run_interface_analyses_parallel,
    run_layer_stack_nvt,
    run_layer_stack_relaxation,
    run_solvated_ion_umbrella,
    run_tg_scan_gmx,
)

__all__ = [
    '__version__',
    'assign_charges',
    'assign_forcefield',
    'AnalyzeResult',
    'CleanupResult',
    'CMCNAXYBulkSlabResult',
    'CMCNAXYSlabRelaxationSpec',
    'IOAnalysisPolicy',
    'audit_default_moldb_sync',
    'audit_bundled_oplsaa_parameter_sanity',
    'audit_oplsaa_assignment',
    'audit_oplsaa_reference',
    'analyze_layer_stack_interface',
    'analyze_umbrella_pmf',
    'build_graphite',
    'build_layer_stack',
    'clean_md_trajectory_files',
    'conformation_search',
    'doctor',
    'ElectrodeChargeSpec',
    'EnhancedSamplingPlan',
    'FixedChargeRegionSpec',
    'GraphiteLayerSpec',
    'GraphiteRestraintSpec',
    'InterdiffusionStartSpec',
    'InterfaceBuilder',
    'InterfaceAnalysis',
    'InterfaceAnalysisBatchResult',
    'InterfaceAnalysisTask',
    'InterfaceAnalysisTaskResult',
    'InterfaceDynamics',
    'InterfaceProtocol',
    'InterfaceRouteSpec',
    'LayerStackNvtResult',
    'LayerStackRelaxationResult',
    'LayerStackRelaxationSpec',
    'LayerStackResult',
    'LayerStackSpec',
    'get_ff',
    'get_run_options',
    'list_charge_methods',
    'list_forcefields',
    'build_interface',
    'build_interface_from_workdirs',
    'load_from_moldb',
    'mol_from_smiles',
    'MoleculeSpec',
    'MolecularLayerSpec',
    'oplsaa_stability_preflight',
    'parameterize_smiles',
    'print_mechanics_result_summary',
    'prepare_solvated_ion_pull',
    'prepare_solvated_ion_umbrella',
    'prepare_cmcna_xy_bulk_slab',
    'qm',
    'resolve_prepared_system',
    'resolve_io_analysis_policy',
    'run_elongation_gmx',
    'run_interface_analyses_parallel',
    'run_layer_stack_nvt',
    'run_layer_stack_relaxation',
    'run_options',
    'run_solvated_ion_umbrella',
    'run_tg_scan_gmx',
    'set_run_options',
    'SolvatedIonPullSpec',
    'SolvatedIonUmbrellaSpec',
    'UmbrellaPmfResult',
    'UmbrellaSamplingPlan',
    'VacuumLayerSpec',
    'XYSlabEquilibrationSpec',
    'ZCompressionAnnealSpec',
    'format_mechanics_result_summary',
]
