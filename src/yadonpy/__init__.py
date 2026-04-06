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
from .interface import InterfaceBuilder, InterfaceDynamics, InterfaceProtocol, InterfaceRouteSpec, build_graphite_cmcna_example_case, build_graphite_cmcna_glucose6_periodic_case, build_graphite_cmcna_electrolyte_sandwich, build_graphite_peo_electrolyte_sandwich, build_graphite_peo_example_case, build_graphite_polymer_electrolyte_sandwich, build_interface, build_interface_from_workdirs, print_sandwich_result_summary  # noqa: F401
from .runtime import get_run_options, set_run_options, run_options  # noqa: F401
from .api import (  # noqa: F401
    audit_default_moldb_sync,
    assign_charges,
    assign_forcefield,
    build_graphite,
    build_graphite_cmcna_example_case,
    build_graphite_cmcna_glucose6_periodic_case,
    build_graphite_cmcna_electrolyte_sandwich,
    build_graphite_peo_example_case,
    build_graphite_peo_electrolyte_sandwich,
    build_graphite_polymer_electrolyte_sandwich,
    conformation_search,
    get_ff,
    list_charge_methods,
    list_forcefields,
    load_from_moldb,
    mol_from_smiles,
    parameterize_smiles,
)

__all__ = [
    '__version__',
    'assign_charges',
    'assign_forcefield',
    'audit_default_moldb_sync',
    'build_graphite',
    'build_graphite_cmcna_example_case',
    'build_graphite_cmcna_glucose6_periodic_case',
    'build_graphite_cmcna_electrolyte_sandwich',
    'build_graphite_peo_example_case',
    'build_graphite_peo_electrolyte_sandwich',
    'build_graphite_polymer_electrolyte_sandwich',
    'conformation_search',
    'InterfaceBuilder',
    'InterfaceDynamics',
    'InterfaceProtocol',
    'InterfaceRouteSpec',
    'get_ff',
    'get_run_options',
    'list_charge_methods',
    'list_forcefields',
    'build_interface',
    'build_interface_from_workdirs',
    'load_from_moldb',
    'mol_from_smiles',
    'parameterize_smiles',
    'print_sandwich_result_summary',
    'qm',
    'run_options',
    'set_run_options',
]
