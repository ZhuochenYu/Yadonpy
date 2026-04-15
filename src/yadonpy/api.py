"""High-level convenience API for script-friendly workflows."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from .ff.registry import available_forcefields, create_forcefield
from .core.charge_models import supported_quick_charge_methods


_PSI4_CHARGE_METHODS = ('RESP', 'ESP', 'Mulliken', 'Lowdin')
_BASELINE_CHARGE_METHODS = ('zero', 'gasteiger')


def get_ff(ff_name: str, **kwargs):
    """Return a force-field object by canonical name or alias."""
    return create_forcefield(ff_name, **kwargs)


def list_forcefields() -> tuple[str, ...]:
    """Return canonical force-field names supported by YadonPy."""
    return tuple(available_forcefields())


def list_charge_methods() -> tuple[str, ...]:
    """Return commonly supported charge-method tokens.

    Included entries cover built-in non-QM methods, Psi4-based methods, and the
    lightweight scaled forms understood by ``qm.assign_charges``.
    """
    return _BASELINE_CHARGE_METHODS + _PSI4_CHARGE_METHODS + supported_quick_charge_methods()


def mol_from_smiles(smiles: str, *, coord: bool = True, name: str | None = None):
    """Build an RDKit molecule from a SMILES string using YadonPy defaults."""
    from .core import utils

    return utils.mol_from_smiles(smiles, coord=coord, name=name)


def build_graphite(**kwargs):
    """Thin wrapper around :func:`yadonpy.core.graphite.build_graphite`."""
    from .core.graphite import build_graphite as _build_graphite

    return _build_graphite(**kwargs)


def build_graphite_polymer_electrolyte_sandwich(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.sandwich.build_graphite_polymer_electrolyte_sandwich`."""
    from .interface.sandwich import build_graphite_polymer_electrolyte_sandwich as _build_sandwich

    return _build_sandwich(**kwargs)


def prepare_graphite_substrate(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.prepare_graphite_substrate`."""
    from .interface import prepare_graphite_substrate as _prepare_graphite_substrate

    return _prepare_graphite_substrate(**kwargs)


def calibrate_polymer_bulk_phase(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.calibrate_polymer_bulk_phase`."""
    from .interface import calibrate_polymer_bulk_phase as _calibrate_polymer_bulk_phase

    return _calibrate_polymer_bulk_phase(**kwargs)


def calibrate_electrolyte_bulk_phase(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.calibrate_electrolyte_bulk_phase`."""
    from .interface import calibrate_electrolyte_bulk_phase as _calibrate_electrolyte_bulk_phase

    return _calibrate_electrolyte_bulk_phase(**kwargs)


def default_peo_polymer_spec(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.default_peo_polymer_spec`."""
    from .interface import default_peo_polymer_spec as _default_peo_polymer_spec

    return _default_peo_polymer_spec(**kwargs)


def default_peo_electrolyte_spec(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.default_peo_electrolyte_spec`."""
    from .interface import default_peo_electrolyte_spec as _default_peo_electrolyte_spec

    return _default_peo_electrolyte_spec(**kwargs)


def default_cmcna_polymer_spec(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.default_cmcna_polymer_spec`."""
    from .interface import default_cmcna_polymer_spec as _default_cmcna_polymer_spec

    return _default_cmcna_polymer_spec(**kwargs)


def default_carbonate_lipf6_electrolyte_spec(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.default_carbonate_lipf6_electrolyte_spec`."""
    from .interface import default_carbonate_lipf6_electrolyte_spec as _default_carbonate_lipf6_electrolyte_spec

    return _default_carbonate_lipf6_electrolyte_spec(**kwargs)


def build_graphite_polymer_interphase(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.build_graphite_polymer_interphase`."""
    from .interface import build_graphite_polymer_interphase as _build_graphite_polymer_interphase

    return _build_graphite_polymer_interphase(**kwargs)


def build_graphite_cmc_interphase(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.build_graphite_cmc_interphase`."""
    from .interface import build_graphite_cmc_interphase as _build_graphite_cmc_interphase

    return _build_graphite_cmc_interphase(**kwargs)


def build_polymer_electrolyte_interphase(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.build_polymer_electrolyte_interphase`."""
    from .interface import build_polymer_electrolyte_interphase as _build_polymer_electrolyte_interphase

    return _build_polymer_electrolyte_interphase(**kwargs)


def build_cmc_electrolyte_interphase(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.build_cmc_electrolyte_interphase`."""
    from .interface import build_cmc_electrolyte_interphase as _build_cmc_electrolyte_interphase

    return _build_cmc_electrolyte_interphase(**kwargs)


def release_graphite_polymer_electrolyte_stack(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.release_graphite_polymer_electrolyte_stack`."""
    from .interface import release_graphite_polymer_electrolyte_stack as _release_graphite_polymer_electrolyte_stack

    return _release_graphite_polymer_electrolyte_stack(**kwargs)


def release_graphite_cmc_electrolyte_stack(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.release_graphite_cmc_electrolyte_stack`."""
    from .interface import release_graphite_cmc_electrolyte_stack as _release_graphite_cmc_electrolyte_stack

    return _release_graphite_cmc_electrolyte_stack(**kwargs)


def analyze_interface_transport(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.analyze_interface_transport`."""
    from .interface import analyze_interface_transport as _analyze_interface_transport

    return _analyze_interface_transport(**kwargs)


def print_interface_result_summary(*args, **kwargs):
    """Thin wrapper around :func:`yadonpy.interface.print_interface_result_summary`."""
    from .interface import print_interface_result_summary as _print_interface_result_summary

    return _print_interface_result_summary(*args, **kwargs)


def build_graphite_cmcna_electrolyte_sandwich(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.sandwich.build_graphite_cmcna_electrolyte_sandwich`."""
    from .interface.sandwich import build_graphite_cmcna_electrolyte_sandwich as _build_sandwich

    return _build_sandwich(**kwargs)


def build_graphite_cmcna_glucose6_periodic_case(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.sandwich.build_graphite_cmcna_glucose6_periodic_case`."""
    from .interface.sandwich import build_graphite_cmcna_glucose6_periodic_case as _build_sandwich

    return _build_sandwich(**kwargs)


def build_graphite_peo_example_case(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.build_graphite_peo_example_case`."""
    from .interface import build_graphite_peo_example_case as _build_sandwich

    return _build_sandwich(**kwargs)


def build_graphite_cmcna_example_case(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.build_graphite_cmcna_example_case`."""
    from .interface import build_graphite_cmcna_example_case as _build_sandwich

    return _build_sandwich(**kwargs)


def audit_default_moldb_sync(**kwargs):
    """Audit the active user MolDB against the bundled default catalog."""
    from .core.data_dir import audit_active_bundle_sync as _audit_bundle

    return _audit_bundle(**kwargs)


def build_graphite_peo_electrolyte_sandwich(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.sandwich.build_graphite_peo_electrolyte_sandwich`."""
    from .interface.sandwich import build_graphite_peo_electrolyte_sandwich as _build_sandwich

    return _build_sandwich(**kwargs)


def resolve_prepared_system(**kwargs):
    """Resolve a prepared GROMACS system from explicit paths or a workflow work_dir."""
    from .workflow import resolve_prepared_system as _resolve_prepared_system

    return _resolve_prepared_system(**kwargs)


def run_tg_scan_gmx(**kwargs):
    """High-level Tg study wrapper around the GROMACS Tg workflow."""
    from .workflow import run_tg_scan_gmx as _run_tg_scan_gmx

    return _run_tg_scan_gmx(**kwargs)


def run_elongation_gmx(**kwargs):
    """High-level elongation study wrapper around the GROMACS deform workflow."""
    from .workflow import run_elongation_gmx as _run_elongation_gmx

    return _run_elongation_gmx(**kwargs)


def format_mechanics_result_summary(result) -> tuple[str, ...]:
    """Format a compact summary for a Tg or elongation study result."""
    from .workflow import format_mechanics_result_summary as _format_mechanics_result_summary

    return _format_mechanics_result_summary(result)


def print_mechanics_result_summary(result) -> None:
    """Print a compact summary for a Tg or elongation study result."""
    from .workflow import print_mechanics_result_summary as _print_mechanics_result_summary

    return _print_mechanics_result_summary(result)


def conformation_search(mol, **kwargs):
    """Thin wrapper around :func:`yadonpy.sim.qm.conformation_search`."""
    from .sim import qm

    return qm.conformation_search(mol, **kwargs)


def assign_charges(mol, *, charge: str = 'RESP', **kwargs):
    """Thin wrapper around :func:`yadonpy.sim.qm.assign_charges`."""
    from .sim import qm

    return qm.assign_charges(mol, charge=charge, **kwargs)


def assign_forcefield(mol, *, ff_name: str = 'gaff2_mod', charge: str | None = None, **kwargs):
    """Instantiate a force field and call its ``ff_assign`` method."""
    ff = get_ff(ff_name)
    ok = bool(ff.ff_assign(mol, charge=charge, **kwargs))
    return ff, ok


def load_from_moldb(
    smiles: str,
    *,
    charge: str = 'RESP',
    basis_set: str | None = None,
    method: str | None = None,
    require_ready: bool = True,
    return_record: bool = False,
    polyelectrolyte_mode: bool | None = None,
    polyelectrolyte_detection: str | None = None,
):
    """Load a molecule from MolDB using the requested charge variant.

    Parameters
    ----------
    smiles:
        SMILES or PSMILES lookup key.
    charge, basis_set, method:
        Requested charge variant. If the MolDB entry also stores a DRIH or
        mseminario bonded patch for the same variant, it is restored onto the
        returned molecule automatically.
    require_ready:
        If ``True``, require the variant to be marked ready in MolDB.
    return_record:
        If ``False`` (default), return only the RDKit molecule. If ``True``,
        return ``(mol, record)`` where ``record`` is the underlying
        :class:`yadonpy.moldb.store.MolRecord`.
    """
    from .moldb import MolDB

    db = MolDB()
    mol, record = db.load_mol(
        smiles,
        require_ready=require_ready,
        charge=charge,
        basis_set=basis_set,
        method=method,
        polyelectrolyte_mode=polyelectrolyte_mode,
        polyelectrolyte_detection=polyelectrolyte_detection,
    )
    if return_record:
        return mol, record
    return mol


def parameterize_smiles(
    smiles: str,
    *,
    ff_name: str = 'gaff2_mod',
    charge_method: str = 'RESP',
    work_dir: str = './',
    total_charge: Optional[int] = None,
    total_multiplicity: Optional[int] = None,
    name: str | None = None,
    allow_ff_without_requested_charges: bool = False,
    polyelectrolyte_mode: bool = False,
    polyelectrolyte_detection: str = 'auto',
):
    """Script-friendly helper: SMILES -> charges -> ``ff_assign``.

    Notes:
      - This function does **not** cache to any deprecated topology library.
      - For reuse, write the result to MolDB explicitly.
      - By default, a requested charge-assignment failure is treated as fatal so
        callers do not silently continue with an incompletely prepared molecule.
        Set ``allow_ff_without_requested_charges=True`` to opt into the legacy
        warning-and-continue behavior explicitly.
    """
    from .sim import qm

    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)

    ff = get_ff(ff_name)
    mol = mol_from_smiles(smiles, coord=True, name=name)
    try:
        qm.assign_charges(
            mol,
            charge=charge_method,
            opt=False,
            work_dir=str(wd),
            total_charge=total_charge,
            total_multiplicity=total_multiplicity,
            polyelectrolyte_mode=polyelectrolyte_mode,
            polyelectrolyte_detection=polyelectrolyte_detection,
        )
    except Exception as exc:
        msg = (
            f"Charge assignment failed for {smiles!r} with method {charge_method!r}: {exc}"
        )
        if not bool(allow_ff_without_requested_charges):
            raise RuntimeError(msg) from exc
        warnings.warn(
            msg + '; continuing with FF assignment only because allow_ff_without_requested_charges=True',
            RuntimeWarning,
            stacklevel=2,
        )
    ok = bool(ff.ff_assign(mol))
    return mol, ok


__all__ = [
    'audit_default_moldb_sync',
    'analyze_interface_transport',
    'assign_charges',
    'assign_forcefield',
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
    'calibrate_electrolyte_bulk_phase',
    'calibrate_polymer_bulk_phase',
    'conformation_search',
    'default_carbonate_lipf6_electrolyte_spec',
    'default_cmcna_polymer_spec',
    'default_peo_electrolyte_spec',
    'default_peo_polymer_spec',
    'format_mechanics_result_summary',
    'get_ff',
    'list_charge_methods',
    'list_forcefields',
    'load_from_moldb',
    'mol_from_smiles',
    'parameterize_smiles',
    'prepare_graphite_substrate',
    'print_interface_result_summary',
    'print_mechanics_result_summary',
    'release_graphite_cmc_electrolyte_stack',
    'release_graphite_polymer_electrolyte_stack',
    'resolve_prepared_system',
    'run_elongation_gmx',
    'run_tg_scan_gmx',
]
