"""High-level convenience API for script-friendly workflows."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

from .ff.registry import available_forcefields, create_forcefield
from .core.charge_models import supported_quick_charge_methods


_PSI4_CHARGE_METHODS = ('RESP', 'RESP2', 'ESP', 'Mulliken', 'Lowdin')
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


def build_layer_stack(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.build_layer_stack`."""
    from .interface import build_layer_stack as _build_layer_stack

    return _build_layer_stack(**kwargs)


def analyze_layer_stack_interface(**kwargs):
    """Thin wrapper around :func:`yadonpy.interface.analyze_layer_stack_interface`."""
    from .interface import analyze_layer_stack_interface as _analyze_layer_stack_interface

    return _analyze_layer_stack_interface(**kwargs)


def run_layer_stack_nvt(*args, **kwargs):
    """Thin wrapper around :func:`yadonpy.interface.run_layer_stack_nvt`."""
    from .interface import run_layer_stack_nvt as _run_layer_stack_nvt

    return _run_layer_stack_nvt(*args, **kwargs)


def clean_md_trajectory_files(*args, **kwargs):
    """Remove large trajectory streams after analysis has finished."""
    from .sim.cleanup import clean_md_trajectory_files as _clean_md_trajectory_files

    return _clean_md_trajectory_files(*args, **kwargs)


def audit_default_moldb_sync(**kwargs):
    """Audit the active user MolDB against the bundled default catalog."""
    from .core.data_dir import audit_active_bundle_sync as _audit_bundle

    return _audit_bundle(**kwargs)


def audit_oplsaa_reference(**kwargs):
    """Audit the current OPLS-AA implementation against GROMACS and Moltemplate references."""
    from .ff.oplsaa_reference import audit_oplsaa_reference as _audit_oplsaa_reference

    return _audit_oplsaa_reference(**kwargs)


def audit_bundled_oplsaa_parameter_sanity():
    """Check the bundled OPLS-AA table for common unit-conversion regressions."""
    from .ff.oplsaa_reference import audit_bundled_oplsaa_parameter_sanity as _audit_sanity

    return _audit_sanity()


def audit_oplsaa_assignment(mol, **kwargs):
    """Audit one OPLS-AA assigned molecule for missing terms and local refinements."""
    from .ff.oplsaa_reference import audit_oplsaa_assignment as _audit_oplsaa_assignment

    return _audit_oplsaa_assignment(mol, **kwargs)


def oplsaa_stability_preflight(**kwargs):
    """Run the short GROMACS OPLS-AA 1 fs/2 fs stability preflight matrix."""
    from .gmx.workflows.eq import oplsaa_stability_preflight as _oplsaa_stability_preflight

    return _oplsaa_stability_preflight(**kwargs)


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


def assign_charges(mol, *, charge: str = 'RESP', resp_profile: str = 'adaptive', **kwargs):
    """Thin wrapper around :func:`yadonpy.sim.qm.assign_charges`."""
    from .sim import qm

    return qm.assign_charges(mol, charge=charge, resp_profile=resp_profile, **kwargs)


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
    resp_profile: str | None = None,
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
        resp_profile=resp_profile,
    )
    if return_record:
        return mol, record
    return mol


def parameterize_smiles(
    smiles: str,
    *,
    ff_name: str = 'gaff2_mod',
    charge_method: str = 'RESP',
    resp_profile: str = 'adaptive',
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
            resp_profile=resp_profile,
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
    'audit_bundled_oplsaa_parameter_sanity',
    'audit_oplsaa_assignment',
    'audit_oplsaa_reference',
    'analyze_layer_stack_interface',
    'assign_charges',
    'assign_forcefield',
    'build_graphite',
    'build_layer_stack',
    'clean_md_trajectory_files',
    'conformation_search',
    'format_mechanics_result_summary',
    'get_ff',
    'list_charge_methods',
    'list_forcefields',
    'load_from_moldb',
    'mol_from_smiles',
    'oplsaa_stability_preflight',
    'parameterize_smiles',
    'print_mechanics_result_summary',
    'resolve_prepared_system',
    'run_layer_stack_nvt',
    'run_elongation_gmx',
    'run_tg_scan_gmx',
]
