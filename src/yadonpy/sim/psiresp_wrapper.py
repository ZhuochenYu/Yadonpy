from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from ..core import utils
from ..core.polyelectrolyte import annotate_polyelectrolyte_metadata
from ..core.polyelectrolyte import _normalize_resp_profile
from ..core.polyelectrolyte import uses_localized_charge_groups
from ..diagnostics import _psiresp_hint

psiresp = None
_psiresp_import_error: Exception | None = None
try:  # pragma: no cover
    import psiresp  # type: ignore
except Exception as e:  # pragma: no cover
    _psiresp_import_error = e
    psiresp = None


def psiresp_available() -> bool:
    return psiresp is not None


def _require_psiresp() -> None:
    if psiresp is None:
        detail = f" (root cause: {_psiresp_import_error!r})" if _psiresp_import_error else ""
        raise ImportError(
            "RESP/ESP fitting now requires optional dependency 'psiresp'. "
            + _psiresp_hint(repr(_psiresp_import_error) if _psiresp_import_error else None)
            + detail
        )


def _ensure_psiresp_numpy_compat() -> None:
    if hasattr(np, "in1d"):
        return

    def _in1d(ar1, ar2, assume_unique=False, invert=False):
        return np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert)

    np.in1d = _in1d  # type: ignore[attr-defined]


def _make_psiresp_molecule(
    mol,
    *,
    charge: int,
    multiplicity: int,
):
    mol_copy = Chem.Mol(mol)
    pmol = psiresp.Molecule.from_rdkit(
        mol_copy,
        charge=charge,
        multiplicity=multiplicity,
        optimize_geometry=False,
        keep_original_orientation=True,
    )
    return mol_copy, pmol


def _charge_constraints_for_molecule(
    mol,
    *,
    pmol,
    polyelectrolyte_mode: bool,
    polyelectrolyte_detection: str,
    resp_profile: str,
) -> tuple[Any, dict[str, Any] | None]:
    profile = _normalize_resp_profile(resp_profile)
    annotated = annotate_polyelectrolyte_metadata(
        mol,
        detection=polyelectrolyte_detection,
        resp_profile=profile,
    )
    summary = annotated["summary"]
    constraints_meta = annotated["constraints"]
    options = psiresp.ChargeConstraintOptions(
        symmetric_methyls=False,
        symmetric_methylenes=False,
        symmetric_atoms_are_equivalent=False,
    )
    use_group_charge_constraints = bool(polyelectrolyte_mode) or (
        str(constraints_meta.get("mode") or "").strip().lower() == "grouped"
        or uses_localized_charge_groups(summary)
    )
    if use_group_charge_constraints:
        for grp in constraints_meta.get("charged_group_constraints", []):
            options.add_charge_sum_constraint_for_molecule(
                pmol,
                charge=float(grp["target_charge"]),
                indices=[int(i) for i in grp["atom_indices"]],
            )
        neutral = [int(i) for i in constraints_meta.get("neutral_remainder_indices", [])]
        if neutral:
            options.add_charge_sum_constraint_for_molecule(
                pmol,
                charge=float(constraints_meta.get("neutral_remainder_charge", 0.0)),
                indices=neutral,
            )
    for eq in constraints_meta.get("equivalence_groups", []):
        indices = [int(i) for i in eq]
        if len(indices) > 1:
            options.add_charge_equivalence_constraint_for_molecule(pmol, indices=indices)
    if not summary.get("groups"):
        summary["fallback"] = summary.get("fallback") or "whole_molecule_scale"
        constraints_meta["fallback"] = summary["fallback"]
    return options, {"summary": summary, "constraints": constraints_meta, "resp_profile": profile}


def _build_psiresp_job(
    *,
    pmol,
    fit_kind: str,
    constraints,
    run_dir: Path,
):
    job_cls = psiresp.TwoStageRESP if fit_kind == "RESP" else psiresp.ESP
    job = job_cls(
        molecules=[pmol],
        charge_constraints=constraints,
        working_directory=run_dir,
    )
    job.grid_options.use_radii = "msk"
    job.grid_options.vdw_scale_factors = [1.4, 1.6, 1.8, 2.0]
    job.grid_options.vdw_point_density = 20.0
    return job


def _default_pcm_input(*, solvent: str = "Water") -> str:
    return (
        f"\n"
        f"Units = Angstrom\n"
        f"Medium {{\n"
        f"    SolverType = IEFPCM\n"
        f"    Solvent = {str(solvent).strip() or 'Water'}\n"
        f"}}\n"
        f"Cavity {{\n"
        f"    RadiiSet = UFF\n"
        f"    Type = GePol\n"
        f"    Scaling = False\n"
        f"    Area = 0.3\n"
        f"    Mode = Implicit\n"
        f"}}\n"
    )


def _ensure_orientation_grid(orientation, *, grid_options) -> np.ndarray:
    if orientation.grid is None:
        orientation.compute_grid(grid_options=grid_options)
    return np.asarray(orientation.grid, dtype=float)


def _compute_orientation_esp_from_psi4(
    orientation,
    *,
    method: str,
    basis: str,
    run_dir: Path | None = None,
    ncores: int | None = None,
    memory_mib: int | float | None = None,
    pcm_solvent: str | None = None,
) -> np.ndarray:
    import psi4  # type: ignore
    from psiresp import psi4utils  # type: ignore

    if orientation.grid is None:
        raise ValueError("Orientation grid must be prepared before computing ESP.")
    grid = np.asarray(orientation.grid, dtype=float)

    try:
        psi4.core.clean()
        psi4.core.clean_options()
    except Exception:
        pass
    if ncores is not None:
        try:
            psi4.set_num_threads(max(1, int(ncores)))
        except Exception:
            pass
    if memory_mib is not None:
        try:
            psi4.set_memory(f"{max(int(float(memory_mib)), 100)} MB")
        except Exception:
            pass
    try:
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            psi4.core.set_output_file(str(run_dir / "psiresp_psi4_sp.log"), False)
    except Exception:
        pass

    pmol = psi4utils.psi4mol_from_qcmol(orientation.qcmol)
    try:
        pmol.reset_point_group("c1")
    except Exception:
        pass
    try:
        pmol.fix_orientation(True)
        pmol.fix_com(True)
    except Exception:
        pass
    try:
        pmol.update_geometry()
    except Exception:
        pass

    options: dict[str, Any] = {
        "basis": str(basis).strip(),
        "scf_type": "df",
        "fail_on_maxiter": True,
    }
    if pcm_solvent:
        options.update(
            {
                "pcm": True,
                "pcm_scf_type": "total",
                "pcm__input": _default_pcm_input(solvent=str(pcm_solvent)),
            }
        )
    psi4.set_options(options)
    _energy, wfn = psi4.energy(str(method).strip().lower(), molecule=pmol, return_wfn=True)
    esp_calc = psi4.core.ESPPropCalc(wfn)
    psi4grid = psi4.core.Matrix.from_array(grid)
    esp = np.asarray(esp_calc.compute_esp_over_grid_in_memory(psi4grid), dtype=float)
    try:
        psi4.core.clean()
        psi4.core.clean_options()
    except Exception:
        pass
    return esp


def _populate_orientation_with_precomputed_esp(
    orientation,
    *,
    grid_options,
    method: str,
    basis: str,
    run_dir: Path | None = None,
    ncores: int | None = None,
    memory_mib: int | float | None = None,
    pcm_solvent: str | None = None,
) -> None:
    grid = _ensure_orientation_grid(orientation, grid_options=grid_options)
    esp = _compute_orientation_esp_from_psi4(
        orientation,
        method=method,
        basis=basis,
        run_dir=run_dir,
        ncores=ncores,
        memory_mib=memory_mib,
        pcm_solvent=pcm_solvent,
    )
    orientation.grid = grid
    orientation.esp = esp


def _extract_psiresp_charge_arrays(*, fit_kind: str, pmol, job) -> tuple[np.ndarray, np.ndarray]:
    esp_charges = getattr(pmol, "stage_1_unrestrained_charges", None)
    final_resp = (
        getattr(pmol, "stage_2_restrained_charges", None)
        if str(fit_kind).strip().upper() == "RESP"
        else getattr(pmol, "stage_1_unrestrained_charges", None)
    )
    if final_resp is None:
        charges = job.charges
        if isinstance(charges, (list, tuple)):
            final_resp = np.asarray(charges[0], dtype=float)
        else:
            final_resp = np.asarray(charges, dtype=float)
    if esp_charges is None:
        esp_charges = np.asarray(final_resp, dtype=float)
    return np.asarray(final_resp, dtype=float), np.asarray(esp_charges, dtype=float)


def _run_precomputed_psiresp_job(
    *,
    mol,
    fit_kind: str,
    charge: int,
    multiplicity: int,
    run_dir: Path,
    method: str,
    basis: str,
    polyelectrolyte_mode: bool,
    polyelectrolyte_detection: str,
    resp_profile: str,
    ncores: int | None,
    memory_mib: int | float | None,
    pcm_solvent: str | None = None,
) -> dict[str, Any]:
    mol_copy, pmol = _make_psiresp_molecule(
        mol,
        charge=charge,
        multiplicity=multiplicity,
    )
    constraints, constraint_meta = _charge_constraints_for_molecule(
        mol_copy,
        pmol=pmol,
        polyelectrolyte_mode=bool(polyelectrolyte_mode),
        polyelectrolyte_detection=str(polyelectrolyte_detection or "auto"),
        resp_profile=str(resp_profile or "adaptive"),
    )
    if isinstance(constraint_meta, dict):
        constraint_meta["fit_kind"] = str(fit_kind).strip().upper()
    job = _build_psiresp_job(
        pmol=pmol,
        fit_kind=str(fit_kind).strip().upper(),
        constraints=constraints,
        run_dir=run_dir,
    )
    job.generate_orientations()
    for orientation in job.iter_orientations():
        if orientation.esp is None:
            _populate_orientation_with_precomputed_esp(
                orientation,
                grid_options=job.grid_options,
                method=str(method),
                basis=str(basis),
                run_dir=run_dir,
                ncores=ncores,
                memory_mib=memory_mib,
                pcm_solvent=pcm_solvent,
            )
    job.compute_charges(update_molecules=True)
    resp_arr, esp_arr = _extract_psiresp_charge_arrays(fit_kind=fit_kind, pmol=pmol, job=job)
    return {
        "resp": resp_arr,
        "esp": esp_arr,
        "constraint_meta": constraint_meta,
    }


def run_psiresp_fit(
    mol,
    *,
    fit_kind: str = "RESP",
    method: str = "hf",
    basis: str = "6-31g*",
    total_charge: int | None = None,
    total_multiplicity: int | None = None,
    work_dir: str | Path | None = None,
    name: str = "yadonpy",
    polyelectrolyte_mode: bool = False,
    polyelectrolyte_detection: str = "auto",
    resp_profile: str = "adaptive",
    ncores: int | None = None,
    memory_mib: int | float | None = None,
) -> dict[str, Any]:
    """Run PsiRESP fitting with Psi4-computed ESPs.

    This follows the PsiRESP examples where orientations/grids are prepared on
    the PsiRESP side, QM is executed externally, and only the precomputed ESPs
    are handed back to `compute_charges()`.
    """
    _require_psiresp()
    _ensure_psiresp_numpy_compat()
    fit_kind_up = str(fit_kind).strip().upper()
    profile = _normalize_resp_profile(resp_profile)
    if fit_kind_up not in {"RESP", "RESP2", "ESP"}:
        raise ValueError(f"Unsupported psiresp fit kind: {fit_kind}")

    charge = int(total_charge) if total_charge is not None else int(Chem.GetFormalCharge(mol))
    multiplicity = int(total_multiplicity) if total_multiplicity is not None else 1
    work_root = Path(work_dir or ".").expanduser().resolve()
    run_dir = work_root / "psiresp"
    run_dir.mkdir(parents=True, exist_ok=True)

    dt1 = datetime.datetime.now()
    utils.radon_print(
        f"PsiRESP {fit_kind_up} is running... name={name} charge={charge} mult={multiplicity} method={method} basis={basis}",
        level=1,
    )
    if fit_kind_up == "RESP2":
        gas = _run_precomputed_psiresp_job(
            mol=mol,
            fit_kind="RESP",
            charge=charge,
            multiplicity=multiplicity,
            run_dir=run_dir / "vacuum",
            method=str(method),
            basis=str(basis),
            polyelectrolyte_mode=bool(polyelectrolyte_mode),
            polyelectrolyte_detection=str(polyelectrolyte_detection or "auto"),
            resp_profile=profile,
            ncores=ncores,
            memory_mib=memory_mib,
            pcm_solvent=None,
        )
        solv = _run_precomputed_psiresp_job(
            mol=mol,
            fit_kind="RESP",
            charge=charge,
            multiplicity=multiplicity,
            run_dir=run_dir / "solvated_water",
            method=str(method),
            basis=str(basis),
            polyelectrolyte_mode=bool(polyelectrolyte_mode),
            polyelectrolyte_detection=str(polyelectrolyte_detection or "auto"),
            resp_profile=profile,
            ncores=ncores,
            memory_mib=memory_mib,
            pcm_solvent="Water",
        )
        resp2_arr = 0.6 * np.asarray(solv["resp"], dtype=float) + 0.4 * np.asarray(gas["resp"], dtype=float)
        esp_arr = np.asarray(gas["esp"], dtype=float)
        constraint_meta = gas.get("constraint_meta") or solv.get("constraint_meta")
    else:
        result = _run_precomputed_psiresp_job(
            mol=mol,
            fit_kind=fit_kind_up,
            charge=charge,
            multiplicity=multiplicity,
            run_dir=run_dir,
            method=str(method),
            basis=str(basis),
            polyelectrolyte_mode=bool(polyelectrolyte_mode),
            polyelectrolyte_detection=str(polyelectrolyte_detection or "auto"),
            resp_profile=profile,
            ncores=ncores,
            memory_mib=memory_mib,
            pcm_solvent=None,
        )
        resp_arr = np.asarray(result["resp"], dtype=float)
        esp_arr = np.asarray(result["esp"], dtype=float)
        constraint_meta = result.get("constraint_meta")
    dt2 = datetime.datetime.now()
    utils.radon_print(
        f"Normal termination of PsiRESP {fit_kind_up} charge calculation. Elapsed time = {str(dt2-dt1)}",
        level=1,
    )
    if fit_kind_up == "RESP2":
        return {
            "resp": np.asarray(resp2_arr, dtype=float),
            "resp2": np.asarray(resp2_arr, dtype=float),
            "resp_gas": np.asarray(gas["resp"], dtype=float),
            "resp_solvated": np.asarray(solv["resp"], dtype=float),
            "esp": esp_arr,
            "constraint_meta": constraint_meta,
            "working_directory": str(run_dir),
        }
    return {
        "resp": resp_arr,
        "esp": esp_arr,
        "constraint_meta": constraint_meta,
        "working_directory": str(run_dir),
    }
