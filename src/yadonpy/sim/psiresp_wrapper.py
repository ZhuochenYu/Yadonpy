from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from ..core import utils
from ..core.polyelectrolyte import annotate_polyelectrolyte_metadata

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
            "Install e.g. via: conda install -c conda-forge psiresp && conda install -c psi4 psi4"
            + detail
        )


def _charge_constraints_for_molecule(
    mol,
    *,
    pmol,
    polyelectrolyte_mode: bool,
    polyelectrolyte_detection: str,
) -> tuple[Any, dict[str, Any] | None]:
    if not polyelectrolyte_mode:
        return psiresp.ChargeConstraintOptions(), None

    annotated = annotate_polyelectrolyte_metadata(mol, detection=polyelectrolyte_detection)
    summary = annotated["summary"]
    constraints_meta = annotated["constraints"]
    options = psiresp.ChargeConstraintOptions(
        symmetric_methyls=False,
        symmetric_methylenes=False,
        symmetric_atoms_are_equivalent=False,
    )
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
    return options, {"summary": summary, "constraints": constraints_meta}


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
) -> dict[str, Any]:
    _require_psiresp()
    fit_kind_up = str(fit_kind).strip().upper()
    if fit_kind_up not in {"RESP", "ESP"}:
        raise ValueError(f"Unsupported psiresp fit kind: {fit_kind}")

    charge = int(total_charge) if total_charge is not None else int(Chem.GetFormalCharge(mol))
    multiplicity = int(total_multiplicity) if total_multiplicity is not None else 1
    work_root = Path(work_dir or ".").expanduser().resolve()
    run_dir = work_root / "psiresp"
    run_dir.mkdir(parents=True, exist_ok=True)

    mol_copy = Chem.Mol(mol)
    pmol = psiresp.Molecule.from_rdkit(
        mol_copy,
        charge=charge,
        multiplicity=multiplicity,
        optimize_geometry=False,
        keep_original_orientation=True,
    )
    constraints, constraint_meta = _charge_constraints_for_molecule(
        mol_copy,
        pmol=pmol,
        polyelectrolyte_mode=bool(polyelectrolyte_mode),
        polyelectrolyte_detection=str(polyelectrolyte_detection or "auto"),
    )
    job_cls = psiresp.TwoStageRESP if fit_kind_up == "RESP" else psiresp.ESP
    job = job_cls(
        molecules=[pmol],
        charge_constraints=constraints,
        working_directory=run_dir,
    )
    job.qm_optimization_options.method = str(method)
    job.qm_optimization_options.basis = str(basis)
    job.qm_esp_options.method = str(method)
    job.qm_esp_options.basis = str(basis)
    job.grid_options.use_radii = "msk"
    job.grid_options.vdw_scale_factors = [1.4, 1.6, 1.8, 2.0]
    job.grid_options.vdw_point_density = 20.0

    dt1 = datetime.datetime.now()
    utils.radon_print(
        f"PsiRESP {fit_kind_up} is running... name={name} charge={charge} mult={multiplicity} method={method} basis={basis}",
        level=1,
    )
    job.generate_orientations()
    job.compute_esps_and_charges(update_molecules=True)
    dt2 = datetime.datetime.now()
    utils.radon_print(
        f"Normal termination of PsiRESP {fit_kind_up} charge calculation. Elapsed time = {str(dt2-dt1)}",
        level=1,
    )

    # PsiRESP keeps multiple charge arrays. For YadonPy we expose:
    #   ESP  -> unrestrained first-stage fit
    #   RESP -> final restrained fit (or ESP charges if no stage-2 exists)
    esp_charges = getattr(pmol, "stage_1_unrestrained_charges", None)
    final_resp = (
        getattr(pmol, "stage_2_restrained_charges", None)
        if fit_kind_up == "RESP"
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

    resp_arr = np.asarray(final_resp, dtype=float)
    esp_arr = np.asarray(esp_charges, dtype=float)
    return {
        "resp": resp_arr,
        "esp": esp_arr,
        "constraint_meta": constraint_meta,
        "working_directory": str(run_dir),
    }
