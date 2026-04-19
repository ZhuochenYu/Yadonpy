from __future__ import annotations

"""CMC random copolymer electrolyte example with a MoldDB-backed additive.

The anionic glucose monomers are expected to be RESP-ready in MolDB with
``polyelectrolyte_mode=True``. This keeps the expensive monomer QM step out of
the system-build script and directly exercises the MolDB -> ITP export path.

Use ``YADONPY_BUILD_ONLY=1`` to stop after amorphous-cell construction.
Use ``YADONPY_EXPORT_ONLY=1`` to stop after exporting ``02_system``.
These modes are useful for checking topology / ITP generation on machines
without GROMACS.

Remote mixed-system debug ladder:
- set a unique ``YADONPY_WORK_DIR`` for every run
- start with ``YADONPY_EXPORT_ONLY=1``
- then ``YADONPY_EQ21_STAGE_CAP=2`` for ``EM + preNVT``
- then a short production run on GPU
- repeat the short production run with ``YADONPY_GPU=0`` if the failure needs
  to be classified as topology/physics vs runtime/environment
"""

import json
import os
from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import utils, poly, workdir
from yadonpy.core.polymer_audit import audit_polymer_state, compare_exported_charge_groups, write_polymer_audit
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2, GAFF2_mod, MERZ
from yadonpy.sim.analyzer import AnalyzeResult
from yadonpy.sim import qm
from yadonpy.sim.preset import eq


def _env_flag(name: str, default: bool = False) -> bool:
    token = str(os.environ.get(name, "")).strip().lower()
    if not token:
        return bool(default)
    return token in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    return float(raw)


def _env_optional_float(name: str) -> float | None:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    return float(raw)


def _env_int_list(name: str, expected_len: int) -> list[int] | None:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    vals = [int(tok.strip()) for tok in raw.split(",") if str(tok).strip()]
    if len(vals) != int(expected_len):
        raise ValueError(f"{name} expects {expected_len} comma-separated integers, got {len(vals)}")
    return vals


def _env_text(name: str, default: str) -> str:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return str(default)
    return raw


def _load_ready_from_moldb(
    ff,
    smiles: str,
    *,
    label: str,
    bonded: str | None = None,
    polyelectrolyte_mode: bool = False,
    repo_db_dir: Path,
):
    last_exc: Exception | None = None
    for db_dir, db_label in ((None, "default"), (repo_db_dir, "repo")):
        try:
            mol = ff.mol_rdkit(
                smiles,
                name=label,
                db_dir=db_dir,
                charge="RESP",
                require_ready=True,
                prefer_db=True,
                polyelectrolyte_mode=polyelectrolyte_mode,
                polyelectrolyte_detection="auto",
            )
            mol = ff.ff_assign(
                mol,
                bonded=bonded,
                polyelectrolyte_mode=polyelectrolyte_mode,
            )
            if not mol:
                raise RuntimeError(f"Cannot assign force field parameters for {label}.")
            print(f"[MolDB] loaded {label} from {db_label} db")
            return mol
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"{label} is expected to be RESP-ready in MolDB. "
        "Please refresh it with examples/07_moldb_precompute_and_reuse/"
        "03_refresh_cmc_glucose_polyelectrolytes.py."
    ) from last_exc


# ---------------- user inputs ----------------
restart_status = True
build_only = _env_flag("YADONPY_BUILD_ONLY", default=False)
export_only = _env_flag("YADONPY_EXPORT_ONLY", default=False)
analysis_only = _env_flag("YADONPY_ANALYSIS_ONLY", default=False)
smoke_mode = _env_flag("YADONPY_SMOKE", default=False)
fast_analysis = _env_flag("YADONPY_FAST_ANALYSIS", default=False)

skip_rdf = _env_flag("YADONPY_SKIP_RDF", default=fast_analysis)
skip_den_dis = _env_flag("YADONPY_SKIP_DEN_DIS", default=fast_analysis)
skip_sigma = _env_flag("YADONPY_SKIP_SIGMA", default=fast_analysis)

set_run_options(restart=restart_status)

ff = GAFF2_mod() if _env_text("YADONPY_FORCEFIELD", "GAFF2_MOD").strip().upper() == "GAFF2_MOD" else GAFF2()
ion_ff = MERZ()

# ---- CMC monomers (two connection points '*...*') ----
glucose_smiles = "*OC1OC(CO)C(*)C(O)C1O"
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"

DTD_smiles = "O=S1(=O)OC=CO1"
VC_smiles = "O=c1occo1"

feed_ratio = [12, 26, 27, 35]
feed_prob = poly.ratio_to_prob(feed_ratio)

target_mw = 10000.0
ter_smiles = "[H][*]"

# ---- Solvents ----
EC_smiles = "O=C1OCCO1"
EMC_smiles = "CCOC(=O)OC"
DEC_smiles = "CCOC(=O)OCC"

# ---- Salt / ions ----
Li_smiles = "[Li+]"
PF6_smiles = "F[P-](F)(F)(F)(F)F"
Na_smiles = "[Na+]"

temp = 318.15
press = 1.0
mpi = _env_int("YADONPY_MPI", 1)
omp = _env_int("YADONPY_OMP", 14)
gpu = _env_int("YADONPY_GPU", 1)
gpu_id = _env_int("YADONPY_GPU_ID", 0)
prod_ns = _env_float("YADONPY_PROD_NS", 20.0)
prod_traj_ps = _env_float("YADONPY_PROD_TRAJ_PS", 2.0)
prod_energy_ps = _env_float("YADONPY_PROD_ENERGY_PS", 2.0)
prod_log_ps = _env_optional_float("YADONPY_PROD_LOG_PS")
prod_trr_ps = _env_optional_float("YADONPY_PROD_TRR_PS")
prod_velocity_ps = _env_optional_float("YADONPY_PROD_VELOCITY_PS")
prod_checkpoint_min = _env_float("YADONPY_PROD_CPT_MIN", 5.0)
msd_drift = _env_text("YADONPY_MSD_DRIFT", "off")
msd_compare_drift_off = _env_flag(
    "YADONPY_COMPARE_DRIFT_OFF",
    default=(str(msd_drift).strip().lower() != "off" and not smoke_mode),
)
additive_name = _env_text("YADONPY_ADDITIVE", "DTD").strip().upper()
forcefield_name = _env_text("YADONPY_FORCEFIELD", "GAFF2_MOD").strip().upper()
system_variant = _env_text("YADONPY_SYSTEM_VARIANT", "full").strip().lower()
prod_ensemble = _env_text("YADONPY_PROD_ENSEMBLE", "npt").strip().lower()
gpu_offload_mode = _env_text("YADONPY_GPU_OFFLOAD_MODE", "conservative").strip().lower()
prod_bridge_ps = _env_float("YADONPY_PROD_BRIDGE_PS", 100.0)
prod_bridge_dt_fs = _env_float("YADONPY_PROD_BRIDGE_DT_FS", 1.0)
prod_bridge_lincs_iter = _env_int("YADONPY_PROD_BRIDGE_LINCS_ITER", 4)
prod_bridge_lincs_order = _env_int("YADONPY_PROD_BRIDGE_LINCS_ORDER", 12)
nvt_density_control = _env_flag("YADONPY_NVT_DENSITY_CONTROL", default=False)
counts_override = _env_int_list("YADONPY_COUNTS", 8)
eq21_final_ns = _env_float("YADONPY_EQ21_FINAL_NS", 0.8)
eq21_npt_time_scale = _env_float("YADONPY_EQ21_NPT_TIME_SCALE", 2.0)
eq21_stage_cap = _env_int("YADONPY_EQ21_STAGE_CAP", 0)
additional_ns = _env_float("YADONPY_ADDITIONAL_NS", 1.0)
additional_rounds = _env_int("YADONPY_ADDITIONAL_MAX_ROUNDS", 4)

omp_psi4 = _env_int("YADONPY_PSI4_OMP", 20)
mem_mb = _env_int("YADONPY_PSI4_MEMORY_MB", 20000)

BASE_DIR = Path(__file__).resolve().parent
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"
_work_dir_override = str(os.environ.get("YADONPY_WORK_DIR", "")).strip()
_shared_polymer_root_override = str(os.environ.get("YADONPY_SHARED_POLYMER_ROOT", "")).strip()
_ADDITIVE_LIBRARY = {
    "DTD": {"label": "DTD", "smiles": DTD_smiles, "default_count": 4, "smoke_count": 1},
    "VC": {"label": "VC", "smiles": VC_smiles, "default_count": 4, "smoke_count": 1},
}
if additive_name not in _ADDITIVE_LIBRARY:
    raise ValueError(f"Unsupported YADONPY_ADDITIVE={additive_name!r}; expected one of {sorted(_ADDITIVE_LIBRARY)}")
if forcefield_name not in {"GAFF2", "GAFF2_MOD"}:
    raise ValueError("YADONPY_FORCEFIELD must be GAFF2 or GAFF2_MOD")
if system_variant not in {"full", "electrolyte_additive", "polymer_ions", "polymer_solvents"}:
    raise ValueError("YADONPY_SYSTEM_VARIANT must be full, electrolyte_additive, polymer_ions, or polymer_solvents")
if prod_ensemble not in {"npt", "nvt"}:
    raise ValueError("YADONPY_PROD_ENSEMBLE must be npt or nvt")
if gpu_offload_mode not in {"full", "conservative", "cpu"}:
    raise ValueError("YADONPY_GPU_OFFLOAD_MODE must be full, conservative, or cpu")
ADDITIVE = dict(_ADDITIVE_LIBRARY[additive_name])
transport_labels = [str(ADDITIVE["label"]), "EMC", "DEC", "EC"]
work_dir = (
    Path(_work_dir_override).expanduser()
    if _work_dir_override
    else (BASE_DIR / f"work_dir_{str(ADDITIVE['label']).lower()}_moldb")
)
shared_polymer_root = (
    Path(_shared_polymer_root_override).expanduser()
    if _shared_polymer_root_override
    else None
)


def _formal_charge(mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def _write_polymer_checkpoint_audit(audit_dir: Path, label: str, mol) -> Path:
    return write_polymer_audit(
        audit_polymer_state(mol, label=label, radius=2),
        audit_dir / f"{label}.json",
    )


def _formulation_counts(
    *,
    smoke: bool,
    additive_default_count: int,
    n_na: int,
    variant: str,
    override: list[int] | None,
) -> list[int]:
    if override is not None:
        return list(override)
    if smoke:
        base = [1, 6, 6, 6, 2, 2, n_na, int(ADDITIVE["smoke_count"])]
    else:
        base = [8, 40, 50, 20, 10, 10, n_na, int(additive_default_count)]
    if variant == "full":
        return base
    if variant == "electrolyte_additive":
        return [0, base[1], base[2], base[3], base[4], base[5], 0, base[7]]
    if variant == "polymer_ions":
        return [base[0], 0, 0, 0, base[4], base[5], base[6], 0]
    if variant == "polymer_solvents":
        return [base[0], base[1], base[2], base[3], base[4], base[5], base[6], 0]
    raise ValueError(f"Unsupported system variant: {variant}")


def _default_msd_begin_ps(prod_time_ns: float) -> float:
    total_ps = max(0.0, float(prod_time_ns) * 1000.0)
    return min(5000.0, 0.25 * total_ps)


def _extract_species_metric(msd_payload: dict, moltype: str) -> dict:
    rec = dict(msd_payload.get(str(moltype)) or {})
    metric_name = str(rec.get("default_metric") or rec.get("metric") or "")
    metrics = dict(rec.get("metrics") or {})
    metric = dict(metrics.get(metric_name) or {})
    return {
        "moltype": str(moltype),
        "n_molecules": int(rec.get("n_molecules") or 0),
        "metric": metric_name,
        "n_groups": int(metric.get("n_groups") or 0),
        "D_m2_s": metric.get("D_m2_s"),
        "confidence": metric.get("confidence"),
        "status": metric.get("status"),
        "warning": metric.get("warning"),
        "alpha_mean": metric.get("alpha_mean"),
        "fit_t_start_ps": metric.get("fit_t_start_ps"),
        "fit_t_end_ps": metric.get("fit_t_end_ps"),
        "preprocessing": dict(metric.get("preprocessing") or {}),
    }


def _rank_transport_species(msd_payload: dict, moltypes: list[str]) -> list[dict]:
    rows = [_extract_species_metric(msd_payload, moltype) for moltype in moltypes]
    rows = [row for row in rows if row.get("D_m2_s") is not None]
    rows.sort(key=lambda row: float(row["D_m2_s"]), reverse=True)
    return rows


def _write_transport_diagnostics(
    analysis_dir: Path,
    *,
    moltypes: list[str],
    primary_msd: dict,
    primary_label: str,
    secondary_msd: dict | None = None,
    secondary_label: str | None = None,
    begin_ps: float | None,
    end_ps: float | None,
) -> Path:
    primary_rows = [_extract_species_metric(primary_msd, moltype) for moltype in moltypes]
    payload: dict[str, object] = {
        "primary": {
            "label": str(primary_label),
            "begin_ps": begin_ps,
            "end_ps": end_ps,
            "species": primary_rows,
            "ranking": _rank_transport_species(primary_msd, moltypes),
        }
    }
    if secondary_msd is not None and secondary_label:
        secondary_rows = [_extract_species_metric(secondary_msd, moltype) for moltype in moltypes]
        sensitivity = []
        for primary_row, secondary_row in zip(primary_rows, secondary_rows):
            p = primary_row.get("D_m2_s")
            s = secondary_row.get("D_m2_s")
            delta = None
            ratio = None
            if p is not None and s is not None:
                delta = float(s) - float(p)
                if abs(float(p)) > 1.0e-30:
                    ratio = float(s) / float(p)
            sensitivity.append(
                {
                    "moltype": primary_row["moltype"],
                    "primary_D_m2_s": p,
                    "secondary_D_m2_s": s,
                    "delta_D_m2_s": delta,
                    "ratio_secondary_to_primary": ratio,
                    "primary_confidence": primary_row.get("confidence"),
                    "secondary_confidence": secondary_row.get("confidence"),
                }
            )
        payload["secondary"] = {
            "label": str(secondary_label),
            "species": secondary_rows,
            "ranking": _rank_transport_species(secondary_msd, moltypes),
            "sensitivity": sensitivity,
        }

    out = analysis_dir / "transport_diagnostics.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def _print_transport_summary(*, msd_payload: dict, label: str, moltypes: list[str]) -> None:
    ranking = _rank_transport_species(msd_payload, moltypes)
    if not ranking:
        print(f"[TRANSPORT] {label}: no diffusion coefficients available")
        return
    print(f"[TRANSPORT] {label} ranking")
    for idx, row in enumerate(ranking, start=1):
        d_val = float(row["D_m2_s"]) if row.get("D_m2_s") is not None else float("nan")
        print(
            f"  {idx}. {row['moltype']:<4} D={d_val:.3e} m^2/s "
            f"confidence={row.get('confidence')} status={row.get('status')} "
            f"n_groups={row.get('n_groups')}"
        )


def _warn_if_transport_is_fragile(primary_msd: dict, *, additive_label: str, moltypes: list[str], secondary_msd: dict | None = None) -> None:
    additive_row = _extract_species_metric(primary_msd, additive_label)
    if not additive_row:
        return
    if int(additive_row.get("n_groups") or 0) <= 4:
        print(
            f"[TRANSPORT][WARNING] {additive_label} diffusion is being estimated from 4 or fewer molecular COM groups; "
            "treat ranking against bulk solvents as low-statistics."
        )
    if str(additive_row.get("confidence") or "").lower() not in {"high", "medium"}:
        print(
            f"[TRANSPORT][WARNING] {additive_label} diffusion fit is not high-confidence; "
            "check msd.json / transport_diagnostics.json before trusting solvent-order conclusions."
        )
    if secondary_msd is None:
        return

    primary_rank = [row["moltype"] for row in _rank_transport_species(primary_msd, moltypes)]
    secondary_rank = [row["moltype"] for row in _rank_transport_species(secondary_msd, moltypes)]
    if primary_rank and secondary_rank and primary_rank != secondary_rank:
        print(
            "[TRANSPORT][WARNING] Solvent/additive ranking changes when drift correction is toggled; "
            "treat this run as drift-sensitive and prefer longer sampling or tail-window reanalysis."
        )

    for moltype in moltypes:
        p = _extract_species_metric(primary_msd, moltype)
        s = _extract_species_metric(secondary_msd, moltype)
        p_d = p.get("D_m2_s")
        s_d = s.get("D_m2_s")
        if p_d is None or s_d is None:
            continue
        p_d = float(p_d)
        s_d = float(s_d)
        if abs(p_d) <= 1.0e-30:
            continue
        ratio = abs(s_d / p_d)
        if ratio >= 5.0 or ratio <= 0.2:
            print(
                f"[TRANSPORT][WARNING] {moltype} diffusion changes by more than 5x when drift correction is toggled "
                f"(ratio={ratio:.3g}); inspect transport_diagnostics.json before comparing solvent order."
            )


def _transport_probe_mols(additive_smiles: str):
    return [
        utils.mol_from_smiles(additive_smiles),
        utils.mol_from_smiles(EMC_smiles),
        utils.mol_from_smiles(DEC_smiles),
        utils.mol_from_smiles(EC_smiles),
    ]


def _extract_cmd_from_exception(message: str) -> str | None:
    for line in str(message).splitlines():
        if line.strip().startswith("cmd:"):
            return line.split("cmd:", 1)[1].strip()
    return None


def _read_log_tail(path: Path, *, tail_chars: int = 12000) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if len(text) <= int(tail_chars):
        return text
    return text[-int(tail_chars):]


def _write_failure_diagnostics(*, work_root: Path, stage: str, stage_dir: Path, exc: BaseException) -> Path:
    log_path = Path(stage_dir) / "md.log"
    checkpoint_path = Path(stage_dir) / "md.cpt"
    payload = {
        "stage": str(stage),
        "stage_dir": str(stage_dir),
        "exception_type": exc.__class__.__name__,
        "exception": str(exc),
        "command": _extract_cmd_from_exception(str(exc)),
        "checkpoint_exists": bool(checkpoint_path.exists()),
        "log_exists": bool(log_path.exists()),
        "lincs_fallback_eligible": bool(checkpoint_path.exists()) and ("lincs" in str(exc).lower() or "constraint" in str(exc).lower()),
        "log_tail": _read_log_tail(log_path),
    }
    out = Path(work_root) / "failure_diagnostics.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[FAILURE] wrote diagnostics to {out}")
    return out


def _run_partial_eq21(
    *,
    eqmd: eq.EQ21step,
    temp: float,
    press: float,
    mpi: int,
    omp: int,
    gpu: int,
    gpu_id: int | None,
    stage_cap: int,
    final_ns: float,
):
    exp = eqmd.ensure_system_exported()
    run_dir = Path(eqmd.work_dir) / "03_EQ21_partial"
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)

    cfg = eq.EQ21ProtocolConfig(
        t_max_k=float(temp),
        t_anneal_k=float(temp),
        p_max_bar=float(press),
        p_anneal_bar=float(press),
        npt_time_scale=float(eq21_npt_time_scale),
    )
    stages, records, params = eq._build_eq21_stages(
        temp=float(temp),
        press=float(press),
        final_ns=float(final_ns),
        cfg=cfg,
    )
    capped_stages = list(stages[: int(stage_cap)])
    capped_records = list(records[: int(stage_cap)])
    if not capped_stages:
        raise ValueError(f"YADONPY_EQ21_STAGE_CAP={stage_cap} does not select any stage.")

    eq._write_eq21_schedule(run_dir, capped_records, params)
    use_gpu, gid = eq._parse_gpu_args(gpu, gpu_id)
    res = eq.RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid)
    job = eq.EquilibrationJob(
        gro=exp.system_gro,
        top=exp.system_top,
        provenance_ndx=exp.system_ndx,
        out_dir=run_dir,
        stages=capped_stages,
        resources=res,
    )
    job.run(restart=False)
    print(
        "[EQ21-PARTIAL] completed stages: "
        + ", ".join(str(stage.name) for stage in capped_stages)
        + f" | output={run_dir}"
    )
    return run_dir


def main() -> int:
    doctor(print_report=True)
    ensure_initialized()

    wd = workdir(work_dir, restart=restart_status)
    analysis_dir = Path(wd) / "06_analysis"
    audit_dir = Path(wd) / "07_polymer_audit"
    if shared_polymer_root is not None:
        shared_polymer_wd = workdir(shared_polymer_root, restart=restart_status)
        cmc_rw_dir = shared_polymer_wd.child("CMC_rw")
        cmc_term_dir = shared_polymer_wd.child("CMC_term")
    else:
        cmc_rw_dir = wd.child("CMC_rw")
        cmc_term_dir = wd.child("CMC_term")
    ac_build_dir = wd.child("00_build_cell")

    msd_begin_ps = _env_optional_float("YADONPY_MSD_BEGIN_PS")
    if msd_begin_ps is None:
        msd_begin_ps = _default_msd_begin_ps(prod_ns)
    msd_end_ps = _env_optional_float("YADONPY_MSD_END_PS")

    if analysis_only:
        analy = AnalyzeResult.from_work_dir(wd)
        _ = analy.get_all_prop(temp=temp, press=press, save=True)
        if not skip_rdf:
            li_probe = utils.mol_from_smiles(Li_smiles)
            _ = analy.rdf(center_mol=li_probe)
        transport_mols = _transport_probe_mols(str(ADDITIVE["smiles"]))
        secondary_msd = None
        if msd_compare_drift_off and str(msd_drift).strip().lower() != "off":
            secondary_msd = analy.msd(
                mols=transport_mols,
                geometry="3d",
                unwrap="on",
                begin_ps=msd_begin_ps,
                end_ps=msd_end_ps,
                drift="off",
            )
        primary_msd = analy.msd(
            mols=transport_mols,
            geometry="3d",
            unwrap="on",
            begin_ps=msd_begin_ps,
            end_ps=msd_end_ps,
            drift=msd_drift,
        )
        if not skip_sigma:
            sigma_msd = analy.msd(
                geometry="3d",
                unwrap="on",
                begin_ps=msd_begin_ps,
                end_ps=msd_end_ps,
                drift=msd_drift,
            )
            _ = analy.sigma(temp_k=temp, msd=sigma_msd, drift=msd_drift)
        if not skip_den_dis:
            _ = analy.den_dis()
        diag_path = _write_transport_diagnostics(
            analysis_dir,
            moltypes=transport_labels,
            primary_msd=primary_msd,
            primary_label=f"drift={msd_drift}",
            secondary_msd=secondary_msd,
            secondary_label="drift=off" if secondary_msd is not None else None,
            begin_ps=msd_begin_ps,
            end_ps=msd_end_ps,
        )
        _print_transport_summary(msd_payload=primary_msd, label=f"primary drift={msd_drift}", moltypes=transport_labels)
        if secondary_msd is not None:
            _print_transport_summary(msd_payload=secondary_msd, label="secondary drift=off", moltypes=transport_labels)
        _warn_if_transport_is_fragile(
            primary_msd,
            additive_label=str(ADDITIVE["label"]),
            moltypes=transport_labels,
            secondary_msd=secondary_msd,
        )
        print(f"[ANALYSIS-ONLY] Transport diagnostics written to {diag_path}")
        return 0

    # ---------------- build monomers ----------------
    glucose = utils.mol_from_smiles(glucose_smiles)
    glucose = ff.ff_assign(glucose)
    if not glucose:
        raise RuntimeError("Can not assign force field parameters for glucose.")

    glucose_2 = _load_ready_from_moldb(
        ff,
        glucose_2_smiles,
        label="glucose_2",
        polyelectrolyte_mode=True,
        repo_db_dir=REPO_DB_DIR,
    )
    glucose_3 = _load_ready_from_moldb(
        ff,
        glucose_3_smiles,
        label="glucose_3",
        polyelectrolyte_mode=True,
        repo_db_dir=REPO_DB_DIR,
    )
    glucose_6 = _load_ready_from_moldb(
        ff,
        glucose_6_smiles,
        label="glucose_6",
        polyelectrolyte_mode=True,
        repo_db_dir=REPO_DB_DIR,
    )
    for name, mol in (
        ("glucose_2_ready", glucose_2),
        ("glucose_3_ready", glucose_3),
        ("glucose_6_ready", glucose_6),
    ):
        _write_polymer_checkpoint_audit(audit_dir, name, mol)

    # termination
    ter1 = utils.mol_from_smiles(ter_smiles)
    qm.assign_charges(
        ter1,
        charge="RESP",
        opt=True,
        work_dir=wd,
        omp=omp_psi4,
        memory=mem_mb,
        log_name="ter1",
    )

    # DP from target polymer Mw
    dp = poly.calc_n_from_mol_weight(
        [glucose, glucose_2, glucose_3, glucose_6],
        target_mw,
        ratio=feed_prob,
        terminal1=ter1,
    )
    print(f"[CMC] estimated DP from target Mw = {dp}")

    # random copolymerization (self-avoiding RW), then terminate
    chain_len = 12 if smoke_mode else 50
    CMC = poly.random_copolymerize_rw(
        [glucose, glucose_2, glucose_3, glucose_6],
        chain_len,
        ratio=feed_prob,
        tacticity="atactic",
        work_dir=cmc_rw_dir,
    )
    _write_polymer_checkpoint_audit(audit_dir, "cmc_random_walk", CMC)
    CMC = poly.terminate_rw(CMC, ter1, work_dir=cmc_term_dir)
    _write_polymer_checkpoint_audit(audit_dir, "cmc_terminated", CMC)
    CMC = ff.ff_assign(CMC, polyelectrolyte_mode=True)
    if not CMC:
        raise RuntimeError("Can not assign force field parameters for CMC.")
    _write_polymer_checkpoint_audit(audit_dir, "cmc_final_assigned", CMC)
    q_poly = _formal_charge(CMC)

    # ---------------- build solvents / additive ----------------
    EC = ff.ff_assign(utils.mol_from_smiles(EC_smiles))
    EMC = ff.ff_assign(utils.mol_from_smiles(EMC_smiles))
    DEC = ff.ff_assign(utils.mol_from_smiles(DEC_smiles))
    if not EC or not EMC or not DEC:
        raise RuntimeError("Can not assign force field parameters for carbonate solvents.")

    additive = _load_ready_from_moldb(
        ff,
        str(ADDITIVE["smiles"]),
        label=str(ADDITIVE["label"]),
        repo_db_dir=REPO_DB_DIR,
    )

    # ---------------- ions ----------------
    Li = ion_ff.mol(Li_smiles)
    Li = ion_ff.ff_assign(Li)
    if not Li:
        raise RuntimeError("Can not assign MERZ force field parameters for Li+.")

    Na = ion_ff.mol(Na_smiles)
    Na = ion_ff.ff_assign(Na)
    if not Na:
        raise RuntimeError("Can not assign MERZ force field parameters for Na+.")

    PF6 = _load_ready_from_moldb(ff, PF6_smiles, label="PF6", bonded="DRIH", repo_db_dir=REPO_DB_DIR)

    # ---------------- compute counts ----------------
    n_cmc = 1 if smoke_mode else 8
    n_na = abs(q_poly) * n_cmc
    additive_count = int(ADDITIVE["smoke_count"] if smoke_mode else ADDITIVE["default_count"])
    counts = _formulation_counts(
        smoke=smoke_mode,
        additive_default_count=additive_count,
        n_na=n_na,
        variant=system_variant,
        override=counts_override,
    )
    charge_scale = [0.7, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 1.0]

    print(
        f"[FORMULATION] ff={forcefield_name} variant={system_variant} additive={ADDITIVE['label']} "
        f"smoke_mode={smoke_mode} q_poly={q_poly} counts={counts}"
    )
    print(
        f"[RUNCFG] mpi={mpi} omp={omp} gpu={gpu} gpu_id={gpu_id} "
        f"eq21_final_ns={eq21_final_ns} eq21_npt_time_scale={eq21_npt_time_scale} "
        f"additional_ns={additional_ns} additional_rounds={additional_rounds} prod_ns={prod_ns} "
        f"shared_polymer_root={shared_polymer_root if shared_polymer_root is not None else '(none)'}"
    )
    print(
        f"[PRODMODE] ensemble={prod_ensemble} gpu_offload_mode={gpu_offload_mode} "
        f"bridge_ps={prod_bridge_ps} bridge_dt_fs={prod_bridge_dt_fs} "
        f"bridge_lincs_iter={prod_bridge_lincs_iter} bridge_lincs_order={prod_bridge_lincs_order} "
        f"nvt_density_control={nvt_density_control}"
    )
    print(
        f"[ANALYSISCFG] msd_begin_ps={msd_begin_ps} msd_end_ps={msd_end_ps} "
        f"msd_drift={msd_drift} compare_drift_off={msd_compare_drift_off} "
        f"skip_rdf={skip_rdf} skip_sigma={skip_sigma} skip_den_dis={skip_den_dis}"
    )
    print(
        f"[PRODOUT] traj_ps={prod_traj_ps} energy_ps={prod_energy_ps} "
        f"log_ps={prod_log_ps if prod_log_ps is not None else prod_energy_ps} "
        f"trr_ps={prod_trr_ps} velocity_ps={prod_velocity_ps} checkpoint_min={prod_checkpoint_min}"
    )

    # ---------------- pack amorphous cell ----------------
    species = [CMC, EC, EMC, DEC, Li, PF6, Na, additive]
    active = [(mol, cnt, scl) for mol, cnt, scl in zip(species, counts, charge_scale) if int(cnt) > 0]
    active_mols = [mol for mol, _cnt, _scl in active]
    active_counts = [int(cnt) for _mol, cnt, _scl in active]
    active_charge_scale = [float(scl) for _mol, _cnt, scl in active]
    ac = poly.amorphous_cell(
        active_mols,
        active_counts,
        charge_scale=active_charge_scale,
        polyelectrolyte_mode=True,
        density=0.05,
        neutralize=False,
        work_dir=ac_build_dir,
    )

    if build_only:
        print(f"[BUILD-ONLY] Finished cell construction at {ac_build_dir}")
        return 0

    # ---------------- run equilibration preset ----------------
    eqmd = eq.EQ21step(ac, work_dir=wd)
    if export_only:
        exported = eqmd.ensure_system_exported()
        write_polymer_audit(
            compare_exported_charge_groups(system_dir=exported.system_top.parent, moltype="CMC", mol=CMC),
            audit_dir / "cmc_export_charge_groups.json",
        )
        print(f"[EXPORT-ONLY] Exported 02_system at {exported.system_top.parent}")
        return 0
    if eq21_stage_cap > 0:
        _run_partial_eq21(
            eqmd=eqmd,
            temp=temp,
            press=press,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            stage_cap=eq21_stage_cap,
            final_ns=eq21_final_ns,
        )
        return 0
    try:
        ac = eqmd.exec(
            temp=temp,
            press=press,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            time=eq21_final_ns,
            eq21_npt_time_scale=eq21_npt_time_scale,
        )
    except Exception as exc:
        _write_failure_diagnostics(work_root=Path(wd), stage="eq21", stage_dir=Path(wd) / "03_EQ21", exc=exc)
        raise

    exported = eqmd.ensure_system_exported()
    write_polymer_audit(
        compare_exported_charge_groups(system_dir=exported.system_top.parent, moltype="CMC", mol=CMC),
        audit_dir / "cmc_export_charge_groups.json",
    )

    analy = eqmd.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    result = analy.check_eq()

    for _i in range(additional_rounds):
        if result:
            break
        eqmd = eq.Additional(ac, work_dir=wd)
        try:
            ac = eqmd.exec(
                temp=temp,
                press=press,
                mpi=mpi,
                omp=omp,
                gpu=gpu,
                gpu_id=gpu_id,
                time=additional_ns,
            )
        except Exception as exc:
            _write_failure_diagnostics(
                work_root=Path(wd),
                stage=f"additional_round_{_i + 1:02d}",
                stage_dir=Path(wd) / "04_eq_additional",
                exc=exc,
            )
            raise
        analy = eqmd.analyze()
        _ = analy.get_all_prop(temp=temp, press=press, save=True)
        result = analy.check_eq()

    if not result:
        print("[WARNING] Did not reach an equilibrium state after EQ21 + Additional cycles.")

    # ---------------- Production + analysis ----------------
    prod_runner: eq.NPT | eq.NVT
    if prod_ensemble == "nvt":
        prod_runner = eq.NVT(ac, work_dir=wd)
        prod_stage_name = "nvt_production"
        prod_stage_dir = Path(wd) / "05_nvt_production"
        prod_kwargs = {
            "temp": temp,
            "mpi": mpi,
            "omp": omp,
            "gpu": gpu,
            "gpu_id": gpu_id,
            "time": prod_ns,
            "traj_ps": prod_traj_ps,
            "energy_ps": prod_energy_ps,
            "log_ps": prod_log_ps,
            "trr_ps": prod_trr_ps,
            "velocity_ps": prod_velocity_ps,
            "checkpoint_min": prod_checkpoint_min,
            "gpu_offload_mode": gpu_offload_mode,
            "bridge_ps": prod_bridge_ps,
            "bridge_dt_fs": prod_bridge_dt_fs,
            "bridge_lincs_iter": prod_bridge_lincs_iter,
            "bridge_lincs_order": prod_bridge_lincs_order,
            "density_control": nvt_density_control,
        }
    else:
        prod_runner = eq.NPT(ac, work_dir=wd)
        prod_stage_name = "npt_production"
        prod_stage_dir = Path(wd) / "05_npt_production"
        prod_kwargs = {
            "temp": temp,
            "press": press,
            "mpi": mpi,
            "omp": omp,
            "gpu": gpu,
            "gpu_id": gpu_id,
            "time": prod_ns,
            "traj_ps": prod_traj_ps,
            "energy_ps": prod_energy_ps,
            "log_ps": prod_log_ps,
            "trr_ps": prod_trr_ps,
            "velocity_ps": prod_velocity_ps,
            "checkpoint_min": prod_checkpoint_min,
            "gpu_offload_mode": gpu_offload_mode,
            "bridge_ps": prod_bridge_ps,
            "bridge_dt_fs": prod_bridge_dt_fs,
            "bridge_lincs_iter": prod_bridge_lincs_iter,
            "bridge_lincs_order": prod_bridge_lincs_order,
        }
    try:
        ac = prod_runner.exec(**prod_kwargs)
    except Exception as exc:
        _write_failure_diagnostics(
            work_root=Path(wd),
            stage=prod_stage_name,
            stage_dir=prod_stage_dir,
            exc=exc,
        )
        raise

    analy = prod_runner.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    if not skip_rdf and counts[4] > 0:
        _ = analy.rdf(center_mol=Li)
    transport_mols = []
    present_transport_labels: list[str] = []
    if counts[7] > 0:
        transport_mols.append(additive)
        present_transport_labels.append(str(ADDITIVE["label"]))
    if counts[2] > 0:
        transport_mols.append(EMC)
        present_transport_labels.append("EMC")
    if counts[3] > 0:
        transport_mols.append(DEC)
        present_transport_labels.append("DEC")
    if counts[1] > 0:
        transport_mols.append(EC)
        present_transport_labels.append("EC")
    msd_drift_off = None
    if transport_mols and msd_compare_drift_off and str(msd_drift).strip().lower() != "off":
        msd_drift_off = analy.msd(
            mols=transport_mols,
            geometry="3d",
            unwrap="on",
            begin_ps=msd_begin_ps,
            end_ps=msd_end_ps,
            drift="off",
        )
    msd = (
        analy.msd(
            mols=transport_mols,
            geometry="3d",
            unwrap="on",
            begin_ps=msd_begin_ps,
            end_ps=msd_end_ps,
            drift=msd_drift,
        )
        if transport_mols
        else {}
    )
    if not skip_sigma:
        sigma_msd = analy.msd(
            geometry="3d",
            unwrap="on",
            begin_ps=msd_begin_ps,
            end_ps=msd_end_ps,
            drift=msd_drift,
        )
        _ = analy.sigma(temp_k=temp, msd=sigma_msd, drift=msd_drift)
    if not skip_den_dis:
        _ = analy.den_dis()
    diag_path = None
    if transport_mols:
        diag_path = _write_transport_diagnostics(
            analysis_dir,
            moltypes=present_transport_labels,
            primary_msd=msd,
            primary_label=f"drift={msd_drift}",
            secondary_msd=msd_drift_off,
            secondary_label="drift=off" if msd_drift_off is not None else None,
            begin_ps=msd_begin_ps,
            end_ps=msd_end_ps,
        )
        _print_transport_summary(msd_payload=msd, label=f"primary drift={msd_drift}", moltypes=present_transport_labels)
        if msd_drift_off is not None:
            _print_transport_summary(msd_payload=msd_drift_off, label="secondary drift=off", moltypes=present_transport_labels)
        _warn_if_transport_is_fragile(
            msd,
            additive_label=str(ADDITIVE["label"]),
            moltypes=present_transport_labels,
            secondary_msd=msd_drift_off,
        )
        print(f"[TRANSPORT] diagnostics written to {diag_path}")
    else:
        print("[TRANSPORT] skipped transport diagnostics because this system variant has no transport probes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
