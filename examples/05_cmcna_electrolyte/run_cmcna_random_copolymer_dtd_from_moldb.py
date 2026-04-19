from __future__ import annotations

"""CMC random copolymer + DTD electrolyte example backed by MolDB monomers.

The anionic glucose monomers are expected to be RESP-ready in MolDB with
``polyelectrolyte_mode=True``. This keeps the expensive monomer QM step out of
the system-build script and directly exercises the MolDB -> ITP export path.

Use ``YADONPY_BUILD_ONLY=1`` to stop after amorphous-cell construction.
Use ``YADONPY_EXPORT_ONLY=1`` to stop after exporting ``02_system``.
These modes are useful for checking topology / ITP generation on machines
without GROMACS.
"""

import json
import os
from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import utils, poly, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2, MERZ
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

ff = GAFF2()
ion_ff = MERZ()

# ---- CMC monomers (two connection points '*...*') ----
glucose_smiles = "*OC1OC(CO)C(*)C(O)C1O"
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"

DTD_smiles = "O=S1(=O)OC=CO1"

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
msd_drift = _env_text("YADONPY_MSD_DRIFT", "off")
msd_compare_drift_off = _env_flag(
    "YADONPY_COMPARE_DRIFT_OFF",
    default=(str(msd_drift).strip().lower() != "off" and not smoke_mode),
)
eq21_final_ns = _env_float("YADONPY_EQ21_FINAL_NS", 0.8)
eq21_npt_time_scale = _env_float("YADONPY_EQ21_NPT_TIME_SCALE", 2.0)
additional_ns = _env_float("YADONPY_ADDITIONAL_NS", 1.0)
additional_rounds = _env_int("YADONPY_ADDITIONAL_MAX_ROUNDS", 4)

omp_psi4 = _env_int("YADONPY_PSI4_OMP", 20)
mem_mb = _env_int("YADONPY_PSI4_MEMORY_MB", 20000)

BASE_DIR = Path(__file__).resolve().parent
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"
_work_dir_override = str(os.environ.get("YADONPY_WORK_DIR", "")).strip()
work_dir = Path(_work_dir_override).expanduser() if _work_dir_override else (BASE_DIR / "work_dir_dtd_moldb")


def _formal_charge(mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


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
    primary_msd: dict,
    primary_label: str,
    secondary_msd: dict | None = None,
    secondary_label: str | None = None,
    begin_ps: float | None,
    end_ps: float | None,
) -> Path:
    moltypes = ["DTD", "EMC", "DEC", "EC"]
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


def _print_transport_summary(*, msd_payload: dict, label: str) -> None:
    ranking = _rank_transport_species(msd_payload, ["DTD", "EMC", "DEC", "EC"])
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


def _warn_if_transport_is_fragile(primary_msd: dict, secondary_msd: dict | None = None) -> None:
    dtd = _extract_species_metric(primary_msd, "DTD")
    if not dtd:
        return
    if int(dtd.get("n_groups") or 0) <= 4:
        print(
            "[TRANSPORT][WARNING] DTD diffusion is being estimated from 4 or fewer molecular COM groups; "
            "treat ranking against bulk solvents as low-statistics."
        )
    if str(dtd.get("confidence") or "").lower() not in {"high", "medium"}:
        print(
            "[TRANSPORT][WARNING] DTD diffusion fit is not high-confidence; "
            "check msd.json / transport_diagnostics.json before trusting solvent-order conclusions."
        )
    if secondary_msd is None:
        return

    primary_rank = [row["moltype"] for row in _rank_transport_species(primary_msd, ["DTD", "EMC", "DEC", "EC"])]
    secondary_rank = [row["moltype"] for row in _rank_transport_species(secondary_msd, ["DTD", "EMC", "DEC", "EC"])]
    if primary_rank and secondary_rank and primary_rank != secondary_rank:
        print(
            "[TRANSPORT][WARNING] Solvent/additive ranking changes when drift correction is toggled; "
            "treat this run as drift-sensitive and prefer longer sampling or tail-window reanalysis."
        )

    for moltype in ("EC", "EMC", "DEC", "DTD"):
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


def _transport_probe_mols():
    return [
        utils.mol_from_smiles(DTD_smiles),
        utils.mol_from_smiles(EMC_smiles),
        utils.mol_from_smiles(DEC_smiles),
        utils.mol_from_smiles(EC_smiles),
    ]


def main() -> int:
    doctor(print_report=True)
    ensure_initialized()

    wd = workdir(work_dir, restart=restart_status)
    analysis_dir = Path(wd) / "06_analysis"
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
        transport_mols = _transport_probe_mols()
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
            primary_msd=primary_msd,
            primary_label=f"drift={msd_drift}",
            secondary_msd=secondary_msd,
            secondary_label="drift=off" if secondary_msd is not None else None,
            begin_ps=msd_begin_ps,
            end_ps=msd_end_ps,
        )
        _print_transport_summary(msd_payload=primary_msd, label=f"primary drift={msd_drift}")
        if secondary_msd is not None:
            _print_transport_summary(msd_payload=secondary_msd, label="secondary drift=off")
        _warn_if_transport_is_fragile(primary_msd, secondary_msd)
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
    CMC = poly.terminate_rw(CMC, ter1, work_dir=cmc_term_dir)
    CMC = ff.ff_assign(CMC, polyelectrolyte_mode=True)
    if not CMC:
        raise RuntimeError("Can not assign force field parameters for CMC.")
    q_poly = _formal_charge(CMC)

    # ---------------- build solvents / additive ----------------
    EC = ff.ff_assign(utils.mol_from_smiles(EC_smiles))
    EMC = ff.ff_assign(utils.mol_from_smiles(EMC_smiles))
    DEC = ff.ff_assign(utils.mol_from_smiles(DEC_smiles))
    if not EC or not EMC or not DEC:
        raise RuntimeError("Can not assign force field parameters for carbonate solvents.")

    DTD = _load_ready_from_moldb(ff, DTD_smiles, label="DTD", repo_db_dir=REPO_DB_DIR)

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
    if smoke_mode:
        counts = [n_cmc, 6, 6, 6, 2, 2, n_na, 1]
    else:
        counts = [n_cmc, 40, 50, 20, 10, 10, n_na, 4]
    charge_scale = [0.7, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 1.0]

    print(f"[FORMULATION] smoke_mode={smoke_mode} q_poly={q_poly} counts={counts}")
    print(
        f"[RUNCFG] mpi={mpi} omp={omp} gpu={gpu} gpu_id={gpu_id} "
        f"eq21_final_ns={eq21_final_ns} eq21_npt_time_scale={eq21_npt_time_scale} "
        f"additional_ns={additional_ns} additional_rounds={additional_rounds} prod_ns={prod_ns}"
    )
    print(
        f"[ANALYSISCFG] msd_begin_ps={msd_begin_ps} msd_end_ps={msd_end_ps} "
        f"msd_drift={msd_drift} compare_drift_off={msd_compare_drift_off} "
        f"skip_rdf={skip_rdf} skip_sigma={skip_sigma} skip_den_dis={skip_den_dis}"
    )

    # ---------------- pack amorphous cell ----------------
    ac = poly.amorphous_cell(
        [CMC, EC, EMC, DEC, Li, PF6, Na, DTD],
        counts,
        charge_scale=charge_scale,
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
        print(f"[EXPORT-ONLY] Exported 02_system at {exported.system_top.parent}")
        return 0
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

    analy = eqmd.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    result = analy.check_eq()

    for _i in range(additional_rounds):
        if result:
            break
        eqmd = eq.Additional(ac, work_dir=wd)
        ac = eqmd.exec(
            temp=temp,
            press=press,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            time=additional_ns,
        )
        analy = eqmd.analyze()
        _ = analy.get_all_prop(temp=temp, press=press, save=True)
        result = analy.check_eq()

    if not result:
        print("[WARNING] Did not reach an equilibrium state after EQ21 + Additional cycles.")

    # ---------------- Production NPT + analysis ----------------
    npt = eq.NPT(ac, work_dir=wd)
    ac = npt.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=prod_ns)

    analy = npt.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    if not skip_rdf:
        _ = analy.rdf(center_mol=Li)
    transport_mols = [DTD, EMC, DEC, EC]
    msd_drift_off = None
    if msd_compare_drift_off and str(msd_drift).strip().lower() != "off":
        msd_drift_off = analy.msd(
            mols=transport_mols,
            geometry="3d",
            unwrap="on",
            begin_ps=msd_begin_ps,
            end_ps=msd_end_ps,
            drift="off",
        )
    msd = analy.msd(
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
        primary_msd=msd,
        primary_label=f"drift={msd_drift}",
        secondary_msd=msd_drift_off,
        secondary_label="drift=off" if msd_drift_off is not None else None,
        begin_ps=msd_begin_ps,
        end_ps=msd_end_ps,
    )
    _print_transport_summary(msd_payload=msd, label=f"primary drift={msd_drift}")
    if msd_drift_off is not None:
        _print_transport_summary(msd_payload=msd_drift_off, label="secondary drift=off")
    _warn_if_transport_is_fragile(msd, msd_drift_off)
    print(f"[TRANSPORT] diagnostics written to {diag_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
