from __future__ import annotations

"""CMC random copolymer + DTD electrolyte example using OPLS-AA.

This variant keeps RESP charges for all non-ionic species by loading them from
MolDB. Li+ and Na+ use the built-in OPLS-AA ion parameters directly. PF6-
reuses its MolDB DRIH bonded topology, then swaps its atom types / charges to
the built-in OPLS-AA ion values because the bundled OPLS-AA table does not yet
ship PF6 bonded terms.

Use ``YADONPY_BUILD_ONLY=1`` to stop after amorphous-cell construction.
Use ``YADONPY_EXPORT_ONLY=1`` to stop after exporting ``02_system``.
Use ``YADONPY_SMOKE=1`` for a smaller polymer / solvent composition.
Use ``YADONPY_EQ21_STAGE_CAP=<N>`` to run only the first ``N`` EQ21 stages
(for example ``3`` = ``01_em`` + ``02_preNVT`` + ``03_EQ21/step_01``).
"""

import os
import shutil
from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2_mod, OPLSAA
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


def _env_text(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    if raw is None:
        return str(default)
    text = str(raw).strip()
    return text if text else str(default)


def _env_int_list(name: str, expected_len: int) -> list[int] | None:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    vals = [int(tok.strip()) for tok in raw.split(",") if str(tok).strip()]
    if len(vals) != int(expected_len):
        raise ValueError(f"{name} expects {expected_len} comma-separated integers, got {len(vals)}")
    return vals


def _normalize_opls_charge_mode(raw: str | None) -> str:
    mode = str(raw or "resp").strip().lower()
    if mode in {"native", "opls", "oplsaa"}:
        return "opls"
    return "resp"


def _load_ready_resp_from_moldb(
    ff: OPLSAA,
    smiles: str,
    *,
    label: str,
    polyelectrolyte_mode: bool = False,
    repo_db_dir: Path,
    charge_mode: str = "resp",
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
            assign_charge = "opls" if str(charge_mode).strip().lower() == "opls" else None
            mol = ff.ff_assign(mol, charge=assign_charge, report=False)
            if not mol:
                raise RuntimeError(f"Cannot assign OPLS-AA parameters for {label}.")
            if assign_charge == "opls":
                print(f"[MolDB] loaded {label} geometry from {db_label} db and switched to built-in OPLS-AA charges")
            else:
                print(f"[MolDB] loaded {label} with RESP charges from {db_label} db")
            return mol
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"{label} is expected to be RESP-ready in MolDB for the OPLS-AA workflow."
    ) from last_exc


def _assign_builtin_opls_ion(ff: OPLSAA, smiles: str, *, label: str):
    mol = ff.mol(smiles, charge="opls", require_ready=False, prefer_db=False)
    mol = ff.ff_assign(mol, charge="opls", report=False)
    if not mol:
        raise RuntimeError(f"Cannot assign built-in OPLS-AA ion parameters for {label}.")
    print(f"[OPLS-AA] assigned built-in ion parameters for {label}")
    return mol


def _load_pf6_with_opls_builtin_charges(*, ion_ff: OPLSAA, repo_db_dir: Path):
    gaff_ff = GAFF2_mod()
    last_exc: Exception | None = None
    opls_probe = ion_ff.mol(PF6_smiles, charge="opls", require_ready=False, prefer_db=False)
    if not ion_ff.assign_ptypes(opls_probe, charge="opls"):
        raise RuntimeError("Cannot build the PF6 OPLS-AA atom-type probe from SMILES.")

    for db_dir, db_label in ((None, "default"), (repo_db_dir, "repo")):
        try:
            pf6 = gaff_ff.mol_rdkit(
                PF6_smiles,
                name="PF6",
                db_dir=db_dir,
                charge="RESP",
                require_ready=True,
                prefer_db=True,
            )
            pf6 = gaff_ff.ff_assign(pf6, bonded="DRIH", report=False)
            if not pf6:
                raise RuntimeError("Cannot restore PF6 DRIH bonded topology from MolDB.")

            if pf6.GetNumAtoms() != opls_probe.GetNumAtoms():
                raise RuntimeError("PF6 probe atom count does not match the MolDB topology.")

            for src_atom, dst_atom in zip(opls_probe.GetAtoms(), pf6.GetAtoms()):
                if src_atom.GetSymbol() != dst_atom.GetSymbol():
                    raise RuntimeError("PF6 probe atom ordering does not match the MolDB topology.")
                dst_atom.SetProp("ff_btype", src_atom.GetProp("ff_btype"))
                dst_atom.SetProp("ff_type", src_atom.GetProp("ff_type"))
                dst_atom.SetDoubleProp("ff_sigma", src_atom.GetDoubleProp("ff_sigma"))
                dst_atom.SetDoubleProp("ff_epsilon", src_atom.GetDoubleProp("ff_epsilon"))
                dst_atom.SetDoubleProp("AtomicCharge", src_atom.GetDoubleProp("AtomicCharge"))
                if src_atom.HasProp("ff_desc"):
                    dst_atom.SetProp("ff_desc", src_atom.GetProp("ff_desc"))

            pf6.SetProp("ff_name", str(ion_ff.name))
            pf6.SetProp("ff_class", str(ion_ff.ff_class))
            pf6.SetProp("pair_style", str(ion_ff.pair_style))
            print(
                "[OPLS-AA] loaded PF6 bonded topology from "
                f"{db_label} db and replaced atom types / charges with built-in OPLS-AA values"
            )
            return pf6
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        "PF6 is expected to exist in MolDB with bonded='DRIH' for this OPLS-AA example."
    ) from last_exc


# ---------------- user inputs ----------------
restart_status = _env_flag("YADONPY_RESTART", default=False)
build_only = _env_flag("YADONPY_BUILD_ONLY", default=False)
export_only = _env_flag("YADONPY_EXPORT_ONLY", default=False)
smoke_mode = _env_flag("YADONPY_SMOKE", default=False)
fast_analysis = _env_flag("YADONPY_FAST_ANALYSIS", default=False)
skip_rdf = _env_flag("YADONPY_SKIP_RDF", default=fast_analysis)
skip_sigma = _env_flag("YADONPY_SKIP_SIGMA", default=fast_analysis)
skip_den_dis = _env_flag("YADONPY_SKIP_DEN_DIS", default=fast_analysis)
eq21_stage_cap = _env_int("YADONPY_EQ21_STAGE_CAP", 0)
eq21_dt_ps = _env_float("YADONPY_EQ21_DT_PS", 0.0005)
eq21_pre_nvt_ps = _env_float("YADONPY_EQ21_PRE_NVT_PS", 10.0)
eq21_final_ns = _env_float("YADONPY_EQ21_FINAL_NS", 0.8)
eq21_tmax = _env_float("YADONPY_EQ21_TMAX_K", 1000.0)
eq21_pmax = _env_float("YADONPY_EQ21_PMAX_BAR", 50000.0)
eq21_npt_time_scale = _env_float("YADONPY_EQ21_NPT_TIME_SCALE", 2.0)
eq21_robust = _env_flag("YADONPY_EQ21_ROBUST", default=True)
eq21_em_nsteps = _env_int("YADONPY_EQ21_EM_NSTEPS", 50000)
eq21_em_emtol = _env_float("YADONPY_EQ21_EM_EMTOL", 1000.0)
eq21_em_emstep = _env_float("YADONPY_EQ21_EM_EMSTEP", 0.001)
counts_override = _env_int_list("YADONPY_COUNTS", 8)
prod_ns = _env_float("YADONPY_PROD_NS", 20.0)
gpu_offload_mode = _env_text("YADONPY_GPU_OFFLOAD_MODE", "auto")

set_run_options(restart=restart_status)

ff = OPLSAA()
ion_ff = OPLSAA()
opls_charge_mode = _normalize_opls_charge_mode(os.environ.get("YADONPY_OPLS_CHARGE_MODE"))

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

omp_psi4 = _env_int("YADONPY_PSI4_OMP", 20)
mem_mb = _env_int("YADONPY_PSI4_MEMORY_MB", 20000)

BASE_DIR = Path(__file__).resolve().parent
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"
_work_dir_override = str(os.environ.get("YADONPY_WORK_DIR", "")).strip()
_shared_polymer_root_override = str(os.environ.get("YADONPY_SHARED_POLYMER_ROOT", "")).strip()
work_dir = (
    Path(_work_dir_override).expanduser()
    if _work_dir_override
    else (BASE_DIR / "work_dir_dtd_oplsaa_moldb")
)
shared_polymer_root = (
    Path(_shared_polymer_root_override).expanduser() if _shared_polymer_root_override else None
)


def _formal_charge(mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


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
        shutil.rmtree(run_dir, ignore_errors=True)

    cfg = eq.EQ21ProtocolConfig(
        t_max_k=float(eq21_tmax),
        t_anneal_k=float(temp),
        p_max_bar=float(eq21_pmax),
        p_anneal_bar=float(press),
        dt_ps=float(eq21_dt_ps),
        pre_nvt_ps=float(eq21_pre_nvt_ps),
        robust=bool(eq21_robust),
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
    first_stage = capped_stages[0]
    if first_stage.kind == "minim":
        capped_stages[0] = eq.EqStage(
            first_stage.name,
            first_stage.kind,
            type(first_stage.mdp)(
                first_stage.mdp.template,
                {
                    **first_stage.mdp.params,
                    "nsteps": int(eq21_em_nsteps),
                    "emtol": float(eq21_em_emtol),
                    "emstep": float(eq21_em_emstep),
                },
            ),
        )

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
    if shared_polymer_root is not None:
        shared_polymer_wd = workdir(shared_polymer_root, restart=restart_status)
        cmc_rw_dir = shared_polymer_wd.child("CMC_rw")
        cmc_term_dir = shared_polymer_wd.child("CMC_term")
    else:
        cmc_rw_dir = wd.child("CMC_rw")
        cmc_term_dir = wd.child("CMC_term")
    ac_build_dir = wd.child("00_build_cell")

    # ---------------- build monomers ----------------
    glucose = _load_ready_resp_from_moldb(
        ff,
        glucose_smiles,
        label="glucose",
        repo_db_dir=REPO_DB_DIR,
        charge_mode=opls_charge_mode,
    )
    glucose_2 = _load_ready_resp_from_moldb(
        ff,
        glucose_2_smiles,
        label="glucose_2",
        polyelectrolyte_mode=True,
        repo_db_dir=REPO_DB_DIR,
        charge_mode=opls_charge_mode,
    )
    glucose_3 = _load_ready_resp_from_moldb(
        ff,
        glucose_3_smiles,
        label="glucose_3",
        polyelectrolyte_mode=True,
        repo_db_dir=REPO_DB_DIR,
        charge_mode=opls_charge_mode,
    )
    glucose_6 = _load_ready_resp_from_moldb(
        ff,
        glucose_6_smiles,
        label="glucose_6",
        polyelectrolyte_mode=True,
        repo_db_dir=REPO_DB_DIR,
        charge_mode=opls_charge_mode,
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
        log_name="ter1_oplsaa",
    )

    dp = poly.calc_n_from_mol_weight(
        [glucose, glucose_2, glucose_3, glucose_6],
        target_mw,
        ratio=feed_prob,
        terminal1=ter1,
    )
    print(f"[CMC] estimated DP from target Mw = {dp}")

    chain_len = 12 if smoke_mode else 50
    CMC = poly.random_copolymerize_rw(
        [glucose, glucose_2, glucose_3, glucose_6],
        chain_len,
        ratio=feed_prob,
        tacticity="atactic",
        work_dir=cmc_rw_dir,
    )
    CMC = poly.terminate_rw(CMC, ter1, work_dir=cmc_term_dir)
    CMC_charge = "opls" if opls_charge_mode == "opls" else None
    CMC = ff.ff_assign(CMC, charge=CMC_charge, report=False)
    if not CMC:
        raise RuntimeError("Cannot assign OPLS-AA parameters for CMC.")
    q_poly = _formal_charge(CMC)

    # ---------------- build solvents / additive ----------------
    EC = _load_ready_resp_from_moldb(ff, EC_smiles, label="EC", repo_db_dir=REPO_DB_DIR, charge_mode=opls_charge_mode)
    EMC = _load_ready_resp_from_moldb(ff, EMC_smiles, label="EMC", repo_db_dir=REPO_DB_DIR, charge_mode=opls_charge_mode)
    DEC = _load_ready_resp_from_moldb(ff, DEC_smiles, label="DEC", repo_db_dir=REPO_DB_DIR, charge_mode=opls_charge_mode)
    DTD = _load_ready_resp_from_moldb(ff, DTD_smiles, label="DTD", repo_db_dir=REPO_DB_DIR, charge_mode=opls_charge_mode)

    # ---------------- ions ----------------
    Li = _assign_builtin_opls_ion(ion_ff, Li_smiles, label="Li")
    Na = _assign_builtin_opls_ion(ion_ff, Na_smiles, label="Na")
    PF6 = _load_pf6_with_opls_builtin_charges(ion_ff=ion_ff, repo_db_dir=REPO_DB_DIR)

    # ---------------- compute counts ----------------
    n_cmc = 1 if smoke_mode else 8
    n_na = abs(q_poly) * n_cmc
    if counts_override is not None:
        counts = list(counts_override)
    elif smoke_mode:
        counts = [n_cmc, 6, 6, 6, 2, 2, n_na, 1]
    else:
        counts = [n_cmc, 40, 50, 20, 10, 10, n_na, 4]
    charge_scale = [0.7, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 1.0]

    print(
        f"[FORMULATION] smoke_mode={smoke_mode} q_poly={q_poly} counts={counts}"
        + (" | source=YADONPY_COUNTS" if counts_override is not None else "")
    )

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

    eqmd = eq.EQ21step(ac, work_dir=wd)
    if export_only:
        exported = eqmd.ensure_system_exported()
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

    ac = eqmd.exec(
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        eq21_tmax=eq21_tmax,
        eq21_pmax=eq21_pmax,
        eq21_dt_ps=eq21_dt_ps,
        eq21_pre_nvt_ps=eq21_pre_nvt_ps,
        eq21_robust=eq21_robust,
        eq21_npt_time_scale=eq21_npt_time_scale,
        sim_time=eq21_final_ns,
    )

    analy = eqmd.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    result = analy.check_eq()

    for _i in range(4):
        if result:
            break
        eqmd = eq.Additional(ac, work_dir=wd)
        ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
        analy = eqmd.analyze()
        _ = analy.get_all_prop(temp=temp, press=press, save=True)
        result = analy.check_eq()

    if not result:
        print("[WARNING] Did not reach an equilibrium state after EQ21 + Additional cycles.")

    npt = eq.NPT(ac, work_dir=wd)
    ac = npt.exec(
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        time=float(prod_ns),
        gpu_offload_mode=gpu_offload_mode,
    )

    analy = npt.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    if not skip_rdf:
        _ = analy.rdf(center_mol=Li)
    msd = analy.msd()
    if not skip_sigma:
        _ = analy.sigma(temp_k=temp, msd=msd)
    if not skip_den_dis:
        _ = analy.den_dis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
