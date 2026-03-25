from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yadonpy as yp


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
EXPECTED_SRC_ROOT = (REPO_ROOT / "src").resolve()
IMPORTED_SRC_ROOT = Path(yp.__file__).resolve().parents[1]
if IMPORTED_SRC_ROOT != EXPECTED_SRC_ROOT:
    raise RuntimeError(
        "Example 12 must run against the current source tree under "
        f"{EXPECTED_SRC_ROOT}. Use run_eg12_local_cuda.bat or run_eg12_remote_cuda.sh, "
        "or set PYTHONPATH=<repo>/src before running this script."
    )

from yadonpy.runtime import recommend_local_resources, set_run_options
from yadonpy.core import molecular_weight, poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2, MERZ
from yadonpy.interface import (
    InterfaceBuilder,
    InterfaceDynamics,
    build_bulk_equilibrium_profile,
    equilibrate_bulk_with_eq21,
    format_cell_charge_audit,
    make_orthorhombic_pack_cell,
    plan_probe_polymer_matched_interface_preparation,
    plan_resized_polymer_matched_interface_from_probe,
    read_equilibrated_box_nm,
    recommend_polymer_diffusion_interface_recipe,
)
from yadonpy.io.mol2 import write_mol2
from yadonpy.sim import qm


profile_default = str(os.environ.get("YADONPY_EG12_PROFILE", "full")).strip().lower() or "full"
restart_default = str(os.environ.get("YADONPY_RESTART", "1")).strip().lower() not in {"0", "false", "no", "off"}
term_qm_default = str(os.environ.get("YADONPY_EG12_TERM_QM", "0")).strip().lower() in {"1", "true", "yes", "on"}
cpu_cap_default = os.environ.get("YADONPY_CPU_CAP")
n_cmc_default = os.environ.get("YADONPY_EG12_N_CMC")
dp_default = os.environ.get("YADONPY_EG12_DP")
bulk_loops_default = os.environ.get("YADONPY_EG12_BULK_LOOPS")
bulk_final_npt_default = os.environ.get("YADONPY_EG12_BULK_FINAL_NPT_NS")
bottom_thickness_default = os.environ.get("YADONPY_EG12_BOTTOM_THICKNESS_NM")
top_thickness_default = os.environ.get("YADONPY_EG12_TOP_THICKNESS_NM")
gap_default = os.environ.get("YADONPY_EG12_INTERFACE_GAP_NM")
vacuum_default = os.environ.get("YADONPY_EG12_INTERFACE_VACUUM_NM")
work_dir_name_default = str(os.environ.get("YADONPY_EG12_WORKDIR", "")).strip()

parser = argparse.ArgumentParser(
    description="Example 12: CMC-Na vs 1 M LiPF6 interface workflow.",
)
parser.add_argument("--profile", choices=("full", "smoke"), default=profile_default)
parser.add_argument(
    "--stop-after",
    choices=("full", "polymer_bulk", "probe_bulk", "electrolyte_bulk", "interface_build"),
    default="full",
)
parser.add_argument("--n-cmc", type=int, default=(int(n_cmc_default) if n_cmc_default else None))
parser.add_argument("--dp", type=int, default=(int(dp_default) if dp_default else None))
parser.add_argument("--bulk-loops", type=int, default=(int(bulk_loops_default) if bulk_loops_default else None))
parser.add_argument(
    "--bulk-final-npt-ns",
    type=float,
    default=(float(bulk_final_npt_default) if bulk_final_npt_default else None),
)
parser.add_argument(
    "--bottom-thickness-nm",
    type=float,
    default=(float(bottom_thickness_default) if bottom_thickness_default else None),
)
parser.add_argument(
    "--top-thickness-nm",
    type=float,
    default=(float(top_thickness_default) if top_thickness_default else None),
)
parser.add_argument(
    "--interface-gap-nm",
    type=float,
    default=(float(gap_default) if gap_default else None),
)
parser.add_argument(
    "--interface-vacuum-nm",
    type=float,
    default=(float(vacuum_default) if vacuum_default else None),
)
parser.add_argument("--work-dir-name", default=(work_dir_name_default or None))
parser.add_argument("--cpu-cap", type=int, default=(int(cpu_cap_default) if cpu_cap_default else None))
parser.add_argument("--mpi", type=int, default=None)
parser.add_argument("--omp", type=int, default=None)
parser.add_argument("--gpu", type=int, choices=(0, 1), default=None)
parser.add_argument("--gpu-id", type=int, default=None)
parser.add_argument("--omp-psi4", type=int, default=None)
parser.add_argument("--restart", dest="restart", action="store_true")
parser.add_argument("--fresh", dest="restart", action="store_false")
parser.add_argument("--with-term-qm", dest="term_qm", action="store_true")
parser.add_argument("--without-term-qm", dest="term_qm", action="store_false")
parser.set_defaults(restart=restart_default, term_qm=term_qm_default)
args = parser.parse_args()

resources = recommend_local_resources(cpu_cap=args.cpu_cap, gpu_default=1, gpu_id_default=0, omp_psi4_cap=8)

restart = bool(args.restart)
set_run_options(restart=restart)

ff = GAFF2()
ion_ff = MERZ()

glucose_smiles = "*OC1OC(CO)C(*)C(O)C1O"
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"
feed_ratio = [12, 26, 27, 35]
feed_prob = poly.ratio_to_prob(feed_ratio)

ter_smiles = "[H][*]"

EC_smiles = "O=C1OCCO1"
DEC_smiles = "CCOC(=O)OCC"
EMC_smiles = "CCOC(=O)OC"

Li_smiles = "[Li+]"
PF6_smiles = "F[P-](F)(F)(F)(F)F"
Na_smiles = "[Na+]"

temp = 300.0
press = 1.0
mpi = int(args.mpi if args.mpi is not None else resources.mpi)
omp = int(args.omp if args.omp is not None else resources.omp)
gpu = int(args.gpu if args.gpu is not None else resources.gpu)
gpu_id = int(args.gpu_id if args.gpu_id is not None else (resources.gpu_id if resources.gpu_id is not None else 0))
omp_psi4 = int(args.omp_psi4 if args.omp_psi4 is not None else resources.omp_psi4)
mem_mb = 20000

n_CMC = 6
dp = 150
polymer_initial_pack_density_g_cm3 = 0.015
polymer_pack_retry = 80
polymer_pack_retry_step = 6000
polymer_pack_threshold_ang = 1.65
polymer_pack_dec_rate = 0.75

solvent_mass_ratio = (3.0, 2.0, 5.0)
salt_molarity_M = 1.0
min_salt_pairs = 20
electrolyte_probe_density_g_cm3 = 1.00
electrolyte_probe_volume_scale = 2.30
electrolyte_probe_pack_density_g_cm3 = 0.42
electrolyte_probe_z_padding_factor = 1.25

electrolyte_pack_retry = 80
electrolyte_pack_retry_step = 4000
electrolyte_pack_threshold_ang = 1.55
electrolyte_pack_dec_rate = 0.70

interface_gap_nm = 0.80
interface_vacuum_nm = 14.0
interface_bottom_thickness_nm = 7.0
interface_top_thickness_nm = 8.0
interface_surface_shell_nm = 1.0
interface_core_guard_nm = 0.8

bulk_additional_loops = 4
bulk_final_npt_ns = 0.0
bulk_eq21_exec_kwargs: dict[str, float] = {}
probe_fixed_xy_npt_ns = 5.0
recipe_pre_contact_ps = 180.0
recipe_density_relax_ps = 500.0
recipe_contact_ps = 500.0
recipe_release_ps = 500.0
recipe_exchange_ns = 4.0
recipe_production_ns = 8.0

if args.profile == "smoke":
    n_CMC = 2
    dp = 40
    polymer_initial_pack_density_g_cm3 = 0.020
    polymer_pack_retry = 40
    polymer_pack_retry_step = 2000
    polymer_pack_threshold_ang = 1.55
    min_salt_pairs = 2
    electrolyte_probe_volume_scale = 1.55
    electrolyte_probe_pack_density_g_cm3 = 0.30
    electrolyte_probe_z_padding_factor = 1.30
    electrolyte_pack_retry = 40
    electrolyte_pack_retry_step = 1200
    electrolyte_pack_threshold_ang = 1.60
    electrolyte_pack_dec_rate = 0.65
    interface_gap_nm = 0.60
    interface_vacuum_nm = 8.0
    interface_bottom_thickness_nm = 4.0
    interface_top_thickness_nm = 4.5
    interface_surface_shell_nm = 0.8
    interface_core_guard_nm = 0.5
    bulk_additional_loops = 0
    bulk_final_npt_ns = 0.25
    bulk_eq21_exec_kwargs = {
        "eq21_tmax": 600.0,
        "eq21_pmax": 10000.0,
        "eq21_pre_nvt_ps": 2.0,
        "sim_time": 0.05,
    }
    probe_fixed_xy_npt_ns = 0.5
    recipe_pre_contact_ps = 10.0
    recipe_density_relax_ps = 20.0
    recipe_contact_ps = 20.0
    recipe_release_ps = 20.0
    recipe_exchange_ns = 0.02
    recipe_production_ns = 0.02

if args.n_cmc is not None:
    n_CMC = int(args.n_cmc)
if args.dp is not None:
    dp = int(args.dp)
if args.bulk_loops is not None:
    bulk_additional_loops = int(args.bulk_loops)
if args.bulk_final_npt_ns is not None:
    bulk_final_npt_ns = float(args.bulk_final_npt_ns)
if args.bottom_thickness_nm is not None:
    interface_bottom_thickness_nm = float(args.bottom_thickness_nm)
if args.top_thickness_nm is not None:
    interface_top_thickness_nm = float(args.top_thickness_nm)
if args.interface_gap_nm is not None:
    interface_gap_nm = float(args.interface_gap_nm)
if args.interface_vacuum_nm is not None:
    interface_vacuum_nm = float(args.interface_vacuum_nm)

work_dir_name = args.work_dir_name or ("work_dir_smoke" if args.profile == "smoke" else "work_dir")
work_dir = workdir(BASE_DIR / work_dir_name, clean=not restart)


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    print("[CONFIG] profile:", args.profile)
    print("[CONFIG] stop_after:", args.stop_after)
    print("[CONFIG] restart:", restart)
    print("[CONFIG] resources:", {"mpi": mpi, "omp": omp, "gpu": gpu, "gpu_id": gpu_id, "omp_psi4": omp_psi4})
    print("[CONFIG] cpu_budget:", {"detected": resources.cpu_total, "cap": resources.cpu_cap})

    mol2_dir = work_dir / "00_molecules"
    workflow_summary_path = work_dir / "workflow_summary.json"
    workflow_summary = {
        "profile": args.profile,
        "restart": restart,
        "stop_after": args.stop_after,
        "status": "running",
        "resources": {"mpi": mpi, "omp": omp, "gpu": gpu, "gpu_id": gpu_id, "omp_psi4": omp_psi4},
        "cpu_budget": {"detected": resources.cpu_total, "cap": resources.cpu_cap},
        "targets": {
            "n_CMC": n_CMC,
            "dp": dp,
            "bulk_additional_loops": bulk_additional_loops,
            "bulk_final_npt_ns": bulk_final_npt_ns,
            "salt_molarity_M": salt_molarity_M,
            "solvent_mass_ratio": solvent_mass_ratio,
            "min_salt_pairs": min_salt_pairs,
            "interface_gap_nm": interface_gap_nm,
            "interface_vacuum_nm": interface_vacuum_nm,
            "interface_bottom_thickness_nm": interface_bottom_thickness_nm,
            "interface_top_thickness_nm": interface_top_thickness_nm,
        },
    }
    if restart and workflow_summary_path.exists():
        try:
            existing_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))
            if isinstance(existing_summary, dict):
                existing_summary.update(workflow_summary)
                workflow_summary = existing_summary
        except Exception:
            pass
    workflow_summary["work_dirs"] = {
        "molecules": str(work_dir / "00_molecules"),
        "polymer_bulk": str(work_dir / "ac_CMC"),
        "probe_bulk": str(work_dir / "probe_electrolyte"),
        "electrolyte_bulk": str(work_dir / "ac_electrolyte"),
        "interface_build": str(work_dir / "interface_route_b"),
        "interface_md": str(work_dir / "interface_route_b_md"),
    }
    workflow_summary.setdefault("phases", {})
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")

    ac_CMC_dir = work_dir.child("ac_CMC")
    ac_CMC_build_dir = ac_CMC_dir.child("00_build_cell")
    probe_electrolyte_dir = work_dir.child("probe_electrolyte")
    probe_electrolyte_build_dir = probe_electrolyte_dir.child("00_build_cell")
    ac_electrolyte_dir = work_dir.child("ac_electrolyte")
    ac_electrolyte_build_dir = ac_electrolyte_dir.child("00_build_cell")
    interface_dir = work_dir.child("interface_route_b")
    interface_md_dir = work_dir.child("interface_route_b_md")

    glucose = ff.mol(glucose_smiles)
    glucose = ff.ff_assign(glucose, report=False)
    if not glucose:
        raise RuntimeError("Can not assign force field parameters for glucose.")
    write_mol2(mol=glucose, out_dir=mol2_dir)

    glucose_2 = ff.mol(glucose_2_smiles)
    glucose_2 = ff.ff_assign(glucose_2, report=False)
    if not glucose_2:
        raise RuntimeError("Can not assign force field parameters for glucose_2.")
    write_mol2(mol=glucose_2, out_dir=mol2_dir)

    glucose_3 = ff.mol(glucose_3_smiles)
    glucose_3 = ff.ff_assign(glucose_3, report=False)
    if not glucose_3:
        raise RuntimeError("Can not assign force field parameters for glucose_3.")
    write_mol2(mol=glucose_3, out_dir=mol2_dir)

    glucose_6 = ff.mol(glucose_6_smiles)
    glucose_6 = ff.ff_assign(glucose_6, report=False)
    if not glucose_6:
        raise RuntimeError("Can not assign force field parameters for glucose_6.")
    write_mol2(mol=glucose_6, out_dir=mol2_dir)

    ter1 = utils.mol_from_smiles(ter_smiles)
    if args.term_qm:
        qm.assign_charges(
            ter1,
            charge="RESP",
            opt=True,
            work_dir=work_dir,
            omp=omp_psi4,
            memory=mem_mb,
            log_name=None,
        )
        print("[CONFIG] terminal_charge_mode: RESP via QM")
    else:
        print("[CONFIG] terminal_charge_mode: skipped QM for [H][*] termination")

    cmc_rw_dir = work_dir.child("CMC_rw")
    cmc_term_dir = work_dir.child("CMC_term")
    CMC = poly.random_copolymerize_rw(
        [glucose, glucose_2, glucose_3, glucose_6],
        dp,
        ratio=feed_prob,
        tacticity="atactic",
        name="CMC",
        work_dir=cmc_rw_dir,
    )
    CMC = poly.terminate_rw(CMC, ter1, name="CMC", work_dir=cmc_term_dir)
    CMC = ff.ff_assign(CMC, report=False)
    if not CMC:
        raise RuntimeError("Can not assign force field parameters for CMC.")
    write_mol2(mol=CMC, out_dir=mol2_dir)

    EC = ff.mol(EC_smiles)
    EC = ff.ff_assign(EC, report=False)
    if not EC:
        raise RuntimeError("Can not assign force field parameters for EC.")
    write_mol2(mol=EC, out_dir=mol2_dir)

    DEC = ff.mol(DEC_smiles)
    DEC = ff.ff_assign(DEC, report=False)
    if not DEC:
        raise RuntimeError("Can not assign force field parameters for DEC.")
    write_mol2(mol=DEC, out_dir=mol2_dir)

    EMC = ff.mol(EMC_smiles)
    EMC = ff.ff_assign(EMC, report=False)
    if not EMC:
        raise RuntimeError("Can not assign force field parameters for EMC.")
    write_mol2(mol=EMC, out_dir=mol2_dir)

    Li = ion_ff.mol(Li_smiles)
    Li = ion_ff.ff_assign(Li, report=False)
    if not Li:
        raise RuntimeError("Can not assign force field parameters for Li.")
    write_mol2(mol=Li, out_dir=mol2_dir)

    Na = ion_ff.mol(Na_smiles)
    Na = ion_ff.ff_assign(Na, report=False)
    if not Na:
        raise RuntimeError("Can not assign force field parameters for Na.")
    write_mol2(mol=Na, out_dir=mol2_dir)

    try:
        PF6 = ff.mol(PF6_smiles, charge="RESP", require_ready=True, prefer_db=True)
        PF6 = ff.ff_assign(PF6, bonded="DRIH", report=False)
    except Exception as exc:
        raise RuntimeError(
            "PF6 is expected to be precomputed in MolDB for Example 12. "
            "Please build it first with examples/01_Li_salt/run_pf6_to_moldb.py."
        ) from exc
    if not PF6:
        raise RuntimeError("Can not assign force field parameters for MolDB-backed PF6.")
    write_mol2(mol=PF6, out_dir=mol2_dir)

    q_poly = int(sum(int(atom.GetFormalCharge()) for atom in CMC.GetAtoms()))
    n_Na = int(abs(q_poly) * n_CMC) if q_poly != 0 else 0

    mw_CMC = molecular_weight(CMC, strict=True)
    mw_Na = molecular_weight(Na, strict=True)
    mw_EC = molecular_weight(EC, strict=True)
    mw_DEC = molecular_weight(DEC, strict=True)
    mw_EMC = molecular_weight(EMC, strict=True)
    mw_Li = molecular_weight(Li, strict=True)
    mw_PF6 = molecular_weight(PF6, strict=True)

    ac_CMC = poly.amorphous_cell(
        [CMC, Na],
        [n_CMC, n_Na],
        density=polymer_initial_pack_density_g_cm3,
        retry=polymer_pack_retry,
        retry_step=polymer_pack_retry_step,
        threshold=polymer_pack_threshold_ang,
        dec_rate=polymer_pack_dec_rate,
        charge_scale=[0.8, 0.8],
        neutralize=False,
        work_dir=ac_CMC_build_dir,
    )
    print(format_cell_charge_audit("CMC packed cell", ac_CMC))
    equilibrate_bulk_with_eq21(
        label="CMC",
        ac=ac_CMC,
        work_dir=ac_CMC_dir,
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        additional_loops=bulk_additional_loops,
        final_npt_ns=bulk_final_npt_ns,
        eq21_exec_kwargs=bulk_eq21_exec_kwargs,
    )
    cmc_box_nm = read_equilibrated_box_nm(work_dir=ac_CMC_dir)
    cmc_profile = build_bulk_equilibrium_profile(
        counts=[n_CMC, n_Na],
        mol_weights=[mw_CMC, mw_Na],
        species_names=["CMC", "Na"],
        work_dir=ac_CMC_dir,
    )
    workflow_summary["polymer_bulk"] = {
        "work_dir": str(ac_CMC_dir),
        "box_nm": tuple(round(float(x), 6) for x in cmc_box_nm),
        "density_g_cm3": round(float(cmc_profile.density_g_cm3), 6),
        "counts": {"CMC": int(n_CMC), "Na": int(n_Na)},
    }
    workflow_summary["phases"]["polymer_bulk"] = "done"
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")

    if args.stop_after == "polymer_bulk":
        workflow_summary["status"] = "stopped_after_polymer_bulk"
        workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")
        print("[INFO] stop_after=polymer_bulk reached.")
        raise SystemExit(0)

    probe_interface_prep = plan_probe_polymer_matched_interface_preparation(
        reference_box_nm=cmc_box_nm,
        bottom_thickness_nm=interface_bottom_thickness_nm,
        top_thickness_nm=interface_top_thickness_nm,
        gap_nm=interface_gap_nm,
        surface_shell_nm=interface_surface_shell_nm,
        target_density_g_cm3=electrolyte_probe_density_g_cm3,
        solvent_mol_weights=[mw_EC, mw_DEC, mw_EMC],
        solvent_mass_ratio=solvent_mass_ratio,
        salt_mol_weights=[mw_Li, mw_PF6],
        salt_molarity_M=salt_molarity_M,
        min_salt_pairs=min_salt_pairs,
        solvent_species_names=["EC", "DEC", "EMC"],
        salt_species_names=["Li", "PF6"],
        min_solvent_counts=(1, 1, 1),
        probe_volume_scale=electrolyte_probe_volume_scale,
        initial_pack_density_g_cm3=electrolyte_probe_pack_density_g_cm3,
        z_padding_factor=electrolyte_probe_z_padding_factor,
        is_polyelectrolyte=True,
        minimum_margin_nm=1.0,
        fixed_xy_npt_ns=probe_fixed_xy_npt_ns,
    )
    for note in probe_interface_prep.notes:
        print("[PLAN]", note)
    workflow_summary["probe_plan"] = {
        "probe_box_nm": tuple(round(float(x), 6) for x in probe_interface_prep.probe_prep.probe_box_nm),
        "target_box_nm": tuple(round(float(x), 6) for x in probe_interface_prep.interface_plan.electrolyte_target_box_nm),
        "counts": dict(zip(probe_interface_prep.probe_prep.direct_plan.species_names, probe_interface_prep.probe_prep.direct_plan.target_counts)),
        "notes": list(probe_interface_prep.notes),
    }
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")

    ac_electrolyte_probe = poly.amorphous_cell(
        [EC, DEC, EMC, Li, PF6],
        list(probe_interface_prep.probe_prep.direct_plan.target_counts),
        density=probe_interface_prep.probe_prep.build_density_g_cm3,
        retry=electrolyte_pack_retry,
        retry_step=electrolyte_pack_retry_step,
        threshold=electrolyte_pack_threshold_ang,
        dec_rate=electrolyte_pack_dec_rate,
        charge_scale=[1.0, 1.0, 1.0, 0.8, 0.8],
        neutralize=False,
        work_dir=probe_electrolyte_build_dir,
    )
    print(format_cell_charge_audit("Electrolyte probe packed cell", ac_electrolyte_probe))
    equilibrate_bulk_with_eq21(
        label="Electrolyte probe",
        ac=ac_electrolyte_probe,
        work_dir=probe_electrolyte_dir,
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        additional_loops=bulk_additional_loops,
        final_npt_ns=bulk_final_npt_ns,
        eq21_exec_kwargs=bulk_eq21_exec_kwargs,
    )
    probe_profile = build_bulk_equilibrium_profile(
        counts=list(probe_interface_prep.probe_prep.direct_plan.target_counts),
        mol_weights=[mw_EC, mw_DEC, mw_EMC, mw_Li, mw_PF6],
        species_names=list(probe_interface_prep.probe_prep.direct_plan.species_names),
        work_dir=probe_electrolyte_dir,
    )
    workflow_summary["probe_bulk"] = {
        "work_dir": str(probe_electrolyte_dir),
        "box_nm": tuple(round(float(x), 6) for x in probe_profile.box_nm),
        "density_g_cm3": round(float(probe_profile.density_g_cm3), 6),
    }
    workflow_summary["phases"]["probe_bulk"] = "done"
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")

    if args.stop_after == "probe_bulk":
        workflow_summary["status"] = "stopped_after_probe_bulk"
        workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")
        print("[INFO] stop_after=probe_bulk reached.")
        raise SystemExit(0)

    resized_interface_prep = plan_resized_polymer_matched_interface_from_probe(
        reference_box_nm=cmc_box_nm,
        interface_plan=probe_interface_prep.interface_plan,
        probe_work_dir=probe_electrolyte_dir,
        probe_counts=probe_interface_prep.probe_prep.direct_plan.target_counts,
        mol_weights=[mw_EC, mw_DEC, mw_EMC, mw_Li, mw_PF6],
        species_names=["EC", "DEC", "EMC", "Li", "PF6"],
        solvent_indices=(0, 1, 2),
        salt_pair_indices=(3, 4),
        min_solvent_counts=(1, 1, 1),
        min_salt_pairs=min_salt_pairs,
        pressure_bar=press,
    )
    for note in resized_interface_prep.notes:
        print("[PLAN]", note)
    workflow_summary["resized_plan"] = {
        "target_box_nm": tuple(round(float(x), 6) for x in resized_interface_prep.interface_plan.electrolyte_target_box_nm),
        "initial_pack_box_nm": tuple(round(float(x), 6) for x in resized_interface_prep.resized_prep.pack_plan.initial_pack_box_nm),
        "counts": dict(zip(resized_interface_prep.resized_prep.resize_plan.species_names, resized_interface_prep.resized_prep.resize_plan.target_counts)),
        "notes": list(resized_interface_prep.notes),
    }
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")

    ac_electrolyte = poly.amorphous_cell(
        [EC, DEC, EMC, Li, PF6],
        list(resized_interface_prep.resized_prep.resize_plan.target_counts),
        cell=make_orthorhombic_pack_cell(resized_interface_prep.resized_prep.pack_plan.initial_pack_box_nm),
        density=None,
        retry=electrolyte_pack_retry,
        retry_step=electrolyte_pack_retry_step,
        threshold=electrolyte_pack_threshold_ang,
        dec_rate=electrolyte_pack_dec_rate,
        charge_scale=[1.0, 1.0, 1.0, 0.8, 0.8],
        neutralize=False,
        work_dir=ac_electrolyte_build_dir,
    )
    print(format_cell_charge_audit("Electrolyte packed cell", ac_electrolyte))
    equilibrate_bulk_with_eq21(
        label="Electrolyte",
        ac=ac_electrolyte,
        work_dir=ac_electrolyte_dir,
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        additional_loops=bulk_additional_loops,
        eq21_npt_mdp_overrides=resized_interface_prep.resized_prep.relax_mdp_overrides,
        additional_mdp_overrides=resized_interface_prep.resized_prep.relax_mdp_overrides,
        final_npt_ns=max(
            float(probe_interface_prep.interface_plan.electrolyte_alignment.fixed_xy_npt_ns),
            float(bulk_final_npt_ns),
        ),
        final_npt_mdp_overrides=resized_interface_prep.resized_prep.relax_mdp_overrides,
        eq21_exec_kwargs=bulk_eq21_exec_kwargs,
    )
    electrolyte_profile = build_bulk_equilibrium_profile(
        counts=list(resized_interface_prep.resized_prep.resize_plan.target_counts),
        mol_weights=[mw_EC, mw_DEC, mw_EMC, mw_Li, mw_PF6],
        species_names=list(resized_interface_prep.resized_prep.resize_plan.species_names),
        work_dir=ac_electrolyte_dir,
    )
    workflow_summary["electrolyte_bulk"] = {
        "work_dir": str(ac_electrolyte_dir),
        "box_nm": tuple(round(float(x), 6) for x in electrolyte_profile.box_nm),
        "density_g_cm3": round(float(electrolyte_profile.density_g_cm3), 6),
    }
    workflow_summary["phases"]["electrolyte_bulk"] = "done"
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")

    if args.stop_after == "electrolyte_bulk":
        workflow_summary["status"] = "stopped_after_electrolyte_bulk"
        workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")
        print("[INFO] stop_after=electrolyte_bulk reached.")
        raise SystemExit(0)

    recipe = recommend_polymer_diffusion_interface_recipe(
        interface_plan=resized_interface_prep.interface_plan,
        temperature_k=temp,
        pressure_bar=press,
        vacuum_nm=interface_vacuum_nm,
        core_guard_nm=interface_core_guard_nm,
        top_lateral_shift_fraction=(0.31, 0.57),
        pre_contact_ps=recipe_pre_contact_ps,
        density_relax_ps=recipe_density_relax_ps,
        contact_ps=recipe_contact_ps,
        release_ps=recipe_release_ps,
        exchange_ns=recipe_exchange_ns,
        production_ns=recipe_production_ns,
    )
    for note in recipe.notes:
        print("[ROUTE]", note)
    workflow_summary["interface_recipe"] = {
        "route": recipe.route_spec.route,
        "notes": list(recipe.notes),
        "stages": [stage.name for stage in recipe.protocol.stages()],
    }
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")

    built = InterfaceBuilder(work_dir=interface_dir).build_from_bulk_workdirs(
        name="CMC_vs_LiPF6_electrolyte",
        bottom_name="ac_CMC",
        bottom_work_dir=ac_CMC_dir,
        top_name="ac_electrolyte",
        top_work_dir=ac_electrolyte_dir,
        route=recipe.route_spec,
    )
    workflow_summary["interface_build"] = {
        "system_gro": str(built.system_gro),
        "system_top": str(built.system_top),
        "system_ndx": str(built.system_ndx),
        "system_meta": str(built.system_meta),
    }
    workflow_summary["phases"]["interface_build"] = "done"
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")

    if args.stop_after == "interface_build":
        workflow_summary["status"] = "stopped_after_interface_build"
        workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")
        print("[INFO] stop_after=interface_build reached.")
        print("  GRO:", built.system_gro)
        print("  TOP:", built.system_top)
        print("  NDX:", built.system_ndx)
        raise SystemExit(0)

    final_interface_gro = InterfaceDynamics(built=built, work_dir=interface_md_dir).run(
        protocol=recipe.protocol,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
    )
    workflow_summary["interface_md"] = {
        "work_dir": str(interface_md_dir),
        "final_interface_gro": str(final_interface_gro),
    }
    workflow_summary["phases"]["interface_md"] = "done"
    workflow_summary["status"] = "done"
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2) + "\n", encoding="utf-8")

    print("[INFO] Example 12 finished.")
    print("  CMC formal charge per chain:", q_poly)
    print("  Na counter ions:", n_Na)
    print("  CMC box (nm):", tuple(round(float(x), 4) for x in cmc_box_nm))
    print(
        "  interface XY (nm):",
        tuple(round(float(x), 4) for x in resized_interface_prep.interface_plan.interface_xy_nm),
    )
    print(
        "  probe electrolyte box (nm):",
        tuple(round(float(x), 4) for x in probe_interface_prep.probe_prep.probe_box_nm),
    )
    print(
        "  final electrolyte target box (nm):",
        tuple(round(float(x), 4) for x in resized_interface_prep.interface_plan.electrolyte_target_box_nm),
    )
    print(
        "  probe electrolyte counts:",
        dict(
            zip(
                probe_interface_prep.probe_prep.direct_plan.species_names,
                probe_interface_prep.probe_prep.direct_plan.target_counts,
            )
        ),
    )
    print(
        "  resized electrolyte counts:",
        dict(
            zip(
                resized_interface_prep.resized_prep.resize_plan.species_names,
                resized_interface_prep.resized_prep.resize_plan.target_counts,
            )
        ),
    )
    print("  route:", recipe.route_spec.route)
    print("  staged protocol:", [stage.name for stage in recipe.protocol.stages()])
    print("  GRO:", built.system_gro)
    print("  TOP:", built.system_top)
    print("  NDX:", built.system_ndx)
    print("  final interface GRO:", final_interface_gro)
