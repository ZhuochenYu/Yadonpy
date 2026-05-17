from __future__ import annotations

"""Example 12-01: umbrella PMF for solvated Li+ entering CMC-Na."""

import os
from pathlib import Path

from yadonpy.core import poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.interface import (
    GraphiteLayerSpec,
    LayerStackRelaxationSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    SolvatedIonUmbrellaSpec,
    ZCompressionAnnealSpec,
    analyze_umbrella_pmf,
    build_layer_stack,
    prepare_solvated_ion_umbrella,
    run_layer_stack_relaxation,
    run_solvated_ion_umbrella,
)
from yadonpy.runtime import set_run_options


def _env_flag(name: str, default: bool = False) -> bool:
    token = str(os.environ.get(name, "")).strip().lower()
    if not token:
        return bool(default)
    return token in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    return int(raw) if raw else int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    return float(raw) if raw else float(default)


# ---------------- user inputs ----------------
restart_status = _env_flag("YADONPY_RESTART", True)
set_run_options(restart=restart_status)

temp = _env_float("YADONPY_TEMP_K", 318.15)
mpi = _env_int("YADONPY_MPI", 1)
omp = _env_int("YADONPY_OMP", 14)
gpu = _env_int("YADONPY_GPU", 1)
gpu_id = os.environ.get("YADONPY_GPU_ID", "0").strip() or None

analysis_only = _env_flag("YADONPY_ANALYSIS_ONLY", False)
skip_sampling = _env_flag("YADONPY_PREPARE_ONLY", False)

relax_ns = _env_float("YADONPY_RELAX_NS", 2.0)
umbrella_windows = _env_int("YADONPY_UMBRELLA_WINDOWS", 31)
umbrella_steering_ns = _env_float("YADONPY_UMBRELLA_STEERING_NS", 0.50)
umbrella_window_eq_ns = _env_float("YADONPY_UMBRELLA_WINDOW_EQ_NS", 0.20)
umbrella_window_ns = _env_float("YADONPY_UMBRELLA_WINDOW_NS", 1.00)
umbrella_k = _env_float("YADONPY_UMBRELLA_K", 1000.0)
wham_skip_ps = _env_float("YADONPY_WHAM_SKIP_PS", 200.0)
wham_bins = _env_int("YADONPY_WHAM_BINS", 200)

ff = GAFF2_mod()
ion_ff = MERZ()

glucose6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"
ter_smiles = "[H][*]"
EC_smiles = "O=C1OCCO1"
EMC_smiles = "CCOC(=O)OC"
DEC_smiles = "CCOC(=O)OCC"
Li_smiles = "[Li+]"
Na_smiles = "[Na+]"
PF6_smiles = "F[P-](F)(F)(F)(F)F"

cmc_dp = _env_int("YADONPY_CMC_DP", 5)
cmc_chain_count = _env_int("YADONPY_CMC_CHAINS", 1)
solvent_counts = (
    _env_int("YADONPY_EC_COUNT", 8),
    _env_int("YADONPY_EMC_COUNT", 6),
    _env_int("YADONPY_DEC_COUNT", 14),
)
salt_pairs = _env_int("YADONPY_SALT_PAIRS", 3)
charge_scale = _env_float("YADONPY_CHARGE_SCALE", 0.7)

BASE_DIR = Path(__file__).resolve().parent
work_dir = Path(os.environ.get("YADONPY_WORK_DIR", BASE_DIR / "work_dir" / "li_to_cmc_umbrella_pmf"))


def _build_and_relax_interface(root_dir: Path):
    cmc_rw_dir = root_dir / "00_interface_build" / "00_cmc_rw"
    cmc_term_dir = root_dir / "00_interface_build" / "01_cmc_term"

    glucose6 = ff.mol(glucose6_smiles, charge="RESP", prefer_db=True, require_ready=True, polyelectrolyte_mode=True)
    glucose6 = ff.ff_assign(glucose6, report=False)
    ter = utils.mol_from_smiles(ter_smiles)
    EC = ff.mol(EC_smiles, charge="RESP", prefer_db=True, require_ready=True)
    EC = ff.ff_assign(EC, report=False)
    EMC = ff.mol(EMC_smiles, charge="RESP", prefer_db=True, require_ready=True)
    EMC = ff.ff_assign(EMC, report=False)
    DEC = ff.mol(DEC_smiles, charge="RESP", prefer_db=True, require_ready=True)
    DEC = ff.ff_assign(DEC, report=False)
    PF6 = ff.mol(PF6_smiles, charge="RESP", prefer_db=True, require_ready=True)
    PF6 = ff.ff_assign(PF6, bonded="DRIH", report=False)
    Li = ion_ff.mol(Li_smiles)
    Li = ion_ff.ff_assign(Li, report=False)
    Na = ion_ff.mol(Na_smiles)
    Na = ion_ff.ff_assign(Na, report=False)
    if not all((glucose6, ter, EC, EMC, DEC, PF6, Li, Na)):
        raise RuntimeError("MolDB/FF assignment failed for one or more CMC/electrolyte species.")

    CMC = poly.random_copolymerize_rw([glucose6], cmc_dp, ratio=[1.0], tacticity="atactic", work_dir=cmc_rw_dir)
    CMC = poly.terminate_rw(CMC, ter, work_dir=cmc_term_dir)
    CMC = ff.ff_assign(CMC, report=False)
    if not CMC:
        raise RuntimeError("Cannot assign GAFF2_mod parameters to the CMC-Na chain.")

    graphite = GraphiteLayerSpec(
        name="GRAPHITE_BASAL",
        nx=6,
        ny=5,
        n_layers=3,
        orientation="basal",
        periodic_xy=True,
    )
    cmcna = MolecularLayerSpec(
        name="CMCNA",
        species=(CMC, Na),
        counts=(cmc_chain_count, cmc_chain_count * cmc_dp),
        thickness_nm=1.8,
        density_target_g_cm3=1.0,
        layer_kind="cmcna",
        charge_scale=(charge_scale, charge_scale),
        polyelectrolyte_mode=True,
        counterion_contact_mode="carboxylate",
    )
    electrolyte = MolecularLayerSpec(
        name="ELECTROLYTE",
        species=(EC, EMC, DEC, Li, PF6),
        counts=(*solvent_counts, salt_pairs, salt_pairs),
        thickness_nm=2.2,
        density_target_g_cm3=1.2,
        layer_kind="electrolyte",
        charge_scale=(1.0, 1.0, 1.0, charge_scale, charge_scale),
    )
    stack = LayerStackSpec(
        layers=(graphite, cmcna, electrolyte),
        order="bottom_to_top",
        pbc_mode="xyz",
        name="eg12_graphite_cmcna_electrolyte_li_umbrella",
        default_gap_nm=0.35,
        molecular_packing_expand="z",
    )
    relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=relax_ns)
    built = build_layer_stack(
        stack=stack,
        relaxation=relaxation,
        work_dir=root_dir / "00_interface_build",
        restart=restart_status,
    )
    relaxed = run_layer_stack_relaxation(
        built,
        work_dir=root_dir / "01_relaxation",
        time_ns=relax_ns,
        temp=temp,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=None if gpu_id is None else int(gpu_id),
        run_analysis=True,
        relax_z=True,
        compression_anneal=ZCompressionAnnealSpec(
            enabled=True,
            cycles=8,
            tmax_K=380.0,
            pmax_bar=3000.0,
            max_z_shrink_per_cycle=0.04,
        ),
        restart=restart_status,
    )
    return built, relaxed


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()
    root = Path(workdir(work_dir, restart=restart_status))

    if analysis_only:
        plan = root / "02_solvated_li_selection" / "umbrella_sampling_manifest.json"
        result = analyze_umbrella_pmf(plan)
        print(f"umbrella_pmf_summary = {result.summary_path}")
        raise SystemExit(0)

    built, relaxed = _build_and_relax_interface(root)
    print(f"layer_stack_manifest = {built.manifest_path}")
    print(f"relaxation_summary = {relaxed.summary_path}")

    umbrella_spec = SolvatedIonUmbrellaSpec(
        target_group="CMCNA",
        target_coordination_number=4,
        target_offset_nm=0.0,
        steering_ns=umbrella_steering_ns,
        window_count=umbrella_windows,
        window_equilibration_ns=umbrella_window_eq_ns,
        window_production_ns=umbrella_window_ns,
        temperature_K=temp,
        dt_ps=0.001,
        constraints="none",
        umbrella_k_kj_mol_nm2=umbrella_k,
        wham_skip_ps=wham_skip_ps,
        wham_bins=wham_bins,
    )
    plan = prepare_solvated_ion_umbrella(
        system_dir=relaxed.work_dir / "02_system",
        gro_path=relaxed.final_gro,
        top_path=relaxed.work_dir / "02_system" / "system.top",
        ndx_path=relaxed.work_dir / "02_system" / "system.ndx",
        manifest_path=built.manifest_path,
        out_dir=root,
        spec=umbrella_spec,
    )
    print(f"umbrella_manifest = {plan.manifest_path}")
    print(f"selected_li_atom = {plan.selected_center_atom}")
    print(f"window_count = {len(plan.windows)}")

    if skip_sampling:
        print("YADONPY_PREPARE_ONLY is set; umbrella inputs were generated but MD was not launched.")
        raise SystemExit(0)

    pmf = run_solvated_ion_umbrella(
        plan,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        restart=restart_status,
    )
    print(f"umbrella_pmf_summary = {pmf.summary_path}")
    print(f"pmf_csv = {pmf.pmf_csv}")
