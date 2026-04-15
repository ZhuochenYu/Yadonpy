from __future__ import annotations

import os
from pathlib import Path

import yadonpy as yp

from _env import env_bool, env_int, env_path

restart = env_bool("YADONPY_RESTART", True)
route = os.environ.get("YADONPY_ROUTE", "screening").strip().lower()
matrix_fast = env_bool("YADONPY_MATRIX_FAST", False)
yp.set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
work_dir = env_path("YADONPY_WORK_DIR", BASE_DIR / "work_dir" / "03_cmcna_smoke")


if __name__ == "__main__":
    yp.doctor(print_report=True)
    ff = yp.get_ff("gaff2_mod")
    ion_ff = yp.get_ff("merz")

    graphite = yp.GraphiteSubstrateSpec(
        nx=(3 if matrix_fast else 4),
        ny=(3 if matrix_fast else 4),
        n_layers=2,
        edge_cap="H",
        name="GRAPH",
    )
    polymer = yp.default_cmcna_polymer_spec(
        dp=(12 if matrix_fast else 24),
        target_density_g_cm3=1.50,
        slab_z_nm=(2.8 if matrix_fast else 3.8),
        min_chain_count=2,
        initial_pack_z_scale=(1.70 if matrix_fast else 1.30),
        pack_retry=(24 if matrix_fast else 36),
        pack_retry_step=(800 if matrix_fast else 2200),
        pack_threshold_ang=(1.35 if matrix_fast else 1.60),
        pack_dec_rate=0.60,
    )
    electrolyte = yp.default_carbonate_lipf6_electrolyte_spec(
        slab_z_nm=(3.0 if matrix_fast else 4.0),
        min_salt_pairs=2,
        target_density_g_cm3=1.28,
        initial_pack_density_g_cm3=(0.72 if matrix_fast else 0.82),
        pack_retry=(24 if matrix_fast else 34),
        pack_retry_step=(800 if matrix_fast else 1800),
    )
    relax = yp.SandwichRelaxationSpec(
        omp=env_int("YADONPY_OMP", 16),
        gpu=env_int("YADONPY_GPU", 1),
        gpu_id=env_int("YADONPY_GPU_ID", 0),
        psi4_omp=env_int("YADONPY_PSI4_OMP", 36),
        psi4_memory_mb=env_int("YADONPY_PSI4_MEMORY_MB", 24000),
        bulk_eq21_final_ns=0.0,
        bulk_additional_loops=0,
        bulk_eq21_exec_kwargs={
            "eq21_tmax": 600.0,
            "eq21_pmax": 10000.0,
            "eq21_pre_nvt_ps": 2.0,
            "sim_time": (0.02 if matrix_fast else 0.04),
            "eq21_npt_time_scale": (0.10 if matrix_fast else 0.20),
        },
        stacked_pre_nvt_ps=(4.0 if matrix_fast else 10.0),
        stacked_z_relax_ps=(8.0 if matrix_fast else 40.0),
        stacked_exchange_ps=(12.0 if matrix_fast else 60.0),
    )

    graphite_stage = yp.prepare_graphite_substrate(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite,
        polymer=polymer,
        electrolyte=electrolyte,
        relax=relax,
        route=route,
        restart=restart,
    )
    polymer_bulk = yp.calibrate_polymer_bulk_phase(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite_stage,
        polymer=polymer,
        relax=relax,
        restart=restart,
    )
    electrolyte_bulk = yp.calibrate_electrolyte_bulk_phase(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite_stage,
        electrolyte=electrolyte,
        relax=relax,
        restart=restart,
    )
    cmc_interphase = yp.build_graphite_cmc_interphase(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite_stage,
        polymer=polymer,
        polymer_bulk=polymer_bulk,
        relax=relax,
        route=route,
        restart=restart,
    )
    top_interphase = yp.build_cmc_electrolyte_interphase(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite_stage,
        electrolyte=electrolyte,
        electrolyte_bulk=electrolyte_bulk,
        relax=relax,
        route=route,
        restart=restart,
    )
    result = yp.release_graphite_cmc_electrolyte_stack(
        work_dir=work_dir,
        ff=ff,
        graphite=graphite_stage,
        polymer_interphase=cmc_interphase,
        electrolyte_interphase=top_interphase,
        relax=relax,
        route=route,
        restart=restart,
    )
    yp.print_interface_result_summary(result, profile="smoke")
