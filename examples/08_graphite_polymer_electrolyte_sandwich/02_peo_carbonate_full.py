from __future__ import annotations

import os
from pathlib import Path

import yadonpy as yp

from _env import env_bool, env_int, env_path

restart = env_bool("YADONPY_RESTART", True)
route = os.environ.get("YADONPY_ROUTE", "production").strip().lower()
yp.set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
work_dir = env_path("YADONPY_WORK_DIR", BASE_DIR / "work_dir" / "02_peo_carbonate_full")


if __name__ == "__main__":
    yp.doctor(print_report=True)
    ff = yp.get_ff("gaff2_mod")
    ion_ff = yp.get_ff("merz")

    graphite = yp.GraphiteSubstrateSpec(
        nx=6,
        ny=5,
        n_layers=3,
        edge_cap="H",
        name="GRAPH",
    )
    polymer = yp.default_peo_polymer_spec(
        name="PEO",
        chain_target_atoms=320,
        min_chain_count=3,
        target_density_g_cm3=1.08,
        slab_z_nm=4.0,
        initial_pack_z_scale=1.20,
        pack_retry=48,
        pack_retry_step=2600,
    )
    electrolyte = yp.default_carbonate_lipf6_electrolyte_spec(
        slab_z_nm=4.8,
        min_salt_pairs=4,
        target_density_g_cm3=1.30,
        initial_pack_density_g_cm3=0.86,
        pack_retry=44,
        pack_retry_step=2400,
    )
    relax = yp.SandwichRelaxationSpec(
        omp=env_int("YADONPY_OMP", 16),
        gpu=env_int("YADONPY_GPU", 1),
        gpu_id=env_int("YADONPY_GPU_ID", 0),
        psi4_omp=env_int("YADONPY_PSI4_OMP", 24),
        psi4_memory_mb=env_int("YADONPY_PSI4_MEMORY_MB", 24000),
        bulk_eq21_final_ns=0.10,
        bulk_additional_loops=1,
        stacked_pre_nvt_ps=20.0,
        stacked_z_relax_ps=80.0,
        stacked_exchange_ps=120.0,
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
    polymer_interphase = yp.build_graphite_polymer_interphase(
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
    electrolyte_interphase = yp.build_polymer_electrolyte_interphase(
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
    result = yp.release_graphite_polymer_electrolyte_stack(
        work_dir=work_dir,
        ff=ff,
        graphite=graphite_stage,
        polymer_interphase=polymer_interphase,
        electrolyte_interphase=electrolyte_interphase,
        relax=relax,
        route=route,
        restart=restart,
    )
    yp.print_interface_result_summary(result, profile="full")
