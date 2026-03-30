from __future__ import annotations

from pathlib import Path

from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2_mod, MERZ
from yadonpy.interface import (
    GraphiteSubstrateSpec,
    SandwichRelaxationSpec,
    build_graphite_cmcna_electrolyte_sandwich,
    default_carbonate_lipf6_electrolyte_spec,
    default_cmcna_polymer_spec,
)
from yadonpy.runtime import set_run_options


restart = True
set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir" / "03_cmcna_smoke"


if __name__ == "__main__":
    doctor(print_report=True)
    ff = GAFF2_mod()
    ion_ff = MERZ()

    result = build_graphite_cmcna_electrolyte_sandwich(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=GraphiteSubstrateSpec(nx=4, ny=4, n_layers=2, edge_cap="H", name="GRAPH"),
        polymer=default_cmcna_polymer_spec(
            dp=24,
            min_chain_count=2,
            slab_z_nm=3.8,
            target_density_g_cm3=1.38,
            initial_pack_z_scale=1.30,
            pack_retry=36,
            pack_retry_step=2200,
            pack_threshold_ang=1.60,
        ),
        electrolyte=default_carbonate_lipf6_electrolyte_spec(
            slab_z_nm=4.0,
            min_salt_pairs=2,
            target_density_g_cm3=1.28,
            initial_pack_density_g_cm3=0.82,
            pack_retry=34,
            pack_retry_step=1800,
        ),
        relax=SandwichRelaxationSpec(
            omp=16,
            gpu=1,
            gpu_id=0,
            psi4_omp=36,
            psi4_memory_mb=24000,
            bulk_eq21_final_ns=0.0,
            bulk_additional_loops=0,
            bulk_eq21_exec_kwargs={
                "eq21_tmax": 600.0,
                "eq21_pmax": 10000.0,
                "eq21_pre_nvt_ps": 2.0,
                "sim_time": 0.04,
                "eq21_npt_time_scale": 0.20,
            },
            stacked_pre_nvt_ps=10.0,
            stacked_z_relax_ps=40.0,
            stacked_exchange_ps=60.0,
        ),
        restart=restart,
    )

    print("manifest_path =", result.manifest_path)
    print("relaxed_gro =", result.relaxed_gro)
    print("polymer_density_g_cm3 =", round(result.polymer_phase.density_g_cm3, 4))
    print("electrolyte_density_g_cm3 =", round(result.electrolyte_phase.density_g_cm3, 4))
    print("stack_checks =", result.stack_checks)
