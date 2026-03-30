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
work_dir = BASE_DIR / "work_dir" / "04_cmcna_full"


if __name__ == "__main__":
    doctor(print_report=True)
    ff = GAFF2_mod()
    ion_ff = MERZ()

    result = build_graphite_cmcna_electrolyte_sandwich(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=GraphiteSubstrateSpec(nx=6, ny=5, n_layers=3, edge_cap="H", name="GRAPH"),
        polymer=default_cmcna_polymer_spec(
            dp=60,
            target_density_g_cm3=1.42,
            slab_z_nm=4.6,
            min_chain_count=2,
            initial_pack_z_scale=1.34,
            pack_retry=60,
            pack_retry_step=3200,
            pack_threshold_ang=1.60,
            pack_dec_rate=0.68,
        ),
        electrolyte=default_carbonate_lipf6_electrolyte_spec(
            slab_z_nm=5.2,
            min_salt_pairs=4,
            target_density_g_cm3=1.30,
            initial_pack_density_g_cm3=0.86,
            pack_retry=44,
            pack_retry_step=2400,
        ),
        relax=SandwichRelaxationSpec(
            omp=16,
            gpu=1,
            gpu_id=0,
            psi4_omp=24,
            psi4_memory_mb=24000,
            bulk_eq21_final_ns=0.10,
            bulk_additional_loops=1,
            stacked_pre_nvt_ps=20.0,
            stacked_z_relax_ps=80.0,
            stacked_exchange_ps=120.0,
        ),
        restart=restart,
    )

    print("manifest_path =", result.manifest_path)
    print("relaxed_gro =", result.relaxed_gro)
    print("polymer_density_g_cm3 =", round(result.polymer_phase.density_g_cm3, 4))
    print("electrolyte_density_g_cm3 =", round(result.electrolyte_phase.density_g_cm3, 4))
    print("stack_checks =", result.stack_checks)
