from __future__ import annotations

import os
from pathlib import Path

import yadonpy as yp

from _env import env_bool, env_int, env_path

restart = env_bool("YADONPY_RESTART", True)
yp.set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
PROFILE = os.environ.get("YADONPY_PROFILE", "full").strip().lower()
SMOKE = PROFILE == "smoke"
MATRIX_FAST = SMOKE and env_bool("YADONPY_MATRIX_FAST", False)
ROUTE = os.environ.get("YADONPY_ROUTE", "screening" if SMOKE else "production").strip().lower()
RUN_NAME = "05_cmcna_glucose6_periodic_smoke" if SMOKE else "05_cmcna_glucose6_periodic_case"
work_dir = env_path("YADONPY_WORK_DIR", BASE_DIR / "work_dir" / RUN_NAME)


if __name__ == "__main__":
    yp.doctor(print_report=True)
    ff = yp.get_ff("gaff2_mod")
    ion_ff = yp.get_ff("merz")

    graphite = yp.GraphiteSubstrateSpec(
        nx=(8 if MATRIX_FAST else (10 if SMOKE else 16)),
        ny=(8 if MATRIX_FAST else (10 if SMOKE else 14)),
        n_layers=4,
        edge_cap="periodic",
        name="GRAPH",
        top_padding_ang=15.0,
    )
    polymer = yp.default_cmcna_polymer_spec(
        name="CMC6",
        monomers=(
            yp.MoleculeSpec(
                name="glucose_6",
                smiles="*OC1OC(COCC(=O)[O-])C(*)C(O)C1O",
                prefer_db=True,
                require_ready=True,
                polyelectrolyte_mode=True,
            ),
        ),
        monomer_ratio=(1.0,),
        dp=(12 if MATRIX_FAST else (20 if SMOKE else 50)),
        chain_count=(2 if MATRIX_FAST else (None if SMOKE else 8)),
        target_density_g_cm3=1.50,
        slab_z_nm=(3.0 if MATRIX_FAST else (4.2 if SMOKE else 5.0)),
        min_chain_count=2,
        initial_pack_z_scale=(1.80 if MATRIX_FAST else (1.55 if SMOKE else 1.40)),
        pack_retry=(36 if MATRIX_FAST else 80),
        pack_retry_step=(900 if MATRIX_FAST else 3600),
        pack_threshold_ang=(1.35 if MATRIX_FAST else 1.60),
        pack_dec_rate=0.68,
        counterion=yp.MoleculeSpec(name="Na", smiles="[Na+]", use_ion_ff=True, charge_scale=1.0),
    )
    electrolyte = yp.default_carbonate_lipf6_electrolyte_spec(
        solvents=(
            yp.MoleculeSpec(name="EC", smiles="O=C1OCCO1", prefer_db=True, require_ready=True),
            yp.MoleculeSpec(name="EMC", smiles="CCOC(=O)OC", prefer_db=True, require_ready=True),
            yp.MoleculeSpec(name="DEC", smiles="CCOC(=O)OCC", prefer_db=True, require_ready=True),
        ),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        salt_anion=yp.MoleculeSpec(
            name="PF6",
            smiles="F[P-](F)(F)(F)(F)F",
            bonded="DRIH",
            charge_scale=0.8,
            prefer_db=True,
            require_ready=True,
        ),
        slab_z_nm=(3.2 if MATRIX_FAST else (4.6 if SMOKE else 5.4)),
        min_salt_pairs=(2 if MATRIX_FAST else (4 if SMOKE else 8)),
        target_density_g_cm3=1.32,
        initial_pack_density_g_cm3=(0.72 if MATRIX_FAST else 0.86),
        pack_retry=(28 if MATRIX_FAST else 44),
        pack_retry_step=(900 if MATRIX_FAST else 2800),
        pack_threshold_ang=(1.35 if MATRIX_FAST else 1.55),
        pack_dec_rate=0.70,
    )
    relax = yp.SandwichRelaxationSpec(
        omp=env_int("YADONPY_OMP", 16 if SMOKE else 24),
        gpu=env_int("YADONPY_GPU", 1),
        gpu_id=env_int("YADONPY_GPU_ID", 0),
        psi4_omp=env_int("YADONPY_PSI4_OMP", 24 if SMOKE else 36),
        psi4_memory_mb=env_int("YADONPY_PSI4_MEMORY_MB", 24000 if SMOKE else 32000),
        bulk_eq21_final_ns=(0.02 if MATRIX_FAST else (0.06 if SMOKE else 0.12)),
        bulk_additional_loops=(0 if MATRIX_FAST else 1),
        stacked_pre_nvt_ps=(4.0 if MATRIX_FAST else (10.0 if SMOKE else 20.0)),
        stacked_z_relax_ps=(8.0 if MATRIX_FAST else (40.0 if SMOKE else 100.0)),
        stacked_exchange_ps=(12.0 if MATRIX_FAST else (60.0 if SMOKE else 160.0)),
    )

    graphite_stage = yp.prepare_graphite_substrate(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite,
        polymer=polymer,
        electrolyte=electrolyte,
        relax=relax,
        route=ROUTE,
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
        route=ROUTE,
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
        route=ROUTE,
        restart=restart,
    )
    result = yp.release_graphite_cmc_electrolyte_stack(
        work_dir=work_dir,
        ff=ff,
        graphite=graphite_stage,
        polymer_interphase=cmc_interphase,
        electrolyte_interphase=top_interphase,
        relax=relax,
        route=ROUTE,
        restart=restart,
    )
    yp.print_interface_result_summary(result, profile=PROFILE)
