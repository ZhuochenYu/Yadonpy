from __future__ import annotations

import os
from pathlib import Path

from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2_mod, MERZ
from yadonpy.interface import (
    GraphiteSubstrateSpec,
    MoleculeSpec,
    SandwichRelaxationSpec,
    build_graphite_cmcna_electrolyte_sandwich,
    default_carbonate_lipf6_electrolyte_spec,
    default_cmcna_polymer_spec,
)
from yadonpy.runtime import set_run_options


restart = True
set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
PROFILE = os.environ.get("YADONPY_PROFILE", "full").strip().lower()
SMOKE = PROFILE == "smoke"
RUN_NAME = "05_cmcna_glucose6_periodic_smoke" if SMOKE else "05_cmcna_glucose6_periodic_case"
work_dir = BASE_DIR / "work_dir" / RUN_NAME


if __name__ == "__main__":
    doctor(print_report=True)
    ff = GAFF2_mod()
    ion_ff = MERZ()

    result = build_graphite_cmcna_electrolyte_sandwich(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=GraphiteSubstrateSpec(
            nx=(10 if SMOKE else 16),
            ny=(10 if SMOKE else 14),
            n_layers=4,
            edge_cap="periodic",
            name="GRAPH",
            top_padding_ang=15.0,
        ),
        polymer=default_cmcna_polymer_spec(
            name="CMC6",
            monomers=(
                MoleculeSpec(
                    name="glucose_6",
                    smiles="*OC1OC(COCC(=O)[O-])C(*)C(O)C1O",
                    prefer_db=True,
                    require_ready=False,
                ),
            ),
            monomer_ratio=(1.0,),
            dp=(20 if SMOKE else 50),
            chain_count=(2 if SMOKE else 8),
            target_density_g_cm3=1.45,
            slab_z_nm=(4.2 if SMOKE else 5.0),
            min_chain_count=2,
            initial_pack_z_scale=1.28,
            pack_retry=80,
            pack_retry_step=3600,
            pack_threshold_ang=1.60,
            pack_dec_rate=0.68,
            counterion=MoleculeSpec(name="Na", smiles="[Na+]", use_ion_ff=True, charge_scale=1.0),
        ),
        electrolyte=default_carbonate_lipf6_electrolyte_spec(
            solvents=(
                MoleculeSpec(name="EC", smiles="O=C1OCCO1", prefer_db=True, require_ready=False),
                MoleculeSpec(name="EMC", smiles="CCOC(=O)OC", prefer_db=True, require_ready=False),
                MoleculeSpec(name="DEC", smiles="CCOC(=O)OCC", prefer_db=True, require_ready=False),
            ),
            solvent_mass_ratio=(3.0, 2.0, 5.0),
            salt_anion=MoleculeSpec(
                name="PF6",
                smiles="F[P-](F)(F)(F)(F)F",
                bonded="DRIH",
                charge_scale=0.8,
                prefer_db=True,
                require_ready=False,
            ),
            slab_z_nm=(4.6 if SMOKE else 5.4),
            min_salt_pairs=(4 if SMOKE else 8),
            target_density_g_cm3=1.32,
            initial_pack_density_g_cm3=0.86,
            pack_retry=44,
            pack_retry_step=2800,
            pack_threshold_ang=1.55,
            pack_dec_rate=0.70,
        ),
        relax=SandwichRelaxationSpec(
            omp=(16 if SMOKE else 24),
            gpu=1,
            gpu_id=0,
            psi4_omp=(24 if SMOKE else 36),
            psi4_memory_mb=(24000 if SMOKE else 32000),
            bulk_eq21_final_ns=(0.06 if SMOKE else 0.12),
            bulk_additional_loops=1,
            stacked_pre_nvt_ps=(10.0 if SMOKE else 20.0),
            stacked_z_relax_ps=(40.0 if SMOKE else 100.0),
            stacked_exchange_ps=(60.0 if SMOKE else 160.0),
        ),
        restart=restart,
    )

    print("profile =", PROFILE)
    print("manifest_path =", result.manifest_path)
    print("relaxed_gro =", result.relaxed_gro)
    print("polymer_density_g_cm3 =", round(result.polymer_phase.density_g_cm3, 4))
    print("electrolyte_density_g_cm3 =", round(result.electrolyte_phase.density_g_cm3, 4))
    print("stack_checks =", result.stack_checks)
