from __future__ import annotations

from pathlib import Path

from yadonpy.core.workdir import workdir
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2_mod, MERZ
from yadonpy.interface import (
    ElectrolyteSlabSpec,
    GraphiteSubstrateSpec,
    MoleculeSpec,
    PolymerSlabSpec,
    SandwichRelaxationSpec,
    build_graphite_peo_electrolyte_sandwich,
)


def main() -> None:
    base = Path(__file__).resolve().parent.parent / "remote_runs" / "graphite_peo_electrolyte_smoke_fast"
    wd = workdir(base, restart=True)
    ff = GAFF2_mod()
    ion_ff = MERZ()

    result = build_graphite_peo_electrolyte_sandwich(
        work_dir=wd,
        ff=ff,
        ion_ff=ion_ff,
        graphite=GraphiteSubstrateSpec(nx=4, ny=4, n_layers=2, edge_cap="H", name="GRAPH"),
        polymer=PolymerSlabSpec(
            name="PEO",
            monomer_smiles="*CCO*",
            terminal_smiles="[H][*]",
            chain_target_atoms=220,
            target_density_g_cm3=1.08,
            slab_z_nm=3.2,
            min_chain_count=2,
        ),
        electrolyte=ElectrolyteSlabSpec(
            solvents=(MoleculeSpec(name="DME", smiles="COCCOC"),),
            salt_cation=MoleculeSpec(name="Li", smiles="[Li+]", use_ion_ff=True, charge_scale=0.8),
            salt_anion=MoleculeSpec(
                name="FSI",
                smiles="FS(=O)(=O)[N-]S(=O)(=O)F",
                charge_scale=0.8,
            ),
            solvent_mass_ratio=(1.0,),
            target_density_g_cm3=1.12,
            slab_z_nm=3.6,
            salt_molarity_M=1.0,
            min_salt_pairs=2,
            initial_pack_density_g_cm3=0.82,
        ),
        relax=SandwichRelaxationSpec(
            temperature_k=300.0,
            pressure_bar=1.0,
            mpi=1,
            omp=16,
            gpu=1,
            gpu_id=0,
            psi4_omp=36,
            psi4_memory_mb=24000,
            bulk_eq21_final_ns=0.0,
            bulk_additional_loops=0,
            stacked_pre_nvt_ps=10.0,
            stacked_z_relax_ps=40.0,
            stacked_exchange_ps=60.0,
        ),
        restart=True,
    )

    print("manifest_path =", result.manifest_path)
    print("relaxed_gro =", result.relaxed_gro)
    print("polymer_density_g_cm3 =", round(result.polymer_phase.density_g_cm3, 4))
    print("polymer_target_density_g_cm3 =", round(float(result.polymer_phase.target_density_g_cm3 or 0.0), 4))
    print("electrolyte_density_g_cm3 =", round(result.electrolyte_phase.density_g_cm3, 4))
    print("electrolyte_target_density_g_cm3 =", round(float(result.electrolyte_phase.target_density_g_cm3 or 0.0), 4))


if __name__ == "__main__":
    doctor(print_report=True)
    main()
