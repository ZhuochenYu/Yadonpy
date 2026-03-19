from __future__ import annotations

# Standalone PF6- test script (YadonPy style)
# Goal: SMILES -> (OpenBabel 3D) -> QM opt+RESP -> FF assign -> write .itp/.top/.gro (+mol2)
# Stops BEFORE packing any simulation box.

import shutil
from pathlib import Path

from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.core import utils
from yadonpy.sim import qm
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.io.mol2 import write_mol2_from_rdkit
from yadonpy.io.gromacs_molecule import write_gromacs_single_molecule_topology


def main():
    # ---------------- user inputs ----------------
    restart_status = False

    smiles_pf6 = "F[P-](F)(F)(F)(F)F"

    # NOTE on naming:
    #   You may set an explicit name at creation time (recommended for scripts),
    #   or simply rely on variable-name inference downstream.

    # Psi4 settings (adjust to your machine)
    omp_psi4 = 8
    mem_mb = 8000

    # Output folder
    work_dir = Path("work_pf6_only").resolve()
    # ------------------------------------------------

    doctor(print_report=True)
    ensure_initialized()

    if not restart_status and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    mol2_dir = work_dir / "00_molecules"
    out_dir = work_dir / "01_basic_top_pf6"
    mol2_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build PF6- with OpenBabel-preferred 3D coords
    PF6 = utils.mol_from_smiles(smiles_pf6, coord=False, name="PF6")
    utils.ensure_3d_coords(PF6, smiles_hint=smiles_pf6, engine="openbabel")

    # 2) QM OPT + RESP
    #    For inorganic anions, YadonPy defaults are aligned to RadonPy:
    #      OPT  : wb97m-d3bj / 6-31+G(d,p)
    #      RESP : wb97m-d3bj / 6-311+G(2d,p)
    #    - Also auto-generates bond+angle params via modified Seminario
    #      and will patch them into the final .itp (robust for PF6-/BF4-/ClO4-...).
    qm.assign_charges(
        PF6,
        charge="RESP",
        opt=True,
        work_dir=work_dir,
        log_name=None,
        omp=omp_psi4,
        memory=mem_mb,
        total_charge=-1,
        total_multiplicity=1,
        symmetrize=True,
        auto_level=True,
        bonded_params="auto",
    )

    # 3) Assign GAFF2-family parameters (default: GAFF2_mod)
    ff = GAFF2_mod()
    result = ff.ff_assign(PF6)
    if not result:
        raise RuntimeError('FF assignment failed for PF6')
    # 4) Write artifacts
    mol2_path = write_mol2_from_rdkit(mol=PF6, out_dir=mol2_dir)

    gro_path, itp_path, top_path = write_gromacs_single_molecule_topology(
        PF6,
        out_dir=out_dir,
        mol_name="PF6",
    )

    print("\n[DONE] PF6- basic topology generated:")
    print(f"  mol2: {mol2_path}")
    print(f"  gro : {gro_path}")
    print(f"  itp : {itp_path}")
    print(f"  top : {top_path}")

    print("\nNotes:")
    print("  - This script stops here (no pack/box build).")
    print("  - For visualization, prefer using the .gro with the .itp/.top.\n")


if __name__ == "__main__":
    main()
