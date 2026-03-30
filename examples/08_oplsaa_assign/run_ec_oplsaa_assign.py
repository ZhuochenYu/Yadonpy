from __future__ import annotations

from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.ff import OPLSAA


def main():
    set_run_options(restart=False)

    smiles_ec = "O=C1OCCO1"

    from rdkit import Chem

    EC = Chem.AddHs(Chem.MolFromSmiles(smiles_ec))

    ff = OPLSAA()
    EC = ff.ff_assign(EC, charge="opls")
    if not EC:
        raise RuntimeError("OPLS-AA assignment failed for ethylene carbonate (EC)")

    from yadonpy.io.gmx import write_gmx

    out_dir = Path(__file__).resolve().parent / "work_dir" / "90_EC_gmx"
    gro_path, itp_path, top_path = write_gmx(mol=EC, out_dir=out_dir)

    print("[DONE] OPLS-AA assignment completed for EC.\n")
    print(f"  gro : {gro_path}")
    print(f"  itp : {itp_path}")
    print(f"  top : {top_path}")
    n_angles = len(getattr(EC, "angles", {}))
    n_dihedrals = len(getattr(EC, "dihedrals", {}))
    print(f"\nAssigned angles: {n_angles}")
    print(f"Assigned dihedrals: {n_dihedrals}")
    print("\nThis example stops after OPLS-AA typing + bonded assignment.")


if __name__ == "__main__":
    main()
