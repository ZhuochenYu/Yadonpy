from __future__ import annotations

from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.ff import OPLSAA
from yadonpy.api import load_from_moldb


def main():
    set_run_options(restart=False)

    ff = OPLSAA()

    # Example A: load a RESP-charged EC molecule from MolDB and preserve those
    # external charges while assigning OPLS-AA atom types / bonded parameters.
    smiles_ec = "O=C1OCCO1"
    ec = load_from_moldb(
        smiles_ec,
        charge="RESP",
        require_ready=True,
    )
    print("[MolDB] loaded EC with external RESP charges.")
    if not ff.ff_assign(ec, charge=None):
        raise RuntimeError("OPLS-AA assignment failed for MolDB-backed EC with external RESP charges")
    from yadonpy.io.gmx import write_gmx

    ec_out = Path(__file__).resolve().parent / "work_dir" / "90_EC_gmx"
    ec_gro, ec_itp, ec_top = write_gmx(mol=ec, out_dir=ec_out, mol_name="EC")
    print("[DONE] OPLS-AA assignment completed for MolDB-backed EC while preserving RESP charges.\n")
    print(f"  EC gro : {ec_gro}")
    print(f"  EC itp : {ec_itp}")
    print(f"  EC top : {ec_top}")

    # Example B: monatomic metal ions can use OPLS-AA directly without a ready
    # MolDB entry. If no external charges exist, ff_assign() falls back to the
    # built-in OPLS-AA type charge automatically.
    na = ff.mol("[Na+]")
    if not ff.ff_assign(na):
        raise RuntimeError("OPLS-AA assignment failed for Na+")
    na_out = Path(__file__).resolve().parent / "work_dir" / "90_Na_gmx"
    na_gro, na_itp, na_top = write_gmx(mol=na, out_dir=na_out, mol_name="Na")
    print("[DONE] OPLS-AA assignment completed for Na+ using built-in OPLS-AA ion parameters.")
    print(f"  Na gro : {na_gro}")
    print(f"  Na itp : {na_itp}")
    print(f"  Na top : {na_top}")


if __name__ == "__main__":
    main()
