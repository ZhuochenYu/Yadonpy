from __future__ import annotations

"""Example 09 / Step 2: OPLS-AA with MolDB-backed charges and a simple ion."""

from pathlib import Path

import yadonpy as yp
from yadonpy.core import workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.io.gmx import write_gmx
from yadonpy.runtime import set_run_options


restart_status = False
set_run_options(restart=restart_status)

ff = yp.get_ff("oplsaa")

smiles_ec = "O=C1OCCO1"
smiles_na = "[Na+]"

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir"


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    example_wd = workdir(work_dir, restart=restart_status)

    ec = yp.load_from_moldb(
        smiles_ec,
        charge="RESP",
        require_ready=True,
    )
    print("[MolDB] loaded EC with external RESP charges.")
    if not ff.ff_assign(ec, charge=None):
        raise RuntimeError("OPLS-AA assignment failed for MolDB-backed EC with external RESP charges.")

    ec_out = example_wd.child("02_ec_from_moldb_gmx")
    ec_gro, ec_itp, ec_top = write_gmx(mol=ec, out_dir=ec_out, mol_name="EC")

    na = ff.mol(smiles_na)
    if not ff.ff_assign(na):
        raise RuntimeError("OPLS-AA assignment failed for Na+.")

    na_out = example_wd.child("03_na_gmx")
    na_gro, na_itp, na_top = write_gmx(mol=na, out_dir=na_out, mol_name="Na")

    print("[DONE] OPLS-AA assignment completed for MolDB-backed EC while preserving RESP charges.")
    print(f"  EC gro : {ec_gro}")
    print(f"  EC itp : {ec_itp}")
    print(f"  EC top : {ec_top}")
    print("[DONE] OPLS-AA assignment completed for Na+ using the built-in OPLS-AA ion parameters.")
    print(f"  Na gro : {na_gro}")
    print(f"  Na itp : {na_itp}")
    print(f"  Na top : {na_top}")
