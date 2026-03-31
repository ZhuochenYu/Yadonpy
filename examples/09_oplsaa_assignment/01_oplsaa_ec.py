from __future__ import annotations

"""Example 09 / Step 1: OPLS-AA assignment for ethylene carbonate."""

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

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir"


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    example_wd = workdir(work_dir, restart=restart_status)

    ec = yp.mol_from_smiles(smiles_ec)
    if not ff.ff_assign(ec, charge="opls"):
        raise RuntimeError("OPLS-AA assignment failed for ethylene carbonate (EC).")

    out_dir = example_wd.child("01_ec_gmx")
    gro_path, itp_path, top_path = write_gmx(mol=ec, out_dir=out_dir, mol_name="EC")

    print("[DONE] OPLS-AA assignment completed for EC.")
    print(f"  gro : {gro_path}")
    print(f"  itp : {itp_path}")
    print(f"  top : {top_path}")
    print(f"  angles: {len(getattr(ec, 'angles', {}))}")
    print(f"  dihedrals: {len(getattr(ec, 'dihedrals', {}))}")
    print("This example stops after OPLS-AA typing, charge assignment, and export.")
