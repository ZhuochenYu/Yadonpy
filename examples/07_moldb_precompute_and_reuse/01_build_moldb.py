from __future__ import annotations

"""Example 07 / Step 1: Build MolDB entries from a CSV.

This script is intentionally minimal.

CSV columns (required):
  - name, smiles

CSV columns (optional):
  - opt (0/1)
  - confsearch (0/1)
  - charge_method (default: RESP)
  - basis_set (default: RESP default)
  - method (default: RESP default)

MolDB default directory:
  ~/.yadonpy/moldb (or $YADONPY_MOLDB)
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

from yadonpy.moldb import MolDB


if __name__ == "__main__":
    db = MolDB()  # default: ~/.yadonpy/moldb (or $YADONPY_MOLDB)
    print(f"MolDB directory: {db.db_dir}")

    work_root = HERE / "work_dir" / "01_build_moldb"
    work_root.mkdir(parents=True, exist_ok=True)

    db.read_calc_temp = str(HERE / "template.csv")
    # Explicitly add results into ~/.yadonpy/moldb
    db.autocalculate(work_root, add_to_moldb=True)

    print("All done.")
