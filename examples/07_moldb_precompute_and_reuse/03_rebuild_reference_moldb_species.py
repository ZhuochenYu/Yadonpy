from __future__ import annotations

"""Legacy alias for the consolidated Example 07 MolDB builder."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    print("[INFO] Example 07 Step 3 is a legacy alias. Redirecting to 01_build_moldb.py.")
    runpy.run_path(str(Path(__file__).with_name("01_build_moldb.py")), run_name="__main__")
