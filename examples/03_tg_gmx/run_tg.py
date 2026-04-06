from __future__ import annotations

from pathlib import Path

import yadonpy as yp
from yadonpy.diagnostics import doctor


restart_status = True
yp.set_run_options(restart=restart_status)

BASE_DIR = Path(__file__).resolve().parent
SYSTEM_WORK_DIR = BASE_DIR.parent / "02_polymer_electrolyte" / "work_dir"
OUT_DIR = BASE_DIR / "work_dir"


def main() -> None:
    doctor(print_report=True)
    prepared = yp.resolve_prepared_system(
        work_dir=SYSTEM_WORK_DIR,
        source_name="example02_equilibrated_system",
    )
    result = yp.run_tg_scan_gmx(
        prepared=prepared,
        out_dir=OUT_DIR,
        profile="default",
        restart=restart_status,
    )
    yp.print_mechanics_result_summary(result)


if __name__ == "__main__":
    main()
