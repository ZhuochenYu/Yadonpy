from __future__ import annotations

from pathlib import Path

import yadonpy as yp


restart = True
yp.set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir" / "01_peo_smoke"


if __name__ == "__main__":
    yp.doctor(print_report=True)
    ff = yp.get_ff("gaff2_mod")
    ion_ff = yp.get_ff("merz")

    result = yp.build_graphite_peo_example_case(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        profile="smoke",
        restart=restart,
    )
    yp.print_sandwich_result_summary(result, profile="smoke")
