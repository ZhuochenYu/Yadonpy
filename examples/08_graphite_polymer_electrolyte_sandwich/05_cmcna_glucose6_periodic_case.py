from __future__ import annotations

import os
from pathlib import Path

import yadonpy as yp


restart = True
yp.set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
PROFILE = os.environ.get("YADONPY_PROFILE", "full").strip().lower()
SMOKE = PROFILE == "smoke"
RUN_NAME = "05_cmcna_glucose6_periodic_smoke" if SMOKE else "05_cmcna_glucose6_periodic_case"
work_dir = BASE_DIR / "work_dir" / RUN_NAME


if __name__ == "__main__":
    yp.doctor(print_report=True)
    ff = yp.get_ff("gaff2_mod")
    ion_ff = yp.get_ff("merz")

    result = yp.build_graphite_cmcna_glucose6_periodic_case(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        profile=PROFILE,
        restart=restart,
    )
    yp.print_sandwich_result_summary(result, profile=PROFILE)
