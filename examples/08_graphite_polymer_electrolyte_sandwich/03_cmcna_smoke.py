from __future__ import annotations

from pathlib import Path

from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2_mod, MERZ
from yadonpy.interface import build_graphite_cmcna_example_case, print_sandwich_result_summary
from yadonpy.runtime import set_run_options


restart = True
set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir" / "03_cmcna_smoke"


if __name__ == "__main__":
    doctor(print_report=True)
    ff = GAFF2_mod()
    ion_ff = MERZ()

    result = build_graphite_cmcna_example_case(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        profile="smoke",
        restart=restart,
    )
    print_sandwich_result_summary(result, profile="smoke")
