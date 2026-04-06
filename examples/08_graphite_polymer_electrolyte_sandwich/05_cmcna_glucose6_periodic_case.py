from __future__ import annotations

import os
from pathlib import Path

from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2_mod, MERZ
from yadonpy.interface import build_graphite_cmcna_glucose6_periodic_case
from yadonpy.runtime import set_run_options


restart = True
set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
PROFILE = os.environ.get("YADONPY_PROFILE", "full").strip().lower()
SMOKE = PROFILE == "smoke"
RUN_NAME = "05_cmcna_glucose6_periodic_smoke" if SMOKE else "05_cmcna_glucose6_periodic_case"
work_dir = BASE_DIR / "work_dir" / RUN_NAME


if __name__ == "__main__":
    doctor(print_report=True)
    ff = GAFF2_mod()
    ion_ff = MERZ()

    result = build_graphite_cmcna_glucose6_periodic_case(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        profile=PROFILE,
        restart=restart,
    )

    print("profile =", PROFILE)
    print("manifest_path =", result.manifest_path)
    print("relaxed_gro =", result.relaxed_gro)
    print("polymer_density_g_cm3 =", round(result.polymer_phase.density_g_cm3, 4))
    print("electrolyte_density_g_cm3 =", round(result.electrolyte_phase.density_g_cm3, 4))
    print("stack_checks =", result.stack_checks)
