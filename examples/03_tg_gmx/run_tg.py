from pathlib import Path

from yadonpy.diagnostics import doctor
from yadonpy.workflow import steps
"""Tg scan example (GROMACS).

Since yadonpy v0.3.0, SVG plots are generated automatically:

  - Per temperature:  T??_xxxK/plots/density_time.svg (and thermo plots)
  - Global curve:     plots/tg_density_vs_T.svg

So this example focuses on running the workflow.
"""

# Example 03: Tg scan (GROMACS)
# This example consumes the equilibrated structure from Example 02.

# --- global restart switch ---
restart_status = True  # set False to force re-run

# --- resources ---
mpi = 1
omp = 16
# GPU switch: 1 enables GPU, 0 disables GPU
gpu = 1
# Which GPU card to use (only when gpu==1)
gpu_id = 0


def main() -> None:
    doctor(print_report=True)

    base = Path(__file__).resolve().parent
    ex2 = base.parent / "02_polymer_electrolyte" / "work_dir"
    gro = ex2 / "01_equilibration" / "04_md" / "md.gro"
    top = ex2 / "00_system" / "system.top"

    out_dir = base / "work_dir"

    temps = [500, 480, 460, 440, 420, 400, 380, 360, 340, 320, 300]

    steps.tg_scan_gmx(
        gro=gro,
        top=top,
        out_dir=out_dir,
        temperatures_k=temps,
        pressure_bar=1.0,
        npt_ns=2.0,
        frac_last=0.5,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        
    )

    print(f"[OK] TgJob finished: {out_dir / 'summary.json'}")
    print(f"[OK] Plots: {out_dir / 'plots'} and each T*/plots")


if __name__ == "__main__":
    main()
