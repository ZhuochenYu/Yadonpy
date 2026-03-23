from pathlib import Path

from yadonpy.diagnostics import doctor
from yadonpy.workflow import steps
"""Uniaxial elongation example (GROMACS).

Since yadonpy v0.3.0, the workflow writes `stress_strain.svg` automatically.
"""

# Example 04: Uniaxial elongation (GROMACS)
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

    steps.elongation_gmx(
        gro=gro,
        top=top,
        out_dir=out_dir,
        temperature_k=300.0,
        pressure_bar=1.0,
        strain_rate_1_ps=1e-6,
        total_strain=0.5,
        dt_ps=0.002,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        
    )

    print(f"[OK] ElongationJob finished: {out_dir / 'summary.json'}")
    print(f"[OK] Plot: {out_dir / 'stress_strain.svg'}")


if __name__ == "__main__":
    main()
