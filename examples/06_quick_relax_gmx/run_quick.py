from pathlib import Path

from yadonpy.diagnostics import doctor
from yadonpy.workflow import steps
from yadonpy.gmx.analysis.plot import plot_xvg_svg, plot_xvg_split_svg

# Example 04: Quick relax (GROMACS)
# This example consumes the packed system from Example 01.

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
    ex1 = base.parent / "01_full_workflow_smiles" / "work_dir"
    gro = ex1 / "00_system" / "system.gro"
    top = ex1 / "00_system" / "system.top"

    out_dir = base / "work_dir"

    steps.quick_relax_gmx(
        gro=gro,
        top=top,
        out_dir=out_dir,
        temperature_k=300.0,
        dt_ps=0.002,
        nvt_ps=50.0,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        restart=restart_status,
    )

    # --- plotting (SVG) ---
    thermo_xvg = out_dir / "thermo.xvg"
    if thermo_xvg.exists():
        plot_xvg_svg(thermo_xvg, out_svg=out_dir / "thermo.svg", title="QuickRelax: Thermo", xlabel="Time (ps)")
        plot_xvg_split_svg(thermo_xvg, out_dir=out_dir, title_prefix="QuickRelax")

    print(f"[OK] QuickRelaxJob finished: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()