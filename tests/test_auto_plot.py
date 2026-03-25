from __future__ import annotations

from pathlib import Path

from yadonpy.gmx.analysis.auto_plot import plot_thermo_stage


def test_plot_thermo_stage_emits_npt_convergence_svg(tmp_path: Path):
    thermo_xvg = tmp_path / "thermo.xvg"
    thermo_xvg.write_text(
        "\n".join(
            [
                '@    xaxis  label "Time (ps)"',
                '@    yaxis  label "Thermo"',
                '@    s0 legend "Density"',
                '@    s1 legend "Volume"',
                '@    s2 legend "Box-X"',
                '@    s3 legend "Box-Y"',
                '@    s4 legend "Box-Z"',
                "0  1100  10.0  2.00  2.10  2.20",
                "1  1080  10.2  2.02  2.09  2.18",
                "2  1060  10.4  2.04  2.08  2.16",
                "3  1050  10.5  2.05  2.07  2.14",
                "4  1045  10.6  2.05  2.06  2.13",
                "5  1040  10.7  2.06  2.06  2.12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plots = plot_thermo_stage(thermo_xvg, out_dir=tmp_path / "plots", title_prefix="npt_stage", frac_last=0.4)

    assert "thermo_svg" in plots
    assert "npt_convergence_svg" in plots
    assert Path(plots["npt_convergence_svg"]).exists()
