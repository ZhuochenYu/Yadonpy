from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yadonpy.gmx.analysis.rg_convergence import find_rg_convergence
from yadonpy.sim.analyzer import AnalyzeResult


def test_find_rg_convergence_accepts_short_stable_series():
    t_ps = np.arange(0.0, 70.0, 10.0)
    rg_nm = np.array([1.274, 1.276, 1.279, 1.277, 1.278, 1.276, 1.275], dtype=float)

    res = find_rg_convergence(t_ps, rg_nm)

    assert res.ok is True
    assert res.converged_by in {"trend", "sd_max", "both"}
    assert np.isfinite(res.mean)


def test_analyze_result_rg_series_accepts_short_valid_series(tmp_path: Path, monkeypatch):
    work_dir = tmp_path / "work"
    system_dir = work_dir / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "system_meta.json").write_text(
        json.dumps({"species": [{"moltype": "polymer_chain", "kind": "polymer", "smiles": "*CCO*"}]}),
        encoding="utf-8",
    )
    ndx = system_dir / "system.ndx"
    ndx.write_text("[ System ]\n1 2 3\n[ MOL_polymer_chain ]\n1 2 3\n", encoding="utf-8")

    tpr = work_dir / "md.tpr"
    xtc = work_dir / "md.xtc"
    edr = work_dir / "md.edr"
    top = system_dir / "system.top"
    for path in (tpr, xtc, edr, top):
        path.write_text("", encoding="utf-8")

    def _fake_gyrate(self, *, tpr, xtc, out_xvg, ndx=None, group=0, begin_ps=None, end_ps=None, cwd=None):
        out_xvg = Path(out_xvg)
        out_xvg.write_text(
            "\n".join(
                [
                    '@    xaxis  label "Time (ps)"',
                    '@    s0 legend "Rg"',
                    "0 1.274",
                    "10 1.276",
                    "20 1.279",
                    "30 1.277",
                    "40 1.278",
                    "50 1.276",
                    "60 1.275",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr("yadonpy.sim.analyzer.GromacsRunner.gyrate", _fake_gyrate)

    analy = AnalyzeResult(work_dir=work_dir, tpr=tpr, xtc=xtc, edr=edr, top=top, ndx=ndx)
    series = analy._rg_series()

    assert series is not None
    assert len(series["rg_nm"]) == 7
    assert series["group"] == 1
