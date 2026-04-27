from __future__ import annotations

import json
from pathlib import Path

import pytest

import yadonpy.sim.preset.eq as eqmod
from yadonpy.io.gromacs_system import SystemExportResult
from yadonpy.sim.analyzer import AnalyzeResult
from yadonpy.sim.performance import resolve_io_analysis_policy


def test_auto_policy_keeps_short_small_runs_lean_but_dense():
    policy = resolve_io_analysis_policy(prod_ns=20.0, atom_count=20_000)

    assert policy.policy_level == "small"
    assert policy.traj_ps == pytest.approx(2.0)
    assert policy.energy_ps == pytest.approx(2.0)
    assert policy.analysis_profile == "transport_fast"
    assert policy.rdf_frame_stride == 5


def test_auto_policy_coarsens_long_large_runs():
    policy = resolve_io_analysis_policy(prod_ns=300.0, atom_count=60_000)

    assert policy.policy_level == "efficient"
    assert policy.traj_ps == pytest.approx(20.0)
    assert policy.energy_ps == pytest.approx(20.0)
    assert policy.analysis_profile == "transport_fast"
    assert policy.rdf_frame_stride == 10
    assert policy.include_polymer_metrics is False


def test_auto_policy_uses_minimal_for_paper_scale_very_large_runs():
    policy = resolve_io_analysis_policy(prod_ns=600.0, atom_count=200_000)

    assert policy.policy_level == "minimal"
    assert policy.traj_ps == pytest.approx(50.0)
    assert policy.analysis_profile == "minimal"
    assert policy.rdf_frame_stride == 20


def test_policy_explicit_overrides_win():
    policy = resolve_io_analysis_policy(
        prod_ns=600.0,
        atom_count=200_000,
        performance_profile="auto",
        analysis_profile="full",
        traj_ps=2.0,
        energy_ps=4.0,
        rdf_frame_stride=1,
    )

    assert policy.traj_ps == pytest.approx(2.0)
    assert policy.energy_ps == pytest.approx(4.0)
    assert policy.analysis_profile == "full"
    assert policy.rdf_frame_stride == 1
    assert policy.overrides["traj_ps"] == pytest.approx(2.0)


def test_npt_auto_policy_writes_coarse_output_cadence(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "system.gro").write_text("test\n60000\n1.0 1.0 1.0\n", encoding="utf-8")
    for name in ("system.top", "system.ndx"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    (system_dir / "system_meta.json").write_text(json.dumps({"species": []}), encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeCell:
        def GetNumAtoms(self):
            return 60_000

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["out_dir"] = out_dir
            self.out_dir = Path(out_dir)
            self.stages = list(stages)

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "01_npt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NPT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    npt = eqmod.NPT(ac=_FakeCell(), work_dir=tmp_path)
    monkeypatch.setattr(npt._resume, "run", lambda spec, fn: fn())

    npt.exec(temp=300.0, press=1.0, mpi=1, omp=1, gpu=0, time=300.0, dt_ps=0.002)

    stage = captured["stages"][0]
    mdp_text = stage.mdp.render()
    assert "nstxout-compressed       = 10000" in mdp_text
    assert "nstenergy                = 10000" in mdp_text
    assert "nstlog                   = 10000" in mdp_text
    summary = json.loads((tmp_path / "05_npt_production" / "summary.json").read_text(encoding="utf-8"))
    assert summary["performance_policy"]["policy_level"] == "efficient"


def test_analyzer_auto_profile_reads_production_policy(tmp_path: Path):
    summary_dir = tmp_path / "05_npt_production"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(
            {
                "performance_policy": {
                    "policy_level": "minimal",
                    "performance_profile": "auto",
                    "analysis_profile": "minimal",
                    "traj_ps": 50.0,
                }
            }
        ),
        encoding="utf-8",
    )
    analyzer = AnalyzeResult(
        work_dir=tmp_path,
        tpr=tmp_path / "md.tpr",
        xtc=tmp_path / "md.xtc",
        edr=tmp_path / "md.edr",
        top=tmp_path / "system.top",
        ndx=tmp_path / "system.ndx",
    )

    assert analyzer._resolve_analysis_profile("auto") == "minimal"
    assert analyzer._analysis_policy_cache_meta()["output_traj_ps"] == pytest.approx(50.0)
