from __future__ import annotations

from pathlib import Path

from yadonpy.sim import analyzer as analyzer_mod
from yadonpy.sim import parallel_postprocess as parallel_mod
from yadonpy.sim.parallel_postprocess import InterfaceAnalysisTask, run_interface_analyses_parallel


def test_parallel_interface_postprocess_runs_serial_task_with_full_kwargs(monkeypatch, tmp_path: Path):
    calls = []

    class FakeInterface:
        def summary(self, *, time_series_analysis=False):
            calls.append(("summary", time_series_analysis))
            return {
                "outputs": {
                    "interface_profile_summary_json": str(tmp_path / "case" / "06_analysis" / "interface_profile_summary.json")
                }
            }

        def geometry_health(self, *, time_series_analysis=False):
            calls.append(("geometry_health", time_series_analysis))
            return {"phase_order_ok": True}

        def membrane_permeation(self, *, species=None, time_series_analysis=False):
            calls.append(("membrane_permeation", tuple(species or ()), time_series_analysis))
            return {"available": True}

    class FakeAnalyzer:
        def __init__(self, work_dir):
            self.work_dir = Path(work_dir)

        def interface(self, **kwargs):
            calls.append(("interface", kwargs))
            return FakeInterface()

    monkeypatch.setattr(
        analyzer_mod.AnalyzeResult,
        "from_work_dir",
        classmethod(lambda cls, work_dir: FakeAnalyzer(work_dir)),
    )

    task = InterfaceAnalysisTask(
        name="case-a",
        work_dir=tmp_path / "case",
        manifest_path=tmp_path / "case" / "layer_stack_manifest.json",
        split_electrodes=True,
        penetration_species=("EC", "PF6"),
        time_series_analysis=True,
        methods=("geometry_health", "membrane_permeation", "summary"),
    )
    result = run_interface_analyses_parallel([task], workers=1, thread_limit=1)

    assert result.ok is True
    assert result.workers == 1
    assert result.results[0].name == "case-a"
    assert result.results[0].summary_path.endswith("interface_profile_summary.json")
    interface_call = next(item for item in calls if item[0] == "interface")
    assert interface_call[1]["manifest_path"] == tmp_path / "case" / "layer_stack_manifest.json"
    assert interface_call[1]["split_electrodes"] is True
    assert ("summary", True) in calls
    assert ("geometry_health", True) in calls
    assert ("membrane_permeation", ("EC", "PF6"), True) in calls


def test_parallel_interface_postprocess_records_task_failures(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        analyzer_mod.AnalyzeResult,
        "from_work_dir",
        classmethod(lambda cls, work_dir: (_ for _ in ()).throw(RuntimeError("boom"))),
    )

    result = run_interface_analyses_parallel(
        [InterfaceAnalysisTask(name="bad", work_dir=tmp_path / "bad")],
        workers=1,
        thread_limit=1,
    )

    assert result.ok is False
    assert result.results[0].ok is False
    assert "RuntimeError: boom" in str(result.results[0].error)


def test_parallel_postprocess_auto_workers_caps_at_case_count(monkeypatch):
    monkeypatch.setattr(parallel_mod.os, "cpu_count", lambda: 48)

    assert parallel_mod._resolve_workers("auto", task_count=4) == 4
    assert parallel_mod._resolve_workers(None, task_count=4) == 4
    assert parallel_mod._resolve_workers("auto", task_count=20) == 12
