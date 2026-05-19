"""Process-level parallel helpers for interface post-processing batches.

The heavy layer-stack interface analyzer intentionally keeps a single trajectory
analysis deterministic and cacheable.  For charge sweeps and replicated
interface runs, the safe acceleration point is one process per independent case:
each worker reads its own trajectory, writes its own analysis directory, and
limits BLAS/OpenMP helper threads to avoid CPU oversubscription.
"""

from __future__ import annotations

import concurrent.futures as confu
import json
import os
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class InterfaceAnalysisTask:
    """One independent layer-stack interface analysis job.

    ``summary()`` is the only method required to produce all standard Eg08
    outputs.  The optional ``methods`` list can request facade views after the
    shared cached summary has been generated, mainly for scripts that want a
    small per-task JSON with selected sections.
    """

    work_dir: str | Path
    name: str | None = None
    manifest_path: str | Path | None = None
    out_dir: str | Path | None = None
    analysis_profile: str = "interface_fast"
    bin_nm: float = 0.05
    frame_stride: int | str = "auto"
    surface_distance_nm: float = 0.50
    region_width_nm: float = 0.75
    surface_grid_nm: float = 0.5
    penetration_threshold_nm: float = 0.20
    adsorption_min_residence_ps: float = 10.0
    potential_reference: str = "zero_mean"
    split_electrodes: bool = False
    report_potential_drop: bool = False
    penetration_species: tuple[str, ...] | None = None
    adsorption_species: tuple[str, ...] | None = None
    phase_groups: tuple[str, ...] | None = None
    compute_transport: bool = True
    time_series_analysis: bool = False
    time_series_sample_count: int = 10
    time_series_fps: float = 1.0
    time_series_rdf: bool = True
    time_series_concentration: bool = True
    time_series_angles: bool = True
    time_series_charge_potential: bool = True
    time_series_rdf_rmax_nm: float = 1.2
    time_series_rdf_bin_nm: float = 0.02
    resume: bool = True
    methods: tuple[str, ...] = ("summary",)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def task_name(self) -> str:
        if self.name:
            return str(self.name)
        return Path(self.work_dir).name


@dataclass(frozen=True)
class InterfaceAnalysisTaskResult:
    """Result record for one interface analysis task."""

    name: str
    work_dir: str
    ok: bool
    summary_path: str | None = None
    outputs: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterfaceAnalysisBatchResult:
    """Aggregate result returned by :func:`run_interface_analyses_parallel`."""

    ok: bool
    workers: int
    results: tuple[InterfaceAnalysisTaskResult, ...]
    summary_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "workers": int(self.workers),
            "summary_path": self.summary_path,
            "results": [asdict(result) for result in self.results],
        }


def _resolve_workers(workers: int | str | None, *, task_count: int) -> int:
    if int(task_count) <= 0:
        return 1
    if workers is None or str(workers).strip().lower() in {"", "auto", "default"}:
        cpu_count = os.cpu_count() or 1
        resolved = max(1, min(int(task_count), max(1, int(cpu_count) // 4)))
    else:
        try:
            resolved = int(workers)
        except Exception:
            resolved = 1
    return max(1, min(int(resolved), int(task_count)))


def _limit_worker_threads(thread_limit: int | None) -> None:
    if thread_limit is None:
        return
    value = str(max(1, int(thread_limit)))
    for key in _THREAD_ENV_VARS:
        os.environ[key] = value


def _interface_kwargs(task: InterfaceAnalysisTask) -> dict[str, Any]:
    return {
        "manifest_path": task.manifest_path,
        "analysis_profile": task.analysis_profile,
        "bin_nm": task.bin_nm,
        "frame_stride": task.frame_stride,
        "surface_distance_nm": task.surface_distance_nm,
        "region_width_nm": task.region_width_nm,
        "surface_grid_nm": task.surface_grid_nm,
        "penetration_threshold_nm": task.penetration_threshold_nm,
        "adsorption_min_residence_ps": task.adsorption_min_residence_ps,
        "potential_reference": task.potential_reference,
        "split_electrodes": task.split_electrodes,
        "report_potential_drop": task.report_potential_drop,
        "penetration_species": task.penetration_species,
        "adsorption_species": task.adsorption_species,
        "phase_groups": task.phase_groups,
        "out_dir": task.out_dir,
        "compute_transport": task.compute_transport,
        "time_series_sample_count": task.time_series_sample_count,
        "time_series_fps": task.time_series_fps,
        "time_series_rdf": task.time_series_rdf,
        "time_series_concentration": task.time_series_concentration,
        "time_series_angles": task.time_series_angles,
        "time_series_charge_potential": task.time_series_charge_potential,
        "time_series_rdf_rmax_nm": task.time_series_rdf_rmax_nm,
        "time_series_rdf_bin_nm": task.time_series_rdf_bin_nm,
        "resume": task.resume,
    }


def _run_one_interface_task(task: InterfaceAnalysisTask, *, thread_limit: int | None) -> InterfaceAnalysisTaskResult:
    _limit_worker_threads(thread_limit)
    name = task.task_name()
    try:
        from .analyzer import AnalyzeResult

        analyzer = AnalyzeResult.from_work_dir(task.work_dir)
        interface = analyzer.interface(**_interface_kwargs(task))
        collected: dict[str, Any] = {}
        summary = interface.summary(time_series_analysis=bool(task.time_series_analysis))
        collected["summary"] = summary
        method_names = tuple(dict.fromkeys(str(method) for method in task.methods if str(method)))
        for method in method_names:
            if method == "summary":
                continue
            func = getattr(interface, method)
            if method in {"penetration", "membrane_permeation"}:
                collected[method] = func(
                    species=task.penetration_species,
                    time_series_analysis=bool(task.time_series_analysis),
                )
            elif method == "graphite_adsorption":
                collected[method] = func(
                    species=task.adsorption_species,
                    time_series_analysis=bool(task.time_series_analysis),
                )
            elif method == "edl_profiles":
                collected[method] = func(
                    split_electrodes=task.split_electrodes,
                    potential_reference=task.potential_reference,
                    report_potential_drop=task.report_potential_drop,
                    time_series_analysis=bool(task.time_series_analysis),
                )
            else:
                collected[method] = func(time_series_analysis=bool(task.time_series_analysis))
        outputs = dict(summary.get("outputs") or {}) if isinstance(summary, dict) else {}
        summary_path = outputs.get("interface_profile_summary_json")
        return InterfaceAnalysisTaskResult(
            name=name,
            work_dir=str(task.work_dir),
            ok=True,
            summary_path=None if summary_path is None else str(summary_path),
            outputs=outputs,
            metadata=dict(task.metadata or {}),
        )
    except Exception as exc:
        return InterfaceAnalysisTaskResult(
            name=name,
            work_dir=str(task.work_dir),
            ok=False,
            error=f"{exc.__class__.__name__}: {exc}",
            traceback=traceback.format_exc(),
            metadata=dict(task.metadata or {}),
        )


def run_interface_analyses_parallel(
    tasks: Sequence[InterfaceAnalysisTask | Mapping[str, Any]],
    *,
    workers: int | str | None = "auto",
    thread_limit: int | None = 1,
    fail_fast: bool = False,
    summary_path: str | Path | None = None,
) -> InterfaceAnalysisBatchResult:
    """Run independent interface analyses concurrently.

    This is the recommended post-processing entry point for Eg08 charge sweeps,
    repeated seeds, or any collection of layer-stack trajectories.  Parallelism
    is deliberately at the case level; each trajectory still runs through the
    normal :class:`AnalyzeResult` cache and writes the same files as the serial
    facade.
    """

    normalized = tuple(task if isinstance(task, InterfaceAnalysisTask) else InterfaceAnalysisTask(**dict(task)) for task in tasks)
    task_count = len(normalized)
    resolved_workers = _resolve_workers(workers, task_count=task_count)
    if task_count == 0:
        batch = InterfaceAnalysisBatchResult(ok=True, workers=resolved_workers, results=tuple())
        if summary_path is not None:
            path = Path(summary_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(batch.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            batch = InterfaceAnalysisBatchResult(ok=True, workers=resolved_workers, results=tuple(), summary_path=str(path))
        return batch

    results_by_name: dict[str, InterfaceAnalysisTaskResult] = {}
    if resolved_workers <= 1 or task_count <= 1:
        for task in normalized:
            result = _run_one_interface_task(task, thread_limit=thread_limit)
            results_by_name[task.task_name()] = result
            if fail_fast and not result.ok:
                break
    else:
        with confu.ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            future_map = {
                executor.submit(_run_one_interface_task, task, thread_limit=thread_limit): task
                for task in normalized
            }
            for future in confu.as_completed(future_map):
                task = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = InterfaceAnalysisTaskResult(
                        name=task.task_name(),
                        work_dir=str(task.work_dir),
                        ok=False,
                        error=f"{exc.__class__.__name__}: {exc}",
                        traceback=traceback.format_exc(),
                        metadata=dict(task.metadata or {}),
                    )
                results_by_name[task.task_name()] = result
                if fail_fast and not result.ok:
                    for other in future_map:
                        other.cancel()
                    break

    ordered_results = tuple(results_by_name.get(task.task_name()) for task in normalized if task.task_name() in results_by_name)
    ok = all(bool(result.ok) for result in ordered_results)
    batch = InterfaceAnalysisBatchResult(ok=ok, workers=resolved_workers, results=ordered_results)
    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = batch.to_dict()
        payload["summary_path"] = str(path)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        batch = InterfaceAnalysisBatchResult(ok=ok, workers=resolved_workers, results=ordered_results, summary_path=str(path))
    return batch


__all__ = [
    "InterfaceAnalysisBatchResult",
    "InterfaceAnalysisTask",
    "InterfaceAnalysisTaskResult",
    "run_interface_analyses_parallel",
]
