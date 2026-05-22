"""Parallel post-processing helper for Eg08 charge-sweep runs.

This script is meant for the common workflow where several charge states were
sampled from the same t=0 structure on different GPUs.  The MD jobs are already
independent, so the post-processing should also run one process per case instead
of analyzing charge states serially.

Typical remote use:

    export EG08_SWEEP_ROOT=/path/to/eg08_07_cmc_facing_charge_100ns_v1
    export EG08_POSTPROCESS_WORKERS=4
    python postprocess_charge_sweep_parallel.py

The script follows the explicit public API style used by the example scripts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from yadonpy import InterfaceAnalysisTask, run_interface_analyses_parallel


DEFAULT_CASES = (
    ("0 uC/cm2", "cmcface_00_uC_cm2", 0.0),
    ("-3 uC/cm2", "cmcface_m3p0_uC_cm2", -3.0),
    ("-9 uC/cm2", "cmcface_m9p0_uC_cm2", -9.0),
    ("-18 uC/cm2", "cmcface_m18p0_uC_cm2", -18.0),
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _case_specs() -> tuple[tuple[str, str, float], ...]:
    """Return charge cases, optionally overridden by JSON in the environment."""

    raw = os.environ.get("EG08_SWEEP_CASES_JSON")
    if not raw:
        return DEFAULT_CASES
    payload = json.loads(raw)
    cases: list[tuple[str, str, float]] = []
    for item in payload:
        cases.append((str(item["label"]), str(item["dir"]), float(item["charge_uC_cm2"])))
    return tuple(cases)


def _wait_for_done(root: Path, cases: tuple[tuple[str, str, float], ...], poll_s: float) -> None:
    """Wait until every case has written ``charge_case_done.json``."""

    report_dir = root / "99_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    while True:
        missing = []
        for _label, dirname, _charge in cases:
            done = root / dirname / "charge_case_done.json"
            if not done.is_file():
                missing.append(str(done))
        status = {
            "root": str(root),
            "missing_done": missing,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (report_dir / "parallel_postprocess_wait_status.json").write_text(
            json.dumps(status, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not missing:
            return
        time.sleep(max(30.0, float(poll_s)))


def _tasks(root: Path, cases: tuple[tuple[str, str, float], ...]) -> list[InterfaceAnalysisTask]:
    """Build one independent analysis task per charge state."""

    penetration_species = _env_tuple("EG08_PENETRATION_SPECIES", ("EC", "EMC", "DEC", "Li", "PF6", "Na"))
    adsorption_species = _env_tuple("EG08_ADSORPTION_SPECIES", ("EC", "EMC", "DEC"))
    tasks: list[InterfaceAnalysisTask] = []
    for label, dirname, charge in cases:
        case_dir = root / dirname
        relax_dir = case_dir / "03_relaxation_sampling"
        tasks.append(
            InterfaceAnalysisTask(
                name=label,
                work_dir=relax_dir,
                manifest_path=relax_dir / "layer_stack_manifest.json",
                analysis_profile=os.environ.get("EG08_ANALYSIS_PROFILE", "interface_fast"),
                bin_nm=float(os.environ.get("EG08_ANALYSIS_BIN_NM", "0.05")),
                frame_stride=os.environ.get("EG08_ANALYSIS_FRAME_STRIDE", "auto"),
                region_width_nm=float(os.environ.get("EG08_REGION_WIDTH_NM", "0.75")),
                surface_grid_nm=float(os.environ.get("EG08_SURFACE_GRID_NM", "0.50")),
                surface_distance_nm=float(os.environ.get("EG08_SURFACE_DISTANCE_NM", "0.50")),
                penetration_threshold_nm=float(os.environ.get("EG08_PENETRATION_THRESHOLD_NM", "0.20")),
                adsorption_min_residence_ps=float(os.environ.get("EG08_ADSORPTION_MIN_RESIDENCE_PS", "10.0")),
                potential_reference=os.environ.get("EG08_POTENTIAL_REFERENCE", "zero_mean"),
                split_electrodes=_env_bool("EG08_SPLIT_ELECTRODES", True),
                report_potential_drop=_env_bool("EG08_REPORT_POTENTIAL_DROP", True),
                penetration_species=penetration_species,
                adsorption_species=adsorption_species,
                compute_transport=_env_bool("EG08_COMPUTE_TRANSPORT", True),
                time_series_analysis=_env_bool("EG08_TIME_SERIES_ANALYSIS", True),
                time_series_sample_count=int(os.environ.get("EG08_TIME_SERIES_SAMPLE_COUNT", "10")),
                time_series_fps=float(os.environ.get("EG08_TIME_SERIES_FPS", "1.0")),
                time_series_rdf=_env_bool("EG08_TIME_SERIES_RDF", True),
                time_series_concentration=_env_bool("EG08_TIME_SERIES_CONCENTRATION", True),
                time_series_angles=_env_bool("EG08_TIME_SERIES_ANGLES", True),
                time_series_charge_potential=_env_bool("EG08_TIME_SERIES_CHARGE_POTENTIAL", True),
                time_series_rdf_rmax_nm=float(os.environ.get("EG08_TIME_SERIES_RDF_RMAX_NM", "1.2")),
                time_series_rdf_bin_nm=float(os.environ.get("EG08_TIME_SERIES_RDF_BIN_NM", "0.02")),
                resume=_env_bool("EG08_POSTPROCESS_RESUME", True),
                metadata={"charge_uC_cm2": charge, "case_dir": str(case_dir)},
            )
        )
    return tasks


def main() -> None:
    root = Path(os.environ.get("EG08_SWEEP_ROOT", ".")).resolve()
    cases = _case_specs()
    if _env_bool("EG08_WAIT_FOR_DONE", True):
        _wait_for_done(root, cases, poll_s=float(os.environ.get("EG08_WAIT_POLL_S", "600")))

    report_dir = root / "99_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    batch = run_interface_analyses_parallel(
        _tasks(root, cases),
        workers=os.environ.get("EG08_POSTPROCESS_WORKERS", "auto"),
        thread_limit=int(os.environ.get("EG08_POSTPROCESS_THREAD_LIMIT", "1")),
        fail_fast=_env_bool("EG08_POSTPROCESS_FAIL_FAST", False),
        summary_path=report_dir / "parallel_interface_analysis_summary.json",
    )
    print(json.dumps(batch.to_dict(), indent=2, ensure_ascii=False))
    if not batch.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
