from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
SRC_DIR = REPO_ROOT / "src"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    script: str
    gpu_id: int
    profile: str | None = None
    route: str = "screening"
    omp: int = 12
    psi4_omp: int = 16
    psi4_memory_mb: int = 20000


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(name="01_peo_smoke", script="01_peo_smoke.py", gpu_id=0),
    CaseSpec(name="03_cmcna_smoke", script="03_cmcna_smoke.py", gpu_id=1),
    CaseSpec(
        name="05_cmcna_glucose6_periodic_smoke",
        script="05_cmcna_glucose6_periodic_case.py",
        gpu_id=2,
        profile="smoke",
        omp=16,
        psi4_omp=20,
        psi4_memory_mb=24000,
    ),
)

DEFAULT_PHASES: tuple[str, ...] = ("fresh", "restart")


def parse_phases(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_PHASES
    if isinstance(raw, str):
        tokens = [item.strip().lower() for item in raw.split(",")]
    else:
        tokens = [str(item).strip().lower() for item in raw]
    phases = tuple(item for item in tokens if item)
    invalid = [item for item in phases if item not in {"fresh", "restart"}]
    if invalid:
        raise ValueError(f"Unsupported matrix phase(s): {', '.join(invalid)}")
    return phases or DEFAULT_PHASES


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _case_env(case: CaseSpec, *, phase: str, work_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        str(SRC_DIR)
        if not existing_pythonpath
        else os.pathsep.join([str(SRC_DIR), existing_pythonpath])
    )
    env["YADONPY_WORK_DIR"] = str(work_dir)
    env["YADONPY_RESTART"] = "1" if phase == "restart" else "0"
    env["YADONPY_ROUTE"] = str(case.route)
    env["YADONPY_GPU"] = "1"
    env["YADONPY_GPU_ID"] = str(case.gpu_id)
    env["YADONPY_OMP"] = str(case.omp)
    env["YADONPY_PSI4_OMP"] = str(case.psi4_omp)
    env["YADONPY_PSI4_MEMORY_MB"] = str(case.psi4_memory_mb)
    if case.profile is not None:
        env["YADONPY_PROFILE"] = str(case.profile)
    return env


def _collect_case_result(
    *,
    case: CaseSpec,
    phase: str,
    work_dir: Path,
    log_path: Path,
    returncode: int,
    timed_out: bool,
    started_at: float,
) -> dict[str, object]:
    manifest_path = work_dir / "06_full_stack_release" / "interface_manifest.json"
    progress_path = work_dir / "06_full_stack_release" / "interface_progress.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else None
    progress = _read_json(progress_path) if progress_path.exists() else None
    acceptance = (manifest or {}).get("acceptance") or {}
    return {
        "case": case.name,
        "phase": phase,
        "gpu_id": case.gpu_id,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_s": round(time.time() - started_at, 3),
        "work_dir": str(work_dir),
        "log_path": str(log_path),
        "manifest_path": str(manifest_path),
        "progress_path": str(progress_path),
        "manifest_exists": manifest_path.exists(),
        "progress_exists": progress_path.exists(),
        "progress_stage": None if progress is None else progress.get("stage"),
        "accepted": acceptance.get("accepted"),
        "order_ok": acceptance.get("order_ok"),
        "wrapped_ok": acceptance.get("wrapped_ok"),
    }


def write_matrix_metadata(
    *,
    base_dir: Path,
    timeout_s: int,
    cases: Sequence[CaseSpec] = CASES,
    phases: Sequence[str] = DEFAULT_PHASES,
) -> dict[str, object]:
    metadata = {
        "started_at": _timestamp(),
        "python": sys.executable,
        "repo_root": str(REPO_ROOT),
        "src_dir": str(SRC_DIR),
        "base_dir": str(base_dir),
        "timeout_s": int(timeout_s),
        "phases": list(phases),
        "cases": [asdict(case) for case in cases],
    }
    (base_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def run_matrix_iteration(
    *,
    base_dir: Path,
    iteration: int,
    timeout_s: int,
    cases: Sequence[CaseSpec] = CASES,
    phases: Sequence[str] = DEFAULT_PHASES,
) -> dict[str, object]:
    status_jsonl = base_dir / "status.jsonl"
    latest_status = base_dir / "latest_status.json"
    iter_dir = base_dir / f"iter_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    iteration_results: list[dict[str, object]] = []

    for phase in phases:
        phase_dir = iter_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            running: list[tuple[subprocess.Popen[str], CaseSpec, Path, Path, float]] = []
            for case in cases:
                case_dir = phase_dir / case.name
                work_dir = case_dir / "work_dir"
                log_dir = case_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                work_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{phase}.log"
                handle = stack.enter_context(log_path.open("w", encoding="utf-8"))
                cmd = [sys.executable, str(BASE_DIR / case.script)]
                handle.write(
                    f"[{_timestamp()}] iteration={iteration} phase={phase} case={case.name} "
                    f"gpu={case.gpu_id} route={case.route} cwd={REPO_ROOT} "
                    f"pythonpath={SRC_DIR} cmd={' '.join(cmd)}\n"
                )
                handle.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    env=_case_env(case, phase=phase, work_dir=work_dir),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running.append((proc, case, work_dir, log_path, time.time()))

            for proc, case, work_dir, log_path, started_at in running:
                timed_out = False
                try:
                    returncode = proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    returncode = 124
                    timed_out = True
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"\n[{_timestamp()}] timeout after {timeout_s} s\n")
                result = _collect_case_result(
                    case=case,
                    phase=phase,
                    work_dir=work_dir,
                    log_path=log_path,
                    returncode=int(returncode),
                    timed_out=timed_out,
                    started_at=started_at,
                )
                iteration_results.append(result)
                with status_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    latest = {
        "updated_at": _timestamp(),
        "iteration": iteration,
        "results": iteration_results,
    }
    latest_status.write_text(json.dumps(latest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (iter_dir / "iteration_summary.json").write_text(
        json.dumps(latest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return latest


def run_matrix_loop(
    *,
    base_dir: Path,
    hours: float,
    timeout_min: int = 180,
    max_iterations: int = 0,
    dry_run: bool = False,
    cases: Sequence[CaseSpec] = CASES,
    phases: Sequence[str] = DEFAULT_PHASES,
) -> int:
    base_dir = base_dir.expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    timeout_s = max(600, int(timeout_min) * 60)
    deadline = time.time() + max(0.25, float(hours)) * 3600.0

    selected_phases = parse_phases(phases)
    metadata = write_matrix_metadata(base_dir=base_dir, timeout_s=timeout_s, cases=cases, phases=selected_phases)
    if dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return 0

    iteration = 0
    while time.time() < deadline:
        if max_iterations > 0 and iteration >= max_iterations:
            break
        iteration += 1
        run_matrix_iteration(
            base_dir=base_dir,
            iteration=iteration,
            timeout_s=timeout_s,
            cases=cases,
            phases=selected_phases,
        )

    finished = {
        "finished_at": _timestamp(),
        "iterations_completed": iteration,
        "base_dir": str(base_dir),
    }
    (base_dir / "finished.json").write_text(json.dumps(finished, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Overnight fast-iteration matrix for Example 08.")
    parser.add_argument("--mode", choices=("observe", "autofix"), default="observe")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "autofix_config.json")
    parser.add_argument("--hours", type=float, default=10.0, help="Target wall time in hours.")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR / "overnight_iteration_runs")
    parser.add_argument("--timeout-min", type=int, default=180, help="Per case timeout in minutes.")
    parser.add_argument("--max-iterations", type=int, default=0, help="Optional hard stop; 0 means unlimited.")
    parser.add_argument(
        "--phases",
        default="fresh,restart",
        help="Comma-separated phases to run. Use 'fresh' for fast autofix screening.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write metadata and print the planned matrix without running cases.")
    args = parser.parse_args()
    if args.mode == "autofix":
        from _autofix import run_autofix_loop

        return run_autofix_loop(
            config_path=args.config,
            base_dir=args.base_dir,
            total_hours=args.hours,
            dry_run=args.dry_run,
        )
    return run_matrix_loop(
        base_dir=args.base_dir,
        hours=args.hours,
        timeout_min=args.timeout_min,
        max_iterations=args.max_iterations,
        dry_run=args.dry_run,
        cases=CASES,
        phases=parse_phases(args.phases),
    )


if __name__ == "__main__":
    raise SystemExit(main())
