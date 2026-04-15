from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

try:
    import pexpect
except Exception:  # pragma: no cover - optional at import time
    pexpect = None


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "autofix_config.json"


@dataclass(frozen=True)
class WorkspaceConfig:
    mode: str = "snapshot_clone"
    path: str = "autofix_workspace"
    base_branch: str = "main"


@dataclass(frozen=True)
class RemoteConfig:
    host: str = "yadonpy-gpu"
    repo_root: str = "/home/zcyu/Yadonpy_EX08_AUTOFIX"
    run_root: str = "/home/zcyu/remote_runs/example08_autofix"
    conda_prefix: str = "/home/zcyu/anaconda3/bin/conda run -n yadonpy_test"
    password_env_var: str = "YADONPY_REMOTE_PASSWORD"


@dataclass(frozen=True)
class MatrixConfig:
    hours_per_round: float = 0.25
    timeout_min: int = 180
    max_iterations_per_round: int = 1
    phases: tuple[str, ...] = ("fresh",)


@dataclass(frozen=True)
class DiffLimits:
    max_changed_files: int = 6
    max_changed_lines: int = 260


@dataclass(frozen=True)
class PushPolicy:
    enabled: bool = True
    remote: str = "origin"
    branch: str = "main"
    require_env_var: str = "YADONPY_AUTOFIX_ALLOW_MAIN_PUSH"


@dataclass(frozen=True)
class StopPolicy:
    stagnant_rounds: int = 3
    repeated_failure_rounds: int = 2


@dataclass(frozen=True)
class AutofixConfig:
    workspace: WorkspaceConfig = WorkspaceConfig()
    remote: RemoteConfig = RemoteConfig()
    matrix: MatrixConfig = MatrixConfig()
    diff_limits: DiffLimits = DiffLimits()
    push: PushPolicy = PushPolicy()
    stop: StopPolicy = StopPolicy()
    total_hours: float = 10.0
    max_rounds: int = 0
    recipe_enabled: dict[str, bool] = field(
        default_factory=lambda: {
            "tighten_or_relax_confined_selection": True,
            "fix_stack_gap_estimation": True,
            "reduce_release_gap_or_padding": True,
            "adjust_rescue_acceptance_preserve_best_round": True,
            "trust_periodic_xy_semantics": True,
            "refine_wrap_density_scoring_or_candidate_ranking": True,
            "adjust_screening_route_fail_fast": True,
            "repair_harness_import_sync_runtime_wiring": True,
        }
    )
    test_commands: tuple[str, ...] = (
        "conda run -n yadonpy python -m pytest tests/test_sandwich_phase_build.py tests/test_sandwich_workflow.py tests/test_release_sanity.py tests/test_api.py -q",
    )


@dataclass(frozen=True)
class RecipeSpec:
    name: str
    risk: str
    files: tuple[str, ...]
    tests: tuple[str, ...]
    push_allowed: bool
    selector: Callable[[dict[str, object], Path], bool]
    applier: Callable[[Path], dict[str, object]]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_local(cmd: str, *, cwd: Path, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


def _interactive_password(password_env_var: str) -> str | None:
    return os.environ.get(password_env_var)


def _run_interactive_command(
    cmd: str,
    *,
    password_env_var: str,
    cwd: Path | None = None,
    timeout: int = 1200,
) -> dict[str, object]:
    if pexpect is None:
        raise RuntimeError("pexpect is required for unattended remote commands")
    password = _interactive_password(password_env_var)
    if password is None:
        raise RuntimeError(f"Remote password not available in environment variable {password_env_var!r}")
    child = pexpect.spawn(cmd, encoding="utf-8", timeout=timeout, cwd=(None if cwd is None else str(cwd)))
    transcript: list[str] = []
    prompts = [r"continue connecting", r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT]
    while True:
        idx = child.expect(prompts)
        transcript.append(child.before or "")
        if idx == 0:
            child.sendline("yes")
        elif idx == 1:
            child.sendline(password)
        elif idx == 2:
            break
        else:
            raise RuntimeError(f"Timeout while running command: {cmd}\n{''.join(transcript)}")
    return {
        "command": cmd,
        "stdout": "".join(transcript),
        "returncode": 0 if child.exitstatus in (0, None) else int(child.exitstatus),
    }


def _run_remote_command(config: RemoteConfig, remote_cmd: str, *, timeout: int = 7200) -> dict[str, object]:
    host = shlex.quote(config.host)
    cmd = f"ssh {host} {shlex.quote(remote_cmd)}"
    return _run_interactive_command(cmd, timeout=timeout, password_env_var=config.password_env_var)


def _rsync_path(
    src: str,
    dst: str,
    *,
    password_env_var: str,
    timeout: int = 7200,
    delete: bool = False,
) -> dict[str, object]:
    delete_flag = " --delete" if delete else ""
    cmd = f"rsync -az{delete_flag} --exclude='.git/' --exclude='.codex/' --exclude='__pycache__/' {src} {dst}"
    return _run_interactive_command(cmd, timeout=timeout, password_env_var=password_env_var)


def load_autofix_config(path: Path = DEFAULT_CONFIG_PATH) -> AutofixConfig:
    raw = _read_json(path) or {}
    workspace = WorkspaceConfig(**raw.get("workspace", {}))
    remote = RemoteConfig(**raw.get("remote", {}))
    matrix_raw = dict(raw.get("matrix", {}))
    phases_raw = matrix_raw.pop("phases", MatrixConfig().phases)
    if isinstance(phases_raw, str):
        phases = tuple(item.strip().lower() for item in phases_raw.split(",") if item.strip())
    else:
        phases = tuple(str(item).strip().lower() for item in phases_raw if str(item).strip())
    matrix = MatrixConfig(**matrix_raw, phases=phases or MatrixConfig().phases)
    diff_limits = DiffLimits(**raw.get("diff_limits", {}))
    push = PushPolicy(**raw.get("push", {}))
    stop = StopPolicy(**raw.get("stop", {}))
    test_commands = tuple(raw.get("test_commands", AutofixConfig().test_commands))
    recipe_enabled = dict(AutofixConfig().recipe_enabled)
    recipe_enabled.update(raw.get("recipe_enabled", {}))
    return AutofixConfig(
        workspace=workspace,
        remote=remote,
        matrix=matrix,
        diff_limits=diff_limits,
        push=push,
        stop=stop,
        total_hours=float(raw.get("total_hours", AutofixConfig().total_hours)),
        max_rounds=int(raw.get("max_rounds", AutofixConfig().max_rounds)),
        recipe_enabled=recipe_enabled,
        test_commands=test_commands,
    )


def _ensure_snapshot_workspace(*, source_repo: Path, workspace_root: Path, base_branch: str) -> Path:
    if workspace_root.exists() and (workspace_root / ".git").exists():
        return workspace_root
    source_origin = _run_local("git remote get-url origin", cwd=source_repo).stdout.strip()
    workspace_root.parent.mkdir(parents=True, exist_ok=True)
    _run_local(f"git clone --shared {shlex.quote(str(source_repo))} {shlex.quote(str(workspace_root))}", cwd=source_repo, check=True)
    if source_origin:
        _run_local(f"git remote set-url origin {shlex.quote(source_origin)}", cwd=workspace_root, check=True)
    _run_local(f"git checkout {shlex.quote(base_branch)}", cwd=workspace_root, check=True)
    exclude_path = workspace_root / ".git/info/exclude"
    exclude_text = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    additions = [".codex", ".codex/**", "__pycache__/", "*.pyc"]
    missing = [item for item in additions if item not in exclude_text.splitlines()]
    if missing:
        exclude_path.write_text(exclude_text.rstrip() + "\n" + "\n".join(missing) + "\n", encoding="utf-8")
    rsync_proc = subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude=.git/",
            "--exclude=.codex/",
            "--exclude=__pycache__/",
            f"{source_repo}/",
            f"{workspace_root}/",
        ],
        text=True,
        capture_output=True,
    )
    if rsync_proc.returncode != 0:
        raise RuntimeError(f"Failed to seed snapshot workspace:\n{rsync_proc.stdout}\n{rsync_proc.stderr}")
    if _run_local("git diff --quiet && git diff --cached --quiet", cwd=workspace_root).returncode != 0:
        _git_add_safe(workspace_root)
        _run_local("git commit -m 'autofix baseline snapshot'", cwd=workspace_root, check=True)
    return workspace_root


def _git_head(repo_root: Path) -> str:
    return _run_local("git rev-parse HEAD", cwd=repo_root, check=True).stdout.strip()


def _git_diff_stats(repo_root: Path) -> dict[str, object]:
    proc = _run_local("git diff --numstat", cwd=repo_root, check=True)
    changed_files = 0
    changed_lines = 0
    files: list[str] = []
    for raw in proc.stdout.splitlines():
        parts = raw.split("\t")
        if len(parts) != 3:
            continue
        add_s, del_s, path = parts
        add = 0 if add_s == "-" else int(add_s)
        delete = 0 if del_s == "-" else int(del_s)
        changed_files += 1
        changed_lines += add + delete
        files.append(path)
    return {
        "changed_files": changed_files,
        "changed_lines": changed_lines,
        "files": files,
    }


def _git_add_safe(repo_root: Path) -> None:
    codex_path = repo_root / ".codex"
    if codex_path.is_dir():
        shutil.rmtree(codex_path)
    elif codex_path.exists():
        codex_path.unlink()
    for pycache in repo_root.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
    for pyc in repo_root.rglob("*.pyc"):
        if pyc.exists():
            pyc.unlink()
    _run_local("git add -A", cwd=repo_root, check=True)


def _recipe_by_name(recipe_name: str) -> RecipeSpec | None:
    return next((item for item in RECIPES if item.name == recipe_name), None)


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-max_chars:]


def _map_remote_path(path_value: str | None, *, remote_base: Path, local_base: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    try:
        rel = path.relative_to(remote_base)
    except Exception:
        return None
    return local_base / rel


def _classify_failure_case(*, result: dict[str, object], manifest: dict[str, object] | None, progress: dict[str, object] | None, log_tail: str) -> dict[str, object]:
    acceptance = dict((manifest or {}).get("acceptance") or {})
    failed_checks = list(acceptance.get("failed_checks") or [])
    lower = log_tail.lower()
    failure_class = "success"
    if bool(acceptance.get("accepted")):
        failure_class = "success"
    elif bool(result.get("manifest_exists")) and manifest is not None:
        failure_class = "acceptance_failure"
    elif "excessive lateral compression" in lower:
        failure_class = "lateral_compression_failure"
    elif "rescue failed" in lower or "round_01_rescue" in lower:
        failure_class = "confined_rescue_failure"
    elif "failed to build final-xy walled" in lower or "catastrophic" in lower or "01_em" in lower:
        failure_class = "pack_overlap"
    elif "importerror" in lower or "no module named" in lower or "attributeerror" in lower:
        failure_class = "runtime_import_failure"
    elif int(result.get("returncode", 1)) != 0:
        failure_class = "runtime_failure"
    stack_gap = dict((manifest or {}).get("stack_gap_ang") or {})
    polymer_summary = dict((manifest or {}).get("polymer_phase_confined") or {})
    electrolyte_summary = dict((manifest or {}).get("electrolyte_phase_confined") or {})
    return {
        "case": result.get("case"),
        "phase": result.get("phase"),
        "returncode": int(result.get("returncode", 1)),
        "timed_out": bool(result.get("timed_out", False)),
        "progress_stage": None if progress is None else progress.get("stage"),
        "failure_class": failure_class,
        "accepted": bool(acceptance.get("accepted", False)),
        "failed_checks": failed_checks,
        "stack_gap_ang": {
            "graphite_to_polymer": float(stack_gap.get("graphite_to_polymer", 0.0) or 0.0),
            "polymer_to_electrolyte": float(stack_gap.get("polymer_to_electrolyte", 0.0) or 0.0),
        },
        "selected_rounds": {
            "polymer": (manifest or {}).get("polymer_phase_confined", {}).get("selected_round"),
            "electrolyte": (manifest or {}).get("electrolyte_phase_confined", {}).get("selected_round"),
        },
        "densities_g_cm3": {
            "polymer": float(acceptance.get("polymer_density_g_cm3", 0.0) or 0.0),
            "electrolyte": float(acceptance.get("electrolyte_density_g_cm3", 0.0) or 0.0),
        },
        "wrapped": {
            "polymer": bool((acceptance.get("wrapped_across_z_boundary") or {}).get("polymer", polymer_summary.get("wrapped_across_z_boundary", False))),
            "electrolyte": bool((acceptance.get("wrapped_across_z_boundary") or {}).get("electrolyte", electrolyte_summary.get("wrapped_across_z_boundary", False))),
        },
        "log_tail": log_tail,
    }


def build_failure_signature(*, matrix_dir: Path, remote_base_dir: Path) -> dict[str, object]:
    latest = _read_json(matrix_dir / "latest_status.json") or {"results": []}
    cases: list[dict[str, object]] = []
    acceptance_breakdown = {"polymer_density_ok": 0, "electrolyte_density_ok": 0, "wrapped_ok": 0, "order_ok": 0, "core_gaps_ok": 0}
    failure_counts: dict[str, int] = {}
    stack_gap_gp: list[float] = []
    stack_gap_pe: list[float] = []
    density_penalty = 0.0
    for result in latest.get("results", []):
        local_manifest = _map_remote_path(result.get("manifest_path"), remote_base=remote_base_dir, local_base=matrix_dir)
        local_progress = _map_remote_path(result.get("progress_path"), remote_base=remote_base_dir, local_base=matrix_dir)
        local_log = _map_remote_path(result.get("log_path"), remote_base=remote_base_dir, local_base=matrix_dir)
        manifest = _read_json(local_manifest) if local_manifest and local_manifest.exists() else None
        progress = _read_json(local_progress) if local_progress and local_progress.exists() else None
        log_tail = _tail_text(local_log) if local_log and local_log.exists() else ""
        case = _classify_failure_case(result=result, manifest=manifest, progress=progress, log_tail=log_tail)
        cases.append(case)
        failure_counts[case["failure_class"]] = failure_counts.get(case["failure_class"], 0) + 1
        for key in acceptance_breakdown:
            if key in case["failed_checks"]:
                acceptance_breakdown[key] += 1
        stack_gap_gp.append(float(case["stack_gap_ang"]["graphite_to_polymer"]))
        stack_gap_pe.append(float(case["stack_gap_ang"]["polymer_to_electrolyte"]))
        density_penalty += abs(case["densities_g_cm3"]["polymer"] - 1.50) + abs(case["densities_g_cm3"]["electrolyte"] - 1.28)
    primary_failure = "success"
    failing_counts = {k: v for k, v in failure_counts.items() if k != "success"}
    if failing_counts:
        primary_failure = max(sorted(failing_counts), key=lambda item: failing_counts[item])
    return {
        "generated_at": _timestamp(),
        "matrix_dir": str(matrix_dir),
        "iteration": latest.get("iteration"),
        "total_cases": len(cases),
        "accepted_cases": sum(1 for case in cases if case["accepted"]),
        "completed_cases": sum(1 for case in cases if case["failure_class"] in {"success", "acceptance_failure"}),
        "primary_failure_class": primary_failure,
        "failure_counts": failure_counts,
        "acceptance_failure_breakdown": acceptance_breakdown,
        "stack_gap_ang_stats": {
            "graphite_to_polymer_mean": sum(stack_gap_gp) / max(len(stack_gap_gp), 1),
            "polymer_to_electrolyte_mean": sum(stack_gap_pe) / max(len(stack_gap_pe), 1),
        },
        "density_penalty_total": density_penalty,
        "cases": cases,
    }


def build_round_metrics(signature: dict[str, object]) -> dict[str, object]:
    cases = list(signature.get("cases", []))
    total_failed_checks = sum(len(case.get("failed_checks", [])) for case in cases)
    wrapped_ok_cases = sum(1 for case in cases if not any(case.get("wrapped", {}).values()))
    metrics = {
        "accepted_cases": int(signature.get("accepted_cases", 0)),
        "completed_cases": int(signature.get("completed_cases", 0)),
        "wrapped_ok_cases": wrapped_ok_cases,
        "total_failed_checks": total_failed_checks,
        "density_penalty_total": float(signature.get("density_penalty_total", 0.0)),
        "primary_failure_class": str(signature.get("primary_failure_class", "success")),
    }
    metrics["comparison_key"] = (
        metrics["accepted_cases"],
        metrics["completed_cases"],
        wrapped_ok_cases,
        -total_failed_checks,
        -float(metrics["density_penalty_total"]),
    )
    return metrics


def compare_round_metrics(previous: dict[str, object] | None, current: dict[str, object]) -> dict[str, object]:
    if previous is None:
        return {"improved": True, "reason": "initial baseline", "comparison": [None, current.get("comparison_key")]}
    prev_key = tuple(previous.get("comparison_key", ()))
    curr_key = tuple(current.get("comparison_key", ()))
    return {
        "improved": curr_key > prev_key,
        "reason": "lexicographic metrics comparison",
        "comparison": [prev_key, curr_key],
    }


def _file_contains(repo_root: Path, relpath: str, needle: str) -> bool:
    path = repo_root / relpath
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8")


def _replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Patch anchor not found in {path}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def _recipe_fix_stack_gap_estimation(repo_root: Path) -> dict[str, object]:
    changed: list[str] = []
    metrics_path = repo_root / "src/yadonpy/interface/sandwich_metrics.py"
    sandwich_path = repo_root / "src/yadonpy/interface/sandwich.py"
    changed_any = False
    changed_any |= _replace_once(
        metrics_path,
        '"center_window_nm": 0.0,\n            "center_bulk_like_density_g_cm3": 0.0,',
        '"center_window_nm": 0.0,\n            "center_bulk_like_window_nm": [0.0, 0.0],\n            "center_bulk_like_density_g_cm3": 0.0,',
    )
    changed_any |= _replace_once(
        metrics_path,
        '"center_window_nm": center_window_nm,\n        "center_bulk_like_density_g_cm3": center_density,',
        '"center_window_nm": center_window_nm,\n        "center_bulk_like_window_nm": [center_lo, center_hi],\n        "center_bulk_like_density_g_cm3": center_density,',
    )
    changed_any |= _replace_once(
        sandwich_path,
        "    if not isinstance(window, (list, tuple)) or len(window) != 2:\n        return 0.0\n",
        "    if isinstance(window, (int, float)):\n        return max(0.0, float(window))\n    if not isinstance(window, (list, tuple)) or len(window) != 2:\n        return 0.0\n",
    )
    changed_any |= _replace_once(
        sandwich_path,
        '    core = _window_size_nm(summary.get("center_bulk_like_window_nm"))\n',
        '    core = _window_size_nm(summary.get("center_bulk_like_window_nm", summary.get("center_window_nm")))\n',
    )
    if changed_any:
        changed.extend(["src/yadonpy/interface/sandwich_metrics.py", "src/yadonpy/interface/sandwich.py"])
    return {"status": "applied" if changed_any else "noop", "files": changed}


def _recipe_trust_periodic_xy(repo_root: Path) -> dict[str, object]:
    path = repo_root / "src/yadonpy/interface/sandwich.py"
    changed = _replace_once(
        path,
        "        counts=counts,\n    )\n",
        "        counts=counts,\n        trust_periodic_xy=True,\n    )\n",
    )
    return {"status": "applied" if changed else "noop", "files": ["src/yadonpy/interface/sandwich.py"] if changed else []}


def _recipe_repair_harness(repo_root: Path) -> dict[str, object]:
    path = repo_root / "examples/08_graphite_polymer_electrolyte_sandwich/run_iteration_matrix.py"
    changed = False
    changed |= _replace_once(
        path,
        "BASE_DIR = Path(__file__).resolve().parent\n\n\n@dataclass(frozen=True)\n",
        "BASE_DIR = Path(__file__).resolve().parent\nREPO_ROOT = BASE_DIR.parent.parent\nSRC_DIR = REPO_ROOT / \"src\"\n\n\n@dataclass(frozen=True)\n",
    )
    changed |= _replace_once(
        path,
        '    env = os.environ.copy()\n    env["YADONPY_WORK_DIR"] = str(work_dir)\n',
        '    env = os.environ.copy()\n    existing_pythonpath = env.get("PYTHONPATH", "").strip()\n    env["PYTHONPATH"] = (\n        str(SRC_DIR)\n        if not existing_pythonpath\n        else os.pathsep.join([str(SRC_DIR), existing_pythonpath])\n    )\n    env["YADONPY_WORK_DIR"] = str(work_dir)\n',
    )
    changed |= _replace_once(
        path,
        "                        cwd=str(BASE_DIR),\n",
        "                        cwd=str(REPO_ROOT),\n",
    )
    return {"status": "applied" if changed else "noop", "files": ["examples/08_graphite_polymer_electrolyte_sandwich/run_iteration_matrix.py"] if changed else []}


def _recipe_reduce_release_gap(repo_root: Path) -> dict[str, object]:
    path = repo_root / "src/yadonpy/interface/sandwich.py"
    changed = False
    changed |= _replace_once(
        path,
        "    graphite_polymer_gap_nm = (float(relax.graphite_to_polymer_gap_ang) / 10.0) + 0.35 * polymer_shell_nm\n",
        "    graphite_polymer_gap_nm = (float(relax.graphite_to_polymer_gap_ang) / 10.0) + 0.18 * polymer_shell_nm\n",
    )
    changed |= _replace_once(
        path,
        "    polymer_electrolyte_gap_nm = (float(relax.polymer_to_electrolyte_gap_ang) / 10.0) + 0.35 * (\n",
        "    polymer_electrolyte_gap_nm = (float(relax.polymer_to_electrolyte_gap_ang) / 10.0) + 0.18 * (\n",
    )
    if changed:
        return {"status": "applied", "files": ["src/yadonpy/interface/sandwich.py"]}
    return {"status": "noop", "files": []}


def _recipe_tighten_confined_selection(repo_root: Path) -> dict[str, object]:
    path = repo_root / "src/yadonpy/interface/sandwich_metrics.py"
    changed = False
    changed |= _replace_once(path, "    if occupied_density < 0.80 * float(target_density_g_cm3):\n        score += 0.50\n", "    if occupied_density < 0.90 * float(target_density_g_cm3):\n        score += 0.90\n")
    changed |= _replace_once(path, "    if bool(summary.get(\"wrapped_across_z_boundary\", False)):\n        score += 1.00\n", "    if bool(summary.get(\"wrapped_across_z_boundary\", False)):\n        score += 1.75\n")
    changed |= _replace_once(path, "    if occupied_thickness > 1.08 * float(target_thickness_nm):\n", "    if occupied_thickness > 1.04 * float(target_thickness_nm):\n")
    changed |= _replace_once(path, "    return center_density < 0.85 * float(target_density_g_cm3)\n", "    return center_density < 0.90 * float(target_density_g_cm3)\n")
    return {"status": "applied" if changed else "noop", "files": ["src/yadonpy/interface/sandwich_metrics.py"] if changed else []}


def _always_false(_signature: dict[str, object], _repo_root: Path) -> bool:
    return False


def _has_missing_stack_gap_fix(signature: dict[str, object], repo_root: Path) -> bool:
    if float(signature.get("stack_gap_ang_stats", {}).get("polymer_to_electrolyte_mean", 0.0)) < 25.0:
        return False
    return not _file_contains(repo_root, "src/yadonpy/interface/sandwich_metrics.py", "center_bulk_like_window_nm")


def _needs_confined_selection_tightening(signature: dict[str, object], repo_root: Path) -> bool:
    if str(signature.get("primary_failure_class")) != "acceptance_failure":
        return False
    if _file_contains(repo_root, "src/yadonpy/interface/sandwich_metrics.py", "score += 1.75"):
        return False
    wrapped_cases = sum(1 for case in signature.get("cases", []) if any(case.get("wrapped", {}).values()))
    return wrapped_cases >= 2


def _needs_gap_reduction(signature: dict[str, object], repo_root: Path) -> bool:
    if str(signature.get("primary_failure_class")) != "acceptance_failure":
        return False
    if _needs_confined_selection_tightening(signature, repo_root):
        return False
    return float(signature.get("stack_gap_ang_stats", {}).get("polymer_to_electrolyte_mean", 0.0)) > 25.0


def _needs_periodic_xy_fix(signature: dict[str, object], repo_root: Path) -> bool:
    return str(signature.get("primary_failure_class")) == "lateral_compression_failure" and not _file_contains(
        repo_root,
        "src/yadonpy/interface/sandwich.py",
        "trust_periodic_xy=True",
    )


def _needs_harness_fix(signature: dict[str, object], repo_root: Path) -> bool:
    return str(signature.get("primary_failure_class")) == "runtime_import_failure" and not _file_contains(
        repo_root,
        "examples/08_graphite_polymer_electrolyte_sandwich/run_iteration_matrix.py",
        'env["PYTHONPATH"]',
    )


RECIPES: tuple[RecipeSpec, ...] = (
    RecipeSpec(
        name="repair_harness_import_sync_runtime_wiring",
        risk="low",
        files=("examples/08_graphite_polymer_electrolyte_sandwich/run_iteration_matrix.py",),
        tests=("python -m py_compile examples/08_graphite_polymer_electrolyte_sandwich/run_iteration_matrix.py",),
        push_allowed=True,
        selector=_needs_harness_fix,
        applier=_recipe_repair_harness,
    ),
    RecipeSpec(
        name="trust_periodic_xy_semantics",
        risk="medium",
        files=("src/yadonpy/interface/sandwich.py",),
        tests=("conda run -n yadonpy python -m pytest tests/test_sandwich_workflow.py -k normalize_confined_block_for_stack -q",),
        push_allowed=True,
        selector=_needs_periodic_xy_fix,
        applier=_recipe_trust_periodic_xy,
    ),
    RecipeSpec(
        name="fix_stack_gap_estimation",
        risk="medium",
        files=("src/yadonpy/interface/sandwich.py", "src/yadonpy/interface/sandwich_metrics.py"),
        tests=("conda run -n yadonpy python -m pytest tests/test_sandwich_workflow.py -k center_bulk_like_window_nm -q",),
        push_allowed=True,
        selector=_has_missing_stack_gap_fix,
        applier=_recipe_fix_stack_gap_estimation,
    ),
    RecipeSpec(
        name="tighten_or_relax_confined_selection",
        risk="medium",
        files=("src/yadonpy/interface/sandwich_metrics.py",),
        tests=("conda run -n yadonpy python -m pytest tests/test_sandwich_workflow.py -q",),
        push_allowed=True,
        selector=_needs_confined_selection_tightening,
        applier=_recipe_tighten_confined_selection,
    ),
    RecipeSpec(
        name="reduce_release_gap_or_padding",
        risk="medium",
        files=("src/yadonpy/interface/sandwich.py",),
        tests=("conda run -n yadonpy python -m pytest tests/test_sandwich_workflow.py tests/test_release_sanity.py -q",),
        push_allowed=True,
        selector=_needs_gap_reduction,
        applier=_recipe_reduce_release_gap,
    ),
    RecipeSpec(
        name="adjust_rescue_acceptance_preserve_best_round",
        risk="medium",
        files=("src/yadonpy/interface/sandwich.py",),
        tests=(),
        push_allowed=True,
        selector=_always_false,
        applier=lambda repo_root: {"status": "noop", "files": []},
    ),
    RecipeSpec(
        name="refine_wrap_density_scoring_or_candidate_ranking",
        risk="medium",
        files=("src/yadonpy/interface/sandwich_metrics.py",),
        tests=(),
        push_allowed=True,
        selector=_always_false,
        applier=lambda repo_root: {"status": "noop", "files": []},
    ),
    RecipeSpec(
        name="adjust_screening_route_fail_fast",
        risk="low",
        files=("src/yadonpy/interface/sandwich.py",),
        tests=(),
        push_allowed=True,
        selector=_always_false,
        applier=lambda repo_root: {"status": "noop", "files": []},
    ),
)


def select_recipe(*, signature: dict[str, object], repo_root: Path, config: AutofixConfig, attempted: set[str]) -> dict[str, object]:
    for recipe in RECIPES:
        if not config.recipe_enabled.get(recipe.name, False):
            continue
        if recipe.name in attempted:
            continue
        if recipe.selector(signature, repo_root):
            return {
                "selected": True,
                "recipe": recipe.name,
                "risk": recipe.risk,
                "files": list(recipe.files),
                "push_allowed": recipe.push_allowed,
            }
    return {
        "selected": False,
        "recipe": "unclassified_failure",
        "risk": "high",
        "files": [],
        "push_allowed": False,
    }


def apply_recipe(*, recipe_name: str, repo_root: Path) -> dict[str, object]:
    recipe = next((item for item in RECIPES if item.name == recipe_name), None)
    if recipe is None:
        return {"status": "unknown_recipe", "files": []}
    return recipe.applier(repo_root)


def _enforce_diff_limits(*, repo_root: Path, recipe_name: str, config: AutofixConfig) -> dict[str, object]:
    stats = _git_diff_stats(repo_root)
    recipe = next((item for item in RECIPES if item.name == recipe_name), None)
    allowed = set(recipe.files) if recipe is not None else set()
    bad_files = [path for path in stats["files"] if path not in allowed]
    ok = (
        stats["changed_files"] <= int(config.diff_limits.max_changed_files)
        and stats["changed_lines"] <= int(config.diff_limits.max_changed_lines)
        and not bad_files
    )
    return {
        **stats,
        "allowed_files": sorted(allowed),
        "unexpected_files": bad_files,
        "ok": ok,
    }


def run_verifier(*, repo_root: Path, recipe_name: str, config: AutofixConfig) -> dict[str, object]:
    diff_check = _enforce_diff_limits(repo_root=repo_root, recipe_name=recipe_name, config=config)
    commands = list(config.test_commands)
    recipe = next((item for item in RECIPES if item.name == recipe_name), None)
    if recipe is not None:
        for cmd in recipe.tests:
            if cmd not in commands:
                commands.append(cmd)
    runs = []
    all_ok = bool(diff_check["ok"])
    for cmd in commands:
        proc = _run_local(cmd, cwd=repo_root)
        entry = {"command": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
        runs.append(entry)
        if proc.returncode != 0:
            all_ok = False
    return {
        "ok": all_ok,
        "diff_check": diff_check,
        "runs": runs,
    }


def _build_push_safety_report(
    *,
    repo_root: Path,
    push: PushPolicy,
    recipe_name: str,
    before_metrics: dict[str, object],
    after_metrics: dict[str, object],
    comparison: dict[str, object],
) -> dict[str, object]:
    recipe = _recipe_by_name(recipe_name)
    stats = _git_diff_stats(repo_root)
    code_files = [
        path
        for path in stats["files"]
        if path.startswith(("src/", "examples/08_graphite_polymer_electrolyte_sandwich/", "tests/"))
    ]
    failures: list[str] = []
    if not push.enabled:
        return {"ok": True, "push_enabled": False, "checks": ["push disabled"], "failures": []}
    if recipe is None:
        failures.append("unknown recipe")
    elif not recipe.push_allowed:
        failures.append("recipe does not allow push")
    elif recipe.risk == "high":
        failures.append("high-risk recipe cannot auto-push")
    if not bool(comparison.get("improved")):
        failures.append("remote matrix did not improve")
    if int(after_metrics.get("completed_cases", 0)) < int(before_metrics.get("completed_cases", 0)):
        failures.append("completed case count regressed")
    if after_metrics.get("primary_failure_class") == "unclassified_failure":
        failures.append("unclassified failure cannot auto-push")
    if not stats["files"]:
        failures.append("no code changes to commit")
    if not code_files:
        failures.append("diff is docs-only or outside protected code/test paths")
    if push.branch == "main" and push.require_env_var:
        unlocked = os.environ.get(push.require_env_var, "").strip().lower()
        if unlocked not in {"1", "true", "yes", "main"}:
            failures.append(f"main push requires {push.require_env_var}=1")
    return {
        "ok": not failures,
        "push_enabled": bool(push.enabled),
        "recipe": recipe_name,
        "recipe_risk": recipe.risk if recipe is not None else "unknown",
        "recipe_push_allowed": bool(recipe.push_allowed) if recipe is not None else False,
        "changed_files": stats["files"],
        "changed_lines": stats["changed_lines"],
        "comparison": comparison,
        "failures": failures,
    }


def _git_commit_and_push(*, repo_root: Path, push: PushPolicy, round_id: int, recipe_name: str, before_metrics: dict[str, object], after_metrics: dict[str, object]) -> dict[str, object]:
    _git_add_safe(repo_root)
    message = (
        f"autofix(example08): round {round_id:03d} {recipe_name}\n\n"
        f"before={json.dumps(before_metrics, ensure_ascii=False)}\n"
        f"after={json.dumps(after_metrics, ensure_ascii=False)}\n"
    )
    _run_local(f"git commit -m {shlex.quote(message)}", cwd=repo_root, check=True)
    if push.enabled:
        _run_local(f"git push {shlex.quote(push.remote)} {shlex.quote(push.branch)}", cwd=repo_root, check=True)
    return {"head": _git_head(repo_root), "pushed": bool(push.enabled)}


def _sync_workspace_to_remote(*, workspace_root: Path, remote: RemoteConfig) -> dict[str, object]:
    return _rsync_path(
        f"{workspace_root}/",
        f"{remote.host}:{remote.repo_root}/",
        delete=True,
        password_env_var=remote.password_env_var,
    )


def _run_remote_round(*, config: AutofixConfig, remote_round_dir: str) -> dict[str, object]:
    remote_cmd = (
        f"rm -rf {shlex.quote(remote_round_dir)} && "
        f"mkdir -p {shlex.quote(remote_round_dir)} && "
        f"cd {shlex.quote(config.remote.repo_root)} && "
        f"{config.remote.conda_prefix} python examples/08_graphite_polymer_electrolyte_sandwich/run_iteration_matrix.py "
        f"--mode observe --base-dir {shlex.quote(remote_round_dir)} "
        f"--hours {float(config.matrix.hours_per_round):.3f} "
        f"--max-iterations {int(config.matrix.max_iterations_per_round)} "
        f"--timeout-min {int(config.matrix.timeout_min)} "
        f"--phases {shlex.quote(','.join(config.matrix.phases))}"
    )
    return _run_remote_command(config.remote, remote_cmd, timeout=max(3600, int(config.matrix.timeout_min) * 60 * 2))


def _sync_remote_round_to_local(*, config: AutofixConfig, remote_round_dir: str, local_matrix_dir: Path) -> dict[str, object]:
    local_matrix_dir.mkdir(parents=True, exist_ok=True)
    src = f"{config.remote.host}:{remote_round_dir}/"
    dst = f"{local_matrix_dir}/"
    cmd = (
        "rsync -az --delete "
        "--include='*/' --include='*.json' --include='*.jsonl' --include='*.log' "
        f"--exclude='*' {src} {dst}"
    )
    return _run_interactive_command(cmd, timeout=7200, password_env_var=config.remote.password_env_var)


def _write_stop_files(*, base_dir: Path, stop_reason: dict[str, object], last_good_round: dict[str, object] | None) -> None:
    _write_json(base_dir / "stop_reason.json", stop_reason)
    if last_good_round is not None:
        _write_json(base_dir / "last_good_round.json", last_good_round)
    hint = base_dir / "next_manual_hint.md"
    hint.write_text(
        "\n".join(
            [
                "# Next Manual Hint",
                "",
                f"- stop reason: `{stop_reason.get('reason', 'unknown')}`",
                f"- timestamp: `{stop_reason.get('stopped_at', _timestamp())}`",
                f"- inspect: `{base_dir}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_autofix_loop(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    base_dir: Path = BASE_DIR / "autofix_runs",
    total_hours: float | None = None,
    dry_run: bool = False,
) -> int:
    config = load_autofix_config(config_path)
    if total_hours is not None:
        config = replace(config, total_hours=float(total_hours))
    base_dir = base_dir.expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    _write_json(base_dir / "config_snapshot.json", json.loads(json.dumps(asdict(config))))

    workspace_root = _ensure_snapshot_workspace(
        source_repo=REPO_ROOT,
        workspace_root=(base_dir / config.workspace.path),
        base_branch=config.workspace.base_branch,
    )
    state = {
        "started_at": _timestamp(),
        "workspace_root": str(workspace_root),
        "source_repo": str(REPO_ROOT),
        "remote_repo_root": config.remote.repo_root,
        "remote_run_root": config.remote.run_root,
    }
    _write_json(base_dir / "autofix_state.json", state)
    if dry_run:
        return 0

    deadline = time.time() + max(0.25, float(config.total_hours)) * 3600.0
    round_idx = 0
    baseline_signature: dict[str, object] | None = None
    baseline_metrics: dict[str, object] | None = None
    pending_recipe: str | None = None
    pending_baseline_round: int | None = None
    attempted_for_baseline: set[str] = set()
    stagnant_rounds = 0
    repeated_primary_failure = 0
    last_primary_failure: str | None = None
    last_good_round: dict[str, object] | None = None
    last_good_commit = _git_head(workspace_root)

    while time.time() < deadline:
        if int(config.max_rounds) > 0 and round_idx >= int(config.max_rounds):
            break
        round_idx += 1
        round_dir = base_dir / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        remote_round_dir = f"{config.remote.run_root}/round_{round_idx:03d}"
        local_matrix_dir = round_dir / "matrix"

        sync_result = _sync_workspace_to_remote(workspace_root=workspace_root, remote=config.remote)
        _write_json(round_dir / "sync_to_remote.json", sync_result)
        if int(sync_result.get("returncode", 1)) != 0:
            _write_stop_files(
                base_dir=base_dir,
                stop_reason={"reason": "remote_sync_failed", "round": round_idx, "stopped_at": _timestamp()},
                last_good_round=last_good_round,
            )
            return 1

        remote_run = _run_remote_round(config=config, remote_round_dir=remote_round_dir)
        _write_json(round_dir / "remote_run.json", remote_run)
        if int(remote_run.get("returncode", 1)) != 0:
            _write_stop_files(
                base_dir=base_dir,
                stop_reason={"reason": "remote_round_failed", "round": round_idx, "stopped_at": _timestamp()},
                last_good_round=last_good_round,
            )
            return 1

        sync_back = _sync_remote_round_to_local(config=config, remote_round_dir=remote_round_dir, local_matrix_dir=local_matrix_dir)
        _write_json(round_dir / "sync_from_remote.json", sync_back)
        signature = build_failure_signature(matrix_dir=local_matrix_dir, remote_base_dir=Path(remote_round_dir))
        metrics = build_round_metrics(signature)
        _write_json(round_dir / "failure_signature.json", signature)
        _write_json(round_dir / "round_metrics.json", metrics)

        if pending_recipe is not None and baseline_metrics is not None and baseline_signature is not None:
            comparison = compare_round_metrics(baseline_metrics, metrics)
            _write_json(round_dir / "comparison.json", comparison)
            if comparison["improved"]:
                push_safety = _build_push_safety_report(
                    repo_root=workspace_root,
                    push=config.push,
                    recipe_name=pending_recipe,
                    before_metrics=baseline_metrics,
                    after_metrics=metrics,
                    comparison=comparison,
                )
                _write_json(round_dir / "push_safety.json", push_safety)
                if not push_safety["ok"]:
                    _write_stop_files(
                        base_dir=base_dir,
                        stop_reason={
                            "reason": "push_safety_failed",
                            "round": round_idx,
                            "failures": push_safety.get("failures", []),
                            "stopped_at": _timestamp(),
                        },
                        last_good_round=last_good_round,
                    )
                    return 1
                commit_info = _git_commit_and_push(
                    repo_root=workspace_root,
                    push=config.push,
                    round_id=round_idx,
                    recipe_name=pending_recipe,
                    before_metrics=baseline_metrics,
                    after_metrics=metrics,
                )
                _write_json(round_dir / "git_commit.json", commit_info)
                last_good_commit = _git_head(workspace_root)
                last_good_round = {"round": round_idx, "metrics": metrics, "signature_path": str(round_dir / "failure_signature.json")}
                baseline_signature = signature
                baseline_metrics = metrics
                pending_recipe = None
                pending_baseline_round = None
                attempted_for_baseline = set()
                stagnant_rounds = 0
            else:
                _run_local(f"git reset --hard {shlex.quote(last_good_commit)}", cwd=workspace_root, check=True)
                pending_recipe = None
                pending_baseline_round = None
                stagnant_rounds += 1
                if stagnant_rounds >= int(config.stop.stagnant_rounds):
                    _write_stop_files(
                        base_dir=base_dir,
                        stop_reason={"reason": "stagnant_rounds_exceeded", "round": round_idx, "stopped_at": _timestamp()},
                        last_good_round=last_good_round,
                    )
                    return 0
                signature = baseline_signature
                metrics = baseline_metrics

        if baseline_signature is None:
            baseline_signature = signature
            baseline_metrics = metrics

        primary_failure = str(signature.get("primary_failure_class", "success"))
        if primary_failure == "success" and int(signature.get("accepted_cases", 0)) == int(signature.get("total_cases", 0)):
            _write_stop_files(
                base_dir=base_dir,
                stop_reason={"reason": "all_cases_accepted", "round": round_idx, "stopped_at": _timestamp()},
                last_good_round=last_good_round or {"round": round_idx, "metrics": metrics},
            )
            return 0

        if primary_failure == last_primary_failure:
            repeated_primary_failure += 1
        else:
            repeated_primary_failure = 1
            last_primary_failure = primary_failure
        if repeated_primary_failure >= int(config.stop.repeated_failure_rounds) and primary_failure not in {"acceptance_failure", "success"}:
            _write_stop_files(
                base_dir=base_dir,
                stop_reason={"reason": "repeated_crash_class", "failure_class": primary_failure, "round": round_idx, "stopped_at": _timestamp()},
                last_good_round=last_good_round,
            )
            return 0

        decision_attempts: list[dict[str, object]] = []
        mutation_attempts: list[dict[str, object]] = []
        decision: dict[str, object] | None = None
        mutation: dict[str, object] | None = None
        while True:
            candidate = select_recipe(
                signature=baseline_signature,
                repo_root=workspace_root,
                config=config,
                attempted=attempted_for_baseline,
            )
            decision_attempts.append(candidate)
            if not candidate["selected"]:
                _write_json(round_dir / "decision_attempts.json", {"attempts": decision_attempts, "mutations": mutation_attempts})
                _write_json(round_dir / "decision.json", candidate)
                _write_stop_files(
                    base_dir=base_dir,
                    stop_reason={"reason": "unclassified_failure", "round": round_idx, "stopped_at": _timestamp()},
                    last_good_round=last_good_round,
                )
                return 0
            attempted_for_baseline.add(str(candidate["recipe"]))
            candidate_mutation = apply_recipe(recipe_name=str(candidate["recipe"]), repo_root=workspace_root)
            mutation_attempts.append({"recipe": candidate["recipe"], **candidate_mutation})
            if candidate_mutation["status"] == "noop":
                continue
            if candidate_mutation["status"] != "applied":
                _write_json(round_dir / "decision_attempts.json", {"attempts": decision_attempts, "mutations": mutation_attempts})
                _write_json(round_dir / "decision.json", candidate)
                _write_json(round_dir / "mutation.json", candidate_mutation)
                _write_stop_files(
                    base_dir=base_dir,
                    stop_reason={"reason": "mutation_failed", "recipe": candidate["recipe"], "round": round_idx, "stopped_at": _timestamp()},
                    last_good_round=last_good_round,
                )
                return 0
            decision = candidate
            mutation = candidate_mutation
            break

        _write_json(round_dir / "decision_attempts.json", {"attempts": decision_attempts, "mutations": mutation_attempts})
        _write_json(round_dir / "decision.json", decision)
        _write_json(round_dir / "mutation.json", mutation)

        verify = run_verifier(repo_root=workspace_root, recipe_name=str(decision["recipe"]), config=config)
        _write_json(round_dir / "verify.json", verify)
        if not verify["ok"]:
            _run_local(f"git reset --hard {shlex.quote(last_good_commit)}", cwd=workspace_root, check=True)
            _write_stop_files(
                base_dir=base_dir,
                stop_reason={"reason": "verification_failed", "recipe": decision["recipe"], "round": round_idx, "stopped_at": _timestamp()},
                last_good_round=last_good_round,
            )
            return 0

        pending_recipe = str(decision["recipe"])
        pending_baseline_round = round_idx
        history_entry = {
            "round": round_idx,
            "timestamp": _timestamp(),
            "primary_failure_class": primary_failure,
            "selected_recipe": pending_recipe,
            "pending_baseline_round": pending_baseline_round,
        }
        with (base_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history_entry, ensure_ascii=False) + "\n")

    _write_stop_files(
        base_dir=base_dir,
        stop_reason={"reason": "time_budget_exhausted", "round": round_idx, "stopped_at": _timestamp()},
        last_good_round=last_good_round,
    )
    return 0
