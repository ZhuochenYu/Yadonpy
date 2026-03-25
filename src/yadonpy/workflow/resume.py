"""Resumable workflow state manager.

Design goals:
  - Make example scripts and user workflows restartable without re-running
    expensive steps.
  - Be conservative: only skip a step if all expected outputs exist.
  - Avoid heavy dependencies.

The manager records a small JSON state file under the workflow directory.
Each step stores an input signature and a list of expected outputs. On rerun,
if outputs exist and the signature matches, the step is skipped.

This is *not* a job scheduler; it is a thin helper for sequential scripts.
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

from ..core.logging_utils import yadon_print

# -----------------------------------------------------------------------------
# Timing helpers (lightweight, no external deps)
# -----------------------------------------------------------------------------
_SCRIPT_T0 = time.perf_counter()
_TIMING_REGISTERED = False

def _fmt_duration(seconds: float) -> str:
    # format as H:MM:SS (or M:SS for short runs)
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"

def _now_hms() -> str:
    return time.strftime('%H:%M:%S', time.localtime())

def _register_total_timer_once() -> None:
    global _TIMING_REGISTERED
    if _TIMING_REGISTERED:
        return
    _TIMING_REGISTERED = True
    import atexit
    def _print_total() -> None:
        total = time.perf_counter() - _SCRIPT_T0
        yadon_print(f"[TIME] Total elapsed: {_fmt_duration(total)} | finished_at={_now_hms()}", level=1)
    atexit.register(_print_total)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _is_jsonable(x: Any) -> bool:
    try:
        json.dumps(x, ensure_ascii=False, sort_keys=True)
        return True
    except Exception:
        return False


def _normalize_inputs(inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not inputs:
        return {}
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, Path):
            out[k] = {"__path__": str(v)}
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        elif _is_jsonable(v):
            out[k] = v
        else:
            # best effort stringification (keeps state file readable)
            out[k] = repr(v)
    return out


def file_signature(path: Path) -> Dict[str, Any]:
    """Stable signature for a file: path + size + content digest.

    Resume checks should not be invalidated just because a workflow rewrote the
    same bytes and bumped the file mtime. We therefore key on file content
    rather than timestamps.
    """
    try:
        path = Path(path)
        st = path.stat()
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                if not chunk:
                    break
                h.update(chunk)
        return {"path": str(path), "size": int(st.st_size), "sha256": h.hexdigest()}
    except FileNotFoundError:
        return {"path": str(path), "missing": True}


def dir_signature(path: Path, *, patterns: Sequence[str] = ("*",), max_files: int = 2000) -> Dict[str, Any]:
    """Signature for a directory by sampling file stats.

    This is intended for job folders (e.g. GROMACS stages). We cap the number of
    files to avoid huge state.
    """
    if not path.exists():
        return {"path": str(path), "missing": True}
    files: list[Dict[str, Any]] = []
    n = 0
    for pat in patterns:
        for p in sorted(path.glob(pat)):
            if p.is_file():
                files.append(file_signature(p))
                n += 1
                if n >= max_files:
                    break
        if n >= max_files:
            break
    return {"path": str(path), "files": files, "truncated": n >= max_files}


def inputs_hash(inputs: Optional[Dict[str, Any]]) -> str:
    """Stable hash for a dict of inputs (JSON serialized)."""
    norm = _normalize_inputs(inputs)
    blob = json.dumps(norm, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class StepSpec:
    """Specification for a resumable step."""

    name: str
    outputs: Sequence[Path]
    inputs: Optional[Dict[str, Any]] = None
    description: str = ""


class ResumeManager:
    """Manage step-level resume for a workflow directory."""

    def __init__(
        self,
        root: Union[Path, str],
        *,
        # Store resume state in a hidden subdir by default, to keep `work_dir/`
        # clean and focused on scientific artifacts.
        state_name: str = ".yadonpy/resume_state.json",
        strict_inputs: bool = False,
        env_disable: str = "YADONPY_NO_RESUME",
        enabled: bool = True,
    ):
        self.root = Path(root).expanduser().resolve()
        _register_total_timer_once()  # step-level timing + total runtime at exit
        self.root.mkdir(parents=True, exist_ok=True)

        # Backward compatible migration: if an old-style state file exists at
        # the root, move it into the new hidden folder (best-effort).
        desired = self.root / state_name
        legacy = self.root / "resume_state.json"
        if state_name != "resume_state.json" and (not desired.exists()) and legacy.exists():
            try:
                desired.parent.mkdir(parents=True, exist_ok=True)
                os.replace(legacy, desired)
            except Exception:
                # If migration fails, keep using the legacy location.
                desired = legacy

        self.state_path = desired
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.strict_inputs = bool(strict_inputs)
        self.env_disable = env_disable
        # Global switch for resume logic. When disabled, steps are never skipped.
        self.enabled = bool(enabled)
        self._state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                # corrupted state -> keep a backup and start fresh
                bak = self.state_path.with_suffix(self.state_path.suffix + ".bak")
                try:
                    bak.write_text(self.state_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                except Exception:
                    pass
        return {"schema_version": 1, "root": str(self.root), "steps": {}}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.state_path)

    def disable_resume(self) -> bool:
        v = os.environ.get(self.env_disable)
        return bool(v and v.strip() not in ("0", "false", "False", "no", "NO"))

    @staticmethod
    def _outputs_exist(outputs: Sequence[Path]) -> bool:
        return all(Path(p).exists() for p in outputs)

    def reuse_status(self, spec: StepSpec) -> str:
        """Return a small status label describing whether a step can be reused."""
        outputs_ok = self._outputs_exist(spec.outputs)
        if not outputs_ok:
            return "missing_outputs"
        if not self.enabled:
            return "resume_disabled"
        if self.disable_resume():
            return "resume_env_disabled"
        rec = self._state.get("steps", {}).get(spec.name)
        if not rec:
            return "done" if not self.strict_inputs else "no_record"
        if rec.get("status") != "done":
            return "not_done"
        if self.strict_inputs:
            want = inputs_hash(spec.inputs)
            have = rec.get("inputs_hash")
            if have != want:
                return "inputs_mismatch"
        return "done"

    def needs_fresh_run(self, spec: StepSpec) -> bool:
        """Whether stale artifacts exist and the step should rebuild from scratch."""
        status = self.reuse_status(spec)
        if status == "done":
            return False
        rec = self._state.get("steps", {}).get(spec.name)
        outputs_ok = self._outputs_exist(spec.outputs)
        return bool(rec or outputs_ok)

    def is_done(self, spec: StepSpec) -> bool:
        return self.reuse_status(spec) == "done"

    def mark_done(self, spec: StepSpec, *, meta: Optional[Dict[str, Any]] = None) -> None:
        self._state.setdefault("steps", {})
        self._state["steps"][spec.name] = {
            "status": "done",
            "inputs_hash": inputs_hash(spec.inputs),
            "outputs": [str(Path(p)) for p in spec.outputs],
            "updated_at": _now_iso(),
            "meta": meta or {},
        }
        self._save_state()

    def mark_failed(self, spec: StepSpec, *, error: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self._state.setdefault("steps", {})
        self._state["steps"][spec.name] = {
            "status": "failed",
            "inputs_hash": inputs_hash(spec.inputs),
            "outputs": [str(Path(p)) for p in spec.outputs],
            "updated_at": _now_iso(),
            "error": str(error),
            "meta": meta or {},
        }
        self._save_state()

    def invalidate_steps(
        self,
        *,
        names: Sequence[str] = (),
        prefixes: Sequence[str] = (),
    ) -> list[str]:
        """Forget cached step records so downstream stages must rebuild.

        This only mutates the lightweight resume state; it intentionally does
        not delete any scientific artifacts on disk. Callers can decide whether
        stale folders should also be cleaned when the invalidated step reruns.
        """
        wanted_names = {str(name) for name in names}
        wanted_prefixes = tuple(str(prefix) for prefix in prefixes if str(prefix))
        if not wanted_names and not wanted_prefixes:
            return []

        steps = self._state.setdefault("steps", {})
        removed: list[str] = []
        for key in list(steps.keys()):
            if key in wanted_names or any(key.startswith(prefix) for prefix in wanted_prefixes):
                steps.pop(key, None)
                removed.append(key)
        if removed:
            self._save_state()
        return removed

    def run(
        self,
        spec: StepSpec,
        func: Callable[[], Any],
        *,
        meta: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> Any:
        """Run a step if needed.

        Returns the function return value if executed, or None if skipped.
        """
        if self.is_done(spec):
            if verbose:
                total = time.perf_counter() - _SCRIPT_T0
                yadon_print(f"[SKIP] {spec.name} | already done | total={_fmt_duration(total)} | at={_now_hms()}", level=1)
            return None

        t0 = time.perf_counter()
        if verbose:
            total = t0 - _SCRIPT_T0
            yadon_print(f"[RUN] {spec.name} | start_at={_now_hms()} | total={_fmt_duration(total)}", level=1)

        try:
            out = func()
        except Exception as e:
            self.mark_failed(spec, error=str(e), meta=meta)
            raise

        # Only mark done if outputs exist.
        if not self._outputs_exist(spec.outputs):
            err = f"Step '{spec.name}' finished but expected outputs are missing: {[str(p) for p in spec.outputs]}"
            self.mark_failed(spec, error=err, meta=meta)
            raise RuntimeError(err)

        self.mark_done(spec, meta=meta)
        if verbose:
            elapsed = time.perf_counter() - t0
            total = time.perf_counter() - _SCRIPT_T0
            yadon_print(f"[DONE] {spec.name} | elapsed={_fmt_duration(elapsed)} | total={_fmt_duration(total)} | end_at={_now_hms()}", level=1)
        return out
