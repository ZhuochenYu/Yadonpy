"""Runtime options (global defaults) for YadonPy workflows.

This module exists so users can set `restart` once (globally) and then omit
`restart=` in downstream function calls.

- Programmatic: `set_run_options(restart=True)`
- Environment:  `YADONPY_RESTART=1` (read at import time)

Explicit keyword arguments in API calls always override these defaults.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Iterator, Optional


def _parse_bool(value: str, *, default: bool) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


@dataclass(frozen=True)
class RunOptions:
    restart: bool = True
    strict_inputs: bool = True


@dataclass(frozen=True)
class RecommendedResources:
    mpi: int = 1
    omp: int = 1
    gpu: int = 1
    gpu_id: int | None = 0
    omp_psi4: int = 1
    cpu_total: int = 1
    cpu_cap: int | None = None


def _default_run_options() -> RunOptions:
    opt = RunOptions()
    env_restart = os.getenv("YADONPY_RESTART")
    if env_restart is not None:
        opt = replace(opt, restart=_parse_bool(env_restart, default=opt.restart))
    env_strict = os.getenv("YADONPY_STRICT_INPUTS")
    if env_strict is not None:
        opt = replace(opt, strict_inputs=_parse_bool(env_strict, default=opt.strict_inputs))
    return opt


_RUN_OPTIONS: ContextVar[RunOptions] = ContextVar("yadonpy_run_options", default=_default_run_options())


def get_run_options() -> RunOptions:
    """Return the current run options (context-local)."""
    return _RUN_OPTIONS.get()


def set_run_options(*, restart: Optional[bool] = None, strict_inputs: Optional[bool] = None) -> RunOptions:
    """Update global defaults for the current context and return the new options."""
    cur = _RUN_OPTIONS.get()
    new = cur
    if restart is not None:
        new = replace(new, restart=bool(restart))
    if strict_inputs is not None:
        new = replace(new, strict_inputs=bool(strict_inputs))
    _RUN_OPTIONS.set(new)
    return new


def resolve_restart(restart: Optional[bool]) -> bool:
    """Resolve an optional `restart` argument against the global default."""
    return get_run_options().restart if restart is None else bool(restart)


def resolve_strict_inputs(strict_inputs: Optional[bool]) -> bool:
    """Resolve an optional `strict_inputs` argument against the global default."""
    return get_run_options().strict_inputs if strict_inputs is None else bool(strict_inputs)


def _parse_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def recommend_local_resources(
    *,
    cpu_cap: int | None = None,
    mpi_default: int = 1,
    gpu_default: int = 1,
    gpu_id_default: int | None = 0,
    omp_psi4_cap: int | None = 8,
) -> RecommendedResources:
    """Recommend a conservative local resource layout for example scripts.

    The recommendation is intentionally simple:
      - default to one thread-MPI rank;
      - keep OpenMP within the visible CPU budget;
      - allow env vars to override the defaults;
      - cap the optional QM thread count more aggressively than MD threads.
    """

    detected_cpu_total = max(1, _parse_int(os.cpu_count(), default=1))
    cap = None if cpu_cap is None else max(1, _parse_int(cpu_cap, default=detected_cpu_total))
    visible_cpu_total = min(detected_cpu_total, cap) if cap is not None else detected_cpu_total

    mpi = max(1, _parse_int(os.environ.get("YADONPY_MPI"), default=mpi_default))
    omp_default = max(1, visible_cpu_total // max(1, mpi))
    omp = max(1, _parse_int(os.environ.get("YADONPY_OMP"), default=omp_default))
    omp = min(omp, visible_cpu_total)

    gpu_env = os.environ.get("YADONPY_GPU")
    if gpu_env is None:
        gpu = int(bool(gpu_default))
    else:
        gpu = 1 if _parse_bool(gpu_env, default=bool(gpu_default)) else 0

    gpu_id_env = os.environ.get("YADONPY_GPU_ID")
    if gpu and gpu_id_env is not None and str(gpu_id_env).strip() != "":
        gpu_id = _parse_int(gpu_id_env, default=(gpu_id_default if gpu_id_default is not None else 0))
    else:
        gpu_id = int(gpu_id_default) if (gpu and gpu_id_default is not None) else None

    omp_psi4_default = min(omp, max(1, _parse_int(omp_psi4_cap, default=omp)))
    omp_psi4 = max(1, _parse_int(os.environ.get("YADONPY_OMP_PSI4"), default=omp_psi4_default))
    omp_psi4 = min(omp_psi4, omp)

    return RecommendedResources(
        mpi=int(mpi),
        omp=int(omp),
        gpu=int(gpu),
        gpu_id=(int(gpu_id) if gpu_id is not None else None),
        omp_psi4=int(omp_psi4),
        cpu_total=int(detected_cpu_total),
        cpu_cap=(int(cap) if cap is not None else None),
    )


@contextmanager
def run_options(*, restart: Optional[bool] = None, strict_inputs: Optional[bool] = None) -> Iterator[RunOptions]:
    """Temporarily override run options within a `with` block."""
    cur = _RUN_OPTIONS.get()
    new = cur
    if restart is not None:
        new = replace(new, restart=bool(restart))
    if strict_inputs is not None:
        new = replace(new, strict_inputs=bool(strict_inputs))
    token = _RUN_OPTIONS.set(new)
    try:
        yield new
    finally:
        _RUN_OPTIONS.reset(token)


__all__ = [
    "RecommendedResources",
    "RunOptions",
    "get_run_options",
    "recommend_local_resources",
    "run_options",
    "resolve_restart",
    "resolve_strict_inputs",
    "set_run_options",
]
