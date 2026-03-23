from __future__ import annotations

"""Runtime options (global defaults) for YadonPy workflows.

This module exists so users can set `restart` once (globally) and then omit
`restart=` in downstream function calls.

- Programmatic: `set_run_options(restart=True)`
- Environment:  `YADONPY_RESTART=1` (read at import time)

Explicit keyword arguments in API calls always override these defaults.
"""

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
    strict_inputs: bool = False


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
    "RunOptions",
    "get_run_options",
    "run_options",
    "resolve_restart",
    "resolve_strict_inputs",
    "set_run_options",
]
