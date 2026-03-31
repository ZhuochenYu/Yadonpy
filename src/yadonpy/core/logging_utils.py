"""Lightweight printing/logging helpers.

This module is used by many parts of YadonPy very early in the import chain.
Keep it dependency-light and avoid importing heavy modules here.
"""

from __future__ import annotations

import os
import re
import sys

from .._version import __version__ as _YADONPY_VERSION
from . import const
from .exceptions import YadonPyError


_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
}


def _use_color() -> bool:
    # Respect NO_COLOR (https://no-color.org/)
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


_TAG_STYLE = {
    "CMD": _ANSI["blue"] + _ANSI["bold"],
    "RUN": _ANSI["cyan"] + _ANSI["bold"],
    "DONE": _ANSI["green"] + _ANSI["bold"],
    "OK": _ANSI["green"] + _ANSI["bold"],
    "SKIP": _ANSI["gray"] + _ANSI["bold"],
    "WARN": _ANSI["yellow"] + _ANSI["bold"],
    "ERROR": _ANSI["red"] + _ANSI["bold"],
    "SECTION": _ANSI["magenta"] + _ANSI["bold"],
    "STEP": _ANSI["cyan"] + _ANSI["bold"],
    "ITEM": _ANSI["blue"] + _ANSI["bold"],
    "STAT": _ANSI["green"] + _ANSI["bold"],
    "NOTE": _ANSI["gray"] + _ANSI["bold"],
    "INFO": _ANSI["blue"] + _ANSI["bold"],
    "TIME": _ANSI["magenta"] + _ANSI["bold"],
    "ANALYZE": _ANSI["cyan"] + _ANSI["bold"],
}


_TAG_RE = re.compile(r"\[(?P<tag>[A-Za-z0-9_+-]{2,10})\s*\]")
_version_banner_printed = False


def _colorize_tags(s: str) -> str:
    if not s or (not _use_color()):
        return s

    def _repl(m: re.Match) -> str:
        tag_raw = (m.group("tag") or "").strip()
        tag_key = tag_raw.upper()
        style = _TAG_STYLE.get(tag_key)
        if not style:
            return m.group(0)
        return f"[{style}{tag_raw}{_ANSI['reset']}]"

    return _TAG_RE.sub(_repl, s)


def emit_version_banner() -> None:
    global _version_banner_printed
    if _version_banner_printed:
        return
    print(_colorize_tags(f"YadonPy info: [INFO] version={_YADONPY_VERSION}"), flush=True)
    _version_banner_printed = True


def yadon_print(text, level=0):
    """Unified console logger.

    Levels:
      0 debug, 1 info, 2 warning, 3 error (raise)
    """
    if level == 0:
        text = 'YadonPy debug info: ' + str(text)
    elif level == 1:
        text = 'YadonPy info: ' + str(text)
    elif level == 2:
        text = 'YadonPy warning: ' + str(text)
    elif level == 3:
        emit_version_banner()
        raise YadonPyError(str(text))

    if level >= const.print_level or const.debug:
        emit_version_banner()
        print(_colorize_tags(str(text)), flush=True)


def radon_print(text, level=0):
    return yadon_print(text, level=level)


def tqdm_stub(it, **kwargs):
    return it


def format_elapsed(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def compact_path(path_like, *, keep_parts: int = 4) -> str:
    try:
        from pathlib import Path as _Path
        p = _Path(path_like)
        parts = list(p.parts)
        if len(parts) <= int(max(1, keep_parts)):
            return str(p)
        return str(_Path(*parts[-int(max(1, keep_parts)):]))
    except Exception:
        return str(path_like)


def shorten(text, *, max_len: int = 96) -> str:
    s = str(text)
    if len(s) <= max_len:
        return s
    return s[: max(0, int(max_len) - 3)] + '...'


def print_rule(*, char: str = '=', width: int = 88, level: int = 1) -> None:
    yadon_print((char or '=')[0] * int(max(8, width)), level=level)


def print_section(title: str, *, detail: str | None = None, level: int = 1, width: int = 88) -> None:
    print_rule(char='=', width=width, level=level)
    yadon_print(f"[SECTION] {title}", level=level)
    if detail:
        yadon_print(f"[NOTE] {detail}", level=level)


def print_item(label: str, value, *, level: int = 1) -> None:
    yadon_print(f"[ITEM] {label:<18}: {shorten(value)}", level=level)


def print_stat(label: str, value, *, level: int = 1) -> None:
    yadon_print(f"[STAT] {label:<18}: {shorten(value)}", level=level)


def print_step(title: str, *, detail: str | None = None, level: int = 1) -> None:
    msg = f"[STEP] {title}"
    if detail:
        msg += f" | {detail}"
    yadon_print(msg, level=level)


def print_done(title: str, *, elapsed: float | None = None, detail: str | None = None, level: int = 1) -> None:
    msg = f"[DONE] {title}"
    if elapsed is not None:
        msg += f" | elapsed={format_elapsed(elapsed)}"
    if detail:
        msg += f" | {detail}"
    yadon_print(msg, level=level)


def print_skip(title: str, *, detail: str | None = None, level: int = 1) -> None:
    msg = f"[SKIP] {title}"
    if detail:
        msg += f" | {detail}"
    yadon_print(msg, level=level)


def print_warn(title: str, *, detail: str | None = None, level: int = 2) -> None:
    msg = f"[WARN] {title}"
    if detail:
        msg += f" | {detail}"
    yadon_print(msg, level=level)
