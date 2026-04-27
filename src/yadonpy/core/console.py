"""Small ASCII formatting helpers for terminal-facing workflow reports.

The functions here intentionally return plain strings instead of printing
directly. Callers can route the formatted banners/tables through YadonPy's
normal logging layer, tests can compare exact output, and examples can stay
readable without depending on a rich terminal renderer.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def ascii_rule(char: str = '=', width: int = 88) -> str:
    char = (char or '=')[0]
    return char * int(max(8, width))


def banner(title: str, subtitle: str | None = None, *, char: str = '=', width: int = 88) -> str:
    lines = [ascii_rule(char, width), str(title)]
    if subtitle:
        lines.append(str(subtitle))
    lines.append(ascii_rule(char, width))
    return '\n'.join(lines)


def ascii_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    rows = [list(r) for r in rows]
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    out = [sep]
    out.append('| ' + ' | '.join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + ' |')
    out.append(sep)
    for row in rows:
        out.append('| ' + ' | '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + ' |')
    out.append(sep)
    return '\n'.join(out)
