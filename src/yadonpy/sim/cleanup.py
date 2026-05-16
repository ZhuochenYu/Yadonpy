"""Cleanup helpers for large, reproducible MD trajectory artifacts.

These utilities are intentionally conservative: they remove trajectory streams
that can be regenerated from a completed workflow, while leaving topology,
coordinate snapshots, energy files, analysis CSV/JSON/SVG, and restart metadata
in place.  Use them at the very end of a script, after analysis has finished.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_TRAJECTORY_PATTERNS: tuple[str, ...] = ("*.xtc", "*.trr", "*.trj", "*.tng")


@dataclass(frozen=True)
class CleanupResult:
    """Summary returned by :func:`clean_md_trajectory_files`."""

    root: str
    enabled: bool
    dry_run: bool
    patterns: list[str]
    removed_files: list[str]
    removed_bytes: int
    kept_files: list[str]
    summary_path: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable cleanup summary."""

        return asdict(self)


def _iter_candidate_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in patterns:
        for path in root.rglob(str(pattern)):
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if resolved in seen or not path.is_file():
                continue
            # Never crawl or clean VCS/plugin internals if a user accidentally
            # points the helper at the repository root.
            if any(part in {".git", ".hg", ".svn", "__pycache__"} for part in path.parts):
                continue
            seen.add(resolved)
            out.append(path)
    out.sort()
    return out


def clean_md_trajectory_files(
    work_dir: str | Path,
    *,
    enabled: bool = True,
    dry_run: bool = False,
    patterns: Iterable[str] = DEFAULT_TRAJECTORY_PATTERNS,
    write_summary: bool = True,
    summary_name: str = "trajectory_cleanup_summary.json",
) -> CleanupResult:
    """Remove large trajectory files under a workflow directory.

    Parameters
    ----------
    work_dir:
        Workflow directory to clean recursively.
    enabled:
        When ``False``, perform no deletion and return an empty summary.  This
        makes it easy to put the call at the end of example scripts behind a
        ``clean = False`` flag.
    dry_run:
        Report files that would be removed without deleting them.
    patterns:
        Glob patterns to remove.  The default targets trajectory streams only:
        ``*.xtc``, ``*.trr``, ``*.trj``, and ``*.tng``.
    write_summary:
        Write a small JSON audit file to ``work_dir``.  Summary files are kept
        even when trajectory streams are removed.
    summary_name:
        Name of the JSON audit file.
    """

    root = Path(work_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Cannot clean missing work_dir: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Cleanup target is not a directory: {root}")

    pattern_list = [str(p) for p in patterns]
    candidates = _iter_candidate_files(root, pattern_list) if enabled else []
    removed: list[str] = []
    kept: list[str] = []
    removed_bytes = 0

    for path in candidates:
        rel = str(path.relative_to(root))
        size = int(path.stat().st_size) if path.exists() else 0
        if dry_run:
            kept.append(rel)
            removed_bytes += size
            continue
        try:
            path.unlink()
            removed.append(rel)
            removed_bytes += size
        except Exception:
            kept.append(rel)

    summary_path = root / summary_name if write_summary else None
    result = CleanupResult(
        root=str(root),
        enabled=bool(enabled),
        dry_run=bool(dry_run),
        patterns=pattern_list,
        removed_files=removed,
        removed_bytes=int(removed_bytes),
        kept_files=kept,
        summary_path=str(summary_path) if summary_path is not None else None,
    )
    if summary_path is not None:
        summary_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


__all__ = ["CleanupResult", "DEFAULT_TRAJECTORY_PATTERNS", "clean_md_trajectory_files"]
