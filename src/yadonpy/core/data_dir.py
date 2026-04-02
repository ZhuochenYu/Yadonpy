"""User data directory helpers.

YadonPy is MolDB-first:
  - the only persistent, user-level cache is the molecule database (MolDB)
    storing geometry and charge variants.

The source tree ships a default MolDB folder beside ``examples/``. During
initialization YadonPy seeds that bundled directory into ``~/.yadonpy/moldb``
in a non-destructive way so editable/source installs start with a ready
reference catalog.
"""

from __future__ import annotations

import os
import json
import filecmp
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def get_data_root() -> Path:
    """Return YadonPy data root directory.

    Priority:
      1) $YADONPY_HOME
      2) $YADONPY_DATA_DIR (compat)
      3) ~/.yadonpy

    Notes:
      - We no longer migrate from legacy locations automatically.
      - We never delete existing user data.
    """
    env_home = (os.environ.get("YADONPY_HOME") or "").strip()
    env = (os.environ.get("YADONPY_DATA_DIR") or "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".yadonpy").resolve()


@dataclass(frozen=True)
class DataLayout:
    root: Path

    @property
    def moldb_dir(self) -> Path:
        return self.root / "moldb"

    @property
    def marker(self) -> Path:
        return self.root / ".initialized"

    @property
    def bundle_state(self) -> Path:
        return self.root / ".moldb_bundle_state.json"


def _normalize_search_seed(path_like: Optional[Path | str]) -> Optional[Path]:
    if not path_like:
        return None
    path = Path(path_like).expanduser()
    try:
        path = path.resolve()
    except Exception:
        path = path.absolute()
    if path.exists() and path.is_file():
        return path.parent
    return path


def _candidate_bundle_dirs(seed: Optional[Path | str]) -> list[Path]:
    base = _normalize_search_seed(seed)
    if base is None:
        return []

    candidates: list[Path] = []
    seen: set[Path] = set()

    def _push(path: Path) -> None:
        path = path.expanduser()
        try:
            path = path.resolve()
        except Exception:
            path = path.absolute()
        if path not in seen:
            seen.add(path)
            candidates.append(path)

    if base.name == "moldb":
        _push(base)
    _push(base / "moldb")
    for parent in base.parents:
        _push(parent / "moldb")
    return candidates


def _is_bundle_dir(path: Path) -> bool:
    return path.is_dir() and (path / "objects").is_dir()


def find_bundle_dir(*, cwd: Optional[Path | str] = None, argv0: Optional[Path | str] = None,
                    module_file: Optional[Path | str] = None) -> Optional[Path]:
    """Locate the bundled default MolDB directory, if present."""
    env_bundle = (os.environ.get("YADONPY_DEFAULT_MOLDB") or "").strip()
    seeds: list[Optional[Path | str]] = [
        env_bundle or None,
        cwd,
        argv0,
        module_file,
        Path(os.sys.prefix) / "share" / "yadonpy" / "moldb",
    ]
    for seed in seeds:
        for candidate in _candidate_bundle_dirs(seed):
            if _is_bundle_dir(candidate):
                return candidate
    return None


def find_bundle_archive(*, cwd: Optional[Path | str] = None, argv0: Optional[Path | str] = None,
                        module_file: Optional[Path | str] = None) -> Optional[Path]:
    """Backward-compatible alias for the old archive discovery helper."""
    return find_bundle_dir(cwd=cwd, argv0=argv0, module_file=module_file)


def _load_bundle_state(layout: DataLayout) -> dict[str, Any]:
    try:
        if layout.bundle_state.exists():
            payload = json.loads(layout.bundle_state.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    except Exception:
        pass
    return {}


def _write_bundle_state(layout: DataLayout, payload: dict[str, Any]) -> None:
    layout.bundle_state.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def sync_bundle_dir(layout: DataLayout, bundle_dir: Path) -> dict[str, Any]:
    """Seed or refresh the default MolDB directory into the active data root."""
    bundle_dir = bundle_dir.expanduser().resolve()
    if not _is_bundle_dir(bundle_dir):
        raise RuntimeError(f"Bundled MolDB directory is invalid: {bundle_dir}")

    previous_state = _load_bundle_state(layout)
    managed_files = set(previous_state.get("files", []))
    copied = 0
    updated = 0
    skipped = 0
    tracked_files: list[str] = []

    for src in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        rel = src.relative_to(bundle_dir).as_posix()
        dst = layout.moldb_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tracked_files.append(rel)

        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
            continue

        if rel in managed_files:
            same = False
            try:
                same = filecmp.cmp(src, dst, shallow=False)
            except Exception:
                same = False
            if not same:
                shutil.copy2(src, dst)
                updated += 1
            continue

        skipped += 1

    summary = {
        "source": str(bundle_dir),
        "files": tracked_files,
        "file_count": len(tracked_files),
        "copied": copied,
        "updated": updated,
        "skipped_existing_user_files": skipped,
    }
    _write_bundle_state(layout, summary)
    return summary


def import_bundle_archive(layout: DataLayout, archive: Path) -> dict[str, Any]:
    """Backward-compatible wrapper for callers still using the old helper name."""
    if archive.is_dir():
        return sync_bundle_dir(layout, archive)
    raise RuntimeError(
        "Bundled MolDB is now directory-based. Point YadonPy at a 'moldb/' "
        "directory instead of a .tar archive."
    )


def ensure_initialized() -> DataLayout:
    """Ensure the data root and MolDB directories exist (idempotent)."""
    layout = DataLayout(get_data_root())
    layout.root.mkdir(parents=True, exist_ok=True)
    (layout.moldb_dir / "objects").mkdir(parents=True, exist_ok=True)
    bundle_dir = find_bundle_dir(module_file=__file__)
    if bundle_dir is not None:
        try:
            sync_bundle_dir(layout, bundle_dir)
        except Exception:
            pass
    # Marker is informational only (no migration/copying behavior).
    try:
        if not layout.marker.exists():
            layout.marker.write_text("ok\n", encoding="utf-8")
    except Exception:
        pass
    return layout
