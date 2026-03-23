"""User data directory helpers.

As of v0.6.6 YadonPy is **MolDB-first**:
  - The only persistent, user-level cache is the *molecule database* (MolDB)
    storing geometry + charges.

The older "basic_top" subsystem (pre-baked .itp/.gro/.top templates) and the
user-level copy of force-field resources have been removed to keep the project
lean and to avoid stale caches.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
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


def _normalize_existing_path(path_like: Optional[Path | str]) -> Optional[Path]:
    if not path_like:
        return None
    try:
        path = Path(path_like).expanduser().resolve()
    except Exception:
        return None
    return path if path.exists() else None


def _candidate_bundle_paths(*, cwd: Optional[Path | str] = None, argv0: Optional[Path | str] = None,
                            module_file: Optional[Path | str] = None) -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []

    def _append(root_like: Optional[Path | str]) -> None:
        root = _normalize_existing_path(root_like)
        if root is None:
            return
        roots = [root]
        if root.is_file():
            roots = [root.parent]
        for base in roots:
            for cur in [base, *base.parents[:4]]:
                cand = (cur / "yd_moldb.tar").resolve()
                if cand in seen:
                    continue
                seen.add(cand)
                candidates.append(cand)

    _append(cwd or Path.cwd())
    _append(argv0 or Path(sys.argv[0]).resolve() if sys.argv else None)
    _append(module_file or __file__)

    env_archive = _normalize_existing_path(os.environ.get("YADONPY_MOLDB_ARCHIVE"))
    if env_archive is not None:
        return [env_archive, *[p for p in candidates if p != env_archive]]
    return candidates


def find_bundle_archive(*, cwd: Optional[Path | str] = None, argv0: Optional[Path | str] = None,
                        module_file: Optional[Path | str] = None) -> Optional[Path]:
    for cand in _candidate_bundle_paths(cwd=cwd, argv0=argv0, module_file=module_file):
        if cand.exists() and cand.is_file():
            return cand
    return None


def _bundle_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_bundle_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_bundle_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _moldb_relpath_from_tar_member(member_name: str) -> Optional[Path]:
    try:
        parts = [p for p in Path(member_name).parts if p not in ("", ".", "..")]
    except Exception:
        return None
    for idx, part in enumerate(parts):
        if part == "moldb" and idx + 1 < len(parts):
            rel = Path(*parts[idx + 1:])
            return rel if rel.parts else None
    return None


def _bundle_record_keys(archive: Path) -> list[str]:
    keys: set[str] = set()
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            rel = _moldb_relpath_from_tar_member(member.name)
            if rel is None or len(rel.parts) < 2:
                continue
            if rel.parts[0] != "objects":
                continue
            keys.add(str(rel.parts[1]))
    return sorted(keys)


def _remove_managed_bundle_records(layout: DataLayout, keys: list[str]) -> None:
    for key in keys:
        try:
            target = (layout.moldb_dir / "objects" / str(key)).resolve()
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        except Exception:
            continue


def import_bundle_archive(layout: DataLayout, archive: Path) -> dict[str, Any]:
    archive_p = archive.expanduser().resolve()
    digest = _bundle_sha256(archive_p)
    state = _load_bundle_state(layout.bundle_state)
    prev_digest = str(state.get("archive_sha256", "")).strip()
    prev_keys = [str(x) for x in (state.get("managed_keys") or [])]

    managed_keys = _bundle_record_keys(archive_p)
    if prev_digest and prev_digest != digest and prev_keys:
        _remove_managed_bundle_records(layout, prev_keys)

    imported_files = 0
    with tarfile.open(archive_p, "r:*") as tf:
        for member in tf.getmembers():
            rel = _moldb_relpath_from_tar_member(member.name)
            if rel is None or not rel.parts or rel.parts[0] != "objects":
                continue
            dest = layout.moldb_dir / rel
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            imported_files += 1

    _save_bundle_state(
        layout.bundle_state,
        {
            "archive_name": archive_p.name,
            "archive_path": str(archive_p),
            "archive_sha256": digest,
            "managed_keys": managed_keys,
        },
    )
    return {
        "archive": archive_p,
        "archive_sha256": digest,
        "managed_keys": managed_keys,
        "imported_files": imported_files,
    }


def ensure_initialized() -> DataLayout:
    """Ensure the data root and MolDB directories exist (idempotent)."""
    layout = DataLayout(get_data_root())
    layout.root.mkdir(parents=True, exist_ok=True)
    (layout.moldb_dir / "objects").mkdir(parents=True, exist_ok=True)
    archive = find_bundle_archive()
    if archive is not None:
        try:
            import_bundle_archive(layout, archive)
        except Exception:
            pass
    # Marker is informational only (no migration/copying behavior).
    try:
        if not layout.marker.exists():
            layout.marker.write_text("ok\n", encoding="utf-8")
    except Exception:
        pass
    return layout
