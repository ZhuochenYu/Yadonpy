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


def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _manifest_ready_score(payload: dict[str, Any]) -> tuple[int, int, int]:
    variants = payload.get("variants") or {}
    ready_variants = 0
    total_variants = 0
    if isinstance(variants, dict):
        total_variants = len(variants)
        for meta in variants.values():
            if isinstance(meta, dict) and bool(meta.get("ready", False)):
                ready_variants += 1
    legacy_ready = 1 if bool(payload.get("ready", False)) else 0
    return (legacy_ready, int(ready_variants), int(total_variants))


def _manifest_variants(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants = payload.get("variants") or {}
    if not isinstance(variants, dict):
        return {}
    return {str(k): v for k, v in variants.items() if isinstance(v, dict)}


def _manifest_variant_ids(payload: dict[str, Any]) -> set[str]:
    return set(_manifest_variants(payload).keys())


def _manifest_ready_variant_ids(payload: dict[str, Any]) -> set[str]:
    return {
        str(vid)
        for vid, meta in _manifest_variants(payload).items()
        if bool(meta.get("ready", False))
    }


def _manifest_bonded_variant_ids(payload: dict[str, Any]) -> set[str]:
    return {
        str(vid)
        for vid, meta in _manifest_variants(payload).items()
        if isinstance(meta.get("bonded"), dict) and bool(meta.get("bonded"))
    }


def _record_variant_delta(
    bundle_payload: dict[str, Any],
    user_payload: dict[str, Any],
) -> dict[str, list[str]]:
    bundle_variant_ids = _manifest_variant_ids(bundle_payload)
    user_variant_ids = _manifest_variant_ids(user_payload)
    bundle_ready_ids = _manifest_ready_variant_ids(bundle_payload)
    user_ready_ids = _manifest_ready_variant_ids(user_payload)
    bundle_bonded_ids = _manifest_bonded_variant_ids(bundle_payload)
    user_bonded_ids = _manifest_bonded_variant_ids(user_payload)
    return {
        "bundle_only_variant_ids": sorted(bundle_variant_ids - user_variant_ids),
        "user_only_variant_ids": sorted(user_variant_ids - bundle_variant_ids),
        "bundle_only_ready_variant_ids": sorted(bundle_ready_ids - user_ready_ids),
        "user_only_ready_variant_ids": sorted(user_ready_ids - bundle_ready_ids),
        "bundle_only_bonded_variant_ids": sorted(bundle_bonded_ids - user_bonded_ids),
        "user_only_bonded_variant_ids": sorted(user_bonded_ids - bundle_bonded_ids),
    }


def _bundle_record_is_more_complete(
    *,
    bundle_payload: dict[str, Any],
    user_payload: dict[str, Any],
) -> bool:
    if _manifest_ready_score(bundle_payload) > _manifest_ready_score(user_payload):
        return True
    delta = _record_variant_delta(bundle_payload, user_payload)
    return any(
        bool(delta[key])
        for key in (
            "bundle_only_variant_ids",
            "bundle_only_ready_variant_ids",
            "bundle_only_bonded_variant_ids",
        )
    )


def audit_bundle_sync(layout: DataLayout, bundle_dir: Path) -> dict[str, Any]:
    bundle_records: dict[str, dict[str, Any]] = {}
    user_records: dict[str, dict[str, Any]] = {}
    for src_manifest in sorted(bundle_dir.glob("objects/*/manifest.json")):
        prefix = src_manifest.relative_to(bundle_dir).parent.as_posix()
        bundle_records[prefix] = _safe_load_json(src_manifest)
    for dst_manifest in sorted(layout.moldb_dir.glob("objects/*/manifest.json")):
        prefix = dst_manifest.relative_to(layout.moldb_dir).parent.as_posix()
        user_records[prefix] = _safe_load_json(dst_manifest)

    bundle_keys = set(bundle_records.keys())
    user_keys = set(user_records.keys())
    stale_variants: dict[str, dict[str, list[str]]] = {}
    bundle_more_complete_records: list[str] = []
    for prefix in sorted(bundle_keys & user_keys):
        delta = _record_variant_delta(bundle_records[prefix], user_records[prefix])
        if any(bool(values) for values in delta.values()):
            stale_variants[prefix] = delta
        if _bundle_record_is_more_complete(
            bundle_payload=bundle_records[prefix],
            user_payload=user_records[prefix],
        ):
            bundle_more_complete_records.append(prefix)
    return {
        "missing_objects": sorted(bundle_keys - user_keys),
        "stale_variants": stale_variants,
        "bundled_more_complete_records": sorted(bundle_more_complete_records),
        "user_only_records": sorted(user_keys - bundle_keys),
    }


def _bundle_record_prefixes_to_refresh(
    *,
    layout: DataLayout,
    bundle_dir: Path,
    managed_files: set[str],
) -> set[str]:
    refresh_prefixes: set[str] = set()
    for src_manifest in sorted(bundle_dir.glob("objects/*/manifest.json")):
        rel = src_manifest.relative_to(bundle_dir).as_posix()
        if rel in managed_files:
            continue
        dst_manifest = layout.moldb_dir / rel
        if not dst_manifest.exists():
            continue
        src_payload = _safe_load_json(src_manifest)
        dst_payload = _safe_load_json(dst_manifest)
        if _bundle_record_is_more_complete(
            bundle_payload=src_payload,
            user_payload=dst_payload,
        ):
            refresh_prefixes.add(src_manifest.relative_to(bundle_dir).parent.as_posix())
    return refresh_prefixes


def sync_bundle_dir(layout: DataLayout, bundle_dir: Path) -> dict[str, Any]:
    """Seed or refresh the default MolDB directory into the active data root."""
    bundle_dir = bundle_dir.expanduser().resolve()
    if not _is_bundle_dir(bundle_dir):
        raise RuntimeError(f"Bundled MolDB directory is invalid: {bundle_dir}")

    previous_state = _load_bundle_state(layout)
    managed_files = set(previous_state.get("files", []))
    refresh_prefixes = _bundle_record_prefixes_to_refresh(
        layout=layout,
        bundle_dir=bundle_dir,
        managed_files=managed_files,
    )
    copied = 0
    updated = 0
    skipped = 0
    tracked_files: list[str] = []

    for src in sorted(p for p in bundle_dir.rglob("*") if p.is_file()):
        rel = src.relative_to(bundle_dir).as_posix()
        dst = layout.moldb_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tracked_files.append(rel)
        rel_parts = Path(rel).parts
        record_prefix = "/".join(rel_parts[:2]) if len(rel_parts) >= 2 else rel

        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
            continue

        if rel in managed_files or record_prefix in refresh_prefixes:
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
        "refreshed_stale_records": sorted(refresh_prefixes),
        "audit": audit_bundle_sync(layout, bundle_dir),
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
