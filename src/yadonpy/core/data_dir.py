"""User data directory helpers.

YadonPy is MolDB-first:
  - the only persistent, user-level cache is the molecule database (MolDB)
    storing geometry and charge variants.

The package no longer ships or auto-imports a bundled MolDB archive at
initialization time. Reference MolDB species are rebuilt explicitly through the
example scripts and stored into the active user MolDB directory.
"""

from __future__ import annotations

import os
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


def find_bundle_archive(*, cwd: Optional[Path | str] = None, argv0: Optional[Path | str] = None,
                        module_file: Optional[Path | str] = None) -> Optional[Path]:
    # Bundled MolDB archive discovery was removed in v0.8.75.
    return None


def import_bundle_archive(layout: DataLayout, archive: Path) -> dict[str, Any]:
    raise RuntimeError(
        "Bundled MolDB archive import was removed in v0.8.75. "
        "Use the Example 07 reference-species rebuild script instead."
    )


def ensure_initialized() -> DataLayout:
    """Ensure the data root and MolDB directories exist (idempotent)."""
    layout = DataLayout(get_data_root())
    layout.root.mkdir(parents=True, exist_ok=True)
    (layout.moldb_dir / "objects").mkdir(parents=True, exist_ok=True)
    # Marker is informational only (no migration/copying behavior).
    try:
        if not layout.marker.exists():
            layout.marker.write_text("ok\n", encoding="utf-8")
    except Exception:
        pass
    return layout
