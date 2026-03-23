"""User data directory helpers.

As of v0.6.6 YadonPy is **MolDB-first**:
  - The only persistent, user-level cache is the *molecule database* (MolDB)
    storing geometry + charges.

The older "basic_top" subsystem (pre-baked .itp/.gro/.top templates) and the
user-level copy of force-field resources have been removed to keep the project
lean and to avoid stale caches.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
