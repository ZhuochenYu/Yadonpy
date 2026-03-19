"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def get_data_root() -> Path:
    """Return yadonpy data root directory.

    Priority:
      1) $YADONPY_DATA_DIR
      2) ~/.local/share/yadonpy
    """
    env = os.environ.get("YADONPY_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".local" / "share" / "yadonpy").resolve()


@dataclass(frozen=True)
class DataLayout:
    root: Path

    @property
    def ff_dir(self) -> Path:
        return self.root / "ff"

    @property
    def ff_dat_dir(self) -> Path:
        return self.ff_dir / "ff_dat"

    @property
    def library_json(self) -> Path:
        return self.ff_dir / "library.json"

    @property
    def gmx_forcefields_dir(self) -> Path:
        """Directory for exported GROMACS-style forcefield folders.

        Example layout:
          $YADONPY_DATA_DIR/ff/gmx_forcefields/gaff/forcefield.itp
        """
        return self.ff_dir / "gmx_forcefields"

    @property
    def basic_top_dir(self) -> Path:
        return self.root / "basic_top"

    @property
    def marker(self) -> Path:
        return self.root / ".initialized"


def _package_ff_dat_dir() -> Path:
    # ff_dat lives inside the installed package.
    return Path(__file__).resolve().parents[1] / "ff" / "ff_dat"


def _package_resource_dir() -> Path:
    # resources live inside the installed package.
    return Path(__file__).resolve().parents[1] / "resources"


def ensure_initialized(force: bool = False) -> DataLayout:
    """Create data root and copy built-in FF data on first use."""
    layout = DataLayout(get_data_root())
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.ff_dir.mkdir(parents=True, exist_ok=True)
    layout.gmx_forcefields_dir.mkdir(parents=True, exist_ok=True)
    layout.basic_top_dir.mkdir(parents=True, exist_ok=True)

    if force or (not layout.marker.exists()):
        # Copy ff_dat json files
        pkg_ff_dat = _package_ff_dat_dir()
        layout.ff_dat_dir.mkdir(parents=True, exist_ok=True)
        if pkg_ff_dat.exists():
            for p in pkg_ff_dat.glob("*.json"):
                shutil.copy2(p, layout.ff_dat_dir / p.name)

        # Initialize library.json if missing
        # Copy built-in basic_top library (precomputed .gro/.itp/.top)
        pkg_res = _package_resource_dir()
        pkg_basic_top = pkg_res / "basic_top"
        if pkg_basic_top.exists():
            # copytree with dirs_exist_ok to allow updates
            shutil.copytree(pkg_basic_top, layout.basic_top_dir, dirs_exist_ok=True)

        # Initialize library.json from packaged resource if available
        pkg_library_json = pkg_res / "ff" / "library.json"
        if (not layout.library_json.exists()) or force:
            if pkg_library_json.exists():
                shutil.copy2(pkg_library_json, layout.library_json)
            else:
                lib = {
                    "schema_version": 1,
                    "force_fields": {
                        "gaff": {"basic": []},
                        "gaff2": {"basic": []},
                        "gaff2_mod": {"basic": []},
                        "merz": {"basic": []},
                    },
                }
                layout.library_json.write_text(json.dumps(lib, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        layout.marker.write_text("ok\n", encoding="utf-8")

    return layout
