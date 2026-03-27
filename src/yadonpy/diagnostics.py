"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
import shutil
import sys
from typing import Any, Optional

from .core.data_dir import get_data_root, DataLayout, find_bundle_archive


@dataclass(frozen=True)
class DepStatus:
    name: str
    installed: bool
    import_ok: bool = False
    version: Optional[str] = None
    import_error: Optional[str] = None
    hint: Optional[str] = None


def _try_import_version(mod: str, attr: str = "__version__") -> Optional[str]:
    try:
        m = import_module(mod)
        return getattr(m, attr, None)
    except Exception:
        return None


def _try_import(mod: str) -> tuple[bool, Optional[str]]:
    try:
        import_module(mod)
        return True, None
    except Exception as e:
        return False, repr(e)


def check_python_module(mod: str, *, import_name: Optional[str] = None, hint: Optional[str] = None) -> DepStatus:
    """Check if a Python module is importable.

    Args:
        mod: The module name for importlib.find_spec.
        import_name: If different, the module name used to fetch __version__.
        hint: Install hint.
    """
    installed = find_spec(mod) is not None
    import_ok, err = _try_import(import_name or mod) if installed else (False, None)
    ver = _try_import_version(import_name or mod) if import_ok else None
    return DepStatus(name=mod, installed=installed, import_ok=import_ok, version=ver, import_error=err, hint=hint)


def require_rdkit() -> None:
    if find_spec("rdkit") is None:
        raise ImportError(
            "RDKit is required by yadonpy (SMILES parsing/typing/polymer building).\n"
            "Recommended install (conda):\n"
            "  conda install -c conda-forge rdkit\n"
        )

    # Enforce a modern RDKit for robust 3D embedding of inorganic/polyatomic ions
    # (e.g., PF6-, BF4-). Older builds (e.g., 2020.03) are known to produce
    # degenerate conformers for some hypervalent species.
    try:
        from rdkit import rdBase

        ver = str(getattr(rdBase, "rdkitVersion", ""))
        # rdkitVersion is typically "YYYY.MM.x".
        parts = [int(p) for p in ver.split(".")[:3] if p.isdigit()]
        while len(parts) < 3:
            parts.append(0)
        y, m, p = parts[:3]
        if (y, m, p) < (2025, 3, 1):
            raise ImportError(
                "RDKit >= 2025.03.1 is required by yadonpy for stable 3D embedding and ion handling.\n"
                f"Detected RDKit: {ver!r}.\n"
                "Please upgrade (recommended conda):\n"
                "  conda install -c conda-forge rdkit>=2025.03.1\n"
            )
    except ImportError:
        raise
    except Exception:
        # If we cannot parse the version string, do not hard-fail.
        pass


def require_psi4_resp() -> None:
    """Require Psi4 + PsiRESP stack for QM-derived charges."""
    if find_spec("psi4") is None:
        raise ImportError(
            "Psi4 is required for QM-derived charges (RESP/ESP/Mulliken/Lowdin).\n"
            "Recommended install (conda):\n"
            "  conda install -c psi4 psi4\n"
            "If you only want a quick test, switch to charge_method='gasteiger'\n"
            "or charge_method='zero'.\n"
        )
    if find_spec("psiresp") is None:
        raise ImportError(
            "Python package 'psiresp' is required for RESP/ESP charge fitting.\n"
            "Try (conda):\n"
            "  conda install -c conda-forge psiresp\n"
        )


def doctor(*, print_report: bool = True) -> dict[str, Any]:
    """Run a small set of sanity checks and optionally print a report."""
    layout = DataLayout(get_data_root())

    checks = {
        "python": sys.version.split()[0],
        "data_root": str(layout.root),
        "moldb_dir": str(layout.moldb_dir),
        "initialized": layout.marker.exists(),
        "bundle_archive": (str(find_bundle_archive()) if find_bundle_archive() is not None else None),
        "bundle_state": str(layout.bundle_state) if layout.bundle_state.exists() else None,
        "executables": {
            "gmx": shutil.which("gmx"),
            "gmx_mpi": shutil.which("gmx_mpi"),
        },
        "modules": {
            "rdkit": check_python_module(
                "rdkit",
                hint="conda install -c conda-forge rdkit",
            ).__dict__,
            "psi4": check_python_module(
                "psi4",
                hint="conda install -c psi4 psi4",
            ).__dict__,
            "psiresp": check_python_module(
                "psiresp",
                hint="conda install -c conda-forge psiresp",
            ).__dict__,
        },
    }

    if print_report:
        print("[yadonpy] doctor report", flush=True)
        print(f"  python: {checks['python']}", flush=True)
        print(f"  data_root: {checks['data_root']}", flush=True)
        print(f"  moldb_dir: {checks['moldb_dir']}", flush=True)
        print(f"  initialized: {checks['initialized']}", flush=True)
        print(f"  bundle_archive: {checks['bundle_archive'] if checks['bundle_archive'] else 'NOT FOUND'}", flush=True)
        print(f"  bundle_state: {checks['bundle_state'] if checks['bundle_state'] else 'NOT FOUND'}", flush=True)
        print("  executables:", flush=True)
        for k, v in checks["executables"].items():
            print(f"    {k}: {v if v else 'NOT FOUND'}", flush=True)
        print("  python modules:", flush=True)
        for k, d in checks["modules"].items():
            if not d["installed"]:
                status = "MISSING"
            else:
                status = "OK" if d.get("import_ok", False) else "BROKEN"
            ver = d.get("version")
            extra = f" (v{ver})" if ver else ""
            print(f"    {k}: {status}{extra}", flush=True)
            if (not d["installed"]) and d.get("hint"):
                print(f"      hint: {d['hint']}", flush=True)
            if d.get("installed") and (not d.get("import_ok", False)) and d.get("import_error"):
                print(f"      import error: {d['import_error']}", flush=True)

    return checks
