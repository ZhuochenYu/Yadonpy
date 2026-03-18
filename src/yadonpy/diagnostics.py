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
from pathlib import Path
import shutil
import sys
from typing import Any, Optional

from .core.data_dir import get_data_root, DataLayout


@dataclass(frozen=True)
class DepStatus:
    name: str
    installed: bool
    version: Optional[str] = None
    hint: Optional[str] = None


def _try_import_version(mod: str, attr: str = "__version__") -> Optional[str]:
    try:
        m = import_module(mod)
        return getattr(m, attr, None)
    except Exception:
        return None


def check_python_module(mod: str, *, import_name: Optional[str] = None, hint: Optional[str] = None) -> DepStatus:
    """Check if a Python module is importable.

    Args:
        mod: The module name for importlib.find_spec.
        import_name: If different, the module name used to fetch __version__.
        hint: Install hint.
    """
    installed = find_spec(mod) is not None
    ver = _try_import_version(import_name or mod) if installed else None
    return DepStatus(name=mod, installed=installed, version=ver, hint=hint)


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
    """Require Psi4 + RESP stack for QM-derived charges."""
    if find_spec("psi4") is None:
        raise ImportError(
            "Psi4 is required for QM-derived charges (RESP/ESP/Mulliken/Lowdin).\n"
            "Recommended install (conda):\n"
            "  conda install -c psi4 psi4\n"
            "If you only want a quick test, switch to charge_method='gasteiger'\n"
            "or charge_method='zero'.\n"
        )
    # 'resp' is commonly installed together with Psi4 in YadonPy workflows.
    if find_spec("resp") is None:
        raise ImportError(
            "Python package 'resp' is required for RESP charge fitting.\n"
            "Try (conda):\n"
            "  conda install -c psi4 resp\n"
        )


def doctor(*, print_report: bool = True) -> dict[str, Any]:
    """Run a small set of sanity checks and optionally print a report."""
    layout = DataLayout(get_data_root())

    checks = {
        "python": sys.version.split()[0],
        "data_root": str(layout.root),
        "initialized": layout.marker.exists(),
        "executables": {
            "gmx": shutil.which("gmx"),
            "gmx_mpi": shutil.which("gmx_mpi"),
        },
        "gromacs_forcefields": {
            "gaff": str((layout.gmx_forcefields_dir / "gaff" / "forcefield.itp")) if (layout.gmx_forcefields_dir / "gaff" / "forcefield.itp").exists() else None,
            "gaff2": str((layout.gmx_forcefields_dir / "gaff2" / "forcefield.itp")) if (layout.gmx_forcefields_dir / "gaff2" / "forcefield.itp").exists() else None,
            "gaff2_mod": str((layout.gmx_forcefields_dir / "gaff2_mod" / "forcefield.itp")) if (layout.gmx_forcefields_dir / "gaff2_mod" / "forcefield.itp").exists() else None,
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
            "resp": check_python_module(
                "resp",
                hint="conda install -c psi4 resp",
            ).__dict__,
        },
    }

    if print_report:
        print("[yadonpy] doctor report", flush=True)
        print(f"  python: {checks['python']}", flush=True)
        print(f"  data_root: {checks['data_root']}", flush=True)
        print(f"  initialized: {checks['initialized']}", flush=True)
        print("  gromacs forcefields:", flush=True)
        for k, v in checks["gromacs_forcefields"].items():
            print(f"    {k}: {v if v else 'NOT GENERATED'}", flush=True)
        print("  executables:", flush=True)
        for k, v in checks["executables"].items():
            print(f"    {k}: {v if v else 'NOT FOUND'}", flush=True)
        print("  python modules:", flush=True)
        for k, d in checks["modules"].items():
            status = "OK" if d["installed"] else "MISSING"
            ver = d.get("version")
            extra = f" (v{ver})" if ver else ""
            print(f"    {k}: {status}{extra}", flush=True)
            if (not d["installed"]) and d.get("hint"):
                print(f"      hint: {d['hint']}", flush=True)

    return checks
