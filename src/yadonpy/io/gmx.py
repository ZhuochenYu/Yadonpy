"""Convenience helpers for standalone GROMACS molecule export.

This module provides a thin, user-facing wrapper around the lower-level
single-molecule topology writer. It is intended for users who only want to
parameterize a molecule and export ``.itp`` / ``.top`` / ``.gro`` artifacts,
without running the full workflow.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Tuple


def _safe_gmx_name(name: str) -> str:
    s = str(name or "").strip()
    s = s.replace('/', '_').replace('\\', '_')
    s = re.sub(r"[^A-Za-z0-9._+\-]+", "_", s)
    s = s.strip("_")
    return s or "MOL"


def write_gmx(
    *,
    mol,
    out_dir: Path,
    name: str | None = None,
    mol_name: str | None = None,
) -> Tuple[Path, Path, Path]:
    """Write standalone ``.gro`` / ``.itp`` / ``.top`` files for one molecule.

    Typical usage::

        from yadonpy.io.gmx import write_gmx
        write_gmx(mol=solvent_B, out_dir=work_dir / "90_solvent_B_gmx")

    Notes:
      - ``mol`` is expected to already have force-field parameters assigned
        (for example via ``ff.ff_assign(...)``).
      - The wrapper infers a stable molecule name from the caller's variable
        name when possible, so filenames match user scripts naturally.

    Returns:
        ``(gro_path, itp_path, top_path)``
    """
    from ..core import utils
    from .artifacts import _ensure_bonded_terms_for_export
    from .gromacs_molecule import write_gromacs_single_molecule_topology

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = utils.ensure_name(mol, name=name, depth=2, prefer_var=True)
    if mol_name is None:
        mol_name = stem
    mol_name = _safe_gmx_name(mol_name)

    ff_name = None
    try:
        if hasattr(mol, 'HasProp') and mol.HasProp('ff_name'):
            ff_name = str(mol.GetProp('ff_name')).strip()
    except Exception:
        ff_name = None
    if ff_name:
        try:
            _ensure_bonded_terms_for_export(mol, ff_name)
        except Exception:
            pass

    return write_gromacs_single_molecule_topology(mol, out_dir, mol_name=mol_name)


__all__ = ['write_gmx']
