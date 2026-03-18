from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore

from .artifacts import write_molecule_artifacts
from ..core.naming import is_bad_default_name


def _default_cache_root() -> Path:
    """
    Default on-disk cache for per-molecule GROMACS artifacts.

    We deliberately keep this outside a simulation work_dir so that:
      - molecules parameterized once can be reused across workflows,
      - system export can always recover correct ITP even if RDKit mol objects
        are later rebuilt/split (which drops Python-level topology containers).
    """
    env = os.environ.get("YADONPY_MOL_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / ".yadonpy_cache" / "molecules").resolve()



def _mol_name(mol, fallback: str) -> str:
    # Prefer explicit molecule naming props used throughout yadonpy.
    for key in ("_yadonpy_name", "name", "_yadonpy_resname", "mol_name", "resname", "_Name"):
        try:
            if mol is not None and hasattr(mol, "HasProp") and mol.HasProp(key):
                n = str(mol.GetProp(key)).strip()
                if n and not is_bad_default_name(n):
                    return n
        except Exception:
            pass
    return fallback


def _mol_ff_name(mol, fallback: str = "gaff2_mod") -> str:
    for key in ("ff", "ff_name", "forcefield"):
        try:
            if mol is not None and hasattr(mol, "HasProp") and mol.HasProp(key):
                v = str(mol.GetProp(key)).strip()
                if v:
                    return v.lower()
        except Exception:
            pass
    return str(fallback).lower()


def _mol_smiles_hint(mol) -> str:
    # Prefer explicit hint if present (works for polymers where RDKit smiles may be invalid).
    for key in ("smiles", "_yadonpy_smiles"):
        try:
            if mol is not None and hasattr(mol, "HasProp") and mol.HasProp(key):
                v = str(mol.GetProp(key)).strip()
                if v:
                    return v
        except Exception:
            pass
    return ""


def _fingerprint_mol(mol, ff_name: str) -> str:
    """
    Stable identifier for cached artifacts.

    Order of preference:
      1) stored smiles hint on mol
      2) RDKit canonical SMILES (if possible)
      3) hash of (natoms, bonds, ff_type list)
    """
    smiles = _mol_smiles_hint(mol)
    if smiles:
        key = f"{ff_name}|smiles|{smiles}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    if Chem is not None and mol is not None:
        try:
            m0 = Chem.RemoveHs(mol)
            smi = Chem.MolToSmiles(m0, canonical=True)
            if smi:
                key = f"{ff_name}|rdkit|{smi}"
                return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        except Exception:
            pass

    # Structural fallback
    nat = 0
    try:
        nat = int(mol.GetNumAtoms())
    except Exception:
        nat = 0
    bonds = []
    try:
        for b in mol.GetBonds():
            bonds.append((int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx()), int(b.GetBondTypeAsDouble())))
        bonds.sort()
    except Exception:
        bonds = []
    ffts = []
    try:
        for a in mol.GetAtoms():
            if a.HasProp("ff_type"):
                ffts.append(a.GetProp("ff_type"))
            else:
                ffts.append("")
    except Exception:
        ffts = []
    key = f"{ff_name}|fallback|{nat}|{bonds}|{ffts}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _infer_charge_method(mol) -> str:
    # Best-effort: used only for meta.json bookkeeping.
    try:
        for a in mol.GetAtoms():
            if a.HasProp("RESP"):
                return "RESP"
    except Exception:
        pass
    return "UNKNOWN"


def stamp_molecule_id(mol, mol_id: str, artifact_dir: Path) -> None:
    """
    Stamp identifiers onto atoms so they survive Chem.GetMolFrags/asMols and other RDKit rebuilds.
    """
    try:
        if hasattr(mol, "SetProp"):
            mol.SetProp("_yadonpy_molid", str(mol_id))
            mol.SetProp("_yadonpy_artifact_dir", str(artifact_dir))
    except Exception:
        pass
    try:
        for a in mol.GetAtoms():
            a.SetProp("_yadonpy_molid", str(mol_id))
    except Exception:
        pass


def ensure_cached_artifacts(
    mol,
    *,
    ff_name: Optional[str] = None,
    mol_name: Optional[str] = None,
    charge_method: Optional[str] = None,
    total_charge: Optional[int] = None,
) -> Tuple[str, Path]:
    """
    Ensure per-molecule artifacts exist on disk and stamp mol/atoms with an id.

    Returns:
      (mol_id, artifact_dir)
    """
    ff = _mol_ff_name(mol, ff_name or "gaff2_mod")
    mid = _fingerprint_mol(mol, ff)
    root = _default_cache_root()
    out_dir = (root / ff / mid).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect existing
    have_itp = any(out_dir.glob("*.itp"))
    have_gro = any(out_dir.glob("*.gro"))
    have_top = any(out_dir.glob("*.top"))

    if not (have_itp and have_gro and have_top):
        name = str(mol_name or _mol_name(mol, fallback=mid))
        cm = str(charge_method or _infer_charge_method(mol))
        write_molecule_artifacts(
            mol,
            out_dir,
            smiles=_mol_smiles_hint(mol),
            ff_name=ff,
            charge_method=cm,
            total_charge=total_charge,
            mol_name=name,
            charge_scale=1.0,  # library artifacts are always unscaled
        )

    stamp_molecule_id(mol, mid, out_dir)
    return mid, out_dir
