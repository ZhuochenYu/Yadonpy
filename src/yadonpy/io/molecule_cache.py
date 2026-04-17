from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None  # type: ignore

from .artifacts import (
    _molecule_compatibility_context,
    _prefers_order_sensitive_artifact_cache,
    write_molecule_artifacts,
)
from ..core.naming import is_bad_default_name


def _mol_bonded_signature(mol) -> str:
    """Return a cache-relevant bonded signature for this molecule."""
    try:
        if mol is not None and hasattr(mol, 'HasProp'):
            for key in ('_yadonpy_bonded_signature', '_yadonpy_bonded_requested', '_yadonpy_bonded_method'):
                if mol.HasProp(key):
                    v = str(mol.GetProp(key)).strip().lower()
                    if v:
                        return v
    except Exception:
        pass
    return 'plain'



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
    bonded_sig = _mol_bonded_signature(mol)
    if _prefers_order_sensitive_artifact_cache(mol):
        compat = _molecule_compatibility_context(mol)
        atom_sig = str(compat.get("atom_order_signature") or "")
        group_sig = str(compat.get("charge_group_signature") or "")
        residue_sig = str(compat.get("residue_signature") or "")
        if atom_sig and group_sig:
            key = f"{ff_name}|bonded|{bonded_sig}|ordered|{atom_sig}|groups|{group_sig}|residues|{residue_sig}"
            return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    smiles = _mol_smiles_hint(mol)
    if smiles:
        key = f"{ff_name}|bonded|{bonded_sig}|smiles|{smiles}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    if Chem is not None and mol is not None:
        try:
            m0 = Chem.RemoveHs(mol)
            smi = Chem.MolToSmiles(m0, canonical=True)
            if smi:
                key = f"{ff_name}|bonded|{bonded_sig}|rdkit|{smi}"
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
    key = f"{ff_name}|bonded|{bonded_sig}|fallback|{nat}|{bonds}|{ffts}"
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


def _mol_charge_abs_and_sig(mol) -> tuple[float, str]:
    """Compute a compact charge fingerprint from atom properties.

    We use AtomicCharge if present, else RESP, else _GasteigerCharge.
    """
    try:
        from ..core import chem_utils as core_utils

        _, charges = core_utils.select_best_charge_property(mol)
        qs = [round(float(q), 6) for q in charges]
    except Exception:
        qs = []

    abs_sum = float(sum(abs(x) for x in qs))
    payload = ",".join(f"{x:.6f}" for x in qs).encode('utf-8')
    sig = hashlib.sha1(payload).hexdigest()[:16]
    return abs_sum, sig


def _itp_charge_abs_sum(itp_path: Path) -> float:
    """Parse [ atoms ] charges from an ITP and return sum(abs(q))."""
    try:
        txt = itp_path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return 0.0

    in_atoms = False
    abs_sum = 0.0
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith(';'):
            continue
        if line.startswith('[') and line.endswith(']'):
            sec = line.strip('[]').strip().lower()
            in_atoms = (sec == 'atoms')
            continue
        if not in_atoms:
            continue
        body = raw.split(';', 1)[0].strip()
        cols = body.split()
        if len(cols) < 7:
            continue
        try:
            q = float(cols[6])
            abs_sum += abs(q)
        except Exception:
            continue
    return float(abs_sum)


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

    # Validate cached artifacts (especially atomic charges). Old caches produced
    # during a failed charge assignment can contain all-zero charges and will be
    # silently reused unless we invalidate them.
    valid_cache = bool(have_itp and have_gro and have_top)
    if valid_cache:
        meta_path = out_dir / 'meta.json'
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
            except Exception:
                meta = {}

        # Atom count mismatch is always a regeneration trigger.
        want_bonded = _mol_bonded_signature(mol)
        have_bonded = str(meta.get('bonded_signature', meta.get('_yadonpy_bonded_signature', 'plain'))).strip().lower() or 'plain'
        if want_bonded != have_bonded:
            valid_cache = False
        try:
            nat = int(mol.GetNumAtoms())
        except Exception:
            nat = None
        try:
            meta_nat = int(meta.get('n_atoms')) if meta.get('n_atoms') is not None else None
        except Exception:
            meta_nat = None
        if nat is not None and meta_nat is not None and nat != meta_nat:
            valid_cache = False

        compat = _molecule_compatibility_context(mol, mol_name=mol_name)
        if _prefers_order_sensitive_artifact_cache(mol, mol_name=mol_name):
            for key in ("atom_order_signature", "charge_group_signature"):
                current_sig = str(compat.get(key) or "").strip()
                cached_sig = str(meta.get(key) or "").strip()
                if not current_sig or not cached_sig or current_sig != cached_sig:
                    valid_cache = False
                    break
            if valid_cache:
                current_residue_sig = str(compat.get("residue_signature") or "").strip()
                cached_residue_sig = str(meta.get("residue_signature") or "").strip()
                if current_residue_sig and (not cached_residue_sig or current_residue_sig != cached_residue_sig):
                    valid_cache = False

        # Charge mismatch / all-zero charge detection.
        mol_abs, mol_sig = _mol_charge_abs_and_sig(mol)
        cache_sig = str(meta.get('charge_signature', '')).strip()
        try:
            cache_abs = float(meta.get('charge_abs_sum')) if meta.get('charge_abs_sum') is not None else None
        except Exception:
            cache_abs = None

        # Only enforce charge validation if the current molecule carries charges.
        # (If the user did not assign charges, we do not force regeneration.)
        if mol_abs > 1e-3:
            # If cached meta has a signature, it must match.
            if cache_sig and cache_sig != mol_sig:
                valid_cache = False
            # If cached meta says charges are ~0, it's invalid.
            elif cache_abs is not None and cache_abs < 1e-6:
                valid_cache = False
            # Legacy caches: no signature stored. Parse the ITP as a last resort.
            elif (not cache_sig) and (cache_abs is None):
                itps = sorted(out_dir.glob('*.itp'))
                if itps:
                    itp_abs = _itp_charge_abs_sum(itps[0])
                    if itp_abs < 1e-6:
                        valid_cache = False

    # Regenerate if needed
    if valid_cache is False:
        try:
            for p in list(out_dir.iterdir()):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass
        have_itp = have_gro = have_top = False

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
