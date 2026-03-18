"""Molecule naming helpers (consistent artifact naming)."""

from __future__ import annotations
import inspect
import hashlib
import re

def set_name_from_var(obj, *, depth: int = 1, prop: str = '_Name'):
    """Best-effort: set RDKit Mol name from the caller's Python variable name.

    This enables workflows where folder/file naming follows the variable names
    used in example scripts, e.g. `anion_A = mol_from_smiles(...)`.

    Args:
        obj: RDKit Mol (or any object with SetProp/HasProp)
        depth: stack depth to look up (1 = direct caller)
        prop: RDKit property key to set (default: _Name)

    Returns:
        str or None: the inferred variable name.
    """
    try:
        import inspect
        frame = inspect.currentframe()
        # move to caller frame
        for _ in range(max(1, int(depth))):
            if frame is None or frame.f_back is None:
                break
            frame = frame.f_back
        if frame is None:
            return None
        for k, v in frame.f_locals.items():
            if v is obj and isinstance(k, str) and k:
                try:
                    if hasattr(obj, 'SetProp'):
                        obj.SetProp(prop, k)
                except Exception:
                    pass
                return k
    except Exception:
        return None
    return None



# Generic placeholders commonly seen from RDKit / implicit constructors.
BAD_DEFAULTS = {"mol", "molecule", "rdkit", "none", "null", ""}


def is_bad_default_name(name: str) -> bool:
    """Return True if `name` is a generic placeholder (e.g., RDKit default 'mol')."""
    try:
        return str(name).strip().lower() in BAD_DEFAULTS
    except Exception:
        return True


def get_name(obj, *, default: str | None = None) -> str | None:
    """Best-effort name getter.

    We standardize on these property keys (in priority order):
      1) _yadonpy_name      (explicit, yadonpy-wide)
      2) name              (user-friendly alias)
      3) _yadonpy_resname   (GROMACS residue/moleculetype name)
      4) _Name             (RDKit conventional name; often a generic placeholder)
    """
    keys = ("_yadonpy_name", "name", "_yadonpy_resname", "_Name")
    try:
        if hasattr(obj, "HasProp") and hasattr(obj, "GetProp"):
            for k in keys:
                try:
                    if obj.HasProp(k):
                        v = str(obj.GetProp(k)).strip()
                        if not v:
                            continue
                        if is_bad_default_name(v):
                            continue
                        return v
                except Exception:
                    continue
    except Exception:
        pass
    return default


def _auto_name(obj) -> str:
    """Generate a stable, non-generic name for an unnamed molecule.

    This is used when:
      - the object has no explicit name props, and
      - variable-name inference fails (common inside loops/lists).

    We prefer a short deterministic name based on (formula + SMILES hash).
    For very large molecules (e.g., polymers), we fall back to a hash-only name.
    """
    # Default fallback
    base = "molecule"
    try:
        # RDKit imports are optional here to keep import-time light.
        from rdkit.Chem import rdMolDescriptors, MolToSmiles
        try:
            formula = rdMolDescriptors.CalcMolFormula(obj)
        except Exception:
            formula = ""
        # Prefer original input SMILES if present; otherwise canonicalize.
        smi = ""
        try:
            if hasattr(obj, "HasProp") and obj.HasProp("_yadonpy_input_smiles"):
                smi = str(obj.GetProp("_yadonpy_input_smiles")).strip()
        except Exception:
            smi = ""
        if not smi:
            try:
                smi = MolToSmiles(obj, isomericSmiles=True)
            except Exception:
                smi = ""
        h = hashlib.sha1((smi or formula or "molecule").encode("utf-8")).hexdigest()[:6]
        if formula and len(formula) <= 20:
            base = f"{formula}_{h}"
        else:
            base = f"mol_{h}"
    except Exception:
        base = "molecule"

    # filesystem-friendly
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(base)).strip("_")
    return base or "molecule"

def ensure_name(obj, *, name: str | None = None, depth: int = 1) -> str:
    """Ensure an object has a stable name.

    If `name` is provided, it is written to all supported property keys.
    Otherwise, we try to read an existing name; if missing, we infer the
    caller's Python variable name (best-effort) and persist it.

    Args:
        obj: RDKit Mol (or any object with SetProp/HasProp)
        name: explicit name override
        depth: stack depth for variable-name inference (1 = direct caller)

    Returns:
        The resolved name (never empty; falls back to "molecule").
    """
    resolved = (str(name).strip() if name is not None else None)
    if not resolved:
        resolved = get_name(obj, default=None)
        if resolved and str(resolved).strip().lower() in {"mol","molecule","rdkit","polymer"}:
            resolved = None

    if not resolved:
        # infer from Python variable name in the *caller's* frame
        inferred = set_name_from_var(obj, depth=depth + 1, prop="_Name")
        resolved = (str(inferred).strip() if inferred else None)

    if not resolved:
        resolved = _auto_name(obj)

    # Persist to all relevant keys (best-effort).
    try:
        if hasattr(obj, "SetProp"):
            for k in ("_yadonpy_name", "name", "_yadonpy_resname", "_Name"):
                try:
                    obj.SetProp(k, resolved)
                except Exception:
                    pass
    except Exception:
        pass

    return resolved


def named(obj, name: str) -> any:
    """Convenience wrapper: set a stable name on an object and return it.

    Example:
        solvent_A = utils.named(utils.mol_from_smiles(...), "solvent_A")

    This is useful when the variable-name inference is not desired.
    """
    ensure_name(obj, name=str(name), depth=1)
    return obj
