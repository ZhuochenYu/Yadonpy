"""Molecule naming helpers (consistent artifact naming)."""

from __future__ import annotations
import inspect
import hashlib
import re


def _infer_var_name(obj, *, depth: int = 1, max_depth: int = 12) -> str | None:
    """Infer a Python variable name referencing `obj` (robust best-effort).

    Why this exists:
      - In user scripts, we often see calls like:
          ac = poly.amorphous_cell([copoly, EC, Li, PF6], ...)
        Stack-based token inference fails here because the argument is a list
        expression, not a single variable.
      - We therefore walk up the call stack and try to find *any* local/global
        name that is identical (``is``) to the given object.

    The search starts from the caller's frame (skipping `depth` frames) and
    continues upward (up to `max_depth` frames). We ignore generic placeholders
    (loop indices, internal temporaries like ``_m``) via `is_bad_default_name`.

    This helper does **not** mutate any object properties.
    """
    try:
        frame = inspect.currentframe()
        # Move to the requested starting frame.
        for _ in range(max(0, int(depth))):
            if frame is None or frame.f_back is None:
                break
            frame = frame.f_back
        if frame is None:
            return None

        candidates: list[str] = []

        # Walk up a limited number of frames to find a good variable name.
        for _ in range(max(1, int(max_depth))):
            if frame is None:
                break

            # Prefer locals, but also check globals (important for module-scope scripts).
            for scope in (getattr(frame, "f_locals", {}), getattr(frame, "f_globals", {})):
                try:
                    for k, v in scope.items():
                        if v is obj and isinstance(k, str) and k:
                            if not is_bad_default_name(k):
                                candidates.append(k)
                except Exception:
                    pass

            if candidates:
                # Choose the "best" one: prefer longer, non-underscore names.
                candidates.sort(key=lambda s: (s.startswith("_"), len(s)))
                return candidates[-1]

            frame = frame.f_back

    except Exception:
        return None
    return None



def _resolve_inferred_name(obj, *, depth: int = 1, max_depth: int = 16) -> str | None:
    """Return a sanitized variable-derived name or None."""
    inferred = _infer_var_name(obj, depth=depth, max_depth=max_depth)
    if inferred is None:
        return None
    inferred_s = str(inferred).strip()
    if not inferred_s or is_bad_default_name(inferred_s):
        return None
    return inferred_s


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
    inferred = _resolve_inferred_name(obj, depth=depth, max_depth=16)
    if inferred is None:
        return None
    try:
        if hasattr(obj, 'SetProp'):
            obj.SetProp(prop, inferred)
    except Exception:
        pass
    return inferred



# Generic placeholders commonly seen from RDKit / implicit constructors.
BAD_DEFAULTS = {
    "mol", "molecule", "rdkit", "none", "null", "",
    # common loop / generic variable names
    "m", "mon", "monomer", "sp", "species", "item", "x", "y", "z",
    "a", "b", "c", "i", "j", "k",
    # internal placeholder names frequently introduced inside helper methods
    "spec", "molspec", "handle", "obj", "resolved",
    # common placeholders used in yadonpy internals / examples
    "poly", "polymer", "copolymer", "solvent", "cation", "anion",
}
def is_bad_default_name(name: str) -> bool:
    """Return True if `name` is a generic placeholder / internal temporary.

    We use this to avoid persisting names like ``_m`` or ``i`` onto molecules,
    which would break artifact naming (e.g., producing ``_m.itp``).
    """
    try:
        s = str(name).strip()
        if not s:
            return True
        # Internal temporaries and dunder
        if s.startswith("_"):
            return True

        sl = s.lower()

        # Common placeholders
        if sl in BAD_DEFAULTS:
            return True

        # Pattern placeholders frequently appearing in polymer builders
        # (e.g., monomer_A, monomer_B, monomer_1, ...).
        if re.match(r"^monomer_[a-z0-9]+$", sl):
            return True

        return False
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
        from .molspec import as_rdkit_mol

        rdkit_obj = as_rdkit_mol(obj, strict=False)
        if rdkit_obj is None:
            rdkit_obj = obj
        try:
            formula = rdMolDescriptors.CalcMolFormula(rdkit_obj)
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
                smi = MolToSmiles(rdkit_obj, isomericSmiles=True)
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

def ensure_name(
    obj,
    *,
    name: str | None = None,
    depth: int = 1,
    prefer_var: bool = False,
) -> str:
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
        if resolved and str(resolved).strip().lower() in {"mol", "molecule", "rdkit", "polymer"}:
            resolved = None

    # Prefer the user's script variable name when requested.
    # This is useful for polymers that may accidentally inherit a monomer-like
    # name from upstream builders. The variable name in the user script is
    # usually the intended moltype/resname for exported artifacts.
    if prefer_var and (name is None):
        # Robust variable-name inference: scan up the stack (handles list expressions
        # like `[copoly, EC, Li]` where token-based inference fails).
        inferred_s = _resolve_inferred_name(obj, depth=depth + 1, max_depth=16)
        if inferred_s is not None:
            if resolved != inferred_s:
                resolved = inferred_s
            try:
                if hasattr(obj, "SetProp"):
                    obj.SetProp("_Name", inferred_s)
            except Exception:
                pass

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


def infer_var_name(obj, *, depth: int = 1) -> str | None:
    """Best-effort inference of the caller's variable name for *any* Python object.

    This is useful for lightweight handles (e.g., MolSpec) that do not support
    RDKit's SetProp/HasProp interface.

    Args:
        obj: any python object
        depth: stack depth (1 = direct caller)

    Returns:
        Inferred variable name, or None if not found.
    """
    try:
        return _resolve_inferred_name(obj, depth=depth + 1, max_depth=16)
    except Exception:
        return None


def named(obj, name: str) -> any:
    """Convenience wrapper: set a stable name on an object and return it.

    Example:
        solvent_A = utils.named(utils.mol_from_smiles(...), "solvent_A")

    This is useful when the variable-name inference is not desired.
    """
    ensure_name(obj, name=str(name), depth=1)
    return obj
