"""Molecule naming helpers (consistent artifact naming)."""

from __future__ import annotations
import inspect
import json
import hashlib
import os
from pathlib import Path
import re
from typing import Any

from ..runtime import resolve_restart


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
    "result", "results", "res", "ok", "ret", "out", "tmp", "value", "data", "record",
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


def _coerce_work_dir_path(candidate) -> Path | None:
    try:
        if candidate is None:
            return None
        if hasattr(candidate, "path_obj"):
            return Path(candidate.path_obj).expanduser().resolve()
        return Path(os.fspath(candidate)).expanduser().resolve()
    except Exception:
        return None


def infer_work_dir(*, depth: int = 1, max_depth: int = 12) -> Path | None:
    """Best-effort lookup of a caller-visible ``work_dir`` style variable."""
    try:
        frame = inspect.currentframe()
        for _ in range(max(0, int(depth))):
            if frame is None or frame.f_back is None:
                break
            frame = frame.f_back
        if frame is None:
            return None

        preferred_keys = ("work_dir", "wd", "workspace_dir")
        for _ in range(max(1, int(max_depth))):
            if frame is None:
                break
            scopes = (getattr(frame, "f_locals", {}), getattr(frame, "f_globals", {}))
            for key in preferred_keys:
                for scope in scopes:
                    try:
                        if key in scope:
                            path = _coerce_work_dir_path(scope.get(key))
                            if path is not None:
                                return path
                    except Exception:
                        continue
            frame = frame.f_back
    except Exception:
        return None
    return None


def suggest_name_from_work_dir(work_dir) -> str | None:
    """Suggest a stable molecule name from a work-dir basename."""
    path = _coerce_work_dir_path(work_dir)
    if path is None:
        return None
    stem = str(path.name).strip()
    if not stem:
        return None
    for suffix in ("_rw", "_term", "_gmx", "_mol2", "_build_cell", "_build", "_prep", "_system"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"[^A-Za-z0-9._+\-]+", "_", stem).strip("_")
    if not stem or is_bad_default_name(stem):
        return None
    return stem


_ASSIGNED_STATE_FILENAME = "assigned_state.json"
_ATOM_DOUBLE_PROPS = (
    "AtomicCharge",
    "AtomicCharge_raw",
    "RESP",
    "RESP_raw",
    "_GasteigerCharge",
    "ff_sigma",
    "ff_epsilon",
    "ff_mass",
)
_BOND_DOUBLE_PROPS = (
    "ff_k",
    "ff_r0",
)


def _assignment_paths(root: Path, stem: str) -> tuple[Path, Path, Path, Path, Path]:
    mol2_path = root / "00_molecules" / f"{stem}.mol2"
    gmx_dir = root / f"90_{stem}_gmx"
    return (
        mol2_path,
        gmx_dir / f"{stem}.gro",
        gmx_dir / f"{stem}.itp",
        gmx_dir / f"{stem}.top",
        gmx_dir / _ASSIGNED_STATE_FILENAME,
    )


def _string_props(holder) -> dict[str, str]:
    props: dict[str, str] = {}
    try:
        for key in holder.GetPropNames(includePrivate=True, includeComputed=False):
            try:
                props[str(key)] = str(holder.GetProp(key))
            except Exception:
                continue
    except Exception:
        pass
    return props


def _atom_state(atom) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "atomic_num": int(atom.GetAtomicNum()),
        "formal_charge": int(atom.GetFormalCharge()),
        "string_props": _string_props(atom),
        "double_props": {},
    }
    for key in _ATOM_DOUBLE_PROPS:
        try:
            if atom.HasProp(key):
                entry["double_props"][key] = float(atom.GetDoubleProp(key))
        except Exception:
            continue
    return entry


def _bond_state(bond) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "begin": int(bond.GetBeginAtomIdx()),
        "end": int(bond.GetEndAtomIdx()),
        "order": float(bond.GetBondTypeAsDouble()),
        "string_props": _string_props(bond),
        "double_props": {},
    }
    for key in _BOND_DOUBLE_PROPS:
        try:
            if bond.HasProp(key):
                entry["double_props"][key] = float(bond.GetDoubleProp(key))
        except Exception:
            continue
    return entry


def _serialize_assigned_state(obj, *, artifact_dir: Path) -> dict[str, Any]:
    bonds = [_bond_state(bond) for bond in obj.GetBonds()]
    atoms = [_atom_state(atom) for atom in obj.GetAtoms()]
    return {
        "schema_version": 1,
        "ff_name": str(getattr(obj, "GetProp", lambda *_: "")("ff_name") if hasattr(obj, "HasProp") and obj.HasProp("ff_name") else ""),
        "artifact_dir": str(artifact_dir),
        "mol_props": _string_props(obj),
        "atom_count": int(obj.GetNumAtoms()),
        "atom_numbers": [int(atom.GetAtomicNum()) for atom in obj.GetAtoms()],
        "bond_pairs": sorted((min(b["begin"], b["end"]), max(b["begin"], b["end"])) for b in bonds),
        "atoms": atoms,
        "bonds": bonds,
    }


def _apply_assigned_state(obj, state: dict[str, Any], *, artifact_dir: Path) -> bool:
    try:
        if int(state.get("atom_count", -1)) != int(obj.GetNumAtoms()):
            return False
        atom_numbers = list(state.get("atom_numbers") or [])
        if atom_numbers and atom_numbers != [int(atom.GetAtomicNum()) for atom in obj.GetAtoms()]:
            return False
        state_pairs = sorted(
            (min(int(p[0]), int(p[1])), max(int(p[0]), int(p[1])))
            for p in list(state.get("bond_pairs") or [])
            if isinstance(p, (list, tuple)) and len(p) >= 2
        )
        current_pairs = sorted(
            (min(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())), max(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())))
            for b in obj.GetBonds()
        )
        if state_pairs != current_pairs:
            return False
    except Exception:
        return False

    for key, value in dict(state.get("mol_props") or {}).items():
        try:
            obj.SetProp(str(key), str(value))
        except Exception:
            continue
    try:
        obj.SetProp("_yadonpy_artifact_dir", str(artifact_dir))
    except Exception:
        pass

    atoms_state = list(state.get("atoms") or [])
    if len(atoms_state) != int(obj.GetNumAtoms()):
        return False
    for atom, entry in zip(obj.GetAtoms(), atoms_state):
        for key, value in dict(entry.get("string_props") or {}).items():
            try:
                atom.SetProp(str(key), str(value))
            except Exception:
                continue
        for key, value in dict(entry.get("double_props") or {}).items():
            try:
                atom.SetDoubleProp(str(key), float(value))
            except Exception:
                continue

    bond_lookup = {
        (min(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())), max(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx()))): b
        for b in obj.GetBonds()
    }
    for entry in list(state.get("bonds") or []):
        key = (
            min(int(entry.get("begin", -1)), int(entry.get("end", -1))),
            max(int(entry.get("begin", -1)), int(entry.get("end", -1))),
        )
        bond = bond_lookup.get(key)
        if bond is None:
            continue
        for prop_key, value in dict(entry.get("string_props") or {}).items():
            try:
                bond.SetProp(str(prop_key), str(value))
            except Exception:
                continue
        for prop_key, value in dict(entry.get("double_props") or {}).items():
            try:
                bond.SetDoubleProp(str(prop_key), float(value))
            except Exception:
                continue
    return True


def try_restore_assigned_mol(obj, *, depth: int = 1, work_dir=None, ff_name: str | None = None) -> bool:
    """Best-effort restart shortcut for ``ff_assign``.

    When restart is enabled and auto-exported per-molecule artifacts already
    exist under the current ``work_dir``, reuse the recorded assigned state
    instead of recomputing the force-field assignment.
    """
    if not resolve_restart(None):
        return False
    try:
        root = _coerce_work_dir_path(work_dir)
        if root is None:
            root = infer_work_dir(depth=depth + 1, max_depth=12)
        if root is None:
            return False
        current_name = get_name(obj, default=None)
        if current_name is None or is_bad_default_name(current_name):
            current_name = suggest_name_from_work_dir(root)
        stem = ensure_name(obj, name=current_name, depth=depth + 1, prefer_var=(current_name is None))
        mol2_path, gro_path, itp_path, top_path, state_path = _assignment_paths(root, stem)
        if not (mol2_path.exists() and gro_path.exists() and itp_path.exists() and top_path.exists() and state_path.exists()):
            return False
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state_ff_name = str((state.get("mol_props") or {}).get("ff_name") or "").strip().lower()
        want_ff_name = str(ff_name or "").strip().lower()
        if want_ff_name and state_ff_name and want_ff_name != state_ff_name:
            return False
        return _apply_assigned_state(obj, state, artifact_dir=top_path.parent)
    except Exception:
        return False


def auto_export_assigned_mol(obj, *, depth: int = 1, work_dir=None) -> Path | None:
    """Best-effort auto export after successful ``ff_assign``.

    If a caller-visible ``work_dir`` exists, write:
    - ``work_dir/00_molecules/<name>.mol2``
    - ``work_dir/90_<name>_gmx/*``
    """
    try:
        root = _coerce_work_dir_path(work_dir)
        if root is None:
            root = infer_work_dir(depth=depth + 1, max_depth=12)
        if root is None:
            return None

        current_name = get_name(obj, default=None)
        if current_name is None or is_bad_default_name(current_name):
            current_name = suggest_name_from_work_dir(root)
        stem = ensure_name(obj, name=current_name, depth=depth + 1, prefer_var=(current_name is None))

        from ..io.mol2 import write_mol2
        from ..io.gmx import write_gmx

        mol2_dir = root / "00_molecules"
        gmx_dir = root / f"90_{stem}_gmx"
        mol2_path, gro_path, itp_path, top_path, state_path = _assignment_paths(root, stem)
        if resolve_restart(None) and mol2_path.exists() and gro_path.exists() and itp_path.exists() and top_path.exists() and state_path.exists():
            return root
        write_mol2(mol=obj, out_dir=mol2_dir, name=stem, mol_name=stem)
        write_gmx(mol=obj, out_dir=gmx_dir, name=stem, mol_name=stem)
        state_path.write_text(
            json.dumps(_serialize_assigned_state(obj, artifact_dir=gmx_dir), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return root
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
