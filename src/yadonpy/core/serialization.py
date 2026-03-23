"""Serialization helpers (pickle + backward-compat)."""

from __future__ import annotations
import pickle

from rdkit import Chem

from . import const
from .exceptions import YadonPyError
from .logging_utils import radon_print

def picklable(mol=None):
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    return mol


def restore_picklable(mol=None):
    # Backward campatibility
    return mol


def pickle_dump(mol, path) -> None:
    Chem.SanitizeMol(mol)
    mol = picklable(mol)
    with open(str(path), mode='wb') as f:
        pickle.dump(mol, f)


def pickle_load(path, *, allow_unsafe: bool = True):
    """Load a molecule pickle.

    Security note: Python pickle is unsafe for untrusted files.
    Only load pickles you created yourself or trust.

    Args:
        path: File path.
        allow_unsafe: Keep backward-compatible behavior. If False, raise.
    """
    if not allow_unsafe:
        raise YadonPyError('Refusing to load pickle with allow_unsafe=False')
    try:
        with open(str(path), mode='rb') as f:
            mol = pickle.load(f)
    except Exception as e:
        radon_print('Cannot load pickle file %s. %s' % (path, e), level=2)
        return None

    # Backward campatibility from version 0.2 to 1.0
    if hasattr(mol, 'angles'):
        if isinstance(mol.angles, list):
            mol.angles = {'%i,%i,%i' % (ang.a, ang.b, ang.c): ang for ang in mol.angles}

    if hasattr(mol, 'dihedrals'):
        if isinstance(mol.dihedrals, list):
            mol.dihedrals = {'%i,%i,%i,%i' % (dih.a, dih.b, dih.c, dih.d): dih for dih in mol.dihedrals}

    if hasattr(mol, 'impropers'):
        if isinstance(mol.impropers, list):
            mol.impropers = {'%i,%i,%i,%i' % (imp.a, imp.b, imp.c, imp.d): imp for imp in mol.impropers}

    return mol


def deepcopy_mol(mol):
    """Deep-copy an RDKit molecule *without losing* YadonPy's Python-side attributes.

    Why this exists:
      - RDKit Mol objects support copy/deepcopy, but their pickle/copy protocol does NOT
        preserve arbitrary Python attributes attached to the Mol instance (e.g. `mol.angles`,
        `mol.dihedrals`, ...).
      - Many YadonPy workflows (e.g. Example 01) rely on `deepcopy_mol` while building
        polymers/mixtures/cells. If bonded terms are stored as Python attributes, they can be
        silently dropped, and later `.itp` export will miss [ angles ] / [ dihedrals ].

    Implementation:
      - Use `Chem.Mol(mol)` to clone the RDKit molecule (conformers + RDKit props).
      - Explicitly copy a small, curated set of Python-side attributes that YadonPy uses.

    This keeps the public API identical while making the workflow robust.
    """
    from rdkit import Chem
    from copy import deepcopy as _deepcopy

    # 1) Clone RDKit core state (atoms/bonds/conformers + RDKit props)
    copy_mol = Chem.Mol(mol)

    # 2) Preserve Python-side attributes used by YadonPy
    #    (Keep this list tight to avoid copying accidental heavy objects.)
    _ATTRS = (
        "angles",
        "dihedrals",
        "impropers",
        "cmaps",
        "cell",
    )
    for _k in _ATTRS:
        if hasattr(mol, _k):
            try:
                setattr(copy_mol, _k, _deepcopy(getattr(mol, _k)))
            except Exception:
                # Best-effort: if an attribute can't be deep-copied, keep a shallow reference.
                # This is still better than silently losing it.
                try:
                    setattr(copy_mol, _k, getattr(mol, _k))
                except Exception:
                    pass

    return copy_mol


def picklable_const():
    c = {}
    for v in dir(const):
        if v.count('__') != 2 and v != 'os':
            c[v] = getattr(const, v)
    return c


def restore_const(c):
    for k, v in c.items():
        setattr(const, k, v)
    return True
