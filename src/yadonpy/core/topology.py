"""Topology attribute containers and helpers."""

from __future__ import annotations
from copy import deepcopy

def copy_topology_attributes(src_mol, dst_mol) -> None:
    """Copy RadonPy-style topology attributes stored as Python attributes.

    RDKit's ``Chem.Mol(src_mol)`` copies RDKit-owned fields/properties but **drops
    Python-level attributes** attached to the molecule. YadonPy uses Python attrs
    such as ``mol.angles`` and ``mol.dihedrals`` to carry bonded parameters.

    If these attributes are lost before writing GROMACS artifacts, the resulting
    `.itp` will miss `[ angles ]`/`[ dihedrals ]` sections, leading to unphysical
    simulations.

    This helper deep-copies the known topology containers when present.
    """

    for attr in ("angles", "dihedrals", "impropers", "pairs", "constraints"):
        if hasattr(src_mol, attr):
            try:
                setattr(dst_mol, attr, deepcopy(getattr(src_mol, attr)))
            except Exception:
                # Best effort: skip if not deepcopy-able
                try:
                    setattr(dst_mol, attr, getattr(src_mol, attr))
                except Exception:
                    pass

    # Also mirror common style props if present (these are RDKit props already,
    # but copying doesn't hurt and keeps behavior consistent).
    for prop in ("angle_style", "dihedral_style", "improper_style", "pair_style"):
        try:
            if hasattr(src_mol, "HasProp") and src_mol.HasProp(prop):
                dst_mol.SetProp(prop, src_mol.GetProp(prop))
        except Exception:
            pass


class Angle():
    """
        utils.Angle() object
    """
    def __init__(self, a, b, c, ff):
        self.a = a
        self.b = b
        self.c = c
        self.ff = ff
    
    def to_dict(self):
        dic = {
            'a': int(self.a),
            'b': int(self.b),
            'c': int(self.c),
            'ff': self.ff.to_dict()
        }
        return dic


class Dihedral():
    """
        utils.Dihedral() object
    """
    def __init__(self, a, b, c, d, ff):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.ff = ff

    def to_dict(self):
        dic = {
            'a': int(self.a),
            'b': int(self.b),
            'c': int(self.c),
            'd': int(self.d),
            'ff': self.ff.to_dict()
        }
        return dic


class Improper():
    """
        utils.Improper() object
    """
    def __init__(self, a, b, c, d, ff):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.ff = ff

    def to_dict(self):
        dic = {
            'a': int(self.a),
            'b': int(self.b),
            'c': int(self.c),
            'd': int(self.d),
            'ff': self.ff.to_dict()
        }
        return dic


class CMAP():
    """
        utils.CMAP() object
    """
    def __init__(self, a, b, c, d, e, ff):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.ff = ff

    def to_dict(self):
        dic = {
            'a': int(self.a),
            'b': int(self.b),
            'c': int(self.c),
            'd': int(self.d),
            'e': int(self.e),
            'ff': self.ff.to_dict()
        }
        return dic


class Cell():
    def __init__(self, xhi, xlo, yhi, ylo, zhi, zlo):
        self.xhi = xhi
        self.xlo = xlo
        self.yhi = yhi
        self.ylo = ylo
        self.zhi = zhi
        self.zlo = zlo
        self.dx = xhi-xlo
        self.dy = yhi-ylo
        self.dz = zhi-zlo
        self.volume = self.dx * self.dy * self.dz

    def to_dict(self):
        dic = {
            'xhi': float(self.xhi),
            'xlo': float(self.xlo),
            'yhi': float(self.yhi),
            'ylo': float(self.ylo),
            'zhi': float(self.zhi),
            'zlo': float(self.zlo),
        }
        return dic
