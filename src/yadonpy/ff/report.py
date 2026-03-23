from __future__ import annotations

"""Formatted force-field assignment reports."""

from collections import OrderedDict

from ..core import utils
from ..core.console import ascii_table, banner


def _fmt_float(value, digits: int = 6) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _atom_desc(atom, ff_obj=None) -> str:
    for prop in ("ff_desc", "ff_type_desc", "desc"):
        try:
            if atom.HasProp(prop):
                val = atom.GetProp(prop)
                if val:
                    return str(val)
        except Exception:
            pass

    ff_type = None
    try:
        if atom.HasProp("ff_type"):
            ff_type = atom.GetProp("ff_type")
    except Exception:
        ff_type = None

    if ff_type and ff_obj is not None:
        try:
            param = getattr(getattr(ff_obj, "param", None), "pt", {}).get(ff_type)
            if param is not None:
                desc = getattr(param, "desc", None)
                if desc:
                    return str(desc)
        except Exception:
            pass

    return "-"



def render_ff_assignment_report(mol, ff_obj=None) -> str:
    name = None
    try:
        name = utils.get_name(mol, default=None)
    except Exception:
        pass
    if not name:
        try:
            if mol.HasProp("_Name"):
                name = mol.GetProp("_Name")
        except Exception:
            pass
    name = name or "molecule"

    ff_name = "-"
    try:
        if mol.HasProp("ff_name"):
            ff_name = mol.GetProp("ff_name")
        elif ff_obj is not None:
            ff_name = getattr(ff_obj, "name", "-")
    except Exception:
        pass

    atom_rows = []
    unique = OrderedDict()
    for atom in mol.GetAtoms():
        ff_type = atom.GetProp("ff_type") if atom.HasProp("ff_type") else "-"
        ff_btype = atom.GetProp("ff_btype") if atom.HasProp("ff_btype") else "-"
        charge = atom.GetDoubleProp("AtomicCharge") if atom.HasProp("AtomicCharge") else None
        sigma = atom.GetDoubleProp("ff_sigma") if atom.HasProp("ff_sigma") else None
        epsilon = atom.GetDoubleProp("ff_epsilon") if atom.HasProp("ff_epsilon") else None
        desc = _atom_desc(atom, ff_obj=ff_obj)
        atom_rows.append([
            atom.GetIdx(),
            atom.GetSymbol(),
            ff_type,
            ff_btype,
            _fmt_float(charge, 5),
            _fmt_float(sigma),
            _fmt_float(epsilon),
            desc,
        ])
        unique.setdefault(ff_type, (_fmt_float(sigma), _fmt_float(epsilon), desc))

    lines = [banner(f'FF ASSIGN | molecule={name} | ff={ff_name} | atoms={mol.GetNumAtoms()}', char='=')]
    lines.append(
        ascii_table(
            ["idx", "elem", "ff_type", "ff_btype", "charge", "sigma", "epsilon", "description"],
            atom_rows,
        )
    )
    uniq_rows = [[ff_type, sigma, epsilon, desc] for ff_type, (sigma, epsilon, desc) in unique.items()]
    lines.append("Unique nonbonded types:")
    lines.append(ascii_table(["ff_type", "sigma", "epsilon", "description"], uniq_rows))
    return "\n".join(lines)


def print_ff_assignment_report(mol, ff_obj=None) -> None:
    print(render_ff_assignment_report(mol, ff_obj=ff_obj))
