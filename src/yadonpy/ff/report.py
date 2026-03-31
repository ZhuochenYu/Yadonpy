from __future__ import annotations

"""Formatted force-field assignment reports."""

from collections import Counter, OrderedDict

from ..core import utils
from ..core.console import ascii_table, banner
from ..core.polyelectrolyte import detect_charged_groups


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


def _total_formal_charge(mol) -> int:
    total = 0
    for atom in mol.GetAtoms():
        try:
            total += int(atom.GetFormalCharge())
        except Exception:
            continue
    return int(total)


def _total_assigned_charge(mol) -> float | None:
    charges = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("AtomicCharge"):
            return None
        try:
            charges.append(float(atom.GetDoubleProp("AtomicCharge")))
        except Exception:
            return None
    return float(sum(charges))


def _formula_for_indices(mol, atom_indices) -> str:
    counts = Counter()
    for idx in atom_indices:
        try:
            atom = mol.GetAtomWithIdx(int(idx))
        except Exception:
            continue
        counts[str(atom.GetSymbol())] += 1
    if not counts:
        return "-"

    ordered_symbols = []
    if "C" in counts:
        ordered_symbols.append("C")
    if "H" in counts:
        ordered_symbols.append("H")
    ordered_symbols.extend(sorted(sym for sym in counts if sym not in {"C", "H"}))

    parts = []
    for sym in ordered_symbols:
        n = int(counts[sym])
        parts.append(sym if n == 1 else f"{sym}{n}")
    return "".join(parts)


def _charged_group_summary(mol) -> tuple[dict, list[list[str]]]:
    summary = detect_charged_groups(mol, detection="auto")
    total_formal = int(summary.get("molecule_formal_charge", _total_formal_charge(mol)))
    total_assigned = _total_assigned_charge(mol)

    if not bool(summary.get("is_polymer")):
        return (
            {
                "total_formal_charge": total_formal,
                "total_assigned_charge": total_assigned,
                "has_side_groups": False,
            },
            [],
        )

    grouped = OrderedDict()
    for grp in summary.get("groups", []) or []:
        indices = [int(i) for i in grp.get("atom_indices", [])]
        label = str(grp.get("label", grp.get("group_id", "group")) or "group")
        composition = _formula_for_indices(mol, indices)
        formal_charge = int(grp.get("formal_charge", 0))
        assigned_charge = None
        if total_assigned is not None:
            assigned_charge = 0.0
            for idx in indices:
                atom = mol.GetAtomWithIdx(idx)
                assigned_charge += float(atom.GetDoubleProp("AtomicCharge"))
        bucket_key = (label, composition, formal_charge)
        bucket = grouped.setdefault(
            bucket_key,
            {
                "label": label,
                "composition": composition,
                "count": 0,
                "formal_charge": formal_charge,
                "assigned_total": 0.0,
                "assigned_values": [],
            },
        )
        bucket["count"] += 1
        if assigned_charge is not None:
            bucket["assigned_total"] += float(assigned_charge)
            bucket["assigned_values"].append(float(assigned_charge))

    rows: list[list[str]] = []
    for item in grouped.values():
        assigned_values = item["assigned_values"]
        assigned_avg = None
        delta_avg = None
        assigned_total = None
        if assigned_values:
            assigned_avg = item["assigned_total"] / float(item["count"])
            delta_avg = assigned_avg - float(item["formal_charge"])
            assigned_total = item["assigned_total"]
        rows.append(
            [
                str(item["label"]),
                str(item["composition"]),
                str(item["count"]),
                str(item["formal_charge"]),
                _fmt_float(assigned_avg, 5),
                _fmt_float(delta_avg, 5),
                _fmt_float(assigned_total, 5),
            ]
        )

    return (
        {
            "total_formal_charge": total_formal,
            "total_assigned_charge": total_assigned,
            "has_side_groups": bool(rows),
        },
        rows,
    )


def _render_charge_check_section(mol) -> str:
    meta, group_rows = _charged_group_summary(mol)
    lines = ["Charge check:"]
    total_assigned = meta.get("total_assigned_charge")
    if total_assigned is None:
        lines.append("  total_assigned_charge: not-set")
    else:
        lines.append(f"  total_assigned_charge: {_fmt_float(total_assigned, 5)}")
    lines.append(f"  total_formal_charge  : {int(meta.get('total_formal_charge', 0))}")

    if group_rows:
        lines.append("Charged side groups:")
        lines.append(
            ascii_table(
                ["label", "composition", "count", "formal/group", "assigned/group(avg)", "delta/group(avg)", "assigned/total"],
                group_rows,
            )
        )
    return "\n".join(lines)



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
    lines.append(_render_charge_check_section(mol))
    return "\n".join(lines)


def print_ff_assignment_report(mol, ff_obj=None) -> None:
    print(render_ff_assignment_report(mol, ff_obj=ff_obj))
