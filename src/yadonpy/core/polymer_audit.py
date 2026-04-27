"""Sanity audits for polymer charge groups and exported bonded topology.

These helpers compare RDKit-side polymer metadata with exported GROMACS
artifacts. They are used as defensive diagnostics before trusting transport
simulations, especially for charged repeat units where missing bonds, missing
nonbonded parameters, or inconsistent charge groups can silently bias MD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from rdkit import Chem

from .chem_utils import select_best_charge_property
from .polyelectrolyte import get_charge_groups


def _residue_id(atom) -> tuple[int, str]:
    info = atom.GetPDBResidueInfo()
    if info is None:
        return (1, "MOL")
    return (int(info.GetResidueNumber()), str(info.GetResidueName()).strip() or "MOL")


def _atom_label(atom) -> str:
    info = atom.GetPDBResidueInfo()
    if info is not None:
        name = str(info.GetName()).strip()
        if name:
            return name
    return f"{atom.GetSymbol()}{int(atom.GetIdx()) + 1}"


def _canonical_angle_key(i: int, j: int, k: int) -> tuple[int, int, int]:
    return (i, j, k) if i <= k else (k, j, i)


def _canonical_dihedral_key(i: int, j: int, k: int, l: int) -> tuple[int, int, int, int]:
    forward = (i, j, k, l)
    reverse = (l, k, j, i)
    return forward if forward <= reverse else reverse


def _improper_items(mol) -> list[tuple[int, int, int, int]]:
    raw = getattr(mol, "impropers", {}) or {}
    if isinstance(raw, Mapping):
        values = list(raw.values())
    else:
        values = list(raw)
    out: list[tuple[int, int, int, int]] = []
    for imp in values:
        try:
            out.append((int(imp.a), int(imp.b), int(imp.c), int(imp.d)))
        except Exception:
            continue
    return out


def _junction_bonds(mol) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        ai = int(a.GetIdx())
        bi = int(b.GetIdx())
        key = (ai, bi) if ai <= bi else (bi, ai)
        if key in seen:
            continue
        seen.add(key)
        marked_new = False
        try:
            marked_new = bond.HasProp("new_bond") and bond.GetBoolProp("new_bond")
        except Exception:
            marked_new = False
        res_a = _residue_id(a)
        res_b = _residue_id(b)
        if not marked_new and res_a == res_b:
            continue
        rows.append(
            {
                "bond_atoms": [key[0], key[1]],
                "atom_labels": [_atom_label(a), _atom_label(b)],
                "residues": [
                    {"residue_number": res_a[0], "residue_name": res_a[1]},
                    {"residue_number": res_b[0], "residue_name": res_b[1]},
                ],
                "marked_new_bond": bool(marked_new),
            }
        )
    return rows


def _local_neighborhood(mol, seed: Iterable[int], radius: int) -> list[int]:
    radius = max(int(radius), 0)
    seen = {int(idx) for idx in seed}
    frontier = set(seen)
    for _ in range(radius):
        nxt: set[int] = set()
        for idx in frontier:
            atom = mol.GetAtomWithIdx(int(idx))
            for nb in atom.GetNeighbors():
                j = int(nb.GetIdx())
                if j not in seen:
                    seen.add(j)
                    nxt.add(j)
        frontier = nxt
        if not frontier:
            break
    return sorted(seen)


def audit_charge_groups(mol) -> dict[str, Any]:
    charge_prop, charges = select_best_charge_property(mol)
    groups = list(get_charge_groups(mol) or [])
    atoms = list(mol.GetAtoms())
    total_formal_charge = int(sum(int(atom.GetFormalCharge()) for atom in atoms))
    total_selected_charge = float(sum(float(q) for q in charges)) if charges else None
    rows: list[dict[str, Any]] = []
    for grp in groups:
        atom_indices = [int(i) for i in grp.get("atom_indices", [])]
        grp_charge = None
        if charges and len(charges) == len(atoms):
            grp_charge = float(sum(float(charges[idx]) for idx in atom_indices))
        rows.append(
            {
                "group_id": grp.get("group_id"),
                "label": grp.get("label"),
                "source": grp.get("source"),
                "atom_indices": atom_indices,
                "formal_charge": int(grp.get("formal_charge", 0)),
                "selected_charge_total": grp_charge,
            }
        )
    return {
        "selected_charge_prop": charge_prop,
        "total_selected_charge": total_selected_charge,
        "total_formal_charge": total_formal_charge,
        "groups": rows,
    }


def audit_junction_bonded_terms(mol, *, radius: int = 2) -> dict[str, Any]:
    angle_keys: set[tuple[int, int, int]] = set()
    for ang in (getattr(mol, "angles", {}) or {}).values():
        try:
            angle_keys.add(_canonical_angle_key(int(ang.a), int(ang.b), int(ang.c)))
        except Exception:
            continue

    dihedral_keys: set[tuple[int, int, int, int]] = set()
    for dih in (getattr(mol, "dihedrals", {}) or {}).values():
        try:
            dihedral_keys.add(_canonical_dihedral_key(int(dih.a), int(dih.b), int(dih.c), int(dih.d)))
        except Exception:
            continue

    improper_items = _improper_items(mol)
    rows: list[dict[str, Any]] = []
    total_missing_angles = 0
    total_missing_dihedrals = 0

    for item in _junction_bonds(mol):
        a, b = (int(item["bond_atoms"][0]), int(item["bond_atoms"][1]))
        atom_a = mol.GetAtomWithIdx(a)
        atom_b = mol.GetAtomWithIdx(b)
        neigh_a = sorted(int(nb.GetIdx()) for nb in atom_a.GetNeighbors() if int(nb.GetIdx()) != b)
        neigh_b = sorted(int(nb.GetIdx()) for nb in atom_b.GetNeighbors() if int(nb.GetIdx()) != a)

        expected_angles = {
            _canonical_angle_key(n, a, b) for n in neigh_a
        } | {
            _canonical_angle_key(a, b, n) for n in neigh_b
        }
        missing_angles = sorted(key for key in expected_angles if key not in angle_keys)

        expected_dihedrals = {
            _canonical_dihedral_key(i, a, b, j)
            for i in neigh_a
            for j in neigh_b
        }
        missing_dihedrals = sorted(key for key in expected_dihedrals if key not in dihedral_keys)

        local_atoms = _local_neighborhood(mol, (a, b), radius)
        local_set = set(local_atoms)
        local_impropers = [
            list(imp)
            for imp in improper_items
            if set(int(idx) for idx in imp).intersection(local_set)
        ]

        total_missing_angles += len(missing_angles)
        total_missing_dihedrals += len(missing_dihedrals)
        rows.append(
            {
                **item,
                "local_atoms_radius": int(radius),
                "local_atom_indices": local_atoms,
                "expected_angle_count": len(expected_angles),
                "missing_angles": [list(key) for key in missing_angles],
                "expected_dihedral_count": len(expected_dihedrals),
                "missing_dihedrals": [list(key) for key in missing_dihedrals],
                "local_impropers": local_impropers,
            }
        )

    return {
        "junction_bonds": rows,
        "missing_angle_total": int(total_missing_angles),
        "missing_dihedral_total": int(total_missing_dihedrals),
    }


def audit_nonbonded_assignment(mol) -> dict[str, Any]:
    missing_ff_type: list[int] = []
    missing_sigma: list[int] = []
    missing_epsilon: list[int] = []
    for atom in mol.GetAtoms():
        idx = int(atom.GetIdx())
        if not atom.HasProp("ff_type"):
            missing_ff_type.append(idx)
        if not atom.HasProp("ff_sigma"):
            missing_sigma.append(idx)
        if not atom.HasProp("ff_epsilon"):
            missing_epsilon.append(idx)
    return {
        "missing_ff_type": missing_ff_type,
        "missing_ff_sigma": missing_sigma,
        "missing_ff_epsilon": missing_epsilon,
    }


def audit_planar_charge_groups(mol) -> dict[str, Any]:
    groups = list(get_charge_groups(mol) or [])
    impropers = _improper_items(mol)
    rows: list[dict[str, Any]] = []
    for grp in groups:
        label = str(grp.get("label") or "")
        atom_indices = [int(i) for i in grp.get("atom_indices", [])]
        if "carboxylate" not in label.lower():
            continue
        atom_set = set(atom_indices)
        hit = any(atom_set.issubset(set(imp)) for imp in impropers)
        rows.append(
            {
                "group_id": grp.get("group_id"),
                "label": label,
                "atom_indices": atom_indices,
                "has_improper_covering_group": bool(hit),
            }
        )
    return {"planar_groups": rows}


def audit_polymer_state(mol, *, label: str, radius: int = 2) -> dict[str, Any]:
    atoms = list(mol.GetAtoms())
    try:
        ff_name = mol.GetProp("ff_name") if mol.HasProp("ff_name") else None
    except Exception:
        ff_name = None
    payload = {
        "label": str(label),
        "ff_name": ff_name,
        "num_atoms": len(atoms),
        "num_bonds": int(mol.GetNumBonds()),
        "charge_audit": audit_charge_groups(mol),
        "nonbonded_audit": audit_nonbonded_assignment(mol),
        "junction_audit": audit_junction_bonded_terms(mol, radius=radius),
        "planar_group_audit": audit_planar_charge_groups(mol),
        "topology_counts": {
            "angles": len(getattr(mol, "angles", {}) or {}),
            "dihedrals": len(getattr(mol, "dihedrals", {}) or {}),
            "impropers": len(getattr(mol, "impropers", {}) or {}),
        },
    }
    payload["warnings"] = []
    if payload["nonbonded_audit"]["missing_ff_type"]:
        payload["warnings"].append("missing_ff_type")
    if payload["nonbonded_audit"]["missing_ff_sigma"] or payload["nonbonded_audit"]["missing_ff_epsilon"]:
        payload["warnings"].append("missing_lj_params")
    if payload["junction_audit"]["missing_angle_total"] > 0:
        payload["warnings"].append("missing_junction_angles")
    if payload["junction_audit"]["missing_dihedral_total"] > 0:
        payload["warnings"].append("missing_junction_dihedrals")
    if any(not row["has_improper_covering_group"] for row in payload["planar_group_audit"]["planar_groups"]):
        payload["warnings"].append("missing_planar_group_improper")
    return payload


def write_polymer_audit(report: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def compare_exported_charge_groups(*, system_dir: str | Path, moltype: str, mol) -> dict[str, Any]:
    system_dir = Path(system_dir)
    path = system_dir / "charge_groups.json"
    payload: dict[str, Any] = {"system_dir": str(system_dir), "moltype": str(moltype), "exists": path.exists()}
    if not path.exists():
        payload["match"] = False
        payload["reason"] = "missing_charge_groups_json"
        return payload
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        payload["match"] = False
        payload["reason"] = f"invalid_json:{exc}"
        return payload
    species = list(raw.get("species") or [])
    current_groups = [
        {
            "group_id": grp.get("group_id"),
            "label": grp.get("label"),
            "atom_indices": [int(i) for i in grp.get("atom_indices", [])],
            "formal_charge": int(grp.get("formal_charge", 0)),
        }
        for grp in get_charge_groups(mol)
    ]
    for entry in species:
        if str(entry.get("moltype") or "") != str(moltype):
            continue
        exported_groups = [
            {
                "group_id": grp.get("group_id"),
                "label": grp.get("label"),
                "atom_indices": [int(i) for i in grp.get("atom_indices", [])],
                "formal_charge": int(grp.get("formal_charge", 0)),
            }
            for grp in (entry.get("charge_groups") or [])
        ]
        payload["exported_groups"] = exported_groups
        payload["current_groups"] = current_groups
        payload["match"] = exported_groups == current_groups
        return payload
    payload["match"] = False
    payload["reason"] = "moltype_not_found"
    payload["current_groups"] = current_groups
    return payload


__all__ = [
    "audit_charge_groups",
    "audit_junction_bonded_terms",
    "audit_nonbonded_assignment",
    "audit_planar_charge_groups",
    "audit_polymer_state",
    "compare_exported_charge_groups",
    "write_polymer_audit",
]
