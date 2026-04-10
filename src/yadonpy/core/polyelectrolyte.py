from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from rdkit import Chem


_CHARGE_GROUPS_PROP = "_yadonpy_charge_groups_json"
_RESP_CONSTRAINTS_PROP = "_yadonpy_resp_constraints_json"
_POLYELECTROLYTE_PROP = "_yadonpy_polyelectrolyte_summary_json"


@dataclass(frozen=True)
class ChargedGroup:
    group_id: str
    label: str
    atom_indices: tuple[int, ...]
    formal_charge: int
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "label": self.label,
            "atom_indices": list(self.atom_indices),
            "formal_charge": int(self.formal_charge),
            "source": self.source,
        }


def _stable_smarts(smarts: str):
    try:
        return Chem.MolFromSmarts(smarts)
    except Exception:
        return None


_TEMPLATES: tuple[tuple[str, int, str, Any], ...] = (
    ("carboxylate", -1, "[C:1](=[O:2])[O-:3]", _stable_smarts("[C:1](=[O:2])[O-:3]")),
    ("sulfonate", -1, "[S:1](=[O:2])(=[O:3])[O-:4]", _stable_smarts("[S:1](=[O:2])(=[O:3])[O-:4]")),
    ("phosphate", -1, "[P:1](=[O:2])([O-:3])([O:4])[O:5]", _stable_smarts("[P:1](=[O:2])([O-:3])([O:4])[O:5]")),
    ("phosphonate", -1, "[P:1]([O-:2])([O:3])([O:4])[C:5]", _stable_smarts("[P:1]([O-:2])([O:3])([O:4])[C:5]")),
    ("quaternary_ammonium", +1, "[N+:1]([C:2])([C:3])([C:4])[C:5]", _stable_smarts("[N+:1]([C:2])([C:3])([C:4])[C:5]")),
    ("imidazolium", +1, "[n+:1]1[c,n][c,n][c,n]1", _stable_smarts("[n+:1]1[c,n][c,n][c,n]1")),
    ("pyridinium", +1, "[n+:1]1ccccc1", _stable_smarts("[n+:1]1ccccc1")),
    ("cmc_carboxylate", -1, "[CH2:1][O:2][C:3](=[O:4])[O-:5]", _stable_smarts("[CH2:1][O:2][C:3](=[O:4])[O-:5]")),
)


def _get_smiles_hint(mol) -> str:
    for key in ("_yadonpy_input_smiles", "_yadonpy_smiles"):
        try:
            if mol.HasProp(key):
                return str(mol.GetProp(key))
        except Exception:
            pass
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def _is_polymer_like(mol) -> bool:
    try:
        if mol.HasProp("num_units") and int(mol.GetIntProp("num_units")) > 1:
            return True
    except Exception:
        pass
    return "*" in _get_smiles_hint(mol)


def _formal_charge_sum(mol, indices: Iterable[int]) -> int:
    total = 0
    for idx in indices:
        try:
            total += int(mol.GetAtomWithIdx(int(idx)).GetFormalCharge())
        except Exception:
            continue
    return int(total)


def _expand_local_group(mol, seed: set[int]) -> set[int]:
    group = set(int(i) for i in seed)
    frontier = list(group)
    allowed = {"O", "N", "S", "P", "C", "Si"}
    while frontier:
        idx = frontier.pop()
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            j = int(nb.GetIdx())
            if j in group:
                continue
            bond = mol.GetBondBetweenAtoms(idx, j)
            add = False
            if int(nb.GetFormalCharge()) != 0:
                add = True
            elif nb.GetSymbol() in allowed and atom.GetSymbol() in allowed:
                if bond is not None and (bond.GetBondTypeAsDouble() >= 1.5 or bond.GetIsAromatic()):
                    add = True
                elif atom.GetSymbol() in {"N", "P", "S"}:
                    add = True
                elif atom.GetSymbol() in {"C", "Si"} and nb.GetSymbol() in {"O", "N", "S"}:
                    add = True
            if add:
                group.add(j)
                frontier.append(j)
    return group


def _graph_detect_groups(mol) -> list[ChargedGroup]:
    charged = [a.GetIdx() for a in mol.GetAtoms() if int(a.GetFormalCharge()) != 0]
    groups: list[ChargedGroup] = []
    seen: set[int] = set()
    for idx in charged:
        if idx in seen:
            continue
        atoms = _expand_local_group(mol, {int(idx)})
        seen.update(atoms)
        q = _formal_charge_sum(mol, atoms)
        if q == 0:
            # Zwitterionic local neighborhoods can still be valid grouped charge regions.
            if not any(int(mol.GetAtomWithIdx(i).GetFormalCharge()) != 0 for i in atoms):
                continue
        groups.append(
            ChargedGroup(
                group_id=f"group_{len(groups)+1}",
                label=f"graph_group_{len(groups)+1}",
                atom_indices=tuple(sorted(atoms)),
                formal_charge=int(q),
                source="graph",
            )
        )
    return groups


def _template_detect_groups(mol) -> list[ChargedGroup]:
    groups: list[ChargedGroup] = []
    used: set[int] = set()
    for label, expected_q, _smarts, patt in _TEMPLATES:
        if patt is None:
            continue
        try:
            matches = mol.GetSubstructMatches(patt, uniquify=True)
        except Exception:
            matches = ()
        for match in matches:
            atoms = tuple(sorted(int(i) for i in match))
            if used.intersection(atoms):
                continue
            q = _formal_charge_sum(mol, atoms)
            if q == 0:
                q = int(expected_q)
            if q != 0 and abs(q) != abs(expected_q):
                continue
            groups.append(
                ChargedGroup(
                    group_id=f"group_{len(groups)+1}",
                    label=label,
                    atom_indices=atoms,
                    formal_charge=int(q),
                    source="template",
                )
            )
            used.update(atoms)
    return groups


def _canonical_equivalence_groups(mol, atom_indices: Iterable[int]) -> list[list[int]]:
    selected = set(int(i) for i in atom_indices)
    if not selected:
        return []
    try:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:
        return []
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in selected:
        atom = mol.GetAtomWithIdx(idx)
        grouped[(int(ranks[idx]), int(atom.GetAtomicNum()))].append(idx)
    return [sorted(v) for v in grouped.values() if len(v) > 1]


def detect_charged_groups(mol, *, detection: str = "auto") -> dict[str, Any]:
    groups: list[ChargedGroup] = []
    fallback = None
    requested = str(detection or "auto").strip().lower()

    if requested in {"auto", "template"}:
        groups = _template_detect_groups(mol)

    if requested in {"auto", "graph"} and not groups:
        groups = _graph_detect_groups(mol)

    if requested not in {"auto", "template", "graph"}:
        fallback = f"unsupported_detection:{requested}"

    charged_atoms = sorted({i for grp in groups for i in grp.atom_indices})
    neutral_atoms = [i for i in range(mol.GetNumAtoms()) if i not in set(charged_atoms)]
    equivalence = _canonical_equivalence_groups(mol, neutral_atoms)

    detected = bool(groups)
    if _is_polymer_like(mol) and not detected and any(int(a.GetFormalCharge()) != 0 for a in mol.GetAtoms()):
        fallback = fallback or "whole_molecule_scale"

    summary = {
        "is_polymer": bool(_is_polymer_like(mol)),
        "is_polyelectrolyte": bool(detected and _is_polymer_like(mol)),
        "detection": requested,
        "fallback": fallback,
        "groups": [grp.to_dict() for grp in groups],
        "neutral_remainder": list(neutral_atoms),
        "equivalence_groups": [list(g) for g in equivalence],
        "molecule_formal_charge": int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms())),
    }
    return summary


def annotate_polyelectrolyte_metadata(mol, *, detection: str = "auto") -> dict[str, Any]:
    summary = detect_charged_groups(mol, detection=detection)
    localized_groups = uses_localized_charge_groups(summary)
    constraints = {
        "mode": "grouped" if localized_groups else "whole_molecule_scale",
        "charged_group_constraints": [
            {
                "group_id": grp["group_id"],
                "atom_indices": grp["atom_indices"],
                "target_charge": grp["formal_charge"],
                "source": grp["source"],
            }
            for grp in summary["groups"]
        ],
        "neutral_remainder_charge": int(summary["molecule_formal_charge"] - sum(int(grp["formal_charge"]) for grp in summary["groups"])),
        "neutral_remainder_indices": list(summary["neutral_remainder"]),
        "equivalence_groups": [list(g) for g in summary["equivalence_groups"]],
        "fallback": summary["fallback"],
    }
    if summary["groups"] and not localized_groups and constraints["fallback"] is None:
        constraints["fallback"] = "whole_molecule_scale"
        summary["fallback"] = "whole_molecule_scale"
    try:
        mol.SetProp(_CHARGE_GROUPS_PROP, json.dumps(summary["groups"], ensure_ascii=False))
        mol.SetProp(_RESP_CONSTRAINTS_PROP, json.dumps(constraints, ensure_ascii=False))
        mol.SetProp(_POLYELECTROLYTE_PROP, json.dumps(summary, ensure_ascii=False))
    except Exception:
        pass
    return {"summary": summary, "constraints": constraints}


def get_charge_groups(mol) -> list[dict[str, Any]]:
    try:
        if mol.HasProp(_CHARGE_GROUPS_PROP):
            raw = json.loads(mol.GetProp(_CHARGE_GROUPS_PROP))
            if isinstance(raw, list):
                return raw
    except Exception:
        pass
    return annotate_polyelectrolyte_metadata(mol)["summary"]["groups"]


def get_resp_constraints(mol) -> dict[str, Any]:
    try:
        if mol.HasProp(_RESP_CONSTRAINTS_PROP):
            raw = json.loads(mol.GetProp(_RESP_CONSTRAINTS_PROP))
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass
    return annotate_polyelectrolyte_metadata(mol)["constraints"]


def get_polyelectrolyte_summary(mol) -> dict[str, Any]:
    try:
        if mol.HasProp(_POLYELECTROLYTE_PROP):
            raw = json.loads(mol.GetProp(_POLYELECTROLYTE_PROP))
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass
    return annotate_polyelectrolyte_metadata(mol)["summary"]


def uses_localized_charge_groups(summary_or_mol) -> bool:
    """Return True when grouped charge semantics should be applied.

    The key distinction is between:
      - localized charged functional groups (e.g. carboxylates), where RESP
        constraints / simulation-level charge scaling should target the group only
      - compact whole-ion anions (e.g. PF6-, TFSI-), where the net charge is
        delocalized over the entire ion and grouped scaling would distort RESP
        charges by over-correcting only the graph-detected core.

    Current rule:
      - any polymer-like charged-group summary -> grouped semantics
      - any template-detected charged group -> grouped semantics
      - graph-only groups on ordinary small molecules -> whole-molecule semantics
    """
    if summary_or_mol is None:
        return False
    if isinstance(summary_or_mol, dict):
        summary = summary_or_mol
    else:
        summary = get_polyelectrolyte_summary(summary_or_mol)

    if not isinstance(summary, dict):
        return False
    groups = list(summary.get("groups") or [])
    if not groups:
        return False
    if bool(summary.get("is_polyelectrolyte")):
        return True
    return any(str(grp.get("source") or "").strip().lower() == "template" for grp in groups)


def build_residue_map(mol, *, mol_name: str | None = None) -> dict[str, Any]:
    residues: dict[tuple[int, str], list[int]] = defaultdict(list)
    residue_atoms: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        idx = int(atom.GetIdx())
        info = atom.GetPDBResidueInfo()
        if info is not None:
            resnr = int(info.GetResidueNumber())
            resname = str(info.GetResidueName()).strip() or (mol_name or "MOL")[:5]
            atomname = str(info.GetName()).strip() or f"{atom.GetSymbol()}{idx+1}"
        else:
            resnr = 1
            resname = (mol_name or "MOL")[:5]
            atomname = f"{atom.GetSymbol()}{idx+1}"
        residues[(resnr, resname)].append(idx)
        residue_atoms.append(
            {
                "atom_index": idx,
                "residue_number": resnr,
                "residue_name": resname,
                "atom_name": atomname,
            }
        )
    return {
        "residues": [
            {
                "residue_number": int(resnr),
                "residue_name": str(resname),
                "atom_indices": list(indices),
            }
            for (resnr, resname), indices in sorted(residues.items(), key=lambda x: (x[0][0], x[0][1]))
        ],
        "atoms": residue_atoms,
    }


def scale_charged_groups_inplace(
    mol,
    *,
    scale: float,
    charge_prop: str = "AtomicCharge",
    groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    groups = list(groups if groups is not None else get_charge_groups(mol))
    report = {"scale": float(scale), "groups": [], "fallback": None, "changed_atom_indices": []}
    if abs(float(scale) - 1.0) < 1.0e-12:
        return report
    if not groups:
        report["fallback"] = "whole_molecule_scale"
        return report

    changed: set[int] = set()
    for grp in groups:
        indices = [int(i) for i in grp.get("atom_indices", [])]
        if not indices:
            continue
        qs: list[float] = []
        for idx in indices:
            atom = mol.GetAtomWithIdx(idx)
            if atom.HasProp(charge_prop):
                qs.append(float(atom.GetDoubleProp(charge_prop)))
            elif atom.HasProp("RESP"):
                qs.append(float(atom.GetDoubleProp("RESP")))
            else:
                qs.append(0.0)
        orig_total = float(sum(qs))
        target_total = float(grp.get("formal_charge", 0.0)) * float(scale)
        if abs(orig_total) > 1.0e-12:
            factor = target_total / orig_total
            new_qs = [q * factor for q in qs]
        else:
            delta = target_total / float(len(indices))
            new_qs = [delta for _ in qs]
        for idx, q in zip(indices, new_qs):
            mol.GetAtomWithIdx(idx).SetDoubleProp(charge_prop, float(q))
            changed.add(idx)
        report["groups"].append(
            {
                "group_id": grp.get("group_id"),
                "atom_indices": indices,
                "formal_charge": int(grp.get("formal_charge", 0)),
                "original_total_charge": float(orig_total),
                "scaled_total_charge": float(sum(new_qs)),
                "target_total_charge": float(target_total),
            }
        )
    report["changed_atom_indices"] = sorted(changed)
    return report
