"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

#  Copyright (c) 2026.
#  This file is an extension module for YadonPy to support OPLS-AA atom typing
#  using an external SMARTS rule table (derived from a moltemplate/STaGE rule table).

# ******************************************************************************
# ff.oplsaa module
# ******************************************************************************

import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

from rdkit import Chem
from rdkit import Geometry as Geom
from ..core import utils
from ..core.resources import ff_data_path
from .gaff import GAFF
from . import ff_class
from ..core import utils as core_utils
from .report import print_ff_assignment_report
from .oplsaa_reference import get_oplsaa_improper_templates, iter_source_backed_oplsaa_improper_candidates



# OPLS-AA atom-typing rules are stored as package data so the rule table can
# evolve independently from the SMARTS matching engine.


@dataclass(frozen=True)
class OplsAtomTypingRule:
    element: str
    btype: str
    opls: str
    smarts: str
    charge: float | None = None
    desc: str = ""
    priority: int = 0


@lru_cache(maxsize=1)
def _load_rule_records():
    with open(ff_data_path("ff_dat", "oplsaa_rules.json"), encoding="utf-8") as fh:
        raw_rules = json.load(fh)

    records = []
    for idx, row in enumerate(raw_rules):
        records.append(
            OplsAtomTypingRule(
                element=str(row["element"]),
                btype=str(row["btype"]),
                opls=str(row["opls"]),
                smarts=str(row["smarts"]),
                charge=None if row.get("charge", None) is None else float(row["charge"]),
                desc=str(row.get("desc", "")),
                priority=int(row.get("priority", idx)),
            )
        )
    return tuple(records)


def validate_oplsaa_rule_table(param_pt_names=None):
    rules = _load_rule_records()
    placeholder_types = sorted({rule.opls for rule in rules if rule.opls == "opls_xxx"})
    placeholder_rule_count = sum(1 for rule in rules if rule.opls == "opls_xxx")
    unknown_types = []
    if param_pt_names is not None:
        allowed = set(param_pt_names)
        unknown_types = sorted({rule.opls for rule in rules if rule.opls not in allowed and rule.opls != "opls_xxx"})
    return {
        "rule_count": len(rules),
        "placeholder_types": placeholder_types,
        "placeholder_rule_count": placeholder_rule_count,
        "unknown_types": unknown_types,
    }


# Lazy-compiled RDKit SMARTS queries (compiled on first use)
_COMPILED = None
_BTYPE_ALIASES = {
    "H": "H~",
    "H~": "H",
    "O": "O~",
    "O~": "O",
    "N": "N~",
    "N~": "N",
    "C": "C~",
    "C~": "C",
    "S": "S~",
    "S~": "S",
    "P": "P~",
    "P~": "P",
    "F": "F~",
    "F~": "F",
    "I": "I~",
    "I~": "I",
}

# A strict zero-LJ donor hydrogen is standard in some OPLS-AA distributions,
# but in GROMACS condensed-phase carbohydrate / polyol systems it can collapse
# onto nearby acceptor oxygens during minimization. We therefore materialize a
# tiny GAFF-like repulsive floor for hydroxyl donor hydrogens after typing.
_DONOR_H_LJ_FLOOR = {
    "opls_155": (0.053792, 0.019665),
    "opls_170": (0.053792, 0.019665),
    "opls_172": (0.053792, 0.019665),
    "opls_188": (0.053792, 0.019665),
}


def _param_source(param):
    source = getattr(param, "source", None)
    if source:
        return str(source)
    desc = getattr(param, "desc", None)
    if desc and "source=" in str(desc):
        try:
            return str(desc).split("source=", 1)[1].split()[0].strip()
        except Exception:
            return None
    return None

def _get_compiled_rules():
    global _COMPILED
    if _COMPILED is not None:
        return _COMPILED

    compiled = []
    for rule in sorted(_load_rule_records(), key=lambda item: item.priority):
        q = Chem.MolFromSmarts(rule.smarts)
        if q is None:
            raise ValueError(f"Invalid SMARTS in OPLS-AA rule table: {rule.smarts}")
        compiled.append((rule, q))
    _COMPILED = tuple(compiled)
    return _COMPILED


@lru_cache(maxsize=None)
def _get_rule_by_opls(opls_type: str):
    for rule in _load_rule_records():
        if str(rule.opls) == str(opls_type):
            return rule
    return None


def _iter_unique_params(mapping):
    """Yield unique force-field parameter objects from name->object mappings."""
    seen = set()
    for obj in mapping.values():
        ident = getattr(obj, 'tag', None) or id(obj)
        if ident in seen:
            continue
        seen.add(ident)
        yield obj


def _iter_btype_alias_token_sets(tokens):
    per_token = []
    for token in tokens:
        choices = [token]
        alias = _BTYPE_ALIASES.get(token)
        if alias and alias not in choices:
            choices.append(alias)
        per_token.append(tuple(choices))

    seen = set()
    for combo in product(*per_token):
        if combo in seen:
            continue
        seen.add(combo)
        yield combo


def _match_bonded_pattern(pattern_tokens, actual_tokens):
    """
    Score a bonded-parameter pattern against actual OPLS bonded tokens.

    We treat the token ``X`` as a wildcard (used by some OPLS dihedral records).
    Other tokens such as ``C*`` and ``N*`` are real OPLS bonded labels and are
    matched literally.
    """
    if len(pattern_tokens) != len(actual_tokens):
        return None

    exact = 0
    wildcards = 0
    for pat, actual in zip(pattern_tokens, actual_tokens):
        if pat == actual:
            exact += 1
            continue
        if pat == 'X':
            wildcards += 1
            continue
        return None

    return (exact, -wildcards)


class OPLSAA(GAFF):
    """
    ff.oplsaa.OPLSAA

    OPLS-AA force field assignment for RDKit Mol.
    - Nonbonded parameters are taken from ff_dat/oplsaa.json [atomtypes] (keyed by 'opls_###')
    - Bonded parameters (bond/angle/dihedral) are taken from ff_dat/oplsaa.json [bondtypes]/[angletypes]/[dihedraltypes]
      and are keyed by OPLS 'bond_type' labels (CT, CA, HC, ...).

    Atom typing is done via an external JSON SMARTS rule table.
    """

    def __init__(self, db_file=None):
        if db_file is None:
            db_file = str(ff_data_path("ff_dat", "oplsaa.json"))
        super().__init__(db_file)
        self.name = 'oplsaa'
        self.rule_table_summary = validate_oplsaa_rule_table(self.param.pt.keys())
        if self.rule_table_summary["unknown_types"]:
            raise ValueError(
                "OPLS-AA rule table references unknown particle types: "
                + ", ".join(self.rule_table_summary["unknown_types"])
            )

        # OPLS 1-4 scaling: Coulomb 0.5, LJ 0.5 (typical OPLS-AA convention)
        self.param.c_c12 = 0.0
        self.param.c_c13 = 0.0
        self.param.c_c14 = 0.5
        self.param.lj_c12 = 0.0
        self.param.lj_c13 = 0.0
        self.param.lj_c14 = 0.5

        # Styles
        self.pair_style = 'lj'
        self.bond_style = 'harmonic'
        self.angle_style = 'harmonic'
        self.dihedral_style = 'fourier'
        self.improper_style = 'cvff'
        self._logged_special_overrides = set()


    def mol(
        self,
        smiles_or_psmiles: str,
        *,
        name: str | None = None,
        basis_set: str | None = None,
        method: str | None = None,
        charge: str = "RESP",
        require_ready: bool = True,
        prefer_db: bool = True,
        polyelectrolyte_mode: bool | None = None,
        polyelectrolyte_detection: str | None = None,
        resp_profile: str | None = None,
    ):
        """Create a lightweight MolSpec handle for OPLS-AA workflows.

        Default behavior mirrors the broader YadonPy workflow:
        try to resolve a RESP-ready MolDB entry first.  If the handle is later
        resolved during ``ff_assign()`` and no RESP-ready entry is available,
        YadonPy falls back to OPLS-AA built-in type charges instead of failing.
        """
        s = str(smiles_or_psmiles).strip()
        try:
            m = Chem.MolFromSmiles(s)
        except Exception:
            m = None
        if m is not None and int(m.GetNumAtoms()) == 1:
            a = m.GetAtomWithIdx(0)
            q = int(a.GetFormalCharge())
            has_embedded_h = int(a.GetTotalNumHs()) > 0 or int(a.GetNumExplicitHs()) > 0
            if q != 0 and not has_embedded_h:
                rw = Chem.RWMol()
                atom = Chem.Atom(a.GetSymbol())
                atom.SetFormalCharge(q)
                idx = rw.AddAtom(atom)
                mol = rw.GetMol()
                conf = Chem.Conformer(mol.GetNumAtoms())
                conf.SetAtomPosition(idx, Geom.Point3D(0.0, 0.0, 0.0))
                mol.AddConformer(conf, assignId=True)
                if name:
                    try:
                        mol.SetProp('_Name', str(name))
                    except Exception:
                        pass
                return mol

        try:
            return self.mol_rdkit(
                smiles_or_psmiles,
                name=name,
                prefer_db=prefer_db,
                require_db=False,
                require_ready=require_ready,
                charge=charge,
                basis_set=basis_set,
                method=method,
                resp_profile=resp_profile,
                polyelectrolyte_mode=polyelectrolyte_mode,
                polyelectrolyte_detection=polyelectrolyte_detection,
            )
        except Exception as exc:
            charge_token = str(charge or "").strip().upper()
            if charge_token != "RESP":
                raise
            utils.radon_print(
                f"RESP-ready MolDB entry unavailable for {smiles_or_psmiles}; "
                "falling back to OPLS-AA built-in charges.",
                level=1,
            )
            mol = self.mol_rdkit(
                smiles_or_psmiles,
                name=name,
                prefer_db=prefer_db,
                require_db=False,
                require_ready=False,
                charge="opls",
                basis_set=basis_set,
                method=method,
                resp_profile=resp_profile,
                polyelectrolyte_mode=polyelectrolyte_mode,
                polyelectrolyte_detection=polyelectrolyte_detection,
            )
            try:
                mol.SetProp("_yadonpy_charge_fallback", "opls")
                mol.SetProp("_yadonpy_charge_fallback_reason", str(exc))
            except Exception:
                pass
            return mol

    @staticmethod
    def _has_complete_atomic_charges(mol) -> bool:
        try:
            atoms = list(mol.GetAtoms())
        except Exception:
            return False
        if not atoms:
            return False
        return all(a.HasProp('AtomicCharge') for a in atoms)

    @staticmethod
    def _normalize_charge_mode(charge):
        if charge is None:
            return None
        token = str(charge).strip()
        if not token:
            return None
        low = token.lower()
        if low in ('keep', 'existing', 'preserve', 'external-existing'):
            return None
        if low == 'opls':
            return 'opls'
        return token

    @staticmethod
    def _normalize_bonded_override(bonded):
        if bonded is None:
            return None
        token = str(bonded).strip().lower()
        if not token:
            return None
        if token in ("ms", "mseminario", "seminario"):
            return "mseminario"
        if token in ("drih", "drih-like", "dri"):
            return "drih"
        return token

    @staticmethod
    def _has_marked_new_bond(mol) -> bool:
        if mol is None or not hasattr(mol, "GetBonds"):
            return False
        for bond in mol.GetBonds():
            try:
                if bond.HasProp("new_bond") and bond.GetBoolProp("new_bond"):
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _new_bond_endpoints(mol) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        if mol is None or not hasattr(mol, "GetBonds"):
            return pairs
        for bond in mol.GetBonds():
            try:
                if bond.HasProp("new_bond") and bond.GetBoolProp("new_bond"):
                    pairs.append((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())))
            except Exception:
                continue
        return pairs

    @staticmethod
    def _collect_graph_neighborhood(mol, seeds, *, radius: int) -> set[int]:
        visited = {int(idx) for idx in seeds}
        frontier = set(visited)
        for _ in range(max(int(radius), 0)):
            next_frontier: set[int] = set()
            for idx in frontier:
                atom = mol.GetAtomWithIdx(int(idx))
                for nb in atom.GetNeighbors():
                    nb_idx = int(nb.GetIdx())
                    if nb_idx in visited:
                        continue
                    visited.add(nb_idx)
                    next_frontier.add(nb_idx)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    @staticmethod
    def _has_complete_existing_atom_types(mol) -> bool:
        try:
            atoms = list(mol.GetAtoms())
        except Exception:
            return False
        if not atoms:
            return False
        return all(atom.HasProp("ff_type") and atom.HasProp("ff_btype") for atom in atoms)

    @staticmethod
    def _has_existing_bonded_assignment(mol) -> bool:
        try:
            bonds = list(mol.GetBonds())
        except Exception:
            return False
        if bonds and not all(bond.HasProp("ff_type") for bond in bonds):
            return False
        if mol.GetNumAtoms() >= 3 and not isinstance(getattr(mol, "angles", None), dict):
            return False
        if mol.GetNumAtoms() >= 4 and not isinstance(getattr(mol, "dihedrals", None), dict):
            return False
        return True

    @staticmethod
    def _has_reusable_existing_assignment(mol, *, ff_name: str) -> bool:
        if mol is None or not hasattr(mol, "HasProp"):
            return False
        try:
            current_ff = str(mol.GetProp("ff_name")).strip().lower()
        except Exception:
            return False
        if current_ff != str(ff_name).strip().lower():
            return False
        return OPLSAA._has_complete_existing_atom_types(mol) and OPLSAA._has_existing_bonded_assignment(mol)

    def _assign_opls_charges_from_existing_types(self, mol) -> bool:
        for atom in mol.GetAtoms():
            try:
                ff_type = atom.GetProp("ff_type")
            except Exception:
                return False
            rule = _get_rule_by_opls(ff_type)
            if rule is None or rule.charge is None:
                continue
            atom.SetDoubleProp("AtomicCharge", float(rule.charge))
        return True

    @staticmethod
    def _has_precomputed_bonded_fragment(mol) -> bool:
        if mol is None or not hasattr(mol, "HasProp"):
            return False
        for key in (
            "_yadonpy_bonded_itp",
            "_yadonpy_bonded_json",
            "_yadonpy_mseminario_itp",
            "_yadonpy_mseminario_json",
        ):
            try:
                if mol.HasProp(key) and str(mol.GetProp(key)).strip():
                    return True
            except Exception:
                continue
        if getattr(mol, "angles", None) or getattr(mol, "dihedrals", None):
            return True
        return False

    def _assign_ptypes_subset(self, mol, atom_indices, charge=None):
        if self._has_implicit_h(mol):
            utils.radon_print(
                'OPLS-AA SMARTS typing requires explicit hydrogens (Chem.AddHs). '
                'Found implicit H on at least one heavy atom. Aborting typing.',
                level=3
            )
            return False, set()

        targets = {int(idx) for idx in atom_indices}
        if not targets:
            return True, set()

        compiled = _get_compiled_rules()
        chosen = {}
        for rule, q in compiled:
            matches = mol.GetSubstructMatches(q, uniquify=False)
            if not matches:
                continue
            for match in matches:
                if not match:
                    continue
                idx = int(match[0])
                if idx not in targets:
                    continue
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() != rule.element:
                    continue
                chosen[idx] = rule

        changed_atoms: set[int] = set()
        ok = True
        for idx in sorted(targets):
            atom = mol.GetAtomWithIdx(idx)
            rule = chosen.get(idx)
            if rule is None:
                if atom.HasProp("ff_type") and atom.HasProp("ff_btype"):
                    continue
                utils.radon_print(
                    f'OPLS-AA typing failed: atom {idx} ({atom.GetSymbol()}) did not match any SMARTS rule.',
                    level=2,
                )
                ok = False
                continue

            prev_ff_type = atom.GetProp("ff_type") if atom.HasProp("ff_type") else None
            prev_btype = atom.GetProp("ff_btype") if atom.HasProp("ff_btype") else None
            atom.SetProp("ff_btype", rule.btype)
            if rule.opls not in self.param.pt:
                utils.radon_print(f'OPLS-AA typing failed: nonbonded type {rule.opls} not found in oplsaa.json', level=3)
                return False, set()
            self.set_ptype(atom, rule.opls)
            try:
                atom.SetProp("ff_desc", str(rule.desc))
            except Exception:
                pass
            if charge == "opls":
                q = 0.0 if rule.charge is None else float(rule.charge)
                atom.SetDoubleProp("AtomicCharge", q)

            if prev_ff_type != rule.opls or prev_btype != rule.btype:
                changed_atoms.add(idx)

        return ok, changed_atoms

    @staticmethod
    def _bond_needs_local_refresh(bond, affected_atoms: set[int]) -> bool:
        try:
            if bond.HasProp("new_bond") and bond.GetBoolProp("new_bond"):
                return True
        except Exception:
            pass
        return int(bond.GetBeginAtomIdx()) in affected_atoms or int(bond.GetEndAtomIdx()) in affected_atoms

    def _refresh_local_bonded_terms(self, mol, *, affected_atoms: set[int]) -> bool:
        result = True

        for bond in mol.GetBonds():
            if not self._bond_needs_local_refresh(bond, affected_atoms):
                continue
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if not a1.HasProp("ff_btype") or not a2.HasProp("ff_btype"):
                utils.radon_print("ff_btype missing on atoms. Did you run assign_ptypes first?", level=3)
                return False
            tokens = (a1.GetProp("ff_btype"), a2.GetProp("ff_btype"))
            param = self._lookup_bond_param(tokens)
            if param is None:
                utils.radon_print(f'Cannot assign bond parameters for {tokens[0]},{tokens[1]}', level=2)
                result = False
                continue
            self.set_btype(bond, param)

        existing_angles = {}
        for key, angle in (getattr(mol, "angles", {}) or {}).items():
            if {int(angle.a), int(angle.b), int(angle.c)} & affected_atoms:
                continue
            existing_angles[key] = angle
        setattr(mol, "angles", existing_angles)

        for center in mol.GetAtoms():
            neighbors = list(center.GetNeighbors())
            if len(neighbors) < 2:
                continue
            for a_idx in range(len(neighbors)):
                for c_idx in range(a_idx + 1, len(neighbors)):
                    left = neighbors[a_idx]
                    right = neighbors[c_idx]
                    indices = {int(left.GetIdx()), int(center.GetIdx()), int(right.GetIdx())}
                    if not indices & affected_atoms:
                        continue
                    tokens = (
                        left.GetProp("ff_btype"),
                        center.GetProp("ff_btype"),
                        right.GetProp("ff_btype"),
                    )
                    param = self._lookup_angle_param(tokens)
                    if param is None:
                        utils.radon_print(
                            f'Cannot assign angle parameters for {tokens[0]},{tokens[1]},{tokens[2]}',
                            level=2,
                        )
                        result = False
                        continue
                    self.set_atype(mol, left.GetIdx(), center.GetIdx(), right.GetIdx(), param)

        existing_dihedrals = {}
        for key, dih in (getattr(mol, "dihedrals", {}) or {}).items():
            if {int(dih.a), int(dih.b), int(dih.c), int(dih.d)} & affected_atoms:
                continue
            existing_dihedrals[key] = dih
        setattr(mol, "dihedrals", existing_dihedrals)

        for bond in mol.GetBonds():
            j = bond.GetBeginAtom()
            k = bond.GetEndAtom()
            jn = [atom for atom in j.GetNeighbors() if atom.GetIdx() != k.GetIdx()]
            kn = [atom for atom in k.GetNeighbors() if atom.GetIdx() != j.GetIdx()]
            if not jn or not kn:
                continue
            for i in jn:
                for l in kn:
                    indices = {int(i.GetIdx()), int(j.GetIdx()), int(k.GetIdx()), int(l.GetIdx())}
                    if not indices & affected_atoms:
                        continue
                    tokens = (
                        i.GetProp("ff_btype"),
                        j.GetProp("ff_btype"),
                        k.GetProp("ff_btype"),
                        l.GetProp("ff_btype"),
                    )
                    param = self._lookup_dihedral_param(tokens)
                    if param is None:
                        utils.radon_print(
                            f'Cannot assign dihedral parameters for {tokens[0]},{tokens[1]},{tokens[2]},{tokens[3]}',
                            level=2,
                        )
                        result = False
                        continue
                    self.set_dtype(mol, i.GetIdx(), j.GetIdx(), k.GetIdx(), l.GetIdx(), param)

        self.assign_itypes(mol)
        return result

    @staticmethod
    def _has_angle_key(mol, a: int, b: int, c: int) -> bool:
        angles = getattr(mol, "angles", {}) or {}
        return (f"{a},{b},{c}" in angles) or (f"{c},{b},{a}" in angles)

    @staticmethod
    def _has_dihedral_key(mol, a: int, b: int, c: int, d: int) -> bool:
        dihedrals = getattr(mol, "dihedrals", {}) or {}
        return (f"{a},{b},{c},{d}" in dihedrals) or (f"{d},{c},{b},{a}" in dihedrals)

    def _fill_missing_bonded_terms(self, mol, atom_indices: set[int] | None = None) -> bool:
        result = True
        scope = None if atom_indices is None else {int(idx) for idx in atom_indices}

        if not hasattr(mol, "angles"):
            setattr(mol, "angles", {})
        if not hasattr(mol, "dihedrals"):
            setattr(mol, "dihedrals", {})

        for center in mol.GetAtoms():
            center_idx = int(center.GetIdx())
            neighbors = list(center.GetNeighbors())
            if len(neighbors) < 2:
                continue
            for left_i in range(len(neighbors)):
                for right_i in range(left_i + 1, len(neighbors)):
                    left = neighbors[left_i]
                    right = neighbors[right_i]
                    indices = {int(left.GetIdx()), center_idx, int(right.GetIdx())}
                    if scope is not None and not (indices & scope):
                        continue
                    if self._has_angle_key(mol, left.GetIdx(), center_idx, right.GetIdx()):
                        continue
                    tokens = (
                        left.GetProp("ff_btype"),
                        center.GetProp("ff_btype"),
                        right.GetProp("ff_btype"),
                    )
                    param = self._lookup_angle_param(tokens)
                    if param is None:
                        utils.radon_print(
                            f'Cannot assign angle parameters for {tokens[0]},{tokens[1]},{tokens[2]}',
                            level=2,
                        )
                        result = False
                        continue
                    self.set_atype(mol, left.GetIdx(), center_idx, right.GetIdx(), param)

        for bond in mol.GetBonds():
            j = bond.GetBeginAtom()
            k = bond.GetEndAtom()
            jn = [atom for atom in j.GetNeighbors() if atom.GetIdx() != k.GetIdx()]
            kn = [atom for atom in k.GetNeighbors() if atom.GetIdx() != j.GetIdx()]
            if not jn or not kn:
                continue
            for i in jn:
                for l in kn:
                    indices = {int(i.GetIdx()), int(j.GetIdx()), int(k.GetIdx()), int(l.GetIdx())}
                    if scope is not None and not (indices & scope):
                        continue
                    if self._has_dihedral_key(mol, i.GetIdx(), j.GetIdx(), k.GetIdx(), l.GetIdx()):
                        continue
                    tokens = (
                        i.GetProp("ff_btype"),
                        j.GetProp("ff_btype"),
                        k.GetProp("ff_btype"),
                        l.GetProp("ff_btype"),
                    )
                    param = self._lookup_dihedral_param(tokens)
                    if param is None:
                        utils.radon_print(
                            f'Cannot assign dihedral parameters for {tokens[0]},{tokens[1]},{tokens[2]},{tokens[3]}',
                            level=2,
                        )
                        result = False
                        continue
                    self.set_dtype(mol, i.GetIdx(), j.GetIdx(), k.GetIdx(), l.GetIdx(), param)

        return result

    def refresh_polymer_junction_terms(self, mol, charge=None, *, radius: int = 2):
        new_bonds = self._new_bond_endpoints(mol)
        if not new_bonds:
            return True
        if not self._has_complete_existing_atom_types(mol):
            return False

        endpoints = {idx for pair in new_bonds for idx in pair}
        workset = self._collect_graph_neighborhood(mol, endpoints, radius=radius)
        ok, changed_atoms = self._assign_ptypes_subset(mol, workset, charge=charge)
        if not ok:
            return False

        # Junction edits can invalidate inherited local bonded terms even when
        # nearby atoms keep the same OPLS type after re-typing. Refresh the
        # full local workset instead of only atoms whose type changed.
        affected_atoms = set(workset) | set(endpoints) | set(changed_atoms)
        utils.radon_print(
            f'Applying OPLS-AA local polymer junction refresh for {len(new_bonds)} new bond(s) '
            f'and {len(affected_atoms)} affected atom(s).',
            level=1,
        )
        result = self._refresh_local_bonded_terms(mol, affected_atoms=affected_atoms)
        # The inherited polymer topology may still contain sparse holes in
        # remote angle/dihedral records after repeated atom deletions and index
        # remapping. Fill only the missing bonded terms without disturbing the
        # existing ones.
        result = self._fill_missing_bonded_terms(mol) and result
        return result

    def _resolve_spec(self, mol):
        """Resolve a MolSpec handle into an RDKit Mol, matching GAFF behavior."""
        try:
            from ..core.molspec import MolSpec
            from ..core import naming
        except Exception:
            return mol

        if isinstance(mol, MolSpec):
            spec = mol
            if not spec.name:
                # Skip the current helper frame so user-script variable names win
                # over internal aliases such as `spec`.
                spec.name = naming.infer_var_name(mol, depth=3) or naming.infer_var_name(mol, depth=2) or None
            try:
                resolved = self.mol_rdkit(
                    spec.smiles,
                    name=spec.name,
                    prefer_db=spec.prefer_db,
                    require_ready=spec.require_ready,
                    charge=spec.charge,
                    basis_set=spec.basis_set,
                    method=spec.method,
                    resp_profile=spec.resp_profile,
                )
            except Exception as exc:
                charge_token = str(getattr(spec, "charge", "") or "").strip().upper()
                if charge_token == "RESP":
                    utils.radon_print(
                        f"RESP-ready MolDB entry unavailable for {spec.smiles}; "
                        "falling back to OPLS-AA built-in charges.",
                        level=1,
                    )
                    resolved = self.mol_rdkit(
                        spec.smiles,
                        name=spec.name,
                        prefer_db=spec.prefer_db,
                        require_ready=False,
                        charge="opls",
                        basis_set=spec.basis_set,
                        method=spec.method,
                        resp_profile=spec.resp_profile,
                    )
                    try:
                        resolved.SetProp("_yadonpy_charge_fallback", "opls")
                        resolved.SetProp("_yadonpy_charge_fallback_reason", str(exc))
                    except Exception:
                        pass
                else:
                    raise
            try:
                spec.cache_resolved_mol(resolved)
            except Exception:
                pass
            try:
                naming.ensure_name(resolved, name=spec.name, depth=2, prefer_var=False)
            except Exception:
                pass
            return resolved
        return mol

    def ff_assign(self, mol, charge=None, retryMDL=True, useMDL=True, report: bool = True, **charge_kwargs):
        """
        OPLSAA.ff_assign

        Args:
            mol: RDKit Mol or MolSpec handle.

        Optional:
            charge:
                - None / 'keep' / 'existing': preserve any pre-existing charges.
                  If no complete charges are present, automatically fall back to
                  built-in OPLS-AA type charges.
                - 'opls': assign charges from the embedded OPLS-AA rule table.
                - other: delegate to calc.assign_charges (same as GAFF)
            retryMDL/useMDL: same behavior as GAFF
        """
        from ..core import calc, naming

        mol = self._resolve_spec(mol)
        try:
            current_name = naming.get_name(mol, default=None)
        except Exception:
            current_name = None
        try:
            naming.ensure_name(mol, name=current_name, depth=2, prefer_var=(current_name is None))
        except Exception:
            pass
        try:
            if naming.try_restore_assigned_mol(mol, depth=2, ff_name=self.name):
                if report:
                    print_ff_assignment_report(mol, ff_obj=self)
                return mol
        except Exception:
            pass
        bonded_override = self._normalize_bonded_override(charge_kwargs.get("bonded"))
        preserve_prebuilt_bonded = bool(
            bonded_override in ("drih", "mseminario") and self._has_precomputed_bonded_fragment(mol)
        )
        if bonded_override not in (None, "drih", "mseminario"):
            raise RuntimeError(
                f"bonded={charge_kwargs.get('bonded')!r} requested but unsupported for OPLS-AA "
                "(supported: 'DRIH', 'mseminario')."
            )
        if bonded_override in ("drih", "mseminario"):
            try:
                mol.SetProp("_yadonpy_bonded_override", "1")
                mol.SetProp("_yadonpy_bonded_explicit", "1")
                mol.SetProp("_yadonpy_bonded_requested", str(bonded_override))
                mol.SetProp("_yadonpy_bonded_signature", str(bonded_override))
                if preserve_prebuilt_bonded:
                    mol.SetProp(
                        "_yadonpy_bonded_method",
                        "DRIH" if bonded_override == "drih" else "mseminario",
                    )
            except Exception:
                pass
        effective_charge = self._normalize_charge_mode(charge)
        fallback_to_opls = False
        if effective_charge is None and not self._has_complete_atomic_charges(mol):
            effective_charge = 'opls'
            fallback_to_opls = True

        if useMDL:
            Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
            Chem.rdmolops.SetAromaticity(mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL)

        mol.SetProp('ff_name', str(self.name))
        mol.SetProp('ff_class', str(self.ff_class))

        if fallback_to_opls:
            utils.radon_print(
                'No complete pre-existing atomic charges found; falling back to built-in OPLS-AA type charges.',
                level=1,
            )

        if self._has_reusable_existing_assignment(mol, ff_name=self.name):
            has_new_bond = self._has_marked_new_bond(mol)
            if has_new_bond:
                result = self.refresh_polymer_junction_terms(mol, charge=effective_charge)
            elif effective_charge == "opls":
                result = self._assign_opls_charges_from_existing_types(mol)
            else:
                result = True
            if result and report:
                print_ff_assignment_report(mol, ff_obj=self)
            if result:
                try:
                    naming.auto_export_assigned_mol(mol, depth=2)
                except Exception:
                    pass
            return mol if result else False

        result = self.assign_ptypes(mol, charge=effective_charge)
        if result and not preserve_prebuilt_bonded:
            result = self.assign_btypes(mol)
        if result and not preserve_prebuilt_bonded:
            result = self.assign_atypes(mol)
        if result and not preserve_prebuilt_bonded:
            result = self.assign_dtypes(mol)
        if result and not preserve_prebuilt_bonded:
            result = self.assign_itypes(mol)
        if result and preserve_prebuilt_bonded:
            utils.radon_print(
                'Preserving precomputed bonded fragment during OPLS-AA reassignment; '
                'only nonbonded types/charges were refreshed.',
                level=1,
            )

        # If charge is not 'opls', use YadonPy's generic charge assignment.
        if result and effective_charge is not None and effective_charge != 'opls':
            result = calc.assign_charges(mol, charge=effective_charge, **charge_kwargs)

        if not result and retryMDL and not useMDL:
            utils.radon_print('Retry to assign with MDL aromaticity model', level=1)
            Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
            Chem.rdmolops.SetAromaticity(mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL)

            result = self.assign_ptypes(mol, charge=effective_charge)
            if result and not preserve_prebuilt_bonded:
                result = self.assign_btypes(mol)
            if result and not preserve_prebuilt_bonded:
                result = self.assign_atypes(mol)
            if result and not preserve_prebuilt_bonded:
                result = self.assign_dtypes(mol)
            if result and not preserve_prebuilt_bonded:
                result = self.assign_itypes(mol)
            if result and effective_charge is not None and effective_charge != 'opls':
                result = calc.assign_charges(mol, charge=effective_charge, **charge_kwargs)
            if result:
                utils.radon_print('Success to assign with MDL aromaticity model', level=1)

        if result and report:
            print_ff_assignment_report(mol, ff_obj=self)
        if result:
            try:
                naming.auto_export_assigned_mol(mol, depth=2)
            except Exception:
                pass
        return mol if result else False

    @staticmethod
    def _has_implicit_h(mol):
        # Many SMARTS rules use degree (D4) semantics that require explicit hydrogens.
        # If the molecule has implicit H, those rules will silently fail and typing will be wrong.
        for a in mol.GetAtoms():
            if a.GetSymbol() != 'H' and a.GetNumImplicitHs() > 0:
                return True
        return False

    def assign_ptypes(self, mol, charge=None):
        """
        Assign particle types (nonbonded) using SMARTS rules.

        Sets per-atom:
          - ff_type   : 'opls_###' (used for LJ params)
          - ff_btype  : bond_type label (CT/CA/HC/...) used for bonded lookups
          - ff_sigma/ff_epsilon : from oplsaa.json particle_types
          - AtomicCharge (if charge='opls')
        """
        mol.SetProp('pair_style', self.pair_style)

        if self._has_implicit_h(mol):
            utils.radon_print(
                'OPLS-AA SMARTS typing requires explicit hydrogens (Chem.AddHs). '
                'Found implicit H on at least one heavy atom. Aborting typing.',
                level=3
            )
            return False

        compiled = _get_compiled_rules()

        # assignment tables (later rules overwrite earlier ones)
        chosen = {}  # atom_idx -> rule dict

        for rule, q in compiled:
            # ``uniquify=False`` is intentional here: symmetric ions / neutral
            # molecules such as CO2, PF6-, and cyclic sulfates otherwise expose
            # only one representative match, leaving equivalent atoms untyped.
            matches = mol.GetSubstructMatches(q, uniquify=False)
            if not matches:
                continue
            for m in matches:
                if not m:
                    continue
                idx = m[0]
                a = mol.GetAtomWithIdx(idx)
                if a.GetSymbol() != rule.element:
                    continue
                chosen[idx] = rule

        self._apply_special_inorganic_type_fallbacks(mol, chosen)

        # Check coverage
        ok = True
        for a in mol.GetAtoms():
            idx = a.GetIdx()
            if idx not in chosen:
                utils.radon_print(f'OPLS-AA typing failed: atom {idx} ({a.GetSymbol()}) did not match any SMARTS rule.', level=2)
                ok = False

        if not ok:
            return False

        # Apply assignment
        for a in mol.GetAtoms():
            idx = a.GetIdx()
            rule = chosen[idx]
            opls_type = rule.opls
            btype = rule.btype

            # store bond_type separately for bonded lookup
            a.SetProp('ff_btype', btype)

            # use GAFF's set_ptype to set ff_type + LJ params from json
            if opls_type not in self.param.pt:
                utils.radon_print(f'OPLS-AA typing failed: nonbonded type {opls_type} not found in oplsaa.json', level=3)
                return False

            self.set_ptype(a, opls_type)
            if opls_type in _DONOR_H_LJ_FLOOR:
                sigma_floor, epsilon_floor = _DONOR_H_LJ_FLOOR[opls_type]
                sigma_now = float(a.GetDoubleProp("ff_sigma")) if a.HasProp("ff_sigma") else 0.0
                epsilon_now = float(a.GetDoubleProp("ff_epsilon")) if a.HasProp("ff_epsilon") else 0.0
                if sigma_now <= 0.0 and epsilon_now <= 0.0:
                    a.SetDoubleProp("ff_sigma", float(sigma_floor))
                    a.SetDoubleProp("ff_epsilon", float(epsilon_floor))
                    a.SetProp("ff_provenance", "local_refine:donor_h_lj_floor")
            source = _param_source(self.param.pt[opls_type])
            if source:
                a.SetProp("ff_source", source)
            try:
                a.SetProp('ff_desc', str(rule.desc))
            except Exception:
                pass

            # Optional: assign type charge from rule table
            if charge == 'opls':
                q = rule.charge
                if q is None:
                    q = 0.0
                a.SetDoubleProp('AtomicCharge', float(q))

        return True

    @staticmethod
    def _apply_special_inorganic_type_fallbacks(mol, chosen):
        """Small structural fallbacks for legacy / lossy inorganic-ion graphs."""
        try:
            if mol is None or int(mol.GetNumAtoms()) != 7:
                return
            phosphorus = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P"]
            fluorine = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "F"]
            if len(phosphorus) != 1 or len(fluorine) != 6:
                return
            center = phosphorus[0]
            neighbors = list(center.GetNeighbors())
            if int(center.GetDegree()) != 6 or len(neighbors) != 6:
                return
            if any(nb.GetSymbol() != "F" for nb in neighbors):
                return

            p_rule = _get_rule_by_opls("opls_785")
            f_rule = _get_rule_by_opls("opls_786")
            if p_rule is None or f_rule is None:
                return

            chosen[center.GetIdx()] = p_rule
            for atom in fluorine:
                chosen[atom.GetIdx()] = f_rule
        except Exception:
            return

    # ----------------------------
    # Bonded assignments (use ff_btype)
    # ----------------------------

    def _clone_param(self, template_param, **overrides):
        obj = self.Container()
        for key, value in vars(template_param).items():
            setattr(obj, key, value)
        for key, value in overrides.items():
            setattr(obj, key, value)
        return obj

    @staticmethod
    def _copy_param_provenance(target, param) -> None:
        if isinstance(param, dict):
            source = param.get("source")
            provenance = param.get("provenance")
        else:
            source = _param_source(param)
            provenance = getattr(param, "provenance", None)
        if source:
            try:
                setattr(target, "source", str(source))
            except Exception:
                pass
        if provenance:
            try:
                setattr(target, "provenance", str(provenance))
            except Exception:
                pass

    def _find_param(self, mapping, tokens):
        """Find the best bonded parameter for a token tuple."""
        key = ','.join(tokens)
        if key in mapping:
            return mapping[key], 'exact'

        rkey = ','.join(reversed(tokens))
        if rkey in mapping:
            return mapping[rkey], 'reverse'

        for alias_tokens in _iter_btype_alias_token_sets(tokens):
            if alias_tokens == tuple(tokens):
                continue
            key = ','.join(alias_tokens)
            if key in mapping:
                return mapping[key], 'alias'
            rkey = ','.join(reversed(alias_tokens))
            if rkey in mapping:
                return mapping[rkey], 'alias-reverse'

        best_obj = None
        best_score = None
        for obj in _iter_unique_params(mapping):
            names = []
            for attr in ('tag', 'name', 'rname'):
                value = getattr(obj, attr, None)
                if value and value not in names:
                    names.append(value)
            for name in names:
                score = _match_bonded_pattern(name.split(','), tokens)
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_obj = obj
                    best_score = score

        if best_obj is not None:
            return best_obj, 'wildcard'

        return None, None

    def _log_special_override(self, label, message):
        if label in self._logged_special_overrides:
            return
        self._logged_special_overrides.add(label)
        utils.radon_print(message, level=1)

    @staticmethod
    def _lookup_special_base_tokens(tokens, mapping):
        key = tuple(tokens)
        if key in mapping:
            return mapping[key]

        rkey = tuple(reversed(key))
        if rkey in mapping:
            return tuple(reversed(mapping[rkey]))

        return None

    def _special_angle_param(self, tokens):
        # Cyclic carbonates (EC/PC) in the classic OPLS bonded tables are known
        # to miss the OS-C-OS angle even though the dedicated nonbonded types
        # exist (opls_771-779).  We keep the OPLS force constant from the nearest
        # available carbonyl-carbon angle (O,C,OS) and use the published EC/PC
        # equilibrium angle 110.6 degrees for the missing OS-C-OS term.
        if tuple(tokens) == ('OS', 'C', 'OS'):
            base, _ = self._find_param(self.param.at, ('O', 'C', 'OS'))
            if base is None:
                return None

            self._log_special_override(
                'angle:OS,C,OS',
                'Applying cyclic-carbonate OPLS-AA fallback for missing angle OS,C,OS '
                '(theta0=110.6 deg; force constant copied from O,C,OS).'
            )
            return self._clone_param(
                base,
                tag='OS,C,OS',
                name='OS,C,OS',
                rname='OS,C,OS',
                theta0=110.6,
                source='local_refine',
                provenance='cyclic_carbonate_angle_fallback',
            )

        fallback_map = {
            # Acyclic carbonate esters (EMC/DEC)
            ('OS', 'C_2', 'OS'): ('O_2', 'C_2', 'OS'),
            # Carboxymethyl-glucose sidechain linkage
            ('OS', 'CT', 'CO'): ('CO', 'CT', 'OH'),
            ('CO', 'OH', 'CT'): ('CT', 'OH', 'CO'),
            ('C', 'NT', 'H'): ('C', 'N', 'H'),
            ('NT', 'CA', 'NT'): ('N2', 'CZ', 'NZ'),
            # Methoxide / thiolate methyl groups
            ('HC', 'C3', 'HC'): ('HC', 'CT', 'HC'),
            ('HC', 'C3', 'OH'): ('HC', 'CT', 'OH'),
            ('HC', 'C3', 'SH'): ('HC', 'CT', 'SH'),
            # Nitrate ion
            ('O', 'N', 'O'): ('ON', 'NO', 'ON'),
            # 1,3,2-dioxathiol-2,2-dioxide (DTD) / cyclic sulfate family
            ('OY', 'SY', 'OS'): ('OY', 'SY', 'CT'),
            ('OS', 'SY', 'OS'): ('OS', 'P~', 'OS'),
            ('SY', 'OS', 'CM'): ('CT', 'OS', 'CM'),
        }
        base_tokens = self._lookup_special_base_tokens(tokens, fallback_map)
        if base_tokens is None:
            return None

        base, _ = self._find_param(self.param.at, base_tokens)
        if base is None:
            return None

        pretty = ','.join(tokens)
        self._log_special_override(
            f'angle:{pretty}',
            'Applying OPLS-AA fallback for missing angle '
            f'{pretty} (copied from {",".join(base_tokens)}).'
        )
        return self._clone_param(
            base,
            tag=pretty,
            name=pretty,
            rname=','.join(reversed(tokens)),
            source='local_refine',
            provenance='special_angle_fallback',
        )

    def _special_bond_param(self, tokens):
        fallback_map = {
            # Methoxide / thiolate methyl groups use the ordinary CT-HC bond.
            ('C3', 'HC'): ('CT', 'HC'),
            # Nitrate uses legacy N/O btypes in the SMARTS table while bonded
            # terms are keyed as NO/ON.
            ('O', 'N'): ('ON', 'NO'),
        }
        base_tokens = self._lookup_special_base_tokens(tokens, fallback_map)
        if base_tokens is None:
            return None

        base, _ = self._find_param(self.param.bt, base_tokens)
        if base is None:
            return None

        pretty = ','.join(tokens)
        self._log_special_override(
            f'bond:{pretty}',
            'Applying OPLS-AA fallback for missing bond '
            f'{pretty} (copied from {",".join(base_tokens)}).'
        )
        return self._clone_param(
            base,
            tag=pretty,
            name=pretty,
            rname=','.join(reversed(tokens)),
            source='local_refine',
            provenance='special_bond_fallback',
        )

    def _special_dihedral_param(self, tokens):
        # Cyclic carbonates (EC/PC) also miss the CT-OS-C-OS family in the
        # bonded tables.  Reuse the nearest carbonyl analogue that is already in
        # OPLS-AA: CT-OS-C-O (and its reverse orientation).
        fallback_map = {
            ('CT', 'OS', 'C', 'OS'): ('CT', 'OS', 'C', 'O'),
            ('OS', 'C', 'OS', 'CT'): ('O', 'C', 'OS', 'CT'),
            # Acyclic carbonate esters (EMC/DEC)
            ('CT', 'OS', 'C_2', 'OS'): ('CT', 'OS', 'C_2', 'O_2'),
            # Amides
            ('CT', 'C', 'NT', 'H'): ('CT', 'C', 'N', 'H'),
            ('O', 'C', 'NT', 'H'): ('O', 'C', 'N', 'H'),
            ('HC', 'CT', 'C', 'NT'): ('CT', 'CT', 'C', 'N'),
            # Carboxymethyl-glucose sidechains
            ('HO', 'OH', 'CO', 'CT'): ('HO', 'OH', 'CO', 'OS'),
            ('HO', 'OH', 'CO', 'HC'): ('HO', 'OH', 'CO', 'OS'),
            ('CO', 'CT', 'OS', 'CT'): ('CT', 'OS', 'CO', 'CT'),
            ('OS', 'CT', 'C_3', 'O2'): ('CT', 'CT', 'C_3', 'O2'),
            ('HC', 'CT', 'C_3', 'O2'): ('CT', 'CT', 'C_3', 'O2'),
            ('OS', 'CT', 'CO', 'OH'): ('CT', 'CT', 'CO', 'OH'),
            ('OS', 'CT', 'CO', 'OS'): ('CT', 'CT', 'CO', 'OS'),
            ('OS', 'CT', 'CO', 'HC'): ('CT', 'CT', 'CO', 'HC'),
            ('OH', 'CT', 'CO', 'OH'): ('CT', 'CT', 'CO', 'OH'),
            ('OH', 'CT', 'CO', 'OS'): ('CT', 'CT', 'CO', 'OS'),
            ('OH', 'CT', 'CO', 'HC'): ('CT', 'CT', 'CO', 'HC'),
            ('CT', 'OH', 'CO', 'OS'): ('HO', 'OH', 'CO', 'OS'),
            ('CT', 'OH', 'CO', 'CT'): ('HO', 'OH', 'CO', 'CT'),
            ('CT', 'OH', 'CO', 'HC'): ('HO', 'OH', 'CO', 'HC'),
            ('CT', 'CT', 'OH', 'CO'): ('HO', 'OH', 'CO', 'CT'),
            ('HC', 'CT', 'OH', 'CO'): ('HO', 'OH', 'CO', 'HC'),
            ('HC', 'CT', 'CO', 'OH'): ('HC', 'CT', 'CO', 'OS'),
            ('HC', 'CT', 'CO', 'HC'): ('HC', 'CT', 'CO', 'OS'),
            # 1,3,2-dioxathiol-2,2-dioxide (DTD) / cyclic sulfate family
            ('OY', 'SY', 'OS', 'CM'): ('O~', 'P~', 'OS', 'CA'),
            ('OS', 'SY', 'OS', 'CM'): ('CT', 'P~', 'OS', 'CT'),
            ('SY', 'OS', 'CM', 'CM'): ('CT', 'OS', 'CM', 'CT'),
            ('SY', 'OS', 'CM', 'HC'): ('CT', 'OS', 'CM', 'HC'),
            # Guanidinium / amidinium families
            ('H', 'NT', 'CA', 'NT'): ('H', 'N2', 'CA', 'N2'),
            ('NT', 'CA', 'NT', 'H'): ('N2', 'CA', 'N2', 'H'),
        }
        base_tokens = self._lookup_special_base_tokens(tokens, fallback_map)
        if base_tokens is None:
            return None

        base, _ = self._find_param(self.param.dt, base_tokens)
        if base is None:
            return None

        pretty = ','.join(tokens)
        self._log_special_override(
            f'dihedral:{pretty}',
            'Applying cyclic-carbonate OPLS-AA fallback for missing dihedral '
            f'{pretty} (copied from {",".join(base_tokens)}).'
        )
        return self._clone_param(
            base,
            tag=pretty,
            name=pretty,
            rname=','.join(reversed(tokens)),
            source='local_refine',
            provenance='special_dihedral_fallback',
        )

    def _lookup_bond_param(self, tokens):
        param, _ = self._find_param(self.param.bt, tokens)
        if param is not None:
            return param
        return self._special_bond_param(tokens)

    def _lookup_angle_param(self, tokens):
        param, _ = self._find_param(self.param.at, tokens)
        if param is not None:
            return param
        return self._special_angle_param(tokens)

    def _lookup_dihedral_param(self, tokens):
        param, _ = self._find_param(self.param.dt, tokens)
        if param is not None:
            return param
        return self._special_dihedral_param(tokens)

    def assign_btypes(self, mol):
        mol.SetProp('bond_style', self.bond_style)
        result = True

        for b in mol.GetBonds():
            a1 = b.GetBeginAtom()
            a2 = b.GetEndAtom()
            if not a1.HasProp('ff_btype') or not a2.HasProp('ff_btype'):
                utils.radon_print('ff_btype missing on atoms. Did you run assign_ptypes first?', level=3)
                return False

            tokens = (a1.GetProp('ff_btype'), a2.GetProp('ff_btype'))
            param = self._lookup_bond_param(tokens)
            if param is None:
                utils.radon_print(f'Cannot assign bond parameters for {tokens[0]},{tokens[1]}', level=2)
                result = False
                continue
            self.set_btype(b, param)
        return result

    def set_btype(self, b, param):
        b.SetProp('ff_type', param.tag)
        b.SetDoubleProp('ff_k', param.k)
        b.SetDoubleProp('ff_r0', param.r0)
        source = _param_source(param)
        if source:
            b.SetProp("ff_source", source)
        provenance = getattr(param, "provenance", None)
        if provenance:
            b.SetProp("ff_provenance", str(provenance))
        return True

    def assign_atypes(self, mol):
        mol.SetProp('angle_style', self.angle_style)
        setattr(mol, 'angles', {})

        # enumerate angles i-j-k where j is center
        result = True
        for j in mol.GetAtoms():
            if not j.HasProp('ff_btype'):
                utils.radon_print('ff_btype missing on atoms. Did you run assign_ptypes first?', level=3)
                return False
            nbrs = [n for n in j.GetNeighbors()]
            if len(nbrs) < 2:
                continue
            for a_idx in range(len(nbrs)):
                for c_idx in range(a_idx+1, len(nbrs)):
                    i = nbrs[a_idx]
                    k = nbrs[c_idx]
                    tokens = (i.GetProp('ff_btype'), j.GetProp('ff_btype'), k.GetProp('ff_btype'))
                    param = self._lookup_angle_param(tokens)
                    if param is None:
                        utils.radon_print(f'Cannot assign angle parameters for {tokens[0]},{tokens[1]},{tokens[2]}', level=2)
                        result = False
                        continue
                    self.set_atype(mol, i.GetIdx(), j.GetIdx(), k.GetIdx(), param)
        return result

    def set_atype(self, mol, a, b, c, param):
        ff = ff_class.Angle_harmonic(
            ff_type=param.tag,
            k=param.k,
            theta0=param.theta0
        )
        self._copy_param_provenance(ff, param)
        angle = core_utils.Angle(
            a=a, b=b, c=c,
            ff=ff
        )
        key = f'{a},{b},{c}'
        mol.angles[key] = angle
        return True

    def assign_dtypes(self, mol):
        mol.SetProp('dihedral_style', self.dihedral_style)
        setattr(mol, 'dihedrals', {})

        result = True
        # enumerate dihedrals i-j-k-l by traversing each bond j-k
        for bond in mol.GetBonds():
            j = bond.GetBeginAtom()
            k = bond.GetEndAtom()
            if not j.HasProp('ff_btype') or not k.HasProp('ff_btype'):
                utils.radon_print('ff_btype missing on atoms. Did you run assign_ptypes first?', level=3)
                return False

            jn = [a for a in j.GetNeighbors() if a.GetIdx() != k.GetIdx()]
            kn = [a for a in k.GetNeighbors() if a.GetIdx() != j.GetIdx()]
            if not jn or not kn:
                continue

            for i in jn:
                for l in kn:
                    a = i.GetIdx(); b = j.GetIdx(); c = k.GetIdx(); d = l.GetIdx()
                    key_idx = f'{a},{b},{c},{d}'
                    if key_idx in mol.dihedrals:
                        continue

                    tokens = (
                        i.GetProp('ff_btype'),
                        j.GetProp('ff_btype'),
                        k.GetProp('ff_btype'),
                        l.GetProp('ff_btype'),
                    )
                    param = self._lookup_dihedral_param(tokens)
                    if param is None:
                        utils.radon_print(
                            f'Cannot assign dihedral parameters for {tokens[0]},{tokens[1]},{tokens[2]},{tokens[3]}',
                            level=2,
                        )
                        result = False
                        continue
                    self.set_dtype(mol, a, b, c, d, param)
        return result

    def set_dtype(self, mol, a, b, c, d, param):
        ff = ff_class.Dihedral_fourier(
            ff_type=param.tag,
            k=param.k,
            d0=param.d,
            m=param.m,
            n=param.n
        )
        self._copy_param_provenance(ff, param)
        dih = core_utils.Dihedral(
            a=a, b=b, c=c, d=d,
            ff=ff
        )
        key = f'{a},{b},{c},{d}'
        mol.dihedrals[key] = dih
        return True

    def assign_itypes(self, mol):
        mol.SetProp('improper_style', self.improper_style)
        setattr(mol, 'impropers', {})
        templates = get_oplsaa_improper_templates()
        result = True
        for candidate in iter_source_backed_oplsaa_improper_candidates(mol):
            atoms = tuple(int(x) for x in candidate["atoms"])
            template_name = str(candidate["template"])
            param = templates.get(template_name)
            if param is None:
                utils.radon_print(
                    f"Missing OPLS-AA improper template {template_name} for motif {candidate.get('motif')}",
                    level=2,
                )
                result = False
                continue
            if not self.set_itype(mol, *atoms, param):
                result = False
        return result

    def set_itype(self, mol, a, b, c, d, param):
        ff = ff_class.Improper_cvff(
            ff_type=str(param["tag"]),
            k=float(param["k"]),
            d0=int(param["d"]),
            n=int(param["n"]),
        )
        self._copy_param_provenance(ff, param)
        key = f"{a},{b},{c},{d}"
        mol.impropers[key] = core_utils.Improper(a=a, b=b, c=c, d=d, ff=ff)
        return True
