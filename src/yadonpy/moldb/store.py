
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from rdkit import Chem
from rdkit.Chem import rdmolfiles

from ..core import chem_utils as utils
from ..io.mol2 import write_mol2_from_rdkit


def _default_db_dir() -> Path:
    env = os.environ.get("YADONPY_MOLDB")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".yadonpy" / "moldb").resolve()


def _is_psmiles(s: str) -> bool:
    # Heuristic: polymer SMILES contains connection point wildcard '*'
    return "*" in s


def canonical_key(smiles_or_psmiles: str) -> Tuple[str, str, str]:
    """Return (kind, canonical_string, key).

    kind: 'smiles' or 'psmiles'
    key: content-addressed sha256 over canonical string
    """
    s = smiles_or_psmiles.strip()
    if _is_psmiles(s):
        kind = "psmiles"
        # For psmiles, RDKit canonicalization can reorder wildcard atoms in ways that may
        # lose intended connector labeling. We keep the raw (trimmed) string as canonical.
        canon = s
    else:
        kind = "smiles"
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles_or_psmiles!r}")
        canon = Chem.MolToSmiles(mol, canonical=True)
    key = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
    return kind, canon, key


def _extract_connectors(mol: Chem.Mol) -> List[Dict[str, int]]:
    """Extract connector atoms for PSMILES.

    In YadonPy, '*' connection points are represented as isotopic hydrogens [nH] with n>=3 (see core.chem_utils.star2h).
    MOL2 does not preserve isotope labels, so we persist connector indices/isotopes in manifest.json and restore on load.
    """
    conns: List[Dict[str, int]] = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == "H":
            iso = int(a.GetIsotope() or 0)
            if iso >= 3:
                conns.append({"idx": int(a.GetIdx()), "isotope": iso})
    return conns


def _restore_connectors(mol: Chem.Mol, connectors: Optional[List[Dict[str, int]]]) -> None:
    if not connectors:
        return
    for c in connectors:
        try:
            idx = int(c.get("idx"))
            iso = int(c.get("isotope", 3))
        except Exception:
            continue
        if 0 <= idx < mol.GetNumAtoms():
            a = mol.GetAtomWithIdx(idx)
            if a.GetSymbol() == "H":
                a.SetIsotope(iso)


@dataclass
class MolRecord:
    key: str
    kind: str               # smiles | psmiles
    canonical: str
    name: str
    charge_method: Optional[str] = None
    charge_unit: str = "e"
    ready: bool = False
    connectors: Optional[List[Dict[str, int]]] = None  # for psmiles: linker atom indices/isotopes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "canonical": self.canonical,
            "name": self.name,
            "charge_method": self.charge_method,
            "charge_unit": self.charge_unit,
            "ready": self.ready,
            "connectors": self.connectors,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MolRecord":
        return cls(
            key=d["key"],
            kind=d["kind"],
            canonical=d["canonical"],
            name=d.get("name", d["key"]),
            charge_method=d.get("charge_method"),
            charge_unit=d.get("charge_unit", "e"),
            ready=bool(d.get("ready", False)),
            connectors=d.get("connectors"),
        )


class MolDB:
    """File-based molecule database.

    Stores: canonical smiles/psmiles + initial 3D geometry (mol2) + charges (json).
    Force-field assignment is intentionally NOT stored here; it is fast and can be done on demand.
    """

    def __init__(self, db_dir: Optional[Path] = None):
        self.db_dir = (db_dir or _default_db_dir()).expanduser().resolve()
        self.objects_dir = self.db_dir / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)

    def record_dir(self, key: str) -> Path:
        return self.objects_dir / key

    def manifest_path(self, key: str) -> Path:
        return self.record_dir(key) / "manifest.json"

    def mol2_path(self, key: str) -> Path:
        return self.record_dir(key) / "best.mol2"

    def charges_path(self, key: str) -> Path:
        return self.record_dir(key) / "charges.json"

    def exists(self, key: str) -> bool:
        return self.manifest_path(key).exists()

    def load_record(self, key: str) -> Optional[MolRecord]:
        mp = self.manifest_path(key)
        if not mp.exists():
            return None
        return MolRecord.from_dict(json.loads(mp.read_text()))

    def save_record(self, rec: MolRecord) -> None:
        d = self.record_dir(rec.key)
        d.mkdir(parents=True, exist_ok=True)
        self.manifest_path(rec.key).write_text(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False) + "\n")

    def save_geometry(self, key: str, mol: Chem.Mol, *, name: str) -> Path:
        out_dir = self.record_dir(key)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save as <name>.mol2 for user-friendliness, but keep a stable 'best.mol2' too.
        friendly = out_dir / f"{name}.mol2"
        write_mol2_from_rdkit(mol=mol, out_mol2=friendly, name=name)
        stable = self.mol2_path(key)
        if stable != friendly:
            shutil.copy2(friendly, stable)
        return stable

    def save_charges(self, key: str, mol: Chem.Mol, *, method: str = "RESP") -> None:
        charges: List[float] = []
        for atom in mol.GetAtoms():
            if atom.HasProp("AtomicCharge"):
                charges.append(float(atom.GetDoubleProp("AtomicCharge")))
            elif atom.HasProp(method):
                charges.append(float(atom.GetDoubleProp(method)))
            else:
                charges.append(0.0)
        payload = {
            "key": key,
            "method": method,
            "charges": charges,
        }
        self.charges_path(key).write_text(json.dumps(payload, indent=2) + "\n")

    def attach_charges(self, mol: Chem.Mol, key: str) -> Chem.Mol:
        cp = self.charges_path(key)
        if not cp.exists():
            return mol
        payload = json.loads(cp.read_text())
        charges = payload.get("charges", [])
        for i, atom in enumerate(mol.GetAtoms()):
            if i < len(charges):
                atom.SetDoubleProp("AtomicCharge", float(charges[i]))
                atom.SetDoubleProp(payload.get("method", "RESP"), float(charges[i]))
        return mol

    def load_mol(self, smiles_or_psmiles: str, *, require_ready: bool = False) -> Tuple[Chem.Mol, MolRecord]:
        kind, canon, key = canonical_key(smiles_or_psmiles)
        rec = self.load_record(key)
        if rec is None:
            raise FileNotFoundError(f"Mol not found in DB: key={key} ({kind})")
        if require_ready and not rec.ready:
            raise RuntimeError(f"Mol exists but not ready (charges missing): key={key}")
        # Load geometry
        mol2p = self.mol2_path(key)
        if not mol2p.exists():
            # fallback: any mol2 in folder
            cands = list(self.record_dir(key).glob("*.mol2"))
            if not cands:
                raise FileNotFoundError(f"Geometry not found for key={key}")
            mol2p = max(cands, key=lambda p: p.stat().st_mtime)
        mol = rdmolfiles.MolFromMol2File(str(mol2p), sanitize=True, removeHs=False)
        if mol is None:
            raise RuntimeError(f"Failed to read mol2: {mol2p}")

        # Restore psmiles connector isotopes (MOL2 does not preserve isotopes)
        if rec.kind == "psmiles":
            if not rec.connectors:
                try:
                    ref = utils.mol_from_smiles(rec.canonical)
                    rec.connectors = _extract_connectors(ref)
                    self.save_record(rec)  # one-time migration for old DB entries
                except Exception:
                    rec.connectors = rec.connectors or []
            _restore_connectors(mol, rec.connectors)

        mol = self.attach_charges(mol, key)
        mol.SetProp('_YADONPY_KEY', key)
        mol.SetProp('_YADONPY_CANONICAL', rec.canonical)
        mol.SetProp('_YADONPY_KIND', rec.kind)
        return mol, rec

    def build_or_load(self, smiles_or_psmiles: str, *, name: Optional[str] = None, prefer_db: bool = True) -> Tuple[Chem.Mol, MolRecord]:
        kind, canon, key = canonical_key(smiles_or_psmiles)
        if prefer_db and self.exists(key):
            mol, rec = self.load_mol(smiles_or_psmiles, require_ready=False)
            return mol, rec

        # Build initial mol
        mol = utils.mol_from_smiles(canon)

        rec = MolRecord(key=key, kind=kind, canonical=canon, name=name or key, ready=False)
        if kind == "psmiles":
            rec.connectors = _extract_connectors(mol)
        self.save_record(rec)
        self.save_geometry(key, mol, name=rec.name)
        mol.SetProp('_YADONPY_KEY', key)
        mol.SetProp('_YADONPY_CANONICAL', rec.canonical)
        mol.SetProp('_YADONPY_KIND', rec.kind)
        return mol, rec


    def update_from_mol(
        self,
        mol: Chem.Mol,
        *,
        smiles_or_psmiles: Optional[str] = None,
        name: Optional[str] = None,
        charge_method: str = "RESP",
    ) -> MolRecord:
        """Persist current geometry + charges from an in-memory mol into the DB.

        Intended for workflows where you build a molecule, run conformer search / QM charge assignment,
        and then store the resulting best geometry + charges for later reuse.

        Key resolution:
          1) if mol has '_YADONPY_KEY', use it
          2) else compute from smiles_or_psmiles (required)
        """
        key = mol.GetProp("_YADONPY_KEY") if mol.HasProp("_YADONPY_KEY") else None
        kind = mol.GetProp("_YADONPY_KIND") if mol.HasProp("_YADONPY_KIND") else None
        canonical = mol.GetProp("_YADONPY_CANONICAL") if mol.HasProp("_YADONPY_CANONICAL") else None

        if not key:
            if not smiles_or_psmiles:
                raise ValueError("smiles_or_psmiles is required when mol has no _YADONPY_KEY prop")
            kind, canonical, key = canonical_key(smiles_or_psmiles)

        rec = self.load_record(key)
        if rec is None:
            if not canonical:
                if not smiles_or_psmiles:
                    raise ValueError("smiles_or_psmiles is required to create new DB record")
                kind, canonical, _ = canonical_key(smiles_or_psmiles)
            rec = MolRecord(key=key, kind=kind or "smiles", canonical=canonical or "", name=name or key, ready=False)

        if name:
            rec.name = name

        if rec.kind == "psmiles":
            rec.connectors = _extract_connectors(mol)

        self.save_record(rec)
        self.save_geometry(key, mol, name=rec.name)
        self.save_charges(key, mol, method=charge_method)
        rec.charge_method = charge_method
        rec.ready = True
        self.save_record(rec)

        mol.SetProp("_YADONPY_KEY", key)
        mol.SetProp("_YADONPY_CANONICAL", rec.canonical)
        mol.SetProp("_YADONPY_KIND", rec.kind)
        return rec
