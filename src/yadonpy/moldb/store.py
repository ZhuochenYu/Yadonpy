
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from rdkit import Chem
from rdkit.Chem import rdmolfiles
try:
    from rdkit import RDLogger
except Exception:  # pragma: no cover - optional on some stripped RDKit builds
    RDLogger = None

from ..core import chem_utils as utils
from ..io.mol2 import write_mol2


_BONDED_META_KEYS = (
    "_yadonpy_bonded_signature",
    "_yadonpy_bonded_requested",
    "_yadonpy_bonded_method",
    "_yadonpy_bonded_override",
    "_yadonpy_bonded_explicit",
)

_BONDED_FILE_KEYS = (
    "_yadonpy_bonded_itp",
    "_yadonpy_bonded_json",
    "_yadonpy_mseminario_itp",
    "_yadonpy_mseminario_json",
)


def _bonded_meta_from_mol(mol: Chem.Mol) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    try:
        if hasattr(mol, "HasProp"):
            for key in _BONDED_META_KEYS + _BONDED_FILE_KEYS:
                if mol.HasProp(key):
                    value = str(mol.GetProp(key)).strip()
                    if value:
                        meta[key] = value
    except Exception:
        pass
    return meta


def _default_db_dir() -> Path:
    # Explicit override for MolDB
    env = os.environ.get("YADONPY_MOLDB")
    if env:
        return Path(env).expanduser().resolve()

    # If the user uses a custom YADONPY_HOME or YADONPY_DATA_DIR, keep MolDB under it.
    base = os.environ.get("YADONPY_HOME") or os.environ.get("YADONPY_DATA_DIR")
    if base:
        return (Path(base).expanduser().resolve() / "moldb").resolve()

    # Default (as requested)
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
        prefer_unsanitized = _prefer_unsanitized_mol2(s)
        mol = None
        if prefer_unsanitized:
            try:
                mol = Chem.MolFromSmiles(s, sanitize=False)
            except Exception:
                mol = None
            if mol is not None:
                try:
                    mol.UpdatePropertyCache(strict=False)
                except Exception:
                    pass
                try:
                    Chem.SanitizeMol(
                        mol,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                    )
                except Exception:
                    pass
        if mol is None:
            mol = Chem.MolFromSmiles(s)
        if mol is None:
            try:
                mol = Chem.MolFromSmiles(s, sanitize=False)
            except Exception:
                mol = None
            if mol is not None:
                try:
                    mol.UpdatePropertyCache(strict=False)
                except Exception:
                    pass
                try:
                    Chem.SanitizeMol(
                        mol,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                    )
                except Exception:
                    pass
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


def _prefer_unsanitized_mol2(smiles_hint: str | None) -> bool:
    if not smiles_hint:
        return False
    try:
        probe = Chem.MolFromSmiles(str(smiles_hint), sanitize=False)
    except Exception:
        probe = None
    if probe is None:
        return False
    try:
        probe.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        if utils.is_high_symmetry_polyhedral_ion(probe, smiles_hint=str(smiles_hint)):
            return True
        if utils.is_inorganic_ion_like(probe, smiles_hint=str(smiles_hint)):
            return True
    except Exception:
        pass
    return False


def _load_mol2_candidate(mol2_path: Path, *, smiles_hint: str | None = None) -> Optional[Chem.Mol]:
    prefer_unsanitized = _prefer_unsanitized_mol2(smiles_hint)

    if prefer_unsanitized:
        try:
            if RDLogger is not None:
                RDLogger.DisableLog('rdApp.*')
            mol = rdmolfiles.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
        except Exception:
            mol = None
        finally:
            if RDLogger is not None:
                RDLogger.EnableLog('rdApp.*')
        if mol is None:
            return None
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            pass
        return mol

    try:
        mol = rdmolfiles.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
    except Exception:
        mol = None
    if mol is not None:
        return mol

    try:
        mol = rdmolfiles.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
    except Exception:
        mol = None
    if mol is None:
        return None

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    except Exception:
        pass
    return mol


@dataclass
class MolRecord:
    key: str
    kind: str               # smiles | psmiles
    canonical: str
    name: str
    # "default" charge metadata (kept for backwards compatibility)
    charge_method: Optional[str] = None
    charge_unit: str = "e"
    ready: bool = False

    # Multiple charge variants for the same canonical smiles/psmiles.
    # Key: variant_id (short hash); value: metadata dict
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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
            "variants": self.variants,
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
            variants=d.get("variants") or {},
        )


def variant_id(*, charge: str = "RESP", basis_set: str | None = None, method: str | None = None) -> str:
    """Stable short id for a charge variant."""
    c = (charge or "RESP").strip().upper()
    b = (basis_set or "Default").strip()
    m = (method or "Default").strip()
    payload = f"{c}|{b}|{m}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


class MolDB:
    """File-based molecule database.

    Stores: canonical smiles/psmiles + initial 3D geometry (mol2) + charges (json)
    + optional QM-derived bonded patch fragments (DRIH / mseminario).

    Force-field assignment itself is intentionally NOT stored here; it is fast and can
    be done on demand. Only the expensive, non-default bonded overrides are persisted.
    """

    def __init__(self, db_dir: Optional[Path] = None):
        self.db_dir = (db_dir or _default_db_dir()).expanduser().resolve()
        self.objects_dir = self.db_dir / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)

        # Optional: CSV path used by autocalculate() (user scripts can set this)
        self.read_calc_temp: Optional[str] = None

    def record_dir(self, key: str) -> Path:
        return self.objects_dir / key

    def manifest_path(self, key: str) -> Path:
        return self.record_dir(key) / "manifest.json"

    def mol2_path(self, key: str) -> Path:
        return self.record_dir(key) / "best.mol2"

    def charges_path(self, key: str) -> Path:
        # Backwards-compatible single charges file ("default" variant)
        return self.record_dir(key) / "charges.json"

    def charges_variant_path(self, key: str, vid: str) -> Path:
        d = self.record_dir(key) / "charges"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{vid}.json"

    def bonded_variant_dir(self, key: str, vid: str) -> Path:
        d = self.record_dir(key) / "bonded" / vid
        d.mkdir(parents=True, exist_ok=True)
        return d

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
        write_mol2(mol=mol, out_mol2=friendly, name=name)
        stable = self.mol2_path(key)
        if stable != friendly:
            shutil.copy2(friendly, stable)
        return stable

    def save_charges(
        self,
        key: str,
        mol: Chem.Mol,
        *,
        charge: str = "RESP",
        basis_set: str | None = None,
        method: str | None = None,
    ) -> str:
        charges: List[float] = []
        for atom in mol.GetAtoms():
            if atom.HasProp("AtomicCharge"):
                charges.append(float(atom.GetDoubleProp("AtomicCharge")))
            elif atom.HasProp(charge):
                charges.append(float(atom.GetDoubleProp(charge)))
            else:
                charges.append(0.0)
        vid = variant_id(charge=charge, basis_set=basis_set, method=method)
        payload = {
            "key": key,
            "variant_id": vid,
            "charge": charge,
            "basis_set": basis_set or "Default",
            "method": method or "Default",
            "charges": charges,
        }

        # Save the variant payload
        self.charges_variant_path(key, vid).write_text(json.dumps(payload, indent=2) + "\n")

        # Also keep/update the legacy single file for interoperability
        self.charges_path(key).write_text(json.dumps(payload, indent=2) + "\n")
        return vid

    def attach_charges(
        self,
        mol: Chem.Mol,
        key: str,
        *,
        charge: str = "RESP",
        basis_set: str | None = None,
        method: str | None = None,
    ) -> Chem.Mol:
        vid = variant_id(charge=charge, basis_set=basis_set, method=method)
        cp = self.charges_variant_path(key, vid)
        if not cp.exists():
            # fallback to legacy
            cp = self.charges_path(key)
            if not cp.exists():
                return mol

        payload = json.loads(cp.read_text())
        charges = payload.get("charges", [])
        for i, atom in enumerate(mol.GetAtoms()):
            if i < len(charges):
                atom.SetDoubleProp("AtomicCharge", float(charges[i]))
                atom.SetDoubleProp(payload.get("charge", "RESP"), float(charges[i]))
        return mol

    def _persist_bonded_variant(self, mol: Chem.Mol, *, key: str, vid: str) -> Dict[str, Any]:
        raw = _bonded_meta_from_mol(mol)
        if not raw:
            return {}

        out_dir = self.bonded_variant_dir(key, vid)
        files: Dict[str, str] = {}
        for prop in _BONDED_FILE_KEYS:
            src = raw.get(prop)
            if not src:
                continue
            try:
                src_path = Path(src).expanduser().resolve()
            except Exception:
                continue
            if not src_path.is_file():
                continue
            dst_name = src_path.name
            dst_path = out_dir / dst_name
            shutil.copy2(src_path, dst_path)
            files[prop] = dst_name

        meta = {key: raw[key] for key in _BONDED_META_KEYS if key in raw}
        if files:
            meta["files"] = files
        return meta

    def _restore_bonded_variant(self, mol: Chem.Mol, *, key: str, vid: str, rec: MolRecord) -> None:
        variants = rec.variants or {}
        variant_meta = variants.get(vid) or {}
        bonded_meta = variant_meta.get("bonded") if isinstance(variant_meta, dict) else None
        if not isinstance(bonded_meta, dict) or not bonded_meta:
            return

        for prop in _BONDED_META_KEYS:
            value = bonded_meta.get(prop)
            if isinstance(value, str) and value.strip():
                mol.SetProp(prop, value.strip())

        files = bonded_meta.get("files")
        if isinstance(files, dict):
            for prop, rel_name in files.items():
                if prop not in _BONDED_FILE_KEYS or not isinstance(rel_name, str) or not rel_name.strip():
                    continue
                path = self.record_dir(key) / "bonded" / vid / rel_name.strip()
                if path.is_file():
                    mol.SetProp(prop, str(path.resolve()))

        try:
            jp = None
            if mol.HasProp("_yadonpy_bonded_json"):
                jp = Path(str(mol.GetProp("_yadonpy_bonded_json")).strip())
            elif mol.HasProp("_yadonpy_mseminario_json"):
                jp = Path(str(mol.GetProp("_yadonpy_mseminario_json")).strip())
            if jp is not None and jp.is_file():
                from ..sim.qm import apply_mseminario_params_to_mol

                payload = json.loads(jp.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    apply_mseminario_params_to_mol(mol, payload, overwrite=True)
        except Exception:
            pass

    def load_mol(
        self,
        smiles_or_psmiles: str,
        *,
        require_ready: bool = False,
        charge: str = "RESP",
        basis_set: str | None = None,
        method: str | None = None,
    ) -> Tuple[Chem.Mol, MolRecord]:
        kind, canon, key = canonical_key(smiles_or_psmiles)
        rec = self.load_record(key)
        if rec is None:
            raise FileNotFoundError(f"Mol not found in DB: key={key} ({kind})")

        if require_ready:
            vid = variant_id(charge=charge, basis_set=basis_set, method=method)
            vmeta = (rec.variants or {}).get(vid)
            if vmeta is None or (not bool(vmeta.get("ready", False))):
                # Backwards-compat: respect legacy rec.ready for default variant only
                if not (rec.ready and vid == variant_id(charge="RESP", basis_set=None, method=None)):
                    raise RuntimeError(
                        f"Mol exists but not ready for variant={vid} ({charge}, {basis_set or 'Default'}, {method or 'Default'}): key={key}"
                    )
        # Load geometry. Prefer the stable best.mol2 path, but recover gracefully if
        # that file is missing or corrupted while another MOL2 copy in the same record
        # directory is still readable.
        record_dir = self.record_dir(key)
        mol2_candidates: list[Path] = []
        preferred_mol2 = self.mol2_path(key)
        if preferred_mol2.exists():
            mol2_candidates.append(preferred_mol2)

        for cand in sorted(record_dir.glob("*.mol2"), key=lambda p: p.stat().st_mtime, reverse=True):
            if cand not in mol2_candidates:
                mol2_candidates.append(cand)

        if not mol2_candidates:
            raise FileNotFoundError(f"Geometry not found for key={key}")

        mol = None
        failed_mol2: list[Path] = []
        for mol2p in mol2_candidates:
            mol = _load_mol2_candidate(mol2p, smiles_hint=rec.canonical)
            if mol is not None:
                break
            failed_mol2.append(mol2p)

        if mol is None:
            failed_list = ", ".join(str(p) for p in failed_mol2) or str(preferred_mol2)
            raise RuntimeError(f"Failed to read mol2 from any candidate: {failed_list}")

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

        requested_vid = variant_id(charge=charge, basis_set=basis_set, method=method)
        mol = self.attach_charges(mol, key, charge=charge, basis_set=basis_set, method=method)
        self._restore_bonded_variant(mol, key=key, vid=requested_vid, rec=rec)
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
        charge: str = "RESP",
        basis_set: str | None = None,
        method: str | None = None,
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

        vid = self.save_charges(key, mol, charge=charge, basis_set=basis_set, method=method)

        bonded_meta = self._persist_bonded_variant(mol, key=key, vid=vid)

        # Update manifest: variants
        rec.variants = rec.variants or {}
        rec.variants[vid] = {
            "variant_id": vid,
            "charge": str(charge),
            "basis_set": basis_set or "Default",
            "method": method or "Default",
            "ready": True,
        }
        if bonded_meta:
            rec.variants[vid]["bonded"] = bonded_meta

        # Backwards-compat single-variant fields
        rec.charge_method = str(charge)
        rec.ready = True
        self.save_record(rec)

        mol.SetProp("_YADONPY_KEY", key)
        mol.SetProp("_YADONPY_CANONICAL", rec.canonical)
        mol.SetProp("_YADONPY_KIND", rec.kind)
        return rec

    def mol_gen(
        self,
        mol: Chem.Mol,
        *,
        work_dir: str | os.PathLike | None = None,
        add_to_moldb: bool = False,
        smiles_or_psmiles: Optional[str] = None,
        name: Optional[str] = None,
        charge: str = "RESP",
        basis_set: str | None = None,
        method: str | None = None,
    ) -> Tuple[MolRecord, Path]:
        """Generate MolDB-formatted artifacts from an already-prepared molecule.

        Typical usage (after qm.conformation_search + qm.assign_charges)::

            db = MolDB()
            db.mol_gen(mol, work_dir=work_dir, add_to_moldb=False)

        Behavior:
          - add_to_moldb=False (default): write a *standalone* MolDB under work_dir
            so the user can copy-paste into ~/.yadonpy/moldb later.
          - add_to_moldb=True: directly add/update the entry inside this MolDB.

        Return:
          (record, out_db_dir)
        """
        from datetime import datetime
        from pathlib import Path

        from ..core import naming
        from ..core.logging_utils import yadon_print

        # Resolve SMILES/PSMILES (best-effort).
        smi = smiles_or_psmiles
        if not smi:
            try:
                if mol.HasProp("_yadonpy_input_smiles"):
                    smi = str(mol.GetProp("_yadonpy_input_smiles")).strip()
            except Exception:
                smi = None
        if not smi:
            try:
                if mol.HasProp("_yadonpy_smiles"):
                    smi = str(mol.GetProp("_yadonpy_smiles")).strip()
            except Exception:
                smi = None
        if not smi:
            # Fallback: derive from RDKit and convert connector hydrogens back to '*'.
            try:
                smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                smi = utils.h2star(smi)
            except Exception:
                smi = None

        # Resolve name (best-effort).
        mol_name = name or naming.get_name(mol)

        if bool(add_to_moldb):
            rec = self.update_from_mol(
                mol,
                smiles_or_psmiles=smi,
                name=mol_name,
                charge=charge,
                basis_set=basis_set,
                method=method,
            )
            yadon_print(
                f"[DONE] MolDB.mol_gen(add_to_moldb=True): added/updated {rec.name} (key={rec.key}) in {self.db_dir}",
                level=1,
            )
            return rec, self.db_dir

        # add_to_moldb=False -> write a standalone MolDB under work_dir
        if work_dir is None:
            raise TypeError("MolDB.mol_gen() requires work_dir when add_to_moldb=False")

        work_dir_p = Path(work_dir).expanduser().resolve()
        work_dir_p.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_db_dir = (work_dir_p / f"moldb_snippet_{stamp}").resolve()
        out_db = MolDB(db_dir=out_db_dir)

        rec = out_db.update_from_mol(
            mol,
            smiles_or_psmiles=smi,
            name=mol_name,
            charge=charge,
            basis_set=basis_set,
            method=method,
        )

        # Help message for copy/paste.
        yadon_print(
            f"[DONE] MolDB.mol_gen(add_to_moldb=False): wrote a standalone MolDB under {out_db_dir}",
            level=1,
        )
        yadon_print(
            f"[CMD] Copy into global MolDB (optional): cp -r {out_db_dir}/objects/{rec.key} {self.objects_dir}/",
            level=1,
        )
        return rec, out_db_dir

    # ------------------------------------------------------------------
    # High-level helper: precompute (geometry + charges) from a CSV.
    # ------------------------------------------------------------------
    def autocalculate(
        self,
        work_root: str | os.PathLike | None = None,
        *,
        # Friendly aliases used by examples / user scripts
        work_dir: str | os.PathLike | None = None,
        mem: int | None = None,
        psi4_omp: int | None = None,
        ff=None,
        mpi: int = 1,
        omp: int = 16,
        omp_psi4: int = 16,
        memory_mb: int = 16000,
        add_to_moldb: bool = False,
    ) -> None:
        """Precompute MolDB entries from a template CSV.

        Expected columns:
          - name, smiles
          - opt (0/1, optional)
          - confsearch (0/1, optional)
          - charge_method (optional; default RESP)
          - basis_set (optional; default uses QM defaults)
          - method (optional; default uses QM defaults)

        Notes:
          - total charge is inferred from SMILES formal charges when possible.
          - Polymer PSMILES ('*') are treated as net-neutral unless the SMILES
            explicitly encodes formal charges.
        """
        from pathlib import Path
        from datetime import datetime

        try:
            import pandas as pd
        except Exception as e:
            raise RuntimeError("pandas is required for MolDB.autocalculate()") from e

        from ..ff.gaff2_mod import GAFF2_mod
        from ..sim import qm
        from ..core import naming
        from ..core.logging_utils import yadon_print

        if ff is None:
            ff = GAFF2_mod()

        csv_path = self.read_calc_temp
        if not csv_path:
            raise ValueError("MolDB.read_calc_temp is not set (path to template CSV)")
        csv_path = str(Path(csv_path).expanduser().resolve())

        # --- normalize aliases ---
        if work_dir is not None:
            work_root = work_dir
        if work_root is None:
            raise TypeError("MolDB.autocalculate() missing required argument: work_root/work_dir")
        if mem is not None:
            memory_mb = int(mem)
        if psi4_omp is not None:
            omp_psi4 = int(psi4_omp)

        # Back-compat convenience: many users intuitively pass omp=<psi4_threads>
        # when calling with work_dir=. If omp_psi4 is still default, treat omp as omp_psi4.
        if work_dir is not None and psi4_omp is None and int(omp_psi4) == 16 and int(omp) != 16:
            omp_psi4 = int(omp)

        work_root_p = Path(work_root).expanduser().resolve()
        work_root_p.mkdir(parents=True, exist_ok=True)

        # Decide where to write MolDB artifacts.
        # - Default: write a fresh MolDB under work_root (no side effects in ~/.yadonpy).
        # - If add_to_moldb=True: write into this MolDB instance (usually ~/.yadonpy/moldb).
        if bool(add_to_moldb):
            out_db = self
            out_db_dir = self.db_dir
        else:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_db_dir = (work_root_p / f"moldb_generated_{stamp}").resolve()
            out_db = MolDB(db_dir=out_db_dir)
            yadon_print(
                f"[INFO] MolDB.autocalculate(): add_to_moldb=False -> writing a local MolDB under: {out_db_dir}",
                level=1,
            )

        df = pd.read_csv(csv_path)

        def _norm_default(val):
            """Normalize CSV fields so that 'Default' behaves like an unset value.

            Users can write 'Default' (case-insensitive) for method/basis_set in
            template.csv. Internally we treat it as None so downstream code uses
            YadonPy's built-in default QM levels.
            """
            if val is None:
                return None
            try:
                # pandas may give NaN (float) which is not equal to itself
                if val != val:  # NaN
                    return None
            except Exception:
                pass
            s = str(val).strip()
            if not s:
                return None
            if s.lower() in ("default", "none", "nan", "null"):
                return None
            return s
        for _, row in df.iterrows():
            name = str(row.get("name", "")).strip()
            smiles = str(row.get("smiles", "")).strip()
            if (not name) or (not smiles):
                continue

            opt = bool(int(row.get("opt", 0))) if "opt" in row else False
            confsearch = bool(int(row.get("confsearch", 1))) if "confsearch" in row else True

            charge = str(row.get("charge_method", row.get("charge", "RESP")) or "RESP").strip()
            basis_set = _norm_default(row.get("basis_set", None))
            method = _norm_default(row.get("method", None))

            # Skip if already ready for this variant (in the chosen output DB)
            try:
                _mol0, rec0 = out_db.load_mol(smiles, require_ready=True, charge=charge, basis_set=basis_set, method=method)
                yadon_print(f"[SKIP] {name}: already ready in MolDB (key={rec0.key}) {charge}/{basis_set or 'Default'}/{method or 'Default'}", level=1)
                continue
            except Exception:
                pass

            yadon_print(f"[RUN ] {name}: {smiles} ({charge}/{basis_set or 'Default'}/{method or 'Default'})", level=1)

            # Create or load initial geometry from MolDB (the chosen output DB)
            try:
                mol, _rec = out_db.build_or_load(smiles, name=name, prefer_db=True)
            except Exception:
                # fallback: build raw
                mol = utils.mol_from_smiles(smiles)
                naming.ensure_name(mol, name=name, depth=2)

            # QM scratch dir
            item_dir = work_root_p / name
            item_dir.mkdir(parents=True, exist_ok=True)

            if confsearch:
                try:
                    mol, _energy = qm.conformation_search(
                        mol,
                        ff=ff,
                        work_dir=item_dir,
                        psi4_omp=int(omp_psi4),
                        mpi=int(mpi),
                        omp=int(omp),
                        memory=int(memory_mb),
                        log_name=name,
                    )
                except Exception:
                    # conf search is a convenience; don't fail the whole batch
                    pass

            qm.assign_charges(
                mol,
                charge=str(charge),
                opt=bool(opt),
                work_dir=item_dir,
                omp=int(omp_psi4),
                memory=int(memory_mb),
                log_name=name,
                total_charge=None,
                total_multiplicity=1,
                charge_method=(method if method is not None else "wb97m-d3bj"),
                charge_basis=(basis_set if basis_set is not None else "def2-TZVP"),
            )

            # Persist best geometry + charges (as a variant)
            out_db.update_from_mol(
                mol,
                smiles_or_psmiles=smiles,
                name=name,
                charge=str(charge),
                basis_set=basis_set,
                method=method,
            )

            rec = out_db.load_record(mol.GetProp("_YADONPY_KEY"))
            yadon_print(f"[DONE] {name}: stored (key={getattr(rec, 'key', None)}) -> {out_db_dir}", level=1)

        # Final hint for users
        if bool(add_to_moldb):
            yadon_print(f"[DONE] MolDB.autocalculate(): results added to MolDB: {self.db_dir}", level=1)
        else:
            yadon_print(f"[DONE] MolDB.autocalculate(): results written to local MolDB: {out_db_dir}", level=1)

        return None

    def check(self) -> None:
        """Print a formatted summary of the MolDB contents.

        This is a lightweight, human-facing tool meant to help quickly spot:
          - how many entries you have
          - which SMILES/PSMILES are present
          - which charge variants exist (RESP + method/basis)
          - potential duplicates by canonical string (should be rare for SMILES)

        It only inspects the current MolDB directory (self.db_dir).
        """
        from ..core.logging_utils import yadon_print

        records: List[MolRecord] = []
        for mp in sorted(self.objects_dir.glob('*/manifest.json')):
            try:
                rec = MolRecord.from_dict(json.loads(mp.read_text()))
                records.append(rec)
            except Exception:
                continue

        if not records:
            yadon_print(f"[INFO] MolDB.check(): no records found in {self.db_dir}", level=1)
            return

        def _vfmt(vmeta: Dict[str, Any]) -> str:
            c = str(vmeta.get('charge', 'RESP'))
            b = str(vmeta.get('basis_set', 'Default'))
            m = str(vmeta.get('method', 'Default'))
            r = 'ready' if bool(vmeta.get('ready', False)) else 'not-ready'
            bonded = ''
            bonded_meta = vmeta.get('bonded') if isinstance(vmeta, dict) else None
            if isinstance(bonded_meta, dict):
                bonded_method = bonded_meta.get('_yadonpy_bonded_method') or bonded_meta.get('_yadonpy_bonded_signature')
                if isinstance(bonded_method, str) and bonded_method.strip():
                    bonded = f", bonded={bonded_method.strip()}"
            return f"{c}/{b}/{m}({r}{bonded})"

        rows: List[Dict[str, str]] = []
        for rec in records:
            variants = rec.variants or {}
            vlist = [_vfmt(v) for _k, v in sorted(variants.items())]
            if not vlist:
                if rec.ready:
                    vlist = ["RESP/Default/Default(ready)"]
                else:
                    vlist = ["(no variants)"]

            rows.append(
                {
                    'key': rec.key,
                    'name': rec.name,
                    'kind': rec.kind,
                    'canonical': rec.canonical,
                    'variants': '; '.join(vlist),
                }
            )

        # Detect duplicates by canonical string
        canon_map: Dict[str, List[str]] = {}
        for r in rows:
            canon_map.setdefault(r['canonical'], []).append(r['key'])
        dups = {c: ks for c, ks in canon_map.items() if len(ks) > 1}

        # Pretty print as fixed-width columns
        cols = ['key', 'name', 'kind', 'canonical', 'variants']
        widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
        widths['canonical'] = min(widths['canonical'], 60)
        widths['variants'] = min(widths['variants'], 80)

        def _short(s: str, w: int) -> str:
            s = str(s)
            if len(s) <= w:
                return s
            if w <= 3:
                return s[:w]
            return s[: w - 3] + '...'

        yadon_print(f"[INFO] MolDB.check(): {len(rows)} record(s) in {self.db_dir}", level=1)
        header = ' | '.join(_short(c, widths[c]).ljust(widths[c]) for c in cols)
        sep = '-+-'.join('-' * widths[c] for c in cols)
        print(header)
        print(sep)
        for r in rows:
            line = ' | '.join(_short(r[c], widths[c]).ljust(widths[c]) for c in cols)
            print(line)

        if dups:
            yadon_print(f"[WARN] MolDB.check(): potential duplicates by canonical string: {len(dups)}", level=2)
            for canon, keys in dups.items():
                print(f"  - {canon}: {', '.join(keys)}")
        else:
            yadon_print("[DONE] MolDB.check(): no duplicates by canonical string detected.", level=1)
