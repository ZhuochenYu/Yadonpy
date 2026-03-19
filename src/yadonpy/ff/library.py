"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.data_dir import ensure_initialized


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize smiles.

    We intentionally keep this lightweight (strip only), because RDKit may not
    be available at import time in some environments.
    """
    return smiles.strip()


def is_polymer_smiles(smiles: str) -> bool:
    """Heuristic: polymer monomer SMILES contain '*' connection points."""
    return "*" in smiles


def smiles_to_molid(smiles: str, n: int = 10) -> str:
    """Stable molecule id derived from SMILES."""
    s = canonicalize_smiles(smiles).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:n]


@dataclass
class LibraryEntry:
    mol_id: str
    smiles: str
    ff: str
    artifact_dir: str
    is_polymer_monomer: bool
    is_original_from_lib: bool
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mol_id": self.mol_id,
            "smiles": self.smiles,
            "ff": self.ff,
            "artifact_dir": self.artifact_dir,
            "is_polymer_monomer": self.is_polymer_monomer,
            "is_original_from_lib": self.is_original_from_lib,
            "created_at": self.created_at,
        }


class LibraryDB:
    """User-writable molecule library.

    The library lives under yadonpy's data root and is keyed by SMILES.
    """

    def __init__(self) -> None:
        self.layout = ensure_initialized()
        self.path = self.layout.library_json

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def find_by_smiles(self, ff: str, smiles: str) -> Optional[Dict[str, Any]]:
        smiles = canonicalize_smiles(smiles)
        data = self._load()
        ff_block = (data.get("force_fields") or {}).get(ff) or {}
        for ent in ff_block.get("basic", []):
            if canonicalize_smiles(ent.get("smiles", "")) == smiles:
                return ent
        return None

    def resolve_artifact_dir(self, artifact_dir: str) -> Path:
        """Resolve artifact_dir stored in library.

        We allow storing paths relative to the yadonpy data root (recommended),
        so the library can be shipped as a packaged default.
        """
        p = Path(artifact_dir)
        if p.is_absolute():
            return p
        return self.layout.root / p

    def ensure_registered(
        self,
        ff: str,
        smiles: str,
        artifact_dir: Path,
        *,
        is_original_from_lib: bool = False,
    ) -> LibraryEntry:
        """Add molecule to library if absent.

        For non-polymer molecules (no '*' in SMILES), yadonpy will call this
        after the first successful parameterization/export.
        """
        smiles_c = canonicalize_smiles(smiles)
        existing = self.find_by_smiles(ff, smiles_c)
        if existing is not None:
            return LibraryEntry(
                mol_id=existing["mol_id"],
                smiles=existing["smiles"],
                ff=existing["ff"],
                artifact_dir=existing["artifact_dir"],
                is_polymer_monomer=bool(existing.get("is_polymer_monomer", False)),
                is_original_from_lib=bool(existing.get("is_original_from_lib", False)),
                created_at=float(existing.get("created_at", 0.0)),
            )

        mol_id = smiles_to_molid(smiles_c)
        is_poly = is_polymer_smiles(smiles_c)

        entry = LibraryEntry(
            mol_id=mol_id,
            smiles=smiles_c,
            ff=ff,
            artifact_dir=str(artifact_dir),
            is_polymer_monomer=is_poly,
            is_original_from_lib=is_original_from_lib,
            created_at=time.time(),
        )

        data = self._load()
        data.setdefault("force_fields", {})
        data["force_fields"].setdefault(ff, {"basic": []})
        data["force_fields"][ff].setdefault("basic", [])
        data["force_fields"][ff]["basic"].append(entry.to_dict())
        self._save(data)
        return entry
