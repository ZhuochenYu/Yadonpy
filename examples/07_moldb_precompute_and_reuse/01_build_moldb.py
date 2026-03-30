from __future__ import annotations

"""Example 07 / Step 1: One-shot precompute of a broad electrolyte MolDB set.

This script is intended as the "do it once" entry point for the species that
show up repeatedly across the YadonPy electrolyte examples:

- neutral monomers / terminal groups used by Example 02/06;
- CMC monomer set used by Example 05/12/13;
- the aromatic polymer repeat unit used by Example 10/11;
- common carbonate / ether solvents and additives;
- common Li-salt anions plus monoatomic counter-ions.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem

import yadonpy as yp
from yadonpy.core import chem_utils as core_utils
from yadonpy.core.polyelectrolyte import detect_charged_groups

HERE = Path(__file__).resolve().parent
CATALOG_CSV = HERE / "electrolyte_species.csv"


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    smiles: str
    kind: str
    source: str


def _read_species_csv(path: Path) -> list[SpeciesSpec]:
    rows: list[SpeciesSpec] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            name = str(raw.get("name") or "").strip()
            smiles = str(raw.get("smiles") or "").strip()
            if not name or not smiles:
                continue
            kind = str(raw.get("kind") or ("psmiles" if "*" in smiles else "smiles")).strip()
            source = str(raw.get("source") or "example07").strip()
            rows.append(SpeciesSpec(name=name, smiles=smiles, kind=kind, source=source))
    return rows


def _dedupe_species(items: list[SpeciesSpec]) -> list[SpeciesSpec]:
    by_smiles: dict[str, SpeciesSpec] = {}
    for item in items:
        by_smiles.setdefault(item.smiles, item)
    return sorted(by_smiles.values(), key=lambda x: (x.kind, x.name.lower(), x.smiles))


def _is_monatomic_ion(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumAtoms() == 1 and int(Chem.GetFormalCharge(mol)) != 0


def _use_drih(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        return bool(core_utils.is_high_symmetry_polyhedral_ion(mol, smiles_hint=smiles))
    except Exception:
        return False


def _is_polyelectrolyte_monomer(smiles: str) -> bool:
    if "*" not in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        except Exception:
            mol = None
    if mol is None:
        return False
    summary = detect_charged_groups(mol, detection="auto")
    return bool(summary.get("groups"))


def _assign_and_store(spec: SpeciesSpec, *, db_dir: Path, work_root: Path) -> dict[str, Any]:
    work_dir = work_root / spec.name
    work_dir.mkdir(parents=True, exist_ok=True)

    if _is_monatomic_ion(spec.smiles):
        ff = yp.get_ff("merz")
        mol = ff.mol(spec.smiles)
        ok = bool(ff.ff_assign(mol, report=False))
        if not ok:
            raise RuntimeError(f"MERZ assignment failed for {spec.name} {spec.smiles}")
        ff.__class__.store_to_db(
            mol,
            smiles_or_psmiles=spec.smiles,
            name=spec.name,
            db_dir=db_dir,
            charge="RESP",
        )
        return {
            "name": spec.name,
            "smiles": spec.smiles,
            "source": spec.source,
            "ff": "merz",
            "bonded": None,
            "polyelectrolyte_mode": False,
        }

    ff = yp.get_ff("gaff2_mod")
    mol = ff.mol(spec.smiles)
    poly_mode = _is_polyelectrolyte_monomer(spec.smiles)
    bonded = "DRIH" if _use_drih(spec.smiles) else None
    ok = bool(
        ff.ff_assign(
            mol,
            charge="RESP",
            bonded=bonded,
            report=False,
            polyelectrolyte_mode=poly_mode,
            work_dir=work_dir,
        )
    )
    if not ok:
        raise RuntimeError(f"GAFF2_mod assignment failed for {spec.name} {spec.smiles}")
    ff.__class__.store_to_db(
        mol,
        smiles_or_psmiles=spec.smiles,
        name=spec.name,
        db_dir=db_dir,
        charge="RESP",
        polyelectrolyte_mode=poly_mode,
        polyelectrolyte_detection="auto",
    )
    return {
        "name": spec.name,
        "smiles": spec.smiles,
        "source": spec.source,
        "ff": "gaff2_mod",
        "bonded": bonded,
        "polyelectrolyte_mode": bool(poly_mode),
    }


def main() -> int:
    db = yp.MolDB()
    db_dir = Path(db.db_dir)
    work_root = HERE / "work_dir" / "01_build_moldb"
    db_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    species = _dedupe_species(_read_species_csv(CATALOG_CSV))

    summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in species:
        try:
            result = _assign_and_store(spec, db_dir=db_dir, work_root=work_root)
            summary.append(result)
            print(f"[OK] {spec.name:20s} {spec.smiles}")
        except Exception as exc:
            failures.append({"name": spec.name, "smiles": spec.smiles, "error": repr(exc)})
            print(f"[FAIL] {spec.name:20s} {spec.smiles} :: {exc}")

    out = {
        "catalog_csv": str(CATALOG_CSV.resolve()),
        "db_dir": str(db_dir.resolve()),
        "work_root": str(work_root.resolve()),
        "success_count": len(summary),
        "failure_count": len(failures),
        "success": summary,
        "failures": failures,
    }
    (work_root / "build_moldb_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nMolDB directory: {db_dir}")
    print(f"Catalog CSV   : {CATALOG_CSV}")
    print(f"Success       : {len(summary)}")
    print(f"Failures      : {len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
