from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem

import yadonpy as yp
from yadonpy.core import chem_utils as core_utils
from yadonpy.core.data_dir import find_bundle_archive
from yadonpy.core.polyelectrolyte import detect_charged_groups


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    smiles: str
    kind: str
    source: str


EXTRA_SPECIES: tuple[SpeciesSpec, ...] = (
    SpeciesSpec("ClO4", "[O-][Cl](=O)(=O)=O", "smiles", "extra"),
    SpeciesSpec("BF4", "F[B-](F)(F)F", "smiles", "extra"),
    SpeciesSpec("AsF6", "F[As-](F)(F)(F)(F)F", "smiles", "extra"),
    SpeciesSpec("FSI", "FS(=O)(=O)[N-]S(=O)(=O)F", "smiles", "extra"),
    SpeciesSpec("TFSI", "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F", "smiles", "extra"),
    SpeciesSpec("Li", "[Li+]", "smiles", "extra"),
)


def _bundle_records(archive: Path) -> list[SpeciesSpec]:
    records: list[SpeciesSpec] = []
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith("/manifest.json"):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            try:
                payload = json.load(fh)
            except Exception:
                continue
            canonical = str(payload.get("canonical") or "").strip()
            if not canonical:
                continue
            records.append(
                SpeciesSpec(
                    name=str(payload.get("name") or payload.get("key") or canonical).strip(),
                    smiles=canonical,
                    kind=str(payload.get("kind") or "smiles").strip(),
                    source="bundle",
                )
            )
    return records


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
        return {"name": spec.name, "smiles": spec.smiles, "ff": "merz", "bonded": None, "polyelectrolyte_mode": False}

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
        "ff": "gaff2_mod",
        "bonded": bonded,
        "polyelectrolyte_mode": bool(poly_mode),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild bundled MolDB species and selected extra battery-anion species.")
    parser.add_argument("--archive", type=Path, default=None, help="Path to yd_moldb.tar. Default: auto-detect.")
    parser.add_argument("--db-dir", type=Path, default=Path.home() / ".yadonpy" / "moldb", help="Target MolDB directory.")
    parser.add_argument("--work-root", type=Path, default=Path("./work_rebuild_bundle_species"), help="Per-species working directory root.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for smoke runs.")
    args = parser.parse_args()

    archive = args.archive or find_bundle_archive(cwd=Path.cwd())
    if archive is None or not Path(archive).is_file():
        raise FileNotFoundError("Could not locate yd_moldb.tar. Pass --archive explicitly.")

    bundle_species = _bundle_records(Path(archive))
    species = _dedupe_species(bundle_species + list(EXTRA_SPECIES))
    if args.limit > 0:
        species = species[: int(args.limit)]

    args.db_dir.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in species:
        try:
            result = _assign_and_store(spec, db_dir=args.db_dir, work_root=args.work_root)
            summary.append(result)
            print(f"[OK] {spec.name:16s} {spec.smiles}")
        except Exception as exc:
            failures.append({"name": spec.name, "smiles": spec.smiles, "error": repr(exc)})
            print(f"[FAIL] {spec.name:16s} {spec.smiles} :: {exc}")

    out = {
        "archive": str(Path(archive).resolve()),
        "db_dir": str(args.db_dir.resolve()),
        "work_root": str(args.work_root.resolve()),
        "success_count": len(summary),
        "failure_count": len(failures),
        "success": summary,
        "failures": failures,
    }
    (args.work_root / "rebuild_bundle_species_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
