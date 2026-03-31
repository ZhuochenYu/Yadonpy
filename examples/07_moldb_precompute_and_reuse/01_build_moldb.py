from __future__ import annotations

"""Example 07 / Step 1: One-shot MolDB build for common electrolyte species."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yadonpy as yp
from yadonpy.core import workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.core.polyelectrolyte import detect_charged_groups
from yadonpy.diagnostics import doctor
from yadonpy.moldb import MolDB
from yadonpy.runtime import set_run_options
from yadonpy.sim.qm import _pick_first_available_basis


HERE = Path(__file__).resolve().parent
CATALOG_CSV = HERE / "electrolyte_species.csv"
RESP_FF_NAME = "gaff2_mod"


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    smiles: str
    kind: str
    source: str
    charge: str
    bonded: str | None
    polyelectrolyte_mode: bool


@dataclass(frozen=True)
class QMSpec:
    method: str
    opt_basis: str
    charge_basis: str
    reason: str


def _csv_bool(value: object, *, default: bool = False) -> bool:
    token = str(value or "").strip().lower()
    if not token:
        return bool(default)
    return token in {"1", "true", "t", "yes", "y", "on"}


def _read_species_csv(path: Path) -> list[SpeciesSpec]:
    items: list[SpeciesSpec] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            name = str(raw.get("name") or "").strip()
            smiles = str(raw.get("smiles") or "").strip()
            if not name or not smiles or smiles in seen:
                continue
            seen.add(smiles)
            items.append(
                SpeciesSpec(
                    name=name,
                    smiles=smiles,
                    kind=str(raw.get("kind") or ("psmiles" if "*" in smiles else "smiles")).strip(),
                    source=str(raw.get("source") or "example07").strip(),
                    charge=str(raw.get("charge") or "RESP").strip().upper(),
                    bonded=(str(raw.get("bonded") or "").strip() or None),
                    polyelectrolyte_mode=_csv_bool(raw.get("polyelectrolyte_mode"), default=False),
                )
            )
    return items
def _resolve_qm_spec(smiles: str) -> QMSpec | None:
    mol = yp.mol_from_smiles(smiles, coord=False)
    elements: list[str] = []
    seen: set[str] = set()
    formal_charge = 0
    for atom in mol.GetAtoms():
        formal_charge += int(atom.GetFormalCharge())
        symbol = str(atom.GetSymbol()).strip()
        if not symbol or symbol == "*" or symbol in seen:
            continue
        seen.add(symbol)
        elements.append(symbol)

    if mol.GetNumAtoms() == 1 and formal_charge != 0:
        return None

    method = "wb97m-d3bj"
    if formal_charge < 0:
        opt_candidates = ["def2-SVPD", "def2-SVP"]
        charge_candidates = ["def2-TZVPD", "def2-TZVPPD", "def2-TZVP"]
        reason = "anion diffuse-first"
    else:
        opt_candidates = ["def2-SVP"]
        charge_candidates = ["def2-TZVP"]
        reason = "neutral default"

    opt_basis = _pick_first_available_basis(opt_candidates, elements=elements)
    charge_basis = _pick_first_available_basis(charge_candidates, elements=elements)
    if formal_charge < 0 and (opt_basis != opt_candidates[0] or charge_basis != charge_candidates[0]):
        reason = f"{reason} -> fallback"

    return QMSpec(
        method=method,
        opt_basis=opt_basis,
        charge_basis=charge_basis,
        reason=reason,
    )


def run_one_species(
    spec: SpeciesSpec,
    *,
    db_dir: Path,
    job_wd: Path,
    psi4_omp: int,
    psi4_memory_mb: int,
) -> dict[str, Any]:
    species_wd = workdir(job_wd / spec.name, restart=False)
    mol = yp.mol_from_smiles(spec.smiles, name=spec.name)
    formal_charge = int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))
    charge_groups = detect_charged_groups(mol, detection="auto") if spec.polyelectrolyte_mode else {}
    qm_spec = _resolve_qm_spec(spec.smiles)
    ff = yp.get_ff(RESP_FF_NAME)
    ok = bool(
        ff.ff_assign(
            mol,
            charge=spec.charge,
            bonded=spec.bonded,
            bonded_work_dir=species_wd,
            bonded_omp_psi4=psi4_omp,
            bonded_memory_mb=psi4_memory_mb,
            total_charge=formal_charge,
            total_multiplicity=1,
            report=False,
            work_dir=species_wd,
            omp=psi4_omp,
            memory=psi4_memory_mb,
            opt_method=(qm_spec.method if qm_spec else "wb97m-d3bj"),
            charge_method=(qm_spec.method if qm_spec else "wb97m-d3bj"),
            opt_basis=(qm_spec.opt_basis if qm_spec else "def2-SVP"),
            charge_basis=(qm_spec.charge_basis if qm_spec else "def2-TZVP"),
            polyelectrolyte_mode=spec.polyelectrolyte_mode,
            polyelectrolyte_detection="auto",
        )
    )

    if not ok:
        raise RuntimeError(f"ff_assign failed for {spec.name} {spec.smiles}")

    db_ff = yp.get_ff(RESP_FF_NAME)
    record = db_ff.store_to_db(
        mol,
        smiles_or_psmiles=spec.smiles,
        name=spec.name,
        db_dir=db_dir,
        charge=spec.charge,
        polyelectrolyte_mode=spec.polyelectrolyte_mode,
        polyelectrolyte_detection="auto",
    )
    return {
        "name": spec.name,
        "smiles": spec.smiles,
        "kind": spec.kind,
        "source": spec.source,
        "charge": spec.charge,
        "bonded": spec.bonded,
        "polyelectrolyte_mode": spec.polyelectrolyte_mode,
        "formal_charge": formal_charge,
        "charge_group_count": len(charge_groups.get("groups") or []),
        "qm_method": (qm_spec.method if qm_spec else None),
        "qm_opt_basis": (qm_spec.opt_basis if qm_spec else None),
        "qm_charge_basis": (qm_spec.charge_basis if qm_spec else None),
        "qm_policy": (qm_spec.reason if qm_spec else None),
        "record_key": record.key,
        "psi4_omp": int(psi4_omp),
        "psi4_memory_mb": int(psi4_memory_mb),
    }


def main() -> int:
    restart_status = False
    set_run_options(restart=restart_status)

    psi4_omp = 36
    psi4_memory_mb = 20000

    doctor(print_report=True)
    ensure_initialized()

    db = MolDB()
    db_dir = Path(db.db_dir)
    example_wd = workdir(HERE / "work_dir", restart=restart_status)
    job_wd = example_wd.child("01_build_moldb")
    species = _read_species_csv(CATALOG_CSV)

    summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for spec in species:
        try:
            summary.append(
                run_one_species(
                    spec,
                    db_dir=db_dir,
                    job_wd=Path(job_wd),
                    psi4_omp=psi4_omp,
                    psi4_memory_mb=psi4_memory_mb,
                )
            )
            print(
                f"[OK] {spec.name:20s} charge={spec.charge:5s} bonded={spec.bonded or '-'}"
            )
        except Exception as exc:
            failures.append(
                {
                    "name": spec.name,
                    "smiles": spec.smiles,
                    "charge": spec.charge,
                    "bonded": spec.bonded,
                    "error": repr(exc),
                }
            )
            print(
                f"[FAIL] {spec.name:20s} charge={spec.charge:5s} bonded={spec.bonded or '-'} :: {exc}"
            )

    out = {
        "catalog_csv": str(CATALOG_CSV.resolve()),
        "db_dir": str(db_dir.resolve()),
        "work_root": str(job_wd.resolve()),
        "psi4_omp": psi4_omp,
        "psi4_memory_mb": psi4_memory_mb,
        "success_count": len(summary),
        "failure_count": len(failures),
        "success": summary,
        "failures": failures,
    }
    (job_wd / "build_moldb_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nMolDB directory: {db_dir}")
    print(f"Catalog CSV   : {CATALOG_CSV}")
    print(f"Psi4 OMP      : {psi4_omp}")
    print(f"Success       : {len(summary)}")
    print(f"Failures      : {len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
