from __future__ import annotations

"""Refresh CMC anionic glucose monomers in both default and repo MolDB.

Targets:
  - glucose_2
  - glucose_3
  - glucose_6

All three species are recomputed with RESP charges under
``polyelectrolyte_mode=True`` and then written to:
  1. the user's default MolDB (typically ``~/.yadonpy/moldb``)
  2. the repository-tracked ``moldb/`` directory
"""

from dataclasses import dataclass
import os
from pathlib import Path

import yadonpy as yp
from yadonpy.core import workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.core.polyelectrolyte import detect_charged_groups
from yadonpy.diagnostics import doctor
from yadonpy.moldb import MolDB
from yadonpy.runtime import set_run_options


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    smiles: str


SPECIES: tuple[SpeciesSpec, ...] = (
    SpeciesSpec("glucose_2", "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"),
    SpeciesSpec("glucose_3", "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"),
    SpeciesSpec("glucose_6", "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"),
)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    return int(raw)


def _env_tokens(name: str) -> set[str]:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return set()
    return {tok.strip() for tok in raw.split(",") if tok.strip()}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "t", "yes", "y", "on"}


def _env_str(name: str, default: str) -> str:
    raw = str(os.environ.get(name, "")).strip()
    return raw or str(default)


def _formal_charge(mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def _is_ready(db: MolDB, smiles: str) -> bool:
    try:
        db.load_mol(
            smiles,
            require_ready=True,
            charge="RESP",
            polyelectrolyte_mode=True,
            polyelectrolyte_detection="auto",
        )
        return True
    except Exception:
        return False


def _refresh_one(spec: SpeciesSpec, *, default_db: MolDB, repo_db: MolDB, job_wd: Path) -> None:
    species_wd = workdir(job_wd / spec.name, restart=False)
    source = "fresh-smiles"
    need_opt = True
    mol = None
    for db, label in ((default_db, "default"), (repo_db, "repo")):
        try:
            mol, _rec = db.load_mol(
                spec.smiles,
                require_ready=False,
                charge="RESP",
                polyelectrolyte_mode=True,
                polyelectrolyte_detection="auto",
            )
            source = f"{label}-db-geometry"
            need_opt = False
            break
        except Exception:
            continue

    if mol is None:
        mol = yp.mol_from_smiles(spec.smiles, name=spec.name)

    charge_groups = detect_charged_groups(mol, detection="auto")
    formal_charge = _formal_charge(mol)

    print(
        f"[RUN] {spec.name:10s} formal_charge={formal_charge:+d} "
        f"localized_groups={len(charge_groups.get('groups') or [])} "
        f"source={source} opt={need_opt}"
    )

    ok = yp.assign_charges(
        mol,
        charge="RESP",
        opt=need_opt,
        work_dir=species_wd,
        log_name=spec.name,
        omp=_env_int("YADONPY_PSI4_OMP", 20),
        memory=_env_int("YADONPY_PSI4_MEMORY_MB", 20000),
        opt_method=_env_str("YADONPY_OPT_METHOD", "wb97m-d3bj"),
        charge_method=_env_str("YADONPY_CHARGE_METHOD", "wb97m-d3bj"),
        opt_basis=_env_str("YADONPY_OPT_BASIS", "def2-SVPD"),
        charge_basis=_env_str("YADONPY_CHARGE_BASIS", "def2-TZVPD"),
        total_charge=formal_charge,
        total_multiplicity=1,
        polyelectrolyte_mode=True,
        polyelectrolyte_detection="auto",
        bonded_params="ff_assigned",
    )
    if not ok:
        raise RuntimeError(f"assign_charges failed for {spec.name}")

    for db, label in ((default_db, "default"), (repo_db, "repo")):
        rec = db.update_from_mol(
            mol,
            smiles_or_psmiles=spec.smiles,
            name=spec.name,
            charge="RESP",
            polyelectrolyte_mode=True,
            polyelectrolyte_detection="auto",
        )
        loaded, loaded_rec = db.load_mol(
            spec.smiles,
            require_ready=True,
            charge="RESP",
            polyelectrolyte_mode=True,
            polyelectrolyte_detection="auto",
        )
        print(
            f"  [OK:{label}] key={rec.key} atoms={loaded.GetNumAtoms()} "
            f"ready_name={loaded_rec.name}"
        )


def main() -> int:
    set_run_options(restart=False)
    doctor(print_report=True)
    ensure_initialized()

    default_db = MolDB()
    repo_db = MolDB(REPO_ROOT / "moldb")
    job_wd = workdir(HERE / "work_dir" / "03_refresh_cmc_glucose_polyelectrolytes", restart=False)

    print(f"[DB] default = {default_db.db_dir}")
    print(f"[DB] repo    = {repo_db.db_dir}")
    print(
        "[QM] psi4_omp="
        f"{_env_int('YADONPY_PSI4_OMP', 20)} "
        f"psi4_memory_mb={_env_int('YADONPY_PSI4_MEMORY_MB', 20000)} "
        f"opt={_env_str('YADONPY_OPT_METHOD', 'wb97m-d3bj')}/"
        f"{_env_str('YADONPY_OPT_BASIS', 'def2-SVPD')} "
        f"charge={_env_str('YADONPY_CHARGE_METHOD', 'wb97m-d3bj')}/"
        f"{_env_str('YADONPY_CHARGE_BASIS', 'def2-TZVPD')}"
    )

    only = _env_tokens("YADONPY_ONLY")
    force = _env_flag("YADONPY_FORCE", default=False)

    for spec in SPECIES:
        if only and spec.name not in only:
            print(f"[SKIP] {spec.name:10s} filtered by YADONPY_ONLY")
            continue
        default_ready = _is_ready(default_db, spec.smiles)
        repo_ready = _is_ready(repo_db, spec.smiles)
        if default_ready and repo_ready and not force:
            print(f"[SKIP] {spec.name:10s} already ready in both MolDB locations")
            continue
        _refresh_one(spec, default_db=default_db, repo_db=repo_db, job_wd=Path(job_wd))

    print("[DONE] Refreshed glucose_2 / glucose_3 / glucose_6 in both MolDB locations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
