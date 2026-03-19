from __future__ import annotations

# Example 10: Batch-build initial geometries + RESP charges into a shared molecule DB from a CSV.
#
# The MolDB stores only:
#   - canonical smiles/psmiles
#   - a best initial 3D geometry (mol2)
#   - charges (RESP, etc.)
#
# Force-field assignment is NOT stored here (it is fast and is done on demand in later workflows).

from pathlib import Path
import sys

import pandas as pd

from yadonpy.ff.gaff2 import GAFF2
from yadonpy.moldb import MolDB
from yadonpy import qm

HERE = Path(__file__).resolve().parent

# Allow overriding the input CSV from command line:
#   python run_batch_build_db.py template.csv
csv_file = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (HERE / "template.csv")

# For portability, we use a project-local DB directory:
db_dir = HERE / "moldb"
db = MolDB(db_dir)

# Human-readable index mapping DB key <-> name <-> smiles for this run.
index_csv = HERE / "index.csv"
try:
    _idx_df = pd.read_csv(index_csv) if index_csv.exists() else pd.DataFrame()
    _index_map = {str(r["key"]): dict(r) for _, r in _idx_df.iterrows()} if not _idx_df.empty else {}
except Exception:
    _index_map = {}

def _update_index(*, rec, smiles: str, name: str):
    _index_map[str(rec.key)] = {
        "key": str(rec.key),
        "name": str(name),
        "smiles": str(smiles),
        "kind": str(getattr(rec, "kind", "")),
        "canonical": str(getattr(rec, "canonical", "")),
        "ready": bool(getattr(rec, "ready", False)),
        "charge_method": getattr(rec, "charge_method", None),
    }
    # Write on every update so partial progress is preserved even if the run crashes.
    pd.DataFrame(list(_index_map.values())).to_csv(index_csv, index=False)


# Per-molecule scratch directory for QM jobs
work_root = HERE / "work_dir"
work_root.mkdir(parents=True, exist_ok=True)

# Resources (adjust as needed)
mpi = 1
omp = 16
omp_psi4 = 16
mem = 16000  # MB

ff = GAFF2()

df = pd.read_csv(csv_file)
for _, row in df.iterrows():
    name = str(row["name"]).strip()
    smiles = str(row["smiles"]).strip()

    print(f"=== {name}: {smiles} ===")

    # If DB already has charges, reuse and skip expensive steps
    try:
        mol_ready, rec = db.load_mol(smiles, require_ready=True)
        print(f"  -> found in DB (ready), key={rec.key}; skipping QM.")
        _update_index(rec=rec, smiles=smiles, name=name)
        continue
    except Exception:
        pass

    # Build (or load) initial geometry (no charges yet)
    mol = GAFF2.mol(smiles, name=name, db_dir=db_dir, prefer_db=True)

    key = mol.GetProp("_YADONPY_KEY") if mol.HasProp("_YADONPY_KEY") else None
    rec = db.load_record(key) if key else None

    # Follow the same workflow as Example 01 for conformers + charges:
    work_dir = work_root / name
    work_dir.mkdir(parents=True, exist_ok=True)

    mol, energy = qm.conformation_search(
        mol, ff=ff, work_dir=work_dir,
        psi4_omp=omp_psi4, mpi=mpi, omp=omp,
        memory=mem, log_name=None
    )
    qm.assign_charges(
        mol, charge="RESP", opt=False, work_dir=work_dir,
        omp=omp_psi4, memory=mem, log_name=None
    )

    # Persist best geometry + charges into DB
    GAFF2.store_to_db(mol, smiles_or_psmiles=smiles, name=name, db_dir=db_dir, charge_method="RESP")
    rec2 = db.load_record(mol.GetProp("_YADONPY_KEY"))
    if rec2 is not None:
        _update_index(rec=rec2, smiles=smiles, name=name)
    print(f"  -> stored to DB.")

print(f"Done. DB is at: {db_dir}")
