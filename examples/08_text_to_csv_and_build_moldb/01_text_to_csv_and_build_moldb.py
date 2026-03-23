from __future__ import annotations

"""Example 08: Convert a pasted text table -> template.csv, then build MolDB.

This script intentionally keeps *everything* in one place.

Notes
-----
- MolDB default directory: ~/.yadonpy/moldb (or $YADONPY_MOLDB)
- Output scratch: ./work_dir/01_build_moldb/
"""

import csv
import io
from pathlib import Path

from yadonpy.moldb import MolDB

HERE = Path(__file__).resolve().parent


RAW = """Name,SMILES,opt,confsearch,charge_method,basis_set,method
EC,O=C1OCCO1,0,1,RESP,Default,Default
PC,CC1COC(=O)O1,0,1,RESP,Default,Default
DEC,CCOC(=O)OCC,0,1,RESP,Default,Default
EMC,CCOC(=O)OC,0,1,RESP,Default,Default
DMC,COC(=O)OC,0,1,RESP,Default,Default
DME,COCCOC,0,1,RESP,Default,Default
VC,O=C1OC=CO1,0,1,RESP,Default,Default
DTD,O=S1(=O)OC=CO1,0,1,RESP,Default,Default
FEC,O=C1OCC(F)O1,0,1,RESP,Default,Default
Diglyme,COCCOCCOC,0,1,RESP,Default,Default
Triglyme,COCCOCCOCCOC,0,1,RESP,Default,Default
Tetraglyme,COCCOCCOCCOCCOC,0,1,RESP,Default,Default
DOL,C1COCO1,0,1,RESP,Default,Default
THF,C1CCOC1,0,1,RESP,Default,Default
Dioxane,C1COCCO1,0,1,RESP,Default,Default
CPME,COC1CCCC1,0,1,RESP,Default,Default
TTE,FC(F)C(F)(F)COC(F)(F)C(F)F,0,1,RESP,Default,Default
glucose_0,*OC1OC(CO)C(*)C(O)C1O,0,1,RESP,Default,Default
glucose_2,*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-],0,1,RESP,Default,Default
glucose_3,*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O,0,1,RESP,Default,Default
glucose_6,*OC1OC(COCC(=O)[O-])C(*)C(O)C1O,0,1,RESP,Default,Default
glucose_23,*OC1OC(CO)C(*)C(OCC(=O)[O-])C1OCC(=O)[O-],0,1,RESP,Default,Default
glucose_26,*OC1OC(COCC(=O)[O-])C(*)C(O)C1OCC(=O)[O-],0,1,RESP,Default,Default
glucose_36,*OC1OC(COCC(=O)[O-])C(*)C(OCC(=O)[O-])C1O,0,1,RESP,Default,Default
glucose_236,*OC1OC(COCC(=O)[O-])C(*)C(OCC(=O)[O-])C1OCC(=O)[O-],0,1,RESP,Default,Default
"""


def write_template_csv(text: str, out_csv: Path) -> None:
    """Normalize headers to lower-case and write to out_csv."""
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n") + "\n"
    reader = csv.DictReader(io.StringIO(text))

    # Normalize header
    fieldnames = [str(h).strip().lower() for h in (reader.fieldnames or [])]
    # Expected by MolDB.autocalculate
    if "name" not in fieldnames or "smiles" not in fieldnames:
        raise ValueError(f"Header must include Name/SMILES (got: {reader.fieldnames})")

    rows = []
    for r in reader:
        rr = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in r.items()}
        if not rr.get("name") or not rr.get("smiles"):
            continue
        rows.append(rr)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


if __name__ == "__main__":
    template_csv = HERE / "template.csv"
    write_template_csv(RAW, template_csv)
    print(f"Wrote: {template_csv}")

    db = MolDB()
    print(f"MolDB directory: {db.db_dir}")

    work_root = HERE / "work_dir" / "01_build_moldb"
    work_root.mkdir(parents=True, exist_ok=True)

    # Resource knobs (Psi4)
    omp_psi4 = 64
    mem_mb = 20000

    db.read_calc_temp = str(template_csv)
    # Explicitly add results into ~/.yadonpy/moldb
    db.autocalculate(work_dir=work_root, omp=omp_psi4, mem=mem_mb, add_to_moldb=True)

    print("All done.")
