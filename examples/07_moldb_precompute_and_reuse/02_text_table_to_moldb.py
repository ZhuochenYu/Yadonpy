from __future__ import annotations

"""Example 07 / Step 2: pasted text table -> CSV -> MolDB.

This keeps the quick text-import path in the same merged Example 07 workflow as
the curated electrolyte catalog builder.
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
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n") + "\n"
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = [str(h).strip().lower() for h in (reader.fieldnames or [])]
    if "name" not in fieldnames or "smiles" not in fieldnames:
        raise ValueError(f"Header must include Name/SMILES (got: {reader.fieldnames})")

    rows = []
    for row in reader:
        normalized = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in row.items()}
        if not normalized.get("name") or not normalized.get("smiles"):
            continue
        rows.append(normalized)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    template_csv = HERE / "template.csv"
    write_template_csv(RAW, template_csv)
    print(f"Wrote: {template_csv}")

    db = MolDB()
    print(f"MolDB directory: {db.db_dir}")

    work_root = HERE / "work_dir" / "02_text_table_to_moldb"
    work_root.mkdir(parents=True, exist_ok=True)

    omp_psi4 = 64
    mem_mb = 20000

    db.read_calc_temp = str(template_csv)
    db.autocalculate(work_dir=work_root, omp=omp_psi4, mem=mem_mb, add_to_moldb=True)
    print("All done.")
