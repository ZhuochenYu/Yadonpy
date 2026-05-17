from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.core.topology import Cell
from yadonpy.io.gromacs_molecule import (
    itp_has_invalid_bond_parameters,
    itp_has_legacy_drih_ax6_angles,
    write_gromacs_single_molecule_topology,
)
from yadonpy.io.gromacs_system import export_system_from_cell_meta


ROOT = Path(__file__).resolve().parents[1]
REPO_DB_DIR = ROOT / "moldb"


def _section_rows(text: str, section: str) -> list[list[str]]:
    rows: list[list[str]] = []
    current: str | None = None
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            current = line[1 : line.index("]")].strip().lower()
            continue
        if current == section:
            rows.append(line.split())
    return rows


def test_pf6_drih_exports_harmonic_angles_and_bond_bond_cross_terms(tmp_path: Path):
    ff = GAFF2_mod()
    pf6 = ff.mol_rdkit(
        "F[P-](F)(F)(F)(F)F",
        db_dir=REPO_DB_DIR,
        charge="RESP",
        require_ready=True,
        prefer_db=True,
    )
    pf6 = ff.ff_assign(pf6, bonded="DRIH", report=False)

    _, itp_path, _ = write_gromacs_single_molecule_topology(pf6, tmp_path, mol_name="PF6")
    text = itp_path.read_text(encoding="utf-8")
    angles = _section_rows(text, "angles")
    harmonic = [row for row in angles if int(float(row[3])) == 1]
    cross = [row for row in angles if int(float(row[3])) == 3]
    cis = [row for row in harmonic if abs(float(row[4]) - 90.0) < 1.0e-6]
    trans = [row for row in harmonic if abs(float(row[4]) - 180.0) < 1.0e-6]

    assert len(harmonic) == 15
    assert len(cis) == 12
    assert len(trans) == 3
    assert len(cross) == 15
    assert not any(int(float(row[3])) == 9 for row in angles)
    assert {round(float(row[5]), 4) for row in cis} == {524.2033, 539.1223}
    assert all(abs(float(row[5]) - 249.0480) < 1.0e-4 for row in trans)
    assert all(abs(float(row[4]) - 0.164607) < 1.0e-6 for row in cross)
    assert all(abs(float(row[5]) - 0.164607) < 1.0e-6 for row in cross)
    assert sum(1 for row in cross if float(row[6]) < 0.0) == 3
    assert sum(1 for row in cross if float(row[6]) > 0.0) == 12
    assert not itp_has_legacy_drih_ax6_angles(text)
    assert not itp_has_invalid_bond_parameters(itp_path)


def test_legacy_drih_pf6_topology_without_cross_terms_is_invalid():
    legacy = """; patched by yadonpy DRIH (bond+angle)
[ moleculetype ]
PF6 3
[ atoms ]
1 f 1 PF6 F1 1 -0.3 18.998
2 p5 1 PF6 P2 2 1.0 30.974
3 f 1 PF6 F3 3 -0.3 18.998
4 f 1 PF6 F4 4 -0.3 18.998
5 f 1 PF6 F5 5 -0.3 18.998
6 f 1 PF6 F6 6 -0.3 18.998
7 f 1 PF6 F7 7 -0.3 18.998
[ bonds ]
1 2 1 0.164 300000
2 3 1 0.164 300000
2 4 1 0.164 300000
2 5 1 0.164 300000
2 6 1 0.164 300000
2 7 1 0.164 300000
[ angles ]
1 2 4 1 90.0 3000
1 2 5 1 90.0 3000
1 2 6 1 90.0 3000
1 2 7 1 90.0 3000
3 2 4 1 90.0 3000
3 2 5 1 90.0 3000
3 2 6 1 90.0 3000
3 2 7 1 90.0 3000
4 2 6 1 90.0 3000
4 2 7 1 90.0 3000
5 2 6 1 90.0 3000
5 2 7 1 90.0 3000
"""

    assert itp_has_legacy_drih_ax6_angles(legacy)


def test_pf6_drih_survives_fragment_system_export_with_charge_scaling(tmp_path: Path):
    ff = GAFF2_mod()
    pf6 = ff.mol_rdkit(
        "F[P-](F)(F)(F)(F)F",
        db_dir=REPO_DB_DIR,
        charge="RESP",
        require_ready=True,
        prefer_db=True,
    )
    pf6 = ff.ff_assign(pf6, bonded="DRIH", report=False)
    li = MERZ().ff_assign(MERZ().mol("[Li+]", name="Li"), report=False)

    cell = Chem.CombineMols(li, pf6)
    conf = Chem.Conformer(cell.GetNumAtoms())
    conf.SetAtomPosition(0, (2.0, 2.0, 2.0))
    pf6_conf = pf6.GetConformer()
    for idx in range(pf6.GetNumAtoms()):
        pos = pf6_conf.GetAtomPosition(idx)
        conf.SetAtomPosition(1 + idx, (pos.x + 12.0, pos.y + 12.0, pos.z + 12.0))
    cell.RemoveAllConformers()
    cell.AddConformer(conf, assignId=True)
    setattr(cell, "cell", Cell(30.0, 0.0, 30.0, 0.0, 30.0, 0.0))
    cell.SetProp(
        "_yadonpy_cell_meta",
        json.dumps(
            {
                "density_g_cm3": 0.1,
                "species": [
                    {
                        "smiles": "[Li+]",
                        "n": 1,
                        "natoms": 1,
                        "charge_scale": 0.7,
                        "name": "Li_0000",
                        "ff_name": "merz",
                        "charge_method": "RESP",
                        "force_write_from_fragment": True,
                        "fragment_index": 0,
                    },
                    {
                        "smiles": "F[P-](F)(F)(F)(F)F",
                        "n": 1,
                        "natoms": 7,
                        "charge_scale": 0.7,
                        "name": "PF6_0001",
                        "ff_name": "gaff2_mod",
                        "charge_method": "RESP",
                        "force_write_from_fragment": True,
                        "fragment_index": 1,
                        "bonded_requested": "drih",
                        "bonded_method": "DRIH",
                        "bonded_explicit": True,
                        "bonded_signature": "drih",
                        "bonded_itp": pf6.GetProp("_yadonpy_bonded_itp"),
                        "bonded_json": pf6.GetProp("_yadonpy_bonded_json"),
                    },
                ],
            }
        ),
    )

    out = export_system_from_cell_meta(
        cell_mol=cell,
        out_dir=tmp_path / "system",
        ff_name="gaff2_mod",
        charge_method="RESP",
        write_system_mol2=False,
    )
    text = (out.molecules_dir / "PF6_0001" / "PF6_0001.itp").read_text(encoding="utf-8")
    angles = _section_rows(text, "angles")
    harmonic = [row for row in angles if int(float(row[3])) == 1]
    cross = [row for row in angles if int(float(row[3])) == 3]

    assert "-0.26281010" in text
    assert "92.220" not in text
    assert "773.26" not in text
    assert len(harmonic) == 15
    assert len(cross) == 15
    assert sum(1 for row in harmonic if abs(float(row[4]) - 180.0) < 1.0e-6) == 3
    assert not itp_has_legacy_drih_ax6_angles(text)
