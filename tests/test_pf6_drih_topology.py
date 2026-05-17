from __future__ import annotations

from pathlib import Path

from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.io.gromacs_molecule import (
    itp_has_invalid_bond_parameters,
    itp_has_legacy_drih_ax6_angles,
    write_gromacs_single_molecule_topology,
)


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
