from __future__ import annotations

from pathlib import Path

from yadonpy.io.mol2 import read_mol2_with_charges


def test_read_mol2_with_charges_falls_back_for_gaff_style_atom_types(tmp_path: Path):
    mol2 = tmp_path / "gaffish.mol2"
    mol2.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "gaffish",
                "2 1 0 0 0",
                "SMALL",
                "USER_CHARGES",
                "",
                "@<TRIPOS>ATOM",
                "1 H1 0.0000 0.0000 0.0000 ho 1 RES 0.300000",
                "2 O1 0.0000 0.0000 0.9600 os 1 RES -0.300000",
                "@<TRIPOS>BOND",
                "1 1 2 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mol = read_mol2_with_charges(mol2, sanitize=False, removeHs=False)

    assert mol.GetNumAtoms() == 2
    assert mol.GetAtomWithIdx(0).GetSymbol() == "H"
    assert mol.GetAtomWithIdx(1).GetSymbol() == "O"
    assert mol.GetAtomWithIdx(0).GetDoubleProp("AtomicCharge") == 0.3
    assert mol.GetAtomWithIdx(1).GetDoubleProp("RESP") == -0.3
