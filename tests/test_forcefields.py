from rdkit import Chem

from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.oplsaa import OPLSAA, validate_oplsaa_rule_table


def _assign_gaff2_mod(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ff = GAFF2_mod()
    out = ff.ff_assign(mol, charge=None)
    assert out is not False
    return out


def test_gaff2_mod_assigns_silicon_atom_types():
    mol = _assign_gaff2_mod("C[Si](C)(C)O")
    atom_types = [atom.GetProp("ff_type") for atom in mol.GetAtoms()]

    assert "si" in atom_types
    assert "ci" in atom_types
    assert "oi" in atom_types


def test_gaff2_mod_assigns_disiloxane_bridge_oxygen():
    mol = _assign_gaff2_mod("C[Si](C)(C)O[Si](C)(C)C")
    atom_types = [atom.GetProp("ff_type") for atom in mol.GetAtoms()]

    assert "si" in atom_types
    assert "oss" in atom_types


def test_oplsaa_rule_table_matches_parameter_table():
    ff = OPLSAA()
    summary = validate_oplsaa_rule_table(ff.param.pt.keys())

    assert summary["rule_count"] > 600
    assert summary["unknown_types"] == []
    assert summary["placeholder_types"] == ["opls_xxx"]
    assert summary["placeholder_rule_count"] == 13


def test_oplsaa_assign_ptypes_from_external_rule_table():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    ff = OPLSAA()

    assert ff.assign_ptypes(mol, charge="opls")
    assert all(atom.HasProp("ff_btype") for atom in mol.GetAtoms())
    assert any(atom.GetProp("ff_type") == "opls_154" for atom in mol.GetAtoms())
