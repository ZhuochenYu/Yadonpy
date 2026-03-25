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


def test_gaff2_mod_assigns_silicon_hydride_bonds():
    mol = _assign_gaff2_mod("[SiH4]")

    atom_types = [atom.GetProp("ff_type") for atom in mol.GetAtoms()]
    bond_types = [bond.GetProp("ff_type") for bond in mol.GetBonds()]
    angle_types = {angle.ff.type for angle in mol.angles.values()}

    assert atom_types.count("si") == 1
    assert atom_types.count("hi") == 4
    assert len(bond_types) == 4
    assert set(bond_types) == {"si,hi"}
    assert angle_types == {"hi,si,hi"}


def test_gaff2_mod_assigns_methylsilane_without_missing_si_hi_bonds():
    mol = _assign_gaff2_mod("C[SiH3]")

    silicon_h_bonds = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom().GetProp("ff_type")
        b = bond.GetEndAtom().GetProp("ff_type")
        if {a, b} == {"si", "hi"}:
            silicon_h_bonds.append(bond.GetProp("ff_type"))

    assert len(silicon_h_bonds) == 3
    assert set(silicon_h_bonds) == {"si,hi"}
    assert {angle.ff.type for angle in mol.angles.values()} >= {"ci,si,hi", "hi,si,hi"}


def test_gaff2_mod_assigns_disiloxane_hydride_terms_explicitly():
    mol = _assign_gaff2_mod("[SiH3]O[SiH3]")

    assert {bond.GetProp("ff_type") for bond in mol.GetBonds()} == {"si,hi", "si,oss"}
    assert {angle.ff.type for angle in mol.angles.values()} == {"oss,si,hi", "hi,si,hi", "si,oss,si"}
    assert {dih.ff.type for dih in mol.dihedrals.values()} == {"hi,si,oss,si"}


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
