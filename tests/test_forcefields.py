import json

from rdkit import Chem

from yadonpy.ff.report import render_ff_assignment_report
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.oplsaa import OPLSAA, validate_oplsaa_rule_table
from yadonpy.core.resources import ff_data_path


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


def test_gaff2_mod_records_qm_backed_si_h_sources():
    with open(ff_data_path("ff_dat", "gaff2_mod.json"), "r", encoding="utf-8") as fh:
        data = json.load(fh)

    def _by_tag(section: str, tag: str):
        for rec in data[section]:
            if rec.get("tag") == tag:
                return rec
        raise AssertionError(f"missing {section}:{tag}")

    si_hi = _by_tag("bond_types", "si,hi")
    hi_si_hi = _by_tag("angle_types", "hi,si,hi")
    ci_si_hi = _by_tag("angle_types", "ci,si,hi")
    oi_si_hi = _by_tag("angle_types", "oi,si,hi")
    oss_si_hi = _by_tag("angle_types", "oss,si,hi")
    torsion = _by_tag("dihedral_types", "hi,si,oss,si")

    assert "Psi4 modified Seminario" in si_hi["source"]
    assert abs(float(si_hi["r0"]) - 0.148585) < 1.0e-6
    assert abs(float(si_hi["k"]) - 166875.184453) < 1.0e-6
    assert "Psi4 modified Seminario" in hi_si_hi["source"]
    assert abs(float(hi_si_hi["theta0"]) - 109.122028) < 1.0e-6
    assert abs(float(hi_si_hi["k"]) - 388.318040) < 1.0e-6
    assert "C[SiH3]" in ci_si_hi["source"]
    assert "O[SiH3]" in oi_si_hi["source"]
    assert "[SiH3]O[SiH3]" in oss_si_hi["source"]
    assert "surrogate" in torsion["source"]


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


def test_oplsaa2024_integrates_new_particles_and_rules():
    with open(ff_data_path("ff_dat", "oplsaa.json"), "r", encoding="utf-8") as fh:
        ff_data = json.load(fh)
    with open(ff_data_path("ff_dat", "oplsaa_rules.json"), "r", encoding="utf-8") as fh:
        rule_data = json.load(fh)

    particle_tags = {row["tag"] for row in ff_data["particle_types"]}
    bond_tags = {row["tag"] for row in ff_data["bond_types"]}
    angle_tags = {row["tag"] for row in ff_data["angle_types"]}
    dihedral_tags = {row["tag"] for row in ff_data["dihedral_types"]}

    assert {"opls_1060", "opls_1078", "opls_1106", "opls_1114", "opls_1160"} <= particle_tags
    assert {"Si,CT", "Si,H~", "Si,OH", "Si,OS", "Si,Si"} <= bond_tags
    assert {"CT,Si,CT", "H~,Si,H~", "CT,Si,OH", "CT,Si,OS"} <= angle_tags
    assert {"CT,Si,OS,CT", "H~,Si,OS,CT", "HC,CT,Si,OS", "CT,Si,Si,CT"} <= dihedral_tags

    by_smarts = {row["smarts"]: row for row in rule_data}
    assert by_smarts["[Li+]"]["opls"] == "opls_1106"
    assert by_smarts["[Na+]"]["opls"] == "opls_1107"
    assert by_smarts["[F-]"]["opls"] == "opls_1100"
    assert by_smarts["[Cl-]"]["opls"] == "opls_1101"
    assert by_smarts["[SiH4]"]["opls"] == "opls_1083"
    assert by_smarts["[O;H1][Si]"]["opls"] == "opls_1073"
    assert by_smarts["[C](=[O])=[O]"]["opls"] == "opls_1160"


def test_oplsaa2024_assigns_silane_co2_and_updated_ions():
    ff = OPLSAA()

    silane = Chem.AddHs(Chem.MolFromSmiles("[SiH4]"))
    assert ff.assign_ptypes(silane, charge="opls")
    assert ff.assign_btypes(silane)
    assert ff.assign_atypes(silane)
    assert ff.assign_dtypes(silane)
    assert [atom.GetProp("ff_type") for atom in silane.GetAtoms()].count("opls_1083") == 1
    assert [atom.GetProp("ff_type") for atom in silane.GetAtoms()].count("opls_1064") == 4
    assert {bond.GetProp("ff_type") for bond in silane.GetBonds()} == {"Si,H~"}
    assert {angle.ff.type for angle in silane.angles.values()} == {"H~,Si,H~"}

    co2 = Chem.MolFromSmiles("O=C=O")
    assert ff.assign_ptypes(co2, charge="opls")
    assert [atom.GetProp("ff_type") for atom in co2.GetAtoms()] == ["opls_1161", "opls_1160", "opls_1161"]

    li = Chem.MolFromSmiles("[Li+]")
    assert ff.assign_ptypes(li, charge="opls")
    assert li.GetAtomWithIdx(0).GetProp("ff_type") == "opls_1106"


def test_render_ff_assignment_report_summarizes_charged_side_groups():
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    mol.SetIntProp("num_units", 4)
    mol.SetProp("_yadonpy_smiles", "*CC(=O)[O-]*")
    charges = [0.0, 0.2, -0.6, -0.6]
    for atom, charge in zip(mol.GetAtoms(), charges):
        atom.SetProp("ff_type", "test")
        atom.SetProp("ff_btype", "test")
        atom.SetDoubleProp("AtomicCharge", charge)

    report = render_ff_assignment_report(mol)

    assert "Charge check:" in report
    assert "Charged side groups:" in report
    assert "carboxylate" in report
    assert "CO2" in report
    assert "total_assigned_charge: -1.00000" in report


def test_render_ff_assignment_report_only_checks_total_charge_for_neutral_molecules():
    mol = Chem.MolFromSmiles("CCO")
    charges = [0.05, -0.10, 0.05]
    for atom, charge in zip(mol.GetAtoms(), charges):
        atom.SetProp("ff_type", "test")
        atom.SetProp("ff_btype", "test")
        atom.SetDoubleProp("AtomicCharge", charge)

    report = render_ff_assignment_report(mol)

    assert "Charge check:" in report
    assert "total_assigned_charge: 0.00000" in report
    assert "Charged side groups:" not in report
