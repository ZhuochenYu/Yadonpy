import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from yadonpy.core import poly, workdir
from yadonpy.moldb import MolDB
from yadonpy.ff.report import render_ff_assignment_report
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.oplsaa import OPLSAA, validate_oplsaa_rule_table
from yadonpy.ff.oplsaa_reference import audit_oplsaa_reference
from yadonpy.core.resources import ff_data_path
from yadonpy.io.gmx import write_gmx
from yadonpy.io.gromacs_top import defaults_for_ff_name


def _assign_gaff2_mod(smiles: str):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ff = GAFF2_mod()
    out = ff.ff_assign(mol, charge=None)
    assert out is not False
    return out


_OPLSAA_DONOR_H_TYPES = {"opls_155", "opls_170", "opls_172", "opls_188"}


def _assert_nonzero_opls_lj_for_materialized_types(mol):
    missing = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("ff_type"):
            missing.append((atom.GetIdx(), atom.GetSymbol(), "missing_ff_type"))
            continue
        ff_type = atom.GetProp("ff_type")
        sigma = atom.GetDoubleProp("ff_sigma") if atom.HasProp("ff_sigma") else None
        epsilon = atom.GetDoubleProp("ff_epsilon") if atom.HasProp("ff_epsilon") else None
        if sigma is None or epsilon is None or float(sigma) <= 0.0 or float(epsilon) <= 0.0:
            missing.append((atom.GetIdx(), atom.GetSymbol(), ff_type, sigma, epsilon))
    assert missing == [], f"OPLS-AA atoms lost LJ parameters: {missing[:20]}"


def _assert_hydroxyl_donor_angles_present(mol):
    angles = getattr(mol, "angles", {}) or {}
    missing = []
    for atom in mol.GetAtoms():
        ff_type = atom.GetProp("ff_type")
        if ff_type not in _OPLSAA_DONOR_H_TYPES:
            continue
        idx = int(atom.GetIdx())
        neighbors = list(atom.GetNeighbors())
        assert len(neighbors) == 1
        donor = neighbors[0]
        donor_idx = int(donor.GetIdx())
        donor_angles = []
        for angle in angles.values():
            triple = (int(angle.a), int(angle.b), int(angle.c))
            if triple[1] == donor_idx and idx in (triple[0], triple[2]):
                donor_angles.append(triple)
        if not donor_angles:
            missing.append((idx, ff_type, donor_idx, donor.GetProp("ff_type")))
    assert missing == [], f"Missing O-H angle terms for donor hydrogens: {missing[:20]}"


def test_oplsaa_defaults_use_jorgensen_rule():
    defaults = defaults_for_ff_name("oplsaa")
    assert defaults.comb_rule == 3
    assert defaults.gen_pairs == "yes"
    assert defaults.fudge_lj == pytest.approx(0.5)
    assert defaults.fudge_qq == pytest.approx(0.5)


def test_write_gmx_uses_oplsaa_defaults_block(tmp_path: Path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0DE) == 0
    ff = OPLSAA()
    out = ff.ff_assign(mol, charge="opls", report=False)
    assert out is not False
    _, _, top_path = write_gmx(mol=out, out_dir=tmp_path, mol_name="ETH")
    top_txt = top_path.read_text(encoding="utf-8")
    assert "1  3  yes  0.500000  0.5000000000" in top_txt


def test_oplsaa_materializes_small_lj_floor_for_hydroxyl_donor_hydrogens():
    ff = OPLSAA()
    mol = ff.ff_assign(Chem.AddHs(Chem.MolFromSmiles("OCCO")), charge="opls", report=False)
    assert mol is not False
    donor_h = []
    for atom in mol.GetAtoms():
        ff_type = atom.GetProp("ff_type")
        if ff_type in _OPLSAA_DONOR_H_TYPES:
            donor_h.append(
                (
                    ff_type,
                    atom.GetDoubleProp("ff_sigma"),
                    atom.GetDoubleProp("ff_epsilon"),
                )
            )
    assert donor_h, "Expected at least one OPLS-AA hydroxyl donor hydrogen"
    assert all(sigma > 0.0 and epsilon > 0.0 for _, sigma, epsilon in donor_h)
    _assert_hydroxyl_donor_angles_present(mol)


def test_oplsaa_assigns_source_backed_planar_improper_for_acetate():
    ff = OPLSAA()
    mol = ff.ff_assign(Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]")), charge="opls", report=False)

    assert mol is not False
    impropers = getattr(mol, "impropers", {}) or {}
    assert len(impropers) >= 1
    assert any(imp.ff.type == "improper_O_C_X_Y" for imp in impropers.values())
    assert any(getattr(imp.ff, "source", "") == "gromacs_oplsaa" for imp in impropers.values())


def test_oplsaa_assigns_amide_angles_and_planar_impropers():
    ff = OPLSAA()
    mol = ff.ff_assign(Chem.AddHs(Chem.MolFromSmiles("CC(=O)N")), charge="opls", report=False)

    assert mol is not False
    assert len(getattr(mol, "angles", {}) or {}) > 0
    assert len(getattr(mol, "dihedrals", {}) or {}) > 0
    impropers = getattr(mol, "impropers", {}) or {}
    assert len(impropers) >= 2
    assert {imp.ff.type for imp in impropers.values()} >= {"improper_O_C_X_Y", "improper_Z_N_X_Y"}


def test_oplsaa_assigns_guanidinium_angles_and_planar_impropers():
    ff = OPLSAA()
    mol = ff.ff_assign(Chem.AddHs(Chem.MolFromSmiles("NC(=[NH2+])N")), charge="opls", report=False)

    assert mol is not False
    assert len(getattr(mol, "angles", {}) or {}) > 0
    impropers = getattr(mol, "impropers", {}) or {}
    assert len(impropers) >= 3
    assert {imp.ff.type for imp in impropers.values()} >= {"improper_O_C_X_Y", "improper_Z_N_X_Y"}


def test_write_gmx_exports_oplsaa_impropers_as_gromacs_dihedral_funct4(tmp_path: Path):
    ff = OPLSAA()
    mol = ff.ff_assign(Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]")), charge="opls", report=False)
    assert mol is not False

    _, itp_path, _ = write_gmx(mol=mol, out_dir=tmp_path, mol_name="ACE")
    txt = itp_path.read_text(encoding="utf-8")

    assert txt.count("[ dihedrals ]") >= 2
    assert "; impropers" in txt
    assert "  4  180.0" in txt or "  4   180.0" in txt


def test_oplsaa_reference_audit_tracks_assignment_and_topology_for_acetate():
    report = audit_oplsaa_reference(smiles="CC(=O)[O-]", export_topology=True)

    assert report["defaults_parity"]["matches"] is True
    assert report["assignment"]["assignment_complete"] is True
    assert report["topology"]["topology_complete"] is True
    assert "donor_h_lj_floor" in {item["kind"] for item in report["locally_patched"]}


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


def test_gaff2_mod_assigns_tfsi_sulfonimide_nitrogen():
    mol = _assign_gaff2_mod("FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F")
    atom_types = [atom.GetProp("ff_type") for atom in mol.GetAtoms()]

    assert "n" in atom_types
    assert atom_types.count("s6") == 2
    assert atom_types.count("o") == 4


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


def test_oplsaa_mol_defaults_to_resp_then_falls_back_to_builtin_path(monkeypatch):
    ff = OPLSAA()
    calls = []

    def fake_mol_rdkit(smiles, **kwargs):
        calls.append((smiles, dict(kwargs)))
        if kwargs.get("require_ready"):
            raise RuntimeError("no RESP-ready variant")
        return Chem.AddHs(Chem.MolFromSmiles(smiles))

    monkeypatch.setattr(ff, "mol_rdkit", fake_mol_rdkit)

    mol = ff.mol("CCO")

    assert mol is not None
    assert len(calls) == 2
    assert calls[0][0] == "CCO"
    assert calls[0][1]["charge"] == "RESP"
    assert calls[0][1]["require_ready"] is True
    assert calls[1][1]["charge"] == "opls"
    assert calls[1][1]["require_ready"] is False
    assert mol.HasProp("_yadonpy_charge_fallback")
    assert mol.GetProp("_yadonpy_charge_fallback") == "opls"


def test_oplsaa_default_resp_handle_falls_back_to_builtin_charges_when_resp_is_unavailable():
    ff = OPLSAA()

    assigned = ff.ff_assign(ff.mol("C[N+](C)(C)C"), report=False)

    assert assigned is not False
    atom_types = [atom.GetProp("ff_type") for atom in assigned.GetAtoms()]
    atom_charges = [atom.GetDoubleProp("AtomicCharge") for atom in assigned.GetAtoms()]
    assert atom_types.count("opls_288") == 1
    assert atom_types.count("opls_291") == 4
    assert abs(sum(atom_charges) - 1.0) < 1.0e-8
    assert assigned.HasProp("_yadonpy_charge_fallback")
    assert assigned.GetProp("_yadonpy_charge_fallback") == "opls"


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


def test_oplsaa_assigns_pf6_with_builtin_ion_types():
    ff = OPLSAA()
    pf6 = Chem.MolFromSmiles("F[P-](F)(F)(F)(F)F")

    assert ff.assign_ptypes(pf6, charge="opls")
    atom_types = [atom.GetProp("ff_type") for atom in pf6.GetAtoms()]
    atom_charges = [atom.GetDoubleProp("AtomicCharge") for atom in pf6.GetAtoms()]
    assert atom_types.count("opls_786") == 6
    assert atom_types.count("opls_785") == 1
    assert abs(sum(atom_charges) + 1.0) < 1.0e-8


@pytest.mark.parametrize(
    ("smiles", "expected_charge"),
    [
        ("[F-]", -1.0),
        ("[Cl-]", -1.0),
        ("[Br-]", -1.0),
        ("[I-]", -1.0),
        ("[Li+]", 1.0),
        ("[Na+]", 1.0),
        ("[Mg+2]", 2.0),
        ("[Ca+2]", 2.0),
    ],
)
def test_oplsaa_assigns_supported_simple_inorganic_ions(smiles, expected_charge):
    ff = OPLSAA()
    mol = Chem.MolFromSmiles(smiles)

    assigned = ff.ff_assign(mol, charge="opls", report=False)

    assert assigned is not False
    total_charge = sum(atom.GetDoubleProp("AtomicCharge") for atom in assigned.GetAtoms())
    assert abs(total_charge - expected_charge) < 1.0e-8


def test_oplsaa_assigns_hydroxide_when_explicit_hydrogen_is_materialized():
    ff = OPLSAA()
    hydroxide = ff.mol("[OH-]", require_ready=False, prefer_db=False)

    assigned = ff.ff_assign(hydroxide, charge="opls", report=False)

    assert assigned is not False
    atom_types = [atom.GetProp("ff_type") for atom in assigned.GetAtoms()]
    atom_charges = [atom.GetDoubleProp("AtomicCharge") for atom in assigned.GetAtoms()]
    assert atom_types == ["opls_434", "opls_435"]
    assert abs(sum(atom_charges) + 1.0) < 1.0e-8


@pytest.mark.parametrize(
    ("smiles", "expected_charge"),
    [
        ("C[O-]", -1.0),
        ("C[S-]", -1.0),
        ("O=[N+]([O-])[O-]", -1.0),
        ("C[N+](C)(C)C", 1.0),
        ("C[P+](C)(C)C", 1.0),
        ("C[NH3+]", 1.0),
    ],
)
def test_oplsaa_assigns_supported_organic_ions(smiles, expected_charge):
    ff = OPLSAA()
    base = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(base) if base.GetNumAtoms() > 1 else base

    assigned = ff.ff_assign(mol, charge="opls", report=False)

    assert assigned is not False
    total_charge = sum(atom.GetDoubleProp("AtomicCharge") for atom in assigned.GetAtoms())
    assert abs(total_charge - expected_charge) < 1.0e-8
    if smiles == "C[N+](C)(C)C":
        atom_types = [atom.GetProp("ff_type") for atom in assigned.GetAtoms()]
        assert atom_types.count("opls_288") == 1
        assert atom_types.count("opls_291") == 4
        assert atom_types.count("opls_156") == 12


def test_oplsaa_mol_preserves_embedded_hydrogen_ions_for_assignment():
    ff = OPLSAA()

    for smiles, expected_types, expected_charge in (
        ("[NH4+]", ["opls_286", "opls_289"], 1.0),
        ("[OH-]", ["opls_434", "opls_435"], -1.0),
    ):
        mol = ff.mol(smiles, require_ready=False, prefer_db=False)
        assigned = ff.ff_assign(mol, charge="opls", report=False)

        assert assigned is not False
        atom_types = {atom.GetProp("ff_type") for atom in assigned.GetAtoms()}
        total_charge = sum(atom.GetDoubleProp("AtomicCharge") for atom in assigned.GetAtoms())
        assert atom_types == set(expected_types)
        assert abs(total_charge - expected_charge) < 1.0e-8


@pytest.mark.parametrize("smiles", ["F[B-](F)(F)F", "[O-][Cl+3]([O-])([O-])[O-]"])
def test_oplsaa_explicitly_rejects_unsupported_inorganic_ions(smiles):
    ff = OPLSAA()
    mol = Chem.MolFromSmiles(smiles)

    assert mol is not None
    assert ff.assign_ptypes(mol, charge="opls") is False


def test_moldb_roundtrip_restores_pf6_anion_graph_semantics():
    repo_db_dir = Path(__file__).resolve().parents[1] / "moldb"
    db = MolDB(repo_db_dir)

    pf6, rec = db.load_mol("F[P-](F)(F)(F)(F)F", require_ready=True, charge="RESP")

    assert rec.canonical == "F[P-](F)(F)(F)(F)F"
    assert Chem.MolToSmiles(Chem.RemoveHs(pf6), isomericSmiles=True) == "F[P-](F)(F)(F)(F)F"
    phosphorus = [atom for atom in pf6.GetAtoms() if atom.GetSymbol() == "P"]
    assert len(phosphorus) == 1
    assert phosphorus[0].GetFormalCharge() == -1


def test_oplsaa_assigns_moldb_backed_pf6_with_preserved_drih_bonded_fragment():
    ff = OPLSAA()
    repo_db_dir = Path(__file__).resolve().parents[1] / "moldb"
    pf6 = ff.mol_rdkit(
        "F[P-](F)(F)(F)(F)F",
        db_dir=repo_db_dir,
        charge="RESP",
        require_ready=True,
        prefer_db=True,
    )

    assigned = ff.ff_assign(pf6, charge="opls", bonded="DRIH", report=False)

    assert assigned is not False
    atom_types = [atom.GetProp("ff_type") for atom in assigned.GetAtoms()]
    atom_charges = [atom.GetDoubleProp("AtomicCharge") for atom in assigned.GetAtoms()]
    assert atom_types.count("opls_786") == 6
    assert atom_types.count("opls_785") == 1
    assert abs(sum(atom_charges) + 1.0) < 1.0e-8
    assert assigned.HasProp("_yadonpy_bonded_itp")
    assert assigned.HasProp("_yadonpy_bonded_json")


def test_oplsaa_pf6_structural_fallback_handles_legacy_positive_p_mol2_graph():
    from rdkit.Chem import rdmolfiles

    ff = OPLSAA()
    mol2_path = Path(__file__).resolve().parents[1] / "moldb" / "objects" / "a262cd2921905bc6" / "best.mol2"
    pf6 = rdmolfiles.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=False)
    assert pf6 is not None

    assert ff.assign_ptypes(pf6, charge="opls")
    atom_types = [atom.GetProp("ff_type") for atom in pf6.GetAtoms()]
    assert atom_types.count("opls_786") == 6
    assert atom_types.count("opls_785") == 1


def test_oplsaa_assigns_acyclic_carbonates_with_fallback_bonded_terms():
    ff = OPLSAA()

    for smiles in ("CCOC(=O)OC", "CCOC(=O)OCC"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assigned = ff.ff_assign(mol, charge="opls", report=False)
        assert assigned is not False
        assert assigned.angles
        assert assigned.dihedrals


def test_oplsaa_assigns_resp_backed_cmc_carboxylate_monomers_from_repo_moldb():
    ff = OPLSAA()
    repo_db_dir = Path(__file__).resolve().parents[1] / "moldb"

    for smiles in (
        "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]",
        "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O",
        "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O",
    ):
        mol = ff.mol_rdkit(
            smiles,
            db_dir=repo_db_dir,
            charge="RESP",
            require_ready=True,
            prefer_db=True,
            polyelectrolyte_mode=True,
            polyelectrolyte_detection="auto",
        )
        before = [atom.GetDoubleProp("AtomicCharge") for atom in mol.GetAtoms()]

        assigned = ff.ff_assign(mol, charge=None, report=False)

        assert assigned is not False
        after = [atom.GetDoubleProp("AtomicCharge") for atom in assigned.GetAtoms()]
        assert len(assigned.angles) > 0
        assert len(assigned.dihedrals) > 0
        assert before == after


def test_oplsaa_assigns_dtd_cyclic_sulfate_with_targeted_rules():
    ff = OPLSAA()
    dtd = Chem.AddHs(Chem.MolFromSmiles("O=S1(=O)OC=CO1"))

    assigned = ff.ff_assign(dtd, charge="opls", report=False)

    assert assigned is not False
    btypes = [atom.GetProp("ff_btype") for atom in assigned.GetAtoms()]
    assert btypes.count("SY") == 1
    assert btypes.count("OY") == 2
    assert btypes.count("OS") == 2
    assert btypes.count("CM") == 2
    assert len(assigned.angles) > 0
    assert len(assigned.dihedrals) > 0


def test_oplsaa_polymer_junction_refresh_reuses_existing_monomer_assignment(capsys):
    ff = OPLSAA()
    monomer = ff.mol('*OCC*', require_ready=False, prefer_db=False)
    monomer = ff.ff_assign(monomer, charge='opls', report=False)

    assert monomer is not False
    capsys.readouterr()

    polymer = poly.random_copolymerize_rw(
        [monomer],
        3,
        ratio=[1.0],
        tacticity='atactic',
        name='opls_poly_acetal',
        retry=1,
        retry_step=20,
        retry_opt_step=0,
    )

    captured = capsys.readouterr()
    assert polymer is not None
    assert 'OPLS-AA typing failed' not in captured.out


def test_oplsaa_assigns_resp_backed_cmc_short_copolymer_without_losing_types_or_charges():
    ff = OPLSAA()
    repo_db_dir = Path(__file__).resolve().parents[1] / "moldb"

    monomers = []
    for smiles in (
        "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]",
        "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O",
        "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O",
    ):
        mol = ff.mol_rdkit(
            smiles,
            db_dir=repo_db_dir,
            charge="RESP",
            require_ready=True,
            prefer_db=True,
            polyelectrolyte_mode=True,
            polyelectrolyte_detection="auto",
        )
        before = [atom.GetDoubleProp("AtomicCharge") for atom in mol.GetAtoms()]
        assigned = ff.ff_assign(mol, charge=None, report=False)
        after = [atom.GetDoubleProp("AtomicCharge") for atom in assigned.GetAtoms()]
        assert assigned is not False
        assert before == after
        monomers.append(assigned)

    polymer = poly.random_copolymerize_rw(
        monomers,
        6,
        ratio=[1 / 3, 1 / 3, 1 / 3],
        tacticity="atactic",
        name="opls_cmc_short",
        retry=1,
        retry_step=30,
        retry_opt_step=0,
    )
    assigned_polymer = ff.ff_assign(polymer, charge=None, report=False)

    assert assigned_polymer is not False
    assert len(getattr(assigned_polymer, "angles", {})) > 0
    assert len(getattr(assigned_polymer, "dihedrals", {})) > 0
    assert all(atom.HasProp("ff_type") and atom.HasProp("ff_btype") for atom in assigned_polymer.GetAtoms())
    _assert_nonzero_opls_lj_for_materialized_types(assigned_polymer)
    _assert_hydroxyl_donor_angles_present(assigned_polymer)


def test_oplsaa_random_walk_restart_cache_preserves_lj_parameters(tmp_path):
    ff = OPLSAA()
    monomer = ff.mol("*OCC*", require_ready=False, prefer_db=False)
    monomer = ff.ff_assign(monomer, charge="opls", report=False)

    assert monomer is not False

    wd = workdir(tmp_path / "opls_rw_cache", restart=True)
    polymer_1 = poly.random_copolymerize_rw(
        [monomer],
        3,
        ratio=[1.0],
        tacticity="atactic",
        name="opls_poly_cache",
        retry=1,
        retry_step=20,
        retry_opt_step=0,
        work_dir=wd,
    )
    polymer_2 = poly.random_copolymerize_rw(
        [monomer],
        3,
        ratio=[1.0],
        tacticity="atactic",
        name="opls_poly_cache",
        retry=1,
        retry_step=20,
        retry_opt_step=0,
        work_dir=wd,
    )

    assigned_1 = ff.ff_assign(polymer_1, charge=None, report=False)
    assigned_2 = ff.ff_assign(polymer_2, charge=None, report=False)

    assert assigned_1 is not False
    assert assigned_2 is not False
    _assert_nonzero_opls_lj_for_materialized_types(assigned_1)
    _assert_nonzero_opls_lj_for_materialized_types(assigned_2)


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
