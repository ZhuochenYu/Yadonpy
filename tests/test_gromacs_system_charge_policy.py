import json

from rdkit import Chem
from rdkit import Geometry as Geom
from rdkit.Chem import AllChem

from yadonpy.ff import OPLSAA
from yadonpy.io.artifacts import _artifact_meta_compatibility_fields
from yadonpy.io.gromacs_molecule import write_gromacs_single_molecule_topology
from yadonpy.io.gromacs_system import _cached_artifact_compatible, _species_charge_policy, export_system_from_cell_meta


def _single_ion_cell_meta() -> Chem.Mol:
    cell = Chem.MolFromSmiles("[Li+]")
    conf = Chem.Conformer(cell.GetNumAtoms())
    conf.SetAtomPosition(0, Geom.Point3D(0.0, 0.0, 0.0))
    cell.AddConformer(conf)
    payload = {
        "schema_version": "test",
        "density_g_cm3": 0.05,
        "species": [
            {
                "smiles": "[Li+]",
                "n": 1,
                "natoms": 1,
                "name": "Li",
                "ff_name": "oplsaa",
                "charge_method": "opls",
                "prefer_db": False,
                "require_db": False,
                "require_ready": False,
            }
        ],
    }
    cell.SetProp("_yadonpy_cell_meta", json.dumps(payload))
    return cell


def test_species_charge_policy_prefers_species_specific_route():
    policy = _species_charge_policy(
        {
            "charge_method": "opls",
            "prefer_db": False,
            "require_db": False,
            "require_ready": False,
        },
        "RESP",
    )

    assert policy["charge_method"] == "opls"
    assert policy["resp_profile"] is None
    assert policy["prefer_db"] is False
    assert policy["require_db"] is False
    assert policy["require_ready"] is False


def test_species_charge_policy_keeps_resp_profile_variant():
    policy = _species_charge_policy(
        {
            "charge_method": "RESP",
            "resp_profile": "Adaptive",
        },
        "RESP",
    )

    assert policy["charge_method"] == "RESP"
    assert policy["resp_profile"] == "adaptive"


def _ec_with_exportable_ff_props() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("O=C1OCCO1"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    for atom in mol.GetAtoms():
        atom.SetProp("ff_type", atom.GetSymbol().lower())
        atom.SetDoubleProp("ff_sigma", 0.3)
        atom.SetDoubleProp("ff_epsilon", 0.2)
        atom.SetDoubleProp("AtomicCharge", 0.0)
        atom.SetDoubleProp("RESP", 0.0)
        atom.SetDoubleProp("ESP", 0.0)
    for bond in mol.GetBonds():
        bond.SetDoubleProp("ff_r0", 0.15)
        bond.SetDoubleProp("ff_k", 1000.0)
    return mol


def _itp_atom_charges(path):
    section = None
    charges = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw.startswith("[") and raw.endswith("]"):
            section = raw.strip("[]").strip()
            continue
        if section != "atoms" or not raw or raw.startswith(";"):
            continue
        parts = raw.split()
        charges.append(float(parts[6]))
    return charges


def test_gromacs_topology_export_repairs_resp_equivalent_charge_props(tmp_path):
    mol = _ec_with_exportable_ff_props()
    mol.GetAtomWithIdx(2).SetDoubleProp("AtomicCharge", -0.50)
    mol.GetAtomWithIdx(5).SetDoubleProp("AtomicCharge", -0.30)
    mol.GetAtomWithIdx(2).SetDoubleProp("RESP", -0.60)
    mol.GetAtomWithIdx(5).SetDoubleProp("RESP", -0.20)
    mol.GetAtomWithIdx(2).SetDoubleProp("ESP", -0.70)
    mol.GetAtomWithIdx(5).SetDoubleProp("ESP", -0.10)
    mol.SetProp("_yadonpy_resp_profile", "adaptive")
    mol.SetProp(
        "_yadonpy_resp_constraints_json",
        json.dumps({"mode": "whole_molecule_scale", "equivalence_groups": [[2, 5]]}),
    )
    mol.SetProp("_yadonpy_qm_recipe_json", json.dumps({"resp_profile": "adaptive", "charge_method": "wb97m-v"}))

    _, itp, _ = write_gromacs_single_molecule_topology(mol, tmp_path / "ec", mol_name="ec")
    charges = _itp_atom_charges(itp)

    assert charges[2] == charges[5]
    for prop in ("AtomicCharge", "RESP", "ESP"):
        assert mol.GetAtomWithIdx(2).GetDoubleProp(prop) == mol.GetAtomWithIdx(5).GetDoubleProp(prop)


def test_artifact_cache_metadata_includes_resp_profile_recipe_and_constraints():
    mol = _ec_with_exportable_ff_props()
    mol.SetProp("_yadonpy_resp_profile", "adaptive")
    mol.SetProp("_yadonpy_qm_recipe_json", json.dumps({"resp_profile": "adaptive", "charge_method": "wb97m-v"}))
    mol.SetProp(
        "_yadonpy_resp_constraints_json",
        json.dumps({"mode": "whole_molecule_scale", "equivalence_groups": [[2, 5]]}),
    )

    fields = _artifact_meta_compatibility_fields(mol, mol_name="ec")

    assert fields["resp_profile"] == "adaptive"
    assert fields["qm_recipe_signature"]
    assert fields["resp_constraints_signature"]


def test_system_export_rejects_legacy_artifact_cache_for_adaptive_constraints(tmp_path):
    cache_dir = tmp_path / "legacy_cache"
    cache_dir.mkdir()
    (cache_dir / "meta.json").write_text(json.dumps({"n_atoms": 10}), encoding="utf-8")

    assert not _cached_artifact_compatible(
        cache_dir,
        species_payload={
            "natoms": 10,
            "resp_constraints": {"mode": "whole_molecule_scale", "equivalence_groups": [[2, 5]]},
        },
        kind="solvent",
        rep_mol=None,
        mol_name="ec",
    )


def test_export_system_uses_species_specific_charge_policy_for_built_in_opls_ions(tmp_path, monkeypatch):
    calls: dict[str, object] = {}
    ff = OPLSAA()

    def _fake_mol_rdkit(
        smiles,
        *,
        name=None,
        prefer_db=True,
        require_db=False,
        require_ready=False,
        charge=None,
        **kwargs,
    ):
        calls["smiles"] = smiles
        calls["prefer_db"] = prefer_db
        calls["require_db"] = require_db
        calls["require_ready"] = require_ready
        calls["charge"] = charge
        return ff.mol(smiles, charge="opls", require_ready=False, prefer_db=False, name=name)

    monkeypatch.setattr(OPLSAA, "mol_rdkit", staticmethod(_fake_mol_rdkit))

    out = export_system_from_cell_meta(
        cell_mol=_single_ion_cell_meta(),
        out_dir=tmp_path / "sys",
        ff_name="oplsaa",
        charge_method="RESP",
        write_system_mol2=False,
    )

    assert out.system_top.exists()
    assert calls["smiles"] == "[Li+]"
    assert calls["charge"] == "opls"
    assert calls["prefer_db"] is False
    assert calls["require_db"] is False
    assert calls["require_ready"] is False


def test_export_system_forwards_resp_profile_to_moldb_fallback(tmp_path, monkeypatch):
    from yadonpy.ff import GAFF2_mod

    calls: dict[str, object] = {}
    cell = Chem.MolFromSmiles("CCO")
    conf = Chem.Conformer(cell.GetNumAtoms())
    for idx in range(cell.GetNumAtoms()):
        conf.SetAtomPosition(idx, Geom.Point3D(float(idx) * 0.1, 0.0, 0.0))
    cell.AddConformer(conf)
    payload = {
        "schema_version": "test",
        "density_g_cm3": 0.5,
        "species": [
            {
                "smiles": "CCO",
                "n": 1,
                "natoms": 3,
                "name": "ETH",
                "ff_name": "gaff2_mod",
                "charge_method": "RESP",
                "resp_profile": "adaptive",
                "prefer_db": True,
                "require_db": True,
                "require_ready": True,
            }
        ],
    }
    cell.SetProp("_yadonpy_cell_meta", json.dumps(payload))

    def _fake_mol_rdkit(
        smiles,
        *,
        name=None,
        prefer_db=True,
        require_db=False,
        require_ready=False,
        charge=None,
        resp_profile=None,
        **kwargs,
    ):
        calls["smiles"] = smiles
        calls["prefer_db"] = prefer_db
        calls["require_db"] = require_db
        calls["require_ready"] = require_ready
        calls["charge"] = charge
        calls["resp_profile"] = resp_profile
        mol = Chem.MolFromSmiles(smiles)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for idx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(idx, Geom.Point3D(float(idx) * 0.1, 0.0, 0.0))
        mol.AddConformer(conf)
        mol.SetProp("_Name", str(name or "ETH"))
        return mol

    def _fake_ff_assign(self, mol, *args, **kwargs):
        for atom in mol.GetAtoms():
            atom.SetProp("ff_type", atom.GetSymbol().lower())
            atom.SetDoubleProp("ff_sigma", 0.3)
            atom.SetDoubleProp("ff_epsilon", 0.2)
            atom.SetDoubleProp("AtomicCharge", 0.0)
        for bond in mol.GetBonds():
            bond.SetDoubleProp("ff_r0", 0.15)
            bond.SetDoubleProp("ff_k", 1000.0)
        return True

    monkeypatch.setattr(GAFF2_mod, "mol_rdkit", staticmethod(_fake_mol_rdkit))
    monkeypatch.setattr(GAFF2_mod, "ff_assign", _fake_ff_assign)

    out = export_system_from_cell_meta(
        cell_mol=cell,
        out_dir=tmp_path / "sys",
        ff_name="gaff2_mod",
        charge_method="RESP",
        write_system_mol2=False,
    )

    assert out.system_top.exists()
    assert calls["smiles"] == "CCO"
    assert calls["charge"] == "RESP"
    assert calls["resp_profile"] == "adaptive"
    assert calls["prefer_db"] is True
    assert calls["require_db"] is True
    assert calls["require_ready"] is True
