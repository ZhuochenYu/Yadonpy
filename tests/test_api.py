from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

import yadonpy.api as api
import yadonpy.moldb as moldb
from yadonpy.core.polyelectrolyte import annotate_polyelectrolyte_metadata
import yadonpy
from yadonpy.sim.analyzer import AnalyzeResult


class _DummyDB:
    def __init__(self):
        self.calls = []

    def load_mol(self, smiles, **kwargs):
        self.calls.append((smiles, kwargs))
        return 'MOL', {'key': 'dummy'}


def test_load_from_moldb_returns_molecule_by_default(monkeypatch):
    db = _DummyDB()
    monkeypatch.setattr(moldb, 'MolDB', lambda: db)

    mol = api.load_from_moldb('O=C1OCCO1', charge='RESP', basis_set='def2-TZVP', method='wb97m-d3bj')

    assert mol == 'MOL'
    assert db.calls == [
        (
            'O=C1OCCO1',
            {
                'require_ready': True,
                'charge': 'RESP',
                'basis_set': 'def2-TZVP',
                'method': 'wb97m-d3bj',
                'polyelectrolyte_mode': None,
                'polyelectrolyte_detection': None,
                'resp_profile': None,
            },
        )
    ]


def test_load_from_moldb_can_return_record(monkeypatch):
    db = _DummyDB()
    monkeypatch.setattr(moldb, 'MolDB', lambda: db)

    mol, record = api.load_from_moldb('O=C1OCCO1', return_record=True, require_ready=False)

    assert mol == 'MOL'
    assert record == {'key': 'dummy'}
    assert db.calls[0][1]['require_ready'] is False
    assert db.calls[0][1]['polyelectrolyte_mode'] is None
    assert db.calls[0][1]['polyelectrolyte_detection'] is None
    assert db.calls[0][1]['resp_profile'] is None


def test_top_level_assign_charges_passes_resp_profile(monkeypatch):
    calls = []

    class DummyQM:
        @staticmethod
        def assign_charges(*args, **kwargs):
            calls.append(kwargs)
            return True

    monkeypatch.setattr(__import__('yadonpy.sim', fromlist=['qm']), 'qm', DummyQM)

    assert api.assign_charges("mol-object", charge="RESP", resp_profile="legacy") is True
    assert calls[0]["resp_profile"] == "legacy"


def test_assign_forcefield_returns_ff_instance_and_status(monkeypatch):
    class DummyFF:
        def __init__(self):
            self.calls = []

        def ff_assign(self, mol, **kwargs):
            self.calls.append((mol, kwargs))
            return 7

    ff = DummyFF()
    monkeypatch.setattr(api, 'get_ff', lambda ff_name, **kwargs: ff)

    out_ff, ok = api.assign_forcefield('mol-object', ff_name='gaff2_mod', charge='RESP', report=False)

    assert out_ff is ff
    assert ok is True
    assert ff.calls == [('mol-object', {'charge': 'RESP', 'report': False})]


def test_list_charge_methods_exposes_scaled_charge_tokens():
    methods = api.list_charge_methods()
    assert 'RESP' in methods
    assert 'RESP2' in methods
    assert 'CM1A' in methods
    assert 'CM5' in methods
    assert '1.14*CM1A' in methods
    assert '1.2*CM5' in methods


def test_top_level_api_exports_mechanics_helpers():
    assert hasattr(yadonpy, 'doctor')
    assert hasattr(yadonpy, 'AnalyzeResult')
    assert hasattr(api, 'audit_oplsaa_reference')
    assert hasattr(yadonpy, 'audit_oplsaa_reference')
    assert hasattr(api, 'resolve_prepared_system')
    assert hasattr(api, 'run_tg_scan_gmx')
    assert hasattr(api, 'run_elongation_gmx')
    assert hasattr(api, 'format_mechanics_result_summary')
    assert hasattr(api, 'print_mechanics_result_summary')
    assert 'resolve_prepared_system' in api.__all__
    assert 'run_tg_scan_gmx' in api.__all__
    assert 'run_elongation_gmx' in api.__all__
    assert 'format_mechanics_result_summary' in api.__all__
    assert 'print_mechanics_result_summary' in api.__all__
    assert 'audit_oplsaa_reference' in api.__all__
    assert hasattr(yadonpy, 'resolve_prepared_system')
    assert hasattr(yadonpy, 'run_tg_scan_gmx')
    assert hasattr(yadonpy, 'run_elongation_gmx')
    assert 'AnalyzeResult' in yadonpy.__all__
    assert 'doctor' in yadonpy.__all__
    assert hasattr(api, 'prepare_graphite_substrate')
    assert hasattr(api, 'calibrate_polymer_bulk_phase')
    assert hasattr(api, 'calibrate_electrolyte_bulk_phase')
    assert hasattr(api, 'build_graphite_polymer_interphase')
    assert hasattr(api, 'build_polymer_electrolyte_interphase')
    assert hasattr(api, 'release_graphite_polymer_electrolyte_stack')
    assert hasattr(api, 'print_interface_result_summary')
    assert hasattr(yadonpy, 'GraphiteSubstrateSpec')
    assert hasattr(yadonpy, 'PolymerSlabSpec')
    assert hasattr(yadonpy, 'ElectrolyteSlabSpec')
    assert hasattr(yadonpy, 'SandwichRelaxationSpec')


def test_analyzer_does_not_expose_transport_bundle_api():
    assert not hasattr(AnalyzeResult, 'transport')
    assert hasattr(AnalyzeResult, 'migration')
    assert hasattr(AnalyzeResult, 'migration_markov')
    assert hasattr(AnalyzeResult, 'migration_residence')


def test_parameterize_smiles_raises_when_charge_assignment_fails_by_default(monkeypatch, tmp_path):
    class DummyFF:
        def __init__(self):
            self.called = False

        def ff_assign(self, mol, **kwargs):
            self.called = True
            return True

    ff = DummyFF()
    monkeypatch.setattr(api, 'get_ff', lambda ff_name, **kwargs: ff)
    monkeypatch.setattr(api, 'mol_from_smiles', lambda smiles, coord=True, name=None: object())

    class DummyQM:
        @staticmethod
        def assign_charges(*args, **kwargs):
            raise RuntimeError('boom')

    monkeypatch.setattr(api, 'warnings', __import__('warnings'))
    monkeypatch.setattr(__import__('yadonpy.sim', fromlist=['qm']), 'qm', DummyQM)

    with pytest.raises(RuntimeError, match='Charge assignment failed'):
        api.parameterize_smiles('CCO', work_dir=str(tmp_path))

    assert ff.called is False


def test_parameterize_smiles_can_opt_in_to_legacy_fallback(monkeypatch, tmp_path):
    class DummyFF:
        def __init__(self):
            self.called = False

        def ff_assign(self, mol, **kwargs):
            self.called = True
            return True

    ff = DummyFF()
    monkeypatch.setattr(api, 'get_ff', lambda ff_name, **kwargs: ff)
    monkeypatch.setattr(api, 'mol_from_smiles', lambda smiles, coord=True, name=None: 'mol-object')

    class DummyQM:
        @staticmethod
        def assign_charges(*args, **kwargs):
            raise RuntimeError('boom')

    monkeypatch.setattr(__import__('yadonpy.sim', fromlist=['qm']), 'qm', DummyQM)

    with pytest.warns(RuntimeWarning, match='allow_ff_without_requested_charges=True'):
        mol, ok = api.parameterize_smiles(
            'CCO',
            work_dir=str(tmp_path),
            allow_ff_without_requested_charges=True,
        )

    assert mol == 'mol-object'
    assert ok is True
    assert ff.called is True


def test_moldb_update_and_load_preserves_bonded_patch_files(tmp_path: Path):
    db = moldb.MolDB(db_dir=tmp_path / 'moldb')
    mol = api.mol_from_smiles('CCO', coord=True, name='ethanol')
    assert isinstance(mol, Chem.Mol)

    charges = (-0.2, 0.1, 0.1)
    for atom, charge in zip(mol.GetAtoms(), charges):
        atom.SetDoubleProp('AtomicCharge', float(charge))

    patch_root = tmp_path / 'patch_src'
    patch_root.mkdir(parents=True, exist_ok=True)
    patch_itp = patch_root / 'bonded_drih_patch.itp'
    patch_json = patch_root / 'bonded_drih_params.json'
    patch_itp.write_text('; fake bonded patch\n', encoding='utf-8')
    patch_json.write_text('{"meta": {"method": "DRIH"}, "bonds": [], "angles": []}\n', encoding='utf-8')

    mol.SetProp('_yadonpy_bonded_itp', str(patch_itp.resolve()))
    mol.SetProp('_yadonpy_bonded_json', str(patch_json.resolve()))
    mol.SetProp('_yadonpy_bonded_method', 'DRIH')
    mol.SetProp('_yadonpy_bonded_requested', 'drih')
    mol.SetProp('_yadonpy_bonded_signature', 'drih')
    mol.SetProp('_yadonpy_bonded_override', '1')
    mol.SetProp('_yadonpy_bonded_explicit', '1')

    rec = db.update_from_mol(mol, smiles_or_psmiles='CCO', name='ethanol', charge='RESP')
    vid = next(iter(rec.variants))
    bonded_meta = rec.variants[vid].get('bonded')

    assert isinstance(bonded_meta, dict)
    assert bonded_meta['_yadonpy_bonded_method'] == 'DRIH'
    assert (db.record_dir(rec.key) / 'bonded' / vid / 'bonded_drih_patch.itp').exists()
    assert (db.record_dir(rec.key) / 'bonded' / vid / 'bonded_drih_params.json').exists()

    loaded, loaded_rec = db.load_mol('CCO', require_ready=True, charge='RESP')

    assert loaded_rec.key == rec.key
    assert loaded.HasProp('_yadonpy_bonded_itp')
    assert loaded.HasProp('_yadonpy_bonded_json')
    assert loaded.GetProp('_yadonpy_bonded_method') == 'DRIH'
    assert Path(loaded.GetProp('_yadonpy_bonded_itp')).is_file()
    assert Path(loaded.GetProp('_yadonpy_bonded_json')).is_file()


def test_moldb_load_falls_back_when_best_mol2_is_corrupted(tmp_path: Path):
    db = moldb.MolDB(db_dir=tmp_path / 'moldb')
    mol = api.mol_from_smiles('CCO', coord=True, name='ethanol')
    assert isinstance(mol, Chem.Mol)

    for atom, charge in zip(mol.GetAtoms(), (-0.2, 0.1, 0.1)):
        atom.SetDoubleProp('AtomicCharge', float(charge))

    rec = db.update_from_mol(mol, smiles_or_psmiles='CCO', name='ethanol', charge='RESP')
    best_mol2 = db.mol2_path(rec.key)
    fallback_mol2 = db.record_dir(rec.key) / 'ethanol.mol2'

    assert best_mol2.exists()
    assert fallback_mol2.exists()

    best_mol2.write_text('corrupted mol2\n', encoding='utf-8')

    loaded, loaded_rec = db.load_mol('CCO', require_ready=True, charge='RESP')

    assert loaded_rec.key == rec.key
    assert loaded.GetNumAtoms() == mol.GetNumAtoms()


def test_moldb_polyelectrolyte_variants_are_distinguished(tmp_path: Path):
    db = moldb.MolDB(db_dir=tmp_path / 'moldb')
    smiles = '*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]'

    mol_plain = api.mol_from_smiles(smiles, coord=True, name='glucose_6')
    for atom in mol_plain.GetAtoms():
        atom.SetDoubleProp('AtomicCharge', float(atom.GetFormalCharge()) * 0.1)
    rec_plain = db.update_from_mol(
        mol_plain,
        smiles_or_psmiles=smiles,
        name='glucose_6',
        charge='RESP',
    )

    mol_poly = api.mol_from_smiles(smiles, coord=True, name='glucose_6')
    for atom in mol_poly.GetAtoms():
        atom.SetDoubleProp('AtomicCharge', float(atom.GetFormalCharge()) * 0.1)
    annotate_polyelectrolyte_metadata(mol_poly)
    rec_poly = db.update_from_mol(
        mol_poly,
        smiles_or_psmiles=smiles,
        name='glucose_6',
        charge='RESP',
        polyelectrolyte_mode=True,
    )

    assert rec_plain.key == rec_poly.key
    assert len(rec_poly.variants) >= 2
    metas = list(rec_poly.variants.values())
    assert any(bool(v.get('polyelectrolyte_mode')) for v in metas)
    assert any(not bool(v.get('polyelectrolyte_mode', False)) for v in metas)


def test_moldb_load_restores_polyelectrolyte_metadata(tmp_path: Path):
    db = moldb.MolDB(db_dir=tmp_path / 'moldb')
    smiles = '*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]'
    mol = api.mol_from_smiles(smiles, coord=True, name='glucose_6')
    for atom in mol.GetAtoms():
        atom.SetDoubleProp('AtomicCharge', float(atom.GetFormalCharge()) * 0.1)
    annotate_polyelectrolyte_metadata(mol)

    rec = db.update_from_mol(
        mol,
        smiles_or_psmiles=smiles,
        name='glucose_6',
        charge='RESP',
        polyelectrolyte_mode=True,
    )
    loaded, loaded_rec = db.load_mol(smiles, require_ready=True, charge='RESP')

    assert loaded_rec.key == rec.key
    assert loaded.HasProp('_YADONPY_VARIANT_ID')
    assert loaded.HasProp('_yadonpy_charge_groups_json')
    assert loaded.HasProp('_yadonpy_resp_constraints_json')
    assert loaded.HasProp('_yadonpy_polyelectrolyte_summary_json')


def test_moldb_load_regenerates_localized_charge_groups_for_stale_polyelectrolyte_variant(tmp_path: Path):
    db = moldb.MolDB(db_dir=tmp_path / 'moldb')
    smiles = '*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]'
    mol = api.mol_from_smiles(smiles, coord=True, name='glucose_6')
    for atom in mol.GetAtoms():
        atom.SetDoubleProp('AtomicCharge', float(atom.GetFormalCharge()) * 0.1)

    rec = db.update_from_mol(
        mol,
        smiles_or_psmiles=smiles,
        name='glucose_6',
        charge='RESP',
        polyelectrolyte_mode=True,
    )
    vid = next(iter(rec.variants))
    rec.variants[vid]['charge_groups'] = []
    rec.variants[vid]['resp_constraints'] = {}
    rec.variants[vid]['polyelectrolyte_summary'] = {}
    db.save_record(rec)

    loaded, _ = db.load_mol(
        smiles,
        require_ready=True,
        charge='RESP',
        polyelectrolyte_mode=True,
    )

    groups = json.loads(loaded.GetProp('_yadonpy_charge_groups_json'))
    constraints = json.loads(loaded.GetProp('_yadonpy_resp_constraints_json'))
    summary = json.loads(loaded.GetProp('_yadonpy_polyelectrolyte_summary_json'))

    assert groups
    assert constraints.get('mode') == 'grouped'
    assert summary.get('groups')


def test_moldb_load_upgrades_legacy_false_variant_when_localized_groups_are_requested(tmp_path: Path):
    db = moldb.MolDB(db_dir=tmp_path / "moldb")
    smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"
    mol = api.mol_from_smiles(smiles, coord=True, name="glucose_6")
    for atom in mol.GetAtoms():
        atom.SetDoubleProp("AtomicCharge", float(atom.GetFormalCharge()) * 0.1)

    rec = db.update_from_mol(
        mol,
        smiles_or_psmiles=smiles,
        name="glucose_6",
        charge="RESP",
        polyelectrolyte_mode=False,
    )
    only_meta = next(iter(rec.variants.values()))
    assert bool(only_meta.get("polyelectrolyte_mode", False)) is False

    loaded, _ = db.load_mol(
        smiles,
        require_ready=True,
        charge="RESP",
        polyelectrolyte_mode=True,
    )

    groups = json.loads(loaded.GetProp("_yadonpy_charge_groups_json"))
    constraints = json.loads(loaded.GetProp("_yadonpy_resp_constraints_json"))
    summary = json.loads(loaded.GetProp("_yadonpy_polyelectrolyte_summary_json"))

    assert groups
    assert constraints.get("mode") == "grouped"
    assert summary.get("groups")
    assert loaded.GetProp("_YADONPY_POLYELECTROLYTE_MODE") == "1"


def test_moldb_load_can_select_resp2_variant_without_explicit_level(tmp_path: Path):
    db = moldb.MolDB(db_dir=tmp_path / "moldb")
    smiles = "O=C1OCCO1"
    mol = api.mol_from_smiles(smiles, coord=True, name="EC")
    charges = [-0.59, 0.89, -0.40, 0.11, 0.10, -0.40, 0.07, 0.07, 0.07, 0.08]
    assert mol.GetNumAtoms() == len(charges)
    for atom, charge in zip(mol.GetAtoms(), charges):
        atom.SetDoubleProp("RESP2", float(charge))
        atom.SetDoubleProp("AtomicCharge", float(charge))

    rec = db.update_from_mol(
        mol,
        smiles_or_psmiles=smiles,
        name="EC",
        charge="RESP2",
        basis_set="def2-TZVP",
        method="wb97m-v",
    )

    loaded, loaded_rec = db.load_mol(smiles, require_ready=True, charge="RESP2")

    assert loaded_rec.key == rec.key
    assert all(atom.HasProp("RESP2") for atom in loaded.GetAtoms())
    got = [atom.GetDoubleProp("RESP2") for atom in loaded.GetAtoms()]
    assert got == pytest.approx(charges)


def test_moldb_resp_profile_variants_are_distinguished(tmp_path: Path):
    db = moldb.MolDB(db_dir=tmp_path / "moldb")
    smiles = "O=C1OCCO1"

    mol_adaptive = api.mol_from_smiles(smiles, coord=True, name="EC")
    for atom, charge in zip(mol_adaptive.GetAtoms(), [-0.6, 0.9, -0.4, 0.1, 0.1, -0.4, 0.1, 0.1, 0.1, 0.1]):
        atom.SetDoubleProp("AtomicCharge", float(charge))
    mol_adaptive.SetProp("_yadonpy_resp_profile", "adaptive")
    mol_adaptive.SetProp(
        "_yadonpy_qm_recipe_json",
        json.dumps({"resp_profile": "adaptive", "opt_method": "wb97m-v", "charge_method": "wb97m-v"}),
    )
    rec = db.update_from_mol(mol_adaptive, smiles_or_psmiles=smiles, name="EC", charge="RESP", method="Default", basis_set="Default")

    mol_legacy = api.mol_from_smiles(smiles, coord=True, name="EC")
    for atom, charge in zip(mol_legacy.GetAtoms(), [-0.5, 0.8, -0.35, 0.1, 0.1, -0.35, 0.1, 0.1, 0.1, 0.1]):
        atom.SetDoubleProp("AtomicCharge", float(charge))
    mol_legacy.SetProp("_yadonpy_resp_profile", "legacy")
    mol_legacy.SetProp(
        "_yadonpy_qm_recipe_json",
        json.dumps({"resp_profile": "legacy", "opt_method": "wb97m-d3bj", "charge_method": "wb97m-d3bj"}),
    )
    rec = db.update_from_mol(mol_legacy, smiles_or_psmiles=smiles, name="EC", charge="RESP", method="Default", basis_set="Default")

    metas = list(rec.variants.values())
    assert len(metas) >= 2
    assert {meta.get("resp_profile") for meta in metas} >= {"adaptive", "legacy"}

    loaded_adaptive, _ = db.load_mol(smiles, require_ready=True, charge="RESP", resp_profile="adaptive")
    loaded_legacy, _ = db.load_mol(smiles, require_ready=True, charge="RESP", resp_profile="legacy")

    assert loaded_adaptive.GetProp("_yadonpy_resp_profile") == "adaptive"
    assert loaded_legacy.GetProp("_yadonpy_resp_profile") == "legacy"
    assert loaded_adaptive.GetProp("_YADONPY_VARIANT_ID") != loaded_legacy.GetProp("_YADONPY_VARIANT_ID")


def test_moldb_loads_old_legacy_style_variant_without_rewriting_profile_metadata(tmp_path: Path):
    db = moldb.MolDB(db_dir=tmp_path / "moldb")
    smiles = "O=C1OCCO1"
    mol = api.mol_from_smiles(smiles, coord=True, name="EC")
    charges = [-0.55, 0.85, -0.38, 0.1, 0.1, -0.38, 0.08, 0.08, 0.05, 0.05]
    for atom, charge in zip(mol.GetAtoms(), charges):
        atom.SetDoubleProp("AtomicCharge", float(charge))

    rec = db.update_from_mol(mol, smiles_or_psmiles=smiles, name="EC", charge="RESP")
    vid = next(iter(rec.variants))
    rec.variants[vid].pop("resp_profile", None)
    rec.variants[vid].pop("qm_recipe", None)
    rec.variants[vid].pop("resp_constraints", None)
    rec.variants[vid].pop("polyelectrolyte_summary", None)
    rec.variants[vid].pop("charge_groups", None)
    rec.variants[vid].pop("psiresp_constraints", None)
    db.save_record(rec)

    payload_path = db.charges_variant_path(rec.key, vid)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("resp_profile", None)
    payload.pop("qm_recipe", None)
    payload.pop("resp_constraints", None)
    payload.pop("polyelectrolyte_summary", None)
    payload.pop("charge_groups", None)
    payload.pop("psiresp_constraints", None)
    payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    loaded, _ = db.load_mol(smiles, require_ready=True, charge="RESP", resp_profile="legacy")

    assert [atom.GetDoubleProp("AtomicCharge") for atom in loaded.GetAtoms()] == pytest.approx(charges)
    assert not loaded.HasProp("_yadonpy_resp_profile")

    reloaded_rec = db.load_record(rec.key)
    assert reloaded_rec is not None
    assert "resp_profile" not in reloaded_rec.variants[vid]
