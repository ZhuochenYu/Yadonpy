from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

import yadonpy.api as api
import yadonpy.moldb as moldb
from yadonpy.core.polyelectrolyte import annotate_polyelectrolyte_metadata
import yadonpy


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
    assert 'CM1A' in methods
    assert 'CM5' in methods
    assert '1.14*CM1A' in methods
    assert '1.2*CM5' in methods


def test_top_level_api_exports_mechanics_helpers():
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
    assert hasattr(yadonpy, 'resolve_prepared_system')
    assert hasattr(yadonpy, 'run_tg_scan_gmx')
    assert hasattr(yadonpy, 'run_elongation_gmx')


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
