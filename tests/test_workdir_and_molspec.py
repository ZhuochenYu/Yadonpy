import json
from pathlib import Path

import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from yadonpy.core import as_rdkit_mol, molecular_weight, workdir, utils, poly
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.interface.charge_audit import format_cell_charge_audit
from yadonpy.io.gromacs_molecule import _format_gro_atom_line as format_single_gro_atom_line
from yadonpy.io.gromacs_system import _format_gro_atom_line as format_system_gro_atom_line
from yadonpy.io.gromacs_system import _load_gro_species_templates
from yadonpy.io.gromacs_system import export_system_from_cell_meta
from yadonpy.io.mol2 import write_mol2_from_rdkit
import yadonpy.sim.qm as qm_mod


def test_workdir_is_pathlike_and_non_destructive(tmp_path: Path):
    root = tmp_path / 'work'
    root.mkdir(parents=True)
    marker = root / 'keep.txt'
    marker.write_text('sentinel', encoding='utf-8')

    wd = workdir(root, restart=False)

    assert Path(wd) == root.resolve()
    assert (wd / 'keep.txt').exists()
    assert marker.read_text(encoding='utf-8') == 'sentinel'
    assert wd.metadata_path.exists()


def test_gro_atom_line_wraps_overflow_indices_without_shifting_coordinates():
    line = format_system_gro_atom_line(
        resnr=100000,
        resname='polymer',
        atomname='C1000',
        atomnr=100000,
        x=4.921,
        y=0.001,
        z=21.617,
    )

    assert len(line) == 44
    assert line[:5] == '    0'
    assert line[5:10] == 'polym'
    assert line[10:15] == 'C1000'
    assert line[15:20] == '    0'
    assert line[20:28] == '   4.921'
    assert line[28:36] == '   0.001'
    assert line[36:44] == '  21.617'


def test_single_molecule_gro_atom_line_uses_same_overflow_safe_format():
    line = format_single_gro_atom_line(
        resnr=1,
        resname='polymer',
        atomname='F1000',
        atomnr=100001,
        x=5.014,
        y=0.161,
        z=21.648,
    )

    assert len(line) == 44
    assert line[15:20] == '    1'
    assert line[20:28] == '   5.014'
    assert line[28:36] == '   0.161'
    assert line[36:44] == '  21.648'


def test_load_gro_species_templates_precomputes_atom_counts_once(tmp_path: Path):
    gro = tmp_path / 'mol.gro'
    gro.write_text(
        '\n'.join(
            [
                'mol',
                '    2',
                '    1MOL     C    1   0.100   0.200   0.300',
                '    1MOL     H    2   0.200   0.200   0.300',
                '   1.00000   1.00000   1.00000',
            ]
        ) + '\n',
        encoding='utf-8',
    )

    templates, nat_total = _load_gro_species_templates(
        [{'n': 3}],
        ['solvent_A'],
        [gro],
    )

    assert nat_total == 6
    assert len(templates) == 1
    assert templates[0].atom_names == ('C', 'H')
    assert templates[0].coords0.shape == (2, 3)


def test_random_walk_polymerization_reuses_current_poly_dmat_between_steps(monkeypatch):
    monomer = Chem.MolFromSmiles('CC')
    assert monomer is not None

    monkeypatch.setattr(poly, 'set_linker_flag', lambda *args, **kwargs: None)
    monkeypatch.setattr(poly, '_rw_finalize_bonded_terms', lambda mol: mol)
    monkeypatch.setattr(poly, '_rw_save', lambda *args, **kwargs: None)
    monkeypatch.setattr(poly, '_effective_restart_flag', lambda *args, **kwargs: False)
    monkeypatch.setattr(poly.calc, 'mirror_inversion_mol', lambda mol, confId=0: Chem.Mol(mol))
    monkeypatch.setattr(poly, '_prepare_connect_trial', lambda *args, **kwargs: {'poly_coord': np.zeros((1, 3)), 'mon_coord': np.zeros((1, 3)), 'keep_idx1': [0], 'keep_idx2': [0]})
    monkeypatch.setattr(poly, 'check_3d_structure_connect_trial', lambda *args, **kwargs: True)
    monkeypatch.setattr(poly, 'check_3d_structure_poly', lambda *args, **kwargs: True)
    monkeypatch.setattr(poly, '_materialize_connected_mols', lambda left, right, trial, **kwargs: Chem.CombineMols(left, right))

    orig_get_distance_matrix = Chem.GetDistanceMatrix
    calls = {'n': 0}

    def _counted_get_distance_matrix(mol, *args, **kwargs):
        calls['n'] += 1
        return orig_get_distance_matrix(mol, *args, **kwargs)

    monkeypatch.setattr(poly.Chem, 'GetDistanceMatrix', _counted_get_distance_matrix)

    out = poly.random_walk_polymerization(
        [monomer],
        [0, 0],
        [False, False],
        init_poly=Chem.Mol(monomer),
        dist_min=1.1,
        retry=1,
        rollback=1,
        retry_step=1,
        retry_opt_step=0,
    )

    assert out is not None
    # Expected calls with reuse:
    #   1 init_poly current dmat
    #   2 monomer template dmat + inverted monomer dmat
    #   2 accepted-step post-materialization dmat updates
    assert calls['n'] == 5


def test_resolved_molspec_is_accepted_by_poly_helpers(tmp_path: Path):
    ff = GAFF2_mod()
    spec = ff.mol('*CCO*', require_ready=False, prefer_db=False)
    ok = ff.ff_assign(spec, report=False)
    assert ok
    assert spec.resolved_mol is not None

    ter = utils.mol_from_smiles('[H][*]')
    dp = poly.calc_n_from_num_atoms(spec, 30, terminal1=ter)
    assert int(dp) >= 1

    wd = workdir(tmp_path / 'rw', restart=True)
    p1 = poly.polymerize_rw(spec, int(dp), tacticity='atactic', work_dir=wd, retry=1, retry_step=2, retry_opt_step=0)
    p2 = poly.polymerize_rw(spec, int(dp), tacticity='atactic', work_dir=wd, retry=1, retry_step=2, retry_opt_step=0)

    assert p1.GetNumAtoms() == p2.GetNumAtoms()
    cache_root = Path(wd) / '.yadonpy' / 'random_walk'
    assert cache_root.exists()


def test_as_rdkit_mol_unwraps_resolved_molspec_for_rdkit_descriptors():
    ff = GAFF2_mod()
    spec = ff.mol('CCO', require_ready=False, prefer_db=False)
    ok = ff.ff_assign(spec, report=False)

    assert ok
    rdkit_mol = as_rdkit_mol(spec, strict=True)
    assert rdkit_mol is spec.resolved_mol
    assert float(Descriptors.MolWt(rdkit_mol)) > 0.0


def test_as_rdkit_mol_prepares_merz_ion_for_rdkit_descriptors():
    ion = MERZ().mol('[Li+]')

    assert float(Descriptors.MolWt(as_rdkit_mol(ion, strict=True))) == pytest.approx(6.941, rel=1.0e-6)


def test_merz_ion_molwt_works_without_manual_cache_update():
    ion = MERZ().mol('[Na+]')

    assert float(Descriptors.MolWt(ion)) == pytest.approx(22.99, rel=1.0e-6)


def test_merz_mol_accepts_modern_name_and_ignores_moldb_style_kwargs():
    ion_ff = MERZ()
    ion = ion_ff.mol('[Li+]', name='Li', charge='RESP', prefer_db=True, require_ready=False)

    assert ion.GetNumAtoms() == 1
    assert ion.HasProp('name')
    assert ion.GetProp('name') == 'Li'
    assert ion.HasProp('_Name')
    assert ion.GetProp('_Name') == 'Li'
    assert ion.HasProp('mol_name')
    assert ion.GetProp('mol_name') == 'Li'
    assert ion.HasProp('merz_molecule_type')
    assert ion_ff.ff_assign(ion, report=False) is ion
    assert ion.GetProp('ff_name') == 'merz'


def test_mol_net_charge_recognizes_resp_only_atoms():
    mol = Chem.MolFromSmiles('CC')
    atom0 = mol.GetAtomWithIdx(0)
    atom1 = mol.GetAtomWithIdx(1)
    atom0.SetDoubleProp('RESP', 0.35)
    atom1.SetDoubleProp('RESP', -0.35)

    assert poly._mol_net_charge(mol) == pytest.approx(0.0, abs=1.0e-12)


def test_molecular_weight_helper_handles_merz_ions_and_unsanitized_hypervalent_species():
    li = MERZ().mol('[Li+]')
    pf6 = Chem.MolFromSmiles('F[P-](F)(F)(F)(F)F', sanitize=False)

    assert molecular_weight(li, strict=True) == pytest.approx(6.941, rel=1.0e-6)
    assert molecular_weight(pf6, strict=True) > 100.0


def test_forcefield_assignment_is_chainable_for_requested_script_style(tmp_path: Path):
    ff = GAFF2_mod()
    cation_ff = MERZ()

    monomer = ff.ff_assign(ff.mol('*CCO*', require_ready=False, prefer_db=False), report=False)
    assert monomer is not False
    assert hasattr(monomer, 'GetNumAtoms')
    assert monomer.GetNumAtoms() >= 3

    dp = poly.calc_n_from_num_atoms(monomer, 60, terminal1=utils.mol_from_smiles('[H][*]'))
    assert int(dp) >= 1

    cation = cation_ff.ff_assign(cation_ff.mol('[Li+]'), report=False)
    assert cation is not False
    assert cation.GetNumAtoms() == 1

    out_path = write_mol2_from_rdkit(mol=cation, out_dir=tmp_path / '00_molecules')
    assert out_path.exists()
    assert out_path.parent == (tmp_path / '00_molecules')


def test_pf6_can_roundtrip_through_moldb_and_direct_molspec_api_with_gasteiger_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setenv('YADONPY_MOLDB', str(tmp_path / 'moldb'))

    ff = GAFF2_mod()
    pf6_smiles = 'F[P-](F)(F)(F)(F)F'
    built = utils.mol_from_smiles(pf6_smiles, coord=False, name='PF6')
    utils.ensure_3d_coords(built, smiles_hint=pf6_smiles, engine='openbabel')

    built = ff.ff_assign(built, charge='gasteiger', bonded='DRIH', report=False)
    assert built is not False

    record = ff.store_to_db(built, smiles_or_psmiles=pf6_smiles, name='PF6', charge='gasteiger')
    assert record.ready is True

    spec = ff.mol(pf6_smiles, charge='gasteiger')
    loaded = ff.ff_assign(spec, bonded='DRIH', report=False)

    assert loaded is not False
    assert loaded.GetNumAtoms() == built.GetNumAtoms()
    assert loaded.HasProp('_YADONPY_KEY')
    assert loaded.GetProp('_YADONPY_KEY') == record.key
    assert loaded.GetAtomWithIdx(0).HasProp('AtomicCharge')


def test_amorphous_cell_explicit_cell_keeps_charge_meta_when_density_is_none(tmp_path: Path):
    mol = utils.mol_from_smiles('O', coord=True, name='water_like')
    cell = Chem.Mol()
    setattr(cell, 'cell', utils.Cell(3.0, 0.0, 3.0, 0.0, 3.0, 0.0))

    ac = poly.amorphous_cell(
        [mol],
        [1],
        cell=cell,
        density=None,
        retry=1,
        retry_step=4,
        threshold=1.0,
        neutralize=False,
        work_dir=tmp_path / 'explicit_cell_pack',
    )

    assert ac.HasProp('_yadonpy_cell_meta')
    meta = json.loads(ac.GetProp('_yadonpy_cell_meta'))
    assert meta['density_g_cm3'] is None
    assert meta['species'][0]['n'] == 1
    assert 'raw=' in format_cell_charge_audit('explicit cell', ac)


def test_export_system_accepts_explicit_cell_density_none(tmp_path: Path):
    ff = GAFF2_mod()
    mol = ff.mol('O', require_ready=False, prefer_db=False, name='water_like')
    assert ff.ff_assign(mol, report=False)

    cell = Chem.Mol()
    setattr(cell, 'cell', utils.Cell(3.0, 0.0, 3.0, 0.0, 3.0, 0.0))

    ac = poly.amorphous_cell(
        [mol],
        [1],
        cell=cell,
        density=None,
        retry=1,
        retry_step=4,
        threshold=1.0,
        neutralize=False,
        work_dir=tmp_path / 'explicit_cell_export',
    )

    out = export_system_from_cell_meta(
        cell_mol=ac,
        out_dir=tmp_path / 'sys',
        ff_name=ff.name,
        charge_method='RESP',
        write_system_mol2=False,
    )

    assert out.system_top.exists()
    assert out.system_gro.exists()
    assert out.box_nm >= 2.0


def test_gaff_bonded_override_default_cache_avoids_legacy_workdir_folder(tmp_path: Path, monkeypatch):
    ff = GAFF2_mod()
    mol = utils.mol_from_smiles('CCO', coord=True, name='ethanol')

    def _ok(*args, **kwargs):
        return True

    monkeypatch.setattr(ff, 'assign_ptypes', _ok)
    monkeypatch.setattr(ff, 'assign_btypes', _ok)
    monkeypatch.setattr(ff, 'assign_atypes', _ok)
    monkeypatch.setattr(ff, 'assign_dtypes', _ok)
    monkeypatch.setattr(ff, 'assign_itypes', _ok)

    captured = {}

    def _fake_drih(mol_obj, *, work_dir, log_name, **kwargs):
        work_root = Path(work_dir)
        captured['work_dir'] = work_root
        task_dir = work_root / '01_qm' / '07_bonded_params' / str(log_name)
        task_dir.mkdir(parents=True, exist_ok=True)
        itp = task_dir / 'bonded_drih_patch.itp'
        js = task_dir / 'bonded_drih_params.json'
        itp.write_text('; fake bonded patch\n', encoding='utf-8')
        js.write_text('{"meta": {"method": "DRIH"}, "bonds": [], "angles": []}\n', encoding='utf-8')
        mol_obj.SetProp('_yadonpy_bonded_itp', str(itp.resolve()))
        mol_obj.SetProp('_yadonpy_bonded_json', str(js.resolve()))
        return {'itp': str(itp.resolve()), 'json': str(js.resolve())}

    monkeypatch.setattr(qm_mod, 'bond_angle_params_drih', _fake_drih)
    monkeypatch.chdir(tmp_path)

    out = ff.ff_assign(mol, bonded='DRIH', report=False)

    assert out is not False
    assert '.yadonpy_cache' in str(captured['work_dir'])
    assert 'bonded_params' not in str(captured['work_dir']).lower()
    assert not (tmp_path / 'work_dir' / 'bonded_params').exists()


def test_manual_bulk_workflow_style_builds_polymer_and_cell_without_wf(tmp_path: Path):
    ff = GAFF2_mod()
    ion_ff = MERZ()
    wd = workdir(tmp_path / 'manual_bulk', restart=True)

    monomer_A = ff.ff_assign(ff.mol('*CCO*', require_ready=False, prefer_db=False, name='monomer_A'), report=False)
    monomer_B = ff.ff_assign(ff.mol('*COC*', require_ready=False, prefer_db=False, name='monomer_B'), report=False)
    ter1 = utils.mol_from_smiles('[H][*]')
    solvent_A = ff.ff_assign(ff.mol('CCOC(=O)OC', require_ready=False, prefer_db=False, name='solvent_A'), report=False)
    solvent_B = ff.ff_assign(ff.mol('O=C1OCCO1', require_ready=False, prefer_db=False, name='solvent_B'), report=False)
    cation_A = ion_ff.ff_assign(ion_ff.mol('[Li+]'), report=False)

    assert monomer_A is not False
    assert monomer_B is not False
    assert solvent_A is not False
    assert solvent_B is not False
    assert cation_A is not False

    dp = max(1, int(poly.calc_n_from_num_atoms([monomer_A, monomer_B], 40, ratio=[0.5, 0.5], terminal1=ter1)))
    copoly = poly.random_copolymerize_rw(
        [monomer_A, monomer_B],
        dp,
        ratio=[0.5, 0.5],
        tacticity='atactic',
        name='copoly',
        work_dir=wd.child('copoly_rw'),
        retry=2,
        retry_step=4,
        retry_opt_step=0,
    )
    copoly = poly.terminate_rw(copoly, ter1, name='copoly', work_dir=wd.child('copoly_term'))
    assert ff.ff_assign(copoly, report=False) is not False

    ac = poly.amorphous_cell(
        [copoly, solvent_A, solvent_B, cation_A],
        [1, 2, 2, 1],
        charge_scale=[1.0, 1.0, 1.0, 0.8],
        density=0.05,
        work_dir=wd.child('00_build_cell'),
        retry=1,
        retry_step=20,
    )

    assert ac.GetNumAtoms() > copoly.GetNumAtoms()
    assert (Path(wd.child('copoly_rw')) / '.yadonpy' / 'random_walk').exists()



def test_molspec_names_roundtrip_into_cell_meta_and_system_top(tmp_path: Path):
    import json

    from yadonpy.core import naming
    from yadonpy.io.gromacs_system import export_system_from_cell_meta

    ff = GAFF2_mod()

    solvent_A = ff.mol('CCO', require_ready=False, prefer_db=False)
    solvent_B = ff.mol('CC', require_ready=False, prefer_db=False)
    solvent_C = ff.mol('CO', require_ready=False, prefer_db=False)
    solvent_D = ff.mol('CCOC', require_ready=False, prefer_db=False)

    for sp in (solvent_A, solvent_B, solvent_C, solvent_D):
        assert ff.ff_assign(sp, report=False)
        assert sp.resolved_mol is not None

    assert naming.get_name(solvent_A.resolved_mol) == 'solvent_A'
    assert naming.get_name(solvent_B.resolved_mol) == 'solvent_B'
    assert naming.get_name(solvent_C.resolved_mol) == 'solvent_C'
    assert naming.get_name(solvent_D.resolved_mol) == 'solvent_D'

    ac = poly.amorphous_cell(
        [solvent_A, solvent_B, solvent_C, solvent_D],
        [1, 1, 1, 1],
        density=0.2,
        retry=1,
        retry_step=20,
    )
    meta = json.loads(ac.GetProp('_yadonpy_cell_meta'))
    names = [sp['name'] for sp in meta['species']]
    assert names == ['solvent_A', 'solvent_B', 'solvent_C', 'solvent_D']
    assert 'spec' not in names

    out = export_system_from_cell_meta(
        cell_mol=ac,
        out_dir=tmp_path / 'sys',
        ff_name=ff.name,
        charge_method='RESP',
    )
    top_text = out.system_top.read_text(encoding='utf-8')
    assert 'molecules/spec/spec.itp' not in top_text
    for molname in ('solvent_A', 'solvent_B', 'solvent_C', 'solvent_D'):
        assert f'molecules/{molname}/{molname}.itp' in top_text


def test_gaff2_bridge_oxygen_prefers_os_even_with_unsanitized_h_count():
    from yadonpy.ff import GAFF2

    mol = Chem.MolFromSmiles('C[O](C)[H]', sanitize=False)
    assert mol is not None
    oxygen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')

    ff = GAFF2()
    assert ff.assign_ptypes_atom(oxygen)
    assert oxygen.GetProp('ff_type') == 'os'


def test_export_system_fast_path_reuses_raw_artifacts_and_box_files(tmp_path: Path, monkeypatch):
    import yadonpy.io.gromacs_system as gsys

    ff = GAFF2_mod()
    solvent = ff.mol('CCO', require_ready=False, prefer_db=False)
    salt = ff.mol('CO', require_ready=False, prefer_db=False)

    assert ff.ff_assign(solvent, report=False)
    assert ff.ff_assign(salt, report=False)

    ac = poly.amorphous_cell(
        [solvent, salt],
        [3, 2],
        density=0.05,
        retry=1,
        retry_step=20,
    )

    raw = gsys.export_system_from_cell_meta(
        cell_mol=ac,
        out_dir=tmp_path / 'raw',
        ff_name=ff.name,
        charge_method='RESP',
        charge_scale=1.0,
        write_system_mol2=False,
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError('fast-path export should reuse existing molecule artifacts without FF/MolDB regeneration')

    monkeypatch.setattr(gsys, 'write_molecule_artifacts', _unexpected)
    monkeypatch.setattr(gsys, 'get_ff', _unexpected)

    scaled = gsys.export_system_from_cell_meta(
        cell_mol=ac,
        out_dir=tmp_path / 'scaled',
        ff_name=ff.name,
        charge_method='RESP',
        charge_scale=0.8,
        source_molecules_dir=raw.molecules_dir,
        system_gro_template=raw.system_gro,
        system_ndx_template=raw.system_ndx,
        write_system_mol2=False,
    )

    assert scaled.system_gro.read_text(encoding='utf-8') == raw.system_gro.read_text(encoding='utf-8')
    assert scaled.system_ndx.read_text(encoding='utf-8') == raw.system_ndx.read_text(encoding='utf-8')
    assert scaled.box_nm == pytest.approx(raw.box_nm)

    moltype = str(next(sp['moltype'] for sp in scaled.species if int(sp['n']) > 0))
    scaled_itp = (scaled.molecules_dir / moltype / f'{moltype}.itp').read_text(encoding='utf-8')
    raw_itp = (raw.molecules_dir / moltype / f'{moltype}.itp').read_text(encoding='utf-8')
    assert scaled_itp != raw_itp


def test_export_system_resolves_mixed_forcefield_species_from_cell_metadata(tmp_path: Path):
    from yadonpy.io.gromacs_system import export_system_from_cell_meta

    ff = GAFF2_mod()
    ion_ff = MERZ()

    solvent = ff.mol('CCO', require_ready=False, prefer_db=False, name='solvent_A')
    lithium = ion_ff.mol('[Li+]')

    assert ff.ff_assign(solvent, report=False)
    assert ion_ff.ff_assign(lithium, report=False)

    ac = poly.amorphous_cell(
        [solvent, lithium],
        [2, 1],
        density=0.2,
        retry=1,
        retry_step=20,
        neutralize=False,
    )

    out = export_system_from_cell_meta(
        cell_mol=ac,
        out_dir=tmp_path / 'mixed_ff',
        ff_name=ff.name,
        charge_method='RESP',
        write_system_mol2=False,
    )

    top_text = out.system_top.read_text(encoding='utf-8')
    assert out.system_top.exists()
    assert out.system_gro.exists()
    assert 'solvent_A' in top_text
    assert 'lithium' in top_text
    assert (out.molecules_dir / 'lithium' / 'lithium.itp').exists()


def test_export_system_normalizes_embedded_parameter_blocks_with_compact_moleculetype_headers(tmp_path: Path):
    from yadonpy.io.gromacs_system import export_system_from_cell_meta

    ff = GAFF2_mod()
    solvent = ff.mol('CCO', require_ready=False, prefer_db=False, name='solvent_A')
    assert ff.ff_assign(solvent, report=False)

    ac = poly.amorphous_cell(
        [solvent],
        [2],
        density=0.2,
        retry=1,
        retry_step=20,
    )

    raw = export_system_from_cell_meta(
        cell_mol=ac,
        out_dir=tmp_path / 'raw',
        ff_name=ff.name,
        charge_method='RESP',
        write_system_mol2=False,
    )

    moltype = str(next(sp['moltype'] for sp in raw.species if int(sp['n']) > 0))
    raw_itp_path = raw.molecules_dir / moltype / f'{moltype}.itp'
    raw_itp_path.write_text(
        (
            '; embedded params\n'
            '[ atomtypes ]\n'
            'c3 12.0110 0.0000 A 0.339967 0.457730\n\n'
            '[moleculetype]\n'
            f'{moltype} 3\n\n'
            '[atoms]\n'
            f'1  c3  1  {moltype}  C1  1  0.0000  12.011\n'
        ),
        encoding='utf-8',
    )

    scaled = export_system_from_cell_meta(
        cell_mol=ac,
        out_dir=tmp_path / 'scaled',
        ff_name=ff.name,
        charge_method='RESP',
        charge_scale=0.8,
        source_molecules_dir=raw.molecules_dir,
        system_gro_template=raw.system_gro,
        system_ndx_template=raw.system_ndx,
        write_system_mol2=False,
    )

    top_text = scaled.system_top.read_text(encoding='utf-8')
    norm_itp = (scaled.molecules_dir / moltype / f'{moltype}.itp').read_text(encoding='utf-8')
    ff_params = (scaled.system_top.parent / 'ff_parameters.itp').read_text(encoding='utf-8')

    assert '#include "ff_parameters.itp"' in top_text
    assert top_text.index('#include "ff_parameters.itp"') < top_text.index(f'#include "molecules/{moltype}/{moltype}.itp"')
    assert '[ atomtypes ]' in ff_params
    assert '[ atomtypes ]' not in norm_itp
    assert norm_itp.lstrip().startswith('[moleculetype]')


def test_export_system_fast_path_reuses_raw_ff_parameters_include(tmp_path: Path):
    from yadonpy.io.gromacs_system import export_system_from_cell_meta

    ff = GAFF2_mod()
    solvent = ff.mol('CCO', require_ready=False, prefer_db=False, name='solvent_A')
    assert ff.ff_assign(solvent, report=False)

    ac = poly.amorphous_cell([solvent], [2], density=0.2, retry=1, retry_step=20)
    raw = export_system_from_cell_meta(
        cell_mol=ac,
        out_dir=tmp_path / 'raw',
        ff_name=ff.name,
        charge_method='RESP',
        write_system_mol2=False,
    )
    (raw.system_top.parent / 'ff_parameters.itp').write_text(
        '; synthetic shared parameters\n[ atomtypes ]\nc3 12.0110 0.0000 A 0.339967 0.457730\n',
        encoding='utf-8',
    )

    scaled = export_system_from_cell_meta(
        cell_mol=ac,
        out_dir=tmp_path / 'scaled',
        ff_name=ff.name,
        charge_method='RESP',
        charge_scale=0.8,
        source_molecules_dir=raw.molecules_dir,
        system_gro_template=raw.system_gro,
        system_ndx_template=raw.system_ndx,
        write_system_mol2=False,
    )

    top_text = scaled.system_top.read_text(encoding='utf-8')
    ff_params = (scaled.system_top.parent / 'ff_parameters.itp').read_text(encoding='utf-8')
    assert '#include "ff_parameters.itp"' in top_text
    assert '[ atomtypes ]' in ff_params


def test_eq21_export_pair_skips_system_mol2_for_large_box_fast_path(tmp_path: Path, monkeypatch):
    import yadonpy.sim.preset.eq as eqmod
    from yadonpy.io.gromacs_system import SystemExportResult

    ff = GAFF2_mod()
    solvent = ff.mol('CCO', require_ready=False, prefer_db=False)
    assert ff.ff_assign(solvent, report=False)
    ac = poly.amorphous_cell([solvent], [2], density=0.2, retry=1, retry_step=20)

    calls: list[bool] = []

    def _fake_export_system_from_cell_meta(**kwargs):
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        mol_dir = out_dir / 'molecules'
        mol_dir.mkdir(parents=True, exist_ok=True)
        calls.append(bool(kwargs.get('write_system_mol2', True)))
        for name in ('system.gro', 'system.top', 'system.ndx', 'system_meta.json'):
            (out_dir / name).write_text('', encoding='utf-8')
        return SystemExportResult(
            system_gro=out_dir / 'system.gro',
            system_top=out_dir / 'system.top',
            system_ndx=out_dir / 'system.ndx',
            molecules_dir=mol_dir,
            system_meta=out_dir / 'system_meta.json',
            box_nm=1.0,
            species=[],
        )

    monkeypatch.setattr(eqmod, 'export_system_from_cell_meta', _fake_export_system_from_cell_meta)

    job = eqmod.EQ21step(ac, work_dir=tmp_path / 'eq')
    job._ensure_system_exported()

    assert calls == [False, False]


def test_eq21_rebuilds_invalid_cached_system_export(tmp_path: Path, monkeypatch):
    import yadonpy.sim.preset.eq as eqmod
    from yadonpy.io.gromacs_system import SystemExportResult

    ff = GAFF2_mod()
    solvent = ff.mol('CCO', require_ready=False, prefer_db=False)
    assert ff.ff_assign(solvent, report=False)
    ac = poly.amorphous_cell([solvent], [1], density=0.2, retry=1, retry_step=20)

    sys_root = tmp_path / 'eq' / '02_system'
    raw_root = sys_root / '01_raw_non_scaled'
    for root in (sys_root, raw_root):
        mol_dir = root / 'molecules'
        mol_dir.mkdir(parents=True, exist_ok=True)
        (root / 'system.gro').write_text('', encoding='utf-8')
        (root / 'system.ndx').write_text('', encoding='utf-8')
        (root / 'system_meta.json').write_text('{}\n', encoding='utf-8')
        (root / 'export_manifest.json').write_text('{}\n', encoding='utf-8')
        (mol_dir / 'SOL.itp').write_text('[ moleculetype ]\nSOL 3\n', encoding='utf-8')
        (root / 'system.top').write_text('#include "molecules/SOL.itp"\n\n[ system ]\ninvalid\n\n[ molecules ]\nSOL 1\n', encoding='utf-8')

    calls: list[Path] = []

    def _fake_export_system_from_cell_meta(**kwargs):
        out_dir = Path(kwargs['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        mol_dir = out_dir / 'molecules'
        mol_dir.mkdir(parents=True, exist_ok=True)
        calls.append(out_dir)
        (out_dir / 'system.gro').write_text('', encoding='utf-8')
        (out_dir / 'system.ndx').write_text('', encoding='utf-8')
        (out_dir / 'system_meta.json').write_text('{}\n', encoding='utf-8')
        (mol_dir / 'SOL.itp').write_text('[ moleculetype ]\nSOL 3\n', encoding='utf-8')
        (out_dir / 'system.top').write_text('[ defaults ]\n1 2 yes 0.5 0.8333333333\n\n#include "molecules/SOL.itp"\n\n[ system ]\nvalid\n\n[ molecules ]\nSOL 1\n', encoding='utf-8')
        return SystemExportResult(
            system_gro=out_dir / 'system.gro',
            system_top=out_dir / 'system.top',
            system_ndx=out_dir / 'system.ndx',
            molecules_dir=mol_dir,
            system_meta=out_dir / 'system_meta.json',
            box_nm=1.0,
            species=[],
        )

    monkeypatch.setattr(eqmod, 'export_system_from_cell_meta', _fake_export_system_from_cell_meta)

    job = eqmod.EQ21step(ac, work_dir=tmp_path / 'eq')
    exp = job.ensure_system_exported()

    assert len(calls) == 2
    assert exp.system_top.read_text(encoding='utf-8').startswith('[ defaults ]')


def test_exported_system_charge_correction_rewrites_copied_itp(tmp_path: Path):
    from yadonpy.io.gromacs_system import _itp_total_charge, _neutralize_exported_system_charge

    mol_a = tmp_path / 'CMC.itp'
    mol_b = tmp_path / 'Na.itp'
    mol_a.write_text(
        """
[ moleculetype ]
CMC 3

[ atoms ]
1  C   1  CMC  C1  1  -0.00040000  12.011
2  H   1  CMC  H1  1   0.00000000   1.008
""".strip() + "\n",
        encoding='utf-8',
    )
    mol_b.write_text(
        """
[ moleculetype ]
Na 3

[ atoms ]
1  Na  1  Na  Na1  1   1.00000000  22.990
""".strip() + "\n",
        encoding='utf-8',
    )

    info = _neutralize_exported_system_charge(
        species=[{'n': 50}, {'n': 0}],
        mol_names=['CMC', 'Na'],
        mol_itp_paths=[mol_a, mol_b],
    )

    assert info is not None
    assert info['target_moltype'] == 'CMC'
    assert abs(info['system_charge_before'] + 0.02) < 1.0e-9
    assert abs(info['system_charge_after']) < 1.0e-9
    assert abs(_itp_total_charge(mol_a.read_text(encoding='utf-8'))) < 1.0e-9


def test_amorphous_cell_reuses_cached_result_on_restart(tmp_path: Path, monkeypatch):
    ff = GAFF2_mod()
    spec = ff.mol('CCO', require_ready=False, prefer_db=False)
    assert ff.ff_assign(spec, report=False)

    wd = workdir(tmp_path / 'cell_cache', restart=True)
    ac1 = poly.amorphous_cell([spec], [2], density=0.2, retry=1, retry_step=20, work_dir=wd)

    def _unexpected_check(*args, **kwargs):
        raise AssertionError('placement should be skipped when amorphous_cell cache is reused')

    monkeypatch.setattr(poly, 'check_3d_structure_cell', _unexpected_check)
    ac2 = poly.amorphous_cell([spec], [2], density=0.2, retry=1, retry_step=20, work_dir=wd)

    assert ac1.GetNumAtoms() == ac2.GetNumAtoms()
    assert hasattr(ac2, 'cell')
    assert float(ac1.cell.dx) == pytest.approx(float(ac2.cell.dx))
    assert float(ac1.cell.dy) == pytest.approx(float(ac2.cell.dy))
    assert float(ac1.cell.dz) == pytest.approx(float(ac2.cell.dz))
    cache_root = Path(wd) / '.yadonpy' / 'random_walk'
    assert any(path.name.endswith('.state.json') for path in cache_root.iterdir())


def test_amorphous_cell_restart_logs_skip_label(tmp_path: Path, monkeypatch):
    ff = GAFF2_mod()
    spec = ff.mol('CCO', require_ready=False, prefer_db=False)
    assert ff.ff_assign(spec, report=False)

    wd = workdir(tmp_path / 'cell_cache_labels', restart=True)
    poly.amorphous_cell([spec], [1], density=0.2, retry=1, retry_step=10, work_dir=wd)

    logs: list[str] = []

    def _capture(msg, *args, **kwargs):
        logs.append(str(msg))

    monkeypatch.setattr(poly.utils, 'radon_print', _capture)
    poly.amorphous_cell([spec], [1], density=0.2, retry=1, retry_step=10, work_dir=wd)

    assert any('[SKIP] poly.amorphous_cell: restored cached result' in msg for msg in logs)
    assert not any('[PACK] Retry placing a molecule in cell.' in msg for msg in logs)


def test_amorphous_cell_uses_stable_cache_when_hashed_restart_cache_misses(tmp_path: Path, monkeypatch):
    ff = GAFF2_mod()
    spec = ff.mol('CCO', require_ready=False, prefer_db=False)
    assert ff.ff_assign(spec, report=False)

    wd = workdir(tmp_path / 'cell_cache_stable', restart=True)
    ac1 = poly.amorphous_cell([spec], [2], density=0.2, retry=1, retry_step=20, work_dir=wd)

    monkeypatch.setattr(poly, '_rw_load', lambda *args, **kwargs: None)

    def _unexpected_check(*args, **kwargs):
        raise AssertionError('placement should be skipped when stable amorphous_cell cache is reused')

    monkeypatch.setattr(poly, 'check_3d_structure_cell', _unexpected_check)
    ac2 = poly.amorphous_cell([spec], [2], density=0.2, retry=1, retry_step=20, work_dir=wd)

    assert ac1.GetNumAtoms() == ac2.GetNumAtoms()
    assert hasattr(ac2, 'cell')


def test_ff_assign_surfaces_molspec_resolution_errors_instead_of_kekulize_type_errors(monkeypatch):
    ff = GAFF2_mod()
    spec = ff.mol('F[P-](F)(F)(F)(F)F')

    def _boom(*args, **kwargs):
        raise RuntimeError('Failed to read mol2: broken PF6 entry')

    monkeypatch.setattr(ff, 'mol_rdkit', _boom)

    with pytest.raises(RuntimeError, match='Failed to read mol2: broken PF6 entry'):
        ff.ff_assign(spec, bonded='DRIH', report=False)


def test_next_amorphous_retry_target_reduces_density_when_density_is_set():
    target = poly._next_amorphous_retry_target(cell=None, density=0.4, dec_rate=0.8)

    assert target['cell'] is None
    assert target['density'] == pytest.approx(0.32)
    assert 'density is reduced' in target['log']


def test_next_amorphous_retry_target_expands_only_z_for_fixed_cell():
    cell = Chem.Mol()
    setattr(cell, 'cell', utils.Cell(3.7, 0.0, 3.7, 0.0, 5.3, 0.0))

    target = poly._next_amorphous_retry_target(cell=cell, density=None, dec_rate=0.8)

    assert target['density'] is None
    assert hasattr(target['cell'], 'cell')
    assert float(target['cell'].cell.dx) == pytest.approx(3.7)
    assert float(target['cell'].cell.dy) == pytest.approx(3.7)
    assert float(target['cell'].cell.dz) == pytest.approx(5.3 / 0.8)
    assert 'keeping XY fixed' in target['log']


def test_check_3d_proximity_respects_dmat_mask_for_cross_distance_checks():
    coord1 = np.asarray([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    coord2 = np.asarray([
        [0.5, 0.0, 0.0],
    ])

    assert poly.check_3d_proximity(coord1, coord2=coord2, dist_min=1.0) is False
    assert poly.check_3d_proximity(
        coord1,
        coord2=coord2,
        dist_min=1.0,
        dmat=np.asarray([[1], [4]]),
        ignore_rad=3,
    ) is True


def test_rw_emit_heartbeat_and_step_progress_logs(monkeypatch):
    logs: list[str] = []

    def _capture(msg, *args, **kwargs):
        logs.append(str(msg))

    monkeypatch.setattr(poly.utils, 'radon_print', _capture)
    monkeypatch.setattr(poly.const, 'rw_heartbeat_seconds', 5.0)
    monkeypatch.setattr(poly.const, 'tqdm_disable', True)

    state = {'last_heartbeat': 0.0, 'step_interval': 10}
    poly._rw_emit_heartbeat(
        step_idx=12,
        total_steps=500,
        retry_idx=41,
        retry_total=360,
        step_started_at=1.0,
        walk_started_at=0.0,
        state=state,
        now=7.5,
    )
    poly._rw_emit_step_progress(
        step_idx=20,
        total_steps=500,
        retries_used=7,
        step_started_at=2.0,
        walk_started_at=0.0,
        state=state,
        now=9.0,
    )

    assert any('[RW] progress step 12/500' in msg for msg in logs)
    assert any('[RW] accepted step 20/500' in msg for msg in logs)


def test_bond_triangle_bbox_pairs_filters_far_apart_combinations(monkeypatch):
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def _capture(bond, tri):
        calls.append((np.asarray(bond), np.asarray(tri)))
        return False

    monkeypatch.setattr(poly, 'MollerTrumbore', _capture)

    bonds = np.asarray([
        [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        [[100.0, 100.0, 100.0], [101.0, 100.0, 100.0]],
    ])
    tris = np.asarray([
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        [[-100.0, -100.0, -100.0], [-99.0, -100.0, -100.0], [-100.0, -99.0, -100.0]],
    ])

    assert poly._has_bond_triangle_intersection(bonds, tris, mp=0) is False
    assert len(calls) == 1


def test_rdkit_clone_mol_preserves_distance_matrix_inputs():
    mol = Chem.AddHs(Chem.MolFromSmiles('CC'))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0

    cloned = poly._rdkit_clone_mol(mol)

    assert isinstance(cloned, Chem.Mol)
    assert cloned.GetNumAtoms() == mol.GetNumAtoms()
    assert np.asarray(Chem.GetDistanceMatrix(cloned)).shape == np.asarray(Chem.GetDistanceMatrix(mol)).shape


def test_connect_trial_cross_dmat_matches_materialized_polymer():
    ff = GAFF2_mod()
    monomer = ff.ff_assign(ff.mol('*CCO*', require_ready=False, prefer_db=False), report=False)
    assert monomer is not False

    poly_mol = utils.deepcopy_mol(monomer)
    mon_mol = utils.deepcopy_mol(monomer)
    trial = poly._prepare_connect_trial(poly_mol, mon_mol, random_rot=False)

    assert trial is not None
    cross_dmat = poly._connect_trial_cross_dmat(
        poly_mol,
        mon_mol,
        trial,
        poly_dmat=np.asarray(Chem.GetDistanceMatrix(poly_mol)),
        mon_dmat=np.asarray(Chem.GetDistanceMatrix(mon_mol)),
    )

    connected = poly.connect_mols(utils.deepcopy_mol(poly_mol), utils.deepcopy_mol(mon_mol), random_rot=False)
    assert connected is not None
    connected_dmat = np.asarray(Chem.GetDistanceMatrix(connected))
    n_old = len(trial['keep_idx1'])

    assert cross_dmat.shape == connected_dmat[:n_old, n_old:].shape
    assert np.array_equal(cross_dmat, connected_dmat[:n_old, n_old:])


def test_ring_intersection_accepts_explicit_monomer_coords_without_wrap():
    poly_mol = Chem.AddHs(Chem.MolFromSmiles('c1ccccc1'))
    mon_mol = Chem.AddHs(Chem.MolFromSmiles('C1CC1'))

    assert poly_mol is not None
    assert mon_mol is not None
    assert AllChem.EmbedMolecule(poly_mol, randomSeed=0xAA01) == 0
    assert AllChem.EmbedMolecule(mon_mol, randomSeed=0xAA02) == 0

    check, tri_coord, bond_coord = poly.check_3d_bond_ring_intersection(
        poly_mol,
        mon=mon_mol,
        poly_coord=np.asarray(poly_mol.GetConformer().GetPositions()),
        mon_coord=np.asarray(mon_mol.GetConformer().GetPositions()) + np.array([5.0, 0.0, 0.0]),
    )

    assert isinstance(check, bool)
    assert tri_coord is not None
    assert bond_coord is not None


def test_ring_intersection_accepts_trial_coords_with_deleted_linker_atoms():
    ff = GAFF2_mod()
    monomer_spec = ff.mol(r'*C1=C(F)C(F)=C(C2=C(F)C(F)=C(*)C(F)=C2F)C(F)=C1F', require_ready=False, prefer_db=False)

    assert ff.ff_assign(monomer_spec, report=False) is not False
    monomer = as_rdkit_mol(monomer_spec, strict=True)

    trial = poly._prepare_connect_trial(monomer, monomer, random_rot=False)

    assert trial is not None
    assert len(trial['poly_coord']) == monomer.GetNumAtoms() - 1
    assert len(trial['mon_coord']) == monomer.GetNumAtoms() - 1

    check, tri_coord, bond_coord = poly.check_3d_bond_ring_intersection(
        monomer,
        mon=monomer,
        poly_coord=trial['poly_coord'],
        mon_coord=trial['mon_coord'],
        poly_atom_indices=trial['keep_idx1'],
        mon_atom_indices=trial['keep_idx2'],
    )

    assert isinstance(check, bool)
    assert tri_coord is not None
    assert bond_coord is not None


def test_rw_finalize_bonded_terms_restores_lightweight_clone_metadata():
    ff = GAFF2_mod()
    monomer = ff.ff_assign(ff.mol('*CCO*', require_ready=False, prefer_db=False), report=False)
    assert monomer is not False
    assert bool(getattr(monomer, 'angles', {}) or {})

    clone = poly._rw_growth_clone_mol(monomer)
    assert not bool(getattr(clone, 'angles', {}) or {})

    restored = poly._rw_finalize_bonded_terms(clone)
    assert bool(getattr(restored, 'angles', {}) or {})


def test_terminate_rw_recovers_missing_residue_info_from_cached_polymer(monkeypatch):
    poly_mol = Chem.MolFromSmiles('CC')
    assert poly_mol is not None
    assert all(atom.GetPDBResidueInfo() is None for atom in poly_mol.GetAtoms())

    poly_mol.SetIntProp('head_idx', 0)
    poly_mol.SetIntProp('tail_idx', 1)
    poly_mol.SetIntProp('num_units', 2)

    terminal = utils.mol_from_smiles('[H][3H]')
    monkeypatch.setattr(poly, 'set_terminal_idx', lambda mol: None)

    out = poly.terminate_rw(poly_mol, terminal, restart=False)

    head_info = out.GetAtomWithIdx(0).GetPDBResidueInfo()
    tail_info = out.GetAtomWithIdx(1).GetPDBResidueInfo()
    assert head_info is not None
    assert tail_info is not None
    assert head_info.GetResidueName().strip() == 'TU0'
    assert tail_info.GetResidueName().strip() == 'TU1'
    assert head_info.GetResidueNumber() == 3
    assert tail_info.GetResidueNumber() == 4
    assert out.GetIntProp('num_units') == 4


def test_check_3d_structure_poly_limits_proximity_to_local_candidates(monkeypatch):
    chain = Chem.MolFromSmiles('CCCCCC')
    mon = Chem.MolFromSmiles('CC')

    assert chain is not None
    assert mon is not None
    assert AllChem.EmbedMolecule(chain, randomSeed=0xF00D) == 0

    conf = chain.GetConformer()
    coords = [
        (0.0, 0.0, 0.0),
        (1.2, 0.0, 0.0),
        (100.2, 0.0, 0.0),
        (101.3, 0.0, 0.0),
        (100.0, 0.0, 0.0),
        (101.0, 0.0, 0.0),
    ]
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, (x, y, z))

    seen = {}

    def _fake_check(coord1, coord2=None, dist_min=1.5, wrap=None, ignore_rad=3, dmat=None):
        seen['poly_atoms_checked'] = len(coord1)
        seen['mon_atoms_checked'] = 0 if coord2 is None else len(coord2)
        return True

    monkeypatch.setattr(poly, 'check_3d_proximity', _fake_check)

    assert poly.check_3d_structure_poly(chain, mon, dist_min=0.7)
    assert seen['poly_atoms_checked'] == 2
    assert seen['mon_atoms_checked'] == 2


def test_resolve_rw_retry_budget_scales_with_rigidity():
    flexible = Chem.MolFromSmiles('CCCCCCCC')
    rigid = Chem.MolFromSmiles('c1ccc(cc1)c2ccccc2')

    assert flexible is not None
    assert rigid is not None

    flexible_budget = poly._resolve_rw_retry_budget(
        [flexible],
        retry=100,
        rollback=5,
        rollback_shaking=False,
        retry_step=200,
        retry_opt_step=20,
    )
    rigid_budget = poly._resolve_rw_retry_budget(
        [rigid],
        retry=100,
        rollback=5,
        rollback_shaking=False,
        retry_step=200,
        retry_opt_step=20,
    )

    assert flexible_budget['retry'] <= 40
    assert flexible_budget['rollback'] <= 3
    assert flexible_budget['retry_step'] <= 60
    assert flexible_budget['retry_opt_step'] <= 2
    assert rigid_budget['retry_step'] > flexible_budget['retry_step']
    assert rigid_budget['retry_opt_step'] > flexible_budget['retry_opt_step']
    assert rigid_budget['rigidity'] > flexible_budget['rigidity']
