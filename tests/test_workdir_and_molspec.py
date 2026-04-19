import json
from pathlib import Path

import pytest
import numpy as np
from rdkit import Chem
from rdkit import Geometry as Geom
from rdkit.Chem import AllChem, Descriptors

from yadonpy.core import as_rdkit_mol, molecular_weight, workdir, utils, poly
from yadonpy.core.polyelectrolyte import (
    annotate_polyelectrolyte_metadata,
    build_residue_map,
    get_charge_groups,
    get_polyelectrolyte_summary,
)
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.interface.charge_audit import format_cell_charge_audit
from yadonpy.io.artifacts import _artifact_meta_compatibility_fields, write_molecule_artifacts
import yadonpy.io.artifacts as artifacts_mod
from yadonpy.io.molecule_cache import _fingerprint_mol, ensure_cached_artifacts
from yadonpy.io.gromacs_molecule import _format_gro_atom_line as format_single_gro_atom_line
from yadonpy.io.gromacs_system import _format_gro_atom_line as format_system_gro_atom_line
from yadonpy.io.gromacs_system import _load_gro_species_templates
from yadonpy.io.gromacs_system import canonicalize_smiles
from yadonpy.io.gromacs_system import _species_signature_from_smiles
from yadonpy.io.gromacs_system import export_system_from_cell_meta
import yadonpy.io.gromacs_system as gromacs_system_mod
from yadonpy.io.mol2 import read_mol2_with_charges, write_mol2_from_rdkit
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.runtime import get_run_options, run_options, set_run_options
from yadonpy.workflow.resume import file_signature as resume_file_signature
import yadonpy.sim.qm as qm_mod


def _parse_itp_atom_charges(itp_path: Path) -> list[float]:
    charges: list[float] = []
    in_atoms = False
    for raw in itp_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_atoms = stripped.strip("[]").strip().lower() == "atoms"
            continue
        if not in_atoms or not stripped or stripped.startswith(";"):
            continue
        cols = raw.split(";", 1)[0].split()
        if len(cols) < 7:
            continue
        charges.append(float(cols[6]))
    return charges


def _write_minimal_species_artifacts(base_dir: Path, *, mol_name: str, charges: list[float]) -> Path:
    mol_dir = base_dir / mol_name
    mol_dir.mkdir(parents=True, exist_ok=True)

    itp_lines = [
        "[ moleculetype ]",
        f"{mol_name} 3",
        "",
        "[ atoms ]",
        "; nr  type  resnr  residue  atom  cgnr  charge  mass",
    ]
    for idx, charge in enumerate(charges, start=1):
        itp_lines.append(
            f"{idx:5d}  XX  1  {mol_name[:5]:<5}  A{idx:<3d}  {idx:5d}  {charge: .6f}  12.011"
        )
    (mol_dir / f"{mol_name}.itp").write_text("\n".join(itp_lines) + "\n", encoding="utf-8")

    gro_lines = [mol_name, f"{len(charges):5d}"]
    for idx in range(1, len(charges) + 1):
        gro_lines.append(
            format_system_gro_atom_line(
                resnr=1,
                resname=mol_name[:5],
                atomname=f"A{idx}",
                atomnr=idx,
                x=0.01 * idx,
                y=0.0,
                z=0.0,
            )
        )
    gro_lines.append("   4.00000   4.00000   4.00000")
    (mol_dir / f"{mol_name}.gro").write_text("\n".join(gro_lines) + "\n", encoding="utf-8")
    (mol_dir / f"{mol_name}.top").write_text(
        f'#include "{mol_name}.itp"\n\n[ system ]\n{mol_name}\n\n[ molecules ]\n{mol_name} 1\n',
        encoding="utf-8",
    )
    return mol_dir


def _make_poly_like_localized_mol(smiles: str = "*OCC(=O)[O-]"):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol.SetProp("_yadonpy_smiles", smiles)
    mol.SetIntProp("num_units", 4)
    if mol.GetNumConformers() == 0:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for idx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(idx, Geom.Point3D(0.1 * idx, 0.0, 0.0))
        mol.AddConformer(conf, assignId=True)
    annotate_polyelectrolyte_metadata(mol)

    groups = get_charge_groups(mol)
    grouped = {int(idx) for grp in groups for idx in grp.get("atom_indices", [])}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8:
            atom.SetProp("ff_type", "oh")
        elif atom.GetAtomicNum() == 6:
            atom.SetProp("ff_type", "c3")
        else:
            atom.SetProp("ff_type", "du")
        atom.SetDoubleProp("AtomicCharge", 0.0)
        atom.SetDoubleProp("RESP", 0.0)

    for grp in groups:
        atom_indices = [int(i) for i in grp.get("atom_indices", [])]
        assert atom_indices
        per_atom = float(grp.get("formal_charge", 0)) / float(len(atom_indices))
        for idx in atom_indices:
            mol.GetAtomWithIdx(idx).SetDoubleProp("AtomicCharge", per_atom)
            mol.GetAtomWithIdx(idx).SetDoubleProp("RESP", per_atom)

    for idx in range(mol.GetNumAtoms()):
        if idx not in grouped:
            mol.GetAtomWithIdx(idx).SetDoubleProp("AtomicCharge", 0.0)
            mol.GetAtomWithIdx(idx).SetDoubleProp("RESP", 0.0)

    return mol


def _set_stale_polyelectrolyte_metadata(mol, atom_indices: list[int]):
    bad_group = {
        "group_id": "group_1",
        "label": "stale_group",
        "atom_indices": [int(i) for i in atom_indices],
        "formal_charge": -1,
        "source": "stale",
    }
    molecule_formal_charge = int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))
    neutral_remainder = [
        idx for idx in range(mol.GetNumAtoms()) if idx not in {int(i) for i in atom_indices}
    ]
    summary = {
        "is_polymer": True,
        "is_polyelectrolyte": True,
        "detection": "auto",
        "fallback": None,
        "groups": [bad_group],
        "neutral_remainder": neutral_remainder,
        "equivalence_groups": [],
        "molecule_formal_charge": molecule_formal_charge,
    }
    constraints = {
        "mode": "grouped",
        "charged_group_constraints": [
            {
                "group_id": bad_group["group_id"],
                "atom_indices": list(bad_group["atom_indices"]),
                "target_charge": -1,
                "source": "stale",
            }
        ],
        "neutral_remainder_charge": int(molecule_formal_charge + 1),
        "neutral_remainder_indices": neutral_remainder,
        "equivalence_groups": [],
        "fallback": None,
    }
    mol.SetProp("_yadonpy_charge_groups_json", json.dumps([bad_group], ensure_ascii=False))
    mol.SetProp("_yadonpy_resp_constraints_json", json.dumps(constraints, ensure_ascii=False))
    mol.SetProp("_yadonpy_polyelectrolyte_summary_json", json.dumps(summary, ensure_ascii=False))


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


def test_polymerize_rw_refreshes_polyelectrolyte_metadata_for_full_chain(tmp_path: Path):
    monomer = utils.mol_from_smiles("*CC(C(=O)[O-])*", name="poly_anion")
    summary = annotate_polyelectrolyte_metadata(monomer)["summary"]
    groups = list(summary.get("groups") or [])
    assert len(groups) == 1

    for atom in monomer.GetAtoms():
        atom.SetDoubleProp("AtomicCharge", 0.0)
        atom.SetDoubleProp("RESP", 0.0)
    for grp in groups:
        atom_indices = [int(i) for i in grp.get("atom_indices", [])]
        assert atom_indices
        per_atom = -1.0 / float(len(atom_indices))
        for idx in atom_indices:
            monomer.GetAtomWithIdx(idx).SetDoubleProp("AtomicCharge", per_atom)
            monomer.GetAtomWithIdx(idx).SetDoubleProp("RESP", per_atom)

    chain = poly.polymerize_rw(monomer, 4, work_dir=tmp_path / "rw_refresh")

    refreshed_groups = get_charge_groups(chain)
    assert len(refreshed_groups) == 4
    for grp in refreshed_groups:
        total = 0.0
        for idx in grp["atom_indices"]:
            total += float(chain.GetAtomWithIdx(int(idx)).GetDoubleProp("AtomicCharge"))
        assert total == pytest.approx(-1.0, abs=1.0e-8)


def test_localized_polymer_cache_fingerprint_tracks_atom_order_even_with_same_smiles_hint():
    mol_a = _make_poly_like_localized_mol()
    reverse_order = list(reversed(range(mol_a.GetNumAtoms())))
    mol_b = Chem.RenumberAtoms(mol_a, reverse_order)
    mol_b.SetProp("_yadonpy_smiles", mol_a.GetProp("_yadonpy_smiles"))
    mol_b.SetIntProp("num_units", 4)
    annotate_polyelectrolyte_metadata(mol_b)

    assert _fingerprint_mol(mol_a, "gaff2_mod") != _fingerprint_mol(mol_b, "gaff2_mod")


def test_rw_load_refreshes_stale_polyelectrolyte_metadata_from_cached_state(tmp_path: Path):
    mol = _make_poly_like_localized_mol()
    expected_groups = get_charge_groups(mol)
    assert expected_groups
    assert expected_groups[0]["atom_indices"] != [0, 1, 2]

    _set_stale_polyelectrolyte_metadata(mol, [0, 1, 2])
    payload = {"probe": "refresh_stale_charge_groups"}
    poly._rw_save(tmp_path / "rw_cache_refresh", "terminate_rw", payload, mol)

    restored = poly._rw_load(tmp_path / "rw_cache_refresh", "terminate_rw", payload)

    assert restored is not None
    assert get_charge_groups(restored) == expected_groups


def test_gaff_ff_assign_refreshes_stale_polyelectrolyte_metadata_when_requested():
    mol = utils.mol_from_smiles("CC(=O)[O-]")
    expected_groups = get_charge_groups(mol)
    assert expected_groups
    assert expected_groups[0]["atom_indices"] != [0, 1, 2]

    _set_stale_polyelectrolyte_metadata(mol, [0, 1, 2])
    ff = GAFF2_mod()

    assigned = ff.ff_assign(mol, charge=None, polyelectrolyte_mode=True, report=False)

    assert assigned is not False
    assert get_charge_groups(assigned) == expected_groups


def test_write_molecule_artifacts_uses_best_available_charge_property(tmp_path: Path):
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.UFFOptimizeMolecule(mol)
    for atom in mol.GetAtoms():
        atom.SetDoubleProp("AtomicCharge", 0.0)
        atom.SetDoubleProp("RESP", 0.1 if atom.GetIdx() == 0 else -0.01)
    out = tmp_path / "art"
    write_molecule_artifacts(
        mol,
        out,
        smiles="CCO",
        ff_name="gaff2_mod",
        charge_method="RESP",
        total_charge=0,
        mol_name="EtOH",
        write_mol2=False,
    )
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["charge_abs_sum"] > 0.0
    assert meta["charge_signature"]


def test_write_molecule_artifacts_retargets_localized_polyelectrolyte_total_charge(tmp_path: Path, monkeypatch):
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    assert mol is not None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=11)
    AllChem.UFFOptimizeMolecule(mol)
    annotate_polyelectrolyte_metadata(mol)

    for atom in mol.GetAtoms():
        atom.SetDoubleProp("AtomicCharge", -0.30)
        atom.SetDoubleProp("RESP", -0.30)

    captured: dict[str, float] = {}

    def _fake_topology_writer(mol_obj, out_dir: Path, *, mol_name: str):
        total_q = 0.0
        for atom in mol_obj.GetAtoms():
            total_q += float(atom.GetDoubleProp("AtomicCharge")) if atom.HasProp("AtomicCharge") else 0.0
        captured["total_q"] = float(total_q)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{mol_name}.itp").write_text("[ moleculetype ]\nTEST 3\n", encoding="utf-8")
        (out_dir / f"{mol_name}.gro").write_text("TEST\n0\n   1.0   1.0   1.0\n", encoding="utf-8")
        (out_dir / f"{mol_name}.top").write_text("", encoding="utf-8")
        return out_dir / f"{mol_name}.gro", out_dir / f"{mol_name}.itp", out_dir / f"{mol_name}.top"

    monkeypatch.setattr(artifacts_mod, "_ensure_bonded_terms_for_export", lambda mol_obj, ff_name: None)
    monkeypatch.setattr(artifacts_mod, "_reapply_bonded_patch_to_mol", lambda mol_obj: None)
    import yadonpy.io.gromacs_molecule as gmx_mol_mod
    monkeypatch.setattr(gmx_mol_mod, "write_gromacs_single_molecule_topology", _fake_topology_writer)

    out = tmp_path / "art_pe"
    write_molecule_artifacts(
        mol,
        out,
        smiles="CC(=O)[O-]",
        ff_name="gaff2_mod",
        charge_method="RESP",
        mol_name="AcO",
        write_mol2=False,
    )

    assert captured["total_q"] == pytest.approx(-1.0, abs=1.0e-6)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["charge_target_policy"] == "formal_charge_for_localized_polyelectrolyte"


def test_write_molecule_artifacts_records_order_sensitive_compatibility_signatures(tmp_path: Path, monkeypatch):
    mol = _make_poly_like_localized_mol()

    monkeypatch.setattr(artifacts_mod, "_ensure_bonded_terms_for_export", lambda mol_obj, ff_name: None)
    monkeypatch.setattr(artifacts_mod, "_reapply_bonded_patch_to_mol", lambda mol_obj: None)

    def _fake_topology_writer(mol_obj, out_dir: Path, *, mol_name: str):
        out_dir.mkdir(parents=True, exist_ok=True)
        charges = [float(atom.GetDoubleProp("AtomicCharge")) for atom in mol_obj.GetAtoms()]
        itp_lines = [
            "[ moleculetype ]",
            f"{mol_name} 3",
            "",
            "[ atoms ]",
            "; nr  type  resnr  residue  atom  cgnr  charge  mass",
        ]
        for idx, charge in enumerate(charges, start=1):
            itp_lines.append(
                f"{idx:5d}  XX  1  {mol_name[:5]:<5}  A{idx:<3d}  {idx:5d}  {charge: .6f}  12.011"
            )
        (out_dir / f"{mol_name}.itp").write_text("\n".join(itp_lines) + "\n", encoding="utf-8")
        (out_dir / f"{mol_name}.gro").write_text(f"{mol_name}\n{len(charges):5d}\n   4.00000   4.00000   4.00000\n", encoding="utf-8")
        (out_dir / f"{mol_name}.top").write_text("", encoding="utf-8")
        return out_dir / f"{mol_name}.gro", out_dir / f"{mol_name}.itp", out_dir / f"{mol_name}.top"

    import yadonpy.io.gromacs_molecule as gmx_mol_mod

    monkeypatch.setattr(gmx_mol_mod, "write_gromacs_single_molecule_topology", _fake_topology_writer)

    out = tmp_path / "art_cache_sig"
    write_molecule_artifacts(
        mol,
        out,
        smiles=mol.GetProp("_yadonpy_smiles"),
        ff_name="gaff2_mod",
        charge_method="RESP",
        mol_name="PolyPE",
        write_mol2=False,
    )

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["atom_order_signature"]
    assert meta["charge_group_signature"]
    assert meta["residue_signature"]


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


def test_large_system_mode_auto_enables_only_above_99999_atoms():
    assert poly._resolve_large_system_mode('auto', 99999) is False
    assert poly._resolve_large_system_mode('auto', 100000) is True
    assert poly._resolve_large_system_mode(True, 10) is True
    assert poly._resolve_large_system_mode(False, 500000) is False


def test_large_pack_state_matches_direct_periodic_clash_check():
    cell = Chem.MolFromSmiles('[Li+]')
    assert cell is not None
    conf = Chem.Conformer(cell.GetNumAtoms())
    conf.SetAtomPosition(0, Geom.Point3D(0.2, 5.0, 5.0))
    cell.AddConformer(conf)
    setattr(cell, 'cell', utils.Cell(10.0, 0.0, 10.0, 0.0, 10.0, 0.0))

    pack_state = poly._build_large_pack_state(cell, 1.0, enabled=True)
    wrapped_clash_coord = np.array([[10.5, 5.0, 5.0]], dtype=float)
    periodic_clash_coord = np.array([[9.5, 5.0, 5.0]], dtype=float)
    far_coord = np.array([[7.0, 5.0, 5.0]], dtype=float)

    assert poly.check_3d_structure_cell(cell, wrapped_clash_coord, dist_min=1.0) is False
    assert poly.check_3d_structure_cell(cell, wrapped_clash_coord, dist_min=1.0, pack_state=pack_state) is False
    assert poly.check_3d_structure_cell(cell, periodic_clash_coord, dist_min=1.0, pack_state=pack_state) is False
    assert poly.check_3d_structure_cell(cell, far_coord, dist_min=1.0) is True
    assert poly.check_3d_structure_cell(cell, far_coord, dist_min=1.0, pack_state=pack_state) is True


def test_random_copolymerize_rw_refreshes_bridge_oxygen_types_after_new_bonds(capsys):
    ff = GAFF2_mod()
    monomer = ff.mol('*OCC*', require_ready=False, prefer_db=False)
    monomer = ff.ff_assign(monomer, report=False)

    assert monomer is not False
    capsys.readouterr()

    polymer = poly.random_copolymerize_rw(
        [monomer],
        3,
        ratio=[1.0],
        tacticity='atactic',
        name='poly_acetal',
        retry=1,
        retry_step=20,
        retry_opt_step=0,
    )

    captured = capsys.readouterr()
    assert 'c3,oh,c3' not in captured.out
    assert polymer is not None

    bad_oxygens = []
    for atom in polymer.GetAtoms():
        if atom.GetSymbol() != 'O' or not atom.HasProp('ff_type'):
            continue
        heavy_neighbors = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1)
        if heavy_neighbors >= 2 and atom.GetProp('ff_type') == 'oh':
            bad_oxygens.append(atom.GetIdx())

    assert bad_oxygens == []


def test_ensure_name_ignores_generic_result_alias():
    ff = MERZ()
    Na = ff.mol("[Na+]")
    result = Na
    utils.ensure_name(Na, depth=1, prefer_var=True)

    assert result is Na
    assert utils.get_name(Na) == "Na"


def test_ff_assign_auto_exports_to_work_dir(tmp_path: Path):
    work_dir = tmp_path / "run"
    work_dir.mkdir(parents=True)

    ff = MERZ()
    Na = ff.mol("[Na+]")
    Na = ff.ff_assign(Na)

    assert Na is not False
    assert (work_dir / "00_molecules" / "Na.mol2").exists()
    assert (work_dir / "90_Na_gmx" / "Na.gro").exists()
    assert (work_dir / "90_Na_gmx" / "Na.itp").exists()
    assert (work_dir / "90_Na_gmx" / "Na.top").exists()


def test_auto_export_assigned_mol_skips_rewrite_when_restart_enabled(tmp_path: Path, monkeypatch):
    work_dir = tmp_path / "run"
    work_dir.mkdir(parents=True)

    with run_options(restart=False):
        ff = MERZ()
        Na = utils.named(ff.mol("[Na+]"), "Na")
        Na = ff.ff_assign(Na, report=False)
        assert Na is not False

    import yadonpy.io.mol2 as mol2_mod
    import yadonpy.io.gmx as gmx_mod

    def _boom(*args, **kwargs):
        raise AssertionError("auto_export_assigned_mol should reuse existing outputs during restart")

    monkeypatch.setattr(mol2_mod, "write_mol2", _boom)
    monkeypatch.setattr(gmx_mod, "write_gmx", _boom)

    with run_options(restart=True):
        reused = utils.auto_export_assigned_mol(Na, work_dir=work_dir)

    assert reused == work_dir
    assert (work_dir / "90_Na_gmx" / "assigned_state.json").exists()


def test_ff_assign_restart_reuses_assigned_state_and_skips_retyping(tmp_path: Path, monkeypatch):
    work_dir = tmp_path / "run"
    work_dir.mkdir(parents=True)

    ff = GAFF2_mod()
    smiles = "CCO"
    prev = get_run_options()

    try:
        set_run_options(restart=False, strict_inputs=prev.strict_inputs)
        ethanol = utils.named(Chem.AddHs(Chem.MolFromSmiles(smiles)), "ethanol")
        assert AllChem.EmbedMolecule(ethanol, randomSeed=0xC0DE) == 0
        assert ff.ff_assign(ethanol, charge="gasteiger", report=False) is not False

        restored_ff = GAFF2_mod()
        restored = utils.named(Chem.AddHs(Chem.MolFromSmiles(smiles)), "ethanol")
        assert AllChem.EmbedMolecule(restored, randomSeed=0xC0DE) == 0

        def _should_not_retype(*args, **kwargs):
            raise AssertionError("restart restore should bypass assign_ptypes")

        monkeypatch.setattr(restored_ff, "assign_ptypes", _should_not_retype)

        set_run_options(restart=True, strict_inputs=prev.strict_inputs)
        result = restored_ff.ff_assign(restored, charge="gasteiger", report=False)
    finally:
        set_run_options(restart=prev.restart, strict_inputs=prev.strict_inputs)

    assert result is not False
    assert restored.HasProp("_yadonpy_artifact_dir")
    assert restored.GetProp("_yadonpy_artifact_dir").endswith("90_ethanol_gmx")
    assert all(atom.HasProp("ff_type") for atom in restored.GetAtoms())


def test_random_copolymerize_rw_uses_work_dir_basename_as_default_name(tmp_path: Path, monkeypatch):
    monomer = utils.mol_from_smiles("*CC*")
    assert monomer is not None

    fake_poly = Chem.MolFromSmiles("CC")
    assert fake_poly is not None

    monkeypatch.setattr(poly, "_effective_restart_flag", lambda *args, **kwargs: False)
    monkeypatch.setattr(poly, "_rw_load_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(poly, "_rw_save_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(poly, "_rw_save", lambda *args, **kwargs: None)
    monkeypatch.setattr(poly, "gen_monomer_array", lambda *args, **kwargs: [0])
    monkeypatch.setattr(poly, "gen_chiral_inv_array", lambda *args, **kwargs: ([False], True))
    monkeypatch.setattr(poly, "random_walk_polymerization", lambda *args, **kwargs: Chem.Mol(fake_poly))

    CMC = poly.random_copolymerize_rw([monomer], 1, work_dir=tmp_path / "CMC_rw")

    assert utils.get_name(CMC) == "CMC"


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
    rng_state = np.random.get_state()
    np.random.seed(7)
    try:
        p1 = poly.polymerize_rw(spec, int(dp), tacticity='atactic', work_dir=wd, retry=2, retry_step=4, retry_opt_step=0)
        np.random.seed(7)
        p2 = poly.polymerize_rw(spec, int(dp), tacticity='atactic', work_dir=wd, retry=2, retry_step=4, retry_opt_step=0)
    finally:
        np.random.set_state(rng_state)

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


def test_merz_ff_assign_infers_variable_name_without_explicit_name():
    from yadonpy.core import naming

    ion_ff = MERZ()
    Li = ion_ff.mol('[Li+]')
    Li = ion_ff.ff_assign(Li, report=False)

    assert Li is not False
    assert naming.get_name(Li) == 'Li'


def test_gaff_ff_assign_infers_variable_name_for_direct_rdkit_mol():
    from yadonpy.core import naming

    ff = GAFF2_mod()
    solvent_A = utils.mol_from_smiles('CCO')
    solvent_A = ff.ff_assign(solvent_A, report=False)

    assert solvent_A is not False
    assert naming.get_name(solvent_A) == 'solvent_A'


def test_mol_net_charge_recognizes_resp_only_atoms():
    mol = Chem.MolFromSmiles('CC')
    atom0 = mol.GetAtomWithIdx(0)
    atom1 = mol.GetAtomWithIdx(1)
    atom0.SetDoubleProp('RESP', 0.35)
    atom1.SetDoubleProp('RESP', -0.35)

    assert poly._mol_net_charge(mol) == pytest.approx(0.0, abs=1.0e-12)


def test_mol_net_charge_prefers_nonzero_resp_over_zero_atomiccharge():
    mol = Chem.MolFromSmiles('CC')
    atom0 = mol.GetAtomWithIdx(0)
    atom1 = mol.GetAtomWithIdx(1)
    atom0.SetDoubleProp('AtomicCharge', 0.0)
    atom1.SetDoubleProp('AtomicCharge', 0.0)
    atom0.SetDoubleProp('RESP', 0.45)
    atom1.SetDoubleProp('RESP', -1.45)

    assert poly._mol_net_charge(mol) == pytest.approx(-1.0, abs=1.0e-12)


def test_write_mol2_prefers_resp_when_atomiccharge_is_zero_placeholder(tmp_path: Path):
    mol = Chem.AddHs(Chem.MolFromSmiles('O'))
    assert mol is not None
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetDoubleProp('AtomicCharge', 0.0)
        atom.SetDoubleProp('RESP', -0.8 if idx == 0 else 0.4)

    out = write_mol2_from_rdkit(mol=mol, out_dir=tmp_path / 'mol2')
    loaded = read_mol2_with_charges(out, sanitize=False)
    charges = [float(atom.GetDoubleProp('AtomicCharge')) for atom in loaded.GetAtoms()]

    assert sum(charges) == pytest.approx(0.0, abs=1.0e-8)
    assert max(abs(q) for q in charges) > 0.1


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


def test_pf6_moldb_load_prefers_unsanitized_mol2_for_hypervalent_ions(tmp_path: Path, monkeypatch):
    monkeypatch.setenv('YADONPY_MOLDB', str(tmp_path / 'moldb'))

    ff = GAFF2_mod()
    pf6_smiles = 'F[P-](F)(F)(F)(F)F'
    built = utils.mol_from_smiles(pf6_smiles, coord=False, name='PF6')
    utils.ensure_3d_coords(built, smiles_hint=pf6_smiles, engine='openbabel')
    built = ff.ff_assign(built, charge='gasteiger', bonded='DRIH', report=False)
    assert built is not False

    ff.store_to_db(built, smiles_or_psmiles=pf6_smiles, name='PF6', charge='gasteiger')

    from yadonpy.moldb import store as storemod

    real = storemod.rdmolfiles.MolFromMol2File
    calls = []

    def _wrapped(*args, **kwargs):
        calls.append(bool(kwargs.get('sanitize', True)))
        return real(*args, **kwargs)

    monkeypatch.setattr(storemod.rdmolfiles, 'MolFromMol2File', _wrapped)

    loaded = ff.ff_assign(ff.mol(pf6_smiles, charge='gasteiger'), bonded='DRIH', report=False)

    assert loaded is not False
    assert calls
    assert calls[0] is False


def test_species_signature_from_smiles_handles_pf6_without_sanitized_parser_failure():
    sig = _species_signature_from_smiles('F[P-](F)(F)(F)(F)F')
    assert sig is not None
    nat, bond_sig = sig
    assert nat == 7
    assert bond_sig


def test_canonicalize_smiles_preserves_pf6_input_for_hypervalent_ions():
    smi = 'F[P-](F)(F)(F)(F)F'
    assert canonicalize_smiles(smi) == smi


def test_ensure_cached_artifacts_preserves_pf6_charge_sum_when_formal_charge_is_wrong(tmp_path: Path, monkeypatch):
    ensure_initialized()
    monkeypatch.setenv('YADONPY_MOL_CACHE_DIR', str(tmp_path / 'mol_cache'))

    ff = GAFF2_mod()
    pf6 = ff.ff_assign(
        ff.mol('F[P-](F)(F)(F)(F)F', charge='RESP', require_ready=True, prefer_db=True),
        bonded='DRIH',
        report=False,
    )

    q_before = sum(float(atom.GetDoubleProp('AtomicCharge')) for atom in pf6.GetAtoms() if atom.HasProp('AtomicCharge'))
    formal_q = sum(int(atom.GetFormalCharge()) for atom in pf6.GetAtoms())

    assert q_before == pytest.approx(-1.0, abs=1.0e-6)
    assert formal_q == -1

    ensure_cached_artifacts(pf6, mol_name='PF6')

    q_after = sum(float(atom.GetDoubleProp('AtomicCharge')) for atom in pf6.GetAtoms() if atom.HasProp('AtomicCharge'))
    assert q_after == pytest.approx(q_before, abs=1.0e-6)


def test_ensure_cached_artifacts_still_cleans_small_neutral_charge_drift(tmp_path: Path, monkeypatch):
    monkeypatch.setenv('YADONPY_MOL_CACHE_DIR', str(tmp_path / 'mol_cache'))

    ff = GAFF2_mod()
    mol = ff.ff_assign(ff.mol('CCO', require_ready=False, prefer_db=False), report=False)
    assert mol is not False

    seeded = [0.10, -0.03, -0.04, -0.01, -0.01, -0.01, 0.02, -0.01, -0.008]
    assert len(seeded) == mol.GetNumAtoms()
    for atom, q in zip(mol.GetAtoms(), seeded):
        atom.SetDoubleProp('AtomicCharge', float(q))

    q_before = sum(float(atom.GetDoubleProp('AtomicCharge')) for atom in mol.GetAtoms())
    assert q_before == pytest.approx(0.0, abs=0.1)
    assert abs(q_before) > 1.0e-3

    ensure_cached_artifacts(mol, mol_name='ethanol_like')

    q_after = sum(float(atom.GetDoubleProp('AtomicCharge')) for atom in mol.GetAtoms())
    assert q_after == pytest.approx(0.0, abs=1.0e-6)


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
    assert out.box_nm == pytest.approx(0.3, abs=1.0e-6)
    assert out.box_lengths_nm == pytest.approx((0.3, 0.3, 0.3), abs=1.0e-6)
    system_meta = json.loads(out.system_meta.read_text(encoding='utf-8'))
    assert tuple(system_meta['box_lengths_nm']) == pytest.approx((0.3, 0.3, 0.3), abs=1.0e-6)


def test_export_system_skips_fragment_materialization_when_cached_artifacts_exist(tmp_path: Path, monkeypatch):
    import yadonpy.io.gromacs_system as gsys

    ff = GAFF2_mod()
    solvent = ff.mol('CCO', require_ready=False, prefer_db=False, name='solvent_A')
    assert ff.ff_assign(solvent, report=False)

    ac = poly.amorphous_cell([solvent], [3], density=0.2, retry=1, retry_step=20, work_dir=tmp_path / 'cell')

    def _unexpected_get_fragments(*args, **kwargs):
        raise AssertionError('fragment materialization should stay lazy when cached artifacts are available')

    monkeypatch.setattr(gsys.Chem, 'GetMolFrags', _unexpected_get_fragments)

    out = gsys.export_system_from_cell_meta(
        cell_mol=ac,
        out_dir=tmp_path / 'sys',
        ff_name=ff.name,
        charge_method='RESP',
        write_system_mol2=False,
    )

    assert out.system_top.exists()
    assert out.system_gro.exists()
    assert (out.molecules_dir / 'solvent_A' / 'solvent_A.itp').exists()


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


def test_amorphous_cell_does_not_consume_implicit_global_ion_registry(tmp_path: Path):
    ff = GAFF2_mod()
    ion_ff = MERZ()
    solvent = ff.mol('CCO', require_ready=False, prefer_db=False)
    assert ff.ff_assign(solvent, report=False)

    orphan_pack = poly.ion(ion='Na+', n_ion=1, ff=ion_ff)
    assert orphan_pack.n == 1

    ac = poly.amorphous_cell(
        [solvent],
        [1],
        density=0.2,
        retry=1,
        retry_step=20,
        work_dir=tmp_path / 'cell_no_implicit_ions',
    )
    meta = json.loads(ac.GetProp('_yadonpy_cell_meta'))
    assert len(meta['species']) == 1
    assert meta['species'][0]['smiles']


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


def test_export_system_recovers_localized_charge_groups_from_smiles_when_cached_metadata_are_incomplete(tmp_path: Path):
    acetate_charges = [0.10, 0.20, -0.60, -0.70]
    acetate_smiles = "CC(=O)[O-]"
    source_root = tmp_path / "source_molecules"
    _write_minimal_species_artifacts(source_root, mol_name="Acetate", charges=acetate_charges)

    cell = Chem.Mol()
    setattr(cell, "cell", utils.Cell(4.0, 0.0, 4.0, 0.0, 4.0, 0.0))
    meta = {
        "density_g_cm3": 1.0,
        "polyelectrolyte_mode": True,
        "species": [
            {
                "smiles": acetate_smiles,
                "n": 2,
                "natoms": len(acetate_charges),
                "name": "Acetate",
                "charge_scale": 0.8,
                "charge_groups": None,
                "resp_constraints": None,
                "polyelectrolyte_summary": None,
                "polyelectrolyte_mode": True,
            }
        ],
    }
    cell.SetProp("_yadonpy_cell_meta", json.dumps(meta, ensure_ascii=False))

    out = export_system_from_cell_meta(
        cell_mol=cell,
        out_dir=tmp_path / "scaled_recovered_groups",
        ff_name="gaff2_mod",
        charge_method="RESP",
        source_molecules_dir=source_root,
        write_system_mol2=False,
    )

    exported = _parse_itp_atom_charges(out.molecules_dir / "Acetate" / "Acetate.itp")
    assert exported[0] == pytest.approx(acetate_charges[0], abs=1.0e-8)
    assert sum(exported[1:]) == pytest.approx(-0.8, abs=1.0e-8)

    report = json.loads((out.system_top.parent / "charge_scaling_report.json").read_text(encoding="utf-8"))
    species_report = report["species"][0]
    assert species_report["used_group_scaling"] is True
    assert species_report["report"]["groups"][0]["target_total_charge"] == pytest.approx(-0.8, abs=1.0e-8)


def test_export_system_preserves_precomputed_polymer_charge_groups_during_fast_path_scaling(tmp_path: Path):
    poly_mol = _make_poly_like_localized_mol()
    poly_name = "PolyPE"
    poly_smiles = poly_mol.GetProp("_yadonpy_smiles")
    charges = [float(atom.GetDoubleProp("AtomicCharge")) for atom in poly_mol.GetAtoms()]
    source_root = tmp_path / "source_molecules"
    _write_minimal_species_artifacts(source_root, mol_name=poly_name, charges=charges)

    expected_groups = get_charge_groups(poly_mol)
    expected_indices = [list(group["atom_indices"]) for group in expected_groups]
    assert expected_indices
    assert expected_indices[0] != [0, 1, 2]

    cell = Chem.Mol()
    setattr(cell, "cell", utils.Cell(4.0, 0.0, 4.0, 0.0, 4.0, 0.0))
    cell.SetProp(
        "_yadonpy_cell_meta",
        json.dumps(
            {
                "density_g_cm3": 1.0,
                "polyelectrolyte_mode": True,
                "species": [
                    {
                        "smiles": poly_smiles,
                        "n": 1,
                        "natoms": poly_mol.GetNumAtoms(),
                        "name": poly_name,
                        "charge_scale": 0.7,
                        "charge_groups": expected_groups,
                        "resp_constraints": None,
                        "polyelectrolyte_summary": get_polyelectrolyte_summary(poly_mol),
                        "residue_map": build_residue_map(poly_mol, mol_name=poly_name),
                        "polyelectrolyte_mode": True,
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )

    out = export_system_from_cell_meta(
        cell_mol=cell,
        out_dir=tmp_path / "scaled_preserve_groups",
        ff_name="gaff2_mod",
        charge_method="RESP",
        source_molecules_dir=source_root,
        write_system_mol2=False,
    )

    report = json.loads((out.system_top.parent / "charge_scaling_report.json").read_text(encoding="utf-8"))
    species_report = report["species"][0]
    actual_indices = [list(group["atom_indices"]) for group in species_report["report"]["groups"]]
    assert actual_indices == expected_indices

    exported = _parse_itp_atom_charges(out.molecules_dir / poly_name / f"{poly_name}.itp")
    assert max(abs(q) for q in exported) < 5.0


def test_export_system_fails_closed_for_polyelectrolyte_scaling_without_charge_groups_when_smiles_cannot_be_recovered(tmp_path: Path):
    source_root = tmp_path / "source_molecules"
    _write_minimal_species_artifacts(source_root, mol_name="OpaquePE", charges=[-0.4, -0.6, 0.0])

    cell = Chem.Mol()
    setattr(cell, "cell", utils.Cell(4.0, 0.0, 4.0, 0.0, 4.0, 0.0))
    meta = {
        "density_g_cm3": 1.0,
        "polyelectrolyte_mode": True,
        "species": [
            {
                "smiles": "not-a-valid-smiles",
                "n": 1,
                "natoms": 3,
                "name": "OpaquePE",
                "charge_scale": 0.8,
                "charge_groups": None,
                "resp_constraints": None,
                "polyelectrolyte_summary": {
                    "is_polyelectrolyte": False,
                    "groups": [
                        {
                            "group_id": "group_1",
                            "label": "carboxylate",
                            "atom_indices": [0, 1],
                            "formal_charge": -1,
                            "source": "template",
                        }
                    ],
                },
                "polyelectrolyte_mode": True,
            }
        ],
    }
    cell.SetProp("_yadonpy_cell_meta", json.dumps(meta, ensure_ascii=False))

    with pytest.raises(RuntimeError, match="requires charge_groups"):
        export_system_from_cell_meta(
            cell_mol=cell,
            out_dir=tmp_path / "scaled_missing_groups",
            ff_name="gaff2_mod",
            charge_method="RESP",
            source_molecules_dir=source_root,
            write_system_mol2=False,
        )


def test_export_system_does_not_apply_polyelectrolyte_group_scaling_to_tfsi(tmp_path: Path):
    tfsi_charges = [
        0.429088,
        -0.168652,
        -0.168652,
        -0.168652,
        0.885514,
        -0.508368,
        -0.508368,
        -0.583821,
        0.885514,
        -0.508368,
        -0.508368,
        0.429088,
        -0.168652,
        -0.168652,
        -0.168652,
    ]
    tfsi_smiles = "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F"
    source_root = tmp_path / "source_molecules"
    _write_minimal_species_artifacts(source_root, mol_name="TFSI", charges=tfsi_charges)

    cell = Chem.Mol()
    setattr(cell, "cell", utils.Cell(4.0, 0.0, 4.0, 0.0, 4.0, 0.0))
    meta = {
        "density_g_cm3": 1.2,
        "species": [
            {
                "smiles": tfsi_smiles,
                "n": 4,
                "natoms": len(tfsi_charges),
                "name": "TFSI",
                "charge_scale": 0.75,
                "charge_groups": [
                    {
                        "group_id": "group_1",
                        "label": "graph_group_1",
                        "atom_indices": [1, 4, 5, 6, 7, 8, 9, 10, 11],
                        "formal_charge": -1,
                        "source": "graph",
                    }
                ],
                "polyelectrolyte_summary": {
                    "is_polyelectrolyte": False,
                    "groups": [
                        {
                            "group_id": "group_1",
                            "label": "graph_group_1",
                            "atom_indices": [1, 4, 5, 6, 7, 8, 9, 10, 11],
                            "formal_charge": -1,
                            "source": "graph",
                        }
                    ],
                },
                "polyelectrolyte_mode": False,
            }
        ],
    }
    cell.SetProp("_yadonpy_cell_meta", json.dumps(meta, ensure_ascii=False))

    out = export_system_from_cell_meta(
        cell_mol=cell,
        out_dir=tmp_path / "scaled",
        ff_name="gaff2_mod",
        charge_method="RESP",
        source_molecules_dir=source_root,
        write_system_mol2=False,
    )

    itp_path = out.molecules_dir / "TFSI" / "TFSI.itp"
    exported_charges = _parse_itp_atom_charges(itp_path)
    assert len(exported_charges) == len(tfsi_charges)
    for actual, expected in zip(exported_charges, tfsi_charges):
        assert actual == pytest.approx(expected * 0.75, abs=1.0e-8)
    assert max(abs(q) for q in exported_charges) < 1.0

    report = json.loads((out.system_top.parent / "charge_scaling_report.json").read_text(encoding="utf-8"))
    assert report["species"][0]["used_group_scaling"] is False


def test_export_system_applies_group_scaling_to_localized_small_molecule_anion(tmp_path: Path):
    acetate_charges = [0.10, 0.20, -0.60, -0.70]
    acetate_smiles = "CC(=O)[O-]"
    source_root = tmp_path / "source_molecules"
    _write_minimal_species_artifacts(source_root, mol_name="Acetate", charges=acetate_charges)

    cell = Chem.Mol()
    setattr(cell, "cell", utils.Cell(4.0, 0.0, 4.0, 0.0, 4.0, 0.0))
    meta = {
        "density_g_cm3": 1.0,
        "species": [
            {
                "smiles": acetate_smiles,
                "n": 3,
                "natoms": len(acetate_charges),
                "name": "Acetate",
                "charge_scale": 0.8,
                "charge_groups": [
                    {
                        "group_id": "group_1",
                        "label": "carboxylate",
                        "atom_indices": [1, 2, 3],
                        "formal_charge": -1,
                        "source": "template",
                    }
                ],
                "polyelectrolyte_summary": {
                    "is_polyelectrolyte": False,
                    "groups": [
                        {
                            "group_id": "group_1",
                            "label": "carboxylate",
                            "atom_indices": [1, 2, 3],
                            "formal_charge": -1,
                            "source": "template",
                        }
                    ],
                },
                "polyelectrolyte_mode": False,
            }
        ],
    }
    cell.SetProp("_yadonpy_cell_meta", json.dumps(meta, ensure_ascii=False))

    out = export_system_from_cell_meta(
        cell_mol=cell,
        out_dir=tmp_path / "scaled_localized",
        ff_name="gaff2_mod",
        charge_method="RESP",
        source_molecules_dir=source_root,
        write_system_mol2=False,
    )

    exported = _parse_itp_atom_charges(out.molecules_dir / "Acetate" / "Acetate.itp")
    assert exported[0] == pytest.approx(acetate_charges[0], abs=1.0e-8)
    assert sum(exported[1:]) == pytest.approx(-0.8, abs=1.0e-8)

    report = json.loads((out.system_top.parent / "charge_scaling_report.json").read_text(encoding="utf-8"))
    species_report = report["species"][0]
    assert species_report["used_group_scaling"] is True
    assert species_report["report"]["groups"][0]["target_total_charge"] == pytest.approx(-0.8, abs=1.0e-8)


def test_export_system_rebuilds_mismatched_cached_polymer_artifact_before_group_scaling(tmp_path: Path, monkeypatch):
    current = _make_poly_like_localized_mol()
    old = Chem.RenumberAtoms(current, list(reversed(range(current.GetNumAtoms()))))
    old.SetProp("_yadonpy_smiles", current.GetProp("_yadonpy_smiles"))
    old.SetIntProp("num_units", 4)
    annotate_polyelectrolyte_metadata(old)

    old_charges = [0.65, -0.10, 0.25, -0.80, 0.15, -0.15][: old.GetNumAtoms()]
    old_dir = _write_minimal_species_artifacts(tmp_path / "cached_polymer", mol_name="PolyPE", charges=old_charges)
    (old_dir / "meta.json").write_text(
        json.dumps(
            {
                "n_atoms": int(old.GetNumAtoms()),
                **_artifact_meta_compatibility_fields(old, mol_name="PolyPE"),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    current_charges = [float(atom.GetDoubleProp("AtomicCharge")) for atom in current.GetAtoms()]
    current_groups = get_charge_groups(current)
    group_atoms = {int(idx) for grp in current_groups for idx in grp.get("atom_indices", [])}
    expected = list(current_charges)
    for grp in current_groups:
        indices = [int(i) for i in grp.get("atom_indices", [])]
        factor = 0.7
        for idx in indices:
            expected[idx] *= factor

    calls = {"count": 0}

    def _fake_write_molecule_artifacts(
        mol_obj,
        out_dir: Path,
        *,
        mol_name: str,
        **kwargs,
    ):
        calls["count"] += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        charges = []
        for atom in mol_obj.GetAtoms():
            charges.append(float(atom.GetDoubleProp("AtomicCharge")) if atom.HasProp("AtomicCharge") else 0.0)

        itp_lines = [
            "[ moleculetype ]",
            f"{mol_name} 3",
            "",
            "[ atoms ]",
            "; nr  type  resnr  residue  atom  cgnr  charge  mass",
        ]
        for idx, charge in enumerate(charges, start=1):
            itp_lines.append(
                f"{idx:5d}  XX  1  {mol_name[:5]:<5}  A{idx:<3d}  {idx:5d}  {charge: .6f}  12.011"
            )
        (out_dir / f"{mol_name}.itp").write_text("\n".join(itp_lines) + "\n", encoding="utf-8")

        gro_lines = [mol_name, f"{len(charges):5d}"]
        for idx in range(1, len(charges) + 1):
            gro_lines.append(
                format_system_gro_atom_line(
                    resnr=1,
                    resname=mol_name[:5],
                    atomname=f"A{idx}",
                    atomnr=idx,
                    x=0.01 * idx,
                    y=0.0,
                    z=0.0,
                )
            )
        gro_lines.append("   4.00000   4.00000   4.00000")
        (out_dir / f"{mol_name}.gro").write_text("\n".join(gro_lines) + "\n", encoding="utf-8")
        (out_dir / f"{mol_name}.top").write_text(
            f'#include "{mol_name}.itp"\n\n[ system ]\n{mol_name}\n\n[ molecules ]\n{mol_name} 1\n',
            encoding="utf-8",
        )
        return out_dir

    monkeypatch.setattr(gromacs_system_mod, "write_molecule_artifacts", _fake_write_molecule_artifacts)
    monkeypatch.setattr(gromacs_system_mod, "_ensure_bonded_terms_from_types", lambda mol_obj, ff_name: None)

    cell = Chem.Mol(current)
    setattr(cell, "cell", utils.Cell(4.0, 0.0, 4.0, 0.0, 4.0, 0.0))
    cell.SetProp(
        "_yadonpy_cell_meta",
        json.dumps(
            {
                "density_g_cm3": 1.0,
                "polyelectrolyte_mode": True,
                "species": [
                    {
                        "smiles": current.GetProp("_yadonpy_smiles"),
                        "n": 1,
                        "natoms": int(current.GetNumAtoms()),
                        "name": "PolyPE",
                        "charge_scale": 0.7,
                        "charge_groups": current_groups,
                        "resp_constraints": None,
                        "polyelectrolyte_summary": get_polyelectrolyte_summary(current),
                        "residue_map": build_residue_map(current, mol_name="PolyPE"),
                        "polyelectrolyte_mode": True,
                        "cached_artifact_dir": str(old_dir),
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )

    out = export_system_from_cell_meta(
        cell_mol=cell,
        out_dir=tmp_path / "polymer_rebuilt",
        ff_name="gaff2_mod",
        charge_method="RESP",
        write_system_mol2=False,
    )

    exported = _parse_itp_atom_charges(out.molecules_dir / "PolyPE" / "PolyPE.itp")
    assert calls["count"] == 1
    assert len(exported) == len(expected)
    for actual, target in zip(exported, expected):
        assert actual == pytest.approx(target, abs=1.0e-6)
    assert sum(exported[idx] for idx in group_atoms) == pytest.approx(-0.7, abs=1.0e-6)
    assert max(abs(q) for q in exported) < 5.0


def test_export_system_rebuilds_polymer_cache_when_legacy_meta_lacks_order_signatures(tmp_path: Path, monkeypatch):
    current = _make_poly_like_localized_mol()
    legacy_dir = _write_minimal_species_artifacts(
        tmp_path / "legacy_polymer",
        mol_name="PolyPE",
        charges=[0.80, -0.10, 0.20, -0.70, -0.10, -0.10][: current.GetNumAtoms()],
    )
    (legacy_dir / "meta.json").write_text(json.dumps({"n_atoms": int(current.GetNumAtoms())}, indent=2) + "\n", encoding="utf-8")

    calls = {"count": 0}

    def _fake_write_molecule_artifacts(mol_obj, out_dir: Path, *, mol_name: str, **kwargs):
        calls["count"] += 1
        out_dir.mkdir(parents=True, exist_ok=True)
        charges = [float(atom.GetDoubleProp("AtomicCharge")) for atom in mol_obj.GetAtoms()]
        _write_minimal_species_artifacts(out_dir.parent, mol_name=mol_name, charges=charges)
        return out_dir

    monkeypatch.setattr(gromacs_system_mod, "write_molecule_artifacts", _fake_write_molecule_artifacts)
    monkeypatch.setattr(gromacs_system_mod, "_ensure_bonded_terms_from_types", lambda mol_obj, ff_name: None)

    cell = Chem.Mol(current)
    setattr(cell, "cell", utils.Cell(4.0, 0.0, 4.0, 0.0, 4.0, 0.0))
    cell.SetProp(
        "_yadonpy_cell_meta",
        json.dumps(
            {
                "density_g_cm3": 1.0,
                "polyelectrolyte_mode": True,
                "species": [
                    {
                        "smiles": current.GetProp("_yadonpy_smiles"),
                        "n": 1,
                        "natoms": int(current.GetNumAtoms()),
                        "name": "PolyPE",
                        "charge_scale": 0.7,
                        "charge_groups": get_charge_groups(current),
                        "polyelectrolyte_summary": get_polyelectrolyte_summary(current),
                        "residue_map": build_residue_map(current, mol_name="PolyPE"),
                        "polyelectrolyte_mode": True,
                        "cached_artifact_dir": str(legacy_dir),
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )

    out = export_system_from_cell_meta(
        cell_mol=cell,
        out_dir=tmp_path / "polymer_legacy_rebuilt",
        ff_name="gaff2_mod",
        charge_method="RESP",
        write_system_mol2=False,
    )

    assert calls["count"] == 1
    exported = _parse_itp_atom_charges(out.molecules_dir / "PolyPE" / "PolyPE.itp")
    assert max(abs(q) for q in exported) < 5.0


def test_export_system_keeps_small_molecule_cached_copy_path_without_new_signatures(tmp_path: Path, monkeypatch):
    cached_dir = _write_minimal_species_artifacts(tmp_path / "cached_small", mol_name="EC", charges=[0.10, -0.20, 0.10])

    def _should_not_write(*args, **kwargs):
        raise AssertionError("small-molecule cached artifact should have been reused")

    monkeypatch.setattr(gromacs_system_mod, "write_molecule_artifacts", _should_not_write)

    cell = Chem.MolFromSmiles("CCO")
    assert cell is not None
    setattr(cell, "cell", utils.Cell(4.0, 0.0, 4.0, 0.0, 4.0, 0.0))
    cell.SetProp(
        "_yadonpy_cell_meta",
        json.dumps(
            {
                "density_g_cm3": 1.0,
                "species": [
                    {
                        "smiles": "CCO",
                        "n": 1,
                        "natoms": 3,
                        "name": "EC",
                        "charge_scale": 1.0,
                        "polyelectrolyte_mode": False,
                        "cached_artifact_dir": str(cached_dir),
                    }
                ],
            },
            ensure_ascii=False,
        ),
    )

    out = export_system_from_cell_meta(
        cell_mol=cell,
        out_dir=tmp_path / "small_cache_hit",
        ff_name="gaff2_mod",
        charge_method="RESP",
        write_system_mol2=False,
    )

    exported = _parse_itp_atom_charges(out.molecules_dir / "EC" / "EC.itp")
    assert exported == pytest.approx([0.10, -0.20, 0.10], abs=1.0e-8)


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
    assert (scaled.system_top.parent / 'site_map.json').exists()


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


def test_eq21_forces_fresh_stage_rebuild_when_resume_inputs_change(tmp_path: Path):
    import yadonpy.sim.preset.eq as eqmod

    ff = GAFF2_mod()
    solvent = ff.mol('CCO', require_ready=False, prefer_db=False)
    assert ff.ff_assign(solvent, report=False)
    ac = poly.amorphous_cell([solvent], [1], density=0.2, retry=1, retry_step=20)

    job = eqmod.EQ21step(ac, work_dir=tmp_path / 'eq')
    final_gro = tmp_path / 'eq' / '03_EQ21' / '03_EQ21' / 'step_21' / 'md.gro'
    final_gro.parent.mkdir(parents=True, exist_ok=True)
    final_gro.write_text('mock\n', encoding='utf-8')

    old_spec = eqmod.StepSpec(
        name='equilibration_eq21',
        outputs=[final_gro],
        inputs={'input_gro_sig': {'path': 'old.gro', 'size': 1, 'sha256': 'old'}},
    )
    new_spec = eqmod.StepSpec(
        name='equilibration_eq21',
        outputs=[final_gro],
        inputs={'input_gro_sig': {'path': 'new.gro', 'size': 2, 'sha256': 'new'}},
    )
    job._resume.mark_done(old_spec)

    assert job._resume.needs_fresh_run(new_spec) is True
    assert job._job_restart_flag(new_spec, True) is False
    assert job._job_restart_flag(new_spec, False) is False


def test_eq21_recovers_completed_workflow_when_resume_record_is_missing(tmp_path: Path):
    import yadonpy.sim.preset.eq as eqmod

    job = eqmod.EQ21step(ac=object(), work_dir=tmp_path / 'eq')
    final_dir = tmp_path / 'eq' / '03_EQ21' / '03_EQ21' / 'step_21'
    final_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        final_dir / 'md.tpr',
        final_dir / 'md.xtc',
        final_dir / 'md.edr',
        final_dir / 'md.gro',
    ]
    for path in outputs:
        path.write_text('mock\n', encoding='utf-8')

    input_gro_sig = {'path': 'system.gro', 'size': 11, 'sha256': 'gro'}
    input_top_sig = {'path': 'system.top', 'size': 22, 'sha256': 'top'}
    input_ndx_sig = {'path': 'system.ndx', 'size': 33, 'sha256': 'ndx'}
    spec = eqmod.StepSpec(
        name='equilibration_eq21',
        outputs=outputs,
        inputs={
            'input_gro_sig': input_gro_sig,
            'input_top_sig': input_top_sig,
            'input_ndx_sig': input_ndx_sig,
        },
    )

    summary_path = tmp_path / 'eq' / '03_EQ21' / 'summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                'provenance': {
                    'input_gro_sig': input_gro_sig,
                    'input_top_sig': input_top_sig,
                    'input_ndx_sig': input_ndx_sig,
                }
            }
        ) + '\n',
        encoding='utf-8',
    )

    assert job._resume.reuse_status(spec) == 'no_record'
    recovered = eqmod._recover_completed_workflow_step(
        job._resume,
        spec,
        summary_path=summary_path,
        input_gro_sig=input_gro_sig,
        input_top_sig=input_top_sig,
        input_ndx_sig=input_ndx_sig,
        label='EQ21 workflow',
    )

    assert recovered is True
    assert job._resume.is_done(spec) is True
    assert job._job_restart_flag(spec, True) is True


def test_file_signature_is_stable_across_mtime_only_rewrites(tmp_path: Path):
    payload = tmp_path / 'system.gro'
    payload.write_text('same-content\n', encoding='utf-8')
    sig1 = resume_file_signature(payload)

    original = payload.stat().st_mtime
    payload.write_text('same-content\n', encoding='utf-8')
    payload.touch()
    sig2 = resume_file_signature(payload)

    assert sig1 == sig2
    assert sig1['size'] == payload.stat().st_size
    assert sig1['sha256'] == sig2['sha256']
    assert payload.stat().st_mtime >= original


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


def test_amorphous_cell_cache_uses_stable_molecule_identity_when_smiles_renderer_changes(tmp_path: Path, monkeypatch):
    ff = GAFF2_mod()
    spec = ff.mol('CCO', require_ready=False, prefer_db=False)
    assert ff.ff_assign(spec, report=False)
    spec.SetProp('_yadonpy_molid', 'demo-ethanol')

    wd = workdir(tmp_path / 'cell_cache_stable_id', restart=True)
    ac1 = poly.amorphous_cell([spec], [2], density=0.2, retry=1, retry_step=20, work_dir=wd)

    monkeypatch.setattr(poly, '_rw_load', lambda *args, **kwargs: None)
    monkeypatch.setattr(poly.Chem, 'MolToSmiles', lambda *args, **kwargs: 'changed-between-runs')

    def _unexpected_check(*args, **kwargs):
        raise AssertionError('placement should be skipped when stable molecule identity is available')

    monkeypatch.setattr(poly, 'check_3d_structure_cell', _unexpected_check)
    ac2 = poly.amorphous_cell([spec], [2], density=0.2, retry=1, retry_step=20, work_dir=wd)

    assert ac1.GetNumAtoms() == ac2.GetNumAtoms()
    assert hasattr(ac2, 'cell')


def test_amorphous_cell_stable_cache_ignores_changed_runtime_molid(tmp_path: Path, monkeypatch):
    ff = GAFF2_mod()
    spec1 = ff.mol('CCO', require_ready=False, prefer_db=False)
    assert ff.ff_assign(spec1, report=False)
    spec1.SetProp('_yadonpy_source_smiles', 'CCO')
    spec1.SetProp('_yadonpy_molid', 'runtime-a')

    wd = workdir(tmp_path / 'cell_cache_molid_drift', restart=True)
    ac1 = poly.amorphous_cell([spec1], [2], density=0.2, retry=1, retry_step=20, work_dir=wd)

    spec2 = ff.mol('CCO', require_ready=False, prefer_db=False)
    assert ff.ff_assign(spec2, report=False)
    spec2.SetProp('_yadonpy_source_smiles', 'CCO')
    spec2.SetProp('_yadonpy_molid', 'runtime-b')

    monkeypatch.setattr(poly, '_rw_load', lambda *args, **kwargs: None)

    def _unexpected_check(*args, **kwargs):
        raise AssertionError('placement should be skipped when only runtime molid changes')

    monkeypatch.setattr(poly, 'check_3d_structure_cell', _unexpected_check)
    ac2 = poly.amorphous_cell([spec2], [2], density=0.2, retry=1, retry_step=20, work_dir=wd)

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


def test_rw_cache_restores_atom_charge_props_and_residue_info(tmp_path: Path):
    mol = utils.mol_from_smiles('CO', coord=True, name='cache_probe')
    assert mol is not None
    mol.SetIntProp('head_idx', 0)
    mol.SetIntProp('tail_idx', mol.GetNumAtoms() - 1)
    mol.SetIntProp('num_units', 3)

    seeded = np.linspace(-0.3, 0.3, mol.GetNumAtoms())
    for idx, (atom, q) in enumerate(zip(mol.GetAtoms(), seeded)):
        atom.SetDoubleProp('AtomicCharge', float(q))
        atom.SetDoubleProp('RESP', float(q))
        atom.SetProp('ff_type', 'c3' if atom.GetSymbol() == 'C' else 'oh')
        atom.SetIntProp('mol_id', 7)
        atom.SetBoolProp('head', idx == 0)
        atom.SetBoolProp('tail', idx == mol.GetNumAtoms() - 1)
        atom.SetMonomerInfo(
            Chem.AtomPDBResidueInfo(
                f'{atom.GetSymbol():>4s}',
                residueName='TST',
                residueNumber=11,
                isHeteroAtom=False,
            )
        )

    wd = workdir(tmp_path / 'rw_cache_props', restart=True)
    payload = poly._rw_payload('cache_probe', smiles='CO')
    poly._rw_save(wd, 'cache_probe', payload, mol)
    cached = poly._rw_load(wd, 'cache_probe', payload)

    assert cached is not None
    assert poly._mol_net_charge(cached) == pytest.approx(poly._mol_net_charge(mol), abs=1.0e-12)
    assert cached.GetAtomWithIdx(0).HasProp('AtomicCharge')
    assert cached.GetAtomWithIdx(0).GetProp('ff_type') == 'c3'
    assert cached.GetAtomWithIdx(0).GetIntProp('mol_id') == 7
    assert cached.GetAtomWithIdx(0).GetBoolProp('head') is True
    assert cached.GetAtomWithIdx(cached.GetNumAtoms() - 1).GetBoolProp('tail') is True
    assert cached.GetAtomWithIdx(0).GetPDBResidueInfo() is not None
    assert cached.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName().strip() == 'TST'
    assert cached.GetIntProp('head_idx') == 0
    assert cached.GetIntProp('tail_idx') == cached.GetNumAtoms() - 1
    assert cached.GetIntProp('num_units') == 3


def test_cell_cache_save_handles_large_cell_meta_and_restores_charge_props(tmp_path: Path):
    mol = utils.mol_from_smiles('O', coord=True, name='cell_cache_probe')
    assert mol is not None
    setattr(mol, 'cell', utils.Cell(4.0, 0.0, 5.0, 0.0, 6.0, 0.0))
    mol.SetProp(
        '_yadonpy_cell_meta',
        json.dumps(
            {
                'species': [{'name': 'probe', 'n': 1, 'charge_scale': 0.8}] * 40,
                'net_charge_raw': -1.0,
                'net_charge_scaled': -0.8,
            },
            ensure_ascii=False,
        ),
    )
    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetDoubleProp('AtomicCharge', -0.2 if idx == 0 else 0.1)
        atom.SetDoubleProp('RESP', -0.2 if idx == 0 else 0.1)
        atom.SetProp('ff_type', 'oh' if atom.GetSymbol() == 'O' else 'ho')

    wd = workdir(tmp_path / 'cell_cache_props', restart=True)
    payload = poly._rw_payload('amorphous_cell', demo=True)
    state = poly._cell_state_from_mol(mol)
    poly._cell_cache_save(wd, 'amorphous_cell', payload, mol, state=state)
    cached, cached_state = poly._cell_cache_load(wd, 'amorphous_cell', payload)
    cached = poly._restore_cached_cell_state(cached, cached_state)

    assert cached is not None
    assert (Path(wd) / '.yadonpy' / 'cell_cache' / 'amorphous_cell.sdf').exists()
    assert cached.HasProp('_yadonpy_cell_meta')
    assert cached.GetAtomWithIdx(0).HasProp('AtomicCharge')
    assert poly._mol_net_charge(cached) == pytest.approx(poly._mol_net_charge(mol), abs=1.0e-12)
    assert hasattr(cached, 'cell')
    assert float(cached.cell.dx) == pytest.approx(float(mol.cell.dx))


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
        retry=20,
        rollback=1,
        rollback_shaking=False,
        retry_step=20,
        retry_opt_step=0,
    )
    rigid_budget = poly._resolve_rw_retry_budget(
        [rigid],
        retry=20,
        rollback=1,
        rollback_shaking=False,
        retry_step=20,
        retry_opt_step=0,
    )

    assert flexible_budget['retry'] == 60
    assert flexible_budget['rollback'] == 3
    assert flexible_budget['retry_step'] == 80
    assert flexible_budget['retry_opt_step'] == 4
    assert flexible_budget['changed'] == {
        'retry': (20, 60),
        'rollback': (1, 3),
        'retry_step': (20, 80),
        'retry_opt_step': (0, 4),
    }
    assert rigid_budget['retry'] == 80
    assert rigid_budget['rollback'] == 4
    assert rigid_budget['retry_step'] > flexible_budget['retry_step']
    assert rigid_budget['retry_opt_step'] > flexible_budget['retry_opt_step']
    assert rigid_budget['rigidity'] > flexible_budget['rigidity']


def test_export_helpers_prefer_unsanitized_smiles_parse_for_pf6(monkeypatch):
    import yadonpy.io.gromacs_system as gsys

    original = gsys.Chem.MolFromSmiles
    calls: list[tuple[str, object]] = []

    def _wrapped(smiles, *args, **kwargs):
        calls.append((str(smiles), kwargs.get('sanitize', True)))
        return original(smiles, *args, **kwargs)

    monkeypatch.setattr(gsys.Chem, 'MolFromSmiles', _wrapped)

    assert gsys.canonicalize_smiles('F[P-](F)(F)(F)(F)F')
    assert gsys._formal_charge_from_smiles('F[P-](F)(F)(F)(F)F') == -1

    pf6_calls = [sanitize for smiles, sanitize in calls if smiles == 'F[P-](F)(F)(F)(F)F']
    assert pf6_calls
    assert pf6_calls[0] is False


def test_moldb_canonical_key_prefers_unsanitized_smiles_parse_for_pf6(monkeypatch):
    import yadonpy.moldb.store as store_mod

    original = store_mod.Chem.MolFromSmiles
    calls: list[tuple[str, object]] = []

    def _wrapped(smiles, *args, **kwargs):
        calls.append((str(smiles), kwargs.get('sanitize', True)))
        return original(smiles, *args, **kwargs)

    monkeypatch.setattr(store_mod.Chem, 'MolFromSmiles', _wrapped)

    kind, canon, key = store_mod.canonical_key('F[P-](F)(F)(F)(F)F')

    assert kind == 'smiles'
    assert canon
    assert key
    pf6_calls = [sanitize for smiles, sanitize in calls if smiles == 'F[P-](F)(F)(F)(F)F']
    assert pf6_calls
    assert pf6_calls[0] is False
