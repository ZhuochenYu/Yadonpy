import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit import Geometry as Geom

import yadonpy as yp
from yadonpy.core import poly, utils
from yadonpy.core.graphite import _graphite_cif_path, register_cell_species_metadata, stack_cell_blocks
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.io.gromacs_system import export_system_from_cell_meta


def _translated_copy(mol: Chem.Mol, shift_xyz) -> Chem.Mol:
    dup = utils.deepcopy_mol(mol)
    conf = dup.GetConformer()
    sx, sy, sz = [float(v) for v in shift_xyz]
    for idx in range(dup.GetNumAtoms()):
        pos = conf.GetAtomPosition(idx)
        conf.SetAtomPosition(idx, Geom.Point3D(pos.x + sx, pos.y + sy, pos.z + sz))
    return dup


def test_build_graphite_basal_assigns_gaff_and_metadata():
    ff = GAFF2_mod()

    out = yp.build_graphite(
        nx=3,
        ny=2,
        n_layers=2,
        orientation="basal",
        edge_cap="H",
        ff=ff,
        name="graphite_layer",
    )

    atom_types = [atom.GetProp("ff_type") for atom in out.layer_mol.GetAtoms()]
    meta = json.loads(out.cell.GetProp("_yadonpy_cell_meta"))

    assert out.orientation == "basal"
    assert out.layer_count == 2
    assert meta["species"][0]["n"] == 2
    assert meta["species"][0]["name"] == "graphite_layer"
    assert all(atom_types)
    assert any(atom.GetSymbol() == "H" for atom in out.layer_mol.GetAtoms())
    assert out.box_nm[0] > 0.0
    assert out.box_nm[1] > 0.0
    assert out.box_nm[2] > 0.0


def test_bundled_graphite_cif_exists():
    path = _graphite_cif_path()
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "data_9000046" in text


def test_build_graphite_edge_supports_random_oxygen_caps():
    ff = GAFF2_mod()

    out = yp.build_graphite(
        nx=2,
        ny=2,
        n_layers=2,
        orientation="edge",
        edge_cap="random",
        random_cap_probs={"OH": 1.0},
        ff=ff,
        random_seed=7,
        name="graphite_edge",
    )

    symbols = [atom.GetSymbol() for atom in out.layer_mol.GetAtoms()]

    assert out.orientation == "edge"
    assert out.edge_cap_summary["OH"] > 0
    assert "O" in symbols
    assert out.box_nm[2] > out.box_nm[1] * 0.3


def test_build_graphite_periodic_exports_without_hydrogen_caps(tmp_path: Path):
    ff = GAFF2_mod()

    graphite = yp.build_graphite(
        nx=4,
        ny=4,
        n_layers=2,
        orientation="basal",
        edge_cap="periodic",
        ff=ff,
        name="graphite_periodic",
        top_padding_ang=4.0,
    )
    register_cell_species_metadata(
        graphite.cell,
        [graphite.layer_mol],
        [graphite.layer_count],
        pack_mode="graphite_periodic_test",
    )
    out = export_system_from_cell_meta(
        cell_mol=graphite.cell,
        out_dir=tmp_path / "graphite_periodic",
        ff_name=ff.name,
        charge_method="RESP",
        write_system_mol2=False,
    )

    symbols = [atom.GetSymbol() for atom in graphite.layer_mol.GetAtoms()]
    top_text = out.system_top.read_text(encoding="utf-8")

    assert graphite.edge_cap_summary["PERIODIC"] > 0
    assert set(symbols) == {"C"}
    assert all(atom.HasProp("ff_type") for atom in graphite.layer_mol.GetAtoms())
    assert out.system_top.exists()
    assert "graphite_periodic" in top_text or "M1" in top_text


def test_stack_cell_blocks_and_export_graphite_plus_solvent(tmp_path: Path):
    ff = GAFF2_mod()

    graphite = yp.build_graphite(
        nx=2,
        ny=2,
        n_layers=2,
        orientation="basal",
        edge_cap="H",
        ff=ff,
        name="graphite_layer",
        top_padding_ang=4.0,
    )
    solvent = ff.mol("CCO", require_ready=False, prefer_db=False, name="solvent_A")
    solvent = ff.ff_assign(solvent, report=False)
    assert solvent is not False

    solvent_block = _translated_copy(solvent, (0.0, 0.0, 0.0))
    stacked = stack_cell_blocks([graphite.cell, solvent_block], z_gaps_ang=[4.0], top_padding_ang=6.0)
    register_cell_species_metadata(
        stacked.cell,
        [graphite.layer_mol, solvent],
        [graphite.layer_count, 1],
        pack_mode="graphite_stack_test",
    )

    out = export_system_from_cell_meta(
        cell_mol=stacked.cell,
        out_dir=tmp_path / "graphite_sys",
        ff_name=ff.name,
        charge_method="RESP",
        write_system_mol2=False,
    )

    top_text = out.system_top.read_text(encoding="utf-8")
    ndx_text = out.system_ndx.read_text(encoding="utf-8")
    meta = json.loads(stacked.cell.GetProp("_yadonpy_cell_meta"))

    assert out.system_top.exists()
    assert out.system_gro.exists()
    assert "[ System ]" in ndx_text
    assert meta["species"][0]["name"] == "graphite_layer"
    assert meta["species"][1]["name"] == "solvent_A"
    assert "graphite_layer" in top_text
    assert "solvent_A" in top_text
