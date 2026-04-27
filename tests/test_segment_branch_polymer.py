from __future__ import annotations

import json
from pathlib import Path

import pytest

from yadonpy.core import poly, utils


def _linker_indices(mol, label: int) -> list[int]:
    isotope = int(label) + 2
    return [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "H" and int(atom.GetIsotope()) == isotope
    ]


def _seed_charges(mol, *, scale: float = 0.01):
    for atom in mol.GetAtoms():
        value = float(scale) * float(atom.GetIdx() + 1)
        atom.SetDoubleProp("AtomicCharge", value)
        atom.SetDoubleProp("RESP", value)
        atom.SetDoubleProp("ESP", value)
        atom.SetDoubleProp("RESP2", value)
    return mol


def _total_charge(mol, prop: str = "AtomicCharge") -> float:
    return float(sum(atom.GetDoubleProp(prop) for atom in mol.GetAtoms() if atom.HasProp(prop)))


def test_seg_gen_preserves_main_linkers_and_branch_markers(tmp_path: Path):
    monomer_a = _seed_charges(utils.mol_from_smiles("*CC*"))
    branchable = _seed_charges(utils.mol_from_smiles("*C([2*])C*"))

    segment = poly.seg_gen(
        [monomer_a, monomer_a, branchable],
        name="seg_aab",
        work_dir=tmp_path / "seg",
        dist_min=0.1,
        retry=2,
        retry_step=8,
        retry_opt_step=0,
        restart=False,
    )

    assert len(_linker_indices(segment, 1)) == 2
    assert len(_linker_indices(segment, 2)) == 1
    assert segment.GetProp("_Name") == "seg_aab"
    meta = json.loads(segment.GetProp("_yadonpy_segment_metadata_json"))
    assert meta["kind"] == "segment"
    assert meta["unit_count"] == 3
    assert meta["branch_labels"] == [2]


def test_seg_gen_cap_tail_keeps_one_attach_linker_and_charge_props(tmp_path: Path):
    branch_unit = _seed_charges(utils.mol_from_smiles("*CO*"))
    before = _total_charge(branch_unit)

    branch_segment = poly.seg_gen(
        [branch_unit],
        cap_tail="[H][*]",
        work_dir=tmp_path / "branch_segment",
        restart=False,
    )

    assert len(_linker_indices(branch_segment, 1)) == 1
    assert _total_charge(branch_segment) == pytest.approx(before, abs=1.0e-10)
    assert _total_charge(branch_segment, "RESP") == pytest.approx(before, abs=1.0e-10)
    assert _total_charge(branch_segment, "ESP") == pytest.approx(before, abs=1.0e-10)
    assert _total_charge(branch_segment, "RESP2") == pytest.approx(before, abs=1.0e-10)


def test_branch_segment_rw_ds_handles_multiple_same_label_sites(tmp_path: Path):
    base_unit = _seed_charges(utils.mol_from_smiles("*C([2*])C*"))
    base = poly.seg_gen(
        [base_unit, base_unit],
        work_dir=tmp_path / "base",
        dist_min=0.1,
        retry=2,
        retry_step=8,
        retry_opt_step=0,
        restart=False,
    )
    assert len(_linker_indices(base, 2)) == 2
    branch = poly.seg_gen([_seed_charges(utils.mol_from_smiles("*CO*"))], cap_tail="[H][*]", restart=False)

    branched = poly.branch_segment_rw(
        base,
        [branch],
        position=2,
        ds=[1.0],
        work_dir=tmp_path / "branch_all",
        dist_min=0.1,
        retry_step=8,
        retry_opt_step=0,
        restart=False,
    )

    assert len(_linker_indices(branched, 1)) == 2
    assert len(_linker_indices(branched, 2)) == 0
    meta = json.loads(branched.GetProp("_yadonpy_branch_metadata_json"))
    assert meta["selected_site_count"] == 2
    assert {site["source"] for site in meta["selected_sites"]} == {"ds"}


def test_branch_segment_rw_exact_map_and_prebranch_remain_polymerizable(tmp_path: Path):
    base_unit = _seed_charges(utils.mol_from_smiles("*C([2*])C*"))
    base = poly.seg_gen(
        [base_unit, base_unit],
        work_dir=tmp_path / "base_exact",
        dist_min=0.1,
        retry=2,
        retry_step=8,
        retry_opt_step=0,
        restart=False,
    )
    branch = poly.seg_gen([_seed_charges(utils.mol_from_smiles("*CO*"))], cap_tail="[H][*]", restart=False)

    prebranched = poly.branch_segment_rw(
        base,
        [branch],
        mode="pre",
        position=2,
        exact_map={"position": 2, "site_index": 0, "branch": 0},
        work_dir=tmp_path / "prebranch",
        dist_min=0.1,
        retry_step=8,
        retry_opt_step=0,
        restart=False,
    )

    assert len(_linker_indices(prebranched, 1)) == 2
    assert len(_linker_indices(prebranched, 2)) == 1
    branch_meta = json.loads(prebranched.GetProp("_yadonpy_branch_metadata_json"))
    assert branch_meta["mode"] == "pre"
    assert branch_meta["selected_site_count"] == 1
    assert branch_meta["selected_sites"][0]["source"] == "exact_map"

    polymer = poly.random_copolymerize_rw(
        [prebranched],
        2,
        work_dir=tmp_path / "prebranch_poly",
        dist_min=0.1,
        retry=2,
        retry_step=8,
        retry_opt_step=0,
        restart=False,
    )
    assert polymer is not None
    assert len(_linker_indices(polymer, 1)) == 2


def test_block_segment_rw_builds_long_block_from_segments(tmp_path: Path):
    seg_a = poly.seg_gen([_seed_charges(utils.mol_from_smiles("*CC*"))], name="seg_a", restart=False)
    seg_b = poly.seg_gen([_seed_charges(utils.mol_from_smiles("*CO*"))], name="seg_b", restart=False)

    block = poly.block_segment_rw(
        [seg_a, seg_b],
        [2, 1],
        name="block_aab",
        work_dir=tmp_path / "block",
        dist_min=0.1,
        retry=2,
        retry_step=8,
        retry_opt_step=0,
        restart=False,
    )

    assert block.GetProp("_Name") == "block_aab"
    assert len(_linker_indices(block, 1)) == 2
    meta = json.loads(block.GetProp("_yadonpy_segment_metadata_json"))
    assert meta["kind"] == "block_segment_polymer"
    assert meta["block_lengths"] == [2, 1]
