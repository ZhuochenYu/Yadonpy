from __future__ import annotations

import json

import pytest
from rdkit import Chem

from yadonpy.core import utils
import yadonpy.sim.qm as qm


def test_assign_charges_shortcuts_h_terminator_placeholder(monkeypatch, tmp_path):
    mol = utils.mol_from_smiles("[H][*]")

    def fail_calc_assign_charges(*args, **kwargs):
        raise AssertionError("QM backend should not run for hydrogen terminator placeholder")

    monkeypatch.setattr(qm.calc, "assign_charges", fail_calc_assign_charges)

    ok = qm.assign_charges(
        mol,
        charge="RESP",
        opt=True,
        work_dir=str(tmp_path),
        log_name="h_terminator",
    )

    assert ok is True
    assert qm.is_h_terminator_placeholder(mol) is True
    assert [atom.GetDoubleProp("AtomicCharge") for atom in mol.GetAtoms()] == [0.0, 0.0]
    assert [atom.GetDoubleProp("RESP") for atom in mol.GetAtoms()] == [0.0, 0.0]
    assert [atom.GetDoubleProp("ESP") for atom in mol.GetAtoms()] == [0.0, 0.0]

    charged_sdf = tmp_path / "01_qm" / "02_charge" / "h_terminator" / "h_terminator.charged.sdf"
    charges_json = tmp_path / "01_qm" / "90_charged_mol2" / "h_terminator.charges.json"
    assert charged_sdf.exists()
    assert charges_json.exists()

    payload = json.loads(charges_json.read_text(encoding="utf-8"))
    assert payload["meta"]["shortcut"] == "h_terminator_placeholder"
    assert payload["charges"] == [0.0, 0.0]


def test_h_terminator_placeholder_detection_is_not_triggered_for_normal_end_groups():
    assert qm.is_h_terminator_placeholder(utils.mol_from_smiles("[H][*]")) is True
    assert qm.is_h_terminator_placeholder(utils.mol_from_smiles("*C")) is False
    assert qm.is_h_terminator_placeholder(utils.mol_from_smiles("*O")) is False
    assert qm.is_h_terminator_placeholder(utils.mol_from_smiles("*CO")) is False


@pytest.mark.parametrize("smiles", ["O=C1OCCO1", "COC(=O)OCC", "CCOC(=O)OCC"])
def test_adaptive_resp_recipe_selects_carbonate_recipe(smiles):
    mol = utils.mol_from_smiles(smiles)

    recipe = qm._resolve_resp_qm_recipe(
        mol,
        resp_profile="adaptive",
        charge_model="RESP",
        opt_method="wb97m-d3bj",
        opt_basis="def2-SVP",
        opt_basis_gen={"Br": "def2-SVP", "I": "def2-SVP"},
        charge_method="wb97m-d3bj",
        charge_basis="def2-TZVP",
        charge_basis_gen={"Br": "def2-TZVP", "I": "def2-TZVP"},
        auto_level=True,
        total_charge=0,
    )

    assert recipe["adaptive_carbonate_recipe"] is True
    assert recipe["opt_method"] == "wb97m-v"
    assert recipe["charge_method"] == "wb97m-v"
    assert recipe["opt_basis"] == "def2-TZVP"
    assert recipe["charge_basis"] == "def2-TZVP"


def test_adaptive_resp_recipe_does_not_select_carbonate_recipe_for_unrelated_neutrals():
    mol = utils.mol_from_smiles("CCO")

    recipe = qm._resolve_resp_qm_recipe(
        mol,
        resp_profile="adaptive",
        charge_model="RESP",
        opt_method="wb97m-d3bj",
        opt_basis="def2-SVP",
        opt_basis_gen={"Br": "def2-SVP", "I": "def2-SVP"},
        charge_method="wb97m-d3bj",
        charge_basis="def2-TZVP",
        charge_basis_gen={"Br": "def2-TZVP", "I": "def2-TZVP"},
        auto_level=True,
        total_charge=0,
    )

    assert recipe["adaptive_carbonate_recipe"] is False
    assert recipe["opt_method"] == "wb97m-d3bj"
    assert recipe["charge_method"] == "wb97m-d3bj"


def test_legacy_resp_recipe_keeps_legacy_defaults_for_carbonates():
    mol = utils.mol_from_smiles("CCOC(=O)OCC")

    recipe = qm._resolve_resp_qm_recipe(
        mol,
        resp_profile="legacy",
        charge_model="RESP",
        opt_method="wb97m-d3bj",
        opt_basis="def2-SVP",
        opt_basis_gen={"Br": "def2-SVP", "I": "def2-SVP"},
        charge_method="wb97m-d3bj",
        charge_basis="def2-TZVP",
        charge_basis_gen={"Br": "def2-TZVP", "I": "def2-TZVP"},
        auto_level=True,
        total_charge=0,
    )

    assert recipe["adaptive_carbonate_recipe"] is False
    assert recipe["opt_method"] == "wb97m-d3bj"
    assert recipe["charge_method"] == "wb97m-d3bj"
    assert recipe["opt_basis"] == "def2-SVP"
    assert recipe["charge_basis"] == "def2-TZVP"


def test_symmetrize_charge_properties_updates_all_charge_props():
    mol = Chem.MolFromSmiles("CC")
    values = {
        "AtomicCharge": (-0.3, 0.1),
        "RESP": (-0.4, 0.2),
        "RESP2": (-0.5, 0.3),
        "ESP": (-0.6, 0.4),
        "AtomicCharge_raw": (-0.7, 0.5),
        "RESP_raw": (-0.8, 0.6),
        "RESP2_raw": (-0.9, 0.7),
        "ESP_raw": (-1.0, 0.8),
    }
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        for prop, pair in values.items():
            atom.SetDoubleProp(prop, float(pair[atom_idx]))

    changed = qm._symmetrize_charge_properties(mol, equivalence_groups=[[0, 1]])

    assert changed == 1
    for prop, pair in values.items():
        expected = sum(pair) / 2.0
        got = [mol.GetAtomWithIdx(i).GetDoubleProp(prop) for i in (0, 1)]
        assert got == pytest.approx([expected, expected])
