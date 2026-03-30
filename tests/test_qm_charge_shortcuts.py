from __future__ import annotations

import json

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
