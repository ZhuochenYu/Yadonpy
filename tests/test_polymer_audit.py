from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from rdkit import Chem

from yadonpy.core import utils
from yadonpy.core.polyelectrolyte import annotate_polyelectrolyte_metadata
from yadonpy.core.polymer_audit import (
    audit_charge_groups,
    audit_junction_bonded_terms,
    compare_exported_charge_groups,
)


def _set_residue(atom, *, residue_number: int, residue_name: str, atom_name: str) -> None:
    info = Chem.AtomPDBResidueInfo()
    info.SetResidueNumber(int(residue_number))
    info.SetResidueName(str(residue_name))
    info.SetName(str(atom_name).rjust(4))
    atom.SetMonomerInfo(info)


def test_audit_junction_bonded_terms_flags_missing_local_terms():
    # Use an implicit-H heavy-atom backbone so the expected local bonded terms
    # stay focused on the inter-residue junction itself.
    mol = Chem.MolFromSmiles("CCCC")
    assert mol is not None
    for idx in (0, 1):
        _set_residue(mol.GetAtomWithIdx(idx), residue_number=1, residue_name="R1", atom_name=f"C{idx+1}")
    for idx in (2, 3):
        _set_residue(mol.GetAtomWithIdx(idx), residue_number=2, residue_name="R2", atom_name=f"C{idx+1}")

    mol.angles = {
        "0,1,2": SimpleNamespace(a=0, b=1, c=2),
    }
    mol.dihedrals = {}
    mol.impropers = {}

    report = audit_junction_bonded_terms(mol, radius=1)

    assert report["missing_angle_total"] == 1
    assert report["missing_dihedral_total"] == 1
    assert len(report["junction_bonds"]) == 1
    row = report["junction_bonds"][0]
    assert row["bond_atoms"] == [1, 2]
    assert row["missing_angles"] == [[1, 2, 3]]
    assert row["missing_dihedrals"] == [[0, 1, 2, 3]]


def test_audit_charge_groups_reports_selected_charge_totals():
    mol = utils.mol_from_smiles("CC(=O)[O-]", coord=False)
    assert mol is not None
    annotate_polyelectrolyte_metadata(mol)

    charges = {
        0: 0.00,
        1: 0.82,
        2: -0.41,
        3: -1.41,
        4: 0.00,
        5: 0.00,
        6: 0.00,
    }
    for atom in mol.GetAtoms():
        atom.SetDoubleProp("AtomicCharge", float(charges[int(atom.GetIdx())]))

    report = audit_charge_groups(mol)

    assert report["selected_charge_prop"] == "AtomicCharge"
    assert round(float(report["total_selected_charge"]), 6) == -1.0
    assert int(report["total_formal_charge"]) == -1
    assert len(report["groups"]) == 1
    grp = report["groups"][0]
    assert grp["label"] == "carboxylate"
    assert grp["atom_indices"] == [1, 2, 3]
    assert round(float(grp["selected_charge_total"]), 6) == -1.0


def test_compare_exported_charge_groups_matches_in_memory_totals(tmp_path: Path):
    mol = utils.mol_from_smiles("CC(=O)[O-]", coord=False)
    assert mol is not None
    summary = annotate_polyelectrolyte_metadata(mol)["summary"]

    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "charge_groups.json").write_text(
        json.dumps(
            {
                "species": [
                    {
                        "moltype": "CMC",
                        "charge_groups": summary["groups"],
                    }
                ]
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = compare_exported_charge_groups(system_dir=system_dir, moltype="CMC", mol=mol)

    assert report["exists"] is True
    assert report["match"] is True
    assert report["exported_groups"] == report["current_groups"]
