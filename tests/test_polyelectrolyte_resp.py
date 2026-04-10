from __future__ import annotations

import json

from rdkit import Chem

from yadonpy.core.polyelectrolyte import (
    annotate_polyelectrolyte_metadata,
    build_residue_map,
    detect_charged_groups,
    scale_charged_groups_inplace,
    uses_localized_charge_groups,
)
from yadonpy.io.gromacs_system import (
    _rewrite_gro_resname,
    _rewrite_itp_moltype_and_resname,
    _scale_itp_charge_groups,
)


def test_detect_charged_groups_finds_carboxylate():
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    summary = detect_charged_groups(mol, detection="auto")
    assert len(summary["groups"]) == 1
    assert summary["groups"][0]["formal_charge"] == -1
    assert summary["groups"][0]["label"] in {"carboxylate", "graph_group_1"}


def test_annotate_polyelectrolyte_metadata_persists_props():
    mol = Chem.MolFromSmiles("*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]")
    mol.SetProp("_yadonpy_smiles", "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]")
    mol.SetIntProp("num_units", 4)
    annotated = annotate_polyelectrolyte_metadata(mol)
    assert annotated["summary"]["is_polymer"] is True
    assert mol.HasProp("_yadonpy_charge_groups_json")
    assert mol.HasProp("_yadonpy_resp_constraints_json")
    groups = json.loads(mol.GetProp("_yadonpy_charge_groups_json"))
    assert groups


def test_build_residue_map_uses_pdb_residue_info():
    mol = Chem.MolFromSmiles("CC")
    for idx, atom in enumerate(mol.GetAtoms(), start=1):
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueNumber(idx)
        info.SetResidueName(f"R{idx}")
        info.SetName(f"C{idx}")
        atom.SetMonomerInfo(info)
    residue_map = build_residue_map(mol, mol_name="POL")
    assert [r["residue_number"] for r in residue_map["residues"]] == [1, 2]
    assert residue_map["atoms"][0]["atom_name"] == "C1"


def test_scale_itp_charge_groups_only_changes_group_atoms():
    itp = """[ moleculetype ]
POL 3

[ atoms ]
1 C 1 RES C1 1 -0.10 12.011
2 O 1 RES O1 1 -0.45 15.999
3 O 1 RES O2 1 -0.45 15.999
4 C 1 RES C2 1 0.20 12.011
"""
    scaled, report = _scale_itp_charge_groups(
        itp,
        [{"group_id": "g1", "atom_indices": [1, 2], "formal_charge": -1}],
        0.8,
    )
    assert "0.20000000" in scaled  # atom 4 remains untouched
    assert report["groups"][0]["target_total_charge"] == -0.8
    assert report["groups"][0]["atom_indices"] == [1, 2]


def test_rewrite_preserve_residues_keeps_polymer_residue_names():
    itp = """[ moleculetype ]
POL 3

[ atoms ]
1 C 1 RU1 C1 1 -0.1 12.011
2 O 2 RU2 O1 1 -0.2 15.999
"""
    gro = """test
2
    1RU1    C1    1   0.000   0.000   0.000
    2RU2    O1    2   0.100   0.100   0.100
   1.00000   1.00000   1.00000
"""
    rewritten_itp = _rewrite_itp_moltype_and_resname(itp, "POLYMER", preserve_residues=True)
    rewritten_gro = _rewrite_gro_resname(gro, "POLYMER", preserve_residues=True)
    assert "RU1" in rewritten_itp and "RU2" in rewritten_itp
    assert "RU1" in rewritten_gro and "RU2" in rewritten_gro


def test_scale_charged_groups_inplace_scales_only_target_group():
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    charges = [0.1, 0.2, -0.6, -0.7]
    for atom, q in zip(mol.GetAtoms(), charges):
        atom.SetDoubleProp("AtomicCharge", q)
    summary = annotate_polyelectrolyte_metadata(mol)["summary"]
    report = scale_charged_groups_inplace(mol, scale=0.8, groups=summary["groups"])
    assert report["groups"]
    unchanged = mol.GetAtomWithIdx(0).GetDoubleProp("AtomicCharge")
    assert abs(unchanged - 0.1) < 1.0e-12


def test_localized_charge_group_semantics_distinguish_carboxylate_from_tfsi():
    acetate = Chem.MolFromSmiles("CC(=O)[O-]")
    tfsi = Chem.MolFromSmiles("FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F")
    pf6 = Chem.MolFromSmiles("F[P-](F)(F)(F)(F)F")

    acetate_summary = detect_charged_groups(acetate, detection="auto")
    tfsi_summary = detect_charged_groups(tfsi, detection="auto")
    pf6_summary = detect_charged_groups(pf6, detection="auto")

    assert uses_localized_charge_groups(acetate_summary) is True
    assert uses_localized_charge_groups(tfsi_summary) is False
    assert uses_localized_charge_groups(pf6_summary) is False


def test_annotate_polyelectrolyte_metadata_marks_graph_only_small_ions_as_whole_molecule_scale():
    tfsi = Chem.MolFromSmiles("FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F")
    annotated = annotate_polyelectrolyte_metadata(tfsi)
    assert annotated["constraints"]["mode"] == "whole_molecule_scale"
    assert annotated["constraints"]["fallback"] == "whole_molecule_scale"

    acetate = Chem.MolFromSmiles("CC(=O)[O-]")
    annotated_acetate = annotate_polyelectrolyte_metadata(acetate)
    assert annotated_acetate["constraints"]["mode"] == "grouped"
