from __future__ import annotations

import importlib.util
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "02_polymer_electrolyte"
    / "benchmark_carbonate_lipf6_gaff2.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_carbonate_lipf6_gaff2", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_charge_recipe_family_map():
    mod = _load_module()
    recipe = mod._charge_recipe_from_family("wb97m_v")
    assert recipe["family"] == "wb97m_v"
    assert recipe["opt_method"] == "wb97m-v"
    assert recipe["charge_basis"] == "def2-TZVP"

    recipe = mod._charge_recipe_from_family("m06-2x")
    assert recipe["family"] == "m06_2x"
    assert recipe["charge_method"] == "m06-2x"


def test_normalize_gaff_variant():
    mod = _load_module()
    assert mod._normalize_gaff_variant("classic") == "classic"
    assert mod._normalize_gaff_variant("gaff2_mod") == "mod"
    assert mod._normalize_gaff_variant(None) == "classic"


def test_normalize_resp_profile():
    mod = _load_module()
    assert mod._normalize_resp_profile("adaptive") == "adaptive"
    assert mod._normalize_resp_profile("legacy") == "legacy"
    assert mod._normalize_resp_profile(None) == "adaptive"


def test_normalize_solvent_source():
    mod = _load_module()
    assert mod._normalize_solvent_source("qm") == "qm"
    assert mod._normalize_solvent_source("moldb") == "moldb"
    assert mod._normalize_solvent_source("ready_db") == "moldb"
    assert mod._normalize_solvent_source(None) == "qm"


def test_normalize_db_priority():
    mod = _load_module()
    assert mod._normalize_db_priority("auto") == "auto"
    assert mod._normalize_db_priority("repo_first") == "repo_first"
    assert mod._normalize_db_priority("repo") == "repo_first"
    assert mod._normalize_db_priority("default_first") == "default_first"
    assert mod._normalize_db_priority(None) == "auto"


def test_normalize_equilibration_mode_and_constraints():
    mod = _load_module()
    assert mod._normalize_equilibration_mode(None) == "auto"
    assert mod._normalize_equilibration_mode("cemp_like") == "liquid_anneal"
    assert mod._normalize_equilibration_mode("EQ21") == "eq21"
    assert mod._normalize_constraints(None) == "h-bonds"
    assert mod._normalize_constraints("hbonds") == "h-bonds"
    assert mod._normalize_constraints("all_bonds") == "all-bonds"


def test_summarize_carbonate_charge_features():
    mod = _load_module()
    mol = Chem.AddHs(Chem.MolFromSmiles("O=C1OCCO1"))
    assert mol is not None
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.UFFOptimizeMolecule(mol)

    for atom in mol.GetAtoms():
        q = 0.0
        if atom.GetSymbol() == "O":
            q = -0.4
            for bond in atom.GetBonds():
                if bond.GetOtherAtom(atom).GetSymbol() == "C" and bond.GetBondTypeAsDouble() >= 1.5:
                    q = -0.6
                    break
        elif atom.GetSymbol() == "C":
            for bond in atom.GetBonds():
                if bond.GetOtherAtom(atom).GetSymbol() == "O" and bond.GetBondTypeAsDouble() >= 1.5:
                    q = 0.9
                    break
        elif atom.GetSymbol() == "H":
            q = 0.05
        atom.SetDoubleProp("AtomicCharge", float(q))

    summary = mod._summarize_carbonate_charge_features(mol, label="EC")
    assert summary["label"] == "EC"
    assert summary["carbonyl_oxygen_charge_e"] == -0.6
    assert summary["carbonyl_carbon_charge_e"] == 0.9
    assert summary["ether_oxygen_charge_mean_e"] == -0.4
    assert summary["point_charge_dipole_debye"] is not None


def test_normalize_charge_mode_rejects_resp2():
    mod = _load_module()
    assert mod._normalize_charge_mode("resp") == "resp"
    try:
        mod._normalize_charge_mode("RESP2")
    except ValueError as exc:
        assert "RESP2" in str(exc)
    else:
        raise AssertionError("RESP2 should be rejected for this benchmark")


def test_equivalence_spread_diagnostic_uses_resp_constraints():
    mod = _load_module()
    mol = Chem.MolFromSmiles("CC")
    for atom, charge in zip(mol.GetAtoms(), (-0.2, 0.1)):
        atom.SetDoubleProp("AtomicCharge", float(charge))
    mol.SetProp("_yadonpy_resp_constraints_json", '{"equivalence_groups": [[0, 1]], "mode": "whole_molecule_scale"}')

    diagnostic = mod._equivalence_spread_diagnostic(mol, label="ETH")

    assert diagnostic["label"] == "ETH"
    assert diagnostic["group_count"] == 1
    assert abs(float(diagnostic["max_spread_e"]) - 0.3) < 1.0e-12


def test_solvent_diffusion_diagnostic_reports_expected_order():
    mod = _load_module()
    diagnostic = mod._solvent_diffusion_diagnostic(
        {"EC": 1.0e-10, "EMC": 3.0e-10, "DEC": 2.0e-10, "Li": 0.5e-10}
    )

    assert diagnostic["observed_order_fast_to_slow"] == ["EMC", "DEC", "EC"]
    assert diagnostic["matches_expected_for_present_species"] is True
    assert diagnostic["relative_to_slowest"]["EMC"] == 3.0


def test_solvent_diffusion_diagnostic_flags_dec_ec_inversion():
    mod = _load_module()
    diagnostic = mod._solvent_diffusion_diagnostic({"EC": 2.0e-10, "EMC": 3.0e-10, "DEC": 1.0e-10})

    assert diagnostic["observed_order_fast_to_slow"] == ["EMC", "EC", "DEC"]
    assert diagnostic["matches_expected_for_present_species"] is False
    assert diagnostic["pairwise_expected"][1]["faster"] == "DEC"
    assert diagnostic["pairwise_expected"][1]["slower"] == "EC"
    assert diagnostic["pairwise_expected"][1]["ok"] is False


def test_msd_block_diffusion_summary_reports_order_stability():
    mod = _load_module()
    blocks = [
        {
            "block_index": 0,
            "time_start_ps": 0.0,
            "time_end_ps": 1000.0,
            "diffusion_m2_s": {"EC": 1.0e-10, "EMC": 3.0e-10, "DEC": 2.0e-10},
        },
        {
            "block_index": 1,
            "time_start_ps": 1000.0,
            "time_end_ps": 2000.0,
            "diffusion_m2_s": {"EC": 1.1e-10, "EMC": 2.8e-10, "DEC": 1.9e-10},
        },
        {
            "block_index": 2,
            "time_start_ps": 2000.0,
            "time_end_ps": 3000.0,
            "diffusion_m2_s": {"EC": 2.2e-10, "EMC": 3.1e-10, "DEC": 1.7e-10},
        },
    ]

    summary = mod._summarize_msd_block_diffusion(blocks)

    assert summary["status"] == "ok"
    assert summary["species"]["EMC"]["mean_D_m2_s"] > summary["species"]["DEC"]["mean_D_m2_s"]
    assert summary["expected_order_match_fraction"] == 2 / 3
    assert summary["ranking_confidence"] == "ambiguous"
    assert summary["block_order_counts"]["EMC>DEC>EC"] == 2
    assert summary["pairwise_expected"][0]["faster"] == "EMC"
    assert summary["pairwise_expected"][0]["ok_fraction"] == 1.0
    assert summary["pairwise_expected"][1]["faster"] == "DEC"
    assert summary["pairwise_expected"][1]["ok_fraction"] == 2 / 3


def test_default_msd_trajectory_bounds_prefers_stored_absolute_times(tmp_path):
    mod = _load_module()
    csv_path = tmp_path / "msd_EC.csv"
    csv_path.write_text("time_ps,msd_nm2\n0,0\n10,1\n", encoding="utf-8")
    msd = {
        "EC": {
            "default_metric": "molecule_com_msd",
            "metrics": {
                "molecule_com_msd": {
                    "series_csv": str(csv_path),
                    "trajectory_time_start_ps": 100.0,
                    "trajectory_time_end_ps": 600.0,
                }
            },
        }
    }

    assert mod._default_msd_trajectory_bounds(msd) == (100.0, 600.0)
