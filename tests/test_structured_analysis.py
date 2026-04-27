from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from yadonpy.gmx.analysis.conductivity import EHFit, classify_eh_confidence
from yadonpy.gmx.analysis import conductivity as conductivity_mod
from yadonpy.gmx.engine import GromacsError
from yadonpy.gmx.analysis.auto_plot import plot_msd_series, plot_rdf_cn_series
from yadonpy.gmx.analysis.structured import (
    _compute_mobile_drift_series,
    GroupSpec,
    build_ne_conductivity_from_msd,
    build_site_map,
    compute_msd_series,
    detect_first_shell,
    preprocess_group_positions,
    resolve_moltypes_from_mols,
    select_diffusive_window,
)
from yadonpy.gmx.topology import MoleculeType, SystemTopology
from yadonpy.sim.analyzer import AnalyzeResult


def test_select_diffusive_window_accepts_linear_diffusion():
    t_ps = np.linspace(0.0, 1000.0, 201)
    msd_nm2 = 0.0 + 0.012 * t_ps
    fit = select_diffusive_window(t_ps, msd_nm2)
    assert fit["confidence"] in {"high", "medium"}
    assert fit["D_m2_s"] is not None
    assert fit["fit_slope_nm2_ps"] > 0.0


def test_select_diffusive_window_respects_geometry_divisor():
    t_ps = np.linspace(0.0, 1000.0, 201)
    msd_nm2 = 0.0 + 0.012 * t_ps
    fit_3d = select_diffusive_window(t_ps, msd_nm2, geometry="3d")
    fit_xy = select_diffusive_window(t_ps, msd_nm2, geometry="xy")
    fit_z = select_diffusive_window(t_ps, msd_nm2, geometry="z")

    assert fit_3d["geometry"] == "3d"
    assert fit_xy["geometry"] == "xy"
    assert fit_z["geometry"] == "z"
    assert fit_xy["D_m2_s"] > fit_3d["D_m2_s"]
    assert fit_z["D_m2_s"] > fit_xy["D_m2_s"]


def test_select_diffusive_window_rejects_plateau():
    t_ps = np.linspace(0.0, 1000.0, 201)
    msd_nm2 = 0.02 * np.power(np.maximum(t_ps, 1.0), 0.6)
    fit = select_diffusive_window(t_ps, msd_nm2)
    assert fit["D_m2_s"] is not None
    assert fit["selection_basis"] == "loglog_slope_closest_to_one"
    assert fit["status"] == "subdiffusive_risk"
    assert fit["confidence"] == "low"
    assert fit["alpha_mean"] < 1.0


def test_compute_msd_series_respects_begin_end_window(monkeypatch):
    t_ps = np.arange(0.0, 10.0, dtype=float)
    positions = np.zeros((10, 1, 3), dtype=float)
    positions[:, 0, 0] = t_ps

    def _fake_preprocess_group_positions(**kwargs):
        return {
            "t_ps": t_ps,
            "positions_nm": positions,
            "box_lengths_nm": np.ones((10, 3), dtype=float),
            "preprocessing": {
                "used_unwrapped_positions": True,
                "drift_correction_mode": "off",
                "drift_reference_group": None,
                "geometry_mode": "3d",
            },
        }

    monkeypatch.setattr(
        "yadonpy.gmx.analysis.structured.preprocess_group_positions",
        _fake_preprocess_group_positions,
    )

    out = compute_msd_series(
        gro_path=Path("system.gro"),
        xtc_path=Path("md.xtc"),
        top_path=Path("system.top"),
        system_dir=Path("02_system"),
        group_specs=[],
        begin_ps=3.0,
        end_ps=7.0,
    )

    assert np.allclose(out["t_ps"], np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert out["trajectory_time_start_ps"] == pytest.approx(3.0)
    assert out["trajectory_time_end_ps"] == pytest.approx(7.0)
    assert out["n_groups"] == 0


def test_preprocess_group_positions_makes_small_molecule_com_whole_across_pbc(monkeypatch, tmp_path: Path):
    class _FakeTrajectory:
        time = np.asarray([0.0, 1.0], dtype=float)
        unitcell_lengths = np.asarray([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=float)
        xyz = np.asarray(
            [
                [[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]],
            ],
            dtype=float,
        )

    fake_md = types.SimpleNamespace(iterload=lambda *args, **kwargs: iter([_FakeTrajectory()]))
    monkeypatch.setitem(sys.modules, "mdtraj", fake_md)

    spec = GroupSpec(
        group_id="tfsi:0",
        label="tfsi",
        moltype="tfsi",
        species_kind="ion",
        atom_indices_0=np.asarray([0, 1], dtype=int),
        masses=np.asarray([1.0, 1.0], dtype=float),
        formal_charge_e=-1.0,
    )

    out = preprocess_group_positions(
        gro_path=tmp_path / "system.gro",
        xtc_path=tmp_path / "md.xtc",
        top_path=tmp_path / "system.top",
        system_dir=tmp_path / "02_system",
        group_specs=[spec],
        unwrap="auto",
        drift="off",
    )

    x = np.asarray(out["positions_nm"], dtype=float)[:, 0, 0]
    assert x[0] == pytest.approx(10.0)
    assert x[1] == pytest.approx(10.3)
    assert x[1] - x[0] == pytest.approx(0.3)


def test_detect_first_shell_returns_confident_minimum():
    r = np.linspace(0.02, 1.20, 300)
    g = 1.0 + 2.4 * np.exp(-((r - 0.23) / 0.045) ** 2) - 0.35 * np.exp(-((r - 0.42) / 0.05) ** 2)
    g = np.clip(g, 0.2, None)
    cn = np.cumsum(g) * (r[1] - r[0])
    shell = detect_first_shell(r, g, cn)
    assert shell["r_peak_nm"] is not None
    assert shell["r_shell_nm"] is not None
    assert shell["confidence"] in {"high", "medium"}


def test_build_ne_conductivity_from_msd_prefers_polymer_charged_groups():
    msd_payload = {
        "CMC": {
            "kind": "polymer",
            "natoms": 400,
            "n_molecules": 2,
            "formal_charge_e": -20.0,
            "default_metric": "chain_com_msd",
            "metrics": {
                "chain_com_msd": {"D_m2_s": 1.0e-12},
                "charged_group_com_msd": {
                    "component_metrics": {
                        "anion:carboxylate:q-1": {
                            "component_label": "anion:carboxylate",
                            "formal_charge_e": -1.0,
                            "charge_sign": "anion",
                            "n_groups": 20,
                            "D_m2_s": 2.0e-10,
                        }
                    }
                },
            },
        },
        "Li": {
            "kind": "cation",
            "natoms": 1,
            "n_molecules": 20,
            "formal_charge_e": 1.0,
            "default_metric": "ion_atomic_msd",
            "metrics": {"ion_atomic_msd": {"D_m2_s": 1.5e-9}},
        },
    }
    out = build_ne_conductivity_from_msd(msd_payload=msd_payload, volume_nm3=100.0, temp_k=300.0)
    kinds = {c["component_kind"] for c in out["components"]}
    assert "polymer_charged_group" in kinds
    assert "species_default" in kinds
    assert out["NE_is_upper_bound"] is True
    assert out["sigma_ne_upper_bound_S_m"] == pytest.approx(out["sigma_S_m"])
    assert out["polymer_charged_group_self_ne_contribution_S_m"] > 0.0
    assert not any(c.get("moltype") == "CMC" and c.get("component_kind") == "species_default" for c in out["components"])
    assert all(c["charge_e"] in {-1.0, 1.0} for c in out["components"])
    assert {c.get("charge_sign") for c in out["components"] if c.get("component_kind") == "polymer_charged_group"} == {"anion"}
    assert all(c.get("interpretation") == "self_upper_bound" for c in out["components"])
    assert all(c.get("component_semantics") for c in out["components"])


def test_build_ne_conductivity_from_msd_rejects_charged_polymer_without_group_metric():
    msd_payload = {
        "PQA": {
            "kind": "polymer",
            "natoms": 120,
            "n_molecules": 1,
            "formal_charge_e": 8.0,
            "default_metric": "chain_com_msd",
            "metrics": {
                "chain_com_msd": {"D_m2_s": 8.0e-12},
            },
        },
    }
    out = build_ne_conductivity_from_msd(msd_payload=msd_payload, volume_nm3=120.0, temp_k=300.0)
    assert out["sigma_S_m"] == pytest.approx(0.0)
    assert out["components"] == []
    assert any(
        item.get("moltype") == "PQA"
        and item.get("component_kind") == "polymer_charged_group"
        and "whole-chain conductivity is disabled" in str(item.get("reason"))
        for item in out["ignored_components"]
    )


def test_build_ne_conductivity_from_msd_accepts_netzero_polymer_group_metrics():
    msd_payload = {
        "ZW": {
            "kind": "polymer",
            "natoms": 240,
            "n_molecules": 1,
            "formal_charge_e": 0.0,
            "default_metric": "chain_com_msd",
            "metrics": {
                "chain_com_msd": {"D_m2_s": 6.0e-12},
                "charged_group_com_msd": {
                    "component_metrics": {
                        "cation:quat:q+1": {
                            "component_label": "cation:quat",
                            "formal_charge_e": 1.0,
                            "charge_sign": "cation",
                            "n_groups": 8,
                            "D_m2_s": 9.0e-11,
                        },
                        "anion:sulfonate:q-1": {
                            "component_label": "anion:sulfonate",
                            "formal_charge_e": -1.0,
                            "charge_sign": "anion",
                            "n_groups": 8,
                            "D_m2_s": 7.5e-11,
                        },
                    }
                },
            },
        },
    }
    out = build_ne_conductivity_from_msd(msd_payload=msd_payload, volume_nm3=150.0, temp_k=300.0)
    assert len(out["components"]) == 2
    assert {c["charge_sign"] for c in out["components"]} == {"cation", "anion"}
    assert all(c["component_kind"] == "polymer_charged_group" for c in out["components"])


def test_build_ne_conductivity_from_msd_marks_subdiffusive_mobile_ion_risk():
    msd_payload = {
        "Li": {
            "kind": "cation",
            "natoms": 1,
            "n_molecules": 20,
            "formal_charge_e": 1.0,
            "default_metric": "ion_atomic_msd",
            "metrics": {
                "ion_atomic_msd": {
                    "D_m2_s": 1.2e-10,
                    "status": "subdiffusive_risk",
                    "confidence": "low",
                    "alpha_mean": 0.72,
                }
            },
        },
    }
    out = build_ne_conductivity_from_msd(msd_payload=msd_payload, volume_nm3=120.0, temp_k=353.0)
    assert out["sigma_ne_upper_bound_S_m"] > 0.0
    assert out["mobile_ion_subdiffusive_risk"] is True
    assert "risk:" in str(out["sigma_ne_upper_bound_display"])
    assert "subdiffusive" in str(out["sigma_ne_upper_bound_note"])
    assert out["risk_annotations"]


def test_build_site_map_classifies_common_sites(tmp_path: Path):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "system_meta.json").write_text(
        json.dumps({"species": [{"moltype": "EC", "smiles": "O=C1OCCO1", "kind": "solvent"}]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (system_dir / "residue_map.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    (system_dir / "charge_groups.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    top = SystemTopology(
        moleculetypes={
            "EC": MoleculeType(
                name="EC",
                atomtypes=["o", "c", "os", "c3", "c3", "os"],
                atomnames=["O1", "C1", "O2", "C2", "C3", "O3"],
                charges=[-0.5, 0.7, -0.3, 0.1, 0.1, -0.3],
                masses=[15.999, 12.011, 15.999, 12.011, 12.011, 15.999],
                bonds=[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 2)],
            )
        },
        molecules=[("EC", 2)],
    )
    site_map = build_site_map(top, system_dir)
    labels = {sp["site_label"] for sp in site_map["site_groups"]}
    assert "carbonyl_oxygen" in labels or "oxygen_site" in labels
    assert "ether_oxygen" in labels or "oxygen_site" in labels


def test_build_site_map_marks_fluorine_as_weaker_coordination_site(tmp_path: Path):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "system_meta.json").write_text(
        json.dumps(
            {
                "species": [
                    {"moltype": "TFSI", "smiles": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "kind": "ion"}
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (system_dir / "residue_map.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    (system_dir / "charge_groups.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    top = SystemTopology(
        moleculetypes={
            "TFSI": MoleculeType(
                name="TFSI",
                atomtypes=["f", "c3f", "f", "f", "s6", "o", "o", "n", "s6", "o", "o", "c3f", "f", "f", "f"],
                atomnames=["F1", "C2", "F3", "F4", "S5", "O6", "O7", "N8", "S9", "O10", "O11", "C12", "F13", "F14", "F15"],
                charges=[0.0] * 15,
                masses=[18.998, 12.011, 18.998, 18.998, 32.067, 15.999, 15.999, 14.007, 32.067, 15.999, 15.999, 12.011, 18.998, 18.998, 18.998],
                bonds=[(1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 8), (8, 9), (9, 10), (9, 11), (9, 12), (12, 13), (12, 14), (12, 15)],
            )
        },
        molecules=[("TFSI", 1)],
    )
    site_map = build_site_map(top, system_dir)
    entries = {entry["site_label"]: entry for entry in site_map["site_groups"]}
    assert entries["fluorine_site"]["coordination_relevance"] == "weak"
    assert entries["sulfonyl_oxygen"]["coordination_relevance"] == "primary"
    assert entries["fluorine_site"]["coordination_priority"] > entries["sulfonyl_oxygen"]["coordination_priority"]


def test_build_site_map_recovers_elements_from_opls_atom_labels(tmp_path: Path):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "system_meta.json").write_text(
        json.dumps({"species": [{"moltype": "EMC", "smiles": "CCOC(=O)OC", "kind": "solvent"}]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (system_dir / "residue_map.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    (system_dir / "charge_groups.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    top = SystemTopology(
        moleculetypes={
            "EMC": MoleculeType(
                name="EMC",
                atomtypes=[
                    "opls_135",
                    "opls_490",
                    "opls_467",
                    "opls_465",
                    "opls_466",
                    "opls_467",
                    "opls_468",
                    "opls_140",
                    "opls_140",
                    "opls_140",
                    "opls_469",
                    "opls_469",
                    "opls_469",
                    "opls_469",
                    "opls_469",
                ],
                atomnames=[
                    "opls_135",
                    "opls_490",
                    "opls_467",
                    "opls_465",
                    "opls_466",
                    "opls_467",
                    "opls_468",
                    "opls_140",
                    "opls_140",
                    "opls_140",
                    "opls_469",
                    "opls_469",
                    "opls_469",
                    "opls_469",
                    "opls_469",
                ],
                charges=[0.0] * 15,
                masses=[
                    12.011,
                    12.011,
                    15.999,
                    12.011,
                    15.999,
                    15.999,
                    12.011,
                    1.008,
                    1.008,
                    1.008,
                    1.008,
                    1.008,
                    1.008,
                    1.008,
                    1.008,
                ],
                bonds=[
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (4, 6),
                    (6, 7),
                    (1, 8),
                    (1, 9),
                    (1, 10),
                    (2, 11),
                    (2, 12),
                    (7, 13),
                    (7, 14),
                    (7, 15),
                ],
            )
        },
        molecules=[("EMC", 1)],
    )
    site_map = build_site_map(top, system_dir)
    labels = {sp["site_label"] for sp in site_map["site_groups"]}
    assert "carbonyl_oxygen" in labels
    assert "ether_oxygen" in labels


def test_resolve_moltypes_from_mols_uses_smiles_from_system_meta(tmp_path: Path):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "system_meta.json").write_text(
        json.dumps({"species": [{"moltype": "Li", "smiles": "[Li+]", "kind": "cation"}]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (system_dir / "residue_map.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    (system_dir / "charge_groups.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    li = Chem.MolFromSmiles("[Li+]")
    assert resolve_moltypes_from_mols(system_dir, li) == ["Li"]


def test_plot_series_helpers_emit_svg(tmp_path: Path):
    t_ps = np.linspace(0.0, 1000.0, 101)
    msd_nm2 = 0.01 * t_ps
    plots = plot_msd_series(
        t_ps=t_ps,
        msd_nm2=msd_nm2,
        out_dir=tmp_path / "plots",
        group="Li_ion_atomic_msd",
        fit_t_start_ps=200.0,
        fit_t_end_ps=800.0,
        confidence="high",
        status="ok",
        geometry="3d",
        alpha_mean=1.0,
        selection_basis="loglog_slope_closest_to_one",
        D_m2_s=2.0e-9,
    )
    assert Path(plots["msd_svg"]).exists()
    assert Path(plots["msd_loglog_svg"]).exists()

    r = np.linspace(0.02, 1.20, 300)
    g = 1.0 + 2.4 * np.exp(-((r - 0.23) / 0.045) ** 2) - 0.35 * np.exp(-((r - 0.42) / 0.05) ** 2)
    g = np.clip(g, 0.2, None)
    cn = np.cumsum(g) * (r[1] - r[0])
    svg = plot_rdf_cn_series(r_nm=r, g_r=g, cn_curve=cn, out_svg=tmp_path / "plots" / "rdf.svg", title="test")
    assert svg is not None and Path(svg).exists()


def test_rdf_strict_center_raises_when_species_is_missing(tmp_path: Path):
    work_dir = tmp_path
    system_dir = work_dir / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "system_meta.json").write_text(
        json.dumps({"species": [{"moltype": "EC", "smiles": "O=C1OCCO1", "kind": "solvent"}]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (system_dir / "residue_map.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    (system_dir / "charge_groups.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    top = work_dir / "system.top"
    itp = system_dir / "EC.itp"
    itp.write_text(
        "[ moleculetype ]\nEC 3\n\n[ atoms ]\n1 c 1 EC C1 1 0.0 12.011\n",
        encoding="utf-8",
    )
    top.write_text(f'#include "{itp.name}"\n\n[ molecules ]\nEC 1\n', encoding="utf-8")
    analyzer = AnalyzeResult(
        work_dir=work_dir,
        tpr=work_dir / "md.tpr",
        xtc=work_dir / "md.xtc",
        edr=work_dir / "md.edr",
        top=top,
        ndx=work_dir / "system.ndx",
    )
    li = Chem.MolFromSmiles("[Li+]")
    with pytest.raises(ValueError):
        analyzer.rdf(li, strict_center=True)


def test_rdf_accepts_center_only_call_style(tmp_path: Path, monkeypatch):
    work_dir = tmp_path
    analyzer = AnalyzeResult(
        work_dir=work_dir,
        tpr=work_dir / "md.tpr",
        xtc=work_dir / "md.xtc",
        edr=work_dir / "md.edr",
        top=work_dir / "system.top",
        ndx=work_dir / "system.ndx",
    )
    calls: dict[str, object] = {}
    monkeypatch.setattr(analyzer, "_transport_rdf_region", lambda region="auto": "global")
    monkeypatch.setattr(analyzer, "_strict_center_moltypes", lambda center_input, strict_center=True: ["Li"])
    monkeypatch.setattr(analyzer, "_resolve_species_moltypes", lambda mols: ["Li"])
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["parse_system_top"]),
        "parse_system_top",
        lambda top: object(),
    )
    monkeypatch.setattr(analyzer, "_system_dir", lambda: work_dir / "02_system")
    (work_dir / "02_system").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["build_species_catalog"]),
        "build_species_catalog",
        lambda topo, system_dir: {"Li": {"instances": [{"atom_indices_0": [0]}]}},
    )
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["build_site_map"]),
        "build_site_map",
        lambda topo, system_dir, include_h=False, selected_moltypes=None: {"site_groups": []},
    )
    monkeypatch.setattr(
        AnalyzeResult,
        "_update_summary_sections",
        lambda self, **kwargs: calls.setdefault("summary", kwargs),
    )

    out = analyzer.rdf(center_mol=Chem.MolFromSmiles("[Li+]"))
    assert isinstance(out, dict)
    assert "_overlay" in out


def test_rdf_defaults_site_workers_to_analyzer_omp(tmp_path: Path, monkeypatch):
    work_dir = tmp_path
    system_dir = work_dir / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)

    analyzer = AnalyzeResult(
        work_dir=work_dir,
        tpr=work_dir / "md.tpr",
        xtc=work_dir / "md.xtc",
        edr=work_dir / "md.edr",
        top=work_dir / "system.top",
        ndx=work_dir / "system.ndx",
        omp=6,
    )

    monkeypatch.setattr(analyzer, "_transport_rdf_region", lambda region="auto": "global")
    monkeypatch.setattr(analyzer, "_strict_center_moltypes", lambda center_input, strict_center=True: ["Li"])
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["parse_system_top"]),
        "parse_system_top",
        lambda top: object(),
    )
    monkeypatch.setattr(analyzer, "_system_dir", lambda: system_dir)
    monkeypatch.setattr(analyzer, "_analysis_xtc_path", lambda: work_dir / "md.xtc")
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["build_species_catalog"]),
        "build_species_catalog",
        lambda topo, system_dir: {"Li": {"instances": [{"atom_indices_0": [0]}]}},
    )
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["build_site_map"]),
        "build_site_map",
        lambda topo, system_dir, include_h=False, selected_moltypes=None: {
            "site_groups": [
                {
                    "site_id": "EC:carbonyl_oxygen",
                    "moltype": "EC",
                    "site_label": "carbonyl_oxygen",
                    "count": 4,
                    "coordination_priority": 0,
                    "coordination_relevance": "primary",
                    "coordination_note": "",
                    "atom_indices": [1, 2],
                },
                {
                    "site_id": "PF6:fluorine",
                    "moltype": "PF6",
                    "site_label": "fluorine",
                    "count": 6,
                    "coordination_priority": 1,
                    "coordination_relevance": "secondary",
                    "coordination_note": "",
                    "atom_indices": [3, 4],
                },
            ]
        },
    )

    calls: dict[str, object] = {}

    def _fake_compute_records(
        self,
        *,
        site_groups,
        center_group,
        center_indices,
        gro_path,
        xtc_path,
        bin_nm,
        r_max_nm,
        region_mode,
        workers,
        frame_stride=1,
    ):
        calls["workers"] = workers
        out = []
        for idx, site in enumerate(site_groups, start=1):
            out.append(
                {
                    "idx": idx,
                    "site": dict(site),
                    "site_id": str(site["site_id"]),
                    "rdf_data": {
                        "r_nm": np.asarray([0.10, 0.20, 0.30], dtype=float),
                        "g_r": np.asarray([0.0, 2.0, 0.8], dtype=float),
                        "cn_curve": np.asarray([0.0, 1.2, 1.8], dtype=float),
                        "rho_target_nm3": 5.0,
                        "shell": {
                            "r_peak_nm": 0.20,
                            "g_peak": 2.0,
                            "r_shell_nm": 0.30,
                            "cn_shell": 1.8,
                            "confidence": "high",
                        },
                    },
                }
            )
        return out

    monkeypatch.setattr(AnalyzeResult, "_compute_site_rdf_records", _fake_compute_records)

    out = analyzer.rdf(center_mol=Chem.MolFromSmiles("[Li+]"))

    assert calls["workers"] == 2
    assert "EC:carbonyl_oxygen" in out
    assert "PF6:fluorine" in out


def test_rdf_transport_fast_filters_sites_and_coarsens_defaults(tmp_path: Path, monkeypatch):
    work_dir = tmp_path
    system_dir = work_dir / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)

    analyzer = AnalyzeResult(
        work_dir=work_dir,
        tpr=work_dir / "md.tpr",
        xtc=work_dir / "md.xtc",
        edr=work_dir / "md.edr",
        top=work_dir / "system.top",
        ndx=work_dir / "system.ndx",
        omp=8,
    )

    monkeypatch.setattr(analyzer, "_transport_rdf_region", lambda region="auto": "global")
    monkeypatch.setattr(analyzer, "_strict_center_moltypes", lambda center_input, strict_center=True: ["Li"])
    monkeypatch.setattr(analyzer, "_system_dir", lambda: system_dir)
    monkeypatch.setattr(analyzer, "_analysis_xtc_path", lambda: work_dir / "md.xtc")
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["parse_system_top"]),
        "parse_system_top",
        lambda top: object(),
    )
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["build_species_catalog"]),
        "build_species_catalog",
        lambda topo, system_dir: {"Li": {"instances": [{"atom_indices_0": [0]}]}},
    )
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["build_site_map"]),
        "build_site_map",
        lambda topo, system_dir, include_h=False, selected_moltypes=None: {
            "site_groups": [
                {"site_id": "PEO:ether_oxygen", "moltype": "PEO", "site_label": "ether_oxygen", "count": 20, "atom_indices": [1, 2]},
                {"site_id": "TFSI:sulfonyl_oxygen", "moltype": "TFSI", "site_label": "sulfonyl_oxygen", "count": 8, "atom_indices": [3, 4]},
                {"site_id": "TFSI:anion_nitrogen", "moltype": "TFSI", "site_label": "anion_nitrogen", "count": 2, "atom_indices": [5]},
                {"site_id": "TFSI:fluorine_site", "moltype": "TFSI", "site_label": "fluorine_site", "count": 12, "atom_indices": [6, 7]},
            ]
        },
    )

    calls: dict[str, object] = {}

    def _fake_compute_records(
        self,
        *,
        site_groups,
        center_group,
        center_indices,
        gro_path,
        xtc_path,
        bin_nm,
        r_max_nm,
        region_mode,
        workers,
        frame_stride=1,
    ):
        calls["site_ids"] = [str(site["site_id"]) for site in site_groups]
        calls["bin_nm"] = bin_nm
        calls["r_max_nm"] = r_max_nm
        calls["frame_stride"] = frame_stride
        calls["workers"] = workers
        out = []
        for idx, site in enumerate(site_groups, start=1):
            out.append(
                {
                    "idx": idx,
                    "site": dict(site),
                    "site_id": str(site["site_id"]),
                    "rdf_data": {
                        "r_nm": np.asarray([0.10, 0.20, 0.30], dtype=float),
                        "g_r": np.asarray([0.0, 2.0, 0.8], dtype=float),
                        "cn_curve": np.asarray([0.0, 1.2, 1.8], dtype=float),
                        "rho_target_nm3": 5.0,
                        "shell": {"r_shell_nm": 0.30, "cn_shell": 1.8, "confidence": "high"},
                    },
                }
            )
        return out

    monkeypatch.setattr(AnalyzeResult, "_compute_site_rdf_records", _fake_compute_records)

    out = analyzer.rdf(center_mol=Chem.MolFromSmiles("[Li+]"), analysis_profile="transport_fast")

    assert calls["site_ids"] == ["PEO:ether_oxygen", "TFSI:sulfonyl_oxygen", "TFSI:anion_nitrogen"]
    assert calls["bin_nm"] == pytest.approx(0.005)
    assert calls["r_max_nm"] == pytest.approx(1.5)
    assert calls["frame_stride"] == 5
    assert calls["workers"] == 3
    assert "TFSI:fluorine_site" not in out
    assert out["_analysis"]["analysis_profile"] == "transport_fast"


def test_rdf_resume_reuses_matching_fresh_cache(tmp_path: Path, monkeypatch):
    work_dir = tmp_path
    system_dir = work_dir / "02_system"
    analysis_dir = work_dir / "06_analysis"
    system_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    xtc = work_dir / "md.xtc"
    top = work_dir / "system.top"
    gro = system_dir / "system.gro"
    meta = system_dir / "system_meta.json"
    for path in (xtc, top, gro, meta):
        path.write_text("x\n", encoding="utf-8")

    cached = {
        "PEO:ether_oxygen": {"center_group": "Li", "moltype": "PEO", "site_label": "ether_oxygen", "formal_cn_shell": 2.0},
        "_analysis": {
            "analysis_profile": "transport_fast",
            "center_group": "Li",
            "bin_nm": 0.005,
            "r_max_nm": 1.5,
            "frame_stride": 5,
            "site_filter": ["ether_oxygen", "sulfonyl_oxygen", "anion_nitrogen"],
            "region": "global",
        },
    }
    (analysis_dir / "rdf_first_shell.json").write_text(json.dumps(cached), encoding="utf-8")

    analyzer = AnalyzeResult(work_dir=work_dir, tpr=work_dir / "md.tpr", xtc=xtc, edr=work_dir / "md.edr", top=top, ndx=work_dir / "system.ndx")
    monkeypatch.setattr(analyzer, "_transport_rdf_region", lambda region="auto": "global")
    monkeypatch.setattr(analyzer, "_strict_center_moltypes", lambda center_input, strict_center=True: ["Li"])
    monkeypatch.setattr(analyzer, "_system_dir", lambda: system_dir)
    monkeypatch.setattr(analyzer, "_analysis_xtc_path", lambda: xtc)
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["parse_system_top"]),
        "parse_system_top",
        lambda top: object(),
    )
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["build_species_catalog"]),
        "build_species_catalog",
        lambda topo, system_dir: {"Li": {"instances": [{"atom_indices_0": [0]}]}},
    )
    monkeypatch.setattr(
        __import__("yadonpy.sim.analyzer", fromlist=["build_site_map"]),
        "build_site_map",
        lambda topo, system_dir, include_h=False, selected_moltypes=None: {
            "site_groups": [
                {"site_id": "PEO:ether_oxygen", "moltype": "PEO", "site_label": "ether_oxygen", "count": 20, "atom_indices": [1, 2]},
            ]
        },
    )

    def _should_not_compute(*args, **kwargs):
        raise AssertionError("RDF cache should have been reused")

    monkeypatch.setattr(AnalyzeResult, "_compute_site_rdf_records", _should_not_compute)

    out = analyzer.rdf(center_mol=Chem.MolFromSmiles("[Li+]"), analysis_profile="transport_fast", resume=True)
    assert out["PEO:ether_oxygen"]["formal_cn_shell"] == pytest.approx(2.0)


def test_get_all_prop_can_skip_polymer_metrics(tmp_path: Path, monkeypatch):
    work_dir = tmp_path
    system_dir = work_dir / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "system.gro").write_text("test\n0\n1.0 1.0 1.0\n", encoding="utf-8")

    class _FakeRunner:
        def energy_xvg(self, *, edr, out_xvg, terms, allow_missing=True):
            Path(out_xvg).write_text(
                '@ s0 legend "Temperature"\n'
                '@ s1 legend "Pressure"\n'
                '@ s2 legend "Density"\n'
                '@ s3 legend "Volume"\n'
                "0 300 1 1000 10\n"
                "1 301 1 1001 10\n",
                encoding="utf-8",
            )
            return {"resolved_terms": ["Temperature", "Pressure", "Density", "Volume"], "missing_terms": []}

    analyzer_mod = __import__("yadonpy.sim.analyzer", fromlist=["GromacsRunner"])
    monkeypatch.setattr(analyzer_mod, "GromacsRunner", _FakeRunner)
    monkeypatch.setattr(analyzer_mod, "compute_cell_summary", lambda **kwargs: {"lengths_nm": {}, "volume_nm3": {"mean": 10.0, "std": 0.0}})
    called = {"polymer_metrics": False}

    def _fake_polymer_metrics(**kwargs):
        called["polymer_metrics"] = True
        raise AssertionError("polymer metrics should be skipped")

    monkeypatch.setattr(analyzer_mod, "compute_polymer_metrics", _fake_polymer_metrics)

    analyzer = AnalyzeResult(work_dir=work_dir, tpr=work_dir / "md.tpr", xtc=work_dir / "md.xtc", edr=work_dir / "md.edr", top=work_dir / "system.top", ndx=work_dir / "system.ndx")
    monkeypatch.setattr(analyzer, "_system_dir", lambda: system_dir)
    monkeypatch.setattr(analyzer, "_polymer_moltypes_from_meta", lambda: ["PEO"])

    out = analyzer.get_all_prop(temp=300.0, press=1.0, include_polymer_metrics=False, analysis_profile="transport_fast")
    assert called["polymer_metrics"] is False
    assert out["polymer_metrics"] == {}
    assert out["_analysis"]["include_polymer_metrics"] is False


def test_msd_transport_fast_selects_transport_species_and_default_metrics(tmp_path: Path, monkeypatch):
    work_dir = tmp_path
    system_dir = work_dir / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system_meta.json", "residue_map.json", "charge_groups.json"):
        (system_dir / name).write_text("{}\n", encoding="utf-8")
    xtc = work_dir / "md.xtc"
    xtc.write_text("x\n", encoding="utf-8")

    analyzer = AnalyzeResult(work_dir=work_dir, tpr=work_dir / "md.tpr", xtc=xtc, edr=work_dir / "md.edr", top=work_dir / "system.top", ndx=work_dir / "system.ndx")
    monkeypatch.setattr(analyzer, "_system_dir", lambda: system_dir)
    monkeypatch.setattr(analyzer, "_analysis_xtc_path", lambda: xtc)
    monkeypatch.setattr(analyzer, "_transport_geometry_mode", lambda geometry="auto": "3d")

    analyzer_mod = __import__("yadonpy.sim.analyzer", fromlist=["parse_system_top"])
    monkeypatch.setattr(analyzer_mod, "parse_system_top", lambda top: object())
    monkeypatch.setattr(
        analyzer_mod,
        "build_msd_metric_catalog",
        lambda topo, system_dir: {
            "PEO": {
                "kind": "polymer",
                "smiles": "*CCO*",
                "n_molecules": 1,
                "natoms": 10,
                "formal_charge_e": 0.0,
                "default_metric": "chain_com_msd",
                "metrics": {"chain_com_msd": {"groups": [object()]}, "residue_com_msd": {"groups": [object()]}},
            },
            "Li": {
                "kind": "ion",
                "smiles": "[Li+]",
                "n_molecules": 2,
                "natoms": 1,
                "formal_charge_e": 1.0,
                "default_metric": "ion_atomic_msd",
                "metrics": {"ion_atomic_msd": {"groups": [object()]}},
            },
            "SOL": {
                "kind": "solvent",
                "smiles": "CO",
                "n_molecules": 3,
                "natoms": 6,
                "formal_charge_e": 0.0,
                "default_metric": "molecule_com_msd",
                "metrics": {"molecule_com_msd": {"groups": [object()]}},
            },
        },
    )
    calls: list[tuple[str, int]] = []

    def _fake_compute_msd_series(*, group_specs, **kwargs):
        calls.append((kwargs.get("geometry_mode"), len(group_specs)))
        return {
            "t_ps": np.asarray([0.0, 1.0, 2.0]),
            "msd_nm2": np.asarray([0.0, 0.1, 0.2]),
            "geometry": "3d",
            "n_groups": len(group_specs),
            "fit": {"D_m2_s": 1.0e-10, "D_nm2_ps": 1.0e-4, "status": "ok", "confidence": "high"},
            "preprocessing": {},
        }

    monkeypatch.setattr(analyzer_mod, "compute_msd_series", _fake_compute_msd_series)
    monkeypatch.setattr(analyzer_mod, "plot_msd_series", lambda **kwargs: {})
    monkeypatch.setattr(analyzer_mod, "plot_msd_series_summary", lambda **kwargs: None)

    out = analyzer.msd(analysis_profile="transport_fast")
    assert set(k for k in out if not k.startswith("_")) == {"PEO", "Li"}
    assert set(out["PEO"]["metrics"]) == {"chain_com_msd"}
    assert len(calls) == 2
    assert out["_analysis"]["analysis_profile"] == "transport_fast"


def test_dielectric_runs_gmx_dipoles_and_writes_summary(tmp_path: Path, monkeypatch):
    work_dir = tmp_path
    system_dir = work_dir / "02_system"
    analysis_dir = work_dir / "06_analysis"
    system_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        system_dir / "system_meta.json",
        system_dir / "system.gro",
        work_dir / "md.tpr",
        work_dir / "md.xtc",
        work_dir / "md.edr",
        work_dir / "system.ndx",
        work_dir / "system.top",
    ):
        path.write_text("x\n", encoding="utf-8")
    (analysis_dir / "basic_properties.json").write_text(
        json.dumps({"temperature_K": 333.15}, indent=2) + "\n",
        encoding="utf-8",
    )

    class _FakeProc:
        stdout = b"Epsilon = 12.34\n"
        stderr = b""

    class _FakeRunner:
        def dipoles(self, **kwargs):
            Path(kwargs["out_epsilon_xvg"]).write_text(
                "0 10.0 1.0 0.9\n"
                "100 12.34 1.5 1.2\n",
                encoding="utf-8",
            )
            Path(kwargs["out_mtot_xvg"]).write_text("0 0 0 0 0\n", encoding="utf-8")
            Path(kwargs["out_average_xvg"]).write_text("0 0 0\n", encoding="utf-8")
            Path(kwargs["out_distribution_xvg"]).write_text("0 0\n", encoding="utf-8")
            assert kwargs["temp_k"] == pytest.approx(333.15)
            assert kwargs["group"] == "System"
            return _FakeProc()

    analyzer_mod = __import__("yadonpy.sim.analyzer", fromlist=["GromacsRunner"])
    monkeypatch.setattr(analyzer_mod, "GromacsRunner", lambda: _FakeRunner())

    analyzer = AnalyzeResult(
        work_dir=work_dir,
        tpr=work_dir / "md.tpr",
        xtc=work_dir / "md.xtc",
        edr=work_dir / "md.edr",
        top=work_dir / "system.top",
        ndx=work_dir / "system.ndx",
    )
    monkeypatch.setattr(analyzer, "_analysis_dir", lambda: analysis_dir)
    monkeypatch.setattr(analyzer, "_system_dir", lambda: system_dir)
    monkeypatch.setattr(analyzer, "_analysis_xtc_path", lambda: work_dir / "md.xtc")

    out = analyzer.dielectric(resume=False)

    assert out["epsilon_static"] == pytest.approx(12.34)
    assert out["finite_system_kirkwood_g"] == pytest.approx(1.5)
    assert out["infinite_system_kirkwood_g"] == pytest.approx(1.2)
    assert out["temperature_source"] == "analysis_cache"
    assert (analysis_dir / "dielectric.json").exists()
    assert (analysis_dir / "dielectric" / "gmx_dipoles.log").exists()


def test_dielectric_resume_reuses_matching_cache(tmp_path: Path, monkeypatch):
    work_dir = tmp_path
    system_dir = work_dir / "02_system"
    analysis_dir = work_dir / "06_analysis"
    system_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    deps = [
        system_dir / "system_meta.json",
        work_dir / "md.tpr",
        work_dir / "md.xtc",
        work_dir / "system.ndx",
    ]
    for path in deps:
        path.write_text("x\n", encoding="utf-8")
    cached = {
        "epsilon_static": 8.8,
        "_analysis": {
            "group": "System",
            "temperature_K": 300.0,
            "begin_ps": None,
            "end_ps": None,
            "dt_ps": None,
            "epsilon_rf": 0.0,
        },
    }
    (analysis_dir / "dielectric.json").write_text(json.dumps(cached), encoding="utf-8")

    analyzer_mod = __import__("yadonpy.sim.analyzer", fromlist=["GromacsRunner"])

    class _UnexpectedRunner:
        def dipoles(self, **kwargs):
            raise AssertionError("dielectric cache should have been reused")

    monkeypatch.setattr(analyzer_mod, "GromacsRunner", lambda: _UnexpectedRunner())

    analyzer = AnalyzeResult(
        work_dir=work_dir,
        tpr=work_dir / "md.tpr",
        xtc=work_dir / "md.xtc",
        edr=work_dir / "md.edr",
        top=work_dir / "system.top",
        ndx=work_dir / "system.ndx",
    )
    monkeypatch.setattr(analyzer, "_analysis_dir", lambda: analysis_dir)
    monkeypatch.setattr(analyzer, "_system_dir", lambda: system_dir)
    monkeypatch.setattr(analyzer, "_analysis_xtc_path", lambda: work_dir / "md.xtc")

    out = analyzer.dielectric(temp_k=300.0, resume=True)
    assert out["epsilon_static"] == pytest.approx(8.8)


def test_sigma_falls_back_to_position_helfand_when_gmx_current_fails(tmp_path: Path, monkeypatch):
    work_dir = tmp_path
    system_dir = work_dir / "02_system"
    analysis_dir = work_dir / "06_analysis"
    system_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    gro = system_dir / "system.gro"
    gro.write_text(
        "test\n"
        "1\n"
        "    1LIT     Li    1   0.000   0.000   0.000\n"
        "   1.00000   1.00000   1.00000\n",
        encoding="utf-8",
    )
    top = work_dir / "system.top"
    top.write_text("", encoding="utf-8")
    ndx = work_dir / "system.ndx"
    ndx.write_text("[ IONS ]\n1\n", encoding="utf-8")
    tpr = work_dir / "md.tpr"
    xtc = work_dir / "md.xtc"
    trr = work_dir / "md.trr"
    edr = work_dir / "md.edr"
    for path in (tpr, xtc, trr, edr):
        path.write_text("", encoding="utf-8")

    analyzer = AnalyzeResult(work_dir=work_dir, tpr=tpr, xtc=xtc, trr=trr, edr=edr, top=top, ndx=ndx)

    monkeypatch.setattr(analyzer, "_analysis_dir", lambda: analysis_dir)
    monkeypatch.setattr(analyzer, "_system_dir", lambda: system_dir)
    monkeypatch.setattr(analyzer, "_analysis_xtc_path", lambda: xtc)

    import yadonpy.sim.analyzer as analyzer_mod

    monkeypatch.setattr(
        analyzer_mod,
        "build_ne_conductivity_from_msd",
        lambda **kwargs: {
            "sigma_S_m": 0.0,
            "sigma_ne_upper_bound_S_m": 0.0,
            "components": [],
            "ignored_components": [],
            "polymer_charged_group_self_ne_contribution_S_m": 0.0,
        },
    )

    class _BrokenRunner:
        def current(self, **kwargs):
            Path(kwargs["out_xvg"]).write_text("", encoding="utf-8")
            Path(kwargs["out_dsp"]).write_text("", encoding="utf-8")
            (Path(kwargs["out_xvg"]).parent / "_nojump.trr").write_text("temporary", encoding="utf-8")
            raise GromacsError("gmx current produced an empty -dsp output")

    monkeypatch.setattr(analyzer_mod, "GromacsRunner", lambda: _BrokenRunner())

    def _fake_write_eh_dsp_from_unwrapped_positions(**kwargs):
        out = Path(kwargs["out_dsp_xvg"])
        out.write_text(
            "0.0 0.0\n"
            "1000.0 1.0e-11\n"
            "2000.0 2.0e-11\n"
            "3000.0 3.0e-11\n"
            "4000.0 4.0e-11\n",
            encoding="utf-8",
        )
        return out

    monkeypatch.setattr(analyzer_mod, "write_eh_dsp_from_unwrapped_positions", _fake_write_eh_dsp_from_unwrapped_positions)
    monkeypatch.setattr(
        analyzer_mod,
        "conductivity_from_current_dsp",
        lambda path: EHFit(
            sigma_S_m=0.123,
            window_start_ps=1000.0,
            window_end_ps=4000.0,
            slope_per_ps=1.23e-13,
            intercept=0.0,
            r2=0.99,
            note="test-fit",
        ),
    )

    out = analyzer.sigma(msd={}, temp_k=353.0)

    assert out["sigma_eh_total_S_m"] == pytest.approx(0.123)
    assert out["collective_conductivity_unavailable"] is False
    assert out["eh"]["method"] == "helfand_unwrapped_positions"
    assert "gmx current produced an empty -dsp output" in str(out["eh"].get("gmx_current_warning"))
    cleaned = out["eh"].get("gmx_current_artifacts_cleaned") or []
    assert any(str(item).endswith("_nojump.trr") for item in cleaned)
    assert not (analyzer._analysis_dir() / "sigma" / "_nojump.trr").exists()


def test_sigma_gmx_current_only_marks_eh_unavailable_without_fallback(tmp_path: Path, monkeypatch):
    work_dir = tmp_path / "work_dir"
    analysis_dir = work_dir / "06_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    system_dir = work_dir / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)

    (analysis_dir / "thermo_summary.json").write_text(json.dumps({"Volume": {"mean": 123.0}}, indent=2) + "\n", encoding="utf-8")
    (system_dir / "system_meta.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    top = system_dir / "system.top"
    top.write_text("", encoding="utf-8")
    ndx = work_dir / "system.ndx"
    ndx.write_text("[ IONS ]\n1\n", encoding="utf-8")
    tpr = work_dir / "md.tpr"
    xtc = work_dir / "md.xtc"
    trr = work_dir / "md.trr"
    edr = work_dir / "md.edr"
    for path in (tpr, xtc, trr, edr):
        path.write_text("", encoding="utf-8")

    analyzer = AnalyzeResult(work_dir=work_dir, tpr=tpr, xtc=xtc, trr=trr, edr=edr, top=top, ndx=ndx)
    monkeypatch.setattr(analyzer, "_analysis_dir", lambda: analysis_dir)
    monkeypatch.setattr(analyzer, "_system_dir", lambda: system_dir)
    monkeypatch.setattr(analyzer, "_analysis_xtc_path", lambda: xtc)

    import yadonpy.sim.analyzer as analyzer_mod

    monkeypatch.setattr(
        analyzer_mod,
        "build_ne_conductivity_from_msd",
        lambda **kwargs: {
            "sigma_S_m": 1.0e-3,
            "sigma_ne_upper_bound_S_m": 1.0e-3,
            "components": [],
            "ignored_components": [],
            "polymer_charged_group_self_ne_contribution_S_m": 0.0,
        },
    )

    class _BrokenRunner:
        def current(self, **kwargs):
            Path(kwargs["out_xvg"]).write_text("", encoding="utf-8")
            Path(kwargs["out_dsp"]).write_text("", encoding="utf-8")
            (Path(kwargs["out_xvg"]).parent / "_nojump.trr").write_text("temporary", encoding="utf-8")
            raise GromacsError("gmx current failed for benchmark mode")

    monkeypatch.setattr(analyzer_mod, "GromacsRunner", lambda: _BrokenRunner())

    def _unexpected_fallback(**kwargs):
        raise AssertionError("positions-based EH fallback must not run when eh_mode='gmx_current_only'")

    monkeypatch.setattr(analyzer_mod, "write_eh_dsp_from_unwrapped_positions", _unexpected_fallback)

    out = analyzer.sigma(msd={}, temp_k=333.15, eh_mode="gmx_current_only")

    assert out["sigma_ne_upper_bound_S_m"] == pytest.approx(1.0e-3)
    assert out["sigma_eh_total_S_m"] is None
    assert out["collective_conductivity_unavailable"] is True
    assert out["eh"]["method"] is None
    assert out["eh"]["eh_mode"] == "gmx_current_only"
    assert "fallback is disabled" in str(out["eh"].get("reason"))
    cleaned = out["eh"].get("gmx_current_artifacts_cleaned") or []
    assert any(str(item).endswith("_nojump.trr") for item in cleaned)
    assert not (analyzer._analysis_dir() / "sigma" / "_nojump.trr").exists()


def test_classify_eh_confidence_marks_best_r2_fallback_as_low():
    fit = EHFit(
        sigma_S_m=1.0e-3,
        window_start_ps=1000.0,
        window_end_ps=3000.0,
        slope_per_ps=1.0e-15,
        intercept=0.0,
        r2=0.91,
        note="Auto EH window: no window passed r2>=thr (thr=0.980); using best-r2 window.",
    )
    confidence, reason = classify_eh_confidence(fit)
    assert confidence == "low"
    assert reason == "fallback_best_r2_window"


def test_compute_mobile_drift_series_unwraps_wrapped_mobile_com(tmp_path: Path, monkeypatch):
    class _FakeTraj:
        def __init__(self):
            self.xyz = np.asarray(
                [
                    [[0.90, 0.00, 0.00], [0.95, 0.00, 0.00]],
                    [[0.05, 0.00, 0.00], [0.10, 0.00, 0.00]],
                    [[0.15, 0.00, 0.00], [0.20, 0.00, 0.00]],
                ],
                dtype=float,
            )
            self.unitcell_lengths = np.ones((3, 3), dtype=float)
            self.time = np.asarray([0.0, 1.0, 2.0], dtype=float)

    fake_mdtraj = types.SimpleNamespace(iterload=lambda *args, **kwargs: [_FakeTraj()])
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)

    import yadonpy.gmx.analysis.structured as structured_mod

    monkeypatch.setattr(structured_mod, "parse_system_top", lambda path: object())
    monkeypatch.setattr(
        structured_mod,
        "_build_mobile_atom_payload",
        lambda top, system_dir: (np.asarray([0, 1], dtype=int), np.asarray([1.0, 1.0], dtype=float)),
    )
    structured_mod._MOBILE_DRIFT_CACHE.clear()

    t_ps, drift = _compute_mobile_drift_series(
        gro_path=tmp_path / "system.gro",
        xtc_path=tmp_path / "md.xtc",
        top_path=tmp_path / "system.top",
        system_dir=tmp_path,
        chunk=3,
    )

    assert np.allclose(t_ps, np.asarray([0.0, 1.0, 2.0], dtype=float))
    assert np.allclose(drift[:, 0], np.asarray([0.925, 1.075, 1.175], dtype=float))
    assert np.allclose(drift[:, 1:], 0.0)


def test_write_eh_dsp_from_unwrapped_positions_uses_gro_topology(tmp_path: Path, monkeypatch):
    xtc = tmp_path / "md.xtc"
    tpr = tmp_path / "md.tpr"
    top = tmp_path / "system.top"
    gro = tmp_path / "system.gro"
    out = tmp_path / "eh_dsp.xvg"
    for path in (xtc, tpr, top, gro):
        path.write_text("", encoding="utf-8")

    topo = SystemTopology(
        moleculetypes={
            "Li": MoleculeType(
                name="Li",
                atomtypes=["li"],
                atomnames=["Li"],
                charges=[1.0],
                masses=[6.94],
                bonds=[],
            )
        },
        molecules=[("Li", 1)],
    )
    monkeypatch.setattr(conductivity_mod, "parse_system_top", lambda path: topo)

    class _FakeTraj:
        def __init__(self, start: int, stop: int):
            self.n_atoms = 1
            self.time = np.asarray([float(i * 50.0) for i in range(start, stop)], dtype=float)
            self.xyz = np.asarray([[[0.1 * float(i), 0.0, 0.0]] for i in range(start, stop)], dtype=float)
            self.unitcell_lengths = np.ones((stop - start, 3), dtype=float)

    calls: dict[str, str] = {}

    def _fake_iterload(path: str, *, top: str, chunk: int):
        calls["path"] = path
        calls["top"] = top
        return [_FakeTraj(0, 5), _FakeTraj(5, 10)]

    fake_mdtraj = types.SimpleNamespace(iterload=_fake_iterload)
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)

    conductivity_mod.write_eh_dsp_from_unwrapped_positions(
        xtc=xtc,
        tpr=tpr,
        top=top,
        gro=gro,
        out_dsp_xvg=out,
        temp_k=353.0,
        vol_m3=1.0e-24,
    )

    assert calls["path"] == str(xtc)
    assert calls["top"] == str(gro)
    assert out.exists()
