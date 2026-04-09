from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from yadonpy.gmx.analysis.auto_plot import plot_msd_series, plot_rdf_cn_series
from yadonpy.gmx.analysis.structured import (
    build_ne_conductivity_from_msd,
    build_site_map,
    detect_first_shell,
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
    msd_nm2 = 1.0 - np.exp(-t_ps / 50.0)
    fit = select_diffusive_window(t_ps, msd_nm2)
    assert fit["D_m2_s"] is None
    assert fit["status"] in {"failed", "no_formal_diffusion"}


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
