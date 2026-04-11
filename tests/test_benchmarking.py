from __future__ import annotations

import json
from pathlib import Path

import pytest

from yadonpy.sim.benchmarking import (
    build_benchmark_compare,
    build_coordination_partition,
    build_screening_compare,
    build_transport_summary,
    collect_force_balance_report,
    literature_band_peo_litfsi_60c,
    load_benchmark_analysis_dir,
)


def test_build_coordination_partition_detects_anion_rich_bias():
    rdf = {
        "PEO:ether_oxygen": {
            "moltype": "PEO",
            "site_label": "ether_oxygen",
            "coordination_priority": 0,
            "coordination_relevance": "primary",
            "formal_cn_shell": 1.2,
        },
        "TFSI:sulfonyl_oxygen": {
            "moltype": "TFSI",
            "site_label": "sulfonyl_oxygen",
            "coordination_priority": 0,
            "coordination_relevance": "primary",
            "formal_cn_shell": 3.4,
        },
        "TFSI:fluorine_site": {
            "moltype": "TFSI",
            "site_label": "fluorine_site",
            "coordination_priority": 3,
            "coordination_relevance": "weak",
            "formal_cn_shell": 15.0,
        },
    }
    out = build_coordination_partition(rdf, polymer_moltype="PEO", anion_moltype="TFSI")
    assert out["coordination_bias"] == "anion_rich"
    assert out["anion_first_shell_cn"] > out["polymer_first_shell_cn"]
    assert out["primary_sites"][0]["site_label"] in {"ether_oxygen", "sulfonyl_oxygen"}


def test_build_transport_summary_marks_sampling_flags_and_compares_to_band(tmp_path: Path):
    thermo_xvg = tmp_path / "thermo.xvg"
    thermo_xvg.write_text(
        "@    title \"thermo\"\n"
        "@ s0 legend \"Density\"\n"
        "0.0 1200.0\n"
        "1.0 1210.0\n"
        "2.0 1235.0\n"
        "3.0 1255.0\n"
        "4.0 1280.0\n",
        encoding="utf-8",
    )
    msd = {
        "Li": {
            "default_metric": "ion_atomic_msd",
            "metrics": {
                "ion_atomic_msd": {
                    "D_m2_s": 8.0e-13,
                    "status": "subdiffusive_risk",
                    "alpha_mean": 0.52,
                }
            },
        }
    }
    sigma = {
        "sigma_ne_upper_bound_S_m": 2.0e-3,
        "sigma_eh_total_S_m": 8.0e-4,
        "haven_ratio": 0.4,
        "collective_conductivity_unavailable": False,
        "eh": {"confidence": "low", "quality_note": "fallback_best_r2_window", "method": "gmx current -dsp"},
    }
    rdf = {
        "PEO:ether_oxygen": {"moltype": "PEO", "site_label": "ether_oxygen", "coordination_priority": 0, "coordination_relevance": "primary", "formal_cn_shell": 1.0},
        "TFSI:sulfonyl_oxygen": {"moltype": "TFSI", "site_label": "sulfonyl_oxygen", "coordination_priority": 0, "coordination_relevance": "primary", "formal_cn_shell": 2.0},
    }
    out = build_transport_summary(
        msd=msd,
        sigma=sigma,
        rdf=rdf,
        polymer_moltype="PEO",
        anion_moltype="TFSI",
        thermo_xvg=thermo_xvg,
        literature_band=literature_band_peo_litfsi_60c(),
    )
    assert out["sampling_flags"]["li_subdiffusive"] is True
    assert out["sampling_flags"]["eh_low_confidence"] is True
    assert out["sampling_flags"]["density_drift_exceeds_2pct"] is True
    assert out["within_literature_band"] is False
    assert out["factor_below_literature_min"] > 1.0


def test_collect_force_balance_report_reads_charge_and_lj_params(tmp_path: Path):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    molecules = system_dir / "molecules"
    (molecules / "PEO").mkdir(parents=True, exist_ok=True)
    (molecules / "Li").mkdir(parents=True, exist_ok=True)
    (molecules / "TFSI").mkdir(parents=True, exist_ok=True)

    (system_dir / "system_meta.json").write_text(
        json.dumps(
            {
                "species": [
                    {"name": "PEO", "moltype": "PEO", "kind": "polymer", "smiles": "*CCO*", "n": 2, "formal_charge": 0.0, "charge_scale": 1.0},
                    {"name": "Li", "moltype": "Li", "kind": "ion", "smiles": "[Li+]", "n": 2, "formal_charge": 1.0, "charge_scale": 0.8},
                    {"name": "TFSI", "moltype": "TFSI", "kind": "ion", "smiles": "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F", "n": 2, "formal_charge": -1.0, "charge_scale": 0.8},
                ],
                "net_charge_raw": 0.0,
                "net_charge_scaled": 0.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    top = system_dir / "system.top"
    top.write_text(
        '#include "molecules/PEO/PEO.itp"\n'
        '#include "molecules/Li/Li.itp"\n'
        '#include "molecules/TFSI/TFSI.itp"\n\n'
        '[ molecules ]\n'
        'PEO 2\n'
        'Li 2\n'
        'TFSI 2\n',
        encoding="utf-8",
    )
    (molecules / "PEO" / "PEO.itp").write_text(
        "[ atomtypes ]\n"
        "c3 12.011 0.0000 A 0.339967 0.457730\n"
        "os 15.999 0.0000 A 0.296000 0.711280\n\n"
        "[ moleculetype ]\nPEO 3\n\n"
        "[ atoms ]\n"
        "1 c3 1 PEO C1 1 0.10 12.011\n"
        "2 os 1 PEO O2 2 -0.20 15.999\n"
        "3 c3 1 PEO C3 3 0.10 12.011\n\n"
        "[ bonds ]\n1 2\n2 3\n",
        encoding="utf-8",
    )
    (molecules / "Li" / "Li.itp").write_text(
        "[ atomtypes ]\n"
        "Li 6.94 0.0000 A 0.150000 0.076000\n\n"
        "[ moleculetype ]\nLi 3\n\n"
        "[ atoms ]\n"
        "1 Li 1 Li Li1 1 0.800000 6.94\n",
        encoding="utf-8",
    )
    (molecules / "TFSI" / "TFSI.itp").write_text(
        "[ atomtypes ]\n"
        "s6 32.067 0.0000 A 0.355000 1.046000\n"
        "o 15.999 0.0000 A 0.296000 0.878640\n"
        "n 14.007 0.0000 A 0.325000 0.711280\n"
        "f 18.998 0.0000 A 0.305000 0.255224\n"
        "c3f 12.011 0.0000 A 0.339967 0.457730\n\n"
        "[ moleculetype ]\nTFSI 3\n\n"
        "[ atoms ]\n"
        "1 c3f 1 TFSI C1 1 0.429088 12.011\n"
        "2 f   1 TFSI F2 2 -0.168652 18.998\n"
        "3 f   1 TFSI F3 3 -0.168652 18.998\n"
        "4 f   1 TFSI F4 4 -0.168652 18.998\n"
        "5 s6  1 TFSI S5 5 0.885514 32.067\n"
        "6 o   1 TFSI O6 6 -0.508368 15.999\n"
        "7 o   1 TFSI O7 7 -0.508368 15.999\n"
        "8 n   1 TFSI N8 8 -0.583821 14.007\n"
        "9 s6  1 TFSI S9 9 0.885514 32.067\n"
        "10 o  1 TFSI O10 10 -0.508368 15.999\n"
        "11 o  1 TFSI O11 11 -0.508368 15.999\n"
        "12 c3f 1 TFSI C12 12 0.429088 12.011\n"
        "13 f  1 TFSI F13 13 -0.168652 18.998\n"
        "14 f  1 TFSI F14 14 -0.168652 18.998\n"
        "15 f  1 TFSI F15 15 -0.168652 18.998\n\n"
        "[ bonds ]\n"
        "1 5\n5 6\n5 7\n5 8\n8 9\n9 10\n9 11\n9 12\n1 2\n1 3\n1 4\n12 13\n12 14\n12 15\n",
        encoding="utf-8",
    )

    out = collect_force_balance_report(
        system_dir=system_dir,
        top_path=top,
        species_pre_export=[{"label": "TFSI", "net_charge_e": -1.0, "scaled_net_charge_e": -0.8}],
        moltype_hints={"polymer": "PEO", "cation": "Li", "anion": "TFSI"},
    )

    labels = {(row["role"], row["site_label"]) for row in out["site_charge_lj_table"]}
    assert ("cation", "cation_center") in labels or ("cation", "cationic_site") in labels
    assert ("polymer", "ether_oxygen") in labels
    assert ("anion", "sulfonyl_oxygen") in labels
    assert "Li_sulfonyl_oxygen" in out["pair_energy_proxies"]
    assert out["system_meta_charge_audit"]["kind"] == "meta_scaled_raw"


def test_build_benchmark_compare_prefers_force_balance_when_anion_bias_is_strong():
    out = build_benchmark_compare(
        force_balance_report={
            "diagnosis": {"primary_force_balance_flag": "anion_proxy_stronger"},
        },
        coordination_partition={"coordination_bias": "anion_rich"},
        transport_summary={
            "sigma_eh_total_S_m": 8.0e-4,
            "sigma_ne_upper_bound_S_m": 4.0e-3,
            "factor_below_literature_min": 10.0,
            "sampling_flags": {
                "density_drift_fraction": 0.0,
                "density_drift_exceeds_2pct": False,
                "li_alpha_mean": 0.85,
                "li_subdiffusive": False,
                "eh_low_confidence": False,
                "haven_gt_one": False,
            },
        },
        charge_scale_polymer=1.0,
        charge_scale_li=0.8,
        charge_scale_anion=0.8,
        production_ns=10.0,
    )
    assert out["primary_cause"] == "force_balance_likely_anion_biased"
    assert out["factor_below_literature_min"] == pytest.approx(10.0)


def test_build_screening_compare_detects_force_balance_recovery():
    runs = [
        {
            "analysis_dir": "/tmp/s100",
            "benchmark_compare": {
                "charge_scale_polymer": 1.0,
                "charge_scale_li": 1.0,
                "charge_scale_anion": 1.0,
                "sigma_eh_total_S_m": None,
                "sigma_ne_upper_bound_S_m": 1.0e-3,
                "production_ns": 10.0,
            },
            "coordination_partition": {
                "coordination_bias": "anion_rich",
                "anion_to_polymer_cn_ratio": 3.0,
            },
            "transport_summary": {
                "li_alpha_mean": 0.30,
                "sampling_flags": {
                    "li_subdiffusive": True,
                },
            },
            "force_balance_report": {
                "diagnosis": {"primary_force_balance_flag": "anion_proxy_stronger"},
            },
        },
        {
            "analysis_dir": "/tmp/s080",
            "benchmark_compare": {
                "charge_scale_polymer": 1.0,
                "charge_scale_li": 0.8,
                "charge_scale_anion": 0.8,
                "sigma_eh_total_S_m": None,
                "sigma_ne_upper_bound_S_m": 1.2e-2,
                "production_ns": 10.0,
            },
            "coordination_partition": {
                "coordination_bias": "mixed",
                "anion_to_polymer_cn_ratio": 1.5,
            },
            "transport_summary": {
                "li_alpha_mean": 0.82,
                "sampling_flags": {
                    "li_subdiffusive": False,
                },
            },
            "force_balance_report": {
                "diagnosis": {"primary_force_balance_flag": "anion_proxy_stronger"},
            },
        },
    ]
    out = build_screening_compare(runs=runs)
    assert out["diagnosis"]["primary_diagnosis"] == "force_balance_overbinding_likely"
    assert out["best_candidate_run"]["charge_scale_li"] == pytest.approx(0.8)
    assert out["best_candidate_run"]["charge_scale_polymer"] == pytest.approx(1.0)
    assert out["gains_vs_baseline"]["sigma_ne_gain"] == pytest.approx(12.0)


def test_load_benchmark_analysis_dir_reads_compare_payload(tmp_path: Path):
    analysis_dir = tmp_path / "06_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "benchmark_compare.json").write_text(
        json.dumps(
            {
                "metadata": {"prod_ns": 10.0, "charge_scale": {"li": 0.8, "tfsi": 0.8}},
                "compare": {"charge_scale_polymer": 1.0, "charge_scale_li": 0.8, "charge_scale_anion": 0.8, "sigma_ne_upper_bound_S_m": 1.2e-2},
            }
        ),
        encoding="utf-8",
    )
    (analysis_dir / "coordination_partition.json").write_text(json.dumps({"coordination_bias": "mixed"}), encoding="utf-8")
    (analysis_dir / "transport_summary.json").write_text(json.dumps({"li_alpha_mean": 0.8}), encoding="utf-8")
    (analysis_dir / "force_balance_report.json").write_text(json.dumps({"diagnosis": {"primary_force_balance_flag": "anion_proxy_stronger"}}), encoding="utf-8")

    out = load_benchmark_analysis_dir(analysis_dir)
    assert out["metadata"]["prod_ns"] == pytest.approx(10.0)
    assert out["benchmark_compare"]["charge_scale_li"] == pytest.approx(0.8)
    assert out["benchmark_compare"]["charge_scale_polymer"] == pytest.approx(1.0)
    assert out["coordination_partition"]["coordination_bias"] == "mixed"
