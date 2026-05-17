from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import yadonpy.gmx.analysis.interface_profile as interface_profile
from yadonpy.gmx.analysis.interface_profile import compute_interface_profile
from yadonpy.interface import analyze_layer_stack_interface
from yadonpy.sim.analyzer import AnalyzeResult


def _write_itp(system_dir: Path, moltype: str, atoms: list[tuple[str, str, float, float]]) -> None:
    mol_dir = system_dir / "molecules" / moltype
    mol_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "[ moleculetype ]",
        f"{moltype} 3",
        "",
        "[ atoms ]",
    ]
    for idx, (atype, aname, charge, mass) in enumerate(atoms, start=1):
        lines.append(f"{idx:5d} {atype:<6} 1 {moltype:<6} {aname:<6} {idx:5d} {charge: .6f} {mass:.4f}")
    lines.append("")
    lines.append("[ bonds ]")
    for idx in range(1, len(atoms)):
        lines.append(f"{idx} {idx + 1}")
    (mol_dir / f"{moltype}.itp").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gro_line(resid: int, resname: str, atomname: str, atomnr: int, xyz: tuple[float, float, float]) -> str:
    return f"{resid:5d}{resname:<5}{atomname:>5}{atomnr:5d}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"


def _write_synthetic_stack(system_dir: Path, *, direct_contact: bool = False) -> None:
    system_dir.mkdir(parents=True, exist_ok=True)
    _write_itp(system_dir, "GRAPH", [("cgraph", "C1", 0.0, 12.011), ("cgraph", "C2", 0.0, 12.011)])
    _write_itp(system_dir, "POLY", [("c3", "C1", 0.1, 12.011), ("os", "O2", -0.2, 15.999), ("c3", "C3", 0.1, 12.011)])
    _write_itp(system_dir, "Li", [("Li", "Li1", 1.0, 6.94)])
    _write_itp(system_dir, "SOLV", [("c", "C1", 0.3, 12.011), ("o", "O2", -0.3, 15.999)])
    _write_itp(system_dir, "PF6", [("p", "P1", 1.0, 30.974), ("f", "F2", -1.0, 18.998)])
    (system_dir / "system.top").write_text(
        '#include "molecules/GRAPH/GRAPH.itp"\n'
        '#include "molecules/POLY/POLY.itp"\n'
        '#include "molecules/Li/Li.itp"\n'
        '#include "molecules/SOLV/SOLV.itp"\n'
        '#include "molecules/PF6/PF6.itp"\n\n'
        "[ molecules ]\n"
        "GRAPH 1\n"
        "POLY 2\n"
        "Li 1\n"
        "SOLV 1\n"
        "PF6 1\n",
        encoding="utf-8",
    )
    (system_dir / "system_meta.json").write_text(
        json.dumps(
            {
                "species": [
                    {"moltype": "GRAPH", "kind": "substrate", "smiles": "graphite", "n": 1, "formal_charge": 0.0},
                    {"moltype": "POLY", "kind": "polymer", "smiles": "*CCO*", "n": 2, "formal_charge": 0.0},
                    {"moltype": "Li", "kind": "ion", "smiles": "[Li+]", "n": 1, "formal_charge": 1.0},
                    {"moltype": "SOLV", "kind": "solvent", "smiles": "CO", "n": 1, "formal_charge": 0.0},
                    {"moltype": "PF6", "kind": "ion", "smiles": "F[P-](F)(F)(F)(F)F", "n": 1, "formal_charge": -1.0},
                ],
                "box_lengths_nm": [3.0, 3.0, 5.0],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    electrolyte_z = 0.42 if direct_contact else 2.66
    atoms = [
        ("GRAPH", "C1", (1.00, 1.00, 0.20)),
        ("GRAPH", "C2", (1.40, 1.00, 0.30)),
        ("POLY", "C1", (1.00, 1.00, 1.10)),
        ("POLY", "O2", (1.00, 1.00, 1.50)),
        ("POLY", "C3", (1.00, 1.00, 1.90)),
        ("POLY", "C1", (1.00, 1.00, 2.30)),
        ("POLY", "O2", (1.00, 1.00, 2.64)),
        ("POLY", "C3", (1.00, 1.00, 2.80)),
        ("Li", "Li1", (1.00, 1.00, electrolyte_z)),
        ("SOLV", "C1", (1.20, 1.00, max(electrolyte_z, 2.70))),
        ("SOLV", "O2", (1.02, 1.00, max(electrolyte_z + 0.02, 2.68))),
        ("PF6", "P1", (1.00, 1.30, max(electrolyte_z, 2.74))),
        ("PF6", "F2", (1.00, 1.16, max(electrolyte_z, 2.66))),
    ]
    gro_lines = ["synthetic interface stack", f"{len(atoms):5d}"]
    for atomnr, (resname, atomname, xyz) in enumerate(atoms, start=1):
        gro_lines.append(_gro_line(atomnr, resname, atomname, atomnr, xyz))
    gro_lines.append("   3.00000   3.00000   5.00000")
    (system_dir / "system.gro").write_text("\n".join(gro_lines) + "\n", encoding="utf-8")
    (system_dir / "system.ndx").write_text(
        "[ GRAPHITE ]\n1 2\n\n"
        "[ POLYMER ]\n3 4 5 6 7 8\n\n"
        "[ ELECTROLYTE ]\n9 10 11 12 13\n",
        encoding="utf-8",
    )


def test_interface_profile_extracts_phase_order_density_overlap_and_coordination(tmp_path: Path):
    system_dir = tmp_path / "00_stack"
    _write_synthetic_stack(system_dir)
    manifest_path = tmp_path / "layer_stack_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "synthetic_manifest",
                "pbc_mode": "xyz",
                "box_nm": [3.0, 3.0, 5.0],
                "layers": [
                    {"name": "GRAPHITE", "kind": "graphite"},
                    {"name": "POLYMER", "kind": "polymer"},
                    {"name": "ELECTROLYTE", "kind": "electrolyte"},
                ],
                "layer_intervals_nm": [
                    {"name": "GRAPHITE", "z_lo_nm": 0.0, "z_hi_nm": 0.5},
                    {"name": "POLYMER", "z_lo_nm": 0.8, "z_hi_nm": 2.8},
                    {"name": "ELECTROLYTE", "z_lo_nm": 2.6, "z_hi_nm": 3.0},
                ],
                "fixed_charge_regions": [
                    {
                        "label": "graphite_inner_face",
                        "layer_name": "GRAPHITE",
                        "region": "top",
                        "target_charge_e": 0.25,
                        "selected_atom_count": 2,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = compute_interface_profile(
        gro_path=system_dir / "system.gro",
        top_path=system_dir / "system.top",
        ndx_path=system_dir / "system.ndx",
        system_dir=system_dir,
        out_dir=tmp_path / "analysis",
        xtc_path=None,
        bin_nm=0.10,
        compute_transport=False,
        manifest_path=manifest_path,
    )

    assert out["geometry_health"]["phase_order"] == ["GRAPHITE", "POLYMER", "ELECTROLYTE"]
    assert out["geometry_health"]["phase_order_ok"] is True
    assert out["geometry_health"]["direct_graphite_electrolyte_contact"] is False
    assert out["region_summary"]["interpenetration"]["overlap_width_nm"] > 0.0
    assert out["membrane_permeation"]["available"] is True
    assert out["membrane_permeation"]["membrane_thickness_nm"] > 0.0
    assert "SOLV" in out["membrane_permeation"]["summary_by_species"]
    assert out["coordination_by_region"]["available"] is True
    assert out["parameters"]["time_series_analysis"] is False
    assert out["manifest"]["available"] is True
    assert out["manifest"]["fixed_charge_regions"][0]["label"] == "graphite_inner_face"
    assert out["time_series"]["available"] is False
    assert out["time_series"]["reason"] == "disabled"
    assert (tmp_path / "analysis" / "z_density_profiles.csv").exists()
    assert (tmp_path / "analysis" / "region_summary.json").exists()
    assert (tmp_path / "analysis" / "membrane_permeation_summary.json").exists()
    assert (tmp_path / "analysis" / "membrane_permeation_timeseries.csv").exists()
    assert (tmp_path / "analysis" / "coordination_by_region.json").exists()


def test_interface_profile_detects_graphite_electrolyte_direct_contact(tmp_path: Path):
    system_dir = tmp_path / "00_stack"
    _write_synthetic_stack(system_dir, direct_contact=True)

    out = compute_interface_profile(
        gro_path=system_dir / "system.gro",
        top_path=system_dir / "system.top",
        ndx_path=system_dir / "system.ndx",
        system_dir=system_dir,
        out_dir=tmp_path / "analysis",
        xtc_path=None,
        bin_nm=0.10,
        compute_transport=False,
    )

    assert out["geometry_health"]["direct_graphite_electrolyte_contact"] is True


def test_analyze_layer_stack_interface_wrapper_accepts_static_stack(tmp_path: Path):
    system_dir = tmp_path / "case" / "02_system"
    _write_synthetic_stack(system_dir)

    out = analyze_layer_stack_interface(
        work_dir=tmp_path / "case",
        system_gro=system_dir / "system.gro",
        system_ndx=system_dir / "system.ndx",
        phase_groups=("GRAPHITE", "POLYMER", "ELECTROLYTE"),
        out_dir=tmp_path / "case" / "06_analysis" / "layer_stack_interface",
        compute_transport=False,
        bin_nm=0.10,
    )

    assert out["geometry_health"]["phase_order_ok"] is True
    assert (tmp_path / "case" / "06_analysis" / "layer_stack_interface" / "interface_profile_summary.json").exists()


def test_interface_facade_exposes_stepwise_outputs(tmp_path: Path):
    system_dir = tmp_path / "case" / "02_system"
    _write_synthetic_stack(system_dir)

    analy = AnalyzeResult(
        work_dir=tmp_path / "case",
        tpr=tmp_path / "case" / "missing.tpr",
        xtc=tmp_path / "case" / "missing.xtc",
        edr=tmp_path / "case" / "missing.edr",
        top=system_dir / "system.top",
        ndx=system_dir / "system.ndx",
    )
    interface = analy.interface(
        analysis_profile="interface_fast",
        bin_nm=0.10,
        penetration_threshold_nm=0.33,
        phase_groups=("GRAPHITE", "POLYMER", "ELECTROLYTE"),
        compute_transport=False,
    )

    health = interface.geometry_health()
    z_profiles = interface.z_profiles()
    edl = interface.edl_profiles(report_potential_drop=True)
    penetration = interface.penetration(species=("SOLV", "PF6"))
    membrane = interface.membrane_permeation(species=("SOLV", "PF6"))
    adsorption = interface.graphite_adsorption(species=("SOLV",))
    coordination = interface.coordination_by_region()
    transport = interface.region_transport()
    time_series_default = interface.time_series()
    summary = interface.summary()
    time_series_enabled = interface.time_series(time_series_analysis=True)
    enabled_summary = interface.summary(time_series_analysis=True)

    assert health["phase_order_ok"] is True
    assert "z_density_profiles_csv" in z_profiles["outputs"]
    assert edl["available"] is True
    assert "charge_potential" in edl
    assert penetration["available"] is True
    assert penetration["penetration_threshold_nm"] == 0.33
    assert membrane["available"] is True
    assert "summary_by_species" in membrane
    assert adsorption["available"] is True
    assert coordination["available"] is True
    assert transport["available"] is False
    assert time_series_default["available"] is False
    assert time_series_default["reason"] == "disabled"
    assert summary["parameters"]["time_series_analysis"] is False
    assert time_series_enabled["available"] is False
    assert time_series_enabled["reason"] == "too_few_time_windows"
    assert enabled_summary["parameters"]["time_series_analysis"] is True
    outputs = summary["outputs"]
    for key in (
        "geometry_health_json",
        "charge_density_profiles_csv",
        "edl_summary_json",
        "integrated_charge_csv",
        "electrostatic_potential_csv",
        "penetration_summary_json",
        "membrane_permeation_summary_json",
        "membrane_permeation_timeseries_csv",
        "adsorption_summary_json",
        "coordination_z_profile_csv",
    ):
        assert Path(outputs[key]).exists()


def test_interface_time_series_writes_decile_csv_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(interface_profile, "_mp4_writer", lambda fps: None)
    coords = np.asarray(
        [
            [1.00, 1.00, 1.00],  # cation center
            [1.18, 1.00, 1.00],  # polymer O
            [1.42, 1.00, 1.00],  # solvent O
            [1.78, 1.00, 1.00],  # anion F
        ],
        dtype=float,
    )
    frames = [
        (float(t), coords + np.asarray([0.0, 0.0, 0.02 * i]), (3.0, 3.0, 5.0))
        for i, t in enumerate((0.0, 10.0, 20.0, 30.0))
    ]
    instances = [
        {"moltype": "Li", "kind": "ion", "atom_indices_0": np.asarray([0]), "masses": np.asarray([6.94])},
        {
            "moltype": "SOLV",
            "kind": "solvent",
            "atom_indices_0": np.asarray([1, 2]),
            "masses": np.asarray([15.999, 15.999]),
        },
    ]
    categories = {
        "cation": np.asarray([0], dtype=int),
        "polymer_o": np.asarray([1], dtype=int),
        "solvent_o": np.asarray([2], dtype=int),
        "anion_f": np.asarray([3], dtype=int),
    }
    adsorption_rows = [
        {
            "time_ps": float(t),
            "adsorbed": True,
            "orientation_available": True,
            "carbonyl_angle_deg": 30.0 + i,
            "dipole_proxy_angle_deg": 75.0 + i,
        }
        for i, t in enumerate((0.0, 10.0, 20.0, 30.0))
    ]

    out = interface_profile._time_series_animations(
        out_dir=tmp_path,
        frames=frames,
        bins=np.arange(0.0, 5.5, 0.5),
        instances=instances,
        categories=categories,
        adsorption_rows=adsorption_rows,
        sample_count=2,
        fps=1.0,
        rdf_rmax_nm=1.0,
        rdf_bin_nm=0.1,
    )

    assert out["sample_windows"] == 2
    assert (tmp_path / "time_series" / "z_concentration_timeseries.csv").exists()
    assert (tmp_path / "time_series" / "rdf_cn_curves_timeseries.csv").exists()
    assert (tmp_path / "time_series" / "rdf_cn_shell_timeseries.csv").exists()
    assert (tmp_path / "time_series" / "adsorbed_orientation_angle_timeseries.csv").exists()
    assert out["outputs"]["rdf_cn"]["shell_csv"].endswith("rdf_cn_shell_timeseries.csv")
