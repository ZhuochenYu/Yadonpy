from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_report_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "08_graphite_polymer_electrolyte_sandwich" / "make_charge_sweep_report_ppt.py"
    spec = importlib.util.spec_from_file_location("eg08_charge_sweep_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_case(tmp_path: Path, *, layer: str, region: str, z_window: tuple[float, float], charge: float, lz: float = 10.0) -> Path:
    case_dir = tmp_path / "case"
    system_dir = case_dir / "02_system"
    system_dir.mkdir(parents=True)
    (system_dir / "system.gro").write_text(
        "test\n0\n   5.00000   5.00000  %8.5f\n" % lz,
        encoding="utf-8",
    )
    (system_dir / "charge_patch_report.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "label": "cmc_facing_graphite_inner_face",
                        "layer_name": layer,
                        "region": region,
                        "z_window_nm": list(z_window),
                        "surface_charge_uC_cm2": charge,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return case_dir


def test_cmc_facing_top_bottom_uses_decreasing_periodic_axis(tmp_path: Path) -> None:
    mod = _load_report_module()
    case_dir = _write_case(tmp_path, layer="GRAPHITE_TOP", region="bottom", z_window=(8.0, 8.2), charge=-9.0)
    zref = mod._z_axis_from_patch(case_dir, {})
    payload = mod.CasePayload(
        label="-9 uC/cm2",
        dirname="case",
        charge=-9.0,
        analysis_dir=case_dir,
        relax_dir=case_dir,
        summary={},
        z0_nm=zref.z0_nm,
        box_z_nm=zref.box_z_nm,
        z_axis_direction=zref.direction,
        z_axis_source=zref.source,
        z_axis_surface_charge_uC_cm2=zref.surface_charge_uC_cm2,
    )
    assert zref.direction == "decreasing"
    assert zref.z0_nm == 8.0
    assert mod._z_plot_nm(payload, 7.8) == pytest.approx(0.2)
    assert mod._z_plot_nm(payload, 8.2) == pytest.approx(9.8)


def test_cmc_facing_bottom_top_uses_increasing_periodic_axis(tmp_path: Path) -> None:
    mod = _load_report_module()
    case_dir = _write_case(tmp_path, layer="GRAPHITE_BOTTOM", region="top", z_window=(1.0, 1.2), charge=-3.0)
    zref = mod._z_axis_from_patch(case_dir, {})
    payload = mod.CasePayload(
        label="-3 uC/cm2",
        dirname="case",
        charge=-3.0,
        analysis_dir=case_dir,
        relax_dir=case_dir,
        summary={},
        z0_nm=zref.z0_nm,
        box_z_nm=zref.box_z_nm,
        z_axis_direction=zref.direction,
        z_axis_source=zref.source,
        z_axis_surface_charge_uC_cm2=zref.surface_charge_uC_cm2,
    )
    assert zref.direction == "increasing"
    assert zref.z0_nm == 1.2
    assert mod._z_plot_nm(payload, 1.4) == pytest.approx(0.2)
    assert mod._z_plot_nm(payload, 1.0) == pytest.approx(9.8)


def test_edl_recompute_outputs_one_nonnegative_period(tmp_path: Path) -> None:
    mod = _load_report_module()
    case_dir = _write_case(tmp_path, layer="GRAPHITE_TOP", region="bottom", z_window=(8.0, 8.2), charge=-18.0)
    zref = mod._z_axis_from_patch(case_dir, {})
    payload = mod.CasePayload(
        label="-18 uC/cm2",
        dirname="case",
        charge=-18.0,
        analysis_dir=case_dir,
        relax_dir=case_dir,
        summary={},
        z0_nm=zref.z0_nm,
        box_z_nm=zref.box_z_nm,
        z_axis_direction=zref.direction,
        z_axis_source=zref.source,
        z_axis_surface_charge_uC_cm2=zref.surface_charge_uC_cm2,
    )
    rows = [
        {"z_nm": "7.8", "charge_density_e_nm3": "-0.5"},
        {"z_nm": "8.0", "charge_density_e_nm3": "-1.0"},
        {"z_nm": "8.2", "charge_density_e_nm3": "0.2"},
    ]
    out = mod._recompute_edl_on_plot_axis(rows, payload)
    z = [float(row["z_plot_nm"]) for row in out]
    assert min(z) >= 0.0
    assert max(z) <= 10.0
    assert 0.0 in z


def test_adaptive_fraction_ylim_magnifies_small_membrane_uptake() -> None:
    mod = _load_report_module()
    arr_a = mod.np.asarray([[0.5, 0.002], [1.5, 0.006]], dtype=float)
    arr_b = mod.np.asarray([[0.5, 0.004], [1.5, 0.008]], dtype=float)
    lo, hi = mod._adaptive_fraction_ylim([arr_a, arr_b])
    assert lo == pytest.approx(0.0, abs=0.005)
    assert hi <= 0.06
    assert hi >= 0.05


def test_penetration_schematic_is_generated(tmp_path: Path) -> None:
    mod = _load_report_module()
    out = mod._draw_penetration_schematic(tmp_path)
    assert out["png"].is_file()
    assert out["svg"].is_file()


def test_orientation_low_sample_fallback_writes_diagnostic(tmp_path: Path) -> None:
    mod = _load_report_module()
    case_dir = _write_case(tmp_path, layer="GRAPHITE_TOP", region="bottom", z_window=(8.0, 8.2), charge=-9.0)
    analysis = case_dir / "analysis"
    analysis.mkdir(parents=True)
    zref = mod._z_axis_from_patch(case_dir, {})
    payload = mod.CasePayload(
        label="-9 uC/cm2",
        dirname="case",
        charge=-9.0,
        analysis_dir=analysis,
        relax_dir=case_dir,
        summary={},
        z0_nm=zref.z0_nm,
        box_z_nm=zref.box_z_nm,
        z_axis_direction=zref.direction,
        z_axis_source=zref.source,
        z_axis_surface_charge_uC_cm2=zref.surface_charge_uC_cm2,
    )
    summary, samples, plots = mod._orientation_rows([payload], tmp_path)
    assert samples == []
    assert plots == {}
    assert any(row["species"] == "EC" and int(row["sample_count"]) == 0 for row in summary)
    assert (tmp_path / "carbonyl_orientation_summary.csv").is_file()
