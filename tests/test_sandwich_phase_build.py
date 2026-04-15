from __future__ import annotations

import json

import pytest

from yadonpy.interface.sandwich_phase_build import (
    BulkCalibrationSummary,
    recommend_initial_walled_pack_density,
    solve_phase_target_z_nm,
    write_bulk_calibration_summary,
)


def test_solve_phase_target_z_nm_matches_density_relation():
    z_nm = solve_phase_target_z_nm(
        total_mass_amu=1200.0,
        target_density_g_cm3=1.50,
        target_xy_nm=(2.0, 3.0),
    )
    assert z_nm == pytest.approx(0.221405, rel=1e-5)


def test_recommend_initial_walled_pack_density_prefers_polymer_and_electrolyte_safe_ranges():
    polymer_density = recommend_initial_walled_pack_density(
        phase="polymer",
        target_density_g_cm3=1.50,
        selected_bulk_pack_density_g_cm3=0.48,
    )
    electrolyte_density = recommend_initial_walled_pack_density(
        phase="electrolyte",
        target_density_g_cm3=1.30,
        selected_bulk_pack_density_g_cm3=0.86,
    )
    assert polymer_density == pytest.approx(0.36)
    assert electrolyte_density == pytest.approx(0.731)


def test_write_bulk_calibration_summary_round_trips_json(tmp_path):
    summary = BulkCalibrationSummary(
        label="polymer",
        phase_preparation_mode="bulk_calibrate_walled_phase",
        master_xy_nm=(3.0, 4.0),
        bulk_reference_box_nm=(3.0, 4.0, 5.0),
        target_density_g_cm3=1.5,
        total_mass_amu=1000.0,
        target_z_nm=1.2,
        initial_walled_pack_density_g_cm3=0.45,
        selected_bulk_pack_density_g_cm3=0.50,
        charged_phase=True,
        notes=("ok",),
    )
    out = write_bulk_calibration_summary(summary, tmp_path / "summary.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["label"] == "polymer"
    assert payload["phase_preparation_mode"] == "bulk_calibrate_walled_phase"
    assert payload["master_xy_nm"] == [3.0, 4.0]
