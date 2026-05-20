from __future__ import annotations

from pathlib import Path

import json

import numpy as np
from rdkit import Chem

from yadonpy.interface.cmcna_slab import CMCNAXYSlabRelaxationSpec, prepare_cmcna_xy_bulk_slab
from yadonpy.sim.preset.eq import (
    XYSlabEquilibrationSpec,
    _active_density_gate,
    _active_density_rows_from_coords,
    _connected_void_report,
    _estimate_total_mass_amu_from_top,
    _export_xy_slab_prepared_gro,
    _surface_flatness_report,
    _xy_slab_geometry_gate,
    _xy_slab_z_schedule,
    xy_slab_mdp_overrides,
)


def test_xy_slab_mdp_overrides_emit_walls_and_3dc(tmp_path: Path):
    top = tmp_path / "system.top"
    top.write_text("[ atomtypes ]\nOW  8 15.999 0 A 0.3 0.5\nHW  1 1.008 0 A 0.1 0.1\n", encoding="utf-8")

    overrides = xy_slab_mdp_overrides(top_path=top, spec=XYSlabEquilibrationSpec(), pressure_bar=1.0)
    rendered_wall = str(overrides["wall_mdp"])

    assert overrides["pbc"] == "xy"
    assert overrides["periodic_molecules"] == "yes"
    assert "nwall                    = 2" in rendered_wall
    assert "ewald-geometry           = 3dc" in rendered_wall
    assert "wall_atomtype            = OW OW" in rendered_wall


def test_xy_slab_auto_schedule_respects_max_shrink():
    spec = XYSlabEquilibrationSpec(target_density_g_cm3=0.5, max_z_shrink_per_cycle=0.10, max_cycles=30)
    schedule = _xy_slab_z_schedule(10.0, 2.0, spec)

    assert schedule
    prev = 10.0
    for z in schedule:
        assert z >= 2.0
        assert (prev - z) / prev <= 0.1000001
        prev = z
    assert schedule[-1] == 2.0


def test_xy_slab_xy_npt_mode_only_couples_xy_area(tmp_path: Path):
    top = tmp_path / "system.top"
    top.write_text("[ atomtypes ]\nC  6 12.011 0 A 0.3 0.5\n", encoding="utf-8")
    spec = XYSlabEquilibrationSpec(xy_area_mode="xy_npt")

    overrides = xy_slab_mdp_overrides(top_path=top, spec=spec, pressure_bar=1.0, npt_like=True)

    assert overrides["pcoupltype"] == "semiisotropic"
    assert overrides["ref_p"] == "1 1"
    assert overrides["compressibility"] == "4.5e-05 0"


def test_xy_slab_fixed_xy_z_npt_mode_only_couples_z(tmp_path: Path):
    top = tmp_path / "system.top"
    top.write_text("[ atomtypes ]\nC  6 12.011 0 A 0.3 0.5\n", encoding="utf-8")
    spec = XYSlabEquilibrationSpec(density_mode="wall_z_npt", pressure_axis_mode="fixed_xy_z_npt")

    overrides = xy_slab_mdp_overrides(top_path=top, spec=spec, pressure_bar=1.0, npt_like=True)

    assert overrides["pbc"] == "xy"
    assert overrides["pcoupltype"] == "semiisotropic"
    assert overrides["ref_p"] == "1 1"
    assert overrides["compressibility"] == "0 4.5e-05"


def test_xy_slab_prepared_export_wraps_xy_but_keeps_z_open(tmp_path: Path):
    src = tmp_path / "whole.gro"
    dst = tmp_path / "prepared_slab.gro"
    rows = [
        f"{1:5d}{'POL':<5}{'C':>5}{1:5d}{-0.200:8.3f}{0.100:8.3f}{-0.400:8.3f}",
        f"{1:5d}{'POL':<5}{'O':>5}{2:5d}{2.200:8.3f}{2.100:8.3f}{1.200:8.3f}",
        f"{1:5d}{'POL':<5}{'H':>5}{3:5d}{1.000:8.3f}{-0.300:8.3f}{3.000:8.3f}",
    ]
    src.write_text("\n".join(["whole slab", f"{len(rows):5d}", *rows, f"{2.00000:10.5f}{2.00000:10.5f}{4.00000:10.5f}"]) + "\n", encoding="utf-8")

    report = _export_xy_slab_prepared_gro(src, dst, policy="wrapped_xy_z_open")
    coords = []
    lines = dst.read_text(encoding="utf-8").splitlines()
    for line in lines[2:5]:
        coords.append((float(line[20:28]), float(line[28:36]), float(line[36:44])))

    assert report["coordinate_export_policy"] == "wrapped_xy_z_open"
    assert report["outside_xy_atom_count_before_wrap"] == 3
    assert report["xy_wrapped_ok"] is True
    assert all(0.0 <= xyz[0] < 2.0 and 0.0 <= xyz[1] < 2.0 for xyz in coords)
    assert [round(xyz[2], 3) for xyz in coords] == [-0.4, 1.2, 3.0]


def test_xy_slab_wall_atomtype_falls_back_to_molecule_atoms(tmp_path: Path):
    top = tmp_path / "system.top"
    mol_dir = tmp_path / "molecules" / "eth"
    mol_dir.mkdir(parents=True)
    top.write_text('#include "ff_parameters.itp"\n#include "molecules/eth/ETH.itp"\n\n[ molecules ]\nETH 1\n', encoding="utf-8")
    (tmp_path / "ff_parameters.itp").write_text(
        "[ atomtypes ]\nc3 6 12.011 0 A 0.3 0.5\nh1 1 1.008 0 A 0.1 0.1\n",
        encoding="utf-8",
    )
    (mol_dir / "ETH.itp").write_text(
        "[ moleculetype ]\nETH 3\n\n[ atoms ]\n1 c3 1 ETH C1 1 0.0 12.011\n2 h1 1 ETH H1 1 0.0 1.008\n",
        encoding="utf-8",
    )

    overrides = xy_slab_mdp_overrides(top_path=top, spec=XYSlabEquilibrationSpec())

    assert "wall_atomtype            = c3 c3" in str(overrides["wall_mdp"])


def test_xy_slab_mass_estimate_reads_nested_molecule_itps(tmp_path: Path):
    top = tmp_path / "system.top"
    mol_dir = tmp_path / "molecules" / "polymer"
    mol_dir.mkdir(parents=True)
    top.write_text('#include "molecules/polymer/CMC.itp"\n\n[ molecules ]\nCMC 3\nNA  6\n', encoding="utf-8")
    (mol_dir / "CMC.itp").write_text(
        "[ moleculetype ]\nCMC 3\n\n[ atoms ]\n1 c 1 CMC C1 1 0.0 12.000\n2 o 1 CMC O1 2 0.0 16.000\n",
        encoding="utf-8",
    )
    (mol_dir / "NA.itp").write_text(
        "[ moleculetype ]\nNA 1\n\n[ atoms ]\n1 na 1 NA NA 1 1.0 23.000\n",
        encoding="utf-8",
    )

    assert _estimate_total_mass_amu_from_top(top) == 3 * 28.0 + 6 * 23.0


def test_xy_slab_active_density_uses_active_extent_not_total_box():
    target_density = 1.5
    area_nm2 = 4.0
    active_z_nm = 2.0
    avogadro = 6.02214076e23
    total_mass_amu = target_density * area_nm2 * active_z_nm * 1.0e-21 * avogadro
    coords = np.asarray(
        [
            [0.1, 0.1, 1.0],
            [0.2, 0.2, 1.0],
            [0.3, 0.3, 3.0],
            [0.4, 0.4, 3.0],
        ],
        dtype=float,
    )

    rows = _active_density_rows_from_coords(
        coords_nm=coords,
        time_ps=[0.0],
        box_nm=np.asarray([2.0, 2.0, 20.0], dtype=float),
        total_mass_amu=total_mass_amu,
        q_low=0.0,
        q_high=1.0,
    )

    assert rows[0]["active_z_extent_nm"] == 2.0
    assert rows[0]["box_z_nm"] == 20.0
    assert rows[0]["active_density_g_cm3"] == target_density


def test_xy_slab_active_density_gate_accepts_stable_tail():
    spec = XYSlabEquilibrationSpec(
        target_density_g_cm3=1.5,
        active_density_tolerance_fraction=0.05,
        active_density_rel_std_max=0.03,
        active_density_tail_fraction=0.5,
    )
    rows = [
        {"time_ps": 0.0, "active_density_g_cm3": 1.40},
        {"time_ps": 1.0, "active_density_g_cm3": 1.49},
        {"time_ps": 2.0, "active_density_g_cm3": 1.51},
        {"time_ps": 3.0, "active_density_g_cm3": 1.50},
    ]

    gate = _active_density_gate(rows, target_density_g_cm3=1.5, spec=spec)

    assert gate["ok"] is True
    assert gate["mean_g_cm3"] == 1.505


def test_xy_slab_active_density_wall_npt_gate_uses_plateau_not_target():
    spec = XYSlabEquilibrationSpec(
        density_mode="wall_z_npt",
        target_density_g_cm3=None,
        active_density_rel_std_max=0.03,
        active_density_tail_fraction=0.5,
    )
    rows = [
        {"time_ps": 0.0, "active_density_g_cm3": 0.80},
        {"time_ps": 1.0, "active_density_g_cm3": 0.91},
        {"time_ps": 2.0, "active_density_g_cm3": 0.90},
        {"time_ps": 3.0, "active_density_g_cm3": 0.92},
    ]

    gate = _active_density_gate(rows, target_density_g_cm3=None, spec=spec)

    assert gate["ok"] is True
    assert gate["mode"] == "plateau_only"
    assert gate["target_density_g_cm3"] is None


def test_xy_slab_wall_gap_gate_can_use_density_floor_without_exact_target():
    spec = XYSlabEquilibrationSpec(
        density_mode="wall_gap_compression",
        target_density_g_cm3=0.8,
        active_density_min_g_cm3=0.8,
        active_density_rel_std_max=0.05,
        active_density_tail_fraction=0.5,
    )
    rows = [
        {"time_ps": 0.0, "active_density_g_cm3": 0.40},
        {"time_ps": 1.0, "active_density_g_cm3": 0.81},
        {"time_ps": 2.0, "active_density_g_cm3": 0.82},
        {"time_ps": 3.0, "active_density_g_cm3": 0.83},
    ]

    gate = _active_density_gate(rows, target_density_g_cm3=0.8, spec=spec)

    assert gate["ok"] is True
    assert gate["mode"] == "plateau_only"
    assert gate["target_density_g_cm3"] == 0.8
    assert gate["min_density_g_cm3"] == 0.8


def test_xy_slab_geometry_gate_checks_lateral_occupancy_when_requested(tmp_path: Path):
    gro = tmp_path / "membrane.gro"
    rows = []
    idx = 1
    for x in (0.1, 0.6, 1.1, 1.6):
        for y in (0.1, 0.6, 1.1, 1.6):
            rows.append(f"{1:5d}{'CMC':<5}{'C':>5}{idx:5d}{x:8.3f}{y:8.3f}{1.000:8.3f}")
            idx += 1
    gro.write_text("\n".join(["wrapped membrane", f"{len(rows):5d}", *rows, f"{2.00000:10.5f}{2.00000:10.5f}{3.00000:10.5f}"]) + "\n", encoding="utf-8")
    spec = XYSlabEquilibrationSpec(
        lateral_occupancy_convergence=True,
        lateral_occupancy_grid_nm=0.5,
        min_lateral_occupancy_fraction=0.85,
        min_edge_occupancy_fraction=0.80,
    )

    gate = _xy_slab_geometry_gate(gro, spec=spec)

    assert gate["ok"] is True
    assert gate["lateral_occupancy_ok"] is True
    assert gate["z_open_ok"] is True


def test_xy_slab_geometry_gate_can_enforce_surface_flatness_and_voids(tmp_path: Path):
    gro = tmp_path / "flat_dense_membrane.gro"
    rows = []
    idx = 1
    for x in (0.1, 0.6, 1.1, 1.6):
        for y in (0.1, 0.6, 1.1, 1.6):
            rows.append(f"{1:5d}{'CMC':<5}{'C':>5}{idx:5d}{x:8.3f}{y:8.3f}{0.900:8.3f}")
            idx += 1
            rows.append(f"{1:5d}{'CMC':<5}{'O':>5}{idx:5d}{x:8.3f}{y:8.3f}{1.100:8.3f}")
            idx += 1
    gro.write_text("\n".join(["flat membrane", f"{len(rows):5d}", *rows, f"{2.00000:10.5f}{2.00000:10.5f}{2.00000:10.5f}"]) + "\n", encoding="utf-8")
    spec = XYSlabEquilibrationSpec(
        lateral_occupancy_convergence=True,
        surface_flatness_convergence=True,
        connected_void_convergence=True,
        lateral_occupancy_grid_nm=0.5,
        surface_flatness_grid_nm=0.5,
        void_grid_nm=0.5,
        void_atom_radius_nm=0.5,
        max_surface_rms_nm=0.05,
        max_surface_peak_to_peak_nm=0.10,
        max_connected_void_fraction=0.05,
    )

    gate = _xy_slab_geometry_gate(gro, spec=spec)

    assert gate["ok"] is True
    assert gate["surface_flatness_ok"] is True
    assert gate["connected_void_ok"] is True


def test_xy_slab_geometry_gate_rejects_rough_surface_when_enforced(tmp_path: Path):
    gro = tmp_path / "rough_membrane.gro"
    rows = []
    idx = 1
    for n, x in enumerate((0.1, 0.6, 1.1, 1.6)):
        for y in (0.1, 0.6, 1.1, 1.6):
            z = 0.5 if n % 2 == 0 else 1.3
            rows.append(f"{1:5d}{'CMC':<5}{'C':>5}{idx:5d}{x:8.3f}{y:8.3f}{z:8.3f}")
            idx += 1
    gro.write_text("\n".join(["rough membrane", f"{len(rows):5d}", *rows, f"{2.00000:10.5f}{2.00000:10.5f}{2.00000:10.5f}"]) + "\n", encoding="utf-8")
    spec = XYSlabEquilibrationSpec(
        lateral_occupancy_convergence=True,
        surface_flatness_convergence=True,
        connected_void_convergence=False,
        lateral_occupancy_grid_nm=0.5,
        surface_flatness_grid_nm=0.5,
        max_surface_rms_nm=0.05,
        max_surface_peak_to_peak_nm=0.10,
    )

    gate = _xy_slab_geometry_gate(gro, spec=spec)

    assert gate["ok"] is False
    assert gate["surface_flatness_ok"] is False


def test_connected_void_report_detects_through_channel():
    coords = []
    for x in (0.25, 0.75, 1.25, 1.75):
        for y in (0.25, 0.75, 1.25, 1.75):
            if x < 0.6 and y < 0.6:
                continue
            coords.append([x, y, 0.5])
            coords.append([x, y, 1.5])
    report = _connected_void_report(np.asarray(coords, dtype=float), (2.0, 2.0, 2.0), grid_nm=0.5, atom_radius_nm=0.0)

    assert report["available"] is True
    assert report["through_void"] is True


def test_surface_flatness_report_measures_surface_rms():
    coords = np.asarray([[0.1, 0.1, 0.8], [0.6, 0.1, 0.8], [0.1, 0.6, 1.2], [0.6, 0.6, 1.2]], dtype=float)
    report = _surface_flatness_report(coords, (1.0, 1.0, 2.0), grid_nm=0.5)

    assert report["available"] is True
    assert report["max_surface_rms_nm"] > 0.0


def test_cmcna_relaxation_spec_maps_to_xy_slab_defaults():
    spec = CMCNAXYSlabRelaxationSpec()
    xy = spec.to_xy_slab_spec()

    assert xy.density_mode == "wall_gap_compression"
    assert xy.target_density_g_cm3 == 0.8
    assert xy.active_density_min_g_cm3 == 0.8
    assert xy.pressure_axis_mode == "fixed_xy_z_npt"
    assert xy.active_density_convergence is True
    assert xy.rg_convergence is True
    assert xy.lateral_occupancy_convergence is True
    assert xy.surface_flatness_convergence is True
    assert xy.connected_void_convergence is True
    assert xy.max_convergence_rounds == 8


def test_prepare_cmcna_xy_bulk_slab_uses_fixed_xy_and_writes_result(monkeypatch, tmp_path: Path):
    import yadonpy.interface.cmcna_slab as mod

    calls = {}

    def fake_cell(lengths):
        calls["cell_lengths"] = tuple(float(v) for v in lengths)
        return {"cell": calls["cell_lengths"]}

    def fake_amorphous(species, counts, **kwargs):
        calls["counts"] = tuple(int(v) for v in counts)
        calls["cell"] = kwargs["cell"]
        return object()

    class FakeEQ21:
        def __init__(self, ac, work_dir):
            self.work_dir = Path(work_dir)

        def exec(self, **kwargs):
            calls["exec_kwargs"] = kwargs
            run_dir = self.work_dir / "03_EQ21_XY_SLAB"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "prepared_slab.gro").write_text("gro\n", encoding="utf-8")
            (run_dir / "prepared_slab.top").write_text("top\n", encoding="utf-8")
            (run_dir / "xy_slab_summary.json").write_text("{}\n", encoding="utf-8")
            (run_dir / "cmcna_slab_convergence.json").write_text(
                json.dumps({"ready_for_layer_stack": True}) + "\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(mod, "make_orthorhombic_pack_cell", fake_cell)
    monkeypatch.setattr(mod.poly, "amorphous_cell", fake_amorphous)
    monkeypatch.setattr(mod.eq, "EQ21step", FakeEQ21)

    cmc = Chem.MolFromSmiles("CC")
    na = Chem.MolFromSmiles("[Na+]")
    out = prepare_cmcna_xy_bulk_slab(
        cmc_chain_mol=cmc,
        na_mol=na,
        chain_count=2,
        dp=3,
        xy_nm=(5.0, 7.0),
        work_dir=tmp_path / "cmc",
        restart=True,
    )

    assert out.ready_for_layer_stack is True
    assert out.xy_nm == (5.0, 7.0)
    assert calls["counts"] == (2, 6)
    assert calls["cell_lengths"][0:2] == (5.0, 7.0)
    assert calls["exec_kwargs"]["periodicity"] == "xy"
    assert calls["exec_kwargs"]["xy_slab"].density_mode == "wall_gap_compression"
    assert calls["exec_kwargs"]["xy_slab"].target_density_g_cm3 == 0.8
