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
    _estimate_total_mass_amu_from_top,
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
    assert overrides["compressibility"] == "4.5e-5 0"


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


def test_cmcna_relaxation_spec_maps_to_xy_slab_defaults():
    spec = CMCNAXYSlabRelaxationSpec()
    xy = spec.to_xy_slab_spec()

    assert xy.target_density_g_cm3 == 1.5
    assert xy.max_z_shrink_per_cycle == 0.06
    assert xy.active_density_convergence is True
    assert xy.rg_convergence is True
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
    assert calls["exec_kwargs"]["xy_slab"].target_density_g_cm3 == 1.5
