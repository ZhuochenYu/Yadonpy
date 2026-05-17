from __future__ import annotations

from pathlib import Path

from yadonpy.sim.preset.eq import XYSlabEquilibrationSpec, _estimate_total_mass_amu_from_top, _xy_slab_z_schedule, xy_slab_mdp_overrides


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
