from __future__ import annotations

from yadonpy.sim.preset.eq import (
    EQ21ProtocolConfig,
    _build_polymer_chain_relaxation_stages,
    _build_polymer_density_recovery_compaction_stages,
    _build_polymer_density_recovery_release_stages,
    _build_eq21_records,
    _build_liquid_anneal_stages,
    _build_liquid_recovery_compaction_stages,
    _build_liquid_recovery_release_stages,
    _build_liquid_target_relaxation_stages,
    _build_production_stages,
    _compression_state_from_trends,
    _eq21_params_payload,
    _find_latest_equilibrated_gro,
    _next_additional_round,
    _resolve_production_bridge_ps,
    cell_meta_contains_polymer,
    select_relaxation_strategy,
)
from yadonpy.gmx.mdp_templates import NPT_MDP, default_mdp_params


def _find(records, label):
    for rec in records:
        if rec['stage_label'] == label:
            return rec
    raise AssertionError(f'missing record: {label}')


class _Cell:
    def __init__(self, payload):
        import json

        self.payload = json.dumps(payload)

    def HasProp(self, name):
        return name == "_yadonpy_cell_meta"

    def GetProp(self, name):
        assert name == "_yadonpy_cell_meta"
        return self.payload


def test_eq21_robust_schedule_has_expected_shape_and_safety_controls():
    cfg = EQ21ProtocolConfig()
    records = _build_eq21_records(300.0, 1.0, final_ns=0.8, cfg=cfg)
    params = _eq21_params_payload(300.0, 1.0, final_ns=0.8, cfg=cfg)

    assert len(records) == 23
    assert records[0]['folder'] == '01_em'
    assert records[1]['folder'] == '02_preNVT'
    assert records[1]['velocity_reseed'] is True
    assert sum(1 for rec in records if rec['velocity_reseed']) == 1
    assert params['robust'] is True

    high_pressure = _find(records, 'step_09')
    assert high_pressure['pressure_bar'] == 50000.0
    assert high_pressure['pcoupl'] == 'Berendsen'
    assert high_pressure['dt_ps'] == 0.0005
    assert high_pressure['tau_p_ps'] >= 12.0
    assert high_pressure['compressibility_bar_inv'] < cfg.compressibility_bar_inv
    assert high_pressure['time_ps'] == 100.0

    final_stage = _find(records, 'step_21')
    assert final_stage['pcoupl'] == 'C-rescale'
    assert final_stage['pressure_bar'] == 1.0
    assert final_stage['tau_p_ps'] == 1.0
    assert final_stage['velocity_reseed'] is False
    assert final_stage['safety_mode'] == 'final-production'
    assert final_stage['time_ps'] == 1600.0
    assert params['npt_time_scale'] == 2.0


def test_eq21_legacy_mode_preserves_requested_barostat_and_stage_reseed():
    cfg = EQ21ProtocolConfig(robust=False, reseed_each_stage=True, barostat='C-rescale', dt_ps=0.001)
    records = _build_eq21_records(300.0, 1.0, final_ns=0.8, cfg=cfg)

    assert records[1]['dt_ps'] == 0.001
    assert all(rec['velocity_reseed'] for rec in records[2:])

    stage_03 = _find(records, 'step_03')
    assert stage_03['pcoupl'] == 'C-rescale'
    assert stage_03['tau_p_ps'] == 1.0
    assert stage_03['compressibility_bar_inv'] == cfg.compressibility_bar_inv


def test_eq21_npt_time_scale_can_be_overridden():
    cfg = EQ21ProtocolConfig(npt_time_scale=1.5)
    records = _build_eq21_records(300.0, 1.0, final_ns=0.8, cfg=cfg)

    stage_03 = _find(records, 'step_03')
    final_stage = _find(records, 'step_21')

    assert stage_03['time_ps'] == 75.0
    assert final_stage['time_ps'] == 1200.0


def test_cell_meta_contains_polymer_distinguishes_liquids():
    liquid = _Cell({"species": [{"kind": "solvent", "smiles": "CCOC(=O)OCC", "natoms": 18}]})
    polymer = _Cell({"species": [{"kind": "polymer", "smiles": "*CCO*", "natoms": 80}]})

    assert cell_meta_contains_polymer(liquid) is False
    assert cell_meta_contains_polymer(polymer) is True


def test_electrolyte_liquid_gets_short_production_bridge_by_default():
    electrolyte = _Cell(
        {
            "species": [
                {"kind": "solvent", "smiles": "CCOC(=O)OCC", "natoms": 18},
                {"kind": "ion", "label": "Li", "smiles": "[Li+]", "natoms": 1},
            ]
        }
    )
    solvent = _Cell({"species": [{"kind": "solvent", "smiles": "CCOC(=O)OCC", "natoms": 18}]})
    polymer = _Cell({"species": [{"kind": "polymer", "smiles": "*CCO*", "natoms": 80}]})

    assert _resolve_production_bridge_ps(electrolyte, None) == 20.0
    assert _resolve_production_bridge_ps(solvent, None) == 0.0
    assert _resolve_production_bridge_ps(polymer, None) == 100.0
    assert _resolve_production_bridge_ps(electrolyte, 0.0) == 0.0


def test_liquid_anneal_schedule_uses_hbond_2fs_final_npt():
    stages = _build_liquid_anneal_stages(
        temp=318.0,
        press=1.0,
        final_ns=0.8,
        hot_temp=600.0,
        hot_pressure_bar=1000.0,
        compact_pressure_bar=5000.0,
        hot_nvt_ns=0.05,
        compact_npt_ns=0.15,
        hot_npt_ns=0.20,
        cooling_npt_ns=0.10,
        dt_ps=0.002,
        hot_dt_ps=0.001,
        constraints="h-bonds",
        lincs_iter=None,
        lincs_order=None,
        checkpoint_min=5.0,
    )

    assert [stage.name for stage in stages[:4]] == ["01_em", "02_hot_nvt", "03_compact_npt", "04_hot_npt"]
    assert stages[2].mdp.params["pcoupl"] == "Berendsen"
    assert stages[2].mdp.params["ref_p"] == 5000.0
    assert any("cool" in stage.name for stage in stages)
    final = stages[-1]
    assert final.name.endswith("final_npt")
    assert final.kind == "npt"
    assert final.mdp.params["dt"] == 0.002
    assert final.mdp.params["constraints"] == "h-bonds"
    assert final.mdp.params["nsteps"] == 400000
    assert final.lincs_retry is not None


def test_liquid_anneal_all_bonds_is_explicitly_supported():
    stages = _build_liquid_anneal_stages(
        temp=300.0,
        press=1.0,
        final_ns=0.1,
        hot_temp=600.0,
        hot_pressure_bar=1000.0,
        compact_pressure_bar=5000.0,
        hot_nvt_ns=0.01,
        compact_npt_ns=0.01,
        hot_npt_ns=0.01,
        cooling_npt_ns=0.01,
        dt_ps=0.002,
        hot_dt_ps=0.001,
        constraints="all-bonds",
        lincs_iter=None,
        lincs_order=None,
        checkpoint_min=5.0,
    )

    assert stages[-1].mdp.params["constraints"] == "all-bonds"


def test_liquid_density_recovery_schedule_compacts_then_releases():
    compact = _build_liquid_recovery_compaction_stages(
        hot_temp=600.0,
        compact_pressure_bar=5000.0,
        hot_nvt_ns=0.03,
        compact_npt_ns=0.25,
        hot_dt_ps=0.001,
        constraints="h-bonds",
        lincs_iter=None,
        lincs_order=None,
    )
    release = _build_liquid_recovery_release_stages(
        temp=318.0,
        press=1.0,
        final_ns=1.0,
        hot_temp=600.0,
        hot_pressure_bar=1000.0,
        cooling_npt_ns=0.10,
        dt_ps=0.002,
        hot_dt_ps=0.001,
        constraints="h-bonds",
        lincs_iter=None,
        lincs_order=None,
        checkpoint_min=5.0,
    )

    assert [stage.name for stage in compact] == ["01_minim", "02_hot_nvt", "03_compact_npt"]
    assert compact[-1].mdp.params["pcoupl"] == "Berendsen"
    assert compact[-1].mdp.params["ref_p"] == 5000.0
    assert release[0].name == "04_hot_release_npt"
    assert release[0].mdp.params["ref_p"] == 1000.0
    assert release[-1].name.endswith("final_npt")
    assert release[-1].mdp.params["ref_p"] == 1.0
    assert release[-1].mdp.params["dt"] == 0.002
    assert release[-1].mdp.params["constraints"] == "h-bonds"


def test_eq21_extension_uses_box_volume_trend_when_density_is_flat():
    state = _compression_state_from_trends(
        density_trend={
            "ok": True,
            "slope_per_ps": 0.0,
            "tail_delta_kg_m3": 0.0,
        },
        volume_trend={
            "ok": True,
            "slope_rel_per_ps": -1.0e-4,
            "tail_rel_delta": -0.01,
        },
        density_slope_threshold_per_ps=5.0e-2,
        density_delta_threshold_kg_m3=2.0,
    )

    assert state["still_compressing"] is True
    assert state["density_still_compressing"] is False
    assert state["box_volume_still_compressing"] is True


def test_select_relaxation_strategy_routes_liquid_and_polymer_failures():
    density_fail = {
        "ok": False,
        "density_gate": {"ok": False},
        "relaxation_state": {"density_or_volume_still_compressing": True},
    }
    rg_fail = {
        "ok": False,
        "density_gate": {"ok": True},
        "rg_gate": {"ok": False},
        "relaxation_state": {"density_or_volume_still_compressing": False},
    }
    liquid_plateau_fail = {
        "ok": False,
        "density_gate": {"ok": False},
        "relaxation_state": {"density_or_volume_still_compressing": False},
    }
    converged = {
        "ok": True,
        "density_gate": {"ok": True},
        "rg_gate": {"ok": True},
        "relaxation_state": {"density_or_volume_still_compressing": False},
    }

    assert select_relaxation_strategy(density_fail, has_polymer=False) == "liquid_density_recovery"
    assert select_relaxation_strategy(liquid_plateau_fail, has_polymer=False) == "additional_npt"
    assert select_relaxation_strategy(density_fail, has_polymer=True) == "polymer_density_recovery"
    assert select_relaxation_strategy(rg_fail, has_polymer=True) == "polymer_chain_relaxation"
    assert select_relaxation_strategy(converged, has_polymer=True) == "production"


def test_liquid_target_relaxation_micro_starts_cold_and_small_timestep():
    stages = _build_liquid_target_relaxation_stages(
        temp=318.0,
        press=1.0,
        final_ns=0.2,
        dt_ps=0.001,
    )

    assert [stage.name for stage in stages] == [
        "01_minim",
        "02_cold_nvt_0p1fs",
        "03_warm_nvt_0p1fs",
        "04_target_nvt_0p2fs",
        "05_soft_npt_0p25fs",
        "06_final_npt",
    ]
    assert stages[0].mdp.params["emstep"] == 0.0002
    assert stages[1].mdp.params["dt"] == 0.0001
    assert stages[1].mdp.params["ref_t"] < 150.0
    assert stages[1].mdp.params["tcoupl"] == "Berendsen"
    assert stages[2].mdp.params["continuation"] == "yes"
    assert stages[4].mdp.params["pcoupl"] == "Berendsen"
    assert stages[4].mdp.params["compressibility"] == 1.0e-5
    assert stages[-1].mdp.params["dt"] == 0.0005
    assert stages[-1].mdp.params["pcoupl"] == "C-rescale"
    assert stages[-1].mdp.params["constraints"] == "none"
    assert "tcoupl                    = Berendsen" in stages[1].mdp.render()


def test_polymer_density_recovery_schedule_is_unconstrained_and_conservative():
    compact = _build_polymer_density_recovery_compaction_stages(
        warm_temp=450.0,
        compact_pressure_bar=1000.0,
        warm_nvt_ns=0.05,
        compact_npt_ns=0.25,
        dt_ps=0.001,
    )
    release = _build_polymer_density_recovery_release_stages(
        temp=318.0,
        press=1.0,
        final_ns=1.0,
        dt_ps=0.001,
        checkpoint_min=5.0,
    )

    assert [stage.name for stage in compact] == ["01_minim", "02_warm_nvt", "03_compact_npt"]
    assert compact[1].mdp.params["constraints"] == "none"
    assert compact[2].mdp.params["constraints"] == "none"
    assert compact[2].mdp.params["pcoupl"] == "Berendsen"
    assert compact[2].mdp.params["ref_p"] == 1000.0
    assert compact[2].mdp.params["tau_p"] == 8.0
    assert release[-1].name == "04_final_npt"
    assert release[-1].mdp.params["constraints"] == "none"
    assert release[-1].mdp.params["ref_p"] == 1.0


def test_polymer_chain_relaxation_schedule_is_unconstrained_without_high_pressure():
    stages = _build_polymer_chain_relaxation_stages(
        temp=318.0,
        press=1.0,
        final_ns=1.0,
        warm_temp=450.0,
        warm_nvt_ns=0.10,
        dt_ps=0.001,
        checkpoint_min=5.0,
    )

    assert [stage.name for stage in stages] == ["01_warm_nvt", "02_final_npt"]
    assert all(stage.mdp.params["constraints"] == "none" for stage in stages)
    assert stages[1].mdp.params["ref_p"] == 1.0
    assert stages[1].mdp.params["pcoupl"] == "C-rescale"


def test_production_bridge_generates_velocities_only_for_first_stage():
    params = default_mdp_params()
    params.update({"gen_vel": "yes", "gen_temp": 318.15, "constraints": "h-bonds"})
    stages = _build_production_stages(
        stage_name="npt",
        template=NPT_MDP,
        params=params,
        prod_ns=0.1,
        checkpoint_min=5.0,
        constraints_mode="h-bonds",
        bridge_ps=20.0,
        first_stage_gen_vel="yes",
    )

    assert [stage.name for stage in stages] == ["01_settle_constraints", "02_bridge_npt", "03_npt"]
    assert stages[0].kind == "minim"
    assert stages[0].strict_constraints is True
    assert stages[0].mdp.params["constraints"] == "h-bonds"
    assert stages[0].mdp.params["lincs_iter"] == 4
    assert stages[0].mdp.params["lincs_order"] == 12
    assert stages[1].mdp.params["gen_vel"] == "yes"
    assert stages[1].mdp.params["continuation"] == "no"
    assert stages[2].mdp.params["gen_vel"] == "no"
    assert stages[2].mdp.params["continuation"] == "yes"
    assert "continuation              = no" in stages[1].mdp.render()
    assert "continuation              = yes" in stages[2].mdp.render()


def test_production_without_constraints_does_not_add_settle_stage():
    params = default_mdp_params()
    params.update({"gen_vel": "yes", "gen_temp": 318.15, "constraints": "none"})
    stages = _build_production_stages(
        stage_name="npt",
        template=NPT_MDP,
        params=params,
        prod_ns=0.1,
        checkpoint_min=5.0,
        constraints_mode="none",
        bridge_ps=0.0,
        first_stage_gen_vel="yes",
    )

    assert [stage.name for stage in stages] == ["01_npt"]
    assert stages[0].mdp.params["gen_vel"] == "yes"
    assert stages[0].mdp.params["continuation"] == "no"


def test_latest_equilibrated_gro_prefers_liquid_anneal_before_production(tmp_path):
    liquid_gro = tmp_path / "03_liquid_anneal" / "06_final_npt" / "md.gro"
    liquid_gro.parent.mkdir(parents=True)
    liquid_gro.write_text("liquid\n", encoding="utf-8")
    prod_gro = tmp_path / "05_npt_production" / "01_npt" / "md.gro"
    prod_gro.parent.mkdir(parents=True)
    prod_gro.write_text("production\n", encoding="utf-8")

    assert _find_latest_equilibrated_gro(tmp_path, exclude_dirs=[prod_gro.parent.parent]) == liquid_gro


def test_next_additional_round_treats_liquid_recovery_final_npt_as_complete(tmp_path):
    round0 = tmp_path / "04_eq_additional" / "round_00" / "07_final_npt"
    round0.mkdir(parents=True)
    for name in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
        (round0 / name).write_text(name, encoding="utf-8")

    idx, path = _next_additional_round(tmp_path, restart=True)

    assert idx == 1
    assert path == tmp_path / "04_eq_additional" / "round_01"


def test_latest_equilibrated_gro_ignores_incomplete_additional_round(tmp_path):
    liquid_gro = tmp_path / "03_liquid_anneal" / "07_final_npt" / "md.gro"
    liquid_gro.parent.mkdir(parents=True)
    liquid_gro.write_text("liquid\n", encoding="utf-8")

    partial_gro = tmp_path / "04_eq_additional" / "round_00" / "01_minim" / "md.gro"
    partial_gro.parent.mkdir(parents=True)
    partial_gro.write_text("partial\n", encoding="utf-8")

    assert _find_latest_equilibrated_gro(tmp_path) == liquid_gro
