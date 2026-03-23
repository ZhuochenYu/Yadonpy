from __future__ import annotations

from yadonpy.sim.preset.eq import EQ21ProtocolConfig, _build_eq21_records, _eq21_params_payload


def _find(records, label):
    for rec in records:
        if rec['stage_label'] == label:
            return rec
    raise AssertionError(f'missing record: {label}')


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
