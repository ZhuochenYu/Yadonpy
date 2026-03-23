import numpy as np

from yadonpy.gmx.analysis.plateau import find_plateau_start


def test_density_plateau_relaxed_slope_threshold_accepts_slow_tail():
    t_ps = np.linspace(0.0, 1000.0, 1001)
    y = 1143.88 + 0.0453 * (t_ps - t_ps[0])
    res = find_plateau_start(
        t_ps,
        y,
        min_window_frac=0.2,
        step_frac=0.02,
        slope_threshold_per_ps=5e-2,
        rel_std_threshold=0.01,
    )
    assert res.ok is True
    assert abs(res.slope - 0.0453) < 1e-6


def test_density_plateau_relaxed_slope_threshold_still_rejects_fast_tail():
    t_ps = np.linspace(0.0, 1000.0, 1001)
    y = 1143.88 + 0.0600 * (t_ps - t_ps[0])
    res = find_plateau_start(
        t_ps,
        y,
        min_window_frac=0.2,
        step_frac=0.02,
        slope_threshold_per_ps=5e-2,
        rel_std_threshold=0.01,
    )
    assert res.ok is False
    assert abs(res.slope - 0.0600) < 1e-6
