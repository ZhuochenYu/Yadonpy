"""Regression tests for the CMC-Na carbonate benchmark defaults."""

from __future__ import annotations

import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "examples/05_cmcna_electrolyte/benchmark_cmcna_carbonate_lipf6_bulk.py"


def _load_defaults(monkeypatch, forcefield: str, **env: str):
    for key in list(env):
        monkeypatch.setenv(key, str(env[key]))
    monkeypatch.setenv("YADONPY_FORCEFIELD", forcefield)
    return runpy.run_path(str(SCRIPT))


def test_oplsaa_defaults_use_stable_cmc_production_timestep(monkeypatch):
    ns = _load_defaults(monkeypatch, "oplsaa")

    assert ns["prod_dt_ps"] == 0.001
    assert ns["prod_lincs_iter"] == 4
    assert ns["prod_lincs_order"] == 12
    assert ns["gpu_offload_mode"] == "conservative"
    assert ns["eq_gpu_offload_mode"] == "conservative"


def test_gaff2_defaults_keep_hbonds_two_fs(monkeypatch):
    ns = _load_defaults(monkeypatch, "gaff2")

    assert ns["prod_dt_ps"] == 0.002
    assert ns["prod_lincs_iter"] == 2
    assert ns["prod_lincs_order"] == 8
    assert ns["gpu_offload_mode"] == "full"


def test_cmc_production_controls_remain_user_overridable(monkeypatch):
    ns = _load_defaults(
        monkeypatch,
        "oplsaa",
        YADONPY_PROD_DT_PS="0.002",
        YADONPY_PROD_LINCS_ITER="6",
        YADONPY_PROD_LINCS_ORDER="14",
        YADONPY_GPU_OFFLOAD_MODE="balanced",
    )

    assert ns["prod_dt_ps"] == 0.002
    assert ns["prod_lincs_iter"] == 6
    assert ns["prod_lincs_order"] == 14
    assert ns["gpu_offload_mode"] == "balanced"
