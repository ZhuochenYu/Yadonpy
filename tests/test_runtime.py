from __future__ import annotations

import importlib
import json
from pathlib import Path

import yadonpy.runtime as runtime
import yadonpy.core.const as const
from yadonpy.gmx.engine import GromacsRunner
from yadonpy.sim.analyzer import AnalyzeResult
from yadonpy.workflow.resume import ResumeManager, StepSpec


def test_parse_bool_accepts_common_tokens():
    assert runtime._parse_bool('yes', default=False) is True
    assert runtime._parse_bool('OFF', default=True) is False
    assert runtime._parse_bool('unexpected', default=True) is True


def test_run_options_context_manager_restores_previous_state():
    original = runtime.get_run_options()
    runtime.set_run_options(restart=True, strict_inputs=True)

    with runtime.run_options(restart=False, strict_inputs=True) as opts:
        assert opts.restart is False
        assert opts.strict_inputs is True
        cur = runtime.get_run_options()
        assert cur.restart is False
        assert cur.strict_inputs is True

    restored = runtime.get_run_options()
    assert restored.restart is True
    assert restored.strict_inputs is True

    runtime.set_run_options(restart=original.restart, strict_inputs=original.strict_inputs)


def test_default_run_options_reads_environment(monkeypatch):
    monkeypatch.setenv('YADONPY_RESTART', '0')
    monkeypatch.setenv('YADONPY_STRICT_INPUTS', '1')
    mod = importlib.reload(runtime)
    try:
        opts = mod._default_run_options()
        assert opts.restart is False
        assert opts.strict_inputs is True
    finally:
        importlib.reload(mod)


def test_default_run_options_are_strict_by_default():
    opts = runtime._default_run_options()
    assert opts.restart is True
    assert opts.strict_inputs is True


def test_resume_manager_does_not_reuse_outputs_without_state_record(tmp_path: Path):
    out = tmp_path / "done.txt"
    out.write_text("ok\n", encoding="utf-8")
    mgr = ResumeManager(tmp_path, strict_inputs=True, enabled=True)
    spec = StepSpec(name="demo", outputs=[out], inputs={"x": 1})
    assert mgr.reuse_status(spec) == "no_record"
    assert mgr.is_done(spec) is False


def test_recommend_local_resources_respects_cpu_cap_and_defaults(monkeypatch):
    monkeypatch.delenv('YADONPY_MPI', raising=False)
    monkeypatch.delenv('YADONPY_OMP', raising=False)
    monkeypatch.delenv('YADONPY_GPU', raising=False)
    monkeypatch.delenv('YADONPY_GPU_ID', raising=False)
    monkeypatch.delenv('YADONPY_OMP_PSI4', raising=False)
    monkeypatch.setattr(runtime.os, 'cpu_count', lambda: 24)

    res = runtime.recommend_local_resources(cpu_cap=12, gpu_default=1, gpu_id_default=0, omp_psi4_cap=8)

    assert res.cpu_total == 24
    assert res.cpu_cap == 12
    assert res.mpi == 1
    assert res.omp == 12
    assert res.gpu == 1
    assert res.gpu_id == 0
    assert res.omp_psi4 == 8


def test_recommend_local_resources_honors_environment_overrides(monkeypatch):
    monkeypatch.setenv('YADONPY_MPI', '2')
    monkeypatch.setenv('YADONPY_OMP', '6')
    monkeypatch.setenv('YADONPY_GPU', '0')
    monkeypatch.setenv('YADONPY_OMP_PSI4', '4')
    monkeypatch.setattr(runtime.os, 'cpu_count', lambda: 12)

    res = runtime.recommend_local_resources(cpu_cap=12, gpu_default=1, gpu_id_default=3, omp_psi4_cap=8)

    assert res.mpi == 2
    assert res.omp == 6
    assert res.gpu == 0
    assert res.gpu_id is None
    assert res.omp_psi4 == 4


def test_tqdm_is_enabled_by_default_and_can_be_disabled_via_environment(monkeypatch):
    monkeypatch.delenv('YADONPY_DISABLE_TQDM', raising=False)
    mod = importlib.reload(const)
    try:
        assert mod.tqdm_disable is False

        monkeypatch.setenv('YADONPY_DISABLE_TQDM', '1')
        mod = importlib.reload(mod)
        assert mod.tqdm_disable is True
    finally:
        monkeypatch.delenv('YADONPY_DISABLE_TQDM', raising=False)
        importlib.reload(mod)


def test_rg_analysis_skips_non_polymer_systems(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / '02_system'
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / 'system_meta.json').write_text(
        json.dumps(
            {
                'species': [
                    {'moltype': 'EC', 'kind': 'solvent', 'smiles': 'O=C1OCCO1'},
                    {'moltype': 'PF6', 'kind': 'anion', 'smiles': 'F[P-](F)(F)(F)(F)F'},
                ]
            },
            indent=2,
        ) + '\n',
        encoding='utf-8',
    )

    def _unexpected_gyrate(*args, **kwargs):
        raise AssertionError('gyrate should not run for non-polymer systems')

    monkeypatch.setattr(GromacsRunner, 'gyrate', _unexpected_gyrate)
    res = AnalyzeResult(
        work_dir=tmp_path,
        tpr=Path(tmp_path / 'md.tpr'),
        xtc=Path(tmp_path / 'md.xtc'),
        edr=Path(tmp_path / 'md.edr'),
        top=Path(tmp_path / 'system.top'),
        ndx=Path(tmp_path / 'system.ndx'),
    ).rg()

    assert res['skipped'] is True
    assert 'no polymer moltypes' in res['reason']


def test_pick_rg_group_prefers_whole_polymer_group_over_rep_atom_group(tmp_path: Path):
    system_dir = tmp_path / '02_system'
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / 'system_meta.json').write_text(
        json.dumps(
            {
                'species': [
                    {'moltype': 'CMC', 'kind': 'polymer', 'smiles': '*CC*'},
                ]
            },
            indent=2,
        ) + '\n',
        encoding='utf-8',
    )
    ndx = tmp_path / 'system.ndx'
    ndx.write_text(
        '\n'.join(
            [
                '[ System ]',
                '1 2 3',
                '[ REP_CMC ]',
                '1',
                '[ MOL_CMC ]',
                '1 2 3',
            ]
        ) + '\n',
        encoding='utf-8',
    )

    analyzer = AnalyzeResult(
        work_dir=tmp_path,
        tpr=Path(tmp_path / 'md.tpr'),
        xtc=Path(tmp_path / 'md.xtc'),
        edr=Path(tmp_path / 'md.edr'),
        top=Path(tmp_path / 'system.top'),
        ndx=ndx,
    )

    assert analyzer._pick_rg_group() == 2


def test_polymer_density_gate_is_more_relaxed_than_liquid_gate(tmp_path: Path):
    analyzer = AnalyzeResult(
        work_dir=tmp_path,
        tpr=Path(tmp_path / 'md.tpr'),
        xtc=Path(tmp_path / 'md.xtc'),
        edr=Path(tmp_path / 'md.edr'),
        top=Path(tmp_path / 'system.top'),
        ndx=Path(tmp_path / 'system.ndx'),
    )

    polymer_kwargs = analyzer._density_plateau_kwargs(has_polymer=True)
    liquid_kwargs = analyzer._density_plateau_kwargs(has_polymer=False)

    assert polymer_kwargs['slope_threshold_per_ps'] > liquid_kwargs['slope_threshold_per_ps']
    assert polymer_kwargs['rel_std_threshold'] > liquid_kwargs['rel_std_threshold']


def test_system_dir_can_be_resolved_from_parent_work_root(tmp_path: Path):
    work_root = tmp_path / 'benchmark' / 'work_dir'
    system_dir = work_root / '02_system'
    stage_dir = work_root / '05_cool_to_60c' / '03_prod_60c'
    system_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    analyzer = AnalyzeResult(
        work_dir=stage_dir,
        tpr=stage_dir / 'md.tpr',
        xtc=stage_dir / 'md.xtc',
        edr=stage_dir / 'md.edr',
        top=work_root / '02_system' / 'system.top',
        ndx=work_root / '02_system' / 'system.ndx',
    )

    assert analyzer._system_dir() == system_dir
