from __future__ import annotations

import json
from pathlib import Path

import pytest

from yadonpy.gmx.mdp_templates import MINIM_CG_MDP, MINIM_STEEP_MDP, MdpSpec, NVT_MDP, default_mdp_params
from yadonpy.gmx.workflows._util import RunResources
from yadonpy.gmx.workflows.eq import EqStage, EquilibrationJob, StageLincsRetryPolicy
from yadonpy.interface import InterfaceProtocol


class FakeRunner:
    def __init__(self):
        self.grompp_calls = 0
        self.mdrun_calls = 0
        self.logs = []

    def _log(self, msg: str) -> None:
        self.logs.append(str(msg))

    def grompp(self, *, out_tpr: Path, **kwargs) -> None:
        self.grompp_calls += 1
        out_tpr.write_text('fake tpr\n', encoding='utf-8')

    def mdrun(self, *, deffnm: str, cwd: Path, **kwargs) -> None:
        self.mdrun_calls += 1
        cwd = Path(cwd)
        (cwd / f'{deffnm}.gro').write_text('fake gro\n', encoding='utf-8')
        (cwd / f'{deffnm}.cpt').write_text('fake cpt\n', encoding='utf-8')


class FakeMinimRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.em_commands = []

    def _tool_has_option(self, command: str, option: str | None = None, cwd: Path | None = None) -> bool:
        return True

    def _run_capture_tee(self, args, cwd: Path | None = None):
        self.em_commands.append(list(args))
        deffnm = None
        if '-deffnm' in args:
            deffnm = args[args.index('-deffnm') + 1]
        cwd = Path(cwd or '.')
        if deffnm == 'md_steep_hbonds':
            return 1, 'Fatal error: Too many LINCS warnings'
        if deffnm:
            (cwd / f'{deffnm}.gro').write_text('fake gro\n', encoding='utf-8')
            (cwd / f'{deffnm}.cpt').write_text('fake cpt\n', encoding='utf-8')
        return 0, ''


class FakeWholeMoleculeRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.grompp_gro_texts = []

    def grompp(self, *, gro: Path, out_tpr: Path, **kwargs) -> None:
        self.grompp_calls += 1
        self.grompp_gro_texts.append(Path(gro).read_text(encoding='utf-8'))
        out_tpr.write_text('fake tpr\n', encoding='utf-8')

    def mdrun(self, *, deffnm: str, cwd: Path, **kwargs) -> None:
        self.mdrun_calls += 1
        cwd = Path(cwd)
        if self.mdrun_calls == 1:
            (cwd / f'{deffnm}.gro').write_text(
                '\n'.join(
                    [
                        'broken across boundary',
                        '    2',
                        '    1MOL     A    1   0.950   0.100   0.100',
                        '    1MOL     B    2   0.050   0.100   0.100',
                        '   1.00000   1.00000   1.00000',
                    ]
                ) + '\n',
                encoding='utf-8',
            )
        else:
            (cwd / f'{deffnm}.gro').write_text(
                '\n'.join(
                    [
                        'final',
                        '    2',
                        '    1MOL     A    1   0.100   0.100   0.100',
                        '    1MOL     B    2   0.200   0.100   0.100',
                        '   1.00000   1.00000   1.00000',
                    ]
                ) + '\n',
                encoding='utf-8',
            )
        (cwd / f'{deffnm}.cpt').write_text('fake cpt\n', encoding='utf-8')


class FakeCutoffRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.grompp_mdp_texts = []

    def grompp(self, *, mdp: Path, out_tpr: Path, **kwargs) -> None:
        self.grompp_calls += 1
        self.grompp_mdp_texts.append(Path(mdp).read_text(encoding='utf-8'))
        out_tpr.write_text('fake tpr\n', encoding='utf-8')

    def mdrun(self, *, deffnm: str, cwd: Path, **kwargs) -> None:
        self.mdrun_calls += 1
        cwd = Path(cwd)
        (cwd / f'{deffnm}.gro').write_text(
            '\n'.join(
                [
                    'final',
                    '    1',
                    '    1MOL     A    1   0.100   0.100   0.100',
                    '   1.80000   2.00000   2.10000',
                ]
            ) + '\n',
            encoding='utf-8',
        )
        (cwd / f'{deffnm}.cpt').write_text('fake cpt\n', encoding='utf-8')


class FakeInvalidMinimRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.invalid_mdrun_calls = 0

    def mdrun(self, *, deffnm: str, cwd: Path, **kwargs) -> None:
        self.invalid_mdrun_calls += 1
        cwd = Path(cwd)
        (cwd / f'{deffnm}.gro').write_text('fake gro\n', encoding='utf-8')
        (cwd / f'{deffnm}.cpt').write_text('fake cpt\n', encoding='utf-8')
        (cwd / f'{deffnm}.log').write_text(
            '\n'.join(
                [
                    'Steepest Descents:',
                    'Energy minimization has stopped because the force on at least one atom is not finite.',
                    'This usually means atoms are overlapping.',
                    'Maximum force     =            inf',
                    'Norm of force     =            inf',
                ]
            ) + '\n',
            encoding='utf-8',
        )


class FakeLincsRetryRunner(FakeRunner):
    def __init__(self, *, write_checkpoint_on_failure: bool = True):
        super().__init__()
        self.write_checkpoint_on_failure = write_checkpoint_on_failure
        self.grompp_records = []
        self.mdrun_records = []

    def grompp(self, *, mdp: Path, out_tpr: Path, **kwargs) -> None:
        self.grompp_calls += 1
        self.grompp_records.append(
            {
                "out_tpr": Path(out_tpr),
                "mdp_text": Path(mdp).read_text(encoding="utf-8"),
                "kwargs": dict(kwargs),
            }
        )
        Path(out_tpr).write_text('fake tpr\n', encoding='utf-8')

    def mdrun(self, *, tpr: Path, deffnm: str, cwd: Path, cpi=None, append=True, **kwargs) -> None:
        self.mdrun_calls += 1
        cwd = Path(cwd)
        self.mdrun_records.append(
            {
                "tpr": Path(tpr),
                "deffnm": deffnm,
                "cwd": cwd,
                "cpi": None if cpi is None else Path(cpi),
                "append": bool(append),
                "kwargs": dict(kwargs),
            }
        )
        if self.mdrun_calls == 1:
            if self.write_checkpoint_on_failure:
                (cwd / f'{deffnm}.cpt').write_text('fake cpt\n', encoding='utf-8')
            (cwd / f'{deffnm}.log').write_text(
                "Step 4000, time 8.000 (ps)\nFatal error: Too many LINCS warnings\n",
                encoding='utf-8',
            )
            raise RuntimeError('Fatal error: Too many LINCS warnings')

        (cwd / f'{deffnm}.gro').write_text('fake gro\n', encoding='utf-8')
        (cwd / f'{deffnm}.cpt').write_text('fake cpt\n', encoding='utf-8')
        (cwd / f'{deffnm}.edr').write_text('fake edr\n', encoding='utf-8')
        (cwd / f'{deffnm}.log').write_text('resumed successfully\n', encoding='utf-8')


class FakeConservativeGpuRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.mdrun_records = []

    def mdrun(self, *, deffnm: str, cwd: Path, **kwargs) -> None:
        self.mdrun_calls += 1
        self.mdrun_records.append(dict(kwargs))
        cwd = Path(cwd)
        (cwd / f'{deffnm}.gro').write_text('fake gro\n', encoding='utf-8')
        (cwd / f'{deffnm}.cpt').write_text('fake cpt\n', encoding='utf-8')
        (cwd / f'{deffnm}.edr').write_text('fake edr\n', encoding='utf-8')
        (cwd / f'{deffnm}.log').write_text('ok\n', encoding='utf-8')


class FakeCheckpointHandoffRunner(FakeRunner):
    def __init__(self):
        super().__init__()
        self.grompp_cpt_by_tpr = {}
        self.mdrun_records = []

    def grompp(self, *, out_tpr: Path, cpt=None, **kwargs) -> None:
        self.grompp_calls += 1
        out_tpr = Path(out_tpr)
        self.grompp_cpt_by_tpr[str(out_tpr)] = None if cpt is None else str(cpt)
        out_tpr.write_text('fake tpr\n', encoding='utf-8')

    def mdrun(self, *, tpr: Path, deffnm: str, cwd: Path, append=True, **kwargs) -> None:
        self.mdrun_calls += 1
        cwd = Path(cwd)
        tpr = Path(tpr)
        self.mdrun_records.append(
            {
                'tpr': str(tpr),
                'cwd': str(cwd),
                'append': bool(append),
                'kwargs': dict(kwargs),
                'grompp_cpt': self.grompp_cpt_by_tpr.get(str(tpr)),
            }
        )
        if self.mdrun_calls == 2 and self.grompp_cpt_by_tpr.get(str(tpr)) is not None:
            (cwd / f'{deffnm}.log').write_text('starting mdrun\n', encoding='utf-8')
            raise RuntimeError('GROMACS mdrun failed\n  reason: terminated by signal 11 (SIGSEGV)\n')

        (cwd / f'{deffnm}.gro').write_text('fake gro\n', encoding='utf-8')
        (cwd / f'{deffnm}.cpt').write_text('fake cpt\n', encoding='utf-8')
        (cwd / f'{deffnm}.edr').write_text('fake edr\n', encoding='utf-8')
        (cwd / f'{deffnm}.log').write_text('ok\n', encoding='utf-8')


def _write_diatomic_topology(root: Path) -> tuple[Path, Path]:
    gro = root / 'input.gro'
    mol_dir = root / 'molecules'
    mol_dir.mkdir(parents=True, exist_ok=True)
    top = root / 'input.top'
    top.write_text(
        '#include "molecules/MOL.itp"\n\n[ system ]\nwhole test\n\n[ molecules ]\nMOL 1\n',
        encoding='utf-8',
    )
    (mol_dir / 'MOL.itp').write_text(
        '\n'.join(
            [
                '[ moleculetype ]',
                'MOL   3',
                '',
                '[ atoms ]',
                '1   C   1  MOL  A   1  0.0  12.011',
                '2   C   1  MOL  B   1  0.0  12.011',
                '',
                '[ bonds ]',
                '1 2',
            ]
        ) + '\n',
        encoding='utf-8',
    )
    gro.write_text(
        '\n'.join(
            [
                'initial',
                '    2',
                '    1MOL     A    1   0.100   0.100   0.100',
                '    1MOL     B    2   0.200   0.100   0.100',
                '   1.00000   1.00000   1.00000',
            ]
        ) + '\n',
        encoding='utf-8',
    )
    return gro, top


def _gro_x_coords(text: str) -> tuple[float, float]:
    lines = text.splitlines()
    first = float(lines[2][20:28])
    second = float(lines[3][20:28])
    return first, second


def test_default_stages_keep_single_velocity_seed_and_c_rescale_npt():
    stages = EquilibrationJob.default_stages(temperature_k=300.0, pressure_bar=1.0, dt_ps=0.001)

    assert [stage.name for stage in stages] == ['01_minim', '02_nvt', '03_npt', '04_md']
    assert stages[1].mdp.params['gen_vel'] == 'yes'
    assert stages[2].mdp.params['gen_vel'] == 'no'
    assert stages[3].mdp.params['gen_vel'] == 'no'
    assert stages[2].mdp.params['pcoupl'] == 'C-rescale'
    assert stages[3].mdp.params['pcoupl'] == 'C-rescale'
    assert stages[1].mdp.params['constraints'] == 'none'
    assert stages[2].mdp.params['constraints'] == 'none'
    assert stages[3].mdp.params['constraints'] == 'none'


def test_equilibration_job_smoke_run_and_restart_skip(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro = tmp_path / 'input.gro'
    top = tmp_path / 'input.top'
    gro.write_text('fake input gro\n', encoding='utf-8')
    top.write_text('fake input top\n', encoding='utf-8')

    params = default_mdp_params()
    params.update(
        {
            'nsteps': 1000,
            'dt': 0.001,
            'ref_t': 300.0,
            'gen_vel': 'yes',
            'gen_temp': 300.0,
            'gen_seed': -1,
        }
    )
    stage = EqStage(name='01_nvt', kind='nvt', mdp=MdpSpec(NVT_MDP, params))
    runner = FakeRunner()
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_job', stages=[stage], runner=runner)

    summary_path = job.run(restart=False)
    assert summary_path.exists()
    assert runner.grompp_calls == 1
    assert runner.mdrun_calls == 1
    assert job.final_gro().exists()

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['job'] == 'EquilibrationJob'
    assert len(summary['stages']) == 1
    assert summary['stages'][0]['name'] == '01_nvt'

    second_summary = job.run(restart=True)
    assert second_summary == summary_path
    assert runner.grompp_calls == 1
    assert runner.mdrun_calls == 1


def test_equilibration_job_conservative_gpu_mode_passes_single_offload_override(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro = tmp_path / 'input.gro'
    top = tmp_path / 'input.top'
    gro.write_text('fake input gro\n', encoding='utf-8')
    top.write_text('fake input top\n', encoding='utf-8')

    params = default_mdp_params()
    params.update(
        {
            'nsteps': 1000,
            'dt': 0.001,
            'ref_t': 300.0,
            'gen_vel': 'no',
        }
    )
    stage = EqStage(name='01_md', kind='md', mdp=MdpSpec(NVT_MDP, params))
    runner = FakeConservativeGpuRunner()
    job = EquilibrationJob(
        gro=gro,
        top=top,
        out_dir=tmp_path / 'eq_job_conservative',
        stages=[stage],
        runner=runner,
        resources=RunResources(ntmpi=1, ntomp=2, use_gpu=True, gpu_id='0', gpu_offload_mode='conservative'),
    )

    summary_path = job.run(restart=False)

    assert summary_path.exists()
    assert runner.mdrun_calls == 1
    kwargs = runner.mdrun_records[0]
    assert kwargs['use_gpu'] is True
    assert kwargs['prefer_gpu_update'] is False
    assert kwargs['nb'] == 'gpu'
    assert kwargs['bonded'] == 'cpu'
    assert kwargs['pme'] == 'gpu'
    assert kwargs['pmefft'] == 'gpu'
    assert kwargs['update'] == 'cpu'


def test_equilibration_job_retries_unconstrained_stage_without_checkpoint_handoff_after_sigsegv(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro = tmp_path / 'input.gro'
    top = tmp_path / 'input.top'
    gro.write_text('fake input gro\n', encoding='utf-8')
    top.write_text('fake input top\n', encoding='utf-8')

    params = default_mdp_params()
    params.update(
        {
            'nsteps': 1000,
            'dt': 0.001,
            'ref_t': 300.0,
            'gen_vel': 'no',
            'constraints': 'none',
        }
    )
    stages = [
        EqStage(name='01_nvt', kind='nvt', mdp=MdpSpec(NVT_MDP, {**params, 'gen_vel': 'yes'})),
        EqStage(name='02_nvt', kind='nvt', mdp=MdpSpec(NVT_MDP, params)),
    ]
    runner = FakeCheckpointHandoffRunner()
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_job_checkpoint_retry', stages=stages, runner=runner)

    summary_path = job.run(restart=False)

    assert summary_path.exists()
    assert runner.grompp_calls == 3
    assert runner.mdrun_calls == 3
    assert runner.mdrun_records[1]['grompp_cpt'] is not None
    assert runner.mdrun_records[2]['grompp_cpt'] is None
    assert runner.mdrun_records[2]['append'] is False

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    fallback = summary['stages'][1]['checkpoint_handoff_fallback']
    assert fallback['triggered'] is True
    assert fallback['original_cpt'].endswith('md.cpt')
    assert fallback['retry_tpr'].endswith('md_nocpt_retry.tpr')


def test_equilibration_job_skips_stage_mol2_export_for_large_systems(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'normalize_gro_molecules_inplace', lambda *args, **kwargs: {'applied': False, 'error': None, 'normalized_molecules': 0})
    mol2_calls = []
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: mol2_calls.append(kwargs) or None)
    monkeypatch.setattr(eqmod.EquilibrationJob, '_gro_atom_count', staticmethod(lambda gro: 100000))

    gro = tmp_path / 'input.gro'
    top = tmp_path / 'input.top'
    gro.write_text('fake input gro\n', encoding='utf-8')
    top.write_text('fake input top\n', encoding='utf-8')

    params = default_mdp_params()
    params.update(
        {
            'nsteps': 1000,
            'dt': 0.001,
            'ref_t': 300.0,
            'gen_vel': 'yes',
            'gen_temp': 300.0,
            'gen_seed': -1,
        }
    )
    stage = EqStage(name='01_nvt', kind='nvt', mdp=MdpSpec(NVT_MDP, params))
    runner = FakeRunner()
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_job_large', stages=[stage], runner=runner)

    summary_path = job.run(restart=False)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))

    assert not mol2_calls
    assert summary['stages'][0]['mol2'] is None
    assert 'large system' in str(summary['stages'][0]['mol2_skipped'])


def test_equilibration_job_minim_bridge_lincs_failure_falls_back_cleanly(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro = tmp_path / 'input.gro'
    top = tmp_path / 'input.top'
    gro.write_text('fake input gro\n', encoding='utf-8')
    top.write_text('fake input top\n', encoding='utf-8')

    params = default_mdp_params()
    params.update({'nsteps': 1000, 'emtol': 1000.0, 'emstep': 0.001})
    stage = EqStage(name='01_pre_contact_em', kind='minim', mdp=MdpSpec(MINIM_CG_MDP, params))
    runner = FakeMinimRunner()
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_job_minim', stages=[stage], runner=runner)

    summary_path = job.run(restart=False)

    assert summary_path.exists()
    assert job.final_gro().exists()
    assert any('steep_hbonds bridge failed' in msg for msg in runner.logs)
    assert any('Replacing the final CG minimization with an unconstrained steep minimization' in msg for msg in runner.logs)
    assert runner.em_commands
    assert all('-nb' not in cmd for cmd in runner.em_commands)
    assert job.final_gro().exists()

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['job'] == 'EquilibrationJob'
    assert len(summary['stages']) == 1
    assert summary['stages'][0]['name'] == '01_pre_contact_em'

    second_summary = job.run(restart=True)
    assert second_summary == summary_path
    assert runner.grompp_calls >= 3
    assert runner.mdrun_calls == 1


def test_equilibration_job_rejects_nonfinite_minimization_results(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro = tmp_path / 'input.gro'
    top = tmp_path / 'input.top'
    gro.write_text('fake input gro\n', encoding='utf-8')
    top.write_text('fake input top\n', encoding='utf-8')

    params = default_mdp_params()
    params.update({'nsteps': 1000, 'emtol': 1000.0, 'emstep': 0.001})
    stage = EqStage(name='01_em', kind='minim', mdp=MdpSpec(MINIM_STEEP_MDP, params))
    runner = FakeInvalidMinimRunner()
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_job_invalid_minim', stages=[stage], runner=runner)

    with pytest.raises(RuntimeError, match='Invalid energy minimization detected'):
        job.run(restart=False)


def test_equilibration_job_reruns_invalid_cached_minimization_outputs(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro = tmp_path / 'input.gro'
    top = tmp_path / 'input.top'
    gro.write_text('fake input gro\n', encoding='utf-8')
    top.write_text('fake input top\n', encoding='utf-8')

    params = default_mdp_params()
    params.update({'nsteps': 1000, 'emtol': 1000.0, 'emstep': 0.001})
    stage = EqStage(name='01_em', kind='minim', mdp=MdpSpec(MINIM_CG_MDP, params))
    runner = FakeMinimRunner()
    out_dir = tmp_path / 'eq_job_cached_invalid'
    stage_dir = out_dir / '01_em'
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / 'md.gro').write_text('stale gro\n', encoding='utf-8')
    (stage_dir / 'md.tpr').write_text('stale tpr\n', encoding='utf-8')
    (stage_dir / 'summary.json').write_text('{"stale": true}\n', encoding='utf-8')
    (stage_dir / 'md.log').write_text(
        'Energy minimization has stopped because the force on at least one atom is not finite.\n',
        encoding='utf-8',
    )

    job = EquilibrationJob(gro=gro, top=top, out_dir=out_dir, stages=[stage], runner=runner)
    summary_path = job.run(restart=True)

    assert summary_path.exists()
    assert runner.grompp_calls >= 2
    assert runner.em_commands


def test_equilibration_job_unconstrained_minim_skips_hbond_bridge(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro = tmp_path / 'input.gro'
    top = tmp_path / 'input.top'
    gro.write_text('fake input gro\n', encoding='utf-8')
    top.write_text('fake input top\n', encoding='utf-8')

    params = default_mdp_params()
    params.update({'nsteps': 1000, 'emtol': 1000.0, 'emstep': 0.0005})
    stage = EqStage(name='01_pre_contact_em', kind='minim', mdp=MdpSpec(MINIM_STEEP_MDP, params))
    runner = FakeMinimRunner()
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_job_unconstrained', stages=[stage], runner=runner)

    summary_path = job.run(restart=False)

    assert summary_path.exists()
    assert job.final_gro().exists()
    assert not any('steep_hbonds bridge failed' in msg for msg in runner.logs)
    assert not any('Replacing the final CG minimization' in msg for msg in runner.logs)
    assert all('md_steep_hbonds' not in cmd for cmd in (' '.join(parts) for parts in runner.em_commands))


def test_equilibration_job_canonicalizes_stage_output_gro_between_stages(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro, top = _write_diatomic_topology(tmp_path)
    params = default_mdp_params()
    params.update(
        {
            'nsteps': 1000,
            'dt': 0.001,
            'ref_t': 300.0,
            'gen_vel': 'yes',
            'gen_temp': 300.0,
            'gen_seed': -1,
        }
    )
    stages = [
        EqStage(name='01_nvt', kind='nvt', mdp=MdpSpec(NVT_MDP, params)),
        EqStage(name='02_nvt', kind='nvt', mdp=MdpSpec(NVT_MDP, {**params, 'gen_vel': 'no'})),
    ]
    runner = FakeWholeMoleculeRunner()
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_whole', stages=stages, runner=runner)

    summary_path = job.run(restart=False)

    assert summary_path.exists()
    assert runner.grompp_calls == 2
    stage2_x1, stage2_x2 = _gro_x_coords(runner.grompp_gro_texts[1])
    assert abs(stage2_x2 - stage2_x1) < 0.2
    assert any('Whole-molecule canonicalization applied' in msg for msg in runner.logs)


def test_equilibration_job_auto_shrinks_cutoffs_for_small_box(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'normalize_gro_molecules_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro = tmp_path / 'small_box.gro'
    top = tmp_path / 'input.top'
    gro.write_text(
        '\n'.join(
            [
                'small box',
                '    1',
                '    1MOL     A    1   0.100   0.100   0.100',
                '   1.80000   2.00000   2.10000',
            ]
        ) + '\n',
        encoding='utf-8',
    )
    top.write_text('fake input top\n', encoding='utf-8')

    params = default_mdp_params()
    params.update(
        {
            'nsteps': 1000,
            'dt': 0.001,
            'ref_t': 300.0,
            'gen_vel': 'yes',
            'gen_temp': 300.0,
            'gen_seed': -1,
        }
    )
    stage = EqStage(name='01_nvt', kind='nvt', mdp=MdpSpec(NVT_MDP, params))
    runner = FakeCutoffRunner()
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_cutoff', stages=[stage], runner=runner)

    summary_path = job.run(restart=False)

    assert summary_path.exists()
    assert runner.grompp_calls == 1
    assert runner.mdrun_calls == 1
    assert any('rlist                    = 0.81' in text for text in runner.grompp_mdp_texts)
    assert any('rcoulomb                 = 0.81' in text for text in runner.grompp_mdp_texts)
    assert any('rvdw                     = 0.81' in text for text in runner.grompp_mdp_texts)
    assert any('nstlist                  = 10' in text for text in runner.grompp_mdp_texts)
    assert any('verlet-buffer-tolerance  = 0.02' in text for text in runner.grompp_mdp_texts)

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['stages'][0]['auto_cutoff_adjustments']
    assert summary['stages'][0]['auto_cutoff_adjustments'][0]['cutoff_cap_nm'] == 0.81
    assert summary['stages'][0]['auto_cutoff_adjustments'][0]['verlet_safety']['nstlist']['new'] == 10.0
    assert summary['stages'][0]['auto_cutoff_adjustments'][0]['verlet_safety']['verlet_buffer_tolerance']['new'] == 0.02
    assert any('Auto-shrinking nonbond cutoffs' in msg for msg in runner.logs)


def test_equilibration_job_retries_production_stage_with_stronger_lincs_after_failure(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'normalize_gro_molecules_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro, top = _write_diatomic_topology(tmp_path)
    params = default_mdp_params()
    params.update(
        {
            'nsteps': 6000,
            'dt': 0.002,
            'ref_t': 300.0,
            'gen_vel': 'yes',
            'gen_temp': 300.0,
            'gen_seed': -1,
        }
    )
    stage = EqStage(
        name='01_npt',
        kind='md',
        mdp=MdpSpec(NVT_MDP, params),
        lincs_retry=StageLincsRetryPolicy(),
    )
    runner = FakeLincsRetryRunner()
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_lincs_retry', stages=[stage], runner=runner)

    summary_path = job.run(restart=False)

    assert summary_path.exists()
    assert runner.grompp_calls == 2
    assert runner.mdrun_calls == 2
    assert runner.mdrun_records[1]["tpr"].name == "md_lincs_retry.tpr"
    assert runner.mdrun_records[1]["cpi"] == job.out_dir / "01_npt" / "md.cpt"
    assert runner.mdrun_records[1]["append"] is True
    retry_mdp = runner.grompp_records[-1]["mdp_text"]
    assert "dt                       = 0.001" in retry_mdp
    assert "nsteps                   = 4000" in retry_mdp
    assert "lincs_iter               = 4" in retry_mdp
    assert "lincs_order              = 12" in retry_mdp

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    fallback = summary["stages"][0]["lincs_fallback"]
    assert fallback["triggered"] is True
    assert fallback["failed_time_ps"] == pytest.approx(8.0)
    assert fallback["remaining_time_ps"] == pytest.approx(4.0)
    assert fallback["retry_nsteps"] == 4000
    assert fallback["resumed_from_pending_state"] is False
    assert not (job.out_dir / "01_npt" / "lincs_fallback_state.json").exists()
    assert any("Production LINCS fallback triggered" in msg for msg in runner.logs)


def test_equilibration_job_passes_stage_checkpoint_minutes_to_mdrun(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'normalize_gro_molecules_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro, top = _write_diatomic_topology(tmp_path)
    params = default_mdp_params()
    params.update(
        {
            'nsteps': 1000,
            'dt': 0.002,
            'ref_t': 300.0,
            'gen_vel': 'yes',
            'gen_temp': 300.0,
            'gen_seed': -1,
        }
    )
    stage = EqStage(
        name='01_npt',
        kind='md',
        mdp=MdpSpec(NVT_MDP, params),
        checkpoint_minutes=2.5,
    )
    class _RecordingRunner(FakeRunner):
        def __init__(self):
            super().__init__()
            self.mdrun_records = []

        def mdrun(self, *, tpr: Path, deffnm: str, cwd: Path, cpi=None, append=True, **kwargs) -> None:
            self.mdrun_calls += 1
            cwd = Path(cwd)
            self.mdrun_records.append(
                {
                    "tpr": Path(tpr),
                    "deffnm": deffnm,
                    "cwd": cwd,
                    "cpi": None if cpi is None else Path(cpi),
                    "append": bool(append),
                    "kwargs": dict(kwargs),
                }
            )
            (cwd / f'{deffnm}.gro').write_text('fake gro\n', encoding='utf-8')
            (cwd / f'{deffnm}.cpt').write_text('fake cpt\n', encoding='utf-8')
            (cwd / f'{deffnm}.edr').write_text('fake edr\n', encoding='utf-8')
            (cwd / f'{deffnm}.log').write_text('ok\n', encoding='utf-8')

    runner = _RecordingRunner()
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_checkpoint_minutes', stages=[stage], runner=runner)

    summary_path = job.run(restart=False)

    assert summary_path.exists()
    assert runner.mdrun_records
    assert runner.mdrun_records[0]["kwargs"]["checkpoint_minutes"] == pytest.approx(2.5)


def test_equilibration_job_refuses_lincs_retry_without_checkpoint(tmp_path, monkeypatch):
    import yadonpy.gmx.workflows.eq as eqmod

    monkeypatch.setattr(eqmod, 'pbc_mol_fix_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'normalize_gro_molecules_inplace', lambda *args, **kwargs: {'applied': False, 'error': None})
    monkeypatch.setattr(eqmod, 'write_mol2_from_top_gro_parmed', lambda **kwargs: None)

    gro, top = _write_diatomic_topology(tmp_path)
    params = default_mdp_params()
    params.update(
        {
            'nsteps': 6000,
            'dt': 0.002,
            'ref_t': 300.0,
            'gen_vel': 'yes',
            'gen_temp': 300.0,
            'gen_seed': -1,
        }
    )
    stage = EqStage(
        name='01_npt',
        kind='md',
        mdp=MdpSpec(NVT_MDP, params),
        lincs_retry=StageLincsRetryPolicy(),
    )
    runner = FakeLincsRetryRunner(write_checkpoint_on_failure=False)
    job = EquilibrationJob(gro=gro, top=top, out_dir=tmp_path / 'eq_lincs_retry_no_cpt', stages=[stage], runner=runner)

    with pytest.raises(RuntimeError, match="no checkpoint"):
        job.run(restart=False)


def test_interface_protocol_enables_periodic_molecules_for_stage_handoffs():
    stages = InterfaceProtocol.route_a().stages()
    assert stages
    assert all(stage.mdp.params['periodic-molecules'] == 'yes' for stage in stages)
