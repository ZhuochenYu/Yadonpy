from __future__ import annotations

import json
from pathlib import Path

from yadonpy.gmx.mdp_templates import MINIM_CG_MDP, MINIM_STEEP_MDP, MdpSpec, NVT_MDP, default_mdp_params
from yadonpy.gmx.workflows.eq import EqStage, EquilibrationJob
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

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['stages'][0]['auto_cutoff_adjustments']
    assert summary['stages'][0]['auto_cutoff_adjustments'][0]['cutoff_cap_nm'] == 0.81
    assert any('Auto-shrinking nonbond cutoffs' in msg for msg in runner.logs)


def test_interface_protocol_enables_periodic_molecules_for_stage_handoffs():
    stages = InterfaceProtocol.route_a().stages()
    assert stages
    assert all(stage.mdp.params['periodic-molecules'] == 'yes' for stage in stages)
