from __future__ import annotations

from pathlib import Path

from yadonpy.gmx.engine import GromacsExec, GromacsRunner
from yadonpy.sim.preset import eq
from yadonpy.workflow.config import EnvReader, ResourceConfig


def test_parse_gpu_args_gpu_zero_ignores_gpu_id():
    use_gpu, gpu_id = eq._parse_gpu_args(0, 3)

    assert use_gpu is False
    assert gpu_id is None


def test_parse_gpu_args_gpu_zero_ignores_invalid_gpu_id_token():
    use_gpu, gpu_id = eq._parse_gpu_args("0", "stale-gpu-id")

    assert use_gpu is False
    assert gpu_id is None


def test_resource_config_ignores_gpu_id_when_gpu_disabled():
    cfg = ResourceConfig.from_env(
        EnvReader(
            {
                "MPI": "1",
                "OMP": "4",
                "GPU": "0",
                "GPU_ID": "not-a-valid-id",
            }
        ),
        default_gpu=1,
    )

    assert cfg.gpu == 0
    assert cfg.gpu_id is None


def test_resource_config_keeps_gpu_id_when_gpu_enabled():
    cfg = ResourceConfig.from_env(
        EnvReader(
            {
                "MPI": "1",
                "OMP": "4",
                "GPU": "1",
                "GPU_ID": "2",
            }
        )
    )

    assert cfg.gpu == 1
    assert cfg.gpu_id == 2


def test_mdrun_cpu_mode_omits_gpu_id_even_if_supplied(monkeypatch, tmp_path: Path):
    runner = GromacsRunner(exec_=GromacsExec("gmx"), verbose=False)
    calls: list[tuple[list[str], dict[str, str]]] = []

    monkeypatch.setattr(runner, "_tool_has_option", lambda command, option=None, cwd=None: True)

    def _fake_run_capture_tee(args, *, cwd=None, env=None, tail_chars=8000):
        calls.append((list(args), dict(env or {})))
        return 0, ""

    monkeypatch.setattr(runner, "_run_capture_tee", _fake_run_capture_tee)

    runner.mdrun(
        tpr=tmp_path / "md.tpr",
        deffnm="md",
        cwd=tmp_path,
        ntomp=4,
        ntmpi=1,
        use_gpu=False,
        gpu_id="2",
    )

    assert len(calls) == 1
    cmd, env = calls[0]
    assert "-gpu_id" not in cmd
    assert cmd[cmd.index("-nb") + 1] == "cpu"
    assert cmd[cmd.index("-bonded") + 1] == "cpu"
    assert cmd[cmd.index("-update") + 1] == "cpu"
    assert cmd[cmd.index("-pme") + 1] == "cpu"
    assert env["GMX_DISABLE_GPU_DETECTION"] == "1"
