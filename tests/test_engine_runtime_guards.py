from __future__ import annotations

import io
from pathlib import Path

import pytest

from yadonpy.core import poly
from yadonpy.gmx.engine import GromacsExec, GromacsRunner
from yadonpy.runtime import run_options


class _FakeBuffer:
    def __init__(self):
        self.chunks = []

    def write(self, data: bytes) -> None:
        self.chunks.append(bytes(data))

    def flush(self) -> None:
        return None


class _FakeStdout:
    def __init__(self):
        self.buffer = _FakeBuffer()


class _FakePopen:
    def __init__(self, *args, **kwargs):
        self.stdout = io.BytesIO(b'progress\rnext\n')
        self.returncode = 0

    def wait(self):
        return 0



def test_run_capture_tee_respects_verbose_flag(monkeypatch, tmp_path: Path):
    import subprocess
    import sys
    import yadonpy.gmx.engine as engine_mod

    runner = GromacsRunner(exec_=GromacsExec('gmx'), verbose=False)

    monkeypatch.setattr(subprocess, 'Popen', _FakePopen)
    fake_stdout = _FakeStdout()
    monkeypatch.setattr(sys, 'stdout', fake_stdout)

    rc, tail = runner._run_capture_tee(['mdrun', '-deffnm', 'md'], cwd=tmp_path)

    assert rc == 0
    assert 'progress' in tail
    assert fake_stdout.buffer.chunks == []



def test_effective_restart_flag_uses_runtime_default_when_no_explicit_override():
    with run_options(restart=True):
        assert poly._effective_restart_flag(None, None) is True
    with run_options(restart=False):
        assert poly._effective_restart_flag(None, None) is False



def test_effective_restart_flag_detects_conflicting_controls():
    with pytest.raises(ValueError, match='Conflicting restart controls'):
        poly._effective_restart_flag(None, True, restart_flag=False)


def test_mdrun_cuda_internal_error_falls_back_to_cpu_kernels(monkeypatch, tmp_path: Path):
    runner = GromacsRunner(exec_=GromacsExec('gmx'), verbose=False)
    calls: list[tuple[list[str], dict[str, str]]] = []

    monkeypatch.setattr(runner, '_tool_has_option', lambda command, option=None, cwd=None: True)

    def _fake_run_capture_tee(args, *, cwd=None, env=None, tail_chars=8000):
        calls.append((list(args), dict(env or {})))
        if len(calls) == 1:
            return (
                -6,
                "terminate called after throwing an instance of 'gmx::InternalError'\n"
                "what():  Freeing of the device buffer failed. CUDA error #1 "
                "(cudaErrorInvalidValue): invalid argument.\n",
            )
        return 0, ""

    monkeypatch.setattr(runner, '_run_capture_tee', _fake_run_capture_tee)

    runner.mdrun(
        tpr=tmp_path / 'md.tpr',
        deffnm='md',
        cwd=tmp_path,
        ntomp=4,
        ntmpi=1,
        use_gpu=True,
        gpu_id='0',
    )

    assert len(calls) == 2
    first_cmd, _ = calls[0]
    second_cmd, second_env = calls[1]

    assert first_cmd[first_cmd.index('-nb') + 1] == 'gpu'
    assert first_cmd[first_cmd.index('-bonded') + 1] == 'gpu'
    assert first_cmd[first_cmd.index('-update') + 1] == 'gpu'
    assert first_cmd[first_cmd.index('-pme') + 1] == 'gpu'
    assert first_cmd[first_cmd.index('-pmefft') + 1] == 'gpu'
    assert '-gpu_id' in first_cmd

    assert second_cmd[second_cmd.index('-nb') + 1] == 'cpu'
    assert second_cmd[second_cmd.index('-bonded') + 1] == 'cpu'
    assert second_cmd[second_cmd.index('-update') + 1] == 'cpu'
    assert second_cmd[second_cmd.index('-pme') + 1] == 'cpu'
    assert second_cmd[second_cmd.index('-pmefft') + 1] == 'cpu'
    assert '-gpu_id' not in second_cmd
    assert second_env.get('GMX_DISABLE_GPU_DETECTION') == '1'
