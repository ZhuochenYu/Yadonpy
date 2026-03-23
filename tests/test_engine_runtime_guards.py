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
