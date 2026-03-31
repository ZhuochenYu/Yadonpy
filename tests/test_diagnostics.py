from __future__ import annotations

import yadonpy
import yadonpy.diagnostics as diagnostics
from yadonpy.core import logging_utils


def test_psiresp_hint_reports_pydantic_v2_conflict(monkeypatch):
    monkeypatch.setattr(diagnostics, "_pydantic_version", lambda: "2.11.0")

    hint = diagnostics._psiresp_hint(
        "PydanticUserError(\"A non-annotated attribute was detected: `jobname = 'optimization'`.\")"
    )

    assert 'pydantic==1.10.26' in hint
    assert "Known incompatibility" in hint
    assert "2.11.0" in hint


def test_check_psiresp_module_surfaces_dynamic_hint(monkeypatch):
    monkeypatch.setattr(diagnostics, "find_spec", lambda mod: object() if mod == "psiresp" else None)
    monkeypatch.setattr(
        diagnostics,
        "_try_import",
        lambda mod: (
            False,
            "PydanticUserError(\"A non-annotated attribute was detected: `jobname = 'optimization'`.\")",
        ),
    )
    monkeypatch.setattr(diagnostics, "_try_import_version", lambda mod, attr="__version__": "2.8.2" if mod == "pydantic" else None)

    status = diagnostics.check_psiresp_module()

    assert status.installed is True
    assert status.import_ok is False
    assert status.hint is not None
    assert 'pydantic==1.10.26' in status.hint


def test_emit_version_banner_only_prints_once(monkeypatch, capsys):
    monkeypatch.setattr(logging_utils, "_version_banner_printed", False)

    logging_utils.emit_version_banner()
    logging_utils.emit_version_banner()

    captured = capsys.readouterr().out
    assert captured.count("version=") == 1
    assert yadonpy.__version__ in captured


def test_doctor_prints_current_version(monkeypatch, capsys):
    monkeypatch.setattr(logging_utils, "_version_banner_printed", False)
    monkeypatch.setattr(diagnostics, "check_python_module", lambda *args, **kwargs: diagnostics.DepStatus(name="x", installed=False))
    monkeypatch.setattr(
        diagnostics,
        "check_psiresp_module",
        lambda: diagnostics.DepStatus(name="psiresp", installed=False, hint="mock"),
    )

    diagnostics.doctor(print_report=True)

    captured = capsys.readouterr().out
    assert f"version={yadonpy.__version__}" in captured
    assert f"[yadonpy] doctor report (v{yadonpy.__version__})" in captured
