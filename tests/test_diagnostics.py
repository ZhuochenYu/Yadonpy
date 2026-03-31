from __future__ import annotations

import yadonpy.diagnostics as diagnostics


def test_psiresp_hint_reports_pydantic_v2_conflict(monkeypatch):
    monkeypatch.setattr(diagnostics, "_pydantic_version", lambda: "2.11.0")

    hint = diagnostics._psiresp_hint(
        "PydanticUserError(\"A non-annotated attribute was detected: `jobname = 'optimization'`.\")"
    )

    assert "pydantic<2" in hint
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
    assert "pydantic<2" in status.hint
