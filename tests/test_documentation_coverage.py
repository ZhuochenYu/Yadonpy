from __future__ import annotations

import ast
from pathlib import Path


def test_all_source_modules_have_intent_docstrings():
    """Every package source file should explain its role before implementation details."""

    missing: list[str] = []
    for path in sorted((Path(__file__).resolve().parents[1] / "src" / "yadonpy").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if ast.get_docstring(tree) is None:
            missing.append(str(path.relative_to(Path(__file__).resolve().parents[1])))

    assert missing == []
