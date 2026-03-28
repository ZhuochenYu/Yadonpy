from __future__ import annotations

from pathlib import Path

import pytest

from yadonpy.core import data_dir


def test_find_bundle_archive_disabled(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "yd_moldb.tar").write_bytes(b"archive")

    found = data_dir.find_bundle_archive(cwd=root)

    assert found is None


def test_ensure_initialized_creates_plain_layout(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("YADONPY_HOME", str(data_root))
    monkeypatch.delenv("YADONPY_DATA_DIR", raising=False)
    monkeypatch.delenv("YADONPY_MOLDB_ARCHIVE", raising=False)

    layout = data_dir.ensure_initialized()

    assert layout.root == data_root.resolve()
    assert (layout.moldb_dir / "objects").exists()
    assert layout.marker.exists()
    assert not layout.bundle_state.exists()


def test_import_bundle_archive_raises(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("YADONPY_HOME", str(data_root))
    layout = data_dir.ensure_initialized()

    with pytest.raises(RuntimeError, match="removed in v0.8.75"):
        data_dir.import_bundle_archive(layout, tmp_path / "yd_moldb.tar")
