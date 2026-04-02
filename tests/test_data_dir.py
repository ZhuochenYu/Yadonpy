from __future__ import annotations

import json
from pathlib import Path

from yadonpy.core import data_dir


def _make_bundle(root: Path, *, value: str = "seed") -> Path:
    bundle = root / "moldb"
    obj = bundle / "objects" / "abc123"
    obj.mkdir(parents=True, exist_ok=True)
    (obj / "manifest.json").write_text(json.dumps({"name": value}), encoding="utf-8")
    (obj / "charges.json").write_text(json.dumps({"total_charge": 0}), encoding="utf-8")
    return bundle


def test_find_bundle_dir_prefers_repo_style_moldb_folder(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    bundle = _make_bundle(root)

    found = data_dir.find_bundle_dir(cwd=root)

    assert found == bundle.resolve()
    assert data_dir.find_bundle_archive(cwd=root) == bundle.resolve()


def test_ensure_initialized_seeds_default_moldb(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data_root"
    bundle = _make_bundle(tmp_path / "seed_repo", value="default")
    monkeypatch.setenv("YADONPY_HOME", str(data_root))
    monkeypatch.setenv("YADONPY_DEFAULT_MOLDB", str(bundle))
    monkeypatch.delenv("YADONPY_DATA_DIR", raising=False)

    layout = data_dir.ensure_initialized()

    manifest = layout.moldb_dir / "objects" / "abc123" / "manifest.json"
    state = json.loads(layout.bundle_state.read_text(encoding="utf-8"))

    assert layout.root == data_root.resolve()
    assert (layout.moldb_dir / "objects").exists()
    assert layout.marker.exists()
    assert manifest.exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["name"] == "default"
    assert state["file_count"] == 2
    assert state["copied"] == 2
    assert state["updated"] == 0


def test_ensure_initialized_only_updates_managed_bundle_files(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data_root"
    bundle = _make_bundle(tmp_path / "seed_repo", value="v1")
    monkeypatch.setenv("YADONPY_HOME", str(data_root))
    monkeypatch.setenv("YADONPY_DEFAULT_MOLDB", str(bundle))

    layout = data_dir.ensure_initialized()
    manifest = layout.moldb_dir / "objects" / "abc123" / "manifest.json"
    user_file = layout.moldb_dir / "objects" / "abc123" / "user_note.txt"
    user_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.write_text("keep me", encoding="utf-8")

    (bundle / "objects" / "abc123" / "manifest.json").write_text(
        json.dumps({"name": "v2"}),
        encoding="utf-8",
    )

    layout = data_dir.ensure_initialized()
    state = json.loads(layout.bundle_state.read_text(encoding="utf-8"))

    assert json.loads(manifest.read_text(encoding="utf-8"))["name"] == "v2"
    assert user_file.read_text(encoding="utf-8") == "keep me"
    assert state["updated"] == 1
    assert state["skipped_existing_user_files"] == 0


def test_import_bundle_archive_requires_directory(tmp_path: Path):
    layout = data_dir.DataLayout(tmp_path / "data_root")
    layout.root.mkdir(parents=True, exist_ok=True)
    (layout.moldb_dir / "objects").mkdir(parents=True, exist_ok=True)

    archive = tmp_path / "default_moldb.tar"
    archive.write_bytes(b"tar")

    try:
        data_dir.import_bundle_archive(layout, archive)
    except RuntimeError as exc:
        assert "directory-based" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for archive input")
