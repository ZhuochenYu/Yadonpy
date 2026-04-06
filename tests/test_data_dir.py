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


def test_ensure_initialized_refreshes_stale_unmanaged_ready_record(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data_root"
    bundle = tmp_path / "seed_repo" / "moldb"
    bundle_obj = bundle / "objects" / "abc123"
    bundle_obj.mkdir(parents=True, exist_ok=True)
    (bundle_obj / "manifest.json").write_text(
        json.dumps(
            {
                "key": "abc123",
                "name": "ready-record",
                "ready": True,
                "variants": {
                    "resp_default": {
                        "variant_id": "resp_default",
                        "ready": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_obj / "charges.json").write_text(
        json.dumps({"variant_id": "resp_default", "total_charge": -1.0}),
        encoding="utf-8",
    )

    monkeypatch.setenv("YADONPY_HOME", str(data_root))
    monkeypatch.setenv("YADONPY_DEFAULT_MOLDB", str(bundle))

    layout = data_dir.ensure_initialized()
    assert (layout.moldb_dir / "objects" / "abc123" / "manifest.json").exists()

    stale_obj = layout.moldb_dir / "objects" / "abc123"
    (stale_obj / "manifest.json").write_text(
        json.dumps(
            {
                "key": "abc123",
                "name": "stale-record",
                "ready": False,
                "variants": {
                    "resp_default": {
                        "variant_id": "resp_default",
                        "ready": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (stale_obj / "charges.json").write_text(
        json.dumps({"variant_id": "resp_default", "total_charge": 0.0}),
        encoding="utf-8",
    )
    user_note = stale_obj / "user_note.txt"
    user_note.write_text("keep me", encoding="utf-8")
    layout.bundle_state.unlink()

    layout = data_dir.ensure_initialized()
    state = json.loads(layout.bundle_state.read_text(encoding="utf-8"))
    manifest = json.loads((stale_obj / "manifest.json").read_text(encoding="utf-8"))
    charges = json.loads((stale_obj / "charges.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "ready-record"
    assert manifest["ready"] is True
    assert manifest["variants"]["resp_default"]["ready"] is True
    assert charges["total_charge"] == -1.0
    assert user_note.read_text(encoding="utf-8") == "keep me"
    assert state["refreshed_stale_records"] == ["objects/abc123"]
    assert state["updated"] == 2


def test_ensure_initialized_refreshes_when_bundle_has_more_complete_variants(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data_root"
    bundle = tmp_path / "seed_repo" / "moldb"
    bundle_obj = bundle / "objects" / "abc123"
    bundle_obj.mkdir(parents=True, exist_ok=True)
    (bundle_obj / "manifest.json").write_text(
        json.dumps(
            {
                "key": "abc123",
                "name": "variant-rich",
                "ready": True,
                "variants": {
                    "resp_default": {
                        "variant_id": "resp_default",
                        "ready": True,
                    },
                    "resp_polyelec": {
                        "variant_id": "resp_polyelec",
                        "ready": True,
                        "bonded": {"mode": "DRIH"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_obj / "charges.json").write_text(json.dumps({"variant_id": "resp_default"}), encoding="utf-8")
    monkeypatch.setenv("YADONPY_HOME", str(data_root))
    monkeypatch.setenv("YADONPY_DEFAULT_MOLDB", str(bundle))

    layout = data_dir.ensure_initialized()
    stale_obj = layout.moldb_dir / "objects" / "abc123"
    (stale_obj / "manifest.json").write_text(
        json.dumps(
            {
                "key": "abc123",
                "name": "stale",
                "ready": True,
                "variants": {
                    "resp_default": {
                        "variant_id": "resp_default",
                        "ready": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    layout.bundle_state.unlink()

    layout = data_dir.ensure_initialized()
    state = json.loads(layout.bundle_state.read_text(encoding="utf-8"))
    manifest = json.loads((stale_obj / "manifest.json").read_text(encoding="utf-8"))

    assert "resp_polyelec" in manifest["variants"]
    assert manifest["variants"]["resp_polyelec"]["bonded"]["mode"] == "DRIH"
    assert state["refreshed_stale_records"] == ["objects/abc123"]
    assert state["audit"]["bundled_more_complete_records"] == []


def test_audit_bundle_sync_reports_missing_stale_and_user_only_records(tmp_path: Path):
    layout = data_dir.DataLayout(tmp_path / "data_root")
    layout.root.mkdir(parents=True, exist_ok=True)
    (layout.moldb_dir / "objects").mkdir(parents=True, exist_ok=True)

    bundle = tmp_path / "seed_repo" / "moldb"
    bundle_a = bundle / "objects" / "abc123"
    bundle_a.mkdir(parents=True, exist_ok=True)
    (bundle_a / "manifest.json").write_text(
        json.dumps(
            {
                "key": "abc123",
                "variants": {
                    "resp_default": {"variant_id": "resp_default", "ready": True},
                    "resp_patch": {"variant_id": "resp_patch", "ready": True, "bonded": {"mode": "DRIH"}},
                },
            }
        ),
        encoding="utf-8",
    )
    bundle_missing = bundle / "objects" / "missing1"
    bundle_missing.mkdir(parents=True, exist_ok=True)
    (bundle_missing / "manifest.json").write_text(json.dumps({"key": "missing1"}), encoding="utf-8")

    user_a = layout.moldb_dir / "objects" / "abc123"
    user_a.mkdir(parents=True, exist_ok=True)
    (user_a / "manifest.json").write_text(
        json.dumps(
            {
                "key": "abc123",
                "variants": {
                    "resp_default": {"variant_id": "resp_default", "ready": True},
                },
            }
        ),
        encoding="utf-8",
    )
    user_only = layout.moldb_dir / "objects" / "useronly"
    user_only.mkdir(parents=True, exist_ok=True)
    (user_only / "manifest.json").write_text(json.dumps({"key": "useronly"}), encoding="utf-8")

    audit = data_dir.audit_bundle_sync(layout, bundle)

    assert audit["missing_objects"] == ["objects/missing1"]
    assert audit["user_only_records"] == ["objects/useronly"]
    assert audit["bundled_more_complete_records"] == ["objects/abc123"]
    assert audit["stale_variants"]["objects/abc123"]["bundle_only_variant_ids"] == ["resp_patch"]
    assert audit["stale_variants"]["objects/abc123"]["bundle_only_bonded_variant_ids"] == ["resp_patch"]


def test_audit_active_bundle_sync_wraps_layout_and_bundle_metadata(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data_root"
    bundle = _make_bundle(tmp_path / "seed_repo", value="default")
    monkeypatch.setenv("YADONPY_HOME", str(data_root))
    monkeypatch.setenv("YADONPY_DEFAULT_MOLDB", str(bundle))

    audit = data_dir.audit_active_bundle_sync()

    assert audit["layout_root"] == str(data_root.resolve())
    assert audit["moldb_dir"] == str((data_root / "moldb").resolve())
    assert audit["bundle_dir"] == str(bundle.resolve())
    assert audit["missing_objects"] == []
    assert audit["user_only_records"] == []
