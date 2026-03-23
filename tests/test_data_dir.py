from __future__ import annotations

import json
import tarfile
from pathlib import Path

from yadonpy.core import data_dir


def _write_bundle_archive(path: Path, keys: list[str]) -> Path:
    src_root = path.parent / f'{path.stem}_src'
    moldb_root = src_root / 'home' / 'yuzc' / '.yadonpy' / 'moldb' / 'objects'
    for key in keys:
        rec_dir = moldb_root / key
        rec_dir.mkdir(parents=True, exist_ok=True)
        (rec_dir / 'manifest.json').write_text(
            json.dumps(
                {
                    'key': key,
                    'kind': 'smiles',
                    'canonical': key,
                    'name': key,
                    'ready': True,
                    'variants': {},
                },
                indent=2,
            ) + '\n',
            encoding='utf-8',
        )
        (rec_dir / 'best.mol2').write_text('@<TRIPOS>MOLECULE\nstub\n', encoding='utf-8')
    with tarfile.open(path, 'w') as tf:
        tf.add(src_root / 'home', arcname='home')
    return path


def test_find_bundle_archive_prefers_example_sibling_root(tmp_path: Path):
    root = tmp_path / 'repo'
    examples_run = root / 'examples' / '12_case' / 'run.py'
    module_file = root / 'src' / 'yadonpy' / 'core' / 'data_dir.py'
    bundle = root / 'yd_moldb.tar'
    examples_run.parent.mkdir(parents=True, exist_ok=True)
    module_file.parent.mkdir(parents=True, exist_ok=True)
    examples_run.write_text('print("stub")\n', encoding='utf-8')
    module_file.write_text('# stub\n', encoding='utf-8')
    bundle.write_bytes(b'archive')

    found = data_dir.find_bundle_archive(
        cwd=examples_run.parent,
        argv0=examples_run,
        module_file=module_file,
    )

    assert found == bundle.resolve()


def test_ensure_initialized_imports_bundle_archive_from_env(tmp_path: Path, monkeypatch):
    archive = _write_bundle_archive(tmp_path / 'yd_moldb.tar', ['key_a'])
    data_root = tmp_path / 'data_root'

    monkeypatch.setenv('YADONPY_HOME', str(data_root))
    monkeypatch.setenv('YADONPY_MOLDB_ARCHIVE', str(archive))
    monkeypatch.delenv('YADONPY_DATA_DIR', raising=False)

    layout = data_dir.ensure_initialized()

    assert layout.root == data_root.resolve()
    assert (layout.moldb_dir / 'objects' / 'key_a' / 'manifest.json').exists()
    state = json.loads(layout.bundle_state.read_text(encoding='utf-8'))
    assert state['managed_keys'] == ['key_a']
    assert state['archive_name'] == 'yd_moldb.tar'


def test_bundle_update_replaces_previous_managed_records_only(tmp_path: Path, monkeypatch):
    archive_a = _write_bundle_archive(tmp_path / 'bundle_a.tar', ['key_a'])
    archive_b = _write_bundle_archive(tmp_path / 'bundle_b.tar', ['key_b'])
    data_root = tmp_path / 'data_root'

    monkeypatch.setenv('YADONPY_HOME', str(data_root))
    monkeypatch.delenv('YADONPY_DATA_DIR', raising=False)
    monkeypatch.setenv('YADONPY_MOLDB_ARCHIVE', str(archive_a))

    layout = data_dir.ensure_initialized()
    user_dir = layout.moldb_dir / 'objects' / 'user_manual'
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / 'manifest.json').write_text('{"key":"user_manual"}\n', encoding='utf-8')

    monkeypatch.setenv('YADONPY_MOLDB_ARCHIVE', str(archive_b))
    layout = data_dir.ensure_initialized()

    assert not (layout.moldb_dir / 'objects' / 'key_a').exists()
    assert (layout.moldb_dir / 'objects' / 'key_b' / 'manifest.json').exists()
    assert (layout.moldb_dir / 'objects' / 'user_manual' / 'manifest.json').exists()
