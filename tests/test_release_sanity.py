from __future__ import annotations

from pathlib import Path
import re

import yadonpy


def test_release_version_is_consistent_in_core_files():
    root = Path(__file__).resolve().parents[1]
    version = yadonpy.__version__

    assert f'Current release: **v{version}**' in (root / 'README.md').read_text(encoding='utf-8')
    assert f'# YadonPy API (v{version})' in (root / 'docs' / f'Yadonpy_API_v{version}.md').read_text(encoding='utf-8')
    assert f'# YadonPy Manual (v{version})' in (root / 'docs' / 'Yadonpy_manul.md').read_text(encoding='utf-8')
    assert f'# YadonPy User Guide (v{version})' in (root / 'docs' / 'Yaonpyd_user_guide.md').read_text(encoding='utf-8')
    assert f'version = "{version}"' in (root / 'pyproject.toml').read_text(encoding='utf-8')


def test_release_declares_python_311_minimum_and_docs_match():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / 'pyproject.toml').read_text(encoding='utf-8')
    readme = (root / 'README.md').read_text(encoding='utf-8')
    manual = (root / 'docs' / 'Yadonpy_manul.md').read_text(encoding='utf-8')
    guide = (root / 'docs' / 'Yaonpyd_user_guide.md').read_text(encoding='utf-8')

    assert 'requires-python = ">=3.11"' in pyproject
    assert 'Python 3.11+' in readme
    assert 'Python 3.11+' in manual
    assert 'Python 3.11+' in guide



def test_release_manifest_excludes_cached_and_temp_artifacts():
    root = Path(__file__).resolve().parents[1]
    manifest = (root / 'MANIFEST.in').read_text(encoding='utf-8')

    assert 'prune .pytest_cache' in manifest
    assert 'prune .yadonpy_cache' in manifest
    assert 'prune src/yadonpy.egg-info' in manifest
    assert 'prune tmp_workdir_smoke' in manifest
    assert 'global-exclude __pycache__' in manifest


def test_examples_do_not_fallback_to_local_src_injection():
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in sorted((root / 'examples').rglob('*.py')):
        text = path.read_text(encoding='utf-8')
        if '_SRC = (Path(__file__).resolve().parent / ".." / ".." / "src").resolve()' in text:
            offenders.append(str(path.relative_to(root)))
            continue
        if 'sys.path.append(str(_SRC))' in text or 'sys.path.insert(0, str(_SRC))' in text:
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_examples_do_not_rebuild_pf6_outside_example01():
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in sorted((root / 'examples').rglob('*.py')):
        rel = path.relative_to(root).as_posix()
        if rel.startswith('examples/01_Li_salt/'):
            continue
        text = path.read_text(encoding='utf-8')
        if 'utils.ensure_3d_coords(PF6' in text:
            offenders.append(rel)
            continue
        if 'utils.ensure_3d_coords(anion_A, smiles_hint=anion_smiles_A' in text:
            offenders.append(rel)
            continue
        if 'qm.assign_charges(PF6' in text:
            offenders.append(rel)
            continue
        if 'anion_A = utils.mol_from_smiles(anion_smiles_A, coord=False)' in text:
            offenders.append(rel)

    assert offenders == []


def test_examples_do_not_wrap_pf6_moldb_loading_in_helpers():
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in sorted((root / 'examples').rglob('*.py')):
        text = path.read_text(encoding='utf-8')
        if 'def load_pf6_from_moldb' in text or 'def prepare_pf6(' in text:
            offenders.append(str(path.relative_to(root)))

    assert offenders == []


def test_interface_examples_keep_linear_script_style():
    root = Path(__file__).resolve().parents[1]
    helper_patterns = (
        'def _named(',
        'def _resolved(',
        'def assign_template_species(',
        'def prepare_template_species(',
        'def build_cmc(',
    )
    offenders: list[str] = []
    for rel in (
        'examples/10_interface_route_a/run_interface_route_a.py',
        'examples/11_interface_route_b/run_interface_route_b.py',
        'examples/12_cmcna_interface/run_cmcna_interface.py',
    ):
        text = (root / rel).read_text(encoding='utf-8')
        if any(pattern in text for pattern in helper_patterns):
            offenders.append(rel)

    assert offenders == []


def test_examples_do_not_manually_set_names_or_pass_name_into_ff_mol():
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    ff_mol_name = re.compile(r"\b(?:ff|ion_ff|cation_ff)\.mol\([\s\S]{0,160}?name\s*=", re.MULTILINE)
    manual_setprop = re.compile(r"\.SetProp\(\s*['\"](?:_Name|name|_yadonpy_name|_yadonpy_resname)['\"]")

    for path in sorted((root / 'examples').rglob('*.py')):
        text = path.read_text(encoding='utf-8')
        if ff_mol_name.search(text) or manual_setprop.search(text):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []
