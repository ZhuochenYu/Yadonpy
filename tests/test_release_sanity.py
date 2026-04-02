from __future__ import annotations

from pathlib import Path
import re

import yadonpy


def test_release_metadata_is_consistent_in_core_files():
    root = Path(__file__).resolve().parents[1]
    version = yadonpy.__version__
    readme = (root / 'README.md').read_text(encoding='utf-8')
    api = (root / 'docs' / 'API_REFERENCE.md').read_text(encoding='utf-8')
    guide = (root / 'docs' / 'USER_GUIDE.md').read_text(encoding='utf-8')
    architecture = (root / 'docs' / 'ARCHITECTURE.md').read_text(encoding='utf-8')
    technical = (root / 'docs' / 'TECHNICAL_NOTES.md').read_text(encoding='utf-8')

    assert '# YadonPy' in readme
    assert '# YadonPy API Reference' in api
    assert '# YadonPy User Guide' in guide
    assert '# YadonPy Architecture' in architecture
    assert '# YadonPy Technical Notes' in technical
    assert 'Current release:' not in readme
    assert f'version = "{version}"' in (root / 'pyproject.toml').read_text(encoding='utf-8')


def test_release_declares_python_311_minimum_and_docs_match():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / 'pyproject.toml').read_text(encoding='utf-8')
    readme = (root / 'README.md').read_text(encoding='utf-8')
    guide = (root / 'docs' / 'USER_GUIDE.md').read_text(encoding='utf-8')
    api = (root / 'docs' / 'API_REFERENCE.md').read_text(encoding='utf-8')

    assert 'requires-python = ">=3.11"' in pyproject
    assert 'python=3.11' in readme
    assert 'python=3.11' in guide
    assert 'psiresp-base' in readme
    assert 'psiresp-base' in guide
    assert 'psi4=1.10' in readme
    assert 'psi4=1.10' in guide
    assert 'python -m pip install "pydantic==1.10.26"' in readme
    assert 'python -m pip install "pydantic==1.10.26"' in guide
    assert '~/.yadonpy/moldb' in readme
    assert '~/.yadonpy/moldb' in guide
    assert 'pydantic<2' not in readme
    assert 'pip install pybel' not in readme
    assert 'pip install pybel' not in guide
    assert 'psiresp-base' in api


def test_repo_uses_openbabel_bindings_and_not_standalone_pybel_package():
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in sorted(root.rglob('*.py')):
        if '.git' in path.parts or '__pycache__' in path.parts:
            continue
        if path.name == 'test_release_sanity.py':
            continue
        text = path.read_text(encoding='utf-8')
        if 'from pybel import' in text:
            offenders.append(str(path.relative_to(root)))
            continue
        if 'import pybel' in text and 'from openbabel import pybel' not in text:
            offenders.append(str(path.relative_to(root)))

    assert offenders == []
    chem_utils = (root / 'src' / 'yadonpy' / 'core' / 'chem_utils.py').read_text(encoding='utf-8')
    assert 'from openbabel import pybel as openbabel_pybel' in chem_utils



def test_repo_excludes_local_only_release_files_from_git():
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / '.gitignore').read_text(encoding='utf-8')

    assert not (root / 'MANIFEST.in').exists()
    assert not (root / 'YADONPY_MAINTENANCE_PRINCIPLES.md').exists()
    assert 'MANIFEST.in' in gitignore
    assert 'YADONPY_MAINTENANCE_PRINCIPLES.md' in gitignore
    assert not (root / 'moldb.tar').exists()
    assert (root / 'moldb' / 'objects').is_dir()
    assert list(root.rglob('*.tar')) == []


def test_docs_have_single_current_structure_and_old_release_docs_are_removed():
    root = Path(__file__).resolve().parents[1]
    docs = root / 'docs'

    assert (docs / 'API_REFERENCE.md').is_file()
    assert (docs / 'USER_GUIDE.md').is_file()
    assert (docs / 'ARCHITECTURE.md').is_file()
    assert (docs / 'TECHNICAL_NOTES.md').is_file()

    old = (
        'Yadonpy_manul.md',
        'Yaonpyd_user_guide.md',
        'oplsaa2024_moltemplate_import.md',
        'si_h_qm_probe_20260325.md',
        'si_h_qm_probe_20260325_summary.json',
        'si_h_qm_probe_20260325_typed_summary.json',
        'Yadonpy_API_v0.8.69.md',
        'Yadonpy_API_v0.8.70.md',
        'Yadonpy_API_v0.8.71.md',
        'Yadonpy_API_v0.8.72.md',
        'Yadonpy_API_v0.8.73.md',
        'Yadonpy_API_v0.8.75.md',
        'Yadonpy_API_v0.8.76.md',
    )
    for name in old:
        assert not (docs / name).exists()


def test_oplsaa_rule_table_is_externalized():
    root = Path(__file__).resolve().parents[1]
    opls_py = (root / 'src' / 'yadonpy' / 'ff' / 'oplsaa.py').read_text(encoding='utf-8')
    opls_rules = root / 'src' / 'yadonpy' / 'ff' / 'ff_dat' / 'oplsaa_rules.json'

    assert opls_rules.is_file()
    assert 'RULES = [' not in opls_py
    assert 'oplsaa_rules.json' in opls_py


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
        'examples/08_graphite_polymer_electrolyte_sandwich/01_peo_smoke.py',
        'examples/08_graphite_polymer_electrolyte_sandwich/02_peo_carbonate_full.py',
        'examples/08_graphite_polymer_electrolyte_sandwich/03_cmcna_smoke.py',
        'examples/08_graphite_polymer_electrolyte_sandwich/04_cmcna_full.py',
        'examples/08_graphite_polymer_electrolyte_sandwich/05_cmcna_glucose6_periodic_case.py',
    ):
        text = (root / rel).read_text(encoding='utf-8')
        if any(pattern in text for pattern in helper_patterns):
            offenders.append(rel)

    assert offenders == []


def test_example05_cmcna_periodic_case_is_moldb_only_for_core_species():
    root = Path(__file__).resolve().parents[1]
    text = (
        root / 'examples' / '08_graphite_polymer_electrolyte_sandwich' / '05_cmcna_glucose6_periodic_case.py'
    ).read_text(encoding='utf-8')

    assert 'name="glucose_6"' in text
    assert 'name="EC"' in text
    assert 'name="EMC"' in text
    assert 'name="DEC"' in text
    assert 'name="PF6"' in text
    assert text.count('prefer_db=True') >= 5
    assert text.count('require_ready=True') >= 5


def test_oplsaa_examples_use_script_first_yadonpy_style():
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for rel in (
        'examples/09_oplsaa_assignment/01_oplsaa_ec.py',
        'examples/09_oplsaa_assignment/02_oplsaa_moldb_and_ion.py',
    ):
        text = (root / rel).read_text(encoding='utf-8')
        if 'from rdkit' in text or 'import rdkit' in text or 'Chem.' in text:
            offenders.append(rel)
            continue
        if 'def _' in text:
            offenders.append(rel)

    assert offenders == []


def test_release_docs_do_not_reference_retired_example_paths():
    root = Path(__file__).resolve().parents[1]
    readme = (root / 'README.md').read_text(encoding='utf-8')
    guide = (root / 'docs' / 'USER_GUIDE.md').read_text(encoding='utf-8')

    retired = (
        'examples/08_text_to_csv_and_build_moldb',
        'examples/08_oplsaa_assign',
        'examples/10_interface_route_a',
        'examples/11_interface_route_b',
        'examples/12_cmcna_interface',
        'examples/13_graphite_cmc_electrolyte',
    )

    for rel in retired:
        assert rel not in readme
        assert rel not in guide


def test_docs_reference_current_doc_set_and_not_retired_doc_names():
    root = Path(__file__).resolve().parents[1]
    readme = (root / 'README.md').read_text(encoding='utf-8')
    api = (root / 'docs' / 'API_REFERENCE.md').read_text(encoding='utf-8')
    guide = (root / 'docs' / 'USER_GUIDE.md').read_text(encoding='utf-8')
    architecture = (root / 'docs' / 'ARCHITECTURE.md').read_text(encoding='utf-8')
    technical = (root / 'docs' / 'TECHNICAL_NOTES.md').read_text(encoding='utf-8')

    for text in (readme, api, guide, architecture, technical):
        assert 'Yadonpy_manul.md' not in text
        assert 'Yaonpyd_user_guide.md' not in text
        assert 'Yadonpy_API_v' not in text

    assert 'docs/USER_GUIDE.md' in readme
    assert 'docs/API_REFERENCE.md' in readme
    assert 'docs/ARCHITECTURE.md' in readme
    assert 'docs/TECHNICAL_NOTES.md' in readme


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
