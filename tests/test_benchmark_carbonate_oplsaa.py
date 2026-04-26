from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "02_polymer_electrolyte"
    / "benchmark_carbonate_lipf6_oplsaa.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_carbonate_lipf6_oplsaa", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_oplsaa_benchmark_prefers_repo_moldb_for_reproducible_solvent_charges(tmp_path: Path):
    mod = _load_module()
    repo_db = tmp_path / "repo_moldb"

    class FakeFF:
        name = "fake-opls"

        def __init__(self):
            self.calls: list[Path | None] = []

        def mol_rdkit(self, smiles, *, name, db_dir, charge, require_ready, prefer_db):
            self.calls.append(db_dir)
            return SimpleNamespace(smiles=smiles, name=name, db_dir=db_dir, charge=charge)

        def ff_assign(self, mol, *, charge=None, report=False):
            return mol

    ff = FakeFF()

    mol = mod._load_ready_opls_species(ff, "O=C1OCCO1", label="EC", repo_db_dir=repo_db, charge_mode="resp")

    assert mol.db_dir == repo_db
    assert ff.calls == [repo_db]
