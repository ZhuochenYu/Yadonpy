from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_example07_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "07_moldb_precompute_and_reuse"
        / "01_build_moldb.py"
    )
    spec = importlib.util.spec_from_file_location("example07_build_moldb", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example07_catalog_includes_new_polymer_and_salt_entries():
    mod = _load_example07_module()
    items = mod._read_species_csv(mod.CATALOG_CSV)
    names = {item.name for item in items}

    assert "SbF6" in names
    assert "BOB" in names
    assert "DFOB" in names
    assert "NO3" in names
    assert "OTf" in names
    assert "PAA" in names
    assert "PVDF" in names
    assert "PTMC" in names


def test_example07_catalog_exposes_bonded_and_forcefield_columns():
    mod = _load_example07_module()
    items = {item.name: item for item in mod._read_species_csv(mod.CATALOG_CSV)}

    assert items["PF6"].bonded == "DRIH"
    assert items["ClO4"].bonded == "DRIH"
    assert items["Li"].ff_name == "merz"
    assert items["PAA"].polyelectrolyte_mode is True


def test_example07_removes_legacy_text_table_inputs():
    root = Path(__file__).resolve().parents[1]
    example_dir = root / "examples" / "07_moldb_precompute_and_reuse"

    assert not (example_dir / "02_text_table_to_moldb.py").exists()
    assert not (example_dir / "template.csv").exists()
    assert not (example_dir / "reference_species.csv").exists()


def test_example07_qm_policy_prefers_diffuse_def2_for_asf6(monkeypatch):
    mod = _load_example07_module()
    seen = []

    def fake_pick(candidates, elements=None):
        seen.append((tuple(candidates), tuple(elements or ())))
        return candidates[0]

    monkeypatch.setattr(mod, "_pick_first_available_basis", fake_pick)

    qm_spec = mod._resolve_qm_spec("F[As-](F)(F)(F)(F)F")

    assert qm_spec is not None
    assert qm_spec.opt_basis == "def2-SVPD"
    assert qm_spec.charge_basis == "def2-TZVPD"
    assert qm_spec.reason == "anion diffuse-first"
    assert seen == [
        (("def2-SVPD", "def2-SVP"), ("F", "As")),
        (("def2-TZVPD", "def2-TZVPPD", "def2-TZVP"), ("F", "As")),
    ]


def test_example07_qm_policy_marks_fallback_when_diffuse_basis_is_unavailable(monkeypatch):
    mod = _load_example07_module()

    def fake_pick(candidates, elements=None):
        return candidates[-1]

    monkeypatch.setattr(mod, "_pick_first_available_basis", fake_pick)

    qm_spec = mod._resolve_qm_spec("F[Sb-](F)(F)(F)(F)F")

    assert qm_spec is not None
    assert qm_spec.opt_basis == "def2-SVP"
    assert qm_spec.charge_basis == "def2-TZVP"
    assert qm_spec.reason.endswith("fallback")


def test_example07_qm_policy_skips_monatomic_merz_path():
    mod = _load_example07_module()
    assert mod._resolve_qm_spec("[Li+]") is None


def test_example07_qm_policy_uses_diffuse_route_for_bob_family():
    mod = _load_example07_module()

    qm_bob = mod._resolve_qm_spec("O=C1O[B-]2(OC1=O)OC(=O)C(=O)O2")
    qm_dfob = mod._resolve_qm_spec("O=C1C(=O)O[B-](F)(F)O1")

    assert qm_bob is not None
    assert qm_dfob is not None
    assert qm_bob.opt_basis == "def2-SVPD"
    assert qm_bob.charge_basis == "def2-TZVPD"
    assert qm_dfob.opt_basis == "def2-SVPD"
    assert qm_dfob.charge_basis == "def2-TZVPD"
