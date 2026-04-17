from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import csv
from types import SimpleNamespace


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


def _load_example07_parallel_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "07_moldb_precompute_and_reuse"
        / "02_build_moldb_parallel.py"
    )
    spec = importlib.util.spec_from_file_location("example07_build_moldb_parallel", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_example07_ffcheck_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "07_moldb_precompute_and_reuse"
        / "05_check_forcefield_assignment.py"
    )
    spec = importlib.util.spec_from_file_location("example07_ffcheck", path)
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
    assert "Li" not in names
    assert "Na" not in names


def test_example07_catalog_exposes_charge_and_bonded_columns_without_forcefield_column():
    mod = _load_example07_module()
    items = {item.name: item for item in mod._read_species_csv(mod.CATALOG_CSV)}
    with mod.CATALOG_CSV.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

    assert items["PF6"].bonded == "DRIH"
    assert items["ClO4"].bonded == "DRIH"
    assert items["PAA"].polyelectrolyte_mode is True
    assert "ff_name" not in fieldnames


def test_example07_catalog_auto_promotes_localized_carboxylate_species_even_if_csv_flag_is_false(tmp_path):
    mod = _load_example07_module()
    csv_path = tmp_path / "species.csv"
    csv_path.write_text(
        (
            "name,smiles,kind,source,charge,bonded,polyelectrolyte_mode\n"
            "legacy_glucose_6,*OC1OC(COCC(=O)[O-])C(*)C(O)C1O,psmiles,test,RESP,,false\n"
            "legacy_tfsi,FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F,smiles,test,RESP,,false\n"
        ),
        encoding="utf-8",
    )

    items = {item.name: item for item in mod._read_species_csv(csv_path)}

    assert items["legacy_glucose_6"].polyelectrolyte_mode is True
    assert items["legacy_tfsi"].polyelectrolyte_mode is False


def test_example07_removes_legacy_text_table_inputs():
    root = Path(__file__).resolve().parents[1]
    example_dir = root / "examples" / "07_moldb_precompute_and_reuse"

    assert not (example_dir / "02_text_table_to_moldb.py").exists()
    assert not (example_dir / "template.csv").exists()
    assert not (example_dir / "reference_species.csv").exists()
    assert (example_dir / "02_build_moldb_parallel.py").exists()
    assert (example_dir / "05_check_forcefield_assignment.py").exists()


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


def test_example07_qm_policy_uses_neutral_and_anion_routes_only():
    mod = _load_example07_module()
    qm_neutral = mod._resolve_qm_spec("O=C1OCCO1")
    qm_anion = mod._resolve_qm_spec("F[P-](F)(F)(F)(F)F")

    assert qm_neutral is not None
    assert qm_neutral.opt_basis == "def2-SVP"
    assert qm_anion is not None
    assert qm_anion.opt_basis == "def2-SVPD"


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


def test_example07_parallel_planner_assigns_profiles_and_core_budgets():
    build_mod = _load_example07_module()
    parallel_mod = _load_example07_parallel_module()

    species = build_mod._read_species_csv(build_mod.CATALOG_CSV)
    tasks = {task.name: task for task in parallel_mod._build_parallel_tasks(species, cpu_total=34)}

    assert parallel_mod._reserved_driver_cores(36) == 2
    assert parallel_mod._planner_cpu_budget(36) == 34
    assert tasks["PF6"].profile == "drih"
    assert tasks["PF6"].batch == "heavy_qm"
    assert tasks["PF6"].required_cores == 8
    assert tasks["PAA"].profile == "polyelectrolyte"
    assert tasks["PAA"].batch == "charged_polymer_qm"
    assert tasks["PAA"].required_cores == 6
    assert tasks["monomer_A"].profile == "polymer"
    assert tasks["monomer_A"].batch == "polymer_qm"
    assert tasks["monomer_A"].required_cores == 5
    assert tasks["EC"].profile == "standard"
    assert tasks["EC"].batch == "standard_qm"
    assert tasks["EC"].required_cores == 4


def test_example07_parallel_planner_builds_pending_payloads_without_zip_ordering():
    build_mod = _load_example07_module()
    parallel_mod = _load_example07_parallel_module()

    species = build_mod._read_species_csv(build_mod.CATALOG_CSV)
    tasks = parallel_mod._build_parallel_tasks(species, cpu_total=18)
    pending = parallel_mod._build_pending_payloads(species, tasks)

    assert len(pending) == len(species)
    assert {item["name"] for item in pending} == {item.name for item in species}
    assert pending[0]["priority"] <= pending[-1]["priority"]
    assert pending[0]["attempt"] == 1
    assert pending[0]["max_attempts"] == 2


def test_example07_parallel_planner_uses_priority_batches_with_backfill():
    parallel_mod = _load_example07_parallel_module()

    pending = [
        {"name": "standard", "priority": 3, "required_cores": 3},
        {"name": "drih_big", "priority": 0, "required_cores": 8},
        {"name": "drih_small", "priority": 0, "required_cores": 4},
        {"name": "standard_small", "priority": 3, "required_cores": 2},
    ]
    parallel_mod._sort_pending_in_place(pending)

    eligible = parallel_mod._eligible_pending_for_launch(pending, available_cores=4)

    assert [item["name"] for item in eligible] == ["drih_small"]

    eligible_backfill = parallel_mod._eligible_pending_for_launch(pending, available_cores=2)

    assert [item["name"] for item in eligible_backfill] == ["standard_small"]


def test_example07_parallel_planner_retries_once_with_reduced_threads():
    parallel_mod = _load_example07_parallel_module()

    task = {
        "name": "PF6",
        "required_cores": 8,
        "psi4_omp": 8,
        "attempt": 1,
        "max_attempts": 2,
    }
    retry_task = parallel_mod._maybe_schedule_retry(task, error="boom")

    assert retry_task is not None
    assert retry_task["required_cores"] == 4
    assert retry_task["psi4_omp"] == 4
    assert retry_task["attempt"] == 2
    assert retry_task["retry_reason"] == "boom"

    assert parallel_mod._maybe_schedule_retry(retry_task, error="boom again") is None


def test_example07_forcefield_check_groups_are_explicit():
    build_mod = _load_example07_module()
    ffcheck_mod = _load_example07_ffcheck_module()

    items = {item.name: item for item in build_mod._read_species_csv(build_mod.CATALOG_CSV)}

    assert ffcheck_mod.GROUP_ORDER == (
        "neutral_molecules",
        "drih_anions",
        "polyelectrolyte_monomers",
        "monatomic_ions",
    )
    assert [item.name for item in ffcheck_mod.DIRECT_ION_SPECS] == ["Li", "Na"]
    assert ffcheck_mod._catalog_report_group(items["EC"], formal_charge=0) == "neutral_molecules"
    assert ffcheck_mod._catalog_report_group(items["PF6"], formal_charge=-1) == "drih_anions"
    assert ffcheck_mod._catalog_report_group(items["PAA"], formal_charge=-1) == "polyelectrolyte_monomers"


def test_example07_run_one_species_uses_charge_first_then_moldb(monkeypatch, tmp_path):
    mod = _load_example07_module()
    captured: dict[str, object] = {}

    def fake_assign_charges(mol, **kwargs):
        captured["assign_kwargs"] = dict(kwargs)
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("AtomicCharge", float(atom.GetFormalCharge()))
        if kwargs.get("bonded_params") == "drih":
            mol.SetProp("_yadonpy_bonded_method", "DRIH")
            mol.SetProp("_yadonpy_bonded_requested", "drih")
            mol.SetProp("_yadonpy_bonded_signature", "drih")
        return True

    class FakeMolDB:
        def __init__(self, db_dir):
            captured["db_dir"] = Path(db_dir)

        def update_from_mol(self, mol, **kwargs):
            captured["update_kwargs"] = dict(kwargs)
            captured["bonded_method"] = mol.GetProp("_yadonpy_bonded_method") if mol.HasProp("_yadonpy_bonded_method") else None
            return SimpleNamespace(key="fake-key")

    monkeypatch.setattr(mod.yp, "assign_charges", fake_assign_charges)
    monkeypatch.setattr(mod, "MolDB", FakeMolDB)

    spec = mod.SpeciesSpec(
        name="BF4",
        smiles="F[B-](F)(F)F",
        kind="smiles",
        source="test",
        charge="RESP",
        bonded="DRIH",
        polyelectrolyte_mode=False,
    )

    result = mod.run_one_species(
        spec,
        db_dir=tmp_path / "moldb",
        job_wd=tmp_path / "work",
        psi4_omp=4,
        psi4_memory_mb=4000,
    )

    assert captured["assign_kwargs"]["bonded_params"] == "drih"
    assert captured["assign_kwargs"]["log_name"] == "BF4"
    assert captured["update_kwargs"]["charge"] == "RESP"
    assert captured["bonded_method"] == "DRIH"
    assert result["record_key"] == "fake-key"
    assert result["bonded_mode"] == "drih"
