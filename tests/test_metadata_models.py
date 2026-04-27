from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

from yadonpy.core.metadata import (
    BenchmarkSummary,
    ChargeMetadata,
    EquilibrationState,
    RespConstraintMetadata,
)
from yadonpy.schema_versions import (
    BENCHMARK_SUMMARY_SCHEMA_VERSION,
    EQUILIBRIUM_SCHEMA_VERSION,
    METADATA_SCHEMA_VERSION,
    SPECIES_FORCEFIELD_SUMMARY_SCHEMA_VERSION,
)
from yadonpy.sim.benchmarking import _dump_json


def test_resp_constraint_metadata_round_trips_equivalence_groups():
    meta = RespConstraintMetadata.from_dict(
        {
            "mode": "whole_molecule_scale",
            "resp_profile": "Adaptive",
            "equivalence_groups": [[1, 3], [2, 4]],
            "charged_group_constraints": [{"group": "carboxylate", "charge": -1}],
        }
    )

    payload = meta.to_dict()

    assert payload["schema_version"] == METADATA_SCHEMA_VERSION
    assert payload["resp_profile"] == "adaptive"
    assert payload["equivalence_groups"] == [[1, 3], [2, 4]]
    assert payload["charged_group_constraints"][0]["charge"] == -1


def test_charge_metadata_round_trips_through_rdkit_props():
    mol = Chem.MolFromSmiles("COC")
    assert mol is not None
    constraints = RespConstraintMetadata.from_dict(
        {"mode": "whole_molecule_scale", "resp_profile": "adaptive", "equivalence_groups": [[0, 2]]}
    )
    meta = ChargeMetadata(
        charge_model="RESP",
        resp_profile="adaptive",
        qm_recipe={"opt_method": "wb97m-v", "charge_basis": "def2-TZVP"},
        resp_constraints=constraints,
        psiresp_constraints={"charge_equivalence": [[0, 2]]},
        source_kind="repo_moldb",
    )

    meta.apply_to_mol(mol)
    recovered = ChargeMetadata.from_mol(mol)

    assert recovered.resp_profile == "adaptive"
    assert recovered.qm_recipe["opt_method"] == "wb97m-v"
    assert recovered.resp_constraints.equivalence_groups == [[0, 2]]
    assert recovered.psiresp_constraints["charge_equivalence"] == [[0, 2]]


def test_equilibration_state_adds_schema_without_overwriting_payload():
    state = EquilibrationState.from_dict({"ok": False, "severity": "high"})
    payload = state.to_dict()

    assert payload["ok"] is False
    assert payload["severity"] == "high"
    assert payload["schema_version"] == EQUILIBRIUM_SCHEMA_VERSION
    assert payload["summary_kind"] == "equilibration_state"


def test_benchmark_summary_schema_is_selected_from_filename(tmp_path: Path):
    species_payload = BenchmarkSummary.for_path("species_forcefield_summary.json", {"species": []}).to_dict()
    benchmark_payload = BenchmarkSummary.for_path("benchmark_summary.json", {"ok": True}).to_dict()

    assert species_payload["schema_version"] == SPECIES_FORCEFIELD_SUMMARY_SCHEMA_VERSION
    assert species_payload["summary_kind"] == "species_forcefield_summary"
    assert benchmark_payload["schema_version"] == BENCHMARK_SUMMARY_SCHEMA_VERSION
    assert benchmark_payload["summary_kind"] == "benchmark_summary"

    out = _dump_json(tmp_path / "benchmark_summary.json", {"ok": True})
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["schema_version"] == BENCHMARK_SUMMARY_SCHEMA_VERSION
    assert saved["ok"] is True
