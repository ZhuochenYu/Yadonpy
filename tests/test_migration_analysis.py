from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
from rdkit import Chem

from yadonpy.gmx.topology import MoleculeType, SystemTopology
from yadonpy.sim.analyzer import AnalyzeResult
from yadonpy.sim.migration import (
    AnchorSpec,
    StateSpec,
    _build_site_state_trajectory,
    _normalize_transition_counts,
    _predict_event_counts,
    _transition_counts,
    infer_role_moltypes,
    run_migration_analysis,
    summarize_role_residence,
)


def _identity_resolver(obj):
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, set)):
        return [str(x) for x in obj]
    return [str(obj)]


def _make_anchor(
    *,
    anchor_id: str,
    role: str,
    moltype: str,
    site_id: str,
    chain_key: str,
    residue_number: int | None = None,
) -> AnchorSpec:
    return AnchorSpec(
        anchor_id=anchor_id,
        role=role,
        moltype=moltype,
        site_id=site_id,
        site_label=site_id.split(":")[-1],
        atom_indices_0=np.asarray([0], dtype=int),
        instance_index=0,
        residue_number=residue_number,
        residue_name="RES" if residue_number is not None else None,
        chain_key=chain_key,
        coordination_priority=1,
        coordination_relevance="primary",
        coordination_note="test",
        anchor_label=anchor_id,
        cutoff_nm=0.33,
    )


def test_infer_role_moltypes_supports_pure_electrolyte():
    catalog = {
        "Li": {"kind": "cation", "formal_charge_e": 1.0},
        "EC": {"kind": "solvent", "formal_charge_e": 0.0},
        "TFSI": {"kind": "anion", "formal_charge_e": -1.0},
    }
    roles = infer_role_moltypes(
        catalog=catalog,
        center_moltype="Li",
        resolve_moltypes=_identity_resolver,
    )
    assert roles["polymer"] == []
    assert roles["solvent"] == ["EC"]
    assert roles["anion"] == ["TFSI"]


def test_infer_role_moltypes_supports_polymer_and_composite():
    catalog = {
        "Li": {"kind": "cation", "formal_charge_e": 1.0},
        "PEO": {"kind": "polymer", "formal_charge_e": 0.0},
        "EC": {"kind": "solvent", "formal_charge_e": 0.0},
        "PF6": {"kind": "anion", "formal_charge_e": -1.0},
    }
    roles = infer_role_moltypes(
        catalog=catalog,
        center_moltype="Li",
        resolve_moltypes=_identity_resolver,
    )
    assert roles["polymer"] == ["PEO"]
    assert roles["solvent"] == ["EC"]
    assert roles["anion"] == ["PF6"]


def test_summarize_role_residence_handles_unavailable(tmp_path: Path):
    rec = summarize_role_residence(
        role="solvent",
        time_ps=np.linspace(0.0, 10.0, 6),
        contact_matrix=None,
        out_dir=tmp_path,
        available=False,
    )
    assert rec["available"] is False
    assert "not present" in str(rec["note"])


class _FakeTraj:
    def __init__(self, xyz: np.ndarray, time_ps: np.ndarray, box_nm: np.ndarray):
        self.xyz = np.asarray(xyz, dtype=float)
        self.time = np.asarray(time_ps, dtype=float)
        self.unitcell_lengths = np.asarray(box_nm, dtype=float)
        self.n_frames = int(self.xyz.shape[0])

    def __getitem__(self, item):
        return _FakeTraj(self.xyz[item], self.time[item], self.unitcell_lengths[item])


def _write_migration_system_files(tmp_path: Path) -> Path:
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "system_meta.json").write_text(
        json.dumps(
            {
                "species": [
                    {"moltype": "Li", "kind": "cation", "smiles": "[Li+]", "formal_charge": 1.0},
                    {"moltype": "EC", "kind": "solvent", "smiles": "O=C1OCCO1", "formal_charge": 0.0},
                    {
                        "moltype": "TFSI",
                        "kind": "anion",
                        "smiles": "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",
                        "formal_charge": -1.0,
                    },
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (system_dir / "residue_map.json").write_text(json.dumps({"species": []}, indent=2) + "\n", encoding="utf-8")
    (system_dir / "charge_groups.json").write_text(
        json.dumps(
            {
                "species": [
                    {
                        "moltype": "TFSI",
                        "charge_groups": [
                            {"group_id": "tfsi", "label": "tfsi", "formal_charge": -1.0, "atom_indices": [0, 1, 2]}
                        ],
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return system_dir


def _toy_topology() -> SystemTopology:
    return SystemTopology(
        moleculetypes={
            "Li": MoleculeType(
                name="Li",
                atomtypes=["li"],
                atomnames=["Li1"],
                charges=[1.0],
                masses=[6.94],
                bonds=[],
            ),
            "EC": MoleculeType(
                name="EC",
                atomtypes=["o", "c", "os"],
                atomnames=["O1", "C2", "O3"],
                charges=[-0.5, 0.8, -0.3],
                masses=[15.999, 12.011, 15.999],
                bonds=[(1, 2), (2, 3)],
            ),
            "TFSI": MoleculeType(
                name="TFSI",
                atomtypes=["s6", "o", "o"],
                atomnames=["S1", "O2", "O3"],
                charges=[0.9, -0.45, -0.45],
                masses=[32.067, 15.999, 15.999],
                bonds=[(1, 2), (1, 3)],
            ),
        },
        molecules=[("Li", 1), ("EC", 1), ("TFSI", 1)],
    )


def test_run_migration_analysis_supports_pure_electrolyte_without_polymer(tmp_path: Path, monkeypatch):
    system_dir = _write_migration_system_files(tmp_path)
    gro = system_dir / "system.gro"
    gro.write_text("toy\n", encoding="utf-8")
    xtc = tmp_path / "md.xtc"
    xtc.write_text("xtc\n", encoding="utf-8")

    xyz = np.asarray(
        [
            [
                [0.10, 0.00, 0.00],
                [0.00, 0.00, 0.00],
                [0.10, 0.10, 0.00],
                [0.20, 0.00, 0.00],
                [1.00, 0.00, 0.00],
                [1.05, 0.00, 0.00],
                [1.05, 0.10, 0.00],
            ],
            [
                [0.12, 0.00, 0.00],
                [0.00, 0.00, 0.00],
                [0.10, 0.10, 0.00],
                [0.20, 0.00, 0.00],
                [1.00, 0.00, 0.00],
                [1.05, 0.00, 0.00],
                [1.05, 0.10, 0.00],
            ],
            [
                [1.02, 0.00, 0.00],
                [0.00, 0.00, 0.00],
                [0.10, 0.10, 0.00],
                [0.20, 0.00, 0.00],
                [1.00, 0.00, 0.00],
                [1.05, 0.00, 0.00],
                [1.05, 0.10, 0.00],
            ],
            [
                [1.03, 0.02, 0.00],
                [0.00, 0.00, 0.00],
                [0.10, 0.10, 0.00],
                [0.20, 0.00, 0.00],
                [1.00, 0.00, 0.00],
                [1.05, 0.00, 0.00],
                [1.05, 0.10, 0.00],
            ],
        ],
        dtype=float,
    )
    time_ps = np.asarray([0.0, 5.0, 10.0, 15.0], dtype=float)
    box_nm = np.repeat(np.asarray([[5.0, 5.0, 5.0]], dtype=float), 4, axis=0)

    fake_mdtraj = types.SimpleNamespace(iterload=lambda *args, **kwargs: [_FakeTraj(xyz, time_ps, box_nm)])
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)

    rdf_summary = {
        "EC:carbonyl_oxygen": {
            "center_group": "Li",
            "r_peak_nm": 0.20,
            "r_shell_nm": 0.35,
            "formal_cn_shell": 1.0,
            "confidence": "high",
        },
        "TFSI:sulfonyl_oxygen": {
            "center_group": "Li",
            "r_peak_nm": 0.22,
            "r_shell_nm": 0.32,
            "formal_cn_shell": 2.0,
            "confidence": "high",
        },
    }
    out = run_migration_analysis(
        top=_toy_topology(),
        system_dir=system_dir,
        gro_path=gro,
        xtc_path=xtc,
        center_moltype="Li",
        rdf_summary=rdf_summary,
        resolve_moltypes=_identity_resolver,
        out_dir=tmp_path / "06_analysis" / "migration",
    )

    assert out["residence_summary"]["polymer"]["available"] is False
    assert out["residence_summary"]["solvent"]["available"] is True
    assert out["residence_summary"]["anion"]["available"] is True
    assert out["markov_role_summary"]["selected_lag_frames"] >= 1
    assert out["markov_site_summary"]["state_count"] >= 2
    assert out["event_flux_summary"]["available"] is True
    assert Path(out["outputs"]["transition_matrix_role_csv"]).exists()
    assert Path(out["outputs"]["transition_matrix_site_csv"]).exists()
    assert Path(out["outputs"]["predicted_event_counts_csv"]).exists()
    assert Path(out["outputs"]["coordination_summary_json"]).exists()

    with Path(out["outputs"]["transition_matrix_role_csv"]).open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    matrix = np.asarray([[float(x) for x in row[1:]] for row in rows[1:]], dtype=float)
    nonzero_rows = np.sum(matrix, axis=1) > 0.0
    if np.any(nonzero_rows):
        assert np.allclose(np.sum(matrix[nonzero_rows], axis=1), 1.0)


def test_site_level_sparse_states_are_lumped_into_other():
    anchors = [
        _make_anchor(
            anchor_id=f"polymer:PEO:0:{idx}:ether_oxygen",
            role="polymer",
            moltype="PEO",
            site_id="PEO:ether_oxygen",
            chain_key="PEO:chain:0",
            residue_number=idx,
        )
        for idx in range(6)
    ]
    dominant = np.asarray([[0], [1], [2], [3], [4], [5], [0], [1], [2], [3]], dtype=int)
    site_states, site_specs, anchor_to_state = _build_site_state_trajectory(
        dominant,
        anchors,
        max_states_per_role=2,
        min_occ_fraction=0.20,
        min_transition_count=10,
    )

    assert site_states.shape == dominant.shape
    assert anchor_to_state[0] != 0
    assert any(spec.state_id == "polymer:OTHER" for spec in site_specs)


def test_markov_transition_counts_and_prediction_recover_simple_chain():
    states = np.asarray([[0], [0], [1], [1], [1], [0], [0], [1]], dtype=int)
    counts = _transition_counts(states, n_states=2, lag_frames=1)
    matrix = _normalize_transition_counts(counts)

    assert counts.shape == (2, 2)
    assert counts[0, 1] > 0
    assert counts[1, 0] > 0
    assert np.allclose(np.sum(matrix, axis=1), 1.0)

    state_specs = [
        StateSpec(
            state_index=0,
            state_id="polymer:a0",
            state_label="polymer:a0",
            role="polymer",
            bucket="anchor",
            anchor_id="polymer:a0",
            site_id="PEO:ether_oxygen",
            moltype="PEO",
            chain_key="PEO:chain:0",
            occupancy_fraction=0.5,
            note=None,
        ),
        StateSpec(
            state_index=1,
            state_id="polymer:a1",
            state_label="polymer:a1",
            role="polymer",
            bucket="anchor",
            anchor_id="polymer:a1",
            site_id="PEO:ether_oxygen",
            moltype="PEO",
            chain_key="PEO:chain:1",
            occupancy_fraction=0.5,
            note=None,
        ),
    ]
    out = _predict_event_counts(
        matrix,
        state_specs=state_specs,
        initial_occupancy=np.asarray([0.5, 0.5], dtype=float),
        n_centers=1,
        lag_ps=5.0,
    )

    assert out["selected_lag_ps"] == 5.0
    assert out["predicted_event_counts"]
    final_row = out["predicted_event_counts"][-1]
    assert final_row["polymer_interchain_hop"] > 0.0


def test_analyze_result_migration_integrates_with_api(tmp_path: Path, monkeypatch):
    system_dir = _write_migration_system_files(tmp_path)
    (system_dir / "system.top").write_text("; top\n", encoding="utf-8")
    (system_dir / "system.ndx").write_text("[ System ]\n1 2 3\n", encoding="utf-8")
    (system_dir / "system.gro").write_text("toy\n", encoding="utf-8")
    tpr = tmp_path / "md.tpr"
    xtc = tmp_path / "md.xtc"
    edr = tmp_path / "md.edr"
    for path in (tpr, xtc, edr):
        path.write_text("stub\n", encoding="utf-8")

    analyzer = AnalyzeResult(
        work_dir=tmp_path,
        tpr=tpr,
        xtc=xtc,
        edr=edr,
        top=system_dir / "system.top",
        ndx=system_dir / "system.ndx",
    )
    monkeypatch.setattr("yadonpy.sim.analyzer.parse_system_top", lambda _path: _toy_topology())
    monkeypatch.setattr(
        AnalyzeResult,
        "rdf",
        lambda self, **kwargs: {
            "EC:carbonyl_oxygen": {"center_group": "Li", "r_shell_nm": 0.35, "formal_cn_shell": 1.0, "confidence": "high"},
            "TFSI:sulfonyl_oxygen": {"center_group": "Li", "r_shell_nm": 0.32, "formal_cn_shell": 2.0, "confidence": "high"},
        },
    )
    monkeypatch.setattr(
        "yadonpy.sim.migration.run_migration_analysis",
        lambda **kwargs: {
            "migration_summary": {
                "center_moltype": "Li",
                "center_count": 2,
                "selected_lag_ps": 10.0,
                "outputs": {"transition_matrix_role_csv": "transition_role.csv"},
            },
            "residence_summary": {"polymer": {"available": False}},
            "coordination_summary": {"roles": {}},
            "markov_role_summary": {"selected_lag_ps": 10.0, "markov_confidence": "medium"},
            "markov_site_summary": {"selected_lag_ps": 10.0, "markov_confidence": "medium"},
            "event_flux_summary": {"available": True, "event_counts_observed": {"site_stay": 1}, "predicted_event_counts": []},
            "state_catalog": {"role_states": [], "site_states": []},
            "event_counts": {"site_stay": 1},
            "outputs": {"transition_matrix_role_csv": "transition_role.csv"},
        },
    )

    result = analyzer.migration(center_mol=Chem.MolFromSmiles("[Li+]"))
    assert result["migration_summary"]["center_moltype"] == "Li"

    markov_result = analyzer.migration_markov(center_mol=Chem.MolFromSmiles("[Li+]"))
    assert markov_result["markov_role_summary"]["selected_lag_ps"] == 10.0

    hopping_result = analyzer.migration_hopping(center_mol=Chem.MolFromSmiles("[Li+]"))
    assert hopping_result["deprecated"] is True
    assert hopping_result["event_counts"]["site_stay"] == 1

    summary = json.loads((tmp_path / "06_analysis" / "summary.json").read_text(encoding="utf-8"))
    assert "migration" in summary
