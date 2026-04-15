from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


EX08_DIR = Path(__file__).resolve().parents[1] / "examples" / "08_graphite_polymer_electrolyte_sandwich"
if str(EX08_DIR) not in sys.path:
    sys.path.insert(0, str(EX08_DIR))

import _autofix as autofix  # noqa: E402
import run_iteration_matrix as matrix  # noqa: E402


def test_build_failure_signature_classifies_acceptance_failure(tmp_path: Path):
    matrix_dir = tmp_path / "matrix"
    remote_base = Path("/remote/round_001")
    local_case = matrix_dir / "iter_001" / "fresh" / "01_peo_smoke" / "work_dir" / "06_full_stack_release"
    local_case.mkdir(parents=True, exist_ok=True)
    manifest = {
        "acceptance": {
            "accepted": False,
            "failed_checks": ["polymer_density_ok", "wrapped_ok"],
            "polymer_density_g_cm3": 0.82,
            "electrolyte_density_g_cm3": 1.07,
            "wrapped_across_z_boundary": {"polymer": False, "electrolyte": True},
        },
        "stack_gap_ang": {
            "graphite_to_polymer": 18.0,
            "polymer_to_electrolyte": 42.0,
        },
    }
    (local_case / "interface_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (local_case / "interface_progress.json").write_text(json.dumps({"stage": "completed"}), encoding="utf-8")
    logs_dir = matrix_dir / "iter_001" / "fresh" / "01_peo_smoke" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "fresh.log").write_text("done\n", encoding="utf-8")
    latest = {
        "iteration": 1,
        "results": [
            {
                "case": "01_peo_smoke",
                "phase": "fresh",
                "returncode": 0,
                "timed_out": False,
                "manifest_exists": True,
                "manifest_path": str(remote_base / "iter_001" / "fresh" / "01_peo_smoke" / "work_dir" / "06_full_stack_release" / "interface_manifest.json"),
                "progress_path": str(remote_base / "iter_001" / "fresh" / "01_peo_smoke" / "work_dir" / "06_full_stack_release" / "interface_progress.json"),
                "log_path": str(remote_base / "iter_001" / "fresh" / "01_peo_smoke" / "logs" / "fresh.log"),
            }
        ],
    }
    (matrix_dir / "latest_status.json").parent.mkdir(parents=True, exist_ok=True)
    (matrix_dir / "latest_status.json").write_text(json.dumps(latest), encoding="utf-8")

    signature = autofix.build_failure_signature(matrix_dir=matrix_dir, remote_base_dir=remote_base)
    assert signature["primary_failure_class"] == "acceptance_failure"
    assert signature["acceptance_failure_breakdown"]["polymer_density_ok"] == 1
    assert signature["acceptance_failure_breakdown"]["wrapped_ok"] == 1
    assert signature["stack_gap_ang_stats"]["polymer_to_electrolyte_mean"] == 42.0


def test_compare_round_metrics_prefers_improved_acceptance():
    previous = {"comparison_key": (0, 2, 1, -5, -3.0)}
    current = {"comparison_key": (1, 2, 1, -3, -2.5)}
    comparison = autofix.compare_round_metrics(previous, current)
    assert comparison["improved"] is True


def test_select_recipe_prefers_stack_gap_fix_when_missing_repo_fix(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "src/yadonpy/interface").mkdir(parents=True, exist_ok=True)
    (repo_root / "src/yadonpy/interface/sandwich_metrics.py").write_text("center_window_nm\n", encoding="utf-8")
    (repo_root / "src/yadonpy/interface/sandwich.py").write_text('summary.get("center_bulk_like_window_nm")\n', encoding="utf-8")
    signature = {
        "primary_failure_class": "acceptance_failure",
        "stack_gap_ang_stats": {"polymer_to_electrolyte_mean": 38.0},
        "cases": [],
    }
    config = autofix.AutofixConfig()
    decision = autofix.select_recipe(signature=signature, repo_root=repo_root, config=config, attempted=set())
    assert decision["recipe"] == "fix_stack_gap_estimation"


def test_select_recipe_chooses_confined_selection_for_wrapped_acceptance(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "src/yadonpy/interface").mkdir(parents=True, exist_ok=True)
    (repo_root / "src/yadonpy/interface/sandwich_metrics.py").write_text("center_bulk_like_window_nm\n", encoding="utf-8")
    (repo_root / "src/yadonpy/interface/sandwich.py").write_text('summary.get("center_bulk_like_window_nm", summary.get("center_window_nm"))\n', encoding="utf-8")
    signature = {
        "primary_failure_class": "acceptance_failure",
        "stack_gap_ang_stats": {"polymer_to_electrolyte_mean": 12.0},
        "cases": [
            {"wrapped": {"polymer": True, "electrolyte": False}},
            {"wrapped": {"polymer": False, "electrolyte": True}},
        ],
    }
    config = autofix.AutofixConfig()
    decision = autofix.select_recipe(signature=signature, repo_root=repo_root, config=config, attempted=set())
    assert decision["recipe"] == "tighten_or_relax_confined_selection"


def test_select_recipe_returns_unclassified_when_attempted_exhausted(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "src/yadonpy/interface").mkdir(parents=True, exist_ok=True)
    (repo_root / "src/yadonpy/interface/sandwich_metrics.py").write_text("center_bulk_like_window_nm\n", encoding="utf-8")
    (repo_root / "src/yadonpy/interface/sandwich.py").write_text('summary.get("center_bulk_like_window_nm", summary.get("center_window_nm"))\n', encoding="utf-8")
    signature = {
        "primary_failure_class": "acceptance_failure",
        "stack_gap_ang_stats": {"polymer_to_electrolyte_mean": 12.0},
        "cases": [
            {"wrapped": {"polymer": True, "electrolyte": False}},
            {"wrapped": {"polymer": False, "electrolyte": True}},
        ],
    }
    config = autofix.AutofixConfig()
    decision = autofix.select_recipe(
        signature=signature,
        repo_root=repo_root,
        config=config,
        attempted={"tighten_or_relax_confined_selection"},
    )
    assert decision["selected"] is False
    assert decision["recipe"] == "unclassified_failure"


def test_select_recipe_skips_already_applied_confined_selection_for_gap(tmp_path: Path):
    repo_root = tmp_path / "repo"
    (repo_root / "src/yadonpy/interface").mkdir(parents=True, exist_ok=True)
    (repo_root / "src/yadonpy/interface/sandwich_metrics.py").write_text(
        "center_bulk_like_window_nm\nscore += 1.75\n",
        encoding="utf-8",
    )
    (repo_root / "src/yadonpy/interface/sandwich.py").write_text(
        "    graphite_polymer_gap_nm = (float(relax.graphite_to_polymer_gap_ang) / 10.0) + 0.35 * polymer_shell_nm\n"
        "    polymer_electrolyte_gap_nm = (float(relax.polymer_to_electrolyte_gap_ang) / 10.0) + 0.35 * (\n",
        encoding="utf-8",
    )
    signature = {
        "primary_failure_class": "acceptance_failure",
        "stack_gap_ang_stats": {"polymer_to_electrolyte_mean": 32.0},
        "cases": [{"wrapped": {"polymer": True, "electrolyte": True}}],
    }
    decision = autofix.select_recipe(
        signature=signature,
        repo_root=repo_root,
        config=autofix.AutofixConfig(),
        attempted=set(),
    )
    assert decision["recipe"] == "reduce_release_gap_or_padding"


def test_push_safety_requires_explicit_main_unlock(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    (repo_root / "src/yadonpy/interface").mkdir(parents=True, exist_ok=True)
    tracked = repo_root / "src/yadonpy/interface/sandwich.py"
    tracked.write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "autofix@example.invalid"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "Example 08 Autofix"], cwd=repo_root, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=repo_root, check=True, capture_output=True)
    tracked.write_text("new\n", encoding="utf-8")

    previous = {"completed_cases": 1, "primary_failure_class": "acceptance_failure"}
    current = {"completed_cases": 1, "primary_failure_class": "acceptance_failure"}
    comparison = {"improved": True}
    push = autofix.PushPolicy(enabled=True, branch="main", require_env_var="YADONPY_TEST_MAIN_PUSH")

    locked = autofix._build_push_safety_report(
        repo_root=repo_root,
        push=push,
        recipe_name="adjust_screening_route_fail_fast",
        before_metrics=previous,
        after_metrics=current,
        comparison=comparison,
    )
    assert locked["ok"] is False
    assert "YADONPY_TEST_MAIN_PUSH" in locked["failures"][-1]

    monkeypatch.setenv("YADONPY_TEST_MAIN_PUSH", "1")
    unlocked = autofix._build_push_safety_report(
        repo_root=repo_root,
        push=push,
        recipe_name="adjust_screening_route_fail_fast",
        before_metrics=previous,
        after_metrics=current,
        comparison=comparison,
    )
    assert unlocked["ok"] is True


def test_matrix_dry_run_records_selected_phases(tmp_path: Path):
    rc = matrix.run_matrix_loop(
        base_dir=tmp_path,
        hours=0.25,
        max_iterations=1,
        dry_run=True,
        phases=("fresh",),
    )
    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert metadata["phases"] == ["fresh"]
    assert metadata["matrix_fast_env"] == "YADONPY_MATRIX_FAST=1"


def test_autofix_config_defaults_to_fresh_only():
    config = autofix.load_autofix_config()
    assert config.matrix.phases == ("fresh",)
