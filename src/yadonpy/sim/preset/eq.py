"""YadonPy equilibration presets.

These presets are thin wrappers over the pure-GROMACS workflows in
:mod:`yadonpy.gmx.workflows`.

Key design goals
---------------
- Accept an amorphous cell returned by :func:`yadonpy.core.poly.amorphous_cell`.
- Provide a consistent (restart, gpu, gpu_id) interface.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from ...core import utils
from ...gmx.mdp_templates import MdpSpec
from ...gmx.workflows._util import RunResources
from ...gmx.workflows.eq import EqStage, EquilibrationJob, StageLincsRetryPolicy
from ...io.gromacs_system import SystemExportResult, export_system_from_cell_meta, validate_exported_system_dir
from ...runtime import resolve_restart
from ...workflow import ResumeManager, StepSpec
from ...workflow.resume import file_signature
from ..analyzer import AnalyzeResult
from ..performance import IOAnalysisPolicy, resolve_io_analysis_policy


_EXPORT_SYSTEM_SCHEMA_VERSION = "0.8.61-export-v2"


def _preset_log(message: str, *, level: int = 1) -> None:
    utils.radon_print(message, level=level)


def _fmt_elapsed(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def _preset_item(label: str, value: object) -> None:
    _preset_log(f"[ITEM] {label:<18}: {value}")


def _preset_section(title: str, *, detail: Optional[str] = None) -> float:
    _preset_log('=' * 88)
    _preset_log(f"[SECTION] {title}")
    if detail:
        _preset_log(f"[NOTE] {detail}")
    return __import__('time').perf_counter()


def _preset_done(title: str, t0: float, *, detail: Optional[str] = None) -> None:
    import time as _time

    msg = f"[DONE] {title} | elapsed={_fmt_elapsed(_time.perf_counter() - float(t0))}"
    if detail:
        msg += f" | {detail}"
    _preset_log(msg)


def _write_export_manifest(sys_dir: Path, *, schema_version: str, ff_name: str, charge_method: str, charge_scale: Any, issues: list[str]) -> Path:
    manifest = sys_dir / "export_manifest.json"
    payload = {
        "schema_version": str(schema_version),
        "ff_name": str(ff_name),
        "charge_method": str(charge_method),
        "charge_scale": charge_scale,
        "topology_issues": list(issues),
    }
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
    _preset_log('=' * 88)


def _cell_resume_signature(ac) -> dict[str, Any]:
    sig: dict[str, Any] = {}
    try:
        sig["num_atoms"] = int(ac.GetNumAtoms())
    except Exception:
        sig["num_atoms"] = None
    try:
        if hasattr(ac, "HasProp") and ac.HasProp("_yadonpy_cell_meta"):
            cell_meta = ac.GetProp("_yadonpy_cell_meta")
            sig["cell_meta_sha256"] = hashlib.sha256(str(cell_meta).encode("utf-8", errors="replace")).hexdigest()
    except Exception:
        sig["cell_meta_sha256"] = None
    try:
        cell = getattr(ac, "cell", None)
        if cell is not None:
            sig["cell"] = {
                "xhi": float(cell.xhi),
                "xlo": float(cell.xlo),
                "yhi": float(cell.yhi),
                "ylo": float(cell.ylo),
                "zhi": float(cell.zhi),
                "zlo": float(cell.zlo),
            }
    except Exception:
        sig["cell"] = None
    try:
        conf = ac.GetConformer()
        coords = np.asarray(conf.GetPositions(), dtype=np.float32)
        rounded = np.round(coords, 3)
        sig["coord_sha256"] = hashlib.sha256(rounded.tobytes()).hexdigest()
    except Exception:
        sig["coord_sha256"] = None
    return sig


def _workflow_summary_matches(summary_path: Path, *, input_gro_sig: dict[str, Any], input_top_sig: dict[str, Any], input_ndx_sig: dict[str, Any] | None = None) -> bool:
    if not summary_path.exists():
        return False
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        return False
    if provenance.get("input_gro_sig") != input_gro_sig:
        return False
    if provenance.get("input_top_sig") != input_top_sig:
        return False
    if provenance.get("input_ndx_sig") != input_ndx_sig:
        return False
    return True


def _outputs_newer_than_inputs(
    outputs: Sequence[Path],
    *,
    input_gro_sig: dict[str, Any],
    input_top_sig: dict[str, Any],
    input_ndx_sig: dict[str, Any] | None = None,
) -> bool:
    try:
        output_paths = [Path(p) for p in outputs]
        if not output_paths or not all(path.exists() for path in output_paths):
            return False
        oldest_output = min(path.stat().st_mtime for path in output_paths)
    except Exception:
        return False

    input_paths: list[Path] = []
    for sig in (input_gro_sig, input_top_sig, input_ndx_sig):
        if not isinstance(sig, dict):
            continue
        raw = sig.get("path")
        if not raw:
            continue
        try:
            path = Path(str(raw))
        except Exception:
            continue
        if not path.exists():
            return False
        input_paths.append(path)

    if not input_paths:
        return False

    try:
        newest_input = max(path.stat().st_mtime for path in input_paths)
    except Exception:
        return False
    return oldest_output >= newest_input


def _stage_done_tag_path(stage_dir: Path) -> Path:
    return Path(stage_dir) / ".yadonpy_stage_done.json"


def _workflow_provenance_matches(
    payload: dict[str, Any],
    *,
    input_gro_sig: dict[str, Any],
    input_top_sig: dict[str, Any],
    input_ndx_sig: dict[str, Any] | None = None,
) -> bool:
    provenance = payload.get("workflow_provenance")
    if not isinstance(provenance, dict):
        provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return False
    if provenance.get("input_gro_sig") != input_gro_sig:
        return False
    if provenance.get("input_top_sig") != input_top_sig:
        return False
    if provenance.get("input_ndx_sig") != input_ndx_sig:
        return False
    return True


def _workflow_marker_matches(
    marker_path: Path,
    *,
    outputs: Sequence[Path],
    input_gro_sig: dict[str, Any],
    input_top_sig: dict[str, Any],
    input_ndx_sig: dict[str, Any] | None = None,
) -> bool:
    if not marker_path.exists():
        return False
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if _workflow_provenance_matches(
        payload,
        input_gro_sig=input_gro_sig,
        input_top_sig=input_top_sig,
        input_ndx_sig=input_ndx_sig,
    ):
        return True
    return _outputs_newer_than_inputs(
        outputs,
        input_gro_sig=input_gro_sig,
        input_top_sig=input_top_sig,
        input_ndx_sig=input_ndx_sig,
    )


def _latest_reusable_stage_progress(
    run_dir: Path,
    stage_names: Sequence[str],
    *,
    input_gro_sig: dict[str, Any],
    input_top_sig: dict[str, Any],
    input_ndx_sig: dict[str, Any] | None = None,
) -> tuple[Optional[str], Optional[str]]:
    latest_completed: Optional[str] = None
    next_stage: Optional[str] = None
    for idx, stage_name in enumerate(stage_names):
        stage_dir = Path(run_dir) / str(stage_name)
        stage_gro = stage_dir / "md.gro"
        if not stage_gro.exists():
            break
        stage_outputs = tuple(
            path
            for path in (
                stage_dir / "md.tpr",
                stage_dir / "md.xtc",
                stage_dir / "md.edr",
                stage_dir / "md.gro",
            )
            if path.exists()
        )
        marker_ok = _workflow_marker_matches(
            _stage_done_tag_path(stage_dir),
            outputs=stage_outputs,
            input_gro_sig=input_gro_sig,
            input_top_sig=input_top_sig,
            input_ndx_sig=input_ndx_sig,
        )
        if not marker_ok:
            marker_ok = _workflow_marker_matches(
                stage_dir / "summary.json",
                outputs=stage_outputs,
                input_gro_sig=input_gro_sig,
                input_top_sig=input_top_sig,
                input_ndx_sig=input_ndx_sig,
            )
        if not marker_ok:
            break
        latest_completed = str(stage_name)
        next_stage = str(stage_names[idx + 1]) if idx + 1 < len(stage_names) else None
    return latest_completed, next_stage


def _recover_completed_workflow_step(
    resume: ResumeManager,
    spec: StepSpec,
    *,
    summary_path: Path,
    input_gro_sig: dict[str, Any],
    input_top_sig: dict[str, Any],
    input_ndx_sig: dict[str, Any] | None = None,
    label: str,
    fallback_markers: Sequence[Path] = (),
) -> bool:
    """Recover a finished workflow step when outputs exist but resume state is missing.

    This handles the common case where the scientific artifacts were fully written
    but the process was terminated before ``resume_state.json`` was updated.
    """
    status = resume.reuse_status(spec)
    if status == "done":
        return False
    if not all(Path(p).exists() for p in spec.outputs):
        return False
    if _workflow_summary_matches(
        summary_path,
        input_gro_sig=input_gro_sig,
        input_top_sig=input_top_sig,
        input_ndx_sig=input_ndx_sig,
    ):
        _preset_log(
            f"[RESTART] Recovered completed {label} from existing outputs; "
            f"resume state status was {status}.",
            level=1,
        )
        resume.mark_done(
            spec,
            meta={
                "recovered_from_summary": True,
                "previous_resume_status": status,
            },
        )
        return True

    marker_paths = [Path(p) for p in fallback_markers]
    if not marker_paths:
        return False
    if not all(path.exists() for path in marker_paths):
        return False
    if not _outputs_newer_than_inputs(
        spec.outputs,
        input_gro_sig=input_gro_sig,
        input_top_sig=input_top_sig,
        input_ndx_sig=input_ndx_sig,
    ):
        return False

    _preset_log(
        f"[RESTART] Recovered completed {label} from stage artifacts; "
        f"resume state status was {status}.",
        level=1,
    )
    resume.mark_done(
        spec,
        meta={
            "recovered_from_stage_artifacts": True,
            "previous_resume_status": status,
            "fallback_markers": [str(path) for path in marker_paths],
        },
    )
    return True


def _parse_gpu_args(gpu: int, gpu_id: Optional[int]) -> tuple[bool, Optional[str]]:
    """Parse legacy + new GPU semantics.

    New semantics (preferred):
      - gpu: 1 (default) -> enable GPU
      - gpu: 0 -> disable GPU
      - gpu_id: int -> select GPU device id used by GROMACS (-gpu_id)

    Backward compatibility:
      - if gpu not in {0, 1} and gpu_id is None -> treat gpu as gpu_id and enable GPU.
    """
    try:
        g = int(gpu)
    except Exception:
        g = 1

    gid = gpu_id
    if gid is None and g not in (0, 1):
        gid = g
        g = 1

    use_gpu = bool(g)
    gid_s = str(int(gid)) if (use_gpu and gid is not None) else None
    return use_gpu, gid_s


def _normalize_gpu_offload_mode(mode: object) -> str:
    token = str(mode or "auto").strip().lower()
    if token in {"auto", "adaptive"}:
        return "auto"
    if token in {"full", "default"}:
        return "full"
    if token in {"conservative", "safe"}:
        return "conservative"
    if token in {"cpu", "none"}:
        return "cpu"
    raise ValueError(f"Unsupported gpu_offload_mode={mode!r}")


def _cell_meta_payload(ac) -> dict[str, Any]:
    try:
        if hasattr(ac, "HasProp") and ac.HasProp("_yadonpy_cell_meta"):
            raw = json.loads(ac.GetProp("_yadonpy_cell_meta"))
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass
    return {}


def _cell_meta_contains_polymer(ac) -> bool:
    payload = _cell_meta_payload(ac)
    species = list(payload.get("species") or [])
    for sp in species:
        if not isinstance(sp, dict):
            continue
        kind = str(sp.get("kind") or "").strip().lower()
        smiles = str(sp.get("smiles") or sp.get("psmiles") or "")
        if kind == "polymer" or "*" in smiles:
            return True
        residue_map = sp.get("residue_map")
        if isinstance(residue_map, dict):
            try:
                if len(list(residue_map.get("residues") or [])) > 1:
                    return True
            except Exception:
                pass
        try:
            if bool(sp.get("polyelectrolyte_mode")) and int(sp.get("natoms") or 0) >= 40:
                return True
        except Exception:
            pass
    return False


def _cell_meta_contains_mobile_ions(ac) -> bool:
    payload = _cell_meta_payload(ac)
    species = list(payload.get("species") or [])
    ion_kinds = {"ion", "salt", "cation", "anion"}
    ion_names = {"li", "na", "k", "pf6", "fsi", "tfsi", "bf4", "cl", "br", "i"}
    for sp in species:
        if not isinstance(sp, dict):
            continue
        kind = str(sp.get("kind") or "").strip().lower()
        if kind in ion_kinds:
            return True
        label = str(sp.get("label") or sp.get("name") or sp.get("moltype") or sp.get("mol_name") or "").strip().lower()
        if label in ion_names:
            return True
        smiles = str(sp.get("smiles") or sp.get("psmiles") or "")
        if "+" in smiles or "-" in smiles:
            return True
        try:
            charge = float(sp.get("charge") or sp.get("net_charge") or sp.get("formal_charge") or 0.0)
        except Exception:
            charge = 0.0
        if abs(charge) > 1.0e-6:
            return True
    return False


def cell_meta_contains_polymer(ac) -> bool:
    """Return whether an amorphous cell metadata payload looks polymeric.

    This public wrapper lets examples and higher-level workflows choose an
    equilibration strategy without relying on private helper names.
    """

    return _cell_meta_contains_polymer(ac)


def select_relaxation_strategy(equilibrium_payload: dict[str, Any] | None, *, has_polymer: bool) -> str:
    """Choose the next additional-equilibration strategy from gate diagnostics."""

    payload = equilibrium_payload if isinstance(equilibrium_payload, dict) else {}
    if bool(payload.get("ok")):
        return "production"
    density_gate = payload.get("density_gate") if isinstance(payload.get("density_gate"), dict) else {}
    rg_gate = payload.get("rg_gate") if isinstance(payload.get("rg_gate"), dict) else {}
    state = payload.get("relaxation_state") if isinstance(payload.get("relaxation_state"), dict) else {}
    density_ok = bool(density_gate.get("ok"))
    rg_ok = True if not has_polymer else bool(rg_gate.get("ok"))
    still_compressing = bool(state.get("density_or_volume_still_compressing"))

    if not has_polymer:
        if still_compressing:
            return "liquid_density_recovery"
        if not density_ok:
            return "additional_npt"
        return "additional_npt"
    if (not density_ok) or still_compressing:
        return "polymer_density_recovery"
    if not rg_ok:
        return "polymer_chain_relaxation"
    return "additional_npt"


def _resolve_production_gpu_offload_mode(ac, requested_mode: object) -> str:
    token = _normalize_gpu_offload_mode(requested_mode)
    if token != "auto":
        return token
    # Public YadonPy default: every non-EM MD stage should use full GPU offload
    # when GPU execution is enabled.  The conservative CPU bonded/update path is
    # still available, but only when explicitly requested by the caller.
    return "full"


def _resolve_production_bridge_ps(ac, bridge_ps: float | None) -> float:
    if bridge_ps is not None:
        return float(bridge_ps)
    if _cell_meta_contains_polymer(ac):
        return 100.0
    if _cell_meta_contains_mobile_ions(ac):
        return 20.0
    return 0.0


def _resolve_nvt_density_control(ac, density_control: bool | None) -> bool:
    if density_control is not None:
        return bool(density_control)
    return True


def _normalize_constraints_mode(value: object) -> str:
    if value is None:
        return "none"
    if not isinstance(value, str):
        raise TypeError(f"constraints must be a string, got {type(value).__name__}")
    token = value.strip().lower().replace("_", "-")
    aliases = {
        "none": "none",
        "no": "none",
        "off": "none",
        "hbonds": "h-bonds",
        "h-bonds": "h-bonds",
        "allbonds": "all-bonds",
        "all-bonds": "all-bonds",
        "hangles": "h-angles",
        "h-angles": "h-angles",
    }
    normalized = aliases.get(token, token)
    if not normalized:
        raise ValueError("constraints must not be empty")
    return normalized


def _constraints_use_lincs(mode: object) -> bool:
    return _normalize_constraints_mode(mode) != "none"


def _interval_ps_to_nsteps(dt_ps: float, interval_ps: Optional[float], *, disabled_value: int = 0) -> int:
    if interval_ps is None:
        return int(disabled_value)
    interval = float(interval_ps)
    if interval <= 0.0:
        return int(disabled_value)
    return max(int(round(interval / float(dt_ps))), 1)


def _apply_production_output_cadence(
    params: dict[str, object],
    *,
    traj_ps: float,
    energy_ps: float,
    log_ps: Optional[float],
    trr_ps: Optional[float],
    velocity_ps: Optional[float],
) -> None:
    dt_ps = float(params["dt"])
    params["nstxout"] = _interval_ps_to_nsteps(dt_ps, float(traj_ps))
    params["nstenergy"] = _interval_ps_to_nsteps(dt_ps, float(energy_ps))
    params["nstlog"] = _interval_ps_to_nsteps(dt_ps, float(log_ps) if log_ps is not None else float(energy_ps))
    params["nstxout_trr"] = _interval_ps_to_nsteps(dt_ps, trr_ps, disabled_value=0)
    params["nstvout"] = _interval_ps_to_nsteps(dt_ps, velocity_ps, disabled_value=0)


def _estimate_atom_count_for_policy(ac: object, exp: SystemExportResult | None = None) -> int | None:
    try:
        n = int(ac.GetNumAtoms())  # type: ignore[attr-defined]
        if n > 0:
            return n
    except Exception:
        pass
    if exp is not None:
        try:
            lines = Path(exp.system_gro).read_text(encoding="utf-8", errors="replace").splitlines()
            if len(lines) >= 2:
                n = int(lines[1].strip())
                if n > 0:
                    return n
        except Exception:
            pass
        try:
            payload = json.loads(Path(exp.system_meta).read_text(encoding="utf-8"))
            species = payload.get("species") if isinstance(payload, dict) else None
            total = 0
            if isinstance(species, list):
                for row in species:
                    if not isinstance(row, dict):
                        continue
                    count = int(row.get("count") or row.get("n_molecules") or 0)
                    natoms = int(row.get("natoms") or row.get("num_atoms") or 0)
                    total += count * natoms
            if total > 0:
                return total
        except Exception:
            pass
    return None


def _write_production_policy_summary(run_dir: Path, policy: IOAnalysisPolicy) -> None:
    summary_path = Path(run_dir) / "summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    payload["performance_policy"] = policy.to_dict()
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception:
        pass


def _prepare_production_mdp_params(
    *,
    base_params: dict[str, object],
    dt_ps: float,
    constraints: str,
    lincs_iter: int | None,
    lincs_order: int | None,
    traj_ps: float,
    energy_ps: float,
    log_ps: Optional[float],
    trr_ps: Optional[float],
    velocity_ps: Optional[float],
    mdp_overrides: Optional[dict[str, object]] = None,
) -> tuple[dict[str, object], str]:
    params = dict(base_params)
    params["dt"] = float(dt_ps)
    params["constraints"] = _normalize_constraints_mode(constraints)
    if lincs_iter is not None:
        params["lincs_iter"] = int(lincs_iter)
    if lincs_order is not None:
        params["lincs_order"] = int(lincs_order)

    overrides = dict(mdp_overrides or {})
    if overrides:
        params.update(overrides)

    params["dt"] = float(params["dt"])
    constraints_mode = _normalize_constraints_mode(params.get("constraints", "none"))
    params["constraints"] = constraints_mode
    params["constraint_algorithm"] = "lincs" if _constraints_use_lincs(constraints_mode) else "none"

    _apply_production_output_cadence(
        params,
        traj_ps=float(traj_ps),
        energy_ps=float(energy_ps),
        log_ps=log_ps,
        trr_ps=trr_ps,
        velocity_ps=velocity_ps,
    )

    for key in ("nstxout", "nstenergy", "nstlog", "nstxout_trr", "nstvout"):
        if key in overrides:
            params[key] = overrides[key]

    return params, constraints_mode


def _build_production_stages(
    *,
    stage_name: str,
    template: str,
    params: dict[str, object],
    prod_ns: float,
    checkpoint_min: float,
    constraints_mode: str,
    bridge_ps: float = 0.0,
    bridge_dt_fs: float = 1.0,
    bridge_lincs_iter: int = 4,
    bridge_lincs_order: int = 12,
    first_stage_gen_vel: str = "no",
    settle_constraints: bool = True,
    settle_nsteps: int = 5000,
) -> list[EqStage]:
    from ...gmx.mdp_templates import MINIM_STEEP_HBONDS_MDP

    dt_ps = float(params["dt"])
    constraints_mode = _normalize_constraints_mode(constraints_mode)
    constraints_active = _constraints_use_lincs(constraints_mode)
    prod_steps = max(int((float(prod_ns) * 1000.0) / dt_ps), 1000)
    stages: list[EqStage] = []
    stage_index = 1
    if constraints_active and bool(settle_constraints):
        settle_params = {
            **params,
            "nsteps": int(max(1, int(settle_nsteps))),
            "emtol": 1000.0,
            "emstep": 0.0005,
            "constraints": constraints_mode,
            "constraint_algorithm": "lincs",
            "lincs_iter": max(int(params.get("lincs_iter", 2)), int(bridge_lincs_iter)),
            "lincs_order": max(int(params.get("lincs_order", 8)), int(bridge_lincs_order)),
        }
        stages.append(
            EqStage(
                f"{stage_index:02d}_settle_constraints",
                "minim",
                MdpSpec(MINIM_STEEP_HBONDS_MDP, settle_params),
                strict_constraints=True,
            )
        )
        stage_index += 1
    if float(bridge_ps) > 0.0:
        bridge_dt_ps = max(float(bridge_dt_fs), 0.1) / 1000.0
        bridge_steps = max(int(round(float(bridge_ps) / bridge_dt_ps)), 1)
        bridge_params = {
            **params,
            "dt": bridge_dt_ps,
            "nsteps": bridge_steps,
            "continuation": "no" if str(first_stage_gen_vel).strip().lower() == "yes" else "yes",
            "gen_vel": str(first_stage_gen_vel),
        }
        if constraints_active:
            bridge_params["lincs_iter"] = max(int(params.get("lincs_iter", 2)), int(bridge_lincs_iter))
            bridge_params["lincs_order"] = max(int(params.get("lincs_order", 8)), int(bridge_lincs_order))
        stages.append(
            EqStage(
                f"{stage_index:02d}_bridge_{stage_name}",
                "md",
                MdpSpec(template, bridge_params),
                checkpoint_minutes=(float(checkpoint_min) if float(checkpoint_min) > 0.0 else None),
            )
        )
        stage_index += 1
    main_name = f"{stage_index:02d}_{stage_name}"
    stages.append(
        EqStage(
            main_name,
            "md",
            MdpSpec(
                template,
                {
                    **params,
                    "nsteps": prod_steps,
                    "continuation": "yes" if float(bridge_ps) > 0.0 else ("no" if str(first_stage_gen_vel).strip().lower() == "yes" else "yes"),
                    "gen_vel": ("no" if float(bridge_ps) > 0.0 else str(first_stage_gen_vel)),
                },
            ),
            lincs_retry=StageLincsRetryPolicy() if constraints_active else None,
            checkpoint_minutes=(float(checkpoint_min) if float(checkpoint_min) > 0.0 else None),
        )
    )
    return stages


def _has_md_outputs(stage_dir: Path) -> bool:
    return all((Path(stage_dir) / name).exists() for name in ("md.tpr", "md.xtc", "md.edr", "md.gro"))


def _is_additional_final_stage(stage_dir: Path) -> bool:
    name = Path(stage_dir).name
    return name == "04_md" or name.endswith("_final_npt")


def _next_additional_round(work_dir: Path, *, restart: bool) -> tuple[int, Path]:
    """Pick (round_idx, out_dir) for an additional equilibration run.

    Rules:
    - If restart=True and the latest round exists but is incomplete, reuse it.
    - Otherwise create a new round directory.
    """
    # Keep top-level work_dir tidy with numbered module folders.
    base = Path(work_dir) / "04_eq_additional"
    base.mkdir(parents=True, exist_ok=True)

    rounds = []
    for d in base.glob("round_*"):
        if d.is_dir():
            try:
                idx = int(d.name.split("_")[-1])
            except Exception:
                continue
            rounds.append((idx, d))
    rounds.sort(key=lambda x: x[0])

    def _round_complete(d: Path) -> bool:
        for stage_dir in sorted(Path(d).iterdir() if Path(d).exists() else ()):
            if stage_dir.is_dir() and _is_additional_final_stage(stage_dir) and _has_md_outputs(stage_dir):
                return True
        return False

    if rounds and restart:
        idx, d = rounds[-1]
        # consider complete if final md files exist
        if not _round_complete(d):
            return idx, d
        return idx + 1, base / f"round_{idx + 1:02d}"

    if not rounds:
        return 0, base / "round_00"

    idx, _ = rounds[-1]
    return idx + 1, base / f"round_{idx + 1:02d}"


def _is_within_any(path: Path, roots: Sequence[Path]) -> bool:
    candidate = Path(path)
    for root in roots:
        try:
            candidate.relative_to(Path(root))
            return True
        except Exception:
            continue
    return False


def _find_latest_equilibrated_gro(work_dir: Path, *, exclude_dirs: Sequence[Path] | None = None) -> Optional[Path]:
    """Find the latest equilibrated coordinate file under work_dir.

    Priority:
            1) Production NPT run (05_npt_production/01_npt/md.gro)
            2) Additional rounds (highest round idx)
            3) Liquid anneal final NPT
            4) Main EQ21 run (new layout: 03_EQ21/03_EQ21/step_*/md.gro)
            5) Legacy main EQ run (03_eq/04_md/md.gro)
    """
    wd = Path(work_dir)
    excluded = tuple(Path(p) for p in (exclude_dirs or ()))
    prod = wd / "05_npt_production" / "01_npt" / "md.gro"
    if prod.exists() and not _is_within_any(prod, excluded):
        return prod

    add_base = wd / "04_eq_additional"
    if add_base.exists():
        rounds = []
        for d in add_base.glob("round_*"):
            if not d.is_dir():
                continue
            try:
                idx = int(d.name.split("_")[-1])
            except Exception:
                continue
            candidates = []
            for stage_dir in sorted(d.iterdir()):
                if not stage_dir.is_dir() or not _is_additional_final_stage(stage_dir):
                    continue
                gro = stage_dir / "md.gro"
                if gro.is_file():
                    candidates.append(gro)
            if not candidates:
                continue
            gro = candidates[-1]
            if gro.exists() and not _is_within_any(gro, excluded):
                rounds.append((idx, gro))
        if rounds:
            rounds.sort(key=lambda x: x[0])
            return rounds[-1][1]

    liquid_root = wd / "03_liquid_anneal"
    if liquid_root.exists():
        candidates = [
            p
            for p in liquid_root.glob("*final_npt/md.gro")
            if p.is_file() and not _is_within_any(p, excluded)
        ]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime)
            return candidates[-1]

    eq21_root = wd / "03_EQ21"
    if eq21_root.exists():
        ext_rounds = []
        for d in eq21_root.glob('final_extend/round_*'):
            if not d.is_dir():
                continue
            try:
                idx = int(d.name.split("_")[-1])
            except Exception:
                continue
            gro = d / "01_npt" / "md.gro"
            if gro.exists() and not _is_within_any(gro, excluded):
                ext_rounds.append((idx, gro))
        if ext_rounds:
            ext_rounds.sort(key=lambda x: x[0])
            return ext_rounds[-1][1]
        candidates = [p for p in eq21_root.glob('03_EQ21/step_*/md.gro') if p.is_file() and not _is_within_any(p, excluded)]
        if candidates:
            def _step_key(path: Path) -> int:
                try:
                    return int(path.parent.name.split('_')[-1])
                except Exception:
                    return -1
            candidates.sort(key=_step_key)
            return candidates[-1]

    gro = wd / "03_eq" / "04_md" / "md.gro"
    if gro.exists() and not _is_within_any(gro, excluded):
        return gro
    return None


def _invalidate_downstream_resume_steps(
    resume: ResumeManager,
    *,
    names: Sequence[str] = (),
    prefixes: Sequence[str] = (),
) -> None:
    removed = resume.invalidate_steps(names=names, prefixes=prefixes)
    if removed:
        _preset_log(
            "[RESTART] Invalidated downstream cached steps: " + ", ".join(sorted(removed)),
            level=1,
        )


@dataclass(frozen=True)
class EQ21ProtocolConfig:
    """Configurable parameters for the GROMACS EQ21 preset.

    Notes
    -----
    ``robust`` is enabled by default because the original 21-step ladder is
    often too aggressive for packed polymer/ionic cells in GROMACS: the box can
    over-react to the high-pressure NPT stages and fail with repeated
    ``pressure scaling more than 1%`` warnings before the density has settled.

    In robust mode YadonPy keeps the same *formal* 21-step ladder, but changes
    how each GROMACS stage is integrated:

    - velocities are generated only once (pre-NVT), then propagated across
      stages instead of being re-randomized every step;
    - high-temperature / high-pressure stages use a smaller MD time step;
    - intermediate high-pressure EQ21 stages use a damped Berendsen compaction
      phase with larger ``tau_p`` and reduced compressibility;
    - the final ambient-pressure stage still uses the requested barostat
      (default: ``C-rescale``).
    """

    t_max_k: float = 1000.0
    t_anneal_k: float = 300.0
    p_max_bar: float = 50000.0
    p_anneal_bar: float = 1.0
    dt_ps: float = 0.001
    pre_nvt_ps: float = 10.0
    tau_t_ps: float = 0.1
    tau_p_ps: float = 1.0
    compressibility_bar_inv: float = 4.5e-5
    barostat: str = "C-rescale"
    robust: bool = True
    reseed_each_stage: bool = False
    npt_time_scale: float = 2.0


def _ns_to_steps(ns: float, dt_ps: float) -> int:
    return max(int(round((float(ns) * 1000.0) / float(dt_ps))), 1)


def _ps_to_steps(ps: float, dt_ps: float) -> int:
    return max(int(round(float(ps) / float(dt_ps))), 1)


def _eq21_stage_dt_ps(ensemble: str, temperature_k: Optional[float], pressure_bar: Optional[float], cfg: EQ21ProtocolConfig) -> float:
    dt = float(cfg.dt_ps)
    if not cfg.robust:
        return dt

    t = float(temperature_k or 0.0)
    p = float(abs(pressure_bar or 0.0))
    if ensemble == 'NPT' and p >= max(0.10 * float(cfg.p_max_bar), 5000.0):
        return min(dt, 0.0005)
    if t >= 0.85 * float(cfg.t_max_k):
        return min(dt, 0.0005)
    if ensemble == 'NPT' and p >= max(0.01 * float(cfg.p_max_bar), 500.0):
        return min(dt, 0.00075)
    return dt


def _eq21_stage_tau_t_ps(temperature_k: Optional[float], cfg: EQ21ProtocolConfig) -> float:
    tau_t = float(cfg.tau_t_ps)
    if cfg.robust and float(temperature_k or 0.0) >= 0.85 * float(cfg.t_max_k):
        return max(tau_t, 0.2)
    return tau_t


def _eq21_barostat_controls(target_pressure_bar: float, prev_pressure_bar: float, *, is_final: bool, cfg: EQ21ProtocolConfig) -> dict[str, float | str]:
    target = float(target_pressure_bar)
    prev = float(prev_pressure_bar)
    shock = abs(target - prev)
    base_tau = float(cfg.tau_p_ps)
    base_comp = max(float(cfg.compressibility_bar_inv), 1.0e-8)

    if (not cfg.robust) or is_final:
        return {
            'pcoupl': str(cfg.barostat),
            'tau_p_ps': max(base_tau, 1.0) if is_final else base_tau,
            'compressibility_bar_inv': base_comp,
            'safety_mode': ('final-production' if is_final else 'legacy'),
        }

    if target >= max(0.60 * float(cfg.p_max_bar), 15000.0) or shock >= max(0.40 * float(cfg.p_max_bar), 15000.0):
        return {
            'pcoupl': 'Berendsen',
            'tau_p_ps': max(base_tau, 12.0),
            'compressibility_bar_inv': base_comp * 0.10,
            'safety_mode': 'high-pressure-damped',
        }
    if target >= max(0.15 * float(cfg.p_max_bar), 5000.0) or shock >= max(0.10 * float(cfg.p_max_bar), 5000.0):
        return {
            'pcoupl': 'Berendsen',
            'tau_p_ps': max(base_tau, 8.0),
            'compressibility_bar_inv': base_comp * 0.20,
            'safety_mode': 'pressure-ramp',
        }
    if target >= max(0.01 * float(cfg.p_max_bar), 500.0) or shock >= max(0.005 * float(cfg.p_max_bar), 500.0):
        return {
            'pcoupl': 'Berendsen',
            'tau_p_ps': max(base_tau, 4.0),
            'compressibility_bar_inv': base_comp * 0.35,
            'safety_mode': 'moderate-pressure-damped',
        }
    return {
        'pcoupl': 'Berendsen',
        'tau_p_ps': max(base_tau, 2.0),
        'compressibility_bar_inv': base_comp * 0.50,
        'safety_mode': 'gentle-compaction',
    }


def _build_eq21_records(temp: float, press: float, *, final_ns: float, cfg: EQ21ProtocolConfig) -> list[dict[str, Any]]:
    t_anneal = float(temp if temp is not None else cfg.t_anneal_k)
    p_anneal = float(press if press is not None else cfg.p_anneal_bar)
    t_max = float(cfg.t_max_k)
    p_max = float(cfg.p_max_bar)
    pre_dt = _eq21_stage_dt_ps('NVT', t_anneal, None, cfg)
    pre_tau_t = _eq21_stage_tau_t_ps(t_anneal, cfg)

    recs: list[dict[str, Any]] = [
        {
            "folder": "01_em",
            "protocol_step": "pre_em",
            "stage_label": "01_em",
            "display_name": "01_em",
            "segment": "pre",
            "ensemble": "EM",
            "temperature_k": None,
            "pressure_bar": None,
            "dt_ps": None,
            "nsteps": 50000,
            "time_ps": None,
            "time_ns": None,
            "tcoupl": None,
            "pcoupl": None,
            "tau_t_ps": None,
            "tau_p_ps": None,
            "compressibility_bar_inv": None,
            "velocity_reseed": False,
            "safety_mode": "minimize",
            "notes": "Steepest-descent energy minimization with constraints=none.",
        },
        {
            "folder": "02_preNVT",
            "protocol_step": "pre_nvt",
            "stage_label": "02_preNVT",
            "display_name": "02_preNVT",
            "segment": "pre",
            "ensemble": "NVT",
            "temperature_k": t_anneal,
            "pressure_bar": None,
            "dt_ps": pre_dt,
            "nsteps": _ps_to_steps(cfg.pre_nvt_ps, pre_dt),
            "time_ps": float(cfg.pre_nvt_ps),
            "time_ns": float(cfg.pre_nvt_ps) / 1000.0,
            "tcoupl": "V-rescale",
            "pcoupl": "no",
            "tau_t_ps": pre_tau_t,
            "tau_p_ps": None,
            "compressibility_bar_inv": None,
            "velocity_reseed": True,
            "safety_mode": "pre-thermalize",
            "notes": "Short pre-equilibration NVT before the formal 21-step ladder; this is the only stage that re-generates velocities by default.",
        },
    ]

    npt_scale = max(float(cfg.npt_time_scale), 1.0)

    eq21_steps = [
        ("NVT", t_max, None, 50.0),
        ("NVT", t_anneal, None, 50.0),
        ("NPT", t_anneal, 0.02 * p_max, 50.0),
        ("NVT", t_max, None, 50.0),
        ("NVT", t_anneal, None, 100.0),
        ("NPT", t_anneal, 0.6 * p_max, 50.0),
        ("NVT", t_max, None, 50.0),
        ("NVT", t_anneal, None, 100.0),
        ("NPT", t_anneal, 1.0 * p_max, 50.0),
        ("NVT", t_max, None, 50.0),
        ("NVT", t_anneal, None, 100.0),
        ("NPT", t_anneal, 0.5 * p_max, 5.0),
        ("NVT", t_max, None, 5.0),
        ("NVT", t_anneal, None, 10.0),
        ("NPT", t_anneal, 0.1 * p_max, 5.0),
        ("NVT", t_max, None, 5.0),
        ("NVT", t_anneal, None, 10.0),
        ("NPT", t_anneal, 0.01 * p_max, 5.0),
        ("NVT", t_max, None, 5.0),
        ("NVT", t_anneal, None, 10.0),
        ("NPT", t_anneal, p_anneal, float(final_ns) * 1000.0),
    ]

    prev_pressure = float(p_anneal)
    for i, (ensemble, t, p, time_ps) in enumerate(eq21_steps, start=1):
        stage_time_ps = float(time_ps) * (npt_scale if ensemble == 'NPT' else 1.0)
        folder = f"03_EQ21/step_{i:02d}"
        dt_stage = _eq21_stage_dt_ps(ensemble, t, p, cfg)
        tau_t_stage = _eq21_stage_tau_t_ps(t, cfg)
        is_final = bool(ensemble == 'NPT' and i == len(eq21_steps))
        if ensemble == 'NPT':
            ctrl = _eq21_barostat_controls(float(p), prev_pressure, is_final=is_final, cfg=cfg)
            tau_p_stage = float(ctrl['tau_p_ps'])
            comp_stage = float(ctrl['compressibility_bar_inv'])
            pcoupl = str(ctrl['pcoupl'])
            safety_mode = str(ctrl['safety_mode'])
            prev_pressure = float(p)
            notes = (
                f"Formal EQ21 stage {i:02d}; target pressure={float(p):.3f} bar | "
                f"barostat={pcoupl} | tau_p={tau_p_stage:.3f} ps | "
                f"compressibility={comp_stage:.3e} 1/bar | npt_time_scale={npt_scale:.2f} | mode={safety_mode}."
            )
        else:
            tau_p_stage = None
            comp_stage = None
            pcoupl = 'no'
            safety_mode = 'stage-chain' if cfg.robust else 'legacy'
            notes = (
                f"Formal EQ21 stage {i:02d}; NVT stage chaining uses dt={dt_stage:.4f} ps | "
                f"reseed_each_stage={bool(cfg.reseed_each_stage)}."
            )

        velocity_reseed = bool(cfg.reseed_each_stage)
        recs.append(
            {
                "folder": folder,
                "protocol_step": f"EQ21_{i:02d}",
                "stage_label": f"step_{i:02d}",
                "display_name": f"03_EQ21/step_{i:02d}",
                "segment": "eq21",
                "ensemble": ensemble,
                "temperature_k": float(t),
                "pressure_bar": (None if p is None else float(p)),
                "dt_ps": dt_stage,
                "nsteps": _ps_to_steps(stage_time_ps, dt_stage),
                "time_ps": float(stage_time_ps),
                "time_ns": float(stage_time_ps) / 1000.0,
                "tcoupl": "V-rescale",
                "pcoupl": pcoupl,
                "tau_t_ps": tau_t_stage,
                "tau_p_ps": tau_p_stage,
                "compressibility_bar_inv": comp_stage,
                "velocity_reseed": velocity_reseed,
                "safety_mode": safety_mode,
                "notes": notes,
            }
        )
    return recs


def _eq21_params_payload(temp: float, press: float, *, final_ns: float, cfg: EQ21ProtocolConfig) -> dict[str, Any]:
    return {
        "protocol": "EQ21_GROMACS",
        "t_max_k": float(cfg.t_max_k),
        "t_anneal_k": float(temp if temp is not None else cfg.t_anneal_k),
        "p_max_bar": float(cfg.p_max_bar),
        "p_anneal_bar": float(press if press is not None else cfg.p_anneal_bar),
        "dt_ps": float(cfg.dt_ps),
        "pre_nvt_ps": float(cfg.pre_nvt_ps),
        "final_stage_ns": float(final_ns),
        "tau_t_ps": float(cfg.tau_t_ps),
        "tau_p_ps": float(cfg.tau_p_ps),
        "barostat": str(cfg.barostat),
        "compressibility_bar_inv": float(cfg.compressibility_bar_inv),
        "robust": bool(cfg.robust),
        "reseed_each_stage": bool(cfg.reseed_each_stage),
        "npt_time_scale": float(cfg.npt_time_scale),
        "notes": [
            "Folders are created as 03_EQ21/01_em, 03_EQ21/02_preNVT, and 03_EQ21/03_EQ21/step_01 ... step_21.",
            "The 21-step protocol follows the provided Tmax/Tanal/Pmax/Panal ladder, translated to GROMACS NVT/NPT stages.",
            "EQ21 stages use constraints=none, so the preset does not invoke LINCS internally.",
            "Robust mode preserves the formal ladder but uses stage chaining, smaller dt at hot/high-pressure stages, and a softened barostat schedule for intermediate densification steps.",
            "All NPT stages are lengthened by eq21_npt_time_scale (default 2.0) to give the box more time to densify and remove cavities without making pressure coupling more aggressive.",
        ],
    }


def _print_eq21_schedule(records: list[dict[str, Any]], params: dict[str, Any]) -> None:
    print("\n[EQ21] Protocol parameters")
    print(
        f"  Tmax={params['t_max_k']:.1f} K | Tanal={params['t_anneal_k']:.1f} K | "
        f"Pmax={params['p_max_bar']:.1f} bar | Panal={params['p_anneal_bar']:.3f} bar | "
        f"base_dt={params['dt_ps']:.3f} ps | robust={bool(params.get('robust', True))} | "
        f"stage_reseed={bool(params.get('reseed_each_stage', False))} | npt_time_scale={float(params.get('npt_time_scale', 1.0)):.2f}"
    )
    print("[EQ21] Folder layout")
    print("  01_em       -> energy minimization")
    print("  02_preNVT   -> short pre-equilibration NVT")
    print("  03_EQ21     -> 21-step formal protocol (step_01 ... step_21)")
    print("[EQ21] Stage schedule")
    print("display_name          ens.   T(K)       P(bar)     dt(ps)   barostat     tau_p(ps)   nsteps      time(ps)   mode")
    print("-" * 132)
    for rec in records:
        t = "-" if rec['temperature_k'] is None else f"{rec['temperature_k']:.1f}"
        p = "-" if rec['pressure_bar'] is None else f"{rec['pressure_bar']:.3f}"
        dt = "-" if rec['dt_ps'] is None else f"{rec['dt_ps']:.4f}"
        pc = "-" if rec.get('pcoupl') in (None, 'no') else str(rec.get('pcoupl'))
        tau_p = "-" if rec.get('tau_p_ps') is None else f"{float(rec['tau_p_ps']):.2f}"
        ns = "-" if rec['nsteps'] is None else str(int(rec['nsteps']))
        tp = "-" if rec['time_ps'] is None else f"{rec['time_ps']:.1f}"
        mode = str(rec.get('safety_mode') or '-')
        print(f"{rec['display_name']:<20} {rec['ensemble']:<5} {t:>8}  {p:>11}  {dt:>8}  {pc:<11}  {tau_p:>10}  {ns:>10}  {tp:>11}  {mode}")
    print()


def _write_eq21_schedule(run_dir: Path, records: list[dict[str, Any]], params: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eq21_parameters.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "eq21_schedule.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "folder",
        "display_name",
        "protocol_step",
        "stage_label",
        "segment",
        "ensemble",
        "temperature_k",
        "pressure_bar",
        "dt_ps",
        "nsteps",
        "time_ps",
        "time_ns",
        "tcoupl",
        "pcoupl",
        "tau_t_ps",
        "tau_p_ps",
        "compressibility_bar_inv",
        "velocity_reseed",
        "safety_mode",
        "notes",
    ]
    with (run_dir / "eq21_schedule.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fieldnames})

    md_lines = [
        "# EQ21 schedule\n",
        "\n",
        "## Folder layout\n",
        "\n",
        "- `01_em`: energy minimization\n",
        "- `02_preNVT`: short pre-equilibration NVT\n",
        "- `03_EQ21/step_01 ... step_21`: formal 21-step ladder\n",
        "\n",
        "## Protocol parameters\n",
        "\n",
        f"- Tmax = {params['t_max_k']} K\n",
        f"- Tanal = {params['t_anneal_k']} K\n",
        f"- Pmax = {params['p_max_bar']} bar\n",
        f"- Panal = {params['p_anneal_bar']} bar\n",
        f"- base dt = {params['dt_ps']} ps\n",
        f"- pre-NVT = {params['pre_nvt_ps']} ps\n",
        f"- final EQ21 stage = {params['final_stage_ns']} ns\n",
        f"- robust = {bool(params.get('robust', True))}\n",
        f"- stage reseed = {bool(params.get('reseed_each_stage', False))}\n",
        f"- NPT time scale = {float(params.get('npt_time_scale', 1.0))}\n",
        "\n",
        "## Stage table\n",
        "\n",
        "| display_name | ensemble | T (K) | P (bar) | dt (ps) | barostat | tau_p (ps) | compressibility (1/bar) | reseed | mode | nsteps | time (ps) | notes |\n",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---|\n",
    ]
    for rec in records:
        md_lines.append(
            f"| {rec['display_name']} | {rec['ensemble']} | "
            f"{('-' if rec['temperature_k'] is None else rec['temperature_k'])} | "
            f"{('-' if rec['pressure_bar'] is None else rec['pressure_bar'])} | "
            f"{('-' if rec['dt_ps'] is None else rec['dt_ps'])} | "
            f"{('-' if rec.get('pcoupl') in (None, 'no') else rec.get('pcoupl'))} | "
            f"{('-' if rec.get('tau_p_ps') is None else rec.get('tau_p_ps'))} | "
            f"{('-' if rec.get('compressibility_bar_inv') is None else rec.get('compressibility_bar_inv'))} | "
            f"{bool(rec.get('velocity_reseed', False))} | "
            f"{rec.get('safety_mode') or '-'} | "
            f"{('-' if rec['nsteps'] is None else rec['nsteps'])} | "
            f"{('-' if rec['time_ps'] is None else rec['time_ps'])} | "
            f"{rec.get('notes') or '-'} |\n"
        )
    (run_dir / "eq21_schedule.md").write_text(''.join(md_lines), encoding="utf-8")


def _build_eq21_stages(temp: float, press: float, *, final_ns: float, cfg: EQ21ProtocolConfig) -> tuple[list[EqStage], list[dict[str, Any]], dict[str, Any]]:
    from ...gmx.mdp_templates import MINIM_STEEP_MDP, NVT_NO_CONSTRAINTS_MDP, NPT_NO_CONSTRAINTS_MDP, MdpSpec, default_mdp_params
    base = default_mdp_params()
    base["tau_t"] = float(cfg.tau_t_ps)
    base["tau_p"] = float(cfg.tau_p_ps)
    base["compressibility"] = float(cfg.compressibility_bar_inv)

    records = _build_eq21_records(temp=temp, press=press, final_ns=final_ns, cfg=cfg)
    params = _eq21_params_payload(temp=temp, press=press, final_ns=final_ns, cfg=cfg)

    stages: list[EqStage] = []
    stages.append(
        EqStage(
            records[0]['folder'],
            "minim",
            MdpSpec(MINIM_STEEP_MDP, {**base, "nsteps": 50000, "emtol": 1000.0, "emstep": 0.001}),
        )
    )

    pre = records[1]
    stages.append(
        EqStage(
            pre['folder'],
            'nvt',
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                {
                    **base,
                    'dt': float(pre['dt_ps']),
                    'tau_t': float(pre.get('tau_t_ps') or cfg.tau_t_ps),
                    'nsteps': int(pre['nsteps']),
                    'ref_t': float(pre['temperature_k']),
                    'gen_vel': 'yes',
                    'gen_temp': float(pre['temperature_k']),
                    'gen_seed': -1,
                },
            ),
        )
    )

    for idx, rec in enumerate(records[2:], start=1):
        seed = -(1000 + idx)
        gen_vel = 'yes' if bool(rec.get('velocity_reseed', False)) else 'no'
        common = {
            **base,
            'dt': float(rec['dt_ps']),
            'tau_t': float(rec.get('tau_t_ps') or cfg.tau_t_ps),
            'nsteps': int(rec['nsteps']),
            'ref_t': float(rec['temperature_k']),
            'gen_vel': gen_vel,
            'gen_temp': float(rec['temperature_k']),
            'gen_seed': seed,
        }
        if rec['ensemble'] == 'NVT':
            stages.append(
                EqStage(
                    rec['folder'],
                    'nvt',
                    MdpSpec(NVT_NO_CONSTRAINTS_MDP, common),
                )
            )
        else:
            stages.append(
                EqStage(
                    rec['folder'],
                    'npt',
                    MdpSpec(
                        NPT_NO_CONSTRAINTS_MDP,
                        {
                            **common,
                            'ref_p': float(rec['pressure_bar']),
                            'tau_p': float(rec.get('tau_p_ps') or cfg.tau_p_ps),
                            'compressibility': float(rec.get('compressibility_bar_inv') or cfg.compressibility_bar_inv),
                            'pcoupl': str(rec.get('pcoupl') or cfg.barostat),
                        },
                    ),
                )
            )

    return stages, records, params


def _ns_to_steps_for_dt(ns: float, dt_ps: float) -> int:
    return max(int(round((float(ns) * 1000.0) / float(dt_ps))), 1)


def _liquid_cooling_temperatures(target_temp: float, hot_temp: float) -> list[float]:
    """Default CEMP-like cooling ladder for simple liquids."""

    target = float(target_temp)
    hot = float(hot_temp)
    candidates = [450.0, target + 50.0]
    out: list[float] = []
    for t in candidates:
        t = float(t)
        if t <= target + 5.0 or t >= hot - 5.0:
            continue
        if all(abs(t - prev) > 5.0 for prev in out):
            out.append(t)
    return out


def _build_liquid_anneal_stages(
    *,
    temp: float,
    press: float,
    final_ns: float,
    hot_temp: float,
    hot_pressure_bar: float,
    compact_pressure_bar: float,
    hot_nvt_ns: float,
    compact_npt_ns: float,
    hot_npt_ns: float,
    cooling_npt_ns: float,
    dt_ps: float,
    hot_dt_ps: float,
    constraints: str,
    lincs_iter: Optional[int],
    lincs_order: Optional[int],
    checkpoint_min: float,
    mdp_overrides: Optional[dict[str, object]] = None,
) -> list[EqStage]:
    from ...gmx.mdp_templates import MINIM_STEEP_MDP, NPT_MDP, NPT_NO_CONSTRAINTS_MDP, NVT_MDP, NVT_NO_CONSTRAINTS_MDP, MdpSpec, default_mdp_params

    constraints_mode = _normalize_constraints_mode(constraints)
    use_lincs = _constraints_use_lincs(constraints_mode)
    nvt_template = NVT_MDP if use_lincs else NVT_NO_CONSTRAINTS_MDP
    npt_template = NPT_MDP if use_lincs else NPT_NO_CONSTRAINTS_MDP
    effective_lincs_iter = int(lincs_iter) if lincs_iter is not None else 2
    effective_lincs_order = int(lincs_order) if lincs_order is not None else 8

    def base_params(
        *,
        temperature: float,
        pressure: Optional[float],
        stage_dt_ps: float,
        nsteps: int,
        gen_vel: str,
        pcoupl: str = "C-rescale",
        tau_p_ps: Optional[float] = None,
        compressibility: float = 4.5e-5,
    ) -> dict[str, object]:
        p = default_mdp_params()
        p.update(
            {
                "dt": float(stage_dt_ps),
                "nsteps": int(max(nsteps, 1000)),
                "ref_t": float(temperature),
                "gen_temp": float(temperature),
                "gen_vel": str(gen_vel),
                "gen_seed": -1,
                "constraints": constraints_mode,
                "constraint_algorithm": "lincs" if use_lincs else "none",
                "lincs_iter": effective_lincs_iter,
                "lincs_order": effective_lincs_order,
                "tau_t": 0.5,
            }
        )
        if pressure is not None:
            p.update(
                {
                    "pcoupl": str(pcoupl),
                    "tau_p": float(tau_p_ps) if tau_p_ps is not None else 5.0,
                    "ref_p": float(pressure),
                    "compressibility": float(compressibility),
                }
            )
        return p

    stages: list[EqStage] = [
        EqStage(
            "01_em",
            "minim",
            MdpSpec(MINIM_STEEP_MDP, {**default_mdp_params(), "nsteps": 50000, "emtol": 1000.0, "emstep": 0.001}),
        )
    ]

    stages.append(
        EqStage(
            "02_hot_nvt",
            "nvt",
            MdpSpec(
                nvt_template,
                base_params(
                    temperature=float(hot_temp),
                    pressure=None,
                    stage_dt_ps=float(hot_dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(hot_nvt_ns), float(hot_dt_ps)),
                    gen_vel="yes",
                ),
            ),
            lincs_retry=StageLincsRetryPolicy() if use_lincs else None,
        )
    )
    stages.append(
        EqStage(
            "03_compact_npt",
            "npt",
            MdpSpec(
                npt_template,
                base_params(
                    temperature=float(hot_temp),
                    pressure=float(max(float(compact_pressure_bar), float(hot_pressure_bar))),
                    stage_dt_ps=float(hot_dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(compact_npt_ns), float(hot_dt_ps)),
                    gen_vel="no",
                    pcoupl="Berendsen",
                    tau_p_ps=5.0,
                    compressibility=4.5e-5,
                ),
            ),
            lincs_retry=StageLincsRetryPolicy() if use_lincs else None,
        )
    )
    stages.append(
        EqStage(
            "04_hot_npt",
            "npt",
            MdpSpec(
                npt_template,
                base_params(
                    temperature=float(hot_temp),
                    pressure=float(hot_pressure_bar),
                    stage_dt_ps=float(hot_dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(hot_npt_ns), float(hot_dt_ps)),
                    gen_vel="no",
                    pcoupl="C-rescale",
                    tau_p_ps=4.0,
                ),
            ),
            lincs_retry=StageLincsRetryPolicy() if use_lincs else None,
        )
    )

    stage_idx = 5
    for cool_temp in _liquid_cooling_temperatures(float(temp), float(hot_temp)):
        stages.append(
            EqStage(
                f"{stage_idx:02d}_cool_{int(round(cool_temp))}K_npt",
                "npt",
                MdpSpec(
                    npt_template,
                    base_params(
                        temperature=float(cool_temp),
                        pressure=float(press),
                        stage_dt_ps=float(dt_ps),
                        nsteps=_ns_to_steps_for_dt(float(cooling_npt_ns), float(dt_ps)),
                        gen_vel="no",
                    ),
                ),
                lincs_retry=StageLincsRetryPolicy() if use_lincs else None,
            )
        )
        stage_idx += 1

    stages.append(
        EqStage(
            f"{stage_idx:02d}_final_npt",
            "npt",
            MdpSpec(
                npt_template,
                base_params(
                    temperature=float(temp),
                    pressure=float(press),
                    stage_dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(final_ns), float(dt_ps)),
                    gen_vel="no",
                ),
            ),
            lincs_retry=StageLincsRetryPolicy() if use_lincs else None,
            checkpoint_minutes=float(checkpoint_min),
        )
    )

    return _apply_stage_mdp_overrides(stages, mdp_overrides)


def _liquid_templates_and_controls(
    *,
    constraints: str,
    lincs_iter: Optional[int],
    lincs_order: Optional[int],
) -> tuple[str, bool, int, int, object, object]:
    from ...gmx.mdp_templates import NPT_MDP, NPT_NO_CONSTRAINTS_MDP, NVT_MDP, NVT_NO_CONSTRAINTS_MDP

    constraints_mode = _normalize_constraints_mode(constraints)
    use_lincs = _constraints_use_lincs(constraints_mode)
    nvt_template = NVT_MDP if use_lincs else NVT_NO_CONSTRAINTS_MDP
    npt_template = NPT_MDP if use_lincs else NPT_NO_CONSTRAINTS_MDP
    effective_lincs_iter = int(lincs_iter) if lincs_iter is not None else 2
    effective_lincs_order = int(lincs_order) if lincs_order is not None else 8
    return constraints_mode, use_lincs, effective_lincs_iter, effective_lincs_order, nvt_template, npt_template


def _liquid_md_params(
    *,
    constraints_mode: str,
    use_lincs: bool,
    lincs_iter: int,
    lincs_order: int,
    temperature: float,
    pressure: Optional[float],
    dt_ps: float,
    nsteps: int,
    gen_vel: str,
    pcoupl: str = "C-rescale",
    tau_t_ps: float = 0.5,
    tau_p_ps: float = 2.0,
    compressibility: float = 4.5e-5,
) -> dict[str, object]:
    from ...gmx.mdp_templates import default_mdp_params

    p = default_mdp_params()
    p.update(
        {
            "dt": float(dt_ps),
            "nsteps": int(max(nsteps, 1000)),
            "ref_t": float(temperature),
            "gen_temp": float(temperature),
            "gen_vel": str(gen_vel),
            "gen_seed": -1,
            "constraints": constraints_mode,
            "constraint_algorithm": "lincs" if use_lincs else "none",
            "lincs_iter": int(lincs_iter),
            "lincs_order": int(lincs_order),
            "tau_t": float(tau_t_ps),
        }
    )
    if pressure is not None:
        p.update(
            {
                "pcoupl": str(pcoupl),
                "tau_p": float(tau_p_ps),
                "ref_p": float(pressure),
                "compressibility": float(compressibility),
            }
        )
    return p


def _build_liquid_recovery_compaction_stages(
    *,
    hot_temp: float,
    compact_pressure_bar: float,
    hot_nvt_ns: float,
    compact_npt_ns: float,
    hot_dt_ps: float,
    constraints: str,
    lincs_iter: Optional[int],
    lincs_order: Optional[int],
) -> list[EqStage]:
    from ...gmx.mdp_templates import MINIM_STEEP_MDP, MdpSpec, default_mdp_params

    constraints_mode, use_lincs, effective_lincs_iter, effective_lincs_order, nvt_template, npt_template = _liquid_templates_and_controls(
        constraints=constraints,
        lincs_iter=lincs_iter,
        lincs_order=lincs_order,
    )
    retry = StageLincsRetryPolicy() if use_lincs else None
    return [
        EqStage(
            "01_minim",
            "minim",
            MdpSpec(MINIM_STEEP_MDP, {**default_mdp_params(), "nsteps": 50000, "emtol": 1000.0, "emstep": 0.001}),
        ),
        EqStage(
            "02_hot_nvt",
            "nvt",
            MdpSpec(
                nvt_template,
                _liquid_md_params(
                    constraints_mode=constraints_mode,
                    use_lincs=use_lincs,
                    lincs_iter=effective_lincs_iter,
                    lincs_order=effective_lincs_order,
                    temperature=float(hot_temp),
                    pressure=None,
                    dt_ps=float(hot_dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(hot_nvt_ns), float(hot_dt_ps)),
                    gen_vel="yes",
                ),
            ),
            lincs_retry=retry,
        ),
        EqStage(
            "03_compact_npt",
            "npt",
            MdpSpec(
                npt_template,
                _liquid_md_params(
                    constraints_mode=constraints_mode,
                    use_lincs=use_lincs,
                    lincs_iter=effective_lincs_iter,
                    lincs_order=effective_lincs_order,
                    temperature=float(hot_temp),
                    pressure=float(compact_pressure_bar),
                    dt_ps=float(hot_dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(compact_npt_ns), float(hot_dt_ps)),
                    gen_vel="no",
                    pcoupl="Berendsen",
                    tau_p_ps=5.0,
                    compressibility=4.5e-5,
                ),
            ),
            lincs_retry=retry,
        ),
    ]


def _build_liquid_recovery_release_stages(
    *,
    temp: float,
    press: float,
    final_ns: float,
    hot_temp: float,
    hot_pressure_bar: float,
    cooling_npt_ns: float,
    dt_ps: float,
    hot_dt_ps: float,
    constraints: str,
    lincs_iter: Optional[int],
    lincs_order: Optional[int],
    checkpoint_min: float,
    mdp_overrides: Optional[dict[str, object]] = None,
) -> list[EqStage]:
    constraints_mode, use_lincs, effective_lincs_iter, effective_lincs_order, _nvt_template, npt_template = _liquid_templates_and_controls(
        constraints=constraints,
        lincs_iter=lincs_iter,
        lincs_order=lincs_order,
    )
    retry = StageLincsRetryPolicy() if use_lincs else None
    stages: list[EqStage] = [
        EqStage(
            "04_hot_release_npt",
            "npt",
            MdpSpec(
                npt_template,
                _liquid_md_params(
                    constraints_mode=constraints_mode,
                    use_lincs=use_lincs,
                    lincs_iter=effective_lincs_iter,
                    lincs_order=effective_lincs_order,
                    temperature=float(hot_temp),
                    pressure=float(hot_pressure_bar),
                    dt_ps=float(hot_dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(cooling_npt_ns), float(hot_dt_ps)),
                    gen_vel="no",
                    pcoupl="C-rescale",
                    tau_p_ps=4.0,
                ),
            ),
            lincs_retry=retry,
        )
    ]

    stage_idx = 5
    for cool_temp in _liquid_cooling_temperatures(float(temp), float(hot_temp)):
        stages.append(
            EqStage(
                f"{stage_idx:02d}_cool_{int(round(cool_temp))}K_npt",
                "npt",
                MdpSpec(
                    npt_template,
                    _liquid_md_params(
                        constraints_mode=constraints_mode,
                        use_lincs=use_lincs,
                        lincs_iter=effective_lincs_iter,
                        lincs_order=effective_lincs_order,
                        temperature=float(cool_temp),
                        pressure=float(press),
                        dt_ps=float(dt_ps),
                        nsteps=_ns_to_steps_for_dt(float(cooling_npt_ns), float(dt_ps)),
                        gen_vel="no",
                        pcoupl="C-rescale",
                        tau_p_ps=5.0,
                    ),
                ),
                lincs_retry=retry,
            )
        )
        stage_idx += 1

    stages.append(
        EqStage(
            f"{stage_idx:02d}_final_npt",
            "npt",
            MdpSpec(
                npt_template,
                _liquid_md_params(
                    constraints_mode=constraints_mode,
                    use_lincs=use_lincs,
                    lincs_iter=effective_lincs_iter,
                    lincs_order=effective_lincs_order,
                    temperature=float(temp),
                    pressure=float(press),
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(final_ns), float(dt_ps)),
                    gen_vel="no",
                    pcoupl="C-rescale",
                    tau_p_ps=5.0,
                ),
            ),
            lincs_retry=retry,
            checkpoint_minutes=float(checkpoint_min),
        )
    )
    return _apply_stage_mdp_overrides(stages, mdp_overrides)


def _polymer_warm_temperature(target_temp: float, warm_temp: Optional[float]) -> float:
    if warm_temp is not None and float(warm_temp) > 0.0:
        return float(warm_temp)
    return float(min(600.0, max(float(target_temp) + 100.0, 450.0)))


def _polymer_recovery_pressure(round_idx: int, pressure_ladder: Sequence[float] = (500.0, 1000.0, 2000.0, 5000.0)) -> float:
    ladder = [float(x) for x in pressure_ladder if float(x) > 0.0]
    if not ladder:
        return 1000.0
    idx = min(max(int(round_idx), 0), len(ladder) - 1)
    return float(ladder[idx])


def _no_constraint_md_params(
    *,
    temperature: float,
    pressure: Optional[float],
    dt_ps: float,
    nsteps: int,
    gen_vel: str,
    pcoupl: str = "C-rescale",
    tau_t_ps: float = 0.5,
    tau_p_ps: float = 2.0,
    compressibility: float = 4.5e-5,
) -> dict[str, object]:
    return _liquid_md_params(
        constraints_mode="none",
        use_lincs=False,
        lincs_iter=2,
        lincs_order=8,
        temperature=float(temperature),
        pressure=pressure,
        dt_ps=float(dt_ps),
        nsteps=int(nsteps),
        gen_vel=str(gen_vel),
        pcoupl=str(pcoupl),
        tau_t_ps=float(tau_t_ps),
        tau_p_ps=float(tau_p_ps),
        compressibility=float(compressibility),
    )


def _build_polymer_density_recovery_compaction_stages(
    *,
    warm_temp: float,
    compact_pressure_bar: float,
    warm_nvt_ns: float,
    compact_npt_ns: float,
    dt_ps: float,
) -> list[EqStage]:
    from ...gmx.mdp_templates import MINIM_STEEP_MDP, MdpSpec, NPT_NO_CONSTRAINTS_MDP, NVT_NO_CONSTRAINTS_MDP, default_mdp_params

    return [
        EqStage(
            "01_minim",
            "minim",
            MdpSpec(MINIM_STEEP_MDP, {**default_mdp_params(), "nsteps": 50000, "emtol": 1000.0, "emstep": 0.001}),
        ),
        EqStage(
            "02_warm_nvt",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                _no_constraint_md_params(
                    temperature=float(warm_temp),
                    pressure=None,
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(warm_nvt_ns), float(dt_ps)),
                    gen_vel="yes",
                    tau_t_ps=0.5,
                ),
            ),
        ),
        EqStage(
            "03_compact_npt",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                _no_constraint_md_params(
                    temperature=float(warm_temp),
                    pressure=float(compact_pressure_bar),
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(compact_npt_ns), float(dt_ps)),
                    gen_vel="no",
                    pcoupl="Berendsen",
                    tau_p_ps=8.0,
                    compressibility=4.5e-5 * 0.35,
                ),
            ),
        ),
    ]


def _build_polymer_density_recovery_release_stages(
    *,
    temp: float,
    press: float,
    final_ns: float,
    dt_ps: float,
    checkpoint_min: float,
) -> list[EqStage]:
    from ...gmx.mdp_templates import MdpSpec, NPT_NO_CONSTRAINTS_MDP

    return [
        EqStage(
            "04_final_npt",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                _no_constraint_md_params(
                    temperature=float(temp),
                    pressure=float(press),
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(final_ns), float(dt_ps)),
                    gen_vel="no",
                    pcoupl="C-rescale",
                    tau_p_ps=2.0,
                    compressibility=4.5e-5,
                ),
            ),
            checkpoint_minutes=float(checkpoint_min),
        )
    ]


def _build_polymer_chain_relaxation_stages(
    *,
    temp: float,
    press: float,
    final_ns: float,
    warm_temp: float,
    warm_nvt_ns: float,
    dt_ps: float,
    checkpoint_min: float,
) -> list[EqStage]:
    from ...gmx.mdp_templates import MdpSpec, NPT_NO_CONSTRAINTS_MDP, NVT_NO_CONSTRAINTS_MDP

    return [
        EqStage(
            "01_warm_nvt",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                _no_constraint_md_params(
                    temperature=float(warm_temp),
                    pressure=None,
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(warm_nvt_ns), float(dt_ps)),
                    gen_vel="yes",
                    tau_t_ps=0.5,
                ),
            ),
        ),
        EqStage(
            "02_final_npt",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                _no_constraint_md_params(
                    temperature=float(temp),
                    pressure=float(press),
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(final_ns), float(dt_ps)),
                    gen_vel="no",
                    pcoupl="C-rescale",
                    tau_p_ps=2.0,
                    compressibility=4.5e-5,
                ),
            ),
            checkpoint_minutes=float(checkpoint_min),
        ),
    ]


def _build_liquid_target_relaxation_stages(
    *,
    temp: float,
    press: float,
    final_ns: float,
    dt_ps: float,
    checkpoint_min: float = 5.0,
    mdp_overrides: Optional[dict[str, object]] = None,
) -> list[EqStage]:
    """Conservative target-temperature relaxation for already-compacted liquids.

    This is intentionally slower than the generic ``Additional`` schedule.  It is
    used after a liquid/electrolyte box is dense enough to be physically
    meaningful but still fails the plateau gate.  The early stages are
    unconstrained, cold, and strongly thermostatted so GROMACS does not have to
    survive a one-shot handoff from a rough structure to 1 fs dynamics.
    """

    from ...gmx.mdp_templates import MINIM_STEEP_MDP, MdpSpec, NPT_NO_CONSTRAINTS_MDP, NVT_NO_CONSTRAINTS_MDP, default_mdp_params

    target_temp = float(temp)
    cold_temp = float(max(80.0, min(120.0, 0.35 * target_temp)))
    warm_temp = float(max(cold_temp + 50.0, min(250.0, 0.70 * target_temp)))
    micro_dt = 0.0001
    target_nvt_dt = min(float(dt_ps), 0.0002)
    soft_npt_dt = min(float(dt_ps), 0.00025)
    final_dt = min(float(dt_ps), 0.0005)

    def params(
        *,
        temperature: float,
        pressure: Optional[float],
        step_dt: float,
        nsteps: int,
        gen_vel: str,
        continuation: str,
        tcoupl: str,
        tau_t: float,
        pcoupl: str = "C-rescale",
        tau_p: float = 5.0,
        compressibility: float = 4.5e-5,
    ) -> dict[str, object]:
        p = _no_constraint_md_params(
            temperature=float(temperature),
            pressure=pressure,
            dt_ps=float(step_dt),
            nsteps=int(nsteps),
            gen_vel=str(gen_vel),
            pcoupl=str(pcoupl),
            tau_t_ps=float(tau_t),
            tau_p_ps=float(tau_p),
            compressibility=float(compressibility),
        )
        p.update(
            {
                "continuation": str(continuation),
                "tcoupl": str(tcoupl),
                "nstenergy": 1000,
                "nstlog": 1000,
                "nstxout": 10000,
                "nstxout_trr": 10000,
                "nstvout": 10000,
            }
        )
        return p

    stages = [
        EqStage(
            "01_minim",
            "minim",
            MdpSpec(
                MINIM_STEEP_MDP,
                {
                    **default_mdp_params(),
                    "nsteps": 200000,
                    "emtol": 100.0,
                    "emstep": 0.0002,
                },
            ),
        ),
        EqStage(
            "02_cold_nvt_0p1fs",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                params(
                    temperature=cold_temp,
                    pressure=None,
                    step_dt=micro_dt,
                    nsteps=_ps_to_steps(2.0, micro_dt),
                    gen_vel="yes",
                    continuation="no",
                    tcoupl="Berendsen",
                    tau_t=0.02,
                ),
            ),
        ),
        EqStage(
            "03_warm_nvt_0p1fs",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                params(
                    temperature=warm_temp,
                    pressure=None,
                    step_dt=micro_dt,
                    nsteps=_ps_to_steps(3.0, micro_dt),
                    gen_vel="no",
                    continuation="yes",
                    tcoupl="Berendsen",
                    tau_t=0.02,
                ),
            ),
        ),
        EqStage(
            "04_target_nvt_0p2fs",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                params(
                    temperature=target_temp,
                    pressure=None,
                    step_dt=target_nvt_dt,
                    nsteps=_ps_to_steps(10.0, target_nvt_dt),
                    gen_vel="no",
                    continuation="yes",
                    tcoupl="Berendsen",
                    tau_t=0.05,
                ),
            ),
        ),
        EqStage(
            "05_soft_npt_0p25fs",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                params(
                    temperature=target_temp,
                    pressure=float(press),
                    step_dt=soft_npt_dt,
                    nsteps=_ps_to_steps(20.0, soft_npt_dt),
                    gen_vel="no",
                    continuation="yes",
                    tcoupl="Berendsen",
                    tau_t=0.05,
                    pcoupl="Berendsen",
                    tau_p=12.0,
                    compressibility=1.0e-5,
                ),
            ),
        ),
        EqStage(
            "06_final_npt",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                params(
                    temperature=target_temp,
                    pressure=float(press),
                    step_dt=final_dt,
                    nsteps=_ns_to_steps_for_dt(float(final_ns), final_dt),
                    gen_vel="no",
                    continuation="yes",
                    tcoupl="V-rescale",
                    tau_t=0.1,
                    pcoupl="C-rescale",
                    tau_p=5.0,
                    compressibility=4.5e-5,
                ),
            ),
            checkpoint_minutes=float(checkpoint_min),
        ),
    ]

    return _apply_stage_mdp_overrides(stages, mdp_overrides, stage_kinds=("npt", "md"))


def _apply_stage_mdp_overrides(stages: Sequence[EqStage], overrides: Optional[dict[str, object]], *, stage_kinds: Sequence[str] = ("npt", "md")) -> list[EqStage]:
    from ...gmx.mdp_templates import MdpSpec

    if not overrides:
        return list(stages)
    target_kinds = {str(kind) for kind in stage_kinds}
    patched: list[EqStage] = []
    for stage in stages:
        if stage.kind in target_kinds:
            patched.append(
                EqStage(
                    stage.name,
                    stage.kind,
                    MdpSpec(stage.mdp.template, {**stage.mdp.params, **dict(overrides)}),
                    lincs_retry=stage.lincs_retry,
                    checkpoint_minutes=stage.checkpoint_minutes,
                    strict_constraints=bool(getattr(stage, "strict_constraints", False)),
                )
            )
        else:
            patched.append(stage)
    return patched

def _eq21_stage_time_offsets(records: list[dict[str, Any]]) -> dict[str, float]:
    offsets: dict[str, float] = {}
    current = 0.0
    for rec in records:
        offsets[str(rec['folder'])] = current
        if rec.get('time_ps') is not None:
            current += float(rec['time_ps'])
    return offsets


def _eq21_load_box_series(stage_dir: Path):
    try:
        import numpy as np
        import mdtraj as md

        xtc = stage_dir / 'md.xtc'
        gro = stage_dir / 'md.gro'
        if not xtc.exists() or not gro.exists():
            return None
        traj = md.load(str(xtc), top=str(gro))
        if traj.n_frames == 0:
            return None
        return {
            'time_ps': np.asarray(traj.time, dtype=float),
            'lengths_nm': np.asarray(traj.unitcell_lengths, dtype=float),
            'angles_deg': np.asarray(traj.unitcell_angles, dtype=float),
        }
    except Exception:
        return None


def _eq21_load_thermo_series(stage_dir: Path):
    try:
        import numpy as np
        from ...gmx.analysis.xvg import read_xvg

        xvg = stage_dir / 'thermo.xvg'
        if not xvg.exists():
            return None
        df = read_xvg(xvg).df
        if 'x' not in df.columns:
            return None
        out = {'time_ps': np.asarray(df['x'].to_numpy(dtype=float), dtype=float)}
        for col in ('Density', 'Volume', 'Total Energy', 'Box-X', 'Box-Y', 'Box-Z'):
            if col in df.columns:
                out[col] = np.asarray(df[col].to_numpy(dtype=float), dtype=float)
        return out
    except Exception:
        return None


def _tail_linear_trend(t_ps: np.ndarray, y: np.ndarray, *, tail_frac: float = 0.35) -> dict[str, object]:
    try:
        t = np.asarray(t_ps, dtype=float)
        yv = np.asarray(y, dtype=float)
        mask = np.isfinite(t) & np.isfinite(yv)
        t = t[mask]
        yv = yv[mask]
        if t.size < 8 or yv.size < 8:
            return {"ok": False, "reason": "series_too_short"}
        start = max(0, int(np.floor((1.0 - float(tail_frac)) * t.size)))
        tt = t[start:]
        yy = yv[start:]
        if tt.size < 5:
            return {"ok": False, "reason": "tail_too_short"}
        A = np.vstack([tt, np.ones_like(tt)]).T
        slope, intercept = np.linalg.lstsq(A, yy, rcond=None)[0]
        half = max(1, yy.size // 2)
        first_mean = float(np.mean(yy[:half]))
        second_mean = float(np.mean(yy[-half:]))
        mean = float(np.mean(yy))
        delta = float(second_mean - first_mean)
        return {
            "ok": True,
            "slope_per_ps": float(slope),
            "slope_rel_per_ps": float(slope / abs(mean)) if mean != 0.0 else float("inf"),
            "intercept": float(intercept),
            "tail_delta": delta,
            "tail_rel_delta": float(delta / abs(mean)) if mean != 0.0 else float("inf"),
            "tail_start_ps": float(tt[0]),
            "tail_end_ps": float(tt[-1]),
            "tail_mean": mean,
            "tail_rel_std": float(np.std(yy) / abs(mean)) if mean != 0.0 else float("inf"),
        }
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def _eq21_density_trend(stage_dir: Path, *, tail_frac: float = 0.35) -> dict[str, object]:
    """Return density trend diagnostics for an EQ21/extension stage."""

    data = _eq21_load_thermo_series(stage_dir)
    if not isinstance(data, dict) or "Density" not in data:
        return {"ok": False, "reason": "density_series_unavailable"}
    trend = _tail_linear_trend(np.asarray(data.get("time_ps"), dtype=float), np.asarray(data.get("Density"), dtype=float), tail_frac=tail_frac)
    if bool(trend.get("ok")):
        trend["tail_delta_kg_m3"] = float(trend.get("tail_delta") or 0.0)
        trend["tail_mean_kg_m3"] = float(trend.get("tail_mean") or 0.0)
    return trend


def _eq21_box_volume_trend(stage_dir: Path, *, tail_frac: float = 0.35) -> dict[str, object]:
    """Return box-volume trend diagnostics for an EQ21/extension stage."""

    data = _eq21_load_thermo_series(stage_dir)
    if not isinstance(data, dict):
        return {"ok": False, "reason": "thermo_series_unavailable"}
    t = np.asarray(data.get("time_ps"), dtype=float)
    if "Volume" in data:
        y = np.asarray(data.get("Volume"), dtype=float)
    elif all(k in data for k in ("Box-X", "Box-Y", "Box-Z")):
        y = np.asarray(data["Box-X"], dtype=float) * np.asarray(data["Box-Y"], dtype=float) * np.asarray(data["Box-Z"], dtype=float)
    else:
        return {"ok": False, "reason": "volume_series_unavailable"}
    return _tail_linear_trend(t, y, tail_frac=tail_frac)


def _compression_state_from_trends(
    *,
    density_trend: dict[str, object] | None,
    volume_trend: dict[str, object] | None,
    density_slope_threshold_per_ps: float,
    density_delta_threshold_kg_m3: float,
    volume_rel_slope_threshold_per_ps: float = 5.0e-5,
    volume_rel_delta_threshold: float = 2.0e-3,
) -> dict[str, object]:
    density_trend = density_trend if isinstance(density_trend, dict) else {}
    volume_trend = volume_trend if isinstance(volume_trend, dict) else {}
    density_still_compressing = False
    if bool(density_trend.get("ok")):
        density_still_compressing = bool(
            float(density_trend.get("slope_per_ps") or 0.0) > float(density_slope_threshold_per_ps)
            and float(density_trend.get("tail_delta_kg_m3", density_trend.get("tail_delta") or 0.0)) > float(density_delta_threshold_kg_m3)
        )
    volume_still_compressing = False
    if bool(volume_trend.get("ok")):
        volume_still_compressing = bool(
            float(volume_trend.get("slope_rel_per_ps") or 0.0) < -float(volume_rel_slope_threshold_per_ps)
            and float(volume_trend.get("tail_rel_delta") or 0.0) < -float(volume_rel_delta_threshold)
        )
    return {
        "still_compressing": bool(density_still_compressing or volume_still_compressing),
        "density_still_compressing": bool(density_still_compressing),
        "box_volume_still_compressing": bool(volume_still_compressing),
        "criteria": {
            "density_slope_threshold_per_ps": float(density_slope_threshold_per_ps),
            "density_delta_threshold_kg_m3": float(density_delta_threshold_kg_m3),
            "volume_rel_slope_threshold_per_ps": float(volume_rel_slope_threshold_per_ps),
            "volume_rel_delta_threshold": float(volume_rel_delta_threshold),
        },
    }


def _run_eq21_final_density_extensions(
    *,
    work_dir: Path,
    exp: SystemExportResult,
    initial_job: EquilibrationJob,
    resources: RunResources,
    temp: float,
    press: float,
    dt_ps: float,
    tau_p_ps: float,
    compressibility: float,
    max_rounds: int,
    round_ns: float,
    slope_threshold_per_ps: float,
    delta_threshold_kg_m3: float,
    restart: bool,
) -> tuple[EquilibrationJob, list[dict[str, object]]]:
    """Extend EQ21 final NPT while density is still clearly increasing."""

    from ...gmx.mdp_templates import NPT_NO_CONSTRAINTS_MDP, MdpSpec, default_mdp_params

    current_job = initial_job
    history: list[dict[str, object]] = []
    if int(max_rounds) <= 0 or float(round_ns) <= 0.0:
        return current_job, history
    if not (hasattr(current_job, "final_stage_dir") and hasattr(current_job, "final_gro")):
        return current_job, history

    ext_root = Path(work_dir) / "03_EQ21" / "final_extend"
    ext_root.mkdir(parents=True, exist_ok=True)
    for round_idx in range(int(max_rounds)):
        stage_dir = current_job.final_stage_dir()
        trend = _eq21_density_trend(stage_dir)
        volume_trend = _eq21_box_volume_trend(stage_dir)
        compression_state = _compression_state_from_trends(
            density_trend=trend,
            volume_trend=volume_trend,
            density_slope_threshold_per_ps=float(slope_threshold_per_ps),
            density_delta_threshold_kg_m3=float(delta_threshold_kg_m3),
        )
        should_extend = bool(compression_state.get("still_compressing"))
        rec = {
            "round": int(round_idx),
            "stage_dir": str(stage_dir),
            "density_trend": trend,
            "box_volume_trend": volume_trend,
            "compression_state": compression_state,
            "extended": should_extend,
        }
        history.append(rec)
        if not should_extend:
            break

        p = default_mdp_params()
        p.update(
            {
                "dt": float(dt_ps),
                "nsteps": _ns_to_steps_for_dt(float(round_ns), float(dt_ps)),
                "ref_t": float(temp),
                "gen_temp": float(temp),
                "gen_vel": "no",
                "gen_seed": -1,
                "pcoupl": "C-rescale",
                "tau_p": float(tau_p_ps),
                "ref_p": float(press),
                "compressibility": float(compressibility),
                "constraints": "none",
                "constraint_algorithm": "none",
            }
        )
        round_dir = ext_root / f"round_{round_idx:02d}"
        if not restart and round_dir.exists():
            shutil.rmtree(round_dir, ignore_errors=True)
        ext_job = EquilibrationJob(
            gro=current_job.final_gro(),
            top=exp.system_top,
            provenance_ndx=exp.system_ndx,
            out_dir=round_dir,
            stages=[
                EqStage(
                    "01_npt",
                    "npt",
                    MdpSpec(NPT_NO_CONSTRAINTS_MDP, p),
                )
            ],
            resources=resources,
        )
        _preset_log(
            "[EQ21] Final NPT box still compressing; extending "
            f"round={round_idx} | slope={float(trend.get('slope_per_ps') or 0.0):.4g} kg/m^3/ps | "
            f"tail_delta={float(trend.get('tail_delta_kg_m3') or 0.0):.4g} kg/m^3 | "
            f"volume_rel_delta={float(volume_trend.get('tail_rel_delta') or 0.0):.4g} | "
            f"time={float(round_ns):.3f} ns",
            level=1,
        )
        ext_job.run(restart=bool(restart))
        current_job = ext_job

    try:
        (ext_root / "final_density_extension.json").write_text(
            json.dumps({"history": history}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    return current_job, history


def _run_liquid_compaction_extensions(
    *,
    ext_root: Path,
    exp: SystemExportResult,
    initial_job: EquilibrationJob,
    resources: RunResources,
    hot_temp: float,
    compact_pressure_bar: float,
    hot_dt_ps: float,
    constraints: str,
    lincs_iter: Optional[int],
    lincs_order: Optional[int],
    max_rounds: int,
    round_ns: float,
    slope_threshold_per_ps: float,
    delta_threshold_kg_m3: float,
    restart: bool,
) -> tuple[EquilibrationJob, list[dict[str, object]]]:
    """Continue high-pressure liquid compaction while density is still rising.

    This deliberately uses density-trend diagnostics rather than an absolute
    density floor. Some legitimate systems are light, but a low-density box that
    is still compressing has not finished returning from the packing vacuum.
    """

    from ...gmx.mdp_templates import MdpSpec

    current_job = initial_job
    history: list[dict[str, object]] = []
    if int(max_rounds) <= 0 or float(round_ns) <= 0.0:
        return current_job, history
    if not (hasattr(current_job, "final_stage_dir") and hasattr(current_job, "final_gro")):
        return current_job, history

    constraints_mode, use_lincs, effective_lincs_iter, effective_lincs_order, _nvt_template, npt_template = _liquid_templates_and_controls(
        constraints=constraints,
        lincs_iter=lincs_iter,
        lincs_order=lincs_order,
    )
    retry = StageLincsRetryPolicy() if use_lincs else None
    ext_root = Path(ext_root)
    ext_root.mkdir(parents=True, exist_ok=True)

    for round_idx in range(int(max_rounds)):
        stage_dir = current_job.final_stage_dir()
        trend = _eq21_density_trend(stage_dir)
        volume_trend = _eq21_box_volume_trend(stage_dir)
        compression_state = _compression_state_from_trends(
            density_trend=trend,
            volume_trend=volume_trend,
            density_slope_threshold_per_ps=float(slope_threshold_per_ps),
            density_delta_threshold_kg_m3=float(delta_threshold_kg_m3),
        )
        should_extend = bool(compression_state.get("still_compressing"))
        rec = {
            "round": int(round_idx),
            "stage_dir": str(stage_dir),
            "density_trend": trend,
            "box_volume_trend": volume_trend,
            "compression_state": compression_state,
            "extended": should_extend,
        }
        history.append(rec)
        if not should_extend:
            break

        round_dir = ext_root / f"round_{round_idx:02d}"
        if not restart and round_dir.exists():
            shutil.rmtree(round_dir, ignore_errors=True)
        p = _liquid_md_params(
            constraints_mode=constraints_mode,
            use_lincs=use_lincs,
            lincs_iter=effective_lincs_iter,
            lincs_order=effective_lincs_order,
            temperature=float(hot_temp),
            pressure=float(compact_pressure_bar),
            dt_ps=float(hot_dt_ps),
            nsteps=_ns_to_steps_for_dt(float(round_ns), float(hot_dt_ps)),
            gen_vel="no",
            pcoupl="Berendsen",
            tau_p_ps=2.0,
            compressibility=4.5e-5,
        )
        ext_job = EquilibrationJob(
            gro=current_job.final_gro(),
            top=exp.system_top,
            provenance_ndx=exp.system_ndx,
            out_dir=round_dir,
            stages=[EqStage("01_compact_npt", "npt", MdpSpec(npt_template, p), lincs_retry=retry)],
            resources=resources,
        )
        _preset_log(
            "[LIQUID-RECOVERY] High-pressure compaction box still compressing; extending "
            f"round={round_idx} | slope={float(trend.get('slope_per_ps') or 0.0):.4g} kg/m^3/ps | "
            f"tail_delta={float(trend.get('tail_delta_kg_m3') or 0.0):.4g} kg/m^3 | "
            f"volume_rel_delta={float(volume_trend.get('tail_rel_delta') or 0.0):.4g} | "
            f"P={float(compact_pressure_bar):.1f} bar | time={float(round_ns):.3f} ns",
            level=1,
        )
        ext_job.run(restart=bool(restart))
        current_job = ext_job

    try:
        (ext_root / "liquid_compaction_extension.json").write_text(
            json.dumps(
                {
                    "criterion": {
                        "slope_threshold_per_ps": float(slope_threshold_per_ps),
                        "delta_threshold_kg_m3": float(delta_threshold_kg_m3),
                    },
                    "history": history,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    return current_job, history


def _run_polymer_compaction_extensions(
    *,
    ext_root: Path,
    exp: SystemExportResult,
    initial_job: EquilibrationJob,
    resources: RunResources,
    warm_temp: float,
    compact_pressure_bar: float,
    dt_ps: float,
    max_rounds: int,
    round_ns: float,
    slope_threshold_per_ps: float,
    delta_threshold_kg_m3: float,
    restart: bool,
) -> tuple[EquilibrationJob, list[dict[str, object]]]:
    from ...gmx.mdp_templates import MdpSpec, NPT_NO_CONSTRAINTS_MDP

    current_job = initial_job
    history: list[dict[str, object]] = []
    if int(max_rounds) <= 0 or float(round_ns) <= 0.0:
        return current_job, history
    if not (hasattr(current_job, "final_stage_dir") and hasattr(current_job, "final_gro")):
        return current_job, history

    ext_root = Path(ext_root)
    ext_root.mkdir(parents=True, exist_ok=True)
    for round_idx in range(int(max_rounds)):
        stage_dir = current_job.final_stage_dir()
        trend = _eq21_density_trend(stage_dir)
        volume_trend = _eq21_box_volume_trend(stage_dir)
        compression_state = _compression_state_from_trends(
            density_trend=trend,
            volume_trend=volume_trend,
            density_slope_threshold_per_ps=float(slope_threshold_per_ps),
            density_delta_threshold_kg_m3=float(delta_threshold_kg_m3),
        )
        should_extend = bool(compression_state.get("still_compressing"))
        rec = {
            "round": int(round_idx),
            "stage_dir": str(stage_dir),
            "density_trend": trend,
            "box_volume_trend": volume_trend,
            "compression_state": compression_state,
            "extended": should_extend,
        }
        history.append(rec)
        if not should_extend:
            break

        round_dir = ext_root / f"round_{round_idx:02d}"
        if not restart and round_dir.exists():
            shutil.rmtree(round_dir, ignore_errors=True)
        p = _no_constraint_md_params(
            temperature=float(warm_temp),
            pressure=float(compact_pressure_bar),
            dt_ps=float(dt_ps),
            nsteps=_ns_to_steps_for_dt(float(round_ns), float(dt_ps)),
            gen_vel="no",
            pcoupl="Berendsen",
            tau_p_ps=8.0,
            compressibility=4.5e-5 * 0.35,
        )
        ext_job = EquilibrationJob(
            gro=current_job.final_gro(),
            top=exp.system_top,
            provenance_ndx=exp.system_ndx,
            out_dir=round_dir,
            stages=[EqStage("01_compact_npt", "npt", MdpSpec(NPT_NO_CONSTRAINTS_MDP, p))],
            resources=resources,
        )
        _preset_log(
            "[POLYMER-RECOVERY] Conservative compaction box still compressing; extending "
            f"round={round_idx} | slope={float(trend.get('slope_per_ps') or 0.0):.4g} kg/m^3/ps | "
            f"tail_delta={float(trend.get('tail_delta_kg_m3') or 0.0):.4g} kg/m^3 | "
            f"volume_rel_delta={float(volume_trend.get('tail_rel_delta') or 0.0):.4g} | "
            f"P={float(compact_pressure_bar):.1f} bar | time={float(round_ns):.3f} ns",
            level=1,
        )
        ext_job.run(restart=bool(restart))
        current_job = ext_job

    try:
        (ext_root / "polymer_compaction_extension.json").write_text(
            json.dumps(
                {
                    "criterion": {
                        "slope_threshold_per_ps": float(slope_threshold_per_ps),
                        "delta_threshold_kg_m3": float(delta_threshold_kg_m3),
                    },
                    "history": history,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    return current_job, history


def _write_eq21_overview_plot(run_dir: Path, records: list[dict[str, Any]], params: dict[str, Any]) -> Optional[Path]:
    try:
        import numpy as np
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        from ...plotting.style import apply_matplotlib_style, golden_figsize
        from ...plotting.legend import place_legend

        offsets = _eq21_stage_time_offsets(records)
        thermo_t = []
        density = []
        energy = []
        length_t = []
        lengths = []
        angle_t = []
        angles = []
        boundaries = []
        labels = []

        current_end = 0.0
        for rec in records:
            folder = str(rec['folder'])
            stage_dir = run_dir / folder
            offset = offsets.get(folder, current_end)
            labels.append((offset, str(rec.get('display_name') or folder)))
            thermo = _eq21_load_thermo_series(stage_dir)
            if thermo is not None and thermo.get('time_ps') is not None and len(thermo['time_ps']) > 0:
                t = np.asarray(thermo['time_ps'], dtype=float) + float(offset)
                if 'Density' in thermo:
                    thermo_t.append(t)
                    density.append(np.asarray(thermo['Density'], dtype=float))
                if 'Total Energy' in thermo:
                    energy.append((t, np.asarray(thermo['Total Energy'], dtype=float)))
                current_end = max(current_end, float(t[-1]))
            box = _eq21_load_box_series(stage_dir)
            if box is not None and box.get('time_ps') is not None and len(box['time_ps']) > 0:
                t_box = np.asarray(box['time_ps'], dtype=float) + float(offset)
                length_t.append(t_box)
                lengths.append(np.asarray(box['lengths_nm'], dtype=float))
                angle_t.append(t_box)
                angles.append(np.asarray(box['angles_deg'], dtype=float))
                current_end = max(current_end, float(t_box[-1]))
            if rec.get('time_ps') is not None:
                boundaries.append(float(offset) + float(rec['time_ps']))

        if not density and not energy and not lengths and not angles:
            return None

        apply_matplotlib_style()
        fig, axes = plt.subplots(4, 1, figsize=golden_figsize(12.0), sharex=True)
        fig.suptitle(
            f"EQ21 overview | Tmax={params['t_max_k']:.0f} K | Tanal={params['t_anneal_k']:.0f} K | "
            f"Pmax={params['p_max_bar']:.0f} bar | Panal={params['p_anneal_bar']:.3f} bar",
            fontsize=12,
        )

        if thermo_t and density:
            for t, y in zip(thermo_t, density):
                axes[0].plot(t / 1000.0, y, linewidth=1.2)
        axes[0].set_ylabel('Density (kg/m$^3$)')
        axes[0].set_title('Density vs time')
        axes[0].grid(True, alpha=0.25)

        if lengths:
            first = True
            for t_box, arr in zip(length_t, lengths):
                lab = ('a', 'b', 'c') if first else (None, None, None)
                axes[1].plot(t_box / 1000.0, arr[:, 0], linewidth=1.1, label=lab[0])
                axes[1].plot(t_box / 1000.0, arr[:, 1], linewidth=1.1, label=lab[1])
                axes[1].plot(t_box / 1000.0, arr[:, 2], linewidth=1.1, label=lab[2])
                first = False
        axes[1].set_ylabel('Box length (nm)')
        axes[1].set_title('Box lengths vs time')
        axes[1].grid(True, alpha=0.25)
        place_legend(axes[1])

        if angles:
            first = True
            for t_box, arr in zip(angle_t, angles):
                lab = ('alpha', 'beta', 'gamma') if first else (None, None, None)
                axes[2].plot(t_box / 1000.0, arr[:, 0], linewidth=1.1, label=lab[0])
                axes[2].plot(t_box / 1000.0, arr[:, 1], linewidth=1.1, label=lab[1])
                axes[2].plot(t_box / 1000.0, arr[:, 2], linewidth=1.1, label=lab[2])
                first = False
        axes[2].set_ylabel('Angle (deg)')
        axes[2].set_title('Box angles vs time')
        axes[2].grid(True, alpha=0.25)
        place_legend(axes[2])

        if energy:
            for t, y in energy:
                axes[3].plot(t / 1000.0, y, linewidth=1.2)
        axes[3].set_ylabel('Total energy (kJ/mol)')
        axes[3].set_title('Total energy vs time')
        axes[3].set_xlabel('Time (ns)')
        axes[3].grid(True, alpha=0.25)

        x_bounds = sorted(set(b for b in boundaries if b > 0.0))
        for ax in axes:
            for b in x_bounds[:-1]:
                ax.axvline(b / 1000.0, linestyle='--', linewidth=0.7, alpha=0.25)
        # annotate only top-level folders and every 5th EQ step to avoid clutter
        for start_ps, label in labels:
            if label in ('01_em', '02_preNVT') or label.endswith('step_01') or label.endswith('step_05') or label.endswith('step_10') or label.endswith('step_15') or label.endswith('step_20') or label.endswith('step_21'):
                axes[0].text(start_ps / 1000.0, 0.98, label, transform=axes[0].get_xaxis_transform(), fontsize=7, va='top', ha='left')

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = run_dir / 'eq21_overview.svg'
        fig.savefig(out, format='svg')
        plt.close(fig)
        return out
    except Exception:
        return None

@dataclass
class EQ21step:
    """YadonPy-style equilibration preset wrapper (GROMACS engine)."""

    ac: object
    work_dir: Union[str, Path] = "./"
    ff_name: str = "gaff2_mod"
    charge_method: str = "RESP"
    system_name: str = "system"
    include_h_atomtypes: bool = False
    charge_scale: Optional[Any] = None

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Two system exports are maintained (organized under work_dir/02_system):
        #   - scaled (production): work_dir/02_system/
        #   - raw (no scaling, reference): work_dir/02_system/01_raw_non_scaled/
        # The workflow runs on the scaled system by default.
        self._export: Optional[SystemExportResult] = None  # scaled
        self._export_raw: Optional[SystemExportResult] = None
        self._job: Optional[EquilibrationJob] = None
        self._resume = ResumeManager(self.work_dir, enabled=True, strict_inputs=True)

    def _load_export_from_disk(self, sys_dir: Path) -> SystemExportResult:
        meta_path = sys_dir / "system_meta.json"
        box_nm = 0.0
        box_lengths_nm = None
        species: list[dict] = []
        if meta_path.exists():
            try:
                import json

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                raw_box = meta.get("box_lengths_nm", meta.get("box_nm"))
                if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 3:
                    box_lengths_nm = (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]))
                    box_nm = float(max(box_lengths_nm))
                else:
                    box_nm = float(meta.get("box_nm") or 0.0)
                    if box_nm > 0.0:
                        box_lengths_nm = (box_nm, box_nm, box_nm)
                species = list(meta.get("species") or [])
            except Exception:
                pass
        return SystemExportResult(
            system_gro=sys_dir / "system.gro",
            system_top=sys_dir / "system.top",
            system_ndx=sys_dir / "system.ndx",
            molecules_dir=sys_dir / "molecules",
            system_meta=meta_path,
            box_nm=box_nm,
            species=species,
            box_lengths_nm=box_lengths_nm,
        )

    def _ensure_system_exported(self) -> SystemExportResult:
        if self._export is not None and self._export_raw is not None:
            return self._export

        # Keep the exported GROMACS system in a single, sortable module folder.
        # - Scaled (production) system lives at: work_dir/02_system/
        # - Raw/unscaled system is stored under: work_dir/02_system/01_raw_non_scaled/
        sys_root = self.work_dir / "02_system"
        raw_dir = sys_root / "01_raw_non_scaled"

        spec = StepSpec(
            name="export_system",
            outputs=[
                sys_root / "system.gro",
                sys_root / "system.top",
                sys_root / "system.ndx",
                sys_root / "system_meta.json",
                sys_root / "export_manifest.json",
                raw_dir / "system.gro",
                raw_dir / "system.top",
                raw_dir / "system.ndx",
                raw_dir / "system_meta.json",
                raw_dir / "export_manifest.json",
            ],
            inputs={
                "export_schema_version": _EXPORT_SYSTEM_SCHEMA_VERSION,
                "ff_name": self.ff_name,
                "charge_method": self.charge_method,
                "charge_scale": str(self.charge_scale),
                "include_h_atomtypes": bool(self.include_h_atomtypes),
                "cell_signature": _cell_resume_signature(self.ac),
            },
            description="Export mixed system into GROMACS gro/top/ndx (scaled + raw)",
        )

        def _run_pair() -> tuple[SystemExportResult, SystemExportResult]:
            exp_raw_local = export_system_from_cell_meta(
                cell_mol=self.ac,
                out_dir=raw_dir,
                ff_name=self.ff_name,
                charge_method=self.charge_method,
                charge_scale=1.0,
                include_h_atomtypes=self.include_h_atomtypes,
                write_system_mol2=False,
            )
            exp_scaled_local = export_system_from_cell_meta(
                cell_mol=self.ac,
                out_dir=sys_root,
                ff_name=self.ff_name,
                charge_method=self.charge_method,
                charge_scale=self.charge_scale,
                include_h_atomtypes=self.include_h_atomtypes,
                source_molecules_dir=exp_raw_local.molecules_dir,
                system_gro_template=exp_raw_local.system_gro,
                system_ndx_template=exp_raw_local.system_ndx,
                write_system_mol2=False,
            )
            raw_issues = validate_exported_system_dir(raw_dir)
            scaled_issues = validate_exported_system_dir(sys_root)
            _write_export_manifest(
                raw_dir,
                schema_version=_EXPORT_SYSTEM_SCHEMA_VERSION,
                ff_name=self.ff_name,
                charge_method=self.charge_method,
                charge_scale=1.0,
                issues=raw_issues,
            )
            _write_export_manifest(
                sys_root,
                schema_version=_EXPORT_SYSTEM_SCHEMA_VERSION,
                ff_name=self.ff_name,
                charge_method=self.charge_method,
                charge_scale=self.charge_scale,
                issues=scaled_issues,
            )
            issues = raw_issues + scaled_issues
            if issues:
                raise RuntimeError(f"Invalid exported GROMACS topology: {'; '.join(issues)}")
            return exp_raw_local, exp_scaled_local

        if self._resume.is_done(spec):
            exp_scaled = self._load_export_from_disk(sys_root)
            exp_raw = self._load_export_from_disk(raw_dir)
            cached_issues = validate_exported_system_dir(raw_dir) + validate_exported_system_dir(sys_root)
            if cached_issues:
                _preset_log(
                    f"[RESTART] Invalid cached 02_system export detected; rebuilding exports. issues={' | '.join(cached_issues)}",
                    level=1,
                )
                exp_raw, exp_scaled = _run_pair()
                self._resume.mark_done(spec, meta={"schema_version": _EXPORT_SYSTEM_SCHEMA_VERSION, "recovered_from_invalid_export": True})
        else:
            exp_raw, exp_scaled = self._resume.run(spec, _run_pair, meta={"schema_version": _EXPORT_SYSTEM_SCHEMA_VERSION})

            # Note: system-level MOL2 export is handled inside export_system_from_cell_meta
            # (best-effort, via ParmEd). We keep EQ presets lean here.


        self._export_raw = exp_raw
        self._export = exp_scaled
        return exp_scaled

    def ensure_system_exported(self) -> SystemExportResult:
        return self._ensure_system_exported()

    def _job_restart_flag(self, spec: StepSpec, requested_restart: bool) -> bool:
        rst_flag = bool(requested_restart)
        if not rst_flag:
            return False
        if self._resume.needs_fresh_run(spec):
            _preset_log(
                f"[RESTART] {spec.name} inputs changed or cached outputs are stale; rebuilding stage workflow from scratch.",
                level=1,
            )
            return False
        return True

    def exec(
        self,
        *,
        temp: float,
        press: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        sim_time: Optional[float] = None,
        time: Optional[float] = None,
        charge_scale: Optional[Any] = None,
        restart: Optional[bool] = None,
        eq21_tmax: float = 1000.0,
        eq21_tanal: Optional[float] = None,
        eq21_pmax: float = 50000.0,
        eq21_panal: Optional[float] = None,
        eq21_dt_ps: float = 0.001,
        eq21_pre_nvt_ps: float = 10.0,
        eq21_tau_t_ps: float = 0.1,
        eq21_tau_p_ps: float = 1.0,
        eq21_barostat: str = "C-rescale",
        eq21_compressibility: float = 4.5e-5,
        eq21_robust: bool = True,
        eq21_stage_reseed: Optional[bool] = None,
        eq21_npt_time_scale: float = 2.0,
        eq21_npt_mdp_overrides: Optional[dict[str, object]] = None,
        eq21_final_extend: bool = True,
        eq21_final_extend_max_rounds: int = 4,
        eq21_final_extend_ns: float = 0.2,
        eq21_final_extend_density_slope_per_ps: float = 5.0e-2,
        eq21_final_extend_density_delta_kg_m3: float = 2.0,
    ):
        """Run the EQ21 multi-stage equilibration.

        The new EQ21 layout is:
          work_dir/03_EQ21/01_em
          work_dir/03_EQ21/02_preNVT
          work_dir/03_EQ21/03_EQ21/step_01 ... step_21

        A schedule table is written to ``03_EQ21/eq21_schedule.(csv|json|md)`` before the run.
        """
        t_all = _preset_section("EQ21 equilibration preset", detail=f"restart={bool(resolve_restart(restart))} | work_dir={self.work_dir}")
        rst_flag = resolve_restart(restart)
        self._resume.enabled = bool(rst_flag)

        if charge_scale is not None and str(charge_scale) != str(self.charge_scale):
            self.charge_scale = charge_scale
            self._export = None
            self._export_raw = None

        exp = self._ensure_system_exported()

        run_dir = self.work_dir / "03_EQ21"
        if not rst_flag and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        final_ns = 0.8
        if sim_time is not None:
            final_ns = float(sim_time)
        if time is not None:
            final_ns = float(time)

        cfg = EQ21ProtocolConfig(
            t_max_k=float(eq21_tmax),
            t_anneal_k=float(temp if eq21_tanal is None else eq21_tanal),
            p_max_bar=float(eq21_pmax),
            p_anneal_bar=float(press if eq21_panal is None else eq21_panal),
            dt_ps=float(eq21_dt_ps),
            pre_nvt_ps=float(eq21_pre_nvt_ps),
            tau_t_ps=float(eq21_tau_t_ps),
            tau_p_ps=float(eq21_tau_p_ps),
            compressibility_bar_inv=float(eq21_compressibility),
            barostat=str(eq21_barostat),
            robust=bool(eq21_robust),
            reseed_each_stage=(bool(eq21_stage_reseed) if eq21_stage_reseed is not None else (not bool(eq21_robust))),
            npt_time_scale=float(eq21_npt_time_scale),
        )
        stages, stage_records, params = _build_eq21_stages(
            temp=float(temp if eq21_tanal is None else eq21_tanal),
            press=float(press if eq21_panal is None else eq21_panal),
            final_ns=float(final_ns),
            cfg=cfg,
        )
        stages = _apply_stage_mdp_overrides(stages, eq21_npt_mdp_overrides)
        _write_eq21_schedule(run_dir, stage_records, params)
        _preset_item("run_dir", run_dir)
        _preset_item("temperature_K", float(temp))
        _preset_item("pressure_bar", float(press))
        _preset_item("final_stage_ns", float(final_ns))
        if eq21_npt_mdp_overrides:
            _preset_item("eq21_npt_mdp_overrides", eq21_npt_mdp_overrides)
        _preset_item(
            "eq21_final_extend",
            f"{bool(eq21_final_extend)} | max_rounds={int(eq21_final_extend_max_rounds)} | "
            f"round_ns={float(eq21_final_extend_ns):.3f} | "
            f"slope>{float(eq21_final_extend_density_slope_per_ps):.4g} kg/m^3/ps | "
            f"delta>{float(eq21_final_extend_density_delta_kg_m3):.4g} kg/m^3",
        )
        _preset_item("resources", f"mpi={int(mpi)} | omp={int(omp)} | gpu={int(gpu)} | gpu_id={gpu_id}")
        _print_eq21_schedule(stage_records, params)

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid)
        job = EquilibrationJob(gro=exp.system_gro, top=exp.system_top, provenance_ndx=exp.system_ndx, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name if stages else run_dir
        eq_spec = StepSpec(
            name="equilibration_eq21",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "input_gro_sig": file_signature(exp.system_gro),
                "input_top_sig": file_signature(exp.system_top),
                "temp": float(temp),
                "press": float(press),
                "final_ns": float(final_ns),
                "eq21_tmax": float(cfg.t_max_k),
                "eq21_tanal": float(cfg.t_anneal_k),
                "eq21_pmax": float(cfg.p_max_bar),
                "eq21_panal": float(cfg.p_anneal_bar),
                "eq21_dt_ps": float(cfg.dt_ps),
                "eq21_pre_nvt_ps": float(cfg.pre_nvt_ps),
                "eq21_tau_t_ps": float(cfg.tau_t_ps),
                "eq21_tau_p_ps": float(cfg.tau_p_ps),
                "eq21_barostat": str(cfg.barostat),
                "eq21_compressibility": float(cfg.compressibility_bar_inv),
                "eq21_robust": bool(cfg.robust),
                "eq21_stage_reseed": bool(cfg.reseed_each_stage),
                "eq21_npt_time_scale": float(cfg.npt_time_scale),
                "eq21_npt_mdp_overrides": json.dumps(eq21_npt_mdp_overrides, sort_keys=True, default=str) if eq21_npt_mdp_overrides else None,
                "eq21_final_extend": bool(eq21_final_extend),
                "eq21_final_extend_max_rounds": int(eq21_final_extend_max_rounds),
                "eq21_final_extend_ns": float(eq21_final_extend_ns),
                "eq21_final_extend_density_slope_per_ps": float(eq21_final_extend_density_slope_per_ps),
                "eq21_final_extend_density_delta_kg_m3": float(eq21_final_extend_density_delta_kg_m3),
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="EQ21step pre-EM + pre-NVT + 21-step equilibration",
        )

        expected_gro_sig = file_signature(exp.system_gro)
        expected_top_sig = file_signature(exp.system_top)
        expected_ndx_sig = file_signature(exp.system_ndx)
        _recover_completed_workflow_step(
            self._resume,
            eq_spec,
            summary_path=run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
            label="EQ21 workflow",
            fallback_markers=(final_dir / "summary.json",),
        )
        if self._resume.is_done(eq_spec) and not _workflow_summary_matches(
            run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
        ):
            _preset_log(
                "[RESTART] Cached EQ21 workflow summary does not match the current exported system. Rebuilding from scratch.",
                level=1,
            )
            self._resume.mark_failed(eq_spec, error="stale EQ21 workflow summary", meta={"auto_rebuild": True})
        if self._resume.reuse_status(eq_spec) != "done":
            _invalidate_downstream_resume_steps(
                self._resume,
                names=("npt_production", "nvt_production"),
                prefixes=(
                    "equilibration_additional_",
                    "equilibration_liquid_density_recovery_",
                    "equilibration_polymer_density_recovery_",
                    "equilibration_polymer_chain_relaxation_",
                ),
            )
        job_restart = self._job_restart_flag(eq_spec, bool(rst_flag))
        if bool(rst_flag) and not job_restart:
            latest_stage, next_stage = _latest_reusable_stage_progress(
                run_dir,
                [stage.name for stage in stages],
                input_gro_sig=expected_gro_sig,
                input_top_sig=expected_top_sig,
                input_ndx_sig=expected_ndx_sig,
            )
            if latest_stage is not None:
                detail = f"resuming from {next_stage}" if next_stage is not None else "all stages already present"
                _preset_log(
                    f"[RESTART] Reusing completed EQ21 substeps through {latest_stage}; {detail}.",
                    level=1,
                )
                job_restart = True
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        self._resume.run(eq_spec, lambda: job.run(restart=job_restart))
        final_job = job
        final_extension_history: list[dict[str, object]] = []
        if bool(eq21_final_extend):
            final_job, final_extension_history = _run_eq21_final_density_extensions(
                work_dir=self.work_dir,
                exp=exp,
                initial_job=job,
                resources=res,
                temp=float(temp),
                press=float(press),
                dt_ps=float(cfg.dt_ps),
                tau_p_ps=max(float(cfg.tau_p_ps), 1.0),
                compressibility=float(cfg.compressibility_bar_inv),
                max_rounds=int(eq21_final_extend_max_rounds),
                round_ns=float(eq21_final_extend_ns),
                slope_threshold_per_ps=float(eq21_final_extend_density_slope_per_ps),
                delta_threshold_kg_m3=float(eq21_final_extend_density_delta_kg_m3),
                restart=bool(rst_flag),
            )
        overview_svg = _write_eq21_overview_plot(run_dir, stage_records, params)
        if overview_svg is not None:
            print(f"[EQ21] Overview plot written: {overview_svg}")
        if final_extension_history:
            _preset_item("eq21_final_extension_rounds", sum(1 for rec in final_extension_history if rec.get("extended")))
        self._job = final_job
        _preset_done("EQ21 preset", t_all, detail=f"output={run_dir}")
        return self.ac

    def analyze(self) -> AnalyzeResult:
        if self._job is None:
            raise RuntimeError("EQ21step.exec() must be called before analyze().")
        exp = self._ensure_system_exported()
        tpr, xtc, edr = self._job.final_outputs()
        trr = None
        try:
            trr = self._job.final_trr()
        except Exception:
            trr = None
        return AnalyzeResult(
            work_dir=self.work_dir,
            tpr=tpr,
            xtc=xtc,
            trr=trr,
            edr=edr,
            top=exp.system_top,
            ndx=exp.system_ndx,
            omp=int(getattr(getattr(self._job, "resources", None), "ntomp", 1) or 1),
        )

    def final_gro(self) -> Path:
        """Return the final coordinate file from the most recent preset run."""

        if self._job is None:
            raise RuntimeError("exec() must be called before final_gro().")
        return self._job.final_gro()


class LiquidAnneal(EQ21step):
    """Fast CEMP-like annealing preset for non-polymer liquids/electrolytes."""

    def exec(
        self,
        *,
        temp: float,
        press: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        time: float = 0.8,
        hot_temp: float = 600.0,
        hot_pressure_bar: float = 1000.0,
        compact_pressure_bar: Optional[float] = None,
        hot_nvt_ns: float = 0.05,
        compact_npt_ns: float = 0.15,
        hot_npt_ns: float = 0.20,
        cooling_npt_ns: float = 0.10,
        dt_ps: float = 0.002,
        hot_dt_ps: float = 0.001,
        constraints: str = "h-bonds",
        lincs_iter: Optional[int] = None,
        lincs_order: Optional[int] = None,
        checkpoint_min: float = 5.0,
        gpu_offload_mode: str = "auto",
        mdp_overrides: Optional[dict[str, object]] = None,
        start_gro: Optional[Union[str, Path]] = None,
        restart: Optional[bool] = None,
    ):
        t_all = _preset_section("Liquid anneal equilibration preset", detail=f"restart={bool(resolve_restart(restart))} | work_dir={self.work_dir}")
        rst_flag = resolve_restart(restart)
        self._resume.enabled = bool(rst_flag)
        exp = self._ensure_system_exported()

        run_dir = self.work_dir / "03_liquid_anneal"
        if not rst_flag and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        constraints_mode = _normalize_constraints_mode(constraints)
        resolved_compact_pressure = float(compact_pressure_bar) if compact_pressure_bar is not None else max(float(hot_pressure_bar), 5000.0)
        resolved_gpu_offload_mode = _resolve_production_gpu_offload_mode(self.ac, gpu_offload_mode)
        _preset_item("run_dir", run_dir)
        _preset_item("temperature_K", float(temp))
        _preset_item("pressure_bar", float(press))
        _preset_item("hot_temperature_K", float(hot_temp))
        _preset_item("hot_pressure_bar", float(hot_pressure_bar))
        _preset_item("compact_pressure_bar", float(resolved_compact_pressure))
        _preset_item("final_stage_ns", float(time))
        _preset_item("compact_npt_ns", float(compact_npt_ns))
        _preset_item("dt_ps", float(dt_ps))
        _preset_item("hot_dt_ps", float(hot_dt_ps))
        _preset_item("constraints", constraints_mode)
        _preset_item("gpu_offload_mode", resolved_gpu_offload_mode)
        if mdp_overrides:
            _preset_item("mdp_overrides", mdp_overrides)
        _preset_item("resources", f"mpi={int(mpi)} | omp={int(omp)} | gpu={int(gpu)} | gpu_id={gpu_id}")

        stages = _build_liquid_anneal_stages(
            temp=float(temp),
            press=float(press),
            final_ns=float(time),
            hot_temp=float(hot_temp),
            hot_pressure_bar=float(hot_pressure_bar),
            compact_pressure_bar=float(resolved_compact_pressure),
            hot_nvt_ns=float(hot_nvt_ns),
            compact_npt_ns=float(compact_npt_ns),
            hot_npt_ns=float(hot_npt_ns),
            cooling_npt_ns=float(cooling_npt_ns),
            dt_ps=float(dt_ps),
            hot_dt_ps=float(hot_dt_ps),
            constraints=constraints_mode,
            lincs_iter=lincs_iter,
            lincs_order=lincs_order,
            checkpoint_min=float(checkpoint_min),
            mdp_overrides=mdp_overrides,
        )

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=use_gpu,
            gpu_id=gid,
            gpu_offload_mode=resolved_gpu_offload_mode,
        )
        job = EquilibrationJob(gro=exp.system_gro, top=exp.system_top, provenance_ndx=exp.system_ndx, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name
        spec = StepSpec(
            name="equilibration_liquid_anneal",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "input_gro_sig": file_signature(exp.system_gro),
                "input_top_sig": file_signature(exp.system_top),
                "temp": float(temp),
                "press": float(press),
                "time": float(time),
                "hot_temp": float(hot_temp),
                "hot_pressure_bar": float(hot_pressure_bar),
                "compact_pressure_bar": float(resolved_compact_pressure),
                "hot_nvt_ns": float(hot_nvt_ns),
                "compact_npt_ns": float(compact_npt_ns),
                "hot_npt_ns": float(hot_npt_ns),
                "cooling_npt_ns": float(cooling_npt_ns),
                "dt_ps": float(dt_ps),
                "hot_dt_ps": float(hot_dt_ps),
                "constraints": constraints_mode,
                "lincs_iter": int(lincs_iter) if lincs_iter is not None else None,
                "lincs_order": int(lincs_order) if lincs_order is not None else None,
                "gpu_offload_mode": resolved_gpu_offload_mode,
                "checkpoint_min": float(checkpoint_min),
                "mdp_overrides": json.dumps(mdp_overrides, sort_keys=True, default=str) if mdp_overrides else None,
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="CEMP-like liquid annealing workflow",
        )

        expected_gro_sig = file_signature(exp.system_gro)
        expected_top_sig = file_signature(exp.system_top)
        expected_ndx_sig = file_signature(exp.system_ndx)
        _recover_completed_workflow_step(
            self._resume,
            spec,
            summary_path=run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
            label="liquid anneal workflow",
            fallback_markers=(final_dir / "summary.json",),
        )
        if self._resume.is_done(spec) and not _workflow_summary_matches(
            run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
        ):
            _preset_log(
                "[RESTART] Cached liquid anneal summary does not match the current exported system. Rebuilding from scratch.",
                level=1,
            )
            self._resume.mark_failed(spec, error="stale liquid anneal workflow summary", meta={"auto_rebuild": True})
        if self._resume.reuse_status(spec) != "done":
            _invalidate_downstream_resume_steps(
                self._resume,
                names=("npt_production", "nvt_production"),
                prefixes=(
                    "equilibration_additional_",
                    "equilibration_liquid_density_recovery_",
                    "equilibration_polymer_density_recovery_",
                    "equilibration_polymer_chain_relaxation_",
                ),
            )
        job_restart = self._job_restart_flag(spec, bool(rst_flag))
        if bool(rst_flag) and not job_restart:
            latest_stage, next_stage = _latest_reusable_stage_progress(
                run_dir,
                [stage.name for stage in stages],
                input_gro_sig=expected_gro_sig,
                input_top_sig=expected_top_sig,
                input_ndx_sig=expected_ndx_sig,
            )
            if latest_stage is not None:
                detail = f"resuming from {next_stage}" if next_stage is not None else "all stages already present"
                _preset_log(
                    f"[RESTART] Reusing completed liquid-anneal substeps through {latest_stage}; {detail}.",
                    level=1,
                )
                job_restart = True
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        self._resume.run(spec, lambda: job.run(restart=job_restart))
        self._job = job
        _preset_done("Liquid anneal preset", t_all, detail=f"output={run_dir}")
        return self.ac


class LiquidDensityRecovery(EQ21step):
    """No-polymer liquid recovery round for very low-density starting boxes.

    The round is intentionally different from generic ``Additional``:
    it reheats, compacts at elevated pressure, extends that compaction while the
    density tail is still increasing, then releases/cools back to the target
    temperature and pressure before the normal density plateau gate is checked.
    """

    def exec(
        self,
        *,
        temp: float,
        press: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        time: float = 1.0,
        hot_temp: float = 600.0,
        hot_pressure_bar: float = 1000.0,
        compact_pressure_bar: float = 5000.0,
        hot_nvt_ns: float = 0.03,
        compact_npt_ns: float = 0.25,
        cooling_npt_ns: float = 0.10,
        compact_extend: bool = True,
        compact_extend_max_rounds: int = 4,
        compact_extend_ns: float = 0.20,
        compact_extend_density_slope_per_ps: float = 5.0e-2,
        compact_extend_density_delta_kg_m3: float = 2.0,
        dt_ps: float = 0.002,
        hot_dt_ps: float = 0.001,
        constraints: str = "none",
        lincs_iter: Optional[int] = None,
        lincs_order: Optional[int] = None,
        checkpoint_min: float = 5.0,
        gpu_offload_mode: str = "auto",
        mdp_overrides: Optional[dict[str, object]] = None,
        start_gro: Optional[Union[str, Path]] = None,
        restart: Optional[bool] = None,
    ):
        t_all = _preset_section("Liquid density recovery preset", detail=f"restart={bool(resolve_restart(restart))} | work_dir={self.work_dir}")
        rst_flag = resolve_restart(restart)
        self._resume.enabled = bool(rst_flag)
        exp = self._ensure_system_exported()
        start_gro = Path(start_gro) if start_gro is not None else (_find_latest_equilibrated_gro(self.work_dir) or exp.system_gro)

        round_idx, run_dir = _next_additional_round(self.work_dir, restart=bool(rst_flag))
        constraints_mode = _normalize_constraints_mode(constraints)
        resolved_gpu_offload_mode = _resolve_production_gpu_offload_mode(self.ac, gpu_offload_mode)
        compact_pressure = max(float(compact_pressure_bar), float(hot_pressure_bar))
        _preset_item("round", round_idx)
        _preset_item("run_dir", run_dir)
        _preset_item("start_gro", start_gro)
        _preset_item("temperature_K", float(temp))
        _preset_item("pressure_bar", float(press))
        _preset_item("hot_temperature_K", float(hot_temp))
        _preset_item("hot_pressure_bar", float(hot_pressure_bar))
        _preset_item("compact_pressure_bar", float(compact_pressure))
        _preset_item("final_stage_ns", float(time))
        _preset_item("compact_npt_ns", float(compact_npt_ns))
        _preset_item(
            "compact_extend",
            f"{bool(compact_extend)} | max_rounds={int(compact_extend_max_rounds)} | "
            f"round_ns={float(compact_extend_ns):.3f} | "
            f"slope>{float(compact_extend_density_slope_per_ps):.4g} kg/m^3/ps | "
            f"delta>{float(compact_extend_density_delta_kg_m3):.4g} kg/m^3",
        )
        _preset_item("dt_ps", float(dt_ps))
        _preset_item("hot_dt_ps", float(hot_dt_ps))
        _preset_item("constraints", constraints_mode)
        _preset_item("gpu_offload_mode", resolved_gpu_offload_mode)
        if mdp_overrides:
            _preset_item("mdp_overrides", mdp_overrides)
        _preset_item("resources", f"mpi={int(mpi)} | omp={int(omp)} | gpu={int(gpu)} | gpu_id={gpu_id}")

        compaction_stages = _build_liquid_recovery_compaction_stages(
            hot_temp=float(hot_temp),
            compact_pressure_bar=float(compact_pressure),
            hot_nvt_ns=float(hot_nvt_ns),
            compact_npt_ns=float(compact_npt_ns),
            hot_dt_ps=float(hot_dt_ps),
            constraints=constraints_mode,
            lincs_iter=lincs_iter,
            lincs_order=lincs_order,
        )
        release_stages = _build_liquid_recovery_release_stages(
            temp=float(temp),
            press=float(press),
            final_ns=float(time),
            hot_temp=float(hot_temp),
            hot_pressure_bar=float(hot_pressure_bar),
            cooling_npt_ns=float(cooling_npt_ns),
            dt_ps=float(dt_ps),
            hot_dt_ps=float(hot_dt_ps),
            constraints=constraints_mode,
            lincs_iter=lincs_iter,
            lincs_order=lincs_order,
            checkpoint_min=float(checkpoint_min),
            mdp_overrides=mdp_overrides,
        )

        final_dir = run_dir / release_stages[-1].name
        spec = StepSpec(
            name=f"equilibration_liquid_density_recovery_{round_idx:02d}",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "start_gro_sig": file_signature(Path(start_gro)),
                "input_top_sig": file_signature(exp.system_top),
                "temp": float(temp),
                "press": float(press),
                "time": float(time),
                "hot_temp": float(hot_temp),
                "hot_pressure_bar": float(hot_pressure_bar),
                "compact_pressure_bar": float(compact_pressure),
                "hot_nvt_ns": float(hot_nvt_ns),
                "compact_npt_ns": float(compact_npt_ns),
                "cooling_npt_ns": float(cooling_npt_ns),
                "compact_extend": bool(compact_extend),
                "compact_extend_max_rounds": int(compact_extend_max_rounds),
                "compact_extend_ns": float(compact_extend_ns),
                "compact_extend_density_slope_per_ps": float(compact_extend_density_slope_per_ps),
                "compact_extend_density_delta_kg_m3": float(compact_extend_density_delta_kg_m3),
                "dt_ps": float(dt_ps),
                "hot_dt_ps": float(hot_dt_ps),
                "constraints": constraints_mode,
                "lincs_iter": int(lincs_iter) if lincs_iter is not None else None,
                "lincs_order": int(lincs_order) if lincs_order is not None else None,
                "gpu_offload_mode": resolved_gpu_offload_mode,
                "checkpoint_min": float(checkpoint_min),
                "mdp_overrides": json.dumps(mdp_overrides, sort_keys=True, default=str) if mdp_overrides else None,
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="No-polymer liquid density recovery round",
        )

        expected_gro_sig = file_signature(Path(start_gro))
        expected_top_sig = file_signature(exp.system_top)
        expected_ndx_sig = file_signature(exp.system_ndx)
        _recover_completed_workflow_step(
            self._resume,
            spec,
            summary_path=run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
            label=f"liquid density recovery round {round_idx:02d}",
            fallback_markers=(final_dir / "summary.json",),
        )
        if self._resume.reuse_status(spec) != "done":
            _invalidate_downstream_resume_steps(
                self._resume,
                names=("npt_production", "nvt_production"),
                prefixes=(
                    "equilibration_additional_",
                    "equilibration_liquid_density_recovery_",
                    "equilibration_polymer_density_recovery_",
                    "equilibration_polymer_chain_relaxation_",
                ),
            )

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=use_gpu,
            gpu_id=gid,
            gpu_offload_mode=resolved_gpu_offload_mode,
        )

        release_job_for_cached = EquilibrationJob(
            gro=start_gro,
            top=exp.system_top,
            provenance_ndx=exp.system_ndx,
            out_dir=run_dir,
            stages=release_stages,
            resources=res,
        )

        job_restart = self._job_restart_flag(spec, bool(rst_flag))
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        def _run_recovery() -> EquilibrationJob:
            compaction_job = EquilibrationJob(
                gro=start_gro,
                top=exp.system_top,
                provenance_ndx=exp.system_ndx,
                out_dir=run_dir,
                stages=compaction_stages,
                resources=res,
            )
            compaction_job.run(restart=job_restart)
            current_job = compaction_job
            extension_history: list[dict[str, object]] = []
            if bool(compact_extend):
                current_job, extension_history = _run_liquid_compaction_extensions(
                    ext_root=run_dir / "03_compact_extend",
                    exp=exp,
                    initial_job=compaction_job,
                    resources=res,
                    hot_temp=float(hot_temp),
                    compact_pressure_bar=float(compact_pressure),
                    hot_dt_ps=float(hot_dt_ps),
                    constraints=constraints_mode,
                    lincs_iter=lincs_iter,
                    lincs_order=lincs_order,
                    max_rounds=int(compact_extend_max_rounds),
                    round_ns=float(compact_extend_ns),
                    slope_threshold_per_ps=float(compact_extend_density_slope_per_ps),
                    delta_threshold_kg_m3=float(compact_extend_density_delta_kg_m3),
                    restart=bool(rst_flag),
                )
            if extension_history:
                _preset_item("compact_extension_rounds", sum(1 for rec in extension_history if rec.get("extended")))
            release_job = EquilibrationJob(
                gro=current_job.final_gro(),
                top=exp.system_top,
                provenance_ndx=exp.system_ndx,
                out_dir=run_dir,
                stages=release_stages,
                resources=res,
            )
            release_job.run(restart=job_restart)
            try:
                (run_dir / "liquid_density_recovery.json").write_text(
                    json.dumps(
                        {
                            "round": int(round_idx),
                            "start_gro": str(start_gro),
                            "compact_pressure_bar": float(compact_pressure),
                            "hot_pressure_bar": float(hot_pressure_bar),
                            "target_pressure_bar": float(press),
                            "target_temperature_K": float(temp),
                            "compact_extension_history": extension_history,
                            "final_gro": str(release_job.final_gro()),
                            "notes": (
                                "Recovery uses density-trend extensions during high-pressure compaction, "
                                "then releases/cools to the target T/P before the standard density plateau gate."
                            ),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
            return release_job

        maybe_job = self._resume.run(spec, _run_recovery)
        self._job = maybe_job if isinstance(maybe_job, EquilibrationJob) else release_job_for_cached
        _preset_done("Liquid density recovery preset", t_all, detail=f"output={run_dir}")
        return self.ac


class PolymerDensityRecovery(EQ21step):
    """Conservative no-constraint density recovery for sparse polymer boxes.

    This preset is intentionally LINCS-free. It applies a modest pressure ladder
    across additional rounds, and within each round it may extend the compaction
    segment while density/box-volume diagnostics still show active compression.
    """

    def exec(
        self,
        *,
        temp: float,
        press: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        time: float = 1.0,
        warm_temp: Optional[float] = None,
        pressure_ladder: Sequence[float] = (500.0, 1000.0, 2000.0, 5000.0),
        warm_nvt_ns: float = 0.05,
        compact_npt_ns: float = 0.25,
        compact_extend: bool = True,
        compact_extend_max_rounds: int = 3,
        compact_extend_ns: float = 0.20,
        compact_extend_density_slope_per_ps: float = 5.0e-2,
        compact_extend_density_delta_kg_m3: float = 2.0,
        dt_ps: float = 0.001,
        checkpoint_min: float = 5.0,
        gpu_offload_mode: str = "auto",
        mdp_overrides: Optional[dict[str, object]] = None,
        start_gro: Optional[Union[str, Path]] = None,
        restart: Optional[bool] = None,
    ):
        t_all = _preset_section("Polymer density recovery preset", detail=f"restart={bool(resolve_restart(restart))} | work_dir={self.work_dir}")
        rst_flag = resolve_restart(restart)
        self._resume.enabled = bool(rst_flag)
        exp = self._ensure_system_exported()
        start_gro = Path(start_gro) if start_gro is not None else (_find_latest_equilibrated_gro(self.work_dir) or exp.system_gro)

        round_idx, run_dir = _next_additional_round(self.work_dir, restart=bool(rst_flag))
        resolved_warm_temp = _polymer_warm_temperature(float(temp), warm_temp)
        compact_pressure = _polymer_recovery_pressure(round_idx, pressure_ladder)
        resolved_gpu_offload_mode = _resolve_production_gpu_offload_mode(self.ac, gpu_offload_mode)
        _preset_item("round", round_idx)
        _preset_item("run_dir", run_dir)
        _preset_item("start_gro", start_gro)
        _preset_item("temperature_K", float(temp))
        _preset_item("pressure_bar", float(press))
        _preset_item("warm_temperature_K", float(resolved_warm_temp))
        _preset_item("compact_pressure_bar", float(compact_pressure))
        _preset_item("final_stage_ns", float(time))
        _preset_item("warm_nvt_ns", float(warm_nvt_ns))
        _preset_item("compact_npt_ns", float(compact_npt_ns))
        _preset_item(
            "compact_extend",
            f"{bool(compact_extend)} | max_rounds={int(compact_extend_max_rounds)} | "
            f"round_ns={float(compact_extend_ns):.3f} | "
            f"slope>{float(compact_extend_density_slope_per_ps):.4g} kg/m^3/ps | "
            f"delta>{float(compact_extend_density_delta_kg_m3):.4g} kg/m^3",
        )
        _preset_item("dt_ps", float(dt_ps))
        _preset_item("constraints", "none")
        _preset_item("gpu_offload_mode", resolved_gpu_offload_mode)
        if mdp_overrides:
            _preset_item("mdp_overrides", mdp_overrides)
        _preset_item("resources", f"mpi={int(mpi)} | omp={int(omp)} | gpu={int(gpu)} | gpu_id={gpu_id}")

        compaction_stages = _build_polymer_density_recovery_compaction_stages(
            warm_temp=float(resolved_warm_temp),
            compact_pressure_bar=float(compact_pressure),
            warm_nvt_ns=float(warm_nvt_ns),
            compact_npt_ns=float(compact_npt_ns),
            dt_ps=float(dt_ps),
        )
        release_stages = _build_polymer_density_recovery_release_stages(
            temp=float(temp),
            press=float(press),
            final_ns=float(time),
            dt_ps=float(dt_ps),
            checkpoint_min=float(checkpoint_min),
        )
        compaction_stages = _apply_stage_mdp_overrides(compaction_stages, mdp_overrides)
        release_stages = _apply_stage_mdp_overrides(release_stages, mdp_overrides)

        final_dir = run_dir / release_stages[-1].name
        spec = StepSpec(
            name=f"equilibration_polymer_density_recovery_{round_idx:02d}",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "start_gro_sig": file_signature(Path(start_gro)),
                "input_top_sig": file_signature(exp.system_top),
                "temp": float(temp),
                "press": float(press),
                "time": float(time),
                "warm_temp": float(resolved_warm_temp),
                "pressure_ladder": json.dumps([float(x) for x in pressure_ladder], default=str),
                "compact_pressure_bar": float(compact_pressure),
                "warm_nvt_ns": float(warm_nvt_ns),
                "compact_npt_ns": float(compact_npt_ns),
                "compact_extend": bool(compact_extend),
                "compact_extend_max_rounds": int(compact_extend_max_rounds),
                "compact_extend_ns": float(compact_extend_ns),
                "compact_extend_density_slope_per_ps": float(compact_extend_density_slope_per_ps),
                "compact_extend_density_delta_kg_m3": float(compact_extend_density_delta_kg_m3),
                "dt_ps": float(dt_ps),
                "constraints": "none",
                "gpu_offload_mode": resolved_gpu_offload_mode,
                "checkpoint_min": float(checkpoint_min),
                "mdp_overrides": json.dumps(mdp_overrides, sort_keys=True, default=str) if mdp_overrides else None,
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="Polymer density recovery round",
        )

        expected_gro_sig = file_signature(Path(start_gro))
        expected_top_sig = file_signature(exp.system_top)
        expected_ndx_sig = file_signature(exp.system_ndx)
        _recover_completed_workflow_step(
            self._resume,
            spec,
            summary_path=run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
            label=f"polymer density recovery round {round_idx:02d}",
            fallback_markers=(final_dir / "summary.json",),
        )
        if self._resume.reuse_status(spec) != "done":
            _invalidate_downstream_resume_steps(
                self._resume,
                names=("npt_production", "nvt_production"),
                prefixes=(
                    "equilibration_additional_",
                    "equilibration_liquid_density_recovery_",
                    "equilibration_polymer_density_recovery_",
                    "equilibration_polymer_chain_relaxation_",
                ),
            )

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=use_gpu,
            gpu_id=gid,
            gpu_offload_mode=resolved_gpu_offload_mode,
        )
        release_job_for_cached = EquilibrationJob(
            gro=start_gro,
            top=exp.system_top,
            provenance_ndx=exp.system_ndx,
            out_dir=run_dir,
            stages=release_stages,
            resources=res,
        )

        job_restart = self._job_restart_flag(spec, bool(rst_flag))
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        def _run_recovery() -> EquilibrationJob:
            compaction_job = EquilibrationJob(
                gro=start_gro,
                top=exp.system_top,
                provenance_ndx=exp.system_ndx,
                out_dir=run_dir,
                stages=compaction_stages,
                resources=res,
            )
            compaction_job.run(restart=job_restart)
            current_job = compaction_job
            extension_history: list[dict[str, object]] = []
            if bool(compact_extend):
                current_job, extension_history = _run_polymer_compaction_extensions(
                    ext_root=run_dir / "03_compact_extend",
                    exp=exp,
                    initial_job=compaction_job,
                    resources=res,
                    warm_temp=float(resolved_warm_temp),
                    compact_pressure_bar=float(compact_pressure),
                    dt_ps=float(dt_ps),
                    max_rounds=int(compact_extend_max_rounds),
                    round_ns=float(compact_extend_ns),
                    slope_threshold_per_ps=float(compact_extend_density_slope_per_ps),
                    delta_threshold_kg_m3=float(compact_extend_density_delta_kg_m3),
                    restart=bool(rst_flag),
                )
            if extension_history:
                _preset_item("compact_extension_rounds", sum(1 for rec in extension_history if rec.get("extended")))
            release_job = EquilibrationJob(
                gro=current_job.final_gro(),
                top=exp.system_top,
                provenance_ndx=exp.system_ndx,
                out_dir=run_dir,
                stages=release_stages,
                resources=res,
            )
            release_job.run(restart=job_restart)
            try:
                (run_dir / "polymer_density_recovery.json").write_text(
                    json.dumps(
                        {
                            "round": int(round_idx),
                            "start_gro": str(start_gro),
                            "warm_temperature_K": float(resolved_warm_temp),
                            "compact_pressure_bar": float(compact_pressure),
                            "target_pressure_bar": float(press),
                            "target_temperature_K": float(temp),
                            "constraints": "none",
                            "compact_extension_history": extension_history,
                            "final_gro": str(release_job.final_gro()),
                            "notes": (
                                "Polymer recovery is no-constraint, moderate-pressure compaction. "
                                "It is selected only when density/box-volume diagnostics still indicate compression."
                            ),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
            return release_job

        maybe_job = self._resume.run(spec, _run_recovery)
        self._job = maybe_job if isinstance(maybe_job, EquilibrationJob) else release_job_for_cached
        _preset_done("Polymer density recovery preset", t_all, detail=f"output={run_dir}")
        return self.ac


class PolymerChainRelaxation(EQ21step):
    """No-constraint warm chain relaxation when density is stable but Rg is not."""

    def exec(
        self,
        *,
        temp: float,
        press: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        time: float = 1.0,
        warm_temp: Optional[float] = None,
        warm_nvt_ns: float = 0.10,
        dt_ps: float = 0.001,
        checkpoint_min: float = 5.0,
        gpu_offload_mode: str = "auto",
        mdp_overrides: Optional[dict[str, object]] = None,
        start_gro: Optional[Union[str, Path]] = None,
        restart: Optional[bool] = None,
    ):
        t_all = _preset_section("Polymer chain relaxation preset", detail=f"restart={bool(resolve_restart(restart))} | work_dir={self.work_dir}")
        rst_flag = resolve_restart(restart)
        self._resume.enabled = bool(rst_flag)
        exp = self._ensure_system_exported()
        start_gro = Path(start_gro) if start_gro is not None else (_find_latest_equilibrated_gro(self.work_dir) or exp.system_gro)

        round_idx, run_dir = _next_additional_round(self.work_dir, restart=bool(rst_flag))
        resolved_warm_temp = _polymer_warm_temperature(float(temp), warm_temp)
        resolved_gpu_offload_mode = _resolve_production_gpu_offload_mode(self.ac, gpu_offload_mode)
        _preset_item("round", round_idx)
        _preset_item("run_dir", run_dir)
        _preset_item("start_gro", start_gro)
        _preset_item("temperature_K", float(temp))
        _preset_item("pressure_bar", float(press))
        _preset_item("warm_temperature_K", float(resolved_warm_temp))
        _preset_item("warm_nvt_ns", float(warm_nvt_ns))
        _preset_item("final_stage_ns", float(time))
        _preset_item("dt_ps", float(dt_ps))
        _preset_item("constraints", "none")
        _preset_item("gpu_offload_mode", resolved_gpu_offload_mode)
        if mdp_overrides:
            _preset_item("mdp_overrides", mdp_overrides)
        _preset_item("resources", f"mpi={int(mpi)} | omp={int(omp)} | gpu={int(gpu)} | gpu_id={gpu_id}")

        stages = _build_polymer_chain_relaxation_stages(
            temp=float(temp),
            press=float(press),
            final_ns=float(time),
            warm_temp=float(resolved_warm_temp),
            warm_nvt_ns=float(warm_nvt_ns),
            dt_ps=float(dt_ps),
            checkpoint_min=float(checkpoint_min),
        )
        stages = _apply_stage_mdp_overrides(stages, mdp_overrides, stage_kinds=("minim", "nvt", "npt", "md"))

        final_dir = run_dir / stages[-1].name
        spec = StepSpec(
            name=f"equilibration_polymer_chain_relaxation_{round_idx:02d}",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "start_gro_sig": file_signature(Path(start_gro)),
                "input_top_sig": file_signature(exp.system_top),
                "temp": float(temp),
                "press": float(press),
                "time": float(time),
                "warm_temp": float(resolved_warm_temp),
                "warm_nvt_ns": float(warm_nvt_ns),
                "dt_ps": float(dt_ps),
                "constraints": "none",
                "gpu_offload_mode": resolved_gpu_offload_mode,
                "checkpoint_min": float(checkpoint_min),
                "mdp_overrides": json.dumps(mdp_overrides, sort_keys=True, default=str) if mdp_overrides else None,
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="Polymer Rg/chain relaxation round",
        )

        expected_gro_sig = file_signature(Path(start_gro))
        expected_top_sig = file_signature(exp.system_top)
        expected_ndx_sig = file_signature(exp.system_ndx)
        _recover_completed_workflow_step(
            self._resume,
            spec,
            summary_path=run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
            label=f"polymer chain relaxation round {round_idx:02d}",
            fallback_markers=(final_dir / "summary.json",),
        )
        if self._resume.is_done(spec) and not _workflow_summary_matches(
            run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
        ):
            _preset_log(
                f"[RESTART] Cached polymer chain relaxation summary for round {round_idx:02d} does not match current inputs. Rebuilding from scratch.",
                level=1,
            )
            self._resume.mark_failed(spec, error="stale polymer chain relaxation summary", meta={"auto_rebuild": True})
        if self._resume.reuse_status(spec) != "done":
            _invalidate_downstream_resume_steps(
                self._resume,
                names=("npt_production", "nvt_production"),
                prefixes=(
                    "equilibration_additional_",
                    "equilibration_liquid_density_recovery_",
                    "equilibration_polymer_density_recovery_",
                    "equilibration_polymer_chain_relaxation_",
                ),
            )

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=use_gpu,
            gpu_id=gid,
            gpu_offload_mode=resolved_gpu_offload_mode,
        )
        job = EquilibrationJob(gro=start_gro, top=exp.system_top, provenance_ndx=exp.system_ndx, out_dir=run_dir, stages=stages, resources=res)
        job_restart = self._job_restart_flag(spec, bool(rst_flag))
        if bool(rst_flag) and not job_restart:
            latest_stage, next_stage = _latest_reusable_stage_progress(
                run_dir,
                [stage.name for stage in stages],
                input_gro_sig=expected_gro_sig,
                input_top_sig=expected_top_sig,
                input_ndx_sig=expected_ndx_sig,
            )
            if latest_stage is not None:
                detail = f"resuming from {next_stage}" if next_stage is not None else "all stages already present"
                _preset_log(
                    f"[RESTART] Reusing completed polymer-chain-relaxation substeps through {latest_stage}; {detail}.",
                    level=1,
                )
                job_restart = True
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        self._resume.run(spec, lambda: job.run(restart=job_restart))
        try:
            (run_dir / "polymer_chain_relaxation.json").write_text(
                json.dumps(
                    {
                        "round": int(round_idx),
                        "start_gro": str(start_gro),
                        "warm_temperature_K": float(resolved_warm_temp),
                        "target_pressure_bar": float(press),
                        "target_temperature_K": float(temp),
                        "constraints": "none",
                        "final_gro": str(job.final_gro()),
                        "notes": "Selected when density is stable but polymer Rg has not plateaued.",
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        self._job = job
        _preset_done("Polymer chain relaxation preset", t_all, detail=f"output={run_dir}")
        return self.ac


class Additional(EQ21step):
    """Additional equilibration rounds.

    The additional rounds are stored under:
      work_dir/04_eq_additional/round_XX/

    This keeps each round self-contained and makes restart behavior robust.
    """

    def exec(
        self,
        *,
        temp: float,
        press: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        sim_time: float = 1.0,
        time: Optional[float] = None,
        dt_ps: float = 0.001,
        constraints: str = "none",
        lincs_iter: Optional[int] = None,
        lincs_order: Optional[int] = None,
        gpu_offload_mode: str = "auto",
        mdp_overrides: Optional[dict[str, object]] = None,
        start_gro: Optional[Union[str, Path]] = None,
        skip_rebuild: Optional[bool] = None,
        micro_relax: bool = False,
        restart: Optional[bool] = None,
    ):
        t_all = _preset_section("Additional equilibration preset", detail=f"restart={bool(resolve_restart(restart))} | work_dir={self.work_dir}")
        rst_flag = resolve_restart(restart)
        self._resume.enabled = bool(rst_flag)

        if time is not None:
            sim_time = float(time)

        exp = self._ensure_system_exported()

        # Prefer continuing from the latest equilibrated structure already present in work_dir.
        # This is important when the user creates a new `Additional(...)` instance in scripts
        # (common in examples), in which case self._job is None.
        start_gro = Path(start_gro) if start_gro is not None else (_find_latest_equilibrated_gro(self.work_dir) or exp.system_gro)

        # 判定标签：若已有可用的平衡结构（即 start_gro 不是原始 system.gro），
        # 则无需重复建盒子/EM/NVT/NPT 等流程。
        # 只在原有基础上重复跑最终平衡阶段（04_md）。
        _skip_rebuild = bool(start_gro != exp.system_gro) if skip_rebuild is None else bool(skip_rebuild)

        round_idx, run_dir = _next_additional_round(self.work_dir, restart=bool(rst_flag))
        _preset_item("round", round_idx)
        _preset_item("run_dir", run_dir)
        _preset_item("start_gro", start_gro)
        _preset_item("temperature_K", float(temp))
        _preset_item("pressure_bar", float(press))
        _preset_item("production_ns", float(sim_time))
        _preset_item("dt_ps", float(dt_ps))
        _preset_item("constraints", _normalize_constraints_mode(constraints))
        _preset_item("lincs_iter", int(lincs_iter) if lincs_iter is not None else None)
        _preset_item("lincs_order", int(lincs_order) if lincs_order is not None else None)
        resolved_gpu_offload_mode = _resolve_production_gpu_offload_mode(self.ac, gpu_offload_mode)
        _preset_item("gpu_offload_mode", resolved_gpu_offload_mode)
        if mdp_overrides:
            _preset_item("mdp_overrides", mdp_overrides)
        if skip_rebuild is not None:
            _preset_item("skip_rebuild", bool(skip_rebuild))
        if micro_relax:
            _preset_item("micro_relax", "cold/warm 0.1 fs NVT -> 0.2 fs target NVT -> 0.25/0.5 fs NPT")

        if bool(micro_relax) and not _skip_rebuild:
            stages = _build_liquid_target_relaxation_stages(
                temp=float(temp),
                press=float(press),
                final_ns=float(sim_time),
                dt_ps=float(dt_ps),
                checkpoint_min=5.0,
                mdp_overrides=mdp_overrides,
            )
        else:
            stages = EquilibrationJob.default_stages(
                temperature_k=float(temp),
                pressure_bar=float(press),
                dt_ps=float(dt_ps),
                constraints=constraints,
                lincs_iter=lincs_iter,
                lincs_order=lincs_order,
                nvt_ns=0.1,
                npt_ns=0.2,
                prod_ns=float(sim_time),
            )
            stages = _apply_stage_mdp_overrides(stages, mdp_overrides, stage_kinds=("minim", "nvt", "npt", "md"))

        # 判定标签：只保留最终平衡阶段（04_md）
        if _skip_rebuild and stages:
            stages = [stages[-1]]

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=use_gpu,
            gpu_id=gid,
            gpu_offload_mode=resolved_gpu_offload_mode,
        )
        job = EquilibrationJob(gro=start_gro, top=exp.system_top, provenance_ndx=exp.system_ndx, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name if stages else run_dir
        spec = StepSpec(
            name=f"equilibration_additional_{round_idx:02d}",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "start_gro_sig": file_signature(Path(start_gro)),
                "input_top_sig": file_signature(exp.system_top),
                "temp": float(temp),
                "press": float(press),
                "sim_time": float(sim_time),
                "dt_ps": float(dt_ps),
                "constraints": _normalize_constraints_mode(constraints),
                "lincs_iter": int(lincs_iter) if lincs_iter is not None else None,
                "lincs_order": int(lincs_order) if lincs_order is not None else None,
                "explicit_start_gro": str(start_gro) if start_gro is not None else None,
                "skip_rebuild": bool(_skip_rebuild),
                "micro_relax": bool(micro_relax),
                "gpu_offload_mode": resolved_gpu_offload_mode,
                "mdp_overrides": json.dumps(mdp_overrides, sort_keys=True, default=str) if mdp_overrides else None,
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="Additional equilibration round",
        )

        expected_gro_sig = file_signature(Path(start_gro))
        expected_top_sig = file_signature(exp.system_top)
        expected_ndx_sig = file_signature(exp.system_ndx)
        _recover_completed_workflow_step(
            self._resume,
            spec,
            summary_path=run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
            label=f"additional equilibration round {round_idx:02d}",
            fallback_markers=(final_dir / "summary.json",),
        )
        if self._resume.is_done(spec) and not _workflow_summary_matches(
            run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
        ):
            _preset_log(
                f"[RESTART] Cached additional equilibration summary for round {round_idx:02d} does not match current inputs. Rebuilding from scratch.",
                level=1,
            )
            self._resume.mark_failed(spec, error="stale additional equilibration summary", meta={"auto_rebuild": True})
        if self._resume.reuse_status(spec) != "done":
            _invalidate_downstream_resume_steps(
                self._resume,
                names=("npt_production", "nvt_production"),
                prefixes=(
                    "equilibration_additional_",
                    "equilibration_liquid_density_recovery_",
                    "equilibration_polymer_density_recovery_",
                    "equilibration_polymer_chain_relaxation_",
                ),
            )
        job_restart = self._job_restart_flag(spec, bool(rst_flag))
        if bool(rst_flag) and not job_restart:
            latest_stage, next_stage = _latest_reusable_stage_progress(
                run_dir,
                [stage.name for stage in stages],
                input_gro_sig=expected_gro_sig,
                input_top_sig=expected_top_sig,
                input_ndx_sig=expected_ndx_sig,
            )
            if latest_stage is not None:
                detail = f"resuming from {next_stage}" if next_stage is not None else "all stages already present"
                _preset_log(
                    f"[RESTART] Reusing completed additional-equilibration substeps through {latest_stage}; {detail}.",
                    level=1,
                )
                job_restart = True
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        self._resume.run(spec, lambda: job.run(restart=job_restart))
        self._job = job
        _preset_done("Additional equilibration preset", t_all, detail=f"output={run_dir}")
        return self.ac


@dataclass
class NPT(EQ21step):
    """NPT production run after equilibration has converged."""

    def exec(
        self,
        *,
        temp: float,
        press: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        time: float = 5.0,
        traj_ps: Union[float, str, None] = "auto",
        energy_ps: Union[float, str, None] = "auto",
        log_ps: Union[float, str, None] = "auto",
        trr_ps: Union[float, str, None] = None,
        velocity_ps: Union[float, str, None] = None,
        checkpoint_min: float = 5.0,
        dt_ps: float = 0.001,
        constraints: str = "none",
        lincs_iter: Optional[int] = None,
        lincs_order: Optional[int] = None,
        gpu_offload_mode: str = "auto",
        performance_profile: str = "auto",
        analysis_profile: str = "auto",
        max_trajectory_frames: Optional[int] = None,
        max_atom_frames: Optional[float] = None,
        bridge_ps: Optional[float] = None,
        bridge_dt_fs: float = 1.0,
        bridge_lincs_iter: int = 4,
        bridge_lincs_order: int = 12,
        mdp_overrides: Optional[dict[str, object]] = None,
        start_gro: Optional[Union[str, Path]] = None,
        restart: Optional[bool] = None,
    ):
        t_all = _preset_section("NPT production preset", detail=f"restart={bool(resolve_restart(restart))} | work_dir={self.work_dir}")
        rst_flag = resolve_restart(restart)
        self._resume.enabled = bool(rst_flag)
        exp = self._ensure_system_exported()

        run_dir = self.work_dir / "05_npt_production"
        # Production must restart from upstream equilibration, not from its own prior output.
        start_gro = Path(start_gro) if start_gro is not None else (_find_latest_equilibrated_gro(self.work_dir, exclude_dirs=[run_dir]) or exp.system_gro)
        policy = resolve_io_analysis_policy(
            prod_ns=float(time),
            atom_count=_estimate_atom_count_for_policy(self.ac, exp),
            performance_profile=performance_profile,
            analysis_profile=analysis_profile,
            traj_ps=traj_ps,
            energy_ps=energy_ps,
            log_ps=log_ps,
            trr_ps=trr_ps,
            velocity_ps=velocity_ps,
            max_trajectory_frames=max_trajectory_frames,
            max_atom_frames=max_atom_frames,
        )
        _preset_item("run_dir", run_dir)
        _preset_item("start_gro", start_gro)
        _preset_item("temperature_K", float(temp))
        _preset_item("pressure_bar", float(press))
        _preset_item("production_ns", float(time))
        _preset_item("dt_ps", float(dt_ps))
        _preset_item("constraints", _normalize_constraints_mode(constraints))
        _preset_item("lincs_iter", int(lincs_iter) if lincs_iter is not None else None)
        _preset_item("lincs_order", int(lincs_order) if lincs_order is not None else None)
        resolved_gpu_offload_mode = _resolve_production_gpu_offload_mode(self.ac, gpu_offload_mode)
        resolved_bridge_ps = _resolve_production_bridge_ps(self.ac, bridge_ps)
        _preset_item(
            "performance_policy",
            f"{policy.policy_level} | profile={policy.performance_profile} | "
            f"analysis={policy.analysis_profile} | frames~{policy.estimated_frames}",
        )
        _preset_item(
            "output_cadence",
            f"xtc={policy.traj_ps:.3f} ps | energy={policy.energy_ps:.3f} ps | "
            f"log={policy.log_ps:.3f} ps | "
            f"trr={'off' if policy.trr_ps is None else f'{policy.trr_ps:.3f} ps'} | "
            f"vel={'off' if policy.velocity_ps is None else f'{policy.velocity_ps:.3f} ps'} | "
            f"cpt={float(checkpoint_min):.2f} min"
        )
        _preset_item("gpu_offload_mode", resolved_gpu_offload_mode)
        _preset_item(
            "bridge",
            f"{float(resolved_bridge_ps):.1f} ps | dt={float(bridge_dt_fs):.3f} fs | "
            f"lincs_iter={int(bridge_lincs_iter)} | lincs_order={int(bridge_lincs_order)}",
        )
        if mdp_overrides:
            _preset_item("mdp_overrides", mdp_overrides)
        _preset_item("resources", f"mpi={int(mpi)} | omp={int(omp)} | gpu={int(gpu)} | gpu_id={gpu_id}")
        if not rst_flag and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        # Build a single-stage NPT MD using the caller-selected production regime.
        from ...gmx.mdp_templates import NPT_MDP, NPT_NO_CONSTRAINTS_MDP, default_mdp_params

        p = default_mdp_params()
        p["ref_t"] = float(temp)
        p["ref_p"] = float(press)
        # Production starts from an equilibrated coordinate snapshot, not from
        # the previous stage checkpoint. Generate a clean target-temperature
        # Maxwell distribution for the first production/bridge stage.
        p["gen_vel"] = "yes"
        p["gen_temp"] = float(temp)
        p["continuation"] = "no"
        p, constraints_mode = _prepare_production_mdp_params(
            base_params=p,
            dt_ps=float(dt_ps),
            constraints=constraints,
            lincs_iter=lincs_iter,
            lincs_order=lincs_order,
            traj_ps=policy.traj_ps,
            energy_ps=policy.energy_ps,
            log_ps=policy.log_ps,
            trr_ps=policy.trr_ps,
            velocity_ps=policy.velocity_ps,
            mdp_overrides=mdp_overrides,
        )
        template = NPT_NO_CONSTRAINTS_MDP if not _constraints_use_lincs(constraints_mode) else NPT_MDP
        effective_dt_ps = float(p["dt"])
        effective_lincs_iter = int(p["lincs_iter"]) if _constraints_use_lincs(constraints_mode) else None
        effective_lincs_order = int(p["lincs_order"]) if _constraints_use_lincs(constraints_mode) else None

        stages = _build_production_stages(
            stage_name="npt",
            template=template,
            params=p,
            prod_ns=float(time),
            checkpoint_min=float(checkpoint_min),
            constraints_mode=constraints_mode,
            bridge_ps=float(resolved_bridge_ps),
            bridge_dt_fs=float(bridge_dt_fs),
            bridge_lincs_iter=int(bridge_lincs_iter),
            bridge_lincs_order=int(bridge_lincs_order),
            first_stage_gen_vel=str(p.get("gen_vel", "yes")),
        )

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=use_gpu,
            gpu_id=gid,
            gpu_offload_mode=resolved_gpu_offload_mode,
        )
        job = EquilibrationJob(gro=start_gro, top=exp.system_top, provenance_ndx=exp.system_ndx, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name
        spec = StepSpec(
            name="npt_production",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "start_gro_sig": file_signature(Path(start_gro)),
                "input_top_sig": file_signature(exp.system_top),
                "temp": float(temp),
                "press": float(press),
                "time": float(time),
                "dt_ps": float(effective_dt_ps),
                "constraints": constraints_mode,
                "lincs_iter": int(effective_lincs_iter) if effective_lincs_iter is not None else None,
                "lincs_order": int(effective_lincs_order) if effective_lincs_order is not None else None,
                "traj_ps": float(policy.traj_ps),
                "energy_ps": float(policy.energy_ps),
                "log_ps": float(policy.log_ps),
                "trr_ps": float(policy.trr_ps) if policy.trr_ps is not None else None,
                "velocity_ps": float(policy.velocity_ps) if policy.velocity_ps is not None else None,
                "performance_policy": json.dumps(policy.to_dict(), sort_keys=True, default=str),
                "checkpoint_min": float(checkpoint_min),
                "gpu_offload_mode": resolved_gpu_offload_mode,
                "bridge_ps": float(resolved_bridge_ps),
                "bridge_dt_fs": float(bridge_dt_fs),
                "bridge_lincs_iter": int(bridge_lincs_iter),
                "bridge_lincs_order": int(bridge_lincs_order),
                "mdp_overrides": json.dumps(mdp_overrides, sort_keys=True, default=str) if mdp_overrides else None,
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="NPT production run",
        )

        expected_gro_sig = file_signature(Path(start_gro))
        expected_top_sig = file_signature(exp.system_top)
        expected_ndx_sig = file_signature(exp.system_ndx)
        _recover_completed_workflow_step(
            self._resume,
            spec,
            summary_path=run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
            label="NPT production",
            fallback_markers=(final_dir / "summary.json",),
        )
        if self._resume.is_done(spec) and not _workflow_summary_matches(
            run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
        ):
            _preset_log(
                "[RESTART] Cached NPT production summary does not match current inputs. Rebuilding from scratch.",
                level=1,
            )
            self._resume.mark_failed(spec, error="stale NPT production summary", meta={"auto_rebuild": True})
        if self._resume.reuse_status(spec) != "done":
            _invalidate_downstream_resume_steps(self._resume, names=("nvt_production",))
        job_restart = self._job_restart_flag(spec, bool(rst_flag))
        if bool(rst_flag) and not job_restart:
            latest_stage, next_stage = _latest_reusable_stage_progress(
                run_dir,
                [stage.name for stage in stages],
                input_gro_sig=expected_gro_sig,
                input_top_sig=expected_top_sig,
                input_ndx_sig=expected_ndx_sig,
            )
            if latest_stage is not None:
                detail = f"resuming from {next_stage}" if next_stage is not None else "all stages already present"
                _preset_log(
                    f"[RESTART] Reusing completed NPT production substeps through {latest_stage}; {detail}.",
                    level=1,
                )
                job_restart = True
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        self._resume.run(spec, lambda: job.run(restart=job_restart))
        _write_production_policy_summary(run_dir, policy)
        self._job = job
        _preset_done("NPT production preset", t_all, detail=f"output={run_dir}")
        return self.ac



@dataclass
class NVT(EQ21step):
    """NVT production run with density fixed to an equilibrium-average target.

    Workflow (robust):
      1) Read the density time series from the *previous* equilibrated NPT/MD EDR.
      2) Determine a safe averaging window:
           - Prefer the density plateau window from `work_dir/06_analysis/equilibrium.json` (if present)
           - Fallback to the last `density_frac_last` fraction of the series.
      3) Compute the target mean density (rho_mean) over that window.
      4) Compute a *tail-mean* density (rho_tail) near the end of the run (to avoid a single-frame spike).
      5) Compute the scaling factor:
             s = (rho_tail / rho_mean)^(1/3)
         and scale the starting gro (box + coordinates) with `gmx editconf -scale s s s`.
      6) Run a single-stage NVT production MD.

    Notes:
      - We scale from an end-of-run *tail mean* density (rho_tail) rather than a single
        last-point value, so the scaling is stable even when the density time series is noisy.
      - This behavior is inspired by yuzc's in-house yzc-gmx-gen workflow.
    """

    def exec(
        self,
        *,
        temp: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        time: float = 5.0,
        traj_ps: Union[float, str, None] = "auto",
        energy_ps: Union[float, str, None] = "auto",
        log_ps: Union[float, str, None] = "auto",
        trr_ps: Union[float, str, None] = None,
        velocity_ps: Union[float, str, None] = None,
        checkpoint_min: float = 5.0,
        dt_ps: float = 0.001,
        constraints: str = "none",
        lincs_iter: Optional[int] = None,
        lincs_order: Optional[int] = None,
        gpu_offload_mode: str = "auto",
        performance_profile: str = "auto",
        analysis_profile: str = "auto",
        max_trajectory_frames: Optional[int] = None,
        max_atom_frames: Optional[float] = None,
        bridge_ps: Optional[float] = None,
        bridge_dt_fs: float = 1.0,
        bridge_lincs_iter: int = 4,
        bridge_lincs_order: int = 12,
        mdp_overrides: Optional[dict[str, object]] = None,
        restart: Optional[bool] = None,
        density_control: Optional[bool] = None,
        density_frac_last: float = 0.3,
    ):
        t_all = _preset_section("EQ21 equilibration preset", detail=f"restart={bool(resolve_restart(restart))} | work_dir={self.work_dir}")
        rst_flag = resolve_restart(restart)
        self._resume.enabled = bool(rst_flag)
        exp = self._ensure_system_exported()

        run_dir = self.work_dir / "05_nvt_production"
        # NVT production can reuse upstream NPT production, but must never point at
        # its own stale output when a rebuild is required.
        start_gro = _find_latest_equilibrated_gro(self.work_dir, exclude_dirs=[run_dir]) or exp.system_gro
        policy = resolve_io_analysis_policy(
            prod_ns=float(time),
            atom_count=_estimate_atom_count_for_policy(self.ac, exp),
            performance_profile=performance_profile,
            analysis_profile=analysis_profile,
            traj_ps=traj_ps,
            energy_ps=energy_ps,
            log_ps=log_ps,
            trr_ps=trr_ps,
            velocity_ps=velocity_ps,
            max_trajectory_frames=max_trajectory_frames,
            max_atom_frames=max_atom_frames,
        )
        _preset_item("run_dir", run_dir)
        _preset_item("start_gro", start_gro)
        _preset_item("temperature_K", float(temp))
        _preset_item("production_ns", float(time))
        _preset_item("dt_ps", float(dt_ps))
        _preset_item("constraints", _normalize_constraints_mode(constraints))
        _preset_item("lincs_iter", int(lincs_iter) if lincs_iter is not None else None)
        _preset_item("lincs_order", int(lincs_order) if lincs_order is not None else None)
        resolved_gpu_offload_mode = _resolve_production_gpu_offload_mode(self.ac, gpu_offload_mode)
        resolved_bridge_ps = _resolve_production_bridge_ps(self.ac, bridge_ps)
        resolved_density_control = _resolve_nvt_density_control(self.ac, density_control)
        _preset_item("density_control", bool(resolved_density_control))
        _preset_item("density_frac_last", float(density_frac_last))
        _preset_item(
            "performance_policy",
            f"{policy.policy_level} | profile={policy.performance_profile} | "
            f"analysis={policy.analysis_profile} | frames~{policy.estimated_frames}",
        )
        _preset_item(
            "output_cadence",
            f"xtc={policy.traj_ps:.3f} ps | energy={policy.energy_ps:.3f} ps | "
            f"log={policy.log_ps:.3f} ps | "
            f"trr={'off' if policy.trr_ps is None else f'{policy.trr_ps:.3f} ps'} | "
            f"vel={'off' if policy.velocity_ps is None else f'{policy.velocity_ps:.3f} ps'} | "
            f"cpt={float(checkpoint_min):.2f} min"
        )
        _preset_item("gpu_offload_mode", resolved_gpu_offload_mode)
        _preset_item(
            "bridge",
            f"{float(resolved_bridge_ps):.1f} ps | dt={float(bridge_dt_fs):.3f} fs | "
            f"lincs_iter={int(bridge_lincs_iter)} | lincs_order={int(bridge_lincs_order)}",
        )
        if mdp_overrides:
            _preset_item("mdp_overrides", mdp_overrides)
        _preset_item("resources", f"mpi={int(mpi)} | omp={int(omp)} | gpu={int(gpu)} | gpu_id={gpu_id}")
        if not rst_flag and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        # Density control (optional): scale the starting gro to an equilibrium-average density.
        scaled_gro = start_gro
        if resolved_density_control:
            from ...gmx.engine import GromacsRunner
            from ...gmx.analysis.xvg import read_xvg
            from ...gmx.analysis.thermo import stats_from_xvg
            import numpy as np
            import uuid

            runner = GromacsRunner()
            tmp = None
            scale_root = self.work_dir / ".yadonpy" / "scratch"
            scale_root.mkdir(parents=True, exist_ok=True)

            # Try to locate the previous edr (same directory as start_gro).
            prev_dir = Path(start_gro).parent
            prev_edr = prev_dir / "md.edr"

            # Fallback: search under work_dir for the newest md.edr.
            if not prev_edr.exists():
                candidates = list(Path(self.work_dir).rglob("md.edr"))
                candidates = [c for c in candidates if c.is_file()]
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    prev_edr = candidates[0]

            if prev_edr.exists():
                # Detect density term name (can vary across GROMACS versions, e.g. "Density (kg/m^3)").
                mapping = runner.list_energy_terms(edr=prev_edr)
                dens_term = None
                for k in mapping.keys():
                    if "density" in k.lower():
                        dens_term = k
                        break

                if dens_term is not None:
                    tmp = run_dir / f"_yadonpy_density_tmp_{uuid.uuid4().hex[:8]}.xvg"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    runner.energy_xvg(edr=prev_edr, out_xvg=tmp, terms=[dens_term])

                    df = read_xvg(tmp).df
                    dens_cols = [c for c in df.columns if c != "x"]
                    if dens_cols:
                        col = dens_cols[0]
                        # --- determine the averaging window ---
                        t_ps = df["x"].to_numpy(dtype=float)
                        rho = df[col].to_numpy(dtype=float)

                        # Prefer the density plateau start from equilibrium.json if available.
                        window_start_ps = None
                        try:
                            eq_json = self.work_dir / "06_analysis" / "equilibrium.json"
                            if eq_json.exists():
                                payload = json.loads(eq_json.read_text(encoding="utf-8"))
                                dg = payload.get("density_gate") or {}
                                ws = dg.get("window_start_time_ps")
                                if ws is not None:
                                    window_start_ps = float(ws)
                        except Exception:
                            window_start_ps = None

                        if window_start_ps is not None:
                            mask = t_ps >= float(window_start_ps)
                            if mask.any():
                                rho_mean = float(np.mean(rho[mask]))
                            else:
                                # Fallback: last fraction
                                st = stats_from_xvg(tmp, col=col, frac_last=float(density_frac_last))
                                rho_mean = float(st.mean)
                        else:
                            st = stats_from_xvg(tmp, col=col, frac_last=float(density_frac_last))
                            rho_mean = float(st.mean)

                        # Use a tail-mean density to avoid single-point noise.
                        # Take the last min(2% of frames, 50 ps) as tail window.
                        rho_tail = float(rho[-1])
                        try:
                            if len(t_ps) >= 5:
                                t_end = float(t_ps[-1])
                                t_start_tail = max(float(t_ps[0]), t_end - 50.0)
                                mask_tail = t_ps >= t_start_tail
                                # Ensure we include at least 2% frames.
                                if mask_tail.sum() < max(int(0.02 * len(t_ps)), 1):
                                    mask_tail = np.zeros_like(mask_tail, dtype=bool)
                                    mask_tail[-max(int(0.02 * len(t_ps)), 1):] = True
                                rho_tail = float(np.mean(rho[mask_tail]))
                        except Exception:
                            rho_tail = float(rho[-1])

                        # Avoid divide-by-zero or pathological scaling.
                        if rho_tail > 0 and rho_mean > 0:
                            # After scaling with factor s, density becomes rho_tail / s^3 == rho_mean
                            s = float((rho_tail / rho_mean) ** (1.0 / 3.0))
                            scaled_gro = scale_root / "nvt_start_scaled_to_eq_density.gro"
                            runner.run(
                                [
                                    "editconf",
                                    "-f",
                                    str(start_gro),
                                    "-o",
                                    str(scaled_gro),
                                    "-scale",
                                    f"{s}",
                                    f"{s}",
                                    f"{s}",
                                ],
                                cwd=run_dir,
                                check=True,
                                capture=True,
                            )

                            # Note: system-level MOL2 export is handled at the export_system stage.

                # Best-effort cleanup
                if tmp is not None:
                    try:
                        tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
                    except Exception:
                        pass

        # Build a single-stage NVT MD using the caller-selected production regime.
        from ...gmx.mdp_templates import NVT_MDP, NVT_NO_CONSTRAINTS_MDP, default_mdp_params

        p = default_mdp_params()
        p["ref_t"] = float(temp)
        p, constraints_mode = _prepare_production_mdp_params(
            base_params=p,
            dt_ps=float(dt_ps),
            constraints=constraints,
            lincs_iter=lincs_iter,
            lincs_order=lincs_order,
            traj_ps=policy.traj_ps,
            energy_ps=policy.energy_ps,
            log_ps=policy.log_ps,
            trr_ps=policy.trr_ps,
            velocity_ps=policy.velocity_ps,
            mdp_overrides=mdp_overrides,
        )
        template = NVT_NO_CONSTRAINTS_MDP if not _constraints_use_lincs(constraints_mode) else NVT_MDP
        effective_dt_ps = float(p["dt"])
        effective_lincs_iter = int(p["lincs_iter"]) if _constraints_use_lincs(constraints_mode) else None
        effective_lincs_order = int(p["lincs_order"]) if _constraints_use_lincs(constraints_mode) else None

        stages = _build_production_stages(
            stage_name="nvt",
            template=template,
            params=p,
            prod_ns=float(time),
            checkpoint_min=float(checkpoint_min),
            constraints_mode=constraints_mode,
            bridge_ps=float(resolved_bridge_ps),
            bridge_dt_fs=float(bridge_dt_fs),
            bridge_lincs_iter=int(bridge_lincs_iter),
            bridge_lincs_order=int(bridge_lincs_order),
        )

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=use_gpu,
            gpu_id=gid,
            gpu_offload_mode=resolved_gpu_offload_mode,
        )
        job = EquilibrationJob(gro=scaled_gro, top=exp.system_top, provenance_ndx=exp.system_ndx, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name
        spec = StepSpec(
            name="nvt_production",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "start_gro_sig": file_signature(Path(scaled_gro)),
                "input_top_sig": file_signature(exp.system_top),
                "temp": float(temp),
                "time": float(time),
                "dt_ps": float(effective_dt_ps),
                "constraints": constraints_mode,
                "lincs_iter": int(effective_lincs_iter) if effective_lincs_iter is not None else None,
                "lincs_order": int(effective_lincs_order) if effective_lincs_order is not None else None,
                "traj_ps": float(policy.traj_ps),
                "energy_ps": float(policy.energy_ps),
                "log_ps": float(policy.log_ps),
                "trr_ps": float(policy.trr_ps) if policy.trr_ps is not None else None,
                "velocity_ps": float(policy.velocity_ps) if policy.velocity_ps is not None else None,
                "performance_policy": json.dumps(policy.to_dict(), sort_keys=True, default=str),
                "checkpoint_min": float(checkpoint_min),
                "gpu_offload_mode": resolved_gpu_offload_mode,
                "bridge_ps": float(resolved_bridge_ps),
                "bridge_dt_fs": float(bridge_dt_fs),
                "bridge_lincs_iter": int(bridge_lincs_iter),
                "bridge_lincs_order": int(bridge_lincs_order),
                "mdp_overrides": json.dumps(mdp_overrides, sort_keys=True, default=str) if mdp_overrides else None,
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
                "density_control": bool(resolved_density_control),
                "density_frac_last": float(density_frac_last),
            },
            description="NVT production run (density fixed to equilibrium mean)",
        )

        expected_gro_sig = file_signature(Path(scaled_gro))
        expected_top_sig = file_signature(exp.system_top)
        expected_ndx_sig = file_signature(exp.system_ndx)
        _recover_completed_workflow_step(
            self._resume,
            spec,
            summary_path=run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
            label="NVT production",
            fallback_markers=(final_dir / "summary.json",),
        )
        if self._resume.is_done(spec) and not _workflow_summary_matches(
            run_dir / "summary.json",
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
        ):
            _preset_log(
                "[RESTART] Cached NVT production summary does not match current inputs. Rebuilding from scratch.",
                level=1,
            )
            self._resume.mark_failed(spec, error="stale NVT production summary", meta={"auto_rebuild": True})
        job_restart = self._job_restart_flag(spec, bool(rst_flag))
        if bool(rst_flag) and not job_restart:
            latest_stage, next_stage = _latest_reusable_stage_progress(
                run_dir,
                [stage.name for stage in stages],
                input_gro_sig=expected_gro_sig,
                input_top_sig=expected_top_sig,
                input_ndx_sig=expected_ndx_sig,
            )
            if latest_stage is not None:
                detail = f"resuming from {next_stage}" if next_stage is not None else "all stages already present"
                _preset_log(
                    f"[RESTART] Reusing completed NVT production substeps through {latest_stage}; {detail}.",
                    level=1,
                )
                job_restart = True
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        self._resume.run(spec, lambda: job.run(restart=job_restart))
        _write_production_policy_summary(run_dir, policy)
        self._job = job
        _preset_done("NVT production preset", t_all, detail=f"output={run_dir}")
        return self.ac
