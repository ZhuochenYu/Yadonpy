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
from ...gmx.workflows._util import RunResources
from ...gmx.workflows.eq import EqStage, EquilibrationJob, StageLincsRetryPolicy
from ...io.gromacs_system import SystemExportResult, export_system_from_cell_meta, validate_exported_system_dir
from ...runtime import resolve_restart
from ...workflow import ResumeManager, StepSpec
from ...workflow.resume import file_signature
from ..analyzer import AnalyzeResult


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


def _recover_completed_workflow_step(
    resume: ResumeManager,
    spec: StepSpec,
    *,
    summary_path: Path,
    input_gro_sig: dict[str, Any],
    input_top_sig: dict[str, Any],
    input_ndx_sig: dict[str, Any] | None = None,
    label: str,
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
    if not _workflow_summary_matches(
        summary_path,
        input_gro_sig=input_gro_sig,
        input_top_sig=input_top_sig,
        input_ndx_sig=input_ndx_sig,
    ):
        return False

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

    if rounds and restart:
        idx, d = rounds[-1]
        # consider complete if final md files exist
        if not ((d / "04_md" / "md.tpr").exists() and (d / "04_md" / "md.xtc").exists() and (d / "04_md" / "md.edr").exists() and (d / "04_md" / "md.gro").exists()):
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
            3) Main EQ21 run (new layout: 03_EQ21/03_EQ21/step_*/md.gro)
            4) Legacy main EQ run (03_eq/04_md/md.gro)
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
            candidates = sorted([p for p in d.glob('*/md.gro') if p.is_file()])
            gro = candidates[-1] if candidates else (d / "04_md" / "md.gro")
            if gro.exists() and not _is_within_any(gro, excluded):
                rounds.append((idx, gro))
        if rounds:
            rounds.sort(key=lambda x: x[0])
            return rounds[-1][1]

    eq21_root = wd / "03_EQ21"
    if eq21_root.exists():
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
            'tau_p_ps': max(base_tau, 2.0 if is_final else base_tau),
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


def _apply_stage_mdp_overrides(stages: Sequence[EqStage], overrides: Optional[dict[str, object]], *, stage_kinds: Sequence[str] = ("npt", "md")) -> list[EqStage]:
    from ...gmx.mdp_templates import MdpSpec

    if not overrides:
        return list(stages)
    target_kinds = {str(kind) for kind in stage_kinds}
    patched: list[EqStage] = []
    for stage in stages:
        if stage.kind in target_kinds:
            patched.append(EqStage(stage.name, stage.kind, MdpSpec(stage.mdp.template, {**stage.mdp.params, **dict(overrides)})))
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
        for col in ('Density', 'Total Energy', 'Box-X', 'Box-Y', 'Box-Z'):
            if col in df.columns:
                out[col] = np.asarray(df[col].to_numpy(dtype=float), dtype=float)
        return out
    except Exception:
        return None


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
                prefixes=("equilibration_additional_",),
            )
        job_restart = self._job_restart_flag(eq_spec, bool(rst_flag))
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        self._resume.run(eq_spec, lambda: job.run(restart=job_restart))
        overview_svg = _write_eq21_overview_plot(run_dir, stage_records, params)
        if overview_svg is not None:
            print(f"[EQ21] Overview plot written: {overview_svg}")
        self._job = job
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
        mdp_overrides: Optional[dict[str, object]] = None,
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
        start_gro = _find_latest_equilibrated_gro(self.work_dir) or exp.system_gro

        # 判定标签：若已有可用的平衡结构（即 start_gro 不是原始 system.gro），
        # 则无需重复建盒子/EM/NVT/NPT 等流程。
        # 只在原有基础上重复跑最终平衡阶段（04_md）。
        _skip_rebuild = bool(start_gro != exp.system_gro)

        round_idx, run_dir = _next_additional_round(self.work_dir, restart=bool(rst_flag))
        _preset_item("round", round_idx)
        _preset_item("run_dir", run_dir)
        _preset_item("start_gro", start_gro)
        _preset_item("temperature_K", float(temp))
        _preset_item("pressure_bar", float(press))
        _preset_item("production_ns", float(sim_time))
        if mdp_overrides:
            _preset_item("mdp_overrides", mdp_overrides)

        stages = EquilibrationJob.default_stages(
            temperature_k=float(temp),
            pressure_bar=float(press),
            nvt_ns=0.1,
            npt_ns=0.2,
            prod_ns=float(sim_time),
        )
        stages = _apply_stage_mdp_overrides(stages, mdp_overrides)

        # 判定标签：只保留最终平衡阶段（04_md）
        if _skip_rebuild and stages:
            stages = [stages[-1]]

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid)
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
                prefixes=("equilibration_additional_",),
            )
        job_restart = self._job_restart_flag(spec, bool(rst_flag))
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
        traj_ps: float = 2.0,
        energy_ps: float = 2.0,
        log_ps: Optional[float] = None,
        trr_ps: Optional[float] = None,
        velocity_ps: Optional[float] = None,
        checkpoint_min: float = 5.0,
        mdp_overrides: Optional[dict[str, object]] = None,
        restart: Optional[bool] = None,
    ):
        t_all = _preset_section("NPT production preset", detail=f"restart={bool(resolve_restart(restart))} | work_dir={self.work_dir}")
        rst_flag = resolve_restart(restart)
        self._resume.enabled = bool(rst_flag)
        exp = self._ensure_system_exported()

        run_dir = self.work_dir / "05_npt_production"
        # Production must restart from upstream equilibration, not from its own prior output.
        start_gro = _find_latest_equilibrated_gro(self.work_dir, exclude_dirs=[run_dir]) or exp.system_gro
        _preset_item("run_dir", run_dir)
        _preset_item("start_gro", start_gro)
        _preset_item("temperature_K", float(temp))
        _preset_item("pressure_bar", float(press))
        _preset_item("production_ns", float(time))
        _preset_item(
            "output_cadence",
            f"xtc={float(traj_ps):.3f} ps | energy={float(energy_ps):.3f} ps | "
            f"log={float(log_ps if log_ps is not None else energy_ps):.3f} ps | "
            f"trr={'off' if trr_ps is None else f'{float(trr_ps):.3f} ps'} | "
            f"vel={'off' if velocity_ps is None else f'{float(velocity_ps):.3f} ps'} | "
            f"cpt={float(checkpoint_min):.2f} min"
        )
        if mdp_overrides:
            _preset_item("mdp_overrides", mdp_overrides)
        _preset_item("resources", f"mpi={int(mpi)} | omp={int(omp)} | gpu={int(gpu)} | gpu_id={gpu_id}")
        if not rst_flag and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        # Build a single-stage NPT MD using the same NPT MDP template as equilibration.
        from ...gmx.mdp_templates import NPT_MDP, MdpSpec, default_mdp_params
        from ...gmx.workflows.eq import EqStage

        p = default_mdp_params()
        p["ref_t"] = float(temp)
        p["ref_p"] = float(press)
        _apply_production_output_cadence(
            p,
            traj_ps=float(traj_ps),
            energy_ps=float(energy_ps),
            log_ps=log_ps,
            trr_ps=trr_ps,
            velocity_ps=velocity_ps,
        )
        if mdp_overrides:
            p.update(dict(mdp_overrides))

        def ns_to_steps(ns: float) -> int:
            return int((ns * 1000.0) / float(p["dt"]))

        stages = [
            EqStage(
                "01_npt",
                "md",
                MdpSpec(NPT_MDP, {**p, "nsteps": max(ns_to_steps(float(time)), 1000)}),
                lincs_retry=StageLincsRetryPolicy(),
                checkpoint_minutes=(float(checkpoint_min) if float(checkpoint_min) > 0.0 else None),
            )
        ]

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid)
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
                "traj_ps": float(traj_ps),
                "energy_ps": float(energy_ps),
                "log_ps": float(log_ps) if log_ps is not None else None,
                "trr_ps": float(trr_ps) if trr_ps is not None else None,
                "velocity_ps": float(velocity_ps) if velocity_ps is not None else None,
                "checkpoint_min": float(checkpoint_min),
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
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        self._resume.run(spec, lambda: job.run(restart=job_restart))
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
        traj_ps: float = 2.0,
        energy_ps: float = 2.0,
        log_ps: Optional[float] = None,
        trr_ps: Optional[float] = None,
        velocity_ps: Optional[float] = None,
        checkpoint_min: float = 5.0,
        restart: Optional[bool] = None,
        density_control: bool = True,
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
        _preset_item("run_dir", run_dir)
        _preset_item("start_gro", start_gro)
        _preset_item("temperature_K", float(temp))
        _preset_item("production_ns", float(time))
        _preset_item("density_control", bool(density_control))
        _preset_item("density_frac_last", float(density_frac_last))
        _preset_item(
            "output_cadence",
            f"xtc={float(traj_ps):.3f} ps | energy={float(energy_ps):.3f} ps | "
            f"log={float(log_ps if log_ps is not None else energy_ps):.3f} ps | "
            f"trr={'off' if trr_ps is None else f'{float(trr_ps):.3f} ps'} | "
            f"vel={'off' if velocity_ps is None else f'{float(velocity_ps):.3f} ps'} | "
            f"cpt={float(checkpoint_min):.2f} min"
        )
        _preset_item("resources", f"mpi={int(mpi)} | omp={int(omp)} | gpu={int(gpu)} | gpu_id={gpu_id}")
        if not rst_flag and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        # Density control (optional): scale the starting gro to an equilibrium-average density.
        scaled_gro = start_gro
        if density_control:
            from ...gmx.engine import GromacsRunner
            from ...gmx.analysis.xvg import read_xvg
            from ...gmx.analysis.thermo import stats_from_xvg
            import numpy as np
            import uuid
            import json

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

        # Build a single-stage NVT MD.
        from ...gmx.mdp_templates import NVT_MDP, MdpSpec, default_mdp_params
        from ...gmx.workflows.eq import EqStage

        p = default_mdp_params()
        p["ref_t"] = float(temp)
        _apply_production_output_cadence(
            p,
            traj_ps=float(traj_ps),
            energy_ps=float(energy_ps),
            log_ps=log_ps,
            trr_ps=trr_ps,
            velocity_ps=velocity_ps,
        )

        def ns_to_steps(ns: float) -> int:
            return int((ns * 1000.0) / float(p["dt"]))

        stages = [
            EqStage(
                "01_nvt",
                "md",
                MdpSpec(NVT_MDP, {**p, "nsteps": max(ns_to_steps(float(time)), 1000)}),
                lincs_retry=StageLincsRetryPolicy(),
                checkpoint_minutes=(float(checkpoint_min) if float(checkpoint_min) > 0.0 else None),
            )
        ]

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid)
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
                "traj_ps": float(traj_ps),
                "energy_ps": float(energy_ps),
                "log_ps": float(log_ps) if log_ps is not None else None,
                "trr_ps": float(trr_ps) if trr_ps is not None else None,
                "velocity_ps": float(velocity_ps) if velocity_ps is not None else None,
                "checkpoint_min": float(checkpoint_min),
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
                "density_control": bool(density_control),
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
        if not job_restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        self._resume.run(spec, lambda: job.run(restart=job_restart))
        self._job = job
        _preset_done("NVT production preset", t_all, detail=f"output={run_dir}")
        return self.ac
