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
import math
import os
import shutil
import subprocess
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np

from ...core import utils
from ...gmx.mdp_templates import MdpSpec
from ...gmx.workflows._util import RunResources, _GroAtomRecord, _GroFrameRecord, _read_gro_frame, _write_gro_frame
from ...gmx.workflows.eq import EqStage, EquilibrationJob, StageLincsRetryPolicy
from ...io.gromacs_system import SystemExportResult, export_system_from_cell_meta, validate_exported_system_dir
from ...runtime import resolve_restart
from ...workflow import ResumeManager, StepSpec
from ...workflow.resume import file_signature
from ..analyzer import AnalyzeResult
from ..performance import IOAnalysisPolicy, resolve_io_analysis_policy


_EXPORT_SYSTEM_SCHEMA_VERSION = "0.8.61-export-v2"
_AVOGADRO = 6.02214076e23


@dataclass(frozen=True)
class XYSlabEquilibrationSpec:
    """Wall-confined ``pbc=xy`` slab equilibration for stack-ready polymers."""

    density_mode: Literal["target_active_density", "wall_gap_compression", "wall_z_npt"] = "target_active_density"
    coordinate_export_policy: Literal["wrapped_xy_z_open", "as_is"] = "wrapped_xy_z_open"
    target_density_g_cm3: float | None = 0.50
    target_active_z_nm: float | None = None
    target_box_z_nm: float | None = None
    active_density_min_g_cm3: float | None = None
    cycles: int | Literal["auto"] = "auto"
    max_cycles: int = 30
    max_z_shrink_per_cycle: float = 0.10
    wall_padding_nm: float = 0.40
    wall_mode: str = "12-6"
    wall_atomtype: str | Literal["auto"] | None = "auto"
    wall_r_linpot_nm: float = 0.05
    ewald_geometry: str = "3dc"
    xy_area_mode: Literal["fixed", "xy_npt"] = "fixed"
    pressure_axis_mode: Literal["fixed_xy_z_npt", "xy_npt", "off"] = "fixed_xy_z_npt"
    tau_p_ps: float = 5.0
    z_compressibility_bar_inv: float = 4.5e-5
    xy_compressibility_bar_inv: float = 4.5e-5
    pmax_bar: float = 2000.0
    pre_nvt_ns: float = 0.01
    wall_npt_ns: float = 0.05
    hot_nvt_ns: float = 0.01
    cool_nvt_ns: float = 0.01
    final_relax_ns: float = 0.20
    minimize_nsteps: int = 5000
    final_minimize_nsteps: int = 10000
    active_density_convergence: bool = False
    rg_convergence: bool = False
    lateral_occupancy_convergence: bool = False
    max_convergence_rounds: int = 0
    extra_relax_ns_per_round: float = 0.20
    active_density_tolerance_fraction: float = 0.08
    active_density_rel_std_max: float = 0.08
    active_density_tail_fraction: float = 0.50
    active_density_quantile_low: float = 0.02
    active_density_quantile_high: float = 0.98
    lateral_occupancy_grid_nm: float = 0.50
    min_lateral_occupancy_fraction: float = 0.85
    min_edge_occupancy_fraction: float = 0.80
    surface_flatness_convergence: bool = False
    surface_flatness_grid_nm: float = 0.50
    max_surface_rms_nm: float = 0.35
    max_surface_peak_to_peak_nm: float = 1.00
    connected_void_convergence: bool = False
    void_grid_nm: float = 0.35
    void_atom_radius_nm: float = 0.22
    max_connected_void_fraction: float = 0.20
    xy_compaction_npt: bool = False
    xy_compaction_pressure_bar: float = 1000.0
    xy_compaction_temp_K: float | None = None
    xy_compaction_npt_ns: float = 0.05
    xy_compaction_final_npt_ns: float = 0.02
    xy_compaction_tau_p_ps: float = 5.0
    xy_compaction_compressibility_bar_inv: float = 4.5e-5
    surface_mold_nvt: bool = False
    surface_mold_cycles: int = 0
    surface_mold_z_shrink_per_cycle: float = 0.03
    surface_mold_hot_temp_K: float | None = 420.0
    surface_mold_hot_nvt_ns: float = 0.02
    surface_mold_cool_nvt_ns: float = 0.02
    surface_mold_max_active_density_g_cm3: float | None = 1.80
    surface_mold_stop_when_flat: bool = True
    na_coo_contact_cutoff_nm: float = 0.35
    na_coo_contact_min_fraction: float = 0.75
    rollback_on_failure: bool = True
    write_compression_animation: bool = True
    animation_fps: float = 1.0


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


def _normalize_periodicity(periodicity: str | None) -> Literal["xyz", "xy"]:
    mode = str(periodicity or "xyz").strip().lower()
    if mode not in {"xyz", "xy"}:
        raise ValueError("periodicity must be either 'xyz' or 'xy'.")
    return mode  # type: ignore[return-value]


def _resolve_xy_slab_spec(spec: XYSlabEquilibrationSpec | None) -> XYSlabEquilibrationSpec:
    return spec if spec is not None else XYSlabEquilibrationSpec()


def _merge_mdp_overrides(*items: Optional[dict[str, object]]) -> dict[str, object] | None:
    merged: dict[str, object] = {}
    for item in items:
        if item:
            merged.update(dict(item))
    return merged or None


def _extract_topology_include_path(line: str, *, base: Path) -> Path | None:
    text = line.split(";", 1)[0].strip()
    if not text.startswith("#include"):
        return None
    rest = text[len("#include") :].strip()
    if not rest:
        return None
    if rest[0] in {'"', "<"}:
        closing = '"' if rest[0] == '"' else ">"
        end = rest.find(closing, 1)
        if end > 1:
            rest = rest[1:end]
    else:
        rest = rest.split()[0]
    if not rest:
        return None
    candidate = Path(rest)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


def _iter_topology_related_files(top_path: Path) -> tuple[Path, ...]:
    root = Path(top_path).resolve()
    seen: set[Path] = set()
    candidates: list[Path] = []

    def _add_candidate(path: Path) -> None:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen or not resolved.is_file():
            return
        seen.add(resolved)
        candidates.append(resolved)

    def _visit(path: Path) -> None:
        _add_candidate(path)
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return
        for raw in lines:
            included = _extract_topology_include_path(raw, base=path.parent)
            if included is not None:
                _visit(included)

    _visit(root)
    _add_candidate(root.parent / "ff_parameters.itp")
    mol_dir = root.parent / "molecules"
    if mol_dir.is_dir():
        for path in sorted(mol_dir.glob("**/*.itp")):
            _add_candidate(path)
    return tuple(candidates)


def _iter_top_atomtypes(top_path: Path) -> tuple[str, ...]:
    names: list[str] = []
    atom_section_names: list[str] = []
    for candidate in _iter_topology_related_files(top_path):
        section = ""
        for raw in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]").strip().lower()
                continue
            if section != "atomtypes":
                if section == "atoms" and len(line.split()) >= 2:
                    token = line.split()[1]
                    if token and token not in names and token not in atom_section_names:
                        atom_section_names.append(token)
                continue
            token = line.split()[0]
            if token and token not in names:
                names.append(token)
    return tuple(names + atom_section_names)


def _resolve_xy_wall_atomtype(top_path: Path, requested: str | None) -> str | None:
    available = _iter_top_atomtypes(top_path)
    req = None if requested is None else str(requested).strip()
    if req and req.lower() != "auto":
        if not available or req in available:
            return req
        raise ValueError(
            f"Requested wall_atomtype={req!r} is not present in {top_path}; "
            f"available atomtypes: {', '.join(available[:20])}"
        )
    for name in available:
        if not str(name).upper().startswith("H"):
            return name
    return available[0] if available else None


def xy_slab_mdp_overrides(
    *,
    top_path: str | Path,
    spec: XYSlabEquilibrationSpec | None = None,
    pressure_bar: float = 1.0,
    npt_like: bool = False,
) -> dict[str, object]:
    """Return GROMACS-ready MDP overrides for a z-open, wall-confined slab."""

    slab = _resolve_xy_slab_spec(spec)
    wall_atomtype = _resolve_xy_wall_atomtype(Path(top_path), None if slab.wall_atomtype is None else str(slab.wall_atomtype))
    wall_lines = [
        "nwall                    = 2",
        f"wall_type                = {str(slab.wall_mode)}",
        f"ewald-geometry           = {str(slab.ewald_geometry)}",
    ]
    if wall_atomtype:
        wall_lines.append(f"wall_atomtype            = {wall_atomtype} {wall_atomtype}")
    else:
        raise ValueError(
            f"Could not auto-select a wall atomtype from {Path(top_path)}. "
            "Set XYSlabEquilibrationSpec(wall_atomtype='...') to an atom type present in the topology."
        )
    if slab.wall_r_linpot_nm is not None:
        wall_lines.append(f"wall-r-linpot            = {float(slab.wall_r_linpot_nm):.6g}")
    overrides: dict[str, object] = {
        "pbc": "xy",
        "periodic_molecules": "yes",
        "periodic-molecules": "yes",
        "wall_mdp": "\n".join(wall_lines) + "\n",
    }
    if npt_like:
        axis_mode = str(getattr(slab, "pressure_axis_mode", "fixed_xy_z_npt")).strip().lower()
        if str(slab.xy_area_mode).lower() == "xy_npt":
            axis_mode = "xy_npt"
        if axis_mode == "xy_npt":
            p = float(pressure_bar)
            overrides.update(
                {
                    "pcoupl": "C-rescale",
                    "pcoupltype": "semiisotropic",
                    "tau_p": float(slab.tau_p_ps),
                    "ref_p": f"{p:.6g} {p:.6g}",
                    "compressibility": f"{float(slab.xy_compressibility_bar_inv):.6g} 0",
                }
            )
        elif axis_mode == "fixed_xy_z_npt":
            p = float(pressure_bar)
            overrides.update(
                {
                    "pcoupl": "C-rescale",
                    "pcoupltype": "semiisotropic",
                    "tau_p": float(slab.tau_p_ps),
                    "ref_p": f"{p:.6g} {p:.6g}",
                    "compressibility": f"0 {float(slab.z_compressibility_bar_inv):.6g}",
                }
            )
        else:
            overrides["pcoupl"] = "no"
    return overrides


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(str(name))
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _allow_production_cpu_fallback() -> bool:
    """Return whether production may intentionally continue on CPU after GPU failure."""

    return _env_flag("ALLOW_CPU_FALLBACK_PRODUCTION", False)


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


def _read_gro_lines_coords_box(gro_path: Path) -> tuple[list[str], np.ndarray, tuple[float, float, float]]:
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {gro_path}")
    try:
        nat = int(lines[1].strip())
    except Exception as exc:
        raise ValueError(f"Invalid GRO atom count in {gro_path}") from exc
    atom_lines = lines[2 : 2 + nat]
    if len(atom_lines) != nat:
        raise ValueError(f"GRO atom count mismatch in {gro_path}: expected {nat}, got {len(atom_lines)}")
    coords = np.zeros((nat, 3), dtype=float)
    for i, line in enumerate(atom_lines):
        try:
            coords[i, 0] = float(line[20:28])
            coords[i, 1] = float(line[28:36])
            coords[i, 2] = float(line[36:44])
        except Exception:
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Cannot parse GRO coordinates in {gro_path}: {line!r}")
            coords[i, :] = [float(parts[-3]), float(parts[-2]), float(parts[-1])]
    try:
        raw_box = [float(tok) for tok in lines[2 + nat].split()]
    except Exception as exc:
        raise ValueError(f"Invalid GRO box line in {gro_path}") from exc
    if len(raw_box) < 3:
        raise ValueError(f"GRO box line must contain at least 3 values in {gro_path}")
    return lines, coords, (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]))


def _lateral_occupancy_report(coords_nm: np.ndarray, box_nm: tuple[float, float, float], *, grid_nm: float = 0.50) -> dict[str, Any]:
    coords = np.asarray(coords_nm, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return {
            "grid_nm": float(grid_nm),
            "grid_shape": [0, 0],
            "occupied_cell_count": 0,
            "total_cell_count": 0,
            "occupied_cell_fraction": 0.0,
            "edge_cell_count": 0,
            "edge_occupied_cell_count": 0,
            "edge_occupied_cell_fraction": 0.0,
            "empty_edge_cell_fraction": 1.0,
            "warning": True,
        }
    box_x = max(float(box_nm[0]), 1.0e-9)
    box_y = max(float(box_nm[1]), 1.0e-9)
    grid = max(float(grid_nm), 1.0e-6)
    nx = max(1, int(math.ceil(box_x / grid)))
    ny = max(1, int(math.ceil(box_y / grid)))
    occupied: set[tuple[int, int]] = set()
    for xyz in coords:
        ix = min(nx - 1, max(0, int(math.floor((float(xyz[0]) % box_x) / box_x * nx))))
        iy = min(ny - 1, max(0, int(math.floor((float(xyz[1]) % box_y) / box_y * ny))))
        occupied.add((ix, iy))
    edge_cells = {
        (ix, iy)
        for ix in range(nx)
        for iy in range(ny)
        if ix == 0 or iy == 0 or ix == nx - 1 or iy == ny - 1
    }
    total = max(nx * ny, 1)
    edge_total = max(len(edge_cells), 1)
    edge_occupied = len(occupied.intersection(edge_cells))
    occupied_fraction = float(len(occupied)) / float(total)
    edge_occupied_fraction = float(edge_occupied) / float(edge_total)
    return {
        "grid_nm": float(grid),
        "grid_shape": [int(nx), int(ny)],
        "occupied_cell_count": int(len(occupied)),
        "total_cell_count": int(total),
        "occupied_cell_fraction": float(occupied_fraction),
        "edge_cell_count": int(edge_total),
        "edge_occupied_cell_count": int(edge_occupied),
        "edge_occupied_cell_fraction": float(edge_occupied_fraction),
        "empty_edge_cell_fraction": float(1.0 - edge_occupied_fraction),
        "warning": bool(occupied_fraction < 0.85 or edge_occupied_fraction < 0.80),
    }


def _surface_flatness_report(coords_nm: np.ndarray, box_nm: tuple[float, float, float], *, grid_nm: float = 0.50) -> dict[str, Any]:
    coords = np.asarray(coords_nm, dtype=float)
    grid = max(float(grid_nm), 1.0e-6)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return {
            "available": False,
            "reason": "no_coordinates",
            "grid_nm": float(grid),
            "occupied_cell_count": 0,
            "bottom_surface_rms_nm": None,
            "top_surface_rms_nm": None,
            "max_surface_rms_nm": None,
            "bottom_peak_to_peak_nm": None,
            "top_peak_to_peak_nm": None,
            "max_peak_to_peak_nm": None,
        }
    box_x = max(float(box_nm[0]), 1.0e-9)
    box_y = max(float(box_nm[1]), 1.0e-9)
    nx = max(1, int(math.ceil(box_x / grid)))
    ny = max(1, int(math.ceil(box_y / grid)))
    bottom: dict[tuple[int, int], float] = {}
    top: dict[tuple[int, int], float] = {}
    for xyz in coords:
        ix = min(nx - 1, max(0, int(math.floor((float(xyz[0]) % box_x) / box_x * nx))))
        iy = min(ny - 1, max(0, int(math.floor((float(xyz[1]) % box_y) / box_y * ny))))
        key = (ix, iy)
        z = float(xyz[2])
        bottom[key] = min(bottom.get(key, z), z)
        top[key] = max(top.get(key, z), z)
    if not bottom or not top:
        return {
            "available": False,
            "reason": "no_occupied_cells",
            "grid_nm": float(grid),
            "occupied_cell_count": 0,
            "bottom_surface_rms_nm": None,
            "top_surface_rms_nm": None,
            "max_surface_rms_nm": None,
            "bottom_peak_to_peak_nm": None,
            "top_peak_to_peak_nm": None,
            "max_peak_to_peak_nm": None,
        }
    bottom_values = np.asarray(list(bottom.values()), dtype=float)
    top_values = np.asarray(list(top.values()), dtype=float)
    bottom_rms = float(np.std(bottom_values))
    top_rms = float(np.std(top_values))
    bottom_p2p = float(np.max(bottom_values) - np.min(bottom_values)) if bottom_values.size else 0.0
    top_p2p = float(np.max(top_values) - np.min(top_values)) if top_values.size else 0.0
    return {
        "available": True,
        "grid_nm": float(grid),
        "grid_shape": [int(nx), int(ny)],
        "occupied_cell_count": int(len(bottom)),
        "total_cell_count": int(nx * ny),
        "bottom_surface_mean_nm": float(np.mean(bottom_values)),
        "top_surface_mean_nm": float(np.mean(top_values)),
        "bottom_surface_rms_nm": bottom_rms,
        "top_surface_rms_nm": top_rms,
        "max_surface_rms_nm": float(max(bottom_rms, top_rms)),
        "bottom_peak_to_peak_nm": bottom_p2p,
        "top_peak_to_peak_nm": top_p2p,
        "max_peak_to_peak_nm": float(max(bottom_p2p, top_p2p)),
    }


def _connected_void_report(
    coords_nm: np.ndarray,
    box_nm: tuple[float, float, float],
    *,
    grid_nm: float = 0.35,
    atom_radius_nm: float = 0.22,
) -> dict[str, Any]:
    coords = np.asarray(coords_nm, dtype=float)
    grid = max(float(grid_nm), 1.0e-6)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return {
            "available": False,
            "reason": "no_coordinates",
            "grid_nm": float(grid),
            "atom_radius_nm": float(atom_radius_nm),
            "through_void": True,
            "connected_void_fraction": 1.0,
        }
    box_x = max(float(box_nm[0]), 1.0e-9)
    box_y = max(float(box_nm[1]), 1.0e-9)
    z_min = float(np.min(coords[:, 2]))
    z_max = float(np.max(coords[:, 2]))
    active_z = max(z_max - z_min, grid)
    nx = max(1, int(math.ceil(box_x / grid)))
    ny = max(1, int(math.ceil(box_y / grid)))
    nz = max(2, int(math.ceil(active_z / grid)))
    occupied = np.zeros((nx, ny, nz), dtype=bool)
    radius_cells = max(0, int(math.ceil(float(atom_radius_nm) / grid)))
    for xyz in coords:
        ix0 = min(nx - 1, max(0, int(math.floor((float(xyz[0]) % box_x) / box_x * nx))))
        iy0 = min(ny - 1, max(0, int(math.floor((float(xyz[1]) % box_y) / box_y * ny))))
        iz0 = min(nz - 1, max(0, int(math.floor((float(xyz[2]) - z_min) / active_z * nz))))
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    if dx * dx + dy * dy + dz * dz > radius_cells * radius_cells:
                        continue
                    iz = iz0 + dz
                    if iz < 0 or iz >= nz:
                        continue
                    occupied[(ix0 + dx) % nx, (iy0 + dy) % ny, iz] = True
    empty = ~occupied
    visited = np.zeros_like(empty, dtype=bool)
    q: deque[tuple[int, int, int]] = deque()
    for ix in range(nx):
        for iy in range(ny):
            if empty[ix, iy, 0]:
                visited[ix, iy, 0] = True
                q.append((ix, iy, 0))
    through = False
    count = 0
    while q:
        ix, iy, iz = q.popleft()
        count += 1
        if iz == nz - 1:
            through = True
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            jx = (ix + dx) % nx
            jy = (iy + dy) % ny
            jz = iz + dz
            if jz < 0 or jz >= nz:
                continue
            if empty[jx, jy, jz] and not visited[jx, jy, jz]:
                visited[jx, jy, jz] = True
                q.append((jx, jy, jz))
    total = max(nx * ny * nz, 1)
    empty_count = int(np.count_nonzero(empty))
    return {
        "available": True,
        "grid_nm": float(grid),
        "atom_radius_nm": float(atom_radius_nm),
        "grid_shape": [int(nx), int(ny), int(nz)],
        "active_z_extent_nm": float(active_z),
        "empty_cell_count": int(empty_count),
        "total_cell_count": int(total),
        "empty_cell_fraction": float(empty_count / total),
        "bottom_connected_empty_cell_count": int(count),
        "connected_void_fraction": float(count / total),
        "through_void": bool(through),
    }


def _coord_extent_report(coords_nm: np.ndarray) -> dict[str, Any]:
    coords = np.asarray(coords_nm, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        mins = np.zeros(3, dtype=float)
        maxs = np.zeros(3, dtype=float)
    else:
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
    extent = np.maximum(maxs - mins, 0.0)
    return {
        "coord_min_nm": [float(v) for v in mins],
        "coord_max_nm": [float(v) for v in maxs],
        "active_extent_nm": [float(v) for v in extent],
        "active_z_extent_nm": float(extent[2]),
    }


def _export_xy_slab_prepared_gro(src_gro: Path, prepared_gro: Path, *, policy: str) -> dict[str, Any]:
    """Write the stack-facing prepared slab GRO.

    Stage handoff structures can be whole-molecule canonicalized, which is useful
    for continuing MD but leaves long polymers outside the primary XY image.  A
    stack-facing slab should instead preserve the periodic XY footprint and keep
    only z open, so wrap x/y into the primary box while leaving z untouched.
    """

    frame = _read_gro_frame(Path(src_gro))
    coords = np.asarray([atom.xyz_nm for atom in frame.atoms], dtype=float)
    raw_extent = _coord_extent_report(coords)
    box = tuple(float(v) for v in frame.box_nm)
    outside_xy = int(
        np.count_nonzero(
            (coords[:, 0] < -1.0e-6)
            | (coords[:, 0] >= float(box[0]) + 1.0e-6)
            | (coords[:, 1] < -1.0e-6)
            | (coords[:, 1] >= float(box[1]) + 1.0e-6)
        )
    )
    normalized = str(policy or "wrapped_xy_z_open").strip().lower()
    if normalized not in {"wrapped_xy_z_open", "as_is"}:
        raise ValueError("XYSlabEquilibrationSpec.coordinate_export_policy must be 'wrapped_xy_z_open' or 'as_is'.")
    new_atoms: list[_GroAtomRecord] = []
    if normalized == "wrapped_xy_z_open":
        wrapped_coords = np.array(coords, copy=True)
        wrapped_coords[:, 0] = np.mod(wrapped_coords[:, 0], max(float(box[0]), 1.0e-9))
        wrapped_coords[:, 1] = np.mod(wrapped_coords[:, 1], max(float(box[1]), 1.0e-9))
        title = f"{frame.title[:55]} | yadonpy_wrapped_xy_z_open"
    else:
        wrapped_coords = np.array(coords, copy=True)
        title = frame.title
    for atom, xyz in zip(frame.atoms, wrapped_coords):
        new_atoms.append(
            _GroAtomRecord(
                resnr=atom.resnr,
                resname=atom.resname,
                atomname=atom.atomname,
                atomnr=atom.atomnr,
                xyz_nm=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                vxyz_nm_ps=atom.vxyz_nm_ps,
            )
        )
    _write_gro_frame(Path(prepared_gro), _GroFrameRecord(title=title, atoms=new_atoms, box_nm=box))
    wrapped_extent = _coord_extent_report(wrapped_coords)
    xy_wrapped_ok = bool(
        np.all(wrapped_coords[:, 0] >= -1.0e-6)
        and np.all(wrapped_coords[:, 0] < float(box[0]) + 1.0e-6)
        and np.all(wrapped_coords[:, 1] >= -1.0e-6)
        and np.all(wrapped_coords[:, 1] < float(box[1]) + 1.0e-6)
    )
    report = {
        "coordinate_export_policy": normalized,
        "source_gro": str(Path(src_gro)),
        "prepared_slab_gro": str(Path(prepared_gro)),
        "box_nm": [float(v) for v in box],
        "outside_xy_atom_count_before_wrap": int(outside_xy),
        "xy_wrapped_ok": bool(xy_wrapped_ok),
        "z_open_ok": True,
        "raw_active_extent_nm": raw_extent.get("active_extent_nm"),
        "raw_active_z_extent_nm": raw_extent.get("active_z_extent_nm"),
        "wrapped_active_extent_nm": wrapped_extent.get("active_extent_nm"),
        "wrapped_active_z_extent_nm": wrapped_extent.get("active_z_extent_nm"),
        "raw_coord_min_nm": raw_extent.get("coord_min_nm"),
        "raw_coord_max_nm": raw_extent.get("coord_max_nm"),
        "wrapped_coord_min_nm": wrapped_extent.get("coord_min_nm"),
        "wrapped_coord_max_nm": wrapped_extent.get("coord_max_nm"),
        "lateral_occupancy_after_wrap": _lateral_occupancy_report(wrapped_coords, box),
    }
    return report


def _write_z_rescaled_gro(
    src_gro: Path,
    dst_gro: Path,
    *,
    target_box_z_nm: float,
    wall_padding_nm: float,
) -> dict[str, float]:
    lines, coords, box = _read_gro_lines_coords_box(Path(src_gro))
    nat = int(coords.shape[0])
    old_z_lo = float(np.min(coords[:, 2])) if nat else 0.0
    old_z_hi = float(np.max(coords[:, 2])) if nat else 0.0
    old_extent = max(old_z_hi - old_z_lo, 1.0e-6)
    target_box_z = max(float(target_box_z_nm), 2.0 * float(wall_padding_nm) + 0.10)
    target_lo = float(wall_padding_nm)
    target_hi = max(target_lo + 0.05, target_box_z - float(wall_padding_nm))
    target_extent = max(target_hi - target_lo, 1.0e-6)
    scale = target_extent / old_extent
    new_coords = np.array(coords, copy=True)
    new_coords[:, 2] = target_lo + (coords[:, 2] - old_z_lo) * scale
    new_coords[:, 0] = np.mod(new_coords[:, 0], max(float(box[0]), 1.0e-6))
    new_coords[:, 1] = np.mod(new_coords[:, 1], max(float(box[1]), 1.0e-6))

    out = list(lines)
    for i in range(nat):
        raw = out[2 + i]
        head = raw[:20].ljust(20)
        tail = raw[44:] if len(raw) > 44 else ""
        out[2 + i] = f"{head}{new_coords[i,0]:8.3f}{new_coords[i,1]:8.3f}{new_coords[i,2]:8.3f}{tail}"
    out[2 + nat] = f"{float(box[0]):10.5f}{float(box[1]):10.5f}{target_box_z:10.5f}"
    dst_gro = Path(dst_gro)
    dst_gro.parent.mkdir(parents=True, exist_ok=True)
    dst_gro.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {
        "old_box_z_nm": float(box[2]),
        "new_box_z_nm": float(target_box_z),
        "old_active_z_nm": float(old_extent),
        "new_active_z_nm": float(target_extent),
        "z_scale": float(scale),
    }


def _read_cell_meta_density(ac: object) -> float | None:
    try:
        if hasattr(ac, "HasProp") and ac.HasProp("_yadonpy_cell_meta"):
            meta = json.loads(ac.GetProp("_yadonpy_cell_meta"))
            for key in ("density_g_cm3", "requested_density_g_cm3"):
                value = meta.get(key)
                if value is not None and float(value) > 0.0:
                    return float(value)
    except Exception:
        return None
    return None


def _parse_molecule_counts_from_top(top_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not Path(top_path).is_file():
        return counts
    section = ""
    for raw in Path(top_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip().lower()
            continue
        if section == "molecules":
            parts = line.split()
            if len(parts) >= 2:
                try:
                    counts[parts[0]] = counts.get(parts[0], 0) + int(parts[1])
                except Exception:
                    continue
    return counts


def _parse_molecule_masses_from_itps(top_path: Path) -> dict[str, float]:
    masses: dict[str, float] = {}
    for itp in _iter_topology_related_files(top_path):
        section = ""
        moltype: str | None = None
        mass = 0.0
        for raw in itp.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]").strip().lower()
                continue
            parts = line.split()
            if section == "moleculetype" and moltype is None and parts:
                moltype = parts[0]
            elif section == "atoms" and len(parts) >= 8:
                try:
                    mass += float(parts[7])
                except Exception:
                    continue
        if moltype and mass > 0.0:
            masses[moltype] = mass
    return masses


def _estimate_total_mass_amu_from_top(top_path: Path) -> float | None:
    counts = _parse_molecule_counts_from_top(top_path)
    masses = _parse_molecule_masses_from_itps(top_path)
    total = 0.0
    for name, count in counts.items():
        mw = masses.get(name)
        if mw is None:
            continue
        total += float(mw) * float(count)
    return float(total) if total > 0.0 else None


def _target_xy_slab_box_z_nm(
    *,
    ac: object,
    top_path: Path,
    gro_path: Path,
    spec: XYSlabEquilibrationSpec,
) -> tuple[float, dict[str, object]]:
    _lines, coords, box = _read_gro_lines_coords_box(gro_path)
    density0 = _read_cell_meta_density(ac)
    area_nm2 = max(float(box[0]) * float(box[1]), 1.0e-9)
    target_box = getattr(spec, "target_box_z_nm", None)
    target_active = getattr(spec, "target_active_z_nm", None)
    if target_box is not None and float(target_box) > 0.0:
        target_box_z = max(float(target_box), 2.0 * float(spec.wall_padding_nm) + 0.10)
        active_z = max(target_box_z - 2.0 * float(spec.wall_padding_nm), 0.10)
        source = "explicit_target_box_z_nm"
        target_density = None if spec.target_density_g_cm3 is None else float(spec.target_density_g_cm3)
    elif target_active is not None and float(target_active) > 0.0:
        active_z = max(float(target_active), 0.10)
        target_box_z = float(active_z) + 2.0 * float(spec.wall_padding_nm)
        source = "explicit_target_active_z_nm"
        target_density = None if spec.target_density_g_cm3 is None else float(spec.target_density_g_cm3)
    else:
        if spec.target_density_g_cm3 is None:
            raise ValueError(
                "XYSlabEquilibrationSpec needs one of target_box_z_nm, target_active_z_nm, "
                "or target_density_g_cm3 for explicit wall-gap compression."
            )
        target_density = max(float(spec.target_density_g_cm3), 1.0e-6)
        if density0 is not None and density0 > 0.0:
            active_z = max(float(box[2]) * float(density0) / target_density, 0.10)
            source = "cell_meta_density"
        else:
            mass_amu = _estimate_total_mass_amu_from_top(top_path)
            if mass_amu is None:
                z_extent = float(np.max(coords[:, 2]) - np.min(coords[:, 2])) if coords.size else float(box[2])
                active_z = max(z_extent, 0.10)
                source = "coordinate_extent_fallback"
            else:
                mass_g = float(mass_amu) / _AVOGADRO
                vol_cm3 = mass_g / target_density
                vol_nm3 = vol_cm3 * 1.0e21
                active_z = max(vol_nm3 / area_nm2, 0.10)
                source = "topology_mass"
        target_box_z = float(active_z) + 2.0 * float(spec.wall_padding_nm)
    return target_box_z, {
        "source": source,
        "initial_box_nm": [float(box[0]), float(box[1]), float(box[2])],
        "density_mode": str(spec.density_mode),
        "target_density_g_cm3": None if target_density is None else float(target_density),
        "target_active_z_nm": float(active_z),
        "wall_padding_nm": float(spec.wall_padding_nm),
        "target_box_z_nm": float(target_box_z),
        "target_active_z_nm_input": None if target_active is None else float(target_active),
        "target_box_z_nm_input": None if target_box is None else float(target_box),
    }


def _xy_slab_z_schedule(current_z_nm: float, target_z_nm: float, spec: XYSlabEquilibrationSpec) -> list[float]:
    current = float(current_z_nm)
    target = min(float(target_z_nm), current)
    max_shrink = min(max(float(spec.max_z_shrink_per_cycle), 1.0e-4), 0.95)
    if target >= current * (1.0 - 1.0e-6):
        return []
    if spec.cycles == "auto":
        out: list[float] = []
        z = current
        while z > target * (1.0 + 1.0e-6) and len(out) < int(spec.max_cycles):
            z = max(target, z * (1.0 - max_shrink))
            out.append(float(z))
        return out
    n = max(int(spec.cycles), 0)
    if n <= 0:
        return []
    out = []
    z = current
    for i in range(1, n + 1):
        desired = current + (target - current) * (float(i) / float(n))
        z = max(desired, z * (1.0 - max_shrink))
        out.append(float(max(target, z)))
    if out and out[-1] > target * (1.0 + 1.0e-6):
        out.append(float(target))
    return out[: int(spec.max_cycles)]


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
      - gpu: 0 always wins over gpu_id; gpu_id is ignored and the run is CPU-only

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
    if token in {"balanced", "bonded-gpu", "bonded_gpu"}:
        return "balanced"
    if token in {"conservative", "safe"}:
        return "conservative"
    if token in {"nb-only", "nb_only", "pme-cpu", "pme_cpu", "no-pme-gpu", "no_pme_gpu"}:
        return "nb-only"
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
    traj_ps: Optional[float],
    energy_ps: float,
    log_ps: Optional[float],
    trr_ps: Optional[float],
    velocity_ps: Optional[float],
) -> None:
    dt_ps = float(params["dt"])
    # ``nstxout`` in YadonPy's templates maps to ``nstxout-compressed`` (XTC).
    # It may be disabled when users explicitly select TRR-only coordinates.
    params["nstxout"] = _interval_ps_to_nsteps(dt_ps, traj_ps, disabled_value=0)
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


def _system_needs_gentle_oplsaa_pre_nvt(exp: SystemExportResult | None) -> bool:
    """Return True for OPLS-AA polyelectrolyte cells that need a softer start.

    CMC-Na/OPLS-AA test systems showed that the first unconstrained NVT can
    blow up at 1 fs even when the same topology is stable after a short 0.5 fs
    settling segment.  Restrict the automatic timestep reduction to systems
    that explicitly advertise both OPLS-AA parameters and localized polymer
    charge metadata, so ordinary GAFF/solvent workflows keep their historical
    schedule.
    """

    if exp is None:
        return False
    try:
        species = list(exp.species or [])
    except Exception:
        species = []
    if not species:
        try:
            payload = json.loads(Path(exp.system_meta).read_text(encoding="utf-8"))
            raw_species = payload.get("species") if isinstance(payload, dict) else []
            species = list(raw_species or []) if isinstance(raw_species, list) else []
        except Exception:
            species = []
    for row in species:
        if not isinstance(row, dict):
            continue
        ff_name = str(row.get("ff_name") or row.get("forcefield") or row.get("ff") or "").strip().lower()
        if "opls" not in ff_name:
            continue
        is_polyion = bool(row.get("polyelectrolyte_mode")) or bool(row.get("charge_groups"))
        is_polymer = str(row.get("kind") or "").strip().lower() in {"polymer", "polyelectrolyte"}
        if is_polyion or is_polymer:
            return True
    return False


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


def _format_ps_interval(value: Optional[float]) -> str:
    return "off" if value is None else f"{float(value):.3f} ps"


def _production_required_outputs(final_dir: Path, policy: IOAnalysisPolicy) -> list[Path]:
    """Return restart markers that match the active coordinate stream.

    Production now writes adaptive TRR by default so conductivity workflows can
    use ``gmx current`` directly.  Requiring ``md.xtc`` for TRR-only runs would
    make a successful production stage look incomplete to the restart layer.
    """

    outputs = [final_dir / "md.tpr", final_dir / "md.edr", final_dir / "md.gro"]
    if policy.xtc_ps is not None:
        outputs.append(final_dir / "md.xtc")
    if policy.trr_ps is not None or policy.velocity_ps is not None:
        outputs.append(final_dir / "md.trr")
    return outputs


def _prepare_production_mdp_params(
    *,
    base_params: dict[str, object],
    dt_ps: float,
    constraints: str,
    lincs_iter: int | None,
    lincs_order: int | None,
    traj_ps: Optional[float],
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
        traj_ps=traj_ps,
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
    pre_minimize: bool = False,
) -> list[EqStage]:
    from ...gmx.mdp_templates import MINIM_STEEP_HBONDS_MDP, MINIM_STEEP_MDP

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
    elif bool(pre_minimize):
        pre_min_params = {
            **params,
            "nsteps": int(max(1, int(settle_nsteps))),
            "emtol": 1000.0,
            "emstep": 0.0005,
            "constraints": "none",
            "constraint_algorithm": "none",
        }
        stages.append(
            EqStage(
                f"{stage_index:02d}_pre_minimize",
                "minim",
                MdpSpec(MINIM_STEEP_MDP, pre_min_params),
                strict_constraints=False,
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
    pre_nvt_dt_ps: Optional[float] = None
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


def _eq21_pre_nvt_dt_ps(cfg: EQ21ProtocolConfig) -> float:
    """Resolve the unconstrained pre-NVT timestep.

    The formal EQ21 ladder is normally run with ``dt_ps`` as the base
    timestep.  Some OPLS-AA polyelectrolyte cells are fragile during the first
    unconstrained thermalization because ionized polymer side groups and X-H
    vibrations can turn small residual packing strain into a fast local blow-up.
    ``pre_nvt_dt_ps`` lets the workflow use a gentler first NVT without changing
    the rest of EQ21 or the production timestep.
    """

    if cfg.pre_nvt_dt_ps is not None:
        return min(float(cfg.pre_nvt_dt_ps), float(cfg.dt_ps))
    return _eq21_stage_dt_ps('NVT', cfg.t_anneal_k, None, cfg)


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
    pre_dt = _eq21_pre_nvt_dt_ps(cfg)
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
            "safety_mode": "pre-thermalize-gentle" if cfg.pre_nvt_dt_ps is not None else "pre-thermalize",
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
        "pre_nvt_dt_ps": (None if cfg.pre_nvt_dt_ps is None else float(cfg.pre_nvt_dt_ps)),
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
            "pre_nvt_dt_ps can be smaller than the base dt for fragile OPLS-AA polyelectrolytes; later EQ21 stages still use the regular robust dt policy.",
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
        f"- pre-NVT dt = {params.get('pre_nvt_dt_ps') if params.get('pre_nvt_dt_ps') is not None else params['dt_ps']} ps\n",
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


def _xy_slab_nvt_params(
    *,
    temp: float,
    dt_ps: float,
    nsteps: int,
    gen_vel: str,
    continuation: str,
    wall_overrides: dict[str, object],
) -> dict[str, object]:
    from ...gmx.mdp_templates import default_mdp_params

    p = {
        **default_mdp_params(),
        **dict(wall_overrides),
        "dt": float(dt_ps),
        "nsteps": int(nsteps),
        "ref_t": float(temp),
        "tau_t": 0.1,
        "tcoupl": "V-rescale",
        "gen_vel": str(gen_vel),
        "gen_temp": float(temp),
        "gen_seed": -1,
        "continuation": str(continuation),
        "nstenergy": 1000,
        "nstlog": 1000,
        "nstxout": 10000,
        "nstxout_trr": 10000,
        "nstvout": 10000,
    }
    return p


def _xy_slab_wall_npt_params(
    *,
    temp: float,
    pressure_bar: float,
    dt_ps: float,
    nsteps: int,
    gen_vel: str,
    continuation: str,
    wall_overrides: dict[str, object],
) -> dict[str, object]:
    from ...gmx.mdp_templates import default_mdp_params

    p = {
        **default_mdp_params(),
        **dict(wall_overrides),
        "dt": float(dt_ps),
        "nsteps": int(nsteps),
        "ref_t": float(temp),
        "tau_t": 0.1,
        "tcoupl": "V-rescale",
        "gen_vel": str(gen_vel),
        "gen_temp": float(temp),
        "gen_seed": -1,
        "continuation": str(continuation),
        "ref_p": f"{float(pressure_bar):.6g} {float(pressure_bar):.6g}",
        "nstenergy": 1000,
        "nstlog": 1000,
        "nstxout": 10000,
        "nstxout_trr": 10000,
        "nstvout": 10000,
    }
    return p


def _wall_z_npt_cycle_count(spec: XYSlabEquilibrationSpec) -> int:
    if spec.cycles == "auto":
        return max(1, min(int(spec.max_cycles), 6))
    return max(1, min(int(spec.cycles), int(spec.max_cycles)))


def _wall_z_npt_schedule(
    *,
    temp: float,
    hot_temp: float,
    press: float,
    spec: XYSlabEquilibrationSpec,
) -> list[dict[str, float]]:
    n = _wall_z_npt_cycle_count(spec)
    out: list[dict[str, float]] = []
    for idx in range(n):
        frac = 0.0 if n <= 1 else float(idx) / float(n - 1)
        stage_temp = float(temp) + (float(hot_temp) - float(temp)) * (1.0 - frac)
        stage_press = float(press) + (float(spec.pmax_bar) - float(press)) * (1.0 - frac)
        out.append({"cycle": float(idx + 1), "temperature_K": stage_temp, "pressure_bar": stage_press})
    return out


def _build_xy_slab_wall_z_npt_initial_stages(
    *,
    temp: float,
    press: float,
    hot_temp: float,
    dt_ps: float,
    spec: XYSlabEquilibrationSpec,
    top_path: Path,
) -> tuple[list[EqStage], list[dict[str, float]]]:
    from ...gmx.mdp_templates import MINIM_STEEP_MDP, NPT_NO_CONSTRAINTS_MDP, NVT_NO_CONSTRAINTS_MDP, MdpSpec, default_mdp_params

    base_wall = xy_slab_mdp_overrides(top_path=top_path, spec=spec, pressure_bar=float(press), npt_like=False)
    stages: list[EqStage] = [
        EqStage(
            "01_minimize",
            "minim",
            MdpSpec(
                MINIM_STEEP_MDP,
                {
                    **default_mdp_params(),
                    **dict(base_wall),
                    "nsteps": int(spec.minimize_nsteps),
                    "emtol": 500.0,
                    "emstep": 0.001,
                },
            ),
        ),
        EqStage(
            "02_pre_nvt",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                _xy_slab_nvt_params(
                    temp=float(temp),
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(spec.pre_nvt_ns), float(dt_ps)),
                    gen_vel="yes",
                    continuation="no",
                    wall_overrides=base_wall,
                ),
            ),
        ),
    ]
    schedule = _wall_z_npt_schedule(temp=float(temp), hot_temp=float(hot_temp), press=float(press), spec=spec)
    for rec in schedule:
        idx = int(rec["cycle"])
        wall_npt = xy_slab_mdp_overrides(top_path=top_path, spec=spec, pressure_bar=float(rec["pressure_bar"]), npt_like=True)
        stages.append(
            EqStage(
                f"03_wall_z_npt_c{idx:02d}",
                "npt",
                MdpSpec(
                    NPT_NO_CONSTRAINTS_MDP,
                    _xy_slab_wall_npt_params(
                        temp=float(rec["temperature_K"]),
                        pressure_bar=float(rec["pressure_bar"]),
                        dt_ps=float(dt_ps),
                        nsteps=_ns_to_steps_for_dt(float(spec.wall_npt_ns), float(dt_ps)),
                        gen_vel="no",
                        continuation="yes",
                        wall_overrides=wall_npt,
                    ),
                ),
            )
        )
    final_wall = xy_slab_mdp_overrides(top_path=top_path, spec=spec, pressure_bar=float(press), npt_like=True)
    stages.append(
        EqStage(
            "04_final_wall_z_npt",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                _xy_slab_wall_npt_params(
                    temp=float(temp),
                    pressure_bar=float(press),
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(spec.final_relax_ns), float(dt_ps)),
                    gen_vel="no",
                    continuation="yes",
                    wall_overrides=final_wall,
                ),
            ),
        )
    )
    return stages, schedule


def _build_xy_slab_wall_z_npt_convergence_stage(
    *,
    temp: float,
    press: float,
    dt_ps: float,
    spec: XYSlabEquilibrationSpec,
    top_path: Path,
    round_idx: int,
) -> EqStage:
    from ...gmx.mdp_templates import NPT_NO_CONSTRAINTS_MDP, MdpSpec

    wall_npt = xy_slab_mdp_overrides(top_path=top_path, spec=spec, pressure_bar=float(press), npt_like=True)
    return EqStage(
        f"round_{int(round_idx):02d}_wall_z_npt",
        "npt",
        MdpSpec(
            NPT_NO_CONSTRAINTS_MDP,
            _xy_slab_wall_npt_params(
                temp=float(temp),
                pressure_bar=float(press),
                dt_ps=float(dt_ps),
                nsteps=_ns_to_steps_for_dt(float(spec.extra_relax_ns_per_round), float(dt_ps)),
                gen_vel="no",
                continuation="yes",
                wall_overrides=wall_npt,
            ),
        ),
    )


def _build_xy_slab_cycle_stages(
    *,
    temp: float,
    hot_temp: float,
    dt_ps: float,
    spec: XYSlabEquilibrationSpec,
    wall_overrides: dict[str, object],
    cycle: int,
) -> list[EqStage]:
    from ...gmx.mdp_templates import MINIM_STEEP_MDP, NVT_NO_CONSTRAINTS_MDP, MdpSpec, default_mdp_params

    hot_steps = _ns_to_steps_for_dt(float(spec.hot_nvt_ns), float(dt_ps))
    cool_steps = _ns_to_steps_for_dt(float(spec.cool_nvt_ns), float(dt_ps))
    prefix = f"cycle_{int(cycle):02d}"
    return [
        EqStage(
            f"{prefix}_01_minimize",
            "minim",
            MdpSpec(
                MINIM_STEEP_MDP,
                {
                    **default_mdp_params(),
                    **dict(wall_overrides),
                    "nsteps": int(spec.minimize_nsteps),
                    "emtol": 500.0,
                    "emstep": 0.001,
                },
            ),
        ),
        EqStage(
            f"{prefix}_02_hot_nvt",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                _xy_slab_nvt_params(
                    temp=float(hot_temp),
                    dt_ps=float(dt_ps),
                    nsteps=int(hot_steps),
                    gen_vel="yes" if int(cycle) == 1 else "no",
                    continuation="no" if int(cycle) == 1 else "yes",
                    wall_overrides=wall_overrides,
                ),
            ),
        ),
        EqStage(
            f"{prefix}_03_cool_nvt",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                _xy_slab_nvt_params(
                    temp=float(temp),
                    dt_ps=float(dt_ps),
                    nsteps=int(cool_steps),
                    gen_vel="no",
                    continuation="yes",
                    wall_overrides=wall_overrides,
                ),
            ),
        ),
    ]


def _build_xy_slab_final_stages(
    *,
    temp: float,
    dt_ps: float,
    spec: XYSlabEquilibrationSpec,
    wall_overrides: dict[str, object],
) -> list[EqStage]:
    from ...gmx.mdp_templates import MINIM_STEEP_MDP, NVT_NO_CONSTRAINTS_MDP, MdpSpec, default_mdp_params

    return [
        EqStage(
            "final_01_minimize",
            "minim",
            MdpSpec(
                MINIM_STEEP_MDP,
                {
                    **default_mdp_params(),
                    **dict(wall_overrides),
                    "nsteps": int(spec.final_minimize_nsteps),
                    "emtol": 250.0,
                    "emstep": 0.001,
                },
            ),
        ),
        EqStage(
            "final_02_nvt_release",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                _xy_slab_nvt_params(
                    temp=float(temp),
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(spec.final_relax_ns), float(dt_ps)),
                    gen_vel="no",
                    continuation="yes",
                    wall_overrides=wall_overrides,
                ),
            ),
        ),
    ]


def _build_xy_slab_xy_compaction_stages(
    *,
    temp: float,
    normal_pressure_bar: float,
    dt_ps: float,
    spec: XYSlabEquilibrationSpec,
    hot_wall_overrides: dict[str, object],
    final_wall_overrides: dict[str, object],
) -> list[EqStage]:
    from ...gmx.mdp_templates import MINIM_STEEP_MDP, NPT_NO_CONSTRAINTS_MDP, MdpSpec, default_mdp_params

    hot_temp = float(spec.xy_compaction_temp_K) if spec.xy_compaction_temp_K is not None else float(temp)
    return [
        EqStage(
            "xy_compaction_01_minimize",
            "minim",
            MdpSpec(
                MINIM_STEEP_MDP,
                {
                    **default_mdp_params(),
                    **dict(hot_wall_overrides),
                    "nsteps": int(spec.final_minimize_nsteps),
                    "emtol": 250.0,
                    "emstep": 0.001,
                },
            ),
        ),
        EqStage(
            "xy_compaction_02_hot_xy_npt",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                _xy_slab_wall_npt_params(
                    temp=hot_temp,
                    pressure_bar=float(spec.xy_compaction_pressure_bar),
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(spec.xy_compaction_npt_ns), float(dt_ps)),
                    gen_vel="no",
                    continuation="yes",
                    wall_overrides=hot_wall_overrides,
                ),
            ),
        ),
        EqStage(
            "xy_compaction_03_normal_xy_npt",
            "npt",
            MdpSpec(
                NPT_NO_CONSTRAINTS_MDP,
                _xy_slab_wall_npt_params(
                    temp=float(temp),
                    pressure_bar=float(normal_pressure_bar),
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(spec.xy_compaction_final_npt_ns), float(dt_ps)),
                    gen_vel="no",
                    continuation="yes",
                    wall_overrides=final_wall_overrides,
                ),
            ),
        ),
    ]


def _build_xy_slab_surface_mold_stages(
    *,
    temp: float,
    dt_ps: float,
    spec: XYSlabEquilibrationSpec,
    wall_overrides: dict[str, object],
    cycle: int,
) -> list[EqStage]:
    from ...gmx.mdp_templates import MINIM_STEEP_MDP, NVT_NO_CONSTRAINTS_MDP, MdpSpec, default_mdp_params

    hot_temp = float(spec.surface_mold_hot_temp_K) if spec.surface_mold_hot_temp_K is not None else float(temp)
    prefix = f"surface_mold_{int(cycle):02d}"
    return [
        EqStage(
            f"{prefix}_01_minimize",
            "minim",
            MdpSpec(
                MINIM_STEEP_MDP,
                {
                    **default_mdp_params(),
                    **dict(wall_overrides),
                    "nsteps": int(spec.final_minimize_nsteps),
                    "emtol": 250.0,
                    "emstep": 0.001,
                },
            ),
        ),
        EqStage(
            f"{prefix}_02_hot_nvt",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                _xy_slab_nvt_params(
                    temp=hot_temp,
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(spec.surface_mold_hot_nvt_ns), float(dt_ps)),
                    gen_vel="no",
                    continuation="yes",
                    wall_overrides=wall_overrides,
                ),
            ),
        ),
        EqStage(
            f"{prefix}_03_cool_nvt",
            "nvt",
            MdpSpec(
                NVT_NO_CONSTRAINTS_MDP,
                _xy_slab_nvt_params(
                    temp=float(temp),
                    dt_ps=float(dt_ps),
                    nsteps=_ns_to_steps_for_dt(float(spec.surface_mold_cool_nvt_ns), float(dt_ps)),
                    gen_vel="no",
                    continuation="yes",
                    wall_overrides=wall_overrides,
                ),
            ),
        ),
    ]


def _ns_to_steps_for_dt(ns: float, dt_ps: float) -> int:
    return max(int(round((float(ns) * 1000.0) / float(dt_ps))), 1)


def _gro_atom_records(gro_path: Path) -> tuple[list[dict[str, Any]], tuple[float, float, float]]:
    lines, coords, box = _read_gro_lines_coords_box(Path(gro_path))
    records: list[dict[str, Any]] = []
    for idx, raw in enumerate(lines[2 : 2 + coords.shape[0]], start=1):
        records.append(
            {
                "index": int(idx),
                "resid": raw[:5].strip(),
                "resname": raw[5:10].strip(),
                "atom": raw[10:15].strip(),
                "atomnr": raw[15:20].strip(),
                "xyz_nm": coords[idx - 1].copy(),
            }
        )
    return records, box


def _active_density_rows_from_coords(
    *,
    coords_nm: np.ndarray,
    time_ps: Sequence[float],
    box_nm: np.ndarray,
    total_mass_amu: float,
    q_low: float,
    q_high: float,
) -> list[dict[str, float]]:
    qlo = min(max(float(q_low), 0.0), 0.49)
    qhi = max(min(float(q_high), 1.0), 0.51)
    if qhi <= qlo:
        qlo, qhi = 0.02, 0.98
    coords = np.asarray(coords_nm, dtype=float)
    if coords.ndim == 2:
        coords = coords.reshape((1, coords.shape[0], coords.shape[1]))
    boxes = np.asarray(box_nm, dtype=float)
    if boxes.ndim == 1:
        boxes = np.tile(boxes.reshape((1, 3)), (coords.shape[0], 1))
    mass_g = float(total_mass_amu) / _AVOGADRO
    rows: list[dict[str, float]] = []
    for frame_idx in range(int(coords.shape[0])):
        xyz = coords[frame_idx]
        box = boxes[min(frame_idx, boxes.shape[0] - 1)]
        z = np.asarray(xyz[:, 2], dtype=float)
        if z.size:
            z_lo = float(np.quantile(z, qlo))
            z_hi = float(np.quantile(z, qhi))
            z_min = float(np.min(z))
            z_max = float(np.max(z))
        else:
            z_lo = z_hi = z_min = z_max = 0.0
        active_z = max(float(z_hi - z_lo), 1.0e-9)
        area = max(float(box[0]) * float(box[1]), 1.0e-9)
        density = mass_g / (area * active_z * 1.0e-21)
        rows.append(
            {
                "time_ps": float(time_ps[frame_idx] if frame_idx < len(time_ps) else frame_idx),
                "box_x_nm": float(box[0]),
                "box_y_nm": float(box[1]),
                "box_z_nm": float(box[2]),
                "z_min_nm": z_min,
                "z_max_nm": z_max,
                "z_q_low_nm": z_lo,
                "z_q_high_nm": z_hi,
                "active_z_extent_nm": float(active_z),
                "active_density_g_cm3": float(density),
            }
        )
    return rows


def _active_density_row_from_gro(
    gro_path: Path,
    *,
    total_mass_amu: float,
    spec: XYSlabEquilibrationSpec,
) -> dict[str, float] | None:
    if float(total_mass_amu) <= 0.0:
        return None
    _lines, coords, box = _read_gro_lines_coords_box(Path(gro_path))
    rows = _active_density_rows_from_coords(
        coords_nm=coords,
        time_ps=[0.0],
        box_nm=np.asarray(box, dtype=float),
        total_mass_amu=float(total_mass_amu),
        q_low=float(spec.active_density_quantile_low),
        q_high=float(spec.active_density_quantile_high),
    )
    return rows[0] if rows else None


def _active_density_series(
    *,
    gro_path: Path,
    trajectory_path: Path | None,
    total_mass_amu: float,
    spec: XYSlabEquilibrationSpec,
) -> tuple[list[dict[str, float]], str]:
    if trajectory_path is not None and Path(trajectory_path).is_file():
        try:
            import mdtraj as md

            traj = md.load(str(trajectory_path), top=str(gro_path))
            boxes = traj.unitcell_lengths
            if boxes is None:
                _lines, _coords0, box = _read_gro_lines_coords_box(Path(gro_path))
                boxes = np.tile(np.asarray(box, dtype=float).reshape((1, 3)), (traj.n_frames, 1))
            times = np.asarray(traj.time, dtype=float)
            rows = _active_density_rows_from_coords(
                coords_nm=np.asarray(traj.xyz, dtype=float),
                time_ps=times.tolist(),
                box_nm=np.asarray(boxes, dtype=float),
                total_mass_amu=float(total_mass_amu),
                q_low=float(spec.active_density_quantile_low),
                q_high=float(spec.active_density_quantile_high),
            )
            if rows:
                return rows, "trajectory"
        except Exception:
            pass
    _lines, coords, box = _read_gro_lines_coords_box(Path(gro_path))
    rows = _active_density_rows_from_coords(
        coords_nm=np.asarray(coords, dtype=float),
        time_ps=[0.0],
        box_nm=np.asarray(box, dtype=float),
        total_mass_amu=float(total_mass_amu),
        q_low=float(spec.active_density_quantile_low),
        q_high=float(spec.active_density_quantile_high),
    )
    return rows, "final_gro"


def _active_density_gate(rows: Sequence[dict[str, float]], *, target_density_g_cm3: float | None, spec: XYSlabEquilibrationSpec) -> dict[str, Any]:
    values = np.asarray([float(row.get("active_density_g_cm3", float("nan"))) for row in rows], dtype=float)
    times = np.asarray([float(row.get("time_ps", idx)) for idx, row in enumerate(rows)], dtype=float)
    mask = np.isfinite(values)
    values = values[mask]
    times = times[mask]
    if values.size == 0:
        return {
            "ok": False,
            "reason": "no_active_density_values",
            "target_density_g_cm3": None if target_density_g_cm3 is None else float(target_density_g_cm3),
        }
    tail_fraction = min(max(float(spec.active_density_tail_fraction), 0.05), 1.0)
    tail_n = max(1, int(math.ceil(float(values.size) * tail_fraction)))
    y = values[-tail_n:]
    t = times[-tail_n:]
    mean = float(np.mean(y))
    std = float(np.std(y))
    rel_std = float(std / abs(mean)) if abs(mean) > 1.0e-12 else float("inf")
    target = None if target_density_g_cm3 is None else float(target_density_g_cm3)
    use_target = target is not None and target > 0.0 and str(getattr(spec, "density_mode", "target_active_density")) == "target_active_density"
    min_density = getattr(spec, "active_density_min_g_cm3", None)
    min_density_value = None if min_density is None else float(min_density)
    rel_error = float(abs(mean - float(target)) / float(target)) if use_target else None
    slope = 0.0
    if y.size >= 3 and float(np.max(t) - np.min(t)) > 1.0e-9:
        try:
            slope = float(np.polyfit(t, y, 1)[0])
        except Exception:
            slope = 0.0
    if use_target:
        ok = bool(
            rel_error is not None
            and rel_error <= float(spec.active_density_tolerance_fraction)
            and rel_std <= float(spec.active_density_rel_std_max)
        )
    else:
        ok = bool(rel_std <= float(spec.active_density_rel_std_max))
        if min_density_value is not None and min_density_value > 0.0:
            ok = bool(ok and mean >= min_density_value)
    return {
        "ok": ok,
        "mode": "target_active_density" if use_target else "plateau_only",
        "target_density_g_cm3": target,
        "min_density_g_cm3": min_density_value,
        "mean_g_cm3": mean,
        "std_g_cm3": std,
        "rel_std": rel_std,
        "rel_error": rel_error,
        "slope_g_cm3_per_ps": slope,
        "tail_frame_count": int(y.size),
        "frame_count": int(values.size),
        "criteria": {
            "active_density_tolerance_fraction": float(spec.active_density_tolerance_fraction),
            "active_density_rel_std_max": float(spec.active_density_rel_std_max),
            "active_density_tail_fraction": float(spec.active_density_tail_fraction),
            "active_density_quantile_low": float(spec.active_density_quantile_low),
            "active_density_quantile_high": float(spec.active_density_quantile_high),
        },
        "recommended_action": "ready" if ok else "continue_wall_confined_nvt_or_rebuild_with_more_gentle_compression",
    }


def _xy_slab_geometry_gate(gro_path: Path, *, spec: XYSlabEquilibrationSpec) -> dict[str, Any]:
    _lines, coords, box = _read_gro_lines_coords_box(Path(gro_path))
    if coords.size:
        wrapped = np.array(coords, copy=True)
        wrapped[:, 0] = np.mod(wrapped[:, 0], max(float(box[0]), 1.0e-9))
        wrapped[:, 1] = np.mod(wrapped[:, 1], max(float(box[1]), 1.0e-9))
        outside_xy = int(
            np.count_nonzero(
                (coords[:, 0] < -1.0e-6)
                | (coords[:, 0] >= float(box[0]) + 1.0e-6)
                | (coords[:, 1] < -1.0e-6)
                | (coords[:, 1] >= float(box[1]) + 1.0e-6)
            )
        )
        z_min = float(np.min(coords[:, 2]))
        z_max = float(np.max(coords[:, 2]))
    else:
        wrapped = coords
        outside_xy = 0
        z_min = z_max = 0.0
    occupancy = _lateral_occupancy_report(
        wrapped,
        box,
        grid_nm=float(getattr(spec, "lateral_occupancy_grid_nm", 0.50)),
    )
    flatness = _surface_flatness_report(
        wrapped,
        box,
        grid_nm=float(getattr(spec, "surface_flatness_grid_nm", getattr(spec, "lateral_occupancy_grid_nm", 0.50))),
    )
    voids = _connected_void_report(
        wrapped,
        box,
        grid_nm=float(getattr(spec, "void_grid_nm", 0.35)),
        atom_radius_nm=float(getattr(spec, "void_atom_radius_nm", 0.22)),
    )
    xy_ok = bool(outside_xy == 0 or str(spec.coordinate_export_policy) == "wrapped_xy_z_open")
    z_open_ok = bool(z_min >= -1.0e-5 and z_max <= float(box[2]) + 1.0e-5 and (z_max - z_min) < float(box[2]) - 1.0e-5)
    lateral_ok = bool(
        float(occupancy.get("occupied_cell_fraction", 0.0)) >= float(getattr(spec, "min_lateral_occupancy_fraction", 0.85))
        and float(occupancy.get("edge_occupied_cell_fraction", 0.0)) >= float(getattr(spec, "min_edge_occupancy_fraction", 0.80))
    )
    enforce_lateral = bool(getattr(spec, "lateral_occupancy_convergence", False))
    flatness_rms = flatness.get("max_surface_rms_nm")
    flatness_p2p = flatness.get("max_peak_to_peak_nm")
    surface_ok = bool(
        flatness.get("available")
        and flatness_rms is not None
        and flatness_p2p is not None
        and float(flatness_rms) <= float(getattr(spec, "max_surface_rms_nm", 0.35))
        and float(flatness_p2p) <= float(getattr(spec, "max_surface_peak_to_peak_nm", 1.00))
    )
    enforce_surface = bool(getattr(spec, "surface_flatness_convergence", False))
    void_ok = bool(
        voids.get("available")
        and not bool(voids.get("through_void"))
        and float(voids.get("connected_void_fraction", 1.0)) <= float(getattr(spec, "max_connected_void_fraction", 0.20))
    )
    enforce_void = bool(getattr(spec, "connected_void_convergence", False))
    ok = bool(
        xy_ok
        and z_open_ok
        and (lateral_ok or not enforce_lateral)
        and (surface_ok or not enforce_surface)
        and (void_ok or not enforce_void)
    )
    return {
        "ok": ok,
        "xy_wrapped_ok_after_export": bool(xy_ok),
        "z_open_ok": bool(z_open_ok),
        "lateral_occupancy_ok": bool(lateral_ok),
        "lateral_occupancy_enforced": bool(enforce_lateral),
        "surface_flatness_ok": bool(surface_ok),
        "surface_flatness_enforced": bool(enforce_surface),
        "connected_void_ok": bool(void_ok),
        "connected_void_enforced": bool(enforce_void),
        "outside_xy_atom_count_before_wrap": int(outside_xy),
        "box_nm": [float(v) for v in box],
        "z_min_nm": float(z_min),
        "z_max_nm": float(z_max),
        "active_z_extent_nm": float(max(z_max - z_min, 0.0)),
        "lateral_occupancy": occupancy,
        "surface_flatness": flatness,
        "connected_void": voids,
        "criteria": {
            "min_lateral_occupancy_fraction": float(getattr(spec, "min_lateral_occupancy_fraction", 0.85)),
            "min_edge_occupancy_fraction": float(getattr(spec, "min_edge_occupancy_fraction", 0.80)),
            "lateral_occupancy_grid_nm": float(getattr(spec, "lateral_occupancy_grid_nm", 0.50)),
            "max_surface_rms_nm": float(getattr(spec, "max_surface_rms_nm", 0.35)),
            "max_surface_peak_to_peak_nm": float(getattr(spec, "max_surface_peak_to_peak_nm", 1.00)),
            "surface_flatness_grid_nm": float(getattr(spec, "surface_flatness_grid_nm", 0.50)),
            "void_grid_nm": float(getattr(spec, "void_grid_nm", 0.35)),
            "void_atom_radius_nm": float(getattr(spec, "void_atom_radius_nm", 0.22)),
            "max_connected_void_fraction": float(getattr(spec, "max_connected_void_fraction", 0.20)),
        },
        "recommended_action": (
            "ready"
            if ok
            else "increase_chain_count_reduce_xy_run_graphite_mold_shaping_or_extend_wall_gap_relaxation_before_stack_assembly"
        ),
    }


def _write_active_density_artifacts(
    *,
    out_dir: Path,
    rows: Sequence[dict[str, float]],
    gate: Mapping[str, Any],
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "active_density_timeseries.csv"
    fields = [
        "time_ps",
        "box_x_nm",
        "box_y_nm",
        "box_z_nm",
        "z_min_nm",
        "z_max_nm",
        "z_q_low_nm",
        "z_q_high_nm",
        "active_z_extent_nm",
        "active_density_g_cm3",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    svg_path = out_dir / "active_density_timeseries.svg"
    try:
        import matplotlib.pyplot as plt

        t = [float(row["time_ps"]) for row in rows]
        y = [float(row["active_density_g_cm3"]) for row in rows]
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.plot(t, y, color="#1f77b4", linewidth=1.5, marker="o" if len(rows) <= 8 else None)
        target = gate.get("target_density_g_cm3")
        if target is not None:
            ax.axhline(float(target), color="#333333", linestyle="--", linewidth=1.0, label="target")
        ax.set_xlabel("time / ps")
        ax.set_ylabel("active density / g cm$^{-3}$")
        ax.set_title("CMC-Na active slab density")
        ax.grid(True, alpha=0.25)
        if target is not None:
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(svg_path)
        plt.close(fig)
    except Exception:
        svg_path.write_text("", encoding="utf-8")
    return {"csv": str(csv_path), "svg": str(svg_path)}


def _write_z_density_profile_final(
    *,
    out_dir: Path,
    gro_path: Path,
    total_mass_amu: float,
    bin_nm: float = 0.05,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _lines, coords, box = _read_gro_lines_coords_box(Path(gro_path))
    z = np.asarray(coords[:, 2], dtype=float) if coords.size else np.asarray([], dtype=float)
    z_min = 0.0
    z_max = float(box[2])
    bins = np.arange(z_min, z_max + float(bin_nm), float(bin_nm))
    if bins.size < 2:
        bins = np.asarray([0.0, max(float(box[2]), float(bin_nm))], dtype=float)
    counts, edges = np.histogram(z, bins=bins)
    mass_per_atom_amu = float(total_mass_amu) / max(int(coords.shape[0]), 1)
    area = max(float(box[0]) * float(box[1]), 1.0e-9)
    rows = []
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        volume_nm3 = area * max(float(hi - lo), 1.0e-9)
        density = (float(count) * mass_per_atom_amu / _AVOGADRO) / (volume_nm3 * 1.0e-21)
        rows.append({"z_mid_nm": 0.5 * float(lo + hi), "mass_density_g_cm3": density, "atom_count": int(count)})
    csv_path = out_dir / "z_density_profile_final.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["z_mid_nm", "mass_density_g_cm3", "atom_count"])
        writer.writeheader()
        writer.writerows(rows)
    svg_path = out_dir / "z_density_profile_final.svg"
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.plot([r["z_mid_nm"] for r in rows], [r["mass_density_g_cm3"] for r in rows], color="#2ca02c", linewidth=1.4)
        ax.set_xlabel("z / nm")
        ax.set_ylabel("mass density / g cm$^{-3}$")
        ax.set_title("Final CMC-Na z-density profile")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(svg_path)
        plt.close(fig)
    except Exception:
        svg_path.write_text("", encoding="utf-8")
    return {"csv": str(csv_path), "svg": str(svg_path)}


def _write_xy_slab_compression_animation(
    *,
    run_dir: Path,
    frames: Sequence[Mapping[str, Any]],
    total_mass_amu: float,
    spec: XYSlabEquilibrationSpec,
) -> dict[str, Any]:
    out_dir = Path(run_dir) / "cmcna_eq21_wall_compression_png_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    if not bool(getattr(spec, "write_compression_animation", True)):
        return {"available": False, "reason": "disabled"}
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except Exception as exc:
        return {"available": False, "reason": f"matplotlib_unavailable: {exc}"}

    def _element_color(atom_name: str) -> str:
        name = str(atom_name or "").strip().upper()
        if name.startswith("NA"):
            return "#7E3F98"
        if name.startswith("O"):
            return "#D62728"
        if name.startswith("H"):
            return "#C7C7C7"
        if name.startswith("C"):
            return "#2CA02C"
        return "#1F77B4"

    def _box_edges(box: tuple[float, float, float]) -> list[list[tuple[float, float, float]]]:
        lx, ly, lz = (float(box[0]), float(box[1]), float(box[2]))
        corners = [
            (0.0, 0.0, 0.0),
            (lx, 0.0, 0.0),
            (lx, ly, 0.0),
            (0.0, ly, 0.0),
            (0.0, 0.0, lz),
            (lx, 0.0, lz),
            (lx, ly, lz),
            (0.0, ly, lz),
        ]
        pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        return [[corners[i], corners[j]] for i, j in pairs]

    png_paths: list[Path] = []
    for frame_idx, rec in enumerate(frames):
        gro_path = Path(str(rec.get("gro", "")))
        if not gro_path.is_file():
            continue
        try:
            atoms, box = _gro_atom_records(gro_path)
        except Exception:
            continue
        if not atoms:
            continue
        coords = np.asarray([atom["xyz_nm"] for atom in atoms], dtype=float)
        coords[:, 0] = np.mod(coords[:, 0], max(float(box[0]), 1.0e-9))
        coords[:, 1] = np.mod(coords[:, 1], max(float(box[1]), 1.0e-9))
        density_row = _active_density_rows_from_coords(
            coords_nm=coords,
            time_ps=[float(frame_idx)],
            box_nm=np.asarray(box, dtype=float),
            total_mass_amu=float(total_mass_amu),
            q_low=float(spec.active_density_quantile_low),
            q_high=float(spec.active_density_quantile_high),
        )[0]
        occupancy = _lateral_occupancy_report(
            coords,
            box,
            grid_nm=float(getattr(spec, "lateral_occupancy_grid_nm", 0.50)),
        )
        n_atoms = coords.shape[0]
        max_points = 6000
        if n_atoms > max_points:
            idx = np.linspace(0, n_atoms - 1, max_points).astype(int)
            plot_coords = coords[idx]
            plot_atoms = [atoms[int(i)] for i in idx]
            point_size = 2.0
        else:
            plot_coords = coords
            plot_atoms = atoms
            point_size = 5.0 if n_atoms < 1500 else 2.5
        colors = [_element_color(str(atom.get("atom"))) for atom in plot_atoms]
        fig = plt.figure(figsize=(7.2, 5.6), dpi=130)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(plot_coords[:, 0], plot_coords[:, 1], plot_coords[:, 2], c=colors, s=point_size, alpha=0.75, linewidths=0.0)
        ax.add_collection3d(Line3DCollection(_box_edges(box), colors="#222222", linewidths=0.9, alpha=0.75))
        ax.set_xlim(0.0, float(box[0]))
        ax.set_ylim(0.0, float(box[1]))
        ax.set_zlim(0.0, float(box[2]))
        ax.set_xlabel("x / nm")
        ax.set_ylabel("y / nm")
        ax.set_zlabel("z / nm")
        try:
            ax.set_box_aspect((float(box[0]), float(box[1]), float(box[2])))
        except Exception:
            pass
        ax.view_init(elev=22, azim=-135)
        label = str(rec.get("label", f"frame {frame_idx:03d}"))
        ax.set_title(
            f"{label}\n"
            f"L=({float(box[0]):.2f}, {float(box[1]):.2f}, {float(box[2]):.2f}) nm | "
            f"rho_active={float(density_row['active_density_g_cm3']):.2f} g/cm3 | "
            f"occ={float(occupancy['occupied_cell_fraction']):.2f}",
            fontsize=10,
        )
        fig.tight_layout()
        png_path = out_dir / f"frame_{len(png_paths):04d}.png"
        fig.savefig(png_path)
        plt.close(fig)
        png_paths.append(png_path)
        frame_rows.append(
            {
                "frame": int(len(png_paths) - 1),
                "label": label,
                "gro": str(gro_path),
                "png": str(png_path),
                "box_x_nm": float(box[0]),
                "box_y_nm": float(box[1]),
                "box_z_nm": float(box[2]),
                "active_density_g_cm3": float(density_row["active_density_g_cm3"]),
                "active_z_extent_nm": float(density_row["active_z_extent_nm"]),
                "lateral_occupancy_fraction": float(occupancy["occupied_cell_fraction"]),
                "edge_occupied_fraction": float(occupancy["edge_occupied_cell_fraction"]),
            }
        )
    csv_path = Path(run_dir) / "cmcna_eq21_wall_compression_frames.csv"
    if frame_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_rows)
    else:
        csv_path.write_text("", encoding="utf-8")
        return {"available": False, "reason": "no_renderable_frames", "frames_dir": str(out_dir), "frames_csv": str(csv_path)}
    mp4_path = Path(run_dir) / "cmcna_eq21_wall_compression.mp4"
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        fps = max(float(getattr(spec, "animation_fps", 1.0)), 0.1)
        cmd = [
            str(ffmpeg),
            "-y",
            "-framerate",
            f"{fps:.6g}",
            "-i",
            str(out_dir / "frame_%04d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "24",
            str(mp4_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode != 0:
            return {
                "available": False,
                "reason": "ffmpeg_failed",
                "returncode": int(proc.returncode),
                "stderr_tail": proc.stderr[-2000:],
                "frames_dir": str(out_dir),
                "frames_csv": str(csv_path),
            }
    except Exception as exc:
        return {"available": False, "reason": str(exc), "frames_dir": str(out_dir), "frames_csv": str(csv_path)}
    return {
        "available": True,
        "mp4": str(mp4_path),
        "frames_dir": str(out_dir),
        "frames_csv": str(csv_path),
        "frame_count": int(len(frame_rows)),
        "fps": float(max(float(getattr(spec, "animation_fps", 1.0)), 0.1)),
    }


def _na_coo_contact_report(gro_path: Path, *, cutoff_nm: float, min_fraction: float) -> dict[str, Any]:
    records, box = _gro_atom_records(Path(gro_path))
    na = []
    oxy = []
    for rec in records:
        atom = str(rec.get("atom") or "").strip().upper()
        res = str(rec.get("resname") or "").strip().upper()
        if atom in {"NA", "NA+"} or res in {"NA", "SOD", "YU1"} or atom.startswith("NA"):
            na.append(rec)
        elif atom.startswith("O"):
            oxy.append(rec)
    if not na:
        return {"available": False, "ok": False, "reason": "no_na_atoms_found"}
    if not oxy:
        return {"available": False, "ok": False, "reason": "no_oxygen_atoms_found", "na_count": len(na)}
    box_arr = np.asarray(box, dtype=float)
    oxy_xyz = np.asarray([rec["xyz_nm"] for rec in oxy], dtype=float)
    contacts = 0
    min_distances: list[float] = []
    for rec in na:
        delta = oxy_xyz - np.asarray(rec["xyz_nm"], dtype=float).reshape((1, 3))
        for dim in (0, 1):
            length = float(box_arr[dim])
            if length > 0.0:
                delta[:, dim] -= np.rint(delta[:, dim] / length) * length
        # z is intentionally nonperiodic for an xy slab.
        dist = np.linalg.norm(delta, axis=1)
        nearest = float(np.min(dist)) if dist.size else float("inf")
        min_distances.append(nearest)
        if nearest <= float(cutoff_nm):
            contacts += 1
    fraction = float(contacts) / float(max(len(na), 1))
    return {
        "available": True,
        "ok": bool(fraction >= float(min_fraction)),
        "mode": "na_to_polymer_oxygen_best_effort",
        "cutoff_nm": float(cutoff_nm),
        "min_fraction": float(min_fraction),
        "na_count": int(len(na)),
        "oxygen_candidate_count": int(len(oxy)),
        "contact_count": int(contacts),
        "contact_fraction": fraction,
        "min_distance_mean_nm": float(np.mean(min_distances)) if min_distances else None,
        "min_distance_min_nm": float(np.min(min_distances)) if min_distances else None,
        "min_distance_max_nm": float(np.max(min_distances)) if min_distances else None,
    }


def _rg_convergence_report_for_job(
    *,
    job: EquilibrationJob,
    exp: SystemExportResult,
    out_dir: Path,
    omp: int,
    required: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not bool(required):
        return {"requested": False, "ok": True, "reason": "disabled"}
    try:
        from ..analyzer import AnalyzeResult
        from ...gmx.analysis.rg_convergence import find_rg_convergence, plot_rg_convergence_svg

        tpr, xtc, edr = job.final_outputs()
        trr = None
        try:
            trr = job.final_trr()
        except Exception:
            trr = None
        analy = AnalyzeResult(
            work_dir=Path(out_dir),
            tpr=tpr,
            xtc=xtc,
            trr=trr,
            edr=edr,
            top=Path(exp.system_top),
            ndx=Path(exp.system_ndx),
            omp=int(omp),
        )
        series = analy._rg_series()
        if series is None:
            return {"requested": True, "ok": False, "reason": "rg_series_unavailable"}
        t = np.asarray(series["t_ps"], dtype=float)
        rg = np.asarray(series["rg_nm"], dtype=float)
        comps = series.get("rg_components_nm")
        res = find_rg_convergence(t, rg, comps)
        csv_path = out_dir / "rg_convergence.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            header = ["time_ps", "rg_nm"]
            arrs = [t, rg]
            if comps is not None:
                comp_arr = np.asarray(comps, dtype=float)
                if comp_arr.ndim == 2 and comp_arr.shape[1] >= 3:
                    header.extend(["rg_x_nm", "rg_y_nm", "rg_z_nm"])
                    arrs.extend([comp_arr[:, 0], comp_arr[:, 1], comp_arr[:, 2]])
            writer.writerow(header)
            for i in range(len(t)):
                writer.writerow([float(arr[i]) for arr in arrs])
        svg_path = out_dir / "rg_convergence.svg"
        try:
            plot_rg_convergence_svg(t=t, rg=rg, rg_components=comps, res=res, out_svg=svg_path)
        except Exception:
            svg_path.write_text("", encoding="utf-8")
        return {
            "requested": True,
            "ok": bool(res.ok),
            "converged_by": str(res.converged_by),
            "plateau_start_time_ps": float(res.plateau_start_time),
            "mean_nm": float(res.mean),
            "std_nm": float(res.std),
            "rel_std": float(res.rel_std),
            "slope_per_ps": float(res.slope),
            "group": str(series.get("group")),
            "csv": str(csv_path),
            "svg": str(svg_path),
        }
    except Exception as exc:
        return {"requested": True, "ok": False, "reason": str(exc)}


def _xy_slab_convergence_report(
    *,
    job: EquilibrationJob,
    exp: SystemExportResult,
    run_dir: Path,
    spec: XYSlabEquilibrationSpec,
    omp: int,
) -> dict[str, Any]:
    analysis_dir = Path(run_dir) / "convergence_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    total_mass_amu = _estimate_total_mass_amu_from_top(Path(exp.system_top))
    if total_mass_amu is None:
        total_mass_amu = 0.0
    final_gro = job.final_gro()
    try:
        _tpr, traj, _edr = job.final_outputs()
    except Exception:
        traj = None
    if total_mass_amu > 0.0:
        rows, source = _active_density_series(
            gro_path=final_gro,
            trajectory_path=traj,
            total_mass_amu=float(total_mass_amu),
            spec=spec,
        )
        density_gate = _active_density_gate(
            rows,
            target_density_g_cm3=(None if spec.target_density_g_cm3 is None else float(spec.target_density_g_cm3)),
            spec=spec,
        )
        density_artifacts = _write_active_density_artifacts(out_dir=analysis_dir, rows=rows, gate=density_gate)
        profile_artifacts = _write_z_density_profile_final(
            out_dir=analysis_dir,
            gro_path=final_gro,
            total_mass_amu=float(total_mass_amu),
        )
    else:
        rows = []
        source = "unavailable"
        density_gate = {
            "ok": not bool(spec.active_density_convergence),
            "reason": "topology_mass_unavailable",
            "target_density_g_cm3": None if spec.target_density_g_cm3 is None else float(spec.target_density_g_cm3),
        }
        density_artifacts = {}
        profile_artifacts = {}
    rg_gate = _rg_convergence_report_for_job(
        job=job,
        exp=exp,
        out_dir=analysis_dir,
        omp=int(omp),
        required=bool(spec.rg_convergence),
    )
    contact = _na_coo_contact_report(
        final_gro,
        cutoff_nm=float(spec.na_coo_contact_cutoff_nm),
        min_fraction=float(spec.na_coo_contact_min_fraction),
    )
    geometry_gate = _xy_slab_geometry_gate(final_gro, spec=spec)
    ready = bool(
        (density_gate.get("ok") or not bool(spec.active_density_convergence))
        and (rg_gate.get("ok") or not bool(spec.rg_convergence))
        and (contact.get("ok") if contact.get("available") else True)
        and bool(geometry_gate.get("ok"))
    )
    payload = {
        "schema_version": "0.9.25-cmcna-xy-slab-convergence-v2",
        "ready_for_layer_stack": ready,
        "final_gro": str(final_gro),
        "final_top": str(exp.system_top),
        "trajectory": str(traj) if traj is not None else None,
        "total_mass_amu": float(total_mass_amu),
        "timeseries_source": source,
        "active_density_gate": density_gate,
        "active_density_artifacts": density_artifacts,
        "z_density_profile_final": profile_artifacts,
        "rg_gate": rg_gate,
        "na_coo_contact": contact,
        "geometry_gate": geometry_gate,
    }
    return payload


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

    def _exec_xy_slab(
        self,
        *,
        temp: float,
        press: float,
        mpi: int,
        omp: int,
        gpu: int,
        gpu_id: Optional[int],
        xy_slab: XYSlabEquilibrationSpec | None,
        eq21_tmax: float,
        eq21_dt_ps: float,
        gpu_offload_mode: str,
        restart: bool,
    ):
        slab = _resolve_xy_slab_spec(xy_slab)
        exp = self._ensure_system_exported()
        run_dir = self.work_dir / "03_EQ21_XY_SLAB"
        final_dir = run_dir / "final"
        final_stage_dir = final_dir / "final_02_nvt_release"
        convergence_path = run_dir / "cmcna_slab_convergence.json"
        membrane_quality_path = run_dir / "cmcna_membrane_quality.json"
        prepared_gro = run_dir / "prepared_slab.gro"
        prepared_whole_gro = run_dir / "prepared_slab_whole.gro"
        prepared_top = run_dir / "prepared_slab.top"
        coordinate_report_path = run_dir / "prepared_slab_coordinate_report.json"
        if str(slab.density_mode) == "wall_z_npt":
            wall_dir = run_dir / "wall_z_npt"
            final_stage_dir = wall_dir / "04_final_wall_z_npt"
            summary_path = run_dir / "xy_slab_summary.json"
            _, schedule = _build_xy_slab_wall_z_npt_initial_stages(
                temp=float(temp),
                press=float(press),
                hot_temp=float(eq21_tmax),
                dt_ps=float(eq21_dt_ps),
                spec=slab,
                top_path=exp.system_top,
            )
            outputs = [
                final_stage_dir / "md.tpr",
                final_stage_dir / "md.gro",
                summary_path,
                convergence_path,
                prepared_gro,
                prepared_whole_gro,
                prepared_top,
                coordinate_report_path,
            ]
            xy_spec = StepSpec(
                name="equilibration_eq21_xy_slab_wall_z_npt",
                outputs=outputs,
                inputs={
                    "input_gro_sig": file_signature(exp.system_gro),
                    "input_top_sig": file_signature(exp.system_top),
                    "input_ndx_sig": file_signature(exp.system_ndx),
                    "temp": float(temp),
                    "press": float(press),
                    "eq21_tmax": float(eq21_tmax),
                    "eq21_dt_ps": float(eq21_dt_ps),
                    "xy_slab": json.dumps(asdict(slab), sort_keys=True, default=str),
                    "wall_z_npt_schedule": schedule,
                    "gpu_offload_mode": str(gpu_offload_mode),
                },
                description="Wall-confined pbc=xy fixed-XY/z-only NPT CMC/polymer slab equilibration",
            )
            _preset_item("run_dir", run_dir)
            _preset_item("periodicity", "xy")
            _preset_item("xy_slab_density_mode", "wall_z_npt")
            _preset_item("xy_slab_pressure_axis_mode", str(slab.pressure_axis_mode))
            _preset_item("xy_slab_wall_z_npt_cycles", len(schedule))

            use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
            res = RunResources(
                ntmpi=int(mpi),
                ntomp=int(omp),
                use_gpu=use_gpu,
                gpu_id=gid,
                gpu_offload_mode=_normalize_gpu_offload_mode(gpu_offload_mode),
            )

            def _run_wall_z_npt() -> Path:
                if run_dir.exists():
                    shutil.rmtree(run_dir, ignore_errors=True)
                run_dir.mkdir(parents=True, exist_ok=True)
                stages, schedule_rows = _build_xy_slab_wall_z_npt_initial_stages(
                    temp=float(temp),
                    press=float(press),
                    hot_temp=float(eq21_tmax),
                    dt_ps=float(eq21_dt_ps),
                    spec=slab,
                    top_path=exp.system_top,
                )
                job = EquilibrationJob(
                    gro=exp.system_gro,
                    top=exp.system_top,
                    provenance_ndx=exp.system_ndx,
                    out_dir=wall_dir,
                    stages=stages,
                    resources=res,
                )
                job.run(restart=False)
                current_job = job
                convergence_reports: list[dict[str, Any]] = []
                report = _xy_slab_convergence_report(
                    job=current_job,
                    exp=exp,
                    run_dir=run_dir,
                    spec=slab,
                    omp=int(omp),
                )
                report["round"] = 0
                convergence_reports.append(report)
                convergence_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                for round_idx in range(1, max(int(slab.max_convergence_rounds), 0) + 1):
                    if bool(report.get("ready_for_layer_stack")):
                        break
                    round_dir = run_dir / f"convergence_round_{round_idx:02d}"
                    extra_job = EquilibrationJob(
                        gro=current_job.final_gro(),
                        top=exp.system_top,
                        provenance_ndx=exp.system_ndx,
                        out_dir=round_dir,
                        stages=[
                            _build_xy_slab_wall_z_npt_convergence_stage(
                                temp=float(temp),
                                press=float(press),
                                dt_ps=float(eq21_dt_ps),
                                spec=slab,
                                top_path=exp.system_top,
                                round_idx=round_idx,
                            )
                        ],
                        resources=res,
                    )
                    extra_job.run(restart=False)
                    current_job = extra_job
                    report = _xy_slab_convergence_report(
                        job=current_job,
                        exp=exp,
                        run_dir=run_dir,
                        spec=slab,
                        omp=int(omp),
                    )
                    report["round"] = int(round_idx)
                    convergence_reports.append(report)
                    convergence_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                shutil.copy2(current_job.final_gro(), prepared_whole_gro)
                coordinate_report = _export_xy_slab_prepared_gro(
                    current_job.final_gro(),
                    prepared_gro,
                    policy=str(slab.coordinate_export_policy),
                )
                coordinate_report["prepared_slab_whole_gro"] = str(prepared_whole_gro)
                coordinate_report_path.write_text(json.dumps(coordinate_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                shutil.copy2(exp.system_top, prepared_top)
                self._job = current_job
                payload = {
                    "schema_version": "0.9.25-xy-slab-wall-z-npt-v1",
                    "periodicity": "xy",
                    "density_mode": "wall_z_npt",
                    "spec": asdict(slab),
                    "wall_z_npt_schedule": schedule_rows,
                    "final_gro": str(current_job.final_gro()),
                    "prepared_slab_gro": str(prepared_gro),
                    "prepared_slab_whole_gro": str(prepared_whole_gro),
                    "prepared_slab_top": str(prepared_top),
                    "prepared_slab_coordinate_report": str(coordinate_report_path),
                    "coordinate_export": coordinate_report,
                    "convergence_summary": str(convergence_path),
                    "convergence_history": convergence_reports,
                    "ready_for_layer_stack": bool((convergence_reports[-1] or {}).get("ready_for_layer_stack")),
                }
                summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                return summary_path

            expected_gro_sig = file_signature(exp.system_gro)
            expected_top_sig = file_signature(exp.system_top)
            expected_ndx_sig = file_signature(exp.system_ndx)
            if self._resume.is_done(xy_spec) and not summary_path.exists():
                self._resume.mark_failed(xy_spec, error="missing xy_slab_summary.json", meta={"auto_rebuild": True})
            if not bool(restart) and run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
            self._resume.run(xy_spec, _run_wall_z_npt)
            _recover_completed_workflow_step(
                self._resume,
                xy_spec,
                summary_path=summary_path,
                input_gro_sig=expected_gro_sig,
                input_top_sig=expected_top_sig,
                input_ndx_sig=expected_ndx_sig,
                label="EQ21 xy slab wall-z-NPT workflow",
                fallback_markers=(final_stage_dir / "summary.json",),
            )
            return self.ac

        target_box_z, target_report = _target_xy_slab_box_z_nm(
            ac=self.ac,
            top_path=exp.system_top,
            gro_path=exp.system_gro,
            spec=slab,
        )
        _lines, _coords, initial_box = _read_gro_lines_coords_box(exp.system_gro)
        schedule = _xy_slab_z_schedule(float(initial_box[2]), float(target_box_z), slab)
        wall_overrides = xy_slab_mdp_overrides(top_path=exp.system_top, spec=slab, pressure_bar=float(press), npt_like=False)
        summary_path = run_dir / "xy_slab_summary.json"
        outputs = [
            final_stage_dir / "md.tpr",
            final_stage_dir / "md.gro",
            summary_path,
            convergence_path,
            membrane_quality_path,
            prepared_gro,
            prepared_whole_gro,
            prepared_top,
            coordinate_report_path,
        ]
        xy_spec = StepSpec(
            name="equilibration_eq21_xy_slab",
            outputs=outputs,
            inputs={
                "input_gro_sig": file_signature(exp.system_gro),
                "input_top_sig": file_signature(exp.system_top),
                "input_ndx_sig": file_signature(exp.system_ndx),
                "temp": float(temp),
                "press": float(press),
                "eq21_tmax": float(eq21_tmax),
                "eq21_dt_ps": float(eq21_dt_ps),
                "xy_slab": json.dumps(asdict(slab), sort_keys=True, default=str),
                "target_box_z_nm": float(target_box_z),
                "schedule": [float(v) for v in schedule],
                "gpu_offload_mode": str(gpu_offload_mode),
            },
            description="Wall-confined pbc=xy CMC/polymer slab compression anneal",
        )
        _preset_item("run_dir", run_dir)
        _preset_item("periodicity", "xy")
        _preset_item("xy_slab_density_mode", str(slab.density_mode))
        _preset_item("xy_slab_target_density_g_cm3", None if slab.target_density_g_cm3 is None else float(slab.target_density_g_cm3))
        _preset_item("xy_slab_target_box_z_nm", f"{float(target_box_z):.4f}")
        _preset_item("xy_slab_cycles", len(schedule))
        _preset_item("xy_slab_wall_overrides", wall_overrides)

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=use_gpu,
            gpu_id=gid,
            gpu_offload_mode=_normalize_gpu_offload_mode(gpu_offload_mode),
        )

        def _run() -> Path:
            from ...gmx.mdp_templates import NVT_NO_CONSTRAINTS_MDP

            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            current_gro = Path(exp.system_gro)
            cycle_reports: list[dict[str, object]] = []
            animation_frames: list[dict[str, object]] = [{"label": "initial sparse AC", "gro": str(current_gro)}]
            for i, z_target in enumerate(schedule, start=1):
                cycle_dir = run_dir / f"cycle_{i:02d}"
                start_gro = cycle_dir / "00_geometry" / "start.gro"
                geom = _write_z_rescaled_gro(
                    current_gro,
                    start_gro,
                    target_box_z_nm=float(z_target),
                    wall_padding_nm=float(slab.wall_padding_nm),
                )
                stages = _build_xy_slab_cycle_stages(
                    temp=float(temp),
                    hot_temp=float(eq21_tmax),
                    dt_ps=float(eq21_dt_ps),
                    spec=slab,
                    wall_overrides=wall_overrides,
                    cycle=i,
                )
                job = EquilibrationJob(
                    gro=start_gro,
                    top=exp.system_top,
                    provenance_ndx=exp.system_ndx,
                    out_dir=cycle_dir,
                    stages=stages,
                    resources=res,
                )
                try:
                    job.run(restart=False)
                except Exception:
                    if not bool(slab.rollback_on_failure):
                        raise
                    retry_z = 0.5 * (float(geom["old_box_z_nm"]) + float(z_target))
                    retry_dir = run_dir / f"cycle_{i:02d}_retry"
                    retry_start = retry_dir / "00_geometry" / "start.gro"
                    retry_geom = _write_z_rescaled_gro(
                        current_gro,
                        retry_start,
                        target_box_z_nm=retry_z,
                        wall_padding_nm=float(slab.wall_padding_nm),
                    )
                    retry_job = EquilibrationJob(
                        gro=retry_start,
                        top=exp.system_top,
                        provenance_ndx=exp.system_ndx,
                        out_dir=retry_dir,
                        stages=stages,
                        resources=res,
                    )
                    retry_job.run(restart=False)
                    current_gro = retry_job.final_gro()
                    animation_frames.append({"label": f"cycle {i:02d} geometry retry", "gro": str(retry_start)})
                    animation_frames.append({"label": f"cycle {i:02d} relaxed retry", "gro": str(current_gro)})
                    cycle_reports.append(
                        {
                            "cycle": i,
                            "requested_box_z_nm": float(z_target),
                            "used_box_z_nm": float(retry_z),
                            "retried": True,
                            "geometry": retry_geom,
                            "final_gro": str(current_gro),
                        }
                    )
                    continue
                current_gro = job.final_gro()
                animation_frames.append({"label": f"cycle {i:02d} geometry", "gro": str(start_gro)})
                animation_frames.append({"label": f"cycle {i:02d} relaxed", "gro": str(current_gro)})
                cycle_reports.append(
                    {
                        "cycle": i,
                        "requested_box_z_nm": float(z_target),
                        "used_box_z_nm": float(z_target),
                        "retried": False,
                        "geometry": geom,
                        "final_gro": str(current_gro),
                    }
                )

            xy_compaction_report: dict[str, Any] | None = None
            if bool(getattr(slab, "xy_compaction_npt", False)):
                compaction_dir = run_dir / "xy_compaction"
                compaction_spec = replace(
                    slab,
                    xy_area_mode="xy_npt",
                    pressure_axis_mode="xy_npt",
                    tau_p_ps=float(getattr(slab, "xy_compaction_tau_p_ps", slab.tau_p_ps)),
                    xy_compressibility_bar_inv=float(
                        getattr(slab, "xy_compaction_compressibility_bar_inv", slab.xy_compressibility_bar_inv)
                    ),
                )
                hot_xy_wall = xy_slab_mdp_overrides(
                    top_path=exp.system_top,
                    spec=compaction_spec,
                    pressure_bar=float(slab.xy_compaction_pressure_bar),
                    npt_like=True,
                )
                final_xy_wall = xy_slab_mdp_overrides(
                    top_path=exp.system_top,
                    spec=compaction_spec,
                    pressure_bar=float(press),
                    npt_like=True,
                )
                box_before = _read_gro_lines_coords_box(current_gro)[2]
                xy_job = EquilibrationJob(
                    gro=current_gro,
                    top=exp.system_top,
                    provenance_ndx=exp.system_ndx,
                    out_dir=compaction_dir,
                    stages=_build_xy_slab_xy_compaction_stages(
                        temp=float(temp),
                        normal_pressure_bar=float(press),
                        dt_ps=float(eq21_dt_ps),
                        spec=compaction_spec,
                        hot_wall_overrides=hot_xy_wall,
                        final_wall_overrides=final_xy_wall,
                    ),
                    resources=res,
                )
                xy_job.run(restart=False)
                current_gro = xy_job.final_gro()
                box_after = _read_gro_lines_coords_box(current_gro)[2]
                animation_frames.append({"label": "xy compaction NPT", "gro": str(current_gro)})
                area_before = max(float(box_before[0]) * float(box_before[1]), 1.0e-9)
                area_after = max(float(box_after[0]) * float(box_after[1]), 1.0e-9)
                xy_compaction_report = {
                    "enabled": True,
                    "pressure_bar": float(slab.xy_compaction_pressure_bar),
                    "normal_pressure_bar": float(press),
                    "temperature_K": (
                        float(slab.xy_compaction_temp_K)
                        if slab.xy_compaction_temp_K is not None
                        else float(temp)
                    ),
                    "npt_ns": float(slab.xy_compaction_npt_ns),
                    "final_npt_ns": float(slab.xy_compaction_final_npt_ns),
                    "tau_p_ps": float(compaction_spec.tau_p_ps),
                    "compressibility_bar_inv": float(compaction_spec.xy_compressibility_bar_inv),
                    "box_before_nm": [float(v) for v in box_before],
                    "box_after_nm": [float(v) for v in box_after],
                    "xy_area_before_nm2": float(area_before),
                    "xy_area_after_nm2": float(area_after),
                    "xy_area_scale": float(area_after / area_before),
                    "final_gro": str(current_gro),
                    "wall_mdp_overrides_hot": hot_xy_wall,
                    "wall_mdp_overrides_final": final_xy_wall,
                }
            else:
                xy_compaction_report = {"enabled": False}

            surface_mold_report: dict[str, Any] = {"enabled": False}
            surface_mold_applied = False
            if bool(getattr(slab, "surface_mold_nvt", False)) and int(getattr(slab, "surface_mold_cycles", 0)) > 0:
                mold_reports: list[dict[str, Any]] = []
                mold_dir = run_dir / "surface_mold"
                total_mass_amu = _estimate_total_mass_amu_from_top(Path(exp.system_top)) or 0.0
                max_density = getattr(slab, "surface_mold_max_active_density_g_cm3", None)
                max_density_value = None if max_density is None else float(max_density)
                for mold_idx in range(1, int(slab.surface_mold_cycles) + 1):
                    gate_before = _xy_slab_geometry_gate(current_gro, spec=slab)
                    density_before = _active_density_row_from_gro(
                        current_gro,
                        total_mass_amu=float(total_mass_amu),
                        spec=slab,
                    )
                    if bool(getattr(slab, "surface_mold_stop_when_flat", True)) and bool(gate_before.get("surface_flatness_ok")):
                        mold_reports.append(
                            {
                                "cycle": int(mold_idx),
                                "skipped": True,
                                "reason": "surface_flatness_already_ok",
                                "geometry_gate_before": gate_before,
                                "active_density_before": density_before,
                            }
                        )
                        break
                    if (
                        max_density_value is not None
                        and density_before is not None
                        and float(density_before.get("active_density_g_cm3", 0.0)) >= max_density_value
                    ):
                        mold_reports.append(
                            {
                                "cycle": int(mold_idx),
                                "skipped": True,
                                "reason": "surface_mold_max_active_density_reached",
                                "geometry_gate_before": gate_before,
                                "active_density_before": density_before,
                                "max_active_density_g_cm3": float(max_density_value),
                            }
                        )
                        break
                    box_before = _read_gro_lines_coords_box(current_gro)[2]
                    shrink = min(max(float(slab.surface_mold_z_shrink_per_cycle), 0.0), 0.25)
                    target_z = max(
                        float(box_before[2]) * (1.0 - shrink),
                        2.0 * float(slab.wall_padding_nm) + 0.15,
                    )
                    cycle_dir = mold_dir / f"cycle_{mold_idx:02d}"
                    start_gro = cycle_dir / "00_geometry" / "start.gro"
                    geom = _write_z_rescaled_gro(
                        current_gro,
                        start_gro,
                        target_box_z_nm=float(target_z),
                        wall_padding_nm=float(slab.wall_padding_nm),
                    )
                    mold_job = EquilibrationJob(
                        gro=start_gro,
                        top=exp.system_top,
                        provenance_ndx=exp.system_ndx,
                        out_dir=cycle_dir,
                        stages=_build_xy_slab_surface_mold_stages(
                            temp=float(temp),
                            dt_ps=float(eq21_dt_ps),
                            spec=slab,
                            wall_overrides=wall_overrides,
                            cycle=mold_idx,
                        ),
                        resources=res,
                    )
                    mold_job.run(restart=False)
                    current_gro = mold_job.final_gro()
                    surface_mold_applied = True
                    animation_frames.append({"label": f"surface mold {mold_idx:02d}", "gro": str(current_gro)})
                    gate_after = _xy_slab_geometry_gate(current_gro, spec=slab)
                    density_after = _active_density_row_from_gro(
                        current_gro,
                        total_mass_amu=float(total_mass_amu),
                        spec=slab,
                    )
                    mold_reports.append(
                        {
                            "cycle": int(mold_idx),
                            "skipped": False,
                            "target_box_z_nm": float(target_z),
                            "geometry": geom,
                            "active_density_before": density_before,
                            "active_density_after": density_after,
                            "geometry_gate_before": gate_before,
                            "geometry_gate_after": gate_after,
                            "final_gro": str(current_gro),
                        }
                    )
                    if bool(getattr(slab, "surface_mold_stop_when_flat", True)) and bool(gate_after.get("surface_flatness_ok")):
                        break
                surface_mold_report = {
                    "enabled": True,
                    "cycles_requested": int(slab.surface_mold_cycles),
                    "z_shrink_per_cycle": float(slab.surface_mold_z_shrink_per_cycle),
                    "hot_temp_K": (
                        float(slab.surface_mold_hot_temp_K)
                        if slab.surface_mold_hot_temp_K is not None
                        else float(temp)
                    ),
                    "hot_nvt_ns": float(slab.surface_mold_hot_nvt_ns),
                    "cool_nvt_ns": float(slab.surface_mold_cool_nvt_ns),
                    "max_active_density_g_cm3": max_density_value,
                    "cycles": mold_reports,
                    "applied": bool(surface_mold_applied),
                }

            final_start = final_dir / "00_geometry" / "start.gro"
            final_target_box_z = (
                float(_read_gro_lines_coords_box(current_gro)[2][2])
                if surface_mold_applied
                else float(target_box_z)
            )
            final_geom = _write_z_rescaled_gro(
                current_gro,
                final_start,
                target_box_z_nm=float(final_target_box_z),
                wall_padding_nm=float(slab.wall_padding_nm),
            )
            final_job = EquilibrationJob(
                gro=final_start,
                top=exp.system_top,
                provenance_ndx=exp.system_ndx,
                out_dir=final_dir,
                stages=_build_xy_slab_final_stages(
                    temp=float(temp),
                    dt_ps=float(eq21_dt_ps),
                    spec=slab,
                    wall_overrides=wall_overrides,
                ),
                resources=res,
            )
            final_job.run(restart=False)
            current_job = final_job
            animation_frames.append({"label": "final geometry", "gro": str(final_start)})
            animation_frames.append({"label": "final wall NVT", "gro": str(current_job.final_gro())})
            convergence_reports: list[dict[str, Any]] = []
            report = _xy_slab_convergence_report(
                job=current_job,
                exp=exp,
                run_dir=run_dir,
                spec=slab,
                omp=int(omp),
            )
            report["round"] = 0
            convergence_reports.append(report)
            convergence_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            for round_idx in range(1, max(int(slab.max_convergence_rounds), 0) + 1):
                if bool(report.get("ready_for_layer_stack")):
                    break
                round_dir = run_dir / f"convergence_round_{round_idx:02d}"
                stages = [
                    EqStage(
                        f"round_{round_idx:02d}_nvt",
                        "nvt",
                        MdpSpec(
                            NVT_NO_CONSTRAINTS_MDP,
                            _xy_slab_nvt_params(
                                temp=float(temp),
                                dt_ps=float(eq21_dt_ps),
                                nsteps=_ns_to_steps_for_dt(float(slab.extra_relax_ns_per_round), float(eq21_dt_ps)),
                                gen_vel="no",
                                continuation="yes",
                                wall_overrides=wall_overrides,
                            ),
                        ),
                    )
                ]
                extra_job = EquilibrationJob(
                    gro=current_job.final_gro(),
                    top=exp.system_top,
                    provenance_ndx=exp.system_ndx,
                    out_dir=round_dir,
                    stages=stages,
                    resources=res,
                )
                extra_job.run(restart=False)
                current_job = extra_job
                animation_frames.append({"label": f"convergence round {round_idx:02d}", "gro": str(current_job.final_gro())})
                report = _xy_slab_convergence_report(
                    job=current_job,
                    exp=exp,
                    run_dir=run_dir,
                    spec=slab,
                    omp=int(omp),
                )
                report["round"] = int(round_idx)
                convergence_reports.append(report)
                convergence_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            shutil.copy2(current_job.final_gro(), prepared_whole_gro)
            coordinate_report = _export_xy_slab_prepared_gro(
                current_job.final_gro(),
                prepared_gro,
                policy=str(slab.coordinate_export_policy),
            )
            coordinate_report["prepared_slab_whole_gro"] = str(prepared_whole_gro)
            coordinate_report_path.write_text(json.dumps(coordinate_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            shutil.copy2(exp.system_top, prepared_top)
            total_mass_amu = _estimate_total_mass_amu_from_top(Path(exp.system_top)) or 0.0
            animation_report = _write_xy_slab_compression_animation(
                run_dir=run_dir,
                frames=animation_frames,
                total_mass_amu=float(total_mass_amu),
                spec=slab,
            )
            self._job = current_job
            initial_box_z = float(initial_box[2])
            final_box_z = float(_read_gro_lines_coords_box(current_job.final_gro())[2][2])
            box_z_changed_ok = bool(abs(final_box_z - initial_box_z) > 1.0e-4)
            membrane_quality = {
                "schema_version": "0.9.27-cmcna-membrane-quality-v1",
                "density_mode": str(slab.density_mode),
                "box_z_changed_ok": bool(box_z_changed_ok),
                "initial_box_z_nm": float(initial_box_z),
                "final_box_z_nm": float(final_box_z),
                "target_box_z_nm": float(target_box_z),
                "box_z_change_fraction": float((final_box_z - initial_box_z) / initial_box_z) if initial_box_z > 0.0 else None,
                "final_convergence": convergence_reports[-1] if convergence_reports else {},
                "coordinate_export": coordinate_report,
                "animation": animation_report,
                "xy_compaction": xy_compaction_report,
                "surface_mold": surface_mold_report,
            }
            membrane_quality_path.write_text(json.dumps(membrane_quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            payload = {
                "schema_version": "0.9.27-xy-slab-wall-gap-compression-v1",
                "periodicity": "xy",
                "density_mode": str(slab.density_mode),
                "target": target_report,
                "spec": asdict(slab),
                "wall_mdp_overrides": wall_overrides,
                "cycles": cycle_reports,
                "final_geometry": final_geom,
                "final_gro": str(current_job.final_gro()),
                "prepared_slab_gro": str(prepared_gro),
                "prepared_slab_whole_gro": str(prepared_whole_gro),
                "prepared_slab_top": str(prepared_top),
                "prepared_slab_coordinate_report": str(coordinate_report_path),
                "coordinate_export": coordinate_report,
                "membrane_quality": str(membrane_quality_path),
                "box_z_changed_ok": bool(box_z_changed_ok),
                "compression_animation": animation_report,
                "xy_compaction": xy_compaction_report,
                "surface_mold": surface_mold_report,
                "convergence_summary": str(convergence_path),
                "convergence_history": convergence_reports,
                "ready_for_layer_stack": bool((convergence_reports[-1] or {}).get("ready_for_layer_stack")),
            }
            summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return summary_path

        expected_gro_sig = file_signature(exp.system_gro)
        expected_top_sig = file_signature(exp.system_top)
        expected_ndx_sig = file_signature(exp.system_ndx)
        if self._resume.is_done(xy_spec) and not summary_path.exists():
            self._resume.mark_failed(xy_spec, error="missing xy_slab_summary.json", meta={"auto_rebuild": True})
        if not bool(restart) and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        self._resume.run(xy_spec, _run)
        if self._job is None and (final_stage_dir / "md.gro").exists():
            self._job = EquilibrationJob(
                gro=exp.system_gro,
                top=exp.system_top,
                provenance_ndx=exp.system_ndx,
                out_dir=final_dir,
                stages=_build_xy_slab_final_stages(
                    temp=float(temp),
                    dt_ps=float(eq21_dt_ps),
                    spec=slab,
                    wall_overrides=wall_overrides,
                ),
                resources=res,
            )
        _recover_completed_workflow_step(
            self._resume,
            xy_spec,
            summary_path=summary_path,
            input_gro_sig=expected_gro_sig,
            input_top_sig=expected_top_sig,
            input_ndx_sig=expected_ndx_sig,
            label="EQ21 xy slab workflow",
            fallback_markers=(final_stage_dir / "summary.json",),
        )
        return self.ac

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
        eq21_pre_nvt_dt_ps: Optional[float] = None,
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
        periodicity: Literal["xyz", "xy"] = "xyz",
        xy_slab: XYSlabEquilibrationSpec | None = None,
        gpu_offload_mode: str = "full",
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
        resolved_periodicity = _normalize_periodicity(periodicity)
        if resolved_periodicity == "xy":
            out = self._exec_xy_slab(
                temp=float(temp),
                press=float(press),
                mpi=int(mpi),
                omp=int(omp),
                gpu=int(gpu),
                gpu_id=gpu_id,
                xy_slab=xy_slab,
                eq21_tmax=float(eq21_tmax),
                eq21_dt_ps=float(eq21_dt_ps),
                gpu_offload_mode=str(gpu_offload_mode),
                restart=bool(rst_flag),
            )
            _preset_done("EQ21 xy-slab preset", t_all, detail=f"output={self.work_dir / '03_EQ21_XY_SLAB'}")
            return out

        run_dir = self.work_dir / "03_EQ21"
        if not rst_flag and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        final_ns = 0.8
        if sim_time is not None:
            final_ns = float(sim_time)
        if time is not None:
            final_ns = float(time)

        resolved_pre_nvt_dt_ps = None if eq21_pre_nvt_dt_ps is None else float(eq21_pre_nvt_dt_ps)
        if resolved_pre_nvt_dt_ps is None and _system_needs_gentle_oplsaa_pre_nvt(exp):
            resolved_pre_nvt_dt_ps = min(float(eq21_dt_ps), 0.0005)

        cfg = EQ21ProtocolConfig(
            t_max_k=float(eq21_tmax),
            t_anneal_k=float(temp if eq21_tanal is None else eq21_tanal),
            p_max_bar=float(eq21_pmax),
            p_anneal_bar=float(press if eq21_panal is None else eq21_panal),
            dt_ps=float(eq21_dt_ps),
            pre_nvt_dt_ps=resolved_pre_nvt_dt_ps,
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
        resolved_gpu_offload_mode = _normalize_gpu_offload_mode(gpu_offload_mode)
        _preset_item(
            "resources",
            f"mpi={int(mpi)} | omp={int(omp)} | gpu={int(gpu)} | gpu_id={gpu_id} | gpu_offload_mode={resolved_gpu_offload_mode}",
        )
        _print_eq21_schedule(stage_records, params)

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid, gpu_offload_mode=resolved_gpu_offload_mode)
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
                "eq21_pre_nvt_dt_ps": (None if cfg.pre_nvt_dt_ps is None else float(cfg.pre_nvt_dt_ps)),
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
                "gpu_offload_mode": str(resolved_gpu_offload_mode),
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

        prepared = self.work_dir / "03_EQ21_XY_SLAB" / "prepared_slab.gro"
        if prepared.is_file():
            return prepared
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
        trajectory_format: Union[str, None] = "auto",
        max_trajectory_frames: Optional[int] = None,
        max_atom_frames: Optional[float] = None,
        bridge_ps: Optional[float] = None,
        bridge_dt_fs: float = 1.0,
        bridge_lincs_iter: int = 4,
        bridge_lincs_order: int = 12,
        mdp_overrides: Optional[dict[str, object]] = None,
        periodicity: Literal["xyz", "xy"] = "xyz",
        xy_slab: XYSlabEquilibrationSpec | None = None,
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
            trajectory_format=trajectory_format,
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
        resolved_periodicity = _normalize_periodicity(periodicity)
        _preset_item("periodicity", resolved_periodicity)
        _preset_item("lincs_iter", int(lincs_iter) if lincs_iter is not None else None)
        _preset_item("lincs_order", int(lincs_order) if lincs_order is not None else None)
        resolved_gpu_offload_mode = _resolve_production_gpu_offload_mode(self.ac, gpu_offload_mode)
        resolved_bridge_ps = _resolve_production_bridge_ps(self.ac, bridge_ps)
        allow_cpu_fallback = _allow_production_cpu_fallback()
        _preset_item(
            "performance_policy",
            f"{policy.policy_level} | profile={policy.performance_profile} | "
            f"analysis={policy.analysis_profile} | format={policy.trajectory_format} | frames~{policy.estimated_frames}",
        )
        _preset_item(
            "output_cadence",
            f"xtc={_format_ps_interval(policy.xtc_ps)} | energy={policy.energy_ps:.3f} ps | "
            f"log={policy.log_ps:.3f} ps | "
            f"trr={_format_ps_interval(policy.trr_ps)} | "
            f"vel={_format_ps_interval(policy.velocity_ps)} | "
            f"cpt={float(checkpoint_min):.2f} min"
        )
        if policy.large_file_warnings:
            _preset_item("large_file_warnings", ", ".join(policy.large_file_warnings))
        _preset_item("gpu_offload_mode", resolved_gpu_offload_mode)
        _preset_item("cpu_fallback", "allowed" if allow_cpu_fallback else "disabled for production GPU failures")
        _preset_item(
            "bridge",
            f"{float(resolved_bridge_ps):.1f} ps | dt={float(bridge_dt_fs):.3f} fs | "
            f"lincs_iter={int(bridge_lincs_iter)} | lincs_order={int(bridge_lincs_order)}",
        )
        effective_mdp_overrides = mdp_overrides
        if resolved_periodicity == "xy":
            effective_mdp_overrides = _merge_mdp_overrides(
                xy_slab_mdp_overrides(
                    top_path=exp.system_top,
                    spec=xy_slab,
                    pressure_bar=float(press),
                    npt_like=True,
                ),
                mdp_overrides,
            )
        if effective_mdp_overrides:
            _preset_item("mdp_overrides", effective_mdp_overrides)
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
            traj_ps=policy.xtc_ps,
            energy_ps=policy.energy_ps,
            log_ps=policy.log_ps,
            trr_ps=policy.trr_ps,
            velocity_ps=policy.velocity_ps,
            mdp_overrides=effective_mdp_overrides,
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
            allow_cpu_fallback_on_gpu_error=allow_cpu_fallback,
        )
        job = EquilibrationJob(gro=start_gro, top=exp.system_top, provenance_ndx=exp.system_ndx, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name
        spec = StepSpec(
            name="npt_production",
            outputs=_production_required_outputs(final_dir, policy),
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
                "xtc_ps": float(policy.xtc_ps) if policy.xtc_ps is not None else None,
                "energy_ps": float(policy.energy_ps),
                "log_ps": float(policy.log_ps),
                "trr_ps": float(policy.trr_ps) if policy.trr_ps is not None else None,
                "velocity_ps": float(policy.velocity_ps) if policy.velocity_ps is not None else None,
                "trajectory_format": policy.trajectory_format,
                "performance_policy": json.dumps(policy.to_dict(), sort_keys=True, default=str),
                "checkpoint_min": float(checkpoint_min),
                "gpu_offload_mode": resolved_gpu_offload_mode,
                "allow_cpu_fallback_on_gpu_error": bool(allow_cpu_fallback),
                "bridge_ps": float(resolved_bridge_ps),
                "bridge_dt_fs": float(bridge_dt_fs),
                "bridge_lincs_iter": int(bridge_lincs_iter),
                "bridge_lincs_order": int(bridge_lincs_order),
                "periodicity": str(resolved_periodicity),
                "xy_slab": json.dumps(asdict(_resolve_xy_slab_spec(xy_slab)), sort_keys=True, default=str) if resolved_periodicity == "xy" else None,
                "mdp_overrides": json.dumps(effective_mdp_overrides, sort_keys=True, default=str) if effective_mdp_overrides else None,
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
        _write_production_policy_summary(final_dir, policy)
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
        trajectory_format: Union[str, None] = "auto",
        max_trajectory_frames: Optional[int] = None,
        max_atom_frames: Optional[float] = None,
        bridge_ps: Optional[float] = None,
        bridge_dt_fs: float = 1.0,
        bridge_lincs_iter: int = 4,
        bridge_lincs_order: int = 12,
        pre_minimize: bool = False,
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
            trajectory_format=trajectory_format,
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
        allow_cpu_fallback = _allow_production_cpu_fallback()
        _preset_item("density_control", bool(resolved_density_control))
        _preset_item("density_frac_last", float(density_frac_last))
        _preset_item(
            "performance_policy",
            f"{policy.policy_level} | profile={policy.performance_profile} | "
            f"analysis={policy.analysis_profile} | format={policy.trajectory_format} | frames~{policy.estimated_frames}",
        )
        _preset_item(
            "output_cadence",
            f"xtc={_format_ps_interval(policy.xtc_ps)} | energy={policy.energy_ps:.3f} ps | "
            f"log={policy.log_ps:.3f} ps | "
            f"trr={_format_ps_interval(policy.trr_ps)} | "
            f"vel={_format_ps_interval(policy.velocity_ps)} | "
            f"cpt={float(checkpoint_min):.2f} min"
        )
        if policy.large_file_warnings:
            _preset_item("large_file_warnings", ", ".join(policy.large_file_warnings))
        _preset_item("gpu_offload_mode", resolved_gpu_offload_mode)
        _preset_item("cpu_fallback", "allowed" if allow_cpu_fallback else "disabled for production GPU failures")
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
            traj_ps=policy.xtc_ps,
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
            pre_minimize=bool(pre_minimize),
        )

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=use_gpu,
            gpu_id=gid,
            gpu_offload_mode=resolved_gpu_offload_mode,
            allow_cpu_fallback_on_gpu_error=allow_cpu_fallback,
        )
        job = EquilibrationJob(gro=scaled_gro, top=exp.system_top, provenance_ndx=exp.system_ndx, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name
        spec = StepSpec(
            name="nvt_production",
            outputs=_production_required_outputs(final_dir, policy),
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
                "xtc_ps": float(policy.xtc_ps) if policy.xtc_ps is not None else None,
                "energy_ps": float(policy.energy_ps),
                "log_ps": float(policy.log_ps),
                "trr_ps": float(policy.trr_ps) if policy.trr_ps is not None else None,
                "velocity_ps": float(policy.velocity_ps) if policy.velocity_ps is not None else None,
                "trajectory_format": policy.trajectory_format,
                "performance_policy": json.dumps(policy.to_dict(), sort_keys=True, default=str),
                "checkpoint_min": float(checkpoint_min),
                "gpu_offload_mode": resolved_gpu_offload_mode,
                "allow_cpu_fallback_on_gpu_error": bool(allow_cpu_fallback),
                "bridge_ps": float(resolved_bridge_ps),
                "bridge_dt_fs": float(bridge_dt_fs),
                "bridge_lincs_iter": int(bridge_lincs_iter),
                "bridge_lincs_order": int(bridge_lincs_order),
                "pre_minimize": bool(pre_minimize),
                "mdp_overrides": json.dumps(mdp_overrides, sort_keys=True, default=str) if mdp_overrides else None,
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
        _write_production_policy_summary(final_dir, policy)
        self._job = job
        _preset_done("NVT production preset", t_all, detail=f"output={run_dir}")
        return self.ac
