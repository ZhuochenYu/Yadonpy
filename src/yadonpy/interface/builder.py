from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np

try:
    from numba import jit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def jit(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator

from ..core.logging_utils import print_done, print_item, print_section, print_stat, print_step
from ..core.workdir import WorkDir, workdir
from ..gmx.engine import GromacsRunner
from ..gmx.index import _write_ndx
from ..gmx.topology import SystemTopology, parse_system_top
from ..gmx.workflows._util import normalize_gro_molecules_inplace
from ..runtime import resolve_restart
from ..sim.preset.eq import _find_latest_equilibrated_gro
from ..workflow.resume import ResumeManager, StepSpec
from .postprocess import export_interface_group_catalog

Axis = Literal["X", "Y", "Z"]
Route = Literal["route_a", "route_b"]
_AXIS_TO_INDEX = {"X": 0, "Y": 1, "Z": 2}
_INTERFACE_BUILD_SCHEMA_VERSION = "0.8.54-interface-build-v3"
_ROUTE_B_WALL_CLEARANCE_NM = 0.05
_PARAMETER_SECTION_ORDER = [
    "defaults",
    "atomtypes",
    "bondtypes",
    "constrainttypes",
    "angletypes",
    "dihedraltypes",
    "impropertypes",
    "improper_types",
    "pairtypes",
    "nonbond_params",
    "cmaptypes",
]
_PARAMETER_SECTIONS = set(_PARAMETER_SECTION_ORDER)
_MERGE_BANNER_LINES = {
    "; yadonpy merged interface FF parameter blocks",
    "; yadonpy combined FF parameter blocks",
    "; yadonpy generated system.top",
}


@dataclass(frozen=True)
class AreaMismatchPolicy:
    max_lateral_strain: float = 0.03
    prefer_larger_area: bool = True
    max_lateral_replicas_xy: tuple[int, int] = (8, 8)
    reference_side: Literal["larger", "smaller", "bottom", "top"] = "larger"


@dataclass(frozen=True)
class LateralSizingPlan:
    target_lengths_nm: tuple[float, float]
    bottom_replicas_xy: tuple[int, int]
    top_replicas_xy: tuple[int, int]
    bottom_scales_xy: tuple[float, float]
    top_scales_xy: tuple[float, float]


@dataclass(frozen=True)
class BulkSource:
    name: str
    work_dir: Path
    system_top: Path
    system_ndx: Path
    system_meta: Path
    system_dir: Path
    representative_gro: Path
    representative_tpr: Optional[Path]
    representative_xtc: Optional[Path]
    representative_time_ps: Optional[float]
    used_equilibrium_window: bool
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlabBuildSpec:
    axis: Axis = "Z"
    target_thickness_nm: Optional[float] = None
    target_density_g_cm3: Optional[float] = None
    gap_nm: float = 0.60
    vacuum_nm: float = 0.0
    surface_shell_nm: float = 0.80
    core_guard_nm: float = 0.50
    min_replicas_xy: tuple[int, int] = (1, 1)
    prefer_densest_window: bool = True
    lateral_recentering: bool = True


@dataclass(frozen=True)
class InterfaceRouteSpec:
    route: Route
    axis: Axis = "Z"
    area_policy: AreaMismatchPolicy = field(default_factory=AreaMismatchPolicy)
    bottom: SlabBuildSpec = field(default_factory=SlabBuildSpec)
    top: SlabBuildSpec = field(default_factory=SlabBuildSpec)
    top_lateral_shift_fraction: tuple[float, float] = (0.5, 0.5)

    @staticmethod
    def route_a(
        *,
        axis: Axis = "Z",
        gap_nm: float = 0.60,
        bottom_thickness_nm: float = 4.0,
        top_thickness_nm: float = 4.0,
        surface_shell_nm: float = 0.80,
        core_guard_nm: float = 0.50,
        top_lateral_shift_fraction: tuple[float, float] = (0.5, 0.5),
        area_policy: Optional[AreaMismatchPolicy] = None,
    ) -> "InterfaceRouteSpec":
        pol = area_policy or AreaMismatchPolicy()
        return InterfaceRouteSpec(
            route="route_a",
            axis=axis,
            area_policy=pol,
            bottom=SlabBuildSpec(
                axis=axis,
                gap_nm=gap_nm,
                target_thickness_nm=bottom_thickness_nm,
                surface_shell_nm=surface_shell_nm,
                core_guard_nm=core_guard_nm,
            ),
            top=SlabBuildSpec(
                axis=axis,
                gap_nm=gap_nm,
                target_thickness_nm=top_thickness_nm,
                surface_shell_nm=surface_shell_nm,
                core_guard_nm=core_guard_nm,
            ),
            top_lateral_shift_fraction=top_lateral_shift_fraction,
        )

    @staticmethod
    def route_b(
        *,
        axis: Axis = "Z",
        gap_nm: float = 0.60,
        vacuum_nm: float = 8.0,
        bottom_thickness_nm: float = 4.0,
        top_thickness_nm: float = 4.0,
        surface_shell_nm: float = 0.80,
        core_guard_nm: float = 0.50,
        top_lateral_shift_fraction: tuple[float, float] = (0.5, 0.5),
        area_policy: Optional[AreaMismatchPolicy] = None,
    ) -> "InterfaceRouteSpec":
        pol = area_policy or AreaMismatchPolicy()
        return InterfaceRouteSpec(
            route="route_b",
            axis=axis,
            area_policy=pol,
            bottom=SlabBuildSpec(
                axis=axis,
                gap_nm=gap_nm,
                target_thickness_nm=bottom_thickness_nm,
                surface_shell_nm=surface_shell_nm,
                core_guard_nm=core_guard_nm,
            ),
            top=SlabBuildSpec(
                axis=axis,
                gap_nm=gap_nm,
                vacuum_nm=vacuum_nm,
                target_thickness_nm=top_thickness_nm,
                surface_shell_nm=surface_shell_nm,
                core_guard_nm=core_guard_nm,
            ),
            top_lateral_shift_fraction=top_lateral_shift_fraction,
        )


@dataclass(frozen=True)
class FragmentRecord:
    moltype: str
    natoms: int
    coords_nm: np.ndarray
    atom_names: list[str]
    charges: list[float]
    masses: list[float]
    source_instance: int
    bonds: tuple[tuple[int, int], ...] = ()

    def com(self, axis_idx: Optional[int] = None) -> np.ndarray | float:
        xyz = np.asarray(self.coords_nm, dtype=float)
        if xyz.size == 0:
            return 0.0 if axis_idx is not None else np.zeros(3, dtype=float)
        w = np.asarray(self.masses, dtype=float)
        if w.size != xyz.shape[0] or float(w.sum()) <= 0.0:
            c = xyz.mean(axis=0)
        else:
            c = np.average(xyz, axis=0, weights=w)
        if axis_idx is None:
            return c
        return float(c[axis_idx])

    @property
    def net_charge(self) -> float:
        try:
            return float(sum(float(x) for x in self.charges))
        except Exception:
            return 0.0


@dataclass(frozen=True)
class PreparedSlab:
    name: str
    axis: Axis
    source_name: str
    source_representative_gro: Path
    gro_path: Path
    top_path: Path
    ndx_path: Path
    meta_path: Path
    thickness_nm: float
    target_thickness_nm: float
    box_nm: tuple[float, float, float]
    selected_fragments: int
    target_area_nm2: float
    actual_area_nm2: float
    target_density_g_cm3: Optional[float]
    route: Route


@dataclass(frozen=True)
class BuiltInterface:
    name: str
    route: Route
    axis: Axis
    out_dir: Path
    system_gro: Path
    system_top: Path
    system_ndx: Path
    system_meta: Path
    bottom_slab: PreparedSlab
    top_slab: PreparedSlab
    protocol_manifest: Path
    box_nm: tuple[float, float, float]
    notes: tuple[str, ...] = ()


@dataclass
class _GroAtom:
    resnr: int
    resname: str
    atomname: str
    atomnr: int
    xyz_nm: np.ndarray


@dataclass
class _GroFrame:
    title: str
    atoms: list[_GroAtom]
    box_nm: tuple[float, float, float]


def _safe_name(name: str) -> str:
    s = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name).strip())
    s = s.strip("._")
    return s or "interface"


def _group_token(name: str) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in str(name or "").strip())
    s = s.strip("_")
    return s or "UNK"


def _add_group_atoms(group_order: list[str], groups: dict[str, list[int]], name: str, atom_ids: int | Iterable[int]) -> None:
    if isinstance(atom_ids, int):
        seq = [int(atom_ids)]
    else:
        seq = [int(idx) for idx in atom_ids]
    if not seq:
        return
    if name not in groups:
        groups[name] = []
        group_order.append(name)
    groups[name].extend(seq)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _gro_wrap_index(value: int) -> int:
    value = int(value)
    if value < 0:
        return -((-value) % 100000)
    return value % 100000


def _read_gro_frame(path: Path) -> _GroFrame:
    lines = _read_text_lines(path)
    if len(lines) < 3:
        raise ValueError(f"Invalid .gro file: {path}")
    nat = int(lines[1].strip())
    atoms: list[_GroAtom] = []
    for i in range(nat):
        raw = lines[2 + i]
        try:
            xyz = np.array([
                float(raw[20:28]),
                float(raw[28:36]),
                float(raw[36:44]),
            ], dtype=float)
        except Exception:
            if len(raw) < 24:
                raise
            xyz = np.array([
                float(raw[-24:-16]),
                float(raw[-16:-8]),
                float(raw[-8:]),
            ], dtype=float)
        atoms.append(
            _GroAtom(
                resnr=int(raw[0:5].strip() or 0),
                resname=raw[5:10].strip() or "RES",
                atomname=raw[10:15].strip() or f"A{i + 1}",
                atomnr=i + 1,
                xyz_nm=xyz,
            )
        )
    box_line = lines[2 + nat].split()
    if len(box_line) < 3:
        raise ValueError(f"Invalid .gro box line: {path}")
    return _GroFrame(title=lines[0].strip(), atoms=atoms, box_nm=(float(box_line[0]), float(box_line[1]), float(box_line[2])))


def _write_gro_frame(path: Path, title: str, atoms: Iterable[_GroAtom], box_nm: tuple[float, float, float]) -> None:
    arr = list(atoms)
    lines = [title, f"{len(arr):5d}"]
    for i, atom in enumerate(arr, start=1):
        x, y, z = atom.xyz_nm
        lines.append(
            f"{_gro_wrap_index(int(atom.resnr)):5d}{str(atom.resname)[:5]:<5}{str(atom.atomname)[:5]:>5}{_gro_wrap_index(i):5d}{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append(f"{box_nm[0]:10.5f}{box_nm[1]:10.5f}{box_nm[2]:10.5f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wrap_gro_atoms_into_primary_box(
    gro_path: Path,
    *,
    dims: Iterable[int] | None = None,
) -> dict[str, Any]:
    frame = _read_gro_frame(gro_path)
    target_dims = tuple(int(d) for d in (dims if dims is not None else (0, 1, 2)))
    wrapped_atoms: list[_GroAtom] = []
    wrapped_components = 0
    max_shift = 0.0
    for atom in frame.atoms:
        xyz = np.asarray(atom.xyz_nm, dtype=float).copy()
        for dim in target_dims:
            if dim < 0 or dim >= 3:
                continue
            box_len = float(frame.box_nm[dim])
            if box_len <= 0.0:
                continue
            original = float(xyz[dim])
            wrapped = original - box_len * math.floor(original / box_len)
            if abs(wrapped - original) > 1.0e-9:
                wrapped_components += 1
                max_shift = max(max_shift, abs(wrapped - original))
                xyz[dim] = wrapped
        wrapped_atoms.append(
            _GroAtom(
                resnr=atom.resnr,
                resname=atom.resname,
                atomname=atom.atomname,
                atomnr=atom.atomnr,
                xyz_nm=xyz,
            )
        )
    if wrapped_components > 0:
        title = frame.title
        if "yadonpy_boxwrap" not in title:
            title = f"{title[:62]} | yadonpy_boxwrap"
        _write_gro_frame(gro_path, title, wrapped_atoms, frame.box_nm)
    return {
        "applied": bool(wrapped_components > 0),
        "wrapped_components": int(wrapped_components),
        "max_shift_nm": float(max_shift),
        "dims": target_dims,
    }


def _read_include_lines(top_path: Path) -> list[str]:
    out: list[str] = []
    for raw in _read_text_lines(top_path):
        line = raw.strip()
        if line.lower().startswith("#include"):
            out.append(line)
    return out


def _include_target_path(include_line: str, *, base_dir: Path) -> Path | None:
    try:
        rel = include_line.split(None, 1)[1].strip().strip('"')
    except Exception:
        return None
    return (base_dir / rel).resolve()


def _include_kinds(include_line: str, *, base_dir: Path) -> dict[str, bool]:
    inc_path = _include_target_path(include_line, base_dir=base_dir)
    if inc_path is None:
        return {
            "has_parameters": False,
            "has_moleculetype": False,
            "parameters_before_moleculetype": False,
            "parameters_after_moleculetype": False,
        }
    if "ff_parameters" in inc_path.name.lower():
        return {
            "has_parameters": True,
            "has_moleculetype": False,
            "parameters_before_moleculetype": True,
            "parameters_after_moleculetype": False,
        }

    has_parameters = False
    has_moleculetype = False
    parameters_before_moleculetype = False
    parameters_after_moleculetype = False
    saw_moleculetype = False
    try:
        for raw in _read_text_lines(inc_path):
            line = raw.strip().lower()
            if not line or line.startswith(";") or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                sec = line.strip("[]").strip()
                if sec == "moleculetype":
                    has_moleculetype = True
                    saw_moleculetype = True
                elif sec in _PARAMETER_SECTIONS:
                    has_parameters = True
                    if saw_moleculetype:
                        parameters_after_moleculetype = True
                    else:
                        parameters_before_moleculetype = True
    except Exception:
        return {
            "has_parameters": False,
            "has_moleculetype": False,
            "parameters_before_moleculetype": False,
            "parameters_after_moleculetype": False,
        }
    return {
        "has_parameters": has_parameters,
        "has_moleculetype": has_moleculetype,
        "parameters_before_moleculetype": parameters_before_moleculetype,
        "parameters_after_moleculetype": parameters_after_moleculetype,
    }


def _is_parameter_include(include_line: str, *, base_dir: Path) -> bool:
    kinds = _include_kinds(include_line, base_dir=base_dir)
    return bool(kinds["has_parameters"] and not kinds["has_moleculetype"])


def _split_include_parameter_and_molecule_text(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    mt_idx = None
    for i, raw in enumerate(lines):
        stripped = raw.strip().lower()
        if stripped.startswith("[") and stripped.endswith("]") and stripped.strip("[]").strip() == "moleculetype":
            mt_idx = i
            break
    if mt_idx is None:
        return text.strip() + ("\n" if text.strip() else ""), ""
    params_text = "\n".join(lines[:mt_idx]).strip()
    molecule_text = "\n".join(lines[mt_idx:]).strip()
    return (params_text + "\n") if params_text else "", (molecule_text + "\n") if molecule_text else ""


def _parameter_record_key(section: str, line: str) -> tuple[str, ...]:
    cols = line.split()
    if not cols:
        return (section, "")
    widths = {
        "atomtypes": 1,
        "bondtypes": 2,
        "constrainttypes": 2,
        "pairtypes": 2,
        "nonbond_params": 2,
        "angletypes": 3,
        "dihedraltypes": 4,
        "impropertypes": 4,
        "improper_types": 4,
        "cmaptypes": 5,
    }
    return (section, *cols[: widths.get(section, len(cols))])


def _collect_parameter_sections(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in _read_text_lines(path):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            if current is not None:
                sections.setdefault(current, []).append("")
            continue
        if stripped.startswith(";") or stripped.startswith("#"):
            if current is None:
                if line not in _MERGE_BANNER_LINES and line not in preamble:
                    preamble.append(line)
            else:
                sections.setdefault(current, []).append(line)
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            sec = stripped.strip("[]").strip().lower()
            if sec == "moleculetype":
                break
            if sec == "defaults":
                current = None
                continue
            current = sec if sec in _PARAMETER_SECTIONS else None
            if current is not None:
                sections.setdefault(current, [])
            continue
        if current is not None:
            sections.setdefault(current, []).append(line)
    return preamble, sections


def _write_merged_parameter_include(*, dest_path: Path, source_paths: list[Path]) -> bool:
    if not source_paths:
        return False

    preamble: list[str] = []
    merged_sections: dict[str, list[str]] = {}
    seen_records: dict[str, dict[tuple[str, ...], str]] = {}

    for src in source_paths:
        src_preamble, src_sections = _collect_parameter_sections(src)
        for line in src_preamble:
            if line not in preamble:
                preamble.append(line)
        for section, lines in src_sections.items():
            target_lines = merged_sections.setdefault(section, [])
            section_seen = seen_records.setdefault(section, {})
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith(";") or stripped.startswith("#"):
                    if line not in target_lines:
                        target_lines.append(line)
                    continue
                key = _parameter_record_key(section, stripped)
                prev = section_seen.get(key)
                if prev is None:
                    section_seen[key] = stripped
                    target_lines.append(line)
                    continue
                if prev != stripped:
                    raise RuntimeError(
                        f"Conflicting parameter definition while merging interface FF includes: section={section} key={' '.join(key[1:])}"
                    )

    ordered_sections = list(_PARAMETER_SECTION_ORDER)
    for section in sorted(merged_sections.keys()):
        if section not in ordered_sections:
            ordered_sections.append(section)

    out_lines = [
        "; yadonpy merged interface FF parameter blocks",
        "[ defaults ]",
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ",
        "1 2 yes 0.5 0.8333333333",
    ]
    out_lines.extend(preamble)
    out_lines.append("")
    for section in ordered_sections:
        lines = merged_sections.get(section) or []
        if not lines:
            continue
        out_lines.append(f"[ {section} ]")
        out_lines.extend(lines)
        out_lines.append("")
    dest_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    return True


def _iter_relevant_topology_directives(path: Path, *, seen: set[Path] | None = None) -> Iterable[tuple[str, Path]]:
    visited = seen if seen is not None else set()
    resolved = path.resolve()
    if resolved in visited or not resolved.exists():
        return
    visited.add(resolved)

    for raw in _read_text_lines(resolved):
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.lower().startswith("#include"):
            inc = _include_target_path(line, base_dir=resolved.parent)
            if inc is not None:
                yield from _iter_relevant_topology_directives(inc, seen=visited)
            continue
        if line.startswith("#"):
            continue
        if not (line.startswith("[") and line.endswith("]")):
            continue
        section = line.strip("[]").strip().lower()
        if section in _PARAMETER_SECTIONS or section in {"moleculetype", "system", "molecules"}:
            yield section, resolved


def validate_topology_include_order(top_path: Path) -> list[str]:
    issues: list[str] = []
    stage_rank = {
        "defaults": 0,
        "atomtypes": 1,
        "bondtypes": 1,
        "constrainttypes": 1,
        "angletypes": 1,
        "dihedraltypes": 1,
        "impropertypes": 1,
        "improper_types": 1,
        "pairtypes": 1,
        "nonbond_params": 1,
        "cmaptypes": 1,
        "moleculetype": 2,
        "system": 3,
        "molecules": 4,
    }
    directives = list(_iter_relevant_topology_directives(top_path))
    highest_stage = -1
    for idx, (section, src_path) in enumerate(directives):
        rank = stage_rank.get(section)
        if rank is None:
            continue
        if section == "defaults" and idx != 0:
            prev_section, prev_src = directives[idx - 1]
            issues.append(
                f"[ defaults ] must be the first directive, but appears after [{prev_section}] in {src_path.name} (previous directive from {prev_src.name})"
            )
        if rank < highest_stage:
            if section in _PARAMETER_SECTIONS and highest_stage >= 2:
                issues.append(f"parameter section [{section}] appears after [ moleculetype ] in {src_path.name}")
            elif section == "moleculetype" and highest_stage >= 3:
                issues.append(f"[ moleculetype ] appears after top-level system directives in {src_path.name}")
            elif section == "system" and highest_stage >= 4:
                issues.append(f"[ system ] appears after [ molecules ] in {src_path.name}")
            else:
                issues.append(f"directive order regressed at [{section}] in {src_path.name}")
        highest_stage = max(highest_stage, rank)
    return issues


def _ordered_include_lines(include_lines: list[str], *, base_dir: Path) -> list[str]:
    param_lines: list[str] = []
    molecule_lines: list[str] = []
    for line in include_lines:
        if _is_parameter_include(line, base_dir=base_dir):
            param_lines.append(line)
        else:
            molecule_lines.append(line)
    return param_lines + molecule_lines


def _copy_include_file(*, src: Path, dest_dir: Path, source_tag: str, copied_names: dict[tuple[str, str], str], content_by_name: dict[str, bytes], collision_mode: str, data: bytes | None = None, name: str | None = None, key_suffix: str = "raw") -> str:
    key = (str(source_tag), f"{src.resolve()}::{key_suffix}")
    if key in copied_names:
        return copied_names[key]

    payload = data if data is not None else src.read_bytes()
    out_name = name or src.name
    if out_name in content_by_name and content_by_name[out_name] != payload:
        if collision_mode == "error":
            raise RuntimeError(f"Conflicting include file name during interface assembly: {src.name}")
        stem = Path(src.name).stem
        suffix = ''.join(Path(src.name).suffixes)
        tag = _safe_name(source_tag)
        out_name = f"{tag}_{stem}{suffix}"
        idx = 2
        while out_name in content_by_name and content_by_name[out_name] != payload:
            out_name = f"{tag}_{stem}_{idx}{suffix}"
            idx += 1

    dst = dest_dir / out_name
    if not dst.exists():
        if data is None:
            shutil.copy2(src, dst)
        else:
            dst.write_bytes(payload)
    content_by_name[out_name] = payload
    copied_names[key] = out_name
    return out_name


def _collect_normalized_include_lines(*, top_path: Path, dest_dir: Path, source_tag: str, seen: set[str], copied_names: dict[tuple[str, str], str], content_by_name: dict[str, bytes], collision_mode: str, parameter_paths: list[Path]) -> list[str]:
    include_lines: list[str] = []
    for raw in _read_include_lines(top_path):
        rel = raw.split(None, 1)[1].strip().strip('"')
        src = (top_path.parent / rel).resolve()
        if not src.exists():
            continue
        kinds = _include_kinds(raw, base_dir=top_path.parent)
        if kinds["parameters_after_moleculetype"]:
            raise RuntimeError(f"Invalid GROMACS include: parameter blocks appear after [ moleculetype ] in {src}")
        if kinds["has_parameters"] and src not in parameter_paths:
            parameter_paths.append(src)
        if not kinds["has_moleculetype"]:
            continue
        if kinds["parameters_before_moleculetype"]:
            _, molecule_text = _split_include_parameter_and_molecule_text(src)
            if not molecule_text.strip():
                continue
            copied_name = _copy_include_file(
                src=src,
                dest_dir=dest_dir,
                source_tag=source_tag,
                copied_names=copied_names,
                content_by_name=content_by_name,
                collision_mode=collision_mode,
                data=molecule_text.encode("utf-8"),
                name=src.name,
                key_suffix="molecule-only",
            )
        else:
            copied_name = _copy_include_file(
                src=src,
                dest_dir=dest_dir,
                source_tag=source_tag,
                copied_names=copied_names,
                content_by_name=content_by_name,
                collision_mode=collision_mode,
            )
        inc_line = f'#include "molecules/{copied_name}"'
        if inc_line not in seen:
            include_lines.append(inc_line)
            seen.add(inc_line)
    return include_lines


def _copy_include_tree(top_path: Path, dest_dir: Path) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    copied_names: dict[tuple[str, str], str] = {}
    content_by_name: dict[str, bytes] = {}
    parameter_paths: list[Path] = []
    source_tag = top_path.stem or "system"
    include_lines = _collect_normalized_include_lines(
        top_path=top_path,
        dest_dir=dest_dir,
        source_tag=source_tag,
        seen=seen,
        copied_names=copied_names,
        content_by_name=content_by_name,
        collision_mode="error",
        parameter_paths=parameter_paths,
    )
    if _write_merged_parameter_include(dest_path=dest_dir / "ff_parameters.itp", source_paths=parameter_paths):
        include_lines.insert(0, '#include "molecules/ff_parameters.itp"')
    return include_lines


def _latest_system_dir(root: Path) -> Path:
    if (root / "02_system" / "system.top").exists():
        return root / "02_system"
    if (root / "system.top").exists():
        return root
    raise FileNotFoundError(f"Cannot locate system.top under {root}")


def _load_equilibrium_window_ps(work_dir: Path) -> Optional[tuple[float, float]]:
    eq_json = Path(work_dir) / "06_analysis" / "equilibrium.json"
    if not eq_json.exists():
        return None
    try:
        payload = json.loads(eq_json.read_text(encoding="utf-8"))
        dg = payload.get("density_gate") or {}
        start = dg.get("window_start_time_ps")
        end = payload.get("trajectory_time_ps")
        if start is None or end is None:
            return None
        return float(start), float(end)
    except Exception:
        return None


def _extract_snapshot(*, name: str, bulk_root: Path, out_dir: Path, restart: Optional[bool] = None, selection_group: str = "System") -> BulkSource:
    rst_flag = resolve_restart(restart)
    bulk_root = Path(bulk_root).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = ResumeManager(out_dir, enabled=bool(rst_flag), strict_inputs=True)
    system_dir = _latest_system_dir(bulk_root)
    system_top = system_dir / "system.top"
    system_ndx = system_dir / "system.ndx"
    system_meta = system_dir / "system_meta.json"
    latest_gro = _find_latest_equilibrated_gro(bulk_root)
    if latest_gro is None:
        latest_gro = system_dir / "system.gro"
    if not latest_gro.exists():
        raise FileNotFoundError(f"Cannot find an equilibrated gro for {bulk_root}")
    latest_dir = latest_gro.parent
    tpr = latest_dir / "md.tpr"
    xtc = latest_dir / "md.xtc"
    rep_raw = out_dir / "representative_raw.gro"
    rep_whole = out_dir / "representative_whole.gro"
    manifest = out_dir / "snapshot_manifest.json"
    window = _load_equilibrium_window_ps(bulk_root)
    representative_time_ps: Optional[float] = None
    used_equilibrium = False
    notes: list[str] = []

    spec = StepSpec(
        name=f"extract_snapshot_{_safe_name(name)}",
        outputs=[rep_raw, rep_whole, manifest],
        inputs={"latest_gro": str(latest_gro), "tpr": str(tpr), "xtc": str(xtc), "window": window, "selection_group": selection_group},
        description="Extract representative bulk snapshot for interface preparation",
    )

    def _run() -> None:
        nonlocal representative_time_ps, used_equilibrium, notes
        if window is not None and tpr.exists() and xtc.exists():
            representative_time_ps = float(window[0] + (window[1] - window[0]) / 2.0)
            used_equilibrium = True
            runner = GromacsRunner(verbose=False)
            try:
                runner.run(["trjconv", "-s", str(tpr), "-f", str(xtc), "-o", str(rep_raw), "-dump", f"{representative_time_ps:.3f}"], cwd=out_dir, stdin_text=f"{selection_group}\n", check=True, capture=True)
                runner.run(["trjconv", "-s", str(tpr), "-f", str(rep_raw), "-o", str(rep_whole), "-pbc", "mol"], cwd=out_dir, stdin_text=f"{selection_group}\n", check=True, capture=True)
            except Exception as exc:
                notes.append(f"equilibrium-window snapshot extraction failed: {exc}")
                used_equilibrium = False
        if (not used_equilibrium) or (not rep_whole.exists()):
            representative_time_ps = None
            shutil.copy2(latest_gro, rep_raw)
            shutil.copy2(latest_gro, rep_whole)
            notes.append("fell back to latest available structure because trajectory-based midpoint extraction was unavailable")
        whole_fix = normalize_gro_molecules_inplace(top=system_top, gro=rep_whole)
        if whole_fix.get("applied"):
            notes.append(
                f"topology-guided whole-molecule canonicalization applied to representative snapshot; normalized_molecules={int(whole_fix.get('normalized_molecules', 0))}"
            )
        elif whole_fix.get("error"):
            notes.append(f"representative snapshot canonicalization skipped: {whole_fix['error']}")
        _write_json(manifest, {"name": name, "bulk_root": bulk_root, "system_dir": system_dir, "latest_gro": latest_gro, "representative_raw_gro": rep_raw, "representative_whole_gro": rep_whole, "representative_time_ps": representative_time_ps, "used_equilibrium_window": bool(used_equilibrium), "notes": notes})

    resume.run(spec, _run)
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        representative_time_ps = payload.get("representative_time_ps")
        used_equilibrium = bool(payload.get("used_equilibrium_window", False))
        notes = list(payload.get("notes") or [])
    return BulkSource(
        name=name,
        work_dir=bulk_root,
        system_top=system_top,
        system_ndx=system_ndx,
        system_meta=system_meta,
        system_dir=system_dir,
        representative_gro=rep_whole,
        representative_tpr=tpr if tpr.exists() else None,
        representative_xtc=xtc if xtc.exists() else None,
        representative_time_ps=float(representative_time_ps) if representative_time_ps is not None else None,
        used_equilibrium_window=bool(used_equilibrium),
        notes=tuple(notes),
    )


def _split_fragments_from_frame(frame: _GroFrame, topo: SystemTopology) -> list[FragmentRecord]:
    out: list[FragmentRecord] = []
    cursor = 0
    instance = 0
    atoms = frame.atoms
    for moltype_name, count in topo.molecules:
        mt = topo.moleculetypes.get(moltype_name)
        if mt is None:
            raise KeyError(f"Molecule type '{moltype_name}' not found in topology")
        nat = mt.natoms
        for _ in range(int(count)):
            block = atoms[cursor: cursor + nat]
            if len(block) != nat:
                raise ValueError(f"Atom count mismatch while parsing {moltype_name}: expected {nat} atoms, got {len(block)}")
            coords = _unwrap_fragment_coords(np.vstack([a.xyz_nm for a in block]), bonds=tuple(mt.bonds), box_nm=frame.box_nm)
            atom_names = [a.atomname for a in block]
            out.append(FragmentRecord(moltype=moltype_name, natoms=nat, coords_nm=coords, atom_names=atom_names, charges=list(mt.charges), masses=list(mt.masses), source_instance=instance, bonds=tuple(mt.bonds)))
            cursor += nat
            instance += 1
    if cursor != len(atoms):
        raise ValueError(f"Unparsed atoms remain in representative frame: parsed {cursor}, total {len(atoms)}")
    return out


def _minimal_image_delta(delta: float, box_len: float) -> float:
    if box_len <= 0.0:
        return float(delta)
    return float(delta - box_len * round(float(delta) / float(box_len)))


def _unwrap_fragment_coords(coords_nm: np.ndarray, *, bonds: tuple[tuple[int, int], ...], box_nm: tuple[float, float, float]) -> np.ndarray:
    coords = np.asarray(coords_nm, dtype=float)
    if coords.ndim != 2 or coords.shape[0] <= 1:
        return coords.copy()
    if not bonds:
        return coords.copy()

    out = coords.copy()
    adjacency: list[list[int]] = [[] for _ in range(coords.shape[0])]
    for ai, aj in bonds:
        i = int(ai) - 1
        j = int(aj) - 1
        if i < 0 or j < 0 or i >= coords.shape[0] or j >= coords.shape[0]:
            continue
        adjacency[i].append(j)
        adjacency[j].append(i)

    seen = [False] * coords.shape[0]
    for root in range(coords.shape[0]):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            idx = stack.pop()
            for nxt in adjacency[idx]:
                if seen[nxt]:
                    continue
                delta = coords[nxt] - coords[idx]
                for dim, box_len in enumerate(box_nm):
                    delta[dim] = _minimal_image_delta(float(delta[dim]), float(box_len))
                out[nxt] = out[idx] + delta
                seen[nxt] = True
                stack.append(nxt)
    return out


def _reference_lateral_length(bottom_length: float, top_length: float, *, policy: AreaMismatchPolicy) -> float:
    mode = str(getattr(policy, "reference_side", "larger")).strip().lower()
    if mode == "bottom":
        return float(bottom_length)
    if mode == "top":
        return float(top_length)
    if mode == "smaller":
        return min(float(bottom_length), float(top_length))
    if mode == "larger":
        return max(float(bottom_length), float(top_length))
    return max(float(bottom_length), float(top_length)) if policy.prefer_larger_area else min(float(bottom_length), float(top_length))


def _target_area_lengths_nm(bottom_box_nm: tuple[float, float, float], top_box_nm: tuple[float, float, float], *, axis: Axis, policy: AreaMismatchPolicy) -> tuple[float, float]:
    ai = _AXIS_TO_INDEX[str(axis)]
    dims = [0, 1, 2]
    lateral = [d for d in dims if d != ai]
    return (
        _reference_lateral_length(float(bottom_box_nm[lateral[0]]), float(top_box_nm[lateral[0]]), policy=policy),
        _reference_lateral_length(float(bottom_box_nm[lateral[1]]), float(top_box_nm[lateral[1]]), policy=policy),
    )


def _resolve_lateral_length_match(
    bottom_length: float,
    top_length: float,
    *,
    policy: AreaMismatchPolicy,
    bottom_min_replica: int,
    top_min_replica: int,
    bottom_max_replica: int,
    top_max_replica: int,
    required_length: float = 0.0,
) -> tuple[float, int, int, float, float]:
    strain = float(max(policy.max_lateral_strain, 0.0))
    max_bottom = max(int(bottom_min_replica), int(bottom_max_replica), 1)
    max_top = max(int(top_min_replica), int(top_max_replica), 1)
    preferred_base = _reference_lateral_length(float(bottom_length), float(top_length), policy=policy)
    best: tuple[tuple[float, int, float, float], tuple[float, int, int, float, float]] | None = None

    for bottom_replica in range(max(int(bottom_min_replica), 1), max_bottom + 1):
        bottom_replicated = float(bottom_length) * float(bottom_replica)
        bottom_low = max(float(required_length), bottom_replicated * (1.0 - strain))
        bottom_high = bottom_replicated * (1.0 + strain)
        for top_replica in range(max(int(top_min_replica), 1), max_top + 1):
            top_replicated = float(top_length) * float(top_replica)
            low = max(bottom_low, float(required_length), top_replicated * (1.0 - strain))
            high = min(bottom_high, top_replicated * (1.0 + strain))
            if low > high + 1.0e-9:
                continue
            preferred = max(preferred_base, float(required_length))
            target = min(max(preferred, low), high)
            bottom_scale = target / max(bottom_replicated, 1.0e-12)
            top_scale = target / max(top_replicated, 1.0e-12)
            objective = (
                float(target),
                int(bottom_replica + top_replica),
                float(max(abs(bottom_scale - 1.0), abs(top_scale - 1.0))),
                float(abs(abs(bottom_scale - 1.0) - abs(top_scale - 1.0))),
            )
            payload = (float(target), int(bottom_replica), int(top_replica), float(bottom_scale), float(top_scale))
            if best is None or objective < best[0]:
                best = (objective, payload)

    if best is not None:
        return best[1]

    target = max(float(required_length), preferred_base)
    bottom_replica = max(int(bottom_min_replica), int(math.ceil(target / max(float(bottom_length) * (1.0 + strain), 1.0e-12))))
    top_replica = max(int(top_min_replica), int(math.ceil(target / max(float(top_length) * (1.0 + strain), 1.0e-12))))
    bottom_replicated = float(bottom_length) * float(bottom_replica)
    top_replicated = float(top_length) * float(top_replica)
    bottom_scale = target / max(bottom_replicated, 1.0e-12)
    top_scale = target / max(top_replicated, 1.0e-12)
    return float(target), int(bottom_replica), int(top_replica), float(bottom_scale), float(top_scale)


def _resolve_lateral_sizing_plan(
    *,
    bottom_box_nm: tuple[float, float, float],
    top_box_nm: tuple[float, float, float],
    axis: Axis,
    policy: AreaMismatchPolicy,
    bottom_min_replicas_xy: tuple[int, int],
    top_min_replicas_xy: tuple[int, int],
) -> LateralSizingPlan:
    ai = _AXIS_TO_INDEX[str(axis)]
    lateral = [d for d in [0, 1, 2] if d != ai]
    x_target, bottom_rx, top_rx, bottom_sx, top_sx = _resolve_lateral_length_match(
        float(bottom_box_nm[lateral[0]]),
        float(top_box_nm[lateral[0]]),
        policy=policy,
        bottom_min_replica=int(bottom_min_replicas_xy[0]),
        top_min_replica=int(top_min_replicas_xy[0]),
        bottom_max_replica=int(policy.max_lateral_replicas_xy[0]),
        top_max_replica=int(policy.max_lateral_replicas_xy[0]),
    )
    y_target, bottom_ry, top_ry, bottom_sy, top_sy = _resolve_lateral_length_match(
        float(bottom_box_nm[lateral[1]]),
        float(top_box_nm[lateral[1]]),
        policy=policy,
        bottom_min_replica=int(bottom_min_replicas_xy[1]),
        top_min_replica=int(top_min_replicas_xy[1]),
        bottom_max_replica=int(policy.max_lateral_replicas_xy[1]),
        top_max_replica=int(policy.max_lateral_replicas_xy[1]),
    )
    return LateralSizingPlan(
        target_lengths_nm=(float(x_target), float(y_target)),
        bottom_replicas_xy=(int(bottom_rx), int(bottom_ry)),
        top_replicas_xy=(int(top_rx), int(top_ry)),
        bottom_scales_xy=(float(bottom_sx), float(bottom_sy)),
        top_scales_xy=(float(top_sx), float(top_sy)),
    )


def _fragment_mass(frag: FragmentRecord) -> float:
    try:
        masses = np.asarray(frag.masses, dtype=float)
        if masses.size == frag.natoms and float(masses.sum()) > 0.0:
            return float(masses.sum())
    except Exception:
        pass
    return float(max(frag.natoms, 1))


def _format_elapsed_seconds(seconds: float) -> str:
    return f"{float(max(seconds, 0.0)):.3f} s"


@jit(nopython=True, cache=True)
def _searchsorted_left_numba(values: np.ndarray, target: float) -> int:
    left = 0
    right = int(values.shape[0])
    while left < right:
        mid = (left + right) // 2
        if values[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


@jit(nopython=True, cache=True)
def _searchsorted_right_numba(values: np.ndarray, target: float) -> int:
    left = 0
    right = int(values.shape[0])
    while left < right:
        mid = (left + right) // 2
        if values[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left


@jit(nopython=True, cache=True)
def _find_best_dense_window_numba(candidate_starts: np.ndarray, box_len: float, thickness: float, center: float, sorted_coms: np.ndarray, prefix_mass: np.ndarray) -> tuple[float, float]:
    best_start = 0.0
    best_stop = min(thickness, box_len)
    best_mass = -1.0
    best_secondary = -1.0e300
    for idx in range(candidate_starts.shape[0]):
        start = candidate_starts[idx]
        stop = min(start + thickness, box_len)
        start = max(0.0, stop - thickness)
        left = _searchsorted_left_numba(sorted_coms, start)
        right = _searchsorted_right_numba(sorted_coms, stop)
        mass = prefix_mass[right] - prefix_mass[left]
        count = float(right - left)
        secondary = -abs((start + stop) * 0.5 - center) - 1.0e-6 * count
        if mass > best_mass or (mass == best_mass and secondary > best_secondary):
            best_mass = mass
            best_secondary = secondary
            best_start = start
            best_stop = stop
    return best_start, best_stop


@jit(nopython=True, cache=True)
def _pick_charge_rebalance_candidate_numba(charges: np.ndarray, coms: np.ndarray, used_mask: np.ndarray, total_charge: float, zmin: float, zmax: float) -> int:
    best_idx = -1
    best_abs_total = 1.0e300
    best_distance = 1.0e300
    best_improvement = -1.0e300
    best_abs_charge = -1.0e300
    for idx in range(charges.shape[0]):
        if used_mask[idx]:
            continue
        frag_charge = charges[idx]
        if abs(frag_charge) <= 1.0e-8:
            continue
        if total_charge * frag_charge >= 0.0:
            continue
        new_total = total_charge + frag_charge
        improvement = abs(total_charge) - abs(new_total)
        if improvement <= 1.0e-8:
            continue
        com = coms[idx]
        distance = 0.0
        if com < zmin:
            distance = zmin - com
        elif com > zmax:
            distance = com - zmax
        abs_total = abs(new_total)
        abs_charge = abs(frag_charge)
        if (
            abs_total < best_abs_total
            or (abs_total == best_abs_total and distance < best_distance)
            or (abs_total == best_abs_total and distance == best_distance and improvement > best_improvement)
            or (abs_total == best_abs_total and distance == best_distance and improvement == best_improvement and abs_charge > best_abs_charge)
        ):
            best_idx = idx
            best_abs_total = abs_total
            best_distance = distance
            best_improvement = improvement
            best_abs_charge = abs_charge
    return best_idx


def _fragment_distance_to_window(frag: FragmentRecord, *, axis: Axis, zmin: float, zmax: float) -> float:
    ai = _AXIS_TO_INDEX[str(axis)]
    com = float(frag.com(ai))
    if zmin <= com <= zmax:
        return 0.0
    return float(min(abs(com - zmin), abs(com - zmax)))


def _rebalance_fragment_selection_for_charge(
    selected: list[FragmentRecord],
    *,
    all_fragments: list[FragmentRecord],
    axis: Axis,
    charge_tol: float = 0.5,
    max_extra_fragments: int = 256,
    charge_scale: float = 1.0,
) -> tuple[list[FragmentRecord], float, list[str]]:
    if not selected:
        return selected, 0.0, []

    charge_scale = float(max(charge_scale, 1.0))
    selected_ids = {id(frag) for frag in selected}
    charges = np.asarray([float(frag.net_charge) for frag in all_fragments], dtype=float) * charge_scale
    if charges.size == 0:
        return selected, 0.0, []
    coms = np.asarray([float(frag.com(_AXIS_TO_INDEX[str(axis)])) for frag in all_fragments], dtype=float)
    ai = _AXIS_TO_INDEX[str(axis)]
    frag_mins = np.asarray([float(np.min(frag.coords_nm[:, ai])) for frag in all_fragments], dtype=float)
    frag_maxs = np.asarray([float(np.max(frag.coords_nm[:, ai])) for frag in all_fragments], dtype=float)
    used_mask = np.asarray([id(frag) in selected_ids for frag in all_fragments], dtype=bool)
    total_charge = float(charges[used_mask].sum())
    if abs(total_charge) <= float(charge_tol):
        return selected, total_charge, []

    selected_indices = [idx for idx, used in enumerate(used_mask) if bool(used)]
    zmin = float(np.min(frag_mins[selected_indices]))
    zmax = float(np.max(frag_maxs[selected_indices]))
    added = 0

    while abs(total_charge) > float(charge_tol) and added < int(max_extra_fragments):
        if _HAS_NUMBA:
            best_idx = int(_pick_charge_rebalance_candidate_numba(charges, coms, used_mask, float(total_charge), float(zmin), float(zmax)))
        else:
            candidate_mask = ~used_mask
            candidate_mask &= np.abs(charges) > 1.0e-8
            candidate_mask &= (total_charge * charges) < 0.0
            if not np.any(candidate_mask):
                break

            new_total = total_charge + charges
            improvement = abs(total_charge) - np.abs(new_total)
            candidate_mask &= improvement > 1.0e-8
            if not np.any(candidate_mask):
                break

            distances = np.where(coms < zmin, zmin - coms, np.where(coms > zmax, coms - zmax, 0.0))
            candidate_idx = np.flatnonzero(candidate_mask)
            order = np.lexsort(
                (
                    -np.abs(charges[candidate_idx]),
                    -improvement[candidate_idx],
                    distances[candidate_idx],
                    np.abs(new_total[candidate_idx]),
                )
            )
            best_idx = int(candidate_idx[int(order[0])])
        if best_idx < 0:
            break

        used_mask[best_idx] = True
        frag = all_fragments[best_idx]
        selected.append(frag)
        total_charge += float(charges[best_idx])
        zmin = min(zmin, float(frag_mins[best_idx]))
        zmax = max(zmax, float(frag_maxs[best_idx]))
        added += 1

    notes: list[str] = []
    if added > 0:
        notes.append(
            f"charge-balanced slab selection by adding {added} nearby counter-fragments; effective_net_charge_e={total_charge:.6f}"
        )
    elif abs(total_charge) > float(charge_tol):
        notes.append(f"slab selection remained charge-imbalanced; effective_net_charge_e={total_charge:.6f}")
    return selected, float(total_charge), notes


def _replicate_xy(fragments: list[FragmentRecord], *, box_nm: tuple[float, float, float], axis: Axis, replicas_xy: tuple[int, int] = (1, 1)) -> tuple[list[FragmentRecord], tuple[float, float, float]]:
    ai = _AXIS_TO_INDEX[str(axis)]
    lateral = [d for d in [0, 1, 2] if d != ai]
    lengths = list(map(float, box_nm))
    rx = max(int(replicas_xy[0]), 1)
    ry = max(int(replicas_xy[1]), 1)
    if rx <= 1 and ry <= 1:
        return fragments, tuple(lengths)
    out: list[FragmentRecord] = []
    for ix in range(rx):
        for iy in range(ry):
            shift = np.zeros(3, dtype=float)
            shift[lateral[0]] = ix * lengths[lateral[0]]
            shift[lateral[1]] = iy * lengths[lateral[1]]
            for frag in fragments:
                out.append(FragmentRecord(moltype=frag.moltype, natoms=frag.natoms, coords_nm=np.asarray(frag.coords_nm, dtype=float) + shift, atom_names=list(frag.atom_names), charges=list(frag.charges), masses=list(frag.masses), source_instance=frag.source_instance, bonds=tuple(frag.bonds)))
    lengths[lateral[0]] *= rx
    lengths[lateral[1]] *= ry
    return out, tuple(lengths)


def _fit_lateral_lengths(fragments: list[FragmentRecord], *, box_nm: tuple[float, float, float], target_lengths_nm: tuple[float, float], axis: Axis, policy: AreaMismatchPolicy) -> tuple[list[FragmentRecord], tuple[float, float, float]]:
    ai = _AXIS_TO_INDEX[str(axis)]
    lateral = [d for d in [0, 1, 2] if d != ai]
    lengths = list(map(float, box_nm))
    scales = [target_lengths_nm[0] / max(lengths[lateral[0]], 1.0e-9), target_lengths_nm[1] / max(lengths[lateral[1]], 1.0e-9)]
    if any(abs(s - 1.0) > float(policy.max_lateral_strain) for s in scales):
        return fragments, tuple(lengths)
    out: list[FragmentRecord] = []
    for frag in fragments:
        coords = np.asarray(frag.coords_nm, dtype=float).copy()
        coords[:, lateral[0]] *= scales[0]
        coords[:, lateral[1]] *= scales[1]
        out.append(FragmentRecord(moltype=frag.moltype, natoms=frag.natoms, coords_nm=coords, atom_names=list(frag.atom_names), charges=list(frag.charges), masses=list(frag.masses), source_instance=frag.source_instance, bonds=tuple(frag.bonds)))
    lengths[lateral[0]] = target_lengths_nm[0]
    lengths[lateral[1]] = target_lengths_nm[1]
    return out, tuple(lengths)


def _select_fragments_for_slab(fragments: list[FragmentRecord], *, axis: Axis, box_nm: tuple[float, float, float], target_thickness_nm: float, prefer_densest_window: bool = True) -> tuple[list[FragmentRecord], float, float]:
    if not fragments:
        raise RuntimeError("Representative bulk snapshot did not yield any molecular fragments for slab preparation.")
    ai = _AXIS_TO_INDEX[str(axis)]
    box_len = float(box_nm[ai])
    center = 0.5 * box_len
    selected: list[FragmentRecord] = []
    half = 0.5 * float(target_thickness_nm)
    window_start = max(0.0, center - half)
    window_stop = min(box_len, center + half)
    coms = np.asarray([float(frag.com(ai)) for frag in fragments], dtype=float) if fragments else np.empty(0, dtype=float)
    frag_mins = np.asarray([float(np.min(frag.coords_nm[:, ai])) for frag in fragments], dtype=float) if fragments else np.empty(0, dtype=float)
    frag_maxs = np.asarray([float(np.max(frag.coords_nm[:, ai])) for frag in fragments], dtype=float) if fragments else np.empty(0, dtype=float)
    frag_masses = np.asarray([_fragment_mass(frag) for frag in fragments], dtype=float) if fragments else np.empty(0, dtype=float)
    if prefer_densest_window and fragments and float(target_thickness_nm) < box_len:
        max_start = max(box_len - float(target_thickness_nm), 0.0)
        candidate_starts = np.concatenate(
            [
                np.asarray([window_start], dtype=float),
                np.clip(coms - half, 0.0, max_start),
                np.clip(frag_mins, 0.0, max_start),
                np.clip(frag_maxs - float(target_thickness_nm), 0.0, max_start),
            ]
        )
        candidate_starts = np.unique(candidate_starts)
        sorted_idx = np.argsort(coms, kind="mergesort")
        sorted_coms = coms[sorted_idx]
        sorted_masses = frag_masses[sorted_idx]
        prefix_mass = np.concatenate(([0.0], np.cumsum(sorted_masses)))
        if _HAS_NUMBA:
            window_start, window_stop = _find_best_dense_window_numba(candidate_starts, float(box_len), float(target_thickness_nm), float(center), sorted_coms, prefix_mass)
        else:
            prefix_count = np.arange(sorted_coms.size + 1, dtype=float)
            best_score: tuple[float, float] | None = None
            best_window = (window_start, window_stop)
            for start in candidate_starts:
                stop = min(start + float(target_thickness_nm), box_len)
                start = max(0.0, stop - float(target_thickness_nm))
                left = int(np.searchsorted(sorted_coms, start, side="left"))
                right = int(np.searchsorted(sorted_coms, stop, side="right"))
                mass = float(prefix_mass[right] - prefix_mass[left])
                count = int(prefix_count[right] - prefix_count[left])
                score = (mass, -abs((start + stop) * 0.5 - center) - 1.0e-6 * float(count))
                if best_score is None or score > best_score:
                    best_score = score
                    best_window = (start, stop)
            window_start, window_stop = best_window
    selected_idx = np.flatnonzero((coms >= float(window_start)) & (coms <= float(window_stop))) if fragments else np.empty(0, dtype=int)
    if selected_idx.size:
        selected = [fragments[int(idx)] for idx in selected_idx]
    else:
        nearest = int(np.argmin(np.abs(coms - center)))
        selected_idx = np.asarray([nearest], dtype=int)
        selected = [fragments[nearest]]
    zmins = frag_mins[selected_idx]
    zmaxs = frag_maxs[selected_idx]
    return selected, float(np.min(zmins)), float(np.max(zmaxs))


def _wrap_fragments_lateral(fragments: list[FragmentRecord], *, box_nm: tuple[float, float, float], axis: Axis) -> list[FragmentRecord]:
    ai = _AXIS_TO_INDEX[str(axis)]
    lateral = [d for d in [0, 1, 2] if d != ai]
    out: list[FragmentRecord] = []
    for frag in fragments:
        coords = np.asarray(frag.coords_nm, dtype=float).copy()
        com = np.asarray(frag.com(), dtype=float)
        for li in lateral:
            length = float(box_nm[li])
            if length <= 0.0:
                continue
            while com[li] < 0.0:
                coords[:, li] += length
                com[li] += length
            while com[li] >= length:
                coords[:, li] -= length
                com[li] -= length
        out.append(FragmentRecord(moltype=frag.moltype, natoms=frag.natoms, coords_nm=coords, atom_names=list(frag.atom_names), charges=list(frag.charges), masses=list(frag.masses), source_instance=frag.source_instance, bonds=tuple(frag.bonds)))
    return out


def _shift_coords_into_box_by_com(coords_nm: np.ndarray, *, box_nm: tuple[float, float, float], dims: Iterable[int]) -> np.ndarray:
    coords = np.asarray(coords_nm, dtype=float).copy()
    if coords.ndim != 2 or coords.shape[0] == 0:
        return coords
    com = coords.mean(axis=0)
    for dim in dims:
        box_len = float(box_nm[int(dim)])
        if box_len <= 0.0:
            continue
        while float(com[int(dim)]) < 0.0:
            coords[:, int(dim)] += box_len
            com[int(dim)] += box_len
        while float(com[int(dim)]) >= box_len:
            coords[:, int(dim)] -= box_len
            com[int(dim)] -= box_len
    return coords


def _validate_assembled_interface_geometry(
    *,
    gro_path: Path,
    box_nm: tuple[float, float, float],
    axis: Axis,
    bottom_atom_ids: Iterable[int],
    top_atom_ids: Iterable[int],
    assembled_gap_nm: float | None = None,
    tolerance_nm: float = 1.0e-3,
) -> dict[str, Any]:
    frame = _read_gro_frame(gro_path)
    ai = _AXIS_TO_INDEX[str(axis)]
    tol = float(max(tolerance_nm, 0.0))
    outside = 0
    max_overflow = 0.0
    for atom in frame.atoms:
        for dim, box_len in enumerate(box_nm):
            val = float(atom.xyz_nm[dim])
            limit = float(box_len)
            if val < -tol:
                outside += 1
                max_overflow = max(max_overflow, -val)
            elif val > limit + tol:
                outside += 1
                max_overflow = max(max_overflow, val - limit)

    return {
        "atoms_outside_primary_box": int(outside),
        "max_primary_box_overflow_nm": float(max_overflow),
        "assembled_gap_nm": assembled_gap_nm,
        "tolerance_nm": tol,
    }


def _recenter_fragments_lateral(fragments: list[FragmentRecord], *, box_nm: tuple[float, float, float], axis: Axis) -> list[FragmentRecord]:
    if not fragments:
        return fragments
    ai = _AXIS_TO_INDEX[str(axis)]
    lateral = [d for d in [0, 1, 2] if d != ai]
    total_mass = 0.0
    weighted = np.zeros(3, dtype=float)
    for frag in fragments:
        mass = _fragment_mass(frag)
        weighted += np.asarray(frag.com(), dtype=float) * mass
        total_mass += mass
    if total_mass <= 0.0:
        return fragments
    com = weighted / total_mass
    shift = np.zeros(3, dtype=float)
    for li in lateral:
        shift[li] = 0.5 * float(box_nm[li]) - float(com[li])
    shifted: list[FragmentRecord] = []
    for frag in fragments:
        shifted.append(FragmentRecord(moltype=frag.moltype, natoms=frag.natoms, coords_nm=np.asarray(frag.coords_nm, dtype=float) + shift, atom_names=list(frag.atom_names), charges=list(frag.charges), masses=list(frag.masses), source_instance=frag.source_instance, bonds=tuple(frag.bonds)))
    return _wrap_fragments_lateral(shifted, box_nm=box_nm, axis=axis)


def _build_slab_groups(
    fragments: list[FragmentRecord],
    *,
    start_atom: int,
    axis: Axis,
    thickness_nm: float,
    surface_shell_nm: float,
    core_guard_nm: float,
) -> tuple[list[_GroAtom], dict[str, list[int]], Counter[str], list[str]]:
    ai = _AXIS_TO_INDEX[str(axis)]
    atoms: list[_GroAtom] = []
    groups: dict[str, list[int]] = {"SYSTEM": [], "SURFACE": [], "CORE": []}
    mol_counts: Counter[str] = Counter()
    mol_sequence: list[str] = []
    atomnr = int(start_atom)
    resnr = 1
    for frag in fragments:
        mol_counts[frag.moltype] += 1
        mol_sequence.append(str(frag.moltype))
        frag_min = float(np.min(frag.coords_nm[:, ai]))
        frag_max = float(np.max(frag.coords_nm[:, ai]))
        surface = (frag_min <= float(surface_shell_nm)) or (frag_max >= float(thickness_nm - surface_shell_nm))
        core = (frag_min >= float(core_guard_nm)) and (frag_max <= float(thickness_nm - core_guard_nm))
        for j, xyz in enumerate(np.asarray(frag.coords_nm, dtype=float)):
            atoms.append(_GroAtom(resnr=resnr, resname=str(frag.moltype)[:5], atomname=str(frag.atom_names[j])[:5], atomnr=atomnr, xyz_nm=np.asarray(xyz, dtype=float)))
            groups["SYSTEM"].append(atomnr)
            if surface:
                groups["SURFACE"].append(atomnr)
            if core:
                groups["CORE"].append(atomnr)
            atomnr += 1
        resnr += 1
    return atoms, groups, mol_counts, mol_sequence


def _merge_topology_molecule_counts(*topologies: SystemTopology) -> Counter[str]:
    counts: Counter[str] = Counter()
    for topo in topologies:
        for name, count in topo.molecules:
            counts[str(name)] += int(count)
    return counts


def _compress_molecule_sequence(sequence: Iterable[str]) -> list[tuple[str, int]]:
    entries: list[tuple[str, int]] = []
    current_name: str | None = None
    current_count = 0
    for raw_name in sequence:
        name = str(raw_name)
        if current_name is None:
            current_name = name
            current_count = 1
            continue
        if name == current_name:
            current_count += 1
            continue
        entries.append((current_name, int(current_count)))
        current_name = name
        current_count = 1
    if current_name is not None and current_count > 0:
        entries.append((current_name, int(current_count)))
    return entries


def _write_local_top(path: Path, include_lines: list[str], molecule_entries: list[tuple[str, int]], system_name: str) -> None:
    lines = ["; yadonpy generated system.top"]
    lines.extend(_ordered_include_lines(list(include_lines), base_dir=path.parent))
    lines += ["", "[ system ]", system_name, "", "[ molecules ]"]
    for name, count in molecule_entries:
        lines.append(f"{name} {int(count)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    issues = validate_topology_include_order(path)
    if issues:
        raise RuntimeError(f"Invalid include order in generated topology {path}: {'; '.join(issues)}")


def _prepare_slab(*, source: BulkSource, spec: SlabBuildSpec, route: Route, name: str, out_dir: Path, target_lengths_nm: tuple[float, float], replicas_xy: tuple[int, int], target_thickness_nm: float, area_policy: AreaMismatchPolicy) -> PreparedSlab:
    slab_t0 = time.perf_counter()
    print_stat(f"{name}_numba_accel", _HAS_NUMBA)
    phase_t0 = time.perf_counter()
    frame = _read_gro_frame(source.representative_gro)
    topo = parse_system_top(source.system_top)
    fragments = _split_fragments_from_frame(frame, topo)
    print_stat(f"{name}_split_fragments", len(fragments))
    print_stat(f"{name}_split_elapsed", _format_elapsed_seconds(time.perf_counter() - phase_t0))

    ai = _AXIS_TO_INDEX[str(spec.axis)]
    phase_t0 = time.perf_counter()
    selected, zmin, zmax = _select_fragments_for_slab(fragments, axis=spec.axis, box_nm=frame.box_nm, target_thickness_nm=target_thickness_nm, prefer_densest_window=bool(spec.prefer_densest_window))
    print_stat(f"{name}_selected_fragments_precharge", len(selected))
    print_stat(f"{name}_selected_window_nm", (round(float(zmin), 4), round(float(zmax), 4)))
    print_stat(f"{name}_select_elapsed", _format_elapsed_seconds(time.perf_counter() - phase_t0))

    phase_t0 = time.perf_counter()
    replica_factor = int(max(int(replicas_xy[0]), 1) * max(int(replicas_xy[1]), 1))
    selected, slab_charge_e, charge_notes = _rebalance_fragment_selection_for_charge(selected, all_fragments=fragments, axis=spec.axis, charge_scale=float(replica_factor))
    print_stat(f"{name}_selected_fragments", len(selected))
    print_stat(f"{name}_replica_factor_xy", replica_factor)
    print_stat(f"{name}_slab_charge_e", f"{slab_charge_e:.6f}")
    print_stat(f"{name}_charge_balance_elapsed", _format_elapsed_seconds(time.perf_counter() - phase_t0))

    phase_t0 = time.perf_counter()
    replicated, rep_box = _replicate_xy(selected, box_nm=frame.box_nm, axis=spec.axis, replicas_xy=replicas_xy)
    print_stat(f"{name}_replicated_fragments", len(replicated))
    print_stat(f"{name}_replicated_box_nm", tuple(round(float(x), 4) for x in rep_box))
    print_stat(f"{name}_replicate_elapsed", _format_elapsed_seconds(time.perf_counter() - phase_t0))

    phase_t0 = time.perf_counter()
    fitted, fit_box = _fit_lateral_lengths(replicated, box_nm=rep_box, target_lengths_nm=target_lengths_nm, axis=spec.axis, policy=area_policy)
    fitted = _wrap_fragments_lateral(fitted, box_nm=fit_box, axis=spec.axis)
    print_stat(f"{name}_fit_box_nm", tuple(round(float(x), 4) for x in fit_box))
    print_stat(f"{name}_fit_elapsed", _format_elapsed_seconds(time.perf_counter() - phase_t0))

    zmin = min(float(np.min(f.coords_nm[:, ai])) for f in selected)
    zmax = max(float(np.max(f.coords_nm[:, ai])) for f in selected)
    shift = np.zeros(3, dtype=float)
    shift[ai] = -float(zmin)
    shifted: list[FragmentRecord] = []
    for frag in fitted:
        shifted.append(FragmentRecord(moltype=frag.moltype, natoms=frag.natoms, coords_nm=np.asarray(frag.coords_nm, dtype=float) + shift, atom_names=list(frag.atom_names), charges=list(frag.charges), masses=list(frag.masses), source_instance=frag.source_instance, bonds=tuple(frag.bonds)))
    actual_thickness = float(zmax - zmin)
    lengths = list(fit_box)
    lengths[ai] = max(actual_thickness, target_thickness_nm)
    phase_t0 = time.perf_counter()
    if spec.lateral_recentering:
        shifted = _recenter_fragments_lateral(shifted, box_nm=tuple(lengths), axis=spec.axis)
    else:
        shifted = _wrap_fragments_lateral(shifted, box_nm=tuple(lengths), axis=spec.axis)
    print_stat(f"{name}_recenter_elapsed", _format_elapsed_seconds(time.perf_counter() - phase_t0))

    phase_t0 = time.perf_counter()
    slab_atoms, groups, mol_counts, mol_sequence = _build_slab_groups(shifted, start_atom=1, axis=spec.axis, thickness_nm=actual_thickness, surface_shell_nm=float(spec.surface_shell_nm), core_guard_nm=float(spec.core_guard_nm))
    gro_path = out_dir / f"{name}.gro"
    top_path = out_dir / f"{name}.top"
    ndx_path = out_dir / f"{name}.ndx"
    meta_path = out_dir / f"{name}_meta.json"
    mol_dir = out_dir / "molecules"
    include_lines = _copy_include_tree(source.system_top, mol_dir)
    _write_gro_frame(gro_path, f"prepared slab {name}", slab_atoms, tuple(lengths))
    _write_local_top(top_path, include_lines, _compress_molecule_sequence(mol_sequence), f"prepared slab {name}")
    _write_ndx(ndx_path, [(name.upper(), groups["SYSTEM"]), (f"{name.upper()}_CORE", groups["CORE"]), (f"{name.upper()}_SURFACE", groups["SURFACE"])])
    print_stat(f"{name}_write_elapsed", _format_elapsed_seconds(time.perf_counter() - phase_t0))
    print_stat(f"{name}_prepare_total_elapsed", _format_elapsed_seconds(time.perf_counter() - slab_t0))
    target_area = float(target_lengths_nm[0]) * float(target_lengths_nm[1])
    actual_area = float(lengths[[i for i in range(3) if i != ai][0]]) * float(lengths[[i for i in range(3) if i != ai][1]])
    density = None
    total_mass_amu = sum(sum(f.masses) for f in shifted)
    if target_area > 0 and actual_thickness > 0 and total_mass_amu > 0:
        # 1 amu = 1 g/mol ; 1 mol / NA -> g
        na = 6.02214076e23
        mass_g = total_mass_amu / na
        volume_cm3 = target_area * actual_thickness * 1.0e-21  # nm^3 -> cm^3
        if volume_cm3 > 0:
            density = mass_g / volume_cm3
    _write_json(meta_path, {
        "name": name,
        "axis": spec.axis,
        "route": route,
        "box_nm": tuple(lengths),
        "replicas_xy": [int(replicas_xy[0]), int(replicas_xy[1])],
        "target_thickness_nm": float(target_thickness_nm),
        "actual_thickness_nm": float(actual_thickness),
        "target_area_nm2": target_area,
        "actual_area_nm2": actual_area,
        "density_g_cm3": density,
        "selected_fragments": len(shifted),
        "molecule_counts": dict(mol_counts),
        "net_charge_e": float(slab_charge_e),
        "charge_balance_notes": charge_notes,
        "prefer_densest_window": bool(spec.prefer_densest_window),
        "lateral_recentering": bool(spec.lateral_recentering),
        "source": {"name": source.name, "representative_gro": source.representative_gro, "work_dir": source.work_dir},
    })
    return PreparedSlab(name=name, axis=spec.axis, source_name=source.name, source_representative_gro=source.representative_gro, gro_path=gro_path, top_path=top_path, ndx_path=ndx_path, meta_path=meta_path, thickness_nm=float(actual_thickness), target_thickness_nm=float(target_thickness_nm), box_nm=tuple(lengths), selected_fragments=len(shifted), target_area_nm2=target_area, actual_area_nm2=actual_area, target_density_g_cm3=density, route=route)


def _load_prepared(meta_path: Path) -> PreparedSlab:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    parent = meta_path.parent
    return PreparedSlab(name=str(data["name"]), axis=str(data["axis"]), source_name=str(data["source"]["name"]), source_representative_gro=Path(data["source"]["representative_gro"]), gro_path=parent / f"{data['name']}.gro", top_path=parent / f"{data['name']}.top", ndx_path=parent / f"{data['name']}.ndx", meta_path=meta_path, thickness_nm=float(data["actual_thickness_nm"]), target_thickness_nm=float(data["target_thickness_nm"]), box_nm=tuple(float(x) for x in data["box_nm"]), selected_fragments=int(data["selected_fragments"]), target_area_nm2=float(data["target_area_nm2"]), actual_area_nm2=float(data["actual_area_nm2"]), target_density_g_cm3=(float(data["density_g_cm3"]) if data.get("density_g_cm3") is not None else None), route=str(data["route"]))


def _read_group_from_ndx(path: Path, group_name: str) -> list[int]:
    target = f"[ {group_name} ]"
    lines = _read_text_lines(path)
    capture = False
    out: list[int] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            capture = (line == target)
            continue
        if capture:
            out.extend(int(x) for x in line.split())
    return out


def _build_interface_region_group_inventory(
    *,
    topo: SystemTopology,
    region: str,
    layer_token: str,
    atom_offset: int,
    surface_local_ids: set[int],
    core_local_ids: set[int],
) -> tuple[list[str], dict[str, list[int]], int]:
    region_token = _group_token(region).upper()
    layer_alias = _group_token(layer_token).upper()
    group_order: list[str] = []
    groups: dict[str, list[int]] = {}
    cursor = 0

    for molname, count in topo.molecules:
        mt = topo.moleculetypes.get(molname)
        if mt is None:
            continue
        mol_token = _group_token(molname)
        for inst_idx in range(1, int(count) + 1):
            instance_ids: list[int] = []
            instance_surface_ids: list[int] = []
            instance_core_ids: list[int] = []

            for atom_i in range(mt.natoms):
                global_idx = int(atom_offset + cursor + 1)
                local_idx = int(cursor + 1)
                instance_ids.append(global_idx)

                atomtype_token = _group_token(mt.atomtypes[atom_i])
                _add_group_atoms(group_order, groups, f"{region_token}_ATYPE_{atomtype_token}", global_idx)
                _add_group_atoms(group_order, groups, f"{region_token}_TYPE_{mol_token}_{atomtype_token}", global_idx)
                _add_group_atoms(group_order, groups, f"{layer_alias}_ATYPE_{atomtype_token}", global_idx)
                _add_group_atoms(group_order, groups, f"{layer_alias}_TYPE_{mol_token}_{atomtype_token}", global_idx)
                _add_group_atoms(group_order, groups, f"ATYPE_{atomtype_token}", global_idx)
                _add_group_atoms(group_order, groups, f"TYPE_{mol_token}_{atomtype_token}", global_idx)

                if local_idx in surface_local_ids:
                    instance_surface_ids.append(global_idx)
                    _add_group_atoms(group_order, groups, f"{region_token}_SURFACE_ATYPE_{atomtype_token}", global_idx)
                    _add_group_atoms(group_order, groups, f"{region_token}_SURFACE_TYPE_{mol_token}_{atomtype_token}", global_idx)
                    _add_group_atoms(group_order, groups, f"{layer_alias}_SURFACE_ATYPE_{atomtype_token}", global_idx)
                    _add_group_atoms(group_order, groups, f"{layer_alias}_SURFACE_TYPE_{mol_token}_{atomtype_token}", global_idx)
                if local_idx in core_local_ids:
                    instance_core_ids.append(global_idx)
                    _add_group_atoms(group_order, groups, f"{region_token}_CORE_ATYPE_{atomtype_token}", global_idx)
                    _add_group_atoms(group_order, groups, f"{region_token}_CORE_TYPE_{mol_token}_{atomtype_token}", global_idx)
                    _add_group_atoms(group_order, groups, f"{layer_alias}_CORE_ATYPE_{atomtype_token}", global_idx)
                    _add_group_atoms(group_order, groups, f"{layer_alias}_CORE_TYPE_{mol_token}_{atomtype_token}", global_idx)

                cursor += 1

            _add_group_atoms(group_order, groups, f"{region_token}_MOL_{mol_token}", instance_ids)
            _add_group_atoms(group_order, groups, f"{layer_alias}_{mol_token}", instance_ids)
            _add_group_atoms(group_order, groups, f"MOL_{mol_token}", instance_ids)
            _add_group_atoms(group_order, groups, f"{region_token}_INST_{mol_token}_{inst_idx:04d}", instance_ids)
            _add_group_atoms(group_order, groups, f"{layer_alias}_INST_{mol_token}_{inst_idx:04d}", instance_ids)
            if instance_ids:
                _add_group_atoms(group_order, groups, f"{region_token}_REP_{mol_token}", instance_ids[0])
                _add_group_atoms(group_order, groups, f"{layer_alias}_REP_{mol_token}", instance_ids[0])
                _add_group_atoms(group_order, groups, f"REP_{mol_token}", instance_ids[0])
            if instance_surface_ids:
                _add_group_atoms(group_order, groups, f"{region_token}_SURFACE_MOL_{mol_token}", instance_surface_ids)
                _add_group_atoms(group_order, groups, f"{layer_alias}_SURFACE_{mol_token}", instance_surface_ids)
            if instance_core_ids:
                _add_group_atoms(group_order, groups, f"{region_token}_CORE_MOL_{mol_token}", instance_core_ids)
                _add_group_atoms(group_order, groups, f"{layer_alias}_CORE_{mol_token}", instance_core_ids)

    return group_order, groups, cursor


def _assemble_interface(*, name: str, out_dir: Path, route_spec: InterfaceRouteSpec, bottom: PreparedSlab, top: PreparedSlab) -> BuiltInterface:
    interface_dir = out_dir / "03_interface"
    interface_dir.mkdir(parents=True, exist_ok=True)
    mol_dir = interface_dir / "molecules"
    mol_dir.mkdir(parents=True, exist_ok=True)
    include_lines: list[str] = []
    parameter_paths: list[Path] = []
    seen: set[str] = set()
    copied_names: dict[tuple[str, str], str] = {}
    content_by_name: dict[str, bytes] = {}
    for source_tag, top_path in [(bottom.source_name, bottom.top_path), (top.source_name, top.top_path)]:
        include_lines.extend(
            _collect_normalized_include_lines(
                top_path=top_path,
                dest_dir=mol_dir,
                source_tag=source_tag,
                seen=seen,
                copied_names=copied_names,
                content_by_name=content_by_name,
                collision_mode="rename",
                parameter_paths=parameter_paths,
            )
        )

    if _write_merged_parameter_include(dest_path=mol_dir / "ff_parameters.itp", source_paths=parameter_paths):
        include_lines.insert(0, '#include "molecules/ff_parameters.itp"')

    bottom_frame = _read_gro_frame(bottom.gro_path)
    top_frame = _read_gro_frame(top.gro_path)
    bottom_topology = parse_system_top(bottom.top_path)
    top_topology = parse_system_top(top.top_path)
    ai = _AXIS_TO_INDEX[str(route_spec.axis)]
    lateral = [d for d in [0, 1, 2] if d != ai]
    box = [0.0, 0.0, 0.0]
    box[lateral[0]] = max(float(bottom.box_nm[lateral[0]]), float(top.box_nm[lateral[0]]))
    box[lateral[1]] = max(float(bottom.box_nm[lateral[1]]), float(top.box_nm[lateral[1]]))
    gap = max(float(route_spec.bottom.gap_nm), float(route_spec.top.gap_nm))
    vacuum = max(float(route_spec.bottom.vacuum_nm), float(route_spec.top.vacuum_nm))
    if route_spec.route == "route_a":
        box[ai] = float(bottom.thickness_nm + top.thickness_nm + gap)
        bottom_offset = np.zeros(3, dtype=float)
        top_offset = np.zeros(3, dtype=float)
        top_offset[ai] = float(bottom.thickness_nm + gap)
    else:
        box[ai] = float(bottom.thickness_nm + top.thickness_nm + gap + vacuum)
        bottom_offset = np.zeros(3, dtype=float)
        top_offset = np.zeros(3, dtype=float)
        top_offset[ai] = float(bottom.thickness_nm + gap)
    lateral_shift = np.zeros(3, dtype=float)
    lateral_shift[lateral[0]] = float(route_spec.top_lateral_shift_fraction[0]) * float(box[lateral[0]])
    lateral_shift[lateral[1]] = float(route_spec.top_lateral_shift_fraction[1]) * float(box[lateral[1]])

    atoms: list[_GroAtom] = []
    bottom_ids: list[int] = []
    top_ids: list[int] = []
    bottom_surface = set(_read_group_from_ndx(bottom.ndx_path, "BOTTOM_SURFACE"))
    bottom_core = set(_read_group_from_ndx(bottom.ndx_path, "BOTTOM_CORE"))
    top_surface = set(_read_group_from_ndx(top.ndx_path, "TOP_SURFACE"))
    top_core = set(_read_group_from_ndx(top.ndx_path, "TOP_CORE"))
    groups = {"BOTTOM": [], "TOP": [], "BOTTOM_SURFACE": [], "BOTTOM_CORE": [], "TOP_SURFACE": [], "TOP_CORE": [], "INTERFACE_ZONE": []}
    atomnr = 1
    resnr = 1
    for src_atom in bottom_frame.atoms:
        xyz = np.asarray(src_atom.xyz_nm, dtype=float) + bottom_offset
        atom = _GroAtom(resnr=resnr, resname=src_atom.resname, atomname=src_atom.atomname, atomnr=atomnr, xyz_nm=xyz)
        atoms.append(atom)
        groups["BOTTOM"].append(atomnr)
        bottom_ids.append(atomnr)
        if src_atom.atomnr in bottom_surface:
            groups["BOTTOM_SURFACE"].append(atomnr)
            groups["INTERFACE_ZONE"].append(atomnr)
        if src_atom.atomnr in bottom_core:
            groups["BOTTOM_CORE"].append(atomnr)
        atomnr += 1
        resnr = max(resnr, src_atom.resnr + 1)
    resnr += 1
    top_cursor = 0
    for molname, count in top_topology.molecules:
        mt = top_topology.moleculetypes.get(molname)
        if mt is None:
            raise KeyError(f"Molecule type '{molname}' not found in top slab topology")
        nat = int(mt.natoms)
        for _ in range(int(count)):
            block = top_frame.atoms[top_cursor: top_cursor + nat]
            if len(block) != nat:
                raise RuntimeError(
                    f"Top slab topology/frame mismatch for {molname}: expected {nat} atoms, got {len(block)}"
                )
            block_xyz = np.vstack([np.asarray(src_atom.xyz_nm, dtype=float) for src_atom in block]) + top_offset + lateral_shift
            block_xyz = _shift_coords_into_box_by_com(block_xyz, box_nm=tuple(box), dims=lateral)
            for src_atom, xyz in zip(block, block_xyz):
                atom = _GroAtom(resnr=resnr, resname=src_atom.resname, atomname=src_atom.atomname, atomnr=atomnr, xyz_nm=xyz)
                atoms.append(atom)
                groups["TOP"].append(atomnr)
                top_ids.append(atomnr)
                if src_atom.atomnr in top_surface:
                    groups["TOP_SURFACE"].append(atomnr)
                    groups["INTERFACE_ZONE"].append(atomnr)
                if src_atom.atomnr in top_core:
                    groups["TOP_CORE"].append(atomnr)
                atomnr += 1
            top_cursor += nat
            resnr += 1
    if top_cursor != len(top_frame.atoms):
        raise RuntimeError(
            f"Top slab frame has trailing atoms after topology consumption: consumed {top_cursor}, total {len(top_frame.atoms)}"
        )

    inventory_order: list[str] = []
    inventory_groups: dict[str, list[int]] = {}
    bottom_order, bottom_inventory, bottom_consumed = _build_interface_region_group_inventory(
        topo=bottom_topology,
        region="BOTTOM",
        layer_token="BOTTOM1",
        atom_offset=0,
        surface_local_ids=bottom_surface,
        core_local_ids=bottom_core,
    )
    top_order, top_inventory, top_consumed = _build_interface_region_group_inventory(
        topo=top_topology,
        region="TOP",
        layer_token="TOP1",
        atom_offset=len(bottom_frame.atoms),
        surface_local_ids=top_surface,
        core_local_ids=top_core,
    )
    if bottom_consumed != len(bottom_frame.atoms):
        raise RuntimeError(
            f"Bottom slab topology/frame mismatch: consumed {bottom_consumed} atoms, frame has {len(bottom_frame.atoms)}"
        )
    if top_consumed != len(top_frame.atoms):
        raise RuntimeError(
            f"Top slab topology/frame mismatch: consumed {top_consumed} atoms, frame has {len(top_frame.atoms)}"
        )
    for group_name in bottom_order:
        _add_group_atoms(inventory_order, inventory_groups, group_name, bottom_inventory[group_name])
    for group_name in top_order:
        _add_group_atoms(inventory_order, inventory_groups, group_name, top_inventory[group_name])

    atom_by_id = {int(atom.atomnr): atom for atom in atoms}
    bottom_z = [float(atom_by_id[idx].xyz_nm[ai]) for idx in groups["BOTTOM"] if int(idx) in atom_by_id]
    top_z = [float(atom_by_id[idx].xyz_nm[ai]) for idx in groups["TOP"] if int(idx) in atom_by_id]
    assembled_gap_nm = float(min(top_z) - max(bottom_z)) if bottom_z and top_z else None
    route_b_wall_clearance_note: str | None = None
    if route_spec.route == "route_b" and atoms:
        axis_coords = [float(atom.xyz_nm[ai]) for atom in atoms]
        min_axis = min(axis_coords)
        max_axis = max(axis_coords)
        lower_shift = max(0.0, float(_ROUTE_B_WALL_CLEARANCE_NM) - min_axis)
        upper_padding = max(0.0, float(_ROUTE_B_WALL_CLEARANCE_NM) - (float(box[ai]) - max_axis))
        if lower_shift > 0.0 or upper_padding > 0.0:
            for atom in atoms:
                atom.xyz_nm[ai] = float(atom.xyz_nm[ai]) + lower_shift
            box[ai] = float(box[ai]) + lower_shift + upper_padding
            route_b_wall_clearance_note = (
                "route_b interface geometry was shifted away from the z-wall before MD "
                f"(lower_shift_nm={lower_shift:.4f}, upper_padding_nm={upper_padding:.4f})"
            )

    system_gro = interface_dir / "system.gro"
    _write_gro_frame(system_gro, f"interface {name}", atoms, tuple(box))

    bottom_meta = json.loads(bottom.meta_path.read_text(encoding="utf-8"))
    top_meta = json.loads(top.meta_path.read_text(encoding="utf-8"))
    molecule_entries = _compress_molecule_sequence(
        [name for name, count in bottom_topology.molecules for _ in range(int(count))]
        + [name for name, count in top_topology.molecules for _ in range(int(count))]
    )
    mol_counts: Counter[str] = Counter()
    for molname, count in molecule_entries:
        mol_counts[str(molname)] += int(count)
    system_top = interface_dir / "system.top"
    _write_local_top(system_top, include_lines, molecule_entries, f"interface {name}")
    whole_fix = normalize_gro_molecules_inplace(top=system_top, gro=system_gro)
    primary_box_wrap = _wrap_gro_atoms_into_primary_box(system_gro, dims=lateral)
    geometry_validation = _validate_assembled_interface_geometry(
        gro_path=system_gro,
        box_nm=tuple(box),
        axis=route_spec.axis,
        bottom_atom_ids=groups["BOTTOM"],
        top_atom_ids=groups["TOP"],
        assembled_gap_nm=assembled_gap_nm,
    )
    if int(geometry_validation.get("atoms_outside_primary_box", 0)) > 0:
        raise RuntimeError(
            "Assembled interface geometry left atoms outside the primary box after canonicalization: "
            f"atoms_outside_primary_box={int(geometry_validation['atoms_outside_primary_box'])}, "
            f"max_primary_box_overflow_nm={float(geometry_validation.get('max_primary_box_overflow_nm', 0.0)):.6f}"
        )
    if geometry_validation.get("assembled_gap_nm") is not None and float(geometry_validation["assembled_gap_nm"]) < -0.05:
        raise RuntimeError(
            "Assembled interface geometry produced a strongly negative slab gap before MD: "
            f"assembled_gap_nm={float(geometry_validation['assembled_gap_nm']):.6f}"
        )

    system_ndx = interface_dir / "system.ndx"
    ndx_groups = [
        ("System", list(range(1, len(atoms) + 1))),
        ("BOTTOM", groups["BOTTOM"]),
        ("BOTTOM1", groups["BOTTOM"]),
        ("TOP", groups["TOP"]),
        ("TOP1", groups["TOP"]),
        ("BOTTOM_CORE", groups["BOTTOM_CORE"]),
        ("BOTTOM1_CORE", groups["BOTTOM_CORE"]),
        ("BOTTOM_SURFACE", groups["BOTTOM_SURFACE"]),
        ("BOTTOM1_SURFACE", groups["BOTTOM_SURFACE"]),
        ("TOP_CORE", groups["TOP_CORE"]),
        ("TOP1_CORE", groups["TOP_CORE"]),
        ("TOP_SURFACE", groups["TOP_SURFACE"]),
        ("TOP1_SURFACE", groups["TOP_SURFACE"]),
        ("INTERFACE_ZONE", sorted(set(groups["INTERFACE_ZONE"]))),
    ]
    for group_name in inventory_order:
        ndx_groups.append((group_name, inventory_groups[group_name]))
    _write_ndx(system_ndx, ndx_groups)
    export_interface_group_catalog(system_ndx, interface_dir / "system_ndx_groups.json")

    system_meta = interface_dir / "system_meta.json"
    protocol_manifest = interface_dir / "protocol_manifest.json"
    notes = ["slabs were prepared from representative bulk snapshots using dense-window slab selection and lateral recentring before interface assembly", f"top slab lateral phase shift fractions = {tuple(float(x) for x in route_spec.top_lateral_shift_fraction)}"]
    if whole_fix.get("applied"):
        notes.append(
            f"topology-guided whole-molecule canonicalization applied to assembled interface geometry; normalized_molecules={int(whole_fix.get('normalized_molecules', 0))}"
        )
    elif whole_fix.get("error"):
        notes.append(f"assembled interface canonicalization skipped: {whole_fix['error']}")
    if primary_box_wrap.get("applied"):
        notes.append(
            "final interface GRO was wrapped back into the primary box along lateral dimensions "
            f"{tuple(int(d) for d in lateral)} after whole-molecule canonicalization; "
            f"wrapped_components={int(primary_box_wrap.get('wrapped_components', 0))}"
        )
    if geometry_validation.get("assembled_gap_nm") is not None:
        notes.append(
            f"assembled interface validation: atoms_outside_primary_box={int(geometry_validation.get('atoms_outside_primary_box', 0))}, assembled_gap_nm={float(geometry_validation['assembled_gap_nm']):.6f}"
        )
    if route_b_wall_clearance_note:
        notes.append(route_b_wall_clearance_note)
    if route_spec.route == "route_b":
        notes.append("route_b geometry adds vacuum padding only; wall settings belong to the MD protocol stage")
    total_charge = 0.0
    for topo in (bottom_topology, top_topology):
        for molname, count in topo.molecules:
            mt = topo.moleculetypes.get(molname)
            if mt is None:
                continue
            total_charge += float(mt.net_charge) * int(count)
    _write_json(system_meta, {"name": name, "route": route_spec.route, "axis": route_spec.axis, "box_nm": tuple(box), "gap_nm": gap, "vacuum_nm": vacuum, "top_lateral_shift_fraction": tuple(float(x) for x in route_spec.top_lateral_shift_fraction), "bottom_slab": bottom.meta_path, "top_slab": top.meta_path, "molecule_counts": dict(mol_counts), "net_charge_e": float(total_charge), "geometry_validation": geometry_validation, "notes": notes})
    _write_json(
        protocol_manifest,
        {
            "route": route_spec.route,
            "stages": {
                "pre_contact": "Gap-preserving EM/NVT scaffold immediately after interface assembly",
                "density_relax": "Optional phase-wise density relaxation while early slab support is still active",
                "contact": "Gentle contact stage after the initial vacuum gap",
                "release": "Optional staged removal of early slab support before unrestricted exchange",
                "exchange": "Unrestricted semiisotropic interface exchange equilibration",
                "production": "Production interface sampling",
            },
            "builder_notes": notes,
        },
    )
    return BuiltInterface(name=name, route=route_spec.route, axis=route_spec.axis, out_dir=interface_dir, system_gro=system_gro, system_top=system_top, system_ndx=system_ndx, system_meta=system_meta, bottom_slab=bottom, top_slab=top, protocol_manifest=protocol_manifest, box_nm=tuple(box), notes=tuple(notes))


class InterfaceBuilder:
    """Build interface geometries from equilibrated bulk simulation folders.

    The builder intentionally stops at geometry / topology / index export.
    Interfacial MD settings (for example wall parameters) belong to the
    interface protocol layer.
    """

    def __init__(self, *, work_dir: str | Path | WorkDir, restart: Optional[bool] = None):
        self.work_dir = workdir(work_dir, restart=restart)
        self.restart = bool(self.work_dir.restart)

    def bulk_source(self, *, name: str, work_dir: str | Path) -> BulkSource:
        snap_dir = Path(self.work_dir) / "01_snapshots" / _safe_name(name)
        return _extract_snapshot(name=name, bulk_root=Path(work_dir), out_dir=snap_dir, restart=self.restart)

    def build_from_bulk_workdirs(self, *, name: str, bottom_name: str, bottom_work_dir: str | Path, top_name: str, top_work_dir: str | Path, route: InterfaceRouteSpec) -> BuiltInterface:
        bottom = self.bulk_source(name=bottom_name, work_dir=bottom_work_dir)
        top = self.bulk_source(name=top_name, work_dir=top_work_dir)
        return self.build(name=name, bottom=bottom, top=top, route=route)

    def build(self, *, name: str, bottom: BulkSource, top: BulkSource, route: InterfaceRouteSpec) -> BuiltInterface:
        root = Path(self.work_dir)
        resume = ResumeManager(root, enabled=self.restart, strict_inputs=True)
        built_meta = root / "03_interface" / "system_meta.json"
        spec = StepSpec(
            name=f"build_interface_{_safe_name(name)}",
            outputs=[root / "03_interface" / "system.gro", root / "03_interface" / "system.top", root / "03_interface" / "system.ndx", root / "03_interface" / "system_ndx_groups.json", built_meta],
            inputs={
                "builder_schema_version": _INTERFACE_BUILD_SCHEMA_VERSION,
                "name": name,
                "route": route.route,
                "axis": route.axis,
                "bottom_snapshot": str(bottom.representative_gro),
                "top_snapshot": str(top.representative_gro),
                "bottom_time_ps": bottom.representative_time_ps,
                "top_time_ps": top.representative_time_ps,
                "gap_nm": max(route.bottom.gap_nm, route.top.gap_nm),
                "vacuum_nm": max(route.bottom.vacuum_nm, route.top.vacuum_nm),
                "bottom_target_thickness_nm": route.bottom.target_thickness_nm,
                "top_target_thickness_nm": route.top.target_thickness_nm,
                "bottom_surface_shell_nm": route.bottom.surface_shell_nm,
                "top_surface_shell_nm": route.top.surface_shell_nm,
                "bottom_core_guard_nm": route.bottom.core_guard_nm,
                "top_core_guard_nm": route.top.core_guard_nm,
                "bottom_min_replicas_xy": list(route.bottom.min_replicas_xy),
                "top_min_replicas_xy": list(route.top.min_replicas_xy),
                "bottom_prefer_densest_window": bool(route.bottom.prefer_densest_window),
                "top_prefer_densest_window": bool(route.top.prefer_densest_window),
                "bottom_lateral_recentering": bool(route.bottom.lateral_recentering),
                "top_lateral_recentering": bool(route.top.lateral_recentering),
                "top_lateral_shift_fraction": list(route.top_lateral_shift_fraction),
                "prefer_larger_area": bool(route.area_policy.prefer_larger_area),
                "max_lateral_strain": route.area_policy.max_lateral_strain,
                "max_lateral_replicas_xy": list(route.area_policy.max_lateral_replicas_xy),
            },
            description="Build interface slabs and assemble the interface system",
        )
        result_cache: dict[str, Any] = {}

        def _run() -> None:
            print_section("Interface build workflow", detail=f"name={name} | route={route.route} | axis={route.axis}")
            print_item("bottom_source", bottom.name)
            print_item("top_source", top.name)
            bottom_frame = _read_gro_frame(bottom.representative_gro)
            top_frame = _read_gro_frame(top.representative_gro)
            print_step("1/4 Resolve lateral target box")
            lateral_plan = _resolve_lateral_sizing_plan(
                bottom_box_nm=bottom_frame.box_nm,
                top_box_nm=top_frame.box_nm,
                axis=route.axis,
                policy=route.area_policy,
                bottom_min_replicas_xy=route.bottom.min_replicas_xy,
                top_min_replicas_xy=route.top.min_replicas_xy,
            )
            target_lengths = lateral_plan.target_lengths_nm
            print_stat("target_lengths_nm", tuple(round(float(x), 4) for x in target_lengths))
            print_stat("bottom_replicas_xy", lateral_plan.bottom_replicas_xy)
            print_stat("top_replicas_xy", lateral_plan.top_replicas_xy)
            print_step("2/4 Prepare bottom slab", detail=bottom.name)
            bottom_slab = _prepare_slab(source=bottom, spec=route.bottom, route=route.route, name="bottom", out_dir=root / "02_slabs" / "bottom", target_lengths_nm=target_lengths, replicas_xy=lateral_plan.bottom_replicas_xy, target_thickness_nm=float(route.bottom.target_thickness_nm or 4.0), area_policy=route.area_policy)
            print_stat("bottom_fragments", bottom_slab.selected_fragments)
            print_stat("bottom_thickness", f"{bottom_slab.thickness_nm:.3f} nm")
            print_item("bottom_out", bottom_slab.gro_path)
            print_step("3/4 Prepare top slab", detail=top.name)
            top_slab = _prepare_slab(source=top, spec=route.top, route=route.route, name="top", out_dir=root / "02_slabs" / "top", target_lengths_nm=target_lengths, replicas_xy=lateral_plan.top_replicas_xy, target_thickness_nm=float(route.top.target_thickness_nm or 4.0), area_policy=route.area_policy)
            print_stat("top_fragments", top_slab.selected_fragments)
            print_stat("top_thickness", f"{top_slab.thickness_nm:.3f} nm")
            print_item("top_out", top_slab.gro_path)
            print_step("4/4 Assemble interface system")
            result_cache["built"] = _assemble_interface(name=name, out_dir=root, route_spec=route, bottom=bottom_slab, top=top_slab)
            print_done("Interface build workflow", detail=f"system={result_cache['built'].system_gro}")

        resume.run(spec, _run)
        if "built" in result_cache:
            return result_cache["built"]
        payload = json.loads(built_meta.read_text(encoding="utf-8"))
        return BuiltInterface(name=name, route=str(payload["route"]), axis=str(payload["axis"]), out_dir=root / "03_interface", system_gro=root / "03_interface" / "system.gro", system_top=root / "03_interface" / "system.top", system_ndx=root / "03_interface" / "system.ndx", system_meta=built_meta, bottom_slab=_load_prepared(Path(payload["bottom_slab"])), top_slab=_load_prepared(Path(payload["top_slab"])), protocol_manifest=root / "03_interface" / "protocol_manifest.json", box_nm=tuple(float(x) for x in payload["box_nm"]), notes=tuple(payload.get("notes") or []))


def build_interface(*, work_dir: str | Path | WorkDir, name: str, bottom: BulkSource, top: BulkSource, route: InterfaceRouteSpec, restart: Optional[bool] = None) -> BuiltInterface:
    builder = InterfaceBuilder(work_dir=work_dir, restart=restart)
    return builder.build(name=name, bottom=bottom, top=top, route=route)


def build_interface_from_workdirs(*, work_dir: str | Path | WorkDir, name: str, bottom_name: str, bottom_work_dir: str | Path, top_name: str, top_work_dir: str | Path, route: InterfaceRouteSpec, restart: Optional[bool] = None) -> BuiltInterface:
    builder = InterfaceBuilder(work_dir=work_dir, restart=restart)
    return builder.build_from_bulk_workdirs(name=name, bottom_name=bottom_name, bottom_work_dir=bottom_work_dir, top_name=top_name, top_work_dir=top_work_dir, route=route)


__all__ = ["AreaMismatchPolicy", "BulkSource", "BuiltInterface", "InterfaceBuilder", "build_interface", "build_interface_from_workdirs", "InterfaceRouteSpec", "PreparedSlab", "SlabBuildSpec"]
