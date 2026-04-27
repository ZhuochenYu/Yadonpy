"""Small study-level configuration objects for repeatable workflows.

These dataclasses capture resources, restart behavior, and study metadata that
span multiple workflow steps. They give scripts a typed place to store run-level
intent without turning YadonPy's script-first style into a large framework.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Literal

from . import steps


@dataclass(frozen=True)
class StudyResources:
    mpi: int = 1
    omp: int = 16
    gpu: int = 1
    gpu_id: int | None = 0


@dataclass(frozen=True)
class PreparedSystem:
    gro: Path
    top: Path
    source: str | None = None
    work_dir: Path | None = None


@dataclass(frozen=True)
class MechanicsStudyResult:
    kind: Literal["tg", "elongation"]
    summary_path: Path
    out_dir: Path
    prepared: PreparedSystem
    summary: Mapping[str, Any] = field(default_factory=dict)


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _resolve_system_top(work_dir: Path) -> Path:
    candidates = [
        work_dir / "00_system" / "system.top",
        work_dir / "02_system" / "system.top",
        work_dir / "system.top",
    ]
    for path in candidates:
        if path.exists():
            return path
    fallback = sorted(work_dir.glob("**/system.top"))
    if fallback:
        return fallback[0]
    raise FileNotFoundError(f"Could not resolve system.top under {work_dir}")


def resolve_prepared_system(
    *,
    gro: str | Path | None = None,
    top: str | Path | None = None,
    work_dir: str | Path | None = None,
    source_name: str | None = None,
) -> PreparedSystem:
    gro_path = _as_path(gro)
    top_path = _as_path(top)
    work_dir_path = _as_path(work_dir)

    if gro_path is not None or top_path is not None:
        if gro_path is None or top_path is None:
            raise ValueError("gro and top must be provided together when not using work_dir resolution")
        if not gro_path.exists():
            raise FileNotFoundError(f"GRO file not found: {gro_path}")
        if not top_path.exists():
            raise FileNotFoundError(f"TOP file not found: {top_path}")
        return PreparedSystem(
            gro=gro_path,
            top=top_path,
            source=(str(source_name).strip() if source_name else None),
            work_dir=work_dir_path,
        )

    if work_dir_path is None:
        raise ValueError("resolve_prepared_system requires either gro+top or work_dir")
    if not work_dir_path.exists():
        raise FileNotFoundError(f"Work directory not found: {work_dir_path}")

    from ..sim.preset.eq import _find_latest_equilibrated_gro

    gro_path = _find_latest_equilibrated_gro(work_dir_path)
    if gro_path is None:
        raise FileNotFoundError(f"Could not resolve an equilibrated GRO under {work_dir_path}")
    top_path = _resolve_system_top(work_dir_path)
    source = str(source_name).strip() if source_name else work_dir_path.name
    return PreparedSystem(gro=gro_path, top=top_path, source=source, work_dir=work_dir_path)


def _tg_profile_defaults(profile: str) -> dict[str, object]:
    key = str(profile).strip().lower()
    if key == "smoke":
        return {
            "temperatures_k": [460.0, 420.0, 380.0, 340.0, 300.0],
            "pressure_bar": 1.0,
            "npt_ns": 0.2,
            "frac_last": 0.5,
        }
    if key == "production":
        return {
            "temperatures_k": [520.0, 500.0, 480.0, 460.0, 440.0, 420.0, 400.0, 380.0, 360.0, 340.0, 320.0, 300.0, 280.0],
            "pressure_bar": 1.0,
            "npt_ns": 3.0,
            "frac_last": 0.5,
        }
    return {
        "temperatures_k": [500.0, 480.0, 460.0, 440.0, 420.0, 400.0, 380.0, 360.0, 340.0, 320.0, 300.0],
        "pressure_bar": 1.0,
        "npt_ns": 2.0,
        "frac_last": 0.5,
    }


def _elongation_profile_defaults(profile: str) -> dict[str, object]:
    key = str(profile).strip().lower()
    if key == "smoke":
        return {
            "temperature_k": 300.0,
            "pressure_bar": 1.0,
            "strain_rate_1_ps": 2.0e-5,
            "total_strain": 0.10,
            "dt_ps": 0.002,
        }
    if key == "production":
        return {
            "temperature_k": 300.0,
            "pressure_bar": 1.0,
            "strain_rate_1_ps": 5.0e-7,
            "total_strain": 0.60,
            "dt_ps": 0.002,
        }
    return {
        "temperature_k": 300.0,
        "pressure_bar": 1.0,
        "strain_rate_1_ps": 1.0e-6,
        "total_strain": 0.50,
        "dt_ps": 0.002,
    }


def _load_summary(path: Path) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_tg_scan_gmx(
    *,
    prepared: PreparedSystem | None = None,
    gro: str | Path | None = None,
    top: str | Path | None = None,
    work_dir: str | Path | None = None,
    source_name: str | None = None,
    out_dir: str | Path,
    profile: str = "default",
    resources: StudyResources = StudyResources(),
    temperatures_k: list[float] | tuple[float, ...] | None = None,
    pressure_bar: float | None = None,
    npt_ns: float | None = None,
    frac_last: float | None = None,
    restart: bool | None = None,
    auto_plot: bool = True,
    fit_metric: Literal["density", "specific_volume"] = "density",
) -> MechanicsStudyResult:
    prepared_system = prepared or resolve_prepared_system(
        gro=gro,
        top=top,
        work_dir=work_dir,
        source_name=source_name,
    )
    defaults = _tg_profile_defaults(profile)
    summary_path = steps.tg_scan_gmx(
        gro=prepared_system.gro,
        top=prepared_system.top,
        out_dir=Path(out_dir),
        temperatures_k=list(temperatures_k or defaults["temperatures_k"]),
        pressure_bar=float(defaults["pressure_bar"] if pressure_bar is None else pressure_bar),
        npt_ns=float(defaults["npt_ns"] if npt_ns is None else npt_ns),
        frac_last=float(defaults["frac_last"] if frac_last is None else frac_last),
        mpi=int(resources.mpi),
        omp=int(resources.omp),
        gpu=int(resources.gpu),
        gpu_id=resources.gpu_id,
        restart=restart,
        auto_plot=bool(auto_plot),
        fit_metric=str(fit_metric),
    )
    return MechanicsStudyResult(
        kind="tg",
        summary_path=Path(summary_path),
        out_dir=Path(out_dir),
        prepared=prepared_system,
        summary=_load_summary(Path(summary_path)),
    )


def run_elongation_gmx(
    *,
    prepared: PreparedSystem | None = None,
    gro: str | Path | None = None,
    top: str | Path | None = None,
    work_dir: str | Path | None = None,
    source_name: str | None = None,
    out_dir: str | Path,
    profile: str = "default",
    resources: StudyResources = StudyResources(),
    temperature_k: float | None = None,
    pressure_bar: float | None = None,
    strain_rate_1_ps: float | None = None,
    total_strain: float | None = None,
    dt_ps: float | None = None,
    restart: bool | None = None,
    auto_plot: bool = True,
    direction: Literal["x", "y", "z"] = "x",
    modulus_fit_max_strain: float = 0.02,
    modulus_fit_min_points: int = 5,
) -> MechanicsStudyResult:
    prepared_system = prepared or resolve_prepared_system(
        gro=gro,
        top=top,
        work_dir=work_dir,
        source_name=source_name,
    )
    defaults = _elongation_profile_defaults(profile)
    summary_path = steps.elongation_gmx(
        gro=prepared_system.gro,
        top=prepared_system.top,
        out_dir=Path(out_dir),
        temperature_k=float(defaults["temperature_k"] if temperature_k is None else temperature_k),
        pressure_bar=float(defaults["pressure_bar"] if pressure_bar is None else pressure_bar),
        strain_rate_1_ps=float(defaults["strain_rate_1_ps"] if strain_rate_1_ps is None else strain_rate_1_ps),
        total_strain=float(defaults["total_strain"] if total_strain is None else total_strain),
        dt_ps=float(defaults["dt_ps"] if dt_ps is None else dt_ps),
        mpi=int(resources.mpi),
        omp=int(resources.omp),
        gpu=int(resources.gpu),
        gpu_id=resources.gpu_id,
        restart=restart,
        auto_plot=bool(auto_plot),
        direction=str(direction),
        modulus_fit_max_strain=float(modulus_fit_max_strain),
        modulus_fit_min_points=int(modulus_fit_min_points),
    )
    return MechanicsStudyResult(
        kind="elongation",
        summary_path=Path(summary_path),
        out_dir=Path(out_dir),
        prepared=prepared_system,
        summary=_load_summary(Path(summary_path)),
    )


def format_mechanics_result_summary(result: MechanicsStudyResult) -> tuple[str, ...]:
    lines = [
        f"study = {result.kind}",
        f"summary_path = {result.summary_path}",
        f"source_gro = {result.prepared.gro}",
        f"source_top = {result.prepared.top}",
    ]
    summary = dict(result.summary)
    if result.kind == "tg":
        fit = dict(summary.get("fit") or {})
        lines.extend(
            [
                f"fit_metric = {fit.get('fit_metric', 'density')}",
                f"tg_k = {fit.get('tg_k')}",
                f"split_index = {fit.get('split_index')}",
                f"total_sse = {fit.get('total_sse')}",
            ]
        )
    else:
        material = dict(summary.get("material_summary") or {})
        lines.extend(
            [
                f"direction = {summary.get('direction')}",
                f"youngs_modulus_gpa = {material.get('youngs_modulus_gpa')}",
                f"max_stress_gpa = {material.get('max_stress_gpa')}",
                f"strain_at_max_stress = {material.get('strain_at_max_stress')}",
            ]
        )
    return tuple(lines)


def print_mechanics_result_summary(result: MechanicsStudyResult) -> None:
    for line in format_mechanics_result_summary(result):
        print(line)


__all__ = [
    "MechanicsStudyResult",
    "PreparedSystem",
    "StudyResources",
    "format_mechanics_result_summary",
    "print_mechanics_result_summary",
    "resolve_prepared_system",
    "run_elongation_gmx",
    "run_tg_scan_gmx",
]
