from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class PackBackoffPolicy:
    phase: str
    charged: bool
    initial_density_g_cm3: float
    max_attempts: int
    backoff_factor: float
    floor_density_g_cm3: float


@dataclass(frozen=True)
class PackBackoffResult:
    cell: object
    selected_density_g_cm3: float
    selected_attempt_index: int
    attempts: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    summary_path: Path


def initial_bulk_pack_density(
    *,
    target_density_g_cm3: float,
    phase: str,
    requested_density_g_cm3: float | None = None,
    z_scale: float | None = None,
    charged: bool = False,
) -> float:
    if requested_density_g_cm3 is not None and float(requested_density_g_cm3) > 0.0:
        return float(requested_density_g_cm3)
    phase_key = str(phase).strip().lower()
    target = float(target_density_g_cm3)
    if phase_key == "polymer":
        if bool(charged):
            density = max(0.34, min(0.56, target * 0.40))
            if z_scale is not None and float(z_scale) > 1.0:
                density = max(0.30, float(density) / float(z_scale))
            return float(density)
        density = max(0.45, min(0.68, target * 0.52))
        if z_scale is not None and float(z_scale) > 1.0:
            density = max(0.38, float(density) / float(z_scale))
        return float(density)
    return max(0.65, min(0.90, target * 0.80))


def build_pack_density_ladder(
    *,
    phase: str,
    target_density_g_cm3: float,
    requested_density_g_cm3: float | None = None,
    z_scale: float | None = None,
    charged: bool = False,
    max_attempts: int | None = None,
    backoff_factor: float | None = None,
    floor_density_g_cm3: float | None = None,
) -> tuple[PackBackoffPolicy, tuple[float, ...]]:
    phase_key = str(phase).strip().lower()
    if phase_key == "polymer":
        if bool(charged):
            attempts = 5 if max_attempts is None else max(1, int(max_attempts))
            factor = 0.86 if backoff_factor is None else float(backoff_factor)
            floor = 0.30 if floor_density_g_cm3 is None else float(floor_density_g_cm3)
        else:
            attempts = 4 if max_attempts is None else max(1, int(max_attempts))
            factor = 0.88 if backoff_factor is None else float(backoff_factor)
            floor = 0.40 if floor_density_g_cm3 is None else float(floor_density_g_cm3)
    else:
        attempts = 3 if max_attempts is None else max(1, int(max_attempts))
        factor = 0.90 if backoff_factor is None else float(backoff_factor)
        floor = 0.60 if floor_density_g_cm3 is None else float(floor_density_g_cm3)

    density0 = initial_bulk_pack_density(
        target_density_g_cm3=float(target_density_g_cm3),
        phase=phase_key,
        requested_density_g_cm3=requested_density_g_cm3,
        z_scale=z_scale,
        charged=bool(charged),
    )
    densities: list[float] = []
    current = float(density0)
    for _ in range(attempts):
        current = max(float(floor), float(current))
        rounded = round(float(current), 6)
        if not densities or rounded != round(float(densities[-1]), 6):
            densities.append(float(current))
        current *= float(factor)
    policy = PackBackoffPolicy(
        phase=phase_key,
        charged=bool(charged),
        initial_density_g_cm3=float(density0),
        max_attempts=int(attempts),
        backoff_factor=float(factor),
        floor_density_g_cm3=float(floor),
    )
    return policy, tuple(float(x) for x in densities)


def run_amorphous_cell_with_density_backoff(
    *,
    label: str,
    pack_fn: Callable[..., object],
    mols: Sequence[object],
    counts: Sequence[int],
    charge_scale: Sequence[float],
    phase: str,
    target_density_g_cm3: float,
    work_dir: Path | str,
    retry: int,
    retry_step: int,
    threshold: float,
    dec_rate: float,
    neutralize: bool = False,
    requested_density_g_cm3: float | None = None,
    z_scale: float | None = None,
    charged: bool = False,
    max_attempts: int | None = None,
    backoff_factor: float | None = None,
    floor_density_g_cm3: float | None = None,
    summary_name: str = "pack_backoff_summary.json",
) -> PackBackoffResult:
    work_root = Path(work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    policy, densities = build_pack_density_ladder(
        phase=phase,
        target_density_g_cm3=float(target_density_g_cm3),
        requested_density_g_cm3=requested_density_g_cm3,
        z_scale=z_scale,
        charged=bool(charged),
        max_attempts=max_attempts,
        backoff_factor=backoff_factor,
        floor_density_g_cm3=floor_density_g_cm3,
    )
    attempts: list[dict[str, Any]] = []
    summary_path = work_root / summary_name
    last_error: Exception | None = None
    selected_cell = None
    selected_density = None
    selected_attempt_index = None

    for attempt_index, density in enumerate(densities):
        attempt_dir = work_root / f"attempt_{attempt_index:02d}"
        attempt_record: dict[str, Any] = {
            "attempt_index": int(attempt_index),
            "density_g_cm3": float(density),
            "work_dir": str(attempt_dir),
        }
        try:
            cell = pack_fn(
                list(mols),
                list(counts),
                density=float(density),
                neutralize=bool(neutralize),
                charge_scale=list(charge_scale),
                work_dir=attempt_dir,
                retry=int(retry),
                retry_step=int(retry_step),
                threshold=float(threshold),
                dec_rate=float(dec_rate),
            )
            attempt_record["success"] = True
            selected_cell = cell
            selected_density = float(density)
            selected_attempt_index = int(attempt_index)
            attempts.append(attempt_record)
            break
        except Exception as exc:
            attempt_record["success"] = False
            attempt_record["error"] = repr(exc)
            attempts.append(attempt_record)
            last_error = exc

    summary = {
        "label": str(label),
        "phase": str(phase),
        "target_density_g_cm3": float(target_density_g_cm3),
        "policy": asdict(policy),
        "attempts": attempts,
        "success": selected_cell is not None,
        "selected_density_g_cm3": selected_density,
        "selected_attempt_index": selected_attempt_index,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if selected_cell is None or selected_density is None or selected_attempt_index is None:
        raise RuntimeError(
            f"Failed to pack {label} after {len(attempts)} density-backoff attempts. "
            f"See {summary_path}."
        ) from last_error

    return PackBackoffResult(
        cell=selected_cell,
        selected_density_g_cm3=float(selected_density),
        selected_attempt_index=int(selected_attempt_index),
        attempts=tuple(attempts),
        summary=summary,
        summary_path=summary_path,
    )
