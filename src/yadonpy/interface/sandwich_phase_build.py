from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


_AVOGADRO = 6.02214076e23


@dataclass(frozen=True)
class BulkCalibrationSummary:
    label: str
    phase_preparation_mode: str
    master_xy_nm: tuple[float, float]
    bulk_reference_box_nm: tuple[float, float, float]
    target_density_g_cm3: float
    total_mass_amu: float
    target_z_nm: float
    initial_walled_pack_density_g_cm3: float
    selected_bulk_pack_density_g_cm3: float
    charged_phase: bool
    notes: tuple[str, ...] = ()


def solve_phase_target_z_nm(
    *,
    total_mass_amu: float,
    target_density_g_cm3: float,
    target_xy_nm: tuple[float, float],
    min_z_nm: float | None = None,
) -> float:
    area_nm2 = float(target_xy_nm[0]) * float(target_xy_nm[1])
    if area_nm2 <= 0.0 or float(target_density_g_cm3) <= 0.0:
        return 0.0 if min_z_nm is None else max(0.0, float(min_z_nm))
    mass_g = float(total_mass_amu) / _AVOGADRO
    volume_cm3 = mass_g / float(target_density_g_cm3)
    z_nm = float(volume_cm3 / (area_nm2 * 1.0e-21))
    if min_z_nm is not None:
        z_nm = max(float(min_z_nm), float(z_nm))
    return float(z_nm)


def recommend_initial_walled_pack_density(
    *,
    phase: str,
    target_density_g_cm3: float,
    selected_bulk_pack_density_g_cm3: float,
) -> float:
    phase_key = str(phase).strip().lower()
    target = float(target_density_g_cm3)
    bulk_density = float(selected_bulk_pack_density_g_cm3)
    if phase_key == "polymer":
        # Keep the direct final-XY polymer build deliberately loose.
        # The confined NPT stage is a better place to densify than the initial
        # pack, especially for charged CMC-like phases that otherwise enter EM
        # with severe overlaps.
        return float(max(0.30, min(0.74 * target, max(0.75 * bulk_density, 0.34))))
    return float(max(0.65, min(0.92 * target, max(0.85 * bulk_density, 0.72))))


def write_bulk_calibration_summary(summary: BulkCalibrationSummary, path: Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
