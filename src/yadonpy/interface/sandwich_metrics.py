from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.molspec import molecular_weight
from .sandwich_specs import SandwichPhaseReport


_AVOGADRO = 6.02214076e23


def _read_gro_z_coords(gro_path: Path) -> list[float]:
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid .gro file: {gro_path}")
    nat = int(lines[1].strip())
    z: list[float] = []
    for i in range(nat):
        raw = lines[2 + i]
        try:
            z.append(float(raw[36:44]))
        except Exception:
            z.append(float(raw[-8:]))
    return z


def _read_gro_box_nm(gro_path: Path) -> tuple[float, float, float]:
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid .gro file: {gro_path}")
    parts = lines[-1].split()
    if len(parts) < 3:
        raise ValueError(f"Invalid .gro box line: {gro_path}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _unwrap_phase_z(values: Sequence[float], *, box_z_nm: float) -> list[float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or box_z_nm <= 0.0:
        return [float(x) for x in arr]
    if float(np.max(arr) - np.min(arr)) <= 0.5 * float(box_z_nm):
        return [float(x) for x in arr]
    ordered = np.sort(arr)
    cyclic = np.concatenate([ordered, ordered[:1] + float(box_z_nm)])
    gaps = np.diff(cyclic)
    split = int(np.argmax(gaps))
    if float(gaps[split]) <= 0.5 * float(box_z_nm):
        return [float(x) for x in arr]
    threshold = float(ordered[split])
    arr = np.where(arr <= threshold, arr + float(box_z_nm), arr)
    return [float(x) for x in arr]


def build_stack_checks(*, gro_path: Path, ndx_groups: dict[str, list[int]]) -> dict[str, object]:
    z_coords = _read_gro_z_coords(gro_path)
    _box_x_nm, _box_y_nm, box_z_nm = _read_gro_box_nm(gro_path)
    payload: dict[str, object] = {"gro_path": str(gro_path)}
    phase_stats: dict[str, dict[str, float]] = {}
    for name in ("GRAPHITE", "POLYMER", "ELECTROLYTE"):
        members = [int(idx) for idx in ndx_groups.get(name, []) if 1 <= int(idx) <= len(z_coords)]
        if not members:
            continue
        values = _unwrap_phase_z([float(z_coords[idx - 1]) for idx in members], box_z_nm=float(box_z_nm))
        ordered = sorted(values)
        p05 = float(np.percentile(ordered, 5.0))
        p95 = float(np.percentile(ordered, 95.0))
        phase_stats[name] = {
            "min_z_nm": min(values),
            "mean_z_nm": sum(values) / float(len(values)),
            "max_z_nm": max(values),
            "p05_z_nm": p05,
            "p95_z_nm": p95,
        }
    payload["phases"] = phase_stats
    if len(phase_stats) == 3:
        observed = [
            name
            for name, _mean in sorted(
                ((name, data["mean_z_nm"]) for name, data in phase_stats.items()),
                key=lambda item: item[1],
            )
        ]
        payload["observed_order"] = observed
        payload["expected_order"] = ["GRAPHITE", "POLYMER", "ELECTROLYTE"]
        payload["is_expected_order"] = observed == ["GRAPHITE", "POLYMER", "ELECTROLYTE"]
        payload["graphite_polymer_gap_nm"] = float(phase_stats["POLYMER"]["min_z_nm"] - phase_stats["GRAPHITE"]["max_z_nm"])
        payload["polymer_electrolyte_gap_nm"] = float(phase_stats["ELECTROLYTE"]["min_z_nm"] - phase_stats["POLYMER"]["max_z_nm"])
        payload["graphite_polymer_core_gap_nm"] = float(phase_stats["POLYMER"]["p05_z_nm"] - phase_stats["GRAPHITE"]["p95_z_nm"])
        payload["polymer_electrolyte_core_gap_nm"] = float(phase_stats["ELECTROLYTE"]["p05_z_nm"] - phase_stats["POLYMER"]["p95_z_nm"])
    return payload


def phase_local_density_summary(*, gro_path: Path, species: Sequence, counts: Sequence[int]) -> dict[str, object]:
    coords = np.asarray(_read_gro_z_coords(gro_path), dtype=float)
    if coords.size == 0:
        return {
            "box_nm": list(_read_gro_box_nm(gro_path)),
            "occupied_thickness_nm": 0.0,
            "occupied_density_g_cm3": 0.0,
            "center_window_nm": 0.0,
            "center_bulk_like_density_g_cm3": 0.0,
            "wrapped_across_z_boundary": False,
        }

    box_nm = _read_gro_box_nm(gro_path)
    box_z_nm = max(float(box_nm[2]), 1.0e-9)
    ordered = np.sort(coords)
    wrapped = bool(float(np.max(coords) - np.min(coords)) > 0.5 * box_z_nm)
    if wrapped:
        coords = np.asarray(_unwrap_phase_z(coords.tolist(), box_z_nm=box_z_nm), dtype=float)
        ordered = np.sort(coords)

    total_atoms = int(sum(int(mol.GetNumAtoms()) * int(count) for mol, count in zip(species, counts)))
    if total_atoms <= 0:
        return {
            "box_nm": list(box_nm),
            "occupied_thickness_nm": 0.0,
            "occupied_density_g_cm3": 0.0,
            "center_window_nm": 0.0,
            "center_bulk_like_density_g_cm3": 0.0,
            "wrapped_across_z_boundary": wrapped,
        }

    species_masses = [float(molecular_weight(mol, strict=True)) for mol in species]
    atom_masses: list[float] = []
    for mol, count, mass_amu in zip(species, counts, species_masses):
        nat = max(int(mol.GetNumAtoms()), 1)
        per_atom_mass = float(mass_amu) / float(nat)
        for _ in range(int(count)):
            atom_masses.extend([per_atom_mass] * nat)
    if len(atom_masses) != len(coords):
        atom_masses = [float(sum(species_masses)) / float(max(len(coords), 1))] * len(coords)
    masses = np.asarray(atom_masses, dtype=float)

    occupied_min = float(np.min(coords))
    occupied_max = float(np.max(coords))
    occupied_thickness_nm = max(0.0, occupied_max - occupied_min)
    occupied_volume_cm3 = float(box_nm[0] * box_nm[1] * occupied_thickness_nm) * 1.0e-21
    total_mass_amu = float(np.sum(masses))
    occupied_density = 0.0 if occupied_volume_cm3 <= 0.0 else float(total_mass_amu / _AVOGADRO / occupied_volume_cm3)

    if occupied_thickness_nm <= 1.0e-9:
        return {
            "box_nm": list(box_nm),
            "occupied_thickness_nm": occupied_thickness_nm,
            "occupied_density_g_cm3": occupied_density,
            "center_window_nm": 0.0,
            "center_bulk_like_density_g_cm3": 0.0,
            "wrapped_across_z_boundary": wrapped,
        }

    center_window_nm = min(max(0.35 * occupied_thickness_nm, 0.40), occupied_thickness_nm)
    center_mid = 0.5 * (occupied_min + occupied_max)
    center_lo = center_mid - 0.5 * center_window_nm
    center_hi = center_mid + 0.5 * center_window_nm
    center_mask = (coords >= center_lo) & (coords <= center_hi)
    center_mass_amu = float(np.sum(masses[center_mask]))
    center_volume_cm3 = float(box_nm[0] * box_nm[1] * center_window_nm) * 1.0e-21
    center_density = 0.0 if center_volume_cm3 <= 0.0 else float(center_mass_amu / _AVOGADRO / center_volume_cm3)

    return {
        "box_nm": list(box_nm),
        "occupied_thickness_nm": occupied_thickness_nm,
        "occupied_density_g_cm3": occupied_density,
        "center_window_nm": center_window_nm,
        "center_bulk_like_density_g_cm3": center_density,
        "wrapped_across_z_boundary": wrapped,
    }


def representative_phase_density(summary: dict[str, object]) -> float:
    try:
        center_density = float(summary.get("center_bulk_like_density_g_cm3", 0.0))
    except Exception:
        center_density = 0.0
    if center_density > 0.0:
        return center_density
    try:
        return float(summary.get("occupied_density_g_cm3", 0.0))
    except Exception:
        return 0.0


def phase_gap_penalty_nm(summary: dict[str, object], *, target_density_g_cm3: float | None) -> float:
    occupied_thickness = float(summary.get("occupied_thickness_nm", 0.0) or 0.0)
    if occupied_thickness <= 0.0:
        return 0.0
    occupied_density = representative_phase_density(summary)
    if target_density_g_cm3 is not None and float(target_density_g_cm3) > 0.0:
        if occupied_density < 0.85 * float(target_density_g_cm3):
            return 0.12 * float(occupied_thickness)
    return 0.05 * float(occupied_thickness)


def confined_summary_score(*, summary: dict[str, object], target_density_g_cm3: float, target_thickness_nm: float) -> float:
    try:
        center_density = float(summary.get("center_bulk_like_density_g_cm3", 0.0))
    except Exception:
        center_density = 0.0
    try:
        occupied_density = float(summary.get("occupied_density_g_cm3", 0.0))
    except Exception:
        occupied_density = 0.0
    occupied_thickness = float(summary.get("occupied_thickness_nm", target_thickness_nm))
    density_ref = max(float(target_density_g_cm3), 1.0e-6)

    score = abs(center_density - float(target_density_g_cm3)) / density_ref
    score += abs(occupied_thickness - float(target_thickness_nm)) / max(float(target_thickness_nm), 1.0e-6)
    if occupied_density < 0.80 * float(target_density_g_cm3):
        score += 0.50
    if bool(summary.get("wrapped_across_z_boundary", False)):
        score += 1.00
    return float(score)


def needs_confined_rescue(*, summary: dict[str, object], target_density_g_cm3: float, target_thickness_nm: float) -> bool:
    try:
        center_density = float(summary.get("center_bulk_like_density_g_cm3", 0.0))
    except Exception:
        center_density = 0.0
    occupied_thickness = float(summary.get("occupied_thickness_nm", target_thickness_nm))
    if bool(summary.get("wrapped_across_z_boundary", False)):
        return True
    if occupied_thickness > 1.08 * float(target_thickness_nm):
        return True
    return center_density < 0.85 * float(target_density_g_cm3)


def confined_phase_report(
    *,
    label: str,
    species_names: Sequence[str],
    counts: Sequence[int],
    target_density_g_cm3: float | None,
    summary: dict[str, object],
) -> SandwichPhaseReport:
    confined_box = tuple(float(x) for x in summary.get("box_nm", (0.0, 0.0, 0.0)))
    occupied_thickness = float(summary.get("occupied_thickness_nm", confined_box[2]))
    occupied_box = (float(confined_box[0]), float(confined_box[1]), float(occupied_thickness))
    occupied_density = float(summary.get("occupied_density_g_cm3", 0.0))
    bulk_like_density = float(summary.get("center_bulk_like_density_g_cm3", occupied_density))
    return SandwichPhaseReport(
        label=str(label),
        box_nm=occupied_box,
        density_g_cm3=float(bulk_like_density if bulk_like_density > 0.0 else occupied_density),
        species_names=tuple(str(x) for x in species_names),
        counts=tuple(int(x) for x in counts),
        target_density_g_cm3=(None if target_density_g_cm3 is None else float(target_density_g_cm3)),
        occupied_density_g_cm3=float(occupied_density),
        bulk_like_density_g_cm3=float(bulk_like_density if bulk_like_density > 0.0 else occupied_density),
    )
