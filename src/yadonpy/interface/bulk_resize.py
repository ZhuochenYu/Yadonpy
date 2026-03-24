from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..sim.preset.eq import _find_latest_equilibrated_gro

_AVOGADRO = 6.02214076e23


@dataclass(frozen=True)
class BulkEquilibriumProfile:
    gro_path: Path
    box_nm: tuple[float, float, float]
    volume_nm3: float
    counts: tuple[int, ...]
    mol_weights: tuple[float, ...]
    species_names: tuple[str, ...]
    density_g_cm3: float
    total_mass_amu: float


@dataclass(frozen=True)
class BulkRescalePlan:
    probe_box_nm: tuple[float, float, float]
    target_box_nm: tuple[float, float, float]
    volume_scale: float
    raw_counts: tuple[float, ...]
    target_counts: tuple[int, ...]
    species_names: tuple[str, ...]
    probe_density_g_cm3: float
    target_density_g_cm3: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ElectrolyteAlignmentPlan:
    target_z_margin_nm: float
    target_z_nm: float
    fixed_xy_npt_ns: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DirectElectrolytePlan:
    target_box_nm: tuple[float, float, float]
    target_volume_nm3: float
    target_density_g_cm3: float
    target_counts: tuple[int, ...]
    species_names: tuple[str, ...]
    solvent_counts: tuple[int, ...]
    salt_pair_count: int
    estimated_density_g_cm3: float
    estimated_salt_molarity_M: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixedXYDirectPackPlan:
    reference_box_nm: tuple[float, float, float]
    initial_pack_box_nm: tuple[float, float, float]
    fixed_xy_nm: tuple[float, float]
    target_counts: tuple[int, ...]
    species_names: tuple[str, ...]
    initial_pack_density_g_cm3: float
    estimated_initial_density_g_cm3: float
    notes: tuple[str, ...] = ()


def recommend_fixed_xy_pack_parameters(
    *,
    target_density_g_cm3: float,
) -> tuple[float, float, float, float]:
    """Recommend a compact but still packable fixed-XY starting box.

    The fixed-XY electrolyte rebuild should start slightly looser than the
    target density, but not from a dramatically dilute slab-like box. The
    returned tuple is:

      (initial_pack_density_g_cm3, z_padding_factor, minimum_pack_z_factor, maximum_pack_z_factor)
    """
    density = float(target_density_g_cm3)
    if density <= 0.0:
        return (0.75, 1.15, 1.15, 1.45)

    pack_density = max(0.65, min(0.92, density * 0.82))
    z_padding_factor = 1.12 if density >= 0.9 else 1.15
    density_ratio = max(density / max(pack_density, 1.0e-12), 1.0)
    minimum_pack_z_factor = max(1.12, min(1.28, density_ratio * 1.02))
    maximum_pack_z_factor = max(minimum_pack_z_factor + 0.12, min(1.55, density_ratio * 1.20))
    return (
        float(pack_density),
        float(z_padding_factor),
        float(minimum_pack_z_factor),
        float(maximum_pack_z_factor),
    )


def _largest_remainder_allocate(raw_counts: np.ndarray, *, target_total: int, min_counts: np.ndarray) -> np.ndarray:
    if raw_counts.ndim != 1:
        raise ValueError("raw_counts must be a 1D array")
    out = np.maximum(np.floor(raw_counts).astype(int), min_counts.astype(int))
    target_total = int(max(target_total, int(min_counts.sum())))
    diff = int(target_total - int(out.sum()))
    if diff > 0:
        frac = raw_counts - np.floor(raw_counts)
        order = np.argsort(-frac, kind="mergesort")
        cursor = 0
        while diff > 0 and order.size > 0:
            out[int(order[cursor % order.size])] += 1
            diff -= 1
            cursor += 1
    elif diff < 0:
        frac = raw_counts - np.floor(raw_counts)
        order = np.argsort(frac, kind="mergesort")
        cursor = 0
        guard = 0
        while diff < 0 and order.size > 0 and guard < 100000:
            idx = int(order[cursor % order.size])
            if out[idx] > int(min_counts[idx]):
                out[idx] -= 1
                diff += 1
            cursor += 1
            guard += 1
    return out


def _normalize_ints(values: Iterable[int], *, label: str) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if not out:
        raise ValueError(f"{label} must not be empty")
    return out


def _normalize_floats(values: Iterable[float], *, label: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not out:
        raise ValueError(f"{label} must not be empty")
    return out


def _normalize_index_groups(
    *,
    size: int,
    indices: Sequence[int] | None,
    groups: Sequence[Sequence[int]] | None,
    label: str,
    required: bool = False,
) -> tuple[tuple[int, ...], ...]:
    source = groups if groups is not None else ((indices,) if indices is not None else ())
    normalized: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for group in source:
        if isinstance(group, (list, tuple, np.ndarray)):
            group_t = tuple(int(idx) for idx in group)
        else:
            group_t = (int(group),)
        if not group_t:
            continue
        local_seen: set[int] = set()
        for idx in group_t:
            if idx < 0 or idx >= int(size):
                raise ValueError(f"{label} index {idx} is out of range for size {size}")
            if idx in local_seen:
                raise ValueError(f"{label} group {group_t} contains duplicate indices")
            if idx in seen:
                raise ValueError(f"{label} indices must not overlap across groups")
            local_seen.add(idx)
            seen.add(idx)
        normalized.append(group_t)
    if required and not normalized:
        raise ValueError(f"{label} must not be empty")
    return tuple(normalized)


def _normalize_group_min_counts(
    *,
    groups: tuple[tuple[int, ...], ...],
    probe_counts: np.ndarray,
    legacy_min_counts: Sequence[int] | None,
    grouped_min_counts: Sequence[Sequence[int]] | None,
) -> tuple[np.ndarray, ...]:
    if grouped_min_counts is not None:
        if len(grouped_min_counts) != len(groups):
            raise ValueError("min_solvent_group_counts must match the number of solvent groups")
        out: list[np.ndarray] = []
        for group, mins in zip(groups, grouped_min_counts):
            arr = np.asarray([int(v) for v in mins], dtype=int)
            if arr.shape[0] != len(group):
                raise ValueError("Each min_solvent_group_counts entry must match its solvent group length")
            out.append(arr)
        return tuple(out)

    if legacy_min_counts is not None:
        flat = tuple(int(v) for v in legacy_min_counts)
        if len(groups) == 1:
            if len(flat) != len(groups[0]):
                raise ValueError("min_solvent_counts and solvent_indices must have the same length")
            return (np.asarray(flat, dtype=int),)
        total_size = sum(len(group) for group in groups)
        if len(flat) != total_size:
            raise ValueError("For multiple solvent groups, min_solvent_counts must cover all grouped solvent indices in order")
        out = []
        cursor = 0
        for group in groups:
            out.append(np.asarray(flat[cursor:cursor + len(group)], dtype=int))
            cursor += len(group)
        return tuple(out)

    return tuple(
        np.asarray([1 if int(probe_counts[idx]) > 0 else 0 for idx in group], dtype=int)
        for group in groups
    )


def _normalize_group_scalar_mins(*, groups: tuple[tuple[int, ...], ...], minima: int | Sequence[int], label: str) -> tuple[int, ...]:
    if isinstance(minima, Sequence) and not isinstance(minima, (str, bytes)):
        out = tuple(int(v) for v in minima)
        if len(out) != len(groups):
            raise ValueError(f"{label} must match the number of groups")
        return out
    return tuple(int(minima) for _ in groups)


def _resolve_gro_path(*, work_dir: Path | None = None, gro_path: Path | None = None) -> Path:
    if gro_path is not None:
        path = Path(gro_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if work_dir is None:
        raise ValueError("Either work_dir or gro_path must be provided.")
    latest = _find_latest_equilibrated_gro(Path(work_dir).expanduser().resolve())
    if latest is None or not latest.exists():
        raise FileNotFoundError(f"Cannot locate an equilibrated GRO under {work_dir}")
    return latest


def read_equilibrated_box_nm(*, work_dir: Path | None = None, gro_path: Path | None = None) -> tuple[float, float, float]:
    path = _resolve_gro_path(work_dir=work_dir, gro_path=gro_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {path}")
    fields = lines[-1].split()
    if len(fields) < 3:
        raise ValueError(f"Invalid GRO box line: {path}")
    return float(fields[0]), float(fields[1]), float(fields[2])


def build_bulk_equilibrium_profile(
    *,
    counts: Sequence[int],
    mol_weights: Sequence[float],
    species_names: Sequence[str] | None = None,
    work_dir: Path | None = None,
    gro_path: Path | None = None,
) -> BulkEquilibriumProfile:
    counts_t = _normalize_ints(counts, label="counts")
    mw_t = _normalize_floats(mol_weights, label="mol_weights")
    if len(counts_t) != len(mw_t):
        raise ValueError("counts and mol_weights must have the same length")
    if species_names is None:
        names_t = tuple(f"species_{idx}" for idx in range(len(counts_t)))
    else:
        names_t = tuple(str(name) for name in species_names)
        if len(names_t) != len(counts_t):
            raise ValueError("species_names and counts must have the same length")

    resolved_gro = _resolve_gro_path(work_dir=work_dir, gro_path=gro_path)
    box_nm = read_equilibrated_box_nm(gro_path=resolved_gro)
    volume_nm3 = float(box_nm[0] * box_nm[1] * box_nm[2])
    total_mass_amu = float(sum(float(nc) * float(mw) for nc, mw in zip(counts_t, mw_t)))
    density = 0.0
    if volume_nm3 > 0.0 and total_mass_amu > 0.0:
        density = float((total_mass_amu / _AVOGADRO) / (volume_nm3 * 1.0e-21))
    return BulkEquilibriumProfile(
        gro_path=resolved_gro,
        box_nm=tuple(float(x) for x in box_nm),
        volume_nm3=volume_nm3,
        counts=counts_t,
        mol_weights=mw_t,
        species_names=names_t,
        density_g_cm3=density,
        total_mass_amu=total_mass_amu,
    )


def plan_rescaled_bulk_counts(
    *,
    profile: BulkEquilibriumProfile,
    target_xy_nm: tuple[float, float],
    target_z_nm: float,
    min_counts: Sequence[int] | None = None,
    tied_groups: Sequence[Sequence[int]] | None = None,
    keep_nonzero_species: bool = True,
) -> BulkRescalePlan:
    target_box_nm = (float(target_xy_nm[0]), float(target_xy_nm[1]), float(target_z_nm))
    if min(target_box_nm) <= 0.0:
        raise ValueError("Target box lengths must be positive.")
    probe_counts = np.asarray(profile.counts, dtype=int)
    mol_weights = np.asarray(profile.mol_weights, dtype=float)
    probe_volume = float(max(profile.volume_nm3, 1.0e-12))
    target_volume = float(target_box_nm[0] * target_box_nm[1] * target_box_nm[2])
    scale = float(target_volume / probe_volume)
    raw_counts = probe_counts.astype(float) * scale
    rounded = np.rint(raw_counts).astype(int)

    if keep_nonzero_species:
        rounded = np.where(probe_counts > 0, np.maximum(rounded, 1), rounded)

    if min_counts is not None:
        min_arr = np.asarray([int(v) for v in min_counts], dtype=int)
        if min_arr.shape[0] != rounded.shape[0]:
            raise ValueError("min_counts and profile.counts must have the same length")
        rounded = np.maximum(rounded, min_arr)
    else:
        min_arr = np.zeros_like(rounded)

    if tied_groups:
        for group in tied_groups:
            idxs = [int(idx) for idx in group]
            if not idxs:
                continue
            tie_target = max(int(rounded[idx]) for idx in idxs)
            tie_target = max(tie_target, max(int(min_arr[idx]) for idx in idxs))
            for idx in idxs:
                rounded[idx] = tie_target

    target_mass_amu = float(np.dot(rounded.astype(float), mol_weights))
    target_density = 0.0
    if target_volume > 0.0 and target_mass_amu > 0.0:
        target_density = float((target_mass_amu / _AVOGADRO) / (target_volume * 1.0e-21))

    notes: list[str] = []
    if scale < 1.0:
        notes.append(f"reduced bulk volume by scale factor {scale:.4f} to match target interface footprint")
    elif scale > 1.0:
        notes.append(f"expanded bulk volume by scale factor {scale:.4f} to match target interface footprint")
    density_delta = abs(float(target_density) - float(profile.density_g_cm3))
    if density_delta > 0.05:
        notes.append(
            f"rounded molecule counts shift estimated density from {profile.density_g_cm3:.4f} to {target_density:.4f} g/cm^3"
        )

    return BulkRescalePlan(
        probe_box_nm=tuple(float(x) for x in profile.box_nm),
        target_box_nm=target_box_nm,
        volume_scale=scale,
        raw_counts=tuple(float(x) for x in raw_counts.tolist()),
        target_counts=tuple(int(x) for x in rounded.tolist()),
        species_names=tuple(profile.species_names),
        probe_density_g_cm3=float(profile.density_g_cm3),
        target_density_g_cm3=float(target_density),
        notes=tuple(notes),
    )


def plan_resized_electrolyte_counts(
    *,
    profile: BulkEquilibriumProfile,
    target_xy_nm: tuple[float, float],
    target_z_nm: float,
    solvent_indices: Sequence[int],
    solvent_groups: Sequence[Sequence[int]] | None = None,
    salt_pair_indices: Sequence[int] | None = None,
    salt_pair_groups: Sequence[Sequence[int]] | None = None,
    min_solvent_counts: Sequence[int] | None = None,
    min_solvent_group_counts: Sequence[Sequence[int]] | None = None,
    min_salt_pairs: int | Sequence[int] = 1,
) -> BulkRescalePlan:
    target_box_nm = (float(target_xy_nm[0]), float(target_xy_nm[1]), float(target_z_nm))
    if min(target_box_nm) <= 0.0:
        raise ValueError("Target box lengths must be positive.")

    probe_counts = np.asarray(profile.counts, dtype=int)
    mol_weights = np.asarray(profile.mol_weights, dtype=float)
    raw_counts = probe_counts.astype(float) * float(target_box_nm[0] * target_box_nm[1] * target_box_nm[2] / max(profile.volume_nm3, 1.0e-12))
    rounded = np.zeros_like(probe_counts)

    solvent_group_list = _normalize_index_groups(
        size=probe_counts.shape[0],
        indices=solvent_indices,
        groups=solvent_groups,
        label="solvent groups",
        required=True,
    )
    solvent_group_mins = _normalize_group_min_counts(
        groups=solvent_group_list,
        probe_counts=probe_counts,
        legacy_min_counts=min_solvent_counts,
        grouped_min_counts=min_solvent_group_counts,
    )
    for group, group_min in zip(solvent_group_list, solvent_group_mins):
        solvent_idx = np.asarray(group, dtype=int)
        solvent_raw = raw_counts[solvent_idx]
        solvent_target_total = int(max(int(round(float(solvent_raw.sum()))), int(group_min.sum())))
        rounded[solvent_idx] = _largest_remainder_allocate(solvent_raw, target_total=solvent_target_total, min_counts=group_min)

    salt_group_list = _normalize_index_groups(
        size=probe_counts.shape[0],
        indices=salt_pair_indices,
        groups=salt_pair_groups,
        label="salt pair groups",
        required=False,
    )
    salt_pair_targets: list[int] = []
    salt_pair_mins = _normalize_group_scalar_mins(groups=salt_group_list, minima=min_salt_pairs, label="min_salt_pairs")
    for group, group_min in zip(salt_group_list, salt_pair_mins):
        salt_idx = np.asarray(group, dtype=int)
        salt_pair_target = max(int(group_min), int(round(float(raw_counts[salt_idx].mean()))))
        salt_pair_targets.append(int(salt_pair_target))
        for idx in salt_idx:
            rounded[int(idx)] = int(salt_pair_target)

    assigned = {
        int(idx)
        for group in solvent_group_list + salt_group_list
        for idx in group
    }
    for idx in range(probe_counts.shape[0]):
        if idx in assigned:
            continue
        rounded[idx] = max(int(round(float(raw_counts[idx]))), 1 if probe_counts[idx] > 0 else 0)

    target_volume = float(target_box_nm[0] * target_box_nm[1] * target_box_nm[2])
    target_mass_amu = float(np.dot(rounded.astype(float), mol_weights))
    target_density = 0.0
    if target_volume > 0.0 and target_mass_amu > 0.0:
        target_density = float((target_mass_amu / _AVOGADRO) / (target_volume * 1.0e-21))

    scale = float(target_volume / max(profile.volume_nm3, 1.0e-12))
    notes = [
        f"rescaled electrolyte volume by factor {scale:.4f} using probe-equilibrated box and composition",
    ]
    if len(solvent_group_list) == 1:
        notes.append(f"preserved solvent composition across indices {solvent_group_list[0]}")
    else:
        notes.append(f"preserved grouped solvent composition across index sets {solvent_group_list}")
    if salt_group_list:
        if len(salt_group_list) == 1:
            notes.append(
                f"preserved salt-pair concentration across coupled indices {salt_group_list[0]}; target_pairs={salt_pair_targets[0]}"
            )
        else:
            notes.append(
                "preserved salt-pair concentration across coupled index sets "
                f"{salt_group_list}; target_pairs={tuple(int(x) for x in salt_pair_targets)}"
            )
    density_delta = abs(float(target_density) - float(profile.density_g_cm3))
    if density_delta > 0.05:
        notes.append(
            f"rounded grouped counts shift estimated density from {profile.density_g_cm3:.4f} to {target_density:.4f} g/cm^3"
        )

    return BulkRescalePlan(
        probe_box_nm=tuple(float(x) for x in profile.box_nm),
        target_box_nm=target_box_nm,
        volume_scale=scale,
        raw_counts=tuple(float(x) for x in raw_counts.tolist()),
        target_counts=tuple(int(x) for x in rounded.tolist()),
        species_names=tuple(profile.species_names),
        probe_density_g_cm3=float(profile.density_g_cm3),
        target_density_g_cm3=float(target_density),
        notes=tuple(notes),
    )


def recommend_electrolyte_alignment(
    *,
    top_thickness_nm: float,
    gap_nm: float,
    surface_shell_nm: float,
    is_polyelectrolyte: bool = False,
    minimum_margin_nm: float = 1.0,
    fixed_xy_npt_ns: float | None = None,
) -> ElectrolyteAlignmentPlan:
    top_thickness = float(top_thickness_nm)
    gap = float(gap_nm)
    surface_shell = float(surface_shell_nm)
    if top_thickness <= 0.0 or gap < 0.0 or surface_shell < 0.0 or minimum_margin_nm <= 0.0:
        raise ValueError("top_thickness_nm must be positive and gap/surface_shell/minimum_margin_nm must be non-negative")

    target_z_margin = max(float(minimum_margin_nm), surface_shell + 0.5 * gap)
    relax_ns = float(fixed_xy_npt_ns) if fixed_xy_npt_ns is not None else (2.5 if is_polyelectrolyte else 1.5)
    notes = [
        f"target_z_margin_nm={target_z_margin:.3f} derived from max(minimum_margin_nm={float(minimum_margin_nm):.3f}, surface_shell_nm + 0.5*gap_nm={surface_shell + 0.5 * gap:.3f})",
        f"fixed_xy_npt_ns={relax_ns:.3f} selected for {'polyelectrolyte' if is_polyelectrolyte else 'neutral polymer'} interface alignment",
    ]
    return ElectrolyteAlignmentPlan(
        target_z_margin_nm=float(target_z_margin),
        target_z_nm=float(top_thickness + target_z_margin),
        fixed_xy_npt_ns=relax_ns,
        notes=tuple(notes),
    )


def plan_direct_electrolyte_counts(
    *,
    target_box_nm: tuple[float, float, float],
    target_density_g_cm3: float,
    solvent_mol_weights: Sequence[float],
    solvent_mass_ratio: Sequence[float],
    salt_mol_weights: Sequence[float],
    salt_molarity_M: float,
    min_salt_pairs: int = 1,
    solvent_species_names: Sequence[str] | None = None,
    salt_species_names: Sequence[str] | None = None,
    min_solvent_counts: Sequence[int] | None = None,
) -> DirectElectrolytePlan:
    box_nm = tuple(float(x) for x in target_box_nm)
    if len(box_nm) != 3 or min(box_nm) <= 0.0:
        raise ValueError("target_box_nm must contain three positive lengths")

    solvent_mw = _normalize_floats(solvent_mol_weights, label="solvent_mol_weights")
    mass_ratio = _normalize_floats(solvent_mass_ratio, label="solvent_mass_ratio")
    salt_mw = _normalize_floats(salt_mol_weights, label="salt_mol_weights")
    if len(solvent_mw) != len(mass_ratio):
        raise ValueError("solvent_mol_weights and solvent_mass_ratio must have the same length")
    if len(salt_mw) != 2:
        raise ValueError("salt_mol_weights must contain exactly two entries for the paired salt species")

    if solvent_species_names is None:
        solvent_names = tuple(f"solvent_{idx}" for idx in range(len(solvent_mw)))
    else:
        solvent_names = tuple(str(name) for name in solvent_species_names)
        if len(solvent_names) != len(solvent_mw):
            raise ValueError("solvent_species_names and solvent_mol_weights must have the same length")

    if salt_species_names is None:
        salt_names = ("salt_cation", "salt_anion")
    else:
        salt_names = tuple(str(name) for name in salt_species_names)
        if len(salt_names) != 2:
            raise ValueError("salt_species_names must contain exactly two entries")

    if min_solvent_counts is None:
        solvent_min = np.ones(len(solvent_mw), dtype=int)
    else:
        solvent_min = np.asarray([int(v) for v in min_solvent_counts], dtype=int)
        if solvent_min.shape[0] != len(solvent_mw):
            raise ValueError("min_solvent_counts and solvent_mol_weights must have the same length")

    volume_nm3 = float(box_nm[0] * box_nm[1] * box_nm[2])
    volume_L = float(volume_nm3 * 1.0e-24)
    target_mass_g = float(target_density_g_cm3) * volume_nm3 * 1.0e-21
    target_mass_amu = float(target_mass_g * _AVOGADRO)

    salt_pair_count = max(int(min_salt_pairs), int(round(float(salt_molarity_M) * volume_L * _AVOGADRO)))
    salt_pair_mass_amu = float(salt_pair_count * sum(salt_mw))
    if salt_pair_mass_amu >= target_mass_amu:
        raise ValueError(
            "Requested salt concentration leaves no mass budget for solvents in the target box; "
            "increase target box volume, reduce target density, or lower the salt loading."
        )

    solvent_mass_amu = float(target_mass_amu - salt_pair_mass_amu)
    ratio = np.asarray(mass_ratio, dtype=float)
    raw_solvent_counts = solvent_mass_amu * ratio / max(float(ratio.sum()), 1.0e-12) / np.asarray(solvent_mw, dtype=float)
    solvent_target_total = int(max(int(round(float(raw_solvent_counts.sum()))), int(solvent_min.sum())))
    solvent_counts = _largest_remainder_allocate(
        raw_solvent_counts,
        target_total=solvent_target_total,
        min_counts=solvent_min,
    )

    all_counts = tuple(int(x) for x in solvent_counts.tolist()) + (int(salt_pair_count), int(salt_pair_count))
    all_names = tuple(solvent_names) + tuple(salt_names)
    all_mw = np.asarray(solvent_mw + salt_mw, dtype=float)
    estimated_mass_amu = float(np.dot(np.asarray(all_counts, dtype=float), all_mw))
    estimated_density = float((estimated_mass_amu / _AVOGADRO) / (volume_nm3 * 1.0e-21)) if volume_nm3 > 0.0 else 0.0
    estimated_salt_molarity = float((float(salt_pair_count) / _AVOGADRO) / max(volume_L, 1.0e-30))

    notes = [
        f"directly planned electrolyte counts for target box {tuple(round(x, 4) for x in box_nm)} nm",
        f"held XY fixed at ({box_nm[0]:.4f}, {box_nm[1]:.4f}) nm and used target Z={box_nm[2]:.4f} nm during packing",
        f"target density={float(target_density_g_cm3):.4f} g/cm^3, estimated packed density after integer rounding={estimated_density:.4f} g/cm^3",
        f"requested salt molarity={float(salt_molarity_M):.4f} M, estimated salt molarity after integer rounding={estimated_salt_molarity:.4f} M",
    ]

    return DirectElectrolytePlan(
        target_box_nm=box_nm,
        target_volume_nm3=volume_nm3,
        target_density_g_cm3=float(target_density_g_cm3),
        target_counts=all_counts,
        species_names=all_names,
        solvent_counts=tuple(int(x) for x in solvent_counts.tolist()),
        salt_pair_count=int(salt_pair_count),
        estimated_density_g_cm3=estimated_density,
        estimated_salt_molarity_M=estimated_salt_molarity,
        notes=tuple(notes),
    )


def plan_fixed_xy_direct_pack_box(
    *,
    reference_box_nm: tuple[float, float, float],
    target_counts: Sequence[int],
    mol_weights: Sequence[float],
    initial_pack_density_g_cm3: float,
    species_names: Sequence[str] | None = None,
    z_padding_factor: float = 1.05,
    minimum_z_nm: float | None = None,
    maximum_z_nm: float | None = None,
) -> FixedXYDirectPackPlan:
    ref_box = tuple(float(x) for x in reference_box_nm)
    if len(ref_box) != 3 or min(ref_box) <= 0.0:
        raise ValueError("reference_box_nm must contain three positive lengths")

    counts = _normalize_ints(target_counts, label="target_counts")
    mol_weights_f = _normalize_floats(mol_weights, label="mol_weights")
    if len(counts) != len(mol_weights_f):
        raise ValueError("target_counts and mol_weights must have the same length")
    if species_names is None:
        names = tuple(f"species_{idx}" for idx in range(len(counts)))
    else:
        names = tuple(str(name) for name in species_names)
        if len(names) != len(counts):
            raise ValueError("species_names and target_counts must have the same length")

    pack_density = float(initial_pack_density_g_cm3)
    if pack_density <= 0.0:
        raise ValueError("initial_pack_density_g_cm3 must be positive")
    if z_padding_factor < 1.0:
        raise ValueError("z_padding_factor must be >= 1.0")
    if maximum_z_nm is not None and float(maximum_z_nm) <= 0.0:
        raise ValueError("maximum_z_nm must be positive when provided")

    total_mass_amu = float(np.dot(np.asarray(counts, dtype=float), np.asarray(mol_weights_f, dtype=float)))
    required_volume_nm3 = float((total_mass_amu / _AVOGADRO) / (pack_density * 1.0e-21)) if total_mass_amu > 0.0 else 0.0
    xy_area_nm2 = float(ref_box[0] * ref_box[1])
    required_z_nm = float(required_volume_nm3 / max(xy_area_nm2, 1.0e-30)) if required_volume_nm3 > 0.0 else 0.0
    lower_bound_z_nm = max(float(ref_box[2]), float(minimum_z_nm) if minimum_z_nm is not None else 0.0)
    density_padded_z_nm = max(lower_bound_z_nm, float(required_z_nm * float(z_padding_factor)))
    upper_bound_z_nm = None
    if maximum_z_nm is not None:
        upper_bound_z_nm = max(lower_bound_z_nm, float(maximum_z_nm))
    pack_z_nm = min(density_padded_z_nm, upper_bound_z_nm) if upper_bound_z_nm is not None else density_padded_z_nm
    pack_box_nm = (float(ref_box[0]), float(ref_box[1]), float(pack_z_nm))
    pack_volume_nm3 = float(pack_box_nm[0] * pack_box_nm[1] * pack_box_nm[2])
    estimated_density = float((total_mass_amu / _AVOGADRO) / (pack_volume_nm3 * 1.0e-21)) if pack_volume_nm3 > 0.0 else 0.0

    notes = [
        f"kept fixed XY at ({ref_box[0]:.4f}, {ref_box[1]:.4f}) nm and derived an initial pack Z from the rounded molecule counts",
        f"reference box Z={ref_box[2]:.4f} nm, density-derived Z={required_z_nm:.4f} nm, z_padding_factor={float(z_padding_factor):.3f}, selected initial pack Z={pack_z_nm:.4f} nm",
        f"initial pack density target={pack_density:.4f} g/cm^3, estimated density in the selected pack box={estimated_density:.4f} g/cm^3",
    ]
    if upper_bound_z_nm is not None and density_padded_z_nm > upper_bound_z_nm + 1.0e-12:
        notes.append(
            f"capped initial pack Z from {density_padded_z_nm:.4f} to {upper_bound_z_nm:.4f} nm to avoid an excessively dilute fixed-XY box before semiisotropic relaxation"
        )

    return FixedXYDirectPackPlan(
        reference_box_nm=ref_box,
        initial_pack_box_nm=pack_box_nm,
        fixed_xy_nm=(float(ref_box[0]), float(ref_box[1])),
        target_counts=counts,
        species_names=names,
        initial_pack_density_g_cm3=pack_density,
        estimated_initial_density_g_cm3=estimated_density,
        notes=tuple(notes),
    )


def fixed_xy_semiisotropic_npt_overrides(*, pressure_bar: float, z_compressibility_bar_inv: float = 4.5e-5) -> dict[str, object]:
    p = float(pressure_bar)
    comp_z = float(z_compressibility_bar_inv)
    return {
        "pcoupltype": "semiisotropic",
        "ref_p": f"{p:.6g} {p:.6g}",
        "compressibility": f"0 {comp_z:.6g}",
    }
