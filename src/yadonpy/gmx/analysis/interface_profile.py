"""Interface-specific profile statistics for graphite/polymer/electrolyte stacks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..topology import SystemTopology, parse_system_top
from .structured import build_msd_metric_catalog, build_species_catalog, compute_msd_series


_AVOGADRO = 6.02214076e23
_AMU_PER_NM3_TO_G_CM3 = 1.66053906660e-3
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_EPS0_F_M = 8.8541878128e-12


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return [_jsonify(x) for x in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _read_ndx_groups(ndx_path: Path) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    current: str | None = None
    for raw in Path(ndx_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]").strip()
            groups.setdefault(current, [])
            continue
        if current is not None:
            groups[current].extend(int(tok) for tok in line.split())
    return groups


def _read_gro_frame(gro_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {gro_path}")
    natoms = int(lines[1].strip())
    coords: list[tuple[float, float, float]] = []
    for idx in range(natoms):
        raw = lines[2 + idx]
        try:
            coords.append((float(raw[20:28]), float(raw[28:36]), float(raw[36:44])))
        except Exception:
            parts = raw.split()
            coords.append((float(parts[-3]), float(parts[-2]), float(parts[-1])))
    box = tuple(float(x) for x in lines[-1].split()[:3])
    return np.asarray(coords, dtype=float), (float(box[0]), float(box[1]), float(box[2]))


def _iter_frames(
    *,
    gro_path: Path,
    xtc_path: Path | None,
    frame_stride: int,
    chunk: int = 50,
):
    stride = max(1, int(frame_stride or 1))
    if xtc_path is None or not Path(xtc_path).exists():
        coords, box = _read_gro_frame(gro_path)
        yield 0.0, coords, box
        return
    try:
        import mdtraj as md
    except Exception:
        coords, box = _read_gro_frame(gro_path)
        yield 0.0, coords, box
        return

    frame_offset = 0
    yielded = False
    try:
        iterator = md.iterload(str(xtc_path), top=str(gro_path), chunk=int(max(1, chunk)))
        for trj_raw in iterator:
            raw_n = int(getattr(trj_raw, "n_frames", np.asarray(getattr(trj_raw, "xyz", [])).shape[0]))
            if raw_n <= 0:
                continue
            keep = [i for i in range(raw_n) if ((frame_offset + i) % stride) == 0]
            frame_offset += raw_n
            if not keep:
                continue
            trj = trj_raw[keep]
            xyz = np.asarray(trj.xyz, dtype=float)
            boxes = np.asarray(getattr(trj, "unitcell_lengths", None), dtype=float)
            times = np.asarray(getattr(trj, "time", np.arange(xyz.shape[0], dtype=float)), dtype=float)
            for i in range(xyz.shape[0]):
                if boxes.ndim == 2 and boxes.shape[0] > i and boxes.shape[1] >= 3:
                    box = (float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]))
                else:
                    _coords0, box = _read_gro_frame(gro_path)
                yielded = True
                yield float(times[i]), np.asarray(xyz[i], dtype=float), box
    except Exception:
        if not yielded:
            coords, box = _read_gro_frame(gro_path)
            yield 0.0, coords, box
            return
        raise
    if not yielded:
        coords, box = _read_gro_frame(gro_path)
        yield 0.0, coords, box


def _atom_payload(top: SystemTopology, system_dir: Path) -> dict[str, Any]:
    catalog = build_species_catalog(top, system_dir)
    total_atoms = sum(
        int(top.moleculetypes[str(moltype)].natoms) * int(count)
        for moltype, count in top.molecules
        if str(moltype) in top.moleculetypes
    )
    moltypes = np.asarray([""] * total_atoms, dtype=object)
    kinds = np.asarray([""] * total_atoms, dtype=object)
    masses = np.ones(total_atoms, dtype=float)
    charges = np.zeros(total_atoms, dtype=float)
    instances: list[dict[str, Any]] = []
    for moltype, entry in catalog.items():
        mt = top.moleculetypes.get(str(moltype))
        if mt is None:
            continue
        local_masses = np.asarray(mt.masses if mt.masses else entry.get("masses"), dtype=float)
        if local_masses.size != int(mt.natoms):
            local_masses = np.ones(int(mt.natoms), dtype=float)
        local_charges = np.asarray(mt.charges if mt.charges else np.zeros(int(mt.natoms)), dtype=float)
        if local_charges.size != int(mt.natoms):
            local_charges = np.zeros(int(mt.natoms), dtype=float)
        kind = str(entry.get("kind") or "")
        for inst in entry.get("instances", []) or []:
            idx = np.asarray(inst.get("atom_indices_0"), dtype=int)
            if idx.size != int(mt.natoms):
                continue
            moltypes[idx] = str(moltype)
            kinds[idx] = kind
            masses[idx] = local_masses
            charges[idx] = local_charges
            instances.append(
                {
                    "moltype": str(moltype),
                    "kind": kind,
                    "formal_charge_e": float(entry.get("formal_charge_e") or 0.0),
                    "atom_indices_0": idx,
                    "masses": local_masses,
                    "charges": local_charges,
                    "atomnames": list(mt.atomnames),
                    "atomtypes": list(mt.atomtypes),
                }
            )
    return {
        "catalog": catalog,
        "moltypes": moltypes,
        "kinds": kinds,
        "masses": masses,
        "charges": charges,
        "instances": instances,
    }


def _phase_index_masks(*, ndx_groups: dict[str, list[int]], natoms: int, phase_groups: Sequence[str]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name in phase_groups:
        idx0 = np.asarray([int(i) - 1 for i in ndx_groups.get(str(name), []) if 1 <= int(i) <= natoms], dtype=int)
        mask = np.zeros(natoms, dtype=bool)
        if idx0.size:
            mask[idx0] = True
        masks[str(name)] = mask
    return masks


def _z_quantiles(z: np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(z, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{q:02d}_z_nm": None for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}
    return {f"p{q:02d}_z_nm": float(np.percentile(arr, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}


def _min_distance_between(
    coords: np.ndarray,
    a_mask: np.ndarray,
    b_mask: np.ndarray,
    box_nm: tuple[float, float, float],
) -> float | None:
    a = np.asarray(coords[a_mask], dtype=float)
    b = np.asarray(coords[b_mask], dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    best = float("inf")
    bx = max(float(box_nm[0]), 1.0e-12)
    by = max(float(box_nm[1]), 1.0e-12)
    for start in range(0, len(a), 512):
        aa = a[start : start + 512]
        dx = aa[:, None, 0] - b[None, :, 0]
        dy = aa[:, None, 1] - b[None, :, 1]
        dz = aa[:, None, 2] - b[None, :, 2]
        dx -= bx * np.round(dx / bx)
        dy -= by * np.round(dy / by)
        val = float(np.sqrt(np.min(dx * dx + dy * dy + dz * dz)))
        if val < best:
            best = val
    return None if not np.isfinite(best) else best


def _profile_rows(
    *,
    frames: list[tuple[float, np.ndarray, tuple[float, float, float]]],
    bins: np.ndarray,
    phase_masks: dict[str, np.ndarray],
    moltypes: np.ndarray,
    masses: np.ndarray,
    charges: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    labels = [f"phase:{name}" for name in phase_masks]
    labels.extend(f"moltype:{name}" for name in sorted(set(str(x) for x in moltypes if str(x))))
    accum: dict[str, dict[str, np.ndarray]] = {
        label: {
            "mass_amu": np.zeros(len(bins) - 1, dtype=float),
            "charge_e": np.zeros(len(bins) - 1, dtype=float),
            "atom_count": np.zeros(len(bins) - 1, dtype=float),
        }
        for label in labels
    }
    volume_acc = np.zeros(len(bins) - 1, dtype=float)
    for _time_ps, coords, box in frames:
        z = np.asarray(coords[:, 2], dtype=float)
        widths = np.diff(bins)
        volume_acc += float(box[0]) * float(box[1]) * widths
        for phase, mask in phase_masks.items():
            label = f"phase:{phase}"
            _accumulate_hist(accum[label], z, mask, bins, masses, charges)
        for moltype in sorted(set(str(x) for x in moltypes if str(x))):
            label = f"moltype:{moltype}"
            mask = moltypes == moltype
            _accumulate_hist(accum[label], z, mask, bins, masses, charges)
    n_frames = max(1, len(frames))
    volume_mean = np.maximum(volume_acc / float(n_frames), 1.0e-12)
    rows: list[dict[str, Any]] = []
    for label, payload in accum.items():
        kind, name = label.split(":", 1)
        for i in range(len(bins) - 1):
            mass = float(payload["mass_amu"][i]) / float(n_frames)
            charge = float(payload["charge_e"][i]) / float(n_frames)
            atoms = float(payload["atom_count"][i]) / float(n_frames)
            rows.append(
                {
                    "entity_kind": kind,
                    "entity": name,
                    "z_lo_nm": float(bins[i]),
                    "z_hi_nm": float(bins[i + 1]),
                    "z_mid_nm": float(0.5 * (bins[i] + bins[i + 1])),
                    "mass_density_g_cm3": mass * _AMU_PER_NM3_TO_G_CM3 / float(volume_mean[i]),
                    "charge_density_e_nm3": charge / float(volume_mean[i]),
                    "atom_number_density_nm3": atoms / float(volume_mean[i]),
                }
            )
    density_arrays: dict[str, dict[str, np.ndarray]] = {}
    for label, payload in accum.items():
        mass_mean = payload["mass_amu"] / float(n_frames)
        charge_mean = payload["charge_e"] / float(n_frames)
        density_arrays[label] = {
            "z_mid_nm": 0.5 * (bins[:-1] + bins[1:]),
            "mass_density_g_cm3": mass_mean * _AMU_PER_NM3_TO_G_CM3 / volume_mean,
            "charge_density_e_nm3": charge_mean / volume_mean,
        }
    return rows, density_arrays


def _accumulate_hist(
    target: dict[str, np.ndarray],
    z: np.ndarray,
    mask: np.ndarray,
    bins: np.ndarray,
    masses: np.ndarray,
    charges: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    target["mass_amu"] += np.histogram(z[mask], bins=bins, weights=masses[mask])[0]
    target["charge_e"] += np.histogram(z[mask], bins=bins, weights=charges[mask])[0]
    target["atom_count"] += np.histogram(z[mask], bins=bins)[0]


def _write_profile_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "entity_kind",
        "entity",
        "z_lo_nm",
        "z_hi_nm",
        "z_mid_nm",
        "mass_density_g_cm3",
        "charge_density_e_nm3",
        "atom_number_density_nm3",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(str(key))
        fields = keys
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[str(x) for x in fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({str(key): row.get(str(key)) for key in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_z_profile_svg(path: Path, rows: Sequence[dict[str, Any]]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    phase_rows: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if str(row.get("entity_kind") or "") != "phase":
            continue
        phase_rows.setdefault(str(row.get("entity") or ""), []).append(
            (float(row.get("z_mid_nm") or 0.0), float(row.get("mass_density_g_cm3") or 0.0))
        )
    if not phase_rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for label, series in phase_rows.items():
        series = sorted(series)
        ax.plot([x for x, _y in series], [y for _x, y in series], label=label)
    ax.set_xlabel("z / nm")
    ax.set_ylabel("mass density / g cm$^{-3}$")
    ax.set_title("Layer-stack z density profiles")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_fraction_bar_svg(path: Path, summary_by_species: dict[str, Any], key: str, ylabel: str, title: str) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    labels = [str(k) for k in summary_by_species.keys()]
    values = [float((summary_by_species.get(k) or {}).get(key) or 0.0) for k in labels]
    if not labels:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4.0, 0.6 * len(labels)), 3.2))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, max(1.0, max(values) * 1.1 if values else 1.0))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _find_crossing(z: np.ndarray, a: np.ndarray, b: np.ndarray) -> float | None:
    if z.size == 0 or a.size != z.size or b.size != z.size:
        return None
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if float(np.max(aa)) > 0.0:
        aa = aa / float(np.max(aa))
    if float(np.max(bb)) > 0.0:
        bb = bb / float(np.max(bb))
    diff = aa - bb
    valid = np.isfinite(diff)
    significant = (aa > 0.02) | (bb > 0.02)
    for i in range(1, len(z)):
        if not (valid[i - 1] and valid[i]):
            continue
        if not (bool(significant[i - 1]) or bool(significant[i])):
            continue
        if diff[i - 1] == 0.0 and bool(significant[i - 1]):
            return float(z[i - 1])
        if diff[i - 1] * diff[i] < 0.0:
            denom = abs(diff[i - 1]) + abs(diff[i])
            frac = 0.5 if denom <= 0.0 else abs(diff[i - 1]) / denom
            return float(z[i - 1] + frac * (z[i] - z[i - 1]))
    return None


def _regions(
    *,
    phase_stats: dict[str, dict[str, Any]],
    density_arrays: dict[str, dict[str, np.ndarray]],
    box_z_nm: float,
    region_width_nm: float,
    phase_groups: Sequence[str],
) -> dict[str, dict[str, float]]:
    graphite, polymer, electrolyte = [str(x) for x in phase_groups[:3]]
    graphite_top = float(phase_stats.get(graphite, {}).get("p95_z_nm") or phase_stats.get(graphite, {}).get("p100_z_nm") or 0.0)
    p_stats = phase_stats.get(polymer, {})
    e_stats = phase_stats.get(electrolyte, {})
    p_arr = density_arrays.get(f"phase:{polymer}", {})
    e_arr = density_arrays.get(f"phase:{electrolyte}", {})
    dividing = _find_crossing(
        np.asarray(p_arr.get("z_mid_nm", []), dtype=float),
        np.asarray(p_arr.get("mass_density_g_cm3", []), dtype=float),
        np.asarray(e_arr.get("mass_density_g_cm3", []), dtype=float),
    )
    if dividing is None:
        p95 = float(p_stats.get("p95_z_nm") or p_stats.get("p100_z_nm") or graphite_top + region_width_nm)
        e05 = float(e_stats.get("p05_z_nm") or e_stats.get("p00_z_nm") or p95)
        dividing = 0.5 * (p95 + e05)
    half = 0.5 * float(region_width_nm)
    graphite_near_hi = float(min(box_z_nm, graphite_top + float(region_width_nm)))
    mixed_lo = float(max(graphite_near_hi, dividing - half))
    mixed_hi = float(max(mixed_lo, min(box_z_nm, dividing + half)))
    return {
        "graphite_near": {
            "z_lo_nm": float(graphite_top),
            "z_hi_nm": graphite_near_hi,
        },
        "polymer_core": {
            "z_lo_nm": float(min(box_z_nm, graphite_near_hi)),
            "z_hi_nm": float(max(graphite_near_hi, mixed_lo)),
        },
        "polymer_electrolyte_mixed": {
            "z_lo_nm": mixed_lo,
            "z_hi_nm": mixed_hi,
        },
        "electrolyte_core": {
            "z_lo_nm": float(min(box_z_nm, mixed_hi)),
            "z_hi_nm": float(box_z_nm),
        },
    }


def _generic_regions(
    *,
    phase_stats: dict[str, dict[str, Any]],
    box_z_nm: float,
    region_width_nm: float,
    phase_groups: Sequence[str],
) -> dict[str, dict[str, float]]:
    regions: dict[str, dict[str, float]] = {}
    ordered = [
        str(name)
        for name in phase_groups
        if phase_stats.get(str(name), {}).get("p50_z_nm") is not None
    ]
    ordered.sort(key=lambda name: float(phase_stats[name]["p50_z_nm"]))
    for name in ordered:
        stats = phase_stats.get(name, {})
        lo = stats.get("p05_z_nm")
        hi = stats.get("p95_z_nm")
        if lo is None or hi is None:
            continue
        regions[f"{name.lower()}_core"] = {
            "z_lo_nm": float(max(0.0, float(lo))),
            "z_hi_nm": float(min(float(box_z_nm), float(hi))),
        }
    half = 0.5 * float(region_width_nm)
    for left, right in zip(ordered, ordered[1:]):
        left_hi = phase_stats.get(left, {}).get("p95_z_nm")
        right_lo = phase_stats.get(right, {}).get("p05_z_nm")
        if left_hi is None or right_lo is None:
            continue
        mid = 0.5 * (float(left_hi) + float(right_lo))
        regions[f"{left.lower()}__{right.lower()}_interface"] = {
            "z_lo_nm": float(max(0.0, mid - half)),
            "z_hi_nm": float(min(float(box_z_nm), mid + half)),
        }
    return regions


def _adjacent_interface_summary(
    *,
    phase_groups: Sequence[str],
    phase_stats: dict[str, dict[str, Any]],
    phase_masks: dict[str, np.ndarray],
    coords: np.ndarray,
    box_nm: tuple[float, float, float],
) -> list[dict[str, Any]]:
    ordered = [
        str(name)
        for name in phase_groups
        if phase_stats.get(str(name), {}).get("p50_z_nm") is not None
    ]
    ordered.sort(key=lambda name: float(phase_stats[name]["p50_z_nm"]))
    out: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        left_stats = phase_stats.get(left, {})
        right_stats = phase_stats.get(right, {})
        left_hi = left_stats.get("p95_z_nm")
        right_lo = right_stats.get("p05_z_nm")
        gap = None if left_hi is None or right_lo is None else float(float(right_lo) - float(left_hi))
        dmin = _min_distance_between(coords, phase_masks[left], phase_masks[right], box_nm)
        out.append(
            {
                "left": left,
                "right": right,
                "gap_from_quantiles_nm": gap,
                "min_distance_nm": dmin,
                "overlap_from_quantiles": bool(gap is not None and float(gap) < 0.0),
                "severe_overlap": bool(dmin is not None and float(dmin) < 0.055),
            }
        )
    return out


def _edl_diagnostics(
    *,
    phase_groups: Sequence[str],
    density_arrays: dict[str, dict[str, np.ndarray]],
    phase_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Cheap fixed-charge EDL diagnostics for graphite/electrolyte neighbors."""

    ordered = [
        str(name)
        for name in phase_groups
        if phase_stats.get(str(name), {}).get("p50_z_nm") is not None
    ]
    ordered.sort(key=lambda name: float(phase_stats[name]["p50_z_nm"]))
    pairs: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        left_is_g = "GRAPHITE" in left.upper()
        right_is_g = "GRAPHITE" in right.upper()
        left_is_e = "ELECTROLYTE" in left.upper()
        right_is_e = "ELECTROLYTE" in right.upper()
        if not ((left_is_g and right_is_e) or (right_is_g and left_is_e)):
            continue
        electrolyte = right if right_is_e else left
        arr = density_arrays.get(f"phase:{electrolyte}", {})
        z = np.asarray(arr.get("z_mid_nm", []), dtype=float)
        q = np.asarray(arr.get("charge_density_e_nm3", []), dtype=float)
        if z.size == 0 or q.size != z.size:
            cumulative = []
        else:
            dz = np.gradient(z) if z.size >= 2 else np.ones_like(z)
            cumulative = np.cumsum(q * dz).tolist()
        pairs.append(
            {
                "interface": f"{left}-{right}",
                "electrolyte_phase": electrolyte,
                "integrated_electrolyte_charge_e_per_nm2": cumulative,
                "z_mid_nm": z.tolist(),
            }
        )
    return {
        "available": bool(pairs),
        "pairs": pairs,
        "note": "Fixed-charge surface diagnostic; not a constant-potential model.",
    }


def _total_charge_density_from_phases(
    *,
    phase_groups: Sequence[str],
    density_arrays: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    z_ref: np.ndarray | None = None
    q_total: np.ndarray | None = None
    for phase in phase_groups:
        arr = density_arrays.get(f"phase:{phase}", {})
        z = np.asarray(arr.get("z_mid_nm", []), dtype=float)
        q = np.asarray(arr.get("charge_density_e_nm3", []), dtype=float)
        if z.size == 0 or q.size != z.size:
            continue
        if z_ref is None:
            z_ref = z
            q_total = np.zeros_like(q, dtype=float)
        if q_total is not None and q_total.size == q.size:
            q_total += q
    if z_ref is None or q_total is None:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return z_ref, q_total


def _charge_potential_profiles(
    *,
    phase_groups: Sequence[str],
    density_arrays: dict[str, dict[str, np.ndarray]],
    potential_reference: str = "zero_mean",
) -> dict[str, Any]:
    """Integrate z charge density into cumulative charge, field, and potential.

    This is a fixed-charge slab diagnostic, not a constant-potential electrode
    solver.  The potential is a one-dimensional vacuum-permittivity reference
    potential derived from the sampled charge distribution.
    """

    z, q_e_nm3 = _total_charge_density_from_phases(phase_groups=phase_groups, density_arrays=density_arrays)
    if z.size == 0 or q_e_nm3.size != z.size:
        return {"available": False, "reason": "missing_charge_density"}
    dz_nm = np.gradient(z) if z.size >= 2 else np.ones_like(z)
    integrated_e_nm2 = np.cumsum(q_e_nm3 * dz_nm)
    rho_c_m3 = q_e_nm3 * _ELEMENTARY_CHARGE_C / 1.0e-27
    dz_m = dz_nm * 1.0e-9
    electric_field_v_m = np.cumsum(rho_c_m3 * dz_m / _EPS0_F_M)
    potential_v = -np.cumsum(electric_field_v_m * dz_m)
    ref = str(potential_reference or "zero_mean").strip().lower()
    if ref == "zero_start" and potential_v.size:
        potential_v = potential_v - potential_v[0]
    else:
        potential_v = potential_v - float(np.mean(potential_v))
    potential_drop_v = None
    if potential_v.size >= 2:
        potential_drop_v = float(potential_v[-1] - potential_v[0])
    rows = [
        {
            "z_nm": float(z[i]),
            "charge_density_e_nm3": float(q_e_nm3[i]),
            "integrated_charge_e_nm2": float(integrated_e_nm2[i]),
            "electric_field_V_m": float(electric_field_v_m[i]),
            "electrostatic_potential_V": float(potential_v[i]),
        }
        for i in range(int(z.size))
    ]
    return {
        "available": True,
        "potential_reference": ref,
        "potential_drop_V": potential_drop_v,
        "rows": rows,
        "note": "One-dimensional fixed-charge diagnostic using vacuum permittivity; not a constant-potential model.",
    }


def _phase_quantile_rows(phase_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, stats in phase_stats.items():
        row = {"phase": phase}
        for key, value in stats.items():
            if key.endswith("_z_nm") or key == "atom_count":
                row[key] = value
        rows.append(row)
    return rows


def _charge_profile_rows(profile_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "entity_kind": row.get("entity_kind"),
            "entity": row.get("entity"),
            "z_lo_nm": row.get("z_lo_nm"),
            "z_hi_nm": row.get("z_hi_nm"),
            "z_mid_nm": row.get("z_mid_nm"),
            "charge_density_e_nm3": row.get("charge_density_e_nm3"),
        }
        for row in profile_rows
    ]


def _edl_species_rows(profile_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in profile_rows:
        if str(row.get("entity_kind") or "") != "moltype":
            continue
        out.append(
            {
                "species": row.get("entity"),
                "z_mid_nm": row.get("z_mid_nm"),
                "number_density_nm3": row.get("atom_number_density_nm3"),
                "charge_density_e_nm3": row.get("charge_density_e_nm3"),
                "mass_density_g_cm3": row.get("mass_density_g_cm3"),
            }
        )
    return out


def _interpenetration(
    *,
    density_arrays: dict[str, dict[str, np.ndarray]],
    polymer: str,
    electrolyte: str,
    graphite: str,
    phase_stats: dict[str, dict[str, Any]],
    min_graphite_electrolyte_nm: float | None,
) -> dict[str, Any]:
    p = density_arrays.get(f"phase:{polymer}", {})
    e = density_arrays.get(f"phase:{electrolyte}", {})
    z = np.asarray(p.get("z_mid_nm", []), dtype=float)
    pd = np.asarray(p.get("mass_density_g_cm3", []), dtype=float)
    ed = np.asarray(e.get("mass_density_g_cm3", []), dtype=float)
    if z.size == 0 or pd.size != z.size or ed.size != z.size:
        return {"overlap_width_nm": 0.0, "overlap_integral_nm": 0.0}
    pn = pd / max(float(np.max(pd)), 1.0e-12)
    en = ed / max(float(np.max(ed)), 1.0e-12)
    mask = (pn > 0.05) & (en > 0.05)
    width = 0.0
    if np.any(mask):
        dz = float(np.median(np.diff(z))) if z.size >= 2 else 0.0
        width = float((np.max(z[mask]) - np.min(z[mask])) + dz)
    dz_arr = np.gradient(z) if z.size >= 2 else np.ones_like(z)
    overlap_integral = float(np.sum(np.minimum(pn, en) * dz_arr))
    graphite_top = float(phase_stats.get(graphite, {}).get("p95_z_nm") or 0.0)
    electrolyte_min = float(phase_stats.get(electrolyte, {}).get("p00_z_nm") or 0.0)
    return {
        "overlap_width_nm": float(max(0.0, width)),
        "overlap_integral_nm": float(overlap_integral),
        "electrolyte_approaches_graphite": bool(
            (min_graphite_electrolyte_nm is not None and float(min_graphite_electrolyte_nm) < 0.35)
            or electrolyte_min <= graphite_top + 0.35
        ),
        "electrolyte_min_minus_graphite_top_nm": float(electrolyte_min - graphite_top),
    }


def _instance_com_z(coords: np.ndarray, inst: dict[str, Any]) -> float:
    idx = np.asarray(inst["atom_indices_0"], dtype=int)
    masses = np.asarray(inst.get("masses"), dtype=float)
    if masses.size != idx.size or float(np.sum(masses)) <= 0.0:
        masses = np.ones(idx.size, dtype=float)
    return float(np.average(coords[idx, 2], weights=masses))


def _region_for_z(z: float, regions: dict[str, dict[str, float]]) -> str | None:
    for name, bounds in regions.items():
        if float(bounds["z_lo_nm"]) <= float(z) < float(bounds["z_hi_nm"]):
            return str(name)
    # Include the top boundary in the final region.
    for name, bounds in regions.items():
        if abs(float(z) - float(bounds["z_hi_nm"])) <= 1.0e-9:
            return str(name)
    return None


def _depth_inside_region(z: float, bounds: dict[str, float]) -> float | None:
    lo = float(bounds["z_lo_nm"])
    hi = float(bounds["z_hi_nm"])
    zz = float(z)
    if zz < lo or zz > hi:
        return None
    return float(max(0.0, min(zz - lo, hi - zz)))


def _enrichment(
    *,
    frames: list[tuple[float, np.ndarray, tuple[float, float, float]]],
    instances: Sequence[dict[str, Any]],
    regions: dict[str, dict[str, float]],
    box_xy_nm2: float,
) -> dict[str, Any]:
    counts: dict[str, dict[str, float]] = {}
    volumes = {
        name: float(box_xy_nm2) * max(float(bounds["z_hi_nm"]) - float(bounds["z_lo_nm"]), 1.0e-12)
        for name, bounds in regions.items()
    }
    for _time_ps, coords, _box in frames:
        for inst in instances:
            moltype = str(inst.get("moltype") or "")
            region = _region_for_z(_instance_com_z(coords, inst), regions)
            if region is None:
                continue
            counts.setdefault(moltype, {}).setdefault(region, 0.0)
            counts[moltype][region] += 1.0
    n_frames = max(1, len(frames))
    out: dict[str, Any] = {}
    for moltype, region_counts in counts.items():
        conc = {
            region: float(region_counts.get(region, 0.0)) / float(n_frames) / max(float(vol), 1.0e-12)
            for region, vol in volumes.items()
        }
        core_candidates = [conc.get("electrolyte_core", 0.0), conc.get("polymer_core", 0.0)]
        reference = max([float(x) for x in core_candidates if float(x) > 0.0] or [sum(conc.values()) / max(len(conc), 1)])
        out[moltype] = {
            "concentration_nm3": conc,
            "enrichment_vs_core": {
                region: (None if reference <= 0.0 else float(value) / float(reference))
                for region, value in conc.items()
            },
        }
    return out


def _species_matches(moltype: str, species: Sequence[str] | None) -> bool:
    if species is None:
        return True
    label = str(moltype).lower()
    return any(str(item).lower() == label or str(item).lower() in label for item in species)


def _frame_interval_ps(frames: Sequence[tuple[float, np.ndarray, tuple[float, float, float]]]) -> float | None:
    times = [float(item[0]) for item in frames]
    if len(times) < 2:
        return None
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return None
    return float(np.median(deltas))


def _penetration_analysis(
    *,
    frames: Sequence[tuple[float, np.ndarray, tuple[float, float, float]]],
    instances: Sequence[dict[str, Any]],
    regions: dict[str, dict[str, float]],
    species: Sequence[str] | None = None,
    penetration_threshold_nm: float = 0.20,
) -> dict[str, Any]:
    polymer_regions = [name for name in regions if ("polymer" in name.lower() or "cmc" in name.lower() or "mixed" in name.lower())]
    electrolyte_regions = [name for name in regions if "electrolyte" in name.lower()]
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    dt_ps = _frame_interval_ps(frames)
    threshold = max(0.0, float(penetration_threshold_nm))
    for inst_id, inst in enumerate(instances):
        moltype = str(inst.get("moltype") or "")
        kind = str(inst.get("kind") or "").lower()
        if not moltype or "graph" in moltype.lower() or "substrate" in kind:
            continue
        if not _species_matches(moltype, species):
            continue
        in_polymer_series: list[bool] = []
        in_electrolyte_series: list[bool] = []
        min_polymer_distance = float("inf")
        for time_ps, coords, _box in frames:
            z = _instance_com_z(coords, inst)
            region = _region_for_z(z, regions) or "outside_regions"
            polymer_depths = [
                depth
                for reg in polymer_regions
                for depth in [_depth_inside_region(z, regions[reg])]
                if depth is not None
            ]
            polymer_region_depth = max(polymer_depths) if polymer_depths else None
            in_polymer = bool(polymer_region_depth is not None and float(polymer_region_depth) >= threshold)
            in_electrolyte = region in electrolyte_regions
            in_polymer_series.append(bool(in_polymer))
            in_electrolyte_series.append(bool(in_electrolyte))
            if polymer_regions:
                distances = []
                for reg in polymer_regions:
                    bounds = regions[reg]
                    if bounds["z_lo_nm"] <= z <= bounds["z_hi_nm"]:
                        distances.append(0.0)
                    else:
                        distances.append(min(abs(z - bounds["z_lo_nm"]), abs(z - bounds["z_hi_nm"])))
                min_polymer_distance = min(min_polymer_distance, min(distances))
            rows.append(
                {
                    "time_ps": float(time_ps),
                    "moltype": moltype,
                    "instance_index": int(inst_id),
                    "com_z_nm": float(z),
                    "region": region,
                    "in_polymer_region": bool(in_polymer),
                    "in_electrolyte_region": bool(in_electrolyte),
                    "polymer_region_depth_nm": polymer_region_depth,
                }
            )
        rec = summary.setdefault(
            moltype,
            {
                "molecule_count": 0,
                "polymer_frame_count": 0,
                "electrolyte_frame_count": 0,
                "sample_frame_count": 0,
                "entry_event_count": 0,
                "min_distance_to_polymer_region_nm": None,
            },
        )
        rec["molecule_count"] += 1
        rec["polymer_frame_count"] += int(sum(in_polymer_series))
        rec["electrolyte_frame_count"] += int(sum(in_electrolyte_series))
        rec["sample_frame_count"] += int(len(in_polymer_series))
        rec["entry_event_count"] += int(sum((not prev) and cur for prev, cur in zip([False] + in_polymer_series[:-1], in_polymer_series)))
        if np.isfinite(min_polymer_distance):
            old = rec.get("min_distance_to_polymer_region_nm")
            rec["min_distance_to_polymer_region_nm"] = float(min_polymer_distance) if old is None else min(float(old), float(min_polymer_distance))
    for moltype, rec in summary.items():
        denom = max(1, int(rec.get("sample_frame_count") or 0))
        rec["polymer_frame_fraction"] = float(rec.get("polymer_frame_count") or 0) / float(denom)
        rec["electrolyte_frame_fraction"] = float(rec.get("electrolyte_frame_count") or 0) / float(denom)
        rec["estimated_polymer_residence_ps"] = (
            None if dt_ps is None else float(rec.get("polymer_frame_count") or 0) * float(dt_ps)
        )
    return {
        "available": bool(rows),
        "species": None if species is None else [str(x) for x in species],
        "penetration_threshold_nm": float(threshold),
        "polymer_regions": polymer_regions,
        "electrolyte_regions": electrolyte_regions,
        "rows": rows,
        "summary_by_species": summary,
        "note": "Penetration uses molecule COM assignment to polymer/mixed regions; it is a region-residence diagnostic.",
    }


def _graphite_surface_positions(phase_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    surfaces: list[dict[str, Any]] = []
    for phase, stats in phase_stats.items():
        if "GRAPHITE" not in str(phase).upper():
            continue
        lo = stats.get("p05_z_nm")
        hi = stats.get("p95_z_nm")
        if lo is not None:
            surfaces.append({"phase": phase, "side": "bottom", "z_nm": float(lo)})
        if hi is not None:
            surfaces.append({"phase": phase, "side": "top", "z_nm": float(hi)})
    return surfaces


def _nearest_graphite_surface(z_nm: float, surfaces: Sequence[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None]:
    if not surfaces:
        return None, None
    best = min(surfaces, key=lambda item: abs(float(z_nm) - float(item["z_nm"])))
    return best, float(abs(float(z_nm) - float(best["z_nm"])))


def _angle_to_surface_normal(vector: np.ndarray, surface: dict[str, Any] | None) -> float | None:
    if surface is None:
        return None
    vec = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return None
    normal = np.asarray([0.0, 0.0, 1.0 if str(surface.get("side") or "top") == "top" else -1.0], dtype=float)
    cosang = float(np.dot(vec, normal) / norm)
    cosang = max(-1.0, min(1.0, cosang))
    return float(math.degrees(math.acos(cosang)))


def _molecule_orientation_angles(
    *,
    coords: np.ndarray,
    inst: dict[str, Any],
    com: np.ndarray,
    surface: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    idx = np.asarray(inst["atom_indices_0"], dtype=int)
    local = np.asarray(coords[idx], dtype=float)
    charges = np.asarray(inst.get("charges"), dtype=float)
    if charges.size != idx.size:
        charges = np.zeros(idx.size, dtype=float)
    rel = local - np.asarray(com, dtype=float)
    dipole = np.sum(charges[:, None] * rel, axis=0)
    dipole_angle = _angle_to_surface_normal(dipole, surface)

    atomnames = [str(x).lower() for x in inst.get("atomnames") or []]
    oxygen_candidates = [i for i, name in enumerate(atomnames) if name.startswith("o")]
    if not oxygen_candidates:
        oxygen_candidates = [i for i, q in enumerate(charges.tolist()) if float(q) < -0.05]
    carbonyl_angle = None
    if oxygen_candidates and charges.size == idx.size and idx.size >= 2:
        o_local = min(oxygen_candidates, key=lambda i: float(charges[i]))
        c_local = int(np.argmax(charges))
        if c_local != o_local:
            carbonyl_angle = _angle_to_surface_normal(local[o_local] - local[c_local], surface)
    return carbonyl_angle, dipole_angle


def _adsorption_analysis(
    *,
    frames: Sequence[tuple[float, np.ndarray, tuple[float, float, float]]],
    instances: Sequence[dict[str, Any]],
    phase_stats: dict[str, dict[str, Any]],
    species: Sequence[str] | None = None,
    surface_distance_nm: float = 0.50,
    min_residence_ps: float = 10.0,
    surface_grid_nm: float = 0.5,
) -> dict[str, Any]:
    surfaces = _graphite_surface_positions(phase_stats)
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    map_counts: dict[tuple[str, str, int, int], int] = {}
    dt_ps = _frame_interval_ps(frames)
    grid_nm = max(float(surface_grid_nm), 1.0e-6)
    for inst_id, inst in enumerate(instances):
        moltype = str(inst.get("moltype") or "")
        kind = str(inst.get("kind") or "").lower()
        if not moltype or "graph" in moltype.lower() or "substrate" in kind:
            continue
        if not _species_matches(moltype, species):
            continue
        adsorbed_series: list[bool] = []
        for time_ps, coords, box in frames:
            idx = np.asarray(inst["atom_indices_0"], dtype=int)
            masses = np.asarray(inst.get("masses"), dtype=float)
            if masses.size != idx.size or float(np.sum(masses)) <= 0.0:
                masses = np.ones(idx.size, dtype=float)
            com = np.average(coords[idx], axis=0, weights=masses)
            surface, dist = _nearest_graphite_surface(float(com[2]), surfaces)
            adsorbed = bool(dist is not None and float(dist) <= float(surface_distance_nm))
            adsorbed_series.append(adsorbed)
            if surface is not None and adsorbed:
                ix = int(math.floor((float(com[0]) % max(float(box[0]), 1.0e-12)) / grid_nm))
                iy = int(math.floor((float(com[1]) % max(float(box[1]), 1.0e-12)) / grid_nm))
                map_counts[(str(surface["phase"]), str(surface["side"]), ix, iy)] = map_counts.get((str(surface["phase"]), str(surface["side"]), ix, iy), 0) + 1
            carbonyl_angle, dipole_angle = _molecule_orientation_angles(coords=coords, inst=inst, com=com, surface=surface)
            rows.append(
                {
                    "time_ps": float(time_ps),
                    "moltype": moltype,
                    "instance_index": int(inst_id),
                    "com_x_nm": float(com[0]),
                    "com_y_nm": float(com[1]),
                    "com_z_nm": float(com[2]),
                    "nearest_graphite_phase": None if surface is None else surface.get("phase"),
                    "nearest_graphite_side": None if surface is None else surface.get("side"),
                    "surface_distance_nm": dist,
                    "adsorbed": adsorbed,
                    "orientation_available": bool(carbonyl_angle is not None or dipole_angle is not None),
                    "carbonyl_angle_deg": carbonyl_angle,
                    "dipole_proxy_angle_deg": dipole_angle,
                }
            )
        rec = summary.setdefault(moltype, {"molecule_count": 0, "sample_frame_count": 0, "adsorbed_frame_count": 0, "event_count": 0})
        rec["molecule_count"] += 1
        rec["sample_frame_count"] += int(len(adsorbed_series))
        rec["adsorbed_frame_count"] += int(sum(adsorbed_series))
        rec["event_count"] += int(sum((not prev) and cur for prev, cur in zip([False] + adsorbed_series[:-1], adsorbed_series)))
    for rec in summary.values():
        denom = max(1, int(rec.get("sample_frame_count") or 0))
        rec["adsorbed_frame_fraction"] = float(rec.get("adsorbed_frame_count") or 0) / float(denom)
        rec["estimated_adsorbed_residence_ps"] = None if dt_ps is None else float(rec.get("adsorbed_frame_count") or 0) * float(dt_ps)
        rec["passes_min_residence"] = bool(
            rec.get("estimated_adsorbed_residence_ps") is not None
            and float(rec["estimated_adsorbed_residence_ps"]) >= float(min_residence_ps)
        )
    for moltype, rec in summary.items():
        carbonyl = [
            float(row["carbonyl_angle_deg"])
            for row in rows
            if row.get("moltype") == moltype and row.get("adsorbed") and row.get("carbonyl_angle_deg") is not None
        ]
        dipole = [
            float(row["dipole_proxy_angle_deg"])
            for row in rows
            if row.get("moltype") == moltype and row.get("adsorbed") and row.get("dipole_proxy_angle_deg") is not None
        ]
        rec["mean_adsorbed_carbonyl_angle_deg"] = None if not carbonyl else float(np.mean(carbonyl))
        rec["mean_adsorbed_dipole_proxy_angle_deg"] = None if not dipole else float(np.mean(dipole))
    surface_map_rows = [
        {
            "graphite_phase": phase,
            "surface_side": side,
            "grid_x": ix,
            "grid_y": iy,
            "count": count,
        }
        for (phase, side, ix, iy), count in sorted(map_counts.items())
    ]
    return {
        "available": bool(surfaces),
        "species": None if species is None else [str(x) for x in species],
        "surface_distance_nm": float(surface_distance_nm),
        "min_residence_ps": float(min_residence_ps),
        "surfaces": surfaces,
        "rows": rows,
        "surface_map_rows": surface_map_rows,
        "summary_by_species": summary,
        "orientation_note": "Angles are measured relative to the nearest graphite surface normal. Carbonyl uses a charge-guided C-to-O proxy; dipole uses the molecular charge-dipole proxy.",
    }


def _atom_indices_by_category(top: SystemTopology, instances: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    li: list[int] = []
    polymer_o: list[int] = []
    solvent_o: list[int] = []
    anion_f: list[int] = []
    for inst in instances:
        idx = np.asarray(inst["atom_indices_0"], dtype=int)
        moltype = str(inst.get("moltype") or "")
        kind = str(inst.get("kind") or "").lower()
        formal = float(inst.get("formal_charge_e") or 0.0)
        atomnames = [str(x) for x in inst.get("atomnames") or []]
        atomtypes = [str(x) for x in inst.get("atomtypes") or []]
        label = " ".join([moltype, kind]).lower()
        if idx.size == 1 and (formal > 0.0 or "li" in label):
            li.extend(idx.tolist())
            continue
        for local, atom_idx in enumerate(idx.tolist()):
            name = atomnames[local].lower() if local < len(atomnames) else ""
            atype = atomtypes[local].lower() if local < len(atomtypes) else ""
            element_hint = name[:1] or atype[:1]
            if element_hint == "o":
                if kind == "polymer" or "*" in label or "peo" in label or "cmc" in label:
                    polymer_o.append(atom_idx)
                else:
                    solvent_o.append(atom_idx)
            if element_hint == "f" and (formal < 0.0 or "pf6" in label or "tfsi" in label):
                anion_f.append(atom_idx)
    return {
        "li": np.asarray(sorted(set(li)), dtype=int),
        "polymer_o": np.asarray(sorted(set(polymer_o)), dtype=int),
        "solvent_o": np.asarray(sorted(set(solvent_o)), dtype=int),
        "anion_f": np.asarray(sorted(set(anion_f)), dtype=int),
    }


def _count_contacts(
    coords: np.ndarray,
    centers: np.ndarray,
    targets: np.ndarray,
    box: tuple[float, float, float],
    cutoff_nm: float,
) -> np.ndarray:
    if centers.size == 0 or targets.size == 0:
        return np.zeros(centers.size, dtype=int)
    c = coords[centers]
    t = coords[targets]
    bx = max(float(box[0]), 1.0e-12)
    by = max(float(box[1]), 1.0e-12)
    out = np.zeros(centers.size, dtype=int)
    for start in range(0, len(c), 256):
        cc = c[start : start + 256]
        dx = cc[:, None, 0] - t[None, :, 0]
        dy = cc[:, None, 1] - t[None, :, 1]
        dz = cc[:, None, 2] - t[None, :, 2]
        dx -= bx * np.round(dx / bx)
        dy -= by * np.round(dy / by)
        dist2 = dx * dx + dy * dy + dz * dz
        out[start : start + len(cc)] = np.sum(dist2 <= float(cutoff_nm) ** 2, axis=1)
    return out


def _li_coordination(
    *,
    frames: list[tuple[float, np.ndarray, tuple[float, float, float]]],
    top: SystemTopology,
    instances: Sequence[dict[str, Any]],
    regions: dict[str, dict[str, float]],
) -> dict[str, Any]:
    categories = _atom_indices_by_category(top, instances)
    li = categories["li"]
    if li.size == 0:
        return {"available": False, "reason": "no_lithium_indices"}
    state_counts: dict[str, dict[str, int]] = {}
    coordination_sums: dict[str, dict[str, int]] = {}
    for _time_ps, coords, box in frames:
        p = _count_contacts(coords, li, categories["polymer_o"], box, 0.28)
        s = _count_contacts(coords, li, categories["solvent_o"], box, 0.28)
        a = _count_contacts(coords, li, categories["anion_f"], box, 0.32)
        for idx, li_atom in enumerate(li):
            region = _region_for_z(float(coords[int(li_atom), 2]), regions) or "outside_regions"
            n_roles = int(p[idx] > 0) + int(s[idx] > 0) + int(a[idx] > 0)
            if n_roles > 1:
                state = "mixed"
            elif p[idx] > 0:
                state = "polymer_bound"
            elif s[idx] > 0:
                state = "solvent_bound"
            elif a[idx] > 0:
                state = "anion_paired"
            else:
                state = "free_like"
            state_counts.setdefault(region, {}).setdefault(state, 0)
            state_counts[region][state] += 1
            coordination_sums.setdefault(region, {}).setdefault("polymer_o", 0)
            coordination_sums[region]["polymer_o"] += int(p[idx])
            coordination_sums[region].setdefault("solvent_o", 0)
            coordination_sums[region]["solvent_o"] += int(s[idx])
            coordination_sums[region].setdefault("anion_f", 0)
            coordination_sums[region]["anion_f"] += int(a[idx])
            coordination_sums[region].setdefault("samples", 0)
            coordination_sums[region]["samples"] += 1
    by_region: dict[str, Any] = {}
    for region, counts in state_counts.items():
        total = max(1, int(sum(counts.values())))
        sums = coordination_sums.get(region, {})
        samples = max(1, int(sums.get("samples", total)))
        by_region[region] = {
            "state_counts": counts,
            "state_fraction": {key: float(value) / float(total) for key, value in counts.items()},
            "mean_contacts": {
                "polymer_o": float(sums.get("polymer_o", 0)) / float(samples),
                "solvent_o": float(sums.get("solvent_o", 0)) / float(samples),
                "anion_f": float(sums.get("anion_f", 0)) / float(samples),
            },
        }
    return {
        "available": True,
        "cutoff_source": "fallback",
        "cutoffs_nm": {"Li-O": 0.28, "Li-F": 0.32},
        "li_count": int(li.size),
        "donor_counts": {key: int(value.size) for key, value in categories.items() if key != "li"},
        "by_region": by_region,
    }


def _anisotropic_msd(
    *,
    gro_path: Path,
    xtc_path: Path | None,
    top_path: Path,
    system_dir: Path,
    out_dir: Path,
    frame_stride: int,
    analysis_profile: str,
) -> dict[str, Any]:
    if xtc_path is None or not Path(xtc_path).exists():
        return {"available": False, "reason": "missing_xtc"}
    top = parse_system_top(top_path)
    catalog = build_msd_metric_catalog(top, system_dir)
    out: dict[str, Any] = {
        "available": True,
        "analysis_profile": str(analysis_profile),
        "frame_stride": int(max(1, frame_stride)),
        "species": {},
        "interpretation_note": "Dxy is the preferred interface transport metric; Dz is confined-direction mobility.",
    }
    for moltype, entry in catalog.items():
        label = " ".join([str(moltype), str(entry.get("kind") or ""), str(entry.get("smiles") or "")]).lower()
        if any(tok in label for tok in ("graph", "graphite", "graphene", "substrate", "wall")):
            continue
        metric_name = str(entry.get("default_metric") or "")
        metric = (entry.get("metrics") or {}).get(metric_name)
        if not metric:
            continue
        group_specs = list(metric.get("groups") or [])
        if not group_specs:
            continue
        species_rec: dict[str, Any] = {"default_metric": metric_name}
        for geometry in ("xy", "z"):
            try:
                data = compute_msd_series(
                    gro_path=gro_path,
                    xtc_path=xtc_path,
                    top_path=top_path,
                    system_dir=system_dir,
                    group_specs=group_specs,
                    geometry_mode=geometry,
                    unwrap="auto",
                    drift="auto",
                    frame_stride=max(1, int(frame_stride)),
                )
                fit = dict(data.get("fit") or {})
                species_rec[geometry] = {
                    "geometry": geometry,
                    "n_groups": int(data.get("n_groups") or 0),
                    "frame_interval_ps": data.get("frame_interval_ps"),
                    "D_m2_s": fit.get("D_m2_s"),
                    "D_nm2_ps": fit.get("D_nm2_ps"),
                    "alpha_mean": fit.get("alpha_mean"),
                    "confidence": fit.get("confidence"),
                    "status": fit.get("status"),
                    "fit_t_start_ps": fit.get("fit_t_start_ps"),
                    "fit_t_end_ps": fit.get("fit_t_end_ps"),
                }
            except Exception as exc:
                species_rec[geometry] = {"error": str(exc)}
        out["species"][str(moltype)] = species_rec
    (out_dir / "anisotropic_msd_summary.json").write_text(json.dumps(_jsonify(out), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def compute_interface_profile(
    *,
    gro_path: Path,
    top_path: Path,
    ndx_path: Path,
    system_dir: Path,
    out_dir: Path,
    xtc_path: Path | None = None,
    bin_nm: float = 0.05,
    frame_stride: int | str = "auto",
    region_width_nm: float = 0.75,
    surface_grid_nm: float = 0.5,
    surface_distance_nm: float = 0.50,
    penetration_threshold_nm: float = 0.20,
    adsorption_min_residence_ps: float = 10.0,
    potential_reference: str = "zero_mean",
    split_electrodes: bool = False,
    report_potential_drop: bool = False,
    penetration_species: Sequence[str] | None = None,
    adsorption_species: Sequence[str] | None = None,
    analysis_profile: str = "interface_fast",
    phase_groups: Sequence[str] = ("GRAPHITE", "POLYMER", "ELECTROLYTE"),
    compute_transport: bool = True,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Compute cheap interface statistics and write JSON/CSV artifacts."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top = parse_system_top(Path(top_path))
    atom_payload = _atom_payload(top, Path(system_dir))
    coords0, box0 = _read_gro_frame(Path(gro_path))
    natoms = int(coords0.shape[0])
    ndx_groups = _read_ndx_groups(Path(ndx_path))
    phase_masks = _phase_index_masks(ndx_groups=ndx_groups, natoms=natoms, phase_groups=phase_groups)

    stride = 1 if str(frame_stride).strip().lower() == "auto" else max(1, int(frame_stride))
    frames = list(_iter_frames(gro_path=Path(gro_path), xtc_path=xtc_path, frame_stride=stride))
    if not frames:
        frames = [(0.0, coords0, box0)]
    final_time, final_coords, final_box = frames[-1]
    bins = np.arange(0.0, max(float(final_box[2]), float(box0[2])) + float(bin_nm), float(bin_nm))
    if bins.size < 2:
        bins = np.asarray([0.0, max(float(final_box[2]), float(bin_nm))], dtype=float)

    phase_groups_norm = [str(x) for x in phase_groups]
    manifest_order: list[str] | None = None
    if manifest_path is not None and Path(manifest_path).is_file():
        try:
            manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            intervals = manifest_payload.get("layer_intervals_nm", []) if isinstance(manifest_payload, dict) else []
            if isinstance(intervals, list):
                ordered_names = [str(v.get("name")) for v in intervals if isinstance(v, dict) and v.get("name") is not None]
                manifest_order = [name for name in ordered_names if name in phase_groups_norm]
        except Exception:
            manifest_order = None

    phase_stats: dict[str, dict[str, Any]] = {}
    for phase, mask in phase_masks.items():
        phase_stats[phase] = _z_quantiles(final_coords[mask, 2])
        phase_stats[phase]["atom_count"] = int(np.sum(mask))
    phase_quantiles_csv = out_dir / "phase_z_quantiles.csv"
    _write_rows_csv(phase_quantiles_csv, _phase_quantile_rows(phase_stats))
    trajectory_phase_order = sorted(
        [name for name in phase_groups if phase_stats.get(str(name), {}).get("p50_z_nm") is not None],
        key=lambda name: float(phase_stats[str(name)]["p50_z_nm"]),
    )
    # Closed xyz stacks can wrap the bottom layer across the periodic z
    # boundary during whole-molecule post-processing.  In that case raw final
    # z-quantiles are still useful diagnostics, but the manifest is the more
    # reliable source of intended layer order.
    phase_order = list(manifest_order or [str(x) for x in trajectory_phase_order])
    phase_order_source = "manifest" if manifest_order else "trajectory_quantiles"

    profile_rows, density_arrays = _profile_rows(
        frames=frames,
        bins=bins,
        phase_masks=phase_masks,
        moltypes=np.asarray(atom_payload["moltypes"], dtype=object),
        masses=np.asarray(atom_payload["masses"], dtype=float),
        charges=np.asarray(atom_payload["charges"], dtype=float),
    )
    profile_csv = out_dir / "z_density_profiles.csv"
    _write_profile_csv(profile_csv, profile_rows)
    charge_profile_csv = out_dir / "charge_density_profiles.csv"
    _write_rows_csv(charge_profile_csv, _charge_profile_rows(profile_rows))
    z_profiles_svg = _write_z_profile_svg(out_dir / "z_profiles.svg", profile_rows)

    adjacent_interfaces = _adjacent_interface_summary(
        phase_groups=phase_groups_norm,
        phase_stats=phase_stats,
        phase_masks=phase_masks,
        coords=final_coords,
        box_nm=final_box,
    )
    old_three_phase = (
        len(phase_groups_norm) >= 3
        and "GRAPHITE" in phase_groups_norm[0].upper()
        and (
            "POLYMER" in phase_groups_norm[1].upper()
            or "CMC" in phase_groups_norm[1].upper()
            or "PEO" in phase_groups_norm[1].upper()
        )
        and "ELECTROLYTE" in phase_groups_norm[2].upper()
    )
    min_distances: dict[str, float | None] = {
        f"{item['left']}-{item['right']}": item.get("min_distance_nm") for item in adjacent_interfaces
    }
    direct_graphite_electrolyte_contact = False
    min_ge = None
    if old_three_phase:
        graphite, polymer, electrolyte = phase_groups_norm[:3]
        min_gp = _min_distance_between(final_coords, phase_masks[graphite], phase_masks[polymer], final_box)
        min_pe = _min_distance_between(final_coords, phase_masks[polymer], phase_masks[electrolyte], final_box)
        min_ge = _min_distance_between(final_coords, phase_masks[graphite], phase_masks[electrolyte], final_box)
        min_distances.update({f"{graphite}-{polymer}": min_gp, f"{polymer}-{electrolyte}": min_pe, f"{graphite}-{electrolyte}": min_ge})
        regions = _regions(
            phase_stats=phase_stats,
            density_arrays=density_arrays,
            box_z_nm=float(final_box[2]),
            region_width_nm=float(region_width_nm),
            phase_groups=phase_groups_norm,
        )
        interpenetration = _interpenetration(
            density_arrays=density_arrays,
            polymer=polymer,
            electrolyte=electrolyte,
            graphite=graphite,
            phase_stats=phase_stats,
            min_graphite_electrolyte_nm=min_ge,
        )
        direct_graphite_electrolyte_contact = bool(min_ge is not None and float(min_ge) < 0.35)
    else:
        regions = _generic_regions(
            phase_stats=phase_stats,
            box_z_nm=float(final_box[2]),
            region_width_nm=float(region_width_nm),
            phase_groups=phase_groups_norm,
        )
        interpenetration = {
            "available": False,
            "reason": "generic_layer_stack_without_graphite_polymer_electrolyte_order",
        }
        for item in adjacent_interfaces:
            left = str(item.get("left", "")).upper()
            right = str(item.get("right", "")).upper()
            if (("GRAPHITE" in left and "ELECTROLYTE" in right) or ("GRAPHITE" in right and "ELECTROLYTE" in left)) and item.get("min_distance_nm") is not None:
                direct_graphite_electrolyte_contact = direct_graphite_electrolyte_contact or float(item["min_distance_nm"]) < 0.35
                min_ge = float(item["min_distance_nm"])
    enrichment = _enrichment(
        frames=frames,
        instances=atom_payload["instances"],
        regions=regions,
        box_xy_nm2=float(final_box[0]) * float(final_box[1]),
    )
    coordination = _li_coordination(
        frames=frames,
        top=top,
        instances=atom_payload["instances"],
        regions=regions,
    )
    penetration = _penetration_analysis(
        frames=frames,
        instances=atom_payload["instances"],
        regions=regions,
        species=penetration_species,
        penetration_threshold_nm=float(penetration_threshold_nm),
    )
    adsorption = _adsorption_analysis(
        frames=frames,
        instances=atom_payload["instances"],
        phase_stats=phase_stats,
        species=adsorption_species,
        surface_distance_nm=float(surface_distance_nm),
        min_residence_ps=float(adsorption_min_residence_ps),
        surface_grid_nm=float(surface_grid_nm),
    )
    transport = (
        _anisotropic_msd(
            gro_path=Path(gro_path),
            xtc_path=xtc_path,
            top_path=Path(top_path),
            system_dir=Path(system_dir),
            out_dir=out_dir,
            frame_stride=stride,
            analysis_profile=str(analysis_profile),
        )
        if bool(compute_transport)
        else {"available": False, "reason": "disabled"}
    )

    geometry_health = {
        "phase_order": phase_order,
        "trajectory_phase_order": [str(x) for x in trajectory_phase_order],
        "phase_order_source": phase_order_source,
        "expected_phase_order": [str(x) for x in phase_groups_norm],
        "phase_order_ok": phase_order == [str(x) for x in phase_groups_norm],
        "adjacent_interfaces": adjacent_interfaces,
        "min_interphase_distance_nm": min_distances,
        "direct_graphite_electrolyte_contact": bool(direct_graphite_electrolyte_contact),
        "severe_overlap": bool(
            min((float(v) for v in min_distances.values() if v is not None), default=1.0) < 0.055
        ),
        "box_nm": [float(x) for x in final_box],
        "frame_count_analyzed": int(len(frames)),
        "last_frame_time_ps": float(final_time),
    }
    _write_json(out_dir / "geometry_health.json", geometry_health)
    edl = _edl_diagnostics(phase_groups=phase_groups_norm, density_arrays=density_arrays, phase_stats=phase_stats)
    potential = _charge_potential_profiles(
        phase_groups=phase_groups_norm,
        density_arrays=density_arrays,
        potential_reference=str(potential_reference),
    )
    edl["charge_potential"] = {key: value for key, value in potential.items() if key != "rows"}
    edl["available"] = bool(edl.get("available") or potential.get("available"))
    edl["split_electrodes"] = bool(split_electrodes)
    edl["report_potential_drop"] = bool(report_potential_drop)
    edl_species_csv = out_dir / "edl_species_profiles.csv"
    integrated_charge_csv = out_dir / "integrated_charge.csv"
    potential_csv = out_dir / "electrostatic_potential.csv"
    _write_rows_csv(edl_species_csv, _edl_species_rows(profile_rows))
    potential_rows = list(potential.get("rows") or []) if isinstance(potential, dict) else []
    _write_rows_csv(
        integrated_charge_csv,
        [
            {
                "z_nm": row.get("z_nm"),
                "charge_density_e_nm3": row.get("charge_density_e_nm3"),
                "integrated_charge_e_nm2": row.get("integrated_charge_e_nm2"),
            }
            for row in potential_rows
        ],
        fields=("z_nm", "charge_density_e_nm3", "integrated_charge_e_nm2"),
    )
    _write_rows_csv(
        potential_csv,
        potential_rows,
        fields=("z_nm", "charge_density_e_nm3", "integrated_charge_e_nm2", "electric_field_V_m", "electrostatic_potential_V"),
    )
    _write_json(out_dir / "edl_summary.json", edl)
    _write_rows_csv(out_dir / "penetration_events.csv", penetration.get("rows") or [])
    _write_json(out_dir / "penetration_summary.json", {key: value for key, value in penetration.items() if key != "rows"})
    penetration_depth_svg = _write_fraction_bar_svg(
        out_dir / "penetration_depth.svg",
        penetration.get("summary_by_species") or {},
        "polymer_frame_fraction",
        "polymer-region frame fraction",
        "Molecular penetration into polymer-rich regions",
    )
    _write_rows_csv(out_dir / "adsorption_events.csv", adsorption.get("rows") or [])
    _write_rows_csv(out_dir / "adsorption_surface_map.csv", adsorption.get("surface_map_rows") or [])
    _write_rows_csv(
        out_dir / "adsorbed_orientation.csv",
        [
            {
                key: row.get(key)
                for key in (
                    "time_ps",
                    "moltype",
                    "instance_index",
                    "nearest_graphite_phase",
                    "nearest_graphite_side",
                    "surface_distance_nm",
                    "adsorbed",
                    "orientation_available",
                    "carbonyl_angle_deg",
                    "dipole_proxy_angle_deg",
                )
            }
            for row in adsorption.get("rows", [])
        ],
    )
    _write_json(
        out_dir / "adsorption_summary.json",
        {key: value for key, value in adsorption.items() if key not in {"rows", "surface_map_rows"}},
    )
    adsorbed_orientation_svg = _write_fraction_bar_svg(
        out_dir / "adsorbed_orientation.svg",
        adsorption.get("summary_by_species") or {},
        "adsorbed_frame_fraction",
        "adsorbed frame fraction",
        "Graphite-near adsorption occupancy",
    )
    _write_json(out_dir / "region_transport_summary.json", transport)
    coord_rows: list[dict[str, Any]] = []
    if isinstance(coordination, dict):
        for region, payload in (coordination.get("by_region") or {}).items():
            frac = payload.get("state_fraction") or {}
            contacts = payload.get("mean_contacts") or {}
            coord_rows.append(
                {
                    "region": region,
                    "polymer_bound_fraction": frac.get("polymer_bound", 0.0),
                    "solvent_bound_fraction": frac.get("solvent_bound", 0.0),
                    "anion_paired_fraction": frac.get("anion_paired", 0.0),
                    "mixed_fraction": frac.get("mixed", 0.0),
                    "free_like_fraction": frac.get("free_like", 0.0),
                    "mean_polymer_o_contacts": contacts.get("polymer_o", 0.0),
                    "mean_solvent_o_contacts": contacts.get("solvent_o", 0.0),
                    "mean_anion_f_contacts": contacts.get("anion_f", 0.0),
                }
            )
    _write_rows_csv(out_dir / "coordination_z_profile.csv", coord_rows)
    region_summary = {
        "regions": regions,
        "phase_stats": phase_stats,
        "interpenetration": interpenetration,
        "enrichment": enrichment,
        "edl_diagnostics": edl,
        "penetration": {key: value for key, value in penetration.items() if key != "rows"},
        "adsorption": {key: value for key, value in adsorption.items() if key not in {"rows", "surface_map_rows"}},
    }

    _write_json(out_dir / "region_summary.json", region_summary)
    _write_json(out_dir / "coordination_by_region.json", coordination)
    summary = {
        "analysis_profile": str(analysis_profile),
        "parameters": {
            "bin_nm": float(bin_nm),
            "frame_stride": int(stride),
            "region_width_nm": float(region_width_nm),
            "surface_grid_nm": float(surface_grid_nm),
            "surface_distance_nm": float(surface_distance_nm),
            "penetration_threshold_nm": float(penetration_threshold_nm),
            "adsorption_min_residence_ps": float(adsorption_min_residence_ps),
            "potential_reference": str(potential_reference),
            "split_electrodes": bool(split_electrodes),
            "report_potential_drop": bool(report_potential_drop),
            "penetration_species": None if penetration_species is None else [str(x) for x in penetration_species],
            "adsorption_species": None if adsorption_species is None else [str(x) for x in adsorption_species],
            "phase_groups": [str(x) for x in phase_groups],
        },
        "inputs": {
            "gro_path": str(gro_path),
            "xtc_path": None if xtc_path is None else str(xtc_path),
            "top_path": str(top_path),
            "ndx_path": str(ndx_path),
            "system_dir": str(system_dir),
            "manifest_path": None if manifest_path is None else str(manifest_path),
        },
        "geometry_health": geometry_health,
        "region_summary": region_summary,
        "coordination_by_region": coordination,
        "anisotropic_msd_summary": transport,
        "edl_profiles": edl,
        "penetration": {key: value for key, value in penetration.items() if key != "rows"},
        "graphite_adsorption": {key: value for key, value in adsorption.items() if key not in {"rows", "surface_map_rows"}},
        "region_transport_summary": transport,
        "outputs": {
            "z_density_profiles_csv": str(profile_csv),
            "z_profiles_svg": None if z_profiles_svg is None else str(z_profiles_svg),
            "charge_density_profiles_csv": str(charge_profile_csv),
            "phase_z_quantiles_csv": str(phase_quantiles_csv),
            "geometry_health_json": str(out_dir / "geometry_health.json"),
            "edl_species_profiles_csv": str(edl_species_csv),
            "integrated_charge_csv": str(integrated_charge_csv),
            "electrostatic_potential_csv": str(potential_csv),
            "edl_summary_json": str(out_dir / "edl_summary.json"),
            "penetration_events_csv": str(out_dir / "penetration_events.csv"),
            "penetration_summary_json": str(out_dir / "penetration_summary.json"),
            "penetration_depth_svg": None if penetration_depth_svg is None else str(penetration_depth_svg),
            "adsorption_summary_json": str(out_dir / "adsorption_summary.json"),
            "adsorption_events_csv": str(out_dir / "adsorption_events.csv"),
            "adsorbed_orientation_csv": str(out_dir / "adsorbed_orientation.csv"),
            "adsorbed_orientation_svg": None if adsorbed_orientation_svg is None else str(adsorbed_orientation_svg),
            "region_transport_summary_json": str(out_dir / "region_transport_summary.json"),
            "region_summary_json": str(out_dir / "region_summary.json"),
            "coordination_by_region_json": str(out_dir / "coordination_by_region.json"),
            "coordination_z_profile_csv": str(out_dir / "coordination_z_profile.csv"),
            "anisotropic_msd_summary_json": str(out_dir / "anisotropic_msd_summary.json"),
            "interface_profile_summary_json": str(out_dir / "interface_profile_summary.json"),
        },
    }
    _write_json(out_dir / "interface_profile_summary.json", summary)
    return _jsonify(summary)


__all__ = ["compute_interface_profile"]
