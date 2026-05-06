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
    for trj_raw in md.iterload(str(xtc_path), top=str(gro_path), chunk=int(max(1, chunk))):
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
        density_arrays[label] = {
            "z_mid_nm": 0.5 * (bins[:-1] + bins[1:]),
            "mass_density_g_cm3": mass_mean * _AMU_PER_NM3_TO_G_CM3 / volume_mean,
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
    analysis_profile: str = "interface_fast",
    phase_groups: Sequence[str] = ("GRAPHITE", "POLYMER", "ELECTROLYTE"),
    compute_transport: bool = True,
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

    phase_stats: dict[str, dict[str, Any]] = {}
    for phase, mask in phase_masks.items():
        phase_stats[phase] = _z_quantiles(final_coords[mask, 2])
        phase_stats[phase]["atom_count"] = int(np.sum(mask))
    phase_order = sorted(
        [name for name in phase_groups if phase_stats.get(str(name), {}).get("p50_z_nm") is not None],
        key=lambda name: float(phase_stats[str(name)]["p50_z_nm"]),
    )

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

    graphite, polymer, electrolyte = [str(x) for x in phase_groups[:3]]
    min_gp = _min_distance_between(final_coords, phase_masks[graphite], phase_masks[polymer], final_box)
    min_pe = _min_distance_between(final_coords, phase_masks[polymer], phase_masks[electrolyte], final_box)
    min_ge = _min_distance_between(final_coords, phase_masks[graphite], phase_masks[electrolyte], final_box)
    regions = _regions(
        phase_stats=phase_stats,
        density_arrays=density_arrays,
        box_z_nm=float(final_box[2]),
        region_width_nm=float(region_width_nm),
        phase_groups=phase_groups,
    )
    interpenetration = _interpenetration(
        density_arrays=density_arrays,
        polymer=polymer,
        electrolyte=electrolyte,
        graphite=graphite,
        phase_stats=phase_stats,
        min_graphite_electrolyte_nm=min_ge,
    )
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
        "expected_phase_order": [str(x) for x in phase_groups],
        "phase_order_ok": phase_order == [str(x) for x in phase_groups],
        "min_interphase_distance_nm": {
            f"{graphite}-{polymer}": min_gp,
            f"{polymer}-{electrolyte}": min_pe,
            f"{graphite}-{electrolyte}": min_ge,
        },
        "direct_graphite_electrolyte_contact": bool(min_ge is not None and float(min_ge) < 0.35),
        "severe_overlap": bool(
            min((x for x in (min_gp, min_pe, min_ge) if x is not None), default=1.0) < 0.055
        ),
        "box_nm": [float(x) for x in final_box],
        "frame_count_analyzed": int(len(frames)),
        "last_frame_time_ps": float(final_time),
    }
    region_summary = {
        "regions": regions,
        "phase_stats": phase_stats,
        "interpenetration": interpenetration,
        "enrichment": enrichment,
    }

    (out_dir / "region_summary.json").write_text(json.dumps(_jsonify(region_summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "coordination_by_region.json").write_text(json.dumps(_jsonify(coordination), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "analysis_profile": str(analysis_profile),
        "parameters": {
            "bin_nm": float(bin_nm),
            "frame_stride": int(stride),
            "region_width_nm": float(region_width_nm),
            "surface_grid_nm": float(surface_grid_nm),
            "phase_groups": [str(x) for x in phase_groups],
        },
        "inputs": {
            "gro_path": str(gro_path),
            "xtc_path": None if xtc_path is None else str(xtc_path),
            "top_path": str(top_path),
            "ndx_path": str(ndx_path),
            "system_dir": str(system_dir),
        },
        "geometry_health": geometry_health,
        "region_summary": region_summary,
        "coordination_by_region": coordination,
        "anisotropic_msd_summary": transport,
        "outputs": {
            "z_density_profiles_csv": str(profile_csv),
            "region_summary_json": str(out_dir / "region_summary.json"),
            "coordination_by_region_json": str(out_dir / "coordination_by_region.json"),
            "anisotropic_msd_summary_json": str(out_dir / "anisotropic_msd_summary.json"),
            "interface_profile_summary_json": str(out_dir / "interface_profile_summary.json"),
        },
    }
    (out_dir / "interface_profile_summary.json").write_text(
        json.dumps(_jsonify(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return _jsonify(summary)


__all__ = ["compute_interface_profile"]
