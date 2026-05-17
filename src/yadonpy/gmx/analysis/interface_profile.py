"""Interface-specific profile statistics for graphite/polymer/electrolyte stacks."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..topology import SystemTopology, parse_system_top
from .structured import build_msd_metric_catalog, build_species_catalog, compute_msd_series, detect_first_shell


_AVOGADRO = 6.02214076e23
_AMU_PER_NM3_TO_G_CM3 = 1.66053906660e-3
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_EPS0_F_M = 8.8541878128e-12


@dataclass(frozen=True)
class InterfaceTimeSeriesOptions:
    """Controls optional decile-sampled interface animation outputs."""

    enabled: bool = False
    sample_count: int = 10
    fps: float = 1.0
    rdf: bool = True
    concentration: bool = True
    angles: bool = True
    rdf_rmax_nm: float = 1.2
    rdf_bin_nm: float = 0.02

    @classmethod
    def from_parameters(
        cls,
        *,
        time_series_analysis: bool = False,
        time_series_sample_count: int = 10,
        time_series_fps: float = 1.0,
        time_series_rdf: bool = True,
        time_series_concentration: bool = True,
        time_series_angles: bool = True,
        time_series_rdf_rmax_nm: float = 1.2,
        time_series_rdf_bin_nm: float = 0.02,
    ) -> "InterfaceTimeSeriesOptions":
        return cls(
            enabled=bool(time_series_analysis),
            sample_count=int(max(1, time_series_sample_count)),
            fps=float(time_series_fps),
            rdf=bool(time_series_rdf),
            concentration=bool(time_series_concentration),
            angles=bool(time_series_angles),
            rdf_rmax_nm=float(time_series_rdf_rmax_nm),
            rdf_bin_nm=float(time_series_rdf_bin_nm),
        )


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


def _write_membrane_permeation_svg(path: Path, summary_by_species: dict[str, Any]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    labels = [str(k) for k in summary_by_species.keys()]
    if not labels:
        return None
    feed = [100.0 * float((summary_by_species.get(k) or {}).get("feed_frame_fraction") or 0.0) for k in labels]
    membrane = [100.0 * float((summary_by_species.get(k) or {}).get("membrane_frame_fraction") or 0.0) for k in labels]
    permeate = [100.0 * float((summary_by_species.get(k) or {}).get("permeate_frame_fraction") or 0.0) for k in labels]
    entry = [100.0 * float((summary_by_species.get(k) or {}).get("entry_event_fraction_per_initial_feed") or 0.0) for k in labels]
    trans = [
        100.0 * float((summary_by_species.get(k) or {}).get("translocation_fraction_per_initial_feed") or 0.0)
        for k in labels
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(max(7.0, 0.8 * len(labels)), 3.2))
    axes[0].bar(labels, feed, label="feed")
    axes[0].bar(labels, membrane, bottom=feed, label="membrane")
    axes[0].bar(labels, permeate, bottom=[a + b for a, b in zip(feed, membrane)], label="permeate")
    axes[0].set_ylabel("sampled molecule-frame fraction / %")
    axes[0].set_ylim(0.0, 100.0)
    axes[0].set_title("Permeant state occupancy")
    axes[0].legend(loc="best", fontsize="small")
    width = 0.38
    xpos = np.arange(len(labels), dtype=float)
    axes[1].bar((xpos - 0.5 * width).tolist(), entry, width=width, label="entry")
    axes[1].bar((xpos + 0.5 * width).tolist(), trans, width=width, label="translocation")
    axes[1].set_xticks(xpos.tolist(), labels)
    axes[1].set_ylabel("fraction of initially fed molecules / %")
    axes[1].set_ylim(0.0, max(100.0, max(entry + trans) * 1.15 if (entry or trans) else 100.0))
    axes[1].set_title("Crossing-event yield")
    axes[1].legend(loc="best", fontsize="small")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_membrane_timeseries_svg(path: Path, rows: Sequence[dict[str, Any]]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    by_species: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_species.setdefault(str(row.get("moltype") or ""), []).append(dict(row))
    if not by_species:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharex=True)
    for moltype, series in by_species.items():
        series = sorted(series, key=lambda item: float(item.get("time_ps") or 0.0))
        t_ns = [float(item.get("time_ps") or 0.0) / 1000.0 for item in series]
        membrane_count = [float(item.get("membrane_count") or 0.0) for item in series]
        cumulative_entries = [float(item.get("cumulative_entry_events") or 0.0) for item in series]
        axes[0].plot(t_ns, membrane_count, marker="o", linewidth=1.2, label=moltype)
        axes[1].plot(t_ns, cumulative_entries, marker="o", linewidth=1.2, label=moltype)
    axes[0].set_ylabel("molecules in membrane")
    axes[0].set_title("Membrane uptake vs time")
    axes[1].set_ylabel("cumulative entries")
    axes[1].set_title("Entry events")
    for ax in axes:
        ax.set_xlabel("time / ns")
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_penetration_depth_distribution(
    *,
    out_csv: Path,
    out_svg: Path,
    rows: Sequence[dict[str, Any]],
    bin_nm: float = 0.10,
) -> tuple[Path, Path | None]:
    by_species: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        moltype = str(row.get("moltype") or "")
        if moltype:
            by_species.setdefault(moltype, []).append(dict(row))
    max_depth = max(
        [
            float(row.get("polymer_region_depth_nm") or 0.0)
            for species_rows in by_species.values()
            for row in species_rows
            if row.get("polymer_region_depth_nm") is not None
        ]
        or [0.0]
    )
    step = max(float(bin_nm), 1.0e-6)
    hi = max(step, math.ceil(max_depth / step) * step)
    edges = np.arange(0.0, hi + 0.5 * step, step)
    if edges.size < 2:
        edges = np.asarray([0.0, step], dtype=float)
    out_rows: list[dict[str, Any]] = []
    for moltype, species_rows in sorted(by_species.items()):
        sample_count = max(1, len(species_rows))
        depths = np.asarray(
            [
                float(row.get("polymer_region_depth_nm"))
                for row in species_rows
                if row.get("polymer_region_depth_nm") is not None and float(row.get("polymer_region_depth_nm") or 0.0) > 0.0
            ],
            dtype=float,
        )
        hist = np.histogram(depths, bins=edges)[0].astype(float)
        penetrated = max(1, int(np.sum(hist)))
        for i, count in enumerate(hist.tolist()):
            out_rows.append(
                {
                    "moltype": moltype,
                    "depth_lo_nm": float(edges[i]),
                    "depth_hi_nm": float(edges[i + 1]),
                    "depth_mid_nm": float(0.5 * (edges[i] + edges[i + 1])),
                    "sample_frame_count": int(sample_count),
                    "penetrated_frame_count": int(np.sum(hist)),
                    "bin_frame_count": int(count),
                    "percent_of_all_sampled_frames": float(100.0 * count / sample_count),
                    "percent_of_penetrated_frames": float(100.0 * count / penetrated) if np.sum(hist) > 0 else 0.0,
                }
            )
    _write_rows_csv(out_csv, out_rows)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return out_csv, None
    if not out_rows:
        return out_csv, None
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for moltype in sorted(by_species):
        series = [row for row in out_rows if row.get("moltype") == moltype]
        x = [float(row["depth_mid_nm"]) for row in series]
        y = [float(row["percent_of_all_sampled_frames"]) for row in series]
        if any(val > 0.0 for val in y):
            ax.step(x, y, where="mid", linewidth=1.8, label=moltype)
    ax.set_xlabel("COM depth inside CMC/polymer-rich region / nm")
    ax.set_ylabel("sampled molecule-frame fraction / %")
    ax.set_title("Penetration depth distribution")
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_svg)
    plt.close(fig)
    return out_csv, out_svg


def _write_adsorbed_orientation_distribution(
    *,
    out_csv: Path,
    out_svg: Path,
    rows: Sequence[dict[str, Any]],
    bin_deg: float = 15.0,
) -> tuple[Path, Path | None]:
    selected = [dict(row) for row in rows if bool(row.get("adsorbed")) and bool(row.get("orientation_available"))]
    edges = np.arange(0.0, 180.0 + float(bin_deg), float(bin_deg))
    if edges[-1] < 180.0:
        edges = np.append(edges, 180.0)
    angle_specs = (("carbonyl", "carbonyl_angle_deg"), ("dipole_proxy", "dipole_proxy_angle_deg"))
    out_rows: list[dict[str, Any]] = []
    moltypes = sorted({str(row.get("moltype") or "") for row in selected if row.get("moltype")})
    for moltype in moltypes:
        species_rows = [row for row in selected if str(row.get("moltype") or "") == moltype]
        for angle_kind, key in angle_specs:
            angles = np.asarray(
                [float(row[key]) for row in species_rows if row.get(key) is not None and np.isfinite(float(row[key]))],
                dtype=float,
            )
            hist = np.histogram(angles, bins=edges)[0].astype(float)
            denom = max(1, int(np.sum(hist)))
            for i, count in enumerate(hist.tolist()):
                out_rows.append(
                    {
                        "moltype": moltype,
                        "orientation_kind": angle_kind,
                        "angle_lo_deg": float(edges[i]),
                        "angle_hi_deg": float(edges[i + 1]),
                        "angle_mid_deg": float(0.5 * (edges[i] + edges[i + 1])),
                        "adsorbed_oriented_sample_count": int(np.sum(hist)),
                        "bin_sample_count": int(count),
                        "percent_of_adsorbed_oriented_frames": float(100.0 * count / denom) if np.sum(hist) > 0 else 0.0,
                    }
                )
    _write_rows_csv(out_csv, out_rows)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return out_csv, None
    if not out_rows:
        return out_csv, None
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, (angle_kind, _key) in zip(axes, angle_specs):
        for moltype in moltypes:
            series = [
                row
                for row in out_rows
                if row.get("moltype") == moltype and row.get("orientation_kind") == angle_kind
            ]
            x = [float(row["angle_mid_deg"]) for row in series]
            y = [float(row["percent_of_adsorbed_oriented_frames"]) for row in series]
            if any(val > 0.0 for val in y):
                ax.step(x, y, where="mid", linewidth=1.8, label=moltype)
        ax.set_title("carbonyl C->O" if angle_kind == "carbonyl" else "charge-dipole proxy")
        ax.set_xlabel("angle to graphite surface normal / deg")
        ax.set_xlim(0.0, 180.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="best", fontsize="small")
    axes[0].set_ylabel("adsorbed EDL oriented-frame fraction / %")
    fig.suptitle("Adsorbed orientation distribution within graphite EDL cutoff")
    fig.tight_layout()
    fig.savefig(out_svg)
    plt.close(fig)
    return out_csv, out_svg


def _mp4_writer(fps: float):
    try:
        from matplotlib import animation
    except Exception:
        return None
    try:
        try:
            import imageio_ffmpeg
            import matplotlib as mpl

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_exe:
                mpl.rcParams["animation.ffmpeg_path"] = str(ffmpeg_exe)
        except Exception:
            pass
        if not animation.writers.is_available("ffmpeg"):
            return None
        return animation.FFMpegWriter(fps=max(1, int(round(float(fps)))), bitrate=1800)
    except Exception:
        return None


def _animation_result(*, path: Path | None, reason: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mp4": None if path is None else str(path),
        "available": bool(path is not None and Path(path).exists()),
    }
    if reason:
        payload["reason"] = str(reason)
    if extra:
        payload.update(extra)
    return payload


def _save_animation_png_frames(
    *,
    fig: Any,
    update,
    frame_count: int,
    frame_dir: Path,
    dpi: int = 160,
) -> dict[str, Any]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame_idx in range(max(0, int(frame_count))):
        update(int(frame_idx))
        fig.savefig(frame_dir / f"frame_{frame_idx:03d}.png", dpi=int(dpi))
        count += 1
    return {"frames_dir": str(frame_dir), "frame_png_count": int(count)}


def _time_windows(
    frames: Sequence[tuple[float, np.ndarray, tuple[float, float, float]]],
    *,
    sample_count: int = 10,
) -> list[dict[str, Any]]:
    if len(frames) < 2:
        return []
    target = min(max(1, int(sample_count or 10)), len(frames))
    times = np.asarray([float(item[0]) for item in frames], dtype=float)
    if not np.all(np.isfinite(times)) or float(np.max(times) - np.min(times)) <= 0.0:
        edges_idx = np.linspace(0, len(frames), min(target, len(frames)) + 1, dtype=int)
        windows = []
        for i in range(len(edges_idx) - 1):
            lo = int(edges_idx[i])
            hi = int(max(edges_idx[i + 1], lo + 1))
            subset = list(frames[lo:hi])
            if subset:
                windows.append(
                    {
                        "window_index": len(windows),
                        "time_start_ps": float(subset[0][0]),
                        "time_end_ps": float(subset[-1][0]),
                        "frames": subset,
                    }
                )
        return windows

    t0 = float(times[0])
    t1 = float(times[-1])
    edges = np.linspace(t0, t1, target + 1)
    windows: list[dict[str, Any]] = []
    used_nearest: set[int] = set()
    for i in range(target):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == target - 1:
            idx = np.where((times >= lo) & (times <= hi))[0]
        else:
            idx = np.where((times >= lo) & (times < hi))[0]
        if idx.size == 0:
            mid = 0.5 * (lo + hi)
            nearest = int(np.argmin(np.abs(times - mid)))
            if nearest in used_nearest:
                continue
            idx = np.asarray([nearest], dtype=int)
            used_nearest.add(nearest)
        subset = [frames[int(j)] for j in idx.tolist()]
        windows.append(
            {
                "window_index": len(windows),
                "time_start_ps": float(lo),
                "time_end_ps": float(hi),
                "frames": subset,
            }
        )
    return windows


def _mobile_instance(inst: dict[str, Any]) -> bool:
    moltype = str(inst.get("moltype") or "")
    kind = str(inst.get("kind") or "").lower()
    label = f"{moltype} {kind}".lower()
    return bool(moltype) and not any(tok in label for tok in ("graph", "graphite", "substrate", "wall"))


def _selected_mobile_moltypes(instances: Sequence[dict[str, Any]], *, max_moltypes: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for inst in instances:
        if not _mobile_instance(dict(inst)):
            continue
        moltype = str(inst.get("moltype") or "")
        if moltype not in counts:
            order.append(moltype)
            counts[moltype] = 0
        counts[moltype] += 1
    order.sort(key=lambda name: (-int(counts.get(name, 0)), name))
    return order[: max(1, int(max_moltypes))]


def _instance_com(coords: np.ndarray, inst: dict[str, Any]) -> np.ndarray:
    idx = np.asarray(inst["atom_indices_0"], dtype=int)
    masses = np.asarray(inst.get("masses"), dtype=float)
    if masses.size != idx.size or float(np.sum(masses)) <= 0.0:
        masses = np.ones(idx.size, dtype=float)
    return np.asarray(np.average(coords[idx], axis=0, weights=masses), dtype=float)


def _write_concentration_timeseries(
    *,
    out_dir: Path,
    windows: Sequence[dict[str, Any]],
    bins: np.ndarray,
    instances: Sequence[dict[str, Any]],
    fps: float,
    max_moltypes: int = 6,
) -> dict[str, Any]:
    labels = _selected_mobile_moltypes(instances, max_moltypes=max_moltypes)
    if not windows or not labels:
        return _animation_result(path=None, reason="no_time_windows_or_mobile_species")
    z_mid = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    profiles: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    by_label = {label: [inst for inst in instances if str(inst.get("moltype") or "") == label] for label in labels}
    for win in windows:
        density: dict[str, np.ndarray] = {label: np.zeros(len(widths), dtype=float) for label in labels}
        volume = np.zeros(len(widths), dtype=float)
        win_frames = list(win.get("frames") or [])
        for _time_ps, coords, box in win_frames:
            volume += float(box[0]) * float(box[1]) * widths
            for label, label_instances in by_label.items():
                z_vals = [_instance_com(coords, inst)[2] for inst in label_instances]
                if z_vals:
                    density[label] += np.histogram(np.asarray(z_vals, dtype=float), bins=bins)[0]
        volume = np.maximum(volume, 1.0e-12)
        for label in labels:
            density[label] = density[label] / volume
            for i, value in enumerate(density[label].tolist()):
                rows.append(
                    {
                        "window_index": int(win["window_index"]),
                        "time_start_ps": float(win["time_start_ps"]),
                        "time_end_ps": float(win["time_end_ps"]),
                        "moltype": label,
                        "z_mid_nm": float(z_mid[i]),
                        "concentration_nm3": float(value),
                    }
                )
        profiles.append({"window": dict(win), "density": density})

    csv_path = out_dir / "time_series" / "z_concentration_timeseries.csv"
    _write_rows_csv(csv_path, rows)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _animation_result(path=None, reason="matplotlib_unavailable", extra={"csv": str(csv_path)})
    mp4_path = out_dir / "time_series" / "z_concentration_timeseries.mp4"
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    ymax = max([float(np.max(p["density"][label])) for p in profiles for label in labels] or [1.0])
    ymax = max(1.0e-12, ymax) * 1.15
    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    def _update(frame_idx: int):
        ax.clear()
        payload = profiles[int(frame_idx)]
        win = payload["window"]
        for label in labels:
            ax.plot(z_mid, payload["density"][label], label=label, lw=1.8)
        ax.set_xlim(float(z_mid[0]), float(z_mid[-1]))
        ax.set_ylim(0.0, ymax)
        ax.set_xlabel("z / nm")
        ax.set_ylabel("molecule COM concentration / nm$^{-3}$")
        ax.set_title(f"z concentration, {float(win['time_start_ps']):.1f}-{float(win['time_end_ps']):.1f} ps")
        ax.legend(loc="best", fontsize="small", ncols=2)
        fig.tight_layout()

    frame_meta = _save_animation_png_frames(
        fig=fig,
        update=_update,
        frame_count=len(profiles),
        frame_dir=out_dir / "time_series" / "frames" / "z_concentration",
    )
    extra = {"csv": str(csv_path), "sample_windows": len(profiles), "moltypes": labels, **frame_meta}
    writer = _mp4_writer(fps)
    if writer is None:
        plt.close(fig)
        return _animation_result(path=None, reason="ffmpeg_writer_unavailable", extra=extra)
    try:
        from matplotlib.animation import FuncAnimation
    except Exception:
        plt.close(fig)
        return _animation_result(path=None, reason="matplotlib_unavailable", extra=extra)
    ani = FuncAnimation(fig, _update, frames=len(profiles), interval=1000)
    try:
        ani.save(mp4_path, writer=writer)
    except Exception as exc:
        plt.close(fig)
        return _animation_result(
            path=None,
            reason=f"mp4_write_failed:{exc.__class__.__name__}",
            extra=extra,
        )
    plt.close(fig)
    return _animation_result(path=mp4_path, extra=extra)


def _rdf_curve_from_frames(
    *,
    frames: Sequence[tuple[float, np.ndarray, tuple[float, float, float]]],
    center_indices: np.ndarray,
    target_indices: np.ndarray,
    r_max_nm: float,
    bin_nm: float,
) -> dict[str, Any]:
    center = np.asarray(sorted(set(int(i) for i in center_indices)), dtype=int)
    target = np.asarray(sorted(set(int(i) for i in target_indices)), dtype=int)
    nbins = max(8, int(np.ceil(float(r_max_nm) / float(bin_nm))))
    edges = np.linspace(0.0, float(r_max_nm), nbins + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(nbins, dtype=float)
    volume_samples: list[float] = []
    effective_ref = 0.0
    if center.size == 0 or target.size == 0 or not frames:
        cn = np.zeros_like(r_mid)
        return {"r_nm": r_mid, "g_r": cn.copy(), "cn_curve": cn, "shell": detect_first_shell(r_mid, cn, cn)}
    pair_mask = center[:, None] != target[None, :]
    for _time_ps, coords, box in frames:
        volume_samples.append(float(box[0]) * float(box[1]) * float(box[2]))
        c = np.asarray(coords[center], dtype=float)
        t = np.asarray(coords[target], dtype=float)
        delta = c[:, None, :] - t[None, :, :]
        box_arr = np.asarray(box, dtype=float)
        delta -= box_arr[None, None, :] * np.round(delta / np.maximum(box_arr[None, None, :], 1.0e-12))
        if not np.any(pair_mask):
            continue
        dist = np.linalg.norm(delta[pair_mask], axis=1)
        hist += np.histogram(dist, bins=edges)[0]
        effective_ref += float(center.size)
    rho_target = float(target.size) / max(float(np.mean(volume_samples)) if volume_samples else 0.0, 1.0e-12)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    denom = max(float(effective_ref) * max(rho_target, 1.0e-12), 1.0e-12)
    g_r = hist / (denom * shell_vol)
    dr = np.gradient(r_mid) if r_mid.size else np.zeros_like(r_mid)
    cn_curve = 4.0 * np.pi * rho_target * np.cumsum(g_r * r_mid * r_mid * dr)
    shell = detect_first_shell(r_mid, g_r, cn_curve)
    return {
        "r_nm": r_mid,
        "g_r": g_r,
        "cn_curve": cn_curve,
        "rho_target_nm3": float(rho_target),
        "shell": shell,
    }


def _representative_site(inst: dict[str, Any], sign: int) -> dict[str, Any] | None:
    idx = np.asarray(inst.get("atom_indices_0"), dtype=int)
    if idx.size == 0:
        return None
    charges = np.asarray(inst.get("charges"), dtype=float)
    if charges.size != idx.size:
        charges = np.zeros(idx.size, dtype=float)
    atomnames = [str(x) for x in inst.get("atomnames") or []]
    atomtypes = [str(x) for x in inst.get("atomtypes") or []]
    if sign > 0:
        local = int(np.argmax(charges)) if charges.size else 0
        charge = float(charges[local]) if charges.size else 0.0
        if charge <= 0.05 and idx.size > 1:
            return None
    else:
        local = int(np.argmin(charges)) if charges.size else 0
        charge = float(charges[local]) if charges.size else 0.0
        if charge >= -0.05 and idx.size > 1:
            return None
    name = atomnames[local] if local < len(atomnames) else f"atom{local + 1}"
    atype = atomtypes[local] if local < len(atomtypes) else ""
    return {
        "local_index": int(local),
        "atom_name": str(name),
        "atom_type": str(atype),
        "charge_e": float(charge),
    }


def _edl_rdf_pair_specs(instances: Sequence[dict[str, Any]], *, max_pairs: int = 8) -> list[dict[str, Any]]:
    by_moltype: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for inst in instances:
        if not _mobile_instance(dict(inst)):
            continue
        moltype = str(inst.get("moltype") or "")
        if not moltype:
            continue
        if moltype not in by_moltype:
            by_moltype[moltype] = []
            order.append(moltype)
        by_moltype[moltype].append(dict(inst))
    order.sort(key=lambda name: (-len(by_moltype.get(name, [])), name))
    site_info: dict[str, dict[str, Any]] = {}
    positive: list[str] = []
    negative: list[str] = []
    for moltype in order:
        inst = by_moltype[moltype][0]
        pos = _representative_site(inst, +1)
        neg = _representative_site(inst, -1)
        formal = float(inst.get("formal_charge_e") or 0.0)
        site_info[moltype] = {"positive": pos, "negative": neg, "formal_charge_e": formal}
        label = f"{moltype} {inst.get('kind') or ''}".lower()
        if pos is not None and (formal > 0.0 or "li" in label or "na" in label or len(by_moltype[moltype][0].get("atom_indices_0", [])) == 1):
            positive.append(moltype)
        if neg is not None:
            negative.append(moltype)
    specs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()

    def add_pair(center: str, target: str, center_sign: int, target_sign: int) -> None:
        center_site = site_info.get(center, {}).get("positive" if center_sign > 0 else "negative")
        target_site = site_info.get(target, {}).get("positive" if target_sign > 0 else "negative")
        if center_site is None or target_site is None:
            return
        key = (center, target, int(center_site["local_index"]), int(target_site["local_index"]))
        if key in seen:
            return
        seen.add(key)
        specs.append(
            {
                "pair": f"{center}:{center_site['atom_name']}-{target}:{target_site['atom_name']}",
                "center_moltype": center,
                "target_moltype": target,
                "center_local_index": int(center_site["local_index"]),
                "target_local_index": int(target_site["local_index"]),
                "center_atom_name": str(center_site["atom_name"]),
                "target_atom_name": str(target_site["atom_name"]),
                "center_site_charge_e": float(center_site["charge_e"]),
                "target_site_charge_e": float(target_site["charge_e"]),
            }
        )

    for center in positive:
        for target in negative:
            if target == center:
                continue
            add_pair(center, target, +1, -1)
            if len(specs) >= int(max_pairs):
                return specs
    for center in negative:
        for target in positive:
            if target == center:
                continue
            add_pair(center, target, -1, +1)
            if len(specs) >= int(max_pairs):
                return specs
    return specs


def _edl_center_indices_for_frame(
    *,
    coords: np.ndarray,
    instances: Sequence[dict[str, Any]],
    moltype: str,
    local_index: int,
    surfaces: Sequence[dict[str, Any]],
    cutoff_nm: float,
) -> np.ndarray:
    out: list[int] = []
    for inst in instances:
        if str(inst.get("moltype") or "") != str(moltype):
            continue
        idx = np.asarray(inst.get("atom_indices_0"), dtype=int)
        local = int(local_index)
        if local < 0 or local >= idx.size:
            continue
        com = _instance_com(coords, inst)
        _surface, dist = _nearest_graphite_surface(float(com[2]), surfaces)
        if dist is not None and float(dist) <= float(cutoff_nm):
            out.append(int(idx[local]))
    return np.asarray(sorted(set(out)), dtype=int)


def _site_indices_for_moltype(
    instances: Sequence[dict[str, Any]],
    *,
    moltype: str,
    local_index: int,
) -> np.ndarray:
    out: list[int] = []
    local = int(local_index)
    for inst in instances:
        if str(inst.get("moltype") or "") != str(moltype):
            continue
        idx = np.asarray(inst.get("atom_indices_0"), dtype=int)
        if 0 <= local < idx.size:
            out.append(int(idx[local]))
    return np.asarray(sorted(set(out)), dtype=int)


def _rdf_curve_from_dynamic_edl_frames(
    *,
    frames: Sequence[tuple[float, np.ndarray, tuple[float, float, float]]],
    instances: Sequence[dict[str, Any]],
    surfaces: Sequence[dict[str, Any]],
    spec: dict[str, Any],
    surface_distance_nm: float,
    r_max_nm: float,
    bin_nm: float,
) -> dict[str, Any]:
    nbins = max(8, int(np.ceil(float(r_max_nm) / float(bin_nm))))
    edges = np.linspace(0.0, float(r_max_nm), nbins + 1)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(nbins, dtype=float)
    volume_samples: list[float] = []
    effective_ref = 0.0
    target_indices = _site_indices_for_moltype(
        instances,
        moltype=str(spec["target_moltype"]),
        local_index=int(spec["target_local_index"]),
    )
    for _time_ps, coords, box in frames:
        center_indices = _edl_center_indices_for_frame(
            coords=coords,
            instances=instances,
            moltype=str(spec["center_moltype"]),
            local_index=int(spec["center_local_index"]),
            surfaces=surfaces,
            cutoff_nm=float(surface_distance_nm),
        )
        if center_indices.size == 0 or target_indices.size == 0:
            continue
        volume_samples.append(float(box[0]) * float(box[1]) * float(box[2]))
        c = np.asarray(coords[center_indices], dtype=float)
        t = np.asarray(coords[target_indices], dtype=float)
        delta = c[:, None, :] - t[None, :, :]
        box_arr = np.asarray(box, dtype=float)
        delta -= box_arr[None, None, :] * np.round(delta / np.maximum(box_arr[None, None, :], 1.0e-12))
        pair_mask = center_indices[:, None] != target_indices[None, :]
        if not np.any(pair_mask):
            continue
        dist = np.linalg.norm(delta[pair_mask], axis=1)
        hist += np.histogram(dist, bins=edges)[0]
        effective_ref += float(center_indices.size)
    if effective_ref <= 0.0:
        cn = np.zeros_like(r_mid)
        return {
            "r_nm": r_mid,
            "g_r": cn.copy(),
            "cn_curve": cn,
            "edl_center_samples": 0,
            "target_site_count": int(target_indices.size),
            "shell": detect_first_shell(r_mid, cn, cn),
        }
    rho_target = float(target_indices.size) / max(float(np.mean(volume_samples)) if volume_samples else 0.0, 1.0e-12)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    denom = np.maximum(float(effective_ref) * max(rho_target, 1.0e-12) * shell_vol, 1.0e-12)
    g_r = hist / denom
    cn_curve = np.cumsum(hist) / max(float(effective_ref), 1.0e-12)
    shell = detect_first_shell(r_mid, g_r, cn_curve)
    return {
        "r_nm": r_mid,
        "g_r": g_r,
        "cn_curve": cn_curve,
        "rho_target_nm3": float(rho_target),
        "edl_center_samples": int(effective_ref),
        "target_site_count": int(target_indices.size),
        "shell": shell,
    }


def _write_edl_rdf_cn_timeseries(
    *,
    out_dir: Path,
    windows: Sequence[dict[str, Any]],
    instances: Sequence[dict[str, Any]],
    surfaces: Sequence[dict[str, Any]],
    surface_distance_nm: float,
    fps: float,
    r_max_nm: float = 1.2,
    bin_nm: float = 0.02,
    max_pairs: int = 8,
) -> dict[str, Any]:
    pair_specs = _edl_rdf_pair_specs(instances, max_pairs=max_pairs)
    if not windows or not surfaces or not pair_specs:
        return _animation_result(path=None, reason="no_edl_rdf_pairs_or_surfaces")
    curves: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    for win in windows:
        win_curves: dict[str, dict[str, Any]] = {}
        for spec in pair_specs:
            pair_id = str(spec["pair"])
            data = _rdf_curve_from_dynamic_edl_frames(
                frames=list(win.get("frames") or []),
                instances=instances,
                surfaces=surfaces,
                spec=spec,
                surface_distance_nm=float(surface_distance_nm),
                r_max_nm=float(r_max_nm),
                bin_nm=float(bin_nm),
            )
            win_curves[pair_id] = data
            r = np.asarray(data.get("r_nm"), dtype=float)
            g = np.asarray(data.get("g_r"), dtype=float)
            cn = np.asarray(data.get("cn_curve"), dtype=float)
            for i in range(int(r.size)):
                curve_rows.append(
                    {
                        "window_index": int(win["window_index"]),
                        "time_start_ps": float(win["time_start_ps"]),
                        "time_end_ps": float(win["time_end_ps"]),
                        "pair": pair_id,
                        "center_moltype": spec["center_moltype"],
                        "target_moltype": spec["target_moltype"],
                        "center_atom_name": spec["center_atom_name"],
                        "target_atom_name": spec["target_atom_name"],
                        "r_nm": float(r[i]),
                        "g_r": float(g[i]),
                        "coordination_number": float(cn[i]),
                    }
                )
            shell = dict(data.get("shell") or {})
            shell_rows.append(
                {
                    "window_index": int(win["window_index"]),
                    "time_start_ps": float(win["time_start_ps"]),
                    "time_end_ps": float(win["time_end_ps"]),
                    "pair": pair_id,
                    "center_moltype": spec["center_moltype"],
                    "target_moltype": spec["target_moltype"],
                    "center_atom_name": spec["center_atom_name"],
                    "target_atom_name": spec["target_atom_name"],
                    "edl_center_samples": data.get("edl_center_samples"),
                    "target_site_count": data.get("target_site_count"),
                    "r_peak_nm": shell.get("r_peak_nm"),
                    "r_shell_nm": shell.get("r_shell_nm"),
                    "cn_shell": shell.get("cn_shell"),
                    "confidence": shell.get("confidence"),
                    "status": shell.get("status"),
                }
            )
        curves.append({"window": dict(win), "curves": win_curves})
    curve_csv = out_dir / "time_series" / "edl_rdf_cn_curves_timeseries.csv"
    shell_csv = out_dir / "time_series" / "edl_rdf_cn_shell_timeseries.csv"
    _write_rows_csv(curve_csv, curve_rows)
    _write_rows_csv(shell_csv, shell_rows)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _animation_result(
            path=None,
            reason="matplotlib_unavailable",
            extra={"curves_csv": str(curve_csv), "shell_csv": str(shell_csv), "sample_windows": len(curves)},
        )
    mp4_path = out_dir / "time_series" / "edl_rdf_cn_timeseries.mp4"
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    ymax_g = max([float(np.max(payload["curves"][spec["pair"]]["g_r"])) for payload in curves for spec in pair_specs] or [1.0])
    fig, ax_g = plt.subplots(figsize=(7.2, 4.2))
    ax_cn = ax_g.twinx()

    def _update(frame_idx: int):
        ax_g.clear()
        ax_cn.clear()
        payload = curves[int(frame_idx)]
        win = payload["window"]
        for spec in pair_specs:
            pair_id = str(spec["pair"])
            data = payload["curves"][pair_id]
            r = np.asarray(data["r_nm"], dtype=float)
            g = np.asarray(data["g_r"], dtype=float)
            cn = np.asarray(data["cn_curve"], dtype=float)
            line = ax_g.plot(r, g, lw=1.6, label=pair_id)[0]
            ax_cn.plot(r, cn, lw=1.2, ls="--", color=line.get_color())
            shell = dict(data.get("shell") or {})
            r_peak = shell.get("r_peak_nm")
            if r_peak is not None and np.isfinite(float(r_peak)):
                y_peak = float(np.interp(float(r_peak), r, g)) if r.size else 0.0
                ax_g.axvline(float(r_peak), color=line.get_color(), ls=":", lw=0.8, alpha=0.65)
                ax_g.text(
                    float(r_peak),
                    y_peak,
                    f"{float(r_peak):.2f}",
                    color=line.get_color(),
                    fontsize=7,
                    rotation=90,
                    va="bottom",
                    ha="center",
                )
        ax_g.set_xlim(0.0, float(r_max_nm))
        ax_g.set_ylim(0.0, max(1.0, ymax_g * 1.15))
        ax_cn.set_ylim(0.0, 6.0)
        ax_g.set_xlabel("r / nm")
        ax_g.set_ylabel("EDL RDF g(r), solid")
        ax_cn.set_ylabel("CN(r), dashed")
        ax_g.set_title(
            f"Graphite EDL RDF/CN, {float(win['time_start_ps']):.1f}-{float(win['time_end_ps']):.1f} ps"
        )
        ax_g.legend(loc="upper right", fontsize="x-small", ncols=2)
        fig.tight_layout()

    frame_meta = _save_animation_png_frames(
        fig=fig,
        update=_update,
        frame_count=len(curves),
        frame_dir=out_dir / "time_series" / "frames" / "edl_rdf_cn",
    )
    extra = {
        "curves_csv": str(curve_csv),
        "shell_csv": str(shell_csv),
        "sample_windows": len(curves),
        "pairs": [str(spec["pair"]) for spec in pair_specs],
        "surface_distance_nm": float(surface_distance_nm),
        "cn_axis_ylim": [0.0, 6.0],
        **frame_meta,
    }
    writer = _mp4_writer(fps)
    if writer is None:
        plt.close(fig)
        return _animation_result(path=None, reason="ffmpeg_writer_unavailable", extra=extra)
    try:
        from matplotlib.animation import FuncAnimation
    except Exception:
        plt.close(fig)
        return _animation_result(path=None, reason="matplotlib_unavailable", extra=extra)
    ani = FuncAnimation(fig, _update, frames=len(curves), interval=1000)
    try:
        ani.save(mp4_path, writer=writer)
    except Exception as exc:
        plt.close(fig)
        return _animation_result(path=None, reason=f"mp4_write_failed:{exc.__class__.__name__}", extra=extra)
    plt.close(fig)
    return _animation_result(path=mp4_path, extra=extra)


def _write_rdf_cn_timeseries(
    *,
    out_dir: Path,
    windows: Sequence[dict[str, Any]],
    categories: dict[str, np.ndarray],
    fps: float,
    r_max_nm: float = 1.2,
    bin_nm: float = 0.02,
) -> dict[str, Any]:
    centers = np.asarray(categories.get("cation", categories.get("li", [])), dtype=int)
    pair_specs = [
        ("cation-polymer_o", "cation", "polymer O", np.asarray(categories.get("polymer_o", []), dtype=int)),
        ("cation-solvent_o", "cation", "solvent O", np.asarray(categories.get("solvent_o", []), dtype=int)),
        ("cation-anion_f", "cation", "anion F", np.asarray(categories.get("anion_f", []), dtype=int)),
    ]
    pair_specs = [spec for spec in pair_specs if centers.size and spec[3].size]
    if not windows or not pair_specs:
        return _animation_result(path=None, reason="no_time_windows_or_rdf_pairs")
    curves: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    for win in windows:
        win_curves: dict[str, dict[str, Any]] = {}
        for pair_id, center_label, target_label, target_idx in pair_specs:
            data = _rdf_curve_from_frames(
                frames=list(win.get("frames") or []),
                center_indices=centers,
                target_indices=target_idx,
                r_max_nm=float(r_max_nm),
                bin_nm=float(bin_nm),
            )
            win_curves[pair_id] = data
            r = np.asarray(data.get("r_nm"), dtype=float)
            g = np.asarray(data.get("g_r"), dtype=float)
            cn = np.asarray(data.get("cn_curve"), dtype=float)
            for i in range(int(r.size)):
                curve_rows.append(
                    {
                        "window_index": int(win["window_index"]),
                        "time_start_ps": float(win["time_start_ps"]),
                        "time_end_ps": float(win["time_end_ps"]),
                        "pair": pair_id,
                        "r_nm": float(r[i]),
                        "g_r": float(g[i]),
                        "coordination_number": float(cn[i]),
                    }
                )
            shell = dict(data.get("shell") or {})
            shell_rows.append(
                {
                    "window_index": int(win["window_index"]),
                    "time_start_ps": float(win["time_start_ps"]),
                    "time_end_ps": float(win["time_end_ps"]),
                    "pair": pair_id,
                    "center": center_label,
                    "target": target_label,
                    "r_peak_nm": shell.get("r_peak_nm"),
                    "r_shell_nm": shell.get("r_shell_nm"),
                    "cn_shell": shell.get("cn_shell"),
                    "confidence": shell.get("confidence"),
                    "status": shell.get("status"),
                }
            )
        curves.append({"window": dict(win), "curves": win_curves})
    curve_csv = out_dir / "time_series" / "rdf_cn_curves_timeseries.csv"
    shell_csv = out_dir / "time_series" / "rdf_cn_shell_timeseries.csv"
    _write_rows_csv(curve_csv, curve_rows)
    _write_rows_csv(shell_csv, shell_rows)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _animation_result(
            path=None,
            reason="matplotlib_unavailable",
            extra={"curves_csv": str(curve_csv), "shell_csv": str(shell_csv)},
        )
    mp4_path = out_dir / "time_series" / "rdf_cn_timeseries.mp4"
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    ymax_g = max([float(np.max(payload["curves"][pid]["g_r"])) for payload in curves for pid, *_rest in pair_specs] or [1.0])
    ymax_cn = max(
        [float(np.max(payload["curves"][pid]["cn_curve"])) for payload in curves for pid, *_rest in pair_specs]
        or [1.0]
    )
    fig, (ax_g, ax_cn) = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True)

    def _update(frame_idx: int):
        ax_g.clear()
        ax_cn.clear()
        payload = curves[int(frame_idx)]
        win = payload["window"]
        for pair_id, _center_label, target_label, _target_idx in pair_specs:
            data = payload["curves"][pair_id]
            r = np.asarray(data["r_nm"], dtype=float)
            ax_g.plot(r, np.asarray(data["g_r"], dtype=float), lw=1.8, label=target_label)
            ax_cn.plot(r, np.asarray(data["cn_curve"], dtype=float), lw=1.8, label=target_label)
        ax_g.set_ylim(0.0, max(1.0, ymax_g * 1.15))
        ax_cn.set_ylim(0.0, max(1.0, ymax_cn * 1.15))
        ax_cn.set_xlim(0.0, float(r_max_nm))
        ax_g.set_ylabel("g(r)")
        ax_cn.set_ylabel("CN(r)")
        ax_cn.set_xlabel("r / nm")
        ax_g.set_title(f"RDF/CN, {float(win['time_start_ps']):.1f}-{float(win['time_end_ps']):.1f} ps")
        ax_g.legend(loc="best", fontsize="small", ncols=3)
        fig.tight_layout()

    frame_meta = _save_animation_png_frames(
        fig=fig,
        update=_update,
        frame_count=len(curves),
        frame_dir=out_dir / "time_series" / "frames" / "rdf_cn",
    )
    extra = {
        "curves_csv": str(curve_csv),
        "shell_csv": str(shell_csv),
        "sample_windows": len(curves),
        "pairs": [p[0] for p in pair_specs],
        **frame_meta,
    }
    writer = _mp4_writer(fps)
    if writer is None:
        plt.close(fig)
        return _animation_result(path=None, reason="ffmpeg_writer_unavailable", extra=extra)
    try:
        from matplotlib.animation import FuncAnimation
    except Exception:
        plt.close(fig)
        return _animation_result(path=None, reason="matplotlib_unavailable", extra=extra)
    ani = FuncAnimation(fig, _update, frames=len(curves), interval=1000)
    try:
        ani.save(mp4_path, writer=writer)
    except Exception as exc:
        plt.close(fig)
        return _animation_result(
            path=None,
            reason=f"mp4_write_failed:{exc.__class__.__name__}",
            extra=extra,
        )
    plt.close(fig)
    return _animation_result(path=mp4_path, extra=extra)


def _write_angle_timeseries(
    *,
    out_dir: Path,
    windows: Sequence[dict[str, Any]],
    adsorption_rows: Sequence[dict[str, Any]],
    fps: float,
    bin_deg: float = 10.0,
) -> dict[str, Any]:
    rows = [dict(row) for row in adsorption_rows if bool(row.get("orientation_available")) and bool(row.get("adsorbed"))]
    if not windows or not rows:
        return _animation_result(path=None, reason="no_adsorbed_orientation_samples")
    edges = np.arange(0.0, 180.0 + float(bin_deg), float(bin_deg))
    mids = 0.5 * (edges[:-1] + edges[1:])
    profiles: list[dict[str, Any]] = []
    out_rows: list[dict[str, Any]] = []
    for win in windows:
        lo = float(win["time_start_ps"])
        hi = float(win["time_end_ps"])
        subset = [row for row in rows if lo <= float(row.get("time_ps") or 0.0) <= hi]
        carbonyl = [float(row["carbonyl_angle_deg"]) for row in subset if row.get("carbonyl_angle_deg") is not None]
        dipole = [float(row["dipole_proxy_angle_deg"]) for row in subset if row.get("dipole_proxy_angle_deg") is not None]
        c_hist = np.histogram(np.asarray(carbonyl, dtype=float), bins=edges)[0].astype(float)
        d_hist = np.histogram(np.asarray(dipole, dtype=float), bins=edges)[0].astype(float)
        c_prob = c_hist / max(float(np.sum(c_hist)), 1.0)
        d_prob = d_hist / max(float(np.sum(d_hist)), 1.0)
        profiles.append(
            {
                "window": dict(win),
                "carbonyl": c_prob,
                "dipole": d_prob,
                "n_carbonyl": len(carbonyl),
                "n_dipole": len(dipole),
            }
        )
        for i in range(int(mids.size)):
            out_rows.append(
                {
                    "window_index": int(win["window_index"]),
                    "time_start_ps": lo,
                    "time_end_ps": hi,
                    "angle_mid_deg": float(mids[i]),
                    "carbonyl_probability": float(c_prob[i]),
                    "dipole_proxy_probability": float(d_prob[i]),
                    "carbonyl_samples": int(len(carbonyl)),
                    "dipole_proxy_samples": int(len(dipole)),
                }
            )
    csv_path = out_dir / "time_series" / "adsorbed_orientation_angle_timeseries.csv"
    _write_rows_csv(csv_path, out_rows)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _animation_result(path=None, reason="matplotlib_unavailable", extra={"csv": str(csv_path)})
    mp4_path = out_dir / "time_series" / "adsorbed_orientation_angle_timeseries.mp4"
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    ymax = max(
        [float(np.max(p["carbonyl"])) for p in profiles]
        + [float(np.max(p["dipole"])) for p in profiles]
        + [1.0e-12]
    ) * 1.15
    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    def _update(frame_idx: int):
        ax.clear()
        payload = profiles[int(frame_idx)]
        win = payload["window"]
        ax.plot(mids, payload["carbonyl"], lw=1.8, label=f"carbonyl ({payload['n_carbonyl']})")
        ax.plot(mids, payload["dipole"], lw=1.8, label=f"dipole proxy ({payload['n_dipole']})")
        ax.set_xlim(0.0, 180.0)
        ax.set_ylim(0.0, max(0.2, ymax))
        ax.set_xlabel("angle to nearest graphite surface normal / deg")
        ax.set_ylabel("probability")
        ax.set_title(f"Adsorbed orientation, {float(win['time_start_ps']):.1f}-{float(win['time_end_ps']):.1f} ps")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()

    frame_meta = _save_animation_png_frames(
        fig=fig,
        update=_update,
        frame_count=len(profiles),
        frame_dir=out_dir / "time_series" / "frames" / "adsorbed_orientation_angle",
    )
    extra = {"csv": str(csv_path), "sample_windows": len(profiles), **frame_meta}
    writer = _mp4_writer(fps)
    if writer is None:
        plt.close(fig)
        return _animation_result(path=None, reason="ffmpeg_writer_unavailable", extra=extra)
    try:
        from matplotlib.animation import FuncAnimation
    except Exception:
        plt.close(fig)
        return _animation_result(path=None, reason="matplotlib_unavailable", extra=extra)
    ani = FuncAnimation(fig, _update, frames=len(profiles), interval=1000)
    try:
        ani.save(mp4_path, writer=writer)
    except Exception as exc:
        plt.close(fig)
        return _animation_result(
            path=None,
            reason=f"mp4_write_failed:{exc.__class__.__name__}",
            extra=extra,
        )
    plt.close(fig)
    return _animation_result(path=mp4_path, extra=extra)


def _time_series_animations(
    *,
    out_dir: Path,
    frames: Sequence[tuple[float, np.ndarray, tuple[float, float, float]]],
    bins: np.ndarray,
    instances: Sequence[dict[str, Any]],
    categories: dict[str, np.ndarray],
    adsorption_rows: Sequence[dict[str, Any]],
    graphite_surfaces: Sequence[dict[str, Any]],
    surface_distance_nm: float,
    sample_count: int,
    fps: float,
    rdf_rmax_nm: float,
    rdf_bin_nm: float,
    enable_rdf: bool = True,
    enable_concentration: bool = True,
    enable_angles: bool = True,
) -> dict[str, Any]:
    windows = _time_windows(frames, sample_count=sample_count)
    ts_dir = Path(out_dir) / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "available": bool(windows),
        "sample_count_requested": int(max(1, sample_count)),
        "sample_windows": int(len(windows)),
        "sampling_note": (
            "The analyzed trajectory is divided into up to ten equal time windows by default; "
            "each animation frame is one window-average."
        ),
        "fps": float(fps),
    }
    if not windows:
        meta["reason"] = "too_few_time_windows"
        _write_json(ts_dir / "time_series_summary.json", meta)
        return meta
    outputs: dict[str, Any] = {}
    if enable_concentration:
        outputs["z_concentration"] = _write_concentration_timeseries(
            out_dir=out_dir,
            windows=windows,
            bins=bins,
            instances=instances,
            fps=float(fps),
        )
    if enable_rdf:
        outputs["rdf_cn"] = _write_rdf_cn_timeseries(
            out_dir=out_dir,
            windows=windows,
            categories=categories,
            fps=float(fps),
            r_max_nm=float(rdf_rmax_nm),
            bin_nm=float(rdf_bin_nm),
        )
        outputs["edl_rdf_cn"] = _write_edl_rdf_cn_timeseries(
            out_dir=out_dir,
            windows=windows,
            instances=instances,
            surfaces=graphite_surfaces,
            surface_distance_nm=float(surface_distance_nm),
            fps=float(fps),
            r_max_nm=float(rdf_rmax_nm),
            bin_nm=float(rdf_bin_nm),
        )
    if enable_angles:
        outputs["adsorbed_orientation_angles"] = _write_angle_timeseries(
            out_dir=out_dir,
            windows=windows,
            adsorption_rows=adsorption_rows,
            fps=float(fps),
        )
    meta["outputs"] = outputs
    _write_json(ts_dir / "time_series_summary.json", meta)
    return meta


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


def _membrane_permeation_analysis(
    *,
    frames: Sequence[tuple[float, np.ndarray, tuple[float, float, float]]],
    instances: Sequence[dict[str, Any]],
    regions: dict[str, dict[str, float]],
    box_nm: tuple[float, float, float],
    species: Sequence[str] | None = None,
    penetration_threshold_nm: float = 0.20,
) -> dict[str, Any]:
    membrane_regions = [
        name
        for name in regions
        if any(token in name.lower() for token in ("polymer", "cmc", "membrane", "separator", "mixed"))
    ]
    if not frames:
        return {"available": False, "reason": "no_frames", "species": None if species is None else [str(x) for x in species]}
    if not membrane_regions:
        return {
            "available": False,
            "reason": "no_polymer_or_membrane_regions",
            "species": None if species is None else [str(x) for x in species],
        }
    membrane_lo = min(float(regions[name]["z_lo_nm"]) for name in membrane_regions)
    membrane_hi = max(float(regions[name]["z_hi_nm"]) for name in membrane_regions)
    membrane_thickness = max(0.0, membrane_hi - membrane_lo)
    box_x, box_y, box_z = (float(box_nm[0]), float(box_nm[1]), float(box_nm[2]))
    area_nm2 = max(box_x * box_y, 1.0e-12)
    membrane_volume_nm3 = max(area_nm2 * membrane_thickness, 1.0e-12)
    threshold = max(0.0, float(penetration_threshold_nm))
    selected: list[tuple[int, dict[str, Any]]] = []
    for inst_id, inst in enumerate(instances):
        moltype = str(inst.get("moltype") or "")
        kind = str(inst.get("kind") or "").lower()
        label = moltype.lower()
        if not moltype:
            continue
        if "graph" in label or "substrate" in kind:
            continue
        if "polymer" in kind or "poly" in label or "cmc" in label:
            continue
        if not _species_matches(moltype, species):
            continue
        selected.append((int(inst_id), inst))
    if not selected:
        return {
            "available": False,
            "reason": "no_matching_mobile_permeant_instances",
            "species": None if species is None else [str(x) for x in species],
            "membrane_regions": membrane_regions,
        }

    first_coords = frames[0][1]
    first_side_counts = {"below": 0, "above": 0}
    for _inst_id, inst in selected:
        z0 = _instance_com_z(first_coords, inst)
        if z0 < membrane_lo:
            first_side_counts["below"] += 1
        elif z0 > membrane_hi:
            first_side_counts["above"] += 1
    if first_side_counts["below"] > first_side_counts["above"]:
        feed_side = "below"
    elif first_side_counts["above"] > first_side_counts["below"]:
        feed_side = "above"
    else:
        electrolyte_regions = [name for name in regions if "electrolyte" in name.lower()]
        electrolyte_mid = [
            0.5 * (float(regions[name]["z_lo_nm"]) + float(regions[name]["z_hi_nm"]))
            for name in electrolyte_regions
        ]
        below_score = sum(1 for z in electrolyte_mid if z < membrane_lo)
        above_score = sum(1 for z in electrolyte_mid if z > membrane_hi)
        feed_side = "below" if below_score > above_score else "above"
    permeate_side = "above" if feed_side == "below" else "below"

    def classify_z(z_nm: float) -> tuple[str, float | None]:
        z = float(z_nm)
        if membrane_lo <= z <= membrane_hi:
            depth = max(0.0, min(z - membrane_lo, membrane_hi - z))
            if depth >= threshold:
                return "membrane", float(depth)
            return "membrane_interface", float(depth)
        side = "below" if z < membrane_lo else "above"
        if side == feed_side:
            return "feed", None
        if side == permeate_side:
            return "permeate", None
        return "outside", None

    times = [float(item[0]) for item in frames]
    duration_ps = max(0.0, float(times[-1] - times[0])) if len(times) >= 2 else 0.0
    duration_ns = duration_ps / 1000.0
    dt_ps = _frame_interval_ps(frames)
    species_records: dict[str, dict[str, Any]] = {}
    event_rows: list[dict[str, Any]] = []
    per_instance: list[dict[str, Any]] = []
    for inst_id, inst in selected:
        moltype = str(inst.get("moltype") or "")
        states: list[str] = []
        depths: list[float | None] = []
        z_values: list[float] = []
        for _time_ps, coords, _box in frames:
            z = _instance_com_z(coords, inst)
            state, depth = classify_z(z)
            states.append(state)
            depths.append(depth)
            z_values.append(float(z))
        feed_seen = False
        membrane_seen_after_feed = False
        first_entry_time = None
        first_arrival_time = None
        entry_count = 0
        for idx, state in enumerate(states):
            prev = states[idx - 1] if idx > 0 else None
            if state == "feed":
                feed_seen = True
            if state == "membrane" and prev != "membrane" and (feed_seen or prev in {"feed", "membrane_interface"}):
                entry_count += 1
                if first_entry_time is None:
                    first_entry_time = float(times[idx])
            if state == "membrane" and feed_seen:
                membrane_seen_after_feed = True
            if state == "permeate" and feed_seen and membrane_seen_after_feed and first_arrival_time is None:
                first_arrival_time = float(times[idx])
        membrane_flags = [state == "membrane" for state in states]
        feed_flags = [state == "feed" for state in states]
        permeate_flags = [state == "permeate" for state in states]
        valid_depths = [float(depth) for depth in depths if depth is not None and float(depth) >= threshold]
        residence_ps = None if dt_ps is None else float(sum(1 for flag in membrane_flags if flag)) * float(dt_ps)
        event = {
            "moltype": moltype,
            "instance_index": int(inst_id),
            "initial_state": states[0] if states else None,
            "final_state": states[-1] if states else None,
            "entry_event_count": int(entry_count),
            "translocation_event_count": int(1 if first_arrival_time is not None else 0),
            "first_entry_time_ps": first_entry_time,
            "first_permeate_arrival_time_ps": first_arrival_time,
            "membrane_frame_count": int(sum(1 for flag in membrane_flags if flag)),
            "feed_frame_count": int(sum(1 for flag in feed_flags if flag)),
            "permeate_frame_count": int(sum(1 for flag in permeate_flags if flag)),
            "membrane_residence_time_ps": residence_ps,
            "max_membrane_depth_nm": None if not valid_depths else float(max(valid_depths)),
            "mean_membrane_depth_nm": None if not valid_depths else float(sum(valid_depths) / len(valid_depths)),
            "min_z_nm": float(min(z_values)) if z_values else None,
            "max_z_nm": float(max(z_values)) if z_values else None,
        }
        event_rows.append(event)
        per_instance.append({**event, "states": states, "depths": depths})
        rec = species_records.setdefault(
            moltype,
            {
                "molecule_count": 0,
                "sample_frame_count": 0,
                "membrane_frame_count": 0,
                "feed_frame_count": 0,
                "permeate_frame_count": 0,
                "entry_event_count": 0,
                "translocation_event_count": 0,
                "membrane_residence_time_ps_total": 0.0,
                "membrane_residence_time_ps_counted": 0,
                "first_entry_times_ps": [],
                "first_permeate_arrival_times_ps": [],
                "max_membrane_depth_nm": None,
            },
        )
        rec["molecule_count"] += 1
        rec["sample_frame_count"] += len(states)
        rec["membrane_frame_count"] += int(event["membrane_frame_count"])
        rec["feed_frame_count"] += int(event["feed_frame_count"])
        rec["permeate_frame_count"] += int(event["permeate_frame_count"])
        rec["entry_event_count"] += int(event["entry_event_count"])
        rec["translocation_event_count"] += int(event["translocation_event_count"])
        if residence_ps is not None:
            rec["membrane_residence_time_ps_total"] += float(residence_ps)
            rec["membrane_residence_time_ps_counted"] += 1
        if first_entry_time is not None:
            rec["first_entry_times_ps"].append(float(first_entry_time))
        if first_arrival_time is not None:
            rec["first_permeate_arrival_times_ps"].append(float(first_arrival_time))
        if event["max_membrane_depth_nm"] is not None:
            old = rec.get("max_membrane_depth_nm")
            rec["max_membrane_depth_nm"] = float(event["max_membrane_depth_nm"]) if old is None else max(float(old), float(event["max_membrane_depth_nm"]))

    timeseries_rows: list[dict[str, Any]] = []
    species_names = sorted(species_records)
    for frame_idx, time_ps in enumerate(times):
        for moltype in species_names:
            relevant = [item for item in per_instance if str(item.get("moltype")) == moltype]
            feed_count = sum(1 for item in relevant if item["states"][frame_idx] == "feed")
            membrane_count = sum(1 for item in relevant if item["states"][frame_idx] == "membrane")
            permeate_count = sum(1 for item in relevant if item["states"][frame_idx] == "permeate")
            depths_at_t = [
                float(item["depths"][frame_idx])
                for item in relevant
                if item["depths"][frame_idx] is not None and item["states"][frame_idx] == "membrane"
            ]
            cumulative_entries = sum(
                int((item.get("first_entry_time_ps") is not None) and float(item["first_entry_time_ps"]) <= float(time_ps))
                for item in relevant
            )
            cumulative_translocations = sum(
                int(
                    (item.get("first_permeate_arrival_time_ps") is not None)
                    and float(item["first_permeate_arrival_time_ps"]) <= float(time_ps)
                )
                for item in relevant
            )
            timeseries_rows.append(
                {
                    "time_ps": float(time_ps),
                    "moltype": moltype,
                    "feed_count": int(feed_count),
                    "membrane_count": int(membrane_count),
                    "permeate_count": int(permeate_count),
                    "cumulative_entry_events": int(cumulative_entries),
                    "cumulative_translocation_events": int(cumulative_translocations),
                    "mean_membrane_depth_nm": None if not depths_at_t else float(sum(depths_at_t) / len(depths_at_t)),
                }
            )

    feed_thickness_nm = membrane_lo if feed_side == "below" else max(0.0, box_z - membrane_hi)
    feed_volume_nm3 = max(area_nm2 * feed_thickness_nm, 1.0e-12)
    summary_by_species: dict[str, dict[str, Any]] = {}
    for moltype, rec in species_records.items():
        sample_frames = max(1, int(rec.get("sample_frame_count") or 0))
        mean_membrane_count = float(rec.get("membrane_frame_count") or 0) / float(max(1, len(frames)))
        initial_feed_count = sum(
            1
            for item in per_instance
            if str(item.get("moltype")) == moltype and item["states"] and item["states"][0] == "feed"
        )
        feed_density_nm3 = float(initial_feed_count) / feed_volume_nm3 if feed_volume_nm3 > 0.0 else None
        entry_flux = (
            None
            if duration_ns <= 0.0
            else float(rec.get("entry_event_count") or 0) / area_nm2 / float(duration_ns)
        )
        translocation_flux = (
            None
            if duration_ns <= 0.0
            else float(rec.get("translocation_event_count") or 0) / area_nm2 / float(duration_ns)
        )
        permeability = (
            None
            if translocation_flux is None or feed_density_nm3 is None or feed_density_nm3 <= 0.0
            else float(translocation_flux) / float(feed_density_nm3)
        )
        entry_permeability = (
            None
            if entry_flux is None or feed_density_nm3 is None or feed_density_nm3 <= 0.0
            else float(entry_flux) / float(feed_density_nm3)
        )
        first_entry_times = [float(x) for x in rec.get("first_entry_times_ps", [])]
        first_arrival_times = [float(x) for x in rec.get("first_permeate_arrival_times_ps", [])]
        residence_counted = int(rec.get("membrane_residence_time_ps_counted") or 0)
        total_residence = float(rec.get("membrane_residence_time_ps_total") or 0.0)
        loading_nm3 = mean_membrane_count / membrane_volume_nm3
        initial_feed_denom = max(1, int(initial_feed_count))
        summary_by_species[moltype] = {
            "molecule_count": int(rec.get("molecule_count") or 0),
            "initial_feed_count": int(initial_feed_count),
            "feed_frame_fraction": float(rec.get("feed_frame_count") or 0) / float(sample_frames),
            "membrane_frame_fraction": float(rec.get("membrane_frame_count") or 0) / float(sample_frames),
            "permeate_frame_fraction": float(rec.get("permeate_frame_count") or 0) / float(sample_frames),
            "entry_event_count": int(rec.get("entry_event_count") or 0),
            "translocation_event_count": int(rec.get("translocation_event_count") or 0),
            "entry_event_fraction_per_initial_feed": float(rec.get("entry_event_count") or 0) / float(initial_feed_denom),
            "translocation_fraction_per_initial_feed": float(rec.get("translocation_event_count") or 0) / float(initial_feed_denom),
            "mean_membrane_count": float(mean_membrane_count),
            "mean_membrane_loading_molecules_nm3": float(loading_nm3),
            "mean_membrane_loading_mol_L": float(loading_nm3 * 1.0e24 / _AVOGADRO),
            "feed_number_density_nm3": feed_density_nm3,
            "apparent_entry_flux_events_nm2_ns": entry_flux,
            "apparent_translocation_flux_events_nm2_ns": translocation_flux,
            "apparent_entry_permeability_nm_ns": entry_permeability,
            "apparent_translocation_permeability_nm_ns": permeability,
            "apparent_entry_permeability_m_s": entry_permeability,
            "apparent_translocation_permeability_m_s": permeability,
            "first_entry_time_ps_min": None if not first_entry_times else float(min(first_entry_times)),
            "first_entry_time_ps_mean": None if not first_entry_times else float(sum(first_entry_times) / len(first_entry_times)),
            "first_permeate_arrival_time_ps_min": None if not first_arrival_times else float(min(first_arrival_times)),
            "first_permeate_arrival_time_ps_mean": None if not first_arrival_times else float(sum(first_arrival_times) / len(first_arrival_times)),
            "membrane_residence_time_ps_total": None if residence_counted <= 0 else float(total_residence),
            "membrane_residence_time_ps_mean_per_molecule": None if residence_counted <= 0 else float(total_residence / residence_counted),
            "max_membrane_depth_nm": rec.get("max_membrane_depth_nm"),
        }
    reference_species = next(
        (
            name
            for name, rec in summary_by_species.items()
            if rec.get("apparent_entry_flux_events_nm2_ns") is not None
            and float(rec.get("apparent_entry_flux_events_nm2_ns") or 0.0) > 0.0
        ),
        next(iter(summary_by_species), None),
    )
    if reference_species is not None:
        ref_entry = summary_by_species[reference_species].get("apparent_entry_flux_events_nm2_ns")
        ref_perm = summary_by_species[reference_species].get("apparent_translocation_permeability_nm_ns")
        for rec in summary_by_species.values():
            entry = rec.get("apparent_entry_flux_events_nm2_ns")
            perm = rec.get("apparent_translocation_permeability_nm_ns")
            rec["selectivity_reference_species"] = reference_species
            rec["entry_flux_selectivity_vs_reference"] = (
                None if ref_entry is None or float(ref_entry) <= 0.0 or entry is None else float(entry) / float(ref_entry)
            )
            rec["translocation_permeability_selectivity_vs_reference"] = (
                None if ref_perm is None or float(ref_perm) <= 0.0 or perm is None else float(perm) / float(ref_perm)
            )

    return {
        "available": True,
        "species": None if species is None else [str(x) for x in species],
        "membrane_regions": membrane_regions,
        "membrane_z_lo_nm": float(membrane_lo),
        "membrane_z_hi_nm": float(membrane_hi),
        "membrane_thickness_nm": float(membrane_thickness),
        "membrane_area_nm2": float(area_nm2),
        "membrane_volume_nm3": float(membrane_volume_nm3),
        "feed_side": feed_side,
        "permeate_side": permeate_side,
        "feed_volume_nm3": float(feed_volume_nm3),
        "duration_ps": float(duration_ps),
        "penetration_threshold_nm": float(threshold),
        "summary_by_species": summary_by_species,
        "events": event_rows,
        "time_series_rows": timeseries_rows,
        "note": (
            "Finite-slab membrane diagnostic from molecule COM trajectories. "
            "Flux and permeability are apparent reservoir estimates, not pressure-gradient macroscopic permeabilities."
        ),
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
    cation: list[int] = []
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
        if idx.size == 1 and (formal > 0.0 or "li" in label or "na" in label):
            cation.extend(idx.tolist())
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
        "li": np.asarray(sorted(set(cation)), dtype=int),
        "cation": np.asarray(sorted(set(cation)), dtype=int),
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
    time_series_analysis: bool = False,
    time_series_sample_count: int = 10,
    time_series_fps: float = 1.0,
    time_series_rdf: bool = True,
    time_series_concentration: bool = True,
    time_series_angles: bool = True,
    time_series_rdf_rmax_nm: float = 1.2,
    time_series_rdf_bin_nm: float = 0.02,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Compute cheap interface statistics and write JSON/CSV artifacts."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    time_series_options = InterfaceTimeSeriesOptions.from_parameters(
        time_series_analysis=bool(time_series_analysis),
        time_series_sample_count=int(time_series_sample_count),
        time_series_fps=float(time_series_fps),
        time_series_rdf=bool(time_series_rdf),
        time_series_concentration=bool(time_series_concentration),
        time_series_angles=bool(time_series_angles),
        time_series_rdf_rmax_nm=float(time_series_rdf_rmax_nm),
        time_series_rdf_bin_nm=float(time_series_rdf_bin_nm),
    )
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
    manifest_payload: dict[str, Any] = {}
    manifest_order: list[str] | None = None
    if manifest_path is not None and Path(manifest_path).is_file():
        try:
            manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            intervals = manifest_payload.get("layer_intervals_nm", []) if isinstance(manifest_payload, dict) else []
            if isinstance(intervals, list):
                ordered_names = [str(v.get("name")) for v in intervals if isinstance(v, dict) and v.get("name") is not None]
                manifest_order = [name for name in ordered_names if name in phase_groups_norm]
        except Exception:
            manifest_payload = {}
            manifest_order = None
    manifest_summary = {
        "available": bool(manifest_payload),
        "path": None if manifest_path is None else str(manifest_path),
        "name": manifest_payload.get("name") if manifest_payload else None,
        "pbc_mode": manifest_payload.get("pbc_mode") if manifest_payload else None,
        "box_nm": manifest_payload.get("box_nm") if manifest_payload else None,
        "layers": manifest_payload.get("layers", []) if manifest_payload else [],
        "layer_intervals_nm": manifest_payload.get("layer_intervals_nm", []) if manifest_payload else [],
        "fixed_charge_regions": manifest_payload.get("fixed_charge_regions", []) if manifest_payload else [],
        "acceptance": manifest_payload.get("acceptance", {}) if manifest_payload else {},
        "z_compaction": manifest_payload.get("z_compaction", {}) if manifest_payload else {},
    }

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
    atom_categories = _atom_indices_by_category(top, atom_payload["instances"])
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
    membrane_permeation = _membrane_permeation_analysis(
        frames=frames,
        instances=atom_payload["instances"],
        regions=regions,
        box_nm=final_box,
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
    time_series = (
        _time_series_animations(
            out_dir=out_dir,
            frames=frames,
            bins=bins,
            instances=atom_payload["instances"],
            categories=atom_categories,
            adsorption_rows=adsorption.get("rows") or [],
            graphite_surfaces=adsorption.get("surfaces") or [],
            surface_distance_nm=float(surface_distance_nm),
            sample_count=int(time_series_options.sample_count),
            fps=float(time_series_options.fps),
            rdf_rmax_nm=float(time_series_options.rdf_rmax_nm),
            rdf_bin_nm=float(time_series_options.rdf_bin_nm),
            enable_rdf=bool(time_series_options.rdf),
            enable_concentration=bool(time_series_options.concentration),
            enable_angles=bool(time_series_options.angles),
        )
        if bool(time_series_options.enabled)
        else {"available": False, "reason": "disabled"}
    )
    if not bool(time_series_options.enabled):
        _write_json(out_dir / "time_series" / "time_series_summary.json", time_series)
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
    penetration_depth_csv, penetration_depth_svg = _write_penetration_depth_distribution(
        out_csv=out_dir / "penetration_depth_distribution.csv",
        out_svg=out_dir / "penetration_depth.svg",
        rows=penetration.get("rows") or [],
        bin_nm=max(0.05, float(bin_nm)),
    )
    _write_rows_csv(out_dir / "membrane_permeation_events.csv", membrane_permeation.get("events") or [])
    _write_rows_csv(out_dir / "membrane_permeation_timeseries.csv", membrane_permeation.get("time_series_rows") or [])
    _write_json(
        out_dir / "membrane_permeation_summary.json",
        {key: value for key, value in membrane_permeation.items() if key not in {"events", "time_series_rows"}},
    )
    membrane_permeation_svg = _write_membrane_permeation_svg(
        out_dir / "membrane_permeation_summary.svg",
        membrane_permeation.get("summary_by_species") or {},
    )
    membrane_permeation_timeseries_svg = _write_membrane_timeseries_svg(
        out_dir / "membrane_permeation_timeseries.svg",
        membrane_permeation.get("time_series_rows") or [],
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
    adsorbed_orientation_distribution_csv, adsorbed_orientation_svg = _write_adsorbed_orientation_distribution(
        out_csv=out_dir / "adsorbed_orientation_distribution.csv",
        out_svg=out_dir / "adsorbed_orientation.svg",
        rows=adsorption.get("rows") or [],
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
        "membrane_permeation": {
            key: value for key, value in membrane_permeation.items() if key not in {"events", "time_series_rows"}
        },
        "adsorption": {key: value for key, value in adsorption.items() if key not in {"rows", "surface_map_rows"}},
        "time_series": time_series,
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
            "membrane_permeation_analysis": True,
            "phase_groups": [str(x) for x in phase_groups],
            "time_series_analysis": bool(time_series_options.enabled),
            "time_series_sample_count": int(time_series_options.sample_count),
            "time_series_fps": float(time_series_options.fps),
            "time_series_rdf": bool(time_series_options.rdf),
            "time_series_concentration": bool(time_series_options.concentration),
            "time_series_angles": bool(time_series_options.angles),
            "time_series_rdf_rmax_nm": float(time_series_options.rdf_rmax_nm),
            "time_series_rdf_bin_nm": float(time_series_options.rdf_bin_nm),
            "edl_rdf_cn_time_series": True,
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
        "manifest": manifest_summary,
        "region_summary": region_summary,
        "coordination_by_region": coordination,
        "anisotropic_msd_summary": transport,
        "edl_profiles": edl,
        "penetration": {key: value for key, value in penetration.items() if key != "rows"},
        "membrane_permeation": {
            key: value for key, value in membrane_permeation.items() if key not in {"events", "time_series_rows"}
        },
        "graphite_adsorption": {key: value for key, value in adsorption.items() if key not in {"rows", "surface_map_rows"}},
        "region_transport_summary": transport,
        "time_series": time_series,
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
            "penetration_depth_distribution_csv": str(penetration_depth_csv),
            "penetration_depth_svg": None if penetration_depth_svg is None else str(penetration_depth_svg),
            "membrane_permeation_events_csv": str(out_dir / "membrane_permeation_events.csv"),
            "membrane_permeation_timeseries_csv": str(out_dir / "membrane_permeation_timeseries.csv"),
            "membrane_permeation_summary_json": str(out_dir / "membrane_permeation_summary.json"),
            "membrane_permeation_summary_svg": None if membrane_permeation_svg is None else str(membrane_permeation_svg),
            "membrane_permeation_timeseries_svg": None if membrane_permeation_timeseries_svg is None else str(membrane_permeation_timeseries_svg),
            "adsorption_summary_json": str(out_dir / "adsorption_summary.json"),
            "adsorption_events_csv": str(out_dir / "adsorption_events.csv"),
            "adsorbed_orientation_csv": str(out_dir / "adsorbed_orientation.csv"),
            "adsorbed_orientation_distribution_csv": str(adsorbed_orientation_distribution_csv),
            "adsorbed_orientation_svg": None if adsorbed_orientation_svg is None else str(adsorbed_orientation_svg),
            "region_transport_summary_json": str(out_dir / "region_transport_summary.json"),
            "region_summary_json": str(out_dir / "region_summary.json"),
            "coordination_by_region_json": str(out_dir / "coordination_by_region.json"),
            "coordination_z_profile_csv": str(out_dir / "coordination_z_profile.csv"),
            "anisotropic_msd_summary_json": str(out_dir / "anisotropic_msd_summary.json"),
            "time_series_summary_json": str(out_dir / "time_series" / "time_series_summary.json"),
            "interface_profile_summary_json": str(out_dir / "interface_profile_summary.json"),
        },
    }
    _write_json(out_dir / "interface_profile_summary.json", summary)
    return _jsonify(summary)


__all__ = ["InterfaceTimeSeriesOptions", "compute_interface_profile"]
