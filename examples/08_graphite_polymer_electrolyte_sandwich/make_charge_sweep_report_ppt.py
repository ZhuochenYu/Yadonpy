"""Build a strict Eg08.07 charge-sweep PowerPoint report.

The report intentionally does more than paste figures into slides.  It checks
whether all charge states really start from the same t=0 coordinates, rewrites
z-axis data onto a single CMC-facing graphite reference, redraws comparison
figures with consistent axes and colors, and writes a visual audit so blank
data slides are caught automatically.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_CASES = (
    ("-18 uC/cm2", "cmcface_m18p0_uC_cm2", -18.0),
    ("-9 uC/cm2", "cmcface_m9p0_uC_cm2", -9.0),
    ("-3 uC/cm2", "cmcface_m3p0_uC_cm2", -3.0),
    ("0 uC/cm2", "cmcface_00_uC_cm2", 0.0),
)

SPECIES_ORDER = ("EC", "EMC", "DEC", "LI", "PF6", "NA", "CMCNA")
PERMEANT_SPECIES = ("EC", "EMC", "DEC", "LI", "PF6")
CHARGE_PALETTE = {
    -18.0: "#2B5C8A",
    -9.0: "#2A9D8F",
    -3.0: "#E76F51",
    0.0: "#6D597A",
}
SPECIES_PALETTE = {
    "EC": "#4C78A8",
    "EMC": "#72B7B2",
    "DEC": "#F58518",
    "LI": "#D62728",
    "PF6": "#7F3C8D",
    "NA": "#54A24B",
    "CMCNA": "#8C6D31",
}
LI_SOLVATION_TARGETS = ("solvent_o", "cmc_o", "pf6_f")
LI_SOLVATION_PALETTE = {
    "solvent_o": "#3B82B8",
    "cmc_o": "#9A7B39",
    "pf6_f": "#7F3C8D",
}


@dataclass(frozen=True)
class CasePayload:
    label: str
    dirname: str
    charge: float
    analysis_dir: Path
    relax_dir: Path
    summary: dict[str, Any]
    z0_nm: float


def _final_nvt_dir(item: CasePayload) -> Path:
    workflow = item.relax_dir / "05_relaxation_workflow"
    candidates = [
        workflow / "03_final_nvt_release",
        workflow / "04_final_nvt_release",
        workflow / "01_final_nvt",
        workflow / "final_nvt",
    ]
    for path in candidates:
        if (path / "md.xtc").is_file() and (path / "md.gro").is_file():
            return path
    if workflow.is_dir():
        for path in sorted(workflow.iterdir()):
            lname = path.name.lower()
            if path.is_dir() and "final" in lname and "nvt" in lname and (path / "md.xtc").is_file() and (path / "md.gro").is_file():
                return path
    return candidates[0]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    if y.size < 2 or x.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def _canonical_species(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("LI"):
        return "LI"
    if text.startswith("NA"):
        return "NA"
    if text.startswith("PF6"):
        return "PF6"
    if "CMC" in text or "POLY" in text:
        return "CMCNA"
    for key in ("EMC", "DEC", "EC"):
        if text == key or text.startswith(f"{key}_"):
            return key
    return text


def _species_matches(value: Any, target: str) -> bool:
    return _canonical_species(value) == _canonical_species(target)


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_fig(fig, base_path: Path, *, dpi: int = 220) -> dict[str, Path]:
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    return {"png": png, "svg": svg}


def _charge_color(charge: float) -> str:
    return CHARGE_PALETTE.get(float(charge), "#4D4D4D")


def _species_color(species: str) -> str:
    return SPECIES_PALETTE.get(_canonical_species(species), "#4D4D4D")


def _infer_z0_nm(summary: dict[str, Any]) -> float:
    phase_stats = dict(summary.get("phase_stats") or {})
    candidates = []
    for name, stats in phase_stats.items():
        lname = str(name).lower()
        if "graphite_bottom" in lname:
            z = stats.get("p95_z_nm") or stats.get("p100_z_nm") or stats.get("p99_z_nm")
            if z is not None:
                return float(z)
        if "graphite" in lname:
            z = stats.get("p95_z_nm") or stats.get("p100_z_nm") or stats.get("p99_z_nm")
            if z is not None:
                candidates.append(float(z))
    if candidates:
        return float(min(candidates))
    membrane = dict(summary.get("membrane_permeation") or {})
    if membrane.get("membrane_z_lo_nm") is not None:
        return float(membrane["membrane_z_lo_nm"])
    return 0.0


def _load_payloads(root: Path, cases: tuple[tuple[str, str, float], ...]) -> list[CasePayload]:
    payloads: list[CasePayload] = []
    for label, dirname, charge in cases:
        relax_dir = root / dirname / "03_relaxation_sampling"
        analysis_dir = relax_dir / "06_analysis" / "layer_stack_interface"
        summary = _read_json(analysis_dir / "interface_profile_summary.json")
        payloads.append(
            CasePayload(
                label=label,
                dirname=dirname,
                charge=float(charge),
                analysis_dir=analysis_dir,
                relax_dir=relax_dir,
                summary=summary,
                z0_nm=_infer_z0_nm(summary),
            )
        )
    return payloads


def _validate_t0(payloads: list[CasePayload], out_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "ok": False,
        "tolerance_rmsd_nm": 1.0e-6,
        "cases": [],
        "system_gro_md5": [],
        "trajectory_first_frame": {},
        "reason": None,
    }
    gro_hashes: list[tuple[CasePayload, str, Path]] = []
    for item in payloads:
        system_gro = item.relax_dir.parent / "02_system" / "system.gro"
        if not system_gro.is_file():
            result["cases"].append({"case": item.label, "available": False, "reason": "missing_02_system_system_gro", "path": str(system_gro)})
            continue
        digest = hashlib.md5(system_gro.read_bytes()).hexdigest()
        gro_hashes.append((item, digest, system_gro))
        result["system_gro_md5"].append({"case": item.label, "md5": digest, "path": str(system_gro)})
        result["cases"].append({"case": item.label, "available": True, "system_gro_md5": digest, "path": str(system_gro)})
    if len(gro_hashes) != len(payloads):
        result["reason"] = "missing_shared_t0_system_gro"
        _write_json(out_dir / "shared_t0_validation.json", result)
        return result
    ref_digest = gro_hashes[0][1]
    system_ok = all(digest == ref_digest for _item, digest, _path in gro_hashes)
    result["available"] = True
    result["ok"] = bool(system_ok)
    result["reason"] = None if system_ok else "02_system_system_gro_not_identical"

    # The exact shared-t0 contract is the case-level system.gro/topology.  The
    # first saved trajectory frame is only a diagnostic, because an xtc may start
    # after several MD steps depending on output cadence.
    try:
        import mdtraj as md
    except Exception as exc:
        result["trajectory_first_frame"] = {"available": False, "reason": f"mdtraj_unavailable: {exc}"}
        _write_json(out_dir / "shared_t0_validation.json", result)
        return result
    frames: list[tuple[CasePayload, np.ndarray, np.ndarray, float]] = []
    for item in payloads:
        base = _final_nvt_dir(item)
        xtc = base / "md.xtc"
        gro = base / "md.gro"
        if not xtc.is_file() or not gro.is_file():
            result.setdefault("trajectory_cases", []).append({"case": item.label, "available": False, "reason": "missing_final_nvt_xtc_or_gro"})
            continue
        try:
            frame = md.load_frame(str(xtc), 0, top=str(gro))
            frames.append((item, frame.xyz[0].copy(), frame.unitcell_lengths[0].copy(), float(frame.time[0])))
            result.setdefault("trajectory_cases", []).append(
                {
                    "case": item.label,
                    "available": True,
                    "natoms": int(frame.n_atoms),
                    "time_ps": float(frame.time[0]),
                    "box_nm": [float(x) for x in frame.unitcell_lengths[0][:3]],
                }
            )
        except Exception as exc:
            result.setdefault("trajectory_cases", []).append({"case": item.label, "available": False, "reason": str(exc)})
    if len(frames) != len(payloads):
        result["trajectory_first_frame"] = {"available": False, "reason": "missing_first_frames"}
        _write_json(out_dir / "shared_t0_validation.json", result)
        return result
    ref_item, ref_xyz, ref_box, _ref_time = frames[0]
    comparisons = []
    traj_ok = True
    for item, xyz, box, _time_ps in frames:
        if xyz.shape != ref_xyz.shape:
            rmsd = math.inf
            max_abs = math.inf
        else:
            delta = xyz - ref_xyz
            rmsd = float(np.sqrt(np.mean(delta * delta)))
            max_abs = float(np.max(np.abs(delta)))
        box_delta = [float(x) for x in (box - ref_box)]
        row = {
            "case": item.label,
            "reference_case": ref_item.label,
            "first_frame_rmsd_nm": rmsd,
            "max_abs_coord_delta_nm": max_abs,
            "box_delta_nm": box_delta,
            "ok": bool(rmsd <= result["tolerance_rmsd_nm"] and max(abs(x) for x in box_delta) <= 1.0e-6),
        }
        comparisons.append(row)
        traj_ok = traj_ok and bool(row["ok"])
    result["trajectory_first_frame"] = {
        "available": True,
        "ok": bool(traj_ok),
        "comparisons": comparisons,
        "note": "Diagnostic only; shared t=0 acceptance is based on identical 02_system/system.gro files.",
    }
    _write_json(out_dir / "shared_t0_validation.json", result)
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _aggregate_z_profiles(payloads: list[CasePayload], out_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    by_case: dict[str, dict[str, dict[float, float]]] = {}
    all_x: list[float] = []
    for item in payloads:
        rows = _read_rows(item.analysis_dir / "z_density_profiles.csv")
        grouped: dict[str, dict[float, float]] = {}
        out_rows = []
        for row in rows:
            z_abs = _safe_float(row.get("z_mid_nm"))
            if not np.isfinite(z_abs):
                continue
            z_rel = float(z_abs - item.z0_nm)
            species = _canonical_species(row.get("entity"))
            if str(row.get("entity_kind", "")).lower() == "phase":
                species = _canonical_species(row.get("entity"))
            if species not in SPECIES_ORDER:
                continue
            density = _safe_float(row.get("mass_density_g_cm3"), 0.0)
            grouped.setdefault(species, {})[round(z_rel, 6)] = grouped.setdefault(species, {}).get(round(z_rel, 6), 0.0) + float(density)
            out_rows.append({**row, "z_rel_nm": z_rel, "canonical_species": species})
            all_x.append(z_rel)
        _write_rows(out_dir / "zrel_csv" / f"{item.dirname}_z_density_profiles_zrel.csv", out_rows)
        by_case[item.label] = grouped
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for item in payloads:
        arrays[item.label] = {}
        for species in SPECIES_ORDER:
            points = by_case.get(item.label, {}).get(species, {})
            if not points:
                arrays[item.label][species] = np.empty((0, 2), dtype=float)
                continue
            x = np.asarray(sorted(points), dtype=float)
            y = np.asarray([points[float(round(v, 6))] for v in x], dtype=float)
            arrays[item.label][species] = np.column_stack([x, y])
    return arrays


def _common_limits(series: Iterable[np.ndarray], *, y_pad: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for arr in series:
        if arr.size == 0:
            continue
        xs.extend([float(np.nanmin(arr[:, 0])), float(np.nanmax(arr[:, 0]))])
        ys.extend([float(np.nanmin(arr[:, 1])), float(np.nanmax(arr[:, 1]))])
    if not xs:
        return (0.0, 1.0), (0.0, 1.0)
    xlim = (float(min(xs)), float(max(xs)))
    ymin, ymax = float(min(ys)), float(max(ys))
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = (ymax - ymin) * y_pad
    return xlim, (max(0.0, ymin - pad), ymax + pad)


def _plot_z_facets_by_charge(z_arrays: dict[str, dict[str, np.ndarray]], payloads: list[CasePayload], fig_dir: Path) -> dict[str, Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), sharex=True, sharey=True)
    all_series = [z_arrays[item.label][sp] for item in payloads for sp in SPECIES_ORDER if sp != "CMCNA"]
    xlim, ylim = _common_limits(all_series)
    for ax, item in zip(axes.flat, payloads):
        for sp in SPECIES_ORDER:
            arr = z_arrays[item.label].get(sp, np.empty((0, 2)))
            if arr.size == 0:
                continue
            ax.fill_between(arr[:, 0], arr[:, 1], step="mid", alpha=0.18, color=_species_color(sp))
            ax.step(arr[:, 0], arr[:, 1], where="mid", lw=1.2, color=_species_color(sp), label=sp)
        ax.set_title(item.label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.20)
    axes[0, 0].legend(ncol=4, frameon=False, fontsize=7)
    fig.supxlabel("z_rel / nm (0 = CMC-facing graphite surface)")
    fig.supylabel("mass density / g cm$^{-3}$")
    fig.suptitle("z distribution by charge: species compared within each charge state")
    fig.tight_layout()
    out = _save_fig(fig, fig_dir / "z_distribution_by_charge_facets")
    plt.close(fig)
    return out


def _plot_z_facets_by_species(z_arrays: dict[str, dict[str, np.ndarray]], payloads: list[CasePayload], fig_dir: Path) -> dict[str, Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    species = ("EC", "EMC", "DEC", "LI", "PF6", "NA")
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.0), sharex=True)
    all_series = [z_arrays[item.label][sp] for item in payloads for sp in species]
    xlim, _ylim = _common_limits(all_series)
    ymax_by_species = {}
    for sp in species:
        _x, y = _common_limits([z_arrays[item.label].get(sp, np.empty((0, 2))) for item in payloads])
        ymax_by_species[sp] = y
    for ax, sp in zip(axes.flat, species):
        for item in payloads:
            arr = z_arrays[item.label].get(sp, np.empty((0, 2)))
            if arr.size == 0:
                continue
            ax.step(arr[:, 0], arr[:, 1], where="mid", lw=1.5, color=_charge_color(item.charge), label=item.label)
        ax.set_title(sp)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ymax_by_species[sp])
        ax.grid(True, alpha=0.20)
    axes[0, 0].legend(ncol=2, frameon=False, fontsize=7)
    fig.supxlabel("z_rel / nm (0 = CMC-facing graphite surface)")
    fig.supylabel("mass density / g cm$^{-3}$")
    fig.suptitle("z distribution by species: charge sweep compared within each species")
    fig.tight_layout()
    out = _save_fig(fig, fig_dir / "z_distribution_by_species_facets")
    plt.close(fig)
    return out


def _membrane_timeseries(payloads: list[CasePayload], out_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    data: dict[str, dict[str, list[tuple[float, float, float, float, float]]]] = {}
    for item in payloads:
        rows = _read_rows(item.analysis_dir / "membrane_permeation_timeseries.csv")
        by_bin: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
        for row in rows:
            sp = _canonical_species(row.get("species") or row.get("moltype"))
            if sp not in PERMEANT_SPECIES and sp != "NA":
                continue
            t_ns = _safe_float(row.get("time_ps")) / 1000.0
            if not np.isfinite(t_ns):
                continue
            bin_idx = int(math.floor(t_ns / 1.0))
            feed = _safe_float(row.get("feed_count"), 0.0)
            mem = _safe_float(row.get("membrane_count"), 0.0)
            perm = _safe_float(row.get("permeate_count"), 0.0)
            entries = _safe_float(row.get("cumulative_entry_events"), 0.0)
            by_bin.setdefault((sp, bin_idx), []).append((feed, mem, perm, entries))
        for (sp, bin_idx), vals in by_bin.items():
            arr = np.asarray(vals, dtype=float)
            feed, mem, perm, entries = np.nanmean(arr, axis=0)
            total = feed + mem + perm
            frac = mem / total if total > 0 else np.nan
            data.setdefault(item.label, {}).setdefault(sp, []).append((bin_idx + 0.5, frac, entries, mem, total))
    final: dict[str, dict[str, np.ndarray]] = {}
    rows_out = []
    for label, by_sp in data.items():
        final[label] = {}
        for sp, vals in by_sp.items():
            vals = sorted(vals)
            arr = np.asarray(vals, dtype=float)
            if arr.shape[0] >= 3:
                smooth = np.convolve(arr[:, 1], np.ones(3) / 3.0, mode="same")
                arr[:, 1] = smooth
            final[label][sp] = arr
            for row in arr:
                rows_out.append(
                    {
                        "case": label,
                        "species": sp,
                        "time_ns": float(row[0]),
                        "membrane_fraction_smoothed": float(row[1]),
                        "cumulative_entry_events": float(row[2]),
                        "membrane_count": float(row[3]),
                        "total_count": float(row[4]),
                    }
                )
    _write_rows(out_dir / "membrane_fraction_1ns_3point_smooth.csv", rows_out)
    return final


def _plot_membrane_fraction(data: dict[str, dict[str, np.ndarray]], payloads: list[CasePayload], fig_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    by_species: dict[str, Path] = {}
    for sp in PERMEANT_SPECIES:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        for item in payloads:
            arr = data.get(item.label, {}).get(sp, np.empty((0, 5)))
            if arr.size:
                ax.plot(arr[:, 0], arr[:, 1], lw=1.8, color=_charge_color(item.charge), label=item.label)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("time / ns")
        ax.set_ylabel(f"f_mem({sp})")
        ax.set_title(f"{sp}: membrane fraction across charge states")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        out = _save_fig(fig, fig_dir / f"membrane_fraction_species_{sp}")
        by_species[sp] = out["png"]
        plt.close(fig)

    by_charge: dict[str, Path] = {}
    for item in payloads:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        for sp in PERMEANT_SPECIES:
            arr = data.get(item.label, {}).get(sp, np.empty((0, 5)))
            if arr.size:
                ax.plot(arr[:, 0], arr[:, 1], lw=1.8, color=_species_color(sp), label=sp)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("time / ns")
        ax.set_ylabel("f_mem")
        ax.set_title(f"{item.label}: membrane fraction by species")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=5)
        fig.tight_layout()
        out = _save_fig(fig, fig_dir / f"membrane_fraction_charge_{item.dirname}")
        by_charge[item.label] = out["png"]
        plt.close(fig)
    return by_species, by_charge


def _penetration_metrics(payloads: list[CasePayload], out_dir: Path) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for item in payloads:
        summary = _read_json(item.analysis_dir / "membrane_permeation_summary.json")
        by_species = dict(summary.get("summary_by_species") or {})
        hist_rows = _read_rows(item.analysis_dir / "penetration_depth_distribution.csv")
        hist_by_sp: dict[str, list[dict[str, str]]] = {}
        for row in hist_rows:
            sp = _canonical_species(row.get("species"))
            hist_by_sp.setdefault(sp, []).append(row)
        for sp in PERMEANT_SPECIES:
            rec = dict(by_species.get(sp) or {})
            rows = hist_by_sp.get(sp, [])
            depths = np.asarray([_safe_float(row.get("depth_mid_nm")) for row in rows], dtype=float)
            pct = np.asarray([_safe_float(row.get("percent_of_penetrated_frames")) for row in rows], dtype=float) / 100.0
            finite = np.isfinite(depths) & np.isfinite(pct)
            depths = depths[finite]
            pct = pct[finite]
            auc = _trapz(pct, depths) if depths.size else np.nan
            d95 = np.nan
            if depths.size and np.nansum(pct) > 0:
                order = np.argsort(depths)
                cum = np.cumsum(pct[order]) / np.nansum(pct)
                idx = min(int(np.searchsorted(cum, 0.95, side="left")), int(depths.size - 1))
                d95 = float(depths[order][idx])
            initial_feed = _safe_float(rec.get("initial_feed_count"), 0.0)
            entry_count = _safe_float(rec.get("entry_event_count"), 0.0)
            metrics.append(
                {
                    "case": item.label,
                    "charge_uC_cm2": item.charge,
                    "species": sp,
                    "P_entry": entry_count / initial_feed if initial_feed > 0 else np.nan,
                    "entry_event_count": entry_count,
                    "translocation_event_count": _safe_float(rec.get("translocation_event_count"), 0.0),
                    "D_mean_nm": _safe_float(rec.get("mean_membrane_depth_nm"), np.nan),
                    "D_max_nm": _safe_float(rec.get("max_membrane_depth_nm"), np.nan),
                    "D95_nm": d95,
                    "AUC_depth_nm": auc,
                    "loading_molecules_nm3": _safe_float(rec.get("mean_membrane_loading_molecules_nm3"), np.nan),
                    "apparent_entry_flux_events_nm2_ns": _safe_float(rec.get("apparent_entry_flux_events_nm2_ns"), np.nan),
                }
            )
    _write_rows(out_dir / "penetration_capability_metrics.csv", metrics)
    return metrics


def _plot_penetration_metrics(metrics: list[dict[str, Any]], payloads: list[CasePayload], fig_dir: Path) -> dict[str, Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    outputs: dict[str, Path] = {}
    for metric, ylabel in (
        ("P_entry", "entry events / initial feed"),
        ("D95_nm", "D95 penetration depth / nm"),
        ("AUC_depth_nm", "depth-distribution AUC / nm"),
        ("loading_molecules_nm3", "mean loading / molecules nm$^{-3}$"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), sharey=True)
        vals = [_safe_float(row.get(metric)) for row in metrics]
        finite = [v for v in vals if np.isfinite(v)]
        ymax = max(finite) * 1.18 if finite else 1.0
        for ax, item in zip(axes.flat, payloads):
            rows = [row for row in metrics if row["case"] == item.label]
            x = np.arange(len(PERMEANT_SPECIES), dtype=float)
            y = [_safe_float(next((row.get(metric) for row in rows if row["species"] == sp), np.nan), np.nan) for sp in PERMEANT_SPECIES]
            ax.bar(x, y, color=[_species_color(sp) for sp in PERMEANT_SPECIES], width=0.72)
            ax.set_xticks(x, PERMEANT_SPECIES, rotation=0)
            ax.set_ylim(0.0, ymax if ymax > 0 else 1.0)
            ax.set_title(item.label)
            ax.grid(True, axis="y", alpha=0.25)
        fig.supxlabel("species")
        fig.supylabel(ylabel)
        fig.suptitle(f"Penetration capability metric: {metric}")
        fig.tight_layout()
        out = _save_fig(fig, fig_dir / f"penetration_metric_{metric}")
        outputs[metric] = out["png"]
        plt.close(fig)
    return outputs


def _find_topology_file(item: CasePayload, final_dir: Path) -> Path | None:
    candidates = [
        final_dir / "system.top",
        final_dir / "topol.top",
        item.relax_dir / "system.top",
        item.relax_dir / "topol.top",
        item.relax_dir.parent / "system.top",
        item.relax_dir.parent / "topol.top",
    ]
    for path in candidates:
        if path.is_file():
            return path
    for root in (item.relax_dir, item.relax_dir.parent):
        if not root.is_dir():
            continue
        for name in ("system.top", "topol.top"):
            hits = sorted(root.rglob(name))
            if hits:
                return hits[0]
    return None


def _membrane_bounds(item: CasePayload) -> tuple[float, float, str] | None:
    summary = _read_json(item.analysis_dir / "membrane_permeation_summary.json")
    lo = summary.get("membrane_z_lo_nm")
    hi = summary.get("membrane_z_hi_nm")
    feed_side = str(summary.get("feed_side") or "").lower()
    if lo is None or hi is None:
        nested = item.summary.get("membrane_permeation") if isinstance(item.summary, dict) else {}
        if isinstance(nested, dict):
            lo = lo if lo is not None else nested.get("membrane_z_lo_nm")
            hi = hi if hi is not None else nested.get("membrane_z_hi_nm")
            feed_side = feed_side or str(nested.get("feed_side") or "").lower()
    try:
        lo_f = float(lo)
        hi_f = float(hi)
    except Exception:
        return None
    if not np.isfinite(lo_f) or not np.isfinite(hi_f) or hi_f <= lo_f:
        return None
    if feed_side not in {"below", "above"}:
        feed_side = "above"
    return lo_f, hi_f, feed_side


def _atom_is_oxygen_like(name: str, atype: str, charge: float) -> bool:
    lname = str(name or "").strip().lower()
    latype = str(atype or "").strip().lower()
    return lname.startswith("o") or latype.startswith("o") or float(charge) <= -0.20


def _atom_is_fluorine_like(name: str, atype: str, charge: float) -> bool:
    lname = str(name or "").strip().lower()
    latype = str(atype or "").strip().lower()
    return lname.startswith("f") or latype.startswith("f") or float(charge) < -0.10


def _li_solvation_indices(instances: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    li: list[int] = []
    solvent_o: list[int] = []
    cmc_o: list[int] = []
    pf6_f: list[int] = []
    for inst in instances:
        moltype = str(inst.get("moltype") or "")
        kind = str(inst.get("kind") or "").lower()
        species = _canonical_species(moltype)
        if "polymer" in kind or "cmc" in moltype.lower():
            species = "CMCNA"
        idx = np.asarray(inst.get("atom_indices_0"), dtype=int)
        charges = np.asarray(inst.get("charges"), dtype=float)
        names = [str(x) for x in inst.get("atomnames") or []]
        atypes = [str(x) for x in inst.get("atomtypes") or []]
        if idx.size == 0:
            continue
        if species == "LI" or (idx.size == 1 and ("li" in moltype.lower() or (charges.size and float(charges[0]) > 0.3))):
            li.append(int(idx[0]))
            continue
        if species in {"EC", "EMC", "DEC"}:
            candidates: list[tuple[float, int]] = []
            for local, atom_idx in enumerate(idx):
                charge = float(charges[local]) if local < charges.size else 0.0
                name = names[local] if local < len(names) else ""
                atype = atypes[local] if local < len(atypes) else ""
                if _atom_is_oxygen_like(name, atype, charge):
                    candidates.append((charge, int(atom_idx)))
            if candidates:
                solvent_o.append(int(sorted(candidates, key=lambda item: item[0])[0][1]))
            continue
        if species == "CMCNA":
            for local, atom_idx in enumerate(idx):
                charge = float(charges[local]) if local < charges.size else 0.0
                name = names[local] if local < len(names) else ""
                atype = atypes[local] if local < len(atypes) else ""
                if _atom_is_oxygen_like(name, atype, charge):
                    cmc_o.append(int(atom_idx))
            continue
        if species == "PF6":
            for local, atom_idx in enumerate(idx):
                charge = float(charges[local]) if local < charges.size else 0.0
                name = names[local] if local < len(names) else ""
                atype = atypes[local] if local < len(atypes) else ""
                if _atom_is_fluorine_like(name, atype, charge):
                    pf6_f.append(int(atom_idx))
    return {
        "li": np.asarray(sorted(set(li)), dtype=int),
        "solvent_o": np.asarray(sorted(set(solvent_o)), dtype=int),
        "cmc_o": np.asarray(sorted(set(cmc_o)), dtype=int),
        "pf6_f": np.asarray(sorted(set(pf6_f)), dtype=int),
    }


def _count_xy_pbc_z_open_neighbors(
    *,
    coords: np.ndarray,
    center_index: int,
    target_indices: np.ndarray,
    box: tuple[float, float, float],
    cutoff_nm: float,
) -> int:
    if target_indices.size == 0:
        return 0
    target = np.asarray(coords[target_indices], dtype=float)
    center = np.asarray(coords[int(center_index)], dtype=float)
    delta = target - center[None, :]
    lx, ly = max(float(box[0]), 1.0e-12), max(float(box[1]), 1.0e-12)
    delta[:, 0] -= lx * np.round(delta[:, 0] / lx)
    delta[:, 1] -= ly * np.round(delta[:, 1] / ly)
    dist2 = np.einsum("ij,ij->i", delta, delta)
    return int(np.count_nonzero(dist2 <= float(cutoff_nm) ** 2))


def _li_solvation_depth_profiles(payloads: list[CasePayload], fig_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    rows_out: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    cutoff_nm = float(os.environ.get("EG08_LI_SOLVATION_CUTOFF_NM", "0.32"))
    depth_bin_nm = float(os.environ.get("EG08_LI_SOLVATION_DEPTH_BIN_NM", "0.10"))
    frame_stride = max(1, int(os.environ.get("EG08_LI_SOLVATION_FRAME_STRIDE", os.environ.get("EG08_ANALYSIS_FRAME_STRIDE", "4"))))
    for item in payloads:
        bounds = _membrane_bounds(item)
        final_dir = _final_nvt_dir(item)
        top_path = _find_topology_file(item, final_dir)
        gro = final_dir / "md.gro"
        xtc = final_dir / "md.xtc"
        if bounds is None or top_path is None or not gro.is_file():
            summary_rows.append(
                {
                    "case": item.label,
                    "available": False,
                    "reason": "missing_membrane_bounds_topology_or_final_gro",
                    "final_dir": str(final_dir),
                }
            )
            continue
        try:
            from yadonpy.gmx.analysis.interface_profile import _atom_payload, _iter_frames
            from yadonpy.gmx.topology import parse_system_top

            top = parse_system_top(top_path)
            atom_payload = _atom_payload(top, top_path.parent)
            site_indices = _li_solvation_indices(list(atom_payload.get("instances") or []))
        except Exception as exc:
            summary_rows.append({"case": item.label, "available": False, "reason": f"topology_parse_failed: {exc}", "topology": str(top_path)})
            continue
        li_indices = site_indices["li"]
        if li_indices.size == 0:
            summary_rows.append({"case": item.label, "available": False, "reason": "no_li_indices", "topology": str(top_path)})
            continue
        membrane_lo, membrane_hi, feed_side = bounds
        thickness = max(0.0, membrane_hi - membrane_lo)
        edges = np.arange(0.0, thickness + depth_bin_nm * 1.5, depth_bin_nm, dtype=float)
        if edges.size < 2:
            edges = np.asarray([0.0, max(depth_bin_nm, thickness)], dtype=float)
        sums = {target: np.zeros(edges.size - 1, dtype=float) for target in LI_SOLVATION_TARGETS}
        sample_counts = np.zeros(edges.size - 1, dtype=float)
        frame_count = 0
        li_inside_total = 0
        try:
            frame_iter = _iter_frames(gro_path=gro, xtc_path=xtc if xtc.is_file() else None, frame_stride=frame_stride, chunk=25)
            for _time_ps, coords, box in frame_iter:
                frame_count += 1
                z = np.asarray(coords[li_indices, 2], dtype=float)
                inside = (z >= membrane_lo) & (z <= membrane_hi)
                if not np.any(inside):
                    continue
                for li_idx, z_li in zip(li_indices[inside], z[inside]):
                    depth = (membrane_hi - float(z_li)) if feed_side == "above" else (float(z_li) - membrane_lo)
                    if depth < 0.0:
                        continue
                    bin_idx = int(np.searchsorted(edges, depth, side="right") - 1)
                    if bin_idx < 0 or bin_idx >= sample_counts.size:
                        continue
                    sample_counts[bin_idx] += 1.0
                    li_inside_total += 1
                    for target in LI_SOLVATION_TARGETS:
                        sums[target][bin_idx] += float(
                            _count_xy_pbc_z_open_neighbors(
                                coords=coords,
                                center_index=int(li_idx),
                                target_indices=site_indices[target],
                                box=box,
                                cutoff_nm=cutoff_nm,
                            )
                        )
        except Exception as exc:
            summary_rows.append({"case": item.label, "available": False, "reason": f"trajectory_scan_failed: {exc}", "final_dir": str(final_dir)})
            continue
        for bin_idx in range(sample_counts.size):
            n = float(sample_counts[bin_idx])
            means = {target: (float(sums[target][bin_idx]) / n if n > 0 else np.nan) for target in LI_SOLVATION_TARGETS}
            total_cn = float(sum(v for v in means.values() if np.isfinite(v)))
            row = {
                "case": item.label,
                "charge_uC_cm2": float(item.charge),
                "depth_bin_lo_nm": float(edges[bin_idx]),
                "depth_bin_hi_nm": float(edges[bin_idx + 1]),
                "depth_mid_nm": float(0.5 * (edges[bin_idx] + edges[bin_idx + 1])),
                "li_frame_samples": int(n),
                "cutoff_nm": cutoff_nm,
                "feed_side": feed_side,
                "cn_solvent_o": means["solvent_o"],
                "cn_cmc_o": means["cmc_o"],
                "cn_pf6_f": means["pf6_f"],
                "cn_total": total_cn if n > 0 else np.nan,
                "fraction_solvent_o": means["solvent_o"] / total_cn if total_cn > 0 and np.isfinite(means["solvent_o"]) else np.nan,
                "fraction_cmc_o": means["cmc_o"] / total_cn if total_cn > 0 and np.isfinite(means["cmc_o"]) else np.nan,
                "fraction_pf6_f": means["pf6_f"] / total_cn if total_cn > 0 and np.isfinite(means["pf6_f"]) else np.nan,
            }
            rows_out.append(row)
        summary_rows.append(
            {
                "case": item.label,
                "available": True,
                "frame_count": int(frame_count),
                "frame_stride": int(frame_stride),
                "li_atom_count": int(li_indices.size),
                "li_inside_cmc_frame_samples": int(li_inside_total),
                "membrane_z_lo_nm": float(membrane_lo),
                "membrane_z_hi_nm": float(membrane_hi),
                "feed_side": feed_side,
                "cutoff_nm": cutoff_nm,
            }
        )
    _write_rows(fig_dir / "li_solvation_by_cmc_depth.csv", rows_out)
    _write_rows(fig_dir / "li_solvation_by_cmc_depth_summary.csv", summary_rows)
    plots = _plot_li_solvation_depth(rows_out, payloads, fig_dir)
    return summary_rows, plots


def _plot_li_solvation_depth(rows: list[dict[str, Any]], payloads: list[CasePayload], fig_dir: Path) -> dict[str, Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    valid_rows = [row for row in rows if _safe_float(row.get("li_frame_samples"), 0.0) > 0]
    outputs: dict[str, Path] = {}
    if not valid_rows:
        return outputs
    cn_keys = {
        "solvent_o": "cn_solvent_o",
        "cmc_o": "cn_cmc_o",
        "pf6_f": "cn_pf6_f",
    }
    frac_keys = {
        "solvent_o": "fraction_solvent_o",
        "cmc_o": "fraction_cmc_o",
        "pf6_f": "fraction_pf6_f",
    }
    max_cn = max((_safe_float(row.get(key), 0.0) for row in valid_rows for key in cn_keys.values()), default=1.0)
    ymax = max(6.0, max_cn * 1.15)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8), sharex=True, sharey=True)
    for ax, item in zip(axes.flat, payloads):
        case_rows = [row for row in valid_rows if row.get("case") == item.label]
        for target, key in cn_keys.items():
            x = np.asarray([_safe_float(row.get("depth_mid_nm")) for row in case_rows], dtype=float)
            y = np.asarray([_safe_float(row.get(key)) for row in case_rows], dtype=float)
            samples = np.asarray([_safe_float(row.get("li_frame_samples"), 0.0) for row in case_rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (samples > 0)
            if np.any(mask):
                ax.plot(x[mask], y[mask], marker="o", ms=3.0, lw=1.6, color=LI_SOLVATION_PALETTE[target], label=target.replace("_", "-"))
        ax.axhline(4.0, color="#808080", lw=0.9, ls=":", alpha=0.8)
        ax.axhline(6.0, color="#808080", lw=0.9, ls="--", alpha=0.8)
        ax.set_ylim(0.0, ymax)
        ax.set_title(item.label)
        ax.grid(True, alpha=0.22)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=7)
    fig.supxlabel("Li penetration depth into CMC from electrolyte-side boundary / nm")
    fig.supylabel("mean coordination count within cutoff")
    fig.suptitle("Depth-resolved Li solvation structure inside CMC membrane")
    fig.tight_layout()
    out = _save_fig(fig, fig_dir / "li_solvation_cn_by_cmc_depth")
    outputs["cn_depth"] = out["png"]
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), sharex=True, sharey=True)
    for ax, target in zip(axes, LI_SOLVATION_TARGETS):
        key = cn_keys[target]
        for item in payloads:
            case_rows = [row for row in valid_rows if row.get("case") == item.label]
            x = np.asarray([_safe_float(row.get("depth_mid_nm")) for row in case_rows], dtype=float)
            y = np.asarray([_safe_float(row.get(key)) for row in case_rows], dtype=float)
            samples = np.asarray([_safe_float(row.get("li_frame_samples"), 0.0) for row in case_rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (samples > 0)
            if np.any(mask):
                ax.plot(x[mask], y[mask], marker="o", ms=2.8, lw=1.5, color=_charge_color(item.charge), label=item.label)
        ax.axhline(4.0, color="#808080", lw=0.9, ls=":", alpha=0.8)
        ax.axhline(6.0, color="#808080", lw=0.9, ls="--", alpha=0.8)
        ax.set_title(target.replace("_", "-"))
        ax.set_ylim(0.0, ymax)
        ax.grid(True, alpha=0.22)
    axes[0].legend(frameon=False, fontsize=7)
    fig.supxlabel("Li penetration depth into CMC from electrolyte-side boundary / nm")
    fig.supylabel("mean coordination count within cutoff")
    fig.suptitle("Depth-resolved Li solvation: charge sweep by coordinating site")
    fig.tight_layout()
    out = _save_fig(fig, fig_dir / "li_solvation_cn_by_site_charge_facets")
    outputs["cn_site_charge_facets"] = out["png"]
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8), sharex=True, sharey=True)
    for ax, item in zip(axes.flat, payloads):
        case_rows = [row for row in valid_rows if row.get("case") == item.label]
        for target, key in frac_keys.items():
            x = np.asarray([_safe_float(row.get("depth_mid_nm")) for row in case_rows], dtype=float)
            y = np.asarray([_safe_float(row.get(key)) for row in case_rows], dtype=float)
            samples = np.asarray([_safe_float(row.get("li_frame_samples"), 0.0) for row in case_rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (samples > 0)
            if np.any(mask):
                ax.plot(x[mask], y[mask], marker="o", ms=3.0, lw=1.6, color=LI_SOLVATION_PALETTE[target], label=target.replace("_", "-"))
        ax.set_ylim(0.0, 1.0)
        ax.set_title(item.label)
        ax.grid(True, alpha=0.22)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=7)
    fig.supxlabel("Li penetration depth into CMC from electrolyte-side boundary / nm")
    fig.supylabel("coordination composition fraction")
    fig.suptitle("Depth-resolved Li solvation composition inside CMC membrane")
    fig.tight_layout()
    out = _save_fig(fig, fig_dir / "li_solvation_fraction_by_cmc_depth")
    outputs["fraction_depth"] = out["png"]
    plt.close(fig)
    return outputs


def _plot_edl_overlays(payloads: list[CasePayload], fig_dir: Path) -> dict[str, Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    columns = (
        ("charge_density_e_nm3", "charge density / e nm$^{-3}$", "edl_charge_density_zrel"),
        ("integrated_charge_e_nm2", "integrated charge / e nm$^{-2}$", "edl_integrated_charge_zrel"),
        ("electrostatic_potential_V", "potential / V", "edl_potential_zrel"),
    )
    outputs: dict[str, Path] = {}
    all_x: list[float] = []
    series_by_col: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {col: {} for col, _yl, _name in columns}
    for item in payloads:
        rows = _read_rows(item.analysis_dir / "electrostatic_potential.csv")
        out_rows = []
        for row in rows:
            z = _safe_float(row.get("z_nm"))
            if not np.isfinite(z):
                continue
            z_rel = z - item.z0_nm
            all_x.append(float(z_rel))
            out_rows.append({**row, "z_rel_nm": z_rel})
        _write_rows(fig_dir / "zrel_csv" / f"{item.dirname}_electrostatic_potential_zrel.csv", out_rows)
        for col, _yl, _name in columns:
            x = np.asarray([_safe_float(row.get("z_rel_nm")) for row in out_rows], dtype=float)
            y = np.asarray([_safe_float(row.get(col)) for row in out_rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            series_by_col[col][item.label] = (x[mask], y[mask])
    xlim = (min(all_x), max(all_x)) if all_x else (0.0, 1.0)
    for col, ylabel, name in columns:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ys = []
        for item in payloads:
            x, y = series_by_col[col].get(item.label, (np.asarray([]), np.asarray([])))
            if x.size:
                ax.plot(x, y, lw=1.8, color=_charge_color(item.charge), label=item.label)
                ys.extend([float(np.nanmin(y)), float(np.nanmax(y))])
        ax.set_xlim(*xlim)
        if ys:
            pad = (max(ys) - min(ys)) * 0.08 or 1.0
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_xlabel("z_rel / nm (0 = CMC-facing graphite surface)")
        ax.set_ylabel(ylabel)
        ax.set_title(name.replace("_", " "))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        out = _save_fig(fig, fig_dir / name)
        outputs[name] = out["png"]
        plt.close(fig)
    return outputs


def _plot_rdf_cn_last_window(payloads: list[CasePayload], fig_dir: Path) -> dict[str, Path]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    pairs = ("cation-polymer_o", "cation-solvent_o", "cation-anion_f")
    outputs: dict[str, Path] = {}
    for pair in pairs:
        fig, ax = plt.subplots(figsize=(7.2, 4.1))
        ax_cn = ax.twinx()
        for item in payloads:
            rows = [row for row in _read_rows(item.analysis_dir / "time_series" / "rdf_cn_curves_timeseries.csv") if row.get("pair") == pair]
            if not rows:
                continue
            max_end = max(_safe_float(row.get("time_end_ps"), -1.0) for row in rows)
            selected = [row for row in rows if abs(_safe_float(row.get("time_end_ps"), -1.0) - max_end) < 1.0e-6]
            r = np.asarray([_safe_float(row.get("r_nm")) for row in selected], dtype=float)
            g = np.asarray([_safe_float(row.get("g_r")) for row in selected], dtype=float)
            cn = np.asarray([_safe_float(row.get("cn_r")) for row in selected], dtype=float)
            mask = np.isfinite(r) & np.isfinite(g) & np.isfinite(cn)
            if not np.any(mask):
                continue
            color = _charge_color(item.charge)
            ax.plot(r[mask], g[mask], lw=1.6, color=color, label=f"{item.label} RDF")
            ax_cn.plot(r[mask], cn[mask], lw=1.4, ls="--", color=color, alpha=0.9)
            if np.any(mask):
                peak_idx = int(np.nanargmax(g[mask]))
                rr = r[mask][peak_idx]
                gg = g[mask][peak_idx]
                ax.annotate(f"{rr:.2f} nm", xy=(rr, gg), xytext=(rr, gg * 1.05), fontsize=7, color=color)
        ax.set_xlabel("r / nm")
        ax.set_ylabel("RDF g(r), solid")
        ax_cn.set_ylabel("CN(r), dashed")
        ax_cn.set_ylim(0.0, 6.0)
        ax.set_title(f"Interface RDF/CN last window: {pair}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        out = _save_fig(fig, fig_dir / f"rdf_cn_last_window_{pair.replace('-', '_')}")
        outputs[pair] = out["png"]
        plt.close(fig)
    return outputs


def _add_textbox(slide, left, top, width, height, text: str, font_size: int = 13) -> None:
    from pptx.util import Pt

    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for idx, line in enumerate(str(text).splitlines() or [""]):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = line
        for run in p.runs:
            run.font.name = "Arial"
            run.font.size = Pt(font_size)


def _add_picture(slide, path: Path | None, left, top, width=None, height=None) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        slide.shapes.add_picture(str(path), left, top, width=width, height=height)
        return True
    except Exception:
        return False


def _add_movie_tile(slide, mp4_path: Path, poster: Path | None, left, top, width, height) -> None:
    from pptx.util import Inches

    shown = _add_picture(slide, poster, left, top, width=width, height=height)
    if mp4_path.is_file():
        try:
            slide.shapes.add_movie(
                str(mp4_path),
                left + width - Inches(0.82),
                top + Inches(0.07),
                Inches(0.72),
                Inches(0.50),
                poster_frame_image=str(poster) if poster and poster.is_file() else None,
                mime_type="video/mp4",
            )
        except Exception:
            pass
        _add_textbox(slide, left + Inches(0.07), top + height - Inches(0.28), width - Inches(0.14), Inches(0.22), f"MP4: {mp4_path.name}", font_size=7)
    elif not shown:
        _add_textbox(slide, left, top, width, height, f"Missing MP4/poster:\n{mp4_path}", font_size=9)


def _add_table(slide, rows: list[list[Any]], left, top, width, height, *, font_size: int = 8) -> None:
    from pptx.util import Pt

    if not rows:
        return
    table = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width, height).table
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(font_size)


def _result_slide(prs, title: str, image: Path | None, text: str, *, source_svg: Path | None = None):
    from pptx.util import Inches

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    shown = _add_picture(slide, image, Inches(0.45), Inches(0.82), width=Inches(7.25))
    if not shown:
        _add_table(
            slide,
            [["figure status", "not generated"], ["expected source", source_svg or image or "not available"]],
            Inches(0.75),
            Inches(1.25),
            Inches(6.4),
            Inches(1.1),
            font_size=9,
        )
    note = text
    if source_svg is not None:
        note += f"\n\nSVG source:\n{source_svg}"
    _add_textbox(slide, Inches(8.0), Inches(0.9), Inches(5.0), Inches(5.95), note, font_size=12)
    return slide


def _audit_ppt(path: Path, out_json: Path) -> dict[str, Any]:
    from pptx import Presentation

    prs = Presentation(str(path))
    slides = []
    ok = True
    for idx, slide in enumerate(prs.slides, start=1):
        title = ""
        pics = movies = charts = tables = 0
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and not title:
                title = shape.text_frame.text.strip().split("\n")[0]
            if shape.shape_type == 13:
                pics += 1
            elif shape.shape_type in (16, 17):
                movies += 1
            elif shape.shape_type == 3:
                charts += 1
            elif shape.shape_type == 19:
                tables += 1
        visual = pics + movies + charts + tables
        data_slide = idx > 2 and "Notes" not in title
        if data_slide and visual == 0:
            ok = False
        slides.append({"index": idx, "title": title, "pictures": pics, "movies": movies, "charts": charts, "tables": tables, "visual_count": visual})
    payload = {"ok": ok, "ppt": str(path), "slides": slides, "size_mb": path.stat().st_size / 1024 / 1024}
    _write_json(out_json, payload)
    return payload


def _trajectory_assets(item: CasePayload, fig_dir: Path) -> tuple[Path | None, Path | None]:
    movie_dir = fig_dir / "trajectory_overview"
    mp4 = movie_dir / f"{item.dirname}_trajectory_overview.mp4"
    poster = movie_dir / item.dirname / "frame_000.png"
    if mp4.is_file() and poster.is_file():
        return mp4, poster
    return None, None


def main() -> None:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    _configure_matplotlib()
    root = Path(os.environ.get("EG08_SWEEP_ROOT", ".")).resolve()
    out_dir = root / "99_report"
    fig_dir = out_dir / "ppt_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    payloads = _load_payloads(root, DEFAULT_CASES)

    t0_validation = _validate_t0(payloads, out_dir)
    z_arrays = _aggregate_z_profiles(payloads, fig_dir)
    z_by_charge = _plot_z_facets_by_charge(z_arrays, payloads, fig_dir)
    z_by_species = _plot_z_facets_by_species(z_arrays, payloads, fig_dir)
    edl = _plot_edl_overlays(payloads, fig_dir)
    mem_data = _membrane_timeseries(payloads, fig_dir)
    mem_by_species, mem_by_charge = _plot_membrane_fraction(mem_data, payloads, fig_dir)
    penetration_metrics = _penetration_metrics(payloads, fig_dir)
    penetration_plots = _plot_penetration_metrics(penetration_metrics, payloads, fig_dir)
    li_solvation_summary, li_solvation_plots = _li_solvation_depth_profiles(payloads, fig_dir)
    rdf_plots = _plot_rdf_cn_last_window(payloads, fig_dir)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Eg08.07 CMC-facing charge sweep"
    slide.placeholders[1].text = f"Strict post-processing report\nroot: {root}\nvalid_shared_t0={t0_validation.get('ok')}"

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Shared t=0 Validation"
    rows = [["case", "02_system/system.gro md5", "shared t0 OK"]]
    ref_md5 = None
    for row in t0_validation.get("system_gro_md5") or []:
        ref_md5 = ref_md5 or row.get("md5")
        rows.append([row.get("case"), row.get("md5"), bool(row.get("md5") == ref_md5)])
    if len(rows) == 1:
        rows.append(["not available", t0_validation.get("reason"), False])
    _add_table(slide, rows, Inches(0.45), Inches(0.85), Inches(12.4), Inches(1.7), font_size=8)
    traj = t0_validation.get("trajectory_first_frame") or {}
    traj_rows = [["trajectory first-frame diagnostic", "value"]]
    traj_rows.append(["available", traj.get("available")])
    traj_rows.append(["identical first saved xtc frame", traj.get("ok")])
    if traj.get("comparisons"):
        worst = max(float(row.get("first_frame_rmsd_nm") or 0.0) for row in traj.get("comparisons") or [])
        traj_rows.append(["worst first-frame RMSD / nm", f"{worst:.3e}"])
    else:
        traj_rows.append(["reason/note", traj.get("reason") or traj.get("note") or "not checked"])
    _add_table(slide, traj_rows, Inches(0.45), Inches(2.75), Inches(5.6), Inches(1.1), font_size=8)
    status = "PASS" if t0_validation.get("ok") else "FAIL"
    _add_textbox(
        slide,
        Inches(6.35),
        Inches(2.75),
        Inches(6.35),
        Inches(2.7),
        f"Observation: shared t=0 validation status is {status}.\n"
        "Analysis: the strict criterion is identical 02_system/system.gro files, because this is the shared assembled structure before charge-specific production. The first saved xtc frame is only a diagnostic; it may be after several integration steps.\n"
        "Conclusion: if the system.gro md5 check fails, all following physical trends are diagnostic-only and the four cases must be rerun from one shared_t0 structure.",
        font_size=14,
    )

    _result_slide(
        prs,
        "EDL Charge Density Overlay",
        edl.get("edl_charge_density_zrel"),
        "Observation: all charge states are plotted on the same z_rel axis.\nAnalysis: charge density localizes electrode and ionic charge relative to the CMC-facing graphite surface.\nConclusion: use sign and peak position to identify screening/overscreening.",
        source_svg=(fig_dir / "edl_charge_density_zrel.svg"),
    )
    _result_slide(
        prs,
        "EDL Integrated Charge Overlay",
        edl.get("edl_integrated_charge_zrel"),
        "Observation: integrated charge is the cumulative area charge from z_rel=0.\nAnalysis: plateaus and sign reversals show how the electrolyte/CMC compensates surface charge.\nConclusion: compare these curves only after t=0 validation passes.",
        source_svg=(fig_dir / "edl_integrated_charge_zrel.svg"),
    )
    _result_slide(
        prs,
        "EDL Electrostatic Potential Overlay",
        edl.get("edl_potential_zrel"),
        "Observation: potential is obtained by integrating the charge-derived electric field under the selected reference.\nAnalysis: the absolute zero is arbitrary; the gradient and charge-dependent differences are meaningful.\nConclusion: stronger CMC-facing negative charge should reshape the potential near the CMC-side EDL.",
        source_svg=(fig_dir / "edl_potential_zrel.svg"),
    )

    _result_slide(
        prs,
        "z Distribution: Charge Facets",
        z_by_charge["png"],
        "Observation: each subplot is one surface charge and compares all species.\nAnalysis: step-filled profiles are z-bin histograms, avoiding misleading overlaid line spaghetti.\nConclusion: this view answers which species occupy each region under one charge state.",
        source_svg=z_by_charge["svg"],
    )
    _result_slide(
        prs,
        "z Distribution: Species Facets",
        z_by_species["png"],
        "Observation: each subplot is one species and overlays all charge states.\nAnalysis: shared z_rel axis makes charge-induced redistribution easier to read.\nConclusion: this view answers whether a given species responds monotonically to surface charge.",
        source_svg=z_by_species["svg"],
    )

    for sp, png in mem_by_species.items():
        _result_slide(
            prs,
            f"{sp} Membrane Fraction Across Charges",
            png,
            f"Definition: f_mem({sp},t)=N_membrane({sp},t)/(N_feed+N_membrane+N_permeate).\n"
            "Observation: curves use 1 ns bins with a 3-point rolling mean.\n"
            "Analysis: this is a molecule-count fraction, not mass fraction or permeability.\n"
            "Conclusion: sustained increases indicate membrane uptake; early differences require shared t=0 validation.",
            source_svg=png.with_suffix(".svg"),
        )
    for label, png in mem_by_charge.items():
        _result_slide(
            prs,
            f"{label} Membrane Fraction By Species",
            png,
            "Observation: all permeant species are compared within one charge state.\nAnalysis: solvent/cation/anion asymmetry reveals selective uptake.\nConclusion: this complements the per-species charge-sweep plots.",
            source_svg=png.with_suffix(".svg"),
        )

    for metric, png in penetration_plots.items():
        _result_slide(
            prs,
            f"Penetration Capability: {metric}",
            png,
            "Method: molecule COM is classified as feed, membrane, or permeate relative to the CMCNA membrane interval.\n"
            "Entry event: feed -> membrane transition. Translocation: feed -> membrane -> permeate path.\n"
            "AUC_depth integrates the normalized penetration-depth distribution and summarizes overall penetration ability.",
            source_svg=png.with_suffix(".svg"),
        )

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Li Solvation vs CMC Penetration Depth: Method"
    rows = [["case", "available", "Li-in-CMC samples", "stride", "cutoff / nm", "feed side"]]
    for row in li_solvation_summary:
        rows.append(
            [
                row.get("case"),
                row.get("available"),
                row.get("li_inside_cmc_frame_samples", row.get("reason")),
                row.get("frame_stride", ""),
                row.get("cutoff_nm", ""),
                row.get("feed_side", ""),
            ]
        )
    _add_table(slide, rows[:8], Inches(0.45), Inches(0.82), Inches(12.4), Inches(2.0), font_size=8)
    _add_textbox(
        slide,
        Inches(0.6),
        Inches(3.15),
        Inches(12.0),
        Inches(3.2),
        "Definition: only Li atoms located inside the CMC membrane interval are counted. "
        "Penetration depth is measured from the electrolyte-side CMC boundary toward the CMC interior.\n"
        "Solvation structure: for each depth bin, average the number of solvent representative O sites, CMC oxygen-like sites, and PF6 fluorine sites within the cutoff around Li. "
        "The plotted composition fractions are each component divided by solvent-O + CMC-O + PF6-F coordination.\n"
        "Interpretation: decreasing solvent-O with increasing CMC-O means Li sheds carbonate solvation and coordinates to the CMC matrix; PF6-F growth indicates ion-pair participation inside the membrane.",
        font_size=13,
    )

    for key, title in (
        ("cn_depth", "Li Solvation CN vs CMC Penetration Depth"),
        ("cn_site_charge_facets", "Li Solvation CN: Site Facets Across Charges"),
        ("fraction_depth", "Li Solvation Composition vs Depth"),
    ):
        png = li_solvation_plots.get(key)
        _result_slide(
            prs,
            title,
            png,
            "Observation: curves are binned by Li penetration depth inside CMC, not by simulation time.\n"
            "Analysis: CN values are local coordination counts around Li within the stated cutoff; values above four can occur when broad site classes are counted.\n"
            "Conclusion: this plot directly shows how the average Li solvation shell changes after entering the CMC membrane.",
            source_svg=(png.with_suffix(".svg") if png else None),
        )

    for pair, png in rdf_plots.items():
        _result_slide(
            prs,
            f"Interface RDF/CN: {pair}",
            png,
            "Observation: RDF is solid, CN is dashed on the right axis fixed to 0-6.\n"
            "Analysis: interface RDF uses local slab-aware normalization rather than bulk gmx rdf assumptions.\n"
            "Conclusion: first peak labels report preferred coordination distances; CN above 6 is recorded in CSV rather than stretching the plot.",
            source_svg=png.with_suffix(".svg"),
        )

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Time-Series MP4 Panels"
    positions = [(Inches(0.45), Inches(0.85)), (Inches(6.85), Inches(0.85)), (Inches(0.45), Inches(4.05)), (Inches(6.85), Inches(4.05))]
    for item, (left, top) in zip(payloads, positions):
        mp4 = item.analysis_dir / "time_series" / "rdf_cn_timeseries.mp4"
        poster = item.analysis_dir / "time_series" / "frames" / "rdf_cn" / "frame_000.png"
        _add_movie_tile(slide, mp4, poster, left, top, Inches(5.85), Inches(2.5))
        _add_textbox(slide, left, top + Inches(2.55), Inches(5.85), Inches(0.30), f"{item.label}: RDF/CN time series", font_size=9)

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = "Source Data And Audit"
    _add_table(
        slide,
        [
            ["artifact", "path"],
            ["z_rel CSV", fig_dir / "zrel_csv"],
            ["membrane fraction CSV", fig_dir / "membrane_fraction_1ns_3point_smooth.csv"],
            ["penetration metrics CSV", fig_dir / "penetration_capability_metrics.csv"],
            ["Li solvation-depth CSV", fig_dir / "li_solvation_by_cmc_depth.csv"],
            ["t0 validation JSON", out_dir / "shared_t0_validation.json"],
            ["visual audit JSON", out_dir / "ppt_visual_audit.json"],
        ],
        Inches(0.45),
        Inches(0.9),
        Inches(12.4),
        Inches(2.2),
        font_size=8,
    )
    _add_textbox(
        slide,
        Inches(0.6),
        Inches(3.55),
        Inches(12.0),
        Inches(2.4),
        "Observation: all generated figures have SVG sources and PNG previews for PowerPoint compatibility.\n"
        "Analysis: PPT itself is a reading layer; final plots can be redrawn from the listed CSV/SVG files.\n"
        "Conclusion: do not accept the PPT unless visual audit passes and file size remains below 50 MB.",
        font_size=14,
    )

    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text_frame"):
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.name = "Arial"
                        if run.font.size is None:
                            run.font.size = Pt(12)

    out_ppt = out_dir / "eg08_07_charge_sweep_full_report.pptx"
    prs.save(out_ppt)
    audit = _audit_ppt(out_ppt, out_dir / "ppt_visual_audit.json")
    final = {
        "ppt": str(out_ppt),
        "figure_dir": str(fig_dir),
        "root": str(root),
        "case_count": len(payloads),
        "valid_shared_t0": bool(t0_validation.get("ok")),
        "ppt_size_mb": audit["size_mb"],
        "visual_audit_ok": bool(audit["ok"]),
        "li_solvation_depth_csv": str(fig_dir / "li_solvation_by_cmc_depth.csv"),
    }
    _write_json(out_dir / "charge_sweep_full_ppt_paths.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
