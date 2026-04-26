"""Markov-driven ion migration analysis helpers.

This module powers :meth:`yadonpy.sim.analyzer.AnalyzeResult.migration` and is
designed to be robust across pure electrolytes, polymer-electrolyte composites,
and solid polymer electrolytes. The default path builds two linked Markov
models:

- role-level states: polymer / solvent / anion / none
- anchor/site-level states: specific donor anchors or role-specific OTHER

Residence, event flux, and prediction summaries are all derived from the same
discrete state trajectories generated in a single trajectory scan.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..gmx.analysis.structured import (
    _autocorr_fft,
    _build_adjacency,
    _site_coordination_metadata,
    _site_label_for_atom,
    _unwrap_position_series,
    build_species_catalog,
)
from ..gmx.topology import MoleculeType, SystemTopology


ROLE_ORDER = ("polymer", "solvent", "anion")
ROLE_STATE_ORDER = ROLE_ORDER + ("none",)
ROLE_TO_STATE = {role: idx for idx, role in enumerate(ROLE_STATE_ORDER)}
ROLE_LABELS = {
    "polymer": "polymer_donor",
    "solvent": "solvent_donor",
    "anion": "anion_donor",
    "none": "uncoordinated",
}
DEFAULT_SITE_CUTOFF_NM = {
    "ether_oxygen": 0.34,
    "carbonyl_oxygen": 0.34,
    "carboxylate_oxygen": 0.31,
    "sulfonyl_oxygen": 0.31,
    "phosphate_or_oxo_oxygen": 0.31,
    "oxo_anion_oxygen": 0.31,
    "oxygen_site": 0.34,
    "nitrogen_site": 0.30,
    "anion_nitrogen": 0.30,
    "coordination_fluorine": 0.28,
    "fluorine_site": 0.28,
}
EVENT_TYPES = (
    "site_stay",
    "site_exchange",
    "polymer_intrachain_hop",
    "polymer_interchain_hop",
    "solvent_assisted_exchange",
    "anion_assisted_exchange",
    "ambiguous_exchange",
)
DEFAULT_ROLE_LAG_MULTIPLIERS = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_PREDICTION_STEP_MULTIPLIERS = (1, 5, 10, 20)
MARKOV_MAX_SITE_STATES_PER_ROLE = 12
MARKOV_MIN_SITE_OCCUPANCY_FRACTION = 0.002
MARKOV_MIN_SITE_TRANSITIONS = 2
MARKOV_ACTIVE_STATE_PLOT_LIMIT = 28


@dataclass(frozen=True)
class AnchorSpec:
    anchor_id: str
    role: str
    moltype: str
    site_id: str
    site_label: str
    atom_indices_0: np.ndarray
    instance_index: int
    residue_number: Optional[int]
    residue_name: Optional[str]
    chain_key: str
    coordination_priority: int
    coordination_relevance: str
    coordination_note: str
    anchor_label: str
    cutoff_nm: float

    def to_record(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "role": self.role,
            "role_label": ROLE_LABELS.get(self.role, self.role),
            "moltype": self.moltype,
            "site_id": self.site_id,
            "site_label": self.site_label,
            "instance_index": int(self.instance_index),
            "residue_number": int(self.residue_number) if self.residue_number is not None else None,
            "residue_name": self.residue_name,
            "chain_key": self.chain_key,
            "coordination_priority": int(self.coordination_priority),
            "coordination_relevance": self.coordination_relevance,
            "coordination_note": self.coordination_note,
            "anchor_label": self.anchor_label,
            "cutoff_nm": float(self.cutoff_nm),
            "n_atoms": int(np.asarray(self.atom_indices_0, dtype=int).size),
        }


@dataclass(frozen=True)
class StateSpec:
    state_index: int
    state_id: str
    state_label: str
    role: str
    bucket: str
    anchor_id: Optional[str]
    site_id: Optional[str]
    moltype: Optional[str]
    chain_key: Optional[str]
    occupancy_fraction: float
    note: Optional[str] = None

    def to_record(self) -> dict[str, Any]:
        return {
            "state_index": int(self.state_index),
            "state_id": self.state_id,
            "state_label": self.state_label,
            "role": self.role,
            "role_label": ROLE_LABELS.get(self.role, self.role),
            "bucket": self.bucket,
            "anchor_id": self.anchor_id,
            "site_id": self.site_id,
            "moltype": self.moltype,
            "chain_key": self.chain_key,
            "occupancy_fraction": float(self.occupancy_fraction),
            "note": self.note,
        }


def _dedupe_keep_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        token = str(raw or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _explicit_moltypes(
    resolve_moltypes: Callable[[object], Sequence[str]],
    mols: object | None,
) -> Optional[list[str]]:
    if mols is None:
        return None
    return _dedupe_keep_order(resolve_moltypes(mols))


def infer_role_moltypes(
    *,
    catalog: dict[str, dict[str, Any]],
    center_moltype: str,
    resolve_moltypes: Callable[[object], Sequence[str]],
    polymer_mols: object | None = None,
    solvent_mols: object | None = None,
    anion_mols: object | None = None,
    cation_mols: object | None = None,
) -> dict[str, list[str]]:
    """Infer donor roles from species metadata, with user overrides winning."""

    center_token = str(center_moltype)
    explicit_polymer = _explicit_moltypes(resolve_moltypes, polymer_mols)
    explicit_solvent = _explicit_moltypes(resolve_moltypes, solvent_mols)
    explicit_anion = _explicit_moltypes(resolve_moltypes, anion_mols)
    explicit_cation = _explicit_moltypes(resolve_moltypes, cation_mols)

    def _auto_pick(role: str) -> list[str]:
        out: list[str] = []
        for moltype, payload in catalog.items():
            if str(moltype) == center_token:
                continue
            kind = str(payload.get("kind") or "").strip().lower()
            formal_q = float(payload.get("formal_charge_e") or 0.0)
            if role == "polymer":
                if kind == "polymer":
                    out.append(str(moltype))
                continue
            if role == "anion":
                if kind == "anion" or formal_q < -1.0e-12:
                    out.append(str(moltype))
                continue
            if role == "solvent":
                if kind == "solvent":
                    out.append(str(moltype))
                    continue
                if kind in {"polymer", "anion", "cation"}:
                    continue
                if abs(formal_q) <= 1.0e-12:
                    out.append(str(moltype))
                continue
        return _dedupe_keep_order(out)

    roles = {
        "polymer": explicit_polymer if explicit_polymer is not None else _auto_pick("polymer"),
        "solvent": explicit_solvent if explicit_solvent is not None else _auto_pick("solvent"),
        "anion": explicit_anion if explicit_anion is not None else _auto_pick("anion"),
        "cation": explicit_cation if explicit_cation is not None else [],
    }
    for role_name, moltypes in list(roles.items()):
        roles[role_name] = [mt for mt in _dedupe_keep_order(moltypes) if mt in catalog and mt != center_token]
    return roles


def _site_cutoff_from_rdf_record(site_label: str, rdf_record: dict[str, Any] | None) -> float:
    rec = dict(rdf_record or {})
    candidates = [
        rec.get("r_shell_nm"),
        rec.get("r_peak_nm"),
        DEFAULT_SITE_CUTOFF_NM.get(str(site_label or "").strip().lower(), None),
        0.33,
    ]
    for value in candidates:
        try:
            num = float(value)
        except Exception:
            continue
        if num > 0.0:
            return num
    return 0.33


def build_anchor_catalog(
    *,
    top: SystemTopology,
    system_dir: Path,
    center_moltype: str,
    role_moltypes: dict[str, Sequence[str]],
    rdf_summary: dict[str, Any] | None = None,
    include_h: bool = False,
    include_weak_sites: bool = False,
) -> dict[str, Any]:
    """Build donor anchor catalog using existing site-label infrastructure."""

    catalog = build_species_catalog(top, system_dir)
    rdf_payload = dict(rdf_summary or {})
    anchors: list[AnchorSpec] = []
    role_site_counts: dict[str, int] = {role: 0 for role in ROLE_ORDER}

    role_lookup: dict[str, str] = {}
    for role in ROLE_ORDER:
        for moltype in role_moltypes.get(role, []) or []:
            role_lookup[str(moltype)] = role

    for moltype, payload in catalog.items():
        role = role_lookup.get(str(moltype))
        if role is None or str(moltype) == str(center_moltype):
            continue
        mt: MoleculeType | None = payload.get("moleculetype")
        if mt is None:
            continue
        adj = _build_adjacency(mt)
        charge_group_lookup: dict[int, dict[str, Any]] = {}
        for grp in payload.get("charge_groups", []) or []:
            for idx in grp.get("atom_indices", []) or []:
                charge_group_lookup[int(idx)] = dict(grp)

        local_site_map: dict[str, list[int]] = {}
        for atom_idx0 in range(int(payload.get("natoms") or 0)):
            site_label = _site_label_for_atom(mt, atom_idx0, adj, charge_group_lookup, include_h=include_h)
            if not site_label:
                continue
            coord_meta = _site_coordination_metadata(site_label)
            if (not include_weak_sites) and str(coord_meta.get("coordination_relevance") or "") == "weak":
                continue
            local_site_map.setdefault(str(site_label), []).append(int(atom_idx0))

        residue_entries = list(((payload.get("residue_map") or {}).get("residues", []) or []))
        if role == "polymer" and residue_entries:
            for inst in payload.get("instances", []) or []:
                atom_offset = int(np.asarray(inst.get("atom_indices_0"), dtype=int)[0])
                chain_key = f"{moltype}:chain:{int(inst.get('instance_index', 0))}"
                for residue in residue_entries:
                    residue_atoms = set(int(i) for i in residue.get("atom_indices", []) or [])
                    residue_number = int(residue.get("residue_number") or 0)
                    residue_name = str(residue.get("residue_name") or "RES")
                    for site_label, local_indices in local_site_map.items():
                        local_sel = np.asarray([idx for idx in local_indices if idx in residue_atoms], dtype=int)
                        if local_sel.size == 0:
                            continue
                        site_id = f"{moltype}:{site_label}"
                        rdf_record = rdf_payload.get(site_id) if isinstance(rdf_payload.get(site_id), dict) else {}
                        coord_meta = _site_coordination_metadata(site_label)
                        cutoff_nm = _site_cutoff_from_rdf_record(site_label, rdf_record)
                        anchors.append(
                            AnchorSpec(
                                anchor_id=f"{role}:{moltype}:{int(inst.get('instance_index', 0))}:{residue_number}:{site_label}",
                                role=role,
                                moltype=str(moltype),
                                site_id=site_id,
                                site_label=str(site_label),
                                atom_indices_0=atom_offset + local_sel,
                                instance_index=int(inst.get("instance_index", 0)),
                                residue_number=residue_number,
                                residue_name=residue_name,
                                chain_key=chain_key,
                                coordination_priority=int(coord_meta.get("coordination_priority") or 0),
                                coordination_relevance=str(coord_meta.get("coordination_relevance") or ""),
                                coordination_note=str(coord_meta.get("coordination_note") or ""),
                                anchor_label=f"{moltype}[{int(inst.get('instance_index', 0))}]/{residue_name}{residue_number}:{site_label}",
                                cutoff_nm=float(cutoff_nm),
                            )
                        )
                        role_site_counts[role] += 1
            continue

        for inst in payload.get("instances", []) or []:
            atom_offset = int(np.asarray(inst.get("atom_indices_0"), dtype=int)[0])
            inst_index = int(inst.get("instance_index", 0))
            for site_label, local_indices in local_site_map.items():
                local_sel = np.asarray(local_indices, dtype=int)
                if local_sel.size == 0:
                    continue
                site_id = f"{moltype}:{site_label}"
                rdf_record = rdf_payload.get(site_id) if isinstance(rdf_payload.get(site_id), dict) else {}
                coord_meta = _site_coordination_metadata(site_label)
                cutoff_nm = _site_cutoff_from_rdf_record(site_label, rdf_record)
                anchors.append(
                    AnchorSpec(
                        anchor_id=f"{role}:{moltype}:{inst_index}:{site_label}",
                        role=role,
                        moltype=str(moltype),
                        site_id=site_id,
                        site_label=str(site_label),
                        atom_indices_0=atom_offset + local_sel,
                        instance_index=inst_index,
                        residue_number=None,
                        residue_name=None,
                        chain_key=f"{moltype}:mol:{inst_index}",
                        coordination_priority=int(coord_meta.get("coordination_priority") or 0),
                        coordination_relevance=str(coord_meta.get("coordination_relevance") or ""),
                        coordination_note=str(coord_meta.get("coordination_note") or ""),
                        anchor_label=f"{moltype}[{inst_index}]:{site_label}",
                        cutoff_nm=float(cutoff_nm),
                    )
                )
                role_site_counts[role] += 1

    anchors.sort(
        key=lambda item: (
            ROLE_ORDER.index(item.role) if item.role in ROLE_ORDER else 99,
            item.coordination_priority,
            item.moltype,
            item.instance_index,
            item.residue_number if item.residue_number is not None else -1,
            item.site_label,
        )
    )
    return {
        "anchors": anchors,
        "role_site_counts": role_site_counts,
        "catalog": catalog,
    }


def _build_center_groups(center_entry: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    natoms = int(center_entry.get("natoms") or 0)
    masses = np.asarray(center_entry.get("masses"), dtype=float)
    if masses.size != natoms or not np.isfinite(masses).all() or float(np.sum(masses)) <= 0.0:
        masses = np.ones(natoms, dtype=float)
    for inst in center_entry.get("instances", []) or []:
        atom_idx = np.asarray(inst.get("atom_indices_0"), dtype=int)
        groups.append(
            {
                "group_id": f"{center_entry.get('moltype')}:{int(inst.get('instance_index', 0))}",
                "atom_indices_0": atom_idx,
                "masses": masses[: atom_idx.size] if atom_idx.size == masses.size else np.ones(atom_idx.size, dtype=float),
                "label": f"{center_entry.get('moltype')}[{int(inst.get('instance_index', 0))}]",
            }
        )
    return groups


def _group_positions_from_xyz(xyz_nm: np.ndarray, groups: Sequence[dict[str, Any]]) -> np.ndarray:
    frames = int(np.asarray(xyz_nm).shape[0])
    out = np.zeros((frames, len(groups), 3), dtype=float)
    xyz = np.asarray(xyz_nm, dtype=float)
    for gi, group in enumerate(groups):
        atom_idx = np.asarray(group.get("atom_indices_0"), dtype=int)
        if atom_idx.size == 1:
            out[:, gi, :] = xyz[:, atom_idx[0], :]
            continue
        masses = np.asarray(group.get("masses"), dtype=float)
        if masses.size != atom_idx.size or not np.isfinite(masses).all() or float(np.sum(masses)) <= 0.0:
            masses = np.ones(atom_idx.size, dtype=float)
        weights = masses / float(np.sum(masses))
        out[:, gi, :] = np.tensordot(xyz[:, atom_idx, :], weights, axes=(1, 0))
    return out


def _run_lengths(binary_1d: np.ndarray) -> list[int]:
    arr = np.asarray(binary_1d, dtype=bool)
    if arr.size == 0:
        return []
    lengths: list[int] = []
    current = 0
    for flag in arr:
        if bool(flag):
            current += 1
        elif current:
            lengths.append(int(current))
            current = 0
    if current:
        lengths.append(int(current))
    return lengths


def summarize_role_residence(
    *,
    role: str,
    time_ps: np.ndarray,
    contact_matrix: np.ndarray | None,
    out_dir: Path,
    available: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not available or contact_matrix is None or np.asarray(contact_matrix).size == 0:
        return {
            "role": role,
            "role_label": ROLE_LABELS.get(role, role),
            "available": False,
            "note": "not present in this system",
            "continuous_residence_time_ps": None,
            "intermittent_residence_time_ps": None,
            "contact_fraction": None,
            "contact_autocorrelation": {"csv": None, "n_points": 0},
        }

    t = np.asarray(time_ps, dtype=float)
    contact = np.asarray(contact_matrix, dtype=bool)
    if contact.ndim == 1:
        contact = contact[:, None]
    dt_ps = float(np.median(np.diff(t))) if t.size >= 2 else 0.0
    contact_fraction = float(np.mean(contact))
    continuous_lengths = []
    for col in range(contact.shape[1]):
        continuous_lengths.extend(_run_lengths(contact[:, col]))
    continuous_res_ps = float(np.mean(continuous_lengths) * dt_ps) if continuous_lengths else 0.0

    ac = _autocorr_fft(contact.astype(float))
    ac_mean = np.mean(ac, axis=1) if ac.ndim == 2 else np.asarray(ac, dtype=float)
    if ac_mean.size and float(ac_mean[0]) > 0.0:
        ac_norm = ac_mean / float(ac_mean[0])
    else:
        ac_norm = np.zeros_like(ac_mean, dtype=float)
    positive = np.where(ac_norm > 0.0)[0]
    stop = int(positive[-1]) if positive.size else 0
    intermittent_ps = float(np.trapezoid(ac_norm[: stop + 1], t[: stop + 1])) if stop > 0 else 0.0

    csv_path = out_dir / f"{role}_contact_autocorrelation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_ps", "contact_autocorrelation"])
        for time_value, corr_value in zip(t.tolist(), ac_norm.tolist()):
            writer.writerow([f"{float(time_value):.8f}", f"{float(corr_value):.12e}"])

    return {
        "role": role,
        "role_label": ROLE_LABELS.get(role, role),
        "available": True,
        "continuous_residence_time_ps": float(continuous_res_ps),
        "intermittent_residence_time_ps": float(intermittent_ps),
        "contact_fraction": float(contact_fraction),
        "contact_autocorrelation": {
            "csv": str(csv_path),
            "n_points": int(ac_norm.size),
        },
    }


def _write_coordination_timeline_svg(
    *,
    time_ps: np.ndarray,
    role_contact_series: dict[str, np.ndarray],
    out_svg: Path,
) -> Optional[Path]:
    if time_ps.size == 0 or not role_contact_series:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for role in ROLE_ORDER:
        series = np.asarray(role_contact_series.get(role), dtype=float)
        if series.size != time_ps.size:
            continue
        ax.plot(time_ps, series, lw=1.5, label=ROLE_LABELS.get(role, role))
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Contact fraction")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Coordination timeline")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg)
    plt.close(fig)
    return out_svg


def _write_residence_curves_svg(
    *,
    residence_summary: dict[str, Any],
    out_svg: Path,
) -> Optional[Path]:
    curves = []
    for role in ROLE_ORDER:
        rec = residence_summary.get(role) or {}
        csv_path = ((rec.get("contact_autocorrelation") or {}).get("csv") if isinstance(rec, dict) else None)
        if not csv_path:
            continue
        try:
            arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            if arr.ndim == 1 and arr.size == 2:
                arr = arr.reshape(1, 2)
            curves.append((role, arr))
        except Exception:
            continue
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for role, arr in curves:
        ax.plot(arr[:, 0], arr[:, 1], lw=1.5, label=ROLE_LABELS.get(role, role))
    ax.set_xlabel("Time lag (ps)")
    ax.set_ylabel("Normalized contact autocorrelation")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Residence autocorrelation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg)
    plt.close(fig)
    return out_svg


def _rdf_coordination_summary(
    *,
    rdf_summary: dict[str, Any],
    anchors: Sequence[AnchorSpec],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_ORDER}
    seen_site_ids: set[str] = set()
    for anchor in anchors:
        if anchor.site_id in seen_site_ids:
            continue
        seen_site_ids.add(anchor.site_id)
        rdf_record = rdf_summary.get(anchor.site_id) if isinstance(rdf_summary.get(anchor.site_id), dict) else {}
        grouped[anchor.role].append(
            {
                "site_id": anchor.site_id,
                "moltype": anchor.moltype,
                "site_label": anchor.site_label,
                "cutoff_nm": float(anchor.cutoff_nm),
                "coordination_priority": int(anchor.coordination_priority),
                "coordination_relevance": anchor.coordination_relevance,
                "coordination_note": anchor.coordination_note,
                "r_peak_nm": rdf_record.get("r_peak_nm"),
                "r_shell_nm": rdf_record.get("r_shell_nm"),
                "formal_cn_shell": rdf_record.get("formal_cn_shell"),
                "rdf_confidence": rdf_record.get("confidence"),
            }
        )
    for role in ROLE_ORDER:
        grouped[role].sort(
            key=lambda item: (
                int(item.get("coordination_priority") or 0),
                str(item.get("moltype") or ""),
                str(item.get("site_label") or ""),
            )
        )
    return grouped


def _choose_stride_auto(n_frames: int, center_count: int, anchor_count: int) -> int:
    work_units = max(1, int(n_frames)) * max(1, int(center_count)) * max(1, int(anchor_count))
    target_units = 2_000_000
    if work_units <= target_units:
        return 1
    return int(max(1, math.ceil(work_units / target_units)))


def _build_role_state_trajectory(
    dominant_anchor_idx: np.ndarray,
    anchors: Sequence[AnchorSpec],
) -> np.ndarray:
    role_states = np.full(np.asarray(dominant_anchor_idx).shape, ROLE_TO_STATE["none"], dtype=int)
    for anchor_idx, anchor in enumerate(anchors):
        role_states[np.asarray(dominant_anchor_idx) == int(anchor_idx)] = ROLE_TO_STATE.get(anchor.role, ROLE_TO_STATE["none"])
    return role_states


def _observed_transition_counts_by_anchor(dominant_anchor_idx: np.ndarray) -> np.ndarray:
    arr = np.asarray(dominant_anchor_idx, dtype=int)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.zeros(int(max(-1, arr.max(initial=-1))) + 1, dtype=int)
    prev = arr[:-1, :]
    nxt = arr[1:, :]
    max_anchor = int(max(prev.max(initial=-1), nxt.max(initial=-1)))
    if max_anchor < 0:
        return np.zeros(0, dtype=int)
    counts = np.zeros(max_anchor + 1, dtype=int)
    mask = (prev >= 0) & (nxt >= 0) & (prev != nxt)
    if np.any(mask):
        ids = np.concatenate([prev[mask], nxt[mask]]).astype(int, copy=False)
        counts = np.bincount(ids, minlength=max_anchor + 1)
    return counts


def _build_site_state_trajectory(
    dominant_anchor_idx: np.ndarray,
    anchors: Sequence[AnchorSpec],
    *,
    max_states_per_role: int = MARKOV_MAX_SITE_STATES_PER_ROLE,
    min_occ_fraction: float = MARKOV_MIN_SITE_OCCUPANCY_FRACTION,
    min_transition_count: int = MARKOV_MIN_SITE_TRANSITIONS,
) -> tuple[np.ndarray, list[StateSpec], dict[int, int]]:
    arr = np.asarray(dominant_anchor_idx, dtype=int)
    total_points = max(1, int(arr.size))
    n_anchors = len(anchors)
    anchor_occ_counts = np.zeros(n_anchors, dtype=int)
    if n_anchors and np.any(arr >= 0):
        anchor_occ_counts = np.bincount(arr[arr >= 0], minlength=n_anchors)
    anchor_transition_counts = _observed_transition_counts_by_anchor(arr)
    anchor_to_state: dict[int, int] = {}
    states: list[StateSpec] = [
        StateSpec(
            state_index=0,
            state_id="none",
            state_label="none",
            role="none",
            bucket="none",
            anchor_id=None,
            site_id=None,
            moltype=None,
            chain_key=None,
            occupancy_fraction=float(np.mean(arr < 0)) if arr.size else 1.0,
            note="No donor anchor within first-shell cutoff.",
        )
    ]
    next_state_idx = 1

    for role in ROLE_ORDER:
        role_anchor_indices = [idx for idx, anchor in enumerate(anchors) if anchor.role == role]
        if not role_anchor_indices:
            continue
        ranked = sorted(
            role_anchor_indices,
            key=lambda idx: (
                -int(anchor_occ_counts[idx]) if idx < anchor_occ_counts.size else 0,
                -int(anchor_transition_counts[idx]) if idx < anchor_transition_counts.size else 0,
                anchors[idx].coordination_priority,
                anchors[idx].anchor_id,
            ),
        )
        keep: list[int] = []
        other_pool: list[int] = []
        for anchor_idx in ranked:
            occ = int(anchor_occ_counts[anchor_idx]) if anchor_idx < anchor_occ_counts.size else 0
            occ_frac = float(occ) / float(total_points)
            trans = int(anchor_transition_counts[anchor_idx]) if anchor_idx < anchor_transition_counts.size else 0
            if len(keep) < int(max_states_per_role) and (occ_frac >= float(min_occ_fraction) or trans >= int(min_transition_count)):
                keep.append(anchor_idx)
            else:
                other_pool.append(anchor_idx)
        if (not keep) and ranked:
            keep = [ranked[0]]
            other_pool = ranked[1:]
        for anchor_idx in keep:
            anchor = anchors[anchor_idx]
            occ = int(anchor_occ_counts[anchor_idx]) if anchor_idx < anchor_occ_counts.size else 0
            states.append(
                StateSpec(
                    state_index=next_state_idx,
                    state_id=f"{role}:{anchor.anchor_id}",
                    state_label=anchor.anchor_label,
                    role=role,
                    bucket="anchor",
                    anchor_id=anchor.anchor_id,
                    site_id=anchor.site_id,
                    moltype=anchor.moltype,
                    chain_key=anchor.chain_key,
                    occupancy_fraction=float(occ) / float(total_points),
                    note=None,
                )
            )
            anchor_to_state[int(anchor_idx)] = int(next_state_idx)
            next_state_idx += 1
        other_occ = int(sum(int(anchor_occ_counts[idx]) for idx in other_pool if idx < anchor_occ_counts.size))
        if other_pool and other_occ > 0:
            other_state_idx = next_state_idx
            states.append(
                StateSpec(
                    state_index=other_state_idx,
                    state_id=f"{role}:OTHER",
                    state_label=f"{role}:OTHER",
                    role=role,
                    bucket="other",
                    anchor_id=None,
                    site_id=None,
                    moltype=None,
                    chain_key=None,
                    occupancy_fraction=float(other_occ) / float(total_points),
                    note="Sparse or low-occupancy donor anchors were lumped into this bucket.",
                )
            )
            for anchor_idx in other_pool:
                anchor_to_state[int(anchor_idx)] = int(other_state_idx)
            next_state_idx += 1

    site_states = np.zeros(arr.shape, dtype=int)
    valid_mask = arr >= 0
    if np.any(valid_mask):
        mapper = np.vectorize(lambda idx: anchor_to_state.get(int(idx), 0), otypes=[int])
        site_states[valid_mask] = mapper(arr[valid_mask])

    occ_counts = np.bincount(site_states.ravel(), minlength=len(states))
    updated_states = []
    for spec in states:
        occ_frac = float(occ_counts[spec.state_index]) / float(total_points)
        updated_states.append(
            StateSpec(
                state_index=spec.state_index,
                state_id=spec.state_id,
                state_label=spec.state_label,
                role=spec.role,
                bucket=spec.bucket,
                anchor_id=spec.anchor_id,
                site_id=spec.site_id,
                moltype=spec.moltype,
                chain_key=spec.chain_key,
                occupancy_fraction=occ_frac,
                note=spec.note,
            )
        )
    return site_states, updated_states, anchor_to_state


def _transition_counts(states: np.ndarray, n_states: int, lag_frames: int) -> np.ndarray:
    arr = np.asarray(states, dtype=int)
    counts = np.zeros((int(n_states), int(n_states)), dtype=float)
    if arr.ndim != 2 or arr.shape[0] <= int(lag_frames) or int(lag_frames) < 1:
        return counts
    prev = arr[:-int(lag_frames), :]
    nxt = arr[int(lag_frames) :, :]
    for center_idx in range(int(prev.shape[1])):
        flat = prev[:, center_idx] * int(n_states) + nxt[:, center_idx]
        counts += np.bincount(flat, minlength=int(n_states) * int(n_states)).reshape(int(n_states), int(n_states))
    return counts


def _normalize_transition_counts(counts: np.ndarray) -> np.ndarray:
    arr = np.asarray(counts, dtype=float)
    row_sums = np.sum(arr, axis=1, keepdims=True)
    out = np.zeros_like(arr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(arr, row_sums, out=out, where=row_sums > 0.0)
    return out


def _occupancy_from_states(states: np.ndarray, n_states: int) -> np.ndarray:
    arr = np.asarray(states, dtype=int)
    counts = np.bincount(arr.ravel(), minlength=int(n_states)).astype(float)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros(int(n_states), dtype=float)
    return counts / total


def _candidate_lag_frames(
    *,
    dt_ps: float,
    n_frames: int,
    lag_ps: str | float | int,
) -> list[int]:
    if n_frames < 3:
        return [1]
    if not isinstance(lag_ps, str):
        frames = max(1, int(round(float(lag_ps) / max(float(dt_ps), 1.0e-12))))
        return [min(frames, max(1, n_frames - 2))]
    if str(lag_ps).strip().lower() != "auto":
        frames = max(1, int(round(float(lag_ps) / max(float(dt_ps), 1.0e-12))))
        return [min(frames, max(1, n_frames - 2))]
    max_lag = max(1, min((n_frames - 1) // 4, 200))
    candidates = []
    for mult in DEFAULT_ROLE_LAG_MULTIPLIERS:
        lag = int(mult)
        if lag <= max_lag:
            candidates.append(lag)
    if 1 not in candidates:
        candidates.insert(0, 1)
    return sorted(set(candidates))


def _rowwise_ck_distance(p_tau: np.ndarray, p_2tau: np.ndarray) -> float:
    if p_tau.size == 0 or p_2tau.size == 0:
        return float("inf")
    pred = np.asarray(p_tau, dtype=float) @ np.asarray(p_tau, dtype=float)
    obs = np.asarray(p_2tau, dtype=float)
    active_rows = (np.sum(obs, axis=1) > 0.0) | (np.sum(pred, axis=1) > 0.0)
    if not np.any(active_rows):
        return 0.0
    diff = np.abs(pred[active_rows, :] - obs[active_rows, :])
    norm = np.maximum(np.sum(np.abs(obs[active_rows, :]), axis=1), 1.0)
    return float(np.mean(np.sum(diff, axis=1) / norm))


def _select_markov_lag(
    states: np.ndarray,
    *,
    dt_ps: float,
    lag_ps: str | float | int,
    n_states: int,
) -> dict[str, Any]:
    arr = np.asarray(states, dtype=int)
    n_frames = int(arr.shape[0]) if arr.ndim == 2 else 0
    candidates = _candidate_lag_frames(dt_ps=dt_ps, n_frames=n_frames, lag_ps=lag_ps)
    candidate_records: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    for lag_frames in candidates:
        counts = _transition_counts(arr, n_states, lag_frames)
        matrix = _normalize_transition_counts(counts)
        ck_distance = None
        if (2 * int(lag_frames)) < n_frames:
            matrix_2tau = _normalize_transition_counts(_transition_counts(arr, n_states, 2 * int(lag_frames)))
            ck_distance = _rowwise_ck_distance(matrix, matrix_2tau)
        record = {
            "lag_frames": int(lag_frames),
            "lag_ps": float(lag_frames * dt_ps),
            "transition_count_total": float(np.sum(counts)),
            "ck_distance": ck_distance,
            "counts": counts,
            "matrix": matrix,
        }
        candidate_records.append(record)
        passes = ck_distance is not None and ck_distance <= 0.15 and float(np.sum(counts)) > 0.0
        if passes:
            best = record
            break
        if best is None:
            best = record
        else:
            cur = float(record["ck_distance"]) if record["ck_distance"] is not None else float("inf")
            old = float(best["ck_distance"]) if best["ck_distance"] is not None else float("inf")
            if cur < old:
                best = record
    best = best or {
        "lag_frames": 1,
        "lag_ps": float(dt_ps),
        "transition_count_total": 0.0,
        "ck_distance": None,
        "counts": np.zeros((n_states, n_states), dtype=float),
        "matrix": np.zeros((n_states, n_states), dtype=float),
    }
    ck = best.get("ck_distance")
    if ck is None:
        confidence = "low"
    elif float(ck) <= 0.08:
        confidence = "high"
    elif float(ck) <= 0.15:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "selected_lag_frames": int(best["lag_frames"]),
        "selected_lag_ps": float(best["lag_ps"]),
        "markov_confidence": confidence,
        "ck_distance": ck,
        "counts": np.asarray(best["counts"], dtype=float),
        "matrix": np.asarray(best["matrix"], dtype=float),
        "candidate_records": [
            {
                "lag_frames": int(rec["lag_frames"]),
                "lag_ps": float(rec["lag_ps"]),
                "transition_count_total": float(rec["transition_count_total"]),
                "ck_distance": rec["ck_distance"],
            }
            for rec in candidate_records
        ],
    }


def _event_type_from_states(prev_state: StateSpec, next_state: StateSpec) -> str:
    if prev_state.anchor_id == next_state.anchor_id:
        return "site_stay"
    if prev_state.bucket in {"none", "other"} or next_state.bucket in {"none", "other"}:
        return "ambiguous_exchange"
    if prev_state.role == "polymer" and next_state.role == "polymer":
        if prev_state.chain_key == next_state.chain_key:
            return "polymer_intrachain_hop"
        return "polymer_interchain_hop"
    if "solvent" in {prev_state.role, next_state.role}:
        return "solvent_assisted_exchange"
    if "anion" in {prev_state.role, next_state.role}:
        return "anion_assisted_exchange"
    if prev_state.site_id != next_state.site_id:
        return "site_exchange"
    return "ambiguous_exchange"


def _event_masks_from_state_catalog(state_specs: Sequence[StateSpec]) -> dict[str, np.ndarray]:
    n_states = len(state_specs)
    masks = {name: np.zeros((n_states, n_states), dtype=bool) for name in EVENT_TYPES}
    for i, prev_state in enumerate(state_specs):
        for j, next_state in enumerate(state_specs):
            event_type = _event_type_from_states(prev_state, next_state)
            masks[event_type][i, j] = True
    return masks


def _observed_event_counts(counts: np.ndarray, state_specs: Sequence[StateSpec]) -> dict[str, int]:
    arr = np.asarray(counts, dtype=float)
    masks = _event_masks_from_state_catalog(state_specs)
    return {event: int(round(float(np.sum(arr[masks[event]])))) for event in EVENT_TYPES}


def _predict_event_counts(
    matrix: np.ndarray,
    *,
    state_specs: Sequence[StateSpec],
    initial_occupancy: np.ndarray,
    n_centers: int,
    lag_ps: float,
) -> dict[str, Any]:
    p = np.asarray(matrix, dtype=float)
    pi = np.asarray(initial_occupancy, dtype=float).copy()
    if pi.ndim != 1:
        pi = np.ravel(pi)
    if pi.size != p.shape[0]:
        pi = np.zeros(p.shape[0], dtype=float)
    masks = _event_masks_from_state_catalog(state_specs)
    max_step = int(max(DEFAULT_PREDICTION_STEP_MULTIPLIERS))
    cumulative = {name: 0.0 for name in EVENT_TYPES}
    rows: list[dict[str, Any]] = []
    for step in range(1, max_step + 1):
        flux = pi[:, None] * p
        per_step = {
            name: float(np.sum(flux[masks[name]])) * float(max(1, n_centers))
            for name in EVENT_TYPES
        }
        for name in EVENT_TYPES:
            cumulative[name] += per_step[name]
        if step in DEFAULT_PREDICTION_STEP_MULTIPLIERS:
            row = {
                "horizon_steps": int(step),
                "horizon_ps": float(step * lag_ps),
            }
            for name in EVENT_TYPES:
                row[name] = float(cumulative[name])
            rows.append(row)
        pi = pi @ p
    return {
        "selected_lag_ps": float(lag_ps),
        "prediction_horizons_steps": [int(row["horizon_steps"]) for row in rows],
        "prediction_horizons_ps": [float(row["horizon_ps"]) for row in rows],
        "predicted_event_counts": rows,
    }


def _write_matrix_csv(labels: Sequence[str], matrix: np.ndarray, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["state"] + list(labels))
        for label, row in zip(labels, np.asarray(matrix, dtype=float).tolist()):
            writer.writerow([label] + [f"{float(x):.12e}" for x in row])
    return out_csv


def _write_event_summary_csv(event_counts: dict[str, int], out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["event_type", "count"])
        for event_type in EVENT_TYPES:
            writer.writerow([event_type, int(event_counts.get(event_type, 0))])
    return out_csv


def _write_predicted_event_counts_csv(rows: Sequence[dict[str, Any]], out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["horizon_steps", "horizon_ps"] + list(EVENT_TYPES)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return out_csv


def _plot_markov_matrix_svg(
    *,
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    out_svg: Path,
    max_states_display: int = MARKOV_ACTIVE_STATE_PLOT_LIMIT,
) -> Optional[Path]:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return None
    active = np.where(np.sum(arr, axis=1) > 0.0)[0]
    if active.size == 0:
        active = np.arange(arr.shape[0], dtype=int)
    if active.size > int(max_states_display):
        return None
    arr = arr[np.ix_(active, active)]
    lab = [str(labels[idx]) for idx in active.tolist()]
    fig, ax = plt.subplots(figsize=(1.1 * max(4, arr.shape[0]), 1.0 * max(4, arr.shape[0])))
    im = ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=max(1.0e-12, float(np.max(arr))))
    ax.set_xticks(np.arange(len(lab), dtype=float))
    ax.set_yticks(np.arange(len(lab), dtype=float))
    ax.set_xticklabels(lab, rotation=45, ha="right")
    ax.set_yticklabels(lab)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Transition probability", rotation=90)
    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg)
    plt.close(fig)
    return out_svg


def _plot_event_flux_summary_svg(
    *,
    predicted_rows: Sequence[dict[str, Any]],
    out_svg: Path,
) -> Optional[Path]:
    rows = list(predicted_rows or [])
    if not rows:
        return None
    horizons = np.asarray([float(row.get("horizon_ps") or 0.0) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for event_type in EVENT_TYPES:
        values = np.asarray([float(row.get(event_type) or 0.0) for row in rows], dtype=float)
        if not np.any(values > 0.0):
            continue
        ax.plot(horizons, values, marker="o", lw=1.6, label=event_type)
    ax.set_xlabel("Prediction horizon (ps)")
    ax.set_ylabel("Expected event count")
    ax.set_title("Markov-predicted migration event counts")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg)
    plt.close(fig)
    return out_svg


def _state_labels(state_specs: Sequence[StateSpec]) -> list[str]:
    return [spec.state_label for spec in state_specs]


def _role_state_specs(role_state_occ: np.ndarray) -> list[StateSpec]:
    occ = np.asarray(role_state_occ, dtype=float)
    specs: list[StateSpec] = []
    for idx, role in enumerate(ROLE_STATE_ORDER):
        specs.append(
            StateSpec(
                state_index=int(idx),
                state_id=role,
                state_label=ROLE_LABELS.get(role, role),
                role=role,
                bucket="role" if role != "none" else "none",
                anchor_id=None,
                site_id=None,
                moltype=None,
                chain_key=None,
                occupancy_fraction=float(occ[idx]) if idx < occ.size else 0.0,
                note=None,
            )
        )
    return specs


def _build_markov_summary(
    *,
    model_name: str,
    state_specs: Sequence[StateSpec],
    occupancy: np.ndarray,
    lag_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "selected_lag_frames": int(lag_info.get("selected_lag_frames") or 1),
        "selected_lag_ps": float(lag_info.get("selected_lag_ps") or 0.0),
        "markov_confidence": str(lag_info.get("markov_confidence") or "low"),
        "chapman_kolmogorov_distance": lag_info.get("ck_distance"),
        "state_count": int(len(state_specs)),
        "active_state_count": int(np.sum(np.asarray(occupancy, dtype=float) > 0.0)),
        "states": [spec.to_record() for spec in state_specs],
        "candidate_lags": list(lag_info.get("candidate_records") or []),
    }


def run_migration_analysis(
    *,
    top: SystemTopology,
    system_dir: Path,
    gro_path: Path,
    xtc_path: Path,
    center_moltype: str,
    rdf_summary: dict[str, Any],
    resolve_moltypes: Callable[[object], Sequence[str]],
    polymer_mols: object | None = None,
    solvent_mols: object | None = None,
    anion_mols: object | None = None,
    cation_mols: object | None = None,
    stride: int | str = "auto",
    rdf_stride: int = 10,
    lag_ps: str | float | int = "auto",
    state_basis: str = "dual",
    residence: bool = True,
    markov: bool = True,
    expert_mode: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    try:
        import mdtraj as md
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"mdtraj is required for migration analysis: {exc}") from exc

    basis = str(state_basis or "dual").strip().lower()
    if basis not in {"dual", "role", "site"}:
        raise ValueError(f"Unsupported migration state_basis: {state_basis!r}")

    analysis_dir = Path(out_dir) if out_dir is not None else (Path(system_dir).parent / "06_analysis" / "migration")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    catalog = build_species_catalog(top, system_dir)
    center_entry = catalog.get(str(center_moltype))
    if center_entry is None:
        raise ValueError(f"Center moltype {center_moltype!r} is not present in the exported topology.")

    role_moltypes = infer_role_moltypes(
        catalog=catalog,
        center_moltype=str(center_moltype),
        resolve_moltypes=resolve_moltypes,
        polymer_mols=polymer_mols,
        solvent_mols=solvent_mols,
        anion_mols=anion_mols,
        cation_mols=cation_mols,
    )
    anchor_bundle = build_anchor_catalog(
        top=top,
        system_dir=system_dir,
        center_moltype=str(center_moltype),
        role_moltypes=role_moltypes,
        rdf_summary=rdf_summary,
        include_h=False,
        include_weak_sites=bool(expert_mode),
    )
    anchors: list[AnchorSpec] = list(anchor_bundle.get("anchors") or [])
    role_available = {
        role: bool(role_moltypes.get(role)) and any(anchor.role == role for anchor in anchors)
        for role in ROLE_ORDER
    }
    coordination_summary = _rdf_coordination_summary(rdf_summary=rdf_summary, anchors=anchors)

    center_groups = _build_center_groups(center_entry)
    center_count = int(len(center_groups))
    if center_count == 0:
        raise RuntimeError(f"Center moltype {center_moltype!r} has no instances to analyze.")

    donor_atom_indices: list[int] = []
    donor_atom_anchor_idx: list[int] = []
    donor_atom_role_idx: list[int] = []
    donor_atom_cutoffs: list[float] = []
    for anchor_idx, anchor in enumerate(anchors):
        donor_atom_indices.extend(int(i) for i in np.asarray(anchor.atom_indices_0, dtype=int).tolist())
        donor_atom_anchor_idx.extend([int(anchor_idx)] * int(np.asarray(anchor.atom_indices_0, dtype=int).size))
        donor_atom_role_idx.extend([ROLE_ORDER.index(anchor.role)] * int(np.asarray(anchor.atom_indices_0, dtype=int).size))
        donor_atom_cutoffs.extend([float(anchor.cutoff_nm)] * int(np.asarray(anchor.atom_indices_0, dtype=int).size))

    donor_atom_indices_arr = np.asarray(donor_atom_indices, dtype=int)
    donor_atom_anchor_idx_arr = np.asarray(donor_atom_anchor_idx, dtype=int)
    donor_atom_role_idx_arr = np.asarray(donor_atom_role_idx, dtype=int)
    donor_atom_cutoffs_arr = np.asarray(donor_atom_cutoffs, dtype=float)

    chunk_size = max(5, int(rdf_stride))
    try:
        probe_iter = iter(md.iterload(str(xtc_path), top=str(gro_path), chunk=chunk_size))
        probe = next(probe_iter)
    except StopIteration as exc:
        raise RuntimeError("No trajectory frames were available for migration analysis.") from exc
    sample_stride = (
        _choose_stride_auto(
            n_frames=int(getattr(probe, "n_frames", 0)),
            center_count=center_count,
            anchor_count=max(1, len(anchors)),
        )
        if isinstance(stride, str) and str(stride).strip().lower() == "auto"
        else int(max(1, int(stride)))
    )

    times_chunks: list[np.ndarray] = []
    center_position_chunks: list[np.ndarray] = []
    box_chunks: list[np.ndarray] = []
    dominant_anchor_chunks: list[np.ndarray] = []
    role_contact_chunks: dict[str, list[np.ndarray]] = {role: [] for role in ROLE_ORDER}

    for raw_trj in md.iterload(str(xtc_path), top=str(gro_path), chunk=chunk_size):
        trj = raw_trj[::sample_stride]
        if int(getattr(trj, "n_frames", 0)) == 0:
            continue
        xyz = np.asarray(trj.xyz, dtype=float)
        box = np.asarray(getattr(trj, "unitcell_lengths", None), dtype=float)
        if box.ndim != 2 or box.shape[0] != xyz.shape[0] or box.shape[1] < 3:
            raise RuntimeError("Migration analysis requires trajectory unit-cell lengths.")

        center_positions = _group_positions_from_xyz(xyz, center_groups)
        dominant_anchor = np.full((int(trj.n_frames), center_count), -1, dtype=int)
        role_contacts = {
            role: np.zeros((int(trj.n_frames), center_count), dtype=bool)
            for role in ROLE_ORDER
        }

        if donor_atom_indices_arr.size:
            donor_xyz = xyz[:, donor_atom_indices_arr, :]
            for frame_idx in range(int(trj.n_frames)):
                centers_f = center_positions[frame_idx, :, :]
                donors_f = donor_xyz[frame_idx, :, :]
                delta = centers_f[:, None, :] - donors_f[None, :, :]
                cell = np.asarray(box[frame_idx, :3], dtype=float)
                delta -= cell[None, None, :] * np.round(delta / np.maximum(cell[None, None, :], 1.0e-12))
                dist = np.linalg.norm(delta, axis=2)
                within = dist <= donor_atom_cutoffs_arr[None, :]
                if np.any(within):
                    masked = np.where(within, dist, np.inf)
                    nearest_atom = np.argmin(masked, axis=1)
                    nearest_val = masked[np.arange(center_count), nearest_atom]
                    active = np.isfinite(nearest_val)
                    dominant_anchor[frame_idx, active] = donor_atom_anchor_idx_arr[nearest_atom[active]]
                    for role_idx, role in enumerate(ROLE_ORDER):
                        atom_mask = donor_atom_role_idx_arr == int(role_idx)
                        if np.any(atom_mask):
                            role_contacts[role][frame_idx, :] = np.any(within[:, atom_mask], axis=1)

        times_chunks.append(np.asarray(trj.time, dtype=float))
        center_position_chunks.append(np.asarray(center_positions, dtype=float))
        box_chunks.append(np.asarray(box[:, :3], dtype=float))
        dominant_anchor_chunks.append(np.asarray(dominant_anchor, dtype=int))
        for role in ROLE_ORDER:
            role_contact_chunks[role].append(np.asarray(role_contacts[role], dtype=bool))

    if not times_chunks:
        raise RuntimeError("No trajectory frames were available for migration analysis.")

    time_ps = np.concatenate(times_chunks, axis=0)
    center_positions_wrapped = np.concatenate(center_position_chunks, axis=0)
    box_lengths_nm = np.concatenate(box_chunks, axis=0)
    dominant_anchor_idx = np.concatenate(dominant_anchor_chunks, axis=0)
    center_positions_nm = _unwrap_position_series(center_positions_wrapped, box_lengths_nm, geometry_mode="3d")
    dt_ps = float(np.median(np.diff(time_ps))) if time_ps.size >= 2 else 1.0

    residence_summary: dict[str, Any] = {}
    role_contact_series: dict[str, np.ndarray] = {}
    for role in ROLE_ORDER:
        role_matrix = (
            np.concatenate(role_contact_chunks[role], axis=0)
            if role_contact_chunks.get(role)
            else np.zeros((time_ps.size, center_count), dtype=bool)
        )
        role_contact_series[role] = (
            np.mean(role_matrix.astype(float), axis=1) if role_matrix.size else np.zeros(time_ps.size, dtype=float)
        )
        if residence:
            residence_summary[role] = summarize_role_residence(
                role=role,
                time_ps=time_ps,
                contact_matrix=role_matrix,
                out_dir=analysis_dir / "residence",
                available=bool(role_available.get(role)),
            )
        else:
            residence_summary[role] = {
                "role": role,
                "role_label": ROLE_LABELS.get(role, role),
                "available": bool(role_available.get(role)),
                "note": "residence analysis disabled",
            }

    role_states = _build_role_state_trajectory(dominant_anchor_idx, anchors)
    role_occ = _occupancy_from_states(role_states, len(ROLE_STATE_ORDER))
    role_specs = _role_state_specs(role_occ)

    site_states, site_specs, anchor_to_state = _build_site_state_trajectory(dominant_anchor_idx, anchors)
    site_occ = _occupancy_from_states(site_states, len(site_specs))

    role_markov_summary: dict[str, Any] = {}
    site_markov_summary: dict[str, Any] = {}
    event_flux_summary: dict[str, Any] = {
        "available": False,
        "reason": "markov analysis disabled",
        "predicted_event_counts": [],
    }
    observed_event_counts = {event: 0 for event in EVENT_TYPES}

    transition_role_csv: Optional[Path] = None
    transition_site_csv: Optional[Path] = None
    markov_role_svg: Optional[Path] = None
    markov_site_svg: Optional[Path] = None
    event_flux_svg: Optional[Path] = None
    predicted_counts_csv: Optional[Path] = None

    if markov:
        role_lag = _select_markov_lag(
            role_states,
            dt_ps=dt_ps,
            lag_ps=lag_ps,
            n_states=len(role_specs),
        )
        role_matrix = np.asarray(role_lag["matrix"], dtype=float)
        role_counts = np.asarray(role_lag["counts"], dtype=float)
        role_markov_summary = _build_markov_summary(
            model_name="role",
            state_specs=role_specs,
            occupancy=role_occ,
            lag_info=role_lag,
        )

        site_lag_ps = role_lag["selected_lag_ps"] if str(lag_ps).strip().lower() == "auto" else lag_ps
        site_lag = _select_markov_lag(
            site_states,
            dt_ps=dt_ps,
            lag_ps=site_lag_ps,
            n_states=len(site_specs),
        )
        site_matrix = np.asarray(site_lag["matrix"], dtype=float)
        site_counts = np.asarray(site_lag["counts"], dtype=float)
        site_markov_summary = _build_markov_summary(
            model_name="site",
            state_specs=site_specs,
            occupancy=site_occ,
            lag_info=site_lag,
        )
        observed_event_counts = _observed_event_counts(site_counts, site_specs)
        event_flux_summary = _predict_event_counts(
            site_matrix,
            state_specs=site_specs,
            initial_occupancy=site_occ,
            n_centers=center_count,
            lag_ps=float(site_lag["selected_lag_ps"]),
        )
        event_flux_summary.update(
            {
                "available": True,
                "event_counts_observed": observed_event_counts,
                "markov_confidence": str(site_lag.get("markov_confidence") or "low"),
            }
        )

        transition_role_csv = _write_matrix_csv(
            _state_labels(role_specs),
            role_matrix,
            analysis_dir / "transition_matrix_role.csv",
        )
        transition_site_csv = _write_matrix_csv(
            _state_labels(site_specs),
            site_matrix,
            analysis_dir / "transition_matrix_site.csv",
        )
        predicted_counts_csv = _write_predicted_event_counts_csv(
            event_flux_summary.get("predicted_event_counts") or [],
            analysis_dir / "predicted_event_counts.csv",
        )
        markov_role_svg = _plot_markov_matrix_svg(
            matrix=role_matrix,
            labels=_state_labels(role_specs),
            title="Role-level Markov transition matrix",
            out_svg=plots_dir / "markov_role_matrix.svg",
            max_states_display=8,
        )
        markov_site_svg = _plot_markov_matrix_svg(
            matrix=site_matrix,
            labels=_state_labels(site_specs),
            title="Site-level Markov transition matrix",
            out_svg=plots_dir / "markov_site_matrix.svg",
        )
        event_flux_svg = _plot_event_flux_summary_svg(
            predicted_rows=event_flux_summary.get("predicted_event_counts") or [],
            out_svg=plots_dir / "event_flux_summary.svg",
        )
    else:
        role_markov_summary = {
            "model": "role",
            "available": False,
            "reason": "markov analysis disabled",
            "states": [spec.to_record() for spec in role_specs],
        }
        site_markov_summary = {
            "model": "site",
            "available": False,
            "reason": "markov analysis disabled",
            "states": [spec.to_record() for spec in site_specs],
        }

    coordination_timeline_svg = _write_coordination_timeline_svg(
        time_ps=time_ps,
        role_contact_series=role_contact_series,
        out_svg=plots_dir / "coordination_timeline.svg",
    )
    residence_curves_svg = _write_residence_curves_svg(
        residence_summary=residence_summary,
        out_svg=plots_dir / "residence_curves.svg",
    )

    coordination_payload = {
        "center_moltype": str(center_moltype),
        "roles": {
            role: {
                "available": bool(role_available.get(role)),
                "role_label": ROLE_LABELS.get(role, role),
                "moltypes": list(role_moltypes.get(role) or []),
                "sites": coordination_summary.get(role) or [],
            }
            for role in ROLE_ORDER
        },
        "anchors": [anchor.to_record() for anchor in anchors] if expert_mode else None,
    }
    state_catalog = {
        "role_states": [spec.to_record() for spec in role_specs],
        "site_states": [spec.to_record() for spec in site_specs],
        "site_state_basis": basis,
        "site_state_lumping": {
            "max_states_per_role": int(MARKOV_MAX_SITE_STATES_PER_ROLE),
            "min_occupancy_fraction": float(MARKOV_MIN_SITE_OCCUPANCY_FRACTION),
            "min_transition_count": int(MARKOV_MIN_SITE_TRANSITIONS),
        },
    }

    (analysis_dir / "coordination_summary.json").write_text(
        json.dumps(coordination_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "residence_summary.json").write_text(
        json.dumps(residence_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "markov_role_summary.json").write_text(
        json.dumps(role_markov_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "markov_site_summary.json").write_text(
        json.dumps(site_markov_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "event_flux_summary.json").write_text(
        json.dumps(event_flux_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "state_catalog.json").write_text(
        json.dumps(state_catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    event_summary_csv = _write_event_summary_csv(
        observed_event_counts,
        analysis_dir / "migration_event_summary.csv",
    )
    migration_events_csv = _write_predicted_event_counts_csv(
        event_flux_summary.get("predicted_event_counts") or [],
        analysis_dir / "migration_events.csv",
    )

    migration_summary = {
        "center_moltype": str(center_moltype),
        "center_count": int(center_count),
        "n_frames": int(time_ps.size),
        "frame_interval_ps": float(dt_ps),
        "effective_stride": int(sample_stride),
        "state_basis": basis,
        "markov_enabled": bool(markov),
        "expert_mode": bool(expert_mode),
        "roles": {
            role: {
                "role_label": ROLE_LABELS.get(role, role),
                "available": bool(role_available.get(role)),
                "moltypes": list(role_moltypes.get(role) or []),
            }
            for role in ROLE_ORDER
        },
        "selected_lag_ps": (
            float(site_markov_summary.get("selected_lag_ps") or role_markov_summary.get("selected_lag_ps") or 0.0)
            if markov
            else None
        ),
        "selected_lag_frames": (
            int(site_markov_summary.get("selected_lag_frames") or role_markov_summary.get("selected_lag_frames") or 0)
            if markov
            else None
        ),
        "markov_confidence": (
            str(site_markov_summary.get("markov_confidence") or role_markov_summary.get("markov_confidence") or "low")
            if markov
            else None
        ),
        "role_state_count": int(len(role_specs)),
        "site_state_count": int(len(site_specs)),
        "event_counts": observed_event_counts,
        "prediction_horizon_ps": list(event_flux_summary.get("prediction_horizons_ps") or []),
        "residence_summary": {
            role: {
                "available": bool((residence_summary.get(role) or {}).get("available")),
                "continuous_residence_time_ps": (residence_summary.get(role) or {}).get("continuous_residence_time_ps"),
                "intermittent_residence_time_ps": (residence_summary.get(role) or {}).get("intermittent_residence_time_ps"),
                "contact_fraction": (residence_summary.get(role) or {}).get("contact_fraction"),
            }
            for role in ROLE_ORDER
        },
        "outputs": {
            "migration_summary_json": str(analysis_dir / "migration_summary.json"),
            "residence_summary_json": str(analysis_dir / "residence_summary.json"),
            "coordination_summary_json": str(analysis_dir / "coordination_summary.json"),
            "markov_role_summary_json": str(analysis_dir / "markov_role_summary.json"),
            "markov_site_summary_json": str(analysis_dir / "markov_site_summary.json"),
            "event_flux_summary_json": str(analysis_dir / "event_flux_summary.json"),
            "state_catalog_json": str(analysis_dir / "state_catalog.json"),
            "transition_matrix_role_csv": str(transition_role_csv) if transition_role_csv is not None else None,
            "transition_matrix_site_csv": str(transition_site_csv) if transition_site_csv is not None else None,
            "predicted_event_counts_csv": str(predicted_counts_csv) if predicted_counts_csv is not None else None,
            "migration_events_csv": str(migration_events_csv),
            "migration_event_summary_csv": str(event_summary_csv),
            "residence_curves_svg": str(residence_curves_svg) if residence_curves_svg is not None else None,
            "markov_role_matrix_svg": str(markov_role_svg) if markov_role_svg is not None else None,
            "markov_site_matrix_svg": str(markov_site_svg) if markov_site_svg is not None else None,
            "event_flux_summary_svg": str(event_flux_svg) if event_flux_svg is not None else None,
            "coordination_timeline_svg": str(coordination_timeline_svg) if coordination_timeline_svg is not None else None,
        },
    }
    if expert_mode:
        raw_anchor_path = analysis_dir / "dominant_anchor_states.npy"
        raw_role_path = analysis_dir / "role_states.npy"
        raw_site_path = analysis_dir / "site_states.npy"
        np.save(raw_anchor_path, dominant_anchor_idx)
        np.save(raw_role_path, role_states)
        np.save(raw_site_path, site_states)
        migration_summary["expert_outputs"] = {
            "dominant_anchor_states_npy": str(raw_anchor_path),
            "role_states_npy": str(raw_role_path),
            "site_states_npy": str(raw_site_path),
        }

    (analysis_dir / "migration_summary.json").write_text(
        json.dumps(migration_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "migration_summary": migration_summary,
        "residence_summary": residence_summary,
        "coordination_summary": coordination_payload,
        "markov_role_summary": role_markov_summary,
        "markov_site_summary": site_markov_summary,
        "event_flux_summary": event_flux_summary,
        "state_catalog": state_catalog,
        "events": [],
        "event_counts": observed_event_counts,
        "outputs": migration_summary.get("outputs") or {},
    }
