"""Adaptive output and analysis policy helpers.

The policy in this module deliberately changes only I/O and post-processing
sampling density. It must not change MD physics such as the timestep, force
field, thermostat, or barostat.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


AUTO = "auto"


def _normalize_profile(value: object, *, default: str = AUTO) -> str:
    token = str(default if value is None else value).strip().lower().replace("-", "_")
    if token in {"", "default"}:
        token = default
    aliases = {
        "screening": "fast",
        "transport": "fast",
        "transport_fast": "fast",
        "full": "full",
        "fast": "fast",
        "efficient": "efficient",
        "minimal": "minimal",
        "auto": "auto",
    }
    if token not in aliases:
        raise ValueError(f"Unsupported performance/profile setting: {value!r}")
    return aliases[token]


def _is_auto(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "auto", "default"}


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if _is_auto(value):
        return None
    return float(value)


def _nice_interval_ps(raw_ps: float) -> float:
    """Round up to a readable ps interval that keeps MDP files human-friendly."""

    raw = float(max(raw_ps, 0.001))
    candidates = [
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        50.0,
        100.0,
        200.0,
        500.0,
        1000.0,
    ]
    for value in candidates:
        if raw <= value:
            return float(value)
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        value = factor * magnitude
        if raw <= value:
            return float(value)
    return float(math.ceil(raw))


@dataclass(frozen=True)
class IOAnalysisPolicy:
    """Resolved output cadence and post-analysis sampling policy.

    Attributes
    ----------
    performance_profile:
        User-requested profile token after normalization. ``"auto"`` means the
        thresholds selected ``policy_level`` from size and duration.
    policy_level:
        Effective cost tier: ``small``, ``fast``, ``efficient``, ``minimal``, or
        ``full``. This tier drives the default output interval and analysis
        coarsening.
    analysis_profile:
        Effective analyzer profile to pass to ``AnalyzeResult``. ``minimal`` is
        intentionally more aggressive than ``transport_fast``.
    traj_ps, energy_ps, log_ps:
        GROMACS output intervals in picoseconds for compressed trajectory,
        energy, and log streams.
    trr_ps, velocity_ps:
        Optional TRR coordinate and velocity intervals. ``None`` disables those
        heavy outputs.
    rdf_frame_stride, rdf_rmax_nm, rdf_bin_nm:
        RDF downsampling and discretization defaults used by benchmark scripts.
    include_polymer_metrics:
        Whether expensive Rg/end-to-end/persistence metrics should run during
        ``get_all_prop``.
    msd_selected_species, msd_default_metric_only:
        Optional species narrowing and metric narrowing hints for MSD analysis.
    estimated_atoms, production_ns, estimated_frames, estimated_atom_frames:
        Cost diagnostics recorded in summaries so users can see why a policy was
        selected.
    max_trajectory_frames, max_atom_frames:
        User or default caps used when deriving automatic intervals.
    reasons:
        Human-readable threshold triggers that explain the chosen tier.
    overrides:
        Explicit user-provided settings that replaced policy defaults.
    """

    performance_profile: str
    policy_level: str
    analysis_profile: str
    traj_ps: float
    energy_ps: float
    log_ps: float
    trr_ps: float | None
    velocity_ps: float | None
    rdf_frame_stride: int
    rdf_rmax_nm: float | None
    rdf_bin_nm: float
    include_polymer_metrics: bool
    msd_selected_species: list[str] | None
    msd_default_metric_only: bool
    estimated_atoms: int | None
    production_ns: float
    estimated_frames: int
    estimated_atom_frames: float | None
    max_trajectory_frames: int
    max_atom_frames: float
    reasons: list[str]
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for workflow summaries."""

        return asdict(self)


def _auto_level(*, prod_ns: float, atom_count: int | None, max_trajectory_frames: int, max_atom_frames: float) -> tuple[str, list[str]]:
    total_ps = max(float(prod_ns), 0.0) * 1000.0
    base_frames = int(math.ceil(total_ps / 2.0)) if total_ps > 0 else 0
    base_atom_frames = float(base_frames * int(atom_count)) if atom_count and atom_count > 0 else None
    reasons: list[str] = []

    level = "small"
    if float(prod_ns) > 50.0:
        level = "fast"
        reasons.append("production_ns>50")
    if atom_count is not None and int(atom_count) > 50_000:
        level = "fast"
        reasons.append("atoms>50000")
    if base_frames > max(25_000, int(0.5 * max_trajectory_frames)):
        level = "fast"
        reasons.append("2ps_frames_large")
    if base_atom_frames is not None and base_atom_frames > 5.0e8:
        level = "fast"
        reasons.append("2ps_atom_frames>5e8")

    if float(prod_ns) >= 200.0:
        level = "efficient"
        reasons.append("production_ns>=200")
    if atom_count is not None and int(atom_count) >= 80_000:
        level = "efficient"
        reasons.append("atoms>=80000")
    if base_frames > max_trajectory_frames:
        level = "efficient"
        reasons.append("2ps_frames_exceed_cap")
    if base_atom_frames is not None and base_atom_frames > min(max_atom_frames, 2.0e9):
        level = "efficient"
        reasons.append("2ps_atom_frames_exceed_cap")

    if float(prod_ns) >= 500.0:
        level = "minimal"
        reasons.append("production_ns>=500")
    if atom_count is not None and int(atom_count) >= 150_000:
        level = "minimal"
        reasons.append("atoms>=150000")
    if base_frames > 4 * max_trajectory_frames:
        level = "minimal"
        reasons.append("2ps_frames_far_exceed_cap")
    if base_atom_frames is not None and base_atom_frames > 2.0e10:
        level = "minimal"
        reasons.append("2ps_atom_frames>2e10")

    if not reasons:
        reasons.append("small_or_short_system")
    return level, reasons


def _defaults_for_level(level: str) -> dict[str, Any]:
    level = str(level)
    if level == "full":
        return {
            "traj_ps": 2.0,
            "energy_ps": 2.0,
            "rdf_frame_stride": 1,
            "rdf_rmax_nm": None,
            "rdf_bin_nm": 0.002,
            "analysis_profile": "full",
            "include_polymer_metrics": True,
            "msd_default_metric_only": False,
        }
    if level == "fast":
        return {
            "traj_ps": 10.0,
            "energy_ps": 10.0,
            "rdf_frame_stride": 5,
            "rdf_rmax_nm": 1.5,
            "rdf_bin_nm": 0.005,
            "analysis_profile": "transport_fast",
            "include_polymer_metrics": False,
            "msd_default_metric_only": True,
        }
    if level == "efficient":
        return {
            "traj_ps": 20.0,
            "energy_ps": 20.0,
            "rdf_frame_stride": 10,
            "rdf_rmax_nm": 1.5,
            "rdf_bin_nm": 0.005,
            "analysis_profile": "transport_fast",
            "include_polymer_metrics": False,
            "msd_default_metric_only": True,
        }
    if level == "minimal":
        return {
            "traj_ps": 50.0,
            "energy_ps": 50.0,
            "rdf_frame_stride": 20,
            "rdf_rmax_nm": 1.2,
            "rdf_bin_nm": 0.01,
            "analysis_profile": "minimal",
            "include_polymer_metrics": False,
            "msd_default_metric_only": True,
        }
    return {
        "traj_ps": 2.0,
        "energy_ps": 2.0,
        "rdf_frame_stride": 5,
        "rdf_rmax_nm": 1.5,
        "rdf_bin_nm": 0.005,
        "analysis_profile": "transport_fast",
        "include_polymer_metrics": False,
        "msd_default_metric_only": True,
    }


def resolve_io_analysis_policy(
    *,
    prod_ns: float,
    atom_count: int | None = None,
    performance_profile: object = AUTO,
    analysis_profile: object = AUTO,
    traj_ps: object = AUTO,
    energy_ps: object = AUTO,
    log_ps: object = AUTO,
    trr_ps: object = None,
    velocity_ps: object = None,
    rdf_frame_stride: object = AUTO,
    rdf_rmax_nm: object = AUTO,
    rdf_bin_nm: object = AUTO,
    include_polymer_metrics: object = AUTO,
    msd_selected_species: list[str] | None = None,
    max_trajectory_frames: int | None = None,
    max_atom_frames: float | None = None,
) -> IOAnalysisPolicy:
    """Resolve production output cadence and analyzer defaults.

    Parameters
    ----------
    prod_ns:
        Production length in nanoseconds. Longer runs can tolerate coarser frame
        spacing for transport statistics, so this is the main duration signal.
    atom_count:
        Estimated total atom count. When available, it lets the policy cap
        atom-frame cost instead of looking only at trajectory-frame count.
    performance_profile:
        ``"auto"`` chooses a tier from thresholds. ``"full"``, ``"fast"``,
        ``"efficient"``, and ``"minimal"`` force a tier.
    analysis_profile:
        ``"auto"`` follows the tier. Explicit ``"full"``,
        ``"transport_fast"``/``"fast"``, or ``"minimal"`` override analyzer
        behavior without changing MD physics.
    traj_ps, energy_ps, log_ps:
        Output intervals in ps. Numeric values override the policy; ``"auto"``
        keeps the tier default.
    trr_ps, velocity_ps:
        Optional TRR and velocity output intervals in ps. ``None`` disables
        these large files unless the caller explicitly opts in.
    rdf_frame_stride, rdf_rmax_nm, rdf_bin_nm:
        RDF analysis defaults. Numeric values override tier defaults.
    include_polymer_metrics:
        ``"auto"`` follows the tier; truthy/falsy values explicitly enable or
        disable expensive polymer conformation metrics.
    msd_selected_species:
        Optional benchmark-level species hint recorded in metadata. The analyzer
        can still resolve its own defaults when this is ``None``.
    max_trajectory_frames, max_atom_frames:
        Soft caps used to increase ``traj_ps`` when the initial tier would still
        write too many frames.

    Returns
    -------
    IOAnalysisPolicy
        A fully resolved, JSON-friendly policy object. Explicit overrides always
        win over automatic tier choices.
    """

    prod_ns_f = float(prod_ns)
    atom_count_i = int(atom_count) if atom_count is not None and int(atom_count) > 0 else None
    max_frames = int(max_trajectory_frames or 50_000)
    max_atom_frames_f = float(max_atom_frames or 5.0e9)

    perf = _normalize_profile(performance_profile, default=AUTO)
    if perf == "auto":
        level, reasons = _auto_level(
            prod_ns=prod_ns_f,
            atom_count=atom_count_i,
            max_trajectory_frames=max_frames,
            max_atom_frames=max_atom_frames_f,
        )
    elif perf == "full":
        level, reasons = "full", ["performance_profile=full"]
    else:
        level, reasons = perf, [f"performance_profile={perf}"]

    defaults = _defaults_for_level(level)
    overrides: dict[str, Any] = {}

    resolved_traj_ps = float(defaults["traj_ps"])
    if not _is_auto(traj_ps):
        maybe = _optional_float(traj_ps)
        if maybe is not None:
            resolved_traj_ps = float(maybe)
            overrides["traj_ps"] = resolved_traj_ps
    resolved_energy_ps = float(defaults["energy_ps"])
    if not _is_auto(energy_ps):
        maybe = _optional_float(energy_ps)
        if maybe is not None:
            resolved_energy_ps = float(maybe)
            overrides["energy_ps"] = resolved_energy_ps

    total_ps = max(prod_ns_f, 0.0) * 1000.0
    if total_ps > 0.0:
        cap_interval = _nice_interval_ps(total_ps / float(max(max_frames, 1)))
        if resolved_traj_ps < cap_interval and _is_auto(traj_ps):
            resolved_traj_ps = cap_interval
            reasons.append("max_trajectory_frames_cap")
        if atom_count_i is not None and atom_count_i > 0:
            atom_cap_interval = _nice_interval_ps((total_ps * float(atom_count_i)) / float(max(max_atom_frames_f, 1.0)))
            if resolved_traj_ps < atom_cap_interval and _is_auto(traj_ps):
                resolved_traj_ps = atom_cap_interval
                reasons.append("max_atom_frames_cap")
    if _is_auto(energy_ps):
        resolved_energy_ps = max(resolved_energy_ps, resolved_traj_ps)

    if _is_auto(log_ps):
        resolved_log_ps = resolved_energy_ps
    else:
        maybe_log = _optional_float(log_ps)
        resolved_log_ps = resolved_energy_ps if maybe_log is None else float(maybe_log)
        overrides["log_ps"] = resolved_log_ps

    resolved_trr_ps = None if trr_ps is None or _is_auto(trr_ps) else float(trr_ps)
    if resolved_trr_ps is not None:
        overrides["trr_ps"] = resolved_trr_ps
    resolved_velocity_ps = None if velocity_ps is None or _is_auto(velocity_ps) else float(velocity_ps)
    if resolved_velocity_ps is not None:
        overrides["velocity_ps"] = resolved_velocity_ps

    requested_analysis = _normalize_profile(analysis_profile, default=AUTO)
    if requested_analysis == "auto":
        resolved_analysis = str(defaults["analysis_profile"])
    elif requested_analysis == "full":
        resolved_analysis = "full"
        overrides["analysis_profile"] = "full"
    elif requested_analysis == "minimal":
        resolved_analysis = "minimal"
        overrides["analysis_profile"] = "minimal"
    else:
        resolved_analysis = "transport_fast"
        overrides["analysis_profile"] = "transport_fast"

    rdf_stride = int(defaults["rdf_frame_stride"])
    if not _is_auto(rdf_frame_stride):
        rdf_stride = max(1, int(rdf_frame_stride))
        overrides["rdf_frame_stride"] = rdf_stride
    rdf_rmax = defaults["rdf_rmax_nm"]
    if not _is_auto(rdf_rmax_nm):
        rdf_rmax = _optional_float(rdf_rmax_nm)
        overrides["rdf_rmax_nm"] = rdf_rmax
    rdf_bin = float(defaults["rdf_bin_nm"])
    if not _is_auto(rdf_bin_nm):
        rdf_bin = float(rdf_bin_nm)
        overrides["rdf_bin_nm"] = rdf_bin

    include_metrics = bool(defaults["include_polymer_metrics"])
    if not _is_auto(include_polymer_metrics):
        include_metrics = str(include_polymer_metrics).strip().lower() in {"1", "true", "yes", "on", "full"}
        overrides["include_polymer_metrics"] = include_metrics

    frames = int(math.ceil(total_ps / resolved_traj_ps)) if total_ps > 0.0 and resolved_traj_ps > 0.0 else 0
    atom_frames = float(frames * atom_count_i) if atom_count_i is not None else None

    return IOAnalysisPolicy(
        performance_profile=perf,
        policy_level=level,
        analysis_profile=resolved_analysis,
        traj_ps=float(resolved_traj_ps),
        energy_ps=float(resolved_energy_ps),
        log_ps=float(resolved_log_ps),
        trr_ps=resolved_trr_ps,
        velocity_ps=resolved_velocity_ps,
        rdf_frame_stride=int(rdf_stride),
        rdf_rmax_nm=None if rdf_rmax is None else float(rdf_rmax),
        rdf_bin_nm=float(rdf_bin),
        include_polymer_metrics=bool(include_metrics),
        msd_selected_species=msd_selected_species,
        msd_default_metric_only=bool(defaults["msd_default_metric_only"]),
        estimated_atoms=atom_count_i,
        production_ns=float(prod_ns_f),
        estimated_frames=int(frames),
        estimated_atom_frames=atom_frames,
        max_trajectory_frames=int(max_frames),
        max_atom_frames=float(max_atom_frames_f),
        reasons=list(dict.fromkeys(reasons)),
        overrides=overrides,
    )
