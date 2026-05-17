"""Generic layer-stack construction for graphite/polymer/electrolyte interfaces.

This is the public interface-building entry point: callers describe an ordered
list of layers and YadonPy turns that description into a physically separated,
GROMACS-ready stacked cell with a manifest and layer-aware index groups.

This first implementation is deliberately conservative.  It focuses on
deterministic geometry planning, MolDB/force-field provenance propagation,
explicit graphite surface charge, vacuum layers, and robust artifacts.  Heavy MD
relaxation still happens through the existing EQ/NVT/NPT helpers so the script
style remains close to Examples 02/05.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from ..core import poly, utils
from ..core.graphite import build_graphite, stack_cell_blocks
from ..gmx.index import _write_ndx
from ..io.gromacs_system import SystemExportResult, export_system_from_cell_meta
from .postprocess import read_ndx_groups
from .prep import make_orthorhombic_pack_cell

_AVOGADRO = 6.02214076e23
_UCM2_TO_E_PER_NM2 = 0.06241509074460765
_LAYER_META_PROP = "_yadonpy_layer_stack_metadata_json"


@dataclass(frozen=True)
class ElectrodeChargeSpec:
    """Fixed-charge model for graphitic electrodes.

    Parameters
    ----------
    mode:
        ``"total_charge"`` distributes ``top_charge_e`` and/or
        ``bottom_charge_e`` directly.  ``"surface_charge_density"`` converts
        ``surface_charge_uC_cm2`` into charge per surface using the final XY
        area.
    top_charge_e, bottom_charge_e:
        Explicit total charge assigned to the top or bottom surface atoms of a
        graphite layer.
    surface_charge_uC_cm2:
        Surface charge density in micro-C/cm^2.  Positive values assign
        ``+sigma`` to the top surface and ``-sigma`` to the bottom surface when
        explicit top/bottom charges are not provided.
    top_surface_charge_uC_cm2, bottom_surface_charge_uC_cm2:
        Side-specific surface charge densities.  These are preferred for
        graphite-electrolyte-graphite stacks because only the interior surfaces
        should be charged; the outer slab faces can remain neutral.
    """

    mode: Literal["total_charge", "surface_charge_density"] = "total_charge"
    top_charge_e: float | None = None
    bottom_charge_e: float | None = None
    surface_charge_uC_cm2: float | None = None
    top_surface_charge_uC_cm2: float | None = None
    bottom_surface_charge_uC_cm2: float | None = None


@dataclass(frozen=True)
class GraphiteLayerSpec:
    """Graphite layer in a generic stack.

    ``orientation="basal"`` builds a basal plane surface, while ``"edge"``
    rotates the graphitic slab so exposed edge chemistry can face the adjacent
    layer.  ``periodic_xy=None`` means basal graphite is periodic in XY and edge
    graphite is a finite, capped slab.  ``edge_cap`` supports ``H``, ``OH``,
    ``O``/carbonyl, ``CHO``, ``COOH``, random mixtures, and the legacy
    ``periodic`` basal-plane mode.
    """

    name: str = "GRAPHITE"
    nx: int = 6
    ny: int = 5
    n_layers: int = 3
    orientation: Literal["basal", "edge"] = "basal"
    edge_cap: str | Sequence[str] = "H"
    periodic_xy: bool | None = None
    lateral_padding_nm: float | None = None
    random_cap_probs: Mapping[str, float] | None = None
    electrode_charge: ElectrodeChargeSpec | None = None
    ff_name: str = "gaff2_mod"
    charge_method: str = "RESP"
    top_padding_ang: float = 0.5


@dataclass(frozen=True)
class MolecularLayerSpec:
    """Molecular/polymer layer packed under the stack master XY footprint.

    Parameters
    ----------
    species:
        RDKit molecules or MolDB-ready species objects.  The molecules should
        already carry force-field and charge information, as in Examples 05/07.
    counts:
        Molecule counts aligned to ``species``.
    density_target_g_cm3:
        Initial packing-density target.  For the default fixed-XY layer-stack
        path this is used to expand the molecular layer in z when the requested
        molecules would otherwise be too dense to insert under the graphite
        footprint.  It is not a final confined-layer density constraint.
    charge_scale:
        Optional global or per-species charge scaling, propagated to the system
        export metadata.
    layer_kind:
        Semantic label used for index groups and analysis routing.
    """

    name: str
    species: Sequence[Chem.Mol]
    counts: Sequence[int]
    thickness_nm: float
    density_target_g_cm3: float | None = None
    layer_kind: Literal["electrolyte", "polymer", "cmcna", "generic"] = "generic"
    charge_scale: float | Sequence[float] | Mapping[str, float] | None = None
    polyelectrolyte_mode: bool | None = None
    retry: int = 20
    retry_step: int = 1000
    threshold_ang: float = 2.0
    large_system_mode: str = "auto"


@dataclass(frozen=True)
class VacuumLayerSpec:
    """Explicit vacuum spacer layer.

    Vacuum contributes only z-thickness.  By default the generated system still
    uses ``pbc=xyz`` with an explicit empty region; walls are not inserted unless
    a later MD preset explicitly requests them.
    """

    thickness_nm: float
    name: str = "VACUUM"


LayerSpec = GraphiteLayerSpec | MolecularLayerSpec | VacuumLayerSpec


@dataclass(frozen=True)
class LayerStackRelaxationSpec:
    """Bookkeeping for post-build relaxation defaults.

    The generic builder itself stays build/export focused.  Scripts can pass
    these values to the existing EQ/NVT/NPT helpers or future layer-stack MD
    wrappers without changing the stack manifest schema.
    """

    temperature_K: float = 318.15
    pressure_bar: float = 1.0
    early_dt_ps: float = 0.001
    early_constraints: str = "none"
    final_dt_ps: float = 0.002
    final_constraints: str = "h-bonds"
    freeze_layers: tuple[str, ...] = ()
    run_relaxation: bool = False
    sample_ns: float = 0.0


@dataclass(frozen=True)
class ZCompressionAnnealSpec:
    """Controlled fixed-XY z-compression annealing for closed layer stacks."""

    enabled: bool | Literal["auto"] = "auto"
    cycles: int = 6
    normal_temp_K: float | None = None
    normal_pressure_bar: float | None = None
    tmax_K: float = 380.0
    pmax_bar: float = 2000.0
    max_z_shrink_per_cycle: float = 0.04
    target_z_nm: float | Literal["auto"] = "auto"
    target_z_tolerance: float = 0.03
    min_interlayer_gap_nm: float = 0.20
    hot_nvt_ns: float = 0.01
    compression_npt_ns: float = 0.05
    cool_nvt_ns: float = 0.02
    pressure_schedule: Literal["linear"] = "linear"
    temperature_schedule: Literal["linear"] = "linear"
    geometry_compression: Literal["auto", "inter_electrode", "global"] = "auto"
    rollback_on_failure: bool = True


@dataclass(frozen=True)
class LayerStackSpec:
    """Ordered stack recipe.

    ``order="bottom_to_top"`` is the natural z-order used in the output cell.
    ``"top_to_bottom"`` lets users write the experimental stack order directly
    and have the builder reverse it internally.
    """

    layers: Sequence[LayerSpec]
    order: Literal["bottom_to_top", "top_to_bottom"] = "bottom_to_top"
    pbc_mode: Literal["auto", "xyz", "xy"] = "auto"
    name: str = "layer_stack"
    default_gap_nm: float = 0.35
    bottom_padding_nm: float = 0.0
    top_padding_nm: float = 0.0
    auto_expand_graphite: bool = True
    molecular_packing_expand: Literal["z", "xy"] = "z"


@dataclass(frozen=True)
class LayerStackResult:
    """Result returned by :func:`build_layer_stack`."""

    work_dir: Path
    stack_spec: LayerStackSpec
    system_export: SystemExportResult
    system_gro: Path
    system_top: Path
    system_ndx: Path
    manifest_path: Path
    stacked_cell: Chem.Mol
    layer_reports: tuple[dict[str, Any], ...]
    acceptance: dict[str, Any]
    box_nm: tuple[float, float, float]

    @property
    def relaxed_gro(self) -> Path:
        """Compatibility alias used by follow-up MD helpers."""

        return self.system_gro

    @property
    def stack_export(self) -> SystemExportResult:
        """Alias for scripts that pass the export object to follow-up helpers."""

        return self.system_export


@dataclass(frozen=True)
class LayerStackNvtResult:
    """Result from :func:`run_layer_stack_nvt`.

    The helper is intentionally thin: it starts from the exported layer-stack
    cell, runs one NVT segment with the existing GROMACS preset, records the
    actual artifact paths, and optionally launches interface analysis.
    """

    work_dir: Path
    final_gro: Path | None
    trajectory: Path | None
    xtc: Path | None
    trr: Path | None
    summary_path: Path
    analysis_summary: Path | None

    def analyze(self):
        """Return an :class:`AnalyzeResult` rooted at this NVT follow-up run."""

        from ..sim.analyzer import AnalyzeResult

        return AnalyzeResult.from_work_dir(self.work_dir)


@dataclass(frozen=True)
class LayerStackRelaxationResult:
    """Result from :func:`run_layer_stack_relaxation`."""

    work_dir: Path
    final_gro: Path | None
    trajectory: Path | None
    xtc: Path | None
    trr: Path | None
    summary_path: Path
    analysis_summary: Path | None
    diagnostics: dict[str, Any]

    def analyze(self):
        """Return an :class:`AnalyzeResult` rooted at this relaxation run."""

        from ..sim.analyzer import AnalyzeResult

        return AnalyzeResult.from_work_dir(self.work_dir)


def _safe_name(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip().upper())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "LAYER"


def _mol_name(mol: Chem.Mol, fallback: str) -> str:
    try:
        return str(utils.get_name(mol, default=fallback) or fallback)
    except Exception:
        return str(fallback)


def _coords(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(mol.GetConformer(0).GetPositions(), dtype=float)


def _box_lengths_ang(mol: Chem.Mol) -> tuple[float, float, float] | None:
    cell = getattr(mol, "cell", None)
    if cell is None:
        return None
    try:
        return (
            abs(float(cell.xhi) - float(cell.xlo)),
            abs(float(cell.yhi) - float(cell.ylo)),
            abs(float(cell.zhi) - float(cell.zlo)),
        )
    except Exception:
        return None


def _read_cell_meta(cell: Chem.Mol) -> dict[str, Any]:
    try:
        return json.loads(cell.GetProp("_yadonpy_cell_meta"))
    except Exception:
        return {"species": []}


def _layer_order(layers: Sequence[LayerSpec], order: str) -> list[LayerSpec]:
    ordered = list(layers)
    token = str(order).strip().lower()
    if token == "bottom_to_top":
        return ordered
    if token == "top_to_bottom":
        return list(reversed(ordered))
    raise ValueError("LayerStackSpec.order must be 'bottom_to_top' or 'top_to_bottom'.")


def _molecular_mass_g(spec: MolecularLayerSpec) -> float:
    mass_da = 0.0
    for mol, count in zip(spec.species, spec.counts):
        try:
            mw = float(Descriptors.MolWt(mol))
        except Exception:
            mw = float(sum(atom.GetMass() for atom in mol.GetAtoms()))
        mass_da += mw * float(count)
    return mass_da / _AVOGADRO


def _required_area_nm2(layer: LayerSpec) -> float:
    if not isinstance(layer, MolecularLayerSpec):
        return 0.0
    density = layer.density_target_g_cm3
    if density is None or float(density) <= 0.0:
        return 0.0
    thickness_nm = max(float(layer.thickness_nm), 1.0e-6)
    volume_cm3 = _molecular_mass_g(layer) / float(density)
    area_cm2 = volume_cm3 / (thickness_nm * 1.0e-7)
    return max(float(area_cm2) * 1.0e14, 0.0)


def _required_thickness_nm(layer: MolecularLayerSpec, area_nm2: float) -> float:
    density = layer.density_target_g_cm3
    if density is None or float(density) <= 0.0 or float(area_nm2) <= 0.0:
        return float(layer.thickness_nm)
    volume_cm3 = _molecular_mass_g(layer) / float(density)
    area_cm2 = float(area_nm2) * 1.0e-14
    thickness_cm = volume_cm3 / max(area_cm2, 1.0e-30)
    return max(float(thickness_cm) * 1.0e7, float(layer.thickness_nm))


def _graphite_periodic_xy(spec: GraphiteLayerSpec) -> bool:
    """Resolve the graphite XY periodicity default.

    Basal-plane graphite is normally an infinite electrode model in XY, whereas
    edge-plane graphite exposes finite edges and therefore needs explicit caps.
    """

    if spec.periodic_xy is not None:
        return bool(spec.periodic_xy)
    return str(spec.orientation).strip().lower() == "basal"


def _effective_graphite_edge_cap(spec: GraphiteLayerSpec) -> str | Sequence[str]:
    periodic_xy = _graphite_periodic_xy(spec)
    if periodic_xy and str(spec.orientation).strip().lower() != "basal":
        raise ValueError("periodic_xy=True is only supported for basal graphite layers.")
    if periodic_xy:
        return "periodic"
    if str(spec.edge_cap).strip().lower() == "periodic":
        raise ValueError("edge_cap='periodic' requires basal graphite with periodic_xy=True.")
    return spec.edge_cap


def _graphite_lateral_margin_ang(spec: GraphiteLayerSpec) -> float:
    if _graphite_periodic_xy(spec):
        return 0.0
    if spec.lateral_padding_nm is not None:
        return 10.0 * max(float(spec.lateral_padding_nm), 0.0)
    return 10.0


def _graphite_xy_nm(spec: GraphiteLayerSpec) -> tuple[float, float, Any]:
    result = build_graphite(
        nx=int(spec.nx),
        ny=int(spec.ny),
        n_layers=int(spec.n_layers),
        orientation=spec.orientation,
        edge_cap=_effective_graphite_edge_cap(spec),
        random_cap_probs=spec.random_cap_probs,
        ff_name=spec.ff_name,
        charge=None,
        name=spec.name,
        lateral_margin_ang=_graphite_lateral_margin_ang(spec),
        bottom_margin_ang=0.0,
        top_padding_ang=float(spec.top_padding_ang),
    )
    return (float(result.box_nm[0]), float(result.box_nm[1]), result)


def _plan_master_xy(
    layers: Sequence[LayerSpec],
    *,
    auto_expand_graphite: bool,
    molecular_packing_expand: str = "z",
) -> tuple[float, float, dict[str, Any]]:
    expand_axis = str(molecular_packing_expand or "z").strip().lower()
    if expand_axis not in {"z", "xy"}:
        raise ValueError("LayerStackSpec.molecular_packing_expand must be 'z' or 'xy'.")
    graphite_entries: list[tuple[GraphiteLayerSpec, float, float, Any]] = []
    required_area = 0.0
    for layer in layers:
        required_area = max(required_area, _required_area_nm2(layer))
        if isinstance(layer, GraphiteLayerSpec):
            gx, gy, graphite_result = _graphite_xy_nm(layer)
            graphite_entries.append((layer, gx, gy, graphite_result))

    if graphite_entries:
        planned_dims = {str(v[0].name): {"nx": int(v[0].nx), "ny": int(v[0].ny)} for v in graphite_entries}
        master_x = max(v[1] for v in graphite_entries)
        master_y = max(v[2] for v in graphite_entries)
        reason = "largest_graphite_footprint"
        area = master_x * master_y
        if expand_axis == "xy" and required_area > area + 1.0e-9:
            if not auto_expand_graphite:
                raise ValueError(
                    f"Molecular layers require {required_area:.3f} nm^2 but graphite footprint is {area:.3f} nm^2."
                )
            scale = math.sqrt(required_area / max(area, 1.0e-12))
            rebuilt_xy: list[tuple[float, float]] = []
            for graphite_spec, gx, gy, _built in graphite_entries:
                nx = max(int(graphite_spec.nx), int(math.ceil(int(graphite_spec.nx) * scale)))
                ny = max(int(graphite_spec.ny), int(math.ceil(int(graphite_spec.ny) * scale)))
                planned_dims[str(graphite_spec.name)] = {"nx": int(nx), "ny": int(ny)}
                rebuilt = build_graphite(
                    nx=nx,
                    ny=ny,
                    n_layers=int(graphite_spec.n_layers),
                    orientation=graphite_spec.orientation,
                    edge_cap=_effective_graphite_edge_cap(graphite_spec),
                    random_cap_probs=graphite_spec.random_cap_probs,
                    ff_name=graphite_spec.ff_name,
                    charge=None,
                    name=graphite_spec.name,
                    lateral_margin_ang=_graphite_lateral_margin_ang(graphite_spec),
                    bottom_margin_ang=0.0,
                    top_padding_ang=float(graphite_spec.top_padding_ang),
                )
                rebuilt_xy.append((float(rebuilt.box_nm[0]), float(rebuilt.box_nm[1])))
            master_x = max(v[0] for v in rebuilt_xy)
            master_y = max(v[1] for v in rebuilt_xy)
            reason = "expanded_from_molecular_density_targets"
        elif expand_axis == "z" and required_area > area + 1.0e-9:
            reason = "fixed_graphite_xy_z_expanded_molecular_layers"
    else:
        planned_dims = {}
        side = math.sqrt(max(required_area, 1.0))
        master_x = side
        master_y = side
        reason = "molecular_density_targets"

    return (
        float(master_x),
        float(master_y),
        {
            "required_area_nm2": float(required_area),
            "master_xy_nm": [float(master_x), float(master_y)],
            "reason": reason,
            "has_graphite": bool(graphite_entries),
            "graphite_dimensions": planned_dims,
            "molecular_packing_expand": expand_axis,
        },
    )


def _charge_atom(atom: Chem.Atom, delta: float) -> None:
    current = 0.0
    for key in ("AtomicCharge", "RESP", "RESP2", "ESP"):
        try:
            if atom.HasProp(key):
                current = float(atom.GetDoubleProp(key))
                break
        except Exception:
            continue
    new_q = current + float(delta)
    for key in ("AtomicCharge", "RESP"):
        try:
            atom.SetDoubleProp(key, float(new_q))
        except Exception:
            pass


def _surface_atom_indices_by_z(mol: Chem.Mol, *, side: Literal["bottom", "top"]) -> list[int]:
    z = _coords(mol)[:, 2]
    if z.size == 0:
        return []
    target = float(np.min(z) if side == "bottom" else np.max(z))
    tol = max(0.08, 0.02 * max(float(np.ptp(z)), 1.0))
    if side == "bottom":
        return [int(i) for i, zi in enumerate(z) if float(zi) <= target + tol]
    return [int(i) for i, zi in enumerate(z) if float(zi) >= target - tol]


def _resolve_electrode_charges(
    spec: ElectrodeChargeSpec | None,
    *,
    area_nm2: float,
) -> tuple[float, float]:
    if spec is None:
        return (0.0, 0.0)
    mode = str(spec.mode).strip().lower()
    if mode == "total_charge":
        return (float(spec.bottom_charge_e or 0.0), float(spec.top_charge_e or 0.0))
    if mode == "surface_charge_density":
        area = float(area_nm2)
        legacy_sigma = None
        if spec.surface_charge_uC_cm2 is not None:
            legacy_sigma = float(spec.surface_charge_uC_cm2) * _UCM2_TO_E_PER_NM2
        if spec.bottom_charge_e is not None:
            bottom = float(spec.bottom_charge_e)
        elif spec.bottom_surface_charge_uC_cm2 is not None:
            bottom = float(spec.bottom_surface_charge_uC_cm2) * _UCM2_TO_E_PER_NM2 * area
        elif legacy_sigma is not None:
            bottom = -legacy_sigma * area
        else:
            bottom = 0.0
        if spec.top_charge_e is not None:
            top = float(spec.top_charge_e)
        elif spec.top_surface_charge_uC_cm2 is not None:
            top = float(spec.top_surface_charge_uC_cm2) * _UCM2_TO_E_PER_NM2 * area
        elif legacy_sigma is not None:
            top = legacy_sigma * area
        else:
            top = 0.0
        return (bottom, top)
    raise ValueError("ElectrodeChargeSpec.mode must be 'total_charge' or 'surface_charge_density'.")


def _split_graphite_species_entries(cell: Chem.Mol, *, layer_name: str, ff_name: str, charge_method: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        frags = list(Chem.GetMolFrags(cell, asMols=True, sanitizeFrags=False))
    except Exception:
        frags = []
    for idx, frag in enumerate(frags):
        try:
            smiles = Chem.MolToSmiles(frag, canonical=True)
        except Exception:
            smiles = ""
        entries.append(
            {
                "smiles": smiles,
                "n": 1,
                "natoms": int(frag.GetNumAtoms()),
                "name": f"{_safe_name(layer_name)}_{idx:02d}",
                "ff_name": str(ff_name),
                "charge_method": str(charge_method),
                "prefer_db": False,
                "require_db": False,
                "require_ready": False,
                "layer_name": str(layer_name),
                "layer_kind": "graphite",
            }
        )
    return entries


def _apply_graphite_surface_charge(
    cell: Chem.Mol,
    *,
    layer_name: str,
    charge_spec: ElectrodeChargeSpec | None,
    area_nm2: float,
) -> dict[str, Any]:
    bottom_q, top_q = _resolve_electrode_charges(charge_spec, area_nm2=area_nm2)
    report = {
        "mode": (charge_spec.mode if charge_spec is not None else None),
        "bottom_charge_e": float(bottom_q),
        "top_charge_e": float(top_q),
        "surface_atom_counts": {"bottom": 0, "top": 0},
        "applied": bool(abs(bottom_q) > 1.0e-12 or abs(top_q) > 1.0e-12),
    }
    if not report["applied"]:
        return report
    bottom_atoms = _surface_atom_indices_by_z(cell, side="bottom")
    top_atoms = _surface_atom_indices_by_z(cell, side="top")
    report["surface_atom_counts"] = {"bottom": len(bottom_atoms), "top": len(top_atoms)}
    if bottom_q and not bottom_atoms:
        raise RuntimeError(f"Could not identify bottom surface atoms for graphite layer {layer_name!r}.")
    if top_q and not top_atoms:
        raise RuntimeError(f"Could not identify top surface atoms for graphite layer {layer_name!r}.")
    for idx in bottom_atoms:
        _charge_atom(cell.GetAtomWithIdx(idx), bottom_q / float(len(bottom_atoms)))
    for idx in top_atoms:
        _charge_atom(cell.GetAtomWithIdx(idx), top_q / float(len(top_atoms)))
    return report


def _prepare_graphite_layer(
    layer: GraphiteLayerSpec,
    *,
    master_xy_nm: tuple[float, float],
    planned_dimensions: Mapping[str, Any] | None,
    work_dir: Path,
) -> tuple[Chem.Mol, list[dict[str, Any]], dict[str, Any]]:
    dims = dict((planned_dimensions or {}).get(str(layer.name), {}) or {})
    nx = int(dims.get("nx", layer.nx))
    ny = int(dims.get("ny", layer.ny))
    built = build_graphite(
        nx=nx,
        ny=ny,
        n_layers=int(layer.n_layers),
        orientation=layer.orientation,
        edge_cap=_effective_graphite_edge_cap(layer),
        random_cap_probs=layer.random_cap_probs,
        ff_name=layer.ff_name,
        charge=None,
        name=layer.name,
        lateral_margin_ang=_graphite_lateral_margin_ang(layer),
        bottom_margin_ang=0.0,
        top_padding_ang=float(layer.top_padding_ang),
    )
    charge_report = _apply_graphite_surface_charge(
        built.cell,
        layer_name=layer.name,
        charge_spec=layer.electrode_charge,
        area_nm2=float(master_xy_nm[0] * master_xy_nm[1]),
    )
    species_entries = _split_graphite_species_entries(
        built.cell,
        layer_name=layer.name,
        ff_name=layer.ff_name,
        charge_method=layer.charge_method,
    )
    report = {
        "name": layer.name,
        "kind": "graphite",
        "nx": int(nx),
        "ny": int(ny),
        "n_layers": int(layer.n_layers),
        "orientation": layer.orientation,
        "edge_cap": layer.edge_cap,
        "periodic_xy": bool(_graphite_periodic_xy(layer)),
        "lateral_padding_nm": (
            0.0
            if _graphite_periodic_xy(layer)
            else (float(layer.lateral_padding_nm) if layer.lateral_padding_nm is not None else 1.0)
        ),
        "effective_edge_cap": _effective_graphite_edge_cap(layer),
        "edge_cap_summary": dict(built.edge_cap_summary),
        "box_nm": [float(v) for v in built.box_nm],
        "electrode_charge": charge_report,
        "work_dir": str(work_dir),
    }
    return built.cell, species_entries, report


def _prepare_molecular_layer(
    layer: MolecularLayerSpec,
    *,
    master_xy_nm: tuple[float, float],
    molecular_packing_expand: str,
    work_dir: Path,
    restart: bool | None,
) -> tuple[Chem.Mol, list[dict[str, Any]], dict[str, Any]]:
    if len(layer.species) != len(layer.counts):
        raise ValueError(f"MolecularLayerSpec {layer.name!r}: species and counts must have the same length.")
    if float(layer.thickness_nm) <= 0.0:
        raise ValueError(f"MolecularLayerSpec {layer.name!r}: thickness_nm must be positive.")
    layer_work_dir = work_dir / _safe_name(layer.name).lower()
    layer_work_dir.mkdir(parents=True, exist_ok=True)
    target_thickness = float(layer.thickness_nm)
    expand_axis = str(molecular_packing_expand or "z").strip().lower()
    packing_thickness = (
        _required_thickness_nm(layer, float(master_xy_nm[0]) * float(master_xy_nm[1]))
        if expand_axis == "z"
        else target_thickness
    )
    pack_cell = make_orthorhombic_pack_cell((float(master_xy_nm[0]), float(master_xy_nm[1]), float(packing_thickness)))
    pe_mode = (
        bool(layer.polyelectrolyte_mode)
        if layer.polyelectrolyte_mode is not None
        else str(layer.layer_kind).lower() in {"cmcna", "polyelectrolyte"}
    )
    cell = poly.amorphous_cell(
        list(layer.species),
        list(int(v) for v in layer.counts),
        cell=pack_cell,
        density=None,
        retry=int(layer.retry),
        retry_step=int(layer.retry_step),
        threshold=float(layer.threshold_ang),
        charge_scale=layer.charge_scale,
        polyelectrolyte_mode=pe_mode,
        large_system_mode=layer.large_system_mode,
        work_dir=layer_work_dir,
        restart=restart,
    )
    meta = _read_cell_meta(cell)
    species_entries = []
    for entry in meta.get("species") or []:
        copied = dict(entry)
        copied["layer_name"] = layer.name
        copied["layer_kind"] = layer.layer_kind
        copied["polyelectrolyte_mode"] = bool(pe_mode or copied.get("polyelectrolyte_mode", False))
        species_entries.append(copied)
    report = {
        "name": layer.name,
        "kind": layer.layer_kind,
        "counts": [int(v) for v in layer.counts],
        "species_names": [_mol_name(m, f"M{i+1}") for i, m in enumerate(layer.species)],
        "thickness_nm": float(target_thickness),
        "packing_thickness_nm": float(packing_thickness),
        "packing_z_expansion_factor": (
            float(packing_thickness) / float(target_thickness) if float(target_thickness) > 0.0 else None
        ),
        "molecular_packing_expand": expand_axis,
        "density_target_g_cm3": (float(layer.density_target_g_cm3) if layer.density_target_g_cm3 is not None else None),
        "density_target_role": (
            "initial_z_packing_density_at_fixed_xy"
            if expand_axis == "z"
            else "initial_xy_footprint_planning_density"
        ),
        "charge_scale": layer.charge_scale,
        "polyelectrolyte_mode": bool(pe_mode),
        "work_dir": str(layer_work_dir),
    }
    return cell, species_entries, report


def _z_extent_nm(block: Chem.Mol) -> tuple[float, float]:
    z = _coords(block)[:, 2] * 0.1
    return float(np.min(z)), float(np.max(z))


def _stack_non_vacuum_layers(
    prepared: Sequence[tuple[int, LayerSpec, Chem.Mol, list[dict[str, Any]], dict[str, Any]]],
    layers: Sequence[LayerSpec],
    *,
    master_xy_nm: tuple[float, float],
    default_gap_nm: float,
    bottom_padding_nm: float,
    top_padding_nm: float,
    periodic_closing_gap_nm: float = 0.0,
) -> tuple[Chem.Mol, tuple[float, float, float], list[dict[str, Any]], list[dict[str, Any]]]:
    nonvac_by_index = {idx: (layer, block, species, report) for idx, layer, block, species, report in prepared}
    blocks: list[Chem.Mol] = []
    block_input_indices: list[int] = []
    gaps_ang: list[float] = []
    pending_gap_nm = float(bottom_padding_nm)
    leading_padding_nm = 0.0
    trailing_padding_nm = float(top_padding_nm)
    previous_nonvac_seen = False

    for idx, layer in enumerate(layers):
        if isinstance(layer, VacuumLayerSpec):
            if previous_nonvac_seen:
                pending_gap_nm += float(layer.thickness_nm)
            else:
                leading_padding_nm += float(layer.thickness_nm)
            continue
        if idx not in nonvac_by_index:
            continue
        if blocks:
            gaps_ang.append(10.0 * max(pending_gap_nm, float(default_gap_nm)))
            pending_gap_nm = 0.0
        else:
            leading_padding_nm += pending_gap_nm
            pending_gap_nm = 0.0
        _layer, block, _species, _report = nonvac_by_index[idx]
        blocks.append(block)
        block_input_indices.append(idx)
        previous_nonvac_seen = True
    trailing_padding_nm += pending_gap_nm
    closing_gap_target_nm = max(0.0, float(periodic_closing_gap_nm))
    if closing_gap_target_nm > 0.0:
        # With pbc=xyz the top of the stack neighbors the bottom image.  Without
        # an explicit closing spacer, graphite|electrolyte|...|graphite stacks
        # can start with the two outer surfaces effectively on top of each
        # other, yielding enormous LJ energy at step 0.  Treat the periodic
        # boundary like another interface unless the user already supplied
        # enough top/bottom vacuum or padding.
        boundary_padding_nm = float(leading_padding_nm) + float(trailing_padding_nm)
        trailing_padding_nm += max(0.0, closing_gap_target_nm - boundary_padding_nm)

    if not blocks:
        raise ValueError("Layer stack contains no non-vacuum layers.")

    stacked = stack_cell_blocks(
        blocks,
        z_gaps_ang=gaps_ang,
        lateral_margin_ang=0.0,
        bottom_margin_ang=10.0 * float(leading_padding_nm),
        top_padding_ang=10.0 * float(trailing_padding_nm),
        fixed_xy_ang=(10.0 * float(master_xy_nm[0]), 10.0 * float(master_xy_nm[1])),
    )

    # Reconstruct z intervals by replaying stack_cell_blocks' z cursor logic.
    intervals: list[dict[str, Any]] = []
    z_cursor_ang = 10.0 * float(leading_padding_nm)
    for out_idx, (input_idx, block) in enumerate(zip(block_input_indices, blocks)):
        coord = _coords(block)
        mins = np.min(coord, axis=0)
        maxs = np.max(coord, axis=0)
        height_ang = float(maxs[2] - mins[2])
        lo_nm = 0.1 * z_cursor_ang
        hi_nm = 0.1 * (z_cursor_ang + height_ang)
        layer = layers[input_idx]
        intervals.append(
            {
                "layer_index": int(input_idx),
                "stack_nonvacuum_index": int(out_idx),
                "name": getattr(layer, "name", f"LAYER_{input_idx:02d}"),
                "kind": ("vacuum" if isinstance(layer, VacuumLayerSpec) else ("graphite" if isinstance(layer, GraphiteLayerSpec) else layer.layer_kind)),
                "z_lo_nm": float(lo_nm),
                "z_hi_nm": float(hi_nm),
                "z_mid_nm": float(0.5 * (lo_nm + hi_nm)),
            }
        )
        z_cursor_ang += height_ang
        if out_idx < len(gaps_ang):
            z_cursor_ang += gaps_ang[out_idx]

    gap_reports = [
        {"after_nonvacuum_index": i, "gap_nm": float(gaps_ang[i] * 0.1)} for i in range(len(gaps_ang))
    ]
    if closing_gap_target_nm > 0.0:
        gap_reports.append(
            {
                "after_nonvacuum_index": len(blocks) - 1,
                "before_nonvacuum_index": 0,
                "gap_nm": float(leading_padding_nm + trailing_padding_nm),
                "pbc_closing": True,
            }
        )
    return stacked.cell, tuple(float(v) for v in stacked.box_nm), intervals, gap_reports


def _write_combined_cell_meta(
    cell: Chem.Mol,
    *,
    species_entries: Sequence[dict[str, Any]],
    stack_name: str,
    master_xy_nm: tuple[float, float],
    box_nm: tuple[float, float, float],
) -> None:
    q_raw = 0.0
    q_scaled = 0.0
    for entry in species_entries:
        # Keep the summary conservative: exact charge is audited by the exporter
        # after artifacts are written, while this manifest mainly tracks intent.
        q_raw += float(entry.get("net_charge_raw", 0.0) or 0.0)
        q_scaled += float(entry.get("net_charge_scaled", entry.get("net_charge_raw", 0.0)) or 0.0)
    payload = {
        "density_g_cm3": None,
        "species": [dict(v) for v in species_entries],
        "pack_mode": "layer_stack",
        "layer_stack_name": str(stack_name),
        "master_xy_nm": [float(master_xy_nm[0]), float(master_xy_nm[1])],
        "box_nm": [float(v) for v in box_nm],
        "target_atoms": int(sum(int(v.get("natoms", 0) or 0) * int(v.get("n", 0) or 0) for v in species_entries)),
        "net_charge_raw": float(q_raw),
        "net_charge_scaled": float(q_scaled),
        "charge_tolerance": 1.0e-2,
        "net_charge_ok": bool(abs(q_scaled) <= 1.0e-2),
    }
    cell.SetProp("_yadonpy_cell_meta", json.dumps(payload, ensure_ascii=False))


def _read_gro_atoms(gro_path: Path) -> tuple[np.ndarray, int]:
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        return np.zeros((0, 3), dtype=float), 0
    try:
        nat = int(lines[1].strip())
    except Exception:
        nat = max(len(lines) - 3, 0)
    coords = np.zeros((nat, 3), dtype=float)
    for i in range(nat):
        line = lines[2 + i]
        coords[i, 0] = float(line[20:28])
        coords[i, 1] = float(line[28:36])
        coords[i, 2] = float(line[36:44])
    return coords, nat


def _layer_group_names(layer: LayerSpec, index: int) -> tuple[str, str]:
    base = _safe_name(getattr(layer, "name", f"LAYER_{index:02d}"))
    return (f"LAYER_{index:02d}_{base}", base)


def _write_layer_ndx_groups(
    ndx_path: Path,
    *,
    gro_path: Path,
    layers: Sequence[LayerSpec],
    intervals: Sequence[dict[str, Any]],
) -> dict[str, list[int]]:
    coords, nat = _read_gro_atoms(gro_path)
    existing = read_ndx_groups(ndx_path) if Path(ndx_path).is_file() else {}
    groups: dict[str, list[int]] = {str(k): list(v) for k, v in existing.items()}
    system = list(range(1, nat + 1))
    groups.setdefault("System", system)
    groups["SYSTEM"] = system
    mobile: set[int] = set()

    interval_by_index = {int(v["layer_index"]): v for v in intervals}
    for idx, layer in enumerate(layers):
        group_name, alias = _layer_group_names(layer, idx)
        if isinstance(layer, VacuumLayerSpec) or idx not in interval_by_index:
            groups[group_name] = []
            groups[alias] = []
            continue
        interval = interval_by_index[idx]
        zlo = float(interval["z_lo_nm"]) - 1.0e-4
        zhi = float(interval["z_hi_nm"]) + 1.0e-4
        atom_ids = [int(i + 1) for i, z in enumerate(coords[:, 2]) if zlo <= float(z) <= zhi]
        groups[group_name] = atom_ids
        groups[alias] = sorted(set(groups.get(alias, [])) | set(atom_ids))
        kind = "GRAPHITE" if isinstance(layer, GraphiteLayerSpec) else str(layer.layer_kind).upper()
        groups[kind] = sorted(set(groups.get(kind, [])) | set(atom_ids))
        if kind != "GRAPHITE":
            mobile.update(atom_ids)

    groups["MOBILE"] = sorted(mobile)
    ordered = [(name, idxs) for name, idxs in groups.items()]
    _write_ndx(ndx_path, ordered)
    return groups


def _acceptance_summary(
    *,
    intervals: Sequence[dict[str, Any]],
    layer_groups: Mapping[str, Sequence[int]],
    layers: Sequence[LayerSpec],
    box_nm: tuple[float, float, float],
    pbc_mode: str,
    min_gap_nm: float,
) -> dict[str, Any]:
    nonvac = [v for v in intervals]
    ordered_ok = all(float(nonvac[i]["z_hi_nm"]) <= float(nonvac[i + 1]["z_lo_nm"]) + 1.0e-6 for i in range(len(nonvac) - 1))
    gaps = []
    for left, right in zip(nonvac, nonvac[1:]):
        gaps.append(
            {
                "left": str(left["name"]),
                "right": str(right["name"]),
                "gap_nm": float(float(right["z_lo_nm"]) - float(left["z_hi_nm"])),
            }
        )
    resolved_pbc = "xyz" if str(pbc_mode).lower() == "auto" else str(pbc_mode).lower()
    closing_gap_nm = None
    closing_ok = True
    if resolved_pbc == "xyz" and nonvac:
        closing_gap_nm = float(box_nm[2]) - float(nonvac[-1]["z_hi_nm"]) + float(nonvac[0]["z_lo_nm"])
        # This is a geometry safety check rather than a thermodynamic criterion.
        # A small positive periodic closing gap avoids step-0 overlap explosions,
        # while explicit vacuum stacks can naturally have a much larger value.
        required_gap_nm = max(0.05, min(0.20, 0.5 * float(min_gap_nm)))
        closing_ok = closing_gap_nm >= required_gap_nm
    has_mobile = bool(layer_groups.get("MOBILE"))
    ok = bool(ordered_ok and has_mobile and closing_ok)
    return {
        "ok": ok,
        "phase_order_ok": bool(ordered_ok),
        "has_mobile_atoms": bool(has_mobile),
        "adjacent_gaps_nm": gaps,
        "pbc_closing_gap_nm": closing_gap_nm,
        "pbc_closing_gap_ok": bool(closing_ok),
        "recommended_action": ("ready_for_relaxation" if ok else "inspect_layer_stack_manifest"),
        "checked_layers": [getattr(layer, "name", f"LAYER_{i:02d}") for i, layer in enumerate(layers)],
    }


def _manifest_payload(
    *,
    spec: LayerStackSpec,
    relaxation: LayerStackRelaxationSpec,
    planning: Mapping[str, Any],
    layer_reports: Sequence[dict[str, Any]],
    intervals: Sequence[dict[str, Any]],
    gap_reports: Sequence[dict[str, Any]],
    box_nm: tuple[float, float, float],
    export: SystemExportResult,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    pbc_mode = spec.pbc_mode
    if pbc_mode == "auto":
        pbc_mode = "xyz"
    layer_payloads, z_compaction = _layer_stack_compaction_metadata(
        layer_reports=layer_reports,
        intervals=intervals,
        gap_reports=gap_reports,
        box_nm=box_nm,
    )
    return {
        "schema_version": 1,
        "builder": "yadonpy.interface.layer_stack",
        "name": spec.name,
        "order": spec.order,
        "pbc_mode": pbc_mode,
        "master_xy_nm": list(planning.get("master_xy_nm", [])),
        "box_nm": [float(v) for v in box_nm],
        "layers": layer_payloads,
        "layer_intervals_nm": [dict(v) for v in intervals],
        "gaps": [dict(v) for v in gap_reports],
        "acceptance": dict(acceptance),
        "z_compaction": z_compaction,
        "relaxation_defaults": {
            "temperature_K": float(relaxation.temperature_K),
            "pressure_bar": float(relaxation.pressure_bar),
            "early_dt_ps": float(relaxation.early_dt_ps),
            "early_constraints": str(relaxation.early_constraints),
            "final_dt_ps": float(relaxation.final_dt_ps),
            "final_constraints": str(relaxation.final_constraints),
            "freeze_layers": list(relaxation.freeze_layers),
            "run_relaxation": bool(relaxation.run_relaxation),
            "sample_ns": float(relaxation.sample_ns),
        },
        "planning": dict(planning),
        "artifacts": {
            "system_gro": str(export.system_gro),
            "system_top": str(export.system_top),
            "system_ndx": str(export.system_ndx),
            "system_meta": str(export.system_meta),
            "molecules_dir": str(export.molecules_dir),
        },
    }


def _layer_stack_compaction_metadata(
    *,
    layer_reports: Sequence[dict[str, Any]],
    intervals: Sequence[dict[str, Any]],
    gap_reports: Sequence[dict[str, Any]],
    box_nm: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    interval_by_index = {int(v.get("layer_index", i)): dict(v) for i, v in enumerate(intervals)}
    payloads: list[dict[str, Any]] = []
    target_nonvac_z = 0.0
    actual_nonvac_z = 0.0
    for idx, report in enumerate(layer_reports):
        payload = dict(report)
        interval = interval_by_index.get(idx)
        actual = None
        if interval is not None:
            actual = max(0.0, float(interval.get("z_hi_nm", 0.0)) - float(interval.get("z_lo_nm", 0.0)))
            actual_nonvac_z += float(actual)
        kind = str(payload.get("kind") or "").strip().lower()
        if kind == "vacuum":
            target = float(payload.get("thickness_nm", 0.0) or 0.0)
        elif kind == "graphite":
            target = float(actual or payload.get("thickness_nm", 0.0) or 0.0)
        else:
            target = float(payload.get("thickness_nm", actual or 0.0) or 0.0)
        if kind != "vacuum":
            target_nonvac_z += float(target)
        payload["target_thickness_nm"] = float(target)
        payload["actual_thickness_nm"] = float(actual) if actual is not None else None
        payload["z_expansion_factor"] = (float(actual) / float(target)) if actual is not None and float(target) > 0.0 else None
        payloads.append(payload)

    target_gap_z = 0.0
    actual_gap_z = 0.0
    for gap in gap_reports:
        try:
            g = max(float(gap.get("gap_nm", 0.0) or 0.0), 0.0)
        except Exception:
            g = 0.0
        target_gap_z += g
        actual_gap_z += g
    target_box_z = float(target_nonvac_z + target_gap_z)
    if target_box_z <= 0.0:
        target_box_z = float(box_nm[2])
    return payloads, {
        "initial_box_z_nm": float(box_nm[2]),
        "target_box_z_nm": float(target_box_z),
        "target_nonvacuum_z_nm": float(target_nonvac_z),
        "actual_nonvacuum_z_nm": float(actual_nonvac_z),
        "target_gap_z_nm": float(target_gap_z),
        "actual_gap_z_nm": float(actual_gap_z),
        "box_z_expansion_factor": (float(box_nm[2]) / target_box_z) if target_box_z > 0.0 else None,
    }


def build_layer_stack(
    stack: LayerStackSpec | None = None,
    *,
    layers: Sequence[LayerSpec] | None = None,
    order: Literal["bottom_to_top", "top_to_bottom"] = "bottom_to_top",
    pbc_mode: Literal["auto", "xyz", "xy"] = "auto",
    name: str = "layer_stack",
    relaxation: LayerStackRelaxationSpec | None = None,
    work_dir: str | Path = "layer_stack_work",
    ff_name: str = "gaff2_mod",
    charge_method: str = "RESP",
    restart: bool | None = None,
    export: bool = True,
) -> LayerStackResult:
    """Build and export a generic z-stacked interface system.

    The function accepts either a pre-built :class:`LayerStackSpec` or an inline
    ``layers=[...]`` sequence.  Molecules are packed/prepared under one master
    XY footprint, stacked in z, exported to GROMACS, and annotated with layer
    groups in ``system.ndx``.
    """

    if stack is None:
        if layers is None:
            raise ValueError("build_layer_stack requires either stack=... or layers=...")
        stack = LayerStackSpec(layers=tuple(layers), order=order, pbc_mode=pbc_mode, name=name)
    if not stack.layers:
        raise ValueError("LayerStackSpec.layers must not be empty.")

    relaxation = relaxation or LayerStackRelaxationSpec()
    resolved_pbc_mode = "xyz" if str(stack.pbc_mode).lower() == "auto" else str(stack.pbc_mode).lower()
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    ordered_layers = _layer_order(stack.layers, stack.order)
    master_x_nm, master_y_nm, planning = _plan_master_xy(
        ordered_layers,
        auto_expand_graphite=stack.auto_expand_graphite,
        molecular_packing_expand=stack.molecular_packing_expand,
    )
    master_xy_nm = (master_x_nm, master_y_nm)

    prepared: list[tuple[int, LayerSpec, Chem.Mol, list[dict[str, Any]], dict[str, Any]]] = []
    all_species: list[dict[str, Any]] = []
    layer_reports: list[dict[str, Any]] = []
    for idx, layer in enumerate(ordered_layers):
        layer_dir = work_dir / "01_layers" / f"{idx:02d}_{_safe_name(getattr(layer, 'name', 'VACUUM')).lower()}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(layer, VacuumLayerSpec):
            report = {"name": layer.name, "kind": "vacuum", "thickness_nm": float(layer.thickness_nm), "work_dir": str(layer_dir)}
            layer_reports.append(report)
            continue
        if isinstance(layer, GraphiteLayerSpec):
            block, species_entries, report = _prepare_graphite_layer(
                layer,
                master_xy_nm=master_xy_nm,
                planned_dimensions=planning.get("graphite_dimensions", {}),
                work_dir=layer_dir,
            )
        elif isinstance(layer, MolecularLayerSpec):
            block, species_entries, report = _prepare_molecular_layer(
                layer,
                master_xy_nm=master_xy_nm,
                molecular_packing_expand=stack.molecular_packing_expand,
                work_dir=layer_dir,
                restart=restart,
            )
        else:  # pragma: no cover - defensive for future spec classes.
            raise TypeError(f"Unsupported layer spec type: {type(layer)!r}")
        for entry in species_entries:
            copied = dict(entry)
            copied["layer_index"] = int(idx)
            copied["layer_group"] = _layer_group_names(layer, idx)[0]
            all_species.append(copied)
        prepared.append((idx, layer, block, species_entries, report))
        layer_reports.append(report)

    stacked_cell, box_nm, intervals, gap_reports = _stack_non_vacuum_layers(
        prepared,
        ordered_layers,
        master_xy_nm=master_xy_nm,
        default_gap_nm=float(stack.default_gap_nm),
        bottom_padding_nm=float(stack.bottom_padding_nm),
        top_padding_nm=float(stack.top_padding_nm),
        periodic_closing_gap_nm=(float(stack.default_gap_nm) if resolved_pbc_mode == "xyz" else 0.0),
    )
    _write_combined_cell_meta(
        stacked_cell,
        species_entries=all_species,
        stack_name=stack.name,
        master_xy_nm=master_xy_nm,
        box_nm=box_nm,
    )
    stacked_cell.SetProp(
        _LAYER_META_PROP,
        json.dumps(
            {
                "schema_version": 1,
                "layers": [dict(v) for v in layer_reports],
                "layer_intervals_nm": [dict(v) for v in intervals],
                "master_xy_nm": [float(master_xy_nm[0]), float(master_xy_nm[1])],
                "box_nm": [float(v) for v in box_nm],
            },
            ensure_ascii=False,
        ),
    )

    if not export:
        raise ValueError("build_layer_stack(export=False) is not supported yet because callers need GROMACS artifacts.")

    export_dir = work_dir / "02_system"
    system_export = export_system_from_cell_meta(
        cell_mol=stacked_cell,
        out_dir=export_dir,
        ff_name=ff_name,
        charge_method=charge_method,
        polyelectrolyte_mode=any(
            isinstance(layer, MolecularLayerSpec) and str(layer.layer_kind).lower() in {"cmcna", "polyelectrolyte"}
            for layer in ordered_layers
        ),
    )
    layer_groups = _write_layer_ndx_groups(
        Path(system_export.system_ndx),
        gro_path=Path(system_export.system_gro),
        layers=ordered_layers,
        intervals=intervals,
    )
    acceptance = _acceptance_summary(
        intervals=intervals,
        layer_groups=layer_groups,
        layers=ordered_layers,
        box_nm=box_nm,
        pbc_mode=resolved_pbc_mode,
        min_gap_nm=float(stack.default_gap_nm),
    )
    manifest = _manifest_payload(
        spec=LayerStackSpec(
            layers=tuple(ordered_layers),
            order="bottom_to_top",
            pbc_mode=stack.pbc_mode,
            name=stack.name,
            default_gap_nm=stack.default_gap_nm,
            bottom_padding_nm=stack.bottom_padding_nm,
            top_padding_nm=stack.top_padding_nm,
            auto_expand_graphite=stack.auto_expand_graphite,
            molecular_packing_expand=stack.molecular_packing_expand,
        ),
        relaxation=relaxation,
        planning=planning,
        layer_reports=layer_reports,
        intervals=intervals,
        gap_reports=gap_reports,
        box_nm=box_nm,
        export=system_export,
        acceptance=acceptance,
    )
    manifest_path = work_dir / "layer_stack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return LayerStackResult(
        work_dir=work_dir,
        stack_spec=stack,
        system_export=system_export,
        system_gro=Path(system_export.system_gro),
        system_top=Path(system_export.system_top),
        system_ndx=Path(system_export.system_ndx),
        manifest_path=manifest_path,
        stacked_cell=stacked_cell,
        layer_reports=tuple(layer_reports),
        acceptance=acceptance,
        box_nm=box_nm,
    )


def _latest_existing_file(paths: Sequence[Path]) -> Path | None:
    found = [Path(p) for p in paths if Path(p).is_file()]
    if not found:
        return None
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0]


def _copy_text_file(src: Path, dst: Path) -> None:
    if Path(src).is_file():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text(Path(src).read_text(encoding="utf-8", errors="replace"), encoding="utf-8")


def _copy_system_export_dir(result: LayerStackResult, dst: Path) -> Path:
    """Copy a stack export so relative topology includes remain valid."""

    src_dir = Path(result.system_gro).parent
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src_dir.is_dir():
        shutil.copytree(src_dir, dst, dirs_exist_ok=True)
    _copy_text_file(Path(result.system_gro), dst / "system.gro")
    _copy_text_file(Path(result.system_top), dst / "system.top")
    _copy_text_file(Path(result.system_ndx), dst / "system.ndx")
    return dst


def _read_gro_lines_coords_box(gro_path: Path) -> tuple[list[str], np.ndarray, tuple[float, float, float]]:
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {gro_path}")
    nat = int(lines[1].strip())
    coords = np.zeros((nat, 3), dtype=float)
    for i in range(nat):
        line = lines[2 + i]
        coords[i, 0] = float(line[20:28])
        coords[i, 1] = float(line[28:36])
        coords[i, 2] = float(line[36:44])
    raw_box = [float(tok) for tok in lines[2 + nat].split()]
    if len(raw_box) < 3:
        raise ValueError(f"Invalid GRO box line in {gro_path}")
    return lines, coords, (float(raw_box[0]), float(raw_box[1]), float(raw_box[2]))


def _write_gro_lines_coords_box(src_lines: Sequence[str], coords: np.ndarray, box_nm: tuple[float, float, float], out_path: Path) -> None:
    nat = int(coords.shape[0])
    lines = list(src_lines)
    for i in range(nat):
        old = lines[2 + i]
        suffix = old[44:] if len(old) > 44 else ""
        lines[2 + i] = f"{old[:20]}{coords[i, 0]:8.3f}{coords[i, 1]:8.3f}{coords[i, 2]:8.3f}{suffix}"
    lines[2 + nat] = f"{float(box_nm[0]):10.5f}{float(box_nm[1]):10.5f}{float(box_nm[2]):10.5f}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _graphite_layer_group_names(manifest: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for idx, layer in enumerate(manifest.get("layers") or []):
        if str(layer.get("kind") or "").strip().lower() != "graphite":
            continue
        name = str(layer.get("name") or f"LAYER_{idx:02d}")
        out.append(f"LAYER_{idx:02d}_{_safe_name(name)}")
    return out


def _auto_target_z_nm(manifest: Mapping[str, Any], fallback_box_z_nm: float) -> float:
    compaction = manifest.get("z_compaction") if isinstance(manifest, Mapping) else None
    if isinstance(compaction, Mapping):
        try:
            target = float(compaction.get("target_box_z_nm"))
            if target > 0.0:
                return target
        except Exception:
            pass
    return float(fallback_box_z_nm)


def _minimal_periodic_z_span(z_values: np.ndarray, box_z_nm: float) -> tuple[float, float, float]:
    """Return the shortest circular z interval containing a group's atoms."""

    vals = np.sort(np.mod(np.asarray(z_values, dtype=float), float(box_z_nm)))
    if vals.size == 0:
        raise ValueError("Cannot compute a periodic z span for an empty group.")
    if vals.size == 1:
        z = float(vals[0])
        return z, z, 0.0
    gaps = np.diff(vals)
    wrap_gap = float(vals[0] + float(box_z_nm) - vals[-1])
    all_gaps = np.concatenate([gaps, np.asarray([wrap_gap], dtype=float)])
    largest_gap_idx = int(np.argmax(all_gaps))
    start_idx = (largest_gap_idx + 1) % int(vals.size)
    if start_idx == 0:
        ordered = vals
    else:
        ordered = np.concatenate([vals[start_idx:], vals[:start_idx] + float(box_z_nm)])
    start = float(ordered[0])
    end = float(ordered[-1])
    return start, end, float(max(end - start, 0.0))


def _unwrap_z_after(z_values: np.ndarray, lower_bound_nm: float, box_z_nm: float) -> np.ndarray:
    z = np.asarray(z_values, dtype=float).copy()
    period = float(box_z_nm)
    if period <= 0.0:
        return z
    mask = z <= float(lower_bound_nm)
    if np.any(mask):
        z[mask] += np.ceil((float(lower_bound_nm) - z[mask] + 1.0e-9) / period) * period
    return z


def _write_z_compressed_gro(
    *,
    input_gro: Path,
    output_gro: Path,
    ndx_path: Path,
    manifest: Mapping[str, Any],
    target_z_nm: float,
    max_z_shrink_per_cycle: float,
    target_z_tolerance: float,
    min_interlayer_gap_nm: float,
    geometry_mode: str,
) -> dict[str, Any]:
    lines, coords, box = _read_gro_lines_coords_box(input_gro)
    current_z = float(box[2])
    target_z = float(target_z_nm)
    if current_z <= target_z * (1.0 + float(target_z_tolerance)):
        shutil.copyfile(input_gro, output_gro)
        return {
            "applied": False,
            "reason": "target_z_reached",
            "input_box_z_nm": current_z,
            "output_box_z_nm": current_z,
            "target_z_nm": target_z,
            "z_shrink_fraction": 0.0,
        }

    shrink_fraction = min(max(float(max_z_shrink_per_cycle), 0.0), 0.50)
    requested_shrink_nm = min(current_z - target_z, current_z * shrink_fraction)
    if requested_shrink_nm <= 1.0e-6:
        shutil.copyfile(input_gro, output_gro)
        return {
            "applied": False,
            "reason": "zero_shrink",
            "input_box_z_nm": current_z,
            "output_box_z_nm": current_z,
            "target_z_nm": target_z,
            "z_shrink_fraction": 0.0,
        }

    mode = "inter_electrode" if str(geometry_mode).strip().lower() == "auto" else str(geometry_mode).strip().lower()
    groups = read_ndx_groups(ndx_path) if Path(ndx_path).is_file() else {}
    graphite_names = _graphite_layer_group_names(manifest)
    graphite_spans: list[dict[str, Any]] = []
    for name in graphite_names:
        atom_ids = [int(v) for v in groups.get(name, []) if 1 <= int(v) <= int(coords.shape[0])]
        if not atom_ids:
            continue
        z = coords[[i - 1 for i in atom_ids], 2]
        start, end, span = _minimal_periodic_z_span(z, current_z)
        graphite_spans.append(
            {
                "name": name,
                "atom_indices0": [int(i - 1) for i in atom_ids],
                "start_nm": float(start),
                "end_nm": float(end),
                "span_nm": float(span),
                "start_mod_nm": float(start % current_z),
                "end_mod_nm": float(end % current_z),
            }
        )

    new_coords = np.asarray(coords, dtype=float).copy()
    applied_mode = mode
    if mode == "inter_electrode" and len(graphite_spans) >= 2:
        bottom_span = graphite_spans[0]
        top_span = graphite_spans[-1]
        soft_lo = float(bottom_span["end_nm"])
        soft_hi = float(top_span["start_nm"])
        top_end = float(top_span["end_nm"])
        while soft_hi <= soft_lo + 1.0e-6:
            soft_hi += current_z
            top_end += current_z
        soft_span = max(soft_hi - soft_lo, 0.0)
        max_allowed_shrink = max(0.0, soft_span - float(min_interlayer_gap_nm))
        shrink_nm = min(float(requested_shrink_nm), max_allowed_shrink)
        if shrink_nm <= 1.0e-6:
            shutil.copyfile(input_gro, output_gro)
            return {
                "applied": False,
                "reason": "min_interlayer_gap_reached",
                "input_box_z_nm": current_z,
                "output_box_z_nm": current_z,
                "target_z_nm": target_z,
                "soft_region_nm": [float(soft_lo % current_z), float(soft_hi % current_z)],
                "soft_region_unwrapped_nm": [soft_lo, soft_hi],
                "graphite_spans_nm": [
                    {k: v for k, v in span.items() if k != "atom_indices0"} for span in graphite_spans
                ],
                "z_shrink_fraction": 0.0,
            }
        scale = max((soft_span - shrink_nm) / max(soft_span, 1.0e-12), 1.0e-6)
        bottom_mask = np.zeros(int(new_coords.shape[0]), dtype=bool)
        for idx0 in bottom_span["atom_indices0"]:
            if 0 <= int(idx0) < int(bottom_mask.size):
                bottom_mask[int(idx0)] = True
        z_unwrapped = _unwrap_z_after(new_coords[:, 2], soft_lo, current_z)
        in_soft = (~bottom_mask) & (z_unwrapped > soft_lo) & (z_unwrapped < soft_hi)
        above = (~bottom_mask) & (z_unwrapped >= soft_hi)
        z_unwrapped[in_soft] = soft_lo + (z_unwrapped[in_soft] - soft_lo) * scale
        z_unwrapped[above] -= shrink_nm
        new_coords[:, 2] = z_unwrapped
        soft_region = [float(soft_lo % current_z), float(soft_hi % current_z)]
        soft_region_unwrapped = [soft_lo, soft_hi]
    else:
        applied_mode = "global"
        shrink_nm = float(requested_shrink_nm)
        scale = max((current_z - shrink_nm) / max(current_z, 1.0e-12), 1.0e-6)
        new_coords[:, 2] *= scale
        soft_region = [0.0, current_z]
        soft_region_unwrapped = [0.0, current_z]

    new_z = max(current_z - float(shrink_nm), target_z)
    new_coords[:, 2] = np.mod(new_coords[:, 2], new_z)
    _write_gro_lines_coords_box(lines, new_coords, (float(box[0]), float(box[1]), float(new_z)), output_gro)
    return {
        "applied": True,
        "mode": applied_mode,
        "input_gro": str(input_gro),
        "output_gro": str(output_gro),
        "input_box_z_nm": current_z,
        "output_box_z_nm": float(new_z),
        "target_z_nm": target_z,
        "soft_region_nm": soft_region,
        "soft_region_unwrapped_nm": soft_region_unwrapped,
        "graphite_spans_nm": [
            {k: v for k, v in span.items() if k != "atom_indices0"} for span in graphite_spans
        ],
        "z_shrink_nm": float(shrink_nm),
        "z_shrink_fraction": float(shrink_nm) / current_z if current_z > 0.0 else None,
        "requested_z_shrink_nm": float(requested_shrink_nm),
        "scale": float(scale),
    }


def _ns_to_nsteps(time_ns: float, dt_ps: float) -> int:
    return max(1, int(round(float(time_ns) * 1000.0 / max(float(dt_ps), 1.0e-12))))


def _interval_to_nst(value: float | str | None, *, dt_ps: float, default_ps: float, disabled_if_none: bool = False) -> int:
    if value is None:
        return 0 if disabled_if_none else max(1, int(round(float(default_ps) / max(float(dt_ps), 1.0e-12))))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "auto"}:
            ps = float(default_ps)
        elif token in {"none", "off", "no", "false", "0"}:
            return 0
        else:
            ps = float(token)
    else:
        ps = float(value)
    if ps <= 0.0:
        return 0
    return max(1, int(round(ps / max(float(dt_ps), 1.0e-12))))


def _layer_stack_md_params(
    *,
    time_ns: float,
    dt_ps: float,
    temp: float,
    constraints: str,
    pbc_mode: str,
    gen_vel: str,
    continuation: str,
    traj_ps: float | str | None,
    energy_ps: float | str | None,
    log_ps: float | str | None,
    trr_ps: float | str | None,
    velocity_ps: float | str | None,
) -> dict[str, object]:
    from ..gmx.mdp_templates import default_mdp_params

    p = default_mdp_params()
    p.update(
        {
            "nsteps": _ns_to_nsteps(time_ns, dt_ps),
            "dt": float(dt_ps),
            "ref_t": float(temp),
            "gen_temp": float(temp),
            "constraints": str(constraints),
            "pbc": str(pbc_mode or "xyz"),
            "periodic_molecules": "yes",
            "gen_vel": str(gen_vel),
            "continuation": str(continuation),
            "nstxout": _interval_to_nst(traj_ps, dt_ps=dt_ps, default_ps=20.0),
            "nstxout_trr": _interval_to_nst(trr_ps, dt_ps=dt_ps, default_ps=20.0, disabled_if_none=True),
            "nstvout": _interval_to_nst(velocity_ps, dt_ps=dt_ps, default_ps=20.0, disabled_if_none=True),
            "nstenergy": _interval_to_nst(energy_ps, dt_ps=dt_ps, default_ps=10.0),
            "nstlog": _interval_to_nst(log_ps, dt_ps=dt_ps, default_ps=10.0),
        }
    )
    return p


def _summarize_density_profile(
    analysis_payload: dict[str, Any] | None,
    *,
    low_density_threshold_g_cm3: float = 0.02,
    extended_vacuum_span_nm: float = 0.75,
    cmc_reference_density_g_cm3: float = 1.5,
    cmc_warning_floor_g_cm3: float = 0.90,
    cmc_severe_floor_g_cm3: float = 0.75,
) -> dict[str, Any]:
    outputs = (analysis_payload or {}).get("outputs") if isinstance(analysis_payload, dict) else None
    profile_csv = Path(str((outputs or {}).get("z_density_profiles_csv") or ""))
    if not profile_csv.is_file():
        return {"available": False, "reason": "missing_z_density_profile"}

    phase_values: dict[str, list[float]] = {}
    totals: dict[tuple[float, float], float] = {}
    phase_by_bin: dict[tuple[float, float], dict[str, float]] = {}
    try:
        with profile_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("entity_kind") or "").strip().lower() != "phase":
                    continue
                entity = str(row.get("entity") or "").strip()
                rho = float(row.get("mass_density_g_cm3") or 0.0)
                z_lo = float(row.get("z_lo_nm") or 0.0)
                z_hi = float(row.get("z_hi_nm") or z_lo)
                phase_values.setdefault(entity, []).append(rho)
                key = (z_lo, z_hi)
                totals[key] = float(totals.get(key, 0.0) + rho)
                phase_by_bin.setdefault(key, {})[entity] = rho
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    phase_density: dict[str, dict[str, float | None]] = {}
    for phase, values in phase_values.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        positive = arr[arr > 1.0e-6]
        max_v = float(np.max(arr)) if arr.size else 0.0
        rich = arr[arr >= max(1.0e-6, 0.10 * max_v)] if max_v > 0.0 else np.asarray([], dtype=float)
        core = arr[arr >= max(1.0e-6, 0.50 * max_v)] if max_v > 0.0 else np.asarray([], dtype=float)
        payload: dict[str, float | None] = {
            "mean_nonzero_g_cm3": float(np.mean(positive)) if positive.size else None,
            "rich_region_mean_g_cm3": float(np.mean(rich)) if rich.size else None,
            "core_region_mean_g_cm3": float(np.mean(core)) if core.size else None,
            "max_g_cm3": max_v,
        }
        if "CMC" in str(phase).upper() and max_v > 0.0:
            cmc_bins = [
                (key, values)
                for key, values in phase_by_bin.items()
                if float(values.get(phase, 0.0)) >= max(1.0e-6, 0.50 * max_v)
            ]
            cmc_rich_bins = [
                (key, values)
                for key, values in phase_by_bin.items()
                if float(values.get(phase, 0.0)) >= max(1.0e-6, 0.10 * max_v)
            ]
            payload["core_region_total_density_g_cm3"] = (
                float(np.mean([totals.get(key, 0.0) for key, _values in cmc_bins])) if cmc_bins else None
            )
            payload["rich_region_total_density_g_cm3"] = (
                float(np.mean([totals.get(key, 0.0) for key, _values in cmc_rich_bins])) if cmc_rich_bins else None
            )
        phase_density[phase] = payload

    bins = sorted((zlo, zhi, rho) for (zlo, zhi), rho in totals.items())
    low = [item for item in bins if float(item[2]) < float(low_density_threshold_g_cm3)]
    max_span = 0.0
    if low:
        current_lo, current_hi = float(low[0][0]), float(low[0][1])
        for zlo, zhi, _rho in low[1:]:
            if float(zlo) <= current_hi + 1.0e-6:
                current_hi = max(current_hi, float(zhi))
            else:
                max_span = max(max_span, current_hi - current_lo)
                current_lo, current_hi = float(zlo), float(zhi)
        max_span = max(max_span, current_hi - current_lo)

    focus = {
        name: payload
        for name, payload in phase_density.items()
        if any(token in name.upper() for token in ("CMC", "ELECTROLYTE"))
    }
    return {
        "available": True,
        "phase_density_g_cm3": phase_density,
        "focus_phase_density_g_cm3": focus,
        "cmc_density_gate": _cmc_density_gate(
            phase_density,
            reference_density_g_cm3=float(cmc_reference_density_g_cm3),
            warning_floor_g_cm3=float(cmc_warning_floor_g_cm3),
            severe_floor_g_cm3=float(cmc_severe_floor_g_cm3),
        ),
        "vacuum_like": {
            "threshold_g_cm3": float(low_density_threshold_g_cm3),
            "extended_span_threshold_nm": float(extended_vacuum_span_nm),
            "low_density_bin_fraction": (float(len(low)) / float(len(bins))) if bins else None,
            "max_contiguous_span_nm": float(max_span),
            "has_extended_vacuum_like_region": bool(max_span >= float(extended_vacuum_span_nm)),
        },
    }


def _cmc_density_gate(
    phase_density: Mapping[str, Mapping[str, float | None]],
    *,
    reference_density_g_cm3: float = 1.5,
    warning_floor_g_cm3: float = 0.90,
    severe_floor_g_cm3: float = 0.75,
) -> dict[str, Any]:
    """Return a CMCNA density sanity gate without enforcing exact bulk density."""

    metric_priority = ("core_region_mean_g_cm3", "rich_region_mean_g_cm3", "mean_nonzero_g_cm3", "max_g_cm3")
    candidates: list[dict[str, Any]] = []
    for name, payload in phase_density.items():
        if "CMC" not in str(name).upper():
            continue
        metric = None
        value = None
        for key in metric_priority:
            raw = payload.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except Exception:
                continue
            metric = key
            break
        if value is None or metric is None:
            continue
        candidates.append(
            {
                "phase": str(name),
                "metric": metric,
                "density_g_cm3": float(value),
                "fraction_of_reference": (
                    float(value) / float(reference_density_g_cm3) if float(reference_density_g_cm3) > 0.0 else None
                ),
                "all_metrics": dict(payload),
            }
        )

    if not candidates:
        return {
            "available": False,
            "reason": "no_cmc_phase_density",
            "reference_bulk_density_g_cm3": float(reference_density_g_cm3),
            "warning_floor_g_cm3": float(warning_floor_g_cm3),
            "severe_floor_g_cm3": float(severe_floor_g_cm3),
        }

    primary = min(candidates, key=lambda item: float(item["density_g_cm3"]))
    primary_density = float(primary["density_g_cm3"])
    if primary_density < float(severe_floor_g_cm3):
        severity = "severe"
    elif primary_density < float(warning_floor_g_cm3):
        severity = "warning"
    else:
        severity = "ok"
    return {
        "available": True,
        "ok": severity == "ok",
        "severity": severity,
        "reference_bulk_density_g_cm3": float(reference_density_g_cm3),
        "warning_floor_g_cm3": float(warning_floor_g_cm3),
        "severe_floor_g_cm3": float(severe_floor_g_cm3),
        "primary_phase": primary["phase"],
        "primary_metric": primary["metric"],
        "primary_density_g_cm3": primary_density,
        "primary_fraction_of_reference": primary["fraction_of_reference"],
        "phases": candidates,
        "message": (
            "CMCNA density is a layer-model sanity check, not a requirement to equal the "
            f"{float(reference_density_g_cm3):.2f} g/cm^3 bulk reference."
        ),
    }


def _relaxation_diagnostics(
    *,
    initial_gro: Path,
    final_gro: Path | None,
    top: Path,
    analysis_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    from ..gmx.workflows._util import gro_topology_bond_geometry, read_gro_box_nm

    diagnostics: dict[str, Any] = {}
    try:
        initial_box = read_gro_box_nm(initial_gro)
        diagnostics["initial_box_nm"] = list(initial_box)
    except Exception as exc:
        initial_box = None
        diagnostics["initial_box_error"] = str(exc)
    if final_gro is not None and Path(final_gro).is_file():
        try:
            final_box = read_gro_box_nm(final_gro)
            diagnostics["final_box_nm"] = list(final_box)
            if initial_box is not None and float(initial_box[2]) > 0.0:
                diagnostics["box_z_ratio_final_over_initial"] = float(final_box[2]) / float(initial_box[2])
        except Exception as exc:
            diagnostics["final_box_error"] = str(exc)
        try:
            diagnostics["periodic_bond_geometry"] = gro_topology_bond_geometry(
                top=top,
                gro=Path(final_gro),
                periodic_dimensions=(True, True, True),
            )
        except Exception as exc:
            diagnostics["periodic_bond_geometry"] = {"ok": False, "error": str(exc)}
    if isinstance(analysis_payload, dict):
        health = analysis_payload.get("geometry_health") or {}
        diagnostics["geometry_health"] = health
        diagnostics["phase_order_ok"] = health.get("phase_order_ok")
        diagnostics["adjacent_gaps_nm"] = health.get("adjacent_gaps_nm")
        diagnostics["pbc_closing_gap_nm"] = health.get("pbc_closing_gap_nm")
        density_summary = _summarize_density_profile(analysis_payload)
        diagnostics["density_profile"] = density_summary
        diagnostics["cmc_density_gate"] = density_summary.get("cmc_density_gate")
    return diagnostics


def _resolve_relax_z(result: LayerStackResult, relax_z: bool | Literal["auto"]) -> tuple[bool, str]:
    """Resolve whether the layer-stack workflow should include z-NPT."""

    if isinstance(relax_z, bool):
        return bool(relax_z), "explicit_true" if relax_z else "explicit_false"
    token = str(relax_z).strip().lower()
    if token not in {"auto", ""}:
        if token in {"1", "true", "yes", "on"}:
            return True, "explicit_true"
        if token in {"0", "false", "no", "off"}:
            return False, "explicit_false"
        raise ValueError("relax_z must be True, False, or 'auto'.")
    if any(isinstance(layer, VacuumLayerSpec) for layer in result.stack_spec.layers):
        return False, "auto_explicit_vacuum_layer"
    pbc_mode = str(result.stack_spec.pbc_mode or "auto").strip().lower()
    if pbc_mode == "xy":
        return False, "auto_xy_pbc_open_z"
    return True, "auto_closed_nonvacuum_stack"


def _normalize_compression_anneal_spec(value: bool | ZCompressionAnnealSpec) -> ZCompressionAnnealSpec:
    if isinstance(value, ZCompressionAnnealSpec):
        return value
    if isinstance(value, bool):
        return ZCompressionAnnealSpec(enabled=bool(value))
    raise TypeError("compression_anneal must be a bool or ZCompressionAnnealSpec.")


def _resolve_compression_anneal(
    *,
    result: LayerStackResult,
    compression_anneal: bool | ZCompressionAnnealSpec,
    relax_z_enabled: bool,
) -> tuple[ZCompressionAnnealSpec, bool, str]:
    spec = _normalize_compression_anneal_spec(compression_anneal)
    enabled_raw = spec.enabled
    if isinstance(enabled_raw, bool):
        requested = bool(enabled_raw)
    else:
        token = str(enabled_raw).strip().lower()
        if token in {"1", "true", "yes", "on"}:
            requested = True
        elif token in {"0", "false", "no", "off"}:
            requested = False
        elif token in {"", "auto"}:
            requested = None
        else:
            raise ValueError("ZCompressionAnnealSpec.enabled must be True, False, or 'auto'.")
    if not relax_z_enabled:
        return spec, False, "relax_z_disabled"
    if requested is False:
        return spec, False, "explicit_false"

    layers = list(result.stack_spec.layers)
    if any(isinstance(layer, VacuumLayerSpec) for layer in layers):
        return spec, bool(requested is True), "explicit_true_with_vacuum" if requested is True else "auto_explicit_vacuum_layer"
    pbc_mode = str(result.stack_spec.pbc_mode or "auto").strip().lower()
    if pbc_mode == "xy":
        return spec, bool(requested is True), "explicit_true_with_xy_pbc" if requested is True else "auto_xy_pbc_open_z"

    graphite_count = sum(1 for layer in layers if isinstance(layer, GraphiteLayerSpec))
    molecular_count = sum(1 for layer in layers if isinstance(layer, MolecularLayerSpec))
    sandwich = graphite_count >= 2 and molecular_count >= 1
    if requested is True:
        return spec, True, "explicit_true"
    if sandwich:
        return spec, True, "auto_closed_graphite_sandwich"
    return spec, False, "auto_no_two_graphite_boundaries"


def _anneal_schedule_value(kind: str, cycle: int, cycles: int, normal: float, maximum: float) -> float:
    if str(kind).strip().lower() != "linear":
        raise ValueError("Only linear compression anneal schedules are supported.")
    if int(cycles) <= 0:
        return float(normal)
    frac = float(cycle) / float(max(int(cycles), 1))
    return float(normal) + (float(maximum) - float(normal)) * frac


def run_layer_stack_relaxation(
    result: LayerStackResult,
    *,
    work_dir: str | Path | None = None,
    time_ns: float = 2.0,
    pre_nvt_ns: float = 0.05,
    z_npt_ns: float = 0.50,
    final_nvt_ns: float | None = None,
    temp: float = 318.15,
    pressure_bar: float = 1.0,
    z_compressibility_bar_inv: float = 4.5e-5,
    xy_compressibility: float = 0.0,
    mpi: int = 1,
    omp: int = 14,
    gpu: int = 1,
    gpu_id: int | None = 0,
    restart: bool | None = None,
    dt_ps: float = 0.001,
    constraints: str = "none",
    traj_ps: float | str | None = "auto",
    energy_ps: float | str | None = "auto",
    log_ps: float | str | None = "auto",
    trr_ps: float | str | None = None,
    velocity_ps: float | str | None = None,
    gpu_offload_mode: str = "full",
    analysis_profile: str = "interface_fast",
    run_analysis: bool = True,
    relax_z: bool | Literal["auto"] = "auto",
    compression_anneal: bool | ZCompressionAnnealSpec = False,
) -> LayerStackRelaxationResult:
    """Relax an exported layer stack, optionally including fixed-XY z-NPT.

    The initial layer densities are geometry targets.  When ``relax_z`` resolves
    true, a z-NPT stage keeps the graphite-defined XY footprint fixed while the
    box length in z responds to pressure; when false, explicit-vacuum or open-z
    systems keep their constructed z spacing and go straight to final NVT.
    """

    from ..gmx.mdp_templates import (
        MINIM_STEEP_MDP,
        NPT_MDP,
        NPT_NO_CONSTRAINTS_MDP,
        NVT_MDP,
        NVT_NO_CONSTRAINTS_MDP,
        MdpSpec,
        default_mdp_params,
    )
    from ..gmx.workflows.eq import EqStage, EquilibrationJob
    from ..gmx.workflows._util import RunResources
    from ..runtime import resolve_restart
    from .bulk_resize import fixed_xy_semiisotropic_npt_overrides

    rst_flag = resolve_restart(restart)
    run_dir = Path(work_dir).expanduser().resolve() if work_dir is not None else Path(result.work_dir) / "03_relaxation_sampling"
    if (not rst_flag) and run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    system_dir = _copy_system_export_dir(result, run_dir / "02_system")
    _copy_text_file(Path(result.manifest_path), run_dir / "layer_stack_manifest.json")
    manifest_path = run_dir / "layer_stack_manifest.json"
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest_payload = {}

    pbc_mode = str(result.stack_spec.pbc_mode or "xyz")
    if pbc_mode == "auto":
        pbc_mode = "xyz"
    relax_z_enabled, relax_z_reason = _resolve_relax_z(result, relax_z)
    anneal_spec, anneal_enabled, anneal_reason = _resolve_compression_anneal(
        result=result,
        compression_anneal=compression_anneal,
        relax_z_enabled=relax_z_enabled,
    )
    constraints_mode = str(constraints)
    dynamic_template_nvt = NVT_NO_CONSTRAINTS_MDP if constraints_mode.strip().lower() == "none" else NVT_MDP
    dynamic_template_npt = NPT_NO_CONSTRAINTS_MDP if constraints_mode.strip().lower() == "none" else NPT_MDP

    minim = default_mdp_params()
    minim.update(
        {
            "nsteps": 50000,
            "emtol": 1000.0,
            "emstep": 0.01,
            "pbc": pbc_mode,
            "periodic_molecules": "yes",
            "constraints": "none",
        }
    )

    pre_nvt = _layer_stack_md_params(
        time_ns=float(pre_nvt_ns),
        dt_ps=float(dt_ps),
        temp=float(temp),
        constraints=constraints_mode,
        pbc_mode=pbc_mode,
        gen_vel="yes",
        continuation="no",
        traj_ps=traj_ps,
        energy_ps=energy_ps,
        log_ps=log_ps,
        trr_ps=trr_ps,
        velocity_ps=velocity_ps,
    )
    z_npt: dict[str, object] | None = None
    if relax_z_enabled:
        z_npt = _layer_stack_md_params(
            time_ns=float(z_npt_ns),
            dt_ps=float(dt_ps),
            temp=float(temp),
            constraints=constraints_mode,
            pbc_mode=pbc_mode,
            gen_vel="no",
            continuation="yes",
            traj_ps=traj_ps,
            energy_ps=energy_ps,
            log_ps=log_ps,
            trr_ps=trr_ps,
            velocity_ps=velocity_ps,
        )
        z_npt["pcoupl"] = "C-rescale"
        z_npt["ref_p"] = f"{float(pressure_bar):.6g} {float(pressure_bar):.6g}"
        z_npt.update(
            fixed_xy_semiisotropic_npt_overrides(
                pressure_bar=float(pressure_bar),
                z_compressibility_bar_inv=float(z_compressibility_bar_inv),
            )
        )
        z_npt["compressibility"] = f"{float(xy_compressibility):.6g} {float(z_compressibility_bar_inv):.6g}"

    final_nvt = _layer_stack_md_params(
        time_ns=float(time_ns if final_nvt_ns is None else final_nvt_ns),
        dt_ps=float(dt_ps),
        temp=float(temp),
        constraints=constraints_mode,
        pbc_mode=pbc_mode,
        gen_vel="no",
        continuation="yes",
        traj_ps=traj_ps,
        energy_ps=energy_ps,
        log_ps=log_ps,
        trr_ps=trr_ps,
        velocity_ps=velocity_ps,
    )

    use_gpu = bool(int(gpu))
    resources = RunResources(
        ntmpi=int(mpi),
        ntomp=int(omp),
        use_gpu=use_gpu,
        gpu_id=(None if gpu_id is None else str(gpu_id)),
        gpu_offload_mode=str(gpu_offload_mode),
    )
    workflow_dir = run_dir / "05_relaxation_workflow"
    stage_order: list[str] = []
    compression_cycles: list[dict[str, Any]] = []

    def _run_segment(start_gro: Path, segment_stages: Sequence[Any]) -> Path:
        job = EquilibrationJob(
            gro=Path(start_gro),
            top=system_dir / "system.top",
            ndx=system_dir / "system.ndx",
            provenance_ndx=system_dir / "system.ndx",
            out_dir=workflow_dir,
            stages=segment_stages,
            resources=resources,
        )
        job.run(restart=rst_flag)
        last_dir = workflow_dir / segment_stages[-1].name
        return last_dir / "md.gro"

    def _make_nvt_params(*, time_ns_value: float, temp_value: float, gen_vel: str, continuation: str) -> dict[str, object]:
        return _layer_stack_md_params(
            time_ns=float(time_ns_value),
            dt_ps=float(dt_ps),
            temp=float(temp_value),
            constraints=constraints_mode,
            pbc_mode=pbc_mode,
            gen_vel=gen_vel,
            continuation=continuation,
            traj_ps=traj_ps,
            energy_ps=energy_ps,
            log_ps=log_ps,
            trr_ps=trr_ps,
            velocity_ps=velocity_ps,
        )

    def _make_z_npt_params(
        *,
        time_ns_value: float,
        temp_value: float,
        pressure_xy_bar: float,
        pressure_z_bar: float,
        gen_vel: str,
        continuation: str,
    ) -> dict[str, object]:
        params = _layer_stack_md_params(
            time_ns=float(time_ns_value),
            dt_ps=float(dt_ps),
            temp=float(temp_value),
            constraints=constraints_mode,
            pbc_mode=pbc_mode,
            gen_vel=gen_vel,
            continuation=continuation,
            traj_ps=traj_ps,
            energy_ps=energy_ps,
            log_ps=log_ps,
            trr_ps=trr_ps,
            velocity_ps=velocity_ps,
        )
        params["pcoupl"] = "C-rescale"
        params["pcoupltype"] = "semiisotropic"
        params["ref_p"] = f"{float(pressure_xy_bar):.6g} {float(pressure_z_bar):.6g}"
        params["compressibility"] = f"{float(xy_compressibility):.6g} {float(z_compressibility_bar_inv):.6g}"
        return params

    if not anneal_enabled:
        stages = [
            EqStage(name="01_pre_minimize", kind="minim", mdp=MdpSpec(MINIM_STEEP_MDP, minim)),
            EqStage(name="02_pre_nvt", kind="nvt", mdp=MdpSpec(dynamic_template_nvt, pre_nvt)),
        ]
        final_stage_name = "03_final_nvt"
        if relax_z_enabled and z_npt is not None:
            stages.append(EqStage(name="03_z_npt", kind="npt", mdp=MdpSpec(dynamic_template_npt, z_npt)))
            final_stage_name = "04_final_nvt"
        stages.append(EqStage(name=final_stage_name, kind="nvt", mdp=MdpSpec(dynamic_template_nvt, final_nvt)))
        stage_order = [stage.name for stage in stages]
        _run_segment(system_dir / "system.gro", stages)
    else:
        workflow_dir.mkdir(parents=True, exist_ok=True)
        stage_index = 1

        def _stage_name(label: str) -> str:
            nonlocal stage_index
            name = f"{stage_index:02d}_{label}"
            stage_index += 1
            return name

        initial_stages = [
            EqStage(name=_stage_name("pre_minimize"), kind="minim", mdp=MdpSpec(MINIM_STEEP_MDP, minim)),
            EqStage(name=_stage_name("pre_nvt"), kind="nvt", mdp=MdpSpec(dynamic_template_nvt, pre_nvt)),
        ]
        stage_order.extend(stage.name for stage in initial_stages)
        current_gro = _run_segment(system_dir / "system.gro", initial_stages)

        normal_temp = float(temp if anneal_spec.normal_temp_K is None else anneal_spec.normal_temp_K)
        normal_pressure = float(pressure_bar if anneal_spec.normal_pressure_bar is None else anneal_spec.normal_pressure_bar)
        try:
            current_box = _read_gro_lines_coords_box(current_gro)[2]
            target_z = (
                _auto_target_z_nm(manifest_payload, current_box[2])
                if str(anneal_spec.target_z_nm).strip().lower() == "auto"
                else float(anneal_spec.target_z_nm)
            )
        except Exception:
            target_z = _auto_target_z_nm(manifest_payload, float(result.box_nm[2]))

        for cycle in range(1, max(int(anneal_spec.cycles), 0) + 1):
            hot_temp = _anneal_schedule_value(
                anneal_spec.temperature_schedule,
                cycle,
                int(anneal_spec.cycles),
                normal_temp,
                float(anneal_spec.tmax_K),
            )
            hot_pressure = _anneal_schedule_value(
                anneal_spec.pressure_schedule,
                cycle,
                int(anneal_spec.cycles),
                normal_pressure,
                float(anneal_spec.pmax_bar),
            )
            cycle_report: dict[str, Any] = {
                "cycle": int(cycle),
                "temperature_K": float(hot_temp),
                "pressure_z_bar": float(hot_pressure),
                "normal_temperature_K": float(normal_temp),
                "normal_pressure_bar": float(normal_pressure),
            }

            attempts = 2 if bool(anneal_spec.rollback_on_failure) else 1
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                shrink = float(anneal_spec.max_z_shrink_per_cycle) / (2.0 ** (attempt - 1))
                pressure_attempt = normal_pressure + (float(hot_pressure) - normal_pressure) / (2.0 ** (attempt - 1))
                geom_name = _stage_name(f"compress_c{cycle:02d}_geometry" if attempt == 1 else f"compress_c{cycle:02d}_geometry_retry{attempt}")
                geom_dir = workflow_dir / geom_name
                geom_gro = geom_dir / "md.gro"
                geom_dir.mkdir(parents=True, exist_ok=True)
                geometry_report = _write_z_compressed_gro(
                    input_gro=current_gro,
                    output_gro=geom_gro,
                    ndx_path=system_dir / "system.ndx",
                    manifest=manifest_payload,
                    target_z_nm=float(target_z),
                    max_z_shrink_per_cycle=shrink,
                    target_z_tolerance=float(anneal_spec.target_z_tolerance),
                    min_interlayer_gap_nm=float(anneal_spec.min_interlayer_gap_nm),
                    geometry_mode=str(anneal_spec.geometry_compression),
                )
                (geom_dir / "summary.json").write_text(json.dumps(geometry_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                stage_order.append(geom_name)
                cycle_report.setdefault("attempts", []).append(
                    {
                        "attempt": int(attempt),
                        "geometry_stage": geom_name,
                        "max_z_shrink_per_cycle": float(shrink),
                        "pressure_z_bar": float(pressure_attempt),
                        "geometry": geometry_report,
                    }
                )
                if not geometry_report.get("applied"):
                    cycle_report["stopped"] = True
                    cycle_report["reason"] = geometry_report.get("reason")
                    compression_cycles.append(cycle_report)
                    current_gro = geom_gro
                    break
                cycle_stages = [
                    EqStage(name=_stage_name(f"compress_c{cycle:02d}_minimize"), kind="minim", mdp=MdpSpec(MINIM_STEEP_MDP, minim)),
                    EqStage(
                        name=_stage_name(f"compress_c{cycle:02d}_hot_nvt"),
                        kind="nvt",
                        mdp=MdpSpec(
                            dynamic_template_nvt,
                            _make_nvt_params(
                                time_ns_value=float(anneal_spec.hot_nvt_ns),
                                temp_value=float(hot_temp),
                                gen_vel="yes",
                                continuation="no",
                            ),
                        ),
                    ),
                    EqStage(
                        name=_stage_name(f"compress_c{cycle:02d}_hot_z_npt"),
                        kind="npt",
                        mdp=MdpSpec(
                            dynamic_template_npt,
                            _make_z_npt_params(
                                time_ns_value=float(anneal_spec.compression_npt_ns),
                                temp_value=float(hot_temp),
                                pressure_xy_bar=normal_pressure,
                                pressure_z_bar=float(pressure_attempt),
                                gen_vel="no",
                                continuation="yes",
                            ),
                        ),
                    ),
                    EqStage(
                        name=_stage_name(f"compress_c{cycle:02d}_cool_nvt"),
                        kind="nvt",
                        mdp=MdpSpec(
                            dynamic_template_nvt,
                            _make_nvt_params(
                                time_ns_value=float(anneal_spec.cool_nvt_ns),
                                temp_value=normal_temp,
                                gen_vel="no",
                                continuation="yes",
                            ),
                        ),
                    ),
                ]
                cycle_report["attempts"][-1]["md_stages"] = [stage.name for stage in cycle_stages]
                try:
                    stage_order.extend(stage.name for stage in cycle_stages)
                    current_gro = _run_segment(geom_gro, cycle_stages)
                    cycle_report["accepted_attempt"] = int(attempt)
                    cycle_report["final_gro"] = str(current_gro)
                    compression_cycles.append(cycle_report)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    cycle_report["attempts"][-1]["error"] = str(exc)
                    if attempt >= attempts:
                        cycle_report["failed"] = True
                        cycle_report["error"] = str(exc)
                        compression_cycles.append(cycle_report)
                        raise
            if last_exc is None and cycle_report.get("stopped"):
                break

        final_z_npt_name = _stage_name("final_z_npt")
        final_stage_name = _stage_name("final_nvt")
        final_stages = [
            EqStage(
                name=final_z_npt_name,
                kind="npt",
                mdp=MdpSpec(
                    dynamic_template_npt,
                    _make_z_npt_params(
                        time_ns_value=float(z_npt_ns),
                        temp_value=normal_temp,
                        pressure_xy_bar=normal_pressure,
                        pressure_z_bar=normal_pressure,
                        gen_vel="yes",
                        continuation="no",
                    ),
                ),
            ),
            EqStage(
                name=final_stage_name,
                kind="nvt",
                mdp=MdpSpec(
                    dynamic_template_nvt,
                    _make_nvt_params(
                        time_ns_value=float(time_ns if final_nvt_ns is None else final_nvt_ns),
                        temp_value=normal_temp,
                        gen_vel="no",
                        continuation="yes",
                    ),
                ),
            ),
        ]
        stage_order.extend(stage.name for stage in final_stages)
        _run_segment(current_gro, final_stages)

    final_stage_dir = workflow_dir / final_stage_name
    final_gro = (
        final_stage_dir / "md.gro"
        if (final_stage_dir / "md.gro").is_file()
        else _latest_existing_file(list(workflow_dir.rglob("md.gro")))
    )
    xtc = (
        final_stage_dir / "md.xtc"
        if (final_stage_dir / "md.xtc").is_file()
        else _latest_existing_file(list(final_stage_dir.rglob("md.xtc")))
    )
    trr = (
        final_stage_dir / "md.trr"
        if (final_stage_dir / "md.trr").is_file()
        else _latest_existing_file(list(final_stage_dir.rglob("md.trr")))
    )
    trajectory = xtc if xtc is not None else trr

    analysis_summary: Path | None = None
    analysis_payload: dict[str, Any] | None = None
    if run_analysis and final_gro is not None:
        try:
            analysis_payload = analyze_layer_stack_interface(
                work_dir=run_dir,
                system_gro=final_gro,
                system_ndx=system_dir / "system.ndx",
                trajectory=trajectory,
                analysis_profile=analysis_profile,
                compute_transport=False,
            )
            out_path = analysis_payload.get("summary_path") if isinstance(analysis_payload, dict) else None
            if out_path:
                analysis_summary = Path(out_path)
        except Exception as exc:
            analysis_payload = {"error": str(exc)}

    diagnostics = _relaxation_diagnostics(
        initial_gro=system_dir / "system.gro",
        final_gro=final_gro,
        top=system_dir / "system.top",
        analysis_payload=analysis_payload,
    )
    summary_path = run_dir / "relaxation_followup_summary.json"
    anneal_payload = {
        "requested": bool(compression_anneal) if isinstance(compression_anneal, bool) else asdict(compression_anneal),
        "resolved": bool(anneal_enabled),
        "reason": anneal_reason,
        "spec": asdict(anneal_spec),
        "cycles": compression_cycles,
        "initial_z_compaction": manifest_payload.get("z_compaction") if isinstance(manifest_payload, dict) else None,
    }
    payload = {
        "schema_version": 1,
        "workflow": "layer_stack_relaxation",
        "work_dir": str(run_dir),
        "source_layer_stack": str(result.work_dir),
        "stage_order": stage_order,
        "time_ns": {
            "pre_nvt": float(pre_nvt_ns),
            "compression_anneal_hot_nvt_per_cycle": float(anneal_spec.hot_nvt_ns) if anneal_enabled else 0.0,
            "compression_anneal_z_npt_per_cycle": float(anneal_spec.compression_npt_ns) if anneal_enabled else 0.0,
            "compression_anneal_cool_nvt_per_cycle": float(anneal_spec.cool_nvt_ns) if anneal_enabled else 0.0,
            "z_npt": float(z_npt_ns) if relax_z_enabled else 0.0,
            "final_nvt": float(time_ns if final_nvt_ns is None else final_nvt_ns),
        },
        "relax_z": {
            "requested": relax_z,
            "resolved": bool(relax_z_enabled),
            "reason": relax_z_reason,
        },
        "compression_anneal": anneal_payload,
        "temperature_K": float(temp),
        "pressure_bar": float(pressure_bar),
        "dt_ps": float(dt_ps),
        "constraints": constraints_mode,
        "z_npt_mdp_overrides": (
            {
                "pcoupl": "C-rescale",
                "pcoupltype": "semiisotropic",
                "ref_p": z_npt["ref_p"],
                "compressibility": z_npt["compressibility"],
            }
            if relax_z_enabled and z_npt is not None
            else None
        ),
        "gpu_offload_mode": str(gpu_offload_mode),
        "resources": {"mpi": int(mpi), "omp": int(omp), "gpu": int(gpu), "gpu_id": gpu_id},
        "artifacts": {
            "system_gro": str(system_dir / "system.gro"),
            "system_top": str(system_dir / "system.top"),
            "system_ndx": str(system_dir / "system.ndx"),
            "workflow_summary": str(workflow_dir / "summary.json"),
            "final_gro": str(final_gro) if final_gro is not None else None,
            "trajectory": str(trajectory) if trajectory is not None else None,
            "xtc": str(xtc) if xtc is not None else None,
            "trr": str(trr) if trr is not None else None,
            "manifest": str(run_dir / "layer_stack_manifest.json"),
        },
        "diagnostics": diagnostics,
        "analysis": analysis_payload,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return LayerStackRelaxationResult(
        work_dir=run_dir,
        final_gro=final_gro,
        trajectory=trajectory,
        xtc=xtc,
        trr=trr,
        summary_path=summary_path,
        analysis_summary=analysis_summary,
        diagnostics=diagnostics,
    )


def run_layer_stack_nvt(
    result: LayerStackResult,
    *,
    work_dir: str | Path | None = None,
    time_ns: float = 2.0,
    temp: float = 318.15,
    mpi: int = 1,
    omp: int = 14,
    gpu: int = 1,
    gpu_id: int | None = 0,
    restart: bool | None = None,
    dt_ps: float = 0.002,
    constraints: str = "h-bonds",
    traj_ps: float | str | None = "auto",
    energy_ps: float | str | None = "auto",
    log_ps: float | str | None = "auto",
    trr_ps: float | str | None = None,
    velocity_ps: float | str | None = None,
    gpu_offload_mode: str = "full",
    performance_profile: str = "auto",
    analysis_profile: str = "interface_fast",
    run_analysis: bool = True,
) -> LayerStackNvtResult:
    """Run a short NVT sampling segment from a built layer-stack artifact.

    Parameters mirror the normal NVT preset so Example 08 can keep the same
    script-first style as Example 02.  The function copies the layer manifest
    and layer-aware NDX groups into the follow-up directory, which lets the
    interface analyzer keep using layer names after MD.
    """

    from ..sim.preset import eq

    run_dir = Path(work_dir).expanduser().resolve() if work_dir is not None else Path(result.work_dir) / "03_nvt_sampling"
    run_dir.mkdir(parents=True, exist_ok=True)
    nvt = eq.NVT(result.stacked_cell, work_dir=run_dir)

    # Prime the export and replace the generic index with layer-aware groups.
    try:
        exp = nvt._ensure_system_exported()  # type: ignore[attr-defined]
        _copy_text_file(Path(result.system_ndx), Path(exp.system_ndx))
    except Exception:
        exp = None

    nvt.exec(
        temp=float(temp),
        mpi=int(mpi),
        omp=int(omp),
        gpu=int(gpu),
        gpu_id=gpu_id,
        time=float(time_ns),
        traj_ps=traj_ps,
        energy_ps=energy_ps,
        log_ps=log_ps,
        trr_ps=trr_ps,
        velocity_ps=velocity_ps,
        dt_ps=float(dt_ps),
        constraints=str(constraints),
        gpu_offload_mode=str(gpu_offload_mode),
        performance_profile=str(performance_profile),
        analysis_profile="auto",
        pre_minimize=True,
        restart=restart,
    )

    system_dir = run_dir / "02_system"
    _copy_text_file(Path(result.system_ndx), system_dir / "system.ndx")
    _copy_text_file(Path(result.manifest_path), run_dir / "layer_stack_manifest.json")

    production_dir = run_dir / "05_nvt_production"
    final_gro = _latest_existing_file(list(production_dir.rglob("md.gro")))
    xtc = _latest_existing_file(list(production_dir.rglob("md.xtc")))
    trr = _latest_existing_file(list(production_dir.rglob("md.trr")))
    trajectory = xtc if xtc is not None else trr
    summary_path = run_dir / "nvt_followup_summary.json"
    analysis_summary: Path | None = None
    analysis_payload: dict[str, Any] | None = None
    if run_analysis and final_gro is not None:
        try:
            analysis_payload = analyze_layer_stack_interface(
                work_dir=run_dir,
                system_gro=final_gro,
                system_ndx=system_dir / "system.ndx",
                trajectory=trajectory,
                analysis_profile=analysis_profile,
            )
            out_path = analysis_payload.get("summary_path") if isinstance(analysis_payload, dict) else None
            if out_path:
                analysis_summary = Path(out_path)
        except Exception as exc:
            analysis_payload = {"error": str(exc)}
    payload = {
        "schema_version": 1,
        "work_dir": str(run_dir),
        "source_layer_stack": str(result.work_dir),
        "time_ns": float(time_ns),
        "temperature_K": float(temp),
        "dt_ps": float(dt_ps),
        "constraints": str(constraints),
        "gpu_offload_mode": str(gpu_offload_mode),
        "resources": {"mpi": int(mpi), "omp": int(omp), "gpu": int(gpu), "gpu_id": gpu_id},
        "artifacts": {
            "system_gro": str(system_dir / "system.gro"),
            "system_top": str(system_dir / "system.top"),
            "system_ndx": str(system_dir / "system.ndx"),
            "final_gro": str(final_gro) if final_gro is not None else None,
            "trajectory": str(trajectory) if trajectory is not None else None,
            "xtc": str(xtc) if xtc is not None else None,
            "trr": str(trr) if trr is not None else None,
            "manifest": str(run_dir / "layer_stack_manifest.json"),
        },
        "analysis": analysis_payload,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return LayerStackNvtResult(
        work_dir=run_dir,
        final_gro=final_gro,
        trajectory=trajectory,
        xtc=xtc,
        trr=trr,
        summary_path=summary_path,
        analysis_summary=analysis_summary,
    )


def _resolve_stack_artifacts(
    *,
    result: LayerStackResult | None = None,
    work_dir: str | Path | None = None,
    system_gro: str | Path | None = None,
    system_ndx: str | Path | None = None,
    trajectory: str | Path | None = None,
) -> tuple[Path, Path, Path | None, Path | None]:
    if result is not None:
        base = Path(result.work_dir)
        gro = Path(system_gro or result.system_gro)
        ndx = Path(system_ndx or result.system_ndx)
        manifest = Path(result.manifest_path)
    elif work_dir is not None:
        base = Path(work_dir).expanduser().resolve()
        gro = Path(system_gro or base / "02_system" / "system.gro")
        ndx = Path(system_ndx or base / "02_system" / "system.ndx")
        manifest = base / "layer_stack_manifest.json"
    else:
        raise ValueError("Provide result=... or work_dir=... for layer-stack analysis.")
    traj = Path(trajectory) if trajectory is not None else None
    return gro, ndx, traj, manifest


def analyze_layer_stack_interface(
    *,
    result: LayerStackResult | None = None,
    work_dir: str | Path | None = None,
    system_gro: str | Path | None = None,
    system_ndx: str | Path | None = None,
    trajectory: str | Path | None = None,
    manifest_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    bin_nm: float = 0.05,
    frame_stride: int | str = "auto",
    region_width_nm: float = 0.75,
    surface_distance_nm: float = 0.50,
    surface_grid_nm: float = 0.5,
    penetration_threshold_nm: float = 0.20,
    adsorption_min_residence_ps: float = 10.0,
    potential_reference: str = "zero_mean",
    split_electrodes: bool = False,
    report_potential_drop: bool = False,
    penetration_species: Sequence[str] | None = None,
    adsorption_species: Sequence[str] | None = None,
    analysis_profile: str = "interface_fast",
    phase_groups: Sequence[str] | None = None,
    compute_transport: bool = True,
    time_series_analysis: bool = False,
    time_series_sample_count: int = 10,
    time_series_fps: float = 1.0,
    time_series_rdf: bool = True,
    time_series_concentration: bool = True,
    time_series_angles: bool = True,
    time_series_rdf_rmax_nm: float = 1.2,
    time_series_rdf_bin_nm: float = 0.02,
) -> dict[str, Any]:
    """Analyze z profiles and adjacent-interface diagnostics for a layer stack."""

    gro, ndx, traj, resolved_manifest_path = _resolve_stack_artifacts(
        result=result,
        work_dir=work_dir,
        system_gro=system_gro,
        system_ndx=system_ndx,
        trajectory=trajectory,
    )
    manifest_path = Path(manifest_path) if manifest_path is not None else resolved_manifest_path
    manifest = {}
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if phase_groups is None:
        phase_groups = [
            str(v.get("name"))
            for v in manifest.get("layers", [])
            if str(v.get("kind", "")).lower() != "vacuum" and v.get("name")
        ]
    if out_dir is None:
        root = Path(work_dir).expanduser().resolve() if work_dir is not None else Path(gro).parent.parent
        out_dir = root / "06_analysis" / "layer_stack_interface"
    from ..gmx.analysis.interface_profile import compute_interface_profile

    return compute_interface_profile(
        gro_path=gro,
        top_path=(Path(work_dir).expanduser().resolve() / "02_system" / "system.top") if work_dir is not None else (Path(gro).parent / "system.top"),
        ndx_path=ndx,
        system_dir=(Path(work_dir).expanduser().resolve() / "02_system") if work_dir is not None else Path(gro).parent,
        xtc_path=traj,
        out_dir=Path(out_dir),
        bin_nm=float(bin_nm),
        frame_stride=frame_stride,
        region_width_nm=float(region_width_nm),
        surface_distance_nm=float(surface_distance_nm),
        surface_grid_nm=float(surface_grid_nm),
        penetration_threshold_nm=float(penetration_threshold_nm),
        adsorption_min_residence_ps=float(adsorption_min_residence_ps),
        potential_reference=str(potential_reference),
        split_electrodes=bool(split_electrodes),
        report_potential_drop=bool(report_potential_drop),
        penetration_species=penetration_species,
        adsorption_species=adsorption_species,
        analysis_profile=str(analysis_profile),
        phase_groups=tuple(phase_groups),
        manifest_path=manifest_path if manifest_path and manifest_path.is_file() else None,
        compute_transport=bool(compute_transport),
        time_series_analysis=bool(time_series_analysis),
        time_series_sample_count=int(time_series_sample_count),
        time_series_fps=float(time_series_fps),
        time_series_rdf=bool(time_series_rdf),
        time_series_concentration=bool(time_series_concentration),
        time_series_angles=bool(time_series_angles),
        time_series_rdf_rmax_nm=float(time_series_rdf_rmax_nm),
        time_series_rdf_bin_nm=float(time_series_rdf_bin_nm),
    )


__all__ = [
    "ElectrodeChargeSpec",
    "GraphiteLayerSpec",
    "LayerStackNvtResult",
    "LayerStackRelaxationResult",
    "LayerStackRelaxationSpec",
    "LayerStackResult",
    "LayerStackSpec",
    "MolecularLayerSpec",
    "VacuumLayerSpec",
    "ZCompressionAnnealSpec",
    "analyze_layer_stack_interface",
    "build_layer_stack",
    "run_layer_stack_relaxation",
    "run_layer_stack_nvt",
]
