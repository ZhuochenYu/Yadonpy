"""Preparation routines for reusable bulk phases and interface inputs.

The functions in this module turn user-facing specifications into equilibrated,
validated GROMACS-ready phases. They bridge molecule preparation, system export,
bulk equilibration, and restart-friendly artifact discovery for later slab or
sandwich assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import math

from rdkit import Chem

from ..core import utils
from ..io.gromacs_system import SystemExportResult, validate_exported_system_dir
from ..sim.preset import eq
from .bulk_resize import (
    BulkEquilibriumProfile,
    BulkRescalePlan,
    DirectElectrolytePlan,
    ElectrolyteAlignmentPlan,
    FixedXYDirectPackPlan,
    build_bulk_equilibrium_profile,
    fixed_xy_semiisotropic_npt_overrides,
    plan_direct_electrolyte_counts,
    plan_fixed_xy_direct_pack_box,
    plan_resized_electrolyte_counts,
    recommend_fixed_xy_pack_parameters,
    recommend_electrolyte_alignment,
)

if TYPE_CHECKING:
    from .builder import AreaMismatchPolicy, InterfaceRouteSpec
    from .protocol import InterfaceProtocol


@dataclass(frozen=True)
class FixedXYElectrolytePreparation:
    reference_box_nm: tuple[float, float, float]
    direct_plan: DirectElectrolytePlan
    pack_plan: FixedXYDirectPackPlan
    relax_mdp_overrides: dict[str, Any]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolymerAnchoredInterfacePreparation:
    reference_box_nm: tuple[float, float, float]
    interface_xy_nm: tuple[float, float]
    bottom_thickness_nm: float
    top_thickness_nm: float
    gap_nm: float
    surface_shell_nm: float
    polymer_margin_nm: float
    is_polyelectrolyte: bool
    polymer_target_box_nm: tuple[float, float, float]
    electrolyte_target_box_nm: tuple[float, float, float]
    electrolyte_alignment: ElectrolyteAlignmentPlan
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DirectPolymerMatchedInterfacePreparation:
    reference_box_nm: tuple[float, float, float]
    interface_plan: PolymerAnchoredInterfacePreparation
    electrolyte_prep: FixedXYElectrolytePreparation
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbePolymerMatchedInterfacePreparation:
    reference_box_nm: tuple[float, float, float]
    interface_plan: PolymerAnchoredInterfacePreparation
    probe_prep: "ProbeElectrolytePreparation"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResizedPolymerMatchedInterfacePreparation:
    reference_box_nm: tuple[float, float, float]
    interface_plan: PolymerAnchoredInterfacePreparation
    resized_prep: "ResizedElectrolytePreparation"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeElectrolytePreparation:
    reference_box_nm: tuple[float, float, float]
    target_box_nm: tuple[float, float, float]
    probe_box_nm: tuple[float, float, float]
    direct_plan: DirectElectrolytePlan
    pack_plan: FixedXYDirectPackPlan
    build_density_g_cm3: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResizedElectrolytePreparation:
    reference_box_nm: tuple[float, float, float]
    target_box_nm: tuple[float, float, float]
    profile: BulkEquilibriumProfile
    resize_plan: BulkRescalePlan
    pack_plan: FixedXYDirectPackPlan
    relax_mdp_overrides: dict[str, Any]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BulkEq21Outcome:
    final_cell: Any
    system_export: SystemExportResult
    raw_system_meta: Path


@dataclass(frozen=True)
class PolymerDiffusionInterfaceRecipe:
    interface_plan: PolymerAnchoredInterfacePreparation
    route_spec: "InterfaceRouteSpec"
    protocol: "InterfaceProtocol"
    notes: tuple[str, ...] = ()


def _derive_isotropic_probe_box(
    *,
    target_box_nm: tuple[float, float, float],
    volume_scale: float,
    minimum_box_nm: float,
) -> tuple[float, float, float]:
    target_box = tuple(float(x) for x in target_box_nm)
    if len(target_box) != 3 or min(target_box) <= 0.0:
        raise ValueError("target_box_nm must contain three positive lengths")
    scale = float(volume_scale)
    if scale < 1.0:
        raise ValueError("probe_volume_scale must be >= 1.0")
    minimum_edge = float(minimum_box_nm)
    if minimum_edge <= 0.0:
        raise ValueError("minimum_probe_box_nm must be positive")

    target_volume = float(target_box[0] * target_box[1] * target_box[2])
    edge_nm = max(minimum_edge, float((target_volume * scale) ** (1.0 / 3.0)))
    return (edge_nm, edge_nm, edge_nm)


def plan_probe_electrolyte_preparation(
    *,
    reference_box_nm: tuple[float, float, float],
    target_box_nm: tuple[float, float, float],
    target_density_g_cm3: float,
    solvent_mol_weights: Sequence[float],
    solvent_mass_ratio: Sequence[float],
    salt_mol_weights: Sequence[float],
    salt_molarity_M: float,
    min_salt_pairs: int,
    solvent_species_names: Sequence[str] | None = None,
    salt_species_names: Sequence[str] | None = None,
    min_solvent_counts: Sequence[int] | None = None,
    probe_volume_scale: float = 1.75,
    minimum_probe_box_nm: float | None = None,
    initial_pack_density_g_cm3: float = 0.70,
    z_padding_factor: float = 1.10,
) -> ProbeElectrolytePreparation:
    ref_box = tuple(float(x) for x in reference_box_nm)
    target_box = tuple(float(x) for x in target_box_nm)
    probe_box = _derive_isotropic_probe_box(
        target_box_nm=target_box,
        volume_scale=float(probe_volume_scale),
        minimum_box_nm=float(minimum_probe_box_nm) if minimum_probe_box_nm is not None else max(target_box),
    )
    direct_plan = plan_direct_electrolyte_counts(
        target_box_nm=probe_box,
        target_density_g_cm3=float(target_density_g_cm3),
        solvent_mol_weights=solvent_mol_weights,
        solvent_mass_ratio=solvent_mass_ratio,
        salt_mol_weights=salt_mol_weights,
        salt_molarity_M=float(salt_molarity_M),
        min_salt_pairs=int(min_salt_pairs),
        solvent_species_names=solvent_species_names,
        salt_species_names=salt_species_names,
        min_solvent_counts=min_solvent_counts,
    )
    pack_plan = plan_fixed_xy_direct_pack_box(
        reference_box_nm=probe_box,
        target_counts=direct_plan.target_counts,
        mol_weights=tuple(float(x) for x in solvent_mol_weights) + tuple(float(x) for x in salt_mol_weights),
        species_names=direct_plan.species_names,
        initial_pack_density_g_cm3=float(initial_pack_density_g_cm3),
        z_padding_factor=float(z_padding_factor),
        minimum_z_nm=float(probe_box[2]),
    )
    build_density = float(initial_pack_density_g_cm3)
    notes = (
        "build the electrolyte probe bulk in a density-driven isotropic pack instead of locking XY to the polymer footprint too early",
        f"probe_box_nm={tuple(round(x, 4) for x in probe_box)} derived from target electrolyte box {tuple(round(x, 4) for x in target_box)} with probe_volume_scale={float(probe_volume_scale):.3f}",
        f"the probe stage uses density={build_density:.4f} g/cm^3 for amorphous_cell so retry fallback can lower density and expand the whole box isotropically before the later polymer-footprint resize",
        f"the probe stage uses the same chemistry targets but delays the polymer-footprint resize until after probe equilibration; reference polymer box remains {tuple(round(x, 4) for x in ref_box)}",
    )
    return ProbeElectrolytePreparation(
        reference_box_nm=ref_box,
        target_box_nm=target_box,
        probe_box_nm=probe_box,
        direct_plan=direct_plan,
        pack_plan=pack_plan,
        build_density_g_cm3=build_density,
        notes=notes,
    )


def plan_resized_electrolyte_preparation_from_probe(
    *,
    reference_box_nm: tuple[float, float, float],
    target_box_nm: tuple[float, float, float],
    probe_work_dir,
    probe_counts: Sequence[int],
    mol_weights: Sequence[float],
    species_names: Sequence[str],
    solvent_indices: Sequence[int],
    solvent_groups: Sequence[Sequence[int]] | None = None,
    salt_pair_indices: Sequence[int] | None = None,
    salt_pair_groups: Sequence[Sequence[int]] | None = None,
    min_solvent_counts: Sequence[int] | None = None,
    min_solvent_group_counts: Sequence[Sequence[int]] | None = None,
    min_salt_pairs: int | Sequence[int] = 1,
    initial_pack_density_g_cm3: float | None = None,
    z_padding_factor: float | None = None,
    minimum_pack_z_factor: float | None = None,
    minimum_pack_z_nm: float | None = None,
    maximum_pack_z_factor: float | None = None,
    maximum_pack_z_nm: float | None = None,
    pressure_bar: float = 1.0,
) -> ResizedElectrolytePreparation:
    ref_box = tuple(float(x) for x in reference_box_nm)
    target_box = tuple(float(x) for x in target_box_nm)
    profile = build_bulk_equilibrium_profile(
        counts=probe_counts,
        mol_weights=mol_weights,
        species_names=species_names,
        work_dir=probe_work_dir,
    )
    resize_plan = plan_resized_electrolyte_counts(
        profile=profile,
        target_xy_nm=(float(target_box[0]), float(target_box[1])),
        target_z_nm=float(target_box[2]),
        solvent_indices=solvent_indices,
        solvent_groups=solvent_groups,
        salt_pair_indices=salt_pair_indices,
        salt_pair_groups=salt_pair_groups,
        min_solvent_counts=min_solvent_counts,
        min_solvent_group_counts=min_solvent_group_counts,
        min_salt_pairs=min_salt_pairs,
    )
    (
        recommended_pack_density,
        recommended_z_padding_factor,
        recommended_minimum_pack_z_factor,
        recommended_maximum_pack_z_factor,
    ) = recommend_fixed_xy_pack_parameters(target_density_g_cm3=float(resize_plan.target_density_g_cm3))
    pack_density = float(initial_pack_density_g_cm3) if initial_pack_density_g_cm3 is not None else recommended_pack_density
    z_padding = float(z_padding_factor) if z_padding_factor is not None else recommended_z_padding_factor
    min_pack_z_factor = float(minimum_pack_z_factor) if minimum_pack_z_factor is not None else recommended_minimum_pack_z_factor
    max_pack_z_factor = (
        float(maximum_pack_z_factor)
        if maximum_pack_z_factor is not None
        else recommended_maximum_pack_z_factor
    )
    min_pack_z_nm = max(
        float(target_box[2]) * float(min_pack_z_factor),
        float(minimum_pack_z_nm) if minimum_pack_z_nm is not None else 0.0,
    )
    max_pack_z_candidates = []
    if max_pack_z_factor is not None:
        max_pack_z_candidates.append(float(target_box[2]) * float(max_pack_z_factor))
    if maximum_pack_z_nm is not None:
        max_pack_z_candidates.append(float(maximum_pack_z_nm))
    max_pack_z_nm = max(max_pack_z_candidates) if max_pack_z_candidates else None
    pack_plan = plan_fixed_xy_direct_pack_box(
        reference_box_nm=(float(target_box[0]), float(target_box[1]), float(target_box[2])),
        target_counts=resize_plan.target_counts,
        mol_weights=mol_weights,
        species_names=species_names,
        initial_pack_density_g_cm3=pack_density,
        z_padding_factor=z_padding,
        minimum_z_nm=min_pack_z_nm,
        maximum_z_nm=max_pack_z_nm,
    )
    note_list = [
        "resized the electrolyte from the equilibrated probe bulk to the polymer-anchored target box instead of planning counts directly in the final fixed-XY box",
        f"probe_density_g_cm3={profile.density_g_cm3:.4f} from probe box {tuple(round(x, 4) for x in profile.box_nm)} -> target_box_nm={tuple(round(x, 4) for x in target_box)}",
        f"final fixed-XY pack uses initial_pack_density_g_cm3={pack_density:.4f} with minimum initial Z={min_pack_z_nm:.4f} nm",
    ]
    if initial_pack_density_g_cm3 is None or z_padding_factor is None or minimum_pack_z_factor is None or maximum_pack_z_factor is None:
        note_list.append(
            "auto-selected the fixed-XY initial pack density and Z guards from the probe-equilibrated target density to avoid rebuilding the final electrolyte from an excessively dilute box"
        )
    if max_pack_z_nm is not None:
        note_list.append(f"capped the initial fixed-XY pack box to a maximum Z of {max_pack_z_nm:.4f} nm before standalone electrolyte relaxation")
    if tuple(round(x, 6) for x in ref_box[:2]) != tuple(round(x, 6) for x in target_box[:2]):
        note_list.append(
            f"reference polymer XY={tuple(round(x, 4) for x in ref_box[:2])} while final electrolyte XY is locked to {tuple(round(x, 4) for x in target_box[:2])}"
        )
    return ResizedElectrolytePreparation(
        reference_box_nm=ref_box,
        target_box_nm=target_box,
        profile=profile,
        resize_plan=resize_plan,
        pack_plan=pack_plan,
        relax_mdp_overrides=fixed_xy_semiisotropic_npt_overrides(pressure_bar=float(pressure_bar)),
        notes=tuple(note_list),
    )


def make_orthorhombic_pack_cell(box_nm: tuple[float, float, float]):
    """Build a poly.amorphous_cell-compatible orthorhombic cell from nm input.

    High-level interface planning is expressed in nm, while the legacy
    ``poly.amorphous_cell`` packer uses Angstrom cell lengths and coordinates.
    Convert here so explicit ``cell=...`` packing follows the same internal
    unit convention as the density-driven path.
    """
    box_ang = tuple(float(x) * 10.0 for x in box_nm)
    cell = Chem.Mol()
    setattr(
        cell,
        "cell",
        utils.Cell(
            float(box_ang[0]),
            0.0,
            float(box_ang[1]),
            0.0,
            float(box_ang[2]),
            0.0,
        ),
    )
    return cell


def plan_fixed_xy_direct_electrolyte_preparation(
    *,
    reference_box_nm: tuple[float, float, float],
    target_box_nm: tuple[float, float, float] | None = None,
    target_density_g_cm3: float,
    solvent_mol_weights: Sequence[float],
    solvent_mass_ratio: Sequence[float],
    salt_mol_weights: Sequence[float],
    salt_molarity_M: float,
    min_salt_pairs: int,
    solvent_species_names: Sequence[str] | None = None,
    salt_species_names: Sequence[str] | None = None,
    min_solvent_counts: Sequence[int] | None = None,
    initial_pack_density_g_cm3: float | None = None,
    z_padding_factor: float | None = None,
    minimum_pack_z_factor: float | None = None,
    minimum_pack_z_nm: float | None = None,
    maximum_pack_z_factor: float | None = None,
    maximum_pack_z_nm: float | None = None,
    pressure_bar: float = 1.0,
) -> FixedXYElectrolytePreparation:
    ref_box = tuple(float(x) for x in reference_box_nm)
    target_box = tuple(float(x) for x in (target_box_nm or reference_box_nm))
    direct_plan = plan_direct_electrolyte_counts(
        target_box_nm=target_box,
        target_density_g_cm3=float(target_density_g_cm3),
        solvent_mol_weights=solvent_mol_weights,
        solvent_mass_ratio=solvent_mass_ratio,
        salt_mol_weights=salt_mol_weights,
        salt_molarity_M=float(salt_molarity_M),
        min_salt_pairs=int(min_salt_pairs),
        solvent_species_names=solvent_species_names,
        salt_species_names=salt_species_names,
        min_solvent_counts=min_solvent_counts,
    )
    (
        recommended_pack_density,
        recommended_z_padding_factor,
        recommended_minimum_pack_z_factor,
        recommended_maximum_pack_z_factor,
    ) = recommend_fixed_xy_pack_parameters(target_density_g_cm3=float(direct_plan.estimated_density_g_cm3))
    pack_density = float(initial_pack_density_g_cm3) if initial_pack_density_g_cm3 is not None else recommended_pack_density
    z_padding = float(z_padding_factor) if z_padding_factor is not None else recommended_z_padding_factor
    min_pack_z_factor = float(minimum_pack_z_factor) if minimum_pack_z_factor is not None else recommended_minimum_pack_z_factor
    if min_pack_z_factor < 1.0:
        raise ValueError("minimum_pack_z_factor must be >= 1.0")
    min_pack_z_nm = max(
        float(target_box[2]) * min_pack_z_factor,
        float(minimum_pack_z_nm) if minimum_pack_z_nm is not None else 0.0,
    )
    max_pack_z_candidates = []
    resolved_max_pack_z_factor = (
        float(maximum_pack_z_factor)
        if maximum_pack_z_factor is not None
        else recommended_maximum_pack_z_factor
    )
    if resolved_max_pack_z_factor is not None:
        max_pack_z_candidates.append(float(target_box[2]) * float(resolved_max_pack_z_factor))
    if maximum_pack_z_nm is not None:
        max_pack_z_candidates.append(float(maximum_pack_z_nm))
    max_pack_z_nm = max(max_pack_z_candidates) if max_pack_z_candidates else None
    pack_plan = plan_fixed_xy_direct_pack_box(
        reference_box_nm=(float(ref_box[0]), float(ref_box[1]), float(target_box[2])),
        target_counts=direct_plan.target_counts,
        mol_weights=tuple(float(x) for x in solvent_mol_weights) + tuple(float(x) for x in salt_mol_weights),
        species_names=direct_plan.species_names,
        initial_pack_density_g_cm3=pack_density,
        z_padding_factor=z_padding,
        minimum_z_nm=min_pack_z_nm,
        maximum_z_nm=max_pack_z_nm,
    )
    note_list = [
        "planned electrolyte counts against a compact polymer-anchored target box, then derived an XY-locked initial pack box for robust explicit-cell packing",
        "use pack_plan.initial_pack_box_nm for the first amorphous_cell build and keep relax_mdp_overrides through standalone electrolyte relaxation",
    ]
    if initial_pack_density_g_cm3 is None or z_padding_factor is None or minimum_pack_z_factor is None or maximum_pack_z_factor is None:
        note_list.append(
            "auto-selected the fixed-XY initial pack density and Z guards from the target electrolyte density so the standalone electrolyte bulk starts close enough to its expected compact box"
        )
    if min_pack_z_nm > float(target_box[2]) + 1.0e-12:
        note_list.append(
            f"forced the initial electrolyte pack box Z to be at least {min_pack_z_nm:.4f} nm before fixed-XY semiisotropic relaxation"
        )
    if max_pack_z_nm is not None:
        note_list.append(
            f"capped the initial electrolyte pack box Z to at most {max_pack_z_nm:.4f} nm to avoid an excessively dilute fixed-XY relaxation box"
        )
    notes = tuple(note_list)
    return FixedXYElectrolytePreparation(
        reference_box_nm=ref_box,
        direct_plan=direct_plan,
        pack_plan=pack_plan,
        relax_mdp_overrides=fixed_xy_semiisotropic_npt_overrides(pressure_bar=float(pressure_bar)),
        notes=notes,
    )


def plan_direct_polymer_matched_interface_preparation(
    *,
    reference_box_nm: tuple[float, float, float],
    bottom_thickness_nm: float,
    top_thickness_nm: float,
    gap_nm: float,
    surface_shell_nm: float,
    target_density_g_cm3: float,
    solvent_mol_weights: Sequence[float],
    solvent_mass_ratio: Sequence[float],
    salt_mol_weights: Sequence[float],
    salt_molarity_M: float,
    min_salt_pairs: int,
    solvent_species_names: Sequence[str] | None = None,
    salt_species_names: Sequence[str] | None = None,
    min_solvent_counts: Sequence[int] | None = None,
    initial_pack_density_g_cm3: float | None = None,
    z_padding_factor: float | None = None,
    minimum_pack_z_factor: float | None = None,
    minimum_pack_z_nm: float | None = None,
    maximum_pack_z_factor: float | None = None,
    maximum_pack_z_nm: float | None = None,
    pressure_bar: float = 1.0,
    is_polyelectrolyte: bool = False,
    minimum_margin_nm: float = 1.0,
    fixed_xy_npt_ns: float | None = None,
    polymer_margin_nm: float | None = None,
) -> DirectPolymerMatchedInterfacePreparation:
    ref_box = tuple(float(x) for x in reference_box_nm)
    interface_plan = plan_polymer_anchored_interface_preparation(
        reference_box_nm=ref_box,
        bottom_thickness_nm=float(bottom_thickness_nm),
        top_thickness_nm=float(top_thickness_nm),
        gap_nm=float(gap_nm),
        surface_shell_nm=float(surface_shell_nm),
        is_polyelectrolyte=bool(is_polyelectrolyte),
        minimum_margin_nm=float(minimum_margin_nm),
        fixed_xy_npt_ns=fixed_xy_npt_ns,
        polymer_margin_nm=polymer_margin_nm,
    )
    electrolyte_prep = plan_fixed_xy_direct_electrolyte_preparation(
        reference_box_nm=ref_box,
        target_box_nm=interface_plan.electrolyte_target_box_nm,
        target_density_g_cm3=float(target_density_g_cm3),
        solvent_mol_weights=solvent_mol_weights,
        solvent_mass_ratio=solvent_mass_ratio,
        salt_mol_weights=salt_mol_weights,
        salt_molarity_M=float(salt_molarity_M),
        min_salt_pairs=int(min_salt_pairs),
        solvent_species_names=solvent_species_names,
        salt_species_names=salt_species_names,
        min_solvent_counts=min_solvent_counts,
        initial_pack_density_g_cm3=initial_pack_density_g_cm3,
        z_padding_factor=z_padding_factor,
        minimum_pack_z_factor=minimum_pack_z_factor,
        minimum_pack_z_nm=minimum_pack_z_nm,
        maximum_pack_z_factor=maximum_pack_z_factor,
        maximum_pack_z_nm=maximum_pack_z_nm,
        pressure_bar=float(pressure_bar),
    )
    notes = (
        "equilibrate the polymer bulk first, then lock the electrolyte XY box to the equilibrated polymer footprint",
        "build the standalone electrolyte bulk against the compact top-side target box, relax it separately to density equilibrium, and only then assemble the interface with an explicit vacuum gap",
    ) + tuple(interface_plan.notes) + tuple(electrolyte_prep.notes)
    return DirectPolymerMatchedInterfacePreparation(
        reference_box_nm=ref_box,
        interface_plan=interface_plan,
        electrolyte_prep=electrolyte_prep,
        notes=notes,
    )


def plan_polymer_anchored_interface_preparation(
    *,
    reference_box_nm: tuple[float, float, float],
    bottom_thickness_nm: float,
    top_thickness_nm: float,
    gap_nm: float,
    surface_shell_nm: float,
    is_polyelectrolyte: bool = False,
    minimum_margin_nm: float = 1.0,
    fixed_xy_npt_ns: float | None = None,
    polymer_margin_nm: float | None = None,
) -> PolymerAnchoredInterfacePreparation:
    ref_box = tuple(float(x) for x in reference_box_nm)
    if len(ref_box) != 3 or min(ref_box) <= 0.0:
        raise ValueError("reference_box_nm must contain three positive lengths")
    bottom_thickness = float(bottom_thickness_nm)
    if bottom_thickness <= 0.0:
        raise ValueError("bottom_thickness_nm must be positive")

    alignment = recommend_electrolyte_alignment(
        top_thickness_nm=float(top_thickness_nm),
        gap_nm=float(gap_nm),
        surface_shell_nm=float(surface_shell_nm),
        is_polyelectrolyte=bool(is_polyelectrolyte),
        minimum_margin_nm=float(minimum_margin_nm),
        fixed_xy_npt_ns=fixed_xy_npt_ns,
    )
    common_xy = (float(ref_box[0]), float(ref_box[1]))
    polymer_margin = float(alignment.target_z_margin_nm if polymer_margin_nm is None else polymer_margin_nm)
    polymer_target_box = (common_xy[0], common_xy[1], float(bottom_thickness + polymer_margin))
    electrolyte_target_box = (common_xy[0], common_xy[1], float(alignment.target_z_nm))
    notes = (
        "use the equilibrated polymer bulk XY lengths as the interface reference footprint",
        "trim the polymer side by slab thickness during interface preparation and plan the electrolyte counts only against the compact top-side target box",
        "keep the electrolyte XY locked to the polymer footprint during packing and relaxation so both sides converge to one shared lateral reference",
    )
    return PolymerAnchoredInterfacePreparation(
        reference_box_nm=ref_box,
        interface_xy_nm=common_xy,
        bottom_thickness_nm=float(bottom_thickness),
        top_thickness_nm=float(top_thickness_nm),
        gap_nm=float(gap_nm),
        surface_shell_nm=float(surface_shell_nm),
        polymer_margin_nm=float(polymer_margin),
        is_polyelectrolyte=bool(is_polyelectrolyte),
        polymer_target_box_nm=polymer_target_box,
        electrolyte_target_box_nm=electrolyte_target_box,
        electrolyte_alignment=alignment,
        notes=notes,
    )


def plan_probe_polymer_matched_interface_preparation(
    *,
    reference_box_nm: tuple[float, float, float],
    bottom_thickness_nm: float,
    top_thickness_nm: float,
    gap_nm: float,
    surface_shell_nm: float,
    target_density_g_cm3: float,
    solvent_mol_weights: Sequence[float],
    solvent_mass_ratio: Sequence[float],
    salt_mol_weights: Sequence[float],
    salt_molarity_M: float,
    min_salt_pairs: int,
    solvent_species_names: Sequence[str] | None = None,
    salt_species_names: Sequence[str] | None = None,
    min_solvent_counts: Sequence[int] | None = None,
    probe_volume_scale: float = 1.75,
    minimum_probe_box_nm: float | None = None,
    initial_pack_density_g_cm3: float = 0.70,
    z_padding_factor: float = 1.10,
    is_polyelectrolyte: bool = False,
    minimum_margin_nm: float = 1.0,
    fixed_xy_npt_ns: float | None = None,
    polymer_margin_nm: float | None = None,
) -> ProbePolymerMatchedInterfacePreparation:
    ref_box = tuple(float(x) for x in reference_box_nm)
    interface_plan = plan_polymer_anchored_interface_preparation(
        reference_box_nm=ref_box,
        bottom_thickness_nm=float(bottom_thickness_nm),
        top_thickness_nm=float(top_thickness_nm),
        gap_nm=float(gap_nm),
        surface_shell_nm=float(surface_shell_nm),
        is_polyelectrolyte=bool(is_polyelectrolyte),
        minimum_margin_nm=float(minimum_margin_nm),
        fixed_xy_npt_ns=fixed_xy_npt_ns,
        polymer_margin_nm=polymer_margin_nm,
    )
    probe_prep = plan_probe_electrolyte_preparation(
        reference_box_nm=ref_box,
        target_box_nm=interface_plan.electrolyte_target_box_nm,
        target_density_g_cm3=float(target_density_g_cm3),
        solvent_mol_weights=solvent_mol_weights,
        solvent_mass_ratio=solvent_mass_ratio,
        salt_mol_weights=salt_mol_weights,
        salt_molarity_M=float(salt_molarity_M),
        min_salt_pairs=int(min_salt_pairs),
        solvent_species_names=solvent_species_names,
        salt_species_names=salt_species_names,
        min_solvent_counts=min_solvent_counts,
        probe_volume_scale=float(probe_volume_scale),
        minimum_probe_box_nm=minimum_probe_box_nm,
        initial_pack_density_g_cm3=float(initial_pack_density_g_cm3),
        z_padding_factor=float(z_padding_factor),
    )
    notes = (
        "equilibrate the polymer bulk first, then build an isotropic electrolyte probe bulk before locking the final XY footprint",
        "use the probe bulk only to learn a physically reasonable composition and density response before rebuilding the final electrolyte bulk against the polymer-matched interface box",
    ) + tuple(interface_plan.notes) + tuple(probe_prep.notes)
    return ProbePolymerMatchedInterfacePreparation(
        reference_box_nm=ref_box,
        interface_plan=interface_plan,
        probe_prep=probe_prep,
        notes=notes,
    )


def plan_resized_polymer_matched_interface_from_probe(
    *,
    reference_box_nm: tuple[float, float, float],
    interface_plan: PolymerAnchoredInterfacePreparation,
    probe_work_dir,
    probe_counts: Sequence[int],
    mol_weights: Sequence[float],
    species_names: Sequence[str],
    solvent_indices: Sequence[int],
    solvent_groups: Sequence[Sequence[int]] | None = None,
    salt_pair_indices: Sequence[int] | None = None,
    salt_pair_groups: Sequence[Sequence[int]] | None = None,
    min_solvent_counts: Sequence[int] | None = None,
    min_solvent_group_counts: Sequence[Sequence[int]] | None = None,
    min_salt_pairs: int | Sequence[int] = 1,
    initial_pack_density_g_cm3: float | None = None,
    z_padding_factor: float | None = None,
    minimum_pack_z_factor: float | None = None,
    minimum_pack_z_nm: float | None = None,
    maximum_pack_z_factor: float | None = 2.5,
    maximum_pack_z_nm: float | None = None,
    pressure_bar: float = 1.0,
) -> ResizedPolymerMatchedInterfacePreparation:
    ref_box = tuple(float(x) for x in reference_box_nm)
    resized_prep = plan_resized_electrolyte_preparation_from_probe(
        reference_box_nm=ref_box,
        target_box_nm=interface_plan.electrolyte_target_box_nm,
        probe_work_dir=probe_work_dir,
        probe_counts=probe_counts,
        mol_weights=mol_weights,
        species_names=species_names,
        solvent_indices=solvent_indices,
        solvent_groups=solvent_groups,
        salt_pair_indices=salt_pair_indices,
        salt_pair_groups=salt_pair_groups,
        min_solvent_counts=min_solvent_counts,
        min_solvent_group_counts=min_solvent_group_counts,
        min_salt_pairs=min_salt_pairs,
        initial_pack_density_g_cm3=initial_pack_density_g_cm3,
        z_padding_factor=z_padding_factor,
        minimum_pack_z_factor=minimum_pack_z_factor,
        minimum_pack_z_nm=minimum_pack_z_nm,
        maximum_pack_z_factor=maximum_pack_z_factor,
        maximum_pack_z_nm=maximum_pack_z_nm,
        pressure_bar=float(pressure_bar),
    )
    notes = (
        "resize the final electrolyte composition only after the probe bulk has equilibrated, then rebuild the final electrolyte with polymer-matched XY and extra initial Z slack",
    ) + tuple(interface_plan.notes) + tuple(resized_prep.notes)
    return ResizedPolymerMatchedInterfacePreparation(
        reference_box_nm=ref_box,
        interface_plan=interface_plan,
        resized_prep=resized_prep,
        notes=notes,
    )


def equilibrate_bulk_with_eq21(
    *,
    label: str,
    ac,
    work_dir,
    temp: float,
    press: float,
    mpi: int,
    omp: int,
    gpu: int,
    gpu_id: int,
    additional_loops: int = 4,
    eq21_npt_mdp_overrides=None,
    additional_mdp_overrides=None,
    final_npt_ns: float = 0.0,
    final_npt_mdp_overrides=None,
    eq21_exec_kwargs: dict[str, Any] | None = None,
) -> BulkEq21Outcome:
    eqmd_job = eq.EQ21step(ac, work_dir=work_dir)
    export = eqmd_job.ensure_system_exported()
    export_issues = validate_exported_system_dir(export.system_top.parent)
    raw_export_root = export.system_top.parent / "01_raw_non_scaled"
    raw_export_issues = validate_exported_system_dir(raw_export_root)
    if export_issues:
        raise RuntimeError(f"{label} scaled export topology is invalid before EQ21: {'; '.join(export_issues)}")
    if raw_export_issues:
        raise RuntimeError(f"{label} raw export topology is invalid before EQ21: {'; '.join(raw_export_issues)}")

    exec_kwargs = dict(eq21_exec_kwargs or {})
    ac = eqmd_job.exec(
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        eq21_npt_mdp_overrides=eq21_npt_mdp_overrides,
        **exec_kwargs,
    )
    analy = eqmd_job.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    result = analy.check_eq()
    for _ in range(int(additional_loops)):
        if result:
            break
        add_job = eq.Additional(ac, work_dir=work_dir)
        ac = add_job.exec(
            temp=temp,
            press=press,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            mdp_overrides=additional_mdp_overrides,
        )
        analy = add_job.analyze()
        _ = analy.get_all_prop(temp=temp, press=press, save=True)
        result = analy.check_eq()

    if float(final_npt_ns) > 0.0:
        npt_job = eq.NPT(ac, work_dir=work_dir)
        ac = npt_job.exec(
            temp=temp,
            press=press,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            time=float(final_npt_ns),
            mdp_overrides=final_npt_mdp_overrides,
        )
        try:
            analy = npt_job.analyze()
            _ = analy.get_all_prop(temp=temp, press=press, save=True)
        except Exception:
            pass

    return BulkEq21Outcome(
        final_cell=ac,
        system_export=export,
        raw_system_meta=raw_export_root / "system_meta.json",
    )


def recommend_polymer_diffusion_interface_recipe(
    *,
    interface_plan: PolymerAnchoredInterfacePreparation,
    temperature_k: float = 300.0,
    pressure_bar: float = 1.0,
    axis: str = "Z",
    prefer_vacuum: bool | None = None,
    vacuum_nm: float | None = None,
    area_policy: "AreaMismatchPolicy | None" = None,
    max_lateral_strain: float = 0.08,
    core_guard_nm: float = 0.50,
    top_lateral_shift_fraction: tuple[float, float] = (0.35, 0.65),
    wall_mode: str = "12-6",
    wall_atomtype: str | None = None,
    wall_density_nm3: float | None = None,
    pre_contact_ps: float | None = None,
    pre_contact_dt_ps: float = 0.001,
    density_relax_ps: float | None = None,
    contact_ps: float | None = None,
    release_ps: float | None = None,
    exchange_ns: float | None = None,
    production_ns: float | None = None,
    freeze_cores_pre_contact: bool = True,
    use_region_thermostat_early: bool = True,
) -> PolymerDiffusionInterfaceRecipe:
    from .builder import AreaMismatchPolicy, InterfaceRouteSpec
    from .protocol import InterfaceProtocol

    plan = interface_plan
    route_axis = str(axis or "Z").strip().upper()
    use_vacuum = bool(plan.is_polyelectrolyte) if prefer_vacuum is None else bool(prefer_vacuum)
    if area_policy is None:
        area_policy = AreaMismatchPolicy(reference_side="bottom", max_lateral_strain=float(max_lateral_strain))

    default_pre_contact_ps = 160.0 if plan.is_polyelectrolyte else 120.0
    default_density_relax_ps = 450.0 if plan.is_polyelectrolyte else 250.0
    default_contact_ps = 450.0 if plan.is_polyelectrolyte else 250.0
    default_release_ps = 450.0 if plan.is_polyelectrolyte else 250.0
    default_exchange_ns = 4.0 if plan.is_polyelectrolyte else 2.0
    default_production_ns = 8.0 if plan.is_polyelectrolyte else 5.0

    if use_vacuum:
        chosen_vacuum_nm = float(
            vacuum_nm
            if vacuum_nm is not None
            else max(
                10.0,
                plan.top_thickness_nm + plan.gap_nm + plan.surface_shell_nm + 2.0,
            )
        )
        route_spec = InterfaceRouteSpec.route_b(
            axis=route_axis,
            gap_nm=float(plan.gap_nm),
            vacuum_nm=chosen_vacuum_nm,
            bottom_thickness_nm=float(plan.bottom_thickness_nm),
            top_thickness_nm=float(plan.top_thickness_nm),
            surface_shell_nm=float(plan.surface_shell_nm),
            core_guard_nm=float(core_guard_nm),
            top_lateral_shift_fraction=top_lateral_shift_fraction,
            area_policy=area_policy,
        )
        protocol = InterfaceProtocol.route_b_wall_diffusion(
            axis=route_axis,
            temperature_k=float(temperature_k),
            pressure_bar=float(pressure_bar),
            pre_contact_ps=float(default_pre_contact_ps if pre_contact_ps is None else pre_contact_ps),
            pre_contact_dt_ps=float(pre_contact_dt_ps),
            density_relax_ps=float(default_density_relax_ps if density_relax_ps is None else density_relax_ps),
            contact_ps=float(default_contact_ps if contact_ps is None else contact_ps),
            release_ps=float(default_release_ps if release_ps is None else release_ps),
            exchange_ns=float(default_exchange_ns if exchange_ns is None else exchange_ns),
            production_ns=float(default_production_ns if production_ns is None else production_ns),
            wall_mode=wall_mode,
            wall_atomtype=wall_atomtype,
            wall_density_nm3=wall_density_nm3,
            freeze_cores_pre_contact=bool(freeze_cores_pre_contact),
            use_region_thermostat_early=bool(use_region_thermostat_early),
        )
        notes = (
            "selected route_b so the assembled interface keeps an explicit gap plus an external vacuum buffer under pbc=xy",
            "used the staged wall-backed diffusion protocol to let each phase relax density before unrestricted interdiffusion begins",
        )
    else:
        route_spec = InterfaceRouteSpec.route_a(
            axis=route_axis,
            gap_nm=float(plan.gap_nm),
            bottom_thickness_nm=float(plan.bottom_thickness_nm),
            top_thickness_nm=float(plan.top_thickness_nm),
            surface_shell_nm=float(plan.surface_shell_nm),
            core_guard_nm=float(core_guard_nm),
            top_lateral_shift_fraction=top_lateral_shift_fraction,
            area_policy=area_policy,
        )
        protocol = InterfaceProtocol.route_a_diffusion(
            axis=route_axis,
            temperature_k=float(temperature_k),
            pressure_bar=float(pressure_bar),
            pre_contact_ps=float(default_pre_contact_ps if pre_contact_ps is None else pre_contact_ps),
            pre_contact_dt_ps=float(pre_contact_dt_ps),
            density_relax_ps=float(default_density_relax_ps if density_relax_ps is None else density_relax_ps),
            contact_ps=float(default_contact_ps if contact_ps is None else contact_ps),
            release_ps=float(default_release_ps if release_ps is None else release_ps),
            exchange_ns=float(default_exchange_ns if exchange_ns is None else exchange_ns),
            production_ns=float(default_production_ns if production_ns is None else production_ns),
            freeze_cores_pre_contact=bool(freeze_cores_pre_contact),
            use_region_thermostat_early=bool(use_region_thermostat_early),
        )
        notes = (
            "selected route_a for a fully periodic diffusion interface while still preserving a staged initial gap-hold and density-relax path",
            "used an asymmetric top lateral phase shift to avoid face-to-face registry between independently equilibrated slabs",
        )

    return PolymerDiffusionInterfaceRecipe(
        interface_plan=plan,
        route_spec=route_spec,
        protocol=protocol,
        notes=notes + tuple(plan.notes) + tuple(plan.electrolyte_alignment.notes),
    )


__all__ = [
    "BulkEq21Outcome",
    "DirectPolymerMatchedInterfacePreparation",
    "FixedXYElectrolytePreparation",
    "PolymerDiffusionInterfaceRecipe",
    "PolymerAnchoredInterfacePreparation",
    "ProbePolymerMatchedInterfacePreparation",
    "ProbeElectrolytePreparation",
    "ResizedPolymerMatchedInterfacePreparation",
    "ResizedElectrolytePreparation",
    "equilibrate_bulk_with_eq21",
    "make_orthorhombic_pack_cell",
    "plan_direct_polymer_matched_interface_preparation",
    "plan_fixed_xy_direct_electrolyte_preparation",
    "plan_probe_polymer_matched_interface_preparation",
    "plan_probe_electrolyte_preparation",
    "plan_polymer_anchored_interface_preparation",
    "plan_resized_polymer_matched_interface_from_probe",
    "plan_resized_electrolyte_preparation_from_probe",
    "recommend_polymer_diffusion_interface_recipe",
]
