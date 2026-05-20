"""CMC-Na z-open slab preparation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem

from ..core import poly
from ..sim.preset import eq
from ..sim.preset.eq import XYSlabEquilibrationSpec
from .prep import make_orthorhombic_pack_cell

_AVOGADRO = 6.02214076e23


@dataclass(frozen=True)
class CMCNAXYSlabRelaxationSpec:
    """Defaults for a reusable z-open CMC-Na slab prepared at fixed XY."""

    initial_density_g_cm3: float = 0.05
    density_mode: str = "wall_gap_compression"
    coordinate_export_policy: str = "wrapped_xy_z_open"
    target_density_g_cm3: float | None = 0.80
    target_active_z_nm: float | None = None
    target_box_z_nm: float | None = None
    active_density_min_g_cm3: float | None = 0.80
    wall_padding_nm: float = 0.40
    cycles: int | str = "auto"
    max_cycles: int = 40
    max_z_shrink_per_cycle: float = 0.06
    tmax_K: float = 450.0
    pmax_bar: float = 2000.0
    tau_p_ps: float = 5.0
    z_compressibility_bar_inv: float = 4.5e-5
    pre_nvt_ns: float = 0.01
    wall_npt_ns: float = 0.05
    hot_nvt_ns: float = 0.01
    cool_nvt_ns: float = 0.01
    final_relax_ns: float = 0.50
    max_convergence_rounds: int = 8
    extra_relax_ns_per_round: float = 0.50
    active_density_convergence: bool = True
    rg_convergence: bool = True
    lateral_occupancy_convergence: bool = True
    active_density_tolerance_fraction: float = 0.08
    active_density_rel_std_max: float = 0.08
    lateral_occupancy_grid_nm: float = 0.50
    min_lateral_occupancy_fraction: float = 0.85
    min_edge_occupancy_fraction: float = 0.80
    na_coo_contact_cutoff_nm: float = 0.35
    na_coo_contact_min_fraction: float = 0.75
    write_compression_animation: bool = True
    animation_fps: float = 1.0

    def to_xy_slab_spec(self) -> XYSlabEquilibrationSpec:
        return XYSlabEquilibrationSpec(
            density_mode=str(self.density_mode),  # type: ignore[arg-type]
            coordinate_export_policy=str(self.coordinate_export_policy),  # type: ignore[arg-type]
            target_density_g_cm3=(None if self.target_density_g_cm3 is None else float(self.target_density_g_cm3)),
            target_active_z_nm=(None if self.target_active_z_nm is None else float(self.target_active_z_nm)),
            target_box_z_nm=(None if self.target_box_z_nm is None else float(self.target_box_z_nm)),
            active_density_min_g_cm3=(None if self.active_density_min_g_cm3 is None else float(self.active_density_min_g_cm3)),
            cycles=self.cycles,  # type: ignore[arg-type]
            max_cycles=int(self.max_cycles),
            max_z_shrink_per_cycle=float(self.max_z_shrink_per_cycle),
            wall_padding_nm=float(self.wall_padding_nm),
            xy_area_mode="fixed",
            pressure_axis_mode="fixed_xy_z_npt",
            tau_p_ps=float(self.tau_p_ps),
            z_compressibility_bar_inv=float(self.z_compressibility_bar_inv),
            pmax_bar=float(self.pmax_bar),
            pre_nvt_ns=float(self.pre_nvt_ns),
            wall_npt_ns=float(self.wall_npt_ns),
            hot_nvt_ns=float(self.hot_nvt_ns),
            cool_nvt_ns=float(self.cool_nvt_ns),
            final_relax_ns=float(self.final_relax_ns),
            active_density_convergence=bool(self.active_density_convergence),
            rg_convergence=bool(self.rg_convergence),
            lateral_occupancy_convergence=bool(self.lateral_occupancy_convergence),
            max_convergence_rounds=int(self.max_convergence_rounds),
            extra_relax_ns_per_round=float(self.extra_relax_ns_per_round),
            active_density_tolerance_fraction=float(self.active_density_tolerance_fraction),
            active_density_rel_std_max=float(self.active_density_rel_std_max),
            lateral_occupancy_grid_nm=float(self.lateral_occupancy_grid_nm),
            min_lateral_occupancy_fraction=float(self.min_lateral_occupancy_fraction),
            min_edge_occupancy_fraction=float(self.min_edge_occupancy_fraction),
            na_coo_contact_cutoff_nm=float(self.na_coo_contact_cutoff_nm),
            na_coo_contact_min_fraction=float(self.na_coo_contact_min_fraction),
            write_compression_animation=bool(self.write_compression_animation),
            animation_fps=float(self.animation_fps),
        )


@dataclass(frozen=True)
class CMCNAXYBulkSlabResult:
    """Artifacts from :func:`prepare_cmcna_xy_bulk_slab`."""

    work_dir: Path
    prepared_slab_gro: Path
    prepared_slab_whole_gro: Path
    prepared_slab_top: Path
    xy_slab_summary: Path
    convergence_summary: Path
    coordinate_summary: Path
    ready_for_layer_stack: bool
    target_density_g_cm3: float | None
    xy_nm: tuple[float, float]
    initial_z_nm: float
    summary: dict[str, Any]


def _mol_mass_amu(mol: Chem.Mol) -> float:
    return float(sum(float(atom.GetMass()) for atom in mol.GetAtoms()))


def prepare_cmcna_xy_bulk_slab(
    *,
    cmc_chain_mol: Chem.Mol,
    na_mol: Chem.Mol,
    chain_count: int,
    dp: int,
    xy_nm: tuple[float, float],
    work_dir: str | Path,
    temp: float = 318.15,
    pressure_bar: float = 1.0,
    mpi: int = 1,
    omp: int = 14,
    gpu: int = 1,
    gpu_id: int | None = 0,
    charge_scale: float | tuple[float, float] = 1.0,
    relaxation: CMCNAXYSlabRelaxationSpec | None = None,
    retry: int = 30,
    retry_step: int = 2000,
    threshold_ang: float = 2.0,
    large_system_mode: str = "large",
    restart: bool | None = None,
) -> CMCNAXYBulkSlabResult:
    """Prepare a fixed-XY, z-open CMC-Na slab with wall-gap compression gates."""

    spec = relaxation if relaxation is not None else CMCNAXYSlabRelaxationSpec()
    wd = Path(work_dir).expanduser().resolve()
    wd.mkdir(parents=True, exist_ok=True)
    xy = (float(xy_nm[0]), float(xy_nm[1]))
    if xy[0] <= 0.0 or xy[1] <= 0.0:
        raise ValueError("xy_nm must contain positive x/y lengths in nm.")
    chains = int(chain_count)
    degree = int(dp)
    if chains <= 0 or degree <= 0:
        raise ValueError("chain_count and dp must be positive integers.")
    na_count = chains * degree
    total_mass_amu = _mol_mass_amu(cmc_chain_mol) * chains + _mol_mass_amu(na_mol) * na_count
    initial_density = max(float(spec.initial_density_g_cm3), 1.0e-6)
    volume_nm3 = (total_mass_amu / _AVOGADRO) / initial_density * 1.0e21
    initial_z_nm = max(float(volume_nm3) / max(xy[0] * xy[1], 1.0e-9), 0.10)
    cell = make_orthorhombic_pack_cell((xy[0], xy[1], initial_z_nm))
    if isinstance(charge_scale, tuple):
        scales = charge_scale
    else:
        scales = (float(charge_scale), float(charge_scale))
    ac = poly.amorphous_cell(
        [cmc_chain_mol, na_mol],
        [chains, na_count],
        cell=cell,
        density=None,
        retry=int(retry),
        retry_step=int(retry_step),
        threshold=float(threshold_ang),
        charge_scale=scales,
        polyelectrolyte_mode=True,
        large_system_mode=str(large_system_mode),
        work_dir=wd / "00_sparse_ac",
        restart=restart,
    )
    job = eq.EQ21step(ac, work_dir=wd)
    job.exec(
        temp=float(temp),
        press=float(pressure_bar),
        mpi=int(mpi),
        omp=int(omp),
        gpu=int(gpu),
        gpu_id=gpu_id,
        sim_time=float(spec.final_relax_ns),
        eq21_tmax=float(spec.tmax_K),
        eq21_dt_ps=0.001,
        periodicity="xy",
        xy_slab=spec.to_xy_slab_spec(),
        restart=restart,
    )
    run_dir = wd / "03_EQ21_XY_SLAB"
    summary_path = run_dir / "xy_slab_summary.json"
    convergence_path = run_dir / "cmcna_slab_convergence.json"
    prepared_gro = run_dir / "prepared_slab.gro"
    prepared_whole_gro = run_dir / "prepared_slab_whole.gro"
    prepared_top = run_dir / "prepared_slab.top"
    coordinate_path = run_dir / "prepared_slab_coordinate_report.json"
    summary: dict[str, Any] = {}
    if convergence_path.is_file():
        try:
            summary = json.loads(convergence_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    ready = bool(summary.get("ready_for_layer_stack")) if summary else False
    return CMCNAXYBulkSlabResult(
        work_dir=wd,
        prepared_slab_gro=prepared_gro,
        prepared_slab_whole_gro=prepared_whole_gro,
        prepared_slab_top=prepared_top,
        xy_slab_summary=summary_path,
        convergence_summary=convergence_path,
        coordinate_summary=coordinate_path,
        ready_for_layer_stack=ready,
        target_density_g_cm3=(None if spec.target_density_g_cm3 is None else float(spec.target_density_g_cm3)),
        xy_nm=xy,
        initial_z_nm=float(initial_z_nm),
        summary=summary,
    )


def prepare_cmcna_xy_membrane(**kwargs: Any) -> CMCNAXYBulkSlabResult:
    """Prepare a stack-ready CMC-Na membrane with XY periodicity and z walls.

    This is the membrane-focused public alias for
    :func:`prepare_cmcna_xy_bulk_slab`.  Its default relaxation spec uses explicit
    wall-gap/box-z compression plus wall-confined NVT relaxation; the older
    wall-z-NPT route remains available only when users request it explicitly in
    ``relaxation.density_mode``.
    """

    return prepare_cmcna_xy_bulk_slab(**kwargs)


__all__ = [
    "CMCNAXYBulkSlabResult",
    "CMCNAXYSlabRelaxationSpec",
    "prepare_cmcna_xy_membrane",
    "prepare_cmcna_xy_bulk_slab",
]
