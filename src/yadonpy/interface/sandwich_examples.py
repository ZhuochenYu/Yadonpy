from __future__ import annotations

from .sandwich import (
    build_graphite_cmcna_electrolyte_sandwich,
    build_graphite_peo_electrolyte_sandwich,
)
from .sandwich_specs import (
    ElectrolyteSlabSpec,
    GraphitePolymerElectrolyteSandwichResult,
    GraphiteSubstrateSpec,
    MoleculeSpec,
    PolymerSlabSpec,
    SandwichRelaxationSpec,
    default_carbonate_lipf6_electrolyte_spec,
    default_cmcna_polymer_spec,
    default_peo_polymer_spec,
)


def build_graphite_peo_example_case(
    *,
    work_dir,
    ff,
    ion_ff,
    profile: str = "smoke",
    graphite: GraphiteSubstrateSpec | None = None,
    polymer: PolymerSlabSpec | None = None,
    electrolyte: ElectrolyteSlabSpec | None = None,
    relax: SandwichRelaxationSpec | None = None,
    restart: bool | None = None,
) -> GraphitePolymerElectrolyteSandwichResult:
    smoke = str(profile).strip().lower() == "smoke"
    default_graphite = GraphiteSubstrateSpec(
        nx=(4 if smoke else 6),
        ny=(4 if smoke else 5),
        n_layers=(2 if smoke else 3),
        edge_cap="H",
        name="GRAPH",
    )
    default_polymer = default_peo_polymer_spec(
        name="PEO",
        chain_target_atoms=(220 if smoke else 320),
        min_chain_count=(4 if smoke else 3),
        target_density_g_cm3=1.08,
        slab_z_nm=(3.2 if smoke else 4.0),
        initial_pack_z_scale=(1.18 if smoke else 1.20),
        pack_retry=(30 if smoke else 48),
        pack_retry_step=(2000 if smoke else 2600),
    )
    default_electrolyte = default_carbonate_lipf6_electrolyte_spec(
        slab_z_nm=(3.8 if smoke else 4.8),
        min_salt_pairs=(2 if smoke else 4),
        target_density_g_cm3=(1.28 if smoke else 1.30),
        initial_pack_density_g_cm3=(0.82 if smoke else 0.86),
        pack_retry=(30 if smoke else 44),
        pack_retry_step=(1800 if smoke else 2400),
    )
    default_relax = SandwichRelaxationSpec(
        omp=16,
        gpu=1,
        gpu_id=0,
        psi4_omp=(36 if smoke else 24),
        psi4_memory_mb=24000,
        bulk_eq21_final_ns=(0.0 if smoke else 0.10),
        bulk_additional_loops=(0 if smoke else 1),
        bulk_eq21_exec_kwargs=(
            {
                "eq21_tmax": 600.0,
                "eq21_pmax": 10000.0,
                "eq21_pre_nvt_ps": 2.0,
                "sim_time": 0.04,
                "eq21_npt_time_scale": 0.20,
            }
            if smoke
            else {}
        ),
        stacked_pre_nvt_ps=(10.0 if smoke else 20.0),
        stacked_z_relax_ps=(40.0 if smoke else 80.0),
        stacked_exchange_ps=(60.0 if smoke else 120.0),
    )
    return build_graphite_peo_electrolyte_sandwich(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=(graphite if graphite is not None else default_graphite),
        polymer=(polymer if polymer is not None else default_polymer),
        electrolyte=(electrolyte if electrolyte is not None else default_electrolyte),
        relax=(relax if relax is not None else default_relax),
        restart=restart,
    )


def build_graphite_cmcna_example_case(
    *,
    work_dir,
    ff,
    ion_ff,
    profile: str = "smoke",
    graphite: GraphiteSubstrateSpec | None = None,
    polymer: PolymerSlabSpec | None = None,
    electrolyte: ElectrolyteSlabSpec | None = None,
    relax: SandwichRelaxationSpec | None = None,
    restart: bool | None = None,
) -> GraphitePolymerElectrolyteSandwichResult:
    smoke = str(profile).strip().lower() == "smoke"
    default_graphite = GraphiteSubstrateSpec(
        nx=(4 if smoke else 6),
        ny=(4 if smoke else 5),
        n_layers=(2 if smoke else 3),
        edge_cap="H",
        name="GRAPH",
    )
    default_polymer = default_cmcna_polymer_spec(
        dp=(24 if smoke else 60),
        target_density_g_cm3=1.50,
        slab_z_nm=(3.8 if smoke else 4.6),
        min_chain_count=2,
        initial_pack_z_scale=(1.30 if smoke else 1.34),
        pack_retry=(36 if smoke else 60),
        pack_retry_step=(2200 if smoke else 3200),
        pack_threshold_ang=1.60,
        pack_dec_rate=(0.60 if smoke else 0.68),
    )
    default_electrolyte = default_carbonate_lipf6_electrolyte_spec(
        slab_z_nm=(4.0 if smoke else 5.2),
        min_salt_pairs=(2 if smoke else 4),
        target_density_g_cm3=(1.28 if smoke else 1.30),
        initial_pack_density_g_cm3=(0.82 if smoke else 0.86),
        pack_retry=(34 if smoke else 44),
        pack_retry_step=(1800 if smoke else 2400),
    )
    default_relax = SandwichRelaxationSpec(
        omp=16,
        gpu=1,
        gpu_id=0,
        psi4_omp=(36 if smoke else 24),
        psi4_memory_mb=24000,
        bulk_eq21_final_ns=(0.0 if smoke else 0.10),
        bulk_additional_loops=(0 if smoke else 1),
        bulk_eq21_exec_kwargs=(
            {
                "eq21_tmax": 600.0,
                "eq21_pmax": 10000.0,
                "eq21_pre_nvt_ps": 2.0,
                "sim_time": 0.04,
                "eq21_npt_time_scale": 0.20,
            }
            if smoke
            else {}
        ),
        stacked_pre_nvt_ps=(10.0 if smoke else 20.0),
        stacked_z_relax_ps=(40.0 if smoke else 80.0),
        stacked_exchange_ps=(60.0 if smoke else 120.0),
    )
    return build_graphite_cmcna_electrolyte_sandwich(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=(graphite if graphite is not None else default_graphite),
        polymer=(polymer if polymer is not None else default_polymer),
        electrolyte=(electrolyte if electrolyte is not None else default_electrolyte),
        relax=(relax if relax is not None else default_relax),
        restart=restart,
    )


def build_graphite_cmcna_glucose6_periodic_case(
    *,
    work_dir,
    ff,
    ion_ff=None,
    profile: str = "full",
    graphite: GraphiteSubstrateSpec | None = None,
    polymer: PolymerSlabSpec | None = None,
    electrolyte: ElectrolyteSlabSpec | None = None,
    relax: SandwichRelaxationSpec | None = None,
    restart: bool | None = None,
) -> GraphitePolymerElectrolyteSandwichResult:
    smoke = str(profile).strip().lower() == "smoke"
    default_graphite = GraphiteSubstrateSpec(
        nx=(10 if smoke else 16),
        ny=(10 if smoke else 14),
        n_layers=4,
        edge_cap="periodic",
        name="GRAPH",
        top_padding_ang=15.0,
    )
    default_polymer = default_cmcna_polymer_spec(
        name="CMC6",
        monomers=(
            MoleculeSpec(
                name="glucose_6",
                smiles="*OC1OC(COCC(=O)[O-])C(*)C(O)C1O",
                prefer_db=True,
                require_ready=True,
                polyelectrolyte_mode=True,
            ),
        ),
        monomer_ratio=(1.0,),
        dp=(20 if smoke else 50),
        chain_count=(None if smoke else 8),
        target_density_g_cm3=1.50,
        slab_z_nm=(4.2 if smoke else 5.0),
        min_chain_count=2,
        initial_pack_z_scale=(1.55 if smoke else 1.40),
        pack_retry=80,
        pack_retry_step=3600,
        pack_threshold_ang=1.60,
        pack_dec_rate=0.68,
        counterion=MoleculeSpec(name="Na", smiles="[Na+]", use_ion_ff=True, charge_scale=1.0),
    )
    default_electrolyte = default_carbonate_lipf6_electrolyte_spec(
        solvents=(
            MoleculeSpec(name="EC", smiles="O=C1OCCO1", prefer_db=True, require_ready=True),
            MoleculeSpec(name="EMC", smiles="CCOC(=O)OC", prefer_db=True, require_ready=True),
            MoleculeSpec(name="DEC", smiles="CCOC(=O)OCC", prefer_db=True, require_ready=True),
        ),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        salt_anion=MoleculeSpec(
            name="PF6",
            smiles="F[P-](F)(F)(F)(F)F",
            bonded="DRIH",
            charge_scale=0.8,
            prefer_db=True,
            require_ready=True,
        ),
        slab_z_nm=(4.6 if smoke else 5.4),
        min_salt_pairs=(4 if smoke else 8),
        target_density_g_cm3=1.32,
        initial_pack_density_g_cm3=0.86,
        pack_retry=44,
        pack_retry_step=2800,
        pack_threshold_ang=1.55,
        pack_dec_rate=0.70,
    )
    default_relax = SandwichRelaxationSpec(
        omp=(16 if smoke else 24),
        gpu=1,
        gpu_id=0,
        psi4_omp=(24 if smoke else 36),
        psi4_memory_mb=(24000 if smoke else 32000),
        bulk_eq21_final_ns=(0.06 if smoke else 0.12),
        bulk_additional_loops=1,
        stacked_pre_nvt_ps=(10.0 if smoke else 20.0),
        stacked_z_relax_ps=(40.0 if smoke else 100.0),
        stacked_exchange_ps=(60.0 if smoke else 160.0),
    )
    return build_graphite_cmcna_electrolyte_sandwich(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=(graphite if graphite is not None else default_graphite),
        polymer=(polymer if polymer is not None else default_polymer),
        electrolyte=(electrolyte if electrolyte is not None else default_electrolyte),
        relax=(relax if relax is not None else default_relax),
        restart=restart,
    )


def format_sandwich_result_summary(
    result: GraphitePolymerElectrolyteSandwichResult,
    *,
    profile: str | None = None,
) -> tuple[str, ...]:
    lines = []
    acceptance = dict(getattr(result, "acceptance", {}) or {})
    stack_checks = dict(getattr(result, "stack_checks", {}) or {})
    if profile is not None:
        lines.append(f"profile = {str(profile).strip().lower()}")
    lines.append(f"manifest_path = {result.manifest_path}")
    lines.append(f"relaxed_gro = {result.relaxed_gro}")
    lines.append(f"polymer_density_g_cm3 = {round(float(result.polymer_phase.density_g_cm3), 4)}")
    lines.append(f"electrolyte_density_g_cm3 = {round(float(result.electrolyte_phase.density_g_cm3), 4)}")
    if acceptance:
        lines.append(f"accepted = {bool(acceptance.get('accepted', False))}")
        lines.append(f"failed_checks = {list(acceptance.get('failed_checks', []))}")
        lines.append(f"order_ok = {bool(acceptance.get('order_ok', False))}")
        lines.append(f"wrapped_ok = {bool(acceptance.get('wrapped_ok', False))}")
        lines.append(f"polymer_density_ok = {bool(acceptance.get('polymer_density_ok', False))}")
        lines.append(f"electrolyte_density_ok = {bool(acceptance.get('electrolyte_density_ok', False))}")
        lines.append(f"core_gaps_ok = {bool(acceptance.get('core_gaps_ok', False))}")
        if "graphite_polymer_core_gap_nm" in acceptance:
            lines.append(
                "graphite_polymer_core_gap_nm = "
                f"{round(float(acceptance.get('graphite_polymer_core_gap_nm', 0.0)), 4)}"
            )
        if "polymer_electrolyte_core_gap_nm" in acceptance:
            lines.append(
                "polymer_electrolyte_core_gap_nm = "
                f"{round(float(acceptance.get('polymer_electrolyte_core_gap_nm', 0.0)), 4)}"
            )
    else:
        lines.append(f"stack_checks = {stack_checks}")
    return tuple(lines)


def print_sandwich_result_summary(
    result: GraphitePolymerElectrolyteSandwichResult,
    *,
    profile: str | None = None,
) -> None:
    for line in format_sandwich_result_summary(result, profile=profile):
        print(line)


__all__ = [
    "build_graphite_peo_example_case",
    "build_graphite_cmcna_example_case",
    "build_graphite_cmcna_glucose6_periodic_case",
    "format_sandwich_result_summary",
    "print_sandwich_result_summary",
]
