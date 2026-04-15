from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from ..core.graphite import GraphiteBuildResult


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    smiles: str
    charge_method: str = "RESP"
    bonded: str | None = None
    prefer_db: bool = False
    require_ready: bool = False
    use_ion_ff: bool = False
    charge_scale: float = 1.0
    basis_set: str | None = None
    method: str | None = None
    polyelectrolyte_mode: bool | None = None
    polyelectrolyte_detection: str | None = None


@dataclass(frozen=True)
class GraphiteSubstrateSpec:
    nx: int = 4
    ny: int = 4
    n_layers: int = 3
    edge_cap: str = "H"
    orientation: str = "basal"
    name: str = "GRAPH"
    top_padding_ang: float = 15.0


@dataclass(frozen=True)
class PolymerSlabSpec:
    name: str = "PEO"
    monomer_smiles: str = "*CCO*"
    monomers: tuple[MoleculeSpec, ...] = ()
    monomer_ratio: tuple[float, ...] = (1.0,)
    terminal_smiles: str = "[H][*]"
    terminal: MoleculeSpec | None = None
    chain_target_atoms: int = 280
    dp: int | None = None
    chain_count: int | None = None
    counterion: MoleculeSpec | None = None
    target_density_g_cm3: float = 1.10
    slab_z_nm: float = 3.6
    min_chain_count: int = 2
    tacticity: str = "atactic"
    charge_scale: float = 1.0
    initial_pack_z_scale: float = 1.18
    pack_retry: int = 30
    pack_retry_step: int = 2400
    pack_threshold_ang: float = 1.55
    pack_dec_rate: float = 0.72


@dataclass(frozen=True)
class ElectrolyteSlabSpec:
    solvents: tuple[MoleculeSpec, ...] = field(
        default_factory=lambda: (
            MoleculeSpec(name="DME", smiles="COCCOC"),
        )
    )
    salt_cation: MoleculeSpec = field(
        default_factory=lambda: MoleculeSpec(name="Li", smiles="[Li+]", use_ion_ff=True, charge_scale=0.8)
    )
    salt_anion: MoleculeSpec = field(
        default_factory=lambda: MoleculeSpec(
            name="FSI",
            smiles="FS(=O)(=O)[N-]S(=O)(=O)F",
            charge_scale=0.8,
        )
    )
    solvent_mass_ratio: tuple[float, ...] = (1.0,)
    target_density_g_cm3: float = 1.18
    slab_z_nm: float = 4.0
    salt_molarity_M: float = 1.0
    min_salt_pairs: int = 3
    initial_pack_density_g_cm3: float | None = None
    pack_retry: int = 30
    pack_retry_step: int = 2400
    pack_threshold_ang: float = 1.55
    pack_dec_rate: float = 0.72


@dataclass(frozen=True)
class SandwichRelaxationSpec:
    temperature_k: float = 300.0
    pressure_bar: float = 1.0
    mpi: int = 1
    omp: int = 8
    gpu: int = 1
    gpu_id: int | None = 0
    psi4_omp: int = 8
    psi4_memory_mb: int = 16000
    bulk_eq21_final_ns: float = 0.10
    bulk_additional_loops: int = 1
    bulk_eq21_exec_kwargs: dict[str, float] = field(default_factory=dict)
    graphite_to_polymer_gap_ang: float = 3.8
    polymer_to_electrolyte_gap_ang: float = 4.2
    top_padding_ang: float = 12.0
    stacked_pre_nvt_ps: float = 20.0
    stacked_z_relax_ps: float = 80.0
    stacked_exchange_ps: float = 120.0


@dataclass(frozen=True)
class SandwichPhaseReport:
    label: str
    box_nm: tuple[float, float, float]
    density_g_cm3: float
    species_names: tuple[str, ...]
    counts: tuple[int, ...]
    target_density_g_cm3: float | None = None
    occupied_density_g_cm3: float | None = None
    bulk_like_density_g_cm3: float | None = None


@dataclass(frozen=True)
class GraphitePolymerElectrolyteSandwichResult:
    graphite: GraphiteBuildResult
    polymer_phase: SandwichPhaseReport
    electrolyte_phase: SandwichPhaseReport
    stack_export: object
    relaxed_gro: Path
    manifest_path: Path
    stack_checks: dict[str, object] = field(default_factory=dict)
    acceptance: dict[str, object] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphitePreparationResult:
    work_dir: Path
    summary_path: Path
    graphite_spec: GraphiteSubstrateSpec
    graphite: GraphiteBuildResult
    master_xy_nm: tuple[float, float]
    box_nm: tuple[float, float, float]
    route: str = "screening"
    footprint_negotiations: tuple[dict[str, object], ...] = ()
    notes: tuple[str, ...] = ()
    context: dict[str, object] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class BulkCalibrationResult:
    label: str
    work_dir: Path
    summary_path: Path
    phase_preparation_mode: str
    target_xy_nm: tuple[float, float]
    bulk_reference_box_nm: tuple[float, float, float]
    target_z_nm: float
    target_density_g_cm3: float
    selected_bulk_pack_density_g_cm3: float
    charged_phase: bool
    species_names: tuple[str, ...]
    counts: tuple[int, ...]
    notes: tuple[str, ...] = ()
    context: dict[str, object] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class InterphaseBuildResult:
    label: str
    work_dir: Path
    summary_path: Path
    report: SandwichPhaseReport
    top_path: Path
    gro_path: Path
    phase_preparation_mode: str
    occupied_thickness_nm: float
    route: str = "screening"
    notes: tuple[str, ...] = ()
    context: dict[str, object] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class StackReleaseResult:
    work_dir: Path
    manifest_path: Path
    relaxed_gro: Path
    graphite: GraphiteBuildResult
    polymer_phase: SandwichPhaseReport
    electrolyte_phase: SandwichPhaseReport
    stack_checks: dict[str, object] = field(default_factory=dict)
    acceptance: dict[str, object] = field(default_factory=dict)
    route: str = "screening"
    notes: tuple[str, ...] = ()
    sandwich_result: GraphitePolymerElectrolyteSandwichResult | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class InterfaceTransportResult:
    work_dir: Path
    analysis_dir: Path
    summary_path: Path
    rdf: dict[str, object]
    msd: dict[str, object]
    sigma: dict[str, object]
    migration: dict[str, object] | None = None


def default_peo_polymer_spec(**kwargs) -> PolymerSlabSpec:
    return PolymerSlabSpec(**kwargs)


def default_peo_electrolyte_spec(**kwargs) -> ElectrolyteSlabSpec:
    return ElectrolyteSlabSpec(**kwargs)


def default_cmcna_polymer_spec(**kwargs) -> PolymerSlabSpec:
    base = PolymerSlabSpec(
        name="CMC",
        monomers=(
            MoleculeSpec(
                name="glucose_0",
                smiles="*OC1OC(CO)C(*)C(O)C1O",
                prefer_db=True,
                require_ready=True,
                polyelectrolyte_mode=False,
            ),
            MoleculeSpec(
                name="glucose_2",
                smiles="*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]",
                prefer_db=True,
                require_ready=True,
                polyelectrolyte_mode=True,
            ),
            MoleculeSpec(
                name="glucose_6",
                smiles="*OC1OC(COCC(=O)[O-])C(*)C(O)C1O",
                prefer_db=True,
                require_ready=True,
                polyelectrolyte_mode=True,
            ),
        ),
        monomer_ratio=(12.0, 26.0, 35.0),
        terminal=MoleculeSpec(name="CMC_terminal", smiles="[H][*]", require_ready=False, prefer_db=False),
        dp=60,
        target_density_g_cm3=1.50,
        slab_z_nm=4.2,
        min_chain_count=2,
        charge_scale=1.0,
        initial_pack_z_scale=1.24,
        pack_retry=60,
        pack_retry_step=3200,
        pack_threshold_ang=1.58,
        pack_dec_rate=0.70,
        counterion=MoleculeSpec(name="Na", smiles="[Na+]", use_ion_ff=True, charge_scale=1.0),
    )
    return replace(base, **kwargs)


def default_carbonate_lipf6_electrolyte_spec(**kwargs) -> ElectrolyteSlabSpec:
    base = ElectrolyteSlabSpec(
        solvents=(
            MoleculeSpec(name="EC", smiles="O=C1OCCO1", prefer_db=True, require_ready=True),
            MoleculeSpec(name="DEC", smiles="CCOC(=O)OCC", prefer_db=True, require_ready=True),
            MoleculeSpec(name="EMC", smiles="CCOC(=O)OC", prefer_db=True, require_ready=True),
        ),
        salt_cation=MoleculeSpec(name="Li", smiles="[Li+]", use_ion_ff=True, charge_scale=0.8),
        salt_anion=MoleculeSpec(
            name="PF6",
            smiles="F[P-](F)(F)(F)(F)F",
            bonded="DRIH",
            charge_scale=0.8,
            prefer_db=True,
            require_ready=True,
        ),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        target_density_g_cm3=1.32,
        slab_z_nm=4.8,
        salt_molarity_M=1.0,
        min_salt_pairs=6,
        initial_pack_density_g_cm3=0.88,
        pack_retry=40,
        pack_retry_step=2600,
        pack_threshold_ang=1.55,
        pack_dec_rate=0.70,
    )
    return replace(base, **kwargs)
