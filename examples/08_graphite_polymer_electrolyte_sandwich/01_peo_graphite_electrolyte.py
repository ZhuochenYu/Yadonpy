from __future__ import annotations

"""Example 08A: graphite | PEO | carbonate/LiPF6 three-phase stack.

This script mirrors the Example 02/07 style: user inputs live near the top,
MolDB-backed molecules are checked explicitly, and the main block follows the
scientific workflow linearly. It does not run DFT; required RESP entries should
already exist in MolDB.
"""

from pathlib import Path

from yadonpy.core import workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.interface import (
    GraphiteSubstrateSpec,
    InterfaceBuildPolicy,
    MoleculeSpec,
    SandwichRelaxationSpec,
    build_graphite_peo_electrolyte_sandwich,
    default_carbonate_lipf6_electrolyte_spec,
    default_peo_polymer_spec,
    print_interface_result_summary,
    run_sandwich_nvt_followup,
)
from yadonpy.runtime import set_run_options


def _ready_moldb_spec(
    name: str,
    smiles: str,
    *,
    bonded: str | None = None,
    charge_scale: float = 1.0,
    polyelectrolyte_mode: bool | None = None,
) -> MoleculeSpec:
    return MoleculeSpec(
        name=name,
        smiles=smiles,
        bonded=bonded,
        charge_scale=float(charge_scale),
        prefer_db=True,
        require_ready=True,
        polyelectrolyte_mode=polyelectrolyte_mode,
    )


def _ion_spec(name: str, smiles: str, *, charge_scale: float = 1.0) -> MoleculeSpec:
    return MoleculeSpec(name=name, smiles=smiles, use_ion_ff=True, charge_scale=float(charge_scale))


def _check_ready_from_moldb(ff, spec: MoleculeSpec) -> None:
    """Example-07-style fail-fast check for MolDB-backed species."""

    if spec.use_ion_ff or not spec.require_ready:
        return
    try:
        mol = ff.mol(
            spec.smiles,
            charge=spec.charge_method,
            require_ready=True,
            prefer_db=True,
            polyelectrolyte_mode=spec.polyelectrolyte_mode,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{spec.name} is expected to be precomputed in MolDB. "
            "Refresh the relevant species with examples/07_moldb_precompute_and_reuse "
            "before running Example 08."
        ) from exc
    if not ff.ff_assign(mol, bonded=spec.bonded, report=False):
        raise RuntimeError(f"Cannot assign force-field parameters for MolDB-backed {spec.name}.")


# ---------------- user inputs ----------------
restart_status = True
set_run_options(restart=restart_status)

ff = GAFF2_mod()
ion_ff = MERZ()

# Choose "smoke" for a small validation build, or "production" for the normal case.
profile = "production"
smoke_mode = profile == "smoke"
route = "screening" if smoke_mode else "production"

# MD resources
temp = 318.15
mpi = 1
omp = 12
gpu = 1
gpu_id = 0

# Psi4 settings are only used if a required molecule is not already cached by
# an upstream builder path. The Example 08 public scripts are intended to be
# MolDB-ready and should not launch fresh DFT.
psi4_omp = 24 if smoke_mode else 36
psi4_memory_mb = 24000 if smoke_mode else 32000

run_nvt_after_stack = False
nvt_ns = 4.0

BASE_DIR = Path(__file__).resolve().parent
RUN_NAME = "01_peo_graphite_electrolyte_smoke" if smoke_mode else "01_peo_graphite_electrolyte"
work_dir = BASE_DIR / "work_dir" / RUN_NAME


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    work_dir = workdir(work_dir, restart=restart_status)

    peo_monomer = _ready_moldb_spec("PEO_monomer", "*CCO*")
    EC = _ready_moldb_spec("EC", "O=C1OCCO1")
    EMC = _ready_moldb_spec("EMC", "CCOC(=O)OC")
    DEC = _ready_moldb_spec("DEC", "CCOC(=O)OCC")
    Li = _ion_spec("Li", "[Li+]", charge_scale=0.8)
    PF6 = _ready_moldb_spec("PF6", "F[P-](F)(F)(F)(F)F", bonded="DRIH", charge_scale=0.8)

    for spec in (peo_monomer, EC, EMC, DEC, PF6):
        _check_ready_from_moldb(ff, spec)

    graphite = GraphiteSubstrateSpec(
        nx=(4 if smoke_mode else 6),
        ny=(4 if smoke_mode else 5),
        n_layers=(2 if smoke_mode else 3),
        edge_cap="H",
        name="GRAPH",
    )

    polymer = default_peo_polymer_spec(
        name="PEO",
        monomers=(peo_monomer,),
        monomer_ratio=(1.0,),
        chain_target_atoms=(220 if smoke_mode else 320),
        min_chain_count=(4 if smoke_mode else 3),
        target_density_g_cm3=1.08,
        slab_z_nm=(3.2 if smoke_mode else 4.0),
        initial_pack_z_scale=(1.18 if smoke_mode else 1.20),
        pack_retry=(30 if smoke_mode else 48),
        pack_retry_step=(2000 if smoke_mode else 2600),
    )

    electrolyte = default_carbonate_lipf6_electrolyte_spec(
        solvents=(EC, EMC, DEC),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        salt_cation=Li,
        salt_anion=PF6,
        slab_z_nm=(3.8 if smoke_mode else 4.8),
        min_salt_pairs=(2 if smoke_mode else 4),
        target_density_g_cm3=(1.28 if smoke_mode else 1.30),
        initial_pack_density_g_cm3=(0.82 if smoke_mode else 0.86),
        pack_retry=(30 if smoke_mode else 44),
        pack_retry_step=(1800 if smoke_mode else 2400),
    )

    relax = SandwichRelaxationSpec(
        temperature_k=temp,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        psi4_omp=psi4_omp,
        psi4_memory_mb=psi4_memory_mb,
        bulk_eq21_final_ns=(0.0 if smoke_mode else 0.10),
        bulk_additional_loops=(0 if smoke_mode else 1),
        bulk_eq21_exec_kwargs=(
            {
                "eq21_tmax": 600.0,
                "eq21_pmax": 10000.0,
                "eq21_pre_nvt_ps": 2.0,
                "sim_time": 0.04,
                "eq21_npt_time_scale": 0.20,
            }
            if smoke_mode
            else {}
        ),
        stacked_pre_nvt_ps=(10.0 if smoke_mode else 20.0),
        stacked_z_relax_ps=(40.0 if smoke_mode else 80.0),
        stacked_exchange_ps=(60.0 if smoke_mode else 120.0),
    )

    policy = InterfaceBuildPolicy(
        phase_preparation="final_xy_walled",
        stack_relaxation="natural_contact",
        acceptance_required=True,
        retry_profile="conservative",
        max_stack_rescue_rounds=(0 if smoke_mode else 1),
    )

    result = build_graphite_peo_electrolyte_sandwich(
        work_dir=work_dir,
        ff=ff,
        ion_ff=ion_ff,
        graphite=graphite,
        polymer=polymer,
        electrolyte=electrolyte,
        relax=relax,
        policy=policy,
        route=route,
        restart=restart_status,
    )
    print_interface_result_summary(result, profile=profile)
    print(f"stack_gmx_dir = {Path(result.stack_export.system_gro).parent}")
    print(f"stack_relaxed_gro = {result.relaxed_gro}")

    if run_nvt_after_stack:
        followup = run_sandwich_nvt_followup(
            result,
            work_dir=work_dir / "07_nvt_followup",
            time_ns=nvt_ns,
            temp=temp,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            restart=restart_status,
        )
        print(f"nvt_followup_dir = {followup.work_dir}")
        print(f"nvt_followup_gro = {followup.final_gro}")
        print(f"nvt_followup_summary = {followup.summary_path}")
