from __future__ import annotations

"""Example 08B: graphite | CMC-Na | carbonate/LiPF6 three-phase stack.

The public CMC-Na interface example follows the same script-first pattern as
Examples 02 and 07. All chemically expensive species are expected to be
RESP-ready in MolDB; this script only builds, packs, relaxes, and optionally runs
a short NVT follow-up for the assembled interface.
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
    build_cmcna_graphite_electrolyte_stack,
    default_carbonate_lipf6_electrolyte_spec,
    default_cmcna_polymer_spec,
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

# Choose "smoke" for syntax/build validation, "compact" for a <20k-atom
# three-phase screening model, or "production" for the normal case.
profile = "production"
smoke_mode = profile == "smoke"
compact_mode = profile == "compact"
route = "screening" if smoke_mode else "production"

# MD resources
temp = 318.15
mpi = 1
omp = 12 if smoke_mode else 14
gpu = 1
gpu_id = 0

psi4_omp = 24 if (smoke_mode or compact_mode) else 36
psi4_memory_mb = 24000 if (smoke_mode or compact_mode) else 32000

run_nvt_after_stack = False
nvt_ns = 4.0
# CMC-Na has many hydroxyl constraints on relatively rigid sugar rings.  Keep
# the optional follow-up conservative; PEO can usually use the 2 fs default.
nvt_dt_ps = 0.001
# This graphite model is bonded and should move naturally with the soft phases.
# Do not freeze it: GROMACS can then keep update/bonded/PME on GPU.
stack_freeze_group = None
stack_gpu_offload_mode = "full"

BASE_DIR = Path(__file__).resolve().parent
RUN_NAME = (
    "02_cmcna_graphite_electrolyte_smoke"
    if smoke_mode
    else ("02_cmcna_graphite_electrolyte_compact" if compact_mode else "02_cmcna_graphite_electrolyte")
)
work_dir = BASE_DIR / "work_dir" / RUN_NAME


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    work_dir = workdir(work_dir, restart=restart_status)

    glucose_6 = _ready_moldb_spec(
        "glucose_6",
        "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O",
        polyelectrolyte_mode=True,
    )
    Na = _ion_spec("Na", "[Na+]", charge_scale=1.0)
    EC = _ready_moldb_spec("EC", "O=C1OCCO1")
    EMC = _ready_moldb_spec("EMC", "CCOC(=O)OC")
    DEC = _ready_moldb_spec("DEC", "CCOC(=O)OCC")
    Li = _ion_spec("Li", "[Li+]", charge_scale=0.8)
    PF6 = _ready_moldb_spec("PF6", "F[P-](F)(F)(F)(F)F", bonded="DRIH", charge_scale=0.8)

    for spec in (glucose_6, EC, EMC, DEC, PF6):
        _check_ready_from_moldb(ff, spec)

    graphite = GraphiteSubstrateSpec(
        nx=(10 if smoke_mode else (6 if compact_mode else 16)),
        ny=(10 if smoke_mode else (6 if compact_mode else 14)),
        n_layers=(4 if not compact_mode else 3),
        edge_cap="periodic",
        name="GRAPH",
        top_padding_ang=(10.0 if compact_mode else 15.0),
    )

    polymer = default_cmcna_polymer_spec(
        name="CMC6",
        monomers=(glucose_6,),
        monomer_ratio=(1.0,),
        chain_target_atoms=(160 if compact_mode else 280),
        dp=(20 if smoke_mode else (12 if compact_mode else 50)),
        chain_count=(None if smoke_mode else (2 if compact_mode else 8)),
        target_density_g_cm3=1.50,
        slab_z_nm=(4.2 if smoke_mode else (2.6 if compact_mode else 5.0)),
        min_chain_count=2,
        initial_pack_z_scale=(1.55 if smoke_mode else (1.35 if compact_mode else 1.40)),
        pack_retry=(60 if compact_mode else 80),
        pack_retry_step=(2600 if compact_mode else 3600),
        pack_threshold_ang=1.60,
        pack_dec_rate=0.68,
        counterion=Na,
    )

    electrolyte = default_carbonate_lipf6_electrolyte_spec(
        solvents=(EC, EMC, DEC),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        salt_cation=Li,
        salt_anion=PF6,
        slab_z_nm=(4.6 if smoke_mode else (2.4 if compact_mode else 5.4)),
        min_salt_pairs=(4 if smoke_mode else (2 if compact_mode else 8)),
        target_density_g_cm3=1.32,
        initial_pack_density_g_cm3=0.86,
        pack_retry=(36 if compact_mode else 44),
        pack_retry_step=(2200 if compact_mode else 2800),
        pack_threshold_ang=1.55,
        pack_dec_rate=0.70,
    )

    relax = SandwichRelaxationSpec(
        temperature_k=temp,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        psi4_omp=psi4_omp,
        psi4_memory_mb=psi4_memory_mb,
        bulk_eq21_final_ns=(0.06 if smoke_mode else (0.08 if compact_mode else 0.12)),
        bulk_additional_loops=1,
        stacked_pre_nvt_ps=(10.0 if smoke_mode else (12.0 if compact_mode else 20.0)),
        stacked_z_relax_ps=(40.0 if smoke_mode else (60.0 if compact_mode else 100.0)),
        stacked_exchange_ps=(60.0 if smoke_mode else (90.0 if compact_mode else 160.0)),
        stack_freeze_group=stack_freeze_group,
        stack_frozen_gpu_offload_mode=stack_gpu_offload_mode,
        stack_final_gpu_offload_mode=stack_gpu_offload_mode,
    )

    policy = InterfaceBuildPolicy(
        phase_preparation="final_xy_walled",
        stack_relaxation="natural_contact",
        acceptance_required=True,
        retry_profile="conservative",
        max_stack_rescue_rounds=(0 if smoke_mode else (0 if compact_mode else 1)),
    )

    result = build_cmcna_graphite_electrolyte_stack(
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
            dt_ps=nvt_dt_ps,
            freeze_group=stack_freeze_group,
            bridge_gpu_offload_mode=stack_gpu_offload_mode,
            final_gpu_offload_mode=stack_gpu_offload_mode,
        )
        print(f"nvt_followup_dir = {followup.work_dir}")
        print(f"nvt_followup_gro = {followup.final_gro}")
        print(f"nvt_followup_summary = {followup.summary_path}")
