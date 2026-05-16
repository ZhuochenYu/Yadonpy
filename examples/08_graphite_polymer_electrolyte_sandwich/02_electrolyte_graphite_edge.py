from __future__ import annotations

"""Example 08-02: edge graphite | carbonate/LiPF6 electrolyte.

The graphite layer is a finite, capped edge slab (`periodic_xy=False`).  Change
`edge_cap` near the top to test H/OH/O/COOH edge chemistry without touching the
rest of the workflow.
"""

from pathlib import Path

from yadonpy.core import workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.interface import (
    GraphiteLayerSpec,
    LayerStackRelaxationSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    analyze_layer_stack_interface,
    build_layer_stack,
    run_layer_stack_nvt,
)
from yadonpy.runtime import set_run_options


# ---------------- user inputs ----------------
restart_status = True
set_run_options(restart=restart_status)

ff = GAFF2_mod()
ion_ff = MERZ()

temp = 318.15
mpi = 1
omp = 14
gpu = 1
gpu_id = 0
run_sampling = True
sample_ns = 2.0
edge_cap = "OH"

EC_smiles = "O=C1OCCO1"
EMC_smiles = "CCOC(=O)OC"
DEC_smiles = "CCOC(=O)OCC"
Li_smiles = "[Li+]"
PF6_smiles = "F[P-](F)(F)(F)(F)F"

solvent_counts = (8, 6, 14)
salt_pairs = 3
charge_scale = 0.8

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir" / "02_electrolyte_graphite_edge"


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()
    work_dir = workdir(work_dir, restart=restart_status)

    EC = ff.mol(EC_smiles, charge="RESP", prefer_db=True, require_ready=True)
    EC = ff.ff_assign(EC, report=False)
    EMC = ff.mol(EMC_smiles, charge="RESP", prefer_db=True, require_ready=True)
    EMC = ff.ff_assign(EMC, report=False)
    DEC = ff.mol(DEC_smiles, charge="RESP", prefer_db=True, require_ready=True)
    DEC = ff.ff_assign(DEC, report=False)
    PF6 = ff.mol(PF6_smiles, charge="RESP", prefer_db=True, require_ready=True)
    PF6 = ff.ff_assign(PF6, bonded="DRIH", report=False)
    Li = ion_ff.mol(Li_smiles)
    Li = ion_ff.ff_assign(Li, report=False)
    if not all((EC, EMC, DEC, PF6, Li)):
        raise RuntimeError("MolDB/FF assignment failed for one or more electrolyte species.")

    graphite = GraphiteLayerSpec(
        name="GRAPHITE_EDGE",
        nx=6,
        ny=8,
        n_layers=3,
        orientation="edge",
        periodic_xy=False,
        edge_cap=edge_cap,
    )
    electrolyte = MolecularLayerSpec(
        name="ELECTROLYTE",
        species=(EC, EMC, DEC, Li, PF6),
        counts=(*solvent_counts, salt_pairs, salt_pairs),
        thickness_nm=2.4,
        density_target_g_cm3=1.2,
        layer_kind="electrolyte",
        charge_scale=(1.0, 1.0, 1.0, charge_scale, charge_scale),
    )
    stack = LayerStackSpec(
        layers=(graphite, electrolyte),
        order="bottom_to_top",
        pbc_mode="xyz",
        name="electrolyte_graphite_edge",
        default_gap_nm=0.35,
    )
    relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=sample_ns)

    result = build_layer_stack(
        stack=stack,
        relaxation=relaxation,
        work_dir=work_dir,
        ff_name="gaff2_mod",
        charge_method="RESP",
        restart=restart_status,
    )
    print(f"layer_stack_manifest = {result.manifest_path}")
    print(f"stack_gmx_dir = {result.system_gro.parent}")
    print(f"acceptance = {result.acceptance}")

    analyze_layer_stack_interface(work_dir=work_dir, analysis_profile="interface_fast")
    if run_sampling:
        nvt = run_layer_stack_nvt(
            result,
            time_ns=sample_ns,
            temp=temp,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            restart=restart_status,
        )
        print(f"nvt_summary = {nvt.summary_path}")
