from __future__ import annotations

# Example 01: PF6- -> QM/RESP -> DRIH-aware MolDB entry -> DB-backed export
#
# This script does two things in one place:
#   1) compute PF6- once from scratch and store the finished result into MolDB;
#   2) immediately demonstrate the later workflow style that reuses MolDB via:
#        PF6 = ff.mol(PF6_smiles)
#        PF6 = ff.ff_assign(PF6, bonded="DRIH")
#
# After step (1), the second pair of lines becomes the recommended way to reuse PF6
# in later scripts. The bonded DRIH patch is restored from MolDB together with the
# stored geometry/charges for the selected variant.

from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor, require_psi4_resp
from yadonpy.core import utils, workdir
from yadonpy.sim import qm
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.io.mol2 import write_mol2
from yadonpy.io.gmx import write_gmx


def select_charge_method() -> tuple[str, str | None]:
    try:
        require_psi4_resp()
    except ImportError as exc:
        return "gasteiger", str(exc)
    return "RESP", None


def build_and_store_pf6(*, ff: GAFF2_mod, pf6_smiles: str, work_dir, omp_psi4: int, mem_mb: int, charge_method: str):
    pf6 = utils.mol_from_smiles(pf6_smiles, coord=False, name="PF6")
    utils.ensure_3d_coords(pf6, smiles_hint=pf6_smiles, engine="openbabel")

    if str(charge_method).strip().upper() == "RESP":
        qm.assign_charges(
            pf6,
            charge="RESP",
            opt=True,
            work_dir=work_dir,
            log_name="PF6_build",
            omp=omp_psi4,
            memory=mem_mb,
            total_charge=-1,
            total_multiplicity=1,
            symmetrize=True,
            auto_level=True,
        )
        pf6 = ff.ff_assign(pf6, bonded="DRIH")
    else:
        pf6 = ff.ff_assign(pf6, charge=charge_method, bonded="DRIH")

    if not pf6:
        raise RuntimeError("FF assignment failed for PF6 during the build phase")

    record = ff.store_to_db(pf6, smiles_or_psmiles=pf6_smiles, name="PF6", charge=charge_method)
    return pf6, record


def main():
    restart_status = False
    set_run_options(restart=restart_status)

    PF6_smiles = "F[P-](F)(F)(F)(F)F"
    omp_psi4 = 8
    mem_mb = 8000
    work_root = Path("work_pf6_only").resolve()

    doctor(print_report=True)
    ensure_initialized()

    work_dir = workdir(work_root, restart=restart_status)
    mol2_dir = work_dir / "00_molecules"
    build_export_dir = work_dir / "01_pf6_build_exports"
    db_export_dir = work_dir / "02_pf6_from_moldb_gmx"
    mol2_dir.mkdir(parents=True, exist_ok=True)
    build_export_dir.mkdir(parents=True, exist_ok=True)
    db_export_dir.mkdir(parents=True, exist_ok=True)

    ff = GAFF2_mod()
    charge_method, qm_fallback_reason = select_charge_method()

    if qm_fallback_reason is None:
        print("[INFO] Example 01 will build PF6 with QM/RESP, then store it into MolDB.")
    else:
        print("[WARN] Optional QM stack is unavailable; Example 01 will fall back to Gasteiger charges so the full MolDB workflow still runs.")
        print(f"[WARN] Missing QM dependency detail: {qm_fallback_reason}")
        print("[WARN] This run still demonstrates the same direct MolDB API, but the stored charge variant is Gasteiger instead of RESP.")

    pf6_built, record = build_and_store_pf6(
        ff=ff,
        pf6_smiles=PF6_smiles,
        work_dir=work_dir,
        omp_psi4=omp_psi4,
        mem_mb=mem_mb,
        charge_method=charge_method,
    )

    built_mol2 = write_mol2(mol=pf6_built, out_dir=mol2_dir, name="PF6_built")
    built_gro, built_itp, built_top = write_gmx(
        mol=pf6_built,
        out_dir=build_export_dir,
        mol_name="PF6",
    )

    if str(charge_method).strip().upper() == "RESP":
        PF6 = ff.mol(PF6_smiles)
    else:
        PF6 = ff.mol(PF6_smiles, charge=charge_method)
    PF6 = ff.ff_assign(PF6, bonded="DRIH")
    if not PF6:
        raise RuntimeError("FF assignment failed for MolDB-backed PF6")
    db_mol2 = write_mol2(mol=PF6, out_dir=mol2_dir, name="PF6_from_moldb")
    db_gro, db_itp, db_top = write_gmx(
        mol=PF6,
        out_dir=db_export_dir,
        mol_name="PF6",
    )

    print("\n[DONE] PF6- was computed, stored to MolDB, then reloaded from MolDB for export:")
    print(f"  MolDB key          : {record.key}")
    print(f"  charge variant     : {charge_method}")
    print(f"  built mol2         : {built_mol2}")
    print(f"  built gro/itp/top  : {built_gro} | {built_itp} | {built_top}")
    print(f"  db mol2            : {db_mol2}")
    print(f"  db gro/itp/top     : {db_gro} | {db_itp} | {db_top}")

    print("\nNotes:")
    print(f"  - 01_pf6_build_exports/ contains the artifacts written immediately after charge assignment ({charge_method}) + DRIH build.")
    print("  - 02_pf6_from_moldb_gmx/ contains the artifacts written after reloading PF6 through the MolDB-backed ff.mol(...) path.")
    print("  - Later workflows can now reuse PF6 with the following pattern:")
    if str(charge_method).strip().upper() == "RESP":
        print("    PF6 = ff.mol(PF6_smiles)")
    else:
        print(f"    PF6 = ff.mol(PF6_smiles, charge=\"{charge_method}\")")
    print("    PF6 = ff.ff_assign(PF6, bonded=\"DRIH\")")
    print("  - If PF6_smiles is the literal string, that call is:")
    if str(charge_method).strip().upper() == "RESP":
        print("    PF6 = ff.mol(\"F[P-](F)(F)(F)(F)F\")")
    else:
        print(f"    PF6 = ff.mol(\"F[P-](F)(F)(F)(F)F\", charge=\"{charge_method}\")")
    print("    PF6 = ff.ff_assign(PF6, bonded=\"DRIH\")\n")


if __name__ == "__main__":
    main()
