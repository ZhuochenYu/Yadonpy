from __future__ import annotations

import json
import os
from pathlib import Path

from yadonpy.core import poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2 import GAFF2
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.runtime import set_run_options
from yadonpy.sim import qm
from yadonpy.sim.benchmarking import (
    _dump_json,
    build_benchmark_compare,
    build_coordination_partition,
    build_transport_summary,
    collect_force_balance_report,
    jpcb2020_peo_litfsi_cases,
    literature_band_peo_litfsi_jpcb2020,
    resolve_jpcb2020_peo_litfsi_case,
    summarize_rdkit_species_forcefield,
)
from yadonpy.sim.preset import eq


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    return float(raw)


def _env_text(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return str(default)
    return str(raw).strip()


def _env_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _make_gaff2():
    variant = str(os.environ.get("GAFF2_VARIANT", "classic") or "classic").strip().lower()
    if variant in {"classic", "gaff2", "orig", "original"}:
        return GAFF2(), "gaff2"
    if variant in {"mod", "gaff2_mod", "modified"}:
        return GAFF2_mod(), "gaff2_mod"
    raise ValueError("GAFF2_VARIANT must be classic or mod.")


BASE_DIR = Path(__file__).resolve().parent

restart_status = _env_bool("RESTART_STATUS", False)
set_run_options(restart=restart_status)

case_labels = _env_list("JPCB_CASES", "P1.00S1.00,P1.00S0.75,P1.20S0.75")
paper_size = _env_bool("PAPER_SIZE", False)
dry_run = _env_bool("DRY_RUN", False)

target_mode = str(os.environ.get("TARGET_MODE", "normalized_inverse") or "normalized_inverse").strip()
target_temp_override = os.environ.get("TARGET_TEMP_K")
target_temp_k = float(target_temp_override) if target_temp_override and target_temp_override.strip() else None
normalized_inverse = _env_float("NORMALIZED_INVERSE", 5.4)

chain_count = _env_int("CHAIN_COUNT", 96)
prod_ns_default = _env_float("PROD_NS", 20.0)
press_bar = _env_float("PRESS_BAR", 1.0)
initial_density_g_cm3 = _env_float("INITIAL_DENSITY_G_CM3", 0.65)
max_melt_additional = _env_int("MAX_MELT_ADDITIONAL", 4)
max_atoms = _env_int("MAX_ATOMS", 30000 if not paper_size else 60000)
min_atoms = _env_int("MIN_ATOMS", 10000 if not _env_bool("SMOKE", False) else 1)

mpi = _env_int("MPI", 1)
omp = _env_int("OMP", 14)
gpu = _env_int("GPU", 1)
gpu_id = _env_int("GPU_ID", 0)
gpu_offload_mode = _env_text("GPU_OFFLOAD_MODE", "auto")
omp_psi4 = _env_int("OMP_PSI4", 32)
mem_mb = _env_int("MEM_MB", 20000)

work_dir_name = os.environ.get("WORK_DIR_NAME", "benchmark_peo_litfsi_jpcb2020_work")
work_root = Path(os.environ.get("WORK_DIR", str(BASE_DIR / work_dir_name))).resolve()


def _resolved_cases() -> list[dict]:
    out = []
    for label in case_labels:
        case = resolve_jpcb2020_peo_litfsi_case(
            label,
            chain_count=chain_count,
            target_mode=target_mode,
            normalized_inverse=normalized_inverse,
            target_temp_k=target_temp_k,
            production_ns=prod_ns_default,
            paper_size=paper_size,
        )
        out.append(case)
    return out


def _write_dry_run_plan(root: Path, cases: list[dict], ff_variant: str) -> None:
    payload = {
        "benchmark_name": "JPCB2020 PEO/LiTFSI charge-scaling reproduction",
        "forcefield": {"polymer_and_tfsi": ff_variant, "cation": "MERZ", "charge_model": "RESP"},
        "target_cases": cases,
        "available_cases": jpcb2020_peo_litfsi_cases(),
        "execution": {
            "dry_run": True,
            "notes": [
                "Set DRY_RUN=0 to run QM/build/equilibration/production.",
                "The paper used 300-600 ns production; PROD_NS defaults to a shorter screening run.",
                "Use PAPER_SIZE=1 for the original 200 PEO25 + 400 LiTFSI system.",
            ],
        },
    }
    _dump_json(root / "jpcb2020_benchmark_plan.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _prepare_species(root, ff, cation_ff):
    species_dir = root.child("00_species")
    poly_rw_dir = species_dir.child("poly_rw")
    poly_term_dir = species_dir.child("poly_term")

    monomer = utils.mol_from_smiles(r"*CCO*")
    monomer, _ = qm.conformation_search(
        monomer,
        ff=ff,
        work_dir=species_dir,
        psi4_omp=omp_psi4,
        mpi=mpi,
        omp=omp,
        memory=mem_mb,
        log_name=None,
    )
    qm.assign_charges(
        monomer,
        charge="RESP",
        opt=False,
        work_dir=species_dir,
        omp=omp_psi4,
        memory=mem_mb,
        log_name=None,
    )

    ter = utils.mol_from_smiles("[H][*]")
    qm.assign_charges(
        ter,
        charge="RESP",
        opt=True,
        work_dir=species_dir,
        omp=omp_psi4,
        memory=mem_mb,
        log_name=None,
    )

    max_dp = max(int(case["chain_dp"]) for case in _resolved_cases())
    peo = poly.random_copolymerize_rw(
        [monomer],
        max_dp,
        ratio=[1.0],
        tacticity="atactic",
        name="PEO",
        work_dir=poly_rw_dir,
    )
    peo = poly.terminate_rw(peo, ter, name="PEO", work_dir=poly_term_dir)
    peo = ff.ff_assign(peo)
    if not peo:
        raise RuntimeError("Failed to assign GAFF2 force-field parameters for PEO.")

    li = cation_ff.mol("[Li+]")
    li = cation_ff.ff_assign(li)
    if not li:
        raise RuntimeError("Failed to assign MERZ parameters for Li+.")

    try:
        tfsi = ff.mol(
            "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",
            charge="RESP",
            require_ready=True,
            prefer_db=True,
        )
        tfsi = ff.ff_assign(tfsi)
    except Exception as exc:
        raise RuntimeError(
            "This benchmark requires a ready RESP-backed TFSI record in MolDB. "
            "Precompute TFSI with GAFF2/RESP first, then rerun this script."
        ) from exc
    if not tfsi:
        raise RuntimeError("Failed to assign GAFF2 force-field parameters for TFSI.")
    return peo, li, tfsi


def _run_case(root, case: dict, peo, li, tfsi, ff_variant: str) -> dict:
    case_key = str(case["case_key"])
    case_root = root.child(f"case_{case_key.replace('.', 'p')}")
    build_dir = case_root.child("00_build_cell")

    counts = [int(case["chain_count"]), int(case["salt_pairs"]), int(case["salt_pairs"])]
    charge_scale = [
        float(case["polymer_charge_scale"]),
        float(case["li_charge_scale"]),
        float(case["anion_charge_scale"]),
    ]

    estimated_atoms = (
        counts[0] * int(peo.GetNumAtoms())
        + counts[1] * int(li.GetNumAtoms())
        + counts[2] * int(tfsi.GetNumAtoms())
    )
    if estimated_atoms < min_atoms or estimated_atoms > max_atoms:
        raise RuntimeError(
            f"Estimated atom count {estimated_atoms} outside [{min_atoms}, {max_atoms}]. "
            "Adjust CHAIN_COUNT, PAPER_SIZE, MIN_ATOMS, or MAX_ATOMS."
        )

    pre_export = [
        summarize_rdkit_species_forcefield(peo, label="PEO", moltype_hint="PEO", charge_scale=charge_scale[0]),
        summarize_rdkit_species_forcefield(li, label="Li", moltype_hint="Li", charge_scale=charge_scale[1]),
        summarize_rdkit_species_forcefield(tfsi, label="TFSI", moltype_hint="TFSI", charge_scale=charge_scale[2]),
    ]

    ac = poly.amorphous_cell(
        [peo, li, tfsi],
        counts,
        charge_scale=charge_scale,
        density=initial_density_g_cm3,
        work_dir=build_dir,
    )

    equil_temp_k = max(400.0, float(case["target_temp_k"]))
    eq_hot = eq.EQ21step(ac, work_dir=case_root)
    ac = eq_hot.exec(temp=equil_temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
    analy_hot = eq_hot.analyze()
    analy_hot.get_all_prop(temp=equil_temp_k, press=press_bar, save=True)
    melt_ok = analy_hot.check_eq()

    additional_rounds = 0
    for _ in range(max_melt_additional):
        if melt_ok:
            break
        additional_rounds += 1
        eq_more = eq.Additional(ac, work_dir=case_root)
        ac = eq_more.exec(
            temp=equil_temp_k,
            press=press_bar,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            gpu_offload_mode=gpu_offload_mode,
        )
        analy_hot = eq_more.analyze()
        analy_hot.get_all_prop(temp=equil_temp_k, press=press_bar, save=True)
        melt_ok = analy_hot.check_eq()

    if abs(equil_temp_k - float(case["target_temp_k"])) > 1.0e-6:
        eq_target = eq.Additional(ac, work_dir=case_root)
        ac = eq_target.exec(
            temp=float(case["target_temp_k"]),
            press=press_bar,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            gpu_offload_mode=gpu_offload_mode,
        )
        analy_target = eq_target.analyze()
        analy_target.get_all_prop(temp=float(case["target_temp_k"]), press=press_bar, save=True)
        analy_target.check_eq()

    npt = eq.NPT(ac, work_dir=case_root)
    ac = npt.exec(
        temp=float(case["target_temp_k"]),
        press=press_bar,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        time=float(case["production_ns"]),
        gpu_offload_mode=gpu_offload_mode,
    )

    analy = npt.analyze()
    prop_data = analy.get_all_prop(temp=float(case["target_temp_k"]), press=press_bar, save=True)
    rdf = analy.rdf(center_mol=li)
    msd = analy.msd()
    sigma = analy.sigma(msd=msd, temp_k=float(case["target_temp_k"]), eh_mode="gmx_current_only")

    analysis_dir = case_root / "06_analysis"
    system_dir = case_root / "02_system"
    top_path = system_dir / "system.top"

    force_balance = collect_force_balance_report(
        system_dir=system_dir,
        top_path=top_path,
        cell=ac,
        species_pre_export=pre_export,
        moltype_hints={"polymer": "PEO", "cation": "Li", "anion": "TFSI"},
    )
    coordination = build_coordination_partition(rdf, polymer_moltype="PEO", anion_moltype="TFSI")
    literature = literature_band_peo_litfsi_jpcb2020(case_key)
    transport = build_transport_summary(
        msd=msd,
        sigma=sigma,
        rdf=rdf,
        polymer_moltype="PEO",
        anion_moltype="TFSI",
        thermo_xvg=analysis_dir / "thermo.xvg",
        literature_band=literature,
    )
    compare = build_benchmark_compare(
        force_balance_report=force_balance,
        coordination_partition=coordination,
        transport_summary=transport,
        charge_scale_polymer=float(case["polymer_charge_scale"]),
        charge_scale_li=float(case["li_charge_scale"]),
        charge_scale_anion=float(case["anion_charge_scale"]),
        production_ns=float(case["production_ns"]),
    )

    metadata = {
        "benchmark_name": "JPCB2020 PEO/LiTFSI charge-scaling reproduction",
        "paper_case": dict(case),
        "forcefield": {"polymer_and_tfsi": ff_variant, "cation": "MERZ", "charge_model": "RESP"},
        "eo_li_ratio": f"{float(case['effective_eo_li_ratio']):.3g}:1",
        "target_temp_k": float(case["target_temp_k"]),
        "normalized_inverse_temperature": float(case["normalized_inverse_temperature"]),
        "prod_ns": float(case["production_ns"]),
        "paper_production_ns": float(case["paper_production_ns"]),
        "chain_dp": int(case["chain_dp"]),
        "chain_count": int(case["chain_count"]),
        "salt_pairs": int(case["salt_pairs"]),
        "estimated_total_atoms": int(estimated_atoms),
        "charge_scale": {"polymer": charge_scale[0], "li": charge_scale[1], "tfsi": charge_scale[2]},
        "initial_density_g_cm3": float(initial_density_g_cm3),
        "melt_equilibrated": bool(melt_ok),
        "additional_rounds": int(additional_rounds),
        "gpu": gpu,
        "gpu_id": gpu_id,
        "gpu_offload_mode": gpu_offload_mode,
    }

    _dump_json(analysis_dir / "force_balance_report.json", force_balance)
    _dump_json(analysis_dir / "coordination_partition.json", coordination)
    _dump_json(analysis_dir / "transport_summary.json", transport)
    _dump_json(
        analysis_dir / "benchmark_compare.json",
        {
            "metadata": metadata,
            "compare": compare,
            "basic_properties": prop_data.get("basic_properties", {}),
        },
    )
    _dump_json(analysis_dir / "benchmark_metadata.json", metadata)
    return {"metadata": metadata, "compare": compare}


if __name__ == "__main__":
    ff, ff_variant = _make_gaff2()
    cation_ff = MERZ()
    cases = _resolved_cases()

    doctor(print_report=True)
    ensure_initialized()
    work_root = workdir(work_root, restart=restart_status)

    if dry_run:
        _write_dry_run_plan(work_root, cases, ff_variant)
    else:
        peo, li, tfsi = _prepare_species(work_root, ff, cation_ff)
        results = [_run_case(work_root, case, peo, li, tfsi, ff_variant) for case in cases]
        summary = {"cases": results}
        _dump_json(work_root / "jpcb2020_screening_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
