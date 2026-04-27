from __future__ import annotations

import json
import os
from pathlib import Path

from yadonpy.core import poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
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
    literature_band_peo_litfsi_60c,
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


def _apply_literature_preset() -> dict[str, float | int | str] | None:
    preset = str(os.environ.get("LITERATURE_PRESET", "") or "").strip()
    if not preset:
        return None
    key = preset.upper()
    presets: dict[str, dict[str, float | int | str]] = {
        "JPCB2020_P1.00S1.00": {
            "paper_label": "P1.00S1.00",
            "polymer_charge_scale": 1.0,
            "li_charge_scale": 1.0,
            "anion_charge_scale": 1.0,
            "chain_dp": 25,
            "chain_count": 96,
            "salt_pairs": 192,
            "melt_temp_k": 400.0,
            "target_temp_k": 333.15,
        },
        "JPCB2020_P1.00S0.75": {
            "paper_label": "P1.00S0.75",
            "polymer_charge_scale": 1.0,
            "li_charge_scale": 0.75,
            "anion_charge_scale": 0.75,
            "chain_dp": 25,
            "chain_count": 96,
            "salt_pairs": 192,
            "melt_temp_k": 400.0,
            "target_temp_k": 333.15,
        },
        "JPCB2020_P1.20S0.75": {
            "paper_label": "P1.20S0.75",
            "polymer_charge_scale": 1.2,
            "li_charge_scale": 0.75,
            "anion_charge_scale": 0.75,
            "chain_dp": 25,
            "chain_count": 96,
            "salt_pairs": 192,
            "melt_temp_k": 400.0,
            "target_temp_k": 333.15,
        },
    }
    if key not in presets:
        raise ValueError(
            f"Unsupported LITERATURE_PRESET={preset!r}. "
            "Expected one of: JPCB2020_P1.00S1.00, JPCB2020_P1.00S0.75, JPCB2020_P1.20S0.75."
        )
    return {"preset_name": key, **presets[key]}


BASE_DIR = Path(__file__).resolve().parent

restart_status = _env_bool("RESTART_STATUS", False)
set_run_options(restart=restart_status)

ff = GAFF2_mod()
cation_ff = MERZ()

melt_temp_k = _env_float("MELT_TEMP_K", 353.15)
target_temp_k = _env_float("TARGET_TEMP_K", 333.15)
press_bar = _env_float("PRESS_BAR", 1.0)
prod_ns = _env_float("PROD_NS", 10.0)
initial_density_g_cm3 = _env_float("INITIAL_DENSITY_G_CM3", 0.05)

mpi = _env_int("MPI", 1)
omp = _env_int("OMP", 16)
gpu = _env_int("GPU", 1)
gpu_id = _env_int("GPU_ID", 0)
omp_psi4 = _env_int("OMP_PSI4", 32)
mem_mb = _env_int("MEM_MB", 20000)

chain_dp = _env_int("CHAIN_DP", 40)
chain_count = _env_int("CHAIN_COUNT", 32)
salt_pairs = _env_int("SALT_PAIRS", 64)
cool_rounds = _env_int("COOL_ROUNDS", 1)
max_melt_additional = _env_int("MAX_MELT_ADDITIONAL", 2)

li_charge_scale = _env_float("LI_CHARGE_SCALE", 1.0)
anion_charge_scale = _env_float("ANION_CHARGE_SCALE", 1.0)
polymer_charge_scale = _env_float("POLYMER_CHARGE_SCALE", 1.0)

literature_preset = _apply_literature_preset()
if literature_preset is not None:
    melt_temp_k = float(literature_preset["melt_temp_k"])
    target_temp_k = float(literature_preset["target_temp_k"])
    chain_dp = int(literature_preset["chain_dp"])
    chain_count = int(literature_preset["chain_count"])
    salt_pairs = int(literature_preset["salt_pairs"])
    polymer_charge_scale = float(literature_preset["polymer_charge_scale"])
    li_charge_scale = float(literature_preset["li_charge_scale"])
    anion_charge_scale = float(literature_preset["anion_charge_scale"])

work_dir_name = os.environ.get("WORK_DIR_NAME", "benchmark_peo_litfsi_60c_work")
work_root = Path(os.environ.get("WORK_DIR", str(BASE_DIR / work_dir_name))).resolve()


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    work_root = workdir(work_root, restart=restart_status)
    build_dir = work_root.child("00_build_cell")
    poly_rw_dir = work_root.child("poly_rw")
    poly_term_dir = work_root.child("poly_term")

    monomer = utils.mol_from_smiles(r"*CCO*")
    monomer, _ = qm.conformation_search(
        monomer,
        ff=ff,
        work_dir=work_root,
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
        work_dir=work_root,
        omp=omp_psi4,
        memory=mem_mb,
        log_name=None,
    )

    ter = utils.mol_from_smiles("[H][*]")
    qm.assign_charges(
        ter,
        charge="RESP",
        opt=True,
        work_dir=work_root,
        omp=omp_psi4,
        memory=mem_mb,
        log_name=None,
    )

    peo = poly.random_copolymerize_rw(
        [monomer],
        chain_dp,
        ratio=[1.0],
        tacticity="atactic",
        name="PEO",
        work_dir=poly_rw_dir,
    )
    peo = poly.terminate_rw(peo, ter, name="PEO", work_dir=poly_term_dir)
    peo = ff.ff_assign(peo)
    if not peo:
        raise RuntimeError("Failed to assign force field parameters for PEO benchmark chain.")

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
            "Precompute TFSI first, then rerun the benchmark."
        ) from exc
    if not tfsi:
        raise RuntimeError("Failed to assign force field parameters for TFSI.")

    counts = [chain_count, salt_pairs, salt_pairs]
    charge_scale = [polymer_charge_scale, li_charge_scale, anion_charge_scale]

    estimated_atoms = chain_count * int(peo.GetNumAtoms()) + salt_pairs * int(li.GetNumAtoms()) + salt_pairs * int(tfsi.GetNumAtoms())
    if estimated_atoms < 10000 or estimated_atoms > 30000:
        raise RuntimeError(
            f"Benchmark atom count must stay within 10k-30k; got estimated_total_atoms={estimated_atoms} "
            f"(chain_count={chain_count}, salt_pairs={salt_pairs}, chain_dp={chain_dp})."
        )

    pre_export = [
        summarize_rdkit_species_forcefield(peo, label="PEO", moltype_hint="PEO", charge_scale=polymer_charge_scale),
        summarize_rdkit_species_forcefield(li, label="Li", moltype_hint="Li", charge_scale=li_charge_scale),
        summarize_rdkit_species_forcefield(tfsi, label="TFSI", moltype_hint="TFSI", charge_scale=anion_charge_scale),
    ]

    ac = poly.amorphous_cell(
        [peo, li, tfsi],
        counts,
        charge_scale=charge_scale,
        density=initial_density_g_cm3,
        work_dir=build_dir,
    )

    eq_hot = eq.EQ21step(ac, work_dir=work_root)
    ac = eq_hot.exec(temp=melt_temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
    analy_hot = eq_hot.analyze()
    analy_hot.get_all_prop(temp=melt_temp_k, press=press_bar, save=True)
    melt_ok = analy_hot.check_eq()

    for _ in range(max_melt_additional):
        if melt_ok:
            break
        eq_more = eq.Additional(ac, work_dir=work_root)
        ac = eq_more.exec(temp=melt_temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
        analy_hot = eq_more.analyze()
        analy_hot.get_all_prop(temp=melt_temp_k, press=press_bar, save=True)
        melt_ok = analy_hot.check_eq()

    for _ in range(cool_rounds):
        eq_cool = eq.Additional(ac, work_dir=work_root)
        ac = eq_cool.exec(temp=target_temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
        analy_cool = eq_cool.analyze()
        analy_cool.get_all_prop(temp=target_temp_k, press=press_bar, save=True)
        analy_cool.check_eq()

    npt = eq.NPT(ac, work_dir=work_root)
    ac = npt.exec(temp=target_temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=prod_ns)

    analy = npt.analyze()
    prop_data = analy.get_all_prop(temp=target_temp_k, press=press_bar, save=True)
    rdf = analy.rdf(center_mol=li)
    msd = analy.msd()
    sigma = analy.sigma(msd=msd, temp_k=target_temp_k, eh_mode="gmx_current_only")

    analysis_dir = work_root / "06_analysis"
    system_dir = work_root / "02_system"
    top_path = system_dir / "system.top"

    force_balance = collect_force_balance_report(
        system_dir=system_dir,
        top_path=top_path,
        cell=ac,
        species_pre_export=pre_export,
        moltype_hints={"polymer": "PEO", "cation": "Li", "anion": "TFSI"},
    )
    coordination = build_coordination_partition(rdf, polymer_moltype="PEO", anion_moltype="TFSI")
    transport = build_transport_summary(
        msd=msd,
        sigma=sigma,
        rdf=rdf,
        polymer_moltype="PEO",
        anion_moltype="TFSI",
        thermo_xvg=analysis_dir / "thermo.xvg",
        literature_band=literature_band_peo_litfsi_60c(),
    )
    compare = build_benchmark_compare(
        force_balance_report=force_balance,
        coordination_partition=coordination,
        transport_summary=transport,
        charge_scale_polymer=polymer_charge_scale,
        charge_scale_li=li_charge_scale,
        charge_scale_anion=anion_charge_scale,
        production_ns=prod_ns,
    )

    effective_eo_li_ratio = float(chain_dp * chain_count) / max(float(salt_pairs), 1.0)
    metadata = {
        "benchmark_name": "PEO/LiTFSI 60C",
        "literature_preset": dict(literature_preset) if literature_preset is not None else None,
        "eo_li_ratio": f"{effective_eo_li_ratio:.3g}:1",
        "melt_temp_k": melt_temp_k,
        "target_temp_k": target_temp_k,
        "prod_ns": prod_ns,
        "chain_dp": chain_dp,
        "chain_count": chain_count,
        "salt_pairs": salt_pairs,
        "effective_eo_li_ratio": effective_eo_li_ratio,
        "estimated_total_atoms": estimated_atoms,
        "charge_scale": {"polymer": polymer_charge_scale, "li": li_charge_scale, "tfsi": anion_charge_scale},
        "gpu": gpu,
        "gpu_id": gpu_id,
        "melt_equilibrated": bool(melt_ok),
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

    print("[BENCHMARK] PEO/LiTFSI 60C benchmark completed")
    print(json.dumps({"metadata": metadata, "compare": compare}, indent=2, ensure_ascii=False))
