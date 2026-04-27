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
from yadonpy.sim.performance import resolve_io_analysis_policy
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


def _env_optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    return float(raw)


def _env_text(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return str(default)
    return str(raw).strip()


def _env_list(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _normalize_analysis_profile(profile: str) -> str:
    token = str(profile or "auto").strip().lower()
    if token in {"auto", "default"}:
        return "auto"
    if token in {"fast", "screening", "transport", "transport-fast", "transport_fast"}:
        return "transport_fast"
    if token in {"minimal", "min"}:
        return "minimal"
    if token == "full":
        return "full"
    raise ValueError("ANALYSIS_PROFILE must be auto, transport_fast, minimal, or full.")


def _json_cache_is_fresh(path: Path, deps: list[Path]) -> bool:
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return False
        dep_mtimes = [Path(dep).stat().st_mtime for dep in deps if Path(dep).exists()]
        return bool(dep_mtimes) and path.stat().st_mtime >= max(dep_mtimes)
    except Exception:
        return False


def _load_json_cache(path: Path, deps: list[Path]) -> dict | None:
    if not _json_cache_is_fresh(path, deps):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


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

literature_preset = _env_text("LITERATURE_PRESET", "")
if literature_preset and "JPCB_CASES" not in os.environ:
    preset_token = literature_preset
    for prefix in ("JPCB2020_", "JPCB_"):
        if preset_token.upper().startswith(prefix):
            preset_token = preset_token[len(prefix):]
            break
    case_labels = [preset_token]
else:
    case_labels = _env_list("JPCB_CASES", "P1.00S1.00,P1.00S0.75,P1.20S0.75")
paper_size = _env_bool("PAPER_SIZE", False)
dry_run = _env_bool("DRY_RUN", False)

target_mode = str(os.environ.get("TARGET_MODE", "normalized_inverse") or "normalized_inverse").strip()
target_temp_override = os.environ.get("TARGET_TEMP_K")
target_temp_k = float(target_temp_override) if target_temp_override and target_temp_override.strip() else None
normalized_inverse = _env_float("NORMALIZED_INVERSE", 5.4)
polymer_charge_scale_override = _env_optional_float("POLYMER_CHARGE_SCALE")
li_charge_scale_override = _env_optional_float("LI_CHARGE_SCALE")
anion_charge_scale_override = _env_optional_float("ANION_CHARGE_SCALE")

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
tfsi_resp_profile = _env_text("TFSI_RESP_PROFILE", "adaptive")

performance_profile = _env_text("PERFORMANCE_PROFILE", "auto")
analysis_profile_requested = _normalize_analysis_profile(_env_text("ANALYSIS_PROFILE", "auto"))
traj_ps_setting = _env_text("TRAJ_PS", os.environ.get("YADONPY_PROD_TRAJ_PS", "auto"))
energy_ps_setting = _env_text("ENERGY_PS", os.environ.get("YADONPY_PROD_ENERGY_PS", "auto"))
log_ps_setting = _env_text("LOG_PS", os.environ.get("YADONPY_PROD_LOG_PS", "auto"))
trr_ps_setting = os.environ.get("TRR_PS")
velocity_ps_setting = os.environ.get("VELOCITY_PS")
max_trajectory_frames = _env_int("MAX_TRAJECTORY_FRAMES", 50000)
max_atom_frames = _env_float("MAX_ATOM_FRAMES", 5.0e9)
rdf_frame_stride_setting = _env_text("RDF_FRAME_STRIDE", "auto")
rdf_bin_nm_setting = _env_text("RDF_BIN_NM", "auto")
rdf_rmax_nm_setting = _env_text("RDF_RMAX_NM", "auto")
resume_analysis = _env_bool("RESUME_ANALYSIS", True)
msd_geometry = _env_text("MSD_GEOMETRY", "auto")
msd_unwrap = _env_text("MSD_UNWRAP", "auto")
msd_drift = _env_text("MSD_DRIFT", "auto")
dielectric_analysis = _env_bool("DIELECTRIC_ANALYSIS", True)
dielectric_group = _env_text("DIELECTRIC_GROUP", "peo")
dielectric_dt_ps_raw = os.environ.get("DIELECTRIC_DT_PS")
dielectric_dt_ps = (
    float(dielectric_dt_ps_raw)
    if dielectric_dt_ps_raw is not None and str(dielectric_dt_ps_raw).strip()
    else None
)

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
        overrides = {}
        if polymer_charge_scale_override is not None:
            case["polymer_charge_scale"] = float(polymer_charge_scale_override)
            overrides["polymer_charge_scale"] = float(polymer_charge_scale_override)
        if li_charge_scale_override is not None:
            case["li_charge_scale"] = float(li_charge_scale_override)
            overrides["li_charge_scale"] = float(li_charge_scale_override)
        if anion_charge_scale_override is not None:
            case["anion_charge_scale"] = float(anion_charge_scale_override)
            overrides["anion_charge_scale"] = float(anion_charge_scale_override)
        if overrides:
            li_scale = float(case.get("li_charge_scale", case.get("salt_charge_scale", 1.0)))
            anion_scale = float(case.get("anion_charge_scale", case.get("salt_charge_scale", 1.0)))
            if abs(li_scale - anion_scale) <= 1.0e-12:
                case["salt_charge_scale"] = li_scale
            case["charge_scale_overrides"] = overrides
        out.append(case)
    return out


def _write_dry_run_plan(root: Path, cases: list[dict], ff_variant: str) -> None:
    payload = {
        "benchmark_name": "JPCB2020 PEO/LiTFSI charge-scaling reproduction",
        "forcefield": {
            "polymer_and_tfsi": ff_variant,
            "cation": "MERZ",
            "charge_model": "RESP",
            "tfsi_resp_profile": tfsi_resp_profile,
        },
        "target_cases": cases,
        "available_cases": jpcb2020_peo_litfsi_cases(),
        "execution": {
            "dry_run": True,
            "performance_profile": performance_profile,
            "analysis_profile_requested": analysis_profile_requested,
            "resume_analysis": bool(resume_analysis),
            "dielectric_analysis": bool(dielectric_analysis),
            "dielectric_group": dielectric_group,
            "literature_preset": literature_preset or None,
            "charge_scale_overrides": {
                "polymer": polymer_charge_scale_override,
                "li": li_charge_scale_override,
                "anion": anion_charge_scale_override,
            },
            "rdf": {
                "bin_nm": rdf_bin_nm_setting,
                "r_max_nm": rdf_rmax_nm_setting,
                "frame_stride": rdf_frame_stride_setting,
            },
            "output_cadence": {
                "traj_ps": traj_ps_setting,
                "energy_ps": energy_ps_setting,
                "log_ps": log_ps_setting,
                "trr_ps": trr_ps_setting,
                "velocity_ps": velocity_ps_setting,
                "max_trajectory_frames": max_trajectory_frames,
                "max_atom_frames": max_atom_frames,
            },
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
            resp_profile=tfsi_resp_profile,
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
    io_policy = resolve_io_analysis_policy(
        prod_ns=float(case["production_ns"]),
        atom_count=int(estimated_atoms),
        performance_profile=performance_profile,
        analysis_profile=analysis_profile_requested,
        traj_ps=traj_ps_setting,
        energy_ps=energy_ps_setting,
        log_ps=log_ps_setting,
        trr_ps=trr_ps_setting,
        velocity_ps=velocity_ps_setting,
        rdf_frame_stride=rdf_frame_stride_setting,
        rdf_rmax_nm=rdf_rmax_nm_setting,
        rdf_bin_nm=rdf_bin_nm_setting,
        msd_selected_species=["PEO", "Li", "TFSI"],
        max_trajectory_frames=max_trajectory_frames,
        max_atom_frames=max_atom_frames,
    )
    analysis_profile = io_policy.analysis_profile
    analysis_fast = analysis_profile in {"transport_fast", "minimal"}

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
        traj_ps=io_policy.traj_ps,
        energy_ps=io_policy.energy_ps,
        log_ps=io_policy.log_ps,
        trr_ps=io_policy.trr_ps,
        velocity_ps=io_policy.velocity_ps,
        performance_profile=io_policy.performance_profile,
        analysis_profile=io_policy.analysis_profile,
        max_trajectory_frames=io_policy.max_trajectory_frames,
        max_atom_frames=io_policy.max_atom_frames,
        gpu_offload_mode=gpu_offload_mode,
    )

    analy = npt.analyze()
    analysis_metadata = {
        "analysis_profile": analysis_profile,
        "performance_policy": io_policy.to_dict(),
        "resume_analysis": bool(resume_analysis),
        "include_polymer_metrics": bool(io_policy.include_polymer_metrics),
        "rdf": {
            "bin_nm": float(io_policy.rdf_bin_nm),
            "r_max_nm": io_policy.rdf_rmax_nm,
            "frame_stride": int(io_policy.rdf_frame_stride),
            "site_filter": (
                ["ether_oxygen", "sulfonyl_oxygen", "anion_nitrogen"]
                if analysis_fast
                else None
            ),
        },
        "msd": {
            "selected_species": io_policy.msd_selected_species if analysis_fast else None,
            "default_metric_only": bool(io_policy.msd_default_metric_only),
            "geometry": msd_geometry,
            "unwrap": msd_unwrap,
            "drift": msd_drift,
        },
        "dielectric": {
            "enabled": bool(dielectric_analysis),
            "group": dielectric_group,
            "dt_ps": dielectric_dt_ps,
            "method": "gmx dipoles",
        },
    }
    prop_data = analy.get_all_prop(
        temp=float(case["target_temp_k"]),
        press=press_bar,
        save=True,
        include_polymer_metrics=bool(io_policy.include_polymer_metrics),
        analysis_profile=analysis_profile,
    )
    rdf = analy.rdf(
        center_mol=li,
        analysis_profile=analysis_profile,
        bin_nm=float(io_policy.rdf_bin_nm),
        r_max_nm=io_policy.rdf_rmax_nm,
        frame_stride=int(io_policy.rdf_frame_stride),
        resume=resume_analysis,
    )
    msd = analy.msd(
        analysis_profile=analysis_profile,
        geometry=msd_geometry,
        unwrap=msd_unwrap,
        drift=msd_drift,
        resume=resume_analysis,
    )
    if dielectric_analysis:
        try:
            dielectric = analy.dielectric(
                temp_k=float(prop_data.get("basic_properties", {}).get("temperature_K") or case["target_temp_k"]),
                group=dielectric_group,
                dt_ps=dielectric_dt_ps,
                resume=resume_analysis,
            )
        except Exception as exc:
            dielectric = {
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "note": (
                    "Dielectric analysis is optional. If this failed with a TPR version error, set "
                    "YADONPY_GMX_CMD to the same GROMACS major version used for production."
                ),
            }
    else:
        dielectric = {"status": "disabled"}

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
    transport_cache = None
    if resume_analysis:
        transport_cache = _load_json_cache(
            analysis_dir / "transport_summary.json",
            [
                analysis_dir / "rdf_first_shell.json",
                analysis_dir / "msd.json",
                analysis_dir / "thermo.xvg",
            ],
        )
        if isinstance(transport_cache, dict):
            cached_meta = transport_cache.get("analysis_metadata")
            cached_profile = (
                cached_meta.get("analysis_profile")
                if isinstance(cached_meta, dict)
                else transport_cache.get("analysis_profile")
            )
            if str(cached_profile or "") != analysis_profile:
                transport_cache = None
    if isinstance(transport_cache, dict):
        transport = transport_cache
    else:
        sigma = analy.sigma(msd=msd, temp_k=float(case["target_temp_k"]), eh_mode="gmx_current_only")
        transport = build_transport_summary(
            msd=msd,
            sigma=sigma,
            rdf=rdf,
            polymer_moltype="PEO",
            anion_moltype="TFSI",
            thermo_xvg=analysis_dir / "thermo.xvg",
            literature_band=literature,
            analysis_metadata=analysis_metadata,
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
        "forcefield": {
            "polymer_and_tfsi": ff_variant,
            "cation": "MERZ",
            "charge_model": "RESP",
            "tfsi_resp_profile": tfsi_resp_profile,
        },
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
        "analysis": analysis_metadata,
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
            "dielectric": dielectric,
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
