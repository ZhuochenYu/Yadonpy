from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

import json  # 导入本例需要的库或 yadonpy 接口。
import os  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.core import poly, utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2 import GAFF2  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2_mod import GAFF2_mod  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.merz import MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim import qm  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.benchmarking import (  # 导入本例需要的库或 yadonpy 接口。
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
from yadonpy.sim.performance import resolve_io_analysis_policy  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.preset import eq  # 导入本例需要的库或 yadonpy 接口。


def _env_bool(name: str, default: bool) -> bool:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None:  # 根据当前状态决定是否进入该分支。
        return bool(default)  # 返回该辅助函数的结果。
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}  # 返回该辅助函数的结果。


def _env_int(name: str, default: int) -> int:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None or not str(raw).strip():  # 根据当前状态决定是否进入该分支。
        return int(default)  # 返回该辅助函数的结果。
    return int(raw)  # 返回该辅助函数的结果。


def _env_float(name: str, default: float) -> float:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None or not str(raw).strip():  # 根据当前状态决定是否进入该分支。
        return float(default)  # 返回该辅助函数的结果。
    return float(raw)  # 返回该辅助函数的结果。


def _env_optional_float(name: str) -> float | None:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None or not str(raw).strip():  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    return float(raw)  # 返回该辅助函数的结果。


def _env_text(name: str, default: str) -> str:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None or not str(raw).strip():  # 根据当前状态决定是否进入该分支。
        return str(default)  # 返回该辅助函数的结果。
    return str(raw).strip()  # 返回该辅助函数的结果。


def _env_list(name: str, default: str) -> list[str]:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name, default)  # 设置中间变量或可调参数，供后续工作流使用。
    return [item.strip() for item in str(raw).split(",") if item.strip()]  # 返回该辅助函数的结果。


def _normalize_analysis_profile(profile: str) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    token = str(profile or "auto").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if token in {"auto", "default"}:  # 根据当前状态决定是否进入该分支。
        return "auto"  # 返回该辅助函数的结果。
    if token in {"fast", "screening", "transport", "transport-fast", "transport_fast"}:  # 根据当前状态决定是否进入该分支。
        return "transport_fast"  # 返回该辅助函数的结果。
    if token in {"minimal", "min"}:  # 根据当前状态决定是否进入该分支。
        return "minimal"  # 返回该辅助函数的结果。
    if token == "full":  # 根据当前状态决定是否进入该分支。
        return "full"  # 返回该辅助函数的结果。
    raise ValueError("ANALYSIS_PROFILE must be auto, transport_fast, minimal, or full.")  # 关键步骤失败时立即报错，避免继续生成错误结果。


def _json_cache_is_fresh(path: Path, deps: list[Path]) -> bool:  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        if not path.exists() or path.stat().st_size <= 0:  # 根据当前状态决定是否进入该分支。
            return False  # 返回该辅助函数的结果。
        dep_mtimes = [Path(dep).stat().st_mtime for dep in deps if Path(dep).exists()]  # 设置中间变量或可调参数，供后续工作流使用。
        return bool(dep_mtimes) and path.stat().st_mtime >= max(dep_mtimes)  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return False  # 返回该辅助函数的结果。


def _load_json_cache(path: Path, deps: list[Path]) -> dict | None:  # 定义本例内部辅助函数，组织重复步骤。
    if not _json_cache_is_fresh(path, deps):  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        payload = json.loads(path.read_text(encoding="utf-8"))  # 设置中间变量或可调参数，供后续工作流使用。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return None  # 返回该辅助函数的结果。
    return payload if isinstance(payload, dict) else None  # 返回该辅助函数的结果。


def _make_gaff2():  # 定义本例内部辅助函数，组织重复步骤。
    variant = str(os.environ.get("GAFF2_VARIANT", "classic") or "classic").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if variant in {"classic", "gaff2", "orig", "original"}:  # 根据当前状态决定是否进入该分支。
        return GAFF2(), "gaff2"  # 返回该辅助函数的结果。
    if variant in {"mod", "gaff2_mod", "modified"}:  # 根据当前状态决定是否进入该分支。
        return GAFF2_mod(), "gaff2_mod"  # 返回该辅助函数的结果。
    raise ValueError("GAFF2_VARIANT must be classic or mod.")  # 关键步骤失败时立即报错，避免继续生成错误结果。


BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。

restart_status = _env_bool("RESTART_STATUS", False)  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

literature_preset = _env_text("LITERATURE_PRESET", "")  # 设置中间变量或可调参数，供后续工作流使用。
if literature_preset and "JPCB_CASES" not in os.environ:  # 根据当前状态决定是否进入该分支。
    preset_token = literature_preset  # 设置中间变量或可调参数，供后续工作流使用。
    for prefix in ("JPCB2020_", "JPCB_"):  # 遍历当前工作流中的一组对象或任务。
        if preset_token.upper().startswith(prefix):  # 根据当前状态决定是否进入该分支。
            preset_token = preset_token[len(prefix):]  # 设置中间变量或可调参数，供后续工作流使用。
            break
    case_labels = [preset_token]  # 设置中间变量或可调参数，供后续工作流使用。
else:  # 处理前面条件都不满足的情况。
    case_labels = _env_list("JPCB_CASES", "P1.00S1.00,P1.00S0.75,P1.20S0.75")  # 设置中间变量或可调参数，供后续工作流使用。
paper_size = _env_bool("PAPER_SIZE", False)  # 设置中间变量或可调参数，供后续工作流使用。
dry_run = _env_bool("DRY_RUN", False)  # 设置中间变量或可调参数，供后续工作流使用。

target_mode = str(os.environ.get("TARGET_MODE", "normalized_inverse") or "normalized_inverse").strip()  # 设置中间变量或可调参数，供后续工作流使用。
target_temp_override = os.environ.get("TARGET_TEMP_K")  # 设置中间变量或可调参数，供后续工作流使用。
target_temp_k = float(target_temp_override) if target_temp_override and target_temp_override.strip() else None  # 设置中间变量或可调参数，供后续工作流使用。
normalized_inverse = _env_float("NORMALIZED_INVERSE", 5.4)  # 设置中间变量或可调参数，供后续工作流使用。
polymer_charge_scale_override = _env_optional_float("POLYMER_CHARGE_SCALE")  # 设置中间变量或可调参数，供后续工作流使用。
li_charge_scale_override = _env_optional_float("LI_CHARGE_SCALE")  # 设置中间变量或可调参数，供后续工作流使用。
anion_charge_scale_override = _env_optional_float("ANION_CHARGE_SCALE")  # 设置中间变量或可调参数，供后续工作流使用。

chain_count = _env_int("CHAIN_COUNT", 96)  # 设置中间变量或可调参数，供后续工作流使用。
prod_ns_default = _env_float("PROD_NS", 20.0)  # 设置中间变量或可调参数，供后续工作流使用。
press_bar = _env_float("PRESS_BAR", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
initial_density_g_cm3 = _env_float("INITIAL_DENSITY_G_CM3", 0.65)  # 设置中间变量或可调参数，供后续工作流使用。
max_melt_additional = _env_int("MAX_MELT_ADDITIONAL", 4)  # 设置中间变量或可调参数，供后续工作流使用。
max_atoms = _env_int("MAX_ATOMS", 30000 if not paper_size else 60000)  # 设置中间变量或可调参数，供后续工作流使用。
min_atoms = _env_int("MIN_ATOMS", 10000 if not _env_bool("SMOKE", False) else 1)  # 设置中间变量或可调参数，供后续工作流使用。

mpi = _env_int("MPI", 1)  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = _env_int("OMP", 14)  # 设置每个 rank 的 OpenMP 线程数。
gpu = _env_int("GPU", 1)  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = _env_int("GPU_ID", 0)  # 选择 GPU 设备编号，多卡节点可修改。
gpu_offload_mode = _env_text("GPU_OFFLOAD_MODE", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
omp_psi4 = _env_int("OMP_PSI4", 32)  # 设置 Psi4/OpenMP 核数。
mem_mb = _env_int("MEM_MB", 20000)  # 设置量子化学内存 MB。
tfsi_resp_profile = _env_text("TFSI_RESP_PROFILE", "adaptive")  # 设置中间变量或可调参数，供后续工作流使用。

performance_profile = _env_text("PERFORMANCE_PROFILE", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
analysis_profile_requested = _normalize_analysis_profile(_env_text("ANALYSIS_PROFILE", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
trajectory_format_setting = _env_text("TRAJECTORY_FORMAT", os.environ.get("YADONPY_TRAJECTORY_FORMAT", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
traj_ps_setting = _env_text("TRAJ_PS", os.environ.get("YADONPY_PROD_TRAJ_PS", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
energy_ps_setting = _env_text("ENERGY_PS", os.environ.get("YADONPY_PROD_ENERGY_PS", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
log_ps_setting = _env_text("LOG_PS", os.environ.get("YADONPY_PROD_LOG_PS", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
trr_ps_setting = os.environ.get("TRR_PS")  # 设置中间变量或可调参数，供后续工作流使用。
velocity_ps_setting = os.environ.get("VELOCITY_PS")  # 设置中间变量或可调参数，供后续工作流使用。
max_trajectory_frames = _env_int("MAX_TRAJECTORY_FRAMES", 50000)  # 设置中间变量或可调参数，供后续工作流使用。
max_atom_frames = _env_float("MAX_ATOM_FRAMES", 5.0e9)  # 设置中间变量或可调参数，供后续工作流使用。
rdf_frame_stride_setting = _env_text("RDF_FRAME_STRIDE", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
rdf_bin_nm_setting = _env_text("RDF_BIN_NM", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
rdf_rmax_nm_setting = _env_text("RDF_RMAX_NM", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
resume_analysis = _env_bool("RESUME_ANALYSIS", True)  # 设置中间变量或可调参数，供后续工作流使用。
msd_geometry = _env_text("MSD_GEOMETRY", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
msd_unwrap = _env_text("MSD_UNWRAP", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
msd_drift = _env_text("MSD_DRIFT", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
dielectric_analysis = _env_bool("DIELECTRIC_ANALYSIS", True)  # 设置中间变量或可调参数，供后续工作流使用。
dielectric_group = _env_text("DIELECTRIC_GROUP", "peo")  # 设置中间变量或可调参数，供后续工作流使用。
dielectric_dt_ps_raw = os.environ.get("DIELECTRIC_DT_PS")  # 设置中间变量或可调参数，供后续工作流使用。
dielectric_dt_ps = (  # 设置中间变量或可调参数，供后续工作流使用。
    float(dielectric_dt_ps_raw)
    if dielectric_dt_ps_raw is not None and str(dielectric_dt_ps_raw).strip()  # 根据当前状态决定是否进入该分支。
    else None
)

work_dir_name = os.environ.get("WORK_DIR_NAME", "benchmark_peo_litfsi_jpcb2020_work")  # 设置中间变量或可调参数，供后续工作流使用。
work_root = Path(os.environ.get("WORK_DIR", str(BASE_DIR / work_dir_name))).resolve()  # 设置中间变量或可调参数，供后续工作流使用。


def _resolved_cases() -> list[dict]:  # 定义本例内部辅助函数，组织重复步骤。
    out = []  # 设置中间变量或可调参数，供后续工作流使用。
    for label in case_labels:  # 遍历当前工作流中的一组对象或任务。
        case = resolve_jpcb2020_peo_litfsi_case(  # 设置中间变量或可调参数，供后续工作流使用。
            label,
            chain_count=chain_count,  # 设置中间变量或可调参数，供后续工作流使用。
            target_mode=target_mode,  # 设置中间变量或可调参数，供后续工作流使用。
            normalized_inverse=normalized_inverse,  # 设置中间变量或可调参数，供后续工作流使用。
            target_temp_k=target_temp_k,  # 设置中间变量或可调参数，供后续工作流使用。
            production_ns=prod_ns_default,  # 设置中间变量或可调参数，供后续工作流使用。
            paper_size=paper_size,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        overrides = {}  # 设置中间变量或可调参数，供后续工作流使用。
        if polymer_charge_scale_override is not None:  # 根据当前状态决定是否进入该分支。
            case["polymer_charge_scale"] = float(polymer_charge_scale_override)  # 设置中间变量或可调参数，供后续工作流使用。
            overrides["polymer_charge_scale"] = float(polymer_charge_scale_override)  # 设置中间变量或可调参数，供后续工作流使用。
        if li_charge_scale_override is not None:  # 根据当前状态决定是否进入该分支。
            case["li_charge_scale"] = float(li_charge_scale_override)  # 设置中间变量或可调参数，供后续工作流使用。
            overrides["li_charge_scale"] = float(li_charge_scale_override)  # 设置中间变量或可调参数，供后续工作流使用。
        if anion_charge_scale_override is not None:  # 根据当前状态决定是否进入该分支。
            case["anion_charge_scale"] = float(anion_charge_scale_override)  # 设置中间变量或可调参数，供后续工作流使用。
            overrides["anion_charge_scale"] = float(anion_charge_scale_override)  # 设置中间变量或可调参数，供后续工作流使用。
        if overrides:  # 根据当前状态决定是否进入该分支。
            li_scale = float(case.get("li_charge_scale", case.get("salt_charge_scale", 1.0)))  # 设置中间变量或可调参数，供后续工作流使用。
            anion_scale = float(case.get("anion_charge_scale", case.get("salt_charge_scale", 1.0)))  # 设置中间变量或可调参数，供后续工作流使用。
            if abs(li_scale - anion_scale) <= 1.0e-12:  # 根据当前状态决定是否进入该分支。
                case["salt_charge_scale"] = li_scale  # 设置中间变量或可调参数，供后续工作流使用。
            case["charge_scale_overrides"] = overrides  # 设置中间变量或可调参数，供后续工作流使用。
        out.append(case)
    return out  # 返回该辅助函数的结果。


def _write_dry_run_plan(root: Path, cases: list[dict], ff_variant: str) -> None:  # 定义本例内部辅助函数，组织重复步骤。
    payload = {  # 设置中间变量或可调参数，供后续工作流使用。
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
    print(json.dumps(payload, indent=2, ensure_ascii=False))  # 打印关键路径或状态，便于人工检查。


def _prepare_species(root, ff, cation_ff):  # 定义本例内部辅助函数，组织重复步骤。
    species_dir = root.child("00_species")  # 设置中间变量或可调参数，供后续工作流使用。
    poly_rw_dir = species_dir.child("poly_rw")  # 设置中间变量或可调参数，供后续工作流使用。
    poly_term_dir = species_dir.child("poly_term")  # 设置中间变量或可调参数，供后续工作流使用。

    monomer = utils.mol_from_smiles(r"*CCO*")  # 从 SMILES 直接构造 RDKit 分子。
    monomer, _ = qm.conformation_search(  # 执行构象搜索/几何优化，为 RESP 做准备。
        monomer,
        ff=ff,  # 选择有机分子/聚合物/部分无机离子的力场对象。
        work_dir=species_dir,  # 设置本例输出目录。
        psi4_omp=omp_psi4,  # 设置中间变量或可调参数，供后续工作流使用。
        mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
        memory=mem_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        log_name=None,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        monomer,
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        opt=False,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=species_dir,  # 设置本例输出目录。
        omp=omp_psi4,  # 设置每个 rank 的 OpenMP 线程数。
        memory=mem_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        log_name=None,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    ter = utils.mol_from_smiles("[H][*]")  # 从 SMILES 直接构造 RDKit 分子。
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        ter,
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        opt=True,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=species_dir,  # 设置本例输出目录。
        omp=omp_psi4,  # 设置每个 rank 的 OpenMP 线程数。
        memory=mem_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        log_name=None,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    max_dp = max(int(case["chain_dp"]) for case in _resolved_cases())  # 设置中间变量或可调参数，供后续工作流使用。
    peo = poly.random_copolymerize_rw(  # 用随机游走生成聚合物链。
        [monomer],
        max_dp,
        ratio=[1.0],  # 设置共聚组成比例。
        tacticity="atactic",  # 设置聚合物立构。
        name="PEO",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=poly_rw_dir,  # 设置本例输出目录。
    )
    peo = poly.terminate_rw(peo, ter, name="PEO", work_dir=poly_term_dir)  # 给聚合物链加端基。
    peo = ff.ff_assign(peo)  # 分配力场参数并写入分子属性。
    if not peo:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Failed to assign GAFF2 force-field parameters for PEO.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    li = cation_ff.mol("[Li+]")  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    li = cation_ff.ff_assign(li)  # 分配力场参数并写入分子属性。
    if not li:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Failed to assign MERZ parameters for Li+.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        tfsi = ff.mol(  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
            "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            resp_profile=tfsi_resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
            require_ready=True,  # 要求 MolDB 物种必须已准备好。
            prefer_db=True,  # 优先从 MolDB 读取已有结果。
        )
        tfsi = ff.ff_assign(tfsi)  # 分配力场参数并写入分子属性。
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
            "This benchmark requires a ready RESP-backed TFSI record in MolDB. "
            "Precompute TFSI with GAFF2/RESP first, then rerun this script."
        ) from exc
    if not tfsi:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Failed to assign GAFF2 force-field parameters for TFSI.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return peo, li, tfsi  # 返回该辅助函数的结果。


def _run_case(root, case: dict, peo, li, tfsi, ff_variant: str) -> dict:  # 定义本例内部辅助函数，组织重复步骤。
    case_key = str(case["case_key"])  # 设置中间变量或可调参数，供后续工作流使用。
    case_root = root.child(f"case_{case_key.replace('.', 'p')}")  # 设置中间变量或可调参数，供后续工作流使用。
    build_dir = case_root.child("00_build_cell")  # 设置中间变量或可调参数，供后续工作流使用。

    counts = [int(case["chain_count"]), int(case["salt_pairs"]), int(case["salt_pairs"])]  # 设置各 species 的数量；顺序必须和 species 列表一致。
    charge_scale = [  # 设置电荷缩放系数；1.0 表示全电荷模型。
        float(case["polymer_charge_scale"]),
        float(case["li_charge_scale"]),
        float(case["anion_charge_scale"]),
    ]

    estimated_atoms = (  # 设置中间变量或可调参数，供后续工作流使用。
        counts[0] * int(peo.GetNumAtoms())
        + counts[1] * int(li.GetNumAtoms())
        + counts[2] * int(tfsi.GetNumAtoms())
    )
    if estimated_atoms < min_atoms or estimated_atoms > max_atoms:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
            f"Estimated atom count {estimated_atoms} outside [{min_atoms}, {max_atoms}]. "
            "Adjust CHAIN_COUNT, PAPER_SIZE, MIN_ATOMS, or MAX_ATOMS."
        )
    io_policy = resolve_io_analysis_policy(  # 设置中间变量或可调参数，供后续工作流使用。
        prod_ns=float(case["production_ns"]),  # 设置中间变量或可调参数，供后续工作流使用。
        atom_count=int(estimated_atoms),  # 设置中间变量或可调参数，供后续工作流使用。
        performance_profile=performance_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=analysis_profile_requested,  # 选择后处理预设；interface_fast 面向 slab/interface。
        trajectory_format=trajectory_format_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        traj_ps=traj_ps_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        energy_ps=energy_ps_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        log_ps=log_ps_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        trr_ps=trr_ps_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        velocity_ps=velocity_ps_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        rdf_frame_stride=rdf_frame_stride_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        rdf_rmax_nm=rdf_rmax_nm_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        rdf_bin_nm=rdf_bin_nm_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        msd_selected_species=["PEO", "Li", "TFSI"],  # 设置中间变量或可调参数，供后续工作流使用。
        max_trajectory_frames=max_trajectory_frames,  # 设置中间变量或可调参数，供后续工作流使用。
        max_atom_frames=max_atom_frames,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    analysis_profile = io_policy.analysis_profile  # 选择后处理预设；interface_fast 面向 slab/interface。
    analysis_fast = analysis_profile in {"transport_fast", "minimal"}  # 设置中间变量或可调参数，供后续工作流使用。

    pre_export = [  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(peo, label="PEO", moltype_hint="PEO", charge_scale=charge_scale[0]),  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(li, label="Li", moltype_hint="Li", charge_scale=charge_scale[1]),  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(tfsi, label="TFSI", moltype_hint="TFSI", charge_scale=charge_scale[2]),  # 设置中间变量或可调参数，供后续工作流使用。
    ]

    ac = poly.amorphous_cell(  # 构建无定形混合体系初始盒子。
        [peo, li, tfsi],
        counts,
        charge_scale=charge_scale,  # 设置电荷缩放系数；1.0 表示全电荷模型。
        density=initial_density_g_cm3,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=build_dir,  # 设置本例输出目录。
    )

    equil_temp_k = max(400.0, float(case["target_temp_k"]))  # 设置中间变量或可调参数，供后续工作流使用。
    eq_hot = eq.EQ21step(ac, work_dir=case_root)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = eq_hot.exec(temp=equil_temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
    analy_hot = eq_hot.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    analy_hot.get_all_prop(temp=equil_temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    melt_ok = analy_hot.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    additional_rounds = 0  # 设置中间变量或可调参数，供后续工作流使用。
    for _ in range(max_melt_additional):  # 遍历当前工作流中的一组对象或任务。
        if melt_ok:  # 根据当前状态决定是否进入该分支。
            break
        additional_rounds += 1  # 设置中间变量或可调参数，供后续工作流使用。
        eq_more = eq.Additional(ac, work_dir=case_root)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = eq_more.exec(  # 设置中间变量或可调参数，供后续工作流使用。
            temp=equil_temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
            press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
            gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
            gpu_offload_mode=gpu_offload_mode,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        analy_hot = eq_more.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        analy_hot.get_all_prop(temp=equil_temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        melt_ok = analy_hot.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    if abs(equil_temp_k - float(case["target_temp_k"])) > 1.0e-6:  # 根据当前状态决定是否进入该分支。
        eq_target = eq.Additional(ac, work_dir=case_root)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = eq_target.exec(  # 设置中间变量或可调参数，供后续工作流使用。
            temp=float(case["target_temp_k"]),  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
            press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
            gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
            gpu_offload_mode=gpu_offload_mode,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        analy_target = eq_target.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        analy_target.get_all_prop(temp=float(case["target_temp_k"]), press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        analy_target.check_eq()

    npt = eq.NPT(ac, work_dir=case_root)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = npt.exec(  # 设置中间变量或可调参数，供后续工作流使用。
        temp=float(case["target_temp_k"]),  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
        press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
        mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
        gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
        gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
        time=float(case["production_ns"]),  # 设置中间变量或可调参数，供后续工作流使用。
        traj_ps=io_policy.traj_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        energy_ps=io_policy.energy_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        log_ps=io_policy.log_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        trr_ps=io_policy.trr_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        velocity_ps=io_policy.velocity_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        trajectory_format=io_policy.trajectory_format,  # 设置中间变量或可调参数，供后续工作流使用。
        performance_profile=io_policy.performance_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=io_policy.analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
        max_trajectory_frames=io_policy.max_trajectory_frames,  # 设置中间变量或可调参数，供后续工作流使用。
        max_atom_frames=io_policy.max_atom_frames,  # 设置中间变量或可调参数，供后续工作流使用。
        gpu_offload_mode=gpu_offload_mode,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    analy = npt.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    analysis_metadata = {  # 设置中间变量或可调参数，供后续工作流使用。
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
                if analysis_fast  # 根据当前状态决定是否进入该分支。
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
    prop_data = analy.get_all_prop(  # 设置中间变量或可调参数，供后续工作流使用。
        temp=float(case["target_temp_k"]),  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
        press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
        save=True,  # 设置中间变量或可调参数，供后续工作流使用。
        include_polymer_metrics=bool(io_policy.include_polymer_metrics),  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
    )
    rdf = analy.rdf(  # 设置中间变量或可调参数，供后续工作流使用。
        center_mol=li,  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
        bin_nm=float(io_policy.rdf_bin_nm),  # 指定 z-profile bin 宽。
        r_max_nm=io_policy.rdf_rmax_nm,  # 设置中间变量或可调参数，供后续工作流使用。
        frame_stride=int(io_policy.rdf_frame_stride),  # 设置中间变量或可调参数，供后续工作流使用。
        resume=resume_analysis,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    msd = analy.msd(  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
        geometry=msd_geometry,  # 设置中间变量或可调参数，供后续工作流使用。
        unwrap=msd_unwrap,  # 设置中间变量或可调参数，供后续工作流使用。
        drift=msd_drift,  # 设置中间变量或可调参数，供后续工作流使用。
        resume=resume_analysis,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    if dielectric_analysis:  # 根据当前状态决定是否进入该分支。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            dielectric = analy.dielectric(  # 设置中间变量或可调参数，供后续工作流使用。
                temp_k=float(prop_data.get("basic_properties", {}).get("temperature_K") or case["target_temp_k"]),  # 设置中间变量或可调参数，供后续工作流使用。
                group=dielectric_group,  # 设置中间变量或可调参数，供后续工作流使用。
                dt_ps=dielectric_dt_ps,  # 设置 MD 时间步长，单位 ps。
                resume=resume_analysis,  # 设置中间变量或可调参数，供后续工作流使用。
            )
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            dielectric = {  # 设置中间变量或可调参数，供后续工作流使用。
                "status": "failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "note": (
                    "Dielectric analysis is optional. If this failed with a TPR version error, set "
                    "YADONPY_GMX_CMD to the same GROMACS major version used for production."
                ),
            }
    else:  # 处理前面条件都不满足的情况。
        dielectric = {"status": "disabled"}  # 设置中间变量或可调参数，供后续工作流使用。

    analysis_dir = case_root / "06_analysis"  # 设置中间变量或可调参数，供后续工作流使用。
    system_dir = case_root / "02_system"  # 设置中间变量或可调参数，供后续工作流使用。
    top_path = system_dir / "system.top"  # 设置中间变量或可调参数，供后续工作流使用。

    force_balance = collect_force_balance_report(  # 设置中间变量或可调参数，供后续工作流使用。
        system_dir=system_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        top_path=top_path,  # 设置中间变量或可调参数，供后续工作流使用。
        cell=ac,  # 设置中间变量或可调参数，供后续工作流使用。
        species_pre_export=pre_export,  # 设置中间变量或可调参数，供后续工作流使用。
        moltype_hints={"polymer": "PEO", "cation": "Li", "anion": "TFSI"},  # 设置中间变量或可调参数，供后续工作流使用。
    )
    coordination = build_coordination_partition(rdf, polymer_moltype="PEO", anion_moltype="TFSI")  # 设置中间变量或可调参数，供后续工作流使用。
    literature = literature_band_peo_litfsi_jpcb2020(case_key)  # 设置中间变量或可调参数，供后续工作流使用。
    transport_cache = None  # 设置中间变量或可调参数，供后续工作流使用。
    if resume_analysis:  # 根据当前状态决定是否进入该分支。
        transport_cache = _load_json_cache(  # 设置中间变量或可调参数，供后续工作流使用。
            analysis_dir / "transport_summary.json",
            [
                analysis_dir / "rdf_first_shell.json",
                analysis_dir / "msd.json",
                analysis_dir / "thermo.xvg",
            ],
        )
        if isinstance(transport_cache, dict):  # 根据当前状态决定是否进入该分支。
            cached_meta = transport_cache.get("analysis_metadata")  # 设置中间变量或可调参数，供后续工作流使用。
            cached_profile = (  # 设置中间变量或可调参数，供后续工作流使用。
                cached_meta.get("analysis_profile")
                if isinstance(cached_meta, dict)  # 根据当前状态决定是否进入该分支。
                else transport_cache.get("analysis_profile")
            )
            if str(cached_profile or "") != analysis_profile:  # 根据当前状态决定是否进入该分支。
                transport_cache = None  # 设置中间变量或可调参数，供后续工作流使用。
    if isinstance(transport_cache, dict):  # 根据当前状态决定是否进入该分支。
        transport = transport_cache  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        sigma = analy.sigma(msd=msd, temp_k=float(case["target_temp_k"]), eh_mode="gmx_current_only")  # 设置中间变量或可调参数，供后续工作流使用。
        transport = build_transport_summary(  # 设置中间变量或可调参数，供后续工作流使用。
            msd=msd,  # 设置中间变量或可调参数，供后续工作流使用。
            sigma=sigma,  # 设置中间变量或可调参数，供后续工作流使用。
            rdf=rdf,  # 设置中间变量或可调参数，供后续工作流使用。
            polymer_moltype="PEO",  # 设置中间变量或可调参数，供后续工作流使用。
            anion_moltype="TFSI",  # 设置中间变量或可调参数，供后续工作流使用。
            thermo_xvg=analysis_dir / "thermo.xvg",  # 设置中间变量或可调参数，供后续工作流使用。
            literature_band=literature,  # 设置中间变量或可调参数，供后续工作流使用。
            analysis_metadata=analysis_metadata,  # 设置中间变量或可调参数，供后续工作流使用。
        )
    compare = build_benchmark_compare(  # 设置中间变量或可调参数，供后续工作流使用。
        force_balance_report=force_balance,  # 设置中间变量或可调参数，供后续工作流使用。
        coordination_partition=coordination,  # 设置中间变量或可调参数，供后续工作流使用。
        transport_summary=transport,  # 设置中间变量或可调参数，供后续工作流使用。
        charge_scale_polymer=float(case["polymer_charge_scale"]),  # 设置中间变量或可调参数，供后续工作流使用。
        charge_scale_li=float(case["li_charge_scale"]),  # 设置中间变量或可调参数，供后续工作流使用。
        charge_scale_anion=float(case["anion_charge_scale"]),  # 设置中间变量或可调参数，供后续工作流使用。
        production_ns=float(case["production_ns"]),  # 设置中间变量或可调参数，供后续工作流使用。
    )

    metadata = {  # 设置中间变量或可调参数，供后续工作流使用。
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
    _dump_json(  # 开始一个多行函数调用或配置块。
        analysis_dir / "benchmark_compare.json",
        {
            "metadata": metadata,
            "compare": compare,
            "basic_properties": prop_data.get("basic_properties", {}),
            "dielectric": dielectric,
        },
    )
    _dump_json(analysis_dir / "benchmark_metadata.json", metadata)
    return {"metadata": metadata, "compare": compare}  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    ff, ff_variant = _make_gaff2()  # 设置中间变量或可调参数，供后续工作流使用。
    cation_ff = MERZ()  # 选择阳离子的力场/参数来源。
    cases = _resolved_cases()  # 设置中间变量或可调参数，供后续工作流使用。

    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。
    work_root = workdir(work_root, restart=restart_status)  # 创建或复用本例工作目录。

    if dry_run:  # 根据当前状态决定是否进入该分支。
        _write_dry_run_plan(work_root, cases, ff_variant)
    else:  # 处理前面条件都不满足的情况。
        peo, li, tfsi = _prepare_species(work_root, ff, cation_ff)  # 设置中间变量或可调参数，供后续工作流使用。
        results = [_run_case(work_root, case, peo, li, tfsi, ff_variant) for case in cases]  # 设置中间变量或可调参数，供后续工作流使用。
        summary = {"cases": results}  # 设置中间变量或可调参数，供后续工作流使用。
        _dump_json(work_root / "jpcb2020_screening_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))  # 打印关键路径或状态，便于人工检查。
