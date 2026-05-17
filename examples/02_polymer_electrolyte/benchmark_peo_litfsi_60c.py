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
    literature_band_peo_litfsi_60c,
    summarize_rdkit_species_forcefield,
)
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


def _apply_literature_preset() -> dict[str, float | int | str] | None:  # 定义本例内部辅助函数，组织重复步骤。
    preset = str(os.environ.get("LITERATURE_PRESET", "") or "").strip()  # 设置中间变量或可调参数，供后续工作流使用。
    if not preset:  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    key = preset.upper()  # 设置中间变量或可调参数，供后续工作流使用。
    presets: dict[str, dict[str, float | int | str]] = {  # 设置中间变量或可调参数，供后续工作流使用。
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
    if key not in presets:  # 根据当前状态决定是否进入该分支。
        raise ValueError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
            f"Unsupported LITERATURE_PRESET={preset!r}. "
            "Expected one of: JPCB2020_P1.00S1.00, JPCB2020_P1.00S0.75, JPCB2020_P1.20S0.75."
        )
    return {"preset_name": key, **presets[key]}  # 返回该辅助函数的结果。


BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。

restart_status = _env_bool("RESTART_STATUS", False)  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # 选择有机分子/聚合物/部分无机离子的力场对象。
cation_ff = MERZ()  # 选择阳离子的力场/参数来源。

melt_temp_k = _env_float("MELT_TEMP_K", 353.15)  # 设置中间变量或可调参数，供后续工作流使用。
target_temp_k = _env_float("TARGET_TEMP_K", 333.15)  # 设置中间变量或可调参数，供后续工作流使用。
press_bar = _env_float("PRESS_BAR", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
prod_ns = _env_float("PROD_NS", 10.0)  # 设置中间变量或可调参数，供后续工作流使用。
initial_density_g_cm3 = _env_float("INITIAL_DENSITY_G_CM3", 0.05)  # 设置中间变量或可调参数，供后续工作流使用。

mpi = _env_int("MPI", 1)  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = _env_int("OMP", 16)  # 设置每个 rank 的 OpenMP 线程数。
gpu = _env_int("GPU", 1)  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = _env_int("GPU_ID", 0)  # 选择 GPU 设备编号，多卡节点可修改。
omp_psi4 = _env_int("OMP_PSI4", 32)  # 设置 Psi4/OpenMP 核数。
mem_mb = _env_int("MEM_MB", 20000)  # 设置量子化学内存 MB。

chain_dp = _env_int("CHAIN_DP", 40)  # 设置中间变量或可调参数，供后续工作流使用。
chain_count = _env_int("CHAIN_COUNT", 32)  # 设置中间变量或可调参数，供后续工作流使用。
salt_pairs = _env_int("SALT_PAIRS", 64)  # 设置盐离子对数；阳离子和阴离子应同步增减。
cool_rounds = _env_int("COOL_ROUNDS", 1)  # 设置中间变量或可调参数，供后续工作流使用。
max_melt_additional = _env_int("MAX_MELT_ADDITIONAL", 2)  # 设置中间变量或可调参数，供后续工作流使用。

li_charge_scale = _env_float("LI_CHARGE_SCALE", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
anion_charge_scale = _env_float("ANION_CHARGE_SCALE", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
polymer_charge_scale = _env_float("POLYMER_CHARGE_SCALE", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。

literature_preset = _apply_literature_preset()  # 设置中间变量或可调参数，供后续工作流使用。
if literature_preset is not None:  # 根据当前状态决定是否进入该分支。
    melt_temp_k = float(literature_preset["melt_temp_k"])  # 设置中间变量或可调参数，供后续工作流使用。
    target_temp_k = float(literature_preset["target_temp_k"])  # 设置中间变量或可调参数，供后续工作流使用。
    chain_dp = int(literature_preset["chain_dp"])  # 设置中间变量或可调参数，供后续工作流使用。
    chain_count = int(literature_preset["chain_count"])  # 设置中间变量或可调参数，供后续工作流使用。
    salt_pairs = int(literature_preset["salt_pairs"])  # 设置盐离子对数；阳离子和阴离子应同步增减。
    polymer_charge_scale = float(literature_preset["polymer_charge_scale"])  # 设置中间变量或可调参数，供后续工作流使用。
    li_charge_scale = float(literature_preset["li_charge_scale"])  # 设置中间变量或可调参数，供后续工作流使用。
    anion_charge_scale = float(literature_preset["anion_charge_scale"])  # 设置中间变量或可调参数，供后续工作流使用。

work_dir_name = os.environ.get("WORK_DIR_NAME", "benchmark_peo_litfsi_60c_work")  # 设置中间变量或可调参数，供后续工作流使用。
work_root = Path(os.environ.get("WORK_DIR", str(BASE_DIR / work_dir_name))).resolve()  # 设置中间变量或可调参数，供后续工作流使用。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    work_root = workdir(work_root, restart=restart_status)  # 创建或复用本例工作目录。
    build_dir = work_root.child("00_build_cell")  # 设置中间变量或可调参数，供后续工作流使用。
    poly_rw_dir = work_root.child("poly_rw")  # 设置中间变量或可调参数，供后续工作流使用。
    poly_term_dir = work_root.child("poly_term")  # 设置中间变量或可调参数，供后续工作流使用。

    monomer = utils.mol_from_smiles(r"*CCO*")  # 从 SMILES 直接构造 RDKit 分子。
    monomer, _ = qm.conformation_search(  # 执行构象搜索/几何优化，为 RESP 做准备。
        monomer,
        ff=ff,  # 选择有机分子/聚合物/部分无机离子的力场对象。
        work_dir=work_root,  # 设置本例输出目录。
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
        work_dir=work_root,  # 设置本例输出目录。
        omp=omp_psi4,  # 设置每个 rank 的 OpenMP 线程数。
        memory=mem_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        log_name=None,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    ter = utils.mol_from_smiles("[H][*]")  # 从 SMILES 直接构造 RDKit 分子。
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        ter,
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        opt=True,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=work_root,  # 设置本例输出目录。
        omp=omp_psi4,  # 设置每个 rank 的 OpenMP 线程数。
        memory=mem_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        log_name=None,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    peo = poly.random_copolymerize_rw(  # 用随机游走生成聚合物链。
        [monomer],
        chain_dp,
        ratio=[1.0],  # 设置共聚组成比例。
        tacticity="atactic",  # 设置聚合物立构。
        name="PEO",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=poly_rw_dir,  # 设置本例输出目录。
    )
    peo = poly.terminate_rw(peo, ter, name="PEO", work_dir=poly_term_dir)  # 给聚合物链加端基。
    peo = ff.ff_assign(peo)  # 分配力场参数并写入分子属性。
    if not peo:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Failed to assign force field parameters for PEO benchmark chain.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    li = cation_ff.mol("[Li+]")  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    li = cation_ff.ff_assign(li)  # 分配力场参数并写入分子属性。
    if not li:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Failed to assign MERZ parameters for Li+.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        tfsi = ff.mol(  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
            "FC(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F",
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            require_ready=True,  # 要求 MolDB 物种必须已准备好。
            prefer_db=True,  # 优先从 MolDB 读取已有结果。
        )
        tfsi = ff.ff_assign(tfsi)  # 分配力场参数并写入分子属性。
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
            "This benchmark requires a ready RESP-backed TFSI record in MolDB. "
            "Precompute TFSI first, then rerun the benchmark."
        ) from exc
    if not tfsi:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Failed to assign force field parameters for TFSI.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    counts = [chain_count, salt_pairs, salt_pairs]  # 设置各 species 的数量；顺序必须和 species 列表一致。
    charge_scale = [polymer_charge_scale, li_charge_scale, anion_charge_scale]  # 设置电荷缩放系数；1.0 表示全电荷模型。

    estimated_atoms = chain_count * int(peo.GetNumAtoms()) + salt_pairs * int(li.GetNumAtoms()) + salt_pairs * int(tfsi.GetNumAtoms())  # 设置中间变量或可调参数，供后续工作流使用。
    if estimated_atoms < 10000 or estimated_atoms > 30000:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
            f"Benchmark atom count must stay within 10k-30k; got estimated_total_atoms={estimated_atoms} "
            f"(chain_count={chain_count}, salt_pairs={salt_pairs}, chain_dp={chain_dp})."
        )

    pre_export = [  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(peo, label="PEO", moltype_hint="PEO", charge_scale=polymer_charge_scale),  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(li, label="Li", moltype_hint="Li", charge_scale=li_charge_scale),  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(tfsi, label="TFSI", moltype_hint="TFSI", charge_scale=anion_charge_scale),  # 设置中间变量或可调参数，供后续工作流使用。
    ]

    ac = poly.amorphous_cell(  # 构建无定形混合体系初始盒子。
        [peo, li, tfsi],
        counts,
        charge_scale=charge_scale,  # 设置电荷缩放系数；1.0 表示全电荷模型。
        density=initial_density_g_cm3,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=build_dir,  # 设置本例输出目录。
    )

    eq_hot = eq.EQ21step(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = eq_hot.exec(temp=melt_temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
    analy_hot = eq_hot.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    analy_hot.get_all_prop(temp=melt_temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    melt_ok = analy_hot.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    for _ in range(max_melt_additional):  # 遍历当前工作流中的一组对象或任务。
        if melt_ok:  # 根据当前状态决定是否进入该分支。
            break
        eq_more = eq.Additional(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = eq_more.exec(temp=melt_temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
        analy_hot = eq_more.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        analy_hot.get_all_prop(temp=melt_temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        melt_ok = analy_hot.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    for _ in range(cool_rounds):  # 遍历当前工作流中的一组对象或任务。
        eq_cool = eq.Additional(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = eq_cool.exec(temp=target_temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
        analy_cool = eq_cool.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        analy_cool.get_all_prop(temp=target_temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        analy_cool.check_eq()

    npt = eq.NPT(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = npt.exec(temp=target_temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=prod_ns)  # 设置中间变量或可调参数，供后续工作流使用。

    analy = npt.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    prop_data = analy.get_all_prop(temp=target_temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    rdf = analy.rdf(center_mol=li)  # 设置中间变量或可调参数，供后续工作流使用。
    msd = analy.msd()  # 设置中间变量或可调参数，供后续工作流使用。
    sigma = analy.sigma(msd=msd, temp_k=target_temp_k, eh_mode="gmx_current_only")  # 设置中间变量或可调参数，供后续工作流使用。

    analysis_dir = work_root / "06_analysis"  # 设置中间变量或可调参数，供后续工作流使用。
    system_dir = work_root / "02_system"  # 设置中间变量或可调参数，供后续工作流使用。
    top_path = system_dir / "system.top"  # 设置中间变量或可调参数，供后续工作流使用。

    force_balance = collect_force_balance_report(  # 设置中间变量或可调参数，供后续工作流使用。
        system_dir=system_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        top_path=top_path,  # 设置中间变量或可调参数，供后续工作流使用。
        cell=ac,  # 设置中间变量或可调参数，供后续工作流使用。
        species_pre_export=pre_export,  # 设置中间变量或可调参数，供后续工作流使用。
        moltype_hints={"polymer": "PEO", "cation": "Li", "anion": "TFSI"},  # 设置中间变量或可调参数，供后续工作流使用。
    )
    coordination = build_coordination_partition(rdf, polymer_moltype="PEO", anion_moltype="TFSI")  # 设置中间变量或可调参数，供后续工作流使用。
    transport = build_transport_summary(  # 设置中间变量或可调参数，供后续工作流使用。
        msd=msd,  # 设置中间变量或可调参数，供后续工作流使用。
        sigma=sigma,  # 设置中间变量或可调参数，供后续工作流使用。
        rdf=rdf,  # 设置中间变量或可调参数，供后续工作流使用。
        polymer_moltype="PEO",  # 设置中间变量或可调参数，供后续工作流使用。
        anion_moltype="TFSI",  # 设置中间变量或可调参数，供后续工作流使用。
        thermo_xvg=analysis_dir / "thermo.xvg",  # 设置中间变量或可调参数，供后续工作流使用。
        literature_band=literature_band_peo_litfsi_60c(),  # 设置中间变量或可调参数，供后续工作流使用。
    )
    compare = build_benchmark_compare(  # 设置中间变量或可调参数，供后续工作流使用。
        force_balance_report=force_balance,  # 设置中间变量或可调参数，供后续工作流使用。
        coordination_partition=coordination,  # 设置中间变量或可调参数，供后续工作流使用。
        transport_summary=transport,  # 设置中间变量或可调参数，供后续工作流使用。
        charge_scale_polymer=polymer_charge_scale,  # 设置中间变量或可调参数，供后续工作流使用。
        charge_scale_li=li_charge_scale,  # 设置中间变量或可调参数，供后续工作流使用。
        charge_scale_anion=anion_charge_scale,  # 设置中间变量或可调参数，供后续工作流使用。
        production_ns=prod_ns,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    effective_eo_li_ratio = float(chain_dp * chain_count) / max(float(salt_pairs), 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
    metadata = {  # 设置中间变量或可调参数，供后续工作流使用。
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
    _dump_json(  # 开始一个多行函数调用或配置块。
        analysis_dir / "benchmark_compare.json",
        {
            "metadata": metadata,
            "compare": compare,
            "basic_properties": prop_data.get("basic_properties", {}),
        },
    )
    _dump_json(analysis_dir / "benchmark_metadata.json", metadata)

    print("[BENCHMARK] PEO/LiTFSI 60C benchmark completed")  # 打印关键路径或状态，便于人工检查。
    print(json.dumps({"metadata": metadata, "compare": compare}, indent=2, ensure_ascii=False))  # 打印关键路径或状态，便于人工检查。
