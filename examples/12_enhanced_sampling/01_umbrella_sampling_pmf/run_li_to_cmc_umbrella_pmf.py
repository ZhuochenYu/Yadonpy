from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 12-01: umbrella PMF for solvated Li+ entering CMC-Na."""

import os  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.core import poly, utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2_mod import GAFF2_mod  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.merz import MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.interface import (  # 导入本例需要的库或 yadonpy 接口。
    GraphiteLayerSpec,
    LayerStackRelaxationSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    SolvatedIonUmbrellaSpec,
    ZCompressionAnnealSpec,
    analyze_umbrella_pmf,
    build_layer_stack,
    prepare_solvated_ion_umbrella,
    run_layer_stack_relaxation,
    run_solvated_ion_umbrella,
)
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


def _env_flag(name: str, default: bool = False) -> bool:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    token = str(os.environ.get(name, "")).strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if not token:  # 根据当前状态决定是否进入该分支。
        return bool(default)  # 返回该辅助函数的结果。
    return token in {"1", "true", "t", "yes", "y", "on"}  # 返回该辅助函数的结果。


def _env_int(name: str, default: int) -> int:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    return int(raw) if raw else int(default)  # 返回该辅助函数的结果。


def _env_float(name: str, default: float) -> float:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    return float(raw) if raw else float(default)  # 返回该辅助函数的结果。


# ---------------- user inputs ----------------
restart_status = _env_flag("YADONPY_RESTART", True)  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

temp = _env_float("YADONPY_TEMP_K", 318.15)  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
mpi = _env_int("YADONPY_MPI", 1)  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = _env_int("YADONPY_OMP", 14)  # 设置每个 rank 的 OpenMP 线程数。
gpu = _env_int("YADONPY_GPU", 1)  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = os.environ.get("YADONPY_GPU_ID", "0").strip() or None  # 选择 GPU 设备编号，多卡节点可修改。

analysis_only = _env_flag("YADONPY_ANALYSIS_ONLY", False)  # 只做后处理，不重新运行采样。
skip_sampling = _env_flag("YADONPY_PREPARE_ONLY", False)  # 跳过采样阶段，仅准备输入或构建体系。

relax_ns = _env_float("YADONPY_RELAX_NS", 2.0)  # 设置界面预松弛时长，单位 ns。
umbrella_windows = _env_int("YADONPY_UMBRELLA_WINDOWS", 31)  # 设置 umbrella sampling 窗口数。
umbrella_steering_ns = _env_float("YADONPY_UMBRELLA_STEERING_NS", 0.50)  # 设置 pulling/steering 阶段时长。
umbrella_window_eq_ns = _env_float("YADONPY_UMBRELLA_WINDOW_EQ_NS", 0.20)  # 设置每个 umbrella 窗口的平衡时长。
umbrella_window_ns = _env_float("YADONPY_UMBRELLA_WINDOW_NS", 1.00)  # 设置每个 umbrella 窗口的生产时长。
umbrella_k = _env_float("YADONPY_UMBRELLA_K", 1000.0)  # 设置 umbrella harmonic force constant。
wham_skip_ps = _env_float("YADONPY_WHAM_SKIP_PS", 200.0)  # 设置 WHAM 丢弃每个窗口前多少 ps。
wham_bins = _env_int("YADONPY_WHAM_BINS", 200)  # 设置 WHAM 反应坐标 bins 数。

ff = GAFF2_mod()  # 选择有机分子/聚合物/部分无机离子的力场对象。
ion_ff = MERZ()  # 选择单原子离子参数来源。

glucose6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。
ter_smiles = "[H][*]"  # 设置中间变量或可调参数，供后续工作流使用。
EC_smiles = "O=C1OCCO1"  # 设置中间变量或可调参数，供后续工作流使用。
EMC_smiles = "CCOC(=O)OC"  # 设置中间变量或可调参数，供后续工作流使用。
DEC_smiles = "CCOC(=O)OCC"  # 设置中间变量或可调参数，供后续工作流使用。
Li_smiles = "[Li+]"  # 设置中间变量或可调参数，供后续工作流使用。
Na_smiles = "[Na+]"  # 设置中间变量或可调参数，供后续工作流使用。
PF6_smiles = "F[P-](F)(F)(F)(F)F"  # 设置中间变量或可调参数，供后续工作流使用。

cmc_dp = _env_int("YADONPY_CMC_DP", 5)  # 设置 CMC 聚合度；增大时通常要同步放大盒子或链数。
cmc_chain_count = _env_int("YADONPY_CMC_CHAINS", 1)  # 设置 CMC 链数；会影响 Na+ counterion 数量和层密度。
solvent_counts = (  # 设置各溶剂分子数量；顺序必须和 species 保持一致。
    _env_int("YADONPY_EC_COUNT", 8),
    _env_int("YADONPY_EMC_COUNT", 6),
    _env_int("YADONPY_DEC_COUNT", 14),
)
salt_pairs = _env_int("YADONPY_SALT_PAIRS", 3)  # 设置盐离子对数；阳离子和阴离子应同步增减。
charge_scale = _env_float("YADONPY_CHARGE_SCALE", 0.7)  # 设置电荷缩放系数；1.0 表示全电荷模型。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = Path(os.environ.get("YADONPY_WORK_DIR", BASE_DIR / "work_dir" / "li_to_cmc_umbrella_pmf"))  # 设置本例输出目录。


def _build_and_relax_interface(root_dir: Path):  # 定义本例内部辅助函数，组织重复步骤。
    cmc_rw_dir = root_dir / "00_interface_build" / "00_cmc_rw"  # 设置中间变量或可调参数，供后续工作流使用。
    cmc_term_dir = root_dir / "00_interface_build" / "01_cmc_term"  # 设置中间变量或可调参数，供后续工作流使用。

    glucose6 = ff.mol(glucose6_smiles, charge="RESP", prefer_db=True, require_ready=True, polyelectrolyte_mode=True)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    glucose6 = ff.ff_assign(glucose6, report=False)  # 分配力场参数并写入分子属性。
    ter = utils.mol_from_smiles(ter_smiles)  # 从 SMILES 直接构造 RDKit 分子。
    EC = ff.mol(EC_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    EC = ff.ff_assign(EC, report=False)  # 分配力场参数并写入分子属性。
    EMC = ff.mol(EMC_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    EMC = ff.ff_assign(EMC, report=False)  # 分配力场参数并写入分子属性。
    DEC = ff.mol(DEC_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    DEC = ff.ff_assign(DEC, report=False)  # 分配力场参数并写入分子属性。
    PF6 = ff.mol(PF6_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    PF6 = ff.ff_assign(PF6, bonded="DRIH", report=False)  # 分配力场参数并写入分子属性。
    Li = ion_ff.mol(Li_smiles)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    Li = ion_ff.ff_assign(Li, report=False)  # 分配力场参数并写入分子属性。
    Na = ion_ff.mol(Na_smiles)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    Na = ion_ff.ff_assign(Na, report=False)  # 分配力场参数并写入分子属性。
    if not all((glucose6, ter, EC, EMC, DEC, PF6, Li, Na)):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("MolDB/FF assignment failed for one or more CMC/electrolyte species.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    CMC = poly.random_copolymerize_rw([glucose6], cmc_dp, ratio=[1.0], tacticity="atactic", work_dir=cmc_rw_dir)  # 用随机游走生成聚合物链。
    CMC = poly.terminate_rw(CMC, ter, work_dir=cmc_term_dir)  # 给聚合物链加端基。
    CMC = ff.ff_assign(CMC, report=False)  # 分配力场参数并写入分子属性。
    if not CMC:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Cannot assign GAFF2_mod parameters to the CMC-Na chain.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    graphite = GraphiteLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="GRAPHITE_BASAL",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        nx=6,  # 设置石墨 x 方向重复数。
        ny=5,  # 设置石墨 y 方向重复数。
        n_layers=3,  # 设置石墨层数。
        orientation="basal",  # 设置石墨暴露面类型。
        periodic_xy=True,  # 控制石墨是否在 XY 方向周期成键。
    )
    cmcna = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="CMCNA",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        species=(CMC, Na),  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。
        counts=(cmc_chain_count, cmc_chain_count * cmc_dp),  # 设置各 species 的数量；顺序必须和 species 列表一致。
        thickness_nm=1.8,  # 设置初始层厚，单位 nm。
        density_target_g_cm3=1.0,  # 设置初始层 packing 目标密度，不等价于最终平衡密度。
        layer_kind="cmcna",  # 设置层的语义类型，供 index 和后处理识别。
        charge_scale=(charge_scale, charge_scale),  # 设置电荷缩放系数；1.0 表示全电荷模型。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        counterion_contact_mode="carboxylate",  # 设置 counterion 初始定位方式，carboxylate 会靠近 COO-。
    )
    electrolyte = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="ELECTROLYTE",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        species=(EC, EMC, DEC, Li, PF6),  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。
        counts=(*solvent_counts, salt_pairs, salt_pairs),  # 设置各 species 的数量；顺序必须和 species 列表一致。
        thickness_nm=2.2,  # 设置初始层厚，单位 nm。
        density_target_g_cm3=1.2,  # 设置初始层 packing 目标密度，不等价于最终平衡密度。
        layer_kind="electrolyte",  # 设置层的语义类型，供 index 和后处理识别。
        charge_scale=(1.0, 1.0, 1.0, charge_scale, charge_scale),  # 设置电荷缩放系数；1.0 表示全电荷模型。
    )
    stack = LayerStackSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        layers=(graphite, cmcna, electrolyte),  # 按从下到上的顺序列出层。
        order="bottom_to_top",  # 声明 layers 的空间顺序。
        pbc_mode="xyz",  # 设置周期边界模式。
        name="eg12_graphite_cmcna_electrolyte_li_umbrella",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        default_gap_nm=0.35,  # 设置相邻层的初始间隙。
        molecular_packing_expand="z",  # 设置分子放不下时优先扩展的方向。
    )
    relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=relax_ns)  # 设置中间变量或可调参数，供后续工作流使用。
    built = build_layer_stack(  # 构建 layer-stack 初始结构、拓扑和索引。
        stack=stack,  # 设置中间变量或可调参数，供后续工作流使用。
        relaxation=relaxation,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=root_dir / "00_interface_build",  # 设置本例输出目录。
        restart=restart_status,  # 传入断点续跑开关。
    )
    relaxed = run_layer_stack_relaxation(  # 运行 layer-stack 松弛和 final NVT 采样。
        built,
        work_dir=root_dir / "01_relaxation",  # 设置本例输出目录。
        time_ns=relax_ns,  # 设置该 MD 阶段的时长，单位 ns。
        temp=temp,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
        mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
        gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
        gpu_id=None if gpu_id is None else int(gpu_id),  # 选择 GPU 设备编号，多卡节点可修改。
        run_analysis=True,  # 控制是否准备后处理入口。
        relax_z=True,  # 控制是否允许 z 方向 NPT 松弛。
        compression_anneal=ZCompressionAnnealSpec(  # 传入循环压缩退火配置。
            enabled=True,  # 控制该功能是否启用。
            cycles=8,  # 设置循环次数。
            tmax_K=380.0,  # 设置退火最高温度。
            pmax_bar=3000.0,  # 设置 z-only 高压上限。
            max_z_shrink_per_cycle=0.04,  # 限制每轮几何压缩比例。
        ),
        restart=restart_status,  # 传入断点续跑开关。
    )
    return built, relaxed  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。
    root = Path(workdir(work_dir, restart=restart_status))  # 创建或复用本例工作目录。

    if analysis_only:  # 根据当前状态决定是否进入该分支。
        plan = root / "02_solvated_li_selection" / "umbrella_sampling_manifest.json"  # 设置中间变量或可调参数，供后续工作流使用。
        result = analyze_umbrella_pmf(plan)  # 执行 WHAM/PMF 和配位后处理。
        print(f"umbrella_pmf_summary = {result.summary_path}")  # 打印关键路径或状态，便于人工检查。
        raise SystemExit(0)  # 关键步骤失败时立即报错，避免继续生成错误结果。

    built, relaxed = _build_and_relax_interface(root)  # 设置中间变量或可调参数，供后续工作流使用。
    print(f"layer_stack_manifest = {built.manifest_path}")  # 打印关键路径或状态，便于人工检查。
    print(f"relaxation_summary = {relaxed.summary_path}")  # 打印关键路径或状态，便于人工检查。

    umbrella_spec = SolvatedIonUmbrellaSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        target_group="CMCNA",  # 设置中间变量或可调参数，供后续工作流使用。
        target_coordination_number=4,  # 设置中间变量或可调参数，供后续工作流使用。
        target_offset_nm=0.0,  # 设置中间变量或可调参数，供后续工作流使用。
        steering_ns=umbrella_steering_ns,  # 设置中间变量或可调参数，供后续工作流使用。
        window_count=umbrella_windows,  # 设置中间变量或可调参数，供后续工作流使用。
        window_equilibration_ns=umbrella_window_eq_ns,  # 设置中间变量或可调参数，供后续工作流使用。
        window_production_ns=umbrella_window_ns,  # 设置中间变量或可调参数，供后续工作流使用。
        temperature_K=temp,  # 设置中间变量或可调参数，供后续工作流使用。
        dt_ps=0.001,  # 设置 MD 时间步长，单位 ps。
        constraints="none",  # 设置约束策略。
        umbrella_k_kj_mol_nm2=umbrella_k,  # 设置中间变量或可调参数，供后续工作流使用。
        wham_skip_ps=wham_skip_ps,  # 设置 WHAM 丢弃每个窗口前多少 ps。
        wham_bins=wham_bins,  # 设置 WHAM 反应坐标 bins 数。
    )
    plan = prepare_solvated_ion_umbrella(  # 准备 solvated-ion umbrella sampling 计划。
        system_dir=relaxed.work_dir / "02_system",  # 设置中间变量或可调参数，供后续工作流使用。
        gro_path=relaxed.final_gro,  # 设置中间变量或可调参数，供后续工作流使用。
        top_path=relaxed.work_dir / "02_system" / "system.top",  # 设置中间变量或可调参数，供后续工作流使用。
        ndx_path=relaxed.work_dir / "02_system" / "system.ndx",  # 设置中间变量或可调参数，供后续工作流使用。
        manifest_path=built.manifest_path,  # 指定 layer_stack_manifest.json 路径。
        out_dir=root,  # 设置中间变量或可调参数，供后续工作流使用。
        spec=umbrella_spec,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    print(f"umbrella_manifest = {plan.manifest_path}")  # 打印关键路径或状态，便于人工检查。
    print(f"selected_li_atom = {plan.selected_center_atom}")  # 打印关键路径或状态，便于人工检查。
    print(f"window_count = {len(plan.windows)}")  # 打印关键路径或状态，便于人工检查。

    if skip_sampling:  # 根据当前状态决定是否进入该分支。
        print("YADONPY_PREPARE_ONLY is set; umbrella inputs were generated but MD was not launched.")  # 打印关键路径或状态，便于人工检查。
        raise SystemExit(0)  # 关键步骤失败时立即报错，避免继续生成错误结果。

    pmf = run_solvated_ion_umbrella(  # 运行 umbrella sampling 窗口。
        plan,
        mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
        gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
        gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
        restart=restart_status,  # 传入断点续跑开关。
    )
    print(f"umbrella_pmf_summary = {pmf.summary_path}")  # 打印关键路径或状态，便于人工检查。
    print(f"pmf_csv = {pmf.pmf_csv}")  # 打印关键路径或状态，便于人工检查。
