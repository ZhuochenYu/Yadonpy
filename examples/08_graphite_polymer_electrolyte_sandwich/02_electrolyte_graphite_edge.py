from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 08-02: edge graphite | carbonate/LiPF6 electrolyte.

The graphite layer is a finite, capped edge slab (`periodic_xy=False`).  Change
`edge_cap` near the top to test H/OH/O/COOH edge chemistry without touching the
rest of the workflow.
"""

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import clean_md_trajectory_files  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2_mod import GAFF2_mod  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.merz import MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.interface import (  # 导入本例需要的库或 yadonpy 接口。
    GraphiteLayerSpec,
    LayerStackRelaxationSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    analyze_layer_stack_interface,
    build_layer_stack,
    run_layer_stack_relaxation,
)
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


# ---------------- user inputs ----------------
restart_status = True  # True 复用已有中间文件；False 会重新生成本例全部工作目录。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # 有机溶剂、PF6- 和石墨端面使用 GAFF2_mod。
ion_ff = MERZ()  # Li+ 使用 Merz 离子力场。

temp = 318.15  # 目标温度 K；高温可加快端面溶剂重排。
mpi = 1  # thread-MPI rank 数；小体系/单卡保持 1。
omp = 14  # OpenMP 线程数；按 CPU 核数调整。
gpu = 1  # 1 使用 GPU，0 使用 CPU。
gpu_id = 0  # 多 GPU 时选择设备编号。
run_sampling = True  # True 跑完整松弛和采样；False 只构建端面界面。
sample_ns = 2.0  # final NVT 采样 ns；端面吸附统计建议后续加长。
edge_cap = "OH"  # 石墨端面封端；可改 "H"、"O"、"COOH" 或混合方案测试边缘化学。
analysis_profile = "interface_fast"  # 界面快速分析预设。
interface_bin_nm = 0.05  # z-profile bin 宽。
interface_region_width_nm = 0.75  # 近界面区域厚度。
graphite_adsorption_cutoff_nm = 0.50  # COM 到石墨小于该距离计为近表面驻留。
time_series_analysis = True  # 输出慢速时间序列动画/CSV。
interface_time_series_sample_count = 10  # 采样十个时间窗口。
interface_time_series_fps = 1.0  # 动画 1 fps，便于观察。
penetration_species = ("EC", "EMC", "DEC", "PF6")  # 渗入/分布分析的物种。
adsorption_species = ("EC", "EMC", "DEC")  # 端面吸附和取向分析的物种。
clean_trajectories_after_analysis = False  # 调试时保留轨迹，空间紧张时可改 True。

EC_smiles = "O=C1OCCO1"  # EC，MolDB RESP 物种。
EMC_smiles = "CCOC(=O)OC"  # EMC，MolDB RESP 物种。
DEC_smiles = "CCOC(=O)OCC"  # DEC，MolDB RESP 物种。
Li_smiles = "[Li+]"  # Li+，MERZ 参数。
PF6_smiles = "F[P-](F)(F)(F)(F)F"  # PF6-，必须使用 DRIH 键角/交叉项。

solvent_counts = (8, 6, 14)  # EC/EMC/DEC 数量；按比例放大可增大电解液层。
salt_pairs = 3  # LiPF6 离子对数；Li 和 PF6 同步增减保持电中性。
charge_scale = 0.8  # 离子电荷缩放；改 1.0 为全电荷。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = BASE_DIR / "work_dir" / "02_electrolyte_graphite_edge"  # 设置本例输出目录。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。
    work_dir = workdir(work_dir, restart=restart_status)  # 设置本例输出目录。

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
    if not all((EC, EMC, DEC, PF6, Li)):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("MolDB/FF assignment failed for one or more electrolyte species.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    graphite = GraphiteLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="GRAPHITE_EDGE",  # 端面石墨层名；后处理据此建立 GRAPHITE 组。
        nx=6,  # 石墨片 x 重复数；增大得到更宽端面。
        ny=8,  # 石墨片 y 重复数；端面模型通常需要比 basal 更长。
        n_layers=3,  # 石墨层数；加厚可降低有限 slab 的柔性。
        orientation="edge",  # edge 表示暴露端面而非基面。
        periodic_xy=False,  # 端面 slab 是有限片，需要边缘封端；不要设成 True。
        edge_cap=edge_cap,  # 使用上面的封端类型控制端面化学。
    )
    electrolyte = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="ELECTROLYTE",  # 电解液层名。
        species=(EC, EMC, DEC, Li, PF6),  # 顺序和 counts/charge_scale 一致。
        counts=(*solvent_counts, salt_pairs, salt_pairs),  # EC/EMC/DEC/Li/PF6 数量。
        thickness_nm=2.4,  # 初始层厚；真实厚度由后续 z 松弛决定。
        density_target_g_cm3=1.2,  # 初始 packing 目标密度。
        layer_kind="electrolyte",  # 语义类型。
        charge_scale=(1.0, 1.0, 1.0, charge_scale, charge_scale),  # 溶剂不缩放，离子缩放。
    )
    stack = LayerStackSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        layers=(graphite, electrolyte),  # 底部端面石墨，上方电解液。
        order="bottom_to_top",  # 明确层顺序。
        pbc_mode="xyz",  # 三维周期；有限端面仍放在周期盒中。
        name="electrolyte_graphite_edge",  # 输出系统名。
        default_gap_nm=0.35,  # 初始层间间隙。
        molecular_packing_expand="z",  # 固定石墨 XY，分子多时扩 z。
    )
    relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=sample_ns)  # 写入温度/采样意图。

    result = build_layer_stack(  # 构建 layer-stack 初始结构、拓扑和索引。
        stack=stack,  # 设置中间变量或可调参数，供后续工作流使用。
        relaxation=relaxation,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=work_dir,  # 设置本例输出目录。
        ff_name="gaff2_mod",  # 设置中间变量或可调参数，供后续工作流使用。
        charge_method="RESP",  # 设置中间变量或可调参数，供后续工作流使用。
        restart=restart_status,  # 传入断点续跑开关。
    )
    print(f"layer_stack_manifest = {result.manifest_path}")  # 打印关键路径或状态，便于人工检查。
    print(f"stack_gmx_dir = {result.system_gro.parent}")  # 打印关键路径或状态，便于人工检查。
    print(f"acceptance = {result.acceptance}")  # 打印关键路径或状态，便于人工检查。

    analyze_layer_stack_interface(  # 对静态或轨迹界面结构执行后处理。
        work_dir=work_dir,  # 设置本例输出目录。
        manifest_path=result.manifest_path,  # 指定 layer_stack_manifest.json 路径。
        analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
        bin_nm=interface_bin_nm,  # 指定 z-profile bin 宽。
        region_width_nm=interface_region_width_nm,  # 指定界面区域宽度。
        surface_distance_nm=graphite_adsorption_cutoff_nm,  # 指定石墨近表面距离阈值。
        penetration_species=penetration_species,  # 列出参与 penetration/区域分布统计的物种。
        adsorption_species=adsorption_species,  # 列出参与石墨吸附/取向统计的物种。
        compute_transport=False,  # 控制是否计算 transport/MSD。
        time_series_analysis=False,  # 控制是否输出时间序列 CSV/MP4。
    )
    if run_sampling:  # 根据当前状态决定是否进入该分支。
        relax = run_layer_stack_relaxation(  # 运行 layer-stack 松弛和 final NVT 采样。
            result,
            time_ns=sample_ns,  # final NVT 时长。
            temp=temp,  # 目标温度。
            mpi=mpi,  # thread-MPI rank。
            omp=omp,  # OpenMP 线程。
            gpu=gpu,  # GPU 开关。
            gpu_id=gpu_id,  # GPU 编号。
            run_analysis=True,  # 保留分析入口。
            relax_z=True,  # 端面/电解液界面仍允许 z-NPT 修正初始层厚。
            restart=restart_status,  # 允许断点续跑。
        )
        print(f"relaxation_summary = {relax.summary_path}")  # 打印关键路径或状态，便于人工检查。
        analy = relax.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        interface = analy.interface(  # 设置中间变量或可调参数，供后续工作流使用。
            manifest_path=result.manifest_path,  # 指定 layer_stack_manifest.json 路径。
            analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
            bin_nm=interface_bin_nm,  # 指定 z-profile bin 宽。
            region_width_nm=interface_region_width_nm,  # 指定界面区域宽度。
            surface_distance_nm=graphite_adsorption_cutoff_nm,  # 指定石墨近表面距离阈值。
            penetration_species=penetration_species,  # 列出参与 penetration/区域分布统计的物种。
            adsorption_species=adsorption_species,  # 列出参与石墨吸附/取向统计的物种。
            time_series_sample_count=interface_time_series_sample_count,  # 设置时间序列窗口数。
            time_series_fps=interface_time_series_fps,  # 设置时间序列动画帧率。
        )
        health = interface.geometry_health(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        z_profile = interface.z_profiles(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        edl = interface.edl_profiles(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        penetration = interface.penetration(species=penetration_species, time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        adsorption = interface.graphite_adsorption(  # 设置中间变量或可调参数，供后续工作流使用。
            species=adsorption_species,  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。
            time_series_analysis=time_series_analysis,  # 控制是否输出时间序列 CSV/MP4。
        )
        coordination = interface.coordination_by_region(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        transport = interface.region_transport(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        time_series = interface.time_series(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        summary = interface.summary(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        print(f"interface_phase_order_ok = {health.get('phase_order_ok')}")  # 打印关键路径或状态，便于人工检查。
        print(f"interface_outputs = {summary.get('outputs', {}).get('interface_profile_summary_json')}")  # 打印关键路径或状态，便于人工检查。
        print(f"interface_time_series = {time_series.get('outputs', {})}")  # 打印关键路径或状态，便于人工检查。

    clean_md_trajectory_files(work_dir, enabled=clean_trajectories_after_analysis)  # 按配置清理 MD 轨迹文件。
