from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 08-01: basal graphite | carbonate/LiPF6 electrolyte.

All molecular species are loaded from MolDB and assigned force-field parameters
locally.  The script follows the Example 02 rhythm: user knobs at the top, then
a linear build -> optional NVT sampling -> interface analysis sequence.
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
restart_status = True  # True 会复用已有中间文件继续跑；改成 False 会清空本例 work_dir 重新生成。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # 小分子电解液和石墨使用 GAFF2_mod；不要在本例中临时换力场。
ion_ff = MERZ()  # Li+ 使用 Merz 离子参数，和碳酸酯/LiPF6 例子保持一致。

temp = 318.15  # 目标温度 K；可改成 298.15 做室温界面，或 333.15 做高温加速松弛。
mpi = 1  # GROMACS thread-MPI rank 数；单 GPU/小体系通常保持 1。
omp = 14  # 每个 rank 的 OpenMP 线程数；按机器 CPU 核数调整。
gpu = 1  # 1 表示允许 mdrun 使用 GPU；没有 GPU 时改成 0。
gpu_id = 0  # 选择第几张 GPU；多卡机器可改成 1、2 等。
run_sampling = True  # True 会跑 relaxation + final NVT；False 只构建初始界面并做静态分析。
sample_ns = 2.0  # final NVT 采样长度 ns；正式统计可增大到 10-50 ns。
analysis_profile = "interface_fast"  # 界面分析预设；保持 fast 可避免调用不适合 slab 的 bulk 分析。
interface_bin_nm = 0.05  # z 方向密度/电荷 profile bin 宽；更小更细但噪声更大。
interface_region_width_nm = 0.75  # 界面附近区域厚度；增大可统计更宽的吸附/渗入区域。
graphite_adsorption_cutoff_nm = 0.50  # 分子 COM 距石墨表面小于该距离时计为 graphite-near。
time_series_analysis = True  # True 会生成按时间分段的 profile/RDF/CN/角度动画和 CSV。
interface_time_series_sample_count = 10  # 时间序列取 10 个窗口，即约每十分之一总时长采样一次。
interface_time_series_fps = 1.0  # MP4 帧率；1 fps 适合慢速观察结构演化。
penetration_species = ("EC", "EMC", "DEC", "PF6")  # 统计进入界面/近石墨区域的物种。
adsorption_species = ("EC", "EMC", "DEC")  # 统计石墨吸附驻留和取向的分子；离子通常不放这里。
clean_trajectories_after_analysis = False  # True 会删大轨迹省空间；调试和复查时保持 False。

EC_smiles = "O=C1OCCO1"  # EC；必须能在 MolDB 中找到 RESP 记录。
EMC_smiles = "CCOC(=O)OC"  # EMC；改变溶剂组成时只改 counts，不建议改这里。
DEC_smiles = "CCOC(=O)OCC"  # DEC；与 EC/EMC 构成默认碳酸酯混合溶剂。
Li_smiles = "[Li+]"  # Li+；由 MERZ 分配离子参数。
PF6_smiles = "F[P-](F)(F)(F)(F)F"  # PF6-；下面必须用 bonded="DRIH" 保持八面体角项。

solvent_counts = (10, 7, 16)  # EC/EMC/DEC 个数；按同一比例放大可做更大电解液层。
salt_pairs = 3  # LiPF6 离子对数；必须同时增加 Li 和 PF6 保持电中性。
charge_scale = 0.8  # Li+/PF6- 电荷缩放；改成 1.0 表示全电荷离子模型。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = BASE_DIR / "work_dir" / "01_electrolyte_graphite_basal"  # 设置本例输出目录。


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
        name="GRAPHITE_BASAL",  # 层名会进入 manifest/ndx；后处理依赖这个语义名称。
        nx=6,  # 石墨 basal 面 x 方向重复数；增大可扩大界面面积。
        ny=5,  # 石墨 basal 面 y 方向重复数；和 nx 一起决定 XY footprint。
        n_layers=3,  # 石墨层数；3 层是轻量 sanity，正式电极可加厚。
        orientation="basal",  # basal 表示基面接触电解液；端面体系见 08-02。
        periodic_xy=True,  # True 表示 XY 周期石墨，避免基面边缘效应。
    )
    electrolyte = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="ELECTROLYTE",  # 电解液层名；后处理会用它识别液相区域。
        species=(EC, EMC, DEC, Li, PF6),  # species 顺序必须和 counts/charge_scale 顺序一致。
        counts=(*solvent_counts, salt_pairs, salt_pairs),  # EC/EMC/DEC/Li/PF6 数量；Li 和 PF6 必须配对。
        thickness_nm=2.4,  # 初始层厚目标；实际厚度可因密度目标和固定 XY footprint 在 z 方向扩展。
        density_target_g_cm3=1.2,  # 初始 packing 密度目标；不是最终 NVT/NPT 后的真实密度约束。
        layer_kind="electrolyte",  # 语义类型；控制 ndx 分组和界面分析路由。
        charge_scale=(1.0, 1.0, 1.0, charge_scale, charge_scale),  # 溶剂全电荷，离子按 charge_scale 缩放。
    )
    stack = LayerStackSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        layers=(graphite, electrolyte),  # 从下到上叠层；本例是石墨底面 + 电解液。
        order="bottom_to_top",  # 明确 layers 顺序，写入 manifest 供后处理判断相对位置。
        pbc_mode="xyz",  # 三维周期；显式真空体系才考虑 xy。
        name="electrolyte_graphite_basal",  # 系统名；用于输出文件和 manifest 标识。
        default_gap_nm=0.35,  # 相邻层初始间隙；太小易重叠，太大需要更长松弛。
        molecular_packing_expand="z",  # 固定石墨 XY；分子放不下时优先扩展 z 而不是扩大石墨面。
    )
    relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=sample_ns)  # 记录默认温度和采样时长。

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
            time_ns=sample_ns,  # final NVT 采样长度；决定后处理统计时间。
            temp=temp,  # MD 温度；和上面的 temp 保持一致。
            mpi=mpi,  # thread-MPI rank 数。
            omp=omp,  # OpenMP 线程数。
            gpu=gpu,  # GPU 开关。
            gpu_id=gpu_id,  # GPU 编号。
            run_analysis=True,  # relaxation 后自动准备 analyze facade。
            relax_z=True,  # basal 石墨 + 电解液用 fixed-XY z-NPT 消除初始层厚误差。
            restart=restart_status,  # 复用已完成阶段，适合长任务断点续跑。
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
