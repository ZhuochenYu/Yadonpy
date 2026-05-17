from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 08-06: large flat fixed-charge graphite | electrolyte | CMC-Na stack.

This is the production-sized counterpart of Example 08-05.  The default system
uses a broad basal-graphite XY footprint, DP=20 CMC-Na, and eight CMC chains.
The initial molecular layers are deliberately loose in z so packing succeeds;
the fixed-XY compression anneal and z-NPT stages then compact the confined
region before the final NVT trajectory is sampled and analyzed.
"""

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import clean_md_trajectory_files  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import poly, utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2_mod import GAFF2_mod  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.merz import MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.interface import (  # 导入本例需要的库或 yadonpy 接口。
    FixedChargeRegionSpec,
    GraphiteLayerSpec,
    LayerStackRelaxationSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    ZCompressionAnnealSpec,
    analyze_layer_stack_interface,
    build_layer_stack,
    run_layer_stack_relaxation,
)
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


# ---------------- user inputs ----------------
restart_status = True  # True 断点续跑；False 会清空 06 的 work_dir 并重新构建/采样。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # CMC、碳酸酯、PF6- 和石墨使用 GAFF2_mod，和其它 eg08 保持一致。
ion_ff = MERZ()  # Li+/Na+ 使用 Merz 离子参数。

temp = 318.15  # 温度 K；正式生产建议先固定这个值，避免和结构问题混在一起。
mpi = 1  # thread-MPI rank 数；当前 GROMACS-2026 thread_mpi 单卡通常用 1。
omp = 14  # OpenMP 线程数；按远端节点 CPU 分配调整。
gpu = 1  # 1 使用 GPU 加速，0 强制 CPU。
gpu_id = 0  # GPU 编号；多卡节点按实际可用卡修改。
run_sampling = True  # True 跑 20 ns 生产；若只想检查构建，把这里改 False。

# This script is meant for a real large-cell production test.  For a quick
# build-only check, set run_sampling=False above.
sample_ns = 20.0  # final NVT 采样时长 ns；这是给你早上收作业看的长跑默认值。

# Keep the first large flat run neutral.  Change this single value to run a
# fixed-charge basal-electrode case after the neutral structure looks healthy.
surface_charge_uC_cm2 = 0.0  # 第一轮大体系保持中性；确认结构健康后可改成 2.0、-2.0、5.0 等。

# ---------------- post-processing controls ----------------
analysis_profile = "interface_fast"  # 使用 slab/interface 分析预设，不跑不适合界面的 bulk 默认项。
interface_bin_nm = 0.05  # z-profile bin 宽；大体系统计足够时可降到 0.025。
interface_region_width_nm = 0.75  # 近石墨/混合/core 区域宽度；CMC 很厚时可试 1.0。
interface_surface_grid_nm = 0.50  # 石墨表面 xy occupancy map 网格尺寸。
graphite_adsorption_cutoff_nm = 0.50  # COM 距石墨表面小于 0.5 nm 计为近表面吸附。
penetration_threshold_nm = 0.20  # 进入 CMC/mixed 区至少 0.2 nm 才计为 penetration。
adsorption_min_residence_ps = 10.0  # adsorption summary 里通过驻留阈值的最短累计时间。
potential_reference = "zero_mean"  # 电势零点；若想看从盒子底部积分，可改 "zero_start"。
split_electrodes_for_edl = True  # 分开统计上下电极附近 EDL，适合 sandwich。
report_potential_drop = True  # 输出 potential drop 诊断；中性体系也可作为 sanity check。
compute_interface_transport = True  # 计算区域 MSD；Dxy 用于界面内扩散趋势。
time_series_analysis = True  # 打开时间序列 CSV/MP4，便于检查 20 ns 内结构是否压实。
interface_time_series_sample_count = 10  # 全轨迹按十分之一时长取 10 帧/窗口。
interface_time_series_fps = 1.0  # MP4 慢速播放，太快不利于判断结构演化。
interface_time_series_rdf = True  # 输出 RDF + CN 时间序列。
interface_time_series_concentration = True  # 输出 z 浓度 profile 时间序列。
interface_time_series_angles = True  # 输出吸附角度分布时间序列。
penetration_species = ("EC", "EMC", "DEC", "PF6", "Li", "Na")  # 渗入 CMC/混合区分析物种。
adsorption_species = ("EC", "EMC", "DEC")  # 石墨吸附取向分析物种；通常只看中性溶剂。
clean_trajectories_after_analysis = False  # True 会删轨迹省空间；长跑验收时应保持 False。

glucose6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"  # CMC 羧酸根重复单元。
ter_smiles = "[H][*]"  # 聚合物端基终止片段。
EC_smiles = "O=C1OCCO1"  # EC。
EMC_smiles = "CCOC(=O)OC"  # EMC。
DEC_smiles = "CCOC(=O)OCC"  # DEC。
Li_smiles = "[Li+]"  # Li+。
Na_smiles = "[Na+]"  # CMC-Na counterion。
PF6_smiles = "F[P-](F)(F)(F)(F)F"  # PF6-；下面必须 bonded="DRIH"，防止八面体角度畸变。

# Large flat-cell composition.
cmc_dp = 20  # 每条 CMC 链聚合度；本例目标是 DP=20。
cmc_chain_count = 8  # CMC 链数；本例目标是 8 条。
solvent_counts = (96, 72, 168)  # EC/EMC/DEC 个数；保持 4:3:7，适配大 XY 面积。
salt_pairs = 36  # LiPF6 离子对数；Li 和 PF6 同步增加保持电中性。
charge_scale = 0.7  # CMC/Na/Li/PF6 电荷缩放；想跑全电荷时改 1.0，但需重新验证稳定性。

BASE_DIR = Path(__file__).resolve().parent  # 例子所在目录。
work_dir = BASE_DIR / "work_dir" / "06_large_flat_charged_graphite_basal_electrolyte_cmcna_graphite_basal"  # 本例输出根目录。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 打印环境诊断，确认 RDKit/GROMACS/依赖可见。
    ensure_initialized()  # 确认 yadonpy 数据目录和 MolDB bundle 已初始化。
    work_dir = workdir(work_dir, restart=restart_status)  # 创建或复用本例输出目录。

    case_name = f"charge_{surface_charge_uC_cm2:+.1f}_uC_cm2".replace("+", "p").replace("-", "m").replace(".", "p")  # 把电荷值变成安全目录名。
    case_dir = work_dir.child(case_name)  # 每个电荷状态一个独立 case 目录。
    cmc_rw_dir = case_dir.child("00_cmc_rw")  # CMC 随机聚合中间文件目录。
    cmc_term_dir = case_dir.child("01_cmc_term")  # CMC 端基终止中间文件目录。

    glucose6 = ff.mol(glucose6_smiles, charge="RESP", prefer_db=True, require_ready=True, polyelectrolyte_mode=True)  # 从 MolDB 读取 CMC 重复单元 RESP。
    glucose6 = ff.ff_assign(glucose6, report=False)  # 给 CMC 重复单元分配 GAFF2_mod 参数。
    ter = utils.mol_from_smiles(ter_smiles)  # 构造聚合物端基，不需要 MolDB。
    EC = ff.mol(EC_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 MolDB 读取 EC。
    EC = ff.ff_assign(EC, report=False)  # 给 EC 分配力场。
    EMC = ff.mol(EMC_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 MolDB 读取 EMC。
    EMC = ff.ff_assign(EMC, report=False)  # 给 EMC 分配力场。
    DEC = ff.mol(DEC_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 MolDB 读取 DEC。
    DEC = ff.ff_assign(DEC, report=False)  # 给 DEC 分配力场。
    PF6 = ff.mol(PF6_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 MolDB 读取 PF6-。
    PF6 = ff.ff_assign(PF6, bonded="DRIH", report=False)  # 给 PF6- 分配 DRIH 键角和 bond-bond cross term。
    Li = ion_ff.mol(Li_smiles)  # 构造 Li+。
    Li = ion_ff.ff_assign(Li, report=False)  # 给 Li+ 分配 Merz 参数。
    Na = ion_ff.mol(Na_smiles)  # 构造 Na+。
    Na = ion_ff.ff_assign(Na, report=False)  # 给 Na+ 分配 Merz 参数。
    if not all((glucose6, ter, EC, EMC, DEC, PF6, Li, Na)):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("MolDB/FF assignment failed for one or more species.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    CMC = poly.random_copolymerize_rw([glucose6], cmc_dp, ratio=[1.0], tacticity="atactic", work_dir=cmc_rw_dir)  # 随机生成 DP=20 CMC 链。
    CMC = poly.terminate_rw(CMC, ter, work_dir=cmc_term_dir)  # 用 H 端基终止 CMC 链。
    CMC = ff.ff_assign(CMC, report=False)  # 给整条 CMC 链分配 GAFF2_mod 参数。
    if not CMC:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Cannot assign GAFF2_mod parameters to the DP=20 CMC-Na chain.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    graphite_bottom = GraphiteLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="GRAPHITE_BOTTOM",  # 下石墨层名；固定电荷区域按这个名字选层。
        nx=20,  # 基面 x 重复数；这是“扁盒子”的主要面积来源。
        ny=18,  # 基面 y 重复数；和 nx 一起给 DP20 x 8 CMC 足够 XY 面积。
        n_layers=3,  # 石墨层数；若要更硬/更厚电极可增加。
        orientation="basal",  # 基面石墨。
        periodic_xy=True,  # XY 周期成键，防止基面边缘断裂。
    )
    graphite_top = GraphiteLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="GRAPHITE_TOP",  # 上石墨层名；与下石墨组成 sandwich。
        nx=20,  # 与下石墨相同，保证上下电极面积一致。
        ny=18,  # 与下石墨相同。
        n_layers=3,  # 与下石墨相同。
        orientation="basal",  # 基面石墨。
        periodic_xy=True,  # XY 周期成键。
    )
    electrolyte = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="ELECTROLYTE",  # 电解液层名。
        species=(EC, EMC, DEC, Li, PF6),  # species 顺序必须对应 counts 和 charge_scale。
        counts=(*solvent_counts, salt_pairs, salt_pairs),  # EC/EMC/DEC/Li/PF6 数量。
        thickness_nm=2.2,  # 初始电解液目标厚度；真实厚度会由 z-NPT 调整。
        density_target_g_cm3=1.15,  # 初始电解液 packing 密度，略低于紧密液体以提高插入成功率。
        layer_kind="electrolyte",  # 电解液语义标签。
        charge_scale=(1.0, 1.0, 1.0, charge_scale, charge_scale),  # 溶剂全电荷，Li/PF6 缩放。
        large_system_mode="large",  # 强制使用大体系 packing 策略，减少大盒子随机插入失败。
    )
    cmcna = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="CMCNA",  # CMC-Na 层名，后处理识别 CMC-rich/core 区域。
        species=(CMC, Na),  # CMC 链和 Na+ counterion。
        counts=(cmc_chain_count, cmc_chain_count * cmc_dp),  # 8 条 DP20 链对应 160 个 Na+。
        thickness_nm=2.6,  # 初始 CMC 目标厚度；先薄而宽，保持扁盒子几何。
        # Loose insertion target: DP=20 x 8 chains should not be packed directly
        # at the bulk CMC-Na reference density.  Compression annealing and z-NPT
        # compact this layer after all molecules have been placed.
        density_target_g_cm3=0.75,  # 初始 CMC packing 要稀疏，后续靠压缩退火/z-NPT 致密化。
        layer_kind="cmcna",  # 启用 CMCNA 专用分组、Na+/COO- 和密度诊断。
        charge_scale=(charge_scale, charge_scale),  # CMC 与 Na+ 使用同一缩放，保持局部配对口径。
        polyelectrolyte_mode=True,  # 按聚电解质体系处理 CMC-Na。
        large_system_mode="large",  # 大体系 packing 策略。
        counterion_contact_mode="carboxylate",  # 构建后把 Na+ 放在 COO- 附近，避免初期跑进电解液。
    )
    stack = LayerStackSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        layers=(graphite_bottom, electrolyte, cmcna, graphite_top),  # 大扁盒子 sandwich 层顺序。
        order="bottom_to_top",  # 按 z 从下到上解释 layers。
        pbc_mode="xyz",  # 闭合三维周期；不是显式真空体系。
        name=f"large_flat_charged_graphite_stack_{case_name}",  # 系统名包含电荷状态。
        default_gap_nm=0.35,  # 层间初始间隙，避免 fresh overlap。
        molecular_packing_expand="z",  # 固定石墨 XY；若分子太多则扩展 z 而非改电极面积。
        fixed_charge_regions=(  # 定义固定电荷选区。
            FixedChargeRegionSpec(  # 开始一个多行函数调用或配置块。
                layer_name="GRAPHITE_BOTTOM",  # 下石墨层。
                region="top",  # 下石墨内侧表面。
                mode="surface_charge_density",  # 以 uC/cm2 指定固定面电荷。
                surface_charge_uC_cm2=surface_charge_uC_cm2,  # 下内表面电荷密度。
                elements=("C",),  # 只给碳原子分配电荷。
                label="bottom_graphite_inner_face",  # manifest 标签，方便检查选区。
            ),
            FixedChargeRegionSpec(  # 开始一个多行函数调用或配置块。
                layer_name="GRAPHITE_TOP",  # 上石墨层。
                region="bottom",  # 上石墨内侧表面。
                mode="surface_charge_density",  # 同样用面电荷密度。
                surface_charge_uC_cm2=-surface_charge_uC_cm2,  # 上内表面取相反电荷，构成电极对。
                elements=("C",),  # 只给碳原子分配电荷。
                label="top_graphite_inner_face",  # manifest 标签。
            ),
        ),
    )
    relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=sample_ns)  # 记录工作流默认温度和生产长度。

    result = build_layer_stack(stack=stack, relaxation=relaxation, work_dir=case_dir, restart=restart_status)  # 构建 GROMACS-ready stack。
    print(f"layer_stack_manifest = {result.manifest_path}")  # manifest 记录层顺序、密度、Na+/COO- 配对和电荷选区。
    print(f"stack_gmx_dir = {result.system_gro.parent}")  # system.gro/top/ndx 所在目录。
    print(f"acceptance = {result.acceptance}")  # packing 接受率/诊断摘要。

    analyze_layer_stack_interface(  # 对静态或轨迹界面结构执行后处理。
        work_dir=case_dir,  # 静态后处理写到 case_dir 下。
        manifest_path=result.manifest_path,  # 使用 manifest 的真实层信息。
        analysis_profile=analysis_profile,  # 界面分析预设。
        bin_nm=interface_bin_nm,  # z-profile bin 宽。
        region_width_nm=interface_region_width_nm,  # 近界面区域宽度。
        surface_grid_nm=interface_surface_grid_nm,  # xy occupancy 网格。
        surface_distance_nm=graphite_adsorption_cutoff_nm,  # 石墨近表面距离阈值。
        penetration_threshold_nm=penetration_threshold_nm,  # penetration 深度阈值。
        adsorption_min_residence_ps=adsorption_min_residence_ps,  # 吸附驻留阈值。
        potential_reference=potential_reference,  # 电势零点。
        split_electrodes=split_electrodes_for_edl,  # 分开上下电极。
        report_potential_drop=report_potential_drop,  # 输出电势差。
        penetration_species=penetration_species,  # penetration 物种。
        adsorption_species=adsorption_species,  # adsorption 物种。
        compute_transport=False,  # 静态结构没有 MSD，所以这里关闭 transport。
        time_series_analysis=False,  # 静态结构只有一帧，不生成时间序列。
    )

    if run_sampling:  # 根据当前状态决定是否进入该分支。
        relax = run_layer_stack_relaxation(  # 运行 layer-stack 松弛和 final NVT 采样。
            result,
            time_ns=sample_ns,  # final NVT 生产长度，默认 20 ns。
            temp=temp,  # 目标温度。
            mpi=mpi,  # thread-MPI rank 数。
            omp=omp,  # OpenMP 线程数。
            gpu=gpu,  # GPU 开关。
            gpu_id=gpu_id,  # GPU 编号。
            dt_ps=0.001,  # 1 fs 步长，优先保证大界面早期稳定。
            constraints="none",  # fresh interface 阶段不使用约束，减少初始失败。
            run_analysis=True,  # 运行后保留 analyze facade。
            relax_z=True,  # fixed-XY z-NPT，让扩展的 z 自然压回致密状态。
            pre_nvt_ns=0.10,  # z-NPT 前的短 NVT，释放局部接触。
            z_npt_ns=1.00,  # 最终生产前的 z-only NPT 时长。
            compression_anneal=ZCompressionAnnealSpec(  # 传入循环压缩退火配置。
                enabled=True,  # 大 CMC sandwich 必须启用循环压缩退火。
                cycles=12,  # 循环数；大体系用 12 轮小步压缩比 6-8 轮更稳。
                tmax_K=400.0,  # 最高退火温度；高于小体系以帮助 DP20 链重排。
                pmax_bar=4000.0,  # z-only 高压上限；用于压实初始稀疏 CMC。
                max_z_shrink_per_cycle=0.03,  # 每轮最多压 3%，比小体系更保守。
                hot_nvt_ns=0.02,  # 每轮高温 NVT 时长。
                compression_npt_ns=0.08,  # 每轮高压 z-NPT 时长。
                cool_nvt_ns=0.03,  # 每轮冷却回正常温度的 NVT 时长。
            ),
            restart=restart_status,  # 允许长任务断点续跑。
        )
        print(f"relaxation_summary = {relax.summary_path}")  # 打印关键路径或状态，便于人工检查。

        analy = relax.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        interface = analy.interface(  # 设置中间变量或可调参数，供后续工作流使用。
            manifest_path=result.manifest_path,  # final NVT 后处理仍用同一个层 manifest。
            analysis_profile=analysis_profile,  # 界面分析预设。
            bin_nm=interface_bin_nm,  # z-profile bin 宽。
            region_width_nm=interface_region_width_nm,  # 区域宽度。
            surface_grid_nm=interface_surface_grid_nm,  # xy occupancy 网格。
            surface_distance_nm=graphite_adsorption_cutoff_nm,  # 吸附距离阈值。
            penetration_threshold_nm=penetration_threshold_nm,  # penetration 深度阈值。
            adsorption_min_residence_ps=adsorption_min_residence_ps,  # 吸附驻留阈值。
            potential_reference=potential_reference,  # 电势零点。
            penetration_species=penetration_species,  # penetration 物种。
            adsorption_species=adsorption_species,  # adsorption 物种。
            split_electrodes=split_electrodes_for_edl,  # 分开上下电极。
            report_potential_drop=report_potential_drop,  # 输出电势差。
            compute_transport=compute_interface_transport,  # 是否计算 MSD/扩散。
            time_series_sample_count=interface_time_series_sample_count,  # 时间序列窗口数。
            time_series_fps=interface_time_series_fps,  # MP4 帧率。
            time_series_rdf=interface_time_series_rdf,  # 是否输出 RDF/CN 动画。
            time_series_concentration=interface_time_series_concentration,  # 是否输出浓度 profile 动画。
            time_series_angles=interface_time_series_angles,  # 是否输出角度分布动画。
        )
        health = interface.geometry_health(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        z_profile = interface.z_profiles(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        edl = interface.edl_profiles(  # 设置中间变量或可调参数，供后续工作流使用。
            split_electrodes=split_electrodes_for_edl,  # 设置中间变量或可调参数，供后续工作流使用。
            potential_reference=potential_reference,  # 设置电势 profile 的零点参考方式。
            report_potential_drop=report_potential_drop,  # 控制是否输出电势差诊断。
            time_series_analysis=time_series_analysis,  # 控制是否输出时间序列 CSV/MP4。
        )
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

    clean_md_trajectory_files(case_dir, enabled=clean_trajectories_after_analysis)  # 按配置清理 MD 轨迹文件。
