from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 08-03: basal graphite | CMC-Na | carbonate/LiPF6 electrolyte."""

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import clean_md_trajectory_files  # 导入本例需要的库或 yadonpy 接口。
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
    ZCompressionAnnealSpec,
    analyze_layer_stack_interface,
    build_layer_stack,
    run_layer_stack_relaxation,
)
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


# ---------------- user inputs ----------------
restart_status = True  # True 复用已有构建/MD 阶段；False 会删除本例 work_dir 后重跑。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # CMC repeat、碳酸酯、PF6- 使用 GAFF2_mod。
ion_ff = MERZ()  # Li+/Na+ 使用 Merz 离子参数。

temp = 318.15  # MD 温度 K；改变温度会同时影响松弛和采样。
mpi = 1  # thread-MPI rank 数。
omp = 14  # OpenMP 线程数。
gpu = 1  # 1 使用 GPU；无 GPU 时改 0。
gpu_id = 0  # GPU 设备编号。
run_sampling = True  # True 跑 relaxation 和 final NVT；False 只构建/静态后处理。
sample_ns = 2.0  # final NVT 采样 ns；界面统计正式跑应更长。
analysis_profile = "interface_fast"  # slab/interface 专用快速分析预设。
interface_bin_nm = 0.05  # z-profile 分辨率。
interface_region_width_nm = 0.75  # graphite-near、mixed、core 区域宽度。
graphite_adsorption_cutoff_nm = 0.50  # COM 到石墨表面小于该值计为吸附/近表面。
time_series_analysis = True  # 生成按时间分段的动画和 CSV。
interface_time_series_sample_count = 10  # 总轨迹分成 10 个时间窗口。
interface_time_series_fps = 1.0  # 慢速 MP4 帧率。
interface_time_series_rdf = True  # True 输出全体系 cation RDF/CN 和 graphite-EDL RDF/CN；RDF 实线、CN 虚线。
interface_time_series_concentration = True  # True 输出 z 浓度/密度 profile 时间序列，用来看 CMC/电解液界面是否提前模糊。
interface_time_series_angles = True  # True 输出 graphite-EDL 吸附取向角分布时间序列；纵坐标为角度窗口百分比。
interface_time_series_charge_potential = True  # True 输出 graphite/EDL 电荷密度、积分电荷和电势 profile 的 CSV/PNG/MP4。
penetration_species = ("EC", "EMC", "DEC", "PF6", "Li", "Na")  # 渗入 CMC/混合区的物种。
adsorption_species = ("EC", "EMC", "DEC")  # 石墨吸附统计的中性溶剂分子。
clean_trajectories_after_analysis = False  # True 可清理大轨迹；检查结构时保持 False。

glucose6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"  # CMC 带羧酸根的重复单元。
ter_smiles = "[H][*]"  # 聚合物端基占位；用于终止随机链。
EC_smiles = "O=C1OCCO1"  # EC。
EMC_smiles = "CCOC(=O)OC"  # EMC。
DEC_smiles = "CCOC(=O)OCC"  # DEC。
Li_smiles = "[Li+]"  # Li+。
Na_smiles = "[Na+]"  # CMC 羧酸根的 Na+ counterion。
PF6_smiles = "F[P-](F)(F)(F)(F)F"  # PF6-；使用 DRIH 保持八面体构型。

cmc_dp = 5  # 每条 CMC 链聚合度；增大需要同步放大石墨面积或允许 z 扩展。
cmc_chain_count = 1  # CMC 链数；Na+ 数量会自动设成 chain_count * dp。
solvent_counts = (8, 6, 14)  # EC/EMC/DEC 数量。
salt_pairs = 3  # LiPF6 离子对数。
charge_scale = 0.7  # CMC-Na 和盐离子电荷缩放；全电荷可改 1.0。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = BASE_DIR / "work_dir" / "03_electrolyte_cmcna_graphite_basal"  # 设置本例输出目录。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。
    work_dir = workdir(work_dir, restart=restart_status)  # 设置本例输出目录。
    cmc_rw_dir = work_dir.child("00_cmc_rw")  # 设置中间变量或可调参数，供后续工作流使用。
    cmc_term_dir = work_dir.child("01_cmc_term")  # 设置中间变量或可调参数，供后续工作流使用。

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
        name="GRAPHITE_BASAL",  # 石墨层名。
        nx=6,  # 基面 x 重复数。
        ny=5,  # 基面 y 重复数。
        n_layers=3,  # 石墨层数。
        orientation="basal",  # 基面接触 CMC/electrolyte。
        periodic_xy=True,  # XY 周期成键，避免基面边缘断键。
    )
    cmcna = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="CMCNA",  # CMC-Na 层名；后处理用 CMCNA 识别聚合物区域。
        species=(CMC, Na),  # CMC 链和 Na+ counterion。
        counts=(cmc_chain_count, cmc_chain_count * cmc_dp),  # Na+ 数等于羧酸根数，保持局部电中性。
        thickness_nm=1.8,  # 初始 CMC 层厚目标。
        # Initial packing target only: keep insertion looser than bulk CMC-Na
        # (~1.5 g/cm3), then let compression annealing/z-NPT densify the layer.
        density_target_g_cm3=1.0,  # 初始 packing 密度，不强行达到 bulk 约 1.5 g/cm3。
        layer_kind="cmcna",  # 触发 CMC-Na 专用分组和诊断。
        charge_scale=(charge_scale, charge_scale),  # CMC 和 Na+ 使用同一缩放。
        polyelectrolyte_mode=True,  # 按聚电解质处理 CMC 链和 counterion。
        # Place Na+ at local carboxylate contact sites after loose packing so
        # early relaxation starts from CMC-Na ion pairs, not free Na in the layer.
        counterion_contact_mode="carboxylate",  # 初始把 Na+ 放到 COO- 附近，避免松弛时跑进电解液。
    )
    electrolyte = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="ELECTROLYTE",  # 电解液层名。
        species=(EC, EMC, DEC, Li, PF6),  # species 顺序对应 counts/charge_scale。
        counts=(*solvent_counts, salt_pairs, salt_pairs),  # EC/EMC/DEC/Li/PF6 个数。
        thickness_nm=2.2,  # 初始电解液层厚。
        density_target_g_cm3=1.2,  # 初始电解液 packing 目标密度。
        layer_kind="electrolyte",  # 电解液语义类型。
        charge_scale=(1.0, 1.0, 1.0, charge_scale, charge_scale),  # 溶剂全电荷，Li/PF6 缩放。
    )
    stack = LayerStackSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        layers=(graphite, cmcna, electrolyte),  # 底部石墨，中间 CMC-Na，上方电解液。
        order="bottom_to_top",  # 按 z 从下到上解释 layers。
        pbc_mode="xyz",  # 闭合三维周期。
        name="electrolyte_cmcna_graphite_basal",  # 输出系统名。
        default_gap_nm=0.35,  # 层间初始间隙。
        molecular_packing_expand="z",  # 固定石墨 XY，分子过多时扩展 z。
    )
    relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=sample_ns)  # 记录工作流默认温度和采样时长。

    result = build_layer_stack(stack=stack, relaxation=relaxation, work_dir=work_dir, restart=restart_status)  # 构建 layer-stack 初始结构、拓扑和索引。
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
            mpi=mpi,  # MPI rank 数。
            omp=omp,  # OpenMP 线程。
            gpu=gpu,  # GPU 开关。
            gpu_id=gpu_id,  # GPU 编号。
            run_analysis=True,  # 生成 relaxation 后处理入口。
            relax_z=True,  # 让 z 方向通过 NPT 自然压缩/拉伸。
            z_compressibility_bar_inv=4.5e-6,  # final z-NPT 用小有效压缩率，避免高聚物界面盒子突变。
            z_npt_tau_p_ps=20.0,  # final z-NPT 使用慢 barostat，和循环压缩阶段保持同等保守。
            compression_anneal=ZCompressionAnnealSpec(  # 传入循环压缩退火配置。
                enabled=True,  # CMC/电解液界面启用循环压缩退火。
                cycles=8,  # 压缩循环数；体系越大可适当增加。
                tmax_K=380.0,  # 热退火最高温度；过高会增加构象扰动。
                pmax_bar=3000.0,  # z-only 高压上限；越高压实越强但风险越大。
                max_z_shrink_per_cycle=0.04,  # 每轮几何压缩最多 4%，避免一次性压坏。
            ),
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
            time_series_rdf=interface_time_series_rdf,  # 控制全体系 RDF/CN 与 graphite-EDL RDF/CN 时间序列。
            time_series_concentration=interface_time_series_concentration,  # 控制 z-profile 时间序列和动画。
            time_series_angles=interface_time_series_angles,  # 控制石墨 EDL 吸附取向角时间序列。
            time_series_charge_potential=interface_time_series_charge_potential,  # 控制 graphite/EDL 电荷-电势时间序列。
        )
        health = interface.geometry_health(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        z_profile = interface.z_profiles(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        edl = interface.edl_profiles(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        penetration = interface.penetration(species=penetration_species, time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        membrane = interface.membrane_permeation(species=penetration_species, time_series_analysis=time_series_analysis)  # 输出 CMCNA 膜/隔膜渗透诊断；包括进入事件、膜内驻留、uptake、表观 flux/permeability。
        adsorption = interface.graphite_adsorption(  # 设置中间变量或可调参数，供后续工作流使用。
            species=adsorption_species,  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。
            time_series_analysis=time_series_analysis,  # 控制是否输出时间序列 CSV/MP4。
        )
        coordination = interface.coordination_by_region(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        transport = interface.region_transport(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        time_series = interface.time_series(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        summary = interface.summary(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        edl_rdf = (time_series.get("outputs") or {}).get("edl_rdf_cn") or {}  # 提取 graphite-EDL RDF/CN 输出，便于确认新统计是否产出。
        charge_potential = (time_series.get("outputs") or {}).get("charge_potential") or {}  # 提取 graphite/EDL 电荷-电势输出。
        print(f"interface_phase_order_ok = {health.get('phase_order_ok')}")  # 打印关键路径或状态，便于人工检查。
        print(f"membrane_permeation_available = {membrane.get('available')}")  # 确认膜渗透统计是否识别到 CMCNA/polymer 区域。
        print(f"interface_outputs = {summary.get('outputs', {}).get('interface_profile_summary_json')}")  # 打印关键路径或状态，便于人工检查。
        print(f"interface_time_series = {time_series.get('outputs', {})}")  # 打印关键路径或状态，便于人工检查。
        print(f"edl_rdf_cn_available = {edl_rdf.get('available')}")  # True 表示 graphite-EDL RDF/CN CSV/MP4 正常生成。
        print(f"edl_rdf_cn_outputs = {edl_rdf.get('curves_csv') or edl_rdf.get('reason')}")  # 打印曲线 CSV 路径或失败原因。
        print(f"charge_potential_available = {charge_potential.get('available')}")  # True 表示电荷-电势 CSV/MP4 正常生成。
        print(f"charge_potential_outputs = {charge_potential.get('csv') or charge_potential.get('reason')}")  # 打印电荷-电势 CSV 路径或失败原因。

    clean_md_trajectory_files(work_dir, enabled=clean_trajectories_after_analysis)  # 按配置清理 MD 轨迹文件。
