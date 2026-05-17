from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 08-05: fixed-charge graphite sweep for a four-layer stack.

The fixed-charge model is not constant potential.  It distributes a prescribed
surface charge density onto the interior graphite faces, leaving the outer
faces neutral.  The default sweep is small and intended for fast EDL sanity
checks before committing to larger production cells.
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
restart_status = True  # True 断点续跑；False 会清空本例 work_dir 后完整重建。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # CMC、碳酸酯、PF6- 和石墨使用 GAFF2_mod。
ion_ff = MERZ()  # Li+/Na+ 使用 Merz 离子参数。

temp = 318.15  # MD 温度 K；影响预松弛、z-NPT 和 final NVT。
mpi = 1  # thread-MPI rank 数；单 GPU 常用 1。
omp = 14  # OpenMP 线程数；按 CPU 资源调整。
gpu = 1  # GPU 开关；无 GPU 改成 0。
gpu_id = 0  # GPU 编号；多卡时可改成 1、2 等。
run_sampling = True  # True 跑每个电荷点的 relaxation/采样/后处理；False 只构建初始结构。
sample_ns = 2.0  # 每个电荷点 final NVT 时长 ns；正式 EDL 统计建议更长。
surface_charge_sweep_uC_cm2 = (0.0, 2.0, -2.0, 5.0, -5.0)  # 固定电荷面密度 sweep；先调试可只留 (0.0,)。

# ---------------- post-processing controls ----------------
# `analysis_profile="interface_fast"` keeps the interface-specific analyses
# focused on robust slab observables instead of bulk 3D transport defaults.
analysis_profile = "interface_fast"  # 界面 slab 专用快速分析预设。

# z-bin width for density, charge, EDL species, integrated charge, and
# electrostatic-potential profiles. Smaller bins give sharper interfaces but
# need more trajectory frames to reduce noise.
interface_bin_nm = 0.05  # z 方向密度/电荷/potential profile bin 宽。

# Width of automatically named near-interface regions. For this four-layer
# stack, the manifest defines layer order, then z-quantiles and density overlap
# define graphite-near, CMC/electrolyte mixed, and core-like regions.
interface_region_width_nm = 0.75  # 自动 near/mixed/core 区域宽度。

# xy grid used for graphite adsorption occupancy maps on the basal planes.
interface_surface_grid_nm = 0.50  # 石墨吸附 xy occupancy map 网格尺寸。

# A molecule is counted as graphite-near adsorbed when its mass-weighted COM is
# within this distance of the nearest graphite quantile surface. This is a
# residence/geometric diagnostic, not a binding free energy.
graphite_adsorption_cutoff_nm = 0.50  # 分子 COM 距石墨表面小于此值计为 graphite-near。

# Minimum COM depth inside a CMC/polymer-rich or mixed region before a frame is
# counted as penetration. This avoids counting molecules that merely touch the
# region boundary due to z-bin noise.
penetration_threshold_nm = 0.20  # 进入 CMC/mixed 区域的最小深度，避免边界噪声。

# Minimum cumulative adsorbed residence used for the `passes_min_residence`
# flag in `adsorption_summary.json`; the raw frame fractions are always written.
adsorption_min_residence_ps = 10.0  # 吸附驻留通过标志的最小累计时间。

# Potential is obtained by one-dimensional integration of sampled fixed-charge
# density using vacuum permittivity. `zero_mean` removes the mean potential;
# `zero_start` pins the first z bin. This is not a constant-potential electrode
# solver and should be interpreted as a slab diagnostic.
potential_reference = "zero_mean"  # 电势零点；可改 "zero_start" 固定第一个 bin。
split_electrodes_for_edl = True  # True 分别报告上下石墨附近 EDL。
report_potential_drop = True  # True 额外输出跨界面 potential drop 诊断。

# Compute anisotropic MSD summaries from the NVT trajectory. Dxy is the main
# in-plane interface mobility metric; Dz is confined-direction mobility.
compute_interface_transport = True  # True 计算区域内各向异性 MSD，Dxy 是主要界面迁移指标。

# Generate slow MP4 animations for interface time evolution. This is off by
# default in the API and must be passed explicitly to each post-processing call.
# The trajectory is split into ten equal time windows by default, so RDF/CN,
# molecule-COM z concentration, and adsorbed-angle distributions are sampled at
# roughly every one-tenth of the total NVT duration instead of producing a dense
# movie.
time_series_analysis = True  # True 生成时间序列 CSV/MP4。
interface_time_series_sample_count = 10  # 轨迹分成 10 个时间窗口。
interface_time_series_fps = 1.0  # MP4 帧率；慢速便于看趋势。
interface_time_series_rdf = True  # True 输出全体系 cation RDF/CN 和 graphite-EDL RDF/CN；RDF 实线、CN 虚线且 CN 轴为 0-6。
interface_time_series_concentration = True  # True 输出 z 浓度分布时间序列动画。
interface_time_series_angles = True  # True 输出吸附角度分布时间序列动画。

penetration_species = ("EC", "EMC", "DEC", "PF6", "Li", "Na")  # 进入 CMC/混合区的物种列表。
adsorption_species = ("EC", "EMC", "DEC")  # 石墨吸附取向统计物种；通常不放离子。
clean_trajectories_after_analysis = False  # True 会清理大轨迹；复查结构时保持 False。

glucose6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"  # CMC 羧酸根重复单元。
ter_smiles = "[H][*]"  # 随机聚合后的端基终止片段。
EC_smiles = "O=C1OCCO1"  # EC。
EMC_smiles = "CCOC(=O)OC"  # EMC。
DEC_smiles = "CCOC(=O)OCC"  # DEC。
Li_smiles = "[Li+]"  # Li+。
Na_smiles = "[Na+]"  # CMC counterion。
PF6_smiles = "F[P-](F)(F)(F)(F)F"  # PF6-；必须用 DRIH 角项保持八面体。

cmc_dp = 5  # 每条 CMC 链聚合度；增大时同步增加 Na+ 和盒子尺寸。
cmc_chain_count = 1  # CMC 链数。
solvent_counts = (8, 6, 14)  # EC/EMC/DEC 个数。
salt_pairs = 3  # LiPF6 离子对数。
charge_scale = 0.7  # CMC/Na/Li/PF6 电荷缩放。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = BASE_DIR / "work_dir" / "05_charged_graphite_basal_electrolyte_cmcna_graphite_basal"  # 设置本例输出目录。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。
    work_dir = workdir(work_dir, restart=restart_status)  # 设置本例输出目录。

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
        raise RuntimeError("MolDB/FF assignment failed for one or more species.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    for surface_charge in surface_charge_sweep_uC_cm2:  # 遍历当前工作流中的一组对象或任务。
        case_name = f"charge_{surface_charge:+.1f}_uC_cm2".replace("+", "p").replace("-", "m").replace(".", "p")  # 设置中间变量或可调参数，供后续工作流使用。
        case_dir = work_dir.child(case_name)  # 设置中间变量或可调参数，供后续工作流使用。
        cmc_rw_dir = case_dir.child("00_cmc_rw")  # 设置中间变量或可调参数，供后续工作流使用。
        cmc_term_dir = case_dir.child("01_cmc_term")  # 设置中间变量或可调参数，供后续工作流使用。

        CMC = poly.random_copolymerize_rw([glucose6], cmc_dp, ratio=[1.0], tacticity="atactic", work_dir=cmc_rw_dir)  # 用随机游走生成聚合物链。
        CMC = poly.terminate_rw(CMC, ter, work_dir=cmc_term_dir)  # 给聚合物链加端基。
        CMC = ff.ff_assign(CMC, report=False)  # 分配力场参数并写入分子属性。
        if not CMC:  # 根据当前状态决定是否进入该分支。
            raise RuntimeError("Cannot assign GAFF2_mod parameters to the CMC-Na chain.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

        graphite_bottom = GraphiteLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
            name="GRAPHITE_BOTTOM",  # 下石墨层名；固定电荷选择器按这个名字定位。
            nx=6,  # 基面 x 重复数；增大可扩大电极面积。
            ny=5,  # 基面 y 重复数。
            n_layers=3,  # 石墨层数；正式电极可加厚。
            orientation="basal",  # 基面模型。
            periodic_xy=True,  # XY 周期石墨，保证基面周期成键。
        )
        graphite_top = GraphiteLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
            name="GRAPHITE_TOP",  # 上石墨层名；固定电荷选择器按这个名字定位。
            nx=6,  # 与下石墨一致，保持上下电极面积相同。
            ny=5,  # 与下石墨一致。
            n_layers=3,  # 与下石墨一致。
            orientation="basal",  # 基面模型。
            periodic_xy=True,  # XY 周期成键。
        )
        electrolyte = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
            name="ELECTROLYTE",  # 电解液层名。
            species=(EC, EMC, DEC, Li, PF6),  # species 顺序对应 counts/charge_scale。
            counts=(*solvent_counts, salt_pairs, salt_pairs),  # EC/EMC/DEC/Li/PF6 数量。
            thickness_nm=2.2,  # 初始电解液层厚。
            density_target_g_cm3=1.2,  # 初始 packing 目标密度。
            layer_kind="electrolyte",  # 电解液语义类型。
            charge_scale=(1.0, 1.0, 1.0, charge_scale, charge_scale),  # 溶剂全电荷，离子缩放。
        )
        cmcna = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
            name="CMCNA",  # CMC-Na 层名；后处理按 CMCNA 识别聚合物相。
            species=(CMC, Na),  # CMC 链和 Na+ counterion。
            counts=(cmc_chain_count, cmc_chain_count * cmc_dp),  # Na+ 数等于总羧酸根数。
            thickness_nm=1.8,  # 初始 CMC 层厚。
            # Initial packing target only: keep insertion looser than bulk
            # CMC-Na (~1.5 g/cm3), then densify via compression annealing/z-NPT.
            density_target_g_cm3=1.0,  # 初始 packing 目标，不是 final 密度约束。
            layer_kind="cmcna",  # 启用 CMCNA 分组和密度诊断。
            charge_scale=(charge_scale, charge_scale),  # CMC 和 Na 同步缩放。
            polyelectrolyte_mode=True,  # 聚电解质模式。
            # Start Na+ next to CMC carboxylates so the charged-interface
            # relaxation preserves physically meaningful local counterions.
            counterion_contact_mode="carboxylate",  # 初始让 Na+ 接触 COO-，防止跑入电解液。
        )
        stack = LayerStackSpec(  # 设置中间变量或可调参数，供后续工作流使用。
            layers=(graphite_bottom, electrolyte, cmcna, graphite_top),  # 四层 sandwich，从下到上。
            order="bottom_to_top",  # 明确 z 方向层顺序。
            pbc_mode="xyz",  # 闭合三维周期。
            name=f"charged_graphite_stack_{case_name}",  # 每个 surface charge 单独命名。
            default_gap_nm=0.35,  # 层间初始间隙。
            molecular_packing_expand="z",  # 固定石墨 XY，分子多时扩展 z。
            # Constant-charge, not constant-potential: assign the prescribed
            # areal charge to the two graphite faces that touch the confined
            # electrolyte/CMC region.  `FixedChargeRegionSpec` is deliberately
            # layer- and region-based, so the same interface can later target a
            # graphite edge face, an amorphous layer slab (`region="z_range"`),
            # or a finite-thickness top/bottom region by changing only this
            # block.  The generated topology carries these charges through
            # compression annealing, final z-NPT, and final NVT sampling.
            fixed_charge_regions=(  # 定义固定电荷选区。
                FixedChargeRegionSpec(  # 开始一个多行函数调用或配置块。
                    layer_name="GRAPHITE_BOTTOM",  # 只给下石墨层施加这一侧电荷。
                    region="top",  # 下石墨的内侧表面是 top。
                    mode="surface_charge_density",  # 用面电荷密度自动换算总电荷。
                    surface_charge_uC_cm2=surface_charge,  # 下内表面 charge density。
                    elements=("C",),  # 只选择碳原子。
                    label="bottom_graphite_inner_face",  # manifest 中的可读标签。
                ),
                FixedChargeRegionSpec(  # 开始一个多行函数调用或配置块。
                    layer_name="GRAPHITE_TOP",  # 只给上石墨层施加这一侧电荷。
                    region="bottom",  # 上石墨的内侧表面是 bottom。
                    mode="surface_charge_density",  # 仍按面电荷密度指定。
                    surface_charge_uC_cm2=-surface_charge,  # 上内表面取相反电荷，保持电极对。
                    elements=("C",),  # 只选择碳原子。
                    label="top_graphite_inner_face",  # manifest 标签。
                ),
            ),
        )
        relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=sample_ns)  # 记录温度和采样长度。

        result = build_layer_stack(stack=stack, relaxation=relaxation, work_dir=case_dir, restart=restart_status)  # 构建 layer-stack 初始结构、拓扑和索引。
        print(f"[{surface_charge:+.1f} uC/cm2] layer_stack_manifest = {result.manifest_path}")  # 打印关键路径或状态，便于人工检查。
        print(f"[{surface_charge:+.1f} uC/cm2] stack_gmx_dir = {result.system_gro.parent}")  # 打印关键路径或状态，便于人工检查。
        print(f"[{surface_charge:+.1f} uC/cm2] acceptance = {result.acceptance}")  # 打印关键路径或状态，便于人工检查。

        # Static-stack post-processing: this reads the freshly built `system.gro`
        # plus `system.top`, `system.ndx`, and `layer_stack_manifest.json`. It is
        # a geometry/charge sanity pass before NVT sampling:
        #   - `manifest_path` preserves the intended bottom-to-top layer order,
        #     which is more reliable than raw z-quantiles when an xyz-periodic
        #     stack wraps around the z boundary.
        #   - `bin_nm` controls the z histogram resolution for mass density,
        #     charge density, EDL species layering, integrated charge, and the
        #     fixed-charge electrostatic-potential diagnostic.
        #   - `region_width_nm` controls the width of graphite-near, mixed, and
        #     core-like z regions used by enrichment, penetration, coordination,
        #     and region transport summaries.
        #   - `surface_distance_nm` and `surface_grid_nm` define graphite
        #     adsorption occupancy: molecule COM within the cutoff of a graphite
        #     surface is counted, and x/y locations are binned on this grid.
        #   - `penetration_threshold_nm` requires a molecule COM to sit at least
        #     this far inside a CMC/polymer-rich or mixed region before counting
        #     it as penetrated.
        #   - `potential_reference`, `split_electrodes`, and
        #     `report_potential_drop` annotate the fixed-charge EDL diagnostic.
        #     The potential is a 1D integral of sampled charge density, not a
        #     constant-potential or Poisson-Boltzmann electrode solution.
        #   - `time_series_analysis=False` is used here because the static stack
        #     has only one coordinate frame; time-series MP4s are generated from
        #     the sampled NVT trajectory below by explicitly passing
        #     `time_series_analysis=True` to the facade methods.
        analyze_layer_stack_interface(  # 对静态或轨迹界面结构执行后处理。
            work_dir=case_dir,  # 设置本例输出目录。
            manifest_path=result.manifest_path,  # 指定 layer_stack_manifest.json 路径。
            analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
            bin_nm=interface_bin_nm,  # 指定 z-profile bin 宽。
            region_width_nm=interface_region_width_nm,  # 指定界面区域宽度。
            surface_grid_nm=interface_surface_grid_nm,  # 指定表面 xy 网格尺寸。
            surface_distance_nm=graphite_adsorption_cutoff_nm,  # 指定石墨近表面距离阈值。
            penetration_threshold_nm=penetration_threshold_nm,  # 设置分子进入目标区域的最小深度阈值。
            adsorption_min_residence_ps=adsorption_min_residence_ps,  # 设置吸附驻留通过标志的最小累计时间。
            potential_reference=potential_reference,  # 设置电势 profile 的零点参考方式。
            split_electrodes=split_electrodes_for_edl,  # 设置中间变量或可调参数，供后续工作流使用。
            report_potential_drop=report_potential_drop,  # 控制是否输出电势差诊断。
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
                mpi=mpi,  # MPI rank。
                omp=omp,  # OpenMP 线程。
                gpu=gpu,  # GPU 开关。
                gpu_id=gpu_id,  # GPU 编号。
                dt_ps=0.001,  # 1 fs 步长，提高紧凑界面早期稳定性。
                constraints="none",  # 初期不加约束，减少 fresh interface 失败。
                run_analysis=True,  # 生成分析入口。
                relax_z=True,  # fixed-XY z-NPT，让层厚自然调整。
                compression_anneal=ZCompressionAnnealSpec(  # 传入循环压缩退火配置。
                    enabled=True,  # sandwich 体系启用循环压缩退火。
                    cycles=8,  # 循环数；大体系可增到 10-12。
                    tmax_K=380.0,  # 退火最高温度。
                    pmax_bar=3000.0,  # z-only 压力上限。
                    max_z_shrink_per_cycle=0.04,  # 每轮最大几何压缩 4%。
                ),
                restart=restart_status,  # 断点续跑。
            )
            print(f"[{surface_charge:+.1f} uC/cm2] relaxation_summary = {relax.summary_path}")  # 打印关键路径或状态，便于人工检查。

            # Sampled-trajectory post-processing facade. `relax.analyze()` resolves
            # the final NVT `md.gro`, `md.tpr`, `md.edr`, topology, index file,
            # and coordinate stream (`md.xtc` by default; `md.trr` is used when
            # the trajectory policy requests full-precision coordinates).
            #
            # The facade keeps expensive work cached per parameter set and lets
            # the script ask physical questions explicitly:
            #   geometry_health(): intended vs sampled layer order, interphase
            #     distances, severe-overlap flags, and direct graphite/electrolyte
            #     contact checks.
            #   z_profiles(): phase/moltype mass density, charge density, number
            #     density, and phase z-quantiles.
            #   edl_profiles(): fixed-charge EDL species profiles, integrated
            #     charge, electric field, and reference-shifted potential.
            #   penetration(...): molecule COM residence in CMC/polymer-rich or
            #     mixed regions, filtered by `penetration_threshold_nm`.
            #   membrane_permeation(...): separator/membrane-specific uptake,
            #     entry events, residence, finite-slab flux, and apparent
            #     permeability diagnostics for liquid species entering CMCNA.
            #   graphite_adsorption(...): graphite-near residence, surface
            #     occupancy map, and simple carbonyl/dipole orientation proxies.
            #   coordination_by_region(): cation donor-state partitioning by z
            #     region using fallback Li/Na-O/F cutoffs.
            #   region_transport(): anisotropic MSD summaries; use Dxy for
            #     in-plane interface mobility and Dz only as confined mobility.
            #   time_series(): slow MP4 animations and CSV data sampled by
            #     trajectory deciles. The RDF outputs include global
            #     cation-centered RDF/CN plus graphite-EDL RDF/CN, where EDL
            #     centers are inside the graphite cutoff, targets use one
            #     strongest opposite-charge site, CN is dashed on a 0-6 axis,
            #     and the first RDF peak is labeled.
            analy = relax.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
            interface = analy.interface(  # 设置中间变量或可调参数，供后续工作流使用。
                manifest_path=result.manifest_path,  # 指定 layer_stack_manifest.json 路径。
                analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
                bin_nm=interface_bin_nm,  # 指定 z-profile bin 宽。
                region_width_nm=interface_region_width_nm,  # 指定界面区域宽度。
                surface_grid_nm=interface_surface_grid_nm,  # 指定表面 xy 网格尺寸。
                surface_distance_nm=graphite_adsorption_cutoff_nm,  # 指定石墨近表面距离阈值。
                penetration_threshold_nm=penetration_threshold_nm,  # 设置分子进入目标区域的最小深度阈值。
                adsorption_min_residence_ps=adsorption_min_residence_ps,  # 设置吸附驻留通过标志的最小累计时间。
                potential_reference=potential_reference,  # 设置电势 profile 的零点参考方式。
                penetration_species=penetration_species,  # 列出参与 penetration/区域分布统计的物种。
                adsorption_species=adsorption_species,  # 列出参与石墨吸附/取向统计的物种。
                split_electrodes=split_electrodes_for_edl,  # 设置中间变量或可调参数，供后续工作流使用。
                report_potential_drop=report_potential_drop,  # 控制是否输出电势差诊断。
                compute_transport=compute_interface_transport,  # 控制是否计算 transport/MSD。
                time_series_sample_count=interface_time_series_sample_count,  # 设置时间序列窗口数。
                time_series_fps=interface_time_series_fps,  # 设置时间序列动画帧率。
                time_series_rdf=interface_time_series_rdf,  # 控制全体系 RDF/CN 与 graphite-EDL RDF/CN 时间序列。
                time_series_concentration=interface_time_series_concentration,  # 控制 z-profile 时间序列和动画。
                time_series_angles=interface_time_series_angles,  # 控制石墨 EDL 吸附取向角时间序列。
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
            print(f"[{surface_charge:+.1f} uC/cm2] interface_phase_order_ok = {health.get('phase_order_ok')}")  # 打印关键路径或状态，便于人工检查。
            print(f"[{surface_charge:+.1f} uC/cm2] membrane_permeation_available = {membrane.get('available')}")  # 确认膜渗透统计是否识别到 CMCNA/polymer 区域。
            print(  # 打印关键路径或状态，便于人工检查。
                f"[{surface_charge:+.1f} uC/cm2] interface_outputs = "
                f"{summary.get('outputs', {}).get('interface_profile_summary_json')}"
            )
            print(f"[{surface_charge:+.1f} uC/cm2] interface_time_series = {time_series.get('outputs', {})}")  # 打印关键路径或状态，便于人工检查。
            print(f"[{surface_charge:+.1f} uC/cm2] edl_rdf_cn_available = {edl_rdf.get('available')}")  # True 表示 graphite-EDL RDF/CN CSV/MP4 正常生成。
            print(f"[{surface_charge:+.1f} uC/cm2] edl_rdf_cn_outputs = {edl_rdf.get('curves_csv') or edl_rdf.get('reason')}")  # 打印曲线 CSV 路径或失败原因。

        clean_md_trajectory_files(case_dir, enabled=clean_trajectories_after_analysis)  # 按配置清理 MD 轨迹文件。
