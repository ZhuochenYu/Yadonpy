from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 08-07: CMC-Na/electrolyte xy slabs first, then matched stack.

This example inverts the usual layer-stack sizing logic.  It first prepares a
wall-confined CMC-Na slab with ``pbc=xy`` and no z periodicity, reads the relaxed
slab XY box, chooses graphite basal-plane repeat counts that match that XY
footprint, prepares an electrolyte slab at the same XY footprint with the same
z-open wall protocol, and only then assembles graphite | electrolyte | CMC-Na |
graphite.  Temporary phase gates are kept during pre-release relaxation and
removed from final NVT, so the first final NVT frame is the electrolyte/CMC
interdiffusion t=0.
"""

import math  # 用于把 CMC slab 的连续 XY 尺寸转成石墨晶格可兼容的 nx/ny。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import (  # 导入本例需要的库或 yadonpy 接口。
    CMCNAXYSlabRelaxationSpec,
    XYSlabEquilibrationSpec,
    clean_md_trajectory_files,
    prepare_cmcna_xy_bulk_slab,
)
from yadonpy.core import poly, utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.graphite import build_graphite  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2_mod import GAFF2_mod  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.merz import MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.interface import (  # 导入本例需要的库或 yadonpy 接口。
    FixedChargeRegionSpec,
    GraphiteLayerSpec,
    GraphiteRestraintSpec,
    InterdiffusionStartSpec,
    LayerStackRelaxationSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    ZCompressionAnnealSpec,
    analyze_layer_stack_interface,
    build_layer_stack,
    make_orthorhombic_pack_cell,
    run_layer_stack_relaxation,
)
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.preset import eq  # 导入 EQ21/NPT preset；这里用于 CMC-Na 的 xy-slab 预平衡。


def read_gro_xy_nm(gro_path: Path) -> tuple[float, float]:  # 从 GROMACS GRO 末行读取 x/y 盒长，单位 nm。
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()  # 读取 GRO 文本。
    if len(lines) < 3:  # GRO 至少应有标题、原子数和 box 行。
        raise ValueError(f"Invalid GRO file: {gro_path}")  # 文件格式错误时立即报错。
    parts = lines[-1].split()  # box 行通常前三列是 x/y/z。
    if len(parts) < 2:  # 至少要有 x/y。
        raise ValueError(f"Cannot read XY box from GRO file: {gro_path}")  # 给出清晰报错。
    return float(parts[0]), float(parts[1])  # 返回 x_nm, y_nm。


def rdkit_mol_mass_amu(mol) -> float:  # 直接按 RDKit 原子质量求分子质量，适合已经展开成全原子的长链。
    return float(sum(float(atom.GetMass()) for atom in mol.GetAtoms()))  # 返回 amu；避免 polymer helper 忽略重复单元造成低估。


def graphite_repeats_for_xy(target_xy_nm: tuple[float, float]) -> tuple[int, int, tuple[float, float]]:  # 选择最小的石墨 nx/ny 覆盖目标 XY。
    probe = build_graphite(  # 用 2x2 周期基面 probe 推断单个石墨重复单元的 box 尺寸。
        nx=2,  # 1x1 周期石墨太小，2x2 是当前 builder 的最小安全 probe。
        ny=2,  # 同上。
        n_layers=3,  # 层数不影响 XY 单元长度。
        orientation="basal",  # 基面石墨。
        edge_cap="periodic",  # 周期边界；box_x = nx * a, box_y = ny * b。
        name="GRAPHITE_UNIT_PROBE",  # probe 不进入最终体系。
    )
    unit_x_nm = float(probe.box_nm[0]) / 2.0  # 单个 x 重复单元长度。
    unit_y_nm = float(probe.box_nm[1]) / 2.0  # 单个 y 重复单元长度。
    nx = max(2, int(math.ceil(float(target_xy_nm[0]) / unit_x_nm - 1.0e-6)))  # 覆盖目标 x 的最小 nx，容忍 GRO 五位小数舍入。
    ny = max(2, int(math.ceil(float(target_xy_nm[1]) / unit_y_nm - 1.0e-6)))  # 覆盖目标 y 的最小 ny，容忍 GRO 五位小数舍入。
    actual_xy_nm = (float(nx) * unit_x_nm, float(ny) * unit_y_nm)  # 石墨晶格兼容的实际 XY。
    return nx, ny, actual_xy_nm  # 返回 repeat 和对应实际 XY。


# ---------------- user inputs ----------------
restart_status = True  # True 断点续跑；False 会清空 07 的 work_dir 并重新构建/采样。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # CMC、碳酸酯、PF6- 和石墨使用 GAFF2_mod，和其它 eg08 保持一致。
ion_ff = MERZ()  # Li+/Na+ 使用 Merz 离子参数。

temp = 318.15  # 温度 K；正式生产建议先固定这个值，避免和结构问题混在一起。
mpi = 1  # thread-MPI rank 数；当前 GROMACS-2026 thread_mpi 单卡通常用 1。
omp = 14  # OpenMP 线程数；按远端节点 CPU 分配调整。
gpu = 1  # 1 使用 GPU 加速，0 强制 CPU。
gpu_id = 0  # GPU 编号；多卡节点按实际可用卡修改。
run_sampling = True  # True 跑 60 ns 生产；若只想检查构建，把这里改 False。
start_interdiffusion_at_final_nvt = True  # True 表示 final NVT 第一帧才定义为电解液/CMC 互扩散 t=0。
pre_equilibrate_cmcna_xy_slab = True  # True 先把 CMC-Na 做成 z 无周期 xy slab，再拼入界面。
pre_equilibrate_electrolyte_xy_slab = True  # True 先把电解液也做成同一 XY 的 z-open slab，避免 final stack fresh packing 缝隙。
cmc_slab_nominal_xy_nm = (5.00, 7.70)  # CMC slab 的名义横向尺寸；脚本会向上取整到石墨晶格兼容 XY。
xy_match_tolerance_nm = 0.02  # relaxed CMC/electrolyte slab 与最终石墨 XY 的允许差；超过说明不应直接拼接。
cmcna_initial_slab_density_g_cm3 = 0.05  # CMC-Na 初始插入密度；越低越容易放入长链，但 slab z 会更长。
cmcna_target_slab_density_g_cm3 = 1.50  # CMC-Na active slab 目标密度；定义为 CMC-Na 质量/(XY 面积*active z 厚度)。
cmcna_slab_eq_tmax_K = 450.0  # CMC-Na slab 退火最高温度；DP20 链需要更强的构象重排能力。
cmcna_slab_eq_ns = 0.50  # 到达目标 active density 后的 wall-confined NVT 松弛时长。
cmcna_slab_max_convergence_rounds = 8  # density/Rg 未收敛时最多追加多少轮 NVT。
cmcna_slab_extra_relax_ns = 0.50  # 每轮追加 NVT 的长度；长链体系可升到 1.0 ns。
cmcna_slab_density_tolerance_fraction = 0.08  # active density 尾段均值允许偏离目标的比例。
cmcna_slab_density_rel_std_max = 0.08  # active density 尾段相对波动上限。
electrolyte_initial_slab_density_g_cm3 = 0.85  # 电解液初始插入密度；略稀疏便于放入，后续 wall EQ21 压到接近 bulk。
electrolyte_target_slab_density_g_cm3 = 1.15  # 电解液 xy-slab 目标密度；接近 carbonate/LiPF6 bulk 液体口径。
electrolyte_slab_eq_tmax_K = 360.0  # 电解液 slab 退火最高温度；比 CMC 低，避免液体阶段过激扰动。
electrolyte_slab_eq_ns = 0.10  # 电解液最终 wall-confined NVT 松弛时长；生产制备可升到 0.2-0.5 ns。

# This script is meant for a real large-cell production test.  For a quick
# build-only check, set run_sampling=False above.
sample_ns = 60.0  # final NVT 采样时长 ns；final NVT 第一帧定义为互扩散 t=0。

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
time_series_analysis = True  # 打开时间序列 CSV/MP4，便于检查 60 ns 内结构是否压实和互扩散。
interface_time_series_sample_count = 10  # 全轨迹按十分之一时长取 10 帧/窗口。
interface_time_series_fps = 1.0  # MP4 慢速播放，太快不利于判断结构演化。
interface_time_series_rdf = True  # 输出全体系 cation RDF/CN 和 graphite-EDL RDF/CN；RDF 实线、CN 虚线且 CN 轴为 0-6。
interface_time_series_concentration = True  # 输出 z 浓度 profile 时间序列。
interface_time_series_angles = True  # 输出吸附角度分布时间序列。
interface_time_series_charge_potential = True  # 输出带电 graphite/EDL 电荷密度、积分电荷和电势 profile 的 CSV/PNG/MP4。
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

# CMC-first matched-cell composition.
cmc_dp = 20  # 每条 CMC 链聚合度；本例目标是 DP=20。
cmc_chain_count = 8  # CMC 链数；本例目标是 8 条。
reference_xy_area_nm2 = 37.60  # 06 号大扁盒子的参考面积；用于按 CMC slab 面积缩放电解液数量。
base_solvent_counts = (96, 72, 168)  # 参考面积下的 EC/EMC/DEC 数量；保持 4:3:7。
base_salt_pairs = 36  # 参考面积下的 LiPF6 离子对数。
charge_scale = 0.7  # CMC/Na/Li/PF6 电荷缩放；想跑全电荷时改 1.0，但需重新验证稳定性。

BASE_DIR = Path(__file__).resolve().parent  # 例子所在目录。
work_dir = BASE_DIR / "work_dir" / "07_cmcna_xy_slab_matched_graphite_electrolyte_cmcna_graphite"  # 本例输出根目录。


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

    pre_slab_nx, pre_slab_ny, graphite_compatible_xy_nm = graphite_repeats_for_xy(cmc_slab_nominal_xy_nm)  # 把名义 CMC XY 转成石墨晶格兼容 XY。
    print(f"requested_cmc_slab_xy_nm = {cmc_slab_nominal_xy_nm}")  # 打印用户请求的名义 CMC XY。
    print(f"graphite_compatible_cmc_slab_xy_nm = {graphite_compatible_xy_nm}")  # 打印实际用于 slab 的 XY。
    print(f"pre_slab_graphite_repeats = nx={pre_slab_nx}, ny={pre_slab_ny}")  # 打印匹配该 XY 所需的石墨 repeat。
    cmcna_prepared_slab_gro = None  # 默认没有 prepared slab；开启预平衡后改为 EQ21 xy-slab 的 final gro。
    if pre_equilibrate_cmcna_xy_slab:  # True 时先独立预平衡 CMC-Na slab。
        cmcna_slab_dir = case_dir.child("02_cmcna_xy_slab_eq21")  # CMC-Na slab 预平衡输出目录。
        cmcna_slab = prepare_cmcna_xy_bulk_slab(  # 新的 CMC-Na 专用入口会完成稀疏 AC、z-wall EQ21、active density/Rg 收敛检查。
            cmc_chain_mol=CMC,  # 已经完成端基和力场分配的 DP20 CMC 链。
            na_mol=Na,  # CMC 羧酸根的 Na+ counterion。
            chain_count=cmc_chain_count,  # CMC 链条数；这里是 8。
            dp=cmc_dp,  # 每条链的重复单元数；Na 数自动按 chain_count*dp 设置。
            xy_nm=graphite_compatible_xy_nm,  # 固定 XY footprint；后续石墨和电解液都按这个横向尺寸匹配。
            work_dir=cmcna_slab_dir,  # CMC-Na slab 全部输出写到该目录。
            temp=temp,  # wall-confined NVT 的目标温度。
            pressure_bar=1.0,  # 仅作为记录和可选 XY-NPT 口径；默认固定 XY，不做 z-NPT。
            mpi=mpi,  # thread-MPI rank。
            omp=omp,  # OpenMP 线程数。
            gpu=gpu,  # GPU 开关。
            gpu_id=gpu_id,  # GPU 编号。
            charge_scale=(charge_scale, charge_scale),  # CMC 与 Na+ 使用同一电荷缩放，保持局域配对口径。
            relaxation=CMCNAXYSlabRelaxationSpec(  # CMC-Na z-open bulk slab 的收敛制备配置。
                initial_density_g_cm3=cmcna_initial_slab_density_g_cm3,  # 稀疏 AC 初始密度；只用于提高插入成功率。
                target_density_g_cm3=cmcna_target_slab_density_g_cm3,  # 目标 active slab density，而不是总 box density。
                wall_padding_nm=0.40,  # z walls 与 active slab 两侧保留 padding，避免原子贴墙。
                cycles="auto",  # 自动根据初始 z 和目标 active z 生成逐轮压缩表。
                max_cycles=40,  # 极端长盒子最多压缩轮数。
                max_z_shrink_per_cycle=0.06,  # 每轮最多缩短 6%，比旧 10% 更稳。
                tmax_K=cmcna_slab_eq_tmax_K,  # 几何压缩退火最高温度。
                hot_nvt_ns=0.01,  # 每轮压缩后的热 NVT。
                cool_nvt_ns=0.01,  # 每轮压缩后回到目标温度的 NVT。
                final_relax_ns=cmcna_slab_eq_ns,  # 到达目标 active density 后的基础松弛时长。
                max_convergence_rounds=cmcna_slab_max_convergence_rounds,  # 未收敛时追加 NVT 的最大轮数。
                extra_relax_ns_per_round=cmcna_slab_extra_relax_ns,  # 每轮追加 NVT 长度。
                active_density_convergence=True,  # 必须检查 active density 尾段是否稳定。
                rg_convergence=True,  # 必须检查 CMC 链 Rg 是否进入平台。
                active_density_tolerance_fraction=cmcna_slab_density_tolerance_fraction,  # active density 均值容差。
                active_density_rel_std_max=cmcna_slab_density_rel_std_max,  # active density 波动容差。
                na_coo_contact_cutoff_nm=0.35,  # Na+/COO- 接触距离阈值。
                na_coo_contact_min_fraction=0.75,  # 大部分 Na+ 应保持在羧酸根附近。
            ),
            retry=30,  # 长链稀疏插入重试次数。
            retry_step=2000,  # 每轮 packing 试探步数。
            threshold_ang=2.0,  # 初始插入排斥阈值。
            large_system_mode="large",  # DP20 x 8 使用大体系 packing 策略。
            restart=restart_status,  # 允许断点复用已经构建好的 AC 和 EQ 阶段。
        )
        cmcna_prepared_slab_gro = cmcna_slab.prepared_slab_gro  # 这个 GRO 将作为 CMCNA layer 的真实初始坐标。
        print(f"cmcna_prepared_slab_gro = {cmcna_prepared_slab_gro}")  # 打印 slab 坐标路径，便于人工检查。
        print(f"cmcna_slab_ready_for_layer_stack = {cmcna_slab.ready_for_layer_stack}")  # 打印 density/Rg/Na-COO 收敛门的总体结果。
        print(f"cmcna_slab_convergence_json = {cmcna_slab.convergence_summary}")  # 打印收敛 JSON，便于后续复查。
    else:
        raise RuntimeError("Example 08-07 is defined around a prepared CMC-Na xy slab; keep pre_equilibrate_cmcna_xy_slab=True.")  # 防止误用为普通 packing 例子。
    cmc_slab_xy_nm = read_gro_xy_nm(Path(cmcna_prepared_slab_gro))  # 从 relaxed CMC slab GRO 读回真实 XY 盒长。
    graphite_nx, graphite_ny, matched_graphite_xy_nm = graphite_repeats_for_xy(cmc_slab_xy_nm)  # 按读回 XY 选择最终石墨 repeat。
    xy_delta_nm = (abs(float(matched_graphite_xy_nm[0]) - float(cmc_slab_xy_nm[0])), abs(float(matched_graphite_xy_nm[1]) - float(cmc_slab_xy_nm[1])))  # 计算石墨和 CMC slab 的 XY 差值。
    if xy_delta_nm[0] > xy_match_tolerance_nm or xy_delta_nm[1] > xy_match_tolerance_nm:  # 如果石墨晶格不能足够贴合 CMC slab。
        raise RuntimeError(  # 直接停止，避免把 XY 周期 slab 放进不兼容的 stack 里。
            f"CMC slab XY {cmc_slab_xy_nm} does not match graphite-compatible XY {matched_graphite_xy_nm}; "
            f"delta={xy_delta_nm}, tolerance={xy_match_tolerance_nm} nm."
        )
    graphite_xy_nm = matched_graphite_xy_nm  # 后续电解液数量和 stack master XY 都以 CMC slab 读回值为准。
    electrolyte_area_scale = (float(graphite_xy_nm[0]) * float(graphite_xy_nm[1])) / float(reference_xy_area_nm2)  # 根据 CMC-first XY 面积缩放电解液数量。
    solvent_counts = tuple(max(1, int(round(v * electrolyte_area_scale))) for v in base_solvent_counts)  # EC/EMC/DEC 数量随面积线性缩放。
    salt_pairs = max(1, int(round(base_salt_pairs * electrolyte_area_scale)))  # LiPF6 数量随面积线性缩放，保持盐浓度口径。
    print(f"cmc_slab_xy_nm = {cmc_slab_xy_nm}")  # 打印 CMC slab 的真实 XY。
    print(f"matched_graphite_repeats = nx={graphite_nx}, ny={graphite_ny}, xy_nm={matched_graphite_xy_nm}")  # 打印最终石墨匹配结果。
    print(f"scaled_electrolyte_counts = {solvent_counts}, salt_pairs={salt_pairs}")  # 打印按面积缩放后的电解液组成。
    electrolyte_prepared_slab_gro = None  # 默认不复用电解液 slab；开启预平衡后改为 EQ21 xy-slab final gro。
    electrolyte_species = (EC, EMC, DEC, Li, PF6)  # 电解液 species 顺序必须和 counts/charge_scale 对齐。
    electrolyte_counts = (*solvent_counts, salt_pairs, salt_pairs)  # EC/EMC/DEC/Li/PF6 数量。
    electrolyte_charge_scale = (1.0, 1.0, 1.0, charge_scale, charge_scale)  # 溶剂全电荷，Li/PF6 与 CMC-Na 使用同一缩放。
    if pre_equilibrate_electrolyte_xy_slab:  # True 时先独立预平衡电解液 slab。
        electrolyte_slab_dir = case_dir.child("03_electrolyte_xy_slab_eq21")  # 电解液 slab 预平衡输出目录。
        electrolyte_total_mass_amu = sum(  # 估算电解液总质量，用于从初始密度反推 z 长度。
            rdkit_mol_mass_amu(mol) * int(count) for mol, count in zip(electrolyte_species, electrolyte_counts)
        )
        electrolyte_initial_volume_nm3 = (electrolyte_total_mass_amu / 6.02214076e23) / electrolyte_initial_slab_density_g_cm3 * 1.0e21  # 初始 0.85 g/cm3 对应体积。
        electrolyte_initial_z_nm = electrolyte_initial_volume_nm3 / (float(graphite_xy_nm[0]) * float(graphite_xy_nm[1]))  # 固定 XY 后所需初始 z。
        electrolyte_initial_cell = make_orthorhombic_pack_cell((float(graphite_xy_nm[0]), float(graphite_xy_nm[1]), electrolyte_initial_z_nm))  # 构造电解液 z-open 初始盒子。
        electrolyte_ac = poly.amorphous_cell(  # 在固定 XY、略稀疏 z 中放入电解液。
            list(electrolyte_species),  # EC/EMC/DEC/Li/PF6。
            list(electrolyte_counts),  # 与 species 对齐的数量。
            cell=electrolyte_initial_cell,  # 固定 XY，z 由初始密度估算。
            density=None,  # 已显式给出 cell，不让 amorphous_cell 再改盒子。
            retry=30,  # 液体组分较多时提高插入重试次数。
            retry_step=2000,  # 每轮 packing 试探步数。
            threshold=2.0,  # 初始排斥阈值；太大更难插入，太小坏接触更多。
            charge_scale=electrolyte_charge_scale,  # Li/PF6 缩放；溶剂保持全电荷。
            polyelectrolyte_mode=False,  # 普通电解液不是聚电解质。
            large_system_mode="large",  # 大 XY slab 使用大体系 packing 策略。
            work_dir=electrolyte_slab_dir.child("00_ac"),  # 初始电解液 AC 输出目录。
            restart=restart_status,  # 允许断点复用已经构建好的 AC。
        )
        electrolyte_slab_eq = eq.EQ21step(electrolyte_ac, work_dir=electrolyte_slab_dir)  # 创建电解液 slab EQ21 任务。
        electrolyte_slab_eq.exec(  # 运行 wall-confined pbc=xy 电解液压实/松弛。
            temp=temp,  # 目标温度。
            press=1.0,  # 仅作为 wall/NVT 记录参数；默认固定 XY，不做 z-NPT。
            mpi=mpi,  # thread-MPI rank。
            omp=omp,  # OpenMP 线程数。
            gpu=gpu,  # GPU 开关。
            gpu_id=gpu_id,  # GPU 编号。
            sim_time=electrolyte_slab_eq_ns,  # 兼容 EQ21 参数；xy-slab 分支主要使用 final_relax_ns。
            eq21_tmax=electrolyte_slab_eq_tmax_K,  # 电解液 slab 退火最高温度。
            eq21_dt_ps=0.001,  # 1 fs 步长，优先保证 z-open wall 初期稳定。
            periodicity="xy",  # 关键：z 方向无周期，使用 z walls。
            xy_slab=XYSlabEquilibrationSpec(  # 电解液 slab 专用配置。
                target_density_g_cm3=electrolyte_target_slab_density_g_cm3,  # 目标接近 bulk 液体密度。
                cycles="auto",  # 自动按每轮最大压缩比例生成 cycles。
                max_cycles=20,  # 电解液通常比 CMC 更容易压实，限制最大轮数。
                max_z_shrink_per_cycle=0.08,  # 每轮最多缩短 8% z，避免液体/离子瞬间过密。
                wall_padding_nm=0.40,  # slab 两侧保留 wall padding。
                xy_area_mode="fixed",  # 固定 XY footprint，确保可直接拼接到 CMC/石墨。
                hot_nvt_ns=0.01,  # 每轮热 NVT 时间。
                cool_nvt_ns=0.01,  # 每轮回到目标温度的 NVT 时间。
                final_relax_ns=electrolyte_slab_eq_ns,  # 到目标 z 后的最终 wall-confined NVT。
            ),
            restart=restart_status,  # 允许断点续跑。
        )
        electrolyte_prepared_slab_gro = electrolyte_slab_eq.final_gro()  # 这个 GRO 将作为 ELECTROLYTE layer 的真实初始坐标。
        electrolyte_slab_xy_nm = read_gro_xy_nm(Path(electrolyte_prepared_slab_gro))  # 从 relaxed electrolyte slab GRO 读回真实 XY。
        electrolyte_xy_delta_nm = (abs(float(electrolyte_slab_xy_nm[0]) - float(graphite_xy_nm[0])), abs(float(electrolyte_slab_xy_nm[1]) - float(graphite_xy_nm[1])))  # 检查电解液 slab 与最终 XY 是否一致。
        if electrolyte_xy_delta_nm[0] > xy_match_tolerance_nm or electrolyte_xy_delta_nm[1] > xy_match_tolerance_nm:  # 电解液 slab 不应在拼接时被横向重缩放。
            raise RuntimeError(  # 直接停止，避免横向缝隙或隐式应变进入 final stack。
                f"Electrolyte slab XY {electrolyte_slab_xy_nm} does not match stack XY {graphite_xy_nm}; "
                f"delta={electrolyte_xy_delta_nm}, tolerance={xy_match_tolerance_nm} nm."
            )
        print(f"electrolyte_prepared_slab_gro = {electrolyte_prepared_slab_gro}")  # 打印电解液 slab 坐标路径，便于人工检查。
    else:
        raise RuntimeError("Example 08-07 requires pre_equilibrate_electrolyte_xy_slab=True to avoid fresh-packing lateral voids.")  # 防止误回到旧拼接方案。
    graphite_bottom = GraphiteLayerSpec(  # 最终下石墨层按 CMC slab XY 自适应。
        name="GRAPHITE_BOTTOM",  # 下石墨层名；固定电荷区域按这个名字选层。
        nx=graphite_nx,  # 从 CMC slab XY 读回后计算的 x repeat。
        ny=graphite_ny,  # 从 CMC slab XY 读回后计算的 y repeat。
        n_layers=3,  # 石墨层数；若要更硬/更厚电极可增加。
        orientation="basal",  # 基面石墨。
        periodic_xy=True,  # XY 周期成键，防止基面边缘断裂。
    )
    graphite_top = GraphiteLayerSpec(  # 最终上石墨层也按 CMC slab XY 自适应。
        name="GRAPHITE_TOP",  # 上石墨层名；与下石墨组成 sandwich。
        nx=graphite_nx,  # 与下石墨相同。
        ny=graphite_ny,  # 与下石墨相同。
        n_layers=3,  # 与下石墨相同。
        orientation="basal",  # 基面石墨。
        periodic_xy=True,  # XY 周期成键。
    )
    electrolyte = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="ELECTROLYTE",  # 电解液层名。
        species=electrolyte_species,  # species 顺序必须对应 counts 和 charge_scale。
        counts=electrolyte_counts,  # EC/EMC/DEC/Li/PF6 数量。
        thickness_nm=2.2,  # 初始电解液目标厚度；真实厚度会由 z-NPT 调整。
        density_target_g_cm3=electrolyte_target_slab_density_g_cm3,  # prepared slab 已接近 bulk 液体密度。
        layer_kind="electrolyte",  # 电解液语义标签。
        charge_scale=electrolyte_charge_scale,  # 溶剂全电荷，Li/PF6 缩放。
        large_system_mode="large",  # 强制使用大体系 packing 策略，减少大盒子随机插入失败。
        prepared_slab_gro=electrolyte_prepared_slab_gro,  # 关键：复用已预平衡电解液 slab，不在 final stack fresh packing。
    )
    cmcna = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="CMCNA",  # CMC-Na 层名，后处理识别 CMC-rich/core 区域。
        species=(CMC, Na),  # CMC 链和 Na+ counterion。
        counts=(cmc_chain_count, cmc_chain_count * cmc_dp),  # 8 条 DP20 链对应 160 个 Na+。
        thickness_nm=2.6,  # prepared_slab_gro 存在时主要作为 manifest 目标厚度记录，不重新按此厚度 packing。
        # The real CMC coordinates come from the wall-confined xy slab above.
        # This density target is retained as provenance and density diagnostic
        # context; it is not used to repack the final stack.
        density_target_g_cm3=cmcna_target_slab_density_g_cm3,  # 与 CMC active slab 目标密度保持一致，便于 manifest/summary 解读。
        layer_kind="cmcna",  # 启用 CMCNA 专用分组、Na+/COO- 和密度诊断。
        charge_scale=(charge_scale, charge_scale),  # CMC 与 Na+ 使用同一缩放，保持局部配对口径。
        polyelectrolyte_mode=True,  # 按聚电解质体系处理 CMC-Na。
        large_system_mode="large",  # 大体系 packing 策略。
        counterion_contact_mode="carboxylate",  # 构建后把 Na+ 放在 COO- 附近，避免初期跑进电解液。
        prepared_slab_gro=cmcna_prepared_slab_gro,  # 若开启 xy-slab 预平衡，则直接复用该 slab 坐标。
    )
    stack = LayerStackSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        layers=(graphite_bottom, electrolyte, cmcna, graphite_top),  # 大扁盒子 sandwich 层顺序。
        order="bottom_to_top",  # 按 z 从下到上解释 layers。
        pbc_mode="xyz",  # 闭合三维周期；不是显式真空体系。
        name=f"cmcna_xy_slab_matched_graphite_stack_{case_name}",  # 系统名包含 CMC-first 匹配策略和电荷状态。
        default_gap_nm=0.35,  # 层间初始间隙，避免 fresh overlap。
        molecular_packing_expand="z",  # 固定 CMC slab 读回的 XY；若分子太多则扩展 z 而非改横向面积。
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
    interdiffusion_start = InterdiffusionStartSpec(  # 定义“预松弛不算互扩散，final NVT 才开始互扩散”的策略。
        enabled=start_interdiffusion_at_final_nvt,  # True 打开预释放 z-gate；False 回到平衡界面直接松弛模式。
        strategy="soft_wall_release",  # 当前实现用 GROMACS z-only position restraint 做软 gate。
        hold_interphase=True,  # pre-release 阶段限制 ELECTROLYTE/CMCNA 在 z 方向明显互穿。
        phase_pre_equilibrate=True,  # 在 summary 中记录相预平衡意图；当前执行路径为 assembled pre-release gate。
        release_at_final_nvt=True,  # final NVT 移除 phase gate，第一帧作为 diffusion t=0。
        phase_gate_k_kj_mol_nm2=1500.0,  # z-gate 力常数；界面过早混合可升到 2500，消重叠困难可降到 500。
        phase_gate_layers=("ELECTROLYTE", "CMCNA"),  # 只 gate 两个软相，不 gate 石墨。
        diffusion_t0_stage="final_nvt",  # 后处理口径：正式互扩散从 final NVT 开始。
        diffusion_t0_ps=0.0,  # final NVT 第一帧定义为 0 ps。
    )
    graphite_restraint = GraphiteRestraintSpec(  # 定义石墨 z 方向平整 restraint，避免 NVT/NPT 中电极褶皱。
        enabled="auto",  # auto 表示只要体系中有 GraphiteLayerSpec 就启用。
        mode="z_position",  # 只约束 z；xy 内可以正常热运动和整体平移。
        k_pre_kj_mol_nm2=5000.0,  # pre-release / anneal / z-NPT 阶段较强，帮助电极保持平整。
        k_final_kj_mol_nm2=1000.0,  # final NVT 阶段较弱，减少对界面动力学的扰动。
        fcx_kj_mol_nm2=0.0,  # x 不约束。
        fcy_kj_mol_nm2=0.0,  # y 不约束。
    )

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
            z_compressibility_bar_inv=4.5e-6,  # final z-NPT 使用小有效压缩率，避免 slab 拼接体系出现大幅 z 缩放。
            z_npt_tau_p_ps=20.0,  # final z-NPT 使用慢 barostat，和循环压缩退火阶段一致。
            graphite_restraint=graphite_restraint,  # 全流程保留石墨 z-only 平整 restraint。
            interdiffusion_start=interdiffusion_start,  # pre-release 阶段不统计互扩散，final NVT 才释放。
            compression_anneal=ZCompressionAnnealSpec(  # prepared CMC/electrolyte slab 已经分别致密预平衡，默认不再强循环压缩。
                enabled=False,  # 若诊断发现明显 vacuum-like span，再手动改 True 做温和循环压缩。
                cycles=4,  # 手动开启时使用少量小步压缩即可。
                tmax_K=360.0,  # 手动开启时只做温和退火，不再用高温强压实。
                pmax_bar=1500.0,  # 保留温和高压上限；比旧强压缩路线更保守。
                max_z_shrink_per_cycle=0.02,  # 手动开启时每轮最多压 2%，避免破坏两个 prepared slab。
                hot_nvt_ns=0.01,  # 手动开启时每轮热 NVT 时间。
                compression_npt_ns=0.04,  # 手动开启时每轮高压 z-NPT 时间。
                cool_nvt_ns=0.02,  # 手动开启时每轮冷却回正常温度的 NVT 时间。
                compression_tau_p_ps=20.0,  # 高压退火使用慢 barostat，使石墨/软相随盒子平稳移动。
                compression_z_compressibility_bar_inv=4.5e-6,  # 高压阶段用较小有效 z 压缩率，降低盒子突变风险。
                geometry_clash_check=True,  # 几何压缩后先筛查原子重叠，严重重叠时自动减半压缩比例。
                geometry_clash_cutoff_nm=0.10,  # 记录 0.10 nm 内的可疑非同残基近接触。
                severe_geometry_clash_cutoff_nm=0.065,  # 低于该距离视为高风险，避免直接进入 minimization。
                max_geometry_clash_retries=3,  # 最多把几何压缩比例连续减半 3 次；仍不安全则停止后续压缩。
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
            time_series_rdf=interface_time_series_rdf,  # 控制全体系 RDF/CN 与 graphite-EDL RDF/CN 时间序列。
            time_series_concentration=interface_time_series_concentration,  # 是否输出浓度 profile 动画。
            time_series_angles=interface_time_series_angles,  # 是否输出角度分布动画。
            time_series_charge_potential=interface_time_series_charge_potential,  # 是否输出 graphite/EDL 电荷-电势动画。
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
        charge_potential = (time_series.get("outputs") or {}).get("charge_potential") or {}  # 提取带电 graphite/EDL 电荷-电势输出。
        print(f"interface_phase_order_ok = {health.get('phase_order_ok')}")  # 打印关键路径或状态，便于人工检查。
        print(f"membrane_permeation_available = {membrane.get('available')}")  # 确认膜渗透统计是否识别到 CMCNA/polymer 区域。
        print(f"interface_outputs = {summary.get('outputs', {}).get('interface_profile_summary_json')}")  # 打印关键路径或状态，便于人工检查。
        print(f"interface_time_series = {time_series.get('outputs', {})}")  # 打印关键路径或状态，便于人工检查。
        print(f"edl_rdf_cn_available = {edl_rdf.get('available')}")  # True 表示 graphite-EDL RDF/CN CSV/MP4 正常生成。
        print(f"edl_rdf_cn_outputs = {edl_rdf.get('curves_csv') or edl_rdf.get('reason')}")  # 打印曲线 CSV 路径或失败原因。
        print(f"charge_potential_available = {charge_potential.get('available')}")  # True 表示电荷-电势 CSV/MP4 正常生成。
        print(f"charge_potential_outputs = {charge_potential.get('csv') or charge_potential.get('reason')}")  # 打印电荷-电势 CSV 路径或失败原因。

    clean_md_trajectory_files(case_dir, enabled=clean_trajectories_after_analysis)  # 按配置清理 MD 轨迹文件。
