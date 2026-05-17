from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

# Example 02: Polymer electrolyte workflow in one script (SMILES -> RESP -> polymer -> cell -> EQ21 -> analysis)
#
# Restart logic
#   restart_status=True  : resume/skip finished steps based on files in work_dir
#   restart_status=False : force re-run (keep work_dir, but steps will overwrite their own outputs)

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import utils, poly, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2_mod import GAFF2_mod  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.merz import MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim import qm  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.cleanup import clean_md_trajectory_files  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.preset import eq  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.moldb import MolDB  # 导入本例需要的库或 yadonpy 接口。


# ---------------- user inputs ----------------
restart_status = False  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # default: GAFF2_mod
cation_ff = MERZ()  # 选择阳离子的力场/参数来源。

# ---- two monomers (both must have two connection points '*...*') ----
smiles_A = r"*CCO*"  # updated per request
smiles_B = r"*COC*"  # 设置中间变量或可调参数，供后续工作流使用。

# ---- solvents / ions (no '*') ----
solvent_smiles_A = "CCOC(=O)OC"  # EMC (example)
solvent_smiles_B = "O=C1OCCO1"  # EC (example)

cation_smiles_A = "[Li+]"  # cation_A (MERZ)
anion_smiles_A = "F[P-](F)(F)(F)(F)F"  # PF6- (GAFF2-family; default uses GAFF2_mod)

# Termination unit (one '*').
# - Hydrogen termination: "[H][*]" or "[*][H]"
# - Other terminations: "*C", "*O", ...
#   The pure hydrogen terminator is treated as a linker placeholder internally,
#   so `qm.assign_charges(..., charge="RESP")` now uses a zero-charge shortcut
#   instead of sending the 2-atom H/H fragment into QM.
ter_smiles = "[H][*]"  # 设置中间变量或可调参数，供后续工作流使用。

# composition in the copolymer chain (mole fraction)
ratio = [0.7, 0.3]  # A:B
reac_ratio: list[float] = []  # optional Mayo–Lewis biasing, e.g. [rA, rB]

# MD settings
temp = 300.0  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
press = 1.0  # 设置压力 bar；用于 NPT/EQ 阶段。
mpi = 1  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = 16  # 设置每个 rank 的 OpenMP 线程数。

# GPU semantics:
#   gpu=1 enables GPU, gpu=0 disables GPU.
#   gpu_id selects which GPU GROMACS uses when GPU is enabled.
gpu = 1  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = 3  # 选择 GPU 设备编号，多卡节点可修改。

sim_time_ns = 5.0  # 设置生产模拟时长，单位 ns。
clean_trajectories_after_analysis = False  # 控制后处理后是否清理大轨迹文件。

# Packing
density_g_cm3 = 0.05  # 设置初始 packing 密度，主要影响构建难度和初始盒子大小。
counts = [4, 20, 20, 20, 20]  # [polymer, solventA, solventB, cation_A, anion_A]
charge_scale = [1.0, 1.0, 1.0, 0.8, 0.8]  # aligned with the species list below

# QM/RESP settings
omp_psi4 = 64  # 设置 Psi4/OpenMP 核数。
mem_mb = 20000  # 设置量子化学内存 MB。
mem = mem_mb  # 把内存设置传给后续 QM 调用。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = BASE_DIR / "work_dir"  # 设置本例输出目录。


if __name__ == '__main__':  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # idempotent

    work_dir = workdir(work_dir, restart=restart_status)  # 设置本例输出目录。
    poly_rw_dir = work_dir.child("copoly_rw")  # 设置中间变量或可调参数，供后续工作流使用。
    poly_term_dir = work_dir.child("copoly_term")  # 设置中间变量或可调参数，供后续工作流使用。
    ac_build_dir = work_dir.child("00_build_cell")  # 设置中间变量或可调参数，供后续工作流使用。

    # Optional: if you want to generate copy-pastable MolDB artifacts from
    # molecules computed in this script (without touching ~/.yadonpy/moldb),
    # use db.mol_gen(mol, work_dir=..., add_to_moldb=False).
    db = MolDB()  # 设置中间变量或可调参数，供后续工作流使用。

    # polymer####################################################################################################
    # --- build monomers ---
    monomer_A = utils.mol_from_smiles(smiles_A)  # 从 SMILES 直接构造 RDKit 分子。
    monomer_B = utils.mol_from_smiles(smiles_B)  # 从 SMILES 直接构造 RDKit 分子。
    # No explicit naming: yadonpy will infer names from variable names when needed.

    # --- conformation search & RESP for each monomer (separate log_name!) ---
    # NOTE (defaults): You can *explicitly* specify the default QM levels like this:
    #   - neutral/cation: OPT wb97m/def2-SVP ; RESP(ESP) wb97m/def2-TZVP
    #   - anion        : OPT wb97m/def2-SVPD; RESP(ESP) wb97m/def2-TZVPD
    # Example (neutral):
    #   monomer_A, _ = qm.conformation_search(monomer_A, ff=ff, opt_method='wb97m', opt_basis='def2-SVP', ...)
    #   qm.assign_charges(monomer_A, charge='RESP', opt=False, charge_method='wb97m', charge_basis='def2-TZVP', ...)
    monomer_A, energy = qm.conformation_search(  # 执行构象搜索/几何优化，为 RESP 做准备。
        monomer_A, ff=ff, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        psi4_omp=omp_psi4, mpi=mpi, omp=omp, memory=mem, log_name=None  # 设置中间变量或可调参数，供后续工作流使用。
    )
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        monomer_A, charge='RESP', opt=False, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        omp=omp_psi4, memory=mem, log_name=None  # 设置每个 rank 的 OpenMP 线程数。
    )

    monomer_B, energy = qm.conformation_search(  # 执行构象搜索/几何优化，为 RESP 做准备。
        monomer_B, ff=ff, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        psi4_omp=omp_psi4, mpi=mpi, omp=omp, memory=mem, log_name=None  # 设置中间变量或可调参数，供后续工作流使用。
    )
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        monomer_B, charge='RESP', opt=False, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        omp=omp_psi4, memory=mem, log_name=None  # 设置每个 rank 的 OpenMP 线程数。
    )

    # --- termination unit RESP (same as sample) ---
    ter1 = utils.mol_from_smiles(ter_smiles)  # 从 SMILES 直接构造 RDKit 分子。
    # (name inferred later)
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        ter1, charge='RESP', opt=True, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        omp=omp_psi4, memory=mem, log_name=None  # 设置每个 rank 的 OpenMP 线程数。
    )

    # --- restart-friendly random copolymerization + termination (manual API style) ---
    dp = max(1, int(poly.calc_n_from_num_atoms([monomer_A, monomer_B], 1000, ratio=ratio, terminal1=ter1)))  # 设置或计算聚合度。
    copoly = poly.random_copolymerize_rw(  # 用随机游走生成聚合物链。
        [monomer_A, monomer_B],
        dp,
        ratio=ratio,  # 设置共聚组成比例。
        reac_ratio=reac_ratio,  # 设置 Mayo-Lewis 反应比；空列表表示不加反应偏置。
        tacticity='atactic',  # 设置聚合物立构。
        name='copoly',  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=poly_rw_dir,  # 设置本例输出目录。
    )
    copoly = poly.terminate_rw(copoly, ter1, name='copoly', work_dir=poly_term_dir)  # 给聚合物链加端基。
    copoly = ff.ff_assign(copoly)  # 分配力场参数并写入分子属性。
    if not copoly:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError('Can not assign force field parameters for copoly.')  # 关键步骤失败时立即报错，避免继续生成错误结果。
    # ############################################################################################################

    # solvent#####################################################################################################
    solvent_A = utils.mol_from_smiles(solvent_smiles_A)  # 从 SMILES 直接构造 RDKit 分子。
    solvent_B = utils.mol_from_smiles(solvent_smiles_B)  # 从 SMILES 直接构造 RDKit 分子。
    # (names inferred later)

    solvent_A, energy = qm.conformation_search(  # 执行构象搜索/几何优化，为 RESP 做准备。
        solvent_A, ff=ff, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        psi4_omp=omp_psi4, mpi=mpi, omp=omp,  # 设置中间变量或可调参数，供后续工作流使用。
        memory=mem, log_name=None  # 设置中间变量或可调参数，供后续工作流使用。
    )
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        solvent_A, charge='RESP', opt=False, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        omp=omp_psi4, memory=mem, log_name=None  # 设置每个 rank 的 OpenMP 线程数。
    )
    # Generate a standalone MolDB snippet under work_dir (copy-paste later if desired)
    db.mol_gen(solvent_A, work_dir=work_dir, add_to_moldb=False)  # 设置中间变量或可调参数，供后续工作流使用。
    solvent_A = ff.ff_assign(solvent_A)  # 分配力场参数并写入分子属性。
    if not solvent_A:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError('Can not assign force field parameters for solvent_A.')  # 关键步骤失败时立即报错，避免继续生成错误结果。

    solvent_B, energy = qm.conformation_search(  # 执行构象搜索/几何优化，为 RESP 做准备。
        solvent_B, ff=ff, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        psi4_omp=omp_psi4, mpi=mpi, omp=omp,  # 设置中间变量或可调参数，供后续工作流使用。
        memory=mem, log_name=None  # 设置中间变量或可调参数，供后续工作流使用。
    )
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        solvent_B, charge='RESP', opt=False, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        omp=omp_psi4, memory=mem, log_name=None  # 设置每个 rank 的 OpenMP 线程数。
    )
    db.mol_gen(solvent_B, work_dir=work_dir, add_to_moldb=False)  # 设置中间变量或可调参数，供后续工作流使用。
    solvent_B = ff.ff_assign(solvent_B)  # 分配力场参数并写入分子属性。
    if not solvent_B:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError('Can not assign force field parameters for solvent_B.')  # 关键步骤失败时立即报错，避免继续生成错误结果。
    # ############################################################################################################

    # cation_A######################################################################################################
    cation_A = cation_ff.mol(cation_smiles_A)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    # (name inferred later)
    cation_A = cation_ff.ff_assign(cation_A)  # 分配力场参数并写入分子属性。
    if not cation_A:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError('Can not assign MERZ parameters to cation_A')  # 关键步骤失败时立即报错，避免继续生成错误结果。
    # ############################################################################################################

    # anion_A#######################################################################################################
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        anion_A = ff.mol(anion_smiles_A, charge='RESP', require_ready=True, prefer_db=True)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
        anion_A = ff.ff_assign(anion_A, bonded='DRIH')  # 分配力场参数并写入分子属性。
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
            "PF6 is expected to be precomputed in MolDB for this example. "
            "Please build it first with examples/01_Li_salt/run_pf6_to_moldb.py."
        ) from exc
    if not anion_A:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError('Can not assign force field parameters for MolDB-backed PF6.')  # 关键步骤失败时立即报错，避免继续生成错误结果。
    # ############################################################################################################

    # build amorphous cell########################################################################################
    ac = poly.amorphous_cell(  # 构建无定形混合体系初始盒子。
        [copoly, solvent_A, solvent_B, cation_A, anion_A],
        counts,
        charge_scale=charge_scale,  # 设置电荷缩放系数；1.0 表示全电荷模型。
        density=density_g_cm3,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=ac_build_dir,  # 设置本例输出目录。
    )
    # ############################################################################################################

    # EQ21 + check equilibrium####################################################################################
    eqmd = eq.EQ21step(ac, work_dir=work_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。

    analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    prop_data = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    # Additional equilibration MD
    for i in range(4):  # 遍历当前工作流中的一组对象或任务。
        if result:  # 根据当前状态决定是否进入该分支。
            break
        eqmd = eq.Additional(ac, work_dir=work_dir)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
        analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        prop_data = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    if not result:  # 根据当前状态决定是否进入该分支。
        print('[ERROR: Did not reach an equilibrium state.]')  # 打印关键路径或状态，便于人工检查。
        raise SystemExit(1)  # 关键步骤失败时立即报错，避免继续生成错误结果。

    # Production NPT (sim_time_ns) + analysis#####################################################################
    npt = eq.NPT(ac, work_dir=work_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = npt.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=sim_time_ns)  # 设置中间变量或可调参数，供后续工作流使用。

    center_molecule = cation_A  # 设置中间变量或可调参数，供后续工作流使用。
    analy = npt.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    prop_data = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。

    rdf = analy.rdf(center_mol=center_molecule)  # 设置中间变量或可调参数，供后续工作流使用。
    msd = analy.msd()  # 设置中间变量或可调参数，供后续工作流使用。
    sigma = analy.sigma(msd=msd, temp_k=temp)  # 设置中间变量或可调参数，供后续工作流使用。
    # Number density distribution (all moltypes by default)
    density_distributionr = analy.den_dis()  # 设置中间变量或可调参数，供后续工作流使用。

    if clean_trajectories_after_analysis:  # 根据当前状态决定是否进入该分支。
        cleanup = clean_md_trajectory_files(work_dir, enabled=True)  # 按配置清理 MD 轨迹文件。
        print(f"[CLEAN] Removed {len(cleanup.removed_files)} trajectory files ({cleanup.removed_bytes} bytes).")  # 打印关键路径或状态，便于人工检查。
