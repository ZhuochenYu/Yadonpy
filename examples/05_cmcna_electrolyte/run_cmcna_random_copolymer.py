from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

# Example 05: CMC-Na random copolymer + 1M LiPF6 in EC/EMC/DEC (1:1:1 mass ratio)
#
# Targets:
#   - Polymer: CMC (random copolymer from 4 glucose-based monomers), Mw ~ 10000 g/mol, 4 chains
#   - Solvent: EC / EMC / DEC with equal mass
#   - Salt: 1 M LiPF6 (min 20 ion pairs)
#   - Counter-ion: Na+ to neutralize polymer formal charge (CMC-Na)
#   - Charge scaling: polymer and all ions scaled by 0.8
#   - Center molecule for RDF: Li+

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import molecular_weight, utils, poly, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff import GAFF2_mod, MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim import qm  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.preset import eq  # 导入本例需要的库或 yadonpy 接口。


# ---------------- user inputs ----------------
restart_status = False  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # or GAFF2() for classic GAFF2 (different parameter DB)
ion_ff = MERZ()  # 选择单原子离子参数来源。

# ---- CMC monomers (two connection points '*...*') ----
glucose_smiles   = "*OC1OC(CO)C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。

# feed ratio (integer)
feed_ratio = [12, 26, 27, 35]  # 设置中间变量或可调参数，供后续工作流使用。
feed_prob = poly.ratio_to_prob(feed_ratio)  # 设置中间变量或可调参数，供后续工作流使用。

# target polymer molecular weight (g/mol)
target_mw = 10000.0  # 设置中间变量或可调参数，供后续工作流使用。

# termination unit (one '*')
ter_smiles = "[H][*]"  # 设置中间变量或可调参数，供后续工作流使用。

# ---- Solvents ----
EC_smiles  = "O=C1OCCO1"  # 设置中间变量或可调参数，供后续工作流使用。
EMC_smiles = "CCOC(=O)OC"  # 设置中间变量或可调参数，供后续工作流使用。
DEC_smiles = "CCOC(=O)OCC"  # 设置中间变量或可调参数，供后续工作流使用。

# ---- Salt / ions ----
Li_smiles  = "[Li+]"                 # MERZ
PF6_smiles = "F[P-](F)(F)(F)(F)F"    # GAFF2_mod
Na_smiles  = "[Na+]"                 # MERZ (counter-ion for CMC-)

# ---- Formulation ----
n_CMC = 4  # 设置中间变量或可调参数，供后续工作流使用。
# Solvent feed: total solvent mass = 6 wt% of polymer feed mass
solvent_wt_over_polymer = 0.06  # 设置中间变量或可调参数，供后续工作流使用。
salt_molarity_M = 1.0      # LiPF6 concentration in mol/L
min_salt_pairs = 20  # 设置中间变量或可调参数，供后续工作流使用。

# packing density (g/cm^3) used to estimate volume for 1M salt count
density_target_g_cm3 = 1.2  # 设置初始层 packing 目标密度，不等价于最终平衡密度。
density_pack_g_cm3 = 0.05  # low packing density to ensure polymer fits in initial box

# charge scaling aligned with species list: [CMC, EC, EMC, DEC, Li, PF6, Na]
charge_scale = [0.8, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8]  # 设置电荷缩放系数；1.0 表示全电荷模型。

# MD settings
temp = 300.0  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
press = 1.0  # 设置压力 bar；用于 NPT/EQ 阶段。
mpi = 1  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = 16  # 设置每个 rank 的 OpenMP 线程数。
gpu = 1  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = 0  # 选择 GPU 设备编号，多卡节点可修改。

# QM settings
omp_psi4 = 32  # 设置 Psi4/OpenMP 核数。
mem_mb = 20000  # 设置量子化学内存 MB。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = BASE_DIR / "work_dir"  # 设置本例输出目录。


def mol_formal_charge(mol) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    """Sum RDKit atom formal charges."""

    q = 0  # 设置中间变量或可调参数，供后续工作流使用。
    for a in mol.GetAtoms():  # 遍历当前工作流中的一组对象或任务。
        q += int(a.GetFormalCharge())  # 设置中间变量或可调参数，供后续工作流使用。
    return int(q)  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    work_dir = workdir(work_dir, restart=restart_status)  # 设置本例输出目录。
    cmc_rw_dir = work_dir.child("CMC_rw")  # 设置中间变量或可调参数，供后续工作流使用。
    cmc_term_dir = work_dir.child("CMC_term")  # 设置中间变量或可调参数，供后续工作流使用。
    ac_build_dir = work_dir.child("00_build_cell")  # 设置中间变量或可调参数，供后续工作流使用。

    # ---------------- build monomers ----------------
    glucose   = utils.mol_from_smiles(glucose_smiles)  # 从 SMILES 直接构造 RDKit 分子。
    glucose_2 = utils.mol_from_smiles(glucose_2_smiles)  # 从 SMILES 直接构造 RDKit 分子。
    glucose_3 = utils.mol_from_smiles(glucose_3_smiles)  # 从 SMILES 直接构造 RDKit 分子。
    glucose_6 = utils.mol_from_smiles(glucose_6_smiles)  # 从 SMILES 直接构造 RDKit 分子。

    # Conformation search + RESP for monomers (anionic monomers included)
    # IMPORTANT:
    #   qm.conformation_search historically returned a *new* RDKit Mol instance.
    #   If we don't rebind the result back, later polymerization would still
    #   see the old (uncharged) objects. Keep the monomer list updated.
    monomers = [glucose, glucose_2, glucose_3, glucose_6]  # 设置中间变量或可调参数，供后续工作流使用。
    for i, mon in enumerate(monomers):  # 遍历当前工作流中的一组对象或任务。
        mon, _ = qm.conformation_search(  # 执行构象搜索/几何优化，为 RESP 做准备。
            mon, ff=ff, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
            psi4_omp=omp_psi4, mpi=mpi, omp=omp, memory=mem_mb, log_name=None  # 设置中间变量或可调参数，供后续工作流使用。
        )
        qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
            mon,
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            opt=False,  # 设置中间变量或可调参数，供后续工作流使用。
            work_dir=work_dir,  # 设置本例输出目录。
            omp=omp_psi4,  # 设置每个 rank 的 OpenMP 线程数。
            memory=mem_mb,  # 设置中间变量或可调参数，供后续工作流使用。
            log_name=None,  # 设置中间变量或可调参数，供后续工作流使用。
            polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        )
        monomers[i] = mon  # 设置中间变量或可调参数，供后续工作流使用。

    glucose, glucose_2, glucose_3, glucose_6 = monomers  # 设置中间变量或可调参数，供后续工作流使用。

    # Quick sanity: make sure charges exist (prevents silent all-zero ITPs)
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        qabs = sum(abs(a.GetDoubleProp('AtomicCharge')) for a in glucose.GetAtoms() if a.HasProp('AtomicCharge'))  # 设置中间变量或可调参数，供后续工作流使用。
        if qabs < 1.0e-6:  # 根据当前状态决定是否进入该分支。
            raise RuntimeError('Monomer charges are missing after QM. Check the psi4/psiresp-base installation and restart cache.')  # 关键步骤失败时立即报错，避免继续生成错误结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        pass

    # termination
    ter1 = utils.mol_from_smiles(ter_smiles)  # 从 SMILES 直接构造 RDKit 分子。
    qm.assign_charges(ter1, charge="RESP", opt=True, work_dir=work_dir, omp=omp_psi4, memory=mem_mb, log_name=None)  # 执行 RESP/ESP 电荷分配。

    # DP from target polymer Mw
    dp = poly.calc_n_from_mol_weight(  # 设置或计算聚合度。
        [glucose, glucose_2, glucose_3, glucose_6],
        target_mw,
        ratio=feed_prob,  # 设置共聚组成比例。
        terminal1=ter1,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    # random copolymerization (self-avoiding RW), then terminate
    CMC = poly.random_copolymerize_rw(  # 用随机游走生成聚合物链。
        [glucose, glucose_2, glucose_3, glucose_6],
        int(dp),
        ratio=feed_prob,  # 设置共聚组成比例。
        tacticity='atactic',  # 设置聚合物立构。
        work_dir=cmc_rw_dir,  # 设置本例输出目录。
    )
    CMC = poly.terminate_rw(CMC, ter1, work_dir=cmc_term_dir)  # 给聚合物链加端基。
    CMC = ff.ff_assign(CMC)  # 分配力场参数并写入分子属性。
    if not CMC:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError('Can not assign force field parameters for CMC.')  # 关键步骤失败时立即报错，避免继续生成错误结果。

    # Sanity-check atomtype coverage (helps diagnose RDF coverage later).
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        atypes = sorted({a.GetProp('ff_type') for a in CMC.GetAtoms() if a.HasProp('ff_type')})  # 设置中间变量或可调参数，供后续工作流使用。
        print(f"[CMC] unique ff_type = {atypes}")  # 打印关键路径或状态，便于人工检查。
        if ('oh' not in atypes) or ('o' not in atypes):  # 根据当前状态决定是否进入该分支。
            print("[WARN] CMC atomtypes missing expected 'oh'/'o'. RDF/ndx coverage may be limited. "  # 打印关键路径或状态，便于人工检查。
                  "Check that hydrogens/valences are present and ff_assign succeeded as intended.")
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        pass

    # ---------------- build solvents ----------------
    solvent_specs = [  # 设置中间变量或可调参数，供后续工作流使用。
        ("EC", utils.mol_from_smiles(EC_smiles)),  # 从 SMILES 直接构造 RDKit 分子。
        ("EMC", utils.mol_from_smiles(EMC_smiles)),  # 从 SMILES 直接构造 RDKit 分子。
        ("DEC", utils.mol_from_smiles(DEC_smiles)),  # 从 SMILES 直接构造 RDKit 分子。
    ]
    solvent_map = {}  # 设置中间变量或可调参数，供后续工作流使用。
    for solvent_name, solvent in solvent_specs:  # 遍历当前工作流中的一组对象或任务。
        solvent, _ = qm.conformation_search(  # 执行构象搜索/几何优化，为 RESP 做准备。
            solvent, ff=ff, work_dir=work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
            psi4_omp=omp_psi4, mpi=mpi, omp=omp, memory=mem_mb, log_name=None  # 设置中间变量或可调参数，供后续工作流使用。
        )
        qm.assign_charges(solvent, charge="RESP", opt=False, work_dir=work_dir, omp=omp_psi4, memory=mem_mb, log_name=None)  # 执行 RESP/ESP 电荷分配。
        solvent = ff.ff_assign(solvent)  # 分配力场参数并写入分子属性。
        if not solvent:  # 根据当前状态决定是否进入该分支。
            raise RuntimeError(f"Can not assign force field parameters for {solvent_name}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
        solvent_map[solvent_name] = solvent  # 设置中间变量或可调参数，供后续工作流使用。
    EC = solvent_map["EC"]  # 设置中间变量或可调参数，供后续工作流使用。
    EMC = solvent_map["EMC"]  # 设置中间变量或可调参数，供后续工作流使用。
    DEC = solvent_map["DEC"]  # 设置中间变量或可调参数，供后续工作流使用。

    # ---------------- ions ----------------
    Li = ion_ff.mol(Li_smiles)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    Li = ion_ff.ff_assign(Li)  # 分配力场参数并写入分子属性。
    if not Li:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Can not assign MERZ force field parameters for Li+.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    Na = ion_ff.mol(Na_smiles)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    Na = ion_ff.ff_assign(Na)  # 分配力场参数并写入分子属性。
    if not Na:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Can not assign MERZ force field parameters for Na+.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        PF6 = ff.mol(PF6_smiles, charge='RESP', require_ready=True, prefer_db=True)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
        PF6 = ff.ff_assign(PF6, bonded='DRIH')  # 分配力场参数并写入分子属性。
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
            "PF6 is expected to be precomputed in MolDB for this example. "
            "Please build it first with examples/01_Li_salt/run_pf6_to_moldb.py."
        ) from exc
    if not PF6:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError('Can not assign force field parameters for MolDB-backed PF6.')  # 关键步骤失败时立即报错，避免继续生成错误结果。

    # ---------------- compute counts ----------------
    # polymer molecular weight from RDKit
    mw_CMC = molecular_weight(CMC, strict=True)  # 设置中间变量或可调参数，供后续工作流使用。
    NA = 6.02214076e23  # 设置中间变量或可调参数，供后续工作流使用。

    # total polymer mass (g) for n_CMC molecules
    m_poly_g = mw_CMC * n_CMC / NA  # 设置中间变量或可调参数，供后续工作流使用。
    # total solvent mass (g) based on polymer feed mass
    m_solvent_total_g = m_poly_g * solvent_wt_over_polymer  # 设置中间变量或可调参数，供后续工作流使用。
    m_total_g = m_poly_g + m_solvent_total_g  # 设置中间变量或可调参数，供后续工作流使用。
    m_solvent_each_g = m_solvent_total_g / 3.0  # 设置中间变量或可调参数，供后续工作流使用。

    # solvent MWs
    mw_EC = molecular_weight(EC, strict=True)  # 设置中间变量或可调参数，供后续工作流使用。
    mw_EMC = molecular_weight(EMC, strict=True)  # 设置中间变量或可调参数，供后续工作流使用。
    mw_DEC = molecular_weight(DEC, strict=True)  # 设置中间变量或可调参数，供后续工作流使用。

    # molecule counts implied by 1:1:1 mass ratio (lower bound 20 each)
    n_EC = max(20, int(round((m_solvent_each_g / mw_EC) * NA)))  # 设置中间变量或可调参数，供后续工作流使用。
    n_EMC = max(20, int(round((m_solvent_each_g / mw_EMC) * NA)))  # 设置中间变量或可调参数，供后续工作流使用。
    n_DEC = max(20, int(round((m_solvent_each_g / mw_DEC) * NA)))  # 设置中间变量或可调参数，供后续工作流使用。

    # estimate volume from total mass and target density
    vol_cm3 = m_total_g / density_target_g_cm3  # 设置中间变量或可调参数，供后续工作流使用。
    vol_L = vol_cm3 / 1000.0  # 设置中间变量或可调参数，供后续工作流使用。

    # 1 M LiPF6 -> ion pairs
    n_LiPF6 = int(round(salt_molarity_M * vol_L * NA))  # 设置中间变量或可调参数，供后续工作流使用。
    if n_LiPF6 < min_salt_pairs:  # 根据当前状态决定是否进入该分支。
        n_LiPF6 = min_salt_pairs  # 设置中间变量或可调参数，供后续工作流使用。

    # Na+ neutralization: polymer formal charge (per chain) * n_CMC
    q_poly = mol_formal_charge(CMC)  # 设置中间变量或可调参数，供后续工作流使用。
    n_Na = int(abs(q_poly) * n_CMC) if q_poly != 0 else 0

    counts = [n_CMC, n_EC, n_EMC, n_DEC, n_LiPF6, n_LiPF6, n_Na]  # 设置各 species 的数量；顺序必须和 species 列表一致。

    print("[FORMULATION]")  # 打印关键路径或状态，便于人工检查。
    print(f"  feed_ratio = {feed_ratio} -> prob = {feed_prob}")  # 打印关键路径或状态，便于人工检查。
    print(f"  dp = {dp}, Mw(CMC)~{mw_CMC:.1f} g/mol, n_CMC={n_CMC}, formal_charge={q_poly}")  # 打印关键路径或状态，便于人工检查。
    print(f"  solvents: n_EC={n_EC}, n_EMC={n_EMC}, n_DEC={n_DEC} (1:1:1 mass)")  # 打印关键路径或状态，便于人工检查。
    print(f"  salt: LiPF6={n_LiPF6} pairs (1M, min {min_salt_pairs})")  # 打印关键路径或状态，便于人工检查。
    print(f"  Na+: {n_Na} (neutralize polymer)")  # 打印关键路径或状态，便于人工检查。
    print(f"  density={density_target_g_cm3} g/cm3 => V~{vol_L:.3e} L")  # 打印关键路径或状态，便于人工检查。

    # ---------------- pack amorphous cell ----------------
    ac = poly.amorphous_cell(  # 构建无定形混合体系初始盒子。
        [CMC, EC, EMC, DEC, Li, PF6, Na],
        counts,
        charge_scale=charge_scale,  # 设置电荷缩放系数；1.0 表示全电荷模型。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        density=density_pack_g_cm3,  # 设置中间变量或可调参数，供后续工作流使用。
        neutralize=False,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=ac_build_dir,  # 设置本例输出目录。
    )

    # ---------------- run equilibration preset ----------------
    eqmd = eq.EQ21step(ac, work_dir=work_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。

    analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    # Additional equilibration MD (up to 4 cycles)
    for i in range(4):  # 遍历当前工作流中的一组对象或任务。
        if result:  # 根据当前状态决定是否进入该分支。
            break
        eqmd = eq.Additional(ac, work_dir=work_dir)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
        analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    if not result:  # 根据当前状态决定是否进入该分支。
        print('[WARNING: Did not reach an equilibrium state after EQ21 + Additional cycles.]')  # 打印关键路径或状态，便于人工检查。

    # ---------------- Production NPT + analysis ----------------
    # IMPORTANT:
    #   Do NOT analyze from the last equilibration stage directory (EQ/Additional).
    #   Production (NPT/NVT) generates the trajectory used for RDF/MSD/sigma.
    npt = eq.NPT(ac, work_dir=work_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = npt.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=20)  # 设置中间变量或可调参数，供后续工作流使用。

    center_molecule = Li  # 设置中间变量或可调参数，供后续工作流使用。
    analy = npt.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。

    rdf = analy.rdf(center_mol=center_molecule)  # 设置中间变量或可调参数，供后续工作流使用。
    msd = analy.msd()  # 设置中间变量或可调参数，供后续工作流使用。
    sigma = analy.sigma(temp_k=temp, msd=msd)  # 设置中间变量或可调参数，供后续工作流使用。
    density_distributionr = analy.den_dis()  # 设置中间变量或可调参数，供后续工作流使用。
