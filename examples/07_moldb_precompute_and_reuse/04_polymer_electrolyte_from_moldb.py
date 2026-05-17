from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 07 / Step 4: Polymer electrolyte workflow using MolDB.

Prerequisite:
  - Run 01_build_moldb.py first so the example species are already "ready" in
    MolDB.

Notes:
    - Example 07 now assumes the precompute step has already filled MolDB with
        a broad electrolyte library, including PF6-.
"""

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import poly, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2_mod import GAFF2_mod  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.merz import MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.preset import eq  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.io.mol2 import write_mol2  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.io.gmx import write_gmx  # 导入本例需要的库或 yadonpy 接口。


# ---------------- user inputs ----------------
restart_status = True  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # default: GAFF2_mod
cation_ff = MERZ()  # 选择阳离子的力场/参数来源。

# ---- two monomers (both must have two connection points '*...*') ----
smiles_A = r"*CCO*"  # 设置中间变量或可调参数，供后续工作流使用。
smiles_B = r"*COC*"  # 设置中间变量或可调参数，供后续工作流使用。

# ---- solvents / ions (no '*') ----
solvent_smiles_A = "CCOC(=O)OC"  # EMC
solvent_smiles_B = "O=C1OCCO1"  # EC

cation_smiles_A = "[Li+]"  # 设置中间变量或可调参数，供后续工作流使用。

ter_smiles = "[H][*]"  # 设置中间变量或可调参数，供后续工作流使用。

ratio = [0.7, 0.3]  # 设置共聚组成比例。
reac_ratio: list[float] = []  # 设置 Mayo-Lewis 反应比；空列表表示不加反应偏置。

# MD settings
temp = 300.0  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
press = 1.0  # 设置压力 bar；用于 NPT/EQ 阶段。
mpi = 1  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = 16  # 设置每个 rank 的 OpenMP 线程数。

gpu = 1  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = 0  # 选择 GPU 设备编号，多卡节点可修改。

sim_time_ns = 5.0  # 设置生产模拟时长，单位 ns。

# Packing
density_g_cm3 = 0.05  # 设置初始 packing 密度，主要影响构建难度和初始盒子大小。
counts = [4, 20, 20, 20]  # 设置各 species 的数量；顺序必须和 species 列表一致。
charge_scale = [1.0, 1.0, 1.0, 0.8]  # 设置电荷缩放系数；1.0 表示全电荷模型。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = BASE_DIR / "work_dir"  # 设置本例输出目录。


def _load_ready_from_moldb(ff, smiles: str, *, label: str, bonded: str | None = None):  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        mol = ff.mol(smiles, charge="RESP", require_ready=True, prefer_db=True)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
            f"{label} is expected to be precomputed in MolDB by "
            "examples/07_moldb_precompute_and_reuse/01_build_moldb.py."
        ) from exc
    mol = ff.ff_assign(mol, bonded=bonded, report=False)  # 分配力场参数并写入分子属性。
    if not mol:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"Cannot assign force field parameters for MolDB-backed {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return mol  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    work_dir = workdir(work_dir, restart=restart_status)  # 设置本例输出目录。
    poly_rw_dir = work_dir.child("copoly_rw")  # 设置中间变量或可调参数，供后续工作流使用。
    poly_term_dir = work_dir.child("copoly_term")  # 设置中间变量或可调参数，供后续工作流使用。
    ac_build_dir = work_dir.child("00_build_cell")  # 设置中间变量或可调参数，供后续工作流使用。

    # --- load precomputed molecules from MolDB (as lightweight handles) ---
    monomer_A = _load_ready_from_moldb(ff, smiles_A, label="monomer_A")  # 设置中间变量或可调参数，供后续工作流使用。
    monomer_B = _load_ready_from_moldb(ff, smiles_B, label="monomer_B")  # 设置中间变量或可调参数，供后续工作流使用。
    ter1 = _load_ready_from_moldb(ff, ter_smiles, label="ter1")  # 设置中间变量或可调参数，供后续工作流使用。
    solvent_A = _load_ready_from_moldb(ff, solvent_smiles_A, label="solvent_A")  # 设置中间变量或可调参数，供后续工作流使用。
    solvent_B = _load_ready_from_moldb(ff, solvent_smiles_B, label="solvent_B")  # 设置中间变量或可调参数，供后续工作流使用。

    # --- Li+ from MERZ ---
    cation_A = cation_ff.mol(cation_smiles_A)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    if not cation_ff.ff_assign(cation_A):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Cannot assign MERZ parameters to Li+")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    # --- Optional: reuse a MolDB-backed anion (example: PF6-) ---
    # anion_smiles_A = "F[P-](F)(F)(F)(F)F"
    # anion_A = _load_ready_from_moldb(ff, anion_smiles_A, label="PF6", bonded="DRIH")

    # --- Build polymer (restart-friendly, manual API style) ---
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
    if not ff.ff_assign(copoly):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Cannot assign force field parameters to copoly")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    # Optional export
    write_mol2(mol=copoly, out_dir=work_dir / "00_molecules", name="copoly")  # 设置中间变量或可调参数，供后续工作流使用。
    write_gmx(mol=copoly, out_dir=work_dir / "90_copoly_gmx", mol_name="copoly")  # 设置中间变量或可调参数，供后续工作流使用。
    write_mol2(mol=solvent_A, out_dir=work_dir / "00_molecules", name="EMC")  # 设置中间变量或可调参数，供后续工作流使用。
    write_gmx(mol=solvent_A, out_dir=work_dir / "90_EMC_gmx", mol_name="EMC")  # 设置中间变量或可调参数，供后续工作流使用。
    write_mol2(mol=solvent_B, out_dir=work_dir / "00_molecules", name="EC")  # 设置中间变量或可调参数，供后续工作流使用。
    write_gmx(mol=solvent_B, out_dir=work_dir / "90_EC_gmx", mol_name="EC")  # 设置中间变量或可调参数，供后续工作流使用。
    write_mol2(mol=cation_A, out_dir=work_dir / "00_molecules", name="Li")  # 设置中间变量或可调参数，供后续工作流使用。
    write_gmx(mol=cation_A, out_dir=work_dir / "90_Li_gmx", mol_name="Li")  # 设置中间变量或可调参数，供后续工作流使用。

    # --- build amorphous cell ---
    ac = poly.amorphous_cell(  # 构建无定形混合体系初始盒子。
        [copoly, solvent_A, solvent_B, cation_A],
        counts,
        charge_scale=charge_scale,  # 设置电荷缩放系数；1.0 表示全电荷模型。
        density=density_g_cm3,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=ac_build_dir,  # 设置本例输出目录。
    )

    # --- EQ21 + check equilibrium ---
    eqmd = eq.EQ21step(ac, work_dir=work_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。

    analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    ok = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    for _i in range(4):  # 遍历当前工作流中的一组对象或任务。
        if ok:  # 根据当前状态决定是否进入该分支。
            break
        eqmd = eq.Additional(ac, work_dir=work_dir)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
        analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        ok = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    if not ok:  # 根据当前状态决定是否进入该分支。
        raise SystemExit("[ERROR] Did not reach an equilibrium state.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    # --- Production NPT + analysis ---
    npt = eq.NPT(ac, work_dir=work_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = npt.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=sim_time_ns)  # 设置中间变量或可调参数，供后续工作流使用。

    analy = npt.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。

    center_molecule = cation_A  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.rdf(center_mol=center_molecule)  # 设置中间变量或可调参数，供后续工作流使用。
    msd = analy.msd()  # 设置中间变量或可调参数，供后续工作流使用。
    sigma = analy.sigma(msd=msd, temp_k=temp)  # 设置中间变量或可调参数，供后续工作流使用。

    print("msd:", msd)  # 打印关键路径或状态，便于人工检查。
    print("sigma:", sigma)  # 打印关键路径或状态，便于人工检查。
