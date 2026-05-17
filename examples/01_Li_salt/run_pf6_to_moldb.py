from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

# Example 01: PF6- -> QM/RESP -> DRIH-aware MolDB entry -> DB-backed export
#
# This script does two things in one place:
#   1) compute PF6- once from scratch and store the finished result into MolDB;
#   2) immediately demonstrate the later workflow style that reuses MolDB via:
#        PF6 = ff.mol(PF6_smiles)
#        PF6 = ff.ff_assign(PF6, bonded="DRIH")
#
# After step (1), the second pair of lines becomes the recommended way to reuse PF6
# in later scripts. The bonded DRIH patch is restored from MolDB together with the
# stored geometry/charges for the selected variant.

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor, require_psi4_resp  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim import qm  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2_mod import GAFF2_mod  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.io.mol2 import write_mol2  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.io.gmx import write_gmx  # 导入本例需要的库或 yadonpy 接口。


def select_charge_method() -> tuple[str, str | None]:  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        require_psi4_resp()
    except ImportError as exc:  # 捕获异常并转成更清楚的示例错误信息。
        return "gasteiger", str(exc)  # 返回该辅助函数的结果。
    return "RESP", None  # 返回该辅助函数的结果。


def build_and_store_pf6(*, ff: GAFF2_mod, pf6_smiles: str, work_dir, omp_psi4: int, mem_mb: int, charge_method: str):  # 定义本例内部辅助函数，组织重复步骤。
    pf6 = utils.mol_from_smiles(pf6_smiles, coord=False, name="PF6")  # 从 SMILES 直接构造 RDKit 分子。
    utils.ensure_3d_coords(pf6, smiles_hint=pf6_smiles, engine="openbabel")  # 设置中间变量或可调参数，供后续工作流使用。

    if str(charge_method).strip().upper() == "RESP":  # 根据当前状态决定是否进入该分支。
        qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
            pf6,
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            opt=True,  # 设置中间变量或可调参数，供后续工作流使用。
            work_dir=work_dir,  # 设置本例输出目录。
            log_name="PF6_build",  # 设置中间变量或可调参数，供后续工作流使用。
            omp=omp_psi4,  # 设置每个 rank 的 OpenMP 线程数。
            memory=mem_mb,  # 设置中间变量或可调参数，供后续工作流使用。
            total_charge=-1,  # 设置中间变量或可调参数，供后续工作流使用。
            total_multiplicity=1,  # 设置中间变量或可调参数，供后续工作流使用。
            symmetrize=True,  # 设置中间变量或可调参数，供后续工作流使用。
            auto_level=True,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        pf6 = ff.ff_assign(pf6, bonded="DRIH")  # 分配力场参数并写入分子属性。
    else:  # 处理前面条件都不满足的情况。
        pf6 = ff.ff_assign(pf6, charge=charge_method, bonded="DRIH")  # 分配力场参数并写入分子属性。

    if not pf6:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("FF assignment failed for PF6 during the build phase")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    record = ff.store_to_db(pf6, smiles_or_psmiles=pf6_smiles, name="PF6", charge=charge_method)  # 设置中间变量或可调参数，供后续工作流使用。
    return pf6, record  # 返回该辅助函数的结果。


def main():  # 定义本例内部辅助函数，组织重复步骤。
    restart_status = False  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
    set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

    PF6_smiles = "F[P-](F)(F)(F)(F)F"  # 设置中间变量或可调参数，供后续工作流使用。
    omp_psi4 = 8  # 设置 Psi4/OpenMP 核数。
    mem_mb = 8000  # 设置量子化学内存 MB。
    work_root = Path("work_pf6_only").resolve()  # 设置中间变量或可调参数，供后续工作流使用。

    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    work_dir = workdir(work_root, restart=restart_status)  # 设置本例输出目录。
    mol2_dir = work_dir / "00_molecules"  # 设置中间变量或可调参数，供后续工作流使用。
    build_export_dir = work_dir / "01_pf6_build_exports"  # 设置中间变量或可调参数，供后续工作流使用。
    db_export_dir = work_dir / "02_pf6_from_moldb_gmx"  # 设置中间变量或可调参数，供后续工作流使用。
    mol2_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    build_export_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    db_export_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。

    ff = GAFF2_mod()  # 选择有机分子/聚合物/部分无机离子的力场对象。
    charge_method, qm_fallback_reason = select_charge_method()  # 设置中间变量或可调参数，供后续工作流使用。

    if qm_fallback_reason is None:  # 根据当前状态决定是否进入该分支。
        print("[INFO] Example 01 will build PF6 with QM/RESP, then store it into MolDB.")  # 打印关键路径或状态，便于人工检查。
    else:  # 处理前面条件都不满足的情况。
        print("[WARN] Optional QM stack is unavailable; Example 01 will fall back to Gasteiger charges so the full MolDB workflow still runs.")  # 打印关键路径或状态，便于人工检查。
        print(f"[WARN] Missing QM dependency detail: {qm_fallback_reason}")  # 打印关键路径或状态，便于人工检查。
        print("[WARN] This run still demonstrates the same direct MolDB API, but the stored charge variant is Gasteiger instead of RESP.")  # 打印关键路径或状态，便于人工检查。

    pf6_built, record = build_and_store_pf6(  # 设置中间变量或可调参数，供后续工作流使用。
        ff=ff,  # 选择有机分子/聚合物/部分无机离子的力场对象。
        pf6_smiles=PF6_smiles,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=work_dir,  # 设置本例输出目录。
        omp_psi4=omp_psi4,  # 设置 Psi4/OpenMP 核数。
        mem_mb=mem_mb,  # 设置量子化学内存 MB。
        charge_method=charge_method,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    built_mol2 = write_mol2(mol=pf6_built, out_dir=mol2_dir, name="PF6_built")  # 设置中间变量或可调参数，供后续工作流使用。
    built_gro, built_itp, built_top = write_gmx(  # 设置中间变量或可调参数，供后续工作流使用。
        mol=pf6_built,  # 设置中间变量或可调参数，供后续工作流使用。
        out_dir=build_export_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        mol_name="PF6",  # 设置中间变量或可调参数，供后续工作流使用。
    )

    if str(charge_method).strip().upper() == "RESP":  # 根据当前状态决定是否进入该分支。
        PF6 = ff.mol(PF6_smiles)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    else:  # 处理前面条件都不满足的情况。
        PF6 = ff.mol(PF6_smiles, charge=charge_method)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    PF6 = ff.ff_assign(PF6, bonded="DRIH")  # 分配力场参数并写入分子属性。
    if not PF6:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("FF assignment failed for MolDB-backed PF6")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    db_mol2 = write_mol2(mol=PF6, out_dir=mol2_dir, name="PF6_from_moldb")  # 设置中间变量或可调参数，供后续工作流使用。
    db_gro, db_itp, db_top = write_gmx(  # 设置中间变量或可调参数，供后续工作流使用。
        mol=PF6,  # 设置中间变量或可调参数，供后续工作流使用。
        out_dir=db_export_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        mol_name="PF6",  # 设置中间变量或可调参数，供后续工作流使用。
    )

    print("\n[DONE] PF6- was computed, stored to MolDB, then reloaded from MolDB for export:")  # 打印关键路径或状态，便于人工检查。
    print(f"  MolDB key          : {record.key}")  # 打印关键路径或状态，便于人工检查。
    print(f"  charge variant     : {charge_method}")  # 打印关键路径或状态，便于人工检查。
    print(f"  built mol2         : {built_mol2}")  # 打印关键路径或状态，便于人工检查。
    print(f"  built gro/itp/top  : {built_gro} | {built_itp} | {built_top}")  # 打印关键路径或状态，便于人工检查。
    print(f"  db mol2            : {db_mol2}")  # 打印关键路径或状态，便于人工检查。
    print(f"  db gro/itp/top     : {db_gro} | {db_itp} | {db_top}")  # 打印关键路径或状态，便于人工检查。

    print("\nNotes:")  # 打印关键路径或状态，便于人工检查。
    print(f"  - 01_pf6_build_exports/ contains the artifacts written immediately after charge assignment ({charge_method}) + DRIH build.")  # 打印关键路径或状态，便于人工检查。
    print("  - 02_pf6_from_moldb_gmx/ contains the artifacts written after reloading PF6 through the MolDB-backed ff.mol(...) path.")  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    print("  - Later workflows can now reuse PF6 with the following pattern:")  # 打印关键路径或状态，便于人工检查。
    if str(charge_method).strip().upper() == "RESP":  # 根据当前状态决定是否进入该分支。
        print("    PF6 = ff.mol(PF6_smiles)")  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    else:  # 处理前面条件都不满足的情况。
        print(f"    PF6 = ff.mol(PF6_smiles, charge=\"{charge_method}\")")  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    print("    PF6 = ff.ff_assign(PF6, bonded=\"DRIH\")")  # 分配力场参数并写入分子属性。
    print("  - If PF6_smiles is the literal string, that call is:")  # 打印关键路径或状态，便于人工检查。
    if str(charge_method).strip().upper() == "RESP":  # 根据当前状态决定是否进入该分支。
        print("    PF6 = ff.mol(\"F[P-](F)(F)(F)(F)F\")")  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    else:  # 处理前面条件都不满足的情况。
        print(f"    PF6 = ff.mol(\"F[P-](F)(F)(F)(F)F\", charge=\"{charge_method}\")")  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    print("    PF6 = ff.ff_assign(PF6, bonded=\"DRIH\")\n")  # 分配力场参数并写入分子属性。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    main()
