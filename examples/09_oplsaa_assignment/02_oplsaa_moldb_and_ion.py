from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 09 / Step 2: OPLS-AA with MolDB-backed charges and a simple ion."""

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import get_ff, load_from_moldb  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.io.gmx import write_gmx  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


restart_status = False  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = get_ff("oplsaa")  # 选择有机分子/聚合物/部分无机离子的力场对象。

smiles_ec = "O=C1OCCO1"  # 设置中间变量或可调参数，供后续工作流使用。
smiles_na = "[Na+]"  # 设置中间变量或可调参数，供后续工作流使用。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = BASE_DIR / "work_dir"  # 设置本例输出目录。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    example_wd = workdir(work_dir, restart=restart_status)  # 创建或复用本例工作目录。

    ec = load_from_moldb(  # 设置中间变量或可调参数，供后续工作流使用。
        smiles_ec,
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        require_ready=True,  # 要求 MolDB 物种必须已准备好。
    )
    print("[MolDB] loaded EC with external RESP charges.")  # 打印关键路径或状态，便于人工检查。
    if not ff.ff_assign(ec, charge=None):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("OPLS-AA assignment failed for MolDB-backed EC with external RESP charges.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    ec_out = example_wd.child("02_ec_from_moldb_gmx")  # 设置中间变量或可调参数，供后续工作流使用。
    ec_gro, ec_itp, ec_top = write_gmx(mol=ec, out_dir=ec_out, mol_name="EC")  # 设置中间变量或可调参数，供后续工作流使用。

    na = ff.mol(smiles_na)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    if not ff.ff_assign(na):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("OPLS-AA assignment failed for Na+.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    na_out = example_wd.child("03_na_gmx")  # 设置中间变量或可调参数，供后续工作流使用。
    na_gro, na_itp, na_top = write_gmx(mol=na, out_dir=na_out, mol_name="Na")  # 设置中间变量或可调参数，供后续工作流使用。

    print("[DONE] OPLS-AA assignment completed for MolDB-backed EC while preserving RESP charges.")  # 打印关键路径或状态，便于人工检查。
    print(f"  EC gro : {ec_gro}")  # 打印关键路径或状态，便于人工检查。
    print(f"  EC itp : {ec_itp}")  # 打印关键路径或状态，便于人工检查。
    print(f"  EC top : {ec_top}")  # 打印关键路径或状态，便于人工检查。
    print("[DONE] OPLS-AA assignment completed for Na+ using the built-in OPLS-AA ion parameters.")  # 打印关键路径或状态，便于人工检查。
    print(f"  Na gro : {na_gro}")  # 打印关键路径或状态，便于人工检查。
    print(f"  Na itp : {na_itp}")  # 打印关键路径或状态，便于人工检查。
    print(f"  Na top : {na_top}")  # 打印关键路径或状态，便于人工检查。
