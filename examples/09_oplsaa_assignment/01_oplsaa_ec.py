from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 09 / Step 1: OPLS-AA assignment for ethylene carbonate."""

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import get_ff, mol_from_smiles  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.io.gmx import write_gmx  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


restart_status = False  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = get_ff("oplsaa")  # 选择有机分子/聚合物/部分无机离子的力场对象。
smiles_ec = "O=C1OCCO1"  # 设置中间变量或可调参数，供后续工作流使用。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = BASE_DIR / "work_dir"  # 设置本例输出目录。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    example_wd = workdir(work_dir, restart=restart_status)  # 创建或复用本例工作目录。

    ec = mol_from_smiles(smiles_ec)  # 设置中间变量或可调参数，供后续工作流使用。
    if not ff.ff_assign(ec, charge="opls"):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("OPLS-AA assignment failed for ethylene carbonate (EC).")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    out_dir = example_wd.child("01_ec_gmx")  # 设置中间变量或可调参数，供后续工作流使用。
    gro_path, itp_path, top_path = write_gmx(mol=ec, out_dir=out_dir, mol_name="EC")  # 设置中间变量或可调参数，供后续工作流使用。

    print("[DONE] OPLS-AA assignment completed for EC.")  # 打印关键路径或状态，便于人工检查。
    print(f"  gro : {gro_path}")  # 打印关键路径或状态，便于人工检查。
    print(f"  itp : {itp_path}")  # 打印关键路径或状态，便于人工检查。
    print(f"  top : {top_path}")  # 打印关键路径或状态，便于人工检查。
    print(f"  angles: {len(getattr(ec, 'angles', {}))}")  # 打印关键路径或状态，便于人工检查。
    print(f"  dihedrals: {len(getattr(ec, 'dihedrals', {}))}")  # 打印关键路径或状态，便于人工检查。
    print("This example stops after OPLS-AA typing, charge assignment, and export.")  # 打印关键路径或状态，便于人工检查。
