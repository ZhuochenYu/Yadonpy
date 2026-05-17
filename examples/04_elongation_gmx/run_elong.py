from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import (  # 导入本例需要的库或 yadonpy 接口。
    print_mechanics_result_summary,
    resolve_prepared_system,
    run_elongation_gmx,
    set_run_options,
)
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。


restart_status = True  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
SYSTEM_WORK_DIR = BASE_DIR.parent / "02_polymer_electrolyte" / "work_dir"  # 设置中间变量或可调参数，供后续工作流使用。
OUT_DIR = BASE_DIR / "work_dir"  # 设置中间变量或可调参数，供后续工作流使用。


def main() -> None:  # 定义本例内部辅助函数，组织重复步骤。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    prepared = resolve_prepared_system(  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=SYSTEM_WORK_DIR,  # 设置本例输出目录。
        source_name="example02_equilibrated_system",  # 设置中间变量或可调参数，供后续工作流使用。
    )
    result = run_elongation_gmx(  # 设置中间变量或可调参数，供后续工作流使用。
        prepared=prepared,  # 设置中间变量或可调参数，供后续工作流使用。
        out_dir=OUT_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
        profile="default",  # 设置中间变量或可调参数，供后续工作流使用。
        restart=restart_status,  # 传入断点续跑开关。
    )
    print_mechanics_result_summary(result)


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    main()
