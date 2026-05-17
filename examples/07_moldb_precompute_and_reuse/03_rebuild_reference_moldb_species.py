from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Legacy alias for the consolidated Example 07 MolDB builder."""

import runpy  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    print("[INFO] Example 07 Step 3 is a legacy alias. Redirecting to 01_build_moldb.py.")  # 打印关键路径或状态，便于人工检查。
    runpy.run_path(str(Path(__file__).with_name("01_build_moldb.py")), run_name="__main__")  # 设置中间变量或可调参数，供后续工作流使用。
