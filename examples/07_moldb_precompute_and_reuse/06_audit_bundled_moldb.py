from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

import json  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.core.data_dir import audit_active_bundle_sync, ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。


BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
work_dir = BASE_DIR / "work_dir" / "06_audit_bundled_moldb"  # 设置本例输出目录。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    layout = ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。
    audit = audit_active_bundle_sync(layout=layout)  # 设置中间变量或可调参数，供后续工作流使用。

    work_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    summary_path = work_dir / "bundle_sync_audit.json"  # 设置中间变量或可调参数，供后续工作流使用。
    summary_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。

    print("\n[MolDB bundled sync audit]")  # 打印关键路径或状态，便于人工检查。
    print(f"layout_root = {audit['layout_root']}")  # 打印关键路径或状态，便于人工检查。
    print(f"moldb_dir = {audit['moldb_dir']}")  # 打印关键路径或状态，便于人工检查。
    print(f"bundle_dir = {audit['bundle_dir']}")  # 打印关键路径或状态，便于人工检查。
    print(f"missing_objects = {len(audit['missing_objects'])}")  # 打印关键路径或状态，便于人工检查。
    print(f"stale_variants = {len(audit['stale_variants'])}")  # 打印关键路径或状态，便于人工检查。
    print(f"bundled_more_complete_records = {len(audit['bundled_more_complete_records'])}")  # 打印关键路径或状态，便于人工检查。
    print(f"user_only_records = {len(audit['user_only_records'])}")  # 打印关键路径或状态，便于人工检查。
    print(f"summary_path = {summary_path}")  # 打印关键路径或状态，便于人工检查。
