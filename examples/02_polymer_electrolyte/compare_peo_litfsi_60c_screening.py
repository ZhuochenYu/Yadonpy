from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

import argparse  # 导入本例需要的库或 yadonpy 接口。
import json  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.sim.benchmarking import _dump_json, build_screening_compare, load_benchmark_analysis_dir  # 导入本例需要的库或 yadonpy 接口。


def _default_output(paths: list[Path]) -> Path:  # 定义本例内部辅助函数，组织重复步骤。
    if not paths:  # 根据当前状态决定是否进入该分支。
        return Path.cwd() / "screening_compare.json"  # 返回该辅助函数的结果。
    parent = paths[0].parent  # 设置中间变量或可调参数，供后续工作流使用。
    if parent.name == "06_analysis":  # 根据当前状态决定是否进入该分支。
        return parent.parent.parent / "screening_compare.json"  # 返回该辅助函数的结果。
    return parent / "screening_compare.json"  # 返回该辅助函数的结果。


def main() -> None:  # 定义本例内部辅助函数，组织重复步骤。
    parser = argparse.ArgumentParser(description="Compare multiple PEO/LiTFSI 60C screening runs.")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument(  # 开始一个多行函数调用或配置块。
        "analysis_dirs",
        nargs="+",  # 设置中间变量或可调参数，供后续工作流使用。
        help="Paths to analysis directories that contain benchmark_compare.json and companion analysis JSON files.",  # 设置中间变量或可调参数，供后续工作流使用。
    )
    parser.add_argument("--out", default=None, help="Output JSON path. Defaults near the provided screening directories.")  # 设置中间变量或可调参数，供后续工作流使用。
    args = parser.parse_args()  # 设置中间变量或可调参数，供后续工作流使用。

    analysis_dirs = [Path(p).resolve() for p in args.analysis_dirs]  # 设置中间变量或可调参数，供后续工作流使用。
    runs = [load_benchmark_analysis_dir(path) for path in analysis_dirs]  # 设置中间变量或可调参数，供后续工作流使用。
    payload = build_screening_compare(runs=runs)  # 设置中间变量或可调参数，供后续工作流使用。
    out_path = Path(args.out).resolve() if args.out else _default_output(analysis_dirs)  # 设置中间变量或可调参数，供后续工作流使用。
    _dump_json(out_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))  # 打印关键路径或状态，便于人工检查。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    main()
