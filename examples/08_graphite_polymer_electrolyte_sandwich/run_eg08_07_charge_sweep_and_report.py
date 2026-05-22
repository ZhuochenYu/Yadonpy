from __future__ import annotations

# YadonPy example annotation:
# - 这个脚本是 Eg08.07 四电荷 sweep 的“收尾入口”，不负责启动 MD。
# - 它会等待 0/-3/-9/-18 uC/cm2 case 完成，并发后处理，再生成完整版 PPT。
# - 保持和示例脚本一致的显式 API/脚本风格。

"""Wait for Eg08.07 charge-sweep jobs, post-process in parallel, and build PPT."""

import json  # 写入 pipeline summary，便于远端直接检查状态。
import os  # 通过环境变量传递 sweep root、worker 数和 PPT 参数。
import subprocess  # 调用 postprocess_charge_sweep_parallel.py 和 make_charge_sweep_report_ppt.py。
import sys  # 使用当前 Python 解释器运行兄弟脚本。
import time  # 记录流水线耗时。
from pathlib import Path  # 处理远端路径。


DEFAULT_CASES = (  # Eg08.07 共享 t=0 电荷 sweep 的标准四个目录。
    ("0 uC/cm2", "cmcface_00_uC_cm2", 0.0),
    ("-3 uC/cm2", "cmcface_m3p0_uC_cm2", -3.0),
    ("-9 uC/cm2", "cmcface_m9p0_uC_cm2", -9.0),
    ("-18 uC/cm2", "cmcface_m18p0_uC_cm2", -18.0),
)


def env_bool(name: str, default: bool) -> bool:  # 从环境变量读取布尔开关。
    value = os.environ.get(name)  # 读取变量。
    if value is None:  # 未设置时使用默认。
        return bool(default)  # 返回默认布尔值。
    return value.strip().lower() not in {"0", "false", "no", "off"}  # 常见否定字符串视为 False。


def sweep_root() -> Path:  # 解析 sweep 根目录。
    default = Path(__file__).resolve().parent / "work_dir" / "07_cmcna_xy_slab_matched_graphite_electrolyte_cmcna_graphite"  # 本地默认路径。
    return Path(os.environ.get("EG08_SWEEP_ROOT", str(default))).expanduser().resolve()  # 远端通常显式设置 EG08_SWEEP_ROOT。


def wait_for_cases(root: Path, *, poll_s: float) -> dict[str, object]:  # 等待四个 case 写出完成标记。
    report_dir = root / "99_report"  # 报告目录。
    report_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在。
    while True:  # 循环等待。
        rows = []  # 每个 case 的状态。
        missing = []  # 尚未完成的 marker。
        for label, dirname, charge in DEFAULT_CASES:  # 遍历标准四电荷。
            done = root / dirname / "charge_case_done.json"  # MD 脚本完成后写出的 marker。
            ok = done.is_file()  # 是否已完成。
            rows.append({"label": label, "dirname": dirname, "charge_uC_cm2": charge, "done": ok, "done_json": str(done)})  # 记录状态。
            if not ok:  # 尚未完成。
                missing.append(str(done))  # 加入等待列表。
        status = {"root": str(root), "cases": rows, "missing_done": missing, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # 状态 payload。
        (report_dir / "charge_sweep_wait_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 写出等待状态。
        if not missing:  # 全部完成。
            return status  # 返回最终状态。
        time.sleep(max(30.0, float(poll_s)))  # 避免频繁轮询。


def run_python(script: Path, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:  # 用当前解释器运行兄弟脚本。
    cmd = [sys.executable, str(script)]  # 命令。
    print("running:", " ".join(cmd))  # 远端日志中记录命令。
    return subprocess.run(cmd, cwd=str(script.parent), env=env, check=True, text=True)  # 失败时抛错。


def main() -> None:  # 主入口。
    start = time.time()  # 记录开始时间。
    root = sweep_root()  # 解析 sweep root。
    root.mkdir(parents=True, exist_ok=True)  # 确保根目录存在。
    report_dir = root / "99_report"  # 汇总报告目录。
    report_dir.mkdir(parents=True, exist_ok=True)  # 创建报告目录。

    if env_bool("EG08_WAIT_FOR_DONE", True):  # 默认等待四个 case 完成。
        wait_for_cases(root, poll_s=float(os.environ.get("EG08_WAIT_POLL_S", "600")))  # 等待 marker。

    script_dir = Path(__file__).resolve().parent  # 当前脚本目录。
    env = dict(os.environ)  # 复制环境变量。
    env["EG08_SWEEP_ROOT"] = str(root)  # 统一传递 sweep root。
    env.setdefault("EG08_POSTPROCESS_WORKERS", "auto")  # 默认 case-level 并发。
    env.setdefault("EG08_POSTPROCESS_THREAD_LIMIT", "1")  # 每个 worker 限制 BLAS/OpenMP 线程，避免过订阅。
    env.setdefault("EG08_ANALYSIS_PROFILE", "interface_fast")  # 长轨迹默认快速界面分析。
    env.setdefault("EG08_ANALYSIS_FRAME_STRIDE", "auto")  # 让 analyzer 根据帧数自适应抽帧。
    env.setdefault("EG08_TIME_SERIES_ANALYSIS", "1")  # 保留时间序列 CSV/MP4。
    env.setdefault("EG08_REPORT_MAX_SIZE_MB", "50")  # PPT 目标大小。

    postprocess_script = script_dir / "postprocess_charge_sweep_parallel.py"  # 并发后处理脚本。
    ppt_script = script_dir / "make_charge_sweep_report_ppt.py"  # PPT 汇总脚本。
    post = run_python(postprocess_script, env=env)  # 先并发后处理。
    ppt = run_python(ppt_script, env=env)  # 再生成 PPT。

    ppt_candidates = sorted((report_dir).glob("*.pptx")) + sorted((report_dir / "ppt").glob("*.pptx"))  # 常见 PPT 输出位置。
    summary = {  # 汇总路径和状态。
        "root": str(root),
        "report_dir": str(report_dir),
        "postprocess_returncode": int(post.returncode),
        "ppt_returncode": int(ppt.returncode),
        "ppt_candidates": [str(path) for path in ppt_candidates],
        "summary_json": str(report_dir / "charge_sweep_pipeline_summary.json"),
        "elapsed_s": float(time.time() - start),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (report_dir / "charge_sweep_pipeline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 写出总摘要。
    (report_dir / "charge_sweep_full_ppt_paths.json").write_text(json.dumps({"ppt_paths": summary["ppt_candidates"]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 按用户要求给出 PPT 路径清单。
    print(json.dumps(summary, indent=2, ensure_ascii=False))  # 终端打印，方便复制路径。


if __name__ == "__main__":  # 直接运行时执行主入口。
    main()  # 执行。
