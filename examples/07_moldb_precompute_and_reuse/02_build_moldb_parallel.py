from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 07 / Step 2: Auto-planned parallel MolDB build."""

import importlib.util  # 导入本例需要的库或 yadonpy 接口。
import json  # 导入本例需要的库或 yadonpy 接口。
import multiprocessing as mp  # 导入本例需要的库或 yadonpy 接口。
import os  # 导入本例需要的库或 yadonpy 接口。
import queue  # 导入本例需要的库或 yadonpy 接口。
import sys  # 导入本例需要的库或 yadonpy 接口。
import time  # 导入本例需要的库或 yadonpy 接口。
from dataclasses import asdict, dataclass  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.core import workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.moldb import MolDB  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


HERE = Path(__file__).resolve().parent  # 定位当前脚本所在目录。
BUILD_SCRIPT = HERE / "01_build_moldb.py"  # 设置中间变量或可调参数，供后续工作流使用。


@dataclass(frozen=True)  # 声明轻量数据类，用于保存配置或任务信息。
class ParallelTask:  # 定义本例内部数据结构或配置对象。
    name: str
    smiles: str
    charge: str
    bonded: str | None
    polyelectrolyte_mode: bool
    profile: str
    batch: str
    priority: int
    required_cores: int
    psi4_omp: int


def _load_build_module():  # 定义本例内部辅助函数，组织重复步骤。
    spec = importlib.util.spec_from_file_location("example07_build_moldb", BUILD_SCRIPT)  # 设置中间变量或可调参数，供后续工作流使用。
    assert spec is not None  # 检查示例假设是否成立。
    assert spec.loader is not None  # 检查示例假设是否成立。
    module = importlib.util.module_from_spec(spec)  # 设置中间变量或可调参数，供后续工作流使用。
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module  # 返回该辅助函数的结果。


def _available_cpu_total() -> int:  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        return max(1, len(os.sched_getaffinity(0)))  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return max(1, int(os.cpu_count() or 1))  # 返回该辅助函数的结果。


def _reserved_driver_cores(cpu_total: int) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    if cpu_total >= 32:  # 根据当前状态决定是否进入该分支。
        return 2  # 返回该辅助函数的结果。
    if cpu_total >= 8:  # 根据当前状态决定是否进入该分支。
        return 1  # 返回该辅助函数的结果。
    return 0  # 返回该辅助函数的结果。


def _planner_cpu_budget(cpu_total: int) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    return max(1, int(cpu_total) - _reserved_driver_cores(int(cpu_total)))  # 返回该辅助函数的结果。


def _task_profile(spec) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    if str(spec.bonded or "").strip().upper() == "DRIH":  # 根据当前状态决定是否进入该分支。
        return "drih"  # 返回该辅助函数的结果。
    if bool(spec.polyelectrolyte_mode):  # 根据当前状态决定是否进入该分支。
        return "polyelectrolyte"  # 返回该辅助函数的结果。
    if "*" in str(spec.smiles):  # 根据当前状态决定是否进入该分支。
        return "polymer"  # 返回该辅助函数的结果。
    return "standard"  # 返回该辅助函数的结果。


def _task_threads(*, profile: str, cpu_total: int) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    if profile == "light":  # 根据当前状态决定是否进入该分支。
        return 1  # 返回该辅助函数的结果。
    if cpu_total <= 8:  # 根据当前状态决定是否进入该分支。
        return max(1, cpu_total)  # 返回该辅助函数的结果。
    if profile == "drih":  # 根据当前状态决定是否进入该分支。
        return min(8, max(4, cpu_total // 4))  # 返回该辅助函数的结果。
    if profile == "polyelectrolyte":  # 根据当前状态决定是否进入该分支。
        return min(8, max(4, cpu_total // 5))  # 返回该辅助函数的结果。
    if profile == "polymer":  # 根据当前状态决定是否进入该分支。
        return min(6, max(3, cpu_total // 6))  # 返回该辅助函数的结果。
    return min(4, max(2, cpu_total // 8))  # 返回该辅助函数的结果。


def _task_batch(profile: str) -> tuple[str, int]:  # 定义本例内部辅助函数，组织重复步骤。
    if profile == "drih":  # 根据当前状态决定是否进入该分支。
        return ("heavy_qm", 0)  # 返回该辅助函数的结果。
    if profile == "polyelectrolyte":  # 根据当前状态决定是否进入该分支。
        return ("charged_polymer_qm", 1)  # 返回该辅助函数的结果。
    if profile == "polymer":  # 根据当前状态决定是否进入该分支。
        return ("polymer_qm", 2)  # 返回该辅助函数的结果。
    return ("standard_qm", 3)  # 返回该辅助函数的结果。


def _retry_threads(current_cores: int) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    current = max(1, int(current_cores))  # 设置中间变量或可调参数，供后续工作流使用。
    if current <= 1:  # 根据当前状态决定是否进入该分支。
        return 1  # 返回该辅助函数的结果。
    return max(1, current // 2)  # 返回该辅助函数的结果。


def _build_parallel_tasks(species, *, cpu_total: int) -> list[ParallelTask]:  # 定义本例内部辅助函数，组织重复步骤。
    tasks: list[ParallelTask] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for spec in species:  # 遍历当前工作流中的一组对象或任务。
        profile = _task_profile(spec)  # 设置中间变量或可调参数，供后续工作流使用。
        batch, priority = _task_batch(profile)  # 设置中间变量或可调参数，供后续工作流使用。
        omp = min(cpu_total, _task_threads(profile=profile, cpu_total=cpu_total))  # 设置每个 rank 的 OpenMP 线程数。
        tasks.append(  # 开始一个多行函数调用或配置块。
            ParallelTask(  # 开始一个多行函数调用或配置块。
                name=spec.name,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                smiles=spec.smiles,  # 设置中间变量或可调参数，供后续工作流使用。
                charge=spec.charge,  # 指定电荷来源或电荷计算方式。
                bonded=spec.bonded,  # 指定特殊 bonded 参数方案，例如 PF6 的 DRIH。
                polyelectrolyte_mode=bool(spec.polyelectrolyte_mode),  # 启用聚电解质处理逻辑。
                profile=profile,  # 设置中间变量或可调参数，供后续工作流使用。
                batch=batch,  # 设置中间变量或可调参数，供后续工作流使用。
                priority=priority,  # 设置中间变量或可调参数，供后续工作流使用。
                required_cores=omp,  # 设置中间变量或可调参数，供后续工作流使用。
                psi4_omp=omp,  # 设置中间变量或可调参数，供后续工作流使用。
            )
        )
    tasks.sort(key=lambda item: (item.priority, -item.required_cores, item.name.lower()))  # 设置中间变量或可调参数，供后续工作流使用。
    return tasks  # 返回该辅助函数的结果。


def _build_pending_payloads(species, tasks: list[ParallelTask]) -> list[dict]:  # 定义本例内部辅助函数，组织重复步骤。
    task_map = {task.name: task for task in tasks}  # 设置中间变量或可调参数，供后续工作流使用。
    pending = []  # 设置中间变量或可调参数，供后续工作流使用。
    for spec in species:  # 遍历当前工作流中的一组对象或任务。
        task = task_map[spec.name]  # 设置中间变量或可调参数，供后续工作流使用。
        pending.append(  # 开始一个多行函数调用或配置块。
            {
                "name": spec.name,
                "smiles": spec.smiles,
                "kind": spec.kind,
                "source": spec.source,
                "charge": spec.charge,
                "bonded": spec.bonded,
                "polyelectrolyte_mode": spec.polyelectrolyte_mode,
                "profile": task.profile,
                "batch": task.batch,
                "priority": task.priority,
                "required_cores": task.required_cores,
                "psi4_omp": task.psi4_omp,
                "attempt": 1,
                "max_attempts": 2,
            }
        )
    pending.sort(  # 开始一个多行函数调用或配置块。
        key=lambda item: (  # 设置中间变量或可调参数，供后续工作流使用。
            int(item["priority"]),
            -int(item["required_cores"]),
            str(item["name"]).lower(),
        )
    )
    return pending  # 返回该辅助函数的结果。


def _sort_pending_in_place(pending: list[dict]) -> None:  # 定义本例内部辅助函数，组织重复步骤。
    pending.sort(  # 开始一个多行函数调用或配置块。
        key=lambda item: (  # 设置中间变量或可调参数，供后续工作流使用。
            int(item["priority"]),
            -int(item["required_cores"]),
            str(item["name"]).lower(),
        )
    )


def _eligible_pending_for_launch(pending: list[dict], available_cores: int) -> list[dict]:  # 定义本例内部辅助函数，组织重复步骤。
    if not pending:  # 根据当前状态决定是否进入该分支。
        return []  # 返回该辅助函数的结果。
    present_priorities = sorted({int(item["priority"]) for item in pending})  # 设置中间变量或可调参数，供后续工作流使用。
    for priority in present_priorities:  # 遍历当前工作流中的一组对象或任务。
        same_priority = [item for item in pending if int(item["priority"]) == priority]
        fitting = [item for item in same_priority if int(item["required_cores"]) <= int(available_cores)]
        if fitting:  # 根据当前状态决定是否进入该分支。
            return fitting  # 返回该辅助函数的结果。
    return [item for item in pending if int(item["required_cores"]) <= int(available_cores)]  # 返回该辅助函数的结果。


def _maybe_schedule_retry(task: dict, *, error: str) -> dict | None:  # 定义本例内部辅助函数，组织重复步骤。
    attempt = int(task.get("attempt", 1))  # 设置中间变量或可调参数，供后续工作流使用。
    max_attempts = int(task.get("max_attempts", 2))  # 设置中间变量或可调参数，供后续工作流使用。
    current_cores = int(task["required_cores"])  # 设置中间变量或可调参数，供后续工作流使用。
    next_cores = _retry_threads(current_cores)  # 设置中间变量或可调参数，供后续工作流使用。
    if attempt >= max_attempts or next_cores >= current_cores:  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。

    retried = dict(task)  # 设置中间变量或可调参数，供后续工作流使用。
    retried["attempt"] = attempt + 1  # 设置中间变量或可调参数，供后续工作流使用。
    retried["required_cores"] = next_cores  # 设置中间变量或可调参数，供后续工作流使用。
    retried["psi4_omp"] = next_cores  # 设置中间变量或可调参数，供后续工作流使用。
    retried["retry_of_cores"] = current_cores  # 设置中间变量或可调参数，供后续工作流使用。
    retried["retry_reason"] = error  # 设置中间变量或可调参数，供后续工作流使用。
    return retried  # 返回该辅助函数的结果。


def _worker_entry(*, task_payload: dict, db_dir: str, job_wd: str, psi4_memory_mb: int, result_queue):  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        os.environ["OMP_NUM_THREADS"] = str(int(task_payload["psi4_omp"]))  # 设置中间变量或可调参数，供后续工作流使用。
        os.environ["YADONPY_OMP_PSI4"] = str(int(task_payload["psi4_omp"]))  # 设置中间变量或可调参数，供后续工作流使用。
        set_run_options(restart=False)  # 设置全局运行选项，例如 restart。
        module = _load_build_module()  # 设置中间变量或可调参数，供后续工作流使用。
        spec = module.SpeciesSpec(  # 设置中间变量或可调参数，供后续工作流使用。
            name=task_payload["name"],  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
            smiles=task_payload["smiles"],  # 设置中间变量或可调参数，供后续工作流使用。
            kind=task_payload["kind"],  # 设置中间变量或可调参数，供后续工作流使用。
            source=task_payload["source"],  # 设置中间变量或可调参数，供后续工作流使用。
            charge=task_payload["charge"],  # 指定电荷来源或电荷计算方式。
            bonded=task_payload["bonded"],  # 指定特殊 bonded 参数方案，例如 PF6 的 DRIH。
            polyelectrolyte_mode=task_payload["polyelectrolyte_mode"],  # 启用聚电解质处理逻辑。
        )
        result = module.run_one_species(  # 设置中间变量或可调参数，供后续工作流使用。
            spec,
            db_dir=Path(db_dir),  # 设置中间变量或可调参数，供后续工作流使用。
            job_wd=Path(job_wd),  # 设置中间变量或可调参数，供后续工作流使用。
            psi4_omp=int(task_payload["psi4_omp"]),  # 设置中间变量或可调参数，供后续工作流使用。
            psi4_memory_mb=int(psi4_memory_mb),  # 设置中间变量或可调参数，供后续工作流使用。
        )
        result_queue.put(  # 开始一个多行函数调用或配置块。
            {
                "name": spec.name,
                "ok": True,
                "result": result,
                "required_cores": int(task_payload["required_cores"]),
                "attempt": int(task_payload.get("attempt", 1)),
            }
        )
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        result_queue.put(  # 开始一个多行函数调用或配置块。
            {
                "name": str(task_payload.get("name")),
                "ok": False,
                "error": repr(exc),
                "required_cores": int(task_payload["required_cores"]),
                "attempt": int(task_payload.get("attempt", 1)),
            }
        )


def main() -> int:  # 定义本例内部辅助函数，组织重复步骤。
    restart_status = False  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
    set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

    psi4_memory_mb = 20000  # 设置中间变量或可调参数，供后续工作流使用。

    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    module = _load_build_module()  # 设置中间变量或可调参数，供后续工作流使用。
    species = module._read_species_csv(module.CATALOG_CSV)  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。
    cpu_total = _available_cpu_total()  # 设置中间变量或可调参数，供后续工作流使用。
    planner_cpu_budget = _planner_cpu_budget(cpu_total)  # 设置中间变量或可调参数，供后续工作流使用。
    tasks = _build_parallel_tasks(species, cpu_total=planner_cpu_budget)  # 设置中间变量或可调参数，供后续工作流使用。

    db = MolDB()  # 设置中间变量或可调参数，供后续工作流使用。
    db_dir = Path(db.db_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    example_wd = workdir(HERE / "work_dir", restart=restart_status)  # 创建或复用本例工作目录。
    job_wd = example_wd.child("02_build_moldb_parallel")  # 设置中间变量或可调参数，供后续工作流使用。

    plan = {  # 设置中间变量或可调参数，供后续工作流使用。
        "catalog_csv": str(module.CATALOG_CSV.resolve()),
        "cpu_total": int(cpu_total),
        "reserved_driver_cores": int(_reserved_driver_cores(cpu_total)),
        "planner_cpu_budget": int(planner_cpu_budget),
        "psi4_memory_mb": int(psi4_memory_mb),
        "tasks": [asdict(task) for task in tasks],
    }
    (Path(job_wd) / "parallel_plan.json").write_text(  # 开始一个多行函数调用或配置块。
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",  # 设置中间变量或可调参数，供后续工作流使用。
        encoding="utf-8",  # 设置中间变量或可调参数，供后续工作流使用。
    )

    print(  # 打印关键路径或状态，便于人工检查。
        f"[PLAN] visible_cpu_total={cpu_total} "
        f"reserved_driver_cores={_reserved_driver_cores(cpu_total)} "
        f"planner_cpu_budget={planner_cpu_budget}"
    )
    for task in tasks:  # 遍历当前工作流中的一组对象或任务。
        print(  # 打印关键路径或状态，便于人工检查。
            f"[PLAN] {task.name:20s} batch={task.batch:18s} profile={task.profile:15s} "
            f"cores={task.required_cores:2d} psi4_omp={task.psi4_omp:2d} bonded={task.bonded or '-'}"
        )

    ctx = mp.get_context("spawn")  # 设置中间变量或可调参数，供后续工作流使用。
    result_queue = ctx.Queue()  # 设置中间变量或可调参数，供后续工作流使用。
    pending = _build_pending_payloads(species, tasks)  # 设置中间变量或可调参数，供后续工作流使用。

    running: dict[str, dict] = {}  # 设置中间变量或可调参数，供后续工作流使用。
    available_cores = int(planner_cpu_budget)  # 设置中间变量或可调参数，供后续工作流使用。
    summary: list[dict] = []  # 设置中间变量或可调参数，供后续工作流使用。
    failures: list[dict] = []  # 设置中间变量或可调参数，供后续工作流使用。
    retry_count = 0  # 设置中间变量或可调参数，供后续工作流使用。

    while pending or running:  # 循环执行直到当前条件不再满足。
        launched = False  # 设置中间变量或可调参数，供后续工作流使用。
        for task in list(_eligible_pending_for_launch(pending, available_cores)):  # 遍历当前工作流中的一组对象或任务。
            required = int(task["required_cores"])  # 设置中间变量或可调参数，供后续工作流使用。
            if required > available_cores:  # 根据当前状态决定是否进入该分支。
                continue
            proc = ctx.Process(  # 设置中间变量或可调参数，供后续工作流使用。
                target=_worker_entry,  # 设置中间变量或可调参数，供后续工作流使用。
                kwargs={  # 设置中间变量或可调参数，供后续工作流使用。
                    "task_payload": task,
                    "db_dir": str(db_dir),
                    "job_wd": str(Path(job_wd)),
                    "psi4_memory_mb": psi4_memory_mb,
                    "result_queue": result_queue,
                },
            )
            proc.start()
            running[str(task["name"])] = {  # 设置中间变量或可调参数，供后续工作流使用。
                "process": proc,
                "required_cores": required,
                "task": task,
                "started_at": time.time(),
            }
            available_cores -= required  # 设置中间变量或可调参数，供后续工作流使用。
            pending.remove(task)
            launched = True  # 设置中间变量或可调参数，供后续工作流使用。
            print(  # 打印关键路径或状态，便于人工检查。
                f"[START] {task['name']:20s} batch={task['batch']:18s} profile={task['profile']:15s} "
                f"attempt={task['attempt']}/{task['max_attempts']} cores={required:2d} remaining={available_cores:2d}"
            )

        if not running and not pending:  # 根据当前状态决定是否进入该分支。
            break

        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            message = result_queue.get(timeout=0.5)  # 设置中间变量或可调参数，供后续工作流使用。
        except queue.Empty:  # 捕获异常并转成更清楚的示例错误信息。
            message = None  # 设置中间变量或可调参数，供后续工作流使用。

        if message is not None:  # 根据当前状态决定是否进入该分支。
            name = str(message["name"])  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
            state = running.pop(name, None)  # 设置中间变量或可调参数，供后续工作流使用。
            if state is not None:  # 根据当前状态决定是否进入该分支。
                state["process"].join(timeout=1.0)  # 设置中间变量或可调参数，供后续工作流使用。
                available_cores += int(state["required_cores"])  # 设置中间变量或可调参数，供后续工作流使用。
            if bool(message.get("ok")):  # 根据当前状态决定是否进入该分支。
                result = dict(message["result"])  # 设置中间变量或可调参数，供后续工作流使用。
                result["attempt"] = int(message.get("attempt", 1))  # 设置中间变量或可调参数，供后续工作流使用。
                summary.append(result)
                print(  # 打印关键路径或状态，便于人工检查。
                    f"[DONE]  {name:20s} released={message['required_cores']:2d} "
                    f"available={available_cores:2d} attempt={message.get('attempt', 1)}"
                )
            else:  # 处理前面条件都不满足的情况。
                retry_task = _maybe_schedule_retry(state["task"], error=str(message.get("error"))) if state is not None else None  # 设置中间变量或可调参数，供后续工作流使用。
                if retry_task is not None:  # 根据当前状态决定是否进入该分支。
                    retry_count += 1  # 设置中间变量或可调参数，供后续工作流使用。
                    pending.append(retry_task)
                    _sort_pending_in_place(pending)
                    print(  # 打印关键路径或状态，便于人工检查。
                        f"[RETRY] {name:20s} failed at {message['required_cores']:2d} cores; "
                        f"retrying with {retry_task['required_cores']:2d} cores "
                        f"(attempt {retry_task['attempt']}/{retry_task['max_attempts']})"
                    )
                else:  # 处理前面条件都不满足的情况。
                    failures.append(  # 开始一个多行函数调用或配置块。
                        {
                            "name": name,
                            "smiles": (state["task"]["smiles"] if state is not None else None),
                            "charge": (state["task"]["charge"] if state is not None else None),
                            "bonded": (state["task"]["bonded"] if state is not None else None),
                            "attempt": int(message.get("attempt", 1)),
                            "error": message.get("error"),
                        }
                    )
                    print(  # 打印关键路径或状态，便于人工检查。
                        f"[FAIL]  {name:20s} released={message['required_cores']:2d} "
                        f"available={available_cores:2d} attempt={message.get('attempt', 1)} :: {message.get('error')}"
                    )

        for name, state in list(running.items()):  # 遍历当前工作流中的一组对象或任务。
            proc = state["process"]  # 设置中间变量或可调参数，供后续工作流使用。
            if proc.is_alive():  # 根据当前状态决定是否进入该分支。
                continue
            proc.join(timeout=0.1)  # 设置中间变量或可调参数，供后续工作流使用。
            running.pop(name, None)
            available_cores += int(state["required_cores"])  # 设置中间变量或可调参数，供后续工作流使用。
            error = f"worker exited without reporting (exitcode={proc.exitcode})"  # 设置中间变量或可调参数，供后续工作流使用。
            retry_task = _maybe_schedule_retry(state["task"], error=error)  # 设置中间变量或可调参数，供后续工作流使用。
            if retry_task is not None:  # 根据当前状态决定是否进入该分支。
                retry_count += 1  # 设置中间变量或可调参数，供后续工作流使用。
                pending.append(retry_task)
                _sort_pending_in_place(pending)
                print(  # 打印关键路径或状态，便于人工检查。
                    f"[RETRY] {name:20s} exited at {state['required_cores']:2d} cores; "
                    f"retrying with {retry_task['required_cores']:2d} cores "
                    f"(attempt {retry_task['attempt']}/{retry_task['max_attempts']})"
                )
            else:  # 处理前面条件都不满足的情况。
                failures.append(  # 开始一个多行函数调用或配置块。
                    {
                        "name": name,
                        "smiles": state["task"]["smiles"],
                        "charge": state["task"]["charge"],
                        "bonded": state["task"]["bonded"],
                        "attempt": int(state["task"].get("attempt", 1)),
                        "error": error,
                    }
                )
                print(  # 打印关键路径或状态，便于人工检查。
                    f"[FAIL]  {name:20s} released={state['required_cores']:2d} available={available_cores:2d} :: "
                    f"{error}"
                )

        if not launched and running:  # 根据当前状态决定是否进入该分支。
            time.sleep(0.1)

    out = {  # 设置中间变量或可调参数，供后续工作流使用。
        "catalog_csv": str(module.CATALOG_CSV.resolve()),
        "db_dir": str(db_dir.resolve()),
        "work_root": str(Path(job_wd).resolve()),
        "cpu_total": int(cpu_total),
        "reserved_driver_cores": int(_reserved_driver_cores(cpu_total)),
        "planner_cpu_budget": int(planner_cpu_budget),
        "psi4_memory_mb": int(psi4_memory_mb),
        "success_count": len(summary),
        "failure_count": len(failures),
        "retry_count": int(retry_count),
        "success": summary,
        "failures": failures,
    }
    (Path(job_wd) / "parallel_build_summary.json").write_text(  # 开始一个多行函数调用或配置块。
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",  # 设置中间变量或可调参数，供后续工作流使用。
        encoding="utf-8",  # 设置中间变量或可调参数，供后续工作流使用。
    )

    print(f"\nMolDB directory: {db_dir}")  # 打印关键路径或状态，便于人工检查。
    print(f"Catalog CSV   : {module.CATALOG_CSV}")  # 打印关键路径或状态，便于人工检查。
    print(f"Visible cores : {cpu_total}")  # 打印关键路径或状态，便于人工检查。
    print(f"Plan budget   : {planner_cpu_budget}")  # 打印关键路径或状态，便于人工检查。
    print(f"Retries       : {retry_count}")  # 打印关键路径或状态，便于人工检查。
    print(f"Success       : {len(summary)}")  # 打印关键路径或状态，便于人工检查。
    print(f"Failures      : {len(failures)}")  # 打印关键路径或状态，便于人工检查。
    return 0 if not failures else 1  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    raise SystemExit(main())  # 关键步骤失败时立即报错，避免继续生成错误结果。
