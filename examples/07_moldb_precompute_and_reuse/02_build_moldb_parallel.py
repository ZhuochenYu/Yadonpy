from __future__ import annotations

"""Example 07 / Step 2: Auto-planned parallel MolDB build."""

import importlib.util
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from yadonpy.core import workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.moldb import MolDB
from yadonpy.runtime import set_run_options


HERE = Path(__file__).resolve().parent
BUILD_SCRIPT = HERE / "01_build_moldb.py"


@dataclass(frozen=True)
class ParallelTask:
    name: str
    smiles: str
    ff_name: str
    bonded: str | None
    polyelectrolyte_mode: bool
    profile: str
    required_cores: int
    psi4_omp: int


def _load_build_module():
    spec = importlib.util.spec_from_file_location("example07_build_moldb", BUILD_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _available_cpu_total() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, int(os.cpu_count() or 1))


def _reserved_driver_cores(cpu_total: int) -> int:
    if cpu_total >= 32:
        return 2
    if cpu_total >= 8:
        return 1
    return 0


def _planner_cpu_budget(cpu_total: int) -> int:
    return max(1, int(cpu_total) - _reserved_driver_cores(int(cpu_total)))


def _task_profile(spec) -> str:
    if str(spec.ff_name).strip().lower() == "merz":
        return "light"
    if str(spec.bonded or "").strip().upper() == "DRIH":
        return "drih"
    if bool(spec.polyelectrolyte_mode):
        return "polyelectrolyte"
    if "*" in str(spec.smiles):
        return "polymer"
    return "standard"


def _task_threads(*, profile: str, cpu_total: int) -> int:
    if profile == "light":
        return 1
    if cpu_total <= 8:
        return max(1, cpu_total)
    if profile == "drih":
        return min(8, max(4, cpu_total // 4))
    if profile == "polyelectrolyte":
        return min(8, max(4, cpu_total // 5))
    if profile == "polymer":
        return min(6, max(3, cpu_total // 6))
    return min(4, max(2, cpu_total // 8))


def _build_parallel_tasks(species, *, cpu_total: int) -> list[ParallelTask]:
    tasks: list[ParallelTask] = []
    for spec in species:
        profile = _task_profile(spec)
        omp = min(cpu_total, _task_threads(profile=profile, cpu_total=cpu_total))
        tasks.append(
            ParallelTask(
                name=spec.name,
                smiles=spec.smiles,
                ff_name=spec.ff_name,
                bonded=spec.bonded,
                polyelectrolyte_mode=bool(spec.polyelectrolyte_mode),
                profile=profile,
                required_cores=omp,
                psi4_omp=omp,
            )
        )
    tasks.sort(key=lambda item: (-item.required_cores, item.name.lower()))
    return tasks


def _build_pending_payloads(species, tasks: list[ParallelTask]) -> list[dict]:
    task_map = {task.name: task for task in tasks}
    pending = []
    for spec in species:
        task = task_map[spec.name]
        pending.append(
            {
                "name": spec.name,
                "smiles": spec.smiles,
                "kind": spec.kind,
                "source": spec.source,
                "ff_name": spec.ff_name,
                "charge": spec.charge,
                "bonded": spec.bonded,
                "polyelectrolyte_mode": spec.polyelectrolyte_mode,
                "profile": task.profile,
                "required_cores": task.required_cores,
                "psi4_omp": task.psi4_omp,
            }
        )
    pending.sort(key=lambda item: (-int(item["required_cores"]), str(item["name"]).lower()))
    return pending


def _worker_entry(*, task_payload: dict, db_dir: str, job_wd: str, psi4_memory_mb: int, result_queue):
    try:
        os.environ["OMP_NUM_THREADS"] = str(int(task_payload["psi4_omp"]))
        os.environ["YADONPY_OMP_PSI4"] = str(int(task_payload["psi4_omp"]))
        set_run_options(restart=False)
        module = _load_build_module()
        spec = module.SpeciesSpec(
            name=task_payload["name"],
            smiles=task_payload["smiles"],
            kind=task_payload["kind"],
            source=task_payload["source"],
            ff_name=task_payload["ff_name"],
            charge=task_payload["charge"],
            bonded=task_payload["bonded"],
            polyelectrolyte_mode=task_payload["polyelectrolyte_mode"],
        )
        result = module.run_one_species(
            spec,
            db_dir=Path(db_dir),
            job_wd=Path(job_wd),
            psi4_omp=int(task_payload["psi4_omp"]),
            psi4_memory_mb=int(psi4_memory_mb),
        )
        result_queue.put(
            {
                "name": spec.name,
                "ok": True,
                "result": result,
                "required_cores": int(task_payload["required_cores"]),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "name": str(task_payload.get("name")),
                "ok": False,
                "error": repr(exc),
                "required_cores": int(task_payload["required_cores"]),
            }
        )


def main() -> int:
    restart_status = False
    set_run_options(restart=restart_status)

    psi4_memory_mb = 20000

    doctor(print_report=True)
    ensure_initialized()

    module = _load_build_module()
    species = module._read_species_csv(module.CATALOG_CSV)
    cpu_total = _available_cpu_total()
    planner_cpu_budget = _planner_cpu_budget(cpu_total)
    tasks = _build_parallel_tasks(species, cpu_total=planner_cpu_budget)

    db = MolDB()
    db_dir = Path(db.db_dir)
    example_wd = workdir(HERE / "work_dir", restart=restart_status)
    job_wd = example_wd.child("02_build_moldb_parallel")

    plan = {
        "catalog_csv": str(module.CATALOG_CSV.resolve()),
        "cpu_total": int(cpu_total),
        "reserved_driver_cores": int(_reserved_driver_cores(cpu_total)),
        "planner_cpu_budget": int(planner_cpu_budget),
        "psi4_memory_mb": int(psi4_memory_mb),
        "tasks": [asdict(task) for task in tasks],
    }
    (Path(job_wd) / "parallel_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"[PLAN] visible_cpu_total={cpu_total} "
        f"reserved_driver_cores={_reserved_driver_cores(cpu_total)} "
        f"planner_cpu_budget={planner_cpu_budget}"
    )
    for task in tasks:
        print(
            f"[PLAN] {task.name:20s} profile={task.profile:15s} "
            f"cores={task.required_cores:2d} psi4_omp={task.psi4_omp:2d} bonded={task.bonded or '-'}"
        )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    pending = _build_pending_payloads(species, tasks)

    running: dict[str, dict] = {}
    available_cores = int(planner_cpu_budget)
    summary: list[dict] = []
    failures: list[dict] = []

    while pending or running:
        launched = False
        for task in list(pending):
            required = int(task["required_cores"])
            if required > available_cores:
                continue
            proc = ctx.Process(
                target=_worker_entry,
                kwargs={
                    "task_payload": task,
                    "db_dir": str(db_dir),
                    "job_wd": str(Path(job_wd)),
                    "psi4_memory_mb": psi4_memory_mb,
                    "result_queue": result_queue,
                },
            )
            proc.start()
            running[str(task["name"])] = {
                "process": proc,
                "required_cores": required,
                "task": task,
                "started_at": time.time(),
            }
            available_cores -= required
            pending.remove(task)
            launched = True
            print(
                f"[START] {task['name']:20s} profile={task['profile']:15s} "
                f"cores={required:2d} remaining={available_cores:2d}"
            )

        if not running and not pending:
            break

        try:
            message = result_queue.get(timeout=0.5)
        except queue.Empty:
            message = None

        if message is not None:
            name = str(message["name"])
            state = running.pop(name, None)
            if state is not None:
                state["process"].join(timeout=1.0)
                available_cores += int(state["required_cores"])
            if bool(message.get("ok")):
                summary.append(message["result"])
                print(f"[DONE]  {name:20s} released={message['required_cores']:2d} available={available_cores:2d}")
            else:
                failures.append(
                    {
                        "name": name,
                        "smiles": (state["task"]["smiles"] if state is not None else None),
                        "ff_name": (state["task"]["ff_name"] if state is not None else None),
                        "charge": (state["task"]["charge"] if state is not None else None),
                        "bonded": (state["task"]["bonded"] if state is not None else None),
                        "error": message.get("error"),
                    }
                )
                print(f"[FAIL]  {name:20s} released={message['required_cores']:2d} available={available_cores:2d} :: {message.get('error')}")

        for name, state in list(running.items()):
            proc = state["process"]
            if proc.is_alive():
                continue
            proc.join(timeout=0.1)
            running.pop(name, None)
            available_cores += int(state["required_cores"])
            failures.append(
                {
                    "name": name,
                    "smiles": state["task"]["smiles"],
                    "ff_name": state["task"]["ff_name"],
                    "charge": state["task"]["charge"],
                    "bonded": state["task"]["bonded"],
                    "error": f"worker exited without reporting (exitcode={proc.exitcode})",
                }
            )
            print(
                f"[FAIL]  {name:20s} released={state['required_cores']:2d} available={available_cores:2d} :: "
                f"worker exited without reporting (exitcode={proc.exitcode})"
            )

        if not launched and running:
            time.sleep(0.1)

    out = {
        "catalog_csv": str(module.CATALOG_CSV.resolve()),
        "db_dir": str(db_dir.resolve()),
        "work_root": str(Path(job_wd).resolve()),
        "cpu_total": int(cpu_total),
        "reserved_driver_cores": int(_reserved_driver_cores(cpu_total)),
        "planner_cpu_budget": int(planner_cpu_budget),
        "psi4_memory_mb": int(psi4_memory_mb),
        "success_count": len(summary),
        "failure_count": len(failures),
        "success": summary,
        "failures": failures,
    }
    (Path(job_wd) / "parallel_build_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nMolDB directory: {db_dir}")
    print(f"Catalog CSV   : {module.CATALOG_CSV}")
    print(f"Visible cores : {cpu_total}")
    print(f"Plan budget   : {planner_cpu_budget}")
    print(f"Success       : {len(summary)}")
    print(f"Failures      : {len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
