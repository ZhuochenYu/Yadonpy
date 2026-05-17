from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

import importlib.util  # 导入本例需要的库或 yadonpy 接口。
import json  # 导入本例需要的库或 yadonpy 接口。
import os  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.core import utils  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff import GAFF2  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim import qm  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.benchmarking import summarize_rdkit_species_forcefield  # 导入本例需要的库或 yadonpy 接口。


def _env_bool(name: str, default: bool) -> bool:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None:  # 根据当前状态决定是否进入该分支。
        return bool(default)  # 返回该辅助函数的结果。
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}  # 返回该辅助函数的结果。


def _env_int(name: str, default: int) -> int:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None or not str(raw).strip():  # 根据当前状态决定是否进入该分支。
        return int(default)  # 返回该辅助函数的结果。
    return int(raw)  # 返回该辅助函数的结果。


def _env_text(name: str, default: str) -> str:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None:  # 根据当前状态决定是否进入该分支。
        return str(default)  # 返回该辅助函数的结果。
    text = str(raw).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    return text if text else str(default)  # 返回该辅助函数的结果。


def _normalize_geometry_source(raw: str | None) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    mode = str(raw or "qm").strip().lower()  # 设置该配置块使用的计算模式。
    if mode in {"db", "moldb", "existing"}:  # 根据当前状态决定是否进入该分支。
        return "moldb"  # 返回该辅助函数的结果。
    if mode != "qm":  # 根据当前状态决定是否进入该分支。
        raise ValueError(f"Unsupported GEOMETRY_SOURCE={raw!r}; expected qm/moldb.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return mode  # 返回该辅助函数的结果。


def _load_benchmark_module():  # 定义本例内部辅助函数，组织重复步骤。
    script_path = Path(__file__).with_name("benchmark_carbonate_lipf6_gaff2.py")  # 设置中间变量或可调参数，供后续工作流使用。
    spec = importlib.util.spec_from_file_location("benchmark_carbonate_lipf6_gaff2", script_path)  # 设置中间变量或可调参数，供后续工作流使用。
    if spec is None or spec.loader is None:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"Cannot load benchmark helper module from {script_path}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    module = importlib.util.module_from_spec(spec)  # 设置中间变量或可调参数，供后续工作流使用。
    spec.loader.exec_module(module)
    return module  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    restart_status = _env_bool("RESTART_STATUS", False)  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
    set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

    bench = _load_benchmark_module()  # 设置中间变量或可调参数，供后续工作流使用。

    target = _env_text("TARGET", "DEC").strip().upper()  # 设置中间变量或可调参数，供后续工作流使用。
    if target not in {"EC", "EMC", "DEC"}:  # 根据当前状态决定是否进入该分支。
        raise ValueError(f"Unsupported TARGET={target!r}; expected EC/EMC/DEC.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    target_smiles = {  # 设置中间变量或可调参数，供后续工作流使用。
        "EC": bench.EC_SMILES,
        "EMC": bench.EMC_SMILES,
        "DEC": bench.DEC_SMILES,
    }[target]

    resp_profile = bench._normalize_resp_profile(os.environ.get("YADONPY_RESP_PROFILE"))  # 设置中间变量或可调参数，供后续工作流使用。
    charge_recipe = bench._charge_recipe_from_family(os.environ.get("YADONPY_CHARGE_DFT_FAMILY"))  # 设置中间变量或可调参数，供后续工作流使用。
    cache_to_repo_db = _env_bool("CACHE_TO_REPO_DB", True)  # 设置中间变量或可调参数，供后续工作流使用。
    geometry_source = _normalize_geometry_source(os.environ.get("GEOMETRY_SOURCE"))  # 设置中间变量或可调参数，供后续工作流使用。

    psi4_omp = _env_int("PSI4_OMP", 8)  # 设置中间变量或可调参数，供后续工作流使用。
    mpi = _env_int("MPI", 1)  # 设置 GROMACS MPI/thread-MPI rank 数。
    omp = _env_int("OMP", 1)  # 设置每个 rank 的 OpenMP 线程数。
    memory_mb = _env_int("MEM_MB", 20000)  # 设置中间变量或可调参数，供后续工作流使用。

    work_dir_name = _env_text("WORK_DIR_NAME", f"probe_single_{target.lower()}_gaff2")  # 设置中间变量或可调参数，供后续工作流使用。
    work_root = workdir(Path(_env_text("WORK_DIR", str(Path(__file__).resolve().parent / work_dir_name))).resolve(), restart=restart_status)  # 创建或复用本例工作目录。

    ff = GAFF2()  # 选择有机分子/聚合物/部分无机离子的力场对象。
    if geometry_source == "moldb":  # 根据当前状态决定是否进入该分支。
        mol = ff.mol_rdkit(  # 设置中间变量或可调参数，供后续工作流使用。
            target_smiles,
            name=target,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
            prefer_db=True,  # 优先从 MolDB 读取已有结果。
            require_ready=False,  # 要求 MolDB 物种必须已准备好。
        )
        log_name = f"{target.lower()}_{charge_recipe['family']}_{ff.name}_refit"  # 设置中间变量或可调参数，供后续工作流使用。
        qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
            mol,
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            opt=False,  # 设置中间变量或可调参数，供后续工作流使用。
            work_dir=work_root,  # 设置本例输出目录。
            log_name=log_name,  # 设置中间变量或可调参数，供后续工作流使用。
            omp=psi4_omp,  # 设置每个 rank 的 OpenMP 线程数。
            memory=memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
            charge_method=charge_recipe["charge_method"],  # 设置中间变量或可调参数，供后续工作流使用。
            charge_basis=charge_recipe["charge_basis"],  # 设置中间变量或可调参数，供后续工作流使用。
            charge_basis_gen=charge_recipe["charge_basis_gen"],  # 设置中间变量或可调参数，供后续工作流使用。
            resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        mol = ff.ff_assign(mol, charge=None, report=False)  # 分配力场参数并写入分子属性。
        if not mol:  # 根据当前状态决定是否进入该分支。
            raise RuntimeError(f"Cannot assign {ff.name} parameters for {target} after RESP refit.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
        if cache_to_repo_db:  # 根据当前状态决定是否进入该分支。
            ff.store_to_db(  # 开始一个多行函数调用或配置块。
                mol,
                smiles_or_psmiles=target_smiles,  # 设置中间变量或可调参数，供后续工作流使用。
                name=target,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                db_dir=bench.REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
                charge="RESP",  # 指定电荷来源或电荷计算方式。
                basis_set=charge_recipe["charge_basis"],  # 设置中间变量或可调参数，供后续工作流使用。
                method=charge_recipe["charge_method"],  # 设置中间变量或可调参数，供后续工作流使用。
            )
            print(f"[MolDB] stored refit {target} RESP entry into repo db: {bench.REPO_DB_DIR}")  # 打印关键路径或状态，便于人工检查。
    else:  # 处理前面条件都不满足的情况。
        mol = bench._build_qm_ready_gaff_species(  # 设置中间变量或可调参数，供后续工作流使用。
            ff,
            target_smiles,
            label=target,  # 给该选区一个可读标签，便于 manifest 检查。
            recipe=charge_recipe,  # 设置中间变量或可调参数，供后续工作流使用。
            resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
            work_root=work_root,  # 设置中间变量或可调参数，供后续工作流使用。
            psi4_omp=psi4_omp,  # 设置中间变量或可调参数，供后续工作流使用。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            memory_mb=memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
            repo_db_dir=bench.REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
            cache_to_repo_db=cache_to_repo_db,  # 设置中间变量或可调参数，供后续工作流使用。
        )

    analysis_dir = work_root / "06_analysis"  # 设置中间变量或可调参数，供后续工作流使用。
    analysis_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    summary = {  # 设置中间变量或可调参数，供后续工作流使用。
        "metadata": {
            "target": target,
            "resp_profile": resp_profile,
            "qm_charge_recipe": charge_recipe,
            "cache_to_repo_db": cache_to_repo_db,
            "geometry_source": geometry_source,
            "repo_db_dir": str(bench.REPO_DB_DIR),
        },
        "resp_route": bench._extract_resp_route(mol, label=target),
        "charge_sanity": bench._summarize_carbonate_charge_features(mol, label=target),
        "equivalence_spread": bench._equivalence_spread_diagnostic(mol, label=target),
        "forcefield_summary": summarize_rdkit_species_forcefield(mol, label=target, moltype_hint=target, charge_scale=1.0),
    }
    out = analysis_dir / "probe_summary.json"  # 设置中间变量或可调参数，供后续工作流使用。
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
    print(json.dumps(summary, indent=2, ensure_ascii=False))  # 打印关键路径或状态，便于人工检查。
