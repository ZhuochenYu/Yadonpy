from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Refresh CMC anionic glucose monomers in both default and repo MolDB.

Targets:
  - glucose_2
  - glucose_3
  - glucose_6

All three species are recomputed with RESP charges under
``polyelectrolyte_mode=True`` and then written to:
  1. the user's default MolDB (typically ``~/.yadonpy/moldb``)
  2. the repository-tracked ``moldb/`` directory
"""

from dataclasses import dataclass  # 导入本例需要的库或 yadonpy 接口。
import os  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import assign_charges, mol_from_smiles  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.polyelectrolyte import detect_charged_groups  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.moldb import MolDB  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


HERE = Path(__file__).resolve().parent  # 定位当前脚本所在目录。
REPO_ROOT = HERE.parents[1]  # 定位仓库根目录。


@dataclass(frozen=True)  # 声明轻量数据类，用于保存配置或任务信息。
class SpeciesSpec:  # 定义本例内部数据结构或配置对象。
    name: str
    smiles: str


SPECIES: tuple[SpeciesSpec, ...] = (  # 设置中间变量或可调参数，供后续工作流使用。
    SpeciesSpec("glucose_2", "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"),  # 设置中间变量或可调参数，供后续工作流使用。
    SpeciesSpec("glucose_3", "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"),  # 设置中间变量或可调参数，供后续工作流使用。
    SpeciesSpec("glucose_6", "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"),  # 设置中间变量或可调参数，供后续工作流使用。
)


def _env_int(name: str, default: int) -> int:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    if not raw:  # 根据当前状态决定是否进入该分支。
        return int(default)  # 返回该辅助函数的结果。
    return int(raw)  # 返回该辅助函数的结果。


def _env_tokens(name: str) -> set[str]:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    if not raw:  # 根据当前状态决定是否进入该分支。
        return set()  # 返回该辅助函数的结果。
    return {tok.strip() for tok in raw.split(",") if tok.strip()}  # 返回该辅助函数的结果。


def _env_flag(name: str, default: bool = False) -> bool:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if not raw:  # 根据当前状态决定是否进入该分支。
        return bool(default)  # 返回该辅助函数的结果。
    return raw in {"1", "true", "t", "yes", "y", "on"}  # 返回该辅助函数的结果。


def _env_str(name: str, default: str) -> str:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    return raw or str(default)  # 返回该辅助函数的结果。


def _formal_charge(mol) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))  # 返回该辅助函数的结果。


def _is_ready(db: MolDB, smiles: str) -> bool:  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        db.load_mol(  # 开始一个多行函数调用或配置块。
            smiles,
            require_ready=True,  # 要求 MolDB 物种必须已准备好。
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
            polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
        )
        return True  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return False  # 返回该辅助函数的结果。


def _refresh_one(spec: SpeciesSpec, *, default_db: MolDB, repo_db: MolDB, job_wd: Path) -> None:  # 定义本例内部辅助函数，组织重复步骤。
    species_wd = workdir(job_wd / spec.name, restart=False)  # 创建或复用本例工作目录。
    source = "fresh-smiles"  # 设置中间变量或可调参数，供后续工作流使用。
    need_opt = True  # 设置中间变量或可调参数，供后续工作流使用。
    mol = None  # 设置中间变量或可调参数，供后续工作流使用。
    for db, label in ((default_db, "default"), (repo_db, "repo")):  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            mol, _rec = db.load_mol(  # 设置中间变量或可调参数，供后续工作流使用。
                spec.smiles,
                require_ready=False,  # 要求 MolDB 物种必须已准备好。
                charge="RESP",  # 指定电荷来源或电荷计算方式。
                polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
                polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
            )
            source = f"{label}-db-geometry"  # 设置中间变量或可调参数，供后续工作流使用。
            need_opt = False  # 设置中间变量或可调参数，供后续工作流使用。
            break
        except Exception:  # 捕获异常并转成更清楚的示例错误信息。
            continue

    if mol is None:  # 根据当前状态决定是否进入该分支。
        mol = mol_from_smiles(spec.smiles, name=spec.name)  # 设置中间变量或可调参数，供后续工作流使用。

    charge_groups = detect_charged_groups(mol, detection="auto")  # 设置中间变量或可调参数，供后续工作流使用。
    formal_charge = _formal_charge(mol)  # 设置中间变量或可调参数，供后续工作流使用。

    print(  # 打印关键路径或状态，便于人工检查。
        f"[RUN] {spec.name:10s} formal_charge={formal_charge:+d} "
        f"localized_groups={len(charge_groups.get('groups') or [])} "
        f"source={source} opt={need_opt}"
    )

    ok = assign_charges(  # 执行电荷分配流程。
        mol,
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        opt=need_opt,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=species_wd,  # 设置本例输出目录。
        log_name=spec.name,  # 设置中间变量或可调参数，供后续工作流使用。
        omp=_env_int("YADONPY_PSI4_OMP", 20),  # 设置每个 rank 的 OpenMP 线程数。
        memory=_env_int("YADONPY_PSI4_MEMORY_MB", 20000),  # 设置中间变量或可调参数，供后续工作流使用。
        opt_method=_env_str("YADONPY_OPT_METHOD", "wb97m-d3bj"),  # 设置中间变量或可调参数，供后续工作流使用。
        charge_method=_env_str("YADONPY_CHARGE_METHOD", "wb97m-d3bj"),  # 设置中间变量或可调参数，供后续工作流使用。
        opt_basis=_env_str("YADONPY_OPT_BASIS", "def2-SVPD"),  # 设置中间变量或可调参数，供后续工作流使用。
        charge_basis=_env_str("YADONPY_CHARGE_BASIS", "def2-TZVPD"),  # 设置中间变量或可调参数，供后续工作流使用。
        total_charge=formal_charge,  # 设置中间变量或可调参数，供后续工作流使用。
        total_multiplicity=1,  # 设置中间变量或可调参数，供后续工作流使用。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
        bonded_params="ff_assigned",  # 设置中间变量或可调参数，供后续工作流使用。
    )
    if not ok:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"assign_charges failed for {spec.name}")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    for db, label in ((default_db, "default"), (repo_db, "repo")):  # 遍历当前工作流中的一组对象或任务。
        rec = db.update_from_mol(  # 设置中间变量或可调参数，供后续工作流使用。
            mol,
            smiles_or_psmiles=spec.smiles,  # 设置中间变量或可调参数，供后续工作流使用。
            name=spec.name,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
            polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
        )
        loaded, loaded_rec = db.load_mol(  # 设置中间变量或可调参数，供后续工作流使用。
            spec.smiles,
            require_ready=True,  # 要求 MolDB 物种必须已准备好。
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
            polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
        )
        print(  # 打印关键路径或状态，便于人工检查。
            f"  [OK:{label}] key={rec.key} atoms={loaded.GetNumAtoms()} "
            f"ready_name={loaded_rec.name}"
        )


def main() -> int:  # 定义本例内部辅助函数，组织重复步骤。
    set_run_options(restart=False)  # 设置全局运行选项，例如 restart。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    default_db = MolDB()  # 设置中间变量或可调参数，供后续工作流使用。
    repo_db = MolDB(REPO_ROOT / "moldb")  # 设置中间变量或可调参数，供后续工作流使用。
    job_wd = workdir(HERE / "work_dir" / "03_refresh_cmc_glucose_polyelectrolytes", restart=False)  # 创建或复用本例工作目录。

    print(f"[DB] default = {default_db.db_dir}")  # 打印关键路径或状态，便于人工检查。
    print(f"[DB] repo    = {repo_db.db_dir}")  # 打印关键路径或状态，便于人工检查。
    print(  # 打印关键路径或状态，便于人工检查。
        "[QM] psi4_omp="
        f"{_env_int('YADONPY_PSI4_OMP', 20)} "
        f"psi4_memory_mb={_env_int('YADONPY_PSI4_MEMORY_MB', 20000)} "
        f"opt={_env_str('YADONPY_OPT_METHOD', 'wb97m-d3bj')}/"
        f"{_env_str('YADONPY_OPT_BASIS', 'def2-SVPD')} "
        f"charge={_env_str('YADONPY_CHARGE_METHOD', 'wb97m-d3bj')}/"
        f"{_env_str('YADONPY_CHARGE_BASIS', 'def2-TZVPD')}"
    )

    only = _env_tokens("YADONPY_ONLY")  # 设置中间变量或可调参数，供后续工作流使用。
    force = _env_flag("YADONPY_FORCE", default=False)  # 设置中间变量或可调参数，供后续工作流使用。

    for spec in SPECIES:  # 遍历当前工作流中的一组对象或任务。
        if only and spec.name not in only:  # 根据当前状态决定是否进入该分支。
            print(f"[SKIP] {spec.name:10s} filtered by YADONPY_ONLY")  # 打印关键路径或状态，便于人工检查。
            continue
        default_ready = _is_ready(default_db, spec.smiles)  # 设置中间变量或可调参数，供后续工作流使用。
        repo_ready = _is_ready(repo_db, spec.smiles)  # 设置中间变量或可调参数，供后续工作流使用。
        if default_ready and repo_ready and not force:  # 根据当前状态决定是否进入该分支。
            print(f"[SKIP] {spec.name:10s} already ready in both MolDB locations")  # 打印关键路径或状态，便于人工检查。
            continue
        _refresh_one(spec, default_db=default_db, repo_db=repo_db, job_wd=Path(job_wd))  # 设置中间变量或可调参数，供后续工作流使用。

    print("[DONE] Refreshed glucose_2 / glucose_3 / glucose_6 in both MolDB locations.")  # 打印关键路径或状态，便于人工检查。
    return 0  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    raise SystemExit(main())  # 关键步骤失败时立即报错，避免继续生成错误结果。
