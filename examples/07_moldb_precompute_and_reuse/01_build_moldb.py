from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 07 / Step 1: One-shot MolDB build for common electrolyte species."""

import csv  # 导入本例需要的库或 yadonpy 接口。
import json  # 导入本例需要的库或 yadonpy 接口。
from dataclasses import dataclass  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。
from typing import Any  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import assign_charges, mol_from_smiles  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.polyelectrolyte import detect_charged_groups, uses_localized_charge_groups  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.moldb import MolDB  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.qm import _pick_first_available_basis  # 导入本例需要的库或 yadonpy 接口。


HERE = Path(__file__).resolve().parent  # 定位当前脚本所在目录。
CATALOG_CSV = HERE / "electrolyte_species.csv"  # 指定示例使用的物种目录表。


@dataclass(frozen=True)  # 声明轻量数据类，用于保存配置或任务信息。
class SpeciesSpec:  # 定义本例内部数据结构或配置对象。
    name: str
    smiles: str
    kind: str
    source: str
    charge: str
    bonded: str | None
    polyelectrolyte_mode: bool


@dataclass(frozen=True)  # 声明轻量数据类，用于保存配置或任务信息。
class QMSpec:  # 定义本例内部数据结构或配置对象。
    method: str
    opt_basis: str
    charge_basis: str
    reason: str


def _csv_bool(value: object, *, default: bool = False) -> bool:  # 定义本例内部辅助函数，组织重复步骤。
    token = str(value or "").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if not token:  # 根据当前状态决定是否进入该分支。
        return bool(default)  # 返回该辅助函数的结果。
    return token in {"1", "true", "t", "yes", "y", "on"}  # 返回该辅助函数的结果。


def _read_species_csv(path: Path) -> list[SpeciesSpec]:  # 定义本例内部辅助函数，组织重复步骤。
    items: list[SpeciesSpec] = []  # 设置中间变量或可调参数，供后续工作流使用。
    seen: set[str] = set()  # 设置中间变量或可调参数，供后续工作流使用。
    with path.open("r", encoding="utf-8", newline="") as fh:  # 用上下文管理器安全打开文件或管理资源。
        reader = csv.DictReader(fh)  # 设置中间变量或可调参数，供后续工作流使用。
        for raw in reader:  # 遍历当前工作流中的一组对象或任务。
            name = str(raw.get("name") or "").strip()  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
            smiles = str(raw.get("smiles") or "").strip()  # 设置中间变量或可调参数，供后续工作流使用。
            if not name or not smiles or smiles in seen:  # 根据当前状态决定是否进入该分支。
                continue
            seen.add(smiles)
            requested_polyelectrolyte_mode = _csv_bool(raw.get("polyelectrolyte_mode"), default=False)  # 设置中间变量或可调参数，供后续工作流使用。
            items.append(  # 开始一个多行函数调用或配置块。
                SpeciesSpec(  # 开始一个多行函数调用或配置块。
                    name=name,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                    smiles=smiles,  # 设置中间变量或可调参数，供后续工作流使用。
                    kind=str(raw.get("kind") or ("psmiles" if "*" in smiles else "smiles")).strip(),  # 设置中间变量或可调参数，供后续工作流使用。
                    source=str(raw.get("source") or "example07").strip(),  # 设置中间变量或可调参数，供后续工作流使用。
                    charge=str(raw.get("charge") or "RESP").strip().upper(),  # 指定电荷来源或电荷计算方式。
                    bonded=(str(raw.get("bonded") or "").strip() or None),  # 指定特殊 bonded 参数方案，例如 PF6 的 DRIH。
                    polyelectrolyte_mode=_resolve_polyelectrolyte_mode(smiles, requested=requested_polyelectrolyte_mode),  # 启用聚电解质处理逻辑。
                )
            )
    return items  # 返回该辅助函数的结果。


def _resolve_polyelectrolyte_mode(smiles: str, *, requested: bool = False) -> bool:  # 定义本例内部辅助函数，组织重复步骤。
    if bool(requested):  # 根据当前状态决定是否进入该分支。
        return True  # 返回该辅助函数的结果。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        mol = mol_from_smiles(smiles, coord=False)  # 设置中间变量或可调参数，供后续工作流使用。
        summary = detect_charged_groups(mol, detection="auto")  # 设置中间变量或可调参数，供后续工作流使用。
        return bool(uses_localized_charge_groups(summary))  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return False  # 返回该辅助函数的结果。


def _bonded_mode(spec: SpeciesSpec) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    token = str(spec.bonded or "").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if token in {"drih", "drih-like", "dri"}:  # 根据当前状态决定是否进入该分支。
        return "drih"  # 返回该辅助函数的结果。
    return "ff_assigned"  # 返回该辅助函数的结果。


def _resolve_qm_spec(smiles: str) -> QMSpec | None:  # 定义本例内部辅助函数，组织重复步骤。
    mol = mol_from_smiles(smiles, coord=False)  # 设置中间变量或可调参数，供后续工作流使用。
    elements: list[str] = []  # 设置中间变量或可调参数，供后续工作流使用。
    seen: set[str] = set()  # 设置中间变量或可调参数，供后续工作流使用。
    formal_charge = 0  # 设置中间变量或可调参数，供后续工作流使用。
    for atom in mol.GetAtoms():  # 遍历当前工作流中的一组对象或任务。
        formal_charge += int(atom.GetFormalCharge())  # 设置中间变量或可调参数，供后续工作流使用。
        symbol = str(atom.GetSymbol()).strip()  # 设置中间变量或可调参数，供后续工作流使用。
        if not symbol or symbol == "*" or symbol in seen:  # 根据当前状态决定是否进入该分支。
            continue
        seen.add(symbol)
        elements.append(symbol)

    if mol.GetNumAtoms() == 1 and formal_charge != 0:  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。

    method = "wb97m-d3bj"  # 设置中间变量或可调参数，供后续工作流使用。
    if formal_charge < 0:  # 根据当前状态决定是否进入该分支。
        opt_candidates = ["def2-SVPD", "def2-SVP"]  # 设置中间变量或可调参数，供后续工作流使用。
        charge_candidates = ["def2-TZVPD", "def2-TZVPPD", "def2-TZVP"]  # 设置中间变量或可调参数，供后续工作流使用。
        reason = "anion diffuse-first"  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        opt_candidates = ["def2-SVP"]  # 设置中间变量或可调参数，供后续工作流使用。
        charge_candidates = ["def2-TZVP"]  # 设置中间变量或可调参数，供后续工作流使用。
        reason = "neutral default"  # 设置中间变量或可调参数，供后续工作流使用。

    opt_basis = _pick_first_available_basis(opt_candidates, elements=elements)  # 设置中间变量或可调参数，供后续工作流使用。
    charge_basis = _pick_first_available_basis(charge_candidates, elements=elements)  # 设置中间变量或可调参数，供后续工作流使用。
    if formal_charge < 0 and (opt_basis != opt_candidates[0] or charge_basis != charge_candidates[0]):  # 根据当前状态决定是否进入该分支。
        reason = f"{reason} -> fallback"  # 设置中间变量或可调参数，供后续工作流使用。

    return QMSpec(  # 返回该辅助函数的结果。
        method=method,  # 设置中间变量或可调参数，供后续工作流使用。
        opt_basis=opt_basis,  # 设置中间变量或可调参数，供后续工作流使用。
        charge_basis=charge_basis,  # 设置中间变量或可调参数，供后续工作流使用。
        reason=reason,  # 设置中间变量或可调参数，供后续工作流使用。
    )


def run_one_species(  # 定义本例内部辅助函数，组织重复步骤。
    spec: SpeciesSpec,
    *,
    db_dir: Path,
    job_wd: Path,
    psi4_omp: int,
    psi4_memory_mb: int,
) -> dict[str, Any]:
    species_wd = workdir(job_wd / spec.name, restart=False)  # 创建或复用本例工作目录。
    mol = mol_from_smiles(spec.smiles, name=spec.name)  # 设置中间变量或可调参数，供后续工作流使用。
    formal_charge = int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))  # 设置中间变量或可调参数，供后续工作流使用。
    charge_groups = detect_charged_groups(mol, detection="auto") if spec.polyelectrolyte_mode else {}  # 设置中间变量或可调参数，供后续工作流使用。
    qm_spec = _resolve_qm_spec(spec.smiles)  # 设置中间变量或可调参数，供后续工作流使用。
    bonded_mode = _bonded_mode(spec)  # 设置中间变量或可调参数，供后续工作流使用。
    ok = bool(  # 设置中间变量或可调参数，供后续工作流使用。
        assign_charges(  # 执行电荷分配流程。
            mol,
            charge=spec.charge,  # 指定电荷来源或电荷计算方式。
            opt=True,  # 设置中间变量或可调参数，供后续工作流使用。
            work_dir=species_wd,  # 设置本例输出目录。
            log_name=spec.name,  # 设置中间变量或可调参数，供后续工作流使用。
            omp=psi4_omp,  # 设置每个 rank 的 OpenMP 线程数。
            memory=psi4_memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
            opt_method=(qm_spec.method if qm_spec else "wb97m-d3bj"),  # 设置中间变量或可调参数，供后续工作流使用。
            charge_method=(qm_spec.method if qm_spec else "wb97m-d3bj"),  # 设置中间变量或可调参数，供后续工作流使用。
            opt_basis=(qm_spec.opt_basis if qm_spec else "def2-SVP"),  # 设置中间变量或可调参数，供后续工作流使用。
            charge_basis=(qm_spec.charge_basis if qm_spec else "def2-TZVP"),  # 设置中间变量或可调参数，供后续工作流使用。
            total_charge=formal_charge,  # 设置中间变量或可调参数，供后续工作流使用。
            total_multiplicity=1,  # 设置中间变量或可调参数，供后续工作流使用。
            polyelectrolyte_mode=spec.polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
            polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
            bonded_params=bonded_mode,  # 设置中间变量或可调参数，供后续工作流使用。
        )
    )

    if not ok:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"assign_charges failed for {spec.name} {spec.smiles}")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    record = MolDB(db_dir).update_from_mol(  # 设置中间变量或可调参数，供后续工作流使用。
        mol,
        smiles_or_psmiles=spec.smiles,  # 设置中间变量或可调参数，供后续工作流使用。
        name=spec.name,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        charge=spec.charge,  # 指定电荷来源或电荷计算方式。
        polyelectrolyte_mode=spec.polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
        polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
    )
    return {  # 返回该辅助函数的结果。
        "name": spec.name,
        "smiles": spec.smiles,
        "kind": spec.kind,
        "source": spec.source,
        "charge": spec.charge,
        "bonded": spec.bonded,
        "polyelectrolyte_mode": spec.polyelectrolyte_mode,
        "formal_charge": formal_charge,
        "charge_group_count": len(charge_groups.get("groups") or []),
        "bonded_mode": bonded_mode,
        "qm_method": (qm_spec.method if qm_spec else None),
        "qm_opt_basis": (qm_spec.opt_basis if qm_spec else None),
        "qm_charge_basis": (qm_spec.charge_basis if qm_spec else None),
        "qm_policy": (qm_spec.reason if qm_spec else None),
        "record_key": record.key,
        "psi4_omp": int(psi4_omp),
        "psi4_memory_mb": int(psi4_memory_mb),
    }


def main() -> int:  # 定义本例内部辅助函数，组织重复步骤。
    restart_status = False  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
    set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

    psi4_omp = 36  # 设置中间变量或可调参数，供后续工作流使用。
    psi4_memory_mb = 20000  # 设置中间变量或可调参数，供后续工作流使用。

    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    db = MolDB()  # 设置中间变量或可调参数，供后续工作流使用。
    db_dir = Path(db.db_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    example_wd = workdir(HERE / "work_dir", restart=restart_status)  # 创建或复用本例工作目录。
    job_wd = example_wd.child("01_build_moldb")  # 设置中间变量或可调参数，供后续工作流使用。
    species = _read_species_csv(CATALOG_CSV)  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。

    summary: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    failures: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。

    for spec in species:  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            summary.append(  # 开始一个多行函数调用或配置块。
                run_one_species(  # 开始一个多行函数调用或配置块。
                    spec,
                    db_dir=db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
                    job_wd=Path(job_wd),  # 设置中间变量或可调参数，供后续工作流使用。
                    psi4_omp=psi4_omp,  # 设置中间变量或可调参数，供后续工作流使用。
                    psi4_memory_mb=psi4_memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
                )
            )
            print(  # 打印关键路径或状态，便于人工检查。
                f"[OK] {spec.name:20s} charge={spec.charge:5s} bonded={spec.bonded or '-'}"
            )
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            failures.append(  # 开始一个多行函数调用或配置块。
                {
                    "name": spec.name,
                    "smiles": spec.smiles,
                    "charge": spec.charge,
                    "bonded": spec.bonded,
                    "error": repr(exc),
                }
            )
            print(  # 打印关键路径或状态，便于人工检查。
                f"[FAIL] {spec.name:20s} charge={spec.charge:5s} bonded={spec.bonded or '-'} :: {exc}"
            )

    out = {  # 设置中间变量或可调参数，供后续工作流使用。
        "catalog_csv": str(CATALOG_CSV.resolve()),
        "db_dir": str(db_dir.resolve()),
        "work_root": str(job_wd.resolve()),
        "psi4_omp": psi4_omp,
        "psi4_memory_mb": psi4_memory_mb,
        "success_count": len(summary),
        "failure_count": len(failures),
        "success": summary,
        "failures": failures,
    }
    (job_wd / "build_moldb_summary.json").write_text(  # 开始一个多行函数调用或配置块。
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",  # 设置中间变量或可调参数，供后续工作流使用。
        encoding="utf-8",  # 设置中间变量或可调参数，供后续工作流使用。
    )

    print(f"\nMolDB directory: {db_dir}")  # 打印关键路径或状态，便于人工检查。
    print(f"Catalog CSV   : {CATALOG_CSV}")  # 打印关键路径或状态，便于人工检查。
    print(f"Psi4 OMP      : {psi4_omp}")  # 打印关键路径或状态，便于人工检查。
    print(f"Success       : {len(summary)}")  # 打印关键路径或状态，便于人工检查。
    print(f"Failures      : {len(failures)}")  # 打印关键路径或状态，便于人工检查。
    return 0 if not failures else 1  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    raise SystemExit(main())  # 关键步骤失败时立即报错，避免继续生成错误结果。
