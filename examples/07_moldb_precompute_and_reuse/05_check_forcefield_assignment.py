from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 07 / Step 5: Spot-check force-field assignment on the precomputed catalog."""

import importlib.util  # 导入本例需要的库或 yadonpy 接口。
import json  # 导入本例需要的库或 yadonpy 接口。
import sys  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。
from dataclasses import dataclass  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import get_ff  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


HERE = Path(__file__).resolve().parent  # 定位当前脚本所在目录。
BUILD_SCRIPT = HERE / "01_build_moldb.py"  # 设置中间变量或可调参数，供后续工作流使用。
GROUP_ORDER = (  # 设置中间变量或可调参数，供后续工作流使用。
    "neutral_molecules",
    "drih_anions",
    "polyelectrolyte_monomers",
    "monatomic_ions",
)


@dataclass(frozen=True)  # 声明轻量数据类，用于保存配置或任务信息。
class DirectIonSpec:  # 定义本例内部数据结构或配置对象。
    name: str
    smiles: str


DIRECT_ION_SPECS = (  # 设置中间变量或可调参数，供后续工作流使用。
    DirectIonSpec(name="Li", smiles="[Li+]"),  # 设置中间变量或可调参数，供后续工作流使用。
    DirectIonSpec(name="Na", smiles="[Na+]"),  # 设置中间变量或可调参数，供后续工作流使用。
)


def _load_build_module():  # 定义本例内部辅助函数，组织重复步骤。
    spec = importlib.util.spec_from_file_location("example07_build_moldb", BUILD_SCRIPT)  # 设置中间变量或可调参数，供后续工作流使用。
    assert spec is not None  # 检查示例假设是否成立。
    assert spec.loader is not None  # 检查示例假设是否成立。
    module = importlib.util.module_from_spec(spec)  # 设置中间变量或可调参数，供后续工作流使用。
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module  # 返回该辅助函数的结果。


def _formal_charge(mol) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))  # 返回该辅助函数的结果。


def _catalog_report_group(spec, *, formal_charge: int) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    if bool(spec.polyelectrolyte_mode):  # 根据当前状态决定是否进入该分支。
        return "polyelectrolyte_monomers"  # 返回该辅助函数的结果。
    if str(spec.bonded or "").strip().upper() == "DRIH":  # 根据当前状态决定是否进入该分支。
        return "drih_anions"  # 返回该辅助函数的结果。
    if int(formal_charge) == 0:  # 根据当前状态决定是否进入该分支。
        return "neutral_molecules"  # 返回该辅助函数的结果。
    return "drih_anions"  # 返回该辅助函数的结果。


def _empty_group_report() -> dict[str, dict[str, object]]:  # 定义本例内部辅助函数，组织重复步骤。
    return {  # 返回该辅助函数的结果。
        group: {
            "success_count": 0,
            "failure_count": 0,
            "success": [],
            "failures": [],
        }
        for group in GROUP_ORDER  # 遍历当前工作流中的一组对象或任务。
    }


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    restart_status = False  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
    set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    ff = get_ff("gaff2_mod")  # 选择有机分子/聚合物/部分无机离子的力场对象。
    ion_ff = get_ff("merz")  # 选择单原子离子参数来源。

    build_mod = _load_build_module()  # 设置中间变量或可调参数，供后续工作流使用。
    species = build_mod._read_species_csv(build_mod.CATALOG_CSV)  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。

    example_wd = workdir(HERE / "work_dir", restart=restart_status)  # 创建或复用本例工作目录。
    job_wd = example_wd.child("05_check_forcefield_assignment")  # 设置中间变量或可调参数，供后续工作流使用。

    summary: list[dict[str, object]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    failures: list[dict[str, object]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    grouped = _empty_group_report()  # 设置中间变量或可调参数，供后续工作流使用。

    for spec in species:  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            mol = ff.mol(  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
                spec.smiles,
                charge=spec.charge,  # 指定电荷来源或电荷计算方式。
                require_ready=True,  # 要求 MolDB 物种必须已准备好。
                prefer_db=True,  # 优先从 MolDB 读取已有结果。
            )
            ok = bool(ff.ff_assign(mol, bonded=spec.bonded, report=False))  # 分配力场参数并写入分子属性。

            if not ok:  # 根据当前状态决定是否进入该分支。
                raise RuntimeError("ff_assign returned False")  # 关键步骤失败时立即报错，避免继续生成错误结果。

            formal_charge = _formal_charge(mol)  # 设置中间变量或可调参数，供后续工作流使用。
            group = _catalog_report_group(spec, formal_charge=formal_charge)  # 设置中间变量或可调参数，供后续工作流使用。
            item = {  # 设置中间变量或可调参数，供后续工作流使用。
                "name": spec.name,
                "smiles": spec.smiles,
                "charge": spec.charge,
                "bonded": spec.bonded,
                "formal_charge": formal_charge,
                "atom_count": int(mol.GetNumAtoms()),
                "report_group": group,
            }
            summary.append(item)
            grouped[group]["success"].append(item)
            print(  # 打印关键路径或状态，便于人工检查。
                f"[OK] {spec.name:20s} group={group:24s} charge={spec.charge:5s} bonded={spec.bonded or '-'}"
            )
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            group = (  # 设置中间变量或可调参数，供后续工作流使用。
                "polyelectrolyte_monomers"
                if spec.polyelectrolyte_mode  # 根据当前状态决定是否进入该分支。
                else "drih_anions"
                if str(spec.bonded or "").strip().upper() == "DRIH"  # 根据当前状态决定是否进入该分支。
                else "neutral_molecules"
            )
            item = {  # 设置中间变量或可调参数，供后续工作流使用。
                "name": spec.name,
                "smiles": spec.smiles,
                "charge": spec.charge,
                "bonded": spec.bonded,
                "report_group": group,
                "error": repr(exc),
            }
            failures.append(item)
            grouped[group]["failures"].append(item)
            print(  # 打印关键路径或状态，便于人工检查。
                f"[FAIL] {spec.name:20s} group={group:24s} charge={spec.charge:5s} bonded={spec.bonded or '-'} :: {exc}"
            )

    for ion_spec in DIRECT_ION_SPECS:  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            mol = ion_ff.mol(ion_spec.smiles)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
            ok = bool(ion_ff.ff_assign(mol, report=False))  # 分配力场参数并写入分子属性。
            if not ok:  # 根据当前状态决定是否进入该分支。
                raise RuntimeError("ff_assign returned False")  # 关键步骤失败时立即报错，避免继续生成错误结果。

            item = {  # 设置中间变量或可调参数，供后续工作流使用。
                "name": ion_spec.name,
                "smiles": ion_spec.smiles,
                "charge": "MERZ",
                "bonded": None,
                "formal_charge": _formal_charge(mol),
                "atom_count": int(mol.GetNumAtoms()),
                "report_group": "monatomic_ions",
            }
            summary.append(item)
            grouped["monatomic_ions"]["success"].append(item)
            print(f"[OK] {ion_spec.name:20s} group=monatomic_ions          charge=MERZ  bonded=-")  # 打印关键路径或状态，便于人工检查。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            item = {  # 设置中间变量或可调参数，供后续工作流使用。
                "name": ion_spec.name,
                "smiles": ion_spec.smiles,
                "charge": "MERZ",
                "bonded": None,
                "report_group": "monatomic_ions",
                "error": repr(exc),
            }
            failures.append(item)
            grouped["monatomic_ions"]["failures"].append(item)
            print(f"[FAIL] {ion_spec.name:20s} group=monatomic_ions          charge=MERZ  bonded=- :: {exc}")  # 打印关键路径或状态，便于人工检查。

    for group_name in GROUP_ORDER:  # 遍历当前工作流中的一组对象或任务。
        grouped[group_name]["success_count"] = len(grouped[group_name]["success"])  # 设置中间变量或可调参数，供后续工作流使用。
        grouped[group_name]["failure_count"] = len(grouped[group_name]["failures"])  # 设置中间变量或可调参数，供后续工作流使用。

        group_out = {  # 设置中间变量或可调参数，供后续工作流使用。
            "group": group_name,
            "success_count": grouped[group_name]["success_count"],
            "failure_count": grouped[group_name]["failure_count"],
            "success": grouped[group_name]["success"],
            "failures": grouped[group_name]["failures"],
        }
        (Path(job_wd) / f"{group_name}.json").write_text(  # 开始一个多行函数调用或配置块。
            json.dumps(group_out, indent=2, ensure_ascii=False) + "\n",  # 设置中间变量或可调参数，供后续工作流使用。
            encoding="utf-8",  # 设置中间变量或可调参数，供后续工作流使用。
        )

    out = {  # 设置中间变量或可调参数，供后续工作流使用。
        "catalog_csv": str(build_mod.CATALOG_CSV.resolve()),
        "work_root": str(Path(job_wd).resolve()),
        "success_count": len(summary),
        "failure_count": len(failures),
        "groups": grouped,
        "success": summary,
        "failures": failures,
    }
    (Path(job_wd) / "forcefield_check_summary.json").write_text(  # 开始一个多行函数调用或配置块。
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",  # 设置中间变量或可调参数，供后续工作流使用。
        encoding="utf-8",  # 设置中间变量或可调参数，供后续工作流使用。
    )

    print(f"\nCatalog CSV   : {build_mod.CATALOG_CSV}")  # 打印关键路径或状态，便于人工检查。
    for group_name in GROUP_ORDER:  # 遍历当前工作流中的一组对象或任务。
        print(  # 打印关键路径或状态，便于人工检查。
            f"{group_name:22s}: "
            f"{grouped[group_name]['success_count']} success / {grouped[group_name]['failure_count']} failure"
        )
    print(f"Success       : {len(summary)}")  # 打印关键路径或状态，便于人工检查。
    print(f"Failures      : {len(failures)}")  # 打印关键路径或状态，便于人工检查。
    raise SystemExit(0 if not failures else 1)  # 关键步骤失败时立即报错，避免继续生成错误结果。
