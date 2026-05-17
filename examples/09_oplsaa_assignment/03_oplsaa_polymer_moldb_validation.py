from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 09 / Step 3: MolDB-first OPLS-AA polymer validation.

This script is intentionally lightweight: it does not run QM or MD.  It checks
that polymer and electrolyte building blocks can be loaded from the repo MolDB,
assigned with OPLS-AA, audited for source-backed parameters, and exported for
GROMACS inspection.
"""

import json  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.core import naming, poly, utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff import OPLSAA  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.oplsaa_reference import audit_oplsaa_assignment  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.io.gmx import write_gmx  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.moldb import MolDB  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


# ---------------- user inputs ----------------
restart_status = False  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
neutral_polymer_dp = 4  # 设置中间变量或可调参数，供后续工作流使用。
polyelectrolyte_dp = 4  # 设置中间变量或可调参数，供后续工作流使用。

PEO_smiles = "*CCO*"  # 设置中间变量或可调参数，供后续工作流使用。
CMC_neutral_smiles = "*OC1OC(CO)C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。
CMC_carboxylate_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。
EC_smiles = "O=C1OCCO1"  # 设置中间变量或可调参数，供后续工作流使用。
EMC_smiles = "CCOC(=O)OC"  # 设置中间变量或可调参数，供后续工作流使用。
PF6_smiles = "F[P-](F)(F)(F)(F)F"  # 设置中间变量或可调参数，供后续工作流使用。
Na_smiles = "[Na+]"  # 设置中间变量或可调参数，供后续工作流使用。
ter_smiles = "[H][*]"  # 设置中间变量或可调参数，供后续工作流使用。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"  # 设置中间变量或可调参数，供后续工作流使用。
work_dir = BASE_DIR / "work_dir"  # 设置本例输出目录。

set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。


def _load_ready_resp(ff: OPLSAA, smiles: str, label: str, *, polyelectrolyte_mode: bool = False):  # 定义本例内部辅助函数，组织重复步骤。
    """Load a RESP-ready species from the repo MolDB and attach a stable name."""

    mol = ff.mol_rdkit(  # 设置中间变量或可调参数，供后续工作流使用。
        smiles,
        db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        require_ready=True,  # 要求 MolDB 物种必须已准备好。
        prefer_db=True,  # 优先从 MolDB 读取已有结果。
        polyelectrolyte_mode=polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
        polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
    )
    naming.ensure_name(mol, name=label)  # 设置中间变量或可调参数，供后续工作流使用。
    return mol  # 返回该辅助函数的结果。


def _assign_and_audit(  # 定义本例内部辅助函数，组织重复步骤。
    ff: OPLSAA,
    mol,
    label: str,
    *,
    charge: str | None = None,  # 设置中间变量或可调参数，供后续工作流使用。
    polyelectrolyte_mode: bool = False,  # 设置中间变量或可调参数，供后续工作流使用。
    bonded_work_dir: Path | None = None,  # 设置中间变量或可调参数，供后续工作流使用。
):
    """Run OPLS-AA assignment and return a compact source/provenance audit."""

    assigned = ff.ff_assign(  # 分配力场参数并写入分子属性。
        mol,
        charge=charge,  # 指定电荷来源或电荷计算方式。
        report=False,  # 控制是否打印详细分配报告。
        polyelectrolyte_mode=polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
        bonded_work_dir=bonded_work_dir,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    if not assigned:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"OPLS-AA assignment returned False for {label}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    naming.ensure_name(assigned, name=label)  # 设置中间变量或可调参数，供后续工作流使用。
    audit = audit_oplsaa_assignment(assigned, strict=True)  # 设置中间变量或可调参数，供后续工作流使用。
    return assigned, audit  # 返回该辅助函数的结果。


def _summary_item(label: str, smiles: str, profile: str, audit: dict[str, object], *, note: str = "") -> dict[str, object]:  # 定义本例内部辅助函数，组织重复步骤。
    """Normalize audit fields used by the JSON summary and terminal report."""

    return {  # 返回该辅助函数的结果。
        "label": label,
        "smiles": smiles,
        "profile": profile,
        "note": note,
        "assignment_complete": bool(audit.get("assignment_complete")),
        "strict_source_clean": bool(audit.get("strict_source_clean")),
        "atom_count": int(audit.get("atom_count", 0)),
        "net_charge": float(audit.get("net_charge", 0.0)),
        "local_refine_count": len(audit.get("local_refines") or []),
        "missing_nonbonded_count": len(audit.get("missing_nonbonded") or []),
        "missing_bonded_count": len(audit.get("missing_bonded") or []),
        "external_bonded": audit.get("external_bonded"),
        "pf6": audit.get("pf6"),
    }


def _write_case_gmx(mol, root: Path, label: str) -> dict[str, str]:  # 定义本例内部辅助函数，组织重复步骤。
    """Export one assigned molecule and return stable output paths."""

    out_dir = root / f"{label}_gmx"  # 设置中间变量或可调参数，供后续工作流使用。
    gro, itp, top = write_gmx(mol=mol, out_dir=out_dir, mol_name=label)  # 设置中间变量或可调参数，供后续工作流使用。
    return {"gro": str(gro), "itp": str(itp), "top": str(top)}  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    db = MolDB(REPO_DB_DIR)  # 设置中间变量或可调参数，供后续工作流使用。
    if not db.objects_dir.is_dir():  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"Repo MolDB is missing: {db.objects_dir}")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    example_wd = workdir(work_dir, restart=restart_status)  # 创建或复用本例工作目录。
    job_wd = Path(example_wd.child("03_polymer_moldb_validation"))  # 设置中间变量或可调参数，供后续工作流使用。

    strict_ff = OPLSAA(profile="strict")  # 设置中间变量或可调参数，供后续工作流使用。
    refine_ff = OPLSAA(profile="refine")  # 设置中间变量或可调参数，供后续工作流使用。
    summary: list[dict[str, object]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    failures: list[dict[str, object]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    exports: dict[str, dict[str, str]] = {}  # 设置中间变量或可调参数，供后续工作流使用。

    # Neutral polymer: this should be fully source-clean in strict mode.
    peo_monomer = _load_ready_resp(strict_ff, PEO_smiles, "PEO_monomer")  # 设置中间变量或可调参数，供后续工作流使用。
    peo_monomer, peo_monomer_audit = _assign_and_audit(strict_ff, peo_monomer, "PEO_monomer")  # 设置中间变量或可调参数，供后续工作流使用。
    summary.append(_summary_item("PEO_monomer", PEO_smiles, "strict", peo_monomer_audit))

    terminator = utils.mol_from_smiles(ter_smiles)  # 从 SMILES 直接构造 RDKit 分子。
    peo_oligomer = poly.random_copolymerize_rw(  # 用随机游走生成聚合物链。
        [peo_monomer],
        neutral_polymer_dp,
        ratio=[1.0],  # 设置共聚组成比例。
        tacticity="atactic",  # 设置聚合物立构。
        name="PEO_oligomer",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=job_wd / "01_peo_rw",  # 设置本例输出目录。
    )
    peo_oligomer = poly.terminate_rw(  # 给聚合物链加端基。
        peo_oligomer,
        terminator,
        name="PEO_oligomer",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=job_wd / "02_peo_term",  # 设置本例输出目录。
    )
    peo_oligomer, peo_oligomer_audit = _assign_and_audit(strict_ff, peo_oligomer, "PEO_oligomer")  # 设置中间变量或可调参数，供后续工作流使用。
    summary.append(_summary_item("PEO_oligomer", PEO_smiles, "strict", peo_oligomer_audit))
    exports["PEO_oligomer"] = _write_case_gmx(peo_oligomer, job_wd, "PEO_oligomer")  # 设置中间变量或可调参数，供后续工作流使用。

    # Polyelectrolyte probe: strict should fail fast until all CMC terms are
    # promoted from refine-only analogs into source-backed OPLS-AA data.
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        strict_cmc = _load_ready_resp(  # 设置中间变量或可调参数，供后续工作流使用。
            strict_ff,
            CMC_carboxylate_smiles,
            "CMC_carboxylate_strict_probe",
            polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        )
        _assign_and_audit(  # 开始一个多行函数调用或配置块。
            strict_ff,
            strict_cmc,
            "CMC_carboxylate_strict_probe",
            polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        )
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        failures.append(  # 开始一个多行函数调用或配置块。
            {
                "label": "CMC_carboxylate_strict_probe",
                "profile": "strict",
                "expected": True,
                "reason": "CMC strict OPLS-AA still has source-backed bonded gaps.",
                "error": repr(exc),
            }
        )

    cmc_neutral = _load_ready_resp(  # 设置中间变量或可调参数，供后续工作流使用。
        refine_ff,
        CMC_neutral_smiles,
        "CMC_neutral_unit",
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
    )
    cmc_carboxylate = _load_ready_resp(  # 设置中间变量或可调参数，供后续工作流使用。
        refine_ff,
        CMC_carboxylate_smiles,
        "CMC_carboxylate_unit",
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
    )
    cmc_neutral, _ = _assign_and_audit(  # 设置中间变量或可调参数，供后续工作流使用。
        refine_ff,
        cmc_neutral,
        "CMC_neutral_unit",
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
    )
    cmc_carboxylate, _ = _assign_and_audit(  # 设置中间变量或可调参数，供后续工作流使用。
        refine_ff,
        cmc_carboxylate,
        "CMC_carboxylate_unit",
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
    )
    cmc_oligomer = poly.random_copolymerize_rw(  # 用随机游走生成聚合物链。
        [cmc_neutral, cmc_carboxylate],
        polyelectrolyte_dp,
        ratio=[1.0, 1.0],  # 设置共聚组成比例。
        tacticity="atactic",  # 设置聚合物立构。
        name="CMC_Na_oligomer",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=job_wd / "03_cmc_rw",  # 设置本例输出目录。
    )
    cmc_oligomer = poly.terminate_rw(  # 给聚合物链加端基。
        cmc_oligomer,
        terminator,
        name="CMC_Na_oligomer",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=job_wd / "04_cmc_term",  # 设置本例输出目录。
    )
    cmc_oligomer, cmc_oligomer_audit = _assign_and_audit(  # 设置中间变量或可调参数，供后续工作流使用。
        refine_ff,
        cmc_oligomer,
        "CMC_Na_oligomer",
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
    )
    summary.append(  # 开始一个多行函数调用或配置块。
        _summary_item(  # 开始一个多行函数调用或配置块。
            "CMC_Na_oligomer",
            f"{CMC_neutral_smiles} + {CMC_carboxylate_smiles}",
            "refine",
            cmc_oligomer_audit,
            note="assignment-complete but not strict-source-clean until CMC local refines are source-backed",  # 设置中间变量或可调参数，供后续工作流使用。
        )
    )
    exports["CMC_Na_oligomer"] = _write_case_gmx(cmc_oligomer, job_wd, "CMC_Na_oligomer")  # 设置中间变量或可调参数，供后续工作流使用。

    # Electrolyte support species used by polymer-electrolyte workflows.
    pf6 = _load_ready_resp(strict_ff, PF6_smiles, "PF6")  # 设置中间变量或可调参数，供后续工作流使用。
    pf6, pf6_audit = _assign_and_audit(strict_ff, pf6, "PF6", bonded_work_dir=job_wd / "05_pf6_drih")  # 设置中间变量或可调参数，供后续工作流使用。
    summary.append(_summary_item("PF6", PF6_smiles, "strict", pf6_audit, note="uses precomputed MolDB DRIH bonded patch"))  # 设置中间变量或可调参数，供后续工作流使用。
    exports["PF6"] = _write_case_gmx(pf6, job_wd, "PF6")  # 设置中间变量或可调参数，供后续工作流使用。

    na = strict_ff.mol(Na_smiles, charge="opls", require_ready=False, prefer_db=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    na, na_audit = _assign_and_audit(strict_ff, na, "Na", charge="opls")  # 设置中间变量或可调参数，供后续工作流使用。
    summary.append(_summary_item("Na", Na_smiles, "strict", na_audit, note="monatomic ion uses built-in OPLS-AA ion data"))  # 设置中间变量或可调参数，供后续工作流使用。
    exports["Na"] = _write_case_gmx(na, job_wd, "Na")  # 设置中间变量或可调参数，供后续工作流使用。

    for label, smiles in (("EC_strict_probe", EC_smiles),):  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            mol = _load_ready_resp(strict_ff, smiles, label)  # 设置中间变量或可调参数，供后续工作流使用。
            _assign_and_audit(strict_ff, mol, label)
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            failures.append(  # 开始一个多行函数调用或配置块。
                {
                    "label": label,
                    "profile": "strict",
                    "expected": True,
                    "reason": "carbonate strict OPLS-AA still has source-backed bonded gaps.",
                    "error": repr(exc),
                }
            )

    emc = _load_ready_resp(refine_ff, EMC_smiles, "EMC_refine_probe")  # 设置中间变量或可调参数，供后续工作流使用。
    emc, emc_audit = _assign_and_audit(refine_ff, emc, "EMC_refine_probe")  # 设置中间变量或可调参数，供后续工作流使用。
    summary.append(  # 开始一个多行函数调用或配置块。
        _summary_item(  # 开始一个多行函数调用或配置块。
            "EMC_refine_probe",
            EMC_smiles,
            "refine",
            emc_audit,
            note="assignment-complete but not strict-source-clean until carbonate local refines are source-backed",  # 设置中间变量或可调参数，供后续工作流使用。
        )
    )
    exports["EMC_refine_probe"] = _write_case_gmx(emc, job_wd, "EMC_refine_probe")  # 设置中间变量或可调参数，供后续工作流使用。

    strict_clean_required = [item for item in summary if item["profile"] == "strict"]
    strict_clean_ok = all(bool(item["strict_source_clean"]) for item in strict_clean_required)  # 设置中间变量或可调参数，供后续工作流使用。
    refine_only = [item for item in summary if item["profile"] == "refine" and not bool(item["strict_source_clean"])]
    out = {  # 设置中间变量或可调参数，供后续工作流使用。
        "repo_moldb": str(REPO_DB_DIR.resolve()),
        "work_root": str(job_wd.resolve()),
        "strict_clean_required_count": len(strict_clean_required),
        "strict_clean_ok": strict_clean_ok,
        "refine_only_count": len(refine_only),
        "expected_strict_gap_count": len([item for item in failures if item.get("expected")]),
        "cases": summary,
        "strict_gaps": failures,
        "exports": exports,
    }
    (job_wd / "oplsaa_polymer_validation_summary.json").write_text(  # 开始一个多行函数调用或配置块。
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",  # 设置中间变量或可调参数，供后续工作流使用。
        encoding="utf-8",  # 设置中间变量或可调参数，供后续工作流使用。
    )

    print("\nOPLS-AA polymer MolDB validation")  # 打印关键路径或状态，便于人工检查。
    print(f"Work root: {job_wd.resolve()}")  # 打印关键路径或状态，便于人工检查。
    for item in summary:  # 遍历当前工作流中的一组对象或任务。
        print(  # 打印关键路径或状态，便于人工检查。
            f"[{item['profile']}] {item['label']:22s} "
            f"complete={item['assignment_complete']} "
            f"strict_source_clean={item['strict_source_clean']} "
            f"local_refines={item['local_refine_count']} "
            f"net={item['net_charge']:.6f}"
        )
    for item in failures:  # 遍历当前工作流中的一组对象或任务。
        marker = "expected strict gap" if item.get("expected") else "unexpected failure"  # 设置中间变量或可调参数，供后续工作流使用。
        print(f"[{marker}] {item['label']}: {item['reason']}")  # 打印关键路径或状态，便于人工检查。
    print(f"Summary: {job_wd / 'oplsaa_polymer_validation_summary.json'}")  # 打印关键路径或状态，便于人工检查。

    unexpected_failures = [item for item in failures if not item.get("expected")]  # 设置中间变量或可调参数，供后续工作流使用。
    raise SystemExit(0 if strict_clean_ok and not unexpected_failures else 1)  # 关键步骤失败时立即报错，避免继续生成错误结果。
