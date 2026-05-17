from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

import json  # 导入本例需要的库或 yadonpy 接口。
import os  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。
from typing import Any  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.core import poly, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.metadata import (  # 导入本例需要的库或 yadonpy 接口。
    QM_RECIPE_PROP,
    RESP_CONSTRAINTS_PROP,
    RESP_PROFILE_PROP,
    read_json_prop,
    read_text_prop,
)
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff import GAFF2_mod, OPLSAA  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.gmx.analysis.structured import build_msd_metric_catalog, compute_msd_series  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.gmx.topology import parse_system_top  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.analyzer import AnalyzeResult  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.benchmarking import _dump_json, summarize_rdkit_species_forcefield  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.preset import eq  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.workflow import EnvReader  # 导入本例需要的库或 yadonpy 接口。


_ENV = EnvReader()  # 设置中间变量或可调参数，供后续工作流使用。


def _env_bool(name: str, default: bool) -> bool:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    return _ENV.bool(name, default)  # 返回该辅助函数的结果。


def _env_int(name: str, default: int) -> int:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    return _ENV.int(name, default)  # 返回该辅助函数的结果。


def _env_float(name: str, default: float) -> float:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    return _ENV.float(name, default)  # 返回该辅助函数的结果。


def _env_text(name: str, default: str) -> str:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    return _ENV.text(name, default)  # 返回该辅助函数的结果。


def _normalize_constraints(raw: str | None, default: str = "h-bonds") -> str:  # 定义本例内部辅助函数，组织重复步骤。
    token = str(raw or default).strip().lower().replace("_", "-")  # 设置中间变量或可调参数，供后续工作流使用。
    aliases = {  # 设置中间变量或可调参数，供后续工作流使用。
        "none": "none",
        "no": "none",
        "off": "none",
        "hbonds": "h-bonds",
        "h-bonds": "h-bonds",
        "allbonds": "all-bonds",
        "all-bonds": "all-bonds",
    }
    if token not in aliases:  # 根据当前状态决定是否进入该分支。
        raise ValueError(f"Unsupported constraints mode: {raw!r}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return aliases[token]  # 返回该辅助函数的结果。


def _normalize_charge_mode(raw: str | None) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    mode = str(raw or "resp").strip().lower()  # 设置该配置块使用的计算模式。
    if mode in {"native", "opls", "oplsaa"}:  # 根据当前状态决定是否进入该分支。
        return "opls"  # 返回该辅助函数的结果。
    if mode in {"resp2", "resp_2"}:  # 根据当前状态决定是否进入该分支。
        return "resp2"  # 返回该辅助函数的结果。
    return "resp"  # 返回该辅助函数的结果。


def _normalize_production_ensemble(raw: str | None) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    token = str(raw or "npt").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if token in {"nvt", "canonical"}:  # 根据当前状态决定是否进入该分支。
        return "nvt"  # 返回该辅助函数的结果。
    if token in {"npt", "isothermal-isobaric", "isothermal_isobaric"}:  # 根据当前状态决定是否进入该分支。
        return "npt"  # 返回该辅助函数的结果。
    raise ValueError(f"Unsupported production ensemble: {raw!r}")  # 关键步骤失败时立即报错，避免继续生成错误结果。


def _json_prop(mol, key: str) -> dict[str, Any] | None:  # 定义本例内部辅助函数，组织重复步骤。
    value = read_json_prop(mol, key)  # 设置中间变量或可调参数，供后续工作流使用。
    return value if isinstance(value, dict) else None  # 返回该辅助函数的结果。


def _extract_resp_route(mol, *, label: str) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    route = {  # 设置中间变量或可调参数，供后续工作流使用。
        "label": str(label),
        "resp_profile": None,
        "qm_recipe": None,
        "constraint_mode": None,
        "equivalence_group_count": 0,
    }
    route["resp_profile"] = read_text_prop(mol, RESP_PROFILE_PROP)  # 设置中间变量或可调参数，供后续工作流使用。
    qm_recipe = _json_prop(mol, QM_RECIPE_PROP)  # 设置中间变量或可调参数，供后续工作流使用。
    if isinstance(qm_recipe, dict):  # 根据当前状态决定是否进入该分支。
        route["qm_recipe"] = qm_recipe  # 设置中间变量或可调参数，供后续工作流使用。
        if route["resp_profile"] is None:  # 根据当前状态决定是否进入该分支。
            route["resp_profile"] = qm_recipe.get("resp_profile")  # 设置中间变量或可调参数，供后续工作流使用。
    constraints = _json_prop(mol, RESP_CONSTRAINTS_PROP)  # 设置约束策略。
    if isinstance(constraints, dict):  # 根据当前状态决定是否进入该分支。
        route["constraint_mode"] = constraints.get("mode")  # 设置中间变量或可调参数，供后续工作流使用。
        route["equivalence_group_count"] = int(len(constraints.get("equivalence_groups") or []))  # 设置中间变量或可调参数，供后续工作流使用。
        if route["resp_profile"] is None:  # 根据当前状态决定是否进入该分支。
            route["resp_profile"] = constraints.get("resp_profile")  # 设置中间变量或可调参数，供后续工作流使用。
    return route  # 返回该辅助函数的结果。


def _equivalence_spread_diagnostic(mol, *, label: str) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    constraints = _json_prop(mol, RESP_CONSTRAINTS_PROP) or {}  # 设置约束策略。
    groups = list(constraints.get("equivalence_groups") or [])  # 设置中间变量或可调参数，供后续工作流使用。
    diagnostics = []  # 设置中间变量或可调参数，供后续工作流使用。
    for group in groups:  # 遍历当前工作流中的一组对象或任务。
        idxs = sorted({int(i) for i in group})  # 设置中间变量或可调参数，供后续工作流使用。
        if len(idxs) <= 1:  # 根据当前状态决定是否进入该分支。
            continue
        values = []  # 设置中间变量或可调参数，供后续工作流使用。
        for idx in idxs:  # 遍历当前工作流中的一组对象或任务。
            atom = mol.GetAtomWithIdx(idx)  # 设置中间变量或可调参数，供后续工作流使用。
            if not atom.HasProp("AtomicCharge"):  # 根据当前状态决定是否进入该分支。
                values = []  # 设置中间变量或可调参数，供后续工作流使用。
                break
            values.append(float(atom.GetDoubleProp("AtomicCharge")))
        diagnostics.append(  # 开始一个多行函数调用或配置块。
            {
                "atom_indices": idxs,
                "symbols": [str(mol.GetAtomWithIdx(idx).GetSymbol()) for idx in idxs],
                "atomic_charge_spread_e": float(max(values) - min(values)) if values else None,
            }
        )
    max_spread = max(  # 设置中间变量或可调参数，供后续工作流使用。
        (float(item["atomic_charge_spread_e"]) for item in diagnostics if item.get("atomic_charge_spread_e") is not None),
        default=0.0,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    return {"label": str(label), "group_count": len(diagnostics), "max_spread_e": float(max_spread), "groups": diagnostics}  # 返回该辅助函数的结果。


def _load_ready_opls_species(  # 定义本例内部辅助函数，组织重复步骤。
    ff: OPLSAA,
    smiles: str,
    *,
    label: str,
    repo_db_dir: Path,
    charge_mode: str,
):
    last_exc: Exception | None = None  # 设置中间变量或可调参数，供后续工作流使用。
    db_charge = "RESP2" if charge_mode == "resp2" else "RESP"
    for db_dir, db_label in ((repo_db_dir, "repo"), (None, "default")):  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            mol = ff.mol_rdkit(  # 设置中间变量或可调参数，供后续工作流使用。
                smiles,
                name=label,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                db_dir=db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
                charge=db_charge,  # 指定电荷来源或电荷计算方式。
                require_ready=True,  # 要求 MolDB 物种必须已准备好。
                prefer_db=True,  # 优先从 MolDB 读取已有结果。
            )
            assign_charge = "opls" if charge_mode == "opls" else None
            mol = ff.ff_assign(mol, charge=assign_charge, report=False)  # 分配力场参数并写入分子属性。
            if not mol:  # 根据当前状态决定是否进入该分支。
                raise RuntimeError(f"Cannot assign OPLS-AA parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
            if assign_charge == "opls":  # 根据当前状态决定是否进入该分支。
                print(f"[MolDB] loaded {label} geometry from {db_label} db and switched to built-in OPLS-AA charges")  # 打印关键路径或状态，便于人工检查。
            else:  # 处理前面条件都不满足的情况。
                print(f"[MolDB] loaded {label} with {db_charge} charges from {db_label} db")  # 打印关键路径或状态，便于人工检查。
            return mol  # 返回该辅助函数的结果。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            last_exc = exc  # 设置中间变量或可调参数，供后续工作流使用。
    raise RuntimeError(f"{label} is expected to be ready in MolDB for the OPLS-AA benchmark.") from last_exc  # 关键步骤失败时立即报错，避免继续生成错误结果。


def _assign_builtin_opls_ion(ff: OPLSAA, smiles: str, *, label: str):  # 定义本例内部辅助函数，组织重复步骤。
    mol = ff.mol(smiles, charge="opls", require_ready=False, prefer_db=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    mol = ff.ff_assign(mol, charge="opls", report=False)  # 分配力场参数并写入分子属性。
    if not mol:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"Cannot assign built-in OPLS-AA ion parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    print(f"[OPLS-AA] assigned built-in ion parameters for {label}")  # 打印关键路径或状态，便于人工检查。
    return mol  # 返回该辅助函数的结果。


def _load_pf6_with_opls_parameters(*, ion_ff: OPLSAA, repo_db_dir: Path, charge_mode: str):  # 定义本例内部辅助函数，组织重复步骤。
    gaff_ff = GAFF2_mod()  # 设置中间变量或可调参数，供后续工作流使用。
    last_exc: Exception | None = None  # 设置中间变量或可调参数，供后续工作流使用。
    use_opls_charges = str(charge_mode).strip().lower() == "opls"
    opls_probe = ion_ff.mol(PF6_SMILES, charge="opls", require_ready=False, prefer_db=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    if not ion_ff.assign_ptypes(opls_probe, charge="opls"):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Cannot build the PF6 OPLS-AA atom-type probe from SMILES.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    for db_dir, db_label in ((repo_db_dir, "repo"), (None, "default")):  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            pf6 = gaff_ff.mol_rdkit(  # 设置中间变量或可调参数，供后续工作流使用。
                PF6_SMILES,
                name="PF6",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                db_dir=db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
                charge="RESP",  # 指定电荷来源或电荷计算方式。
                require_ready=True,  # 要求 MolDB 物种必须已准备好。
                prefer_db=True,  # 优先从 MolDB 读取已有结果。
            )
            pf6 = gaff_ff.ff_assign(pf6, bonded="DRIH", report=False)  # 分配力场参数并写入分子属性。
            if not pf6:  # 根据当前状态决定是否进入该分支。
                raise RuntimeError("Cannot restore PF6 DRIH bonded topology from MolDB.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
            if pf6.GetNumAtoms() != opls_probe.GetNumAtoms():  # 根据当前状态决定是否进入该分支。
                raise RuntimeError("PF6 probe atom count does not match MolDB topology.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

            for src_atom, dst_atom in zip(opls_probe.GetAtoms(), pf6.GetAtoms()):  # 遍历当前工作流中的一组对象或任务。
                if src_atom.GetSymbol() != dst_atom.GetSymbol():  # 根据当前状态决定是否进入该分支。
                    raise RuntimeError("PF6 probe atom ordering does not match MolDB topology.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
                dst_atom.SetProp("ff_btype", src_atom.GetProp("ff_btype"))
                dst_atom.SetProp("ff_type", src_atom.GetProp("ff_type"))
                dst_atom.SetDoubleProp("ff_sigma", src_atom.GetDoubleProp("ff_sigma"))
                dst_atom.SetDoubleProp("ff_epsilon", src_atom.GetDoubleProp("ff_epsilon"))
                if use_opls_charges:  # 根据当前状态决定是否进入该分支。
                    dst_atom.SetDoubleProp("AtomicCharge", src_atom.GetDoubleProp("AtomicCharge"))
                if src_atom.HasProp("ff_desc"):  # 根据当前状态决定是否进入该分支。
                    dst_atom.SetProp("ff_desc", src_atom.GetProp("ff_desc"))
            pf6.SetProp("ff_name", str(ion_ff.name))
            pf6.SetProp("ff_class", str(ion_ff.ff_class))
            pf6.SetProp("pair_style", str(ion_ff.pair_style))
            charge_note = (  # 设置中间变量或可调参数，供后续工作流使用。
                "replaced charges with built-in OPLS-AA values"
                if use_opls_charges  # 根据当前状态决定是否进入该分支。
                else "preserved MolDB RESP charges"
            )
            print(f"[OPLS-AA] loaded PF6 bonded topology from {db_label} db, copied OPLS-AA LJ/types, and {charge_note}")  # 打印关键路径或状态，便于人工检查。
            return pf6  # 返回该辅助函数的结果。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            last_exc = exc  # 设置中间变量或可调参数，供后续工作流使用。

    raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
        "PF6 is expected to exist in MolDB with bonded='DRIH' for this OPLS-AA benchmark."
    ) from last_exc


def _extract_default_diffusivity(msd: dict[str, Any], moltype: str) -> float | None:  # 定义本例内部辅助函数，组织重复步骤。
    record = msd.get(moltype) or msd.get(str(moltype).lower())  # 设置中间变量或可调参数，供后续工作流使用。
    if not isinstance(record, dict):  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        direct = record.get("D_m2_s")  # 设置中间变量或可调参数，供后续工作流使用。
        if direct is not None:  # 根据当前状态决定是否进入该分支。
            return float(direct)  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        pass
    metric_name = str(record.get("default_metric") or "").strip()  # 设置中间变量或可调参数，供后续工作流使用。
    metrics = record.get("metrics")  # 设置中间变量或可调参数，供后续工作流使用。
    if not metric_name or not isinstance(metrics, dict):  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    metric = metrics.get(metric_name)  # 设置中间变量或可调参数，供后续工作流使用。
    if not isinstance(metric, dict):  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        return float(metric.get("D_m2_s"))  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return None  # 返回该辅助函数的结果。


def _extract_default_msd_metric_record(msd: dict[str, Any], moltype: str) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    record = msd.get(moltype) or msd.get(str(moltype).lower())  # 设置中间变量或可调参数，供后续工作流使用。
    if not isinstance(record, dict):  # 根据当前状态决定是否进入该分支。
        return {}  # 返回该辅助函数的结果。
    metric_name = str(record.get("default_metric") or "").strip()  # 设置中间变量或可调参数，供后续工作流使用。
    metrics = record.get("metrics")  # 设置中间变量或可调参数，供后续工作流使用。
    if metric_name and isinstance(metrics, dict) and isinstance(metrics.get(metric_name), dict):  # 根据当前状态决定是否进入该分支。
        return dict(metrics[metric_name])  # 返回该辅助函数的结果。
    return dict(record)  # 返回该辅助函数的结果。


def _default_msd_trajectory_bounds(msd: dict[str, Any], labels: tuple[str, ...] = ("EC", "EMC", "DEC")) -> tuple[float | None, float | None]:  # 定义本例内部辅助函数，组织重复步骤。
    import numpy as np  # 导入本例需要的库或 yadonpy 接口。

    starts: list[float] = []  # 设置中间变量或可调参数，供后续工作流使用。
    ends: list[float] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for label in labels:  # 遍历当前工作流中的一组对象或任务。
        metric = _extract_default_msd_metric_record(msd, label)  # 设置中间变量或可调参数，供后续工作流使用。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            start_raw = metric.get("trajectory_time_start_ps")  # 设置中间变量或可调参数，供后续工作流使用。
            end_raw = metric.get("trajectory_time_end_ps")  # 设置中间变量或可调参数，供后续工作流使用。
            if start_raw is not None and end_raw is not None:  # 根据当前状态决定是否进入该分支。
                start = float(start_raw)  # 设置中间变量或可调参数，供后续工作流使用。
                end = float(end_raw)  # 设置中间变量或可调参数，供后续工作流使用。
                if np.isfinite(start) and np.isfinite(end) and end > start:  # 根据当前状态决定是否进入该分支。
                    starts.append(start)
                    ends.append(end)
                    continue
        except Exception:  # 捕获异常并转成更清楚的示例错误信息。
            pass
    if not starts or not ends:  # 根据当前状态决定是否进入该分支。
        return None, None  # 返回该辅助函数的结果。
    return float(min(starts)), float(max(ends))  # 返回该辅助函数的结果。


def _summarize_msd_block_diffusion(  # 定义本例内部辅助函数，组织重复步骤。
    blocks: list[dict[str, Any]],
    *,
    expected_order: tuple[str, ...] = ("EMC", "DEC", "EC"),  # 设置中间变量或可调参数，供后续工作流使用。
) -> dict[str, Any]:
    import numpy as np  # 导入本例需要的库或 yadonpy 接口。

    valid_blocks = [block for block in blocks if isinstance(block.get("diffusion_m2_s"), dict)]  # 设置中间变量或可调参数，供后续工作流使用。
    if not valid_blocks:  # 根据当前状态决定是否进入该分支。
        return {"status": "skipped", "reason": "no_valid_block_diffusion", "blocks": blocks}  # 返回该辅助函数的结果。
    species_labels = sorted(  # 设置中间变量或可调参数，供后续工作流使用。
        {
            str(label)
            for block in valid_blocks  # 遍历当前工作流中的一组对象或任务。
            for label, value in (block.get("diffusion_m2_s") or {}).items()  # 遍历当前工作流中的一组对象或任务。
            if value is not None  # 根据当前状态决定是否进入该分支。
        }
    )
    species_stats: dict[str, Any] = {}  # 设置中间变量或可调参数，供后续工作流使用。
    for label in species_labels:  # 遍历当前工作流中的一组对象或任务。
        values = []  # 设置中间变量或可调参数，供后续工作流使用。
        for block in valid_blocks:  # 遍历当前工作流中的一组对象或任务。
            try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
                value = (block.get("diffusion_m2_s") or {}).get(label)  # 设置中间变量或可调参数，供后续工作流使用。
                if value is not None and np.isfinite(float(value)):  # 根据当前状态决定是否进入该分支。
                    values.append(float(value))
            except Exception:  # 捕获异常并转成更清楚的示例错误信息。
                continue
        if values:  # 根据当前状态决定是否进入该分支。
            arr = np.asarray(values, dtype=float)  # 设置中间变量或可调参数，供后续工作流使用。
            std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
            mean = float(np.mean(arr))  # 设置中间变量或可调参数，供后续工作流使用。
            species_stats[label] = {  # 设置中间变量或可调参数，供后续工作流使用。
                "n_valid_blocks": int(arr.size),
                "mean_D_m2_s": mean,
                "std_D_m2_s": std,
                "sem_D_m2_s": float(std / np.sqrt(arr.size)) if arr.size >= 2 else 0.0,
                "cv": float(std / abs(mean)) if mean != 0.0 else None,
                "min_D_m2_s": float(np.min(arr)),
                "max_D_m2_s": float(np.max(arr)),
            }

    expected_present = [label for label in expected_order if label in species_stats]  # 设置中间变量或可调参数，供后续工作流使用。
    block_orders = []  # 设置中间变量或可调参数，供后续工作流使用。
    expected_order_matches = []  # 设置中间变量或可调参数，供后续工作流使用。
    for block in valid_blocks:  # 遍历当前工作流中的一组对象或任务。
        diffusion = {  # 设置中间变量或可调参数，供后续工作流使用。
            str(label): float(value)
            for label, value in (block.get("diffusion_m2_s") or {}).items()  # 遍历当前工作流中的一组对象或任务。
            if value is not None  # 根据当前状态决定是否进入该分支。
        }
        order = [label for label, _value in sorted(diffusion.items(), key=lambda item: item[1], reverse=True)]  # 声明 layers 的空间顺序。
        block_orders.append(  # 开始一个多行函数调用或配置块。
            {
                "block_index": block.get("block_index"),
                "time_start_ps": block.get("time_start_ps"),
                "time_end_ps": block.get("time_end_ps"),
                "observed_order_fast_to_slow": order,
            }
        )
        if len(expected_present) >= 2 and all(label in diffusion for label in expected_present):  # 根据当前状态决定是否进入该分支。
            expected_order_matches.append([label for label in order if label in expected_present] == expected_present)

    pairwise_expected = []  # 设置中间变量或可调参数，供后续工作流使用。
    for fast, slow in zip(expected_present, expected_present[1:]):  # 遍历当前工作流中的一组对象或任务。
        comparisons = []  # 设置中间变量或可调参数，供后续工作流使用。
        ratios = []  # 设置中间变量或可调参数，供后续工作流使用。
        for block in valid_blocks:  # 遍历当前工作流中的一组对象或任务。
            diffusion = block.get("diffusion_m2_s") or {}  # 设置中间变量或可调参数，供后续工作流使用。
            try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
                fast_d = diffusion.get(fast)  # 设置中间变量或可调参数，供后续工作流使用。
                slow_d = diffusion.get(slow)  # 设置中间变量或可调参数，供后续工作流使用。
                if fast_d is None or slow_d is None:  # 根据当前状态决定是否进入该分支。
                    continue
                fast_f = float(fast_d)  # 设置中间变量或可调参数，供后续工作流使用。
                slow_f = float(slow_d)  # 设置中间变量或可调参数，供后续工作流使用。
                if not (np.isfinite(fast_f) and np.isfinite(slow_f)):  # 根据当前状态决定是否进入该分支。
                    continue
                comparisons.append(fast_f > slow_f)
                if slow_f != 0.0:  # 根据当前状态决定是否进入该分支。
                    ratios.append(fast_f / slow_f)
            except Exception:  # 捕获异常并转成更清楚的示例错误信息。
                continue
        pairwise_expected.append(  # 开始一个多行函数调用或配置块。
            {
                "faster": fast,
                "slower": slow,
                "n_valid_blocks": len(comparisons),
                "ok_fraction": float(np.mean(comparisons)) if comparisons else None,
                "mean_ratio": float(np.mean(ratios)) if ratios else None,
            }
        )
    order_counts: dict[str, int] = {}  # 设置中间变量或可调参数，供后续工作流使用。
    for row in block_orders:  # 遍历当前工作流中的一组对象或任务。
        key = ">".join(str(x) for x in (row.get("observed_order_fast_to_slow") or []))  # 设置中间变量或可调参数，供后续工作流使用。
        if key:  # 根据当前状态决定是否进入该分支。
            order_counts[key] = int(order_counts.get(key, 0) + 1)  # 设置中间变量或可调参数，供后续工作流使用。
    pairwise_fractions = [  # 设置中间变量或可调参数，供后续工作流使用。
        float(row["ok_fraction"])
        for row in pairwise_expected  # 遍历当前工作流中的一组对象或任务。
        if row.get("ok_fraction") is not None  # 根据当前状态决定是否进入该分支。
    ]
    match_fraction = float(np.mean(expected_order_matches)) if expected_order_matches else None  # 设置中间变量或可调参数，供后续工作流使用。
    if len(expected_present) < 2 or not pairwise_fractions:  # 根据当前状态决定是否进入该分支。
        ranking_confidence = "not_applicable"  # 设置中间变量或可调参数，供后续工作流使用。
    elif all(frac >= 0.75 for frac in pairwise_fractions) and (match_fraction is None or match_fraction >= 0.75):  # 继续判断另一个互斥分支。
        ranking_confidence = "supports_expected"  # 设置中间变量或可调参数，供后续工作流使用。
    elif any(0.25 < frac < 0.75 for frac in pairwise_fractions):  # 继续判断另一个互斥分支。
        ranking_confidence = "ambiguous"  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        ranking_confidence = "contradicts_expected"  # 设置中间变量或可调参数，供后续工作流使用。
    return {  # 返回该辅助函数的结果。
        "status": "ok",
        "n_blocks": len(blocks),
        "n_valid_blocks": len(valid_blocks),
        "species": species_stats,
        "block_orders_fast_to_slow": block_orders,
        "block_order_counts": order_counts,
        "expected_order_fast_to_slow": list(expected_order),
        "expected_order_for_present_species": expected_present,
        "expected_order_match_fraction": match_fraction,
        "pairwise_expected": pairwise_expected,
        "ranking_confidence": ranking_confidence,
        "blocks": blocks,
    }


def _msd_block_diffusion_diagnostic(  # 定义本例内部辅助函数，组织重复步骤。
    analy: AnalyzeResult,
    *,
    full_msd: dict[str, Any],
    n_blocks: int,
    min_block_ps: float = 500.0,  # 设置中间变量或可调参数，供后续工作流使用。
    labels: tuple[str, ...] = ("EC", "EMC", "DEC"),  # 设置中间变量或可调参数，供后续工作流使用。
) -> dict[str, Any]:
    import numpy as np  # 导入本例需要的库或 yadonpy 接口。

    n_blocks = int(max(0, n_blocks))  # 设置中间变量或可调参数，供后续工作流使用。
    if n_blocks < 2:  # 根据当前状态决定是否进入该分支。
        return {"status": "skipped", "reason": "MSD_BLOCKS<2", "n_blocks_requested": n_blocks}  # 返回该辅助函数的结果。
    start_ps, end_ps = _default_msd_trajectory_bounds(full_msd, labels=labels)  # 设置中间变量或可调参数，供后续工作流使用。
    if start_ps is None or end_ps is None or end_ps <= start_ps:  # 根据当前状态决定是否进入该分支。
        return {"status": "skipped", "reason": "trajectory_time_bounds_unavailable", "n_blocks_requested": n_blocks}  # 返回该辅助函数的结果。
    duration_ps = float(end_ps - start_ps)  # 设置中间变量或可调参数，供后续工作流使用。
    block_count = min(n_blocks, int(np.floor(duration_ps / max(float(min_block_ps), 1.0e-12))))  # 设置中间变量或可调参数，供后续工作流使用。
    if block_count < 2:  # 根据当前状态决定是否进入该分支。
        return {  # 返回该辅助函数的结果。
            "status": "skipped",
            "reason": "trajectory_too_short_for_blocks",
            "duration_ps": duration_ps,
            "min_block_ps": float(min_block_ps),
        }
    topo = parse_system_top(Path(analy.top))  # 设置中间变量或可调参数，供后续工作流使用。
    system_dir = analy._system_dir()  # 设置中间变量或可调参数，供后续工作流使用。
    metric_catalog = build_msd_metric_catalog(topo, system_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    xtc_path = analy._analysis_xtc_path()  # 设置中间变量或可调参数，供后续工作流使用。
    catalog_by_lower = {str(key).lower(): (key, value) for key, value in metric_catalog.items()}  # 设置中间变量或可调参数，供后续工作流使用。
    transport = full_msd.get("_transport") if isinstance(full_msd.get("_transport"), dict) else {}  # 设置中间变量或可调参数，供后续工作流使用。
    geometry_mode = str(transport.get("geometry_mode") or "auto")  # 设置中间变量或可调参数，供后续工作流使用。
    unwrap = str(transport.get("unwrap") or "auto")  # 设置中间变量或可调参数，供后续工作流使用。
    drift = str(transport.get("drift") or "auto")  # 设置中间变量或可调参数，供后续工作流使用。
    edges = np.linspace(float(start_ps), float(end_ps), int(block_count) + 1)  # 设置中间变量或可调参数，供后续工作流使用。
    blocks: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for block_idx in range(int(block_count)):  # 遍历当前工作流中的一组对象或任务。
        begin = float(edges[block_idx])  # 设置中间变量或可调参数，供后续工作流使用。
        end = float(edges[block_idx + 1])  # 设置中间变量或可调参数，供后续工作流使用。
        block: dict[str, Any] = {  # 设置中间变量或可调参数，供后续工作流使用。
            "block_index": int(block_idx),
            "time_start_ps": begin,
            "time_end_ps": end,
            "duration_ps": float(end - begin),
            "diffusion_m2_s": {},
            "fit_status": {},
            "fit_confidence": {},
            "errors": {},
        }
        for label in labels:  # 遍历当前工作流中的一组对象或任务。
            catalog_item = catalog_by_lower.get(str(label).lower())  # 设置中间变量或可调参数，供后续工作流使用。
            if catalog_item is None:  # 根据当前状态决定是否进入该分支。
                block["errors"][label] = "species_not_found"  # 设置中间变量或可调参数，供后续工作流使用。
                continue
            _moltype, entry = catalog_item  # 设置中间变量或可调参数，供后续工作流使用。
            metric_name = str(entry.get("default_metric") or "")  # 设置中间变量或可调参数，供后续工作流使用。
            metric_entry = (entry.get("metrics") or {}).get(metric_name)  # 设置中间变量或可调参数，供后续工作流使用。
            group_specs = list((metric_entry or {}).get("groups") or [])  # 设置中间变量或可调参数，供后续工作流使用。
            if not metric_name or not group_specs:  # 根据当前状态决定是否进入该分支。
                block["errors"][label] = "default_metric_or_groups_missing"  # 设置中间变量或可调参数，供后续工作流使用。
                continue
            try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
                metric_data = compute_msd_series(  # 设置中间变量或可调参数，供后续工作流使用。
                    gro_path=system_dir / "system.gro",  # 设置中间变量或可调参数，供后续工作流使用。
                    xtc_path=xtc_path,  # 设置中间变量或可调参数，供后续工作流使用。
                    top_path=Path(analy.top),  # 设置中间变量或可调参数，供后续工作流使用。
                    system_dir=system_dir,  # 设置中间变量或可调参数，供后续工作流使用。
                    group_specs=group_specs,  # 设置中间变量或可调参数，供后续工作流使用。
                    geometry_mode=geometry_mode,  # 设置中间变量或可调参数，供后续工作流使用。
                    unwrap=unwrap,  # 设置中间变量或可调参数，供后续工作流使用。
                    drift=drift,  # 设置中间变量或可调参数，供后续工作流使用。
                    begin_ps=begin,  # 设置中间变量或可调参数，供后续工作流使用。
                    end_ps=end,  # 设置中间变量或可调参数，供后续工作流使用。
                )
                fit = dict(metric_data.get("fit") or {})  # 设置中间变量或可调参数，供后续工作流使用。
                d_val = fit.get("D_m2_s")  # 设置中间变量或可调参数，供后续工作流使用。
                block["diffusion_m2_s"][str(label)] = float(d_val) if d_val is not None else None  # 设置中间变量或可调参数，供后续工作流使用。
                block["fit_status"][str(label)] = fit.get("status")  # 设置中间变量或可调参数，供后续工作流使用。
                block["fit_confidence"][str(label)] = fit.get("confidence")  # 设置中间变量或可调参数，供后续工作流使用。
            except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
                block["errors"][label] = str(exc)  # 设置中间变量或可调参数，供后续工作流使用。
        block["observed_order_fast_to_slow"] = [  # 设置中间变量或可调参数，供后续工作流使用。
            label
            for label, value in sorted(  # 遍历当前工作流中的一组对象或任务。
                {
                    str(label): value
                    for label, value in (block.get("diffusion_m2_s") or {}).items()  # 遍历当前工作流中的一组对象或任务。
                    if value is not None  # 根据当前状态决定是否进入该分支。
                }.items(),
                key=lambda item: float(item[1]),  # 设置中间变量或可调参数，供后续工作流使用。
                reverse=True,  # 设置中间变量或可调参数，供后续工作流使用。
            )
        ]
        blocks.append(block)
    summary = _summarize_msd_block_diffusion(blocks)  # 设置中间变量或可调参数，供后续工作流使用。
    summary.update(  # 开始一个多行函数调用或配置块。
        {
            "n_blocks_requested": int(n_blocks),
            "min_block_ps": float(min_block_ps),
            "trajectory_time_start_ps": float(start_ps),
            "trajectory_time_end_ps": float(end_ps),
            "trajectory_duration_ps": duration_ps,
        }
    )
    return summary  # 返回该辅助函数的结果。


def _extract_rdf_site(rdf: dict[str, Any], site_id: str) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    block = rdf.get(site_id)  # 设置中间变量或可调参数，供后续工作流使用。
    return dict(block) if isinstance(block, dict) else {}  # 返回该辅助函数的结果。


def _extract_primary_oxygen_site(rdf: dict[str, Any], moltype: str) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    token = str(moltype or "").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    for site_id in (f"{token}:carbonyl_oxygen", f"{token}:oxygen_site"):  # 遍历当前工作流中的一组对象或任务。
        block = _extract_rdf_site(rdf, site_id)  # 设置中间变量或可调参数，供后续工作流使用。
        if block:  # 根据当前状态决定是否进入该分支。
            return block  # 返回该辅助函数的结果。
    return {}  # 返回该辅助函数的结果。


def _coordination_preference_summary(coordination: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    labels = {  # 设置中间变量或可调参数，供后续工作流使用。
        "EC": "EC_carbonyl_oxygen",
        "EMC": "EMC_carbonyl_oxygen",
        "DEC": "DEC_carbonyl_oxygen",
    }
    cn_by_species: dict[str, float] = {}  # 设置中间变量或可调参数，供后续工作流使用。
    for species, key in labels.items():  # 遍历当前工作流中的一组对象或任务。
        block = coordination.get(key)  # 设置中间变量或可调参数，供后续工作流使用。
        if not isinstance(block, dict):  # 根据当前状态决定是否进入该分支。
            continue
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            cn = float(block.get("cn_shell"))  # 设置中间变量或可调参数，供后续工作流使用。
        except Exception:  # 捕获异常并转成更清楚的示例错误信息。
            continue
        cn_by_species[species] = cn  # 设置中间变量或可调参数，供后续工作流使用。

    total_cn = sum(cn_by_species.values())  # 设置中间变量或可调参数，供后续工作流使用。
    total_count = sum(int(counts.get(species, 0) or 0) for species in labels)  # 设置中间变量或可调参数，供后续工作流使用。
    out: dict[str, Any] = {  # 设置中间变量或可调参数，供后续工作流使用。
        "total_cn_shell": total_cn,
        "notes": "shell_fraction is the fraction of first-shell carbonyl coordination; enrichment_vs_bulk > 1 means over-represented versus bulk solvent composition.",
    }
    for species, cn in cn_by_species.items():  # 遍历当前工作流中的一组对象或任务。
        bulk_count = int(counts.get(species, 0) or 0)  # 设置中间变量或可调参数，供后续工作流使用。
        bulk_fraction = (bulk_count / total_count) if total_count > 0 else None  # 设置中间变量或可调参数，供后续工作流使用。
        shell_fraction = (cn / total_cn) if total_cn > 0 else None  # 设置中间变量或可调参数，供后续工作流使用。
        enrichment = None  # 设置中间变量或可调参数，供后续工作流使用。
        if bulk_fraction and shell_fraction is not None and bulk_fraction > 0:  # 根据当前状态决定是否进入该分支。
            enrichment = shell_fraction / bulk_fraction  # 设置中间变量或可调参数，供后续工作流使用。
        out[species] = {  # 设置中间变量或可调参数，供后续工作流使用。
            "cn_shell": cn,
            "bulk_fraction": bulk_fraction,
            "shell_fraction": shell_fraction,
            "enrichment_vs_bulk": enrichment,
        }
    return out  # 返回该辅助函数的结果。


def _neutral_charge_anomalies(species_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:  # 定义本例内部辅助函数，组织重复步骤。
    issues: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for row in species_rows:  # 遍历当前工作流中的一组对象或任务。
        label = str(row.get("label") or "")  # 给该选区一个可读标签，便于 manifest 检查。
        if label in {"Li", "PF6"}:  # 根据当前状态决定是否进入该分支。
            continue
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            net_q = float(row.get("net_charge_e") or 0.0)  # 设置中间变量或可调参数，供后续工作流使用。
        except Exception:  # 捕获异常并转成更清楚的示例错误信息。
            net_q = 0.0  # 设置中间变量或可调参数，供后续工作流使用。
        if abs(net_q) > 1.0e-6:  # 根据当前状态决定是否进入该分支。
            issues.append(  # 开始一个多行函数调用或配置块。
                {
                    "label": label,
                    "net_charge_e": net_q,
                    "note": "Neutral electrolyte species should normally remain charge-neutral.",
                }
            )
    return issues  # 返回该辅助函数的结果。


def _stamp_charge_route(mol, *, charge_method: str, prefer_db: bool, require_db: bool, require_ready: bool) -> None:  # 定义本例内部辅助函数，组织重复步骤。
    mol.SetProp("_yadonpy_charge_method", str(charge_method))
    mol.SetProp("_yadonpy_prefer_db", "1" if prefer_db else "0")
    mol.SetProp("_yadonpy_require_db", "1" if require_db else "0")
    mol.SetProp("_yadonpy_require_ready", "1" if require_ready else "0")


BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"  # 设置中间变量或可调参数，供后续工作流使用。

restart_status = _env_bool("RESTART_STATUS", False)  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

species_only = _env_bool("SPECIES_ONLY", False)  # 设置中间变量或可调参数，供后续工作流使用。
build_only = _env_bool("BUILD_ONLY", False)  # 设置中间变量或可调参数，供后续工作流使用。
export_only = _env_bool("EXPORT_ONLY", False)  # 设置中间变量或可调参数，供后续工作流使用。

oplsaa_profile = _env_text("OPLSAA_PROFILE", "strict")  # 设置中间变量或可调参数，供后续工作流使用。
gpu_offload_mode = _env_text("GPU_OFFLOAD_MODE", "auto")  # 设置中间变量或可调参数，供后续工作流使用。

ff = OPLSAA(profile=oplsaa_profile)  # 选择有机分子/聚合物/部分无机离子的力场对象。
ion_ff = OPLSAA(profile=oplsaa_profile)  # 选择单原子离子参数来源。

charge_mode = _normalize_charge_mode(os.environ.get("YADONPY_OPLS_CHARGE_MODE"))  # 设置中间变量或可调参数，供后续工作流使用。

EC_SMILES = "O=C1OCCO1"  # 设置中间变量或可调参数，供后续工作流使用。
EMC_SMILES = "CCOC(=O)OC"  # 设置中间变量或可调参数，供后续工作流使用。
DEC_SMILES = "CCOC(=O)OCC"  # 设置中间变量或可调参数，供后续工作流使用。
LI_SMILES = "[Li+]"  # 设置中间变量或可调参数，供后续工作流使用。
PF6_SMILES = "F[P-](F)(F)(F)(F)F"  # 设置中间变量或可调参数，供后续工作流使用。

temp_k = _env_float("TEMP_K", 298.15)  # 设置中间变量或可调参数，供后续工作流使用。
press_bar = _env_float("PRESS_BAR", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
prod_ns = _env_float("PROD_NS", 5.0)  # 设置中间变量或可调参数，供后续工作流使用。
prod_ensemble = _normalize_production_ensemble(os.environ.get("PROD_ENSEMBLE"))  # 设置中间变量或可调参数，供后续工作流使用。
initial_density_g_cm3 = _env_float("INITIAL_DENSITY_G_CM3", 0.05)  # 设置中间变量或可调参数，供后续工作流使用。
max_additional_rounds = _env_int("MAX_ADDITIONAL_ROUNDS", 4)  # 设置中间变量或可调参数，供后续工作流使用。
prod_constraints = _normalize_constraints(os.environ.get("PROD_CONSTRAINTS"), default="h-bonds")  # 设置中间变量或可调参数，供后续工作流使用。
prod_dt_ps = _env_float("PROD_DT_PS", 0.002)  # 设置中间变量或可调参数，供后续工作流使用。
msd_blocks = _env_int("MSD_BLOCKS", 4)  # 设置中间变量或可调参数，供后续工作流使用。
msd_block_min_ps = _env_float("MSD_BLOCK_MIN_PS", 500.0)  # 设置中间变量或可调参数，供后续工作流使用。

mpi = _env_int("MPI", 1)  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = _env_int("OMP", 16)  # 设置每个 rank 的 OpenMP 线程数。
gpu = _env_int("GPU", 1)  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = _env_int("GPU_ID", 0)  # 选择 GPU 设备编号，多卡节点可修改。

count_ec = _env_int("COUNT_EC", 40)  # 设置中间变量或可调参数，供后续工作流使用。
count_emc = _env_int("COUNT_EMC", 50)  # 设置中间变量或可调参数，供后续工作流使用。
count_dec = _env_int("COUNT_DEC", 20)  # 设置中间变量或可调参数，供后续工作流使用。
salt_pairs = _env_int("SALT_PAIRS", 15)  # 设置盐离子对数；阳离子和阴离子应同步增减。

li_charge_scale = _env_float("LI_CHARGE_SCALE", 0.8)  # 设置中间变量或可调参数，供后续工作流使用。
pf6_charge_scale = _env_float("PF6_CHARGE_SCALE", 0.8)  # 设置中间变量或可调参数，供后续工作流使用。

work_dir_name = _env_text("WORK_DIR_NAME", "benchmark_carbonate_lipf6_oplsaa_work")  # 设置中间变量或可调参数，供后续工作流使用。
work_root = Path(_env_text("WORK_DIR", str(BASE_DIR / work_dir_name))).resolve()  # 设置中间变量或可调参数，供后续工作流使用。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    work_root = workdir(work_root, restart=restart_status)  # 创建或复用本例工作目录。
    build_dir = work_root.child("00_build_cell")  # 设置中间变量或可调参数，供后续工作流使用。

    ec = _load_ready_opls_species(ff, EC_SMILES, label="EC", repo_db_dir=REPO_DB_DIR, charge_mode=charge_mode)  # 设置中间变量或可调参数，供后续工作流使用。
    emc = _load_ready_opls_species(ff, EMC_SMILES, label="EMC", repo_db_dir=REPO_DB_DIR, charge_mode=charge_mode)  # 设置中间变量或可调参数，供后续工作流使用。
    dec = _load_ready_opls_species(ff, DEC_SMILES, label="DEC", repo_db_dir=REPO_DB_DIR, charge_mode=charge_mode)  # 设置中间变量或可调参数，供后续工作流使用。
    li = _assign_builtin_opls_ion(ion_ff, LI_SMILES, label="Li")  # 设置中间变量或可调参数，供后续工作流使用。
    pf6 = _load_pf6_with_opls_parameters(ion_ff=ion_ff, repo_db_dir=REPO_DB_DIR, charge_mode=charge_mode)  # 设置中间变量或可调参数，供后续工作流使用。

    solvent_charge_method = "RESP2" if charge_mode == "resp2" else "RESP"
    pf6_charge_method = "opls" if charge_mode == "opls" else "RESP"
    _stamp_charge_route(ec, charge_method=solvent_charge_method, prefer_db=True, require_db=True, require_ready=True)  # 设置中间变量或可调参数，供后续工作流使用。
    _stamp_charge_route(emc, charge_method=solvent_charge_method, prefer_db=True, require_db=True, require_ready=True)  # 设置中间变量或可调参数，供后续工作流使用。
    _stamp_charge_route(dec, charge_method=solvent_charge_method, prefer_db=True, require_db=True, require_ready=True)  # 设置中间变量或可调参数，供后续工作流使用。
    _stamp_charge_route(li, charge_method="opls", prefer_db=False, require_db=False, require_ready=False)  # 设置中间变量或可调参数，供后续工作流使用。
    _stamp_charge_route(pf6, charge_method=pf6_charge_method, prefer_db=True, require_db=True, require_ready=True)  # 设置中间变量或可调参数，供后续工作流使用。

    species_rows = [  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(ec, label="EC", moltype_hint="EC", charge_scale=1.0),  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(emc, label="EMC", moltype_hint="EMC", charge_scale=1.0),  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(dec, label="DEC", moltype_hint="DEC", charge_scale=1.0),  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(li, label="Li", moltype_hint="Li", charge_scale=li_charge_scale),  # 设置中间变量或可调参数，供后续工作流使用。
        summarize_rdkit_species_forcefield(pf6, label="PF6", moltype_hint="PF6", charge_scale=pf6_charge_scale),  # 设置中间变量或可调参数，供后续工作流使用。
    ]
    neutral_charge_issues = _neutral_charge_anomalies(species_rows)  # 设置中间变量或可调参数，供后续工作流使用。
    solvent_routes = {  # 设置中间变量或可调参数，供后续工作流使用。
        "EC": _extract_resp_route(ec, label="EC"),
        "EMC": _extract_resp_route(emc, label="EMC"),
        "DEC": _extract_resp_route(dec, label="DEC"),
    }
    equivalence_spread = {  # 设置中间变量或可调参数，供后续工作流使用。
        "EC": _equivalence_spread_diagnostic(ec, label="EC"),
        "EMC": _equivalence_spread_diagnostic(emc, label="EMC"),
        "DEC": _equivalence_spread_diagnostic(dec, label="DEC"),
    }

    species_summary = {  # 设置中间变量或可调参数，供后续工作流使用。
        "metadata": {
            "benchmark_name": "carbonate_lipf6_oplsaa",
            "charge_mode": charge_mode,
            "oplsaa_profile": oplsaa_profile,
            "gpu_offload_mode": gpu_offload_mode,
            "solvent_charge_method": solvent_charge_method,
            "resolved_qm_recipes": solvent_routes,
            "pf6_charge_method": pf6_charge_method,
            "species": ["EC", "EMC", "DEC", "Li", "PF6"],
            "counts": {"EC": count_ec, "EMC": count_emc, "DEC": count_dec, "Li": salt_pairs, "PF6": salt_pairs},
            "charge_scale": {"EC": 1.0, "EMC": 1.0, "DEC": 1.0, "Li": li_charge_scale, "PF6": pf6_charge_scale},
            "production_ensemble": prod_ensemble,
            "expected_diffusion_trend": "EMC > DEC > EC (literature-guided target for mixed linear/cyclic carbonate electrolyte)",
            "neutral_charge_issues": neutral_charge_issues,
        },
        "species_pre_export": species_rows,
        "species_equivalence_spread": equivalence_spread,
    }
    analysis_dir = work_root / "06_analysis"  # 设置中间变量或可调参数，供后续工作流使用。
    analysis_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    _dump_json(analysis_dir / "species_forcefield_summary.json", species_summary)

    if species_only:  # 根据当前状态决定是否进入该分支。
        print("[SPECIES-ONLY] Wrote species_forcefield_summary.json")  # 打印关键路径或状态，便于人工检查。
        print(json.dumps(species_summary["metadata"], indent=2, ensure_ascii=False))  # 打印关键路径或状态，便于人工检查。
        raise SystemExit(0)  # 关键步骤失败时立即报错，避免继续生成错误结果。

    cell_mols = []  # 设置中间变量或可调参数，供后续工作流使用。
    counts = []  # 设置各 species 的数量；顺序必须和 species 列表一致。
    charge_scale = []  # 设置电荷缩放系数；1.0 表示全电荷模型。
    for mol, count in ((ec, count_ec), (emc, count_emc), (dec, count_dec)):  # 遍历当前工作流中的一组对象或任务。
        if int(count) > 0:  # 根据当前状态决定是否进入该分支。
            cell_mols.append(mol)
            counts.append(int(count))
            charge_scale.append(1.0)
    if salt_pairs > 0:  # 根据当前状态决定是否进入该分支。
        cell_mols.extend([li, pf6])
        counts.extend([salt_pairs, salt_pairs])
        charge_scale.extend([li_charge_scale, pf6_charge_scale])
    if not cell_mols:  # 根据当前状态决定是否进入该分支。
        raise ValueError("At least one molecule count must be positive.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    ac = poly.amorphous_cell(  # 构建无定形混合体系初始盒子。
        cell_mols,
        counts,
        charge_scale=charge_scale,  # 设置电荷缩放系数；1.0 表示全电荷模型。
        density=initial_density_g_cm3,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=build_dir,  # 设置本例输出目录。
    )

    if build_only:  # 根据当前状态决定是否进入该分支。
        print(f"[BUILD-ONLY] Finished cell construction at {build_dir}")  # 打印关键路径或状态，便于人工检查。
        raise SystemExit(0)  # 关键步骤失败时立即报错，避免继续生成错误结果。

    eqmd = eq.LiquidAnneal(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
    if export_only:  # 根据当前状态决定是否进入该分支。
        exported = eqmd.ensure_system_exported()  # 设置中间变量或可调参数，供后续工作流使用。
        print(f"[EXPORT-ONLY] Exported 02_system at {exported.system_top.parent}")  # 打印关键路径或状态，便于人工检查。
        raise SystemExit(0)  # 关键步骤失败时立即报错，避免继续生成错误结果。

    ac = eqmd.exec(  # 设置中间变量或可调参数，供后续工作流使用。
        temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
        press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
        mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
        gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
        gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
        gpu_offload_mode=gpu_offload_mode,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    latest_equilibrated_gro = eqmd.final_gro()  # 设置中间变量或可调参数，供后续工作流使用。
    analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    analy.get_all_prop(temp=temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    for _ in range(max_additional_rounds):  # 遍历当前工作流中的一组对象或任务。
        if result:  # 根据当前状态决定是否进入该分支。
            break
        eq_more = eq.Additional(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = eq_more.exec(  # 设置中间变量或可调参数，供后续工作流使用。
            temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
            press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
            gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
            gpu_offload_mode=gpu_offload_mode,  # 设置中间变量或可调参数，供后续工作流使用。
            constraints="none",  # 设置约束策略。
            start_gro=latest_equilibrated_gro,  # 设置中间变量或可调参数，供后续工作流使用。
            skip_rebuild=True,  # 设置中间变量或可调参数，供后续工作流使用。
            micro_relax=False,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        latest_equilibrated_gro = eq_more.final_gro()  # 设置中间变量或可调参数，供后续工作流使用。
        analy = eq_more.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        analy.get_all_prop(temp=temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    if prod_ensemble == "nvt":  # 根据当前状态决定是否进入该分支。
        prod_step = eq.NVT(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = prod_step.exec(  # 设置中间变量或可调参数，供后续工作流使用。
            temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
            gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
            time=prod_ns,  # 设置中间变量或可调参数，供后续工作流使用。
            dt_ps=prod_dt_ps,  # 设置 MD 时间步长，单位 ps。
            constraints=prod_constraints,  # 设置约束策略。
            gpu_offload_mode=gpu_offload_mode,  # 设置中间变量或可调参数，供后续工作流使用。
        )
    else:  # 处理前面条件都不满足的情况。
        prod_step = eq.NPT(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = prod_step.exec(  # 设置中间变量或可调参数，供后续工作流使用。
            temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
            press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
            gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
            time=prod_ns,  # 设置中间变量或可调参数，供后续工作流使用。
            dt_ps=prod_dt_ps,  # 设置 MD 时间步长，单位 ps。
            constraints=prod_constraints,  # 设置约束策略。
            gpu_offload_mode=gpu_offload_mode,  # 设置中间变量或可调参数，供后续工作流使用。
        )

    analy = prod_step.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    basic = analy.get_all_prop(temp=temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    msd_species = [ec, emc, dec]  # 设置中间变量或可调参数，供后续工作流使用。
    if salt_pairs > 0:  # 根据当前状态决定是否进入该分支。
        msd_species.extend([li, pf6])
    msd = analy.msd(mols=msd_species)  # 设置中间变量或可调参数，供后续工作流使用。
    msd_block_diagnostic = _msd_block_diffusion_diagnostic(  # 设置中间变量或可调参数，供后续工作流使用。
        analy,
        full_msd=msd,  # 设置中间变量或可调参数，供后续工作流使用。
        n_blocks=msd_blocks,  # 设置中间变量或可调参数，供后续工作流使用。
        min_block_ps=msd_block_min_ps,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    _dump_json(analysis_dir / "msd_block_diffusion.json", msd_block_diagnostic)
    if salt_pairs > 0:  # 根据当前状态决定是否进入该分支。
        rdf = analy.rdf(center_mol=li)  # 设置中间变量或可调参数，供后续工作流使用。
        sigma = analy.sigma(msd=msd, temp_k=temp_k, eh_mode="gmx_current_only")  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        rdf = {}  # 设置中间变量或可调参数，供后续工作流使用。
        sigma = {  # 设置中间变量或可调参数，供后续工作流使用。
            "sigma_ne_upper_bound_S_m": None,
            "sigma_eh_total_S_m": None,
            "haven_ratio": None,
            "eh": {
                "confidence": "not_applicable",
                "quality_note": "No salt pairs were requested; skipped Li-centered RDF and conductivity analysis.",
            },
        }

    summary = {  # 设置中间变量或可调参数，供后续工作流使用。
        "metadata": species_summary["metadata"],
        "basic_properties": basic.get("basic_properties", {}),
        "equilibration_ok": bool(result),
        "transport_confidence": "high" if result else "low_density_not_converged",
        "diffusion_m2_s": {
            "EC": _extract_default_diffusivity(msd, "EC"),
            "EMC": _extract_default_diffusivity(msd, "EMC"),
            "DEC": _extract_default_diffusivity(msd, "DEC"),
            "Li": _extract_default_diffusivity(msd, "Li"),
            "PF6": _extract_default_diffusivity(msd, "PF6"),
        },
        "msd_block_diffusion_diagnostic": msd_block_diagnostic,
        "conductivity": {
            "sigma_ne_upper_bound_S_m": sigma.get("sigma_ne_upper_bound_S_m"),
            "sigma_eh_total_S_m": sigma.get("sigma_eh_total_S_m"),
            "haven_ratio": sigma.get("haven_ratio"),
            "eh_confidence": ((sigma.get("eh") or {}) if isinstance(sigma.get("eh"), dict) else {}).get("confidence"),
            "eh_quality_note": ((sigma.get("eh") or {}) if isinstance(sigma.get("eh"), dict) else {}).get("quality_note"),
        },
        "coordination": {
            "EC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "ec"),
            "EMC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "emc"),
            "DEC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "dec"),
            "PF6_coordination_fluorine": _extract_rdf_site(rdf, "pf6:coordination_fluorine"),
            "PF6_fluorine_site": _extract_rdf_site(rdf, "pf6:fluorine_site"),
        },
    }
    summary["coordination_preference"] = _coordination_preference_summary(  # 设置中间变量或可调参数，供后续工作流使用。
        summary["coordination"],
        {"EC": count_ec, "EMC": count_emc, "DEC": count_dec},
    )
    _dump_json(analysis_dir / "benchmark_summary.json", summary)
    print("[BENCHMARK] carbonate_lipf6_oplsaa completed")  # 打印关键路径或状态，便于人工检查。
    print(json.dumps(summary["diffusion_m2_s"], indent=2, ensure_ascii=False))  # 打印关键路径或状态，便于人工检查。
