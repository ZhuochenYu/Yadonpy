from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

import json  # 导入本例需要的库或 yadonpy 接口。
import os  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。
from typing import Any  # 导入本例需要的库或 yadonpy 接口。

import numpy as np  # 导入本例需要的库或 yadonpy 接口。
from rdkit import Chem  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.core import poly, utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.metadata import (  # 导入本例需要的库或 yadonpy 接口。
    QM_RECIPE_PROP,
    RESP_CONSTRAINTS_PROP,
    RESP_PROFILE_PROP,
    read_json_prop,
    read_text_prop,
)
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff import GAFF2, GAFF2_mod, MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.gmx.analysis.structured import build_msd_metric_catalog, compute_msd_series  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.gmx.topology import parse_system_top  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.analyzer import AnalyzeResult  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim import qm  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.benchmarking import _dump_json, summarize_rdkit_species_forcefield  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.performance import resolve_io_analysis_policy  # 导入本例需要的库或 yadonpy 接口。
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


def _normalize_charge_mode(raw: str | None) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    mode = str(raw or "resp").strip().lower()  # 设置该配置块使用的计算模式。
    if mode in {"resp2", "resp_2"}:  # 根据当前状态决定是否进入该分支。
        raise ValueError("This benchmark is intentionally restricted to GAFF2 + RESP. RESP2 is out of scope for this script.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return "resp"  # 返回该辅助函数的结果。


def _normalize_gaff_variant(raw: str | None) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    variant = str(raw or "classic").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if variant in {"mod", "gaff2_mod"}:  # 根据当前状态决定是否进入该分支。
        return "mod"  # 返回该辅助函数的结果。
    return "classic"  # 返回该辅助函数的结果。


def _normalize_resp_profile(raw: str | None) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    profile = str(raw or "adaptive").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if profile in {"default", "current"}:  # 根据当前状态决定是否进入该分支。
        profile = "adaptive"  # 设置中间变量或可调参数，供后续工作流使用。
    if profile not in {"adaptive", "legacy"}:  # 根据当前状态决定是否进入该分支。
        raise ValueError(f"Unsupported RESP profile: {raw!r}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return profile  # 返回该辅助函数的结果。


def _normalize_solvent_source(raw: str | None) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    source = str(raw or "qm").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if source in {"db", "moldb", "ready", "ready_db"}:  # 根据当前状态决定是否进入该分支。
        return "moldb"  # 返回该辅助函数的结果。
    if source not in {"qm", "moldb"}:  # 根据当前状态决定是否进入该分支。
        raise ValueError(f"Unsupported solvent source: {raw!r}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return source  # 返回该辅助函数的结果。


def _normalize_db_priority(raw: str | None) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    mode = str(raw or "auto").strip().lower()  # 设置该配置块使用的计算模式。
    if mode in {"repo", "repo_first", "local_first"}:  # 根据当前状态决定是否进入该分支。
        return "repo_first"  # 返回该辅助函数的结果。
    if mode in {"default", "default_first", "global_first"}:  # 根据当前状态决定是否进入该分支。
        return "default_first"  # 返回该辅助函数的结果。
    if mode != "auto":  # 根据当前状态决定是否进入该分支。
        raise ValueError(f"Unsupported MolDB priority: {raw!r}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return "auto"  # 返回该辅助函数的结果。


def _normalize_equilibration_mode(raw: str | None) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    mode = str(raw or "auto").strip().lower().replace("-", "_")  # 设置该配置块使用的计算模式。
    aliases = {  # 设置中间变量或可调参数，供后续工作流使用。
        "auto": "auto",
        "eq21": "eq21",
        "liquid": "liquid_anneal",
        "liquid_anneal": "liquid_anneal",
        "cemp": "liquid_anneal",
        "cemp_like": "liquid_anneal",
    }
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        return aliases[mode]  # 返回该辅助函数的结果。
    except KeyError as exc:  # 捕获异常并转成更清楚的示例错误信息。
        raise ValueError(f"Unsupported equilibration mode: {raw!r}") from exc  # 关键步骤失败时立即报错，避免继续生成错误结果。


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


def _load_equilibrium_payload(analysis_dir: Path) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    path = Path(analysis_dir) / "equilibrium.json"  # 设置中间变量或可调参数，供后续工作流使用。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        payload = json.loads(path.read_text(encoding="utf-8"))  # 设置中间变量或可调参数，供后续工作流使用。
        return payload if isinstance(payload, dict) else {}  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return {}  # 返回该辅助函数的结果。


def _restart_latest_equilibrated_gro(work_root: Path, fallback: Path) -> Path:  # 定义本例内部辅助函数，组织重复步骤。
    """Prefer the latest completed equilibration restart point, excluding production."""
    finder = getattr(eq, "_find_latest_equilibrated_gro", None)  # 设置中间变量或可调参数，供后续工作流使用。
    if not callable(finder):  # 根据当前状态决定是否进入该分支。
        return Path(fallback)  # 返回该辅助函数的结果。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        candidate = finder(Path(work_root), exclude_dirs=(Path(work_root) / "05_npt_production",))  # 设置中间变量或可调参数，供后续工作流使用。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        candidate = None  # 设置中间变量或可调参数，供后续工作流使用。
    if candidate is None:  # 根据当前状态决定是否进入该分支。
        return Path(fallback)  # 返回该辅助函数的结果。
    candidate = Path(candidate)  # 设置中间变量或可调参数，供后续工作流使用。
    if not candidate.exists():  # 根据当前状态决定是否进入该分支。
        return Path(fallback)  # 返回该辅助函数的结果。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        if candidate.resolve() == Path(fallback).resolve():  # 根据当前状态决定是否进入该分支。
            return Path(fallback)  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        pass
    print(f"[RESTART] Continuing additional equilibration from latest completed structure: {candidate}")  # 打印关键路径或状态，便于人工检查。
    return candidate  # 返回该辅助函数的结果。


def _analyze_restart_stage(work_root: Path, gro_path: Path) -> AnalyzeResult | None:  # 定义本例内部辅助函数，组织重复步骤。
    stage_dir = Path(gro_path).parent  # 设置中间变量或可调参数，供后续工作流使用。
    tpr = stage_dir / "md.tpr"  # 设置中间变量或可调参数，供后续工作流使用。
    xtc = stage_dir / "md.xtc"  # 设置中间变量或可调参数，供后续工作流使用。
    trr = stage_dir / "md.trr"  # 设置中间变量或可调参数，供后续工作流使用。
    traj = xtc if xtc.exists() else trr  # 设置中间变量或可调参数，供后续工作流使用。
    edr = stage_dir / "md.edr"  # 设置中间变量或可调参数，供后续工作流使用。
    top = Path(work_root) / "02_system" / "system.top"  # 设置中间变量或可调参数，供后续工作流使用。
    ndx = Path(work_root) / "02_system" / "system.ndx"  # 设置中间变量或可调参数，供后续工作流使用。
    if not (tpr.exists() and traj.exists() and edr.exists() and top.exists() and ndx.exists()):  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    return AnalyzeResult(  # 返回该辅助函数的结果。
        work_dir=Path(work_root),  # 设置本例输出目录。
        tpr=tpr,  # 设置中间变量或可调参数，供后续工作流使用。
        xtc=traj,  # 设置中间变量或可调参数，供后续工作流使用。
        edr=edr,  # 设置中间变量或可调参数，供后续工作流使用。
        top=top,  # 设置中间变量或可调参数，供后续工作流使用。
        ndx=ndx,  # 设置中间变量或可调参数，供后续工作流使用。
        trr=trr if trr.exists() else None,  # 设置中间变量或可调参数，供后续工作流使用。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
    )


def _transport_confidence_from_equilibrium(payload: dict[str, Any], ok: bool) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    density_gate = payload.get("density_gate") if isinstance(payload, dict) else None  # 设置中间变量或可调参数，供后续工作流使用。
    density_gate = density_gate if isinstance(density_gate, dict) else {}  # 设置中间变量或可调参数，供后续工作流使用。
    severity = str(density_gate.get("severity") or ("none" if ok else "high"))  # 设置中间变量或可调参数，供后续工作流使用。
    if ok:  # 根据当前状态决定是否进入该分支。
        confidence = "high"  # 设置中间变量或可调参数，供后续工作流使用。
    elif severity == "high":  # 继续判断另一个互斥分支。
        confidence = "low_density_not_converged"  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        confidence = "medium_density_not_converged"  # 设置中间变量或可调参数，供后续工作流使用。
    return {  # 返回该辅助函数的结果。
        "equilibration_ok": bool(ok),
        "density_warning_severity": severity,
        "transport_confidence": confidence,
        "density_gate": density_gate,
    }


def _charge_recipe_from_family(raw: str | None) -> dict[str, str]:  # 定义本例内部辅助函数，组织重复步骤。
    family = str(raw or "wb97m_v").strip().lower().replace("-", "_")  # 设置中间变量或可调参数，供后续工作流使用。
    recipes = {  # 设置中间变量或可调参数，供后续工作流使用。
        "b3lyp_d3bj": {
            "family": "b3lyp_d3bj",
            "label": "B3LYP-D3BJ/def2-TZVP",
            "opt_method": "b3lyp-d3bj",
            "charge_method": "b3lyp-d3bj",
        },
        "wb97m_v": {
            "family": "wb97m_v",
            "label": "wB97M-V/def2-TZVP",
            "opt_method": "wb97m-v",
            "charge_method": "wb97m-v",
        },
        "m06_2x": {
            "family": "m06_2x",
            "label": "M06-2X/def2-TZVP",
            "opt_method": "m06-2x",
            "charge_method": "m06-2x",
        },
        # Hidden compatibility alias for older local probes.
        "wb97m_d3bj": {
            "family": "wb97m_d3bj",
            "label": "wB97M-D3BJ/def2-TZVP",
            "opt_method": "wb97m-d3bj",
            "charge_method": "wb97m-d3bj",
        },
    }
    if family not in recipes:  # 根据当前状态决定是否进入该分支。
        family = "wb97m_v"  # 设置中间变量或可调参数，供后续工作流使用。
    recipe = dict(recipes[family])  # 设置中间变量或可调参数，供后续工作流使用。
    recipe.update(  # 开始一个多行函数调用或配置块。
        {
            "opt_basis": "def2-TZVP",
            "charge_basis": "def2-TZVP",
            "opt_basis_gen": {"Br": "def2-TZVP", "I": "def2-TZVP"},
            "charge_basis_gen": {"Br": "def2-TZVP", "I": "def2-TZVP"},
        }
    )
    return recipe  # 返回该辅助函数的结果。


def _build_ff_variant(variant: str):  # 定义本例内部辅助函数，组织重复步骤。
    if str(variant).strip().lower() == "mod":  # 根据当前状态决定是否进入该分支。
        return GAFF2_mod()  # 返回该辅助函数的结果。
    return GAFF2()  # 返回该辅助函数的结果。


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
    prop_names = ["AtomicCharge", "RESP", "RESP2", "ESP"]  # 设置中间变量或可调参数，供后续工作流使用。
    for group in groups:  # 遍历当前工作流中的一组对象或任务。
        idxs = sorted({int(i) for i in group})  # 设置中间变量或可调参数，供后续工作流使用。
        if len(idxs) <= 1:  # 根据当前状态决定是否进入该分支。
            continue
        spreads = {}  # 设置中间变量或可调参数，供后续工作流使用。
        for prop in prop_names:  # 遍历当前工作流中的一组对象或任务。
            values = []  # 设置中间变量或可调参数，供后续工作流使用。
            for idx in idxs:  # 遍历当前工作流中的一组对象或任务。
                atom = mol.GetAtomWithIdx(idx)  # 设置中间变量或可调参数，供后续工作流使用。
                if not atom.HasProp(prop):  # 根据当前状态决定是否进入该分支。
                    values = []  # 设置中间变量或可调参数，供后续工作流使用。
                    break
                values.append(float(atom.GetDoubleProp(prop)))
            if values:  # 根据当前状态决定是否进入该分支。
                spreads[prop] = float(max(values) - min(values))  # 设置中间变量或可调参数，供后续工作流使用。
        diagnostics.append(  # 开始一个多行函数调用或配置块。
            {
                "atom_indices": idxs,
                "symbols": [str(mol.GetAtomWithIdx(idx).GetSymbol()) for idx in idxs],
                "spreads_e": spreads,
            }
        )
    max_spread = 0.0  # 设置中间变量或可调参数，供后续工作流使用。
    for item in diagnostics:  # 遍历当前工作流中的一组对象或任务。
        for spread in item.get("spreads_e", {}).values():  # 遍历当前工作流中的一组对象或任务。
            max_spread = max(max_spread, float(spread))  # 设置中间变量或可调参数，供后续工作流使用。
    return {  # 返回该辅助函数的结果。
        "label": str(label),
        "group_count": len(diagnostics),
        "max_spread_e": float(max_spread),
        "groups": diagnostics,
    }


def _load_ready_gaff_species(  # 定义本例内部辅助函数，组织重复步骤。
    ff: GAFF2 | GAFF2_mod,
    smiles: str,
    *,
    label: str,
    repo_db_dir: Path,
    charge_mode: str,
    db_priority: str,
):
    last_exc: Exception | None = None  # 设置中间变量或可调参数，供后续工作流使用。
    db_charge = "RESP2" if charge_mode == "resp2" else "RESP"
    search_order = [(None, "default"), (repo_db_dir, "repo")]  # 设置中间变量或可调参数，供后续工作流使用。
    if db_priority == "repo_first":  # 根据当前状态决定是否进入该分支。
        search_order = [(repo_db_dir, "repo"), (None, "default")]  # 设置中间变量或可调参数，供后续工作流使用。
    for db_dir, db_label in search_order:  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            mol = ff.mol_rdkit(  # 设置中间变量或可调参数，供后续工作流使用。
                smiles,
                name=label,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                db_dir=db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
                charge=db_charge,  # 指定电荷来源或电荷计算方式。
                require_ready=True,  # 要求 MolDB 物种必须已准备好。
                prefer_db=True,  # 优先从 MolDB 读取已有结果。
            )
            mol = ff.ff_assign(mol, charge=None, report=False)  # 分配力场参数并写入分子属性。
            if not mol:  # 根据当前状态决定是否进入该分支。
                raise RuntimeError(f"Cannot assign {ff.name} parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
            print(f"[MolDB] loaded {label} with {db_charge} charges from {db_label} db")  # 打印关键路径或状态，便于人工检查。
            return mol  # 返回该辅助函数的结果。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            last_exc = exc  # 设置中间变量或可调参数，供后续工作流使用。
    raise RuntimeError(f"{label} is expected to be ready in MolDB for the GAFF2 benchmark.") from last_exc  # 关键步骤失败时立即报错，避免继续生成错误结果。


def _load_ready_pf6(ff: GAFF2 | GAFF2_mod, *, repo_db_dir: Path, db_priority: str):  # 定义本例内部辅助函数，组织重复步骤。
    last_exc: Exception | None = None  # 设置中间变量或可调参数，供后续工作流使用。
    search_order = [(None, "default"), (repo_db_dir, "repo")]  # 设置中间变量或可调参数，供后续工作流使用。
    if db_priority == "repo_first":  # 根据当前状态决定是否进入该分支。
        search_order = [(repo_db_dir, "repo"), (None, "default")]  # 设置中间变量或可调参数，供后续工作流使用。
    for db_dir, db_label in search_order:  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            mol = ff.mol_rdkit(  # 设置中间变量或可调参数，供后续工作流使用。
                PF6_SMILES,
                name="PF6",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                db_dir=db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
                charge="RESP",  # 指定电荷来源或电荷计算方式。
                require_ready=True,  # 要求 MolDB 物种必须已准备好。
                prefer_db=True,  # 优先从 MolDB 读取已有结果。
            )
            mol = ff.ff_assign(mol, charge=None, bonded="DRIH", report=False)  # 分配力场参数并写入分子属性。
            if not mol:  # 根据当前状态决定是否进入该分支。
                raise RuntimeError("Cannot assign PF6 parameters from MolDB-backed DRIH topology.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
            print(f"[MolDB] loaded PF6 with RESP charges from {db_label} db")  # 打印关键路径或状态，便于人工检查。
            return mol  # 返回该辅助函数的结果。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            last_exc = exc  # 设置中间变量或可调参数，供后续工作流使用。
    raise RuntimeError("PF6 is expected to be ready in MolDB for the GAFF2 benchmark.") from last_exc  # 关键步骤失败时立即报错，避免继续生成错误结果。


def _assign_merz_ion(ff: MERZ, smiles: str, *, label: str):  # 定义本例内部辅助函数，组织重复步骤。
    mol = ff.mol(smiles)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    mol = ff.ff_assign(mol)  # 分配力场参数并写入分子属性。
    if not mol:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"Cannot assign MERZ ion parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    print(f"[MERZ] assigned built-in ion parameters for {label}")  # 打印关键路径或状态，便于人工检查。
    return mol  # 返回该辅助函数的结果。


def _build_qm_ready_gaff_species(  # 定义本例内部辅助函数，组织重复步骤。
    ff: GAFF2 | GAFF2_mod,
    smiles: str,
    *,
    label: str,
    recipe: dict[str, str],
    resp_profile: str,
    work_root: Path,
    psi4_omp: int,
    mpi: int,
    omp: int,
    memory_mb: int,
    repo_db_dir: Path | None = None,  # 设置中间变量或可调参数，供后续工作流使用。
    cache_to_repo_db: bool = False,  # 设置中间变量或可调参数，供后续工作流使用。
):
    mol = utils.mol_from_smiles(smiles)  # 从 SMILES 直接构造 RDKit 分子。
    log_name = f"{label.lower()}_{recipe['family']}_{ff.name}"  # 设置中间变量或可调参数，供后续工作流使用。
    mol, _energy = qm.conformation_search(  # 执行构象搜索/几何优化，为 RESP 做准备。
        mol,
        ff=ff,  # 选择有机分子/聚合物/部分无机离子的力场对象。
        work_dir=work_root,  # 设置本例输出目录。
        log_name=log_name,  # 设置中间变量或可调参数，供后续工作流使用。
        psi4_omp=psi4_omp,  # 设置中间变量或可调参数，供后续工作流使用。
        mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
        memory=memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        opt_method=recipe["opt_method"],  # 设置中间变量或可调参数，供后续工作流使用。
        opt_basis=recipe["opt_basis"],  # 设置中间变量或可调参数，供后续工作流使用。
        opt_basis_gen=recipe["opt_basis_gen"],  # 设置中间变量或可调参数，供后续工作流使用。
    )
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        mol,
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        opt=False,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=work_root,  # 设置本例输出目录。
        log_name=log_name,  # 设置中间变量或可调参数，供后续工作流使用。
        omp=psi4_omp,  # 设置每个 rank 的 OpenMP 线程数。
        memory=memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        charge_method=recipe["charge_method"],  # 设置中间变量或可调参数，供后续工作流使用。
        charge_basis=recipe["charge_basis"],  # 设置中间变量或可调参数，供后续工作流使用。
        charge_basis_gen=recipe["charge_basis_gen"],  # 设置中间变量或可调参数，供后续工作流使用。
        resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    mol = ff.ff_assign(mol, charge=None, report=False)  # 分配力场参数并写入分子属性。
    if not mol:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"Cannot assign {ff.name} parameters for {label} after QM/RESP.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    if cache_to_repo_db and repo_db_dir is not None:  # 根据当前状态决定是否进入该分支。
        ff.store_to_db(  # 开始一个多行函数调用或配置块。
            mol,
            smiles_or_psmiles=smiles,  # 设置中间变量或可调参数，供后续工作流使用。
            name=label,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
            db_dir=repo_db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            basis_set=recipe["charge_basis"],  # 设置中间变量或可调参数，供后续工作流使用。
            method=recipe["charge_method"],  # 设置中间变量或可调参数，供后续工作流使用。
        )
        print(f"[MolDB] stored freshly computed {label} RESP entry into repo db: {repo_db_dir}")  # 打印关键路径或状态，便于人工检查。
    return mol  # 返回该辅助函数的结果。


def _point_charge_dipole_debye(mol) -> float | None:  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        conf = mol.GetConformer()  # 设置中间变量或可调参数，供后续工作流使用。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return None  # 返回该辅助函数的结果。
    dip = np.zeros(3, dtype=float)  # 设置中间变量或可调参数，供后续工作流使用。
    for atom in mol.GetAtoms():  # 遍历当前工作流中的一组对象或任务。
        q = float(atom.GetDoubleProp("AtomicCharge")) if atom.HasProp("AtomicCharge") else 0.0  # 设置中间变量或可调参数，供后续工作流使用。
        pos = conf.GetAtomPosition(atom.GetIdx())  # 设置中间变量或可调参数，供后续工作流使用。
        dip += q * np.asarray([float(pos.x), float(pos.y), float(pos.z)], dtype=float)  # 设置中间变量或可调参数，供后续工作流使用。
    # 1 e*Angstrom = 4.80320427 Debye
    return float(np.linalg.norm(dip) * 4.80320427)  # 返回该辅助函数的结果。


def _summarize_carbonate_charge_features(mol, *, label: str) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    carbonyl_oxygen_charges: list[float] = []  # 设置中间变量或可调参数，供后续工作流使用。
    carbonyl_carbon_charges: list[float] = []  # 设置中间变量或可调参数，供后续工作流使用。
    ether_oxygen_charges: list[float] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for atom in mol.GetAtoms():  # 遍历当前工作流中的一组对象或任务。
        sym = atom.GetSymbol()  # 设置中间变量或可调参数，供后续工作流使用。
        charge = float(atom.GetDoubleProp("AtomicCharge")) if atom.HasProp("AtomicCharge") else 0.0  # 指定电荷来源或电荷计算方式。
        if sym == "O":  # 根据当前状态决定是否进入该分支。
            is_carbonyl = False  # 设置中间变量或可调参数，供后续工作流使用。
            for bond in atom.GetBonds():  # 遍历当前工作流中的一组对象或任务。
                other = bond.GetOtherAtom(atom)  # 设置中间变量或可调参数，供后续工作流使用。
                if other.GetSymbol() == "C" and bond.GetBondTypeAsDouble() >= 1.5:  # 根据当前状态决定是否进入该分支。
                    is_carbonyl = True  # 设置中间变量或可调参数，供后续工作流使用。
                    break
            if is_carbonyl:  # 根据当前状态决定是否进入该分支。
                carbonyl_oxygen_charges.append(charge)
            else:  # 处理前面条件都不满足的情况。
                ether_oxygen_charges.append(charge)
        elif sym == "C":  # 继续判断另一个互斥分支。
            for bond in atom.GetBonds():  # 遍历当前工作流中的一组对象或任务。
                other = bond.GetOtherAtom(atom)  # 设置中间变量或可调参数，供后续工作流使用。
                if other.GetSymbol() == "O" and bond.GetBondTypeAsDouble() >= 1.5:  # 根据当前状态决定是否进入该分支。
                    carbonyl_carbon_charges.append(charge)
                    break
    net_q = 0.0  # 设置中间变量或可调参数，供后续工作流使用。
    for atom in mol.GetAtoms():  # 遍历当前工作流中的一组对象或任务。
        if atom.HasProp("AtomicCharge"):  # 根据当前状态决定是否进入该分支。
            net_q += float(atom.GetDoubleProp("AtomicCharge"))  # 设置中间变量或可调参数，供后续工作流使用。
    return {  # 返回该辅助函数的结果。
        "label": str(label),
        "net_charge_e": float(net_q),
        "carbonyl_oxygen_charge_e": float(np.mean(carbonyl_oxygen_charges)) if carbonyl_oxygen_charges else None,
        "carbonyl_carbon_charge_e": float(np.mean(carbonyl_carbon_charges)) if carbonyl_carbon_charges else None,
        "ether_oxygen_charge_mean_e": float(np.mean(ether_oxygen_charges)) if ether_oxygen_charges else None,
        "point_charge_dipole_debye": _point_charge_dipole_debye(mol),
    }


def _atom_ff_signature(mol) -> list[dict[str, Any]]:  # 定义本例内部辅助函数，组织重复步骤。
    rows: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for atom in mol.GetAtoms():  # 遍历当前工作流中的一组对象或任务。
        rows.append(  # 开始一个多行函数调用或配置块。
            {
                "idx": int(atom.GetIdx()),
                "symbol": str(atom.GetSymbol()),
                "ff_type": str(atom.GetProp("ff_type")) if atom.HasProp("ff_type") else "",
                "ff_sigma_nm": float(atom.GetDoubleProp("ff_sigma")) if atom.HasProp("ff_sigma") else None,
                "ff_epsilon_kj_mol": float(atom.GetDoubleProp("ff_epsilon")) if atom.HasProp("ff_epsilon") else None,
            }
        )
    return rows  # 返回该辅助函数的结果。


def _audit_gaff_variant_differences(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    recipe: dict[str, str],
    resp_profile: str,
    work_root: Path,
    repo_db_dir: Path,
    psi4_omp: int,
    mpi: int,
    omp: int,
    memory_mb: int,
) -> dict[str, Any]:
    audit: dict[str, Any] = {  # 设置中间变量或可调参数，供后续工作流使用。
        "recipe": recipe,
        "species": {},
        "notes": (
            "GAFF2_mod is audited only as a species-level reference. The full MD benchmark uses classic GAFF2."
        ),
    }
    classic_ff = GAFF2()  # 设置中间变量或可调参数，供后续工作流使用。
    mod_ff = GAFF2_mod()  # 设置中间变量或可调参数，供后续工作流使用。
    species_specs = (  # 设置中间变量或可调参数，供后续工作流使用。
        ("EC", EC_SMILES, "solvent"),
        ("EMC", EMC_SMILES, "solvent"),
        ("DEC", DEC_SMILES, "solvent"),
    )
    for label, smiles, role in species_specs:  # 遍历当前工作流中的一组对象或任务。
        classic_mol = _build_qm_ready_gaff_species(  # 设置中间变量或可调参数，供后续工作流使用。
            classic_ff,
            smiles,
            label=label,  # 给该选区一个可读标签，便于 manifest 检查。
            recipe=recipe,  # 设置中间变量或可调参数，供后续工作流使用。
            resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
            work_root=work_root,  # 设置中间变量或可调参数，供后续工作流使用。
            psi4_omp=psi4_omp,  # 设置中间变量或可调参数，供后续工作流使用。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            memory_mb=memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
            repo_db_dir=repo_db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
            cache_to_repo_db=False,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        mod_mol = utils.deepcopy_mol(classic_mol)  # 设置中间变量或可调参数，供后续工作流使用。
        mod_mol = mod_ff.ff_assign(mod_mol, charge=None, report=False)  # 分配力场参数并写入分子属性。
        if not mod_mol:  # 根据当前状态决定是否进入该分支。
            raise RuntimeError(f"Cannot assign GAFF2_mod parameters for {label} during audit.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
        classic_rows = _atom_ff_signature(classic_mol)  # 设置中间变量或可调参数，供后续工作流使用。
        mod_rows = _atom_ff_signature(mod_mol)  # 设置中间变量或可调参数，供后续工作流使用。
        diff = []  # 设置中间变量或可调参数，供后续工作流使用。
        for row_c, row_m in zip(classic_rows, mod_rows):  # 遍历当前工作流中的一组对象或任务。
            changed = {}  # 设置中间变量或可调参数，供后续工作流使用。
            for key in ("ff_type", "ff_sigma_nm", "ff_epsilon_kj_mol"):  # 遍历当前工作流中的一组对象或任务。
                if row_c.get(key) != row_m.get(key):  # 根据当前状态决定是否进入该分支。
                    changed[key] = {"classic": row_c.get(key), "mod": row_m.get(key)}  # 设置中间变量或可调参数，供后续工作流使用。
            if changed:  # 根据当前状态决定是否进入该分支。
                diff.append({"idx": row_c["idx"], "symbol": row_c["symbol"], "changes": changed})
        audit["species"][label] = {  # 设置中间变量或可调参数，供后续工作流使用。
            "role": role,
            "classic_summary": summarize_rdkit_species_forcefield(classic_mol, label=label, moltype_hint=label, charge_scale=1.0),
            "mod_summary": summarize_rdkit_species_forcefield(mod_mol, label=label, moltype_hint=label, charge_scale=1.0),
            "charge_features": _summarize_carbonate_charge_features(classic_mol, label=label),
            "atom_param_differences": diff,
        }
    pf6_classic = _load_ready_pf6(classic_ff, repo_db_dir=repo_db_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    pf6_mod = _load_ready_pf6(mod_ff, repo_db_dir=repo_db_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    diff_pf6 = []  # 设置中间变量或可调参数，供后续工作流使用。
    for row_c, row_m in zip(_atom_ff_signature(pf6_classic), _atom_ff_signature(pf6_mod)):  # 遍历当前工作流中的一组对象或任务。
        changed = {}  # 设置中间变量或可调参数，供后续工作流使用。
        for key in ("ff_type", "ff_sigma_nm", "ff_epsilon_kj_mol"):  # 遍历当前工作流中的一组对象或任务。
            if row_c.get(key) != row_m.get(key):  # 根据当前状态决定是否进入该分支。
                changed[key] = {"classic": row_c.get(key), "mod": row_m.get(key)}  # 设置中间变量或可调参数，供后续工作流使用。
        if changed:  # 根据当前状态决定是否进入该分支。
            diff_pf6.append({"idx": row_c["idx"], "symbol": row_c["symbol"], "changes": changed})
    audit["species"]["PF6"] = {  # 设置中间变量或可调参数，供后续工作流使用。
        "role": "anion",
        "classic_summary": summarize_rdkit_species_forcefield(pf6_classic, label="PF6", moltype_hint="PF6", charge_scale=1.0),
        "mod_summary": summarize_rdkit_species_forcefield(pf6_mod, label="PF6", moltype_hint="PF6", charge_scale=1.0),
        "atom_param_differences": diff_pf6,
    }
    return audit  # 返回该辅助函数的结果。


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
    starts: list[float] = []  # 设置中间变量或可调参数，供后续工作流使用。
    ends: list[float] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for label in labels:  # 遍历当前工作流中的一组对象或任务。
        metric = _extract_default_msd_metric_record(msd, label)  # 设置中间变量或可调参数，供后续工作流使用。
        start_raw = metric.get("trajectory_time_start_ps")  # 设置中间变量或可调参数，供后续工作流使用。
        end_raw = metric.get("trajectory_time_end_ps")  # 设置中间变量或可调参数，供后续工作流使用。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            if start_raw is not None and end_raw is not None:  # 根据当前状态决定是否进入该分支。
                start = float(start_raw)  # 设置中间变量或可调参数，供后续工作流使用。
                end = float(end_raw)  # 设置中间变量或可调参数，供后续工作流使用。
                if np.isfinite(start) and np.isfinite(end) and end > start:  # 根据当前状态决定是否进入该分支。
                    starts.append(start)
                    ends.append(end)
                    continue
        except Exception:  # 捕获异常并转成更清楚的示例错误信息。
            pass
        csv_path = metric.get("series_csv")  # 设置中间变量或可调参数，供后续工作流使用。
        if not csv_path:  # 根据当前状态决定是否进入该分支。
            continue
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            arr = np.genfromtxt(str(csv_path), delimiter=",", names=True)  # 设置中间变量或可调参数，供后续工作流使用。
            t = np.asarray(arr["time_ps"], dtype=float)  # 设置中间变量或可调参数，供后续工作流使用。
            t = t[np.isfinite(t)]  # 设置中间变量或可调参数，供后续工作流使用。
            if t.size >= 2 and float(t[-1]) > float(t[0]):  # 根据当前状态决定是否进入该分支。
                starts.append(float(t[0]))
                ends.append(float(t[-1]))
        except Exception:  # 捕获异常并转成更清楚的示例错误信息。
            continue
    if not starts or not ends:  # 根据当前状态决定是否进入该分支。
        return None, None  # 返回该辅助函数的结果。
    return float(min(starts)), float(max(ends))  # 返回该辅助函数的结果。


def _summarize_msd_block_diffusion(  # 定义本例内部辅助函数，组织重复步骤。
    blocks: list[dict[str, Any]],
    *,
    expected_order: tuple[str, ...] = ("EMC", "DEC", "EC"),  # 设置中间变量或可调参数，供后续工作流使用。
) -> dict[str, Any]:
    valid_blocks = [block for block in blocks if isinstance(block.get("diffusion_m2_s"), dict)]  # 设置中间变量或可调参数，供后续工作流使用。
    if not valid_blocks:  # 根据当前状态决定是否进入该分支。
        return {  # 返回该辅助函数的结果。
            "status": "skipped",
            "reason": "no_valid_block_diffusion",
            "blocks": blocks,
        }

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
        if not values:  # 根据当前状态决定是否进入该分支。
            continue
        arr = np.asarray(values, dtype=float)  # 设置中间变量或可调参数，供后续工作流使用。
        mean = float(np.mean(arr))  # 设置中间变量或可调参数，供后续工作流使用。
        std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
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
        ranking_interpretation = "Not enough solvent species are present to assess the expected order."  # 设置中间变量或可调参数，供后续工作流使用。
    elif all(frac >= 0.75 for frac in pairwise_fractions) and (match_fraction is None or match_fraction >= 0.75):  # 继续判断另一个互斥分支。
        ranking_confidence = "supports_expected"  # 设置中间变量或可调参数，供后续工作流使用。
        ranking_interpretation = "Most blocks support the expected solvent ordering."  # 设置中间变量或可调参数，供后续工作流使用。
    elif any(0.25 < frac < 0.75 for frac in pairwise_fractions):  # 继续判断另一个互斥分支。
        ranking_confidence = "ambiguous"  # 设置中间变量或可调参数，供后续工作流使用。
        ranking_interpretation = "At least one adjacent solvent pair changes order across blocks; extend sampling before interpreting that pair."  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        ranking_confidence = "contradicts_expected"  # 设置中间变量或可调参数，供后续工作流使用。
        ranking_interpretation = "Blockwise ordering consistently disagrees with at least one expected adjacent solvent pair."  # 设置中间变量或可调参数，供后续工作流使用。

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
        "ranking_interpretation": ranking_interpretation,
        "blocks": blocks,
        "notes": (
            "Each block recomputes species COM MSD from the trajectory slice using lag time. "
            "Use block spread/order fractions to judge whether a solvent ranking is stable."
        ),
    }


def _msd_block_diffusion_diagnostic(  # 定义本例内部辅助函数，组织重复步骤。
    analy: AnalyzeResult,
    *,
    full_msd: dict[str, Any],
    n_blocks: int,
    min_block_ps: float = 500.0,  # 设置中间变量或可调参数，供后续工作流使用。
    labels: tuple[str, ...] = ("EC", "EMC", "DEC"),  # 设置中间变量或可调参数，供后续工作流使用。
) -> dict[str, Any]:
    n_blocks = int(max(0, n_blocks))  # 设置中间变量或可调参数，供后续工作流使用。
    if n_blocks < 2:  # 根据当前状态决定是否进入该分支。
        return {"status": "skipped", "reason": "MSD_BLOCKS<2", "n_blocks_requested": n_blocks}  # 返回该辅助函数的结果。
    start_ps, end_ps = _default_msd_trajectory_bounds(full_msd, labels=labels)  # 设置中间变量或可调参数，供后续工作流使用。
    if start_ps is None or end_ps is None or end_ps <= start_ps:  # 根据当前状态决定是否进入该分支。
        return {"status": "skipped", "reason": "trajectory_time_bounds_unavailable", "n_blocks_requested": n_blocks}  # 返回该辅助函数的结果。
    duration_ps = float(end_ps - start_ps)  # 设置中间变量或可调参数，供后续工作流使用。
    max_blocks_by_duration = int(np.floor(duration_ps / max(float(min_block_ps), 1.0e-12)))  # 设置中间变量或可调参数，供后续工作流使用。
    block_count = min(n_blocks, max_blocks_by_duration)  # 设置中间变量或可调参数，供后续工作流使用。
    if block_count < 2:  # 根据当前状态决定是否进入该分支。
        return {  # 返回该辅助函数的结果。
            "status": "skipped",
            "reason": "trajectory_too_short_for_blocks",
            "n_blocks_requested": n_blocks,
            "duration_ps": duration_ps,
            "min_block_ps": float(min_block_ps),
        }

    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        topo = parse_system_top(Path(analy.top))  # 设置中间变量或可调参数，供后续工作流使用。
        system_dir = analy._system_dir()  # 设置中间变量或可调参数，供后续工作流使用。
        metric_catalog = build_msd_metric_catalog(topo, system_dir)  # 设置中间变量或可调参数，供后续工作流使用。
        xtc_path = analy._analysis_xtc_path()  # 设置中间变量或可调参数，供后续工作流使用。
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        return {"status": "skipped", "reason": f"setup_failed: {exc}", "n_blocks_requested": n_blocks}  # 返回该辅助函数的结果。

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
            moltype, entry = catalog_item  # 设置中间变量或可调参数，供后续工作流使用。
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


def _solvent_diffusion_diagnostic(  # 定义本例内部辅助函数，组织重复步骤。
    diffusion_m2_s: dict[str, float | None],
    *,
    expected_order: tuple[str, ...] = ("EMC", "DEC", "EC"),  # 设置中间变量或可调参数，供后续工作流使用。
) -> dict[str, Any]:
    solvents = ("EC", "EMC", "DEC")  # 设置中间变量或可调参数，供后续工作流使用。
    present: dict[str, float] = {}  # 设置中间变量或可调参数，供后续工作流使用。
    for label in solvents:  # 遍历当前工作流中的一组对象或任务。
        value = diffusion_m2_s.get(label)  # 设置中间变量或可调参数，供后续工作流使用。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            if value is not None:  # 根据当前状态决定是否进入该分支。
                present[label] = float(value)  # 设置中间变量或可调参数，供后续工作流使用。
        except Exception:  # 捕获异常并转成更清楚的示例错误信息。
            continue
    observed_order = [label for label, _value in sorted(present.items(), key=lambda item: item[1], reverse=True)]  # 设置中间变量或可调参数，供后续工作流使用。
    slowest = min(present.values()) if present else None  # 设置中间变量或可调参数，供后续工作流使用。
    relative_to_slowest = {  # 设置中间变量或可调参数，供后续工作流使用。
        label: (float(value) / float(slowest) if slowest and slowest > 0.0 else None)
        for label, value in present.items()  # 遍历当前工作流中的一组对象或任务。
    }
    expected_present = [label for label in expected_order if label in present]  # 设置中间变量或可调参数，供后续工作流使用。
    pairwise_expected = []  # 设置中间变量或可调参数，供后续工作流使用。
    for fast, slow in zip(expected_present, expected_present[1:]):  # 遍历当前工作流中的一组对象或任务。
        pairwise_expected.append(  # 开始一个多行函数调用或配置块。
            {
                "faster": fast,
                "slower": slow,
                "ok": bool(present.get(fast, float("-inf")) > present.get(slow, float("inf"))),
                "ratio": (
                    float(present[fast]) / float(present[slow])
                    if slow in present and present[slow] not in (0.0, None)  # 根据当前状态决定是否进入该分支。
                    else None
                ),
            }
        )
    expected_order_observed = [label for label in observed_order if label in expected_present]  # 设置中间变量或可调参数，供后续工作流使用。
    return {  # 返回该辅助函数的结果。
        "observed_order_fast_to_slow": observed_order,
        "expected_order_fast_to_slow": list(expected_order),
        "expected_order_for_present_species": expected_present,
        "matches_expected_for_present_species": bool(expected_order_observed == expected_present) if len(expected_present) >= 2 else None,
        "relative_to_slowest": relative_to_slowest,
        "pairwise_expected": pairwise_expected,
        "notes": "Diffusion ordering from short MD is noisy; use this diagnostic as a screen, not a final transport benchmark.",
    }


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
skip_completed_benchmark = _env_bool("SKIP_COMPLETED_BENCHMARK", True)  # 设置中间变量或可调参数，供后续工作流使用。

gaff_variant = _normalize_gaff_variant(os.environ.get("YADONPY_GAFF_VARIANT"))  # 设置中间变量或可调参数，供后续工作流使用。
charge_mode = _normalize_charge_mode(os.environ.get("YADONPY_GAFF_CHARGE_MODE"))  # 设置中间变量或可调参数，供后续工作流使用。
charge_recipe = _charge_recipe_from_family(os.environ.get("YADONPY_CHARGE_DFT_FAMILY"))  # 设置中间变量或可调参数，供后续工作流使用。
resp_profile = _normalize_resp_profile(os.environ.get("YADONPY_RESP_PROFILE"))  # 设置中间变量或可调参数，供后续工作流使用。
solvent_source = _normalize_solvent_source(os.environ.get("YADONPY_SOLVENT_SOURCE"))  # 设置中间变量或可调参数，供后续工作流使用。
db_priority_mode = _normalize_db_priority(os.environ.get("YADONPY_DB_PRIORITY"))  # 设置中间变量或可调参数，供后续工作流使用。
run_gaff_variant_audit = _env_bool("RUN_GAFF_VARIANT_AUDIT", solvent_source == "qm")
cache_qm_solvents_to_repo_db = _env_bool("CACHE_QM_SOLVENTS_TO_REPO_DB", False)  # 设置中间变量或可调参数，供后续工作流使用。

EC_SMILES = "O=C1OCCO1"  # 设置中间变量或可调参数，供后续工作流使用。
EMC_SMILES = "CCOC(=O)OC"  # 设置中间变量或可调参数，供后续工作流使用。
DEC_SMILES = "CCOC(=O)OCC"  # 设置中间变量或可调参数，供后续工作流使用。
LI_SMILES = "[Li+]"  # 设置中间变量或可调参数，供后续工作流使用。
PF6_SMILES = "F[P-](F)(F)(F)(F)F"  # 设置中间变量或可调参数，供后续工作流使用。

temp_k = _env_float("TEMP_K", 298.15)  # 设置中间变量或可调参数，供后续工作流使用。
press_bar = _env_float("PRESS_BAR", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
prod_ns = _env_float("PROD_NS", 5.0)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_final_ns = _env_float("EQ21_FINAL_NS", 0.8)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_pre_nvt_ps = _env_float("EQ21_PRE_NVT_PS", 10.0)  # 设置中间变量或可调参数，供后续工作流使用。
initial_density_g_cm3 = _env_float("INITIAL_DENSITY_G_CM3", 0.05)  # 设置中间变量或可调参数，供后续工作流使用。
max_additional_rounds = _env_int("MAX_ADDITIONAL_ROUNDS", 4)  # 设置中间变量或可调参数，供后续工作流使用。
equilibration_mode = _normalize_equilibration_mode(os.environ.get("EQUILIBRATION_MODE"))  # 设置中间变量或可调参数，供后续工作流使用。
prod_constraints = _normalize_constraints(os.environ.get("PROD_CONSTRAINTS"), default="h-bonds")  # 设置中间变量或可调参数，供后续工作流使用。
prod_dt_ps = _env_float("PROD_DT_PS", 0.002)  # 设置中间变量或可调参数，供后续工作流使用。
performance_profile = _env_text("PERFORMANCE_PROFILE", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
analysis_profile_requested = _env_text("ANALYSIS_PROFILE", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
trajectory_format_setting = _env_text("TRAJECTORY_FORMAT", os.environ.get("YADONPY_TRAJECTORY_FORMAT", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
traj_ps_setting = _env_text("TRAJ_PS", os.environ.get("YADONPY_PROD_TRAJ_PS", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
energy_ps_setting = _env_text("ENERGY_PS", os.environ.get("YADONPY_PROD_ENERGY_PS", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
log_ps_setting = _env_text("LOG_PS", os.environ.get("YADONPY_PROD_LOG_PS", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
trr_ps_setting = os.environ.get("TRR_PS")  # 设置中间变量或可调参数，供后续工作流使用。
velocity_ps_setting = os.environ.get("VELOCITY_PS")  # 设置中间变量或可调参数，供后续工作流使用。
max_trajectory_frames = _env_int("MAX_TRAJECTORY_FRAMES", 50000)  # 设置中间变量或可调参数，供后续工作流使用。
max_atom_frames = _env_float("MAX_ATOM_FRAMES", 5.0e9)  # 设置中间变量或可调参数，供后续工作流使用。
rdf_frame_stride_setting = _env_text("RDF_FRAME_STRIDE", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
rdf_bin_nm_setting = _env_text("RDF_BIN_NM", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
rdf_rmax_nm_setting = _env_text("RDF_RMAX_NM", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
msd_blocks = _env_int("MSD_BLOCKS", 4)  # 设置中间变量或可调参数，供后续工作流使用。
msd_block_min_ps = _env_float("MSD_BLOCK_MIN_PS", 500.0)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_recovery_constraints = _normalize_constraints(os.environ.get("LIQUID_RECOVERY_CONSTRAINTS"), default="none")  # 设置中间变量或可调参数，供后续工作流使用。
additional_round_ns = _env_float("ADDITIONAL_ROUND_NS", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
allow_unconverged_production = _env_bool("ALLOW_UNCONVERGED_PRODUCTION", False)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_hot_temp_k = _env_float("LIQUID_ANNEAL_HOT_TEMP_K", 600.0)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_hot_press_bar = _env_float("LIQUID_ANNEAL_HOT_PRESS_BAR", 1000.0)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_compact_press_bar = _env_float("LIQUID_ANNEAL_COMPACT_PRESS_BAR", max(liquid_hot_press_bar, 5000.0))  # 设置中间变量或可调参数，供后续工作流使用。
liquid_hot_nvt_ns = _env_float("LIQUID_ANNEAL_HOT_NVT_NS", 0.05)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_compact_npt_ns = _env_float("LIQUID_ANNEAL_COMPACT_NPT_NS", 0.15)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_hot_npt_ns = _env_float("LIQUID_ANNEAL_HOT_NPT_NS", 0.20)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_cooling_npt_ns = _env_float("LIQUID_ANNEAL_COOLING_NPT_NS", 0.10)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_recovery_hot_nvt_ns = _env_float("LIQUID_RECOVERY_HOT_NVT_NS", 0.03)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_recovery_compact_npt_ns = _env_float("LIQUID_RECOVERY_COMPACT_NPT_NS", 0.25)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_recovery_extend_max_rounds = _env_int("LIQUID_RECOVERY_EXTEND_MAX_ROUNDS", 4)  # 设置中间变量或可调参数，供后续工作流使用。
liquid_recovery_extend_ns = _env_float("LIQUID_RECOVERY_EXTEND_NS", 0.20)  # 设置中间变量或可调参数，供后续工作流使用。
polymer_recovery_warm_temp_k = _env_float("POLYMER_RECOVERY_WARM_TEMP_K", 0.0)  # 设置中间变量或可调参数，供后续工作流使用。
polymer_recovery_warm_nvt_ns = _env_float("POLYMER_RECOVERY_WARM_NVT_NS", 0.05)  # 设置中间变量或可调参数，供后续工作流使用。
polymer_recovery_compact_npt_ns = _env_float("POLYMER_RECOVERY_COMPACT_NPT_NS", 0.25)  # 设置中间变量或可调参数，供后续工作流使用。
polymer_recovery_extend_max_rounds = _env_int("POLYMER_RECOVERY_EXTEND_MAX_ROUNDS", 3)  # 设置中间变量或可调参数，供后续工作流使用。
polymer_recovery_extend_ns = _env_float("POLYMER_RECOVERY_EXTEND_NS", 0.20)  # 设置中间变量或可调参数，供后续工作流使用。
polymer_chain_warm_temp_k = _env_float("POLYMER_CHAIN_WARM_TEMP_K", 0.0)  # 设置中间变量或可调参数，供后续工作流使用。
polymer_chain_warm_nvt_ns = _env_float("POLYMER_CHAIN_WARM_NVT_NS", 0.10)  # 设置中间变量或可调参数，供后续工作流使用。

mpi = _env_int("MPI", 1)  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = _env_int("OMP", 16)  # 设置每个 rank 的 OpenMP 线程数。
gpu = _env_int("GPU", 1)  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = _env_int("GPU_ID", 0)  # 选择 GPU 设备编号，多卡节点可修改。

count_ec = _env_int("COUNT_EC", 120)  # 设置中间变量或可调参数，供后续工作流使用。
count_emc = _env_int("COUNT_EMC", 120)  # 设置中间变量或可调参数，供后续工作流使用。
count_dec = _env_int("COUNT_DEC", 120)  # 设置中间变量或可调参数，供后续工作流使用。
salt_pairs = _env_int("SALT_PAIRS", 45)  # 设置盐离子对数；阳离子和阴离子应同步增减。

li_charge_scale = _env_float("LI_CHARGE_SCALE", 0.8)  # 设置中间变量或可调参数，供后续工作流使用。
pf6_charge_scale = _env_float("PF6_CHARGE_SCALE", 0.8)  # 设置中间变量或可调参数，供后续工作流使用。

psi4_omp = _env_int("PSI4_OMP", 64)  # 设置中间变量或可调参数，供后续工作流使用。
memory_mb = _env_int("MEM_MB", 20000)  # 设置中间变量或可调参数，供后续工作流使用。

work_dir_name = _env_text(  # 设置中间变量或可调参数，供后续工作流使用。
    "WORK_DIR_NAME",
    f"benchmark_carbonate_lipf6_gaff2_{gaff_variant}_{charge_recipe['family']}_work",
)
work_root = Path(_env_text("WORK_DIR", str(BASE_DIR / work_dir_name))).resolve()  # 设置中间变量或可调参数，供后续工作流使用。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    work_root = workdir(work_root, restart=restart_status)  # 创建或复用本例工作目录。
    completed_summary = work_root / "06_analysis" / "benchmark_summary.json"  # 设置中间变量或可调参数，供后续工作流使用。
    if restart_status and skip_completed_benchmark and completed_summary.exists() and not any((species_only, build_only, export_only)):  # 根据当前状态决定是否进入该分支。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            completed_payload = json.loads(completed_summary.read_text(encoding="utf-8"))  # 设置中间变量或可调参数，供后续工作流使用。
        except Exception:  # 捕获异常并转成更清楚的示例错误信息。
            completed_payload = {}  # 设置中间变量或可调参数，供后续工作流使用。
        completed_status = str(completed_payload.get("status") or "completed")  # 设置中间变量或可调参数，供后续工作流使用。
        completed_diffusion = completed_payload.get("diffusion_m2_s")  # 设置中间变量或可调参数，供后续工作流使用。
        if completed_status != "failed_equilibration_density_gate" and isinstance(completed_diffusion, dict):  # 根据当前状态决定是否进入该分支。
            print(f"[SKIP] Existing completed benchmark_summary.json found at {completed_summary}")  # 打印关键路径或状态，便于人工检查。
            print(json.dumps(completed_diffusion, indent=2, ensure_ascii=False))  # 打印关键路径或状态，便于人工检查。
            raise SystemExit(0)  # 关键步骤失败时立即报错，避免继续生成错误结果。

    build_dir = work_root.child("00_build_cell")  # 设置中间变量或可调参数，供后续工作流使用。
    ff = _build_ff_variant(gaff_variant)  # 选择有机分子/聚合物/部分无机离子的力场对象。
    ion_ff = MERZ()  # 选择单原子离子参数来源。
    resolved_db_priority = db_priority_mode  # 设置中间变量或可调参数，供后续工作流使用。
    if resolved_db_priority == "auto":  # 根据当前状态决定是否进入该分支。
        resolved_db_priority = "repo_first" if solvent_source == "moldb" and resp_profile == "adaptive" else "default_first"

    if solvent_source == "moldb":  # 根据当前状态决定是否进入该分支。
        ec = _load_ready_gaff_species(  # 设置中间变量或可调参数，供后续工作流使用。
            ff,
            EC_SMILES,
            label="EC",  # 给该选区一个可读标签，便于 manifest 检查。
            repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
            charge_mode=charge_mode,  # 设置中间变量或可调参数，供后续工作流使用。
            db_priority=resolved_db_priority,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        emc = _load_ready_gaff_species(  # 设置中间变量或可调参数，供后续工作流使用。
            ff,
            EMC_SMILES,
            label="EMC",  # 给该选区一个可读标签，便于 manifest 检查。
            repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
            charge_mode=charge_mode,  # 设置中间变量或可调参数，供后续工作流使用。
            db_priority=resolved_db_priority,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        dec = _load_ready_gaff_species(  # 设置中间变量或可调参数，供后续工作流使用。
            ff,
            DEC_SMILES,
            label="DEC",  # 给该选区一个可读标签，便于 manifest 检查。
            repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
            charge_mode=charge_mode,  # 设置中间变量或可调参数，供后续工作流使用。
            db_priority=resolved_db_priority,  # 设置中间变量或可调参数，供后续工作流使用。
        )
    else:  # 处理前面条件都不满足的情况。
        ec = _build_qm_ready_gaff_species(  # 设置中间变量或可调参数，供后续工作流使用。
            ff,
            EC_SMILES,
            label="EC",  # 给该选区一个可读标签，便于 manifest 检查。
            recipe=charge_recipe,  # 设置中间变量或可调参数，供后续工作流使用。
            resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
            work_root=work_root,  # 设置中间变量或可调参数，供后续工作流使用。
            psi4_omp=psi4_omp,  # 设置中间变量或可调参数，供后续工作流使用。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            memory_mb=memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
            repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
            cache_to_repo_db=cache_qm_solvents_to_repo_db,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        emc = _build_qm_ready_gaff_species(  # 设置中间变量或可调参数，供后续工作流使用。
            ff,
            EMC_SMILES,
            label="EMC",  # 给该选区一个可读标签，便于 manifest 检查。
            recipe=charge_recipe,  # 设置中间变量或可调参数，供后续工作流使用。
            resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
            work_root=work_root,  # 设置中间变量或可调参数，供后续工作流使用。
            psi4_omp=psi4_omp,  # 设置中间变量或可调参数，供后续工作流使用。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            memory_mb=memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
            repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
            cache_to_repo_db=cache_qm_solvents_to_repo_db,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        dec = _build_qm_ready_gaff_species(  # 设置中间变量或可调参数，供后续工作流使用。
            ff,
            DEC_SMILES,
            label="DEC",  # 给该选区一个可读标签，便于 manifest 检查。
            recipe=charge_recipe,  # 设置中间变量或可调参数，供后续工作流使用。
            resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
            work_root=work_root,  # 设置中间变量或可调参数，供后续工作流使用。
            psi4_omp=psi4_omp,  # 设置中间变量或可调参数，供后续工作流使用。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            memory_mb=memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
            repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
            cache_to_repo_db=cache_qm_solvents_to_repo_db,  # 设置中间变量或可调参数，供后续工作流使用。
        )
    li = _assign_merz_ion(ion_ff, LI_SMILES, label="Li")  # 设置中间变量或可调参数，供后续工作流使用。
    pf6 = _load_ready_pf6(ff, repo_db_dir=REPO_DB_DIR, db_priority=resolved_db_priority)  # 设置中间变量或可调参数，供后续工作流使用。

    if solvent_source == "moldb":  # 根据当前状态决定是否进入该分支。
        solvent_charge_method = f"{charge_mode.upper()}[MolDB-ready]"  # 设置中间变量或可调参数，供后续工作流使用。
        solvent_db_flags = {"prefer_db": True, "require_db": True, "require_ready": True}  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        solvent_charge_method = f"RESP[{charge_recipe['label']}]"  # 设置中间变量或可调参数，供后续工作流使用。
        solvent_db_flags = {"prefer_db": False, "require_db": False, "require_ready": False}  # 设置中间变量或可调参数，供后续工作流使用。
    _stamp_charge_route(ec, charge_method=solvent_charge_method, **solvent_db_flags)  # 设置中间变量或可调参数，供后续工作流使用。
    _stamp_charge_route(emc, charge_method=solvent_charge_method, **solvent_db_flags)  # 设置中间变量或可调参数，供后续工作流使用。
    _stamp_charge_route(dec, charge_method=solvent_charge_method, **solvent_db_flags)  # 设置中间变量或可调参数，供后续工作流使用。
    _stamp_charge_route(li, charge_method="MERZ", prefer_db=False, require_db=False, require_ready=False)  # 设置中间变量或可调参数，供后续工作流使用。
    _stamp_charge_route(pf6, charge_method="RESP", prefer_db=True, require_db=True, require_ready=True)  # 设置中间变量或可调参数，供后续工作流使用。

    if run_gaff_variant_audit:  # 根据当前状态决定是否进入该分支。
        gaff_variant_audit = _audit_gaff_variant_differences(  # 设置中间变量或可调参数，供后续工作流使用。
            recipe=charge_recipe,  # 设置中间变量或可调参数，供后续工作流使用。
            resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
            work_root=work_root,  # 设置中间变量或可调参数，供后续工作流使用。
            repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
            psi4_omp=psi4_omp,  # 设置中间变量或可调参数，供后续工作流使用。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            memory_mb=memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        )
    else:  # 处理前面条件都不满足的情况。
        gaff_variant_audit = {  # 设置中间变量或可调参数，供后续工作流使用。
            "recipe": charge_recipe,
            "species": {},
            "skipped": True,
            "notes": "GAFF2 classic-vs-mod audit skipped for this run configuration.",
        }
    analysis_dir = work_root / "06_analysis"  # 设置中间变量或可调参数，供后续工作流使用。
    analysis_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    _dump_json(analysis_dir / "gaff_variant_audit.json", gaff_variant_audit)

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
            "benchmark_name": "carbonate_lipf6_gaff2",
            "ff_variant": gaff_variant,
            "charge_mode": charge_mode,
            "resp_profile": resp_profile,
            "solvent_source": solvent_source,
            "db_priority": resolved_db_priority,
            "cache_qm_solvents_to_repo_db": cache_qm_solvents_to_repo_db,
            "run_gaff_variant_audit": run_gaff_variant_audit,
            "qm_charge_recipe": charge_recipe,
            "resolved_qm_recipes": solvent_routes,
            "solvent_charge_method": solvent_charge_method,
            "pf6_charge_method": "RESP",
            "li_charge_method": "MERZ",
            "species": ["EC", "EMC", "DEC", "Li", "PF6"],
            "counts": {"EC": count_ec, "EMC": count_emc, "DEC": count_dec, "Li": salt_pairs, "PF6": salt_pairs},
            "charge_scale": {"EC": 1.0, "EMC": 1.0, "DEC": 1.0, "Li": li_charge_scale, "PF6": pf6_charge_scale},
            "eq21_final_ns": eq21_final_ns,
            "eq21_pre_nvt_ps": eq21_pre_nvt_ps,
            "prod_ns": prod_ns,
            "equilibration_mode_requested": equilibration_mode,
            "max_additional_rounds": max_additional_rounds,
            "additional_round_ns": additional_round_ns,
            "allow_unconverged_production": allow_unconverged_production,
            "prod_constraints": prod_constraints,
            "prod_dt_ps": prod_dt_ps,
            "msd_blocks": msd_blocks,
            "msd_block_min_ps": msd_block_min_ps,
            "liquid_anneal": {
                "hot_temp_K": liquid_hot_temp_k,
                "hot_press_bar": liquid_hot_press_bar,
                "compact_press_bar": liquid_compact_press_bar,
                "hot_nvt_ns": liquid_hot_nvt_ns,
                "compact_npt_ns": liquid_compact_npt_ns,
                "hot_npt_ns": liquid_hot_npt_ns,
                "cooling_npt_ns": liquid_cooling_npt_ns,
                "density_recovery": {
                    "hot_nvt_ns": liquid_recovery_hot_nvt_ns,
                    "compact_npt_ns": liquid_recovery_compact_npt_ns,
                    "extend_max_rounds": liquid_recovery_extend_max_rounds,
                    "extend_ns": liquid_recovery_extend_ns,
                    "constraints": liquid_recovery_constraints,
                    "notes": "No-polymer failed density gates use high-pressure recovery rounds before the normal four-round stop condition is evaluated.",
                },
            },
            "polymer_adaptive_relaxation": {
                "density_recovery": {
                    "warm_temp_K": None if polymer_recovery_warm_temp_k <= 0.0 else polymer_recovery_warm_temp_k,
                    "warm_nvt_ns": polymer_recovery_warm_nvt_ns,
                    "compact_npt_ns": polymer_recovery_compact_npt_ns,
                    "extend_max_rounds": polymer_recovery_extend_max_rounds,
                    "extend_ns": polymer_recovery_extend_ns,
                    "pressure_ladder_bar": [500.0, 1000.0, 2000.0, 5000.0],
                    "constraints": "none",
                },
                "chain_relaxation": {
                    "warm_temp_K": None if polymer_chain_warm_temp_k <= 0.0 else polymer_chain_warm_temp_k,
                    "warm_nvt_ns": polymer_chain_warm_nvt_ns,
                    "constraints": "none",
                },
            },
            "expected_diffusion_trend": "EMC > DEC > EC (literature-guided target for mixed linear/cyclic carbonate electrolyte)",
            "neutral_charge_issues": neutral_charge_issues,
            "literature_alignment_notes": {
                "coordination_target": "EC should remain at least as strong a Li carbonyl ligand as EMC; DEC should not remain systematically stronger than EC.",
                "transport_target": "EMC should remain more transport-favorable than DEC in mixed carbonate electrolyte benchmark.",
            },
        },
        "species_pre_export": species_rows,
        "species_charge_sanity": {
            "EC": _summarize_carbonate_charge_features(ec, label="EC"),
            "EMC": _summarize_carbonate_charge_features(emc, label="EMC"),
            "DEC": _summarize_carbonate_charge_features(dec, label="DEC"),
        },
        "species_equivalence_spread": equivalence_spread,
    }
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
    estimated_atoms = int(sum(int(count) * int(mol.GetNumAtoms()) for mol, count in zip(cell_mols, counts)))  # 设置中间变量或可调参数，供后续工作流使用。
    io_policy = resolve_io_analysis_policy(  # 设置中间变量或可调参数，供后续工作流使用。
        prod_ns=float(prod_ns),  # 设置中间变量或可调参数，供后续工作流使用。
        atom_count=estimated_atoms,  # 设置中间变量或可调参数，供后续工作流使用。
        performance_profile=performance_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=analysis_profile_requested,  # 选择后处理预设；interface_fast 面向 slab/interface。
        trajectory_format=trajectory_format_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        traj_ps=traj_ps_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        energy_ps=energy_ps_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        log_ps=log_ps_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        trr_ps=trr_ps_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        velocity_ps=velocity_ps_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        rdf_frame_stride=rdf_frame_stride_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        rdf_rmax_nm=rdf_rmax_nm_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        rdf_bin_nm=rdf_bin_nm_setting,  # 设置中间变量或可调参数，供后续工作流使用。
        msd_selected_species=["EC", "EMC", "DEC", "Li", "PF6"],  # 设置中间变量或可调参数，供后续工作流使用。
        max_trajectory_frames=max_trajectory_frames,  # 设置中间变量或可调参数，供后续工作流使用。
        max_atom_frames=max_atom_frames,  # 设置中间变量或可调参数，供后续工作流使用。
    )

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

    has_polymer = eq.cell_meta_contains_polymer(ac)  # 设置中间变量或可调参数，供后续工作流使用。
    selected_equilibration_mode = equilibration_mode  # 设置中间变量或可调参数，供后续工作流使用。
    if selected_equilibration_mode == "auto":  # 根据当前状态决定是否进入该分支。
        selected_equilibration_mode = "eq21" if has_polymer else "liquid_anneal"  # 设置中间变量或可调参数，供后续工作流使用。
    species_summary["metadata"]["has_polymer"] = bool(has_polymer)  # 设置中间变量或可调参数，供后续工作流使用。
    species_summary["metadata"]["equilibration_mode"] = selected_equilibration_mode  # 设置中间变量或可调参数，供后续工作流使用。
    species_summary["metadata"]["estimated_total_atoms"] = int(estimated_atoms)  # 设置中间变量或可调参数，供后续工作流使用。
    species_summary["metadata"]["performance_policy"] = io_policy.to_dict()  # 设置中间变量或可调参数，供后续工作流使用。
    _dump_json(analysis_dir / "species_forcefield_summary.json", species_summary)

    eqmd = eq.EQ21step(ac, work_dir=work_root) if selected_equilibration_mode == "eq21" else eq.LiquidAnneal(ac, work_dir=work_root)
    if export_only:  # 根据当前状态决定是否进入该分支。
        exported = eqmd.ensure_system_exported()  # 设置中间变量或可调参数，供后续工作流使用。
        print(f"[EXPORT-ONLY] Exported 02_system at {exported.system_top.parent}")  # 打印关键路径或状态，便于人工检查。
        raise SystemExit(0)  # 关键步骤失败时立即报错，避免继续生成错误结果。

    if selected_equilibration_mode == "eq21":  # 根据当前状态决定是否进入该分支。
        ac = eqmd.exec(  # 设置中间变量或可调参数，供后续工作流使用。
            temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
            press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
            gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
            time=eq21_final_ns,  # 设置中间变量或可调参数，供后续工作流使用。
            eq21_pre_nvt_ps=eq21_pre_nvt_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        )
    else:  # 处理前面条件都不满足的情况。
        ac = eqmd.exec(  # 设置中间变量或可调参数，供后续工作流使用。
            temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
            press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
            gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
            time=eq21_final_ns,  # 设置中间变量或可调参数，供后续工作流使用。
            hot_temp=liquid_hot_temp_k,  # 设置中间变量或可调参数，供后续工作流使用。
            hot_pressure_bar=liquid_hot_press_bar,  # 设置中间变量或可调参数，供后续工作流使用。
            compact_pressure_bar=liquid_compact_press_bar,  # 设置中间变量或可调参数，供后续工作流使用。
            hot_nvt_ns=liquid_hot_nvt_ns,  # 设置每轮高温 NVT 时长。
            compact_npt_ns=liquid_compact_npt_ns,  # 设置中间变量或可调参数，供后续工作流使用。
            hot_npt_ns=liquid_hot_npt_ns,  # 设置中间变量或可调参数，供后续工作流使用。
            cooling_npt_ns=liquid_cooling_npt_ns,  # 设置中间变量或可调参数，供后续工作流使用。
            constraints=prod_constraints,  # 设置约束策略。
            dt_ps=prod_dt_ps,  # 设置 MD 时间步长，单位 ps。
        )
    analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    analy.get_all_prop(temp=temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。
    latest_equilibrated_gro = eqmd.final_gro()  # 设置中间变量或可调参数，供后续工作流使用。
    if restart_status and not result:  # 根据当前状态决定是否进入该分支。
        restart_gro = _restart_latest_equilibrated_gro(work_root, latest_equilibrated_gro)  # 设置中间变量或可调参数，供后续工作流使用。
        if restart_gro != latest_equilibrated_gro:  # 根据当前状态决定是否进入该分支。
            restart_analy = _analyze_restart_stage(work_root, restart_gro)  # 设置中间变量或可调参数，供后续工作流使用。
            if restart_analy is not None:  # 根据当前状态决定是否进入该分支。
                restart_analy.get_all_prop(temp=temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
                result = restart_analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。
        latest_equilibrated_gro = restart_gro  # 设置中间变量或可调参数，供后续工作流使用。

    additional_rounds_run = 0  # 设置中间变量或可调参数，供后续工作流使用。
    additional_round_strategies: list[str] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for _ in range(max_additional_rounds):  # 遍历当前工作流中的一组对象或任务。
        if result:  # 根据当前状态决定是否进入该分支。
            break
        equilibrium_payload = _load_equilibrium_payload(analysis_dir)  # 设置中间变量或可调参数，供后续工作流使用。
        strategy = eq.select_relaxation_strategy(equilibrium_payload, has_polymer=has_polymer)  # 设置中间变量或可调参数，供后续工作流使用。
        if strategy == "production":  # 根据当前状态决定是否进入该分支。
            break
        additional_round_strategies.append(strategy)
        if strategy == "liquid_density_recovery":  # 根据当前状态决定是否进入该分支。
            eq_more = eq.LiquidDensityRecovery(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
            ac = eq_more.exec(  # 设置中间变量或可调参数，供后续工作流使用。
                temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
                press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
                mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
                omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
                gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
                gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
                time=additional_round_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                hot_temp=liquid_hot_temp_k,  # 设置中间变量或可调参数，供后续工作流使用。
                hot_pressure_bar=liquid_hot_press_bar,  # 设置中间变量或可调参数，供后续工作流使用。
                compact_pressure_bar=liquid_compact_press_bar,  # 设置中间变量或可调参数，供后续工作流使用。
                hot_nvt_ns=liquid_recovery_hot_nvt_ns,  # 设置每轮高温 NVT 时长。
                compact_npt_ns=liquid_recovery_compact_npt_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                cooling_npt_ns=liquid_cooling_npt_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                compact_extend=True,  # 设置中间变量或可调参数，供后续工作流使用。
                compact_extend_max_rounds=liquid_recovery_extend_max_rounds,  # 设置中间变量或可调参数，供后续工作流使用。
                compact_extend_ns=liquid_recovery_extend_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                dt_ps=prod_dt_ps,  # 设置 MD 时间步长，单位 ps。
                hot_dt_ps=min(float(prod_dt_ps), 0.001),  # 设置中间变量或可调参数，供后续工作流使用。
                constraints=liquid_recovery_constraints,  # 设置约束策略。
                start_gro=latest_equilibrated_gro,  # 设置中间变量或可调参数，供后续工作流使用。
            )
        elif strategy == "polymer_density_recovery":  # 继续判断另一个互斥分支。
            eq_more = eq.PolymerDensityRecovery(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
            ac = eq_more.exec(  # 设置中间变量或可调参数，供后续工作流使用。
                temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
                press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
                mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
                omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
                gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
                gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
                time=additional_round_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                warm_temp=(None if polymer_recovery_warm_temp_k <= 0.0 else polymer_recovery_warm_temp_k),
                warm_nvt_ns=polymer_recovery_warm_nvt_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                compact_npt_ns=polymer_recovery_compact_npt_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                compact_extend=True,  # 设置中间变量或可调参数，供后续工作流使用。
                compact_extend_max_rounds=polymer_recovery_extend_max_rounds,  # 设置中间变量或可调参数，供后续工作流使用。
                compact_extend_ns=polymer_recovery_extend_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                dt_ps=min(float(prod_dt_ps), 0.001),  # 设置 MD 时间步长，单位 ps。
                start_gro=latest_equilibrated_gro,  # 设置中间变量或可调参数，供后续工作流使用。
            )
        elif strategy == "polymer_chain_relaxation":  # 继续判断另一个互斥分支。
            eq_more = eq.PolymerChainRelaxation(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
            ac = eq_more.exec(  # 设置中间变量或可调参数，供后续工作流使用。
                temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
                press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
                mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
                omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
                gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
                gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
                time=additional_round_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                warm_temp=(None if polymer_chain_warm_temp_k <= 0.0 else polymer_chain_warm_temp_k),
                warm_nvt_ns=polymer_chain_warm_nvt_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                dt_ps=min(float(prod_dt_ps), 0.001),  # 设置 MD 时间步长，单位 ps。
                start_gro=latest_equilibrated_gro,  # 设置中间变量或可调参数，供后续工作流使用。
            )
        else:  # 处理前面条件都不满足的情况。
            additional_constraints = "none" if has_polymer else prod_constraints  # 设置中间变量或可调参数，供后续工作流使用。
            additional_dt_ps = min(float(prod_dt_ps), 0.001) if _normalize_constraints(additional_constraints, default="none") == "none" else float(prod_dt_ps)
            additional_mdp_overrides = None  # 设置中间变量或可调参数，供后续工作流使用。
            additional_gpu_offload_mode = "auto"  # 设置中间变量或可调参数，供后续工作流使用。
            additional_gpu = gpu  # 设置中间变量或可调参数，供后续工作流使用。
            additional_gpu_id = gpu_id  # 设置中间变量或可调参数，供后续工作流使用。
            if not has_polymer:  # 根据当前状态决定是否进入该分支。
                additional_gpu_offload_mode = "auto"  # 设置中间变量或可调参数，供后续工作流使用。
                additional_mdp_overrides = None  # 设置中间变量或可调参数，供后续工作流使用。
            eq_more = eq.Additional(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
            ac = eq_more.exec(  # 设置中间变量或可调参数，供后续工作流使用。
                temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
                press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
                mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
                omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
                gpu=additional_gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
                gpu_id=additional_gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
                time=additional_round_ns,  # 设置中间变量或可调参数，供后续工作流使用。
                dt_ps=additional_dt_ps,  # 设置 MD 时间步长，单位 ps。
                constraints=additional_constraints,  # 设置约束策略。
                gpu_offload_mode=additional_gpu_offload_mode,  # 设置中间变量或可调参数，供后续工作流使用。
                mdp_overrides=additional_mdp_overrides,  # 设置中间变量或可调参数，供后续工作流使用。
                start_gro=latest_equilibrated_gro,  # 设置中间变量或可调参数，供后续工作流使用。
                skip_rebuild=True,  # 设置中间变量或可调参数，供后续工作流使用。
                micro_relax=False,  # 设置中间变量或可调参数，供后续工作流使用。
            )
        latest_equilibrated_gro = eq_more.final_gro()  # 设置中间变量或可调参数，供后续工作流使用。
        additional_rounds_run += 1  # 设置中间变量或可调参数，供后续工作流使用。
        analy = eq_more.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        analy.get_all_prop(temp=temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    equilibrium_payload = _load_equilibrium_payload(analysis_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    equilibrium_status = _transport_confidence_from_equilibrium(equilibrium_payload, result)  # 设置中间变量或可调参数，供后续工作流使用。
    if not result:  # 根据当前状态决定是否进入该分支。
        fail_summary = {  # 设置中间变量或可调参数，供后续工作流使用。
            "metadata": species_summary["metadata"],
            "analysis": {
                "analysis_profile": io_policy.analysis_profile,
                "performance_policy": io_policy.to_dict(),
            },
            "equilibration_ok": False,
            "density_warning_severity": equilibrium_status["density_warning_severity"],
            "transport_confidence": equilibrium_status["transport_confidence"],
            "equilibration": {
                "mode": selected_equilibration_mode,
                "has_polymer": bool(has_polymer),
                "additional_rounds_run": int(additional_rounds_run),
                "max_additional_rounds": int(max_additional_rounds),
                "additional_round_strategies": additional_round_strategies,
                "density_gate": equilibrium_status["density_gate"],
                "rg_gate": equilibrium_payload.get("rg_gate") if isinstance(equilibrium_payload, dict) else None,
                "relaxation_state": equilibrium_payload.get("relaxation_state") if isinstance(equilibrium_payload, dict) else None,
            },
            "status": "failed_equilibration_density_gate",
            "message": "Equilibration did not converge after the configured additional rounds; production/MSD was not run.",
        }
        _dump_json(analysis_dir / "benchmark_summary.json", fail_summary)
        utils.radon_print(  # 打印关键路径或状态，便于人工检查。
            "[ERROR] Equilibration density/Rg gate is still failing after additional rounds; "
            "production/MSD will not run. Set ALLOW_UNCONVERGED_PRODUCTION=1 only for diagnostics.",
            level=2,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        if not allow_unconverged_production:  # 根据当前状态决定是否进入该分支。
            raise SystemExit(2)  # 关键步骤失败时立即报错，避免继续生成错误结果。
        utils.radon_print(  # 打印关键路径或状态，便于人工检查。
            "[WARN] ALLOW_UNCONVERGED_PRODUCTION=1: continuing for diagnostics only; "
            "diffusion/transport values may be severely overestimated.",
            level=2,  # 设置中间变量或可调参数，供后续工作流使用。
        )

    npt = eq.NPT(ac, work_dir=work_root)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = npt.exec(  # 设置中间变量或可调参数，供后续工作流使用。
        temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
        press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
        mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
        gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
        gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
        time=prod_ns,  # 设置中间变量或可调参数，供后续工作流使用。
        dt_ps=prod_dt_ps,  # 设置 MD 时间步长，单位 ps。
        constraints=prod_constraints,  # 设置约束策略。
        traj_ps=io_policy.traj_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        energy_ps=io_policy.energy_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        log_ps=io_policy.log_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        trr_ps=io_policy.trr_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        velocity_ps=io_policy.velocity_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        trajectory_format=io_policy.trajectory_format,  # 设置中间变量或可调参数，供后续工作流使用。
        performance_profile=io_policy.performance_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=io_policy.analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
        max_trajectory_frames=io_policy.max_trajectory_frames,  # 设置中间变量或可调参数，供后续工作流使用。
        max_atom_frames=io_policy.max_atom_frames,  # 设置中间变量或可调参数，供后续工作流使用。
        start_gro=latest_equilibrated_gro,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    analy = npt.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    basic = analy.get_all_prop(  # 设置中间变量或可调参数，供后续工作流使用。
        temp=temp_k,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
        press=press_bar,  # 设置压力 bar；用于 NPT/EQ 阶段。
        save=True,  # 设置中间变量或可调参数，供后续工作流使用。
        include_polymer_metrics=bool(io_policy.include_polymer_metrics),  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=io_policy.analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
    )
    msd_species = list(cell_mols)  # 设置中间变量或可调参数，供后续工作流使用。
    msd = analy.msd(mols=msd_species, analysis_profile=io_policy.analysis_profile)  # 设置中间变量或可调参数，供后续工作流使用。
    msd_block_diagnostic = _msd_block_diffusion_diagnostic(  # 设置中间变量或可调参数，供后续工作流使用。
        analy,
        full_msd=msd,  # 设置中间变量或可调参数，供后续工作流使用。
        n_blocks=msd_blocks,  # 设置中间变量或可调参数，供后续工作流使用。
        min_block_ps=msd_block_min_ps,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    _dump_json(analysis_dir / "msd_block_diffusion.json", msd_block_diagnostic)
    if salt_pairs > 0:  # 根据当前状态决定是否进入该分支。
        rdf = analy.rdf(  # 设置中间变量或可调参数，供后续工作流使用。
            center_mol=li,  # 设置中间变量或可调参数，供后续工作流使用。
            analysis_profile=io_policy.analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
            bin_nm=float(io_policy.rdf_bin_nm),  # 指定 z-profile bin 宽。
            r_max_nm=io_policy.rdf_rmax_nm,  # 设置中间变量或可调参数，供后续工作流使用。
            frame_stride=int(io_policy.rdf_frame_stride),  # 设置中间变量或可调参数，供后续工作流使用。
        )
        sigma = analy.sigma(msd=msd, temp_k=temp_k, eh_mode="gmx_current_only")  # 设置中间变量或可调参数，供后续工作流使用。
        coordination = {  # 设置中间变量或可调参数，供后续工作流使用。
            "EC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "ec"),
            "EMC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "emc"),
            "DEC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "dec"),
            "PF6_coordination_fluorine": _extract_rdf_site(rdf, "pf6:coordination_fluorine"),
            "PF6_fluorine_site": _extract_rdf_site(rdf, "pf6:fluorine_site"),
        }
        coordination_preference = _coordination_preference_summary(  # 设置中间变量或可调参数，供后续工作流使用。
            coordination,
            {"EC": count_ec, "EMC": count_emc, "DEC": count_dec},
        )
    else:  # 处理前面条件都不满足的情况。
        sigma = {  # 设置中间变量或可调参数，供后续工作流使用。
            "sigma_ne_upper_bound_S_m": None,
            "sigma_eh_total_S_m": None,
            "haven_ratio": None,
            "eh": {"confidence": "skipped", "quality_note": "salt-free solvent mixture"},
        }
        coordination = {}  # 设置中间变量或可调参数，供后续工作流使用。
        coordination_preference = {  # 设置中间变量或可调参数，供后续工作流使用。
            "total_cn_shell": None,
            "notes": "Skipped for salt-free solvent mixture because no Li center group is present.",
        }

    diffusion_m2_s = {  # 设置中间变量或可调参数，供后续工作流使用。
        "EC": _extract_default_diffusivity(msd, "EC"),
        "EMC": _extract_default_diffusivity(msd, "EMC"),
        "DEC": _extract_default_diffusivity(msd, "DEC"),
        "Li": _extract_default_diffusivity(msd, "Li"),
        "PF6": _extract_default_diffusivity(msd, "PF6"),
    }
    summary = {  # 设置中间变量或可调参数，供后续工作流使用。
        "metadata": species_summary["metadata"],
        "equilibration_ok": equilibrium_status["equilibration_ok"],
        "density_warning_severity": equilibrium_status["density_warning_severity"],
        "transport_confidence": equilibrium_status["transport_confidence"],
        "equilibration": {
            "mode": selected_equilibration_mode,
            "has_polymer": bool(has_polymer),
            "additional_rounds_run": int(additional_rounds_run),
            "max_additional_rounds": int(max_additional_rounds),
            "additional_round_strategies": additional_round_strategies,
            "density_gate": equilibrium_status["density_gate"],
            "rg_gate": equilibrium_payload.get("rg_gate") if isinstance(equilibrium_payload, dict) else None,
            "relaxation_state": equilibrium_payload.get("relaxation_state") if isinstance(equilibrium_payload, dict) else None,
        },
        "basic_properties": basic.get("basic_properties", {}),
        "analysis": {
            "analysis_profile": io_policy.analysis_profile,
            "performance_policy": io_policy.to_dict(),
            "include_polymer_metrics": bool(io_policy.include_polymer_metrics),
            "rdf": {
                "bin_nm": float(io_policy.rdf_bin_nm),
                "r_max_nm": io_policy.rdf_rmax_nm,
                "frame_stride": int(io_policy.rdf_frame_stride),
            },
            "msd": {
                "selected_species": io_policy.msd_selected_species,
                "default_metric_only": bool(io_policy.msd_default_metric_only),
            },
        },
        "diffusion_m2_s": diffusion_m2_s,
        "solvent_diffusion_diagnostic": _solvent_diffusion_diagnostic(diffusion_m2_s),
        "msd_block_diffusion_diagnostic": msd_block_diagnostic,
        "conductivity": {
            "sigma_ne_upper_bound_S_m": sigma.get("sigma_ne_upper_bound_S_m"),
            "sigma_eh_total_S_m": sigma.get("sigma_eh_total_S_m"),
            "haven_ratio": sigma.get("haven_ratio"),
            "eh_confidence": ((sigma.get("eh") or {}) if isinstance(sigma.get("eh"), dict) else {}).get("confidence"),
            "eh_quality_note": ((sigma.get("eh") or {}) if isinstance(sigma.get("eh"), dict) else {}).get("quality_note"),
        },
        "coordination": coordination,
    }
    summary["coordination_preference"] = coordination_preference  # 设置中间变量或可调参数，供后续工作流使用。
    _dump_json(analysis_dir / "benchmark_summary.json", summary)
    print("[BENCHMARK] carbonate_lipf6_gaff2 completed")  # 打印关键路径或状态，便于人工检查。
    print(json.dumps(summary["diffusion_m2_s"], indent=2, ensure_ascii=False))  # 打印关键路径或状态，便于人工检查。
