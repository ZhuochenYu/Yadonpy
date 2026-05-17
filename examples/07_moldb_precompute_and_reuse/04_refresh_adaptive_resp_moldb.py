from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Refresh existing repo MolDB entries with fully reoptimized adaptive RESP.

This script is intentionally more conservative than the lightweight Example 07
builder:

* it only targets species that already exist in the repository MolDB;
* workers write candidate molecules into a temporary candidate MolDB first;
* the repo MolDB is hard-replaced only after every target succeeds;
* old/new charge differences are written before the final replacement.

The default route performs a fresh DFT geometry optimization plus adaptive RESP
for every selected species. Use ``--reuse-geometry`` only for debugging.
"""

import argparse  # 导入本例需要的库或 yadonpy 接口。
import csv  # 导入本例需要的库或 yadonpy 接口。
import json  # 导入本例需要的库或 yadonpy 接口。
import math  # 导入本例需要的库或 yadonpy 接口。
import multiprocessing as mp  # 导入本例需要的库或 yadonpy 接口。
import os  # 导入本例需要的库或 yadonpy 接口。
import queue  # 导入本例需要的库或 yadonpy 接口。
import shutil  # 导入本例需要的库或 yadonpy 接口。
import subprocess  # 导入本例需要的库或 yadonpy 接口。
import time  # 导入本例需要的库或 yadonpy 接口。
from dataclasses import asdict, dataclass  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。
from typing import Any  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import assign_charges, mol_from_smiles  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import chem_utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.polyelectrolyte import detect_charged_groups  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.moldb import MolDB  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.moldb.store import canonical_key  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。


HERE = Path(__file__).resolve().parent  # 定位当前脚本所在目录。
REPO_ROOT = HERE.parents[1]  # 定位仓库根目录。
CATALOG_CSV = HERE / "electrolyte_species.csv"  # 指定示例使用的物种目录表。

DEFAULT_TARGETS = (  # 设置中间变量或可调参数，供后续工作流使用。
    "PAA",
    "glucose_2",
    "glucose_3",
    "glucose_6",
    "glucose_23",
    "glucose_26",
    "glucose_36",
    "glucose_236",
    "EC",
    "EMC",
    "DEC",
)

@dataclass(frozen=True)  # 声明轻量数据类，用于保存配置或任务信息。
class SpeciesSpec:  # 定义本例内部数据结构或配置对象。
    name: str
    smiles: str
    kind: str
    charge: str
    bonded: str | None
    polyelectrolyte_mode: bool


@dataclass(frozen=True)  # 声明轻量数据类，用于保存配置或任务信息。
class RefreshTask:  # 定义本例内部数据结构或配置对象。
    name: str
    smiles: str
    kind: str
    charge: str
    bonded: str | None
    polyelectrolyte_mode: bool
    profile: str
    priority: int
    heavy_atoms: int
    formal_charge: int
    required_cores: int
    psi4_omp: int
    memory_mb: int


def _csv_bool(value: object, *, default: bool = False) -> bool:  # 定义本例内部辅助函数，组织重复步骤。
    token = str(value or "").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if not token:  # 根据当前状态决定是否进入该分支。
        return bool(default)  # 返回该辅助函数的结果。
    return token in {"1", "true", "t", "yes", "y", "on"}  # 返回该辅助函数的结果。


def _read_catalog(path: Path) -> dict[str, SpeciesSpec]:  # 定义本例内部辅助函数，组织重复步骤。
    out: dict[str, SpeciesSpec] = {}  # 设置中间变量或可调参数，供后续工作流使用。
    with path.open("r", encoding="utf-8", newline="") as fh:  # 用上下文管理器安全打开文件或管理资源。
        reader = csv.DictReader(fh)  # 设置中间变量或可调参数，供后续工作流使用。
        for raw in reader:  # 遍历当前工作流中的一组对象或任务。
            name = str(raw.get("name") or "").strip()  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
            smiles = str(raw.get("smiles") or "").strip()  # 设置中间变量或可调参数，供后续工作流使用。
            if not name or not smiles:  # 根据当前状态决定是否进入该分支。
                continue
            out[name] = SpeciesSpec(  # 设置中间变量或可调参数，供后续工作流使用。
                name=name,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                smiles=smiles,  # 设置中间变量或可调参数，供后续工作流使用。
                kind=str(raw.get("kind") or ("psmiles" if "*" in smiles else "smiles")).strip(),  # 设置中间变量或可调参数，供后续工作流使用。
                charge=str(raw.get("charge") or "RESP").strip().upper(),  # 指定电荷来源或电荷计算方式。
                bonded=(str(raw.get("bonded") or "").strip() or None),  # 指定特殊 bonded 参数方案，例如 PF6 的 DRIH。
                polyelectrolyte_mode=_csv_bool(raw.get("polyelectrolyte_mode"), default=False),  # 启用聚电解质处理逻辑。
            )
    return out  # 返回该辅助函数的结果。


def _env_default(name: str, default: int) -> int:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    return int(raw) if raw else int(default)  # 返回该辅助函数的结果。


def _parse_only(tokens: list[str] | None) -> list[str]:  # 定义本例内部辅助函数，组织重复步骤。
    if not tokens:  # 根据当前状态决定是否进入该分支。
        env = str(os.environ.get("YADONPY_ONLY", "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
        tokens = [env] if env else []  # 设置中间变量或可调参数，供后续工作流使用。
    seen: list[str] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for token in tokens:  # 遍历当前工作流中的一组对象或任务。
        for part in str(token).split(","):  # 遍历当前工作流中的一组对象或任务。
            name = part.strip()  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
            if name and name not in seen:  # 根据当前状态决定是否进入该分支。
                seen.append(name)
    return seen  # 返回该辅助函数的结果。


def _formal_charge(mol) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))  # 返回该辅助函数的结果。


def _heavy_atom_count(mol) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))  # 返回该辅助函数的结果。


def _profile_variant_ready(db: MolDB, spec: SpeciesSpec, *, resp_profile: str) -> bool:  # 定义本例内部辅助函数，组织重复步骤。
    profile = str(resp_profile or "adaptive").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        _kind, _canonical, key = canonical_key(spec.smiles)  # 设置中间变量或可调参数，供后续工作流使用。
        rec = db.load_record(key)  # 设置中间变量或可调参数，供后续工作流使用。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return False  # 返回该辅助函数的结果。
    if rec is None:  # 根据当前状态决定是否进入该分支。
        return False  # 返回该辅助函数的结果。
    for meta in (rec.variants or {}).values():  # 遍历当前工作流中的一组对象或任务。
        if not isinstance(meta, dict):  # 根据当前状态决定是否进入该分支。
            continue
        if str(meta.get("charge") or "RESP").upper() != "RESP":  # 根据当前状态决定是否进入该分支。
            continue
        if str(meta.get("resp_profile") or "").strip().lower() != profile:  # 根据当前状态决定是否进入该分支。
            continue
        if bool(meta.get("polyelectrolyte_mode", False)) != bool(spec.polyelectrolyte_mode):  # 根据当前状态决定是否进入该分支。
            continue
        if bool(meta.get("ready", False)):  # 根据当前状态决定是否进入该分支。
            return True  # 返回该辅助函数的结果。
    return False  # 返回该辅助函数的结果。


def _record_exists(db: MolDB, spec: SpeciesSpec) -> bool:  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        _kind, _canonical, key = canonical_key(spec.smiles)  # 设置中间变量或可调参数，供后续工作流使用。
        return db.load_record(key) is not None  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return False  # 返回该辅助函数的结果。


def _selected_specs(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    catalog: dict[str, SpeciesSpec],
    repo_db: MolDB,
    only: list[str],
    target_mode: str,
) -> list[SpeciesSpec]:
    if only:  # 根据当前状态决定是否进入该分支。
        missing = [name for name in only if name not in catalog]  # 设置中间变量或可调参数，供后续工作流使用。
        if missing:  # 根据当前状态决定是否进入该分支。
            raise SystemExit(f"Unknown species in electrolyte_species.csv: {missing}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
        return [catalog[name] for name in only if _record_exists(repo_db, catalog[name])]  # 返回该辅助函数的结果。
    mode = str(target_mode or "existing-repo").strip().lower()  # 设置该配置块使用的计算模式。
    if mode == "default-targets":  # 根据当前状态决定是否进入该分支。
        return [catalog[name] for name in DEFAULT_TARGETS if name in catalog and _record_exists(repo_db, catalog[name])]  # 返回该辅助函数的结果。
    if mode != "existing-repo":  # 根据当前状态决定是否进入该分支。
        raise SystemExit(f"Unsupported target mode: {target_mode!r}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return [spec for spec in catalog.values() if _record_exists(repo_db, spec)]  # 返回该辅助函数的结果。


def _load_geometry(spec: SpeciesSpec, *, dbs: list[tuple[str, MolDB]], resp_profile: str) -> tuple[Any, str, bool]:  # 定义本例内部辅助函数，组织重复步骤。
    for label, db in dbs:  # 遍历当前工作流中的一组对象或任务。
        for profile in (resp_profile, "legacy", None):  # 遍历当前工作流中的一组对象或任务。
            kwargs: dict[str, Any] = {  # 设置中间变量或可调参数，供后续工作流使用。
                "require_ready": False,
                "charge": "RESP",
                "polyelectrolyte_mode": spec.polyelectrolyte_mode,
                "polyelectrolyte_detection": "auto",
            }
            if profile is not None:  # 根据当前状态决定是否进入该分支。
                kwargs["resp_profile"] = profile  # 设置中间变量或可调参数，供后续工作流使用。
            try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
                mol, _rec = db.load_mol(spec.smiles, **kwargs)  # 设置中间变量或可调参数，供后续工作流使用。
                return mol, f"{label}-db-{profile or 'any'}", False  # 返回该辅助函数的结果。
            except Exception:  # 捕获异常并转成更清楚的示例错误信息。
                continue
    return mol_from_smiles(spec.smiles, name=spec.name), "fresh-smiles", True  # 返回该辅助函数的结果。


def _charge_values(mol, prop: str = "AtomicCharge") -> list[float | None]:  # 定义本例内部辅助函数，组织重复步骤。
    values: list[float | None] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for atom in mol.GetAtoms():  # 遍历当前工作流中的一组对象或任务。
        if atom.HasProp(prop):  # 根据当前状态决定是否进入该分支。
            values.append(float(atom.GetDoubleProp(prop)))
        else:  # 处理前面条件都不满足的情况。
            values.append(None)
    return values  # 返回该辅助函数的结果。


def _net_charge(charges: list[float | None]) -> float | None:  # 定义本例内部辅助函数，组织重复步骤。
    if any(value is None for value in charges):  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    return float(sum(float(value) for value in charges if value is not None))  # 返回该辅助函数的结果。


def _group_spreads(mol, groups: list[list[int]], prop: str = "AtomicCharge") -> list[dict[str, Any]]:  # 定义本例内部辅助函数，组织重复步骤。
    values = _charge_values(mol, prop=prop)  # 设置中间变量或可调参数，供后续工作流使用。
    spreads: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for group in groups:  # 遍历当前工作流中的一组对象或任务。
        idxs = sorted({int(i) for i in group})  # 设置中间变量或可调参数，供后续工作流使用。
        qs = [values[i] for i in idxs if 0 <= i < len(values) and values[i] is not None]
        if len(qs) != len(idxs) or len(qs) <= 1:  # 根据当前状态决定是否进入该分支。
            continue
        spreads.append(  # 开始一个多行函数调用或配置块。
            {
                "indices": idxs,
                "symbols": [mol.GetAtomWithIdx(i).GetSymbol() for i in idxs],
                "charges": [float(q) for q in qs],
                "spread": float(max(qs) - min(qs)),
            }
        )
    return spreads  # 返回该辅助函数的结果。


def _carboxylate_spreads(mol, prop: str = "AtomicCharge") -> list[dict[str, Any]]:  # 定义本例内部辅助函数，组织重复步骤。
    values = _charge_values(mol, prop=prop)  # 设置中间变量或可调参数，供后续工作流使用。
    out: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for atom in mol.GetAtoms():  # 遍历当前工作流中的一组对象或任务。
        if atom.GetSymbol() != "C":  # 根据当前状态决定是否进入该分支。
            continue
        oxygen_atoms = [bond.GetOtherAtom(atom) for bond in atom.GetBonds() if bond.GetOtherAtom(atom).GetSymbol() == "O"]
        if len(oxygen_atoms) != 2:  # 根据当前状态决定是否进入该分支。
            continue
        oxy_idxs = sorted(int(oxygen.GetIdx()) for oxygen in oxygen_atoms)  # 设置中间变量或可调参数，供后续工作流使用。
        if not any(mol.GetAtomWithIdx(i).GetFormalCharge() < 0 for i in oxy_idxs):  # 根据当前状态决定是否进入该分支。
            continue
        qs = [values[i] for i in oxy_idxs if values[i] is not None]  # 设置中间变量或可调参数，供后续工作流使用。
        if len(qs) != 2:  # 根据当前状态决定是否进入该分支。
            continue
        out.append(  # 开始一个多行函数调用或配置块。
            {
                "carbon": int(atom.GetIdx()),
                "oxygen_indices": oxy_idxs,
                "charges": [float(q) for q in qs],
                "spread": float(max(qs) - min(qs)),
            }
        )
    return out  # 返回该辅助函数的结果。


def _validate_equivalence(mol, *, tolerance: float) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    groups = chem_utils.resp_equivalence_groups_from_mol(mol)  # 设置中间变量或可调参数，供后续工作流使用。
    spreads = _group_spreads(mol, groups)  # 设置中间变量或可调参数，供后续工作流使用。
    carboxylates = _carboxylate_spreads(mol)  # 设置中间变量或可调参数，供后续工作流使用。
    max_group_spread = max([item["spread"] for item in spreads] or [0.0])  # 设置中间变量或可调参数，供后续工作流使用。
    max_carboxylate_spread = max([item["spread"] for item in carboxylates] or [0.0])  # 设置中间变量或可调参数，供后续工作流使用。
    ok = max(max_group_spread, max_carboxylate_spread) <= float(tolerance)
    return {  # 返回该辅助函数的结果。
        "ok": bool(ok),
        "tolerance": float(tolerance),
        "equivalence_group_count": len(groups),
        "max_equivalence_spread": float(max_group_spread),
        "max_carboxylate_oxygen_spread": float(max_carboxylate_spread),
        "equivalence_spreads": spreads,
        "carboxylate_spreads": carboxylates,
    }


def _old_resp_snapshot(db: MolDB, spec: SpeciesSpec) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    _kind, _canonical, key = canonical_key(spec.smiles)  # 设置中间变量或可调参数，供后续工作流使用。
    rec = db.load_record(key)  # 设置中间变量或可调参数，供后续工作流使用。
    variants = dict((rec.variants or {}) if rec is not None else {})  # 设置中间变量或可调参数，供后续工作流使用。
    mol, _rec = db.load_mol(  # 设置中间变量或可调参数，供后续工作流使用。
        spec.smiles,
        require_ready=True,  # 要求 MolDB 物种必须已准备好。
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        polyelectrolyte_mode=spec.polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
        polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
    )
    charges = _charge_values(mol)  # 设置中间变量或可调参数，供后续工作流使用。
    resp_variant_ids = [  # 设置中间变量或可调参数，供后续工作流使用。
        vid
        for vid, meta in sorted(variants.items())  # 遍历当前工作流中的一组对象或任务。
        if isinstance(meta, dict) and str(meta.get("charge") or "RESP").strip().upper() == "RESP"  # 根据当前状态决定是否进入该分支。
    ]
    adaptive_ids = [  # 设置中间变量或可调参数，供后续工作流使用。
        vid
        for vid, meta in sorted(variants.items())  # 遍历当前工作流中的一组对象或任务。
        if isinstance(meta, dict)  # 根据当前状态决定是否进入该分支。
        and str(meta.get("charge") or "RESP").strip().upper() == "RESP"
        and str(meta.get("resp_profile") or "").strip().lower() == "adaptive"
    ]
    return {  # 返回该辅助函数的结果。
        "key": key,
        "name": rec.name if rec is not None else spec.name,
        "variant_ids": list(variants.keys()),
        "resp_variant_ids": resp_variant_ids,
        "adaptive_resp_variant_ids": adaptive_ids,
        "selected_resp_profile": "adaptive" if adaptive_ids else "legacy",
        "charges": charges,
        "net_charge": _net_charge(charges),
        "manifest": rec.to_dict() if rec is not None else {},
    }


def _charge_diff(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    spec: SpeciesSpec,
    old_snapshot: dict[str, Any],
    new_mol,
    validation: dict[str, Any],
    new_variant_id: str | None,
) -> dict[str, Any]:
    old = list(old_snapshot.get("charges") or [])  # 设置中间变量或可调参数，供后续工作流使用。
    new = _charge_values(new_mol)  # 设置中间变量或可调参数，供后续工作流使用。
    if len(old) != len(new):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"{spec.name} atom-count mismatch in charge diff: old={len(old)} new={len(new)}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    deltas: list[float] = []  # 设置中间变量或可调参数，供后续工作流使用。
    per_atom: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for idx, (old_q, new_q) in enumerate(zip(old, new)):  # 遍历当前工作流中的一组对象或任务。
        if old_q is None or new_q is None:  # 根据当前状态决定是否进入该分支。
            delta = None  # 设置中间变量或可调参数，供后续工作流使用。
            abs_delta = None  # 设置中间变量或可调参数，供后续工作流使用。
        else:  # 处理前面条件都不满足的情况。
            delta = float(new_q) - float(old_q)  # 设置中间变量或可调参数，供后续工作流使用。
            abs_delta = abs(delta)  # 设置中间变量或可调参数，供后续工作流使用。
            deltas.append(float(delta))
        per_atom.append(  # 开始一个多行函数调用或配置块。
            {
                "species": spec.name,
                "atom_index": int(idx),
                "symbol": new_mol.GetAtomWithIdx(idx).GetSymbol(),
                "old_charge": old_q,
                "new_charge": new_q,
                "delta": delta,
                "abs_delta": abs_delta,
            }
        )
    abs_deltas = [abs(x) for x in deltas]  # 设置中间变量或可调参数，供后续工作流使用。
    rms_delta = math.sqrt(sum(x * x for x in deltas) / len(deltas)) if deltas else None  # 设置中间变量或可调参数，供后续工作流使用。
    summary = {  # 设置中间变量或可调参数，供后续工作流使用。
        "species": spec.name,
        "smiles": spec.smiles,
        "atom_count": len(new),
        "formal_charge": _formal_charge(new_mol),
        "old_net_charge": old_snapshot.get("net_charge"),
        "new_net_charge": _net_charge(new),
        "old_resp_profile": old_snapshot.get("selected_resp_profile"),
        "new_resp_profile": "adaptive",
        "old_resp_variant_ids": old_snapshot.get("resp_variant_ids") or [],
        "new_resp_variant_id": new_variant_id,
        "max_abs_delta": max(abs_deltas) if abs_deltas else None,
        "mean_abs_delta": (sum(abs_deltas) / len(abs_deltas)) if abs_deltas else None,
        "rms_delta": rms_delta,
        "equivalence_ok": bool(validation.get("ok", False)),
        "max_equivalence_spread": validation.get("max_equivalence_spread"),
        "max_carboxylate_oxygen_spread": validation.get("max_carboxylate_oxygen_spread"),
    }
    return {"summary": summary, "per_atom": per_atom}  # 返回该辅助函数的结果。


def _available_cpu_total() -> int:  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        return max(1, len(os.sched_getaffinity(0)))  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return max(1, int(os.cpu_count() or 1))  # 返回该辅助函数的结果。


def _available_memory_mb() -> int:  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():  # 遍历当前工作流中的一组对象或任务。
            if line.startswith("MemTotal:"):  # 根据当前状态决定是否进入该分支。
                return max(1024, int(line.split()[1]) // 1024)  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        pass
    return 96000  # 返回该辅助函数的结果。


def _task_profile(spec: SpeciesSpec) -> tuple[str, int]:  # 定义本例内部辅助函数，组织重复步骤。
    if str(spec.bonded or "").strip().upper() == "DRIH":  # 根据当前状态决定是否进入该分支。
        return "drih", 0  # 返回该辅助函数的结果。
    if bool(spec.polyelectrolyte_mode):  # 根据当前状态决定是否进入该分支。
        return "polyelectrolyte", 1  # 返回该辅助函数的结果。
    if "*" in str(spec.smiles):  # 根据当前状态决定是否进入该分支。
        return "polymer", 2  # 返回该辅助函数的结果。
    return "standard", 3  # 返回该辅助函数的结果。


def _task_omp(*, profile: str, heavy_atoms: int, cpu_budget: int, max_omp: int) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    max_allowed = max(1, min(int(cpu_budget), int(max_omp)))  # 设置中间变量或可调参数，供后续工作流使用。
    if profile in {"drih", "polyelectrolyte"}:  # 根据当前状态决定是否进入该分支。
        requested = 8 if heavy_atoms < 35 else 12  # 设置中间变量或可调参数，供后续工作流使用。
    elif profile == "polymer":  # 继续判断另一个互斥分支。
        requested = 6 if heavy_atoms < 35 else 8  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        requested = 4 if heavy_atoms < 25 else 6  # 设置中间变量或可调参数，供后续工作流使用。
    return max(1, min(max_allowed, requested))  # 返回该辅助函数的结果。


def _task_memory_mb(*, profile: str, heavy_atoms: int, max_memory_mb: int) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    base = 8000 + 220 * int(heavy_atoms)  # 设置中间变量或可调参数，供后续工作流使用。
    if profile in {"drih", "polyelectrolyte"}:  # 根据当前状态决定是否进入该分支。
        base += 4000  # 设置中间变量或可调参数，供后续工作流使用。
    if heavy_atoms >= 45:  # 根据当前状态决定是否进入该分支。
        base += 4000  # 设置中间变量或可调参数，供后续工作流使用。
    return max(6000, min(int(max_memory_mb), int(base)))  # 返回该辅助函数的结果。


def _build_refresh_tasks(  # 定义本例内部辅助函数，组织重复步骤。
    specs: list[SpeciesSpec],
    *,
    cpu_budget: int,
    max_omp: int,
    max_memory_mb: int,
) -> list[RefreshTask]:
    tasks: list[RefreshTask] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for spec in specs:  # 遍历当前工作流中的一组对象或任务。
        mol = mol_from_smiles(spec.smiles, coord=False, name=spec.name)  # 设置中间变量或可调参数，供后续工作流使用。
        heavy_atoms = _heavy_atom_count(mol)  # 设置中间变量或可调参数，供后续工作流使用。
        formal_charge = _formal_charge(mol)  # 设置中间变量或可调参数，供后续工作流使用。
        profile, priority = _task_profile(spec)  # 设置中间变量或可调参数，供后续工作流使用。
        omp = _task_omp(profile=profile, heavy_atoms=heavy_atoms, cpu_budget=cpu_budget, max_omp=max_omp)  # 设置每个 rank 的 OpenMP 线程数。
        memory_mb = _task_memory_mb(profile=profile, heavy_atoms=heavy_atoms, max_memory_mb=max_memory_mb)  # 设置中间变量或可调参数，供后续工作流使用。
        tasks.append(  # 开始一个多行函数调用或配置块。
            RefreshTask(  # 开始一个多行函数调用或配置块。
                name=spec.name,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                smiles=spec.smiles,  # 设置中间变量或可调参数，供后续工作流使用。
                kind=spec.kind,  # 设置中间变量或可调参数，供后续工作流使用。
                charge=spec.charge,  # 指定电荷来源或电荷计算方式。
                bonded=spec.bonded,  # 指定特殊 bonded 参数方案，例如 PF6 的 DRIH。
                polyelectrolyte_mode=bool(spec.polyelectrolyte_mode),  # 启用聚电解质处理逻辑。
                profile=profile,  # 设置中间变量或可调参数，供后续工作流使用。
                priority=priority,  # 设置中间变量或可调参数，供后续工作流使用。
                heavy_atoms=heavy_atoms,  # 设置中间变量或可调参数，供后续工作流使用。
                formal_charge=formal_charge,  # 设置中间变量或可调参数，供后续工作流使用。
                required_cores=omp,  # 设置中间变量或可调参数，供后续工作流使用。
                psi4_omp=omp,  # 设置中间变量或可调参数，供后续工作流使用。
                memory_mb=memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
            )
        )
    tasks.sort(key=lambda task: (task.priority, -task.heavy_atoms, task.name.lower()))  # 设置中间变量或可调参数，供后续工作流使用。
    return tasks  # 返回该辅助函数的结果。


def _pending_payloads(tasks: list[RefreshTask]) -> list[dict[str, Any]]:  # 定义本例内部辅助函数，组织重复步骤。
    pending = [dict(asdict(task), attempt=1, max_attempts=2) for task in tasks]  # 设置中间变量或可调参数，供后续工作流使用。
    _sort_pending_in_place(pending)
    return pending  # 返回该辅助函数的结果。


def _sort_pending_in_place(pending: list[dict[str, Any]]) -> None:  # 定义本例内部辅助函数，组织重复步骤。
    pending.sort(  # 开始一个多行函数调用或配置块。
        key=lambda item: (  # 设置中间变量或可调参数，供后续工作流使用。
            int(item["priority"]),
            -int(item["required_cores"]),
            -int(item.get("memory_mb", 0)),
            str(item["name"]).lower(),
        )
    )


def _eligible_pending_for_launch(pending: list[dict[str, Any]], available_cores: int, available_memory_mb: int) -> list[dict[str, Any]]:  # 定义本例内部辅助函数，组织重复步骤。
    if not pending:  # 根据当前状态决定是否进入该分支。
        return []  # 返回该辅助函数的结果。
    priorities = sorted({int(item["priority"]) for item in pending})  # 设置中间变量或可调参数，供后续工作流使用。
    for priority in priorities:  # 遍历当前工作流中的一组对象或任务。
        same_priority = [item for item in pending if int(item["priority"]) == priority]
        fitting = [  # 设置中间变量或可调参数，供后续工作流使用。
            item
            for item in same_priority  # 遍历当前工作流中的一组对象或任务。
            if int(item["required_cores"]) <= int(available_cores)  # 根据当前状态决定是否进入该分支。
            and int(item.get("memory_mb", 0)) <= int(available_memory_mb)
        ]
        if fitting:  # 根据当前状态决定是否进入该分支。
            return fitting  # 返回该辅助函数的结果。
    return [  # 返回该辅助函数的结果。
        item
        for item in pending  # 遍历当前工作流中的一组对象或任务。
        if int(item["required_cores"]) <= int(available_cores)  # 根据当前状态决定是否进入该分支。
        and int(item.get("memory_mb", 0)) <= int(available_memory_mb)
    ]


def _maybe_schedule_retry(task: dict[str, Any], *, error: str) -> dict[str, Any] | None:  # 定义本例内部辅助函数，组织重复步骤。
    attempt = int(task.get("attempt", 1))  # 设置中间变量或可调参数，供后续工作流使用。
    max_attempts = int(task.get("max_attempts", 2))  # 设置中间变量或可调参数，供后续工作流使用。
    current_cores = max(1, int(task["required_cores"]))  # 设置中间变量或可调参数，供后续工作流使用。
    if attempt >= max_attempts or current_cores <= 1:  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    retried = dict(task)  # 设置中间变量或可调参数，供后续工作流使用。
    retried["attempt"] = attempt + 1  # 设置中间变量或可调参数，供后续工作流使用。
    retried["required_cores"] = max(1, current_cores // 2)  # 设置中间变量或可调参数，供后续工作流使用。
    retried["psi4_omp"] = max(1, current_cores // 2)  # 设置中间变量或可调参数，供后续工作流使用。
    retried["retry_of_cores"] = current_cores  # 设置中间变量或可调参数，供后续工作流使用。
    retried["retry_reason"] = error  # 设置中间变量或可调参数，供后续工作流使用。
    retry_geom_iter = int(task.get("retry_geom_iter", 0) or 0)  # 设置中间变量或可调参数，供后续工作流使用。
    if retry_geom_iter > int(task.get("geom_iter", 50) or 50):  # 根据当前状态决定是否进入该分支。
        retried["geom_iter"] = retry_geom_iter  # 设置中间变量或可调参数，供后续工作流使用。
    return retried  # 返回该辅助函数的结果。


def _worker_refresh_candidate(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    task_payload: dict[str, Any],
    repo_db_dir: str,
    candidate_db_dir: str,
    job_root: str,
    resp_profile: str,
    tolerance: float,
    reuse_geometry: bool,
    result_queue,
) -> None:
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        os.environ["OMP_NUM_THREADS"] = str(int(task_payload["psi4_omp"]))  # 设置中间变量或可调参数，供后续工作流使用。
        os.environ["YADONPY_OMP_PSI4"] = str(int(task_payload["psi4_omp"]))  # 设置中间变量或可调参数，供后续工作流使用。
        set_run_options(restart=False)  # 设置全局运行选项，例如 restart。
        spec = SpeciesSpec(  # 设置中间变量或可调参数，供后续工作流使用。
            name=str(task_payload["name"]),  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
            smiles=str(task_payload["smiles"]),  # 设置中间变量或可调参数，供后续工作流使用。
            kind=str(task_payload["kind"]),  # 设置中间变量或可调参数，供后续工作流使用。
            charge=str(task_payload["charge"]),  # 指定电荷来源或电荷计算方式。
            bonded=task_payload.get("bonded"),  # 指定特殊 bonded 参数方案，例如 PF6 的 DRIH。
            polyelectrolyte_mode=bool(task_payload["polyelectrolyte_mode"]),  # 启用聚电解质处理逻辑。
        )
        repo_db = MolDB(Path(repo_db_dir))  # 设置中间变量或可调参数，供后续工作流使用。
        candidate_db = MolDB(Path(candidate_db_dir))  # 设置中间变量或可调参数，供后续工作流使用。
        old_snapshot = _old_resp_snapshot(repo_db, spec)  # 设置中间变量或可调参数，供后续工作流使用。
        mol, geometry_source, fresh = _load_geometry(spec, dbs=[("repo", repo_db)], resp_profile=resp_profile)  # 设置中间变量或可调参数，供后续工作流使用。
        formal_charge = _formal_charge(mol)  # 设置中间变量或可调参数，供后续工作流使用。
        charge_groups = detect_charged_groups(mol, detection="auto", resp_profile=resp_profile)  # 设置中间变量或可调参数，供后续工作流使用。
        species_wd = workdir(Path(job_root) / spec.name / f"attempt_{int(task_payload.get('attempt', 1)):02d}", restart=False)  # 创建或复用本例工作目录。
        optimize = not bool(reuse_geometry)  # 设置中间变量或可调参数，供后续工作流使用。
        print(  # 打印关键路径或状态，便于人工检查。
            f"[RUN] {spec.name:16s} charge={formal_charge:+d} profile={task_payload['profile']} "
            f"omp={task_payload['psi4_omp']} mem={task_payload['memory_mb']}MB "
            f"source={geometry_source} opt={int(optimize)}",
            flush=True,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        ok = assign_charges(  # 执行电荷分配流程。
            mol,
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
            opt=optimize,  # 设置中间变量或可调参数，供后续工作流使用。
            work_dir=species_wd,  # 设置本例输出目录。
            log_name=spec.name,  # 设置中间变量或可调参数，供后续工作流使用。
            omp=int(task_payload["psi4_omp"]),  # 设置每个 rank 的 OpenMP 线程数。
            memory=int(task_payload["memory_mb"]),  # 设置中间变量或可调参数，供后续工作流使用。
            opt_method="wb97m-d3bj",  # 设置中间变量或可调参数，供后续工作流使用。
            charge_method="wb97m-d3bj",  # 设置中间变量或可调参数，供后续工作流使用。
            opt_basis="def2-SVP",  # 设置中间变量或可调参数，供后续工作流使用。
            charge_basis="def2-TZVP",  # 设置中间变量或可调参数，供后续工作流使用。
            auto_level=True,  # 设置中间变量或可调参数，供后续工作流使用。
            geom_iter=int(task_payload.get("geom_iter", 50) or 50),  # 设置中间变量或可调参数，供后续工作流使用。
            total_charge=formal_charge,  # 设置中间变量或可调参数，供后续工作流使用。
            total_multiplicity=1,  # 设置中间变量或可调参数，供后续工作流使用。
            polyelectrolyte_mode=spec.polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
            polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
            bonded_params=(spec.bonded or "ff_assigned"),  # 设置中间变量或可调参数，供后续工作流使用。
            symmetrize=True,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        if not ok:  # 根据当前状态决定是否进入该分支。
            raise RuntimeError(f"assign_charges failed for {spec.name}")  # 关键步骤失败时立即报错，避免继续生成错误结果。

        groups = chem_utils.resp_equivalence_groups_from_mol(mol)  # 设置中间变量或可调参数，供后续工作流使用。
        repaired_groups = chem_utils.symmetrize_equivalent_charge_props(mol, equivalence_groups=groups)  # 设置中间变量或可调参数，供后续工作流使用。
        validation = _validate_equivalence(mol, tolerance=tolerance)  # 设置中间变量或可调参数，供后续工作流使用。
        if not validation["ok"]:  # 根据当前状态决定是否进入该分支。
            raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
                f"{spec.name} equivalence validation failed: "
                f"max_group_spread={validation['max_equivalence_spread']:.3e}, "
                f"max_carboxylate_spread={validation['max_carboxylate_oxygen_spread']:.3e}"
            )

        rec = candidate_db.update_from_mol(  # 设置中间变量或可调参数，供后续工作流使用。
            mol,
            smiles_or_psmiles=spec.smiles,  # 设置中间变量或可调参数，供后续工作流使用。
            name=spec.name,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
            charge="RESP",  # 指定电荷来源或电荷计算方式。
            polyelectrolyte_mode=spec.polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
            polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
            resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        payload = json.loads(candidate_db.charges_path(rec.key).read_text(encoding="utf-8"))  # 设置中间变量或可调参数，供后续工作流使用。
        new_variant_id = str(payload.get("variant_id") or "")  # 设置中间变量或可调参数，供后续工作流使用。
        diff = _charge_diff(  # 设置中间变量或可调参数，供后续工作流使用。
            spec=spec,  # 设置中间变量或可调参数，供后续工作流使用。
            old_snapshot=old_snapshot,  # 设置中间变量或可调参数，供后续工作流使用。
            new_mol=mol,  # 设置中间变量或可调参数，供后续工作流使用。
            validation=validation,  # 设置中间变量或可调参数，供后续工作流使用。
            new_variant_id=new_variant_id,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        recipe = {}  # 设置中间变量或可调参数，供后续工作流使用。
        if mol.HasProp("_yadonpy_qm_recipe_json"):  # 根据当前状态决定是否进入该分支。
            try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
                recipe = json.loads(mol.GetProp("_yadonpy_qm_recipe_json"))  # 设置中间变量或可调参数，供后续工作流使用。
            except Exception:  # 捕获异常并转成更清楚的示例错误信息。
                recipe = {}  # 设置中间变量或可调参数，供后续工作流使用。
        result = {  # 设置中间变量或可调参数，供后续工作流使用。
            "name": spec.name,
            "smiles": spec.smiles,
            "status": "candidate_ready",
            "geometry_source": geometry_source,
            "fresh_geometry": bool(fresh),
            "optimized": bool(optimize),
            "formal_charge": int(formal_charge),
            "polyelectrolyte_mode": bool(spec.polyelectrolyte_mode),
            "charge_group_count": int(len(charge_groups.get("groups") or [])),
            "resp_profile": resp_profile,
            "qm_recipe": recipe,
            "repaired_equivalence_groups": int(repaired_groups),
            "validation": validation,
            "candidate_key": rec.key,
            "candidate_variant_id": new_variant_id,
            "diff_summary": diff["summary"],
            "diff_per_atom": diff["per_atom"],
            "attempt": int(task_payload.get("attempt", 1)),
            "psi4_omp": int(task_payload["psi4_omp"]),
            "memory_mb": int(task_payload["memory_mb"]),
        }
        result_queue.put({"name": spec.name, "ok": True, "result": result, "task": task_payload})
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        result_queue.put({"name": str(task_payload.get("name")), "ok": False, "error": repr(exc), "task": task_payload})


def _is_resp_variant(meta: dict[str, Any] | None) -> bool:  # 定义本例内部辅助函数，组织重复步骤。
    return isinstance(meta, dict) and str(meta.get("charge") or "RESP").strip().upper() == "RESP"  # 返回该辅助函数的结果。


def _hard_replace_repo_record(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    repo_db: MolDB,
    candidate_db: MolDB,
    spec: SpeciesSpec,
    resp_profile: str,
) -> dict[str, Any]:
    candidate_mol, _candidate_rec = candidate_db.load_mol(  # 设置中间变量或可调参数，供后续工作流使用。
        spec.smiles,
        require_ready=True,  # 要求 MolDB 物种必须已准备好。
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        polyelectrolyte_mode=spec.polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
        polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
    )
    rec = repo_db.update_from_mol(  # 设置中间变量或可调参数，供后续工作流使用。
        candidate_mol,
        smiles_or_psmiles=spec.smiles,  # 设置中间变量或可调参数，供后续工作流使用。
        name=spec.name,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        polyelectrolyte_mode=spec.polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
        polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
        resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    payload = json.loads(repo_db.charges_path(rec.key).read_text(encoding="utf-8"))  # 设置中间变量或可调参数，供后续工作流使用。
    keep_resp_vid = str(payload.get("variant_id") or "")  # 设置中间变量或可调参数，供后续工作流使用。
    removed_resp_variant_ids: list[str] = []  # 设置中间变量或可调参数，供后续工作流使用。
    kept: dict[str, dict[str, Any]] = {}  # 设置中间变量或可调参数，供后续工作流使用。
    for vid, meta in sorted((rec.variants or {}).items()):  # 遍历当前工作流中的一组对象或任务。
        if _is_resp_variant(meta) and str(vid) != keep_resp_vid:  # 根据当前状态决定是否进入该分支。
            removed_resp_variant_ids.append(str(vid))
            continue
        kept[str(vid)] = dict(meta)  # 设置中间变量或可调参数，供后续工作流使用。
    rec.variants = kept  # 设置中间变量或可调参数，供后续工作流使用。
    rec.charge_method = "RESP"  # 设置中间变量或可调参数，供后续工作流使用。
    rec.ready = True  # 设置中间变量或可调参数，供后续工作流使用。
    repo_db.save_record(rec)
    for vid in removed_resp_variant_ids:  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            repo_db.charges_variant_path(rec.key, vid).unlink(missing_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
        except Exception:  # 捕获异常并转成更清楚的示例错误信息。
            pass
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            bonded_dir = repo_db.bonded_variant_dir(rec.key, vid)  # 设置中间变量或可调参数，供后续工作流使用。
            if bonded_dir.exists():  # 根据当前状态决定是否进入该分支。
                shutil.rmtree(bonded_dir)
        except Exception:  # 捕获异常并转成更清楚的示例错误信息。
            pass
    return {  # 返回该辅助函数的结果。
        "name": spec.name,
        "key": rec.key,
        "kept_resp_variant_id": keep_resp_vid,
        "removed_resp_variant_ids": removed_resp_variant_ids,
        "remaining_variant_ids": sorted((rec.variants or {}).keys()),
    }


def _make_repo_backup(repo_db: MolDB, backup_dir: Path) -> Path:  # 定义本例内部辅助函数，组织重复步骤。
    backup_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    stamp = time.strftime("%Y%m%d_%H%M%S")  # 设置中间变量或可调参数，供后续工作流使用。
    src = Path(repo_db.db_dir).resolve()  # 设置中间变量或可调参数，供后续工作流使用。
    target = backup_dir / f"moldb_backup_pre_adaptive_resp_{stamp}.tar.zst"  # 设置中间变量或可调参数，供后续工作流使用。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        subprocess.run(  # 开始一个多行函数调用或配置块。
            ["tar", "--zstd", "-cf", str(target), "-C", str(src.parent), src.name],
            check=True,  # 设置中间变量或可调参数，供后续工作流使用。
            capture_output=True,  # 设置中间变量或可调参数，供后续工作流使用。
            text=True,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        return target  # 返回该辅助函数的结果。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        fallback = backup_dir / f"moldb_backup_pre_adaptive_resp_{stamp}.tar.gz"  # 设置中间变量或可调参数，供后续工作流使用。
        subprocess.run(  # 开始一个多行函数调用或配置块。
            ["tar", "-czf", str(fallback), "-C", str(src.parent), src.name],
            check=True,  # 设置中间变量或可调参数，供后续工作流使用。
            capture_output=True,  # 设置中间变量或可调参数，供后续工作流使用。
            text=True,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        return fallback  # 返回该辅助函数的结果。


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:  # 定义本例内部辅助函数，组织重复步骤。
    path.parent.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    with path.open("w", encoding="utf-8", newline="") as fh:  # 用上下文管理器安全打开文件或管理资源。
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")  # 设置中间变量或可调参数，供后续工作流使用。
        writer.writeheader()
        for row in rows:  # 遍历当前工作流中的一组对象或任务。
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_diff_outputs(out_dir: Path, *, results: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, str]:  # 定义本例内部辅助函数，组织重复步骤。
    out_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    summary_rows = [dict(result["diff_summary"]) for result in results]  # 设置中间变量或可调参数，供后续工作流使用。
    per_atom_rows: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for result in results:  # 遍历当前工作流中的一组对象或任务。
        per_atom_rows.extend(dict(row) for row in result.get("diff_per_atom") or [])

    summary_json = out_dir / "charge_diff_summary.json"  # 设置中间变量或可调参数，供后续工作流使用。
    per_atom_json = out_dir / "charge_diff_per_atom.json"  # 设置中间变量或可调参数，供后续工作流使用。
    failures_json = out_dir / "refresh_failures.json"  # 设置中间变量或可调参数，供后续工作流使用。
    summary_csv = out_dir / "charge_diff_summary.csv"  # 设置中间变量或可调参数，供后续工作流使用。
    per_atom_csv = out_dir / "charge_diff_per_atom.csv"  # 设置中间变量或可调参数，供后续工作流使用。
    summary_md = out_dir / "charge_diff_summary.md"  # 设置中间变量或可调参数，供后续工作流使用。

    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
    per_atom_json.write_text(json.dumps(per_atom_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
    failures_json.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
    _write_csv(  # 开始一个多行函数调用或配置块。
        summary_csv,
        summary_rows,
        [
            "species",
            "smiles",
            "atom_count",
            "formal_charge",
            "old_net_charge",
            "new_net_charge",
            "old_resp_profile",
            "new_resp_profile",
            "max_abs_delta",
            "mean_abs_delta",
            "rms_delta",
            "equivalence_ok",
            "max_equivalence_spread",
            "max_carboxylate_oxygen_spread",
            "new_resp_variant_id",
        ],
    )
    _write_csv(  # 开始一个多行函数调用或配置块。
        per_atom_csv,
        per_atom_rows,
        ["species", "atom_index", "symbol", "old_charge", "new_charge", "delta", "abs_delta"],
    )
    lines = [  # 设置中间变量或可调参数，供后续工作流使用。
        "# Adaptive RESP charge refresh diff",
        "",
        "| species | atoms | old profile | max | mean | rms | eq ok | carboxylate spread |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary_rows:  # 遍历当前工作流中的一组对象或任务。
        lines.append(  # 开始一个多行函数调用或配置块。
            "| {species} | {atom_count} | {old_resp_profile} | {max_abs_delta:.6g} | {mean_abs_delta:.6g} | "
            "{rms_delta:.6g} | {equivalence_ok} | {max_carboxylate_oxygen_spread:.3g} |".format(
                species=row.get("species"),  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。
                atom_count=row.get("atom_count"),  # 设置中间变量或可调参数，供后续工作流使用。
                old_resp_profile=row.get("old_resp_profile"),  # 设置中间变量或可调参数，供后续工作流使用。
                max_abs_delta=float(row.get("max_abs_delta") or 0.0),  # 设置中间变量或可调参数，供后续工作流使用。
                mean_abs_delta=float(row.get("mean_abs_delta") or 0.0),  # 设置中间变量或可调参数，供后续工作流使用。
                rms_delta=float(row.get("rms_delta") or 0.0),  # 设置中间变量或可调参数，供后续工作流使用。
                equivalence_ok=row.get("equivalence_ok"),  # 设置中间变量或可调参数，供后续工作流使用。
                max_carboxylate_oxygen_spread=float(row.get("max_carboxylate_oxygen_spread") or 0.0),  # 设置中间变量或可调参数，供后续工作流使用。
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
    return {  # 返回该辅助函数的结果。
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "per_atom_json": str(per_atom_json),
        "per_atom_csv": str(per_atom_csv),
        "failures_json": str(failures_json),
    }


def _candidate_result_from_db(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    repo_db: MolDB,
    candidate_db: MolDB,
    spec: SpeciesSpec,
    resp_profile: str,
    tolerance: float,
) -> dict[str, Any]:
    """Reconstruct a refresh result from an existing candidate MolDB record."""
    old_snapshot = _old_resp_snapshot(repo_db, spec)  # 设置中间变量或可调参数，供后续工作流使用。
    candidate_mol, rec = candidate_db.load_mol(  # 设置中间变量或可调参数，供后续工作流使用。
        spec.smiles,
        require_ready=True,  # 要求 MolDB 物种必须已准备好。
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        polyelectrolyte_mode=spec.polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
        polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
    )
    validation = _validate_equivalence(candidate_mol, tolerance=tolerance)  # 设置中间变量或可调参数，供后续工作流使用。
    if not validation["ok"]:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
            f"{spec.name} equivalence validation failed: "
            f"max_group_spread={validation['max_equivalence_spread']:.3e}, "
            f"max_carboxylate_spread={validation['max_carboxylate_oxygen_spread']:.3e}"
        )
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        payload = json.loads(candidate_db.charges_path(rec.key).read_text(encoding="utf-8"))  # 设置中间变量或可调参数，供后续工作流使用。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        payload = {}  # 设置中间变量或可调参数，供后续工作流使用。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        recipe = json.loads(candidate_mol.GetProp("_yadonpy_qm_recipe_json"))  # 设置中间变量或可调参数，供后续工作流使用。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        recipe = {}  # 设置中间变量或可调参数，供后续工作流使用。
    charge_groups = detect_charged_groups(candidate_mol, detection="auto", resp_profile=resp_profile)  # 设置中间变量或可调参数，供后续工作流使用。
    diff = _charge_diff(  # 设置中间变量或可调参数，供后续工作流使用。
        spec=spec,  # 设置中间变量或可调参数，供后续工作流使用。
        old_snapshot=old_snapshot,  # 设置中间变量或可调参数，供后续工作流使用。
        new_mol=candidate_mol,  # 设置中间变量或可调参数，供后续工作流使用。
        validation=validation,  # 设置中间变量或可调参数，供后续工作流使用。
        new_variant_id=str(payload.get("variant_id") or ""),  # 设置中间变量或可调参数，供后续工作流使用。
    )
    return {  # 返回该辅助函数的结果。
        "name": spec.name,
        "smiles": spec.smiles,
        "formal_charge": _formal_charge(candidate_mol),
        "polyelectrolyte_mode": bool(spec.polyelectrolyte_mode),
        "charge_group_count": int(len(charge_groups.get("groups") or [])),
        "resp_profile": resp_profile,
        "qm_recipe": recipe,
        "repaired_equivalence_groups": 0,
        "validation": validation,
        "candidate_key": rec.key,
        "candidate_variant_id": str(payload.get("variant_id") or ""),
        "diff_summary": diff["summary"],
        "diff_per_atom": diff["per_atom"],
        "attempt": None,
        "psi4_omp": None,
        "memory_mb": None,
        "source": "candidate_moldb",
    }


def _finalize_candidate_moldb(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    repo_db: MolDB,
    candidate_db_dir: Path,
    specs: list[SpeciesSpec],
    resp_profile: str,
    tolerance: float,
    diff_dir: Path,
    no_commit: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Validate candidate records, write diffs, and optionally hard-replace repo RESP variants."""
    candidate_db = MolDB(candidate_db_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    results: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    failures: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    for spec in specs:  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            results.append(  # 开始一个多行函数调用或配置块。
                _candidate_result_from_db(  # 开始一个多行函数调用或配置块。
                    repo_db=repo_db,  # 设置中间变量或可调参数，供后续工作流使用。
                    candidate_db=candidate_db,  # 设置中间变量或可调参数，供后续工作流使用。
                    spec=spec,  # 设置中间变量或可调参数，供后续工作流使用。
                    resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
                    tolerance=float(tolerance),  # 设置中间变量或可调参数，供后续工作流使用。
                )
            )
            print(f"[CANDIDATE] {spec.name:16s} ready", flush=True)  # 打印关键路径或状态，便于人工检查。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            failures.append({"name": spec.name, "smiles": spec.smiles, "error": repr(exc)})
            print(f"[MISSING]   {spec.name:16s} {exc!r}", flush=True)  # 打印关键路径或状态，便于人工检查。

    diff_outputs = _write_diff_outputs(diff_dir, results=results, failures=failures)  # 设置中间变量或可调参数，供后续工作流使用。
    replacements: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    if failures:  # 根据当前状态决定是否进入该分支。
        print("[STOP] Candidate MolDB is incomplete/invalid; repo MolDB was not modified.", flush=True)  # 打印关键路径或状态，便于人工检查。
        return results, failures, diff_outputs, replacements  # 返回该辅助函数的结果。
    if no_commit:  # 根据当前状态决定是否进入该分支。
        print("[STOP] --no-commit requested; candidate MolDB and charge diffs are ready, repo MolDB unchanged.", flush=True)  # 打印关键路径或状态，便于人工检查。
        return results, failures, diff_outputs, replacements  # 返回该辅助函数的结果。

    for spec in specs:  # 遍历当前工作流中的一组对象或任务。
        replacements.append(  # 开始一个多行函数调用或配置块。
            _hard_replace_repo_record(  # 开始一个多行函数调用或配置块。
                repo_db=repo_db,  # 设置中间变量或可调参数，供后续工作流使用。
                candidate_db=candidate_db,  # 设置中间变量或可调参数，供后续工作流使用。
                spec=spec,  # 设置中间变量或可调参数，供后续工作流使用。
                resp_profile=resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
            )
        )
        print(f"[REPLACE] {spec.name:16s} RESP variants hard-replaced", flush=True)  # 打印关键路径或状态，便于人工检查。
    return results, failures, diff_outputs, replacements  # 返回该辅助函数的结果。


def _run_parallel_refresh(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    tasks: list[RefreshTask],
    repo_db_dir: Path,
    candidate_db_dir: Path,
    job_root: Path,
    resp_profile: str,
    tolerance: float,
    reuse_geometry: bool,
    geom_iter: int,
    retry_geom_iter: int,
    planner_cpu_budget: int,
    planner_memory_budget_mb: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    ctx = mp.get_context("spawn")  # 设置中间变量或可调参数，供后续工作流使用。
    result_queue = ctx.Queue()  # 设置中间变量或可调参数，供后续工作流使用。
    pending = _pending_payloads(tasks)  # 设置中间变量或可调参数，供后续工作流使用。
    for item in pending:  # 遍历当前工作流中的一组对象或任务。
        item["geom_iter"] = int(geom_iter)  # 设置中间变量或可调参数，供后续工作流使用。
        item["retry_geom_iter"] = int(retry_geom_iter)  # 设置中间变量或可调参数，供后续工作流使用。
    running: dict[str, dict[str, Any]] = {}  # 设置中间变量或可调参数，供后续工作流使用。
    available_cores = int(planner_cpu_budget)  # 设置中间变量或可调参数，供后续工作流使用。
    available_memory_mb = int(planner_memory_budget_mb)  # 设置中间变量或可调参数，供后续工作流使用。
    results: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    failures: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    retry_count = 0  # 设置中间变量或可调参数，供后续工作流使用。

    while pending or running:  # 循环执行直到当前条件不再满足。
        launched = False  # 设置中间变量或可调参数，供后续工作流使用。
        for task in list(_eligible_pending_for_launch(pending, available_cores, available_memory_mb)):  # 遍历当前工作流中的一组对象或任务。
            required = int(task["required_cores"])  # 设置中间变量或可调参数，供后续工作流使用。
            memory_mb = int(task["memory_mb"])  # 设置中间变量或可调参数，供后续工作流使用。
            if required > available_cores or memory_mb > available_memory_mb:  # 根据当前状态决定是否进入该分支。
                continue
            proc = ctx.Process(  # 设置中间变量或可调参数，供后续工作流使用。
                target=_worker_refresh_candidate,  # 设置中间变量或可调参数，供后续工作流使用。
                kwargs={  # 设置中间变量或可调参数，供后续工作流使用。
                    "task_payload": task,
                    "repo_db_dir": str(repo_db_dir),
                    "candidate_db_dir": str(candidate_db_dir),
                    "job_root": str(job_root),
                    "resp_profile": resp_profile,
                    "tolerance": float(tolerance),
                    "reuse_geometry": bool(reuse_geometry),
                    "result_queue": result_queue,
                },
            )
            proc.start()
            running[str(task["name"])] = {  # 设置中间变量或可调参数，供后续工作流使用。
                "process": proc,
                "required_cores": required,
                "memory_mb": memory_mb,
                "task": task,
            }
            available_cores -= required  # 设置中间变量或可调参数，供后续工作流使用。
            available_memory_mb -= memory_mb  # 设置中间变量或可调参数，供后续工作流使用。
            pending.remove(task)
            launched = True  # 设置中间变量或可调参数，供后续工作流使用。
            print(  # 打印关键路径或状态，便于人工检查。
                f"[START] {task['name']:16s} profile={task['profile']:15s} "
                f"attempt={task['attempt']}/{task['max_attempts']} "
                f"cores={required:2d} mem={memory_mb:5d}MB "
                f"available={available_cores:2d}/{available_memory_mb:5d}MB",
                flush=True,  # 设置中间变量或可调参数，供后续工作流使用。
            )

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
                available_memory_mb += int(state["memory_mb"])  # 设置中间变量或可调参数，供后续工作流使用。
            if bool(message.get("ok")):  # 根据当前状态决定是否进入该分支。
                result = dict(message["result"])  # 设置中间变量或可调参数，供后续工作流使用。
                results.append(result)
                print(  # 打印关键路径或状态，便于人工检查。
                    f"[DONE]  {name:16s} available={available_cores:2d}/{available_memory_mb:5d}MB "
                    f"attempt={result.get('attempt')}",
                    flush=True,  # 设置中间变量或可调参数，供后续工作流使用。
                )
            else:  # 处理前面条件都不满足的情况。
                retry_task = _maybe_schedule_retry(state["task"], error=str(message.get("error"))) if state else None  # 设置中间变量或可调参数，供后续工作流使用。
                if retry_task is not None:  # 根据当前状态决定是否进入该分支。
                    retry_count += 1  # 设置中间变量或可调参数，供后续工作流使用。
                    pending.append(retry_task)
                    _sort_pending_in_place(pending)
                    print(  # 打印关键路径或状态，便于人工检查。
                        f"[RETRY] {name:16s} failed; retrying with {retry_task['required_cores']} cores "
                        f"(attempt {retry_task['attempt']}/{retry_task['max_attempts']})",
                        flush=True,  # 设置中间变量或可调参数，供后续工作流使用。
                    )
                else:  # 处理前面条件都不满足的情况。
                    failures.append(  # 开始一个多行函数调用或配置块。
                        {
                            "name": name,
                            "smiles": (state["task"]["smiles"] if state else None),
                            "attempt": int((state or {}).get("task", {}).get("attempt", message.get("attempt", 1))),
                            "error": message.get("error"),
                        }
                    )
                    print(f"[FAIL]  {name:16s} {message.get('error')}", flush=True)  # 打印关键路径或状态，便于人工检查。

        for name, state in list(running.items()):  # 遍历当前工作流中的一组对象或任务。
            proc = state["process"]  # 设置中间变量或可调参数，供后续工作流使用。
            if proc.is_alive():  # 根据当前状态决定是否进入该分支。
                continue
            proc.join(timeout=0.1)  # 设置中间变量或可调参数，供后续工作流使用。
            running.pop(name, None)
            available_cores += int(state["required_cores"])  # 设置中间变量或可调参数，供后续工作流使用。
            available_memory_mb += int(state["memory_mb"])  # 设置中间变量或可调参数，供后续工作流使用。
            error = f"worker exited without reporting (exitcode={proc.exitcode})"  # 设置中间变量或可调参数，供后续工作流使用。
            retry_task = _maybe_schedule_retry(state["task"], error=error)  # 设置中间变量或可调参数，供后续工作流使用。
            if retry_task is not None:  # 根据当前状态决定是否进入该分支。
                retry_count += 1  # 设置中间变量或可调参数，供后续工作流使用。
                pending.append(retry_task)
                _sort_pending_in_place(pending)
                print(f"[RETRY] {name:16s} {error}; retrying", flush=True)  # 打印关键路径或状态，便于人工检查。
            else:  # 处理前面条件都不满足的情况。
                failures.append(  # 开始一个多行函数调用或配置块。
                    {
                        "name": name,
                        "smiles": state["task"]["smiles"],
                        "attempt": int(state["task"].get("attempt", 1)),
                        "error": error,
                    }
                )
                print(f"[FAIL]  {name:16s} {error}", flush=True)  # 打印关键路径或状态，便于人工检查。

        if not launched and not running and pending:  # 根据当前状态决定是否进入该分支。
            smallest = min(pending, key=lambda item: (int(item["required_cores"]), int(item.get("memory_mb", 0))))  # 设置中间变量或可调参数，供后续工作流使用。
            if (  # 根据当前状态决定是否进入该分支。
                int(smallest["required_cores"]) <= int(available_cores)
                and int(smallest.get("memory_mb", 0)) <= int(available_memory_mb)
            ):
                # A just-finished worker may have freed enough resources after
                # the launch pass at the top of this loop. Continue so the next
                # iteration can launch the now-eligible task instead of raising
                # a false resource-budget error.
                continue
            raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
                "No pending RESP refresh task fits the available resource budget. "
                f"Smallest pending task is {smallest['name']} requiring "
                f"{smallest['required_cores']} cores / {smallest.get('memory_mb')} MB, "
                f"but only {available_cores} cores / {available_memory_mb} MB are available. "
                "Reduce reserve resources or increase the job memory budget."
            )
        if not launched and running:  # 根据当前状态决定是否进入该分支。
            time.sleep(0.1)

    results.sort(key=lambda item: str(item.get("name", "")).lower())  # 设置中间变量或可调参数，供后续工作流使用。
    failures.sort(key=lambda item: str(item.get("name", "")).lower())  # 设置中间变量或可调参数，供后续工作流使用。
    return results, failures, retry_count  # 返回该辅助函数的结果。


def refresh_one(  # 定义本例内部辅助函数，组织重复步骤。
    spec: SpeciesSpec,
    *,
    default_db: MolDB | None,
    repo_db: MolDB,
    job_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Backward-compatible single-species candidate refresh helper.

    The new production path uses the candidate-DB worker above. This wrapper is
    kept for older tests and ad-hoc debugging, but now writes only to ``repo_db``.
    """

    _ = default_db  # 设置中间变量或可调参数，供后续工作流使用。
    candidate_db = repo_db  # 设置中间变量或可调参数，供后续工作流使用。
    old_snapshot = _old_resp_snapshot(repo_db, spec)  # 设置中间变量或可调参数，供后续工作流使用。
    mol, geometry_source, fresh = _load_geometry(spec, dbs=[("repo", repo_db)], resp_profile=args.resp_profile)  # 设置中间变量或可调参数，供后续工作流使用。
    formal_charge = _formal_charge(mol)  # 设置中间变量或可调参数，供后续工作流使用。
    charge_groups = detect_charged_groups(mol, detection="auto", resp_profile=args.resp_profile)  # 设置中间变量或可调参数，供后续工作流使用。
    needs_new_opt = not bool(getattr(args, "reuse_geometry", False) or getattr(args, "no_opt", False))  # 设置中间变量或可调参数，供后续工作流使用。
    print(  # 打印关键路径或状态，便于人工检查。
        f"[RUN] {spec.name:12s} charge={formal_charge:+d} "
        f"poly={int(spec.polyelectrolyte_mode)} groups={len(charge_groups.get('groups') or [])} "
        f"source={geometry_source} opt={int(needs_new_opt)}"
    )
    ok = assign_charges(  # 执行电荷分配流程。
        mol,
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        resp_profile=args.resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        opt=needs_new_opt,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=workdir(job_root / spec.name, restart=False),  # 设置本例输出目录。
        log_name=spec.name,  # 设置中间变量或可调参数，供后续工作流使用。
        omp=args.omp,  # 设置每个 rank 的 OpenMP 线程数。
        memory=args.memory_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        opt_method="wb97m-d3bj",  # 设置中间变量或可调参数，供后续工作流使用。
        charge_method="wb97m-d3bj",  # 设置中间变量或可调参数，供后续工作流使用。
        opt_basis="def2-SVP",  # 设置中间变量或可调参数，供后续工作流使用。
        charge_basis="def2-TZVP",  # 设置中间变量或可调参数，供后续工作流使用。
        auto_level=True,  # 设置中间变量或可调参数，供后续工作流使用。
        total_charge=formal_charge,  # 设置中间变量或可调参数，供后续工作流使用。
        total_multiplicity=1,  # 设置中间变量或可调参数，供后续工作流使用。
        polyelectrolyte_mode=spec.polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
        polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
        bonded_params=(spec.bonded or "ff_assigned"),  # 设置中间变量或可调参数，供后续工作流使用。
        symmetrize=True,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    if not ok:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"assign_charges failed for {spec.name}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    groups = chem_utils.resp_equivalence_groups_from_mol(mol)  # 设置中间变量或可调参数，供后续工作流使用。
    repaired_groups = chem_utils.symmetrize_equivalent_charge_props(mol, equivalence_groups=groups)  # 设置中间变量或可调参数，供后续工作流使用。
    validation = _validate_equivalence(mol, tolerance=args.tolerance)  # 设置中间变量或可调参数，供后续工作流使用。
    if not validation["ok"]:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"{spec.name} equivalence validation failed")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    rec = candidate_db.update_from_mol(  # 设置中间变量或可调参数，供后续工作流使用。
        mol,
        smiles_or_psmiles=spec.smiles,  # 设置中间变量或可调参数，供后续工作流使用。
        name=spec.name,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        polyelectrolyte_mode=spec.polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
        polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
        resp_profile=args.resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    payload = json.loads(candidate_db.charges_path(rec.key).read_text(encoding="utf-8"))  # 设置中间变量或可调参数，供后续工作流使用。
    diff = _charge_diff(  # 设置中间变量或可调参数，供后续工作流使用。
        spec=spec,  # 设置中间变量或可调参数，供后续工作流使用。
        old_snapshot=old_snapshot,  # 设置中间变量或可调参数，供后续工作流使用。
        new_mol=mol,  # 设置中间变量或可调参数，供后续工作流使用。
        validation=validation,  # 设置中间变量或可调参数，供后续工作流使用。
        new_variant_id=str(payload.get("variant_id") or ""),  # 设置中间变量或可调参数，供后续工作流使用。
    )
    return {  # 返回该辅助函数的结果。
        "name": spec.name,
        "smiles": spec.smiles,
        "status": "refreshed",
        "geometry_source": geometry_source,
        "fresh_geometry": bool(fresh),
        "optimized": bool(needs_new_opt),
        "formal_charge": int(formal_charge),
        "polyelectrolyte_mode": bool(spec.polyelectrolyte_mode),
        "charge_group_count": int(len(charge_groups.get("groups") or [])),
        "resp_profile": args.resp_profile,
        "repaired_equivalence_groups": int(repaired_groups),
        "validation": validation,
        "candidate_key": rec.key,
        "candidate_variant_id": str(payload.get("variant_id") or ""),
        "diff_summary": diff["summary"],
        "diff_per_atom": diff["per_atom"],
    }


def build_arg_parser() -> argparse.ArgumentParser:  # 定义本例内部辅助函数，组织重复步骤。
    parser = argparse.ArgumentParser(description=__doc__)  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--only", nargs="*", help="Species names or comma-separated name lists.")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--target-mode", choices=("existing-repo", "default-targets"), default="existing-repo")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--repo-db", type=Path, default=REPO_ROOT / "moldb")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--work-dir", type=Path, default=HERE / "work_dir" / "04_refresh_adaptive_resp_moldb")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--summary", type=Path, default=None)  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--resp-profile", choices=("adaptive", "legacy"), default="adaptive")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--max-omp", type=int, default=_env_default("YADONPY_REFRESH_MAX_OMP", 12))  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--reserve-cores", type=int, default=_env_default("YADONPY_REFRESH_RESERVE_CORES", 4))  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--reserve-memory-mb", type=int, default=_env_default("YADONPY_REFRESH_RESERVE_MEMORY_MB", 16000))  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--max-memory-mb", type=int, default=_env_default("YADONPY_REFRESH_MAX_MEMORY_MB", 24000))  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--omp", type=int, default=_env_default("YADONPY_PSI4_OMP", 8), help="Single-species compatibility mode OMP.")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--memory-mb", type=int, default=_env_default("YADONPY_PSI4_MEMORY_MB", 12000), help="Single-species compatibility mode memory.")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--tolerance", type=float, default=1.0e-8)  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--reuse-geometry", action="store_true", help="Debug only: skip DFT optimization and refit RESP on existing geometry.")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--force-opt", action="store_true", help="Compatibility flag; full refresh already optimizes by default.")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--no-opt", action="store_true", help="Alias for --reuse-geometry.")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--geom-iter", type=int, default=50, help="Psi4 geometry optimization iteration limit.")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--retry-geom-iter", type=int, default=120, help="Geometry iteration limit used on retry attempts.")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument(  # 开始一个多行函数调用或配置块。
        "--finalize-candidates",
        action="store_true",  # 设置中间变量或可调参数，供后续工作流使用。
        help="Do not run QM. Validate the existing candidate MolDB, write diffs, and optionally hard-replace repo RESP variants.",  # 设置中间变量或可调参数，供后续工作流使用。
    )
    parser.add_argument("--dry-run", action="store_true", help="Write plan only; do not run QM.")  # 设置中间变量或可调参数，供后续工作流使用。
    parser.add_argument("--no-commit", action="store_true", help="Do not hard-replace repo MolDB after candidate success.")  # 设置中间变量或可调参数，供后续工作流使用。
    return parser  # 返回该辅助函数的结果。


def main(argv: list[str] | None = None) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    args = build_arg_parser().parse_args(argv)  # 设置中间变量或可调参数，供后续工作流使用。
    if args.no_opt:  # 根据当前状态决定是否进入该分支。
        args.reuse_geometry = True  # 设置中间变量或可调参数，供后续工作流使用。

    set_run_options(restart=False)  # 设置全局运行选项，例如 restart。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    catalog = _read_catalog(CATALOG_CSV)  # 设置中间变量或可调参数，供后续工作流使用。
    repo_db = MolDB(args.repo_db)  # 设置中间变量或可调参数，供后续工作流使用。
    selected_names = _parse_only(args.only)  # 设置中间变量或可调参数，供后续工作流使用。
    specs = _selected_specs(catalog=catalog, repo_db=repo_db, only=selected_names, target_mode=args.target_mode)  # 设置中间变量或可调参数，供后续工作流使用。
    if not specs:  # 根据当前状态决定是否进入该分支。
        raise SystemExit("No existing repo MolDB entries matched the selected Example 07 catalog species.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    job_root = Path(workdir(args.work_dir, restart=False))  # 创建或复用本例工作目录。
    candidate_db_dir = job_root / "candidate_moldb"  # 设置中间变量或可调参数，供后续工作流使用。
    backup_dir = job_root / "backups"  # 设置中间变量或可调参数，供后续工作流使用。
    diff_dir = job_root / "charge_diffs"  # 设置中间变量或可调参数，供后续工作流使用。
    summary_path = args.summary or (job_root / "adaptive_resp_moldb_refresh_summary.json")  # 设置中间变量或可调参数，供后续工作流使用。

    cpu_total = _available_cpu_total()  # 设置中间变量或可调参数，供后续工作流使用。
    memory_total_mb = _available_memory_mb()  # 设置中间变量或可调参数，供后续工作流使用。
    planner_cpu_budget = max(1, int(cpu_total) - max(0, int(args.reserve_cores)))  # 设置中间变量或可调参数，供后续工作流使用。
    planner_memory_budget_mb = max(4096, int(memory_total_mb) - max(0, int(args.reserve_memory_mb)))  # 设置中间变量或可调参数，供后续工作流使用。
    tasks = _build_refresh_tasks(  # 设置中间变量或可调参数，供后续工作流使用。
        specs,
        cpu_budget=planner_cpu_budget,  # 设置中间变量或可调参数，供后续工作流使用。
        max_omp=max(1, int(args.max_omp)),  # 设置中间变量或可调参数，供后续工作流使用。
        max_memory_mb=max(4096, int(args.max_memory_mb)),  # 设置中间变量或可调参数，供后续工作流使用。
    )
    plan = {  # 设置中间变量或可调参数，供后续工作流使用。
        "catalog_csv": str(CATALOG_CSV.resolve()),
        "repo_db": str(Path(repo_db.db_dir).resolve()),
        "candidate_db": str(candidate_db_dir.resolve()),
        "work_dir": str(job_root.resolve()),
        "resp_profile": args.resp_profile,
        "target_mode": args.target_mode,
        "target_count": len(tasks),
        "cpu_total": int(cpu_total),
        "memory_total_mb": int(memory_total_mb),
        "reserve_cores": int(args.reserve_cores),
        "reserve_memory_mb": int(args.reserve_memory_mb),
        "planner_cpu_budget": int(planner_cpu_budget),
        "planner_memory_budget_mb": int(planner_memory_budget_mb),
        "reuse_geometry": bool(args.reuse_geometry),
        "geom_iter": int(args.geom_iter),
        "retry_geom_iter": int(args.retry_geom_iter),
        "commit": not bool(args.no_commit),
        "tasks": [asdict(task) for task in tasks],
    }
    (job_root / "parallel_refresh_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。

    print(f"[DB] repo      = {repo_db.db_dir}")  # 打印关键路径或状态，便于人工检查。
    print(f"[DB] candidate = {candidate_db_dir}")  # 打印关键路径或状态，便于人工检查。
    print(f"[RUN] targets  = {', '.join(task.name for task in tasks)}")  # 打印关键路径或状态，便于人工检查。
    print(  # 打印关键路径或状态，便于人工检查。
        f"[PLAN] cpu_total={cpu_total} budget={planner_cpu_budget} "
        f"memory_total={memory_total_mb}MB budget={planner_memory_budget_mb}MB"
    )
    for task in tasks:  # 遍历当前工作流中的一组对象或任务。
        print(  # 打印关键路径或状态，便于人工检查。
            f"[PLAN] {task.name:16s} profile={task.profile:15s} heavy={task.heavy_atoms:2d} "
            f"charge={task.formal_charge:+d} omp={task.psi4_omp:2d} mem={task.memory_mb:5d}MB"
        )

    if args.dry_run:  # 根据当前状态决定是否进入该分支。
        print(f"[SUMMARY] dry-run plan written to {job_root / 'parallel_refresh_plan.json'}")  # 打印关键路径或状态，便于人工检查。
        return 0  # 返回该辅助函数的结果。

    backup_path = _make_repo_backup(repo_db, backup_dir)  # 设置中间变量或可调参数，供后续工作流使用。
    print(f"[BACKUP] {backup_path}")  # 打印关键路径或状态，便于人工检查。

    if args.finalize_candidates:  # 根据当前状态决定是否进入该分支。
        results, failures, diff_outputs, replacements = _finalize_candidate_moldb(  # 设置中间变量或可调参数，供后续工作流使用。
            repo_db=repo_db,  # 设置中间变量或可调参数，供后续工作流使用。
            candidate_db_dir=candidate_db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
            specs=specs,  # 设置中间变量或可调参数，供后续工作流使用。
            resp_profile=args.resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
            tolerance=float(args.tolerance),  # 设置中间变量或可调参数，供后续工作流使用。
            diff_dir=diff_dir,  # 设置中间变量或可调参数，供后续工作流使用。
            no_commit=bool(args.no_commit),  # 设置中间变量或可调参数，供后续工作流使用。
        )
        out = {  # 设置中间变量或可调参数，供后续工作流使用。
            **plan,
            "backup_path": str(backup_path),
            "success_count": len(results),
            "failure_count": len(failures),
            "retry_count": 0,
            "committed": bool((not failures) and (not args.no_commit)),
            "finalize_candidates": True,
            "diff_outputs": diff_outputs,
            "results": [
                {key: value for key, value in result.items() if key != "diff_per_atom"}
                for result in results  # 遍历当前工作流中的一组对象或任务。
            ],
            "failures": failures,
            "replacements": replacements,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
        summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
        print(f"[SUMMARY] {summary_path}")  # 打印关键路径或状态，便于人工检查。
        return 2 if failures else 0  # 返回该辅助函数的结果。

    results, failures, retry_count = _run_parallel_refresh(  # 设置中间变量或可调参数，供后续工作流使用。
        tasks=tasks,  # 设置中间变量或可调参数，供后续工作流使用。
        repo_db_dir=Path(repo_db.db_dir),  # 设置中间变量或可调参数，供后续工作流使用。
        candidate_db_dir=candidate_db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        job_root=job_root / "qm_jobs",  # 设置中间变量或可调参数，供后续工作流使用。
        resp_profile=args.resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        tolerance=float(args.tolerance),  # 设置中间变量或可调参数，供后续工作流使用。
        reuse_geometry=bool(args.reuse_geometry),  # 设置中间变量或可调参数，供后续工作流使用。
        geom_iter=int(args.geom_iter),  # 设置中间变量或可调参数，供后续工作流使用。
        retry_geom_iter=int(args.retry_geom_iter),  # 设置中间变量或可调参数，供后续工作流使用。
        planner_cpu_budget=planner_cpu_budget,  # 设置中间变量或可调参数，供后续工作流使用。
        planner_memory_budget_mb=planner_memory_budget_mb,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    diff_outputs = _write_diff_outputs(diff_dir, results=results, failures=failures)  # 设置中间变量或可调参数，供后续工作流使用。

    replacements: list[dict[str, Any]] = []  # 设置中间变量或可调参数，供后续工作流使用。
    if failures:  # 根据当前状态决定是否进入该分支。
        print("[STOP] At least one species failed; repo MolDB was not modified beyond the pre-run backup.")  # 打印关键路径或状态，便于人工检查。
    elif args.no_commit:  # 继续判断另一个互斥分支。
        print("[STOP] --no-commit requested; candidate MolDB and charge diffs are ready, repo MolDB unchanged.")  # 打印关键路径或状态，便于人工检查。
    else:  # 处理前面条件都不满足的情况。
        candidate_db = MolDB(candidate_db_dir)  # 设置中间变量或可调参数，供后续工作流使用。
        spec_by_name = {spec.name: spec for spec in specs}  # 设置中间变量或可调参数，供后续工作流使用。
        for result in results:  # 遍历当前工作流中的一组对象或任务。
            replacements.append(  # 开始一个多行函数调用或配置块。
                _hard_replace_repo_record(  # 开始一个多行函数调用或配置块。
                    repo_db=repo_db,  # 设置中间变量或可调参数，供后续工作流使用。
                    candidate_db=candidate_db,  # 设置中间变量或可调参数，供后续工作流使用。
                    spec=spec_by_name[str(result["name"])],  # 设置中间变量或可调参数，供后续工作流使用。
                    resp_profile=args.resp_profile,  # 设置中间变量或可调参数，供后续工作流使用。
                )
            )
            print(f"[REPLACE] {result['name']:16s} RESP variants hard-replaced")  # 打印关键路径或状态，便于人工检查。

    out = {  # 设置中间变量或可调参数，供后续工作流使用。
        **plan,
        "backup_path": str(backup_path),
        "success_count": len(results),
        "failure_count": len(failures),
        "retry_count": int(retry_count),
        "committed": bool((not failures) and (not args.no_commit)),
        "diff_outputs": diff_outputs,
        "results": [
            {key: value for key, value in result.items() if key != "diff_per_atom"}
            for result in results  # 遍历当前工作流中的一组对象或任务。
        ],
        "failures": failures,
        "replacements": replacements,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
    print(f"[SUMMARY] {summary_path}")  # 打印关键路径或状态，便于人工检查。
    if failures:  # 根据当前状态决定是否进入该分支。
        return 1  # 返回该辅助函数的结果。
    return 0  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    raise SystemExit(main())  # 关键步骤失败时立即报错，避免继续生成错误结果。
