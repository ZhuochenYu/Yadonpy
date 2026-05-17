# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""CMC-Na / carbonate / LiPF6 bulk benchmark.

This script is intentionally close to the example02/example05 style: user
settings live near the top, MoldDB-ready species are reused, and no QM/RESP is
started from the benchmark script.  It builds a swollen CMC-Na bulk box and
reports transport with explicit MSD semantics:

- Li and Na: single-ion atomic MSD.
- PF6, EC, EMC, DEC: molecular COM MSD.
- CMC: whole-chain COM MSD is the polymer self-diffusion observable; residue
  and charged-group COM MSDs are local mobility diagnostics.
"""

from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

import csv  # 导入本例需要的库或 yadonpy 接口。
import json  # 导入本例需要的库或 yadonpy 接口。
import math  # 导入本例需要的库或 yadonpy 接口。
import os  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。
from typing import Any  # 导入本例需要的库或 yadonpy 接口。

import numpy as np  # 导入本例需要的库或 yadonpy 接口。
from rdkit.Chem import Descriptors  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.core import naming, poly, utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.chem_utils import correct_total_charge, symmetrize_equivalent_charge_props  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.polyelectrolyte import annotate_polyelectrolyte_metadata  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.polymer_audit import audit_polymer_state, compare_exported_charge_groups, write_polymer_audit  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff import GAFF2, GAFF2_mod, MERZ, OPLSAA  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.oplsaa_reference import audit_oplsaa_assignment  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.analyzer import AnalyzeResult  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.preset import eq  # 导入本例需要的库或 yadonpy 接口。


def _env_flag(name: str, default: bool = False) -> bool:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    token = str(os.environ.get(name, "")).strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if not token:  # 根据当前状态决定是否进入该分支。
        return bool(default)  # 返回该辅助函数的结果。
    return token in {"1", "true", "t", "yes", "y", "on"}  # 返回该辅助函数的结果。


def _env_int(name: str, default: int) -> int:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    return int(raw) if raw else int(default)  # 返回该辅助函数的结果。


def _env_float(name: str, default: float) -> float:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    return float(raw) if raw else float(default)  # 返回该辅助函数的结果。


def _env_optional_float(name: str) -> float | None:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    return float(raw) if raw else None  # 返回该辅助函数的结果。


def _env_text(name: str, default: str) -> str:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    return raw if raw else str(default)  # 返回该辅助函数的结果。


def _normalize_forcefield(raw: str) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    token = str(raw or "gaff2").strip().lower().replace("-", "_")  # 设置中间变量或可调参数，供后续工作流使用。
    if token in {"gaff2", "gaff"}:  # 根据当前状态决定是否进入该分支。
        return "gaff2"  # 返回该辅助函数的结果。
    if token in {"gaff2_mod", "mod"}:  # 根据当前状态决定是否进入该分支。
        return "gaff2_mod"  # 返回该辅助函数的结果。
    if token in {"opls", "oplsaa", "opls_aa"}:  # 根据当前状态决定是否进入该分支。
        return "oplsaa"  # 返回该辅助函数的结果。
    raise ValueError("YADONPY_FORCEFIELD must be gaff2, gaff2_mod, or oplsaa")  # 关键步骤失败时立即报错，避免继续生成错误结果。


def _formal_charge(mol) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))  # 返回该辅助函数的结果。


def _net_charge(mol, prop: str = "AtomicCharge") -> float:  # 定义本例内部辅助函数，组织重复步骤。
    total = 0.0  # 设置中间变量或可调参数，供后续工作流使用。
    for atom in mol.GetAtoms():  # 遍历当前工作流中的一组对象或任务。
        if atom.HasProp(prop):  # 根据当前状态决定是否进入该分支。
            total += float(atom.GetDoubleProp(prop))  # 设置中间变量或可调参数，供后续工作流使用。
    return float(total)  # 返回该辅助函数的结果。


def _mol_weight(mol) -> float:  # 定义本例内部辅助函数，组织重复步骤。
    return float(Descriptors.MolWt(mol))  # 返回该辅助函数的结果。


def _set_zero_charge_props(mol) -> None:  # 定义本例内部辅助函数，组织重复步骤。
    for atom in mol.GetAtoms():  # 遍历当前工作流中的一组对象或任务。
        for prop in ("AtomicCharge", "RESP", "RESP2", "ESP"):  # 遍历当前工作流中的一组对象或任务。
            atom.SetDoubleProp(prop, 0.0)


def _zero_charge_terminator(smiles: str):  # 定义本例内部辅助函数，组织重复步骤。
    mol = utils.mol_from_smiles(smiles)  # 从 SMILES 直接构造 RDKit 分子。
    _set_zero_charge_props(mol)
    naming.ensure_name(mol, name="TER", prefer_var=False)  # 设置中间变量或可调参数，供后续工作流使用。
    return mol  # 返回该辅助函数的结果。


def _load_ready_gaff_species(  # 定义本例内部辅助函数，组织重复步骤。
    ff,
    smiles: str,
    *,
    label: str,
    repo_db_dir: Path,
    bonded: str | None = None,  # 设置中间变量或可调参数，供后续工作流使用。
    polyelectrolyte_mode: bool = False,  # 设置中间变量或可调参数，供后续工作流使用。
):
    last_exc: Exception | None = None  # 设置中间变量或可调参数，供后续工作流使用。
    for db_dir, db_label in ((repo_db_dir, "repo"), (None, "default")):  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            mol = ff.mol_rdkit(  # 设置中间变量或可调参数，供后续工作流使用。
                smiles,
                name=label,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                db_dir=db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
                charge="RESP",  # 指定电荷来源或电荷计算方式。
                require_ready=True,  # 要求 MolDB 物种必须已准备好。
                prefer_db=True,  # 优先从 MolDB 读取已有结果。
                polyelectrolyte_mode=polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
                polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
            )
            mol = ff.ff_assign(mol, bonded=bonded, polyelectrolyte_mode=polyelectrolyte_mode, report=False)  # 分配力场参数并写入分子属性。
            if not mol:  # 根据当前状态决定是否进入该分支。
                raise RuntimeError(f"Cannot assign GAFF parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
            print(f"[MolDB] loaded {label} from {db_label} db with RESP charges")  # 打印关键路径或状态，便于人工检查。
            return mol  # 返回该辅助函数的结果。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            last_exc = exc  # 设置中间变量或可调参数，供后续工作流使用。
    raise RuntimeError(f"{label} must be RESP-ready in MolDB for this benchmark.") from last_exc  # 关键步骤失败时立即报错，避免继续生成错误结果。


def _load_ready_opls_species(  # 定义本例内部辅助函数，组织重复步骤。
    ff: OPLSAA,
    smiles: str,
    *,
    label: str,
    repo_db_dir: Path,
    bonded: str | None = None,  # 设置中间变量或可调参数，供后续工作流使用。
    polyelectrolyte_mode: bool = False,  # 设置中间变量或可调参数，供后续工作流使用。
):
    last_exc: Exception | None = None  # 设置中间变量或可调参数，供后续工作流使用。
    for db_dir, db_label in ((repo_db_dir, "repo"), (None, "default")):  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            mol = ff.mol_rdkit(  # 设置中间变量或可调参数，供后续工作流使用。
                smiles,
                name=label,  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
                db_dir=db_dir,  # 设置中间变量或可调参数，供后续工作流使用。
                charge="RESP",  # 指定电荷来源或电荷计算方式。
                require_ready=True,  # 要求 MolDB 物种必须已准备好。
                prefer_db=True,  # 优先从 MolDB 读取已有结果。
                polyelectrolyte_mode=polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
                polyelectrolyte_detection="auto",  # 设置中间变量或可调参数，供后续工作流使用。
            )
            mol = ff.ff_assign(  # 分配力场参数并写入分子属性。
                mol,
                charge=None,  # 指定电荷来源或电荷计算方式。
                bonded=bonded,  # 指定特殊 bonded 参数方案，例如 PF6 的 DRIH。
                polyelectrolyte_mode=polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
                report=False,  # 控制是否打印详细分配报告。
            )
            if not mol:  # 根据当前状态决定是否进入该分支。
                raise RuntimeError(f"Cannot assign OPLS-AA parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
            print(f"[MolDB] loaded {label} from {db_label} db with RESP charges and OPLS-AA atom types")  # 打印关键路径或状态，便于人工检查。
            return mol  # 返回该辅助函数的结果。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            last_exc = exc  # 设置中间变量或可调参数，供后续工作流使用。
    raise RuntimeError(f"{label} must be RESP-ready in MolDB for this OPLS-AA benchmark.") from last_exc  # 关键步骤失败时立即报错，避免继续生成错误结果。


def _assign_merz_ion(ff: MERZ, smiles: str, *, label: str):  # 定义本例内部辅助函数，组织重复步骤。
    mol = ff.mol(smiles)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    mol = ff.ff_assign(mol, report=False)  # 分配力场参数并写入分子属性。
    if not mol:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"Cannot assign MERZ parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    naming.ensure_name(mol, name=label, prefer_var=False)  # 设置中间变量或可调参数，供后续工作流使用。
    return mol  # 返回该辅助函数的结果。


def _assign_opls_ion(ff: OPLSAA, smiles: str, *, label: str):  # 定义本例内部辅助函数，组织重复步骤。
    mol = ff.mol(smiles, charge="opls", require_ready=False, prefer_db=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    mol = ff.ff_assign(mol, charge="opls", report=False)  # 分配力场参数并写入分子属性。
    if not mol:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"Cannot assign built-in OPLS-AA parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    naming.ensure_name(mol, name=label, prefer_var=False)  # 设置中间变量或可调参数，供后续工作流使用。
    return mol  # 返回该辅助函数的结果。


def _solvent_counts_for_swelling(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    solvent_mols: list[object],
    mass_ratio: list[float],
    dry_mass_amu: float,
    swelling_fraction: float,
) -> list[int]:
    target_mass = max(0.0, float(dry_mass_amu) * float(swelling_fraction))  # 设置中间变量或可调参数，供后续工作流使用。
    weights = [_mol_weight(mol) for mol in solvent_mols]  # 设置中间变量或可调参数，供后续工作流使用。
    ratio_sum = float(sum(float(x) for x in mass_ratio))  # 设置中间变量或可调参数，供后续工作流使用。
    counts: list[int] = []  # 设置各 species 的数量；顺序必须和 species 列表一致。
    for mw, ratio in zip(weights, mass_ratio):  # 遍历当前工作流中的一组对象或任务。
        target_i = target_mass * float(ratio) / ratio_sum  # 设置中间变量或可调参数，供后续工作流使用。
        counts.append(max(1, int(round(target_i / max(float(mw), 1.0e-12)))))
    return counts  # 返回该辅助函数的结果。


def _salt_pairs_for_1m(*, solvent_mols: list[object], solvent_counts: list[int], density_g_cm3: float, molarity: float) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    solvent_mass_amu = sum(_mol_weight(mol) * int(count) for mol, count in zip(solvent_mols, solvent_counts))  # 设置中间变量或可调参数，供后续工作流使用。
    # molecules = c(mol/L) * volume(L) * Avogadro; with mass in amu this reduces
    # to c * mass_amu / (density_g_cm3 * 1000).
    return max(1, int(round(float(molarity) * float(solvent_mass_amu) / (float(density_g_cm3) * 1000.0))))  # 返回该辅助函数的结果。


def _repair_polymer_net_charge(mol, *, target_q: float, audit_dir: Path, label: str) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    """Keep polymer partial-charge net aligned with its formal charge.

    Random-walk linking correctly carries linker-neighbor charges, but tiny
    per-repeat residuals can accumulate over long CMC chains.  The system
    counterion count is based on formal charge, so we make that invariant
    explicit before packing/export.
    """

    before = {  # 设置中间变量或可调参数，供后续工作流使用。
        "AtomicCharge": _net_charge(mol, "AtomicCharge"),
        "RESP": _net_charge(mol, "RESP"),
        "RESP2": _net_charge(mol, "RESP2"),
        "ESP": _net_charge(mol, "ESP"),
    }
    correction = correct_total_charge(  # 设置中间变量或可调参数，供后续工作流使用。
        mol,
        target_q=float(target_q),  # 设置中间变量或可调参数，供后续工作流使用。
        props=("AtomicCharge", "RESP", "RESP2", "ESP"),  # 设置中间变量或可调参数，供后续工作流使用。
        tol=1.0e-8,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        symmetrize_equivalent_charge_props(mol)
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        pass
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        annotate_polyelectrolyte_metadata(mol, detection="auto", resp_profile="adaptive")  # 设置中间变量或可调参数，供后续工作流使用。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        pass
    after = {  # 设置中间变量或可调参数，供后续工作流使用。
        "AtomicCharge": _net_charge(mol, "AtomicCharge"),
        "RESP": _net_charge(mol, "RESP"),
        "RESP2": _net_charge(mol, "RESP2"),
        "ESP": _net_charge(mol, "ESP"),
    }
    payload = {  # 设置中间变量或可调参数，供后续工作流使用。
        "label": str(label),
        "target_q": float(target_q),
        "before": before,
        "after": after,
        "correction": correction,
    }
    audit_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    (audit_dir / f"{label}_charge_repair.json").write_text(  # 开始一个多行函数调用或配置块。
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",  # 设置中间变量或可调参数，供后续工作流使用。
        encoding="utf-8",  # 设置中间变量或可调参数，供后续工作流使用。
    )
    return payload  # 返回该辅助函数的结果。


def _composition_for_atom_cap(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    cmc,
    na,
    solvent_mols: list[object],
    chain_count: int,
    q_poly: int,
    swelling_fraction: float,
    solvent_mass_ratio: list[float],
    solvent_density_g_cm3: float,
    salt_molarity: float,
    atom_cap: int,
) -> dict[str, Any]:
    n_chain = max(1, int(chain_count))  # 设置中间变量或可调参数，供后续工作流使用。
    while True:  # 循环执行直到当前条件不再满足。
        n_na = int(abs(int(q_poly)) * n_chain)  # 设置中间变量或可调参数，供后续工作流使用。
        dry_mass = _mol_weight(cmc) * n_chain + _mol_weight(na) * n_na  # 设置中间变量或可调参数，供后续工作流使用。
        solvent_counts = _solvent_counts_for_swelling(  # 设置各溶剂分子数量；顺序必须和 species 保持一致。
            solvent_mols=solvent_mols,  # 设置中间变量或可调参数，供后续工作流使用。
            mass_ratio=solvent_mass_ratio,  # 设置中间变量或可调参数，供后续工作流使用。
            dry_mass_amu=dry_mass,  # 设置中间变量或可调参数，供后续工作流使用。
            swelling_fraction=swelling_fraction,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        n_salt = _salt_pairs_for_1m(  # 设置中间变量或可调参数，供后续工作流使用。
            solvent_mols=solvent_mols,  # 设置中间变量或可调参数，供后续工作流使用。
            solvent_counts=solvent_counts,  # 设置各溶剂分子数量；顺序必须和 species 保持一致。
            density_g_cm3=solvent_density_g_cm3,  # 设置初始 packing 密度，主要影响构建难度和初始盒子大小。
            molarity=salt_molarity,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        counts = [n_chain, *solvent_counts, n_salt, n_salt, n_na]  # 设置各 species 的数量；顺序必须和 species 列表一致。
        atom_count = (  # 设置中间变量或可调参数，供后续工作流使用。
            cmc.GetNumAtoms() * counts[0]
            + sum(mol.GetNumAtoms() * count for mol, count in zip(solvent_mols, solvent_counts))
            + LI_NATOMS * n_salt
            + PF6_NATOMS * n_salt
            + na.GetNumAtoms() * n_na
        )
        if atom_count <= int(atom_cap) or n_chain <= 1:  # 根据当前状态决定是否进入该分支。
            return {  # 返回该辅助函数的结果。
                "counts": counts,
                "estimated_atom_count": int(atom_count),
                "chain_count": int(n_chain),
                "dry_mass_amu": float(dry_mass),
                "solvent_counts": solvent_counts,
                "salt_pairs": int(n_salt),
            }
        n_chain -= 1  # 设置中间变量或可调参数，供后续工作流使用。


def _metric_row(msd: dict[str, Any], moltype: str, metric_name: str, *, role: str) -> dict[str, Any]:  # 定义本例内部辅助函数，组织重复步骤。
    species = dict(msd.get(moltype) or {})  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。
    metric = dict((species.get("metrics") or {}).get(metric_name) or {})  # 设置中间变量或可调参数，供后续工作流使用。
    value = metric.get("D_m2_s")  # 设置中间变量或可调参数，供后续工作流使用。
    apparent = metric.get("apparent_D_m2_s")  # 设置中间变量或可调参数，供后续工作流使用。
    return {  # 返回该辅助函数的结果。
        "role": role,
        "moltype": moltype,
        "metric": metric_name,
        "D_m2_s": value,
        "apparent_D_m2_s": apparent,
        "alpha_mean": metric.get("alpha_mean"),
        "alpha_std": metric.get("alpha_std"),
        "confidence": metric.get("confidence"),
        "status": metric.get("status"),
        "warning": metric.get("warning"),
        "fit_t_start_ps": metric.get("fit_t_start_ps"),
        "fit_t_end_ps": metric.get("fit_t_end_ps"),
        "fit_n_points": metric.get("fit_n_points"),
        "min_fit_points": metric.get("min_fit_points"),
        "min_fit_duration_ps": metric.get("min_fit_duration_ps"),
        "n_groups": metric.get("n_groups"),
    }


def _write_transport_table(analysis_dir: Path, msd: dict[str, Any], rdf_li: dict[str, Any] | None, rdf_na: dict[str, Any] | None) -> Path:  # 定义本例内部辅助函数，组织重复步骤。
    # Keep whole-chain polymer self-diffusion separate from local segment or
    # charged-group mobility.  This mirrors the usual polymer-diffusion
    # convention and avoids reporting monomer motion as chain transport.
    rows = [  # 设置中间变量或可调参数，供后续工作流使用。
        _metric_row(msd, "Li", "ion_atomic_msd", role="ion_atomic_diffusion"),  # 设置中间变量或可调参数，供后续工作流使用。
        _metric_row(msd, "Na", "ion_atomic_msd", role="counterion_atomic_diffusion"),  # 设置中间变量或可调参数，供后续工作流使用。
        _metric_row(msd, "PF6", "molecule_com_msd", role="anion_com_diffusion"),  # 设置中间变量或可调参数，供后续工作流使用。
        _metric_row(msd, "EC", "molecule_com_msd", role="solvent_com_diffusion"),  # 设置中间变量或可调参数，供后续工作流使用。
        _metric_row(msd, "EMC", "molecule_com_msd", role="solvent_com_diffusion"),  # 设置中间变量或可调参数，供后续工作流使用。
        _metric_row(msd, "DEC", "molecule_com_msd", role="solvent_com_diffusion"),  # 设置中间变量或可调参数，供后续工作流使用。
        _metric_row(msd, "CMC", "chain_com_msd", role="polymer_chain_com_self_diffusion"),  # 设置中间变量或可调参数，供后续工作流使用。
        _metric_row(msd, "CMC", "residue_com_msd", role="polymer_segment_mobility_diagnostic"),  # 设置中间变量或可调参数，供后续工作流使用。
        _metric_row(msd, "CMC", "charged_group_com_msd", role="polymer_carboxylate_group_mobility"),  # 设置中间变量或可调参数，供后续工作流使用。
    ]
    out_csv = analysis_dir / "cmc_bulk_transport_table.csv"  # 设置中间变量或可调参数，供后续工作流使用。
    with out_csv.open("w", newline="", encoding="utf-8") as handle:  # 用上下文管理器安全打开文件或管理资源。
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))  # 设置中间变量或可调参数，供后续工作流使用。
        writer.writeheader()
        writer.writerows(rows)

    payload = {  # 设置中间变量或可调参数，供后续工作流使用。
        "transport_rows": rows,
        "rdf_li_first_shell": rdf_li or {},
        "rdf_na_first_shell": rdf_na or {},
        "notes": [
            "CMC polymer self-diffusion is reported from each independent chain COM (chain_com_msd).",
            "Residue and charged-group MSD rows are local mobility diagnostics, not whole-chain self-diffusion coefficients.",
            "Report D_m2_s only when alpha_mean is close to 1 and confidence/status are acceptable; otherwise prefer apparent_D_m2_s as a mobility index.",
        ],
    }
    (analysis_dir / "cmc_bulk_transport_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
    return out_csv  # 返回该辅助函数的结果。


# ---------------- user inputs ----------------
restart_status = _env_flag("YADONPY_RESTART", default=True)  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
build_only = _env_flag("YADONPY_BUILD_ONLY", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
export_only = _env_flag("YADONPY_EXPORT_ONLY", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
analysis_only = _env_flag("YADONPY_ANALYSIS_ONLY", default=False)  # 只做后处理，不重新运行采样。
smoke_mode = _env_flag("YADONPY_SMOKE", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
skip_rdf = _env_flag("YADONPY_SKIP_RDF", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
skip_sigma = _env_flag("YADONPY_SKIP_SIGMA", default=True)  # 设置中间变量或可调参数，供后续工作流使用。
skip_den_dis = _env_flag("YADONPY_SKIP_DEN_DIS", default=True)  # 设置中间变量或可调参数，供后续工作流使用。

forcefield_name = _normalize_forcefield(_env_text("YADONPY_FORCEFIELD", "gaff2"))  # 设置中间变量或可调参数，供后续工作流使用。
oplsaa_profile = _env_text("YADONPY_OPLSAA_PROFILE", "strict")  # 设置中间变量或可调参数，供后续工作流使用。
random_seed = _env_int("YADONPY_RANDOM_SEED", 20260507)  # 设置中间变量或可调参数，供后续工作流使用。
chain_len = _env_int("YADONPY_CMC_DP", 12 if smoke_mode else 50)  # 设置中间变量或可调参数，供后续工作流使用。
chain_count = _env_int("YADONPY_CMC_CHAINS", 2 if smoke_mode else 20)  # 设置中间变量或可调参数，供后续工作流使用。
atom_cap = _env_int("YADONPY_ATOM_CAP", 12000 if smoke_mode else 40000)  # 设置中间变量或可调参数，供后续工作流使用。
swelling_fraction = _env_float("YADONPY_SWELLING_FRACTION", 0.12)  # 设置中间变量或可调参数，供后续工作流使用。
salt_molarity = _env_float("YADONPY_SALT_MOLARITY", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
solvent_density_g_cm3 = _env_float("YADONPY_SOLVENT_DENSITY_G_CM3", 1.15)  # 设置中间变量或可调参数，供后续工作流使用。
charge_scale_value = _env_float("YADONPY_CHARGE_SCALE", 0.7)  # 设置中间变量或可调参数，供后续工作流使用。

temp = _env_float("YADONPY_TEMP_K", 318.15)  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
press = _env_float("YADONPY_PRESS_BAR", 1.0)  # 设置压力 bar；用于 NPT/EQ 阶段。
prod_ns = _env_float("YADONPY_PROD_NS", 100.0)  # 设置中间变量或可调参数，供后续工作流使用。
# OPLS-AA for CMC/electrolyte is still under active validation.  The default is
# intentionally conservative here; users can opt into 2 fs once their short
# preflight confirms the specific parameter set is stable.
prod_dt_default = 0.001 if forcefield_name == "oplsaa" else 0.002
prod_lincs_iter_default = 4 if forcefield_name == "oplsaa" else 2
prod_lincs_order_default = 12 if forcefield_name == "oplsaa" else 8
prod_dt_ps = _env_float("YADONPY_PROD_DT_PS", prod_dt_default)  # 设置中间变量或可调参数，供后续工作流使用。
prod_constraints = _env_text("YADONPY_PROD_CONSTRAINTS", "h-bonds")  # 设置中间变量或可调参数，供后续工作流使用。
prod_lincs_iter = _env_int("YADONPY_PROD_LINCS_ITER", prod_lincs_iter_default)  # 设置中间变量或可调参数，供后续工作流使用。
prod_lincs_order = _env_int("YADONPY_PROD_LINCS_ORDER", prod_lincs_order_default)  # 设置中间变量或可调参数，供后续工作流使用。
prod_ensemble = _env_text("YADONPY_PROD_ENSEMBLE", "npt").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
prod_bridge_ps = _env_float("YADONPY_PROD_BRIDGE_PS", 100.0)  # 设置中间变量或可调参数，供后续工作流使用。
gpu_offload_default = "conservative" if forcefield_name == "oplsaa" else "full"
gpu_offload_mode = _env_text("YADONPY_GPU_OFFLOAD_MODE", gpu_offload_default)  # 设置中间变量或可调参数，供后续工作流使用。
eq_gpu_offload_mode = _env_text("YADONPY_EQ_GPU_OFFLOAD_MODE", gpu_offload_mode)  # 设置中间变量或可调参数，供后续工作流使用。
performance_profile = _env_text("PERFORMANCE_PROFILE", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
analysis_profile = _env_text("ANALYSIS_PROFILE", "auto")  # 选择后处理预设；interface_fast 面向 slab/interface。
trajectory_format = _env_text("TRAJECTORY_FORMAT", os.environ.get("YADONPY_TRAJECTORY_FORMAT", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。

eq21_final_ns = _env_float("YADONPY_EQ21_FINAL_NS", 0.8)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_npt_time_scale = _env_float("YADONPY_EQ21_NPT_TIME_SCALE", 2.0)  # 设置中间变量或可调参数，供后续工作流使用。
additional_ns = _env_float("YADONPY_ADDITIONAL_NS", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
additional_rounds = _env_int("YADONPY_ADDITIONAL_MAX_ROUNDS", 4)  # 设置中间变量或可调参数，供后续工作流使用。
allow_unconverged = _env_flag("ALLOW_UNCONVERGED_PRODUCTION", default=False)  # 设置中间变量或可调参数，供后续工作流使用。

mpi = _env_int("YADONPY_MPI", 1)  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = _env_int("YADONPY_OMP", 14)  # 设置每个 rank 的 OpenMP 线程数。
gpu = _env_int("YADONPY_GPU", 1)  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = _env_int("YADONPY_GPU_ID", 0)  # 选择 GPU 设备编号，多卡节点可修改。

msd_begin_ps = _env_optional_float("YADONPY_MSD_BEGIN_PS")  # 设置中间变量或可调参数，供后续工作流使用。
msd_end_ps = _env_optional_float("YADONPY_MSD_END_PS")  # 设置中间变量或可调参数，供后续工作流使用。
msd_drift = _env_text("YADONPY_MSD_DRIFT", "off")  # 设置中间变量或可调参数，供后续工作流使用。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"  # 设置中间变量或可调参数，供后续工作流使用。
WORK_DIR_OVERRIDE = str(os.environ.get("YADONPY_WORK_DIR", "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
work_dir = (  # 设置本例输出目录。
    Path(WORK_DIR_OVERRIDE).expanduser()
    if WORK_DIR_OVERRIDE  # 根据当前状态决定是否进入该分支。
    else BASE_DIR / f"work_dir_cmcna_carbonate_lipf6_{forcefield_name}"
)

glucose_0_smiles = "*OC1OC(CO)C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。
feed_ratio = [12, 26, 27, 35]  # 设置中间变量或可调参数，供后续工作流使用。
feed_prob = poly.ratio_to_prob(feed_ratio)  # 设置中间变量或可调参数，供后续工作流使用。
ter_smiles = "[H][*]"  # 设置中间变量或可调参数，供后续工作流使用。

EC_SMILES = "O=C1OCCO1"  # 设置中间变量或可调参数，供后续工作流使用。
EMC_SMILES = "CCOC(=O)OC"  # 设置中间变量或可调参数，供后续工作流使用。
DEC_SMILES = "CCOC(=O)OCC"  # 设置中间变量或可调参数，供后续工作流使用。
LI_SMILES = "[Li+]"  # 设置中间变量或可调参数，供后续工作流使用。
NA_SMILES = "[Na+]"  # 设置中间变量或可调参数，供后续工作流使用。
PF6_SMILES = "F[P-](F)(F)(F)(F)F"  # 设置中间变量或可调参数，供后续工作流使用。
LI_NATOMS = 1  # 设置中间变量或可调参数，供后续工作流使用。
PF6_NATOMS = 7  # 设置中间变量或可调参数，供后续工作流使用。


def main() -> int:  # 定义本例内部辅助函数，组织重复步骤。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。
    set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。
    np.random.seed(int(random_seed))

    wd = workdir(work_dir, restart=restart_status)  # 创建或复用本例工作目录。
    analysis_dir = Path(wd) / "06_analysis"  # 设置中间变量或可调参数，供后续工作流使用。
    audit_dir = Path(wd) / "07_polymer_audit"  # 设置中间变量或可调参数，供后续工作流使用。
    cmc_rw_dir = wd.child("CMC_rw")  # 设置中间变量或可调参数，供后续工作流使用。
    cmc_term_dir = wd.child("CMC_term")  # 设置中间变量或可调参数，供后续工作流使用。
    ac_build_dir = wd.child("00_build_cell")  # 设置中间变量或可调参数，供后续工作流使用。

    if forcefield_name == "oplsaa":  # 根据当前状态决定是否进入该分支。
        ff = OPLSAA(profile=oplsaa_profile)  # 选择有机分子/聚合物/部分无机离子的力场对象。
        ion_ff = OPLSAA(profile=oplsaa_profile)  # 选择单原子离子参数来源。
        load_species = lambda smiles, label, **kw: _load_ready_opls_species(  # noqa: E731
            ff, smiles, label=label, repo_db_dir=REPO_DB_DIR, **kw  # 设置中间变量或可调参数，供后续工作流使用。
        )
        Li = _assign_opls_ion(ion_ff, LI_SMILES, label="Li")  # 设置中间变量或可调参数，供后续工作流使用。
        Na = _assign_opls_ion(ion_ff, NA_SMILES, label="Na")  # 设置中间变量或可调参数，供后续工作流使用。
        PF6 = load_species(PF6_SMILES, "PF6", bonded="DRIH")  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        ff = GAFF2_mod() if forcefield_name == "gaff2_mod" else GAFF2()  # 选择有机分子/聚合物/部分无机离子的力场对象。
        ion_ff = MERZ()  # 选择单原子离子参数来源。
        load_species = lambda smiles, label, **kw: _load_ready_gaff_species(  # noqa: E731
            ff, smiles, label=label, repo_db_dir=REPO_DB_DIR, **kw  # 设置中间变量或可调参数，供后续工作流使用。
        )
        Li = _assign_merz_ion(ion_ff, LI_SMILES, label="Li")  # 设置中间变量或可调参数，供后续工作流使用。
        Na = _assign_merz_ion(ion_ff, NA_SMILES, label="Na")  # 设置中间变量或可调参数，供后续工作流使用。
        PF6 = load_species(PF6_SMILES, "PF6", bonded="DRIH")  # 设置中间变量或可调参数，供后续工作流使用。

    glucose_0 = load_species(glucose_0_smiles, "glucose_0")  # 设置中间变量或可调参数，供后续工作流使用。
    glucose_2 = load_species(glucose_2_smiles, "glucose_2", polyelectrolyte_mode=True)  # 设置中间变量或可调参数，供后续工作流使用。
    glucose_3 = load_species(glucose_3_smiles, "glucose_3", polyelectrolyte_mode=True)  # 设置中间变量或可调参数，供后续工作流使用。
    glucose_6 = load_species(glucose_6_smiles, "glucose_6", polyelectrolyte_mode=True)  # 设置中间变量或可调参数，供后续工作流使用。

    ter = _zero_charge_terminator(ter_smiles)  # 设置中间变量或可调参数，供后续工作流使用。
    CMC = poly.random_copolymerize_rw(  # 用随机游走生成聚合物链。
        [glucose_0, glucose_2, glucose_3, glucose_6],
        chain_len,
        ratio=feed_prob,  # 设置共聚组成比例。
        tacticity="atactic",  # 设置聚合物立构。
        name="CMC",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=cmc_rw_dir,  # 设置本例输出目录。
    )
    write_polymer_audit(audit_polymer_state(CMC, label="cmc_random_walk", radius=2), audit_dir / "cmc_random_walk.json")  # 设置中间变量或可调参数，供后续工作流使用。
    CMC = poly.terminate_rw(CMC, ter, name="CMC", work_dir=cmc_term_dir)  # 给聚合物链加端基。
    CMC = ff.ff_assign(CMC, charge=None, polyelectrolyte_mode=True, report=False)  # 分配力场参数并写入分子属性。
    if not CMC:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Cannot assign force field parameters for CMC.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    naming.ensure_name(CMC, name="CMC", prefer_var=False)  # 设置中间变量或可调参数，供后续工作流使用。
    q_poly = _formal_charge(CMC)  # 设置中间变量或可调参数，供后续工作流使用。
    charge_repair = _repair_polymer_net_charge(CMC, target_q=float(q_poly), audit_dir=audit_dir, label="cmc_final_assigned")  # 设置中间变量或可调参数，供后续工作流使用。
    write_polymer_audit(audit_polymer_state(CMC, label="cmc_final_assigned", radius=2), audit_dir / "cmc_final_assigned.json")  # 设置中间变量或可调参数，供后续工作流使用。
    if forcefield_name == "oplsaa":  # 根据当前状态决定是否进入该分支。
        (audit_dir / "cmc_oplsaa_assignment_audit.json").write_text(  # 开始一个多行函数调用或配置块。
            json.dumps(audit_oplsaa_assignment(CMC, strict=True), indent=2, ensure_ascii=False) + "\n",  # 设置中间变量或可调参数，供后续工作流使用。
            encoding="utf-8",  # 设置中间变量或可调参数，供后续工作流使用。
        )

    EC = load_species(EC_SMILES, "EC")  # 设置中间变量或可调参数，供后续工作流使用。
    EMC = load_species(EMC_SMILES, "EMC")  # 设置中间变量或可调参数，供后续工作流使用。
    DEC = load_species(DEC_SMILES, "DEC")  # 设置中间变量或可调参数，供后续工作流使用。
    solvent_mols = [EC, EMC, DEC]  # 设置中间变量或可调参数，供后续工作流使用。

    composition = _composition_for_atom_cap(  # 设置中间变量或可调参数，供后续工作流使用。
        cmc=CMC,  # 设置中间变量或可调参数，供后续工作流使用。
        na=Na,  # 设置中间变量或可调参数，供后续工作流使用。
        solvent_mols=solvent_mols,  # 设置中间变量或可调参数，供后续工作流使用。
        chain_count=chain_count,  # 设置中间变量或可调参数，供后续工作流使用。
        q_poly=q_poly,  # 设置中间变量或可调参数，供后续工作流使用。
        swelling_fraction=swelling_fraction,  # 设置中间变量或可调参数，供后续工作流使用。
        solvent_mass_ratio=[3.0, 2.0, 5.0],  # 设置中间变量或可调参数，供后续工作流使用。
        solvent_density_g_cm3=solvent_density_g_cm3,  # 设置中间变量或可调参数，供后续工作流使用。
        salt_molarity=salt_molarity,  # 设置中间变量或可调参数，供后续工作流使用。
        atom_cap=atom_cap,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    counts = [int(x) for x in composition["counts"]]  # 设置各 species 的数量；顺序必须和 species 列表一致。
    species = [CMC, EC, EMC, DEC, Li, PF6, Na]  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。
    charge_scale = [float(charge_scale_value)] * len(species)  # 设置电荷缩放系数；1.0 表示全电荷模型。
    formulation = {  # 设置中间变量或可调参数，供后续工作流使用。
        "forcefield": forcefield_name,
        "oplsaa_profile": oplsaa_profile if forcefield_name == "oplsaa" else None,
        "random_seed": int(random_seed),
        "temperature_K": float(temp),
        "pressure_bar": float(press),
        "cmc_dp": int(chain_len),
        "cmc_chains_requested": int(chain_count),
        "cmc_chains_used": int(composition["chain_count"]),
        "cmc_formal_charge_per_chain": int(q_poly),
        "cmc_partial_charge_per_chain": float(charge_repair["after"].get("AtomicCharge", 0.0)),
        "cmc_charge_repair": charge_repair,
        "swelling_fraction": float(swelling_fraction),
        "salt_molarity_nominal": float(salt_molarity),
        "solvent_mass_ratio_EC_EMC_DEC": [3.0, 2.0, 5.0],
        "counts_CMC_EC_EMC_DEC_Li_PF6_Na": counts,
        "charge_scale": charge_scale,
        "production_dt_ps": float(prod_dt_ps),
        "production_constraints": str(prod_constraints),
        "production_lincs_iter": int(prod_lincs_iter),
        "production_lincs_order": int(prod_lincs_order),
        "production_gpu_offload_mode": str(gpu_offload_mode),
        "eq_gpu_offload_mode": str(eq_gpu_offload_mode),
        "production_stability_note": (
            "OPLS-AA defaults to 1 fs, LINCS 4/12, and conservative GPU offload for this CMC/electrolyte benchmark "
            "because remote diagnostics showed balanced/full GPU offload can trigger CUDA illegal-address failures "
            "for current refine-profile CMC assignments."
            if forcefield_name == "oplsaa"  # 根据当前状态决定是否进入该分支。
            else "GAFF-family default production uses h-bonds with 2 fs."
        ),
        "estimated_atom_count": int(composition["estimated_atom_count"]),
        "dry_mass_amu": float(composition["dry_mass_amu"]),
        "solvent_counts": composition["solvent_counts"],
        "salt_pairs": int(composition["salt_pairs"]),
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)  # 设置中间变量或可调参数，供后续工作流使用。
    (analysis_dir / "formulation.json").write_text(json.dumps(formulation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
    print("[FORMULATION] " + json.dumps(formulation, ensure_ascii=False))  # 打印关键路径或状态，便于人工检查。
    msd_mols = ["CMC"] + [mol for mol, count in zip(species[1:], counts[1:]) if int(count) > 0]  # 设置中间变量或可调参数，供后续工作流使用。

    if analysis_only:  # 根据当前状态决定是否进入该分支。
        analy = AnalyzeResult.from_work_dir(wd)  # 设置中间变量或可调参数，供后续工作流使用。
        _ = analy.get_all_prop(temp=temp, press=press, save=True, analysis_profile=analysis_profile)  # 设置中间变量或可调参数，供后续工作流使用。
        rdf_li = None if skip_rdf else analy.rdf(  # 设置中间变量或可调参数，供后续工作流使用。
            center_mol=Li,  # 设置中间变量或可调参数，供后续工作流使用。
            site_filter=["carboxylate_oxygen", "hydroxyl_oxygen", "carbonyl_oxygen", "ether_oxygen", "coordination_fluorine", "fluorine_site"],  # 设置中间变量或可调参数，供后续工作流使用。
            analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
            r_max_nm=1.5,  # 设置中间变量或可调参数，供后续工作流使用。
            resume=True,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        rdf_na = None if skip_rdf else analy.rdf(  # 设置中间变量或可调参数，供后续工作流使用。
            center_mol=Na,  # 设置中间变量或可调参数，供后续工作流使用。
            site_filter=["carboxylate_oxygen", "hydroxyl_oxygen", "carbonyl_oxygen", "ether_oxygen", "coordination_fluorine", "fluorine_site"],  # 设置中间变量或可调参数，供后续工作流使用。
            analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
            r_max_nm=1.5,  # 设置中间变量或可调参数，供后续工作流使用。
            resume=True,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        msd = analy.msd(  # 设置中间变量或可调参数，供后续工作流使用。
            mols=msd_mols,  # 设置中间变量或可调参数，供后续工作流使用。
            geometry="3d",  # 设置中间变量或可调参数，供后续工作流使用。
            unwrap="on",  # 设置中间变量或可调参数，供后续工作流使用。
            drift=msd_drift,  # 设置中间变量或可调参数，供后续工作流使用。
            begin_ps=msd_begin_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            end_ps=msd_end_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
            resume=True,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        table = _write_transport_table(analysis_dir, msd, rdf_li, rdf_na)  # 设置中间变量或可调参数，供后续工作流使用。
        print(f"[ANALYSIS-ONLY] transport table: {table}")  # 打印关键路径或状态，便于人工检查。
        return 0  # 返回该辅助函数的结果。

    active = [(mol, count, scale) for mol, count, scale in zip(species, counts, charge_scale) if int(count) > 0]  # 设置中间变量或可调参数，供后续工作流使用。
    ac = poly.amorphous_cell(  # 构建无定形混合体系初始盒子。
        [mol for mol, _count, _scale in active],
        [int(count) for _mol, count, _scale in active],
        charge_scale=[float(scale) for _mol, _count, scale in active],  # 设置电荷缩放系数；1.0 表示全电荷模型。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        density=0.05,  # 设置中间变量或可调参数，供后续工作流使用。
        neutralize=False,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=ac_build_dir,  # 设置本例输出目录。
    )
    if build_only:  # 根据当前状态决定是否进入该分支。
        print(f"[BUILD-ONLY] built initial cell at {ac_build_dir}")  # 打印关键路径或状态，便于人工检查。
        return 0  # 返回该辅助函数的结果。

    eqmd = eq.EQ21step(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
    if export_only:  # 根据当前状态决定是否进入该分支。
        exported = eqmd.ensure_system_exported()  # 设置中间变量或可调参数，供后续工作流使用。
        write_polymer_audit(  # 开始一个多行函数调用或配置块。
            compare_exported_charge_groups(system_dir=exported.system_top.parent, moltype="CMC", mol=CMC),  # 设置中间变量或可调参数，供后续工作流使用。
            audit_dir / "cmc_export_charge_groups.json",
        )
        print(f"[EXPORT-ONLY] exported {exported.system_top.parent}")  # 打印关键路径或状态，便于人工检查。
        return 0  # 返回该辅助函数的结果。

    ac = eqmd.exec(  # 设置中间变量或可调参数，供后续工作流使用。
        temp=temp,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
        press=press,  # 设置压力 bar；用于 NPT/EQ 阶段。
        mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
        gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
        gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
        time=eq21_final_ns,  # 设置中间变量或可调参数，供后续工作流使用。
        eq21_npt_time_scale=eq21_npt_time_scale,  # 设置中间变量或可调参数，供后续工作流使用。
        gpu_offload_mode=eq_gpu_offload_mode,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    exported = eqmd.ensure_system_exported()  # 设置中间变量或可调参数，供后续工作流使用。
    write_polymer_audit(  # 开始一个多行函数调用或配置块。
        compare_exported_charge_groups(system_dir=exported.system_top.parent, moltype="CMC", mol=CMC),  # 设置中间变量或可调参数，供后续工作流使用。
        audit_dir / "cmc_export_charge_groups.json",
    )

    analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    eq_ok = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。
    for round_idx in range(int(additional_rounds)):  # 遍历当前工作流中的一组对象或任务。
        if eq_ok:  # 根据当前状态决定是否进入该分支。
            break
        add = eq.Additional(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = add.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=additional_ns)  # 设置中间变量或可调参数，供后续工作流使用。
        analy = add.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        eq_ok = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。
        print(f"[ADDITIONAL] round={round_idx + 1} equilibrium_ok={eq_ok}")  # 打印关键路径或状态，便于人工检查。
    if not eq_ok and not allow_unconverged:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Equilibration did not converge; set ALLOW_UNCONVERGED_PRODUCTION=1 for diagnostic production.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    prod_cls = eq.NVT if prod_ensemble == "nvt" else eq.NPT
    prod = prod_cls(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
    prod_kwargs: dict[str, Any] = {  # 设置中间变量或可调参数，供后续工作流使用。
        "temp": temp,
        "mpi": mpi,
        "omp": omp,
        "gpu": gpu,
        "gpu_id": gpu_id,
        "time": prod_ns,
        "dt_ps": prod_dt_ps,
        "constraints": prod_constraints,
        "lincs_iter": prod_lincs_iter,
        "lincs_order": prod_lincs_order,
        "bridge_ps": prod_bridge_ps,
        "gpu_offload_mode": gpu_offload_mode,
        "performance_profile": performance_profile,
        "analysis_profile": analysis_profile,
        "trajectory_format": trajectory_format,
    }
    if prod_ensemble != "nvt":  # 根据当前状态决定是否进入该分支。
        prod_kwargs["press"] = press  # 设置中间变量或可调参数，供后续工作流使用。
    ac = prod.exec(**prod_kwargs)  # 设置中间变量或可调参数，供后续工作流使用。

    analy = prod.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.get_all_prop(temp=temp, press=press, save=True, analysis_profile=analysis_profile)  # 设置中间变量或可调参数，供后续工作流使用。
    rdf_li = None if skip_rdf else analy.rdf(  # 设置中间变量或可调参数，供后续工作流使用。
        center_mol=Li,  # 设置中间变量或可调参数，供后续工作流使用。
        site_filter=["carboxylate_oxygen", "hydroxyl_oxygen", "carbonyl_oxygen", "ether_oxygen", "coordination_fluorine", "fluorine_site"],  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
        r_max_nm=1.5,  # 设置中间变量或可调参数，供后续工作流使用。
        resume=True,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    rdf_na = None if skip_rdf else analy.rdf(  # 设置中间变量或可调参数，供后续工作流使用。
        center_mol=Na,  # 设置中间变量或可调参数，供后续工作流使用。
        site_filter=["carboxylate_oxygen", "hydroxyl_oxygen", "carbonyl_oxygen", "ether_oxygen", "coordination_fluorine", "fluorine_site"],  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
        r_max_nm=1.5,  # 设置中间变量或可调参数，供后续工作流使用。
        resume=True,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    msd = analy.msd(  # 设置中间变量或可调参数，供后续工作流使用。
        mols=msd_mols,  # 设置中间变量或可调参数，供后续工作流使用。
        geometry="3d",  # 设置中间变量或可调参数，供后续工作流使用。
        unwrap="on",  # 设置中间变量或可调参数，供后续工作流使用。
        drift=msd_drift,  # 设置中间变量或可调参数，供后续工作流使用。
        begin_ps=msd_begin_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        end_ps=msd_end_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
        resume=True,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    if not skip_sigma:  # 根据当前状态决定是否进入该分支。
        _ = analy.sigma(temp_k=temp, msd=msd, drift=msd_drift)  # 设置中间变量或可调参数，供后续工作流使用。
    if not skip_den_dis:  # 根据当前状态决定是否进入该分支。
        _ = analy.den_dis()  # 设置中间变量或可调参数，供后续工作流使用。
    table = _write_transport_table(analysis_dir, msd, rdf_li, rdf_na)  # 设置中间变量或可调参数，供后续工作流使用。
    print(f"[TRANSPORT] table: {table}")  # 打印关键路径或状态，便于人工检查。
    return 0  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    raise SystemExit(main())  # 关键步骤失败时立即报错，避免继续生成错误结果。
