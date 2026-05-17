from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""CMC random copolymer electrolyte example with a MoldDB-backed additive.

The anionic glucose monomers are expected to be RESP-ready in MolDB with
``polyelectrolyte_mode=True``. This keeps the expensive monomer QM step out of
the system-build script and directly exercises the MolDB -> ITP export path.

Use ``YADONPY_BUILD_ONLY=1`` to stop after amorphous-cell construction.
Use ``YADONPY_EXPORT_ONLY=1`` to stop after exporting ``02_system``.
These modes are useful for checking topology / ITP generation on machines
without GROMACS.

Remote mixed-system debug ladder:
- set a unique ``YADONPY_WORK_DIR`` for every run
- start with ``YADONPY_EXPORT_ONLY=1``
- then ``YADONPY_EQ21_STAGE_CAP=2`` for ``EM + preNVT``
- then a short production run on GPU
- repeat the short production run with ``YADONPY_GPU=0`` if the failure needs
  to be classified as topology/physics vs runtime/environment
"""

import json  # 导入本例需要的库或 yadonpy 接口。
import os  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import utils, poly, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.polymer_audit import audit_polymer_state, compare_exported_charge_groups, write_polymer_audit  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff import GAFF2, GAFF2_mod, MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.analyzer import AnalyzeResult  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim import qm  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.performance import resolve_io_analysis_policy  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.preset import eq  # 导入本例需要的库或 yadonpy 接口。


def _env_flag(name: str, default: bool = False) -> bool:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    token = str(os.environ.get(name, "")).strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
    if not token:  # 根据当前状态决定是否进入该分支。
        return bool(default)  # 返回该辅助函数的结果。
    return token in {"1", "true", "t", "yes", "y", "on"}  # 返回该辅助函数的结果。


def _env_int(name: str, default: int) -> int:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    if not raw:  # 根据当前状态决定是否进入该分支。
        return int(default)  # 返回该辅助函数的结果。
    return int(raw)  # 返回该辅助函数的结果。


def _env_float(name: str, default: float) -> float:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    if not raw:  # 根据当前状态决定是否进入该分支。
        return float(default)  # 返回该辅助函数的结果。
    return float(raw)  # 返回该辅助函数的结果。


def _env_optional_float(name: str) -> float | None:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    if not raw:  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    return float(raw)  # 返回该辅助函数的结果。


def _env_int_list(name: str, expected_len: int) -> list[int] | None:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    if not raw:  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    vals = [int(tok.strip()) for tok in raw.split(",") if str(tok).strip()]  # 设置中间变量或可调参数，供后续工作流使用。
    if len(vals) != int(expected_len):  # 根据当前状态决定是否进入该分支。
        raise ValueError(f"{name} expects {expected_len} comma-separated integers, got {len(vals)}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return vals  # 返回该辅助函数的结果。


def _env_text(name: str, default: str) -> str:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    if not raw:  # 根据当前状态决定是否进入该分支。
        return str(default)  # 返回该辅助函数的结果。
    return raw  # 返回该辅助函数的结果。


def _load_ready_from_moldb(  # 定义本例内部辅助函数，组织重复步骤。
    ff,
    smiles: str,
    *,
    label: str,
    bonded: str | None = None,  # 设置中间变量或可调参数，供后续工作流使用。
    polyelectrolyte_mode: bool = False,  # 设置中间变量或可调参数，供后续工作流使用。
    repo_db_dir: Path,
):
    last_exc: Exception | None = None  # 设置中间变量或可调参数，供后续工作流使用。
    for db_dir, db_label in ((None, "default"), (repo_db_dir, "repo")):  # 遍历当前工作流中的一组对象或任务。
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
                bonded=bonded,  # 指定特殊 bonded 参数方案，例如 PF6 的 DRIH。
                polyelectrolyte_mode=polyelectrolyte_mode,  # 启用聚电解质处理逻辑。
            )
            if not mol:  # 根据当前状态决定是否进入该分支。
                raise RuntimeError(f"Cannot assign force field parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
            print(f"[MolDB] loaded {label} from {db_label} db")  # 打印关键路径或状态，便于人工检查。
            return mol  # 返回该辅助函数的结果。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            last_exc = exc  # 设置中间变量或可调参数，供后续工作流使用。
    raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
        f"{label} is expected to be RESP-ready in MolDB. "
        "Please refresh it with examples/07_moldb_precompute_and_reuse/"
        "03_refresh_cmc_glucose_polyelectrolytes.py."
    ) from last_exc


# ---------------- user inputs ----------------
restart_status = True  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
build_only = _env_flag("YADONPY_BUILD_ONLY", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
export_only = _env_flag("YADONPY_EXPORT_ONLY", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
analysis_only = _env_flag("YADONPY_ANALYSIS_ONLY", default=False)  # 只做后处理，不重新运行采样。
smoke_mode = _env_flag("YADONPY_SMOKE", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
fast_analysis = _env_flag("YADONPY_FAST_ANALYSIS", default=False)  # 设置中间变量或可调参数，供后续工作流使用。

skip_rdf = _env_flag("YADONPY_SKIP_RDF", default=fast_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
skip_den_dis = _env_flag("YADONPY_SKIP_DEN_DIS", default=fast_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
skip_sigma = _env_flag("YADONPY_SKIP_SIGMA", default=fast_analysis)  # 设置中间变量或可调参数，供后续工作流使用。

set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod() if _env_text("YADONPY_FORCEFIELD", "GAFF2_MOD").strip().upper() == "GAFF2_MOD" else GAFF2()  # 选择有机分子/聚合物/部分无机离子的力场对象。
ion_ff = MERZ()  # 选择单原子离子参数来源。

# ---- CMC monomers (two connection points '*...*') ----
glucose_smiles = "*OC1OC(CO)C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。

DTD_smiles = "O=S1(=O)OC=CO1"  # 设置中间变量或可调参数，供后续工作流使用。
VC_smiles = "O=c1occo1"  # 设置中间变量或可调参数，供后续工作流使用。

feed_ratio = [12, 26, 27, 35]  # 设置中间变量或可调参数，供后续工作流使用。
feed_prob = poly.ratio_to_prob(feed_ratio)  # 设置中间变量或可调参数，供后续工作流使用。

target_mw = 10000.0  # 设置中间变量或可调参数，供后续工作流使用。
ter_smiles = "[H][*]"  # 设置中间变量或可调参数，供后续工作流使用。

# ---- Solvents ----
EC_smiles = "O=C1OCCO1"  # 设置中间变量或可调参数，供后续工作流使用。
EMC_smiles = "CCOC(=O)OC"  # 设置中间变量或可调参数，供后续工作流使用。
DEC_smiles = "CCOC(=O)OCC"  # 设置中间变量或可调参数，供后续工作流使用。

# ---- Salt / ions ----
Li_smiles = "[Li+]"  # 设置中间变量或可调参数，供后续工作流使用。
PF6_smiles = "F[P-](F)(F)(F)(F)F"  # 设置中间变量或可调参数，供后续工作流使用。
Na_smiles = "[Na+]"  # 设置中间变量或可调参数，供后续工作流使用。

temp = 318.15  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
press = 1.0  # 设置压力 bar；用于 NPT/EQ 阶段。
mpi = _env_int("YADONPY_MPI", 1)  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = _env_int("YADONPY_OMP", 14)  # 设置每个 rank 的 OpenMP 线程数。
gpu = _env_int("YADONPY_GPU", 1)  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = _env_int("YADONPY_GPU_ID", 0)  # 选择 GPU 设备编号，多卡节点可修改。
prod_ns = _env_float("YADONPY_PROD_NS", 20.0)  # 设置中间变量或可调参数，供后续工作流使用。
prod_traj_ps = _env_text("YADONPY_PROD_TRAJ_PS", os.environ.get("TRAJ_PS", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
prod_energy_ps = _env_text("YADONPY_PROD_ENERGY_PS", os.environ.get("ENERGY_PS", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
prod_log_ps = _env_text("YADONPY_PROD_LOG_PS", os.environ.get("LOG_PS", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
prod_trr_ps = os.environ.get("YADONPY_PROD_TRR_PS", os.environ.get("TRR_PS"))  # 设置中间变量或可调参数，供后续工作流使用。
prod_velocity_ps = os.environ.get("YADONPY_PROD_VELOCITY_PS", os.environ.get("VELOCITY_PS"))  # 设置中间变量或可调参数，供后续工作流使用。
performance_profile = _env_text("PERFORMANCE_PROFILE", "auto")  # 设置中间变量或可调参数，供后续工作流使用。
analysis_profile = _env_text("ANALYSIS_PROFILE", "auto")  # 选择后处理预设；interface_fast 面向 slab/interface。
trajectory_format = _env_text("TRAJECTORY_FORMAT", os.environ.get("YADONPY_TRAJECTORY_FORMAT", "auto"))  # 设置中间变量或可调参数，供后续工作流使用。
prod_checkpoint_min = _env_float("YADONPY_PROD_CPT_MIN", 5.0)  # 设置中间变量或可调参数，供后续工作流使用。
msd_drift = _env_text("YADONPY_MSD_DRIFT", "off")  # 设置中间变量或可调参数，供后续工作流使用。
msd_compare_drift_off = _env_flag(  # 设置中间变量或可调参数，供后续工作流使用。
    "YADONPY_COMPARE_DRIFT_OFF",
    default=(str(msd_drift).strip().lower() != "off" and not smoke_mode),
)
additive_name = _env_text("YADONPY_ADDITIVE", "DTD").strip().upper()  # 设置中间变量或可调参数，供后续工作流使用。
forcefield_name = _env_text("YADONPY_FORCEFIELD", "GAFF2_MOD").strip().upper()  # 设置中间变量或可调参数，供后续工作流使用。
system_variant = _env_text("YADONPY_SYSTEM_VARIANT", "full").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
prod_ensemble = _env_text("YADONPY_PROD_ENSEMBLE", "npt").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
gpu_offload_mode = _env_text("YADONPY_GPU_OFFLOAD_MODE", "conservative").strip().lower()  # 设置中间变量或可调参数，供后续工作流使用。
prod_bridge_ps = _env_float("YADONPY_PROD_BRIDGE_PS", 100.0)  # 设置中间变量或可调参数，供后续工作流使用。
prod_bridge_dt_fs = _env_float("YADONPY_PROD_BRIDGE_DT_FS", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
prod_bridge_lincs_iter = _env_int("YADONPY_PROD_BRIDGE_LINCS_ITER", 4)  # 设置中间变量或可调参数，供后续工作流使用。
prod_bridge_lincs_order = _env_int("YADONPY_PROD_BRIDGE_LINCS_ORDER", 12)  # 设置中间变量或可调参数，供后续工作流使用。
nvt_density_control = _env_flag("YADONPY_NVT_DENSITY_CONTROL", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
counts_override = _env_int_list("YADONPY_COUNTS", 8)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_final_ns = _env_float("YADONPY_EQ21_FINAL_NS", 0.8)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_npt_time_scale = _env_float("YADONPY_EQ21_NPT_TIME_SCALE", 2.0)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_stage_cap = _env_int("YADONPY_EQ21_STAGE_CAP", 0)  # 设置中间变量或可调参数，供后续工作流使用。
additional_ns = _env_float("YADONPY_ADDITIONAL_NS", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
additional_rounds = _env_int("YADONPY_ADDITIONAL_MAX_ROUNDS", 4)  # 设置中间变量或可调参数，供后续工作流使用。

omp_psi4 = _env_int("YADONPY_PSI4_OMP", 20)  # 设置 Psi4/OpenMP 核数。
mem_mb = _env_int("YADONPY_PSI4_MEMORY_MB", 20000)  # 设置量子化学内存 MB。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"  # 设置中间变量或可调参数，供后续工作流使用。
_work_dir_override = str(os.environ.get("YADONPY_WORK_DIR", "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
_shared_polymer_root_override = str(os.environ.get("YADONPY_SHARED_POLYMER_ROOT", "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
_ADDITIVE_LIBRARY = {  # 设置中间变量或可调参数，供后续工作流使用。
    "DTD": {"label": "DTD", "smiles": DTD_smiles, "default_count": 4, "smoke_count": 1},
    "VC": {"label": "VC", "smiles": VC_smiles, "default_count": 4, "smoke_count": 1},
}
if additive_name not in _ADDITIVE_LIBRARY:  # 根据当前状态决定是否进入该分支。
    raise ValueError(f"Unsupported YADONPY_ADDITIVE={additive_name!r}; expected one of {sorted(_ADDITIVE_LIBRARY)}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
if forcefield_name not in {"GAFF2", "GAFF2_MOD"}:  # 根据当前状态决定是否进入该分支。
    raise ValueError("YADONPY_FORCEFIELD must be GAFF2 or GAFF2_MOD")  # 关键步骤失败时立即报错，避免继续生成错误结果。
if system_variant not in {"full", "electrolyte_additive", "polymer_ions", "polymer_solvents"}:  # 根据当前状态决定是否进入该分支。
    raise ValueError("YADONPY_SYSTEM_VARIANT must be full, electrolyte_additive, polymer_ions, or polymer_solvents")  # 关键步骤失败时立即报错，避免继续生成错误结果。
if prod_ensemble not in {"npt", "nvt"}:  # 根据当前状态决定是否进入该分支。
    raise ValueError("YADONPY_PROD_ENSEMBLE must be npt or nvt")  # 关键步骤失败时立即报错，避免继续生成错误结果。
if gpu_offload_mode not in {"full", "conservative", "cpu"}:  # 根据当前状态决定是否进入该分支。
    raise ValueError("YADONPY_GPU_OFFLOAD_MODE must be full, conservative, or cpu")  # 关键步骤失败时立即报错，避免继续生成错误结果。
ADDITIVE = dict(_ADDITIVE_LIBRARY[additive_name])  # 设置中间变量或可调参数，供后续工作流使用。
transport_labels = [str(ADDITIVE["label"]), "EMC", "DEC", "EC"]  # 设置中间变量或可调参数，供后续工作流使用。
work_dir = (  # 设置本例输出目录。
    Path(_work_dir_override).expanduser()
    if _work_dir_override  # 根据当前状态决定是否进入该分支。
    else (BASE_DIR / f"work_dir_{str(ADDITIVE['label']).lower()}_moldb")
)
shared_polymer_root = (  # 设置中间变量或可调参数，供后续工作流使用。
    Path(_shared_polymer_root_override).expanduser()
    if _shared_polymer_root_override  # 根据当前状态决定是否进入该分支。
    else None
)


def _formal_charge(mol) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))  # 返回该辅助函数的结果。


def _write_polymer_checkpoint_audit(audit_dir: Path, label: str, mol) -> Path:  # 定义本例内部辅助函数，组织重复步骤。
    return write_polymer_audit(  # 返回该辅助函数的结果。
        audit_polymer_state(mol, label=label, radius=2),  # 设置中间变量或可调参数，供后续工作流使用。
        audit_dir / f"{label}.json",
    )


def _formulation_counts(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    smoke: bool,
    additive_default_count: int,
    n_na: int,
    variant: str,
    override: list[int] | None,
) -> list[int]:
    if override is not None:  # 根据当前状态决定是否进入该分支。
        return list(override)  # 返回该辅助函数的结果。
    if smoke:  # 根据当前状态决定是否进入该分支。
        base = [1, 6, 6, 6, 2, 2, n_na, int(ADDITIVE["smoke_count"])]  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        base = [8, 40, 50, 20, 10, 10, n_na, int(additive_default_count)]  # 设置中间变量或可调参数，供后续工作流使用。
    if variant == "full":  # 根据当前状态决定是否进入该分支。
        return base  # 返回该辅助函数的结果。
    if variant == "electrolyte_additive":  # 根据当前状态决定是否进入该分支。
        return [0, base[1], base[2], base[3], base[4], base[5], 0, base[7]]  # 返回该辅助函数的结果。
    if variant == "polymer_ions":  # 根据当前状态决定是否进入该分支。
        return [base[0], 0, 0, 0, base[4], base[5], base[6], 0]  # 返回该辅助函数的结果。
    if variant == "polymer_solvents":  # 根据当前状态决定是否进入该分支。
        return [base[0], base[1], base[2], base[3], base[4], base[5], base[6], 0]  # 返回该辅助函数的结果。
    raise ValueError(f"Unsupported system variant: {variant}")  # 关键步骤失败时立即报错，避免继续生成错误结果。


def _default_msd_begin_ps(prod_time_ns: float) -> float:  # 定义本例内部辅助函数，组织重复步骤。
    total_ps = max(0.0, float(prod_time_ns) * 1000.0)  # 设置中间变量或可调参数，供后续工作流使用。
    return min(5000.0, 0.25 * total_ps)  # 返回该辅助函数的结果。


def _extract_species_metric(msd_payload: dict, moltype: str) -> dict:  # 定义本例内部辅助函数，组织重复步骤。
    rec = dict(msd_payload.get(str(moltype)) or {})  # 设置中间变量或可调参数，供后续工作流使用。
    metric_name = str(rec.get("default_metric") or rec.get("metric") or "")  # 设置中间变量或可调参数，供后续工作流使用。
    metrics = dict(rec.get("metrics") or {})  # 设置中间变量或可调参数，供后续工作流使用。
    metric = dict(metrics.get(metric_name) or {})  # 设置中间变量或可调参数，供后续工作流使用。
    return {  # 返回该辅助函数的结果。
        "moltype": str(moltype),
        "n_molecules": int(rec.get("n_molecules") or 0),
        "metric": metric_name,
        "n_groups": int(metric.get("n_groups") or 0),
        "D_m2_s": metric.get("D_m2_s"),
        "confidence": metric.get("confidence"),
        "status": metric.get("status"),
        "warning": metric.get("warning"),
        "alpha_mean": metric.get("alpha_mean"),
        "fit_t_start_ps": metric.get("fit_t_start_ps"),
        "fit_t_end_ps": metric.get("fit_t_end_ps"),
        "preprocessing": dict(metric.get("preprocessing") or {}),
    }


def _rank_transport_species(msd_payload: dict, moltypes: list[str]) -> list[dict]:  # 定义本例内部辅助函数，组织重复步骤。
    rows = [_extract_species_metric(msd_payload, moltype) for moltype in moltypes]  # 设置中间变量或可调参数，供后续工作流使用。
    rows = [row for row in rows if row.get("D_m2_s") is not None]  # 设置中间变量或可调参数，供后续工作流使用。
    rows.sort(key=lambda row: float(row["D_m2_s"]), reverse=True)  # 设置中间变量或可调参数，供后续工作流使用。
    return rows  # 返回该辅助函数的结果。


def _write_transport_diagnostics(  # 定义本例内部辅助函数，组织重复步骤。
    analysis_dir: Path,
    *,
    moltypes: list[str],
    primary_msd: dict,
    primary_label: str,
    secondary_msd: dict | None = None,  # 设置中间变量或可调参数，供后续工作流使用。
    secondary_label: str | None = None,  # 设置中间变量或可调参数，供后续工作流使用。
    begin_ps: float | None,
    end_ps: float | None,
) -> Path:
    primary_rows = [_extract_species_metric(primary_msd, moltype) for moltype in moltypes]  # 设置中间变量或可调参数，供后续工作流使用。
    payload: dict[str, object] = {  # 设置中间变量或可调参数，供后续工作流使用。
        "primary": {
            "label": str(primary_label),
            "begin_ps": begin_ps,
            "end_ps": end_ps,
            "species": primary_rows,
            "ranking": _rank_transport_species(primary_msd, moltypes),
        }
    }
    if secondary_msd is not None and secondary_label:  # 根据当前状态决定是否进入该分支。
        secondary_rows = [_extract_species_metric(secondary_msd, moltype) for moltype in moltypes]  # 设置中间变量或可调参数，供后续工作流使用。
        sensitivity = []  # 设置中间变量或可调参数，供后续工作流使用。
        for primary_row, secondary_row in zip(primary_rows, secondary_rows):  # 遍历当前工作流中的一组对象或任务。
            p = primary_row.get("D_m2_s")  # 设置中间变量或可调参数，供后续工作流使用。
            s = secondary_row.get("D_m2_s")  # 设置中间变量或可调参数，供后续工作流使用。
            delta = None  # 设置中间变量或可调参数，供后续工作流使用。
            ratio = None  # 设置共聚组成比例。
            if p is not None and s is not None:  # 根据当前状态决定是否进入该分支。
                delta = float(s) - float(p)  # 设置中间变量或可调参数，供后续工作流使用。
                if abs(float(p)) > 1.0e-30:  # 根据当前状态决定是否进入该分支。
                    ratio = float(s) / float(p)  # 设置共聚组成比例。
            sensitivity.append(  # 开始一个多行函数调用或配置块。
                {
                    "moltype": primary_row["moltype"],
                    "primary_D_m2_s": p,
                    "secondary_D_m2_s": s,
                    "delta_D_m2_s": delta,
                    "ratio_secondary_to_primary": ratio,
                    "primary_confidence": primary_row.get("confidence"),
                    "secondary_confidence": secondary_row.get("confidence"),
                }
            )
        payload["secondary"] = {  # 设置中间变量或可调参数，供后续工作流使用。
            "label": str(secondary_label),
            "species": secondary_rows,
            "ranking": _rank_transport_species(secondary_msd, moltypes),
            "sensitivity": sensitivity,
        }

    out = analysis_dir / "transport_diagnostics.json"  # 设置中间变量或可调参数，供后续工作流使用。
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
    return out  # 返回该辅助函数的结果。


def _print_transport_summary(*, msd_payload: dict, label: str, moltypes: list[str]) -> None:  # 定义本例内部辅助函数，组织重复步骤。
    ranking = _rank_transport_species(msd_payload, moltypes)  # 设置中间变量或可调参数，供后续工作流使用。
    if not ranking:  # 根据当前状态决定是否进入该分支。
        print(f"[TRANSPORT] {label}: no diffusion coefficients available")  # 打印关键路径或状态，便于人工检查。
        return
    print(f"[TRANSPORT] {label} ranking")  # 打印关键路径或状态，便于人工检查。
    for idx, row in enumerate(ranking, start=1):  # 遍历当前工作流中的一组对象或任务。
        d_val = float(row["D_m2_s"]) if row.get("D_m2_s") is not None else float("nan")  # 设置中间变量或可调参数，供后续工作流使用。
        print(  # 打印关键路径或状态，便于人工检查。
            f"  {idx}. {row['moltype']:<4} D={d_val:.3e} m^2/s "
            f"confidence={row.get('confidence')} status={row.get('status')} "
            f"n_groups={row.get('n_groups')}"
        )


def _warn_if_transport_is_fragile(primary_msd: dict, *, additive_label: str, moltypes: list[str], secondary_msd: dict | None = None) -> None:  # 定义本例内部辅助函数，组织重复步骤。
    additive_row = _extract_species_metric(primary_msd, additive_label)  # 设置中间变量或可调参数，供后续工作流使用。
    if not additive_row:  # 根据当前状态决定是否进入该分支。
        return
    if int(additive_row.get("n_groups") or 0) <= 4:  # 根据当前状态决定是否进入该分支。
        print(  # 打印关键路径或状态，便于人工检查。
            f"[TRANSPORT][WARNING] {additive_label} diffusion is being estimated from 4 or fewer molecular COM groups; "
            "treat ranking against bulk solvents as low-statistics."
        )
    if str(additive_row.get("confidence") or "").lower() not in {"high", "medium"}:  # 根据当前状态决定是否进入该分支。
        print(  # 打印关键路径或状态，便于人工检查。
            f"[TRANSPORT][WARNING] {additive_label} diffusion fit is not high-confidence; "
            "check msd.json / transport_diagnostics.json before trusting solvent-order conclusions."
        )
    if secondary_msd is None:  # 根据当前状态决定是否进入该分支。
        return

    primary_rank = [row["moltype"] for row in _rank_transport_species(primary_msd, moltypes)]  # 设置中间变量或可调参数，供后续工作流使用。
    secondary_rank = [row["moltype"] for row in _rank_transport_species(secondary_msd, moltypes)]  # 设置中间变量或可调参数，供后续工作流使用。
    if primary_rank and secondary_rank and primary_rank != secondary_rank:  # 根据当前状态决定是否进入该分支。
        print(  # 打印关键路径或状态，便于人工检查。
            "[TRANSPORT][WARNING] Solvent/additive ranking changes when drift correction is toggled; "
            "treat this run as drift-sensitive and prefer longer sampling or tail-window reanalysis."
        )

    for moltype in moltypes:  # 遍历当前工作流中的一组对象或任务。
        p = _extract_species_metric(primary_msd, moltype)  # 设置中间变量或可调参数，供后续工作流使用。
        s = _extract_species_metric(secondary_msd, moltype)  # 设置中间变量或可调参数，供后续工作流使用。
        p_d = p.get("D_m2_s")  # 设置中间变量或可调参数，供后续工作流使用。
        s_d = s.get("D_m2_s")  # 设置中间变量或可调参数，供后续工作流使用。
        if p_d is None or s_d is None:  # 根据当前状态决定是否进入该分支。
            continue
        p_d = float(p_d)  # 设置中间变量或可调参数，供后续工作流使用。
        s_d = float(s_d)  # 设置中间变量或可调参数，供后续工作流使用。
        if abs(p_d) <= 1.0e-30:  # 根据当前状态决定是否进入该分支。
            continue
        ratio = abs(s_d / p_d)  # 设置共聚组成比例。
        if ratio >= 5.0 or ratio <= 0.2:  # 根据当前状态决定是否进入该分支。
            print(  # 打印关键路径或状态，便于人工检查。
                f"[TRANSPORT][WARNING] {moltype} diffusion changes by more than 5x when drift correction is toggled "
                f"(ratio={ratio:.3g}); inspect transport_diagnostics.json before comparing solvent order."
            )


def _transport_probe_mols(additive_smiles: str):  # 定义本例内部辅助函数，组织重复步骤。
    return [  # 返回该辅助函数的结果。
        utils.mol_from_smiles(additive_smiles),  # 从 SMILES 直接构造 RDKit 分子。
        utils.mol_from_smiles(EMC_smiles),  # 从 SMILES 直接构造 RDKit 分子。
        utils.mol_from_smiles(DEC_smiles),  # 从 SMILES 直接构造 RDKit 分子。
        utils.mol_from_smiles(EC_smiles),  # 从 SMILES 直接构造 RDKit 分子。
    ]


def _extract_cmd_from_exception(message: str) -> str | None:  # 定义本例内部辅助函数，组织重复步骤。
    for line in str(message).splitlines():  # 遍历当前工作流中的一组对象或任务。
        if line.strip().startswith("cmd:"):  # 根据当前状态决定是否进入该分支。
            return line.split("cmd:", 1)[1].strip()  # 返回该辅助函数的结果。
    return None  # 返回该辅助函数的结果。


def _read_log_tail(path: Path, *, tail_chars: int = 12000) -> str | None:  # 定义本例内部辅助函数，组织重复步骤。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        text = path.read_text(encoding="utf-8", errors="replace")  # 设置中间变量或可调参数，供后续工作流使用。
    except Exception:  # 捕获异常并转成更清楚的示例错误信息。
        return None  # 返回该辅助函数的结果。
    if len(text) <= int(tail_chars):  # 根据当前状态决定是否进入该分支。
        return text  # 返回该辅助函数的结果。
    return text[-int(tail_chars):]  # 返回该辅助函数的结果。


def _write_failure_diagnostics(*, work_root: Path, stage: str, stage_dir: Path, exc: BaseException) -> Path:  # 定义本例内部辅助函数，组织重复步骤。
    log_path = Path(stage_dir) / "md.log"  # 设置中间变量或可调参数，供后续工作流使用。
    checkpoint_path = Path(stage_dir) / "md.cpt"  # 设置中间变量或可调参数，供后续工作流使用。
    payload = {  # 设置中间变量或可调参数，供后续工作流使用。
        "stage": str(stage),
        "stage_dir": str(stage_dir),
        "exception_type": exc.__class__.__name__,
        "exception": str(exc),
        "command": _extract_cmd_from_exception(str(exc)),
        "checkpoint_exists": bool(checkpoint_path.exists()),
        "log_exists": bool(log_path.exists()),
        "lincs_fallback_eligible": bool(checkpoint_path.exists()) and ("lincs" in str(exc).lower() or "constraint" in str(exc).lower()),
        "log_tail": _read_log_tail(log_path),
    }
    out = Path(work_root) / "failure_diagnostics.json"  # 设置中间变量或可调参数，供后续工作流使用。
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 设置中间变量或可调参数，供后续工作流使用。
    print(f"[FAILURE] wrote diagnostics to {out}")  # 打印关键路径或状态，便于人工检查。
    return out  # 返回该辅助函数的结果。


def _run_partial_eq21(  # 定义本例内部辅助函数，组织重复步骤。
    *,
    eqmd: eq.EQ21step,
    temp: float,
    press: float,
    mpi: int,
    omp: int,
    gpu: int,
    gpu_id: int | None,
    stage_cap: int,
    final_ns: float,
):
    exp = eqmd.ensure_system_exported()  # 设置中间变量或可调参数，供后续工作流使用。
    run_dir = Path(eqmd.work_dir) / "03_EQ21_partial"  # 设置中间变量或可调参数，供后续工作流使用。
    if run_dir.exists():  # 根据当前状态决定是否进入该分支。
        import shutil  # 导入本例需要的库或 yadonpy 接口。

        shutil.rmtree(run_dir, ignore_errors=True)  # 设置中间变量或可调参数，供后续工作流使用。

    cfg = eq.EQ21ProtocolConfig(  # 设置中间变量或可调参数，供后续工作流使用。
        t_max_k=float(temp),  # 设置中间变量或可调参数，供后续工作流使用。
        t_anneal_k=float(temp),  # 设置中间变量或可调参数，供后续工作流使用。
        p_max_bar=float(press),  # 设置中间变量或可调参数，供后续工作流使用。
        p_anneal_bar=float(press),  # 设置中间变量或可调参数，供后续工作流使用。
        npt_time_scale=float(eq21_npt_time_scale),  # 设置中间变量或可调参数，供后续工作流使用。
    )
    stages, records, params = eq._build_eq21_stages(  # 设置中间变量或可调参数，供后续工作流使用。
        temp=float(temp),  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
        press=float(press),  # 设置压力 bar；用于 NPT/EQ 阶段。
        final_ns=float(final_ns),  # 设置中间变量或可调参数，供后续工作流使用。
        cfg=cfg,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    capped_stages = list(stages[: int(stage_cap)])  # 设置中间变量或可调参数，供后续工作流使用。
    capped_records = list(records[: int(stage_cap)])  # 设置中间变量或可调参数，供后续工作流使用。
    if not capped_stages:  # 根据当前状态决定是否进入该分支。
        raise ValueError(f"YADONPY_EQ21_STAGE_CAP={stage_cap} does not select any stage.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    eq._write_eq21_schedule(run_dir, capped_records, params)
    use_gpu, gid = eq._parse_gpu_args(gpu, gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
    res = eq.RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid)  # 设置中间变量或可调参数，供后续工作流使用。
    job = eq.EquilibrationJob(  # 设置中间变量或可调参数，供后续工作流使用。
        gro=exp.system_gro,  # 设置中间变量或可调参数，供后续工作流使用。
        top=exp.system_top,  # 设置中间变量或可调参数，供后续工作流使用。
        provenance_ndx=exp.system_ndx,  # 设置中间变量或可调参数，供后续工作流使用。
        out_dir=run_dir,  # 设置中间变量或可调参数，供后续工作流使用。
        stages=capped_stages,  # 设置中间变量或可调参数，供后续工作流使用。
        resources=res,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    job.run(restart=False)  # 设置中间变量或可调参数，供后续工作流使用。
    print(  # 打印关键路径或状态，便于人工检查。
        "[EQ21-PARTIAL] completed stages: "
        + ", ".join(str(stage.name) for stage in capped_stages)
        + f" | output={run_dir}"  # 设置中间变量或可调参数，供后续工作流使用。
    )
    return run_dir  # 返回该辅助函数的结果。


def main() -> int:  # 定义本例内部辅助函数，组织重复步骤。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。

    wd = workdir(work_dir, restart=restart_status)  # 创建或复用本例工作目录。
    analysis_dir = Path(wd) / "06_analysis"  # 设置中间变量或可调参数，供后续工作流使用。
    audit_dir = Path(wd) / "07_polymer_audit"  # 设置中间变量或可调参数，供后续工作流使用。
    if shared_polymer_root is not None:  # 根据当前状态决定是否进入该分支。
        shared_polymer_wd = workdir(shared_polymer_root, restart=restart_status)  # 创建或复用本例工作目录。
        cmc_rw_dir = shared_polymer_wd.child("CMC_rw")  # 设置中间变量或可调参数，供后续工作流使用。
        cmc_term_dir = shared_polymer_wd.child("CMC_term")  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        cmc_rw_dir = wd.child("CMC_rw")  # 设置中间变量或可调参数，供后续工作流使用。
        cmc_term_dir = wd.child("CMC_term")  # 设置中间变量或可调参数，供后续工作流使用。
    ac_build_dir = wd.child("00_build_cell")  # 设置中间变量或可调参数，供后续工作流使用。

    msd_begin_ps = _env_optional_float("YADONPY_MSD_BEGIN_PS")  # 设置中间变量或可调参数，供后续工作流使用。
    if msd_begin_ps is None:  # 根据当前状态决定是否进入该分支。
        msd_begin_ps = _default_msd_begin_ps(prod_ns)  # 设置中间变量或可调参数，供后续工作流使用。
    msd_end_ps = _env_optional_float("YADONPY_MSD_END_PS")  # 设置中间变量或可调参数，供后续工作流使用。

    if analysis_only:  # 根据当前状态决定是否进入该分支。
        analy = AnalyzeResult.from_work_dir(wd)  # 设置中间变量或可调参数，供后续工作流使用。
        _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        if not skip_rdf:  # 根据当前状态决定是否进入该分支。
            li_probe = utils.mol_from_smiles(Li_smiles)  # 从 SMILES 直接构造 RDKit 分子。
            _ = analy.rdf(center_mol=li_probe)  # 设置中间变量或可调参数，供后续工作流使用。
        transport_mols = _transport_probe_mols(str(ADDITIVE["smiles"]))  # 设置中间变量或可调参数，供后续工作流使用。
        secondary_msd = None  # 设置中间变量或可调参数，供后续工作流使用。
        if msd_compare_drift_off and str(msd_drift).strip().lower() != "off":  # 根据当前状态决定是否进入该分支。
            secondary_msd = analy.msd(  # 设置中间变量或可调参数，供后续工作流使用。
                mols=transport_mols,  # 设置中间变量或可调参数，供后续工作流使用。
                geometry="3d",  # 设置中间变量或可调参数，供后续工作流使用。
                unwrap="on",  # 设置中间变量或可调参数，供后续工作流使用。
                begin_ps=msd_begin_ps,  # 设置中间变量或可调参数，供后续工作流使用。
                end_ps=msd_end_ps,  # 设置中间变量或可调参数，供后续工作流使用。
                drift="off",  # 设置中间变量或可调参数，供后续工作流使用。
            )
        primary_msd = analy.msd(  # 设置中间变量或可调参数，供后续工作流使用。
            mols=transport_mols,  # 设置中间变量或可调参数，供后续工作流使用。
            geometry="3d",  # 设置中间变量或可调参数，供后续工作流使用。
            unwrap="on",  # 设置中间变量或可调参数，供后续工作流使用。
            begin_ps=msd_begin_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            end_ps=msd_end_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            drift=msd_drift,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        if not skip_sigma:  # 根据当前状态决定是否进入该分支。
            sigma_msd = analy.msd(  # 设置中间变量或可调参数，供后续工作流使用。
                geometry="3d",  # 设置中间变量或可调参数，供后续工作流使用。
                unwrap="on",  # 设置中间变量或可调参数，供后续工作流使用。
                begin_ps=msd_begin_ps,  # 设置中间变量或可调参数，供后续工作流使用。
                end_ps=msd_end_ps,  # 设置中间变量或可调参数，供后续工作流使用。
                drift=msd_drift,  # 设置中间变量或可调参数，供后续工作流使用。
            )
            _ = analy.sigma(temp_k=temp, msd=sigma_msd, drift=msd_drift)  # 设置中间变量或可调参数，供后续工作流使用。
        if not skip_den_dis:  # 根据当前状态决定是否进入该分支。
            _ = analy.den_dis()  # 设置中间变量或可调参数，供后续工作流使用。
        diag_path = _write_transport_diagnostics(  # 设置中间变量或可调参数，供后续工作流使用。
            analysis_dir,
            moltypes=transport_labels,  # 设置中间变量或可调参数，供后续工作流使用。
            primary_msd=primary_msd,  # 设置中间变量或可调参数，供后续工作流使用。
            primary_label=f"drift={msd_drift}",  # 设置中间变量或可调参数，供后续工作流使用。
            secondary_msd=secondary_msd,  # 设置中间变量或可调参数，供后续工作流使用。
            secondary_label="drift=off" if secondary_msd is not None else None,  # 设置中间变量或可调参数，供后续工作流使用。
            begin_ps=msd_begin_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            end_ps=msd_end_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        _print_transport_summary(msd_payload=primary_msd, label=f"primary drift={msd_drift}", moltypes=transport_labels)  # 设置中间变量或可调参数，供后续工作流使用。
        if secondary_msd is not None:  # 根据当前状态决定是否进入该分支。
            _print_transport_summary(msd_payload=secondary_msd, label="secondary drift=off", moltypes=transport_labels)  # 设置中间变量或可调参数，供后续工作流使用。
        _warn_if_transport_is_fragile(  # 开始一个多行函数调用或配置块。
            primary_msd,
            additive_label=str(ADDITIVE["label"]),  # 设置中间变量或可调参数，供后续工作流使用。
            moltypes=transport_labels,  # 设置中间变量或可调参数，供后续工作流使用。
            secondary_msd=secondary_msd,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        print(f"[ANALYSIS-ONLY] Transport diagnostics written to {diag_path}")  # 打印关键路径或状态，便于人工检查。
        return 0  # 返回该辅助函数的结果。

    # ---------------- build monomers ----------------
    glucose = utils.mol_from_smiles(glucose_smiles)  # 从 SMILES 直接构造 RDKit 分子。
    glucose = ff.ff_assign(glucose)  # 分配力场参数并写入分子属性。
    if not glucose:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Can not assign force field parameters for glucose.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    glucose_2 = _load_ready_from_moldb(  # 设置中间变量或可调参数，供后续工作流使用。
        ff,
        glucose_2_smiles,
        label="glucose_2",  # 给该选区一个可读标签，便于 manifest 检查。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    glucose_3 = _load_ready_from_moldb(  # 设置中间变量或可调参数，供后续工作流使用。
        ff,
        glucose_3_smiles,
        label="glucose_3",  # 给该选区一个可读标签，便于 manifest 检查。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    glucose_6 = _load_ready_from_moldb(  # 设置中间变量或可调参数，供后续工作流使用。
        ff,
        glucose_6_smiles,
        label="glucose_6",  # 给该选区一个可读标签，便于 manifest 检查。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    for name, mol in (  # 遍历当前工作流中的一组对象或任务。
        ("glucose_2_ready", glucose_2),
        ("glucose_3_ready", glucose_3),
        ("glucose_6_ready", glucose_6),
    ):
        _write_polymer_checkpoint_audit(audit_dir, name, mol)

    # termination
    ter1 = utils.mol_from_smiles(ter_smiles)  # 从 SMILES 直接构造 RDKit 分子。
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        ter1,
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        opt=True,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=wd,  # 设置本例输出目录。
        omp=omp_psi4,  # 设置每个 rank 的 OpenMP 线程数。
        memory=mem_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        log_name="ter1",  # 设置中间变量或可调参数，供后续工作流使用。
    )

    # DP from target polymer Mw
    dp = poly.calc_n_from_mol_weight(  # 设置或计算聚合度。
        [glucose, glucose_2, glucose_3, glucose_6],
        target_mw,
        ratio=feed_prob,  # 设置共聚组成比例。
        terminal1=ter1,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    print(f"[CMC] estimated DP from target Mw = {dp}")  # 打印关键路径或状态，便于人工检查。

    # random copolymerization (self-avoiding RW), then terminate
    chain_len = 12 if smoke_mode else 50  # 设置中间变量或可调参数，供后续工作流使用。
    CMC = poly.random_copolymerize_rw(  # 用随机游走生成聚合物链。
        [glucose, glucose_2, glucose_3, glucose_6],
        chain_len,
        ratio=feed_prob,  # 设置共聚组成比例。
        tacticity="atactic",  # 设置聚合物立构。
        work_dir=cmc_rw_dir,  # 设置本例输出目录。
    )
    _write_polymer_checkpoint_audit(audit_dir, "cmc_random_walk", CMC)
    CMC = poly.terminate_rw(CMC, ter1, work_dir=cmc_term_dir)  # 给聚合物链加端基。
    _write_polymer_checkpoint_audit(audit_dir, "cmc_terminated", CMC)
    CMC = ff.ff_assign(CMC, polyelectrolyte_mode=True)  # 分配力场参数并写入分子属性。
    if not CMC:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Can not assign force field parameters for CMC.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    _write_polymer_checkpoint_audit(audit_dir, "cmc_final_assigned", CMC)
    q_poly = _formal_charge(CMC)  # 设置中间变量或可调参数，供后续工作流使用。

    # ---------------- build solvents / additive ----------------
    EC = ff.ff_assign(utils.mol_from_smiles(EC_smiles))  # 分配力场参数并写入分子属性。
    EMC = ff.ff_assign(utils.mol_from_smiles(EMC_smiles))  # 分配力场参数并写入分子属性。
    DEC = ff.ff_assign(utils.mol_from_smiles(DEC_smiles))  # 分配力场参数并写入分子属性。
    if not EC or not EMC or not DEC:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Can not assign force field parameters for carbonate solvents.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    additive = _load_ready_from_moldb(  # 设置中间变量或可调参数，供后续工作流使用。
        ff,
        str(ADDITIVE["smiles"]),
        label=str(ADDITIVE["label"]),  # 给该选区一个可读标签，便于 manifest 检查。
        repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    # ---------------- ions ----------------
    Li = ion_ff.mol(Li_smiles)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    Li = ion_ff.ff_assign(Li)  # 分配力场参数并写入分子属性。
    if not Li:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Can not assign MERZ force field parameters for Li+.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    Na = ion_ff.mol(Na_smiles)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    Na = ion_ff.ff_assign(Na)  # 分配力场参数并写入分子属性。
    if not Na:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Can not assign MERZ force field parameters for Na+.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    PF6 = _load_ready_from_moldb(ff, PF6_smiles, label="PF6", bonded="DRIH", repo_db_dir=REPO_DB_DIR)  # 设置中间变量或可调参数，供后续工作流使用。

    # ---------------- compute counts ----------------
    n_cmc = 1 if smoke_mode else 8  # 设置中间变量或可调参数，供后续工作流使用。
    n_na = abs(q_poly) * n_cmc  # 设置中间变量或可调参数，供后续工作流使用。
    additive_count = int(ADDITIVE["smoke_count"] if smoke_mode else ADDITIVE["default_count"])  # 设置中间变量或可调参数，供后续工作流使用。
    counts = _formulation_counts(  # 设置各 species 的数量；顺序必须和 species 列表一致。
        smoke=smoke_mode,  # 设置中间变量或可调参数，供后续工作流使用。
        additive_default_count=additive_count,  # 设置中间变量或可调参数，供后续工作流使用。
        n_na=n_na,  # 设置中间变量或可调参数，供后续工作流使用。
        variant=system_variant,  # 设置中间变量或可调参数，供后续工作流使用。
        override=counts_override,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    charge_scale = [0.7, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 1.0]  # 设置电荷缩放系数；1.0 表示全电荷模型。

    print(  # 打印关键路径或状态，便于人工检查。
        f"[FORMULATION] ff={forcefield_name} variant={system_variant} additive={ADDITIVE['label']} "
        f"smoke_mode={smoke_mode} q_poly={q_poly} counts={counts}"
    )
    print(  # 打印关键路径或状态，便于人工检查。
        f"[RUNCFG] mpi={mpi} omp={omp} gpu={gpu} gpu_id={gpu_id} "
        f"eq21_final_ns={eq21_final_ns} eq21_npt_time_scale={eq21_npt_time_scale} "
        f"additional_ns={additional_ns} additional_rounds={additional_rounds} prod_ns={prod_ns} "
        f"shared_polymer_root={shared_polymer_root if shared_polymer_root is not None else '(none)'}"
    )
    print(  # 打印关键路径或状态，便于人工检查。
        f"[PRODMODE] ensemble={prod_ensemble} gpu_offload_mode={gpu_offload_mode} "
        f"bridge_ps={prod_bridge_ps} bridge_dt_fs={prod_bridge_dt_fs} "
        f"bridge_lincs_iter={prod_bridge_lincs_iter} bridge_lincs_order={prod_bridge_lincs_order} "
        f"nvt_density_control={nvt_density_control}"
    )
    print(  # 打印关键路径或状态，便于人工检查。
        f"[ANALYSISCFG] msd_begin_ps={msd_begin_ps} msd_end_ps={msd_end_ps} "
        f"msd_drift={msd_drift} compare_drift_off={msd_compare_drift_off} "
        f"skip_rdf={skip_rdf} skip_sigma={skip_sigma} skip_den_dis={skip_den_dis}"
    )
    print(  # 打印关键路径或状态，便于人工检查。
        f"[PRODOUT] traj_ps={prod_traj_ps} energy_ps={prod_energy_ps} "
        f"log_ps={prod_log_ps if prod_log_ps is not None else prod_energy_ps} "
        f"trr_ps={prod_trr_ps} velocity_ps={prod_velocity_ps} checkpoint_min={prod_checkpoint_min}"
    )

    # ---------------- pack amorphous cell ----------------
    species = [CMC, EC, EMC, DEC, Li, PF6, Na, additive]  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。
    active = [(mol, cnt, scl) for mol, cnt, scl in zip(species, counts, charge_scale) if int(cnt) > 0]  # 设置中间变量或可调参数，供后续工作流使用。
    active_mols = [mol for mol, _cnt, _scl in active]  # 设置中间变量或可调参数，供后续工作流使用。
    active_counts = [int(cnt) for _mol, cnt, _scl in active]  # 设置中间变量或可调参数，供后续工作流使用。
    active_charge_scale = [float(scl) for _mol, _cnt, scl in active]  # 设置中间变量或可调参数，供后续工作流使用。
    ac = poly.amorphous_cell(  # 构建无定形混合体系初始盒子。
        active_mols,
        active_counts,
        charge_scale=active_charge_scale,  # 设置电荷缩放系数；1.0 表示全电荷模型。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        density=0.05,  # 设置中间变量或可调参数，供后续工作流使用。
        neutralize=False,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=ac_build_dir,  # 设置本例输出目录。
    )

    if build_only:  # 根据当前状态决定是否进入该分支。
        print(f"[BUILD-ONLY] Finished cell construction at {ac_build_dir}")  # 打印关键路径或状态，便于人工检查。
        return 0  # 返回该辅助函数的结果。

    # ---------------- run equilibration preset ----------------
    eqmd = eq.EQ21step(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
    if export_only:  # 根据当前状态决定是否进入该分支。
        exported = eqmd.ensure_system_exported()  # 设置中间变量或可调参数，供后续工作流使用。
        write_polymer_audit(  # 开始一个多行函数调用或配置块。
            compare_exported_charge_groups(system_dir=exported.system_top.parent, moltype="CMC", mol=CMC),  # 设置中间变量或可调参数，供后续工作流使用。
            audit_dir / "cmc_export_charge_groups.json",
        )
        print(f"[EXPORT-ONLY] Exported 02_system at {exported.system_top.parent}")  # 打印关键路径或状态，便于人工检查。
        return 0  # 返回该辅助函数的结果。
    if eq21_stage_cap > 0:  # 根据当前状态决定是否进入该分支。
        _run_partial_eq21(  # 开始一个多行函数调用或配置块。
            eqmd=eqmd,  # 设置中间变量或可调参数，供后续工作流使用。
            temp=temp,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
            press=press,  # 设置压力 bar；用于 NPT/EQ 阶段。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
            gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
            stage_cap=eq21_stage_cap,  # 设置中间变量或可调参数，供后续工作流使用。
            final_ns=eq21_final_ns,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        return 0  # 返回该辅助函数的结果。
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        ac = eqmd.exec(  # 设置中间变量或可调参数，供后续工作流使用。
            temp=temp,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
            press=press,  # 设置压力 bar；用于 NPT/EQ 阶段。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
            gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
            time=eq21_final_ns,  # 设置中间变量或可调参数，供后续工作流使用。
            eq21_npt_time_scale=eq21_npt_time_scale,  # 设置中间变量或可调参数，供后续工作流使用。
        )
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        _write_failure_diagnostics(work_root=Path(wd), stage="eq21", stage_dir=Path(wd) / "03_EQ21", exc=exc)  # 设置中间变量或可调参数，供后续工作流使用。
        raise

    exported = eqmd.ensure_system_exported()  # 设置中间变量或可调参数，供后续工作流使用。
    write_polymer_audit(  # 开始一个多行函数调用或配置块。
        compare_exported_charge_groups(system_dir=exported.system_top.parent, moltype="CMC", mol=CMC),  # 设置中间变量或可调参数，供后续工作流使用。
        audit_dir / "cmc_export_charge_groups.json",
    )

    analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    for _i in range(additional_rounds):  # 遍历当前工作流中的一组对象或任务。
        if result:  # 根据当前状态决定是否进入该分支。
            break
        eqmd = eq.Additional(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            ac = eqmd.exec(  # 设置中间变量或可调参数，供后续工作流使用。
                temp=temp,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
                press=press,  # 设置压力 bar；用于 NPT/EQ 阶段。
                mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
                omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
                gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
                gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
                time=additional_ns,  # 设置中间变量或可调参数，供后续工作流使用。
            )
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            _write_failure_diagnostics(  # 开始一个多行函数调用或配置块。
                work_root=Path(wd),  # 设置中间变量或可调参数，供后续工作流使用。
                stage=f"additional_round_{_i + 1:02d}",  # 设置中间变量或可调参数，供后续工作流使用。
                stage_dir=Path(wd) / "04_eq_additional",  # 设置中间变量或可调参数，供后续工作流使用。
                exc=exc,  # 设置中间变量或可调参数，供后续工作流使用。
            )
            raise
        analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    if not result:  # 根据当前状态决定是否进入该分支。
        print("[WARNING] Did not reach an equilibrium state after EQ21 + Additional cycles.")  # 打印关键路径或状态，便于人工检查。

    # ---------------- Production + analysis ----------------
    estimated_atoms = int(ac.GetNumAtoms()) if hasattr(ac, "GetNumAtoms") else None  # 设置中间变量或可调参数，供后续工作流使用。
    io_policy = resolve_io_analysis_policy(  # 设置中间变量或可调参数，供后续工作流使用。
        prod_ns=float(prod_ns),  # 设置中间变量或可调参数，供后续工作流使用。
        atom_count=estimated_atoms,  # 设置中间变量或可调参数，供后续工作流使用。
        performance_profile=performance_profile,  # 设置中间变量或可调参数，供后续工作流使用。
        analysis_profile=analysis_profile,  # 选择后处理预设；interface_fast 面向 slab/interface。
        trajectory_format=trajectory_format,  # 设置中间变量或可调参数，供后续工作流使用。
        traj_ps=prod_traj_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        energy_ps=prod_energy_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        log_ps=prod_log_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        trr_ps=prod_trr_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        velocity_ps=prod_velocity_ps,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    prod_runner: eq.NPT | eq.NVT
    if prod_ensemble == "nvt":  # 根据当前状态决定是否进入该分支。
        prod_runner = eq.NVT(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
        prod_stage_name = "nvt_production"  # 设置中间变量或可调参数，供后续工作流使用。
        prod_stage_dir = Path(wd) / "05_nvt_production"  # 设置中间变量或可调参数，供后续工作流使用。
        prod_kwargs = {  # 设置中间变量或可调参数，供后续工作流使用。
            "temp": temp,
            "mpi": mpi,
            "omp": omp,
            "gpu": gpu,
            "gpu_id": gpu_id,
            "time": prod_ns,
            "traj_ps": io_policy.traj_ps,
            "energy_ps": io_policy.energy_ps,
            "log_ps": io_policy.log_ps,
            "trr_ps": io_policy.trr_ps,
            "velocity_ps": io_policy.velocity_ps,
            "trajectory_format": io_policy.trajectory_format,
            "performance_profile": io_policy.performance_profile,
            "analysis_profile": io_policy.analysis_profile,
            "checkpoint_min": prod_checkpoint_min,
            "gpu_offload_mode": gpu_offload_mode,
            "bridge_ps": prod_bridge_ps,
            "bridge_dt_fs": prod_bridge_dt_fs,
            "bridge_lincs_iter": prod_bridge_lincs_iter,
            "bridge_lincs_order": prod_bridge_lincs_order,
            "density_control": nvt_density_control,
        }
    else:  # 处理前面条件都不满足的情况。
        prod_runner = eq.NPT(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
        prod_stage_name = "npt_production"  # 设置中间变量或可调参数，供后续工作流使用。
        prod_stage_dir = Path(wd) / "05_npt_production"  # 设置中间变量或可调参数，供后续工作流使用。
        prod_kwargs = {  # 设置中间变量或可调参数，供后续工作流使用。
            "temp": temp,
            "press": press,
            "mpi": mpi,
            "omp": omp,
            "gpu": gpu,
            "gpu_id": gpu_id,
            "time": prod_ns,
            "traj_ps": io_policy.traj_ps,
            "energy_ps": io_policy.energy_ps,
            "log_ps": io_policy.log_ps,
            "trr_ps": io_policy.trr_ps,
            "velocity_ps": io_policy.velocity_ps,
            "trajectory_format": io_policy.trajectory_format,
            "performance_profile": io_policy.performance_profile,
            "analysis_profile": io_policy.analysis_profile,
            "checkpoint_min": prod_checkpoint_min,
            "gpu_offload_mode": gpu_offload_mode,
            "bridge_ps": prod_bridge_ps,
            "bridge_dt_fs": prod_bridge_dt_fs,
            "bridge_lincs_iter": prod_bridge_lincs_iter,
            "bridge_lincs_order": prod_bridge_lincs_order,
        }
    try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
        ac = prod_runner.exec(**prod_kwargs)  # 设置中间变量或可调参数，供后续工作流使用。
    except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
        _write_failure_diagnostics(  # 开始一个多行函数调用或配置块。
            work_root=Path(wd),  # 设置中间变量或可调参数，供后续工作流使用。
            stage=prod_stage_name,  # 设置中间变量或可调参数，供后续工作流使用。
            stage_dir=prod_stage_dir,  # 设置中间变量或可调参数，供后续工作流使用。
            exc=exc,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        raise

    analy = prod_runner.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    if not skip_rdf and counts[4] > 0:  # 根据当前状态决定是否进入该分支。
        _ = analy.rdf(center_mol=Li)  # 设置中间变量或可调参数，供后续工作流使用。
    transport_mols = []  # 设置中间变量或可调参数，供后续工作流使用。
    present_transport_labels: list[str] = []  # 设置中间变量或可调参数，供后续工作流使用。
    if counts[7] > 0:  # 根据当前状态决定是否进入该分支。
        transport_mols.append(additive)
        present_transport_labels.append(str(ADDITIVE["label"]))
    if counts[2] > 0:  # 根据当前状态决定是否进入该分支。
        transport_mols.append(EMC)
        present_transport_labels.append("EMC")
    if counts[3] > 0:  # 根据当前状态决定是否进入该分支。
        transport_mols.append(DEC)
        present_transport_labels.append("DEC")
    if counts[1] > 0:  # 根据当前状态决定是否进入该分支。
        transport_mols.append(EC)
        present_transport_labels.append("EC")
    msd_drift_off = None  # 设置中间变量或可调参数，供后续工作流使用。
    if transport_mols and msd_compare_drift_off and str(msd_drift).strip().lower() != "off":  # 根据当前状态决定是否进入该分支。
        msd_drift_off = analy.msd(  # 设置中间变量或可调参数，供后续工作流使用。
            mols=transport_mols,  # 设置中间变量或可调参数，供后续工作流使用。
            geometry="3d",  # 设置中间变量或可调参数，供后续工作流使用。
            unwrap="on",  # 设置中间变量或可调参数，供后续工作流使用。
            begin_ps=msd_begin_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            end_ps=msd_end_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            drift="off",  # 设置中间变量或可调参数，供后续工作流使用。
        )
    msd = (  # 设置中间变量或可调参数，供后续工作流使用。
        analy.msd(  # 开始一个多行函数调用或配置块。
            mols=transport_mols,  # 设置中间变量或可调参数，供后续工作流使用。
            geometry="3d",  # 设置中间变量或可调参数，供后续工作流使用。
            unwrap="on",  # 设置中间变量或可调参数，供后续工作流使用。
            begin_ps=msd_begin_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            end_ps=msd_end_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            drift=msd_drift,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        if transport_mols  # 根据当前状态决定是否进入该分支。
        else {}
    )
    if not skip_sigma:  # 根据当前状态决定是否进入该分支。
        sigma_msd = analy.msd(  # 设置中间变量或可调参数，供后续工作流使用。
            geometry="3d",  # 设置中间变量或可调参数，供后续工作流使用。
            unwrap="on",  # 设置中间变量或可调参数，供后续工作流使用。
            begin_ps=msd_begin_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            end_ps=msd_end_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            drift=msd_drift,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        _ = analy.sigma(temp_k=temp, msd=sigma_msd, drift=msd_drift)  # 设置中间变量或可调参数，供后续工作流使用。
    if not skip_den_dis:  # 根据当前状态决定是否进入该分支。
        _ = analy.den_dis()  # 设置中间变量或可调参数，供后续工作流使用。
    diag_path = None  # 设置中间变量或可调参数，供后续工作流使用。
    if transport_mols:  # 根据当前状态决定是否进入该分支。
        diag_path = _write_transport_diagnostics(  # 设置中间变量或可调参数，供后续工作流使用。
            analysis_dir,
            moltypes=present_transport_labels,  # 设置中间变量或可调参数，供后续工作流使用。
            primary_msd=msd,  # 设置中间变量或可调参数，供后续工作流使用。
            primary_label=f"drift={msd_drift}",  # 设置中间变量或可调参数，供后续工作流使用。
            secondary_msd=msd_drift_off,  # 设置中间变量或可调参数，供后续工作流使用。
            secondary_label="drift=off" if msd_drift_off is not None else None,  # 设置中间变量或可调参数，供后续工作流使用。
            begin_ps=msd_begin_ps,  # 设置中间变量或可调参数，供后续工作流使用。
            end_ps=msd_end_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        _print_transport_summary(msd_payload=msd, label=f"primary drift={msd_drift}", moltypes=present_transport_labels)  # 设置中间变量或可调参数，供后续工作流使用。
        if msd_drift_off is not None:  # 根据当前状态决定是否进入该分支。
            _print_transport_summary(msd_payload=msd_drift_off, label="secondary drift=off", moltypes=present_transport_labels)  # 设置中间变量或可调参数，供后续工作流使用。
        _warn_if_transport_is_fragile(  # 开始一个多行函数调用或配置块。
            msd,
            additive_label=str(ADDITIVE["label"]),  # 设置中间变量或可调参数，供后续工作流使用。
            moltypes=present_transport_labels,  # 设置中间变量或可调参数，供后续工作流使用。
            secondary_msd=msd_drift_off,  # 设置中间变量或可调参数，供后续工作流使用。
        )
        print(f"[TRANSPORT] diagnostics written to {diag_path}")  # 打印关键路径或状态，便于人工检查。
    else:  # 处理前面条件都不满足的情况。
        print("[TRANSPORT] skipped transport diagnostics because this system variant has no transport probes.")  # 打印关键路径或状态，便于人工检查。
    return 0  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    raise SystemExit(main())  # 关键步骤失败时立即报错，避免继续生成错误结果。
