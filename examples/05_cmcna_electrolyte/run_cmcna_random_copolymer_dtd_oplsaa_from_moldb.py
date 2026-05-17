from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""CMC random copolymer + DTD electrolyte example using OPLS-AA.

This variant keeps RESP charges for all non-ionic species by loading them from
MolDB. Li+ and Na+ use the built-in OPLS-AA ion parameters directly. PF6-
reuses its MolDB DRIH bonded topology, then swaps its atom types / charges to
the built-in OPLS-AA ion values because the bundled OPLS-AA table does not yet
ship PF6 bonded terms.

Use ``YADONPY_BUILD_ONLY=1`` to stop after amorphous-cell construction.
Use ``YADONPY_EXPORT_ONLY=1`` to stop after exporting ``02_system``.
Use ``YADONPY_SMOKE=1`` for a smaller polymer / solvent composition.
Use ``YADONPY_EQ21_STAGE_CAP=<N>`` to run only the first ``N`` EQ21 stages
(for example ``3`` = ``01_em`` + ``02_preNVT`` + ``03_EQ21/step_01``).
"""

import os  # 导入本例需要的库或 yadonpy 接口。
import shutil  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core import poly, utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff import GAFF2_mod, OPLSAA  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim import qm  # 导入本例需要的库或 yadonpy 接口。
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


def _env_text(name: str, default: str = "") -> str:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None:  # 根据当前状态决定是否进入该分支。
        return str(default)  # 返回该辅助函数的结果。
    text = str(raw).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    return text if text else str(default)  # 返回该辅助函数的结果。


def _env_int_list(name: str, expected_len: int) -> list[int] | None:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = str(os.environ.get(name, "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
    if not raw:  # 根据当前状态决定是否进入该分支。
        return None  # 返回该辅助函数的结果。
    vals = [int(tok.strip()) for tok in raw.split(",") if str(tok).strip()]  # 设置中间变量或可调参数，供后续工作流使用。
    if len(vals) != int(expected_len):  # 根据当前状态决定是否进入该分支。
        raise ValueError(f"{name} expects {expected_len} comma-separated integers, got {len(vals)}")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    return vals  # 返回该辅助函数的结果。


def _normalize_opls_charge_mode(raw: str | None) -> str:  # 定义本例内部辅助函数，组织重复步骤。
    mode = str(raw or "resp").strip().lower()  # 设置该配置块使用的计算模式。
    if mode in {"native", "opls", "oplsaa"}:  # 根据当前状态决定是否进入该分支。
        return "opls"  # 返回该辅助函数的结果。
    return "resp"  # 返回该辅助函数的结果。


def _load_ready_resp_from_moldb(  # 定义本例内部辅助函数，组织重复步骤。
    ff: OPLSAA,
    smiles: str,
    *,
    label: str,
    polyelectrolyte_mode: bool = False,  # 设置中间变量或可调参数，供后续工作流使用。
    repo_db_dir: Path,
    charge_mode: str = "resp",  # 设置中间变量或可调参数，供后续工作流使用。
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
            assign_charge = "opls" if str(charge_mode).strip().lower() == "opls" else None
            mol = ff.ff_assign(mol, charge=assign_charge, report=False)  # 分配力场参数并写入分子属性。
            if not mol:  # 根据当前状态决定是否进入该分支。
                raise RuntimeError(f"Cannot assign OPLS-AA parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
            if assign_charge == "opls":  # 根据当前状态决定是否进入该分支。
                print(f"[MolDB] loaded {label} geometry from {db_label} db and switched to built-in OPLS-AA charges")  # 打印关键路径或状态，便于人工检查。
            else:  # 处理前面条件都不满足的情况。
                print(f"[MolDB] loaded {label} with RESP charges from {db_label} db")  # 打印关键路径或状态，便于人工检查。
            return mol  # 返回该辅助函数的结果。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            last_exc = exc  # 设置中间变量或可调参数，供后续工作流使用。

    raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
        f"{label} is expected to be RESP-ready in MolDB for the OPLS-AA workflow."
    ) from last_exc


def _assign_builtin_opls_ion(ff: OPLSAA, smiles: str, *, label: str):  # 定义本例内部辅助函数，组织重复步骤。
    mol = ff.mol(smiles, charge="opls", require_ready=False, prefer_db=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    mol = ff.ff_assign(mol, charge="opls", report=False)  # 分配力场参数并写入分子属性。
    if not mol:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"Cannot assign built-in OPLS-AA ion parameters for {label}.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    print(f"[OPLS-AA] assigned built-in ion parameters for {label}")  # 打印关键路径或状态，便于人工检查。
    return mol  # 返回该辅助函数的结果。


def _load_pf6_with_opls_builtin_charges(*, ion_ff: OPLSAA, repo_db_dir: Path):  # 定义本例内部辅助函数，组织重复步骤。
    gaff_ff = GAFF2_mod()  # 设置中间变量或可调参数，供后续工作流使用。
    last_exc: Exception | None = None  # 设置中间变量或可调参数，供后续工作流使用。
    opls_probe = ion_ff.mol(PF6_smiles, charge="opls", require_ready=False, prefer_db=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    if not ion_ff.assign_ptypes(opls_probe, charge="opls"):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Cannot build the PF6 OPLS-AA atom-type probe from SMILES.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    for db_dir, db_label in ((None, "default"), (repo_db_dir, "repo")):  # 遍历当前工作流中的一组对象或任务。
        try:  # 尝试执行可能失败的步骤，并在 except 中给出明确错误。
            pf6 = gaff_ff.mol_rdkit(  # 设置中间变量或可调参数，供后续工作流使用。
                PF6_smiles,
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
                raise RuntimeError("PF6 probe atom count does not match the MolDB topology.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

            for src_atom, dst_atom in zip(opls_probe.GetAtoms(), pf6.GetAtoms()):  # 遍历当前工作流中的一组对象或任务。
                if src_atom.GetSymbol() != dst_atom.GetSymbol():  # 根据当前状态决定是否进入该分支。
                    raise RuntimeError("PF6 probe atom ordering does not match the MolDB topology.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
                dst_atom.SetProp("ff_btype", src_atom.GetProp("ff_btype"))
                dst_atom.SetProp("ff_type", src_atom.GetProp("ff_type"))
                dst_atom.SetDoubleProp("ff_sigma", src_atom.GetDoubleProp("ff_sigma"))
                dst_atom.SetDoubleProp("ff_epsilon", src_atom.GetDoubleProp("ff_epsilon"))
                dst_atom.SetDoubleProp("AtomicCharge", src_atom.GetDoubleProp("AtomicCharge"))
                if src_atom.HasProp("ff_desc"):  # 根据当前状态决定是否进入该分支。
                    dst_atom.SetProp("ff_desc", src_atom.GetProp("ff_desc"))

            pf6.SetProp("ff_name", str(ion_ff.name))
            pf6.SetProp("ff_class", str(ion_ff.ff_class))
            pf6.SetProp("pair_style", str(ion_ff.pair_style))
            print(  # 打印关键路径或状态，便于人工检查。
                "[OPLS-AA] loaded PF6 bonded topology from "
                f"{db_label} db and replaced atom types / charges with built-in OPLS-AA values"
            )
            return pf6  # 返回该辅助函数的结果。
        except Exception as exc:  # 捕获异常并转成更清楚的示例错误信息。
            last_exc = exc  # 设置中间变量或可调参数，供后续工作流使用。

    raise RuntimeError(  # 关键步骤失败时立即报错，避免继续生成错误结果。
        "PF6 is expected to exist in MolDB with bonded='DRIH' for this OPLS-AA example."
    ) from last_exc


# ---------------- user inputs ----------------
restart_status = _env_flag("YADONPY_RESTART", default=False)  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
build_only = _env_flag("YADONPY_BUILD_ONLY", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
export_only = _env_flag("YADONPY_EXPORT_ONLY", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
smoke_mode = _env_flag("YADONPY_SMOKE", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
fast_analysis = _env_flag("YADONPY_FAST_ANALYSIS", default=False)  # 设置中间变量或可调参数，供后续工作流使用。
skip_rdf = _env_flag("YADONPY_SKIP_RDF", default=fast_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
skip_sigma = _env_flag("YADONPY_SKIP_SIGMA", default=fast_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
skip_den_dis = _env_flag("YADONPY_SKIP_DEN_DIS", default=fast_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_stage_cap = _env_int("YADONPY_EQ21_STAGE_CAP", 0)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_dt_ps = _env_float("YADONPY_EQ21_DT_PS", 0.0005)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_pre_nvt_ps = _env_float("YADONPY_EQ21_PRE_NVT_PS", 10.0)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_final_ns = _env_float("YADONPY_EQ21_FINAL_NS", 0.8)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_tmax = _env_float("YADONPY_EQ21_TMAX_K", 1000.0)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_pmax = _env_float("YADONPY_EQ21_PMAX_BAR", 50000.0)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_npt_time_scale = _env_float("YADONPY_EQ21_NPT_TIME_SCALE", 2.0)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_robust = _env_flag("YADONPY_EQ21_ROBUST", default=True)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_em_nsteps = _env_int("YADONPY_EQ21_EM_NSTEPS", 50000)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_em_emtol = _env_float("YADONPY_EQ21_EM_EMTOL", 1000.0)  # 设置中间变量或可调参数，供后续工作流使用。
eq21_em_emstep = _env_float("YADONPY_EQ21_EM_EMSTEP", 0.001)  # 设置中间变量或可调参数，供后续工作流使用。
counts_override = _env_int_list("YADONPY_COUNTS", 8)  # 设置中间变量或可调参数，供后续工作流使用。
prod_ns = _env_float("YADONPY_PROD_NS", 20.0)  # 设置中间变量或可调参数，供后续工作流使用。
gpu_offload_mode = _env_text("YADONPY_GPU_OFFLOAD_MODE", "auto")  # 设置中间变量或可调参数，供后续工作流使用。

set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = OPLSAA()  # 选择有机分子/聚合物/部分无机离子的力场对象。
ion_ff = OPLSAA()  # 选择单原子离子参数来源。
opls_charge_mode = _normalize_opls_charge_mode(os.environ.get("YADONPY_OPLS_CHARGE_MODE"))  # 设置中间变量或可调参数，供后续工作流使用。

# ---- CMC monomers (two connection points '*...*') ----
glucose_smiles = "*OC1OC(CO)C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"  # 设置中间变量或可调参数，供后续工作流使用。
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"  # 设置中间变量或可调参数，供后续工作流使用。

DTD_smiles = "O=S1(=O)OC=CO1"  # 设置中间变量或可调参数，供后续工作流使用。

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

omp_psi4 = _env_int("YADONPY_PSI4_OMP", 20)  # 设置 Psi4/OpenMP 核数。
mem_mb = _env_int("YADONPY_PSI4_MEMORY_MB", 20000)  # 设置量子化学内存 MB。

BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"  # 设置中间变量或可调参数，供后续工作流使用。
_work_dir_override = str(os.environ.get("YADONPY_WORK_DIR", "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
_shared_polymer_root_override = str(os.environ.get("YADONPY_SHARED_POLYMER_ROOT", "")).strip()  # 设置中间变量或可调参数，供后续工作流使用。
work_dir = (  # 设置本例输出目录。
    Path(_work_dir_override).expanduser()
    if _work_dir_override  # 根据当前状态决定是否进入该分支。
    else (BASE_DIR / "work_dir_dtd_oplsaa_moldb")
)
shared_polymer_root = (  # 设置中间变量或可调参数，供后续工作流使用。
    Path(_shared_polymer_root_override).expanduser() if _shared_polymer_root_override else None
)


def _formal_charge(mol) -> int:  # 定义本例内部辅助函数，组织重复步骤。
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))  # 返回该辅助函数的结果。


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
        shutil.rmtree(run_dir, ignore_errors=True)  # 设置中间变量或可调参数，供后续工作流使用。

    cfg = eq.EQ21ProtocolConfig(  # 设置中间变量或可调参数，供后续工作流使用。
        t_max_k=float(eq21_tmax),  # 设置中间变量或可调参数，供后续工作流使用。
        t_anneal_k=float(temp),  # 设置中间变量或可调参数，供后续工作流使用。
        p_max_bar=float(eq21_pmax),  # 设置中间变量或可调参数，供后续工作流使用。
        p_anneal_bar=float(press),  # 设置中间变量或可调参数，供后续工作流使用。
        dt_ps=float(eq21_dt_ps),  # 设置 MD 时间步长，单位 ps。
        pre_nvt_ps=float(eq21_pre_nvt_ps),  # 设置中间变量或可调参数，供后续工作流使用。
        robust=bool(eq21_robust),  # 设置中间变量或可调参数，供后续工作流使用。
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
    first_stage = capped_stages[0]  # 设置中间变量或可调参数，供后续工作流使用。
    if first_stage.kind == "minim":  # 根据当前状态决定是否进入该分支。
        capped_stages[0] = eq.EqStage(  # 设置中间变量或可调参数，供后续工作流使用。
            first_stage.name,
            first_stage.kind,
            type(first_stage.mdp)(  # 开始一个多行函数调用或配置块。
                first_stage.mdp.template,
                {
                    **first_stage.mdp.params,
                    "nsteps": int(eq21_em_nsteps),
                    "emtol": float(eq21_em_emtol),
                    "emstep": float(eq21_em_emstep),
                },
            ),
        )

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
    if shared_polymer_root is not None:  # 根据当前状态决定是否进入该分支。
        shared_polymer_wd = workdir(shared_polymer_root, restart=restart_status)  # 创建或复用本例工作目录。
        cmc_rw_dir = shared_polymer_wd.child("CMC_rw")  # 设置中间变量或可调参数，供后续工作流使用。
        cmc_term_dir = shared_polymer_wd.child("CMC_term")  # 设置中间变量或可调参数，供后续工作流使用。
    else:  # 处理前面条件都不满足的情况。
        cmc_rw_dir = wd.child("CMC_rw")  # 设置中间变量或可调参数，供后续工作流使用。
        cmc_term_dir = wd.child("CMC_term")  # 设置中间变量或可调参数，供后续工作流使用。
    ac_build_dir = wd.child("00_build_cell")  # 设置中间变量或可调参数，供后续工作流使用。

    # ---------------- build monomers ----------------
    glucose = _load_ready_resp_from_moldb(  # 设置中间变量或可调参数，供后续工作流使用。
        ff,
        glucose_smiles,
        label="glucose",  # 给该选区一个可读标签，便于 manifest 检查。
        repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
        charge_mode=opls_charge_mode,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    glucose_2 = _load_ready_resp_from_moldb(  # 设置中间变量或可调参数，供后续工作流使用。
        ff,
        glucose_2_smiles,
        label="glucose_2",  # 给该选区一个可读标签，便于 manifest 检查。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
        charge_mode=opls_charge_mode,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    glucose_3 = _load_ready_resp_from_moldb(  # 设置中间变量或可调参数，供后续工作流使用。
        ff,
        glucose_3_smiles,
        label="glucose_3",  # 给该选区一个可读标签，便于 manifest 检查。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
        charge_mode=opls_charge_mode,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    glucose_6 = _load_ready_resp_from_moldb(  # 设置中间变量或可调参数，供后续工作流使用。
        ff,
        glucose_6_smiles,
        label="glucose_6",  # 给该选区一个可读标签，便于 manifest 检查。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        repo_db_dir=REPO_DB_DIR,  # 设置中间变量或可调参数，供后续工作流使用。
        charge_mode=opls_charge_mode,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    # termination
    ter1 = utils.mol_from_smiles(ter_smiles)  # 从 SMILES 直接构造 RDKit 分子。
    qm.assign_charges(  # 执行 RESP/ESP 电荷分配。
        ter1,
        charge="RESP",  # 指定电荷来源或电荷计算方式。
        opt=True,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=wd,  # 设置本例输出目录。
        omp=omp_psi4,  # 设置每个 rank 的 OpenMP 线程数。
        memory=mem_mb,  # 设置中间变量或可调参数，供后续工作流使用。
        log_name="ter1_oplsaa",  # 设置中间变量或可调参数，供后续工作流使用。
    )

    dp = poly.calc_n_from_mol_weight(  # 设置或计算聚合度。
        [glucose, glucose_2, glucose_3, glucose_6],
        target_mw,
        ratio=feed_prob,  # 设置共聚组成比例。
        terminal1=ter1,  # 设置中间变量或可调参数，供后续工作流使用。
    )
    print(f"[CMC] estimated DP from target Mw = {dp}")  # 打印关键路径或状态，便于人工检查。

    chain_len = 12 if smoke_mode else 50  # 设置中间变量或可调参数，供后续工作流使用。
    CMC = poly.random_copolymerize_rw(  # 用随机游走生成聚合物链。
        [glucose, glucose_2, glucose_3, glucose_6],
        chain_len,
        ratio=feed_prob,  # 设置共聚组成比例。
        tacticity="atactic",  # 设置聚合物立构。
        work_dir=cmc_rw_dir,  # 设置本例输出目录。
    )
    CMC = poly.terminate_rw(CMC, ter1, work_dir=cmc_term_dir)  # 给聚合物链加端基。
    CMC_charge = "opls" if opls_charge_mode == "opls" else None
    CMC = ff.ff_assign(CMC, charge=CMC_charge, report=False)  # 分配力场参数并写入分子属性。
    if not CMC:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Cannot assign OPLS-AA parameters for CMC.")  # 关键步骤失败时立即报错，避免继续生成错误结果。
    q_poly = _formal_charge(CMC)  # 设置中间变量或可调参数，供后续工作流使用。

    # ---------------- build solvents / additive ----------------
    EC = _load_ready_resp_from_moldb(ff, EC_smiles, label="EC", repo_db_dir=REPO_DB_DIR, charge_mode=opls_charge_mode)  # 设置中间变量或可调参数，供后续工作流使用。
    EMC = _load_ready_resp_from_moldb(ff, EMC_smiles, label="EMC", repo_db_dir=REPO_DB_DIR, charge_mode=opls_charge_mode)  # 设置中间变量或可调参数，供后续工作流使用。
    DEC = _load_ready_resp_from_moldb(ff, DEC_smiles, label="DEC", repo_db_dir=REPO_DB_DIR, charge_mode=opls_charge_mode)  # 设置中间变量或可调参数，供后续工作流使用。
    DTD = _load_ready_resp_from_moldb(ff, DTD_smiles, label="DTD", repo_db_dir=REPO_DB_DIR, charge_mode=opls_charge_mode)  # 设置中间变量或可调参数，供后续工作流使用。

    # ---------------- ions ----------------
    Li = _assign_builtin_opls_ion(ion_ff, Li_smiles, label="Li")  # 设置中间变量或可调参数，供后续工作流使用。
    Na = _assign_builtin_opls_ion(ion_ff, Na_smiles, label="Na")  # 设置中间变量或可调参数，供后续工作流使用。
    PF6 = _load_pf6_with_opls_builtin_charges(ion_ff=ion_ff, repo_db_dir=REPO_DB_DIR)  # 设置中间变量或可调参数，供后续工作流使用。

    # ---------------- compute counts ----------------
    n_cmc = 1 if smoke_mode else 8  # 设置中间变量或可调参数，供后续工作流使用。
    n_na = abs(q_poly) * n_cmc  # 设置中间变量或可调参数，供后续工作流使用。
    if counts_override is not None:  # 根据当前状态决定是否进入该分支。
        counts = list(counts_override)  # 设置各 species 的数量；顺序必须和 species 列表一致。
    elif smoke_mode:  # 继续判断另一个互斥分支。
        counts = [n_cmc, 6, 6, 6, 2, 2, n_na, 1]  # 设置各 species 的数量；顺序必须和 species 列表一致。
    else:  # 处理前面条件都不满足的情况。
        counts = [n_cmc, 40, 50, 20, 10, 10, n_na, 4]  # 设置各 species 的数量；顺序必须和 species 列表一致。
    charge_scale = [0.7, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 1.0]  # 设置电荷缩放系数；1.0 表示全电荷模型。

    print(  # 打印关键路径或状态，便于人工检查。
        f"[FORMULATION] smoke_mode={smoke_mode} q_poly={q_poly} counts={counts}"
        + (" | source=YADONPY_COUNTS" if counts_override is not None else "")  # 设置中间变量或可调参数，供后续工作流使用。
    )

    ac = poly.amorphous_cell(  # 构建无定形混合体系初始盒子。
        [CMC, EC, EMC, DEC, Li, PF6, Na, DTD],
        counts,
        charge_scale=charge_scale,  # 设置电荷缩放系数；1.0 表示全电荷模型。
        polyelectrolyte_mode=True,  # 启用聚电解质处理逻辑。
        density=0.05,  # 设置中间变量或可调参数，供后续工作流使用。
        neutralize=False,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=ac_build_dir,  # 设置本例输出目录。
    )

    if build_only:  # 根据当前状态决定是否进入该分支。
        print(f"[BUILD-ONLY] Finished cell construction at {ac_build_dir}")  # 打印关键路径或状态，便于人工检查。
        return 0  # 返回该辅助函数的结果。

    eqmd = eq.EQ21step(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
    if export_only:  # 根据当前状态决定是否进入该分支。
        exported = eqmd.ensure_system_exported()  # 设置中间变量或可调参数，供后续工作流使用。
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

    ac = eqmd.exec(  # 设置中间变量或可调参数，供后续工作流使用。
        temp=temp,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
        press=press,  # 设置压力 bar；用于 NPT/EQ 阶段。
        mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
        gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
        gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
        eq21_tmax=eq21_tmax,  # 设置中间变量或可调参数，供后续工作流使用。
        eq21_pmax=eq21_pmax,  # 设置中间变量或可调参数，供后续工作流使用。
        eq21_dt_ps=eq21_dt_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        eq21_pre_nvt_ps=eq21_pre_nvt_ps,  # 设置中间变量或可调参数，供后续工作流使用。
        eq21_robust=eq21_robust,  # 设置中间变量或可调参数，供后续工作流使用。
        eq21_npt_time_scale=eq21_npt_time_scale,  # 设置中间变量或可调参数，供后续工作流使用。
        sim_time=eq21_final_ns,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    for _i in range(4):  # 遍历当前工作流中的一组对象或任务。
        if result:  # 根据当前状态决定是否进入该分支。
            break
        eqmd = eq.Additional(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = eqmd.exec(  # 设置中间变量或可调参数，供后续工作流使用。
            temp=temp,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
            press=press,  # 设置压力 bar；用于 NPT/EQ 阶段。
            mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
            omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
            gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
            gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
            dt_ps=eq21_dt_ps,  # 设置 MD 时间步长，单位 ps。
        )
        analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    if not result:  # 根据当前状态决定是否进入该分支。
        print("[WARNING] Did not reach an equilibrium state after EQ21 + Additional cycles.")  # 打印关键路径或状态，便于人工检查。

    npt = eq.NPT(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = npt.exec(  # 设置中间变量或可调参数，供后续工作流使用。
        temp=temp,  # 设置 MD 温度 K；会影响松弛、采样和统计口径。
        press=press,  # 设置压力 bar；用于 NPT/EQ 阶段。
        mpi=mpi,  # 设置 GROMACS MPI/thread-MPI rank 数。
        omp=omp,  # 设置每个 rank 的 OpenMP 线程数。
        gpu=gpu,  # 控制是否使用 GPU；0 表示 CPU-only。
        gpu_id=gpu_id,  # 选择 GPU 设备编号，多卡节点可修改。
        time=float(prod_ns),  # 设置中间变量或可调参数，供后续工作流使用。
        dt_ps=eq21_dt_ps,  # 设置 MD 时间步长，单位 ps。
        bridge_ps=0.0,  # 设置中间变量或可调参数，供后续工作流使用。
        gpu_offload_mode=gpu_offload_mode,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    analy = npt.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    _ = analy.get_all_prop(temp=temp, press=press, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    if not skip_rdf:  # 根据当前状态决定是否进入该分支。
        _ = analy.rdf(center_mol=Li)  # 设置中间变量或可调参数，供后续工作流使用。
    msd = analy.msd()  # 设置中间变量或可调参数，供后续工作流使用。
    if not skip_sigma:  # 根据当前状态决定是否进入该分支。
        _ = analy.sigma(temp_k=temp, msd=msd)  # 设置中间变量或可调参数，供后续工作流使用。
    if not skip_den_dis:  # 根据当前状态决定是否进入该分支。
        _ = analy.den_dis()  # 设置中间变量或可调参数，供后续工作流使用。
    return 0  # 返回该辅助函数的结果。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    raise SystemExit(main())  # 关键步骤失败时立即报错，避免继续生成错误结果。
