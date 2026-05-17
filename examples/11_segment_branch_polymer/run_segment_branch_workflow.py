from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

import os  # 导入本例需要的库或 yadonpy 接口。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from yadonpy.core import poly, utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2_mod import GAFF2_mod  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.io import write_gmx  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.io.mol2 import write_mol2  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.preset import eq  # 导入本例需要的库或 yadonpy 接口。


def _env_bool(name: str, default: bool) -> bool:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None:  # 根据当前状态决定是否进入该分支。
        return bool(default)  # 返回该辅助函数的结果。
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}  # 返回该辅助函数的结果。


def _env_float(name: str, default: float) -> float:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None or not str(raw).strip():  # 根据当前状态决定是否进入该分支。
        return float(default)  # 返回该辅助函数的结果。
    return float(raw)  # 返回该辅助函数的结果。


def _env_int(name: str, default: int) -> int:  # 定义环境变量读取辅助函数，方便命令行覆盖默认参数。
    raw = os.environ.get(name)  # 设置中间变量或可调参数，供后续工作流使用。
    if raw is None or not str(raw).strip():  # 根据当前状态决定是否进入该分支。
        return int(default)  # 返回该辅助函数的结果。
    return int(raw)  # 返回该辅助函数的结果。


BASE_DIR = Path(__file__).resolve().parent  # 定位当前示例脚本所在目录。
restart_status = _env_bool("RESTART_STATUS", False)  # 控制断点续跑；True 复用已有输出，False 重新执行相关步骤。
run_md = _env_bool("RUN_MD", True)  # 设置中间变量或可调参数，供后续工作流使用。

work_root = Path(os.environ.get("WORK_DIR", str(BASE_DIR / "work_segment_branch"))).resolve()  # 设置中间变量或可调参数，供后续工作流使用。

mpi = _env_int("MPI", 1)  # 设置 GROMACS MPI/thread-MPI rank 数。
omp = _env_int("OMP", 8)  # 设置每个 rank 的 OpenMP 线程数。
gpu = _env_int("GPU", 0)  # 控制是否使用 GPU；0 表示 CPU-only。
gpu_id = _env_int("GPU_ID", 0)  # 选择 GPU 设备编号，多卡节点可修改。
temp_k = _env_float("TEMP_K", 300.0)  # 设置中间变量或可调参数，供后续工作流使用。
press_bar = _env_float("PRESS_BAR", 1.0)  # 设置中间变量或可调参数，供后续工作流使用。
prod_ns = _env_float("PROD_NS", 0.2)  # 设置中间变量或可调参数，供后续工作流使用。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。
    ensure_initialized()  # 初始化 yadonpy 数据目录和 MolDB。
    set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

    wd = workdir(work_root, restart=restart_status)  # 创建或复用本例工作目录。
    ff = GAFF2_mod()  # 选择有机分子/聚合物/部分无机离子的力场对象。

    # Main-chain label convention:
    #   * / [1*]  -> consumed during linear segment/polymer growth
    #   [2*]      -> preserved by seg_gen and later used as a branch site
    monomer_a = ff.ff_assign(ff.mol("*CCO*", require_ready=False, prefer_db=False), report=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    monomer_b = ff.ff_assign(ff.mol("*COC*", require_ready=False, prefer_db=False), report=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    branchable = ff.ff_assign(ff.mol("*C([2*])C*", require_ready=False, prefer_db=False), report=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    branch_unit = ff.ff_assign(ff.mol("*CO*", require_ready=False, prefer_db=False), report=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    solvent = ff.ff_assign(ff.mol("CCOC(=O)OC", require_ready=False, prefer_db=False), report=False)  # 从 SMILES 或 MolDB 构造带电荷信息的分子对象。
    terminator = utils.mol_from_smiles("[H][*]")  # 从 SMILES 直接构造 RDKit 分子。

    if not all([monomer_a, monomer_b, branchable, branch_unit, solvent]):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Could not assign force-field parameters to one or more example species.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    segment_aab = poly.seg_gen(  # 设置中间变量或可调参数，供后续工作流使用。
        [monomer_a, monomer_a, monomer_b],
        name="segment_aab",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=wd.child("01_segment_aab"),  # 设置本例输出目录。
    )
    branchable_segment = poly.seg_gen(  # 设置中间变量或可调参数，供后续工作流使用。
        [branchable, branchable, monomer_a],
        name="branchable_segment",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=wd.child("02_branchable_segment"),  # 设置本例输出目录。
    )

    # Cap the tail so this side-chain segment has exactly one attach linker.
    side_segment = poly.seg_gen(  # 设置中间变量或可调参数，供后续工作流使用。
        [branch_unit],
        cap_tail="[H][*]",  # 设置中间变量或可调参数，供后续工作流使用。
        name="side_segment",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=wd.child("03_side_segment"),  # 设置本例输出目录。
    )

    # Pre-branch one deterministic site while preserving the other [2*] sites
    # for later post-polymerization grafting.
    prebranched_segment = poly.branch_segment_rw(  # 设置中间变量或可调参数，供后续工作流使用。
        branchable_segment,
        [side_segment],
        mode="pre",  # 设置该配置块使用的计算模式。
        position=2,  # 设置中间变量或可调参数，供后续工作流使用。
        exact_map={"position": 2, "site_index": 0, "branch": 0},  # 设置中间变量或可调参数，供后续工作流使用。
        name="prebranched_segment",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=wd.child("04_prebranch"),  # 设置本例输出目录。
    )

    # Long block construction from segments; this is intentionally a light
    # wrapper around the existing random-walk engine.
    block_polymer = poly.block_segment_rw(  # 设置中间变量或可调参数，供后续工作流使用。
        [segment_aab, prebranched_segment],
        [3, 2],
        name="segment_block_polymer",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=wd.child("05_block_polymer"),  # 设置本例输出目录。
    )

    # Post-branch remaining [2*] sites statistically.  Use ds=[1.0] to consume
    # all available [2*] sites with side_segment in this demonstrator.
    branched_polymer = poly.branch_segment_rw(  # 设置中间变量或可调参数，供后续工作流使用。
        block_polymer,
        [side_segment],
        mode="post",  # 设置该配置块使用的计算模式。
        position=2,  # 设置中间变量或可调参数，供后续工作流使用。
        ds=[1.0],  # 设置中间变量或可调参数，供后续工作流使用。
        name="segment_branched_polymer",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=wd.child("06_postbranch"),  # 设置本例输出目录。
    )
    branched_polymer = poly.terminate_rw(  # 给聚合物链加端基。
        branched_polymer,
        terminator,
        name="segment_branched_polymer",  # 设置对象/层/任务名称，通常会写入输出目录或 manifest。
        work_dir=wd.child("07_terminate"),  # 设置本例输出目录。
    )

    branched_polymer = ff.ff_assign(branched_polymer, report=False)  # 分配力场参数并写入分子属性。
    if not branched_polymer:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Can not assign force-field parameters for the branched segment polymer.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    write_mol2(mol=branched_polymer, out_dir=wd / "00_molecules")  # 设置中间变量或可调参数，供后续工作流使用。
    write_gmx(mol=branched_polymer, out_dir=wd / "90_polymer_gmx")  # 设置中间变量或可调参数，供后续工作流使用。

    ac = poly.amorphous_cell(  # 构建无定形混合体系初始盒子。
        [branched_polymer, solvent],
        [1, 12],
        density=0.05,  # 设置中间变量或可调参数，供后续工作流使用。
        work_dir=wd.child("08_build_cell"),  # 设置本例输出目录。
        retry=3,  # 设置中间变量或可调参数，供后续工作流使用。
        retry_step=500,  # 设置中间变量或可调参数，供后续工作流使用。
    )

    if not run_md:  # 根据当前状态决定是否进入该分支。
        print("[OK] Segment/branch polymer and amorphous cell were built. Set RUN_MD=1 for EQ/production.")  # 打印关键路径或状态，便于人工检查。
        raise SystemExit(0)  # 关键步骤失败时立即报错，避免继续生成错误结果。

    eqmd = eq.EQ21step(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = eqmd.exec(temp=temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
    analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    analy.get_all_prop(temp=temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    for _ in range(4):  # 遍历当前工作流中的一组对象或任务。
        if result:  # 根据当前状态决定是否进入该分支。
            break
        eqmd = eq.Additional(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
        ac = eqmd.exec(temp=temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
        analy = eqmd.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        analy.get_all_prop(temp=temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
        result = analy.check_eq()  # 设置中间变量或可调参数，供后续工作流使用。

    if not result:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Equilibration did not converge; inspect analysis/equilibrium.json before production.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    prod = eq.NPT(ac, work_dir=wd)  # 设置中间变量或可调参数，供后续工作流使用。
    ac = prod.exec(temp=temp_k, press=press_bar, time=prod_ns, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)  # 设置中间变量或可调参数，供后续工作流使用。
    analy = prod.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
    analy.get_all_prop(temp=temp_k, press=press_bar, save=True)  # 设置中间变量或可调参数，供后续工作流使用。
    analy.msd()
