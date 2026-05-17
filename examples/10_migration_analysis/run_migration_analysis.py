#!/usr/bin/env python3
from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。

from rdkit import Chem  # 导入本例需要的库或 yadonpy 接口。

from yadonpy import AnalyzeResult, doctor  # 导入本例需要的库或 yadonpy 接口。


# Example 10: standalone migration analysis for an existing YadonPy work directory.
#
# This example intentionally mirrors the Example 02 post-processing style:
#   analy = job.analyze()
#   rdf = analy.rdf(center_mol=...)
#   msd = analy.msd()
#   sigma = analy.sigma(msd=msd, temp_k=...)
#   migration = analy.migration(center_mol=...)
#
# The difference is that here we open an already-finished work directory via
# AnalyzeResult.from_work_dir(...).

WORK_DIR = Path(__file__).resolve().parent / "work_dir"  # 设置中间变量或可调参数，供后续工作流使用。
CENTER_SMILES = "[Li+]"  # 设置中间变量或可调参数，供后续工作流使用。
TEMPERATURE_K = 333.15  # 设置中间变量或可调参数，供后续工作流使用。
EXPERT_MODE = False  # 设置中间变量或可调参数，供后续工作流使用。


def main() -> None:  # 定义本例内部辅助函数，组织重复步骤。
    doctor(print_report=True)  # 检查运行环境并打印依赖/GROMACS/Python 诊断。

    analy = AnalyzeResult.from_work_dir(WORK_DIR)  # 设置中间变量或可调参数，供后续工作流使用。
    center_mol = Chem.MolFromSmiles(CENTER_SMILES)  # 设置中间变量或可调参数，供后续工作流使用。
    if center_mol is None:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError(f"Failed to parse center SMILES: {CENTER_SMILES!r}")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    rdf = analy.rdf(center_mol=center_mol)  # 设置中间变量或可调参数，供后续工作流使用。
    msd = analy.msd()  # 设置中间变量或可调参数，供后续工作流使用。
    sigma = analy.sigma(msd=msd, temp_k=TEMPERATURE_K)  # 设置中间变量或可调参数，供后续工作流使用。
    migration = analy.migration(center_mol=center_mol, expert_mode=EXPERT_MODE)  # 设置中间变量或可调参数，供后续工作流使用。

    migration_summary = migration.get("migration_summary") or {}  # 设置中间变量或可调参数，供后续工作流使用。
    event_counts = migration.get("event_counts") or {}  # 设置中间变量或可调参数，供后续工作流使用。
    residence_summary = migration.get("residence_summary") or {}  # 设置中间变量或可调参数，供后续工作流使用。
    role_markov = migration.get("markov_role_summary") or {}  # 设置中间变量或可调参数，供后续工作流使用。
    site_markov = migration.get("markov_site_summary") or {}  # 设置中间变量或可调参数，供后续工作流使用。

    print("\nMigration analysis complete.")  # 打印关键路径或状态，便于人工检查。
    print(f"Work dir: {WORK_DIR}")  # 打印关键路径或状态，便于人工检查。
    print(f"Center: {migration_summary.get('center_moltype')}")  # 打印关键路径或状态，便于人工检查。
    print(f"Frames: {migration_summary.get('n_frames')}")  # 打印关键路径或状态，便于人工检查。
    print(f"Selected lag: {migration_summary.get('selected_lag_ps')} ps")  # 打印关键路径或状态，便于人工检查。
    print(f"Markov confidence: {migration_summary.get('markov_confidence')}")  # 打印关键路径或状态，便于人工检查。
    print(f"RDF targets: {len([k for k in rdf.keys() if not str(k).startswith('_')])}")  # 打印关键路径或状态，便于人工检查。
    print(f"MSD species: {len([k for k in msd.keys() if not str(k).startswith('_')])}")  # 打印关键路径或状态，便于人工检查。
    print(f"Sigma_NE_upper: {sigma.get('sigma_ne_upper_bound_S_m')}")  # 打印关键路径或状态，便于人工检查。
    print(f"Sigma_EH_total: {sigma.get('sigma_eh_total_S_m')}")  # 打印关键路径或状态，便于人工检查。
    print(f"Role states: {role_markov.get('state_count')}")  # 打印关键路径或状态，便于人工检查。
    print(f"Site states: {site_markov.get('state_count')}")  # 打印关键路径或状态，便于人工检查。
    print("Observed event counts:")  # 打印关键路径或状态，便于人工检查。
    for key, value in sorted(event_counts.items()):  # 遍历当前工作流中的一组对象或任务。
        print(f"  - {key}: {value}")  # 打印关键路径或状态，便于人工检查。
    print("Residence:")  # 打印关键路径或状态，便于人工检查。
    for role, rec in residence_summary.items():  # 遍历当前工作流中的一组对象或任务。
        print(  # 打印关键路径或状态，便于人工检查。
            f"  - {role}: available={rec.get('available')} "
            f"continuous={rec.get('continuous_residence_time_ps')} ps "
            f"intermittent={rec.get('intermittent_residence_time_ps')} ps"
        )
    print(f"Outputs: {migration_summary.get('outputs')}")  # 打印关键路径或状态，便于人工检查。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    main()
