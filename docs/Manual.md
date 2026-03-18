## 0.4.30 Updates (2026-02-05)
- Fix: `from yadonpy import qm` import path for batch examples.

# yadonpy 使用手册（v0.4.30）

yadonpy = “YZC 写的、RadonPy 风格的分子动力学工作流”，以 **GROMACS** 作为唯一目标 MD 引擎。
本仓库 vendor 了 RadonPy 的部分代码（BSD-3-Clause），并在此基础上做适配与扩展。


---

## 0.4.27
- Added shared molecule database (MolDB) storing SMILES/PSMILES + initial geometry + charges.
- Added GAFF2.mol / GAFF2.chg_assign helpers and batch CSV examples.

## 0.4.26 更新（2026-02-05）

### 1）新增：Polymer 系统永远启用 Rg 收敛 gate（yzc-gmx-gen 判据）

对于 index 文件中包含 `poly*` 组的体系，`analy.check_eq()` 会**无条件增加**一条 Rg 收敛 gate（不再依赖 Density 是否存在）。

- 判据与绘图移植自 yzc-gmx-gen：趋势平台（slope/rel_std/sma_sd）与 sd_max（RadonPy-like）两套标准，满足任意一套即可判为收敛。
- 会在 `06_analysis/plots/rg_convergence.svg` 输出诊断图（即使最终不收敛也尽量绘制）。
- `06_analysis/equilibrium.json` 会额外写入 `rg_gate` 字段，方便自动化逻辑判定。

---

## 0.4.25 更新（2026-02-05）

### 1）新增：NVT 生产流程（密度锁定到平衡末段平均值）

新增 `eq.NVT` 预设，用法与 `eq.NPT` 类似：

```python
nvt = eq.NVT(ac, work_dir=work_dir)
ac = nvt.exec(
    temp=temp,
    mpi=mpi,
    omp=omp,
    gpu=gpu,
    gpu_id=gpu_id,
    time=sim_time_ns,
    restart=restart_status,
    density_control=True,
    density_frac_last=0.30,
)
```

当 `density_control=True` 时，NVT 会：
- 从上一步平衡的 `md.edr` 中提取密度随时间序列；
- 取末段（默认后 30%）的平均密度作为目标密度；
- 用 `gmx editconf -scale` 对起始结构的盒子和坐标做等比缩放，使密度匹配目标值；
- 然后进行单阶段 NVT 生产模拟。

### 2）新增例子：Example 09（Example 03 的 NVT 版本）

新增目录：`examples/09_polymer_electrolyte_nvt`。



## 0.4.23 更新（2026-02-04）

- **严重 Bug 修复：** 平衡判据/收敛性分析现在会自动“补全定位”缺失的 `md.tpr / md.xtc / md.edr / md.trr` 等文件。
  - 当分析阶段拿到的路径不存在时，会在 `work_dir` 下递归搜索同名/常见文件名，并优先选择最新的那份。
  - 这避免了因为找不到文件导致无法计算收敛性，从而 Additional 轮次一直循环到结束的问题。
- `gmx energy` 的 term 列表探测现在会合并捕获 stdout/stderr，以适配不同 GROMACS 版本/环境下输出流不同导致的解析失败。

## 0.4.22 更新（2026-02-04）

### 1）mdrun -v 屏幕输出：进度改为“单行刷新”

现在在 yadonpy 中运行 `gmx mdrun -v` 时，会像原生 GROMACS 一样仅在**一行**内持续刷新进度，不再每次刷新都额外输出一行。

### 2）Additional 平衡轮次：不重复建盒子/EM/NVT/NPT，只追加最终平衡阶段

当存在已有的平衡结构（比如来自上一轮 `03_eq/04_md/md.gro` 或 `04_eq_additional/round_XX/04_md/md.gro`）时：

- `eq.Additional.exec()` 内会自动打上判定标签 `_skip_rebuild=True`
- 仅执行最终平衡阶段 `04_md`（重复跑 n 次就产生 n 个 round_XX）

### 3）即使不收敛，也输出收敛性曲线

`analy.check_eq()` 现在无论收敛与否，都会 best-effort 生成：

- `analysis/plots/thermo.svg`（以及 split plots）
- `analysis/plots/rg_*.svg`（若可计算）

### 4）修正不同 GROMACS 版本下 thermo 列名差异导致的误判

平衡判据现在对 thermo 列名做 substring 匹配（例如 `Density (kg/m^3)` 也会被识别为 Density）。


## 0.4.21 更新（2026-02-04）

### 1）write_mol2_from_rdkit 新增 `name` 参数（控制导出的 mol2 文件名）

现在你可以这样写：

```python
write_mol2_from_rdkit(mol=copoly, out_dir=work_dir / "00_molecules")
```

默认会（best-effort）使用调用处 `mol=...` 后面的变量名作为文件名 stem，因此会导出：

- `00_molecules/copoly.mol2`

如果你希望用任意名字（不依赖变量名），可以显式指定：

```python
write_mol2_from_rdkit(mol=copoly, out_dir=work_dir / "00_molecules", name="abcd")
```

将导出：

- `00_molecules/abcd.mol2`

说明：`mol_name` 仍保留用于控制 mol2 文件内部的 molecule/residue name；若不指定，则默认等于 `name`。

### 2）所有示例默认资源改为 mpi=1, omp=16

所有 examples 中的默认并行参数从 `mpi = 16, omp = 1` 调整为：

- `mpi = 1`
- `omp = 16`

这样 `gmx mdrun -ntomp` 会与输入的 `omp` 保持一致（也更符合单 GPU 场景的常见设置）。


---

## 0.4.20 更新（2026-02-04）

### 1）修复：Example 01 直接报 `IndentationError`

`yadonpy/gmx/engine.py` 中 `GromacsRunner.list_energy_terms()` 的代码块缩进被破坏，导致 import 时触发 `IndentationError: unexpected indent`。
本版本已修复缩进并保证函数逻辑/异常清理正常。

### 2）变更：所有 EM 相关任务统一使用 `-nb gpu`

在若干工作流中，EM（steep/cg/minim/em）阶段虽然被设置为 `use_gpu=False`（希望保持 bonded/PME/update 在 CPU 上以提升稳定性），
但 `mdrun` 会同时强制 `-nb cpu`。现在统一改为：**只要是 EM 相关任务，默认请求 `-nb gpu`**。

实现方式：为 `GromacsRunner.mdrun()` 新增参数 `nb="gpu"|"cpu"`，用于在不改变其它 offload 选项的情况下独立控制 nonbonded。

示例：

```python
runner.mdrun(
    tpr=...,
    deffnm="minim",
    cwd=...,
    use_gpu=False,   # bonded/PME/update 仍保持 CPU
    nb="gpu",       # 仅 nonbonded 放到 GPU
)
```

---

## 0.2.27 更新（必读）

### 0）严重 Bug 修复：.itp 缺失 [ angles ] / [ dihedrals ]

在 0.2.26 之前的某些流程中，为了保存“未缩放电荷”的缓存拓扑，会用 `Chem.Mol(mol)`
复制 RDKit 分子对象。但 **RDKit 的拷贝不会保留 Python 级别的属性**（例如 `mol.angles`、`mol.dihedrals`），
导致写出的 `.itp` 只包含 `[ bonds ]`，而 **缺失 `[ angles ]` / `[ dihedrals ]`**。
这会让聚合物/小分子在 MD 中出现极不合理的结构行为。

0.2.27 已修复：
- ✅ 新增 `utils.copy_topology_attributes()`，在 RDKit 拷贝后显式保留 `angles/dihedrals/...`。
- ✅ 读取 basic_top 缓存时会检测 `.itp` 是否缺失 `[ angles ]`（对 >=3 原子的分子），若缺失则自动判为无效并重新生成。

> **提示：** 如果你本地 `~/.local/share/yadonpy/basic_top/...` 里已有 0.2.26 生成的错误缓存，
> 建议直接删掉对应目录（或让 yadonpy 重新生成）。

---

## 0.2.24 更新（必读）

本版本主要针对 **高对称性无机阴离子（PF6-/BF4-/ClO4-/AsF6- 等）** 的几何/力场鲁棒性修复，并对齐 RadonPy 的默认泛函与后处理绘图能力。

### 1）QM 默认泛函全面对齐 RadonPy：`wb97m-d3bj`

- **所有物种一致**：优化 / 单点 / 电荷计算默认都使用 `wb97m-d3bj`。
- 你仍可以在脚本中显式覆盖 `opt_method/charge_method/...`。

### 2）高对称性无机阴离子：加入几何对称化（AX4 / AX6）

- 对自动识别为 AX4 / AX6 的无机离子，在 QM 前后会执行一次 **几何对称化**（仅移动配位原子，保持中心原子不动）。
- 目的：避免数值噪声导致的微小畸变在后续步骤（Hessian/FF/MD）中被放大。

### 3）修复：FF 分配覆盖（m）Seminario 键/角势 的流程问题

- 之前的缓存恢复与后续 `ff_assign()` 之间存在信息丢失，导致 **Seminario 键/角势没有被正确注入最终 .itp**，从而出现 PF6- 这类离子“软掉”并在平衡结构中畸变。
- 0.2.24 里：
  - ✅ 对阴离子 polyion，Seminario Hessian 默认 **跟随单点**（sp_basis/sp_basis_gen），进一步提升刚性与对称性。
  - 会把 Seminario 生成的 `.itp` / `.json` 路径写入 `atomic_charges.json` 的 meta 中，并在 resume 时恢复。
  - `.itp` 注入时对 angle key 采用 **(min(i,k), j, max(i,k))** 规范化匹配，避免 i-k 顺序导致的漏补丁。

### 4）阴离子 polyion：Hessian 默认改为更强基组（与单点一致）

- 对 **带负电的 polyion**，(m)Seminario 的 Hessian 计算默认使用与单点/电荷一致的基组（`charge_basis`/`charge_basis_gen`），以提升键角刚性与对称性保持。

### 5）新增：后处理绘图（默认输出 SVG）

- 新增 `yadonpy.plotting` 与 `yadonpy.gmx.analysis.plot_*` 系列函数。
- 默认输出 **SVG**（矢量图，便于论文/报告直接使用）。

---

## 0.2.21 更新（必读）

### 1）mdrun 进度输出（GROMACS 2022+ 更安静）

GROMACS 2022 之后默认的终端输出更“安静”，很多情况下你只会看到开头/结尾，缺少中间的进度信息。

YadonPy 现在会在 `gmx mdrun` 中默认启用 `-v`，从而让屏幕按合理频率输出与进度相关的信息（例如：
`starting mdrun ...`、`steps/ps`、性能统计等）。

如需关闭（例如你希望完全静默，仅写入 md.log），可在调用 `runner.mdrun(...)` 时传入 `loud=False`。

### 1）生产阶段轨迹输出频率：**1 ps/帧**

- NPT 生产段默认写 xtc（`nstxout-compressed`）为 **1 ps/帧**（依据 `dt` 自动换算）。
- 屏幕/日志输出仍保持每 **10000 steps** 一次，避免过度 IO。

### 2）MSD 分析增强兼容性（GROMACS 2025/2022+）

- `gmx msd` 在新版本中不再支持 `-rmcomm` 选项，yadonpy 改为 **自动探测**：仅当 `gmx msd -h` 中存在该选项时才加入。
- yadonpy 会自动设置 `-trestart`，默认基准为 **20 ps**，并在可解析 `md.mdp` 时按帧间隔自动加大：
  `trestart = max(20 ps, 10 × frame_interval)`，避免 `-dt cannot be larger than -trestart` 报错。

---

## 0.2.9 更新（必读）

本版本集中对齐 **yzc-gmx-gen** 的工程化长处，同时保留 yadonpy 原有的 workflow 优点。

### 1）全面取消 Parrinello–Rahman（P-R）控压：统一改为 C-rescale

- YadonPy 生成的 `.mdp` **禁止输出** `pcoupl = Parrinello-Rahman`（或 P-R / PR 等写法）。
- 若用户在参数覆盖中写入 PR，也会在最终写入 `.mdp` 前被自动替换为：

```ini
pcoupl = C-rescale
```

这与 GROMACS 2021+ 的更稳健 NPT 建议一致（在高离子/低密度初始盒子更稳定）。
默认 `tau_p = 2.0 ps`，并建议按需在 `1–10 ps` 范围内调整。

### 2）每个 MD stage 结束后自动做 PBC “分子连续化”处理（防止“看起来断键/错键”）

你反馈的现象（例如跑完 NVT 后看到 `H 连了两根键`）**通常不是 GROMACS 真断键**，而是：

- 分子跨越周期性边界 (PBC wrap)，导致可视化/转换工具（尤其是 OpenBabel 之类会“按距离猜键”的）误判；
- 或者轨迹未处理 `-pbc mol` 时，单个分子被切成两半。

因此 v0.2.9 在以下 workflow（EQ/NPT/Tg/Elongation/QuickRelax）里，**每个 stage** 结束后都会 best-effort 执行：

```bash
gmx trjconv -pbc mol -center -ur compact
```

并将结果 **覆盖写回** 到该 stage 的 `md.gro`（以及存在时的 `md.xtc`）。这样做的目标是：

- 让后续 `gro/xtc -> mol2/xyz` 的转换更可靠
- 让 VMD/OVITO 等看起来“分子连续”，避免误读

### 3）work_dir 输出结构规范化：模块输出分目录 + Psi4 文件进专用 psi4/ 子目录

work_dir 不再堆满零散文件。典型结构：

- `00_molecules/`：示例脚本导出的分子 mol2
- `01_qm/`：QM/RESP/构象/（新增）bond+angle 参数
  - 其中 Psi4 产生的所有文件都落在每个任务目录下的 `psi4/`
- `02_polymer/`、`03_pack/`、`04_gmx/`：聚合、装箱、GROMACS 的阶段性产物

### 4）新增：modified Seminario（mseminario）从 Psi4 Hessian 生成 bond+angle 参数（仅 bond+angle）

新增 API：`yadonpy.sim.qm.bond_angle_params_mseminario()`

```python
from yadonpy.sim import qm

out = qm.bond_angle_params_mseminario(
    mol,
    confId=0,
    opt=True,
    work_dir=work_dir,
    log_name="anion_A",
    total_charge=-1,
    total_multiplicity=1,
)

print(out["itp"])   # .../01_qm/07_bonded_params/anion_A/bond_angle_params.itp
print(out["json"])  # .../bond_angle_params.json
```

说明：
- 该功能只生成 **[ bonds ] + [ angles ]**（不含 dihedral / improper）。
- 输出为一个可 include 的 `.itp` 片段，便于你后续整合进自定义 force field。

### 5）聚合物随机游走拼接更稳健：默认启用 rollback shaking + 失败步的局部优化

你反馈“聚合物初始构型极不合理”的情况，最常见原因不是“键接错了”，而是 **random-walk 过程中某些 step 的局部构型过挤/扭曲**，在后续 pack/NVT 时会被迅速拉扯，导致：

- 体系能量异常高；
- OpenBabel 等“按距离猜键”的工具更容易误判（看起来像断键/多键）。

因此 v0.2.9 将 `core.poly.polymerize_rw()` / `core.poly.random_copolymerize_rw()` 的默认参数调整为：

- `rollback_shaking=True`
- `retry_opt_step=20`

含义：当某一步结构不合格时，不仅回滚，还会对回滚段做轻微“抖动”，并在失败步触发一次局部优化，从而显著提高大体系链的几何合理性（代价是极端困难体系可能稍慢）。

---

## 0.1.40 更新（历史）

本版本主要解决 **GROMACS 调用逻辑不合理/效率偏低** 的问题（参考 yzc_gmx_gen 的成熟实现），并修复
EQ21 多阶段任务在被中断后的 **断点续跑** 行为。

### 1）mdrun 默认启用 GPU 计算加速参数（高性能默认）

当你在脚本中设置：

```python
gpu = 1
gpu_id = 3  # 例：使用第 3 块 GPU
```

YadonPy 在调用 `gmx mdrun` 时会默认加入（若你的 GROMACS 支持这些选项）：

```bash
-v -pin auto \
  -nb gpu -bonded gpu -update gpu -pme gpu -pmefft gpu \
  -gpu_id <...> -g md.log
```

这与 yzc_gmx_gen 的常用加速写法一致，通常在 GPU 节点上能显著提升吞吐。

### 2）更稳健的 GPU 选择与隔离（避免探测其它 GPU 导致 OOM）

如果你提供了 `gpu_id`（并且环境变量 `CUDA_VISIBLE_DEVICES` 尚未由调度器设置），YadonPy 会自动：

- 设置 `CUDA_VISIBLE_DEVICES=<gpu_id>`
- 同时把 `-gpu_id` 重映射为 `0..N-1`

这样可以显著降低 GROMACS “探测/初始化其它 GPU” 导致的显存冲突与 OOM 风险。

### 3）断点续跑修复：中断后可以真正从 checkpoint 继续

对 EQ21 等多阶段任务：若某个 stage 目录下存在 `md.tpr + md.cpt`，则会使用标准方式继续：

```bash
gmx mdrun -s md.tpr -deffnm md -cpi md.cpt -append ...
```

从而避免“中断后重新 grompp 并从头跑”的低效/不可靠行为。

### 4）gmx 可执行文件的选择逻辑更合理（默认优先 gmx，而不是 gmx_mpi）

很多集群同时提供 `gmx` 与 `gmx_mpi`：

- `gmx`（thread-MPI）常用于 `-ntmpi/-ntomp` 方式控制并行（这也是 YadonPy 的默认并行语义）
- `gmx_mpi` 往往需要 `mpirun/srun` 才能真正启用多 MPI rank

因此本版本默认优先使用 `gmx`。如果你确实要用其它可执行文件，可显式设置：

```bash
export YADONPY_GMX_CMD=gmx_mpi
```

---






## 0.1.20 更新（必读）

本版本引入 **通用 ResumeManager（续算管理器）**，并将所有示例脚本重写为“可一键续算”的形式。

- 新增 `yadonpy.workflow.ResumeManager` 与 `StepSpec`：
  - 在 workflow 目录写入 `resume_state.json`，记录每一步的 inputs_hash 和 outputs。
  - 再次运行脚本时：如果输出文件齐全且 state 记录匹配（或非 strict 模式），会自动 `[SKIP]` 这一步。
  - 可通过环境变量关闭续算：`export YADONPY_NO_RESUME=1`。

- EQ21step 接入 ResumeManager：
  - `export_system`（生成 00_system）与 `equilibration_eq21`（01_equilibration）都带 step-level 续算标记。
  - 即使被中断，重新运行会自动从未完成的 stage/步骤继续。

- 分析模块增加“专用输出/marker 文件”，便于断点续算：
  - `analysis/thermo_summary.json`
  - `analysis/rdf_first_shell.json`
  - `analysis/msd.json`
  - `analysis/sigma.json`
  - `analysis/number_density_profile.json`
  - `analysis/equilibrium.json`

- examples 全部更新为一致的操作逻辑：
  - 统一使用 ResumeManager 管理每个步骤
  - `01_full_workflow_smiles` 的单体 pSMILES 保持为 `*CCO*`
  - PF6- 示例力场保持为 `gaff2_mod`


## 0.1.19 更新（必读）

本版本主要增强 **续算/断点重启** 能力，并更新示例体系的离子力场选择：

- 新增可续算的电荷缓存：`sim.qm.assign_charges()` 在输出 `charged_mol2_qm/<name>.mol2` 的同时，会额外写出
  `charged_mol2_qm/<name>.charges.json`（逐原子电荷列表 + 元信息）。
  - 你可以用 `yadonpy.sim.qm.load_atomic_charges_json(mol, path)` 把电荷重新加载到 RDKit Mol（写入 atom double-prop `AtomicCharge`）。

- 示例脚本 `examples/01_full_workflow_smiles/run_full_workflow.py` 支持断点续算：
  - 若发现 `work_dir/amorphous_cell.sdf` 已存在，会跳过单体 RESP、聚合、Pack，直接从 SDF 载入体系继续向下运行。
  - 若发现 `charged_mol2_qm/<name>.charges.json` 或旧版 `charged_mol2_qm/<name>.mol2` 已存在，会跳过对应分子的 RESP 计算。
  - EQ21step 内部的多阶段 GROMACS 任务本身已支持逐阶段续算（检测到某个子目录已有 `md.gro/summary.json` 则自动跳过）。

- 六氟磷酸根（PF6-）示例力场调整为 `gaff2_mod`（此前示例使用 merz）。


## Ion 3D templates (PF6-, BF4-, ClO4-, TFSI-, FSI-, DFOB-)

在 RDKit 2020.x 等较旧版本中，某些无机/高电荷阴离子（例如 PF6-、BF4-、ClO4-）从 SMILES 生成 3D 坐标可能失败。YadonPy 在 `mol_from_smiles()` 内置了这些常见离子的**结构模板**作为兜底：

- PF6-：近似八面体（P–F）
- BF4-：近似四面体（B–F）
- ClO4-：近似四面体（Cl–O）
- TFSI- / FSI-：给出合理的初始构型（S–N–S 框架 + S=O）
- DFOB-：近似平面环 + 四面体 B（B–F）

模板仅用于 RDKit embedding 不稳定/失败时（PF6-/BF4-/ClO4- 会优先使用模板），以保证后续 `ensure_basic_top()`、pack、MD 工作流不中断。


## 0.1.18 更新（必读）

本版本修复了 RDKit MM 预优化阶段在部分分子上返回 `None` 导致的崩溃，并增强小分子/端基的鲁棒性：

- 修复 `core.calc.conformation_search()` 在 MMFF 力场不可用时可能触发的：
  `AttributeError: 'NoneType' object has no attribute 'Minimize'`。
  - 原因：`AllChem.MMFFGetMoleculeForceField()` 在某些体系（端基、单原子/无键分子、缺参数等）会返回 `None`。
  - 现在改为：MMFF → UFF → 保持构型（能量记 0） 的多级 fallback，保证流程不断。

- 对“极小分子”（例如单原子 terminator）自动将 `nconf` 限制为 1，并关闭 DFT 阶段（`dft_nconf=0`），避免无意义的 1000 conformers。


## 0.1.17 更新（必读）

本版本修复了示例脚本在 QM/RESP 预处理阶段的一个参数冲突错误，并增强了兼容性：

- 修复 `core.calc.conformation_search()` / `_conf_search_psi4_worker()` 在构造 `QMw/Psi4w` 时可能触发的：
  `TypeError: QMw() got multiple values for keyword argument 'omp'`。
  - 原因：用户脚本把 **MD 的 `omp/mpi` 参数**误传进 QM helper，且 `core.calc` 同时又显式传入 `omp=psi4_omp`。
  - 现在会自动清理/转换误传入的 `omp/mpi/gpu/...` 等非 QM 参数：
    - 若用户只传了 `omp` 而没传 `psi4_omp`，将把 `omp` 解释为 `psi4_omp`。
    - 若同时传了 `omp` 与 `psi4_omp`，以 `psi4_omp` 为准，并忽略 `omp`。
    - `mpi` 等 MD 参数会被忽略（不会再传给 Psi4）。

- `sim.qm.conformation_search()` 增加兼容处理：自动忽略 `mpi`，并按上述规则处理 `omp`。

- 示例 `examples/01_full_workflow_smiles/run_full_workflow.py` 更新：
  - `try_resp()` 调用 `qm.conformation_search()` 时不再传入 `mpi/omp`（这些参数用于后续 GROMACS，QM 阶段不需要）。
  - 第一个聚合物单体的 pSMILES 已按要求改为 `*CCO*`。

---

## 0.1.16 更新（必读）

本版本修复了 v0.1.15 示例脚本在 `doctor()` 阶段的一个属性错误：

- 修复 `yadonpy.diagnostics.doctor()` 访问 `DataLayout.gmx_forcefields_dir` 时的
  `AttributeError`。
  - 现在 `DataLayout` 明确提供 `gmx_forcefields_dir`，默认指向：
    `$YADONPY_DATA_DIR/ff/gmx_forcefields`。
  - `ensure_initialized()` 会创建该目录（如果不存在）。

> 注：v0.1.15 的更新内容仍然适用，见下节。

---

## 0.1.15 更新（必读）

本版本主要是 **修补不完整实现**、提升失败可诊断性，并修复了示例脚本在 Python 3.9 下的一个
`dataclasses` 初始化错误。

**行为变化与修复点：**

- 修复 `AnalyzeResult` 的 `@dataclass` 字段顺序错误（Python 3.9 会报：
  `TypeError: non-default argument 'edr' follows default argument`）。
- 分子工件（`.itp/.gro/.top`）生成改为 **失败即报错**：不再静默吞异常；失败时会在输出目录写入
  `gromacs_error.txt`（含 traceback）并在 `meta.json` 记录 `gromacs_error`，方便快速定位。
- 当请求 QM/RESP 等电荷方法但外部依赖不可用或计算失败时，会自动 **fallback 到 `gasteiger`**
  电荷，并给出明确提示（避免“无电荷继续跑”导致后续拓扑异常）。
- `system.ndx` 组名兼容性增强：同时生成 `<moltype>` 与 `MOL_<moltype>` 两种别名，避免脚本
  侧按旧约定找不到 group。
- 系统导出时的物种元信息增加/统一 `moltype` 字段，并改进了从 `cell_meta` fallback 重建物种时的
  错误提示（更易定位是哪一个 species 的 SMILES/参数化失败）。

---

## 1. 设计理念

- **无命令行（CLI）**：所有任务都用 Python 脚本组织（RadonPy 的 sample_script 风格）。
- **以 SMILES 为唯一识别符**：
  - 不依赖名字去匹配文件夹。
  - 同一分子只要 SMILES 相同，就认为是同一“物种”（species）。
- **自动库管理（面向后续 RESP/DFT 扩展）**：
  - 非聚合物分子（SMILES 不含 `*`）第一次成功参数化后会自动加入默认库。
  - 新加入的分子会标记 `is_original_from_lib=false`，便于后续一键清理可能失败的外部计算产物。

---

## 2. 安装与依赖

> 本节的 conda 安装风格尽量与 RadonPy 保持一致：先创建环境，再分别安装 **RDKit** 与 **Psi4/resp**。

### 2.1 Python 版本
- Python >= 3.9

### 2.2 推荐的 conda/mamba 环境（RadonPy 风格）

下面给出一个与 RadonPy 风格一致的推荐安装方式：先用 conda 建环境，再在源码目录 `pip install -e .`。

你可以直接保存为 `environment.yml`：

```yaml
name: yadonpy
channels:
  - conda-forge
dependencies:
  - python>=3.9
  - rdkit>=2020.03
  - mdtraj>=1.9
  - matplotlib
  - pint>0.24
  - ipykernel
  - numpy
  - pandas
  - scipy
  - psi4
  - openbabel
  - dftd3-python
  - resp
  - pip:
    - pybel
    - minepy
    - -e .
```

安装步骤：

```bash
conda env create -f environment.yml
conda activate yadonpy

# 在项目根目录
pip install -e .
```

> 说明：`psi4/resp` 只在你需要 **RESP** 或其它 QM 电荷方法时必需。

> 说明：`openbabel/pybel` 不是必须依赖，但强烈推荐用于 **显式带电的 (p)SMILES**：
> yadonpy 会先用 OpenBabel + 内置 UFF 松弛生成初始 3D 构象，然后交给 Psi4 做 OPT + RESP。

### 2.3 依赖总览

**必需：**
- RDKit
- numpy / scipy / pandas / matplotlib

**OpenBabel（建议）：**
- openbabel（conda）
- pybel（pip）

**RESP/QM（可选）：**
- psi4
- resp
- dftd3-python（可选，但推荐）

如果你不想用 `environment.yml`，也可以手动用 conda 安装上述依赖，再 `pip install -e .`。

### 2.3 必需依赖
- **RDKit（必须）**：SMILES 解析、分子/聚合物构建、拓扑生成。

### 2.4 QM / RESP 相关依赖（可选但常用）
当 `charge_method` 选择 `RESP/ESP/Mulliken/Lowdin` 时需要：
- **Psi4**
- **resp**（用于 RESP 拟合）

如果你只想先跑通流程，可使用：
- `charge_method='gasteiger'`（快速、无需 Psi4）
- `charge_method='zero'`（适用于部分离子/测试）

#### 2.4.1 推荐的 RESP 默认级别（v0.2.11 起）
yadonpy 的 RESP 默认不再使用 HF：

- **RESP/ESP 单点能（用于导出 ESP → RESP 拟合）**：默认 `wB97X-D3BJ / def2-TZVP`
- **阴离子（净电荷 < 0）**：会自动优先选用 **带弥散** 的 def2/ma-def2 基组（例如 `def2-TZVPD`、`ma-def2-TZVPPD`）

你仍可以在代码中显式传入 `charge_method/charge_basis` 覆盖默认。

#### 2.4.2 无机多原子离子（PF6- 等）的自动参数策略
对 PF6- / BF4- / ClO4- 等 **高对称、刚性、无机多原子离子**，yadonpy 会在 `auto_level=True` 时自动采取更稳健的 QM 分层：

- OPT：优先 `wB97X-D3BJ / ma-def2-SVPD`（带弥散，小基组，鲁棒）
- RESP 单点：优先 `wB97X-D3BJ / ma-def2-TZVPPD`（更大基组，更稳定的 ESP）

并且在 `bonded_params='auto'` 时，会优先用 **modified Seminario（mseminario）** 从 Psi4 Hessian
导出 **bond + angle** 的力常数，并在写出 `.itp` 时自动覆盖 GAFF 家族默认的 bond/angle k 值。

> 说明：mseminario 仅用于 bond/angle；二面角仍保留 GAFF2_mod 的默认。

### 2.5 GROMACS（唯一 MD 引擎）
要运行平衡/拉伸/Tg 等 MD 工作流，你的系统需要可调用：
- `gmx`（或 `gmx_mpi`）

并建议把可执行文件加入 `PATH`。

### 2.6 可选依赖
- **mpi4py**：若你在 Psi4 计算中希望用 MPI/多进程调度。



## 3. 快速开始

仓库自带示例脚本（v0.3.6）：

### 3.1 Example 01：聚合物溶液（无盐）一条龙
- 路径：`examples/01_polymer_solution/run_full_workflow.py`
- 内容：
  - 两个双连接点单体（`*...*`）→ 各自构象搜索 + RESP → 随机共聚（RW）→ 端基封端
  - 两种溶剂（无 `*`）→ 构象搜索 + RESP → GAFF2-family 参数
  - `poly.amorphous_cell([...], counts, density, charge_scale=...)` 构盒
  - `eq.EQ21step` 平衡（可选断点续算）→ `eq.NPT` 生产段 → 热力学/结构性质（MSD 等）

运行（在项目根目录）：
```bash
python examples/01_polymer_solution/run_full_workflow.py
```

### 3.2 Example 02：PF6- 单分子 basic_top（不构盒）
- 路径：`examples/02_Li_salt/run_pf6_basic_top.py`
- 目标：只验证 **无机多原子阴离子** 的 3D 几何、RESP、电荷对称化、以及 `.itp/.top/.gro` 输出。
- 特点：
  - 用 OpenBabel 生成初始 3D 坐标（避免 RDKit 对 PF6- 这类体系的构象/几何不稳定）
  - `qm.assign_charges(..., opt=True, symmetrize=True, auto_level=True, bonded_params='auto')`
    会在可用时导出 QM-Hessian 并用 modified Seminario 自动生成 bond/angle 参数，写出 `.itp` 时自动覆盖 GAFF-family 的默认 bond/angle。
  - **脚本在写出单分子拓扑后停止**（不进行 pack/MD）。

运行：
```bash
python examples/02_Li_salt/run_pf6_basic_top.py
```

### 3.3 Example 03：聚合物电解质（含盐）一条龙
- 路径：`examples/03_polymer_electrolyte/run_full_workflow.py`
- 内容：在 Example 01 的基础上，增加盐离子并进入完整 MD：
  - `cation_A=[Li+]` 使用 MERZ 参数（`MERZ().mol()` + `MERZ().ff_assign()`）
  - `anion_A=PF6-` 采用 OpenBabel 3D → OPT+RESP →（可选）mseminario bonded patch → GAFF2-family 非键
  - 之后同样 `amorphous_cell` 构盒 → `EQ21step` 平衡 → `NPT` 生产段
  - 后处理包含：RDF（以 Li+ 为中心）、MSD、离子电导率估算（`sigma`）等

运行：
```bash
python examples/03_polymer_electrolyte/run_full_workflow.py
```

> 备注：v0.3.6 修复了 `write_gromacs_single_molecule_topology()` 在 bonded patch 阶段的一个语法错误（缺失 except/finally），该问题会导致 Example 03 在导出系统时直接报 `SyntaxError: expected 'except' or 'finally' block`。

---

## 4. 电荷缩放与 charged MOL2 输出

### 4.1 charge_scale（介电缩放）

yadonpy 支持对 **聚合物 / 溶剂 / 盐离子** 的原子电荷进行 **按物种分别缩放**，用于模拟介电
屏蔽效应（charge scaling）。该功能在 **系统导出阶段** 生效：不修改持久化的 basic_top 库，
而是在运行目录中为每个 moltype 生成一份缩放后的 `.itp` 并用于 GROMACS 模拟。

**推荐用法：在构盒阶段按物种顺序给出缩放列表**（最直接、最不易匹配失败）：

```python
ac = poly.amorphous_cell(
    [copoly, solvent_A, solvent_B, ion_A, ion_B],
    [4, 20, 20, 20, 20],
    density=0.05,
    charge_scale=[1.0, 1.0, 1.0, 0.8, 0.8],  # 与 mols 列表一一对应
)
eqmd = eq.EQ21step(ac, work_dir=..., ff_name='gaff2_mod', charge_method='RESP')
eqmd.exec(temp=300, press=1.0, mpi=16, omp=1, gpu=0, sim_time=5)
```

**高级用法：在 `EQ21step.exec` 中覆盖缩放**（便于扫描参数）。此时 `charge_scale` 可以是：

- `float`：全物种统一缩放
- `dict`：按 SMILES/moltype/类别匹配缩放（更灵活）

```python
charge_scale = {
    "polymer": 1.0,
    "solvent": 1.0,
    "ion": 0.8,
    "[Li+]": 0.8,
    "F[P-](F)(F)(F)(F)F": 0.8,
}
eqmd.exec(temp=300, press=1.0, mpi=16, omp=1, gpu=0, sim_time=5, charge_scale=charge_scale)
```

> 默认不显式指定时，`charge_scale=1.0`（不缩放）。
>
> `charge_scale` 可用的 key 规则（优先级从高到低）：
> 1) 物种 canonical SMILES（建议）
> 2) 导出到 system.top 的 moltype 名称
> 3) 类别：`cation` / `anion` / `ion` / `polymer` / `solvent` / `neutral`
> 4) `all` / `default`

### 4.2 charged_mol2 目录

当系统被导出到 `work_dir/00_system` 后，yadonpy 会生成：

- `work_dir/00_system/charged_mol2/<moltype>.mol2`：**本次模拟实际使用** 的电荷（已应用 charge_scale）
- `work_dir/00_system/charged_mol2/<moltype>.raw.mol2`：库文件（basic_top）中的 **原始电荷**

对于首次参数化的新分子，在对应的 basic_top 目录下也会写出：

- `<basic_top>/<ff>/<mol_id>/charged_mol2/<mol_id>.mol2`

这些文件主要用于：
- 检查 RESP 电荷是否合理
- 在外部软件中可视化电荷分布
- 与其它工具链互操作（例如后续的外部 DFT/RESP 扩展）

该脚本展示：
- 以 SMILES 参数化溶剂/盐（非聚合物）并自动写入默认库
- 参数化聚合物单体（带 `*`）
- 构建随机共聚物链 + 封端
- 生成 amorphous cell（混合溶剂/盐体系）
- **自动导出** `system.gro/system.top/system.ndx`（只依赖 SMILES 匹配）
- EQ21step（多阶段平衡，GROMACS）
- 后处理：RDF（按 atomtype 逐个算）、MSD、离子电导率、数密度分布

---

## 5. 默认库与 library.json

### 5.1 默认库位置
yadonpy 会初始化一个“数据目录”，默认在：
- `~/.local/share/yadonpy/`

其中会包含：
- `basic_top/...`（预置与缓存的 GROMACS 工件：`.itp/.top/.gro`）
- `ff/library.json`（SMILES -> 工件目录索引）

### 5.1.1 yadonpy 的“只用 SMILES 识别”逻辑（核心）

1. 读取 SMILES
2. 先在 `ff/library.json` 中以 **SMILES** 查找对应记录
3. 若记录存在且其 `artifact_dir` 下存在 `.itp/.top/.gro`：直接使用
4. 若找不到：执行 **力场分配 +（可选）RESP 电荷计算**，然后自动生成 `.itp/.top/.gro` 写入 `basic_top/`，并更新 `ff/library.json`

### 5.2 library.json 的关键字段
每条物种记录包含（常用）：
- `mol_id`：库内部 id（内置分子通常是目录名；新生成分子通常是 SMILES 哈希前缀）
- `smiles`：唯一识别符
- `artifact_dir`：工件目录（建议存相对路径，例如 `basic_top/gaff/molecules/EC`）
- `smiles`：唯一识别符
- `is_polymer_monomer`：是否为聚合物单体（来自 system.csv / 或基于 `*` 判断）
- `is_original_from_lib`：是否为软件自带库中原始分子
  - `true`：自带库
  - `false`：后期外部计算/自动生成加入的分子，可用于统一清理

---

## 5. GROMACS 工作流（presets）

### 5.0 EQ21step（RadonPy 风格入口）

从 v0.1.9 起，yadonpy 提供了 RadonPy 风格的封装：

```python
from yadonpy.sim.preset import eq

ac = poly.amorphous_cell([...], [...], density=0.05)
eqmd = eq.EQ21step(ac, work_dir=work_dir, ff_name='gaff2_mod')
eqmd.exec(temp=300, press=1.0, mpi=16, omp=1, gpu=0, sim_time=5)  # ns

an = eqmd.analyze()
an.get_all_prop(temp=300, press=1.0, save=True)
an.rdf(..., center_mol=ion, include_h=False)
an.msd(...)
an.sigma(temp_k=300)
an.density_distribution(...)
```

关键点：

- `poly.amorphous_cell(...)` 会在返回的 cell 对象中写入 `_yadonpy_cell_meta`（species smiles + 数量 + density）。
- `EQ21step` 会读取这个 meta，**先按 SMILES 在内置 basic_top 库里找**：
  - 找到就直接拿 `.itp/.gro` 用；
  - 找不到就参数化并生成工件；若是非聚合物分子（SMILES 不含 `*`），会自动写入默认库并打 `is_original_from_lib=false`。
- 最终会在 `work_dir/00_system/` 下生成：
  - `system.gro` / `system.top` / `system.ndx`

### 5.1 yadonpy 的后处理接口（保留 yzc-gmx-gen 风格）

#### 5.1.2 绘图（默认输出 SVG）

yadonpy 提供了一组轻量的 SVG 绘图工具，主要面向 GROMACS 输出（`.xvg`）与简单的 x-y 数据：

```python
from pathlib import Path
import numpy as np

from yadonpy.gmx.analysis.plot import plot_xvg_svg, plot_xvg_split_svg, plot_xy_svg

# 1) 直接把一个 .xvg（可能有多列）画成一张图
plot_xvg_svg(Path("work_quick/thermo.xvg"), out_svg=Path("work_quick/thermo.svg"))

# 2) 也可以把每一列拆成一张图
plot_xvg_split_svg(Path("work_quick/thermo.xvg"), out_dir=Path("work_quick"))

# 3) 画任意 x-y 数据（例如 density vs T）
T = np.array([500, 480, 460])
rho = np.array([1200, 1210, 1225])
plot_xy_svg(T, rho, out_svg=Path("work_tg/density_vs_T.svg"), title="Density vs T", xlabel="T (K)", ylabel="Density (kg/m^3)")
```

特点：

- **默认 SVG**（矢量图，论文/报告可直接用）
- headless-safe（matplotlib Agg backend）
- 自动读取 `.xvg` 里的 axis label / legend（若存在）

##### 自动绘图（v0.3.0 起默认开启）

从 **v0.3.0** 开始，yadonpy 的主要工作流与后处理会像 `yzc-gmx-gen` 一样：
**每个阶段自动生成 SVG 图**，并写入各自的 `plots/` 目录（不会影响主流程；失败会被记录在 summary）。

典型输出位置：

- **QuickRelaxJob / EquilibrationJob**
  - `<stage>/thermo.xvg`
  - `<stage>/plots/thermo.svg`（全部曲线）
  - `<stage>/plots/thermo__*.svg`（按列拆分）

- **TgJob**
  - `T??_xxxK/plots/density_time.svg`
  - `plots/tg_density_vs_T.svg`

- **ElongationJob**
  - `stress_strain.svg`

- **Analyzer（analysis/）**
  - `analysis/msd/plots/msd_*.svg`、`msd_overlay*.svg`
  - `analysis/conductivity/eh_fit_*.svg`
  - `analysis/rdf_first_shell/plots/rdf_cn_*.svg`
  - `analysis/number_density_distribution/plots/ndens_*.svg`

如果你想关闭自动绘图：在对应 Job 的构造函数里传 `auto_plot=False`。

#### RDF（按 atomtype 逐个绘制）

默认 **不统计氢** 的 RDF（因为很多力场里氢的 type 太碎、噪声大），但保留接口：

```python
an.rdf(species_mols, center_mol=ion, include_h=True)
```

RDF 结果会自动估计第一溶剂化鞘（峰值 + 峰后第一极小值），并把该极小值处的 CN 写入：

- `work_dir/analysis/summary.json` 的 `rdf_first_shell` 字段。



#### 离子电导率（Ionic conductivity）

`sigma()` 会同时给出两种估算：

- **Nernst–Einstein（NE）**：基于 `msd()` 拟合得到的离子扩散系数与拓扑中的净电荷数，计算电导率并分解到各离子组分。
- **Einstein–Helfand（EH）**：调用 `gmx current -dsp` 得到 EH 曲线，并自动选择线性区间做线性拟合（斜率即静态电导率）。

注意：

- EH 需要**带速度的轨迹**（推荐 `.trr`）。因此 yadonpy 的默认 mdp 会写出 `md.trr`（通过 `nstxout/nstvout`）。
- 如果工作目录中找不到 `.trr`，则会自动跳过 EH，仅返回 NE，并在结果里写明原因。

结果会写入：

- `work_dir/analysis/summary.json` 的 `sigma` 字段（包含 `ne` 与 `eh` 两部分）。

#### 数密度分布（Number density profiles）

会在 `work_dir/analysis/number_density_distribution/` 下输出每个 moltype 在 X/Y/Z 的数密度曲线（横坐标为 distance）。


从 **v0.1.4** 起，yadonpy 开始把 RadonPy 的常用工作流迁移到 **GROMACS**，并保持“脚本驱动”的风格。
在 **v0.1.5** 中补强了“从 `.edr` 直接做波动性质/应力相关后处理”的能力。

所有工作流都要求你已经有 GROMACS 可跑的输入（至少 `system.gro` + `system.top`）。

### 5.1 Quick relax（替代 quick-min / quick-md）

```python
from yadonpy.gmx.workflows import QuickRelaxJob

job = QuickRelaxJob(
    gro=Path("system.gro"),
    top=Path("system.top"),
    out_dir=Path("work_quick"),
    do_quick_md=True,
    quick_md_ns=0.05,
)
job.run(restart=True)
```

输出：
- `work_quick/minim.*`（能量最小化）
- `work_quick/quick.*`（短 NVT）
- `work_quick/summary.json`（包含 Temperature/Pressure/Density/Volume 的统计）

### 5.2 Multi-stage equilibration（多阶段平衡）

```python
from yadonpy.gmx.workflows import EquilibrationJob

stages = EquilibrationJob.default_stages(
    temperature_k=298.15,
    pressure_bar=1.0,
    nvt_ns=0.2,
    npt_ns=0.5,
    prod_ns=1.0,
)
job = EquilibrationJob(
    gro=Path("system.gro"),
    top=Path("system.top"),
    out_dir=Path("work_eq"),
    stages=stages,
)
job.run(restart=True)
```

每个 stage 都是一个子目录（例如 `01_minim/`, `02_nvt/`），并写入 stage 级 `summary.json`。

**EM（能量最小化）鲁棒性策略（v0.2.10+，对齐 yzc-gmx-gen）：**

在 `01_minim/` 内部会顺序运行三段 EM（同一个 stage 目录，避免 work_dir 变乱）：

1) `md_steep`：`integrator=steep` + `constraints=none`（对初始重叠最稳）
2) `md_steep_hbonds`：`integrator=steep` + `constraints=h-bonds`（把约束“温和收回来”）
3) `md`：`integrator=cg`（可选收敛；若因 LINCS/约束失败，自动回退为 steep/h-bonds 并继续流程）

这样可以避免你遇到的报错：`Minimizer 'cg' can not handle constraint failures`。

从 `md.edr` 提取的后处理（best-effort，取最后 `frac_last` 做系综平均）：
- Temperature / Pressure / Density / Volume / Energy（`gmx energy`）
- 体积涨落得到等温压缩率 `kappa_t_1_pa` 与体积模量 `bulk_modulus_gpa`
- 若能量项中包含 `Enthalpy`，则给出 `cp_molar_j_mol_k`（以 `Cp = var(H)/(R T^2)` 计算）

### 5.3 Tg 扫描（自动两段线性切分拟合）

```python
from yadonpy.gmx.workflows import TgJob

job = TgJob(
    gro=Path("system.gro"),
    top=Path("system.top"),
    out_dir=Path("work_tg"),
    temperatures_k=[500, 475, 450, 425, 400, 375, 350, 325, 300],
    npt_ns=2.0,
    frac_last=0.5,
)
job.run(restart=True)
```

拟合方法：
- 以 `Density(T)` 曲线为输入
- 遍历所有可能的切分点（保证两侧至少 2 个点）
- 对低温段/高温段分别做最小二乘直线拟合
- 以总 SSE 最小为最优切分
- Tg 为两条线的交点

输出：
- `work_tg/density_vs_T.csv`
- `work_tg/summary.json`（包含 Tg、切分点、两段拟合参数）

### 5.4 拉伸（uniaxial deform）

yadonpy 用 GROMACS 的 `deform`（mdp 里的 `deform = ...`）实现单轴形变。

输出：
- `work_elong/stress_strain.csv`
- `work_elong/summary.json`

注意：能量文件中的压力分量是 **bar**。yadonpy 输出常用符号约定的
`sigma_xx = -Pres-XX`（单位 GPa），并额外给出
`sigma_dev = -(Pres-XX - (Pres-YY+Pres-ZZ)/2)`。

---

## 6. 力场与单位说明（重要）

RadonPy 的 GAFF 系列 JSON 数据库常用长度单位是 **Å**（angstrom），能量单位是 **kcal/mol**。
GROMACS 的 `.top`/`.itp` 常用：
- 长度：**nm**
- 能量：**kJ/mol**

yadonpy 在生成 GROMACS 输入时会做单位转换（见 `yadonpy.io.gromacs_convert` / `yadonpy.io.gromacs_top`）。

---

## 7. yadonpy 中“只保留 GROMACS”的取舍

为了保证依赖简单、避免双引擎维护成本，yadonpy **移除了所有 LAMMPS 后端**。

### 7.1 已迁移 / 仍未迁移的功能

已迁移到 GROMACS（v0.1.4+）：
- 多阶段平衡（`EquilibrationJob`）
- Tg 扫描 + 自动切分拟合（`TgJob`）
- 单轴拉伸（`ElongationJob`）
- quick-min / quick-md（`QuickRelaxJob`）
- 热力学量/压力张量等从 `.edr` 抽取并做系综平均（`gmx energy` + XVG 解析）
- 典型波动性质（示例）：等温压缩率 κT、体积模量 K（可在 NPT 阶段输出）

仍未迁移（如你后续需要，可以继续加）：
- `sp`（更完整的力学/应力相关性质集合）
- `ef_dp`（外场/偶极响应类）
- 其他 RadonPy 中高度 LAMMPS 绑定的 preset

明确移除：
- 热导率（`tc`）——你之前已明确不需要

---

## 8. 自检（debug/doctor）

在示例脚本中你会看到：
```python
from yadonpy.diagnostics import doctor
doctor(print_report=True)
```

它会检查：
- RDKit 是否可用
- Psi4/resp 是否可用（如果你要跑 RESP）
- PATH 中是否有 gmx（如果你要跑 GROMACS）

---

## 9. 常见问题

### 9.1 没装 Psi4/resp，但我只想先跑通流程
把 `charge_method` 改成：
- `gasteiger`：快速、普适
- `zero`：离子（例如 `[Li+]`）可用

### 9.2 聚合物单体怎么判断？
- SMILES 中 **带 `*`**：当作“单体/连接单元（monomer）”
- 不带 `*`：当作普通分子（溶剂/盐），会自动写入默认库

### 9.3 Tg 拟合是什么算法？
v0.1.4 使用 **自动切分的两段线性拟合**（见 5.3）。你只需要给出温度点序列即可。

### 9.4 为什么 minim 阶段会出现 LINCS WARNING / `cg` 直接崩？

你看到的典型报错是：

- `LINCS WARNING ... The coordinates could not be constrained`
- `Minimizer 'cg' can not handle constraint failures, use minimizer 'steep' before using 'cg'.`

这通常意味着：**初始 pack 的结构过“粗糙”**（局部重叠/挤压太严重），导致在开启约束时
（`constraints=h-bonds`）某些键无法被 LINCS 收敛。

v0.2.10 起，yadonpy 的 `EquilibrationJob/QuickRelaxJob` 采用 yzc-gmx-gen 风格的三段 EM：

1) steep + none（抗重叠）
2) steep + h-bonds（抗约束崩溃）
3) cg（可选，失败会自动回退为 steep/h-bonds）

如果你仍然遇到约束崩溃，优先尝试：

- 把 pack 更“松”一点（降低目标密度、增加盒子尺寸）
- 增大 `md_steep`/`md_steep_hbonds` 的 `nsteps`，或进一步减小 `emstep`
- 先用 `QuickRelaxJob` 跑一段短 NVT（小 dt）再进入完整 EQ

### 9.5 生成的 .itp 里只有 [ bonds ]，没有 [ angles ]/[ dihedrals ] 怎么办？

这会导致体系“软成一团”，属于**严重错误**。

从 **v0.3.2** 起，yadonpy 会：

1) 在力场分配后输出一行简要总结（用于你核对是否真的分配到了角/二面角）：

`FF assign summary (gaff2): atoms=... bonds=... angles=... dihedrals=... impropers=...`

2) 在使用 basic_top 缓存时自动检查 `.itp` 是否包含 `[ angles ]/[ dihedrals ]`（对 >=3 / >=4 原子的分子），
   如果缺失会**自动强制重新生成**并覆盖缓存。

如果你仍看到缺失：

- 先确认终端里 `FF assign summary` 的 angles/dihedrals 计数不是 0
- 再删掉本地缓存目录（默认：`~/.local/share/yadonpy/basic_top/...`）后重新运行




## Notes: bonded terms and topology export

- In workflows that build mixtures/cells (e.g. **Example 01 polymer solution**), YadonPy may internally copy RDKit molecules during packing and polymer assembly.
- Starting from **v0.3.7**, `deepcopy_mol()` preserves YadonPy's Python-side bonded-term attributes (`angles/dihedrals/impropers/cmaps`) when copying molecules, preventing `.itp` outputs that contain only `[ bonds ]` but miss `[ angles ]` / `[ dihedrals ]`.

