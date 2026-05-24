from __future__ import annotations  # 启用未来注解语法，减少类型注解带来的运行时负担。

# YadonPy example annotation:
# - 这些示例脚本同时承担教程作用，所以注释会比库代码更详细。
# - 优先修改文件顶部的 user inputs / 参数区；后面的注释说明每个参数的物理意义和可调方向。
# - 带有 MolDB/RESP/DRIH/GROMACS 的行通常不要随意删除，除非你明确知道该阶段的替代流程。

"""Example 08-07: shared-t0 CMC-Na/electrolyte slabs, then charge sweep.

This example inverts the usual layer-stack sizing logic.  It first prepares a
wall-confined CMC-Na slab with ``pbc=xy`` and no z periodicity, reads the relaxed
slab XY box, chooses graphite basal-plane repeat counts that match that XY
footprint, prepares an electrolyte slab at the same XY footprint with the same
z-open wall protocol, and only then assembles graphite | CMC-Na | electrolyte |
graphite.  The intended production protocol is to build one shared, neutral
``t=0`` geometry and then derive charge cases from exactly that coordinate set:
the CMC-facing graphite inner face receives ``0, -3, -9, -18 uC/cm2`` while the
opposite graphite inner face receives the equal-and-opposite charge.  Temporary
phase gates are kept during pre-release relaxation and removed from final NVT,
so the first final NVT frame is the electrolyte/CMC interdiffusion t=0.
"""

import json  # 写入 charge_case_done 和 shared-t0 诊断 JSON。
import math  # 用于把 CMC slab 的连续 XY 尺寸转成石墨晶格可兼容的 nx/ny。
import os  # 远端四卡 sweep 通过环境变量覆盖 GPU、电荷和输出根目录。
import shutil  # shared-t0 charge sweep 只复制一次中性体系，再 patch 电荷。
import time  # 多进程同时启动时用简单目录锁等待 shared stack 构建完成。
from dataclasses import replace  # 从中性 stack 派生 charged stack，不重新手写所有参数。
from pathlib import Path  # 导入本例需要的库或 yadonpy 接口。
from typing import Any  # 本例的 topology patch helper 用到少量 JSON/manifest payload。

from rdkit import Chem  # 从已有 shared-t0 文件恢复 LayerStackResult 时需要一个占位 Mol。

from yadonpy import (  # 导入本例需要的库或 yadonpy 接口。
    CMCNAXYSlabRelaxationSpec,
    XYSlabEquilibrationSpec,
    clean_md_trajectory_files,
    prepare_cmcna_xy_membrane,
    retarget_prepared_slab_xy,
)
from yadonpy.core import poly, utils, workdir  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.data_dir import ensure_initialized  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.core.graphite import build_graphite  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.diagnostics import doctor  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.gaff2_mod import GAFF2_mod  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.ff.merz import MERZ  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.interface import (  # 导入本例需要的库或 yadonpy 接口。
    FixedChargeRegionSpec,
    GraphiteLayerSpec,
    GraphiteRestraintSpec,
    InterdiffusionStartSpec,
    LayerStackRelaxationSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    ZCompressionAnnealSpec,
    analyze_layer_stack_interface,
    build_layer_stack,
    make_orthorhombic_pack_cell,
    run_layer_stack_relaxation,
)
from yadonpy.interface.layer_stack import LayerStackResult  # 由 shared-t0 文件构造轻量 result，避免每个电荷重建 stack。
from yadonpy.io.gromacs_system import SystemExportResult  # 由已存在的 system.gro/top/ndx 构造轻量 export result。
from yadonpy.runtime import set_run_options  # 导入本例需要的库或 yadonpy 接口。
from yadonpy.sim.preset import eq  # 导入 EQ21/NPT preset；这里用于 CMC-Na 的 xy-slab 预平衡。


UCM2_TO_E_PER_NM2 = 0.06241509074460765  # 1 microC/cm2 对应的 e/nm2；用于 topology 层 patch 电极表面电荷。


def read_gro_xy_nm(gro_path: Path) -> tuple[float, float]:  # 从 GROMACS GRO 末行读取 x/y 盒长，单位 nm。
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()  # 读取 GRO 文本。
    if len(lines) < 3:  # GRO 至少应有标题、原子数和 box 行。
        raise ValueError(f"Invalid GRO file: {gro_path}")  # 文件格式错误时立即报错。
    parts = lines[-1].split()  # box 行通常前三列是 x/y/z。
    if len(parts) < 2:  # 至少要有 x/y。
        raise ValueError(f"Cannot read XY box from GRO file: {gro_path}")  # 给出清晰报错。
    return float(parts[0]), float(parts[1])  # 返回 x_nm, y_nm。


def rdkit_mol_mass_amu(mol) -> float:  # 直接按 RDKit 原子质量求分子质量，适合已经展开成全原子的长链。
    return float(sum(float(atom.GetMass()) for atom in mol.GetAtoms()))  # 返回 amu；避免 polymer helper 忽略重复单元造成低估。


def estimate_electrolyte_counts_for_slab(  # 按固定 XY、active 厚度和目标密度反推电解液数量。
    *,
    species: tuple[Any, Any, Any, Any, Any],  # EC/EMC/DEC/Li/PF6。
    base_solvent_counts: tuple[int, int, int],  # EC/EMC/DEC 参考比例。
    base_salt_pairs: int,  # LiPF6 参考比例。
    xy_nm: tuple[float, float],  # fixed XY footprint。
    active_thickness_nm: float,  # 目标 active slab 厚度，不含 wall padding。
    target_density_g_cm3: float,  # 目标电解液密度。
) -> dict[str, Any]:
    avogadro = 6.02214076e23  # mol^-1。
    base_counts = (*base_solvent_counts, int(base_salt_pairs), int(base_salt_pairs))  # EC/EMC/DEC/Li/PF6 比例单元。
    base_mass_amu = sum(rdkit_mol_mass_amu(mol) * int(count) for mol, count in zip(species, base_counts))  # 一个比例单元质量。
    volume_nm3 = float(xy_nm[0]) * float(xy_nm[1]) * float(active_thickness_nm)  # active slab 体积。
    target_mass_amu = float(target_density_g_cm3) * volume_nm3 * 1.0e-21 * avogadro  # 目标质量，g/cm3 * cm3 -> g -> amu。
    scale = max(1.0e-12, target_mass_amu / max(base_mass_amu, 1.0e-12))  # 按质量缩放比例单元。
    solvent_counts = tuple(max(1, int(round(int(count) * scale))) for count in base_solvent_counts)  # 缩放 EC/EMC/DEC。
    salt_pairs = max(1, int(round(int(base_salt_pairs) * scale)))  # 缩放 LiPF6 对数。
    counts = (*solvent_counts, salt_pairs, salt_pairs)  # 与 species 对齐。
    actual_mass_amu = sum(rdkit_mol_mass_amu(mol) * int(count) for mol, count in zip(species, counts))  # 四舍五入后的真实质量。
    actual_density = (actual_mass_amu / avogadro) / max(volume_nm3 * 1.0e-21, 1.0e-30)  # 四舍五入后的密度。
    atom_count = sum(int(mol.GetNumAtoms()) * int(count) for mol, count in zip(species, counts))  # 电解液原子数估计。
    return {  # 返回完整诊断，写入 stdout 和后续检查。
        "xy_nm": [float(xy_nm[0]), float(xy_nm[1])],
        "active_thickness_nm": float(active_thickness_nm),
        "target_density_g_cm3": float(target_density_g_cm3),
        "scale": float(scale),
        "solvent_counts": list(solvent_counts),
        "salt_pairs": int(salt_pairs),
        "counts": list(counts),
        "estimated_density_g_cm3": float(actual_density),
        "estimated_electrolyte_atoms": int(atom_count),
        "estimated_volume_nm3": float(volume_nm3),
    }


def env_bool(name: str, default: bool) -> bool:  # 从环境变量读取布尔值，远端批量作业可覆盖。
    value = os.environ.get(name)  # 读取环境变量。
    if value is None:  # 未设置时保持脚本默认。
        return bool(default)  # 返回默认值。
    return value.strip().lower() not in {"0", "false", "no", "off"}  # 常见否定值视作 False。


def env_float(name: str, default: float) -> float:  # 从环境变量读取浮点数。
    value = os.environ.get(name)  # 读取环境变量。
    return float(default if value is None or value == "" else value)  # 空值回退默认。


def env_int(name: str, default: int) -> int:  # 从环境变量读取整数。
    value = os.environ.get(name)  # 读取环境变量。
    return int(default if value is None or value == "" else value)  # 空值回退默认。


def charge_case_dirname(charge_uC_cm2: float) -> str:  # 把 CMC-facing 电荷转成稳定 case 目录名。
    if abs(float(charge_uC_cm2)) < 1.0e-12:  # 中性 case 单独命名为 00。
        return "cmcface_00_uC_cm2"  # 返回中性目录。
    sign = "m" if float(charge_uC_cm2) < 0 else "p"  # 负号不能直接进目录名。
    mag = f"{abs(float(charge_uC_cm2)):.1f}".replace(".", "p")  # 3.0 -> 3p0。
    return f"cmcface_{sign}{mag}_uC_cm2"  # 例如 -18.0 -> cmcface_m18p0_uC_cm2。


def charge_region_specs(charge_uC_cm2: float) -> tuple[FixedChargeRegionSpec, FixedChargeRegionSpec]:  # 生成 CMC-facing / opposite 两个石墨内表面的电荷定义。
    return (  # 返回 tuple，便于 neutral shared stack 用空 tuple，charge case 用这个 tuple。
        FixedChargeRegionSpec(  # 上石墨贴着 CMCNA 的内侧表面。
            layer_name="GRAPHITE_TOP",  # 正确层序为 graphite | electrolyte | CMCNA | graphite，所以 CMC-facing electrode 是上石墨。
            region="bottom",  # 上石墨的 bottom face 是 CMC-facing surface。
            mode="surface_charge_density",  # 以 microC/cm2 指定表面电荷密度。
            surface_charge_uC_cm2=float(charge_uC_cm2),  # 目标 CMC-facing 电荷。
            elements=("C",),  # 只给石墨碳原子加固定电荷。
            label="cmc_facing_graphite_inner_face",  # 报告标签。
        ),
        FixedChargeRegionSpec(  # 下石墨内侧表面取等量反号，保持电极对口径。
            layer_name="GRAPHITE_BOTTOM",  # 下石墨层。
            region="top",  # 下石墨的 top face 是 opposite inner face。
            mode="surface_charge_density",  # 仍按表面电荷密度。
            surface_charge_uC_cm2=-float(charge_uC_cm2),  # 等量反号。
            elements=("C",),  # 只给石墨碳原子加固定电荷。
            label="opposite_graphite_inner_face",  # 报告标签。
        ),
    )


def wait_for_shared_stack_lock(lock_dir: Path, *, timeout_s: float = 7200.0):  # 多个 charge case 同时启动时，只有一个进程能构建 shared stack。
    start = time.time()  # 记录等待起点。
    lock_dir = Path(lock_dir)  # 规范成 Path。
    while True:  # 目录锁：mkdir 成功者负责构建，失败者等待。
        try:
            lock_dir.mkdir(parents=True)  # 原子创建目录；成功即获得锁。
            return True  # True 表示当前进程持锁，需要 finally 删除。
        except FileExistsError:  # 其它进程正在构建 shared stack。
            if time.time() - start > float(timeout_s):  # 防止死等。
                raise TimeoutError(f"Timed out waiting for shared stack lock: {lock_dir}")  # 超时给出明确路径。
            time.sleep(5.0)  # 每 5 s 检查一次，避免忙等。


def read_gro_atoms(gro_path: Path) -> tuple[list[dict[str, Any]], tuple[float, float, float]]:  # 读取 GRO 原子记录和 box；用于从 shared t=0 选择石墨表面原子。
    lines = Path(gro_path).read_text(encoding="utf-8", errors="replace").splitlines()  # 读取 GRO。
    if len(lines) < 3:  # GRO 基本格式检查。
        raise ValueError(f"Invalid GRO file: {gro_path}")  # 清晰报错。
    natoms = int(lines[1].strip())  # 第二行是原子数。
    atoms: list[dict[str, Any]] = []  # 每个原子保存 1-based index、名称、坐标。
    for offset in range(natoms):  # 遍历原子行。
        line = lines[2 + offset]  # GRO 原子行。
        atom_name = line[10:15].strip()  # atom name 字段。
        atoms.append(  # 写入解析结果。
            {
                "index": offset + 1,  # GROMACS topology 使用 1-based global atom index。
                "resnr": line[0:5].strip(),  # residue number。
                "resname": line[5:10].strip(),  # residue name。
                "atomname": atom_name,  # atom name。
                "element": (atom_name[:1].upper() if atom_name else ""),  # graphite 只需要识别 C。
                "x": float(line[20:28]),  # x / nm。
                "y": float(line[28:36]),  # y / nm。
                "z": float(line[36:44]),  # z / nm。
            }
        )
    box_parts = [float(tok) for tok in lines[2 + natoms].split()]  # box 行。
    if len(box_parts) < 3:  # 需要正交盒前三列。
        raise ValueError(f"Invalid GRO box line in {gro_path}")  # 清晰报错。
    return atoms, (float(box_parts[0]), float(box_parts[1]), float(box_parts[2]))  # 返回 atoms 和 box。


def select_charge_region_atoms(system_gro: Path, manifest: dict[str, Any], spec: FixedChargeRegionSpec) -> tuple[list[int], tuple[float, float]]:  # 根据 manifest 层区间选择 surface atoms。
    atoms, _box = read_gro_atoms(system_gro)  # 读坐标。
    intervals = [dict(v) for v in manifest.get("layer_intervals_nm", [])]  # manifest 中的层 z 区间。
    matches = [v for v in intervals if str(v.get("name", "")).strip() == str(spec.layer_name).strip()]  # 按层名匹配。
    if len(matches) != 1:  # 必须唯一。
        raise ValueError(f"Cannot uniquely find layer {spec.layer_name!r} in manifest intervals.")  # 清晰报错。
    interval = matches[0]  # 目标层区间。
    layer_lo = float(interval["z_lo_nm"])  # 层下边界。
    layer_hi = float(interval["z_hi_nm"])  # 层上边界。
    layer_atoms = [a for a in atoms if layer_lo - 1.0e-4 <= float(a["z"]) <= layer_hi + 1.0e-4]  # 该层原子。
    if not layer_atoms:  # 没选到说明 manifest/GRO 不一致。
        raise ValueError(f"No atoms found in layer interval for {spec.layer_name!r}.")  # 清晰报错。
    z_values = [float(a["z"]) for a in layer_atoms]  # 该层实际原子 z。
    atom_lo = min(z_values)  # 实际下表面。
    atom_hi = max(z_values)  # 实际上表面。
    region = str(spec.region).strip().lower()  # top/bottom/all/z_range。
    if region == "top":  # 顶表面。
        tol = max(0.008, 0.02 * max(atom_hi - atom_lo, 0.1))  # 与 layer_stack builder 保持一致的薄层选择容差。
        z0, z1 = atom_hi - tol, layer_hi + 1.0e-4  # 顶层窗口。
    elif region == "bottom":  # 底表面。
        tol = max(0.008, 0.02 * max(atom_hi - atom_lo, 0.1))  # 同上。
        z0, z1 = layer_lo - 1.0e-4, atom_lo + tol  # 底层窗口。
    elif region == "all":  # 整层。
        z0, z1 = layer_lo - 1.0e-4, layer_hi + 1.0e-4  # 整层窗口。
    elif region == "z_range":  # 层内自定义 z window。
        if spec.z_min_nm is None or spec.z_max_nm is None:  # z_range 必须给范围。
            raise ValueError("z_range charge region requires z_min_nm and z_max_nm.")  # 清晰报错。
        z0 = layer_lo + float(spec.z_min_nm)  # 层内下界。
        z1 = layer_lo + float(spec.z_max_nm)  # 层内上界。
        if z1 < z0:  # 容忍用户写反。
            z0, z1 = z1, z0  # 交换。
    else:  # 未知 region。
        raise ValueError(f"Unsupported charge region: {spec.region}")  # 清晰报错。
    include = {str(v).strip().upper() for v in spec.elements if str(v).strip()}  # 元素过滤。
    selected = [int(a["index"]) for a in atoms if z0 <= float(a["z"]) <= z1 and (not include or str(a["element"]).upper() in include)]  # 全局原子索引。
    return selected, (float(z0), float(z1))  # 返回全局 atom ids 和 z window。


def parse_topology_molecule_blocks(system_top: Path) -> tuple[dict[str, Path], list[tuple[str, int]]]:  # 解析 system.top 的 include 和 [ molecules ] 顺序。
    include_paths: dict[str, Path] = {}  # moltype -> itp path。
    molecule_rows: list[tuple[str, int]] = []  # [molecules] 中的 moltype/count。
    section = ""  # 当前 section。
    top_dir = Path(system_top).parent  # include 相对 system.top。
    for raw in Path(system_top).read_text(encoding="utf-8", errors="replace").splitlines():  # 遍历 top。
        clean = raw.split(";", 1)[0].strip()  # 去掉注释。
        if clean.startswith("#include"):  # include 行。
            parts = clean.split("\"")  # 只处理 #include "path"。
            if len(parts) >= 2 and parts[1].endswith(".itp"):  # molecule itp 或 ff_parameters。
                path = (top_dir / parts[1]).resolve()  # include 文件绝对路径。
                moltype = parse_itp_moltype(path) if path.is_file() else ""  # 解析 moltype。
                if moltype:  # ff_parameters 没有 moleculetype，会被跳过。
                    include_paths[moltype] = path  # 记录。
            continue  # include 行处理完毕。
        if clean.startswith("[") and clean.endswith("]"):  # section header。
            section = clean.strip("[]").strip().lower()  # 记录 section。
            continue  # 进入下一行。
        if section == "molecules" and clean:  # [ molecules ] 数据。
            tokens = clean.split()  # molname count。
            if len(tokens) >= 2:  # 至少两列。
                molecule_rows.append((tokens[0], int(tokens[1])))  # 保存。
    return include_paths, molecule_rows  # 返回 include map 和 molecule rows。


def parse_itp_moltype(itp_path: Path) -> str:  # 从 ITP 读取 [ moleculetype ] 名。
    section = ""  # 当前 section。
    for raw in Path(itp_path).read_text(encoding="utf-8", errors="replace").splitlines():  # 遍历 ITP。
        clean = raw.split(";", 1)[0].strip()  # 去掉注释。
        if not clean:  # 跳过空行。
            continue  # 继续。
        if clean.startswith("[") and clean.endswith("]"):  # section header。
            section = clean.strip("[]").strip().lower()  # 记录 section。
            continue  # 下一行。
        if section == "moleculetype":  # moleculetype 数据行。
            return clean.split()[0]  # 第一列是 moltype。
    return ""  # 没找到。


def parse_itp_atom_count(itp_path: Path) -> int:  # 统计 ITP [ atoms ] 行数，用于 global->local atom 映射。
    section = ""  # 当前 section。
    count = 0  # atom count。
    for raw in Path(itp_path).read_text(encoding="utf-8", errors="replace").splitlines():  # 遍历 ITP。
        clean = raw.split(";", 1)[0].strip()  # 去注释。
        if not clean:  # 跳过空行。
            continue  # 继续。
        if clean.startswith("[") and clean.endswith("]"):  # section header。
            section = clean.strip("[]").strip().lower()  # 记录。
            continue  # 下一行。
        if section == "atoms" and clean.split()[0].isdigit():  # atoms 数据行。
            count += 1  # 计数。
    return count  # 返回 atom count。


def map_global_atoms_to_itp(system_top: Path, selected_atom_ids: list[int]) -> dict[Path, dict[int, float]]:  # 把全局 atom ids 映射到 ITP local atom ids。
    include_paths, molecule_rows = parse_topology_molecule_blocks(system_top)  # 解析 topology。
    atom_counts = {moltype: parse_itp_atom_count(path) for moltype, path in include_paths.items()}  # 每个 moltype 的 atom count。
    selected = set(int(v) for v in selected_atom_ids)  # 目标全局 atom ids。
    mapped: set[int] = set()  # 实际映射到 topology 的全局 atom ids。
    local_by_itp: dict[Path, dict[int, float]] = {}  # itp -> local atom id -> charge placeholder。
    global_idx = 1  # GROMACS 全局 atom index 从 1 开始。
    for moltype, count in molecule_rows:  # 按 [ molecules ] 顺序展开。
        if moltype not in include_paths or moltype not in atom_counts:  # 缺少 ITP 是拓扑错误。
            raise ValueError(f"Cannot find ITP/atom count for moltype {moltype!r} in {system_top}")  # 清晰报错。
        nat = int(atom_counts[moltype])  # 当前 moltype atom count。
        for mol_inst in range(int(count)):  # 展开每个 molecule instance。
            local_hits: list[int] = []  # 这一 molecule instance 内选中的 local atom。
            for local_idx in range(1, nat + 1):  # 遍历 local atom。
                if global_idx in selected:  # 命中目标 atom。
                    local_hits.append(local_idx)  # 记录 local id。
                    mapped.add(global_idx)  # 记录已映射的全局 id。
                global_idx += 1  # 下一个全局 atom。
            if local_hits and int(count) != 1:  # 对 count>1 的 moltype 局部 patch 会影响所有重复分子，不能静默执行。
                raise ValueError(  # 当前 charge patch 只允许 graphite 这种 count=1 的 layer moltype。
                    f"Charge patch selected moltype {moltype!r} instance {mol_inst + 1}/{count}, "
                    "but the moltype appears multiple times. Build graphite layers as unique moltypes before patching."
                )
            if local_hits:  # count=1 graphite moltype 命中。
                local_by_itp.setdefault(include_paths[moltype], {})  # 初始化。
                for local_idx in local_hits:  # 保存 local atom id，charge 稍后填。
                    local_by_itp[include_paths[moltype]][int(local_idx)] = 0.0  # placeholder。
    missing = selected - mapped  # 检查是否所有目标 atom 都在 topology 展开顺序中找到。
    if missing:  # 实际不会发生。
        raise ValueError(f"Internal atom mapping error for {len(missing)} selected atoms.")  # 清晰报错。
    return local_by_itp  # 返回 local mapping。


def patch_itp_atom_charges(itp_path: Path, local_charges: dict[int, float]) -> None:  # 修改 ITP [ atoms ] charge 列。
    lines = Path(itp_path).read_text(encoding="utf-8", errors="replace").splitlines()  # 读取 ITP。
    section = ""  # 当前 section。
    out: list[str] = []  # 输出行。
    patched = set()  # 已 patch 的 local atom id。
    for raw in lines:  # 遍历原始行。
        clean = raw.split(";", 1)[0].strip()  # 去注释后用于判断。
        if clean.startswith("[") and clean.endswith("]"):  # section header。
            section = clean.strip("[]").strip().lower()  # 记录 section。
            out.append(raw)  # 保留原行。
            continue  # 下一行。
        if section == "atoms" and clean:  # atoms section 数据。
            comment = ""  # 注释部分。
            body = raw  # 数据部分。
            if ";" in raw:  # 保留注释。
                body, comment = raw.split(";", 1)  # 分离。
                comment = ";" + comment  # 加回分号。
            tokens = body.split()  # atoms 行字段。
            if len(tokens) >= 7 and tokens[0].isdigit():  # 至少到 charge 列。
                local_idx = int(tokens[0])  # local atom id。
                if local_idx in local_charges:  # 需要修改。
                    tokens[6] = f"{float(local_charges[local_idx]):.8f}"  # charge 列。
                    patched.add(local_idx)  # 记录。
                    out.append("    " + "  ".join(tokens) + ("  " + comment if comment else ""))  # 写回。
                    continue  # 下一行。
        out.append(raw)  # 非目标行原样写回。
    missing = set(local_charges) - patched  # 检查是否全部 patch。
    if missing:  # 有 local atom 没找到。
        raise ValueError(f"Did not find local atoms {sorted(missing)[:10]} in {itp_path}")  # 清晰报错。
    Path(itp_path).write_text("\n".join(out) + "\n", encoding="utf-8")  # 写回 ITP。


def apply_surface_charge_patch(system_dir: Path, manifest_path: Path, specs: tuple[FixedChargeRegionSpec, ...]) -> list[dict[str, Any]]:  # 对复制出的 shared-t0 topology 施加表面电荷。
    system_dir = Path(system_dir)  # 规范路径。
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))  # 读取 manifest。
    atoms, box_nm = read_gro_atoms(system_dir / "system.gro")  # 读取坐标和盒子。
    area_nm2 = float(box_nm[0]) * float(box_nm[1])  # 电极面积。
    reports: list[dict[str, Any]] = []  # patch 报告。
    for idx, spec in enumerate(specs):  # 两个内表面逐个 patch。
        selected, z_window = select_charge_region_atoms(system_dir / "system.gro", manifest, spec)  # 选择全局 atom ids。
        target_q = float(spec.surface_charge_uC_cm2 or 0.0) * UCM2_TO_E_PER_NM2 * area_nm2  # uC/cm2 -> 总电荷 e。
        q_per_atom = float(target_q) / float(len(selected)) if selected else 0.0  # 均分到选区碳原子。
        local_by_itp = map_global_atoms_to_itp(system_dir / "system.top", selected)  # 映射到 ITP local atoms。
        for itp_path, local_map in local_by_itp.items():  # 写入每个 ITP。
            patch_itp_atom_charges(itp_path, {local_idx: q_per_atom for local_idx in local_map})  # 修改 charge 列。
        reports.append(  # 记录报告。
            {
                "index": int(idx),
                "label": spec.label or f"{spec.layer_name}_{spec.region}_{idx}",
                "layer_name": spec.layer_name,
                "region": spec.region,
                "mode": spec.mode,
                "surface_charge_uC_cm2": float(spec.surface_charge_uC_cm2 or 0.0),
                "target_charge_e": float(target_q),
                "charge_per_atom_e": float(q_per_atom),
                "selected_atom_count": int(len(selected)),
                "z_window_nm": [float(z_window[0]), float(z_window[1])],
                "patched_itp_files": [str(path) for path in local_by_itp],
            }
        )
    manifest["fixed_charge_regions"] = reports  # 覆盖 manifest 中的电荷报告。
    manifest["charge_patch"] = {  # 明确这是 shared neutral topology 派生出的 charge case。
        "method": "shared_t0_topology_charge_patch",
        "source": "neutral_shared_stack",
        "surface_area_nm2": float(area_nm2),
        "regions": reports,
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 写回 case manifest。
    (system_dir / "charge_patch_report.json").write_text(json.dumps(manifest["charge_patch"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")  # 同步写入 system 目录。
    return reports  # 返回报告。


def layer_stack_result_from_existing(  # 从已存在的 system.gro/top/ndx/manifest 构造 LayerStackResult，避免重新导出体系。
    *,
    work_dir_path: Path,
    stack_spec: LayerStackSpec,
    manifest_path: Path,
    acceptance: dict[str, Any] | None = None,
    layer_reports: tuple[dict[str, Any], ...] = (),
) -> LayerStackResult:
    system_dir = Path(work_dir_path) / "02_system"  # 标准 system 目录。
    system_gro = system_dir / "system.gro"  # 坐标。
    system_top = system_dir / "system.top"  # 拓扑。
    system_ndx = system_dir / "system.ndx"  # index。
    if not (system_gro.is_file() and system_top.is_file() and system_ndx.is_file() and Path(manifest_path).is_file()):  # 四个关键文件都必须存在。
        raise FileNotFoundError(f"Existing stack is incomplete under {work_dir_path}")  # 清晰报错。
    _atoms, box_nm = read_gro_atoms(system_gro)  # 读 box。
    export = SystemExportResult(  # 构造轻量 export result。
        system_gro=system_gro,
        system_top=system_top,
        system_ndx=system_ndx,
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=max(float(box_nm[0]), float(box_nm[1]), float(box_nm[2])),
        species=[],
        box_lengths_nm=box_nm,
    )
    return LayerStackResult(  # 返回 run_layer_stack_relaxation 可接受的 result。
        work_dir=Path(work_dir_path),
        stack_spec=stack_spec,
        system_export=export,
        system_gro=system_gro,
        system_top=system_top,
        system_ndx=system_ndx,
        manifest_path=Path(manifest_path),
        stacked_cell=Chem.Mol(),
        layer_reports=layer_reports,
        acceptance=acceptance or {},
        box_nm=box_nm,
    )


def derive_charge_case_from_shared_stack(  # 从中性 shared-t0 派生某个表面电荷 case。
    *,
    shared_result: LayerStackResult,
    case_dir: Path,
    charged_stack: LayerStackSpec,
    charge_specs: tuple[FixedChargeRegionSpec, ...],
) -> LayerStackResult:
    case_dir = Path(case_dir)  # 规范路径。
    case_system_dir = case_dir / "02_system"  # case 自己的 system 目录。
    shutil.rmtree(case_system_dir, ignore_errors=True)  # 清掉旧的错误 molecule-per-residue 导出，防止 stale topology 污染。
    shutil.copytree(Path(shared_result.system_gro).parent, case_system_dir, dirs_exist_ok=True)  # 复制 shared neutral system。
    case_manifest = case_dir / "layer_stack_manifest.json"  # case manifest。
    shutil.copy2(shared_result.manifest_path, case_manifest)  # 复制 shared manifest。
    reports = apply_surface_charge_patch(case_system_dir, case_manifest, charge_specs)  # 在 topology 层 patch 电荷。
    result = layer_stack_result_from_existing(  # 构造可传给 run_layer_stack_relaxation 的 result。
        work_dir_path=case_dir,
        stack_spec=charged_stack,
        manifest_path=case_manifest,
        acceptance={**dict(shared_result.acceptance), "shared_t0_charge_patch": True, "fixed_charge_regions": reports},
        layer_reports=tuple(shared_result.layer_reports),
    )
    (case_dir / "shared_t0_charge_patch_summary.json").write_text(  # 顶层再写一份摘要，便于远端检查。
        json.dumps(
            {
                "source_shared_stack": str(shared_result.work_dir),
                "case_dir": str(case_dir),
                "system_dir": str(case_system_dir),
                "manifest": str(case_manifest),
                "regions": reports,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return result  # 返回 case result。


def graphite_repeats_for_xy(target_xy_nm: tuple[float, float]) -> tuple[int, int, tuple[float, float]]:  # 选择最小的石墨 nx/ny 覆盖目标 XY。
    probe = build_graphite(  # 用 2x2 周期基面 probe 推断单个石墨重复单元的 box 尺寸。
        nx=2,  # 1x1 周期石墨太小，2x2 是当前 builder 的最小安全 probe。
        ny=2,  # 同上。
        n_layers=3,  # 层数不影响 XY 单元长度。
        orientation="basal",  # 基面石墨。
        edge_cap="periodic",  # 周期边界；box_x = nx * a, box_y = ny * b。
        name="GRAPHITE_UNIT_PROBE",  # probe 不进入最终体系。
    )
    unit_x_nm = float(probe.box_nm[0]) / 2.0  # 单个 x 重复单元长度。
    unit_y_nm = float(probe.box_nm[1]) / 2.0  # 单个 y 重复单元长度。
    nx = max(2, int(math.ceil(float(target_xy_nm[0]) / unit_x_nm - 1.0e-6)))  # 覆盖目标 x 的最小 nx，容忍 GRO 五位小数舍入。
    ny = max(2, int(math.ceil(float(target_xy_nm[1]) / unit_y_nm - 1.0e-6)))  # 覆盖目标 y 的最小 ny，容忍 GRO 五位小数舍入。
    actual_xy_nm = (float(nx) * unit_x_nm, float(ny) * unit_y_nm)  # 石墨晶格兼容的实际 XY。
    return nx, ny, actual_xy_nm  # 返回 repeat 和对应实际 XY。


# ---------------- user inputs ----------------
restart_status = env_bool("EG08_RESTART", True)  # True 断点续跑；False 会清空 07 的 work_dir 并重新构建/采样。
set_run_options(restart=restart_status)  # 设置全局运行选项，例如 restart。

ff = GAFF2_mod()  # CMC、碳酸酯、PF6- 和石墨使用 GAFF2_mod，和其它 eg08 保持一致。
ion_ff = MERZ()  # Li+/Na+ 使用 Merz 离子参数。

temp = env_float("EG08_TEMP_K", 318.15)  # 温度 K；正式生产建议先固定这个值，避免和结构问题混在一起。
mpi = env_int("EG08_NTMPI", 1)  # thread-MPI rank 数；当前 GROMACS-2026 thread_mpi 单卡通常用 1。
omp = env_int("EG08_OMP", 12)  # OpenMP 线程数；远端 48 逻辑线程四卡并发时每个 case 用 12，避免 CPU 超订阅。
gpu = env_int("EG08_USE_GPU", 1)  # 1 使用 GPU 加速，0 强制 CPU。
gpu_id = env_int("EG08_GPU_ID", 0)  # GPU 编号；多卡节点按实际可用卡修改。
run_sampling = env_bool("EG08_RUN_SAMPLING", True)  # True 跑生产；若只想检查构建，把这里改 False。
shared_t0_charge_sweep_mode = env_bool("EG08_SHARED_T0_MODE", True)  # True：所有电荷 case 共用 prepared slabs，并从 assembled t=0 直接进 final NVT。
start_interdiffusion_at_final_nvt = True  # True 表示 final NVT 第一帧才定义为电解液/CMC 互扩散 t=0。
pre_equilibrate_cmcna_xy_slab = True  # True 先把 CMC-Na 做成 z 无周期 xy slab，再拼入界面。
pre_equilibrate_electrolyte_xy_slab = True  # True 先把电解液也做成同一 XY 的 z-open slab，避免 final stack fresh packing 缝隙。
cmc_slab_nominal_xy_nm = (5.00, 7.70)  # CMC slab 的名义横向尺寸；脚本会向上取整到石墨晶格兼容 XY。
xy_match_tolerance_nm = 0.02  # relaxed CMC/electrolyte slab 与最终石墨 XY 的允许差；超过说明不应直接拼接。
cmcna_snap_to_graphite_xy = True  # True 时把压实后的 CMC slab 小幅 affine-snap 到最近的石墨晶格兼容 XY。
cmcna_snap_max_lateral_strain = 0.06  # snap 的最大允许横向应变；超过说明 CMC 压实后的 XY 和石墨尺寸不该强行拼。
cmcna_initial_slab_density_g_cm3 = 0.05  # CMC-Na 初始插入密度；越低越容易放入长链，但 slab z 会更长。
cmcna_compression_density_g_cm3 = 1.20  # CMC 膜几何压缩参考密度；远端 DP5/8 链测试表明 1.2 左右可闭合贯通孔并压平表面。
cmcna_min_active_density_g_cm3 = 1.00  # CMC 膜最低可接受 active density；低于此值常见横向空洞或贯通孔。
cmcna_slab_eq_tmax_K = 450.0  # CMC-Na slab 退火最高温度；DP20 链需要更强的构象重排能力。
cmcna_slab_eq_pmax_bar = 2000.0  # 保留给旧 wall_z_npt 实验模式；默认 wall_gap_compression 不依赖 barostat 缩盒。
cmcna_xy_compaction_pressure_bar = 3000.0  # Z 压扁后释放 XY 压控的横向压实力；空洞多时升到 4000-5000 bar。
cmcna_xy_compaction_npt_ns = 0.10  # 横向高压 NPT 时间；空洞多时升到 0.2-0.5 ns，但要检查是否过度收缩。
cmcna_xy_compaction_final_npt_ns = 0.05  # 回到 1 bar 的短 XY-NPT 时间，避免高压结构直接进入 final NVT。
cmcna_xy_compaction_temp_K = 380.0  # 横向压实时的温度；比生产温度略高，帮助链段填补 XY 空洞。
cmcna_surface_mold_cycles = 4  # XY 填满后追加几轮小步 Z 表面整形；表面不平时增加到 6-8。
cmcna_surface_mold_z_shrink_per_cycle = 0.03  # 每轮 Z wall-gap 额外缩短比例；小步修平，避免一次性压坏链构象。
cmcna_surface_mold_temp_K = 420.0  # 表面整形 hot NVT 温度；高于横向压实温度以帮助表面链段重排。
cmcna_minimize_nsteps = 5000  # 每轮几何压缩后的 steep minimization 步数；只消坏接触，不做严格能量收敛。
cmcna_final_minimize_nsteps = 10000  # final/XY/mold 阶段 minimization 步数；太大时大体系制备会很慢。
cmcna_slab_eq_ns = 0.50  # 最终 wall-confined NVT 基础松弛时长；active density 只做稳定性诊断。
cmcna_slab_max_convergence_rounds = 8  # density/Rg 未收敛时最多追加多少轮 NVT。
cmcna_slab_extra_relax_ns = 0.50  # 每轮追加 NVT 的长度；长链体系可升到 1.0 ns。
cmcna_slab_density_rel_std_max = 0.08  # active density 尾段相对波动上限。
electrolyte_initial_slab_density_g_cm3 = 0.85  # 电解液初始插入密度；略稀疏便于放入，后续 wall EQ21 压到接近 bulk。
electrolyte_target_slab_density_g_cm3 = 1.15  # 电解液 xy-slab 目标密度；接近 carbonate/LiPF6 bulk 液体口径。
electrolyte_active_thickness_nm = env_float("EG08_ELECTROLYTE_ACTIVE_THICKNESS_NM", 10.0)  # 电解液 prepared slab 的 active 厚度；不含 wall padding，默认 10 nm，可用环境变量调整。
max_total_atoms = env_int("EG08_MAX_TOTAL_ATOMS", 200000)  # 大体系保护阈值；超过该估计值直接停止，避免误生成过大的任务。
electrolyte_slab_eq_tmax_K = 360.0  # 电解液 slab 退火最高温度；比 CMC 低，避免液体阶段过激扰动。
electrolyte_slab_eq_ns = 0.10  # 电解液最终 wall-confined NVT 松弛时长；生产制备可升到 0.2-0.5 ns。

# This script is meant for a real large-cell production test.  For a quick
# build-only check, set run_sampling=False above.
sample_ns = env_float("EG08_SAMPLE_NS", 100.0)  # final NVT 采样时长 ns；正式电荷 sweep 默认跑 100 ns。
production_dt_ps = env_float("EG08_PRODUCTION_DT_PS", 0.002)  # final NVT 生产步长；2 fs + h-bonds 可把 100 ns 步数减半。
production_constraints = os.environ.get("EG08_PRODUCTION_CONSTRAINTS", "h-bonds") or "h-bonds"  # final NVT 生产约束；默认只约束 H-bonds。
production_traj_ps = os.environ.get("EG08_PRODUCTION_TRAJ_PS", "auto") or "auto"  # final XTC 输出间隔；auto 按采样时长控制总帧数。
production_energy_ps = os.environ.get("EG08_PRODUCTION_ENERGY_PS", "auto") or "auto"  # final energy 输出间隔；auto 控制总点数。
production_log_ps = os.environ.get("EG08_PRODUCTION_LOG_PS", "auto") or "auto"  # final log 输出间隔；auto 控制总点数。

# Keep the first large flat run neutral.  Change this single value to run a
# fixed-charge basal-electrode case after the neutral structure looks healthy.
surface_charge_uC_cm2 = env_float("EG08_CMC_FACING_SURFACE_CHARGE_UC_CM2", 0.0)  # CMC-facing graphite inner face 的面电荷；生产 sweep 用 0、-3、-9、-18。
cmc_facing_surface_charge_sweep_uC_cm2 = (0.0, -3.0, -9.0, -18.0)  # 四卡并行 sweep 的推荐电荷序列。

# ---------------- post-processing controls ----------------
analysis_profile = "interface_fast"  # 使用 slab/interface 分析预设；长轨迹会自动抽稀到合理帧数，避免 100 ns sweep 后处理串行跑数小时。
analysis_frame_stride = os.environ.get("EG08_ANALYSIS_FRAME_STRIDE", "auto") or "auto"  # 后处理默认让 interface_fast 自适应抽帧；精细复算可显式设 1。
interface_bin_nm = 0.05  # z-profile bin 宽；大体系统计足够时可降到 0.025。
interface_region_width_nm = 0.75  # 近石墨/混合/core 区域宽度；CMC 很厚时可试 1.0。
interface_surface_grid_nm = 0.50  # 石墨表面 xy occupancy map 网格尺寸。
graphite_adsorption_cutoff_nm = 0.50  # COM 距石墨表面小于 0.5 nm 计为近表面吸附。
penetration_threshold_nm = 0.20  # 进入 CMC/mixed 区至少 0.2 nm 才计为 penetration。
adsorption_min_residence_ps = 10.0  # adsorption summary 里通过驻留阈值的最短累计时间。
potential_reference = "zero_mean"  # 电势零点；若想看从盒子底部积分，可改 "zero_start"。
split_electrodes_for_edl = True  # 分开统计上下电极附近 EDL，适合 sandwich。
report_potential_drop = True  # 输出 potential drop 诊断；中性体系也可作为 sanity check。
compute_interface_transport = True  # 计算区域 MSD；Dxy 用于界面内扩散趋势。
time_series_analysis = True  # 打开时间序列 CSV/MP4，便于检查长时间生产中结构是否压实和互扩散。
interface_time_series_sample_count = 10  # 全轨迹按十分之一时长取 10 帧/窗口。
interface_time_series_fps = 1.0  # MP4 慢速播放，太快不利于判断结构演化。
interface_time_series_rdf = True  # 输出全体系 cation RDF/CN 和 graphite-EDL RDF/CN；RDF 实线、CN 虚线且 CN 轴为 0-6。
interface_time_series_concentration = True  # 输出 z 浓度 profile 时间序列。
interface_time_series_angles = True  # 输出吸附角度分布时间序列。
interface_time_series_charge_potential = True  # 输出带电 graphite/EDL 电荷密度、积分电荷和电势 profile 的 CSV/PNG/MP4。
li_solvation_depth_analysis = True  # sweep PPT 汇总阶段统计“进入 CMC 的 Li 溶剂化结构随进入深度变化”。
li_solvation_depth_cutoff_nm = 0.32  # Li 周围配位壳层 cutoff；统计 solvent-O、CMC-O、PF6-F 三类配位位点。
li_solvation_depth_bin_nm = 0.10  # Li 进入 CMC 的深度分箱；太小会噪声大，100 ns sweep 推荐 0.10 nm。
penetration_species = ("EC", "EMC", "DEC", "PF6", "Li", "Na")  # 渗入 CMC/混合区分析物种。
adsorption_species = ("EC", "EMC", "DEC")  # 石墨吸附取向分析物种；通常只看中性溶剂。
clean_trajectories_after_analysis = False  # True 会删轨迹省空间；长跑验收时应保持 False。
# 多电荷 sweep 不要在一个 Python for-loop 里串行后处理；生产四卡跑完后请用
# postprocess_charge_sweep_parallel.py，并设置 EG08_POSTPROCESS_WORKERS=4。

glucose6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"  # CMC 羧酸根重复单元。
ter_smiles = "[H][*]"  # 聚合物端基终止片段。
EC_smiles = "O=C1OCCO1"  # EC。
EMC_smiles = "CCOC(=O)OC"  # EMC。
DEC_smiles = "CCOC(=O)OCC"  # DEC。
Li_smiles = "[Li+]"  # Li+。
Na_smiles = "[Na+]"  # CMC-Na counterion。
PF6_smiles = "F[P-](F)(F)(F)(F)F"  # PF6-；下面必须 bonded="DRIH"，防止八面体角度畸变。

# CMC-first matched-cell composition.
cmc_dp = 20  # 每条 CMC 链聚合度；本例目标是 DP=20。
cmc_chain_count = 16  # CMC 链数；提高到 16 条，让 CMC 原子数进入 6000-20000 区间并降低表面/横向空洞风险。
base_solvent_counts = (96, 72, 168)  # 参考面积下的 EC/EMC/DEC 数量；保持 4:3:7。
base_salt_pairs = 36  # 参考面积下的 LiPF6 离子对数。
charge_scale = 0.7  # CMC/Na/Li/PF6 电荷缩放；想跑全电荷时改 1.0，但需重新验证稳定性。

BASE_DIR = Path(__file__).resolve().parent  # 例子所在目录。
work_dir = Path(os.environ["EG08_SWEEP_ROOT"]) if os.environ.get("EG08_SWEEP_ROOT") else BASE_DIR / "work_dir" / "07_cmcna_xy_slab_matched_graphite_electrolyte_cmcna_graphite"  # 本例输出根目录；远端 sweep 用 EG08_SWEEP_ROOT 指向 v2 root。


if __name__ == "__main__":  # 只在直接运行该脚本时执行主工作流。
    doctor(print_report=True)  # 打印环境诊断，确认 RDKit/GROMACS/依赖可见。
    ensure_initialized()  # 确认 yadonpy 数据目录和 MolDB bundle 已初始化。
    work_dir = workdir(work_dir, restart=restart_status)  # 创建或复用本例输出目录。

    case_name = charge_case_dirname(surface_charge_uC_cm2) if shared_t0_charge_sweep_mode else f"charge_{surface_charge_uC_cm2:+.1f}_uC_cm2".replace("+", "p").replace("-", "m").replace(".", "p")  # 把电荷值变成安全目录名。
    case_dir = work_dir.child(case_name)  # 每个电荷状态一个独立 case 目录。
    shared_structure_dir = work_dir.child("00_shared_t0_preparation") if shared_t0_charge_sweep_mode else case_dir  # shared 模式下 CMC/electrolyte slab 只制备一次。
    cmc_rw_dir = shared_structure_dir.child("00_cmc_rw")  # CMC 随机聚合中间文件目录；shared 模式下所有电荷复用。
    cmc_term_dir = shared_structure_dir.child("01_cmc_term")  # CMC 端基终止中间文件目录；shared 模式下所有电荷复用。

    glucose6 = ff.mol(glucose6_smiles, charge="RESP", prefer_db=True, require_ready=True, polyelectrolyte_mode=True)  # 从 MolDB 读取 CMC 重复单元 RESP。
    glucose6 = ff.ff_assign(glucose6, report=False)  # 给 CMC 重复单元分配 GAFF2_mod 参数。
    ter = utils.mol_from_smiles(ter_smiles)  # 构造聚合物端基，不需要 MolDB。
    EC = ff.mol(EC_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 MolDB 读取 EC。
    EC = ff.ff_assign(EC, report=False)  # 给 EC 分配力场。
    EMC = ff.mol(EMC_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 MolDB 读取 EMC。
    EMC = ff.ff_assign(EMC, report=False)  # 给 EMC 分配力场。
    DEC = ff.mol(DEC_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 MolDB 读取 DEC。
    DEC = ff.ff_assign(DEC, report=False)  # 给 DEC 分配力场。
    PF6 = ff.mol(PF6_smiles, charge="RESP", prefer_db=True, require_ready=True)  # 从 MolDB 读取 PF6-。
    PF6 = ff.ff_assign(PF6, bonded="DRIH", report=False)  # 给 PF6- 分配 DRIH 键角和 bond-bond cross term。
    Li = ion_ff.mol(Li_smiles)  # 构造 Li+。
    Li = ion_ff.ff_assign(Li, report=False)  # 给 Li+ 分配 Merz 参数。
    Na = ion_ff.mol(Na_smiles)  # 构造 Na+。
    Na = ion_ff.ff_assign(Na, report=False)  # 给 Na+ 分配 Merz 参数。
    if not all((glucose6, ter, EC, EMC, DEC, PF6, Li, Na)):  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("MolDB/FF assignment failed for one or more species.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    CMC = poly.random_copolymerize_rw([glucose6], cmc_dp, ratio=[1.0], tacticity="atactic", work_dir=cmc_rw_dir)  # 随机生成 DP=20 CMC 链。
    CMC = poly.terminate_rw(CMC, ter, work_dir=cmc_term_dir)  # 用 H 端基终止 CMC 链。
    CMC = ff.ff_assign(CMC, report=False)  # 给整条 CMC 链分配 GAFF2_mod 参数。
    if not CMC:  # 根据当前状态决定是否进入该分支。
        raise RuntimeError("Cannot assign GAFF2_mod parameters to the DP=20 CMC-Na chain.")  # 关键步骤失败时立即报错，避免继续生成错误结果。

    pre_slab_nx, pre_slab_ny, graphite_compatible_xy_nm = graphite_repeats_for_xy(cmc_slab_nominal_xy_nm)  # 把名义 CMC XY 转成石墨晶格兼容 XY。
    print(f"requested_cmc_slab_xy_nm = {cmc_slab_nominal_xy_nm}")  # 打印用户请求的名义 CMC XY。
    print(f"graphite_compatible_cmc_slab_xy_nm = {graphite_compatible_xy_nm}")  # 打印实际用于 slab 的 XY。
    print(f"pre_slab_graphite_repeats = nx={pre_slab_nx}, ny={pre_slab_ny}")  # 打印匹配该 XY 所需的石墨 repeat。
    cmcna_prepared_slab_gro = None  # 默认没有 prepared slab；开启预平衡后改为 wrapped-XY/z-open 的 stack-facing GRO。
    if pre_equilibrate_cmcna_xy_slab:  # True 时先独立预平衡 CMC-Na slab。
        cmcna_slab_dir = shared_structure_dir.child("02_cmcna_xy_slab_eq21")  # CMC-Na slab 预平衡输出目录；shared 模式下只做一次。
        cached_cmcna_snap_gro = shared_structure_dir / "02b_cmcna_graphite_xy_snap" / "cmcna_prepared_slab_graphite_xy.gro"  # 最终 stack 使用的是 snap 到石墨 XY 后的 CMC 成品。
        cached_cmcna_raw_gro = cmcna_slab_dir / "03_EQ21_XY_SLAB" / "prepared_slab.gro"  # 若尚未 snap，则退回复用 CMC slab 原始成品。
        cached_cmcna_slab_gro = cached_cmcna_snap_gro if cached_cmcna_snap_gro.is_file() else cached_cmcna_raw_gro  # shared-t0 下游 case 应复用成品，不再并发重跑 CMC slab。
        if shared_t0_charge_sweep_mode and cached_cmcna_slab_gro.is_file():  # 四个电荷 case 同时启动时，必须走只读复用路径。
            cmcna_prepared_slab_gro = cached_cmcna_slab_gro  # 直接使用 shared prepared slab。
            print(f"cmcna_prepared_slab_gro = {cmcna_prepared_slab_gro}")  # 打印复用路径。
            print(f"cmcna_prepared_slab_reused = True")  # 明确记录没有二次制备，避免 race condition。
        else:
            cmcna_slab = prepare_cmcna_xy_membrane(  # CMC-Na 专用入口会完成稀疏 AC、显式 wall-gap 压缩、active density/Rg 收敛检查。
                cmc_chain_mol=CMC,  # 已经完成端基和力场分配的 DP20 CMC 链。
                na_mol=Na,  # CMC 羧酸根的 Na+ counterion。
                chain_count=cmc_chain_count,  # CMC 链条数；默认 16 条，增强横向填充和膜连续性。
                dp=cmc_dp,  # 每条链的重复单元数；Na 数自动按 chain_count*dp 设置。
                xy_nm=graphite_compatible_xy_nm,  # 固定 XY footprint；后续石墨和电解液都按这个横向尺寸匹配。
                work_dir=cmcna_slab_dir,  # CMC-Na slab 全部输出写到该目录。
                temp=temp,  # wall-confined NVT 的目标温度。
                pressure_bar=1.0,  # 正常压力；CMC slab 会先显式压 Z，再用 XY-NPT 横向压实，最后回到该压力。
                mpi=mpi,  # thread-MPI rank。
                omp=omp,  # OpenMP 线程数。
                gpu=gpu,  # GPU 开关。
                gpu_id=gpu_id,  # GPU 编号。
                charge_scale=(charge_scale, charge_scale),  # CMC 与 Na+ 使用同一电荷缩放，保持局域配对口径。
                relaxation=CMCNAXYSlabRelaxationSpec(  # CMC-Na z-open bulk slab 的收敛制备配置。
                    initial_density_g_cm3=cmcna_initial_slab_density_g_cm3,  # 稀疏 AC 初始密度；只用于提高插入成功率。
                    density_mode="wall_gap_compression",  # 关键：显式缩短 wall gap/box_z，避免依赖 GROMACS z barostat。
                    coordinate_export_policy="wrapped_xy_z_open",  # 关键：导出给 stack 的 GRO 只 wrap XY，Z 保持开边界。
                    target_density_g_cm3=cmcna_compression_density_g_cm3,  # 几何压缩参考密度；只用来先把 Z 压扁到膜厚量级。
                    active_density_min_g_cm3=cmcna_min_active_density_g_cm3,  # 低于该 active density 说明膜仍过稀疏。
                    wall_padding_nm=0.40,  # z walls 与 active slab 两侧保留 padding，避免原子贴墙。
                    cycles="auto",  # wall-gap 压缩轮数；auto 按 max_z_shrink_per_cycle 小步缩短 box_z。
                    max_cycles=24,  # 极端长盒子最多允许的退火轮数；1.2 g/cm3 construction target 一般 15-18 轮内到位。
                    max_z_shrink_per_cycle=0.20,  # 每轮最多缩短 20%；远端测试可稳定形成规则 DP5/8 CMC 膜。
                    tmax_K=cmcna_slab_eq_tmax_K,  # 显式压缩循环中的最高 NVT 退火温度。
                    pmax_bar=cmcna_slab_eq_pmax_bar,  # 旧 wall_z_npt 模式参数；默认显式压缩模式仅记录该值。
                    pre_nvt_ns=0.02,  # 压缩前短 NVT，先释放局部坏接触。
                    wall_npt_ns=0.05,  # 旧 wall_z_npt 模式参数；默认显式压缩模式不用 barostat 压缩。
                    hot_nvt_ns=0.01,  # 每轮压缩后的高温 NVT 时长；可升到 0.02-0.05 ns 增强链重排。
                    cool_nvt_ns=0.01,  # 每轮压缩后的降温 NVT 时长；正式膜制备可适当加长。
                    final_relax_ns=cmcna_slab_eq_ns,  # 正常温度 wall-confined NVT 基础松弛时长。
                    max_convergence_rounds=cmcna_slab_max_convergence_rounds,  # 未收敛时追加 NVT 的最大轮数。
                    extra_relax_ns_per_round=cmcna_slab_extra_relax_ns,  # 每轮追加 NVT 长度。
                    active_density_convergence=True,  # 必须检查 active density 尾段是否稳定。
                    rg_convergence=True,  # 必须检查 CMC 链 Rg 是否进入平台。
                    lateral_occupancy_convergence=True,  # 必须检查 CMC 是否铺满 XY footprint，避免横向空洞。
                    surface_flatness_convergence=True,  # 必须检查上下膜表面是否足够平整，便于后续直接贴石墨。
                    connected_void_convergence=True,  # 必须检查是否存在贯通空洞，避免电解液从膜孔/边缘旁路直达石墨。
                    active_density_rel_std_max=cmcna_slab_density_rel_std_max,  # active density 波动容差。
                    max_surface_rms_nm=0.35,  # 上下表面网格高度 RMS 的最大允许值；更严格可降到 0.20-0.25 nm。
                    max_surface_peak_to_peak_nm=1.00,  # 表面峰-谷高度差上限；过大说明膜不是平整 slab。
                    void_grid_nm=0.35,  # 贯通空洞检测网格；越小越严格但更敏感。
                    void_atom_radius_nm=0.22,  # 空洞检测中把原子膨胀成排除体积的半径。
                    max_connected_void_fraction=0.20,  # 与底面连通的空洞体积分数上限；过大说明膜内有明显通道。
                    xy_compaction_npt=True,  # Z 压扁后释放 XY 方向控压，让横向空洞被进一步挤掉。
                    xy_compaction_pressure_bar=cmcna_xy_compaction_pressure_bar,  # 横向压实压力，越高压得越狠。
                    xy_compaction_temp_K=cmcna_xy_compaction_temp_K,  # 横向压实温度，略高温有利于 CMC 链段重排。
                    xy_compaction_npt_ns=cmcna_xy_compaction_npt_ns,  # 高压 XY-NPT 持续时间。
                    xy_compaction_final_npt_ns=cmcna_xy_compaction_final_npt_ns,  # 回到正常压力后的 XY-NPT 时间。
                    surface_mold_nvt=True,  # 横向压实后追加 Z 表面整形，目标是规则、可贴石墨的膜表面。
                    surface_mold_cycles=cmcna_surface_mold_cycles,  # 表面整形最多循环次数。
                    surface_mold_z_shrink_per_cycle=cmcna_surface_mold_z_shrink_per_cycle,  # 每轮小步压扁 Z。
                    surface_mold_hot_temp_K=cmcna_surface_mold_temp_K,  # 表面整形 hot NVT 温度。
                    minimize_nsteps=cmcna_minimize_nsteps,  # 压缩循环 minimization 上限。
                    final_minimize_nsteps=cmcna_final_minimize_nsteps,  # final/XY/mold minimization 上限。
                    write_compression_animation=True,  # 输出带盒子尺寸的 CMC 压缩/松弛 MP4 和 PNG 帧。
                    animation_fps=1.0,  # 低帧率播放，方便人工检查 box_z 是否逐轮缩小。
                    na_coo_contact_cutoff_nm=0.35,  # Na+/COO- 接触距离阈值。
                    na_coo_contact_min_fraction=0.75,  # 大部分 Na+ 应保持在羧酸根附近。
                ),
                retry=30,  # 长链稀疏插入重试次数。
                retry_step=2000,  # 每轮 packing 试探步数。
                threshold_ang=2.0,  # 初始插入排斥阈值。
                large_system_mode="large",  # DP20 x 8 使用大体系 packing 策略。
                restart=restart_status,  # 允许断点复用已经构建好的 AC 和 EQ 阶段。
            )
            cmcna_prepared_slab_gro = cmcna_slab.prepared_slab_gro  # 这个 wrapped-XY/z-open GRO 将作为 CMCNA layer 的真实初始坐标。
            print(f"cmcna_prepared_slab_gro = {cmcna_prepared_slab_gro}")  # 打印 slab 坐标路径，便于人工检查。
            print(f"cmcna_prepared_slab_whole_gro = {cmcna_slab.prepared_slab_whole_gro}")  # whole-molecule GRO 仅用于诊断，不用于拼接。
            print(f"cmcna_prepared_slab_coordinate_report = {cmcna_slab.coordinate_summary}")  # 记录 wrap 前后外包络和 XY occupancy。
            print(f"cmcna_slab_ready_for_layer_stack = {cmcna_slab.ready_for_layer_stack}")  # 打印 density/Rg/Na-COO 收敛门的总体结果。
            print(f"cmcna_slab_convergence_json = {cmcna_slab.convergence_summary}")  # 打印收敛 JSON，便于后续复查。
    else:
        raise RuntimeError("Example 08-07 is defined around a prepared CMC-Na xy slab; keep pre_equilibrate_cmcna_xy_slab=True.")  # 防止误用为普通 packing 例子。
    cmc_slab_xy_nm = read_gro_xy_nm(Path(cmcna_prepared_slab_gro))  # 从 relaxed CMC slab GRO 读回真实 XY 盒长。
    graphite_nx, graphite_ny, matched_graphite_xy_nm = graphite_repeats_for_xy(cmc_slab_xy_nm)  # 按读回 XY 选择最终石墨 repeat。
    xy_delta_nm = (abs(float(matched_graphite_xy_nm[0]) - float(cmc_slab_xy_nm[0])), abs(float(matched_graphite_xy_nm[1]) - float(cmc_slab_xy_nm[1])))  # 计算石墨和 CMC slab 的 XY 差值。
    if cmcna_snap_to_graphite_xy and (xy_delta_nm[0] > xy_match_tolerance_nm or xy_delta_nm[1] > xy_match_tolerance_nm):  # 压实后的 CMC XY 往往不是石墨 repeat 的整数倍。
        snap_dir = shared_structure_dir.child("02b_cmcna_graphite_xy_snap")  # snap 输出独立保存，避免覆盖原始 CMC prepared slab。
        snap_dir.mkdir(parents=True, exist_ok=True)  # 创建 snap 目录。
        snap_report = retarget_prepared_slab_xy(  # 轻微 affine 缩放 x/y 到最近的石墨兼容尺寸，z-open 坐标不动。
            cmcna_prepared_slab_gro,  # 输入为 CMC surface-mold 后的 wrapped-XY prepared GRO。
            matched_graphite_xy_nm,  # 目标为最近可由石墨 basal repeats 铺出的 XY。
            out_gro=snap_dir / "cmcna_prepared_slab_graphite_xy.gro",  # 输出给最终 stack 使用。
            max_abs_strain=cmcna_snap_max_lateral_strain,  # 超过该应变直接报错，避免把膜拉回空洞状态。
        )
        cmcna_prepared_slab_gro = Path(snap_report["retargeted_gro"])  # 后续拼接改用 snap 后的 GRO。
        cmc_slab_xy_nm = read_gro_xy_nm(Path(cmcna_prepared_slab_gro))  # 重新读回 snap 后的 XY，应与石墨完全一致。
        xy_delta_nm = (abs(float(matched_graphite_xy_nm[0]) - float(cmc_slab_xy_nm[0])), abs(float(matched_graphite_xy_nm[1]) - float(cmc_slab_xy_nm[1])))  # 重新计算差值。
        print(f"cmcna_graphite_xy_snap_report = {snap_report.get('report_path')}")  # 打印 snap 诊断，记录应变和 occupancy。
    if xy_delta_nm[0] > xy_match_tolerance_nm or xy_delta_nm[1] > xy_match_tolerance_nm:  # 如果石墨晶格不能足够贴合 CMC slab。
        raise RuntimeError(  # 直接停止，避免把 XY 周期 slab 放进不兼容的 stack 里。
            f"CMC slab XY {cmc_slab_xy_nm} does not match graphite-compatible XY {matched_graphite_xy_nm}; "
            f"delta={xy_delta_nm}, tolerance={xy_match_tolerance_nm} nm."
        )
    graphite_xy_nm = matched_graphite_xy_nm  # 后续电解液数量和 stack master XY 都以 CMC slab 读回值为准。
    print(f"cmc_slab_xy_nm = {cmc_slab_xy_nm}")  # 打印 CMC slab 的真实 XY。
    print(f"matched_graphite_repeats = nx={graphite_nx}, ny={graphite_ny}, xy_nm={matched_graphite_xy_nm}")  # 打印最终石墨匹配结果。
    electrolyte_prepared_slab_gro = None  # 默认不复用电解液 slab；开启预平衡后改为 wrapped-XY/z-open 的 final gro。
    electrolyte_species = (EC, EMC, DEC, Li, PF6)  # 电解液 species 顺序必须和 counts/charge_scale 对齐。
    electrolyte_plan = estimate_electrolyte_counts_for_slab(  # 按指定 active slab 厚度和目标密度反推数量，而不是只按面积缩放。
        species=electrolyte_species,  # EC/EMC/DEC/Li/PF6。
        base_solvent_counts=base_solvent_counts,  # 保持 4:3:7 溶剂比例。
        base_salt_pairs=base_salt_pairs,  # 保持参考 LiPF6/溶剂比例。
        xy_nm=(float(graphite_xy_nm[0]), float(graphite_xy_nm[1])),  # 使用最终 graphite/CMC 共同 XY。
        active_thickness_nm=float(electrolyte_active_thickness_nm),  # 使用脚本/环境变量给定的 active 电解液层厚度。
        target_density_g_cm3=electrolyte_target_slab_density_g_cm3,  # 目标接近 bulk 液体密度。
    )
    solvent_counts = tuple(int(v) for v in electrolyte_plan["solvent_counts"])  # EC/EMC/DEC 数量。
    salt_pairs = int(electrolyte_plan["salt_pairs"])  # LiPF6 离子对数。
    electrolyte_counts = (*solvent_counts, salt_pairs, salt_pairs)  # EC/EMC/DEC/Li/PF6 数量。
    estimated_cmc_atoms = int(CMC.GetNumAtoms()) * int(cmc_chain_count) + int(Na.GetNumAtoms()) * int(cmc_chain_count) * int(cmc_dp)  # CMC-Na 原子数估计。
    estimated_graphite_atoms = int(2 * 3 * 8 * int(graphite_nx) * int(graphite_ny))  # 保守估计上下三层石墨原子数，偏高以保护任务规模。
    estimated_total_atoms = int(electrolyte_plan["estimated_electrolyte_atoms"]) + estimated_cmc_atoms + estimated_graphite_atoms  # 生产体系原子数估计。
    electrolyte_plan["estimated_cmcna_atoms"] = int(estimated_cmc_atoms)  # 写入诊断。
    electrolyte_plan["estimated_graphite_atoms_conservative"] = int(estimated_graphite_atoms)  # 写入诊断。
    electrolyte_plan["estimated_total_atoms_conservative"] = int(estimated_total_atoms)  # 写入诊断。
    print(f"electrolyte_active_thickness_nm = {electrolyte_plan['active_thickness_nm']}")  # 打印 active 厚度口径。
    print(f"scaled_electrolyte_counts = {solvent_counts}, salt_pairs={salt_pairs}")  # 打印按 active volume/density 反推后的电解液组成。
    print(f"electrolyte_count_plan = {json.dumps(electrolyte_plan, ensure_ascii=False)}")  # 打印完整数量/密度/原子数诊断。
    if estimated_total_atoms > int(max_total_atoms):  # 避免误把单机脚本扩成超大任务。
        raise RuntimeError(  # 清晰提示如何降低规模。
            f"Estimated total atom count {estimated_total_atoms} exceeds EG08_MAX_TOTAL_ATOMS={int(max_total_atoms)}. "
            "Reduce EG08_ELECTROLYTE_ACTIVE_THICKNESS_NM, CMC chain count/XY size, or raise EG08_MAX_TOTAL_ATOMS intentionally."
        )
    electrolyte_charge_scale = (1.0, 1.0, 1.0, charge_scale, charge_scale)  # 溶剂全电荷，Li/PF6 与 CMC-Na 使用同一缩放。
    if pre_equilibrate_electrolyte_xy_slab:  # True 时先独立预平衡电解液 slab。
        electrolyte_slab_dir = shared_structure_dir.child("03_electrolyte_xy_slab_eq21")  # 电解液 slab 预平衡输出目录；shared 模式下所有电荷复用。
        cached_electrolyte_slab_gro = electrolyte_slab_dir / "03_EQ21_XY_SLAB" / "prepared_slab.gro"  # shared 成品电解液 slab。
        if shared_t0_charge_sweep_mode and cached_electrolyte_slab_gro.is_file():  # 四个电荷 case 只读复用，不能并发重跑。
            electrolyte_prepared_slab_gro = cached_electrolyte_slab_gro  # 使用已有电解液 slab。
            print(f"electrolyte_prepared_slab_gro = {electrolyte_prepared_slab_gro}")  # 打印复用路径。
            print(f"electrolyte_prepared_slab_reused = True")  # 明确记录复用。
        else:
            electrolyte_total_mass_amu = sum(  # 估算电解液总质量，用于从初始密度反推 z 长度。
                rdkit_mol_mass_amu(mol) * int(count) for mol, count in zip(electrolyte_species, electrolyte_counts)
            )
            electrolyte_initial_volume_nm3 = (electrolyte_total_mass_amu / 6.02214076e23) / electrolyte_initial_slab_density_g_cm3 * 1.0e21  # 初始 0.85 g/cm3 对应体积。
            electrolyte_initial_z_nm = electrolyte_initial_volume_nm3 / (float(graphite_xy_nm[0]) * float(graphite_xy_nm[1]))  # 固定 XY 后所需初始 z。
            electrolyte_initial_cell = make_orthorhombic_pack_cell((float(graphite_xy_nm[0]), float(graphite_xy_nm[1]), electrolyte_initial_z_nm))  # 构造电解液 z-open 初始盒子。
            electrolyte_ac = poly.amorphous_cell(  # 在固定 XY、略稀疏 z 中放入电解液。
                list(electrolyte_species),  # EC/EMC/DEC/Li/PF6。
                list(electrolyte_counts),  # 与 species 对齐的数量。
                cell=electrolyte_initial_cell,  # 固定 XY，z 由初始密度估算。
                density=None,  # 已显式给出 cell，不让 amorphous_cell 再改盒子。
                retry=30,  # 液体组分较多时提高插入重试次数。
                retry_step=2000,  # 每轮 packing 试探步数。
                threshold=2.0,  # 初始排斥阈值；太大更难插入，太小坏接触更多。
                charge_scale=electrolyte_charge_scale,  # Li/PF6 缩放；溶剂保持全电荷。
                polyelectrolyte_mode=False,  # 普通电解液不是聚电解质。
                large_system_mode="large",  # 大 XY slab 使用大体系 packing 策略。
                work_dir=electrolyte_slab_dir.child("00_ac"),  # 初始电解液 AC 输出目录。
                restart=restart_status,  # 允许断点复用已经构建好的 AC。
            )
            electrolyte_slab_eq = eq.EQ21step(electrolyte_ac, work_dir=electrolyte_slab_dir)  # 创建电解液 slab EQ21 任务。
            electrolyte_slab_eq.exec(  # 运行 wall-confined pbc=xy 电解液压实/松弛。
                temp=temp,  # 目标温度。
                press=1.0,  # 仅作为 wall/NVT 记录参数；默认固定 XY，不做 z-NPT。
                mpi=mpi,  # thread-MPI rank。
                omp=omp,  # OpenMP 线程数。
                gpu=gpu,  # GPU 开关。
                gpu_id=gpu_id,  # GPU 编号。
                sim_time=electrolyte_slab_eq_ns,  # 兼容 EQ21 参数；xy-slab 分支主要使用 final_relax_ns。
                eq21_tmax=electrolyte_slab_eq_tmax_K,  # 电解液 slab 退火最高温度。
                eq21_dt_ps=0.001,  # 1 fs 步长，优先保证 z-open wall 初期稳定。
                periodicity="xy",  # 关键：z 方向无周期，使用 z walls。
                xy_slab=XYSlabEquilibrationSpec(  # 电解液 slab 专用配置。
                    coordinate_export_policy="wrapped_xy_z_open",  # 电解液导出也 wrap XY，保证拼接时不会按外包络居中。
                    target_density_g_cm3=electrolyte_target_slab_density_g_cm3,  # 目标接近 bulk 液体密度。
                    cycles="auto",  # 自动按每轮最大压缩比例生成 cycles。
                    max_cycles=20,  # 电解液通常比 CMC 更容易压实，限制最大轮数。
                    max_z_shrink_per_cycle=0.08,  # 每轮最多缩短 8% z，避免液体/离子瞬间过密。
                    wall_padding_nm=0.40,  # slab 两侧保留 wall padding。
                    xy_area_mode="fixed",  # 固定 XY footprint，确保可直接拼接到 CMC/石墨。
                    hot_nvt_ns=0.01,  # 每轮热 NVT 时间。
                    cool_nvt_ns=0.01,  # 每轮回到目标温度的 NVT 时间。
                    final_relax_ns=electrolyte_slab_eq_ns,  # 到目标 z 后的最终 wall-confined NVT。
                ),
                restart=restart_status,  # 允许断点续跑。
            )
            electrolyte_prepared_slab_gro = electrolyte_slab_eq.final_gro()  # 这个 wrapped-XY/z-open GRO 将作为 ELECTROLYTE layer 的真实初始坐标。
        electrolyte_slab_xy_nm = read_gro_xy_nm(Path(electrolyte_prepared_slab_gro))  # 从 relaxed electrolyte slab GRO 读回真实 XY。
        electrolyte_xy_delta_nm = (abs(float(electrolyte_slab_xy_nm[0]) - float(graphite_xy_nm[0])), abs(float(electrolyte_slab_xy_nm[1]) - float(graphite_xy_nm[1])))  # 检查电解液 slab 与最终 XY 是否一致。
        if electrolyte_xy_delta_nm[0] > xy_match_tolerance_nm or electrolyte_xy_delta_nm[1] > xy_match_tolerance_nm:  # 电解液 slab 不应在拼接时被横向重缩放。
            raise RuntimeError(  # 直接停止，避免横向缝隙或隐式应变进入 final stack。
                f"Electrolyte slab XY {electrolyte_slab_xy_nm} does not match stack XY {graphite_xy_nm}; "
                f"delta={electrolyte_xy_delta_nm}, tolerance={xy_match_tolerance_nm} nm."
            )
        print(f"electrolyte_prepared_slab_gro = {electrolyte_prepared_slab_gro}")  # 打印电解液 slab 坐标路径，便于人工检查。
        print(f"electrolyte_prepared_slab_coordinate_report = {Path(electrolyte_prepared_slab_gro).with_name('prepared_slab_coordinate_report.json')}")  # 打印 wrap 诊断路径。
    else:
        raise RuntimeError("Example 08-07 requires pre_equilibrate_electrolyte_xy_slab=True to avoid fresh-packing lateral voids.")  # 防止误回到旧拼接方案。
    graphite_bottom = GraphiteLayerSpec(  # 最终下石墨层按 CMC slab XY 自适应。
        name="GRAPHITE_BOTTOM",  # 下石墨层名；固定电荷区域按这个名字选层。
        nx=graphite_nx,  # 从 CMC slab XY 读回后计算的 x repeat。
        ny=graphite_ny,  # 从 CMC slab XY 读回后计算的 y repeat。
        n_layers=3,  # 石墨层数；若要更硬/更厚电极可增加。
        orientation="basal",  # 基面石墨。
        periodic_xy=True,  # XY 周期成键，防止基面边缘断裂。
    )
    graphite_top = GraphiteLayerSpec(  # 最终上石墨层也按 CMC slab XY 自适应。
        name="GRAPHITE_TOP",  # 上石墨层名；与下石墨组成 sandwich。
        nx=graphite_nx,  # 与下石墨相同。
        ny=graphite_ny,  # 与下石墨相同。
        n_layers=3,  # 与下石墨相同。
        orientation="basal",  # 基面石墨。
        periodic_xy=True,  # XY 周期成键。
    )
    electrolyte = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="ELECTROLYTE",  # 电解液层名。
        species=electrolyte_species,  # species 顺序必须对应 counts 和 charge_scale。
        counts=electrolyte_counts,  # EC/EMC/DEC/Li/PF6 数量。
        thickness_nm=float(electrolyte_plan["active_thickness_nm"]),  # prepared slab active 厚度；本轮默认 10 nm。
        density_target_g_cm3=electrolyte_target_slab_density_g_cm3,  # prepared slab 已接近 bulk 液体密度。
        layer_kind="electrolyte",  # 电解液语义标签。
        charge_scale=electrolyte_charge_scale,  # 溶剂全电荷，Li/PF6 缩放。
        large_system_mode="large",  # 强制使用大体系 packing 策略，减少大盒子随机插入失败。
        prepared_slab_gro=electrolyte_prepared_slab_gro,  # 关键：复用 wrapped-XY 预平衡电解液 slab，不在 final stack fresh packing。
    )
    cmcna = MolecularLayerSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        name="CMCNA",  # CMC-Na 层名，后处理识别 CMC-rich/core 区域。
        species=(CMC, Na),  # CMC 链和 Na+ counterion。
        counts=(cmc_chain_count, cmc_chain_count * cmc_dp),  # 默认 16 条 DP20 链对应 320 个 Na+。
        thickness_nm=2.6,  # prepared_slab_gro 存在时主要作为 manifest 目标厚度记录，不重新按此厚度 packing。
        # The real CMC coordinates come from the wall-confined xy slab above.
        # This density target is retained as provenance and density diagnostic
        # context; it is not used to repack the final stack.
        density_target_g_cm3=cmcna_min_active_density_g_cm3,  # 仅作为最低致密膜参考写入 manifest，不参与 prepared slab 重构。
        layer_kind="cmcna",  # 启用 CMCNA 专用分组、Na+/COO- 和密度诊断。
        charge_scale=(charge_scale, charge_scale),  # CMC 与 Na+ 使用同一缩放，保持局部配对口径。
        polyelectrolyte_mode=True,  # 按聚电解质体系处理 CMC-Na。
        large_system_mode="large",  # 大体系 packing 策略。
        counterion_contact_mode="carboxylate",  # 构建后把 Na+ 放在 COO- 附近，避免初期跑进电解液。
        prepared_slab_gro=cmcna_prepared_slab_gro,  # 复用 wrapped-XY/z-open CMC slab；链可跨 XY 周期，单盒显示被切开是正确表示。
    )
    charge_specs = charge_region_specs(surface_charge_uC_cm2)  # 当前 case 的两个石墨内表面电荷；shared stack 构建时不直接传入。
    stack = LayerStackSpec(  # 设置中间变量或可调参数，供后续工作流使用。
        layers=(graphite_bottom, electrolyte, cmcna, graphite_top),  # 共享 t=0 的层顺序：石墨 | 电解液 | CMCNA | 石墨。
        order="bottom_to_top",  # 按 z 从下到上解释 layers。
        pbc_mode="xyz",  # 闭合三维周期；不是显式真空体系。
        name=f"cmcna_xy_slab_matched_graphite_stack_{case_name}",  # 系统名包含 prepared CMC/electrolyte slab 拼接策略和电荷状态。
        default_gap_nm=0.35,  # 层间初始间隙，避免 fresh overlap。
        molecular_packing_expand="z",  # 固定 CMC slab 读回的 XY；若分子太多则扩展 z 而非改横向面积。
        fixed_charge_regions=charge_specs,  # 非 shared 模式可直接在 build_layer_stack 中施加；shared 模式会改为 topology patch。
    )
    relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=sample_ns)  # 记录工作流默认温度和生产长度。
    interdiffusion_start = InterdiffusionStartSpec(  # 定义“预松弛不算互扩散，final NVT 才开始互扩散”的策略。
        enabled=start_interdiffusion_at_final_nvt,  # True 打开预释放 z-gate；False 回到平衡界面直接松弛模式。
        strategy="soft_wall_release",  # 当前实现用 GROMACS z-only position restraint 做软 gate。
        hold_interphase=True,  # pre-release 阶段限制 ELECTROLYTE/CMCNA 在 z 方向明显互穿。
        phase_pre_equilibrate=True,  # 在 summary 中记录相预平衡意图；当前执行路径为 assembled pre-release gate。
        release_at_final_nvt=True,  # final NVT 移除 phase gate，第一帧作为 diffusion t=0。
        phase_gate_k_kj_mol_nm2=1500.0,  # z-gate 力常数；界面过早混合可升到 2500，消重叠困难可降到 500。
        phase_gate_layers=("ELECTROLYTE", "CMCNA"),  # 只 gate 两个软相，不 gate 石墨。
        diffusion_t0_stage="final_nvt",  # 后处理口径：正式互扩散从 final NVT 开始。
        diffusion_t0_ps=0.0,  # final NVT 第一帧定义为 0 ps。
    )
    graphite_restraint = GraphiteRestraintSpec(  # 定义石墨 z 方向平整 restraint，避免 NVT/NPT 中电极褶皱。
        enabled="auto",  # auto 表示只要体系中有 GraphiteLayerSpec 就启用。
        mode="z_position",  # 只约束 z；xy 内可以正常热运动和整体平移。
        k_pre_kj_mol_nm2=5000.0,  # pre-release / anneal / z-NPT 阶段较强，帮助电极保持平整。
        k_final_kj_mol_nm2=1000.0,  # final NVT 阶段较弱，减少对界面动力学的扰动。
        fcx_kj_mol_nm2=0.0,  # x 不约束。
        fcy_kj_mol_nm2=0.0,  # y 不约束。
    )

    if shared_t0_charge_sweep_mode:  # shared-t0 sweep 的正确调度：中性 stack 只构建一次，四个 case 只复制并 patch 电荷。
        neutral_stack = replace(  # 从 charged stack 派生中性 shared stack。
            stack,
            name="cmcna_xy_slab_matched_graphite_stack_shared_t0_neutral",  # shared t=0 名称固定，便于四个 case 对齐。
            fixed_charge_regions=(),  # 关键：中性 shared stack 不触发 fragment-precise export，避免每个 PF6 单独写 ITP。
        )
        shared_stack_dir = shared_structure_dir.child("04_shared_t0_stack")  # shared neutral stack 输出目录。
        shared_manifest = shared_stack_dir / "layer_stack_manifest.json"  # shared manifest。
        shared_system_dir = shared_stack_dir / "02_system"  # shared system.gro/top/ndx 所在目录。
        shared_ready = (shared_system_dir / "system.gro").is_file() and (shared_system_dir / "system.top").is_file() and (shared_system_dir / "system.ndx").is_file() and shared_manifest.is_file()  # 判断 shared stack 是否已存在。
        lock_dir = shared_structure_dir / ".04_shared_t0_stack.lock"  # 多进程目录锁。
        lock_owned = False  # 当前进程是否持锁。
        if not shared_ready:  # 若 shared stack 还没构建，抢锁后构建。
            lock_owned = wait_for_shared_stack_lock(lock_dir)  # 只有一个进程能进入构建。
            try:
                shared_ready = (shared_system_dir / "system.gro").is_file() and (shared_system_dir / "system.top").is_file() and (shared_system_dir / "system.ndx").is_file() and shared_manifest.is_file()  # 抢到锁后重查，避免重复构建。
                if not shared_ready:  # 仍不存在，当前进程负责构建。
                    shared_result = build_layer_stack(stack=neutral_stack, relaxation=relaxation, work_dir=shared_stack_dir, restart=restart_status)  # 构建一次中性 shared t=0。
                    print(f"shared_t0_stack_built = {shared_result.system_gro.parent}")  # 记录路径。
                else:  # 其它进程已在等待期间构建完成。
                    shared_result = layer_stack_result_from_existing(  # 从已有文件恢复 result。
                        work_dir_path=shared_stack_dir,
                        stack_spec=neutral_stack,
                        manifest_path=shared_manifest,
                    )
            finally:
                if lock_owned:  # 释放目录锁。
                    shutil.rmtree(lock_dir, ignore_errors=True)  # 删除锁目录。
        else:  # shared stack 已存在。
            shared_result = layer_stack_result_from_existing(  # 直接从已有文件恢复 result。
                work_dir_path=shared_stack_dir,
                stack_spec=neutral_stack,
                manifest_path=shared_manifest,
            )
        result = derive_charge_case_from_shared_stack(  # 从同一个 shared neutral stack 派生当前电荷 case。
            shared_result=shared_result,
            case_dir=case_dir,
            charged_stack=stack,
            charge_specs=charge_specs,
        )
        print(f"shared_t0_stack_reused = {shared_result.system_gro.parent}")  # 明确记录复用路径。
        print(f"charge_patch_report = {result.system_gro.parent / 'charge_patch_report.json'}")  # 打印电荷 patch 报告。
    else:  # 单个非 shared case 保持旧接口，直接在 build_layer_stack 中施加电荷。
        result = build_layer_stack(stack=stack, relaxation=relaxation, work_dir=case_dir, restart=restart_status)  # 构建 GROMACS-ready stack。
    print(f"layer_stack_manifest = {result.manifest_path}")  # manifest 记录层顺序、密度、Na+/COO- 配对和电荷选区。
    print(f"stack_gmx_dir = {result.system_gro.parent}")  # system.gro/top/ndx 所在目录。
    print(f"acceptance = {result.acceptance}")  # packing 接受率/诊断摘要。

    if not shared_t0_charge_sweep_mode or env_bool("EG08_STATIC_ANALYSIS", False):  # shared sweep 默认跳过静态分析，避免四个 case 重复做同一张 t=0 图。
        analyze_layer_stack_interface(  # 对静态或轨迹界面结构执行后处理。
            work_dir=case_dir,  # 静态后处理写到 case_dir 下。
            manifest_path=result.manifest_path,  # 使用 manifest 的真实层信息。
            analysis_profile=analysis_profile,  # 界面分析预设。
            bin_nm=interface_bin_nm,  # z-profile bin 宽。
            region_width_nm=interface_region_width_nm,  # 近界面区域宽度。
            surface_grid_nm=interface_surface_grid_nm,  # xy occupancy 网格。
            surface_distance_nm=graphite_adsorption_cutoff_nm,  # 石墨近表面距离阈值。
            penetration_threshold_nm=penetration_threshold_nm,  # penetration 深度阈值。
            adsorption_min_residence_ps=adsorption_min_residence_ps,  # 吸附驻留阈值。
            potential_reference=potential_reference,  # 电势零点。
            split_electrodes=split_electrodes_for_edl,  # 分开上下电极。
            report_potential_drop=report_potential_drop,  # 输出电势差。
            penetration_species=penetration_species,  # penetration 物种。
            adsorption_species=adsorption_species,  # adsorption 物种。
            compute_transport=False,  # 静态结构没有 MSD，所以这里关闭 transport。
            time_series_analysis=False,  # 静态结构只有一帧，不生成时间序列。
        )

    if run_sampling:  # 根据当前状态决定是否进入该分支。
        relax = run_layer_stack_relaxation(  # 运行 layer-stack 松弛和 final NVT 采样。
            result,
            time_ns=sample_ns,  # final NVT 生产长度，默认 100 ns。
            temp=temp,  # 目标温度。
            mpi=mpi,  # thread-MPI rank 数。
            omp=omp,  # OpenMP 线程数。
            gpu=gpu,  # GPU 开关。
            gpu_id=gpu_id,  # GPU 编号。
            dt_ps=0.001,  # 1 fs 步长，优先保证大界面早期稳定。
            constraints="none",  # fresh interface 阶段不使用约束，减少初始失败。
            final_dt_ps=production_dt_ps,  # final NVT 使用 2 fs 步长，提高 100 ns sweep 效率。
            final_constraints=production_constraints,  # final NVT 只约束 H-bonds，允许 2 fs 稳定生产。
            final_traj_ps=production_traj_ps,  # final XTC 输出间隔；auto 控制总帧数，防止大体系文件暴涨。
            final_energy_ps=production_energy_ps,  # final EDR 输出间隔；auto 控制总点数。
            final_log_ps=production_log_ps,  # final log 输出间隔；auto 控制总点数。
            run_analysis=True,  # 运行后保留 analyze facade。
            relax_z=(False if shared_t0_charge_sweep_mode else True),  # shared-t0 sweep 禁止各电荷 case 独立 z-NPT，否则 t=0 会漂移。
            pre_nvt_ns=(0.0 if shared_t0_charge_sweep_mode else 0.10),  # shared-t0 sweep 不再逐 case 预 NVT。
            z_npt_ns=(0.0 if shared_t0_charge_sweep_mode else 1.00),  # shared-t0 sweep 不再逐 case z-NPT。
            z_compressibility_bar_inv=4.5e-6,  # final z-NPT 使用小有效压缩率，避免 slab 拼接体系出现大幅 z 缩放。
            z_npt_tau_p_ps=20.0,  # final z-NPT 使用慢 barostat，和循环压缩退火阶段一致。
            graphite_restraint=graphite_restraint,  # 全流程保留石墨 z-only 平整 restraint。
            interdiffusion_start=(False if shared_t0_charge_sweep_mode else interdiffusion_start),  # shared-t0 sweep 的 t=0 就是 assembled stack，不再加逐 case phase gate。
            compression_anneal=ZCompressionAnnealSpec(  # prepared CMC/electrolyte slab 已经分别致密预平衡，默认不再强循环压缩。
                enabled=False,  # 若诊断发现明显 vacuum-like span，再手动改 True 做温和循环压缩。
                cycles=4,  # 手动开启时使用少量小步压缩即可。
                tmax_K=360.0,  # 手动开启时只做温和退火，不再用高温强压实。
                pmax_bar=1500.0,  # 保留温和高压上限；比旧强压缩路线更保守。
                max_z_shrink_per_cycle=0.02,  # 手动开启时每轮最多压 2%，避免破坏两个 prepared slab。
                hot_nvt_ns=0.01,  # 手动开启时每轮热 NVT 时间。
                compression_npt_ns=0.04,  # 手动开启时每轮高压 z-NPT 时间。
                cool_nvt_ns=0.02,  # 手动开启时每轮冷却回正常温度的 NVT 时间。
                compression_tau_p_ps=20.0,  # 高压退火使用慢 barostat，使石墨/软相随盒子平稳移动。
                compression_z_compressibility_bar_inv=4.5e-6,  # 高压阶段用较小有效 z 压缩率，降低盒子突变风险。
                geometry_clash_check=True,  # 几何压缩后先筛查原子重叠，严重重叠时自动减半压缩比例。
                geometry_clash_cutoff_nm=0.10,  # 记录 0.10 nm 内的可疑非同残基近接触。
                severe_geometry_clash_cutoff_nm=0.065,  # 低于该距离视为高风险，避免直接进入 minimization。
                max_geometry_clash_retries=3,  # 最多把几何压缩比例连续减半 3 次；仍不安全则停止后续压缩。
            ),
            restart=restart_status,  # 允许长任务断点续跑。
            pre_relaxation=(not shared_t0_charge_sweep_mode),  # 关键：shared-t0 sweep 直接从同一 system.gro 进入 final NVT。
        )
        print(f"relaxation_summary = {relax.summary_path}")  # 打印关键路径或状态，便于人工检查。

        analy = relax.analyze()  # 设置中间变量或可调参数，供后续工作流使用。
        interface = analy.interface(  # 设置中间变量或可调参数，供后续工作流使用。
            manifest_path=result.manifest_path,  # final NVT 后处理仍用同一个层 manifest。
            analysis_profile=analysis_profile,  # 界面分析预设。
            frame_stride=analysis_frame_stride,  # 控制长轨迹抽帧，避免默认全帧扫描导致四个 case 后处理排队数小时。
            bin_nm=interface_bin_nm,  # z-profile bin 宽。
            region_width_nm=interface_region_width_nm,  # 区域宽度。
            surface_grid_nm=interface_surface_grid_nm,  # xy occupancy 网格。
            surface_distance_nm=graphite_adsorption_cutoff_nm,  # 吸附距离阈值。
            penetration_threshold_nm=penetration_threshold_nm,  # penetration 深度阈值。
            adsorption_min_residence_ps=adsorption_min_residence_ps,  # 吸附驻留阈值。
            potential_reference=potential_reference,  # 电势零点。
            penetration_species=penetration_species,  # penetration 物种。
            adsorption_species=adsorption_species,  # adsorption 物种。
            split_electrodes=split_electrodes_for_edl,  # 分开上下电极。
            report_potential_drop=report_potential_drop,  # 输出电势差。
            compute_transport=compute_interface_transport,  # 是否计算 MSD/扩散。
            time_series_sample_count=interface_time_series_sample_count,  # 时间序列窗口数。
            time_series_fps=interface_time_series_fps,  # MP4 帧率。
            time_series_rdf=interface_time_series_rdf,  # 控制全体系 RDF/CN 与 graphite-EDL RDF/CN 时间序列。
            time_series_concentration=interface_time_series_concentration,  # 是否输出浓度 profile 动画。
            time_series_angles=interface_time_series_angles,  # 是否输出角度分布动画。
            time_series_charge_potential=interface_time_series_charge_potential,  # 是否输出 graphite/EDL 电荷-电势动画。
        )
        health = interface.geometry_health(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        z_profile = interface.z_profiles(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        edl = interface.edl_profiles(  # 设置中间变量或可调参数，供后续工作流使用。
            split_electrodes=split_electrodes_for_edl,  # 设置中间变量或可调参数，供后续工作流使用。
            potential_reference=potential_reference,  # 设置电势 profile 的零点参考方式。
            report_potential_drop=report_potential_drop,  # 控制是否输出电势差诊断。
            time_series_analysis=time_series_analysis,  # 控制是否输出时间序列 CSV/MP4。
        )
        penetration = interface.penetration(species=penetration_species, time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        membrane = interface.membrane_permeation(species=penetration_species, time_series_analysis=time_series_analysis)  # 输出 CMCNA 膜/隔膜渗透诊断；包括进入事件、膜内驻留、uptake、表观 flux/permeability。
        adsorption = interface.graphite_adsorption(  # 设置中间变量或可调参数，供后续工作流使用。
            species=adsorption_species,  # 列出本层或本体系包含的分子对象，顺序要和 counts 对齐。
            time_series_analysis=time_series_analysis,  # 控制是否输出时间序列 CSV/MP4。
        )
        coordination = interface.coordination_by_region(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        transport = interface.region_transport(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        time_series = interface.time_series(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        summary = interface.summary(time_series_analysis=time_series_analysis)  # 设置中间变量或可调参数，供后续工作流使用。
        edl_rdf = (time_series.get("outputs") or {}).get("edl_rdf_cn") or {}  # 提取 graphite-EDL RDF/CN 输出，便于确认新统计是否产出。
        charge_potential = (time_series.get("outputs") or {}).get("charge_potential") or {}  # 提取带电 graphite/EDL 电荷-电势输出。
        print(f"interface_phase_order_ok = {health.get('phase_order_ok')}")  # 打印关键路径或状态，便于人工检查。
        print(f"membrane_permeation_available = {membrane.get('available')}")  # 确认膜渗透统计是否识别到 CMCNA/polymer 区域。
        print(f"interface_outputs = {summary.get('outputs', {}).get('interface_profile_summary_json')}")  # 打印关键路径或状态，便于人工检查。
        print(f"interface_time_series = {time_series.get('outputs', {})}")  # 打印关键路径或状态，便于人工检查。
        print(f"edl_rdf_cn_available = {edl_rdf.get('available')}")  # True 表示 graphite-EDL RDF/CN CSV/MP4 正常生成。
        print(f"edl_rdf_cn_outputs = {edl_rdf.get('curves_csv') or edl_rdf.get('reason')}")  # 打印曲线 CSV 路径或失败原因。
        print(f"charge_potential_available = {charge_potential.get('available')}")  # True 表示电荷-电势 CSV/MP4 正常生成。
        print(f"charge_potential_outputs = {charge_potential.get('csv') or charge_potential.get('reason')}")  # 打印电荷-电势 CSV 路径或失败原因。
        (case_dir / "charge_case_done.json").write_text(  # 写入并发后处理等待标记。
            json.dumps(
                {
                    "case_name": case_name,
                    "surface_charge_uC_cm2": float(surface_charge_uC_cm2),
                    "shared_t0_charge_sweep_mode": bool(shared_t0_charge_sweep_mode),
                    "sample_ns": float(sample_ns),
                    "gpu_id": int(gpu_id),
                    "omp": int(omp),
                    "relaxation_summary": str(relax.summary_path),
                    "interface_summary": str(summary.get("outputs", {}).get("interface_profile_summary_json")),
                    "li_solvation_depth_analysis": {
                        "enabled": bool(li_solvation_depth_analysis),
                        "computed_by": "make_charge_sweep_report_ppt.py",
                        "definition": "Li atoms inside CMC are binned by depth from the electrolyte-side CMC boundary; each bin reports mean coordination to solvent-O, CMC-O, and PF6-F sites.",
                        "cutoff_nm": float(li_solvation_depth_cutoff_nm),
                        "depth_bin_nm": float(li_solvation_depth_bin_nm),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    clean_md_trajectory_files(case_dir, enabled=clean_trajectories_after_analysis)  # 按配置清理 MD 轨迹文件。
