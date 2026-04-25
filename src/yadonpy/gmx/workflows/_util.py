"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..topology import parse_system_top


@dataclass
class _GroAtomRecord:
    resnr: int
    resname: str
    atomname: str
    atomnr: int
    xyz_nm: tuple[float, float, float]
    vxyz_nm_ps: tuple[float, float, float] | None = None


@dataclass
class _GroFrameRecord:
    title: str
    atoms: list[_GroAtomRecord]
    box_nm: tuple[float, float, float]


def _read_gro_frame(path: Path) -> _GroFrameRecord:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid .gro file: {path}")
    natoms = int(lines[1].strip())
    atoms: list[_GroAtomRecord] = []
    for idx in range(natoms):
        raw = lines[2 + idx]
        try:
            xyz = (
                float(raw[20:28]),
                float(raw[28:36]),
                float(raw[36:44]),
            )
        except Exception:
            if len(raw) < 24:
                raise
            xyz = (
                float(raw[-24:-16]),
                float(raw[-16:-8]),
                float(raw[-8:]),
            )
        atoms.append(
            _GroAtomRecord(
                resnr=int(raw[0:5].strip() or 0),
                resname=raw[5:10].strip() or "RES",
                atomname=raw[10:15].strip() or f"A{idx + 1}",
                atomnr=idx + 1,
                xyz_nm=xyz,
                vxyz_nm_ps=(
                    (
                        float(raw[44:52]),
                        float(raw[52:60]),
                        float(raw[60:68]),
                    )
                    if len(raw) >= 68 and raw[44:68].strip()
                    else None
                ),
            )
        )
    parts = lines[2 + natoms].split()
    if len(parts) < 3:
        raise ValueError(f"Invalid .gro box line: {path}")
    return _GroFrameRecord(
        title=lines[0].strip(),
        atoms=atoms,
        box_nm=(float(parts[0]), float(parts[1]), float(parts[2])),
    )


def _gro_wrap_index(value: int) -> int:
    value = int(value)
    if value < 0:
        return -((-value) % 100000)
    return value % 100000


def _write_gro_frame(path: Path, frame: _GroFrameRecord) -> None:
    lines = [frame.title[:80], f"{len(frame.atoms):5d}"]
    for atom in frame.atoms:
        x, y, z = atom.xyz_nm
        line = (
            f"{_gro_wrap_index(atom.resnr):5d}{str(atom.resname)[:5]:<5}{str(atom.atomname)[:5]:>5}{_gro_wrap_index(atom.atomnr):5d}"
            f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
        )
        if atom.vxyz_nm_ps is not None:
            vx, vy, vz = atom.vxyz_nm_ps
            line += f"{float(vx):8.4f}{float(vy):8.4f}{float(vz):8.4f}"
        lines.append(line)
    lines.append(f"{frame.box_nm[0]:10.5f}{frame.box_nm[1]:10.5f}{frame.box_nm[2]:10.5f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _minimal_image_delta(delta: float, box_len: float) -> float:
    if box_len <= 0.0:
        return float(delta)
    return float(delta - box_len * round(float(delta) / float(box_len)))


def _unwrap_fragment_coords(
    coords_nm: list[tuple[float, float, float]],
    *,
    bonds: list[tuple[int, int]],
    box_nm: tuple[float, float, float],
) -> list[list[float]]:
    coords = [[float(x), float(y), float(z)] for x, y, z in coords_nm]
    if len(coords) <= 1 or not bonds:
        return coords

    out = [list(xyz) for xyz in coords]
    adjacency: list[list[int]] = [[] for _ in range(len(coords))]
    for ai, aj in bonds:
        i = int(ai) - 1
        j = int(aj) - 1
        if i < 0 or j < 0 or i >= len(coords) or j >= len(coords):
            continue
        adjacency[i].append(j)
        adjacency[j].append(i)

    seen = [False] * len(coords)
    for root in range(len(coords)):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            idx = stack.pop()
            for nxt in adjacency[idx]:
                if seen[nxt]:
                    continue
                # Anchor each new atom to the already-unwrapped coordinate of
                # the current atom.  Using the original wrapped coordinate here
                # works for a single crossing, but can re-split multi-bond
                # molecules when the second/third bond also crosses PBC.
                delta = [coords[nxt][dim] - out[idx][dim] for dim in range(3)]
                for dim, box_len in enumerate(box_nm):
                    delta[dim] = _minimal_image_delta(delta[dim], float(box_len))
                out[nxt] = [out[idx][dim] + delta[dim] for dim in range(3)]
                seen[nxt] = True
                stack.append(nxt)
    return out


def _wrap_molecule_com_into_box(coords_nm: list[list[float]], box_nm: tuple[float, float, float]) -> list[list[float]]:
    if not coords_nm:
        return coords_nm
    out = [list(xyz) for xyz in coords_nm]
    natoms = float(len(out))
    for dim, box_len in enumerate(box_nm):
        box_len = float(box_len)
        if box_len <= 0.0:
            continue
        com = sum(float(xyz[dim]) for xyz in out) / natoms
        shift = box_len * math.floor(com / box_len)
        if shift == 0.0:
            continue
        for xyz in out:
            xyz[dim] -= shift
    return out


def normalize_gro_molecules_inplace(
    *,
    top: Path,
    gro: Path,
) -> dict[str, Any]:
    """Rewrite a GRO file so topology-defined molecules are geometrically whole.

    This is a best-effort canonicalization step for stage-to-stage handoff. It
    does not alter topology or box vectors; it only unwraps bonded fragments via
    the system topology and keeps each molecule COM close to the primary box.
    """

    top = Path(top)
    gro = Path(gro)
    if (not top.exists()) or (not gro.exists()):
        return {"applied": False, "error": "missing top or gro", "normalized_molecules": 0}
    try:
        topo = parse_system_top(top)
        frame = _read_gro_frame(gro)
    except Exception as exc:
        return {"applied": False, "error": str(exc), "normalized_molecules": 0}

    atoms_out: list[_GroAtomRecord] = []
    cursor = 0
    normalized_molecules = 0
    try:
        for molname, count in topo.molecules:
            moltype = topo.moleculetypes.get(str(molname))
            if moltype is None:
                raise KeyError(f"Molecule type '{molname}' not found in topology")
            natoms = int(moltype.natoms)
            for _ in range(int(count)):
                block = frame.atoms[cursor: cursor + natoms]
                if len(block) != natoms:
                    raise ValueError(
                        f"Atom count mismatch while canonicalizing {molname}: expected {natoms}, got {len(block)}"
                    )
                coords = _unwrap_fragment_coords(
                    [tuple(atom.xyz_nm) for atom in block],
                    bonds=list(moltype.bonds),
                    box_nm=frame.box_nm,
                )
                if natoms > 1 and moltype.bonds:
                    coords = _wrap_molecule_com_into_box(coords, frame.box_nm)
                    normalized_molecules += 1
                for atom, xyz in zip(block, coords):
                    atoms_out.append(
                        _GroAtomRecord(
                            resnr=atom.resnr,
                            resname=atom.resname,
                            atomname=atom.atomname,
                            atomnr=atom.atomnr,
                            xyz_nm=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                            vxyz_nm_ps=atom.vxyz_nm_ps,
                        )
                    )
                cursor += natoms
        if cursor != len(frame.atoms):
            raise ValueError(f"Unparsed atoms remain in {gro}: parsed {cursor}, total {len(frame.atoms)}")
        title = frame.title
        if "yadonpy_whole" not in title:
            title = f"{title[:63]} | yadonpy_whole"
        _write_gro_frame(gro, _GroFrameRecord(title=title, atoms=atoms_out, box_nm=frame.box_nm))
        return {"applied": bool(normalized_molecules > 0), "error": None, "normalized_molecules": int(normalized_molecules)}
    except Exception as exc:
        return {"applied": False, "error": str(exc), "normalized_molecules": 0}


def gro_topology_bond_geometry(
    *,
    top: Path,
    gro: Path,
    max_bond_nm_threshold: float = 0.8,
) -> dict[str, Any]:
    """Check direct topology bond lengths in a GRO handoff structure.

    GROMACS can fail at step 0 if a handoff GRO contains a molecule split across
    the periodic boundary: LINCS then sees a normal X-H bond as a box-length
    constraint. This lightweight check catches that before the next stage starts.
    """

    top = Path(top)
    gro = Path(gro)
    if (not top.exists()) or (not gro.exists()):
        return {"ok": False, "error": "missing top or gro"}
    try:
        topo = parse_system_top(top)
        frame = _read_gro_frame(gro)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    cursor = 0
    max_bond_nm = 0.0
    worst: dict[str, Any] | None = None
    checked = 0
    try:
        for molname, count in topo.molecules:
            moltype = topo.moleculetypes.get(str(molname))
            if moltype is None:
                raise KeyError(f"Molecule type '{molname}' not found in topology")
            natoms = int(moltype.natoms)
            for mol_idx in range(int(count)):
                block = frame.atoms[cursor: cursor + natoms]
                if len(block) != natoms:
                    raise ValueError(
                        f"Atom count mismatch while checking {molname}: expected {natoms}, got {len(block)}"
                    )
                for ai, aj in moltype.bonds:
                    i = int(ai) - 1
                    j = int(aj) - 1
                    if i < 0 or j < 0 or i >= len(block) or j >= len(block):
                        continue
                    xi = block[i].xyz_nm
                    xj = block[j].xyz_nm
                    dist = math.sqrt(sum((float(xj[dim]) - float(xi[dim])) ** 2 for dim in range(3)))
                    checked += 1
                    if dist > max_bond_nm:
                        max_bond_nm = float(dist)
                        worst = {
                            "molname": str(molname),
                            "mol_index": int(mol_idx),
                            "atom_i": int(cursor + i + 1),
                            "atom_j": int(cursor + j + 1),
                            "bond_i": int(ai),
                            "bond_j": int(aj),
                            "distance_nm": float(dist),
                        }
                cursor += natoms
        if cursor != len(frame.atoms):
            raise ValueError(f"Unparsed atoms remain in {gro}: parsed {cursor}, total {len(frame.atoms)}")
    except Exception as exc:
        return {"ok": False, "error": str(exc), "checked_bonds": int(checked), "max_bond_nm": float(max_bond_nm)}

    threshold = float(max_bond_nm_threshold)
    return {
        "ok": bool(max_bond_nm <= threshold),
        "checked_bonds": int(checked),
        "max_bond_nm": float(max_bond_nm),
        "threshold_nm": threshold,
        "worst": worst,
    }


def pbc_mol_fix_inplace(
    runner: Any,
    *,
    tpr: Path,
    traj_or_gro: Path,
    cwd: Optional[Path] = None,
    group: str = "0",
    pbc: str = "mol",
    # NOTE (2026-02): For robustness and reproducibility, YadonPy now applies
    # only `-pbc mol` during the standard post-processing pass. We intentionally
    # do NOT apply `-center` or `-ur`, because they can change the visual frame
    # of reference and may confuse downstream tooling / comparisons.
    center: bool = False,
    ur: Optional[str] = None,
) -> dict[str, Any]:
    """Best-effort: rewrite a structure/trajectory with sane PBC handling.

    This is primarily for *visualization* and downstream conversion (mol2/xyz),
    where wrapped coordinates can make bonds *look* broken.

    Uses:
      gmx trjconv -pbc mol

    Notes
    -----
    - This does not change topology; it only rewrites coordinates.
    - The function is intentionally best-effort and never raises.
    """

    traj_or_gro = Path(traj_or_gro)
    tpr = Path(tpr)
    if (not tpr.exists()) or (not traj_or_gro.exists()):
        return {"applied": False, "error": "missing tpr or input file"}

    keep_copy = str(os.environ.get("YADONPY_KEEP_PBC_COPY", "")).strip().lower() in {"1", "true", "yes", "on"}
    tmp = traj_or_gro.with_name(traj_or_gro.name + ".pbc_tmp" + traj_or_gro.suffix)
    try:
        runner.trjconv(
            tpr=tpr,
            xtc=traj_or_gro,
            out=tmp,
            pbc=pbc,
            center=False,
            ur=None,
            group=str(group),
            cwd=cwd,
        )
        # Some GROMACS builds may exit with code 0 but still fail to write output
        # (e.g., due to selection prompts / permission / short trajectory). Guard
        # against tmp missing to avoid raising on `replace`.
        if not tmp.exists() or tmp.stat().st_size == 0:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return {"applied": False, "error": f"trjconv produced no output: {tmp.name}"}
        # Also overwrite the original in-place so downstream steps that expect
        # canonical filenames continue to work.
        try:
            import shutil

            shutil.copyfile(tmp, traj_or_gro)
        except Exception:
            # Best-effort: keep tmp even if in-place overwrite fails.
            pass
        pbc_copy = str(tmp) if keep_copy else None
        if not keep_copy:
            try:
                tmp.unlink()
            except Exception:
                pass
        return {"applied": True, "error": None, "pbc_copy": pbc_copy}
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return {"applied": False, "error": str(e), "pbc_copy": None}


def read_gro_box_nm(gro: Path) -> tuple[float, float, float]:
    """Read box lengths (nm) from the last line of a .gro file."""
    last = gro.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1]
    parts = last.split()
    if len(parts) < 3:
        raise ValueError(f"Cannot parse box line from gro: {gro}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RunResources:
    ntomp: Optional[int] = None
    ntmpi: Optional[int] = None
    # GPU execution switch: True enables GPU if supported by the cluster/GROMACS build.
    # This is separate from gpu_id so users can explicitly disable GPU while still
    # keeping a stable gpu_id parameter in scripts.
    use_gpu: bool = True
    gpu_id: Optional[str] = None
    gpu_offload_mode: str = "full"
