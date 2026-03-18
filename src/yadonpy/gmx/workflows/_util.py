"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def pbc_mol_fix_inplace(
    runner: Any,
    *,
    tpr: Path,
    traj_or_gro: Path,
    cwd: Optional[Path] = None,
    group: str = "0",
    pbc: str = "mol",
    center: bool = True,
    ur: str = "compact",
) -> dict[str, Any]:
    """Best-effort: rewrite a structure/trajectory in-place with sane PBC handling.

    This is primarily for *visualization* and downstream conversion (mol2/xyz),
    where wrapped coordinates can make bonds *look* broken.

    Uses:
      gmx trjconv -pbc mol -center -ur compact

    Notes
    -----
    - This does not change topology; it only rewrites coordinates.
    - The function is intentionally best-effort and never raises.
    """

    traj_or_gro = Path(traj_or_gro)
    tpr = Path(tpr)
    if (not tpr.exists()) or (not traj_or_gro.exists()):
        return {"applied": False, "error": "missing tpr or input file"}

    tmp = traj_or_gro.with_name(traj_or_gro.name + ".pbc_tmp")
    try:
        runner.trjconv(
            tpr=tpr,
            xtc=traj_or_gro,
            out=tmp,
            pbc=pbc,
            center=center,
            ur=ur,
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
        tmp.replace(traj_or_gro)
        return {"applied": True, "error": None}
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return {"applied": False, "error": str(e)}


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