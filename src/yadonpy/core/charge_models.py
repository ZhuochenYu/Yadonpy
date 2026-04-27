"""Lightweight non-DFT charge-model backends.

Supported families:
- CM5 / <scale>*CM5 via xTB GFN1-xTB
- CM1A / <scale>*CM1A via LigParGen/BOSS

These backends are intentionally thin wrappers around external tools instead of
re-implementing the original semiempirical methods inside YadonPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
from typing import Iterable

from rdkit import Chem

from . import utils


@dataclass(frozen=True)
class ChargeModelSpec:
    base: str
    scale: float
    label: str


_CHARGE_MODEL_RE = re.compile(
    r"^\s*(?:(?P<scale>[0-9]+(?:\.[0-9]+)?)\s*\*\s*)?(?P<base>cm1a|cm5)\s*$",
    re.IGNORECASE,
)


def parse_charge_model_spec(spec: str | None) -> ChargeModelSpec | None:
    if spec is None:
        return None
    m = _CHARGE_MODEL_RE.match(str(spec))
    if not m:
        return None
    base = m.group("base").upper()
    scale = float(m.group("scale") or 1.0)
    label = f"{scale:g}*{base}" if abs(scale - 1.0) > 1e-12 else base
    return ChargeModelSpec(base=base, scale=scale, label=label)


def is_quick_charge_model(spec: str | None) -> bool:
    return parse_charge_model_spec(spec) is not None


def supported_quick_charge_methods() -> tuple[str, ...]:
    return (
        "CM1A",
        "1.14*CM1A",
        "<scale>*CM1A",
        "CM5",
        "1.2*CM5",
        "<scale>*CM5",
    )


def assign_quick_charges(
    mol,
    *,
    charge: str,
    confId: int = 0,
    opt: bool = False,
    work_dir: str | os.PathLike | None = None,
    tmp_dir: str | os.PathLike | None = None,
    log_name: str | None = None,
    total_charge: int | None = None,
    total_multiplicity: int | None = None,
    **kwargs,
) -> bool:
    spec = parse_charge_model_spec(charge)
    if spec is None:
        raise ValueError(f"Unsupported quick charge model: {charge}")

    if spec.base == "CM5":
        charges = _compute_cm5_with_xtb(
            mol,
            confId=confId,
            opt=opt,
            work_dir=work_dir,
            tmp_dir=tmp_dir,
            log_name=log_name,
            total_charge=total_charge,
            total_multiplicity=total_multiplicity,
        )
        scaled = [float(q) * float(spec.scale) for q in charges]
        _apply_atomic_charges(mol, scaled, label=spec.label)
        return True

    if spec.base == "CM1A":
        charges = _compute_cm1a_with_ligpargen(
            mol,
            confId=confId,
            work_dir=work_dir,
            tmp_dir=tmp_dir,
            log_name=log_name,
            total_charge=total_charge,
        )
        # LigParGen documents that neutral CM1A is automatically scaled by 1.14.
        backend_scale = 1.14 if _infer_total_charge(mol, total_charge=total_charge) == 0 else 1.0
        factor = float(spec.scale) / float(backend_scale)
        scaled = [float(q) * factor for q in charges]
        _apply_atomic_charges(mol, scaled, label=spec.label)
        return True

    raise ValueError(f"Unsupported quick charge model base: {spec.base}")


def _apply_atomic_charges(mol, charges: Iterable[float], *, label: str) -> None:
    charges = list(charges)
    if len(charges) != mol.GetNumAtoms():
        raise ValueError(f"Charge count mismatch: got {len(charges)}, expected {mol.GetNumAtoms()}")
    for i, q in enumerate(charges):
        atom = mol.GetAtomWithIdx(i)
        atom.SetDoubleProp("AtomicCharge", float(q))
        atom.SetDoubleProp(label, float(q))
        atom.SetProp("charge_model", str(label))
    try:
        mol.SetProp("charge_model", str(label))
    except Exception:
        pass


def _infer_total_charge(mol, *, total_charge: int | None = None) -> int:
    if isinstance(total_charge, int):
        return int(total_charge)
    fc = 0
    for atom in mol.GetAtoms():
        fc += int(atom.GetFormalCharge())
    return int(fc)


def _infer_total_multiplicity(mol, *, total_multiplicity: int | None = None) -> int:
    if isinstance(total_multiplicity, int):
        return int(total_multiplicity)
    n_rad = 0
    for atom in mol.GetAtoms():
        n_rad += int(atom.GetNumRadicalElectrons())
    return int(max(1, n_rad + 1))


def _ensure_xyz_input(mol, *, confId: int, xyz_path: Path) -> None:
    try:
        if mol.GetNumConformers() == 0:
            utils.ensure_3d_coords(mol, engine="auto")
    except Exception:
        pass
    if mol.GetNumConformers() == 0:
        raise RuntimeError("No 3D conformer available for charge calculation")
    conf = mol.GetConformer(confId)
    lines = [str(mol.GetNumAtoms()), str(mol.GetProp("_Name")) if mol.HasProp("_Name") else "yadonpy"]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {pos.x:.10f} {pos.y:.10f} {pos.z:.10f}")
    xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _working_directory(*, work_dir=None, tmp_dir=None, log_name=None, prefix="charge_model") -> Path:
    base = Path(tmp_dir or work_dir or tempfile.mkdtemp(prefix="yadonpy_"))
    name = str(log_name or prefix).strip() or prefix
    path = base / f"{prefix}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compute_cm5_with_xtb(
    mol,
    *,
    confId: int,
    opt: bool,
    work_dir,
    tmp_dir,
    log_name,
    total_charge,
    total_multiplicity,
) -> list[float]:
    xtb = shutil.which("xtb")
    if xtb is None:
        raise RuntimeError("CM5 charges require the xTB executable in PATH (GFN1-xTB backend).")

    run_dir = _working_directory(work_dir=work_dir, tmp_dir=tmp_dir, log_name=log_name, prefix="cm5")
    xyz_path = run_dir / "input.xyz"
    _ensure_xyz_input(mol, confId=confId, xyz_path=xyz_path)

    chrg = _infer_total_charge(mol, total_charge=total_charge)
    mult = _infer_total_multiplicity(mol, total_multiplicity=total_multiplicity)
    uhf = max(0, int(mult) - 1)

    cmd = [xtb, xyz_path.name, "--gfn", "1", "--pop", "--chrg", str(chrg), "--uhf", str(uhf)]
    if opt:
        cmd.append("--opt")

    proc = subprocess.run(
        cmd,
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (run_dir / "xtb_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"xTB failed while computing CM5 charges (exit={proc.returncode}). See {run_dir / 'xtb_stdout.log'}."
        )

    charges = _parse_xtb_cm5_output(proc.stdout, expected_atoms=mol.GetNumAtoms())
    if len(charges) != mol.GetNumAtoms():
        raise RuntimeError(f"Parsed {len(charges)} CM5 charges from xTB, expected {mol.GetNumAtoms()}.")
    return charges


def _parse_xtb_cm5_output(text: str, *, expected_atoms: int) -> list[float]:
    charges: list[float] = []
    in_block = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if "Mulliken/CM5 charges" in line:
            in_block = True
            charges = []
            continue
        if not in_block:
            continue
        if not line.strip():
            if charges:
                break
            continue
        m = re.match(r"^\s*(\d+)\s+([A-Za-z][A-Za-z]?)\s+([\-0-9.Ee+]+)\s+([\-0-9.Ee+]+)", line)
        if not m:
            if charges:
                break
            continue
        charges.append(float(m.group(4)))
        if len(charges) >= expected_atoms:
            break
    if len(charges) != expected_atoms:
        raise RuntimeError("Could not parse the CM5 charge block from xTB output.")
    return charges


def _compute_cm1a_with_ligpargen(
    mol,
    *,
    confId: int,
    work_dir,
    tmp_dir,
    log_name,
    total_charge,
) -> list[float]:
    ligpargen = shutil.which("ligpargen")
    if ligpargen is None:
        raise RuntimeError("CM1A charges require the ligpargen executable in PATH (LigParGen/BOSS backend).")

    run_dir = _working_directory(work_dir=work_dir, tmp_dir=tmp_dir, log_name=log_name, prefix="cm1a")
    smiles = _best_effort_smiles(mol)
    name = _best_effort_name(mol, fallback="mol")
    residue = re.sub(r"[^A-Za-z0-9]", "", name.upper())[:3] or "MOL"
    net_charge = _infer_total_charge(mol, total_charge=total_charge)

    cmd = [
        ligpargen,
        "-s", smiles,
        "-n", name,
        "-p", str(run_dir),
        "-r", residue,
        "-c", str(net_charge),
        "-o", "0",
        "-cgen", "CM1A",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (run_dir / "ligpargen_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"LigParGen failed while computing CM1A charges (exit={proc.returncode}). See {run_dir / 'ligpargen_stdout.log'}."
        )

    itp_files = sorted(run_dir.rglob("*.itp"))
    if not itp_files:
        raise RuntimeError(f"LigParGen finished but no .itp file was found in {run_dir}.")

    charges = None
    last_error = None
    for itp_path in itp_files:
        try:
            candidate = _parse_gromacs_itp_charges(itp_path)
            if len(candidate) == mol.GetNumAtoms():
                charges = candidate
                break
        except Exception as exc:
            last_error = exc
    if charges is None:
        raise RuntimeError(
            f"Could not extract {mol.GetNumAtoms()} atom charges from LigParGen output under {run_dir}. "
            f"Last parser error: {last_error}"
        )
    return charges


def _parse_gromacs_itp_charges(path: Path) -> list[float]:
    charges: list[float] = []
    in_atoms = False
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_atoms = line.lower().replace(" ", "") == "[atoms]"
            continue
        if not in_atoms:
            continue
        fields = line.split()
        if len(fields) < 7:
            continue
        if not fields[0].isdigit():
            continue
        charges.append(float(fields[6]))
    if not charges:
        raise RuntimeError(f"No [ atoms ] charges found in {path}")
    return charges


def _best_effort_name(mol, fallback: str = "mol") -> str:
    try:
        name = utils.get_name(mol, default=None)
        if name:
            return str(name)
    except Exception:
        pass
    try:
        if mol.HasProp("_Name"):
            return str(mol.GetProp("_Name"))
    except Exception:
        pass
    return str(fallback)


def _best_effort_smiles(mol) -> str:
    try:
        if mol.HasProp("_yadonpy_input_smiles"):
            return mol.GetProp("_yadonpy_input_smiles")
    except Exception:
        pass
    try:
        if mol.HasProp("_yadonpy_smiles"):
            return mol.GetProp("_yadonpy_smiles")
    except Exception:
        pass
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
