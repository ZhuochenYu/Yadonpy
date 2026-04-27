"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""
from __future__ import annotations

#  Copyright (c) 2026. YadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# sim.qm module
# ******************************************************************************

import numpy as np
import os
import json
import gc
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from rdkit import Chem
from rdkit import Geometry as Geom
from ..core import chem_utils as core_utils
from ..core import utils, const, calc
from ..core.metadata import QM_RECIPE_PROP, RESP_PROFILE_PROP, write_json_prop, write_text_prop
from ..core.logging_utils import format_elapsed as _fmt_elapsed
from ..runtime import resolve_restart
from .qm_wrapper import QMw
from . import seminario


_RESP_PROFILES = {"adaptive", "legacy"}
_DEFAULT_OPT_METHOD = "wb97m-d3bj"
_DEFAULT_OPT_BASIS = "def2-SVP"
_DEFAULT_CHARGE_METHOD = "wb97m-d3bj"
_DEFAULT_CHARGE_BASIS = "def2-TZVP"
_CARBONATE_RECIPE = {
    "opt_method": "wb97m-v",
    "charge_method": "wb97m-v",
    "opt_basis": "def2-TZVP",
    "charge_basis": "def2-TZVP",
}


def _qm_log(message: str, *, level: int = 1) -> None:
    utils.yadon_print(f"[QM] {message}", level=level)


def _qm_smiles(mol) -> str:
    try:
        if mol.HasProp('_yadonpy_input_smiles'):
            return mol.GetProp('_yadonpy_input_smiles')
        if mol.HasProp('_yadonpy_smiles'):
            return mol.GetProp('_yadonpy_smiles')
        return Chem.MolToSmiles(mol)
    except Exception:
        return '?'


def _qm_begin(title: str, *, log_name: str, mol=None, detail: str | None = None) -> float:
    import time as _time
    _qm_log('=' * 88, level=1)
    _qm_log(f"[SECTION] {title}", level=1)
    _qm_log(f"[ITEM] purpose           : {log_name}", level=1)
    if mol is not None:
        _qm_log(f"[ITEM] smiles            : {_qm_smiles(mol)}", level=1)
    if detail:
        _qm_log(f"[NOTE] {detail}", level=1)
    return _time.perf_counter()


def _qm_done(title: str, t0: float, *, detail: str | None = None) -> None:
    import time as _time
    msg = f"[DONE] {title} | elapsed={_fmt_elapsed(_time.perf_counter() - float(t0))}"
    if detail:
        msg += f" | {detail}"
    _qm_log(msg, level=1)
    _qm_log('=' * 88, level=1)


def ensure_has_conformer(mol) -> None:
    try:
        if int(mol.GetNumConformers()) > 0:
            return
    except Exception:
        return
    conf = Chem.Conformer(int(mol.GetNumAtoms()))
    for i in range(int(mol.GetNumAtoms())):
        conf.SetAtomPosition(i, Geom.Point3D(0.0, 0.0, 0.0))
    mol.AddConformer(conf, assignId=True)


def is_h_terminator_placeholder(mol, *, smiles_hint: str | None = None) -> bool:
    """Return True for the special H/H linker placeholder used by ``[H][*]``.

    YadonPy stores ``*`` connection points as isotope-labeled H atoms. Most
    terminal groups like ``*C`` or ``*O`` therefore become normal closed-shell
    molecules and can proceed through QM/RESP as usual. The pure hydrogen
    terminator is different: ``[H][*]`` becomes a two-atom H/H placeholder that
    exists only to mark the linker pattern during polymer termination.
    """
    try:
        if int(mol.GetNumAtoms()) != 2 or int(mol.GetNumBonds()) != 1:
            return False
        atoms = list(mol.GetAtoms())
        if any(a.GetSymbol() != "H" for a in atoms):
            return False
        if any(int(a.GetFormalCharge()) != 0 for a in atoms):
            return False
        if any(int(a.GetNumRadicalElectrons()) != 0 for a in atoms):
            return False
        if any(int(a.GetTotalDegree()) > 1 for a in atoms):
            return False
        return any(int(a.GetIsotope()) >= 3 for a in atoms)
    except Exception:
        return False


def apply_placeholder_zero_charges(mol, *, charge_label: str = "RESP") -> None:
    """Assign stable zero charges to placeholder/linker-only fragments."""
    label = str(charge_label or "").strip().upper()
    ensure_has_conformer(mol)
    for atom in mol.GetAtoms():
        atom.SetDoubleProp("AtomicCharge", 0.0)
        atom.SetDoubleProp("AtomicCharge_raw", 0.0)
        if label in ("RESP", "ESP"):
            atom.SetDoubleProp("RESP", 0.0)
            atom.SetDoubleProp("RESP_raw", 0.0)
            atom.SetDoubleProp("ESP", 0.0)
            atom.SetDoubleProp("ESP_raw", 0.0)
        elif label == "MULLIKEN":
            atom.SetDoubleProp("MullikenCharge", 0.0)
        elif label == "LOWDIN":
            atom.SetDoubleProp("LowdinCharge", 0.0)


def _normalize_resp_profile(resp_profile: str | None) -> str:
    profile = str(resp_profile or "adaptive").strip().lower()
    if profile in {"default", "current"}:
        profile = "adaptive"
    if profile not in _RESP_PROFILES:
        raise ValueError(f"Unsupported RESP profile: {resp_profile!r}")
    return profile


def _json_prop(mol, key: str) -> dict[str, Any] | None:
    try:
        if mol.HasProp(key):
            value = json.loads(mol.GetProp(key))
            if isinstance(value, dict):
                return value
    except Exception:
        pass
    return None


def _atom_heavy_neighbor_count(atom, *, exclude_idx: int | None = None) -> int:
    count = 0
    for nb in atom.GetNeighbors():
        if exclude_idx is not None and int(nb.GetIdx()) == int(exclude_idx):
            continue
        if nb.GetAtomicNum() > 1:
            count += 1
    return count


def _is_neutral_carbonate_like(mol) -> bool:
    try:
        total_q = sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms())
    except Exception:
        return False
    if int(total_q) != 0:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        oxygen_double = []
        oxygen_single = []
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetSymbol() != "O":
                continue
            if bond.GetBondTypeAsDouble() >= 1.5:
                oxygen_double.append(other)
            elif abs(float(bond.GetBondTypeAsDouble()) - 1.0) < 1.0e-8:
                oxygen_single.append(other)
        if len(oxygen_double) != 1 or len(oxygen_single) != 2:
            continue
        if any(_atom_heavy_neighbor_count(oxygen, exclude_idx=atom.GetIdx()) < 1 for oxygen in oxygen_single):
            continue
        return True
    return False


def _basis_gen_with_override(base: dict[str, str] | None, basis: str) -> dict[str, str]:
    out = dict(base or {})
    out.update({"Br": str(basis), "I": str(basis)})
    return out


def _resolve_resp_qm_recipe(
    mol,
    *,
    resp_profile: str,
    charge_model: str,
    opt_method: str,
    opt_basis: str,
    opt_basis_gen: dict[str, str] | None,
    charge_method: str,
    charge_basis: str,
    charge_basis_gen: dict[str, str] | None,
    auto_level: bool,
    total_charge: int,
) -> dict[str, Any]:
    profile = _normalize_resp_profile(resp_profile)
    charge_model_up = str(charge_model or "").strip().upper()
    recipe = {
        "resp_profile": profile,
        "charge_model": charge_model_up,
        "opt_method": str(opt_method),
        "opt_basis": str(opt_basis),
        "charge_method": str(charge_method),
        "charge_basis": str(charge_basis),
        "opt_basis_gen": dict(opt_basis_gen or {}),
        "charge_basis_gen": dict(charge_basis_gen or {}),
        "auto_level": bool(auto_level),
        "anion_diffuse_upgrade": False,
        "adaptive_carbonate_recipe": False,
        "carbonate_like": False,
    }

    if bool(auto_level) and int(total_charge) < 0:
        if str(recipe["opt_basis"]).strip().lower() == "def2-svp":
            recipe["opt_basis"] = "def2-SVPD"
            recipe["opt_basis_gen"] = _basis_gen_with_override(recipe.get("opt_basis_gen"), "def2-SVPD")
            recipe["anion_diffuse_upgrade"] = True
        if str(recipe["charge_basis"]).strip().lower() == "def2-tzvp":
            recipe["charge_basis"] = "def2-TZVPD"
            recipe["charge_basis_gen"] = _basis_gen_with_override(recipe.get("charge_basis_gen"), "def2-TZVPD")
            recipe["anion_diffuse_upgrade"] = True

    carbonate_like = False
    if charge_model_up in {"RESP", "RESP2", "ESP"}:
        carbonate_like = _is_neutral_carbonate_like(mol)
    recipe["carbonate_like"] = bool(carbonate_like)
    if profile == "adaptive" and carbonate_like and int(total_charge) == 0:
        if str(recipe["opt_method"]).strip().lower() == _DEFAULT_OPT_METHOD:
            recipe["opt_method"] = _CARBONATE_RECIPE["opt_method"]
        if str(recipe["charge_method"]).strip().lower() == _DEFAULT_CHARGE_METHOD:
            recipe["charge_method"] = _CARBONATE_RECIPE["charge_method"]
        if str(recipe["opt_basis"]).strip().lower() == _DEFAULT_OPT_BASIS.lower():
            recipe["opt_basis"] = _CARBONATE_RECIPE["opt_basis"]
            recipe["opt_basis_gen"] = _basis_gen_with_override(recipe.get("opt_basis_gen"), _CARBONATE_RECIPE["opt_basis"])
        if str(recipe["charge_basis"]).strip().lower() == _DEFAULT_CHARGE_BASIS.lower():
            recipe["charge_basis"] = _CARBONATE_RECIPE["charge_basis"]
            recipe["charge_basis_gen"] = _basis_gen_with_override(recipe.get("charge_basis_gen"), _CARBONATE_RECIPE["charge_basis"])
        recipe["adaptive_carbonate_recipe"] = True

    return recipe


def _stamp_resp_recipe_metadata(mol, *, resp_profile: str, recipe: dict[str, Any]) -> None:
    try:
        write_text_prop(mol, RESP_PROFILE_PROP, str(resp_profile))
        write_json_prop(mol, QM_RECIPE_PROP, dict(recipe))
    except Exception:
        pass


def _get_psiresp_equivalence_groups(mol) -> list[list[int]]:
    return core_utils.resp_equivalence_groups_from_mol(mol)


def _symmetry_repair_props(mol) -> list[str]:
    props = []
    for base in ("AtomicCharge", "RESP", "RESP2", "ESP"):
        if any(atom.HasProp(base) for atom in mol.GetAtoms()):
            props.append(base)
        raw_key = f"{base}_raw"
        if any(atom.HasProp(raw_key) for atom in mol.GetAtoms()):
            props.append(raw_key)
    return props


def _symmetrize_charge_properties(mol, *, equivalence_groups: list[list[int]]) -> int:
    return core_utils.symmetrize_equivalent_charge_props(mol, equivalence_groups=equivalence_groups)


def _sanitize_dirname(name: str) -> str:
    """Sanitize a user-supplied name into a filesystem-friendly directory name."""
    s = str(name) if name is not None else ""
    s = s.strip() or "unnamed"
    # Replace path separators and whitespace with underscores.
    s = s.replace(os.sep, "_")
    s = s.replace("/", "_")
    s = re.sub(r"\s+", "_", s)
    # Keep a conservative charset.
    s = re.sub(r"[^A-Za-z0-9_.\-]", "_", s)
    return s.strip("_") or "default"


def _task_dirname(task: str) -> str:
    """Map a logical QM task name to a numbered folder name.

    This keeps work_dir/01_qm organized and sortable (like yzc-gmx-gen).
    """
    t = str(task or "task").strip()
    if not t:
        return "99_task"

    # If the caller already provides a numbered prefix, respect it.
    if re.match(r"^\d{2,}_", t):
        return _sanitize_dirname(t)

    tl = t.lower()
    mapping = {
        "confsearch": "01_confsearch",
        "conformation_search": "01_confsearch",
        "charge": "02_charge",
        "assign_charges": "02_charge",
        "resp": "02_charge",
        "sp_prop": "03_sp_prop",
        "polarizability": "04_polarizability",
        "refractive_index": "05_refractive_index",
        "abbe_number_cc2": "06_abbe_number_cc2",
        "bonded_params": "07_bonded_params",
        "bond_angle": "07_bonded_params",
        "bond_angle_params": "07_bonded_params",
    }
    if tl in mapping:
        return mapping[tl]
    return f"99_{_sanitize_dirname(t)}"


def _qm_task_dir(work_dir, *, log_name: str, task: str) -> tuple[Path, Path]:
    """Return (root_work_dir, task_work_dir) for QM outputs.

    To keep `work_dir/` clean, YadonPy places QM artifacts under:
      work_dir/01_qm/<NN_task>/<log_name>/

    This mirrors the "module folders" idea used in yzc-gmx-gen.
    """
    root = Path(work_dir)
    qm_root = root / "01_qm"
    task_dir = qm_root / _task_dirname(task) / _sanitize_dirname(log_name)
    task_dir.mkdir(parents=True, exist_ok=True)
    return root, task_dir


def _save_atomic_charges_json(mol, path, *, charge_label: str, log_name: str, extra_meta: dict[str, Any] | None = None):
    """Persist per-atom charges to JSON for resumable workflows.

    Charges are stored in RDKit atom double-prop 'AtomicCharge'.
    """
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    charges = []
    for a in mol.GetAtoms():
        try:
            charges.append(float(a.GetDoubleProp('AtomicCharge')))
        except Exception:
            charges.append(0.0)
    meta = {
        "log_name": str(log_name),
        "charge": str(charge_label),
        "num_atoms": int(mol.GetNumAtoms()),
    }
    try:
        # store SMILES if present
        if mol.HasProp('_yadonpy_smiles'):
            meta["smiles"] = mol.GetProp('_yadonpy_smiles')
    except Exception:
        pass
    if extra_meta:
        meta.update(dict(extra_meta))

    # IMPORTANT: persist (m)Seminario patch metadata so that a resumed workflow
    # can still inject the QM-derived bond/angle params after ff_assign().
    for k in (
        "_yadonpy_mseminario_itp",
        "_yadonpy_mseminario_json",
        "_yadonpy_bonded_itp",
        "_yadonpy_bonded_json",
        "_yadonpy_bonded_method",
        "_yadonpy_bonded_override",
        "_yadonpy_bonded_requested",
        "_yadonpy_bonded_explicit",
        "_yadonpy_bonded_signature",
    ):
        try:
            if mol.HasProp(k):
                v = str(mol.GetProp(k))
                if v:
                    meta[k] = v
        except Exception:
            pass
    p.write_text(json.dumps({"meta": meta, "charges": charges}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_atomic_charges_json(mol, path, *, strict: bool = True, expected_meta: dict[str, Any] | None = None) -> bool:
    """Load charges written by _save_atomic_charges_json into an RDKit Mol.

    Returns True if charges were applied.
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return False
    try:
        obj = json.loads(p.read_text(encoding='utf-8'))
        charges = obj.get('charges')
        if not isinstance(charges, list):
            return False
        if strict and len(charges) != mol.GetNumAtoms():
            return False
        meta = obj.get('meta') if isinstance(obj, dict) else None
        if expected_meta and isinstance(meta, dict):
            for key, value in expected_meta.items():
                if value is None:
                    continue
                if meta.get(key) != value:
                    return False
        n = min(len(charges), mol.GetNumAtoms())
        charge_label = None
        if isinstance(meta, dict):
            raw_label = str(meta.get("charge") or "").strip().upper()
            if raw_label in {"RESP", "RESP2", "ESP", "MULLIKEN", "LOWDIN"}:
                charge_label = raw_label
        for i in range(n):
            mol.GetAtomWithIdx(i).SetDoubleProp('AtomicCharge', float(charges[i]))
            if charge_label in {"RESP", "RESP2", "ESP"}:
                mol.GetAtomWithIdx(i).SetDoubleProp(charge_label, float(charges[i]))
            elif charge_label == "MULLIKEN":
                mol.GetAtomWithIdx(i).SetDoubleProp("MullikenCharge", float(charges[i]))
            elif charge_label == "LOWDIN":
                mol.GetAtomWithIdx(i).SetDoubleProp("LowdinCharge", float(charges[i]))

        # Restore QM patch metadata (e.g. (m)Seminario bond/angle fragment) so
        # downstream topology writers can inject it.
        try:
            if isinstance(meta, dict):
                for k in (
                    "_yadonpy_mseminario_itp",
                    "_yadonpy_mseminario_json",
                    "_yadonpy_bonded_itp",
                    "_yadonpy_bonded_json",
                    "_yadonpy_bonded_method",
                    "_yadonpy_bonded_override",
                    "_yadonpy_bonded_requested",
                    "_yadonpy_bonded_explicit",
                    "_yadonpy_bonded_signature",
                ):
                    v = meta.get(k)
                    if isinstance(v, str) and v.strip():
                        mol.SetProp(k, v.strip())
                for k in (
                    "_yadonpy_charge_groups_json",
                    "_yadonpy_resp_constraints_json",
                    "_yadonpy_polyelectrolyte_summary_json",
                    "_yadonpy_psiresp_constraints",
                    "_yadonpy_resp_profile",
                    "_yadonpy_qm_recipe_json",
                ):
                    v = meta.get(k)
                    if isinstance(v, str) and v.strip():
                        mol.SetProp(k, v.strip())
        except Exception:
            pass
        try:
            mol.SetProp('_yadonpy_charge_loaded_from', str(p))
        except Exception:
            pass
        return True
    except Exception:
        return False

def _read_sdf_one(path: Path):
    """Read a single-molecule SDF file."""
    sup = Chem.SDMolSupplier(str(path), removeHs=False)
    if not sup or sup[0] is None:
        raise ValueError(f"Cannot read molecule from SDF: {path}")
    return sup[0]


def _write_sdf_one(mol, path: Path) -> None:
    """Write a single RDKit molecule to SDF."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(p))
    w.write(mol)
    w.close()


def _copy_geometry_inplace(dst, src) -> None:
    """Copy the first conformer from ``src`` onto ``dst`` in place."""
    if int(dst.GetNumAtoms()) != int(src.GetNumAtoms()):
        raise ValueError('Atom count mismatch while restoring geometry')
    try:
        dst.RemoveAllConformers()
    except Exception:
        try:
            for cid in range(int(dst.GetNumConformers()) - 1, -1, -1):
                dst.RemoveConformer(cid)
        except Exception:
            pass

    conf_src = src.GetConformer(0)
    conf = Chem.Conformer(int(src.GetNumAtoms()))
    conf.Set3D(bool(conf_src.Is3D()))
    pos = conf_src.GetPositions()
    for i in range(int(src.GetNumAtoms())):
        conf.SetAtomPosition(i, Geom.Point3D(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
    dst.AddConformer(conf, assignId=True)


def _load_energy_json(path: Path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding='utf-8'))
        energy = obj.get('energy')
        if isinstance(energy, list):
            try:
                return np.asarray(energy, dtype=float)
            except Exception:
                return energy
        return energy
    except Exception:
        return None


def _json_safe_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    return value


def _save_energy_json(path: Path, energy, *, log_name: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        'log_name': str(log_name),
        'energy': _json_safe_value(energy),
    }
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _reattach_bonded_patch_metadata(mol, *, work_dir_root, log_name: str) -> bool:
    """Best-effort restore of bonded patch metadata from the standard QM folder."""
    try:
        task_dir = Path(work_dir_root) / '01_qm' / '07_bonded_params' / str(log_name)
    except Exception:
        return False
    if not task_dir.exists():
        return False

    method = None
    itp_path = None
    json_path = None
    candidates = [
        ('DRIH', task_dir / 'bonded_drih_patch.itp', task_dir / 'bonded_drih_params.json'),
        ('mseminario', task_dir / 'bond_angle_params.itp', task_dir / 'bond_angle_params.json'),
    ]
    for meth, itp_cand, json_cand in candidates:
        if itp_cand.exists() or json_cand.exists():
            method = meth
            itp_path = itp_cand if itp_cand.exists() else None
            json_path = json_cand if json_cand.exists() else None
            break

    if method is None:
        return False

    try:
        if itp_path is not None:
            if method == 'mseminario':
                mol.SetProp('_yadonpy_mseminario_itp', str(itp_path.resolve()))
            mol.SetProp('_yadonpy_bonded_itp', str(itp_path.resolve()))
        if json_path is not None:
            if method == 'mseminario':
                mol.SetProp('_yadonpy_mseminario_json', str(json_path.resolve()))
            mol.SetProp('_yadonpy_bonded_json', str(json_path.resolve()))
        mol.SetProp('_yadonpy_bonded_method', str(method))
        mol.SetProp('_yadonpy_bonded_requested', str(method).lower())
        mol.SetProp('_yadonpy_bonded_signature', str(method).lower())
        mol.SetProp('_yadonpy_bonded_override', '1')
        return True
    except Exception:
        return False


def _set_total_charge_multiplicity(mol, total_charge, total_multiplicity, kwargs):
    """Populate kwargs for QMw with correct total charge/multiplicity.

    YadonPy historically assumed neutral molecules unless users manually passed total_charge.
    For polyelectrolyte monomers (SMILES containing e.g. [O-]), Psi4 must be told the net charge.
    """
    # Charge
    if type(total_charge) is int:
        kwargs['charge'] = int(total_charge)
    elif total_charge is None:
        fc = 0
        for _a in mol.GetAtoms():
            fc += int(_a.GetFormalCharge())
        if fc != 0:
            kwargs['charge'] = int(fc)

    # Multiplicity
    if type(total_multiplicity) is int:
        kwargs['multiplicity'] = int(total_multiplicity)
    elif total_multiplicity is None:
        # Best-effort open-shell detection from SMILES/RDKit radical electrons.
        # This is necessarily approximate; users can always override explicitly.
        try:
            n_rad = 0
            for _a in mol.GetAtoms():
                n_rad += int(_a.GetNumRadicalElectrons())
            if n_rad > 0:
                kwargs['multiplicity'] = int(n_rad + 1)
        except Exception:
            pass

    return kwargs




def assign_charges(
    mol,
    charge='RESP',
    confId=0,
    opt=True,
    work_dir=None,
    tmp_dir=None,
    log_name=None,
    qm_solver='psi4',
    # OPT level (geometry)
    # NOTE: Psi4 method keyword must be something it actually provides.
    # In Psi4 v1.10, plain "wb97m" is NOT available, while "wb97m-d3bj" is.
    # We keep the working default and treat "wb97m" as an alias elsewhere.
    opt_method='wb97m-d3bj',
    opt_basis='def2-SVP',
    opt_basis_gen={'Br': 'def2-SVP', 'I': 'def2-SVP'},
    geom_iter=50,
    geom_conv='QCHEM',
    geom_algorithm='RFO',
    # RESP/ESP level (single point)
    charge_method='wb97m-d3bj',
    charge_basis='def2-TZVP',
    charge_basis_gen={'Br': 'def2-TZVP', 'I': 'def2-TZVP'},
    # behavior toggles
    auto_level: bool = True,
    bonded_params: str = 'auto',
    total_charge=None,
    total_multiplicity=None,
    polyelectrolyte_mode: bool = False,
    polyelectrolyte_detection: str = 'auto',
    resp_profile: str = 'adaptive',
    symmetrize=True,
    symmetrize_geometry: bool = True,
    restart: Optional[bool] = None,
    **kwargs,
):
    """
    sim.qm.assign_charges

    Assignment atomic charge for RDKit Mol object
    This is wrapper function of core.calc.assign_charges

    Args:
        mol: RDKit Mol object

    Optional args:
        charge: Select charge type of gasteiger, RESP, ESP, Mulliken, Lowdin, zero, CM1A, <scale>*CM1A, CM5, or <scale>*CM5 (str, default:RESP)
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        charge_method: Using method in the charge calculation (str, default:HF)
        charge_basis: Using basis set in the charge calculation (str, default:6-31G(d))
        charge_basis_gen: Using basis set in the charge calculation for each element

    Returns:
        boolean
    """

    # ------------------------------------------------------------------
    # Normalize "Default" placeholders.
    #
    # In higher-level workflows (e.g., MolDB template.csv) users may write
    # method/basis_set as "Default". Psi4 does not recognize "default" as a
    # real method name, so we map it back to YadonPy's built-in defaults.
    # ------------------------------------------------------------------
    def _is_default_token(x) -> bool:
        try:
            s = str(x).strip().lower()
        except Exception:
            return False
        return s in ("default", "none", "nan", "null", "")

    if _is_default_token(opt_method):
        opt_method = _DEFAULT_OPT_METHOD
    if _is_default_token(opt_basis):
        opt_basis = _DEFAULT_OPT_BASIS
    if _is_default_token(charge_method):
        charge_method = _DEFAULT_CHARGE_METHOD
    if _is_default_token(charge_basis):
        charge_basis = _DEFAULT_CHARGE_BASIS
    resp_profile = _normalize_resp_profile(resp_profile)

    # If the caller didn't provide an explicit name, use (and persist) a stable
    # molecule name. If no name was set, we infer the caller's Python variable
    # name (e.g., solvent_A) best-effort.
    if log_name is None:
        try:
            # Called from this wrapper; use depth=2 to inspect user-script frames.
            log_name = utils.ensure_name(mol, name=None, depth=2, prefer_var=True)
        except Exception:
            log_name = None
    if not log_name:
        log_name = 'charge'

    t_qm = _qm_begin(
        "QM charge assignment",
        log_name=str(log_name),
        mol=mol,
        detail=f"charge={charge} | opt={bool(opt)}",
    )
    _smi = _qm_smiles(mol)

    # ------------------------------------------------------------------
    # Work dir hygiene: keep work_dir clean by writing QM artifacts under
    #   work_dir/01_qm/<log_name>/charge/
    # ------------------------------------------------------------------
    work_dir_root = None
    if work_dir is not None:
        work_dir_root, work_dir = _qm_task_dir(work_dir, log_name=str(log_name), task="charge")
        if tmp_dir is None:
            tmp_dir = work_dir

    restart_flag = resolve_restart(restart)
    charged_sdf = None
    charges_json = None
    cached_charge_hit = False
    if work_dir_root is not None:
        charged_sdf = Path(work_dir) / f"{log_name}.charged.sdf"
        charges_json = Path(work_dir_root) / "01_qm" / "90_charged_mol2" / f"{log_name}.charges.json"

    smiles_hint = _smi if isinstance(_smi, str) and _smi not in ("?", "") else None
    is_inorganic = False
    is_poly_ion = False
    fc = 0
    n_rad = 0
    try:
        is_inorganic = utils.is_inorganic_ion_like(mol, smiles_hint=smiles_hint)
        is_poly_ion = utils.is_inorganic_polyatomic_ion(mol, smiles_hint=smiles_hint)
        for a in mol.GetAtoms():
            fc += int(a.GetFormalCharge())
            n_rad += int(a.GetNumRadicalElectrons())
    except Exception:
        pass

    eff_charge = int(total_charge) if type(total_charge) is int else int(fc)
    if (total_multiplicity is None) and (type(total_multiplicity) is not int) and n_rad > 0:
        total_multiplicity = int(n_rad + 1)

    resolved_recipe = _resolve_resp_qm_recipe(
        mol,
        resp_profile=resp_profile,
        charge_model=str(charge),
        opt_method=str(opt_method),
        opt_basis=str(opt_basis),
        opt_basis_gen=opt_basis_gen,
        charge_method=str(charge_method),
        charge_basis=str(charge_basis),
        charge_basis_gen=charge_basis_gen,
        auto_level=bool(auto_level),
        total_charge=int(eff_charge),
    )
    opt_method = str(resolved_recipe["opt_method"])
    opt_basis = str(resolved_recipe["opt_basis"])
    opt_basis_gen = dict(resolved_recipe.get("opt_basis_gen") or {})
    charge_method = str(resolved_recipe["charge_method"])
    charge_basis = str(resolved_recipe["charge_basis"])
    charge_basis_gen = dict(resolved_recipe.get("charge_basis_gen") or {})
    _stamp_resp_recipe_metadata(mol, resp_profile=resp_profile, recipe=resolved_recipe)

    def _build_charge_cache_meta(recipe: dict[str, Any]) -> dict[str, Any]:
        return {
            "charge_model": str(charge),
            "polyelectrolyte_mode": bool(polyelectrolyte_mode),
            "polyelectrolyte_detection": str(polyelectrolyte_detection or "auto"),
            "resp_profile": str(resp_profile),
            "resolved_qm_recipe": dict(recipe),
        }

    charge_cache_meta = _build_charge_cache_meta(resolved_recipe)

    if is_h_terminator_placeholder(mol, smiles_hint=_smi):
        apply_placeholder_zero_charges(mol, charge_label=str(charge))
        _qm_log("[SKIP] Hydrogen terminator placeholder uses zero-charge shortcut; QM/RESP skipped", level=1)
        try:
            if charged_sdf is not None:
                _write_sdf_one(mol, charged_sdf)
        except Exception as e:
            utils.yadon_print(f"QM restart warning: failed to save charged SDF for {log_name}: {e}", level=2)
        try:
            if charges_json is not None:
                _save_atomic_charges_json(
                    mol,
                    charges_json,
                    charge_label=str(charge),
                    log_name=str(log_name),
                    extra_meta=dict(charge_cache_meta, shortcut="h_terminator_placeholder"),
                )
        except Exception:
            pass
        _qm_done("QM charge assignment", t_qm, detail=f"charge_model={charge} | shortcut=h_terminator")
        return True

    if work_dir_root is not None and restart_flag and charges_json.exists() and ((not bool(opt)) or charged_sdf.exists()):
        try:
            if charged_sdf.exists():
                cached = _read_sdf_one(charged_sdf)
                _copy_geometry_inplace(mol, cached)
            if not load_atomic_charges_json(mol, charges_json, strict=True, expected_meta=charge_cache_meta):
                raise RuntimeError(f"Failed to load cached charges from {charges_json}")
            _reattach_bonded_patch_metadata(mol, work_dir_root=work_dir_root, log_name=str(log_name))
            cached_charge_hit = True
            _qm_log(f"[SKIP] Reused cached charges | file={charges_json.name}", level=1)
        except Exception as e:
            utils.yadon_print(f"QM restart warning: cached assign_charges restore failed for {log_name}: {e}; recomputing.", level=2)

    if not cached_charge_hit:
        # ------------------------------------------------------------------
        # For small inorganic ions (PF6-, BF4-, ClO4-...) RDKit/MMFF can be
        # unstable. Prefer OpenBabel-based 3D building when possible.
        # This enables skipping conformer search for anions in example workflows.
        # ------------------------------------------------------------------
        try:
            if mol.GetNumConformers() == 0 or utils.is_inorganic_ion_like(mol, smiles_hint=smiles_hint):
                utils.ensure_3d_coords(mol, smiles_hint=smiles_hint, engine='openbabel')
        except Exception:
            pass

    # Echo the chosen levels to screen
    _qm_log(
        f"[ITEM] levels            : OPT={str(opt_method)}/{str(opt_basis)} | RESP(ESP)={str(charge_method)}/{str(charge_basis)} | profile={resp_profile}",
        level=1,
    )

    # ------------------------------------------------------------------
    # Optional: geometry symmetrization for high-symmetry inorganic ions
    #
    # PF6-/BF4-/ClO4-/AsF6- ... are prone to tiny numeric distortions which can
    # later be amplified by Hessian-based bonded params and by MD.
    # We symmetrize the local AX4/AX6 polyhedron in Cartesian space.
    # ------------------------------------------------------------------
    if symmetrize_geometry and is_poly_ion:
        try:
            if utils.is_high_symmetry_polyhedral_ion(mol, smiles_hint=smiles_hint):
                utils.symmetrize_polyhedral_ion_geometry(mol, confId=confId)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Basis fallback for Psi4 (esp. ma-* basis availability varies by build)
    # ------------------------------------------------------------------
    def _attempt_levels(ob: str, cb: str) -> bool:
        return calc.assign_charges(
            mol,
            charge=charge,
            confId=confId,
            opt=opt,
            work_dir=work_dir,
            tmp_dir=tmp_dir,
            log_name=log_name,
            qm_solver=qm_solver,
            opt_method=opt_method,
            opt_basis=ob,
            opt_basis_gen=opt_basis_gen,
            geom_iter=geom_iter,
            geom_conv=geom_conv,
            geom_algorithm=geom_algorithm,
            charge_method=charge_method,
            charge_basis=cb,
            charge_basis_gen=charge_basis_gen,
            total_charge=total_charge,
            total_multiplicity=total_multiplicity,
            polyelectrolyte_mode=bool(polyelectrolyte_mode),
            polyelectrolyte_detection=str(polyelectrolyte_detection or 'auto'),
            resp_profile=str(resp_profile),
            **kwargs,
        )

    if cached_charge_hit:
        flag = True
    else:
        flag = _attempt_levels(opt_basis, charge_basis)
    if (not flag) and (not cached_charge_hit) and auto_level and is_inorganic:
        # Minimal, robust fallback ladder (avoid over-complicated basis shopping).
        # 1) Keep the functional, relax basis to commonly available sets.
        trials = [
            ("def2-SVP", "def2-TZVP"),
            ("def2-SVPD", "def2-TZVPD"),
            ("def2-SVPD", "def2-TZVPPD"),
            ("def2-SVP", "def2-TZVPD"),
        ]
        for ob, cb in trials:
            utils.yadon_print(f"QM retry with basis: OPT={ob} | RESP(ESP)={cb}", level=2)
            if _attempt_levels(ob, cb):
                opt_basis, charge_basis, flag = ob, cb, True
                resolved_recipe["opt_basis"] = str(ob)
                resolved_recipe["charge_basis"] = str(cb)
                resolved_recipe["opt_basis_gen"] = _basis_gen_with_override(opt_basis_gen, ob)
                resolved_recipe["charge_basis_gen"] = _basis_gen_with_override(charge_basis_gen, cb)
                _stamp_resp_recipe_metadata(mol, resp_profile=resp_profile, recipe=resolved_recipe)
                charge_cache_meta = _build_charge_cache_meta(resolved_recipe)
                break

    # Re-apply geometry symmetrization after QM optimization (if any).
    if flag and symmetrize_geometry and is_poly_ion:
        try:
            if utils.is_high_symmetry_polyhedral_ion(mol, smiles_hint=smiles_hint):
                utils.symmetrize_polyhedral_ion_geometry(mol, confId=confId)
        except Exception:
            pass

    if flag and symmetrize:
        try:
            eq_groups = _get_psiresp_equivalence_groups(mol)
            n_symm = _symmetrize_charge_properties(mol, equivalence_groups=eq_groups)
            if n_symm:
                utils.yadon_print(f"QM task: symmetrize_charges groups={n_symm} purpose={log_name}", level=1)
        except Exception:
            # Do not fail the workflow if symmetrization is not possible.
            pass

    

    # ------------------------------------------------------------------
    # Bonded parameters (override/patch) for improved rigidity
    #
    # Supported methods:
    #   - ff_assigned : keep force-field assigned bonded terms (default for most molecules)
    #   - mseminario  : Hessian-derived (modified Seminario) bond+angle harmonics
    #   - drih        : geometry-driven robust stiffening for AX4/AX6 high-symmetry ions
    #   - auto        : use drih for high-symmetry inorganic ions; mseminario for other polyions; else ff_assigned
    # ------------------------------------------------------------------
    try:
        _bp = str(bonded_params).lower().strip() if bonded_params is not None else 'auto'
    except Exception:
        _bp = 'auto'

    # normalize aliases
    if _bp in ('on', 'true', '1'):
        _bp = 'mseminario'
    if _bp in ('off', 'false', '0', 'none', ''):
        _bp = 'ff_assigned'

    if _bp == 'auto':
        try:
            if utils.is_high_symmetry_polyhedral_ion(mol, smiles_hint=smiles_hint):
                _bp = 'drih'
            elif is_poly_ion:
                _bp = 'mseminario'
            else:
                _bp = 'ff_assigned'
        except Exception:
            _bp = 'mseminario' if is_poly_ion else 'ff_assigned'

    if flag and work_dir_root is not None and is_poly_ion and (_bp in ('mseminario', 'drih')):
        try:
            # Reuse an existing bonded patch during restart whenever possible.
            existing_method = None
            existing_itp = None
            try:
                if mol.HasProp('_yadonpy_bonded_method'):
                    existing_method = str(mol.GetProp('_yadonpy_bonded_method')).strip().lower()
                if mol.HasProp('_yadonpy_bonded_itp'):
                    cand = Path(str(mol.GetProp('_yadonpy_bonded_itp')).strip())
                    if cand.is_file():
                        existing_itp = cand
            except Exception:
                existing_method = None
                existing_itp = None

            want = 'drih' if _bp == 'drih' else 'mseminario'
            if restart_flag and existing_itp is not None and existing_method == want:
                _qm_log(f"[SKIP] Reused cached bonded params | method={want} | file={existing_itp.name}", level=1)
            else:
                # Always re-symmetrize right before generating bonded params (best-effort).
                if symmetrize_geometry:
                    try:
                        if utils.is_high_symmetry_polyhedral_ion(mol, smiles_hint=smiles_hint):
                            utils.symmetrize_polyhedral_ion_geometry(mol, confId=confId)
                    except Exception:
                        pass

                if _bp == 'drih':
                    utils.yadon_print(f"QM task: bonded_params (DRIH) purpose={log_name}", level=1)
                    res = bond_angle_params_drih(
                        mol,
                        confId=confId,
                        work_dir=str(work_dir_root),
                        log_name=str(log_name),
                        smiles_hint=smiles_hint,
                    )
                    if res.get('itp'):
                        try:
                            mol.SetProp('_yadonpy_bonded_itp', str(res['itp']))
                            mol.SetProp('_yadonpy_bonded_json', str(res.get('json', '')))
                            mol.SetProp('_yadonpy_bonded_method', 'DRIH')
                            mol.SetProp('_yadonpy_bonded_requested', 'drih')
                            mol.SetProp('_yadonpy_bonded_signature', 'drih')
                            mol.SetProp('_yadonpy_bonded_override', '1')
                        except Exception:
                            pass
                else:
                    # For anionic polyions, use a stronger Hessian basis aligned with the RESP/ESP single-point
                    hess_method = str(charge_method) if fc < 0 else str(opt_method)
                    if fc < 0:
                        hess_basis = str(charge_basis)
                        hess_basis_gen = charge_basis_gen
                    else:
                        hess_basis = str(opt_basis)
                        hess_basis_gen = opt_basis_gen

                    utils.yadon_print(
                        f"QM task: bonded_params (mseminario, bond+angle) purpose={log_name} opt={opt_method}/{opt_basis} hess={hess_method}/{hess_basis}",
                        level=1,
                    )
                    res = bond_angle_params_mseminario(
                        mol,
                        confId=confId,
                        opt=False,
                        work_dir=str(work_dir_root),
                        tmp_dir=tmp_dir,
                        log_name=str(log_name),
                        qm_solver=qm_solver,
                        opt_method=str(opt_method),
                        opt_basis=str(opt_basis),
                        hess_method=hess_method,
                        hess_basis=hess_basis,
                        hess_basis_gen=hess_basis_gen,
                        total_charge=total_charge,
                        total_multiplicity=total_multiplicity,
                    )
                    if res.get('itp'):
                        try:
                            mol.SetProp('_yadonpy_mseminario_itp', str(res['itp']))
                            mol.SetProp('_yadonpy_mseminario_json', str(res.get('json', '')))
                            # also populate generic keys for downstream injection
                            mol.SetProp('_yadonpy_bonded_itp', str(res['itp']))
                            mol.SetProp('_yadonpy_bonded_json', str(res.get('json', '')))
                            mol.SetProp('_yadonpy_bonded_method', 'mseminario')
                            mol.SetProp('_yadonpy_bonded_requested', 'mseminario')
                            mol.SetProp('_yadonpy_bonded_signature', 'mseminario')
                            mol.SetProp('_yadonpy_bonded_override', '1')
                        except Exception:
                            pass
        except Exception as e:
            # Non-fatal: keep the workflow moving; topology will fall back to GAFF.
            utils.yadon_print(f"QM warning: bonded-params({_bp}) failed for {log_name}: {e}", level=2)

    # Best-effort: export a charged MOL2 after charge assignment.
    try:
        # Export a charged MOL2 + JSON in a predictable module folder.
        if work_dir_root is not None:
            from ..io.mol2 import write_mol2

            # Keep QM module exports grouped and sortable.
            d = Path(work_dir_root) / "01_qm" / "90_charged_mol2"
            d.mkdir(parents=True, exist_ok=True)
            write_mol2(mol=mol, out_mol2=d / f"{log_name}.mol2", mol_name=str(log_name))

            # Also write JSON charges for easy resuming.
            try:
                extra_meta = dict(charge_cache_meta)
                for key in (
                    "_yadonpy_charge_groups_json",
                    "_yadonpy_resp_constraints_json",
                    "_yadonpy_polyelectrolyte_summary_json",
                    "_yadonpy_psiresp_constraints",
                    "_yadonpy_resp_profile",
                    "_yadonpy_qm_recipe_json",
                ):
                    try:
                        if mol.HasProp(key):
                            extra_meta[key] = str(mol.GetProp(key))
                    except Exception:
                        pass
                _save_atomic_charges_json(
                    mol,
                    d / f"{log_name}.charges.json",
                    charge_label=str(charge),
                    log_name=str(log_name),
                    extra_meta=extra_meta,
                )
            except Exception:
                pass
    except Exception:
        pass

    # Make failure loud by default so user scripts can stay linear and clean
    # (RadonPy-style) without carrying intermediate "ok" variables.
    if not flag:
        raise RuntimeError(f"Charge assignment failed (charge={charge}) for {log_name}")

    try:
        if charged_sdf is not None:
            _write_sdf_one(mol, charged_sdf)
    except Exception as e:
        utils.yadon_print(f"QM restart warning: failed to save charged SDF for {log_name}: {e}", level=2)

    _qm_done("QM charge assignment", t_qm, detail=f"charge_model={charge} | profile={resp_profile}")
    return flag
        

def conformation_search(mol, ff=None, nconf=1000, dft_nconf=4, etkdg_ver=2, rmsthresh=0.5, tfdthresh=0.02, clustering='TFD', qm_solver='psi4',
    opt_method='wb97m-d3bj', opt_basis='def2-SVP', opt_basis_gen={'Br': 'def2-SVP', 'I': 'def2-SVP'},
    geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO', log_name=None, work_dir=None, tmp_dir=None,
    etkdg_omp=-1, psi4_omp=-1, psi4_mp=0, mm_mp=0, memory=1000, mm_solver='rdkit', gmx_refine_n=0, gmx_ntomp=None, gmx_ntmpi=None, gmx_gpu_id=None, total_charge=None, total_multiplicity=None, restart: Optional[bool] = None, **kwargs):
    """
    sim.qm.conformation_search

    Conformation search
    This is wrapper function of core.calc.conformation_search

    Args:
        mol: RDKit Mol object
    
    Optional args:
        ff: Force field instance. If None, MMFF94 optimization is carried out by RDKit
        nconf: Number of generating conformations (int)
        dft_nconf: Number of conformations for DFT optimization (int, default:4)
        work_dir: Path of work directory (str)
        etkdg_omp: Number of threads of OpenMP in ETKDG of RDkit (int)
        mm_mp: Number of parallel execution of RDKit in MM optimization (int)
        psi4_omp: Number of threads of OpenMP in Psi4 (int)
        psi4_mp: Number of parallel execution of Psi4 in DFT optimization (int)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element

    Returns:
        RDKit Mol object
        DFT and MM energy (ndarray, kcal/mol)
    """

    # Default naming: if log_name is not provided, use the molecule's name.
    if log_name is None:
        try:
            # Called from this wrapper; use depth=2 to inspect user-script frames.
            log_name = utils.ensure_name(mol, name=None, depth=2, prefer_var=True)
        except Exception:
            log_name = 'mol'

    t_qm = _qm_begin(
        "QM conformation search",
        log_name=str(log_name),
        mol=mol,
        detail=f"nconf={int(nconf)} | dft_nconf={int(dft_nconf)}",
    )
    _smi = _qm_smiles(mol)
    # Compatibility: some scripts mistakenly pass MD runtime keys (mpi/omp) into QM helpers.
    # - "omp" here should NOT be used for Psi4; use psi4_omp instead. We drop it to avoid crashes.
    # - "mpi" is an MD setting and is ignored here.
    if "mpi" in kwargs:
        kwargs.pop("mpi", None)
    if "omp" in kwargs:
        if psi4_omp in (None, -1):
            try:
                psi4_omp = int(kwargs.pop("omp"))
            except Exception:
                kwargs.pop("omp", None)
        else:
            kwargs.pop("omp", None)

    # Log the goal and SMILES for reproducibility
    try:
        if mol.HasProp('_yadonpy_input_smiles'):
            _smi = mol.GetProp('_yadonpy_input_smiles')
        elif mol.HasProp('_yadonpy_smiles'):
            _smi = mol.GetProp('_yadonpy_smiles')
        else:
            _smi = Chem.MolToSmiles(mol)
    except Exception:
        _smi = '?'
    _qm_log(f"[ITEM] search_config     : smiles={_smi} | nconf={nconf} | dft_nconf={dft_nconf}", level=1)

    # ------------------------------------------------------------------
    # Auto-infer charge and multiplicity from SMILES/RDKit (best-effort)
    # and apply default basis policy (2026-03).
    #
    #   anions : OPT def2-SVPD / wb97m
    #   others : OPT def2-SVP  / wb97m
    # ------------------------------------------------------------------
    try:
        fc = 0
        n_rad = 0
        for a in mol.GetAtoms():
            fc += int(a.GetFormalCharge())
            n_rad += int(a.GetNumRadicalElectrons())
        eff_charge = int(total_charge) if type(total_charge) is int else int(fc)
        if (type(total_charge) is not int) and eff_charge != 0:
            total_charge = int(eff_charge)
        if (type(total_multiplicity) is not int) and (total_multiplicity is None) and (n_rad > 0):
            total_multiplicity = int(n_rad + 1)
        if eff_charge < 0 and str(opt_basis).strip().lower() == 'def2-svp':
            opt_basis = 'def2-SVPD'
            opt_basis_gen = {'Br': 'def2-SVPD', 'I': 'def2-SVPD', **(opt_basis_gen or {})}
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Work dir hygiene: keep work_dir clean by writing QM artifacts under
    #   work_dir/01_qm/<log_name>/confsearch/
    # ------------------------------------------------------------------
    conf_sdf = None
    energy_json = None
    if work_dir is not None:
        _root, work_dir = _qm_task_dir(work_dir, log_name=str(log_name), task="confsearch")
        if tmp_dir is None:
            tmp_dir = work_dir
        conf_sdf = Path(work_dir) / f"{log_name}.opt.sdf"
        energy_json = Path(work_dir) / f"{log_name}.energy.json"

    restart_flag = resolve_restart(restart)
    if restart_flag and conf_sdf is not None and conf_sdf.exists():
        try:
            cached = _read_sdf_one(conf_sdf)
            _copy_geometry_inplace(mol, cached)
            energy = _load_energy_json(energy_json) if energy_json is not None else None
            _qm_log(f"[SKIP] Reused cached conformer | file={conf_sdf.name}", level=1)
            _qm_done("QM conformation search", t_qm, detail="restart cache hit")
            return mol, energy
        except Exception as e:
            utils.yadon_print(f"QM restart warning: cached conformation_search restore failed for {log_name}: {e}; recomputing.", level=2)


    # ------------------------------------------------------------------
    # IMPORTANT robustness note
    # ------------------------------------------------------------------
    # `core.calc.conformation_search` deep-copies the input and returns a *new*
    # RDKit Mol. This is easy to misuse when molecules are iterated in a list.
    # If users do not rebind the returned Mol back to their variables, later
    # steps (RESP, polymerization, export) will operate on the *old* Mol object
    # without 3D geometry/charges.
    #
    # We therefore apply the optimized geometry back onto the *input* Mol in
    # place (best-effort) and return the original object. Scripts that rebind
    # the return value keep working.
    mol_in = mol

    mol_out, energy = calc.conformation_search(mol_in, ff=ff, nconf=nconf, dft_nconf=dft_nconf, etkdg_ver=etkdg_ver, rmsthresh=rmsthresh, qm_solver=qm_solver,
                tfdthresh=tfdthresh, clustering=clustering, opt_method=opt_method, opt_basis=opt_basis,
                opt_basis_gen=opt_basis_gen, geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm, log_name=log_name, work_dir=work_dir, tmp_dir=tmp_dir,
                etkdg_omp=etkdg_omp, psi4_omp=psi4_omp, psi4_mp=psi4_mp, mm_mp=mm_mp, memory=memory,
                mm_solver=mm_solver, gmx_refine_n=gmx_refine_n, gmx_ntomp=gmx_ntomp, gmx_ntmpi=gmx_ntmpi, gmx_gpu_id=gmx_gpu_id,
                total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)

    # Downstream polymer builders (e.g. random_walk_polymerization) expect a *single* 3D conformer.
    # RDKit's CombineMols warns (and may drop coords) if molecules have different numbers of conformers.
    # Keep only the lowest-energy conformer (ID 0) to make behavior deterministic and robust.
    try:
        n_c = int(mol_out.GetNumConformers())
        if n_c > 1:
            for cid in range(n_c - 1, 0, -1):
                mol_out.RemoveConformer(cid)
    except Exception:
        pass

    # Best-effort: copy the resulting geometry back to the input molecule.
    # If this fails (e.g., atom count mismatch), fall back to returning mol_out.
    try:
        if int(mol_in.GetNumAtoms()) == int(mol_out.GetNumAtoms()):
            # Replace conformers on the input molecule
            try:
                mol_in.RemoveAllConformers()
            except Exception:
                try:
                    for cid in range(int(mol_in.GetNumConformers()) - 1, -1, -1):
                        mol_in.RemoveConformer(cid)
                except Exception:
                    pass

            conf = Chem.Conformer(int(mol_out.GetNumAtoms()))
            conf.Set3D(True)
            pos = mol_out.GetConformer(0).GetPositions()
            for i in range(int(mol_out.GetNumAtoms())):
                conf.SetAtomPosition(i, Geom.Point3D(float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
            mol_in.AddConformer(conf, assignId=True)
            try:
                if conf_sdf is not None:
                    _write_sdf_one(mol_in, conf_sdf)
                if energy_json is not None:
                    _save_energy_json(energy_json, energy, log_name=str(log_name))
            except Exception as e:
                utils.yadon_print(f"QM restart warning: failed to save conformation_search cache for {log_name}: {e}", level=2)
            _qm_done("QM conformation search", t_qm, detail="best conformer ready")
            return mol_in, energy
    except Exception:
        pass

    try:
        if conf_sdf is not None:
            _write_sdf_one(mol_out, conf_sdf)
        if energy_json is not None:
            _save_energy_json(energy_json, energy, log_name=str(log_name))
    except Exception as e:
        utils.yadon_print(f"QM restart warning: failed to save conformation_search cache for {log_name}: {e}", level=2)

    _qm_done("QM conformation search", t_qm, detail="best conformer ready")
    return mol_out, energy


def sp_prop(mol, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='sp_prop', qm_solver='psi4',
    opt_method='wb97m-d3bj', opt_basis='def2-SVP', opt_basis_gen={'Br': 'def2-SVP', 'I': 'def2-SVP'}, 
    geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
    sp_method='wb97m-d3bj', sp_basis='def2-TZVP', sp_basis_gen={'Br': 'def2-TZVP', 'I': 'def2-TZVP'},
    total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.sp_prop

    Calculation of total energy, HOMO, LUMO, dipole moment by QM calculation

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        sp_method: Using method in the single point calculation (str, default:wb97m-d3bj)
        sp_basis: Using basis set in the single point calculation (str, default:6-311G(2d,p))
        opt_basis_gen: Using basis set in the single point calculation for each element

    return
        dict
            qm_total_energy (float, kJ/mol)
            qm_homo (float, eV)
            qm_lumo (float, eV)
            qm_dipole (x, y, z) (float, Debye)
    """
    e_prop = {}

    # Keep work_dir clean (module folder layout)
    if work_dir is not None:
        _root, work_dir = _qm_task_dir(work_dir, log_name=str(log_name), task="sp_prop")
        if tmp_dir is None:
            tmp_dir = work_dir

    kwargs = _set_total_charge_multiplicity(mol, total_charge, total_multiplicity, kwargs)

    # Default basis policy (2026-03): use diffuse basis for anions.
    try:
        eff_charge = int(kwargs.get('charge', 0))
    except Exception:
        eff_charge = 0
    if eff_charge < 0:
        try:
            if str(opt_basis).strip().lower() == 'def2-svp':
                opt_basis = 'def2-SVPD'
                opt_basis_gen = {'Br': 'def2-SVPD', 'I': 'def2-SVPD', **(opt_basis_gen or {})}
        except Exception:
            pass
        try:
            if str(sp_basis).strip().lower() == 'def2-tzvp':
                sp_basis = 'def2-TZVPD'
                sp_basis_gen = {'Br': 'def2-TZVPD', 'I': 'def2-TZVPD', **(sp_basis_gen or {})}
        except Exception:
            pass

    psi4mol = QMw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen, qm_solver=qm_solver,
                    name=log_name, **kwargs)
    if opt:
        psi4mol.optimize(geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
        if psi4mol.error_flag:
            utils.radon_print('Psi4 optimization error in sim.qm.sp_prop.', level=2)
            return e_prop

        coord = psi4mol.mol.GetConformer(confId).GetPositions()
        for i, atom in enumerate(psi4mol.mol.GetAtoms()):
            mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    psi4mol.method = sp_method
    psi4mol.basis = sp_basis
    psi4mol.basis_gen = sp_basis_gen

    e_prop['qm_total_energy'] = psi4mol.energy()
    e_prop['qm_homo'] = psi4mol.homo
    e_prop['qm_lumo'] = psi4mol.lumo
    e_prop['qm_dipole_x'], e_prop['qm_dipole_y'], e_prop['qm_dipole_z'] = psi4mol.dipole

    del psi4mol
    gc.collect()

    return e_prop


def polarizability(mol, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='polarizability', qm_solver='psi4', mp=0,
    opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'}, 
    geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
    polar_method='wb97m-d3bj', polar_basis='6-311+G(2d,p)', polar_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
    total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.polarizability

    Calculation of dipole polarizability by QM calculation

    Args:
        mol: RDKit Mol object
    
    Optional args:
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        polar_method: Using method in the polarizability calculation (str, default:wb97m-d3bj)
        polar_basis: Using basis set in the polarizability calculation (str, default:6-311+G(2d,p))
        polar_basis_gen: Using basis set in the polarizability calculation for each element

    return
        dict
            Dipole polarizability (float, angstrom^3)
            Polarizability tensor (xx, yy, zz, xy, xz, yz) (float, angstrom^3)
    """
    polar_data = {}

    # Keep work_dir clean (module folder layout)
    if work_dir is not None:
        _root, work_dir = _qm_task_dir(work_dir, log_name=str(log_name), task="polarizability")
        if tmp_dir is None:
            tmp_dir = work_dir
    kwargs = _set_total_charge_multiplicity(mol, total_charge, total_multiplicity, kwargs)

    psi4mol = QMw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen, qm_solver=qm_solver,
                    name=log_name, **kwargs)
    if opt:
        psi4mol.optimize(geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
        if psi4mol.error_flag:
            utils.radon_print('Psi4 optimization error in calc.polarizability.', level=2)
            return polar_data

        coord = psi4mol.mol.GetConformer(confId).GetPositions()
        for i, atom in enumerate(psi4mol.mol.GetAtoms()):
            mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    psi4mol.method = polar_method
    psi4mol.basis = polar_basis
    psi4mol.basis_gen = polar_basis_gen

    alpha, d_mu = psi4mol.polar(mp=mp)
    if psi4mol.error_flag:
        utils.radon_print('Psi4 polarizability calculation error in sim.qm.polarizability.', level=2)

    polar_data = {
        'qm_polarizability': alpha,
        'qm_polarizability_xx': d_mu[0, 0],
        'qm_polarizability_yy': d_mu[1, 1],
        'qm_polarizability_zz': d_mu[2, 2],
        'qm_polarizability_xy': (d_mu[0, 1]+d_mu[1, 0])/2,
        'qm_polarizability_xz': (d_mu[0, 2]+d_mu[2, 0])/2,
        'qm_polarizability_yz': (d_mu[1, 2]+d_mu[2, 1])/2,
    }

    del psi4mol
    gc.collect()

    return polar_data


def refractive_index(mols, density, ratio=None, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='refractive_index', qm_solver='psi4', mp=0,
        opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'}, 
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
        polar_method='wb97m-d3bj', polar_basis='6-311+G(2d,p)', polar_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.refractive_index

    Calculation of refractive index by QM calculation

    Args:
        mols: List of RDKit Mol object
        density: [g/cm^3]
    
    Optional args:
        ratio: ratio of repeating units in a copolymer
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        polar_method: Using method in the polarizability calculation (str, default:wb97m-d3bj)
        polar_basis: Using basis set in the polarizability calculation (str, default:6-311+G(2d,p))
        polar_basis_gen: Using basis set in the polarizability calculation for each element

    return
        Refractive index data (dict)
            refractive_index (float)
            polarizability of repeating units (float, angstrom^3)
            polarizability tensor of repeating units (float, angstrom^3)
    """
    ri_data = {}

    if type(mols) is Chem.Mol: mols = [mols]
    mol_weight = [calc.molecular_weight(mol) for mol in mols]
    a_list = []
    
    for i, mol in enumerate(mols):
        polar_data = polarizability(mol, confId=confId, opt=opt, work_dir=work_dir, tmp_dir=tmp_dir, log_name='%s_%i' % (log_name, i), qm_solver=qm_solver, mp=mp,
                            opt_method=opt_method, opt_basis=opt_basis, opt_basis_gen=opt_basis_gen, 
                            geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm,
                            polar_method=polar_method, polar_basis=polar_basis, polar_basis_gen=polar_basis_gen,
                            total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)

        a_list.append(polar_data['qm_polarizability'])
        for k in polar_data.keys(): ri_data['%s_monomer%i' % (k, i+1)] = polar_data[k]

    ri_data['refractive_index'] = calc.refractive_index(a_list, density, mol_weight, ratio=ratio)

    return ri_data


def abbe_number_cc2(mol, density, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='abbe_number_cc2', qm_solver='psi4',
        opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'}, 
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
        polar_basis='6-311+G(2d,p)', polar_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.abbe_number_cc2

    Calculation of abbe's number by CC2 calculation

    Args:
        mol: RDKit Mol object
        density: [g/cm^3]
    
    Optional args:
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default: wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        polar_basis: Using basis set in the dynamic polarizability calculation (str, default:6-311+G(2d,p))
        polar_basis_gen: Using basis set in the polarizability calculation for each element

    return
        Abbe's number data (dict)
            abbe_number (float)
            refractive_index_656 (float)
            refractive_index_589 (float)
            refractive_index_486 (float)
            polarizability_656 (float, angstrom^3)
            polarizability_589 (float, angstrom^3)
            polarizability_486 (float, angstrom^3)
    """
    abbe_data = {}

    # Keep work_dir clean (module folder layout)
    if work_dir is not None:
        _root, work_dir = _qm_task_dir(work_dir, log_name=str(log_name), task="abbe_number_cc2")
        if tmp_dir is None:
            tmp_dir = work_dir
    kwargs = _set_total_charge_multiplicity(mol, total_charge, total_multiplicity, kwargs)

    mol_weight = calc.molecular_weight(mol)

    psi4mol = QMw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen, qm_solver=qm_solver,
                    name=log_name, **kwargs)
    if opt:
        psi4mol.optimize(geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
        if psi4mol.error_flag:
            utils.radon_print('Psi4 optimization error in sim.qm.abbe_number_cc2.', level=2)
            return abbe_data

        coord = psi4mol.mol.GetConformer(confId).GetPositions()
        for j, atom in enumerate(psi4mol.mol.GetAtoms()):
            mol.GetConformer(confId).SetAtomPosition(j, Geom.Point3D(coord[j, 0], coord[j, 1], coord[j, 2]))

    psi4mol.basis = polar_basis
    psi4mol.basis_gen = polar_basis_gen

    alpha = psi4mol.cc2_polar(omega=[656, 589, 486])

    n_656 = calc.refractive_index(alpha[0], density, mol_weight)
    n_589 = calc.refractive_index(alpha[1], density, mol_weight)
    n_486 = calc.refractive_index(alpha[2], density, mol_weight)

    abbe_data = {
        'abbe_number_cc2': (n_589 - 1)/(n_486 - n_656),
        'refractive_index_cc2_656': n_656,
        'refractive_index_cc2_589': n_589,
        'refractive_index_cc2_486': n_486,
        'qm_polarizability_cc2_656': alpha[0],
        'qm_polarizability_cc2_589': alpha[1],
        'qm_polarizability_cc2_486': alpha[2],
    }

    del psi4mol
    gc.collect()

    return abbe_data


def polar_sos(res, wavelength=None):
    """
    sim.qm.polar_sos

    Calculation of static/dynamic electric dipole polarizability by sum-over-states approach using TD-DFT results
    J. Phys. Chem. A 2004, 108, 11063-11072

    Args:
        res: Results of TD-DFT calculation
    
    Optional args:
        wavelength: wavelength [nm]. If None, static dipole polarizability is computed. (float)

    return
        Polarizability (float, angstrom^3)
        Polarizability tensor (ndarray, angstrom^3)
    """
    a_conv = 1.648777e-41    # a.u. -> C^2 m^2 J^-1
    pv = (a_conv*const.m2ang**3)/(4*np.pi*const.eps0)    # C^2 m^2 J^-1 -> angstrom^3 (polarizability volume)

    E = np.array([r['EXCITATION ENERGY'] for r in res])
    mu = np.array([r['ELECTRIC DIPOLE TRANSITION MOMENT (LEN)'] for r in res])
    
    if wavelength is None:
        tensor = 2*np.sum( (mu[:, np.newaxis, :] * mu[:, :, np.newaxis]) / E.reshape((-1,1,1)), axis=0 ) * pv
    else:
        Ep = const.h*const.c/(wavelength*1e-9) / 4.3597447222071e-18    # (J s) * (m/s) / (nm->m) = J -> hartree
        tensor = 2*np.sum( (mu[:, np.newaxis, :] * mu[:, :, np.newaxis]) / (E - (Ep**2)/E).reshape((-1,1,1)), axis=0 ) * pv
        
    alpha = np.mean(np.diag(tensor))
    
    return alpha, tensor


def polarizability_sos(mol, wavelength=None, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='polarizability_sos', qm_solver='psi4',
        opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'},
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
        td_method='cam-b3lyp-d3bj', td_basis='6-311+G(2d,p)', td_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
        n_state=1000, p_state=None, tda=False, tdscf_maxiter=60, td_output='polarizability_sos_tddft.json',
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.polarizability_sos

    Calculation of static/dynemic electric dipole polarizability by using TD-DFT calculation

    Args:
        mol: RDKit Mol object
    
    Optional args:
        wavelength: wavelength [nm]. If None, static dipole polarizability is computed.
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default:wb97m-d3bj)
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        opt_basis_gen: Using basis set in the optimize calculation for each element
        td_method: Using method in the TD-DFT calculation (str, default:wb97m-d3bj)
        td_basis: Using basis set in the TD-DFT calculation (str, default:6-311+G(2d,p))
        td_basis_gen: Using basis set in the TD-DFT calculation for each element
        n_state: Number of state in the TD-DFT calculation
        p_state: Number of states, which is determined by [Num. of all excitation states] * p_state (float, 0.0 < p_state <= 1.0).
                 p_state is given priority over n_state.
        tda: Run with Tamm-Dancoff approximation (TDA), uses random-phase approximation (RPA) when false (boolean)
        tdscf_maxiter: Maximum number of TDSCF solver iterations (int)

    return
        list of dict
            Frequency dependent dipole polarizability (float, angstrom^3)
            Frequency dependent dipole polarizability tensor (xx, yy, zz, xy, xz, yz) (float, angstrom^3)
    """
    polar_data = {}
    if wavelength is None:
        wavelength = [None]
    elif type(wavelength) is float or type(wavelength) is int:
        wavelength = [wavelength]
    kwargs = _set_total_charge_multiplicity(mol, total_charge, total_multiplicity, kwargs)

    psi4mol = QMw(mol, confId=confId, work_dir=work_dir, tmp_dir=tmp_dir, method=opt_method, basis=opt_basis, basis_gen=opt_basis_gen, qm_solver=qm_solver,
                    name=log_name, **kwargs)
    if opt:
        psi4mol.optimize(geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm)
        if psi4mol.error_flag:
            utils.radon_print('Psi4 optimization error in calc.polarizability_sos.', level=2)
            return polar_data

        coord = psi4mol.mol.GetConformer(confId).GetPositions()
        for i, atom in enumerate(psi4mol.mol.GetAtoms()):
            mol.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    psi4mol.method = td_method
    psi4mol.basis = td_basis
    psi4mol.basis_gen = td_basis_gen

    res = psi4mol.tddft(n_state=n_state, p_state=p_state, tda=tda, tdscf_maxiter=tdscf_maxiter)

    if psi4mol.error_flag:
        utils.radon_print('Psi4 TD-DFT calculation error in sim.qm.polarizability_sos.', level=2)

    for l in wavelength:
        alpha, tensor = polar_sos(res, wavelength=l)

        if l is None:
            p_data = {
                'qm_polarizability_sos': alpha,
                'qm_polarizability_sos_xx': tensor[0, 0],
                'qm_polarizability_sos_yy': tensor[1, 1],
                'qm_polarizability_sos_zz': tensor[2, 2],
                'qm_polarizability_sos_xy': (tensor[0, 1]+tensor[1, 0])/2,
                'qm_polarizability_sos_xz': (tensor[0, 2]+tensor[2, 0])/2,
                'qm_polarizability_sos_yz': (tensor[1, 2]+tensor[2, 1])/2,
            }
            polar_data.update(p_data)

        else:
            p_data = {
                'qm_polarizability_sos_%i' % int(l): alpha,
                'qm_polarizability_sos_%i_xx' % int(l): tensor[0, 0],
                'qm_polarizability_sos_%i_yy' % int(l): tensor[1, 1],
                'qm_polarizability_sos_%i_zz' % int(l): tensor[2, 2],
                'qm_polarizability_sos_%i_xy' % int(l): (tensor[0, 1]+tensor[1, 0])/2,
                'qm_polarizability_sos_%i_xz' % int(l): (tensor[0, 2]+tensor[2, 0])/2,
                'qm_polarizability_sos_%i_yz' % int(l): (tensor[1, 2]+tensor[2, 1])/2,
            }
            polar_data.update(p_data)

    if td_output:
        json_data = {}
        for i, r in enumerate(res):
            r['ELECTRIC DIPOLE TRANSITION MOMENT (LEN)'] = ','.join(str(x) for x in r['ELECTRIC DIPOLE TRANSITION MOMENT (LEN)'])
            r['ELECTRIC DIPOLE TRANSITION MOMENT (VEL)'] = ','.join(str(x) for x in r['ELECTRIC DIPOLE TRANSITION MOMENT (VEL)'])
            r['MAGNETIC DIPOLE TRANSITION MOMENT'] = ','.join(str(x) for x in r['MAGNETIC DIPOLE TRANSITION MOMENT'])
            del r['RIGHT EIGENVECTOR ALPHA']
            del r['LEFT EIGENVECTOR ALPHA']
            del r['RIGHT EIGENVECTOR BETA']
            del r['LEFT EIGENVECTOR BETA']
            json_data['Excitation state %i' % (i+1)] = r

        with open(os.path.join(work_dir, td_output), 'w') as fh:
            json.dump(json_data, fh, ensure_ascii=False, indent=4, separators=(',', ': '))

    del psi4mol
    gc.collect()

    return polar_data


def refractive_index_sos(mols, density, ratio=None, wavelength=None, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='refractive_index_sos',
        qm_solver='psi4', opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'},
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
        td_method='cam-b3lyp-d3bj', td_basis='6-311+G(2d,p)', td_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
        n_state=1000, p_state=None, tda=False, tdscf_maxiter=60, td_output='refractive_index_sos_tddft.json',
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.refractive_index_sos

    Calculation of refractive index by sum-over-states approach using TD-DFT calculation

    Args:
        mols: List of RDKit Mol object
        density: [g/cm^3]
    
    Optional args:
        ratio: ratio of repeating units in a copolymer
        wavelength: wavelength [nm]. If None, static dipole polarizability is computed.
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default: wb97m-d3bj)
        opt_basis_gen: Using basis set in the optimize calculation for each element
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        td_method: Using method in the TD-DFT calculation (str, default:cam-b3lyp-d3bj)
        td_basis: Using basis set in the TD-DFT calculation (str, default:6-311+G(2d,p))
        td_basis_gen: Using basis set in the TD-DFT calculation for each element
        n_state: Number of state in the TD-DFT calculation
        p_state: Number of states, which is determined by [Num. of all excitation states] * p_state (float, 0.0 < p_state <= 1.0).
                 p_state is given priority over n_state.
        tda: Run with Tamm-Dancoff approximation (TDA), uses random-phase approximation (RPA) when false (boolean)
        tdscf_maxiter: Maximum number of TDSCF solver iterations (int)

    return
        Refractive index data (list of dict)
            frequency dependent refractive index (float)
            frequency dependent dipole polarizability of repeating units (float, angstrom^3)
            frequency dependent dipole polarizability tensor of repeating units (float, angstrom^3)
    """
    if wavelength is None:
        wavelength = [None]
    elif type(wavelength) is float or type(wavelength) is int:
        wavelength = [wavelength]

    if type(mols) is Chem.Mol: mols = [mols]
    mol_weight = [calc.molecular_weight(mol) for mol in mols]
    p_data = {}
    a_list = []
    for i, mol in enumerate(mols):
        polar_data = polarizability_sos(mol, wavelength=wavelength, confId=confId, opt=opt, work_dir=work_dir, tmp_dir=tmp_dir,
                                log_name='%s_%i' % (log_name, i), qm_solver=qm_solver, opt_method=opt_method, opt_basis=opt_basis, opt_basis_gen=opt_basis_gen,
                                geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm,
                                td_method=td_method, td_basis=td_basis, td_basis_gen=td_basis_gen,
                                n_state=n_state, p_state=p_state, tda=tda, tdscf_maxiter=tdscf_maxiter, td_output=td_output,
                                total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)

        for k, v in polar_data.items():
            p_data['%s_monomer%i' % (k, i+1)] = v

        a_tmp = []
        for l in wavelength:
            if l is None:
                a_tmp.append(polar_data['qm_polarizability_sos'])
            else:
                a_tmp.append(polar_data['qm_polarizability_sos_%i' % int(l)])
        a_list.append(a_tmp)
    a_list = np.array(a_list)

    ri_data = {}
    for i, l in enumerate(wavelength): 
        if i is None:
            ri_data['refractive_index_sos'] = calc.refractive_index(a_list[:, i], density, mol_weight, ratio=ratio)
        else:
            ri_data['refractive_index_sos_%i' % int(l)] = calc.refractive_index(a_list[:, i], density, mol_weight, ratio=ratio)

    ri_data.update(p_data)

    return ri_data


def abbe_number_sos(mols, density, ratio=None, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='abbe_number_sos', qm_solver='psi4',
        opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'},
        geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
        td_method='cam-b3lyp-d3bj', td_basis='6-311+G(2d,p)', td_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
        n_state=1000, p_state=0.003, tda=False, tdscf_maxiter=60, td_output='abbe_number_sos_tddft.json',
        total_charge=None, total_multiplicity=None, **kwargs):
    """
    sim.qm.abbe_number_sos

    Calculation of abbe's number by sum-over-states approach using TD-DFT calculation

    Args:
        mols: List of RDKit Mol object
        density: [g/cm^3]
    
    Optional args:
        ratio: ratio of repeating units in a copolymer
        confID: Target conformer ID (int)
        opt: Do optimization (boolean)
        work_dir: Work directory path (str)
        omp: Num of threads of OpenMP in the quantum chemical calculation (int)
        memory: Using memory in the quantum chemical calculation (int, MB)
        opt_method: Using method in the optimize calculation (str, default: wb97m-d3bj)
        opt_basis_gen: Using basis set in the optimize calculation for each element
        opt_basis: Using basis set in the optimize calculation (str, default:6-31G(d,p))
        td_method: Using method in the TD-DFT calculation (str, default:cam-b3lyp-d3bj)
        td_basis: Using basis set in the TD-DFT calculation (str, default:6-311+G(2d,p))
        td_basis_gen: Using basis set in the TD-DFT calculation for each element
        n_state: Number of state in the TD-DFT calculation
        p_state: Number of states, which is determined by [Num. of all excitation states] * p_state (float, 0.0 < p_state <= 1.0).
                 p_state is given priority over n_state.
        tda: Run with Tamm-Dancoff approximation (TDA), uses random-phase approximation (RPA) when false (boolean)
        tdscf_maxiter: Maximum number of TDSCF solver iterations (int)

    return
        Abbe's number data (dict)
            abbe_number (float)
            refractive_index_656 (float)
            refractive_index_589 (float)
            refractive_index_486 (float)
            frequency dependent dipole polarizability of repeating units (float, angstrom^3)
            frequency dependent dipole polarizability tensor of repeating units (float, angstrom^3)
    """
    if type(mols) is Chem.Mol: mols = [mols]
    ri_data = refractive_index_sos(mols, density=density, ratio=ratio, wavelength=[656, 589, 486], confId=confId,
                            opt=opt, work_dir=work_dir, tmp_dir=tmp_dir, log_name=log_name, qm_solver=qm_solver,
                            opt_method=opt_method, opt_basis=opt_basis, opt_basis_gen=opt_basis_gen,
                            geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm,
                            td_method=td_method, td_basis=td_basis, td_basis_gen=td_basis_gen,
                            n_state=n_state, p_state=p_state, tda=tda, tdscf_maxiter=tdscf_maxiter, td_output=td_output,
                            total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)

    n_656 = ri_data['refractive_index_sos_656']
    n_589 = ri_data['refractive_index_sos_589']
    n_486 = ri_data['refractive_index_sos_486']

    abbe_data = {
        'abbe_number_sos': (n_589 - 1)/(n_486 - n_656),
        **ri_data
    }

    return abbe_data


def bond_angle_params_mseminario(
    mol: Chem.Mol,
    *,
    confId: int = 0,
    opt: bool = True,
    work_dir: Optional[str] = None,
    tmp_dir: Optional[str] = None,
    log_name: str = "bond_angle",
    qm_solver: str = "psi4",
    opt_method: str = "wb97m-d3bj",
    opt_basis: str = "6-31G(d,p)",
    opt_basis_gen: Optional[Dict[str, Any]] = None,
    geom_iter: int = 50,
    geom_conv: str = "QCHEM",
    geom_algorithm: str = "RFO",
    # Hessian-level settings (defaults to the optimize level)
    hess_method: Optional[str] = None,
    hess_basis: Optional[str] = None,
    hess_basis_gen: Optional[Dict[str, Any]] = None,
    # Modified-Seminario options
    linear_angle_deg_cutoff: float = 175.0,
    projection_mode: str = "abs",
    keep_linear_angles: bool = True,
    symmetrize_equivalents: bool = True,
    # Charge/multiplicity overrides
    total_charge: Optional[int] = None,
    total_multiplicity: Optional[int] = None,
    # Output controls
    write_itp: bool = True,
    itp_name: str = "bond_angle_params.itp",
    json_name: str = "bond_angle_params.json",
    **kwargs,
):
    """Derive *bond + angle* harmonic parameters from a Psi4 Hessian via modified Seminario.

    What it does
    ------------
    - (Optional) geometry optimization in Psi4
    - Hessian calculation in Psi4
    - Convert Hessian -> bond/angle force constants via (modified) Seminario
    - Write a small .itp fragment with [ bonds ] and [ angles ]

    Where it writes
    --------------
    To keep `work_dir/` clean, outputs go under:
      work_dir/01_qm/07_bonded_params/<log_name>/

    Returns
    -------
    dict with keys: work_dir, json, itp, params

    Notes
    -----
    - This function intentionally does **not** derive dihedrals/impropers.
    - The resulting .itp is a *fragment* intended to be included/merged.
    """

    if qm_solver.lower() != "psi4":
        raise NotImplementedError("bond_angle_params_mseminario currently supports qm_solver='psi4' only")

    # Organize outputs (yzc-gmx-gen style)
    _, task_dir = _qm_task_dir(work_dir, log_name=log_name, task="bonded_params")

    # Psi4 runner (Psi4w already nests all Psi4 artifacts under task_dir/psi4/)
    from .psi4_wrapper import Psi4w

    basis_gen = dict(opt_basis_gen or {})
    h_basis_gen = dict(hess_basis_gen or basis_gen)

    psi4_kwargs: dict = {
        "method": opt_method,
        "basis": opt_basis,
        "basis_gen": basis_gen,
    }
    psi4_kwargs = _set_total_charge_multiplicity(mol, total_charge, total_multiplicity, psi4_kwargs)
    psi4_kwargs.update(kwargs)

    psi4mol = Psi4w(mol, confId=confId, work_dir=str(task_dir), tmp_dir=tmp_dir, name="bond_angle", **psi4_kwargs)

    # Optional geometry optimization (recommended)
    opt_energy_kj = None
    if opt:
        e_kj, coord = psi4mol.optimize(
            ignore_conv_error=True,
            geom_iter=geom_iter,
            geom_conv=geom_conv,
            geom_algorithm=geom_algorithm,
        )
        opt_energy_kj = float(e_kj) if e_kj is not None else None
        try:
            conf = mol.GetConformer(int(confId))
            for ai in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(ai, Geom.Point3D(float(coord[ai, 0]), float(coord[ai, 1]), float(coord[ai, 2])))
        except Exception:
            pass

    # Hessian level (defaults to opt level, with diffuse basis for anions)
    if hess_method:
        psi4mol.method = hess_method

    eff_total_charge = None
    try:
        if total_charge is not None:
            eff_total_charge = int(total_charge)
        else:
            eff_total_charge = int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms()))
    except Exception:
        eff_total_charge = None

    eff_hess_basis = hess_basis
    eff_h_basis_gen = dict(h_basis_gen or {})
    if not eff_hess_basis:
        eff_hess_basis = opt_basis
        if eff_total_charge is not None and eff_total_charge < 0:
            base = str(opt_basis or '').strip()
            if base in ('6-31G(d,p)', '6-31+G(d,p)', 'def2-SVP', 'def2-SVPD', 'def2-TZVP', 'def2-TZVPD'):
                eff_hess_basis = 'def2-TZVPD'
            elif 'TZVP' in base and 'D' not in base:
                eff_hess_basis = base + 'D'
    psi4mol.basis = eff_hess_basis

    if eff_total_charge is not None and eff_total_charge < 0:
        if 'Br' not in eff_h_basis_gen:
            eff_h_basis_gen['Br'] = 'def2-TZVPD'
        if 'I' not in eff_h_basis_gen:
            eff_h_basis_gen['I'] = 'def2-TZVPD'
    if eff_h_basis_gen is not None:
        psi4mol.basis_gen = eff_h_basis_gen

    hess = psi4mol.hessian(wfn=True)

    params = seminario.bond_angle_params_from_hessian(
        mol,
        hess,
        confId=confId,
        linear_angle_deg_cutoff=float(linear_angle_deg_cutoff),
        projection_mode=str(projection_mode),
        keep_linear_angles=bool(keep_linear_angles),
        symmetrize_equivalents=bool(symmetrize_equivalents),
    )
    params.setdefault("meta", {})
    params["meta"].update(
        {
            "opt": bool(opt),
            "opt_energy_kj_mol": opt_energy_kj,
            "opt_method": opt_method,
            "opt_basis": opt_basis,
            "hess_method": hess_method or opt_method,
            "hess_basis": eff_hess_basis,
            "linear_angle_deg_cutoff": float(linear_angle_deg_cutoff),
            "projection_mode": str(projection_mode),
            "keep_linear_angles": bool(keep_linear_angles),
            "symmetrize_equivalents": bool(symmetrize_equivalents),
        }
    )

    json_path = (task_dir / str(json_name)).resolve()
    json_path.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    itp_path = None
    if write_itp:
        itp_path = (task_dir / str(itp_name)).resolve()
        seminario.write_bond_angle_itp(mol, params, itp_path, comment="generated by yadonpy mseminario (bond+angle)")

    # Attach patch file paths to the molecule so downstream writers can inject it.
    try:
        if itp_path:
            mol.SetProp('_yadonpy_mseminario_itp', str(itp_path))
        mol.SetProp('_yadonpy_mseminario_json', str(json_path))
    except Exception:
        pass

    # Best-effort: also apply the derived params to the in-memory mol (bonds/angles)
    # so advanced callers can use them directly.
    try:
        apply_mseminario_params_to_mol(mol, params)
    except Exception:
        pass

    return {
        "work_dir": str(task_dir),
        "json": str(json_path),
        "itp": str(itp_path) if itp_path else None,
        "params": params,
    }



def bond_angle_params_drih(
    mol: Chem.Mol,
    *,
    confId: int = 0,
    work_dir: str,
    log_name: str,
    smiles_hint: str = None,
    k_bond_kj_mol_nm2: float = 350000.0,
    k_angle_kj_mol_rad2: float = 2500.0,
    k_angle_linear_kj_mol_rad2: float = 6000.0,
    stiffness_scale: float = 1.0,
    write_itp: bool = True,
    itp_name: str = "bonded_drih_patch.itp",
    json_name: str = "bonded_drih_params.json",
) -> dict:
    """DRIH-like robust bonded parameterization for high-symmetry inorganic ions.

    The early implementation used one global set of fixed force constants.
    That was simple, but it made all AX4/AX6 ions look mechanically identical and
    was also vulnerable to cache mix-ups with plain GAFF artifacts. This version
    keeps the same low-dependency, no-Hessian philosophy, but strengthens it by:

    - exact geometry symmetrization for AX4/AX6 motifs;
    - species-aware presets for common ions (PF6-, BF4-, ClO4-, AsF6-, SbF6-);
    - mild bond-length scaling so unusually long/short geometries do not reuse a
      completely inappropriate stiffness;
    - explicit metadata that downstream caches/exporters can validate.
    """
    from ..core import utils as _u
    import numpy as np
    from pathlib import Path

    task_dir = Path(work_dir) / "01_qm" / "07_bonded_params" / str(log_name)
    task_dir.mkdir(parents=True, exist_ok=True)

    # Ensure geometry is symmetrized if possible.
    try:
        if _u.is_high_symmetry_polyhedral_ion(mol, smiles_hint=smiles_hint):
            _u.symmetrize_polyhedral_ion_geometry(mol, confId=int(confId))
    except Exception:
        pass

    hit = None
    try:
        hit = _u._detect_ax_polyhedron(mol)  # internal helper (center, lig_idxs, cn)
    except Exception:
        hit = None
    if hit is None:
        raise ValueError("DRIH bonded params currently supports only AX4/AX6 polyhedral ions.")
    center_idx, lig_idxs, cn = hit
    conf = mol.GetConformer(int(confId))

    def _pos(i: int) -> np.ndarray:
        p = conf.GetAtomPosition(int(i))
        return np.array([p.x, p.y, p.z], dtype=float)

    center_atom = mol.GetAtomWithIdx(int(center_idx))
    ligand_atom = mol.GetAtomWithIdx(int(lig_idxs[0]))
    center_sym = str(center_atom.GetSymbol())
    ligand_sym = str(ligand_atom.GetSymbol())

    cpos = _pos(center_idx)
    lig_pos = [_pos(i) for i in lig_idxs]

    # Species-aware presets. Values remain conservative GROMACS harmonic terms.
    preset_map = {
        ('P', 'F', 6): dict(k_bond=430000.0, k_cis=3200.0, k_trans=9500.0, tag='PF6'),
        ('As', 'F', 6): dict(k_bond=390000.0, k_cis=3000.0, k_trans=9000.0, tag='AsF6'),
        ('Sb', 'F', 6): dict(k_bond=370000.0, k_cis=2900.0, k_trans=8600.0, tag='SbF6'),
        ('B', 'F', 4): dict(k_bond=470000.0, k_cis=3400.0, k_trans=3400.0, tag='BF4'),
        ('Cl', 'O', 4): dict(k_bond=520000.0, k_cis=3200.0, k_trans=3200.0, tag='ClO4'),
    }
    preset = dict(preset_map.get((center_sym, ligand_sym, int(cn)), {}))
    if not preset:
        preset = dict(k_bond=float(k_bond_kj_mol_nm2), k_cis=float(k_angle_kj_mol_rad2), k_trans=float(k_angle_linear_kj_mol_rad2), tag='generic')
    else:
        # Allow user-level scaling on top of the species preset.
        preset['k_bond'] = float(preset['k_bond'])
        preset['k_cis'] = float(preset['k_cis'])
        preset['k_trans'] = float(preset['k_trans'])

    # Mild geometry scaling so obviously stretched/compressed coordinates do not
    # inherit an unphysical preset unchanged.
    r0s_a = [float(np.linalg.norm(p - cpos)) for p in lig_pos]
    r0_avg_a = float(np.mean(r0s_a))
    if r0_avg_a <= 1.0e-8:
        raise ValueError('DRIH failed: zero bond length detected')
    ref_a = 1.60 if cn == 6 else 1.45
    length_scale = (ref_a / r0_avg_a) ** 4
    length_scale = float(min(1.40, max(0.70, length_scale)))
    total_scale = float(max(0.10, stiffness_scale)) * length_scale

    k_bond_eff = float(preset['k_bond']) * total_scale
    k_cis_eff = float(preset['k_cis']) * total_scale
    k_trans_eff = float(preset['k_trans']) * total_scale

    # Bonds: all center-ligand, symmetrized to the average radius.
    r0_nm = float(r0_avg_a) * 0.1  # Angstrom -> nm
    bonds = [
        {"i": int(center_idx), "j": int(i), "r0_nm": float(r0_nm), "k_kj_mol_nm2": float(k_bond_eff)}
        for i in lig_idxs
    ]

    # Angles: all ligand-center-ligand.
    angles = []
    n_cis = 0
    n_trans = 0
    for a in range(len(lig_idxs)):
        for b in range(a + 1, len(lig_idxs)):
            i = int(lig_idxs[a])
            k = int(lig_idxs[b])
            v1 = lig_pos[a] - cpos
            v2 = lig_pos[b] - cpos
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-8 or n2 < 1e-8:
                continue
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            cosang = max(-1.0, min(1.0, cosang))
            ang = float(np.degrees(np.arccos(cosang)))
            if cn == 6:
                if ang > 150.0:
                    th0 = 180.0
                    kk = float(k_trans_eff)
                    n_trans += 1
                else:
                    th0 = 90.0
                    kk = float(k_cis_eff)
                    n_cis += 1
            else:
                th0 = 109.471
                kk = float(k_cis_eff)
                n_cis += 1
            angles.append({"i": i, "j": int(center_idx), "k": k, "theta0_deg": float(th0), "k_kj_mol_rad2": float(kk)})

    params = {
        "meta": {
            "method": "DRIH",
            "cn": int(cn),
            "center_idx": int(center_idx),
            "ligand_indices": [int(x) for x in lig_idxs],
            "center_symbol": center_sym,
            "ligand_symbol": ligand_sym,
            "preset": str(preset.get('tag', 'generic')),
            "r0_avg_angstrom": float(r0_avg_a),
            "length_scale": float(length_scale),
            "stiffness_scale": float(stiffness_scale),
            "k_bond_kj_mol_nm2": float(k_bond_eff),
            "k_angle_kj_mol_rad2": float(k_cis_eff),
            "k_angle_linear_kj_mol_rad2": float(k_trans_eff),
            "n_cis_angles": int(n_cis),
            "n_trans_angles": int(n_trans),
        },
        "bonds": bonds,
        "angles": angles,
    }

    json_path = (task_dir / str(json_name)).resolve()
    json_path.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    itp_path = None
    if write_itp:
        itp_path = (task_dir / str(itp_name)).resolve()
        seminario.write_bond_angle_itp(mol, params, itp_path, comment="generated by yadonpy DRIH (bond+angle)")

    try:
        if itp_path:
            mol.SetProp('_yadonpy_bonded_itp', str(itp_path))
        mol.SetProp('_yadonpy_bonded_json', str(json_path))
        mol.SetProp('_yadonpy_bonded_method', 'DRIH')
        mol.SetProp('_yadonpy_bonded_signature', 'drih')
    except Exception:
        pass

    try:
        apply_mseminario_params_to_mol(mol, params)
    except Exception:
        pass

    return {"work_dir": str(task_dir), "json": str(json_path), "itp": str(itp_path) if itp_path else None, "params": params}

def apply_mseminario_params_to_mol(mol: Chem.Mol, params: Dict[str, Any], *, overwrite: bool = True) -> None:
    """Apply (m)Seminario bond+angle parameters onto an RDKit mol.

    This is **best effort** and primarily intended to prevent downstream ff_assign()
    from wiping out the QM-derived stiffness for small rigid ions.

    - bonds: set RDKit bond props "ff_r0" (nm) and "ff_k" (kJ/mol/nm^2)
    - angles: if mol has a .angles dict (RadonPy-style), overwrite corresponding entries
    """

    # Local import to avoid circular deps during module import.
    from ..ff.ff_class import Angle_harmonic
    from ..core import utils as core_utils

    # Bonds (indices are 0-based in params)
    for bpar in params.get("bonds", []) or []:
        try:
            i = int(bpar["i"])
            j = int(bpar["j"])
        except Exception:
            continue
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is None:
            continue
        if (not overwrite) and (bond.HasProp("ff_k") or bond.HasProp("ff_r0")):
            continue
        try:
            bond.SetDoubleProp("ff_r0", float(bpar["r0_nm"]))
            bond.SetDoubleProp("ff_k", float(bpar["k_kj_mol_nm2"]))
        except Exception:
            pass

    # Angles (indices are 0-based)
    if hasattr(mol, "angles") and isinstance(getattr(mol, "angles"), dict):
        # Build a map from (a,b,c) (0-based) to internal key used by mol.angles.
        ang_key_map: Dict[tuple, Any] = {}
        for k, ang in mol.angles.items():
            try:
                a = int(getattr(ang, "a"))
                b = int(getattr(ang, "b"))
                c = int(getattr(ang, "c"))
                ang_key_map[(a, b, c)] = k
                ang_key_map[(c, b, a)] = k
            except Exception:
                continue
        for apar in params.get("angles", []) or []:
            try:
                if "i" in apar or "j" in apar or "k" in apar:
                    a = int(apar["i"])
                    b = int(apar["j"])
                    c = int(apar["k"])
                else:
                    a = int(apar["a"])
                    b = int(apar["b"])
                    c = int(apar["c"])
            except Exception:
                continue
            k = ang_key_map.get((a, b, c))
            if k is None:
                continue
            if (not overwrite) and (k in mol.angles):
                # If caller doesn't want overwrite, keep existing.
                continue
            try:
                # Preserve the container (utils.Angle) if present, otherwise
                # fall back to replacing the value.
                new_ff = Angle_harmonic(
                    ff_type="harm",
                    k=float(apar["k_kj_mol_rad2"]),
                    theta0=float(apar["theta0_deg"]),
                )
                old = mol.angles.get(k)
                if old is not None and hasattr(old, "a") and hasattr(old, "b") and hasattr(old, "c"):
                    mol.angles[k] = core_utils.Angle(a=int(old.a), b=int(old.b), c=int(old.c), ff=new_ff)
                else:
                    mol.angles[k] = new_ff
            except Exception:
                pass

def _psi4_basis_exists(basis: str, elements: Optional[List[str]] = None) -> bool:
    """Return True if Psi4/QCDB can locate the basis set name for given elements.
    Uses a tiny molecule containing requested elements and tries to build the basis.
    """
    try:
        import psi4
        # Build a minimal molecule that contains at least one of each element we care about.
        elems = elements or ["H"]
        geom_lines = []
        for i, el in enumerate(elems):
            geom_lines.append(f"{el} {i*1.5:.3f} 0.0 0.0")
        mol = psi4.geometry("\n".join(geom_lines))
        # Quiet build attempt
        _ = psi4.core.BasisSet.build(mol, "ORBITAL", basis, quiet=True)
        return True
    except Exception:
        # BasisSetNotFound or any build error => treat as missing.
        return False

def _pick_first_available_basis(candidates: List[str], elements: Optional[List[str]] = None) -> str:
    try:
        import psi4  # noqa: F401
    except Exception:
        # In lightweight/test environments without Psi4, keep the requested policy
        # route rather than silently downgrading anions away from diffuse bases.
        return candidates[0]
    for b in candidates:
        if _psi4_basis_exists(b, elements=elements):
            return b
    # Fall back to the last candidate (even if missing), let Psi4 error be explicit.
    return candidates[-1]
