"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

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
from typing import Optional, Dict, Any, List, Tuple
from rdkit import Chem
from rdkit import Geometry as Geom
from ..core import utils, const, calc
from .qm_wrapper import QMw
from . import seminario



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


def _save_atomic_charges_json(mol, path, *, charge_label: str, log_name: str):
    """Persist per-atom charges to JSON for resumable workflows.

    Charges are stored in RDKit atom double-prop 'AtomicCharge'.
    """
    from pathlib import Path

    p = Path(path)
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

    # IMPORTANT: persist (m)Seminario patch metadata so that a resumed workflow
    # can still inject the QM-derived bond/angle params after ff_assign().
    for k in ("_yadonpy_mseminario_itp", "_yadonpy_mseminario_json"):
        try:
            if mol.HasProp(k):
                v = str(mol.GetProp(k))
                if v:
                    meta[k] = v
        except Exception:
            pass
    p.write_text(json.dumps({"meta": meta, "charges": charges}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_atomic_charges_json(mol, path, *, strict: bool = True) -> bool:
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
        n = min(len(charges), mol.GetNumAtoms())
        for i in range(n):
            mol.GetAtomWithIdx(i).SetDoubleProp('AtomicCharge', float(charges[i]))

        # Restore QM patch metadata (e.g. (m)Seminario bond/angle fragment) so
        # downstream topology writers can inject it.
        try:
            meta = obj.get('meta') if isinstance(obj, dict) else None
            if isinstance(meta, dict):
                for k in ("_yadonpy_mseminario_itp", "_yadonpy_mseminario_json"):
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
    opt_method='wb97m-d3bj',
    opt_basis='6-31G(d,p)',
    opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'},
    geom_iter=50,
    geom_conv='QCHEM',
    geom_algorithm='RFO',
    # RESP/ESP level (single point)
    charge_method='wb97m-d3bj',
    charge_basis='def2-TZVP',
    charge_basis_gen={'Br': 'def2-TZVP', 'I': 'lanl2dz'},
    # behavior toggles
    auto_level: bool = True,
    bonded_params: str = 'auto',
    total_charge=None,
    total_multiplicity=None,
    symmetrize=True,
    symmetrize_geometry: bool = True,
    **kwargs,
):
    """
    sim.qm.assign_charges

    Assignment atomic charge for RDKit Mol object
    This is wrapper function of core.calc.assign_charges

    Args:
        mol: RDKit Mol object

    Optional args:
        charge: Select charge type of gasteiger, RESP, ESP, Mulliken, Lowdin, or zero (str, default:RESP)
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

    # If the caller didn't provide an explicit name, use (and persist) a stable
    # molecule name. If no name was set, we infer the caller's Python variable
    # name (e.g., solvent_A) best-effort.
    if log_name is None:
        try:
            log_name = utils.ensure_name(mol, name=None, depth=1)
        except Exception:
            log_name = None
    if not log_name:
        log_name = 'charge'

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
    utils.yadon_print(f"QM task: assign_charges (charge={charge}, opt={opt}) purpose={log_name} smiles={_smi}", level=1)

    # ------------------------------------------------------------------
    # Work dir hygiene: keep work_dir clean by writing QM artifacts under
    #   work_dir/01_qm/<log_name>/charge/
    # ------------------------------------------------------------------
    work_dir_root = None
    if work_dir is not None:
        work_dir_root, work_dir = _qm_task_dir(work_dir, log_name=str(log_name), task="charge")
        if tmp_dir is None:
            tmp_dir = work_dir

    # ------------------------------------------------------------------
    # For small inorganic ions (PF6-, BF4-, ClO4-...) RDKit/MMFF can be
    # unstable. Prefer OpenBabel-based 3D building when possible.
    # This enables skipping conformer search for anions in example workflows.
    # ------------------------------------------------------------------
    try:
        smiles_hint = None
        if isinstance(_smi, str) and _smi not in ("?", ""):
            smiles_hint = _smi
        if mol.GetNumConformers() == 0 or utils.is_inorganic_ion_like(mol, smiles_hint=smiles_hint):
            utils.ensure_3d_coords(mol, smiles_hint=smiles_hint, engine='openbabel')
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Auto-level selection for small inorganic ions (yzc-gmx-gen style)
    #
    # Rationale:
    # - RESP via HF/6-31G(d) is no longer recommended for typical electrolyte
    #   systems; use a modern dispersion-corrected functional and a larger,
    #   diffuse basis at the ESP level.
    # - For small, high-symmetry anions like PF6-, use a robust OPT level
    #   (SVPD, preferably ma-*) and a larger single-point basis.
    #
    # Users can still fully override by passing auto_level=False or explicit
    # method/basis arguments.
    # ------------------------------------------------------------------
    is_inorganic = False
    is_poly_ion = False
    fc = 0
    try:
        is_inorganic = utils.is_inorganic_ion_like(mol, smiles_hint=smiles_hint)
        is_poly_ion = utils.is_inorganic_polyatomic_ion(mol, smiles_hint=smiles_hint)
        for a in mol.GetAtoms():
            fc += int(a.GetFormalCharge())
    except Exception:
        pass

    if auto_level and is_inorganic:
        # Keep functional consistent across species (RadonPy-style).
        # We only auto-adjust **basis sets** for numeric stability (diffuse functions).

        # Inorganic anions (PF6-, BF4-, ClO4-...): keep it SIMPLE and robust.
        # User preference:
        #   OPT: 6-31+G(d,p)
        #   RESP(ESP) single point: 6-311+G(2d,p)
        # Both include diffuse (+) and polarization; functional remains as provided.
        if fc < 0:
            if str(opt_basis).lower() in (
                "6-31g(d,p)", "6-31g(d)", "6-31+g(d,p)", "def2-svp", "def2-svpd",
                "ma-def2-svpd", "ma-def2-svp",
            ):
                opt_basis = "6-31+G(d,p)"
            if str(charge_basis).lower() in (
                "6-31g(d)", "def2-tzvp", "def2-tzvpd", "ma-def2-tzvppd", "ma-def2-tzvpd",
                "6-311+g(2d,p)", "6-311+g(d,p)",
            ):
                charge_basis = "6-311+G(2d,p)"
        else:
            # Inorganic cations: keep moderate ESP basis
            if str(charge_basis).lower() in ("6-31g(d)",):
                charge_basis = "def2-TZVP"

    # Echo the chosen levels to screen
    utils.yadon_print(
        f"QM levels: OPT={str(opt_method)}/{str(opt_basis)} | RESP(ESP)={str(charge_method)}/{str(charge_basis)}",
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
            **kwargs,
        )

    flag = _attempt_levels(opt_basis, charge_basis)
    if (not flag) and auto_level and is_inorganic:
        # Minimal, robust fallback ladder (avoid over-complicated basis shopping).
        # 1) Keep the functional, relax basis to commonly available sets.
        trials = [
            ("6-31G(d,p)", "6-311+G(2d,p)"),
            ("6-31+G(d,p)", "6-311+G(d,p)"),
            ("6-31G(d,p)", "6-311+G(d,p)"),
            ("def2-SVPD", "def2-TZVPPD"),
            ("def2-SVPD", "def2-TZVPD"),
            ("def2-SVPD", "def2-TZVP"),
        ]
        for ob, cb in trials:
            utils.yadon_print(f"QM retry with basis: OPT={ob} | RESP(ESP)={cb}", level=2)
            if _attempt_levels(ob, cb):
                opt_basis, charge_basis, flag = ob, cb, True
                break

    # Re-apply geometry symmetrization after QM optimization (if any).
    if flag and symmetrize_geometry and is_poly_ion:
        try:
            if utils.is_high_symmetry_polyhedral_ion(mol, smiles_hint=smiles_hint):
                utils.symmetrize_polyhedral_ion_geometry(mol, confId=confId)
        except Exception:
            pass

    # Optionally symmetrize charges across topologically equivalent atoms.
    # This makes highly symmetric ions (e.g., PF6-) assign identical charges to symmetry-equivalent atoms.
    if flag and symmetrize:
        try:
            ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
            groups = {}
            for i, r in enumerate(ranks):
                groups.setdefault(int(r), []).append(i)
            # collect charges
            charges = []
            for a in mol.GetAtoms():
                if a.HasProp('AtomicCharge'):
                    charges.append(float(a.GetDoubleProp('AtomicCharge')))
                elif a.HasProp('atom_charge'):
                    charges.append(float(a.GetDoubleProp('atom_charge')))
                else:
                    charges.append(None)
            if any(c is None for c in charges):
                raise RuntimeError('AtomicCharge not found')
            n_symm = 0
            for _, idxs in groups.items():
                if len(idxs) <= 1:
                    continue
                qavg = float(sum(charges[i] for i in idxs) / len(idxs))
                for i in idxs:
                    mol.GetAtomWithIdx(i).SetDoubleProp('AtomicCharge', qavg)
                    charges[i] = qavg
                n_symm += 1
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
                    except Exception:
                        pass
        except Exception as e:
            # Non-fatal: keep the workflow moving; topology will fall back to GAFF.
            utils.yadon_print(f"QM warning: bonded-params({_bp}) failed for {log_name}: {e}", level=2)

    # Best-effort: export a charged MOL2 after charge assignment.
    try:
        # Export a charged MOL2 + JSON in a predictable module folder.
        if work_dir_root is not None:
            from ..io.mol2 import write_mol2_from_rdkit

            # Keep QM module exports grouped and sortable.
            d = Path(work_dir_root) / "01_qm" / "90_charged_mol2"
            d.mkdir(parents=True, exist_ok=True)
            write_mol2_from_rdkit(mol=mol, out_mol2=d / f"{log_name}.mol2", mol_name=str(log_name))

            # Also write JSON charges for easy resuming.
            try:
                _save_atomic_charges_json(mol, d / f"{log_name}.charges.json", charge_label=str(charge), log_name=str(log_name))
            except Exception:
                pass
    except Exception:
        pass

    # Make failure loud by default so user scripts can stay linear and clean
    # (RadonPy-style) without carrying intermediate "ok" variables.
    if not flag:
        raise RuntimeError(f"Charge assignment failed (charge={charge}) for {log_name}")

    return flag
        

def conformation_search(mol, ff=None, nconf=1000, dft_nconf=4, etkdg_ver=2, rmsthresh=0.5, tfdthresh=0.02, clustering='TFD', qm_solver='psi4',
    opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'},
    geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO', log_name=None, work_dir=None, tmp_dir=None,
    etkdg_omp=-1, psi4_omp=-1, psi4_mp=0, mm_mp=0, memory=1000, mm_solver='rdkit', gmx_refine_n=0, gmx_ntomp=None, gmx_ntmpi=None, gmx_gpu_id=None, total_charge=None, total_multiplicity=None, **kwargs):
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
            log_name = utils.ensure_name(mol, name=None, depth=1)
        except Exception:
            log_name = 'mol'

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
    utils.yadon_print(f"QM task: conformation_search purpose={log_name} smiles={_smi}", level=1)
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
    utils.yadon_print(f"QM task: conformation_search (nconf={nconf}, dft_nconf={dft_nconf}) purpose={log_name} smiles={_smi}", level=1)

    # ------------------------------------------------------------------
    # Work dir hygiene: keep work_dir clean by writing QM artifacts under
    #   work_dir/01_qm/<log_name>/confsearch/
    # ------------------------------------------------------------------
    if work_dir is not None:
        _root, work_dir = _qm_task_dir(work_dir, log_name=str(log_name), task="confsearch")
        if tmp_dir is None:
            tmp_dir = work_dir


    mol, energy = calc.conformation_search(mol, ff=ff, nconf=nconf, dft_nconf=dft_nconf, etkdg_ver=etkdg_ver, rmsthresh=rmsthresh, qm_solver=qm_solver,
                tfdthresh=tfdthresh, clustering=clustering, opt_method=opt_method, opt_basis=opt_basis,
                opt_basis_gen=opt_basis_gen, geom_iter=geom_iter, geom_conv=geom_conv, geom_algorithm=geom_algorithm, log_name=log_name, work_dir=work_dir, tmp_dir=tmp_dir,
                etkdg_omp=etkdg_omp, psi4_omp=psi4_omp, psi4_mp=psi4_mp, mm_mp=mm_mp, memory=memory,
                mm_solver=mm_solver, gmx_refine_n=gmx_refine_n, gmx_ntomp=gmx_ntomp, gmx_ntmpi=gmx_ntmpi, gmx_gpu_id=gmx_gpu_id,
                total_charge=total_charge, total_multiplicity=total_multiplicity, **kwargs)

    # Downstream polymer builders (e.g. random_walk_polymerization) expect a *single* 3D conformer.
    # RDKit's CombineMols warns (and may drop coords) if molecules have different numbers of conformers.
    # Keep only the lowest-energy conformer (ID 0) to make behavior deterministic and robust.
    try:
        n_c = int(mol.GetNumConformers())
        if n_c > 1:
            for cid in range(n_c - 1, 0, -1):
                mol.RemoveConformer(cid)
    except Exception:
        pass

    return mol, energy


def sp_prop(mol, confId=0, opt=True, work_dir=None, tmp_dir=None, log_name='sp_prop', qm_solver='psi4',
    opt_method='wb97m-d3bj', opt_basis='6-31G(d,p)', opt_basis_gen={'Br': '6-31G(d,p)', 'I': 'lanl2dz'}, 
    geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO',
    sp_method='wb97m-d3bj', sp_basis='6-311G(d,p)', sp_basis_gen={'Br': '6-311G(d,p)', 'I': 'lanl2dz'},
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
    mol_weight = [calc.molecular_weight(mol) for mol in mols]

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

    # Hessian level (defaults to opt level)
    if hess_method:
        psi4mol.method = hess_method
    if hess_basis:
        psi4mol.basis = hess_basis
    if h_basis_gen is not None:
        psi4mol.basis_gen = h_basis_gen

    hess = psi4mol.hessian(wfn=True)

    params = seminario.bond_angle_params_from_hessian(
        mol,
        hess,
        confId=confId,
        linear_angle_deg_cutoff=float(linear_angle_deg_cutoff),
    )
    params.setdefault("meta", {})
    params["meta"].update(
        {
            "opt": bool(opt),
            "opt_energy_kj_mol": opt_energy_kj,
            "opt_method": opt_method,
            "opt_basis": opt_basis,
            "hess_method": hess_method or opt_method,
            "hess_basis": hess_basis or opt_basis,
            "linear_angle_deg_cutoff": float(linear_angle_deg_cutoff),
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
    write_itp: bool = True,
    itp_name: str = "bonded_drih_patch.itp",
    json_name: str = "bonded_drih_params.json",
) -> dict:
    """DRIH-like robust bonded parameterization for high-symmetry inorganic ions.

    This is a **geometry-driven** stiffening method intended for AX4/AX6 ions
    (PF6-, BF4-, ClO4-, AsF6- ...). It does not require a Hessian, and it
    enforces symmetry by averaging all equivalent bonds/angles.

    Output format matches Seminario writer: params['bonds'/'angles'].
    """
    from ..core import utils as _u
    import numpy as np
    from pathlib import Path

    task_dir = Path(work_dir) / "01_qm" / "07_bonded_params" / str(log_name)
    task_dir.mkdir(parents=True, exist_ok=True)

    # Ensure geometry is symmetrized if possible
    try:
        if _u.is_high_symmetry_polyhedral_ion(mol, smiles_hint=smiles_hint):
            _u.symmetrize_polyhedral_ion_geometry(mol, confId=int(confId))
    except Exception:
        pass

    # Detect AX polyhedron
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

    cpos = _pos(center_idx)
    lig_pos = [_pos(i) for i in lig_idxs]

    # Bonds: all center-ligand
    r0s = [float(np.linalg.norm(p - cpos)) for p in lig_pos]
    r0_nm = float(np.mean(r0s)) * 0.1  # Angstrom -> nm
    bonds = []
    for i in lig_idxs:
        bonds.append({"i": int(center_idx), "j": int(i), "r0_nm": float(r0_nm), "k_kj_mol_nm2": float(k_bond_kj_mol_nm2)})

    # Angles: all ligand-center-ligand
    angles = []
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
                # classify trans vs cis by angle
                if ang > 150.0:
                    th0 = 180.0
                    kk = float(k_angle_linear_kj_mol_rad2)
                else:
                    th0 = 90.0
                    kk = float(k_angle_kj_mol_rad2)
            else:
                # tetrahedral
                th0 = 109.471
                kk = float(k_angle_kj_mol_rad2)
            angles.append({"i": i, "j": int(center_idx), "k": k, "theta0_deg": float(th0), "k_kj_mol_rad2": float(kk)})

    params = {
        "meta": {
            "method": "DRIH",
            "cn": int(cn),
            "center_idx": int(center_idx),
            "ligand_indices": [int(x) for x in lig_idxs],
            "k_bond_kj_mol_nm2": float(k_bond_kj_mol_nm2),
            "k_angle_kj_mol_rad2": float(k_angle_kj_mol_rad2),
            "k_angle_linear_kj_mol_rad2": float(k_angle_linear_kj_mol_rad2),
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

    # Attach patch paths for downstream writers
    try:
        if itp_path:
            mol.SetProp('_yadonpy_bonded_itp', str(itp_path))
        mol.SetProp('_yadonpy_bonded_json', str(json_path))
        mol.SetProp('_yadonpy_bonded_method', 'DRIH')
    except Exception:
        pass

    # Also apply to mol props (best-effort)
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
        from psi4.driver.qcdb.exceptions import BasisSetNotFound
        # Build a minimal molecule that contains at least one of each element we care about.
        elems = elements or ["H"]
        geom_lines = []
        for i, el in enumerate(elems):
            geom_lines.append(f"{el} {i*1.5:.3f} 0.0 0.0")
        mol = psi4.geometry("\n".join(geom_lines))
        # Quiet build attempt
        _ = psi4.core.BasisSet.build(mol, "ORBITAL", basis, quiet=True)
        return True
    except Exception as e:
        # BasisSetNotFound or any build error => treat as missing.
        return False

def _pick_first_available_basis(candidates: List[str], elements: Optional[List[str]] = None) -> str:
    for b in candidates:
        if _psi4_basis_exists(b, elements=elements):
            return b
    # Fall back to the last candidate (even if missing), let Psi4 error be explicit.
    return candidates[-1]



