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
# core.poly module
# ******************************************************************************

import numpy as np
import pandas as pd
import re
import json
from copy import copy
import itertools
import datetime
import time
import multiprocessing as MP
import concurrent.futures as confu
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import Geometry as Geom
from rdkit import RDLogger
from . import calc, const, utils
from .molspec import MolSpec, as_rdkit_mol, molecular_weight
from .polyelectrolyte import annotate_polyelectrolyte_metadata, build_residue_map, get_charge_groups, get_polyelectrolyte_summary, get_resp_constraints, uses_localized_charge_groups
from .resources import core_data_path
from ..ff.gaff2_mod import GAFF2_mod
from ..runtime import resolve_restart
from ..schema_versions import AMORPHOUS_CELL_SCHEMA_VERSION


def _resolve_work_dir_path(work_dir):
    if work_dir is None:
        return None
    try:
        return Path(work_dir).expanduser().resolve()
    except Exception:
        return None


def _resolve_mol_like(mol):
    if isinstance(mol, MolSpec):
        resolved = mol.get_resolved_mol(strict=False)
        if resolved is None:
            utils.radon_print(
                'Input MolSpec has not been resolved into an RDKit Mol yet. '
                'Call ff.ff_assign(spec) first or use ff.mol_rdkit(...).',
                level=3,
            )
        return resolved
    return mol


def _resolve_mol_list(mols):
    if isinstance(mols, Chem.Mol):
        return [mols]
    if isinstance(mols, MolSpec):
        return [_resolve_mol_like(mols)]
    if isinstance(mols, list):
        out = []
        for m in mols:
            out.append(_resolve_mol_like(m))
        return out
    return mols


def _normalize_mol_counts(mols, n):
    mols = _resolve_mol_list(mols)
    if type(mols) is Chem.Mol:
        mols = [mols]
    elif not isinstance(mols, list):
        mols = list(mols)

    if type(n) is int:
        n = [int(n)]
    elif not isinstance(n, list):
        n = [int(x) for x in n]
    else:
        n = [int(x) for x in n]

    if len(mols) != len(n):
        raise ValueError(f'mols/n length mismatch: {len(mols)} species vs {len(n)} counts')
    return mols, n


def _cache_artifacts_best_effort(mols, *, prefer_var: bool = False) -> None:
    try:
        from ..io.molecule_cache import ensure_cached_artifacts

        for mol in mols:
            mol_name = None
            try:
                if prefer_var:
                    mol_name = utils.ensure_name(mol, name=None, depth=2, prefer_var=True)
                else:
                    if hasattr(mol, 'HasProp') and mol.HasProp('_yadonpy_resname'):
                        mol_name = str(mol.GetProp('_yadonpy_resname'))
                    elif hasattr(mol, 'HasProp') and mol.HasProp('_Name'):
                        mol_name = str(mol.GetProp('_Name'))
                    elif hasattr(mol, 'HasProp') and mol.HasProp('name'):
                        mol_name = str(mol.GetProp('name'))
            except Exception:
                mol_name = None
            ensure_cached_artifacts(mol, mol_name=mol_name)
    except Exception:
        # Caching is a best-effort optimization and should never break packing.
        pass


def _amorphous_pack_priority(mol) -> tuple[float, float, float, int]:
    nat = 0
    try:
        nat = int(mol.GetNumAtoms())
    except Exception:
        nat = 0
    max_span = 0.0
    bbox_volume = 0.0
    mw = 0.0
    try:
        conf = mol.GetConformer(0)
        coords = np.asarray(conf.GetPositions(), dtype=float)
        if coords.size > 0:
            span = np.ptp(coords, axis=0)
            max_span = float(np.max(span))
            bbox_volume = float(np.prod(np.maximum(span, 1.0e-3)))
    except Exception:
        pass
    try:
        mw = float(molecular_weight(mol))
    except Exception:
        mw = 0.0
    return (bbox_volume, max_span, mw, nat)


def _amorphous_pack_order(mols, n) -> tuple[int, ...]:
    ranked: list[tuple[tuple[float, float, float, int], int]] = []
    for idx, (mol, count) in enumerate(zip(mols, n)):
        if int(count) <= 0:
            continue
        ranked.append((_amorphous_pack_priority(mol), int(idx)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return tuple(idx for _, idx in ranked)


def _estimate_total_atoms(mols, n) -> int:
    total = 0
    for mol, count in zip(mols, n):
        try:
            total += int(mol.GetNumAtoms()) * int(count)
        except Exception:
            continue
    return int(total)


def _estimate_mol_bbox_span(mol) -> np.ndarray:
    try:
        coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=float)
    except Exception:
        return np.zeros(3, dtype=float)
    if coords.size == 0:
        return np.zeros(3, dtype=float)
    return np.asarray(np.ptp(coords, axis=0), dtype=float)


def _estimate_vdw_reference_clearance(mols, *, dist_min: float) -> float:
    periodic = Chem.GetPeriodicTable()
    best = float(dist_min)
    for mol in mols:
        for atom in mol.GetAtoms():
            try:
                r = float(periodic.GetRvdw(int(atom.GetAtomicNum())))
            except Exception:
                r = 1.5
            best = max(best, 2.0 * r)
    return float(best)


def _classify_pack_phase(mol) -> str:
    try:
        nat = int(mol.GetNumAtoms())
    except Exception:
        nat = 0
    if nat >= 80:
        return 'backbone'
    if nat <= 8:
        return 'filler'
    return 'matrix'


def _amorphous_pack_batches(mols, n) -> list[dict]:
    phased: dict[str, list[int]] = {'backbone': [], 'matrix': [], 'filler': []}
    for idx in _amorphous_pack_order(mols, n):
        phase = _classify_pack_phase(mols[idx])
        phased.setdefault(phase, []).append(int(idx))
    out: list[dict] = []
    for phase in ('backbone', 'matrix', 'filler'):
        if phased.get(phase):
            out.append({'phase': phase, 'indices': tuple(phased[phase])})
    return out


def _resolve_large_system_mode(mode, total_atoms: int) -> bool:
    if isinstance(mode, str):
        token = mode.strip().lower()
        if token in ('', 'auto', 'default'):
            return int(total_atoms) > 99999
        if token in ('1', 'true', 'yes', 'on', 'large', 'enabled'):
            return True
        if token in ('0', 'false', 'no', 'off', 'disabled'):
            return False
    if mode is None:
        return int(total_atoms) > 99999
    return bool(mode)


def _build_large_pack_state(cell, dist_min: float, *, enabled: bool, reference_clearance: float | None = None) -> dict:
    state = {
        'enabled': bool(enabled),
        'wrapped': np.empty((0, 3), dtype=float),
        'cells': {},
        'origin': None,
        'lengths': None,
        'nbins': None,
        'grid': None,
        'neighbor_offsets': ((0, 0, 0),),
    }
    if not enabled or cell is None or not hasattr(cell, 'cell'):
        return state

    lengths = np.array(
        [
            float(cell.cell.xhi) - float(cell.cell.xlo),
            float(cell.cell.yhi) - float(cell.cell.ylo),
            float(cell.cell.zhi) - float(cell.cell.zlo),
        ],
        dtype=float,
    )
    lengths = np.maximum(lengths, 1.0e-6)
    base_clearance = float(reference_clearance) if reference_clearance is not None else float(dist_min)
    grid = max(float(dist_min), base_clearance * 1.05, 2.0)
    nbins = np.maximum(np.floor(lengths / grid).astype(int), 1)
    reach = max(int(np.ceil(float(dist_min) / grid)), 1)
    state.update(
        {
            'origin': np.array([float(cell.cell.xlo), float(cell.cell.ylo), float(cell.cell.zlo)], dtype=float),
            'lengths': lengths,
            'nbins': nbins,
            'grid': float(grid),
            'neighbor_offsets': tuple(itertools.product(range(-reach, reach + 1), repeat=3)),
        }
    )

    if cell.GetNumConformers() > 0 and cell.GetNumAtoms() > 0:
        coord = np.array(cell.GetConformer(0).GetPositions(), dtype=float)
        _append_large_pack_coords(state, coord, cell.cell)
    return state


def _grid_keys_for_coords(coord, state: dict):
    frac = np.floor((coord - state['origin']) / state['grid']).astype(int)
    return np.mod(frac, state['nbins'])


def _append_large_pack_coords(state: dict, coord, cell_box) -> None:
    if not state.get('enabled', False):
        return
    coord = np.asarray(coord, dtype=float)
    if coord.size == 0:
        return
    wrapped = calc.wrap(coord, cell_box.xhi, cell_box.xlo, cell_box.yhi, cell_box.ylo, cell_box.zhi, cell_box.zlo)
    start = int(len(state['wrapped']))
    if start == 0:
        state['wrapped'] = np.array(wrapped, dtype=float, copy=True)
    else:
        state['wrapped'] = np.vstack((state['wrapped'], wrapped))
    keys = _grid_keys_for_coords(wrapped, state)
    for local_idx, key in enumerate(keys):
        bucket = tuple(int(v) for v in key.tolist())
        state['cells'].setdefault(bucket, []).append(start + local_idx)


def _large_pack_clash(state: dict, coord, dist_min: float, cell_box) -> bool:
    if not state.get('enabled', False):
        return False
    if len(state['wrapped']) == 0:
        return False
    coord = np.asarray(coord, dtype=float)
    if coord.size == 0:
        return False

    wrapped = calc.wrap(coord, cell_box.xhi, cell_box.xlo, cell_box.yhi, cell_box.ylo, cell_box.zhi, cell_box.zlo)
    keys = _grid_keys_for_coords(wrapped, state)
    lengths = state['lengths']
    threshold_sq = float(dist_min) * float(dist_min)

    for atom_idx, key in enumerate(keys):
        base = np.asarray(key, dtype=int)
        for off in state['neighbor_offsets']:
            nb_key = tuple(int(v) for v in np.mod(base + np.asarray(off, dtype=int), state['nbins']).tolist())
            idxs = state['cells'].get(nb_key)
            if not idxs:
                continue
            diff = state['wrapped'][idxs] - wrapped[atom_idx]
            diff -= np.round(diff / lengths) * lengths
            dist_sq = np.einsum('ij,ij->i', diff, diff, optimize=True)
            if np.any(dist_sq <= threshold_sq):
                return True
    return False


def _large_pack_min_distance(state: dict, coord, cell_box) -> float:
    if not state.get('enabled', False):
        return float('inf')
    if len(state['wrapped']) == 0:
        return float('inf')
    coord = np.asarray(coord, dtype=float)
    if coord.size == 0:
        return float('inf')

    wrapped = calc.wrap(coord, cell_box.xhi, cell_box.xlo, cell_box.yhi, cell_box.ylo, cell_box.zhi, cell_box.zlo)
    keys = _grid_keys_for_coords(wrapped, state)
    lengths = state['lengths']
    best = float('inf')
    for atom_idx, key in enumerate(keys):
        base = np.asarray(key, dtype=int)
        for off in state['neighbor_offsets']:
            nb_key = tuple(int(v) for v in np.mod(base + np.asarray(off, dtype=int), state['nbins']).tolist())
            idxs = state['cells'].get(nb_key)
            if not idxs:
                continue
            delta = state['wrapped'][idxs] - wrapped[atom_idx]
            delta -= np.round(delta / lengths) * lengths
            dist = np.sqrt(np.sum(delta * delta, axis=1))
            if dist.size:
                best = min(best, float(np.min(dist)))
    return float(best)


def _cell_log_begin(func_name: str, *, restart: bool):
    dt1 = None
    if not restart:
        dt1 = datetime.datetime.now()
        utils.radon_print(f'Start {func_name} generation by poly.{func_name}.', level=1)
    return dt1


def _cell_log_done(func_name: str, dt1, *, restart: bool) -> None:
    if restart or dt1 is None:
        return
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.%s. Elapsed time = %s' % (func_name, str(dt2-dt1)), level=1)


def _rw_cache_root(work_dir):
    wd = _resolve_work_dir_path(work_dir)
    if wd is None:
        return None
    root = wd / '.yadonpy' / 'random_walk'
    root.mkdir(parents=True, exist_ok=True)
    return root


_CACHE_SAFE_ATOM_FLOAT_PROPS = (
    'AtomicCharge',
    'AtomicCharge_raw',
    'RESP',
    'RESP_raw',
    'RESP2',
    'RESP2_raw',
    'ESP',
    'MullikenCharge',
    'LowdinCharge',
    '_GasteigerCharge',
    'ff_sigma',
    'ff_epsilon',
)
_CACHE_SAFE_ATOM_STR_PROPS = (
    'ff_type',
    'ff_btype',
    'ff_ptype',
)
_CACHE_SAFE_ATOM_INT_PROPS = (
    'mol_id',
)
_CACHE_SAFE_ATOM_BOOL_PROPS = (
    'head',
    'tail',
    'head_neighbor',
    'tail_neighbor',
)
def _cache_snapshot_atom_state(atom):
    state = {'float': {}, 'str': {}, 'int': {}, 'bool': {}, 'pdb': None}
    for key in _CACHE_SAFE_ATOM_FLOAT_PROPS:
        if not atom.HasProp(key):
            continue
        try:
            state['float'][key] = float(atom.GetDoubleProp(key))
        except Exception:
            try:
                state['float'][key] = float(atom.GetProp(key))
            except Exception:
                pass
    for key in _CACHE_SAFE_ATOM_STR_PROPS:
        if not atom.HasProp(key):
            continue
        try:
            state['str'][key] = str(atom.GetProp(key))
        except Exception:
            pass
    for key in _CACHE_SAFE_ATOM_INT_PROPS:
        if not atom.HasProp(key):
            continue
        try:
            state['int'][key] = int(atom.GetIntProp(key))
        except Exception:
            try:
                state['int'][key] = int(atom.GetProp(key))
            except Exception:
                pass
    for key in _CACHE_SAFE_ATOM_BOOL_PROPS:
        if not atom.HasProp(key):
            continue
        try:
            state['bool'][key] = bool(atom.GetBoolProp(key))
        except Exception:
            try:
                val = str(atom.GetProp(key)).strip().lower()
                state['bool'][key] = val in ('1', 'true', 't', 'yes', 'y', 'on')
            except Exception:
                pass
    ri = atom.GetPDBResidueInfo()
    if ri is not None:
        try:
            state['pdb'] = {
                'name': ri.GetName(),
                'resName': ri.GetResidueName(),
                'resNum': int(ri.GetResidueNumber()),
                'chain': ri.GetChainId(),
                'insCode': ri.GetInsertionCode(),
                'isHet': bool(ri.GetIsHeteroAtom()),
                'occ': float(ri.GetOccupancy()),
                'temp': float(ri.GetTempFactor()),
                'serial': int(ri.GetSerialNumber()),
                'altLoc': ri.GetAltLoc(),
            }
        except Exception:
            state['pdb'] = None
    return state


def _cache_restore_atom_state(atom, state):
    if atom is None or not isinstance(state, dict):
        return
    for key, value in (state.get('str') or {}).items():
        try:
            atom.SetProp(str(key), str(value))
        except Exception:
            pass
    for key, value in (state.get('int') or {}).items():
        try:
            atom.SetIntProp(str(key), int(value))
        except Exception:
            pass
    for key, value in (state.get('bool') or {}).items():
        try:
            atom.SetBoolProp(str(key), bool(value))
        except Exception:
            pass
    for key, value in (state.get('float') or {}).items():
        try:
            atom.SetDoubleProp(str(key), float(value))
        except Exception:
            try:
                atom.SetProp(str(key), str(value))
            except Exception:
                pass
    pdb = state.get('pdb')
    if isinstance(pdb, dict):
        try:
            atom_name = str(pdb.get('name') or _pdb_atom_name(atom))
            ri = Chem.AtomPDBResidueInfo(
                atom_name,
                residueName=str(pdb.get('resName') or 'RU0'),
                residueNumber=int(pdb.get('resNum', 1)),
                isHeteroAtom=bool(pdb.get('isHet', False)),
            )
            try:
                ri.SetChainId(str(pdb.get('chain', ' ')))
            except Exception:
                pass
            try:
                ri.SetInsertionCode(str(pdb.get('insCode', ' ')))
            except Exception:
                pass
            try:
                ri.SetOccupancy(float(pdb.get('occ', 1.0)))
            except Exception:
                pass
            try:
                ri.SetTempFactor(float(pdb.get('temp', 0.0)))
            except Exception:
                pass
            try:
                ri.SetSerialNumber(int(pdb.get('serial', atom.GetIdx() + 1)))
            except Exception:
                pass
            try:
                ri.SetAltLoc(str(pdb.get('altLoc', ' ')))
            except Exception:
                pass
            atom.SetMonomerInfo(ri)
        except Exception:
            pass


def _cache_snapshot_mol_state(mol):
    if mol is None:
        return None
    out = {'atom_props': [], 'mol_props': {'str': {}, 'int': {}, 'float': {}, 'bool': {}}}
    try:
        for atom in mol.GetAtoms():
            out['atom_props'].append(_cache_snapshot_atom_state(atom))
    except Exception:
        out['atom_props'] = []
    try:
        for key in list(mol.GetPropNames(includePrivate=True, includeComputed=False)):
            try:
                out['mol_props']['bool'][key] = bool(mol.GetBoolProp(key))
                continue
            except Exception:
                pass
            try:
                out['mol_props']['int'][key] = int(mol.GetIntProp(key))
                continue
            except Exception:
                pass
            try:
                out['mol_props']['float'][key] = float(mol.GetDoubleProp(key))
                continue
            except Exception:
                pass
            try:
                out['mol_props']['str'][key] = str(mol.GetProp(key))
            except Exception:
                pass
    except Exception:
        pass
    return out


def _cache_restore_mol_state(mol, state):
    if mol is None or not isinstance(state, dict):
        return mol
    mol_props = state.get('mol_props')
    if isinstance(mol_props, dict):
        for key, value in (mol_props.get('str') or {}).items():
            try:
                mol.SetProp(str(key), str(value))
            except Exception:
                pass
        for key, value in (mol_props.get('int') or {}).items():
            try:
                mol.SetIntProp(str(key), int(value))
            except Exception:
                pass
        for key, value in (mol_props.get('bool') or {}).items():
            try:
                mol.SetBoolProp(str(key), bool(value))
            except Exception:
                pass
        for key, value in (mol_props.get('float') or {}).items():
            try:
                mol.SetDoubleProp(str(key), float(value))
            except Exception:
                try:
                    mol.SetProp(str(key), str(value))
                except Exception:
                    pass
    atom_props = state.get('atom_props')
    if isinstance(atom_props, list):
        for idx, atom_state in enumerate(atom_props):
            if idx >= mol.GetNumAtoms():
                break
            try:
                atom = mol.GetAtomWithIdx(int(idx))
            except Exception:
                continue
            _cache_restore_atom_state(atom, atom_state)
    return mol


def _cache_writer_copy(mol):
    if mol is None:
        return None
    try:
        mol_copy = Chem.Mol(mol)
    except Exception:
        return mol
    try:
        for key in list(mol_copy.GetPropNames(includePrivate=True, includeComputed=True)):
            try:
                mol_copy.ClearProp(key)
            except Exception:
                pass
    except Exception:
        pass
    try:
        for atom in mol_copy.GetAtoms():
            for key in list(atom.GetPropNames(includePrivate=True, includeComputed=True)):
                try:
                    atom.ClearProp(key)
                except Exception:
                    pass
    except Exception:
        pass
    return mol_copy


def _merge_cache_state_dict(existing, update):
    if not isinstance(update, dict):
        return update
    merged = dict(existing) if isinstance(existing, dict) else {}
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            inner = dict(merged[key])
            inner.update(value)
            merged[key] = inner
        else:
            merged[key] = value
    return merged


def _rw_cache_key(kind: str, payload: dict) -> str:
    import hashlib, json
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return f"{kind}-{hashlib.sha256(blob).hexdigest()[:16]}"


def _rw_normalize_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_rw_normalize_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_rw_normalize_value(v) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if v is None:
                continue
            out[str(k)] = _rw_normalize_value(v)
        return out
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(f'{float(value):.12g}')
    return value


def _rw_payload(kind: str, **kwargs):
    return _rw_normalize_value({'kind': kind, **kwargs})


def _rw_compat_payload(kind: str, payload: dict):
    compat = _rw_normalize_value(payload)
    if not isinstance(compat, dict):
        return compat
    compat.pop('name', None)
    if kind in ('random_copolymerize_rw', 'random_copolymerize_rw_plan'):
        smiles = compat.get('smiles') or []
        if compat.get('ratio') is None and isinstance(smiles, list) and len(smiles) > 0:
            compat['ratio'] = _rw_normalize_value(ratio_to_prob([1] * len(smiles)))
    return compat


def _rw_state_path(work_dir, kind: str, payload: dict):
    root = _rw_cache_root(work_dir)
    if root is None:
        return None
    key = _rw_cache_key(kind, payload)
    return root / f'{key}.state.json'


def _rw_load_state(work_dir, kind: str, payload: dict):
    state = _rw_state_path(work_dir, kind, payload)
    if state is None or not state.exists():
        return None
    try:
        import json

        return json.loads(state.read_text(encoding='utf-8'))
    except Exception:
        return None


def _rw_save_state(work_dir, kind: str, payload: dict, data: dict):
    state = _rw_state_path(work_dir, kind, payload)
    if state is None:
        return
    try:
        import json
        normalized = _rw_normalize_value(data)
        if state.exists():
            try:
                existing = json.loads(state.read_text(encoding='utf-8'))
            except Exception:
                existing = None
            normalized = _merge_cache_state_dict(existing, normalized)
        state.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    except Exception:
        pass


def _cache_mol_smiles(mol):
    if mol is None:
        return None
    # `_yadonpy_molid` is a runtime/cache artifact identifier, not a stable
    # chemical identity. Restarted workflows can restamp logically identical
    # molecules with a different molid, so cache payloads must prefer source
    # chemistry/name metadata before falling back to molid.
    prop_keys = (
        '_yadonpy_source_smiles',
        'smiles',
        'input_smiles',
        'mol_name',
        'name',
        '_Name',
        '_yadonpy_molid',
    )
    for key in prop_keys:
        try:
            if mol.HasProp(key):
                value = str(mol.GetProp(key)).strip()
                if value:
                    return f'{key}:{value}'
        except Exception:
            continue
    try:
        return f"smiles:{Chem.MolToSmiles(mol, isomericSmiles=True)}"
    except Exception:
        return None


def _cell_cache_payload(kind: str, *, mols, n, cell, density, threshold, dec_rate, check_bond_ring_intersection, mp,
        neutralize, neutralize_tol, charge_scale, charge_tolerance, ions, large_system_mode):
    ion_payload = []
    if ions is not None:
        seq = ions if isinstance(ions, (list, tuple)) else [ions]
        for pack in seq:
            ion_payload.append({
                'smiles': _cache_mol_smiles(getattr(pack, 'mol', None)),
                'n': getattr(pack, 'n', None),
                'ff_name': getattr(pack, 'ff_name', None),
            })
    return _rw_payload(
        kind,
        schema_version=AMORPHOUS_CELL_SCHEMA_VERSION,
        smiles=[_cache_mol_smiles(mol) for mol in mols],
        n=[int(v) for v in n],
        cell_smiles=_cache_mol_smiles(cell) if isinstance(cell, Chem.Mol) else None,
        density=density,
        threshold=threshold,
        dec_rate=dec_rate,
        check_bond_ring_intersection=bool(check_bond_ring_intersection),
        mp=int(mp),
        neutralize=bool(neutralize),
        neutralize_tol=float(neutralize_tol),
        charge_scale=charge_scale,
        charge_tolerance=float(charge_tolerance),
        ions=ion_payload,
        large_system_mode=large_system_mode,
    )


def _cell_state_from_mol(mol):
    if mol is None or not hasattr(mol, 'cell'):
        return None
    try:
        return {
            'cell': {
                'xhi': float(mol.cell.xhi),
                'xlo': float(mol.cell.xlo),
                'yhi': float(mol.cell.yhi),
                'ylo': float(mol.cell.ylo),
                'zhi': float(mol.cell.zhi),
                'zlo': float(mol.cell.zlo),
            }
        }
    except Exception:
        return None


def _restore_cached_cell_state(mol, state):
    if mol is None:
        return mol
    if isinstance(state, dict):
        mol = _cache_restore_mol_state(mol, state)
    if not isinstance(state, dict):
        return mol
    cell_state = state.get('cell')
    if not isinstance(cell_state, dict):
        return mol
    try:
        setattr(
            mol,
            'cell',
            utils.Cell(
                float(cell_state['xhi']),
                float(cell_state['xlo']),
                float(cell_state['yhi']),
                float(cell_state['ylo']),
                float(cell_state['zhi']),
                float(cell_state['zlo']),
            ),
        )
    except Exception:
        pass
    return mol


def _copy_cell_holder(mol):
    holder = Chem.Mol()
    if mol is None or not hasattr(mol, 'cell'):
        return holder
    try:
        src = mol.cell
        setattr(holder, 'cell', utils.Cell(float(src.xhi), float(src.xlo), float(src.yhi), float(src.ylo), float(src.zhi), float(src.zlo)))
    except Exception:
        pass
    return holder


def _estimate_packing_stress_axes(mols, n, cell_box) -> tuple[int, ...]:
    if cell_box is None:
        return (2, 1, 0)
    lengths = np.asarray(
        [
            float(cell_box.xhi) - float(cell_box.xlo),
            float(cell_box.yhi) - float(cell_box.ylo),
            float(cell_box.zhi) - float(cell_box.zlo),
        ],
        dtype=float,
    )
    lengths = np.maximum(lengths, 1.0e-6)
    stress = np.zeros(3, dtype=float)
    for mol, count in zip(mols, n):
        span = _estimate_mol_bbox_span(mol)
        stress += np.asarray(span, dtype=float) * max(int(count), 0)
    norm = np.divide(stress, lengths, out=np.zeros_like(stress), where=lengths > 0.0)
    order = np.argsort(norm)[::-1]
    return tuple(int(i) for i in order.tolist())


def _next_amorphous_retry_target(cell, density, dec_rate, *, axes: tuple[int, ...] = (2,)):
    if density is not None:
        retry_density = float(density) * float(dec_rate)
        return {
            'cell': cell,
            'density': retry_density,
            'log': '[PACK] Retry poly.amorphous_cell. Remaining %i times. The density is reduced to %f.',
            'log_value': retry_density,
        }

    if cell is None or not hasattr(cell, 'cell'):
        raise ValueError('poly.amorphous_cell retry requires either density or an explicit cell with box lengths.')

    grow = 1.0 / max(float(dec_rate), 1.0e-8)
    src = cell.cell
    lengths = np.array(
        [
            float(src.xhi) - float(src.xlo),
            float(src.yhi) - float(src.ylo),
            float(src.zhi) - float(src.zlo),
        ],
        dtype=float,
    )
    axes = tuple(int(a) for a in axes if int(a) in (0, 1, 2))
    if not axes:
        axes = (2,)
    for axis in axes:
        lengths[axis] = float(lengths[axis]) * grow
    retry_cell = _copy_cell_holder(cell)
    setattr(retry_cell, 'cell', utils.Cell(float(lengths[0]), 0.0, float(lengths[1]), 0.0, float(lengths[2]), 0.0))
    axes_label = ''.join('XYZ'[axis] for axis in axes)
    fixed_axes = ''.join('XYZ'[axis] for axis in (0, 1, 2) if axis not in axes)
    if fixed_axes:
        log = (
            f'[PACK] Retry poly.amorphous_cell. Remaining %i times. Density is fixed by explicit cell; '
            f'expanded {axes_label} lengths to %s while keeping {fixed_axes} fixed.'
        )
    else:
        log = (
            f'[PACK] Retry poly.amorphous_cell. Remaining %i times. Density is fixed by explicit cell; '
            f'expanded {axes_label} lengths to %s.'
        )
    return {
        'cell': retry_cell,
        'density': None,
        'log': log,
        'log_value': tuple(float(v) for v in lengths.tolist()),
    }


def _pdb_atom_name(atom):
    return atom.GetProp('ff_type') if atom.HasProp('ff_type') else atom.GetSymbol()


def _write_pack_diagnostics(work_dir, payload: dict) -> None:
    wd = _resolve_work_dir_path(work_dir)
    if wd is None:
        return
    try:
        import json

        out = wd / '.yadonpy' / 'amorphous_cell_pack_diagnostics.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    except Exception:
        pass


def _infer_residue_defaults(mol, residue_name='RU0', residue_number=1):
    if mol is None:
        return residue_name, residue_number
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        try:
            info_name = str(info.GetResidueName()).strip()
        except Exception:
            info_name = ''
        try:
            info_number = int(info.GetResidueNumber())
        except Exception:
            info_number = 0
        return info_name or residue_name, info_number if info_number > 0 else residue_number
    return residue_name, residue_number


def _ensure_atom_residue_info(atom, residue_name='RU0', residue_number=1):
    atom_name = _pdb_atom_name(atom)
    info = atom.GetPDBResidueInfo()
    if info is None:
        atom.SetMonomerInfo(
            Chem.AtomPDBResidueInfo(
                atom_name,
                residueName=residue_name,
                residueNumber=int(residue_number),
                isHeteroAtom=False,
            )
        )
        return atom.GetPDBResidueInfo()

    info.SetName(atom_name)
    try:
        current_name = str(info.GetResidueName()).strip()
    except Exception:
        current_name = ''
    if not current_name:
        info.SetResidueName(residue_name)
    try:
        current_number = int(info.GetResidueNumber())
    except Exception:
        current_number = 0
    if current_number <= 0:
        info.SetResidueNumber(int(residue_number))
    return info


def _ensure_mol_residue_info(mol, residue_name='RU0', residue_number=1):
    if mol is None:
        return None
    default_name, default_number = _infer_residue_defaults(mol, residue_name=residue_name, residue_number=residue_number)
    for atom in mol.GetAtoms():
        _ensure_atom_residue_info(atom, residue_name=default_name, residue_number=default_number)
    return mol


def _set_atom_residue(mol, atom_idx: int, residue_name: str, residue_number: int):
    atom = mol.GetAtomWithIdx(int(atom_idx))
    info = _ensure_atom_residue_info(atom, residue_name=residue_name, residue_number=residue_number)
    info.SetResidueName(residue_name)
    info.SetResidueNumber(int(residue_number))
    return atom


def _rw_load_sdf(path: Path, kind: str):
    try:
        mol = None
        try:
            sup = Chem.SDMolSupplier(str(path), removeHs=False)
            mol = sup[0] if sup and len(sup) > 0 else None
        except Exception:
            mol = None
        if mol is None:
            sup = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
            mol = sup[0] if sup and len(sup) > 0 else None
            if mol is not None:
                try:
                    Chem.SanitizeMol(
                        mol,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                    )
                except Exception:
                    pass
        if mol is not None:
            _ensure_mol_residue_info(mol)
            utils.radon_print(f'[RESTART] Reusing cached {kind}: {path.name}', level=1)
        return mol
    except Exception:
        return None


def _rw_find_compatible_sdf(root: Path, kind: str, payload: dict):
    want = _rw_compat_payload(kind, payload)
    candidates = []
    for meta in root.glob('*.json'):
        if meta.name.endswith('.state.json'):
            continue
        try:
            import json

            have = json.loads(meta.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(have, dict) or str(have.get('kind')) != str(kind):
            continue
        if _rw_compat_payload(kind, have) != want:
            continue
        sdf = meta.with_suffix('.sdf')
        if sdf.exists():
            candidates.append(sdf)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _rw_load(work_dir, kind: str, payload: dict):
    root = _rw_cache_root(work_dir)
    if root is None:
        return None
    key = _rw_cache_key(kind, payload)
    sdf = root / f'{key}.sdf'
    if sdf.exists():
        mol = _rw_load_sdf(sdf, kind)
        mol = _cache_restore_mol_state(mol, _rw_load_state(work_dir, kind, payload))
        try:
            mol = _rw_finalize_bonded_terms(mol)
        except Exception:
            pass
        return mol
    compat = _rw_find_compatible_sdf(root, kind, payload)
    if compat is None:
        return None
    mol = _rw_load_sdf(compat, kind)
    state_path = compat.with_suffix('.state.json')
    state = None
    if state_path.exists():
        try:
            import json
            state = json.loads(state_path.read_text(encoding='utf-8'))
        except Exception:
            state = None
    mol = _cache_restore_mol_state(mol, state)
    try:
        mol = _rw_finalize_bonded_terms(mol)
    except Exception:
        pass
    return mol


def _rw_save(work_dir, kind: str, payload: dict, mol):
    root = _rw_cache_root(work_dir)
    if root is None or mol is None:
        return
    key = _rw_cache_key(kind, payload)
    sdf = root / f'{key}.sdf'
    meta = root / f'{key}.json'
    try:
        w = Chem.SDWriter(str(sdf))
        w.write(_cache_writer_copy(mol))
        w.close()
        import json
        meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        _rw_save_state(work_dir, kind, payload, _cache_snapshot_mol_state(mol))
    except Exception:
        pass


def _cell_cache_root(work_dir):
    wd = _resolve_work_dir_path(work_dir)
    if wd is None:
        return None
    root = wd / '.yadonpy' / 'cell_cache'
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cell_cache_paths(work_dir, kind: str):
    root = _cell_cache_root(work_dir)
    if root is None:
        return None
    stem = str(kind).strip() or 'cell'
    return {
        'sdf': root / f'{stem}.sdf',
        'meta': root / f'{stem}.json',
        'state': root / f'{stem}.state.json',
    }


def _cell_cache_load(work_dir, kind: str, payload: dict):
    paths = _cell_cache_paths(work_dir, kind)
    if paths is None:
        return None, None
    sdf = paths['sdf']
    meta = paths['meta']
    state = paths['state']
    if not sdf.exists() or not meta.exists():
        return None, None
    try:
        import json

        have = json.loads(meta.read_text(encoding='utf-8'))
    except Exception:
        return None, None
    if not isinstance(have, dict) or str(have.get('kind')) != str(kind):
        return None, None
    if _rw_compat_payload(kind, have) != _rw_compat_payload(kind, payload):
        return None, None
    mol = _rw_load_sdf(sdf, kind)
    if mol is None:
        return None, None
    cell_state = None
    if state.exists():
        try:
            import json

            cell_state = json.loads(state.read_text(encoding='utf-8'))
        except Exception:
            cell_state = None
    return _cache_restore_mol_state(mol, cell_state), cell_state


def _cell_cache_save(work_dir, kind: str, payload: dict, mol, state: dict | None = None):
    paths = _cell_cache_paths(work_dir, kind)
    if paths is None or mol is None:
        return
    try:
        w = Chem.SDWriter(str(paths['sdf']))
        w.write(_cache_writer_copy(mol))
        w.close()
        import json

        meta = {'kind': str(kind), **dict(payload)}
        paths['meta'].write_text(json.dumps(_rw_normalize_value(meta), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        state_payload = _merge_cache_state_dict(_cache_snapshot_mol_state(mol), _rw_normalize_value(state))
        if state_payload is not None:
            paths['state'].write_text(json.dumps(state_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    except Exception:
        pass


def _cell_log_restart_reuse(func_name: str, *, source: str, work_dir=None) -> None:
    wd = _resolve_work_dir_path(work_dir)
    detail = f' | work_dir={wd}' if wd is not None else ''
    utils.radon_print(f'[SKIP] poly.{func_name}: restored cached result from {source}{detail}', level=1)


def _cell_log_restart_miss(func_name: str, *, work_dir=None) -> None:
    wd = _resolve_work_dir_path(work_dir)
    detail = f' at {wd}' if wd is not None else ''
    utils.radon_print(f'[RESTART] No compatible cached {func_name} result found{detail}; rebuilding packed cell.', level=1)


def _effective_restart_flag(work_dir, restart, *, restart_flag=None):
    if restart is not None and restart_flag is not None and bool(restart) != bool(restart_flag):
        raise ValueError('Conflicting restart controls: restart and restart_flag differ')
    if restart is not None:
        return bool(restart)
    if restart_flag is not None:
        return bool(restart_flag)
    try:
        return bool(getattr(work_dir, 'restart'))
    except Exception:
        return bool(resolve_restart(None))


def _count_sssr_rings(mol) -> int:
    try:
        sssr = Chem.GetSSSR(mol)
    except Exception:
        return 0
    if isinstance(sssr, int):
        return int(sssr)
    try:
        return int(len(sssr))
    except Exception:
        return 0


def _estimate_rw_rigidity(mol) -> float:
    mol = _resolve_mol_like(mol)
    if not isinstance(mol, Chem.Mol) or mol.GetNumAtoms() <= 0:
        return 0.5

    total_atoms = max(int(mol.GetNumAtoms()), 1)
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    heavy_atoms = max(int(mol.GetNumHeavyAtoms()), 1)
    ring_count = _count_sssr_rings(mol)
    aromatic_fraction = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()) / total_atoms
    branch_fraction = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2) / total_atoms
    try:
        rotatable = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    except Exception:
        rotatable = 0.0

    ring_density = min(ring_count / max(heavy_atoms / 6.0, 1.0), 1.0)
    flexibility = min(rotatable / max(heavy_atoms - 1, 1), 1.0)
    rigidity = (
        0.45 * ring_density
        + 0.25 * float(aromatic_fraction)
        + 0.15 * float(branch_fraction)
        + 0.15 * (1.0 - flexibility)
    )
    return float(np.clip(rigidity, 0.0, 1.0))


def _resolve_rw_retry_budget(mols, retry, rollback, rollback_shaking, retry_step, retry_opt_step, *, total_steps: int | None = None):
    mols = [mol for mol in _resolve_mol_list(mols) if isinstance(mol, Chem.Mol)]
    rigidity = max((_estimate_rw_rigidity(mol) for mol in mols), default=0.5)

    if rigidity >= 0.72:
        recommended = {'retry': 80, 'rollback': 4, 'retry_step': 120, 'retry_opt_step': 8}
    elif rigidity >= 0.38:
        recommended = {'retry': 60, 'rollback': 3, 'retry_step': 90, 'retry_opt_step': 4}
    else:
        # Keep the historical defaults for flexible chains. Lowering the retry
        # budget here made easy polymerizations less robust in practice.
        recommended = {'retry': 60, 'rollback': 3, 'retry_step': 80, 'retry_opt_step': 4}

    try:
        steps = max(int(total_steps or 0), 0)
    except Exception:
        steps = 0
    if steps >= 120:
        recommended = {
            'retry': max(recommended['retry'], 160),
            'rollback': max(recommended['rollback'], 4),
            'retry_step': max(recommended['retry_step'], 240),
            'retry_opt_step': max(recommended['retry_opt_step'], 8),
        }
    elif steps >= 80:
        recommended = {
            'retry': max(recommended['retry'], 120),
            'rollback': max(recommended['rollback'], 4),
            'retry_step': max(recommended['retry_step'], 160),
            'retry_opt_step': max(recommended['retry_opt_step'], 6),
        }

    original = {
        'retry': int(retry),
        'rollback': int(rollback),
        'retry_step': int(retry_step),
        'retry_opt_step': int(retry_opt_step),
    }
    effective = {
        'retry': max(0, max(original['retry'], recommended['retry'])),
        'rollback': max(1, max(original['rollback'], recommended['rollback'])),
        'retry_step': max(1, max(original['retry_step'], recommended['retry_step'])),
        'retry_opt_step': max(0, max(original['retry_opt_step'], recommended['retry_opt_step'])),
    }
    changed = {key: (original[key], effective[key]) for key in effective if original[key] != effective[key]}

    return {
        'rigidity': rigidity,
        'retry': effective['retry'],
        'rollback': effective['rollback'],
        'rollback_shaking': bool(rollback_shaking),
        'retry_step': effective['retry_step'],
        'retry_opt_step': effective['retry_opt_step'],
        'changed': changed,
    }


def _nudge_first_atom_charge(mol, delta: float) -> None:
    if mol is None or mol.GetNumAtoms() <= 0:
        return
    atom = mol.GetAtomWithIdx(0)
    keys = ['AtomicCharge', 'AtomicCharge_raw', 'RESP', 'RESP_raw', 'RESP2', 'RESP2_raw']
    base = 0.0
    for k in keys:
        try:
            if atom.HasProp(k):
                base = float(atom.GetDoubleProp(k))
                break
        except Exception:
            pass
    for k in keys:
        try:
            if atom.HasProp(k) or k in ('AtomicCharge', 'RESP', 'RESP2'):
                atom.SetDoubleProp(k, float(base + delta))
        except Exception:
            pass


def _format_rw_elapsed(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    if seconds < 60.0:
        return f'{seconds:.1f}s'
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f'{int(minutes)}m {sec:.1f}s'
    hours, minutes = divmod(minutes, 60.0)
    return f'{int(hours)}h {int(minutes)}m {sec:.0f}s'


def _rw_progress_step_interval(total_steps: int) -> int:
    total_steps = max(int(total_steps), 1)
    if total_steps <= 10:
        return 1
    if total_steps <= 100:
        return 10
    return 25


def _rw_emit_heartbeat(*, step_idx: int, total_steps: int, retry_idx: int, retry_total: int,
        step_started_at: float, walk_started_at: float, state: dict, now: float | None = None) -> None:
    interval = float(getattr(const, 'rw_heartbeat_seconds', 0.0) or 0.0)
    if interval <= 0.0:
        return
    now = time.perf_counter() if now is None else float(now)
    last_heartbeat = float(state.get('last_heartbeat', walk_started_at))
    if (now - last_heartbeat) < interval:
        return
    utils.radon_print(
        '[RW] progress step %i/%i: searching for a valid placement; current_retry=%i/%i; step_elapsed=%s; total_elapsed=%s'
        % (
            int(step_idx),
            int(total_steps),
            int(retry_idx),
            int(retry_total),
            _format_rw_elapsed(now - float(step_started_at)),
            _format_rw_elapsed(now - float(walk_started_at)),
        ),
        level=1,
    )
    state['last_heartbeat'] = now


def _rw_emit_step_progress(*, step_idx: int, total_steps: int, retries_used: int,
        step_started_at: float, walk_started_at: float, state: dict, now: float | None = None) -> None:
    if not const.tqdm_disable:
        return
    now = time.perf_counter() if now is None else float(now)
    every = int(state.get('step_interval', _rw_progress_step_interval(total_steps)))
    if step_idx not in {1, int(total_steps)} and (int(step_idx) % max(every, 1)) != 0:
        return
    utils.radon_print(
        '[RW] accepted step %i/%i; retries_used=%i; step_elapsed=%s; total_elapsed=%s'
        % (
            int(step_idx),
            int(total_steps),
            int(retries_used),
            _format_rw_elapsed(now - float(step_started_at)),
            _format_rw_elapsed(now - float(walk_started_at)),
        ),
        level=1,
    )


def _rdkit_clone_mol(mol):
    try:
        return Chem.Mol(mol)
    except Exception:
        return utils.deepcopy_mol(mol)


def _rw_growth_clone_mol(mol):
    if mol is None:
        return None
    clone = _rdkit_clone_mol(mol)
    if hasattr(mol, 'cell'):
        try:
            setattr(clone, 'cell', copy(mol.cell))
        except Exception:
            pass
    return clone


def _has_marked_new_bond(mol):
    if mol is None or not hasattr(mol, 'GetBonds'):
        return False
    for bond in mol.GetBonds():
        try:
            if bond.HasProp('new_bond') and bond.GetBoolProp('new_bond'):
                return True
        except Exception:
            continue
    return False


def _clear_new_bond_marks(mol):
    if mol is None or not hasattr(mol, 'GetBonds'):
        return
    for bond in mol.GetBonds():
        try:
            if bond.HasProp('new_bond'):
                bond.ClearProp('new_bond')
        except Exception:
            continue


def _rw_finalize_bonded_terms(mol):
    if mol is None:
        return None
    has_new_bond = _has_marked_new_bond(mol)
    has_angles = bool(getattr(mol, 'angles', {}) or {})
    has_dihedrals = bool(getattr(mol, 'dihedrals', {}) or {})
    skip_bonded_refresh = bool(not has_new_bond and has_angles and (mol.GetNumAtoms() < 4 or has_dihedrals))
    ff_name = None
    try:
        if hasattr(mol, 'HasProp') and mol.HasProp('ff_name'):
            ff_name = str(mol.GetProp('ff_name')).strip() or None
    except Exception:
        ff_name = None
    if (not skip_bonded_refresh) and ff_name:
        try:
            from ..api import get_ff

            ff_obj = get_ff(ff_name)
            if has_new_bond:
                handled_locally = False
                if str(ff_name).strip().lower() == 'oplsaa' and hasattr(ff_obj, 'refresh_polymer_junction_terms'):
                    handled_locally = bool(ff_obj.refresh_polymer_junction_terms(mol))
                if not handled_locally:
                    if mol.GetNumAtoms() >= 1 and hasattr(ff_obj, 'assign_ptypes'):
                        ff_obj.assign_ptypes(mol)
                    if mol.GetNumAtoms() >= 2 and hasattr(ff_obj, 'assign_btypes'):
                        ff_obj.assign_btypes(mol)
                    if mol.GetNumAtoms() >= 3 and hasattr(ff_obj, 'assign_atypes'):
                        ff_obj.assign_atypes(mol)
                    if mol.GetNumAtoms() >= 4 and hasattr(ff_obj, 'assign_dtypes'):
                        ff_obj.assign_dtypes(mol)
                    if hasattr(ff_obj, 'assign_itypes'):
                        ff_obj.assign_itypes(mol)
                _clear_new_bond_marks(mol)
            else:
                if mol.GetNumAtoms() >= 3 and not has_angles and hasattr(ff_obj, 'assign_atypes'):
                    ff_obj.assign_atypes(mol)
                if mol.GetNumAtoms() >= 4 and not has_dihedrals and hasattr(ff_obj, 'assign_dtypes'):
                    ff_obj.assign_dtypes(mol)
                if hasattr(ff_obj, 'assign_itypes'):
                    ff_obj.assign_itypes(mol)
        except Exception:
            pass
    try:
        has_polyelectrolyte_props = False
        if hasattr(mol, 'HasProp'):
            for key in (
                "_yadonpy_charge_groups_json",
                "_yadonpy_resp_constraints_json",
                "_yadonpy_polyelectrolyte_summary_json",
            ):
                if mol.HasProp(key):
                    has_polyelectrolyte_props = True
                    break
        molecule_formal_charge = int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))
        if has_polyelectrolyte_props or ("*" in Chem.MolToSmiles(mol, isomericSmiles=True)) or molecule_formal_charge != 0:
            annotate_polyelectrolyte_metadata(mol, detection="auto")
    except Exception:
        pass
    return mol

def ratio_to_prob(ratio):
    """Convert an integer (or float) feed ratio list into probabilities.

    Examples
    --------
    >>> ratio_to_prob([1, 1, 2])
    [0.25, 0.25, 0.5]

    Notes
    -----
    - All entries must be non-negative.
    - If the sum is zero, this raises ValueError.
    """
    if ratio is None:
        raise ValueError("ratio must not be None")
    # Accept list/tuple/np array
    arr = np.asarray(ratio, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("ratio must not be empty")
    if np.any(arr < 0):
        raise ValueError(f"ratio must be non-negative, got {ratio}")
    s = float(arr.sum())
    if s <= 0.0:
        raise ValueError(f"sum(ratio) must be > 0, got {ratio}")
    prob = (arr / s).tolist()
    # Clean tiny negatives due to float noise
    prob = [0.0 if p < 0 and abs(p) < 1e-15 else float(p) for p in prob]
    # Renormalize once more to guarantee sum=1 within float
    sp = sum(prob)
    if sp != 0:
        prob = [p / sp for p in prob]
    return prob



MD_avail = True
try:
    from ..sim import md
except ImportError:
    MD_avail = False

if const.tqdm_disable:
    tqdm = utils.tqdm_stub
else:
    try:
        from tqdm.autonotebook import tqdm
    except ImportError:
        tqdm = utils.tqdm_stub




# -----------------------------------------------------------------------------
# Ion injection helper (for polyelectrolytes)
# -----------------------------------------------------------------------------

class IonPack:
    """Container for ions to be added into an amorphous cell."""
    def __init__(self, mol, n, ff_name='merz'):
        self.mol = mol
        self.n = int(n) if n is not None else None
        self.ff_name = str(ff_name)


def ion(ion='Na+', n_ion=None, ff=None):
    """Create an explicit ion pack for ``amorphous_cell(..., ions=[...])``.

    This helper is designed to match a simple user script style:

        ion_ff = MERZ()
        ion_pack = ion(ion='Na+', n_ion=1000, ff=ion_ff)

    Notes:
        - The returned object must be passed explicitly via ``ions=``.
        - ff must provide create_ion_mol(ion) method (MERZ does).
    """
    if ff is None or not hasattr(ff, 'create_ion_mol'):
        raise ValueError('ion(): ff must be an ion FF instance with create_ion_mol(). e.g., MERZ()')

    ion_key = str(ion).strip()
    # Accept SMILES such as "[Li+]".
    try:
        m = Chem.MolFromSmiles(ion_key)
    except Exception:
        m = None
    if m is not None and int(m.GetNumAtoms()) == 1 and int(m.GetAtomWithIdx(0).GetFormalCharge()) != 0:
        a = m.GetAtomWithIdx(0)
        sym = a.GetSymbol()
        q = int(a.GetFormalCharge())
        # Build the canonical ion key (Li+, Ca2+, Cl-)
        if abs(q) == 1:
            ion_key = f"{sym}{'+' if q > 0 else '-'}"
        else:
            ion_key = f"{sym}{abs(q)}{'+' if q > 0 else '-'}"
    elif m is not None and int(m.GetNumAtoms()) != 1:
        raise ValueError('ion(): MERZ ion builder supports monoatomic ions only. Multi-atom ions should be parameterized by GAFF2_mod and passed as normal species.')

    ion_mol = ff.create_ion_mol(ion_key)
    return IonPack(ion_mol, n_ion, ff_name=getattr(ff, 'name', 'ion_ff'))


# Make ion() available without explicit import (so user scripts can call ion(...) directly)
try:
    import builtins as _builtins
    if not hasattr(_builtins, 'ion'):
        _builtins.ion = ion
except Exception:
    pass


def _mol_net_charge(mol, use_atomic_charge=True):
    """Compute net charge of an RDKit Mol.

    Priority:
      1) If any recognized per-atom charge property exists, sum it.
      2) Otherwise, use formal charge from SMILES.
    """
    q = 0.0
    if use_atomic_charge:
        try:
            from . import chem_utils as core_chem_utils

            _, charges = core_chem_utils.select_best_charge_property(mol)
            if charges:
                return float(sum(float(v) for v in charges))
        except Exception:
            pass
    # fallback
    for a in mol.GetAtoms():
        q += float(a.GetFormalCharge())
    return q

def _init_connect_linkers(mol1, mol2, *, set_linker=True, label1=1, label2=1, headhead=False, tailtail=False):
    if headhead:
        set_linker_flag(mol1, reverse=True, label=label1)
    elif set_linker:
        set_linker_flag(mol1, label=label1)
    if tailtail and not headhead:
        set_linker_flag(mol2, reverse=True, label=label2)
    elif set_linker:
        set_linker_flag(mol2, label=label2)

    if mol1.GetIntProp('tail_idx') < 0 or mol1.GetIntProp('tail_ne_idx') < 0:
        utils.radon_print('Cannot connect_mols because mol1 does not have a tail linker atom.', level=2)
        return False
    if mol2.GetIntProp('head_idx') < 0 or mol2.GetIntProp('head_ne_idx') < 0:
        utils.radon_print('Cannot connect_mols because mol2 does not have a head linker atom.', level=2)
        return False
    return True


def _prepare_connect_trial(mol1, mol2, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer', set_linker=True,
                label1=1, label2=1, headhead=False, tailtail=False, confId1=0, confId2=0):
    if mol1 is None or mol2 is None:
        return None
    if not _init_connect_linkers(
        mol1,
        mol2,
        set_linker=set_linker,
        label1=label1,
        label2=label2,
        headhead=headhead,
        tailtail=tailtail,
    ):
        return None

    mol1_n = mol1.GetNumAtoms()
    mol2_n = mol2.GetNumAtoms()
    mol1_coord = np.asarray(mol1.GetConformer(confId1).GetPositions(), dtype=float)
    mol2_coord = np.asarray(mol2.GetConformer(confId2).GetPositions(), dtype=float)
    tail_idx = mol1.GetIntProp('tail_idx')
    tail_ne_idx = mol1.GetIntProp('tail_ne_idx')
    head_idx = mol2.GetIntProp('head_idx')
    head_ne_idx = mol2.GetIntProp('head_ne_idx')

    mol1_tail_vec = mol1_coord[tail_ne_idx] - mol1_coord[tail_idx]
    mol2_head_vec = mol2_coord[head_ne_idx] - mol2_coord[head_idx]

    angle = calc.angle_vec(mol1_tail_vec, mol2_head_vec, rad=True)
    center = mol2_coord[head_ne_idx]
    if angle == 0:
        mol2_coord_rot = (mol2_coord - center) * -1 + center
    elif angle == np.pi:
        mol2_coord_rot = mol2_coord.copy()
    else:
        vcross = np.cross(mol1_tail_vec, mol2_head_vec)
        mol2_coord_rot = calc.rotate_rod(mol2_coord, vcross, (np.pi-angle), center=center)

    trans = mol1_coord[tail_ne_idx] - (bond_length * mol1_tail_vec / np.linalg.norm(mol1_tail_vec))
    mol2_coord_rot = mol2_coord_rot + trans - mol2_coord_rot[head_ne_idx]

    if random_rot == True:
        dih = np.random.uniform(-np.pi, np.pi)
    else:
        if dih_type == 'monomer':
            dih = calc.dihedral_coord(
                mol1_coord[mol1.GetIntProp('head_idx')],
                mol1_coord[tail_ne_idx],
                mol2_coord_rot[head_ne_idx],
                mol2_coord_rot[mol2.GetIntProp('tail_idx')],
                rad=True,
            )
        elif dih_type == 'bond':
            path1 = Chem.GetShortestPath(mol1, mol1.GetIntProp('head_idx'), tail_idx)
            path2 = Chem.GetShortestPath(mol2, head_idx, mol2.GetIntProp('tail_idx'))
            dih = calc.dihedral_coord(
                mol1_coord[path1[-3]],
                mol1_coord[path1[-2]],
                mol2_coord_rot[path2[1]],
                mol2_coord_rot[path2[2]],
                rad=True,
            )
        else:
            utils.radon_print('Illegal option of dih_type=%s.' % str(dih_type), level=3)
            return None
    mol2_coord_rot = calc.rotate_rod(mol2_coord_rot, -mol1_tail_vec, (dihedral-dih), center=mol2_coord_rot[head_ne_idx])

    keep_idx1 = np.delete(np.arange(mol1_n, dtype=int), tail_idx)
    keep_idx2 = np.delete(np.arange(mol2_n, dtype=int), head_idx)
    tail_ne_idx_new = int(tail_ne_idx - 1) if tail_idx < tail_ne_idx else int(tail_ne_idx)
    head_ne_idx_new = int(head_ne_idx - 1) if head_idx < head_ne_idx else int(head_ne_idx)

    return {
        'mol1_coord': mol1_coord,
        'mol2_coord_rot': mol2_coord_rot,
        'mol1_n': int(mol1_n),
        'del_idx1': int(tail_idx),
        'del_idx2': int(head_idx),
        'keep_idx1': keep_idx1,
        'keep_idx2': keep_idx2,
        'poly_coord': mol1_coord[keep_idx1],
        'mon_coord': mol2_coord_rot[keep_idx2],
        'tail_ne_idx_new': tail_ne_idx_new,
        'head_ne_idx_new': head_ne_idx_new,
        'charge_list': ['AtomicCharge', '_GasteigerCharge', 'RESP', 'RESP2', 'ESP', 'MullikenCharge', 'LowdinCharge'],
    }


def _connect_trial_cross_dmat(mol1, mol2, trial, poly_dmat=None, mon_dmat=None):
    if poly_dmat is None or mon_dmat is None or trial is None:
        return None
    try:
        tail_ne_idx = mol1.GetIntProp('tail_ne_idx')
        head_ne_idx = mol2.GetIntProp('head_ne_idx')
        poly_block = np.asarray(poly_dmat)[np.ix_(trial['keep_idx1'], [tail_ne_idx])]
        mon_block = np.asarray(mon_dmat)[np.ix_([head_ne_idx], trial['keep_idx2'])]
        return poly_block + 1 + mon_block
    except Exception:
        return None


def check_3d_structure_connect_trial(mol1, mol2, trial, poly_dmat=None, mon_dmat=None, dist_min=1.0, ignore_rad=3):
    if trial is None:
        return False
    p_coord = trial['poly_coord']
    m_coord = trial['mon_coord']
    candidate_idx = _local_proximity_candidate_indices(p_coord, m_coord, dist_min=dist_min)
    if candidate_idx.size == 0:
        return True

    p_local = p_coord[candidate_idx] if candidate_idx.size < len(p_coord) else p_coord
    cross_dmat = _connect_trial_cross_dmat(mol1, mol2, trial, poly_dmat=poly_dmat, mon_dmat=mon_dmat)
    if cross_dmat is not None and candidate_idx.size < len(p_coord):
        cross_dmat = cross_dmat[candidate_idx, :]
    return check_3d_proximity(p_local, coord2=m_coord, dist_min=dist_min, dmat=cross_dmat, ignore_rad=ignore_rad)


def _materialize_connected_mols(mol1, mol2, trial, *, res_name_1='RU0', res_name_2='RU0'):
    mol = combine_mols(mol1, mol2, res_name_1=res_name_1, res_name_2=res_name_2)
    mol1_n = int(trial['mol1_n'])

    for i in range(mol2.GetNumAtoms()):
        coord = trial['mol2_coord_rot'][i]
        mol.GetConformer(0).SetAtomPosition(i+mol1_n, Geom.Point3D(coord[0], coord[1], coord[2]))

    for charge in trial['charge_list']:
        try:
            idx_head = mol2.GetIntProp('head_idx') + mol1_n
            idx_head_ne = mol2.GetIntProp('head_ne_idx') + mol1_n
            idx_tail = mol1.GetIntProp('tail_idx')
            idx_tail_ne = mol1.GetIntProp('tail_ne_idx')
            need = (idx_head, idx_head_ne, idx_tail, idx_tail_ne)
            if not all(mol.GetAtomWithIdx(i).HasProp(charge) for i in need):
                continue
            head_charge = mol.GetAtomWithIdx(idx_head).GetDoubleProp(charge)
            head_ne_charge = mol.GetAtomWithIdx(idx_head_ne).GetDoubleProp(charge)
            tail_charge = mol.GetAtomWithIdx(idx_tail).GetDoubleProp(charge)
            tail_ne_charge = mol.GetAtomWithIdx(idx_tail_ne).GetDoubleProp(charge)
            mol.GetAtomWithIdx(idx_head_ne).SetDoubleProp(charge, head_charge + head_ne_charge)
            mol.GetAtomWithIdx(idx_tail_ne).SetDoubleProp(charge, tail_charge + tail_ne_charge)
        except Exception:
            continue

    del_idx1 = int(trial['del_idx1'])
    del_idx2 = int(trial['del_idx2']) + mol1_n - 1
    mol = utils.remove_atom(mol, del_idx1, angle_fix=True)
    mol = utils.remove_atom(mol, del_idx2, angle_fix=True)

    tail_ne_idx = int(trial['tail_ne_idx_new'])
    head_ne_idx = int(trial['head_ne_idx_new']) + mol1_n - 1
    if del_idx2 < head_ne_idx:
        head_ne_idx -= 1
    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.SINGLE, preserve_topology=True)
    new_bond = mol.GetBondBetweenAtoms(tail_ne_idx, head_ne_idx)
    new_bond.SetBoolProp('new_bond', True)

    Chem.SanitizeMol(mol)

    mol.GetAtomWithIdx(tail_ne_idx).SetBoolProp('tail_neighbor', False)
    mol.GetAtomWithIdx(head_ne_idx).SetBoolProp('head_neighbor', False)

    head_idx = mol1.GetIntProp('head_idx')
    if del_idx1 < head_idx:
        head_idx -= 1
    head_ne_idx_poly = mol1.GetIntProp('head_ne_idx')
    if del_idx1 < head_ne_idx_poly:
        head_ne_idx_poly -= 1
    tail_idx = mol2.GetIntProp('tail_idx') + mol1_n - 1
    if del_idx2 < tail_idx:
        tail_idx -= 1
    tail_ne_idx_poly = mol2.GetIntProp('tail_ne_idx') + mol1_n - 1
    if del_idx2 < tail_ne_idx_poly:
        tail_ne_idx_poly -= 1

    mol.SetIntProp('head_idx', head_idx)
    mol.SetIntProp('head_ne_idx', head_ne_idx_poly)
    mol.SetIntProp('tail_idx', tail_idx)
    mol.SetIntProp('tail_ne_idx', tail_ne_idx_poly)

    return mol


def connect_mols(mol1, mol2, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer', set_linker=True, label1=1, label2=1,
                headhead=False, tailtail=False, confId1=0, confId2=0, res_name_1='RU0', res_name_2='RU0'):
    """
    poly.connect_mols

    Connect tail atom in mol1 to head atom in mol2

    Args:
        mol1, mol2: RDkit Mol object (requiring AddHs and 3D position)

    Optional args:
        bond_length: Bond length of connecting bond (float, angstrome)
        random_rot: Dihedral angle around connecting bond is rotated randomly (boolean)
        dihedral: Dihedral angle around connecting bond (float, radian)
        dih_type: Definition type of dihedral angle (str; monomer, or bond)
        headhead: Connect with head-to-head
        tailtail: Connect with tail-to-tail
        confId1, confId2: Target conformer ID of mol1 and mol2
        res_name_1, res_name_2: Set residue name of PDB

    Returns:
        RDkit Mol object
    """

    if mol1 is None: return mol2
    if mol2 is None: return mol1
    trial = _prepare_connect_trial(
        mol1,
        mol2,
        bond_length=bond_length,
        dihedral=dihedral,
        random_rot=random_rot,
        dih_type=dih_type,
        set_linker=set_linker,
        label1=label1,
        label2=label2,
        headhead=headhead,
        tailtail=tailtail,
        confId1=confId1,
        confId2=confId2,
    )
    if trial is None:
        return mol1
    return _materialize_connected_mols(mol1, mol2, trial, res_name_1=res_name_1, res_name_2=res_name_2)


def combine_mols(mol1, mol2, res_name_1='RU0', res_name_2='RU0'):
    """
    poly.combine_mols

    Combining mol1 and mol2 taking over the angles, dihedrals, impropers, cmaps and cell data

    Args:
        mol1, mol2: RDkit Mol object

    Optional args:
        res_name_1, res_name_2: Set residue name of PDB

    Returns:
        RDkit Mol object
    """

    # Combine two molecules.
    # RDKit preserves per-atom properties (including SetDoubleProp charges)
    # correctly in CombineMols(). Charge loss that was observed in some
    # workflows actually came from later topology edits (RemoveAtom), which we
    # guard against in utils.remove_atom().
    mol = Chem.rdmolops.CombineMols(mol1, mol2)

    mol1_n = mol1.GetNumAtoms()
    mol2_n = mol2.GetNumAtoms()
    angles = {}
    dihedrals = {}
    impropers = {}
    cmaps = {}
    cell = None

    if hasattr(mol1, 'angles'):
        angles = mol1.angles.copy()
    if hasattr(mol2, 'angles'):
        for angle in mol2.angles.values():
            key = '%i,%i,%i' % (angle.a+mol1_n, angle.b+mol1_n, angle.c+mol1_n)
            angles[key] = utils.Angle(
                                a=angle.a+mol1_n,
                                b=angle.b+mol1_n,
                                c=angle.c+mol1_n,
                                ff=angle.ff
                            )

    if hasattr(mol1, 'dihedrals'):
        dihedrals = mol1.dihedrals.copy()
    if hasattr(mol2, 'dihedrals'):
        for dihedral in mol2.dihedrals.values():
            key = '%i,%i,%i,%i' % (dihedral.a+mol1_n, dihedral.b+mol1_n, dihedral.c+mol1_n, dihedral.d+mol1_n)
            dihedrals[key] = utils.Dihedral(
                                    a=dihedral.a+mol1_n,
                                    b=dihedral.b+mol1_n,
                                    c=dihedral.c+mol1_n,
                                    d=dihedral.d+mol1_n,
                                    ff=dihedral.ff
                                )

    if hasattr(mol1, 'impropers'):
        impropers = mol1.impropers.copy()
    if hasattr(mol2, 'impropers'):
        for improper in mol2.impropers.values():
            key = '%i,%i,%i,%i' % (improper.a+mol1_n, improper.b+mol1_n, improper.c+mol1_n, improper.d+mol1_n)
            impropers[key] = utils.Improper(
                                    a=improper.a+mol1_n,
                                    b=improper.b+mol1_n,
                                    c=improper.c+mol1_n,
                                    d=improper.d+mol1_n,
                                    ff=improper.ff
                                )
    
    if hasattr(mol1, 'cmaps'):
        cmaps = mol1.cmaps.copy()
    if hasattr(mol2, 'cmaps'):
        for cmap in mol2.cmaps.values():
            key = '%i,%i,%i,%i,%i' % (cmap.a+mol1_n, cmap.b+mol1_n, cmap.c+mol1_n, cmap.d+mol1_n, cmap.e+mol1_n)
            cmaps[key] = utils.CMAP(
                               a=cmap.a+mol1_n,
                               b=cmap.b+mol1_n,
                               c=cmap.c+mol1_n,
                               d=cmap.d+mol1_n,
                               e=cmap.e+mol1_n,
                               ff=cmap.ff
                           )

    if hasattr(mol1, 'cell'):
        cell = copy(mol1.cell)
    elif hasattr(mol2, 'cell'):
        cell = copy(mol2.cell)

    # Generate PDB information and repeating unit information
    resid = []
    if mol1.HasProp('num_units'):
        max_resid = mol1.GetIntProp('num_units')
    else:
        for i in range(mol1_n):
            atom = mol.GetAtomWithIdx(i)
            atom_name = atom.GetProp('ff_type') if atom.HasProp('ff_type') else atom.GetSymbol()
            if atom.GetPDBResidueInfo() is None:
                atom.SetMonomerInfo(
                    Chem.AtomPDBResidueInfo(
                        atom_name,
                        residueName=res_name_1,
                        residueNumber=1,
                        isHeteroAtom=False
                    )
                )
                resid.append(1)
            else:
                atom.GetPDBResidueInfo().SetName(atom_name)
                resid1 = atom.GetPDBResidueInfo().GetResidueNumber()
                resid.append(resid1)
        max_resid = max(resid) if len(resid) > 0 else 0

    for i in range(mol2_n):
        atom = mol.GetAtomWithIdx(i+mol1_n)
        atom_name = atom.GetProp('ff_type') if atom.HasProp('ff_type') else atom.GetSymbol()
        if atom.GetPDBResidueInfo() is None:
            atom.SetMonomerInfo(
                Chem.AtomPDBResidueInfo(
                    atom_name,
                    residueName=res_name_2,
                    residueNumber=1+max_resid,
                    isHeteroAtom=False
                )
            )
            resid.append(1+max_resid)
        else:
            atom.GetPDBResidueInfo().SetName(atom_name)
            resid2 = atom.GetPDBResidueInfo().GetResidueNumber()
            atom.GetPDBResidueInfo().SetResidueNumber(resid2+max_resid)
            resid.append(resid2+max_resid)

    max_resid = max(resid) if len(resid) > 0 else 0
    mol.SetIntProp('num_units', max_resid)

    setattr(mol, 'angles', angles)
    setattr(mol, 'dihedrals', dihedrals)
    setattr(mol, 'impropers', impropers)
    if len(cmaps) > 0:
        setattr(mol, 'cmaps', cmaps)
    if mol1.HasProp('pair_style'):
        mol.SetProp('pair_style', mol1.GetProp('pair_style'))
    elif mol2.HasProp('pair_style'):
        mol.SetProp('pair_style', mol2.GetProp('pair_style'))
    if mol1.HasProp('bond_style'):
        mol.SetProp('bond_style', mol1.GetProp('bond_style'))
    elif mol2.HasProp('bond_style'):
        mol.SetProp('bond_style', mol2.GetProp('bond_style'))
    if mol1.HasProp('angle_style'):
        mol.SetProp('angle_style', mol1.GetProp('angle_style'))
    elif mol2.HasProp('angle_style'):
        mol.SetProp('angle_style', mol2.GetProp('angle_style'))
    if mol1.HasProp('dihedral_style'):
        mol.SetProp('dihedral_style', mol1.GetProp('dihedral_style'))
    elif mol2.HasProp('dihedral_style'):
        mol.SetProp('dihedral_style', mol2.GetProp('dihedral_style'))
    if mol1.HasProp('improper_style'):
        mol.SetProp('improper_style', mol1.GetProp('improper_style'))
    elif mol2.HasProp('improper_style'):
        mol.SetProp('improper_style', mol2.GetProp('improper_style'))
    if mol1.HasProp('ff_name'):
        mol.SetProp('ff_name', mol1.GetProp('ff_name'))
    elif mol2.HasProp('ff_name'):
        mol.SetProp('ff_name', mol2.GetProp('ff_name'))
    if mol1.HasProp('ff_class'):
        mol.SetProp('ff_class', mol1.GetProp('ff_class'))
    elif mol2.HasProp('ff_class'):
        mol.SetProp('ff_class', mol2.GetProp('ff_class'))
    if cell is not None:
        setattr(mol, 'cell', cell)

    return mol


##########################################################
# Polymer chain generator (non random walk)
##########################################################
def simple_polymerization(mols, m_idx, chi_inv, start_num=0, init_poly=None, headhead=False, confId=0,
                bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer', res_name_1='RU0', res_name_2=None):
    mols_copy = []
    mols_inv = []
    poly = None

    if type(init_poly) == Chem.Mol:
        poly = utils.deepcopy_mol(init_poly)
        set_linker_flag(poly)

    for mol in mols:
        set_linker_flag(mol)
        mols_copy.append(utils.deepcopy_mol(mol))
        mols_inv.append(calc.mirror_inversion_mol(mol))

    if res_name_2 is None:
        res_name_2 = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]

    utils.radon_print('Start poly.simple_polymerization.')

    for i in tqdm(range(start_num, len(m_idx)), desc='[Polymerization]', disable=const.tqdm_disable):
        if chi_inv[i]:
            mol_c = mols_inv[m_idx[i]]
        else:
            mol_c = mols_copy[m_idx[i]]

        if headhead and i % 2 == 0:
            poly = connect_mols(poly, mol_c, tailtail=True, set_linker=False,
                    bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                    confId1=confId, confId2=confId, res_name_1=res_name_1, res_name_2=res_name_2[m_idx[i]])
        else:
            poly = connect_mols(poly, mol_c, set_linker=False,
                    bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                    confId1=confId, confId2=confId, res_name_1=res_name_1, res_name_2=res_name_2[m_idx[i]])

    return poly


def polymerize_mols(mol, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
                        bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.polymerize_mols.', level=1)

    m_idx = gen_monomer_array(1, n)
    chi_inv, _ = gen_chiral_inv_array([mol], m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    poly = simple_polymerization(
        [mol], m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type
    )

    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.polymerize_mols. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def copolymerize_mols(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
                        bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.copolymerize_mols.', level=1)

    m_idx = gen_monomer_array(len(mols), n, copoly='alt')
    chi_inv, _ = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    poly = simple_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type
    )

    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.copolymerize_mols. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def random_copolymerize_mols(mols, n, ratio=None, reac_ratio=[], init_poly=None, headhead=False, confId=0,
                                tacticity='atactic', atac_ratio=0.5, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.random_copolymerize_mols.', level=1)

    m_idx = gen_monomer_array(len(mols), n, copoly='random', ratio=ratio, reac_ratio=reac_ratio)
    chi_inv, _ = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    poly = simple_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type
    )
    
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.random_copolymerize_mols. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def block_copolymerize_mols(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
                        bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.block_copolymerize_mols.', level=1)

    m_idx = gen_monomer_array(len(mols), n, copoly='block')
    chi_inv, _ = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    poly = simple_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type
    )
    
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.block_copolymerize_mols. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly


def terminate_mols(poly, mol1, mol2=None, confId=0, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer'):
    """
    poly.terminate_mols

    Simple termination function of polymer of RDkit Mol object

    Args:
        poly: polymer (RDkit Mol object)
        mol1: terminated substitute at head (and tail) (RDkit Mol object)

    Optional args:
        mol2: terminated substitute at tail (RDkit Mol object)
        bond_length: Bond length of connecting bond (float, angstrome)
        random_rot: Dihedral angle around connecting bond is rotated randomly (boolean)
        dihedral: Dihedral angle around connecting bond (float, radian)
        dih_type: Definition type of dihedral angle (str; monomer, or bond)

    Returns:
        Rdkit Mol object
    """
    if mol2 is None: mol2 = mol1
    poly_c = utils.deepcopy_mol(poly)
    mol1_c = utils.deepcopy_mol(mol1)
    mol2_c = utils.deepcopy_mol(mol2)
    res_name_1 = 'TU0'
    res_name_2 = 'TU1'
        
    if Chem.MolToSmiles(mol1_c) == '[H][3H]' or Chem.MolToSmiles(mol1_c) == '[3H][H]':
        head_idx = poly_c.GetIntProp('head_idx')
        residue_number = 1 + poly_c.GetIntProp('num_units')
        _set_atom_residue(poly_c, head_idx, res_name_1, residue_number).SetIsotope(0)
        poly_c.SetIntProp('num_units', 1+poly_c.GetIntProp('num_units'))
    else:
        poly_c = connect_mols(mol1_c, poly_c, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                            res_name_1=res_name_1)

    if Chem.MolToSmiles(mol2_c) == '[H][3H]' or Chem.MolToSmiles(mol2_c) == '[3H][H]':
        tail_idx = poly_c.GetIntProp('tail_idx')
        residue_number = 1 + poly_c.GetIntProp('num_units')
        _set_atom_residue(poly_c, tail_idx, res_name_2, residue_number).SetIsotope(0)
        poly_c.SetIntProp('num_units', 1+poly_c.GetIntProp('num_units'))
    else:
        poly_c = connect_mols(poly_c, mol2_c, bond_length=bond_length, dihedral=dihedral, random_rot=random_rot, dih_type=dih_type,
                            res_name_2=res_name_2)

    set_terminal_idx(poly_c)

    return poly_c


##########################################################
# Polymer chain generator with self-avoiding random walk
##########################################################
def random_walk_polymerization(mols, m_idx, chi_inv, start_num=0, init_poly=None, headhead=False, confId=0,
            dist_min=0.7, retry=60, rollback=3, rollback_shaking=False, retry_step=80, retry_opt_step=4, tacticity=None,
            res_name_init='INI', res_name=None, label=None, label_init=1, ff=None, work_dir=None, omp=1, mpi=0, gpu=0, mp_idx=None, restart=None):
    """
    poly.random_walk_polymerization

    Polymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mols: list of RDkit Mol object
        m_idx: Input array of repeating units by index number of mols
        chi_inv: Input boolean array of chiral inversion

    Optional args:
        start_num: Index number of m_idx of starting point
        init_poly: Perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry random_walk_polymerization (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        work_dir: Work directory path of external minimizer (str, requiring when opt is external minimizer)
        ff: Force field object (requiring when opt is external minimizer)
        omp: Number of threads of OpenMP in external minimizer (int)
        mpi: Number of MPI process in external minimizer (int)
        gpu: Number of GPU in external minimizer (int)

    Returns:
        Rdkit Mol object
    """
    utils.radon_print('Start poly.random_walk_polymerization.')

    mols = _resolve_mol_list(mols)
    init_poly = _resolve_mol_like(init_poly) if init_poly is not None else None
    budget = None
    if start_num == 0 and init_poly is None:
        budget = _resolve_rw_retry_budget(
            mols,
            retry,
            rollback,
            rollback_shaking,
            retry_step,
            retry_opt_step,
            total_steps=len(m_idx),
        )
        retry = budget['retry']
        rollback = budget['rollback']
        rollback_shaking = budget['rollback_shaking']
        retry_step = budget['retry_step']
        retry_opt_step = budget['retry_opt_step']
    if budget and budget['changed']:
        details = ', '.join(f'{key} {old}->{new}' for key, (old, new) in budget['changed'].items())
        utils.radon_print(
            f'Adaptive random-walk budget applied (rigidity={budget["rigidity"]:.2f}): {details}',
            level=1,
        )
    rst_flag = _effective_restart_flag(work_dir, restart)
    payload = _rw_payload('random_walk_polymerization', smiles=[Chem.MolToSmiles(m, isomericSmiles=True) for m in mols], m_idx=list(np.asarray(m_idx).tolist()), chi_inv=[bool(x) for x in chi_inv], start_num=int(start_num), headhead=bool(headhead), tacticity=tacticity, label=label, label_init=label_init, init_poly_smiles=(Chem.MolToSmiles(init_poly, isomericSmiles=True) if isinstance(init_poly, Chem.Mol) else None))
    if rst_flag:
        cached = _rw_load(work_dir, 'random_walk_polymerization', payload)
        if cached is not None:
            return cached

    if len(m_idx) != len(chi_inv):
        utils.radon_print('Inconsistency length of m_idx and chi_inv', level=3)
    if len(mols) <= max(m_idx):
        utils.radon_print('Illegal index number was found in m_idx', level=3)

    mols_copy = []
    mols_inv = []
    mols_dmat = []
    mols_inv_dmat = []
    has_ring = False
    retry_flag = False
    tri_coord = None
    bond_coord = None    
    poly = None
    poly_copy = [None]
    poly_dmat_current = None
    total_steps = int(len(m_idx))
    walk_started_at = time.perf_counter()
    progress_state = {
        'last_heartbeat': walk_started_at,
        'step_interval': _rw_progress_step_interval(total_steps),
    }

    if start_num == 0 and const.tqdm_disable and total_steps > 0:
        utils.radon_print(
            '[RW] heartbeat logging active: total_steps=%i; heartbeat=%ss; accepted-step-log-every=%i'
            % (total_steps, f'{float(getattr(const, "rw_heartbeat_seconds", 0.0)):.0f}', progress_state['step_interval']),
            level=1,
        )

    if res_name is None:
        res_name = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]
    else:
        if len(mols) != len(res_name):
            utils.radon_print('Inconsistency length of mols and res_name', level=3)

    if label is None:
        label = [[1, 1] for x in range(len(mols))]
    else:
        if len(mols) != len(label):
            utils.radon_print('Inconsistency length of mols and label', level=3)

    if type(init_poly) == Chem.Mol:
        poly = utils.deepcopy_mol(init_poly)
        set_linker_flag(poly, label=label_init)
        poly_copy = []
        if dist_min > 1.0:
            poly_dmat_current = np.asarray(Chem.GetDistanceMatrix(_rdkit_clone_mol(poly)))
        sssr_tmp = Chem.GetSSSR(poly)
        if type(sssr_tmp) is int:
            if sssr_tmp > 0:
                has_ring = True
        elif len(sssr_tmp) > 0:  # For RDKit version >= 2022.09
            has_ring = True

    for i, mol in enumerate(mols):
        set_linker_flag(mol, label=label[i][0])
        mols_copy.append(utils.deepcopy_mol(mol))
        mols_inv.append(calc.mirror_inversion_mol(mol))
        if dist_min > 1.0:
            mols_dmat.append(np.asarray(Chem.GetDistanceMatrix(_rdkit_clone_mol(mols_copy[-1]))))
            mols_inv_dmat.append(np.asarray(Chem.GetDistanceMatrix(_rdkit_clone_mol(mols_inv[-1]))))
        else:
            mols_dmat.append(None)
            mols_inv_dmat.append(None)
        sssr_tmp = Chem.GetSSSR(mol)
        if type(sssr_tmp) is int:
            if sssr_tmp > 0:
                has_ring = True
        elif len(sssr_tmp) > 0:  # For RDKit version >= 2022.09
            has_ring = True

    for i in tqdm(range(start_num, len(m_idx)), desc='[Polymerization]', disable=const.tqdm_disable):
        dmat = poly_dmat_current
        step_started_at = time.perf_counter()
    
        if chi_inv[i]:
            mol_c = mols_inv[m_idx[i]]
            mon_dmat = mols_inv_dmat[m_idx[i]]
        else:
            mol_c = mols_copy[m_idx[i]]
            mon_dmat = mols_dmat[m_idx[i]]

        if i == 0:
            res_name_1 = res_name_init
        else:
            res_name_1 = res_name[m_idx[i-1]]

        if type(poly) is Chem.Mol:
            poly_copy.append(_rw_growth_clone_mol(poly))
        else:
            poly_copy.append(_rw_growth_clone_mol(mol_c))

        if len(poly_copy) > rollback:
            del poly_copy[0]

        for r in range(retry_step*(1+retry_opt_step)):
            check_3d = False
            _rw_emit_heartbeat(
                step_idx=i+1,
                total_steps=total_steps,
                retry_idx=r+1,
                retry_total=retry_step*(1+retry_opt_step),
                step_started_at=step_started_at,
                walk_started_at=walk_started_at,
                state=progress_state,
            )
            if i > 0:
                label1 = label[m_idx[i-1]][1]
            elif type(init_poly) == Chem.Mol:
                label1 = label_init
            else:
                label1 = 1

            if i == 0 and init_poly is None:
                break

            trial = _prepare_connect_trial(
                poly,
                mol_c,
                random_rot=True,
                set_linker=True,
                tailtail=bool(headhead and i % 2 == 0),
                confId2=confId,
                label1=label1,
                label2=(label[m_idx[i]][1] if headhead and i % 2 == 0 else label[m_idx[i]][0]),
            )
            if trial is None:
                poly = _rw_growth_clone_mol(poly_copy[-1]) if type(poly_copy[-1]) is Chem.Mol else None
                continue

            if dmat is None and dist_min > 1.0:
                dmat = np.asarray(Chem.GetDistanceMatrix(_rdkit_clone_mol(poly)))
                poly_dmat_current = dmat

            check_3d = check_3d_structure_connect_trial(
                poly,
                mol_c,
                trial,
                poly_dmat=dmat,
                mon_dmat=mon_dmat,
                dist_min=dist_min,
            )

            if check_3d and has_ring:
                check_3d, tri_coord_new, bond_coord_new = check_3d_bond_ring_intersection(
                    poly,
                    mon=mol_c,
                    poly_coord=trial['poly_coord'],
                    mon_coord=trial['mon_coord'],
                    tri_coord=tri_coord,
                    bond_coord=None,
                    poly_atom_indices=trial['keep_idx1'],
                    mon_atom_indices=trial['keep_idx2'],
                )

            if check_3d:
                poly = _materialize_connected_mols(poly, mol_c, trial, res_name_1=res_name_1, res_name_2=res_name[m_idx[i]])
                if dmat is not None or tacticity:
                    post_dmat = None
                    if dmat is not None:
                        post_dmat = np.asarray(Chem.GetDistanceMatrix(_rdkit_clone_mol(poly)))
                    check_3d = check_3d_structure_poly(
                        poly,
                        mol_c,
                        post_dmat,
                        dist_min=dist_min,
                        check_bond_length=bool(dmat is not None),
                        tacticity=tacticity,
                    )
                    if not check_3d:
                        poly = _rw_growth_clone_mol(poly_copy[-1]) if type(poly_copy[-1]) is Chem.Mol else None
                        tri_coord = None
                        bond_coord = None
                        continue
                    if post_dmat is not None:
                        poly_dmat_current = post_dmat

            if check_3d:
                if has_ring:
                    tri_coord = tri_coord_new
                    bond_coord = bond_coord_new
                _rw_emit_step_progress(
                    step_idx=i+1,
                    total_steps=total_steps,
                    retries_used=r+1,
                    step_started_at=step_started_at,
                    walk_started_at=walk_started_at,
                    state=progress_state,
                )
                break
            elif r < retry_step * (1 + retry_opt_step) - 1:
                poly = _rw_growth_clone_mol(poly_copy[-1]) if type(poly_copy[-1]) is Chem.Mol else None
                if r == 0 or (r+1) % 100 == 0:
                    utils.radon_print('Retry random walk step %i, %i/%i' % (i+1, r+1, retry_step*(1+retry_opt_step)))
            else:
                retry_flag = True
                utils.radon_print(
                    'Reached maximum number of retrying step in random walk step %i of poly.random_walk_polymerization.' % (i+1),
                    level=1)

        if retry_flag: break

    if retry_flag:
        if retry <= 0:
            utils.radon_print(
                'poly.random_walk_polymerization is failure because reached maximum number of rollback times in random walk step %i.' % (i+1),
                level=3)
        else:
            utils.radon_print(
                'Retry poly.random_walk_polymerization and rollback %i steps. Remaining %i times.' % (len(poly_copy), retry),
                level=1)
            retry -= 1
            start_num = i-len(poly_copy)+1
            if start_num > 0:
                label_init = label[m_idx[start_num-1]][1]
            rb_poly = poly_copy[0]

            if MD_avail and rollback_shaking and type(rb_poly) is Chem.Mol:
                utils.radon_print('Molecular geometry shaking by a short time and high temperature MD simulation')
                if ff is None:
                    ff = GAFF2_mod()
                # Use a robust check: if *any* atom already has AtomicCharge, preserve it.
                has_q = False
                try:
                    has_q = any(a.HasProp('AtomicCharge') for a in rb_poly.GetAtoms())
                except Exception:
                    has_q = False
                if has_q:
                    ff.ff_assign(rb_poly)
                else:
                    ff.ff_assign(rb_poly, charge='gasteiger')
                rb_poly, _ = md.quick_rw(rb_poly, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)

            poly = random_walk_polymerization(
                mols, m_idx, chi_inv, start_num=start_num, init_poly=rb_poly, headhead=headhead, confId=confId,
                dist_min=dist_min, retry=retry, rollback=rollback, rollback_shaking=rollback_shaking,
                retry_step=retry_step, retry_opt_step=retry_opt_step, tacticity=tacticity,
                res_name_init=res_name_init, res_name=res_name, label=label, label_init=label_init,
                ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, restart=restart
            )

    poly = _rw_finalize_bonded_terms(poly)

    _rw_save(work_dir, 'random_walk_polymerization', payload, poly)
    return poly


def polymerize_rw(mol, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=60, rollback=3, rollback_shaking=False, retry_step=80, retry_opt_step=4, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name='RU0', ff=None, work_dir=None, omp=0, mpi=1, gpu=0, mp_idx=None, restart=None):
    """
    poly.polymerize_rw

    Homo-polymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mol: RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        init_poly: polymerize_rw perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry polymerize_rw (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        work_dir: Work directory path of external minimizer (str, requiring when opt is external minimizer)
        ff: Force field object (requiring when opt is external minimizer)
        omp: Number of threads of OpenMP in external minimizer (int)
        mpi: Number of MPI process in external minimizer (int)
        gpu: Number of GPU in external minimizer (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.polymerize_rw.', level=1)

    mol = _resolve_mol_like(mol)
    init_poly = _resolve_mol_like(init_poly) if init_poly is not None else None
    ter1 = _resolve_mol_like(ter1) if ter1 is not None else None
    ter2 = _resolve_mol_like(ter2) if ter2 is not None else None
    rst_flag = _effective_restart_flag(work_dir, restart)
    payload = _rw_payload(
        'polymerize_rw',
        smiles=(Chem.MolToSmiles(mol, isomericSmiles=True) if isinstance(mol, Chem.Mol) else None),
        n=int(n),
        headhead=bool(headhead),
        tacticity=tacticity,
        atac_ratio=float(atac_ratio),
        label=label,
        ter1_smiles=(Chem.MolToSmiles(ter1, isomericSmiles=True) if isinstance(ter1, Chem.Mol) else None),
        ter2_smiles=(Chem.MolToSmiles(ter2, isomericSmiles=True) if isinstance(ter2, Chem.Mol) else None),
        init_poly_smiles=(Chem.MolToSmiles(init_poly, isomericSmiles=True) if isinstance(init_poly, Chem.Mol) else None),
    )
    plan_payload = _rw_payload(
        'polymerize_rw_plan',
        smiles=(Chem.MolToSmiles(mol, isomericSmiles=True) if isinstance(mol, Chem.Mol) else None),
        n=int(n),
        headhead=bool(headhead),
        tacticity=tacticity,
        atac_ratio=float(atac_ratio),
        label=label,
        ter1_smiles=(Chem.MolToSmiles(ter1, isomericSmiles=True) if isinstance(ter1, Chem.Mol) else None),
        ter2_smiles=(Chem.MolToSmiles(ter2, isomericSmiles=True) if isinstance(ter2, Chem.Mol) else None),
        init_poly_smiles=(Chem.MolToSmiles(init_poly, isomericSmiles=True) if isinstance(init_poly, Chem.Mol) else None),
    )
    plan = _rw_load_state(work_dir, 'polymerize_rw_plan', plan_payload) if rst_flag else None
    if plan is not None and isinstance(plan, dict):
        m_idx = [int(x) for x in plan.get('m_idx', [])]
        chi_inv = [bool(x) for x in plan.get('chi_inv', [])]
        tacticity = plan.get('effective_tacticity', tacticity)
    else:
        m_idx = gen_monomer_array(1, n)
        chi_inv, check_chi = gen_chiral_inv_array([mol], m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
        if not check_chi:
            tacticity = None
        _rw_save_state(
            work_dir,
            'polymerize_rw_plan',
            plan_payload,
            {'m_idx': list(m_idx), 'chi_inv': [bool(x) for x in chi_inv], 'effective_tacticity': tacticity},
        )

    if rst_flag:
        cached = _rw_load(work_dir, 'polymerize_rw', payload)
        if cached is not None:
            return cached

    if type(ter1) is Chem.Mol:
        if ter2 is None:
            ter2 = ter1
        mols = [mol, ter1, ter2]
        res_name = [res_name, 'TU0', 'TU1']
        m_idx = [1, *m_idx, 2]
        chi_inv = [False, *chi_inv, False]
        if label is None:
            label = [1, 1]
        label = [label, [label_ter1, label_ter1], [label_ter2, label_ter2]]
    else:
        mols = [mol]
        res_name = [res_name]
        if label is not None:
            label = [label]

    poly = random_walk_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        dist_min=dist_min, retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step,
        tacticity=tacticity, res_name=res_name, label=label, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, mp_idx=mp_idx, restart=restart
    )

    if type(ter1) is Chem.Mol:
        set_terminal_idx(poly)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.polymerize_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    _rw_save(work_dir, 'polymerize_rw', payload, poly)
    return poly


def polymerize_rw_old(mol, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=10, rollback=5, rollback_shaking=False, retry_step=0, retry_opt_step=50, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name='RU0', ff=None, work_dir=None, omp=0, mpi=1, gpu=0, restart=None):
    # Backward campatibility
    return polymerize_rw(mol, n, init_poly=init_poly, headhead=headhead, confId=confId, tacticity=tacticity, atac_ratio=atac_ratio, dist_min=dist_min, 
            retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step, ter1=ter1, ter2=ter2,
            label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name, ff=ff,
            work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, restart=restart)


def polymerize_rw_mp(mol, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=60, rollback=3, rollback_shaking=False, retry_step=80, retry_opt_step=4, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name='RU0',
            ff=None, work_dir=None, omp=0, mpi=1, gpu=0, nchain=1, mp=None, fail_copy=True):

    utils.picklable(mol)
    if type(ter1) is Chem.Mol:
        utils.picklable(ter1)
    if type(ter2) is Chem.Mol:
        utils.picklable(ter2)

    if mp is None:
        mp = utils.cpu_count()
    np = max([nchain, mp])

    c = utils.picklable_const()
    args = [(mol, n, init_poly, headhead, confId, tacticity, atac_ratio, dist_min, retry, rollback, rollback_shaking, retry_step, retry_opt_step,
            ter1, ter2, label, label_ter1, label_ter2, res_name, ff, work_dir, omp, mpi, gpu, mp_idx, c) for mp_idx in range(np)]

    polys = polymerize_mp_exec(_polymerize_rw_mp_worker, args, mp, nchain=nchain, fail_copy=fail_copy)

    return polys


def _polymerize_rw_mp_worker(args):
    (mol, n, init_poly, headhead, confId, tacticity, atac_ratio, dist_min, retry, rollback, rollback_shaking, retry_step, retry_opt_step,
        ter1, ter2, label, label_ter1, label_ter2, res_name, ff, work_dir, omp, mpi, gpu, mp_idx, c) = args
    utils.restore_const(c)

    try:
        poly = polymerize_rw(mol, n, init_poly=init_poly, headhead=headhead, confId=confId,
                    tacticity=tacticity, atac_ratio=atac_ratio, dist_min=dist_min, 
                    retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step,
                    ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name,
                    ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, mp_idx=mp_idx)
        utils.picklable(poly)
    except BaseException as e:
        utils.radon_print('%s' % e)
        poly = None

    return poly


def copolymerize_rw(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=60, rollback=3, rollback_shaking=False, retry_step=80, retry_opt_step=4, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, ff=None, work_dir=None, omp=0, mpi=1, gpu=0, mp_idx=None, restart=None):
    """
    poly.copolymerize_rw

    Alternating copolymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mols: list of RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        init_poly: polymerize_rw perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry polymerize_rw (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        work_dir: Work directory path of external minimizer (str, requiring when opt is external minimizer)
        ff: Force field object (requiring when opt is external minimizer)
        omp: Number of threads of OpenMP in external minimizer (int)
        mpi: Number of MPI process in external minimizer (int)
        gpu: Number of GPU in external minimizer (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.copolymerize_rw.', level=1)

    mols = _resolve_mol_list(mols)
    init_poly = _resolve_mol_like(init_poly) if init_poly is not None else None
    ter1 = _resolve_mol_like(ter1) if ter1 is not None else None
    ter2 = _resolve_mol_like(ter2) if ter2 is not None else None
    rst_flag = _effective_restart_flag(work_dir, restart)
    payload = _rw_payload('copolymerize_rw', smiles=[Chem.MolToSmiles(m, isomericSmiles=True) for m in mols], n=n if isinstance(n, list) else int(n), headhead=bool(headhead), tacticity=tacticity, atac_ratio=float(atac_ratio), label=label, ter1_smiles=(Chem.MolToSmiles(ter1, isomericSmiles=True) if isinstance(ter1, Chem.Mol) else None), ter2_smiles=(Chem.MolToSmiles(ter2, isomericSmiles=True) if isinstance(ter2, Chem.Mol) else None), init_poly_smiles=(Chem.MolToSmiles(init_poly, isomericSmiles=True) if isinstance(init_poly, Chem.Mol) else None))
    plan_payload = _rw_payload('copolymerize_rw_plan', smiles=[Chem.MolToSmiles(m, isomericSmiles=True) for m in mols], n=n if isinstance(n, list) else int(n), headhead=bool(headhead), tacticity=tacticity, atac_ratio=float(atac_ratio), label=label, ter1_smiles=(Chem.MolToSmiles(ter1, isomericSmiles=True) if isinstance(ter1, Chem.Mol) else None), ter2_smiles=(Chem.MolToSmiles(ter2, isomericSmiles=True) if isinstance(ter2, Chem.Mol) else None), init_poly_smiles=(Chem.MolToSmiles(init_poly, isomericSmiles=True) if isinstance(init_poly, Chem.Mol) else None))
    plan = _rw_load_state(work_dir, 'copolymerize_rw_plan', plan_payload) if rst_flag else None
    if plan is not None and isinstance(plan, dict):
        m_idx = [int(x) for x in plan.get('m_idx', [])]
        chi_inv = [bool(x) for x in plan.get('chi_inv', [])]
        tacticity = plan.get('effective_tacticity', tacticity)
    else:
        m_idx = gen_monomer_array(len(mols), n, copoly='alt')
        chi_inv, check_chi = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
        if not check_chi:
            tacticity = None
        _rw_save_state(
            work_dir,
            'copolymerize_rw_plan',
            plan_payload,
            {'m_idx': list(m_idx), 'chi_inv': [bool(x) for x in chi_inv], 'effective_tacticity': tacticity},
        )

    if rst_flag:
        cached = _rw_load(work_dir, 'copolymerize_rw', payload)
        if cached is not None:
            return cached

    if res_name is None:
        res_name = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]

    if type(ter1) is Chem.Mol:
        if ter2 is None:
            ter2 = ter1
        n = len(mols)
        mols = [*mols, ter1, ter2]
        res_name = [*res_name, 'TU0', 'TU1']
        m_idx = [n, *m_idx, n+1]
        chi_inv = [False, *chi_inv, False]
        if label is not None:
            label = [*label, [label_ter1, label_ter1], [label_ter2, label_ter2]]

    poly = random_walk_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId, dist_min=dist_min,
        retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step,
        tacticity=tacticity, res_name=res_name, label=label, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, mp_idx=mp_idx, restart=restart
    )

    if type(ter1) is Chem.Mol:
        set_terminal_idx(poly)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.copolymerize_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    _rw_save(work_dir, 'copolymerize_rw', payload, poly)
    return poly


def copolymerize_rw_old(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=10, rollback=5, rollback_shaking=False, retry_step=0, retry_opt_step=50, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    # Backward campatibility
    return copolymerize_rw(mols, n, init_poly=init_poly, headhead=headhead, confId=confId, tacticity=tacticity, atac_ratio=atac_ratio, dist_min=dist_min, 
            retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step, ter1=ter1, ter2=ter2,
            label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name, ff=ff,
            work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, restart=restart)


def copolymerize_rw_mp(mols, n, init_poly=None, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=100, rollback=5, rollback_shaking=False, retry_step=200, retry_opt_step=0, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, 
            ff=None, work_dir=None, omp=0, mpi=1, gpu=0, nchain=1, mp=None, fail_copy=True):

    for i in range(len(mols)):
        utils.picklable(mols[i])
    if type(ter1) is Chem.Mol:
        utils.picklable(ter1)
        if type(ter2) is Chem.Mol:
            utils.picklable(ter2)
        else:
            ter2 = ter1

    if mp is None:
        mp = utils.cpu_count()
    np = max([nchain, mp])

    c = utils.picklable_const()
    args = [(mols, n, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, rollback_shaking, retry_step, retry_opt_step,
            ter1, ter2, label, label_ter1, label_ter2, res_name, ff, work_dir, omp, mpi, gpu, mp_idx, c) for mp_idx in range(np)]

    polys = polymerize_mp_exec(_copolymerize_rw_mp_worker, args, mp, nchain=nchain, fail_copy=fail_copy)

    return polys


def _copolymerize_rw_mp_worker(args):
    (mols, n, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, rollback_shaking, retry_step, retry_opt_step,
        ter1, ter2, label, label_ter1, label_ter2, res_name, ff, work_dir, omp, mpi, gpu, mp_idx, c) = args
    utils.restore_const(c)

    try:
        poly = copolymerize_rw(mols, n, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio, dist_min=dist_min, 
                    retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step,
                    ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name,
                    ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, mp_idx=mp_idx)
        utils.picklable(poly)
    except BaseException as e:
        utils.radon_print('%s' % e)
        poly = None

    return poly


def random_copolymerize_rw(mols, n, ratio=None, reac_ratio=[], init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=60, rollback=3, rollback_shaking=False, retry_step=80, retry_opt_step=4, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, ff=None, work_dir=None, omp=0, mpi=1, gpu=0, mp_idx=None, name: str | None = None, restart=None):
    """
    poly.random_copolymerize_rw

    Random copolymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mols: List of RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        ratio: List of monomer composition ratio (float)
        reac_ratio: List of monomer reactivity ratio (float)
        init_poly: polymerize_rw perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry polymerize_rw (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        work_dir: Work directory path of external minimizer (str)
        ff: Force field object (requiring when opt is external minimizer)
        omp: Number of threads of OpenMP in external minimizer (int)
        mpi: Number of MPI process in external minimizer (int)
        gpu: Number of GPU in external minimizer (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.random_copolymerize_rw.', level=1)

    mols = _resolve_mol_list(mols)
    init_poly = _resolve_mol_like(init_poly) if init_poly is not None else None
    ter1 = _resolve_mol_like(ter1) if ter1 is not None else None
    ter2 = _resolve_mol_like(ter2) if ter2 is not None else None
    rst_flag = _effective_restart_flag(work_dir, restart)
    if ratio is None:
        ratio = np.full(len(mols), 1 / len(mols))
    ratio = ratio_to_prob(ratio)

    payload = _rw_payload('random_copolymerize_rw', smiles=[Chem.MolToSmiles(m, isomericSmiles=True) for m in mols], n=int(n), ratio=ratio, reac_ratio=list(np.asarray(reac_ratio).tolist()) if len(reac_ratio) > 0 else [], headhead=bool(headhead), tacticity=tacticity, atac_ratio=float(atac_ratio), label=label, ter1_smiles=(Chem.MolToSmiles(ter1, isomericSmiles=True) if isinstance(ter1, Chem.Mol) else None), ter2_smiles=(Chem.MolToSmiles(ter2, isomericSmiles=True) if isinstance(ter2, Chem.Mol) else None), init_poly_smiles=(Chem.MolToSmiles(init_poly, isomericSmiles=True) if isinstance(init_poly, Chem.Mol) else None))
    plan_payload = _rw_payload('random_copolymerize_rw_plan', smiles=[Chem.MolToSmiles(m, isomericSmiles=True) for m in mols], n=int(n), ratio=ratio, reac_ratio=list(np.asarray(reac_ratio).tolist()) if len(reac_ratio) > 0 else [], headhead=bool(headhead), tacticity=tacticity, atac_ratio=float(atac_ratio), label=label, ter1_smiles=(Chem.MolToSmiles(ter1, isomericSmiles=True) if isinstance(ter1, Chem.Mol) else None), ter2_smiles=(Chem.MolToSmiles(ter2, isomericSmiles=True) if isinstance(ter2, Chem.Mol) else None), init_poly_smiles=(Chem.MolToSmiles(init_poly, isomericSmiles=True) if isinstance(init_poly, Chem.Mol) else None))
    plan = _rw_load_state(work_dir, 'random_copolymerize_rw_plan', plan_payload) if rst_flag else None
    if rst_flag:
        cached = _rw_load(work_dir, 'random_copolymerize_rw', payload)
        if cached is not None:
            return cached

    if len(mols) != len(ratio):
        utils.radon_print('Inconsistency length of mols and ratio', level=3)

    if plan is not None and isinstance(plan, dict):
        m_idx = [int(x) for x in plan.get('m_idx', [])]
        chi_inv = [bool(x) for x in plan.get('chi_inv', [])]
        tacticity = plan.get('effective_tacticity', tacticity)
    else:
        m_idx = gen_monomer_array(len(mols), n, copoly='random', ratio=ratio, reac_ratio=reac_ratio)
        chi_inv, check_chi = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
        if not check_chi:
            tacticity = None
        _rw_save_state(
            work_dir,
            'random_copolymerize_rw_plan',
            plan_payload,
            {'m_idx': list(m_idx), 'chi_inv': [bool(x) for x in chi_inv], 'effective_tacticity': tacticity},
        )

    if res_name is None:
        res_name = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]

    if type(ter1) is Chem.Mol:
        if ter2 is None:
            ter2 = ter1
        n = len(mols)
        mols = [*mols, ter1, ter2]
        res_name = [*res_name, 'TU0', 'TU1']
        m_idx = [n, *m_idx, n+1]
        chi_inv = [False, *chi_inv, False]
        if label is not None:
            label = [*label, [label_ter1, label_ter1], [label_ter2, label_ter2]]

    poly = random_walk_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId, dist_min=dist_min,
        retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step,
        tacticity=tacticity, res_name=res_name, label=label, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, mp_idx=mp_idx, restart=restart
    )

    if type(ter1) is Chem.Mol:
        set_terminal_idx(poly)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.random_copolymerize_rw. Elapsed time = %s' % str(dt2-dt1), level=1)
    # ------------------------------------------------------------------
    # yadonpy: Polymer naming hygiene
    #
    # In multi-step workflows (e.g., Example 01), the resulting polymer Mol
    # can inherit the monomer's RDKit _Name. Downstream, amorphous_cell stores
    # species names into cell metadata and system export uses those names to
    # name .itp/.gro and [ molecules ] entries. If the polymer keeps a monomer
    # name (e.g., "monomer_A"), the exported system will incorrectly produce
    # monomer_A.itp for the polymer.
    #
    # Default behavior: if the polymer name matches any input monomer name,
    # or no explicit polymer name is set, rename it to "polymer".
    # Users can override by explicitly setting poly.SetProp('_Name', ...) or
    # poly.SetProp('_yadonpy_resname', ...) after polymerization.
    # ------------------------------------------------------------------
    try:
        _mon_names = set()
        try:
            for _m in mols:
                if hasattr(_m, 'HasProp') and _m.HasProp('_Name'):
                    _mon_names.add(str(_m.GetProp('_Name')))
                if hasattr(_m, 'HasProp') and _m.HasProp('_yadonpy_resname'):
                    _mon_names.add(str(_m.GetProp('_yadonpy_resname')))
        except Exception:
            _mon_names = set()
        _cur = None
        try:
            if hasattr(poly, 'HasProp') and poly.HasProp('_Name'):
                _cur = str(poly.GetProp('_Name'))
        except Exception:
            _cur = None
        _has_explicit = False
        try:
            _has_explicit = (hasattr(poly, 'HasProp') and (poly.HasProp('_yadonpy_resname') or poly.HasProp('_Name')))
        except Exception:
            _has_explicit = False
        if (_cur is None) or (_cur in _mon_names) or (not _has_explicit):
            try:
                poly.SetProp('_Name', 'polymer')
            except Exception:
                pass
            try:
                poly.SetProp('_yadonpy_resname', 'polymer')
            except Exception:
                pass
    except Exception:
        pass



    try:
        resolved_name = str(name).strip() if name is not None else ""
        if not resolved_name:
            resolved_name = utils.suggest_name_from_work_dir(work_dir) or ""
        if resolved_name:
            utils.ensure_name(poly, name=str(resolved_name), depth=1, prefer_var=False)
    except Exception:
        pass




    _rw_save(work_dir, 'random_copolymerize_rw', payload, poly)
    return poly


_SEGMENT_META_PROP = "_yadonpy_segment_metadata_json"
_BRANCH_META_PROP = "_yadonpy_branch_metadata_json"
_SEGMENT_CHARGE_PROPS = ("AtomicCharge", "_GasteigerCharge", "RESP", "RESP2", "ESP", "MullikenCharge", "LowdinCharge")


def _segment_resolve_mol(mol, *, name: str | None = None):
    if isinstance(mol, str):
        return utils.mol_from_smiles(mol, name=name)
    return _resolve_mol_like(mol)


def _segment_smiles(mol) -> str | None:
    try:
        if hasattr(mol, 'HasProp') and mol.HasProp('_yadonpy_input_smiles'):
            return str(mol.GetProp('_yadonpy_input_smiles'))
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def _segment_linker_indices(mol, label: int = 1) -> list[int]:
    out: list[int] = []
    isotope = int(label) + 2
    for atom in mol.GetAtoms():
        try:
            if atom.GetSymbol() == "H" and int(atom.GetIsotope()) == isotope:
                out.append(int(atom.GetIdx()))
            elif int(label) == 1 and atom.GetSymbol() == "*":
                out.append(int(atom.GetIdx()))
        except Exception:
            continue
    return out


def _segment_branch_labels(mol, *, main_label: int = 1) -> list[int]:
    labels: set[int] = set()
    for atom in mol.GetAtoms():
        try:
            if atom.GetSymbol() == "H" and int(atom.GetIsotope()) >= 3:
                label = int(atom.GetIsotope()) - 2
                if label != int(main_label):
                    labels.add(label)
        except Exception:
            continue
    return sorted(labels)


def _segment_validate_linker_count(mol, *, label: int, allowed: tuple[int, ...], role: str) -> list[int]:
    idxs = _segment_linker_indices(mol, label=label)
    if len(idxs) not in set(int(x) for x in allowed):
        raise ValueError(
            f"{role} must contain {allowed} linker(s) for label={label}; found {len(idxs)}. "
            "Use '*'/'[1*]' for main-chain ends and '[2*]'/'[3*]' for branch sites."
        )
    return idxs


def _segment_charge_props(*mols) -> list[str]:
    props: list[str] = []
    for prop in _SEGMENT_CHARGE_PROPS:
        for mol in mols:
            try:
                if mol is not None and any(atom.HasProp(prop) for atom in mol.GetAtoms()):
                    props.append(prop)
                    break
            except Exception:
                continue
    return props


def _segment_harmonize_charge_props(*mols) -> None:
    props = _segment_charge_props(*mols)
    if not props:
        return
    for mol in mols:
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            for prop in props:
                if not atom.HasProp(prop):
                    atom.SetDoubleProp(prop, 0.0)


def _segment_total_charge(mol, prop: str = "AtomicCharge") -> float | None:
    try:
        if not any(atom.HasProp(prop) for atom in mol.GetAtoms()):
            return None
        return float(sum(float(atom.GetDoubleProp(prop)) for atom in mol.GetAtoms() if atom.HasProp(prop)))
    except Exception:
        return None


def _segment_write_json_prop(mol, key: str, payload: dict) -> None:
    try:
        mol.SetProp(key, json.dumps(_rw_normalize_value(payload), ensure_ascii=False, sort_keys=True))
    except Exception:
        pass


def _segment_set_name(mol, name: str | None, *, work_dir=None) -> None:
    try:
        resolved = str(name).strip() if name is not None else ""
        if not resolved:
            resolved = utils.suggest_name_from_work_dir(work_dir) or ""
        if resolved:
            utils.ensure_name(mol, name=resolved, depth=1, prefer_var=False)
    except Exception:
        pass


def _segment_cap_end(segment, cap, *, side: str, label: int = 1, confId: int = 0, **kwargs):
    if cap is None:
        return segment
    cap_mol = _segment_resolve_mol(cap)
    if cap_mol is None:
        raise ValueError(f"Invalid segment cap for {side!r}: {cap!r}")
    segment_c = utils.deepcopy_mol(segment)
    cap_c = utils.deepcopy_mol(cap_mol)
    _segment_harmonize_charge_props(segment_c, cap_c)
    if side == "tail":
        out = connect_mols(
            segment_c,
            cap_c,
            label1=label,
            label2=label,
            confId1=confId,
            confId2=confId,
            res_name_1="SEG",
            res_name_2="CAP",
            **kwargs,
        )
    elif side == "head":
        out = connect_mols(
            cap_c,
            segment_c,
            label1=label,
            label2=label,
            confId1=confId,
            confId2=confId,
            res_name_1="CAP",
            res_name_2="SEG",
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported cap side: {side!r}")
    return _rw_finalize_bonded_terms(out)


def seg_gen(
    units,
    *,
    name: str | None = None,
    label: int = 1,
    cap_head=None,
    cap_tail=None,
    confId: int = 0,
    dist_min: float = 0.7,
    retry: int = 60,
    rollback: int = 3,
    rollback_shaking: bool = False,
    retry_step: int = 80,
    retry_opt_step: int = 4,
    res_name=None,
    ff=None,
    work_dir=None,
    omp: int = 0,
    mpi: int = 1,
    gpu: int = 0,
    restart=None,
    **kwargs,
):
    """Generate a reusable polymer segment from pre-parameterized units.

    The main-chain connection label defaults to ``1`` (``*`` or ``[1*]``).
    Higher labels such as ``[2*]`` are preserved for later branch attachment.
    Existing per-atom charge properties are carried through the connection.
    """

    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.seg_gen.', level=1)
    if isinstance(units, (Chem.Mol, MolSpec, str)):
        units = [units]
    units_resolved = [_segment_resolve_mol(unit) for unit in list(units)]
    if any(unit is None for unit in units_resolved):
        raise ValueError("seg_gen received an unresolved unit.")
    for idx, unit in enumerate(units_resolved):
        _segment_validate_linker_count(unit, label=label, allowed=(2,), role=f"seg_gen unit {idx}")

    payload = _rw_payload(
        'seg_gen',
        unit_smiles=[_segment_smiles(unit) for unit in units_resolved],
        label=int(label),
        cap_head=_segment_smiles(_segment_resolve_mol(cap_head)) if cap_head is not None else None,
        cap_tail=_segment_smiles(_segment_resolve_mol(cap_tail)) if cap_tail is not None else None,
        name=name,
    )
    rst_flag = _effective_restart_flag(work_dir, restart)
    if rst_flag:
        cached = _rw_load(work_dir, 'seg_gen', payload)
        if cached is not None:
            return cached

    if len(units_resolved) == 1:
        seg = utils.deepcopy_mol(units_resolved[0])
    else:
        m_idx = list(range(len(units_resolved)))
        chi_inv = [False for _ in m_idx]
        if res_name is None:
            res_name = ['SG%s' % const.pdb_id[i] for i in range(len(units_resolved))]
        seg = random_walk_polymerization(
            units_resolved,
            m_idx,
            chi_inv,
            confId=confId,
            dist_min=dist_min,
            retry=retry,
            rollback=rollback,
            rollback_shaking=rollback_shaking,
            retry_step=retry_step,
            retry_opt_step=retry_opt_step,
            tacticity=None,
            res_name=res_name,
            label=[[int(label), int(label)] for _ in units_resolved],
            ff=ff,
            work_dir=work_dir,
            omp=omp,
            mpi=mpi,
            gpu=gpu,
            restart=restart,
        )

    if cap_head is not None:
        seg = _segment_cap_end(seg, cap_head, side="head", label=label, confId=confId, **kwargs)
    if cap_tail is not None:
        seg = _segment_cap_end(seg, cap_tail, side="tail", label=label, confId=confId, **kwargs)

    _segment_set_name(seg, name, work_dir=work_dir)
    meta = {
        "kind": "segment",
        "unit_count": len(units_resolved),
        "unit_smiles": [_segment_smiles(unit) for unit in units_resolved],
        "main_label": int(label),
        "main_linker_count": len(_segment_linker_indices(seg, label=label)),
        "branch_labels": _segment_branch_labels(seg, main_label=label),
        "cap_head": payload.get("cap_head"),
        "cap_tail": payload.get("cap_tail"),
        "name": name,
        "net_charge_atomic": _segment_total_charge(seg, "AtomicCharge"),
    }
    _segment_write_json_prop(seg, _SEGMENT_META_PROP, meta)
    _rw_save(work_dir, 'seg_gen', payload, seg)
    utils.radon_print('Normal termination of poly.seg_gen. Elapsed time = %s' % str(datetime.datetime.now() - dt1), level=1)
    return seg


def block_segment_rw(
    segments,
    block_lengths,
    *,
    name: str | None = None,
    label: int = 1,
    confId: int = 0,
    dist_min: float = 0.7,
    retry: int = 60,
    rollback: int = 3,
    rollback_shaking: bool = False,
    retry_step: int = 80,
    retry_opt_step: int = 4,
    res_name=None,
    ff=None,
    work_dir=None,
    omp: int = 0,
    mpi: int = 1,
    gpu: int = 0,
    restart=None,
):
    """Build a block polymer by treating each segment as a pseudo-monomer."""

    segments_resolved = [_segment_resolve_mol(seg) for seg in list(segments)]
    lengths = [int(x) for x in list(block_lengths)]
    if len(segments_resolved) != len(lengths):
        raise ValueError(f"segments/block_lengths mismatch: {len(segments_resolved)} vs {len(lengths)}")
    if any(n <= 0 for n in lengths):
        raise ValueError("All block_lengths must be positive integers.")
    for idx, seg in enumerate(segments_resolved):
        _segment_validate_linker_count(seg, label=label, allowed=(2,), role=f"block segment {idx}")
    m_idx: list[int] = []
    for idx, length in enumerate(lengths):
        m_idx.extend([idx for _ in range(int(length))])
    chi_inv = [False for _ in m_idx]

    payload = _rw_payload(
        'block_segment_rw',
        segment_smiles=[_segment_smiles(seg) for seg in segments_resolved],
        block_lengths=lengths,
        label=int(label),
        name=name,
    )
    rst_flag = _effective_restart_flag(work_dir, restart)
    if rst_flag:
        cached = _rw_load(work_dir, 'block_segment_rw', payload)
        if cached is not None:
            return cached

    if res_name is None:
        res_name = ['BK%s' % const.pdb_id[i] for i in range(len(segments_resolved))]
    out = random_walk_polymerization(
        segments_resolved,
        m_idx,
        chi_inv,
        confId=confId,
        dist_min=dist_min,
        retry=retry,
        rollback=rollback,
        rollback_shaking=rollback_shaking,
        retry_step=retry_step,
        retry_opt_step=retry_opt_step,
        tacticity=None,
        res_name=res_name,
        label=[[int(label), int(label)] for _ in segments_resolved],
        ff=ff,
        work_dir=work_dir,
        omp=omp,
        mpi=mpi,
        gpu=gpu,
        restart=restart,
    )
    _segment_set_name(out, name, work_dir=work_dir)
    _segment_write_json_prop(
        out,
        _SEGMENT_META_PROP,
        {
            "kind": "block_segment_polymer",
            "segment_count": len(segments_resolved),
            "block_lengths": lengths,
            "main_label": int(label),
            "branch_labels": _segment_branch_labels(out, main_label=label),
            "name": name,
        },
    )
    _rw_save(work_dir, 'block_segment_rw', payload, out)
    return out


def _branch_normalize_branches(branches):
    if isinstance(branches, (Chem.Mol, MolSpec, str)):
        branches = [branches]
    out = [_segment_resolve_mol(branch) for branch in list(branches)]
    if any(branch is None for branch in out):
        raise ValueError("branch_segment_rw received an unresolved branch.")
    return out


def _branch_prepare_fragment(branch, *, branch_terminator, label: int, confId: int, **kwargs):
    branch_c = utils.deepcopy_mol(branch)
    n_linkers = len(_segment_linker_indices(branch_c, label=label))
    if n_linkers == 1:
        return branch_c
    if n_linkers == 2:
        if branch_terminator is None:
            raise ValueError("Two-linker branch fragments require branch_terminator or prior cap_tail.")
        return _segment_cap_end(branch_c, branch_terminator, side="tail", label=label, confId=confId, **kwargs)
    raise ValueError(f"Branch fragment must have one or two label={label} linkers; found {n_linkers}.")


def _branch_normalize_positions(position) -> list[int]:
    if isinstance(position, (list, tuple, np.ndarray)):
        return [int(x) for x in position]
    return [int(position)]


def _branch_normalize_ds(ds, *, n_branches: int, n_positions: int) -> np.ndarray:
    if ds is None:
        arr = np.zeros((n_branches, n_positions), dtype=float)
        arr[0, :] = 1.0
        return arr
    arr = np.asarray(ds, dtype=float)
    if arr.ndim == 0:
        arr = np.full((n_branches, n_positions), float(arr))
    elif arr.ndim == 1:
        if n_branches == 1 and arr.size == n_positions:
            arr = arr.reshape(1, n_positions)
        elif n_positions == 1 and arr.size == n_branches:
            arr = arr.reshape(n_branches, 1)
        elif arr.size == n_branches:
            arr = np.tile(arr.reshape(n_branches, 1), (1, n_positions))
        else:
            raise ValueError("Cannot interpret ds shape for branches/positions.")
    elif arr.ndim == 2:
        if arr.shape != (n_branches, n_positions):
            raise ValueError(f"ds must have shape {(n_branches, n_positions)}, got {arr.shape}.")
    else:
        raise ValueError("ds must be scalar, 1D, or 2D.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("ds values must be between 0 and 1.")
    if np.any(np.sum(arr, axis=0) > 1.0 + 1.0e-12):
        raise ValueError("For each branch position, sum(ds[:, position]) must be <= 1.")
    return arr


def _branch_select_sites(base, branches, *, position, ds, exact_map, random_seed=None) -> list[dict]:
    positions = _branch_normalize_positions(position)
    sites_by_pos = {pos: _segment_linker_indices(base, label=pos) for pos in positions}
    rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng()
    selected: list[dict] = []
    if exact_map is not None:
        entries = [exact_map] if isinstance(exact_map, dict) else list(exact_map)
        for entry in entries:
            pos = int(entry.get("position", positions[0]))
            sites = sorted(sites_by_pos.get(pos) or _segment_linker_indices(base, label=pos))
            if "atom_idx" in entry:
                atom_idx = int(entry["atom_idx"])
                if atom_idx not in sites:
                    raise ValueError(f"atom_idx={atom_idx} is not a branch marker for position={pos}.")
            else:
                site_index = int(entry.get("site_index", 0))
                if site_index < 0 or site_index >= len(sites):
                    raise ValueError(f"site_index={site_index} out of range for position={pos}; n_sites={len(sites)}.")
                atom_idx = int(sites[site_index])
            branch_idx = int(entry.get("branch", entry.get("branch_index", 0)))
            if branch_idx < 0 or branch_idx >= len(branches):
                raise ValueError(f"branch index out of range: {branch_idx}")
            selected.append({"position": pos, "atom_idx": atom_idx, "branch": branch_idx, "source": "exact_map"})
    else:
        ds_arr = _branch_normalize_ds(ds, n_branches=len(branches), n_positions=len(positions))
        for pos_j, pos in enumerate(positions):
            sites = list(sorted(sites_by_pos.get(pos, [])))
            if not sites:
                continue
            rng.shuffle(sites)
            cursor = 0
            for branch_idx in range(len(branches)):
                n_take = int(round(len(sites) * float(ds_arr[branch_idx, pos_j])))
                for atom_idx in sites[cursor:cursor + n_take]:
                    selected.append({"position": pos, "atom_idx": int(atom_idx), "branch": branch_idx, "source": "ds"})
                cursor += n_take
    seen: set[int] = set()
    for item in selected:
        atom_idx = int(item["atom_idx"])
        if atom_idx in seen:
            raise ValueError(f"Duplicate branch attachment site selected: atom_idx={atom_idx}")
        seen.add(atom_idx)
    selected.sort(key=lambda item: int(item["atom_idx"]), reverse=True)
    return selected


def _branch_set_single_site_linker(mol, *, atom_idx: int) -> None:
    atom_idx = int(atom_idx)
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = list(atom.GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError(f"Branch marker atom {atom_idx} must have exactly one neighbor.")
    ne_idx = int(neighbors[0].GetIdx())
    for at in mol.GetAtoms():
        at.SetBoolProp('linker', False)
        at.SetBoolProp('head', False)
        at.SetBoolProp('tail', False)
        at.SetBoolProp('head_neighbor', False)
        at.SetBoolProp('tail_neighbor', False)
    atom.SetBoolProp('linker', True)
    atom.SetBoolProp('head', True)
    atom.SetBoolProp('tail', True)
    mol.GetAtomWithIdx(ne_idx).SetBoolProp('head_neighbor', True)
    mol.GetAtomWithIdx(ne_idx).SetBoolProp('tail_neighbor', True)
    mol.SetIntProp('head_idx', atom_idx)
    mol.SetIntProp('tail_idx', atom_idx)
    mol.SetIntProp('head_ne_idx', ne_idx)
    mol.SetIntProp('tail_ne_idx', ne_idx)


def _branch_connect_one_site(
    base,
    branch,
    *,
    atom_idx: int,
    branch_label: int,
    confId: int,
    dist_min: float,
    retry_step: int,
    retry_opt_step: int,
    res_name_base: str,
    res_name_branch: str,
    **kwargs,
):
    last = None
    for _ in range(max(int(retry_step) * (1 + int(retry_opt_step)), 1)):
        base_c = utils.deepcopy_mol(base)
        branch_c = utils.deepcopy_mol(branch)
        _segment_harmonize_charge_props(base_c, branch_c)
        _branch_set_single_site_linker(base_c, atom_idx=atom_idx)
        if not set_linker_flag(branch_c, label=branch_label):
            raise ValueError(f"Branch fragment does not have a usable label={branch_label} attach linker.")
        trial = _prepare_connect_trial(
            base_c,
            branch_c,
            random_rot=True,
            set_linker=False,
            confId1=confId,
            confId2=confId,
            **kwargs,
        )
        if trial is None:
            last = None
            continue
        ok = check_3d_structure_connect_trial(base_c, branch_c, trial, dist_min=dist_min)
        if not ok:
            last = None
            continue
        out = _materialize_connected_mols(base_c, branch_c, trial, res_name_1=res_name_base, res_name_2=res_name_branch)
        try:
            set_linker_flag(out, label=1)
        except Exception:
            pass
        return _rw_finalize_bonded_terms(out)
    raise RuntimeError(f"Could not attach branch at atom_idx={atom_idx} after retry_step={retry_step}.")


def branch_segment_rw(
    base,
    branches,
    *,
    position=2,
    ds=None,
    exact_map=None,
    branch_terminator="[H][*]",
    mode: str = "post",
    branch_label: int = 1,
    name: str | None = None,
    confId: int = 0,
    dist_min: float = 0.7,
    retry_step: int = 80,
    retry_opt_step: int = 4,
    random_seed=None,
    work_dir=None,
    restart=None,
    **kwargs,
):
    """Attach prebuilt branch fragments to labelled branch sites on a segment/polymer."""

    mode_norm = str(mode or "post").strip().lower()
    if mode_norm not in {"pre", "post"}:
        raise ValueError("branch_segment_rw mode must be 'pre' or 'post'.")
    base_mol = _segment_resolve_mol(base)
    if base_mol is None:
        raise ValueError("branch_segment_rw received an unresolved base molecule.")
    branch_mols = _branch_normalize_branches(branches)
    prepared_branches = [
        _branch_prepare_fragment(
            branch,
            branch_terminator=branch_terminator,
            label=branch_label,
            confId=confId,
            **kwargs,
        )
        for branch in branch_mols
    ]

    payload = _rw_payload(
        'branch_segment_rw',
        base_smiles=_segment_smiles(base_mol),
        branch_smiles=[_segment_smiles(branch) for branch in branch_mols],
        position=position,
        ds=np.asarray(ds).tolist() if ds is not None else None,
        exact_map=exact_map,
        branch_terminator=_segment_smiles(_segment_resolve_mol(branch_terminator)) if branch_terminator is not None else None,
        mode=mode_norm,
        name=name,
    )
    rst_flag = _effective_restart_flag(work_dir, restart)
    if rst_flag:
        cached = _rw_load(work_dir, 'branch_segment_rw', payload)
        if cached is not None:
            return cached

    selected = _branch_select_sites(
        base_mol,
        prepared_branches,
        position=position,
        ds=ds,
        exact_map=exact_map,
        random_seed=random_seed,
    )
    out = utils.deepcopy_mol(base_mol)
    for attach in selected:
        out = _branch_connect_one_site(
            out,
            prepared_branches[int(attach["branch"])],
            atom_idx=int(attach["atom_idx"]),
            branch_label=int(branch_label),
            confId=confId,
            dist_min=dist_min,
            retry_step=retry_step,
            retry_opt_step=retry_opt_step,
            res_name_base="BAS",
            res_name_branch="BRN",
            **kwargs,
        )
    _segment_set_name(out, name, work_dir=work_dir)
    meta = {
        "kind": "branched_segment" if mode_norm == "pre" else "branched_polymer",
        "mode": mode_norm,
        "position": _branch_normalize_positions(position),
        "ds": np.asarray(ds).tolist() if ds is not None else None,
        "exact_map": exact_map,
        "selected_site_count": len(selected),
        "selected_sites": selected,
        "branch_smiles": [_segment_smiles(branch) for branch in branch_mols],
        "branch_terminator": payload.get("branch_terminator"),
        "main_linker_count": len(_segment_linker_indices(out, label=1)),
        "remaining_branch_labels": _segment_branch_labels(out, main_label=1),
        "net_charge_atomic": _segment_total_charge(out, "AtomicCharge"),
    }
    _segment_write_json_prop(out, _BRANCH_META_PROP, meta)
    if mode_norm == "pre":
        _segment_write_json_prop(out, _SEGMENT_META_PROP, {**meta, "kind": "prebranched_segment", "main_label": 1})
    _rw_save(work_dir, 'branch_segment_rw', payload, out)
    return out


def random_copolymerize_rw_old(mols, n, ratio=None, reac_ratio=[], init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=10, rollback=5, rollback_shaking=False, retry_step=0, retry_opt_step=50, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    # Backward campatibility
    return random_copolymerize_rw(mols, n, ratio=ratio, reac_ratio=reac_ratio, init_poly=init_poly, headhead=headhead, confId=confId,
            tacticity=tacticity, atac_ratio=atac_ratio, dist_min=dist_min, retry=retry, rollback=rollback, rollback_shaking=rollback_shaking,
            retry_step=retry_step, retry_opt_step=retry_opt_step, ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2,
            res_name=res_name, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)


def random_copolymerize_rw_mp(mols, n, ratio=None, reac_ratio=[], init_poly=None, tacticity='atactic', atac_ratio=0.5,
                dist_min=0.7, retry=60, rollback=3, rollback_shaking=False, retry_step=80, retry_opt_step=4, ter1=None, ter2=None,
                label=None, label_ter1=1, label_ter2=1, res_name=None,
                ff=None, work_dir=None, omp=1, mpi=1, gpu=0, nchain=1, mp=None, fail_copy=True):

    for i in range(len(mols)):
        utils.picklable(mols[i])
    if type(ter1) is Chem.Mol:
        utils.picklable(ter1)
        if type(ter2) is Chem.Mol:
            utils.picklable(ter2)
        else:
            ter2 = ter1

    if mp is None:
        mp = utils.cpu_count()
    np = max([nchain, mp])

    c = utils.picklable_const()
    args = [(mols, n, ratio, reac_ratio, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, rollback_shaking, retry_step, retry_opt_step,
            ter1, ter2, label, label_ter1, label_ter2, res_name, ff, work_dir, omp, mpi, gpu, mp_idx, c) for mp_idx in range(np)]

    polys = polymerize_mp_exec(_random_copolymerize_rw_mp_worker, args, mp, nchain=nchain, fail_copy=fail_copy)

    return polys


def _random_copolymerize_rw_mp_worker(args):
    (mols, n, ratio, reac_ratio, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, rollback_shaking, retry_step, retry_opt_step,
        ter1, ter2, label, label_ter1, label_ter2, res_name, ff, work_dir, omp, mpi, gpu, mp_idx, c) = args
    utils.restore_const(c)

    try:
        poly = random_copolymerize_rw(mols, n, ratio=ratio, reac_ratio=reac_ratio, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio,
                    dist_min=dist_min, retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step,
                    ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name,
                    ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, mp_idx=mp_idx)
        utils.picklable(poly)
    except BaseException as e:
        utils.radon_print('%s' % e)
        poly = None

    return poly


def block_copolymerize_rw(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=60, rollback=3, rollback_shaking=False, retry_step=80, retry_opt_step=4, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, ff=None, work_dir=None, omp=0, mpi=1, gpu=0, mp_idx=None, restart=None):
    """
    poly.block_copolymerize_rw

    Block copolymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mols: List of RDkit Mol object
        n: List of polymerization degree (list, int)

    Optional args:
        init_poly: polymerize_rw perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry polymerize_rw (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        work_dir: Work directory path of external minimizer (str)
        ff: Force field object (requiring when opt is external minimizer)
        omp: Number of threads of OpenMP in external minimizer (int)
        mpi: Number of MPI process in external minimizer (int)
        gpu: Number of GPU in external minimizer (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.block_copolymerize_rw.', level=1)

    mols = _resolve_mol_list(mols)
    init_poly = _resolve_mol_like(init_poly) if init_poly is not None else None
    ter1 = _resolve_mol_like(ter1) if ter1 is not None else None
    ter2 = _resolve_mol_like(ter2) if ter2 is not None else None
    rst_flag = _effective_restart_flag(work_dir, restart)
    payload = _rw_payload('block_copolymerize_rw', smiles=[Chem.MolToSmiles(m, isomericSmiles=True) for m in mols], n=list(np.asarray(n).tolist()) if isinstance(n, list) else int(n), headhead=bool(headhead), tacticity=tacticity, atac_ratio=float(atac_ratio), label=label, ter1_smiles=(Chem.MolToSmiles(ter1, isomericSmiles=True) if isinstance(ter1, Chem.Mol) else None), ter2_smiles=(Chem.MolToSmiles(ter2, isomericSmiles=True) if isinstance(ter2, Chem.Mol) else None), init_poly_smiles=(Chem.MolToSmiles(init_poly, isomericSmiles=True) if isinstance(init_poly, Chem.Mol) else None))
    if rst_flag:
        cached = _rw_load(work_dir, 'block_copolymerize_rw', payload)
        if cached is not None:
            return cached

    if len(mols) != len(n):
        utils.radon_print('Inconsistency length of mols and n', level=3)

    m_idx = gen_monomer_array(len(mols), n, copoly='block')
    chi_inv, check_chi = gen_chiral_inv_array(mols, m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    if not check_chi:
        tacticity = None

    if res_name is None:
        res_name = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]
        
    if type(ter1) is Chem.Mol:
        if ter2 is None:
            ter2 = ter1
        n = len(mols)
        mols = [*mols, ter1, ter2]
        res_name = [*res_name, 'TU0', 'TU1']
        m_idx = [n, *m_idx, n+1]
        chi_inv = [False, *chi_inv, False]
        if label is not None:
            label = [*label, [label_ter1, label_ter1], [label_ter2, label_ter2]]

    poly = random_walk_polymerization(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        dist_min=dist_min, retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step,
        tacticity=tacticity, res_name=res_name, label=label, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, mp_idx=mp_idx, restart=restart
    )

    if type(ter1) is Chem.Mol:
        set_terminal_idx(poly)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.block_copolymerize_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    _rw_save(work_dir, 'block_copolymerize_rw', payload, poly)
    return poly


def block_copolymerize_rw_old(mols, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=10, rollback=5, rollback_shaking=False, retry_step=0, retry_opt_step=50, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, ff=None, work_dir=None, omp=0, mpi=1, gpu=0):
    # Backward campatibility
    return block_copolymerize_rw(mols, n, init_poly=init_poly, headhead=headhead, confId=confId,
            tacticity=tacticity, atac_ratio=atac_ratio, dist_min=dist_min, retry=retry, rollback=rollback, rollback_shaking=rollback_shaking,
            retry_step=retry_step, retry_opt_step=retry_opt_step, ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2,
            res_name=res_name, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)


def block_copolymerize_rw_mp(mols, n, init_poly=None, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=60, rollback=3, rollback_shaking=False, retry_step=80, retry_opt_step=4, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name=None, 
            ff=None, work_dir=None, omp=0, mpi=1, gpu=0, nchain=10, mp=10, fail_copy=True):

    for i in range(len(mols)):
        utils.picklable(mols[i])
    if type(ter1) is Chem.Mol:
        utils.picklable(ter1)
        if type(ter2) is Chem.Mol:
            utils.picklable(ter2)
        else:
            ter2 = ter1

    if mp is None:
        mp = utils.cpu_count()
    np = max([nchain, mp])

    c = utils.picklable_const()
    args = [(mols, n, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, rollback_shaking, retry_step, retry_opt_step,
            ter1, ter2, label, label_ter1, label_ter2, res_name, ff, work_dir, omp, mpi, gpu, mp_idx, c) for mp_idx in range(np)]

    polys = polymerize_mp_exec(_block_copolymerize_rw_mp_worker, args, mp, nchain=nchain, fail_copy=fail_copy)

    return polys


def _block_copolymerize_rw_mp_worker(args):
    (mols, n, init_poly, tacticity, atac_ratio, dist_min, retry, rollback, rollback_shaking, retry_step, retry_opt_step,
        ter1, ter2, label, label_ter1, label_ter2, res_name, ff, work_dir, omp, mpi, gpu, mp_idx, c) = args
    utils.restore_const(c)

    try:
        poly = block_copolymerize_rw(mols, n, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio, dist_min=dist_min, 
                    retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step,
                    ter1=ter1, ter2=ter2, label=label, label_ter1=label_ter1, label_ter2=label_ter2, res_name=res_name,
                    ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, mp_idx=mp_idx)
        utils.picklable(poly)
    except BaseException as e:
        utils.radon_print('%s' % e)
        poly = None

    return poly


def polymerize_mp_exec(func, args, mp, nchain=1, fail_copy=True):
    executor = confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn'))
    futures = [executor.submit(func, arg) for arg in args]

    results = []
    success = 0
    for future in confu.as_completed(futures):
        res = future.result()
        results.append(res)
        if type(res) is Chem.Mol:
            success += 1
        if success >= nchain:
            break

    for future in futures:
        if not future.running():
            future.cancel()

    for process in executor._processes.values():
        process.kill()

    executor.shutdown(wait=False)

    polys = [res for res in results if type(res) is Chem.Mol]

    if len(polys) == 0:
        utils.radon_print('Generation of polymer chain by random walk was failure.', level=3)
    elif len(polys) < nchain:
        if fail_copy:
            for i in range(nchain-len(polys)):
                r_idx = np.random.choice(range(len(polys)))
                polys.append(utils.deepcopy_mol(polys[r_idx]))
            utils.radon_print('%i success, %i copy' % (len(polys), (nchain-len(polys))))
        else:
            for i in range(nchain-len(polys)):
                polys.append(None)
            utils.radon_print('%i success, %i fail' % (len(polys), (nchain-len(polys))))
    elif len(polys) > nchain:
        polys = polys[:nchain]

    return polys


def terminate_rw(poly, mol1, mol2=None, confId=0, dist_min=1.0, retry=100, rollback_shaking=False, retry_step=200, retry_opt_step=0,
            res_name='RU0', label=None, ff=None, work_dir=None, omp=0, mpi=1, gpu=0, name: str | None = None, restart=None):
    """
    poly.terminate_rw

    Termination of polymer of RDkit Mol object by random walk

    Args:
        poly: RDkit Mol object of a polymer
        mol1: RDkit Mol object of a terminal unit (head side or both sides)

    Optional args:
        mol2: RDkit Mol object of a terminal unit (tail side)
        confId: Target conformer ID
        dist_min: (float, angstrom)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        work_dir: Work directory path of external minimizer (str, requiring when opt is external minimizer)
        ff: Force field object (requiring when opt is external minimizer)
        omp: Number of threads of OpenMP in external minimizer (int)
        mpi: Number of MPI process in external minimizer (int)
        gpu: Number of GPU in external minimizer (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.terminate_rw.', level=1)

    poly = _resolve_mol_like(poly)
    mol1 = _resolve_mol_like(mol1)
    mol2 = _resolve_mol_like(mol2) if mol2 is not None else None
    rst_flag = _effective_restart_flag(work_dir, restart)
    payload = _rw_payload('terminate_rw', poly_smiles=(Chem.MolToSmiles(poly, isomericSmiles=True) if isinstance(poly, Chem.Mol) else None), mol1_smiles=(Chem.MolToSmiles(mol1, isomericSmiles=True) if isinstance(mol1, Chem.Mol) else None), mol2_smiles=(Chem.MolToSmiles(mol2, isomericSmiles=True) if isinstance(mol2, Chem.Mol) else None), res_name=res_name, label=label, name=name)
    if rst_flag:
        cached = _rw_load(work_dir, 'terminate_rw', payload)
        if cached is not None:
            return cached

    if mol2 is None:
        mol2 = mol1
    H2_flag1 = False
    H2_flag2 = False
    res_name_1 = 'TU0'
    res_name_2 = 'TU1'
    poly_c = utils.deepcopy_mol(poly)
    _ensure_mol_residue_info(poly_c)

    if Chem.MolToSmiles(mol1) == '[H][3H]' or Chem.MolToSmiles(mol1) == '[3H][H]':
        head_idx = poly_c.GetIntProp('head_idx')
        residue_number = 1 + poly_c.GetIntProp('num_units')
        _set_atom_residue(poly_c, head_idx, res_name_1, residue_number).SetIsotope(0)
        poly_c.SetIntProp('num_units', 1+poly_c.GetIntProp('num_units'))
        H2_flag1 = True
    if Chem.MolToSmiles(mol2) == '[H][3H]' or Chem.MolToSmiles(mol2) == '[3H][H]':
        tail_idx = poly_c.GetIntProp('tail_idx')
        residue_number = 1 + poly_c.GetIntProp('num_units')
        _set_atom_residue(poly_c, tail_idx, res_name_2, residue_number).SetIsotope(0)
        poly_c.SetIntProp('num_units', 1+poly_c.GetIntProp('num_units'))
        H2_flag2 = True

    mols = [poly_c, mol1, mol2]
    res_name = [res_name, res_name_1, res_name_2]

    if H2_flag1 and H2_flag2:
        pass
    elif H2_flag2:
        mon_idx = [1, 0]
        chi_inv = [False, False]
    elif H2_flag1:
        mon_idx = [0, 2]
        chi_inv = [False, False]
    else:
        mon_idx = [1, 0, 2]
        chi_inv = [False, False, False]

    if not H2_flag1 or not H2_flag2:
        poly_c = random_walk_polymerization(
            mols, mon_idx, chi_inv, confId=confId,
            dist_min=dist_min, retry=retry, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step,
            res_name=res_name, label=label, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, restart=restart
        )

    set_terminal_idx(poly_c)
    poly_c = _rw_finalize_bonded_terms(poly_c)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.terminate_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    try:
        resolved_name = str(name).strip() if name is not None else ""
        if not resolved_name:
            resolved_name = utils.get_name(poly, default=None) or utils.suggest_name_from_work_dir(work_dir) or ""
        if resolved_name:
            utils.ensure_name(poly_c, name=str(resolved_name), depth=1, prefer_var=False)
    except Exception:
        pass


    _rw_save(work_dir, 'terminate_rw', payload, poly_c)
    return poly_c


def gen_monomer_array(n_mon, n, copoly='alt', ratio=[], reac_ratio=[]):
    """
    poly.gen_monomer_array

    Monomer array generator for the polymer chain generators
    """
    mon_array = []
    if type(n) is list and len(n) == 1:
        n = int(n[0])

    # Homopolymer
    if n_mon == 1:
        mon_array = np.zeros(n)

    # Alternating copolymer
    elif n_mon > 1 and copoly == 'alt':
        mon_array = np.tile(list(range(n_mon)), n)

    # Random copolymer
    elif n_mon > 1 and copoly == 'random':
        # Monomer reactivity ratio
        if len(reac_ratio) == 2 and n_mon == 2:
            p11 = reac_ratio[0]/(1+reac_ratio[0])
            p22 = reac_ratio[1]/(1+reac_ratio[1])

            if np.random.rand(1)[0] >= 0.5:
                pre = 0
                mon_array.append(0)
            else:
                pre = 1
                mon_array.append(1)

            for i in range(n-1):
                rand = np.random.rand(1)[0]
                if pre == 0:
                    if rand <= p11:
                        pre = 0
                        mon_array.append(0)
                    else:
                        pre = 1
                        mon_array.append(1)
                else:
                    if rand <= p22:
                        pre = 1
                        mon_array.append(1)
                    else:
                        pre = 0
                        mon_array.append(0)
            mon_array = np.array(mon_array)

        # Monomer composition ratio
        else:
            if len(ratio) != n_mon:
                utils.radon_print('Inconsistency length of mols and ratio', level=3)

            for i, r in enumerate(ratio):
                tmp = [i]*int(n*r+0.5)
                mon_array.extend(tmp)
            if len(mon_array) < n:
                d = int(n-len(mon_array))
                imax = np.argmax(ratio)
                tmp = [imax]*d
                mon_array.extend(tmp)
            elif len(mon_array) > n:
                d = int(len(mon_array)-n)
                imax = np.argmax(ratio)
                for i in range(d):
                    mon_array.remove(imax)
            mon_array = np.array(mon_array[:])
            np.random.shuffle(mon_array)

    # Block copolymer
    elif n_mon > 1 and copoly == 'block':
        if type(n) is int:
            n = np.full(n_mon, n)

        if len(n) != n_mon:
            utils.radon_print('Inconsistency length of mols and n', level=3)

        for i in range(n_mon):
            tmp = [i]*n[i]
            mon_array.extend(tmp)
        mon_array = np.array(mon_array[:])

    else:
        utils.radon_print('Illegal input for copoly=%s in poly.gen_monomer_array.' % copoly, level=3)

    mon_array = [int(x) for x in list(mon_array)]

    return mon_array


def gen_chiral_inv_array(mols, mon_array, init_poly=None, tacticity='atactic', atac_ratio=0.5):
    """
    poly.gen_chiral_inv_array

    Chiral inversion array generator for the polymer chain generators
    """
    mon_array = np.array(mon_array)
    chi_inv = np.full(len(mon_array), False)
    chiral_poly = []
    chiral_poly_last = 0
    chiral_mon = []
    n_mon = []
    check_chi = True

    if type(init_poly) is Chem.Mol:
        chiral_poly = get_chiral_list(init_poly)
        chiral_poly_last = chiral_poly[-1] if len(chiral_poly) > 0 else 0

    for i, mol in enumerate(mols):
        if i == 0:
            n_mon.append(len(mon_array) - np.count_nonzero(mon_array))
        else:
            n_mon.append(np.count_nonzero(mon_array == i))
        num_chiral = check_chiral_monomer(mol)
        if num_chiral == 0:
            chiral_mon.append(0)
        elif num_chiral == 1:
            chiral_mon.append(get_chiral_list(mol)[0])
        else:
            chiral_mon.append(0)
            if check_chi:
                utils.radon_print(
                    'Found multiple chiral center in the mainchain of polymer repeating unit. The chirality control is turned off.',
                    level=1)
                check_chi = False

    if tacticity == 'isotactic':
        flag = chiral_poly_last
        for i, mon_idx in enumerate(mon_array):
            if flag == 0 and chiral_mon[mon_idx] != 0:
                flag = chiral_mon[mon_idx]
            if chiral_mon[mon_idx] == 0:
                chi_inv[i] = False
            elif flag == chiral_mon[mon_idx]:
                chi_inv[i] = False
            else:
                chi_inv[i] = True

    elif tacticity == 'syndiotactic':
        flag = chiral_poly_last
        for i, mon_idx in enumerate(mon_array):
            if chiral_mon[mon_idx] == 0:
                chi_inv[i] = False
            elif flag == chiral_mon[mon_idx]:
                chi_inv[i] = True
                flag = 2 if chiral_mon[mon_idx] == 1 else 1
            else:
                chi_inv[i] = False
                flag = 1 if chiral_mon[mon_idx] == 1 else 2

    elif tacticity == 'atactic':
        chi_inv_list = []
        for n, c in zip(n_mon, chiral_mon):
            chi_mon_inv = np.full(n, False)
            if c == 0:
                chi_inv_list.append(list(chi_mon_inv))
            elif c == 1:
                n_inv = n*atac_ratio
                chi_mon_inv[int(n_inv):] = True
                np.random.shuffle(chi_mon_inv)
                chi_inv_list.append(list(chi_mon_inv))
            else:
                n_inv = n*(1-atac_ratio)
                chi_mon_inv[int(n_inv):] = True
                np.random.shuffle(chi_mon_inv)
                chi_inv_list.append(list(chi_mon_inv))

        for i, mon_idx in enumerate(mon_array):
            chi_inv[i] = chi_inv_list[mon_idx].pop()

    else:
        utils.radon_print('%s is illegal input for tacticity.' % str(tacticity), level=3)

    return chi_inv, check_chi


######################################################
# Unit cell generators
######################################################
def amorphous_cell(
        mols,
        n,
        cell=None,
        density=0.1,
        retry=20,
        retry_step=1000,
        threshold=2.0,
        dec_rate=0.8,
        check_bond_ring_intersection=False,
        mp=0,
        restart_flag=None,
        ions=None,
        neutralize=True,
        neutralize_tol=1e-4,
        charge_scale=None,
        polyelectrolyte_mode=False,
        charge_tolerance=1e-2,
        large_system_mode='auto',
        work_dir=None,
        restart=None,
):
    """
    poly.amorphous_cell

    Simple unit cell generator for amorphous system

    Args:
        mols: RDkit Mol object or its list
        n: Number of molecules in the unit cell (int) or its list

    Optional args:
        cell: Initial structure of unit cell (RDkit Mol object)
        density: (float, g/cm3)
        retry: Number of retry for this function when inter-molecular atom-atom distance is below threshold (int)
        retry_step: Number of retry for a random placement step when inter-molecular atom-atom distance is below threshold (int)
        threshold: Threshold of inter-molecular atom-atom distance (float, angstrom)
        dec_rate: Decrease rate of density when retry for this function (float)

    Returns:
        Rdkit Mol object
    """
    rst_flag = _effective_restart_flag(work_dir, restart, restart_flag=restart_flag)
    dt1 = _cell_log_begin('amorphous_cell', restart=rst_flag)

    mols, n = _normalize_mol_counts(mols, n)

    # Stamp stable molecule identities before generating restart/cache payloads.
    # Otherwise the first run sees pre-stamp identities while later runs see
    # stamped `_yadonpy_molid` values, which defeats cache reuse.
    _cache_artifacts_best_effort(mols, prefer_var=True)

    cache_payload = _cell_cache_payload(
        'amorphous_cell',
        mols=mols,
        n=n,
        cell=cell,
        density=density,
        threshold=threshold,
        dec_rate=dec_rate,
        check_bond_ring_intersection=check_bond_ring_intersection,
        mp=mp,
        neutralize=neutralize,
        neutralize_tol=neutralize_tol,
        charge_scale=charge_scale,
        charge_tolerance=charge_tolerance,
        ions=ions,
        large_system_mode=large_system_mode,
    )
    if rst_flag:
        cached = _rw_load(work_dir, 'amorphous_cell', cache_payload)
        if cached is not None:
            cached_state = _rw_load_state(work_dir, 'amorphous_cell', cache_payload)
            _cell_log_restart_reuse('amorphous_cell', source='hashed restart cache', work_dir=work_dir)
            _cell_log_done('amorphous_cell', dt1, restart=rst_flag)
            return _restore_cached_cell_state(cached, cached_state)
        cached, cached_state = _cell_cache_load(work_dir, 'amorphous_cell', cache_payload)
        if cached is not None:
            _cell_log_restart_reuse('amorphous_cell', source='stable cell cache', work_dir=work_dir)
            _cell_log_done('amorphous_cell', dt1, restart=rst_flag)
            return _restore_cached_cell_state(cached, cached_state)
        _cell_log_restart_miss('amorphous_cell', work_dir=work_dir)

    utils.radon_print('[PACK] Building amorphous cell by random molecular placement.', level=1)

    # ------------------------------------------------------------------
    # yadonpy extension: per-species charge scaling specification.
    #
    # Users can provide a list aligned with `mols` to represent dielectric
    # screening (e.g., scale ionic charges to 0.8). This is stored in the
    # returned cell's metadata and later applied during GROMACS export.
    #
    # Supported forms:
    #   - None: no scaling (all 1.0)
    #   - float/int: apply to all species
    #   - list/tuple: per-species list, same length as `mols`
    #   - dict: stored as-is (advanced use)
    # ------------------------------------------------------------------
    _cs = None
    try:
        if charge_scale is None:
            _cs = [1.0] * len(mols)
        elif isinstance(charge_scale, (int, float)):
            _cs = [float(charge_scale)] * len(mols)
        elif isinstance(charge_scale, (list, tuple)):
            if len(charge_scale) != len(mols):
                raise ValueError(
                    f"charge_scale length mismatch: got {len(charge_scale)} but mols has {len(mols)}"
                )
            _cs = [float(x) for x in charge_scale]
        elif isinstance(charge_scale, dict):
            # Keep dict for exporter to interpret if needed.
            _cs = charge_scale
        else:
            _cs = [1.0] * len(mols)
    except Exception:
        _cs = [1.0] * len(mols)

    # ------------------------------------------------------------------
    # Auto-detect (and record) system net charge for mixed systems.
    # This does NOT change the system; it only reports and stores info.
    # - raw: sum of molecule net charges
    # - scaled: applies per-species charge_scale (useful for ionic screening)
    # ------------------------------------------------------------------
    try:
        q_raw = 0.0
        q_scaled = 0.0
        for _i, _mol in enumerate(mols):
            qi = float(_mol_net_charge(_mol))
            ni = float(n[_i]) if _i < len(n) else 1.0
            q_raw += qi * ni
            si = 1.0
            if isinstance(_cs, list) and _i < len(_cs):
                si = float(_cs[_i])
            q_scaled += qi * ni * si

        tol = float(charge_tolerance)
        if abs(q_scaled) > tol:
            utils.radon_print(
                f"Warning: system net charge (scaled) = {q_scaled:.6f} e exceeds tolerance {tol:.1e}",
                level=2,
            )
        # store for later export/analysis
        try:
            setattr(cell, "net_charge", q_scaled)
        except Exception:
            pass
    except Exception:
        q_raw = None
        q_scaled = None
        tol = float(charge_tolerance)

    # Small residual system charge can appear after RESP/rounding or charge scaling.
    # If the scaled net charge is already close to zero, fold the tiny remainder into
    # the first atom of the first ITP-carrying species instead of treating the box as charged.
    try:
        if len(mols) > 0 and q_scaled is not None and abs(float(q_scaled)) < 0.1:
            _nudge_first_atom_charge(mols[0], -float(q_scaled))
            q_scaled = 0.0
            q_raw = None
            utils.radon_print('[PACK] Adjusted the first atom charge of the first species to absorb a residual system charge < 0.1 e.', level=1)
    except Exception:
        pass

    # --- Ion injection / charge neutrality ---------------------------------
    if ions is not None:
        if isinstance(ions, IonPack):
            ions = [ions]
        # Compute total charge of solute (mols * n)
        q_sol = 0.0
        for _m, _mol in enumerate(mols):
            q_sol += _mol_net_charge(_mol) * float(n[_m])
        # Add ions: if count is None and neutralize=True, auto-compute required number
        for pack in ions:
            q_ion = _mol_net_charge(pack.mol)
            if abs(q_ion) < 1e-8:
                utils.radon_print('Ion pack has ~0 charge; cannot neutralize.', level=3)
                raise ValueError('Ion pack has ~0 charge; cannot neutralize.')

            if pack.n is None:
                if not neutralize:
                    raise ValueError('n_ion is None and neutralize=False. Please specify n_ion.')
                # required ions to neutralize solute charge
                req = -q_sol / q_ion
                req_int = int(round(req))
                if abs(req - req_int) > 1e-6:
                    raise ValueError('Cannot neutralize: solute charge %.6f not divisible by ion charge %.6f' % (q_sol, q_ion))
                pack.n = req_int
            # append into mols/n for placement
            mols.append(pack.mol)
            n.append(int(pack.n))

        # final neutrality check
        q_total = 0.0
        for _m, _mol in enumerate(mols):
            q_total += _mol_net_charge(_mol) * float(n[_m])
        if abs(q_total) > float(neutralize_tol):
            raise ValueError('System net charge is not neutral after adding ions. net_charge=%.6f' % q_total)
    # -----------------------------------------------------------------------

    total_atoms_target = _estimate_total_atoms(mols, n)
    large_pack_enabled = _resolve_large_system_mode(large_system_mode, total_atoms_target)
    reference_clearance = _estimate_vdw_reference_clearance(mols, dist_min=threshold)
    if large_pack_enabled:
        utils.radon_print(
            '[PACK] Large-system mode enabled automatically for amorphous_cell '
            f'(target_atoms={total_atoms_target}, threshold={threshold:.3f} A).',
            level=1,
        )

    trial_cell = cell
    trial_density = density
    retries_left = int(retry)
    attempt_index = 0
    pack_diagnostics: dict[str, object] = {
        'schema_version': AMORPHOUS_CELL_SCHEMA_VERSION,
        'target_atoms': int(total_atoms_target),
        'large_system_mode': bool(large_pack_enabled),
        'reference_clearance_angstrom': float(reference_clearance),
        'attempts': [],
    }
    cell_c = Chem.Mol()

    while True:
        attempt_index += 1
        mols_c = [utils.deepcopy_mol(mol) for mol in mols]
        mol_coord = [np.array(mol.GetConformer(0).GetPositions()) for mol in mols_c]
        has_ring = False
        tri_coord = None
        bond_coord = None

        if trial_cell is None:
            cell_c = Chem.Mol()
        else:
            cell_c = utils.deepcopy_mol(trial_cell)
            sssr_tmp = Chem.GetSSSR(cell_c)
            if type(sssr_tmp) is int:
                if sssr_tmp > 0:
                    has_ring = True
            elif len(sssr_tmp) > 0:
                has_ring = True

        for mol_c in mols_c:
            sssr_tmp = Chem.GetSSSR(mol_c)
            if type(sssr_tmp) is int:
                if sssr_tmp > 0:
                    has_ring = True
            elif len(sssr_tmp) > 0:
                has_ring = True

        if trial_density is None and hasattr(cell_c, 'cell'):
            xhi = cell_c.cell.xhi
            xlo = cell_c.cell.xlo
            yhi = cell_c.cell.yhi
            ylo = cell_c.cell.ylo
            zhi = cell_c.cell.zhi
            zlo = cell_c.cell.zlo
        else:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols_c, cell_c], [*n, 1], density=trial_density)
            setattr(cell_c, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

        pack_state = _build_large_pack_state(
            cell_c,
            threshold,
            enabled=large_pack_enabled,
            reference_clearance=reference_clearance,
        )
        pack_plan = _amorphous_pack_batches(mols_c, n)
        box_lengths = (
            float(cell_c.cell.xhi) - float(cell_c.cell.xlo),
            float(cell_c.cell.yhi) - float(cell_c.cell.ylo),
            float(cell_c.cell.zhi) - float(cell_c.cell.zlo),
        )
        attempt_diag: dict[str, object] = {
            'attempt_index': int(attempt_index),
            'density_g_cm3': (float(trial_density) if trial_density is not None else None),
            'box_lengths_angstrom': [float(v) for v in box_lengths],
            'phases': [],
            'placement_retries_total': 0,
            'placement_failures': {},
        }

        retry_flag = False
        placed_counter = 0
        for batch_idx, batch in enumerate(pack_plan, start=1):
            phase = str(batch['phase'])
            indices = tuple(int(v) for v in batch['indices'])
            attempt_diag['phases'].append(
                {
                    'phase': phase,
                    'species_indices': [int(v) for v in indices],
                    'species_names': [str(utils.get_name(mols_c[idx], default=f'M{idx+1}')) for idx in indices],
                }
            )
            for m in indices:
                mol = mols_c[m]
                species_name = str(utils.get_name(mol, default=f'M{m+1}'))
                for i in tqdm(
                    range(n[m]),
                    desc='[Unit cell generation %i/%i]' % (batch_idx, len(pack_plan)),
                    disable=const.tqdm_disable,
                ):
                    accepted = False
                    tries_used = 0
                    for r in range(retry_step):
                        tries_used = r + 1
                        trans = np.array([
                            np.random.uniform(xlo, xhi),
                            np.random.uniform(ylo, yhi),
                            np.random.uniform(zlo, zhi)
                        ])
                        rot = np.random.uniform(-np.pi, np.pi, 3)

                        mol_coord_c = mol_coord[m]
                        mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([1, 0, 0]), rot[0])
                        mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([0, 1, 0]), rot[1])
                        mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([0, 0, 1]), rot[2])
                        mol_coord_c += trans - np.mean(mol_coord_c, axis=0)

                        if cell_c.GetNumConformers() == 0:
                            accepted = True
                            break

                        check_3d = check_3d_structure_cell(cell_c, mol_coord_c, dist_min=threshold, pack_state=pack_state)
                        if check_3d and check_bond_ring_intersection and has_ring:
                            check_3d, tri_coord_new, bond_coord_new = check_3d_bond_ring_intersection(
                                cell_c,
                                mon=mol,
                                mon_coord=mol_coord_c,
                                tri_coord=tri_coord,
                                bond_coord=bond_coord,
                                mp=mp,
                            )
                        if check_3d:
                            if check_bond_ring_intersection and has_ring:
                                tri_coord = tri_coord_new
                                bond_coord = bond_coord_new
                            accepted = True
                            break
                        if r < retry_step - 1 and (r == 0 or (r + 1) % 100 == 0):
                            utils.radon_print('[PACK] Retry placing a molecule in cell. Step=%i, %i/%i' % (placed_counter + 1, r + 1, retry_step), level=1)

                    attempt_diag['placement_retries_total'] = int(attempt_diag['placement_retries_total']) + max(0, tries_used - 1)
                    if not accepted:
                        retry_flag = True
                        bucket = attempt_diag['placement_failures'].setdefault(
                            species_name,
                            {'count': 0, 'phase': phase, 'natoms': int(mol.GetNumAtoms())},
                        )
                        bucket['count'] = int(bucket['count']) + 1
                        utils.radon_print('[PACK] Reached maximum number of retrying in step %i of poly.amorphous_cell.' % (placed_counter + 1), level=1)
                        break

                    cell_n = cell_c.GetNumAtoms()
                    cell_c = combine_mols(cell_c, mol)
                    for j in range(mol.GetNumAtoms()):
                        cell_c.GetConformer(0).SetAtomPosition(
                            cell_n + j,
                            Geom.Point3D(mol_coord_c[j, 0], mol_coord_c[j, 1], mol_coord_c[j, 2])
                        )
                    _append_large_pack_coords(pack_state, mol_coord_c, cell_c.cell)
                    placed_counter += 1

                if retry_flag:
                    break
            if retry_flag:
                break

        box_volume = max(float(np.prod(np.asarray(box_lengths, dtype=float))), 1.0e-12)
        approx_occ = 0.0
        for mol, count in zip(mols_c, n):
            span = np.maximum(_estimate_mol_bbox_span(mol), 1.0e-3)
            approx_occ += float(np.prod(span)) * float(count)
        approx_occ = min(max(approx_occ / box_volume, 0.0), 0.999999)
        attempt_diag['approx_occupied_fraction'] = float(approx_occ)
        attempt_diag['approx_max_cavity_angstrom'] = float(np.cbrt(max(box_volume * (1.0 - approx_occ), 0.0)))
        pack_diagnostics['attempts'].append(attempt_diag)

        if not retry_flag:
            break

        if retries_left <= 0:
            _write_pack_diagnostics(work_dir, pack_diagnostics)
            raise RuntimeError('Reached maximum number of retrying poly.amorphous_cell without finding a valid packing.')

        retry_axes = (2,) if (trial_density is None and trial_cell is not None) else _estimate_packing_stress_axes(mols_c, n, cell_c.cell)
        retry_target = _next_amorphous_retry_target(trial_cell, trial_density, dec_rate, axes=retry_axes)
        retry_value = retry_target['log_value']
        if isinstance(retry_value, tuple):
            retry_value = ', '.join(f'{float(v):.3f}' for v in retry_value)
        utils.radon_print(retry_target['log'] % (retries_left, retry_value), level=1)
        retries_left -= 1
        trial_cell = retry_target['cell']
        trial_density = retry_target['density']

    _cell_log_done('amorphous_cell', dt1, restart=rst_flag)

    # ------------------------------------------------------------------
    # yadonpy extension: attach composition metadata to the returned cell.
    # This lets downstream exporters/analysis reconstruct a mixed system
    # (moleculetype list, counts, smiles) without relying on names.
    # ------------------------------------------------------------------
    try:
        import json
        meta = []
        for idx, (_mol, _count) in enumerate(zip(mols, n)):
            try:
                smi = _mol.GetProp('_yadonpy_smiles') if _mol.HasProp('_yadonpy_smiles') else Chem.MolToSmiles(_mol, isomericSmiles=True)
            except Exception:
                smi = ''
            # Per-species charge scale (if provided as a list) is attached here.
            # If a dict was provided, we store it at the system level below.
            cs_val = None
            if isinstance(_cs, list):
                try:
                    cs_val = float(_cs[idx])
                except Exception:
                    cs_val = 1.0
            _bonded_requested = None
            _bonded_method = None
            _bonded_explicit = False
            _bonded_signature = None
            _cached_mol_id = None
            _cached_artifact_dir = None
            _ff_name = None
            _charge_method = None
            _prefer_db = None
            _require_db = None
            _require_ready = None
            _charge_groups = None
            _resp_constraints = None
            _polyelectrolyte_summary = None
            _residue_map = None
            try:
                if hasattr(_mol, 'HasProp'):
                    if _mol.HasProp('_yadonpy_bonded_requested'):
                        _bonded_requested = str(_mol.GetProp('_yadonpy_bonded_requested')).strip() or None
                    if _mol.HasProp('_yadonpy_bonded_method'):
                        _bonded_method = str(_mol.GetProp('_yadonpy_bonded_method')).strip() or None
                    if _mol.HasProp('_yadonpy_bonded_explicit'):
                        _bonded_explicit = str(_mol.GetProp('_yadonpy_bonded_explicit')).strip().lower() in ('1','true','yes','on')
                    if _mol.HasProp('_yadonpy_bonded_signature'):
                        _bonded_signature = str(_mol.GetProp('_yadonpy_bonded_signature')).strip() or None
                    if _mol.HasProp('_yadonpy_molid'):
                        _cached_mol_id = str(_mol.GetProp('_yadonpy_molid')).strip() or None
                    if _mol.HasProp('_yadonpy_artifact_dir'):
                        _cached_artifact_dir = str(_mol.GetProp('_yadonpy_artifact_dir')).strip() or None
                    if _mol.HasProp('ff_name'):
                        _ff_name = str(_mol.GetProp('ff_name')).strip() or None
                    if _mol.HasProp('_yadonpy_charge_method'):
                        _charge_method = str(_mol.GetProp('_yadonpy_charge_method')).strip() or None
                    if _mol.HasProp('_yadonpy_prefer_db'):
                        _prefer_db = str(_mol.GetProp('_yadonpy_prefer_db')).strip() or None
                    if _mol.HasProp('_yadonpy_require_db'):
                        _require_db = str(_mol.GetProp('_yadonpy_require_db')).strip() or None
                    if _mol.HasProp('_yadonpy_require_ready'):
                        _require_ready = str(_mol.GetProp('_yadonpy_require_ready')).strip() or None
                    try:
                        _charge_groups = get_charge_groups(_mol)
                        _resp_constraints = get_resp_constraints(_mol)
                        _polyelectrolyte_summary = get_polyelectrolyte_summary(_mol)
                        _residue_map = build_residue_map(_mol, mol_name=utils.get_name(_mol, default='MOL'))
                    except Exception:
                        if bool(polyelectrolyte_mode):
                            annotated = annotate_polyelectrolyte_metadata(_mol)
                            _charge_groups = annotated["summary"]["groups"]
                            _resp_constraints = annotated["constraints"]
                            _polyelectrolyte_summary = annotated["summary"]
            except Exception:
                _bonded_requested = None
                _bonded_method = None
                _bonded_explicit = False
                _bonded_signature = None
                _cached_mol_id = None
                _cached_artifact_dir = None
                _ff_name = None
                _charge_method = None
                _prefer_db = None
                _require_db = None
                _require_ready = None
                _charge_groups = None
                _resp_constraints = None
                _polyelectrolyte_summary = None
                _residue_map = None
            _effective_polyelectrolyte_mode = False
            try:
                if isinstance(_polyelectrolyte_summary, dict):
                    _effective_polyelectrolyte_mode = bool(uses_localized_charge_groups(_polyelectrolyte_summary))
            except Exception:
                _effective_polyelectrolyte_mode = False
            meta.append({
                'smiles': smi,
                'n': int(_count),
                'natoms': int(_mol.GetNumAtoms()),
                'charge_scale': cs_val,
                'name': (utils.get_name(_mol, default=None)),
                'bonded_requested': _bonded_requested,
                'bonded_method': _bonded_method,
                'bonded_explicit': bool(_bonded_explicit),
                'bonded_signature': _bonded_signature,
                'cached_mol_id': _cached_mol_id,
                'cached_artifact_dir': _cached_artifact_dir,
                'ff_name': _ff_name,
                'charge_method': _charge_method,
                'prefer_db': _prefer_db,
                'require_db': _require_db,
                'require_ready': _require_ready,
                'charge_groups': _charge_groups,
                'resp_constraints': _resp_constraints,
                'polyelectrolyte_summary': _polyelectrolyte_summary,
                'residue_map': _residue_map,
                'polyelectrolyte_mode': bool(_effective_polyelectrolyte_mode),
            })
        payload = {
            'schema_version': AMORPHOUS_CELL_SCHEMA_VERSION,
            'density_g_cm3': (float(trial_density) if trial_density is not None else None),
            'requested_density_g_cm3': (float(density) if density is not None else None),
            'species': meta,
            'pack_mode': ('large_system' if large_pack_enabled else 'default'),
            'target_atoms': int(total_atoms_target),
            'polyelectrolyte_mode': bool(polyelectrolyte_mode),
            'packing_diagnostics': pack_diagnostics,
        }
        if isinstance(_cs, dict):
            payload['charge_scale'] = _cs

        # -----------------------------
        # Net-charge auto detection
        # -----------------------------
        # We report both:
        #   - raw net charge (AtomicCharge/formal-charge sum)
        #   - scaled net charge (after per-species `charge_scale`)
        #
        # The simulation can still proceed if the system is non-neutral
        # (some users intentionally model charged boxes), but we emit a
        # warning when |q_scaled| > charge_tolerance.
        q_raw = 0.0
        q_scaled = 0.0
        for idx, (_mol, _count) in enumerate(zip(mols, n)):
            qi = float(_mol_net_charge(_mol)) * float(_count)
            q_raw += qi
            sc = 1.0
            if isinstance(_cs, list):
                try:
                    sc = float(_cs[idx])
                except Exception:
                    sc = 1.0
            elif isinstance(_cs, (int, float)):
                sc = float(_cs)
            q_scaled += qi * sc
        payload['net_charge_raw'] = float(q_raw)
        payload['net_charge_scaled'] = float(q_scaled)
        payload['charge_tolerance'] = float(charge_tolerance)
        payload['net_charge_ok'] = bool(abs(q_scaled) <= float(charge_tolerance))

        try:
            if abs(q_scaled) > float(charge_tolerance):
                utils.radon_print(
                    'Warning: system net charge (after charge_scale) is %.6f e (tolerance=%.2e).'
                    % (float(q_scaled), float(charge_tolerance)),
                    level=2,
                )
        except Exception:
            pass
        cell_c.SetProp('_yadonpy_cell_meta', json.dumps(payload, ensure_ascii=False))
        _write_pack_diagnostics(work_dir, payload.get('packing_diagnostics') or {})
    except Exception:
        pass

    _rw_save(work_dir, 'amorphous_cell', cache_payload, cell_c)
    state = _cell_state_from_mol(cell_c)
    if state is not None:
        _rw_save_state(work_dir, 'amorphous_cell', cache_payload, state)
    _cell_cache_save(work_dir, 'amorphous_cell', cache_payload, cell_c, state=state)

    return cell_c


def amorphous_mixture_cell(
        mols,
        n,
        cell=None,
        density=0.1,
        retry=20,
        retry_step=1000,
        threshold=2.0,
        dec_rate=0.8,
        check_bond_ring_intersection=False,
        mp=0,
        charge_scale=None,
        polyelectrolyte_mode=False,
        large_system_mode='auto',
        work_dir=None,
        restart=None,
):
    """
    poly.amorphous_mixture_cell

    This function is alias of poly.amorphous_cell to maintain backward compatibility
    """
    return amorphous_cell(
            mols,
            n,
            cell=cell,
            density=density,
            retry=retry,
            retry_step=retry_step,
            threshold=threshold,
            dec_rate=dec_rate,
            check_bond_ring_intersection=check_bond_ring_intersection,
            mp=mp,
            charge_scale=charge_scale,
            polyelectrolyte_mode=polyelectrolyte_mode,
            large_system_mode=large_system_mode,
            work_dir=work_dir,
            restart=restart,
    )


def nematic_cell(mols, n, cell=None, density=0.1, retry=20, retry_step=1000, threshold=2.0, dec_rate=0.8,
        check_bond_ring_intersection=False, mp=0, restart_flag=None, work_dir=None, restart=None):
    """
    poly.nematic_cell

    Simple unit cell generator with nematic-like ordered structure for x axis

    Args:
        mols: Array of RDkit Mol object
        n: Array of number of molecules in the unit cell (int)

    Optional args:
        cell: Initial structure of unit cell (RDkit Mol object)
        density: (float, g/cm3)
        retry: Number of retry for this function when inter-molecular atom-atom distance is below threshold (int)
        retry_step: Number of retry for a random placement step when inter-molecular atom-atom distance is below threshold (int)
        threshold: Threshold of inter-molecular atom-atom distance (float, angstrom)
        dec_rate: Decrease rate of density when retry for this function (float)

    Returns:
        Rdkit Mol object
    """
    rst_flag = _effective_restart_flag(work_dir, restart, restart_flag=restart_flag)
    dt1 = _cell_log_begin('nematic_cell', restart=rst_flag)

    mols, n = _normalize_mol_counts(mols, n)

    # ------------------------------------------------------------------
    # yadonpy: cache per-molecule GROMACS artifacts early.
    # ------------------------------------------------------------------
    _cache_artifacts_best_effort(mols, prefer_var=False)



    has_ring = False
    tri_coord = None
    bond_coord = None

    if cell is None:
        cell_c = Chem.Mol()
    else:
        cell_c = utils.deepcopy_mol(cell)
        sssr_tmp = Chem.GetSSSR(cell_c)
        if type(sssr_tmp) is int:
            if sssr_tmp > 0:
                has_ring = True
        elif len(sssr_tmp) > 0:  # For RDKit version >= 2022.09
            has_ring = True

    mols_c = [utils.deepcopy_mol(mol) for mol in mols]
    mol_coord = [np.array(mol.GetConformer(0).GetPositions()) for mol in mols_c]
    # Alignment molecules
    for mol in mols_c:
        Chem.rdMolTransforms.CanonicalizeConformer(mol.GetConformer(0), ignoreHs=False)
        sssr_tmp = Chem.GetSSSR(mol)
        if type(sssr_tmp) is int:
            if sssr_tmp > 0:
                has_ring = True
        elif len(sssr_tmp) > 0:  # For RDKit version >= 2022.09
            has_ring = True

    if density is None and hasattr(cell_c, 'cell'):
        xhi = cell_c.cell.xhi
        xlo = cell_c.cell.xlo
        yhi = cell_c.cell.yhi
        ylo = cell_c.cell.ylo
        zhi = cell_c.cell.zhi
        zlo = cell_c.cell.zlo
    else:
        xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols_c, cell_c], [*n, 1], density=density)
        setattr(cell_c, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    if not rst_flag:
        utils.radon_print('Start nematic-like cell generation.', level=1)
    retry_flag = False
    for m, mol in enumerate(mols_c):
        for i in tqdm(range(n[m]), desc='[Unit cell generation %i/%i]' % (m+1, len(mols_c)), disable=const.tqdm_disable):
            for r in range(retry_step):
                # Random translation
                trans = np.array([
                    np.random.uniform(xlo, xhi),
                    np.random.uniform(ylo, yhi),
                    np.random.uniform(zlo, zhi)
                ])

                #Random rotation around x axis
                rot = np.random.uniform(-np.pi, np.pi)

                mol_coord_c = mol_coord[m]
                if i % 2 == 0:
                    mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([1, 0, 0]), rot) + trans
                else:
                    mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([1, 0, 0]), rot)
                    mol_coord_c = calc.rotate_rod(mol_coord_c, np.array([0, 1, 0]), np.pi)
                    mol_coord_c += trans

                if cell_c.GetNumConformers() == 0: break

                check_3d = check_3d_structure_cell(cell_c, mol_coord_c, dist_min=threshold)
                if check_3d and check_bond_ring_intersection and has_ring:
                    check_3d, tri_coord_new, bond_coord_new = check_3d_bond_ring_intersection(cell_c, mon=mol, mon_coord=mol_coord_c,
                                                                    tri_coord=tri_coord, bond_coord=bond_coord, mp=mp)
                if check_3d:
                    if check_bond_ring_intersection and has_ring:
                        tri_coord = tri_coord_new
                        bond_coord = bond_coord_new
                    break
                elif r < retry_step-1:
                    if r == 0 or (r+1) % 100 == 0:
                        step_n = sum(n[:m])+i+1 if m > 0 else i+1
                        utils.radon_print('Retry placing a molecule in cell. Step=%i, %i/%i' % (step_n, r+1, retry_step))
                else:
                    retry_flag = True
                    step_n = sum(n[:m])+i+1 if m > 0 else i+1
                    utils.radon_print('Reached maximum number of retrying in the step %i of poly.nematic_cell.' % step_n, level=1)

            if retry_flag and retry > 0: break

            cell_n = cell_c.GetNumAtoms()

            # Add Mol to cell
            cell_c = combine_mols(cell_c, mol)

            # Set atomic coordinate
            for j in range(mol.GetNumAtoms()):
                cell_c.GetConformer(0).SetAtomPosition(
                    cell_n+j,
                    Geom.Point3D(mol_coord_c[j, 0], mol_coord_c[j, 1], mol_coord_c[j, 2])
                )

        if retry_flag and retry > 0: break

    if retry_flag:
        if retry <= 0:
            utils.radon_print('Reached maximum number of retrying poly.nematic_cell.', level=3)
        else:
            density *= dec_rate
            utils.radon_print('Retry poly.nematic_cell. Remainig %i times. The density is reduced to %f.' % (retry, density), level=1)
            retry -= 1
            cell_c = nematic_cell(mols, n, cell=cell, density=density, retry=retry,
                    retry_step=retry_step, threshold=threshold, dec_rate=dec_rate,
                    check_bond_ring_intersection=check_bond_ring_intersection, mp=mp, restart=True, work_dir=work_dir)

    _cell_log_done('nematic_cell', dt1, restart=rst_flag)

    return cell_c


def nematic_mixture_cell(mols, n, cell=None, density=0.1, retry=20, retry_step=1000, threshold=2.0, dec_rate=0.8,
                            check_bond_ring_intersection=False, mp=0, work_dir=None, restart=None):
    """
    poly.nematic_mixture_cell

    This function is alias of poly.nematic_cell to maintain backward compatibility
    """
    return nematic_cell(mols, n, cell=cell, density=density, retry=retry, retry_step=retry_step, threshold=threshold, dec_rate=dec_rate,
                            check_bond_ring_intersection=check_bond_ring_intersection, mp=mp, work_dir=work_dir, restart=restart)


# DEPRECATION
def polymerize_cell(mol, n, m, terminate=None, terminate2=None, cell=None, density=0.1,
                    ff=None, retry=50, dist_min=0.7, threshold=2.0, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.polymerize_cell

    *** DEPRECATION ***
    Unit cell generator of a homopolymer by random walk

    Args:
        mol: RDkit Mol object
        n: Polymerization degree (int)
        m: Number of polymer chains (int)

    Optional args:
        terminate: terminated substitute at head (and tail) (RDkit Mol object)
        terminate2: terminated substitute at tail (RDkit Mol object)
        cell: Initial structure of unit cell (RDkit Mol object)
        density: (float, g/cm3)
        ff: Force field object (optional; external minimizer is not supported in yadonpy (GROMACS-only))
        retry: Number of retry when generating unsuitable structure (int)
        dist_min: Threshold of intra-molecular atom-atom distance(float, angstrom)
        threshold: Threshold of inter-molecular atom-atom distance (float, angstrom)

    Returns:
        Rdkit Mol object
    """

    mol_n = mol.GetNumAtoms()
    if terminate2 is None: terminate2 = terminate

    if cell is None:
        cell = Chem.Mol()
    else:
        cell = utils.deepcopy_mol(cell)
 
    if density is None and hasattr(cell, 'cell'):
        xhi = cell.cell.xhi
        xlo = cell.cell.xlo
        yhi = cell.cell.yhi
        ylo = cell.cell.ylo
        zhi = cell.cell.zhi
        zlo = cell.cell.zlo
    else:
        if terminate is None:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([mol, cell], [n*m, 1], density=density)
        else:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([mol, terminate, terminate2, cell], [n*m, m, m, 1], density=density)
        setattr(cell, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    for j in range(m):
        cell_n = cell.GetNumAtoms()
        cell = amorphous_cell(mol, 1, cell=cell, density=None, retry=retry, threshold=threshold)

        for i in tqdm(range(n-1), desc='[Unit cell generation %i/%i]' % (j+1, m), disable=const.tqdm_disable):
            cell_copy = utils.deepcopy_mol(cell)

            for r in range(retry):
                cell = connect_mols(cell, mol, random_rot=True)

                if MD_avail:
                    ff.ff_assign(cell)
                    cell, _ = md.quick_rw(cell, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
                else:
                    AllChem.MMFFOptimizeMolecule(cell, maxIters=50, confId=0)

                cell_coord = np.array(cell.GetConformer(0).GetPositions())
                cell_wcoord = calc.wrap(cell_coord, xhi, xlo, yhi, ylo, zhi, zlo)
                dist_matrix1 = calc.distance_matrix(cell_wcoord[:-mol_n], cell_wcoord[-mol_n:])

                if cell_n > 0:
                    dist_matrix2 = calc.distance_matrix(cell_wcoord[:cell_n], cell_wcoord[-mol_n:])
                    dist_min2 = dist_matrix2.min()
                else:
                    dist_min2 = threshold + 1

                if dist_matrix1.min() > dist_min and dist_min2 > threshold:
                    break
                elif r < retry-1:
                    cell = utils.deepcopy_mol(cell_copy)
                    utils.radon_print('Retry random walk step %03d' % (i+1))
                else:
                    utils.radon_print('Reached maximum number of retrying random walk step.', level=2)

        if terminate is not None:
            cell = terminate_rw(cell, terminate, terminate2)

    return cell


# DEPRECATION
def copolymerize_cell(mols, n, m, terminate=None, terminate2=None, cell=None, density=0.1,
                    ff=None, retry=50, dist_min=0.7, threshold=2.0, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.copolymerize_cell

    *** DEPRECATION ***
    Unit cell generator of a copolymer by random walk

    Args:
        mols: Array of RDkit Mol object
        n: Polymerization degree (int)
        m: Number of polymer chains (int)

    Optional args:
        terminate: terminated substitute at head (and tail) (RDkit Mol object)
        terminate2: terminated substitute at tail (RDkit Mol object)
        cell: Initial structure of unit cell (RDkit Mol object)
        density: (float, g/cm3)
        ff: Force field object (optional; external minimizer is not supported in yadonpy (GROMACS-only))
        retry: Number of retry when generating unsuitable structure (int)
        dist_min: Threshold of intra-molecular atom-atom distance(float, angstrom)
        threshold: Threshold of inter-molecular atom-atom distance (float, angstrom)

    Returns:
        Rdkit Mol object
    """
    if terminate2 is None: terminate2 = terminate

    if cell is None:
        cell = Chem.Mol()
    else:
        cell = utils.deepcopy_mol(cell)
 
    if density is None and hasattr(cell, 'cell'):
        xhi = cell.cell.xhi
        xlo = cell.cell.xlo
        yhi = cell.cell.yhi
        ylo = cell.cell.ylo
        zhi = cell.cell.zhi
        zlo = cell.cell.zlo
    else:
        nl = [n*m] * len(mols)
        if terminate is None:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols, cell], [*nl, 1], density=density)
        else:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols, terminate, terminate2, cell], [*nl, m, m, 1], density=density)
        setattr(cell, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    for k in range(m):
        cell_n = cell.GetNumAtoms()
        cell = amorphous_cell(mols[0], 1, cell=cell, density=None, retry=retry, threshold=threshold)

        for i in tqdm(range(n), desc='[Unit cell generation %i/%i]' % (k+1, m), disable=const.tqdm_disable):
            for j, mol in enumerate(mols):
                if i == 0 and j==0: continue
                mol_n = mol.GetNumAtoms()
                cell_copy = utils.deepcopy_mol(cell)

                for r in range(retry):
                    cell = connect_mols(cell, mol, random_rot=True)

                    if MD_avail:
                        ff.ff_assign(cell)
                        cell, _ = md.quick_rw(cell, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
                    else:
                        AllChem.MMFFOptimizeMolecule(cell, maxIters=50, confId=0)

                    cell_coord = np.array(cell.GetConformer(0).GetPositions())
                    cell_wcoord = calc.wrap(cell_coord, xhi, xlo, yhi, ylo, zhi, zlo)
                    dist_matrix1 = calc.distance_matrix(cell_wcoord[:-mol_n], cell_wcoord[-mol_n:])

                    if cell_n > 0:
                        dist_matrix2 = calc.distance_matrix(cell_wcoord[:cell_n], cell_wcoord[-mol_n:])
                        dist_min2 = dist_matrix2.min()
                    else:
                        dist_min2 = threshold + 1

                    if dist_matrix1.min() > dist_min and dist_min2 > threshold:
                        break
                    elif r < retry-1:
                        cell = utils.deepcopy_mol(cell_copy)
                        utils.radon_print('Retry random walk step %03d' % (i+1))
                    else:
                        utils.radon_print('Reached maximum number of retrying random walk step.', level=2)

        if terminate is not None:
            cell = terminate_rw(cell, terminate, terminate2)

    return cell


# DEPRECATION
def random_copolymerize_cell(mols, n, ratio, m, terminate=None, terminate2=None, cell=None, density=0.1,
                    ff=None, retry=50, dist_min=0.7, threshold=2.0, work_dir=None, omp=1, mpi=1, gpu=0):
    """
    poly.random_copolymerize_cell

    *** DEPRECATION ***
    Unit cell generator of a random copolymer by random walk

    Args:
        mols: Array of RDkit Mol object
        n: Polymerization degree (int)
        ratio: Array of monomer ratio (float, sum=1.0)
        m: Number of polymer chains (int)

    Optional args:
        terminate: terminated substitute at head (and tail) (RDkit Mol object)
        terminate2: terminated substitute at tail (RDkit Mol object)
        cell: Initial structure of unit cell (RDkit Mol object)
        density: (float, g/cm3)
        ff: Force field object (optional; external minimizer is not supported in yadonpy (GROMACS-only))
        retry: Number of retry when generating unsuitable structure (int)
        dist_min: Threshold of intra-molecular atom-atom distance(float, angstrom)
        threshold: Threshold of inter-molecular atom-atom distance (float, angstrom)

    Returns:
        Rdkit Mol object
    """

    if len(mols) != len(ratio):
        utils.radon_print('Inconsistency length of mols and ratio', level=3)
        return None

    ratio = np.array(ratio) / np.sum(ratio)

    mol_index = np.random.choice(a=list(range(len(mols))), size=(m, n), p=ratio)

    if terminate2 is None: terminate2 = terminate

    if cell is None:
        cell = Chem.Mol()
    else:
        cell = utils.deepcopy_mol(cell)
 
    if density is None and hasattr(cell, 'cell'):
        xhi = cell.cell.xhi
        xlo = cell.cell.xlo
        yhi = cell.cell.yhi
        ylo = cell.cell.ylo
        zhi = cell.cell.zhi
        zlo = cell.cell.zlo
    else:
        nl = [n*m*r for r in ratio]
        if terminate is None:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols, cell], [*nl, 1], density=density)
        else:
            xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([*mols, terminate, terminate2, cell], [*nl, m, m, 1], density=density)
        setattr(cell, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    for j in range(m):
        cell_n = cell.GetNumAtoms()
        cell = amorphous_cell(mols[mol_index[j, 0]], 1, cell=cell, density=None, retry=retry, threshold=threshold)

        for i in tqdm(range(n-1), desc='[Unit cell generation %i/%i]' % (j+1, m), disable=const.tqdm_disable):
            cell_copy = utils.deepcopy_mol(cell)
            mol = mols[mol_index[j, i+1]]
            mol_n = mol.GetNumAtoms()

            for r in range(retry):
                cell = connect_mols(cell, mol, random_rot=True)

                if MD_avail:
                    ff.ff_assign(cell)
                    cell, _ = md.quick_rw(cell, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)
                else:
                    AllChem.MMFFOptimizeMolecule(cell, maxIters=50, confId=0)

                cell_coord = np.array(cell.GetConformer(0).GetPositions())
                cell_wcoord = calc.wrap(cell_coord, xhi, xlo, yhi, ylo, zhi, zlo)
                dist_matrix1 = calc.distance_matrix(cell_wcoord[:-mol_n], cell_wcoord[-mol_n:])

                if cell_n > 0:
                    dist_matrix2 = calc.distance_matrix(cell_wcoord[:cell_n], cell_wcoord[-mol_n:])
                    dist_min2 = dist_matrix2.min()
                else:
                    dist_min2 = threshold + 1

                if dist_matrix1.min() > dist_min and dist_min2 > threshold:
                    break
                elif r < retry-1:
                    cell = utils.deepcopy_mol(cell_copy)
                    utils.radon_print('Retry random walk step %03d' % (i+1))
                else:
                    utils.radon_print('Reached maximum number of retrying random walk step.', level=2)

        if terminate is not None:
            cell = terminate_rw(cell, terminate, terminate2)

    return cell


def crystal_cell(mol, density=0.7, margin=(0.75, 1.0, 1.0), alpha=1, theta=1, d=1,
                dist_min=2.5, dist_max=3.0, step=0.1, max_iter=100, wrap=True, confId=0,
                chain1rot=False, FixSelfRot=True):
    """
    poly.crystal_cell

    Unit cell generator of crystalline polymer
    J. Phys. Chem. Lett. 2020, 11, 15, 5823–5829

    Args:
        mol: RDkit Mol object

    Optional args:
        density: (float, g/cm3)
        margin: Tuple of Margin of cell along x,y,z axis (float, angstrom)
        alpha: Rotation parameter of chain 2 around chain 1 in the confomation generator (int)
        theta: Rotation parameter of chain 2 around itself in the confomation generator (int)
        d: Translation parameter of chain 2 in the confomation generator (int)
        dist_min: Minimum threshold of inter-molecular atom-atom distance (float, angstrom)
        dist_max: Maximum threshold of inter-molecular atom-atom distance (float, angstrom)
        step: Step size of an iteration in adjusting interchain distance (float, angstrom)
        max_iter: Number of retry when adjusting interchain distance (int)
        wrap: Output coordinates are wrapped in cell against only x-axis (boolean)
        confId: Target conformer ID

    Debug option:
        chain1rot: Rotation parameter alpha works as rotation parameter of chain 1 around itself in the confomation generator (boolean)
        FixSelfRot: (boolean)

    Returns:
        Rdkit Mol object
    """

    # Alignment molecules
    mol_c = utils.deepcopy_mol(mol)
    Chem.rdMolTransforms.CanonicalizeConformer(mol_c.GetConformer(confId), ignoreHs=False)
    mol1_coord = np.array(mol_c.GetConformer(confId).GetPositions())

    # Rotation mol2 to align link vectors and x-axis
    set_linker_flag(mol_c)
    center = mol1_coord[mol_c.GetIntProp('head_idx')]
    link_vec = mol1_coord[mol_c.GetIntProp('tail_idx')] - mol1_coord[mol_c.GetIntProp('head_idx')]
    angle = calc.angle_vec(link_vec, np.array([1.0, 0.0, 0.0]), rad=True)
    if angle == 0 or angle == np.pi:
        pass
    else:
        vcross = np.cross(link_vec, np.array([1.0, 0.0, 0.0]))
        mol1_coord = calc.rotate_rod(mol1_coord, vcross, angle, center=center)

    # Adjusting interchain distance
    mol2_coord = mol1_coord + np.array([0.0, 0.0, dist_min])
    for r in range(max_iter):
        dist_matrix = calc.distance_matrix(mol1_coord, mol2_coord)
        if dist_matrix.min() >= dist_min and dist_matrix.min() <= dist_max:
            break
        elif dist_matrix.min() > dist_max:
            mol2_coord -= np.array([0.0, 0.0, step])
        elif dist_matrix.min() < dist_min:
            mol2_coord += np.array([0.0, 0.0, step])

    # Copy a polymer chain
    cell = combine_mols(mol_c, mol_c)
    cell_n = cell.GetNumAtoms()

    # Set atomic coordinate of chain2 in initial structure
    new_coord = np.vstack((mol1_coord, mol2_coord))
    for i in range(cell_n):
        cell.GetConformer(0).SetAtomPosition(i, Geom.Point3D(new_coord[i, 0], new_coord[i, 1], new_coord[i, 2]))

    # Set cell information
    xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([cell], [1], density=density, margin=margin, fit='x')
    setattr(cell, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    mol1_center = np.mean(mol1_coord, axis=0)
    mol2_center = np.mean(mol2_coord, axis=0)

    # Generate conformers for grobal minimum search
    utils.radon_print('Start crystal cell generation.', level=1)
    if not const.tqdm_disable:
        bar = tqdm(total=alpha*theta*d)
        bar.set_description('Generate confomers')
    for a in range(alpha):
        a_rot = 2*np.pi/alpha * a

        for t in range(theta):
            t_rot = 2*np.pi/theta * t

            for r in range(d):
                if a==0 and t==0 and r==0:
                    _, _, yhi, ylo, zhi, zlo = calc_cell_length([cell], [1], margin=margin, fit='xyz')
                    set_cell_param_conf(cell, 0, xhi, xlo, yhi, ylo, zhi, zlo)
                    if not const.tqdm_disable: bar.update(1)
                    continue

                r_tr = (xhi-xlo)/d * r

                # Translate chain2 along x-axis
                mol2_tr = mol2_coord + np.array([r_tr, 0.0, 0.0])

                # Rotate chain2 around itself
                mol2_rot = calc.rotate_rod(mol2_tr, np.array([1.0, 0.0, 0.0]), t_rot, center=mol2_center)

                if chain1rot:
                    # Rotate chain1 around itself
                    mol1_new = calc.rotate_rod(mol1_coord, np.array([1.0, 0.0, 0.0]), -a_rot, center=mol1_center)
                    mol2_new = mol2_rot
                    if FixSelfRot: mol2_new = calc.rotate_rod(mol2_new, np.array([1.0, 0.0, 0.0]), -a_rot, center=mol2_center)
                else:
                    # Rotate chain2 around chain1
                    mol1_new = mol1_coord
                    mol2_new = calc.rotate_rod(mol2_rot, np.array([1.0, 0.0, 0.0]), -a_rot, center=mol2_center)
                    if FixSelfRot: mol2_new = calc.rotate_rod(mol2_new, np.array([1.0, 0.0, 0.0]), a_rot, center=mol1_center)

                # Adjusting interchain distance
                for i in range(max_iter):
                    if wrap: mol2_new_w = calc.wrap(mol2_new, xhi, xlo, None, None, None, None)
                    else: mol2_new_w = mol2_new
                    dist_matrix = calc.distance_matrix(mol1_new, mol2_new_w)

                    if dist_matrix.min() >= dist_min and dist_matrix.min() <= dist_max:
                        break

                    mol2_new_center = np.mean(mol2_new, axis=0)
                    vec = np.array([0.0, mol2_new_center[1]-mol1_center[1], mol2_new_center[2]-mol1_center[2]])
                    vec = vec/np.linalg.norm(vec)

                    if dist_matrix.min() > dist_max:
                        mol2_new -= vec*step
                    elif dist_matrix.min() < dist_min:
                        mol2_new += vec*step

                # Add new conformer
                new_coord = np.vstack((mol1_new, mol2_new))
                if wrap: new_coord = calc.wrap(new_coord, xhi, xlo, None, None, None, None)
                conf = Chem.rdchem.Conformer(cell_n)
                conf.Set3D(True)
                for i in range(cell_n):
                    conf.SetAtomPosition(i, Geom.Point3D(new_coord[i, 0], new_coord[i, 1], new_coord[i, 2]))
                conf_id = cell.AddConformer(conf, assignId=True)

                # Set cell parameters
                _, _, yhi, ylo, zhi, zlo = calc_cell_length([cell], [1], confId=conf_id, margin=margin, fit='xyz')
                set_cell_param_conf(cell, conf_id, xhi, xlo, yhi, ylo, zhi, zlo)

                if not const.tqdm_disable: bar.update(1)
    if not const.tqdm_disable: bar.close()

    return cell


def single_chain_cell(mol, density=0.8, margin=0.75, confId=0):
    """
    poly.single_chain_cell

    Unit cell generator of single chain polymer (for 1D PBC calculation)

    Args:
        mol: RDkit Mol object

    Optional args:
        density: (float, g/cm3)
        margin: Margin of cell along x-axis (float, angstrom)
        confId: Target conformer ID

    Returns:
        Rdkit Mol object
    """

    cell = utils.deepcopy_mol(mol)

    # Alignment molecules
    Chem.rdMolTransforms.CanonicalizeConformer(cell.GetConformer(confId), ignoreHs=False)
    cell_coord = np.array(cell.GetConformer(confId).GetPositions())

    # Rotation mol to align link vectors and x-axis
    set_linker_flag(cell)
    center = cell_coord[cell.GetIntProp('head_idx')]
    link_vec = cell_coord[cell.GetIntProp('tail_idx')] - cell_coord[cell.GetIntProp('head_idx')]
    angle = calc.angle_vec(link_vec, np.array([1.0, 0.0, 0.0]), rad=True)
    if angle == 0 or angle == np.pi:
        pass
    else:
        vcross = np.cross(link_vec, np.array([1.0, 0.0, 0.0]))
        cell_coord = calc.rotate_rod(cell_coord, vcross, angle, center=center)

    # Set atomic coordinate
    for i in range(cell.GetNumAtoms()):
        cell.GetConformer(confId).SetAtomPosition(i, Geom.Point3D(cell_coord[i, 0], cell_coord[i, 1], cell_coord[i, 2]))

    # Set cell information
    xhi, xlo, yhi, ylo, zhi, zlo = calc_cell_length([cell], [1], density=density, margin=(margin, 1.5, 1.5), fit='x')
    setattr(cell, 'cell', utils.Cell(xhi, xlo, yhi, ylo, zhi, zlo))

    return cell


def super_cell(cell, x=1, y=1, z=1, confId=0):
    """
    poly.super_cell

    Super cell generator of RDkit Mol object

    Args:
        cell: RDkit Mol object

    Optional args:
        x, y, z: Number of replicating super cell (int)
        confId: Target conformer ID

    Returns:
        Rdkit Mol object
    """

    if cell.GetConformer(confId).HasProp('cell_dx'):
        lx = cell.GetConformer(confId).GetDoubleProp('cell_dx')
        ly = cell.GetConformer(confId).GetDoubleProp('cell_dy')
        lz = cell.GetConformer(confId).GetDoubleProp('cell_dz')
    else:
        lx = cell.cell.dx
        ly = cell.cell.dy
        lz = cell.cell.dz

    xcell = utils.deepcopy_mol(cell)
    cell_n = cell.GetNumAtoms()
    cell_coord = np.array(cell.GetConformer(confId).GetPositions())

    xcell.RemoveAllConformers()
    conf = Chem.rdchem.Conformer(cell_n)
    conf.Set3D(True)
    for i in range(cell_n):
        conf.SetAtomPosition(i, Geom.Point3D(cell_coord[i, 0], cell_coord[i, 1], cell_coord[i, 2]))
    xcell.AddConformer(conf, assignId=True)

    for ix in range(x-1):
        xcell_n = xcell.GetNumAtoms()
        xcell = combine_mols(xcell, cell)
        new_coord = cell_coord + np.array([lx*(ix+1), 0.0, 0.0])
        for i in range(cell_n):
            xcell.GetConformer(0).SetAtomPosition(
                xcell_n+i,
                Geom.Point3D(new_coord[i, 0], new_coord[i, 1], new_coord[i, 2])
            )

    ycell = utils.deepcopy_mol(xcell)
    xcell_n = xcell.GetNumAtoms()
    xcell_coord = np.array(xcell.GetConformer(0).GetPositions())
    for iy in range(y-1):
        ycell_n = ycell.GetNumAtoms()
        ycell = combine_mols(ycell, xcell)
        new_coord = xcell_coord + np.array([0.0, ly*(iy+1), 0.0])
        for i in range(xcell_n):
            ycell.GetConformer(0).SetAtomPosition(
                ycell_n+i,
                Geom.Point3D(new_coord[i, 0], new_coord[i, 1], new_coord[i, 2])
            )

    zcell = utils.deepcopy_mol(ycell)
    ycell_n = ycell.GetNumAtoms()
    ycell_coord = np.array(ycell.GetConformer(0).GetPositions())
    for iz in range(z-1):
        zcell_n = zcell.GetNumAtoms()
        zcell = combine_mols(zcell, ycell)
        new_coord = ycell_coord + np.array([0.0, 0.0, lz*(iz+1)])
        for i in range(ycell_n):
            zcell.GetConformer(0).SetAtomPosition(
                zcell_n+i,
                Geom.Point3D(new_coord[i, 0], new_coord[i, 1], new_coord[i, 2])
            )

    zcell.cell = utils.Cell(lx*x, zcell.cell.xlo, ly*y, zcell.cell.ylo, lz*z, zcell.cell.zlo)

    Chem.SanitizeMol(zcell)

    return zcell


def calc_cell_length(mols, n, density=1.0, confId=0, ignoreLinkers=True, fit=None, margin=(1.2, 1.2, 1.2)):
    """
    poly.calc_cell_length

    Calculate cell length

    Args:
        mols: Array of RDkit Mol object
        n: Array of number of molecules into cell (float)
            i.e. (polymerization degree) * (monomer ratio) * (number of polymer chains)
        density: (float, g/cm3)

    Optional args:
        confId: Target conformer ID (int)
        ignoreLinkers: Ignoring linker atoms in the mass calculation (boolean)
        fit: Length of specific axis is fitted for molecular coordinate (str, (x, y, z, xy, xz, yz, xyz, or max_cubic))
             If fit is xyz, density is ignored.
        margin: Margin of cell for fitting axis (float, angstrom)

    Returns:
        xhi, xlo: Higher and lower edges of x axis (float, angstrom)
        yhi, ylo: Higher and lower edges of y axis (float, angstrom)
        zhi, zlo: Higher and lower edges of z axis (float, angstrom)
    """

    # Calculate mass
    mass = 0
    for i, mol in enumerate(mols):
        if ignoreLinkers: set_linker_flag(mol)
        for atom in mol.GetAtoms():
            if ignoreLinkers and atom.GetBoolProp('linker'):
                pass
            else:
                mass += atom.GetMass()*n[i]

    # Check molecular length
    x_min = x_max = y_min = y_max = z_min = z_max = 0
    for mol in mols:
        if mol.GetNumConformers() == 0: continue
        mol_coord = np.array(mol.GetConformer(confId).GetPositions())

        if ignoreLinkers:
            del_coord = []
            for i, atom in enumerate(mol.GetAtoms()):
                if not atom.GetBoolProp('linker'):
                    del_coord.append(mol_coord[i])
            mol_coord = np.array(del_coord)

        mol_coord = calc.fix_trans(mol_coord)
        x_min = mol_coord.min(axis=0)[0] - margin[0] if mol_coord.min(axis=0)[0] - margin[0] < x_min else x_min
        x_max = mol_coord.max(axis=0)[0] + margin[0] if mol_coord.max(axis=0)[0] + margin[0] > x_max else x_max
        y_min = mol_coord.min(axis=0)[1] - margin[1] if mol_coord.min(axis=0)[1] - margin[1] < y_min else y_min
        y_max = mol_coord.max(axis=0)[1] + margin[1] if mol_coord.max(axis=0)[1] + margin[1] > y_max else y_max
        z_min = mol_coord.min(axis=0)[2] - margin[2] if mol_coord.min(axis=0)[2] - margin[2] < z_min else z_min
        z_max = mol_coord.max(axis=0)[2] + margin[2] if mol_coord.max(axis=0)[2] + margin[2] > z_max else z_max

    x_length = x_max - x_min
    y_length = y_max - y_min
    z_length = z_max - z_min

    # Determining cell length (angstrom)
    length = np.cbrt( (mass / const.NA) / (density / const.cm2ang**3)) / 2

    if fit is None:
        xhi = yhi = zhi = length
        xlo = ylo = zlo = -length
    elif fit == 'auto':
        if x_length > length*2:
            xhi = x_length/2
            xlo = -xhi
            yhi = zhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / x_length ) / 2
            ylo = zlo = -yhi
        elif y_length > length*2:
            yhi = y_length/2
            ylo = -yhi
            xhi = zhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / y_length ) / 2
            xlo = zlo = -xhi
        elif z_length > length*2:
            zhi = z_length/2
            zlo = -zhi
            xhi = yhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / z_length ) / 2
            xlo = ylo = -xhi
        else:
            xhi = yhi = zhi = length
            xlo = ylo = zlo = -length
    elif fit == 'x':
        xhi = x_length/2
        xlo = -xhi
        yhi = zhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / x_length ) / 2
        ylo = zlo = -yhi
    elif fit == 'y':
        yhi = y_length/2
        ylo = -yhi
        xhi = zhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / y_length ) / 2
        xlo = zlo = -xhi
    elif fit == 'z':
        zhi = z_length/2
        zlo = -zhi
        xhi = yhi = np.sqrt( ((mass / const.NA) / (density / const.cm2ang**3)) / z_length ) / 2
        xlo = ylo = -xhi
    elif fit == 'xy':
        xhi = x_length/2
        xlo = -xhi
        yhi = y_length/2
        ylo = -yhi
        zhi = (((mass / const.NA) / (density / const.cm2ang**3)) / x_length / y_length) / 2
        zlo = -zhi
    elif fit == 'xz':
        xhi = x_length/2
        xlo = -xhi
        zhi = z_length/2
        zlo = -zhi
        yhi = (((mass / const.NA) / (density / const.cm2ang**3)) / x_length / z_length) / 2
        ylo = -yhi
    elif fit == 'yz':
        yhi = y_length/2
        ylo = -yhi
        zhi = z_length/2
        zlo = -zhi
        xhi = (((mass / const.NA) / (density / const.cm2ang**3)) / y_length / z_length) / 2
        xlo = -xhi
    elif fit == 'xyz':
        xhi = x_length/2
        xlo = -xhi
        yhi = y_length/2
        ylo = -yhi
        zhi = z_length/2
        zlo = -zhi
    elif fit == 'max_cubic':
        cmax = max([x_max, y_max, z_max])
        cmin = min([x_min, y_min, z_min])
        cell_l = cmax if cmax > abs(cmin) else abs(cmin)
        xhi = yhi = zhi = cell_l
        xlo = ylo = zlo = -cell_l

    return xhi, xlo, yhi, ylo, zhi, zlo


def set_cell_param_conf(mol, confId, xhi, xlo, yhi, ylo, zhi, zlo):
    """
    poly.set_cell_param_conf

    Set cell parameters for RDKit Conformer object

    Args:
        mol: RDkit Mol object
        confId: Target conformer ID (int)
        xhi, xlo: Higher and lower edges of x axis (float, angstrom)
        yhi, ylo: Higher and lower edges of y axis (float, angstrom)
        zhi, zlo: Higher and lower edges of z axis (float, angstrom)

    Returns:
        Rdkit Mol object
    """

    conf = mol.GetConformer(confId)
    conf.SetDoubleProp('cell_xhi', xhi)
    conf.SetDoubleProp('cell_xlo', xlo)
    conf.SetDoubleProp('cell_yhi', yhi)
    conf.SetDoubleProp('cell_ylo', ylo)
    conf.SetDoubleProp('cell_zhi', zhi)
    conf.SetDoubleProp('cell_zlo', zlo)
    conf.SetDoubleProp('cell_dx', xhi-xlo)
    conf.SetDoubleProp('cell_dy', yhi-ylo)
    conf.SetDoubleProp('cell_dz', zhi-zlo)

    return mol


##########################################################
# Utility functions for check of 3D structure
##########################################################
def check_3d_proximity(coord1, coord2=None, dist_min=1.5, wrap=None, ignore_rad=3, dmat=None):
    """
    poly.check_3d_proximity

    Checking proximity between atoms

    Args:
        mol1: RDKit Mol object

    Optional args:
        mol2: RDKit Mol object of a flagment unit
        dist_min: Threshold of the minimum atom-atom distance (float, angstrom)
        confId: Target conformer ID of the polymer (int)
        wrap: Input cell object (mol.cell) if use wrapped coordinates

    Returns:
        boolean
            True: Without proximity atoms
            False: Found proximity atoms
    """
    if wrap is not None:
        coord1 = calc.wrap(coord1, wrap.xhi, wrap.xlo,
                        wrap.yhi, wrap.ylo, wrap.zhi, wrap.zlo)
        if coord2 is not None:
            coord2 = calc.wrap(coord2, wrap.xhi, wrap.xlo,
                            wrap.yhi, wrap.ylo, wrap.zhi, wrap.zlo)

    if coord2 is not None:
        if coord1.size == 0 or coord2.size == 0:
            return True
        threshold_sq = float(dist_min) * float(dist_min)
        chunk_size = 256
        for start in range(0, len(coord1), chunk_size):
            stop = min(start + chunk_size, len(coord1))
            diff = coord1[start:stop, None, :] - coord2[None, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff, optimize=True)
            if dmat is not None:
                mask = np.asarray(dmat[start:stop, :]) > ignore_rad
                if not np.any(mask):
                    continue
                dist_sq = np.where(mask, dist_sq, np.inf)
            if np.any(dist_sq <= threshold_sq):
                return False
        return True

    dist_matrix = calc.distance_matrix(coord1)
    np.fill_diagonal(dist_matrix, np.nan)

    if dmat is not None:
        imat = np.where(dmat <= ignore_rad, np.nan, 1)
        dist_matrix = dist_matrix * imat

    # Robustness: empty distance matrices can occur if a connection attempt failed
    # and the candidate fragment/polymer contains no comparable atoms for the
    # proximity check. In that case, treat as "no clash" and let upstream
    # logic decide whether to retry.
    if dist_matrix.size == 0:
        return True

    if np.nanmin(dist_matrix) > dist_min:
        return True
    else:
        return False


def check_3d_bond_length(mol, confId=0, bond_s=2.7, bond_a=1.9, bond_d=1.8, bond_t=1.4):
    """
    poly.check_3d_bond_length

    Args:
        mol: RDkit Mol object

    Optional args:
        bond_s: Threshold of the maximum single bond length (float, angstrom)
        bond_a: Threshold of the maximum aromatic bond length (float, angstrom)
        bond_d: Threshold of the maximum double bond length (float, angstrom)
        bond_t: Threshold of the maximum triple bond length (float, angstrom)

    Returns:
        boolean
    """
    coord = np.array(mol.GetConformer(confId).GetPositions())
    dist_matrix = calc.distance_matrix(coord)
    check = True

    # Cheking bond length
    for b in mol.GetBonds():
        bond_l = dist_matrix[b.GetBeginAtom().GetIdx(), b.GetEndAtom().GetIdx()]
        if b.GetBondTypeAsDouble() == 1.0 and bond_l > bond_s:
            check = False
            break
        elif b.GetBondTypeAsDouble() == 1.5 and bond_l > bond_a:
            check = False
            break
        elif b.GetBondTypeAsDouble() == 2.0 and bond_l > bond_d:
            check = False
            break
        elif b.GetBondTypeAsDouble() == 3.0 and bond_l > bond_t:
            check = False
            break

    return check


def check_3d_bond_ring_intersection(poly, mon=None, poly_coord=None, mon_coord=None, tri_coord=None, bond_coord=None,
                                    poly_atom_indices=None, mon_atom_indices=None, confId=0, wrap=None, mp=0):
    """
    poly.check_3d_bond_ring_intersection

    Checking bond-ring intersection using the Möller–Trumbore ray-triangle intersection algorithm

    Args:
        poly: RDKit Mol object of a polymer

    Optional args:
        mon: RDKit Mol object of a repeating unit
        tri_coord: Constructed atomic coordinates of triangles in the polymer (ndarray)
        bond_coord: Constructed atomic coordinates of bonds in the polymer (ndarray)
        confId: Target conformer ID of the polymer (int)
        wrap: Use wrapped coordinates if the poly having cell information
        mp: Parallel number of multiprocessing

    Return:
        check: (boolean)
            True: Without a penetration structure
            False: Found a penetration structure
        tri_coord: Constructed atomic coordinates of triangles in the polymer (ndarray)
        bond_coord: Constructed atomic coordinates of bonds in the polymer (ndarray)
    """
    check = True
    poly_n = poly.GetNumAtoms()
    mon_idx = poly_n
    ring = Chem.GetSymmSSSR(poly)
    if poly_coord is None:
        poly_coord = np.array(poly.GetConformer(confId).GetPositions())

    if len(ring) == 0:
        return True, None, None

    def _coord_index_map(atom_count, coord, atom_indices):
        if atom_indices is None:
            return {int(i): int(i) for i in range(min(int(atom_count), len(coord)))}
        return {int(atom_idx): int(pos) for pos, atom_idx in enumerate(np.asarray(atom_indices, dtype=int).tolist())}

    def _coords_from_indices(coord, indices, idx_map):
        mapped = []
        for atom_idx in indices:
            pos = idx_map.get(int(atom_idx))
            if pos is None or pos < 0 or pos >= len(coord):
                return None
            mapped.append(coord[pos])
        return mapped

    poly_idx_map = _coord_index_map(poly_n, poly_coord, poly_atom_indices)

    if type(mon) is Chem.Mol:
        mon_n = mon.GetNumAtoms()
        if mon_coord is None:
            mon_idx = int(poly_n - mon_n)
            mon_idx_map = None
        else:
            mon_idx_map = _coord_index_map(mon_n, mon_coord, mon_atom_indices)
    else:
        mon_idx_map = None

    # Construction of bond and ring surface coordinates in the growing chain part
    if tri_coord is None:
        if wrap is not None:
            poly_coord = calc.wrap(poly_coord, wrap.xhi, wrap.xlo,
                            wrap.yhi, wrap.ylo, wrap.zhi, wrap.zlo)
        p_ring_coord = []
        for i in range(len(ring)):
            ring_idx = list(ring[i])
            if ring_idx[0] >= mon_idx:
                continue
            mapped = _coords_from_indices(poly_coord, ring_idx, poly_idx_map)
            if mapped is not None:
                p_ring_coord.append(mapped)
        p_tri_coord = np.array([np.array([r[0], r[x+1], r[x+2]]) for r in p_ring_coord for x in range(len(r)-2)])
    else:
        p_tri_coord = tri_coord

    if bond_coord is None:
        p_bond_list = []
        for b in poly.GetBonds():
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            if begin_idx >= mon_idx or end_idx >= mon_idx:
                continue
            mapped = _coords_from_indices(poly_coord, (begin_idx, end_idx), poly_idx_map)
            if mapped is not None:
                p_bond_list.append(np.array(mapped))
        p_bond_coord = np.array(p_bond_list)
    else:
        p_bond_coord = bond_coord


    if type(mon) is Chem.Mol:
        # Vector construction of bond and ring surface coordinates in the additional monomer part
        if mon_coord is None:
            m_ring_coord = []
            for i in range(len(ring), 0, -1):
                if list(ring[i-1])[0] < mon_idx:
                    break
                mapped = _coords_from_indices(poly_coord, list(ring[i-1]), poly_idx_map)
                if mapped is not None:
                    m_ring_coord.append(mapped)
            m_tri_coord = np.array([np.array([r[0], r[x+1], r[x+2]]) for r in m_ring_coord for x in range(len(r)-2)])

            m_bond_list = []
            for b in poly.GetBonds():
                begin_idx = b.GetBeginAtomIdx()
                end_idx = b.GetEndAtomIdx()
                if begin_idx < mon_idx and end_idx < mon_idx:
                    continue
                mapped = _coords_from_indices(poly_coord, (begin_idx, end_idx), poly_idx_map)
                if mapped is not None:
                    m_bond_list.append(np.array(mapped))
            m_bond_coord = np.array(m_bond_list)
        else:
            if wrap is not None:
                mon_coord = calc.wrap(mon_coord, wrap.xhi, wrap.xlo,
                                wrap.yhi, wrap.ylo, wrap.zhi, wrap.zlo)

            m_ring = Chem.GetSymmSSSR(mon)
            m_ring_coord = []
            for i in range(len(m_ring)):
                mapped = _coords_from_indices(mon_coord, list(m_ring[i]), mon_idx_map)
                if mapped is not None:
                    m_ring_coord.append(mapped)
            m_tri_coord = np.array([np.array([r[0], r[x+1], r[x+2]]) for r in m_ring_coord for x in range(len(r)-2)])

            m_bond_list = []
            for b in mon.GetBonds():
                mapped = _coords_from_indices(mon_coord, (b.GetBeginAtomIdx(), b.GetEndAtomIdx()), mon_idx_map)
                if mapped is not None:
                    m_bond_list.append(np.array(mapped))
            m_bond_coord = np.array(m_bond_list)

        if  len(p_tri_coord) > 0 and len(m_tri_coord) > 0:
            r_tri_coord = np.vstack([p_tri_coord, m_tri_coord])
        elif len(p_tri_coord) > 0:
            r_tri_coord = p_tri_coord
        else:
            r_tri_coord = m_tri_coord
        r_bond_coord = np.vstack([p_bond_coord, m_bond_coord])


        if _has_bond_triangle_intersection(m_bond_coord, p_tri_coord, mp=mp):
            check = False
        elif _has_bond_triangle_intersection(p_bond_coord, m_tri_coord, mp=mp):
            check = False


    else:
        r_tri_coord = p_tri_coord
        r_bond_coord = p_bond_coord

        if _has_bond_triangle_intersection(p_bond_coord, p_tri_coord, mp=mp):
            check = False

    if not check:
        utils.radon_print('A bond-ring intersection was found.')

    return check, r_tri_coord, r_bond_coord


def MollerTrumbore(bond, tri):
    """
    poly.MollerTrumbore

    The Möller–Trumbore ray-triangle intersection algorithm

    Args:
        bond: Atomic coordinates of a bond (2*3 ndarray)
        tri: Atomic coordinates of a triangle (3*3 ndarray)

    Return:
        boolean
    """
    eps = 1e-4
    R   = bond[1, :] - bond[0, :]
    T   = bond[0, :] - tri[0, :]
    E1  = tri[1, :]  - tri[0, :]
    E2  = tri[2, :]  - tri[0, :]
    P   = np.cross(R, E2)
    S   = np.dot(P, E1)
    if -eps < S < eps:
        return False
    
    u   = np.dot(P, T)  / S
    if u < 0:
        return False

    Q   = np.cross(T, E1)
    v   = np.dot(Q, R)  / S
    t   = np.dot(Q, E2) / S

    if ((-eps < u < eps and -eps < v < eps) or
        (-eps < u < eps and 1-eps < v < 1+eps) or
        (1-eps < u < 1+eps and -eps < v < eps)
       ) and (-eps < t < eps or 1-eps < t < 1+eps):
        return False 
    elif v >= 0 and u+v <= 1 and 0 <= t <= 1:
        return True
    else:
        return False


def _MollerTrumbore(args):
    for arg in args:
        bond, tri = arg
        if MollerTrumbore(bond, tri):
            return True
    return False


def _bond_triangle_bbox_pairs(bond_coord, tri_coord, *, padding: float = 1.0e-8, chunk_size: int = 256):
    if len(bond_coord) == 0 or len(tri_coord) == 0:
        return []

    bond_min = np.min(bond_coord, axis=1)
    bond_max = np.max(bond_coord, axis=1)
    tri_min = np.min(tri_coord, axis=1)
    tri_max = np.max(tri_coord, axis=1)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    pad = float(max(padding, 0.0))
    chunk = int(max(chunk_size, 1))

    for start in range(0, len(bond_coord), chunk):
        stop = min(start + chunk, len(bond_coord))
        overlaps = np.all(
            (bond_max[start:stop, None, :] + pad) >= tri_min[None, :, :],
            axis=2,
        ) & np.all(
            (tri_max[None, :, :] + pad) >= bond_min[start:stop, None, :],
            axis=2,
        )
        if not np.any(overlaps):
            continue
        local_bonds, local_tris = np.nonzero(overlaps)
        for b_idx, t_idx in zip(local_bonds.tolist(), local_tris.tolist()):
            pairs.append((bond_coord[start + int(b_idx)], tri_coord[int(t_idx)]))
    return pairs


def _has_bond_triangle_intersection(bond_coord, tri_coord, *, mp: int = 0) -> bool:
    pairs = _bond_triangle_bbox_pairs(bond_coord, tri_coord)
    if not pairs:
        return False

    if int(mp) == 0:
        for bond, tri in pairs:
            if MollerTrumbore(bond, tri):
                return True
        return False

    args = list(np.array_split(np.array(pairs, dtype=object), int(mp)))
    with confu.ProcessPoolExecutor(max_workers=int(mp), mp_context=MP.get_context('spawn')) as executor:
        results = executor.map(_MollerTrumbore, args)
    return bool(np.any(np.array(list(results))))


def _local_proximity_candidate_indices(poly_coord, mon_coord, dist_min=1.0, padding=1.25):
    if poly_coord.size == 0 or mon_coord.size == 0:
        return np.arange(len(poly_coord), dtype=int)

    cutoff = float(max(dist_min, 0.0)) + float(padding)
    lower = np.min(mon_coord, axis=0) - cutoff
    upper = np.max(mon_coord, axis=0) + cutoff
    bbox_mask = np.all((poly_coord >= lower) & (poly_coord <= upper), axis=1)
    if np.any(bbox_mask):
        return np.flatnonzero(bbox_mask)

    mon_center = np.mean(mon_coord, axis=0)
    mon_radius = float(np.max(np.linalg.norm(mon_coord - mon_center, axis=1))) if len(mon_coord) > 0 else 0.0
    shell_mask = np.linalg.norm(poly_coord - mon_center, axis=1) <= (mon_radius + cutoff)
    if np.any(shell_mask):
        return np.flatnonzero(shell_mask)

    return np.empty(0, dtype=int)


def check_3d_structure_poly(poly, mon, poly_dmat=None, dist_min=1.0, ignore_rad=3, check_bond_length=False, tacticity=None):
    """
    poly.check_3d_structure_poly

    Checking proximity between atoms for polymer chain generators

    Args:
        poly: RDKit Mol object of a polymer
        mon: RDKit Mol object of an additional molecular
        poly_dmat: Topological distance matrix of a polymer

    Optional args:
        dist_min: Threshold of the minimum atom-atom distance (float, angstrom)
        ignore_rad: Radius of topological distance is ignored in distance check
        check_bond_length: Perform bond length check
        tacticity: Perform tacticity check

    Returns:
        boolean
            True: Without proximity atoms
            False: Found proximity atoms
    """
    n_mon = mon.GetNumAtoms()
    if poly_dmat is not None:
        poly_dmat = poly_dmat[:-n_mon, -n_mon:]
    coord = np.array(poly.GetConformer(0).GetPositions())
    p_coord = coord[:-n_mon]
    m_coord = coord[-n_mon:]
    candidate_idx = _local_proximity_candidate_indices(p_coord, m_coord, dist_min=dist_min)
    if candidate_idx.size == 0:
        check = True
    else:
        if candidate_idx.size < len(p_coord):
            p_coord = p_coord[candidate_idx]
            if poly_dmat is not None:
                poly_dmat = poly_dmat[candidate_idx, :]
        check = check_3d_proximity(p_coord, coord2=m_coord, dist_min=dist_min, dmat=poly_dmat, ignore_rad=ignore_rad)

    if check and check_bond_length:
        check = check_3d_bond_length(poly)

    if check and tacticity:
        check = check_tacticity(poly, tacticity=tacticity)

    return check


def check_3d_structure_cell(cell, mol_coord, dist_min=2.0, pack_state=None):
    """
    poly.check_3d_structure_cell

    Checking proximity between atoms for unit cell generators

    Args:
        cell: RDKit Mol object of a cell
        mol_coord: Atomic coordinates of an additional molecule

    Optional args:
        dist_min: Threshold of the minimum atom-atom distance (float, angstrom)

    Returns:
        boolean
            True: Without proximity atoms
            False: Found proximity atoms
    """
    check = True

    # Step 1. Self proximity check of an additional molecule (mol_coord) in the PBC cell
    wflag = np.where(
        (mol_coord[:, 0] > cell.cell.xhi) | (mol_coord[:, 0] < cell.cell.xlo) |
        (mol_coord[:, 1] > cell.cell.yhi) | (mol_coord[:, 1] < cell.cell.ylo) |
        (mol_coord[:, 2] > cell.cell.zhi) | (mol_coord[:, 2] < cell.cell.zlo),
        True, False
    )
    wcoord = mol_coord[wflag]
    uwcoord = mol_coord[np.logical_not(wflag)]
    if len(wcoord) > 0 and len(uwcoord) > 0:
        check = check_3d_proximity(uwcoord, coord2=wcoord, wrap=cell.cell, dist_min=dist_min)

    # Step 2. Proximity check between atoms in the cell (cell_coord) and in an addional molecule (mol_coord)
    if check:
        if pack_state is not None and pack_state.get('enabled', False):
            check = not _large_pack_clash(pack_state, mol_coord, dist_min, cell.cell)
        else:
            cell_coord = np.array(cell.GetConformer(0).GetPositions())
            check = check_3d_proximity(cell_coord, coord2=mol_coord, wrap=cell.cell, dist_min=dist_min)

    return check


##########################################################
# Utility functions for calculate polymerization degree 
##########################################################
# def calc_n_from_num_atoms(mols, natom, ratio=[1.0], label=1, terminal1=None, terminal2=None):
#     """
#     poly.calc_n_from_num_atoms

#     Calculate polymerization degree from target number of atoms

#     Args:
#         mols: List of RDkit Mol object
#         natom: Target of number of atoms

#     Optional args:
#         terminal1, terminal2: Terminal substruct of RDkit Mol object
#         ratio: List of monomer ratio

#     Returns:
#         int
#     """

#     if type(mols) is Chem.Mol:
#         mols = [mols]
#     elif type(mols) is not list:
#         utils.radon_print('Input should be an RDKit Mol object or its List', level=3)
#         return None

#     if len(mols) != len(ratio):
#         utils.radon_print('Inconsistency length of mols and ratio', level=3)
#         return None

#     ratio = np.array(ratio) / np.sum(ratio)

#     mol_n = 0.0
#     for i, mol in enumerate(mols):
#         new_mol = remove_linker_atoms(mol, label=label)
#         mol_n += new_mol.GetNumAtoms() * ratio[i]
    
#     if terminal1 is not None:
#         new_ter1 = remove_linker_atoms(terminal1)
#         ter1_n = new_ter1.GetNumAtoms()
#     else:
#         ter1_n = 0

#     if terminal2 is not None:
#         new_ter2 = remove_linker_atoms(terminal2)
#         ter2_n = new_ter2.GetNumAtoms()
#     elif terminal1 is not None:
#         ter2_n = ter1_n
#     else:
#         ter2_n = 0

#     n = int((natom - ter1_n - ter2_n) / mol_n + 0.5)
    
#     return n


# def calc_n_from_mol_weight(mols, mw, ratio=[1.0], terminal1=None, terminal2=None):
#     """
#     poly.calc_n_from_mol_weight

#     Calculate polymerization degree from target molecular weight

#     Args:
#         mols: List of RDkit Mol object
#         mw: Target of molecular weight

#     Optional args:
#         terminal1, terminal2: Terminal substruct of RDkit Mol object
#         ratio: List of monomer ratio

#     Returns:
#         int
#     """
#     if type(mols) is Chem.Mol:
#         mols = [mols]
#     elif type(mols) is not list:
#         utils.radon_print('Input should be an RDKit Mol object or its list', level=3)
#         return None

#     if len(mols) != len(ratio):
#         utils.radon_print('Inconsistency length of mols and ratio', level=3)
#         return None

#     ratio = np.array(ratio) / np.sum(ratio)

#     mol_mw = 0.0
#     for i, mol in enumerate(mols):
#         mol_mw += calc.molecular_weight(mol) * ratio[i]
    
#     ter1_mw = 0.0
#     if terminal1 is not None:
#         ter1_mw = calc.molecular_weight(terminal1)

#     ter2_mw = 0.0
#     if terminal2 is not None:
#         ter2_mw = calc.molecular_weight(terminal2)
#     elif terminal1 is not None:
#         ter2_mw = ter1_mw

#     n = int((mw - ter1_mw - ter2_mw) / mol_mw + 0.5)

#     return n


##########################################################
# Utility function for RDKit Mol object of polymers
##########################################################
def set_linker_flag(mol, reverse=False, label=1):
    """
    poly.set_linker_flag

    Args:
        mol: RDkit Mol object
        reverse: Reversing head and tail (boolean)

    Returns:
        boolean
    """

    # Connection points ("linkers") in YadonPy are typically represented by isotopic
    # hydrogens [nH] (n>=3) produced from "*" in SMILES (see core.chem_utils.star2h).
    #
    # Valid cases:
    #   - Monomer for linear polymerization: 2 linkers
    #   - Terminal unit: 1 linker (head==tail)
    #
    # If more than 2 are present, silently picking first/last can create wrong chemistry.
    # Fail fast instead.
    linker_atoms = []

    mol.SetIntProp('head_idx', -1)
    mol.SetIntProp('tail_idx', -1)
    mol.SetIntProp('head_ne_idx', -1)
    mol.SetIntProp('tail_ne_idx', -1)

    for atom in mol.GetAtoms():
        atom.SetBoolProp('linker', False)
        atom.SetBoolProp('head', False)
        atom.SetBoolProp('tail', False)
        atom.SetBoolProp('head_neighbor', False)
        atom.SetBoolProp('tail_neighbor', False)
        # Linker atom detection
        #
        # In RadonPy upstream, a special atom type (e.g. "MTIP") may be used for
        # non-polymer markers. In YadonPy we treat "*" dummy atoms in SMILES as
        # polymer connection points unconditionally, because excluding them can
        # break polymerization for valid inputs like "*CCO*".
        #
        # Also guard GetProp('ff_type') because not all toolchains set it.
        is_linker = False
        if atom.GetSymbol() == "H" and atom.GetIsotope() == label + 2:
            is_linker = True
        elif atom.GetSymbol() == "*":
            is_linker = True
        elif atom.HasProp('terminal') and atom.GetBoolProp('terminal'):
            is_linker = True

        if is_linker:
            atom.SetBoolProp('linker', True)
            linker_atoms.append(atom.GetIdx())

    if len(linker_atoms) == 0 or len(linker_atoms) > 2:
        # Clean linker flags so downstream code doesn't see a half-valid state
        for atom in mol.GetAtoms():
            atom.SetBoolProp('linker', False)
            atom.SetBoolProp('head', False)
            atom.SetBoolProp('tail', False)
            atom.SetBoolProp('head_neighbor', False)
            atom.SetBoolProp('tail_neighbor', False)
        return False

    if len(linker_atoms) == 1:
        mol_head_idx = mol_tail_idx = linker_atoms[0]
    else:
        if reverse:
            mol_head_idx, mol_tail_idx = linker_atoms[1], linker_atoms[0]
        else:
            mol_head_idx, mol_tail_idx = linker_atoms[0], linker_atoms[1]

    mol.SetIntProp('head_idx', mol_head_idx)
    mol.GetAtomWithIdx(mol_head_idx).SetBoolProp('head', True)

    mol.SetIntProp('tail_idx', mol_tail_idx)
    mol.GetAtomWithIdx(mol_tail_idx).SetBoolProp('tail', True)

    head_ne_atom = mol.GetAtomWithIdx(mol_head_idx).GetNeighbors()[0]
    mol.SetIntProp('head_ne_idx', head_ne_atom.GetIdx())
    head_ne_atom.SetBoolProp('head_neighbor', True)

    tail_ne_atom = mol.GetAtomWithIdx(mol_tail_idx).GetNeighbors()[0]
    mol.SetIntProp('tail_ne_idx', tail_ne_atom.GetIdx())
    tail_ne_atom.SetBoolProp('tail_neighbor', True)

    return True


def remove_linker_atoms(mol, label=1):
    """
    poly.remove_linker_atoms

    Args:
        mol: RDkit Mol object

    Returns:
        RDkit Mol object
    """

    new_mol = utils.deepcopy_mol(mol)

    def recursive_remove_linker_atoms(mol):
        for atom in mol.GetAtoms():
            if (atom.GetSymbol() == "H" and atom.GetIsotope() == label+2) or atom.GetSymbol() == "*":
                mol = utils.remove_atom(mol, atom.GetIdx())
                mol = recursive_remove_linker_atoms(mol)
                break
        return mol

    new_mol = recursive_remove_linker_atoms(new_mol)

    return new_mol


def set_terminal_idx(mol):

    count = 0
    for atom in mol.GetAtoms():
        resinfo = atom.GetPDBResidueInfo()
        if resinfo is None: continue
        resname = resinfo.GetResidueName()

        if resname == 'TU0':
            for na in atom.GetNeighbors():
                if na.GetPDBResidueInfo() is None: continue
                elif na.GetPDBResidueInfo().GetResidueName() != 'TU0':
                    mol.SetIntProp('terminal_idx1', atom.GetIdx())
                    mol.SetIntProp('terminal_ne_idx1', na.GetIdx())
                    count += 1

        elif resname == 'TU1':
            for na in atom.GetNeighbors():
                if na.GetPDBResidueInfo() is None: continue
                elif na.GetPDBResidueInfo().GetResidueName() != 'TU1':
                    mol.SetIntProp('terminal_idx2', atom.GetIdx())
                    mol.SetIntProp('terminal_ne_idx2', na.GetIdx())
                    count += 1
    
    return count


def set_mainchain_flag(mol):

    for atom in mol.GetAtoms():
        atom.SetBoolProp('main_chain', False)
    
    linker_result = set_linker_flag(mol)
    terminal_result = set_terminal_idx(mol)

    # Robustness: some molecules may have terminal residues but miss the expected index props
    # (e.g., partial PDB residue info or failed termination). In that case, fall back to head/tail linkers.
    if terminal_result > 0 and (not mol.HasProp('terminal_ne_idx1') or not mol.HasProp('terminal_ne_idx2')):
        terminal_result = 0

    if not linker_result and terminal_result == 0:
        return False

    if terminal_result > 0:
        try:
            t1 = mol.GetIntProp('terminal_ne_idx1')
            t2 = mol.GetIntProp('terminal_ne_idx2')
        except KeyError:
            terminal_result = 0
        else:
            if t1 == t2:
                path = [t1]
            else:
                path = Chem.GetShortestPath(mol, t1, t2)

    if terminal_result == 0 and mol.GetIntProp('head_idx') == mol.GetIntProp('tail_idx'):
        path = Chem.GetShortestPath(mol, mol.GetIntProp('head_idx'), mol.GetIntProp('head_ne_idx'))
    elif terminal_result == 0:
        path = Chem.GetShortestPath(mol, mol.GetIntProp('head_idx'), mol.GetIntProp('tail_idx'))

    for idx in path:
        atom = mol.GetAtomWithIdx(idx)
        atom.SetBoolProp('main_chain', True)
        for batom in atom.GetNeighbors():
            if batom.GetTotalDegree() == 1:  # Expect -H, =O, =S, -F, -Cl, -Br, -I
                batom.SetBoolProp('main_chain', True)
    
    rings = Chem.GetSymmSSSR(mol)
    m_rings = []
    for ring in rings:
        dup = list(set(path) & set(ring))
        if len(dup) > 0:
            m_rings.append(ring)
            for idx in ring:
                atom = mol.GetAtomWithIdx(idx)
                atom.SetBoolProp('main_chain', True)
                for batom in atom.GetNeighbors():
                    if batom.GetTotalDegree() == 1:  # Expect -H, =O, =S, -F, -Cl, -Br, -I
                        batom.SetBoolProp('main_chain', True)

    for m_ring in m_rings:
        for ring in rings:
            dup = list(set(m_ring) & set(ring))
            if len(dup) > 0:
                for idx in ring:
                    atom = mol.GetAtomWithIdx(idx)
                    atom.SetBoolProp('main_chain', True)
                    for batom in atom.GetNeighbors():
                        if batom.GetTotalDegree() == 1:  # Expect -H, =O, =S, -F, -Cl, -Br, -I
                            batom.SetBoolProp('main_chain', True)

    return True


def check_chiral_monomer(mol):
    """Return the number of chiral centers on the main chain (best-effort).

    Some user workflows pre-process monomers (QM/charge assignment) and may lose linker markers.
    This helper must therefore be defensive and never crash polymerization.
    """
    n_chiral = 0
    ter = utils.mol_from_smiles('*[C]')
    mol_c = utils.deepcopy_mol(mol)

    # If no linker markers exist, we cannot define a main chain reliably.
    try:
        if not set_linker_flag(mol_c):
            return 0
    except Exception:
        return 0

    try:
        mol_c = terminate_mols(mol_c, ter, random_rot=True)
    except Exception:
        # Termination is only used to detect chirality; proceed without termination.
        pass

    try:
        set_mainchain_flag(mol_c)
    except Exception:
        return 0
    for atom in mol_c.GetAtoms():
        if (int(atom.GetChiralTag()) == 1 or int(atom.GetChiralTag()) == 2) and atom.GetBoolProp('main_chain'):
            n_chiral += 1

    return n_chiral


def get_chiral_list(mol, confId=0):
    mol_c = utils.deepcopy_mol(mol)
    set_linker_flag(mol_c)
    if mol_c.GetIntProp('head_idx') >= 0 and mol_c.GetIntProp('tail_idx') >= 0:
        ter = utils.mol_from_smiles('*[C]')
        mol_c = terminate_mols(mol_c, ter, random_rot=True)
    set_mainchain_flag(mol_c)

    Chem.AssignStereochemistryFrom3D(mol_c, confId=confId)
    chiral_centers = np.array(Chem.FindMolChiralCenters(mol_c))

    if len(chiral_centers) == 0:
        return []

    chiral_centers = [int(x) for x in chiral_centers[:, 0]]
    chiral_list = []
    for atom in mol_c.GetAtoms():
        if atom.GetBoolProp('main_chain') and atom.GetIdx() in chiral_centers:
            chiral_list.append(int(atom.GetChiralTag()))
    chiral_list = np.array(chiral_list)

    return chiral_list


def get_tacticity(mol, confId=0):
    """
    poly.get_tacticity

    Get tacticity of polymer

    Args:
        mol: RDkit Mol object

    Optional args:
        confId: Target conformer ID (int)

    Returns:
        tacticity (str; isotactic, syndiotactic, atactic, or none)
    """

    tac = 'none'
    chiral_list = get_chiral_list(mol, confId=confId)

    chiral_cw = np.count_nonzero(chiral_list == 1)
    chiral_ccw = np.count_nonzero(chiral_list == 2)

    if chiral_cw == len(chiral_list) or chiral_ccw == len(chiral_list):
        tac = 'isotactic'

    else:
        chiral_even_s = np.count_nonzero(chiral_list[0::2] == 1)
        chiral_even_r = np.count_nonzero(chiral_list[0::2] == 2)
        chiral_odd_s = np.count_nonzero(chiral_list[1::2] == 1)
        chiral_odd_r = np.count_nonzero(chiral_list[1::2] == 2)
        if ((chiral_even_s == len(chiral_list[0::2]) and chiral_odd_r == len(chiral_list[1::2]))
                or (chiral_even_r == len(chiral_list[0::2]) and chiral_odd_s == len(chiral_list[1::2]))):
            tac = 'syndiotactic'
        else:
            tac = 'atactic'

    return tac


def check_tacticity(mol, tacticity, tac_array=None, confId=0):

    if tacticity == 'atactic' and tac_array is None:
        return True

    tac = get_tacticity(mol, confId=confId)

    check = False
    if tac == 'none':
        check = True

    elif tacticity == 'atactic' and tac_array is not None:
        set_mainchain_flag(mol)
        Chem.AssignStereochemistryFrom3D(mol, confId=confId)
        chiral_list = np.array(Chem.FindMolChiralCenters(mol))
        chiral_centers = [int(x) for x in chiral_list[:, 0]]
        chiral_idx = []

        for atom in mol.GetAtoms():
            if atom.GetBoolProp('main_chain') and atom.GetIdx() in chiral_centers:
                chiral_idx.append(int(atom.GetChiralTag()))
        chiral_idx = np.array(chiral_idx)

        tac_list1 = np.where(np.array(tac_array)[:len(chiral_idx)], 1, 2)
        tac_list2 = np.where(np.array(tac_array)[:len(chiral_idx)], 2, 1)

        if (chiral_idx == tac_list1).all() or (chiral_idx == tac_list2).all():
            check = True

    elif tac == 'isotactic' and tacticity == 'isotactic':
        check = True

    elif tac == 'syndiotactic' and tacticity == 'syndiotactic':
        check = True


    return check


def polymer_stats(mol, df=False, join=False):
    """
    poly.polymer_stats

    Calculate statistics of polymers

    Args:
        mol: RDkit Mol object

    Optional args:
        df: Data output type, True: pandas.DataFrame, False: dict  (boolean)

    Returns:
        dict or pandas.DataFrame
    """

    molcount = utils.count_mols(mol)
    polymer_chains = Chem.GetMolFrags(mol, asMols=True)
    natom = np.array([chain.GetNumAtoms() for chain in polymer_chains])
    molweight = np.array([molecular_weight(chain, strict=True) for chain in polymer_chains])

    poly_stats = {
        'n_mol': molcount,
        'n_atom': natom if not df and not join else '/'.join([str(n) for n in natom]),
        'n_atom_mean': np.mean(natom),
        'n_atom_var': np.var(natom),
        'mol_weight': molweight if not df and not join else '/'.join([str(n) for n in molweight]),
        'Mn': np.mean(molweight),
        'Mw': np.sum(molweight**2)/np.sum(molweight),
        'Mw/Mn': np.sum(molweight**2)/np.sum(molweight)/np.mean(molweight)
    }

    mol.SetIntProp('n_mol', molcount)
    mol.SetDoubleProp('n_atom_mean', poly_stats['n_atom_mean'])
    mol.SetDoubleProp('n_atom_var', poly_stats['n_atom_var'])
    mol.SetDoubleProp('Mn', poly_stats['Mn'])
    mol.SetDoubleProp('Mw', poly_stats['Mw'])
    mol.SetDoubleProp('Mw/Mn', poly_stats['Mw/Mn'])

    return poly_stats if not df else pd.DataFrame(poly_stats, index=[0])


##########################################################
# Utility function for SMILES of polymers
##########################################################
def polymerize_MolFromSmiles(smiles, n=2, terminal='C', label=1):
    """
    poly.polymerize_MolFromSmiles

    Generate polimerized RDkit Mol object from SMILES

    Args:
        smiles: SMILES (str)
        n: Polimerization degree (int)
        terminal: SMILES of terminal substruct (str)

    Returns:
        RDkit Mol object
    """

    poly_smiles = make_linearpolymer(smiles, n=n, terminal=terminal, label=label)

    try:
        mol = Chem.MolFromSmiles(poly_smiles)
        mol = Chem.AddHs(mol)
    except Exception:
        mol = None
    
    return mol


def make_linearpolymer(smiles, n=2, terminal='C', label=1):
    """
    poly.make_linearpolymer

    Generate linearpolymer SMILES from monomer SMILES

    Args:
        smiles: SMILES (str)
        n: Polimerization degree (int)
        terminal: SMILES of terminal substruct (str)

    Returns:
        SMILES
    """

    dummy = '[*]'
    dummy_head = '[Nh]'
    dummy_tail = '[Og]'

    smiles_in = smiles
    smiles = utils.star2h(smiles)
    smiles = smiles.replace('[%iH]' % int(label+2), '*')

    if smiles.count('*') != 2:
        utils.radon_print('Illegal number of connecting points in SMILES. %s' % smiles_in, level=2)
        return None

    smiles = smiles.replace('\\', '\\\\')
    smiles_head = re.sub(r'\*', dummy_head, smiles, 1)
    smiles_tail = re.sub(r'\*', dummy_tail, smiles, 1)
    #smiles_tail = re.sub(r'%s\\\\' % dummy_tail, '%s/' % dummy_tail, smiles_tail, 1)
    #smiles_tail = re.sub(r'%s/' % dummy_tail, '%s\\\\' % dummy_tail, smiles_tail, 1)

    try:
        mol_head = Chem.MolFromSmiles(smiles_head)
        mol_tail = Chem.MolFromSmiles(smiles_tail)
        mol_terminal = Chem.MolFromSmiles(terminal)
        mol_dummy = Chem.MolFromSmiles(dummy)
        mol_dummy_tail = Chem.MolFromSmiles(dummy_tail)

        con_point = 1
        for atom in mol_tail.GetAtoms():
            if atom.GetSymbol() == mol_dummy_tail.GetAtomWithIdx(0).GetSymbol():
                con_point = atom.GetNeighbors()[0].GetIdx()
                break
        
        for poly in range(n-1):
            mol_head = Chem.rdmolops.ReplaceSubstructs(mol_head, mol_dummy, mol_tail, replacementConnectionPoint=con_point)[0]
            mol_head = Chem.RWMol(mol_head)
            for atom in mol_head.GetAtoms():
                if atom.GetSymbol() == mol_dummy_tail.GetAtomWithIdx(0).GetSymbol():
                    idx = atom.GetIdx()
                    break
            mol_head.RemoveAtom(idx)
            Chem.SanitizeMol(mol_head)
        
        mol = mol_head.GetMol()
        mol = Chem.rdmolops.ReplaceSubstructs(mol, mol_dummy, mol_terminal, replacementConnectionPoint=0)[0]

        poly_smiles = Chem.MolToSmiles(mol)
        poly_smiles = poly_smiles.replace(dummy_head, terminal)
        poly_smiles = utils.h2star(poly_smiles)

    except Exception:
        utils.radon_print('Cannot transform to polymer from monomer SMILES. %s' % smiles_in, level=2)
        return None

    return poly_smiles


def make_cyclicpolymer(smiles, n=2, return_mol=False, removeHs=False, label=1):
    """
    poly.make_cyclicpolymer

    Generate cyclicpolymer SMILES from monomer SMILES

    Args:
        smiles: SMILES (str)
        n: Polimerization degree (int)
        return_mol: Return Mol object (True) or SMILES strings (False)

    Returns:
        SMILES or RDKit Mol object
    """

    mol = polymerize_MolFromSmiles(smiles, n=n, terminal='*', label=label)
    # mol = utils.mol_from_smiles(smiles, stereochemistry_control=False)
    # mol = polymerize_mols(mol, n)
    if mol is None:
        return None
        
    set_mainchain_flag(mol)
    set_linker_flag(mol)
    head_idx = mol.GetIntProp('head_idx')
    tail_idx = mol.GetIntProp('tail_idx')
    head_ne_idx = mol.GetIntProp('head_ne_idx')
    tail_ne_idx = mol.GetIntProp('tail_ne_idx')

    # Get bond type of deleting bonds
    bd1type = mol.GetBondBetweenAtoms(head_idx, head_ne_idx).GetBondTypeAsDouble()
    bd2type = mol.GetBondBetweenAtoms(tail_idx, tail_ne_idx).GetBondTypeAsDouble()
    
    # Delete linker atoms and bonds
    mol = utils.remove_atom(mol, head_idx, angle_fix=True)
    if tail_idx > head_idx: tail_idx -= 1
    mol = utils.remove_atom(mol, tail_idx, angle_fix=True)

    # Add a new bond
    if mol.GetIntProp('head_ne_idx') > mol.GetIntProp('tail_idx'): head_ne_idx -= 1
    if mol.GetIntProp('head_ne_idx') > mol.GetIntProp('head_idx'): head_ne_idx -= 1
    if mol.GetIntProp('tail_ne_idx') > mol.GetIntProp('tail_idx'): tail_ne_idx -= 1
    if mol.GetIntProp('tail_ne_idx') > mol.GetIntProp('head_idx'): tail_ne_idx -= 1
    if bd1type == 2.0 or bd2type == 2.0:
        mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.DOUBLE, preserve_topology=True)
    elif bd1type == 3.0 or bd2type == 3.0:
        mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.TRIPLE, preserve_topology=True)
    else:
        mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.SINGLE, preserve_topology=True)

    Chem.SanitizeMol(mol)

    if removeHs:
        try:
            mol = Chem.RemoveHs(mol)
        except:
            return None

    if return_mol:
        return mol
    else:
        poly_smiles = Chem.MolToSmiles(mol)
        return poly_smiles


def make_cyclicpolymer_mp(smiles, n=2, return_mol=False, removeHs=False, mp=None):
    """
    poly.make_cyclicpolymer_mp

    Multiprocessing version of make_cyclicpolymer

    Args:
        smiles: SMILES (list, str)
        n: Polimerization degree (int)
        return_mol: Return Mol object (True) or SMILES strings (False)
        mp: Number of process (int)

    Returns:
        List of SMILES or RDKit Mol object
    """
    if mp is None:
        mp = utils.cpu_count()
    
    c = utils.picklable_const()
    args = [[smi, n, return_mol, removeHs, c] for smi in smiles]

    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_make_cyclicpolymer_worker, args)
        res = [r for r in results]

    return res


def _make_cyclicpolymer_worker(args):
    smi, n, return_mol, removeHs, c = args
    utils.restore_const(c)
    res = make_cyclicpolymer(smi, n=n, return_mol=return_mol, removeHs=removeHs)
    if return_mol:
        utils.picklable()
    return res


def substruct_match_mol(pmol, smol, useChirality=False):

    psmi = Chem.MolToSmiles(pmol)
    pmol = make_cyclicpolymer(psmi, 3, return_mol=True)
    if pmol is None:
        return False

    return pmol.HasSubstructMatch(smol, useChirality=useChirality)


def substruct_match_smiles(poly_smiles, sub_smiles, useChirality=False):
    """
    poly.substruct_match_smiles

    Substruct matching of smiles2 in smiles1 as a polymer structure

    Args:
        poly_smiles: polymer SMILES (str)
        sub_smiles: substruct SMILES (str)

    Optional args:
        useChirality: enables the use of stereochemistry in the matching (boolean)

    Returns:
        RDkit Mol object
    """

    pmol = make_cyclicpolymer(poly_smiles, 3, return_mol=True)
    if pmol is None:
        return False
    smol = Chem.MolFromSmarts(sub_smiles)

    return pmol.HasSubstructMatch(smol, useChirality=useChirality)


def substruct_match_smiles_list(smiles, smi_series, mp=None, boolean=False):
    
    if mp is None:
        mp = utils.cpu_count()

    c = utils.picklable_const()
    args = [[smi, smiles, c] for smi in smi_series]

    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_substruct_match_smiles_worker, args)
        res = [r for r in results]
    
    if boolean:
        return res
    else:
        smi_list = smi_series[res].index.values.tolist()
        return smi_list


def _substruct_match_smiles_worker(args):
    smi, smiles, c = args
    utils.restore_const(c)
    return substruct_match_smiles(smi, smiles)


def full_match_mol(mol1, mol2, monomerize=True):
    smiles1 = Chem.MolToSmiles(mol1)
    smiles2 = Chem.MolToSmiles(mol2)

    return full_match_smiles(smiles1, smiles2, monomerize=monomerize)


def full_match_smiles(smiles1, smiles2, monomerize=True):
    """
    poly.full_match_smiles

    Full matching of smiles1 and smiles2 as a polymer structure

    Args:
        smiles1, smiles2: polymer SMILES (str)

    Returns:
        RDkit Mol object
    """
    if monomerize:
        smiles1 = monomerization_smiles(smiles1)
        smiles2 = monomerization_smiles(smiles2)

    try:
        if Chem.MolFromSmiles(smiles1).GetNumAtoms() != Chem.MolFromSmiles(smiles2).GetNumAtoms():
            return False
    except:
        return False

    smi1 = make_cyclicpolymer(smiles1, n=3)
    smi2 = make_cyclicpolymer(smiles2, n=3)
    if smi1 is None or smi2 is None:
        return False

    # Canonicalize
    smi1 = Chem.MolToSmiles(Chem.MolFromSmiles(smi1))
    smi2 = Chem.MolToSmiles(Chem.MolFromSmiles(smi2))

    if smi1 == smi2:
        return True
    else:
        return False


def full_match_smiles_list(smiles, smi_series, mp=None, monomerize=True):
    
    if mp is None:
        mp = utils.cpu_count()

    c = utils.picklable_const()
    args = [[smiles, smi, monomerize, c] for smi in smi_series]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_full_match_smiles_worker, args)
        res = [r for r in results]
    
    smi_list = smi_series[res].index.values.tolist()

    return smi_list


def _full_match_smiles_worker(args):
    smiles, smi, monomerize, c = args
    utils.restore_const(c)
    return full_match_smiles(smiles, smi, monomerize=monomerize)


def full_match_smiles_listself(smi_series, mp=None, monomerize=True):
    
    if mp is None:
        mp = utils.cpu_count()

    idx_list = smi_series.index.tolist()
    c = utils.picklable_const()
    args = [[smi, monomerize, c] for smi in smi_series]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_make_full_match_smiles, args)
        smi_list = [r for r in results]

    result = []
    i = 1
    for idx1, smi1 in tqdm(zip(idx_list, smi_list), total=len(smi_series), desc='[Full match smiles]', disable=const.tqdm_disable):
        match_list = []
        for idx2, smi2 in zip(idx_list[i:], smi_list[i:]):
            if smi1 == smi2:
                match_list.append(idx2)
        if len(match_list) > 0:
            result.append((idx1, match_list))
        i += 1

    return result


def full_match_smiles_listlist(smi_series1, smi_series2, mp=None, monomerize=True):
    
    if mp is None:
        mp = utils.cpu_count()

    idx_list1 = smi_series1.index.tolist()
    idx_list2 = smi_series2.index.tolist()
    c = utils.picklable_const()

    args = [[smi, monomerize, c] for smi in smi_series1]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_make_full_match_smiles, args)
        smi_list1 = [r for r in results]

    args = [[smi, monomerize, c] for smi in smi_series2]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_make_full_match_smiles, args)
        smi_list2 = [r for r in results]

    result = []
    for idx1, smi1 in tqdm(zip(idx_list1, smi_list1), total=len(smi_series1), desc='[Full match smiles]', disable=const.tqdm_disable):
        match_list = []
        for idx2, smi2 in zip(idx_list2, smi_list2):
            if smi1 == smi2:
                match_list.append(idx2)
        if len(match_list) > 0:
            result.append((idx1, match_list))

    return result


def _make_full_match_smiles(args):
    smi, monomerize, c = args
    utils.restore_const(c)

    if monomerize:
        smi = monomerization_smiles(smi)
    smi = make_cyclicpolymer(smi, n=3)
    if smi is None:
        return args[0]

    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except Exception:
        utils.radon_print('Cannot convert to canonical SMILES from %s' % smi, level=2)
        return args[0]
        
    return smi


def ff_test_mp(smi_list, ff, mp=None):
    if mp is None:
        mp = utils.cpu_count()

    c = utils.picklable_const()
    args = [[smi, ff, c] for smi in smi_list]

    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_ff_test_mp_worker, args)
        res = [r for r in results]

    return res


def _ff_test_mp_worker(args):
    smi, ff, c = args
    utils.restore_const(c)

    try:
        mol = polymerize_MolFromSmiles(smi, n=2)
        mol = Chem.AddHs(mol)
        result = ff.ff_assign(mol)
    except:
        result = False
        
    return result


def monomerization_smiles(smiles, min_length=1, label=1):

    smi = utils.star2h(smiles)
    
    if smi.count('[%iH]' % int(label+2)) != 2:
        utils.radon_print('Illegal number of connecting points in SMILES. %s' % smiles, level=2)
        return smiles

    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
    except:
        utils.radon_print('Cannot convert to Mol object from %s' % smiles, level=2)
        return smiles

    set_linker_flag(mol)
    path = list(Chem.GetShortestPath(mol, mol.GetIntProp('head_idx'), mol.GetIntProp('tail_idx')))
    del path[0], path[-1]

    length = len(path)

    for l in range(min_length, int(length/2+1)):
        if length % l == 0:
            bidx = [mol.GetBondBetweenAtoms(path[l*i-1], path[l*i]).GetIdx() for i in range(1, int(length/l))]
            fmol = Chem.FragmentOnBonds(mol, bidx, addDummies=True)
            try:
                RDLogger.DisableLog('rdApp.*')
                fsmi = [Chem.MolToSmiles(Chem.MolFromSmiles(re.sub(r'\[[0-9]+\*\]', '[*]', x).replace('[%iH]' % int(label+2), '[*]'))) for x in Chem.MolToSmiles(fmol).split('.')]
            except:
                RDLogger.EnableLog('rdApp.*')
                continue

            RDLogger.EnableLog('rdApp.*')
            if len(list(set(fsmi))) == 1 and fsmi[0].count('*') == 2:
                csmi = make_linearpolymer(fsmi[0], n=len(fsmi), terminal='*')
                if csmi is not None:
                    try:
                        if Chem.MolToSmiles(Chem.MolFromSmiles(csmi)) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles)):
                            return fsmi[0]
                    except Exception:
                        utils.radon_print('Cannot convert to canonical SMILES from %s' % smiles, level=2)

    return smiles


def extract_mainchain(smiles, label=1):

    main_smi = None

    fsmi = fragmentation_main_side_chain(smiles, label=label)
    if fsmi is None:
        return main_smi

    for s in fsmi:
        if '[%iH]' % int(label+2) in s:
            try:
                main_smi = Chem.MolToSmiles(Chem.MolFromSmiles(s.replace('[%iH]' % int(label+2), '*')))
            except:
                utils.radon_print('Cannot convert to canonical SMILES from %s' % s, level=2)

    return main_smi


def extract_sidechain(smiles, label=1):

    side_smi = []

    fsmi = fragmentation_main_side_chain(smiles, label=1)
    if fsmi is None:
        return side_smi

    for s in fsmi:
        if '[%iH]' % int(label+2) not in s:
            try:
                side_smi.append(Chem.MolToSmiles(Chem.MolFromSmiles(s)))
            except:
                utils.radon_print('Cannot convert to canonical SMILES from %s' % s, level=2)

    return side_smi


def fragmentation_main_side_chain(smiles, label=1):
    smi = utils.star2h(smiles)
    if smi.count('[%iH]' % int(label+2)) != 2:
        utils.radon_print('Illegal number of connecting points in SMILES. %s' % smiles, level=2)
        return None

    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
    except:
        utils.radon_print('Cannot convert to Mol object from %s' % smiles, level=2)
        return None

    set_mainchain_flag(mol)

    for atom in mol.GetAtoms():
        if atom.GetBoolProp('main_chain'):
            for na in atom.GetNeighbors():
                if not na.GetBoolProp('main_chain'):
                    bidx = mol.GetBondBetweenAtoms(atom.GetIdx(), na.GetIdx()).GetIdx()
                    mol = Chem.FragmentOnBonds(mol, [bidx], addDummies=False)

    RDLogger.DisableLog('rdApp.*')

    try:
        fsmi = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in Chem.MolToSmiles(mol).split('.')]
    except:
        utils.radon_print('Cannot convert to fragmented Mol', level=2)
        RDLogger.EnableLog('rdApp.*')
        return None

    RDLogger.EnableLog('rdApp.*')

    return fsmi


def polyinfo_classifier(smi, return_flag=False):
    """
    poly.polyinfo_classifier

    Classifier of polymer structure to 21 class IDs of PoLyInfo

    Args:
        smi: polymer SMILES (str)

    Returns:
        class ID (int)
    """
    class_id = 0
    
    # Definition of SMARTS of each class. [14C] means a carbon atom in main chain
    styrene = ['*-[14C]([14C]-*)c1[c,n][c,n][c,n][c,n][c,n]1']        # P02
    acryl = ['*-[14C]([14C]-*)C(=[O,S])-[O,S,N,n]']                   # P04
    ether = ['[!$(C(=[O,S]))&!$([X4&S](=O)(=O))&!Si][X2&O,X2&o][!$(C(=[O,S]))&!$([X4&S](=O)(=O))&!Si]'] # P07
    thioether = ['[!$(C(=[O,S]))&!Si][X2&S,X2&s][!$(C(=[O,S]))&!Si]'] # P08
    ester = ['[!$(C(=[O,S]))]-[O,S]C(=[O,S])-[!N&!O&!S]']             # P09
    amide = ['[!$(C(=[O,S]))]-NC(=[O,S])-[!N&!O&!S]']                 # P10
    urethane = ['*-NC(=[O,S])[O,S]-*']                                # P11
    urea = ['*-NC(=[O,S])N-*']                                        # P12
    imide = ['[X3&C,X3&c](=[O,S])[X3&N,X3&n][X3&C,X3&c](=[O,S])']     # P13
    anhyd = ['*-C(=[O,S])[O,S]C(=[O,S])-*']                           # P14
    carbonate = ['*-[O,S]C(=[O,S])[O,S]-*']                           # P15
    amine = ['[X3&N,X2&N,X4&N,X3&n,X2&n;!$([N,n][C,c](=[O,S]));!$(N=P)]'] # P16
    silane = ['*-[X4&Si]-*']                                          # P17
    phosphazene = ['*-N=P-*']                                         # P18
    ketone = ['[!N&!O&!S]-C(=[O,S])-[!N&!O&!S]']                      # P19
    sulfon = ['*-[X4&S](=O)(=O)-*', '*-[X3&S](=O)-*']                 # P20
    phenylene = ['*-c1[c,n][c,n][c,n]([c,n]c1)-*', '*-c1[c,n][c,n]([c,n][c,n]c1)-*', '*-c1[c,n]([c,n][c,n][c,n]c1)-*'] # P21

    m_smi = extract_mainchain(smi)
    if m_smi is None:
        m_smi = smi

    mr_mol = make_cyclicpolymer(m_smi, 4, return_mol=True)
    if mr_mol is None:
        if return_flag:
            return class_id, {}
        else:
            return class_id
    
    flag = {
        'PHYC': False,
        'PSTR': False,
        'PVNL': False,
        'PACR': False,
        'PHAL': False,
        'PDIE': False,
        'POXI': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in ether].count(True) > 0,
        'PSUL': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in thioether].count(True) > 0,
        'PEST': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in ester].count(True) > 0,
        'PAMD': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in amide].count(True) > 0,
        'PURT': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in urethane].count(True) > 0,
        'PURA': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in urea].count(True) > 0,
        'PIMD': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in imide].count(True) > 0,
        'PANH': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in anhyd].count(True) > 0,
        'PCBN': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in carbonate].count(True) > 0,
        'PIMN': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in amine].count(True) > 0,
        'PSIL': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in silane].count(True) > 0,
        'PPHS': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in phosphazene].count(True) > 0,
        'PKTN': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in ketone].count(True) > 0,
        'PSFO': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in sulfon].count(True) > 0,
        'PPNL': [mr_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in phenylene].count(True) > 0,
    }
    
    if list(flag.values()).count(True) == 0:
        try:
            m_mol = Chem.MolFromSmiles(m_smi)
            m_mol = Chem.AddHs(m_mol)
        except:
            utils.radon_print('Cannot convert to Mol object from %s' % m_smi, level=2)
            if return_flag:
                return class_id, {}
            else:
                return class_id

        m_nelem = {'H':0, 'C':0, 'hetero':0, 'halogen':0}
        for atom in m_mol.GetAtoms():
            elem = atom.GetSymbol()
            if elem == '*':
                continue
            elif elem in ['H', 'C']:
                m_nelem[elem] += 1
            else:
                m_nelem['hetero'] += 1
                if elem in ['F', 'Cl', 'Br', 'I']:
                    m_nelem['halogen'] += 1

        try:
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
        except:
            utils.radon_print('Cannot convert to Mol object from %s' % smi, level=2)
            if return_flag:
                return class_id, {}
            else:
                return class_id

        nelem = {'H':0, 'C':0, 'hetero':0, 'halogen':0}
        for atom in mol.GetAtoms():
            elem = atom.GetSymbol()
            if elem == '*':
                continue
            elif elem in ['H', 'C']:
                nelem[elem] += 1
            else:
                nelem['hetero'] += 1
                if elem in ['F', 'Cl', 'Br', 'I']:
                    nelem['halogen'] += 1

        ndbond = 0
        nabond = 0
        ntbond = 0
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                ndbond += 1
            elif bond.GetBondTypeAsDouble() == 1.5:
                nabond += 1
            elif bond.GetBondTypeAsDouble() == 3.0:
                ntbond += 1
        nubond = ndbond + nabond + ntbond

        r_mol = make_cyclicpolymer(smi, 4, return_mol=True)
        if r_mol is None:
            if return_flag:
                return class_id, {}
            else:
                return class_id

        for atom in r_mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetBoolProp('main_chain'):
                atom.SetIsotope(14)
        
        flag['PSTR'] = [r_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in styrene].count(True) > 0
        flag['PACR'] = [r_mol.HasSubstructMatch(Chem.MolFromSmarts(sub_smi)) for sub_smi in acryl].count(True) > 0
        flag['PHAL'] = m_nelem['halogen'] > 0 #(m_nelem['halogen'] > 1 or (nelem['halogen'] - m_nelem['halogen']) > 0)
        flag['PDIE'] = nelem['hetero'] == 0 and (ndbond > 0 or ntbond > 0) and nabond == 0
        flag['PHYC'] = nelem['hetero'] == 0 and nubond == 0
        flag['PVNL'] = m_nelem['halogen'] == 0 and (nelem['hetero'] > 0 or nabond > 0) and not flag['PHYC'] and not flag['PDIE']
    
    if flag['PURT']:
        class_id = 11
    elif flag['PURA']:
        class_id = 12
    elif flag['PIMD']:
        class_id = 13
    elif flag['PANH']:
        class_id = 14
    elif flag['PCBN']:
        class_id = 15

    elif flag['PEST']:
        class_id = 9
    elif flag['PAMD']:
        class_id = 10

    elif flag['PSIL']:
        class_id = 17
    elif flag['PPHS']:
        class_id = 18
    elif flag['PKTN']:
        class_id = 19
    elif flag['PSFO']:
        class_id = 20

    elif flag['POXI']:
        class_id = 7
    elif flag['PSUL']:
        class_id = 8
    elif flag['PIMN']:
        class_id = 16        
    elif flag['PPNL']:
        class_id = 21

    elif flag['PSTR']:
        class_id = 2
    elif flag['PACR']:
        class_id = 4
    elif flag['PHAL']:
        class_id = 5
    elif flag['PDIE']:
        class_id = 6
    elif flag['PHYC']:
        class_id = 1
    elif flag['PVNL']:
        class_id = 3
        
    if return_flag:
        return class_id, flag
    else:
        return class_id


def polyinfo_classifier_list(smi_series, return_flag=False, mp=None):

    if mp is None:
        mp = utils.cpu_count()

    c = utils.picklable_const()
    args = [[None, smi, return_flag, c] for smi in smi_series]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_polyinfo_classifier_worker, args)
        res = [r for r in results]

    return res


def polyinfo_classifier_series(smi_series, return_flag=False, mp=None):
    # class_name = [
    #     'PHYC', 'PSTR', 'PVNL', 'PACR', 'PHAL', 'PDIE', 'POXI',
    #     'PSUL', 'PEST', 'PAMD', 'PURT', 'PURA', 'PIMD', 'PANH',
    #     'PCBN', 'PIMN', 'PSIL', 'PPHS', 'PKTN', 'PSFO', 'PPNL',
    # ]

    if mp is None:
        mp = utils.cpu_count()

    c = utils.picklable_const()
    args = [[idx, smi, return_flag, c] for idx, smi in smi_series.items()]
    with confu.ProcessPoolExecutor(max_workers=mp) as executor:
        results = executor.map(_polyinfo_classifier_worker, args)
        res = [r for r in results]

    if return_flag:
        class_dict = []
        for idx, class_id, pcflag in res:
            dict_tmp = {}
            dict_tmp['index'] = idx
            dict_tmp['polymer_class'] = class_id
            for k, v in pcflag.items():
                dict_tmp['class_%s' % k] = v
            class_dict.append(dict_tmp)
        df_class = pd.DataFrame(class_dict).set_index('index')
    else:
        class_dict = []
        for idx, class_id in res:
            dict_tmp = {}
            dict_tmp['index'] = idx
            dict_tmp['polymer_class'] = class_id
            class_dict.append(dict_tmp)
        df_class = pd.DataFrame(class_dict).set_index('index')

    return df_class


def _polyinfo_classifier_worker(args):
    idx, smi, return_flag, c = args
    utils.restore_const(c)
    if return_flag:
        class_id, pcflag = polyinfo_classifier(smi, return_flag=True)
        if idx is None:
            return class_id, pcflag
        else:
            return idx, class_id, pcflag
    else:
        class_id = polyinfo_classifier(smi, return_flag=False)
        if idx is None:
            return class_id
        else:
            return idx, class_id











def branch_polymerization_rw(poly_mol, mols, n, confId=0, n_dist='uniform',
                             copoly_type='random', tacticity='atactic', atac_ratio=0.5, label=None,
                             position=[2], ds=[1.0], _pf=False, **kwargs):
    # poly_mol: RDKit Mol object of a polymer
    # mols: List of RDKit Mol object of substituents
    # positions: List of labelling index on poly to be connected by substituents
    # ds: List of degree of substitution (axis=0: mols, axis=1: positions)
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.branch_polymerization_rw.', level=1)
    
    bpoly = utils.deepcopy_mol(poly_mol)
    
    if type(mols) is Chem.Mol:
        mols = [[mols]]
    elif type(mols) is list:
        for i, m in enumerate(mols):
            if type(m) is not list:
                mols[i] = [m]
        
    if len(mols) != len(ds):
        utils.radon_print('Inconsistency length of mols and ds', level=3)
        return False
    
    if type(ds[0]) is not list:
        ds = np.array([[x for i in range(len(position))] for x in ds])
    else:
        ds = np.array(ds)
        ds_sum = np.sum(ds, axis=0)
        if np.any(ds_sum > 1.0):
            utils.radon_print('Invalid input value for ds. Sum of ds values on each of a substituent position must be <= 1.0', level=3)
            return False
            
    n_position = {}
    for atom in bpoly.GetAtoms():
        if atom.GetSymbol() == 'H':
            l = int(atom.GetIsotope() - 2)
            if l > 0:
                if l in n_position.keys():
                    n_position[l] += 1
                else:
                    n_position[l] = 1

    for j, pos in enumerate(position):
        func_array = []
        for i in range(len(mols)):
            func_array.extend([i for x in range(round(n_position[int(pos)]*ds[i,j]))])
        if len(func_array) < n_position[int(pos)]:
            func_array.extend([None for x in range(n_position[int(pos)] - len(func_array))])
        func_array = np.array(func_array)
        np.random.shuffle(func_array)

        if not _pf:
            if type(n) is int:
                n_array = np.full(n, len(func_array))
            elif type(n) is list and n_dist == 'uniform':
                n_array = np.random.randint(n[0], n[1], size=len(func_array))
            elif type(n) is list and n_dist == 'normal':
                n_array = [round(x) for x in np.random.normal(n[0], n[1], size=len(func_array))]
            else:
                utils.radon_print('Invalid input value for n.', level=3)

        for i, idx in enumerate(func_array):
            if idx is not None:
                utils.radon_print('Side chain generation. Step %i' % (i+1))
                if _pf:
                    m_idx = [0]
                    chi_inv = [False]
                    tactic = None
                else:
                    m_idx = gen_monomer_array(len(mols[idx]), n_array[i], copoly=copoly_type)
                    chi_inv, check_chi = gen_chiral_inv_array(mols[idx], m_idx, tacticity=tacticity, atac_ratio=atac_ratio)
                    if not check_chi:
                        tactic = None
                    else:
                        tactic = tacticity

                # poly.random_walk_polymerization: Low level API for polymerization
                # 1st arg.: List of using repeating units
                # 2nd arg.: Arrangement of indices of repeating units in the polymer chain
                # 3rd arg.: Whether to perform chiral inversion (List of boolean)
                # label: List of label to be used as connecting points
                #        [[head label of mol1, tail label of mol1], [head label of mol2, tail label of mol2], ...]
                bpoly = random_walk_polymerization(mols[idx], m_idx, chi_inv, confId=confId, label=label, init_poly=bpoly,
                                                        label_init=pos, tacticity=tactic, **kwargs)
            else:
                set_linker_flag(bpoly, label=pos)
                bpoly.GetAtomWithIdx(bpoly.GetIntProp('tail_idx')).SetIsotope(0)

    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.branch_polymerization_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    return bpoly


def post_functionalization_rw(poly_mol, mols, confId=0, position=[2], ds=[1.0], **kwargs):
    pfpoly = branch_polymerization_rw(poly_mol, mols, 1, confId=confId, position=position, ds=ds, _pf=True, **kwargs)
    return pfpoly


def calc_n_from_num_atoms(mols, natom, ratio=[1.0], label=1, terminal1=None, terminal2=None,
                          pf={'sub_mols': [], 'position':[2], 'ds':[1.0]}):

    pf_flag = False

    mols = _resolve_mol_list(mols)
    mols = _resolve_mol_list(mols)
    if type(mols) is Chem.Mol:
        mols = [mols]
    elif type(mols) is not list:
        utils.radon_print('Input should be an RDKit Mol object or its List', level=3)
        return None

    if len(mols) != len(ratio):
        utils.radon_print('Inconsistency length of mols and ratio', level=3)
        return None

    ratio = np.array(ratio) / np.sum(ratio)

    if len(pf['sub_mols']) > 0:
        pf_flag = True
        sub_mols = pf['sub_mols']
        position = pf['position']
        ds = pf['ds']

        if len(sub_mols) != len(ds):
            utils.radon_print('Inconsistency length of sub_mols and ds', level=3)
            return False
    
        if type(ds[0]) is not list:
            ds = np.array([[x for i in range(len(position))] for x in ds])
        else:
            ds = np.array(ds)
            ds_sum = np.sum(ds, axis=0)
            if np.any(ds_sum > 1.0):
                utils.radon_print('Invalid input value for ds. Sum of ds values on each of a substituent position must be <= 1.0', level=3)
                return False

        smol_n = []
        for smol in sub_mols:
            new_mol = remove_linker_atoms(smol)
            smol_n.append(new_mol.GetNumAtoms())
        smol_n = np.array(smol_n)

    mol_n = 0
    for i, mol in enumerate(mols):
        new_mol = remove_linker_atoms(mol, label=label)
        n_atom = new_mol.GetNumAtoms()

        if pf_flag:
            for j, pos in enumerate(position):
                pos_n = 0
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'H' and atom.GetIsotope() == pos+2:
                        pos_n += 1

                n_atom += pos_n * np.sum((smol_n-1) * ds[:, j])

        mol_n += n_atom * ratio[i]

    if terminal1 is not None:
        new_ter1 = remove_linker_atoms(terminal1)
        ter1_n = new_ter1.GetNumAtoms()

        if pf_flag:
            for j, pos in enumerate(position):
                pos_n = 0
                for atom in terminal1.GetAtoms():
                    if atom.GetSymbol() == 'H' and atom.GetIsotope() == pos+2:
                        pos_n += 1

                ter1_n += pos_n * np.sum((smol_n-1) * ds[:, j])

    else:
        ter1_n = 0

    if terminal2 is not None:
        new_ter2 = remove_linker_atoms(terminal2)
        ter2_n = new_ter2.GetNumAtoms()

        if pf_flag:
            for j, pos in enumerate(position):
                pos_n = 0
                for atom in terminal2.GetAtoms():
                    if atom.GetSymbol() == 'H' and atom.GetIsotope() == pos+2:
                        pos_n += 1

                ter2_n += pos_n * np.sum((smol_n-1) * ds[:, j])

    elif terminal1 is not None:
        ter2_n = ter1_n
    else:
        ter2_n = 0

    n = round((natom - ter1_n - ter2_n) / mol_n)
    
    return n


def calc_n_from_mol_weight(mols, mw, ratio=[1.0], terminal1=None, terminal2=None,
                           pf={'sub_mols': [], 'position':[2], 'ds':[1.0]}):
    pf_flag = False

    if type(mols) is Chem.Mol:
        mols = [mols]
    elif type(mols) is not list:
        utils.radon_print('Input should be an RDKit Mol object or its List', level=3)
        return None

    if len(mols) != len(ratio):
        utils.radon_print('Inconsistency length of mols and ratio', level=3)
        return None

    ratio = np.array(ratio) / np.sum(ratio)

    if len(pf['sub_mols']) > 0:
        pf_flag = True
        sub_mols = pf['sub_mols']
        position = pf['position']
        ds = pf['ds']

        if len(sub_mols) != len(ds):
            utils.radon_print('Inconsistency length of sub_mols and ds', level=3)
            return False
        
        if type(ds[0]) is not list:
            ds = np.array([[x for i in range(len(position))] for x in ds])
        else:
            ds = np.array(ds)
            ds_sum = np.sum(ds, axis=0)
            if np.any(ds_sum > 1.0):
                utils.radon_print('Invalid input value for ds. Sum of ds values on each of a substituent position must be <= 1.0', level=3)
                return False

        smol_mw = np.array([calc.molecular_weight(smol) for smol in sub_mols])

    mol_mw = 0.0
    for i, mol in enumerate(mols):
        m_mw = calc.molecular_weight(mol)

        if pf_flag:
            for j, pos in enumerate(position):
                pos_n = 0
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'H' and atom.GetIsotope() == pos+2:
                        pos_n += 1

                m_mw += pos_n * np.sum(smol_mw * ds[:, j])

        mol_mw += m_mw * ratio[i]

    if terminal1 is not None:
        ter1_mw = calc.molecular_weight(terminal1)

        if pf_flag:
            for j, pos in enumerate(position):
                pos_n = 0
                for atom in terminal1.GetAtoms():
                    if atom.GetSymbol() == 'H' and atom.GetIsotope() == pos+2:
                        pos_n += 1

                ter1_mw += pos_n * np.sum(smol_mw * ds[:, j])
    else:
        ter1_mw = 0.0

    if terminal2 is not None:
        ter2_mw = calc.molecular_weight(terminal2)

        if pf_flag:
            for j, pos in enumerate(position):
                pos_n = 0
                for atom in terminal2.GetAtoms():
                    if atom.GetSymbol() == 'H' and atom.GetIsotope() == pos+2:
                        pos_n += 1

                ter2_mw += pos_n * np.sum(smol_mw * ds[:, j])
    elif terminal1 is not None:
        ter2_mw = ter1_mw
    else:
        ter2_mw = 0.0

    n = round((mw - ter1_mw - ter2_mw) / mol_mw)
    
    return n













# For ladder polymer extension
def connect_mols_dev(mol1, mol2, bond_length=1.5, dihedral=np.pi, random_rot=False, dih_type='monomer', set_linker=True, label1=1, label2=1,
                headhead=False, tailtail=False, confId1=0, confId2=0, res_name_1='RU0', res_name_2='RU0', ladder=False, ladder_label1=2, ladder_label2=3):
    """
    poly.connect_mols

    Connect tail atom in mol1 to head atom in mol2

    Args:
        mol1, mol2: RDkit Mol object (requiring AddHs and 3D position)

    Optional args:
        bond_length: Bond length of connecting bond (float, angstrome)
        random_rot: Dihedral angle around connecting bond is rotated randomly (boolean)
        dihedral: Dihedral angle around connecting bond (float, radian)
        dih_type: Definition type of dihedral angle (str; monomer, or bond)
        headhead: Connect with head-to-head
        tailtail: Connect with tail-to-tail
        confId1, confId2: Target conformer ID of mol1 and mol2
        res_name_1, res_name_2: Set residue name of PDB

    Returns:
        RDkit Mol object
    """

    if mol1 is None: return mol2
    if mol2 is None: return mol1

    # Initialize
    if headhead: set_linker_flag(mol1, reverse=True, label=label1)
    elif set_linker: set_linker_flag(mol1, label=label1)
    if tailtail and not headhead: set_linker_flag(mol2, reverse=True, label=label2)
    elif set_linker: set_linker_flag(mol2, label=label2)

    if mol1.GetIntProp('tail_idx') < 0 or mol1.GetIntProp('tail_ne_idx') < 0:
        utils.radon_print('Cannot connect_mols because mol1 does not have a tail linker atom.', level=2)
        return mol1
    elif mol2.GetIntProp('head_idx') < 0 or mol2.GetIntProp('head_ne_idx') < 0:
        utils.radon_print('Cannot connect_mols because mol2 does not have a head linker atom.', level=2)
        return mol1

    mol1_n = mol1.GetNumAtoms()
    mol1_coord = mol1.GetConformer(confId1).GetPositions()
    mol2_coord = mol2.GetConformer(confId2).GetPositions()
    mol1_tail_vec = mol1_coord[mol1.GetIntProp('tail_ne_idx')] - mol1_coord[mol1.GetIntProp('tail_idx')]
    mol2_head_vec = mol2_coord[mol2.GetIntProp('head_ne_idx')] - mol2_coord[mol2.GetIntProp('head_idx')]
    charge_list = ['AtomicCharge', '_GasteigerCharge', 'RESP', 'RESP2', 'ESP', 'MullikenCharge', 'LowdinCharge']
    #bd1type = mol1.GetBondBetweenAtoms(mol1.GetIntProp('tail_idx'), mol1.GetIntProp('tail_ne_idx')).GetBondTypeAsDouble()
    #bd2type = mol2.GetBondBetweenAtoms(mol2.GetIntProp('head_idx'), mol2.GetIntProp('head_ne_idx')).GetBondTypeAsDouble()

    # Rotation mol2 to align bond vectors of head and tail
    angle = calc.angle_vec(mol1_tail_vec, mol2_head_vec, rad=True)
    center = mol2_coord[mol2.GetIntProp('head_ne_idx')]
    if angle == 0:
        mol2_coord_rot = (mol2_coord - center) * -1 + center
    elif angle == np.pi:
        mol2_coord_rot = mol2_coord
    else:
        vcross = np.cross(mol1_tail_vec, mol2_head_vec)
        mol2_coord_rot = calc.rotate_rod(mol2_coord, vcross, (np.pi-angle), center=center)

    # Translation mol2
    trans = mol1_coord[mol1.GetIntProp('tail_ne_idx')] - ( bond_length * mol1_tail_vec / np.linalg.norm(mol1_tail_vec) )
    mol2_coord_rot = mol2_coord_rot + trans - mol2_coord_rot[mol2.GetIntProp('head_ne_idx')]

    # Rotation mol2 around new bond
    if ladder:
        head_ladder_idx, head_ne_ladder_idx = None, None
        tail_ladder_idx, tail_ne_ladder_idx = None, None

        for atom in mol1.GetAtoms():
            if atom.GetSymbol() == 'H' and atom.GetIsotope() == ladder_label1 + 2:
                tail_ladder_idx = atom.GetIdx()                
                tail_ne_ladder_idx = atom.GetNeighbors()[0].GetIdx()
                break
        if tail_ladder_idx is None:
            utils.radon_print('Can not find connection tail atom for a ladder polymer. ladder_label1=%i' % int(ladder_label1), level=3)
        for atom in mol2.GetAtoms():
            if atom.GetSymbol() == 'H' and atom.GetIsotope() == ladder_label2 + 2:
                head_ladder_idx = atom.GetIdx()
                head_ne_ladder_idx = atom.GetNeighbors()[0].GetIdx()
                break
        if head_ladder_idx is None:
            utils.radon_print('Can not find connection head atom for a ladder polymer. ladder_label2=%i' % int(ladder_label2), level=3)

        dih = calc.dihedral_coord(mol1_coord[tail_ladder_idx], mol1_coord[mol1.GetIntProp('tail_ne_idx')],
                                  mol2_coord_rot[mol2.GetIntProp('head_ne_idx')], mol2_coord_rot[head_ladder_idx], rad=True)
        dihedral = np.random.uniform(-np.pi, np.pi) / 6

    elif random_rot == True:
        dih = np.random.uniform(-np.pi, np.pi)
        
    else:
        if dih_type == 'monomer':
            dih = calc.dihedral_coord(mol1_coord[mol1.GetIntProp('head_idx')], mol1_coord[mol1.GetIntProp('tail_ne_idx')],
                                      mol2_coord_rot[mol2.GetIntProp('head_ne_idx')], mol2_coord_rot[mol2.GetIntProp('tail_idx')], rad=True)
        elif dih_type == 'bond':
            path1 = Chem.GetShortestPath(mol1, mol1.GetIntProp('head_idx'), mol1.GetIntProp('tail_idx'))
            path2 = Chem.GetShortestPath(mol2, mol2.GetIntProp('head_idx'), mol2.GetIntProp('tail_idx'))
            dih = calc.dihedral_coord(mol1_coord[path1[-3]], mol1_coord[path1[-2]],
                                      mol2_coord_rot[path2[1]], mol2_coord_rot[path2[2]], rad=True)
        else:
            utils.radon_print('Illegal option of dih_type=%s.' % str(dih_type), level=3)
    mol2_coord_rot = calc.rotate_rod(mol2_coord_rot, -mol1_tail_vec, (dihedral-dih), center=mol2_coord_rot[mol2.GetIntProp('head_ne_idx')])

    # Combining mol1 and mol2
    mol = combine_mols(mol1, mol2, res_name_1=res_name_1, res_name_2=res_name_2)

    # Set atomic coordinate
    for i in range(mol2.GetNumAtoms()):
        mol.GetConformer(0).SetAtomPosition(i+mol1_n, Geom.Point3D(mol2_coord_rot[i, 0], mol2_coord_rot[i, 1], mol2_coord_rot[i, 2]))

    # Set atomic charge
    for charge in charge_list:
        if not mol.GetAtomWithIdx(0).HasProp(charge) or not mol.GetAtomWithIdx(mol.GetNumAtoms()-1).HasProp(charge):
            continue
        head_charge = mol.GetAtomWithIdx(mol2.GetIntProp('head_idx') + mol1_n).GetDoubleProp(charge)
        head_ne_charge = mol.GetAtomWithIdx(mol2.GetIntProp('head_ne_idx') + mol1_n).GetDoubleProp(charge)
        tail_charge = mol.GetAtomWithIdx(mol1.GetIntProp('tail_idx')).GetDoubleProp(charge)
        tail_ne_charge = mol.GetAtomWithIdx(mol1.GetIntProp('tail_ne_idx')).GetDoubleProp(charge)
        mol.GetAtomWithIdx(mol2.GetIntProp('head_ne_idx') + mol1_n).SetDoubleProp(charge, head_charge+head_ne_charge)
        mol.GetAtomWithIdx(mol1.GetIntProp('tail_ne_idx')).SetDoubleProp(charge, tail_charge+tail_ne_charge)
        if ladder:
            head_ladder_charge = mol.GetAtomWithIdx(head_ladder_idx + mol1_n).GetDoubleProp(charge)
            head_ne_ladder_charge = mol.GetAtomWithIdx(head_ne_ladder_idx + mol1_n).GetDoubleProp(charge)
            tail_ladder_charge = mol.GetAtomWithIdx(tail_ladder_idx).GetDoubleProp(charge)
            tail_ne_ladder_charge = mol.GetAtomWithIdx(tail_ne_ladder_idx).GetDoubleProp(charge)
            mol.GetAtomWithIdx(head_ne_ladder_idx + mol1_n).SetDoubleProp(charge, head_ladder_charge+head_ne_ladder_charge)
            mol.GetAtomWithIdx(tail_ne_ladder_idx).SetDoubleProp(charge, tail_ladder_charge+tail_ne_ladder_charge)

    # Delete linker atoms and bonds
    del_idx1 = mol1.GetIntProp('tail_idx')
    del_idx2 = mol2.GetIntProp('head_idx') + mol1_n - 1
    mol = utils.remove_atom(mol, del_idx1, angle_fix=True)
    mol = utils.remove_atom(mol, del_idx2, angle_fix=True)

    # Add a new bond
    tail_ne_idx = mol1.GetIntProp('tail_ne_idx')
    head_ne_idx = mol2.GetIntProp('head_ne_idx') + mol1_n - 1
    if del_idx1 < tail_ne_idx: tail_ne_idx -= 1
    if del_idx2 < head_ne_idx: head_ne_idx -= 1
    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.SINGLE, preserve_topology=True)
    new_bond = mol.GetBondBetweenAtoms(tail_ne_idx, head_ne_idx)
    new_bond.SetBoolProp('new_bond', True)

    if ladder:
        del_ladder_idx1 = tail_ladder_idx
        del_ladder_idx2 = head_ladder_idx + mol1_n - 2
        if del_idx1 < del_ladder_idx1: del_ladder_idx1 -= 1
        if del_idx2 < del_ladder_idx2: del_ladder_idx2 -= 1
        mol = utils.remove_atom(mol, del_ladder_idx1, angle_fix=True)
        mol = utils.remove_atom(mol, del_ladder_idx2, angle_fix=True)

        tail_ladder_new_idx = tail_ne_ladder_idx
        head_ladder_new_idx = head_ne_ladder_idx + mol1_n - 2
        if del_idx1 < tail_ladder_new_idx: tail_ladder_new_idx -= 1
        if del_idx2 < head_ladder_new_idx: head_ladder_new_idx -= 1
        if del_ladder_idx1 < tail_ladder_new_idx: tail_ladder_new_idx -= 1
        if del_ladder_idx2 < head_ladder_new_idx: head_ladder_new_idx -= 1
        mol = utils.add_bond(mol, tail_ladder_new_idx, head_ladder_new_idx, order=Chem.rdchem.BondType.SINGLE, preserve_topology=True)
        new_bond = mol.GetBondBetweenAtoms(tail_ladder_new_idx, head_ladder_new_idx)
        new_bond.SetBoolProp('new_bond', True)

    #if bd1type == 2.0 or bd2type == 2.0:
    #    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.DOUBLE)
    #elif bd1type == 3.0 or bd2type == 3.0:
    #    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.TRIPLE)
    #else:
    #    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.SINGLE)

    # Finalize
    Chem.SanitizeMol(mol)
    #set_linker_flag(mol)

    # Update linker_flag
    mol.GetAtomWithIdx(tail_ne_idx).SetBoolProp('tail_neighbor', False)
    mol.GetAtomWithIdx(head_ne_idx).SetBoolProp('head_neighbor', False)

    head_idx = mol1.GetIntProp('head_idx')
    if del_idx1 < head_idx: head_idx -= 1

    head_ne_idx = mol1.GetIntProp('head_ne_idx')
    if del_idx1 < head_ne_idx: head_ne_idx -= 1

    tail_idx = mol2.GetIntProp('tail_idx') + mol1_n - 1
    if del_idx2 < tail_idx: tail_idx -= 1

    tail_ne_idx = mol2.GetIntProp('tail_ne_idx') + mol1_n - 1
    if del_idx2 < tail_ne_idx: tail_ne_idx -= 1

    mol.SetIntProp('head_idx', head_idx)
    mol.SetIntProp('head_ne_idx', head_ne_idx)
    mol.SetIntProp('tail_idx', tail_idx)
    mol.SetIntProp('tail_ne_idx', tail_ne_idx)

    return mol



def random_walk_polymerization_dev(mols, m_idx, chi_inv, start_num=0, init_poly=None, headhead=False, confId=0,
            dist_min=0.7, retry=100, rollback=5, rollback_shaking=False, retry_step=200, retry_opt_step=0, tacticity=None,
            res_name_init='INI', res_name=None, label=None, label_init=1, ff=None, work_dir=None, omp=1, mpi=0, gpu=0, mp_idx=None,
            ladder=False, ladder_label1=2, ladder_label2=3):
    """
    poly.random_walk_polymerization

    Polymerization of RDkit Mol object by self-avoiding random walk

    Args:
        mols: list of RDkit Mol object
        m_idx: Input array of repeating units by index number of mols
        chi_inv: Input boolean array of chiral inversion

    Optional args:
        start_num: Index number of m_idx of starting point
        init_poly: Perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry random_walk_polymerization (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        work_dir: Work directory path of external minimizer (str, requiring when opt is external minimizer)
        ff: Force field object (requiring when opt is external minimizer)
        omp: Number of threads of OpenMP in external minimizer (int)
        mpi: Number of MPI process in external minimizer (int)
        gpu: Number of GPU in external minimizer (int)

    Returns:
        Rdkit Mol object
    """
    utils.radon_print('Start poly.random_walk_polymerization.')

    mols = _resolve_mol_list(mols)
    init_poly = _resolve_mol_like(init_poly) if init_poly is not None else None
    rst_flag = _effective_restart_flag(work_dir, restart)
    payload = _rw_payload('random_walk_polymerization', smiles=[Chem.MolToSmiles(m, isomericSmiles=True) for m in mols], m_idx=list(np.asarray(m_idx).tolist()), chi_inv=[bool(x) for x in chi_inv], start_num=int(start_num), headhead=bool(headhead), tacticity=tacticity, label=label, label_init=label_init, init_poly_smiles=(Chem.MolToSmiles(init_poly, isomericSmiles=True) if isinstance(init_poly, Chem.Mol) else None))
    if rst_flag:
        cached = _rw_load(work_dir, 'random_walk_polymerization', payload)
        if cached is not None:
            return cached

    if len(m_idx) != len(chi_inv):
        utils.radon_print('Inconsistency length of m_idx and chi_inv', level=3)
    if len(mols) <= max(m_idx):
        utils.radon_print('Illegal index number was found in m_idx', level=3)

    mols_copy = []
    mols_inv = []
    has_ring = False
    retry_flag = False
    tri_coord = None
    bond_coord = None    
    poly = None
    poly_copy = [None]

    if res_name is None:
        res_name = ['RU%s' % const.pdb_id[i] for i in range(len(mols))]
    else:
        if len(mols) != len(res_name):
            utils.radon_print('Inconsistency length of mols and res_name', level=3)

    if label is None:
        label = [[1, 1] for x in range(len(mols))]
    else:
        if len(mols) != len(label):
            utils.radon_print('Inconsistency length of mols and label', level=3)

    if type(init_poly) == Chem.Mol:
        poly = utils.deepcopy_mol(init_poly)
        set_linker_flag(poly, label=label_init)
        poly_copy = []
        sssr_tmp = Chem.GetSSSR(poly)
        if type(sssr_tmp) is int:
            if sssr_tmp > 0:
                has_ring = True
        elif len(sssr_tmp) > 0:  # For RDKit version >= 2022.09
            has_ring = True

    for i, mol in enumerate(mols):
        set_linker_flag(mol, label=label[i][0])
        mols_copy.append(utils.deepcopy_mol(mol))
        mols_inv.append(calc.mirror_inversion_mol(mol))
        sssr_tmp = Chem.GetSSSR(mol)
        if type(sssr_tmp) is int:
            if sssr_tmp > 0:
                has_ring = True
        elif len(sssr_tmp) > 0:  # For RDKit version >= 2022.09
            has_ring = True

    for i in tqdm(range(start_num, len(m_idx)), desc='[Polymerization]', disable=const.tqdm_disable):
        dmat = None
    
        if chi_inv[i]:
            mol_c = mols_inv[m_idx[i]]
        else:
            mol_c = mols_copy[m_idx[i]]

        if i == 0:
            res_name_1 = res_name_init
        else:
            res_name_1 = res_name[m_idx[i-1]]

        if type(poly) is Chem.Mol:
            poly_copy.append(utils.deepcopy_mol(poly))
        else:
            poly_copy.append(utils.deepcopy_mol(mol_c))

        if len(poly_copy) > rollback:
            del poly_copy[0]

        for r in range(retry_step*(1+retry_opt_step)):
            check_3d = False
            if i > 0:
                label1 = label[m_idx[i-1]][1]
            elif type(init_poly) == Chem.Mol:
                label1 = label_init
            else:
                label1 = 1

            if headhead and i % 2 == 0:
                poly = connect_mols_dev(poly, mol_c, tailtail=True, random_rot=True, set_linker=True,
                            confId2=confId, res_name_1=res_name_1, res_name_2=res_name[m_idx[i]],
                            label1=label1, label2=label[m_idx[i]][1],
                            ladder=ladder, ladder_label1=ladder_label1, ladder_label2=ladder_label2)
            else:
                poly = connect_mols_dev(poly, mol_c, random_rot=True, set_linker=True,
                            confId2=confId, res_name_1=res_name_1, res_name_2=res_name[m_idx[i]],
                            label1=label1, label2=label[m_idx[i]][0],
                            ladder=ladder, ladder_label1=ladder_label1, ladder_label2=ladder_label2)

            if i == 0 and init_poly is None:
                break

            if dmat is None and dist_min > 1.0:
                # This deepcopy avoids a bug of RDKit
                dmat = Chem.GetDistanceMatrix(utils.deepcopy_mol(poly))

                if r % retry_step == 0 and r > 0:
                    # Periodically "shake" the geometry to escape poor local minima.
                    # If MD is available, do a short high-T MD; otherwise fall back to a quick RDKit MMFF optimization.
                    if MD_avail:
                        utils.radon_print('Molecular geometry shaking by a short time and high temperature MD simulation')
                        if ff is None:
                            ff = GAFF2_mod()

                        # Robust charge-preservation: if *any* atom already has AtomicCharge, treat the molecule
                        # as pre-charged and do NOT overwrite charges.
                        try:
                            has_q = any(a.HasProp('AtomicCharge') for a in poly.GetAtoms())
                        except Exception:
                            has_q = False

                        if has_q:
                            ff.ff_assign(poly)
                        else:
                            ff.ff_assign(poly, charge='gasteiger')

                        # Always perform the MD shake after FF assignment.
                        poly, _ = md.quick_rw(poly, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, idx=mp_idx)
                    else:
                        utils.radon_print('Molecular geometry optimization by RDKit')
                        AllChem.MMFFOptimizeMolecule(
                            poly,
                            maxIters=50,
                            mmffVariant='MMFF94s',
                            nonBondedThresh=3.0,
                            confId=0,
                        )

                    check_3d = check_3d_structure_poly(
                        poly,
                        mol_c,
                        dmat,
                        dist_min=dist_min,
                        check_bond_length=True,
                        tacticity=tacticity,
                    )
                    tri_coord = None
                    bond_coord = None
            else:
                check_3d = check_3d_structure_poly(poly, mol_c, dmat, dist_min=dist_min, check_bond_length=False)

            if check_3d and has_ring:
                check_3d, tri_coord_new, bond_coord_new = check_3d_bond_ring_intersection(poly, mon=mol_c,
                                                                    tri_coord=tri_coord, bond_coord=bond_coord)

            if check_3d:
                if has_ring:
                    tri_coord = tri_coord_new
                    bond_coord = bond_coord_new
                break
            elif r < retry_step * (1 + retry_opt_step) - 1:
                poly = utils.deepcopy_mol(poly_copy[-1]) if type(poly_copy[-1]) is Chem.Mol else None
                if r == 0 or (r+1) % 100 == 0:
                    utils.radon_print('Retry random walk step %i, %i/%i' % (i+1, r+1, retry_step*(1+retry_opt_step)))
            else:
                retry_flag = True
                utils.radon_print(
                    'Reached maximum number of retrying step in random walk step %i of poly.random_walk_polymerization.' % (i+1),
                    level=1)

        if retry_flag: break

    if retry_flag:
        if retry <= 0:
            utils.radon_print(
                'poly.random_walk_polymerization is failure because reached maximum number of rollback times in random walk step %i.' % (i+1),
                level=3)
        else:
            utils.radon_print(
                'Retry poly.random_walk_polymerization and rollback %i steps. Remaining %i times.' % (len(poly_copy), retry),
                level=1)
            retry -= 1
            start_num = i-len(poly_copy)+1
            if start_num > 0:
                label_init = label[m_idx[start_num-1]][1]
            rb_poly = poly_copy[0]

            if MD_avail and rollback_shaking and type(rb_poly) is Chem.Mol:
                utils.radon_print('Molecular geometry shaking by a short time and high temperature MD simulation')
                if ff is None:
                    ff = GAFF2_mod()
                # Use a robust check: if *any* atom already has AtomicCharge, preserve it.
                has_q = False
                try:
                    has_q = any(a.HasProp('AtomicCharge') for a in rb_poly.GetAtoms())
                except Exception:
                    has_q = False
                if has_q:
                    ff.ff_assign(rb_poly)
                else:
                    ff.ff_assign(rb_poly, charge='gasteiger')
                rb_poly, _ = md.quick_rw(rb_poly, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu)

            poly = random_walk_polymerization_dev(
                mols, m_idx, chi_inv, start_num=start_num, init_poly=rb_poly, headhead=headhead, confId=confId,
                dist_min=dist_min, retry=retry, rollback=rollback, retry_step=retry_step, retry_opt_step=retry_opt_step, tacticity=tacticity,
                res_name_init=res_name_init, res_name=res_name, label=label, label_init=label_init,
                ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu,
                ladder=ladder, ladder_label1=ladder_label1, ladder_label2=ladder_label2
            )
            
    return poly



def polymerize_ladder_rw(mol, n, init_poly=None, headhead=False, confId=0, tacticity='atactic', atac_ratio=0.5,
            dist_min=0.7, retry=100, rollback=5, rollback_shaking=False, retry_step=200, retry_opt_step=0, ter1=None, ter2=None,
            label=None, label_ter1=1, label_ter2=1, res_name='RU0', ff=None, work_dir=None, omp=0, mpi=1, gpu=0, mp_idx=None,
            ladder_label1=2, ladder_label2=3):
    """
    poly.polymerize_ladder_rw

    Homo-polymerization of RDkit Mol object by self-avoiding random walk for ladder polymers

    Args:
        mol: RDkit Mol object
        n: Polymerization degree (int)

    Optional args:
        init_poly: polymerize_rw perform additional polymerization for init_poly (RDkit Mol object)
        headhead: Connect monomer unit by head-to-head
        confId: Target conformer ID
        tacticity: isotactic, syndiotactic, or atactic
        atac_ratio: Chiral inversion ration for atactic polymer
        dist_min: (float, angstrom)
        retry: Number of retry for this function when generating unsuitable structure (int)
        rollback: Number of rollback step when retry polymerize_rw (int)
        retry_step: Number of retry for a random-walk step when generating unsuitable structure (int)
        retry_opt_step: Number of retry for a random-walk step with optimization when generating unsuitable structure (int)
        work_dir: Work directory path of external minimizer (str, requiring when opt is external minimizer)
        ff: Force field object (requiring when opt is external minimizer)
        omp: Number of threads of OpenMP in external minimizer (int)
        mpi: Number of MPI process in external minimizer (int)
        gpu: Number of GPU in external minimizer (int)

    Returns:
        Rdkit Mol object
    """
    dt1 = datetime.datetime.now()
    utils.radon_print('Start poly.polymerize_rw.', level=1)

    m_idx = gen_monomer_array(1, n)
    chi_inv, check_chi = gen_chiral_inv_array([mol], m_idx, init_poly=init_poly, tacticity=tacticity, atac_ratio=atac_ratio)
    if not check_chi:
        tacticity = None

    if type(ter1) is Chem.Mol:
        if ter2 is None:
            ter2 = ter1
        mols = [mol, ter1, ter2]
        res_name = [res_name, 'TU0', 'TU1']
        m_idx = [1, *m_idx, 2]
        chi_inv = [False, *chi_inv, False]
        if label is None:
            label = [1, 1]
        label = [label, [label_ter1, label_ter1], [label_ter2, label_ter2]]
    else:
        mols = [mol]
        res_name = [res_name]
        if label is not None:
            label = [label]

    poly = random_walk_polymerization_dev(
        mols, m_idx, chi_inv, start_num=0, init_poly=init_poly, headhead=headhead, confId=confId,
        dist_min=dist_min, retry=retry, rollback=rollback, rollback_shaking=rollback_shaking, retry_step=retry_step, retry_opt_step=retry_opt_step,
        tacticity=tacticity, res_name=res_name, label=label, ff=ff, work_dir=work_dir, omp=omp, mpi=mpi, gpu=gpu, mp_idx=mp_idx,
        ladder=True, ladder_label1=ladder_label1, ladder_label2=ladder_label2
    )

    if type(ter1) is Chem.Mol:
        set_terminal_idx(poly)
    dt2 = datetime.datetime.now()
    utils.radon_print('Normal termination of poly.polymerize_rw. Elapsed time = %s' % str(dt2-dt1), level=1)

    return poly

def mol_from_amino_residues(residues):

    mols = [utils.mol_from_pdb(str(core_data_path("pdb", res + ".pdb")), charge=True)
            for res in residues]

    mol = block_copolymerize_mols(mols, 1, tacticity='isotactic')
    
    return mol
