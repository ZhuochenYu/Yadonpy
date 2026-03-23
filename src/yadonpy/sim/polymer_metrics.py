from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..gmx.topology import MoleculeType, SystemTopology, parse_system_top


def _is_h(atomname: str, atomtype: str) -> bool:
    an = str(atomname or '').strip().upper()
    at = str(atomtype or '').strip().lower()
    return an.startswith('H') or at.startswith('h')


@dataclass
class PolymerChainSpec:
    moltype: str
    chain_index: int
    atom_indices_0: np.ndarray
    backbone_indices_0: np.ndarray


def polymer_moltypes_from_meta(system_meta_path: Path) -> List[str]:
    try:
        meta = json.loads(Path(system_meta_path).read_text(encoding='utf-8'))
    except Exception:
        return []
    out: List[str] = []
    for sp in meta.get('species', []) or []:
        smi = str(sp.get('smiles') or '')
        kind = str(sp.get('kind') or '')
        mt = sp.get('moltype') or sp.get('mol_name') or sp.get('mol_id')
        if not mt:
            continue
        if kind.lower() == 'polymer' or '*' in smi:
            out.append(str(mt))
    return sorted(set(out))


def _adjacency(mt: MoleculeType, *, heavy_only: bool = True) -> Dict[int, set[int]]:
    adj: Dict[int, set[int]] = {}
    heavy = {i for i, (an, at) in enumerate(zip(mt.atomnames, mt.atomtypes), start=1) if not _is_h(an, at)}
    for ai, aj in mt.bonds:
        if heavy_only and (ai not in heavy or aj not in heavy):
            continue
        adj.setdefault(ai, set()).add(aj)
        adj.setdefault(aj, set()).add(ai)
    if not adj and heavy_only:
        # fallback: all atoms in order when explicit bonds are unavailable
        for i in heavy:
            adj.setdefault(i, set())
    return adj


def _farthest_with_parent(start: int, adj: Dict[int, set[int]]) -> tuple[int, Dict[int, Optional[int]], Dict[int, int]]:
    parent: Dict[int, Optional[int]] = {start: None}
    dist: Dict[int, int] = {start: 0}
    q = [start]
    for node in q:
        for nb in sorted(adj.get(node, ())):
            if nb in parent:
                continue
            parent[nb] = node
            dist[nb] = dist[node] + 1
            q.append(nb)
    far = max(dist, key=lambda k: dist[k])
    return far, parent, dist


def _backbone_path(mt: MoleculeType) -> List[int]:
    adj = _adjacency(mt, heavy_only=True)
    nodes = sorted(adj)
    if not nodes:
        # fallback to all non-h atoms in declared order
        return [i for i, (an, at) in enumerate(zip(mt.atomnames, mt.atomtypes), start=1) if not _is_h(an, at)] or list(range(1, mt.natoms + 1))
    start = nodes[0]
    u, _, _ = _farthest_with_parent(start, adj)
    v, parent, _ = _farthest_with_parent(u, adj)
    path = [v]
    while path[-1] != u and parent.get(path[-1]) is not None:
        path.append(parent[path[-1]])
    path.reverse()
    if len(path) >= 2:
        return path
    return nodes


def build_polymer_chain_specs(top: SystemTopology, system_meta_path: Path) -> Dict[str, List[PolymerChainSpec]]:
    polymer_moltypes = set(polymer_moltypes_from_meta(system_meta_path))
    specs: Dict[str, List[PolymerChainSpec]] = {mt: [] for mt in polymer_moltypes}
    current = 0
    chain_counter: Dict[str, int] = {mt: 0 for mt in polymer_moltypes}
    for molname, count in top.molecules:
        mt = top.moleculetypes.get(molname)
        natoms = int(mt.natoms if mt is not None else 0)
        for _ in range(int(count)):
            if molname in polymer_moltypes and mt is not None and natoms > 0:
                atom_idx = np.arange(current, current + natoms, dtype=int)
                bb_local_1 = _backbone_path(mt)
                bb_local_0 = np.asarray([current + i - 1 for i in bb_local_1], dtype=int)
                specs[molname].append(
                    PolymerChainSpec(
                        moltype=str(molname),
                        chain_index=int(chain_counter[molname]),
                        atom_indices_0=atom_idx,
                        backbone_indices_0=bb_local_0,
                    )
                )
                chain_counter[molname] += 1
            current += natoms
    return {k: v for k, v in specs.items() if v}


def _safe_mean_std(arr: List[float]) -> Dict[str, Optional[float]]:
    if not arr:
        return {'mean': None, 'std': None, 'n': 0}
    vals = np.asarray(arr, dtype=float)
    return {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        'n': int(vals.size),
    }


def _accumulate_corr_from_vectors(vec: np.ndarray, out: Dict[str, Any]) -> None:
    # vec: (n_frames, n_bonds, 3) minimum-image bond vectors ordered along backbone
    if vec.shape[1] < 2:
        return
    bl = np.linalg.norm(vec, axis=2)
    mask = bl > 1e-12
    if not np.any(mask):
        return
    out['lb_sum'] += float(bl[mask].sum())
    out['lb_count'] += int(mask.sum())
    unit = np.zeros_like(vec)
    unit[mask] = vec[mask] / bl[mask, None]
    nb = unit.shape[1]
    if out['corr_sum'] is None:
        out['corr_sum'] = np.zeros(nb, dtype=float)
        out['corr_count'] = np.zeros(nb, dtype=int)
    for lag in range(nb):
        dots = np.sum(unit[:, :nb-lag, :] * unit[:, lag:, :], axis=2)
        out['corr_sum'][lag] += float(np.sum(dots))
        out['corr_count'][lag] += int(dots.size)


def _fit_persistence_length(corr: np.ndarray, lb_nm: float) -> Optional[float]:
    if corr.size < 3 or not np.isfinite(lb_nm) or lb_nm <= 0:
        return None
    x = np.arange(corr.size, dtype=float) * float(lb_nm)
    valid = (corr > 0.0) & np.isfinite(corr)
    valid[0] = False
    if int(np.sum(valid)) < 2:
        return None
    slope, _ = np.polyfit(x[valid], np.log(corr[valid]), 1)
    if slope >= 0:
        return None
    return float(-1.0 / slope)


def compute_polymer_metrics(*, gro_path: Path, xtc_path: Path, top_path: Path, system_meta_path: Path, chunk: int = 50) -> Dict[str, Any]:
    try:
        import mdtraj as md
    except Exception as e:
        return {'warning': f'mdtraj not available: {e}'}

    top = parse_system_top(top_path)
    specs_by_type = build_polymer_chain_specs(top, system_meta_path)
    if not specs_by_type:
        return {}

    accum: Dict[str, Any] = {}
    for moltype, specs in specs_by_type.items():
        pairs = []
        for sp in specs:
            if sp.backbone_indices_0.size >= 2:
                pairs.append([int(sp.backbone_indices_0[0]), int(sp.backbone_indices_0[-1])])
        accum[moltype] = {
            'n_chains': len(specs),
            'chain_backbone_atoms': [int(sp.backbone_indices_0.size) for sp in specs],
            'rg_values': [],
            'e2e_values': [],
            'corr_sum': None,
            'corr_count': None,
            'lb_sum': 0.0,
            'lb_count': 0,
            'pairs': np.asarray(pairs, dtype=int) if pairs else np.zeros((0, 2), dtype=int),
        }

    cell_lengths = []
    cell_angles = []
    n_frames_total = 0
    for trj in md.iterload(str(xtc_path), top=str(gro_path), chunk=int(chunk)):
        n_frames_total += int(trj.n_frames)
        if getattr(trj, 'unitcell_lengths', None) is not None:
            try:
                cell_lengths.append(np.asarray(trj.unitcell_lengths, dtype=float))
                if getattr(trj, 'unitcell_angles', None) is not None:
                    cell_angles.append(np.asarray(trj.unitcell_angles, dtype=float))
            except Exception:
                pass
        for moltype, specs in specs_by_type.items():
            rec = accum[moltype]
            for sp in specs:
                atom_idx = sp.atom_indices_0
                ref = int(atom_idx[0])
                atom_pairs = np.asarray([[ref, int(a)] for a in atom_idx], dtype=int)
                rel = md.compute_displacements(trj, atom_pairs, periodic=True)
                com = np.mean(rel, axis=1, keepdims=True)
                rg = np.sqrt(np.mean(np.sum((rel - com) ** 2, axis=2), axis=1))
                rec['rg_values'].extend(np.asarray(rg, dtype=float).tolist())
                bb = sp.backbone_indices_0
                if bb.size >= 2:
                    bb_pairs = np.asarray([[int(bb[i]), int(bb[i + 1])] for i in range(bb.size - 1)], dtype=int)
                    bb_vec = md.compute_displacements(trj, bb_pairs, periodic=True)
                    e2e = np.linalg.norm(np.sum(bb_vec, axis=1), axis=1)
                    rec['e2e_values'].extend(np.asarray(e2e, dtype=float).tolist())
                    if bb.size >= 3:
                        _accumulate_corr_from_vectors(bb_vec, rec)

    out: Dict[str, Any] = {}
    for moltype, rec in accum.items():
        rg_stats = _safe_mean_std(rec['rg_values'])
        e2e_stats = _safe_mean_std(rec['e2e_values'])
        lb_nm = float(rec['lb_sum'] / rec['lb_count']) if rec['lb_count'] > 0 else None
        corr = None
        if rec['corr_sum'] is not None and rec['corr_count'] is not None:
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = np.divide(rec['corr_sum'], rec['corr_count'], out=np.zeros_like(rec['corr_sum']), where=rec['corr_count'] > 0)
        lp_nm = _fit_persistence_length(corr, lb_nm) if corr is not None and lb_nm is not None else None
        out[moltype] = {
            'n_frames': int(n_frames_total),
            'n_chains': int(rec['n_chains']),
            'chain_backbone_atoms': rec['chain_backbone_atoms'],
            'radius_of_gyration_nm': rg_stats,
            'end_to_end_distance_nm': e2e_stats,
            'persistence_length_nm': lp_nm,
            'average_backbone_bond_length_nm': lb_nm,
            'bond_autocorrelation': [float(x) for x in corr.tolist()] if corr is not None else [],
        }

    if cell_lengths:
        L = np.concatenate(cell_lengths, axis=0)
        cell = {
            'n_frames': int(L.shape[0]),
            'lengths_nm': {
                'a': {'mean': float(np.mean(L[:, 0])), 'std': float(np.std(L[:, 0], ddof=1)) if L.shape[0] > 1 else 0.0},
                'b': {'mean': float(np.mean(L[:, 1])), 'std': float(np.std(L[:, 1], ddof=1)) if L.shape[0] > 1 else 0.0},
                'c': {'mean': float(np.mean(L[:, 2])), 'std': float(np.std(L[:, 2], ddof=1)) if L.shape[0] > 1 else 0.0},
            },
        }
        if cell_angles:
            A = np.concatenate(cell_angles, axis=0)
            cell['angles_deg'] = {
                'alpha': {'mean': float(np.mean(A[:, 0])), 'std': float(np.std(A[:, 0], ddof=1)) if A.shape[0] > 1 else 0.0},
                'beta': {'mean': float(np.mean(A[:, 1])), 'std': float(np.std(A[:, 1], ddof=1)) if A.shape[0] > 1 else 0.0},
                'gamma': {'mean': float(np.mean(A[:, 2])), 'std': float(np.std(A[:, 2], ddof=1)) if A.shape[0] > 1 else 0.0},
            }
        out['_cell'] = cell

    return out


def compute_cell_summary(*, gro_path: Path, xtc_path: Path, chunk: int = 100) -> Dict[str, Any]:
    try:
        import mdtraj as md
    except Exception as e:
        return {'warning': f'mdtraj not available: {e}'}

    cell_lengths = []
    cell_angles = []
    for trj in md.iterload(str(xtc_path), top=str(gro_path), chunk=int(chunk)):
        if getattr(trj, 'unitcell_lengths', None) is not None:
            try:
                cell_lengths.append(np.asarray(trj.unitcell_lengths, dtype=float))
                if getattr(trj, 'unitcell_angles', None) is not None:
                    cell_angles.append(np.asarray(trj.unitcell_angles, dtype=float))
            except Exception:
                pass
    if not cell_lengths:
        return {}
    L = np.concatenate(cell_lengths, axis=0)
    out: Dict[str, Any] = {
        'n_frames': int(L.shape[0]),
        'lengths_nm': {
            'a': {'mean': float(np.mean(L[:, 0])), 'std': float(np.std(L[:, 0], ddof=1)) if L.shape[0] > 1 else 0.0},
            'b': {'mean': float(np.mean(L[:, 1])), 'std': float(np.std(L[:, 1], ddof=1)) if L.shape[0] > 1 else 0.0},
            'c': {'mean': float(np.mean(L[:, 2])), 'std': float(np.std(L[:, 2], ddof=1)) if L.shape[0] > 1 else 0.0},
        },
        'volume_nm3': {'mean': float(np.mean(np.prod(L, axis=1))), 'std': float(np.std(np.prod(L, axis=1), ddof=1)) if L.shape[0] > 1 else 0.0},
    }
    if cell_angles:
        A = np.concatenate(cell_angles, axis=0)
        out['angles_deg'] = {
            'alpha': {'mean': float(np.mean(A[:, 0])), 'std': float(np.std(A[:, 0], ddof=1)) if A.shape[0] > 1 else 0.0},
            'beta': {'mean': float(np.mean(A[:, 1])), 'std': float(np.std(A[:, 1], ddof=1)) if A.shape[0] > 1 else 0.0},
            'gamma': {'mean': float(np.mean(A[:, 2])), 'std': float(np.std(A[:, 2], ddof=1)) if A.shape[0] > 1 else 0.0},
        }
    return out
