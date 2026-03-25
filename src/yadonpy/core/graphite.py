"""Graphite / graphenic substrate builders.

Notes
-----
This module uses an ideal AB-stacked graphite construction derived from the
standard hexagonal graphite lattice rather than vendoring a proprietary
Materials Studio asset. The in-plane C-C bond length is 1.42 A and the
interlayer spacing is 3.35 A, which is sufficient for building finite
graphitic slabs for YadonPy workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit import Geometry as Geom

from . import poly, utils
from .naming import ensure_name, get_name
from .topology import Cell

_CC_BOND = 1.42
_INTERLAYER = 3.35
_A1 = np.array([math.sqrt(3.0) * _CC_BOND, 0.0, 0.0], dtype=float)
_A2 = np.array([0.5 * math.sqrt(3.0) * _CC_BOND, 1.5 * _CC_BOND, 0.0], dtype=float)
_AB_SHIFT = (_A1 + _A2) / 3.0
_ALLOWED_CAPS = ("H", "OH", "CHO", "COOH")


@dataclass(frozen=True)
class GraphiteBuildResult:
    cell: Chem.Mol
    layer_mol: Chem.Mol
    layer_count: int
    orientation: str
    edge_cap_summary: dict[str, int]
    box_nm: tuple[float, float, float]


@dataclass(frozen=True)
class StackedCellResult:
    cell: Chem.Mol
    box_nm: tuple[float, float, float]


def _unit(vec: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1.0e-10:
        arr = np.asarray(fallback, dtype=float)
        norm = float(np.linalg.norm(arr))
        if norm <= 1.0e-10:
            return np.array([1.0, 0.0, 0.0], dtype=float)
    return arr / norm


def _coords(mol: Chem.Mol, conf_id: int = 0) -> np.ndarray:
    return np.asarray(mol.GetConformer(conf_id).GetPositions(), dtype=float)


def _set_coords(mol: Chem.Mol, coord: np.ndarray, conf_id: int = 0) -> None:
    conf = mol.GetConformer(conf_id)
    for i, xyz in enumerate(np.asarray(coord, dtype=float)):
        conf.SetAtomPosition(i, Geom.Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def _translate(mol: Chem.Mol, shift: Sequence[float], conf_id: int = 0) -> Chem.Mol:
    dup = utils.deepcopy_mol(mol)
    coord = _coords(dup, conf_id=conf_id) + np.asarray(shift, dtype=float)
    _set_coords(dup, coord, conf_id=conf_id)
    return dup


def _rotate_x(coord: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ],
        dtype=float,
    )
    return np.asarray(coord, dtype=float) @ rot.T


def _graphene_layer(nx: int, ny: int) -> tuple[Chem.Mol, list[int]]:
    if int(nx) <= 0 or int(ny) <= 0:
        raise ValueError("nx and ny must be positive")

    rw = Chem.RWMol()
    coords: list[np.ndarray] = []
    coord_index: dict[tuple[int, int, int], int] = {}

    def _coord_key(pos: np.ndarray) -> tuple[int, int, int]:
        arr = np.asarray(pos, dtype=float)
        return (int(round(arr[0] * 1000.0)), int(round(arr[1] * 1000.0)), int(round(arr[2] * 1000.0)))

    def _get_or_add_aromatic_c(pos: np.ndarray) -> int:
        key = _coord_key(pos)
        if key in coord_index:
            return int(coord_index[key])
        atom = Chem.Atom("C")
        atom.SetIsAromatic(True)
        atom.SetNoImplicit(True)
        idx = rw.AddAtom(atom)
        coords.append(np.asarray(pos, dtype=float))
        coord_index[key] = int(idx)
        return int(idx)

    def _bond(i1: int, i2: int) -> None:
        if rw.GetBondBetweenAtoms(int(i1), int(i2)) is None:
            rw.AddBond(int(i1), int(i2), rdchem.BondType.AROMATIC)

    dx = math.sqrt(3.0) * _CC_BOND
    dy = 1.5 * _CC_BOND
    vertices = (
        np.array([0.0, _CC_BOND, 0.0], dtype=float),
        np.array([0.5 * math.sqrt(3.0) * _CC_BOND, 0.5 * _CC_BOND, 0.0], dtype=float),
        np.array([0.5 * math.sqrt(3.0) * _CC_BOND, -0.5 * _CC_BOND, 0.0], dtype=float),
        np.array([0.0, -_CC_BOND, 0.0], dtype=float),
        np.array([-0.5 * math.sqrt(3.0) * _CC_BOND, -0.5 * _CC_BOND, 0.0], dtype=float),
        np.array([-0.5 * math.sqrt(3.0) * _CC_BOND, 0.5 * _CC_BOND, 0.0], dtype=float),
    )

    for row in range(int(ny)):
        for col in range(int(nx)):
            center = np.array(
                [
                    float(col) * dx + (0.5 * dx if (row % 2) else 0.0),
                    float(row) * dy,
                    0.0,
                ],
                dtype=float,
            )
            ring = [_get_or_add_aromatic_c(center + vertex) for vertex in vertices]
            for i1, i2 in zip(ring, ring[1:] + ring[:1]):
                _bond(i1, i2)

    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for idx, xyz in enumerate(coords):
        conf.SetAtomPosition(idx, Geom.Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)

    dangling = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "C" and atom.GetDegree() < 3]
    return mol, dangling


def _choose_caps(
    edge_cap: str | Sequence[str],
    site_count: int,
    *,
    random_cap_probs: Mapping[str, float] | None,
    rng: random.Random,
) -> list[str]:
    if isinstance(edge_cap, str):
        token = edge_cap.strip().upper()
        if token == "RANDOM":
            probs = dict(random_cap_probs or {cap: 1.0 for cap in _ALLOWED_CAPS})
            choices = []
            weights = []
            for cap in _ALLOWED_CAPS:
                weight = float(probs.get(cap, 0.0))
                if weight > 0.0:
                    choices.append(cap)
                    weights.append(weight)
            if not choices:
                raise ValueError("random_cap_probs must leave at least one positive cap choice")
            return [rng.choices(choices, weights=weights, k=1)[0] for _ in range(int(site_count))]
        if token not in _ALLOWED_CAPS:
            raise ValueError(f"Unsupported edge_cap {edge_cap!r}. Choose from {_ALLOWED_CAPS} or 'random'.")
        return [token] * int(site_count)

    caps = [str(cap).strip().upper() for cap in edge_cap]
    if len(caps) != int(site_count):
        raise ValueError("edge_cap sequence length must match the number of dangling edge sites")
    for cap in caps:
        if cap not in _ALLOWED_CAPS:
            raise ValueError(f"Unsupported cap {cap!r}. Choose from {_ALLOWED_CAPS}.")
    return caps


def _local_frame(coord: np.ndarray, atom_idx: int, neighbors: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    c = coord[int(atom_idx)]
    nbr = np.asarray([coord[int(i)] for i in neighbors], dtype=float)
    out = _unit((len(nbr) * c) - np.sum(nbr, axis=0), fallback=(0.0, 0.0, 1.0))
    if len(nbr) >= 2:
        normal = np.cross(nbr[0] - c, nbr[1] - c)
    else:
        normal = np.cross(out, np.array([0.0, 0.0, 1.0], dtype=float))
    normal = _unit(normal, fallback=(0.0, 0.0, 1.0))
    tangent = _unit(np.cross(normal, out), fallback=(1.0, 0.0, 0.0))
    return out, tangent


def _add_atom(
    rw: Chem.RWMol,
    coord: list[np.ndarray],
    *,
    symbol: str,
    pos: np.ndarray,
    aromatic: bool = False,
) -> int:
    atom = Chem.Atom(str(symbol))
    atom.SetNoImplicit(True)
    atom.SetIsAromatic(bool(aromatic))
    idx = int(rw.AddAtom(atom))
    coord.append(np.asarray(pos, dtype=float))
    return idx


def _cap_edges(mol: Chem.Mol, edge_cap: str | Sequence[str], *, random_cap_probs: Mapping[str, float] | None, random_seed: int | None) -> tuple[Chem.Mol, dict[str, int]]:
    rng = random.Random(random_seed)
    rw = Chem.RWMol(mol)
    coord = list(_coords(mol))
    dangling = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetSymbol() == "C" and atom.GetDegree() < 3]
    caps = _choose_caps(edge_cap, len(dangling), random_cap_probs=random_cap_probs, rng=rng)
    summary = {cap: 0 for cap in _ALLOWED_CAPS}

    for atom_idx, cap in zip(dangling, caps):
        summary[cap] += 1
        neighbors = [nbr.GetIdx() for nbr in rw.GetAtomWithIdx(int(atom_idx)).GetNeighbors()]
        out, tangent = _local_frame(np.asarray(coord, dtype=float), int(atom_idx), neighbors)
        c = np.asarray(coord[int(atom_idx)], dtype=float)

        if cap == "H":
            h = _add_atom(rw, coord, symbol="H", pos=c + 1.09 * out)
            rw.AddBond(int(atom_idx), h, rdchem.BondType.SINGLE)
            continue

        if cap == "OH":
            o = _add_atom(rw, coord, symbol="O", pos=c + 1.36 * out)
            h = _add_atom(rw, coord, symbol="H", pos=np.asarray(coord[o]) + 0.96 * _unit(out + 0.25 * tangent, fallback=out))
            rw.AddBond(int(atom_idx), o, rdchem.BondType.SINGLE)
            rw.AddBond(o, h, rdchem.BondType.SINGLE)
            continue

        if cap == "CHO":
            c1 = _add_atom(rw, coord, symbol="C", pos=c + 1.46 * out)
            o1 = _add_atom(rw, coord, symbol="O", pos=np.asarray(coord[c1]) + 1.23 * out)
            h1 = _add_atom(
                rw,
                coord,
                symbol="H",
                pos=np.asarray(coord[c1]) + 1.09 * _unit(-0.55 * out + 0.85 * tangent, fallback=tangent),
            )
            rw.AddBond(int(atom_idx), c1, rdchem.BondType.SINGLE)
            rw.AddBond(c1, o1, rdchem.BondType.DOUBLE)
            rw.AddBond(c1, h1, rdchem.BondType.SINGLE)
            continue

        if cap == "COOH":
            c1 = _add_atom(rw, coord, symbol="C", pos=c + 1.50 * out)
            o_d = _add_atom(rw, coord, symbol="O", pos=np.asarray(coord[c1]) + 1.23 * out)
            oh_dir = _unit(-0.45 * out + 0.89 * tangent, fallback=tangent)
            o_h = _add_atom(rw, coord, symbol="O", pos=np.asarray(coord[c1]) + 1.33 * oh_dir)
            h1 = _add_atom(
                rw,
                coord,
                symbol="H",
                pos=np.asarray(coord[o_h]) + 0.96 * _unit(oh_dir + 0.20 * out, fallback=oh_dir),
            )
            rw.AddBond(int(atom_idx), c1, rdchem.BondType.SINGLE)
            rw.AddBond(c1, o_d, rdchem.BondType.DOUBLE)
            rw.AddBond(c1, o_h, rdchem.BondType.SINGLE)
            rw.AddBond(o_h, h1, rdchem.BondType.SINGLE)
            continue

    capped = rw.GetMol()
    conf = Chem.Conformer(capped.GetNumAtoms())
    conf.Set3D(True)
    for idx, xyz in enumerate(coord):
        conf.SetAtomPosition(idx, Geom.Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    capped.RemoveAllConformers()
    capped.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(capped)
    return capped, {key: value for key, value in summary.items() if value > 0}


def _box_from_coords(coord: np.ndarray, *, lateral_margin_ang: float, bottom_margin_ang: float, top_padding_ang: float) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]:
    coord = np.asarray(coord, dtype=float)
    mins = np.min(coord, axis=0)
    maxs = np.max(coord, axis=0)

    shift = np.array(
        [
            float(lateral_margin_ang) - mins[0],
            float(lateral_margin_ang) - mins[1],
            float(bottom_margin_ang) - mins[2],
        ],
        dtype=float,
    )
    moved = coord + shift
    maxs = np.max(moved, axis=0)

    xlo = 0.0
    ylo = 0.0
    zlo = 0.0
    xhi = float(maxs[0] + float(lateral_margin_ang))
    yhi = float(maxs[1] + float(lateral_margin_ang))
    zhi = float(maxs[2] + float(top_padding_ang))
    return moved, (xhi, xlo, yhi, ylo, zhi, zlo)


def stack_cell_blocks(
    blocks: Sequence[Chem.Mol],
    *,
    z_gaps_ang: float | Sequence[float] = 0.0,
    lateral_margin_ang: float = 4.0,
    bottom_margin_ang: float = 2.0,
    top_padding_ang: float = 8.0,
) -> StackedCellResult:
    if not blocks:
        raise ValueError("blocks must not be empty")

    dup_blocks = [utils.deepcopy_mol(block) for block in blocks]
    coords = [_coords(block) for block in dup_blocks]
    extents = []
    for coord in coords:
        mins = np.min(coord, axis=0)
        maxs = np.max(coord, axis=0)
        extents.append((mins, maxs))

    if isinstance(z_gaps_ang, (int, float)):
        gaps = [float(z_gaps_ang)] * max(0, len(dup_blocks) - 1)
    else:
        gaps = [float(gap) for gap in z_gaps_ang]
    if len(gaps) != max(0, len(dup_blocks) - 1):
        raise ValueError("z_gaps_ang must be a scalar or have len(blocks)-1 entries")

    x_span = max(float(maxs[0] - mins[0]) for mins, maxs in extents)
    y_span = max(float(maxs[1] - mins[1]) for mins, maxs in extents)
    x_center = float(lateral_margin_ang) + 0.5 * x_span
    y_center = float(lateral_margin_ang) + 0.5 * y_span
    z_cursor = float(bottom_margin_ang)

    combined = None
    for idx, (block, coord, (mins, maxs)) in enumerate(zip(dup_blocks, coords, extents)):
        shift = np.array(
            [
                x_center - 0.5 * float(mins[0] + maxs[0]),
                y_center - 0.5 * float(mins[1] + maxs[1]),
                z_cursor - float(mins[2]),
            ],
            dtype=float,
        )
        _set_coords(block, coord + shift)
        combined = block if combined is None else poly.combine_mols(combined, block, res_name_1="BLK", res_name_2="BLK")
        z_cursor += float(maxs[2] - mins[2])
        if idx < len(gaps):
            z_cursor += float(gaps[idx])

    xhi = 2.0 * float(lateral_margin_ang) + x_span
    yhi = 2.0 * float(lateral_margin_ang) + y_span
    zhi = z_cursor + float(top_padding_ang)
    setattr(combined, "cell", Cell(xhi, 0.0, yhi, 0.0, zhi, 0.0))
    poly.set_cell_param_conf(combined, 0, xhi, 0.0, yhi, 0.0, zhi, 0.0)
    return StackedCellResult(cell=combined, box_nm=(0.1 * xhi, 0.1 * yhi, 0.1 * zhi))


def register_cell_species_metadata(
    cell: Chem.Mol,
    mols: Sequence[Chem.Mol],
    counts: Sequence[int],
    *,
    density_g_cm3: float | None = None,
    charge_scale: float | Sequence[float] | Mapping[str, float] | None = None,
    pack_mode: str = "custom",
) -> Chem.Mol:
    if len(mols) != len(counts):
        raise ValueError("mols and counts must have the same length")

    mols_c = [utils.deepcopy_mol(mol) for mol in mols]
    counts_i = [int(n) for n in counts]
    poly._cache_artifacts_best_effort(mols_c, prefer_var=False)

    species = []
    for mol, n in zip(mols_c, counts_i):
        if int(n) <= 0:
            continue
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            smiles = ""
        entry = {
            "smiles": str(smiles),
            "n": int(n),
            "natoms": int(mol.GetNumAtoms()),
            "name": get_name(mol, default=None),
            "ff_name": str(mol.GetProp("_yadonpy_ff_name")) if mol.HasProp("_yadonpy_ff_name") else None,
        }
        if charge_scale is not None:
            if isinstance(charge_scale, (list, tuple)):
                entry["charge_scale"] = float(charge_scale[len(species)])
            elif isinstance(charge_scale, dict):
                try:
                    entry["charge_scale"] = float(charge_scale.get(smiles, charge_scale.get(entry["name"], 1.0)))
                except Exception:
                    pass
            else:
                entry["charge_scale"] = float(charge_scale)

        for key in (
            "_yadonpy_bonded_signature",
            "_yadonpy_bonded_requested",
            "_yadonpy_bonded_method",
            "_yadonpy_bonded_override",
            "_yadonpy_molid",
            "_yadonpy_artifact_dir",
        ):
            if mol.HasProp(key):
                value = str(mol.GetProp(key)).strip()
                if not value:
                    continue
                if key == "_yadonpy_molid":
                    entry["cached_mol_id"] = value
                elif key == "_yadonpy_artifact_dir":
                    entry["cached_artifact_dir"] = value
                elif key == "_yadonpy_bonded_signature":
                    entry["bonded_signature"] = value
                elif key == "_yadonpy_bonded_requested":
                    entry["bonded_requested"] = value
                elif key == "_yadonpy_bonded_method":
                    entry["bonded_method"] = value
                elif key == "_yadonpy_bonded_override":
                    entry["bonded_explicit"] = value.lower() in {"1", "true", "yes", "on"}
        species.append(entry)

    q_raw = 0.0
    q_scaled = 0.0
    for idx, (mol, n) in enumerate(zip(mols_c, counts_i)):
        qi = float(poly._mol_net_charge(mol)) * float(n)
        q_raw += qi
        scale = 1.0
        if charge_scale is not None:
            if isinstance(charge_scale, (list, tuple)):
                scale = float(charge_scale[idx])
            elif isinstance(charge_scale, dict):
                try:
                    scale = float(charge_scale.get(Chem.MolToSmiles(mol, canonical=True), charge_scale.get(get_name(mol, default=None), 1.0)))
                except Exception:
                    scale = 1.0
            else:
                scale = float(charge_scale)
        q_scaled += qi * scale

    payload = {
        "density_g_cm3": (float(density_g_cm3) if density_g_cm3 is not None else None),
        "species": species,
        "pack_mode": str(pack_mode),
        "target_atoms": int(sum(int(m.GetNumAtoms()) * int(n) for m, n in zip(mols_c, counts_i))),
        "net_charge_raw": float(q_raw),
        "net_charge_scaled": float(q_scaled),
        "charge_tolerance": 1.0e-2,
        "net_charge_ok": bool(abs(q_scaled) <= 1.0e-2),
    }
    cell.SetProp("_yadonpy_cell_meta", json.dumps(payload, ensure_ascii=False))
    return cell


def build_graphite(
    *,
    nx: int,
    ny: int,
    n_layers: int = 3,
    orientation: str = "basal",
    edge_cap: str | Sequence[str] = "H",
    random_cap_probs: Mapping[str, float] | None = None,
    ff=None,
    ff_name: str = "gaff2",
    charge: str | None = None,
    random_seed: int | None = None,
    name: str | None = None,
    lateral_margin_ang: float = 4.0,
    bottom_margin_ang: float = 2.0,
    top_padding_ang: float = 8.0,
) -> GraphiteBuildResult:
    layer_base, dangling = _graphene_layer(nx=int(nx), ny=int(ny))
    if not dangling:
        raise RuntimeError("Graphite builder failed to identify edge sites in the graphene layer")

    capped_layer, cap_summary = _cap_edges(
        layer_base,
        edge_cap=edge_cap,
        random_cap_probs=random_cap_probs,
        random_seed=random_seed,
    )

    ff_obj = ff
    if ff_obj is None:
        from ..api import get_ff

        ff_obj = get_ff(ff_name)
    layer_mol = utils.deepcopy_mol(capped_layer)
    ok = bool(ff_obj.ff_assign(layer_mol, charge=charge, report=False))
    if not ok:
        raise RuntimeError("Can not assign force field parameters for the graphite layer.")
    ensure_name(layer_mol, name=(str(name) if name else None), prefer_var=False)

    cell = None
    for layer_idx in range(int(n_layers)):
        shift = np.array([0.0, 0.0, float(layer_idx) * _INTERLAYER], dtype=float)
        if layer_idx % 2 == 1:
            shift[:2] += _AB_SHIFT[:2]
        shifted_layer = _translate(layer_mol, shift)
        cell = shifted_layer if cell is None else poly.combine_mols(cell, shifted_layer, res_name_1="GRA", res_name_2="GRA")

    coord = _coords(cell)
    orient = str(orientation).strip().lower()
    if orient not in {"basal", "edge"}:
        raise ValueError("orientation must be 'basal' or 'edge'")
    if orient == "edge":
        coord = _rotate_x(coord, 90.0)

    coord, bounds = _box_from_coords(
        coord,
        lateral_margin_ang=float(lateral_margin_ang),
        bottom_margin_ang=float(bottom_margin_ang),
        top_padding_ang=float(top_padding_ang),
    )
    _set_coords(cell, coord)
    xhi, xlo, yhi, ylo, zhi, zlo = bounds
    setattr(cell, "cell", Cell(xhi, xlo, yhi, ylo, zhi, zlo))
    poly.set_cell_param_conf(cell, 0, xhi, xlo, yhi, ylo, zhi, zlo)

    register_cell_species_metadata(cell, [layer_mol], [int(n_layers)], pack_mode=f"graphite_{orient}")
    box_nm = (0.1 * (xhi - xlo), 0.1 * (yhi - ylo), 0.1 * (zhi - zlo))
    return GraphiteBuildResult(
        cell=cell,
        layer_mol=layer_mol,
        layer_count=int(n_layers),
        orientation=orient,
        edge_cap_summary=cap_summary,
        box_nm=(float(box_nm[0]), float(box_nm[1]), float(box_nm[2])),
    )


__all__ = [
    "GraphiteBuildResult",
    "StackedCellResult",
    "build_graphite",
    "register_cell_species_metadata",
    "stack_cell_blocks",
]
