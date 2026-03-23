"""Bond/angle parameterization from a QM Hessian via (Modified) Seminario.

This implementation is intentionally lightweight and dependency-minimal:
- Input Hessian: Cartesian Hessian in kJ/(mol*Angstrom^2)
  (Psi4w.hessian() already returns this unit).
- Connectivity: RDKit Mol bonds/angles
- Output: bond/angle equilibrium values + harmonic force constants in
  GROMACS units (nm, degrees, kJ/mol/nm^2, kJ/mol/rad^2).

Compared with the early implementation, this version adds three robustness
features that matter for highly symmetric ions such as PF6-:

1) near-linear angles are no longer dropped silently; they are handled with a
   dedicated two-plane fallback;
2) equivalent bonds/angles can be symmetrized by RDKit symmetry rank and coarse
   geometry class (e.g. PF6- cis vs trans);
3) the projection routine supports the classical "Seminario-like" absolute
   projection as well as the quadratic Hessian-projection form, and defaults to
   the more conservative absolute projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from rdkit import Chem


@dataclass(frozen=True)
class BondParam:
    i: int  # 0-based atom index
    j: int
    r0_nm: float
    k_kj_mol_nm2: float


@dataclass(frozen=True)
class AngleParam:
    i: int  # 0-based atom index
    j: int
    k: int
    theta0_deg: float
    k_kj_mol_rad2: float


def _coords_angstrom(mol: Chem.Mol, confId: int = 0) -> np.ndarray:
    conf = mol.GetConformer(int(confId))
    n = mol.GetNumAtoms()
    xyz = np.zeros((n, 3), dtype=float)
    for a in range(n):
        p = conf.GetAtomPosition(a)
        xyz[a, 0] = float(p.x)
        xyz[a, 1] = float(p.y)
        xyz[a, 2] = float(p.z)
    return xyz


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _block(hess: np.ndarray, a: int, b: int) -> np.ndarray:
    """Return 3x3 Cartesian Hessian block for atom pair (a,b)."""
    ia = slice(3 * a, 3 * a + 3)
    ib = slice(3 * b, 3 * b + 3)
    return hess[ia, ib]


def _eig_sym(m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetrize then eigen-decompose (robust)."""
    ms = 0.5 * (m + m.T)
    vals, vecs = np.linalg.eigh(ms)
    vals = np.clip(vals, 0.0, None)
    return vals, vecs


def _projection_force_constant(vals: np.ndarray, vecs: np.ndarray, direction: np.ndarray, *, mode: str = "abs") -> float:
    """Project a direction onto Hessian eigenmodes.

    ``mode='abs'`` follows the more common Seminario-style absolute projection,
    while ``mode='quadratic'`` keeps the older quadratic Hessian-projection form.
    """
    u = _unit(direction)
    if float(np.linalg.norm(u)) < 1.0e-12:
        return 0.0
    proj = vecs.T @ u
    mode_l = str(mode or "abs").strip().lower()
    if mode_l in {"quadratic", "square", "squared", "projection"}:
        k = float(np.sum(vals * (proj ** 2)))
    elif mode_l in {"hybrid", "max"}:
        k_abs = float(np.sum(vals * np.abs(proj)))
        k_quad = float(np.sum(vals * (proj ** 2)))
        k = max(k_abs, k_quad)
    else:
        # Seminario-style absolute projection.
        k = float(np.sum(vals * np.abs(proj)))
    if not np.isfinite(k):
        return 0.0
    return max(k, 0.0)


def _seminario_k_from_block(block_ij: np.ndarray, direction: np.ndarray, *, projection_mode: str = "abs") -> float:
    """Compute projected force constant from a 3x3 block and a unit direction.

    We follow the common convention using the *negative* interatomic block
    (restoring positive curvature). Returns k in the same unit as the input
    block (kJ/mol/Ang^2).
    """
    vals, vecs = _eig_sym(-block_ij)
    return _projection_force_constant(vals, vecs, direction, mode=projection_mode)


def _perpendicular_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal vectors perpendicular to ``axis``."""
    u = _unit(axis)
    if float(np.linalg.norm(u)) < 1.0e-12:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    # Pick a reference vector that is not nearly parallel to the axis.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(u, ref))) > 0.8:
        ref = np.array([0.0, 1.0, 0.0])
    v1 = _unit(np.cross(u, ref))
    if float(np.linalg.norm(v1)) < 1.0e-12:
        ref = np.array([0.0, 0.0, 1.0])
        v1 = _unit(np.cross(u, ref))
    v2 = _unit(np.cross(u, v1))
    return v1, v2


def bond_params_from_hessian(
    mol: Chem.Mol,
    hessian_kj_mol_a2: np.ndarray,
    *,
    confId: int = 0,
    projection_mode: str = "abs",
) -> List[BondParam]:
    """Compute harmonic bond parameters for all RDKit bonds."""
    xyz = _coords_angstrom(mol, confId=confId)
    out: List[BondParam] = []

    for b in mol.GetBonds():
        i = int(b.GetBeginAtomIdx())
        j = int(b.GetEndAtomIdx())
        rij = xyz[j] - xyz[i]
        r0_a = float(np.linalg.norm(rij))
        if r0_a <= 1e-6:
            continue

        k_a2 = _seminario_k_from_block(_block(hessian_kj_mol_a2, i, j), rij, projection_mode=projection_mode)

        # Unit conversion:
        #  - r0: Ang -> nm
        #  - k:  kJ/mol/Ang^2 -> kJ/mol/nm^2  (1 nm = 10 Ang => multiply by 100)
        r0_nm = r0_a / 10.0
        k_nm2 = k_a2 * 100.0

        out.append(BondParam(i=i, j=j, r0_nm=r0_nm, k_kj_mol_nm2=k_nm2))

    return out


def _angle_triplets(mol: Chem.Mol) -> Iterable[Tuple[int, int, int]]:
    """Enumerate all i-j-k angles from RDKit connectivity."""
    for j in range(mol.GetNumAtoms()):
        nbrs = [int(a.GetIdx()) for a in mol.GetAtomWithIdx(j).GetNeighbors()]
        if len(nbrs) < 2:
            continue
        for i, k in combinations(sorted(nbrs), 2):
            yield i, j, k


def _combine_angle_force_constants(k1: float, k2: float, r1: float, r2: float) -> float:
    eps = 1.0e-12
    denom = (1.0 / max(r1 * r1 * max(k1, eps), eps)) + (1.0 / max(r2 * r2 * max(k2, eps), eps))
    if denom <= 0.0:
        return 0.0
    k_theta = 1.0 / denom
    if not np.isfinite(k_theta):
        return 0.0
    return float(max(k_theta, 0.0))


def _linear_angle_force_constant(
    hessian_kj_mol_a2: np.ndarray,
    i: int,
    j: int,
    k: int,
    u1: np.ndarray,
    u2: np.ndarray,
    r1: float,
    r2: float,
    *,
    projection_mode: str = "abs",
) -> float:
    """Fallback for near-linear angles.

    For a 180° angle the in-plane modified-Seminario vectors become ill-conditioned.
    We therefore build two orthogonal bending directions perpendicular to the bond
    axis and average their force constants.
    """
    axis = _unit(u1 - u2)
    if float(np.linalg.norm(axis)) < 1.0e-12:
        axis = _unit(u1 if np.linalg.norm(u1) > np.linalg.norm(u2) else u2)
    v1, v2 = _perpendicular_basis(axis)
    ks: list[float] = []
    for vv in (v1, v2):
        ki = _seminario_k_from_block(_block(hessian_kj_mol_a2, i, j), vv, projection_mode=projection_mode)
        kk = _seminario_k_from_block(_block(hessian_kj_mol_a2, k, j), vv, projection_mode=projection_mode)
        kval = _combine_angle_force_constants(ki, kk, r1, r2)
        if kval > 0.0:
            ks.append(float(kval))
    if not ks:
        return 0.0
    return float(sum(ks) / len(ks))


def angle_params_from_hessian(
    mol: Chem.Mol,
    hessian_kj_mol_a2: np.ndarray,
    *,
    confId: int = 0,
    linear_angle_deg_cutoff: float = 175.0,
    projection_mode: str = "abs",
    keep_linear_angles: bool = True,
) -> List[AngleParam]:
    """Compute harmonic angle parameters for all RDKit angles.

    For regular angles we use the common modified-Seminario projection vectors:
      n_i = normalize(u_jk - cos(theta)*u_ji)
      n_k = normalize(u_ji - cos(theta)*u_jk)

    For near-linear angles we switch to a dedicated two-plane fallback instead of
    dropping the term, which is particularly important for AX6 ions such as PF6-.
    """
    xyz = _coords_angstrom(mol, confId=confId)
    out: List[AngleParam] = []

    for i, j, k in _angle_triplets(mol):
        rji = xyz[i] - xyz[j]
        rjk = xyz[k] - xyz[j]
        r1 = float(np.linalg.norm(rji))
        r2 = float(np.linalg.norm(rjk))
        if r1 <= 1e-8 or r2 <= 1e-8:
            continue

        u1 = rji / r1
        u2 = rjk / r2
        cos_t = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        theta = float(np.arccos(cos_t))
        theta_deg = float(theta * 180.0 / np.pi)

        is_linear = theta_deg >= float(linear_angle_deg_cutoff)
        is_collapsed = theta_deg <= max(1.0e-6, 180.0 - float(linear_angle_deg_cutoff))
        if is_collapsed:
            continue

        if is_linear:
            if not keep_linear_angles:
                continue
            k_theta = _linear_angle_force_constant(
                hessian_kj_mol_a2,
                i,
                j,
                k,
                u1,
                u2,
                r1,
                r2,
                projection_mode=projection_mode,
            )
            if k_theta <= 0.0:
                continue
            out.append(AngleParam(i=i, j=j, k=k, theta0_deg=180.0, k_kj_mol_rad2=k_theta))
            continue

        # In-plane perpendicular directions
        n1 = _unit(u2 - cos_t * u1)
        n2 = _unit(u1 - cos_t * u2)
        if float(np.linalg.norm(n1)) < 1e-12 or float(np.linalg.norm(n2)) < 1e-12:
            continue

        k_i_a2 = _seminario_k_from_block(_block(hessian_kj_mol_a2, i, j), n1, projection_mode=projection_mode)
        k_k_a2 = _seminario_k_from_block(_block(hessian_kj_mol_a2, k, j), n2, projection_mode=projection_mode)
        k_theta = _combine_angle_force_constants(k_i_a2, k_k_a2, r1, r2)
        if k_theta <= 0.0:
            continue

        out.append(AngleParam(i=i, j=j, k=k, theta0_deg=theta_deg, k_kj_mol_rad2=k_theta))

    return out


def _symmetry_ranks(mol: Chem.Mol) -> list[int]:
    try:
        return [int(x) for x in Chem.CanonicalRankAtoms(mol, breakTies=False)]
    except Exception:
        return list(range(int(mol.GetNumAtoms())))


def _bond_group_key(item: Dict[str, Any], ranks: list[int]) -> tuple:
    i = int(item["i"])
    j = int(item["j"])
    return tuple(sorted((ranks[i], ranks[j])))


def _angle_group_key(item: Dict[str, Any], ranks: list[int]) -> tuple:
    i = int(item["i"])
    j = int(item["j"])
    k = int(item["k"])
    th = float(item["theta0_deg"])
    if th >= 175.0:
        bucket = "linear"
    else:
        bucket = int(round(th / 10.0) * 10)
    return (min(ranks[i], ranks[k]), ranks[j], max(ranks[i], ranks[k]), bucket)


def _symmetrize_equivalent_terms(mol: Chem.Mol, params: Dict[str, Any]) -> Dict[str, Any]:
    """Average equivalent bonds/angles by RDKit symmetry rank and coarse geometry.

    This is intentionally conservative: trans and cis angle manifolds remain separate,
    which is essential for AX6 ions.
    """
    ranks = _symmetry_ranks(mol)
    out = {
        "meta": dict(params.get("meta") or {}),
        "bonds": [dict(x) for x in (params.get("bonds") or [])],
        "angles": [dict(x) for x in (params.get("angles") or [])],
    }

    # Bonds
    b_groups: Dict[tuple, list[Dict[str, Any]]] = {}
    for item in out["bonds"]:
        b_groups.setdefault(_bond_group_key(item, ranks), []).append(item)
    for items in b_groups.values():
        if len(items) <= 1:
            continue
        r0 = float(sum(float(x["r0_nm"]) for x in items) / len(items))
        kk = float(sum(float(x["k_kj_mol_nm2"]) for x in items) / len(items))
        for x in items:
            x["r0_nm"] = r0
            x["k_kj_mol_nm2"] = kk

    # Angles
    a_groups: Dict[tuple, list[Dict[str, Any]]] = {}
    for item in out["angles"]:
        a_groups.setdefault(_angle_group_key(item, ranks), []).append(item)
    for items in a_groups.values():
        if len(items) <= 1:
            continue
        th = float(sum(float(x["theta0_deg"]) for x in items) / len(items))
        kk = float(sum(float(x["k_kj_mol_rad2"]) for x in items) / len(items))
        for x in items:
            x["theta0_deg"] = th
            x["k_kj_mol_rad2"] = kk

    out.setdefault("meta", {})
    out["meta"]["symmetrized_equivalents"] = True
    return out


def bond_angle_params_from_hessian(
    mol: Chem.Mol,
    hessian_kj_mol_a2: np.ndarray,
    *,
    confId: int = 0,
    linear_angle_deg_cutoff: float = 175.0,
    projection_mode: str = "abs",
    keep_linear_angles: bool = True,
    symmetrize_equivalents: bool = True,
) -> Dict[str, Any]:
    """Convenience: return both bond+angle params as JSON-serializable dict."""
    bonds = bond_params_from_hessian(
        mol,
        hessian_kj_mol_a2,
        confId=confId,
        projection_mode=projection_mode,
    )
    angles = angle_params_from_hessian(
        mol,
        hessian_kj_mol_a2,
        confId=confId,
        linear_angle_deg_cutoff=float(linear_angle_deg_cutoff),
        projection_mode=projection_mode,
        keep_linear_angles=bool(keep_linear_angles),
    )

    params: Dict[str, Any] = {
        "meta": {
            "confId": int(confId),
            "num_atoms": int(mol.GetNumAtoms()),
            "unit": {
                "bond_r0": "nm",
                "bond_k": "kJ/mol/nm^2",
                "angle_theta0": "deg",
                "angle_k": "kJ/mol/rad^2",
            },
            "method": "mseminario",
            "projection_mode": str(projection_mode),
            "keep_linear_angles": bool(keep_linear_angles),
            "linear_angle_deg_cutoff": float(linear_angle_deg_cutoff),
        },
        "bonds": [
            {
                "i": int(p.i),
                "j": int(p.j),
                "r0_nm": float(p.r0_nm),
                "k_kj_mol_nm2": float(p.k_kj_mol_nm2),
            }
            for p in bonds
        ],
        "angles": [
            {
                "i": int(p.i),
                "j": int(p.j),
                "k": int(p.k),
                "theta0_deg": float(p.theta0_deg),
                "k_kj_mol_rad2": float(p.k_kj_mol_rad2),
            }
            for p in angles
        ],
    }

    if symmetrize_equivalents:
        params = _symmetrize_equivalent_terms(mol, params)
    else:
        params.setdefault("meta", {})["symmetrized_equivalents"] = False
    return params


def write_bond_angle_itp(
    mol: Chem.Mol,
    params: Dict[str, Any],
    out_itp: Path,
    *,
    comment: str = "generated by yadonpy seminario",
) -> Path:
    """Write a minimal .itp fragment with [ bonds ] and [ angles ].

    This file is meant to be *included* or used as a patch reference.
    It uses 1-based atom indices.
    """
    out_itp = Path(out_itp)
    out_itp.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"; {comment}\n")

    bonds = params.get("bonds") or []
    if bonds:
        lines.append("[ bonds ]\n")
        lines.append("; i  j  funct  r0(nm)  k(kJ/mol/nm^2)\n")
        for b in bonds:
            i = int(b["i"]) + 1
            j = int(b["j"]) + 1
            r0 = float(b["r0_nm"])
            kk = float(b["k_kj_mol_nm2"])
            lines.append(f"{i:5d} {j:5d}  1  {r0: .6f}  {kk: .2f}\n")
        lines.append("\n")

    angles = params.get("angles") or []
    if angles:
        lines.append("[ angles ]\n")
        lines.append("; i  j  k  funct  theta0(deg)  k(kJ/mol/rad^2)\n")
        for a in angles:
            i = int(a["i"]) + 1
            j = int(a["j"]) + 1
            k = int(a["k"]) + 1
            th0 = float(a["theta0_deg"])
            kk = float(a["k_kj_mol_rad2"])
            lines.append(f"{i:5d} {j:5d} {k:5d}  1  {th0: .3f}  {kk: .2f}\n")
        lines.append("\n")

    out_itp.write_text("".join(lines), encoding="utf-8")
    return out_itp
