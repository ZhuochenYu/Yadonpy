"""Bond/angle parameterization from a QM Hessian via (Modified) Seminario.

This module is intentionally lightweight and dependency-minimal:
- Input Hessian: Cartesian Hessian in kJ/(mol*Angstrom^2)
  (Psi4w.hessian() already returns this unit).
- Connectivity: RDKit Mol bonds/angles
- Output: bond/angle equilibrium values + harmonic force constants in
  GROMACS units (nm, degrees, kJ/mol/nm^2, kJ/mol/rad^2).

Notes
-----
1) This does *not* attempt to derive dihedrals/impropers.
2) As with all Seminario-family methods, results depend on the QM level
   of theory and on the quality of the optimized geometry.
3) We treat the 3x3 interatomic Hessian block symmetrically and clip
   negative eigenvalues to 0 (best-effort robustness).

References
----------
- Seminario, J. M. (1996) calculation of intramolecular force constants
  from a molecular Hessian.
- "Modified Seminario" variants (Bernetti/Bussi context is barostat; not
  related). Several community implementations use the same projection
  vectors employed here.
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


def _seminario_k_from_block(block_ij: np.ndarray, direction: np.ndarray) -> float:
    """Compute projected force constant from a 3x3 block and a unit direction.

    We follow the common Seminario convention using the *negative* interatomic
    block (restoring positive curvature).

    Returns k in the same unit as the input block (kJ/mol/Ang^2).
    """
    u = _unit(direction)
    if float(np.linalg.norm(u)) < 1e-12:
        return 0.0

    # Use -H_ij and symmetrize
    vals, vecs = _eig_sym(-block_ij)
    # projection of u onto eigenvectors
    proj = vecs.T @ u
    k = float(np.sum(vals * (proj ** 2)))
    if not np.isfinite(k):
        return 0.0
    return max(k, 0.0)


def bond_params_from_hessian(
    mol: Chem.Mol,
    hessian_kj_mol_a2: np.ndarray,
    *,
    confId: int = 0,
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

        k_a2 = _seminario_k_from_block(_block(hessian_kj_mol_a2, i, j), rij)

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


def angle_params_from_hessian(
    mol: Chem.Mol,
    hessian_kj_mol_a2: np.ndarray,
    *,
    confId: int = 0,
    linear_angle_deg_cutoff: float = 175.0,
) -> List[AngleParam]:
    """Compute harmonic angle parameters for all RDKit angles.

    We use the common "modified Seminario" projection vectors:
      n_i = normalize(u_jk - cos(theta)*u_ji)
      n_k = normalize(u_ji - cos(theta)*u_jk)

    Combination rule (widely used in practice):
      k_theta = 1 / ( 1/(r_ji^2*k_i) + 1/(r_jk^2*k_k) )

    k_i and k_k are obtained by projecting (-H_ij) and (-H_kj) onto n_i/n_k.

    Returns k in kJ/mol/rad^2 and theta0 in degrees.
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

        # Skip near-linear angles: projection vectors become ill-conditioned.
        if theta_deg >= linear_angle_deg_cutoff or theta_deg <= (180.0 - linear_angle_deg_cutoff):
            continue

        # In-plane perpendicular directions
        n1 = _unit(u2 - cos_t * u1)
        n2 = _unit(u1 - cos_t * u2)
        if float(np.linalg.norm(n1)) < 1e-12 or float(np.linalg.norm(n2)) < 1e-12:
            continue

        k_i_a2 = _seminario_k_from_block(_block(hessian_kj_mol_a2, i, j), n1)
        k_k_a2 = _seminario_k_from_block(_block(hessian_kj_mol_a2, k, j), n2)

        # Combine to get k_theta (kJ/mol/rad^2)
        eps = 1e-12
        denom = (1.0 / max(r1 * r1 * max(k_i_a2, eps), eps)) + (1.0 / max(r2 * r2 * max(k_k_a2, eps), eps))
        if denom <= 0:
            continue
        k_theta = 1.0 / denom
        if not np.isfinite(k_theta):
            continue
        k_theta = float(max(k_theta, 0.0))

        out.append(AngleParam(i=i, j=j, k=k, theta0_deg=theta_deg, k_kj_mol_rad2=k_theta))

    return out


def bond_angle_params_from_hessian(
    mol: Chem.Mol,
    hessian_kj_mol_a2: np.ndarray,
    *,
    confId: int = 0,
    linear_angle_deg_cutoff: float = 175.0,
) -> Dict[str, Any]:
    """Convenience: return both bond+angle params as JSON-serializable dict."""
    bonds = bond_params_from_hessian(mol, hessian_kj_mol_a2, confId=confId)
    angles = angle_params_from_hessian(
        mol,
        hessian_kj_mol_a2,
        confId=confId,
        linear_angle_deg_cutoff=float(linear_angle_deg_cutoff),
    )

    return {
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
