"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .xvg import read_xvg


# SI constants
K_B_SI = 1.380649e-23  # J/K
R_SI = 8.31446261815324  # J/(mol*K)


@dataclass(frozen=True)
class SeriesStats:
    mean: float
    std: float
    n: int


def _last_window(values: np.ndarray, frac_last: float) -> np.ndarray:
    """Return the last `frac_last` portion of the series."""
    if not (0.0 < frac_last <= 1.0):
        raise ValueError(f"frac_last must be in (0,1], got {frac_last}")
    if values.size == 0:
        return values
    start = int(np.floor(values.size * (1.0 - frac_last)))
    start = max(0, min(start, values.size - 1))
    return values[start:]


def stats_from_xvg(xvg: Path, *, col: Optional[str] = None, frac_last: float = 0.5) -> SeriesStats:
    """Compute mean/std for an XVG column.

    Args:
        xvg: File path.
        col: Column name. If None, uses the first non-x column.
        frac_last: Use only the last fraction of the series.

    Returns:
        SeriesStats
    """
    x = read_xvg(xvg).df
    if col is None:
        candidates = [c for c in x.columns if c != "x"]
        if not candidates:
            raise ValueError(f"No data columns in XVG: {xvg}")
        col = candidates[0]
    if col not in x.columns:
        raise ValueError(f"Column '{col}' not found in {xvg}. Available: {list(x.columns)}")
    vals = _last_window(x[col].to_numpy(dtype=float), frac_last)
    return SeriesStats(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1) if vals.size > 1 else 0.0), n=int(vals.size))


def summarize_terms_xvg(xvg: Path, *, terms: Sequence[str], frac_last: float = 0.5) -> Dict[str, SeriesStats]:
    """Summarize multiple terms in a single XVG.

    `gmx energy` can output multiple series into one xvg; this function computes
    mean/std for each requested term.
    """
    df = read_xvg(xvg).df
    out: Dict[str, SeriesStats] = {}
    for t in terms:
        if t not in df.columns:
            continue
        vals = _last_window(df[t].to_numpy(dtype=float), frac_last)
        out[t] = SeriesStats(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1) if vals.size > 1 else 0.0), n=int(vals.size))
    return out


def kappa_t_from_volume(
    volume_nm3: np.ndarray,
    temperature_k: float,
    *,
    frac_last: float = 0.5,
) -> float:
    """Isothermal compressibility from NPT volume fluctuations.

    Formula (NPT ensemble):
      kappa_T = ( <V^2> - <V>^2 ) / (k_B T <V>)

    Args:
        volume_nm3: Volume time series (nm^3).
        temperature_k: Temperature (K).
        frac_last: Use last fraction for averaging.

    Returns:
        kappa_T in 1/Pa.
    """
    v_nm3 = _last_window(np.asarray(volume_nm3, dtype=float), frac_last)
    v_m3 = v_nm3 * 1e-27
    v_mean = float(np.mean(v_m3))
    v2_mean = float(np.mean(v_m3**2))
    var = v2_mean - v_mean**2
    if v_mean <= 0:
        return float("nan")
    return var / (K_B_SI * float(temperature_k) * v_mean)


def cp_molar_from_enthalpy(
    enthalpy_kj_mol: np.ndarray,
    temperature_k: float,
    *,
    frac_last: float = 0.5,
) -> float:
    """Molar heat capacity Cp from enthalpy fluctuations (NPT).

    Cp = ( <H^2> - <H>^2 ) / (R * T^2)

    Notes:
    - `gmx energy` typically reports energy-like terms in kJ/mol.
    - The returned Cp is in J/(mol*K).
    """
    h = _last_window(np.asarray(enthalpy_kj_mol, dtype=float), frac_last) * 1000.0  # J/mol
    if h.size == 0:
        return float("nan")
    var = float(np.mean(h**2) - np.mean(h) ** 2)
    if temperature_k <= 0:
        return float("nan")
    return var / (R_SI * float(temperature_k) ** 2)


def bulk_modulus_gpa_from_kappa_t(kappa_t_1_pa: float) -> float:
    """Bulk modulus K from isothermal compressibility kappa_T.

    K = 1 / kappa_T.

    Returns:
        Bulk modulus in GPa.
    """
    if not np.isfinite(kappa_t_1_pa) or kappa_t_1_pa <= 0:
        return float("nan")
    k_pa = 1.0 / float(kappa_t_1_pa)
    return k_pa / 1e9
