"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

"""Unit conversion helpers.

YadonPy's GAFF-family parameter json files use AMBER-style units:
  - length: Angstrom
  - energy: kcal/mol

GROMACS uses:
  - length: nm
  - energy: kJ/mol

These helpers implement the conversions needed for direct-parameter
topologies written by yadonpy.
"""

KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1


def angstrom_to_nm(x: float) -> float:
    return float(x) * ANGSTROM_TO_NM


def kcal_to_kj(x: float) -> float:
    return float(x) * KCAL_TO_KJ


def bond_k_kcal_per_a2_to_kj_per_nm2(k: float) -> float:
    """Convert bond force constant.

Amber/GAFF bond k is in kcal/mol/Å^2.
GROMACS harmonic bond uses kJ/mol/nm^2.
"""

    # kcal/mol/Å^2 -> kJ/mol/nm^2
    # multiply by 4.184 and by (Å^2/nm^2) = (0.1^2)^{-1} = 100
    return float(k) * KCAL_TO_KJ * 100.0


def angle_k_kcal_per_rad2_to_kj_per_rad2(k: float) -> float:
    """Convert angle force constant (kcal/mol/rad^2 -> kJ/mol/rad^2)."""
    return float(k) * KCAL_TO_KJ


def dihedral_k_kcal_to_kj(k: float) -> float:
    """Convert dihedral amplitude (kcal/mol -> kJ/mol)."""
    return float(k) * KCAL_TO_KJ
