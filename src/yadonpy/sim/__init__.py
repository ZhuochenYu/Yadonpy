"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from . import qm
from . import seminario
from .analyzer import AnalyzeResult
from .performance import IOAnalysisPolicy, resolve_io_analysis_policy

Psi4w = None
try:
    from .psi4_wrapper import Psi4w as Psi4w
except Exception:
    # Keep import optional; user will get a clear message when calling RESP.
    Psi4w = None

__all__ = ["qm", "seminario", "Psi4w", "AnalyzeResult", "IOAnalysisPolicy", "resolve_io_analysis_policy"]
