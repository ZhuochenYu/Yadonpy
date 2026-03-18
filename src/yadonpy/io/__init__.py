"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from .gromacs_top import AmberGaffScaling, defaults_block
from .artifacts import write_molecule_artifacts

__all__ = ["AmberGaffScaling", "defaults_block", "write_molecule_artifacts"]
