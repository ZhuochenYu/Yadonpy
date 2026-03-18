"""Backward-compatible facade for legacy ``yadonpy.core.utils``.

Historically, this module was a large grab-bag mixing topology containers,
RDKit I/O helpers, serialization, logging, and chemistry utilities.

Starting in 0.3.14, implementations are split into focused submodules
under ``yadonpy.core``. Public names remain available from this module to
avoid breaking downstream code.
"""

from __future__ import annotations

# Keep const accessible here (many callsites expect it)
from . import const  # noqa: F401

# Exceptions
from .exceptions import YadonPyError, RadonPyError  # noqa: F401

# Printing / logging
from .logging_utils import yadon_print, radon_print, tqdm_stub  # noqa: F401

# System
from .system import cpu_count  # noqa: F401

# Topology containers + copying
from .topology import (
    copy_topology_attributes,
    Angle,
    Dihedral,
    Improper,
    CMAP,
    Cell,
)  # noqa: F401

# Molecule ops
from .molops import (
    set_mol_id,
    count_mols,
    remove_atom,
    add_bond,
    remove_bond,
    add_angle,
    remove_angle,
    add_dihedral,
    remove_dihedral,
    add_improper,
    remove_improper,
)  # noqa: F401

# RDKit I/O
from .rdkit_io import (
    MolToPDBBlock,
    MolToPDBFile,
    StructureFromXYZFile,
    MolToExXYZBlock,
    MolToExXYZFile,
    MolToJSON,
    MolToJSON_dict,
    JSONToMol,
    JSONToMol_str,
    JSONToMol_dict,
)  # noqa: F401

# Serialization
from .serialization import (
    picklable,
    restore_picklable,
    pickle_dump,
    pickle_load,
    deepcopy_mol,
    picklable_const,
    restore_const,
)  # noqa: F401

# Chemistry helpers
from .chem_utils import (
    star2h,
    h2star,
    mol_from_smiles,
    is_inorganic_ion_like,
    is_inorganic_polyatomic_ion,
    _detect_ax_polyhedron,
    is_high_symmetry_polyhedral_ion,
    _kabsch_rotation,
    symmetrize_polyhedral_ion_geometry,
    ensure_3d_coords,
    scale_atomic_charges,
    restore_raw_charges,
    mol_from_pdb,
    is_in_ring,
)  # noqa: F401

# Naming helpers
from .naming import set_name_from_var, get_name, ensure_name, named  # noqa: F401
