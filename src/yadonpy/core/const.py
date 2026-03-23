"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

#  Copyright (c) 2026. YadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# core.const module
# ******************************************************************************
import os


def _env_flag(name: str, default: bool) -> bool:
	value = os.getenv(name)
	if value is None:
		return bool(default)
	text = str(value).strip().lower()
	if text in {"1", "true", "t", "yes", "y", "on"}:
		return True
	if text in {"0", "false", "f", "no", "n", "off"}:
		return False
	return bool(default)


def _env_float(name: str, default: float) -> float:
	value = os.getenv(name)
	if value is None:
		return float(default)
	try:
		return float(str(value).strip())
	except Exception:
		return float(default)


print_level = 1
tqdm_disable = _env_flag('YADONPY_DISABLE_TQDM', False)
rw_heartbeat_seconds = max(0.0, _env_float('YADONPY_RW_PROGRESS_INTERVAL', 15.0))
debug = True

# Do not check installing optional packages
check_package_disable = False

# Use mpi4py
mpi4py_avail = False

# Path to GROMACS executable (used by optional runners)
gmx_exec = os.getenv('GMX_EXEC', 'gmx')
gmx_mpi_exec = os.getenv('GMX_MPI_EXEC', 'gmx_mpi')

# Conversion factors
atm2pa = 101325
bohr2ang = 0.52917720859
cal2j = 4.184
au2kcal = 627.5095
au2ev = 27.2114
au2kj = au2kcal * cal2j
au2debye = 2.541765
ang2m = 1e-10
m2ang = 1e+10
ang2cm = 1e-8
cm2ang = 1e+8

# Physical constants
kB = 1.3806504e-23 # J/K
NA = 6.02214076e+23 # mol^-1
R = kB * NA # J/(K mol)
h = 6.62607015e-34 # J s
c = 2.99792458e+8 # m/s
e = 1.602e-19 # C
eps0 = 8.8541878128e-12 # F/m

# Constants
pdb_id = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

