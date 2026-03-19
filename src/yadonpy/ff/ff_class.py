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
# ff.ff_class module
# ******************************************************************************

import numpy as np



class Angle_harmonic():
    """
        Angle_harmonic() class
    """
    def __init__(self, ff_type=None, k=None, theta0=None):
        self.type = str(ff_type)
        self.k = float(k)
        self.theta0 = float(theta0)
        self.theta0_rad = float(theta0*(np.pi/180))
    
    def to_dict(self):
        dic = {
            'ff_type': str(self.type),
            'k': float(self.k),
            'theta0': float(self.theta0),
        }
        return dic
    
    
class Dihedral_fourier():
    """
        Dihedral_fourier() class
    """
    def __init__(self, ff_type=None, k=[], d0=[], m=None, n=[]):
        self.type = str(ff_type)
        self.k = np.array(k, dtype=float)
        self.d0 = np.array(d0, dtype=float)
        self.d0_rad = np.array(d0, dtype=float)*(np.pi/180)
        self.m = int(m)
        self.n = np.array(n, dtype=int)
    
    def to_dict(self):
        dic = {
            'ff_type': str(self.type),
            'k': [float(x) for x in self.k],
            'd0': [float(x) for x in self.d0],
            'm': int(self.m),
            'n': [int(x) for x in self.n],
        }
        return dic


class Dihedral_harmonic():
    """
        Dihedral_harmonic() class
    """
    def __init__(self, ff_type=None, k=None, d0=None, n=None):
        self.type = str(ff_type)
        self.k = float(k)
        self.d0 = int(d0)
        self.n = int(n)
    
    def to_dict(self):
        dic = {
            'ff_type': str(self.type),
            'k': float(self.k),
            'd0': int(self.d0),
            'n': int(self.n),
        }
        return dic

class Improper_cvff():
    """
        Improper_cvff() class
    """
    def __init__(self, ff_type=None, k=None, d0=-1, n=None):
        self.type = str(ff_type)
        self.k = float(k)
        self.d0 = int(d0)
        self.n = int(n)
    
    def to_dict(self):
        dic = {
            'ff_type': str(self.type),
            'k': float(self.k),
            'd0': int(self.d0),
            'n': int(self.n),
        }
        return dic


class Improper_umbrella():
    """
        Improper_umbrella() class
    """
    def __init__(self, ff_type=None, k=None, x0=None):
        self.type = str(ff_type)
        self.k = float(k)
        self.x0 = float(x0)
    
    def to_dict(self):
        dic = {
            'ff_type': str(self.type),
            'k': float(self.k),
            'x0': float(self.x0),
        }
        return dic

class CMAP_24():
    """
        CMAP_24() class
    """
    def __init__(self, ff_type=None, prm=[]):
        self.type = str(ff_type)
        self.prm = np.array(prm, dtype=float)
    
    def to_dict(self):
        dic = {
            'ff_type': str(self.type),
            'prm': [float(x) for x in self.prm],
        }
        return dic

