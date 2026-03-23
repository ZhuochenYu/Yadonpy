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
# ff.tip module
# ******************************************************************************

import numpy as np
from rdkit import Chem
from ..core import utils
from . import ff_class
import sys
from rdkit import Geometry as Geom
from .report import print_ff_assignment_report



def make_atom(symbol, charge, name):
    atom = Chem.Atom(symbol)
    atom.SetDoubleProp('AtomicCharge', charge)
    info = Chem.AtomPDBResidueInfo()
    info.SetResidueName('WAT')
    atom.SetPDBResidueInfo(info)
    atom.SetProp('ff_type', name)

    return atom

class TIP3P():
    """
    tip.TIP3P() class

    Forcefield object with typing rules for TIP3P water model.

    Attributes:
        ff_name: tip3p
        pair_style: lj
        bond_style: harmonic
        angle_style: harmonic
        #dihedral_style: fourier
        #improper_style: cvff
    """

    def __init__(self):
        self.name = 'tip3p'
        self.pair_style = 'lj'
        self.bond_style = 'harmonic'
        self.angle_style = 'harmonic'
        self.dihedral_style = 'fourier'
        self.improper_style = 'cvff'

    @staticmethod
    def mol():
        rwmol = Chem.RWMol()

        rwmol.AddAtom(make_atom('O', -0.834, 'OW'))
        rwmol.AddAtom(make_atom('H',  0.417, 'HW'))
        rwmol.AddAtom(make_atom('H',  0.417, 'HW'))

        rwmol.AddBond(0, 1, Chem.BondType.SINGLE)
        rwmol.AddBond(0, 2, Chem.BondType.SINGLE)
    
        conf = Chem.Conformer()
        conf.SetAtomPosition(0, Geom.Point3D( 0.00000, -0.06556,  0.00000))
        conf.SetAtomPosition(1, Geom.Point3D( 0.75695,  0.52032,  0.00000))
        conf.SetAtomPosition(2, Geom.Point3D(-0.75695,  0.52032,  0.00000))
        rwmol.AddConformer(conf)
    
        return rwmol.GetMol()
        
    def ff_assign(self, mol, report: bool = True):
        """
        TIP3P.ff_assign

        TIP3P force field assignment for RDkit Mol object

        Args:
            mol: rdkit mol object

        Optional args:

        Returns: (boolean)
            True: Success assignment
            False: Failure assignment
        """

        if mol.GetNumAtoms() != 3:
            utils.radon_print('Cannot assign force field for TIP3P water.')
            sys.exit()
        
        mol.SetProp('ff_name', str(self.name))

        result = self.assign_ptypes(mol)
        if result: result = self.assign_btypes(mol)
        if result: result = self.assign_atypes(mol)
        if result: result = self.assign_dtypes(mol)
        if result: result = self.assign_itypes(mol)

        if result and report:
            print_ff_assignment_report(mol, ff_obj=self)

        return mol if result else False


    def assign_ptypes(self, mol):
        """
        TIP3P.assign_ptypes

        TIP3P specific particle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('pair_style', self.pair_style)

        self.set_ptype(mol.GetAtomWithIdx(0), "OTIP", [0.1550, 3.1536])
        self.set_ptype(mol.GetAtomWithIdx(1), "HTIP", [0.0   , 1.0   ])
        self.set_ptype(mol.GetAtomWithIdx(2), "HTIP", [0.0   , 1.0   ])
      
        return result_flag

    def set_ptype(self, p, pt, pair_coeff):
        p.SetProp('ff_type', pt)
        p.SetDoubleProp('ff_epsilon', pair_coeff[0])
        p.SetDoubleProp('ff_sigma', pair_coeff[1])

        return p
        
        
    def assign_btypes(self, mol):
        """
        TIP3P.assign_btypes

        TIP3P specific bond typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('bond_style', self.bond_style)

        self.set_btype(mol.GetBondWithIdx(0), "OTIP,HTIP", [0.0, 0.9574])
        self.set_btype(mol.GetBondWithIdx(1), "OTIP,HTIP", [0.0, 0.9574])
        
        return result_flag
    

    def set_btype(self, b, bt, bond_coeff):

        b.SetProp('ff_type', bt)
        b.SetDoubleProp('ff_k', bond_coeff[0])
        b.SetDoubleProp('ff_r0', bond_coeff[1])

        return True
        
    def assign_atypes(self, mol):
        """
        TIP3P.assign_atypes

        TIP3P specific angle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('angle_style', self.angle_style)
        setattr(mol, 'angles', {})

        at = "HTIP,OIP,HTIP"
        
        result = self.set_atype(mol, 1, 0, 2, at, [0.0, 104.52])
        if not result:
            result_flag = False
            
        return result_flag

    def set_atype(self, mol, a, b, c, at, coeff):
            
        angle = utils.Angle(
            a=a, b=b, c=c,
            ff=ff_class.Angle_harmonic(
                ff_type=at,
                k=coeff[0],
                theta0=coeff[1]
            )
        )
        
        key = '%i,%i,%i' % (a, b, c)
        mol.angles[key] = angle
        
        return True

    def assign_dtypes(self, mol):
        """
        TIP3P.assign_dtypes

        TIP3P specific dihedral typing rules.
        
        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('dihedral_style', self.dihedral_style)
        setattr(mol, 'dihedrals', {})

        return result_flag

    def assign_itypes(self, mol):
        """
        TIP3P.assign_itypes

        TIP3P specific improper typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        mol.SetProp('improper_style', self.improper_style)
        setattr(mol, 'impropers', {})
        
        return True

    
    ## Backward compatibility
    class Angle_ff():
        """
            TIP3P.Angle_ff() object
        """
        def __init__(self, ff_type=None, k=None, theta0=None):
            self.type = ff_type
            self.k = k
            self.theta0 = theta0
            self.theta0_rad = theta0*(np.pi/180)

        def to_dict(self):
            dic = {
                'ff_type': str(self.type),
                'k': float(self.k),
                'theta0': float(self.theta0),
            }
            return dic
        
        
class TIP4P():
    """
    tip.TIP4P() class

    Forcefield object with typing rules for TIP4P water model.

    Attributes:
        ff_name: tip4p
        pair_style: lj
        bond_style: harmonic
        angle_style: harmonic
        #dihedral_style: fourier
        #improper_style: cvff
    """

    def __init__(self):
        self.name = 'tip4p'
        self.pair_style = 'lj'
        self.bond_style = 'harmonic'
        self.angle_style = 'harmonic'
        self.dihedral_style = 'fourier'
        self.improper_style = 'cvff'

    @staticmethod
    def mol():
        rwmol = Chem.RWMol()

        rwmol.AddAtom(make_atom('O',  0.000, 'OW'))
        rwmol.AddAtom(make_atom('H',  0.520, 'HW'))
        rwmol.AddAtom(make_atom('H',  0.520, 'HW'))
        rwmol.AddAtom(make_atom(0  , -1.040, 'MW'))

        rwmol.AddBond(0, 1, Chem.BondType.SINGLE)
        rwmol.AddBond(0, 2, Chem.BondType.SINGLE)
        rwmol.AddBond(0, 3, Chem.BondType.ZERO  )

        conf = Chem.Conformer()
        conf.SetAtomPosition(0, Geom.Point3D( 0.00000, -0.06556,  0.00000))
        conf.SetAtomPosition(1, Geom.Point3D( 0.75695,  0.52032,  0.00000))
        conf.SetAtomPosition(2, Geom.Point3D(-0.75695,  0.52032,  0.00000))
        conf.SetAtomPosition(3, Geom.Point3D( 0.00000,  0.08444,  0.00000))
        rwmol.AddConformer(conf)
    
        return rwmol.GetMol()
        
    def ff_assign(self, mol, report: bool = True):
        """
        TIP4P.ff_assign

        TIP4P force field assignment for RDkit Mol object

        Args:
            mol: rdkit mol object

        Optional args:

        Returns: (boolean)
            True: Success assignment
            False: Failure assignment
        """

        if mol.GetNumAtoms() != 4:
            utils.radon_print('Cannot assign force field for TIP4P water.')
            sys.exit()
        
        mol.SetProp('ff_name', str(self.name))

        result = self.assign_ptypes(mol)
        if result: result = self.assign_btypes(mol)
        if result: result = self.assign_atypes(mol)
        if result: result = self.assign_dtypes(mol)
        if result: result = self.assign_itypes(mol)

        if result and report:
            print_ff_assignment_report(mol, ff_obj=self)

        return mol if result else False


    def assign_ptypes(self, mol):
        """
        TIP4P.assign_ptypes

        TIP4P specific particle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('pair_style', self.pair_style)

        self.set_ptype(mol.GetAtomWithIdx(0), "OTIP", [0.1550, 3.1536])
        self.set_ptype(mol.GetAtomWithIdx(1), "HTIP", [0.0   , 1.0   ])
        self.set_ptype(mol.GetAtomWithIdx(2), "HTIP", [0.0   , 1.0   ])
        self.set_ptype(mol.GetAtomWithIdx(3), "MTIP", [0.0   , 1.0   ])

        return result_flag

    def set_ptype(self, p, pt, pair_coeff):
        p.SetProp('ff_type', pt)
        p.SetDoubleProp('ff_epsilon', pair_coeff[0])
        p.SetDoubleProp('ff_sigma', pair_coeff[1])

        return p
        
        
    def assign_btypes(self, mol):
        """
        TIP4P.assign_btypes

        TIP4P specific bond typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('bond_style', self.bond_style)

        self.set_btype(mol.GetBondWithIdx(0), "OTIP,HTIP", [0.0, 0.9572])
        self.set_btype(mol.GetBondWithIdx(1), "OTIP,HTIP", [0.0, 0.9572])
        self.set_btype(mol.GetBondWithIdx(2), "OTIP,MTIP", [0.0, 0.1500])

        return result_flag
    

    def set_btype(self, b, bt, bond_coeff):

        b.SetProp('ff_type', bt)
        b.SetDoubleProp('ff_k', bond_coeff[0])
        b.SetDoubleProp('ff_r0', bond_coeff[1])

        return True
        
    def assign_atypes(self, mol):
        """
        TIP4P.assign_atypes

        TIP4P specific angle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('angle_style', self.angle_style)
        setattr(mol, 'angles', {})

        at = "HTIP,OTIP,HTIP"
        
        result = self.set_atype(mol, 1, 0, 2, at, [0.0, 104.52])
        if not result:
            result_flag = False
            
        return result_flag

    def set_atype(self, mol, a, b, c, at, coeff):
            
        angle = utils.Angle(
            a=a, b=b, c=c,
            ff=ff_class.Angle_harmonic(
                ff_type=at,
                k=coeff[0],
                theta0=coeff[1]
            )
        )
        
        key = '%i,%i,%i' % (a, b, c)
        mol.angles[key] = angle
        
        return True

    def assign_dtypes(self, mol):
        """
        TIP4P.assign_dtypes

        TIP4P specific dihedral typing rules.
        
        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('dihedral_style', self.dihedral_style)
        setattr(mol, 'dihedrals', {})

        return result_flag

    def assign_itypes(self, mol):
        """
        TIP4P.assign_itypes

        TIP4P specific improper typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        mol.SetProp('improper_style', self.improper_style)
        setattr(mol, 'impropers', {})
        
        return True

    
    ## Backward compatibility
    class Angle_ff():
        """
            TIP4P.Angle_ff() object
        """
        def __init__(self, ff_type=None, k=None, theta0=None):
            self.type = ff_type
            self.k = k
            self.theta0 = theta0
            self.theta0_rad = theta0*(np.pi/180)

        def to_dict(self):
            dic = {
                'ff_type': str(self.type),
                'k': float(self.k),
                'theta0': float(self.theta0),
            }
            return dic
        
class TIP5P():
    """
    tip.TIP5P() class

    Forcefield object with typing rules for TIP5P water model.

    Attributes:
        ff_name: tip5p
        pair_style: lj
        bond_style: harmonic
        angle_style: harmonic
        #dihedral_style: fourier
        #improper_style: cvff
    """

    def __init__(self):
        self.name = 'tip5p'
        self.pair_style = 'lj'
        self.bond_style = 'harmonic'
        self.angle_style = 'harmonic'
        self.dihedral_style = 'fourier'
        self.improper_style = 'cvff'

    @staticmethod
    def mol():
        rwmol = Chem.RWMol()

        rwmol.AddAtom(make_atom('O',  0.000, 'OW'))
        rwmol.AddAtom(make_atom('H',  0.241, 'HW'))
        rwmol.AddAtom(make_atom('H',  0.241, 'HW'))
        rwmol.AddAtom(make_atom(0  , -0.241, 'LW'))
        rwmol.AddAtom(make_atom(0  , -0.241, 'LW'))

        rwmol.AddBond(0, 1, Chem.BondType.SINGLE)
        rwmol.AddBond(0, 2, Chem.BondType.SINGLE)
        rwmol.AddBond(0, 3, Chem.BondType.ZERO  )
        rwmol.AddBond(0, 4, Chem.BondType.ZERO  )

        conf = Chem.Conformer()
        conf.SetAtomPosition(0, Geom.Point3D( 0.00000, -0.06556,  0.00000))
        conf.SetAtomPosition(1, Geom.Point3D( 0.75695,  0.52032,  0.00000))
        conf.SetAtomPosition(2, Geom.Point3D(-0.75695,  0.52032,  0.00000))
        conf.SetAtomPosition(3, Geom.Point3D( 0.00000, -0.46971,  0.57154))
        conf.SetAtomPosition(4, Geom.Point3D( 0.00000, -0.46971, -0.57154))
        rwmol.AddConformer(conf)
    
        return rwmol.GetMol()
        
    def ff_assign(self, mol, report: bool = True):
        """
        TIP5P.ff_assign

        TIP5P force field assignment for RDkit Mol object

        Args:
            mol: rdkit mol object

        Optional args:

        Returns: (boolean)
            True: Success assignment
            False: Failure assignment
        """

        if mol.GetNumAtoms() != 5:
            utils.radon_print('Cannot assign force field for TIP5P water.')
            sys.exit()
        
        mol.SetProp('ff_name', str(self.name))

        result = self.assign_ptypes(mol)
        if result: result = self.assign_btypes(mol)
        if result: result = self.assign_atypes(mol)
        if result: result = self.assign_dtypes(mol)
        if result: result = self.assign_itypes(mol)

        if result and report:
            print_ff_assignment_report(mol, ff_obj=self)

        return mol if result else False


    def assign_ptypes(self, mol):
        """
        TIP5P.assign_ptypes

        TIP5P specific particle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('pair_style', self.pair_style)

        self.set_ptype(mol.GetAtomWithIdx(0), "OTIP", [0.160, 3.12])
        self.set_ptype(mol.GetAtomWithIdx(1), "HTIP", [0.0  , 1.0 ])
        self.set_ptype(mol.GetAtomWithIdx(2), "HTIP", [0.0  , 1.0 ])
        self.set_ptype(mol.GetAtomWithIdx(3), "LTIP", [0.0  , 1.0 ])
        self.set_ptype(mol.GetAtomWithIdx(4), "LTIP", [0.0  , 1.0 ])

        return result_flag

    def set_ptype(self, p, pt, pair_coeff):
        p.SetProp('ff_type', pt)
        p.SetDoubleProp('ff_epsilon', pair_coeff[0])
        p.SetDoubleProp('ff_sigma', pair_coeff[1])

        return p
        
        
    def assign_btypes(self, mol):
        """
        TIP5P.assign_btypes

        TIP5P specific bond typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('bond_style', self.bond_style)

        self.set_btype(mol.GetBondWithIdx(0), "OTIP,HTIP", [0.0, 0.9572])
        self.set_btype(mol.GetBondWithIdx(1), "OTIP,HTIP", [0.0, 0.9572])
        self.set_btype(mol.GetBondWithIdx(2), "OTIP,LTIP", [0.0, 0.7000])
        self.set_btype(mol.GetBondWithIdx(3), "OTIP,LTIP", [0.0, 0.7000])

        return result_flag
    

    def set_btype(self, b, bt, bond_coeff):

        b.SetProp('ff_type', bt)
        b.SetDoubleProp('ff_k', bond_coeff[0])
        b.SetDoubleProp('ff_r0', bond_coeff[1])

        return True
        
    def assign_atypes(self, mol):
        """
        TIP5P.assign_atypes

        TIP5P specific angle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('angle_style', self.angle_style)
        setattr(mol, 'angles', {})

        self.set_atype(mol, 1, 0, 2, "HTIP,OTIP,HTIP", [0.0, 104.52])
        self.set_atype(mol, 3, 0, 4, "LTIP,OTIP,LTIP", [0.0, 109.47])
            
        return result_flag

    def set_atype(self, mol, a, b, c, at, coeff):
            
        angle = utils.Angle(
            a=a, b=b, c=c,
            ff=ff_class.Angle_harmonic(
                ff_type=at,
                k=coeff[0],
                theta0=coeff[1]
            )
        )
        
        key = '%i,%i,%i' % (a, b, c)
        mol.angles[key] = angle
        
        return True

    def assign_dtypes(self, mol):
        """
        TIP5P.assign_dtypes

        TIP5P specific dihedral typing rules.
        
        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('dihedral_style', self.dihedral_style)
        setattr(mol, 'dihedrals', {})

        return result_flag

    def assign_itypes(self, mol):
        """
        TIP5P.assign_itypes

        TIP5P specific improper typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        mol.SetProp('improper_style', self.improper_style)
        setattr(mol, 'impropers', {})
        
        return True

    
    ## Backward compatibility
    class Angle_ff():
        """
            TIP5P.Angle_ff() object
        """
        def __init__(self, ff_type=None, k=None, theta0=None):
            self.type = ff_type
            self.k = k
            self.theta0 = theta0
            self.theta0_rad = theta0*(np.pi/180)

        def to_dict(self):
            dic = {
                'ff_type': str(self.type),
                'k': float(self.k),
                'theta0': float(self.theta0),
            }
            return dic
        
        
