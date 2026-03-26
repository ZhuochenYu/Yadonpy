"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

#  Copyright (c) 2026. YadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.
#  Author: Teruki Tsurimoto @ Sekisui Chemical Co., Ltd.
# ******************************************************************************
# ff.dreiding module
# ******************************************************************************

import numpy as np
import json
from itertools import permutations
from rdkit import Chem
from ..core import calc, utils
from ..core.resources import ff_data_path
from . import ff_class
from .report import print_ff_assignment_report



class Dreiding():
    """
    Dreiding.Dreiding() class

    Forcefield object with typing rules for Dreiding model.
    By default reads data file in forcefields subdirectory.

    Attributes:
        ff_name: dreiding
        pair_style: lj
        bond_style: harmonic
        angle_style: harmonic
        dihedral_style: harmonic
        improper_style: umbrella
        ff_class: 1
    """
    def __init__(self, db_file=None):
        if db_file is None:
            db_file = str(ff_data_path("ff_dat", "dreiding.json"))
        self.param = self.load_ff_json(db_file)
        self.name = 'dreiding'
        self.pair_style = 'lj'
        self.bond_style = 'harmonic'
        self.angle_style = 'harmonic'
        self.dihedral_style = 'harmonic'
        self.improper_style = 'umbrella'
        self.ff_class = '1'
        self.param.c_c12 = 0.0
        self.param.c_c13 = 0.0
        self.param.c_c14 = 5/6
        self.param.lj_c12 = 0.0
        self.param.lj_c13 = 0.0
        self.param.lj_c14 = 0.5
        self.max_ring_size = 14
        self.alt_ptype = {}


    def ff_assign(self, mol, charge=None, retryMDL=True, useMDL=True, report: bool = True):
        """
        Dreiding.ff_assign

        Dreiding force field assignment for RDkit Mol object

        Args:
            mol: rdkit mol object

        Optional args:
            charge: Method of charge assignment. If None, charge assignment is skipped. 
            retryMDL: Retry assignment using MDL aromaticity model if default aromaticity model is failure (boolean)
            useMDL: Assignment using MDL aromaticity model (boolean)

        Returns: (boolean)
            True: Success assignment
            False: Failure assignment
        """

        if useMDL:
            Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
            Chem.rdmolops.SetAromaticity(mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL)

        mol.SetProp('ff_name', str(self.name))
        mol.SetProp('ff_class', str(self.ff_class))
        result = self.assign_ptypes(mol)
        if result: result = self.assign_btypes(mol)
        if result: result = self.assign_atypes(mol)
        if result: result = self.assign_dtypes(mol)
        if result: result = self.assign_itypes(mol)
        if result and charge is not None: result = calc.assign_charges(mol, charge=charge)
        
        if not result and retryMDL and not useMDL:
            utils.radon_print('Retry to assign with MDL aromaticity model', level=1)
            Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
            Chem.rdmolops.SetAromaticity(mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL)

            result = self.assign_ptypes(mol)
            if result: result = self.assign_btypes(mol)
            if result: result = self.assign_atypes(mol)
            if result: result = self.assign_dtypes(mol)
            if result: result = self.assign_itypes(mol)
            if result and charge is not None: result = calc.assign_charges(mol, charge=charge)
            if result: utils.radon_print('Success to assign with MDL aromaticity model', level=1)

        if result and report:
            print_ff_assignment_report(mol, ff_obj=self)
        if result:
            try:
                utils.auto_export_assigned_mol(mol, depth=2)
            except Exception:
                pass
        return mol if result else False


    def assign_ptypes(self, mol):
        """
        Dreiding.assign_ptypes

        Dreiding specific particle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('pair_style', self.pair_style)
        
        for p in mol.GetAtoms():
            p.bond_orders = [x.GetBondTypeAsDouble() for x in p.GetBonds()]
            if p.GetSymbol() == 'H':
                self.set_ptype(p, 'H_')
            elif p.GetSymbol() == 'C':
                if p.bond_orders and (4 in p.bond_orders or p.GetIsAromatic()):
                    self.set_ptype(p, 'C_R')
                elif p.GetTotalDegree()  == 4:
                    self.set_ptype(p, 'C_3')
                elif p.GetTotalDegree()  == 3:
                    self.set_ptype(p, 'C_2')
                elif p.GetTotalDegree()  == 2:
                    self.set_ptype(p, 'C_1')
                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False

            elif p.GetSymbol() == 'N':
                if p.bond_orders and (4 in p.bond_orders or p.GetIsAromatic()):
                    self.set_ptype(p, 'N_R')
                elif 2 in p.bond_orders:
                    self.set_ptype(p, 'N_2')
                elif 3 in p.bond_orders:
                    self.set_ptype(p, 'N_1')
                elif 1 in p.bond_orders:
                    for pb in p.GetNeighbors():
                        if pb.GetSymbol() == 'C' and pb.GetTotalDegree() == 3:
                            self.set_ptype(p, 'N_2')
                    if not p.GetProp('ff_type'):
                        self.set_ptype(p, 'N_3')
                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False

            elif p.GetSymbol() == 'O':
                if p.bond_orders and (4 in p.bond_orders or p.GetIsAromatic()):
                    self.set_ptype(p, 'O_R')
                elif 2 in p.bond_orders:
                    self.set_ptype(p, 'O_2')
                elif 3 in p.bond_orders:
                    self.set_ptype(p, 'O_1')
                elif 1 in p.bond_orders and len(set(p.bond_orders)) == 1:
                    self.set_ptype(p, 'O_3')
                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False

            elif p.GetSymbol() == 'F':
                self.set_ptype(p, 'F_')
            elif p.GetSymbol() == 'P':
                self.set_ptype(p, 'P_3')
            elif p.GetSymbol() == 'S':
                self.set_ptype(p, 'S_3')
            elif p.GetSymbol() == 'Cl':
                self.set_ptype(p, 'Cl')
            elif p.GetSymbol() == 'Br':
                self.set_ptype(p, 'Br')
            elif p.GetSymbol() == 'Si':
                self.set_ptype(p, 'Si3')
            elif p.GetSymbol() == 'B':
                if p.GetTotalDegree() == 4:
                    self.set_ptype(p, 'B_3')
                elif p.GetTotalDegree() == 3:
                    self.set_ptype(p, 'B_2')
                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False

            elif p.GetSymbol() == 'Al':
                self.set_ptype(p, 'Al3')
            elif p.GetSymbol() == 'Ga':
                self.set_ptype(p, 'Ga3')
            elif p.GetSymbol() == 'Ge':
                self.set_ptype(p, 'Ge3')
            elif p.GetSymbol() == 'As':
                self.set_ptype(p, 'As3')
            elif p.GetSymbol() == 'Se':
                self.set_ptype(p, 'Se3')
            elif p.GetSymbol() == 'In':
                self.set_ptype(p, 'In3')
            elif p.GetSymbol() == 'Sn':
                self.set_ptype(p, 'Sn3')
            elif p.GetSymbol() == 'Sb':
                self.set_ptype(p, 'Sb3')
            elif p.GetSymbol() == 'Te':
                self.set_ptype(p, 'Te3')
            elif p.GetSymbol() == 'I':
                self.set_ptype(p, 'I_')
            elif p.GetSymbol() == 'Na':
                self.set_ptype(p, 'Na')
            elif p.GetSymbol() == 'Ca':
                self.set_ptype(p, 'Ca')
            elif p.GetSymbol() == 'Fe':
                self.set_ptype(p, 'Fe')
            elif p.GetSymbol() == 'Zn':
                self.set_ptype(p, 'Zn')
            elif p.GetSymbol() == 'Ru':
                self.set_ptype(p, 'Ru')
            elif p.GetSymbol() == 'Ti':
                self.set_ptype(p, 'Ti')

            elif p.GetSymbol() == '*':
                p.SetProp('ff_type', '*')
                p.SetDoubleProp('ff_epsilon', 0.0)
                p.SetDoubleProp('ff_sigma', 0.0)

            ######################################
            # Assignment error
            ######################################
            else:
                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                result_flag = False


        ###########################################
        # Assignment of special atom type in GAFF
        ###########################################
        #if result_flag: self.assign_special_ptype(mol)
        
        
        return result_flag
        
        
    def set_ptype(self, p, pt):
        p.SetProp('ff_type', pt)
        p.SetDoubleProp('ff_epsilon', self.param.pt[pt].epsilon)
        p.SetDoubleProp('ff_sigma', self.param.pt[pt].sigma)
        
        return p
        
        
    def assign_btypes(self, mol):
        """
        Dreiding.assign_btypes

        Dreiding specific bond typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        alt_ptype = self.alt_ptype
        mol.SetProp('bond_style', self.bond_style)
        for b in mol.GetBonds():
            ba = b.GetBeginAtom().GetProp('ff_type')
            bb = b.GetEndAtom().GetProp('ff_type')
            bt = '%s,%s' % (ba, bb)
            
            result = self.set_btype(b, bt)
            if not result:
                alt1 = alt_ptype[ba] if ba in alt_ptype.keys() else None
                alt2 = alt_ptype[bb] if bb in alt_ptype.keys() else None
                if alt1 is None and alt2 is None:
                    utils.radon_print(('Can not assign this bond %s,%s' % (ba, bb)), level=2)
                    result_flag = False
                    continue
                
                bt_alt = []
                if alt1: bt_alt.append('%s,%s' % (alt1, bb))
                if alt2: bt_alt.append('%s,%s' % (ba, alt2))
                if alt1 and alt2: bt_alt.append('%s,%s' % (alt1, alt2))

                for bt in bt_alt:
                    result = self.set_btype(b, bt)
                    if result:
                        utils.radon_print('Using alternate bond type %s instead of %s,%s' % (bt, ba, bb))
                        break
                        
                if not b.HasProp('ff_type'):
                    utils.radon_print(('Can not assign this bond %s,%s' % (ba, bb)), level=2)
                    result_flag = False
                    
        return result_flag
    
    
    def set_btype(self, b, bt):
        if bt not in self.param.bt:
            return False
            
        b.SetProp('ff_type', self.param.bt[bt].tag)
        b.SetDoubleProp('ff_k', self.param.bt[bt].k)
        b.SetDoubleProp('ff_r0', self.param.bt[bt].r0)
        
        return True
        

    def assign_atypes(self, mol):
        """
        Dreiding.assign_atypes

        Dreiding specific angle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('angle_style', self.angle_style)
        setattr(mol, 'angles', [])

        for p in mol.GetAtoms():
            for p1 in p.GetNeighbors():
                for p2 in p.GetNeighbors():
                    if p1.GetIdx() == p2.GetIdx(): continue
                    unique = True
                    for ang in mol.angles:
                        if ((ang.a == p1.GetIdx() and ang.b == p.GetIdx() and ang.c == p2.GetIdx()) or
                            (ang.c == p1.GetIdx() and ang.b == p.GetIdx() and ang.a == p2.GetIdx())):
                            unique = False
                    if unique:
                        pt1 = p1.GetProp('ff_type')
                        pt = p.GetProp('ff_type')
                        pt2 = p2.GetProp('ff_type')
                        at = '%s,%s,%s' % (pt1, pt, pt2)
                        
                        result = self.set_atype(mol, a=p1.GetIdx(), b=p.GetIdx(), c=p2.GetIdx(), at=at)
                        
                        if not result:
                            utils.radon_print(('Can not assign this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                            result_flag = False
                            continue

        return result_flag
    
    def set_atype(self, mol, a, b, c, at):
        if at not in self.param.at:
            pt = at.split(',')
            at1 = 'X,%s,X' % (pt[1])
            if at1 in self.param.at:
                at = at1
            else:
                return False
    
        angle = utils.Angle(
            a=a, b=b, c=c,
            ff=ff_class.Angle_harmonic(
                ff_type=self.param.at[at].tag,
                k=self.param.at[at].k,
                theta0=self.param.at[at].theta0
            )
        )
        
        mol.angles.append(angle)
        
        return True
    

    def assign_dtypes(self, mol):
        """
        Dreiding.assign_dtypes

        Dreiding specific dihedral typing rules.
        
        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        alt_ptype = self.alt_ptype
        mol.SetProp('dihedral_style', self.dihedral_style)
        setattr(mol, 'dihedrals', [])
        
        for b in mol.GetBonds():
            p1 = b.GetBeginAtom()
            p2 = b.GetEndAtom()
            for p1b in p1.GetNeighbors():
                for p2b in p2.GetNeighbors():
                    if p1.GetIdx() == p2b.GetIdx() or p2.GetIdx() == p1b.GetIdx() or p1b.GetIdx() == p2b.GetIdx(): continue
                    unique = True
                    for dih in mol.dihedrals:
                        if ((dih.a == p1b.GetIdx() and dih.b == p1.GetIdx() and
                             dih.c == p2.GetIdx() and dih.d == p2b.GetIdx()) or
                            (dih.d == p1b.GetIdx() and dih.c == p1.GetIdx() and
                             dih.b == p2.GetIdx() and dih.a == p2b.GetIdx())):
                            unique = False
                    if unique:
                        p1bt = p1b.GetProp('ff_type')
                        p1t = p1.GetProp('ff_type')
                        p2t = p2.GetProp('ff_type')
                        p2bt = p2b.GetProp('ff_type')
                        dt = '%s,%s,%s,%s' % (p1bt, p1t, p2t, p2bt)
                        
                        result = self.set_dtype(mol, a=p1b.GetIdx(), b=p1.GetIdx(), c=p2.GetIdx(), d=p2b.GetIdx(), dt=dt)
                        
                        if not result:
                            alt1 = alt_ptype[p1t] if p1t in alt_ptype.keys() else None
                            alt2 = alt_ptype[p2t] if p2t in alt_ptype.keys() else None
                            if alt1 is None and alt2 is None:
                                utils.radon_print('Can not assign this dihedral %s,%s,%s,%s' % (p1bt, p1t, p2t, p2bt), level=2)
                                result_flag = False
                                continue
                            
                            dt_alt = []
                            if alt1: dt_alt.append('%s,%s,%s,%s' % (p1bt, alt1, p2t, p2bt))
                            if alt2: dt_alt.append('%s,%s,%s,%s' % (p1bt, p1t, alt2, p2bt))
                            if alt1 and alt2: dt_alt.append('%s,%s,%s,%s' % (p1bt, alt1, alt2, p2bt))
                            
                            for dt in dt_alt:
                                result = self.set_dtype(mol, a=p1b.GetIdx(), b=p1.GetIdx(), c=p2.GetIdx(), d=p2b.GetIdx(), dt=dt)
                                if result:
                                    utils.radon_print('Using alternate dihedral type %s instead of %s,%s,%s,%s' % (dt, p1bt, p1t, p2t, p2bt))
                                    break
                                    
                            if not result:
                                utils.radon_print(('Can not assign this dihedral %s,%s,%s,%s' % (p1bt, p1t, p2t, p2bt)), level=2)
                                result_flag = False
        
        return result_flag


    def set_dtype(self, mol, a, b, c, d, dt):
        if dt not in self.param.dt:
            pt = dt.split(',')
            dt1 = 'X,%s,%s,X' % (pt[1], pt[2])
            dt2 = 'X,%s,%s,%s' % (pt[1], pt[2], pt[3])
            dt3 = '%s,%s,%s,X' % (pt[0], pt[1], pt[2])
            if dt1 in self.param.dt:
                dt = dt1
            elif dt2 in self.param.dt:
                dt = dt2
            elif dt3 in self.param.dt:
                dt = dt3
            else:
                return False

        dihedral = utils.Dihedral(
            a=a, b=b, c=c, d=d,
            ff=ff_class.Dihedral_harmonic(
                ff_type=self.param.dt[dt].tag,
                k=self.param.dt[dt].k,
                d0=self.param.dt[dt].d,
                n=self.param.dt[dt].n
            )
        )
        
        mol.dihedrals.append(dihedral)
        
        return True


    def assign_itypes(self, mol):
        """
        Dreiding.assign_itypes

        Dreiding specific improper typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        alt_ptype = self.alt_ptype
        mol.SetProp('improper_style', self.improper_style)
        setattr(mol, 'impropers', [])
        
        for p in mol.GetAtoms():
            if len(p.GetNeighbors()) == 3:
                for perm in permutations(p.GetNeighbors(), 3):
                    pt = p.GetProp('ff_type')
                    p1t = perm[0].GetProp('ff_type')
                    p2t = perm[1].GetProp('ff_type')
                    p3t = perm[2].GetProp('ff_type')
                    it = '%s,%s,%s,%s' % (pt, p1t, p2t, p3t)
                    
                    result = self.set_itype(mol, a=p.GetIdx(), b=perm[0].GetIdx(), c=perm[1].GetIdx(), d=perm[2].GetIdx(), it=it)
                    
                    if not result:
                        alt1 = alt_ptype[pt] if pt in alt_ptype.keys() else None
                        alt2 = alt_ptype[p1t] if p1t in alt_ptype.keys() else None
                        alt3 = alt_ptype[p2t] if p2t in alt_ptype.keys() else None
                        alt4 = alt_ptype[p3t] if p3t in alt_ptype.keys() else None
                        if alt1 is None and alt2 is None and alt3 is None and alt4 is None:
                            break
                        
                        it_alt = []
                        if alt1: it_alt.append('%s,%s,%s,%s' % (alt1, p1t, p2t, p3t))
                        if alt2: it_alt.append('%s,%s,%s,%s' % (pt, alt2, p2t, p3t))
                        if alt3: it_alt.append('%s,%s,%s,%s' % (pt, p1t, alt3, p3t))
                        if alt4: it_alt.append('%s,%s,%s,%s' % (pt, p1t, p2t, alt4))

                        if alt1 and alt2: it_alt.append('%s,%s,%s,%s' % (alt1, alt2, p2t, p3t))
                        if alt1 and alt3: it_alt.append('%s,%s,%s,%s' % (alt1, p1t, alt3, p3t))
                        if alt1 and alt4: it_alt.append('%s,%s,%s,%s' % (alt1, p1t, p2t, alt4))
                        if alt2 and alt3: it_alt.append('%s,%s,%s,%s' % (pt, alt2, alt3, p3t))
                        if alt2 and alt4: it_alt.append('%s,%s,%s,%s' % (pt, alt2, p2t, alt4))
                        if alt3 and alt4: it_alt.append('%s,%s,%s,%s' % (pt, p1t, alt3, alt4))
                        
                        if alt1 and alt2 and alt3: it_alt.append('%s,%s,%s,%s' % (alt1, alt2, alt3, p3t))
                        if alt1 and alt2 and alt4: it_alt.append('%s,%s,%s,%s' % (alt1, alt2, p2t, alt4))
                        if alt1 and alt3 and alt4: it_alt.append('%s,%s,%s,%s' % (alt1, p1t, alt3, alt4))
                        if alt2 and alt3 and alt4: it_alt.append('%s,%s,%s,%s' % (pt, alt2, alt3, alt4))

                        if alt1 and alt2 and alt3 and alt4: it_alt.append('%s,%s,%s,%s' % (alt1, alt2, alt3, alt4))
                        
                        for it in it_alt:
                            result = self.set_itype(mol, a=p.GetIdx(), b=perm[0].GetIdx(), c=perm[1].GetIdx(), d=perm[2].GetIdx(), it=it)
                            if result:
                                utils.radon_print('Using alternate improper type %s instead of %s,%s,%s,%s' % (it, pt, p1t, p2t, p3t))
                                break
                    if result:
                        break
        
        return True            


    def set_itype(self, mol, a, b, c, d, it):
        if it not in self.param.it:
            pt = it.split(',')
            it1 = '%s,X,%s,%s' % (pt[0], pt[2], pt[3])
            it2 = '%s,X,X,%s' % (pt[0], pt[3])
            it3 = '%s,X,X,X' % (pt[0])
            it4 = 'X,X,X,%s' % (pt[3])
            if it1 in self.param.it:
                it = it1
            elif it2 in self.param.it:
                it = it2
            elif it3 in self.param.it:
                it = it3
            elif it4 in self.param.it:
                it = it4
            else:
                return False
        
        improper = utils.Improper(
            a=a, b=b, c=c, d=d,
            ff=ff_class.Improper_umbrella(
                ff_type=self.param.it[it].tag,
                k=self.param.it[it].k,
                x0=self.param.it[it].x0
            )
        )
        
        mol.impropers.append(improper)
        
        return True


    def load_ff_json(self, json_file):
        with open(json_file) as f:
            j = json.loads(f.read())

        ff = self.Container()
        ff.pt = {}
        ff.bt = {}
        ff.at = {}
        ff.dt = {}
        ff.it = {}

        ff.ff_name = j.get('ff_name')
        ff.ff_class = j.get('ff_class')
        ff.pair_style = j.get('pair_style')
        ff.bond_style = j.get('bond_style')
        ff.angle_style = j.get('angle_style')
        ff.dihedral_style = j.get('dihedral_style')
        ff.improper_style = j.get('improper_style')
        
        for pt in j.get('particle_types'):
            pt_obj = self.Container()
            for key in pt.keys():
                setattr(pt_obj, key, pt[key])
            ff.pt[pt['name']] = pt_obj
        
        for bt in j.get('bond_types'):
            bt_obj = self.Container()
            for key in bt.keys():
                setattr(bt_obj, key, bt[key])
            ff.bt[bt['name']] = bt_obj
            ff.bt[bt['rname']] = bt_obj
        
        for at in j.get('angle_types'):
            at_obj = self.Container()
            for key in at.keys():
                setattr(at_obj, key, at[key])
            ff.at[at['name']] = at_obj
            ff.at[at['rname']] = at_obj
        
        for dt in j.get('dihedral_types'):
            dt_obj = self.Container()
            for key in dt.keys():
                setattr(dt_obj, key, dt[key])
            ff.dt[dt['name']] = dt_obj
            ff.dt[dt['rname']] = dt_obj
        
        for it in j.get('improper_types'):
            it_obj = self.Container()
            for key in it.keys():
                setattr(it_obj, key, it[key])
            ff.it[it['name']] = it_obj
        
        return ff
            
    
    class Container(object):
        pass


    ## Backward compatibility
    class Angle_ff():
        """
            Dreiding.Angle_ff() object
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
    
    class Dihedral_ff():
        """
            Dreiding.Dihedral_ff() object
        """
        def __init__(self, ff_type=None, k=[], d0=[], n=[]):
            self.type = ff_type
            self.k = np.array(k)
            self.d0 = np.array(d0)
            self.d0_rad = np.array(d0)*(np.pi/180)
            self.n = np.array(n)
        
        def to_dict(self):
            dic = {
                'ff_type': str(self.type),
                'k': [float(x) for x in self.k],
                'd0': [float(x) for x in self.d0],
                'n': [int(x) for x in self.n],
            }
            return dic

        
    class Improper_ff():
        """
            Dreiding.Improper_ff() object
        """
        def __init__(self, ff_type=None, k=None, x0=None):
            self.type = ff_type
            self.k = k
            self.x0 = x0
        
        def to_dict(self):
            dic = {
                'ff_type': str(self.type),
                'k': float(self.k),
                'x0': int(self.x0),
            }
            return dic


class Dreiding_UT(Dreiding):
    """
    Dreiding.Dreiding_UT() class

    Forcefield object with typing rules for Dreiding model.
    By default reads data file in forcefields subdirectory.
    Dreiding_UT is modified Dreiding model that LJ parameters are re-fitted to use RESP charge model.
    K. Sasaki, T. Yamashita, J. Chem. Inf. Model. 2021, 61, 3, 1172–1179

    Attributes:
        ff_name: dreiding_ut
        pair_style: lj
        bond_style: harmonic
        angle_style: harmonic
        dihedral_style: harmonic
        improper_style: umbrella
        ff_class: 1
    """
    def __init__(self, db_file=None):
        if db_file is None:
            db_file = str(ff_data_path("ff_dat", "dreiding_ut.json"))
        super().__init__(db_file)
        self.name = 'dreiding_ut'
