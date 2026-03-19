"""RDKit I/O helpers (PDB/XYZ/ExtendedXYZ/JSON)."""

from __future__ import annotations
import os
import json
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import rdGeometry as Geom
from .logging_utils import radon_print
from .topology import copy_topology_attributes

def MolToPDBBlock(mol, confId=0):
    """
    utils.MolToPDBBlock

    Convert RDKit Mol object to PDB block

    Args:
        mol: RDkit Mol object

    Optional args:
        confId: Target conformer ID (int)

    Returns:
        PDB block (str, array)
    """

    coord = mol.GetConformer(confId).GetPositions()
    PDBBlock = ['TITLE    pdb written using YadonPy']
    conect = []
    serial = 0
    ter = 0
    chainid_pre = 1
    chainid_pdb_pre = ''

    for i, atom in enumerate(mol.GetAtoms()):
        serial += 1
        resinfo = atom.GetPDBResidueInfo()
        if resinfo is None: return None

        chainid_pdb = resinfo.GetChainId()
        chainid = atom.GetIntProp('mol_id')
        if chainid != chainid_pre:
            if chainid_pdb_pre:
                PDBBlock.append('TER   %5i      %3s %1s%4i%1s' % (serial, resname, chainid_pdb_pre, resnum, icode))
            else:
                PDBBlock.append('TER   %5i      %3s %1s%4i%1s' % (serial, resname, '*', resnum, icode))
            ter += 1
            serial += 1

        record = 'HETATM' if resinfo.GetIsHeteroAtom() else 'ATOM  '
        name = atom.GetProp('ff_type') if atom.HasProp('ff_type') else atom.GetSymbol()
        altLoc = resinfo.GetAltLoc()
        resname = resinfo.GetResidueName()
        resnum = resinfo.GetResidueNumber()
        icode = resinfo.GetInsertionCode()
        x = coord[i][0]
        y = coord[i][1]
        z = coord[i][2]
        occ = resinfo.GetOccupancy() if resinfo.GetOccupancy() else 1.0
        tempf = resinfo.GetTempFactor() if resinfo.GetTempFactor() else 0.0

        if chainid_pdb:
            line = '%-6s%5i %4s%1s%3s %1s%4i%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s' % (
                    record, serial, name, altLoc, resname, chainid_pdb, resnum, icode, x, y, z, occ, tempf, atom.GetSymbol())
        else:
            line = '%-6s%5i %4s%1s%3s %1s%4i%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s' % (
                    record, serial, name, altLoc, resname, '*', resnum, icode, x, y, z, occ, tempf, atom.GetSymbol())

        PDBBlock.append(line)

        chainid_pre = chainid
        chainid_pdb_pre = chainid_pdb

        if len(atom.GetNeighbors()) > 0:
            flag = False
            conect_line = 'CONECT%5i' % (serial)
            for na in atom.GetNeighbors():
                if atom.GetIdx() < na.GetIdx():
                    conect_line += '%5i' % (na.GetIdx()+1+ter)
                    flag = True
            if flag:
                conect.append(conect_line)

    PDBBlock.append('TER   %5i      %3s %1s%4i%1s' % (serial+1, resname, chainid_pre, resnum, icode))
    PDBBlock.extend(conect)
    PDBBlock.append('END')

    return PDBBlock


def MolToPDBFile(mol, filename, confId=0):
    """
    utils.MolToPDBFile

    Convert RDKit Mol object to PDB file

    Args:
        mol: RDkit Mol object
        filename: Output pdb filename (str)

    Optional args:
        confId: Target conformer ID (int)

    Returns:
        Success or fail (boolean)
    """

    mol = set_mol_id(mol)
    PDBBlock = MolToPDBBlock(mol, confId=confId)
    if PDBBlock is None: return False

    with open(filename, 'w') as fh:
        fh.write('\n'.join(PDBBlock)+'\n')
        fh.flush()
        if hasattr(os, 'fdatasync'):
            os.fdatasync(fh.fileno())
        else:
            os.fsync(fh.fileno())

    return True


def StructureFromXYZFile(filename):
    with open(filename, 'r') as fh:
        lines = [s.strip() for s in fh.readlines()]

    strucs = []
    struc = []
    t_flag = False
    n_flag = False
    n_atom = 0
    c_atom = 0
    for line in lines:
        if not n_flag:
            if line.isdecimal():
                n_atom = line
                n_flag = True
        elif not t_flag:
            t_flag = True
            continue
        else:
            c_atom += 1
            element, x, y, z = line.split()
            struc.append([element, x, y, z])
            if c_atom >= n_atom:
                strucs.append(struc)
                t_flag = False
                n_flag = False
                n_atom = 0
                c_atom = 0

    return strucs


def MolToExXYZBlock(mol, confId=0):

    XYZBlock = Chem.MolToXYZBlock(mol, confId=confId)
    XYZBlock = XYZBlock.split('\n')
    if mol.GetConformer(confId).HasProp('cell_xhi'):
        conf = mol.GetConformer(confId)
        cell_line = 'Lattice=\"%.4f 0.0 0.0 0.0 %.4f 0.0 0.0 0.0 %.4f\"' % (
            conf.GetDoubleProp('cell_dx'), conf.GetDoubleProp('cell_dy'), conf.GetDoubleProp('cell_dz'))
        XYZBlock[1] = cell_line
    elif hasattr(mol, 'cell'):
        cell_line = 'Lattice=\"%.4f 0.0 0.0 0.0 %.4f 0.0 0.0 0.0 %.4f\"' % (mol.cell.dx, mol.cell.dy, mol.cell.dz)
        XYZBlock[1] = cell_line

    return XYZBlock


def MolToExXYZFile(mol, filename, confId=0):

    XYZBlock = MolToExXYZBlock(mol, confId=confId)
    if XYZBlock is None: return False
    with open(filename, 'w') as fh:
        fh.write('\n'.join(XYZBlock)+'\n')
        fh.flush()
        if hasattr(os, 'fdatasync'):
            os.fdatasync(fh.fileno())
        else:
            os.fsync(fh.fileno())

    return True


def MolToJSON(mol, file, useRDKitExtensions=False):
    json_dict = MolToJSON_dict(mol, useRDKitExtensions=useRDKitExtensions)
    with open(file, mode='w') as f:
        json.dump(json_dict, f, indent=2)


def MolToJSON_dict(mol, useRDKitExtensions=False):
    Chem.SanitizeMol(mol)
    if hasattr(Chem.rdMolInterchange, 'JSONWriteParameters'):
        params = Chem.rdMolInterchange.JSONWriteParameters()
        params.useRDKitExtensions = useRDKitExtensions
        json_str = Chem.rdMolInterchange.MolToJSON(mol, params=params)
    else:
        json_str = Chem.rdMolInterchange.MolToJSON(mol)
    json_dict = json.loads(json_str)
    
    radonpy_ext = {
        'name': 'radonpy_extention',
        "formatVersion": 1,
        'lib_version': __yadonpy_version__,
    }

    atom_prop = []
    for a in mol.GetAtoms():
        atom_data = {}
        # FF on atoms
        if a.HasProp('ff_type'):
            atom_data['ff_type'] = a.GetProp('ff_type')
        if a.HasProp('ff_epsilon'):
            atom_data['ff_epsilon'] = a.GetDoubleProp('ff_epsilon')
        if a.HasProp('ff_sigma'):
            atom_data['ff_sigma'] = a.GetDoubleProp('ff_sigma')

        # charge
        if a.HasProp('AtomicCharge'):
            atom_data['AtomicCharge'] = a.GetDoubleProp('AtomicCharge')
        if a.HasProp('RESP'):
            atom_data['RESP'] = a.GetDoubleProp('RESP')
        if a.HasProp('ESP'):
            atom_data['ESP'] = a.GetDoubleProp('ESP')
        if a.HasProp('Mulliken'):
            atom_data['Mulliken'] = a.GetDoubleProp('Mulliken')
        if a.HasProp('Lowdin'):
            atom_data['Lowdin'] = a.GetDoubleProp('Lowdin')
        if a.HasProp('_GasteigerCharge'):
            atom_data['_GasteigerCharge'] = a.GetProp('_GasteigerCharge')
    
        # velocity
        if a.HasProp('vx'):
            atom_data['vx'] = a.GetDoubleProp('vx')
            atom_data['vy'] = a.GetDoubleProp('vy')
            atom_data['vz'] = a.GetDoubleProp('vz')

        # crosslinking
        if a.HasProp('CL_remove'):
            atom_data['CL_remove'] = a.GetBoolProp('CL_remove')
        if a.HasProp('CL_react'):
            atom_data['CL_react'] = a.GetBoolProp('CL_react')

        # others
        atom_data['isotope'] = a.GetIsotope()
        if a.HasProp('mol_id'):
            atom_data['mol_id'] = a.GetIntProp('mol_id')

        # PDB
        resinfo = a.GetPDBResidueInfo()
        if resinfo is not None:
            atom_data['ResidueName'] = resinfo.GetResidueName()
            atom_data['ResidueNumber'] = resinfo.GetResidueNumber()

        atom_prop.append(atom_data)
    radonpy_ext['atoms'] = atom_prop


    bond_prop = []
    for b in mol.GetBonds():
        bond_data = {}
        # FF on bonds
        if b.HasProp('ff_type'):
            bond_data['ff_type'] = b.GetProp('ff_type')
        if b.HasProp('ff_k'):
            bond_data['ff_k'] = b.GetDoubleProp('ff_k')
        if b.HasProp('ff_r0'):
            bond_data['ff_r0'] = b.GetDoubleProp('ff_r0')

        # crosslinking
        if b.HasProp('CL_new_bond'):
            bond_data['CL_new_bond'] = b.GetBoolProp('CL_new_bond')

        # others
        if b.HasProp('new_bond'):
            bond_data['new_bond'] = b.GetBoolProp('new_bond')

        bond_prop.append(bond_data)
    radonpy_ext['bonds'] = bond_prop

 
    # angle
    if hasattr(mol, 'angles'):
        if len(mol.angles) > 0 and hasattr(mol.angles[list(mol.angles.keys())[0]], 'to_dict'):
            angle_prop = [ang.to_dict() for key, ang in mol.angles.items()]
            radonpy_ext['angles'] = angle_prop
        else:
            angle_prop = []
            for key, ang in mol.angles.items():
                dic = {
                    'a': int(ang.a),
                    'b': int(ang.b),
                    'c': int(ang.c),
                    'ff': {
                        'ff_type': str(ang.ff.type),
                        'k': float(ang.ff.k),
                        'theta0': float(ang.ff.theta0),
                    }
                }
                angle_prop.append(dic)
            radonpy_ext['angles'] = angle_prop
    else:
        angle_prop = []
    
    # dihedral
    if hasattr(mol, 'dihedrals'):
        if len(mol.dihedrals) > 0 and hasattr(mol.dihedrals[list(mol.dihedrals.keys())[0]], 'to_dict'):
            dihedral_prop = [dih.to_dict() for key, dih in mol.dihedrals.items()]
            radonpy_ext['dihedrals'] = dihedral_prop
        else:
            dihedral_prop = []
            for key, dih in mol.dihedrals.items():
                dic = {
                    'a': int(dih.a),
                    'b': int(dih.b),
                    'c': int(dih.c),
                    'd': int(dih.d),
                    'ff': {
                        'ff_type': str(dih.ff.type),
                        'k': list([float(x) for x in dih.ff.k]),
                        'd0': list([float(x) for x in dih.ff.d0]),
                        'm': int(dih.ff.m),
                        'n': list([int(x) for x in dih.ff.n]),
                    }
                }
                dihedral_prop.append(dic)
            radonpy_ext['dihedrals'] = dihedral_prop
    else:
        dihedral_prop = []

    # improper
    if hasattr(mol, 'impropers'):
        if len(mol.impropers) > 0 and hasattr(mol.impropers[list(mol.impropers.keys())[0]], 'to_dict'):
            improper_prop = [imp.to_dict() for key, imp in mol.impropers.items()]
            radonpy_ext['impropers'] = improper_prop
        else:
            improper_prop = []
            for key, imp in mol.impropers.items():
                dic = {
                    'a': int(imp.a),
                    'b': int(imp.b),
                    'c': int(imp.c),
                    'd': int(imp.d),
                    'ff': {
                        'ff_type': str(imp.ff.type),
                        'k': float(imp.ff.k),
                        'd0': int(imp.ff.d0),
                        'n': int(imp.ff.n),
                    }
                }
                improper_prop.append(dic)
            radonpy_ext['impropers'] = improper_prop
    else:
        improper_prop = []

    # cell
    if hasattr(mol, 'cell'):
        if hasattr(mol.cell, 'to_dict'):
            cell_prop = mol.cell.to_dict()
            radonpy_ext['cell'] = cell_prop
        else:
            cell_prop = {
                'xhi': float(mol.cell.xhi),
                'xlo': float(mol.cell.xlo),
                'yhi': float(mol.cell.yhi),
                'ylo': float(mol.cell.ylo),
                'zhi': float(mol.cell.zhi),
                'zlo': float(mol.cell.zlo),
            }

    json_dict['molecules'][0]['extensions'].append(radonpy_ext)

    return json_dict


def JSONToMol(file):
    with open(file, mode='r') as f:
        json_dict = json.load(f)
    mol = JSONToMol_dict(json_dict)
    return mol


def JSONToMol_str(json_str):
    json_dict = json.loads(json_str)
    mol = JSONToMol_dict(json_dict)
    return mol


def JSONToMol_dict(json_dict):
    radonpy_ext = None
    for ext in json_dict['molecules'][0]['extensions']:
        if 'name' in ext and ext['name'] == 'radonpy_extention':
            radonpy_ext = ext
    if radonpy_ext is None:
        radon_print('YadonPy extension data was not found in JSON file.', level=3)

    # Avoiding bug in RDKit
    for b in json_dict['molecules'][0]['bonds']:
        if 'stereo' in b and 'stereoAtoms' not in b:
            b['stereo'] = 'either'

    mol = Chem.rdMolInterchange.JSONToMols(json.dumps(json_dict))[0]
    Chem.SanitizeMol(mol)

    if not mol.HasProp('pair_style'):
        radon_print('pair_style is missing. Assuming lj for pair_style.', level=2)
        mol.SetProp('pair_style', 'lj')
    for i, a in enumerate(mol.GetAtoms()):
        atom_data = radonpy_ext['atoms'][i]

        # FF on atoms
        if 'ff_type' in atom_data:
            a.SetProp('ff_type', str(atom_data['ff_type']))
        if 'ff_epsilon' in atom_data:
            a.SetDoubleProp('ff_epsilon', float(atom_data['ff_epsilon']))
        if 'ff_sigma' in atom_data:
            a.SetDoubleProp('ff_sigma', float(atom_data['ff_sigma']))

        # charge
        if 'AtomicCharge' in atom_data:
            a.SetDoubleProp('AtomicCharge', float(atom_data['AtomicCharge']))
        if 'RESP' in atom_data:
            a.SetDoubleProp('RESP', float(atom_data['RESP']))
        if 'ESP' in atom_data:
            a.SetDoubleProp('ESP', float(atom_data['ESP']))
        if 'Mulliken' in atom_data:
            a.SetDoubleProp('Mulliken', float(atom_data['Mulliken']))
        if 'Lowdin' in atom_data:
            a.SetDoubleProp('Lowdin', float(atom_data['Lowdin']))
        if '_GasteigerCharge' in atom_data:
            a.SetProp('_GasteigerCharge', str(atom_data['_GasteigerCharge']))
    
        # velocity
        if 'vx' in atom_data:
            a.SetDoubleProp('vx', float(atom_data['vx']))
            a.SetDoubleProp('vy', float(atom_data['vy']))
            a.SetDoubleProp('vz', float(atom_data['vz']))

        # crosslinking
        if 'CL_remove' in atom_data:
            a.SetBoolProp('CL_remove', bool(atom_data['CL_remove']))
        if 'CL_react' in atom_data:
            a.SetBoolProp('CL_react', bool(atom_data['CL_react']))

        # others
        a.SetIsotope(int(atom_data['isotope']))
        if 'mol_id' in atom_data:
            a.SetIntProp('mol_id', int(atom_data['mol_id']))

        # PDB
        atom_name = str(atom_data['ff_type']) if 'ff_type' in atom_data else a.GetSymbol()
        if 'ResidueName' in atom_data and 'ResidueNumber' in atom_data:
            a.SetMonomerInfo(
                Chem.AtomPDBResidueInfo(
                    atom_name,
                    residueName=atom_data['ResidueName'],
                    residueNumber=atom_data['ResidueNumber'],
                    isHeteroAtom=False
                )
            )


    if not mol.HasProp('bond_style'):
        radon_print('bond_style is missing. Assuming harmonic for bond_style.', level=2)
        mol.SetProp('bond_style', 'harmonic')
    for i, b in enumerate(mol.GetBonds()):
        bond_data = radonpy_ext['bonds'][i]

        # FF on bonds
        if 'ff_type' in bond_data:
            b.SetProp('ff_type', str(bond_data['ff_type']))
        if 'ff_k' in bond_data:
            b.SetDoubleProp('ff_k', float(bond_data['ff_k']))
        if 'ff_r0' in bond_data:
            b.SetDoubleProp('ff_r0', float(bond_data['ff_r0']))

        # crosslinking
        if 'CL_new_bond' in bond_data:
            b.SetBoolProp('CL_new_bond', bool(bond_data['CL_new_bond']))

        # others
        if 'new_bond' in bond_data:
            b.SetBoolProp('new_bond', bool(bond_data['new_bond']))

    if not mol.HasProp('angle_style'):
        radon_print('angle_style is missing. Assuming harmonic for angle_style.', level=2)
        mol.SetProp('angle_style', 'harmonic')
    if 'angles' in radonpy_ext:
        if mol.GetProp('angle_style') == 'harmonic':
            angle_class = ff_class.Angle_harmonic
        else:
            radon_print('angle_style %s is not available.' % mol.GetProp('angle_style'), level=3)

        angle_prop = {}
        for ang in radonpy_ext['angles']:
            key = '%i,%i,%i' % (int(ang['a']), int(ang['b']), int(ang['c']))
            angle_prop[key] = Angle(
                a = int(ang['a']),
                b = int(ang['b']),
                c = int(ang['c']),
                ff = angle_class(**ang['ff'])
            )
        setattr(mol, 'angles', angle_prop)


    if not mol.HasProp('dihedral_style'):
        radon_print('dihedral_style is missing. Assuming fourier for dihedral_style.', level=2)
        mol.SetProp('dihedral_style', 'fourier')
    if 'dihedrals' in radonpy_ext:
        if mol.GetProp('dihedral_style') == 'fourier':
            dihedral_class = ff_class.Dihedral_fourier
        elif mol.GetProp('dihedral_style') == 'harmonic':
            dihedral_class = ff_class.Dihedral_harmonic
        else:
            radon_print('dihedral_style %s is not available.' % mol.GetProp('dihedral_style'), level=3)

        dihedral_prop = {}
        for dih in radonpy_ext['dihedrals']:
            key = '%i,%i,%i,%i' % (int(dih['a']), int(dih['b']), int(dih['c']), int(dih['d']))
            dihedral_prop[key] = Dihedral(
                a = int(dih['a']),
                b = int(dih['b']),
                c = int(dih['c']),
                d = int(dih['d']),
                ff = dihedral_class(**dih['ff'])
            )
        setattr(mol, 'dihedrals', dihedral_prop)


    if not mol.HasProp('improper_style'):
        radon_print('improper_style is missing. Assuming cvff for improper_style.', level=2)
        mol.SetProp('improper_style', 'cvff')
    if 'impropers' in radonpy_ext:
        if mol.GetProp('improper_style') == 'cvff':
            improper_class = ff_class.Improper_cvff
        elif mol.GetProp('improper_style') == 'umbrella':
            improper_class = ff_class.Improper_umbrella
        else:
            radon_print('improper_style %s is not available.' % mol.GetProp('improper_style'), level=3)

        improper_prop = {}
        for imp in radonpy_ext['impropers']:
            key = '%i,%i,%i,%i' % (int(imp['a']), int(imp['b']), int(imp['c']), int(imp['d']))
            improper_prop[key] = Improper(
                a = int(imp['a']),
                b = int(imp['b']),
                c = int(imp['c']),
                d = int(imp['d']),
                ff = improper_class(**imp['ff'])
            )
        setattr(mol, 'impropers', improper_prop)


    # cell
    if 'cell' in radonpy_ext and 'xhi' in radonpy_ext['cell']:
        cell_prop = Cell(**radonpy_ext['cell'])
        setattr(mol, 'cell', cell_prop)

    return mol
