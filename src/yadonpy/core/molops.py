"""RDKit molecule editing utilities used by builders and converters."""

from __future__ import annotations
from itertools import permutations

from rdkit import Chem

from . import const
from .topology import Angle, Dihedral, Improper

def set_mol_id(mol, pdb=True):
    """
    utils.set_mol_id

    Set molecular ID

    Args:
        mol: RDkit Mol object

    Optional args:
        pdb: Update the ChainId of PDB (boolean)

    Returns:
        Rdkit Mol object
    """

    molid = 1

    # Clear mol_id
    for atom in mol.GetAtoms():
        atom.SetIntProp('mol_id', 0)

    def recursive_set_mol_id(atom, molid):
        for na in atom.GetNeighbors():
            if na.GetIntProp('mol_id') == 0:
                na.SetIntProp('mol_id', molid)
                if pdb and na.GetPDBResidueInfo() is not None:
                    if molid <= len(const.pdb_id):
                        na.GetPDBResidueInfo().SetChainId(const.pdb_id[molid-1])
                recursive_set_mol_id(na, molid)

    for atom in mol.GetAtoms():
        if atom.GetIntProp('mol_id') == 0:
            atom.SetIntProp('mol_id', molid)
            if pdb and atom.GetPDBResidueInfo() is not None:
                if molid <= len(const.pdb_id):
                    atom.GetPDBResidueInfo().SetChainId(const.pdb_id[molid-1])
            recursive_set_mol_id(atom, molid)
            molid += 1

    return mol


def count_mols(mol):
    """
    utils.count_mols

    Count number of molecules

    Args:
        mol: RDkit Mol object

    Returns:
        Number of molecules (int)
    """

    fragments = Chem.GetMolFrags(mol, asMols=True)
    return len(fragments)


def remove_atom(mol, idx, angle_fix=False):
    """
    utils.remove_atom

    Remove a specific atom from RDkit Mol object

    Args:
        mol: RDkit Mol object
        idx: Atom index of removing atom in RDkit Mol object

    Options:
        angle_fix: Fix information of bond angles, dihedral angles, and improper angles. (boolean)

    Returns:
        RDkit Mol object
    """

    # NOTE (v0.5.1):
    # Some RDKit builds can throw `RuntimeError: bad any_cast` inside
    # `RWMol.RemoveAtom()` when the molecule carries certain computed/private
    # properties. This became visible in polymerization workflows where
    # monomers have gone through QM/charge assignment and then are repeatedly
    # combined/edited.
    #
    # We therefore:
    #   1) snapshot a *minimal safe* set of yadonpy-relevant atom properties
    #      (charges, ff_type, linker flags, PDB residue info),
    #   2) clear all atom properties before editing,
    #   3) perform the atom removal,
    #   4) restore the saved properties onto the surviving atoms.
    #
    # This also ensures partial charges are preserved across RemoveAtom.

    # Properties we must preserve for downstream export/analysis.
    _SAFE_FLOAT_PROPS = [
        'AtomicCharge', 'AtomicCharge_raw',
        'RESP', 'RESP_raw',
        'ESP', 'MullikenCharge', 'LowdinCharge',
        '_GasteigerCharge',
        'ff_sigma', 'ff_epsilon',
    ]
    _SAFE_STR_PROPS = [
        'ff_type', 'ff_btype', 'ff_ptype',
    ]
    _SAFE_INT_PROPS = [
        'mol_id',
    ]
    _SAFE_BOOL_PROPS = [
        'head', 'tail', 'head_neighbor', 'tail_neighbor',
    ]

    def _snapshot_atom(atom: Chem.Atom) -> dict:
        d: dict = {'float': {}, 'str': {}, 'int': {}, 'bool': {}, 'pdb': None}
        # floats
        for k in _SAFE_FLOAT_PROPS:
            if not atom.HasProp(k):
                continue
            try:
                d['float'][k] = float(atom.GetDoubleProp(k))
            except Exception:
                # stored as string
                try:
                    d['float'][k] = float(atom.GetProp(k))
                except Exception:
                    pass
        # strings
        for k in _SAFE_STR_PROPS:
            if atom.HasProp(k):
                try:
                    d['str'][k] = str(atom.GetProp(k))
                except Exception:
                    pass
        # ints
        for k in _SAFE_INT_PROPS:
            if atom.HasProp(k):
                try:
                    d['int'][k] = int(atom.GetIntProp(k))
                except Exception:
                    # stored as string
                    try:
                        d['int'][k] = int(atom.GetProp(k))
                    except Exception:
                        pass
        # bools
        for k in _SAFE_BOOL_PROPS:
            if atom.HasProp(k):
                try:
                    d['bool'][k] = bool(atom.GetBoolProp(k))
                except Exception:
                    # stored as string
                    try:
                        v = str(atom.GetProp(k)).strip().lower()
                        d['bool'][k] = v in ('1', 'true', 't', 'yes', 'y')
                    except Exception:
                        pass

        # PDB residue info
        ri = atom.GetPDBResidueInfo()
        if ri is not None:
            try:
                d['pdb'] = {
                    'name': ri.GetName(),
                    'resName': ri.GetResidueName(),
                    'resNum': int(ri.GetResidueNumber()),
                    'chain': ri.GetChainId(),
                    'insCode': ri.GetInsertionCode(),
                    'isHet': bool(ri.GetIsHeteroAtom()),
                    'occ': float(ri.GetOccupancy()),
                    'temp': float(ri.GetTempFactor()),
                    'serial': int(ri.GetSerialNumber()),
                    'altLoc': ri.GetAltLoc(),
                }
            except Exception:
                d['pdb'] = None
        return d

    # Snapshot properties (excluding the removed atom)
    _saved: dict[int, dict] = {}
    try:
        for a in mol.GetAtoms():
            i = a.GetIdx()
            if i == idx:
                continue
            _saved[i] = _snapshot_atom(a)
    except Exception:
        _saved = {}

    # Copy the extended topology attributes when requested.
    # NOTE: These attributes are not RDKit-owned; they live as Python-side
    # containers on the Mol object and are used when writing .itp files.
    angles_copy: dict = {}
    dihedrals_copy: dict = {}
    impropers_copy: dict = {}
    cell_copy = mol.cell if hasattr(mol, 'cell') else None

    if angle_fix:
        if hasattr(mol, 'impropers'):
            for imp in mol.impropers.values():
                imp_idx = {imp.a, imp.b, imp.c, imp.d}
                if idx in imp_idx:
                    continue

                if max(imp_idx) < idx:
                    key = '%i,%i,%i,%i' % (imp.a, imp.b, imp.c, imp.d)
                    impropers_copy[key] = imp
                else:
                    idx_a = imp.a if imp.a < idx else imp.a-1
                    idx_b = imp.b if imp.b < idx else imp.b-1
                    idx_c = imp.c if imp.c < idx else imp.c-1
                    idx_d = imp.d if imp.d < idx else imp.d-1
                    key = '%i,%i,%i,%i' % (idx_a, idx_b, idx_c, idx_d)
                    impropers_copy[key] = Improper(
                                                a=idx_a,
                                                b=idx_b,
                                                c=idx_c,
                                                d=idx_d,
                                                ff=imp.ff
                                            )

        if hasattr(mol, 'dihedrals'):
            for dih in mol.dihedrals.values():
                dih_idx = {dih.a, dih.b, dih.c, dih.d}
                if idx in dih_idx:
                    continue

                if max(dih_idx) < idx:
                    key = '%i,%i,%i,%i' % (dih.a, dih.b, dih.c, dih.d)
                    dihedrals_copy[key] = dih
                else:
                    idx_a = dih.a if dih.a < idx else dih.a-1
                    idx_b = dih.b if dih.b < idx else dih.b-1
                    idx_c = dih.c if dih.c < idx else dih.c-1
                    idx_d = dih.d if dih.d < idx else dih.d-1
                    key = '%i,%i,%i,%i' % (idx_a, idx_b, idx_c, idx_d)
                    dihedrals_copy[key] = Dihedral(
                                                a=idx_a,
                                                b=idx_b,
                                                c=idx_c,
                                                d=idx_d,
                                                ff=dih.ff
                                            )

        if hasattr(mol, 'angles'):
            for angle in mol.angles.values():
                ang_idx = {angle.a, angle.b, angle.c}
                if idx in ang_idx:
                    continue

                if max(ang_idx) < idx:
                    key = '%i,%i,%i' % (angle.a, angle.b, angle.c)
                    angles_copy[key] = angle
                else:
                    idx_a = angle.a if angle.a < idx else angle.a-1
                    idx_b = angle.b if angle.b < idx else angle.b-1
                    idx_c = angle.c if angle.c < idx else angle.c-1
                    key = '%i,%i,%i' % (idx_a, idx_b, idx_c)
                    angles_copy[key] = Angle(
                                            a=idx_a,
                                            b=idx_b,
                                            c=idx_c,
                                            ff=angle.ff
                                        )

    # Edit on a RWMol with all atom properties cleared to avoid RDKit any_cast
    # issues during atom deletion.
    rwmol = Chem.RWMol(mol)
    try:
        for a in rwmol.GetAtoms():
            # Clear all properties (string/int/double/bool/computed/private).
            for k in list(a.GetPropNames(includePrivate=True, includeComputed=True)):
                try:
                    a.ClearProp(k)
                except Exception:
                    pass
    except Exception:
        pass

    # Remove all bonds from the atom first (defensive for some RDKit builds).
    for pb in list(rwmol.GetAtomWithIdx(idx).GetNeighbors()):
        try:
            rwmol.RemoveBond(idx, pb.GetIdx())
        except Exception:
            pass

    rwmol.RemoveAtom(idx)

    mol = rwmol.GetMol()

    # Restore saved properties onto surviving atoms.
    try:
        for old_i, d in _saved.items():
            new_i = old_i if old_i < idx else old_i - 1
            if new_i < 0 or new_i >= mol.GetNumAtoms():
                continue
            a = mol.GetAtomWithIdx(new_i)
            for k, v in d.get('str', {}).items():
                try:
                    a.SetProp(str(k), str(v))
                except Exception:
                    pass
            for k, v in d.get('int', {}).items():
                try:
                    a.SetIntProp(str(k), int(v))
                except Exception:
                    pass
            for k, v in d.get('bool', {}).items():
                try:
                    a.SetBoolProp(str(k), bool(v))
                except Exception:
                    pass
            for k, v in d.get('float', {}).items():
                try:
                    a.SetDoubleProp(str(k), float(v))
                except Exception:
                    # last resort: store as string
                    try:
                        a.SetProp(str(k), str(v))
                    except Exception:
                        pass

            pdb = d.get('pdb', None)
            if pdb:
                try:
                    ri = Chem.AtomPDBResidueInfo(
                        pdb.get('name', a.GetSymbol()),
                        residueName=pdb.get('resName', 'RU0'),
                        residueNumber=int(pdb.get('resNum', 1)),
                        isHeteroAtom=bool(pdb.get('isHet', False)),
                    )
                    # optional fields
                    try: ri.SetChainId(pdb.get('chain', ' '))
                    except Exception: pass
                    try: ri.SetInsertionCode(pdb.get('insCode', ' '))
                    except Exception: pass
                    try: ri.SetOccupancy(float(pdb.get('occ', 1.0)))
                    except Exception: pass
                    try: ri.SetTempFactor(float(pdb.get('temp', 0.0)))
                    except Exception: pass
                    try: ri.SetSerialNumber(int(pdb.get('serial', 0)))
                    except Exception: pass
                    try: ri.SetAltLoc(pdb.get('altLoc', ' '))
                    except Exception: pass
                    a.SetMonomerInfo(ri)
                except Exception:
                    pass
    except Exception:
        pass
    setattr(mol, 'angles', angles_copy)
    setattr(mol, 'dihedrals', dihedrals_copy)
    setattr(mol, 'impropers', impropers_copy)
    if cell_copy is not None: setattr(mol, 'cell', cell_copy)

    return mol


def add_bond(mol, idx1, idx2, order=Chem.rdchem.BondType.SINGLE, preserve_topology=False):
    """
    utils.add_bond

    Add a new bond in RDkit Mol object

    Args:
        mol: RDkit Mol object
        idx1, idx2: Atom index adding a new bond (int)
        order: bond order (RDkit BondType object, ex. Chem.rdchem.BondType.SINGLE)

    Returns:
        RDkit Mol object
    """

    # Copy the extended attributes when the caller wants to preserve inherited
    # bonded topology across a connection edit (used by polymer junction builds).
    angles_copy = mol.angles.copy() if preserve_topology and hasattr(mol, 'angles') else {}
    dihedrals_copy = mol.dihedrals.copy() if preserve_topology and hasattr(mol, 'dihedrals') else {}
    impropers_copy = mol.impropers.copy() if preserve_topology and hasattr(mol, 'impropers') else {}
    cell_copy = mol.cell if hasattr(mol, 'cell') else None

    rwmol = Chem.RWMol(mol)
    rwmol.AddBond(idx1, idx2, order=order)
    mol = rwmol.GetMol()

    setattr(mol, 'angles', angles_copy)
    setattr(mol, 'dihedrals', dihedrals_copy)
    setattr(mol, 'impropers', impropers_copy)
    if cell_copy is not None: setattr(mol, 'cell', cell_copy)

    return mol


def remove_bond(mol, idx1, idx2):
    """
    utils.remove_bond

    Remove a specific bond in RDkit Mol object

    Args:
        mol: RDkit Mol object
        idx1, idx2: Atom index removing a specific bond (int)

    Returns:
        RDkit Mol object
    """

    # Copy the extended attributes
    #angles_copy = mol.angles.copy() if hasattr(mol, 'angles') else {}
    #dihedrals_copy = mol.dihedrals.copy() if hasattr(mol, 'dihedrals') else {}
    #impropers_copy = mol.impropers.copy() if hasattr(mol, 'impropers') else {}
    cell_copy = mol.cell if hasattr(mol, 'cell') else None

    rwmol = Chem.RWMol(mol)
    rwmol.RemoveBond(idx1, idx2)
    mol = rwmol.GetMol()

    setattr(mol, 'angles', {})
    setattr(mol, 'dihedrals', {})
    setattr(mol, 'impropers', {})
    if cell_copy is not None: setattr(mol, 'cell', cell_copy)

    return mol


def add_angle(mol, a, b, c, ff=None):
    """
    utils.add_angle

    Add a new angle in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c: Atom index adding a new angle (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'angles'):
        setattr(mol, 'angles', {})

    key = '%i,%i,%i' % (a, b, c)
    mol.angles[key] = Angle(a=a, b=b, c=c, ff=ff)

    return True


def remove_angle(mol, a, b, c):
    """
    utils.remove_angle

    Remove a specific angle in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c: Atom index removing a specific angle (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'angles'):
        return False

    # for i, angle in enumerate(mol.angles:
    #     if ((angle.a == a and angle.b == b and angle.c == c) or
    #         (angle.c == a and angle.b == b and angle.a == c)):
    #         del mol.angles[i]
    #         break

    key1 = '%i,%i,%i' % (a, b, c)
    key2 = '%i,%i,%i' % (c, b, a)
    if key1 in mol.angles:
        del mol.angles[key1]
    elif key2 in mol.angles:
        del mol.angles[key2]

    return True


def add_dihedral(mol, a, b, c, d, ff=None):
    """
    utils.add_dihedral

    Add a new dihedral in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c, d: Atom index adding a new dihedral (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'dihedrals'):
        setattr(mol, 'dihedrals', {})

    key = '%i,%i,%i,%i' % (a, b, c, d)
    mol.dihedrals[key] = Dihedral(a=a, b=b, c=c, d=d, ff=ff)

    return True


def remove_dihedral(mol, a, b, c, d):
    """
    utils.remove_dihedral

    Remove a specific dihedral in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c: Atom index removing a specific dihedral (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'dihedrals'):
        return False

    # for i, dihedral in enumerate(mol.dihedrals):
    #     if ((dihedral.a == a and dihedral.b == b and dihedral.c == c and dihedral.d == d) or
    #         (dihedral.d == a and dihedral.c == b and dihedral.b == c and dihedral.a == d)):
    #         del mol.dihedrals[i]
    #         break

    key1 = '%i,%i,%i,%i' % (a, b, c, d)
    key2 = '%i,%i,%i,%i' % (d, c, b, a)
    if key1 in mol.dihedrals:
        del mol.dihedrals[key1]
    elif key2 in mol.dihedrals:
        del mol.dihedrals[key2]

    return True


def add_improper(mol, a, b, c, d, ff=None):
    """
    utils.add_improper

    Add a new imploper in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c, d: Atom index adding a new imploper (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'impropers'):
        setattr(mol, 'impropers', {})

    key = '%i,%i,%i,%i' % (a, b, c, d)
    mol.impropers[key] = Improper(a=a, b=b, c=c, d=d, ff=ff)

    return True


def remove_improper(mol, a, b, c, d):
    """
    utils.remove_improper

    Remove a specific improper in RDkit Mol object

    Args:
        mol: RDkit Mol object
        a, b, c: Atom index removing a specific improper (int)

    Returns:
        boolean
    """

    if not hasattr(mol, 'impropers'):
        return False

    # match = False
    # for i, improper in enumerate(mol.impropers):
    #     if improper.a == a:
    #         for perm in permutations([b, c, d], 3):
    #             if improper.b == perm[0] and improper.c == perm[1] and improper.d == perm[2]:
    #                 del mol.impropers[i]
    #                 match = True
    #                 break
    #         if match: break

    for perm in permutations([b, c, d], 3):
        key = '%i,%i,%i,%i' % (a, perm[0], perm[1], perm[2])
        if key in mol.impropers:
            del mol.impropers[key]
            break

    return True
