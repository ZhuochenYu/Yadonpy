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
# ff.merz module
# ******************************************************************************

"""
MERZ ion force field (OPC/OPC3 ion Lennard-Jones parameters from Merz group).

This module is intentionally minimal:
- It only provides nonbonded parameters (mass, sigma, epsilon) for ions/water-site types.
- It also includes molecule definitions for monoatomic ions (charges from ions.itp).

Typical use case in YadonPy:
    ion_ff = MERZ()
    ion_pack = ion(ion='Na+', n_ion=1000, ff=ion_ff)
    ac = poly.amorphous_cell(polymer, 10, density=0.05)  # ions are injected from registry
"""

import json
from rdkit import Chem
from rdkit import Geometry as Geom
from ..core import naming, utils
from ..core.resources import ff_data_path
from .report import print_ff_assignment_report





class MERZ():
    """
    merz.MERZ() class

    Minimal forcefield object for Merz ion LJ parameters.

    - Pair style: LJ
    - No bonded terms (ions are monoatomic)
    - 1-4 scaling constants are taken from [defaults] if present (fudgeLJ/fudgeQQ)

    The database is stored in ff_dat/Merz.json.
    """

    def __init__(self, db_file=None):
        if db_file is None:
            db_file = str(ff_data_path("ff_dat", "Merz.json"))

        with open(db_file, 'r') as f:
            self._raw = json.loads(f.read())

        self.param = self.load_ff_json(db_file)

        self.name = 'merz'
        self.pair_style = 'lj'
        self.bond_style = 'harmonic'
        self.angle_style = 'harmonic'
        self.dihedral_style = 'fourier'
        self.improper_style = 'cvff'
        self.ff_class = '1'

        # Coulomb/LJ scaling (for compatibility with the rest of YadonPy)
        # Merz forcefield is typically used with gen-pairs=yes and fudgeQQ=0.8333, fudgeLJ=0.5
        dflt = self._raw.get('defaults', {})
        self.param.c_c12 = 0.0
        self.param.c_c13 = 0.0
        self.param.c_c14 = float(dflt.get('fudgeQQ', 1.0))
        self.param.lj_c12 = 0.0
        self.param.lj_c13 = 0.0
        self.param.lj_c14 = float(dflt.get('fudgeLJ', 1.0))

        # Ion alias map: accept 'Na+', 'Cl-', 'CA2', 'Ca2+' etc.
        self.ion_alias = self._raw.get('ion_alias', {})

        # Molecule definitions (from ions.itp)
        self.molecule_types = {m['name']: m for m in self._raw.get('molecule_types', [])}

        # ---------------------------------------------------------------------
        # Convenience builder: MERZ().mol('[Li+]')
        # ---------------------------------------------------------------------


    @staticmethod
    def _ion_key(symbol: str, q: int) -> str:
        q = int(q)
        if q == 0:
            return str(symbol)
        if abs(q) == 1:
            return f"{symbol}{'+' if q > 0 else '-'}"
        return f"{symbol}{abs(q)}{'+' if q > 0 else '-'}"

    def mol(self, ion_smiles_or_name: str, confId: int = 0, *, name: str | None = None, **kwargs):
        """Create a monoatomic ion Mol from either SMILES ("[Li+]") or name ("Li+").

        This mirrors the RadonPy user experience:
            ion = MERZ().mol("[Li+]")

        Compatibility notes:
            - MERZ predates the MolDB-style `ff.mol(...)` interface used by the
              newer force fields in YadonPy.
            - We therefore accept modern keyword arguments such as `name`,
              `prefer_db`, `require_ready`, or `charge` and ignore the ones that
              do not apply to built-in monoatomic ions.

        Notes:
            - Only monoatomic ions are supported in MERZ.
            - Multi-atom ions (e.g. quaternary ammonium, TFSI-) should be treated
              as normal molecules and parameterized by GAFF2_mod.
        """
        if name is None:
            alias_name = kwargs.get('mol_name', None)
            if alias_name is not None and str(alias_name).strip():
                name = str(alias_name).strip()

        s = str(ion_smiles_or_name).strip()
        # Try SMILES first
        try:
            m = Chem.MolFromSmiles(s)
        except Exception:
            m = None
        if m is not None and int(m.GetNumAtoms()) == 1:
            a = m.GetAtomWithIdx(0)
            sym = a.GetSymbol()
            q = int(a.GetFormalCharge())
            if q == 0:
                raise ValueError(f"MERZ.mol expects a charged monoatomic ion, got neutral: {s}")
            ion = self._ion_key(sym, q)
            # IMPORTANT: mol() only constructs geometry + charges.
            # Force-field parameters are assigned by an explicit ff_assign(mol)
            # call, mirroring RadonPy's TIP3P().mol() + ff_assign() usage.
            return self.create_ion_mol(ion=ion, confId=confId, name=name)
        # Fall back: treat as ion name / alias
        return self.create_ion_mol(ion=s, confId=confId, name=name)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def ff_assign(self, mol, charge=None, report: bool = True, **kwargs):
        """
        Assign force field parameters for an ion molecule.

        Notes:
            - For Merz ions, we only set particle type + LJ + charge.
            - 'charge' argument is ignored; charges are fixed by ion definitions.
        """
        try:
            current_name = naming.get_name(mol, default=None)
        except Exception:
            current_name = None
        try:
            naming.ensure_name(mol, name=current_name, depth=2, prefer_var=(current_name is None))
        except Exception:
            pass
        mol.SetProp('ff_name', str(self.name))
        mol.SetProp('ff_class', str(self.ff_class))
        result = self.assign_ptypes(mol)
        if result and report:
            print_ff_assignment_report(mol, ff_obj=self)
        return mol if result else False

    def create_ion_mol(self, ion='Na+', confId=0, *, name: str | None = None):
        """Create a monoatomic ion RDKit Mol with Merz charges + 3D coordinates.

        Args:
            ion: ion identifier string.
                 Recommended forms: 'Na+', 'Cl-', 'Ca2+', 'Mg2+', ...
                 Also accepts: 'NA', 'CL', ... (molecule type names from ions.itp)
            name: optional user-facing molecule name. When provided, YadonPy's
                  standard naming properties are populated for downstream export
                  and cache naming, while the original Merz molecule type is
                  preserved in `merz_molecule_type`.

        Returns:
            RDKit Mol (with one atom and one conformer at origin).

        Notes:
            This function does *not* assign Lennard-Jones/particle types by default.
            Call ``MERZ().ff_assign(mol)`` explicitly (RadonPy-style).
        """
        if ion is None:
            raise ValueError('ion must be specified')

        key = str(ion).strip()
        # Accept bracketed SMILES like '[Li+]' as input.
        # NOTE: Unbracketed forms like 'Li+' are not valid SMILES and will
        # trigger RDKit parse warnings; we treat those as ion labels instead.
        m = None
        if key.startswith('[') and key.endswith(']'):
            try:
                m = Chem.MolFromSmiles(key)
            except Exception:
                m = None
        if m is not None and int(m.GetNumAtoms()) == 1 and int(m.GetAtomWithIdx(0).GetFormalCharge()) != 0:
            a = m.GetAtomWithIdx(0)
            key = self._ion_key(a.GetSymbol(), int(a.GetFormalCharge()))

        # Map alias -> molecule name
        molname = self.ion_alias.get(key, None)
        if molname is None:
            # Fall back: strip trailing sign (Na+, Cl-) and try again
            if key.endswith('+') or key.endswith('-'):
                molname = self.ion_alias.get(key[:-1], None)
        if molname is None:
            # Also try upper-case molname
            molname = self.ion_alias.get(key.upper(), None)

        if molname is None:
            raise ValueError('Unknown ion "%s". Available examples: Na+, Cl-, Ca2+.' % key)

        mdef = self.molecule_types.get(molname)
        if mdef is None or len(mdef.get('atoms', [])) != 1:
            raise ValueError('Ion definition for "%s" is missing or not monoatomic.' % molname)

        at = mdef['atoms'][0]
        at_type = at['type']
        q = float(at['charge'])

        # Build RDKit molecule: one atom
        rw = Chem.RWMol()
        atom = Chem.Atom(str(at_type))  # RDKit understands "Na", "Cl", etc.
        # Keep a consistent formal charge (helps downstream net-charge checks).
        try:
            atom.SetFormalCharge(int(round(q)))
        except Exception:
            pass
        idx = rw.AddAtom(atom)
        mol = rw.GetMol()

        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetAtomPosition(idx, Geom.Point3D(0.0, 0.0, 0.0))
        mol.AddConformer(conf, assignId=True)

        # Assign Merz atomic charge (the FF assignment is done explicitly via ff_assign)
        mol.GetAtomWithIdx(0).SetDoubleProp('AtomicCharge', q)

        # Helpful metadata
        resolved_name = str(name).strip() if name is not None else ''
        mol.SetProp('mol_name', resolved_name or molname)
        mol.SetProp('merz_molecule_type', molname)
        mol.GetAtomWithIdx(0).SetProp('ion_name', molname)

        # RDKit descriptor calls such as MolWt() expect the implicit-valence
        # cache to be initialized even for single-atom charged species.
        mol.UpdatePropertyCache(strict=False)

        if resolved_name:
            try:
                naming.ensure_name(mol, name=resolved_name, depth=2)
            except Exception:
                mol.SetProp('_Name', resolved_name)
                mol.SetProp('name', resolved_name)

        return mol

    # -------------------------------------------------------------------------
    # Assignment
    # -------------------------------------------------------------------------

    def assign_ptypes(self, mol):
        """
        Assign particle types for each atom in mol.

        For Merz ions, the atom type name is usually identical to the element symbol (Na, Cl, Ca2, ...).
        This method:
          - sets ff_type = atom symbol
          - sets ff_sigma/ff_epsilon from database
          - sets ff_mass from database
        """
        for a in mol.GetAtoms():
            pt = a.GetSymbol()

            # Some divalent types in the database are named "Ca2", "Mg2", etc.
            # RDKit atom symbol for calcium is "Ca" (no trailing '2').
            # If symbol not found, try symbol+'2'.
            if pt not in self.param.pt and (pt + '2') in self.param.pt:
                pt = pt + '2'

            if pt not in self.param.pt:
                utils.radon_print('MERZ: particle type %s is not found in database.' % pt, level=3)
                return False

            a.SetProp('ff_type', str(pt))
            a.SetDoubleProp('ff_sigma', float(self.param.pt[pt].sigma))
            a.SetDoubleProp('ff_epsilon', float(self.param.pt[pt].epsilon))
            a.SetDoubleProp('ff_mass', float(self.param.pt[pt].mass))

            # If the ion was built from create_ion_mol, AtomicCharge is already set.
            # Otherwise, keep existing AtomicCharge/formal charge as-is.

        return True

    # -------------------------------------------------------------------------
    # JSON loader (same logic as GAFF)
    # -------------------------------------------------------------------------

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

        for pt in j.get('particle_types', []):
            pt_obj = self.Container()
            for key in pt.keys():
                setattr(pt_obj, key, pt[key])
            ff.pt[pt['name']] = pt_obj

        # bonded terms are intentionally empty for Merz ions
        return ff

    class Container(object):
        pass
