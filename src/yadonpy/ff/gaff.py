"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""
from __future__ import annotations

#  Copyright (c) 2026. YadonPy developers. All rights reserved.
#  Use of this source code is governed by a BSD-3-style
#  license that can be found in the LICENSE file.

# ******************************************************************************
# ff.gaff module
# ******************************************************************************

import numpy as np
import os
import json
from itertools import permutations
from rdkit import Chem
from ..core import calc, utils
from ..core.polyelectrolyte import annotate_polyelectrolyte_metadata
from ..core.resources import ff_data_path
from . import ff_class
from .report import print_ff_assignment_report




class GAFF():
    """
    gaff.GAFF() class

    Forcefield object with typing rules for Gaff model.
    By default reads data file in forcefields subdirectory.

    Attributes:
        ff_name: gaff
        pair_style: lj
        bond_style: harmonic
        angle_style: harmonic
        dihedral_style: fourier
        improper_style: cvff
        ff_class: 1
    """
    def __init__(self, db_file=None):
        if db_file is None:
            db_file = str(ff_data_path("ff_dat", "gaff.json"))
        self.param = self.load_ff_json(db_file)
        self.name = 'gaff'
        self.pair_style = 'lj'
        self.bond_style = 'harmonic'
        self.angle_style = 'harmonic'
        self.dihedral_style = 'fourier'
        self.improper_style = 'cvff'
        self.ff_class = '1'
        self.param.c_c12 = 0.0
        self.param.c_c13 = 0.0
        self.param.c_c14 = 5/6
        self.param.lj_c12 = 0.0
        self.param.lj_c13 = 0.0
        self.param.lj_c14 = 0.5
        self.max_ring_size = 14
        self.elctrwd_elements = {'N', 'O', 'F', 'Cl', 'Br', 'I'}
        self.conj_chain = {'ce', 'cg', 'ne', 'pe'}
        self.conj_ring = {'cc', 'nc', 'pc'}
        self.conj_rep = {'ce': 'cf', 'cg': 'ch', 'ne': 'nf', 'pe': 'pf',
                            'cc': 'cd', 'nc': 'nd', 'pc': 'pd'}
        self.alt_ptype = {
            'cc': 'c2', 'cd': 'c2', 'ce': 'c2', 'cf': 'c2', 'cg': 'c1', 'ch': 'c1',
            'cp': 'ca', 'cq': 'ca', 'cu': 'c2', 'cv': 'c2', 'cx': 'c3', 'cy': 'c3',
            'h1': 'hc', 'h2': 'hc', 'h3': 'hc', 'h4': 'ha', 'h5': 'ha',
            'nb': 'nc', 'nc': 'n2', 'nd': 'n2', 'ne': 'n2', 'nf': 'n2',
            'pb': 'pc', 'pc': 'p2', 'pd': 'p2', 'pe': 'p2', 'pf': 'p2',
            'p4': 'px', 'px': 'p4', 'p5': 'py', 'py': 'p5',
            's4': 'sx', 'sx': 's4', 's6': 'sy', 'sy': 's6'
        }

    def _is_sulfonimide_like_nitrogen(self, atom) -> bool:
        """Return True for N(-SO2-)2 style centers such as FSI-/TFSI-."""
        try:
            if atom.GetSymbol() != 'N' or int(atom.GetTotalDegree()) != 2:
                return False
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 2:
                return False
            for nb in neighbors:
                if nb.GetSymbol() != 'S':
                    return False
                sulfonyl_oxo = 0
                for bond in nb.GetBonds():
                    other = bond.GetBeginAtom() if nb.GetIdx() == bond.GetEndAtom().GetIdx() else bond.GetEndAtom()
                    if other.GetIdx() == atom.GetIdx():
                        continue
                    if other.GetSymbol() == 'O' and bond.GetBondTypeAsDouble() == 2:
                        sulfonyl_oxo += 1
                if sulfonyl_oxo < 2:
                    return False
            return True
        except Exception:
            return False


    def ff_assign(
        self,
        mol,
        charge=None,
        retryMDL=True,
        useMDL=True,
        *,
        bonded: str | None = None,
        bonded_work_dir: str | os.PathLike | None = None,
        bonded_omp_psi4: int = 16,
        bonded_memory_mb: int = 16000,
        total_charge=None,
        total_multiplicity: int = 1,
        report: bool = True,
        **charge_kwargs,
    ):
        """
        GAFF.ff_assign

        GAFF force field assignment for RDkit Mol object

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

        # Allow passing a lightweight MolSpec handle (v0.6.3+)
        _spec_mode = False
        try:
            from ..core.molspec import MolSpec
            from ..core import naming
        except Exception:
            MolSpec = None
            naming = None

        if MolSpec is not None and isinstance(mol, MolSpec):
            _spec_mode = True
            spec = mol
            if not spec.name and naming is not None:
                # Start the stack scan from the *caller* of ff_assign so we do
                # not accidentally capture local helper aliases such as `spec`.
                spec.name = naming.infer_var_name(mol, depth=3) or naming.infer_var_name(mol, depth=2) or None
            mol = self.mol_rdkit(
                spec.smiles,
                name=spec.name,
                prefer_db=spec.prefer_db,
                require_ready=spec.require_ready,
                charge=spec.charge,
                basis_set=spec.basis_set,
                method=spec.method,
                resp_profile=spec.resp_profile,
                polyelectrolyte_mode=(
                    charge_kwargs.get("polyelectrolyte_mode")
                    if "polyelectrolyte_mode" in charge_kwargs
                    else spec.polyelectrolyte_mode
                ),
                polyelectrolyte_detection=(
                    charge_kwargs.get("polyelectrolyte_detection")
                    if "polyelectrolyte_detection" in charge_kwargs
                    else spec.polyelectrolyte_detection
                ),
            )
            # For downstream exports
            try:
                spec.cache_resolved_mol(mol)
            except Exception:
                pass
            if naming is not None:
                try:
                    naming.ensure_name(mol, name=spec.name, depth=2, prefer_var=False)
                except Exception:
                    pass

        if naming is not None:
            try:
                current_name = naming.get_name(mol, default=None)
            except Exception:
                current_name = None
            try:
                naming.ensure_name(mol, name=current_name, depth=2, prefer_var=(current_name is None))
            except Exception:
                pass
            try:
                if naming.try_restore_assigned_mol(mol, depth=2, ff_name=self.name):
                    if report:
                        print_ff_assignment_report(mol, ff_obj=self)
                    return mol
            except Exception:
                pass

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
        if result and charge is not None: result = calc.assign_charges(mol, charge=charge, **charge_kwargs)
        
        if not result and retryMDL and not useMDL:
            utils.radon_print('Retry to assign with MDL aromaticity model', level=1)
            Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True)
            Chem.rdmolops.SetAromaticity(mol, model=Chem.rdmolops.AromaticityModel.AROMATICITY_MDL)

            result = self.assign_ptypes(mol)
            if result: result = self.assign_btypes(mol)
            if result: result = self.assign_atypes(mol)
            if result: result = self.assign_dtypes(mol)
            if result: result = self.assign_itypes(mol)
            if result and charge is not None: result = calc.assign_charges(mol, charge=charge, **charge_kwargs)
            if result: utils.radon_print('Success to assign with MDL aromaticity model', level=1)

        # ------------------------------------------------------------------
        # Optional bonded-parameter override (DRIH / mseminario)
        #
        # Default: follow the force-field's bonded parameters (no patching).
        # If bonded is explicitly provided, we mark the molecule and (best-effort)
        # generate the bonded patch fragments for downstream topology writers.
        # ------------------------------------------------------------------
        if result and bonded is not None:
            try:
                from pathlib import Path
                from ..core import naming
                from ..sim import qm

                b = str(bonded).strip().lower()
                # aliases
                if b in ("ms", "mseminario", "seminario"):
                    b = "mseminario"
                if b in ("drih", "drih-like", "dri"):
                    b = "drih"

                # Mark explicit override so exporters / caches know this was requested.
                try:
                    mol.SetProp("_yadonpy_bonded_override", "1")
                    mol.SetProp("_yadonpy_bonded_explicit", "1")
                    mol.SetProp("_yadonpy_bonded_requested", str(b))
                    mol.SetProp("_yadonpy_bonded_signature", str(b))
                except Exception:
                    pass

                # Resolve a stable work dir
                mol_name = None
                try:
                    mol_name = naming.get_name(mol)
                except Exception:
                    pass
                if not mol_name:
                    mol_name = "mol"

                if bonded_work_dir is None:
                    bonded_work_dir = (Path.cwd() / ".yadonpy_cache" / "bonded" / str(mol_name)).resolve()
                bdir = Path(bonded_work_dir).expanduser().resolve()

                # If patch already exists, just record the method and continue.
                try:
                    if b == "drih" and mol.HasProp("_yadonpy_bonded_itp"):
                        mol.SetProp("_yadonpy_bonded_method", "DRIH")
                        b = "done"
                    if b == "mseminario" and mol.HasProp("_yadonpy_mseminario_itp"):
                        mol.SetProp("_yadonpy_bonded_method", "mseminario")
                        b = "done"
                except Exception:
                    pass

                # Ensure 3D coords for geometry-driven / Hessian-driven methods
                if b in ("drih", "mseminario"):
                    try:
                        smiles_hint = None
                        if mol.HasProp("_YADONPY_CANONICAL"):
                            smiles_hint = mol.GetProp("_YADONPY_CANONICAL")
                        utils.ensure_3d_coords(mol, smiles_hint=smiles_hint, engine="openbabel")
                    except Exception:
                        pass

                if b == "drih":
                    # Geometry-driven stiffening for AX4/AX6 polyhedral ions
                    qm.bond_angle_params_drih(
                        mol,
                        work_dir=str(bdir),
                        log_name=str(mol_name),
                        smiles_hint=(mol.GetProp("_YADONPY_CANONICAL") if mol.HasProp("_YADONPY_CANONICAL") else None),
                    )
                    try:
                        mol.SetProp("_yadonpy_bonded_method", "DRIH")
                    except Exception:
                        pass
                elif b == "mseminario":
                    # Hessian-derived bond/angle harmonics
                    # Use formal charge inference unless user explicitly provides total_charge.
                    if total_charge is None:
                        try:
                            total_charge = int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms()))
                        except Exception:
                            total_charge = 0

                    qm.bond_angle_params_mseminario(
                        mol,
                        confId=0,
                        opt=False,
                        work_dir=str(bdir),
                        log_name=str(mol_name),
                        qm_solver="psi4",
                        opt_method="wb97m-d3bj",
                        opt_basis="6-31G(d,p)",
                        hess_method="wb97m-d3bj",
                        hess_basis=None,
                        total_charge=int(total_charge),
                        total_multiplicity=int(total_multiplicity),
                        psi4_omp=int(bonded_omp_psi4),
                        memory=int(bonded_memory_mb),
                    )
                    try:
                        mol.SetProp("_yadonpy_bonded_method", "mseminario")
                    except Exception:
                        pass
                elif b != "done":
                    raise ValueError(f"Unknown bonded override: {bonded!r} (supported: 'DRIH', 'mseminario')")

                # Final verification: explicit bonded overrides must leave behind
                # a real fragment that downstream exporters can consume.
                try:
                    from pathlib import Path as _Path
                    _frag = None
                    if str(bonded).strip().lower() in ("ms", "mseminario", "seminario"):
                        if mol.HasProp("_yadonpy_mseminario_itp"):
                            _frag = _Path(str(mol.GetProp("_yadonpy_mseminario_itp")).strip())
                        if _frag is not None and _frag.is_file():
                            mol.SetProp("_yadonpy_bonded_itp", str(_frag.resolve()))
                            mol.SetProp("_yadonpy_bonded_method", "mseminario")
                            mol.SetProp("_yadonpy_bonded_signature", "mseminario")
                    else:
                        if mol.HasProp("_yadonpy_bonded_itp"):
                            _frag = _Path(str(mol.GetProp("_yadonpy_bonded_itp")).strip())
                        if _frag is not None and _frag.is_file():
                            mol.SetProp("_yadonpy_bonded_method", "DRIH")
                            mol.SetProp("_yadonpy_bonded_signature", "drih")
                    if _frag is None or (not _frag.is_file()):
                        raise RuntimeError("explicit bonded override did not produce a bonded fragment")
                except Exception as e:
                    raise RuntimeError(f"bonded={bonded!r} requested but fragment verification failed: {e}") from e
            except Exception as e:
                # If the user explicitly requested bonded params, fail loudly.
                raise RuntimeError(f"bonded={bonded!r} requested but failed: {e}") from e

        if result:
            try:
                refresh_detection = str(charge_kwargs.get("polyelectrolyte_detection", "auto") or "auto")
                refresh_requested = bool(charge_kwargs.get("polyelectrolyte_mode", False))
                has_polyelectrolyte_props = False
                if hasattr(mol, "HasProp"):
                    for key in (
                        "_yadonpy_charge_groups_json",
                        "_yadonpy_resp_constraints_json",
                        "_yadonpy_polyelectrolyte_summary_json",
                    ):
                        if mol.HasProp(key):
                            has_polyelectrolyte_props = True
                            break
                molecule_formal_charge = int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))
                smiles_hint = ""
                try:
                    smiles_hint = Chem.MolToSmiles(mol, isomericSmiles=True)
                except Exception:
                    smiles_hint = ""
                if refresh_requested or has_polyelectrolyte_props or molecule_formal_charge != 0 or "*" in smiles_hint:
                    annotate_polyelectrolyte_metadata(mol, detection=refresh_detection)
            except Exception:
                pass

        if _spec_mode:
            return mol if result else False
        if result and report:
            print_ff_assignment_report(mol, ff_obj=self)
        if result:
            try:
                naming.auto_export_assigned_mol(mol, depth=2)
            except Exception:
                pass
        return mol if result else False


    def assign_ptypes(self, mol):
        """
        GAFF.assign_ptypes

        GAFF specific particle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('pair_style', self.pair_style)
        
        for p in mol.GetAtoms():
            if not self.assign_ptypes_atom(p):
                result_flag = False

        ###########################################
        # Assignment of special atom type in GAFF
        ###########################################
        if result_flag:
            self.assign_special_ptype(mol)
        
        return result_flag



    def assign_ptypes_atom(self, p):
        """
        GAFF2.assign_ptypes_atom

        GAFF2 specific particle typing rules for atom.

        Args:
            p: rdkit atom object

        Returns:
            boolean
        """
        result_flag = True

        ######################################
        # Assignment routine of H
        ######################################
        if p.GetSymbol() == 'H':
            nb_sym = p.GetNeighbors()[0].GetSymbol()

            if nb_sym == 'O':
                water = False
                for pb in p.GetNeighbors():
                    if pb.GetSymbol() == 'O' and pb.GetTotalNumHs(includeNeighbors=True) == 2:
                        water = True
                if water:
                    self.set_ptype(p, 'hw')
                else:
                    self.set_ptype(p, 'ho')
                    
            elif nb_sym == 'N':
                self.set_ptype(p, 'hn')
                
            elif nb_sym == 'P':
                self.set_ptype(p, 'hp')
                
            elif nb_sym == 'S':
                self.set_ptype(p, 'hs')
                
            elif nb_sym == 'C':
                for pb in p.GetNeighbors():
                    if pb.GetSymbol() == 'C':
                        elctrwd = 0
                        degree = pb.GetTotalDegree()

                        for pbb in pb.GetNeighbors():
                            pbb_degree = pbb.GetTotalDegree()
                            pbb_sym = pbb.GetSymbol()
                            if pbb_sym in self.elctrwd_elements and pbb_degree < 4: 
                                elctrwd += 1
                        if elctrwd == 0:
                            if str(pb.GetHybridization()) == 'SP2' or str(pb.GetHybridization()) == 'SP':
                                self.set_ptype(p, 'ha')
                            else:
                                self.set_ptype(p, 'hc')
                        elif degree == 4 and elctrwd == 1:
                            self.set_ptype(p, 'h1')
                        elif degree == 4 and elctrwd == 2:
                            self.set_ptype(p, 'h2')
                        elif degree == 4 and elctrwd == 3:
                            self.set_ptype(p, 'h3')
                        elif degree == 3 and elctrwd == 1:
                            self.set_ptype(p, 'h4')
                        elif degree == 3 and elctrwd == 2:
                            self.set_ptype(p, 'h5')
                        else:
                            utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                        % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2)
                            result_flag = False

            else:
                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2)
                result_flag = False
                            
                            
                            
        ######################################
        # Assignment routine of C
        ######################################
        elif p.GetSymbol() == 'C':
            hyb = str(p.GetHybridization())
            
            if hyb == 'SP3':
                if p.IsInRingSize(3):
                    self.set_ptype(p, 'cx')
                elif p.IsInRingSize(4):
                    self.set_ptype(p, 'cy')
                else:
                    self.set_ptype(p, 'c3')
                
            elif hyb == 'SP2':
                p_idx = p.GetIdx()
                carbonyl = False
                conj = 0
                for pb in p.GetNeighbors():
                    if pb.GetSymbol() == 'O':
                        pb_idx = pb.GetIdx()
                        pb_degree = pb.GetTotalDegree()
                        for b in pb.GetBonds():
                            if (
                                (b.GetBeginAtom().GetIdx() == p_idx and b.GetEndAtom().GetIdx() == pb_idx) or
                                (b.GetBeginAtom().GetIdx() == pb_idx and b.GetEndAtom().GetIdx() == p_idx)
                            ):
                                if b.GetBondTypeAsDouble() == 2 and pb_degree == 1:
                                    carbonyl = True

                for b in p.GetBonds():
                    if b.GetIsConjugated():
                        conj += 1

                if carbonyl:
                    self.set_ptype(p, 'c')  # Carbonyl carbon
                elif p.GetIsAromatic():
                    self.set_ptype(p, 'ca')

                    # For biphenyl head atom
                    for b in p.GetBonds():
                        if b.GetBondTypeAsDouble() == 1:
                            bp = b.GetBeginAtom() if p_idx == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                            if bp.GetIsAromatic():
                                self.set_ptype(p, 'cp')
                                if bp.GetSymbol() == 'C':
                                    self.set_ptype(bp, 'cp')

                elif p.IsInRingSize(3):
                    self.set_ptype(p, 'cu')
                elif p.IsInRingSize(4):
                    self.set_ptype(p, 'cv')
                elif conj >= 2:
                    if utils.is_in_ring(p, max_size=self.max_ring_size):
                        self.set_ptype(p, 'cc')
                    else:
                        self.set_ptype(p, 'ce')
                else:
                    self.set_ptype(p, 'c2')  # Other sp2 carbon
                    
            elif hyb == 'SP':
                conj = 0
                for b in p.GetBonds():
                    if b.GetIsConjugated():
                        conj += 1

                if conj >= 2:
                    self.set_ptype(p, 'cg')
                else:
                    self.set_ptype(p, 'c1')
                
            else:
                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                result_flag = False
                
                
                
        ######################################
        # Assignment routine of N
        ######################################
        elif p.GetSymbol() == 'N':
            hyb = str(p.GetHybridization())
            degree = p.GetTotalDegree()

            if hyb == 'SP':
                self.set_ptype(p, 'n1')
                
            elif degree == 2:
                bond_orders = []
                conj = 0
                for b in p.GetBonds():
                    bond_orders.append(b.GetBondTypeAsDouble())
                    if b.GetIsConjugated():
                        conj += 1

                if p.GetIsAromatic():
                    self.set_ptype(p, 'nb')
                elif 2 in bond_orders:
                    if conj >= 2:
                        if utils.is_in_ring(p, max_size=self.max_ring_size):
                            self.set_ptype(p, 'nc')
                        else:
                            self.set_ptype(p, 'ne')
                    else:
                        self.set_ptype(p, 'n2')
                elif self._is_sulfonimide_like_nitrogen(p):
                    # Sulfonimide anions such as FSI-/TFSI- have an amide-like
                    # central nitrogen bonded to two sulfonyl sulfurs.
                    self.set_ptype(p, 'n')
                else:
                    utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                                % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                    result_flag = False
                    
            elif degree == 3:
                amide = False
                aromatic_ring = False
                no2 = 0
                sp2 = 0
                for pb in p.GetNeighbors():
                    pb_sym = pb.GetSymbol()
                    pb_hyb = str(pb.GetHybridization())

                    if pb_sym == 'C':
                        if pb.GetIsAromatic():
                            aromatic_ring = True
                        for b in pb.GetBonds():
                            bp = b.GetBeginAtom() if pb.GetIdx() == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                            if (bp.GetSymbol() == 'O' or bp.GetSymbol() == 'S') and b.GetBondTypeAsDouble() == 2:
                                amide = True
                    elif pb_sym == 'O':
                        no2 += 1
                    if pb_hyb == 'SP2' or pb_hyb == 'SP':
                        sp2 += 1
                if no2 >= 2:
                    self.set_ptype(p, 'no')
                elif amide:
                    self.set_ptype(p, 'n')
                elif p.GetIsAromatic():
                    self.set_ptype(p, 'na')
                elif sp2 >= 2:
                    self.set_ptype(p, 'na')
                elif aromatic_ring:
                    self.set_ptype(p, 'nh')
                else:
                    self.set_ptype(p, 'n3')
                    
            elif degree == 4:
                self.set_ptype(p, 'n4')
                
            else:
                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                result_flag = False



        ######################################
        # Assignment routine of O
        ######################################
        elif p.GetSymbol() == 'O':
            heavy_neighbor_count = sum(1 for nb in p.GetNeighbors() if nb.GetAtomicNum() > 1)
            if p.GetTotalDegree() == 1:
                self.set_ptype(p, 'o')
            elif heavy_neighbor_count >= 2:
                # Bridge/ether oxygens should remain `os` even if an
                # unsanitized intermediate still reports a residual H count.
                self.set_ptype(p, 'os')
            elif p.GetTotalNumHs(includeNeighbors=True) == 2:
                self.set_ptype(p, 'ow')
            elif p.GetTotalNumHs(includeNeighbors=True) == 1:
                self.set_ptype(p, 'oh')
            else:
                self.set_ptype(p, 'os')



        ######################################
        # Assignment routine of F, Cl, Br, I
        ######################################
        elif p.GetSymbol() == 'F':
            self.set_ptype(p, 'f')
        elif p.GetSymbol() == 'Cl':
            self.set_ptype(p, 'cl')
        elif p.GetSymbol() == 'Br':
            self.set_ptype(p, 'br')
        elif p.GetSymbol() == 'I':
            self.set_ptype(p, 'i')
            
            
            
        ######################################
        # Assignment routine of P
        ######################################
        elif p.GetSymbol() == 'P':
            p_idx = p.GetIdx()
            degree = p.GetTotalDegree()

            if p.GetIsAromatic():
                self.set_ptype(p, 'pb')
                
            elif degree == 2:
                conj = 0
                for b in p.GetBonds():
                    if b.GetIsConjugated():
                        conj += 1

                if conj >= 2:
                    if utils.is_in_ring(p, max_size=self.max_ring_size):
                        self.set_ptype(p, 'pc')
                    else:
                        self.set_ptype(p, 'pe')
                else:
                    self.set_ptype(p, 'p2')
                
            elif degree == 3:
                bond_orders = [x.GetBondTypeAsDouble() for x in p.GetBonds()]
                if 2 in bond_orders:
                    conj = False
                    for pb in p.GetNeighbors():
                        for b in pb.GetBonds():
                            if b.GetBeginAtom().GetIdx() != p_idx and b.GetEndAtom().GetIdx() != p_idx:
                                if b.GetBondTypeAsDouble() >= 1.5:
                                    conj = True
                    if conj:
                        self.set_ptype(p, 'px')
                    else:
                        self.set_ptype(p, 'p4')
                else:
                    self.set_ptype(p, 'p3')
                    
            elif degree == 4:
                conj = False
                for pb in p.GetNeighbors():
                    for b in pb.GetBonds():
                        if b.GetBeginAtom().GetIdx() != p_idx and b.GetEndAtom().GetIdx() != p_idx:
                            if b.GetBondTypeAsDouble() >= 1.5:
                                conj = True
                if conj:
                    self.set_ptype(p, 'py')
                else:
                    self.set_ptype(p, 'p5')

            elif degree == 5 or degree == 6:
                self.set_ptype(p, 'p5')
                    
            else:
                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                result_flag = False



        ######################################
        # Assignment routine of S
        ######################################
        elif p.GetSymbol() == 'S':
            p_idx = p.GetIdx()
            degree = p.GetTotalDegree()

            if degree == 1:
                self.set_ptype(p, 's')
                
            elif degree == 2:
                bond_orders = [x.GetBondTypeAsDouble() for x in p.GetBonds()]
                if p.GetIsAromatic():
                    self.set_ptype(p, 'ss')
                elif p.GetTotalNumHs(includeNeighbors=True) == 1:
                    self.set_ptype(p, 'sh')
                elif 2 in bond_orders:
                    self.set_ptype(p, 's2')
                else:
                    self.set_ptype(p, 'ss')
                
            elif degree == 3:
                conj = False
                for pb in p.GetNeighbors():
                    for b in pb.GetBonds():
                        if b.GetBeginAtom().GetIdx() != p_idx and b.GetEndAtom().GetIdx() != p_idx:
                            if b.GetBondTypeAsDouble() >= 1.5:
                                conj = True
                if conj:
                    self.set_ptype(p, 'sx')
                else:
                    self.set_ptype(p, 's4')
                    
            elif degree == 4:
                conj = False
                for pb in p.GetNeighbors():
                    for b in pb.GetBonds():
                        if b.GetBeginAtom().GetIdx() != p_idx and b.GetEndAtom().GetIdx() != p_idx:
                            if b.GetBondTypeAsDouble() >= 1.5:
                                conj = True
                if conj:
                    self.set_ptype(p, 'sy')
                else:
                    self.set_ptype(p, 's6')

            elif degree == 5 or degree == 6:
                self.set_ptype(p, 's6')
                
            else:
                utils.radon_print('Cannot assignment index %i, element %s, num. of bonds %i, hybridization %s'
                            % (p.GetIdx(), p.GetSymbol(), p.GetTotalDegree(), str(p.GetHybridization())), level=2 )
                result_flag = False


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
        
        return result_flag


    def assign_special_ptype(self, mol):
        """
        GAFF.assign_special_ptype
        
            Assignment of special particle type in GAFF
            C: cc, cd, ce, cf, cg, ch, cp, cq
            N: nc, nd, ne, nf
            P: pc, pd, pe, pf
        """
        for p in mol.GetAtoms():
            self.assign_special_ptype_atom(p)
        
        return True
        
        
    def assign_special_ptype_atom(self, p):
        # Replacement of ce to cf
        if p.GetProp('ff_type') in self.conj_chain: # Chain
            for b in p.GetBonds():
                if b.GetBondTypeAsDouble() == 2 or b.GetBondTypeAsDouble() == 3:
                    bp = b.GetBeginAtom() if p.GetIdx() == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                    if bp.GetProp('ff_type') in self.conj_chain:
                        self.set_ptype(bp, self.conj_rep[bp.GetProp('ff_type')])
                        for bpb in bp.GetBonds():
                            if bpb.GetBondTypeAsDouble() == 1:
                                bpbp = bpb.GetBeginAtom() if bp.GetIdx() == bpb.GetEndAtom().GetIdx() else bpb.GetEndAtom()
                                if bpbp.GetProp('ff_type') in self.conj_chain:
                                    self.set_ptype(bpbp, self.conj_rep[bpbp.GetProp('ff_type')])
                                    
        # Replacement of cc to cd
        elif p.GetProp('ff_type') in self.conj_ring: # Kekulized Ring
            for b in p.GetBonds():
                if b.GetBondTypeAsDouble() == 2:
                    bp = b.GetBeginAtom() if p.GetIdx() == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                    if bp.GetProp('ff_type') in self.conj_ring:
                        self.set_ptype(bp, self.conj_rep[bp.GetProp('ff_type')])
                        for bpb in bp.GetBonds():
                            if bpb.GetBondTypeAsDouble() == 1:
                                bpbp = bpb.GetBeginAtom() if bp.GetIdx() == bpb.GetEndAtom().GetIdx() else bpb.GetEndAtom()
                                if bpbp.GetProp('ff_type') in self.conj_ring:
                                    self.set_ptype(bpbp, self.conj_rep[bpbp.GetProp('ff_type')])

        # Replacement of cp to cq
        elif p.GetProp('ff_type') == 'cp':
            for b in p.GetBonds():
                if b.GetBondTypeAsDouble() == 1.5:
                    bp = b.GetBeginAtom() if p.GetIdx() == b.GetEndAtom().GetIdx() else b.GetEndAtom()
                    if bp.GetProp('ff_type') == 'cp':
                        self.set_ptype(bp, 'cq')
                        for cqb in bp.GetBonds():
                            if cqb.GetBondTypeAsDouble() == 1:
                                cqbp = cqb.GetBeginAtom() if bp.GetIdx() == cqb.GetEndAtom().GetIdx() else cqb.GetEndAtom()
                                if cqbp.GetProp('ff_type') == 'cp':
                                    self.set_ptype(cqbp, 'cq')
        return True


    def set_ptype(self, p, pt):
        p.SetProp('ff_type', pt)
        p.SetDoubleProp('ff_epsilon', self.param.pt[pt].epsilon)
        p.SetDoubleProp('ff_sigma', self.param.pt[pt].sigma)
        
        return p
        
        
    def assign_btypes(self, mol):
        """
        GAFF.assign_btypes

        GAFF specific bond typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('bond_style', self.bond_style)

        for b in mol.GetBonds():
            if not self.assign_btypes_bond(b):
                result_flag = False

        return result_flag
    

    def assign_btypes_bond(self, b):
        """
        GAFF.assign_btypes

        GAFF specific bond typing rules for a bond.

        Args:
            b: rdkit bond object

        Returns:
            boolean
        """
        result_flag = True
        ba = b.GetBeginAtom().GetProp('ff_type')
        bb = b.GetEndAtom().GetProp('ff_type')
        bt = '%s,%s' % (ba, bb)
        
        result = self.set_btype(b, bt)
        if not result:
            alt1 = self.alt_ptype[ba] if ba in self.alt_ptype else None
            alt2 = self.alt_ptype[bb] if bb in self.alt_ptype else None
            if alt1 is None and alt2 is None:
                utils.radon_print(('Can not assign this bond %s,%s' % (ba, bb)), level=2)
                result_flag = False
                return result_flag
            
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
        GAFF.assign_atypes

        GAFF specific angle typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('angle_style', self.angle_style)
        setattr(mol, 'angles', {})
        
        for p in mol.GetAtoms():
            if not self.assign_atypes_atom(mol, p):
                result_flag = False

        return result_flag
        

    def assign_atypes_atom(self, mol, p, replace=False):
        """
        GAFF.assign_atypes_atom

        GAFF specific angle typing rules for an atom.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        b = p.GetIdx()

        for p1 in p.GetNeighbors():
            a = p1.GetIdx()

            for p2 in p.GetNeighbors():
                c = p2.GetIdx()

                if a == c:
                    continue

                unique = True
                key1 = '%i,%i,%i' % (a, b, c)
                key2 = '%i,%i,%i' % (c, b, a)

                if key1 in mol.angles:
                    if replace:
                        del mol.angles[key1]
                    else:
                        unique = False
                elif key2 in mol.angles:
                    if replace:
                        del mol.angles[key2]
                    else:
                        unique = False

                if unique or replace:
                    pt1 = p1.GetProp('ff_type')
                    pt = p.GetProp('ff_type')
                    pt2 = p2.GetProp('ff_type')
                    at = '%s,%s,%s' % (pt1, pt, pt2)
                    
                    result = self.set_atype(mol, a=a, b=b, c=c, at=at)
                    
                    if not result:
                        alt1 = self.alt_ptype[pt1] if pt1 in self.alt_ptype else None
                        alt2 = self.alt_ptype[pt] if pt in self.alt_ptype else None
                        alt3 = self.alt_ptype[pt2] if pt2 in self.alt_ptype else None
                        if alt1 is None and alt2 is None and alt3 is None:
                            emp_result = self.empirical_angle_param(mol, p1, p, p2)
                            if not emp_result:
                                utils.radon_print(('Can not assign this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                                result_flag = False
                            continue

                        at_alt = []
                        if alt1: at_alt.append('%s,%s,%s' % (alt1, pt, pt2))
                        if alt2: at_alt.append('%s,%s,%s' % (pt1, alt2, pt2))
                        if alt3: at_alt.append('%s,%s,%s' % (pt1, pt, alt3))
                        if alt1 and alt2: at_alt.append('%s,%s,%s' % (alt1, alt2, pt2))
                        if alt1 and alt3: at_alt.append('%s,%s,%s' % (alt1, pt, alt3))
                        if alt2 and alt3: at_alt.append('%s,%s,%s' % (pt1, alt2, alt3))
                        if alt1 and alt2 and alt3: at_alt.append('%s,%s,%s' % (alt1, alt2, alt3))
                        
                        for at in at_alt:
                            result = self.set_atype(mol, a=a, b=b, c=c, at=at)
                            if result:
                                utils.radon_print('Using alternate angle type %s instead of %s,%s,%s' % (at, pt1, pt, pt2))
                                break
                                
                        if not result:
                            emp_result = self.empirical_angle_param(mol, p1, p, p2)
                            if not emp_result:
                                utils.radon_print(('Can not assign this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                                result_flag = False

        return result_flag


    def empirical_angle_param(self, mol, a, b, c):

        param_C = {'H':0.000, 'C':1.339, 'N':1.300, 'O':1.249, 'F':0.000, 'Cl':0.000, 'Br':0.000, 'I':0.000, 'P':0.906, 'S':1.448, 'Si':0.894}
        param_Z = {'H':0.784, 'C':1.183, 'N':1.212, 'O':1.219, 'F':1.166, 'Cl':1.272, 'Br':1.378, 'I':1.398, 'P':1.620, 'S':1.280, 'Si':1.016}

        emp_theta = None
        emp_k_ang = None

        pt1 = a.GetProp('ff_type')
        pt = b.GetProp('ff_type')
        pt2 = c.GetProp('ff_type')

        at1 = '%s,%s,%s' % (pt1, pt, pt1)
        at2 = '%s,%s,%s' % (pt2, pt, pt2)

        bt1 = '%s,%s' % (pt1, pt)
        bt2 = '%s,%s' % (pt, pt2)

        if b.GetSymbol() in ['H', 'F', 'Cl', 'Br', 'I']:
            utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
            return False

        for atom in (a, b, c):
            if atom.GetSymbol() not in param_C or atom.GetSymbol() not in param_Z:
                utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                return False

        # Estimate theta0
        if at1 not in self.param.at or at2 not in self.param.at or bt1 not in self.param.bt or bt2 not in self.param.bt:
            alt1 = self.alt_ptype[pt1] if pt1 in self.alt_ptype else None
            alt2 = self.alt_ptype[pt] if pt in self.alt_ptype else None
            alt3 = self.alt_ptype[pt2] if pt2 in self.alt_ptype else None
            if alt1 is None and alt2 is None and alt3 is None:
                utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                return False

            if at1 not in self.param.at:
                if alt1 and '%s,%s,%s' % (alt1, pt, alt1) in self.param.at:
                    at1 = '%s,%s,%s' % (alt1, pt, alt1)
                elif alt2 and '%s,%s,%s' % (pt1, alt2, pt1) in self.param.at:
                    at1 = '%s,%s,%s' % (pt1, alt2, pt1)
                elif alt1 and alt2 and '%s,%s,%s' % (alt1, alt2, alt1) in self.param.at:
                    at1 = '%s,%s,%s' % (alt1, alt2, alt1)
                else:
                    utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                    return False

            if at2 not in self.param.at:
                if alt3 and '%s,%s,%s' % (alt3, pt, alt3) in self.param.at:
                    at2 = '%s,%s,%s' % (alt3, pt, alt3)
                elif alt2 and '%s,%s,%s' % (pt2, alt2, pt2) in self.param.at:
                    at2 = '%s,%s,%s' % (pt2, alt2, pt2)
                elif alt3 and alt2 and '%s,%s,%s' % (alt3, alt2, alt3) in self.param.at:
                    at2 = '%s,%s,%s' % (alt3, alt2, alt3)
                else:
                    utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                    return False

            if bt1 not in self.param.bt:
                if alt1 and '%s,%s' % (alt1, pt) in self.param.bt:
                    bt1 = '%s,%s' % (alt1, pt)
                elif alt2 and '%s,%s' % (pt1, alt2) in self.param.bt:
                    bt1 = '%s,%s' % (pt1, alt2)
                elif alt1 and alt2 and '%s,%s' % (alt1, alt2) in self.param.bt:
                    bt1 = '%s,%s' % (alt1, alt2)
                else:
                    utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                    return False

            if bt2 not in self.param.bt:
                if alt2 and '%s,%s' % (alt2, pt2) in self.param.bt:
                    bt2 = '%s,%s' % (alt2, pt2)
                elif alt3 and '%s,%s' % (pt, alt3) in self.param.bt:
                    bt2 = '%s,%s' % (pt, alt3)
                elif alt2 and alt3 and '%s,%s' % (alt2, alt3) in self.param.bt:
                    bt2 = '%s,%s' % (alt2, alt3)
                else:
                    utils.radon_print(('Can not estimate parameters of this angle %s,%s,%s' % (pt1, pt, pt2)), level=2)
                    return False

        emp_theta = (self.param.at[at1].theta0 + self.param.at[at2].theta0)/2
        emp_k_ang = (143.9*param_Z[a.GetSymbol()]*param_C[b.GetSymbol()]*param_Z[c.GetSymbol()]
                    / (self.param.bt[bt1].r0 + self.param.bt[bt2].r0) / np.sqrt(emp_theta*np.pi/180)
                    * np.exp(-2*(self.param.bt[bt1].r0 - self.param.bt[bt2].r0)**2/(self.param.bt[bt1].r0 + self.param.bt[bt2].r0)**2) )

        angle = utils.Angle(
            a=a.GetIdx(), b=b.GetIdx(), c=c.GetIdx(),
            ff=self.Angle_ff(
                ff_type = '%s,%s,%s' % (pt1, pt, pt2),
                k = emp_k_ang,
                theta0 = emp_theta
            )
        )
        
        key = '%i,%i,%i' % (a.GetIdx(), b.GetIdx(), c.GetIdx())
        mol.angles[key] = angle

        utils.radon_print('Using empirical angle parameters theta0 = %f, k_angle = %f for %s,%s,%s'
                    % (emp_theta, emp_k_ang, pt1, pt, pt2), level=1)

        return True
        

    def set_atype(self, mol, a, b, c, at):
        if at not in self.param.at:
            return False
    
        angle = utils.Angle(
            a=a, b=b, c=c,
            ff=ff_class.Angle_harmonic(
                ff_type=self.param.at[at].tag,
                k=self.param.at[at].k,
                theta0=self.param.at[at].theta0
            )
        )
        
        key = '%i,%i,%i' % (a, b, c)
        mol.angles[key] = angle
        
        return True


    def assign_dtypes(self, mol):
        """
        GAFF.assign_dtypes

        GAFF specific dihedral typing rules.
        
        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        result_flag = True
        mol.SetProp('dihedral_style', self.dihedral_style)
        setattr(mol, 'dihedrals', {})

        for b in mol.GetBonds():
            if not self.assign_dtypes_bond(mol, b):
                result_flag = False

        return result_flag


    def assign_dtypes_bond(self, mol, bond, replace=False):
        """
        GAFF.assign_dtypes

        GAFF specific dihedral typing rules for a bond.
        
        Args:
            bond: rdkit bond object

        Returns:
            boolean
        """
        result_flag = True
        p1 = bond.GetBeginAtom()
        p2 = bond.GetEndAtom()
        b = p1.GetIdx()
        c = p2.GetIdx()

        for p1b in p1.GetNeighbors():
            a = p1b.GetIdx()
            if c == a:
                continue

            for p2b in p2.GetNeighbors():
                d = p2b.GetIdx()

                if b == d or a == d:
                    continue

                unique = True
                key1 = '%i,%i,%i,%i' % (a, b, c, d)
                key2 = '%i,%i,%i,%i' % (d, c, b, a)

                if key1 in mol.dihedrals:
                    if replace:
                        del mol.dihedrals[key1]
                    else:
                        unique = False
                elif key2 in mol.dihedrals:
                    if replace:
                        del mol.dihedrals[key2]
                    else:
                        unique = False

                if unique or replace:
                    p1bt = p1b.GetProp('ff_type')
                    p1t = p1.GetProp('ff_type')
                    p2t = p2.GetProp('ff_type')
                    p2bt = p2b.GetProp('ff_type')
                    dt = '%s,%s,%s,%s' % (p1bt, p1t, p2t, p2bt)
                    
                    result = self.set_dtype(mol, a=a, b=b, c=c, d=d, dt=dt)
                    
                    if not result:
                        alt1 = self.alt_ptype[p1t] if p1t in self.alt_ptype else None
                        alt2 = self.alt_ptype[p2t] if p2t in self.alt_ptype else None
                        if alt1 is None and alt2 is None:
                            utils.radon_print('Can not assign this dihedral %s,%s,%s,%s' % (p1bt, p1t, p2t, p2bt), level=2)
                            result_flag = False
                            continue
                        
                        dt_alt = []
                        if alt1: dt_alt.append('%s,%s,%s,%s' % (p1bt, alt1, p2t, p2bt))
                        if alt2: dt_alt.append('%s,%s,%s,%s' % (p1bt, p1t, alt2, p2bt))
                        if alt1 and alt2: dt_alt.append('%s,%s,%s,%s' % (p1bt, alt1, alt2, p2bt))
                        
                        for dt in dt_alt:
                            result = self.set_dtype(mol, a=a, b=b, c=c, d=d, dt=dt)
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
            ff=ff_class.Dihedral_fourier(
                ff_type=self.param.dt[dt].tag,
                k=self.param.dt[dt].k,
                d0=self.param.dt[dt].d,
                m=self.param.dt[dt].m,
                n=self.param.dt[dt].n
            )
        )

        key = '%i,%i,%i,%i' % (a, b, c, d)        
        mol.dihedrals[key] = dihedral
        
        return True


    def assign_itypes(self, mol):
        """
        GAFF.assign_itypes

        GAFF specific improper typing rules.

        Args:
            mol: rdkit mol object

        Returns:
            boolean
        """
        mol.SetProp('improper_style', self.improper_style)
        setattr(mol, 'impropers', {})
        
        for p in mol.GetAtoms():
            self.assign_itypes_atom(mol, p)
        
        return True            


    def assign_itypes_atom(self, mol, p, replace=False):
        """
        GAFF.assign_itypes_atom

        GAFF specific improper typing rules for an atom.

        Args:
            p: rdkit atom object

        Returns:
            boolean
        """
        if p.GetTotalDegree() == 3:
            a = p.GetIdx()

            for perm in permutations(p.GetNeighbors(), 3):
                pt = p.GetProp('ff_type')
                p1t = perm[0].GetProp('ff_type')
                p2t = perm[1].GetProp('ff_type')
                p3t = perm[2].GetProp('ff_type')
                it = '%s,%s,%s,%s' % (pt, p1t, p2t, p3t)
                b = perm[0].GetIdx()
                c = perm[1].GetIdx()
                d = perm[2].GetIdx()
                key = '%i,%i,%i,%i' % (a, b, c, d)

                if replace:
                    if key in mol.impropers:
                        del mol.impropers[key]

                result = self.set_itype(mol, a=a, b=b, c=c, d=d, it=it)
                
                if not result:
                    alt1 = self.alt_ptype[pt] if pt in self.alt_ptype else None
                    alt2 = self.alt_ptype[p1t] if p1t in self.alt_ptype else None
                    alt3 = self.alt_ptype[p2t] if p2t in self.alt_ptype else None
                    alt4 = self.alt_ptype[p3t] if p3t in self.alt_ptype else None
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
                        result = self.set_itype(mol, a=a, b=b, c=c, d=d, it=it)
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
            if it1 in self.param.it:
                it = it1
            elif it2 in self.param.it:
                it = it2
            else:
                return False
            
        improper = utils.Improper(
            a=a, b=b, c=c, d=d,
            ff=ff_class.Improper_cvff(
                ff_type=self.param.it[it].tag,
                k=self.param.it[it].k,
                d0=self.param.it[it].d,
                n=self.param.it[it].n
            )
        )
        
        key = '%i,%i,%i,%i' % (a, b, c, d)
        mol.impropers[key] = improper
        
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
            GAFF.Angle_ff() object
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
            GAFF.Dihedral_ff() object
        """
        def __init__(self, ff_type=None, k=[], d0=[], m=None, n=[]):
            self.type = ff_type
            self.k = np.array(k)
            self.d0 = np.array(d0)
            self.d0_rad = np.array(d0)*(np.pi/180)
            self.m = m
            self.n = np.array(n)
        
        def to_dict(self):
            dic = {
                'ff_type': str(self.type),
                'k': [float(x) for x in self.k],
                'd0': [float(x) for x in self.d0],
                'm': int(self.m),
                'n': [int(x) for x in self.n],
            }
            return dic

        
    class Improper_ff():
        """
            GAFF.Improper_ff() object
        """
        def __init__(self, ff_type=None, k=None, d0=-1, n=None):
            self.type = ff_type
            self.k = k
            self.d0 = d0
            self.n = n
        
        def to_dict(self):
            dic = {
                'ff_type': str(self.type),
                'k': float(self.k),
                'd0': float(self.d0),
                'n': int(self.n),
            }
            return dic



    # ---------------------------------------------------------------------
    # MolDB-backed handle API (v0.6.3+)
    # ---------------------------------------------------------------------
    def mol(
        self,
        smiles_or_psmiles: str,
        *,
        name: str | None = None,
        basis_set: str | None = None,
        method: str | None = None,
        charge: str = "RESP",
        require_ready: bool = True,
        prefer_db: bool = True,
        polyelectrolyte_mode: bool | None = None,
        polyelectrolyte_detection: str | None = None,
        resp_profile: str | None = None,
    ):
        """Create a lightweight MolSpec handle.

        The handle is resolved into an RDKit Mol when you call ff_assign().
        """
        from ..core.molspec import MolSpec

        return MolSpec(
            smiles=str(smiles_or_psmiles).strip(),
            name=(str(name).strip() if name else None),
            charge=str(charge).strip() if charge else "RESP",
            basis_set=(str(basis_set).strip() if basis_set else None),
            method=(str(method).strip() if method else None),
            resp_profile=(str(resp_profile).strip() if resp_profile else None),
            require_ready=bool(require_ready),
            prefer_db=bool(prefer_db),
            polyelectrolyte_mode=polyelectrolyte_mode,
            polyelectrolyte_detection=(str(polyelectrolyte_detection).strip() if polyelectrolyte_detection else None),
        )

    # ---------------------------------------------------------------------
    # Shared molecule database helpers (RDKit Mol; geometry + charges)
    # ---------------------------------------------------------------------
    @classmethod
    def mol_rdkit(
        cls,
        smiles_or_psmiles: str,
        *,
        name: str | None = None,
        db_dir: str | os.PathLike | None = None,
        prefer_db: bool = True,
        require_db: bool = False,
        require_ready: bool = False,
        charge: str = "RESP",
        basis_set: str | None = None,
        method: str | None = None,
        resp_profile: str | None = None,
        polyelectrolyte_mode: bool | None = None,
        polyelectrolyte_detection: str | None = None,
    ):
        """Create or load a molecule (RDKit Mol) from the shared MolDB.

        MolDB stores ONLY:
          - canonical smiles/psmiles
          - a best initial 3D geometry (mol2)
          - charges (RESP etc., if available)

        Force-field assignment is intentionally NOT stored here.

        Args:
            smiles_or_psmiles: SMILES (small molecule / ion) or PSMILES (polymer building block, contains '*').
            name: Optional friendly name stored into the record and used for exports.
            db_dir: Optional DB directory. If not set, uses the default MolDB (typically ~/.yadonpy/moldb).
            prefer_db: If True and the record exists, load it; otherwise (re)build a fresh initial geometry.
            require_db: If True, the entry must exist in DB (geometry required; charges may be missing).
            require_ready: If True, the entry must exist and be marked ready (charges present)
                for the requested (charge,basis_set,method) variant.
        """
        from pathlib import Path
        from ..moldb import MolDB

        db = MolDB(Path(db_dir) if db_dir is not None else None)
        # Priority:
        #   1) require_ready=True: must exist AND have charges (ready)
        #   2) require_db=True   : must exist (charges optional)
        #   3) otherwise         : build-or-load (charges optional)
        if require_ready:
            mol, rec = db.load_mol(
                smiles_or_psmiles,
                require_ready=True,
                charge=charge,
                basis_set=basis_set,
                method=method,
                resp_profile=resp_profile,
                polyelectrolyte_mode=polyelectrolyte_mode,
                polyelectrolyte_detection=polyelectrolyte_detection,
            )
        elif require_db:
            mol, rec = db.load_mol(
                smiles_or_psmiles,
                require_ready=False,
                charge=charge,
                basis_set=basis_set,
                method=method,
                resp_profile=resp_profile,
                polyelectrolyte_mode=polyelectrolyte_mode,
                polyelectrolyte_detection=polyelectrolyte_detection,
            )
        else:
            mol, rec = db.build_or_load(smiles_or_psmiles, name=name, prefer_db=prefer_db)

        # Prefer explicit name; otherwise keep record name for nicer exports.
        if name:
            try:
                mol.SetProp("_Name", str(name))
            except Exception:
                pass
        elif rec and getattr(rec, "name", None):
            try:
                mol.SetProp("_Name", str(rec.name))
            except Exception:
                pass
        return mol

    @classmethod
    def store_to_db(
        cls,
        mol,
        *,
        smiles_or_psmiles: str | None = None,
        name: str | None = None,
        db_dir: str | os.PathLike | None = None,
        charge: str = "RESP",
        basis_set: str | None = None,
        method: str | None = None,
        polyelectrolyte_mode: bool | None = None,
        polyelectrolyte_detection: str | None = None,
    ):
        """Store current geometry + charges of an RDKit mol into shared MolDB."""
        from pathlib import Path
        from ..moldb import MolDB
        db = MolDB(Path(db_dir) if db_dir is not None else None)
        return db.update_from_mol(
            mol,
            smiles_or_psmiles=smiles_or_psmiles,
            name=name,
            charge=charge,
            basis_set=basis_set,
            method=method,
            polyelectrolyte_mode=polyelectrolyte_mode,
            polyelectrolyte_detection=polyelectrolyte_detection,
        )
