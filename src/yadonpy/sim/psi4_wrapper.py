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
# sim.psi4_wrapper module
# ******************************************************************************

import numpy as np
import os
import gc
import datetime
import socket
import multiprocessing as MP
import concurrent.futures as confu
from packaging.version import parse as parse_version
from rdkit import Chem
from rdkit import Geometry as Geom

# Psi4 and PsiRESP are optional dependencies.
# We must not fail at import time when users only want non-QM workflows.
#
# IMPORTANT: Some users only need Psi4 for geometry optimizations / ESP tasks
# that do *not* require the Python package `psiresp`. Therefore:
#   - We require Psi4 at object construction.
#   - We require `psiresp` lazily only when calling Psi4w.resp().
#
# Also, catch broad exceptions and preserve the root cause for better error
# messages (conda environments can have ABI issues that raise OSError).
psi4 = None
_psi4_import_error: Exception | None = None
try:  # pragma: no cover
    import psi4  # type: ignore
except Exception as e:  # pragma: no cover
    _psi4_import_error = e
    psi4 = None

from ..core import const, calc, utils


def _version_key(raw: object):
    return parse_version(str(raw))


if psi4 is not None:
    if _version_key(psi4.__version__) >= _version_key('1.4'):
        import qcengine  # type: ignore

if const.mpi4py_avail:
    try:
        from mpi4py.futures import MPIPoolExecutor
    except ImportError as e:
        utils.radon_print('Cannot import mpi4py. Change to const.mpi4py_avail = False. %s' % e, level=2)
        const.mpi4py_avail = False


class Psi4w():
    def __init__(self, mol, confId=0, work_dir=None, tmp_dir=None, name=None, **kwargs):
        if psi4 is None:
            detail = f" (root cause: {_psi4_import_error!r})" if _psi4_import_error else ""
            raise ImportError(
                "QM features require optional dependency 'psi4'. "
                "Install e.g. via: conda install -c psi4 psi4" + detail
            )
        self.work_dir = work_dir if work_dir is not None else './'
        self.tmp_dir = tmp_dir if tmp_dir is not None else self.work_dir

        # ------------------------------------------------------------------
        # Keep Psi4 scratch/log files contained
        # ------------------------------------------------------------------
        # Psi4 (and OptKing) can emit many auxiliary files (psi.* binaries,
        # timer.dat, scratch matrices, etc.). To keep the user's work_dir
        # tidy, we route *all* Psi4-produced files into a dedicated
        # "psi4/" subfolder under the provided work_dir.
        #
        # This mirrors yzc-gmx-gen's philosophy: module outputs live in
        # predictable subdirectories instead of cluttering the root.
        try:
            w_abs = os.path.abspath(self.work_dir)
            if os.path.basename(w_abs) == 'psi4':
                self.psi4_dir = w_abs
            else:
                self.psi4_dir = os.path.join(w_abs, 'psi4')
            self.psi4_scratch = os.path.join(self.psi4_dir, 'scratch')
            os.makedirs(self.psi4_scratch, exist_ok=True)
        except Exception:
            # Fall back to legacy behavior (best-effort) if path ops fail.
            self.psi4_dir = self.work_dir
            self.psi4_scratch = self.tmp_dir
        self.num_threads = kwargs.get('num_threads', kwargs.get('omp', utils.cpu_count()))
        self.memory = kwargs.get('memory', 1000) # MByte

        self.name = name if name else 'radonpy'
        self.mol = utils.deepcopy_mol(mol)
        self.confId = confId
        self.wfn = kwargs.get('wfn', None)
        self.charge = kwargs.get('charge', Chem.rdmolops.GetFormalCharge(self.mol))
        nr = calc.get_num_radicals(self.mol)
        self.multiplicity = kwargs.get('multiplicity', 1 if nr == 0 else 2 if nr%2 == 1 else 3)

        # Default functional/basis policy (2026-03):
        # - Use wb97m-d3bj for general electrolyte/FF workflows (Psi4 method keyword).
        # - Users sometimes write "wb97m" informally; in Psi4 v1.10 it is NOT
        #   a valid method keyword, so we treat it as an alias of wb97m-d3bj.
        # - Basis selection for anions is handled at the workflow layer (qm.py/calc.py).
        def _normalize_method(m):
            try:
                s = str(m).strip().lower()
            except Exception:
                return m
            if s == 'wb97m':
                return 'wb97m-d3bj'
            return m

        self.method = _normalize_method(kwargs.get('method', 'wb97m-d3bj'))
        self.basis = kwargs.get('basis', 'def2-SVP')
        basis_Br = kwargs.get('basis_Br', self.basis)
        basis_I = kwargs.get('basis_I', self.basis)
        self.basis_gen = {'Br': basis_Br, 'I': basis_I, **kwargs.get('basis_gen', {})}

        self.scf_type = kwargs.get('scf_type', 'df')
        self.scf_maxiter = kwargs.get('scf_maxiter', 128)
        self.scf_fail_on_maxiter = kwargs.get('scf_fail_on_maxiter', True)
        self.cc2wfn = kwargs.get('cc2wfn', None)
        self.cache_level = kwargs.get('cache_level', 2)
        self.cwd = os.getcwd()
        self.error_flag = False

        if _version_key(psi4.__version__) >= _version_key('1.4'):
            self.get_global_org = qcengine.config.get_global

        # Corresponds to Gaussian keyword
        if kwargs.get('dft_integral', None) == 'fine':
            self.dft_spherical_points = 302
            self.dft_radial_points = 75
        elif kwargs.get('dft_integral', None) == 'ultrafine':
            self.dft_spherical_points = 590
            self.dft_radial_points = 99
        elif kwargs.get('dft_integral', None) == 'superfine':
            self.dft_spherical_points = 974
            self.dft_radial_points = 175
        elif kwargs.get('dft_integral', None) == 'coarse':
            self.dft_spherical_points = 110
            self.dft_radial_points = 35
        elif kwargs.get('dft_integral', None) == 'SG1':
            self.dft_spherical_points = 194
            self.dft_radial_points = 50
        else:
            self.dft_spherical_points = kwargs.get('dft_spherical_points', 590)
            self.dft_radial_points = kwargs.get('dft_radial_points', 99)


    def __del__(self):
        # Be defensive: __init__ may have failed (partial construction),
        # or psi4 might not be importable in this environment.
        try:
            if psi4 is not None:
                psi4.core.clean()
                psi4.core.clean_options()
                psi4.core.clean_variables()
        except Exception:
            pass
        try:
            if hasattr(self, "wfn"):
                del self.wfn
            if hasattr(self, "cc2wfn"):
                del self.cc2wfn
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass

    @property
    def get_name(self):
        return 'Psi4'


    @property
    def psi4_version(self):
        return psi4.__version__


    def _init_psi4(self, *args, output=None):

        # Avoiding errors on Fugaku and in mpi4py
        if _version_key(psi4.__version__) >= _version_key('1.4'):
            qcengine.config.get_global = _override_get_global

        psi4.core.clean_options()
        # Route scratch to the dedicated psi4 scratch directory.
        os.environ['PSI_SCRATCH'] = os.path.abspath(self.psi4_scratch)
        psi4.core.IOManager.shared_object().set_default_path(os.path.abspath(self.psi4_scratch))

        pmol = self._mol2psi4(*args)
        try:
            if self._is_high_symmetry_polyhedral_ion():
                pmol = self._force_c1_molecule(pmol)
        except Exception:
            pass
        self.error_flag = False
        self._basis_set_construction()
        psi4.set_num_threads(self.num_threads)
        psi4.set_memory('%i MB' % (self.memory))
        psi4.set_options({
            'dft_spherical_points': self.dft_spherical_points,
            'dft_radial_points': self.dft_radial_points,
            'scf_type': self.scf_type,
            'maxiter': self.scf_maxiter,
            'fail_on_maxiter': self.scf_fail_on_maxiter,
            'cachelevel': self.cache_level,
            'CC_NUM_THREADS': self.num_threads,
            'basis': 'radonpy_basis'
            })

        # Avoiding the bug due to MKL (https://github.com/psi4/psi4/issues/2279)
        if '1.4rc' in str(psi4.__version__):
            psi4.set_options({'wcombine': False})
        elif _version_key('1.4.1') > _version_key(psi4.__version__) >= _version_key('1.4'):
            psi4.set_options({'wcombine': False})

        self.cwd = os.getcwd()
        # Keep Psi4 outputs inside work_dir/psi4/
        os.chdir(self.psi4_dir)
        if output is not None:
            psi4.core.set_output_file(output, False)

        return pmol

    def _fin_psi4(self):
        psi4.core.clean()
        psi4.core.clean_options()
        if _version_key(psi4.__version__) >= _version_key('1.4'):
            qcengine.config.get_global = self.get_global_org

        # Avoiding the bug that optimization is failed due to the optimization binary file remaining.
        opt_bin_file = os.path.join(self.psi4_scratch, 'psi.%i.1' % os.getpid())
        if os.path.isfile(opt_bin_file):
            os.remove(opt_bin_file)

        os.chdir(self.cwd)
        gc.collect()


    def _mol2psi4(self, *args, symmetry: str | None = None, no_reorient: bool = False, no_com: bool = False):
        """
        Psi4w._mol2psi4

        Convert RDkit Mol object to Psi4 Mol object

        Returns:
            Psi4 Mol object
        """

        geom = '%i   %i\n' % (self.charge, self.multiplicity)
        for arg in args:
            geom += '%s\n' % (arg)

        # ---------------------------------------------------------------------
        # Robust geometry guard
        #
        # QCElemental (used by Psi4) validates the input geometry and can raise
        #   qcelemental.exceptions.ValidationError: Following atoms are too close
        # when two or more atoms occupy the same coordinates.
        #
        # This can occur on some RDKit builds for inorganic/polyhedral ions
        # (e.g., PF6-, BF4-, ClO4-) or highly charged species, where the initial
        # 3D embedding occasionally yields a degenerate conformer (all atoms at
        # the origin). To keep the workflow robust, we detect this and
        # re-initialize a non-degenerate 3D geometry using yadonpy's ion
        # templates / fallback embedding.
        # ---------------------------------------------------------------------

        # Ensure there is at least one conformer
        if self.mol.GetNumConformers() == 0:
            try:
                utils.ensure_3d_coords(self.mol, smiles_hint=getattr(self, 'name', None))
            except Exception:
                pass

        # Get conformer safely (confId is a conformer *ID*, not an index)
        try:
            conf = self.mol.GetConformer(int(self.confId))
        except Exception:
            conf = self.mol.GetConformer(0)
            self.confId = 0

        coord = np.array(conf.GetPositions(), dtype=float)

        # Detect degenerate geometry (any pair at ~0 distance)
        try:
            n = coord.shape[0]
            if n >= 2:
                min_d2 = np.inf
                for i in range(n):
                    dv = coord[i+1:] - coord[i]
                    if dv.size == 0:
                        continue
                    d2 = np.sum(dv * dv, axis=1)
                    md2 = float(np.min(d2))
                    if md2 < min_d2:
                        min_d2 = md2
                if (not np.isfinite(min_d2)) or (min_d2 < 1.0e-6):
                    utils.radon_print('Detected degenerate initial geometry; re-initializing 3D coords for QM.', level=2)
                    try:
                        self.mol.RemoveAllConformers()
                        success = utils.ensure_3d_coords(self.mol, smiles_hint=getattr(self, 'name', None))
                        if not success:
                            raise RuntimeError('ensure_3d_coords returned False')
                    except Exception:
                        # Last resort: simple spaced-out coordinates
                        try:
                            from rdkit import Chem
                            from rdkit.Geometry import rdGeometry as Geom
                            conf2 = Chem.Conformer(self.mol.GetNumAtoms())
                            for ai in range(self.mol.GetNumAtoms()):
                                conf2.SetAtomPosition(ai, Geom.Point3D(float(ai) * 1.5, 0.0, 0.0))
                            self.mol.RemoveAllConformers()
                            self.mol.AddConformer(conf2, assignId=True)
                        except Exception:
                            pass

                    conf = self.mol.GetConformer(0)
                    self.confId = 0
                    coord = np.array(conf.GetPositions(), dtype=float)
        except Exception:
            pass
        for i in range(self.mol.GetNumAtoms()):
            geom += '%2s  % .8f  % .8f  % .8f\n' % (self.mol.GetAtomWithIdx(i).GetSymbol(), coord[i, 0], coord[i, 1], coord[i, 2])
        if symmetry:
            geom += f'symmetry {str(symmetry).strip()}\n'
        if no_reorient:
            geom += 'no_reorient\n'
        if no_com:
            geom += 'no_com\n'

        pmol = psi4.geometry(geom)
        pmol.update_geometry()
        pmol.set_name(self.name)

        return pmol


    def _is_high_symmetry_polyhedral_ion(self) -> bool:
        try:
            return bool(utils.is_high_symmetry_polyhedral_ion(self.mol, smiles_hint=getattr(self, 'name', None)))
        except Exception:
            return False


    def _force_c1_molecule(self, pmol):
        """Force a Psi4 molecule to stay in C1 and keep the current frame."""
        try:
            pmol.reset_point_group('c1')
        except Exception:
            pass
        try:
            pmol.fix_orientation(True)
        except Exception:
            pass
        try:
            pmol.fix_com(True)
        except Exception:
            pass
        try:
            pmol.update_geometry()
        except Exception:
            pass
        return pmol


    def _sync_rdkit_from_psi4_mol(self, pmol):
        """Best-effort coordinate sync from a Psi4 molecule back to the RDKit conformer."""
        try:
            coord = np.asarray(pmol.geometry().to_array(), dtype=float) * const.bohr2ang
        except Exception:
            try:
                coord = np.asarray(pmol.geometry(), dtype=float) * const.bohr2ang
            except Exception:
                return None
        try:
            conf = self.mol.GetConformer(int(self.confId))
        except Exception:
            conf = self.mol.GetConformer(0)
            self.confId = 0
        try:
            for i in range(self.mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Geom.Point3D(float(coord[i, 0]), float(coord[i, 1]), float(coord[i, 2])))
        except Exception:
            pass
        return coord


    def _basis_set_construction(self):
        basis = self.basis.replace('(', '_').replace(')', '_').replace(',', '_').replace('+', 'p').replace('*', 's')
        bs = 'assign %s\n' % basis

        for element, basis in self.basis_gen.items():
            basis = basis.replace('(', '_').replace(')', '_').replace(',', '_').replace('+', 'p').replace('*', 's')
            bs += 'assign %s %s\n' % (element, basis)

        psi4.basis_helper(bs, name='radonpy_basis', set_option=True)


    def energy(self, wfn=True, **kwargs):
        """
        Psi4w.energy

        Single point energy calculation by Psi4

        Optional args:
            wfn: Store the wfn object of Psi4 (boolean)

        Returns:
            energy (float, kJ/mol)
        """

        pmol = self._init_psi4(output='./%s_psi4.log' % self.name)
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 single point calculation is running...', level=1)

        try:
            if wfn:
                energy, self.wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
            else:
                energy = psi4.energy(self.method, molecule=pmol, return_wfn=False, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 single point calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            energy = e.wfn.energy()
            self.error_flag = True

        self._fin_psi4()

        return energy*const.au2kj # Hartree -> kJ/mol


    
    def optimize(
        self,
        wfn=True,
        freeze=[],
        ignore_conv_error=False,
        opt_type='min',
        geom_iter=50,
        geom_conv='QCHEM',
        geom_algorithm='RFO',
        dynamic_level=0,
        opt_coordinates=None,
        **kwargs,
    ):
        '''
        Psi4w.optimize

        Structure optimization calculation by Psi4.

        Notes (robustness):
          - Polyhedral / highly symmetric ions (e.g., PF6-) can trigger OptKing internal-coordinate issues
            ("linear bends detected", etc.). Psi4 recommends OPT_COORDINATES=CARTESIAN (or BOTH) to
            avoid internal coordinate difficulties.
          - If OptKing fails anyway, we optionally retry with the geomeTRIC engine (when available).
          - As a last resort (and when ignore_conv_error=True), we return a single-point energy at the
            current geometry so RESP can still proceed.

        Returns:
            energy (float, kJ/mol)
            coord (ndarray(float), angstrom)
        '''

        pmol = self._init_psi4(output='./%s_psi4_opt.log' % self.name)

        # Older Psi4/OptKing stacks can be fragile with linear angles; bump dynamic_level automatically.
        if dynamic_level == 0 and calc.find_liner_angle(self.mol) and _version_key(psi4.__version__) < _version_key('1.8'):
            utils.radon_print("Found a linear angle in the molecule. Psi4 optimization setting 'dynamic_level' was changed to 2.", level=2)
            dynamic_level = 2
            geom_iter = int(2 * geom_iter)

        opt_dict = {
            'OPT_TYPE': opt_type,
            'GEOM_MAXITER': geom_iter,
            'G_CONVERGENCE': geom_conv,
            'STEP_TYPE': geom_algorithm,
            'OPTKING__ENSURE_BT_CONVERGENCE': True,
            'DYNAMIC_LEVEL': dynamic_level,
        }

        # User-requested coordinate system for OptKing
        if opt_coordinates:
            try:
                opt_dict['OPTKING__OPT_COORDINATES'] = str(opt_coordinates).upper()
            except Exception:
                opt_dict['OPTKING__OPT_COORDINATES'] = opt_coordinates

        # Frozen coordinates
        frozen_bond = []
        frozen_angle = []
        frozen_dihedral = []
        for atoms in freeze:
            if len(atoms) == 2:
                frozen_bond.append('%i %i' % (atoms[0] + 1, atoms[1] + 1))
            elif len(atoms) == 3:
                frozen_angle.append('%i %i %i' % (atoms[0] + 1, atoms[1] + 1, atoms[2] + 1))
            elif len(atoms) == 4:
                frozen_dihedral.append('%i %i %i %i' % (atoms[0] + 1, atoms[1] + 1, atoms[2] + 1, atoms[3] + 1))
            else:
                utils.radon_print('Illegal length of array for input atoms. (2, 3, or 4)', level=3)
        if len(frozen_bond) > 0:
            opt_dict['OPTKING__FROZEN_DISTANCE'] = ' '.join(frozen_bond)
        if len(frozen_angle) > 0:
            opt_dict['OPTKING__FROZEN_BEND'] = ' '.join(frozen_angle)
        if len(frozen_dihedral) > 0:
            opt_dict['OPTKING__FROZEN_DIHEDRAL'] = ' '.join(frozen_dihedral)

        def _run_opt(engine=None, pmol_override=None):
            """Run Psi4 optimization with current opt_dict; return (energy_h, coord_ang, wfn)."""
            nonlocal pmol
            if pmol_override is not None:
                pmol = pmol_override
            psi4.set_options(opt_dict)
            dt1 = datetime.datetime.now()
            try:
                _smi = Chem.MolToSmiles(self.mol)
            except Exception:
                _smi = '?'
            _eng = f" engine={engine}" if engine else ""
            utils.radon_print(
                f"Psi4 OPT start: name={self.name} charge={self.charge} mult={self.multiplicity} "
                f"method={self.method} basis={self.basis}{_eng} smiles={_smi}",
                level=1,
            )

            if wfn:
                if engine:
                    e_h, wfn_obj = psi4.optimize(self.method, molecule=pmol, return_wfn=True, engine=engine, **kwargs)
                else:
                    e_h, wfn_obj = psi4.optimize(self.method, molecule=pmol, return_wfn=True, **kwargs)
            else:
                if engine:
                    e_h = psi4.optimize(self.method, molecule=pmol, return_wfn=False, engine=engine, **kwargs)
                    wfn_obj = None
                else:
                    e_h = psi4.optimize(self.method, molecule=pmol, return_wfn=False, **kwargs)
                    wfn_obj = None

            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 optimization. Elapsed time = %s' % str(dt2 - dt1), level=1)
            coord_ang = pmol.geometry().to_array() * const.bohr2ang
            return e_h, coord_ang, wfn_obj

        energy_h = None
        coord = None
        last_err = None

        try:
            # First attempt: as requested
            energy_h, coord, wfn_obj = _run_opt(engine=None)
            if wfn:
                self.wfn = wfn_obj

        except psi4.OptimizationConvergenceError as e:
            utils.radon_print('Psi4 optimization convergence error. %s' % e, level=2)
            if ignore_conv_error:
                energy_h = e.wfn.energy()
            else:
                energy_h = np.nan
            coord = np.array(e.wfn.molecule().geometry()) * const.bohr2ang
            self.error_flag = True

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            if ignore_conv_error:
                energy_h = e.wfn.energy()
            else:
                energy_h = np.nan
            coord = np.array(e.wfn.molecule().geometry()) * const.bohr2ang
            self.error_flag = True

        except BaseException as e:
            last_err = e

            # Retry 0: restart from the current geometry with symmetry disabled if the
            # optimizer reports a point-group change. This keeps symmetric ions such as
            # PF6- robust while still allowing the symmetry increase to happen naturally.
            try:
                _msg = str(e)
                if ('Point group changed!' in _msg) or ('point group changed' in _msg.lower()):
                    utils.radon_print(f'Point group changed during optimization ({e}). Retrying from the latest geometry with symmetry c1...', level=2)
                    pmol_pg = self._mol2psi4(symmetry='c1', no_reorient=True, no_com=True)
                    pmol_pg = self._force_c1_molecule(pmol_pg)
                    energy_h, coord, wfn_obj = _run_opt(engine=None, pmol_override=pmol_pg)
                    if wfn:
                        self.wfn = wfn_obj
                    last_err = None
                else:
                    raise e
            except BaseException as e_pg:
                last_err = e_pg

            # Retry 1: force CARTESIAN coordinates for OptKing if not already requested
            try:
                oc = opt_dict.get('OPTKING__OPT_COORDINATES', None)
                if oc is None or str(oc).upper() not in ('CARTESIAN', 'BOTH'):
                    utils.radon_print(f'OptKing failed ({e}). Retrying with OPT_COORDINATES=CARTESIAN...', level=2)
                    opt_dict['OPTKING__OPT_COORDINATES'] = 'CARTESIAN'
                    energy_h, coord, wfn_obj = _run_opt(engine=None)
                    if wfn:
                        self.wfn = wfn_obj
                    last_err = None
                else:
                    raise e
            except BaseException as e2:
                last_err = e2

            # Final fallback for PF6-/BF4-/ClO4-/AsF6- like ions:
            # keep a C1-symmetrized geometry and continue to the ESP/RESP stage
            # instead of aborting the whole workflow on a symmetry-only optimizer issue.
            if last_err is not None:
                try:
                    _msg = str(last_err)
                    if self._is_high_symmetry_polyhedral_ion() and ('Point group changed!' in _msg or 'point group changed' in _msg.lower()):
                        utils.radon_print(
                            'Psi4 optimization still reports a point-group change for a high-symmetry inorganic ion. '
                            'Falling back to a C1 fixed geometry and continuing to the charge-fitting step.',
                            level=2,
                        )
                        try:
                            utils.symmetrize_polyhedral_ion_geometry(self.mol, confId=int(self.confId))
                        except Exception:
                            pass
                        pmol_c1 = self._mol2psi4(symmetry='c1', no_reorient=True, no_com=True)
                        pmol_c1 = self._force_c1_molecule(pmol_c1)
                        coord = self._sync_rdkit_from_psi4_mol(pmol_c1)
                        if coord is None:
                            try:
                                conf = self.mol.GetConformer(int(self.confId))
                            except Exception:
                                conf = self.mol.GetConformer(0)
                                self.confId = 0
                            coord = np.asarray(conf.GetPositions(), dtype=float)
                        energy_h = float('nan')
                        last_err = None
                        self.error_flag = False
                except Exception:
                    pass

            # Retry 2: geomeTRIC optimizer (optional)
            # - Do NOT retry with geomeTRIC for basis/level errors; let the caller fall back to another basis.
            # - Only retry if the 'geometric' python package is available.
            if last_err is not None:
                try:
                    from psi4.driver.qcdb.exceptions import BasisSetNotFound  # type: ignore
                    if isinstance(last_err, BasisSetNotFound):
                        raise last_err
                except Exception:
                    # If we can't import the exception type, fall back to string match.
                    if 'BasisSetNotFound' in str(last_err) or 'Unable to find a basis set' in str(last_err):
                        raise last_err

                try:
                    import importlib.util
                    if importlib.util.find_spec('geometric') is None:
                        raise ModuleNotFoundError('geometric')
                    utils.radon_print(f'OptKing failed ({last_err}). Retrying with engine=geometric...', level=2)
                    energy_h, coord, wfn_obj = _run_opt(engine='geometric')
                    if wfn:
                        self.wfn = wfn_obj
                    last_err = None
                except (TypeError, ModuleNotFoundError):
                    # engine kw not supported OR geometric not installed -> keep original error
                    last_err = last_err
                except BaseException as e4:
                    last_err = e4
# If still failed, raise a YadonPyError (level=3)
            if last_err is not None:
                self.error_flag = True
                utils.radon_print('Error termination of psi4 optimization. %s' % last_err, level=3)

        finally:
            self._fin_psi4()

        # Update RDKit conformer coordinates (best-effort)
        try:
            if coord is not None:
                for i, atom in enumerate(self.mol.GetAtoms()):
                    self.mol.GetConformer(int(self.confId)).SetAtomPosition(
                        i, Geom.Point3D(float(coord[i, 0]), float(coord[i, 1]), float(coord[i, 2]))
                    )
        except Exception:
            pass

        return float(energy_h) * const.au2kj, coord  # Hartree -> kJ/mol



    def scan(self, atoms, values=[], opt=True, ignore_conv_error=False, geom_iter=50, geom_conv='QCHEM', geom_algorithm='RFO', dynamic_level=0, **kwargs):
        """
        Psi4w.scan

        Scanning potential energy surface by Psi4

        Args:
            atoms: Array of index number of atoms in a scanning bond length, bond angle, or dihedral angle (list(int))

        Optional args:
            values: Array of bond length (angstrom), bond angle (degree), or dihedral angle (degree) values
                    to be calculated potential energies (list(float))
            opt: Perform optimization (boolean)
            ignore_conv_error: If optimization has an convergence error,
                                False: return np.nan, True: return energy without converging (boolean)

        Returns:
            energy (ndarray(float), kJ/mol)
            coord (ndarray(float), angstrom)
        """
        energies = np.array([])
        coords = []

        if dynamic_level == 0 and calc.find_liner_angle(self.mol) and _version_key(psi4.__version__) < _version_key('1.8'):
            utils.radon_print('Found a linear angle in the molecule. Psi4 optimization setting \'dynamic_level\' was changed to 2.')
            dynamic_level = 2
            geom_iter = int(2*geom_iter)

        opt_dict = {
            'GEOM_MAXITER': geom_iter,
            'G_CONVERGENCE': geom_conv,
            'STEP_TYPE': geom_algorithm,
            'OPTKING__ENSURE_BT_CONVERGENCE': True,
            'DYNAMIC_LEVEL': dynamic_level,
#            'PRINT_OPT_PARAMS': True,
        }

        if len(atoms) == 2:
            opt_dict['OPTKING__FROZEN_DISTANCE'] = '%i %i' % (atoms[0]+1, atoms[1]+1)
            scan_type = 'bond length'
        elif len(atoms) == 3:
            opt_dict['OPTKING__FROZEN_BEND'] = '%i %i %i' % (atoms[0]+1, atoms[1]+1, atoms[2]+1)
            scan_type = 'bond angle'
        elif len(atoms) == 4:
            opt_dict['OPTKING__FROZEN_DIHEDRAL'] = '%i %i %i %i' % (atoms[0]+1, atoms[1]+1, atoms[2]+1, atoms[3]+1)
            scan_type = 'dihedral angle'
        else:
            utils.radon_print('Illegal length of array for input atoms. (2, 3, or 4)', level=3)

        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 scan (%s) is running...' % scan_type, level=1)

        for v in values:
            log_name = None
            conf = self.mol.GetConformer(self.confId)

            if len(atoms) == 2:
                Chem.rdMolTransforms.SetBondLength(conf, atoms[0], atoms[1], float(v))
                log_name = './%s_psi4_scan%i-%i_%f.log' % (self.name, atoms[0], atoms[1], float(v))
            elif len(atoms) == 3:
                Chem.rdMolTransforms.SetAngleDeg(conf, atoms[0], atoms[1], atoms[2], float(v))
                log_name = './%s_psi4_scan%i-%i-%i_%i.log' % (self.name, atoms[0], atoms[1], atoms[2], int(v))
            elif len(atoms) == 4:
                Chem.rdMolTransforms.SetDihedralDeg(conf, atoms[0], atoms[1], atoms[2], atoms[3], float(v))
                log_name = './%s_psi4_scan%i-%i-%i-%i_%i.log' % (self.name, atoms[0], atoms[1], atoms[2], atoms[3], int(v))
            
            pmol = self._init_psi4('symmetry c1', output=log_name)
            psi4.set_options(opt_dict)

            try:
                if opt:
                    utils.radon_print('Psi4 optimization (%s = %f) is running...' % (scan_type, float(v)), level=1)
                    dt3 = datetime.datetime.now()
                    energy = psi4.optimize(self.method, molecule=pmol, return_wfn=False, **kwargs)
                    dt4 = datetime.datetime.now()
                    utils.radon_print('Normal termination of psi4 optimization. Elapsed time = %s' % str(dt4-dt3), level=1)
                    coord = pmol.geometry().to_array() * const.bohr2ang
                else:
                    energy = psi4.energy(self.method, molecule=pmol, return_wfn=False, **kwargs)
                    coord = pmol.geometry().to_array() * const.bohr2ang

            except psi4.OptimizationConvergenceError as e:
                utils.radon_print('Psi4 optimization convergence error. %s' % e, level=2)
                if ignore_conv_error:
                    energy = e.wfn.energy()
                else:
                    energy = np.nan
                coord = np.array(e.wfn.molecule().geometry()) * const.bohr2ang
                self.error_flag = True

            except psi4.SCFConvergenceError as e:
                utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
                if ignore_conv_error:
                    energy = e.wfn.energy()
                else:
                    energy = np.nan
                coord = np.array(e.wfn.molecule().geometry()) * const.bohr2ang
                self.error_flag = True

            except BaseException as e:
                self._fin_psi4()
                self.error_flag = True
                utils.radon_print('Error termination of psi4 optimization. %s' % e, level=3)

            energies = np.append(energies, energy)
            coords.append(coord)
            if opt:
                conf = Chem.rdchem.Conformer(self.mol.GetNumAtoms())
                conf.Set3D(True)
                for i in range(self.mol.GetNumAtoms()):
                    self.mol.GetConformer(int(self.confId)).SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))
                    conf.SetAtomPosition(i, Geom.Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))
                self.mol.AddConformer(conf, assignId=True)

            self._fin_psi4()

        dt2 = datetime.datetime.now()
        utils.radon_print('Normal termination of psi4 scan. Elapsed time = %s' % str(dt2-dt1), level=1)

        return energies*const.au2kj, np.array(coords) # Hartree -> kJ/mol


    def force(self, wfn=True, **kwargs):
        """
        Psi4w.force

        Force calculation by Psi4

        Optional args:
            wfn: Return the wfn object of Psi4 (boolean)

        Returns:
            force (float, kJ/(mol angstrom))
        """

        pmol = self._init_psi4(output='./%s_psi4_force.log' % self.name)
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 force calculation is running...', level=1)

        try:
            if wfn:
                grad, self.wfn = psi4.gradient(self.method, molecule=pmol, return_wfn=True, **kwargs)
            else:
                grad = psi4.gradient(self.method, molecule=pmol, return_wfn=False, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 force calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            grad = e.wfn.gradient()
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 force calculation. %s' % e, level=3)

        self._fin_psi4()

        return grad.to_array()*const.au2kj/const.bohr2ang # Hartree/bohr -> kJ/(mol angstrom)


    def frequency(self, wfn=True, **kwargs):
        """
        Psi4w.frequency

        Frequency calculation by Psi4

        Optional args:
            wfn: Return the wfn object of Psi4 (boolean)

        Returns:
            energy (float, kJ/mol)
        """

        pmol = self._init_psi4(output='./%s_psi4_freq.log' % self.name)
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 frequency calculation is running...')

        try:
            if wfn:
                energy, self.wfn = psi4.frequency(self.method, molecule=pmol, return_wfn=True, **kwargs)
            else:
                energy = psi4.frequency(self.method, molecule=pmol, return_wfn=False, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 frequency calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            energy = e.wfn.energy()
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 frequency calculation. %s' % e, level=3)

        self._fin_psi4()

        return energy*const.au2kj # Hartree -> kJ/mol


    def hessian(self, wfn=True, **kwargs):
        """
        Psi4w.hessian

        Hessian calculation by Psi4

        Optional args:
            wfn: Return the wfn object of Psi4 (boolean)

        Returns:
            hessian (float, kJ/(mol angstrom**2))
        """

        pmol = self._init_psi4(output='./%s_psi4_hessian.log' % self.name)
        dt1 = datetime.datetime.now()
        try:
            _smi = Chem.MolToSmiles(self.mol)
        except Exception:
            _smi = '?'
        utils.radon_print(
            f"Psi4 Hessian is running... name={self.name} charge={self.charge} mult={self.multiplicity} "
            f"method={self.method} basis={self.basis} smiles={_smi}",
            level=1,
        )

        try:
            if wfn:
                hessian, self.wfn = psi4.hessian(self.method, molecule=pmol, return_wfn=True, **kwargs)
            else:
                hessian = psi4.hessian(self.method, molecule=pmol, return_wfn=False, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 hessian calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            hessian = e.wfn.hessian()
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 hessian calculation. %s' % e, level=3)

        self._fin_psi4()

        return hessian.to_array()*const.au2kj/const.bohr2ang/const.bohr2ang # Hartree/bohr^2 -> kJ/(mol angstrom^2)


    def tddft(self, n_state=6, p_state=None, triplet='NONE', tda=False, tdscf_maxiter=60, **kwargs):
        """
        Psi4w.tddft

        TD-DFT calculation by Psi4

        Optional args:
            wfn: Store the wfn object of Psi4 (boolean)
            n_state: Number of states (int). If n_state < 0, all excitation states are calculated.
            p_state: Number of states, which is determined by [Num. of all excitation states] * p_state (float, 0.0 < p_state <= 1.0).
                     p_state is given priority over n_state.
            triplet: NONE, ALSO, or ONLY
            tda: Run with Tamm-Dancoff approximation (TDA), uses random-phase approximation (RPA) when false (boolean)
            tdscf_maxiter: Maximum number of TDSCF solver iterations (int)

        Returns:
            TD-DFT result
        """
        if _version_key(psi4.__version__) < _version_key('1.3.100'):
            utils.radon_print('TD-DFT calclation is not implemented in Psi4 of this version (%s).' % str(psi4.__version__), level=3)
            return []

        pmol = self._init_psi4(output='./%s_psi4_tddft.log' % self.name)
        psi4.set_options({
            'wcombine': False,
            'save_jk': True,
            'TDSCF_TRIPLETS': triplet,
            'TDSCF_TDA': tda,
            'TDSCF_MAXITER': tdscf_maxiter
            })
        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 TD-DFT calculation is running...', level=1)

        try:
            energy, self.wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
            max_n_states = int((self.wfn.nmo() - self.wfn.nalpha()) * self.wfn.nalpha())
            if p_state is not None:
                if 0.0 < p_state <= 1.0:
                    n_state = int(max_n_states * p_state)
                    utils.radon_print('n_state of Psi4 TD-DFT calculation set to %i.' % n_state, level=1)
                else:
                    utils.radon_print('p_state=%f of Psi4 TD-DFT calculation is out of range (0.0 < p_state <= 1.0).' % float(p_state), level=3)
            elif n_state > max_n_states or n_state < 0:
                utils.radon_print('n_state of Psi4 TD-DFT calculation set to %i.' % max_n_states, level=1)
                n_state = max_n_states
            res = psi4.procrouting.response.scf_response.tdscf_excitations(self.wfn, states=n_state)

            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 TD-DFT calculation. Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            res = []
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 TD-DFT calculation. %s' % e, level=3)

        self._fin_psi4()

        return res


    def resp(self, **kwargs):
        """
        Psi4w.resp

        RESP charge calculation by PsiRESP/Psi4.

        Returns:
            RESP charge (float, array)
        """
        import json
        from .psiresp_wrapper import run_psiresp_fit

        polyelectrolyte_mode = bool(kwargs.pop("polyelectrolyte_mode", False))
        polyelectrolyte_detection = str(kwargs.pop("polyelectrolyte_detection", "auto") or "auto")

        try:
            result = run_psiresp_fit(
                self.mol,
                fit_kind="RESP",
                method=str(self.method),
                basis=str(self.basis),
                total_charge=int(self.charge),
                total_multiplicity=int(self.multiplicity),
                work_dir=self.work_dir,
                name=str(self.name),
                polyelectrolyte_mode=polyelectrolyte_mode,
                polyelectrolyte_detection=polyelectrolyte_detection,
                ncores=int(self.num_threads),
                memory_mib=float(self.memory),
            )
            resp_q = np.asarray(result["resp"], dtype=float)
            esp_q = np.asarray(result["esp"], dtype=float)
            meta = result.get("constraint_meta")
        except BaseException as e:
            self.error_flag = True
            utils.radon_print(f"Error termination of PsiRESP charge calculation. {e}", level=3)
            nan_arr = np.asarray([np.nan for _ in range(self.mol.GetNumAtoms())], dtype=float)
            for atom in self.mol.GetAtoms():
                atom.SetDoubleProp("ESP", float("nan"))
                atom.SetDoubleProp("RESP", float("nan"))
            return nan_arr

        for i, atom in enumerate(self.mol.GetAtoms()):
            atom.SetDoubleProp("ESP", float(esp_q[i]))
            atom.SetDoubleProp("RESP", float(resp_q[i]))
        try:
            if meta:
                self.mol.SetProp("_yadonpy_psiresp_constraints", json.dumps(meta, ensure_ascii=False))
                summary = meta.get("summary") if isinstance(meta, dict) else None
                constraints = meta.get("constraints") if isinstance(meta, dict) else None
                if isinstance(summary, dict):
                    self.mol.SetProp("_yadonpy_polyelectrolyte_summary_json", json.dumps(summary, ensure_ascii=False))
                    if isinstance(summary.get("groups"), list):
                        self.mol.SetProp("_yadonpy_charge_groups_json", json.dumps(summary.get("groups"), ensure_ascii=False))
                if isinstance(constraints, dict):
                    self.mol.SetProp("_yadonpy_resp_constraints_json", json.dumps(constraints, ensure_ascii=False))
        except Exception:
            pass
        return resp_q


    def mulliken_charge(self, recalc=False, **kwargs):
        """
        Psi4w.mulliken_charge

        Mulliken charge calculation by Psi4

        Optional args:
            recalc: Recalculation of wavefunction (boolean)

        Returns:
            Mulliken charge (float, ndarray)
        """

        if self.wfn is None or recalc:
            pmol = self._init_psi4(output='./%s_psi4.log' % self.name)
            energy, self.wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
            self._fin_psi4()

        psi4.oeprop(self.wfn, 'MULLIKEN_CHARGES')
        mulliken = self.wfn.atomic_point_charges().np

        for i, atom in enumerate(self.mol.GetAtoms()):
            atom.SetDoubleProp('MullikenCharge', mulliken[i])

        return mulliken


    def lowdin_charge(self, recalc=False, **kwargs):
        """
        psi4w.lowdin_charge

        Lowdin charge calculation by Psi4

        Optional args:
            recalc: Recalculation of wavefunction (boolean)

        Returns:
            Lowdin charge (float, ndarray)
        """

        if self.wfn is None or recalc:
            pmol = self._init_psi4(output='./%s_psi4.log' % self.name)
            energy, self.wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
            self._fin_psi4()

        psi4.oeprop(self.wfn, 'LOWDIN_CHARGES')
        lowdin = self.wfn.atomic_point_charges().np

        for i, atom in enumerate(self.mol.GetAtoms()):
            atom.SetDoubleProp('LowdinCharge', lowdin[i])

        return lowdin


    def polar(self, eps=1e-4, mp=0, **kwargs):
        """
        psi4w.polar

        Computation of dipole polarizability by finite field

        Optional args:
            eps: Epsilon of finite field
            mp: Number of multiprocessing

        Return:
            Dipole polarizability (float, angstrom^3)
            Polarizability tensor (ndarray, angstrom^3)
        """
        self.error_flag = False

        # Finit different of d(mu)/dE
        d_mu = np.zeros((3, 3))
        p_mu = np.zeros((2, 3, 3))

        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 polarizability calculation (finite field) is running...', level=1)

        # Dipole moment calculation by perturbed SCF
        # Multiprocessing
        if mp > 0 or const.mpi4py_avail:
            args = []
            utils.picklable(self.mol)
            wfn_copy = self.wfn
            self.wfn = None
            cc2wfn_copy = self.cc2wfn
            self.cc2wfn = None

            c = utils.picklable_const()
            for e in [eps, -eps]:
                for ax in ['x', 'y', 'z']:
                    args.append([e, ax, self, c])

            # mpi4py
            if const.mpi4py_avail:
                utils.radon_print('Parallel method: mpi4py')
                with MPIPoolExecutor(max_workers=mp) as executor:
                    results = executor.map(_polar_mp_worker, args)
                    for i, res in enumerate(results):
                        p_mu[(i // 3), (i % 3)] = res[0]
                        if res[1]: self.error_flag = True

            # concurrent.futures
            else:
                utils.radon_print('Parallel method: concurrent.futures.ProcessPoolExecutor')
                if mp == 1:
                    for i, arg in enumerate(args):
                        with confu.ProcessPoolExecutor(max_workers=1, mp_context=MP.get_context('spawn')) as executor:
                            results = executor.map(_polar_mp_worker, [arg])
                            for res in results:
                                p_mu[(i // 3), (i % 3)] = res[0]
                                if res[1]: self.error_flag = True
                else:
                    with confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn')) as executor:
                        results = executor.map(_polar_mp_worker, args)
                        for i, res in enumerate(results):
                            p_mu[(i // 3), (i % 3)] = res[0]
                            if res[1]: self.error_flag = True

            utils.restore_picklable(self.mol)
            self.wfn = wfn_copy
            self.cc2wfn = cc2wfn_copy

        # Sequential
        else:
            pmol = self._init_psi4('symmetry c1')
            psi4.set_options({
                'perturb_h': True,
                'perturb_with': 'dipole'
                })
            for i, e in enumerate([eps, -eps]):
                for j, ax in enumerate(['x', 'y', 'z']):
                    try:
                        psi4.core.set_output_file('./%s_psi4_polar_%s%i.log' % (self.name, ax, i), False)
                        divec = [0.0, 0.0, 0.0]
                        divec[j] = e
                        psi4.set_options({'perturb_dipole': divec})
                        energy_x, wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
                        psi4.oeprop(wfn, 'DIPOLE')
                        if _version_key(psi4.__version__) < _version_key('1.3.100'):
                            p_mu[i, j, 0] = psi4.variable('SCF DIPOLE X') / const.au2debye
                            p_mu[i, j, 1] = psi4.variable('SCF DIPOLE Y') / const.au2debye
                            p_mu[i, j, 2] = psi4.variable('SCF DIPOLE Z') / const.au2debye
                        else:
                            p_mu[i, j] = np.array(psi4.variable('SCF DIPOLE'))

                    except psi4.SCFConvergenceError as e:
                        utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
                        p_mu[i, j] = np.array([np.nan, np.nan, np.nan])
                        self.error_flag = True

                    except BaseException as e:
                        self._fin_psi4()
                        self.error_flag = True
                        utils.radon_print('Error termination of psi4 polarizability calculation (finite field). %s' % e, level=3)
            self._fin_psi4()

        a_conv = 1.648777e-41    # a.u. -> C^2 m^2 J^-1
        pv = (a_conv*const.m2ang**3)/(4*np.pi*const.eps0)    # C^2 m^2 J^-1 -> angstrom^3 (polarizability volume)

        d_mu = -(p_mu[0] - p_mu[1]) / (2*eps) * pv
        alpha = np.mean(np.diag(d_mu))

        if self.error_flag:
            utils.radon_print('Psi4 polarizability calculation (finite field) failure.', level=2)
        else:
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 polarizability calculation (finite field). Elapsed time = %s' % str(dt2-dt1), level=1)

        return alpha, d_mu


    def cc2_polar(self, omega=[], unit='nm', method='cc2', **kwargs):
        """
        psi4w.cc2_polar

        Computation of dipole polarizability by coupled cluster linear response calculation

        Optional args:
            omega: Computation of dynamic polarizability at the wave lengths (float, list)
            unit: Unit of omega (str; nm, au, ev, or hz)
            method: Coupled cluster method (cc2 | ccsd)
            cache_level: 

        Returns:
            Static or dynamic dipole polarizability (ndarray, angstrom^3)
        """
            
        pmol = self._init_psi4(output='./%s_psi4_cc2polar.log' % self.name)
        if len(omega) > 0:
            omega.append(unit)
            psi4.set_options({'omega': omega})

        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 polarizability calculation (CC linear response) is running...', level=1)

        try:
            energy, self.cc2wfn = psi4.properties(method, properties=['polarizability'], molecule=pmol, return_wfn=True, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 polarizability calculation (CC linear response). Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            self.error_flag = True
            return [np.nan]

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 polarizability calculation (CC linear response). %s' % e, level=3)

        self._fin_psi4()

        a_conv = 1.648777e-41 # a.u. -> C^2 m^2 J^-1
        pv = (a_conv*const.m2ang**3)/(4*np.pi*const.eps0) # C^2 m^2 J^-1 -> angstrom^3 (polarizability volume)

        alpha = []
        if len(omega) == 0:
            if method == 'cc2' or method == 'CC2':
                alpha.append( psi4.variable('CC2 DIPOLE POLARIZABILITY @ INF NM') * pv )
            elif method == 'ccsd' or method == 'CCSD':
                alpha.append( psi4.variable('CCSD DIPOLE POLARIZABILITY @ INF NM') * pv )
        elif len(omega) > 0:
            for i in range(len(omega)-1):
                if unit == 'NM' or unit == 'nm':
                    lamda = round(omega[i])
                elif unit == 'AU' or unit == 'au':
                    lamda = round( (const.h * const.c) / (omega[i] * const.au2ev * const.e) * 1e9 )
                elif unit == 'EV' or unit == 'ev':
                    lamda = round( (const.h * const.c) / (omega[i] * const.au2kj) * 1e6 )
                elif unit == 'HZ' or unit == 'hz':
                    lamda = round( const.c / omega[i] * 1e9 )
                else:
                    utils.radon_print('Illeagal input of unit = %s in cc2_polar.' % str(unit), level=3)

                if method == 'cc2' or method == 'CC2':
                    alpha.append( psi4.variable('CC2 DIPOLE POLARIZABILITY @ %iNM' % lamda) * pv )
                elif method == 'ccsd' or method == 'CCSD':
                    alpha.append( psi4.variable('CCSD DIPOLE POLARIZABILITY @ %iNM' % lamda) * pv )

        return np.array(alpha)


    def cphf_polar(self, **kwargs):
        """
        psi4w.cphf_polar

        Computation of dipole polarizability by linear response CPHF/CPKS calculation

        Returns:
            Static dipole polarizability (ndarray, angstrom^3)
        """            
        pmol = self._init_psi4(output='./%s_psi4_cphfpolar.log' % self.name)

        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 polarizability calculation (CPHF/CPKS) is running...', level=1)

        try:
            energy, self.wfn = psi4.properties(self.method, properties=['DIPOLE_POLARIZABILITIES'], molecule=pmol, return_wfn=True, **kwargs)
            dt2 = datetime.datetime.now()
            utils.radon_print('Normal termination of psi4 polarizability calculation (CPHF/CPKS). Elapsed time = %s' % str(dt2-dt1), level=1)

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            pol = np.full((6), np.nan)
            self.error_flag = True

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 polarizability calculation (CPHF/CPKS). %s' % e, level=3)

        self._fin_psi4()

        a_conv = 1.648777e-41 # a.u. -> C^2 m^2 J^-1
        pv = (a_conv*const.m2ang**3)/(4*np.pi*const.eps0) # C^2 m^2 J^-1 -> angstrom^3 (polarizability volume)

        pol = np.array([psi4.variable('DIPOLE POLARIZABILITY %s' % ax) * pv for ax in ['XX', 'YY', 'ZZ', 'XY', 'XZ', 'YZ']])
        alpha = np.mean(pol[:3])
        tensor = np.array([[pol[0], pol[3], pol[4]], [pol[3], pol[1], pol[5]], [pol[4], pol[5], pol[2]]])

        return alpha, tensor


    # Experimental
    def cphf_hyperpolar(self, eps=1e-4, mp=0, **kwargs):
        """
        psi4w.polar

        Computation of first dipole hyperpolarizability by finite field and CPHF/CPKS hybrid

        Optional args:
            eps: Epsilon of finite field
            mp: Number of multiprocessing

        Return:
            First dipole hyperpolarizability (float, a.u.)
            First hyperpolarizability tensor (ndarray, a.u.)
        """
        self.error_flag = False
        pmol = self._init_psi4('symmetry c1')

        dt1 = datetime.datetime.now()
        utils.radon_print('Psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field) is running...', level=1)

        # Calculate Non-perturbed dipole moment
        np_mu = np.zeros((3))
        try:
            psi4.core.set_output_file('./%s_psi4_hyperpolar.log' % (self.name), False)
            energy_x, wfn = psi4.energy(self.method, molecule=pmol, return_wfn=True, **kwargs)
            psi4.oeprop(wfn, 'DIPOLE')
            if _version_key(psi4.__version__) < _version_key('1.3.100'):
                np_mu[0] = psi4.variable('SCF DIPOLE X') / const.au2debye
                np_mu[1] = psi4.variable('SCF DIPOLE Y') / const.au2debye
                np_mu[2] = psi4.variable('SCF DIPOLE Z') / const.au2debye
            else:
                np_mu = np.array(psi4.variable('SCF DIPOLE'))

        except psi4.SCFConvergenceError as e:
            utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
            self.error_flag = True
            np_mu = np.full((3), np.nan)

        except BaseException as e:
            self._fin_psi4()
            self.error_flag = True
            utils.radon_print('Error termination of psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field). %s' % e, level=3)

        self._fin_psi4()


        # Finit different of polarizability d(alpha)/dE
        tensor = np.zeros((3, 3, 3))
        p_alpha = np.zeros((2, 3, 3, 3))

        # Polarizability calculation by perturbed SCF
        # Multiprocessing
        if mp > 0 or const.mpi4py_avail:
            args = []
            utils.picklable(self.mol)
            wfn_copy = self.wfn
            self.wfn = None
            cc2wfn_copy = self.cc2wfn
            self.cc2wfn = None

            c = utils.picklable_const()
            for e in [eps, -eps]:
                for ax in ['x', 'y', 'z']:
                    args.append([e, ax, self, c])

            # mpi4py
            if const.mpi4py_avail:
                utils.radon_print('Parallel method: mpi4py')
                with MPIPoolExecutor(max_workers=mp) as executor:
                    results = executor.map(_cphf_hyperpolar_mp_worker, args)
                    for i, res in enumerate(results):
                        p_alpha[(i // 3), (i % 3)] = res[0]
                        if res[1]: self.error_flag = True

            # concurrent.futures
            else:
                utils.radon_print('Parallel method: concurrent.futures.ProcessPoolExecutor')
                with confu.ProcessPoolExecutor(max_workers=mp, mp_context=MP.get_context('spawn')) as executor:
                    results = executor.map(_cphf_hyperpolar_mp_worker, args)
                    for i, res in enumerate(results):
                        p_alpha[(i // 3), (i % 3)] = res[0]
                        if res[1]: self.error_flag = True

            utils.restore_picklable(self.mol)
            self.wfn = wfn_copy
            self.cc2wfn = cc2wfn_copy

        # Sequential
        else:
            pmol = self._init_psi4('symmetry c1')
            psi4.set_options({
                'perturb_h': True,
                'perturb_with': 'dipole'
                })

            for i, e in enumerate([eps, -eps]):
                for j, ax in enumerate(['x', 'y', 'z']):
                    try:
                        psi4.core.set_output_file('./%s_psi4_hyperpolar_%s%i.log' % (self.name, ax, i), False)
                        divec = [0.0, 0.0, 0.0]
                        divec[j] = e
                        psi4.set_options({'perturb_dipole': divec})
                        energy_x, wfn = psi4.properties(self.method, properties=['DIPOLE_POLARIZABILITIES'], molecule=pmol, return_wfn=True)
                        p_alpha[i, j, 0, 0] = psi4.variable('DIPOLE POLARIZABILITY XX')
                        p_alpha[i, j, 1, 1] = psi4.variable('DIPOLE POLARIZABILITY YY')
                        p_alpha[i, j, 2, 2] = psi4.variable('DIPOLE POLARIZABILITY ZZ')
                        p_alpha[i, j, 0, 1] = p_alpha[i, j, 1, 0] = psi4.variable('DIPOLE POLARIZABILITY XY')
                        p_alpha[i, j, 0, 2] = p_alpha[i, j, 2, 0] = psi4.variable('DIPOLE POLARIZABILITY XZ')
                        p_alpha[i, j, 1, 2] = p_alpha[i, j, 2, 1] = psi4.variable('DIPOLE POLARIZABILITY YZ')

                    except psi4.SCFConvergenceError as e:
                        utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
                        p_alpha[i, j] = np.full((3,3), np.nan)
                        self.error_flag = True

                    except BaseException as e:
                        self._fin_psi4()
                        self.error_flag = True
                        utils.radon_print('Error termination of psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field). %s' % e, level=3)
            self._fin_psi4()

        tensor = -(p_alpha[0] - p_alpha[1]) / (2*eps)
        b_x = tensor[0,0,0] + tensor[0,1,1] + tensor[0,2,2] + tensor[1,0,1] + tensor[2,0,2] + tensor[1,1,0] + tensor[2,2,0]
        b_y = tensor[1,1,1] + tensor[1,0,0] + tensor[1,2,2] + tensor[0,1,0] + tensor[2,1,2] + tensor[0,0,1] + tensor[2,2,1]
        b_z = tensor[2,2,2] + tensor[2,0,0] + tensor[2,1,1] + tensor[0,2,0] + tensor[1,2,1] + tensor[0,0,2] + tensor[1,1,2]
        beta = np.sqrt(b_x**2 + b_y**2 + b_z**2)

        if self.error_flag:
            utils.radon_print('Psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field) failure.', level=2)
        else:
            dt2 = datetime.datetime.now()
            utils.radon_print(
                'Normal termination of psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field). Elapsed time = %s' % str(dt2-dt1),
                level=1)

        return beta, tensor


    @property
    def homo(self):
        """
        Psi4w.homo

        Returns:
            HOMO (float, eV)
        """
        if self.wfn is None: return np.nan
        else: return self.wfn.epsilon_a_subset('AO', 'ALL').np[self.wfn.nalpha() - 1] * const.au2ev # Hartree -> eV


    @property
    def lumo(self):
        """
        Psi4w.lumo

        Returns:
            LUMO (float, eV)
        """
        if self.wfn is None: return np.nan
        return self.wfn.epsilon_a_subset('AO', 'ALL').np[self.wfn.nalpha()] * const.au2ev # Hartree -> eV


    @property
    def dipole(self):
        """
        Psi4w.dipole

        Returns:
            dipole vector (float, ndarray, debye)
        """
        if self.wfn is None: return np.nan
        psi4.oeprop(self.wfn, 'DIPOLE')
        if _version_key(psi4.__version__) < _version_key('1.3.100'):
            x = psi4.variable('SCF DIPOLE X')
            y = psi4.variable('SCF DIPOLE Y')
            z = psi4.variable('SCF DIPOLE Z')
            mu = np.array([x, y, z])
        else:
            mu = np.array(psi4.variable('SCF DIPOLE')) * const.au2debye

        return mu


    @property
    def quadrupole(self):
        """
        Psi4w.quadrupole

        Returns:
            quadrupole xx, yy, zz, xy, xz, yz (float, ndarray, debye ang)
        """
        if self.wfn is None: return np.nan
        psi4.oeprop(self.wfn, 'QUADRUPOLE')
        if _version_key(psi4.__version__) < _version_key('1.3.100'):
            xx = psi4.variable('SCF QUADRUPOLE XX')
            yy = psi4.variable('SCF QUADRUPOLE YY')
            zz = psi4.variable('SCF QUADRUPOLE ZZ')
            xy = psi4.variable('SCF QUADRUPOLE XY')
            xz = psi4.variable('SCF QUADRUPOLE XZ')
            yz = psi4.variable('SCF QUADRUPOLE YZ')
            quad = np.array([xx, yy, zz, xy, xz, yz])
        else:
            quad = psi4.variable('SCF QUADRUPOLE')

        return quad 


    @property
    def wiberg_bond_index(self):
        """
        Psi4w.wiberg_bond_index

        Returns:
            wiberg bond index (float, ndarray)
        """
        if self.wfn is None: return np.nan
        psi4.oeprop(self.wfn, 'WIBERG_LOWDIN_INDICES')
        return None


    @property
    def mayer_bond_index(self):
        """
        Psi4w.mayer_bond_index

        Returns:
            mayer bond index (float, ndarray)
        """
        if self.wfn is None: return np.nan
        psi4.oeprop(self.wfn, 'MAYER_INDICES')
        return None
        

    @property
    def natural_orbital_occ(self):
        """
        Psi4w.mayer_bond_index

        Returns:
            mayer bond index (float, ndarray)
        """
        if self.wfn is None: return np.nan
        psi4.oeprop(self.wfn, 'NO_OCCUPATIONS')
        no = self.wfn.no_occupations()
        return np.array(no)
        

    @property
    def total_energy(self):
        """
        Psi4w.total_energy

        Returns:
            DFT total energy (float)
        """
        return float(psi4.variable('DFT TOTAL ENERGY'))


    @property
    def scf_energy(self):
        """
        Psi4w.scf_energy

        Returns:
            DFT scf energy (float)
        """
        return float(psi4.variable('SCF TOTAL ENERGY'))


    @property
    def xc_energy(self):
        """
        Psi4w.xc_energy

        Returns:
            DFT xc energy (float)
        """
        return float(psi4.variable('DFT XC ENERGY'))
        

    @property
    def dispersion_energy(self):
        """
        Psi4w.dispersion_energy

        Returns:
            Dispersion correction energy (float)
        """
        return float(psi4.variable('DISPERSION CORRECTION ENERGY'))
        

    @property
    def dh_energy(self):
        """
        Psi4w.dh_energy

        Returns:
            Double hybrid correction energy (float)
        """
        return float(psi4.variable('DOUBLE-HYBRID CORRECTION ENERGY'))
        

    @property
    def NN_energy(self):
        """
        Psi4w.NN_energy

        Returns:
            Nuclear repulsion energy (float)
        """
        return float(psi4.variable('NUCLEAR REPULSION ENERGY'))
        

    @property
    def e1_energy(self):
        """
        Psi4w.e1_energy

        Returns:
            One-electron energy (float)
        """
        return float(psi4.variable('ONE-ELECTRON ENERGY'))
        

    @property
    def e2_energy(self):
        """
        Psi4w.e2_energy

        Returns:
            Two-electron energy (float)
        """
        return float(psi4.variable('TWO-ELECTRON ENERGY'))
        

    @property
    def cc2_energy(self):
        """
        Psi4w.cc2_energy

        Returns:
            CC2 total energy (float)
        """
        return float(psi4.variable('CC2 TOTAL ENERGY'))
        

    @property
    def cc2_corr_energy(self):
        """
        Psi4w.cc2_corr_energy

        Returns:
            CC2 correlation energy (float)
        """
        return float(psi4.variable('CC2 CORRELATION ENERGY'))
        

    @property
    def ccsd_energy(self):
        """
        Psi4w.ccsd_energy

        Returns:
            CCSD total energy (float)
        """
        return float(psi4.variable('CCSD TOTAL ENERGY'))
        

    @property
    def ccsd_corr_energy(self):
        """
        Psi4w.ccsd_corr_energy

        Returns:
            CCSD correlation energy (float)
        """
        return float(psi4.variable('CCSD CORRELATION ENERGY'))
        

    @property
    def ccsd_t_energy(self):
        """
        Psi4w.ccsd_t_energy

        Returns:
            CCSD(T) total energy (float)
        """
        return float(psi4.variable('CCSD(T) TOTAL ENERGY'))
        

    @property
    def ccsd_t_corr_energy(self):
        """
        Psi4w.ccsd_t_corr_energy

        Returns:
            CCSD(T) correlation energy (float)
        """
        return float(psi4.variable('CCSD(T) CORRELATION ENERGY'))
        

def _polar_mp_worker(args):
    eps, ax, psi4obj, c = args
    utils.restore_const(c)
    
    i = 0 if eps > 0 else 1
    j = 0 if ax == 'x' else 1 if ax == 'y' else 2 if ax == 'z' else np.nan
    error_flag = False

    utils.radon_print('Worker process %s%i start on %s. PID: %i' % (ax, i, socket.gethostname(), os.getpid()))

    utils.restore_picklable(psi4obj.mol)
    pmol = psi4obj._init_psi4('symmetry c1', output='./%s_psi4_polar_%s%i.log' % (psi4obj.name, ax, i))
    divec = [0.0, 0.0, 0.0]
    divec[j] = eps
    psi4.set_options({
        'perturb_h': True,
        'perturb_with': 'dipole',
        'perturb_dipole': divec
        })

    dipole = np.zeros((3))

    try:
        energy_x, wfn = psi4.energy(psi4obj.method, molecule=pmol, return_wfn=True)
        psi4.oeprop(wfn, 'DIPOLE')
        if _version_key(psi4.__version__) < _version_key('1.3.100'):
            dipole[0] = psi4.variable('SCF DIPOLE X') / const.au2debye
            dipole[1] = psi4.variable('SCF DIPOLE Y') / const.au2debye
            dipole[2] = psi4.variable('SCF DIPOLE Z') / const.au2debye
        else:
            dipole = np.array(psi4.variable('SCF DIPOLE'))

    except psi4.SCFConvergenceError as e:
        utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
        dipole = np.full((3), np.nan)
        error_flag = True

    except BaseException as e:
        psi4obj._fin_psi4()
        error_flag = True
        utils.radon_print('Error termination of psi4 polarizability calculation (finite field). %s' % e, level=3)

    psi4obj._fin_psi4()

    return dipole, error_flag


def _cphf_hyperpolar_mp_worker(args):
    eps, ax, psi4obj, c = args
    utils.restore_const(c)

    i = 0 if eps > 0 else 1
    j = 0 if ax == 'x' else 1 if ax == 'y' else 2 if ax == 'z' else np.nan
    error_flag = False

    utils.radon_print('Worker process %s%i start on %s. PID: %i' % (ax, i, socket.gethostname(), os.getpid()))

    utils.restore_picklable(psi4obj.mol)
    pmol = psi4obj._init_psi4('symmetry c1', output='./%s_psi4_hyperpolar_%s%i.log' % (psi4obj.name, ax, i))
    divec = [0.0, 0.0, 0.0]
    divec[j] = eps
    psi4.set_options({
        'perturb_h': True,
        'perturb_with': 'dipole',
        'perturb_dipole': divec
        })

    alpha = np.zeros((3,3))

    try:
        energy_x, wfn = psi4.properties(psi4obj.method, properties=['DIPOLE_POLARIZABILITIES'], molecule=pmol, return_wfn=True)
        alpha[0, 0] = psi4.variable('DIPOLE POLARIZABILITY XX')
        alpha[1, 1] = psi4.variable('DIPOLE POLARIZABILITY YY')
        alpha[2, 2] = psi4.variable('DIPOLE POLARIZABILITY ZZ')
        alpha[0, 1] = alpha[1, 0] = psi4.variable('DIPOLE POLARIZABILITY XY')
        alpha[0, 2] = alpha[2, 0] = psi4.variable('DIPOLE POLARIZABILITY XZ')
        alpha[1, 2] = alpha[2, 1] = psi4.variable('DIPOLE POLARIZABILITY YZ')

    except psi4.SCFConvergenceError as e:
        utils.radon_print('Psi4 SCF convergence error. %s' % e, level=2)
        alpha = np.full((3,3), np.nan)
        error_flag = True

    except BaseException as e:
        psi4obj._fin_psi4()
        error_flag = True
        utils.radon_print('Error termination of psi4 first hyperpolarizability calculation (CPHF/CPKS & finite field). %s' % e, level=3)

    psi4obj._fin_psi4()

    return alpha, error_flag


# Override function for qcengine.config.get_global
from typing import Any, Dict, Optional, Union
_global_values = None
def _override_get_global(key: Optional[str] = None) -> Union[str, Dict[str, Any]]:
    import psutil
    import getpass
    import cpuinfo

    # TODO (wardlt): Implement a means of getting CPU information from compute nodes on clusters for MPI tasks
    #  The QC code runs on a different node than the node running this Python function, which may have different info

    global _global_values
    if _global_values is None:
        _global_values = {}
        _global_values["hostname"] = socket.gethostname()
        _global_values["memory"] = round(psutil.virtual_memory().available / (1024 ** 3), 3)
        _global_values["username"] = getpass.getuser()

        # Work through VMs and logical cores.
        if hasattr(psutil.Process(), "cpu_affinity"):
            cpu_cnt = len(psutil.Process().cpu_affinity())

            # For mpi4py
            if const.mpi4py_avail and cpu_cnt == 1:
                cpu_cnt = psutil.cpu_count(logical=False)
                if cpu_cnt is None:
                   cpu_cnt = psutil.cpu_count(logical=True)

        else:
            cpu_cnt = psutil.cpu_count(logical=False)
            if cpu_cnt is None:
                cpu_cnt = psutil.cpu_count(logical=True)

        _global_values["ncores"] = cpu_cnt
        _global_values["nnodes"] = 1

        _global_values["cpuinfo"] = cpuinfo.get_cpu_info()
        try:
            _global_values["cpu_brand"] = _global_values["cpuinfo"]["brand_raw"]
        except KeyError:
            try:
                # Remove this if py-cpuinfo is pinned to >=6.0.0
                _global_values["cpu_brand"] = _global_values["cpuinfo"]["brand"]
            except KeyError:
                # Assuming Fugaku
                _global_values["cpu_brand"] = 'A64FX'

    if key is None:
        return _global_values.copy()
    else:
        return _global_values[key]
