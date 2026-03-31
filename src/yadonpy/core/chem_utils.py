"""Chemistry-focused utilities (RDKit helpers, geometry, charge scaling)."""

from __future__ import annotations
import re
import sys
from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import rdGeometry as Geom
from .logging_utils import radon_print

def star2h(smiles):
    smiles = smiles.replace('[*]', '[3H]')
    smiles = re.sub(r'\[([0-9]+)\*\]', lambda m: '[%iH]' % int(int(m.groups()[0])+2), smiles)
    smiles = smiles.replace('*', '[3H]')
    smiles = smiles.replace('[X]', '[65535H]')
    smiles = smiles.replace('[LP-]', '[65534H-]')
    return smiles


def h2star(smiles):
    smiles = smiles.replace('[65534H-]', '[LP-]')
    smiles = smiles.replace('[65535H]', '[X]')
    smiles = smiles.replace('[3H]', '*')
    smiles = re.sub(
        r'\[([0-9]+)H\]',
        lambda m: '[%i*]' % int(int(m.groups()[0])-2) if int(int(m.groups()[0])) >= 3 else '[%iH]' % int(int(m.groups()[0])),
        smiles)
    return smiles


def mol_from_smiles(smiles, coord=True, version=3, ez='E', chiral='S', stereochemistry_control=True, sanitize=True, name: str | None = None):

    # Convenience: allow simple monoatomic ions in the shorthand form "Li+", "Cl-".
    # RDKit requires brackets for charged atoms, e.g. "[Li+]". We normalize here
    # to avoid confusing parse errors in user scripts.
    try:
        _s0 = str(smiles).strip()
        if _s0 and (not _s0.startswith('[')) and re.fullmatch(r"[A-Z][a-z]?(?:\d+)?[+-](?:\d+)?", _s0):
            smiles = f"[{_s0}]"
    except Exception:
        pass

    # Keep the original SMILES for robust downstream matching (e.g., library lookup).
    # NOTE: this function converts '*' connection points into special hydrogens
    # (see star2h), so Chem.MolToSmiles(mol) may differ from user input.
    _yadonpy_input_smiles = str(smiles).strip()

    smi = star2h(smiles)
    l = re.findall(r'\[([0-9]+)H\]', smi)
    labels = [int(x) for x in l if int(x) >= 3]
    if len(labels) > 0:
        n_conn = smi.count('[%iH]' % min(labels))
    else:
        n_conn = 0

    # Conformer generation parameters.
    # RDKit version compatibility note: ETKDGv3 is not available in older builds
    # (e.g., 2020.03.x on some clusters). Fall back gracefully.
    if version == 3 and hasattr(AllChem, 'ETKDGv3'):
        etkdg = AllChem.ETKDGv3()
    elif version >= 2 and hasattr(AllChem, 'ETKDGv2'):
        etkdg = AllChem.ETKDGv2()
    else:
        etkdg = AllChem.ETKDG()
    etkdg.enforceChirality=True
    etkdg.useRandomCoords = False
    if hasattr(etkdg, 'maxIterations'):
        etkdg.maxIterations = 100
    elif hasattr(etkdg, 'maxAttempts'):
        etkdg.maxAttempts = 100
    else:
        radon_print('The installed RDKit version is not supported.', level=3)

    # Parse SMILES.
    # Hypervalent/ionic species (PF6-, BF4-, ClO4-, some metal complexes) can
    # fail RDKit's default sanitization (valence rules). A robust approach is:
    #   1) try normal parsing; if it fails,
    #   2) retry with sanitize=False and then do *selective* sanitization.
    mol = None
    try:
        if sanitize:
            mol = Chem.MolFromSmiles(smi)
        else:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
    except Exception:
        mol = None

    if mol is None and sanitize:
        # retry with sanitize disabled (RDKit cookbook/book recommend this for hypervalent P/I etc.)
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            # selective sanitization: skip PROPERTIES (includes valence/property derivation)
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                )
            except Exception:
                pass
        except Exception:
            mol = None

    if mol is None:
        radon_print('Cannot transform to RDKit Mol object from %s' % smiles, level=3)
        return None

    if sanitize:
        try:
            mol = Chem.AddHs(mol)
        except Exception:
            # For some unsanitized hypervalent/ionic species, AddHs may fail.
            # Keep the heavy-atom graph and continue.
            pass

    # Store original input SMILES on the molecule for later retrieval.
    # Use a yadonpy-specific key to avoid colliding with RadonPy props.
    try:
        mol.SetProp('_yadonpy_smiles', _yadonpy_input_smiles)
    except Exception:
        pass

    # Attach input SMILES (best-effort). This is used by yadonpy to match species
    # by SMILES even after packing/export.
    try:
        mol.SetProp('_yadonpy_input_smiles', _yadonpy_input_smiles)
    except Exception:
        pass

    if stereochemistry_control:
        ### cis/trans and chirality control
        Chem.AssignStereochemistry(mol)

        # Get polymer backbone
        backbone_atoms = []
        backbone_bonds = []
        backbone_dih = []

        if n_conn == 2:
            link_idx = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "H" and atom.GetIsotope() >= 3:
                    link_idx.append(atom.GetIdx())
            backbone_atoms = Chem.GetShortestPath(mol, link_idx[0], link_idx[1])

            for i in range(len(backbone_atoms)-1):
                bond = mol.GetBondBetweenAtoms(backbone_atoms[i], backbone_atoms[i+1])
                backbone_bonds.append(bond.GetIdx())
                if bond.GetBondTypeAsDouble() == 2 and str(bond.GetStereo()) == 'STEREONONE' and not bond.IsInRing():
                    backbone_dih.append((backbone_atoms[i-1], backbone_atoms[i], backbone_atoms[i+1], backbone_atoms[i+2]))

        # List of unspecified double bonds (except for bonds in polymer backbone and a ring structure)
        db_list = []
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() == 2 and str(bond.GetStereo()) == 'STEREONONE' and not bond.IsInRing():
                if n_conn == 2 and bond.GetIdx() in backbone_bonds:
                    continue
                else:
                    db_list.append(bond.GetIdx())

        # Enumerate stereo isomers
        opts = Chem.EnumerateStereoisomers.StereoEnumerationOptions(unique=True, tryEmbedding=True)
        isomers = tuple(Chem.EnumerateStereoisomers.EnumerateStereoisomers(mol, options=opts))

        if len(isomers) > 1:
            radon_print('%i candidates of stereoisomers were generated.' % len(isomers))
            chiral_num_max = 0
            
            for isomer in isomers:
                ez_flag = False
                chiral_flag = 0

                Chem.AssignStereochemistry(isomer)

                # Contorol unspecified double bonds (except for bonds in polymer backbone and a ring structure)
                ez_list = []
                for idx in db_list:
                    bond = isomer.GetBondWithIdx(idx)
                    if str(bond.GetStereo()) == 'STEREOANY' or str(bond.GetStereo()) == 'STEREONONE':
                        continue
                    elif ez == 'E' and (str(bond.GetStereo()) == 'STEREOE' or str(bond.GetStereo()) == 'STEREOTRANS'):
                        ez_list.append(True)
                    elif ez == 'Z' and (str(bond.GetStereo()) == 'STEREOZ' or str(bond.GetStereo()) == 'STEREOCIS'):
                        ez_list.append(True)
                    else:
                        ez_list.append(False)

                if len(ez_list) > 0:
                    ez_flag = np.all(np.array(ez_list))
                else:
                    ez_flag = True

                # Contorol unspecified chirality
                chiral_list = np.array(Chem.FindMolChiralCenters(isomer))
                if len(chiral_list) > 0:
                    chirality = chiral_list[:, 1]
                    chiral_num = np.count_nonzero(chirality == chiral)
                    if chiral_num == len(chiral_list):
                        chiral_num_max = chiral_num
                        chiral_flag = 2
                    elif chiral_num > chiral_num_max:
                        chiral_num_max = chiral_num
                        chiral_flag = 1
                else:
                    chiral_flag = 2

                if ez_flag and chiral_flag:
                    mol = isomer
                    if chiral_flag == 2:
                        break

    # Generate 3D coordinates
    # NOTE: RDKit (especially older versions like 2020.03.x) may fail to embed
    # some inorganic ions (e.g., PF6-) or highly charged species. yadonpy
    # requires *some* 3D coordinates for topology/artifact generation, so we
    # provide robust fallbacks.
    if coord:
        def _set_conf_from_coords(_mol, xyz):
            try:
                xyz = np.asarray(xyz, dtype=float)
                if xyz.ndim == 1:
                    xyz = xyz.reshape((-1, 3))
                if xyz.shape[0] != _mol.GetNumAtoms() or xyz.shape[1] != 3:
                    return False
                conf = Chem.Conformer(_mol.GetNumAtoms())
                for ai in range(_mol.GetNumAtoms()):
                    conf.SetAtomPosition(ai, Geom.Point3D(float(xyz[ai, 0]), float(xyz[ai, 1]), float(xyz[ai, 2])))
                _mol.RemoveAllConformers()
                _mol.AddConformer(conf, assignId=True)
                return True
            except Exception:
                return False

        
        def _template_ion_coords(_mol):
            """Return template coords for common electrolyte anions when RDKit embedding is unstable.
        
            The templates are used as a fallback (and preferred initializer for rigid polyhedral ions),
            to avoid hard failures on older RDKit builds (e.g., 2020.03.x).
        
            Implemented ions (heuristics):
              - Monoatomic ions: single atom at origin.
              - PF6- : octahedral geometry around P.
              - BF4- : tetrahedral geometry around B.
              - ClO4- : tetrahedral geometry around Cl.
              - FSI-  : FSO2–N–SO2F (approx. geometry).
              - TFSI- : (CF3SO2)2N- (approx. geometry).
              - DFOB- : O=C1O[B-](F)(F)OC1=O (approx. planar ring + tetrahedral B).
            """
            n = _mol.GetNumAtoms()
            if n == 1:
                return np.zeros((1, 3), dtype=float)
        
            def _tetra_dirs(scale):
                v = np.array([[1, 1, 1],
                              [1, -1, -1],
                              [-1, 1, -1],
                              [-1, -1, 1]], dtype=float)
                v /= np.linalg.norm(v[0])
                return v * float(scale)
        
            def _find_center(anum, deg, neigh_anum):
                for a in _mol.GetAtoms():
                    if a.GetAtomicNum() == anum and a.GetDegree() == deg:
                        neigh = [b.GetAtomicNum() for b in a.GetNeighbors()]
                        if len(neigh) == deg and all(x == neigh_anum for x in neigh):
                            return a.GetIdx()
                return None
        
            def _find_ns2_core():
                for a in _mol.GetAtoms():
                    if a.GetAtomicNum() == 7:
                        s_nei = [nb for nb in a.GetNeighbors() if nb.GetAtomicNum() == 16]
                        if len(s_nei) == 2:
                            return a.GetIdx(), s_nei[0].GetIdx(), s_nei[1].GetIdx()
                return None
        
            def _s_has_cf3(n_idx, s_idx):
                s = _mol.GetAtomWithIdx(s_idx)
                for nb in s.GetNeighbors():
                    if nb.GetIdx() == n_idx:
                        continue
                    if nb.GetAtomicNum() == 6:
                        fcnt = sum(1 for x in nb.GetNeighbors() if x.GetAtomicNum() == 9)
                        if fcnt >= 3:
                            return True
                return False
        
            def _s_has_terminal_f(n_idx, s_idx):
                s = _mol.GetAtomWithIdx(s_idx)
                has_f = False
                for nb in s.GetNeighbors():
                    if nb.GetIdx() == n_idx:
                        continue
                    if nb.GetAtomicNum() == 9:
                        has_f = True
                    if nb.GetAtomicNum() == 6:
                        fcnt = sum(1 for x in nb.GetNeighbors() if x.GetAtomicNum() == 9)
                        if fcnt >= 3:
                            return False
                return has_f
        
            def _find_dfob_b():
                if _mol.GetRingInfo() is None or _mol.GetRingInfo().NumRings() < 1:
                    return None
                for a in _mol.GetAtoms():
                    if a.GetAtomicNum() == 5 and a.GetDegree() == 4:
                        nei = list(a.GetNeighbors())
                        if sum(1 for x in nei if x.GetAtomicNum() == 9) == 2 and sum(1 for x in nei if x.GetAtomicNum() == 8) == 2:
                            return a.GetIdx()
                return None
        
            # PF6- (octahedral)
            try:
                p_idx = _find_center(15, 6, 9)
                if p_idx is not None:
                    r = 1.58
                    xyz = np.zeros((n, 3), dtype=float)
                    f_ids = [nb.GetIdx() for nb in _mol.GetAtomWithIdx(p_idx).GetNeighbors()]
                    vecs = [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]
                    for k, fi in enumerate(f_ids[:6]):
                        xyz[fi] = vecs[k]
                    return xyz
            except Exception:
                pass
        
            # BF4- (tetrahedral)
            try:
                b_idx = _find_center(5, 4, 9)
                if b_idx is not None:
                    r = 1.40
                    xyz = np.zeros((n, 3), dtype=float)
                    f_ids = [nb.GetIdx() for nb in _mol.GetAtomWithIdx(b_idx).GetNeighbors()]
                    dirs = _tetra_dirs(r)
                    for k, fi in enumerate(f_ids[:4]):
                        xyz[fi] = dirs[k]
                    return xyz
            except Exception:
                pass
        
            # ClO4- (tetrahedral)
            try:
                cl_idx = _find_center(17, 4, 8)
                if cl_idx is not None:
                    r = 1.44
                    xyz = np.zeros((n, 3), dtype=float)
                    o_ids = [nb.GetIdx() for nb in _mol.GetAtomWithIdx(cl_idx).GetNeighbors()]
                    dirs = _tetra_dirs(r)
                    for k, oi in enumerate(o_ids[:4]):
                        xyz[oi] = dirs[k]
                    return xyz
            except Exception:
                pass
        
            # FSI- / TFSI- (approx geometry)
            try:
                core = _find_ns2_core()
                if core is not None:
                    n_idx, s1_idx, s2_idx = core
                    is_tfsi = _s_has_cf3(n_idx, s1_idx) and _s_has_cf3(n_idx, s2_idx)
                    is_fsi = _s_has_terminal_f(n_idx, s1_idx) and _s_has_terminal_f(n_idx, s2_idx)
                    if is_tfsi or is_fsi:
                        xyz = np.full((n, 3), np.nan, dtype=float)
                        d_ns = 1.60
                        xyz[n_idx] = (0.0, 0.0, 0.0)
                        xyz[s1_idx] = (d_ns, 0.0, 0.0)
                        xyz[s2_idx] = (-d_ns, 0.0, 0.0)
        
                        def _place_s_group(s_idx):
                            s_atom = _mol.GetAtomWithIdx(s_idx)
                            o_nei = [nb for nb in s_atom.GetNeighbors() if nb.GetAtomicNum() == 8]
                            other = [nb for nb in s_atom.GetNeighbors() if nb.GetIdx() != n_idx and nb.GetAtomicNum() != 8]
                            r_so = 1.43
                            r_sf = 1.60
                            r_sc = 1.82
                            sx, sy, sz = xyz[s_idx]
                            if len(o_nei) >= 1:
                                xyz[o_nei[0].GetIdx()] = (sx, sy + r_so, sz)
                            if len(o_nei) >= 2:
                                xyz[o_nei[1].GetIdx()] = (sx, sy, sz + r_so)
                            if len(other) >= 1:
                                sub = other[0]
                                if sub.GetAtomicNum() == 9:
                                    xyz[sub.GetIdx()] = (sx, sy - r_sf / 1.4142, sz - r_sf / 1.4142)
                                elif sub.GetAtomicNum() == 6:
                                    xyz[sub.GetIdx()] = (sx, sy - r_sc, sz)
                                    c_idx = sub.GetIdx()
                                    f_nei = [x for x in sub.GetNeighbors() if x.GetAtomicNum() == 9]
                                    dirs = _tetra_dirs(1.35)
                                    pick = [0, 1, 2]
                                    for k, f_atom in enumerate(f_nei[:3]):
                                        dx, dy, dz = dirs[pick[k]]
                                        xyz[f_atom.GetIdx()] = (xyz[c_idx][0] + dx, xyz[c_idx][1] + dy, xyz[c_idx][2] + dz)
        
                        _place_s_group(s1_idx)
                        _place_s_group(s2_idx)
        
                        for i in range(n):
                            if not np.isfinite(xyz[i]).all():
                                xyz[i] = (0.0, 0.0, 0.0)
        
                        return xyz
            except Exception:
                pass
        
            # DFOB- (approx planar ring + tetrahedral B)
            try:
                b_idx = _find_dfob_b()
                if b_idx is not None:
                    xyz = np.full((n, 3), 0.0, dtype=float)
                    b = _mol.GetAtomWithIdx(b_idx)
                    f_nei = [nb for nb in b.GetNeighbors() if nb.GetAtomicNum() == 9]
                    o_nei = [nb for nb in b.GetNeighbors() if nb.GetAtomicNum() == 8]
                    xyz[b_idx] = (0.0, 0.0, 0.0)
                    r_bf = 1.35
                    if len(f_nei) >= 1:
                        xyz[f_nei[0].GetIdx()] = (0.0, 0.0, r_bf)
                    if len(f_nei) >= 2:
                        xyz[f_nei[1].GetIdx()] = (0.0, 0.0, -r_bf)
                    r_bo = 1.50
                    if len(o_nei) >= 1:
                        xyz[o_nei[0].GetIdx()] = (r_bo, 0.0, 0.0)
                    if len(o_nei) >= 2:
                        xyz[o_nei[1].GetIdx()] = (-r_bo, 0.0, 0.0)
                    for oi, sign in zip(o_nei[:2], [1.0, -1.0]):
                        cands = [nb for nb in oi.GetNeighbors() if nb.GetIdx() != b_idx and nb.GetAtomicNum() == 6]
                        if cands:
                            ci = cands[0].GetIdx()
                            xyz[ci] = (sign * (r_bo + 1.10), sign * 0.90, 0.0)
                            c_atom = _mol.GetAtomWithIdx(ci)
                            o_term = [nb for nb in c_atom.GetNeighbors() if nb.GetAtomicNum() == 8 and nb.GetIdx() != oi.GetIdx()]
                            if o_term:
                                ot = o_term[0].GetIdx()
                                xyz[ot] = (xyz[ci][0], xyz[ci][1] + sign * 1.20, 0.0)
                    return xyz
            except Exception:
                pass
        
            return None
        def _robust_embed(_mol):
            # 0) preferred rigid-ion templates (polyhedral anions) before RDKit embedding
            try:
                xyz0 = _template_ion_coords(_mol)
                if xyz0 is not None:
                    def _is_rigid_polyhedral(_m):
                        for a in _m.GetAtoms():
                            if a.GetAtomicNum() == 15 and a.GetDegree() == 6 and all(nb.GetAtomicNum() == 9 for nb in a.GetNeighbors()):
                                return True  # PF6-
                            if a.GetAtomicNum() == 5 and a.GetDegree() == 4 and all(nb.GetAtomicNum() == 9 for nb in a.GetNeighbors()):
                                return True  # BF4-
                            if a.GetAtomicNum() == 17 and a.GetDegree() == 4 and all(nb.GetAtomicNum() == 8 for nb in a.GetNeighbors()):
                                return True  # ClO4-
                        return False
                    if _is_rigid_polyhedral(_mol) and _set_conf_from_coords(_mol, xyz0):
                        return True
            except Exception:
                pass

            # 1) normal ETKDG
            try:
                if AllChem.EmbedMolecule(_mol, etkdg) != -1:
                    return True
            except Exception:
                pass

            # 2) random coords ETKDG with more attempts
            try:
                etkdg2 = AllChem.ETKDGv3() if hasattr(AllChem, 'ETKDGv3') else AllChem.ETKDG()
                etkdg2.enforceChirality = True
                etkdg2.useRandomCoords = True
                if hasattr(etkdg2, 'maxIterations'):
                    etkdg2.maxIterations = 1000
                elif hasattr(etkdg2, 'maxAttempts'):
                    etkdg2.maxAttempts = 1000
                etkdg2.randomSeed = 0xf00d
                if AllChem.EmbedMolecule(_mol, etkdg2) != -1:
                    return True
            except Exception:
                pass

            # 3) template coords for common inorganic ions
            xyz = _template_ion_coords(_mol)
            if xyz is not None and _set_conf_from_coords(_mol, xyz):
                return True

            # 4) last resort: trivial spaced-out coordinates to keep pipeline alive
            try:
                n = _mol.GetNumAtoms()
                xyz = np.zeros((n, 3), dtype=float)
                for i in range(n):
                    xyz[i, 0] = float(i) * 1.5
                if _set_conf_from_coords(_mol, xyz):
                    return True
            except Exception:
                pass

            return False

        if not _robust_embed(mol):
            radon_print('Cannot generate 3D coordinate of %s' % smiles, level=3)
            return None

    # Dihedral angles of unspecified double bonds in a polymer backbone are modified to 180 degree.
    if stereochemistry_control:
        if len(backbone_dih) > 0:
            for dih_idx in backbone_dih:
                Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), dih_idx[0], dih_idx[1], dih_idx[2], dih_idx[3], 180.0)

                for na in mol.GetAtomWithIdx(dih_idx[2]).GetNeighbors():
                    na_idx = na.GetIdx()
                    if na_idx != dih_idx[1] and na_idx != dih_idx[3]:
                        break
                Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(0), dih_idx[0], dih_idx[1], dih_idx[2], na_idx, 0.0)

    # Optional explicit naming (default naming is inferred later when the molecule
    # is passed into workflows like QM/packing/export).
    if name is not None:
        try:
            _n = str(name).strip()
            if _n:
                for k in ("_yadonpy_name", "name", "_yadonpy_resname", "_Name"):
                    try:
                        mol.SetProp(k, _n)
                    except Exception:
                        pass
        except Exception:
            pass

    return mol


def is_inorganic_ion_like(mol, smiles_hint: str = None) -> bool:
    """Heuristic detector for small inorganic ions that are hard for RDKit/MMFF/UFF and OptKing.

    Typical targets: PF6-, BF4-, ClO4-, simple metal ions.

    Criteria (best-effort):
    - net formal charge != 0
    - contains no carbon atoms
    - small (<= 30 atoms)

    This is intentionally conservative: for organic ions (with carbon), we prefer RDKit embedding.
    """
    try:
        if mol is None:
            return False
        nat = int(mol.GetNumAtoms())
        if nat == 0:
            return False
        if nat > 30:
            return False
        # count C
        has_c = any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
        if has_c:
            return False
        # formal charge
        fc = 0
        for a in mol.GetAtoms():
            try:
                fc += int(a.GetFormalCharge())
            except Exception:
                pass
        if fc == 0:
            return False
        # avoid polymerizable SMILES with '*'
        if smiles_hint and ('*' in str(smiles_hint)):
            return False
        return True
    except Exception:
        return False


def is_inorganic_polyatomic_ion(mol, smiles_hint: str = None) -> bool:
    """Heuristic detector for *polyatomic* inorganic ions.

    This is a stricter variant of :func:`is_inorganic_ion_like` that excludes
    monatomic ions (e.g., Na+, Li+), because bonded parameter derivation (e.g.,
    modified Seminario) only makes sense when internal bonds/angles exist.
    """
    try:
        if not is_inorganic_ion_like(mol, smiles_hint=smiles_hint):
            return False
        return int(mol.GetNumAtoms()) >= 2
    except Exception:
        return False


def _detect_ax_polyhedron(mol) -> Optional[Tuple[int, List[int], int]]:
    """Detect a simple AX4/AX6 polyhedron (high-symmetry inorganic ion).

    Returns:
        (center_idx, ligand_indices, coordination_number) or None
    """
    try:
        if mol is None:
            return None
        # Guard: exclude organics
        for a in mol.GetAtoms():
            if a.GetSymbol() == "C":
                return None
        for a in mol.GetAtoms():
            deg = int(a.GetDegree())
            if deg not in (4, 6):
                continue
            if a.GetSymbol() in ("C", "H"):
                continue
            neigh = list(a.GetNeighbors())
            if len(neigh) != deg:
                continue
            lig_syms = {n.GetSymbol() for n in neigh}
            if len(lig_syms) != 1:
                continue
            return a.GetIdx(), [n.GetIdx() for n in neigh], deg
        return None
    except Exception:
        return None


def is_high_symmetry_polyhedral_ion(mol, smiles_hint: str = None) -> bool:
    """Return True for common high-symmetry polyatomic inorganic ions.

    Intended for PF6-/BF4-/ClO4-/AsF6- like ions where small numeric distortions
    can later be amplified by Hessian-derived force constants and MD.
    """
    try:
        if not is_inorganic_polyatomic_ion(mol, smiles_hint=smiles_hint):
            return False
        return _detect_ax_polyhedron(mol) is not None
    except Exception:
        return False


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Compute optimal rotation R minimizing ||P R - Q|| (Kabsch)."""
    # Center
    P0 = P - np.mean(P, axis=0)
    Q0 = Q - np.mean(Q, axis=0)
    C = P0.T @ Q0
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, float(d)])
    R = V @ D @ Wt
    return R


def symmetrize_polyhedral_ion_geometry(mol, *, confId: int = 0) -> bool:
    """Symmetrize AX4/AX6 polyhedral geometry (tetrahedral/octet)."""
    try:
        hit = _detect_ax_polyhedron(mol)
        if hit is None:
            return False
        center_idx, lig_idxs, cn = hit
        conf = mol.GetConformer(int(confId))
        pc = conf.GetAtomPosition(center_idx)
        center = np.array([pc.x, pc.y, pc.z], dtype=float)
        lig_xyz = np.array([[p.x, p.y, p.z] for p in (conf.GetAtomPosition(i) for i in lig_idxs)], dtype=float)
        P = lig_xyz - center
        # Scale to average radius
        r = np.linalg.norm(P, axis=1)
        r_avg = float(np.mean(r)) if np.all(r > 1.0e-8) else float(np.max(r))
        if r_avg <= 1.0e-8:
            return False

        if cn == 4:
            Q0 = np.array(
                [
                    [1, 1, 1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                ],
                dtype=float,
            )
            Q0 /= np.linalg.norm(Q0[0])
        elif cn == 6:
            Q0 = np.array(
                [
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                ],
                dtype=float,
            )
        else:
            return False

        Q0 = Q0 * r_avg

        best = None
        best_rmsd = float("inf")

        for perm in permutations(range(cn)):
            Q = Q0[list(perm)]
            R = _kabsch_rotation(P, Q)
            diff = P @ R - Q
            rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best = (perm, R)
                if best_rmsd < 1.0e-6:
                    break

        if best is None:
            return False

        perm, R = best
        Q = Q0[list(perm)]
        # Rotate ideal vectors back into the current frame (minimizes jump)
        new_rel = Q @ R.T
        for idx, rel in zip(lig_idxs, new_rel):
            x, y, z = center + rel
            conf.SetAtomPosition(int(idx), Geom.Point3D(float(x), float(y), float(z)))
        return True
    except Exception:
        return False


def ensure_3d_coords(mol, *, smiles_hint: str = None, engine: str = "auto") -> bool:
    """Ensure the molecule has at least one 3D conformer.

    Motivation:
    - RDKit distance geometry is usually best for organic molecules.
    - Some small inorganic ions (PF6-, BF4-, ClO4-, simple metal ions) are notoriously unstable under
      RDKit+UFF/MMFF and can also break OptKing's internal coordinates in QM optimizations.
      For these, OpenBabel's 3D builder + UFF local optimization is often more reliable.

    Strategy (engine='auto'):
      - inorganic-ion-like  : OpenBabel -> RDKit -> templates -> spaced-out coords
      - otherwise           : RDKit -> OpenBabel -> templates -> spaced-out coords

    Args:
        mol: RDKit Mol. Modified in-place.
        smiles_hint: Original SMILES string (recommended). Used by OpenBabel.
        engine: 'auto'|'rdkit'|'openbabel'|'template'.

    Returns:
        True if coordinates were set or already present.
    """
    try:
        if mol is None:
            return False
        if mol.GetNumConformers() > 0:
            return True
        n = int(mol.GetNumAtoms())
        if n <= 0:
            return False

        eng = (engine or 'auto').lower().strip()

        def _add_conf(xyz):
            import numpy as _np
            from rdkit.Geometry import Point3D as _P3D
            xyz = _np.asarray(xyz, dtype=float)
            if xyz.ndim == 1:
                xyz = xyz.reshape((-1, 3))
            if xyz.shape != (n, 3):
                return False
            conf = Chem.Conformer(n)
            for i in range(n):
                conf.SetAtomPosition(i, _P3D(float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])))
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            return True

        def _degenerate_conf_ok() -> bool:
            import numpy as _np
            if mol.GetNumConformers() == 0:
                return False
            xyz = _np.array(mol.GetConformer(0).GetPositions(), dtype=float)
            if xyz.shape != (n, 3):
                return False
            if n < 2:
                return True
            min_d2 = float('inf')
            for i in range(n):
                dv = xyz[i + 1:] - xyz[i]
                if dv.size == 0:
                    continue
                d2 = _np.sum(dv * dv, axis=1)
                md2 = float(_np.min(d2))
                if md2 < min_d2:
                    min_d2 = md2
            return (min_d2 > 1.0e-6)

        def _try_rdkit() -> bool:
            try:
                from rdkit.Chem import AllChem as _AllChem
                if hasattr(_AllChem, 'ETKDGv3'):
                    params = _AllChem.ETKDGv3()
                elif hasattr(_AllChem, 'ETKDGv2'):
                    params = _AllChem.ETKDGv2()
                else:
                    params = _AllChem.ETKDG()
                params.useRandomCoords = True
                params.enforceChirality = True
                if hasattr(params, 'randomSeed'):
                    params.randomSeed = 0xC0FFEE
                if hasattr(params, 'maxIterations'):
                    params.maxIterations = 2000
                elif hasattr(params, 'maxAttempts'):
                    params.maxAttempts = 2000
                if hasattr(params, 'timeout'):
                    params.timeout = 10

                cid = _AllChem.EmbedMolecule(mol, params)
                if int(cid) >= 0 and mol.GetNumConformers() > 0 and _degenerate_conf_ok():
                    return True
                mol.RemoveAllConformers()
            except Exception:
                try:
                    mol.RemoveAllConformers()
                except Exception:
                    pass
            return False

        def _try_openbabel() -> bool:
            # OpenBabel is optional. Only used when available and a SMILES hint is provided.
            try:
                if not smiles_hint:
                    return False
                smi = str(smiles_hint).strip()
                if (not smi) or ('*' in smi):
                    return False
                # Use the Open Babel Python bindings shipped with the `openbabel`
                # package. This is intentionally not the unrelated standalone
                # `pybel` package from PyPI.
                from openbabel import pybel as openbabel_pybel  # type: ignore
            except Exception:
                return False

            try:
                ob = openbabel_pybel.readstring('smi', smi)
                # For some species, adding H stabilizes 3D builder; ignore failures.
                try:
                    ob.addh()
                except Exception:
                    pass

                # 3D build + UFF local opt
                ob.make3D(forcefield='uff', steps=500)
                ob.localopt(forcefield='uff', steps=500)

                molblock = ob.write('mol')
                if not molblock:
                    return False

                # Parse OpenBabel MOL into RDKit (for mapping)
                ob_m = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
                if ob_m is None or ob_m.GetNumConformers() == 0:
                    return False
                try:
                    Chem.SanitizeMol(
                        ob_m,
                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                    )
                except Exception:
                    pass

                # Map atoms by graph isomorphism (robust for ions like PF6-)
                match = ()
                try:
                    match = mol.GetSubstructMatch(ob_m)
                except Exception:
                    match = ()
                if (not match) or (len(match) != n):
                    # try heavy-atom mapping if hydrogens differ
                    try:
                        mol_h = Chem.RemoveHs(mol)
                        ob_h = Chem.RemoveHs(ob_m)
                        mh = mol_h.GetSubstructMatch(ob_h)
                        if mh and len(mh) == mol_h.GetNumAtoms():
                            # build xyz for heavy atoms only; if mol has H, they remain 0 and will be refined later
                            import numpy as _np
                            xyz = _np.zeros((n, 3), dtype=float)
                            xyz_ob = _np.array(ob_m.GetConformer(0).GetPositions(), dtype=float)
                            # naive heavy-atom index mapping via RemoveHs index correspondence
                            # If this path is taken, coordinates may be rough but non-degenerate.
                            for qi, ti in enumerate(mh):
                                xyz[int(ti)] = xyz_ob[int(qi)]
                            return _add_conf(xyz)
                    except Exception:
                        pass
                    return False

                import numpy as _np
                xyz_ob = _np.array(ob_m.GetConformer(0).GetPositions(), dtype=float)
                xyz = _np.zeros((n, 3), dtype=float)
                for qi, ti in enumerate(match):
                    xyz[int(ti)] = xyz_ob[int(qi)]
                ok = _add_conf(xyz)
                if ok and _degenerate_conf_ok():
                    return True
                mol.RemoveAllConformers()
                return False
            except Exception:
                try:
                    mol.RemoveAllConformers()
                except Exception:
                    pass
                return False

        def _try_templates() -> bool:
            import numpy as _np

            def _tetra_dirs(r):
                v = _np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
                v = v / _np.linalg.norm(v, axis=1)[:, None]
                return v * float(r)

            # PF6- octahedral
            for a in mol.GetAtoms():
                if a.GetAtomicNum() == 15 and a.GetDegree() == 6 and all(nb.GetAtomicNum() == 9 for nb in a.GetNeighbors()):
                    f = [nb.GetIdx() for nb in a.GetNeighbors()]
                    r = 1.58
                    xyz = _np.zeros((n, 3), float)
                    dirs = _np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]], float)
                    for k, fi in enumerate(f[:6]):
                        xyz[fi] = dirs[k]
                    return _add_conf(xyz)

            # BF4- tetrahedral
            for a in mol.GetAtoms():
                if a.GetAtomicNum() == 5 and a.GetDegree() == 4 and all(nb.GetAtomicNum() == 9 for nb in a.GetNeighbors()):
                    f = [nb.GetIdx() for nb in a.GetNeighbors()]
                    xyz = _np.zeros((n, 3), float)
                    dirs = _tetra_dirs(1.35)
                    for k, fi in enumerate(f[:4]):
                        xyz[fi] = dirs[k]
                    return _add_conf(xyz)

            # ClO4- tetrahedral
            for a in mol.GetAtoms():
                if a.GetAtomicNum() == 17 and a.GetDegree() == 4 and all(nb.GetAtomicNum() == 8 for nb in a.GetNeighbors()):
                    o = [nb.GetIdx() for nb in a.GetNeighbors()]
                    xyz = _np.zeros((n, 3), float)
                    dirs = _tetra_dirs(1.44)
                    for k, oi in enumerate(o[:4]):
                        xyz[oi] = dirs[k]
                    return _add_conf(xyz)

            return False

        def _try_spaced() -> bool:
            import numpy as _np
            xyz = _np.zeros((n, 3), float)
            for i in range(n):
                xyz[i, 0] = 1.5 * float(i)
            return _add_conf(xyz)

        if eng == 'rdkit':
            order = ['rdkit', 'openbabel', 'template', 'spaced']
        elif eng == 'openbabel':
            order = ['openbabel', 'rdkit', 'template', 'spaced']
        elif eng in ('template', 'templates'):
            order = ['template', 'spaced']
        else:
            if is_inorganic_ion_like(mol, smiles_hint=smiles_hint):
                order = ['openbabel', 'rdkit', 'template', 'spaced']
            else:
                order = ['rdkit', 'openbabel', 'template', 'spaced']

        for step in order:
            if step == 'rdkit' and _try_rdkit():
                return True
            if step == 'openbabel' and _try_openbabel():
                return True
            if step == 'template' and _try_templates():
                return True
            if step == 'spaced' and _try_spaced():
                return True

        return False

    except Exception:
        return False


def scale_atomic_charges(mol, *, scale: float, props=("AtomicCharge", "RESP")):
    """Scale atomic charges in-place.

    This utility is used to model dielectric screening (charge scaling) in MD.

    Implementation details:
    - When scaling is applied, the original values are preserved in
      ``<prop>_raw`` atom properties (only written once).
    - If a charge property does not exist on an atom, it is skipped.

    Args:
        mol: RDKit Mol.
        scale: Multiplicative scale factor.
        props: Atom double properties to scale, checked in order.
    """

    try:
        s = float(scale)
    except Exception:
        raise ValueError(f"Invalid charge scale: {scale}")

    if abs(s - 1.0) < 1.0e-12:
        return mol

    for a in mol.GetAtoms():
        for prop in props:
            if a.HasProp(prop):
                # preserve original
                raw_key = f"{prop}_raw"
                if not a.HasProp(raw_key):
                    try:
                        a.SetDoubleProp(raw_key, float(a.GetDoubleProp(prop)))
                    except Exception:
                        pass
                try:
                    a.SetDoubleProp(prop, float(a.GetDoubleProp(prop)) * s)
                except Exception:
                    pass
    return mol


def restore_raw_charges(mol, *, props=("AtomicCharge", "RESP")):
    """Restore raw charges previously saved by :func:`scale_atomic_charges`.

    If a corresponding ``<prop>_raw`` exists, it overwrites ``<prop>``.
    This is helpful when we want to apply charge scaling for a simulation,
    but keep persistent caches (MolDB / artifact caches) in an unscaled state.
    """
    for a in mol.GetAtoms():
        for prop in props:
            raw_key = f"{prop}_raw"
            if a.HasProp(raw_key):
                try:
                    a.SetDoubleProp(prop, float(a.GetDoubleProp(raw_key)))
                except Exception:
                    pass
    return mol


def correct_total_charge(
    mol,
    *,
    target_q: float | int | None = None,
    props=("AtomicCharge", "RESP"),
    tol: float = 1.0e-3,
    strategy: str = "uniform",
):
    """Correct per-atom partial charges so the molecule net charge matches a target.

    Motivation
    ----------
    In practice, RESP (and some FF assignment pipelines) can yield a net charge
    that slightly deviates from the intended integer formal charge due to
    numerical noise or round-tripping through intermediate formats.
    This becomes problematic for neutral molecules (net != 0) and for
    polyelectrolytes where the total charge should equal the molecule's formal
    charge (often proportional to degree of polymerization).

    This helper applies a small correction *in-place* by distributing the
    required offset across atoms.

    Args:
        mol: RDKit Mol.
        target_q: Target total charge in electron units. If None, we use the
            sum of RDKit atom formal charges.
        props: Atom properties to correct (checked in order). All existing
            properties in this list will be corrected.
        tol: If |target_q - current_q| <= tol, no correction is applied.
        strategy: "uniform" (default) adds the same delta to every atom.

    Returns:
        A dict describing the correction that was applied (or None if skipped).
    """

    try:
        n = int(mol.GetNumAtoms())
    except Exception:
        return None
    if n <= 0:
        return None

    # Determine target from formal charges if not provided.
    if target_q is None:
        tq = 0.0
        try:
            for a in mol.GetAtoms():
                tq += float(a.GetFormalCharge())
        except Exception:
            tq = 0.0
        target_q = tq
    target_q = float(target_q)

    # Compute current total for each property and apply correction.
    applied_any = False
    info = {"target_q": float(target_q), "tol": float(tol), "per_prop": {}}

    for prop in props:
        # Only correct if the property exists on at least one atom.
        has = False
        for a in mol.GetAtoms():
            if a.HasProp(prop):
                has = True
                break
        if not has:
            continue

        cur = 0.0
        for a in mol.GetAtoms():
            if a.HasProp(prop):
                try:
                    cur += float(a.GetDoubleProp(prop))
                except Exception:
                    cur += 0.0
        delta = float(target_q) - float(cur)

        info["per_prop"][prop] = {"current_q": float(cur), "delta": float(delta)}

        if abs(delta) <= float(tol):
            continue

        if strategy != "uniform":
            # For now we only implement uniform distribution (robust, deterministic).
            strategy = "uniform"

        per_atom = delta / float(n)
        for a in mol.GetAtoms():
            if a.HasProp(prop):
                try:
                    a.SetDoubleProp(prop, float(a.GetDoubleProp(prop)) + per_atom)
                except Exception:
                    # If setting fails for a particular atom, keep going.
                    pass

        applied_any = True

        # Recompute for record.
        cur2 = 0.0
        for a in mol.GetAtoms():
            if a.HasProp(prop):
                try:
                    cur2 += float(a.GetDoubleProp(prop))
                except Exception:
                    pass
        info["per_prop"][prop]["corrected_q"] = float(cur2)

    if not applied_any:
        return None

    # Record on mol for downstream debugging.
    try:
        import json as _json

        if hasattr(mol, "SetProp"):
            mol.SetProp("_yadonpy_charge_correction", _json.dumps(info, ensure_ascii=False))
    except Exception:
        pass

    return info


def mol_from_pdb(pdb_file, charge=False):

    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)

    # read charges
    if charge:
        charges = []
        lines = []
        try:
            f = open(pdb_file, 'r')
            lines = f.readlines()
            f.close()
        except:
            f.close()
            print("ERROR: Failed to read " + pdb_file)
            sys.exit()

        try:
            for i, line in enumerate(lines):
                if line[:4] == 'ATOM':
                    q = float(line[80:].strip())
                    charges.append(q)
                    continue
        except:
            print("ERROR: Failed to read charges from " + pdb_file)
            sys.exit()

        natom = mol.GetNumAtoms()
        if natom != len(charges):
            print("ERROR: Failed to read charges from " + pdb_file)

        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetDoubleProp('AtomicCharge', charges[i])

    natom = mol.GetNumAtoms()
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp('ff_type', atom.GetPDBResidueInfo().GetName())
        atom.SetBoolProp('terminal', i == 0 or i == natom - 1)
        
    return mol


def is_in_ring(ab, max_size=10):

    for i in range(3, max_size+1):
        if ab.IsInRingSize(int(i)):
            return True
    return False
