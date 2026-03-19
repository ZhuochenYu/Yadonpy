"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from .core import utils
from .core.utils import YadonPyError
from .core.data_dir import ensure_initialized
from .ff.gaff import GAFF
from .ff.gaff2 import GAFF2
from .ff.gaff2_mod import GAFF2_mod
from .ff.merz import MERZ
from .ff.library import LibraryDB, is_polymer_smiles, smiles_to_molid
from .io.artifacts import write_molecule_artifacts


def ensure_basic_top(
    smiles: str,
    *,
    ff_name: str,
    charge_method: str = "RESP",
    work_dir: Union[str, Path] = "./",
    total_charge: Optional[int] = None,
    charge_scale: float = 1.0,
) -> dict:
    """Ensure GROMACS artifacts (.itp/.top/.gro) exist for a SMILES.

    Behavior (as requested):
      1) Look up by SMILES in the built-in/user basic_top library.
      2) If present, return the library entry (with resolved artifact_dir).
      3) If absent, run FF assignment + (optional) RESP, generate artifacts,
         register into the user library, and return the new entry.
    """
    _, ent = parameterize_smiles(
        smiles,
        ff_name=ff_name,
        charge_method=charge_method,
        work_dir=work_dir,
        auto_register_nonpolymer=True,
        use_basic_top_first=True,
        total_charge=total_charge,
        charge_scale=charge_scale,
    )
    if ent is None:
        raise RuntimeError("Failed to resolve or generate artifacts for SMILES")
    return ent


def get_ff(ff_name: str):
    ff_name = ff_name.lower()
    if ff_name == "gaff":
        return GAFF()
    if ff_name == "gaff2":
        return GAFF2()
    if ff_name in {"gaff2_mod", "gaff2-mod", "gaff2mod"}:
        return GAFF2_mod()
    if ff_name in {"merz", "merzop", "merzopc3"}:
        return MERZ()
    raise ValueError(f"Unknown force field: {ff_name}")


def parameterize_smiles(
    smiles: str,
    *,
    ff_name: str,
    charge_method: str = "RESP",
    work_dir: Union[str, Path] = "./",
    auto_register_nonpolymer: bool = True,
    use_basic_top_first: bool = True,
    total_charge: Optional[int] = None,
    charge_scale: float = 1.0,
) -> Tuple[object, Optional[dict]]:
    """Build+parameterize a molecule from SMILES.

    This is a thin, script-friendly wrapper intended for the new yadonpy
    workflow (no CLI). It also supports the requested behavior:

      - If the SMILES is *not* a polymer monomer (does not contain '*'), then
        after the first successful computation the molecule will be added to
        the default library under the initialized data root.

    Returns:
        (mol, library_entry_dict_or_None)
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    db = LibraryDB()

    # 0) Try basic_top/library first (SMILES-only matching)
    if use_basic_top_first:
        ent = db.find_by_smiles(ff_name.lower(), smiles)
        if ent is None and ff_name.lower() != "merz":
            ent = db.find_by_smiles("merz", smiles)
        if ent is not None:
            art_dir = db.resolve_artifact_dir(ent.get("artifact_dir", ""))
            # validate presence of expected files
            mol_id = ent.get("mol_id") or "MOL"
            # built-in molecules typically named by directory; accept any
            have_itp = any(art_dir.glob("*.itp"))
            have_gro = any(art_dir.glob("*.gro"))
            have_top = any(art_dir.glob("*.top"))
            if art_dir.exists() and have_itp and have_gro and have_top:
                # Validate cached .itp has bonded terms when expected.
                # A bug in <=0.2.26 could drop Python-level topology attrs when copying
                # RDKit mols, producing .itp files with [ bonds ] but missing [ angles ]
                # / [ dihedrals ]. If detected, force regeneration.
                try:
                    mol0 = utils.mol_from_smiles(smiles)
                except Exception:
                    mol0 = None

                try:
                    itp_files = sorted(art_dir.glob("*.itp"))
                    itp_txt = itp_files[0].read_text(encoding="utf-8") if itp_files else ""
                    nat0 = int(getattr(mol0, "GetNumAtoms", lambda: 0)()) if mol0 is not None else 0
                    needs_angles = bool(nat0 >= 3)
                    needs_dihedrals = bool(nat0 >= 4)

                    missing_angles = needs_angles and ("[ angles ]" not in itp_txt)
                    missing_dihedrals = needs_dihedrals and ("[ dihedrals ]" not in itp_txt)

                    if missing_angles or missing_dihedrals:
                        # Invalidate cache (buggy cached topologies from earlier versions)
                        ent = None
                except Exception:
                    pass

                if ent is not None:
                    return mol0, {**ent, "artifact_dir": str(art_dir)}
    ff = get_ff(ff_name)

    # Build RDKit molecule. Some inorganic/charged ions may fail 3D embedding on older RDKit builds.
    try:
        mol = utils.mol_from_smiles(smiles)
    except Exception as e:
        utils.yadon_print(
            f"[WARN] mol_from_smiles 3D embedding failed for {smiles}; falling back to template/trivial 3D. ({e})",
            level=2,
        )
        mol = utils.mol_from_smiles(smiles, coord=False)
        if mol is None:
            raise
        try:
            utils.ensure_3d_coords(mol, smiles_hint=smiles)
        except Exception:
            pass



    # -------------------------------
    # Monoatomic ion shortcut
    # -------------------------------
    # GAFF-family force fields do not define parameters for common inorganic ions (Li+, Na+, Cl-, ...).
    # In those cases we automatically fall back to MERZ ion LJ parameters.
    # We also skip QM/RESP: for a single atom, the charge is just its formal charge.
    _skip_charge = False
    try:
        if mol is not None and int(mol.GetNumAtoms()) == 1:
            q = int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms()))
            if q != 0:
                _skip_charge = True
                a0 = mol.GetAtomWithIdx(0)
                a0.SetDoubleProp('AtomicCharge', float(q))
                a0.SetDoubleProp('AtomicCharge_raw', float(q))
                try:
                    a0.SetDoubleProp('RESP', float(q))
                    a0.SetDoubleProp('RESP_raw', float(q))
                except Exception:
                    pass

                if ff_name.lower() not in {'merz', 'merzop', 'merzopc3'}:
                    utils.radon_print(
                        f"[WARN] {smiles} is a monoatomic ion; overriding ff_name={ff_name} -> merz.",
                        level=2,
                    )
                    ff_name = 'merz'
                    ff = MERZ()
    except Exception:
        _skip_charge = False
    # Assign charges (robust default):
    #   - If QM-based methods (RESP/ESP/Mulliken/Lowdin) fail (e.g. Psi4 not installed),
    #     fall back to RDKit Gasteiger charges.
    #   - If even that fails, continue with 0.0 charges.
    if not _skip_charge:
        try:
            from .sim import qm

            ok_charge = bool(
                qm.assign_charges(
                    mol,
                    charge=charge_method,
                    opt=False,
                    work_dir=str(work_dir),
                    total_charge=total_charge,
                )
            )
            if (not ok_charge) and str(charge_method).strip().upper() not in {"GASTEIGER", "ZERO"}:
                utils.radon_print(
                    f"Charge method {charge_method} failed; falling back to Gasteiger charges.",
                    level=2,
                )
                ok_g = bool(
                    qm.assign_charges(
                        mol,
                        charge="gasteiger",
                        opt=False,
                        work_dir=str(work_dir),
                        total_charge=total_charge,
                    )
                )
                if not ok_g:
                    utils.radon_print(
                        "Gasteiger charge assignment also failed; continuing with zero charges.",
                        level=2,
                    )
        except Exception as e:
            # Keep running: some users may assign charges elsewhere.
            utils.radon_print(f"Charge assignment failed ({e}); continuing with zero charges.", level=2)

    # IMPORTANT: charge scaling is a *simulation-level* choice.
    # Do NOT modify the RDKit molecule charges here (or we'd risk double-scaling
    # when exporting GROMACS topologies). Scaling is applied at system export time.

    # Assign FF parameters
    ok = ff.ff_assign(mol)
    if not ok:
        raise RuntimeError(f"Force-field assignment failed (ff={ff_name})")

    # --- Short FF assignment summary (helps debug missing angles/dihedrals) ---
    try:
        nat = int(mol.GetNumAtoms())
        nb = int(mol.GetNumBonds())
        nang = int(len(getattr(mol, "angles", {}) or {}))
        ndih = int(len(getattr(mol, "dihedrals", {}) or {}))
        nimpr = int(len(getattr(mol, "impropers", {}) or {}))
        ff_tag = ff_name
        try:
            if hasattr(mol, "HasProp") and mol.HasProp("ff_name"):
                ff_tag = mol.GetProp("ff_name")
        except Exception:
            pass

        utils.radon_print(
            f"FF assign summary ({ff_tag}): atoms={nat} bonds={nb} angles={nang} dihedrals={ndih} impropers={nimpr}",
            level=1,
        )

        if nat >= 3 and nang == 0:
            utils.radon_print(
                "[WARN] angles=0 for a multi-atom molecule; resulting .itp will be unphysical. Will force regenerate.",
                level=2,
            )
        if nat >= 4 and ndih == 0:
            utils.radon_print(
                "[WARN] dihedrals=0 for a >=4-atom molecule; check FF assignment/SMILES.",
                level=2,
            )
    except Exception:
        pass

    # User-visible summary to confirm bonded terms were actually assigned.
    # This is intentionally concise but diagnostic.
    try:
        nat = int(mol.GetNumAtoms())
        nb = int(mol.GetNumBonds())
        nang = len(getattr(mol, 'angles', {}) or {})
        ndih = len(getattr(mol, 'dihedrals', {}) or {})
        nimp = len(getattr(mol, 'impropers', {}) or {})
        utils.radon_print(
            f"FF assigned ({ff_name}): atoms={nat} bonds={nb} angles={nang} dihedrals={ndih} impropers={nimp}",
            level=1,
        )
        if nat >= 3 and nang == 0:
            utils.radon_print(
                "[WARN] Angle terms were not assigned (angles=0). Generated .itp will be unphysical.",
                level=2,
            )
        if nat >= 4 and ndih == 0:
            utils.radon_print(
                "[WARN] Dihedral terms were not assigned (dihedrals=0). Generated .itp will be unphysical.",
                level=2,
            )
    except Exception:
        pass

    entry = None
    if auto_register_nonpolymer and (not is_polymer_smiles(smiles)):
        layout = ensure_initialized()
        mol_id = smiles_to_molid(smiles)
        artifact_dir = layout.basic_top_dir / ff_name.lower() / mol_id
        # When registering molecules into the default library, we store *raw* charges
        # (charge_scale is a simulation-level choice, and should not permanently modify the library).
        # If charge scaling was requested, raw charges are preserved in *_raw props and will be used.
        # Create a safe copy for persistent storage and restore raw charges if present.
        mol_lib = mol
        try:
            from rdkit import Chem

            mol_lib = Chem.Mol(mol)
            utils.restore_raw_charges(mol_lib)
            # IMPORTANT: RDKit copy drops Python-level topology attributes
            # (mol.angles/mol.dihedrals/...), which would cause cached .itp to
            # miss [ angles ]/[ dihedrals ]. Preserve them explicitly.
            try:
                utils.copy_topology_attributes(mol, mol_lib)
            except Exception:
                pass
        except Exception:
            mol_lib = mol

        write_molecule_artifacts(
            mol_lib,
            artifact_dir,
            smiles=smiles,
            ff_name=ff_name.lower(),
            charge_method=charge_method,
            total_charge=total_charge,
            mol_name=mol_id,
            charge_scale=1.0,
        )
        db = LibraryDB()
        ent = db.ensure_registered(
            ff=ff_name.lower(),
            smiles=smiles,
            artifact_dir=artifact_dir,
            is_original_from_lib=False,
        )
        entry = ent.to_dict()

    return mol, entry
