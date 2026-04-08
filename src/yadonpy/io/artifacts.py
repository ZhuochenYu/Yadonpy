"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from ..core import chem_utils as core_utils


def _bonded_meta_from_mol(mol) -> Dict[str, Any]:
    """Collect bonded-override metadata that should survive caching."""
    meta: Dict[str, Any] = {}
    try:
        if hasattr(mol, 'HasProp'):
            for key in (
                '_yadonpy_bonded_signature',
                '_yadonpy_bonded_requested',
                '_yadonpy_bonded_method',
                '_yadonpy_bonded_override',
                '_yadonpy_bonded_explicit',
                '_yadonpy_bonded_itp',
                '_yadonpy_bonded_json',
                '_yadonpy_mseminario_itp',
                '_yadonpy_mseminario_json',
            ):
                if mol.HasProp(key):
                    val = str(mol.GetProp(key)).strip()
                    if val:
                        meta[key] = val
    except Exception:
        pass
    return meta


def _reapply_bonded_patch_to_mol(mol) -> None:
    """Best-effort reapplication of a bonded patch after angle/dihedral rebuilds."""
    try:
        if not hasattr(mol, 'HasProp'):
            return
        jp = None
        if mol.HasProp('_yadonpy_bonded_json'):
            jp = str(mol.GetProp('_yadonpy_bonded_json')).strip()
        elif mol.HasProp('_yadonpy_mseminario_json'):
            jp = str(mol.GetProp('_yadonpy_mseminario_json')).strip()
        if not jp:
            return
        from pathlib import Path as _Path
        _p = _Path(jp)
        if not _p.is_file():
            return
        import json as _json
        from ..sim.qm import apply_mseminario_params_to_mol as _apply
        obj = _json.loads(_p.read_text(encoding='utf-8'))
        if isinstance(obj, dict):
            _apply(mol, obj, overwrite=True)
    except Exception:
        pass



def _ensure_bonded_terms_for_export(mol, ff_name: str) -> None:
    """Ensure mol.angles/mol.dihedrals exist before writing .itp.

    In some workflows (notably mixed-system exports), molecules may be obtained via RDKit fragment
    extraction or serialization that drops Python-level attributes (mol.angles/mol.dihedrals) even
    though ff_type/ff_r0/ff_k are still present. This helper rebuilds higher-order bonded terms
    using the requested force-field name.
    """
    try:
        nat = int(mol.GetNumAtoms())
    except Exception:
        nat = 0
    if nat < 3:
        return

    has_angles = bool(getattr(mol, "angles", {}) or {})
    has_dihedrals = bool(getattr(mol, "dihedrals", {}) or {})

    if has_angles and (nat < 4 or has_dihedrals):
        return

    ff_name_l = str(ff_name).lower().strip()
    ff_obj = None
    try:
        if ff_name_l == "gaff":
            from ..ff.gaff import GAFF
            ff_obj = GAFF()
        elif ff_name_l == "gaff2":
            from ..ff.gaff2 import GAFF2
            ff_obj = GAFF2()
        elif ff_name_l in {"gaff2_mod", "gaff2-mod", "gaff2mod"}:
            from ..ff.gaff2_mod import GAFF2_mod
            ff_obj = GAFF2_mod()
        elif ff_name_l in {"merz", "merzop", "merzopc3"}:
            from ..ff.merz import MERZ
            ff_obj = MERZ()
    except Exception:
        ff_obj = None

    if ff_obj is None:
        return

    # Only (re)assign higher-order terms; do not redo ptypes/btypes or charges here.
    try:
        if nat >= 3 and (not has_angles) and hasattr(ff_obj, "assign_atypes"):
            ff_obj.assign_atypes(mol)
    except Exception:
        pass
    try:
        if nat >= 4 and (not has_dihedrals) and hasattr(ff_obj, "assign_dtypes"):
            ff_obj.assign_dtypes(mol)
    except Exception:
        pass
    try:
        if hasattr(ff_obj, "assign_itypes"):
            ff_obj.assign_itypes(mol)
    except Exception:
        pass


def write_molecule_artifacts(
    mol,
    out_dir: Path,
    *,
    smiles: str,
    ff_name: str,
    charge_method: str,
    total_charge: Optional[int] = None,
    mol_name: str = "MOL",
    charge_scale: float = 1.0,
    mol2_root: Optional[Path] = None,
    write_mol2: bool = True,
) -> Path:
    """Write cached artifacts for a single molecule.

    Notes:
      - This function writes *per-molecule* artifacts to a given directory.
      - Higher-level caches (MolDB, molecule_cache) decide *when* to call this.

    Returns:
        out_dir
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Charge fingerprinting (cache validation)
    #
    # Cached artifacts persist across runs. If a cached topology was
    # generated when atomic charges were missing (e.g., a failed RESP step),
    # it will silently be reused and propagate all-zero charges into the
    # system. We store a compact charge signature in meta.json so that the
    # cache layer can invalidate and regenerate artifacts when needed.
    # ------------------------------------------------------------------
    def _mol_charge_abs_and_sig(_mol) -> tuple[float, str]:
        _, selected = core_utils.select_best_charge_property(_mol)
        qs = [round(float(q), 6) for q in selected]
        abs_sum = float(sum(abs(x) for x in qs))
        payload = ",".join(f"{x:.6f}" for x in qs).encode('utf-8')
        sig = hashlib.sha1(payload).hexdigest()[:16]
        return abs_sum, sig

    # Metadata
    ch_abs, ch_sig = _mol_charge_abs_and_sig(mol)
    meta: Dict[str, Any] = {
        "smiles": smiles,
        "ff": ff_name,
        "charge_method": charge_method,
        "total_charge": total_charge,
        "charge_scale": float(charge_scale),
        "mol_name": mol_name,
        "n_atoms": int(mol.GetNumAtoms()) if hasattr(mol, "GetNumAtoms") else None,
        # Cache validation helpers
        "charge_abs_sum": float(ch_abs),
        "charge_signature": str(ch_sig),
    }
    meta.update(_bonded_meta_from_mol(mol))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Tag the molecule with its artifact directory for downstream workflows (best effort)
    try:
        if hasattr(mol, "SetProp"):
            mol.SetProp("_yadonpy_artifact_dir", str(out_dir))
    except Exception:
        pass

    # GROMACS artifacts (required for MD; do not silently ignore failures)
    # NOTE: `charge_scale` is a *simulation-level* choice and MUST NOT be baked into
    # cached per-molecule artifacts. Charge scaling is applied when exporting a *system*
    # (see io.gromacs_system.export_system_from_cell_meta).
    try:
        # Require a 3D conformer for .gro/.itp generation
        if hasattr(mol, "GetNumConformers") and int(mol.GetNumConformers()) <= 0:
            raise ValueError(
                "RDKit molecule has no conformer (3D coordinates). "
                "Generate 3D geometry before writing GROMACS artifacts."
            )

        # ---- charge sanity / auto-correction (unscaled artifacts) ----
        # Many downstream tools assume integer net charge, and small deviations
        # (e.g., -0.0008 e) can accumulate or cause ion classification issues.
        # We correct charges *only* for the unscaled cached artifacts.
        try:
            from ..core.chem_utils import correct_total_charge

            target_q = total_charge
            # IMPORTANT:
            #   If the caller did not provide an explicit target charge, preserve
            #   the current per-atom charge sum instead of falling back to RDKit
            #   formal charges. Hypervalent species such as PF6- can round-trip
            #   through RDKit with an unreliable formal charge, and using that as
            #   the target would silently mutate a correct RESP charge set.
            if target_q is None:
                current_q = 0.0
                has_charge_props = False
                for atom in mol.GetAtoms():
                    try:
                        if atom.HasProp('AtomicCharge'):
                            current_q += float(atom.GetDoubleProp('AtomicCharge'))
                            has_charge_props = True
                            continue
                        if atom.HasProp('RESP'):
                            current_q += float(atom.GetDoubleProp('RESP'))
                            has_charge_props = True
                            continue
                        if atom.HasProp('_GasteigerCharge'):
                            current_q += float(atom.GetProp('_GasteigerCharge'))
                            has_charge_props = True
                    except Exception:
                        continue
                if has_charge_props:
                    formal_q = 0.0
                    try:
                        for atom in mol.GetAtoms():
                            formal_q += float(atom.GetFormalCharge())
                    except Exception:
                        formal_q = float(current_q)

                    # Small RESP / round-trip drift on ordinary neutral or ionic
                    # molecules should still be cleaned up to the intended formal
                    # charge. But if the formal charge strongly disagrees with the
                    # existing charge set, keep the current charge sum instead.
                    if abs(float(current_q) - float(formal_q)) <= 0.25:
                        target_q = float(formal_q)
                    else:
                        target_q = float(current_q)

            corr = correct_total_charge(mol, target_q=target_q)
            if corr is not None:
                meta["charge_correction"] = corr
                # Update charge fingerprint after correction
                ch_abs2, ch_sig2 = _mol_charge_abs_and_sig(mol)
                meta["charge_abs_sum"] = float(ch_abs2)
                meta["charge_signature"] = str(ch_sig2)
                (out_dir / "meta.json").write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
        except Exception:
            pass

        # Ensure bonded terms exist (angles/dihedrals) before writing .itp
        _ensure_bonded_terms_for_export(mol, ff_name)
        _reapply_bonded_patch_to_mol(mol)

        from .gromacs_molecule import write_gromacs_single_molecule_topology

        # Always write *unscaled* charges to the cached topology.
        write_gromacs_single_molecule_topology(mol, out_dir, mol_name=mol_name)

        # Sanity check: must produce at least one .itp and one .gro
        if not any(out_dir.glob('*.itp')):
            raise RuntimeError('No .itp file was produced.')
        if not any(out_dir.glob('*.gro')):
            raise RuntimeError('No .gro file was produced.')
    except Exception as e:
        # Update metadata with the failure reason and re-raise with context
        meta["gromacs_error"] = str(e)
        (out_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        # Also write a short error file for quick inspection
        try:
            import traceback

            (out_dir / "gromacs_error.txt").write_text(
                traceback.format_exc() + "\n", encoding="utf-8"
            )
        except Exception:
            pass
        raise RuntimeError(f"Failed to write GROMACS artifacts under {out_dir}: {e}") from e
    # Charged MOL2 export (best effort)
    if write_mol2:
        try:
            from .mol2 import write_mol2

            mol2_dir = (mol2_root if mol2_root is not None else (out_dir / "charged_mol2"))
            mol2_dir.mkdir(parents=True, exist_ok=True)
            # Always write the current (possibly scaled) charges
            write_mol2(mol=mol, out_mol2=mol2_dir / f"{mol_name}.mol2", mol_name=mol_name)
            # If raw charges exist, also write a raw file for reference
            if any(a.HasProp("AtomicCharge_raw") or a.HasProp("RESP_raw") for a in mol.GetAtoms()):
                write_mol2(
                    mol=mol,
                    out_mol2=mol2_dir / f"{mol_name}.raw.mol2",
                    mol_name=mol_name,
                    use_raw=True,
                )
        except Exception:
            pass

    # SDF geometry (best effort; helpful for debugging)
    try:
        from rdkit import Chem

        w = Chem.SDWriter(str(out_dir / "mol.sdf"))
        w.write(mol)
        w.close()
    except Exception:
        # RDKit might not be installed in some environments; keep metadata only.
        pass

    return out_dir
