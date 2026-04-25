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


def _stable_signature(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _atom_order_signature(mol) -> str:
    atoms: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        atoms.append(
            {
                "idx": int(atom.GetIdx()),
                "atomic_num": int(atom.GetAtomicNum()),
                "formal_charge": int(atom.GetFormalCharge()),
                "isotope": int(atom.GetIsotope()),
                "radicals": int(atom.GetNumRadicalElectrons()),
                "aromatic": bool(atom.GetIsAromatic()),
                "ff_type": (str(atom.GetProp("ff_type")) if atom.HasProp("ff_type") else ""),
                "residue_number": (int(info.GetResidueNumber()) if info is not None else None),
                "residue_name": (str(info.GetResidueName()).strip() if info is not None else ""),
                "atom_name": (str(info.GetName()).strip() if info is not None else ""),
            }
        )
    bonds = [
        {
            "begin": int(bond.GetBeginAtomIdx()),
            "end": int(bond.GetEndAtomIdx()),
            "order": float(bond.GetBondTypeAsDouble()),
            "aromatic": bool(bond.GetIsAromatic()),
        }
        for bond in mol.GetBonds()
    ]
    return _stable_signature({"atoms": atoms, "bonds": bonds})


def _charge_group_signature(groups: Any) -> str | None:
    if not isinstance(groups, list) or not groups:
        return None

    normalized: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        atom_indices = tuple(int(i) for i in group.get("atom_indices", []) if i is not None)
        if not atom_indices:
            continue
        normalized.append(
            {
                "atom_indices": list(atom_indices),
                "formal_charge": int(group.get("formal_charge", 0)),
                "label": str(group.get("label") or ""),
                "source": str(group.get("source") or ""),
            }
        )
    if not normalized:
        return None
    normalized.sort(
        key=lambda item: (
            tuple(item["atom_indices"]),
            item["formal_charge"],
            item["label"],
            item["source"],
        )
    )
    return _stable_signature(normalized)


def _residue_signature(residue_map: Any) -> str | None:
    if not isinstance(residue_map, dict):
        return None
    residues_raw = residue_map.get("residues")
    atoms_raw = residue_map.get("atoms")
    if not isinstance(residues_raw, list) or not isinstance(atoms_raw, list):
        return None

    residues = []
    for residue in residues_raw:
        if not isinstance(residue, dict):
            continue
        residues.append(
            {
                "residue_number": int(residue.get("residue_number", 0)),
                "residue_name": str(residue.get("residue_name") or ""),
                "atom_indices": [int(i) for i in residue.get("atom_indices", []) if i is not None],
            }
        )

    atoms = []
    for atom in atoms_raw:
        if not isinstance(atom, dict):
            continue
        atoms.append(
            {
                "atom_index": int(atom.get("atom_index", 0)),
                "residue_number": int(atom.get("residue_number", 0)),
                "residue_name": str(atom.get("residue_name") or ""),
                "atom_name": str(atom.get("atom_name") or ""),
            }
        )

    if not residues and not atoms:
        return None
    return _stable_signature({"residues": residues, "atoms": atoms})


def _molecule_compatibility_context(mol, *, mol_name: str | None = None) -> Dict[str, Any]:
    context: Dict[str, Any] = {}

    try:
        context["atom_order_signature"] = _atom_order_signature(mol)
    except Exception:
        pass

    residue_map = None
    try:
        from ..core.polyelectrolyte import build_residue_map

        residue_map = build_residue_map(mol, mol_name=mol_name)
    except Exception:
        residue_map = None

    if residue_map is not None:
        try:
            context["_residue_count"] = len(list(residue_map.get("residues") or []))
        except Exception:
            context["_residue_count"] = 0
        residue_sig = _residue_signature(residue_map)
        if residue_sig:
            context["residue_signature"] = residue_sig
    else:
        context["_residue_count"] = 0

    try:
        from ..core.polyelectrolyte import (
            get_charge_groups,
            get_polyelectrolyte_summary,
            uses_localized_charge_groups,
        )

        groups = get_charge_groups(mol)
        group_sig = _charge_group_signature(groups)
        if group_sig:
            context["charge_group_signature"] = group_sig
        summary = get_polyelectrolyte_summary(mol)
        context["_localized_charge_groups"] = bool(uses_localized_charge_groups(summary))
    except Exception:
        context["_localized_charge_groups"] = False

    try:
        if mol.HasProp("_yadonpy_resp_profile"):
            profile = str(mol.GetProp("_yadonpy_resp_profile")).strip().lower()
            if profile:
                context["resp_profile"] = profile
    except Exception:
        pass

    for prop, context_key in (
        ("_yadonpy_qm_recipe_json", "qm_recipe_signature"),
        ("_yadonpy_resp_constraints_json", "resp_constraints_signature"),
        ("_yadonpy_psiresp_constraints", "psiresp_constraints_signature"),
    ):
        try:
            if not mol.HasProp(prop):
                continue
            raw = str(mol.GetProp(prop)).strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                payload = raw
            context[context_key] = _stable_signature(payload)
        except Exception:
            continue

    return context


def _artifact_meta_compatibility_fields(mol, *, mol_name: str | None = None) -> Dict[str, str]:
    context = _molecule_compatibility_context(mol, mol_name=mol_name)
    meta: Dict[str, str] = {}
    for key in (
        "atom_order_signature",
        "charge_group_signature",
        "residue_signature",
        "resp_profile",
        "qm_recipe_signature",
        "resp_constraints_signature",
        "psiresp_constraints_signature",
    ):
        value = context.get(key)
        if value:
            meta[key] = str(value)
    return meta


def _prefers_order_sensitive_artifact_cache(mol, *, mol_name: str | None = None) -> bool:
    context = _molecule_compatibility_context(mol, mol_name=mol_name)
    if not context.get("charge_group_signature"):
        return False
    return bool(context.get("_localized_charge_groups")) or int(context.get("_residue_count", 0)) > 1


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
    """Ensure bond/angle/dihedral parameters exist before writing .itp.

    In some workflows (notably mixed-system exports), molecules may be obtained via RDKit fragment
    extraction or serialization that drops Python-level attributes and sometimes bond-level
    properties. This helper rebuilds the missing GAFF-family bonded terms using the requested
    force-field name, without changing charges.
    """
    try:
        nat = int(mol.GetNumAtoms())
    except Exception:
        nat = 0
    if nat < 3:
        return

    has_angles = bool(getattr(mol, "angles", {}) or {})
    has_dihedrals = bool(getattr(mol, "dihedrals", {}) or {})
    has_valid_bonds = True
    try:
        for bond in mol.GetBonds():
            if (not bond.HasProp("ff_r0")) or (not bond.HasProp("ff_k")):
                has_valid_bonds = False
                break
            if float(bond.GetDoubleProp("ff_r0")) <= 0.0 or float(bond.GetDoubleProp("ff_k")) <= 0.0:
                has_valid_bonds = False
                break
    except Exception:
        has_valid_bonds = False

    atoms_have_types = True
    try:
        for atom in mol.GetAtoms():
            if not atom.HasProp("ff_type"):
                atoms_have_types = False
                break
    except Exception:
        atoms_have_types = False

    if has_valid_bonds and has_angles and (nat < 4 or has_dihedrals):
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
        elif ff_name_l == "oplsaa":
            from ..ff.oplsaa import OPLSAA
            ff_obj = OPLSAA()
    except Exception:
        ff_obj = None

    if ff_obj is None:
        return

    # Rebuild only missing force-field typing/bonded terms; keep charges untouched.
    try:
        if (not atoms_have_types) and hasattr(ff_obj, "assign_ptypes"):
            ff_obj.assign_ptypes(mol)
    except Exception:
        pass
    try:
        if (not has_valid_bonds) and hasattr(ff_obj, "assign_btypes"):
            ff_obj.assign_btypes(mol)
    except Exception:
        pass
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

    # Repair charge properties before fingerprinting or writing topology. This is
    # a safety net for old charge caches that carry adaptive RESP metadata but
    # predate explicit post-fit property synchronization.
    try:
        core_utils.symmetrize_equivalent_charge_props(mol)
    except Exception:
        pass

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
    meta.update(_artifact_meta_compatibility_fields(mol, mol_name=mol_name))
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
            from ..core.polyelectrolyte import get_polyelectrolyte_summary, uses_localized_charge_groups

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

                    localized_charge_groups = False
                    try:
                        localized_charge_groups = bool(
                            uses_localized_charge_groups(get_polyelectrolyte_summary(mol))
                        )
                    except Exception:
                        localized_charge_groups = False

                    # Localized polyelectrolytes (e.g. CMC-like carboxylates)
                    # rely on formal charged-group counts being authoritative for
                    # later grouped scaling during system export. If the current
                    # per-atom charge sum has drifted away from that integer total,
                    # preserve the formal charge here instead of freezing the bad
                    # current total into the cached ITP.
                    if localized_charge_groups:
                        target_q = float(formal_q)
                        meta["charge_target_policy"] = "formal_charge_for_localized_polyelectrolyte"
                    # Small RESP / round-trip drift on ordinary neutral or ionic
                    # molecules should still be cleaned up to the intended formal
                    # charge. But if the formal charge strongly disagrees with the
                    # existing charge set, keep the current charge sum instead.
                    elif abs(float(current_q) - float(formal_q)) <= 0.25:
                        target_q = float(formal_q)
                    else:
                        target_q = float(current_q)

            corr = correct_total_charge(mol, target_q=target_q)
            if corr is not None:
                meta["charge_correction"] = corr
                try:
                    # Uniform net-charge correction should preserve equality, but
                    # run the same metadata-driven repair after any charge mutation
                    # so future strategies cannot desynchronize equivalent atoms.
                    core_utils.symmetrize_equivalent_charge_props(mol)
                except Exception:
                    pass
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
