"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None

from ..core import chem_utils as core_utils
from ..core.polyelectrolyte import (
    annotate_polyelectrolyte_metadata,
    build_residue_map,
    get_charge_groups,
    get_polyelectrolyte_summary,
    get_resp_constraints,
    uses_localized_charge_groups,
)
from ..gmx.topology import parse_system_top
from ..gmx.analysis.structured import build_site_map
from ..schema_versions import EXPORT_SYSTEM_SCHEMA_VERSION
from ..workflow.resume import file_signature

def _mol_signature(m: Chem.Mol) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Return a simple, order-independent signature for a molecule.

    RDKit's `GetMolFrags(..., asMols=True)` does **not** guarantee the order of
    returned fragments. For exporting a packed amorphous cell, we therefore match
    packed fragments back to the requested species using a lightweight signature.

    The signature is:
      (num_atoms, sorted((atomic_number, count), ...))

    This is usually sufficient to disambiguate common solvents/ions in electrolyte
    systems. If your system contains multiple species with identical signatures,
    the exporter may still need a stronger matcher (e.g., canonical SMILES).
    """
    from collections import Counter

    nums = [a.GetAtomicNum() for a in m.GetAtoms()]
    c = Counter(nums)
    return (len(nums), tuple(sorted((int(k), int(v)) for k, v in c.items())))


def _artifact_digest(artifact_dir: Path, moltype: str) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    payload: dict[str, Any] = {"artifact_dir": str(artifact_dir), "moltype": str(moltype)}
    for suffix in (".itp", ".gro", ".top"):
        fp = artifact_dir / f"{moltype}{suffix}"
        if fp.exists():
            payload[suffix.lstrip(".")] = file_signature(fp)
    return payload


def _requires_charge_groups(species_payload: Mapping[str, Any]) -> bool:
    summary = species_payload.get("polyelectrolyte_summary")
    if isinstance(summary, Mapping):
        return bool(uses_localized_charge_groups(dict(summary)))
    if bool(species_payload.get("polyelectrolyte_mode")):
        return True
    return False


def _enrich_species_polyelectrolyte_payload(
    payload: dict[str, Any],
    *,
    smiles: str,
    mol_name: str,
    mol=None,
) -> dict[str, Any]:
    summary = payload.get("polyelectrolyte_summary")
    charge_groups = payload.get("charge_groups")
    needs_probe = (
        mol is not None
        or bool(payload.get("polyelectrolyte_mode"))
        or not isinstance(summary, Mapping)
        or not isinstance(charge_groups, list)
        or (isinstance(summary, Mapping) and not bool(summary))
    )
    if not needs_probe:
        return payload

    probe = mol
    if probe is None:
        try:
            probe = _parse_smiles_for_metadata(smiles)
        except Exception:
            probe = None
    if probe is None:
        return payload

    try:
        regen_groups = get_charge_groups(probe)
        regen_constraints = get_resp_constraints(probe)
        regen_summary = get_polyelectrolyte_summary(probe)
        localized = bool(uses_localized_charge_groups(regen_summary))
    except Exception:
        return payload

    if isinstance(regen_groups, list):
        payload["charge_groups"] = regen_groups
    if isinstance(regen_constraints, Mapping):
        payload["resp_constraints"] = dict(regen_constraints)
    if isinstance(regen_summary, Mapping):
        payload["polyelectrolyte_summary"] = dict(regen_summary)
    payload["polyelectrolyte_mode"] = bool(localized)

    try:
        payload["residue_map"] = build_residue_map(probe, mol_name=mol_name)
    except Exception:
        pass
    return payload


def _ensure_charge_group_ready(species_payload: Mapping[str, Any], *, effective_polyelectrolyte_mode: bool) -> None:
    if not effective_polyelectrolyte_mode:
        return
    if not _requires_charge_groups(species_payload):
        return
    groups = species_payload.get("charge_groups")
    if isinstance(groups, list) and groups:
        return
    moltype = str(species_payload.get("moltype") or species_payload.get("mol_name") or species_payload.get("mol_id") or "UNKNOWN")
    raise RuntimeError(
        f"Polyelectrolyte-aware export requires charge_groups for species '{moltype}', "
        "but no grouped metadata were found. Re-run charge assignment / MolDB generation with polyelectrolyte_mode=True."
    )


def _uses_charge_group_scaling(species_payload: Mapping[str, Any], *, effective_polyelectrolyte_mode: bool) -> bool:
    """Return True only for species that genuinely need grouped charge scaling.

    Ordinary ions such as TFSI- can carry incidental ``charge_groups`` metadata from
    graph detection, but those groups are diagnostic and must not trigger
    polyelectrolyte-style re-targeting of RESP charges. Group scaling is reserved for
    species that explicitly require polyelectrolyte-aware export semantics.
    """
    if not effective_polyelectrolyte_mode:
        return False
    if not _requires_charge_groups(species_payload):
        return False
    groups = species_payload.get("charge_groups")
    return bool(isinstance(groups, list) and groups)

from ..api import get_ff
from .artifacts import write_molecule_artifacts


@dataclass
class SpeciesExportRecord:
    payload: dict[str, Any]
    artifact_dir: Path
    mol_id: str
    mol_name: str
    moltype: str
    formal_charge: int
    kind: str
    source_artifact_digest: dict[str, Any] | None = None


@dataclass
class ChargeScalingDecision:
    moltype: str
    scale: float
    used_group_scaling: bool
    report: dict[str, Any]


@dataclass
class SystemAssemblyPlan:
    out_dir: Path
    molecules_dir: Path
    source_molecules_dir: Path | None
    system_gro_template: Path | None
    system_ndx_template: Path | None
    effective_polyelectrolyte_mode: bool


def _gro_wrap_index(value: int) -> int:
    value = int(value)
    if value < 0:
        return -((-value) % 100000)
    return value % 100000


def _format_gro_atom_line(*, resnr: int, resname: str, atomname: str, atomnr: int, x: float, y: float, z: float) -> str:
    return (
        f"{_gro_wrap_index(resnr):5d}{str(resname)[:5]:<5}{str(atomname)[:5]:>5}{_gro_wrap_index(atomnr):5d}"
        f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
    )


def canonicalize_smiles(smiles_or_psmiles: str) -> str:
    """Canonicalize SMILES; keep PSMILES (contains '*') as-is.

    This helper replaces the removed basic_top/library canonicalizer.
    """
    s = str(smiles_or_psmiles or "").strip()
    if not s:
        return ""
    if "*" in s:
        return s
    if Chem is None:
        return s
    try:
        if _prefer_unsanitized_smiles_parse(s):
            # Preserve the caller-provided representation for hypervalent /
            # inorganic ion-like species. RDKit can emit distorted canonical
            # strings for these graphs (e.g. PF6-) even when the molecule is
            # otherwise usable for metadata matching.
            return s
        m = _parse_smiles_for_metadata(s)
        if m is None:
            return s
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return s


def _prefer_unsanitized_smiles_parse(smiles: str) -> bool:
    if Chem is None:
        return False
    try:
        probe = Chem.MolFromSmiles(str(smiles), sanitize=False)
    except Exception:
        probe = None
    if probe is None:
        return False
    try:
        probe.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        if core_utils.is_high_symmetry_polyhedral_ion(probe, smiles_hint=str(smiles)):
            return True
        if core_utils.is_inorganic_ion_like(probe, smiles_hint=str(smiles)):
            return True
    except Exception:
        pass
    return False


def _parse_smiles_for_metadata(smiles: str):
    if Chem is None:
        return None
    s = str(smiles or "").strip()
    if not s:
        return None
    prefer_unsanitized = _prefer_unsanitized_smiles_parse(s)
    if prefer_unsanitized:
        try:
            mol = Chem.MolFromSmiles(s, sanitize=False)
        except Exception:
            mol = None
        if mol is None:
            return None
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            pass
        return mol
    try:
        mol = Chem.MolFromSmiles(s)
    except Exception:
        mol = None
    if mol is not None:
        return mol
    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)
    except Exception:
        mol = None
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    except Exception:
        pass
    return mol


def _species_signature_from_smiles(smiles: str) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    """Build a fragment-matching signature from SMILES without over-sanitizing ions."""
    if Chem is None:
        return None
    try:
        mol = _parse_smiles_for_metadata(str(smiles or ""))
    except Exception:
        mol = None
    if mol is None:
        return None
    try:
        return _mol_signature(mol)
    except Exception:
        return None


def _fs_safe_mol_name(name: str) -> str:
    """Convert a molecule name into a filesystem- and GROMACS-friendly token.

    Requirements:
      - Stable mapping from user-facing names/variable names to artifact filenames.
      - Works for common ion labels like "Li+", "PF6-".
      - Avoids non-ASCII / whitespace / path separators.

    Notes:
      - Python variable names (e.g., EC, polymer_A) are already safe, so this
        mostly protects users who explicitly set custom names.
    """
    s = str(name or "").strip()
    # Prevent path traversal and odd separators.
    s = s.replace("/", "_").replace("\\", "_")
    # Keep common safe characters. Replace everything else with "_".
    s = re.sub(r"[^A-Za-z0-9._+\-]+", "_", s)
    s = s.strip("_")
    return s or "MOL"


_TOPOLOGY_PARAMETER_SECTIONS = {
    "defaults",
    "atomtypes",
    "bondtypes",
    "constrainttypes",
    "angletypes",
    "dihedraltypes",
    "impropertypes",
    "improper_types",
    "pairtypes",
    "nonbond_params",
    "cmaptypes",
}


def _topology_section_name(line: str) -> str | None:
    stripped = str(line or "").strip().lower()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    return stripped.strip("[]").strip() or None


def _split_itp_params_and_mol_text(itp_text: str) -> tuple[str, str]:
    """Split an ITP into parameter text before the first [moleculetype] and the molecule block."""
    lines = itp_text.splitlines()
    mt_idx = None
    for i, ln in enumerate(lines):
        if _topology_section_name(ln) == "moleculetype":
            mt_idx = i
            break
    if mt_idx is None:
        return "", itp_text
    params = "\n".join(lines[:mt_idx]).strip()
    molblk = "\n".join(lines[mt_idx:]).strip()
    return ((params + "\n") if params else ""), ((molblk + "\n") if molblk else "")


def _iter_relevant_topology_directives(path: Path, *, seen: set[Path] | None = None):
    visited = seen if seen is not None else set()
    resolved = path.resolve()
    if resolved in visited or not resolved.exists():
        return
    visited.add(resolved)

    for raw in resolved.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.lower().startswith("#include"):
            try:
                rel = line.split(None, 1)[1].strip().strip('"')
            except Exception:
                continue
            inc = (resolved.parent / rel).resolve()
            yield from _iter_relevant_topology_directives(inc, seen=visited)
            continue
        if line.startswith("#"):
            continue
        section = _topology_section_name(line)
        if section in _TOPOLOGY_PARAMETER_SECTIONS or section in {"moleculetype", "system", "molecules"}:
            yield section, resolved


def _validate_topology_include_order(top_path: Path) -> list[str]:
    stage_rank = {
        "defaults": 0,
        "atomtypes": 1,
        "bondtypes": 1,
        "constrainttypes": 1,
        "angletypes": 1,
        "dihedraltypes": 1,
        "impropertypes": 1,
        "improper_types": 1,
        "pairtypes": 1,
        "nonbond_params": 1,
        "cmaptypes": 1,
        "moleculetype": 2,
        "system": 3,
        "molecules": 4,
    }
    directives = list(_iter_relevant_topology_directives(top_path))
    issues: list[str] = []
    first_moleculetype_idx = next((i for i, (section, _) in enumerate(directives) if section == "moleculetype"), None)
    if first_moleculetype_idx is not None:
        has_defaults_before_moleculetype = any(
            section == "defaults" for section, _ in directives[:first_moleculetype_idx]
        )
        if not has_defaults_before_moleculetype:
            issues.append("missing [ defaults ] before first [ moleculetype ]")
    highest_stage = -1
    for idx, (section, src_path) in enumerate(directives):
        rank = stage_rank.get(section)
        if rank is None:
            continue
        if section == "defaults" and idx != 0:
            prev_section, prev_src = directives[idx - 1]
            issues.append(
                f"[ defaults ] must be the first directive, but appears after [{prev_section}] in {src_path.name} (previous directive from {prev_src.name})"
            )
        if rank < highest_stage:
            if section in _TOPOLOGY_PARAMETER_SECTIONS and highest_stage >= 2:
                issues.append(f"parameter section [{section}] appears after [ moleculetype ] in {src_path.name}")
            elif section == "moleculetype" and highest_stage >= 3:
                issues.append(f"[ moleculetype ] appears after top-level system directives in {src_path.name}")
            elif section == "system" and highest_stage >= 4:
                issues.append(f"[ system ] appears after [ molecules ] in {src_path.name}")
            else:
                issues.append(f"directive order regressed at [{section}] in {src_path.name}")
        highest_stage = max(highest_stage, rank)
    return issues


def validate_topology_include_order(top_path: Path | str) -> list[str]:
    return _validate_topology_include_order(Path(top_path).expanduser().resolve())


def validate_exported_system_dir(out_dir: Path | str) -> list[str]:
    root = Path(out_dir).expanduser().resolve()
    top_path = root / "system.top"
    issues: list[str] = []
    if not top_path.is_file():
        return [f"missing topology file: {top_path}"]
    issues.extend(validate_topology_include_order(top_path))
    for line in top_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s.lower().startswith("#include"):
            continue
        try:
            rel = s.split(None, 1)[1].strip().strip('"')
        except Exception:
            continue
        inc = (top_path.parent / rel).resolve()
        if not inc.exists():
            issues.append(f"missing include file: {inc}")
    return issues


def _ensure_bonded_terms_from_types(mol, ff_name: str) -> None:
    """Ensure mol.angles/mol.dihedrals exist by re-running GAFF-family angle/dihedral typing.

    This is critical when a representative molecule is obtained via Chem.GetMolFrags on a combined
    cell: RDKit fragment molecules preserve atom/bond properties (ff_type, ff_r0, ff_k, etc.) but
    do not carry Python-level containers like mol.angles/mol.dihedrals. Without rebuilding these,
    exported .itp files would contain only [ bonds ] and miss [ angles ]/[ dihedrals ].
    """
    try:
        nat = int(mol.GetNumAtoms())
    except Exception:
        nat = 0

    # If angles/dihedrals are already present, do nothing.
    has_angles = bool(getattr(mol, "angles", {}) or {})
    has_dihedrals = bool(getattr(mol, "dihedrals", {}) or {})

    if nat < 3 or (has_angles and (nat < 4 or has_dihedrals)):
        return

    try:
        ff = get_ff(str(ff_name))
    except Exception:
        return

    # Only (re)assign higher-order terms; do not redo ptypes/btypes (already present).
    try:
        if nat >= 3 and not has_angles and hasattr(ff, "assign_atypes"):
            ff.assign_atypes(mol)
    except Exception:
        pass
    try:
        if nat >= 4 and not has_dihedrals and hasattr(ff, "assign_dtypes"):
            ff.assign_dtypes(mol)
    except Exception:
        pass
    try:
        if hasattr(ff, "assign_itypes"):
            # impropers are optional for many molecules, but keep consistent if available
            ff.assign_itypes(mol)
    except Exception:
        pass

    _reapply_bonded_patch_to_mol(mol)


def _reapply_bonded_patch_to_mol(mol) -> None:
    """Best-effort reapply of bonded JSON patch after angle/dihedral rebuilds."""
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

@dataclass
class SystemExportResult:
    system_gro: Path
    system_top: Path
    system_ndx: Path
    molecules_dir: Path
    system_meta: Path
    box_nm: float
    species: list[dict]
    box_lengths_nm: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class _GroSpeciesTemplate:
    species: dict
    name: str
    gro_path: Path
    atom_names: tuple[str, ...]
    coords0: np.ndarray


def _load_gro_species_templates(
    species: list[dict],
    mol_names: list[str],
    mol_gro_paths: list[Path],
) -> tuple[list[_GroSpeciesTemplate], int]:
    templates: list[_GroSpeciesTemplate] = []
    nat_total = 0
    for sp, name, gro_path in zip(species, mol_names, mol_gro_paths):
        atom_names, coords0 = _read_gro_single_molecule(gro_path)
        tpl = _GroSpeciesTemplate(
            species=sp,
            name=str(name or ''),
            gro_path=gro_path,
            atom_names=tuple(atom_names),
            coords0=np.asarray(coords0, dtype=float),
        )
        templates.append(tpl)
        nat_total += int(sp.get('n', 0)) * len(tpl.atom_names)
    return templates, nat_total


def _flush_gro_lines(handle, line_buffer: list[str]) -> None:
    if line_buffer:
        handle.write(''.join(line_buffer))
        line_buffer.clear()


def _normalize_box_lengths_nm(values: Any) -> tuple[float, float, float] | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)) and len(values) >= 3:
        try:
            return (float(values[0]), float(values[1]), float(values[2]))
        except Exception:
            return None
    try:
        edge = float(values)
    except Exception:
        return None
    if edge <= 0.0:
        return None
    return (edge, edge, edge)


def _read_gro_box_lengths_nm(path: Path) -> tuple[float, float, float] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) < 3:
            return None
        parts = lines[-1].split()
        if len(parts) < 3:
            return None
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except Exception:
        return None


def _cell_box_lengths_nm_from_rdkit(cell_mol) -> tuple[float, float, float] | None:
    try:
        cell = getattr(cell_mol, "cell", None)
        if cell is None:
            return None
        return (
            0.1 * (float(cell.xhi) - float(cell.xlo)),
            0.1 * (float(cell.yhi) - float(cell.ylo)),
            0.1 * (float(cell.zhi) - float(cell.zlo)),
        )
    except Exception:
        return None


def _cell_box_origin_nm_from_rdkit(cell_mol) -> tuple[float, float, float] | None:
    try:
        cell = getattr(cell_mol, "cell", None)
        if cell is None:
            return None
        return (
            0.1 * float(cell.xlo),
            0.1 * float(cell.ylo),
            0.1 * float(cell.zlo),
        )
    except Exception:
        return None


def _read_gro_single_molecule(gro_path: Path) -> Tuple[List[str], np.ndarray]:
    """Read a single-molecule .gro produced by yadonpy into (atom_names, coords_nm)."""
    lines = gro_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid gro file: {gro_path}")
    nat = int(lines[1].strip())
    if len(lines) < 2 + nat:
        raise ValueError(f"Invalid gro file (atom count mismatch): {gro_path}")
    atom_names: List[str] = []
    coords = np.zeros((nat, 3), dtype=float)
    for i in range(nat):
        l = lines[2 + i]
        # fixed columns: resnr(5) resname(5) atomname(5) atomnr(5) x(8) y(8) z(8)
        atomname = l[10:15].strip() or f"A{i+1}"
        atom_names.append(atomname)
        coords[i, 0] = float(l[20:28])
        coords[i, 1] = float(l[28:36])
        coords[i, 2] = float(l[36:44])
    return atom_names, coords


def _scale_itp_charges(itp_text: str, scale: float, atom_indices: list[int] | None = None) -> str:
    """Scale charges in the `[ atoms ]` section of an ITP.

    We keep the rest of the file untouched. This is used for simulation-level
    dielectric charge scaling without modifying the persistent library.

    The GROMACS [ atoms ] format is typically:
      nr  type  resnr  resid  atom  cgnr  charge  mass
    """
    s = float(scale)
    if abs(s - 1.0) < 1.0e-12:
        return itp_text

    selected = None if atom_indices is None else {int(i) + 1 for i in atom_indices}
    out_lines: list[str] = []
    in_atoms = False
    atom_nr = None
    for raw in itp_text.splitlines():
        line = raw
        stripped = line.strip()

        # Section switching
        if stripped.startswith("[") and stripped.endswith("]"):
            sec = stripped.strip("[]").strip().lower()
            in_atoms = (sec == "atoms")
            out_lines.append(raw)
            continue

        if not in_atoms:
            out_lines.append(raw)
            continue

        # Keep comments/blank lines
        if stripped == "" or stripped.startswith(";"):
            out_lines.append(raw)
            continue

        # Split off trailing comment
        body, *comment = raw.split(";", 1)
        cols = body.split()
        if len(cols) >= 7:
            try:
                atom_nr = int(cols[0])
                if selected is not None and atom_nr not in selected:
                    out_lines.append(raw)
                    continue
                q = float(cols[6])
                cols[6] = f"{q * s:.8f}"
                body2 = "\t".join(cols)
                if comment:
                    out_lines.append(body2 + " ;" + comment[0])
                else:
                    out_lines.append(body2)
                continue
            except Exception:
                # fall back to original
                pass
        out_lines.append(raw)

    return "\n".join(out_lines) + ("\n" if itp_text.endswith("\n") else "")


def _scale_itp_charge_groups(itp_text: str, groups: list[dict[str, Any]], scale: float) -> tuple[str, dict[str, Any]]:
    if abs(float(scale) - 1.0) < 1.0e-12 or not groups:
        return itp_text, {"scale": float(scale), "groups": [], "fallback": None}

    lines = itp_text.splitlines()
    in_atoms = False
    atom_rows: dict[int, dict[str, Any]] = {}
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            sec = stripped.strip("[]").strip().lower()
            in_atoms = sec == "atoms"
            continue
        if not in_atoms or stripped == "" or stripped.startswith(";"):
            continue
        body = raw.split(";", 1)[0]
        cols = body.split()
        if len(cols) < 7:
            continue
        try:
            atom_no = int(cols[0])
            atom_rows[atom_no] = {"line_index": idx, "cols": cols}
        except Exception:
            continue

    report = {"scale": float(scale), "groups": [], "fallback": None}
    for grp in groups:
        group_indices = [int(i) + 1 for i in grp.get("atom_indices", [])]
        if not group_indices:
            continue
        present = [i for i in group_indices if i in atom_rows]
        if not present:
            continue
        orig = []
        for atom_no in present:
            orig.append(float(atom_rows[atom_no]["cols"][6]))
        orig_total = float(sum(orig))
        target_total = float(grp.get("formal_charge", 0.0)) * float(scale)
        if abs(orig_total) > 1.0e-12:
            factor = target_total / orig_total
            new_vals = [q * factor for q in orig]
        else:
            delta = target_total / float(len(present))
            new_vals = [delta for _ in orig]
        for atom_no, q in zip(present, new_vals):
            atom_rows[atom_no]["cols"][6] = f"{float(q):.8f}"
        report["groups"].append(
            {
                "group_id": grp.get("group_id"),
                "atom_indices": [int(i) - 1 for i in present],
                "formal_charge": int(grp.get("formal_charge", 0)),
                "original_total_charge": float(orig_total),
                "scaled_total_charge": float(sum(new_vals)),
                "target_total_charge": float(target_total),
            }
        )

    for atom_no, info in atom_rows.items():
        raw = lines[info["line_index"]]
        comment = ""
        if ";" in raw:
            _, tail = raw.split(";", 1)
            comment = " ;" + tail
        lines[info["line_index"]] = "\t".join(info["cols"]) + comment
    return ("\n".join(lines) + ("\n" if itp_text.endswith("\n") else "")), report


def _itp_total_charge(itp_text: str) -> float:
    total = 0.0
    in_atoms = False
    for raw in itp_text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            sec = stripped.strip("[]").strip().lower()
            in_atoms = (sec == "atoms")
            continue
        if not in_atoms or stripped == "" or stripped.startswith(";"):
            continue
        cols = raw.split(";", 1)[0].split()
        if len(cols) < 7:
            continue
        try:
            total += float(cols[6])
        except Exception:
            continue
    return float(total)


def _nudge_itp_first_atom_charge(itp_text: str, delta_q: float) -> str:
    if abs(float(delta_q)) <= 1.0e-12:
        return itp_text

    out_lines: list[str] = []
    in_atoms = False
    updated = False
    for raw in itp_text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            sec = stripped.strip("[]").strip().lower()
            in_atoms = (sec == "atoms")
            out_lines.append(raw)
            continue
        if not in_atoms or updated or stripped == "" or stripped.startswith(";"):
            out_lines.append(raw)
            continue
        body, *comment = raw.split(";", 1)
        cols = body.split()
        if len(cols) >= 7:
            try:
                cols[6] = f"{float(cols[6]) + float(delta_q):.8f}"
                new_body = "\t".join(cols)
                if comment:
                    out_lines.append(new_body + " ;" + comment[0])
                else:
                    out_lines.append(new_body)
                updated = True
                continue
            except Exception:
                pass
        out_lines.append(raw)
    if not updated:
        raise ValueError("Failed to adjust the first [ atoms ] charge in the exported ITP")
    return "\n".join(out_lines) + ("\n" if itp_text.endswith("\n") else "")


def _neutralize_exported_system_charge(*, species: list[dict], mol_names: list[str], mol_itp_paths: list[Path], tol: float = 0.1) -> dict[str, Any] | None:
    if not species or not mol_itp_paths:
        return None

    system_charge = 0.0
    per_molecule_charge: dict[str, float] = {}
    for sp, name, itp_path in zip(species, mol_names, mol_itp_paths):
        charge = _itp_total_charge(itp_path.read_text(encoding="utf-8", errors="replace"))
        per_molecule_charge[name] = float(charge)
        system_charge += float(charge) * int(sp.get("n", 0))

    if abs(float(system_charge)) <= 1.0e-12 or abs(float(system_charge)) > float(tol):
        return None

    target_idx = None
    for idx, sp in enumerate(species):
        if int(sp.get("n", 0)) > 0:
            target_idx = idx
            break
    if target_idx is None:
        return None

    target_count = int(species[target_idx].get("n", 0))
    if target_count <= 0:
        return None

    target_name = mol_names[target_idx]
    target_itp = mol_itp_paths[target_idx]
    delta_q = -float(system_charge) / float(target_count)
    updated_text = _nudge_itp_first_atom_charge(
        target_itp.read_text(encoding="utf-8", errors="replace"),
        delta_q,
    )
    target_itp.write_text(updated_text, encoding="utf-8")

    corrected_charge = _itp_total_charge(updated_text)
    system_charge_after = 0.0
    for idx, (sp, name) in enumerate(zip(species, mol_names)):
        charge = corrected_charge if idx == target_idx else per_molecule_charge[name]
        system_charge_after += float(charge) * int(sp.get("n", 0))

    return {
        "applied": True,
        "tolerance": float(tol),
        "system_charge_before": float(system_charge),
        "system_charge_after": float(system_charge_after),
        "target_moltype": target_name,
        "target_count": int(target_count),
        "delta_per_molecule": float(delta_q),
    }


def _random_rotation_matrix() -> np.ndarray:
    """Uniform random rotation matrix."""
    u1, u2, u3 = random.random(), random.random(), random.random()
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    # quaternion to rotation matrix
    return np.array(
        [
            [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
        ],
        dtype=float,
    )


def _estimate_box_nm(species: list[dict], density_g_cm3: float) -> float:
    """Estimate cubic box length (nm) from total mass and density."""
    if Chem is None or Descriptors is None:
        # fallback: a conservative small box
        return 10.0

    from ..core.molspec import molecular_weight

    na = 6.02214076e23
    total_mass_g = 0.0
    for sp in species:
        smi = sp.get("smiles", "")
        n = int(sp.get("n", 0))
        if n <= 0:
            continue
        try:
            mol = _parse_smiles_for_metadata(smi)
            if mol is None:
                continue
            mw = float(molecular_weight(mol))
        except Exception:
            continue
        total_mass_g += (mw / na) * n

    if total_mass_g <= 0 or density_g_cm3 <= 0:
        return 10.0

    vol_cm3 = total_mass_g / float(density_g_cm3)
    vol_nm3 = vol_cm3 * 1.0e21
    L_nm = vol_nm3 ** (1.0 / 3.0)
    return float(max(L_nm, 2.0))


def _coerce_density_for_box_estimate(value: object) -> float:
    """Return a safe density value for synthetic box estimation.

    Explicit-cell builds may intentionally store `density_g_cm3 = None` because the
    cell vectors are authoritative and no target packing density was used. The
    exporter should accept that metadata and only fall back to a conservative box
    estimate when it later needs to synthesize coordinates.
    """
    if value is None:
        return 0.0
    try:
        density = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(density):
        return 0.0
    return density


def _formal_charge_from_smiles(smiles: str) -> int:
    """Best-effort formal charge from SMILES.

    Used only for classifying a species as cation/anion/neutral for charge-scaling.
    """
    if Chem is None:
        return 0
    try:
        m = _parse_smiles_for_metadata(smiles)
        if m is None:
            return 0
        return int(sum(int(a.GetFormalCharge()) for a in m.GetAtoms()))
    except Exception:
        return 0


def _infer_species_kind(smiles: str, formal_charge: int) -> str:
    """Return one of: polymer / ion / solvent."""
    if "*" in smiles:
        return "polymer"
    if int(formal_charge) != 0:
        return "ion"
    return "solvent"


def _normalize_charge_scale_spec(spec: Any) -> Union[float, Dict[str, float]]:
    """Normalize user charge scaling spec.

    The returned dict contains:
      - canonical SMILES keys, and/or
      - raw moltype keys (as-is), and/or
      - special category keys in lower-case.
    """
    if spec is None:
        return 1.0
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, Mapping):
        out: dict[str, float] = {}
        for k, v in spec.items():
            if k is None:
                continue
            ks = str(k).strip()
            try:
                fv = float(v)
            except Exception:
                continue
            ksl = ks.lower()
            if ksl in {"polymer", "solvent", "ion", "cation", "anion", "all", "default", "neutral"}:
                out[ksl] = fv
                continue
            # Try canonical SMILES; if it fails, keep the raw key (e.g. moltype name)
            try:
                out[canonicalize_smiles(ks)] = fv
            except Exception:
                out[ks] = fv
        return out
    # last resort
    try:
        return float(spec)
    except Exception:
        return 1.0


def _resolve_charge_scale(
    spec: Union[float, Dict[str, float]],
    *,
    smiles: str,
    moltype: str,
    kind: str,
    formal_charge: int,
) -> float:
    """Resolve a per-species charge scaling factor.

    Priority (highest -> lowest):
      1) exact canonical SMILES match
      2) exact moltype name match
      3) sign-specific ion class: cation / anion
      4) broad kind: polymer / solvent / ion / neutral
      5) all / default
      6) fallback 1.0
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    if not isinstance(spec, dict):
        return 1.0

    csmi = canonicalize_smiles(smiles)
    if csmi in spec:
        return float(spec[csmi])
    if moltype in spec:
        return float(spec[moltype])

    if int(formal_charge) > 0 and "cation" in spec:
        return float(spec["cation"])
    if int(formal_charge) < 0 and "anion" in spec:
        return float(spec["anion"])

    if kind in spec:
        return float(spec[kind])
    if int(formal_charge) == 0 and "neutral" in spec:
        return float(spec["neutral"])

    if "ion" in spec and int(formal_charge) != 0:
        return float(spec["ion"])
    if "all" in spec:
        return float(spec["all"])
    if "default" in spec:
        return float(spec["default"])
    return 1.0

def _rewrite_itp_moltype_and_resname(itp_text: str, new_name: str, *, preserve_residues: bool = False) -> str:
    """Rewrite [ moleculetype ] name and optionally flatten [ atoms ] residue name."""
    lines = itp_text.splitlines()
    out = []
    i = 0
    in_moleculetype = False
    in_atoms = False
    moleculetype_done = False
    new_res = new_name
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        low = s.lower()
        if low.startswith("[") and "moleculetype" in low:
            in_moleculetype = True
            in_atoms = False
            moleculetype_done = False
            out.append(ln)
            i += 1
            continue
        if low.startswith("[") and "atoms" in low:
            in_atoms = True
            in_moleculetype = False
            out.append(ln)
            i += 1
            continue
        if low.startswith("[") and not ("atoms" in low):
            in_atoms = False
            in_moleculetype = False

        if in_moleculetype and (not moleculetype_done):
            if (not s) or s.startswith(";"):
                out.append(ln)
            else:
                parts = ln.split()
                if parts:
                    parts[0] = new_name
                    out.append(" ".join(parts))
                    moleculetype_done = True
                else:
                    out.append(ln)
            i += 1
            continue

        if in_atoms:
            if (not s) or s.startswith(";"):
                out.append(ln)
            else:
                parts = ln.split()
                # Expected: nr type resnr residue atom cgnr charge mass
                if len(parts) >= 4 and not preserve_residues:
                    parts[3] = new_res
                    out.append(" ".join(parts))
                else:
                    out.append(ln)
            i += 1
            continue

        out.append(ln)
        i += 1
    return "\n".join(out) + ("\n" if itp_text.endswith("\n") else "")


def _rewrite_gro_resname(gro_text: str, new_resname: str, *, preserve_residues: bool = False) -> str:
    """Rewrite the 5-char residue name field in a .gro atom lines unless residues are preserved."""
    if preserve_residues:
        return gro_text
    lines = gro_text.splitlines()
    if len(lines) < 3:
        return gro_text
    res5 = (new_resname[:5]).ljust(5)
    out = [lines[0], lines[1]]
    # atom lines: next n lines
    try:
        n = int(lines[1].strip())
    except Exception:
        n = len(lines) - 3
    for j in range(2, 2 + n):
        if j >= len(lines):
            break
        ln = lines[j]
        if len(ln) >= 10:
            out.append(ln[:5] + res5 + ln[10:])
        else:
            out.append(ln)
    # box line
    if 2 + n < len(lines):
        out.extend(lines[2 + n:])
    return "\n".join(out) + ("\n" if gro_text.endswith("\n") else "")


def export_system_from_cell_meta(
    *,
    cell_mol,
    out_dir: Path,
    ff_name: str,
    charge_method: str = "RESP",
    total_charge: Optional[int] = None,
    # Charge scaling can be provided here, but the recommended way is to
    # specify per-species scaling in poly.amorphous_cell(..., charge_scale=[...]).
    # If None, we will read per-species scaling from the cell metadata.
    charge_scale: Optional[Any] = None,
    polyelectrolyte_mode: bool = False,
    include_h_atomtypes: bool = False,
    source_molecules_dir: Optional[Path] = None,
    system_gro_template: Optional[Path] = None,
    system_ndx_template: Optional[Path] = None,
    write_system_mol2: bool = True,
) -> SystemExportResult:
    """Export a mixed system to GROMACS input files.

    This function is the missing glue for the "ideal script" style:

      ac = poly.amorphous_cell([...])
      eqmd = eq.EQ21step(ac, ...)

    We DO NOT rely on the RDKit atom ordering inside `cell_mol`. Instead, we:
      1) read yadonpy's composition metadata from `_yadonpy_cell_meta`
      2) resolve each species by SMILES via MolDB (geometry+charges) when needed
      3) pack a new system.gro by replicating each species' single-molecule gro
      4) write system.top with includes and [molecules]
      5) generate system.ndx for analysis

        Internal fast path:
            - `source_molecules_dir` can point at a previous export's `molecules/` tree.
                When present, matching per-species artifacts are reused directly instead of
                resolving molecules from MolDB/FF again.
            - `system_gro_template` / `system_ndx_template` allow callers to reuse an
                already-built box layout when only charge-scaled topology files differ.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    molecules_dir = out_dir / "molecules"
    molecules_dir.mkdir(parents=True, exist_ok=True)
    source_molecules_dir = (Path(source_molecules_dir).expanduser().resolve() if source_molecules_dir is not None else None)
    system_gro_template = (Path(system_gro_template).expanduser().resolve() if system_gro_template is not None else None)
    system_ndx_template = (Path(system_ndx_template).expanduser().resolve() if system_ndx_template is not None else None)
    assembly_plan = SystemAssemblyPlan(
        out_dir=out_dir,
        molecules_dir=molecules_dir,
        source_molecules_dir=source_molecules_dir,
        system_gro_template=system_gro_template,
        system_ndx_template=system_ndx_template,
        effective_polyelectrolyte_mode=False,
    )

    # Charged mol2 outputs (raw + scaled)

    # ---------------
    # Load composition
    # ---------------
    try:
        meta = json.loads(cell_mol.GetProp("_yadonpy_cell_meta"))
    except Exception as e:
        raise ValueError("Cell is missing _yadonpy_cell_meta. Please build the system with poly.amorphous_cell().") from e

    species_in = meta.get("species") or []
    density_g_cm3 = _coerce_density_for_box_estimate(meta.get("density_g_cm3", 0.1))

    # -------------------------------------------------
    # Determine charge scaling spec
    # -------------------------------------------------
    # Priority (high -> low):
    #   1) explicit `charge_scale` passed to this exporter (float/dict/list)
    #   2) per-species scales stored in cell metadata (from amorphous_cell)
    #   3) default 1.0
    if charge_scale is None:
        # Build a per-SMILES mapping from metadata.
        cs_map: dict[str, float] = {}
        # If the metadata stored a global dict under key "charge_scale", prefer that.
        meta_cspec = meta.get("charge_scale")
        if isinstance(meta_cspec, dict):
            try:
                cspec = _normalize_charge_scale_spec(meta_cspec)
            except Exception:
                cspec = 1.0
        else:
            for sp in species_in:
                try:
                    smi = canonicalize_smiles(sp.get("smiles", ""))
                    if not smi:
                        continue
                    v = sp.get("charge_scale")
                    if v is None:
                        continue
                    cs_map[smi] = float(v)
                except Exception:
                    continue
            cspec = cs_map if cs_map else 1.0
    else:
        # Allow list/tuple aligned with species order.
        if isinstance(charge_scale, (list, tuple)):
            if len(charge_scale) != len(species_in):
                raise ValueError(
                    f"charge_scale length mismatch: got {len(charge_scale)} but species has {len(species_in)}"
                )
            cs_map: dict[str, float] = {}
            for sp, v in zip(species_in, charge_scale):
                smi = canonicalize_smiles(sp.get("smiles", ""))
                if smi:
                    cs_map[smi] = float(v)
            cspec = cs_map if cs_map else 1.0
        else:
            cspec = _normalize_charge_scale_spec(charge_scale)

    # Optionally leverage molecules embedded in the cell itself as artifacts, when they
    # already carry force-field typing (e.g., pre-parameterized polymer chains).
    #
    # Important for large mixed cells:
    #   `Chem.GetMolFrags(..., asMols=True)` duplicates every fragment molecule and can
    #   dominate the export cost. Most modern restart/export paths already have cached
    #   per-species artifacts in metadata, so fragment extraction should stay lazy and
    #   only run when those cached artifacts are unavailable.
    frag_mols = None
    frag_mols_loaded = False
    frag_cursor = 0

    def _ensure_frag_mols() -> list[Chem.Mol] | None:
        nonlocal frag_mols, frag_mols_loaded
        if frag_mols_loaded:
            return frag_mols
        frag_mols_loaded = True
        if Chem is None:
            frag_mols = None
            return frag_mols
        try:
            frag_mols = list(Chem.GetMolFrags(cell_mol, asMols=True, sanitizeFrags=False))
        except Exception:
            frag_mols = None
        return frag_mols

    species: list[dict] = []
    def _species_payload(
        sp_in: dict,
        *,
        artifact_dir: Path,
        mol_id: str,
        mol_name: str,
        moltype: str,
        formal_charge: int,
        kind: str,
        mol=None,
    ) -> dict[str, Any]:
        payload = {
            **sp_in,
            "export_schema_version": EXPORT_SYSTEM_SCHEMA_VERSION,
            "smiles": smiles,
            "n": n,
            "artifact_dir": str(artifact_dir),
            "mol_id": mol_id,
            "mol_name": mol_name,
            "moltype": moltype,
            "formal_charge": int(formal_charge),
            "kind": kind,
            "polyelectrolyte_mode": bool(sp_in.get("polyelectrolyte_mode", False)),
            "source_artifact_digest": _artifact_digest(artifact_dir, moltype),
        }
        for key in ("charge_groups", "resp_constraints", "polyelectrolyte_summary", "residue_map"):
            if key in sp_in and sp_in.get(key) is not None:
                payload[key] = sp_in.get(key)
        if mol is not None:
            try:
                payload["charge_groups"] = get_charge_groups(mol)
                payload["resp_constraints"] = get_resp_constraints(mol)
                payload["polyelectrolyte_summary"] = get_polyelectrolyte_summary(mol)
                payload["residue_map"] = build_residue_map(mol, mol_name=moltype)
            except Exception:
                pass
        return _enrich_species_polyelectrolyte_payload(
            payload,
            smiles=smiles,
            mol_name=moltype,
            mol=mol,
        )

    # Resolve/create per-species artifacts
    for i, sp in enumerate(species_in):
        smiles = canonicalize_smiles(sp.get("smiles", ""))
        n = int(sp.get("n", 0))
        if not smiles or n <= 0:
            continue
        frag_start = frag_cursor
        frag_cursor += n

        species_ff_name = str(sp.get("ff_name") or ff_name).strip() or str(ff_name)

        natoms_meta = int(sp.get("natoms") or 0)
        formal_charge = _formal_charge_from_smiles(smiles)
        # Heuristic: very large, neutral species are typically polymer chains.
        kind = "polymer" if ("*" in smiles or natoms_meta >= 200) else _infer_species_kind(smiles, formal_charge)
        mol_id = str(sp.get('name') or sp.get('resname') or f"M{i+1}")
        mol_name = str(sp.get('name') or sp.get('resname') or mol_id)
        mol_name_fs = _fs_safe_mol_name(mol_name)

        def _copy_from_src_dir(src_dir: Path | None) -> bool:
            if src_dir is None:
                return False
            try:
                src_dir = Path(src_dir).expanduser().resolve()
            except Exception:
                return False
            if not src_dir.exists():
                return False

            src_itp = next(iter(src_dir.glob("*.itp")), None)
            src_gro = next(iter(src_dir.glob("*.gro")), None)
            src_top = next(iter(src_dir.glob("*.top")), None)
            if src_itp is None or src_gro is None:
                return False

            dst_itp = art_dir / f"{mol_name_fs}.itp"
            itp_txt = src_itp.read_text(encoding="utf-8", errors="ignore")
            preserve_residues = bool(kind == "polymer" or sp.get("residue_map"))
            dst_itp.write_text(
                _rewrite_itp_moltype_and_resname(itp_txt, mol_name_fs, preserve_residues=preserve_residues),
                encoding="utf-8",
            )

            dst_gro = art_dir / f"{mol_name_fs}.gro"
            gro_txt = src_gro.read_text(encoding="utf-8", errors="ignore")
            dst_gro.write_text(
                _rewrite_gro_resname(gro_txt, mol_name_fs, preserve_residues=preserve_residues),
                encoding="utf-8",
            )

            if src_top is not None:
                dst_top = art_dir / f"{mol_name_fs}.top"
                shutil.copy2(src_top, dst_top)

            return bool(dst_itp.exists() and dst_gro.exists())

        if source_molecules_dir is not None:
            src_dir = source_molecules_dir / mol_name_fs
            src_itp = (src_dir / f"{mol_name_fs}.itp")
            src_gro = (src_dir / f"{mol_name_fs}.gro")
            if (not src_itp.is_file()) or (not src_gro.is_file()):
                src_itp = next(iter(sorted(src_dir.glob("*.itp"))), None)
                src_gro = next(iter(sorted(src_dir.glob("*.gro"))), None)
            if src_itp is not None and src_gro is not None and src_itp.is_file() and src_gro.is_file():
                species.append(
                    _species_payload(
                        sp,
                        artifact_dir=src_dir,
                        mol_id=mol_id,
                        mol_name=mol_name,
                        moltype=mol_name_fs,
                        formal_charge=formal_charge,
                        kind=kind,
                    )
                )
                continue

        art_dir = (molecules_dir / mol_name_fs)
        art_dir.mkdir(parents=True, exist_ok=True)

        copied = False
        src_meta_dir = str(sp.get('cached_artifact_dir') or '').strip()
        if src_meta_dir:
            copied = _copy_from_src_dir(Path(src_meta_dir))

        if (not copied) and sp.get('cached_mol_id'):
            try:
                from .molecule_cache import _default_cache_root

                ff_name_src = str(sp.get('ff_name') or species_ff_name).lower()
                src_dir = (_default_cache_root() / ff_name_src / str(sp.get('cached_mol_id')).strip()).resolve()
                copied = _copy_from_src_dir(src_dir)
            except Exception:
                copied = False

        if copied:
            species.append(
                _species_payload(
                    sp,
                    artifact_dir=art_dir,
                    mol_id=mol_id,
                    mol_name=mol_name,
                    moltype=mol_name_fs,
                    formal_charge=formal_charge,
                    kind=kind,
                )
            )
            continue

        # Representative fragment molecule from the cell (if available), to preserve atom ordering
        # and reuse pre-typed FF information.
        rep_mol = None
        frag_list = _ensure_frag_mols()
        if frag_list is not None:
            if frag_start + n <= len(frag_list):
                cand = frag_list[frag_start]
                ok = True
                for j in range(n):
                    if frag_list[frag_start + j].GetNumAtoms() != cand.GetNumAtoms():
                        ok = False
                        break
                if ok and (natoms_meta <= 0 or cand.GetNumAtoms() == natoms_meta):
                    rep_mol = cand

        rep_has_ff = False
        if rep_mol is not None:
            try:
                rep_has_ff = any(a.HasProp("ff_type") and a.GetProp("ff_type") for a in rep_mol.GetAtoms())
            except Exception:
                rep_has_ff = False

        if rep_has_ff:
            # IMPORTANT: rep_mol comes from Chem.GetMolFrags(cell_mol, asMols=True),
            # which preserves RDKit props (ff_type, ff_r0, ff_k, ...) but drops Python-level
            # containers like mol.angles / mol.dihedrals. Rebuild them here so exported ITP
            # contains [ angles ] / [ dihedrals ] (Example 01 regression).
            try:
                _ensure_bonded_terms_from_types(rep_mol, ff_name)
            except Exception:
                pass

            # Prefer cached per-molecule artifacts if available (robust for packed-cell exports).
            copied = False
            try:
                # Last resort: recover ids from representative fragment atom props.
                if not copied:
                    molid = None
                    try:
                        a0 = rep_mol.GetAtomWithIdx(0)
                        if a0.HasProp("_yadonpy_molid"):
                            molid = str(a0.GetProp("_yadonpy_molid")).strip()
                    except Exception:
                        molid = None

                    src_dir = None
                    if molid:
                        try:
                            if hasattr(rep_mol, "HasProp") and rep_mol.HasProp("_yadonpy_artifact_dir"):
                                src_dir = Path(rep_mol.GetProp("_yadonpy_artifact_dir")).expanduser().resolve()
                        except Exception:
                            src_dir = None
                        if src_dir is None:
                            from .molecule_cache import _default_cache_root

                            src_dir = (_default_cache_root() / str(ff_name).lower() / molid).resolve()
                    copied = _copy_from_src_dir(src_dir)

            except Exception:
                copied = False

            if not copied:
                write_molecule_artifacts(
                    rep_mol,
                    art_dir,
                    smiles=smiles,
                    ff_name=species_ff_name,
                    charge_method=charge_method,
                    total_charge=total_charge,
                    # Use the filesystem-safe moltype for filenames and the
                    # internal [ moleculetype ] name to keep GROMACS includes
                    # consistent.
                    mol_name=mol_name_fs,
                    write_mol2=False,
                )
            species.append(
                _species_payload(
                    sp,
                    artifact_dir=art_dir,
                    mol_id=mol_id,
                    mol_name=mol_name,
                    moltype=mol_name_fs,
                    formal_charge=formal_charge,
                    kind=kind,
                    mol=rep_mol,
                )
            )
            continue

        # Otherwise: resolve by SMILES via MolDB (geometry + charges), then write artifacts.
        try:
            ff_obj = get_ff(species_ff_name)
        except Exception:
            ff_obj = None

        try:
            # Prefer MolDB for geometry + charges.
            if ff_obj is not None and hasattr(ff_obj.__class__, "mol_rdkit"):
                rep_mol = ff_obj.__class__.mol_rdkit(
                    smiles,
                    name=mol_name,
                    prefer_db=True,
                    require_db=True,
                    require_ready=(str(charge_method).strip().upper() == "RESP"),
                    charge=charge_method,
                )
            else:
                from ..core import utils
                rep_mol = utils.mol_from_smiles(smiles)

            if ff_obj is not None:
                bonded_req = None
                try:
                    if sp.get('bonded_requested'):
                        bonded_req = str(sp.get('bonded_requested')).strip()
                    elif bool(sp.get('bonded_explicit')) and sp.get('bonded_method'):
                        bonded_req = str(sp.get('bonded_method')).strip()
                except Exception:
                    bonded_req = None
                ok = bool(ff_obj.ff_assign(rep_mol, bonded=bonded_req) if bonded_req else ff_obj.ff_assign(rep_mol))
                if not ok:
                    raise RuntimeError("ff_assign failed")
        except Exception as e:
            raise RuntimeError(
                f"Failed to resolve species[{i}] from MolDB. "
                f"Please precompute it into MolDB first (see Examples 07/08). "
                f"smiles={smiles} ff={species_ff_name} charge={charge_method}."
            ) from e

        write_molecule_artifacts(
            rep_mol,
            art_dir,
            smiles=smiles,
            ff_name=species_ff_name,
            charge_method=charge_method,
            total_charge=total_charge,
            mol_name=mol_name_fs,
            write_mol2=False,
        )

        species.append(
            _species_payload(
                sp,
                artifact_dir=art_dir,
                mol_id=mol_id,
                mol_name=mol_name,
                moltype=mol_name_fs,
                formal_charge=formal_charge,
                kind=kind,
                mol=rep_mol,
            )
        )


    if not species:
        raise ValueError("Empty system: no species resolved from cell metadata")
    effective_polyelectrolyte_mode = bool(
        polyelectrolyte_mode
        or meta.get("polyelectrolyte_mode")
        or any(_requires_charge_groups(sp) for sp in species)
    )
    assembly_plan.effective_polyelectrolyte_mode = bool(effective_polyelectrolyte_mode)
    for sp in species:
        _ensure_charge_group_ready(sp, effective_polyelectrolyte_mode=bool(effective_polyelectrolyte_mode))

    # ---------------
    # Copy molecule ITP/TOP/GRO into work directory
    # ---------------
    mol_itp_paths: list[Path] = []
    mol_gro_paths: list[Path] = []
    mol_names: list[str] = []
    # Collect global force-field parameter sections (e.g., [ atomtypes ]) across all molecule ITPs.
    # Some generators emit ITPs that start directly with [ moleculetype ] (no atomtypes), while
    # others include [ atomtypes ] / [ bondtypes ] ... before [ moleculetype ]. Mixing those ITPs
    # and including them sequentially can trigger grompp errors such as
    # "Invalid order for directive atomtypes" when an [ atomtypes ] block appears after a
    # previously included [ moleculetype ] block.
    # We therefore extract any parameter sections that appear before the first [ moleculetype ]
    # and write them once into a combined ff_parameters.itp that is included before any molecule types.
    global_param_preamble: list[str] = []
    global_param_sections: dict[str, list[str]] = {}
    inherited_param_itp = None
    if source_molecules_dir is not None:
        candidate = source_molecules_dir.parent / "ff_parameters.itp"
        if candidate.is_file():
            inherited_param_itp = candidate

    def _accumulate_param_blocks(params_text: str) -> None:
        """Accumulate parameter sections from params_text into global_param_sections with de-dup."""
        if not params_text.strip():
            return
        current = None
        for raw in params_text.splitlines():
            line = raw.rstrip("\n")
            s = line.strip()
            if not s:
                # Keep blank lines inside a section for readability.
                if current is not None:
                    global_param_sections.setdefault(current, []).append("")
                continue
            if s.startswith(";") or s.startswith("#"):
                # Keep comments/includes only in preamble (outside any section) or as-is inside sections.
                if current is None:
                    if line not in global_param_preamble:
                        global_param_preamble.append(line)
                else:
                    global_param_sections.setdefault(current, []).append(line)
                continue
            if s.startswith("[") and s.endswith("]"):
                sec = s.strip("[]").strip().lower()
                # Never allow nested [ defaults ] here; system.top writes it once.
                if sec == "defaults":
                    current = None
                    continue
                current = sec
                global_param_sections.setdefault(current, []).append(f"[ {sec} ]")
                continue
            if current is not None:
                global_param_sections.setdefault(current, []).append(line)

    if inherited_param_itp is not None:
        _accumulate_param_blocks(inherited_param_itp.read_text(encoding="utf-8", errors="replace"))

    charge_scaling_report: dict[str, Any] = {"schema_version": EXPORT_SYSTEM_SCHEMA_VERSION, "species": []}
    for sp in species:
        art_dir = Path(sp["artifact_dir"])
        mol_name = str(sp.get("mol_name") or sp.get("mol_id") or "MOL")
        mol_name_fs = _fs_safe_mol_name(mol_name)
        # Canonical moltype name used in system.top / system.ndx
        sp["moltype"] = mol_name_fs
        formal_charge = int(sp.get("formal_charge", 0))
        kind = str(sp.get("kind") or _infer_species_kind(str(sp.get("smiles", "")), formal_charge))
        scale = _resolve_charge_scale(
            cspec,
            smiles=str(sp.get("smiles", "")),
            moltype=mol_name_fs,
            kind=kind,
            formal_charge=formal_charge,
        )
        # Record the scale actually used for this moltype.
        sp["charge_scale"] = float(scale)
        dst_dir = molecules_dir / mol_name_fs
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Choose the first matching artifacts
        itp = next(iter(sorted(art_dir.glob("*.itp"))), None)
        gro = next(iter(sorted(art_dir.glob("*.gro"))), None)
        if itp is None or gro is None:
            raise RuntimeError(f"Missing .itp or .gro in artifact_dir={art_dir}")

        # Always write canonical filenames based on the resolved molecule name.
        # This keeps exported artifacts consistent and avoids grompp errors when
        # the cached library name differs from the desired species name.
        dst_itp = dst_dir / f"{mol_name_fs}.itp"
        dst_gro = dst_dir / f"{mol_name_fs}.gro"
        # Apply per-species simulation-level charge scaling when copying to the run directory.
        itp_text = itp.read_text(encoding="utf-8", errors="replace")
        charge_groups = sp.get("charge_groups") if isinstance(sp.get("charge_groups"), list) else None
        use_group_scaling = _uses_charge_group_scaling(
            sp,
            effective_polyelectrolyte_mode=bool(effective_polyelectrolyte_mode),
        )
        if use_group_scaling:
            itp_text, scale_report = _scale_itp_charge_groups(itp_text, charge_groups, scale=float(scale))
        else:
            itp_text = _scale_itp_charges(itp_text, scale=float(scale))
            scale_report = {"scale": float(scale), "groups": [], "fallback": ("whole_molecule_scale" if effective_polyelectrolyte_mode else None)}

        # Rewrite moltype + residue name to match `mol_name_fs`.
        # (Important when cached entries use hashed IDs.)
        preserve_residues = bool(kind == "polymer" or sp.get("residue_map"))
        itp_text = _rewrite_itp_moltype_and_resname(itp_text, mol_name_fs, preserve_residues=preserve_residues)

        # Extract any parameter blocks before [ moleculetype ] and accumulate them.
        params_text, mol_text = _split_itp_params_and_mol_text(itp_text)
        _accumulate_param_blocks(params_text)
        # Write molecule part only (starts with [ moleculetype ]).
        dst_itp.write_text(mol_text, encoding="utf-8")

        gro_txt = gro.read_text(encoding="utf-8", errors="replace")
        dst_gro.write_text(_rewrite_gro_resname(gro_txt, mol_name_fs, preserve_residues=preserve_residues), encoding="utf-8")
        sp["artifact_dir"] = str(dst_dir)
        charge_scaling_report["species"].append(
            {
                "moltype": mol_name_fs,
                "smiles": str(sp.get("smiles", "")),
                "kind": kind,
                "charge_scale": float(scale),
                "polyelectrolyte_mode": bool(effective_polyelectrolyte_mode),
                "used_group_scaling": bool(use_group_scaling),
                "report": scale_report,
            }
        )

        mol_itp_paths.append(dst_itp)
        mol_gro_paths.append(dst_gro)
        mol_names.append(mol_name_fs)

    charge_fix_info = _neutralize_exported_system_charge(
        species=species,
        mol_names=mol_names,
        mol_itp_paths=mol_itp_paths,
    )

    # ---------------
    # Build system.top
    # ---------------
    system_top = out_dir / "system.top"
    lines: list[str] = []
    lines.append("; yadonpy generated system.top")
    lines.append("[ defaults ]")
    lines.append("; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ")
    lines.append("1 2 yes 0.5 0.8333333333")
    lines.append("")

    # Write combined parameter include if needed.
    ff_param_itp = out_dir / "ff_parameters.itp"
    if any(v for v in global_param_sections.values()):
        # De-duplicate within each section while preserving the first occurrence.
        section_order = [
            "atomtypes",
            "bondtypes",
            "constrainttypes",
            "angletypes",
            "dihedraltypes",
            "improper_types",
            "pairtypes",
            "nonbond_params",
        ]
        # Also include any other sections we saw, after the known list.
        for sec in sorted(global_param_sections.keys()):
            if sec not in section_order:
                section_order.append(sec)
        out_lines: list[str] = []
        out_lines.append("; yadonpy combined FF parameter blocks")
        out_lines.extend(global_param_preamble)
        out_lines.append("")
        for sec in section_order:
            blk = global_param_sections.get(sec) or []
            if not blk:
                continue
            seen = set()
            for ln in blk:
                key = ln.strip()
                if not key:
                    out_lines.append("")
                    continue
                if key in seen:
                    continue
                seen.add(key)
                out_lines.append(ln)
            out_lines.append("")
        ff_param_itp.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
        lines.append(f'#include "{ff_param_itp.name}"')
        lines.append("")

    for name, itp in zip(mol_names, mol_itp_paths):
        lines.append(f'#include "molecules/{name}/{itp.name}"')
    lines.append("")
    lines.append("[ system ]")
    lines.append("yadonpy_system")
    lines.append("")
    lines.append("[ molecules ]")
    lines.append("; molname  count")
    for sp, name in zip(species, mol_names):
        lines.append(f"{name:<16s} {int(sp['n'])}")
    system_top.write_text("\n".join(lines) + "\n", encoding="utf-8")
    topology_issues = _validate_topology_include_order(system_top)
    if topology_issues:
        raise RuntimeError(f"Invalid include order in generated topology {system_top}: {'; '.join(topology_issues)}")

    # ---------------
    # Write system.gro
    # ---------------
    # NOTE: poly.amorphous_cell already performed packing (density + minimum-distance
    # threshold). Re-packing here with an O(N^2) rejection sampler can become extremely
    # slow for large systems (appearing "hung" at export_system_pair). Therefore we
    # prefer exporting the coordinates directly from the packed `cell_mol` when
    # possible, and only fall back to synthetic packing when the cell lacks usable
    # 3D coordinates.
    box_nm = _estimate_box_nm(species, density_g_cm3)
    box_lengths_nm = _cell_box_lengths_nm_from_rdkit(cell_mol) or (float(box_nm), float(box_nm), float(box_nm))
    box_vec_nm = np.asarray(box_lengths_nm, dtype=float)
    cell_origin_nm = _cell_box_origin_nm_from_rdkit(cell_mol)
    system_gro = out_dir / "system.gro"
    species_templates, nat_total = _load_gro_species_templates(species, mol_names, mol_gro_paths)
    if system_gro_template is not None and system_gro_template.is_file():
        shutil.copy2(system_gro_template, system_gro)
    else:
        atom_counter = 1
        res_counter = 1
        line_buffer: list[str] = []
        flush_every = 8192

        def _mol_coords_from_rdkit(mol) -> Optional[np.ndarray]:
            try:
                conf = mol.GetConformer()
                pts = conf.GetPositions()  # (N,3) float
                arr = np.asarray(pts, dtype=float)
                return arr
            except Exception:
                return None

        def _buffer_line(handle, line: str) -> None:
            line_buffer.append(line)
            if len(line_buffer) >= flush_every:
                _flush_gro_lines(handle, line_buffer)

        header = "yadonpy system"
        with system_gro.open("w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write(f"{nat_total}\n")

            used_cell_coords = False
            # Attempt to use packed coordinates from the cell.
            if Chem is not None:
                try:
                    frags = list(Chem.GetMolFrags(cell_mol, asMols=True, sanitizeFrags=False))
                except Exception:
                    frags = []
                total_needed = int(sum(int(sp.get("n", 0)) for sp in species))
                if frags and len(frags) >= total_needed:
                    # RDKit does not guarantee fragment ordering. Build buckets so we can
                    # match each packed fragment back to the requested species.
                    frags = frags[:total_needed]
                    frag_buckets: dict[tuple[int, tuple[tuple[int, int], ...]], list[tuple[Chem.Mol, np.ndarray]]] = {}
                    big_min: Optional[np.ndarray] = None
                    big_max: Optional[np.ndarray] = None
                    max_abs = 0.0
                    for m in frags:
                        c = _mol_coords_from_rdkit(m)
                        if c is None:
                            frag_buckets = {}
                            break
                        frag_buckets.setdefault(_mol_signature(m), []).append((m, c))
                        cur_min = np.min(c, axis=0)
                        cur_max = np.max(c, axis=0)
                        max_abs = max(max_abs, float(np.max(np.abs(cur_min))), float(np.max(np.abs(cur_max))))
                        if big_min is None:
                            big_min = cur_min
                            big_max = cur_max
                        else:
                            big_min = np.minimum(big_min, cur_min)
                            big_max = np.maximum(big_max, cur_max)

                    if frag_buckets and big_min is not None and big_max is not None:
                        # Heuristic unit detection: amorphous_cell uses Angstrom-like thresholds (e.g. 2.0)
                        # so coordinates are typically in Angstrom. Convert to nm for .gro.
                        to_nm = 0.1 if max_abs > 20.0 else 1.0
                        mn = big_min * to_nm
                        mx = big_max * to_nm
                        span = mx - mn
                        have_explicit_box = cell_origin_nm is not None and np.all(box_vec_nm > 0.0)
                        span_safe = np.where(span > 1.0e-9, span, 1.0)
                        scale_vec = np.where(span > 1.0e-9, box_vec_nm / span_safe, 1.0)

                        for tpl in species_templates:
                            sp = tpl.species
                            name = tpl.name
                            nat_expect = len(tpl.atom_names)
                            sig = _species_signature_from_smiles(str(sp.get("smiles", "") or ""))
                            for _k in range(int(sp["n"])):
                                frag_entry = None
                                if sig is not None and sig in frag_buckets and frag_buckets[sig]:
                                    frag_entry = frag_buckets[sig].pop()
                                else:
                                    for (nat, _sig2), lst in frag_buckets.items():
                                        if nat == nat_expect and lst:
                                            frag_entry = lst.pop()
                                            break
                                if frag_entry is None:
                                    remain = {k: len(v) for k, v in frag_buckets.items() if v}
                                    raise RuntimeError(
                                        f"Packed cell fragment matching failed for {name or 'mol'}: need nat={nat_expect}. Remaining buckets={remain}"
                                    )
                                _m, c = frag_entry
                                if c.shape[0] != nat_expect:
                                    raise RuntimeError(
                                        f"Packed cell coordinates mismatch for {name}: got {c.shape[0]} atoms, expect {nat_expect}."
                                    )
                                if have_explicit_box:
                                    coords = (c * to_nm) - np.asarray(cell_origin_nm, dtype=float)
                                else:
                                    coords = (c * to_nm - mn) * scale_vec
                                coords = np.mod(coords, box_vec_nm)
                                resname = (name[:5] if name else "MOL")
                                for a_name, (x, y, z) in zip(tpl.atom_names, coords):
                                    _buffer_line(
                                        f,
                                        _format_gro_atom_line(resnr=res_counter, resname=resname, atomname=a_name, atomnr=atom_counter, x=x, y=y, z=z) + "\n",
                                    )
                                    atom_counter += 1
                                res_counter += 1
                        used_cell_coords = True

            if not used_cell_coords:
                placed_xyz: np.ndarray = np.zeros((0, 3), dtype=float)
                min_dist_nm = 0.12
                max_place_trials = 200

                for tpl in species_templates:
                    sp = tpl.species
                    name = tpl.name
                    atom_names = tpl.atom_names
                    coords0 = tpl.coords0

                    for _k in range(int(sp["n"])):
                        coords = coords0 - coords0.mean(axis=0, keepdims=True)
                        R = _random_rotation_matrix()
                        coords = coords @ R.T

                        ok = False
                        for _t in range(max_place_trials):
                            shift = np.array(
                                [
                                    random.random() * float(box_vec_nm[0]),
                                    random.random() * float(box_vec_nm[1]),
                                    random.random() * float(box_vec_nm[2]),
                                ]
                            )
                            coords_try = np.mod(coords + shift, box_vec_nm)
                            if placed_xyz.shape[0] == 0:
                                ok = True
                                coords = coords_try
                                break
                            d2 = ((placed_xyz[None, :, :] - coords_try[:, None, :]) ** 2).sum(axis=2)
                            if float(np.sqrt(d2.min())) > min_dist_nm:
                                ok = True
                                coords = coords_try
                                break
                        if not ok:
                            coords = np.mod(
                                coords
                                + np.array(
                                    [
                                        random.random() * float(box_vec_nm[0]),
                                        random.random() * float(box_vec_nm[1]),
                                        random.random() * float(box_vec_nm[2]),
                                    ]
                                ),
                                box_vec_nm,
                            )

                        placed_xyz = np.vstack([placed_xyz, coords])
                        resname = (name[:5] if name else "MOL")
                        for a_name, (x, y, z) in zip(atom_names, coords):
                            _buffer_line(
                                f,
                                _format_gro_atom_line(resnr=res_counter, resname=resname, atomname=a_name, atomnr=atom_counter, x=x, y=y, z=z) + "\n",
                            )
                            atom_counter += 1
                        res_counter += 1

            _flush_gro_lines(f, line_buffer)
            f.write(f"{float(box_vec_nm[0]):10.5f}{float(box_vec_nm[1]):10.5f}{float(box_vec_nm[2]):10.5f}\n")

    # ---------------
    # Best-effort: write a **system-level** MOL2 for the packed box (visualization/interoperability)
    # ---------------
    # Users often want a single MOL2 of the full simulation box right after the cell is built.
    # We generate it from the exported GROMACS topology + coordinates using ParmEd.
    # This does NOT affect downstream MD.
    if write_system_mol2:
        try:
            from .mol2 import write_mol2_from_top_gro_parmed
            from ..core.logging_utils import yadon_print

            out_mol2 = write_mol2_from_top_gro_parmed(
                top_path=system_top,
                gro_path=system_gro,
                out_mol2=(out_dir / "system.mol2"),
                overwrite=True,
            )
            if out_mol2 is None:
                yadon_print(
                    f"Failed to write system.mol2 (best-effort). "
                    f"Check ParmEd installation and whether system.top/system.gro are readable. out_dir={out_dir}",
                    level=2,
                )
        except Exception:
            # Keep export robust; MOL2 is optional.
            pass

    # ---------------
    # Generate system.ndx
    # ---------------
    from ..gmx.index import generate_system_ndx

    system_ndx = out_dir / "system.ndx"
    if system_ndx_template is not None and system_ndx_template.is_file():
        shutil.copy2(system_ndx_template, system_ndx)
    else:
        generate_system_ndx(
            top_path=system_top,
            ndx_path=system_ndx,
            include_h_atomtypes=include_h_atomtypes,
        )

    # system meta (for robust SMILES<->moltype mapping in analysis)
    system_meta = out_dir / "system_meta.json"
    actual_box_lengths_nm = _read_gro_box_lengths_nm(system_gro) or tuple(float(x) for x in box_vec_nm)
    payload = {
        "schema_version": EXPORT_SYSTEM_SCHEMA_VERSION,
        "species": species,
        "box_nm": float(max(actual_box_lengths_nm)),
        "box_lengths_nm": [float(x) for x in actual_box_lengths_nm],
    }
    if charge_fix_info is not None:
        payload["export_charge_correction"] = charge_fix_info
    system_meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    residue_map_path = out_dir / "residue_map.json"
    residue_map_payload = {
        "species": [
            {
                "moltype": str(sp.get("moltype", "")),
                "mol_name": str(sp.get("mol_name", "")),
                "n": int(sp.get("n", 0)),
                "residue_map": sp.get("residue_map"),
            }
            for sp in species
            if sp.get("residue_map") is not None
        ]
    }
    residue_map_path.write_text(json.dumps(residue_map_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    charge_groups_path = out_dir / "charge_groups.json"
    charge_groups_payload = {
        "species": [
            {
                "moltype": str(sp.get("moltype", "")),
                "mol_name": str(sp.get("mol_name", "")),
                "smiles": str(sp.get("smiles", "")),
                "charge_groups": sp.get("charge_groups", []),
                "polyelectrolyte_summary": sp.get("polyelectrolyte_summary"),
            }
            for sp in species
        ]
    }
    charge_groups_path.write_text(json.dumps(charge_groups_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    resp_constraints_path = out_dir / "resp_constraints.json"
    resp_constraints_payload = {
        "species": [
            {
                "moltype": str(sp.get("moltype", "")),
                "mol_name": str(sp.get("mol_name", "")),
                "resp_constraints": sp.get("resp_constraints"),
            }
            for sp in species
        ]
    }
    resp_constraints_path.write_text(json.dumps(resp_constraints_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    charge_scaling_report_path = out_dir / "charge_scaling_report.json"
    charge_scaling_report_path.write_text(json.dumps(charge_scaling_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    site_map_path = out_dir / "site_map.json"
    site_map_payload = build_site_map(parse_system_top(system_top), out_dir)
    site_map_path.write_text(json.dumps(site_map_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    export_manifest = out_dir / "export_manifest.json"
    export_manifest.write_text(
        json.dumps(
            {
                "schema_version": EXPORT_SYSTEM_SCHEMA_VERSION,
                "effective_polyelectrolyte_mode": bool(assembly_plan.effective_polyelectrolyte_mode),
                "out_dir": str(assembly_plan.out_dir),
                "molecules_dir": str(assembly_plan.molecules_dir),
                "source_molecules_dir": (str(assembly_plan.source_molecules_dir) if assembly_plan.source_molecules_dir is not None else None),
                "system_gro_template": (str(assembly_plan.system_gro_template) if assembly_plan.system_gro_template is not None else None),
                "system_ndx_template": (str(assembly_plan.system_ndx_template) if assembly_plan.system_ndx_template is not None else None),
                "system_top_sig": file_signature(system_top),
                "system_gro_sig": file_signature(system_gro),
                "system_ndx_sig": file_signature(system_ndx),
                "species": [
                    {
                        "moltype": str(sp.get("moltype", "")),
                        "artifact_dir": str(sp.get("artifact_dir", "")),
                        "source_artifact_digest": sp.get("source_artifact_digest"),
                    }
                    for sp in species
                ],
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    return SystemExportResult(
        system_gro=system_gro,
        system_top=system_top,
        system_ndx=system_ndx,
        molecules_dir=molecules_dir,
        system_meta=system_meta,
        box_nm=float(max(actual_box_lengths_nm)),
        species=species,
        box_lengths_nm=tuple(float(x) for x in actual_box_lengths_nm),
    )
