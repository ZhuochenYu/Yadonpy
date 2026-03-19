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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception:  # pragma: no cover
    Chem = None
    Descriptors = None

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

from ..ff.library import LibraryDB, canonicalize_smiles
from ..api import ensure_basic_top, get_ff
from .artifacts import write_molecule_artifacts


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


@dataclass
class SystemExportResult:
    system_gro: Path
    system_top: Path
    system_ndx: Path
    molecules_dir: Path
    system_meta: Path
    box_nm: float
    species: list[dict]


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


def _scale_itp_charges(itp_text: str, scale: float) -> str:
    """Scale charges in the `[ atoms ]` section of an ITP.

    We keep the rest of the file untouched. This is used for simulation-level
    dielectric charge scaling without modifying the persistent library.

    The GROMACS [ atoms ] format is typically:
      nr  type  resnr  resid  atom  cgnr  charge  mass
    """
    s = float(scale)
    if abs(s - 1.0) < 1.0e-12:
        return itp_text

    out_lines: list[str] = []
    in_atoms = False
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

    na = 6.02214076e23
    total_mass_g = 0.0
    for sp in species:
        smi = sp.get("smiles", "")
        n = int(sp.get("n", 0))
        if n <= 0:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mw = float(Descriptors.MolWt(mol))
        except Exception:
            continue
        total_mass_g += (mw / na) * n

    if total_mass_g <= 0 or density_g_cm3 <= 0:
        return 10.0

    vol_cm3 = total_mass_g / float(density_g_cm3)
    vol_nm3 = vol_cm3 * 1.0e21
    L_nm = vol_nm3 ** (1.0 / 3.0)
    return float(max(L_nm, 2.0))


def _formal_charge_from_smiles(smiles: str) -> int:
    """Best-effort formal charge from SMILES.

    Used only for classifying a species as cation/anion/neutral for charge-scaling.
    """
    if Chem is None:
        return 0
    try:
        m = Chem.MolFromSmiles(smiles)
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

def _rewrite_itp_moltype_and_resname(itp_text: str, new_name: str) -> str:
    """Rewrite [ moleculetype ] name and [ atoms ] residue name to `new_name`."""
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
                if len(parts) >= 4:
                    parts[3] = new_res
                    out.append(" ".join(parts))
                else:
                    out.append(ln)
            i += 1
            continue

        out.append(ln)
        i += 1
    return "\n".join(out) + ("\n" if itp_text.endswith("\n") else "")


def _rewrite_gro_resname(gro_text: str, new_resname: str) -> str:
    """Rewrite the 5-char residue name field in a .gro atom lines."""
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
    include_h_atomtypes: bool = False,
) -> SystemExportResult:
    """Export a mixed system to GROMACS input files.

    This function is the missing glue for the "ideal script" style:

      ac = poly.amorphous_cell([...])
      eqmd = eq.EQ21step(ac, ...)

    We DO NOT rely on the RDKit atom ordering inside `cell_mol`. Instead, we:
      1) read yadonpy's composition metadata from `_yadonpy_cell_meta`
      2) resolve each species by SMILES in the basic_top library (or generate)
      3) pack a new system.gro by replicating each species' single-molecule gro
      4) write system.top with includes and [molecules]
      5) generate system.ndx for analysis
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    molecules_dir = out_dir / "molecules"
    molecules_dir.mkdir(parents=True, exist_ok=True)

    # Charged mol2 outputs (raw + scaled)

    # ---------------
    # Load composition
    # ---------------
    try:
        meta = json.loads(cell_mol.GetProp("_yadonpy_cell_meta"))
    except Exception as e:
        raise ValueError("Cell is missing _yadonpy_cell_meta. Please build the system with poly.amorphous_cell().") from e

    species_in = meta.get("species") or []
    density_g_cm3 = float(meta.get("density_g_cm3", 0.1))

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

    db = LibraryDB()

    # Optionally leverage molecules embedded in the cell itself as artifacts, when they
    # already carry force-field typing (e.g., pre-parameterized polymer chains).
    frag_mols = None
    frag_cursor = 0
    if Chem is not None:
        try:
            frag_mols = list(Chem.GetMolFrags(cell_mol, asMols=True, sanitizeFrags=False))
        except Exception:
            frag_mols = None

    species: list[dict] = []
    # Resolve/create per-species artifacts
    for i, sp in enumerate(species_in):
        smiles = canonicalize_smiles(sp.get("smiles", ""))
        n = int(sp.get("n", 0))
        if not smiles or n <= 0:
            continue

        natoms_meta = int(sp.get("natoms") or 0)
        formal_charge = _formal_charge_from_smiles(smiles)
        # Heuristic: very large, neutral species are typically polymer chains.
        kind = "polymer" if ("*" in smiles or natoms_meta >= 200) else _infer_species_kind(smiles, formal_charge)

        # Representative fragment molecule from the cell (if available), to preserve atom ordering
        # and reuse pre-typed FF information.
        rep_mol = None
        if frag_mols is not None:
            if frag_cursor + n <= len(frag_mols):
                cand = frag_mols[frag_cursor]
                ok = True
                for j in range(n):
                    if frag_mols[frag_cursor + j].GetNumAtoms() != cand.GetNumAtoms():
                        ok = False
                        break
                if ok and (natoms_meta <= 0 or cand.GetNumAtoms() == natoms_meta):
                    rep_mol = cand
            frag_cursor += n

        rep_has_ff = False
        if rep_mol is not None:
            try:
                rep_has_ff = any(a.HasProp("ff_type") and a.GetProp("ff_type") for a in rep_mol.GetAtoms())
            except Exception:
                rep_has_ff = False

        if rep_has_ff:
            mol_id = str(sp.get('name') or sp.get('resname') or f"M{i+1}")
            mol_name = str(sp.get('name') or sp.get('resname') or mol_id)
            mol_name_fs = mol_name.replace('/', '_')

            art_dir = (molecules_dir / mol_name_fs)
            art_dir.mkdir(parents=True, exist_ok=True)

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
                from .molecule_cache import _default_cache_root
                import shutil as _shutil

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
                        src_dir = (_default_cache_root() / str(ff_name).lower() / molid).resolve()

                if src_dir is not None and src_dir.exists():
                    # Copy a single representative of each artifact type, but rewrite the moltype/resname
                    # to match the current species name (mol_name_fs). This avoids grompp errors like:
                    #   No such moleculetype monomer_A
                    src_itp = next(iter(src_dir.glob("*.itp")), None)
                    src_gro = next(iter(src_dir.glob("*.gro")), None)
                    src_top = next(iter(src_dir.glob("*.top")), None)

                    if src_itp is not None:
                        dst_itp = art_dir / f"{mol_name_fs}.itp"
                        itp_txt = src_itp.read_text(encoding="utf-8", errors="ignore")
                        dst_itp.write_text(_rewrite_itp_moltype_and_resname(itp_txt, mol_name_fs), encoding="utf-8")
                    if src_gro is not None:
                        dst_gro = art_dir / f"{mol_name_fs}.gro"
                        gro_txt = src_gro.read_text(encoding="utf-8", errors="ignore")
                        dst_gro.write_text(_rewrite_gro_resname(gro_txt, mol_name_fs), encoding="utf-8")
                    if src_top is not None:
                        # Per-molecule .top is not included by system.top, but we keep it for completeness
                        dst_top = art_dir / f"{mol_name_fs}.top"
                        _shutil.copy2(src_top, dst_top)

                    copied = bool((art_dir / f"{mol_name_fs}.itp").exists() and (art_dir / f"{mol_name_fs}.gro").exists())

            except Exception:
                copied = False

            if not copied:
                write_molecule_artifacts(
                    rep_mol,
                    art_dir,
                    smiles=smiles,
                    ff_name=ff_name,
                    charge_method=charge_method,
                    total_charge=total_charge,
                    mol_name=mol_name,
                    write_mol2=False,
                )
            species.append(
                {
                    **sp,
                    "smiles": smiles,
                    "n": n,
                    "artifact_dir": str(art_dir),
                    "mol_id": mol_id,
                    "mol_name": mol_name,
                    "moltype": mol_name.replace("/", "_"),
                    "formal_charge": int(formal_charge),
                    "kind": kind,
                }
            )
            continue

        # Otherwise, use (or generate) a cached single-molecule topology from the basic_top library
        try:
            ent = ensure_basic_top(
                smiles,
                ff_name=ff_name,
                charge_method=charge_method,
                work_dir=out_dir,
                total_charge=total_charge,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to prepare basic_top artifacts for species[{i}] smiles={smiles} ff={ff_name} charge={charge_method}. "
                f"If this is a pre-parameterized polymer, make sure the input polymer molecule (passed into poly.amorphous_cell) "
                f"already has ff_type/charges assigned so it can be exported from the cell."
            ) from e

        if ent is None:
            raise RuntimeError(f"Failed to resolve artifacts for species[{i}] smiles={smiles}.")

        art_dir = Path(ent["artifact_dir"]).resolve()

        # --- Validate cached single-molecule topology contains angles/dihedrals when expected ---
        # Some earlier versions could generate .itp missing [ angles ]/[ dihedrals ] (severe).
        # If detected here, force-regenerate the basic_top entry and overwrite artifacts.
        try:
            itp_files = sorted(art_dir.glob("*.itp"))
            itp_txt = itp_files[0].read_text(encoding="utf-8") if itp_files else ""
            # Heuristic: molecules with >=3 atoms should have angles; >=4 atoms should have dihedrals.
            # We only check presence of the blocks here (not counts).
            # NOTE: Do NOT import rdkit.Chem here.
            # If we import `Chem` inside this function, Python treats it as a local
            # variable everywhere in this scope, which can cause
            # `UnboundLocalError: local variable 'Chem' referenced before assignment`
            # when we reference the module-level `Chem` imported at file import time.
            nat_guess = 0
            try:
                if Chem is not None:
                    m0 = Chem.MolFromSmiles(smiles)
                    if m0 is not None:
                        nat_guess = int(m0.GetNumAtoms())
            except Exception:
                nat_guess = 0

            missing_angles = (nat_guess >= 3) and ("[ angles ]" not in itp_txt)
            missing_dihedrals = (nat_guess >= 4) and ("[ dihedrals ]" not in itp_txt)

            if missing_angles or missing_dihedrals:
                from ..api import parameterize_smiles
                from ..core import utils

                utils.radon_print(
                    f"[WARN] cached ITP missing bonded terms (angles={not missing_angles}, dihedrals={not missing_dihedrals}); regenerating basic_top for {smiles}",
                    level=2,
                )
                _, ent2 = parameterize_smiles(
                    smiles,
                    ff_name=ff_name,
                    charge_method=charge_method,
                    work_dir=out_dir,
                    auto_register_nonpolymer=True,
                    use_basic_top_first=False,
                    total_charge=total_charge,
                )
                if ent2 is not None:
                    ent = ent2
                    art_dir = Path(ent2["artifact_dir"]).resolve()
        except Exception:
            pass
        mol_id = str(sp.get('name') or sp.get('resname') or ent.get("mol_id") or f"M{i+1}")
        mol_name = str(sp.get('name') or sp.get('resname') or ent.get("name") or mol_id)
        sp2 = {
            **sp,
            **ent,
            "artifact_dir": str(art_dir),
            "mol_id": mol_id,
            "mol_name": mol_name,
            "moltype": mol_name.replace("/", "_"),
            "n": n,
            "smiles": smiles,
            "formal_charge": int(formal_charge),
            "kind": kind,
        }
        species.append(sp2)


    if not species:
        raise ValueError("Empty system: no species resolved from cell metadata")

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

    def _split_itp_params_and_mol(itp_text: str) -> tuple[str, str]:
        """Split an ITP into (params, mol) where params are any directives before first [ moleculetype ]."""
        lines = itp_text.splitlines()
        mt_idx = None
        for i, ln in enumerate(lines):
            if ln.strip().lower().startswith('[ moleculetype'):
                mt_idx = i
                break
        if mt_idx is None:
            return "", itp_text
        params = "\n".join(lines[:mt_idx]).strip() + "\n"
        molblk = "\n".join(lines[mt_idx:]).strip() + "\n"
        return params, molblk

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

    for sp in species:
        art_dir = Path(sp["artifact_dir"])
        mol_name = str(sp.get("mol_name") or sp.get("mol_id") or "MOL")
        mol_name_fs = mol_name.replace("/", "_")
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

        dst_itp = dst_dir / itp.name
        dst_gro = dst_dir / gro.name
        # Apply per-species simulation-level charge scaling when copying to the run directory.
        itp_text = itp.read_text(encoding="utf-8", errors="replace")
        itp_text = _scale_itp_charges(itp_text, scale=float(scale))

        # Extract any parameter blocks before [ moleculetype ] and accumulate them.
        params_text, mol_text = _split_itp_params_and_mol(itp_text)
        _accumulate_param_blocks(params_text)
        # Write molecule part only (starts with [ moleculetype ]).
        dst_itp.write_text(mol_text, encoding="utf-8")
        dst_gro.write_bytes(gro.read_bytes())

        mol_itp_paths.append(dst_itp)
        mol_gro_paths.append(dst_gro)
        mol_names.append(mol_name_fs)

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
    system_gro = out_dir / "system.gro"

    # Build gro lines from coordinates
    all_atom_lines: list[str] = []
    atom_counter = 1
    res_counter = 1

    def _mol_coords_from_rdkit(mol) -> Optional[np.ndarray]:
        try:
            conf = mol.GetConformer()
            pts = conf.GetPositions()  # (N,3) float
            arr = np.asarray(pts, dtype=float)
            return arr
        except Exception:
            return None

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
            frag_buckets: dict[tuple[int, tuple[tuple[int, int], ...]], list[Chem.Mol]] = {}
            for m in frags:
                frag_buckets.setdefault(_mol_signature(m), []).append(m)

            # Collect all coords to infer current unit and bounding box.
            coords_all: list[np.ndarray] = []
            for m in frags:
                c = _mol_coords_from_rdkit(m)
                if c is None:
                    coords_all = []
                    break
                coords_all.append(c)

            if coords_all:
                big = np.vstack(coords_all)
                # Heuristic unit detection: amorphous_cell uses Angstrom-like thresholds (e.g. 2.0)
                # so coordinates are typically in Angstrom. Convert to nm for .gro.
                max_abs = float(np.max(np.abs(big)))
                to_nm = 0.1 if max_abs > 20.0 else 1.0
                big_nm = big * to_nm
                mn = big_nm.min(axis=0)
                mx = big_nm.max(axis=0)
                span = mx - mn
                # Ensure non-zero span; add a small margin.
                span_max = float(np.max(span)) if float(np.max(span)) > 1e-9 else 1.0
                # Scale packed coords to match the density-derived cubic box.
                scale = float(box_nm) / span_max

                for sp, name, gro_path in zip(species, mol_names, mol_gro_paths):
                    atom_names, _coords0 = _read_gro_single_molecule(gro_path)
                    nat_expect = len(atom_names)
                    # Determine signature from SMILES (preferred) to avoid relying on GRO parsing.
                    sig = None
                    try:
                        smi = str(sp.get("smiles", "") or "")
                        if smi:
                            tmpl = Chem.MolFromSmiles(smi)
                            if tmpl is not None:
                                sig = _mol_signature(tmpl)
                    except Exception:
                        sig = None
                    for _k in range(int(sp["n"])):
                        m = None
                        if sig is not None and sig in frag_buckets and frag_buckets[sig]:
                            m = frag_buckets[sig].pop()
                        else:
                            # Fallback: match by atom count only.
                            # Pick any fragment bucket with matching nat.
                            for (nat, _sig2), lst in frag_buckets.items():
                                if nat == nat_expect and lst:
                                    m = lst.pop()
                                    break
                        if m is None:
                            # Provide a helpful error with remaining bucket keys.
                            remain = {k: len(v) for k, v in frag_buckets.items() if v}
                            raise RuntimeError(
                                f"Packed cell fragment matching failed for {name or 'mol'}: need nat={nat_expect}. Remaining buckets={remain}"
                            )
                        c = _mol_coords_from_rdkit(m)
                        if c is None or c.shape[0] != nat_expect:
                            raise RuntimeError(
                                f"Packed cell coordinates mismatch for {name}: got {None if c is None else c.shape[0]} atoms, expect {nat_expect}."
                            )
                        coords = (c * to_nm - mn) * scale
                        coords = coords % box_nm
                        resname = (name[:5] if name else "MOL")
                        for a_name, (x, y, z) in zip(atom_names, coords):
                            all_atom_lines.append(
                                f"{res_counter:5d}{resname:<5s}{a_name:>5s}{atom_counter:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
                            )
                            atom_counter += 1
                        res_counter += 1
                used_cell_coords = True

    if not used_cell_coords:
        # Fallback: synthetic random packing (slow for huge systems, but keeps behavior
        # defined when cell coordinates are unavailable).
        placed_xyz: np.ndarray = np.zeros((0, 3), dtype=float)
        min_dist_nm = 0.12  # soft overlap threshold
        max_place_trials = 200

        for sp, name, gro_path in zip(species, mol_names, mol_gro_paths):
            atom_names, coords0 = _read_gro_single_molecule(gro_path)

            for _k in range(int(sp["n"])):
                coords = coords0 - coords0.mean(axis=0, keepdims=True)
                R = _random_rotation_matrix()
                coords = coords @ R.T

                ok = False
                for _t in range(max_place_trials):
                    shift = np.array([random.random() * box_nm, random.random() * box_nm, random.random() * box_nm])
                    coords_try = (coords + shift) % box_nm
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
                    coords = (coords + np.array([random.random() * box_nm, random.random() * box_nm, random.random() * box_nm])) % box_nm

                placed_xyz = np.vstack([placed_xyz, coords])

                resname = (name[:5] if name else "MOL")
                for a_name, (x, y, z) in zip(atom_names, coords):
                    all_atom_lines.append(
                        f"{res_counter:5d}{resname:<5s}{a_name:>5s}{atom_counter:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
                    )
                    atom_counter += 1
                res_counter += 1

    header = "yadonpy system"
    nat_total = len(all_atom_lines)
    with system_gro.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(f"{nat_total}\n")
        for l in all_atom_lines:
            f.write(l + "\n")
        f.write(f"{box_nm:10.5f}{box_nm:10.5f}{box_nm:10.5f}\n")

    # ---------------
    # Generate system.ndx
    # ---------------
    from ..gmx.index import generate_system_ndx

    system_ndx = out_dir / "system.ndx"
    generate_system_ndx(
        top_path=system_top,
        ndx_path=system_ndx,
        include_h_atomtypes=include_h_atomtypes,
    )

    # system meta (for robust SMILES<->moltype mapping in analysis)
    system_meta = out_dir / "system_meta.json"
    system_meta.write_text(json.dumps({"species": species}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return SystemExportResult(
        system_gro=system_gro,
        system_top=system_top,
        system_ndx=system_ndx,
        molecules_dir=molecules_dir,
        system_meta=system_meta,
        box_nm=float(box_nm),
        species=species,
    )