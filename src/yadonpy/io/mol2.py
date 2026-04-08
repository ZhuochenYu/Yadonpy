"""Utilities to write charge-bearing MOL2 files.

YadonPy stores atomic partial charges on RDKit atoms (e.g. ``RESP`` or
``AtomicCharge``). For debugging and interoperability, we export MOL2 files
containing the charges used.

Why a custom writer?
--------------------
RDKit's MOL2 writer API differs across versions, and some builds omit it.
To keep yadonpy robust, we implement a minimal MOL2 writer that is sufficient
for downstream tools (it includes ATOM and BOND records and per-atom charges).

The writer supports two sources:

* RDKit molecules (preferred when available)
* GROMACS artifacts (``.gro`` + ``.itp``) when the RDKit object is not present

All coordinates in MOL2 are written in Angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing import List, Optional, Tuple

import re

from ..core import chem_utils as core_utils

_COMMON_FF_TYPE_ELEMENT_MAP = {
    "h": "H",
    "h1": "H",
    "h2": "H",
    "h3": "H",
    "h4": "H",
    "h5": "H",
    "ha": "H",
    "hc": "H",
    "hn": "H",
    "ho": "H",
    "hp": "H",
    "hs": "H",
    "hw": "H",
    "c": "C",
    "c1": "C",
    "c2": "C",
    "c3": "C",
    "ca": "C",
    "cc": "C",
    "cd": "C",
    "ce": "C",
    "cf": "C",
    "cg": "C",
    "ch": "C",
    "cp": "C",
    "cq": "C",
    "cu": "C",
    "cv": "C",
    "cx": "C",
    "cy": "C",
    "cz": "C",
    "o": "O",
    "o2": "O",
    "o3": "O",
    "oh": "O",
    "os": "O",
    "ow": "O",
    "op": "O",
    "oq": "O",
    "n": "N",
    "n1": "N",
    "n2": "N",
    "n3": "N",
    "n4": "N",
    "na": "N",
    "nb": "N",
    "nc": "N",
    "nd": "N",
    "ne": "N",
    "nf": "N",
    "nh": "N",
    "no": "N",
    "s": "S",
    "s2": "S",
    "s4": "S",
    "s6": "S",
    "sh": "S",
    "ss": "S",
    "sx": "S",
    "sy": "S",
    "p": "P",
    "p2": "P",
    "p3": "P",
    "p4": "P",
    "p5": "P",
    "px": "P",
    "py": "P",
    "f": "F",
    "cl": "Cl",
    "br": "Br",
    "i": "I",
    "li": "Li",
    "na+": "Na",
    "na_ion": "Na",
    "k": "K",
    "mg": "Mg",
    "si": "Si",
    "b": "B",
}

_PERMISSIVE_MOL2_TYPE_TOKENS = {
    "h1", "h2", "h3", "h4", "h5", "ha", "hc", "hn", "ho", "hp", "hs", "hw",
    "c1", "c2", "c3", "ca", "cc", "cd", "ce", "cf", "cg", "ch", "cp", "cq", "cu", "cv", "cx", "cy", "cz",
    "o2", "o3", "oh", "os", "ow", "op", "oq",
    "n1", "n2", "n3", "n4", "na", "nb", "nc", "nd", "ne", "nf", "nh", "no",
    "s2", "s4", "s6", "sh", "ss", "sx", "sy",
    "p2", "p3", "p4", "p5", "px", "py",
}

def _elem_from_atom_name(aname: str) -> str:
    """Infer element symbol from an atom name like 'C12', 'Cl3', 'Na1'."""
    s = str(aname).strip()
    m = re.match(r"^([A-Za-z]{1,2})", s)
    if not m:
        return "X"
    e = m.group(1)
    # normalize capitalization: first upper, second lower
    if len(e) == 1:
        return e.upper()
    return e[0].upper() + e[1:].lower()


def _guess_element_from_mol2_fields(atom_name: str, atom_type: str) -> str:
    """Best-effort element inference for permissive MOL2 loading.

    RDKit's native MOL2 parser can reject GAFF/ParmEd-style atom types such as
    ``ho`` or ``ca`` when they appear in exported system MOL2 files. For the
    geometry-reload path we only need chemically plausible elements, connectivity,
    coordinates, and charges, so we infer the element conservatively here.
    """
    atom_name_s = str(atom_name).strip()
    atom_name_head = re.match(r"^([A-Za-z]{1,2})", atom_name_s)
    if atom_name_head:
        token = atom_name_head.group(1)
        mapped = _COMMON_FF_TYPE_ELEMENT_MAP.get(token.lower())
        if mapped is not None:
            return mapped
        name_guess = _elem_from_atom_name(atom_name_s)
        if not atom_name_s.lower().startswith(("xx", "du")):
            # Prefer atom names first when they look element-like; ParmEd/GROMACS
            # names usually preserve the true element even if the MOL2 atom type
            # is a force-field label.
            return name_guess

    raw_type = str(atom_type or "").strip()
    type_head = raw_type.split(".", 1)[0].strip()
    mapped = _COMMON_FF_TYPE_ELEMENT_MAP.get(type_head.lower())
    if mapped:
        return mapped

    if type_head:
        head_guess = _elem_from_atom_name(type_head)
        if head_guess and head_guess != "X":
            return head_guess
    return "C"



import numpy as np


# ---------------------------------------------------------------------------
# MOL2 reader with charge recovery
# ---------------------------------------------------------------------------
def _parse_mol2_atom_charges(mol2_path: Path) -> list[float]:
    """Parse per-atom charges from a MOL2 file.

    We parse @<TRIPOS>ATOM lines and take the last column as charge.
    This is robust across most MOL2 variants YadonPy produces.
    """
    charges: list[float] = []
    in_atom = False
    for raw in Path(mol2_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if line.upper().startswith("@<TRIPOS>") and not line.upper().startswith("@<TRIPOS>ATOM"):
            if in_atom:
                break
            continue
        if in_atom:
            parts = line.split()
            # MOL2 ATOM line typically has 9 columns:
            # id name x y z type subst_id subst_name charge
            if len(parts) < 6:
                continue
            try:
                q = float(parts[-1])
                charges.append(q)
            except Exception:
                # If charge column missing, append 0.0 to keep indices aligned.
                charges.append(0.0)
    return charges


def _mol2_likely_needs_permissive_parser(mol2_path: Path) -> bool:
    in_atom = False
    for raw in Path(mol2_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            continue
        if upper.startswith("@<TRIPOS>") and not upper.startswith("@<TRIPOS>ATOM"):
            if in_atom:
                break
            continue
        if not in_atom:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        atom_type = str(parts[5]).strip().split(".", 1)[0].lower()
        if atom_type in _PERMISSIVE_MOL2_TYPE_TOKENS:
            return True
    return False


def _read_mol2_with_custom_parser(
    mol2_path: Path,
    *,
    charge_prop: str = "AtomicCharge",
    also_resp: bool = True,
):
    """Fallback MOL2 reader that ignores Tripos atom-type chemistry.

    This path exists specifically for exported system MOL2 files where RDKit's
    native MOL2 reader rejects force-field atom types like ``ho`` / ``ca``.
    """
    try:
        from rdkit import Chem
        from rdkit import Geometry as Geom
    except Exception as e:  # pragma: no cover
        raise RuntimeError("RDKit is required to read MOL2") from e

    atoms: list[dict[str, object]] = []
    bonds: list[tuple[int, int, str]] = []
    section = None
    for raw in Path(mol2_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("@<TRIPOS>ATOM"):
            section = "atom"
            continue
        if upper.startswith("@<TRIPOS>BOND"):
            section = "bond"
            continue
        if upper.startswith("@<TRIPOS>"):
            section = None
            continue

        if section == "atom":
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                atom_id = int(parts[0])
                atom_name = str(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                atom_type = str(parts[5])
                charge = float(parts[-1]) if len(parts) >= 9 else 0.0
            except Exception:
                continue
            atoms.append(
                {
                    "id": atom_id,
                    "name": atom_name,
                    "type": atom_type,
                    "x": x,
                    "y": y,
                    "z": z,
                    "charge": charge,
                    "element": _guess_element_from_mol2_fields(atom_name, atom_type),
                }
            )
            continue

        if section == "bond":
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                bonds.append((int(parts[1]), int(parts[2]), str(parts[3]).lower()))
            except Exception:
                continue

    if not atoms:
        raise RuntimeError(f"Failed to read mol2: {mol2_path}")

    rw = Chem.RWMol()
    atom_id_to_idx: dict[int, int] = {}
    for item in atoms:
        rd_atom = Chem.Atom(str(item["element"]))
        rd_atom.SetNoImplicit(True)
        idx = int(rw.AddAtom(rd_atom))
        atom_id_to_idx[int(item["id"])] = idx
        atom_ref = rw.GetAtomWithIdx(idx)
        atom_ref.SetProp("_TriposAtomName", str(item["name"]))
        atom_ref.SetProp("_TriposAtomType", str(item["type"]))
        atom_ref.SetDoubleProp(str(charge_prop), float(item["charge"]))
        if also_resp:
            atom_ref.SetDoubleProp("RESP", float(item["charge"]))

    for a1, a2, btype in bonds:
        if a1 not in atom_id_to_idx or a2 not in atom_id_to_idx:
            continue
        bond_type = Chem.BondType.SINGLE
        if btype == "2":
            bond_type = Chem.BondType.DOUBLE
        elif btype == "3":
            bond_type = Chem.BondType.TRIPLE
        elif btype == "ar":
            bond_type = Chem.BondType.AROMATIC
        try:
            rw.AddBond(atom_id_to_idx[a1], atom_id_to_idx[a2], bond_type)
            if btype == "ar":
                bond = rw.GetBondBetweenAtoms(atom_id_to_idx[a1], atom_id_to_idx[a2])
                if bond is not None:
                    bond.SetIsAromatic(True)
                rw.GetAtomWithIdx(atom_id_to_idx[a1]).SetIsAromatic(True)
                rw.GetAtomWithIdx(atom_id_to_idx[a2]).SetIsAromatic(True)
        except Exception:
            continue

    mol = rw.GetMol()
    conf = Chem.Conformer(int(len(atoms)))
    conf.Set3D(True)
    for idx, item in enumerate(atoms):
        conf.SetAtomPosition(
            idx,
            Geom.Point3D(float(item["x"]), float(item["y"]), float(item["z"])),
        )
    mol.AddConformer(conf, assignId=True)
    return mol


def read_mol2_with_charges(
    mol2_path: Path,
    *,
    sanitize: bool = True,
    removeHs: bool = False,
    charge_prop: str = "AtomicCharge",
    also_resp: bool = True,
):
    """Read MOL2 into RDKit Mol and restore per-atom charges.

    RDKit's MolFromMol2File does not reliably populate charge properties
    across builds. YadonPy therefore re-parses the charge column and writes
    it into atom double props.

    Returns:
        RDKit Mol with charges set on each atom.
    """
    try:
        from rdkit.Chem import rdmolfiles
    except Exception as e:
        raise RuntimeError("RDKit is required to read MOL2") from e

    mol = None
    if not _mol2_likely_needs_permissive_parser(Path(mol2_path)):
        try:
            mol = rdmolfiles.MolFromMol2File(str(mol2_path), sanitize=bool(sanitize), removeHs=bool(removeHs))
        except Exception:
            mol = None
    if mol is None:
        mol = _read_mol2_with_custom_parser(
            Path(mol2_path),
            charge_prop=str(charge_prop),
            also_resp=bool(also_resp),
        )

    charges = _parse_mol2_atom_charges(Path(mol2_path))
    if len(charges) == int(mol.GetNumAtoms()):
        for i, a in enumerate(mol.GetAtoms()):
            try:
                q = float(charges[i])
            except Exception:
                q = 0.0
            a.SetDoubleProp(str(charge_prop), float(q))
            if also_resp:
                try:
                    a.SetDoubleProp("RESP", float(q))
                except Exception:
                    pass
    return mol


@dataclass
class ITPAtomsBonds:
    atom_names: List[str]
    atom_types: List[str]
    charges: List[float]
    coords_ang: np.ndarray  # (n,3)
    bonds: List[Tuple[int, int, str]]  # 1-based indices


def _strip_comment(line: str) -> str:
    if ";" in line:
        line = line.split(";", 1)[0]
    return line.strip()


def _read_gro_coords_ang(gro_path: Path) -> Tuple[List[str], np.ndarray]:
    lines = gro_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid gro file: {gro_path}")
    nat = int(lines[1].strip())
    atom_names: List[str] = []
    coords_nm = np.zeros((nat, 3), dtype=float)
    for i in range(nat):
        l = lines[2 + i]
        atomname = l[10:15].strip() or f"A{i+1}"
        atom_names.append(atomname)
        coords_nm[i, 0] = float(l[20:28])
        coords_nm[i, 1] = float(l[28:36])
        coords_nm[i, 2] = float(l[36:44])
    coords_ang = coords_nm * 10.0
    return atom_names, coords_ang


def _parse_itp_atoms_bonds(itp_path: Path) -> Tuple[List[str], List[str], List[float], List[Tuple[int, int, str]]]:
    """Parse [ atoms ] and [ bonds ] sections from an .itp.

    Returns:
        (atom_names, atom_types, charges, bonds)
    """
    section = None
    atom_names: List[str] = []
    atom_types: List[str] = []
    charges: List[float] = []
    bonds: List[Tuple[int, int, str]] = []

    for raw in itp_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        m = re.match(r"^\[\s*([^\]]+)\s*\]", line)
        if m:
            section = m.group(1).strip().lower()
            continue
        if section == "atoms":
            parts = line.split()
            # nr type resnr resid atom cgnr charge mass
            if len(parts) < 7:
                continue
            atom_types.append(parts[1])
            atom_names.append(parts[4])
            try:
                charges.append(float(parts[6]))
            except Exception:
                charges.append(0.0)
        elif section == "bonds":
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                i = int(parts[0])
                j = int(parts[1])
            except Exception:
                continue
            btype = "1"
            bonds.append((i, j, btype))

    if not atom_names:
        raise ValueError(f"No [ atoms ] parsed from {itp_path}")
    return atom_names, atom_types, charges, bonds


def read_gro_itp_as_mol2_inputs(gro_path: Path, itp_path: Path) -> ITPAtomsBonds:
    atom_names_gro, coords_ang = _read_gro_coords_ang(gro_path)
    atom_names_itp, atom_types, charges, bonds = _parse_itp_atoms_bonds(itp_path)

    # Prefer atom names from itp (they are usually cleaner), but keep consistent length
    atom_names = atom_names_itp if len(atom_names_itp) == len(atom_names_gro) else atom_names_gro
    if len(atom_names) != coords_ang.shape[0]:
        raise ValueError("gro/itp atom count mismatch")
    return ITPAtomsBonds(
        atom_names=atom_names,
        atom_types=atom_types,
        charges=charges,
        coords_ang=coords_ang,
        bonds=bonds,
    )


def write_mol2_from_gro_itp(
    *,
    gro_path: Path,
    itp_path: Path,
    out_mol2: Path,
    mol_name: str,
    charge_scale: float = 1.0,
) -> Path:
    data = read_gro_itp_as_mol2_inputs(gro_path, itp_path)
    charges = [float(q) * float(charge_scale) for q in data.charges]

    out_mol2.parent.mkdir(parents=True, exist_ok=True)
    with out_mol2.open("w", encoding="utf-8") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{mol_name}\n")
        f.write(f"{len(data.atom_names)} {len(data.bonds)} 0 0 0\n")
        f.write("SMALL\n")
        f.write("USER_CHARGES\n\n")

        f.write("@<TRIPOS>ATOM\n")
        for i, (aname, atype, (x, y, z), q) in enumerate(
            zip(data.atom_names, data.atom_types, data.coords_ang, charges), start=1
        ):
            # id name x y z type subst_id subst_name charge
            elem = _elem_from_atom_name(aname)
            atype2 = f"{elem}.{atype}" if atype else elem
            f.write(f"{i:7d} {aname:<8s} {x:10.4f} {y:10.4f} {z:10.4f} {atype2:<10s} 1 {mol_name:<8s} {q: .6f}\n")
        f.write("@<TRIPOS>BOND\n")
        for k, (i, j, btype) in enumerate(data.bonds, start=1):
            f.write(f"{k:6d} {i:4d} {j:4d} {btype}\n")

    return out_mol2


def write_mol2(
    *,
    mol,
    name: str | None = None,
    out_mol2: Path | None = None,
    out_dir: Path | None = None,
    mol_name: str | None = None,
    charge_prop: str = "AtomicCharge",
    charge_scale: float = 1.0,
    use_raw: bool = False,
 ) -> Path:
    """Write a MOL2 using RDKit mol geometry and per-atom charges.

    If the requested charge property does not exist, we fall back to RESP,
    then to 0.0.
    """
    # Resolve naming / output path.
    #
    # Semantics:
    #   - `name` controls the output filename stem (<name>.mol2).
    #     If not provided, we infer it from the caller's Python variable name
    #     (best-effort), falling back to a stable auto-name.
    #   - `mol_name` controls the MOL2 residue / molecule name inside the file.
    #     If not provided, it defaults to the resolved `name`.
    try:
        from ..core import utils
        # Ensure consistent naming with exported GROMACS artifacts.
        # Prefer the caller's Python variable name when available (e.g. copoly, solvent_A).
        # This avoids opaque auto-names like "C2H6O_8d9587" in examples/00_molecules.
        _stem = utils.ensure_name(mol, name=name, depth=2, prefer_var=True)
    except Exception:
        _stem = (name or "molecule")

    # Backward-compatibility: if `mol_name` was used as the naming knob,
    # prefer it as a filename stem when `name` is not explicitly set.
    if name is None and mol_name is not None and str(mol_name).strip() != "":
        _stem = str(mol_name).strip()

    if mol_name is None:
        mol_name = _stem

    if out_mol2 is None:
        if out_dir is None:
            out_dir = Path(".")
        out_mol2 = Path(out_dir) / f"{_stem}.mol2"

    conf = mol.GetConformer()
    out_mol2.parent.mkdir(parents=True, exist_ok=True)

    charge_order = []
    if charge_prop:
        if use_raw:
            charge_order.append(f"{charge_prop}_raw")
        charge_order.append(charge_prop)
    if use_raw:
        charge_order.extend(["RESP_raw", "AtomicCharge_raw"])
    charge_order.extend(["RESP", "AtomicCharge"])
    selected_charge_prop, _ = core_utils.select_best_charge_property(mol, preferred_props=tuple(charge_order))

    def _get_charge(a) -> float:
        """Return atomic charge from RDKit atom properties.

        If use_raw=True and a corresponding *_raw property exists, that value is used.
        """
        if selected_charge_prop:
            v = core_utils._atom_charge_from_prop(a, selected_charge_prop)
            if v is not None:
                return float(v)
        return 0.0

    # Build bonds
    bonds: List[Tuple[int, int, str]] = []
    for b in mol.GetBonds():
        i = int(b.GetBeginAtomIdx()) + 1
        j = int(b.GetEndAtomIdx()) + 1
        bt = "1"
        if b.GetIsAromatic():
            bt = "ar"
        else:
            order = int(b.GetBondTypeAsDouble())
            if order == 2:
                bt = "2"
            elif order == 3:
                bt = "3"
        bonds.append((i, j, bt))

    with out_mol2.open("w", encoding="utf-8") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{mol_name}\n")
        f.write(f"{mol.GetNumAtoms()} {len(bonds)} 0 0 0\n")
        f.write("SMALL\n")
        f.write("USER_CHARGES\n\n")

        f.write("@<TRIPOS>ATOM\n")
        for i, a in enumerate(mol.GetAtoms(), start=1):
            p = conf.GetAtomPosition(i - 1)
            x, y, z = float(p.x), float(p.y), float(p.z)
            aname = f"{a.GetSymbol()}{i}"
            atype = a.GetProp("ff_type") if a.HasProp("ff_type") else a.GetSymbol()
            q = _get_charge(a) * float(charge_scale)
            elem = _elem_from_atom_name(aname)
            atype2 = f"{elem}.{atype}" if atype else elem
            f.write(f"{i:7d} {aname:<8s} {x:10.4f} {y:10.4f} {z:10.4f} {atype2:<10s} 1 {mol_name:<8s} {q: .6f}\n")

        f.write("@<TRIPOS>BOND\n")
        for k, (i, j, bt) in enumerate(bonds, start=1):
            f.write(f"{k:6d} {i:4d} {j:4d} {bt}\n")

    return out_mol2


# ----------------------
# System-level mol2 export (GROMACS top + gro) via ParmEd
# ----------------------


def write_mol2_from_top_gro_parmed(
    *,
    top_path: Path,
    gro_path: Path,
    out_mol2: Optional[Path] = None,
    overwrite: bool = True,
) -> Optional[Path]:
    """Write a **system** MOL2 from a GROMACS ``.top`` + coordinate ``.gro`` using ParmEd.

    This is intended for *debugging / visualization / interoperability*.
    It preserves the full system (all residues/ions/solvents) in a single MOL2.

    Notes
    -----
    - ``top_path`` is resolved to an **absolute path** before passing to ParmEd,
      to avoid surprises with changing working directories.
    - Best-effort: returns ``None`` if ParmEd is unavailable or the conversion fails.

    Args:
        top_path: path to ``system.top`` (or any GROMACS topology file)
        gro_path: coordinate file (e.g. ``md.gro``)
        out_mol2: output path. Default: same folder as gro with ``.mol2`` suffix.
        overwrite: overwrite existing file

    Returns:
        Path to the written mol2, or None on failure.
    """
    top_path = Path(top_path).expanduser().resolve()
    gro_path = Path(gro_path).expanduser().resolve()
    if out_mol2 is None:
        out_mol2 = gro_path.with_suffix(".mol2")
    out_mol2 = Path(out_mol2).expanduser()
    out_mol2.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_mol2 = (out_mol2.parent.resolve() / out_mol2.name)
    except Exception:
        pass

    try:
        import parmed as pmd  # type: ignore
    except Exception:
        return None

    try:
        import warnings
        with warnings.catch_warnings():
            # ParmEd does not generate 1-4 pairs from [defaults] gen-pairs; it warns and sets them to zero
            # internally. This is harmless for MOL2 export (topology is not used downstream) and would otherwise
            # spam the console. We suppress only these warnings.
            try:
                from parmed.gromacs.gromacstop import GromacsWarning  # type: ignore
                warnings.filterwarnings("ignore", category=GromacsWarning)
            except Exception:
                warnings.filterwarnings("ignore", message=r".*1-4 pairs were missing from the \[ pairs \].*")
            s = pmd.load_file(str(top_path), xyz=str(gro_path))
        s.save(str(out_mol2), overwrite=bool(overwrite))
        # Guard against silent failures
        if not out_mol2.exists() or out_mol2.stat().st_size == 0:
            return None
        return out_mol2
    except Exception:
        return None



def write_mol2_from_rdkit(*args, **kwargs):
    """Backward-compatible alias for ``write_mol2``."""
    return write_mol2(*args, **kwargs)


__all__ = [
    "read_mol2_with_charges",
    "write_mol2",
    "write_mol2_from_rdkit",
    "write_mol2_from_gro_itp",
    "write_mol2_from_top_gro_parmed",
]
