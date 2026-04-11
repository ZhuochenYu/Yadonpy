"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class MoleculeType:
    name: str
    atomtypes: List[str]
    atomnames: List[str]
    charges: List[float]
    masses: List[float]
    bonds: List[Tuple[int, int]]

    @property
    def natoms(self) -> int:
        return len(self.atomtypes)

    @property
    def net_charge(self) -> float:
        return float(sum(self.charges))

    @property
    def total_mass(self) -> float:
        return float(sum(self.masses))


@dataclass
class SystemTopology:
    moleculetypes: Dict[str, MoleculeType]
    molecules: List[Tuple[str, int]]  # (molname, count)


@dataclass
class AtomTypeParam:
    name: str
    mass: float
    sigma_nm: float
    epsilon_kj: float


def _strip_comment(line: str) -> str:
    # GROMACS comments use ';'
    return line.split(';', 1)[0].strip()


def parse_itp(itp_path: Path) -> Optional[MoleculeType]:
    """Parse a minimal subset of .itp sufficient for ndx generation and NE conductivity.

    We parse:
      - [ moleculetype ] name
      - [ atoms ]: type, atom name, charge
    """
    lines = itp_path.read_text(encoding="utf-8", errors="replace").splitlines()
    section = None
    mol_name: Optional[str] = None
    atomtypes: List[str] = []
    atomnames: List[str] = []
    charges: List[float] = []
    masses: List[float] = []
    bonds: List[Tuple[int, int]] = []

    for raw in lines:
        line = _strip_comment(raw)
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            section = line.strip('[]').strip().lower()
            continue
        if section == 'moleculetype':
            # first token is name
            parts = line.split()
            if parts:
                mol_name = parts[0]
        elif section == 'atoms':
            # expected format:
            # nr type resnr residue atom cgnr charge mass
            parts = line.split()
            if len(parts) < 7:
                continue
            atype = parts[1]
            aname = parts[4]
            try:
                q = float(parts[6])
            except Exception:
                q = 0.0
            atomtypes.append(atype)
            atomnames.append(aname)
            charges.append(q)
            try:
                m = float(parts[7]) if len(parts) > 7 else 0.0
            except Exception:
                m = 0.0
            masses.append(m)
        elif section == 'bonds':
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                ai = int(parts[0])
                aj = int(parts[1])
            except Exception:
                continue
            if ai >= 1 and aj >= 1:
                bonds.append((ai, aj))

    if mol_name is None or not atomtypes:
        return None
    return MoleculeType(name=mol_name, atomtypes=atomtypes, atomnames=atomnames, charges=charges, masses=masses, bonds=bonds)


def parse_system_top(top_path: Path) -> SystemTopology:
    """Parse system.top and included .itp files (relative includes)."""
    base = top_path.parent
    txt = top_path.read_text(encoding="utf-8", errors="replace").splitlines()

    includes: List[Path] = []
    molecules: List[Tuple[str, int]] = []
    section = None

    for raw in txt:
        line = _strip_comment(raw)
        if not line:
            continue
        if line.lower().startswith('#include'):
            # #include "path"
            p = line.split(None, 1)[1].strip().strip('"')
            inc = (base / p).resolve()
            if inc.exists():
                includes.append(inc)
            continue
        if line.startswith('[') and line.endswith(']'):
            section = line.strip('[]').strip().lower()
            continue
        if section == 'molecules':
            parts = line.split()
            if len(parts) >= 2:
                try:
                    count = int(float(parts[1]))
                except Exception:
                    continue
                molecules.append((parts[0], count))

    moleculetypes: Dict[str, MoleculeType] = {}
    for inc in includes:
        mt = parse_itp(inc)
        if mt is not None:
            moleculetypes[mt.name] = mt

    return SystemTopology(moleculetypes=moleculetypes, molecules=molecules)


def parse_defined_atomtypes_from_itp(itp_path: Path) -> List[str]:
    """Parse [ atomtypes ] names from an .itp file."""

    lines = itp_path.read_text(encoding="utf-8", errors="replace").splitlines()
    section = None
    atomtypes: List[str] = []

    for raw in lines:
        line = _strip_comment(raw)
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip().lower()
            continue
        if section != "atomtypes":
            continue
        parts = line.split()
        if not parts:
            continue
        token = parts[0].strip()
        if not token or token.lower() == "name":
            continue
        atomtypes.append(token)

    return atomtypes


def parse_atomtype_params_from_itp(itp_path: Path) -> Dict[str, AtomTypeParam]:
    """Parse [ atomtypes ] records from a molecule or force-field ITP file."""

    lines = itp_path.read_text(encoding="utf-8", errors="replace").splitlines()
    section = None
    out: Dict[str, AtomTypeParam] = {}

    for raw in lines:
        line = _strip_comment(raw)
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip().lower()
            continue
        if section != "atomtypes":
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        name = parts[0].strip()
        if not name or name.lower() == "name":
            continue
        try:
            mass = float(parts[1])
        except Exception:
            mass = 0.0
        try:
            sigma_nm = float(parts[-2])
            epsilon_kj = float(parts[-1])
        except Exception:
            sigma_nm = 0.0
            epsilon_kj = 0.0
        out[name] = AtomTypeParam(
            name=name,
            mass=float(mass),
            sigma_nm=float(sigma_nm),
            epsilon_kj=float(epsilon_kj),
        )
    return out


def parse_defined_atomtypes_from_system_top(top_path: Path) -> List[str]:
    """Collect unique atomtypes defined by includes referenced from system.top."""

    base = top_path.parent
    txt = top_path.read_text(encoding="utf-8", errors="replace").splitlines()
    seen: Set[str] = set()
    ordered: List[str] = []

    for raw in txt:
        line = _strip_comment(raw)
        if not line or not line.lower().startswith("#include"):
            continue
        inc_rel = line.split(None, 1)[1].strip().strip('"')
        inc = (base / inc_rel).resolve()
        if not inc.exists():
            continue
        for atomtype in parse_defined_atomtypes_from_itp(inc):
            if atomtype not in seen:
                seen.add(atomtype)
                ordered.append(atomtype)
    return ordered


def parse_system_atomtype_params(top_path: Path) -> Dict[str, AtomTypeParam]:
    """Collect atomtype LJ parameters from ITPs included by system.top."""

    base = top_path.parent
    txt = top_path.read_text(encoding="utf-8", errors="replace").splitlines()
    merged: Dict[str, AtomTypeParam] = {}

    for raw in txt:
        line = _strip_comment(raw)
        if not line or not line.lower().startswith("#include"):
            continue
        inc_rel = line.split(None, 1)[1].strip().strip('"')
        inc = (base / inc_rel).resolve()
        if not inc.exists():
            continue
        for name, param in parse_atomtype_params_from_itp(inc).items():
            merged[name] = param
    return merged
