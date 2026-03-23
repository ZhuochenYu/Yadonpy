"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
