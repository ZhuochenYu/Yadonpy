"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .topology import parse_system_top


def _is_h_atom(atomname: str, atomtype: str) -> bool:
    """Heuristic for hydrogen atoms/types.

    We default to excluding hydrogens from RDF counting, but provide a switch
    in analysis to include them.
    """
    an = (atomname or "").strip()
    at = (atomtype or "").strip()
    if an.upper().startswith("H"):
        return True
    # common force-field type prefixes
    if at.lower().startswith("h"):
        return True
    return False


def generate_system_ndx(
    *,
    top_path: Path,
    ndx_path: Path,
    include_h_atomtypes: bool = False,
) -> None:
    """Generate system.ndx based on system.top and included ITPs.

    Groups generated (deterministic):
      - <molname> and MOL_<molname> : all atoms of each molecule type
      - REP_<molname> : representative atoms (first atom of each molecule instance)
      - TYPE_<molname>_<atomtype> : atoms of a given atomtype within a moltype

    This mirrors the grouping strategy used in yzc-gmx-gen and is robust for
    mixed systems because it relies on topology ordering.
    """
    topo = parse_system_top(top_path)

    # Build global atom index mapping assuming the system.gro matches the
    # [molecules] ordering, and each molecule instance uses the atom order
    # defined in its ITP.
    current = 1

    groups: List[Tuple[str, List[int]]] = []
    system_atoms: List[int] = []

    ions: List[int] = []
    cations: List[int] = []
    anions: List[int] = []

    for molname, count in topo.molecules:
        mt = topo.moleculetypes.get(molname)
        if mt is None:
            # silently skip unknown includes
            continue

        mol_atoms: List[int] = []
        rep_atoms: List[int] = []
        type_to_atoms: Dict[str, List[int]] = {}

        for _k in range(int(count)):
            start = current
            # representative atom: the first atom of this molecule instance
            rep_atoms.append(start)

            for i in range(mt.natoms):
                idx = current
                mol_atoms.append(idx)
                system_atoms.append(idx)
                atype = mt.atomtypes[i]
                aname = mt.atomnames[i]
                if (not include_h_atomtypes) and _is_h_atom(aname, atype):
                    pass
                else:
                    type_to_atoms.setdefault(atype, []).append(idx)
                current += 1

            # ion classification by net charge of the molecule type
            # (based on ITP charges). A moltype with |net_charge| < 0.1 is treated as neutral.
        # after instances

        groups.append((f"{molname}", mol_atoms))
        groups.append((f"MOL_{molname}", mol_atoms))
        groups.append((f"REP_{molname}", rep_atoms))
        for atype, idxs in sorted(type_to_atoms.items(), key=lambda x: x[0]):
            groups.append((f"TYPE_{molname}_{atype}", idxs))

        q = mt.net_charge
        if abs(q) >= 0.1:
            ions.extend(mol_atoms)
            if q > 0:
                cations.extend(mol_atoms)
            else:
                anions.extend(mol_atoms)

    # top-level ion groups
    if ions:
        groups.append(("IONS", sorted(set(ions))))
    if cations:
        groups.append(("CATIONS", sorted(set(cations))))
    if anions:
        groups.append(("ANIONS", sorted(set(anions))))

    if system_atoms:
        groups.insert(0, ("System", sorted(set(system_atoms))))

    _write_ndx(ndx_path, groups)


def _write_ndx(path: Path, groups: List[Tuple[str, List[int]]]) -> None:
    lines: List[str] = []
    for name, idxs in groups:
        lines.append(f"[ {name} ]")
        # write 15 per line (gmx style)
        row: List[str] = []
        for i, idx in enumerate(idxs, 1):
            row.append(str(int(idx)))
            if i % 15 == 0:
                lines.append(" ".join(row))
                row = []
        if row:
            lines.append(" ".join(row))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
