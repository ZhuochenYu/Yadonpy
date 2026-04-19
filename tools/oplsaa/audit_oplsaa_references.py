#!/usr/bin/env python3
"""Audit YadonPy's OPLS-AA implementation against GROMACS and Moltemplate references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yadonpy.ff.oplsaa_reference import audit_oplsaa_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", help="Optional SMILES to assign and audit.")
    parser.add_argument("--gromacs-root", help="Path to a GROMACS oplsaa.ff directory.")
    parser.add_argument("--moltemplate-par", help="Path to Moltemplate OPLSAA2024 .par file.")
    parser.add_argument("--moltemplate-lt", help="Path to Moltemplate OPLSAA2024 .lt file.")
    parser.add_argument("--charge", default="opls", help="Charge mode used when assigning the optional molecule.")
    parser.add_argument("--export-topology", action="store_true", help="Also export a temporary topology and audit impropers in the ITP.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    report = audit_oplsaa_reference(
        smiles=args.smiles,
        gromacs_root=args.gromacs_root,
        moltemplate_par_path=args.moltemplate_par,
        moltemplate_lt_path=args.moltemplate_lt,
        charge=args.charge,
        export_topology=bool(args.export_topology),
    )

    summary = {
        "defaults_match": bool(report["defaults_parity"]["matches"]),
        "atomtype_lj_mismatches": len(report["atomtype_lj_parity"]["mismatches"]),
        "bond_mismatches": len(report["bond_angle_dihedral_parity"]["bond_mismatches"]),
        "angle_mismatches": len(report["bond_angle_dihedral_parity"]["angle_mismatches"]),
        "dihedral_mismatches": len(report["bond_angle_dihedral_parity"]["dihedral_mismatches"]),
        "locally_patched": len(report["locally_patched"]),
        "assignment_complete": bool(report.get("assignment", {}).get("assignment_complete", True)),
        "topology_complete": bool(report.get("topology", {}).get("topology_complete", True)),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
