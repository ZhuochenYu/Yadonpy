from __future__ import annotations

import json
from pathlib import Path

from yadonpy.gmx.analysis.interface_profile import compute_interface_profile
from yadonpy.interface import analyze_sandwich_interface


def _write_itp(system_dir: Path, moltype: str, atoms: list[tuple[str, str, float, float]]) -> None:
    mol_dir = system_dir / "molecules" / moltype
    mol_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "[ moleculetype ]",
        f"{moltype} 3",
        "",
        "[ atoms ]",
    ]
    for idx, (atype, aname, charge, mass) in enumerate(atoms, start=1):
        lines.append(f"{idx:5d} {atype:<6} 1 {moltype:<6} {aname:<6} {idx:5d} {charge: .6f} {mass:.4f}")
    lines.append("")
    lines.append("[ bonds ]")
    for idx in range(1, len(atoms)):
        lines.append(f"{idx} {idx + 1}")
    (mol_dir / f"{moltype}.itp").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gro_line(resid: int, resname: str, atomname: str, atomnr: int, xyz: tuple[float, float, float]) -> str:
    return f"{resid:5d}{resname:<5}{atomname:>5}{atomnr:5d}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"


def _write_synthetic_stack(system_dir: Path, *, direct_contact: bool = False) -> None:
    system_dir.mkdir(parents=True, exist_ok=True)
    _write_itp(system_dir, "GRAPH", [("cgraph", "C1", 0.0, 12.011), ("cgraph", "C2", 0.0, 12.011)])
    _write_itp(system_dir, "POLY", [("c3", "C1", 0.1, 12.011), ("os", "O2", -0.2, 15.999), ("c3", "C3", 0.1, 12.011)])
    _write_itp(system_dir, "Li", [("Li", "Li1", 1.0, 6.94)])
    _write_itp(system_dir, "SOLV", [("c", "C1", 0.3, 12.011), ("o", "O2", -0.3, 15.999)])
    _write_itp(system_dir, "PF6", [("p", "P1", 1.0, 30.974), ("f", "F2", -1.0, 18.998)])
    (system_dir / "system.top").write_text(
        '#include "molecules/GRAPH/GRAPH.itp"\n'
        '#include "molecules/POLY/POLY.itp"\n'
        '#include "molecules/Li/Li.itp"\n'
        '#include "molecules/SOLV/SOLV.itp"\n'
        '#include "molecules/PF6/PF6.itp"\n\n'
        "[ molecules ]\n"
        "GRAPH 1\n"
        "POLY 2\n"
        "Li 1\n"
        "SOLV 1\n"
        "PF6 1\n",
        encoding="utf-8",
    )
    (system_dir / "system_meta.json").write_text(
        json.dumps(
            {
                "species": [
                    {"moltype": "GRAPH", "kind": "substrate", "smiles": "graphite", "n": 1, "formal_charge": 0.0},
                    {"moltype": "POLY", "kind": "polymer", "smiles": "*CCO*", "n": 2, "formal_charge": 0.0},
                    {"moltype": "Li", "kind": "ion", "smiles": "[Li+]", "n": 1, "formal_charge": 1.0},
                    {"moltype": "SOLV", "kind": "solvent", "smiles": "CO", "n": 1, "formal_charge": 0.0},
                    {"moltype": "PF6", "kind": "ion", "smiles": "F[P-](F)(F)(F)(F)F", "n": 1, "formal_charge": -1.0},
                ],
                "box_lengths_nm": [3.0, 3.0, 5.0],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    electrolyte_z = 0.42 if direct_contact else 2.66
    atoms = [
        ("GRAPH", "C1", (1.00, 1.00, 0.20)),
        ("GRAPH", "C2", (1.40, 1.00, 0.30)),
        ("POLY", "C1", (1.00, 1.00, 1.10)),
        ("POLY", "O2", (1.00, 1.00, 1.50)),
        ("POLY", "C3", (1.00, 1.00, 1.90)),
        ("POLY", "C1", (1.00, 1.00, 2.30)),
        ("POLY", "O2", (1.00, 1.00, 2.64)),
        ("POLY", "C3", (1.00, 1.00, 2.80)),
        ("Li", "Li1", (1.00, 1.00, electrolyte_z)),
        ("SOLV", "C1", (1.20, 1.00, max(electrolyte_z, 2.70))),
        ("SOLV", "O2", (1.02, 1.00, max(electrolyte_z + 0.02, 2.68))),
        ("PF6", "P1", (1.00, 1.30, max(electrolyte_z, 2.74))),
        ("PF6", "F2", (1.00, 1.16, max(electrolyte_z, 2.66))),
    ]
    gro_lines = ["synthetic interface stack", f"{len(atoms):5d}"]
    for atomnr, (resname, atomname, xyz) in enumerate(atoms, start=1):
        gro_lines.append(_gro_line(atomnr, resname, atomname, atomnr, xyz))
    gro_lines.append("   3.00000   3.00000   5.00000")
    (system_dir / "system.gro").write_text("\n".join(gro_lines) + "\n", encoding="utf-8")
    (system_dir / "system.ndx").write_text(
        "[ GRAPHITE ]\n1 2\n\n"
        "[ POLYMER ]\n3 4 5 6 7 8\n\n"
        "[ ELECTROLYTE ]\n9 10 11 12 13\n",
        encoding="utf-8",
    )


def test_interface_profile_extracts_phase_order_density_overlap_and_coordination(tmp_path: Path):
    system_dir = tmp_path / "00_stack"
    _write_synthetic_stack(system_dir)

    out = compute_interface_profile(
        gro_path=system_dir / "system.gro",
        top_path=system_dir / "system.top",
        ndx_path=system_dir / "system.ndx",
        system_dir=system_dir,
        out_dir=tmp_path / "analysis",
        xtc_path=None,
        bin_nm=0.10,
        compute_transport=False,
    )

    assert out["geometry_health"]["phase_order"] == ["GRAPHITE", "POLYMER", "ELECTROLYTE"]
    assert out["geometry_health"]["phase_order_ok"] is True
    assert out["geometry_health"]["direct_graphite_electrolyte_contact"] is False
    assert out["region_summary"]["interpenetration"]["overlap_width_nm"] > 0.0
    assert out["coordination_by_region"]["available"] is True
    assert (tmp_path / "analysis" / "z_density_profiles.csv").exists()
    assert (tmp_path / "analysis" / "region_summary.json").exists()
    assert (tmp_path / "analysis" / "coordination_by_region.json").exists()


def test_interface_profile_detects_graphite_electrolyte_direct_contact(tmp_path: Path):
    system_dir = tmp_path / "00_stack"
    _write_synthetic_stack(system_dir, direct_contact=True)

    out = compute_interface_profile(
        gro_path=system_dir / "system.gro",
        top_path=system_dir / "system.top",
        ndx_path=system_dir / "system.ndx",
        system_dir=system_dir,
        out_dir=tmp_path / "analysis",
        xtc_path=None,
        bin_nm=0.10,
        compute_transport=False,
    )

    assert out["geometry_health"]["direct_graphite_electrolyte_contact"] is True


def test_analyze_sandwich_interface_wrapper_accepts_static_stack(tmp_path: Path):
    system_dir = tmp_path / "case" / "00_stack"
    _write_synthetic_stack(system_dir)

    out = analyze_sandwich_interface(
        work_dir=tmp_path / "case",
        compute_transport=False,
        bin_nm=0.10,
    )

    assert out["geometry_health"]["phase_order_ok"] is True
    assert (tmp_path / "case" / "06_analysis" / "interface_profile" / "interface_profile_summary.json").exists()
