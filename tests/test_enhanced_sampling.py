from __future__ import annotations

import json
from pathlib import Path

from yadonpy.interface import SolvatedIonPullSpec, prepare_solvated_ion_pull


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
    (mol_dir / f"{moltype}.itp").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _gro_line(resid: int, resname: str, atomname: str, atomnr: int, xyz: tuple[float, float, float]) -> str:
    return f"{resid:5d}{resname:<5}{atomname:>5}{atomnr:5d}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"


def _write_four_coordinate_li_stack(system_dir: Path) -> None:
    system_dir.mkdir(parents=True, exist_ok=True)
    _write_itp(system_dir, "CMCNA", [("c3", "C1", 0.1, 12.011), ("os", "O1", -0.5, 15.999)])
    _write_itp(system_dir, "Li", [("Li", "Li1", 1.0, 6.94)])
    _write_itp(system_dir, "EC", [("o", "O1", -0.3, 15.999)])
    _write_itp(system_dir, "PF6", [("f", "F1", -1.0, 18.998)])
    (system_dir / "system.top").write_text(
        '#include "molecules/CMCNA/CMCNA.itp"\n'
        '#include "molecules/Li/Li.itp"\n'
        '#include "molecules/EC/EC.itp"\n'
        '#include "molecules/PF6/PF6.itp"\n\n'
        "[ molecules ]\n"
        "CMCNA 2\n"
        "Li 2\n"
        "EC 4\n"
        "PF6 1\n",
        encoding="utf-8",
    )
    atoms = [
        ("CMCNA", "C1", (1.50, 1.50, 1.20)),
        ("CMCNA", "O1", (1.50, 1.50, 1.40)),
        ("CMCNA", "C1", (1.80, 1.50, 1.55)),
        ("CMCNA", "O1", (1.80, 1.50, 1.75)),
        ("Li", "Li1", (1.50, 1.50, 2.50)),  # selected: four solvent oxygens nearby
        ("Li", "Li1", (2.60, 2.60, 4.20)),
        ("EC", "O1", (1.68, 1.50, 2.50)),
        ("EC", "O1", (1.32, 1.50, 2.50)),
        ("EC", "O1", (1.50, 1.68, 2.50)),
        ("EC", "O1", (1.50, 1.32, 2.50)),
        ("PF6", "F1", (1.50, 1.50, 3.20)),
    ]
    gro_lines = ["synthetic solvated Li pull", f"{len(atoms):5d}"]
    for atomnr, (resname, atomname, xyz) in enumerate(atoms, start=1):
        gro_lines.append(_gro_line(atomnr, resname, atomname, atomnr, xyz))
    gro_lines.append("   4.00000   4.00000   6.00000")
    (system_dir / "system.gro").write_text("\n".join(gro_lines) + "\n", encoding="utf-8")
    (system_dir / "system.ndx").write_text(
        "[ CMCNA ]\n1 2 3 4\n\n"
        "[ ELECTROLYTE ]\n5 6 7 8 9 10 11\n",
        encoding="utf-8",
    )


def test_prepare_solvated_ion_pull_selects_four_coordinate_li_and_writes_plumed(tmp_path: Path):
    system_dir = tmp_path / "02_system"
    _write_four_coordinate_li_stack(system_dir)

    plan = prepare_solvated_ion_pull(
        system_dir=system_dir,
        spec=SolvatedIonPullSpec(
            target_group="CMCNA",
            target_coordination_number=4,
            step1=2000,
            kappa1_kj_mol_nm2=500.0,
            print_stride=20,
        ),
    )

    assert plan.selected_center_atom == 5
    assert plan.mdrun_extra_args == ("-plumed", str(plan.plumed_dat))
    assert plan.plumed_dat.exists()
    assert plan.ndx_path.exists()
    manifest = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_center"]["solvent_ligand_count"] == 4
    assert manifest["selected_center"]["target_ligand_count"] == 0
    assert manifest["ligand_atom_counts"] == {"solvent": 4, "target": 2, "anion": 1}
    text = plan.plumed_dat.read_text(encoding="utf-8")
    assert "MOVINGRESTRAINT" in text
    assert "cn_solvent: COORDINATION" in text
    assert "cn_target: COORDINATION" in text
    assert "cn_anion: COORDINATION" in text
    assert "PRINT STRIDE=20" in text
    assert "-plumed" in manifest["mdrun_extra_args"]


def test_prepare_solvated_ion_pull_requires_target_group(tmp_path: Path):
    system_dir = tmp_path / "02_system"
    _write_four_coordinate_li_stack(system_dir)

    try:
        prepare_solvated_ion_pull(system_dir=system_dir, spec=SolvatedIonPullSpec(target_group="MISSING"))
    except ValueError as exc:
        assert "Target group" in str(exc)
    else:
        raise AssertionError("prepare_solvated_ion_pull should require the target ndx group")
