from __future__ import annotations

import json
from pathlib import Path

from yadonpy.interface import (
    SolvatedIonPullSpec,
    SolvatedIonUmbrellaSpec,
    analyze_umbrella_pmf,
    prepare_solvated_ion_pull,
    prepare_solvated_ion_umbrella,
    run_solvated_ion_umbrella,
)


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


def test_prepare_solvated_ion_umbrella_writes_windows_mdp_plumed_and_wham_lists(tmp_path: Path):
    system_dir = tmp_path / "02_system"
    _write_four_coordinate_li_stack(system_dir)

    plan = prepare_solvated_ion_umbrella(
        system_dir=system_dir,
        out_dir=tmp_path / "umbrella",
        spec=SolvatedIonUmbrellaSpec(
            target_group="CMCNA",
            target_coordination_number=4,
            target_offset_nm=0.0,
        ),
    )

    assert plan.selected_center_atom == 5
    assert len(plan.window_centers_nm) == 31
    assert abs(plan.window_centers_nm[0] - plan.initial_offset_nm) < 1.0e-9
    assert abs(plan.window_centers_nm[-1] - 0.0) < 1.0e-9
    assert plan.umbrella_ndx_path.exists()
    mdp = Path(plan.windows[0]["production_mdp"]).read_text(encoding="utf-8")
    assert "pull                      = yes" in mdp
    assert "pull-coord1-type          = umbrella" in mdp
    assert "pull-coord1-geometry      = direction" in mdp
    assert "pull-coord1-dim           = N N Y" in mdp
    assert "pull-nstxout" in mdp
    assert "pull-nstfout" in mdp
    plumed = Path(plan.windows[0]["plumed_dat"]).read_text(encoding="utf-8")
    assert "cn_solvent: COORDINATION" in plumed
    assert "cn_target: COORDINATION" in plumed
    assert "cn_anion: COORDINATION" in plumed
    assert "MOVINGRESTRAINT" not in plumed
    assert "RESTRAINT" not in plumed
    assert "PRINT STRIDE=1000" in plumed

    tpr_lines = (plan.wham_dir / "tpr-files.dat").read_text(encoding="utf-8").splitlines()
    pullx_lines = (plan.wham_dir / "pullx-files.dat").read_text(encoding="utf-8").splitlines()
    pullf_lines = (plan.wham_dir / "pullf-files.dat").read_text(encoding="utf-8").splitlines()
    assert tpr_lines == [str(Path(win["production_tpr"])) for win in plan.windows]
    assert pullx_lines == [str(Path(win["production_pullx"])) for win in plan.windows]
    assert pullf_lines == [str(Path(win["production_pullf"])) for win in plan.windows]


def test_analyze_umbrella_pmf_merges_fake_wham_and_colvar_outputs(tmp_path: Path):
    system_dir = tmp_path / "02_system"
    _write_four_coordinate_li_stack(system_dir)
    plan = prepare_solvated_ion_umbrella(
        system_dir=system_dir,
        out_dir=tmp_path / "umbrella",
        spec=SolvatedIonUmbrellaSpec(window_count=5, target_offset_nm=0.0),
    )

    plan.wham_dir.mkdir(parents=True, exist_ok=True)
    plan.wham_dir.joinpath("pmf.xvg").write_text(
        "\n".join(["@ xaxis label \"z\"", "@ yaxis label \"PMF\""] + [f"{i * 0.25:.3f} {float(i):.3f}" for i in range(5)]) + "\n",
        encoding="utf-8",
    )
    hist_lines = ["@ xaxis label \"z\""]
    for i in range(6):
        x = i * 0.2
        values = [max(0.0, 1.0 - abs(x - j * 0.2)) for j in range(5)]
        hist_lines.append(" ".join([f"{x:.3f}", *[f"{v:.3f}" for v in values]]))
    plan.wham_dir.joinpath("histogram.xvg").write_text("\n".join(hist_lines) + "\n", encoding="utf-8")
    for win in plan.windows:
        colvar = Path(win["production_colvar"])
        colvar.parent.mkdir(parents=True, exist_ok=True)
        center = float(win["center_nm"])
        idx = int(win["index"])
        colvar.write_text(
            "#! FIELDS time d_ion_target.z cn_solvent cn_target cn_anion\n"
            f"0.0 {center:.4f} {4.0 - idx * 0.5:.3f} {idx * 0.5:.3f} 0.0\n"
            f"1.0 {center + 0.01:.4f} {3.8 - idx * 0.5:.3f} {idx * 0.5 + 0.2:.3f} 0.0\n",
            encoding="utf-8",
        )

    result = analyze_umbrella_pmf(plan)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["pmf"]["available"] is True
    assert summary["histogram"]["available"] is True
    assert summary["coordination"]["available"] is True
    assert result.pmf_csv is not None and result.pmf_csv.exists()
    assert result.histogram_csv is not None and result.histogram_csv.exists()
    assert (plan.postprocess_dir / "merged_colvar.csv").exists()
    assert (plan.postprocess_dir / "coordination_by_window.csv").exists()
    assert (plan.postprocess_dir / "pmf.svg").exists()
    assert (plan.postprocess_dir / "coordination_vs_reaction_coordinate.svg").exists()
    assert "available" in summary["mp4"]


def test_run_solvated_ion_umbrella_accepts_fake_runner(tmp_path: Path):
    system_dir = tmp_path / "02_system"
    _write_four_coordinate_li_stack(system_dir)
    plan = prepare_solvated_ion_umbrella(
        system_dir=system_dir,
        out_dir=tmp_path / "umbrella",
        spec=SolvatedIonUmbrellaSpec(window_count=3, steering_ns=0.001, window_equilibration_ns=0.001, window_production_ns=0.001),
    )

    class FakeRunner:
        def grompp(self, *, out_tpr, **_kwargs):
            Path(out_tpr).parent.mkdir(parents=True, exist_ok=True)
            Path(out_tpr).write_text("fake tpr\n", encoding="utf-8")

        def mdrun(self, *, cwd, deffnm, mdrun_extra_args=None, **_kwargs):
            cwd = Path(cwd)
            cwd.mkdir(parents=True, exist_ok=True)
            (cwd / f"{deffnm}.gro").write_text("fake\n0\n   4.0   4.0   6.0\n", encoding="utf-8")
            (cwd / f"{deffnm}.xtc").write_bytes(b"fake")
            (cwd / f"{deffnm}.cpt").write_bytes(b"fake")
            (cwd / f"{deffnm}_pullx.xvg").write_text("0 0\n1 0\n", encoding="utf-8")
            (cwd / f"{deffnm}_pullf.xvg").write_text("0 0\n1 0\n", encoding="utf-8")
            if mdrun_extra_args:
                (cwd / "COLVAR").write_text(
                    "#! FIELDS time d_ion_target.z cn_solvent cn_target cn_anion\n0 0 4 0 0\n1 0 3 1 0\n",
                    encoding="utf-8",
                )

        def run(self, args, *, cwd=None, stdin_text=None):
            cwd = Path(cwd or ".")
            if args and args[0] == "trjconv":
                out = Path(args[args.index("-o") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("fake\n0\n   4.0   4.0   6.0\n", encoding="utf-8")
            elif args and args[0] == "wham":
                out = Path(args[args.index("-o") + 1])
                hist = Path(args[args.index("-hist") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text("0 0\n1 1\n", encoding="utf-8")
                hist.write_text("0 1 0.5 0\n1 0 0.5 1\n", encoding="utf-8")
            return None

    result = run_solvated_ion_umbrella(plan, runner=FakeRunner(), gpu=0, omp=1, mpi=1)
    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["pmf"]["available"] is True
    assert summary["coordination"]["available"] is True
