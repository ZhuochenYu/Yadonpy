from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit import Geometry as Geom

from yadonpy.interface.sandwich import (
    ElectrolyteSlabSpec,
    GraphiteSubstrateSpec,
    MoleculeSpec,
    PolymerSlabSpec,
    SandwichPhaseReport,
    SandwichRelaxationSpec,
    _covered_lateral_replicas,
    _compact_packed_cell_z_by_molecule_centers,
    _initial_bulk_pack_density,
    _augment_sandwich_ndx,
    _build_stack_checks,
    _phase_confined_relaxation_stages,
    _rebox_block_for_phase_confinement,
    _sandwich_relaxation_stages,
    build_graphite_polymer_electrolyte_sandwich,
)
from yadonpy.io.gromacs_system import SystemExportResult


def _dummy_mol(name: str, *, z_ang: float = 0.0, cell_box_ang: tuple[float, float, float] = (20.0, 20.0, 20.0)):
    mol = Chem.RWMol()
    atom = Chem.Atom("C")
    atom.SetNoImplicit(True)
    mol.AddAtom(atom)
    out = mol.GetMol()
    conf = Chem.Conformer(out.GetNumAtoms())
    conf.Set3D(True)
    conf.SetAtomPosition(0, Geom.Point3D(0.0, 0.0, float(z_ang)))
    out.AddConformer(conf, assignId=True)
    out.SetProp("_Name", str(name))
    out.SetProp("_yadonpy_name", str(name))
    setattr(out, "cell", SimpleNamespace(
        xhi=float(cell_box_ang[0]),
        xlo=0.0,
        yhi=float(cell_box_ang[1]),
        ylo=0.0,
        zhi=float(cell_box_ang[2]),
        zlo=0.0,
    ))
    return out


def test_augment_sandwich_ndx_adds_phase_groups(tmp_path: Path):
    ndx = tmp_path / "system.ndx"
    ndx.write_text(
        "\n".join(
            [
                "[ GRAPH ]",
                "1 2",
                "",
                "[ PEO ]",
                "3 4 5",
                "",
                "[ DME ]",
                "6 7",
                "",
                "[ Li ]",
                "8",
                "",
                "[ FSI ]",
                "9 10 11",
                "",
            ]
        ),
        encoding="utf-8",
    )
    groups = _augment_sandwich_ndx(
        ndx_path=ndx,
        graphite_name="GRAPH",
        polymer_name="PEO",
        electrolyte_names=["DME", "Li", "FSI"],
    )
    assert groups["System"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert groups["GRAPHITE"] == [1, 2]
    assert groups["POLYMER"] == [3, 4, 5]
    assert groups["ELECTROLYTE"] == [6, 7, 8, 9, 10, 11]
    assert groups["MOBILE"] == [3, 4, 5, 6, 7, 8, 9, 10, 11]


def test_sandwich_relaxation_stages_freeze_graphite_and_keep_xy_fixed():
    stages = _sandwich_relaxation_stages(
        relax=SandwichRelaxationSpec(stacked_pre_nvt_ps=10.0, stacked_z_relax_ps=20.0, stacked_exchange_ps=30.0),
        freeze_group="GRAPHITE",
    )
    assert [stage.name for stage in stages] == ["01_em", "02_pre_nvt", "03_z_relax", "04_exchange"]
    assert "freezegrps               = GRAPHITE" in stages[0].mdp.params["extra_mdp"]
    assert "freezegrps               = GRAPHITE" in stages[2].mdp.params["extra_mdp"]
    assert stages[2].mdp.params["pcoupltype"] == "semiisotropic"
    assert stages[2].mdp.params["compressibility"] == "0 4.5e-05"


def test_phase_confined_relaxation_stages_use_xy_walls_and_fixed_xy_npt():
    stages = _phase_confined_relaxation_stages(
        relax=SandwichRelaxationSpec(stacked_pre_nvt_ps=10.0, stacked_z_relax_ps=40.0),
        wall_atomtype="c3",
    )
    assert [stage.name for stage in stages] == ["01_em", "02_pre_nvt", "03_density_relax"]
    assert stages[1].mdp.params["pbc"] == "xy"
    assert "wall_type                = 12-6" in stages[1].mdp.params["wall_mdp"]
    assert "wall_atomtype            = c3 c3" in stages[1].mdp.params["wall_mdp"]
    assert stages[2].mdp.params["pcoupltype"] == "semiisotropic"
    assert stages[2].mdp.params["compressibility"] == "0 4.5e-05"


def test_build_stack_checks_reports_phase_order(tmp_path: Path):
    gro = tmp_path / "final.gro"
    gro.write_text(
        "\n".join(
            [
                "dummy",
                "6",
                "    1GRA     C    1   0.000   0.000   0.100",
                "    1GRA     C    2   0.000   0.000   0.120",
                "    2PEO     C    3   0.000   0.000   0.450",
                "    2PEO     C    4   0.000   0.000   0.520",
                "    3EL      C    5   0.000   0.000   0.900",
                "    3EL      C    6   0.000   0.000   0.980",
                "2.00000 2.00000 2.00000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checks = _build_stack_checks(
        gro_path=gro,
        ndx_groups={"GRAPHITE": [1, 2], "POLYMER": [3, 4], "ELECTROLYTE": [5, 6]},
    )
    assert checks["is_expected_order"] is True
    assert checks["observed_order"] == ["GRAPHITE", "POLYMER", "ELECTROLYTE"]
    assert checks["graphite_polymer_gap_nm"] > 0.0
    assert checks["polymer_electrolyte_gap_nm"] > 0.0


def test_build_stack_checks_unwraps_phase_crossing_periodic_boundary(tmp_path: Path):
    def _gro_line(resnr: int, resname: str, atomname: str, atomnr: int, z_nm: float) -> str:
        return f"{resnr:5d}{resname:<5}{atomname:>5}{atomnr:5d}{0.0:8.3f}{0.0:8.3f}{z_nm:8.3f}"

    gro = tmp_path / "wrapped.gro"
    gro.write_text(
        "\n".join(
            [
                "dummy",
                "6",
                _gro_line(1, "GRA", "C", 1, 0.200),
                _gro_line(1, "GRA", "C", 2, 0.535),
                _gro_line(2, "PEO", "C", 3, 4.100),
                _gro_line(2, "PEO", "C", 4, 7.346),
                _gro_line(3, "EL", "C", 5, 9.482),
                _gro_line(3, "EL", "C", 6, 0.381),
                "2.00000 2.00000 10.00000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checks = _build_stack_checks(
        gro_path=gro,
        ndx_groups={"GRAPHITE": [1, 2], "POLYMER": [3, 4], "ELECTROLYTE": [5, 6]},
    )
    assert checks["polymer_electrolyte_gap_nm"] > 0.0
    assert checks["phases"]["ELECTROLYTE"]["mean_z_nm"] > checks["phases"]["POLYMER"]["mean_z_nm"]


def test_compact_packed_cell_z_by_molecule_centers_preserves_order_and_target_box():
    species = [_dummy_mol("PEO_monomer")]
    cell = Chem.RWMol()
    for _ in range(2):
        atom = Chem.Atom("C")
        atom.SetNoImplicit(True)
        cell.AddAtom(atom)
    out = cell.GetMol()
    conf = Chem.Conformer(out.GetNumAtoms())
    conf.Set3D(True)
    conf.SetAtomPosition(0, Geom.Point3D(0.0, 0.0, 5.0))
    conf.SetAtomPosition(1, Geom.Point3D(0.0, 0.0, 105.0))
    out.AddConformer(conf, assignId=True)
    setattr(out, "cell", SimpleNamespace(xhi=20.0, xlo=0.0, yhi=20.0, ylo=0.0, zhi=120.0, zlo=0.0))
    sandwich, note = _compact_packed_cell_z_by_molecule_centers(
        cell=out,
        species=species,
        counts=[2],
        target_box_nm=(2.0, 2.0, 3.0),
    )
    z = [float(sandwich.GetConformer(0).GetAtomPosition(i).z) for i in range(sandwich.GetNumAtoms())]
    assert sandwich.cell.zhi == 45.0
    assert 0.0 <= min(z) < max(z) <= 45.0
    assert z[0] < z[1]
    assert note is not None
    assert "pre-relaxation" in note


def test_rebox_block_for_phase_confinement_centers_slab_and_adds_vacuum():
    block = _dummy_mol("PEO", z_ang=6.0, cell_box_ang=(18.0, 18.0, 18.0))
    conf = block.GetConformer(0)
    conf.SetAtomPosition(0, Geom.Point3D(1.0, 2.0, 6.0))
    confined, summary, note = _rebox_block_for_phase_confinement(
        block=block,
        target_xy_nm=(2.0, 2.0),
        target_thickness_nm=1.5,
        vacuum_padding_ang=12.0,
    )
    pos = confined.GetConformer(0).GetAtomPosition(0)
    assert confined.cell.xhi == pytest.approx(20.0)
    assert confined.cell.yhi == pytest.approx(20.0)
    assert confined.cell.zhi == pytest.approx(39.0)
    assert 0.0 < float(pos.x) < 20.0
    assert 0.0 < float(pos.y) < 20.0
    assert 12.0 < float(pos.z) < 27.0
    assert summary["target_xy_nm"] == [2.0, 2.0]
    assert "vacuum" in note


def test_covered_lateral_replicas_ceil_to_cover_target_lengths():
    reps = _covered_lateral_replicas(source_box_nm=(1.8, 2.3, 3.5), target_lengths_nm=(4.0, 4.5))
    assert reps == (3, 2)


def test_initial_bulk_pack_density_defaults_are_more_permissive_than_targets():
    assert _initial_bulk_pack_density(target_density_g_cm3=1.08, phase="polymer") == pytest.approx(0.648)
    assert _initial_bulk_pack_density(target_density_g_cm3=1.12, phase="electrolyte") == pytest.approx(0.896)
    assert _initial_bulk_pack_density(target_density_g_cm3=1.12, phase="electrolyte", requested_density_g_cm3=0.82) == pytest.approx(0.82)


def test_build_graphite_polymer_electrolyte_sandwich_orchestrates_bulk_then_slab_prep(tmp_path: Path, monkeypatch):
    import yadonpy.interface.sandwich as sandwich

    graphite_cell = _dummy_mol("GRAPH", z_ang=1.0, cell_box_ang=(20.0, 20.0, 12.0))
    graphite_layer = _dummy_mol("GRAPH", z_ang=0.0, cell_box_ang=(20.0, 20.0, 4.0))
    polymer_chain = _dummy_mol("PEO", z_ang=0.0, cell_box_ang=(20.0, 20.0, 8.0))
    dme = _dummy_mol("DME", z_ang=0.0, cell_box_ang=(20.0, 20.0, 8.0))
    li = _dummy_mol("Li", z_ang=0.0, cell_box_ang=(20.0, 20.0, 8.0))
    fsi = _dummy_mol("FSI", z_ang=0.0, cell_box_ang=(20.0, 20.0, 8.0))
    polymer_cell = _dummy_mol("PEO", z_ang=5.0, cell_box_ang=(20.0, 20.0, 18.0))
    electrolyte_cell = _dummy_mol("EL", z_ang=8.0, cell_box_ang=(20.0, 20.0, 20.0))

    monkeypatch.setattr(
        sandwich,
        "build_graphite",
        lambda **kwargs: SimpleNamespace(
            cell=graphite_cell,
            layer_mol=graphite_layer,
            layer_count=2,
            orientation="basal",
            edge_cap_summary={"H": 8},
            box_nm=(2.0, 2.0, 1.2),
        ),
    )
    monkeypatch.setattr(
        sandwich,
        "_prepare_polymer_phase_species",
        lambda **kwargs: {
            "chain": polymer_chain,
            "dp": 8,
            "chain_count": 2,
            "species": [polymer_chain],
            "counts": [2],
            "charge_scale": [1.0],
            "notes": (),
        },
    )
    pack_calls: list[dict] = []

    def _fake_amorphous_cell(mols, counts, **kwargs):
        pack_calls.append({"mols": mols, "counts": counts, **kwargs})
        return polymer_cell if len(mols) == 1 else electrolyte_cell

    monkeypatch.setattr(sandwich.poly, "amorphous_cell", _fake_amorphous_cell)
    eq_calls: list[dict] = []

    def _fake_eq(**kwargs):
        eq_calls.append(kwargs)
        ac = kwargs["ac"]
        return SimpleNamespace(final_cell=ac, system_export=None, raw_system_meta=Path("dummy.json"))

    monkeypatch.setattr(sandwich, "equilibrate_bulk_with_eq21", _fake_eq)
    monkeypatch.setattr(sandwich, "_prepare_small_molecule", lambda spec, **kwargs: {"DME": dme, "Li": li, "FSI": fsi}[spec.name])
    monkeypatch.setattr(
        sandwich,
        "plan_fixed_xy_direct_electrolyte_preparation",
        lambda **kwargs: SimpleNamespace(
            direct_plan=SimpleNamespace(target_counts=(4, 2, 2)),
            pack_plan=SimpleNamespace(initial_pack_box_nm=(2.0, 2.0, 4.5)),
            relax_mdp_overrides={"pcoupltype": "semiisotropic", "compressibility": "0 4.5e-05", "ref_p": "1 1"},
        ),
    )
    polymer_prepared = SimpleNamespace(
        top_path=tmp_path / "polymer.top",
        gro_path=tmp_path / "polymer.gro",
        meta_path=tmp_path / "polymer_meta.json",
        box_nm=(2.0, 2.0, 3.0),
    )
    electrolyte_prepared = SimpleNamespace(
        top_path=tmp_path / "electrolyte.top",
        gro_path=tmp_path / "electrolyte.gro",
        meta_path=tmp_path / "electrolyte_meta.json",
        box_nm=(2.0, 2.0, 4.0),
    )
    monkeypatch.setattr(
        sandwich,
        "_prepare_slab_from_equilibrated_bulk",
        lambda **kwargs: (
            polymer_prepared if kwargs["label"] == "polymer" else electrolyte_prepared,
            f"{kwargs['label']} slab prepared",
        ),
    )
    confined_calls: list[dict] = []
    monkeypatch.setattr(
        sandwich,
        "_run_confined_phase_relaxation",
        lambda **kwargs: (
            confined_calls.append(kwargs)
            or SimpleNamespace(
                relaxed_block=polymer_cell if kwargs["label"] == "polymer" else electrolyte_cell,
                report=SandwichPhaseReport(
                    label=kwargs["label"],
                    box_nm=(2.0, 2.0, 3.0 if kwargs["label"] == "polymer" else 4.0),
                    density_g_cm3=(1.02 if kwargs["label"] == "polymer" else 1.11),
                    species_names=("PEO",) if kwargs["label"] == "polymer" else ("DME", "Li", "FSI"),
                    counts=(2,) if kwargs["label"] == "polymer" else (4, 2, 2),
                    target_density_g_cm3=kwargs["target_density_g_cm3"],
                ),
                summary={"occupied_density_g_cm3": 1.02 if kwargs["label"] == "polymer" else 1.11},
                summary_path=tmp_path / f"{kwargs['label']}_summary.json",
            )
        ),
    )
    monkeypatch.setattr(
        sandwich,
        "_prepared_slab_phase_report",
        lambda **kwargs: SandwichPhaseReport(
            label=kwargs["label"],
            box_nm=(2.0, 2.0, 3.0 if kwargs["label"] == "polymer" else 4.0),
            density_g_cm3=(1.02 if kwargs["label"] == "polymer" else 1.11),
            species_names=("PEO",) if kwargs["label"] == "polymer" else ("DME", "Li", "FSI"),
            counts=(2,) if kwargs["label"] == "polymer" else (4, 2, 2),
            target_density_g_cm3=kwargs["target_density_g_cm3"],
        ),
    )
    monkeypatch.setattr(
        sandwich,
        "_load_block_from_top_gro",
        lambda **kwargs: polymer_cell if "polymer" in kwargs["gro_path"].name else electrolyte_cell,
    )

    def _fake_export(**kwargs):
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        system_top = out_dir / "system.top"
        system_gro = out_dir / "system.gro"
        system_ndx = out_dir / "system.ndx"
        system_meta = out_dir / "system_meta.json"
        system_top.write_text("; top\n", encoding="utf-8")
        system_gro.write_text("dummy\n1\n    1RES  C    1   0.000   0.000   0.000\n2.0 2.0 5.0\n", encoding="utf-8")
        system_ndx.write_text("[ GRAPH ]\n1\n\n[ PEO ]\n2\n\n[ DME ]\n3\n\n[ Li ]\n4\n\n[ FSI ]\n5\n", encoding="utf-8")
        system_meta.write_text("{}\n", encoding="utf-8")
        return SystemExportResult(
            system_gro=system_gro,
            system_top=system_top,
            system_ndx=system_ndx,
            molecules_dir=out_dir / "molecules",
            system_meta=system_meta,
            box_nm=5.0,
            species=[],
            box_lengths_nm=(2.0, 2.0, 5.0),
        )

    monkeypatch.setattr(sandwich, "export_system_from_cell_meta", _fake_export)
    monkeypatch.setattr(sandwich, "_run_stacked_relaxation", lambda **kwargs: Path(kwargs["work_dir"]) / "04_exchange" / "md.gro")
    monkeypatch.setattr(sandwich, "_build_stack_checks", lambda **kwargs: {"is_expected_order": True})

    result = build_graphite_polymer_electrolyte_sandwich(
        work_dir=tmp_path / "sandwich",
        ff=SimpleNamespace(name="gaff2_mod"),
        ion_ff=SimpleNamespace(),
        graphite=GraphiteSubstrateSpec(nx=4, ny=4, n_layers=2),
        polymer=PolymerSlabSpec(chain_target_atoms=240, slab_z_nm=3.0, target_density_g_cm3=1.05),
        electrolyte=ElectrolyteSlabSpec(
            solvents=(MoleculeSpec(name="DME", smiles="COCCOC"),),
            salt_cation=MoleculeSpec(name="Li", smiles="[Li+]", use_ion_ff=True, charge_scale=0.8),
            salt_anion=MoleculeSpec(name="FSI", smiles="FS(=O)(=O)[N-]S(=O)(=O)F", charge_scale=0.8),
            solvent_mass_ratio=(1.0,),
            target_density_g_cm3=1.15,
            slab_z_nm=4.0,
            initial_pack_density_g_cm3=0.82,
        ),
        relax=SandwichRelaxationSpec(gpu=0, omp=2, psi4_omp=2),
        restart=False,
    )

    assert len(eq_calls) == 2
    assert len(pack_calls) == 2
    assert len(confined_calls) == 2
    assert pack_calls[0]["density"] < 1.05
    assert pack_calls[1]["density"] == pytest.approx(0.82)
    assert eq_calls[0]["label"] == "Polymer bulk"
    assert eq_calls[1]["label"] == "Electrolyte bulk"
    assert eq_calls[0]["eq21_exec_kwargs"]["eq21_npt_time_scale"] == 0.4
    assert eq_calls[1]["eq21_exec_kwargs"]["eq21_npt_time_scale"] == 0.4
    assert result.polymer_phase.target_density_g_cm3 == 1.05
    assert result.electrolyte_phase.target_density_g_cm3 == 1.15
    assert result.manifest_path.exists()
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert '"polymer_phase"' in manifest
    assert '"electrolyte_phase"' in manifest
    assert '"polymer_phase_confined"' in manifest
    assert '"electrolyte_phase_confined"' in manifest
    assert '"ndx_groups"' in manifest
    assert (tmp_path / "sandwich" / "05_sandwich" / "sandwich_progress.json").exists()
    assert result.stack_checks["is_expected_order"] is True
