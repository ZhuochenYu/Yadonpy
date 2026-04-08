from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import Chem
from rdkit import Geometry as Geom

from yadonpy.core.graphite import stack_cell_blocks
from yadonpy.interface import (
    build_graphite_cmcna_example_case,
    build_graphite_peo_example_case,
    format_sandwich_result_summary,
)
from yadonpy.interface.sandwich import (
    ElectrolyteSlabSpec,
    GraphiteSubstrateSpec,
    MoleculeSpec,
    PolymerSlabSpec,
    SandwichPhaseReport,
    SandwichRelaxationSpec,
    _covered_lateral_replicas,
    _compact_packed_cell_z_by_molecule_centers,
    _ensure_system_group_in_ndx,
    _initial_bulk_pack_density,
    _augment_sandwich_ndx,
    _build_pack_density_ladder,
    _adaptive_stack_gaps_ang,
    _build_stack_checks,
    _compress_phase_block_z_to_target_thickness,
    _confined_summary_score,
    _confined_phase_report,
    _graphite_counts_for_required_xy,
    _graphite_repeat_factors_for_required_xy,
    _needs_confined_rescue,
    _maybe_expand_graphite_for_phase_footprint,
    _preflight_graphite_footprint_from_phase_targets,
    _preflight_linear_headroom_xy,
    _preflight_required_xy_nm_from_target_area,
    _phase_local_density_summary,
    _prepared_slab_required_xy_nm,
    _phase_confined_relaxation_stages,
    _rebox_block_for_phase_confinement,
    _run_amorphous_cell_with_density_backoff,
    _sandwich_relaxation_stages,
    _stack_master_xy_nm,
    build_graphite_cmcna_glucose6_periodic_case,
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


def test_ensure_system_group_in_ndx_promotes_uppercase_system(tmp_path: Path):
    ndx = tmp_path / "system.ndx"
    ndx.write_text(
        "\n".join(
            [
                "[ SYSTEM ]",
                "1 2 3",
                "",
                "[ POLYMER ]",
                "1 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    groups = _ensure_system_group_in_ndx(ndx)
    assert groups["System"] == [1, 2, 3]
    text = ndx.read_text(encoding="utf-8")
    assert "[ System ]" in text


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
    assert checks["graphite_polymer_core_gap_nm"] > 0.0
    assert checks["polymer_electrolyte_core_gap_nm"] > 0.0


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
    assert checks["polymer_electrolyte_core_gap_nm"] > 0.0
    assert checks["phases"]["ELECTROLYTE"]["mean_z_nm"] > checks["phases"]["POLYMER"]["mean_z_nm"]


def test_confined_phase_report_prefers_center_bulk_like_density():
    report = _confined_phase_report(
        label="polymer",
        species_names=("CMC6", "Na"),
        counts=(8, 40),
        target_density_g_cm3=1.50,
        summary={
            "box_nm": [2.3, 4.1, 6.6],
            "occupied_thickness_nm": 4.0,
            "occupied_density_g_cm3": 0.84,
            "center_bulk_like_density_g_cm3": 1.57,
        },
    )
    assert report.density_g_cm3 == pytest.approx(1.57)
    assert report.occupied_density_g_cm3 == pytest.approx(0.84)
    assert report.bulk_like_density_g_cm3 == pytest.approx(1.57)


def test_compress_phase_block_z_to_target_thickness_shrinks_overdilated_slab():
    cell = Chem.RWMol()
    for _ in range(2):
        atom = Chem.Atom("C")
        atom.SetNoImplicit(True)
        cell.AddAtom(atom)
    out = cell.GetMol()
    conf = Chem.Conformer(out.GetNumAtoms())
    conf.Set3D(True)
    conf.SetAtomPosition(0, Geom.Point3D(0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, Geom.Point3D(0.0, 0.0, 60.0))
    out.AddConformer(conf, assignId=True)

    compressed, summary = _compress_phase_block_z_to_target_thickness(
        block=out,
        target_thickness_nm=4.0,
    )

    coords = compressed.GetConformer(0).GetPositions()
    z_span = max(float(pos[2]) for pos in coords) - min(float(pos[2]) for pos in coords)
    assert summary["z_compression_applied"] is True
    assert summary["z_compression_scale"] < 1.0
    assert z_span == pytest.approx(40.0, rel=1e-3)


def test_phase_local_density_summary_uses_atomwise_mass_not_whole_molecule_com(tmp_path: Path):
    mol = Chem.RWMol()
    for _ in range(2):
        atom = Chem.Atom("C")
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    species = [mol.GetMol()]
    gro = tmp_path / "slab.gro"
    gro.write_text(
        "\n".join(
            [
                "dummy",
                "2",
                "    1POL     C    1   0.000   0.000   0.000",
                "    1POL     C    2   0.000   0.000   1.000",
                "2.00000 2.00000 2.00000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = _phase_local_density_summary(gro_path=gro, species=species, counts=[1])
    assert summary["occupied_density_g_cm3"] > 0.0
    assert summary["center_bulk_like_density_g_cm3"] == pytest.approx(0.0)


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


def test_rebox_block_for_phase_confinement_wraps_molecule_centers_into_target_xy():
    mol = Chem.RWMol()
    for _ in range(2):
        atom = Chem.Atom("C")
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    block = mol.GetMol()
    conf = Chem.Conformer(block.GetNumAtoms())
    conf.Set3D(True)
    conf.SetAtomPosition(0, Geom.Point3D(24.0, 21.0, 6.0))
    conf.SetAtomPosition(1, Geom.Point3D(25.2, 21.8, 8.0))
    block.AddConformer(conf, assignId=True)
    setattr(block, "cell", SimpleNamespace(xhi=40.0, xlo=0.0, yhi=40.0, ylo=0.0, zhi=18.0, zlo=0.0))

    species = [_dummy_mol("A"), _dummy_mol("B")]
    confined, summary, _note = _rebox_block_for_phase_confinement(
        block=block,
        target_xy_nm=(2.0, 2.0),
        target_thickness_nm=1.5,
        vacuum_padding_ang=12.0,
        species=species,
        counts=[1, 1],
    )

    coords = confined.GetConformer(0).GetPositions()
    assert confined.cell.xhi == pytest.approx(20.0)
    assert confined.cell.yhi == pytest.approx(20.0)
    assert max(float(x[0]) for x in coords) <= 20.0
    assert max(float(x[1]) for x in coords) <= 20.0
    assert summary["target_xy_nm"] == [2.0, 2.0]


def test_rebox_block_for_phase_confinement_restores_periodic_lateral_coordinates():
    mol = Chem.RWMol()
    for _ in range(2):
        atom = Chem.Atom("C")
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    block = mol.GetMol()
    conf = Chem.Conformer(block.GetNumAtoms())
    conf.Set3D(True)
    conf.SetAtomPosition(0, Geom.Point3D(0.2, 2.0, 4.0))
    conf.SetAtomPosition(1, Geom.Point3D(21.0, 18.5, 6.0))
    block.AddConformer(conf, assignId=True)
    setattr(block, "cell", SimpleNamespace(xhi=20.0, xlo=0.0, yhi=20.0, ylo=0.0, zhi=18.0, zlo=0.0))

    confined, summary, note = _rebox_block_for_phase_confinement(
        block=block,
        target_xy_nm=(2.0, 2.0),
        target_thickness_nm=1.5,
        vacuum_padding_ang=12.0,
    )

    coords = confined.GetConformer(0).GetPositions()
    xs = [float(x[0]) for x in coords]
    ys = [float(x[1]) for x in coords]
    assert max(xs) <= 20.0
    assert max(ys) <= 20.0
    assert summary["periodic_lateral_wrap_applied"] or summary["lateral_scale_xy"][0] < 1.0 or summary["lateral_scale_xy"][1] < 1.0
    assert ("restored lateral periodic coordinates" in note) or ("compressed the soft slab onto the graphite XY footprint" in note)


def test_rebox_block_for_phase_confinement_unwraps_bonded_fragments_across_lateral_boundary():
    mol = Chem.RWMol()
    for _ in range(2):
        atom = Chem.Atom("C")
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    mol.AddBond(0, 1, Chem.BondType.SINGLE)
    species = [mol.GetMol()]

    block = mol.GetMol()
    conf = Chem.Conformer(block.GetNumAtoms())
    conf.Set3D(True)
    conf.SetAtomPosition(0, Geom.Point3D(0.2, 2.0, 4.0))
    conf.SetAtomPosition(1, Geom.Point3D(19.8, 2.1, 4.1))
    block.AddConformer(conf, assignId=True)
    setattr(block, "cell", SimpleNamespace(xhi=20.0, xlo=0.0, yhi=20.0, ylo=0.0, zhi=18.0, zlo=0.0))

    confined, summary, note = _rebox_block_for_phase_confinement(
        block=block,
        target_xy_nm=(2.0, 2.0),
        target_thickness_nm=1.5,
        vacuum_padding_ang=12.0,
        species=species,
        counts=[1],
    )

    coords = confined.GetConformer(0).GetPositions()
    dx = abs(float(coords[1][0]) - float(coords[0][0]))
    assert dx < 1.0
    assert summary["bonded_lateral_unwrap_applied"] is True
    assert "restored bonded lateral periodic coordinates" in note


def test_rebox_block_for_phase_confinement_compresses_lateral_span_to_target_xy():
    block = _dummy_mol("SLAB", z_ang=4.0, cell_box_ang=(40.0, 40.0, 18.0))
    conf = block.GetConformer(0)
    conf.SetAtomPosition(0, Geom.Point3D(2.0, 2.0, 4.0))
    atom = Chem.Atom("C")
    atom.SetNoImplicit(True)
    rw = Chem.RWMol(block)
    rw.AddAtom(atom)
    block = rw.GetMol()
    conf = Chem.Conformer(block.GetNumAtoms())
    conf.Set3D(True)
    conf.SetAtomPosition(0, Geom.Point3D(2.0, 2.0, 4.0))
    conf.SetAtomPosition(1, Geom.Point3D(34.0, 30.0, 6.0))
    block.RemoveAllConformers()
    block.AddConformer(conf, assignId=True)
    setattr(block, "cell", SimpleNamespace(xhi=40.0, xlo=0.0, yhi=40.0, ylo=0.0, zhi=18.0, zlo=0.0))

    confined, summary, note = _rebox_block_for_phase_confinement(
        block=block,
        target_xy_nm=(3.05, 2.75),
        target_thickness_nm=1.5,
        vacuum_padding_ang=12.0,
    )

    coords = confined.GetConformer(0).GetPositions()
    xs = [float(x[0]) for x in coords]
    ys = [float(x[1]) for x in coords]
    assert max(xs) - min(xs) <= 30.5 + 1.0e-6
    assert max(ys) - min(ys) <= 27.5 + 1.0e-6
    assert summary["lateral_scale_xy"][0] < 1.0
    assert summary["lateral_scale_xy"][1] < 1.0
    assert "compressed the soft slab onto the graphite XY footprint" in note


def test_rebox_block_for_phase_confinement_refuses_extreme_lateral_compression():
    block = _dummy_mol("SLAB", z_ang=4.0, cell_box_ang=(40.0, 40.0, 18.0))
    conf = block.GetConformer(0)
    conf.SetAtomPosition(0, Geom.Point3D(2.0, 2.0, 4.0))
    atom = Chem.Atom("C")
    atom.SetNoImplicit(True)
    rw = Chem.RWMol(block)
    rw.AddAtom(atom)
    block = rw.GetMol()
    conf = Chem.Conformer(block.GetNumAtoms())
    conf.Set3D(True)
    conf.SetAtomPosition(0, Geom.Point3D(2.0, 2.0, 4.0))
    conf.SetAtomPosition(1, Geom.Point3D(34.0, 30.0, 6.0))
    block.RemoveAllConformers()
    block.AddConformer(conf, assignId=True)
    setattr(block, "cell", SimpleNamespace(xhi=40.0, xlo=0.0, yhi=40.0, ylo=0.0, zhi=18.0, zlo=0.0))

    with pytest.raises(RuntimeError, match="excessive lateral compression"):
        _rebox_block_for_phase_confinement(
            block=block,
            target_xy_nm=(2.0, 2.0),
            target_thickness_nm=1.5,
            vacuum_padding_ang=12.0,
        )


def test_graphite_repeat_factors_for_required_xy_expand_only_when_needed():
    assert _graphite_repeat_factors_for_required_xy(
        current_box_nm=(2.3, 4.1, 2.0),
        required_xy_nm=(2.1, 3.9),
    ) == (1, 1)
    assert _graphite_repeat_factors_for_required_xy(
        current_box_nm=(2.3, 4.1, 2.0),
        required_xy_nm=(4.0, 3.9),
    ) == (2, 1)
    assert _graphite_repeat_factors_for_required_xy(
        current_box_nm=(2.3, 4.1, 2.0),
        required_xy_nm=(4.0, 8.3),
    ) == (2, 3)


def test_graphite_counts_for_required_xy_expand_by_minimum_lattice_increment():
    graphite = GraphiteSubstrateSpec(nx=10, ny=10, n_layers=4)
    assert _graphite_counts_for_required_xy(
        graphite=graphite,
        current_box_nm=(4.7892, 4.183094328, 2.5),
        required_xy_nm=(4.80, 4.19),
    ) == (11, 11)
    assert _graphite_counts_for_required_xy(
        graphite=graphite,
        current_box_nm=(4.7892, 4.183094328, 2.5),
        required_xy_nm=(9.58, 8.37),
    ) == (21, 21)


def test_preflight_graphite_footprint_from_phase_targets_expands_before_bulk_rounds(monkeypatch, tmp_path: Path):
    import yadonpy.interface.sandwich as sandwich

    polymer_chain = _dummy_mol("CMC6")
    solvent = _dummy_mol("EC")
    cation = _dummy_mol("Li")
    anion = _dummy_mol("PF6")

    monkeypatch.setattr(
        sandwich,
        "_prepare_polymer_phase_species",
        lambda **kwargs: {
            "chain": polymer_chain,
            "dp": 20,
            "chain_count": 1800,
            "species": [polymer_chain],
            "counts": [1800],
            "charge_scale": [1.0],
            "notes": (),
        },
    )
    monkeypatch.setattr(
        sandwich,
        "_prepare_electrolyte_phase_inputs",
        lambda **kwargs: {
            "mols": [solvent, cation, anion],
            "charge_scale": [1.0, 1.0, 1.0],
            "prep": SimpleNamespace(direct_plan=SimpleNamespace(target_counts=(6000, 600, 600))),
        },
    )

    build_calls = []

    def _fake_build_graphite(**kwargs):
        build_calls.append((int(kwargs["nx"]), int(kwargs["ny"])))
        return SimpleNamespace(box_nm=(0.25 * int(kwargs["nx"]), 0.42 * int(kwargs["ny"]), 2.5))

    monkeypatch.setattr(sandwich, "build_graphite", _fake_build_graphite)

    graphite = GraphiteSubstrateSpec(nx=10, ny=10, n_layers=4)
    graphite_result = SimpleNamespace(box_nm=(2.5, 4.2, 2.5))
    expanded_graphite, expanded_result, negotiations = _preflight_graphite_footprint_from_phase_targets(
        graphite=graphite,
        graphite_result=graphite_result,
        ff=SimpleNamespace(),
        ion_ff=SimpleNamespace(),
        polymer=PolymerSlabSpec(target_density_g_cm3=1.50, slab_z_nm=4.2),
        electrolyte=ElectrolyteSlabSpec(target_density_g_cm3=1.32, slab_z_nm=4.6),
        relax=SandwichRelaxationSpec(),
        chain_dir=tmp_path / "chain",
    )

    assert negotiations
    assert expanded_graphite.nx > graphite.nx
    assert expanded_graphite.ny >= graphite.ny
    assert build_calls
    assert negotiations[0]["stage"] == "preflight"
    assert negotiations[0]["graphite_counts_before_xy"] == [10, 10]
    base_polymer_required_xy = _preflight_required_xy_nm_from_target_area(
        current_box_nm=graphite_result.box_nm,
        target_area_nm2=float(negotiations[0]["polymer_target_area_nm2"]) * float(negotiations[0]["area_margin"]),
    )
    assert negotiations[0]["polymer_preflight_required_xy_nm"][0] > base_polymer_required_xy[0]
    assert negotiations[0]["polymer_preflight_required_xy_nm"][1] > base_polymer_required_xy[1]
    assert negotiations[-1]["graphite_counts_after_xy"] == [expanded_graphite.nx, expanded_graphite.ny]
    assert expanded_result.box_nm[0] >= graphite_result.box_nm[0]


def test_preflight_linear_headroom_xy_is_polymer_specific():
    assert _preflight_linear_headroom_xy(label="polymer")[0] > 1.0
    assert _preflight_linear_headroom_xy(label="electrolyte") == (1.0, 1.0)


def test_preflight_required_xy_nm_from_target_area_applies_linear_headroom():
    base = _preflight_required_xy_nm_from_target_area(
        current_box_nm=(2.0, 3.0, 2.0),
        target_area_nm2=6.0,
    )
    inflated = _preflight_required_xy_nm_from_target_area(
        current_box_nm=(2.0, 3.0, 2.0),
        target_area_nm2=6.0,
        linear_headroom_xy=(1.2, 1.1),
    )
    assert inflated[0] == pytest.approx(base[0] * 1.2)
    assert inflated[1] == pytest.approx(base[1] * 1.1)


def test_maybe_expand_graphite_for_phase_footprint_skips_expansion_when_slab_is_compressible(monkeypatch):
    import yadonpy.interface.sandwich as sandwich

    spans = iter(((7.2452, 8.437094328), (6.93136, 13.86272)))
    monkeypatch.setattr(sandwich, "_prepared_slab_lateral_span_nm", lambda **kwargs: next(spans))
    build_calls = []
    monkeypatch.setattr(
        sandwich,
        "build_graphite",
        lambda **kwargs: build_calls.append(dict(kwargs)) or SimpleNamespace(box_nm=(99.0, 99.0, 9.9)),
    )

    graphite = GraphiteSubstrateSpec(nx=10, ny=10, n_layers=4)
    graphite_result = SimpleNamespace(box_nm=(7.2452, 8.437094328, 2.5044))

    _, _, negotiation = _maybe_expand_graphite_for_phase_footprint(
        graphite=graphite,
        graphite_result=graphite_result,
        ff=SimpleNamespace(),
        polymer_slab=object(),
        polymer_species=(),
        polymer_counts=(),
        electrolyte_slab=object(),
        electrolyte_species=(),
        electrolyte_counts=(),
    )

    assert negotiation is None
    assert build_calls == []


def test_maybe_expand_graphite_for_phase_footprint_expands_when_required_compression_is_too_large(monkeypatch):
    import yadonpy.interface.sandwich as sandwich

    spans = iter(((10.0, 10.0), (6.0, 6.0)))
    monkeypatch.setattr(sandwich, "_prepared_slab_lateral_span_nm", lambda **kwargs: next(spans))
    build_calls = []

    def _fake_build_graphite(**kwargs):
        build_calls.append(dict(kwargs))
        scale_x = float(kwargs["nx"]) / 4.0
        scale_y = float(kwargs["ny"]) / 4.0
        return SimpleNamespace(box_nm=(7.0 * scale_x, 8.0 * scale_y, 2.5))

    monkeypatch.setattr(sandwich, "build_graphite", _fake_build_graphite)

    graphite = GraphiteSubstrateSpec(nx=4, ny=4, n_layers=4, orientation="basal", edge_cap="periodic")
    graphite_result = SimpleNamespace(box_nm=(7.0, 8.0, 2.5))

    expanded_graphite, expanded_result, negotiation = _maybe_expand_graphite_for_phase_footprint(
        graphite=graphite,
        graphite_result=graphite_result,
        ff=SimpleNamespace(),
        polymer_slab=object(),
        polymer_species=(),
        polymer_counts=(),
        electrolyte_slab=object(),
        electrolyte_species=(),
        electrolyte_counts=(),
    )

    assert build_calls
    assert expanded_graphite.nx == 5
    assert expanded_graphite.ny == 5
    assert expanded_result.box_nm == pytest.approx((8.75, 10.0, 2.5))
    assert negotiation is not None
    assert negotiation["graphite_counts_before_xy"] == [4, 4]
    assert negotiation["graphite_counts_after_xy"] == [5, 5]
    assert negotiation["graphite_count_scale_xy"] == pytest.approx([1.25, 1.25])
    assert negotiation["polymer_required_xy_nm"] == pytest.approx([10.0, 10.0])
    assert negotiation["polymer_compression_aware_required_xy_nm"] == pytest.approx([8.2, 8.2])
    assert negotiation["electrolyte_compression_aware_required_xy_nm"] == pytest.approx([7.0, 8.0])


def test_build_polymer_chain_forwards_polyelectrolyte_mode_to_db_lookup(monkeypatch):
    import yadonpy.interface.sandwich as sandwich

    class _DummyFF:
        def __init__(self):
            self.mol_calls = []

        def mol(self, smiles, **kwargs):
            self.mol_calls.append((smiles, dict(kwargs)))
            return _dummy_mol(kwargs.get("name") or "monomer")

        def ff_assign(self, mol, **kwargs):
            return mol

    ff = _DummyFF()
    terminal = _dummy_mol("terminal")
    original_mol_from_smiles = sandwich.utils.mol_from_smiles
    monkeypatch.setattr(
        sandwich.utils,
        "mol_from_smiles",
        lambda smiles, name=None, **kwargs: terminal if smiles == "[H][*]" else original_mol_from_smiles(smiles, name=name, **kwargs),
    )
    monkeypatch.setattr(sandwich.qm, "assign_charges", lambda *args, **kwargs: True)
    monkeypatch.setattr(sandwich.poly, "polymerize_rw", lambda monomer, dp, **kwargs: _dummy_mol("polymer"))
    monkeypatch.setattr(sandwich.poly, "terminate_rw", lambda chain, terminal, **kwargs: _dummy_mol("terminated"))

    base = sandwich.default_cmcna_polymer_spec()
    polymer = sandwich.default_cmcna_polymer_spec(
        monomers=(base.monomers[3],),
        monomer_ratio=(1.0,),
        dp=4,
    )
    chain, dp = sandwich._build_polymer_chain(
        ff=ff,
        polymer=polymer,
        relax=SandwichRelaxationSpec(psi4_omp=1, psi4_memory_mb=1000),
        chain_dir=Path("/tmp/polyelectrolyte_forward"),
    )

    assert dp == 4
    assert chain is not None
    assert ff.mol_calls
    assert ff.mol_calls[0][1]["polyelectrolyte_mode"] is True


def test_default_cmcna_and_carbonate_specs_require_ready_db_records():
    import yadonpy.interface.sandwich as sandwich

    polymer = sandwich.default_cmcna_polymer_spec()
    assert polymer.monomers
    assert all(spec.prefer_db for spec in polymer.monomers)
    assert all(spec.require_ready for spec in polymer.monomers)

    electrolyte = sandwich.default_carbonate_lipf6_electrolyte_spec()
    assert electrolyte.solvents
    assert all(spec.prefer_db for spec in electrolyte.solvents)
    assert all(spec.require_ready for spec in electrolyte.solvents)
    assert electrolyte.salt_anion.prefer_db is True
    assert electrolyte.salt_anion.require_ready is True


def test_rebox_block_for_phase_confinement_softens_catastrophic_xy_overlaps():
    mol = Chem.RWMol()
    for _ in range(2):
        atom = Chem.Atom("C")
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    block = mol.GetMol()
    conf = Chem.Conformer(block.GetNumAtoms())
    conf.Set3D(True)
    conf.SetAtomPosition(0, Geom.Point3D(10.000, 10.000, 4.0))
    conf.SetAtomPosition(1, Geom.Point3D(10.005, 10.002, 4.0))
    block.AddConformer(conf, assignId=True)
    setattr(block, "cell", SimpleNamespace(xhi=20.0, xlo=0.0, yhi=20.0, ylo=0.0, zhi=18.0, zlo=0.0))

    confined, summary, note = _rebox_block_for_phase_confinement(
        block=block,
        target_xy_nm=(2.0, 2.0),
        target_thickness_nm=1.5,
        vacuum_padding_ang=12.0,
    )

    coords = confined.GetConformer(0).GetPositions()
    dx = float(coords[1][0] - coords[0][0])
    dy = float(coords[1][1] - coords[0][1])
    dist = (dx * dx + dy * dy) ** 0.5
    assert summary["overlap_softening_applied"] is True
    assert summary["overlap_pairs_softened"] >= 1
    assert dist > 0.02
    assert "softened" in note


def test_stack_cell_blocks_respects_fixed_xy_master_footprint():
    lower = _dummy_mol("LOWER", z_ang=0.0, cell_box_ang=(40.0, 40.0, 20.0))
    upper = _dummy_mol("UPPER", z_ang=0.0, cell_box_ang=(40.0, 40.0, 20.0))
    lower_conf = lower.GetConformer(0)
    upper_conf = upper.GetConformer(0)
    lower_conf.SetAtomPosition(0, Geom.Point3D(18.5, 5.0, 1.0))
    upper_conf.SetAtomPosition(0, Geom.Point3D(19.0, 6.0, 1.5))

    stacked = stack_cell_blocks(
        [lower, upper],
        z_gaps_ang=[6.0],
        top_padding_ang=8.0,
        fixed_xy_ang=(20.0, 20.0),
    )

    assert stacked.box_nm[0] == pytest.approx(2.0)
    assert stacked.box_nm[1] == pytest.approx(2.0)
    coords = stacked.cell.GetConformer(0).GetPositions()
    assert max(float(x[0]) for x in coords) <= 20.0 + 1.0e-6
    assert max(float(x[1]) for x in coords) <= 20.0 + 1.0e-6


def test_adaptive_stack_gaps_expand_when_confined_slabs_have_large_surface_shells():
    relax = SandwichRelaxationSpec(
        graphite_to_polymer_gap_ang=3.8,
        polymer_to_electrolyte_gap_ang=4.2,
    )
    polymer_summary = {
        "occupied_thickness_nm": 4.0,
        "center_bulk_like_window_nm": [1.0, 3.0],
        "occupied_density_g_cm3": 0.90,
        "wrapped_across_z_boundary": False,
    }
    electrolyte_summary = {
        "occupied_thickness_nm": 6.5,
        "center_bulk_like_window_nm": [1.5, 4.5],
        "occupied_density_g_cm3": 0.95,
        "wrapped_across_z_boundary": True,
    }

    graphite_polymer_gap_ang, polymer_electrolyte_gap_ang = _adaptive_stack_gaps_ang(
        relax=relax,
        polymer_summary=polymer_summary,
        polymer_target_density_g_cm3=1.45,
        electrolyte_summary=electrolyte_summary,
        electrolyte_target_density_g_cm3=1.32,
    )

    assert graphite_polymer_gap_ang > 3.8
    assert polymer_electrolyte_gap_ang > 4.2
    assert polymer_electrolyte_gap_ang > graphite_polymer_gap_ang


def test_stack_master_xy_nm_adds_periodic_graphite_seam_clearance():
    periodic = _stack_master_xy_nm(
        graphite=GraphiteSubstrateSpec(edge_cap="periodic"),
        graphite_box_nm=(2.3332, 4.183094328, 2.5044),
    )
    capped = _stack_master_xy_nm(
        graphite=GraphiteSubstrateSpec(edge_cap="H"),
        graphite_box_nm=(2.3332, 4.183094328, 2.5044),
    )
    assert periodic[0] == pytest.approx(2.5132)
    assert periodic[1] == pytest.approx(4.363094328)
    assert capped == pytest.approx((2.3332, 4.183094328))


def test_needs_confined_rescue_flags_wrapped_and_overdilated_phase():
    summary = {
        "occupied_thickness_nm": 6.4,
        "center_bulk_like_density_g_cm3": 1.08,
        "wrapped_across_z_boundary": True,
    }
    assert _needs_confined_rescue(
        summary=summary,
        target_density_g_cm3=1.32,
        target_thickness_nm=4.6,
    ) is True


def test_confined_summary_score_prefers_center_density_and_target_thickness_match():
    better = {
        "occupied_density_g_cm3": 1.02,
        "center_bulk_like_density_g_cm3": 1.28,
        "occupied_thickness_nm": 4.8,
        "wrapped_across_z_boundary": False,
    }
    worse = {
        "occupied_density_g_cm3": 0.93,
        "center_bulk_like_density_g_cm3": 1.08,
        "occupied_thickness_nm": 6.4,
        "wrapped_across_z_boundary": True,
    }
    better_score = _confined_summary_score(
        summary=better,
        target_density_g_cm3=1.32,
        target_thickness_nm=4.6,
    )
    worse_score = _confined_summary_score(
        summary=worse,
        target_density_g_cm3=1.32,
        target_thickness_nm=4.6,
    )
    assert better_score < worse_score


def test_covered_lateral_replicas_prefers_minimal_replicas_that_fit_within_strain():
    reps = _covered_lateral_replicas(
        source_box_nm=(1.8, 2.3, 3.5),
        target_lengths_nm=(4.0, 4.5),
        max_lateral_strain=0.12,
    )
    assert reps == (2, 2)


def test_covered_lateral_replicas_falls_back_to_covering_target_when_no_near_match_exists():
    reps = _covered_lateral_replicas(
        source_box_nm=(1.0, 1.0, 3.5),
        target_lengths_nm=(2.7, 1.4),
        max_lateral_strain=0.05,
    )
    assert reps == (3, 2)


def test_initial_bulk_pack_density_defaults_are_more_permissive_than_targets():
    assert _initial_bulk_pack_density(target_density_g_cm3=1.08, phase="polymer") == pytest.approx(0.5616)
    assert _initial_bulk_pack_density(target_density_g_cm3=1.50, phase="polymer", z_scale=1.28) == pytest.approx(0.68 / 1.28)
    assert _initial_bulk_pack_density(target_density_g_cm3=1.50, phase="polymer", z_scale=1.30, charged=True) == pytest.approx(0.56 / 1.30)
    assert _initial_bulk_pack_density(target_density_g_cm3=1.12, phase="electrolyte") == pytest.approx(0.896)
    assert _initial_bulk_pack_density(target_density_g_cm3=1.12, phase="electrolyte", requested_density_g_cm3=0.82) == pytest.approx(0.82)


def test_build_pack_density_ladder_uses_phase_specific_backoff_policy():
    polymer_policy, polymer_ladder = _build_pack_density_ladder(
        phase="polymer",
        target_density_g_cm3=1.50,
        z_scale=1.25,
    )
    electrolyte_policy, electrolyte_ladder = _build_pack_density_ladder(
        phase="electrolyte",
        target_density_g_cm3=1.20,
        requested_density_g_cm3=0.86,
    )
    assert polymer_policy.max_attempts == 4
    assert polymer_policy.backoff_factor == pytest.approx(0.88)
    assert polymer_policy.floor_density_g_cm3 == pytest.approx(0.40)
    assert polymer_ladder[0] == pytest.approx(0.68 / 1.25)
    assert polymer_ladder[-1] >= 0.40
    assert electrolyte_policy.max_attempts == 3
    assert electrolyte_ladder[0] == pytest.approx(0.86)
    assert electrolyte_ladder[1] == pytest.approx(0.86 * 0.90)


def test_build_pack_density_ladder_uses_more_permissive_policy_for_charged_polymer():
    polymer_policy, polymer_ladder = _build_pack_density_ladder(
        phase="polymer",
        target_density_g_cm3=1.50,
        z_scale=1.30,
        charged=True,
    )
    assert polymer_policy.charged is True
    assert polymer_policy.max_attempts == 5
    assert polymer_policy.backoff_factor == pytest.approx(0.86)
    assert polymer_policy.floor_density_g_cm3 == pytest.approx(0.30)
    assert polymer_ladder[0] == pytest.approx(0.56 / 1.30)
    assert polymer_ladder[-1] >= 0.30


def test_run_amorphous_cell_with_density_backoff_retries_and_writes_summary(tmp_path: Path):
    calls: list[float] = []

    def _fake_pack(*_args, **kwargs):
        calls.append(float(kwargs["density"]))
        if len(calls) == 1:
            raise RuntimeError("too dense")
        return {"packed": True}

    result = _run_amorphous_cell_with_density_backoff(
        label="polymer",
        pack_fn=_fake_pack,
        mols=[_dummy_mol("PEO")],
        counts=[2],
        charge_scale=[1.0],
        phase="polymer",
        target_density_g_cm3=1.40,
        z_scale=1.20,
        work_dir=tmp_path / "poly_pack",
        retry=10,
        retry_step=100,
        threshold=1.5,
        dec_rate=0.7,
    )

    assert result.selected_attempt_index == 1
    assert result.selected_density_g_cm3 == pytest.approx(calls[1])
    assert calls[1] == pytest.approx(calls[0] * 0.88)
    assert result.summary_path.exists()
    assert result.summary["attempts"][0]["success"] is False
    assert result.summary["attempts"][1]["success"] is True


def test_build_graphite_cmcna_glucose6_periodic_case_uses_moldb_ready_defaults(tmp_path: Path, monkeypatch):
    import yadonpy.interface.sandwich_examples as sandwich_examples

    captured: dict[str, object] = {}

    def _fake_builder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            manifest_path=tmp_path / "manifest.json",
            relaxed_gro=tmp_path / "final.gro",
            polymer_phase=SimpleNamespace(density_g_cm3=1.5),
            electrolyte_phase=SimpleNamespace(density_g_cm3=1.3),
            stack_checks={},
        )

    monkeypatch.setattr(sandwich_examples, "build_graphite_cmcna_electrolyte_sandwich", _fake_builder)

    result = build_graphite_cmcna_glucose6_periodic_case(
        work_dir=tmp_path,
        ff=object(),
        ion_ff=object(),
        profile="smoke",
        restart=True,
    )

    assert result.manifest_path == tmp_path / "manifest.json"
    graphite = captured["graphite"]
    polymer = captured["polymer"]
    electrolyte = captured["electrolyte"]
    assert graphite.edge_cap == "periodic"
    assert polymer.monomers[0].name == "glucose_6"
    assert polymer.monomers[0].prefer_db is True
    assert polymer.monomers[0].require_ready is True
    assert polymer.monomers[0].polyelectrolyte_mode is True
    assert polymer.chain_count is None
    assert polymer.initial_pack_z_scale == pytest.approx(1.55)
    assert electrolyte.solvents[0].name == "EC"
    assert all(spec.prefer_db and spec.require_ready for spec in electrolyte.solvents)
    assert electrolyte.salt_anion.name == "PF6"
    assert electrolyte.salt_anion.prefer_db is True
    assert electrolyte.salt_anion.require_ready is True


def test_build_graphite_peo_example_case_chooses_smoke_defaults(tmp_path: Path, monkeypatch):
    import yadonpy.interface.sandwich_examples as sandwich_examples

    captured: dict[str, object] = {}

    def _fake_builder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            manifest_path=tmp_path / "manifest.json",
            relaxed_gro=tmp_path / "final.gro",
            polymer_phase=SimpleNamespace(density_g_cm3=1.05),
            electrolyte_phase=SimpleNamespace(density_g_cm3=1.28),
            stack_checks={"is_expected_order": True},
        )

    monkeypatch.setattr(sandwich_examples, "build_graphite_peo_electrolyte_sandwich", _fake_builder)

    result = build_graphite_peo_example_case(
        work_dir=tmp_path,
        ff=object(),
        ion_ff=object(),
        profile="smoke",
        restart=True,
    )

    assert result.manifest_path == tmp_path / "manifest.json"
    assert captured["graphite"].n_layers == 2
    assert captured["polymer"].chain_target_atoms == 220
    assert captured["electrolyte"].slab_z_nm == pytest.approx(3.8)
    assert captured["relax"].stacked_exchange_ps == pytest.approx(60.0)


def test_build_graphite_cmcna_example_case_chooses_full_defaults(tmp_path: Path, monkeypatch):
    import yadonpy.interface.sandwich_examples as sandwich_examples

    captured: dict[str, object] = {}

    def _fake_builder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            manifest_path=tmp_path / "manifest.json",
            relaxed_gro=tmp_path / "final.gro",
            polymer_phase=SimpleNamespace(density_g_cm3=1.42),
            electrolyte_phase=SimpleNamespace(density_g_cm3=1.24),
            stack_checks={"is_expected_order": True},
        )

    monkeypatch.setattr(sandwich_examples, "build_graphite_cmcna_electrolyte_sandwich", _fake_builder)

    result = build_graphite_cmcna_example_case(
        work_dir=tmp_path,
        ff=object(),
        ion_ff=object(),
        profile="full",
        restart=True,
    )

    assert result.relaxed_gro == tmp_path / "final.gro"
    assert captured["graphite"].n_layers == 3
    assert captured["polymer"].dp == 60
    assert captured["electrolyte"].slab_z_nm == pytest.approx(5.2)
    assert captured["relax"].stacked_exchange_ps == pytest.approx(120.0)


def test_format_sandwich_result_summary_emits_linear_example02_style_lines(tmp_path: Path):
    result = SimpleNamespace(
        manifest_path=tmp_path / "manifest.json",
        relaxed_gro=tmp_path / "final.gro",
        polymer_phase=SimpleNamespace(density_g_cm3=1.45678),
        electrolyte_phase=SimpleNamespace(density_g_cm3=1.23456),
        stack_checks={"is_expected_order": True},
        acceptance={
            "accepted": True,
            "failed_checks": [],
            "order_ok": True,
            "wrapped_ok": True,
            "polymer_density_ok": True,
            "electrolyte_density_ok": True,
            "core_gaps_ok": True,
            "graphite_polymer_core_gap_nm": 0.54321,
            "polymer_electrolyte_core_gap_nm": 0.98765,
        },
    )

    lines = format_sandwich_result_summary(result, profile="smoke")

    assert lines[0] == "profile = smoke"
    assert "manifest_path =" in lines[1]
    assert "relaxed_gro =" in lines[2]
    assert lines[3] == "polymer_density_g_cm3 = 1.4568"
    assert lines[4] == "electrolyte_density_g_cm3 = 1.2346"
    assert lines[5] == "accepted = True"
    assert lines[6] == "failed_checks = []"
    assert lines[7] == "order_ok = True"
    assert lines[8] == "wrapped_ok = True"
    assert lines[9] == "polymer_density_ok = True"
    assert lines[10] == "electrolyte_density_ok = True"
    assert lines[11] == "core_gaps_ok = True"
    assert lines[12] == "graphite_polymer_core_gap_nm = 0.5432"
    assert lines[13] == "polymer_electrolyte_core_gap_nm = 0.9877"


def test_format_sandwich_result_summary_falls_back_to_stack_checks_when_acceptance_missing(tmp_path: Path):
    result = SimpleNamespace(
        manifest_path=tmp_path / "manifest.json",
        relaxed_gro=tmp_path / "final.gro",
        polymer_phase=SimpleNamespace(density_g_cm3=1.4),
        electrolyte_phase=SimpleNamespace(density_g_cm3=1.2),
        stack_checks={"is_expected_order": True},
    )

    lines = format_sandwich_result_summary(result, profile="smoke")

    assert lines[5] == "stack_checks = {'is_expected_order': True}"


def test_build_sandwich_acceptance_reports_failed_checks():
    from yadonpy.interface.sandwich_metrics import build_sandwich_acceptance

    acceptance = build_sandwich_acceptance(
        polymer_summary={
            "center_bulk_like_density_g_cm3": 1.10,
            "wrapped_across_z_boundary": True,
        },
        electrolyte_summary={
            "center_bulk_like_density_g_cm3": 0.95,
            "wrapped_across_z_boundary": False,
        },
        stack_checks={
            "observed_order": ["POLYMER", "GRAPHITE", "ELECTROLYTE"],
            "graphite_polymer_core_gap_nm": -0.1,
            "polymer_electrolyte_core_gap_nm": 0.2,
        },
    )

    assert acceptance["accepted"] is False
    assert acceptance["failed_checks"] == [
        "polymer_density_ok",
        "electrolyte_density_ok",
        "core_gaps_ok",
        "wrapped_ok",
        "order_ok",
    ]


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
    monkeypatch.setattr(
        sandwich,
        "_preflight_graphite_footprint_from_phase_targets",
        lambda **kwargs: (kwargs["graphite"], kwargs["graphite_result"], []),
    )

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
    assert '"polymer_bulk_pack"' in manifest
    assert '"electrolyte_bulk_pack"' in manifest
    assert '"ndx_groups"' in manifest
    assert (tmp_path / "sandwich" / "05_sandwich" / "sandwich_progress.json").exists()
    assert result.stack_checks["is_expected_order"] is True


def test_build_graphite_polymer_electrolyte_sandwich_rebuilds_soft_phases_after_graphite_expansion(tmp_path: Path, monkeypatch):
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
    polymer_box_calls: list[tuple[float, float, float]] = []

    def _fake_prepare_polymer_phase_species(**kwargs):
        polymer_box_calls.append(tuple(float(x) for x in kwargs["box_nm"]))
        return {
            "chain": polymer_chain,
            "dp": 8,
            "chain_count": 2,
            "species": [polymer_chain],
            "counts": [2],
            "charge_scale": [1.0],
            "notes": (),
        }

    monkeypatch.setattr(sandwich, "_prepare_polymer_phase_species", _fake_prepare_polymer_phase_species)

    pack_calls: list[dict] = []

    def _fake_amorphous_cell(mols, counts, **kwargs):
        pack_calls.append({"mols": mols, "counts": counts, **kwargs})
        return polymer_cell if len(mols) == 1 else electrolyte_cell

    monkeypatch.setattr(sandwich.poly, "amorphous_cell", _fake_amorphous_cell)
    monkeypatch.setattr(
        sandwich,
        "equilibrate_bulk_with_eq21",
        lambda **kwargs: SimpleNamespace(final_cell=kwargs["ac"], system_export=None, raw_system_meta=Path("dummy.json")),
    )
    monkeypatch.setattr(sandwich, "_prepare_small_molecule", lambda spec, **kwargs: {"DME": dme, "Li": li, "FSI": fsi}[spec.name])
    electrolyte_ref_boxes: list[tuple[float, float, float]] = []

    def _fake_electrolyte_plan(**kwargs):
        electrolyte_ref_boxes.append(tuple(float(x) for x in kwargs["reference_box_nm"]))
        return SimpleNamespace(
            direct_plan=SimpleNamespace(target_counts=(4, 2, 2)),
            pack_plan=SimpleNamespace(initial_pack_box_nm=(2.0, 2.0, 4.5)),
            relax_mdp_overrides={"pcoupltype": "semiisotropic", "compressibility": "0 4.5e-05", "ref_p": "1 1"},
        )

    monkeypatch.setattr(sandwich, "plan_fixed_xy_direct_electrolyte_preparation", _fake_electrolyte_plan)
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
        "_run_confined_phase_relaxation",
        lambda **kwargs: SimpleNamespace(
            relaxed_block=polymer_cell if kwargs["label"] == "polymer" else electrolyte_cell,
            report=SandwichPhaseReport(
                label=kwargs["label"],
                box_nm=(4.0, 4.0, 3.0 if kwargs["label"] == "polymer" else 4.0),
                density_g_cm3=(1.02 if kwargs["label"] == "polymer" else 1.11),
                species_names=("PEO",) if kwargs["label"] == "polymer" else ("DME", "Li", "FSI"),
                counts=(2,) if kwargs["label"] == "polymer" else (4, 2, 2),
                target_density_g_cm3=kwargs["target_density_g_cm3"],
            ),
            summary={"occupied_density_g_cm3": 1.02 if kwargs["label"] == "polymer" else 1.11},
            summary_path=tmp_path / f"{kwargs['label']}_summary.json",
        ),
    )

    expansion_state = {"calls": 0}

    def _fake_expand_graphite(**kwargs):
        expansion_state["calls"] += 1
        if expansion_state["calls"] == 1:
            return (
                GraphiteSubstrateSpec(nx=8, ny=8, n_layers=2),
                SimpleNamespace(
                    cell=graphite_cell,
                    layer_mol=graphite_layer,
                    layer_count=2,
                    orientation="basal",
                    edge_cap_summary={"H": 8},
                    box_nm=(4.0, 4.0, 1.2),
                ),
                {
                    "repeat_factors_xy": [2, 2],
                    "graphite_box_before_nm": [2.0, 2.0, 1.2],
                    "graphite_box_after_nm": [4.0, 4.0, 1.2],
                },
            )
        return kwargs["graphite"], kwargs["graphite_result"], None

    monkeypatch.setattr(sandwich, "_maybe_expand_graphite_for_phase_footprint", _fake_expand_graphite)

    def _fake_export(**kwargs):
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        system_top = out_dir / "system.top"
        system_gro = out_dir / "system.gro"
        system_ndx = out_dir / "system.ndx"
        system_meta = out_dir / "system_meta.json"
        system_top.write_text("; top\n", encoding="utf-8")
        system_gro.write_text("dummy\n1\n    1RES  C    1   0.000   0.000   0.000\n4.0 4.0 5.0\n", encoding="utf-8")
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
            box_lengths_nm=(4.0, 4.0, 5.0),
        )

    monkeypatch.setattr(sandwich, "export_system_from_cell_meta", _fake_export)
    monkeypatch.setattr(sandwich, "_run_stacked_relaxation", lambda **kwargs: Path(kwargs["work_dir"]) / "04_exchange" / "md.gro")
    monkeypatch.setattr(sandwich, "_build_stack_checks", lambda **kwargs: {"is_expected_order": True})
    monkeypatch.setattr(
        sandwich,
        "_preflight_graphite_footprint_from_phase_targets",
        lambda **kwargs: (kwargs["graphite"], kwargs["graphite_result"], []),
    )

    result = build_graphite_polymer_electrolyte_sandwich(
        work_dir=tmp_path / "sandwich_expand",
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

    assert expansion_state["calls"] == 2
    assert len(polymer_box_calls) == 2
    assert polymer_box_calls[0][:2] == pytest.approx((2.0, 2.0))
    assert polymer_box_calls[1][:2] == pytest.approx((4.0, 4.0))
    assert len(electrolyte_ref_boxes) == 2
    assert electrolyte_ref_boxes[1][:2] == pytest.approx((4.0, 4.0))
    manifest = result.manifest_path.read_text(encoding="utf-8")
    progress = (tmp_path / "sandwich_expand" / "05_sandwich" / "sandwich_progress.json").read_text(encoding="utf-8")
    assert '"phase_preparation_rounds": 2' in manifest
    assert '"graphite_footprint_negotiations"' in progress
    assert '"latest_graphite_footprint_negotiation"' in progress
