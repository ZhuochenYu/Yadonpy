from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from yadonpy.core import utils
from yadonpy.core.graphite import build_graphite
from yadonpy.interface import layer_stack as layer_stack_mod
from yadonpy.interface.layer_stack import (
    ElectrodeChargeSpec,
    FixedChargeRegionSpec,
    GraphiteLayerSpec,
    LayerStackNvtResult,
    LayerStackRelaxationResult,
    LayerStackSpec,
    MolecularLayerSpec,
    VacuumLayerSpec,
    ZCompressionAnnealSpec,
    build_layer_stack,
    run_layer_stack_relaxation,
    run_layer_stack_nvt,
)
from yadonpy.io.gromacs_system import SystemExportResult


def _write_fake_gro_from_cell(cell, path: Path) -> None:
    coords = np.asarray(cell.GetConformer(0).GetPositions(), dtype=float) * 0.1
    nat = int(coords.shape[0])
    cell_box = getattr(cell, "cell", None)
    if cell_box is not None:
        box = (
            0.1 * (float(cell_box.xhi) - float(cell_box.xlo)),
            0.1 * (float(cell_box.yhi) - float(cell_box.ylo)),
            0.1 * (float(cell_box.zhi) - float(cell_box.zlo)),
        )
    else:
        box = (5.0, 5.0, 5.0)
    lines = ["fake layer stack", f"{nat:5d}"]
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(f"{1:5d}{'MOL':<5}{'C':>5}{i:5d}{x:8.3f}{y:8.3f}{z:8.3f}")
    lines.append(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _patch_fake_export(monkeypatch):
    from yadonpy.interface import layer_stack as mod

    def _fake_export_system_from_cell_meta(*, cell_mol, out_dir, ff_name, charge_method, polyelectrolyte_mode=False, **kwargs):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        gro = out_dir / "system.gro"
        top = out_dir / "system.top"
        ndx = out_dir / "system.ndx"
        meta = out_dir / "system_meta.json"
        _write_fake_gro_from_cell(cell_mol, gro)
        top.write_text("[ system ]\nfake\n", encoding="utf-8")
        ndx.write_text("[ System ]\n1\n", encoding="utf-8")
        meta.write_text(json.dumps({"ff_name": ff_name, "charge_method": charge_method}), encoding="utf-8")
        return SystemExportResult(
            system_gro=gro,
            system_top=top,
            system_ndx=ndx,
            molecules_dir=out_dir / "molecules",
            system_meta=meta,
            box_nm=1.0,
            species=[],
            box_lengths_nm=None,
        )

    monkeypatch.setattr(mod, "export_system_from_cell_meta", _fake_export_system_from_cell_meta)


def test_graphite_edge_carbonyl_o_cap_builds():
    result = build_graphite(nx=2, ny=2, n_layers=1, edge_cap="O", ff_name="gaff2_mod")
    assert result.edge_cap_summary["O"] > 0
    assert result.cell.GetNumAtoms() > 0


def test_layer_stack_supports_vacuum_and_arbitrary_order(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    water = utils.mol_from_smiles("O", name="WAT")
    stack = LayerStackSpec(
        layers=(
            VacuumLayerSpec(thickness_nm=0.5, name="VACUUM_TOP"),
            MolecularLayerSpec(
                name="ELECTROLYTE",
                species=(water,),
                counts=(2,),
                thickness_nm=1.0,
                density_target_g_cm3=0.4,
                layer_kind="electrolyte",
            ),
            GraphiteLayerSpec(name="GRAPHITE", nx=2, ny=2, n_layers=1),
        ),
        order="top_to_bottom",
        name="vacuum_electrolyte_graphite",
    )
    result = build_layer_stack(stack=stack, work_dir=tmp_path, restart=False, charge_method="gasteiger")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["layers"][0]["name"] == "GRAPHITE"
    assert manifest["layers"][0]["periodic_xy"] is True
    assert manifest["layers"][0]["effective_edge_cap"] == "periodic"
    assert manifest["layers"][-1]["name"] == "VACUUM_TOP"
    assert "LAYER_00_GRAPHITE" in result.system_ndx.read_text(encoding="utf-8")
    assert "ELECTROLYTE" in result.system_ndx.read_text(encoding="utf-8")


def test_layer_stack_density_target_expands_z_not_graphite_xy_by_default():
    water = utils.mol_from_smiles("O", name="WAT")
    layer = MolecularLayerSpec(
        name="DENSE_WATER",
        species=(water,),
        counts=(200,),
        thickness_nm=0.2,
        density_target_g_cm3=1.0,
        layer_kind="electrolyte",
    )
    graphite = GraphiteLayerSpec(name="GRAPHITE", nx=2, ny=2, n_layers=1)

    master_x, master_y, planning = layer_stack_mod._plan_master_xy(
        (graphite, layer),
        auto_expand_graphite=True,
        molecular_packing_expand="z",
    )

    assert planning["reason"] == "fixed_graphite_xy_z_expanded_molecular_layers"
    assert planning["graphite_dimensions"]["GRAPHITE"] == {"nx": 2, "ny": 2}
    assert layer_stack_mod._required_thickness_nm(layer, master_x * master_y) > layer.thickness_nm


def test_layer_stack_constant_charge_manifest(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(
                GraphiteLayerSpec(
                    name="GRAPHITE",
                    nx=2,
                    ny=2,
                    n_layers=2,
                    electrode_charge=ElectrodeChargeSpec(mode="total_charge", top_charge_e=1.0, bottom_charge_e=-1.0),
                ),
            ),
            name="charged_graphite",
        ),
        work_dir=tmp_path,
        restart=False,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    charge = manifest["layers"][0]["electrode_charge"]
    assert charge["applied"] is True
    assert charge["top_charge_e"] == 1.0
    assert charge["bottom_charge_e"] == -1.0


def test_layer_stack_side_specific_surface_charge_manifest(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(
                GraphiteLayerSpec(
                    name="BOTTOM",
                    nx=2,
                    ny=2,
                    n_layers=2,
                    electrode_charge=ElectrodeChargeSpec(
                        mode="surface_charge_density",
                        top_surface_charge_uC_cm2=2.0,
                    ),
                ),
                GraphiteLayerSpec(
                    name="TOP",
                    nx=2,
                    ny=2,
                    n_layers=2,
                    electrode_charge=ElectrodeChargeSpec(
                        mode="surface_charge_density",
                        bottom_surface_charge_uC_cm2=-2.0,
                    ),
                ),
            ),
            name="side_specific_charged_graphite",
        ),
        work_dir=tmp_path,
        restart=False,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    bottom_charge = manifest["layers"][0]["electrode_charge"]
    top_charge = manifest["layers"][1]["electrode_charge"]
    assert bottom_charge["top_charge_e"] > 0.0
    assert bottom_charge["bottom_charge_e"] == 0.0
    assert top_charge["bottom_charge_e"] < 0.0
    assert top_charge["top_charge_e"] == 0.0


def test_layer_stack_fixed_charge_region_targets_named_layer_face(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(
                GraphiteLayerSpec(name="BOTTOM", nx=2, ny=2, n_layers=2),
                GraphiteLayerSpec(name="TOP", nx=2, ny=2, n_layers=2),
            ),
            name="region_charged_graphite",
            fixed_charge_regions=(
                FixedChargeRegionSpec(
                    layer_name="BOTTOM",
                    region="top",
                    mode="surface_charge_density",
                    surface_charge_uC_cm2=2.0,
                    elements=("C",),
                    label="bottom_inner_face",
                ),
                FixedChargeRegionSpec(
                    layer_name="TOP",
                    region="bottom",
                    mode="surface_charge_density",
                    surface_charge_uC_cm2=-2.0,
                    elements=("C",),
                    label="top_inner_face",
                ),
            ),
        ),
        work_dir=tmp_path,
        restart=False,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    reports = manifest["fixed_charge_regions"]
    assert [r["label"] for r in reports] == ["bottom_inner_face", "top_inner_face"]
    assert reports[0]["selected_atom_count"] > 0
    assert reports[0]["target_charge_e"] > 0.0
    assert reports[1]["target_charge_e"] < 0.0

    meta = json.loads(result.stacked_cell.GetProp("_yadonpy_cell_meta"))
    species = meta["species"]
    assert all(sp.get("fragment_index") is not None for sp in species)
    assert all(sp.get("force_write_from_fragment") is True for sp in species)
    assert abs(float(meta["net_charge_scaled"])) < 1.0e-8


def test_layer_stack_fixed_charge_region_can_target_molecular_slab(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    water = utils.mol_from_smiles("O", name="WAT")
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(
                MolecularLayerSpec(
                    name="AMORPHOUS",
                    species=(water,),
                    counts=(4,),
                    thickness_nm=1.2,
                    density_target_g_cm3=0.2,
                    layer_kind="generic",
                ),
            ),
            name="charged_amorphous_slab",
            fixed_charge_regions=(
                FixedChargeRegionSpec(
                    layer_name="AMORPHOUS",
                    region="top",
                    thickness_nm=0.6,
                    mode="total_charge",
                    charge_e=0.5,
                    exclude_elements=("H",),
                ),
            ),
        ),
        work_dir=tmp_path,
        restart=False,
        charge_method="gasteiger",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    report = manifest["fixed_charge_regions"][0]
    assert report["selected_atom_count"] >= 1
    assert report["target_charge_e"] == 0.5
    meta = json.loads(result.stacked_cell.GetProp("_yadonpy_cell_meta"))
    assert any(sp.get("layer_name") == "AMORPHOUS" and sp.get("fragment_index") is not None for sp in meta["species"])


def test_layer_stack_adds_periodic_closing_gap_for_xyz(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(
                GraphiteLayerSpec(name="BOTTOM", nx=2, ny=2, n_layers=1),
                GraphiteLayerSpec(name="TOP", nx=2, ny=2, n_layers=1),
            ),
            name="closed_periodic_graphite_stack",
            default_gap_nm=0.35,
        ),
        work_dir=tmp_path,
        restart=False,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.acceptance["pbc_closing_gap_ok"] is True
    assert result.acceptance["pbc_closing_gap_nm"] >= 0.35 - 1.0e-6
    assert any(gap.get("pbc_closing") for gap in manifest["gaps"])


def test_graphite_edge_defaults_to_nonperiodic(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(GraphiteLayerSpec(name="EDGE", nx=2, ny=2, n_layers=1, orientation="edge", edge_cap="OH"),),
            name="edge_graphite",
        ),
        work_dir=tmp_path,
        restart=False,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    layer = manifest["layers"][0]
    assert layer["periodic_xy"] is False
    assert layer["effective_edge_cap"] == "OH"
    assert layer["lateral_padding_nm"] == 1.0


def test_cmcna_layer_keeps_polymer_and_na_in_same_layer_group(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    cmc = utils.mol_from_smiles("CC(=O)[O-]", name="CMC_FRAGMENT")
    na = utils.mol_from_smiles("[Na+]", name="Na")
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(
                GraphiteLayerSpec(name="GRAPHITE", nx=2, ny=2, n_layers=1),
                MolecularLayerSpec(
                    name="CMCNA",
                    species=(cmc, na),
                    counts=(1, 1),
                    thickness_nm=1.4,
                    density_target_g_cm3=0.5,
                    layer_kind="cmcna",
                    polyelectrolyte_mode=True,
                ),
            ),
            name="cmcna_stack",
        ),
        work_dir=tmp_path,
        restart=False,
        charge_method="gasteiger",
    )
    ndx_text = result.system_ndx.read_text(encoding="utf-8")
    assert "[ CMCNA ]" in ndx_text
    assert "[ MOBILE ]" in ndx_text
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    cmc_layer = [layer for layer in manifest["layers"] if layer["name"] == "CMCNA"][0]
    assert cmc_layer["polyelectrolyte_mode"] is True
    contact = cmc_layer["counterion_contact"]
    assert contact["enabled"] is True
    assert contact["paired_count"] == 1
    assert contact["unpaired_carboxylate_sites"] == 0
    assert contact["unpaired_na_counterions"] == 0
    assert 0.20 <= contact["o_na_distance_min_nm"] <= 0.27
    assert 0.20 <= contact["o_na_distance_max_nm"] <= 0.27


def test_run_layer_stack_nvt_uses_exported_stack_artifact(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    water = utils.mol_from_smiles("O", name="WAT")
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(
                GraphiteLayerSpec(name="GRAPHITE", nx=2, ny=2, n_layers=1),
                MolecularLayerSpec(
                    name="ELECTROLYTE",
                    species=(water,),
                    counts=(1,),
                    thickness_nm=1.0,
                    density_target_g_cm3=0.4,
                    layer_kind="electrolyte",
                ),
            ),
            name="nvt_input",
        ),
        work_dir=tmp_path / "stack",
        restart=False,
    )

    class DummyNVT:
        def __init__(self, ac, work_dir):
            self.ac = ac
            self.work_dir = Path(work_dir)

        def _ensure_system_exported(self):
            out = self.work_dir / "02_system"
            out.mkdir(parents=True, exist_ok=True)
            gro = out / "system.gro"
            top = out / "system.top"
            ndx = out / "system.ndx"
            meta = out / "system_meta.json"
            gro.write_text("dummy\n0\n   1.0   1.0   1.0\n", encoding="utf-8")
            top.write_text("[ system ]\n", encoding="utf-8")
            ndx.write_text("[ System ]\n", encoding="utf-8")
            meta.write_text("{}", encoding="utf-8")
            return SystemExportResult(
                system_gro=gro,
                system_top=top,
                system_ndx=ndx,
                molecules_dir=out / "molecules",
                system_meta=meta,
                box_nm=1.0,
                species=[],
                box_lengths_nm=(1.0, 1.0, 1.0),
            )

        def exec(self, **kwargs):
            prod = self.work_dir / "05_nvt_production" / "01_nvt"
            prod.mkdir(parents=True, exist_ok=True)
            (prod / "md.gro").write_text("dummy\n0\n   1.0   1.0   1.0\n", encoding="utf-8")
            (prod / "md.trr").write_bytes(b"")
            return self.ac

    from yadonpy.sim.preset import eq

    monkeypatch.setattr(eq, "NVT", DummyNVT)
    monkeypatch.setattr("yadonpy.interface.layer_stack.analyze_layer_stack_interface", lambda **kwargs: {"summary_path": str(tmp_path / "analysis.json")})

    out = run_layer_stack_nvt(result, work_dir=tmp_path / "nvt", time_ns=0.01, run_analysis=True)
    assert isinstance(out, LayerStackNvtResult)
    assert out.final_gro is not None
    assert out.trajectory is not None and out.trajectory.suffix == ".trr"
    assert out.trr is not None
    assert out.xtc is None
    assert out.summary_path.exists()
    copied_ndx = (tmp_path / "nvt" / "02_system" / "system.ndx").read_text(encoding="utf-8")
    assert "LAYER_00_GRAPHITE" in copied_ndx


def test_run_layer_stack_relaxation_builds_fixed_xy_z_npt_workflow(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    water = utils.mol_from_smiles("O", name="WAT")
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(
                GraphiteLayerSpec(name="GRAPHITE", nx=2, ny=2, n_layers=1),
                MolecularLayerSpec(
                    name="ELECTROLYTE",
                    species=(water,),
                    counts=(1,),
                    thickness_nm=1.0,
                    density_target_g_cm3=0.4,
                    layer_kind="electrolyte",
                ),
            ),
            name="relax_input",
        ),
        work_dir=tmp_path / "stack",
        restart=False,
    )

    captured: dict[str, object] = {}

    class DummyJob:
        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["gro"] = Path(gro)
            captured["top"] = Path(top)
            captured["ndx"] = Path(ndx)
            captured["out_dir"] = Path(out_dir)
            captured["stages"] = list(stages)
            captured["resources"] = resources

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"])
            final = out_dir / captured["stages"][-1].name
            final.mkdir(parents=True, exist_ok=True)
            (final / "md.gro").write_text("dummy\n0\n   1.0   1.0   1.5\n", encoding="utf-8")
            (final / "md.xtc").write_bytes(b"")
            (out_dir / "summary.json").write_text("{}", encoding="utf-8")
            return out_dir / "summary.json"

    from yadonpy.gmx.workflows import eq as eqmod

    monkeypatch.setattr(eqmod, "EquilibrationJob", DummyJob)
    monkeypatch.setattr("yadonpy.interface.layer_stack.analyze_layer_stack_interface", lambda **kwargs: {"summary_path": str(tmp_path / "analysis.json"), "geometry_health": {"phase_order_ok": True}, "outputs": {}})

    out = run_layer_stack_relaxation(
        result,
        work_dir=tmp_path / "relax",
        time_ns=0.01,
        pre_nvt_ns=0.002,
        z_npt_ns=0.003,
        run_analysis=True,
        gpu=0,
    )

    stages = captured["stages"]
    assert isinstance(out, LayerStackRelaxationResult)
    assert [stage.name for stage in stages] == ["01_pre_minimize", "02_pre_nvt", "03_z_npt", "04_final_nvt"]
    z_npt_mdp = stages[2].mdp.render()
    final_nvt_mdp = stages[3].mdp.render()
    assert "pcoupltype                = semiisotropic" in z_npt_mdp
    assert "ref_p                     = 1 1" in z_npt_mdp
    assert "compressibility           = 0 4.5e-05" in z_npt_mdp
    assert "pcoupl" not in final_nvt_mdp
    assert captured["gro"] == tmp_path / "relax" / "02_system" / "system.gro"
    assert captured["ndx"] == tmp_path / "relax" / "02_system" / "system.ndx"
    assert out.final_gro == tmp_path / "relax" / "05_relaxation_workflow" / "04_final_nvt" / "md.gro"
    assert out.trajectory == tmp_path / "relax" / "05_relaxation_workflow" / "04_final_nvt" / "md.xtc"
    summary = json.loads(out.summary_path.read_text(encoding="utf-8"))
    assert summary["stage_order"] == ["01_pre_minimize", "02_pre_nvt", "03_z_npt", "04_final_nvt"]
    assert summary["z_npt_mdp_overrides"]["compressibility"] == "0 4.5e-05"
    assert (tmp_path / "relax" / "layer_stack_manifest.json").is_file()
    copied_ndx = (tmp_path / "relax" / "02_system" / "system.ndx").read_text(encoding="utf-8")
    assert "LAYER_00_GRAPHITE" in copied_ndx


def test_run_layer_stack_relaxation_auto_skips_z_npt_for_vacuum_stack(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    water = utils.mol_from_smiles("O", name="WAT")
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(
                VacuumLayerSpec(thickness_nm=1.0, name="VACUUM_BOTTOM"),
                MolecularLayerSpec(
                    name="ELECTROLYTE",
                    species=(water,),
                    counts=(2,),
                    thickness_nm=1.0,
                    density_target_g_cm3=0.4,
                    layer_kind="electrolyte",
                ),
                VacuumLayerSpec(thickness_nm=1.0, name="VACUUM_TOP"),
            ),
            pbc_mode="xy",
            name="vacuum_electrolyte_vacuum",
        ),
        work_dir=tmp_path / "vacuum_stack",
        restart=False,
    )

    captured: dict[str, object] = {}

    class DummyJob:
        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["out_dir"] = Path(out_dir)
            captured["stages"] = list(stages)

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"])
            final = out_dir / captured["stages"][-1].name
            final.mkdir(parents=True, exist_ok=True)
            (final / "md.gro").write_text("dummy\n0\n   1.0   1.0   3.0\n", encoding="utf-8")
            (final / "md.xtc").write_bytes(b"")
            (out_dir / "summary.json").write_text("{}", encoding="utf-8")
            return out_dir / "summary.json"

    from yadonpy.gmx.workflows import eq as eqmod

    monkeypatch.setattr(eqmod, "EquilibrationJob", DummyJob)
    monkeypatch.setattr("yadonpy.interface.layer_stack.analyze_layer_stack_interface", lambda **kwargs: {"summary_path": str(tmp_path / "analysis.json"), "geometry_health": {"phase_order_ok": True}, "outputs": {}})

    out = run_layer_stack_relaxation(
        result,
        work_dir=tmp_path / "vacuum_relax",
        time_ns=0.01,
        z_npt_ns=0.5,
        relax_z="auto",
        compression_anneal=ZCompressionAnnealSpec(enabled="auto", cycles=1),
        gpu=0,
    )

    stages = captured["stages"]
    assert [stage.name for stage in stages] == ["01_pre_minimize", "02_pre_nvt", "03_final_nvt"]
    assert all(stage.kind != "npt" for stage in stages)
    assert out.final_gro == tmp_path / "vacuum_relax" / "05_relaxation_workflow" / "03_final_nvt" / "md.gro"
    summary = json.loads(out.summary_path.read_text(encoding="utf-8"))
    assert summary["relax_z"] == {
        "requested": "auto",
        "resolved": False,
        "reason": "auto_explicit_vacuum_layer",
    }
    assert summary["time_ns"]["z_npt"] == 0.0
    assert summary["z_npt_mdp_overrides"] is None
    assert summary["compression_anneal"]["resolved"] is False
    assert summary["compression_anneal"]["reason"] == "relax_z_disabled"


def test_run_layer_stack_relaxation_compression_anneal_workflow(monkeypatch, tmp_path: Path):
    _patch_fake_export(monkeypatch)
    water = utils.mol_from_smiles("O", name="WAT")
    result = build_layer_stack(
        stack=LayerStackSpec(
            layers=(
                GraphiteLayerSpec(name="BOTTOM", nx=2, ny=2, n_layers=1),
                MolecularLayerSpec(
                    name="ELECTROLYTE",
                    species=(water,),
                    counts=(2,),
                    thickness_nm=2.0,
                    density_target_g_cm3=0.4,
                    layer_kind="electrolyte",
                ),
                GraphiteLayerSpec(name="TOP", nx=2, ny=2, n_layers=1),
            ),
            name="compression_input",
            default_gap_nm=0.35,
        ),
        work_dir=tmp_path / "compression_stack",
        restart=False,
    )
    with result.system_ndx.open("a", encoding="utf-8") as fh:
        top_atoms = " ".join(str(i) for i in range(23, 39))
        fh.write(f"\n[ LAYER_02_TOP ]\n{top_atoms}\n[ TOP ]\n{top_atoms}\n")

    jobs: list[dict[str, object]] = []

    class DummyJob:
        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            self.gro = Path(gro)
            self.out_dir = Path(out_dir)
            self.stages = list(stages)
            jobs.append({"gro": self.gro, "out_dir": self.out_dir, "stages": self.stages})

        def run(self, *, restart=False):
            current = self.gro
            for stage in self.stages:
                stage_dir = self.out_dir / stage.name
                stage_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(current, stage_dir / "md.gro")
                if stage.name.endswith("final_nvt"):
                    (stage_dir / "md.xtc").write_bytes(b"")
                current = stage_dir / "md.gro"
            (self.out_dir / "summary.json").write_text("{}", encoding="utf-8")
            return self.out_dir / "summary.json"

    from yadonpy.gmx.workflows import eq as eqmod

    monkeypatch.setattr(eqmod, "EquilibrationJob", DummyJob)
    monkeypatch.setattr("yadonpy.interface.layer_stack.analyze_layer_stack_interface", lambda **kwargs: {"summary_path": str(tmp_path / "analysis.json"), "geometry_health": {"phase_order_ok": True}, "outputs": {}})

    out = run_layer_stack_relaxation(
        result,
        work_dir=tmp_path / "compression_relax",
        time_ns=0.01,
        pre_nvt_ns=0.001,
        z_npt_ns=0.001,
        relax_z=True,
        compression_anneal=ZCompressionAnnealSpec(
            enabled=True,
            cycles=1,
            tmax_K=380.0,
            pmax_bar=2000.0,
            target_z_nm=float(result.box_nm[2]) * 0.90,
            max_z_shrink_per_cycle=0.04,
            hot_nvt_ns=0.001,
            compression_npt_ns=0.001,
            cool_nvt_ns=0.001,
        ),
        run_analysis=False,
        gpu=0,
    )

    summary = json.loads(out.summary_path.read_text(encoding="utf-8"))
    assert summary["compression_anneal"]["resolved"] is True
    assert summary["compression_anneal"]["reason"] == "explicit_true"
    assert summary["compression_anneal"]["cycles"][0]["accepted_attempt"] == 1
    assert summary["stage_order"] == [
        "01_pre_minimize",
        "02_pre_nvt",
        "03_compress_c01_geometry",
        "04_compress_c01_minimize",
        "05_compress_c01_hot_nvt",
        "06_compress_c01_hot_z_npt",
        "07_compress_c01_cool_nvt",
        "08_final_z_npt",
        "09_final_nvt",
    ]
    geometry = summary["compression_anneal"]["cycles"][0]["attempts"][0]["geometry"]
    assert geometry["applied"] is True
    assert geometry["mode"] == "inter_electrode"
    hot_z_npt = jobs[1]["stages"][2].mdp.render()
    assert "ref_p                     = 1 2000" in hot_z_npt
    assert "compressibility           = 0 4.5e-05" in hot_z_npt
    assert out.final_gro == tmp_path / "compression_relax" / "05_relaxation_workflow" / "09_final_nvt" / "md.gro"


def test_density_profile_flags_low_cmc_density(tmp_path: Path):
    profile = tmp_path / "z_density_profiles.csv"
    profile.write_text(
        "\n".join(
            [
                "entity_kind,entity,z_lo_nm,z_hi_nm,mass_density_g_cm3",
                "phase,CMCNA,0.0,0.1,0.20",
                "phase,CMCNA,0.1,0.2,0.45",
                "phase,CMCNA,0.2,0.3,0.55",
                "phase,ELECTROLYTE,0.0,0.1,1.10",
                "phase,ELECTROLYTE,0.1,0.2,1.20",
                "phase,ELECTROLYTE,0.2,0.3,1.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = layer_stack_mod._summarize_density_profile({"outputs": {"z_density_profiles_csv": str(profile)}})

    gate = summary["cmc_density_gate"]
    assert gate["available"] is True
    assert gate["ok"] is False
    assert gate["severity"] == "severe"
    assert gate["reference_bulk_density_g_cm3"] == 1.5
    assert gate["warning_floor_g_cm3"] == 0.90
    assert gate["severe_floor_g_cm3"] == 0.75
    assert gate["primary_phase"] == "CMCNA"
    assert gate["primary_metric"] == "core_region_mean_g_cm3"
    assert gate["primary_density_g_cm3"] < 0.75
    assert summary["phase_density_g_cm3"]["CMCNA"]["core_region_total_density_g_cm3"] > 1.0
