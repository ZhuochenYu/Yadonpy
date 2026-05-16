from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yadonpy.core import utils
from yadonpy.core.graphite import build_graphite
from yadonpy.interface.layer_stack import (
    ElectrodeChargeSpec,
    GraphiteLayerSpec,
    LayerStackNvtResult,
    LayerStackSpec,
    MolecularLayerSpec,
    VacuumLayerSpec,
    build_layer_stack,
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
