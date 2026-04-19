from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
import yadonpy.interface.builder as builder_mod
import yadonpy.interface.prep as prep_mod
import yadonpy.sim.preset.eq as eqmod

from yadonpy.core import workdir, workunit
from yadonpy.gmx.topology import MoleculeType, SystemTopology, parse_system_top
from yadonpy.interface import AreaMismatchPolicy, BuiltInterface, InterfaceBuilder, InterfaceDynamics, InterfaceProtocol, InterfaceRouteSpec, build_bulk_equilibrium_profile, build_interface_group_catalog, equilibrate_bulk_with_eq21, fixed_xy_semiisotropic_npt_overrides, format_cell_charge_audit, format_charge_meta_audit, make_orthorhombic_pack_cell, plan_direct_electrolyte_counts, plan_direct_polymer_matched_interface_preparation, plan_fixed_xy_direct_electrolyte_preparation, plan_fixed_xy_direct_pack_box, plan_polymer_anchored_interface_preparation, plan_probe_electrolyte_preparation, plan_probe_polymer_matched_interface_preparation, plan_rescaled_bulk_counts, plan_resized_electrolyte_counts, plan_resized_electrolyte_preparation_from_probe, recommend_electrolyte_alignment, recommend_polymer_diffusion_interface_recipe
from yadonpy.interface.builder import FragmentRecord, _compress_molecule_sequence, _rebalance_fragment_selection_for_charge, _select_fragments_for_slab, _write_local_top
from yadonpy.interface.protocol import _resolve_route_b_wall_atomtype
from yadonpy.io.gromacs_system import SystemExportResult
from yadonpy.core import poly


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_bulk_fixture(root: Path, *, lx: float, ly: float, lz: float, ff_parameters_text: str | None = None, mol_itp_text: str | None = None, include_ff_parameters: bool = True, atoms: list[tuple[int, str, str, int, float, float, float]] | None = None, molecule_count: int = 4) -> Path:
    sys_dir = root / "02_system"
    mol_dir = sys_dir / "molecules"
    mol_dir.mkdir(parents=True, exist_ok=True)
    if include_ff_parameters:
        _write_text(
            mol_dir / "ff_parameters.itp",
            (ff_parameters_text or "[ defaults ]\n1 2 yes 0.5 0.5\n").strip() + "\n",
        )
    _write_text(
        mol_dir / "SOL.itp",
        (mol_itp_text or """
[ moleculetype ]
SOL   3

[ atoms ]
1  C   1  SOL  C1  1  0.0  12.011
2  H   1  SOL  H1  1  0.0  1.008

[ bonds ]
1 2
""").strip() + "\n",
    )
    include_lines = []
    if include_ff_parameters:
        include_lines.append('#include "molecules/ff_parameters.itp"')
    include_lines.append('#include "molecules/SOL.itp"')
    gro_atoms = atoms or [
        (1, "SOL", "C1", 1, 0.20, 0.20, 0.20),
        (1, "SOL", "H1", 2, 0.24, 0.20, 0.20),
        (2, "SOL", "C1", 3, 0.70, 0.20, 0.60),
        (2, "SOL", "H1", 4, 0.74, 0.20, 0.60),
        (3, "SOL", "C1", 5, 0.20, 0.70, 0.95),
        (3, "SOL", "H1", 6, 0.24, 0.70, 0.95),
        (4, "SOL", "C1", 7, 0.70, 0.70, 1.30),
        (4, "SOL", "H1", 8, 0.74, 0.70, 1.30),
    ]
    _write_text(sys_dir / "system.top", "\n".join(include_lines) + f'\n\n[ system ]\nfake bulk\n\n[ molecules ]\nSOL {int(molecule_count)}\n')
    _write_text(sys_dir / "system.ndx", "[ System ]\n" + " ".join(str(i) for i in range(1, len(gro_atoms) + 1)) + "\n")
    _write_text(sys_dir / "system_meta.json", json.dumps({"density_g_cm3": 1.0}, indent=2) + "\n")
    gro = root / "03_EQ21" / "03_EQ21" / "step_21" / "md.gro"
    lines = ["fake", f"{len(gro_atoms):5d}"]
    for resnr, resname, atomname, atomnr, x, y, z in gro_atoms:
        lines.append(f"{resnr:5d}{resname:<5}{atomname:>5}{atomnr:5d}{x:8.3f}{y:8.3f}{z:8.3f}")
    lines.append(f"{lx:10.5f}{ly:10.5f}{lz:10.5f}")
    _write_text(gro, "\n".join(lines) + "\n")
    return root


def test_workunit_child_path(tmp_path: Path):
    wd = workdir(tmp_path / "root", restart=True)
    child = workunit(wd, "ac_poly")
    assert child.path_obj.name == "ac_poly"
    assert child.path_obj.parent == wd.path_obj


def test_equilibrate_bulk_with_eq21_forwards_exec_kwargs(tmp_path: Path, monkeypatch):
    calls: dict[str, object] = {}

    class _FakeAnalyze:
        def get_all_prop(self, temp, press, save=True):
            calls["analyze"] = {"temp": temp, "press": press, "save": save}
            return {}

        def check_eq(self):
            return True

    class _FakeEQ21:
        def __init__(self, ac, work_dir):
            self.ac = ac
            self.work_dir = Path(work_dir)

        def ensure_system_exported(self):
            sys_dir = self.work_dir / "02_system"
            mol_dir = sys_dir / "molecules"
            mol_dir.mkdir(parents=True, exist_ok=True)
            _write_text(mol_dir / "ff_parameters.itp", "[ defaults ]\n1 2 yes 0.5 0.5\n")
            _write_text(mol_dir / "SOL.itp", "[ moleculetype ]\nSOL 3\n\n[ atoms ]\n1 C 1 SOL C1 1 0.0 12.011\n")
            _write_text(sys_dir / "system.top", '#include "molecules/ff_parameters.itp"\n#include "molecules/SOL.itp"\n\n[ system ]\nmock\n\n[ molecules ]\nSOL 1\n')
            _write_text(sys_dir / "system.gro", "mock\n    1\n    1SOL     C1    1   0.100   0.100   0.100\n   1.00000   1.00000   1.00000\n")
            _write_text(sys_dir / "system.ndx", "[ System ]\n1\n")
            raw_dir = sys_dir / "01_raw_non_scaled"
            raw_dir.mkdir(parents=True, exist_ok=True)
            _write_text(raw_dir / "system.top", '#include "../molecules/ff_parameters.itp"\n#include "../molecules/SOL.itp"\n\n[ system ]\nmock\n\n[ molecules ]\nSOL 1\n')
            _write_text(raw_dir / "system.gro", "mock\n    1\n    1SOL     C1    1   0.100   0.100   0.100\n   1.00000   1.00000   1.00000\n")
            _write_text(raw_dir / "system.ndx", "[ System ]\n1\n")
            _write_text(raw_dir / "system_meta.json", "{}\n")
            return SystemExportResult(
                system_gro=sys_dir / "system.gro",
                system_top=sys_dir / "system.top",
                system_ndx=sys_dir / "system.ndx",
                molecules_dir=mol_dir,
                system_meta=raw_dir / "system_meta.json",
                box_nm=(1.0, 1.0, 1.0),
                species=[],
            )

        def exec(self, **kwargs):
            calls["exec"] = kwargs
            return self.ac

        def analyze(self):
            return _FakeAnalyze()

    monkeypatch.setattr(eqmod, "EQ21step", _FakeEQ21)

    out = equilibrate_bulk_with_eq21(
        label="mock",
        ac={"cell": "mock"},
        work_dir=tmp_path / "bulk",
        temp=300.0,
        press=1.0,
        mpi=1,
        omp=8,
        gpu=1,
        gpu_id=0,
        additional_loops=0,
        eq21_exec_kwargs={"eq21_tmax": 600.0, "eq21_pre_nvt_ps": 2.5},
    )

    assert calls["exec"]["eq21_tmax"] == 600.0
    assert calls["exec"]["eq21_pre_nvt_ps"] == 2.5
    assert calls["exec"]["omp"] == 8
    assert out.raw_system_meta.name == "system_meta.json"


def test_interface_builder_route_a_and_b(tmp_path: Path):
    bulk_bottom = _make_bulk_fixture(tmp_path / "ac_poly", lx=1.0, ly=1.0, lz=1.6)
    bulk_top = _make_bulk_fixture(tmp_path / "ac_electrolyte", lx=1.2, ly=1.1, lz=1.6)

    builder = InterfaceBuilder(work_dir=tmp_path / "interface_work", restart=True)
    src_bottom = builder.bulk_source(name="ac_poly", work_dir=bulk_bottom)
    src_top = builder.bulk_source(name="ac_electrolyte", work_dir=bulk_top)
    assert (tmp_path / "interface_work" / "01_snapshots" / "ac_poly" / "representative_whole.gro").exists()

    built_a = builder.build(name="route_a_case", bottom=src_bottom, top=src_top, route=InterfaceRouteSpec.route_a(bottom_thickness_nm=0.8, top_thickness_nm=0.8, gap_nm=0.4))
    assert built_a.system_gro.exists()
    assert built_a.system_top.exists()
    assert built_a.system_gro.parent == tmp_path / "interface_work" / "03_interface"
    top_text = built_a.system_top.read_text(encoding="utf-8")
    assert top_text.startswith("; yadonpy generated system.top\n#include \"molecules/ff_parameters.itp\"\n")
    ff_params_text = (built_a.system_top.parent / "molecules" / "ff_parameters.itp").read_text(encoding="utf-8")
    assert ff_params_text.startswith(
        "; yadonpy merged interface FF parameter blocks\n"
        "[ defaults ]\n"
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
        "1 2 yes 0.5 0.8333333333\n"
    )
    ndx_text = built_a.system_ndx.read_text(encoding="utf-8")
    assert "[ BOTTOM ]" in ndx_text
    assert "[ TOP ]" in ndx_text
    assert "[ INTERFACE_ZONE ]" in ndx_text
    assert "[ BOTTOM_MOL_SOL ]" in ndx_text
    assert "[ BOTTOM1_SOL ]" in ndx_text
    assert "[ TOP_MOL_SOL ]" in ndx_text
    assert "[ TOP1_SOL ]" in ndx_text
    assert "[ BOTTOM_TYPE_SOL_C ]" in ndx_text
    assert "[ BOTTOM1_TYPE_SOL_C ]" in ndx_text
    assert "[ TOP_TYPE_SOL_C ]" in ndx_text
    assert "[ TOP1_TYPE_SOL_C ]" in ndx_text
    assert "[ ATYPE_C ]" in ndx_text
    assert "[ BOTTOM1_ATYPE_C ]" in ndx_text
    assert "[ BOTTOM_INST_SOL_0001 ]" in ndx_text
    assert "[ BOTTOM1_INST_SOL_0001 ]" in ndx_text
    assert "[ TOP_INST_SOL_0001 ]" in ndx_text
    assert "[ TOP1_INST_SOL_0001 ]" in ndx_text
    assert "[ REP_SOL ]" in ndx_text
    assert "[ BOTTOM1_REP_SOL ]" in ndx_text
    catalog_path = built_a.system_ndx.parent / "system_ndx_groups.json"
    assert catalog_path.exists()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert "BOTTOM_MOL_SOL" in catalog["categories"]["region_moltypes"]["BOTTOM"]
    assert "BOTTOM_INST_SOL_0001" in catalog["categories"]["region_instances"]["BOTTOM"]
    assert "BOTTOM1_SOL" in catalog["categories"]["layer_moltypes"]["BOTTOM1"]
    assert "BOTTOM1_TYPE_SOL_C" in catalog["categories"]["layer_moltype_atomtypes"]["BOTTOM1"]
    assert catalog["group_sizes"]["BOTTOM_MOL_SOL"] > 0
    meta = json.loads(built_a.system_meta.read_text(encoding="utf-8"))
    assert meta["route"] == "route_a"
    assert meta["axis"] == "Z"
    assert tuple(meta["top_lateral_shift_fraction"]) == (0.5, 0.5)
    assert meta["geometry_validation"]["atoms_outside_primary_box"] == 0
    assert meta["geometry_validation"]["assembled_gap_nm"] >= 0.0
    assert any("dense-window slab selection" in note for note in meta["notes"])

    built_b = builder.build_from_bulk_workdirs(name="route_b_case", bottom_name="ac_poly", bottom_work_dir=bulk_bottom, top_name="ac_electrolyte", top_work_dir=bulk_top, route=InterfaceRouteSpec.route_b(bottom_thickness_nm=0.8, top_thickness_nm=0.8, gap_nm=0.4, vacuum_nm=2.0))
    meta_b = json.loads(built_b.system_meta.read_text(encoding="utf-8"))
    assert meta_b["route"] == "route_b"
    assert meta_b["box_nm"][2] > meta["box_nm"][2]
    gro_lines = built_b.system_gro.read_text(encoding="utf-8").splitlines()
    natoms = int(gro_lines[1].strip())
    z_values = [float(line[36:44]) for line in gro_lines[2:2 + natoms]]
    assert min(z_values) >= 0.05 - 1.0e-6
    assert any("shifted away from the z-wall" in note for note in meta_b["notes"])


def test_interface_builder_merges_parameter_include_files(tmp_path: Path):
    bulk_bottom = _make_bulk_fixture(
        tmp_path / "ac_poly",
        lx=1.0,
        ly=1.0,
        lz=1.6,
        ff_parameters_text=(
            "; bottom params\n"
            "[ atomtypes ]\n"
            "c 12.0110 0.0000 A 0.331521 0.413379\n"
            "Na 22.9900 0.0000 A 0.261746 0.126029\n"
        ),
    )
    bulk_top = _make_bulk_fixture(
        tmp_path / "ac_electrolyte",
        lx=1.0,
        ly=1.0,
        lz=1.6,
        ff_parameters_text=(
            "; top params\n"
            "; patched by yadonpy DRIH (bond+angle)\n"
            "[ atomtypes ]\n"
            "c 12.0110 0.0000 A 0.331521 0.413379\n"
            "Li 6.9400 0.0000 A 0.226288 0.013625\n"
        ),
    )

    builder = InterfaceBuilder(work_dir=tmp_path / "interface_conflict", restart=True)
    built = builder.build_from_bulk_workdirs(
        name="route_a_case",
        bottom_name="ac_poly",
        bottom_work_dir=bulk_bottom,
        top_name="ac_electrolyte",
        top_work_dir=bulk_top,
        route=InterfaceRouteSpec.route_a(bottom_thickness_nm=0.8, top_thickness_nm=0.8, gap_nm=0.4),
    )

    top_text = built.system_top.read_text(encoding="utf-8")
    assert top_text.startswith("; yadonpy generated system.top\n#include \"molecules/ff_parameters.itp\"\n")
    assert '#include "molecules/ff_parameters.itp"' in top_text
    assert (built.system_top.parent / "molecules" / "ff_parameters.itp").exists()
    assert top_text.index('#include "molecules/ff_parameters.itp"') < top_text.index('#include "molecules/SOL.itp"')
    assert 'ac_electrolyte_ff_parameters.itp' not in top_text
    merged = (built.system_top.parent / "molecules" / "ff_parameters.itp").read_text(encoding="utf-8")
    assert merged.startswith(
        "; yadonpy merged interface FF parameter blocks\n"
        "[ defaults ]\n"
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
        "1 2 yes 0.5 0.8333333333\n"
    )
    assert merged.count("[ atomtypes ]") == 1
    assert merged.count("[ defaults ]") == 1
    assert merged.count("; yadonpy merged interface FF parameter blocks") == 1
    assert "Na 22.9900 0.0000 A 0.261746 0.126029" in merged
    assert "Li 6.9400 0.0000 A 0.226288 0.013625" in merged
    assert merged.count("c 12.0110 0.0000 A 0.331521 0.413379") == 1


def test_interface_builder_normalizes_mixed_parameter_and_molecule_itps(tmp_path: Path):
    mixed_itp = (
        "; embedded params\n"
        "[ atomtypes ]\n"
        "c 12.0110 0.0000 A 0.331521 0.413379\n"
        "h 1.0080 0.0000 A 0.250000 0.125520\n\n"
        "[ moleculetype ]\n"
        "SOL   3\n\n"
        "[ atoms ]\n"
        "1  C   1  SOL  C1  1  0.0  12.011\n"
        "2  H   1  SOL  H1  1  0.0  1.008\n\n"
        "[ bonds ]\n"
        "1 2\n"
    )
    bulk_bottom = _make_bulk_fixture(
        tmp_path / "ac_poly",
        lx=1.0,
        ly=1.0,
        lz=1.6,
        mol_itp_text=mixed_itp,
        include_ff_parameters=False,
    )
    bulk_top = _make_bulk_fixture(
        tmp_path / "ac_electrolyte",
        lx=1.0,
        ly=1.0,
        lz=1.6,
        mol_itp_text=mixed_itp,
        include_ff_parameters=False,
    )

    builder = InterfaceBuilder(work_dir=tmp_path / "interface_mixed", restart=True)
    built = builder.build_from_bulk_workdirs(
        name="route_a_case",
        bottom_name="ac_poly",
        bottom_work_dir=bulk_bottom,
        top_name="ac_electrolyte",
        top_work_dir=bulk_top,
        route=InterfaceRouteSpec.route_a(bottom_thickness_nm=0.8, top_thickness_nm=0.8, gap_nm=0.4),
    )

    top_text = built.system_top.read_text(encoding="utf-8")
    assert top_text.startswith("; yadonpy generated system.top\n#include \"molecules/ff_parameters.itp\"\n")
    assert '#include "molecules/ff_parameters.itp"' in top_text
    assert top_text.index('#include "molecules/ff_parameters.itp"') < top_text.index('#include "molecules/SOL.itp"')
    normalized_itp = (built.system_top.parent / "molecules" / "SOL.itp").read_text(encoding="utf-8")
    merged_params = (built.system_top.parent / "molecules" / "ff_parameters.itp").read_text(encoding="utf-8")
    assert merged_params.startswith(
        "; yadonpy merged interface FF parameter blocks\n"
        "[ defaults ]\n"
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
        "1 2 yes 0.5 0.8333333333\n"
    )
    assert "[ atomtypes ]" not in normalized_itp
    assert normalized_itp.lstrip().startswith("[ moleculetype ]")
    assert merged_params.count("[ atomtypes ]") == 1
    assert "c 12.0110 0.0000 A 0.331521 0.413379" in merged_params


def test_interface_builder_schema_change_invalidates_cached_build(tmp_path: Path, monkeypatch):
    bulk_bottom = _make_bulk_fixture(tmp_path / "ac_poly", lx=1.0, ly=1.0, lz=1.6)
    bulk_top = _make_bulk_fixture(tmp_path / "ac_electrolyte", lx=1.0, ly=1.0, lz=1.6)

    builder = InterfaceBuilder(work_dir=tmp_path / "interface_resume", restart=True)
    built = builder.build_from_bulk_workdirs(
        name="route_a_case",
        bottom_name="ac_poly",
        bottom_work_dir=bulk_bottom,
        top_name="ac_electrolyte",
        top_work_dir=bulk_top,
        route=InterfaceRouteSpec.route_a(bottom_thickness_nm=0.8, top_thickness_nm=0.8, gap_nm=0.4),
    )

    built.system_top.write_text("corrupt\n", encoding="utf-8")
    monkeypatch.setattr(builder_mod, "_INTERFACE_BUILD_SCHEMA_VERSION", "test-bump")

    rebuilt = builder.build_from_bulk_workdirs(
        name="route_a_case",
        bottom_name="ac_poly",
        bottom_work_dir=bulk_bottom,
        top_name="ac_electrolyte",
        top_work_dir=bulk_top,
        route=InterfaceRouteSpec.route_a(bottom_thickness_nm=0.8, top_thickness_nm=0.8, gap_nm=0.4),
    )

    top_text = rebuilt.system_top.read_text(encoding="utf-8")
    assert "corrupt" not in top_text
    assert '#include "molecules/ff_parameters.itp"' in top_text


def test_interface_builder_ignores_stale_slab_meta_molecule_counts(tmp_path: Path):
    bulk_bottom = _make_bulk_fixture(tmp_path / "ac_poly", lx=1.0, ly=1.0, lz=1.6)
    bulk_top = _make_bulk_fixture(tmp_path / "ac_electrolyte", lx=1.0, ly=1.0, lz=1.6)

    builder = InterfaceBuilder(work_dir=tmp_path / "interface_meta_counts", restart=True)
    built = builder.build_from_bulk_workdirs(
        name="route_a_case",
        bottom_name="ac_poly",
        bottom_work_dir=bulk_bottom,
        top_name="ac_electrolyte",
        top_work_dir=bulk_top,
        route=InterfaceRouteSpec.route_a(bottom_thickness_nm=0.8, top_thickness_nm=0.8, gap_nm=0.4),
    )

    expected_counts = dict(parse_system_top(built.bottom_slab.top_path).molecules)
    for name, count in parse_system_top(built.top_slab.top_path).molecules:
        expected_counts[name] = expected_counts.get(name, 0) + int(count)

    for meta_path in (built.bottom_slab.meta_path, built.top_slab.meta_path):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["molecule_counts"] = {"SOL": 999}
        meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rebuilt = builder_mod._assemble_interface(
        name="route_a_case_rebuilt",
        out_dir=tmp_path / "reassembled",
        route_spec=InterfaceRouteSpec.route_a(bottom_thickness_nm=0.8, top_thickness_nm=0.8, gap_nm=0.4),
        bottom=built.bottom_slab,
        top=built.top_slab,
    )
    merged_counts = dict(parse_system_top(rebuilt.system_top).molecules)
    assert merged_counts == expected_counts


def test_interface_builder_keeps_top_molecule_whole_after_lateral_shift(tmp_path: Path):
    bulk_bottom = _make_bulk_fixture(tmp_path / "ac_poly", lx=1.2, ly=1.0, lz=1.6)
    bulk_top = _make_bulk_fixture(
        tmp_path / "ac_electrolyte",
        lx=1.2,
        ly=1.0,
        lz=1.6,
        atoms=[
            (1, "SOL", "C1", 1, 0.50, 0.20, 0.20),
            (1, "SOL", "H1", 2, 0.70, 0.20, 0.20),
        ],
        molecule_count=1,
    )

    builder = InterfaceBuilder(work_dir=tmp_path / "interface_wrap", restart=True)
    built = builder.build_from_bulk_workdirs(
        name="route_a_case",
        bottom_name="ac_poly",
        bottom_work_dir=bulk_bottom,
        top_name="ac_electrolyte",
        top_work_dir=bulk_top,
        route=InterfaceRouteSpec.route_a(
            bottom_thickness_nm=0.8,
            top_thickness_nm=0.8,
            gap_nm=0.4,
            top_lateral_shift_fraction=(1.10, 0.0),
        ),
    )

    frame = builder_mod._read_gro_frame(built.system_gro)
    top_atoms = frame.atoms[-2:]
    dx = abs(float(top_atoms[1].xyz_nm[0]) - float(top_atoms[0].xyz_nm[0]))
    assert dx < 0.4
    assert all(0.0 <= float(atom.xyz_nm[0]) < float(built.box_nm[0]) for atom in top_atoms)
    meta = json.loads(built.system_meta.read_text(encoding="utf-8"))
    assert meta["geometry_validation"]["atoms_outside_primary_box"] == 0
    assert 'yadonpy_whole' in built.system_gro.read_text(encoding='utf-8').splitlines()[0]


def test_interface_charge_audit_helpers_format_cell_and_meta(tmp_path: Path):
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C")
    mol.SetProp(
        "_yadonpy_cell_meta",
        json.dumps(
            {
                "net_charge_raw": 1.0,
                "net_charge_scaled": 0.8,
                "charge_tolerance": 1.0e-2,
                "net_charge_ok": False,
            }
        ),
    )
    meta = tmp_path / "system_meta.json"
    meta.write_text(json.dumps({"net_charge_e": 0.25}, indent=2) + "\n", encoding="utf-8")

    cell_line = format_cell_charge_audit("demo cell", mol)
    meta_line = format_charge_meta_audit("demo meta", meta)

    assert "raw=1.000000 e" in cell_line
    assert "scaled=0.800000 e" in cell_line
    assert "net_charge_e=0.250000 e" in meta_line


def test_interface_dynamics_rejects_invalid_topology_include_order(tmp_path: Path):
    built = BuiltInterface(
        name="demo",
        route="route_a",
        axis="Z",
        out_dir=tmp_path / "03_interface",
        system_gro=tmp_path / "03_interface" / "system.gro",
        system_top=tmp_path / "03_interface" / "system.top",
        system_ndx=tmp_path / "03_interface" / "system.ndx",
        system_meta=tmp_path / "03_interface" / "system_meta.json",
        bottom_slab=None,
        top_slab=None,
        protocol_manifest=tmp_path / "03_interface" / "protocol_manifest.json",
        box_nm=(1.0, 1.0, 1.0),
        notes=(),
    )
    mol_dir = built.system_top.parent / "molecules"
    mol_dir.mkdir(parents=True, exist_ok=True)
    built.system_gro.write_text("x\n", encoding="utf-8")
    built.system_meta.write_text("{}\n", encoding="utf-8")
    built.system_ndx.write_text("[ System ]\n1 2\n\n[ BOTTOM ]\n1\n\n[ TOP ]\n2\n", encoding="utf-8")
    (mol_dir / "SOL.itp").write_text("[ moleculetype ]\nSOL 3\n\n[ atoms ]\n1 C 1 SOL C1 1 0 12.011\n", encoding="utf-8")
    (mol_dir / "ff_parameters.itp").write_text("[ defaults ]\n1 2 yes 0.5 0.8333333333\n\n[ atomtypes ]\nc 12.0110 0.0000 A 0.331521 0.413379\n", encoding="utf-8")
    built.system_top.write_text(
        '#include "molecules/SOL.itp"\n#include "molecules/ff_parameters.itp"\n\n[ system ]\ninvalid\n\n[ molecules ]\nSOL 1\n',
        encoding="utf-8",
    )

    try:
        InterfaceDynamics(built=built, work_dir=tmp_path / "interface_md", restart=True).run(
            protocol=InterfaceProtocol.route_a(),
            mpi=1,
            omp=1,
            gpu=0,
            gpu_id=None,
        )
    except ValueError as exc:
        msg = str(exc)
        assert "invalid include order" in msg
        assert "ff_parameters.itp" in msg
    else:
        raise AssertionError("expected topology preflight to reject invalid include order")


def test_build_interface_group_catalog_reads_written_ndx(tmp_path: Path):
    bulk_bottom = _make_bulk_fixture(tmp_path / "ac_poly", lx=1.0, ly=1.0, lz=1.6)
    bulk_top = _make_bulk_fixture(tmp_path / "ac_electrolyte", lx=1.0, ly=1.0, lz=1.6)

    builder = InterfaceBuilder(work_dir=tmp_path / "interface_catalog", restart=True)
    built = builder.build_from_bulk_workdirs(
        name="route_a_case",
        bottom_name="ac_poly",
        bottom_work_dir=bulk_bottom,
        top_name="ac_electrolyte",
        top_work_dir=bulk_top,
        route=InterfaceRouteSpec.route_a(bottom_thickness_nm=0.8, top_thickness_nm=0.8, gap_nm=0.4),
    )

    catalog = build_interface_group_catalog(built.system_ndx)
    assert catalog["total_groups"] >= 10
    assert "BOTTOM_ATYPE_C" in catalog["categories"]["region_atomtypes"]["BOTTOM"]
    assert "BOTTOM1_ATYPE_C" in catalog["categories"]["layer_atomtypes"]["BOTTOM1"]
    assert "REP_SOL" in catalog["categories"]["global_representatives"]


def test_interface_protocol_definitions():
    a = InterfaceProtocol.route_a()
    b = InterfaceProtocol.route_b_wall(wall_mode="12-6", wall_atomtype="OW")
    stages_a = a.stages()
    stages_b = b.stages()
    assert len(stages_a) == 5
    assert len(stages_b) == 5
    assert any(s.name == "04_exchange" for s in stages_a)
    assert any(s.name == "05_production" for s in stages_b)
    assert "freezegrps               = BOTTOM_CORE TOP_CORE" in stages_a[0].mdp.render()
    assert "constraints              = none" in stages_a[0].mdp.render()
    assert "freezegrps" in stages_a[1].mdp.render()
    assert "BOTTOM_CORE TOP_CORE" in stages_a[1].mdp.render()
    assert "constraints              = none" in stages_a[1].mdp.render()
    assert "dt                       = 0.001" in stages_a[1].mdp.render()
    assert "freezegrps" not in stages_a[2].mdp.render()
    assert "pcoupl                    = Berendsen" in stages_a[2].mdp.render()
    assert "pcoupl                    = Berendsen" in stages_a[3].mdp.render()
    assert "pcoupl                    = C-rescale" in stages_a[4].mdp.render()
    assert "pcoupltype                = semiisotropic" in stages_a[2].mdp.render()


def test_interface_diffusion_protocol_definitions():
    a = InterfaceProtocol.route_a_diffusion(axis="Z")
    b = InterfaceProtocol.route_b_wall_diffusion(axis="Z", wall_mode="12-6", wall_atomtype="OW")
    stages_a = a.stages()
    stages_b = b.stages()

    assert len(stages_a) == 7
    assert len(stages_b) == 7
    assert [stage.name for stage in stages_a] == [
        "01_pre_contact_em",
        "02_gap_hold_nvt",
        "03_density_relax",
        "04_contact",
        "05_release",
        "06_exchange",
        "07_production",
    ]
    assert "freezegrps               = BOTTOM_CORE TOP_CORE" in stages_a[0].mdp.render()
    assert "freezedim                = Y Y Y Y Y Y" in stages_a[0].mdp.render()
    assert "tc-grps                   = BOTTOM TOP" in stages_a[1].mdp.render()
    assert "constraints              = none" in stages_a[2].mdp.render()
    assert "freezedim                = N N Y N N Y" in stages_a[2].mdp.render()
    assert "freezegrps" not in stages_a[4].mdp.render()
    assert "tc-grps                   = System" in stages_a[5].mdp.render()
    assert "pcoupl                    = C-rescale" in stages_a[6].mdp.render()
    assert "pbc                      = xy" in stages_b[2].mdp.render()
    assert "periodic-molecules       = yes" in stages_b[2].mdp.render()
    assert "ewald-geometry           = 3dc" in stages_b[2].mdp.render()
    assert "wall-r-linpot            = 0.05" in stages_b[2].mdp.render()
    assert "wall_atomtype            = OW OW" in stages_b[3].mdp.render()
    assert stages_b[3].mdp.params["wall_type"] == "12-6"
    assert stages_b[3].mdp.params["wall_atomtype"] == "OW"


def test_resolve_route_b_wall_atomtype_falls_back_to_defined_heavy_type(tmp_path: Path):
    mol_dir = tmp_path / "molecules"
    mol_dir.mkdir(parents=True, exist_ok=True)
    _write_text(
        mol_dir / "ff_parameters.itp",
        (
            "[ defaults ]\n"
            "1 2 yes 0.5 0.5\n\n"
            "[ atomtypes ]\n"
            "o    15.999  0.0 A  0.30  0.61\n"
            "os   15.999  0.0 A  0.31  0.30\n"
            "c3   12.011  0.0 A  0.34  0.45\n"
            "h1    1.008  0.0 A  0.24  0.08\n"
        ),
    )
    _write_text(
        mol_dir / "SOL.itp",
        (
            "[ moleculetype ]\n"
            "SOL   3\n\n"
            "[ atoms ]\n"
            "1  c3   1  SOL  C1  1  0.0  12.011\n"
            "2  h1   1  SOL  H1  1  0.0   1.008\n"
        ),
    )
    _write_text(
        tmp_path / "system.top",
        '#include "molecules/ff_parameters.itp"\n#include "molecules/SOL.itp"\n\n[ system ]\nmock\n\n[ molecules ]\nSOL 1\n',
    )

    resolved, available = _resolve_route_b_wall_atomtype(tmp_path / "system.top", "OW")

    assert resolved == "o"
    assert available[:3] == ("o", "os", "c3")


def test_write_local_top_preserves_actual_fragment_sequence(tmp_path: Path):
    mol_dir = tmp_path / "molecules"
    mol_dir.mkdir(parents=True, exist_ok=True)
    _write_text(mol_dir / "Li.itp", "[ moleculetype ]\nLi 3\n\n[ atoms ]\n1 Li 1 Li Li1 1 1.0 6.94\n")
    _write_text(
        mol_dir / "PF6.itp",
        (
            "[ moleculetype ]\nPF6 3\n\n"
            "[ atoms ]\n"
            "1 f 1 PF6 F1 1 -0.3 18.998\n"
            "2 p5 1 PF6 P2 2 1.8 30.974\n"
            "3 f 1 PF6 F3 3 -0.3 18.998\n"
            "4 f 1 PF6 F4 4 -0.3 18.998\n"
            "5 f 1 PF6 F5 5 -0.3 18.998\n"
            "6 f 1 PF6 F6 6 -0.3 18.998\n"
            "7 f 1 PF6 F7 7 -0.3 18.998\n\n"
            "[ bonds ]\n1 2\n2 3\n2 4\n2 5\n2 6\n2 7\n"
        ),
    )
    top_path = tmp_path / "system.top"
    sequence = ["Li", "PF6", "Li", "Li", "PF6"]
    _write_local_top(
        top_path,
        ['#include "molecules/Li.itp"', '#include "molecules/PF6.itp"'],
        _compress_molecule_sequence(sequence),
        "ordered",
    )

    topo = parse_system_top(top_path)

    assert topo.molecules == [("Li", 1), ("PF6", 1), ("Li", 2), ("PF6", 1)]


def test_direct_polymer_matched_interface_preparation_wraps_shared_helpers():
    prep = plan_direct_polymer_matched_interface_preparation(
        reference_box_nm=(4.2, 4.0, 6.8),
        bottom_thickness_nm=3.0,
        top_thickness_nm=3.5,
        gap_nm=0.6,
        surface_shell_nm=0.8,
        target_density_g_cm3=1.25,
        solvent_mol_weights=(88.0, 118.0, 104.0),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        salt_mol_weights=(6.94, 144.96),
        salt_molarity_M=1.0,
        min_salt_pairs=8,
        solvent_species_names=("EC", "DEC", "EMC"),
        salt_species_names=("Li", "PF6"),
        min_solvent_counts=(1, 1, 1),
        pressure_bar=1.0,
    )

    assert prep.interface_plan.interface_xy_nm == pytest.approx((4.2, 4.0))
    assert prep.interface_plan.electrolyte_target_box_nm[:2] == pytest.approx((4.2, 4.0))
    assert prep.electrolyte_prep.pack_plan.initial_pack_box_nm[:2] == pytest.approx((4.2, 4.0))
    assert prep.electrolyte_prep.direct_plan.species_names == ("EC", "DEC", "EMC", "Li", "PF6")
    assert any("polymer bulk first" in note for note in prep.notes)


def test_probe_polymer_matched_interface_preparation_reuses_interface_target_box():
    prep = plan_probe_polymer_matched_interface_preparation(
        reference_box_nm=(5.1, 5.1, 8.0),
        bottom_thickness_nm=3.2,
        top_thickness_nm=3.6,
        gap_nm=0.6,
        surface_shell_nm=0.8,
        target_density_g_cm3=1.15,
        solvent_mol_weights=(88.0, 118.0, 104.0),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        salt_mol_weights=(6.94, 144.96),
        salt_molarity_M=0.6,
        min_salt_pairs=6,
        solvent_species_names=("EC", "DEC", "EMC"),
        salt_species_names=("Li", "PF6"),
        min_solvent_counts=(1, 1, 1),
        probe_volume_scale=1.5,
        is_polyelectrolyte=True,
        minimum_margin_nm=0.8,
        fixed_xy_npt_ns=4.0,
    )

    assert prep.interface_plan.interface_xy_nm == pytest.approx((5.1, 5.1))
    assert prep.probe_prep.target_box_nm == pytest.approx(prep.interface_plan.electrolyte_target_box_nm)
    assert prep.probe_prep.probe_box_nm[0] >= prep.probe_prep.target_box_nm[0]
    assert any("isotropic electrolyte probe" in note for note in prep.notes)


def test_recommend_polymer_diffusion_interface_recipe_keeps_neutral_system_periodic_by_default():
    plan = plan_polymer_anchored_interface_preparation(
        reference_box_nm=(5.0, 5.2, 8.8),
        bottom_thickness_nm=3.4,
        top_thickness_nm=3.8,
        gap_nm=0.6,
        surface_shell_nm=0.8,
        is_polyelectrolyte=False,
    )

    recipe = recommend_polymer_diffusion_interface_recipe(
        interface_plan=plan,
        pressure_bar=1.0,
        prefer_vacuum=False,
        max_lateral_strain=0.07,
    )

    assert recipe.route_spec.route == "route_a"
    assert recipe.protocol.route == "route_a"
    assert recipe.route_spec.bottom.target_thickness_nm == pytest.approx(3.4)
    assert recipe.route_spec.top.target_thickness_nm == pytest.approx(3.8)
    assert recipe.route_spec.area_policy.reference_side == "bottom"
    assert recipe.route_spec.area_policy.max_lateral_strain == pytest.approx(0.07)
    assert recipe.protocol.pre_contact_dt_ps == pytest.approx(0.001)
    assert recipe.protocol.freeze_cores_pre_contact is True
    assert any("fully periodic diffusion interface" in note for note in recipe.notes)


def test_recommend_polymer_diffusion_interface_recipe_prefers_vacuum_for_polyelectrolyte():
    plan = plan_polymer_anchored_interface_preparation(
        reference_box_nm=(8.0, 8.0, 16.0),
        bottom_thickness_nm=5.0,
        top_thickness_nm=5.5,
        gap_nm=0.8,
        surface_shell_nm=1.0,
        is_polyelectrolyte=True,
        minimum_margin_nm=1.0,
        fixed_xy_npt_ns=4.0,
    )

    recipe = recommend_polymer_diffusion_interface_recipe(interface_plan=plan, pressure_bar=1.0)

    assert recipe.route_spec.route == "route_b"
    assert recipe.protocol.route == "route_b"
    assert recipe.route_spec.top.vacuum_nm >= 10.0
    assert recipe.protocol.exchange_ns == pytest.approx(4.0)
    assert recipe.protocol.production_ns == pytest.approx(8.0)
    assert recipe.protocol.use_region_thermostat_early is True
    assert any("explicit gap plus an external vacuum buffer" in note for note in recipe.notes)


def test_select_fragments_for_slab_prefers_denser_window():
    fragments = [
        FragmentRecord("A", 1, np.asarray([[0.0, 0.0, 1.0]], dtype=float), ["A"], [0.0], [10.0], 0),
        FragmentRecord("B", 1, np.asarray([[0.0, 0.0, 1.2]], dtype=float), ["B"], [0.0], [10.0], 1),
        FragmentRecord("C", 1, np.asarray([[0.0, 0.0, 3.0]], dtype=float), ["C"], [0.0], [1.0], 2),
    ]
    selected, zmin, zmax = _select_fragments_for_slab(
        fragments,
        axis="Z",
        box_nm=(2.0, 2.0, 4.0),
        target_thickness_nm=1.0,
        prefer_densest_window=True,
    )

    names = {frag.moltype for frag in selected}
    assert names == {"A", "B"}
    assert zmin < zmax


def test_resolve_lateral_sizing_plan_avoids_unnecessary_2x2_replication():
    plan = builder_mod._resolve_lateral_sizing_plan(
        bottom_box_nm=(5.98, 5.98, 8.0),
        top_box_nm=(6.159, 6.159, 8.0),
        axis="Z",
        policy=builder_mod.AreaMismatchPolicy(max_lateral_strain=0.03, prefer_larger_area=True),
        bottom_min_replicas_xy=(1, 1),
        top_min_replicas_xy=(1, 1),
    )

    assert plan.bottom_replicas_xy == (1, 1)
    assert plan.top_replicas_xy == (1, 1)
    assert 5.98 <= float(plan.target_lengths_nm[0]) <= 6.159
    assert 5.98 <= float(plan.target_lengths_nm[1]) <= 6.159


def test_split_fragments_from_frame_unwraps_bonded_chain_across_pbc():
    frame = builder_mod._GroFrame(
        title="wrapped",
        atoms=[
            builder_mod._GroAtom(resnr=1, resname="POL", atomname="C1", atomnr=1, xyz_nm=np.asarray([0.10, 0.20, 0.20], dtype=float)),
            builder_mod._GroAtom(resnr=1, resname="POL", atomname="C2", atomnr=2, xyz_nm=np.asarray([0.90, 0.20, 0.20], dtype=float)),
        ],
        box_nm=(1.0, 1.0, 1.0),
    )
    topo = SystemTopology(
        moleculetypes={
            "POL": MoleculeType(
                name="POL",
                atomtypes=["C", "C"],
                atomnames=["C1", "C2"],
                charges=[0.0, 0.0],
                masses=[12.011, 12.011],
                bonds=[(1, 2)],
            )
        },
        molecules=[("POL", 1)],
    )

    fragments = builder_mod._split_fragments_from_frame(frame, topo)

    assert len(fragments) == 1
    dx = abs(float(fragments[0].coords_nm[1, 0]) - float(fragments[0].coords_nm[0, 0]))
    assert dx < 0.25


def test_rebalance_fragment_selection_for_charge_adds_nearby_counterions():
    fragments = [
        FragmentRecord("CMC", 1, np.asarray([[0.0, 0.0, 1.00]], dtype=float), ["A"], [-1.0], [10.0], 0),
        FragmentRecord("CMC", 1, np.asarray([[0.0, 0.0, 1.15]], dtype=float), ["A"], [-1.0], [10.0], 1),
        FragmentRecord("Na", 1, np.asarray([[0.0, 0.0, 1.25]], dtype=float), ["Na"], [1.0], [23.0], 2),
        FragmentRecord("Na", 1, np.asarray([[0.0, 0.0, 1.35]], dtype=float), ["Na"], [1.0], [23.0], 3),
        FragmentRecord("Na", 1, np.asarray([[0.0, 0.0, 3.50]], dtype=float), ["Na"], [1.0], [23.0], 4),
    ]

    selected = fragments[:2]
    balanced, net_charge, notes = _rebalance_fragment_selection_for_charge(selected, all_fragments=fragments, axis="Z", charge_tol=0.1)

    assert abs(net_charge) < 1.0e-9
    assert len(balanced) == 4
    assert sum(1 for frag in balanced if frag.moltype == "Na") == 2
    assert any("charge-balanced slab selection" in note for note in notes)


def test_rebalance_fragment_selection_for_charge_uses_scaled_effective_charge():
    fragments = [
        FragmentRecord("CMC", 1, np.asarray([[0.0, 0.0, 1.00]], dtype=float), ["A"], [-0.25], [10.0], 0),
        FragmentRecord("CMC", 1, np.asarray([[0.0, 0.0, 1.10]], dtype=float), ["A"], [-0.25], [10.0], 1),
        FragmentRecord("Na", 1, np.asarray([[0.0, 0.0, 1.18]], dtype=float), ["Na"], [0.50], [23.0], 2),
        FragmentRecord("Na", 1, np.asarray([[0.0, 0.0, 2.50]], dtype=float), ["Na"], [0.50], [23.0], 3),
    ]

    balanced, net_charge, notes = _rebalance_fragment_selection_for_charge(
        fragments[:2],
        all_fragments=fragments,
        axis="Z",
        charge_tol=0.1,
        charge_scale=4.0,
    )

    assert abs(net_charge) < 1.0e-9
    assert len(balanced) == 3
    assert balanced[-1].source_instance == 2
    assert any("effective_net_charge_e=0.000000" in note for note in notes)


def test_rebalance_fragment_selection_for_charge_removes_excess_same_sign_fragments():
    fragments = [
        FragmentRecord("CMC", 1, np.asarray([[0.0, 0.0, 1.00]], dtype=float), ["A"], [-35.0], [100.0], 0),
        FragmentRecord("CMC", 1, np.asarray([[0.0, 0.0, 4.00]], dtype=float), ["A"], [-35.0], [100.0], 1),
    ]
    for idx in range(70):
        z = 1.10 + 0.01 * idx
        fragments.append(FragmentRecord("Na", 1, np.asarray([[0.0, 0.0, z]], dtype=float), ["Na"], [1.0], [23.0], idx + 2))

    selected = [fragments[0]] + fragments[2:53]
    balanced, net_charge, notes = _rebalance_fragment_selection_for_charge(
        selected,
        all_fragments=fragments,
        axis="Z",
        charge_tol=0.1,
        charge_scale=4.0,
    )

    assert abs(net_charge) < 1.0e-9
    assert sum(1 for frag in balanced if frag.moltype == "CMC") == 1
    assert sum(1 for frag in balanced if frag.moltype == "Na") == 35
    assert any("removed" in note for note in notes)


def test_read_gro_frame_recovers_coordinates_from_overflowed_atom_number_columns(tmp_path: Path):
    gro = tmp_path / "overflow.gro"
    gro.write_text(
        "overflow test\n"
        "    2\n"
        "    1SOL     C1100000   0.123   0.456   0.789\n"
        "    1SOL     H1100001   1.123   1.456   1.789\n"
        "   2.00000   2.00000   2.00000\n",
        encoding="utf-8",
    )

    frame = builder_mod._read_gro_frame(gro)

    assert len(frame.atoms) == 2
    assert frame.atoms[0].atomnr == 1
    assert frame.atoms[1].atomnr == 2
    assert np.allclose(frame.atoms[0].xyz_nm, np.asarray([0.123, 0.456, 0.789], dtype=float))
    assert np.allclose(frame.atoms[1].xyz_nm, np.asarray([1.123, 1.456, 1.789], dtype=float))


def test_write_gro_frame_wraps_large_indices_without_breaking_fixed_columns(tmp_path: Path):
    gro = tmp_path / "wrapped.gro"
    atoms = [
        builder_mod._GroAtom(resnr=100000, resname="SOL", atomname="C1", atomnr=1, xyz_nm=np.asarray([0.123, 0.456, 0.789], dtype=float)),
    ]

    builder_mod._write_gro_frame(gro, "wrapped", atoms, (2.0, 2.0, 2.0))

    lines = gro.read_text(encoding="utf-8").splitlines()
    assert len(lines[2]) == 44
    assert lines[2][:5] == "    0"
    assert lines[2][20:28].strip() == "0.123"
    assert lines[2][28:36].strip() == "0.456"
    assert lines[2][36:44].strip() == "0.789"


def test_plan_rescaled_bulk_counts_matches_target_footprint(tmp_path: Path):
    gro = tmp_path / "probe.gro"
    gro.write_text(
        "probe\n"
        "    1\n"
        "    1SOL     C1    1   0.100   0.200   0.300\n"
        "   4.00000   4.00000   8.00000\n",
        encoding="utf-8",
    )
    profile = build_bulk_equilibrium_profile(
        gro_path=gro,
        counts=[120, 80, 200, 24, 24],
        mol_weights=[88.0, 118.0, 104.0, 6.94, 144.96],
        species_names=["EC", "DEC", "EMC", "Li", "PF6"],
    )

    plan = plan_rescaled_bulk_counts(
        profile=profile,
        target_xy_nm=(2.0, 2.0),
        target_z_nm=6.0,
        min_counts=[1, 1, 1, 10, 10],
        tied_groups=[(3, 4)],
    )

    assert plan.target_box_nm == (2.0, 2.0, 6.0)
    assert plan.volume_scale < 1.0
    assert sum(plan.target_counts) < sum(profile.counts)
    assert plan.target_counts[3] == plan.target_counts[4]
    assert plan.target_counts[3] >= 10


def test_plan_resized_electrolyte_counts_preserves_solvent_group_and_salt_pairs(tmp_path: Path):
    gro = tmp_path / "probe_electrolyte.gro"
    gro.write_text(
        "probe\n"
        "    1\n"
        "    1SOL     C1    1   0.100   0.200   0.300\n"
        "   5.00000   5.00000   10.00000\n",
        encoding="utf-8",
    )
    profile = build_bulk_equilibrium_profile(
        gro_path=gro,
        counts=[300, 200, 500, 48, 48],
        mol_weights=[88.0, 118.0, 104.0, 6.94, 144.96],
        species_names=["EC", "DEC", "EMC", "Li", "PF6"],
    )

    plan = plan_resized_electrolyte_counts(
        profile=profile,
        target_xy_nm=(2.5, 2.5),
        target_z_nm=6.5,
        solvent_indices=(0, 1, 2),
        salt_pair_indices=(3, 4),
        min_solvent_counts=(1, 1, 1),
        min_salt_pairs=8,
    )

    assert plan.target_counts[3] == plan.target_counts[4]
    assert plan.target_counts[3] >= 8
    assert sum(plan.target_counts[:3]) < sum(profile.counts[:3])
    assert any("preserved solvent composition" in note for note in plan.notes)
    assert any("preserved salt-pair concentration" in note for note in plan.notes)


def test_plan_resized_electrolyte_counts_supports_grouped_solvents_and_multiple_salts(tmp_path: Path):
    gro = tmp_path / "probe_multi_electrolyte.gro"
    gro.write_text(
        "probe\n"
        "    1\n"
        "    1SOL     C1    1   0.100   0.200   0.300\n"
        "   6.00000   6.00000   8.00000\n",
        encoding="utf-8",
    )
    profile = build_bulk_equilibrium_profile(
        gro_path=gro,
        counts=[180, 120, 90, 60, 30, 30, 8, 8],
        mol_weights=[88.0, 104.0, 90.0, 74.0, 6.94, 144.96, 22.99, 147.0],
        species_names=["EC", "EMC", "DME", "DOL", "Li", "PF6", "Na", "FSI"],
    )

    target_scale = float((3.0 * 3.0 * 5.0) / (6.0 * 6.0 * 8.0))
    raw_group_1 = float((180 + 120) * target_scale)
    raw_group_2 = float((90 + 60) * target_scale)

    plan = plan_resized_electrolyte_counts(
        profile=profile,
        target_xy_nm=(3.0, 3.0),
        target_z_nm=5.0,
        solvent_indices=(0, 1, 2, 3),
        solvent_groups=((0, 1), (2, 3)),
        salt_pair_groups=((4, 5), (6, 7)),
        min_solvent_group_counts=((2, 1), (1, 1)),
        min_salt_pairs=(6, 3),
    )

    assert plan.target_counts[0] + plan.target_counts[1] == int(round(raw_group_1))
    assert plan.target_counts[2] + plan.target_counts[3] == int(round(raw_group_2))
    assert plan.target_counts[4] == plan.target_counts[5]
    assert plan.target_counts[4] >= 6
    assert plan.target_counts[6] == plan.target_counts[7] == 3
    assert any("grouped solvent composition" in note for note in plan.notes)
    assert any("coupled index sets" in note for note in plan.notes)


def test_plan_resized_electrolyte_counts_locks_target_xy_to_interface_footprint(tmp_path: Path):
    gro = tmp_path / "probe_resize_xy_lock.gro"
    gro.write_text(
        "probe\n"
        "    1\n"
        "    1SOL     C1    1   0.100   0.200   0.300\n"
        "   4.00000   5.00000  12.00000\n",
        encoding="utf-8",
    )
    profile = build_bulk_equilibrium_profile(
        gro_path=gro,
        counts=[240, 160, 400, 36, 36],
        mol_weights=[88.0, 118.0, 104.0, 6.94, 144.96],
        species_names=["EC", "DEC", "EMC", "Li", "PF6"],
    )

    plan = plan_resized_electrolyte_counts(
        profile=profile,
        target_xy_nm=(3.2, 3.8),
        target_z_nm=7.5,
        solvent_indices=(0, 1, 2),
        salt_pair_indices=(3, 4),
        min_solvent_counts=(1, 1, 1),
        min_salt_pairs=6,
    )

    assert plan.probe_box_nm == (4.0, 5.0, 12.0)
    assert plan.target_box_nm == (3.2, 3.8, 7.5)
    assert abs(plan.volume_scale - ((3.2 * 3.8 * 7.5) / (4.0 * 5.0 * 12.0))) < 1.0e-12
    assert plan.target_counts[3] == plan.target_counts[4]
    assert plan.target_counts[3] >= 6
    assert sum(plan.target_counts[:3]) < sum(profile.counts[:3])
    assert any("probe-equilibrated box and composition" in note for note in plan.notes)


def test_plan_fixed_xy_direct_pack_box_expands_z_when_pack_density_is_lower():
    pack = plan_fixed_xy_direct_pack_box(
        reference_box_nm=(3.2, 3.8, 7.5),
        target_counts=(240, 160, 400, 36, 36),
        mol_weights=(88.0, 118.0, 104.0, 6.94, 144.96),
        species_names=("EC", "DEC", "EMC", "Li", "PF6"),
        initial_pack_density_g_cm3=0.85,
        z_padding_factor=1.05,
    )

    assert pack.fixed_xy_nm == (3.2, 3.8)
    assert pack.initial_pack_box_nm[0] == 3.2
    assert pack.initial_pack_box_nm[1] == 3.8
    assert pack.initial_pack_box_nm[2] > 7.5
    assert pack.estimated_initial_density_g_cm3 < 1.1
    assert any("selected initial pack Z" in note for note in pack.notes)


def test_plan_fixed_xy_direct_electrolyte_preparation_bundles_counts_pack_and_relaxation():
    prep = plan_fixed_xy_direct_electrolyte_preparation(
        reference_box_nm=(3.2, 3.8, 7.5),
        target_density_g_cm3=1.10,
        solvent_mol_weights=(88.0, 118.0, 104.0),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        salt_mol_weights=(6.94, 144.96),
        salt_molarity_M=1.0,
        min_salt_pairs=12,
        solvent_species_names=("EC", "DEC", "EMC"),
        salt_species_names=("Li", "PF6"),
        min_solvent_counts=(1, 1, 1),
        initial_pack_density_g_cm3=0.85,
        pressure_bar=1.0,
    )

    assert prep.reference_box_nm == (3.2, 3.8, 7.5)
    assert prep.direct_plan.target_box_nm == (3.2, 3.8, 7.5)
    assert prep.pack_plan.initial_pack_box_nm[0] == 3.2
    assert prep.pack_plan.initial_pack_box_nm[1] == 3.8
    assert prep.pack_plan.initial_pack_box_nm[2] >= 7.5
    assert prep.relax_mdp_overrides["pcoupltype"] == "semiisotropic"
    assert any("XY-locked initial pack box" in note for note in prep.notes)


def test_plan_fixed_xy_direct_electrolyte_preparation_supports_compact_target_box():
    prep = plan_fixed_xy_direct_electrolyte_preparation(
        reference_box_nm=(4.8, 5.1, 12.0),
        target_box_nm=(4.8, 5.1, 5.6),
        target_density_g_cm3=1.10,
        solvent_mol_weights=(88.0, 118.0, 104.0),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        salt_mol_weights=(6.94, 144.96),
        salt_molarity_M=1.0,
        min_salt_pairs=8,
        solvent_species_names=("EC", "DEC", "EMC"),
        salt_species_names=("Li", "PF6"),
        min_solvent_counts=(1, 1, 1),
        initial_pack_density_g_cm3=0.85,
        z_padding_factor=1.15,
        minimum_pack_z_factor=1.25,
        pressure_bar=1.0,
    )

    assert prep.reference_box_nm == (4.8, 5.1, 12.0)
    assert prep.direct_plan.target_box_nm == (4.8, 5.1, 5.6)
    assert prep.pack_plan.initial_pack_box_nm[0] == 4.8
    assert prep.pack_plan.initial_pack_box_nm[1] == 5.1
    assert prep.pack_plan.initial_pack_box_nm[2] >= 7.0
    assert any("at least 7.0000 nm" in note for note in prep.notes)


def test_plan_probe_electrolyte_preparation_uses_isotropic_probe_box():
    prep = plan_probe_electrolyte_preparation(
        reference_box_nm=(5.2, 5.2, 11.0),
        target_box_nm=(5.2, 5.2, 4.5),
        target_density_g_cm3=0.95,
        solvent_mol_weights=(88.0, 118.0, 104.0),
        solvent_mass_ratio=(3.0, 2.0, 5.0),
        salt_mol_weights=(6.94, 144.96),
        salt_molarity_M=0.6,
        min_salt_pairs=8,
        solvent_species_names=("EC", "DEC", "EMC"),
        salt_species_names=("Li", "PF6"),
        min_solvent_counts=(1, 1, 1),
        probe_volume_scale=1.8,
        initial_pack_density_g_cm3=0.60,
    )

    assert prep.reference_box_nm == (5.2, 5.2, 11.0)
    assert prep.target_box_nm == (5.2, 5.2, 4.5)
    assert prep.build_density_g_cm3 == pytest.approx(0.60)
    assert prep.probe_box_nm[0] == pytest.approx(prep.probe_box_nm[1])
    assert prep.probe_box_nm[1] == pytest.approx(prep.probe_box_nm[2])
    assert prep.pack_plan.initial_pack_box_nm[0] == pytest.approx(prep.probe_box_nm[0])
    assert prep.pack_plan.initial_pack_box_nm[1] == pytest.approx(prep.probe_box_nm[1])
    assert any("density-driven isotropic pack" in note for note in prep.notes)


def test_plan_resized_electrolyte_preparation_from_probe_bundles_resize_and_fixed_xy_pack(tmp_path: Path):
    gro = tmp_path / "probe_resize_bundle.gro"
    gro.write_text(
        "probe\n"
        "    1\n"
        "    1SOL     C1    1   0.100   0.200   0.300\n"
        "   5.50000   5.50000   5.50000\n",
        encoding="utf-8",
    )
    probe_dir = tmp_path / "probe_bulk" / "03_EQ21" / "03_EQ21" / "step_21"
    probe_dir.mkdir(parents=True)
    (probe_dir / "md.gro").write_text(gro.read_text(encoding="utf-8"), encoding="utf-8")

    prep = plan_resized_electrolyte_preparation_from_probe(
        reference_box_nm=(5.2, 5.2, 11.0),
        target_box_nm=(5.2, 5.2, 4.5),
        probe_work_dir=tmp_path / "probe_bulk",
        probe_counts=(60, 40, 100, 12, 12),
        mol_weights=(88.0, 118.0, 104.0, 6.94, 144.96),
        species_names=("EC", "DEC", "EMC", "Li", "PF6"),
        solvent_indices=(0, 1, 2),
        salt_pair_indices=(3, 4),
        min_solvent_counts=(1, 1, 1),
        min_salt_pairs=8,
        initial_pack_density_g_cm3=0.55,
        minimum_pack_z_factor=1.8,
        pressure_bar=1.0,
    )

    assert prep.profile.box_nm == (5.5, 5.5, 5.5)
    assert prep.resize_plan.target_box_nm == (5.2, 5.2, 4.5)
    assert prep.resize_plan.target_counts[3] == prep.resize_plan.target_counts[4]
    assert prep.resize_plan.target_counts[3] >= 8
    assert prep.pack_plan.initial_pack_box_nm[:2] == (5.2, 5.2)
    assert prep.pack_plan.initial_pack_box_nm[2] >= 8.1
    assert prep.relax_mdp_overrides["pcoupltype"] == "semiisotropic"
    assert any("equilibrated probe bulk" in note for note in prep.notes)


def test_plan_fixed_xy_direct_pack_box_can_cap_excessive_z():
    pack = plan_fixed_xy_direct_pack_box(
        reference_box_nm=(5.2, 5.2, 5.6),
        target_counts=(298, 149, 420, 103, 103),
        mol_weights=(88.0, 118.0, 104.0, 6.94, 144.96),
        species_names=("EC", "DEC", "EMC", "Li", "PF6"),
        initial_pack_density_g_cm3=0.16,
        z_padding_factor=1.45,
        minimum_z_nm=10.0,
        maximum_z_nm=14.0,
    )

    assert pack.initial_pack_box_nm == (5.2, 5.2, 14.0)
    assert any("capped initial pack Z" in note for note in pack.notes)


def test_plan_resized_electrolyte_preparation_from_probe_caps_dilute_fixed_xy_box(tmp_path: Path):
    gro = tmp_path / "probe_resize_cap.gro"
    gro.write_text(
        "probe\n"
        "    1\n"
        "    1SOL     C1    1   0.100   0.200   0.300\n"
        "   5.81848   5.81848   5.81848\n",
        encoding="utf-8",
    )
    probe_dir = tmp_path / "probe_bulk" / "03_EQ21" / "03_EQ21" / "step_21"
    probe_dir.mkdir(parents=True)
    (probe_dir / "md.gro").write_text(gro.read_text(encoding="utf-8"), encoding="utf-8")

    prep = plan_resized_electrolyte_preparation_from_probe(
        reference_box_nm=(5.13392, 5.13392, 5.13392),
        target_box_nm=(5.13392, 5.13392, 5.6),
        probe_work_dir=tmp_path / "probe_bulk",
        probe_counts=(398, 198, 561, 138, 138),
        mol_weights=(88.0, 118.0, 104.0, 6.94, 144.96),
        species_names=("EC", "DEC", "EMC", "Li", "PF6"),
        solvent_indices=(0, 1, 2),
        salt_pair_indices=(3, 4),
        min_solvent_counts=(1, 1, 1),
        min_salt_pairs=2,
        initial_pack_density_g_cm3=0.16,
        z_padding_factor=1.45,
        minimum_pack_z_factor=1.8,
        maximum_pack_z_factor=2.5,
        pressure_bar=1.0,
    )

    assert prep.pack_plan.initial_pack_box_nm[:2] == (5.13392, 5.13392)
    assert prep.pack_plan.initial_pack_box_nm[2] == pytest.approx(14.0)
    assert any("maximum Z of 14.0000 nm" in note for note in prep.notes)


def test_plan_resized_electrolyte_preparation_from_probe_auto_selects_compact_pack_defaults(tmp_path: Path):
    gro = tmp_path / "probe_resize_auto_defaults.gro"
    gro.write_text(
        "probe\n"
        "    1\n"
        "    1SOL     C1    1   0.100   0.200   0.300\n"
        "   5.98053   5.98053   5.98053\n",
        encoding="utf-8",
    )
    probe_dir = tmp_path / "probe_bulk" / "03_EQ21" / "03_EQ21" / "step_21"
    probe_dir.mkdir(parents=True)
    (probe_dir / "md.gro").write_text(gro.read_text(encoding="utf-8"), encoding="utf-8")

    prep = plan_resized_electrolyte_preparation_from_probe(
        reference_box_nm=(5.08879, 5.08879, 5.08879),
        target_box_nm=(5.08879, 5.08879, 5.6),
        probe_work_dir=tmp_path / "probe_bulk",
        probe_counts=(391, 194, 552, 135, 135),
        mol_weights=(88.0, 118.0, 104.0, 6.94, 144.96),
        species_names=("EC", "DEC", "EMC", "Li", "PF6"),
        solvent_indices=(0, 1, 2),
        salt_pair_indices=(3, 4),
        min_solvent_counts=(1, 1, 1),
        min_salt_pairs=2,
        pressure_bar=1.0,
    )

    assert prep.resize_plan.target_density_g_cm3 == pytest.approx(1.05, rel=0.1)
    assert prep.pack_plan.initial_pack_density_g_cm3 > 0.65
    assert prep.pack_plan.initial_pack_box_nm[2] < 9.0
    assert any("auto-selected the fixed-XY initial pack density" in note for note in prep.notes)


def test_plan_polymer_anchored_interface_preparation_uses_polymer_xy_for_both_sides():
    plan = plan_polymer_anchored_interface_preparation(
        reference_box_nm=(4.2, 5.3, 10.5),
        bottom_thickness_nm=3.6,
        top_thickness_nm=4.2,
        gap_nm=0.6,
        surface_shell_nm=0.8,
        is_polyelectrolyte=True,
        minimum_margin_nm=0.8,
        fixed_xy_npt_ns=1.5,
    )

    assert plan.interface_xy_nm == (4.2, 5.3)
    assert plan.bottom_thickness_nm == pytest.approx(3.6)
    assert plan.top_thickness_nm == pytest.approx(4.2)
    assert plan.gap_nm == pytest.approx(0.6)
    assert plan.surface_shell_nm == pytest.approx(0.8)
    assert plan.is_polyelectrolyte is True
    assert plan.polymer_target_box_nm[:2] == (4.2, 5.3)
    assert plan.electrolyte_target_box_nm[:2] == (4.2, 5.3)
    assert plan.electrolyte_target_box_nm[2] > 4.2
    assert any("polymer bulk XY lengths" in note for note in plan.notes)


def test_interface_builder_area_policy_can_anchor_to_bottom_lengths(tmp_path: Path):
    bulk_bottom = _make_bulk_fixture(tmp_path / "ac_poly", lx=1.00, ly=1.02, lz=1.6)
    bulk_top = _make_bulk_fixture(tmp_path / "ac_electrolyte", lx=0.99, ly=1.00, lz=1.6)

    builder = InterfaceBuilder(work_dir=tmp_path / "interface_anchor_bottom", restart=True)
    built = builder.build_from_bulk_workdirs(
        name="route_a_case",
        bottom_name="ac_poly",
        bottom_work_dir=bulk_bottom,
        top_name="ac_electrolyte",
        top_work_dir=bulk_top,
        route=InterfaceRouteSpec.route_a(
            bottom_thickness_nm=0.8,
            top_thickness_nm=0.8,
            gap_nm=0.4,
            area_policy=AreaMismatchPolicy(reference_side="bottom", max_lateral_strain=0.05),
        ),
    )

    assert abs(built.bottom_slab.box_nm[0] - 1.0) < 1.0e-12
    assert abs(built.bottom_slab.box_nm[1] - 1.02) < 1.0e-12
    assert abs(built.top_slab.box_nm[0] - 1.0) < 1.0e-12
    assert abs(built.top_slab.box_nm[1] - 1.02) < 1.0e-12


def test_make_orthorhombic_pack_cell_sets_expected_cell_lengths():
    cell = make_orthorhombic_pack_cell((3.2, 3.8, 7.5))

    assert hasattr(cell, 'cell')
    assert cell.cell.xhi == 32.0
    assert cell.cell.yhi == 38.0
    assert cell.cell.zhi == 75.0


def test_wrap_gro_atoms_into_primary_box_wraps_lateral_coordinates(tmp_path: Path):
    gro = tmp_path / "system.gro"
    gro.write_text(
        "\n".join(
            [
                "wrapped",
                "    2",
                "    1SOL     C1    1  -0.200   1.100   0.300",
                "    1SOL     H1    2   1.250  -0.150   0.450",
                "   1.00000   1.00000   2.00000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = builder_mod._wrap_gro_atoms_into_primary_box(gro, dims=(0, 1))
    frame = builder_mod._read_gro_frame(gro)

    assert result["applied"] is True
    assert result["wrapped_components"] == 4
    assert frame.atoms[0].xyz_nm[2] == pytest.approx(0.300)
    assert frame.atoms[1].xyz_nm[2] == pytest.approx(0.450)
    for atom in frame.atoms:
        assert 0.0 <= float(atom.xyz_nm[0]) < 1.0
        assert 0.0 <= float(atom.xyz_nm[1]) < 1.0


def test_amorphous_pack_order_prefers_larger_species_first():
    small = Chem.MolFromSmiles('C')
    large = Chem.MolFromSmiles('CCCCCC')
    assert small is not None
    assert large is not None

    order = poly._amorphous_pack_order([small, large], [10, 1])

    assert order == (1, 0)


def test_recommend_electrolyte_alignment_uses_route_geometry_and_system_type():
    neutral = recommend_electrolyte_alignment(
        top_thickness_nm=5.0,
        gap_nm=0.6,
        surface_shell_nm=0.8,
        is_polyelectrolyte=False,
    )
    cmc = recommend_electrolyte_alignment(
        top_thickness_nm=5.5,
        gap_nm=0.6,
        surface_shell_nm=0.8,
        is_polyelectrolyte=True,
    )

    assert abs(neutral.target_z_margin_nm - 1.1) < 1.0e-9
    assert abs(neutral.target_z_nm - 6.1) < 1.0e-9
    assert abs(neutral.fixed_xy_npt_ns - 1.5) < 1.0e-9
    assert abs(cmc.target_z_margin_nm - 1.1) < 1.0e-9
    assert abs(cmc.target_z_nm - 6.6) < 1.0e-9
    assert abs(cmc.fixed_xy_npt_ns - 2.5) < 1.0e-9
    assert any("neutral polymer" in note for note in neutral.notes)
    assert any("polyelectrolyte" in note for note in cmc.notes)


def test_plan_direct_electrolyte_counts_targets_box_without_probe_workflow():
    plan = plan_direct_electrolyte_counts(
        target_box_nm=(4.0, 4.0, 5.0),
        target_density_g_cm3=0.30,
        solvent_mol_weights=[88.0, 118.0, 104.0],
        solvent_mass_ratio=[3.0, 2.0, 5.0],
        salt_mol_weights=[6.94, 144.96],
        salt_molarity_M=1.0,
        min_salt_pairs=12,
        solvent_species_names=["EC", "DEC", "EMC"],
        salt_species_names=["Li", "PF6"],
        min_solvent_counts=(1, 1, 1),
    )

    assert plan.target_box_nm == (4.0, 4.0, 5.0)
    assert plan.species_names == ("EC", "DEC", "EMC", "Li", "PF6")
    assert plan.target_counts[3] == plan.target_counts[4]
    assert plan.salt_pair_count == plan.target_counts[3]
    assert plan.salt_pair_count >= 12
    assert all(count >= 1 for count in plan.solvent_counts)
    assert abs(plan.estimated_density_g_cm3 - 0.30) < 0.02
    assert any("directly planned electrolyte counts" in note for note in plan.notes)


def test_fixed_xy_semiisotropic_npt_overrides_emit_gromacs_ready_values():
    overrides = fixed_xy_semiisotropic_npt_overrides(pressure_bar=1.0)

    assert overrides["pcoupltype"] == "semiisotropic"
    assert overrides["ref_p"] == "1 1"
    assert overrides["compressibility"] == "0 4.5e-05"


def test_find_latest_equilibrated_gro_prefers_npt_production(tmp_path: Path):
    prod = tmp_path / "05_npt_production" / "01_npt"
    prod.mkdir(parents=True, exist_ok=True)
    (prod / "md.gro").write_text("prod\n", encoding="utf-8")
    eq21 = tmp_path / "03_EQ21" / "03_EQ21" / "step_21"
    eq21.mkdir(parents=True, exist_ok=True)
    (eq21 / "md.gro").write_text("eq21\n", encoding="utf-8")

    latest = eqmod._find_latest_equilibrated_gro(tmp_path)

    assert latest == prod / "md.gro"


def test_find_latest_equilibrated_gro_can_exclude_current_production_dir(tmp_path: Path):
    prod = tmp_path / "05_npt_production" / "01_npt"
    prod.mkdir(parents=True, exist_ok=True)
    (prod / "md.gro").write_text("prod\n", encoding="utf-8")
    eq21 = tmp_path / "03_EQ21" / "03_EQ21" / "step_21"
    eq21.mkdir(parents=True, exist_ok=True)
    (eq21 / "md.gro").write_text("eq21\n", encoding="utf-8")

    latest = eqmod._find_latest_equilibrated_gro(tmp_path, exclude_dirs=[tmp_path / "05_npt_production"])

    assert latest == eq21 / "md.gro"


def test_npt_exec_applies_mdp_overrides(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["out_dir"] = out_dir
            captured["provenance_ndx"] = provenance_ndx

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "01_npt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NPT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    npt = eqmod.NPT(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(npt._resume, "run", lambda spec, fn: fn())

    npt.exec(
        temp=300.0,
        press=1.0,
        mpi=1,
        omp=1,
        gpu=0,
        time=0.1,
        mdp_overrides=fixed_xy_semiisotropic_npt_overrides(pressure_bar=1.0),
    )

    mdp_text = captured["stages"][0].mdp.render()
    assert captured["provenance_ndx"] == exp.system_ndx
    assert "pcoupltype                = semiisotropic" in mdp_text
    assert "ref_p                     = 1 1" in mdp_text
    assert "compressibility           = 0 4.5e-05" in mdp_text


def test_eq21_exec_applies_npt_stage_overrides(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["out_dir"] = out_dir
            captured["provenance_ndx"] = provenance_ndx

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "step_21"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.EQ21step, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)
    monkeypatch.setattr(eqmod, "_write_eq21_schedule", lambda *args, **kwargs: None)
    monkeypatch.setattr(eqmod, "_print_eq21_schedule", lambda *args, **kwargs: None)
    monkeypatch.setattr(eqmod, "_write_eq21_overview_plot", lambda *args, **kwargs: None)

    eq21 = eqmod.EQ21step(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(eq21._resume, "run", lambda spec, fn: fn())

    eq21.exec(
        temp=300.0,
        press=1.0,
        mpi=1,
        omp=1,
        gpu=0,
        time=0.1,
        eq21_npt_mdp_overrides=fixed_xy_semiisotropic_npt_overrides(pressure_bar=1.0),
    )

    npt_like = [stage for stage in captured["stages"] if stage.kind == "npt"]
    assert npt_like
    assert captured["provenance_ndx"] == exp.system_ndx
    assert all("pcoupltype                = semiisotropic" in stage.mdp.render() for stage in npt_like)
    assert all("ref_p                     = 1 1" in stage.mdp.render() for stage in npt_like)


def test_eq21_exec_invalidates_downstream_resume_steps_when_rebuilding(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "step_21"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.EQ21step, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)
    monkeypatch.setattr(eqmod, "_write_eq21_schedule", lambda *args, **kwargs: None)
    monkeypatch.setattr(eqmod, "_print_eq21_schedule", lambda *args, **kwargs: None)
    monkeypatch.setattr(eqmod, "_write_eq21_overview_plot", lambda *args, **kwargs: None)

    eq21 = eqmod.EQ21step(ac=object(), work_dir=tmp_path)
    invalidations: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    monkeypatch.setattr(
        eq21._resume,
        "invalidate_steps",
        lambda *, names=(), prefixes=(): invalidations.append((tuple(names), tuple(prefixes))) or [],
    )
    monkeypatch.setattr(eq21._resume, "run", lambda spec, fn: fn())

    eq21.exec(temp=300.0, press=1.0, mpi=1, omp=1, gpu=0, time=0.1)

    assert invalidations == [(("npt_production", "nvt_production"), ("equilibration_additional_",))]


def test_additional_exec_applies_mdp_overrides_to_relaxation_stages(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["out_dir"] = out_dir
            captured["provenance_ndx"] = provenance_ndx

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / captured["stages"][-1].name
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.Additional, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    add = eqmod.Additional(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(add._resume, "run", lambda spec, fn: fn())

    add.exec(
        temp=300.0,
        press=1.0,
        mpi=1,
        omp=1,
        gpu=0,
        time=0.1,
        mdp_overrides=fixed_xy_semiisotropic_npt_overrides(pressure_bar=1.0),
    )

    npt_like = [stage for stage in captured["stages"] if stage.kind in ("npt", "md")]
    assert npt_like
    assert captured["provenance_ndx"] == exp.system_ndx
    assert any("pcoupltype                = semiisotropic" in stage.mdp.render() for stage in npt_like)


def test_npt_exec_invalidates_nvt_resume_state_when_rebuilding(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "01_npt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NPT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    npt = eqmod.NPT(ac=object(), work_dir=tmp_path)
    invalidations: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    monkeypatch.setattr(
        npt._resume,
        "invalidate_steps",
        lambda *, names=(), prefixes=(): invalidations.append((tuple(names), tuple(prefixes))) or [],
    )
    monkeypatch.setattr(npt._resume, "run", lambda spec, fn: fn())

    npt.exec(temp=300.0, press=1.0, mpi=1, omp=1, gpu=0, time=0.1)

    assert invalidations == [(("nvt_production",), ())]


def test_npt_exec_sets_lean_production_output_cadence_and_checkpoints(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "01_npt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NPT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    npt = eqmod.NPT(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(npt._resume, "run", lambda spec, fn: fn())

    npt.exec(
        temp=300.0,
        press=1.0,
        mpi=1,
        omp=1,
        gpu=0,
        time=0.1,
        traj_ps=2.0,
        energy_ps=4.0,
        log_ps=6.0,
        trr_ps=None,
        velocity_ps=None,
        checkpoint_min=3.0,
    )

    stage = captured["stages"][0]
    mdp_text = stage.mdp.render()
    assert "dt                       = 0.001" in mdp_text
    assert "constraints              = none" in mdp_text
    assert "nstxout-compressed       = 2000" in mdp_text
    assert "nstenergy                = 4000" in mdp_text
    assert "nstlog                   = 6000" in mdp_text
    assert "nstxout                  = 0" in mdp_text
    assert "nstvout                  = 0" in mdp_text
    assert stage.lincs_retry is None
    assert stage.checkpoint_minutes == pytest.approx(3.0)


def test_nvt_exec_sets_lean_production_output_cadence_and_checkpoints(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "01_nvt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NVT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    nvt = eqmod.NVT(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(nvt._resume, "run", lambda spec, fn: fn())

    nvt.exec(
        temp=300.0,
        mpi=1,
        omp=1,
        gpu=0,
        time=0.1,
        traj_ps=1.0,
        energy_ps=2.0,
        log_ps=None,
        trr_ps=5.0,
        velocity_ps=5.0,
        checkpoint_min=4.0,
    )

    stage = captured["stages"][0]
    mdp_text = stage.mdp.render()
    assert "dt                       = 0.001" in mdp_text
    assert "constraints              = none" in mdp_text
    assert "nstxout-compressed       = 1000" in mdp_text
    assert "nstenergy                = 2000" in mdp_text
    assert "nstlog                   = 2000" in mdp_text
    assert "nstxout                  = 5000" in mdp_text
    assert "nstvout                  = 5000" in mdp_text
    assert stage.lincs_retry is None
    assert stage.checkpoint_minutes == pytest.approx(4.0)


def test_npt_exec_default_production_uses_no_constraints_and_1fs(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "01_npt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NPT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    npt = eqmod.NPT(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(npt._resume, "run", lambda spec, fn: fn())

    npt.exec(temp=300.0, press=1.0, mpi=1, omp=1, gpu=0, time=0.1)

    stage = captured["stages"][0]
    assert stage.mdp.params["dt"] == pytest.approx(0.001)
    assert stage.mdp.params["constraints"] == "none"
    assert stage.lincs_retry is None


def test_npt_exec_explicit_constraints_restore_lincs_path(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "01_npt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NPT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    npt = eqmod.NPT(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(npt._resume, "run", lambda spec, fn: fn())

    npt.exec(
        temp=300.0,
        press=1.0,
        mpi=1,
        omp=1,
        gpu=0,
        time=0.1,
        constraints="h-bonds",
        lincs_iter=5,
        lincs_order=10,
    )

    stage = captured["stages"][0]
    mdp_text = stage.mdp.render()
    assert "constraints              = h-bonds" in mdp_text
    assert "constraint_algorithm     = lincs" in mdp_text
    assert "lincs_iter               = 5" in mdp_text
    assert "lincs_order              = 10" in mdp_text
    assert stage.lincs_retry is not None


def test_nvt_exec_default_production_uses_no_constraints_and_1fs(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "01_nvt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NVT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    nvt = eqmod.NVT(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(nvt._resume, "run", lambda spec, fn: fn())

    nvt.exec(temp=300.0, mpi=1, omp=1, gpu=0, time=0.1)

    stage = captured["stages"][0]
    assert stage.mdp.params["dt"] == pytest.approx(0.001)
    assert stage.mdp.params["constraints"] == "none"
    assert stage.lincs_retry is None


def test_nvt_exec_applies_mdp_overrides_and_constraint_selection(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "01_nvt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NVT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    nvt = eqmod.NVT(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(nvt._resume, "run", lambda spec, fn: fn())

    nvt.exec(
        temp=300.0,
        mpi=1,
        omp=1,
        gpu=0,
        time=0.1,
        constraints="all-bonds",
        lincs_iter=6,
        lincs_order=9,
        mdp_overrides={"dt": 0.0015},
    )

    stage = captured["stages"][0]
    mdp_text = stage.mdp.render()
    assert "dt                       = 0.0015" in mdp_text
    assert "constraints              = all-bonds" in mdp_text
    assert "constraint_algorithm     = lincs" in mdp_text
    assert "lincs_iter               = 6" in mdp_text
    assert "lincs_order              = 9" in mdp_text
    assert stage.lincs_retry is not None


def test_npt_exec_supports_bridge_and_conservative_gpu_mode(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["resources"] = resources
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "02_npt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NPT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    npt = eqmod.NPT(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(npt._resume, "run", lambda spec, fn: fn())

    npt.exec(
        temp=300.0,
        press=1.0,
        mpi=1,
        omp=1,
        gpu=1,
        gpu_id=0,
        time=0.1,
        checkpoint_min=2.0,
        gpu_offload_mode="conservative",
        bridge_ps=20.0,
        bridge_dt_fs=1.0,
        bridge_lincs_iter=5,
        bridge_lincs_order=10,
    )

    stages = captured["stages"]
    resources = captured["resources"]
    assert len(stages) == 2
    assert stages[0].name == "01_bridge_npt"
    assert stages[1].name == "02_npt"
    assert stages[0].mdp.params["dt"] == pytest.approx(0.001)
    assert stages[0].mdp.params["constraints"] == "none"
    assert resources.gpu_offload_mode == "conservative"


def test_nvt_exec_supports_bridge_and_density_control_toggle(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["resources"] = resources
            captured["gro"] = gro
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "02_nvt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NVT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    nvt = eqmod.NVT(ac=object(), work_dir=tmp_path)
    monkeypatch.setattr(nvt._resume, "run", lambda spec, fn: fn())

    nvt.exec(
        temp=300.0,
        mpi=1,
        omp=1,
        gpu=1,
        gpu_id=0,
        time=0.1,
        checkpoint_min=2.0,
        gpu_offload_mode="conservative",
        bridge_ps=10.0,
        density_control=False,
    )

    stages = captured["stages"]
    resources = captured["resources"]
    assert len(stages) == 2
    assert stages[0].name == "01_bridge_nvt"
    assert stages[1].name == "02_nvt"
    assert resources.gpu_offload_mode == "conservative"
    assert captured["gro"] == exp.system_gro


def _polymer_like_ac():
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None
    mol.SetProp(
        "_yadonpy_cell_meta",
        json.dumps(
            {
                "species": [
                    {
                        "natoms": 120,
                        "polyelectrolyte_mode": True,
                        "residue_map": {
                            "residues": [
                                {"residue_number": 1, "residue_name": "RU0", "atom_indices": [0]},
                                {"residue_number": 2, "residue_name": "RU1", "atom_indices": [1]},
                            ]
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    return mol


def test_npt_exec_auto_uses_conservative_gpu_and_bridge_for_polymer_system(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["resources"] = resources
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "02_npt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NPT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    npt = eqmod.NPT(ac=_polymer_like_ac(), work_dir=tmp_path)
    monkeypatch.setattr(npt._resume, "run", lambda spec, fn: fn())

    npt.exec(temp=300.0, press=1.0, mpi=1, omp=1, gpu=1, gpu_id=0, time=0.1)

    stages = captured["stages"]
    resources = captured["resources"]
    assert len(stages) == 2
    assert stages[0].name == "01_bridge_npt"
    assert stages[0].mdp.params["dt"] == pytest.approx(0.001)
    assert resources.gpu_offload_mode == "conservative"


def test_nvt_exec_keeps_density_scaling_enabled_by_default_for_polymer_system(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )
    captured: dict[str, object] = {}

    class _FakeJob:
        default_stages = staticmethod(eqmod.EquilibrationJob.default_stages)

        def __init__(self, *, gro, top, ndx=None, provenance_ndx=None, out_dir, stages, resources):
            captured["stages"] = stages
            captured["resources"] = resources
            captured["gro"] = gro
            captured["out_dir"] = out_dir

        def run(self, *, restart=False):
            out_dir = Path(captured["out_dir"]) / "02_nvt"
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ("md.tpr", "md.xtc", "md.edr", "md.gro"):
                (out_dir / suffix).write_text("x\n", encoding="utf-8")
            return out_dir / "md.gro"

    monkeypatch.setattr(eqmod.NVT, "_ensure_system_exported", lambda self: exp)
    monkeypatch.setattr(eqmod, "_find_latest_equilibrated_gro", lambda work_dir, exclude_dirs=None: None)
    monkeypatch.setattr(eqmod, "EquilibrationJob", _FakeJob)

    nvt = eqmod.NVT(ac=_polymer_like_ac(), work_dir=tmp_path)
    monkeypatch.setattr(nvt._resume, "run", lambda spec, fn: fn())

    nvt.exec(temp=300.0, mpi=1, omp=1, gpu=1, gpu_id=0, time=0.1)

    stages = captured["stages"]
    resources = captured["resources"]
    assert len(stages) == 2
    assert stages[0].name == "01_bridge_nvt"
    assert resources.gpu_offload_mode == "conservative"
    assert captured["gro"] == exp.system_gro
    assert eqmod._resolve_nvt_density_control(_polymer_like_ac(), None) is True


def test_equilibrate_bulk_with_eq21_helper_runs_eq_additional_and_optional_npt(tmp_path: Path, monkeypatch):
    system_dir = tmp_path / "02_system"
    raw_dir = system_dir / "01_raw_non_scaled"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("system.gro", "system.top", "system.ndx", "system_meta.json"):
        (system_dir / name).write_text("x\n", encoding="utf-8")
        (raw_dir / name).write_text("x\n", encoding="utf-8")
    exp = SystemExportResult(
        system_gro=system_dir / "system.gro",
        system_top=system_dir / "system.top",
        system_ndx=system_dir / "system.ndx",
        molecules_dir=system_dir / "molecules",
        system_meta=system_dir / "system_meta.json",
        box_nm=1.0,
        species=[],
    )

    calls: dict[str, object] = {"eq21": None, "additional": None, "npt": None}

    class _AnalyzeFalseThenTrue:
        def __init__(self, state):
            self.state = state

        def get_all_prop(self, **kwargs):
            return {}

        def check_eq(self):
            if not self.state["done"]:
                self.state["done"] = True
                return False
            return True

    class _AnalyzeAlwaysTrue:
        def get_all_prop(self, **kwargs):
            return {}

        def check_eq(self):
            return True

    eq_state = {"done": False}

    class _FakeEQ21:
        def __init__(self, ac, work_dir):
            self.ac = ac
            self.work_dir = work_dir

        def ensure_system_exported(self):
            return exp

        def exec(self, **kwargs):
            calls["eq21"] = kwargs
            return "after_eq21"

        def analyze(self):
            return _AnalyzeFalseThenTrue(eq_state)

    class _FakeAdditional:
        def __init__(self, ac, work_dir):
            self.ac = ac
            self.work_dir = work_dir

        def exec(self, **kwargs):
            calls["additional"] = kwargs
            return "after_additional"

        def analyze(self):
            return _AnalyzeAlwaysTrue()

    class _FakeNPT:
        def __init__(self, ac, work_dir):
            self.ac = ac
            self.work_dir = work_dir

        def exec(self, **kwargs):
            calls["npt"] = kwargs
            return "after_npt"

        def analyze(self):
            return _AnalyzeAlwaysTrue()

    monkeypatch.setattr(prep_mod.eq, 'EQ21step', _FakeEQ21)
    monkeypatch.setattr(prep_mod.eq, 'Additional', _FakeAdditional)
    monkeypatch.setattr(prep_mod.eq, 'NPT', _FakeNPT)
    monkeypatch.setattr(prep_mod, 'validate_exported_system_dir', lambda path: [])

    outcome = equilibrate_bulk_with_eq21(
        label="Electrolyte",
        ac="packed_cell",
        work_dir=tmp_path,
        temp=300.0,
        press=1.0,
        mpi=1,
        omp=1,
        gpu=0,
        gpu_id=0,
        additional_loops=2,
        eq21_npt_mdp_overrides=fixed_xy_semiisotropic_npt_overrides(pressure_bar=1.0),
        additional_mdp_overrides={"pcoupltype": "semiisotropic"},
        final_npt_ns=0.2,
        final_npt_mdp_overrides={"pcoupltype": "semiisotropic", "ref_p": "1 1"},
    )

    assert outcome.final_cell == "after_npt"
    assert outcome.system_export.system_top == exp.system_top
    assert outcome.raw_system_meta == raw_dir / "system_meta.json"
    assert calls["eq21"] is not None and calls["eq21"]["eq21_npt_mdp_overrides"]["pcoupltype"] == "semiisotropic"
    assert calls["additional"] is not None and calls["additional"]["mdp_overrides"]["pcoupltype"] == "semiisotropic"
    assert calls["npt"] is not None and calls["npt"]["mdp_overrides"]["ref_p"] == "1 1"


def test_interface_dynamics_passes_ndx_to_equilibration_job(tmp_path: Path, monkeypatch):
    built = BuiltInterface(
        name="demo",
        route="route_a",
        axis="Z",
        out_dir=tmp_path / "03_interface",
        system_gro=tmp_path / "03_interface" / "system.gro",
        system_top=tmp_path / "03_interface" / "system.top",
        system_ndx=tmp_path / "03_interface" / "system.ndx",
        system_meta=tmp_path / "03_interface" / "system_meta.json",
        bottom_slab=None,
        top_slab=None,
        protocol_manifest=tmp_path / "03_interface" / "protocol_manifest.json",
        box_nm=(1.0, 1.0, 1.0),
        notes=(),
    )
    for path in (built.system_gro, built.system_top, built.system_meta):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    built.system_ndx.parent.mkdir(parents=True, exist_ok=True)
    built.system_ndx.write_text("[ System ]\n1 2 3 4\n\n[ BOTTOM ]\n1 2\n\n[ TOP ]\n3 4\n\n[ BOTTOM_CORE ]\n1\n\n[ TOP_CORE ]\n4\n", encoding="utf-8")

    captured: dict[str, object] = {}

    class _FakeEqJob:
        def __init__(self, *, gro, top, ndx, out_dir, stages, resources, runner=None, frac_last=0.5):
            captured["ndx"] = ndx
            captured["out_dir"] = out_dir

        def run(self, *, restart=None):
            final_dir = Path(captured["out_dir"]) / "05_production"
            final_dir.mkdir(parents=True, exist_ok=True)
            final = final_dir / "md.gro"
            final.write_text("done\n", encoding="utf-8")
            return final

    monkeypatch.setattr("yadonpy.interface.protocol.EquilibrationJob", _FakeEqJob)

    final = InterfaceDynamics(built=built, work_dir=tmp_path / "interface_md", restart=True).run(
        protocol=InterfaceProtocol.route_a(),
        mpi=1,
        omp=2,
        gpu=0,
        gpu_id=None,
    )

    assert final.exists()
    assert captured["ndx"] == built.system_ndx


def test_interface_dynamics_disables_core_freeze_when_core_groups_are_empty(tmp_path: Path, monkeypatch):
    built = BuiltInterface(
        name="demo",
        route="route_a",
        axis="Z",
        out_dir=tmp_path / "03_interface",
        system_gro=tmp_path / "03_interface" / "system.gro",
        system_top=tmp_path / "03_interface" / "system.top",
        system_ndx=tmp_path / "03_interface" / "system.ndx",
        system_meta=tmp_path / "03_interface" / "system_meta.json",
        bottom_slab=None,
        top_slab=None,
        protocol_manifest=tmp_path / "03_interface" / "protocol_manifest.json",
        box_nm=(1.0, 1.0, 1.0),
        notes=(),
    )
    built.system_gro.parent.mkdir(parents=True, exist_ok=True)
    built.system_gro.write_text("x\n", encoding="utf-8")
    built.system_top.write_text("x\n", encoding="utf-8")
    built.system_meta.write_text("{}\n", encoding="utf-8")
    built.system_ndx.write_text("[ System ]\n1 2 3 4\n\n[ BOTTOM ]\n1 2\n\n[ TOP ]\n3 4\n\n[ INTERFACE_ZONE ]\n2 3\n", encoding="utf-8")

    captured: dict[str, object] = {}

    class _FakeEqJob:
        def __init__(self, *, gro, top, ndx, out_dir, stages, resources, runner=None, frac_last=0.5):
            captured["stages"] = stages
            captured["out_dir"] = out_dir

        def run(self, *, restart=None):
            final_dir = Path(captured["out_dir"]) / "05_production"
            final_dir.mkdir(parents=True, exist_ok=True)
            final = final_dir / "md.gro"
            final.write_text("done\n", encoding="utf-8")
            return final

    monkeypatch.setattr("yadonpy.interface.protocol.EquilibrationJob", _FakeEqJob)

    md_root = tmp_path / "interface_md"
    final = InterfaceDynamics(built=built, work_dir=md_root, restart=True).run(
        protocol=InterfaceProtocol.route_a(),
        mpi=1,
        omp=2,
        gpu=0,
        gpu_id=None,
    )

    assert final.exists()
    stages = captured["stages"]
    assert "freezegrps" not in stages[0].mdp.render()
    assert "freezegrps" not in stages[1].mdp.render()
    protocol_manifest = json.loads((md_root / "protocol.json").read_text(encoding="utf-8"))
    assert protocol_manifest["effective_protocol"]["freeze_cores_pre_contact"] is False
    assert any("BOTTOM_CORE" in note for note in protocol_manifest["preflight"]["notes"])
