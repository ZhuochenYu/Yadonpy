from __future__ import annotations

import json
from pathlib import Path

import pytest

import yadonpy.workflow.studies as studies
from yadonpy.gmx.workflows.elongation import ElongationJob
from yadonpy.gmx.workflows.tg import TgJob, fit_tg_piecewise_linear


class FakeTgRunner:
    def grompp(self, *, out_tpr: Path, **kwargs) -> None:
        out_tpr.write_text("fake tpr\n", encoding="utf-8")

    def mdrun(self, *, deffnm: str, cwd: Path, **kwargs) -> None:
        cwd = Path(cwd)
        (cwd / f"{deffnm}.gro").write_text(
            "\n".join(
                [
                    "fake",
                    "    1",
                    "    1MOL     A    1   0.100   0.100   0.100",
                    "   1.00000   1.00000   1.00000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (cwd / f"{deffnm}.cpt").write_text("fake cpt\n", encoding="utf-8")
        (cwd / f"{deffnm}.edr").write_text("fake edr\n", encoding="utf-8")

    def energy_xvg(self, *, out_xvg: Path, terms, cwd: Path, **kwargs) -> None:
        try:
            token = Path(cwd).name.split("_")[-1]
            temperature_k = float(token.rstrip("K"))
        except Exception:
            temperature_k = 300.0
        density = 1450.0 - 0.8 * (temperature_k - 300.0)
        volume = 1000.0 / density
        rows = [
            [0.0, density + 5.0, temperature_k, 1.0, volume],
            [1.0, density + 2.0, temperature_k, 1.0, volume],
            [2.0, density, temperature_k, 1.0, volume],
            [3.0, density - 2.0, temperature_k, 1.0, volume],
            [4.0, density - 5.0, temperature_k, 1.0, volume],
        ]
        header = [
            '@    xaxis  label "Time (ps)"',
            '@    yaxis  label "Thermo"',
        ]
        for idx, term in enumerate(terms):
            header.append(f'@    s{idx} legend "{term}"')
        payload = header + [" ".join(map(str, row)) for row in rows]
        Path(out_xvg).write_text("\n".join(payload) + "\n", encoding="utf-8")


class FakeElongationRunner:
    def grompp(self, *, out_tpr: Path, **kwargs) -> None:
        out_tpr.write_text("fake tpr\n", encoding="utf-8")

    def mdrun(self, *, deffnm: str, cwd: Path, **kwargs) -> None:
        cwd = Path(cwd)
        (cwd / f"{deffnm}.gro").write_text(
            "\n".join(
                [
                    "fake",
                    "    1",
                    "    1MOL     A    1   0.100   0.100   0.100",
                    "   1.10000   1.00000   1.00000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (cwd / f"{deffnm}.cpt").write_text("fake cpt\n", encoding="utf-8")
        (cwd / f"{deffnm}.edr").write_text("fake edr\n", encoding="utf-8")

    def energy_xvg(self, *, out_xvg: Path, terms, cwd: Path, **kwargs) -> None:
        rows = []
        for idx in range(10):
            strain = idx * 0.01
            lx = 1.0 + strain
            sigma_xx_gpa = 5.0 * strain
            pres_xx_bar = -sigma_xx_gpa / 1.0e-4
            sigma_dev_gpa = 3.0 * strain
            lateral_bar = (pres_xx_bar + 2.0 * (-sigma_dev_gpa / 1.0e-4)) / 2.0
            rows.append([float(idx), lx, 1.0, 1.0, pres_xx_bar, lateral_bar, lateral_bar])
        header = [
            '@    xaxis  label "Time (ps)"',
            '@    yaxis  label "Stress"',
        ]
        for idx, term in enumerate(terms):
            header.append(f'@    s{idx} legend "{term}"')
        payload = header + [" ".join(map(str, row)) for row in rows]
        Path(out_xvg).write_text("\n".join(payload) + "\n", encoding="utf-8")


def _write_prepared_system(root: Path) -> tuple[Path, Path]:
    gro = root / "system.gro"
    top = root / "system.top"
    gro.write_text(
        "\n".join(
            [
                "prepared",
                "    1",
                "    1MOL     A    1   0.100   0.100   0.100",
                "   1.00000   1.00000   1.00000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    top.write_text("[ system ]\nprepared\n", encoding="utf-8")
    return gro, top


def test_fit_tg_piecewise_linear_supports_density_and_specific_volume():
    temps = [500.0, 460.0, 420.0, 380.0, 340.0, 300.0]
    densities = [1180.0, 1200.0, 1220.0, 1260.0, 1300.0, 1340.0]

    density_fit = fit_tg_piecewise_linear(temps, densities, fit_metric="density")
    sv_fit = fit_tg_piecewise_linear(temps, densities, fit_metric="specific_volume")

    assert density_fit.fit_metric == "density"
    assert sv_fit.fit_metric == "specific_volume"
    assert len(density_fit.specific_volume_cm3_g) == len(temps)
    assert len(sv_fit.fit_values) == len(temps)
    assert density_fit.total_sse >= 0.0
    assert sv_fit.total_sse >= 0.0


def test_fit_tg_piecewise_linear_requires_at_least_five_points():
    with pytest.raises(ValueError, match="at least 5"):
        fit_tg_piecewise_linear([500.0, 450.0, 400.0, 350.0], [1200.0, 1220.0, 1260.0, 1310.0])


def test_resolve_prepared_system_from_work_dir(tmp_path: Path):
    work_dir = tmp_path / "work_dir"
    gro = work_dir / "03_eq" / "04_md" / "md.gro"
    top = work_dir / "00_system" / "system.top"
    gro.parent.mkdir(parents=True, exist_ok=True)
    top.parent.mkdir(parents=True, exist_ok=True)
    gro.write_text(
        "\n".join(
            [
                "prepared",
                "    1",
                "    1MOL     A    1   0.100   0.100   0.100",
                "   1.00000   1.00000   1.00000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    top.write_text("[ system ]\nprepared\n", encoding="utf-8")

    prepared = studies.resolve_prepared_system(work_dir=work_dir, source_name="eq02")

    assert prepared.gro == gro.resolve()
    assert prepared.top == top.resolve()
    assert prepared.source == "eq02"


def test_run_tg_scan_gmx_wrapper_uses_high_level_resolution(monkeypatch, tmp_path: Path):
    gro, top = _write_prepared_system(tmp_path)
    calls: dict[str, object] = {}

    def fake_tg_scan_gmx(**kwargs):
        calls.update(kwargs)
        summary_path = Path(kwargs["out_dir"]) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "job": "TgJob",
                    "fit": {
                        "fit_metric": kwargs["fit_metric"],
                        "tg_k": 355.0,
                        "split_index": 2,
                        "total_sse": 1.25,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return summary_path

    monkeypatch.setattr(studies.steps, "tg_scan_gmx", fake_tg_scan_gmx)

    result = studies.run_tg_scan_gmx(
        gro=gro,
        top=top,
        out_dir=tmp_path / "tg",
        profile="smoke",
        fit_metric="specific_volume",
    )

    assert result.kind == "tg"
    assert result.summary["fit"]["fit_metric"] == "specific_volume"
    assert Path(calls["gro"]) == gro.resolve()
    assert Path(calls["top"]) == top.resolve()
    assert calls["fit_metric"] == "specific_volume"
    assert calls["temperatures_k"] == [460.0, 420.0, 380.0, 340.0, 300.0]


def test_run_elongation_gmx_wrapper_returns_material_summary(monkeypatch, tmp_path: Path):
    gro, top = _write_prepared_system(tmp_path)
    calls: dict[str, object] = {}

    def fake_elongation_gmx(**kwargs):
        calls.update(kwargs)
        summary_path = Path(kwargs["out_dir"]) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "job": "ElongationJob",
                    "direction": kwargs["direction"],
                    "material_summary": {
                        "youngs_modulus_gpa": 4.2,
                        "max_stress_gpa": 0.9,
                        "strain_at_max_stress": 0.18,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return summary_path

    monkeypatch.setattr(studies.steps, "elongation_gmx", fake_elongation_gmx)

    result = studies.run_elongation_gmx(
        gro=gro,
        top=top,
        out_dir=tmp_path / "elong",
        profile="smoke",
        direction="y",
    )

    assert result.kind == "elongation"
    assert result.summary["material_summary"]["youngs_modulus_gpa"] == 4.2
    assert calls["direction"] == "y"
    assert calls["total_strain"] == 0.10


def test_tg_job_writes_specific_volume_and_fit_metadata(monkeypatch, tmp_path: Path):
    import yadonpy.gmx.workflows.tg as tgmod

    gro, top = _write_prepared_system(tmp_path)
    monkeypatch.setattr(tgmod, "pbc_mol_fix_inplace", lambda *args, **kwargs: {"applied": False, "error": None})
    monkeypatch.setattr(tgmod, "write_mol2_from_top_gro_parmed", lambda **kwargs: None)

    job = TgJob(
        gro=gro,
        top=top,
        out_dir=tmp_path / "tg_job",
        temperatures_k=[460.0, 420.0, 380.0, 340.0, 300.0],
        fit_metric="specific_volume",
        runner=FakeTgRunner(),
        auto_plot=False,
    )

    summary_path = job.run(restart=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["scan"]["fit_metric"] == "specific_volume"
    assert len(summary["points"]) == 5
    assert summary["points"][0]["specific_volume_cm3_g"] is not None
    assert summary["fit"]["fit_metric"] == "specific_volume"
    assert "specific_volume_cm3_g" in summary["curve"]


def test_elongation_job_writes_material_summary(monkeypatch, tmp_path: Path):
    import yadonpy.gmx.workflows.elongation as elongmod

    gro, top = _write_prepared_system(tmp_path)
    monkeypatch.setattr(elongmod, "pbc_mol_fix_inplace", lambda *args, **kwargs: {"applied": False, "error": None})
    monkeypatch.setattr(elongmod, "write_mol2_from_top_gro_parmed", lambda **kwargs: None)

    job = ElongationJob(
        gro=gro,
        top=top,
        out_dir=tmp_path / "elong_job",
        direction="x",
        modulus_fit_max_strain=0.03,
        modulus_fit_min_points=3,
        runner=FakeElongationRunner(),
        auto_plot=False,
    )

    summary_path = job.run(restart=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    material = dict(summary["material_summary"])

    assert summary["direction"] == "x"
    assert len(summary["results"]["engineering_strain"]) == 10
    assert material["youngs_modulus_gpa"] is not None
    assert material["youngs_modulus_gpa"] > 0.0
    assert material["max_stress_gpa"] > 0.0
    assert material["strain_at_max_stress"] >= 0.0
