from __future__ import annotations

from pathlib import Path

from yadonpy.ff.oplsaa_reference import audit_oplsaa_reference


def test_oplsaa_reference_audit_reads_fake_gromacs_and_moltemplate_sources(tmp_path: Path):
    gmx_root = tmp_path / "oplsaa.ff"
    gmx_root.mkdir(parents=True, exist_ok=True)
    (gmx_root / "forcefield.itp").write_text(
        "[ defaults ]\n; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n1 3 yes 0.5 0.5\n",
        encoding="utf-8",
    )
    (gmx_root / "ffnonbonded.itp").write_text(
        "[ atomtypes ]\n; name mass charge ptype sigma epsilon\nopls_140 1.0080 0.0 A 0.250000 0.125520\n",
        encoding="utf-8",
    )
    (gmx_root / "ffbonded.itp").write_text(
        "#define improper_O_C_X_Y 180.0 43.93200 2\n"
        "#define improper_Z_N_X_Y 180.0 4.18400 2\n"
        "#define improper_Z_CA_X_Y 180.0 4.60240 2\n",
        encoding="utf-8",
    )
    (gmx_root / "aminoacids.rtp").write_text(
        "[ ARG ]\n"
        "[ impropers ]\n"
        "NE NH1 CZ NH2 improper_O_C_X_Y\n"
        "CZ HH11 NH1 HH12 improper_Z_N_X_Y\n",
        encoding="utf-8",
    )

    lt_path = tmp_path / "oplsaa2024.lt"
    lt_path.write_text(
        "bond_coeff @bond:CT_CT 300.0 1.529\n"
        "angle_coeff @angle:CT_CT_CT 50.0 109.5\n"
        "dihedral_coeff @dihedral:CT_CT_CT_CT 0.0 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    report = audit_oplsaa_reference(
        gromacs_root=gmx_root,
        moltemplate_lt_path=lt_path,
        moltemplate_par_path=tmp_path / "dummy.par",
    )

    assert report["defaults_parity"]["matches"] is True
    assert "improper_O_C_X_Y" in report["improper_template_parity"]["available_in_gromacs"]
    assert report["improper_template_parity"]["template_usage"]["improper_O_C_X_Y"][0]["residue"] == "ARG"
