from __future__ import annotations

import json
from pathlib import Path

from yadonpy.sim.cleanup import clean_md_trajectory_files


def test_clean_md_trajectory_files_removes_only_trajectory_streams(tmp_path: Path):
    keep = [
        tmp_path / "05_npt_production" / "03_npt" / "md.gro",
        tmp_path / "05_npt_production" / "03_npt" / "md.edr",
        tmp_path / "05_npt_production" / "03_npt" / "summary.json",
        tmp_path / "06_analysis" / "msd_summary.json",
        tmp_path / "06_analysis" / "msd.svg",
    ]
    remove = [
        tmp_path / "05_npt_production" / "03_npt" / "md.trr",
        tmp_path / "05_npt_production" / "03_npt" / "md.xtc",
        tmp_path / "06_analysis" / "sigma" / "_nojump.trr",
    ]
    for path in keep + remove:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 7)

    result = clean_md_trajectory_files(tmp_path, enabled=True)

    assert sorted(result.removed_files) == sorted(str(p.relative_to(tmp_path)) for p in remove)
    assert result.removed_bytes == 21
    for path in remove:
        assert not path.exists()
    for path in keep:
        assert path.exists()
    summary = json.loads((tmp_path / "trajectory_cleanup_summary.json").read_text(encoding="utf-8"))
    assert summary["removed_bytes"] == 21


def test_clean_md_trajectory_files_dry_run_keeps_files(tmp_path: Path):
    trr = tmp_path / "md.trr"
    trr.write_bytes(b"123")

    result = clean_md_trajectory_files(tmp_path, dry_run=True)

    assert result.removed_files == []
    assert result.kept_files == ["md.trr"]
    assert result.removed_bytes == 3
    assert trr.exists()


def test_clean_md_trajectory_files_disabled_is_noop(tmp_path: Path):
    xtc = tmp_path / "md.xtc"
    xtc.write_bytes(b"123")

    result = clean_md_trajectory_files(tmp_path, enabled=False)

    assert result.enabled is False
    assert result.removed_files == []
    assert result.removed_bytes == 0
    assert xtc.exists()
