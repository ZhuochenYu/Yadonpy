from __future__ import annotations

import argparse
import json
from pathlib import Path

from yadonpy.sim.benchmarking import _dump_json, build_screening_compare, load_benchmark_analysis_dir


def _default_output(paths: list[Path]) -> Path:
    if not paths:
        return Path.cwd() / "screening_compare.json"
    parent = paths[0].parent
    if parent.name == "06_analysis":
        return parent.parent.parent / "screening_compare.json"
    return parent / "screening_compare.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple PEO/LiTFSI 60C screening runs.")
    parser.add_argument(
        "analysis_dirs",
        nargs="+",
        help="Paths to analysis directories that contain benchmark_compare.json and companion analysis JSON files.",
    )
    parser.add_argument("--out", default=None, help="Output JSON path. Defaults near the provided screening directories.")
    args = parser.parse_args()

    analysis_dirs = [Path(p).resolve() for p in args.analysis_dirs]
    runs = [load_benchmark_analysis_dir(path) for path in analysis_dirs]
    payload = build_screening_compare(runs=runs)
    out_path = Path(args.out).resolve() if args.out else _default_output(analysis_dirs)
    _dump_json(out_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
