from __future__ import annotations

import argparse
from pathlib import Path

from _autofix import DEFAULT_CONFIG_PATH, run_autofix_loop


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous Example 08 autofix loop.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent / "autofix_runs")
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run_autofix_loop(
        config_path=args.config,
        base_dir=args.base_dir,
        total_hours=args.hours,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
