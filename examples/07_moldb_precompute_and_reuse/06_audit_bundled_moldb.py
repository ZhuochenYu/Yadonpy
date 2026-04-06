from __future__ import annotations

import json
from pathlib import Path

from yadonpy.core.data_dir import audit_active_bundle_sync, ensure_initialized
from yadonpy.diagnostics import doctor


BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir" / "06_audit_bundled_moldb"


if __name__ == "__main__":
    doctor(print_report=True)
    layout = ensure_initialized()
    audit = audit_active_bundle_sync(layout=layout)

    work_dir.mkdir(parents=True, exist_ok=True)
    summary_path = work_dir / "bundle_sync_audit.json"
    summary_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[MolDB bundled sync audit]")
    print(f"layout_root = {audit['layout_root']}")
    print(f"moldb_dir = {audit['moldb_dir']}")
    print(f"bundle_dir = {audit['bundle_dir']}")
    print(f"missing_objects = {len(audit['missing_objects'])}")
    print(f"stale_variants = {len(audit['stale_variants'])}")
    print(f"bundled_more_complete_records = {len(audit['bundled_more_complete_records'])}")
    print(f"user_only_records = {len(audit['user_only_records'])}")
    print(f"summary_path = {summary_path}")
