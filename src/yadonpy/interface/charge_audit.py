"""Charge-audit summaries for cells, slabs, and assembled interfaces.

The routines here provide small JSON-friendly reports that make net charge,
species charge balance, and metadata consistency visible before running MD.
They are intentionally lightweight so examples can print or persist them during
preflight checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_cell_charge(cell) -> dict[str, Any]:
    if cell is None or not hasattr(cell, "HasProp") or not cell.HasProp("_yadonpy_cell_meta"):
        return {
            "kind": "cell_missing_meta",
            "net_charge_raw": 0.0,
            "net_charge_scaled": 0.0,
            "charge_tolerance": 0.0,
            "net_charge_ok": False,
        }
    payload = json.loads(cell.GetProp("_yadonpy_cell_meta"))
    return {
        "kind": "cell",
        "net_charge_raw": float(payload.get("net_charge_raw", 0.0)),
        "net_charge_scaled": float(payload.get("net_charge_scaled", 0.0)),
        "charge_tolerance": float(payload.get("charge_tolerance", 0.0)),
        "net_charge_ok": bool(payload.get("net_charge_ok", False)),
    }


def summarize_charge_meta(meta_path: Path) -> dict[str, Any]:
    payload = _load_json(Path(meta_path))
    if "net_charge_e" in payload:
        return {
            "kind": "meta_net_charge",
            "net_charge_e": float(payload.get("net_charge_e", 0.0)),
            "path": str(meta_path),
        }
    correction = payload.get("export_charge_correction")
    if correction:
        return {
            "kind": "meta_export_correction",
            "system_charge_before": float(correction.get("system_charge_before", 0.0)),
            "system_charge_after": float(correction.get("system_charge_after", 0.0)),
            "target_moltype": correction.get("target_moltype"),
            "path": str(meta_path),
        }
    if ("net_charge_scaled" in payload) or ("net_charge_raw" in payload):
        return {
            "kind": "meta_scaled_raw",
            "net_charge_raw": float(payload.get("net_charge_raw", 0.0)),
            "net_charge_scaled": float(payload.get("net_charge_scaled", 0.0)),
            "path": str(meta_path),
        }
    return {
        "kind": "meta_unknown",
        "path": str(meta_path),
    }


def format_cell_charge_audit(label: str, cell) -> str:
    summary = summarize_cell_charge(cell)
    if str(summary.get("kind") or "") == "cell_missing_meta":
        return f"[CHARGE] {label}: no _yadonpy_cell_meta found on the returned cell"
    return (
        f"[CHARGE] {label}: raw={summary['net_charge_raw']:.6f} e | "
        f"scaled={summary['net_charge_scaled']:.6f} e | "
        f"tol={summary['charge_tolerance']:.2e} | ok={summary['net_charge_ok']}"
    )


def format_charge_meta_audit(label: str, meta_path: Path) -> str:
    summary = summarize_charge_meta(meta_path)
    kind = str(summary.get("kind") or "")
    if kind == "meta_net_charge":
        return f"[CHARGE] {label}: net_charge_e={summary['net_charge_e']:.6f} e | meta={summary['path']}"
    if kind == "meta_export_correction":
        return (
            f"[CHARGE] {label}: export_charge_before={summary['system_charge_before']:.6f} e | "
            f"after={summary['system_charge_after']:.6f} e | target={summary['target_moltype']} | meta={summary['path']}"
        )
    if kind == "meta_scaled_raw":
        return (
            f"[CHARGE] {label}: raw={summary['net_charge_raw']:.6f} e | "
            f"scaled={summary['net_charge_scaled']:.6f} e | meta={summary['path']}"
        )
    return f"[CHARGE] {label}: no charge fields found in {summary['path']}"


__all__ = [
    "format_cell_charge_audit",
    "format_charge_meta_audit",
    "summarize_cell_charge",
    "summarize_charge_meta",
]
