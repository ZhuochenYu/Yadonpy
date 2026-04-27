"""Post-processing helpers for interface-specific GROMACS artifacts.

This module collects small parsers and data reshapers used after interface
construction, such as reading index groups and extracting role-specific
diagnostics. Keeping these utilities separate avoids bloating the builders with
analysis-oriented file handling.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def read_ndx_groups(ndx_path: str | Path) -> dict[str, list[int]]:
    path = Path(ndx_path)
    groups: dict[str, list[int]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line.strip("[]").strip()
            groups.setdefault(current, [])
            continue
        if current is None:
            continue
        groups[current].extend(int(tok) for tok in line.split())
    return groups


def _append(cat: dict[str, list[str]], key: str, name: str) -> None:
    cat.setdefault(key, []).append(name)


def _append_region(cat: dict[str, dict[str, list[str]]], key: str, region: str, name: str) -> None:
    cat.setdefault(key, {}).setdefault(region, []).append(name)


def build_interface_group_catalog(ndx_path: str | Path) -> dict[str, Any]:
    path = Path(ndx_path)
    groups = read_ndx_groups(path)
    categories: dict[str, Any] = {
        "system": [],
        "region_labels": [],
        "layer_labels": [],
        "region_moltypes": {"BOTTOM": [], "TOP": []},
        "region_atomtypes": {"BOTTOM": [], "TOP": []},
        "region_moltype_atomtypes": {"BOTTOM": [], "TOP": []},
        "region_surface_moltypes": {"BOTTOM": [], "TOP": []},
        "region_core_moltypes": {"BOTTOM": [], "TOP": []},
        "region_surface_atomtypes": {"BOTTOM": [], "TOP": []},
        "region_core_atomtypes": {"BOTTOM": [], "TOP": []},
        "region_surface_moltype_atomtypes": {"BOTTOM": [], "TOP": []},
        "region_core_moltype_atomtypes": {"BOTTOM": [], "TOP": []},
        "region_instances": {"BOTTOM": [], "TOP": []},
        "region_representatives": {"BOTTOM": [], "TOP": []},
        "layer_moltypes": {},
        "layer_atomtypes": {},
        "layer_moltype_atomtypes": {},
        "layer_surface_moltypes": {},
        "layer_core_moltypes": {},
        "layer_surface_atomtypes": {},
        "layer_core_atomtypes": {},
        "layer_surface_moltype_atomtypes": {},
        "layer_core_moltype_atomtypes": {},
        "layer_instances": {},
        "layer_representatives": {},
        "global_moltypes": [],
        "global_atomtypes": [],
        "global_moltype_atomtypes": [],
        "global_representatives": [],
        "other": [],
    }

    region_labels = {
        "System",
        "BOTTOM",
        "TOP",
        "BOTTOM_CORE",
        "BOTTOM_SURFACE",
        "TOP_CORE",
        "TOP_SURFACE",
        "INTERFACE_ZONE",
    }
    layer_prefixes = tuple(f"{region}{idx}" for region in ("BOTTOM", "TOP") for idx in range(1, 10))
    layer_label_re = re.compile(r"^(BOTTOM|TOP)\d+(?:_(?:CORE|SURFACE))?$")

    for name in sorted(groups.keys()):
        if name in region_labels:
            if name == "System":
                categories["system"].append(name)
            else:
                categories["region_labels"].append(name)
            continue
        if layer_label_re.match(name):
            categories["layer_labels"].append(name)
            continue

        matched = False
        for region in ("BOTTOM", "TOP"):
            prefixes = [
                (f"{region}_SURFACE_TYPE_", "region_surface_moltype_atomtypes"),
                (f"{region}_CORE_TYPE_", "region_core_moltype_atomtypes"),
                (f"{region}_SURFACE_ATYPE_", "region_surface_atomtypes"),
                (f"{region}_CORE_ATYPE_", "region_core_atomtypes"),
                (f"{region}_SURFACE_MOL_", "region_surface_moltypes"),
                (f"{region}_CORE_MOL_", "region_core_moltypes"),
                (f"{region}_TYPE_", "region_moltype_atomtypes"),
                (f"{region}_ATYPE_", "region_atomtypes"),
                (f"{region}_MOL_", "region_moltypes"),
                (f"{region}_INST_", "region_instances"),
                (f"{region}_REP_", "region_representatives"),
            ]
            for prefix, category in prefixes:
                if name.startswith(prefix):
                    _append(categories[category], region, name)
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue

        for layer in layer_prefixes:
            layer_prefix_map = [
                (f"{layer}_SURFACE_TYPE_", "layer_surface_moltype_atomtypes"),
                (f"{layer}_CORE_TYPE_", "layer_core_moltype_atomtypes"),
                (f"{layer}_SURFACE_ATYPE_", "layer_surface_atomtypes"),
                (f"{layer}_CORE_ATYPE_", "layer_core_atomtypes"),
                (f"{layer}_SURFACE_", "layer_surface_moltypes"),
                (f"{layer}_CORE_", "layer_core_moltypes"),
                (f"{layer}_TYPE_", "layer_moltype_atomtypes"),
                (f"{layer}_ATYPE_", "layer_atomtypes"),
                (f"{layer}_INST_", "layer_instances"),
                (f"{layer}_REP_", "layer_representatives"),
                (f"{layer}_", "layer_moltypes"),
            ]
            for prefix, category in layer_prefix_map:
                if name.startswith(prefix):
                    _append_region(categories, category, layer, name)
                    matched = True
                    break
            if matched:
                break
        if matched:
            continue

        if name.startswith("TYPE_"):
            categories["global_moltype_atomtypes"].append(name)
        elif name.startswith("ATYPE_"):
            categories["global_atomtypes"].append(name)
        elif name.startswith("MOL_"):
            categories["global_moltypes"].append(name)
        elif name.startswith("REP_"):
            categories["global_representatives"].append(name)
        else:
            categories["other"].append(name)

    return {
        "ndx_path": str(path),
        "total_groups": len(groups),
        "group_sizes": {name: len(idxs) for name, idxs in groups.items()},
        "categories": categories,
        "naming_notes": [
            "Region labels are coarse spatial labels such as BOTTOM, TOP, CORE, SURFACE, and INTERFACE_ZONE.",
            "Region moltype groups use <REGION>_MOL_<MOLTYPE>.",
            "Region atomtype groups use <REGION>_ATYPE_<ATOMTYPE>.",
            "Region moltype-atomtype groups use <REGION>_TYPE_<MOLTYPE>_<ATOMTYPE>.",
            "Per-instance molecule groups use <REGION>_INST_<MOLTYPE>_<NNNN>.",
            "Representative-atom groups use <REGION>_REP_<MOLTYPE> and REP_<MOLTYPE>.",
            "Layered aliases use <POSITION><N>_<MOLTYPE>, for example BOTTOM1_CMC or TOP1_LIPF6.",
            "Layered TYPE aliases use <POSITION><N>_TYPE_<MOLTYPE>_<ATOMTYPE>.",
        ],
    }


def export_interface_group_catalog(ndx_path: str | Path, out_path: str | Path | None = None) -> Path:
    ndx = Path(ndx_path)
    target = Path(out_path) if out_path is not None else ndx.with_name("system_ndx_groups.json")
    payload = build_interface_group_catalog(ndx)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


__all__ = [
    "read_ndx_groups",
    "build_interface_group_catalog",
    "export_interface_group_catalog",
]
