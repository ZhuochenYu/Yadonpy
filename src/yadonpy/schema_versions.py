from __future__ import annotations

"""Central schema/version constants for cache invalidation sensitive workflows."""

PACKAGE_SERIES = "0.8.74"
AMORPHOUS_CELL_SCHEMA_VERSION = f"{PACKAGE_SERIES}-amorphous-cell-v2"
EXPORT_SYSTEM_SCHEMA_VERSION = f"{PACKAGE_SERIES}-export-system-v2"
INTERFACE_BUILD_SCHEMA_VERSION = f"{PACKAGE_SERIES}-interface-build-v5"
ANALYSIS_SCHEMA_VERSION = f"{PACKAGE_SERIES}-analysis-v2"

__all__ = [
    "PACKAGE_SERIES",
    "AMORPHOUS_CELL_SCHEMA_VERSION",
    "EXPORT_SYSTEM_SCHEMA_VERSION",
    "INTERFACE_BUILD_SCHEMA_VERSION",
    "ANALYSIS_SCHEMA_VERSION",
]
