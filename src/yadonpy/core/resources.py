"""Package resource path helpers.

These helpers keep packaged data lookups in one place so modules do not need to
reconstruct paths from ``__file__`` repeatedly.
"""

from __future__ import annotations

from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def package_path(*parts: str) -> Path:
    """Return an absolute path inside the installed yadonpy package."""
    return _PACKAGE_ROOT.joinpath(*parts)


def core_data_path(*parts: str) -> Path:
    """Return an absolute path inside ``yadonpy/core``."""
    return package_path("core", *parts)


def ff_data_path(*parts: str) -> Path:
    """Return an absolute path inside ``yadonpy/ff``."""
    return package_path("ff", *parts)
