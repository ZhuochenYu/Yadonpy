"""Force-field registry and lazy factory helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Dict, Iterable, Tuple

_FORCEFIELD_SPECS: Dict[str, Tuple[str, str]] = {
    "gaff": ("yadonpy.ff.gaff", "GAFF"),
    "gaff2": ("yadonpy.ff.gaff2", "GAFF2"),
    "gaff2_mod": ("yadonpy.ff.gaff2_mod", "GAFF2_mod"),
    "merz": ("yadonpy.ff.merz", "MERZ"),
    "oplsaa": ("yadonpy.ff.oplsaa", "OPLSAA"),
    "dreiding": ("yadonpy.ff.dreiding", "Dreiding"),
}

_FORCEFIELD_ALIASES = {
    "gaff": "gaff",
    "gaff2": "gaff2",
    "gaff2-mod": "gaff2_mod",
    "gaff2_mod": "gaff2_mod",
    "gaff2mod": "gaff2_mod",
    "mod": "gaff2_mod",
    "classic": "gaff2",
    "orig": "gaff2",
    "original": "gaff2",
    "merz": "merz",
    "merzop": "merz",
    "merzopc3": "merz",
    "opls": "oplsaa",
    "opls-aa": "oplsaa",
    "oplsaa": "oplsaa",
    "oplsaa2024": "oplsaa",
    "opls-aa-2024": "oplsaa",
    "opls2024": "oplsaa",
    "dreiding": "dreiding",
}


def canonical_forcefield_name(name: str) -> str:
    """Normalize a user-supplied force-field name."""
    normalized = str(name or "").strip().lower()
    canonical = _FORCEFIELD_ALIASES.get(normalized)
    if canonical is None:
        raise ValueError(f"Unknown force field: {name}")
    return canonical



def available_forcefields() -> Iterable[str]:
    """Return canonical force-field names supported by the lazy registry."""
    return tuple(_FORCEFIELD_SPECS)



def create_forcefield(name: str, **kwargs):
    """Instantiate a force field by canonical name or alias."""
    canonical = canonical_forcefield_name(name)
    module_name, class_name = _FORCEFIELD_SPECS[canonical]
    module = import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)
