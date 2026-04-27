"""Typed metadata helpers for YadonPy workflow provenance.

The codebase stores scientific provenance in two places: JSON files on disk and
RDKit molecule properties.  This module provides small dataclass wrappers and
centralized prop helpers so new code does not need to repeat ad-hoc
``HasProp/GetProp/json.loads`` logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from ..schema_versions import (
    BENCHMARK_SUMMARY_SCHEMA_VERSION,
    EQUILIBRIUM_SCHEMA_VERSION,
    METADATA_SCHEMA_VERSION,
    SPECIES_FORCEFIELD_SUMMARY_SCHEMA_VERSION,
)


RESP_CONSTRAINTS_PROP = "_yadonpy_resp_constraints_json"
PSIRESP_CONSTRAINTS_PROP = "_yadonpy_psiresp_constraints"
RESP_PROFILE_PROP = "_yadonpy_resp_profile"
QM_RECIPE_PROP = "_yadonpy_qm_recipe_json"


def stable_payload(payload: Any) -> Any:
    """Return a JSON-compatible, deterministic representation."""

    try:
        return json.loads(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        )
    except Exception:
        return payload


def read_json_prop(mol: Any, key: str, *, default: Any = None) -> Any:
    """Read a JSON RDKit molecule property, returning ``default`` on absence/error."""

    try:
        if mol is None or not hasattr(mol, "HasProp") or not mol.HasProp(key):
            return default
        raw = str(mol.GetProp(key)).strip()
        if not raw:
            return default
        return json.loads(raw)
    except Exception:
        return default


def write_json_prop(mol: Any, key: str, payload: Any) -> None:
    """Write a JSON RDKit molecule property using the repository convention."""

    if mol is None or not hasattr(mol, "SetProp"):
        return
    mol.SetProp(key, json.dumps(stable_payload(payload), ensure_ascii=False))


def read_text_prop(mol: Any, key: str, *, default: str | None = None) -> str | None:
    try:
        if mol is None or not hasattr(mol, "HasProp") or not mol.HasProp(key):
            return default
        value = str(mol.GetProp(key)).strip()
        return value if value else default
    except Exception:
        return default


def write_text_prop(mol: Any, key: str, value: str | None) -> None:
    if value is None or mol is None or not hasattr(mol, "SetProp"):
        return
    text = str(value).strip()
    if text:
        mol.SetProp(key, text)


def schema_stamp(payload: Mapping[str, Any], *, schema_version: str, kind: str) -> dict[str, Any]:
    """Return ``payload`` with schema provenance added without overwriting content."""

    out = dict(payload)
    out.setdefault("schema_version", str(schema_version))
    out.setdefault("summary_kind", str(kind))
    return out


def write_schema_json(path: str | Path, payload: Mapping[str, Any], *, schema_version: str, kind: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(schema_stamp(payload, schema_version=schema_version, kind=kind), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return out


@dataclass(frozen=True)
class RespConstraintMetadata:
    """Typed view of RESP/PsiRESP constraint metadata."""

    mode: str | None = None
    resp_profile: str | None = None
    equivalence_groups: list[list[int]] = field(default_factory=list)
    charged_group_constraints: list[dict[str, Any]] = field(default_factory=list)
    charged_region_equivalence_groups: list[dict[str, Any]] = field(default_factory=list)
    neutral_remainder_indices: list[int] = field(default_factory=list)
    neutral_remainder_charge: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    schema_version: str = METADATA_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "RespConstraintMetadata":
        data = dict(payload or {})
        return cls(
            mode=(str(data.get("mode")).strip() if data.get("mode") is not None else None),
            resp_profile=(str(data.get("resp_profile")).strip().lower() if data.get("resp_profile") else None),
            equivalence_groups=[
                [int(i) for i in group]
                for group in (data.get("equivalence_groups") or [])
                if isinstance(group, (list, tuple))
            ],
            charged_group_constraints=[
                dict(item)
                for item in (data.get("charged_group_constraints") or [])
                if isinstance(item, Mapping)
            ],
            charged_region_equivalence_groups=[
                dict(item)
                for item in (data.get("charged_region_equivalence_groups") or [])
                if isinstance(item, Mapping)
            ],
            neutral_remainder_indices=[int(i) for i in (data.get("neutral_remainder_indices") or [])],
            neutral_remainder_charge=(
                int(data["neutral_remainder_charge"]) if data.get("neutral_remainder_charge") is not None else None
            ),
            raw=data,
        )

    @classmethod
    def from_mol(cls, mol: Any) -> "RespConstraintMetadata":
        return cls.from_dict(read_json_prop(mol, RESP_CONSTRAINTS_PROP, default={}))

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.raw)
        out.setdefault("schema_version", self.schema_version)
        if self.mode is not None:
            out["mode"] = self.mode
        if self.resp_profile is not None:
            out["resp_profile"] = self.resp_profile
        out["equivalence_groups"] = [list(group) for group in self.equivalence_groups]
        if self.charged_group_constraints:
            out["charged_group_constraints"] = [dict(item) for item in self.charged_group_constraints]
        if self.charged_region_equivalence_groups:
            out["charged_region_equivalence_groups"] = [dict(item) for item in self.charged_region_equivalence_groups]
        if self.neutral_remainder_indices:
            out["neutral_remainder_indices"] = list(self.neutral_remainder_indices)
        if self.neutral_remainder_charge is not None:
            out["neutral_remainder_charge"] = int(self.neutral_remainder_charge)
        return stable_payload(out)

    def apply_to_mol(self, mol: Any) -> None:
        write_json_prop(mol, RESP_CONSTRAINTS_PROP, self.to_dict())
        write_text_prop(mol, RESP_PROFILE_PROP, self.resp_profile)


@dataclass(frozen=True)
class ChargeMetadata:
    """Typed view of charge-model provenance attached to a molecule."""

    charge_model: str | None = None
    resp_profile: str | None = None
    qm_recipe: dict[str, Any] = field(default_factory=dict)
    resp_constraints: RespConstraintMetadata = field(default_factory=RespConstraintMetadata)
    psiresp_constraints: dict[str, Any] = field(default_factory=dict)
    source_kind: str | None = None
    schema_version: str = METADATA_SCHEMA_VERSION

    @classmethod
    def from_mol(cls, mol: Any) -> "ChargeMetadata":
        recipe = read_json_prop(mol, QM_RECIPE_PROP, default={}) or {}
        constraints = RespConstraintMetadata.from_mol(mol)
        profile = read_text_prop(mol, RESP_PROFILE_PROP)
        if not profile:
            profile = constraints.resp_profile
        if not profile and isinstance(recipe, Mapping):
            raw_profile = recipe.get("resp_profile")
            profile = str(raw_profile).strip().lower() if raw_profile else None
        return cls(
            charge_model=(
                str(recipe.get("charge_model")).strip()
                if isinstance(recipe, Mapping) and recipe.get("charge_model")
                else None
            ),
            resp_profile=profile,
            qm_recipe=dict(recipe) if isinstance(recipe, Mapping) else {},
            resp_constraints=constraints,
            psiresp_constraints=dict(read_json_prop(mol, PSIRESP_CONSTRAINTS_PROP, default={}) or {}),
            source_kind=read_text_prop(mol, "_yadonpy_charge_loaded_from"),
        )

    def to_dict(self) -> dict[str, Any]:
        return stable_payload(
            {
                "schema_version": self.schema_version,
                "charge_model": self.charge_model,
                "resp_profile": self.resp_profile,
                "qm_recipe": dict(self.qm_recipe),
                "resp_constraints": self.resp_constraints.to_dict(),
                "psiresp_constraints": dict(self.psiresp_constraints),
                "source_kind": self.source_kind,
            }
        )

    def apply_to_mol(self, mol: Any) -> None:
        write_text_prop(mol, RESP_PROFILE_PROP, self.resp_profile)
        if self.qm_recipe:
            write_json_prop(mol, QM_RECIPE_PROP, self.qm_recipe)
        if self.resp_constraints.raw or self.resp_constraints.equivalence_groups:
            self.resp_constraints.apply_to_mol(mol)
        if self.psiresp_constraints:
            write_json_prop(mol, PSIRESP_CONSTRAINTS_PROP, self.psiresp_constraints)


@dataclass(frozen=True)
class SystemMeta:
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SystemMeta":
        data = dict(payload or {})
        return cls(
            payload=data,
            schema_version=(str(data.get("schema_version")) if data.get("schema_version") else None),
        )

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.payload)
        if self.schema_version:
            out.setdefault("schema_version", self.schema_version)
        return stable_payload(out)


@dataclass(frozen=True)
class EquilibrationState:
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = EQUILIBRIUM_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "EquilibrationState":
        data = dict(payload or {})
        return cls(payload=data, schema_version=str(data.get("schema_version") or EQUILIBRIUM_SCHEMA_VERSION))

    @property
    def ok(self) -> bool:
        return bool(self.payload.get("ok"))

    def to_dict(self) -> dict[str, Any]:
        return schema_stamp(self.payload, schema_version=self.schema_version, kind="equilibration_state")


@dataclass(frozen=True)
class BenchmarkSummary:
    payload: dict[str, Any] = field(default_factory=dict)
    summary_kind: str = "benchmark_summary"
    schema_version: str = BENCHMARK_SUMMARY_SCHEMA_VERSION

    @classmethod
    def for_path(cls, path: str | Path, payload: Mapping[str, Any]) -> "BenchmarkSummary":
        name = Path(path).name
        if name == "species_forcefield_summary.json":
            return cls(
                dict(payload),
                summary_kind="species_forcefield_summary",
                schema_version=SPECIES_FORCEFIELD_SUMMARY_SCHEMA_VERSION,
            )
        if name == "benchmark_summary.json":
            return cls(dict(payload), summary_kind="benchmark_summary", schema_version=BENCHMARK_SUMMARY_SCHEMA_VERSION)
        return cls(dict(payload), summary_kind=Path(path).stem, schema_version=BENCHMARK_SUMMARY_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, Any]:
        return schema_stamp(self.payload, schema_version=self.schema_version, kind=self.summary_kind)


__all__ = [
    "BENCHMARK_SUMMARY_SCHEMA_VERSION",
    "BenchmarkSummary",
    "ChargeMetadata",
    "EQUILIBRIUM_SCHEMA_VERSION",
    "PSIRESP_CONSTRAINTS_PROP",
    "QM_RECIPE_PROP",
    "RESP_CONSTRAINTS_PROP",
    "RESP_PROFILE_PROP",
    "RespConstraintMetadata",
    "SPECIES_FORCEFIELD_SUMMARY_SCHEMA_VERSION",
    "SystemMeta",
    "EquilibrationState",
    "read_json_prop",
    "read_text_prop",
    "schema_stamp",
    "write_json_prop",
    "write_schema_json",
    "write_text_prop",
]
