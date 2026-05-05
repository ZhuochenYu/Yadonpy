from __future__ import annotations

"""Refresh existing repo MolDB entries with fully reoptimized adaptive RESP.

This script is intentionally more conservative than the lightweight Example 07
builder:

* it only targets species that already exist in the repository MolDB;
* workers write candidate molecules into a temporary candidate MolDB first;
* the repo MolDB is hard-replaced only after every target succeeds;
* old/new charge differences are written before the final replacement.

The default route performs a fresh DFT geometry optimization plus adaptive RESP
for every selected species. Use ``--reuse-geometry`` only for debugging.
"""

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yadonpy as yp
from yadonpy.core import chem_utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.core.polyelectrolyte import detect_charged_groups
from yadonpy.diagnostics import doctor
from yadonpy.moldb import MolDB
from yadonpy.moldb.store import canonical_key
from yadonpy.runtime import set_run_options


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CATALOG_CSV = HERE / "electrolyte_species.csv"

DEFAULT_TARGETS = (
    "PAA",
    "glucose_2",
    "glucose_3",
    "glucose_6",
    "glucose_23",
    "glucose_26",
    "glucose_36",
    "glucose_236",
    "EC",
    "EMC",
    "DEC",
)

@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    smiles: str
    kind: str
    charge: str
    bonded: str | None
    polyelectrolyte_mode: bool


@dataclass(frozen=True)
class RefreshTask:
    name: str
    smiles: str
    kind: str
    charge: str
    bonded: str | None
    polyelectrolyte_mode: bool
    profile: str
    priority: int
    heavy_atoms: int
    formal_charge: int
    required_cores: int
    psi4_omp: int
    memory_mb: int


def _csv_bool(value: object, *, default: bool = False) -> bool:
    token = str(value or "").strip().lower()
    if not token:
        return bool(default)
    return token in {"1", "true", "t", "yes", "y", "on"}


def _read_catalog(path: Path) -> dict[str, SpeciesSpec]:
    out: dict[str, SpeciesSpec] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            name = str(raw.get("name") or "").strip()
            smiles = str(raw.get("smiles") or "").strip()
            if not name or not smiles:
                continue
            out[name] = SpeciesSpec(
                name=name,
                smiles=smiles,
                kind=str(raw.get("kind") or ("psmiles" if "*" in smiles else "smiles")).strip(),
                charge=str(raw.get("charge") or "RESP").strip().upper(),
                bonded=(str(raw.get("bonded") or "").strip() or None),
                polyelectrolyte_mode=_csv_bool(raw.get("polyelectrolyte_mode"), default=False),
            )
    return out


def _env_default(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    return int(raw) if raw else int(default)


def _parse_only(tokens: list[str] | None) -> list[str]:
    if not tokens:
        env = str(os.environ.get("YADONPY_ONLY", "")).strip()
        tokens = [env] if env else []
    seen: list[str] = []
    for token in tokens:
        for part in str(token).split(","):
            name = part.strip()
            if name and name not in seen:
                seen.append(name)
    return seen


def _formal_charge(mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def _heavy_atom_count(mol) -> int:
    return int(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1))


def _profile_variant_ready(db: MolDB, spec: SpeciesSpec, *, resp_profile: str) -> bool:
    profile = str(resp_profile or "adaptive").strip().lower()
    try:
        _kind, _canonical, key = canonical_key(spec.smiles)
        rec = db.load_record(key)
    except Exception:
        return False
    if rec is None:
        return False
    for meta in (rec.variants or {}).values():
        if not isinstance(meta, dict):
            continue
        if str(meta.get("charge") or "RESP").upper() != "RESP":
            continue
        if str(meta.get("resp_profile") or "").strip().lower() != profile:
            continue
        if bool(meta.get("polyelectrolyte_mode", False)) != bool(spec.polyelectrolyte_mode):
            continue
        if bool(meta.get("ready", False)):
            return True
    return False


def _record_exists(db: MolDB, spec: SpeciesSpec) -> bool:
    try:
        _kind, _canonical, key = canonical_key(spec.smiles)
        return db.load_record(key) is not None
    except Exception:
        return False


def _selected_specs(
    *,
    catalog: dict[str, SpeciesSpec],
    repo_db: MolDB,
    only: list[str],
    target_mode: str,
) -> list[SpeciesSpec]:
    if only:
        missing = [name for name in only if name not in catalog]
        if missing:
            raise SystemExit(f"Unknown species in electrolyte_species.csv: {missing}")
        return [catalog[name] for name in only if _record_exists(repo_db, catalog[name])]
    mode = str(target_mode or "existing-repo").strip().lower()
    if mode == "default-targets":
        return [catalog[name] for name in DEFAULT_TARGETS if name in catalog and _record_exists(repo_db, catalog[name])]
    if mode != "existing-repo":
        raise SystemExit(f"Unsupported target mode: {target_mode!r}")
    return [spec for spec in catalog.values() if _record_exists(repo_db, spec)]


def _load_geometry(spec: SpeciesSpec, *, dbs: list[tuple[str, MolDB]], resp_profile: str) -> tuple[Any, str, bool]:
    for label, db in dbs:
        for profile in (resp_profile, "legacy", None):
            kwargs: dict[str, Any] = {
                "require_ready": False,
                "charge": "RESP",
                "polyelectrolyte_mode": spec.polyelectrolyte_mode,
                "polyelectrolyte_detection": "auto",
            }
            if profile is not None:
                kwargs["resp_profile"] = profile
            try:
                mol, _rec = db.load_mol(spec.smiles, **kwargs)
                return mol, f"{label}-db-{profile or 'any'}", False
            except Exception:
                continue
    return yp.mol_from_smiles(spec.smiles, name=spec.name), "fresh-smiles", True


def _charge_values(mol, prop: str = "AtomicCharge") -> list[float | None]:
    values: list[float | None] = []
    for atom in mol.GetAtoms():
        if atom.HasProp(prop):
            values.append(float(atom.GetDoubleProp(prop)))
        else:
            values.append(None)
    return values


def _net_charge(charges: list[float | None]) -> float | None:
    if any(value is None for value in charges):
        return None
    return float(sum(float(value) for value in charges if value is not None))


def _group_spreads(mol, groups: list[list[int]], prop: str = "AtomicCharge") -> list[dict[str, Any]]:
    values = _charge_values(mol, prop=prop)
    spreads: list[dict[str, Any]] = []
    for group in groups:
        idxs = sorted({int(i) for i in group})
        qs = [values[i] for i in idxs if 0 <= i < len(values) and values[i] is not None]
        if len(qs) != len(idxs) or len(qs) <= 1:
            continue
        spreads.append(
            {
                "indices": idxs,
                "symbols": [mol.GetAtomWithIdx(i).GetSymbol() for i in idxs],
                "charges": [float(q) for q in qs],
                "spread": float(max(qs) - min(qs)),
            }
        )
    return spreads


def _carboxylate_spreads(mol, prop: str = "AtomicCharge") -> list[dict[str, Any]]:
    values = _charge_values(mol, prop=prop)
    out: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        oxygen_atoms = [bond.GetOtherAtom(atom) for bond in atom.GetBonds() if bond.GetOtherAtom(atom).GetSymbol() == "O"]
        if len(oxygen_atoms) != 2:
            continue
        oxy_idxs = sorted(int(oxygen.GetIdx()) for oxygen in oxygen_atoms)
        if not any(mol.GetAtomWithIdx(i).GetFormalCharge() < 0 for i in oxy_idxs):
            continue
        qs = [values[i] for i in oxy_idxs if values[i] is not None]
        if len(qs) != 2:
            continue
        out.append(
            {
                "carbon": int(atom.GetIdx()),
                "oxygen_indices": oxy_idxs,
                "charges": [float(q) for q in qs],
                "spread": float(max(qs) - min(qs)),
            }
        )
    return out


def _validate_equivalence(mol, *, tolerance: float) -> dict[str, Any]:
    groups = chem_utils.resp_equivalence_groups_from_mol(mol)
    spreads = _group_spreads(mol, groups)
    carboxylates = _carboxylate_spreads(mol)
    max_group_spread = max([item["spread"] for item in spreads] or [0.0])
    max_carboxylate_spread = max([item["spread"] for item in carboxylates] or [0.0])
    ok = max(max_group_spread, max_carboxylate_spread) <= float(tolerance)
    return {
        "ok": bool(ok),
        "tolerance": float(tolerance),
        "equivalence_group_count": len(groups),
        "max_equivalence_spread": float(max_group_spread),
        "max_carboxylate_oxygen_spread": float(max_carboxylate_spread),
        "equivalence_spreads": spreads,
        "carboxylate_spreads": carboxylates,
    }


def _old_resp_snapshot(db: MolDB, spec: SpeciesSpec) -> dict[str, Any]:
    _kind, _canonical, key = canonical_key(spec.smiles)
    rec = db.load_record(key)
    variants = dict((rec.variants or {}) if rec is not None else {})
    mol, _rec = db.load_mol(
        spec.smiles,
        require_ready=True,
        charge="RESP",
        polyelectrolyte_mode=spec.polyelectrolyte_mode,
        polyelectrolyte_detection="auto",
    )
    charges = _charge_values(mol)
    resp_variant_ids = [
        vid
        for vid, meta in sorted(variants.items())
        if isinstance(meta, dict) and str(meta.get("charge") or "RESP").strip().upper() == "RESP"
    ]
    adaptive_ids = [
        vid
        for vid, meta in sorted(variants.items())
        if isinstance(meta, dict)
        and str(meta.get("charge") or "RESP").strip().upper() == "RESP"
        and str(meta.get("resp_profile") or "").strip().lower() == "adaptive"
    ]
    return {
        "key": key,
        "name": rec.name if rec is not None else spec.name,
        "variant_ids": list(variants.keys()),
        "resp_variant_ids": resp_variant_ids,
        "adaptive_resp_variant_ids": adaptive_ids,
        "selected_resp_profile": "adaptive" if adaptive_ids else "legacy",
        "charges": charges,
        "net_charge": _net_charge(charges),
        "manifest": rec.to_dict() if rec is not None else {},
    }


def _charge_diff(
    *,
    spec: SpeciesSpec,
    old_snapshot: dict[str, Any],
    new_mol,
    validation: dict[str, Any],
    new_variant_id: str | None,
) -> dict[str, Any]:
    old = list(old_snapshot.get("charges") or [])
    new = _charge_values(new_mol)
    if len(old) != len(new):
        raise RuntimeError(f"{spec.name} atom-count mismatch in charge diff: old={len(old)} new={len(new)}")
    deltas: list[float] = []
    per_atom: list[dict[str, Any]] = []
    for idx, (old_q, new_q) in enumerate(zip(old, new)):
        if old_q is None or new_q is None:
            delta = None
            abs_delta = None
        else:
            delta = float(new_q) - float(old_q)
            abs_delta = abs(delta)
            deltas.append(float(delta))
        per_atom.append(
            {
                "species": spec.name,
                "atom_index": int(idx),
                "symbol": new_mol.GetAtomWithIdx(idx).GetSymbol(),
                "old_charge": old_q,
                "new_charge": new_q,
                "delta": delta,
                "abs_delta": abs_delta,
            }
        )
    abs_deltas = [abs(x) for x in deltas]
    rms_delta = math.sqrt(sum(x * x for x in deltas) / len(deltas)) if deltas else None
    summary = {
        "species": spec.name,
        "smiles": spec.smiles,
        "atom_count": len(new),
        "formal_charge": _formal_charge(new_mol),
        "old_net_charge": old_snapshot.get("net_charge"),
        "new_net_charge": _net_charge(new),
        "old_resp_profile": old_snapshot.get("selected_resp_profile"),
        "new_resp_profile": "adaptive",
        "old_resp_variant_ids": old_snapshot.get("resp_variant_ids") or [],
        "new_resp_variant_id": new_variant_id,
        "max_abs_delta": max(abs_deltas) if abs_deltas else None,
        "mean_abs_delta": (sum(abs_deltas) / len(abs_deltas)) if abs_deltas else None,
        "rms_delta": rms_delta,
        "equivalence_ok": bool(validation.get("ok", False)),
        "max_equivalence_spread": validation.get("max_equivalence_spread"),
        "max_carboxylate_oxygen_spread": validation.get("max_carboxylate_oxygen_spread"),
    }
    return {"summary": summary, "per_atom": per_atom}


def _available_cpu_total() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, int(os.cpu_count() or 1))


def _available_memory_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return max(1024, int(line.split()[1]) // 1024)
    except Exception:
        pass
    return 96000


def _task_profile(spec: SpeciesSpec) -> tuple[str, int]:
    if str(spec.bonded or "").strip().upper() == "DRIH":
        return "drih", 0
    if bool(spec.polyelectrolyte_mode):
        return "polyelectrolyte", 1
    if "*" in str(spec.smiles):
        return "polymer", 2
    return "standard", 3


def _task_omp(*, profile: str, heavy_atoms: int, cpu_budget: int, max_omp: int) -> int:
    max_allowed = max(1, min(int(cpu_budget), int(max_omp)))
    if profile in {"drih", "polyelectrolyte"}:
        requested = 8 if heavy_atoms < 35 else 12
    elif profile == "polymer":
        requested = 6 if heavy_atoms < 35 else 8
    else:
        requested = 4 if heavy_atoms < 25 else 6
    return max(1, min(max_allowed, requested))


def _task_memory_mb(*, profile: str, heavy_atoms: int, max_memory_mb: int) -> int:
    base = 8000 + 220 * int(heavy_atoms)
    if profile in {"drih", "polyelectrolyte"}:
        base += 4000
    if heavy_atoms >= 45:
        base += 4000
    return max(6000, min(int(max_memory_mb), int(base)))


def _build_refresh_tasks(
    specs: list[SpeciesSpec],
    *,
    cpu_budget: int,
    max_omp: int,
    max_memory_mb: int,
) -> list[RefreshTask]:
    tasks: list[RefreshTask] = []
    for spec in specs:
        mol = yp.mol_from_smiles(spec.smiles, coord=False, name=spec.name)
        heavy_atoms = _heavy_atom_count(mol)
        formal_charge = _formal_charge(mol)
        profile, priority = _task_profile(spec)
        omp = _task_omp(profile=profile, heavy_atoms=heavy_atoms, cpu_budget=cpu_budget, max_omp=max_omp)
        memory_mb = _task_memory_mb(profile=profile, heavy_atoms=heavy_atoms, max_memory_mb=max_memory_mb)
        tasks.append(
            RefreshTask(
                name=spec.name,
                smiles=spec.smiles,
                kind=spec.kind,
                charge=spec.charge,
                bonded=spec.bonded,
                polyelectrolyte_mode=bool(spec.polyelectrolyte_mode),
                profile=profile,
                priority=priority,
                heavy_atoms=heavy_atoms,
                formal_charge=formal_charge,
                required_cores=omp,
                psi4_omp=omp,
                memory_mb=memory_mb,
            )
        )
    tasks.sort(key=lambda task: (task.priority, -task.heavy_atoms, task.name.lower()))
    return tasks


def _pending_payloads(tasks: list[RefreshTask]) -> list[dict[str, Any]]:
    pending = [dict(asdict(task), attempt=1, max_attempts=2) for task in tasks]
    _sort_pending_in_place(pending)
    return pending


def _sort_pending_in_place(pending: list[dict[str, Any]]) -> None:
    pending.sort(
        key=lambda item: (
            int(item["priority"]),
            -int(item["required_cores"]),
            -int(item.get("memory_mb", 0)),
            str(item["name"]).lower(),
        )
    )


def _eligible_pending_for_launch(pending: list[dict[str, Any]], available_cores: int, available_memory_mb: int) -> list[dict[str, Any]]:
    if not pending:
        return []
    priorities = sorted({int(item["priority"]) for item in pending})
    for priority in priorities:
        same_priority = [item for item in pending if int(item["priority"]) == priority]
        fitting = [
            item
            for item in same_priority
            if int(item["required_cores"]) <= int(available_cores)
            and int(item.get("memory_mb", 0)) <= int(available_memory_mb)
        ]
        if fitting:
            return fitting
    return [
        item
        for item in pending
        if int(item["required_cores"]) <= int(available_cores)
        and int(item.get("memory_mb", 0)) <= int(available_memory_mb)
    ]


def _maybe_schedule_retry(task: dict[str, Any], *, error: str) -> dict[str, Any] | None:
    attempt = int(task.get("attempt", 1))
    max_attempts = int(task.get("max_attempts", 2))
    current_cores = max(1, int(task["required_cores"]))
    if attempt >= max_attempts or current_cores <= 1:
        return None
    retried = dict(task)
    retried["attempt"] = attempt + 1
    retried["required_cores"] = max(1, current_cores // 2)
    retried["psi4_omp"] = max(1, current_cores // 2)
    retried["retry_of_cores"] = current_cores
    retried["retry_reason"] = error
    retry_geom_iter = int(task.get("retry_geom_iter", 0) or 0)
    if retry_geom_iter > int(task.get("geom_iter", 50) or 50):
        retried["geom_iter"] = retry_geom_iter
    return retried


def _worker_refresh_candidate(
    *,
    task_payload: dict[str, Any],
    repo_db_dir: str,
    candidate_db_dir: str,
    job_root: str,
    resp_profile: str,
    tolerance: float,
    reuse_geometry: bool,
    result_queue,
) -> None:
    try:
        os.environ["OMP_NUM_THREADS"] = str(int(task_payload["psi4_omp"]))
        os.environ["YADONPY_OMP_PSI4"] = str(int(task_payload["psi4_omp"]))
        set_run_options(restart=False)
        spec = SpeciesSpec(
            name=str(task_payload["name"]),
            smiles=str(task_payload["smiles"]),
            kind=str(task_payload["kind"]),
            charge=str(task_payload["charge"]),
            bonded=task_payload.get("bonded"),
            polyelectrolyte_mode=bool(task_payload["polyelectrolyte_mode"]),
        )
        repo_db = MolDB(Path(repo_db_dir))
        candidate_db = MolDB(Path(candidate_db_dir))
        old_snapshot = _old_resp_snapshot(repo_db, spec)
        mol, geometry_source, fresh = _load_geometry(spec, dbs=[("repo", repo_db)], resp_profile=resp_profile)
        formal_charge = _formal_charge(mol)
        charge_groups = detect_charged_groups(mol, detection="auto", resp_profile=resp_profile)
        species_wd = workdir(Path(job_root) / spec.name / f"attempt_{int(task_payload.get('attempt', 1)):02d}", restart=False)
        optimize = not bool(reuse_geometry)
        print(
            f"[RUN] {spec.name:16s} charge={formal_charge:+d} profile={task_payload['profile']} "
            f"omp={task_payload['psi4_omp']} mem={task_payload['memory_mb']}MB "
            f"source={geometry_source} opt={int(optimize)}",
            flush=True,
        )
        ok = yp.assign_charges(
            mol,
            charge="RESP",
            resp_profile=resp_profile,
            opt=optimize,
            work_dir=species_wd,
            log_name=spec.name,
            omp=int(task_payload["psi4_omp"]),
            memory=int(task_payload["memory_mb"]),
            opt_method="wb97m-d3bj",
            charge_method="wb97m-d3bj",
            opt_basis="def2-SVP",
            charge_basis="def2-TZVP",
            auto_level=True,
            geom_iter=int(task_payload.get("geom_iter", 50) or 50),
            total_charge=formal_charge,
            total_multiplicity=1,
            polyelectrolyte_mode=spec.polyelectrolyte_mode,
            polyelectrolyte_detection="auto",
            bonded_params=(spec.bonded or "ff_assigned"),
            symmetrize=True,
        )
        if not ok:
            raise RuntimeError(f"assign_charges failed for {spec.name}")

        groups = chem_utils.resp_equivalence_groups_from_mol(mol)
        repaired_groups = chem_utils.symmetrize_equivalent_charge_props(mol, equivalence_groups=groups)
        validation = _validate_equivalence(mol, tolerance=tolerance)
        if not validation["ok"]:
            raise RuntimeError(
                f"{spec.name} equivalence validation failed: "
                f"max_group_spread={validation['max_equivalence_spread']:.3e}, "
                f"max_carboxylate_spread={validation['max_carboxylate_oxygen_spread']:.3e}"
            )

        rec = candidate_db.update_from_mol(
            mol,
            smiles_or_psmiles=spec.smiles,
            name=spec.name,
            charge="RESP",
            polyelectrolyte_mode=spec.polyelectrolyte_mode,
            polyelectrolyte_detection="auto",
            resp_profile=resp_profile,
        )
        payload = json.loads(candidate_db.charges_path(rec.key).read_text(encoding="utf-8"))
        new_variant_id = str(payload.get("variant_id") or "")
        diff = _charge_diff(
            spec=spec,
            old_snapshot=old_snapshot,
            new_mol=mol,
            validation=validation,
            new_variant_id=new_variant_id,
        )
        recipe = {}
        if mol.HasProp("_yadonpy_qm_recipe_json"):
            try:
                recipe = json.loads(mol.GetProp("_yadonpy_qm_recipe_json"))
            except Exception:
                recipe = {}
        result = {
            "name": spec.name,
            "smiles": spec.smiles,
            "status": "candidate_ready",
            "geometry_source": geometry_source,
            "fresh_geometry": bool(fresh),
            "optimized": bool(optimize),
            "formal_charge": int(formal_charge),
            "polyelectrolyte_mode": bool(spec.polyelectrolyte_mode),
            "charge_group_count": int(len(charge_groups.get("groups") or [])),
            "resp_profile": resp_profile,
            "qm_recipe": recipe,
            "repaired_equivalence_groups": int(repaired_groups),
            "validation": validation,
            "candidate_key": rec.key,
            "candidate_variant_id": new_variant_id,
            "diff_summary": diff["summary"],
            "diff_per_atom": diff["per_atom"],
            "attempt": int(task_payload.get("attempt", 1)),
            "psi4_omp": int(task_payload["psi4_omp"]),
            "memory_mb": int(task_payload["memory_mb"]),
        }
        result_queue.put({"name": spec.name, "ok": True, "result": result, "task": task_payload})
    except Exception as exc:
        result_queue.put({"name": str(task_payload.get("name")), "ok": False, "error": repr(exc), "task": task_payload})


def _is_resp_variant(meta: dict[str, Any] | None) -> bool:
    return isinstance(meta, dict) and str(meta.get("charge") or "RESP").strip().upper() == "RESP"


def _hard_replace_repo_record(
    *,
    repo_db: MolDB,
    candidate_db: MolDB,
    spec: SpeciesSpec,
    resp_profile: str,
) -> dict[str, Any]:
    candidate_mol, _candidate_rec = candidate_db.load_mol(
        spec.smiles,
        require_ready=True,
        charge="RESP",
        resp_profile=resp_profile,
        polyelectrolyte_mode=spec.polyelectrolyte_mode,
        polyelectrolyte_detection="auto",
    )
    rec = repo_db.update_from_mol(
        candidate_mol,
        smiles_or_psmiles=spec.smiles,
        name=spec.name,
        charge="RESP",
        polyelectrolyte_mode=spec.polyelectrolyte_mode,
        polyelectrolyte_detection="auto",
        resp_profile=resp_profile,
    )
    payload = json.loads(repo_db.charges_path(rec.key).read_text(encoding="utf-8"))
    keep_resp_vid = str(payload.get("variant_id") or "")
    removed_resp_variant_ids: list[str] = []
    kept: dict[str, dict[str, Any]] = {}
    for vid, meta in sorted((rec.variants or {}).items()):
        if _is_resp_variant(meta) and str(vid) != keep_resp_vid:
            removed_resp_variant_ids.append(str(vid))
            continue
        kept[str(vid)] = dict(meta)
    rec.variants = kept
    rec.charge_method = "RESP"
    rec.ready = True
    repo_db.save_record(rec)
    for vid in removed_resp_variant_ids:
        try:
            repo_db.charges_variant_path(rec.key, vid).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            bonded_dir = repo_db.bonded_variant_dir(rec.key, vid)
            if bonded_dir.exists():
                shutil.rmtree(bonded_dir)
        except Exception:
            pass
    return {
        "name": spec.name,
        "key": rec.key,
        "kept_resp_variant_id": keep_resp_vid,
        "removed_resp_variant_ids": removed_resp_variant_ids,
        "remaining_variant_ids": sorted((rec.variants or {}).keys()),
    }


def _make_repo_backup(repo_db: MolDB, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    src = Path(repo_db.db_dir).resolve()
    target = backup_dir / f"moldb_backup_pre_adaptive_resp_{stamp}.tar.zst"
    try:
        subprocess.run(
            ["tar", "--zstd", "-cf", str(target), "-C", str(src.parent), src.name],
            check=True,
            capture_output=True,
            text=True,
        )
        return target
    except Exception:
        fallback = backup_dir / f"moldb_backup_pre_adaptive_resp_{stamp}.tar.gz"
        subprocess.run(
            ["tar", "-czf", str(fallback), "-C", str(src.parent), src.name],
            check=True,
            capture_output=True,
            text=True,
        )
        return fallback


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_diff_outputs(out_dir: Path, *, results: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [dict(result["diff_summary"]) for result in results]
    per_atom_rows: list[dict[str, Any]] = []
    for result in results:
        per_atom_rows.extend(dict(row) for row in result.get("diff_per_atom") or [])

    summary_json = out_dir / "charge_diff_summary.json"
    per_atom_json = out_dir / "charge_diff_per_atom.json"
    failures_json = out_dir / "refresh_failures.json"
    summary_csv = out_dir / "charge_diff_summary.csv"
    per_atom_csv = out_dir / "charge_diff_per_atom.csv"
    summary_md = out_dir / "charge_diff_summary.md"

    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    per_atom_json.write_text(json.dumps(per_atom_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failures_json.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(
        summary_csv,
        summary_rows,
        [
            "species",
            "smiles",
            "atom_count",
            "formal_charge",
            "old_net_charge",
            "new_net_charge",
            "old_resp_profile",
            "new_resp_profile",
            "max_abs_delta",
            "mean_abs_delta",
            "rms_delta",
            "equivalence_ok",
            "max_equivalence_spread",
            "max_carboxylate_oxygen_spread",
            "new_resp_variant_id",
        ],
    )
    _write_csv(
        per_atom_csv,
        per_atom_rows,
        ["species", "atom_index", "symbol", "old_charge", "new_charge", "delta", "abs_delta"],
    )
    lines = [
        "# Adaptive RESP charge refresh diff",
        "",
        "| species | atoms | old profile | max | mean | rms | eq ok | carboxylate spread |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {species} | {atom_count} | {old_resp_profile} | {max_abs_delta:.6g} | {mean_abs_delta:.6g} | "
            "{rms_delta:.6g} | {equivalence_ok} | {max_carboxylate_oxygen_spread:.3g} |".format(
                species=row.get("species"),
                atom_count=row.get("atom_count"),
                old_resp_profile=row.get("old_resp_profile"),
                max_abs_delta=float(row.get("max_abs_delta") or 0.0),
                mean_abs_delta=float(row.get("mean_abs_delta") or 0.0),
                rms_delta=float(row.get("rms_delta") or 0.0),
                equivalence_ok=row.get("equivalence_ok"),
                max_carboxylate_oxygen_spread=float(row.get("max_carboxylate_oxygen_spread") or 0.0),
            )
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "per_atom_json": str(per_atom_json),
        "per_atom_csv": str(per_atom_csv),
        "failures_json": str(failures_json),
    }


def _candidate_result_from_db(
    *,
    repo_db: MolDB,
    candidate_db: MolDB,
    spec: SpeciesSpec,
    resp_profile: str,
    tolerance: float,
) -> dict[str, Any]:
    """Reconstruct a refresh result from an existing candidate MolDB record."""
    old_snapshot = _old_resp_snapshot(repo_db, spec)
    candidate_mol, rec = candidate_db.load_mol(
        spec.smiles,
        require_ready=True,
        charge="RESP",
        resp_profile=resp_profile,
        polyelectrolyte_mode=spec.polyelectrolyte_mode,
        polyelectrolyte_detection="auto",
    )
    validation = _validate_equivalence(candidate_mol, tolerance=tolerance)
    if not validation["ok"]:
        raise RuntimeError(
            f"{spec.name} equivalence validation failed: "
            f"max_group_spread={validation['max_equivalence_spread']:.3e}, "
            f"max_carboxylate_spread={validation['max_carboxylate_oxygen_spread']:.3e}"
        )
    try:
        payload = json.loads(candidate_db.charges_path(rec.key).read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    try:
        recipe = json.loads(candidate_mol.GetProp("_yadonpy_qm_recipe_json"))
    except Exception:
        recipe = {}
    charge_groups = detect_charged_groups(candidate_mol, detection="auto", resp_profile=resp_profile)
    diff = _charge_diff(
        spec=spec,
        old_snapshot=old_snapshot,
        new_mol=candidate_mol,
        validation=validation,
        new_variant_id=str(payload.get("variant_id") or ""),
    )
    return {
        "name": spec.name,
        "smiles": spec.smiles,
        "formal_charge": _formal_charge(candidate_mol),
        "polyelectrolyte_mode": bool(spec.polyelectrolyte_mode),
        "charge_group_count": int(len(charge_groups.get("groups") or [])),
        "resp_profile": resp_profile,
        "qm_recipe": recipe,
        "repaired_equivalence_groups": 0,
        "validation": validation,
        "candidate_key": rec.key,
        "candidate_variant_id": str(payload.get("variant_id") or ""),
        "diff_summary": diff["summary"],
        "diff_per_atom": diff["per_atom"],
        "attempt": None,
        "psi4_omp": None,
        "memory_mb": None,
        "source": "candidate_moldb",
    }


def _finalize_candidate_moldb(
    *,
    repo_db: MolDB,
    candidate_db_dir: Path,
    specs: list[SpeciesSpec],
    resp_profile: str,
    tolerance: float,
    diff_dir: Path,
    no_commit: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Validate candidate records, write diffs, and optionally hard-replace repo RESP variants."""
    candidate_db = MolDB(candidate_db_dir)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in specs:
        try:
            results.append(
                _candidate_result_from_db(
                    repo_db=repo_db,
                    candidate_db=candidate_db,
                    spec=spec,
                    resp_profile=resp_profile,
                    tolerance=float(tolerance),
                )
            )
            print(f"[CANDIDATE] {spec.name:16s} ready", flush=True)
        except Exception as exc:
            failures.append({"name": spec.name, "smiles": spec.smiles, "error": repr(exc)})
            print(f"[MISSING]   {spec.name:16s} {exc!r}", flush=True)

    diff_outputs = _write_diff_outputs(diff_dir, results=results, failures=failures)
    replacements: list[dict[str, Any]] = []
    if failures:
        print("[STOP] Candidate MolDB is incomplete/invalid; repo MolDB was not modified.", flush=True)
        return results, failures, diff_outputs, replacements
    if no_commit:
        print("[STOP] --no-commit requested; candidate MolDB and charge diffs are ready, repo MolDB unchanged.", flush=True)
        return results, failures, diff_outputs, replacements

    for spec in specs:
        replacements.append(
            _hard_replace_repo_record(
                repo_db=repo_db,
                candidate_db=candidate_db,
                spec=spec,
                resp_profile=resp_profile,
            )
        )
        print(f"[REPLACE] {spec.name:16s} RESP variants hard-replaced", flush=True)
    return results, failures, diff_outputs, replacements


def _run_parallel_refresh(
    *,
    tasks: list[RefreshTask],
    repo_db_dir: Path,
    candidate_db_dir: Path,
    job_root: Path,
    resp_profile: str,
    tolerance: float,
    reuse_geometry: bool,
    geom_iter: int,
    retry_geom_iter: int,
    planner_cpu_budget: int,
    planner_memory_budget_mb: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    pending = _pending_payloads(tasks)
    for item in pending:
        item["geom_iter"] = int(geom_iter)
        item["retry_geom_iter"] = int(retry_geom_iter)
    running: dict[str, dict[str, Any]] = {}
    available_cores = int(planner_cpu_budget)
    available_memory_mb = int(planner_memory_budget_mb)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    retry_count = 0

    while pending or running:
        launched = False
        for task in list(_eligible_pending_for_launch(pending, available_cores, available_memory_mb)):
            required = int(task["required_cores"])
            memory_mb = int(task["memory_mb"])
            if required > available_cores or memory_mb > available_memory_mb:
                continue
            proc = ctx.Process(
                target=_worker_refresh_candidate,
                kwargs={
                    "task_payload": task,
                    "repo_db_dir": str(repo_db_dir),
                    "candidate_db_dir": str(candidate_db_dir),
                    "job_root": str(job_root),
                    "resp_profile": resp_profile,
                    "tolerance": float(tolerance),
                    "reuse_geometry": bool(reuse_geometry),
                    "result_queue": result_queue,
                },
            )
            proc.start()
            running[str(task["name"])] = {
                "process": proc,
                "required_cores": required,
                "memory_mb": memory_mb,
                "task": task,
            }
            available_cores -= required
            available_memory_mb -= memory_mb
            pending.remove(task)
            launched = True
            print(
                f"[START] {task['name']:16s} profile={task['profile']:15s} "
                f"attempt={task['attempt']}/{task['max_attempts']} "
                f"cores={required:2d} mem={memory_mb:5d}MB "
                f"available={available_cores:2d}/{available_memory_mb:5d}MB",
                flush=True,
            )

        try:
            message = result_queue.get(timeout=0.5)
        except queue.Empty:
            message = None

        if message is not None:
            name = str(message["name"])
            state = running.pop(name, None)
            if state is not None:
                state["process"].join(timeout=1.0)
                available_cores += int(state["required_cores"])
                available_memory_mb += int(state["memory_mb"])
            if bool(message.get("ok")):
                result = dict(message["result"])
                results.append(result)
                print(
                    f"[DONE]  {name:16s} available={available_cores:2d}/{available_memory_mb:5d}MB "
                    f"attempt={result.get('attempt')}",
                    flush=True,
                )
            else:
                retry_task = _maybe_schedule_retry(state["task"], error=str(message.get("error"))) if state else None
                if retry_task is not None:
                    retry_count += 1
                    pending.append(retry_task)
                    _sort_pending_in_place(pending)
                    print(
                        f"[RETRY] {name:16s} failed; retrying with {retry_task['required_cores']} cores "
                        f"(attempt {retry_task['attempt']}/{retry_task['max_attempts']})",
                        flush=True,
                    )
                else:
                    failures.append(
                        {
                            "name": name,
                            "smiles": (state["task"]["smiles"] if state else None),
                            "attempt": int((state or {}).get("task", {}).get("attempt", message.get("attempt", 1))),
                            "error": message.get("error"),
                        }
                    )
                    print(f"[FAIL]  {name:16s} {message.get('error')}", flush=True)

        for name, state in list(running.items()):
            proc = state["process"]
            if proc.is_alive():
                continue
            proc.join(timeout=0.1)
            running.pop(name, None)
            available_cores += int(state["required_cores"])
            available_memory_mb += int(state["memory_mb"])
            error = f"worker exited without reporting (exitcode={proc.exitcode})"
            retry_task = _maybe_schedule_retry(state["task"], error=error)
            if retry_task is not None:
                retry_count += 1
                pending.append(retry_task)
                _sort_pending_in_place(pending)
                print(f"[RETRY] {name:16s} {error}; retrying", flush=True)
            else:
                failures.append(
                    {
                        "name": name,
                        "smiles": state["task"]["smiles"],
                        "attempt": int(state["task"].get("attempt", 1)),
                        "error": error,
                    }
                )
                print(f"[FAIL]  {name:16s} {error}", flush=True)

        if not launched and not running and pending:
            smallest = min(pending, key=lambda item: (int(item["required_cores"]), int(item.get("memory_mb", 0))))
            if (
                int(smallest["required_cores"]) <= int(available_cores)
                and int(smallest.get("memory_mb", 0)) <= int(available_memory_mb)
            ):
                # A just-finished worker may have freed enough resources after
                # the launch pass at the top of this loop. Continue so the next
                # iteration can launch the now-eligible task instead of raising
                # a false resource-budget error.
                continue
            raise RuntimeError(
                "No pending RESP refresh task fits the available resource budget. "
                f"Smallest pending task is {smallest['name']} requiring "
                f"{smallest['required_cores']} cores / {smallest.get('memory_mb')} MB, "
                f"but only {available_cores} cores / {available_memory_mb} MB are available. "
                "Reduce reserve resources or increase the job memory budget."
            )
        if not launched and running:
            time.sleep(0.1)

    results.sort(key=lambda item: str(item.get("name", "")).lower())
    failures.sort(key=lambda item: str(item.get("name", "")).lower())
    return results, failures, retry_count


def refresh_one(
    spec: SpeciesSpec,
    *,
    default_db: MolDB | None,
    repo_db: MolDB,
    job_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Backward-compatible single-species candidate refresh helper.

    The new production path uses the candidate-DB worker above. This wrapper is
    kept for older tests and ad-hoc debugging, but now writes only to ``repo_db``.
    """

    _ = default_db
    candidate_db = repo_db
    old_snapshot = _old_resp_snapshot(repo_db, spec)
    mol, geometry_source, fresh = _load_geometry(spec, dbs=[("repo", repo_db)], resp_profile=args.resp_profile)
    formal_charge = _formal_charge(mol)
    charge_groups = detect_charged_groups(mol, detection="auto", resp_profile=args.resp_profile)
    needs_new_opt = not bool(getattr(args, "reuse_geometry", False) or getattr(args, "no_opt", False))
    print(
        f"[RUN] {spec.name:12s} charge={formal_charge:+d} "
        f"poly={int(spec.polyelectrolyte_mode)} groups={len(charge_groups.get('groups') or [])} "
        f"source={geometry_source} opt={int(needs_new_opt)}"
    )
    ok = yp.assign_charges(
        mol,
        charge="RESP",
        resp_profile=args.resp_profile,
        opt=needs_new_opt,
        work_dir=workdir(job_root / spec.name, restart=False),
        log_name=spec.name,
        omp=args.omp,
        memory=args.memory_mb,
        opt_method="wb97m-d3bj",
        charge_method="wb97m-d3bj",
        opt_basis="def2-SVP",
        charge_basis="def2-TZVP",
        auto_level=True,
        total_charge=formal_charge,
        total_multiplicity=1,
        polyelectrolyte_mode=spec.polyelectrolyte_mode,
        polyelectrolyte_detection="auto",
        bonded_params=(spec.bonded or "ff_assigned"),
        symmetrize=True,
    )
    if not ok:
        raise RuntimeError(f"assign_charges failed for {spec.name}")
    groups = chem_utils.resp_equivalence_groups_from_mol(mol)
    repaired_groups = chem_utils.symmetrize_equivalent_charge_props(mol, equivalence_groups=groups)
    validation = _validate_equivalence(mol, tolerance=args.tolerance)
    if not validation["ok"]:
        raise RuntimeError(f"{spec.name} equivalence validation failed")
    rec = candidate_db.update_from_mol(
        mol,
        smiles_or_psmiles=spec.smiles,
        name=spec.name,
        charge="RESP",
        polyelectrolyte_mode=spec.polyelectrolyte_mode,
        polyelectrolyte_detection="auto",
        resp_profile=args.resp_profile,
    )
    payload = json.loads(candidate_db.charges_path(rec.key).read_text(encoding="utf-8"))
    diff = _charge_diff(
        spec=spec,
        old_snapshot=old_snapshot,
        new_mol=mol,
        validation=validation,
        new_variant_id=str(payload.get("variant_id") or ""),
    )
    return {
        "name": spec.name,
        "smiles": spec.smiles,
        "status": "refreshed",
        "geometry_source": geometry_source,
        "fresh_geometry": bool(fresh),
        "optimized": bool(needs_new_opt),
        "formal_charge": int(formal_charge),
        "polyelectrolyte_mode": bool(spec.polyelectrolyte_mode),
        "charge_group_count": int(len(charge_groups.get("groups") or [])),
        "resp_profile": args.resp_profile,
        "repaired_equivalence_groups": int(repaired_groups),
        "validation": validation,
        "candidate_key": rec.key,
        "candidate_variant_id": str(payload.get("variant_id") or ""),
        "diff_summary": diff["summary"],
        "diff_per_atom": diff["per_atom"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Species names or comma-separated name lists.")
    parser.add_argument("--target-mode", choices=("existing-repo", "default-targets"), default="existing-repo")
    parser.add_argument("--repo-db", type=Path, default=REPO_ROOT / "moldb")
    parser.add_argument("--work-dir", type=Path, default=HERE / "work_dir" / "04_refresh_adaptive_resp_moldb")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--resp-profile", choices=("adaptive", "legacy"), default="adaptive")
    parser.add_argument("--max-omp", type=int, default=_env_default("YADONPY_REFRESH_MAX_OMP", 12))
    parser.add_argument("--reserve-cores", type=int, default=_env_default("YADONPY_REFRESH_RESERVE_CORES", 4))
    parser.add_argument("--reserve-memory-mb", type=int, default=_env_default("YADONPY_REFRESH_RESERVE_MEMORY_MB", 16000))
    parser.add_argument("--max-memory-mb", type=int, default=_env_default("YADONPY_REFRESH_MAX_MEMORY_MB", 24000))
    parser.add_argument("--omp", type=int, default=_env_default("YADONPY_PSI4_OMP", 8), help="Single-species compatibility mode OMP.")
    parser.add_argument("--memory-mb", type=int, default=_env_default("YADONPY_PSI4_MEMORY_MB", 12000), help="Single-species compatibility mode memory.")
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--reuse-geometry", action="store_true", help="Debug only: skip DFT optimization and refit RESP on existing geometry.")
    parser.add_argument("--force-opt", action="store_true", help="Compatibility flag; full refresh already optimizes by default.")
    parser.add_argument("--no-opt", action="store_true", help="Alias for --reuse-geometry.")
    parser.add_argument("--geom-iter", type=int, default=50, help="Psi4 geometry optimization iteration limit.")
    parser.add_argument("--retry-geom-iter", type=int, default=120, help="Geometry iteration limit used on retry attempts.")
    parser.add_argument(
        "--finalize-candidates",
        action="store_true",
        help="Do not run QM. Validate the existing candidate MolDB, write diffs, and optionally hard-replace repo RESP variants.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write plan only; do not run QM.")
    parser.add_argument("--no-commit", action="store_true", help="Do not hard-replace repo MolDB after candidate success.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.no_opt:
        args.reuse_geometry = True

    set_run_options(restart=False)
    doctor(print_report=True)
    ensure_initialized()

    catalog = _read_catalog(CATALOG_CSV)
    repo_db = MolDB(args.repo_db)
    selected_names = _parse_only(args.only)
    specs = _selected_specs(catalog=catalog, repo_db=repo_db, only=selected_names, target_mode=args.target_mode)
    if not specs:
        raise SystemExit("No existing repo MolDB entries matched the selected Example 07 catalog species.")

    job_root = Path(workdir(args.work_dir, restart=False))
    candidate_db_dir = job_root / "candidate_moldb"
    backup_dir = job_root / "backups"
    diff_dir = job_root / "charge_diffs"
    summary_path = args.summary or (job_root / "adaptive_resp_moldb_refresh_summary.json")

    cpu_total = _available_cpu_total()
    memory_total_mb = _available_memory_mb()
    planner_cpu_budget = max(1, int(cpu_total) - max(0, int(args.reserve_cores)))
    planner_memory_budget_mb = max(4096, int(memory_total_mb) - max(0, int(args.reserve_memory_mb)))
    tasks = _build_refresh_tasks(
        specs,
        cpu_budget=planner_cpu_budget,
        max_omp=max(1, int(args.max_omp)),
        max_memory_mb=max(4096, int(args.max_memory_mb)),
    )
    plan = {
        "catalog_csv": str(CATALOG_CSV.resolve()),
        "repo_db": str(Path(repo_db.db_dir).resolve()),
        "candidate_db": str(candidate_db_dir.resolve()),
        "work_dir": str(job_root.resolve()),
        "resp_profile": args.resp_profile,
        "target_mode": args.target_mode,
        "target_count": len(tasks),
        "cpu_total": int(cpu_total),
        "memory_total_mb": int(memory_total_mb),
        "reserve_cores": int(args.reserve_cores),
        "reserve_memory_mb": int(args.reserve_memory_mb),
        "planner_cpu_budget": int(planner_cpu_budget),
        "planner_memory_budget_mb": int(planner_memory_budget_mb),
        "reuse_geometry": bool(args.reuse_geometry),
        "geom_iter": int(args.geom_iter),
        "retry_geom_iter": int(args.retry_geom_iter),
        "commit": not bool(args.no_commit),
        "tasks": [asdict(task) for task in tasks],
    }
    (job_root / "parallel_refresh_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[DB] repo      = {repo_db.db_dir}")
    print(f"[DB] candidate = {candidate_db_dir}")
    print(f"[RUN] targets  = {', '.join(task.name for task in tasks)}")
    print(
        f"[PLAN] cpu_total={cpu_total} budget={planner_cpu_budget} "
        f"memory_total={memory_total_mb}MB budget={planner_memory_budget_mb}MB"
    )
    for task in tasks:
        print(
            f"[PLAN] {task.name:16s} profile={task.profile:15s} heavy={task.heavy_atoms:2d} "
            f"charge={task.formal_charge:+d} omp={task.psi4_omp:2d} mem={task.memory_mb:5d}MB"
        )

    if args.dry_run:
        print(f"[SUMMARY] dry-run plan written to {job_root / 'parallel_refresh_plan.json'}")
        return 0

    backup_path = _make_repo_backup(repo_db, backup_dir)
    print(f"[BACKUP] {backup_path}")

    if args.finalize_candidates:
        results, failures, diff_outputs, replacements = _finalize_candidate_moldb(
            repo_db=repo_db,
            candidate_db_dir=candidate_db_dir,
            specs=specs,
            resp_profile=args.resp_profile,
            tolerance=float(args.tolerance),
            diff_dir=diff_dir,
            no_commit=bool(args.no_commit),
        )
        out = {
            **plan,
            "backup_path": str(backup_path),
            "success_count": len(results),
            "failure_count": len(failures),
            "retry_count": 0,
            "committed": bool((not failures) and (not args.no_commit)),
            "finalize_candidates": True,
            "diff_outputs": diff_outputs,
            "results": [
                {key: value for key, value in result.items() if key != "diff_per_atom"}
                for result in results
            ],
            "failures": failures,
            "replacements": replacements,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[SUMMARY] {summary_path}")
        return 2 if failures else 0

    results, failures, retry_count = _run_parallel_refresh(
        tasks=tasks,
        repo_db_dir=Path(repo_db.db_dir),
        candidate_db_dir=candidate_db_dir,
        job_root=job_root / "qm_jobs",
        resp_profile=args.resp_profile,
        tolerance=float(args.tolerance),
        reuse_geometry=bool(args.reuse_geometry),
        geom_iter=int(args.geom_iter),
        retry_geom_iter=int(args.retry_geom_iter),
        planner_cpu_budget=planner_cpu_budget,
        planner_memory_budget_mb=planner_memory_budget_mb,
    )
    diff_outputs = _write_diff_outputs(diff_dir, results=results, failures=failures)

    replacements: list[dict[str, Any]] = []
    if failures:
        print("[STOP] At least one species failed; repo MolDB was not modified beyond the pre-run backup.")
    elif args.no_commit:
        print("[STOP] --no-commit requested; candidate MolDB and charge diffs are ready, repo MolDB unchanged.")
    else:
        candidate_db = MolDB(candidate_db_dir)
        spec_by_name = {spec.name: spec for spec in specs}
        for result in results:
            replacements.append(
                _hard_replace_repo_record(
                    repo_db=repo_db,
                    candidate_db=candidate_db,
                    spec=spec_by_name[str(result["name"])],
                    resp_profile=args.resp_profile,
                )
            )
            print(f"[REPLACE] {result['name']:16s} RESP variants hard-replaced")

    out = {
        **plan,
        "backup_path": str(backup_path),
        "success_count": len(results),
        "failure_count": len(failures),
        "retry_count": int(retry_count),
        "committed": bool((not failures) and (not args.no_commit)),
        "diff_outputs": diff_outputs,
        "results": [
            {key: value for key, value in result.items() if key != "diff_per_atom"}
            for result in results
        ],
        "failures": failures,
        "replacements": replacements,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[SUMMARY] {summary_path}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
