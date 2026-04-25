from __future__ import annotations

"""Refresh selected MolDB entries with adaptive RESP charges.

This helper targets the species where explicit RESP equivalence constraints
matter most for the bundled electrolyte examples:

* polyelectrolyte repeat units with localized carboxylate groups;
* neutral carbonate solvents used in EC/EMC/DEC benchmarks.

The refreshed entries are written as ``resp_profile="adaptive"`` variants, so
legacy MolDB records remain loadable for reproducibility.
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yadonpy as yp
from yadonpy.core import workdir
from yadonpy.core import chem_utils
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

CARBONATE_TARGETS = {"EC", "EMC", "DEC"}


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    smiles: str
    kind: str
    charge: str
    bonded: str | None
    polyelectrolyte_mode: bool


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


def _adaptive_variant_ready(db: MolDB, spec: SpeciesSpec) -> bool:
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
        if str(meta.get("resp_profile") or "").strip().lower() != "adaptive":
            continue
        if bool(meta.get("polyelectrolyte_mode", False)) != bool(spec.polyelectrolyte_mode):
            continue
        if bool(meta.get("ready", False)):
            return True
    return False


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


def _write_to_dbs(mol, spec: SpeciesSpec, *, dbs: list[tuple[str, MolDB]], resp_profile: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, db in dbs:
        rec = db.update_from_mol(
            mol,
            smiles_or_psmiles=spec.smiles,
            name=spec.name,
            charge="RESP",
            polyelectrolyte_mode=spec.polyelectrolyte_mode,
            polyelectrolyte_detection="auto",
            resp_profile=resp_profile,
        )
        adaptive_vids = [
            vid
            for vid, meta in sorted((rec.variants or {}).items())
            if isinstance(meta, dict)
            and str(meta.get("resp_profile") or "").strip().lower() == resp_profile
            and bool(meta.get("polyelectrolyte_mode", False)) == bool(spec.polyelectrolyte_mode)
        ]
        records.append(
            {
                "db": label,
                "db_dir": str(db.db_dir),
                "key": rec.key,
                "adaptive_variant_ids": adaptive_vids,
            }
        )
    return records


def refresh_one(
    spec: SpeciesSpec,
    *,
    default_db: MolDB | None,
    repo_db: MolDB,
    job_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dbs: list[tuple[str, MolDB]] = [("repo", repo_db)]
    if default_db is not None:
        dbs.insert(0, ("default", default_db))

    if (not args.force) and all(_adaptive_variant_ready(db, spec) for _label, db in dbs):
        return {
            "name": spec.name,
            "smiles": spec.smiles,
            "status": "skipped_ready",
            "polyelectrolyte_mode": spec.polyelectrolyte_mode,
        }

    mol, geometry_source, fresh = _load_geometry(spec, dbs=dbs, resp_profile=args.resp_profile)
    if fresh:
        try:
            mol.SetProp("_Name", spec.name)
        except Exception:
            pass

    formal_charge = _formal_charge(mol)
    charge_groups = detect_charged_groups(mol, detection="auto", resp_profile=args.resp_profile)
    needs_new_opt = bool(args.force_opt or fresh)
    if spec.name in CARBONATE_TARGETS and args.optimize_carbonates:
        needs_new_opt = True
    if args.no_opt:
        needs_new_opt = False

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
        raise RuntimeError(
            f"{spec.name} equivalence validation failed: "
            f"max_group_spread={validation['max_equivalence_spread']:.3e}, "
            f"max_carboxylate_spread={validation['max_carboxylate_oxygen_spread']:.3e}"
        )

    records = _write_to_dbs(mol, spec, dbs=dbs, resp_profile=args.resp_profile)
    recipe = {}
    if mol.HasProp("_yadonpy_qm_recipe_json"):
        try:
            recipe = json.loads(mol.GetProp("_yadonpy_qm_recipe_json"))
        except Exception:
            recipe = {}
    return {
        "name": spec.name,
        "smiles": spec.smiles,
        "status": "refreshed",
        "geometry_source": geometry_source,
        "optimized": bool(needs_new_opt),
        "formal_charge": int(formal_charge),
        "polyelectrolyte_mode": bool(spec.polyelectrolyte_mode),
        "charge_group_count": int(len(charge_groups.get("groups") or [])),
        "resp_profile": args.resp_profile,
        "qm_recipe": recipe,
        "repaired_equivalence_groups": int(repaired_groups),
        "validation": validation,
        "records": records,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Species names or comma-separated name lists.")
    parser.add_argument("--force", action="store_true", default=os.environ.get("YADONPY_FORCE", "").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--force-opt", action="store_true", help="Optimize geometry even when MolDB geometry is available.")
    parser.add_argument("--no-opt", action="store_true", help="Never optimize geometry; only recompute RESP on the current geometry.")
    parser.add_argument("--optimize-carbonates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-default-db", action="store_true", help="Only update the repository MolDB.")
    parser.add_argument("--repo-db", type=Path, default=REPO_ROOT / "moldb")
    parser.add_argument("--work-dir", type=Path, default=HERE / "work_dir" / "04_refresh_adaptive_resp_moldb")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--resp-profile", choices=("adaptive", "legacy"), default="adaptive")
    parser.add_argument("--omp", type=int, default=_env_default("YADONPY_PSI4_OMP", 8))
    parser.add_argument("--memory-mb", type=int, default=_env_default("YADONPY_PSI4_MEMORY_MB", 12000))
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.force_opt and args.no_opt:
        raise SystemExit("--force-opt and --no-opt are mutually exclusive")

    set_run_options(restart=False)
    doctor(print_report=True)
    ensure_initialized()

    catalog = _read_catalog(CATALOG_CSV)
    selected_names = _parse_only(args.only) or list(DEFAULT_TARGETS)
    missing = [name for name in selected_names if name not in catalog]
    if missing:
        raise SystemExit(f"Unknown species in electrolyte_species.csv: {missing}")

    default_db = None if args.skip_default_db else MolDB()
    repo_db = MolDB(args.repo_db)
    job_root = Path(workdir(args.work_dir, restart=False))
    summary_path = args.summary or (job_root / "adaptive_resp_moldb_refresh_summary.json")

    print(f"[DB] repo    = {repo_db.db_dir}")
    if default_db is not None:
        print(f"[DB] default = {default_db.db_dir}")
    print(f"[RUN] targets = {', '.join(selected_names)}")
    print(f"[QM] profile={args.resp_profile} omp={args.omp} memory_mb={args.memory_mb}")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name in selected_names:
        spec = catalog[name]
        try:
            result = refresh_one(spec, default_db=default_db, repo_db=repo_db, job_root=job_root, args=args)
            results.append(result)
            print(f"[OK]  {name:12s} {result['status']}")
        except Exception as exc:
            failure = {"name": name, "smiles": spec.smiles, "error": repr(exc)}
            failures.append(failure)
            print(f"[ERR] {name:12s} {exc}")

    out = {
        "catalog_csv": str(CATALOG_CSV.resolve()),
        "repo_db": str(repo_db.db_dir),
        "default_db": str(default_db.db_dir) if default_db is not None else None,
        "work_dir": str(job_root.resolve()),
        "resp_profile": args.resp_profile,
        "target_count": len(selected_names),
        "success_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[SUMMARY] {summary_path}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
