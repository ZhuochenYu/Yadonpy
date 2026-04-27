from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from yadonpy.core import poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.core.metadata import (
    QM_RECIPE_PROP,
    RESP_CONSTRAINTS_PROP,
    RESP_PROFILE_PROP,
    read_json_prop,
    read_text_prop,
)
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2, GAFF2_mod, MERZ
from yadonpy.gmx.analysis.structured import build_msd_metric_catalog, compute_msd_series
from yadonpy.gmx.topology import parse_system_top
from yadonpy.runtime import set_run_options
from yadonpy.sim.analyzer import AnalyzeResult
from yadonpy.sim import qm
from yadonpy.sim.benchmarking import _dump_json, summarize_rdkit_species_forcefield
from yadonpy.sim.performance import resolve_io_analysis_policy
from yadonpy.sim.preset import eq
from yadonpy.workflow import EnvReader


_ENV = EnvReader()


def _env_bool(name: str, default: bool) -> bool:
    return _ENV.bool(name, default)


def _env_int(name: str, default: int) -> int:
    return _ENV.int(name, default)


def _env_float(name: str, default: float) -> float:
    return _ENV.float(name, default)


def _env_text(name: str, default: str) -> str:
    return _ENV.text(name, default)


def _normalize_charge_mode(raw: str | None) -> str:
    mode = str(raw or "resp").strip().lower()
    if mode in {"resp2", "resp_2"}:
        raise ValueError("This benchmark is intentionally restricted to GAFF2 + RESP. RESP2 is out of scope for this script.")
    return "resp"


def _normalize_gaff_variant(raw: str | None) -> str:
    variant = str(raw or "classic").strip().lower()
    if variant in {"mod", "gaff2_mod"}:
        return "mod"
    return "classic"


def _normalize_resp_profile(raw: str | None) -> str:
    profile = str(raw or "adaptive").strip().lower()
    if profile in {"default", "current"}:
        profile = "adaptive"
    if profile not in {"adaptive", "legacy"}:
        raise ValueError(f"Unsupported RESP profile: {raw!r}")
    return profile


def _normalize_solvent_source(raw: str | None) -> str:
    source = str(raw or "qm").strip().lower()
    if source in {"db", "moldb", "ready", "ready_db"}:
        return "moldb"
    if source not in {"qm", "moldb"}:
        raise ValueError(f"Unsupported solvent source: {raw!r}")
    return source


def _normalize_db_priority(raw: str | None) -> str:
    mode = str(raw or "auto").strip().lower()
    if mode in {"repo", "repo_first", "local_first"}:
        return "repo_first"
    if mode in {"default", "default_first", "global_first"}:
        return "default_first"
    if mode != "auto":
        raise ValueError(f"Unsupported MolDB priority: {raw!r}")
    return "auto"


def _normalize_equilibration_mode(raw: str | None) -> str:
    mode = str(raw or "auto").strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "eq21": "eq21",
        "liquid": "liquid_anneal",
        "liquid_anneal": "liquid_anneal",
        "cemp": "liquid_anneal",
        "cemp_like": "liquid_anneal",
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported equilibration mode: {raw!r}") from exc


def _normalize_constraints(raw: str | None, default: str = "h-bonds") -> str:
    token = str(raw or default).strip().lower().replace("_", "-")
    aliases = {
        "none": "none",
        "no": "none",
        "off": "none",
        "hbonds": "h-bonds",
        "h-bonds": "h-bonds",
        "allbonds": "all-bonds",
        "all-bonds": "all-bonds",
    }
    if token not in aliases:
        raise ValueError(f"Unsupported constraints mode: {raw!r}")
    return aliases[token]


def _load_equilibrium_payload(analysis_dir: Path) -> dict[str, Any]:
    path = Path(analysis_dir) / "equilibrium.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _restart_latest_equilibrated_gro(work_root: Path, fallback: Path) -> Path:
    """Prefer the latest completed equilibration restart point, excluding production."""
    finder = getattr(eq, "_find_latest_equilibrated_gro", None)
    if not callable(finder):
        return Path(fallback)
    try:
        candidate = finder(Path(work_root), exclude_dirs=(Path(work_root) / "05_npt_production",))
    except Exception:
        candidate = None
    if candidate is None:
        return Path(fallback)
    candidate = Path(candidate)
    if not candidate.exists():
        return Path(fallback)
    try:
        if candidate.resolve() == Path(fallback).resolve():
            return Path(fallback)
    except Exception:
        pass
    print(f"[RESTART] Continuing additional equilibration from latest completed structure: {candidate}")
    return candidate


def _analyze_restart_stage(work_root: Path, gro_path: Path) -> AnalyzeResult | None:
    stage_dir = Path(gro_path).parent
    tpr = stage_dir / "md.tpr"
    xtc = stage_dir / "md.xtc"
    edr = stage_dir / "md.edr"
    top = Path(work_root) / "02_system" / "system.top"
    ndx = Path(work_root) / "02_system" / "system.ndx"
    if not (tpr.exists() and xtc.exists() and edr.exists() and top.exists() and ndx.exists()):
        return None
    return AnalyzeResult(
        work_dir=Path(work_root),
        tpr=tpr,
        xtc=xtc,
        edr=edr,
        top=top,
        ndx=ndx,
        trr=(stage_dir / "md.trr") if (stage_dir / "md.trr").exists() else None,
        omp=omp,
    )


def _transport_confidence_from_equilibrium(payload: dict[str, Any], ok: bool) -> dict[str, Any]:
    density_gate = payload.get("density_gate") if isinstance(payload, dict) else None
    density_gate = density_gate if isinstance(density_gate, dict) else {}
    severity = str(density_gate.get("severity") or ("none" if ok else "high"))
    if ok:
        confidence = "high"
    elif severity == "high":
        confidence = "low_density_not_converged"
    else:
        confidence = "medium_density_not_converged"
    return {
        "equilibration_ok": bool(ok),
        "density_warning_severity": severity,
        "transport_confidence": confidence,
        "density_gate": density_gate,
    }


def _charge_recipe_from_family(raw: str | None) -> dict[str, str]:
    family = str(raw or "wb97m_v").strip().lower().replace("-", "_")
    recipes = {
        "b3lyp_d3bj": {
            "family": "b3lyp_d3bj",
            "label": "B3LYP-D3BJ/def2-TZVP",
            "opt_method": "b3lyp-d3bj",
            "charge_method": "b3lyp-d3bj",
        },
        "wb97m_v": {
            "family": "wb97m_v",
            "label": "wB97M-V/def2-TZVP",
            "opt_method": "wb97m-v",
            "charge_method": "wb97m-v",
        },
        "m06_2x": {
            "family": "m06_2x",
            "label": "M06-2X/def2-TZVP",
            "opt_method": "m06-2x",
            "charge_method": "m06-2x",
        },
        # Hidden compatibility alias for older local probes.
        "wb97m_d3bj": {
            "family": "wb97m_d3bj",
            "label": "wB97M-D3BJ/def2-TZVP",
            "opt_method": "wb97m-d3bj",
            "charge_method": "wb97m-d3bj",
        },
    }
    if family not in recipes:
        family = "wb97m_v"
    recipe = dict(recipes[family])
    recipe.update(
        {
            "opt_basis": "def2-TZVP",
            "charge_basis": "def2-TZVP",
            "opt_basis_gen": {"Br": "def2-TZVP", "I": "def2-TZVP"},
            "charge_basis_gen": {"Br": "def2-TZVP", "I": "def2-TZVP"},
        }
    )
    return recipe


def _build_ff_variant(variant: str):
    if str(variant).strip().lower() == "mod":
        return GAFF2_mod()
    return GAFF2()


def _json_prop(mol, key: str) -> dict[str, Any] | None:
    value = read_json_prop(mol, key)
    return value if isinstance(value, dict) else None


def _extract_resp_route(mol, *, label: str) -> dict[str, Any]:
    route = {
        "label": str(label),
        "resp_profile": None,
        "qm_recipe": None,
        "constraint_mode": None,
        "equivalence_group_count": 0,
    }
    route["resp_profile"] = read_text_prop(mol, RESP_PROFILE_PROP)
    qm_recipe = _json_prop(mol, QM_RECIPE_PROP)
    if isinstance(qm_recipe, dict):
        route["qm_recipe"] = qm_recipe
        if route["resp_profile"] is None:
            route["resp_profile"] = qm_recipe.get("resp_profile")
    constraints = _json_prop(mol, RESP_CONSTRAINTS_PROP)
    if isinstance(constraints, dict):
        route["constraint_mode"] = constraints.get("mode")
        route["equivalence_group_count"] = int(len(constraints.get("equivalence_groups") or []))
        if route["resp_profile"] is None:
            route["resp_profile"] = constraints.get("resp_profile")
    return route


def _equivalence_spread_diagnostic(mol, *, label: str) -> dict[str, Any]:
    constraints = _json_prop(mol, RESP_CONSTRAINTS_PROP) or {}
    groups = list(constraints.get("equivalence_groups") or [])
    diagnostics = []
    prop_names = ["AtomicCharge", "RESP", "RESP2", "ESP"]
    for group in groups:
        idxs = sorted({int(i) for i in group})
        if len(idxs) <= 1:
            continue
        spreads = {}
        for prop in prop_names:
            values = []
            for idx in idxs:
                atom = mol.GetAtomWithIdx(idx)
                if not atom.HasProp(prop):
                    values = []
                    break
                values.append(float(atom.GetDoubleProp(prop)))
            if values:
                spreads[prop] = float(max(values) - min(values))
        diagnostics.append(
            {
                "atom_indices": idxs,
                "symbols": [str(mol.GetAtomWithIdx(idx).GetSymbol()) for idx in idxs],
                "spreads_e": spreads,
            }
        )
    max_spread = 0.0
    for item in diagnostics:
        for spread in item.get("spreads_e", {}).values():
            max_spread = max(max_spread, float(spread))
    return {
        "label": str(label),
        "group_count": len(diagnostics),
        "max_spread_e": float(max_spread),
        "groups": diagnostics,
    }


def _load_ready_gaff_species(
    ff: GAFF2 | GAFF2_mod,
    smiles: str,
    *,
    label: str,
    repo_db_dir: Path,
    charge_mode: str,
    db_priority: str,
):
    last_exc: Exception | None = None
    db_charge = "RESP2" if charge_mode == "resp2" else "RESP"
    search_order = [(None, "default"), (repo_db_dir, "repo")]
    if db_priority == "repo_first":
        search_order = [(repo_db_dir, "repo"), (None, "default")]
    for db_dir, db_label in search_order:
        try:
            mol = ff.mol_rdkit(
                smiles,
                name=label,
                db_dir=db_dir,
                charge=db_charge,
                require_ready=True,
                prefer_db=True,
            )
            mol = ff.ff_assign(mol, charge=None, report=False)
            if not mol:
                raise RuntimeError(f"Cannot assign {ff.name} parameters for {label}.")
            print(f"[MolDB] loaded {label} with {db_charge} charges from {db_label} db")
            return mol
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"{label} is expected to be ready in MolDB for the GAFF2 benchmark.") from last_exc


def _load_ready_pf6(ff: GAFF2 | GAFF2_mod, *, repo_db_dir: Path, db_priority: str):
    last_exc: Exception | None = None
    search_order = [(None, "default"), (repo_db_dir, "repo")]
    if db_priority == "repo_first":
        search_order = [(repo_db_dir, "repo"), (None, "default")]
    for db_dir, db_label in search_order:
        try:
            mol = ff.mol_rdkit(
                PF6_SMILES,
                name="PF6",
                db_dir=db_dir,
                charge="RESP",
                require_ready=True,
                prefer_db=True,
            )
            mol = ff.ff_assign(mol, charge=None, bonded="DRIH", report=False)
            if not mol:
                raise RuntimeError("Cannot assign PF6 parameters from MolDB-backed DRIH topology.")
            print(f"[MolDB] loaded PF6 with RESP charges from {db_label} db")
            return mol
        except Exception as exc:
            last_exc = exc
    raise RuntimeError("PF6 is expected to be ready in MolDB for the GAFF2 benchmark.") from last_exc


def _assign_merz_ion(ff: MERZ, smiles: str, *, label: str):
    mol = ff.mol(smiles)
    mol = ff.ff_assign(mol)
    if not mol:
        raise RuntimeError(f"Cannot assign MERZ ion parameters for {label}.")
    print(f"[MERZ] assigned built-in ion parameters for {label}")
    return mol


def _build_qm_ready_gaff_species(
    ff: GAFF2 | GAFF2_mod,
    smiles: str,
    *,
    label: str,
    recipe: dict[str, str],
    resp_profile: str,
    work_root: Path,
    psi4_omp: int,
    mpi: int,
    omp: int,
    memory_mb: int,
    repo_db_dir: Path | None = None,
    cache_to_repo_db: bool = False,
):
    mol = utils.mol_from_smiles(smiles)
    log_name = f"{label.lower()}_{recipe['family']}_{ff.name}"
    mol, _energy = qm.conformation_search(
        mol,
        ff=ff,
        work_dir=work_root,
        log_name=log_name,
        psi4_omp=psi4_omp,
        mpi=mpi,
        omp=omp,
        memory=memory_mb,
        opt_method=recipe["opt_method"],
        opt_basis=recipe["opt_basis"],
        opt_basis_gen=recipe["opt_basis_gen"],
    )
    qm.assign_charges(
        mol,
        charge="RESP",
        opt=False,
        work_dir=work_root,
        log_name=log_name,
        omp=psi4_omp,
        memory=memory_mb,
        charge_method=recipe["charge_method"],
        charge_basis=recipe["charge_basis"],
        charge_basis_gen=recipe["charge_basis_gen"],
        resp_profile=resp_profile,
    )
    mol = ff.ff_assign(mol, charge=None, report=False)
    if not mol:
        raise RuntimeError(f"Cannot assign {ff.name} parameters for {label} after QM/RESP.")
    if cache_to_repo_db and repo_db_dir is not None:
        ff.store_to_db(
            mol,
            smiles_or_psmiles=smiles,
            name=label,
            db_dir=repo_db_dir,
            charge="RESP",
            basis_set=recipe["charge_basis"],
            method=recipe["charge_method"],
        )
        print(f"[MolDB] stored freshly computed {label} RESP entry into repo db: {repo_db_dir}")
    return mol


def _point_charge_dipole_debye(mol) -> float | None:
    try:
        conf = mol.GetConformer()
    except Exception:
        return None
    dip = np.zeros(3, dtype=float)
    for atom in mol.GetAtoms():
        q = float(atom.GetDoubleProp("AtomicCharge")) if atom.HasProp("AtomicCharge") else 0.0
        pos = conf.GetAtomPosition(atom.GetIdx())
        dip += q * np.asarray([float(pos.x), float(pos.y), float(pos.z)], dtype=float)
    # 1 e*Angstrom = 4.80320427 Debye
    return float(np.linalg.norm(dip) * 4.80320427)


def _summarize_carbonate_charge_features(mol, *, label: str) -> dict[str, Any]:
    carbonyl_oxygen_charges: list[float] = []
    carbonyl_carbon_charges: list[float] = []
    ether_oxygen_charges: list[float] = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        charge = float(atom.GetDoubleProp("AtomicCharge")) if atom.HasProp("AtomicCharge") else 0.0
        if sym == "O":
            is_carbonyl = False
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if other.GetSymbol() == "C" and bond.GetBondTypeAsDouble() >= 1.5:
                    is_carbonyl = True
                    break
            if is_carbonyl:
                carbonyl_oxygen_charges.append(charge)
            else:
                ether_oxygen_charges.append(charge)
        elif sym == "C":
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if other.GetSymbol() == "O" and bond.GetBondTypeAsDouble() >= 1.5:
                    carbonyl_carbon_charges.append(charge)
                    break
    net_q = 0.0
    for atom in mol.GetAtoms():
        if atom.HasProp("AtomicCharge"):
            net_q += float(atom.GetDoubleProp("AtomicCharge"))
    return {
        "label": str(label),
        "net_charge_e": float(net_q),
        "carbonyl_oxygen_charge_e": float(np.mean(carbonyl_oxygen_charges)) if carbonyl_oxygen_charges else None,
        "carbonyl_carbon_charge_e": float(np.mean(carbonyl_carbon_charges)) if carbonyl_carbon_charges else None,
        "ether_oxygen_charge_mean_e": float(np.mean(ether_oxygen_charges)) if ether_oxygen_charges else None,
        "point_charge_dipole_debye": _point_charge_dipole_debye(mol),
    }


def _atom_ff_signature(mol) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        rows.append(
            {
                "idx": int(atom.GetIdx()),
                "symbol": str(atom.GetSymbol()),
                "ff_type": str(atom.GetProp("ff_type")) if atom.HasProp("ff_type") else "",
                "ff_sigma_nm": float(atom.GetDoubleProp("ff_sigma")) if atom.HasProp("ff_sigma") else None,
                "ff_epsilon_kj_mol": float(atom.GetDoubleProp("ff_epsilon")) if atom.HasProp("ff_epsilon") else None,
            }
        )
    return rows


def _audit_gaff_variant_differences(
    *,
    recipe: dict[str, str],
    resp_profile: str,
    work_root: Path,
    repo_db_dir: Path,
    psi4_omp: int,
    mpi: int,
    omp: int,
    memory_mb: int,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "recipe": recipe,
        "species": {},
        "notes": (
            "GAFF2_mod is audited only as a species-level reference. The full MD benchmark uses classic GAFF2."
        ),
    }
    classic_ff = GAFF2()
    mod_ff = GAFF2_mod()
    species_specs = (
        ("EC", EC_SMILES, "solvent"),
        ("EMC", EMC_SMILES, "solvent"),
        ("DEC", DEC_SMILES, "solvent"),
    )
    for label, smiles, role in species_specs:
        classic_mol = _build_qm_ready_gaff_species(
            classic_ff,
            smiles,
            label=label,
            recipe=recipe,
            resp_profile=resp_profile,
            work_root=work_root,
            psi4_omp=psi4_omp,
            mpi=mpi,
            omp=omp,
            memory_mb=memory_mb,
            repo_db_dir=repo_db_dir,
            cache_to_repo_db=False,
        )
        mod_mol = utils.deepcopy_mol(classic_mol)
        mod_mol = mod_ff.ff_assign(mod_mol, charge=None, report=False)
        if not mod_mol:
            raise RuntimeError(f"Cannot assign GAFF2_mod parameters for {label} during audit.")
        classic_rows = _atom_ff_signature(classic_mol)
        mod_rows = _atom_ff_signature(mod_mol)
        diff = []
        for row_c, row_m in zip(classic_rows, mod_rows):
            changed = {}
            for key in ("ff_type", "ff_sigma_nm", "ff_epsilon_kj_mol"):
                if row_c.get(key) != row_m.get(key):
                    changed[key] = {"classic": row_c.get(key), "mod": row_m.get(key)}
            if changed:
                diff.append({"idx": row_c["idx"], "symbol": row_c["symbol"], "changes": changed})
        audit["species"][label] = {
            "role": role,
            "classic_summary": summarize_rdkit_species_forcefield(classic_mol, label=label, moltype_hint=label, charge_scale=1.0),
            "mod_summary": summarize_rdkit_species_forcefield(mod_mol, label=label, moltype_hint=label, charge_scale=1.0),
            "charge_features": _summarize_carbonate_charge_features(classic_mol, label=label),
            "atom_param_differences": diff,
        }
    pf6_classic = _load_ready_pf6(classic_ff, repo_db_dir=repo_db_dir)
    pf6_mod = _load_ready_pf6(mod_ff, repo_db_dir=repo_db_dir)
    diff_pf6 = []
    for row_c, row_m in zip(_atom_ff_signature(pf6_classic), _atom_ff_signature(pf6_mod)):
        changed = {}
        for key in ("ff_type", "ff_sigma_nm", "ff_epsilon_kj_mol"):
            if row_c.get(key) != row_m.get(key):
                changed[key] = {"classic": row_c.get(key), "mod": row_m.get(key)}
        if changed:
            diff_pf6.append({"idx": row_c["idx"], "symbol": row_c["symbol"], "changes": changed})
    audit["species"]["PF6"] = {
        "role": "anion",
        "classic_summary": summarize_rdkit_species_forcefield(pf6_classic, label="PF6", moltype_hint="PF6", charge_scale=1.0),
        "mod_summary": summarize_rdkit_species_forcefield(pf6_mod, label="PF6", moltype_hint="PF6", charge_scale=1.0),
        "atom_param_differences": diff_pf6,
    }
    return audit


def _extract_default_diffusivity(msd: dict[str, Any], moltype: str) -> float | None:
    record = msd.get(moltype) or msd.get(str(moltype).lower())
    if not isinstance(record, dict):
        return None
    try:
        direct = record.get("D_m2_s")
        if direct is not None:
            return float(direct)
    except Exception:
        pass
    metric_name = str(record.get("default_metric") or "").strip()
    metrics = record.get("metrics")
    if not metric_name or not isinstance(metrics, dict):
        return None
    metric = metrics.get(metric_name)
    if not isinstance(metric, dict):
        return None
    try:
        return float(metric.get("D_m2_s"))
    except Exception:
        return None


def _extract_default_msd_metric_record(msd: dict[str, Any], moltype: str) -> dict[str, Any]:
    record = msd.get(moltype) or msd.get(str(moltype).lower())
    if not isinstance(record, dict):
        return {}
    metric_name = str(record.get("default_metric") or "").strip()
    metrics = record.get("metrics")
    if metric_name and isinstance(metrics, dict) and isinstance(metrics.get(metric_name), dict):
        return dict(metrics[metric_name])
    return dict(record)


def _default_msd_trajectory_bounds(msd: dict[str, Any], labels: tuple[str, ...] = ("EC", "EMC", "DEC")) -> tuple[float | None, float | None]:
    starts: list[float] = []
    ends: list[float] = []
    for label in labels:
        metric = _extract_default_msd_metric_record(msd, label)
        start_raw = metric.get("trajectory_time_start_ps")
        end_raw = metric.get("trajectory_time_end_ps")
        try:
            if start_raw is not None and end_raw is not None:
                start = float(start_raw)
                end = float(end_raw)
                if np.isfinite(start) and np.isfinite(end) and end > start:
                    starts.append(start)
                    ends.append(end)
                    continue
        except Exception:
            pass
        csv_path = metric.get("series_csv")
        if not csv_path:
            continue
        try:
            arr = np.genfromtxt(str(csv_path), delimiter=",", names=True)
            t = np.asarray(arr["time_ps"], dtype=float)
            t = t[np.isfinite(t)]
            if t.size >= 2 and float(t[-1]) > float(t[0]):
                starts.append(float(t[0]))
                ends.append(float(t[-1]))
        except Exception:
            continue
    if not starts or not ends:
        return None, None
    return float(min(starts)), float(max(ends))


def _summarize_msd_block_diffusion(
    blocks: list[dict[str, Any]],
    *,
    expected_order: tuple[str, ...] = ("EMC", "DEC", "EC"),
) -> dict[str, Any]:
    valid_blocks = [block for block in blocks if isinstance(block.get("diffusion_m2_s"), dict)]
    if not valid_blocks:
        return {
            "status": "skipped",
            "reason": "no_valid_block_diffusion",
            "blocks": blocks,
        }

    species_labels = sorted(
        {
            str(label)
            for block in valid_blocks
            for label, value in (block.get("diffusion_m2_s") or {}).items()
            if value is not None
        }
    )
    species_stats: dict[str, Any] = {}
    for label in species_labels:
        values = []
        for block in valid_blocks:
            try:
                value = (block.get("diffusion_m2_s") or {}).get(label)
                if value is not None and np.isfinite(float(value)):
                    values.append(float(value))
            except Exception:
                continue
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
        species_stats[label] = {
            "n_valid_blocks": int(arr.size),
            "mean_D_m2_s": mean,
            "std_D_m2_s": std,
            "sem_D_m2_s": float(std / np.sqrt(arr.size)) if arr.size >= 2 else 0.0,
            "cv": float(std / abs(mean)) if mean != 0.0 else None,
            "min_D_m2_s": float(np.min(arr)),
            "max_D_m2_s": float(np.max(arr)),
        }

    expected_present = [label for label in expected_order if label in species_stats]
    block_orders = []
    expected_order_matches = []
    for block in valid_blocks:
        diffusion = {
            str(label): float(value)
            for label, value in (block.get("diffusion_m2_s") or {}).items()
            if value is not None
        }
        order = [label for label, _value in sorted(diffusion.items(), key=lambda item: item[1], reverse=True)]
        block_orders.append(
            {
                "block_index": block.get("block_index"),
                "time_start_ps": block.get("time_start_ps"),
                "time_end_ps": block.get("time_end_ps"),
                "observed_order_fast_to_slow": order,
            }
        )
        if len(expected_present) >= 2 and all(label in diffusion for label in expected_present):
            expected_order_matches.append([label for label in order if label in expected_present] == expected_present)

    pairwise_expected = []
    for fast, slow in zip(expected_present, expected_present[1:]):
        comparisons = []
        ratios = []
        for block in valid_blocks:
            diffusion = block.get("diffusion_m2_s") or {}
            try:
                fast_d = diffusion.get(fast)
                slow_d = diffusion.get(slow)
                if fast_d is None or slow_d is None:
                    continue
                fast_f = float(fast_d)
                slow_f = float(slow_d)
                if not (np.isfinite(fast_f) and np.isfinite(slow_f)):
                    continue
                comparisons.append(fast_f > slow_f)
                if slow_f != 0.0:
                    ratios.append(fast_f / slow_f)
            except Exception:
                continue
        pairwise_expected.append(
            {
                "faster": fast,
                "slower": slow,
                "n_valid_blocks": len(comparisons),
                "ok_fraction": float(np.mean(comparisons)) if comparisons else None,
                "mean_ratio": float(np.mean(ratios)) if ratios else None,
            }
        )
    order_counts: dict[str, int] = {}
    for row in block_orders:
        key = ">".join(str(x) for x in (row.get("observed_order_fast_to_slow") or []))
        if key:
            order_counts[key] = int(order_counts.get(key, 0) + 1)
    pairwise_fractions = [
        float(row["ok_fraction"])
        for row in pairwise_expected
        if row.get("ok_fraction") is not None
    ]
    match_fraction = float(np.mean(expected_order_matches)) if expected_order_matches else None
    if len(expected_present) < 2 or not pairwise_fractions:
        ranking_confidence = "not_applicable"
        ranking_interpretation = "Not enough solvent species are present to assess the expected order."
    elif all(frac >= 0.75 for frac in pairwise_fractions) and (match_fraction is None or match_fraction >= 0.75):
        ranking_confidence = "supports_expected"
        ranking_interpretation = "Most blocks support the expected solvent ordering."
    elif any(0.25 < frac < 0.75 for frac in pairwise_fractions):
        ranking_confidence = "ambiguous"
        ranking_interpretation = "At least one adjacent solvent pair changes order across blocks; extend sampling before interpreting that pair."
    else:
        ranking_confidence = "contradicts_expected"
        ranking_interpretation = "Blockwise ordering consistently disagrees with at least one expected adjacent solvent pair."

    return {
        "status": "ok",
        "n_blocks": len(blocks),
        "n_valid_blocks": len(valid_blocks),
        "species": species_stats,
        "block_orders_fast_to_slow": block_orders,
        "block_order_counts": order_counts,
        "expected_order_fast_to_slow": list(expected_order),
        "expected_order_for_present_species": expected_present,
        "expected_order_match_fraction": match_fraction,
        "pairwise_expected": pairwise_expected,
        "ranking_confidence": ranking_confidence,
        "ranking_interpretation": ranking_interpretation,
        "blocks": blocks,
        "notes": (
            "Each block recomputes species COM MSD from the trajectory slice using lag time. "
            "Use block spread/order fractions to judge whether a solvent ranking is stable."
        ),
    }


def _msd_block_diffusion_diagnostic(
    analy: AnalyzeResult,
    *,
    full_msd: dict[str, Any],
    n_blocks: int,
    min_block_ps: float = 500.0,
    labels: tuple[str, ...] = ("EC", "EMC", "DEC"),
) -> dict[str, Any]:
    n_blocks = int(max(0, n_blocks))
    if n_blocks < 2:
        return {"status": "skipped", "reason": "MSD_BLOCKS<2", "n_blocks_requested": n_blocks}
    start_ps, end_ps = _default_msd_trajectory_bounds(full_msd, labels=labels)
    if start_ps is None or end_ps is None or end_ps <= start_ps:
        return {"status": "skipped", "reason": "trajectory_time_bounds_unavailable", "n_blocks_requested": n_blocks}
    duration_ps = float(end_ps - start_ps)
    max_blocks_by_duration = int(np.floor(duration_ps / max(float(min_block_ps), 1.0e-12)))
    block_count = min(n_blocks, max_blocks_by_duration)
    if block_count < 2:
        return {
            "status": "skipped",
            "reason": "trajectory_too_short_for_blocks",
            "n_blocks_requested": n_blocks,
            "duration_ps": duration_ps,
            "min_block_ps": float(min_block_ps),
        }

    try:
        topo = parse_system_top(Path(analy.top))
        system_dir = analy._system_dir()
        metric_catalog = build_msd_metric_catalog(topo, system_dir)
        xtc_path = analy._analysis_xtc_path()
    except Exception as exc:
        return {"status": "skipped", "reason": f"setup_failed: {exc}", "n_blocks_requested": n_blocks}

    catalog_by_lower = {str(key).lower(): (key, value) for key, value in metric_catalog.items()}
    transport = full_msd.get("_transport") if isinstance(full_msd.get("_transport"), dict) else {}
    geometry_mode = str(transport.get("geometry_mode") or "auto")
    unwrap = str(transport.get("unwrap") or "auto")
    drift = str(transport.get("drift") or "auto")

    edges = np.linspace(float(start_ps), float(end_ps), int(block_count) + 1)
    blocks: list[dict[str, Any]] = []
    for block_idx in range(int(block_count)):
        begin = float(edges[block_idx])
        end = float(edges[block_idx + 1])
        block: dict[str, Any] = {
            "block_index": int(block_idx),
            "time_start_ps": begin,
            "time_end_ps": end,
            "duration_ps": float(end - begin),
            "diffusion_m2_s": {},
            "fit_status": {},
            "fit_confidence": {},
            "errors": {},
        }
        for label in labels:
            catalog_item = catalog_by_lower.get(str(label).lower())
            if catalog_item is None:
                block["errors"][label] = "species_not_found"
                continue
            moltype, entry = catalog_item
            metric_name = str(entry.get("default_metric") or "")
            metric_entry = (entry.get("metrics") or {}).get(metric_name)
            group_specs = list((metric_entry or {}).get("groups") or [])
            if not metric_name or not group_specs:
                block["errors"][label] = "default_metric_or_groups_missing"
                continue
            try:
                metric_data = compute_msd_series(
                    gro_path=system_dir / "system.gro",
                    xtc_path=xtc_path,
                    top_path=Path(analy.top),
                    system_dir=system_dir,
                    group_specs=group_specs,
                    geometry_mode=geometry_mode,
                    unwrap=unwrap,
                    drift=drift,
                    begin_ps=begin,
                    end_ps=end,
                )
                fit = dict(metric_data.get("fit") or {})
                d_val = fit.get("D_m2_s")
                block["diffusion_m2_s"][str(label)] = float(d_val) if d_val is not None else None
                block["fit_status"][str(label)] = fit.get("status")
                block["fit_confidence"][str(label)] = fit.get("confidence")
            except Exception as exc:
                block["errors"][label] = str(exc)
        block["observed_order_fast_to_slow"] = [
            label
            for label, value in sorted(
                {
                    str(label): value
                    for label, value in (block.get("diffusion_m2_s") or {}).items()
                    if value is not None
                }.items(),
                key=lambda item: float(item[1]),
                reverse=True,
            )
        ]
        blocks.append(block)

    summary = _summarize_msd_block_diffusion(blocks)
    summary.update(
        {
            "n_blocks_requested": int(n_blocks),
            "min_block_ps": float(min_block_ps),
            "trajectory_time_start_ps": float(start_ps),
            "trajectory_time_end_ps": float(end_ps),
            "trajectory_duration_ps": duration_ps,
        }
    )
    return summary


def _solvent_diffusion_diagnostic(
    diffusion_m2_s: dict[str, float | None],
    *,
    expected_order: tuple[str, ...] = ("EMC", "DEC", "EC"),
) -> dict[str, Any]:
    solvents = ("EC", "EMC", "DEC")
    present: dict[str, float] = {}
    for label in solvents:
        value = diffusion_m2_s.get(label)
        try:
            if value is not None:
                present[label] = float(value)
        except Exception:
            continue
    observed_order = [label for label, _value in sorted(present.items(), key=lambda item: item[1], reverse=True)]
    slowest = min(present.values()) if present else None
    relative_to_slowest = {
        label: (float(value) / float(slowest) if slowest and slowest > 0.0 else None)
        for label, value in present.items()
    }
    expected_present = [label for label in expected_order if label in present]
    pairwise_expected = []
    for fast, slow in zip(expected_present, expected_present[1:]):
        pairwise_expected.append(
            {
                "faster": fast,
                "slower": slow,
                "ok": bool(present.get(fast, float("-inf")) > present.get(slow, float("inf"))),
                "ratio": (
                    float(present[fast]) / float(present[slow])
                    if slow in present and present[slow] not in (0.0, None)
                    else None
                ),
            }
        )
    expected_order_observed = [label for label in observed_order if label in expected_present]
    return {
        "observed_order_fast_to_slow": observed_order,
        "expected_order_fast_to_slow": list(expected_order),
        "expected_order_for_present_species": expected_present,
        "matches_expected_for_present_species": bool(expected_order_observed == expected_present) if len(expected_present) >= 2 else None,
        "relative_to_slowest": relative_to_slowest,
        "pairwise_expected": pairwise_expected,
        "notes": "Diffusion ordering from short MD is noisy; use this diagnostic as a screen, not a final transport benchmark.",
    }


def _extract_rdf_site(rdf: dict[str, Any], site_id: str) -> dict[str, Any]:
    block = rdf.get(site_id)
    return dict(block) if isinstance(block, dict) else {}


def _extract_primary_oxygen_site(rdf: dict[str, Any], moltype: str) -> dict[str, Any]:
    token = str(moltype or "").strip().lower()
    for site_id in (f"{token}:carbonyl_oxygen", f"{token}:oxygen_site"):
        block = _extract_rdf_site(rdf, site_id)
        if block:
            return block
    return {}


def _coordination_preference_summary(coordination: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    labels = {
        "EC": "EC_carbonyl_oxygen",
        "EMC": "EMC_carbonyl_oxygen",
        "DEC": "DEC_carbonyl_oxygen",
    }
    cn_by_species: dict[str, float] = {}
    for species, key in labels.items():
        block = coordination.get(key)
        if not isinstance(block, dict):
            continue
        try:
            cn = float(block.get("cn_shell"))
        except Exception:
            continue
        cn_by_species[species] = cn

    total_cn = sum(cn_by_species.values())
    total_count = sum(int(counts.get(species, 0) or 0) for species in labels)
    out: dict[str, Any] = {
        "total_cn_shell": total_cn,
        "notes": "shell_fraction is the fraction of first-shell carbonyl coordination; enrichment_vs_bulk > 1 means over-represented versus bulk solvent composition.",
    }
    for species, cn in cn_by_species.items():
        bulk_count = int(counts.get(species, 0) or 0)
        bulk_fraction = (bulk_count / total_count) if total_count > 0 else None
        shell_fraction = (cn / total_cn) if total_cn > 0 else None
        enrichment = None
        if bulk_fraction and shell_fraction is not None and bulk_fraction > 0:
            enrichment = shell_fraction / bulk_fraction
        out[species] = {
            "cn_shell": cn,
            "bulk_fraction": bulk_fraction,
            "shell_fraction": shell_fraction,
            "enrichment_vs_bulk": enrichment,
        }
    return out


def _neutral_charge_anomalies(species_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in species_rows:
        label = str(row.get("label") or "")
        if label in {"Li", "PF6"}:
            continue
        try:
            net_q = float(row.get("net_charge_e") or 0.0)
        except Exception:
            net_q = 0.0
        if abs(net_q) > 1.0e-6:
            issues.append(
                {
                    "label": label,
                    "net_charge_e": net_q,
                    "note": "Neutral electrolyte species should normally remain charge-neutral.",
                }
            )
    return issues


def _stamp_charge_route(mol, *, charge_method: str, prefer_db: bool, require_db: bool, require_ready: bool) -> None:
    mol.SetProp("_yadonpy_charge_method", str(charge_method))
    mol.SetProp("_yadonpy_prefer_db", "1" if prefer_db else "0")
    mol.SetProp("_yadonpy_require_db", "1" if require_db else "0")
    mol.SetProp("_yadonpy_require_ready", "1" if require_ready else "0")


BASE_DIR = Path(__file__).resolve().parent
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"

restart_status = _env_bool("RESTART_STATUS", False)
set_run_options(restart=restart_status)

species_only = _env_bool("SPECIES_ONLY", False)
build_only = _env_bool("BUILD_ONLY", False)
export_only = _env_bool("EXPORT_ONLY", False)
skip_completed_benchmark = _env_bool("SKIP_COMPLETED_BENCHMARK", True)

gaff_variant = _normalize_gaff_variant(os.environ.get("YADONPY_GAFF_VARIANT"))
charge_mode = _normalize_charge_mode(os.environ.get("YADONPY_GAFF_CHARGE_MODE"))
charge_recipe = _charge_recipe_from_family(os.environ.get("YADONPY_CHARGE_DFT_FAMILY"))
resp_profile = _normalize_resp_profile(os.environ.get("YADONPY_RESP_PROFILE"))
solvent_source = _normalize_solvent_source(os.environ.get("YADONPY_SOLVENT_SOURCE"))
db_priority_mode = _normalize_db_priority(os.environ.get("YADONPY_DB_PRIORITY"))
run_gaff_variant_audit = _env_bool("RUN_GAFF_VARIANT_AUDIT", solvent_source == "qm")
cache_qm_solvents_to_repo_db = _env_bool("CACHE_QM_SOLVENTS_TO_REPO_DB", False)

EC_SMILES = "O=C1OCCO1"
EMC_SMILES = "CCOC(=O)OC"
DEC_SMILES = "CCOC(=O)OCC"
LI_SMILES = "[Li+]"
PF6_SMILES = "F[P-](F)(F)(F)(F)F"

temp_k = _env_float("TEMP_K", 298.15)
press_bar = _env_float("PRESS_BAR", 1.0)
prod_ns = _env_float("PROD_NS", 5.0)
eq21_final_ns = _env_float("EQ21_FINAL_NS", 0.8)
eq21_pre_nvt_ps = _env_float("EQ21_PRE_NVT_PS", 10.0)
initial_density_g_cm3 = _env_float("INITIAL_DENSITY_G_CM3", 0.05)
max_additional_rounds = _env_int("MAX_ADDITIONAL_ROUNDS", 4)
equilibration_mode = _normalize_equilibration_mode(os.environ.get("EQUILIBRATION_MODE"))
prod_constraints = _normalize_constraints(os.environ.get("PROD_CONSTRAINTS"), default="h-bonds")
prod_dt_ps = _env_float("PROD_DT_PS", 0.002)
performance_profile = _env_text("PERFORMANCE_PROFILE", "auto")
analysis_profile_requested = _env_text("ANALYSIS_PROFILE", "auto")
traj_ps_setting = _env_text("TRAJ_PS", os.environ.get("YADONPY_PROD_TRAJ_PS", "auto"))
energy_ps_setting = _env_text("ENERGY_PS", os.environ.get("YADONPY_PROD_ENERGY_PS", "auto"))
log_ps_setting = _env_text("LOG_PS", os.environ.get("YADONPY_PROD_LOG_PS", "auto"))
trr_ps_setting = os.environ.get("TRR_PS")
velocity_ps_setting = os.environ.get("VELOCITY_PS")
max_trajectory_frames = _env_int("MAX_TRAJECTORY_FRAMES", 50000)
max_atom_frames = _env_float("MAX_ATOM_FRAMES", 5.0e9)
rdf_frame_stride_setting = _env_text("RDF_FRAME_STRIDE", "auto")
rdf_bin_nm_setting = _env_text("RDF_BIN_NM", "auto")
rdf_rmax_nm_setting = _env_text("RDF_RMAX_NM", "auto")
msd_blocks = _env_int("MSD_BLOCKS", 4)
msd_block_min_ps = _env_float("MSD_BLOCK_MIN_PS", 500.0)
liquid_recovery_constraints = _normalize_constraints(os.environ.get("LIQUID_RECOVERY_CONSTRAINTS"), default="none")
additional_round_ns = _env_float("ADDITIONAL_ROUND_NS", 1.0)
allow_unconverged_production = _env_bool("ALLOW_UNCONVERGED_PRODUCTION", False)
liquid_hot_temp_k = _env_float("LIQUID_ANNEAL_HOT_TEMP_K", 600.0)
liquid_hot_press_bar = _env_float("LIQUID_ANNEAL_HOT_PRESS_BAR", 1000.0)
liquid_compact_press_bar = _env_float("LIQUID_ANNEAL_COMPACT_PRESS_BAR", max(liquid_hot_press_bar, 5000.0))
liquid_hot_nvt_ns = _env_float("LIQUID_ANNEAL_HOT_NVT_NS", 0.05)
liquid_compact_npt_ns = _env_float("LIQUID_ANNEAL_COMPACT_NPT_NS", 0.15)
liquid_hot_npt_ns = _env_float("LIQUID_ANNEAL_HOT_NPT_NS", 0.20)
liquid_cooling_npt_ns = _env_float("LIQUID_ANNEAL_COOLING_NPT_NS", 0.10)
liquid_recovery_hot_nvt_ns = _env_float("LIQUID_RECOVERY_HOT_NVT_NS", 0.03)
liquid_recovery_compact_npt_ns = _env_float("LIQUID_RECOVERY_COMPACT_NPT_NS", 0.25)
liquid_recovery_extend_max_rounds = _env_int("LIQUID_RECOVERY_EXTEND_MAX_ROUNDS", 4)
liquid_recovery_extend_ns = _env_float("LIQUID_RECOVERY_EXTEND_NS", 0.20)
polymer_recovery_warm_temp_k = _env_float("POLYMER_RECOVERY_WARM_TEMP_K", 0.0)
polymer_recovery_warm_nvt_ns = _env_float("POLYMER_RECOVERY_WARM_NVT_NS", 0.05)
polymer_recovery_compact_npt_ns = _env_float("POLYMER_RECOVERY_COMPACT_NPT_NS", 0.25)
polymer_recovery_extend_max_rounds = _env_int("POLYMER_RECOVERY_EXTEND_MAX_ROUNDS", 3)
polymer_recovery_extend_ns = _env_float("POLYMER_RECOVERY_EXTEND_NS", 0.20)
polymer_chain_warm_temp_k = _env_float("POLYMER_CHAIN_WARM_TEMP_K", 0.0)
polymer_chain_warm_nvt_ns = _env_float("POLYMER_CHAIN_WARM_NVT_NS", 0.10)

mpi = _env_int("MPI", 1)
omp = _env_int("OMP", 16)
gpu = _env_int("GPU", 1)
gpu_id = _env_int("GPU_ID", 0)

count_ec = _env_int("COUNT_EC", 120)
count_emc = _env_int("COUNT_EMC", 120)
count_dec = _env_int("COUNT_DEC", 120)
salt_pairs = _env_int("SALT_PAIRS", 45)

li_charge_scale = _env_float("LI_CHARGE_SCALE", 0.8)
pf6_charge_scale = _env_float("PF6_CHARGE_SCALE", 0.8)

psi4_omp = _env_int("PSI4_OMP", 64)
memory_mb = _env_int("MEM_MB", 20000)

work_dir_name = _env_text(
    "WORK_DIR_NAME",
    f"benchmark_carbonate_lipf6_gaff2_{gaff_variant}_{charge_recipe['family']}_work",
)
work_root = Path(_env_text("WORK_DIR", str(BASE_DIR / work_dir_name))).resolve()


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    work_root = workdir(work_root, restart=restart_status)
    completed_summary = work_root / "06_analysis" / "benchmark_summary.json"
    if restart_status and skip_completed_benchmark and completed_summary.exists() and not any((species_only, build_only, export_only)):
        try:
            completed_payload = json.loads(completed_summary.read_text(encoding="utf-8"))
        except Exception:
            completed_payload = {}
        completed_status = str(completed_payload.get("status") or "completed")
        completed_diffusion = completed_payload.get("diffusion_m2_s")
        if completed_status != "failed_equilibration_density_gate" and isinstance(completed_diffusion, dict):
            print(f"[SKIP] Existing completed benchmark_summary.json found at {completed_summary}")
            print(json.dumps(completed_diffusion, indent=2, ensure_ascii=False))
            raise SystemExit(0)

    build_dir = work_root.child("00_build_cell")
    ff = _build_ff_variant(gaff_variant)
    ion_ff = MERZ()
    resolved_db_priority = db_priority_mode
    if resolved_db_priority == "auto":
        resolved_db_priority = "repo_first" if solvent_source == "moldb" and resp_profile == "adaptive" else "default_first"

    if solvent_source == "moldb":
        ec = _load_ready_gaff_species(
            ff,
            EC_SMILES,
            label="EC",
            repo_db_dir=REPO_DB_DIR,
            charge_mode=charge_mode,
            db_priority=resolved_db_priority,
        )
        emc = _load_ready_gaff_species(
            ff,
            EMC_SMILES,
            label="EMC",
            repo_db_dir=REPO_DB_DIR,
            charge_mode=charge_mode,
            db_priority=resolved_db_priority,
        )
        dec = _load_ready_gaff_species(
            ff,
            DEC_SMILES,
            label="DEC",
            repo_db_dir=REPO_DB_DIR,
            charge_mode=charge_mode,
            db_priority=resolved_db_priority,
        )
    else:
        ec = _build_qm_ready_gaff_species(
            ff,
            EC_SMILES,
            label="EC",
            recipe=charge_recipe,
            resp_profile=resp_profile,
            work_root=work_root,
            psi4_omp=psi4_omp,
            mpi=mpi,
            omp=omp,
            memory_mb=memory_mb,
            repo_db_dir=REPO_DB_DIR,
            cache_to_repo_db=cache_qm_solvents_to_repo_db,
        )
        emc = _build_qm_ready_gaff_species(
            ff,
            EMC_SMILES,
            label="EMC",
            recipe=charge_recipe,
            resp_profile=resp_profile,
            work_root=work_root,
            psi4_omp=psi4_omp,
            mpi=mpi,
            omp=omp,
            memory_mb=memory_mb,
            repo_db_dir=REPO_DB_DIR,
            cache_to_repo_db=cache_qm_solvents_to_repo_db,
        )
        dec = _build_qm_ready_gaff_species(
            ff,
            DEC_SMILES,
            label="DEC",
            recipe=charge_recipe,
            resp_profile=resp_profile,
            work_root=work_root,
            psi4_omp=psi4_omp,
            mpi=mpi,
            omp=omp,
            memory_mb=memory_mb,
            repo_db_dir=REPO_DB_DIR,
            cache_to_repo_db=cache_qm_solvents_to_repo_db,
        )
    li = _assign_merz_ion(ion_ff, LI_SMILES, label="Li")
    pf6 = _load_ready_pf6(ff, repo_db_dir=REPO_DB_DIR, db_priority=resolved_db_priority)

    if solvent_source == "moldb":
        solvent_charge_method = f"{charge_mode.upper()}[MolDB-ready]"
        solvent_db_flags = {"prefer_db": True, "require_db": True, "require_ready": True}
    else:
        solvent_charge_method = f"RESP[{charge_recipe['label']}]"
        solvent_db_flags = {"prefer_db": False, "require_db": False, "require_ready": False}
    _stamp_charge_route(ec, charge_method=solvent_charge_method, **solvent_db_flags)
    _stamp_charge_route(emc, charge_method=solvent_charge_method, **solvent_db_flags)
    _stamp_charge_route(dec, charge_method=solvent_charge_method, **solvent_db_flags)
    _stamp_charge_route(li, charge_method="MERZ", prefer_db=False, require_db=False, require_ready=False)
    _stamp_charge_route(pf6, charge_method="RESP", prefer_db=True, require_db=True, require_ready=True)

    if run_gaff_variant_audit:
        gaff_variant_audit = _audit_gaff_variant_differences(
            recipe=charge_recipe,
            resp_profile=resp_profile,
            work_root=work_root,
            repo_db_dir=REPO_DB_DIR,
            psi4_omp=psi4_omp,
            mpi=mpi,
            omp=omp,
            memory_mb=memory_mb,
        )
    else:
        gaff_variant_audit = {
            "recipe": charge_recipe,
            "species": {},
            "skipped": True,
            "notes": "GAFF2 classic-vs-mod audit skipped for this run configuration.",
        }
    analysis_dir = work_root / "06_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(analysis_dir / "gaff_variant_audit.json", gaff_variant_audit)

    species_rows = [
        summarize_rdkit_species_forcefield(ec, label="EC", moltype_hint="EC", charge_scale=1.0),
        summarize_rdkit_species_forcefield(emc, label="EMC", moltype_hint="EMC", charge_scale=1.0),
        summarize_rdkit_species_forcefield(dec, label="DEC", moltype_hint="DEC", charge_scale=1.0),
        summarize_rdkit_species_forcefield(li, label="Li", moltype_hint="Li", charge_scale=li_charge_scale),
        summarize_rdkit_species_forcefield(pf6, label="PF6", moltype_hint="PF6", charge_scale=pf6_charge_scale),
    ]
    neutral_charge_issues = _neutral_charge_anomalies(species_rows)
    solvent_routes = {
        "EC": _extract_resp_route(ec, label="EC"),
        "EMC": _extract_resp_route(emc, label="EMC"),
        "DEC": _extract_resp_route(dec, label="DEC"),
    }
    equivalence_spread = {
        "EC": _equivalence_spread_diagnostic(ec, label="EC"),
        "EMC": _equivalence_spread_diagnostic(emc, label="EMC"),
        "DEC": _equivalence_spread_diagnostic(dec, label="DEC"),
    }

    species_summary = {
        "metadata": {
            "benchmark_name": "carbonate_lipf6_gaff2",
            "ff_variant": gaff_variant,
            "charge_mode": charge_mode,
            "resp_profile": resp_profile,
            "solvent_source": solvent_source,
            "db_priority": resolved_db_priority,
            "cache_qm_solvents_to_repo_db": cache_qm_solvents_to_repo_db,
            "run_gaff_variant_audit": run_gaff_variant_audit,
            "qm_charge_recipe": charge_recipe,
            "resolved_qm_recipes": solvent_routes,
            "solvent_charge_method": solvent_charge_method,
            "pf6_charge_method": "RESP",
            "li_charge_method": "MERZ",
            "species": ["EC", "EMC", "DEC", "Li", "PF6"],
            "counts": {"EC": count_ec, "EMC": count_emc, "DEC": count_dec, "Li": salt_pairs, "PF6": salt_pairs},
            "charge_scale": {"EC": 1.0, "EMC": 1.0, "DEC": 1.0, "Li": li_charge_scale, "PF6": pf6_charge_scale},
            "eq21_final_ns": eq21_final_ns,
            "eq21_pre_nvt_ps": eq21_pre_nvt_ps,
            "prod_ns": prod_ns,
            "equilibration_mode_requested": equilibration_mode,
            "max_additional_rounds": max_additional_rounds,
            "additional_round_ns": additional_round_ns,
            "allow_unconverged_production": allow_unconverged_production,
            "prod_constraints": prod_constraints,
            "prod_dt_ps": prod_dt_ps,
            "msd_blocks": msd_blocks,
            "msd_block_min_ps": msd_block_min_ps,
            "liquid_anneal": {
                "hot_temp_K": liquid_hot_temp_k,
                "hot_press_bar": liquid_hot_press_bar,
                "compact_press_bar": liquid_compact_press_bar,
                "hot_nvt_ns": liquid_hot_nvt_ns,
                "compact_npt_ns": liquid_compact_npt_ns,
                "hot_npt_ns": liquid_hot_npt_ns,
                "cooling_npt_ns": liquid_cooling_npt_ns,
                "density_recovery": {
                    "hot_nvt_ns": liquid_recovery_hot_nvt_ns,
                    "compact_npt_ns": liquid_recovery_compact_npt_ns,
                    "extend_max_rounds": liquid_recovery_extend_max_rounds,
                    "extend_ns": liquid_recovery_extend_ns,
                    "constraints": liquid_recovery_constraints,
                    "notes": "No-polymer failed density gates use high-pressure recovery rounds before the normal four-round stop condition is evaluated.",
                },
            },
            "polymer_adaptive_relaxation": {
                "density_recovery": {
                    "warm_temp_K": None if polymer_recovery_warm_temp_k <= 0.0 else polymer_recovery_warm_temp_k,
                    "warm_nvt_ns": polymer_recovery_warm_nvt_ns,
                    "compact_npt_ns": polymer_recovery_compact_npt_ns,
                    "extend_max_rounds": polymer_recovery_extend_max_rounds,
                    "extend_ns": polymer_recovery_extend_ns,
                    "pressure_ladder_bar": [500.0, 1000.0, 2000.0, 5000.0],
                    "constraints": "none",
                },
                "chain_relaxation": {
                    "warm_temp_K": None if polymer_chain_warm_temp_k <= 0.0 else polymer_chain_warm_temp_k,
                    "warm_nvt_ns": polymer_chain_warm_nvt_ns,
                    "constraints": "none",
                },
            },
            "expected_diffusion_trend": "EMC > DEC > EC (literature-guided target for mixed linear/cyclic carbonate electrolyte)",
            "neutral_charge_issues": neutral_charge_issues,
            "literature_alignment_notes": {
                "coordination_target": "EC should remain at least as strong a Li carbonyl ligand as EMC; DEC should not remain systematically stronger than EC.",
                "transport_target": "EMC should remain more transport-favorable than DEC in mixed carbonate electrolyte benchmark.",
            },
        },
        "species_pre_export": species_rows,
        "species_charge_sanity": {
            "EC": _summarize_carbonate_charge_features(ec, label="EC"),
            "EMC": _summarize_carbonate_charge_features(emc, label="EMC"),
            "DEC": _summarize_carbonate_charge_features(dec, label="DEC"),
        },
        "species_equivalence_spread": equivalence_spread,
    }
    _dump_json(analysis_dir / "species_forcefield_summary.json", species_summary)

    if species_only:
        print("[SPECIES-ONLY] Wrote species_forcefield_summary.json")
        print(json.dumps(species_summary["metadata"], indent=2, ensure_ascii=False))
        raise SystemExit(0)

    cell_mols = []
    counts = []
    charge_scale = []
    for mol, count in ((ec, count_ec), (emc, count_emc), (dec, count_dec)):
        if int(count) > 0:
            cell_mols.append(mol)
            counts.append(int(count))
            charge_scale.append(1.0)
    if salt_pairs > 0:
        cell_mols.extend([li, pf6])
        counts.extend([salt_pairs, salt_pairs])
        charge_scale.extend([li_charge_scale, pf6_charge_scale])
    if not cell_mols:
        raise ValueError("At least one molecule count must be positive.")
    estimated_atoms = int(sum(int(count) * int(mol.GetNumAtoms()) for mol, count in zip(cell_mols, counts)))
    io_policy = resolve_io_analysis_policy(
        prod_ns=float(prod_ns),
        atom_count=estimated_atoms,
        performance_profile=performance_profile,
        analysis_profile=analysis_profile_requested,
        traj_ps=traj_ps_setting,
        energy_ps=energy_ps_setting,
        log_ps=log_ps_setting,
        trr_ps=trr_ps_setting,
        velocity_ps=velocity_ps_setting,
        rdf_frame_stride=rdf_frame_stride_setting,
        rdf_rmax_nm=rdf_rmax_nm_setting,
        rdf_bin_nm=rdf_bin_nm_setting,
        msd_selected_species=["EC", "EMC", "DEC", "Li", "PF6"],
        max_trajectory_frames=max_trajectory_frames,
        max_atom_frames=max_atom_frames,
    )

    ac = poly.amorphous_cell(
        cell_mols,
        counts,
        charge_scale=charge_scale,
        density=initial_density_g_cm3,
        work_dir=build_dir,
    )

    if build_only:
        print(f"[BUILD-ONLY] Finished cell construction at {build_dir}")
        raise SystemExit(0)

    has_polymer = eq.cell_meta_contains_polymer(ac)
    selected_equilibration_mode = equilibration_mode
    if selected_equilibration_mode == "auto":
        selected_equilibration_mode = "eq21" if has_polymer else "liquid_anneal"
    species_summary["metadata"]["has_polymer"] = bool(has_polymer)
    species_summary["metadata"]["equilibration_mode"] = selected_equilibration_mode
    species_summary["metadata"]["estimated_total_atoms"] = int(estimated_atoms)
    species_summary["metadata"]["performance_policy"] = io_policy.to_dict()
    _dump_json(analysis_dir / "species_forcefield_summary.json", species_summary)

    eqmd = eq.EQ21step(ac, work_dir=work_root) if selected_equilibration_mode == "eq21" else eq.LiquidAnneal(ac, work_dir=work_root)
    if export_only:
        exported = eqmd.ensure_system_exported()
        print(f"[EXPORT-ONLY] Exported 02_system at {exported.system_top.parent}")
        raise SystemExit(0)

    if selected_equilibration_mode == "eq21":
        ac = eqmd.exec(
            temp=temp_k,
            press=press_bar,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            time=eq21_final_ns,
            eq21_pre_nvt_ps=eq21_pre_nvt_ps,
        )
    else:
        ac = eqmd.exec(
            temp=temp_k,
            press=press_bar,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            time=eq21_final_ns,
            hot_temp=liquid_hot_temp_k,
            hot_pressure_bar=liquid_hot_press_bar,
            compact_pressure_bar=liquid_compact_press_bar,
            hot_nvt_ns=liquid_hot_nvt_ns,
            compact_npt_ns=liquid_compact_npt_ns,
            hot_npt_ns=liquid_hot_npt_ns,
            cooling_npt_ns=liquid_cooling_npt_ns,
            constraints=prod_constraints,
            dt_ps=prod_dt_ps,
        )
    analy = eqmd.analyze()
    analy.get_all_prop(temp=temp_k, press=press_bar, save=True)
    result = analy.check_eq()
    latest_equilibrated_gro = eqmd.final_gro()
    if restart_status and not result:
        restart_gro = _restart_latest_equilibrated_gro(work_root, latest_equilibrated_gro)
        if restart_gro != latest_equilibrated_gro:
            restart_analy = _analyze_restart_stage(work_root, restart_gro)
            if restart_analy is not None:
                restart_analy.get_all_prop(temp=temp_k, press=press_bar, save=True)
                result = restart_analy.check_eq()
        latest_equilibrated_gro = restart_gro

    additional_rounds_run = 0
    additional_round_strategies: list[str] = []
    for _ in range(max_additional_rounds):
        if result:
            break
        equilibrium_payload = _load_equilibrium_payload(analysis_dir)
        strategy = eq.select_relaxation_strategy(equilibrium_payload, has_polymer=has_polymer)
        if strategy == "production":
            break
        additional_round_strategies.append(strategy)
        if strategy == "liquid_density_recovery":
            eq_more = eq.LiquidDensityRecovery(ac, work_dir=work_root)
            ac = eq_more.exec(
                temp=temp_k,
                press=press_bar,
                mpi=mpi,
                omp=omp,
                gpu=gpu,
                gpu_id=gpu_id,
                time=additional_round_ns,
                hot_temp=liquid_hot_temp_k,
                hot_pressure_bar=liquid_hot_press_bar,
                compact_pressure_bar=liquid_compact_press_bar,
                hot_nvt_ns=liquid_recovery_hot_nvt_ns,
                compact_npt_ns=liquid_recovery_compact_npt_ns,
                cooling_npt_ns=liquid_cooling_npt_ns,
                compact_extend=True,
                compact_extend_max_rounds=liquid_recovery_extend_max_rounds,
                compact_extend_ns=liquid_recovery_extend_ns,
                dt_ps=prod_dt_ps,
                hot_dt_ps=min(float(prod_dt_ps), 0.001),
                constraints=liquid_recovery_constraints,
                start_gro=latest_equilibrated_gro,
            )
        elif strategy == "polymer_density_recovery":
            eq_more = eq.PolymerDensityRecovery(ac, work_dir=work_root)
            ac = eq_more.exec(
                temp=temp_k,
                press=press_bar,
                mpi=mpi,
                omp=omp,
                gpu=gpu,
                gpu_id=gpu_id,
                time=additional_round_ns,
                warm_temp=(None if polymer_recovery_warm_temp_k <= 0.0 else polymer_recovery_warm_temp_k),
                warm_nvt_ns=polymer_recovery_warm_nvt_ns,
                compact_npt_ns=polymer_recovery_compact_npt_ns,
                compact_extend=True,
                compact_extend_max_rounds=polymer_recovery_extend_max_rounds,
                compact_extend_ns=polymer_recovery_extend_ns,
                dt_ps=min(float(prod_dt_ps), 0.001),
                start_gro=latest_equilibrated_gro,
            )
        elif strategy == "polymer_chain_relaxation":
            eq_more = eq.PolymerChainRelaxation(ac, work_dir=work_root)
            ac = eq_more.exec(
                temp=temp_k,
                press=press_bar,
                mpi=mpi,
                omp=omp,
                gpu=gpu,
                gpu_id=gpu_id,
                time=additional_round_ns,
                warm_temp=(None if polymer_chain_warm_temp_k <= 0.0 else polymer_chain_warm_temp_k),
                warm_nvt_ns=polymer_chain_warm_nvt_ns,
                dt_ps=min(float(prod_dt_ps), 0.001),
                start_gro=latest_equilibrated_gro,
            )
        else:
            additional_constraints = "none" if has_polymer else prod_constraints
            additional_dt_ps = min(float(prod_dt_ps), 0.001) if _normalize_constraints(additional_constraints, default="none") == "none" else float(prod_dt_ps)
            additional_mdp_overrides = None
            additional_gpu_offload_mode = "auto"
            additional_gpu = gpu
            additional_gpu_id = gpu_id
            if not has_polymer:
                additional_gpu_offload_mode = "auto"
                additional_mdp_overrides = None
            eq_more = eq.Additional(ac, work_dir=work_root)
            ac = eq_more.exec(
                temp=temp_k,
                press=press_bar,
                mpi=mpi,
                omp=omp,
                gpu=additional_gpu,
                gpu_id=additional_gpu_id,
                time=additional_round_ns,
                dt_ps=additional_dt_ps,
                constraints=additional_constraints,
                gpu_offload_mode=additional_gpu_offload_mode,
                mdp_overrides=additional_mdp_overrides,
                start_gro=latest_equilibrated_gro,
                skip_rebuild=True,
                micro_relax=False,
            )
        latest_equilibrated_gro = eq_more.final_gro()
        additional_rounds_run += 1
        analy = eq_more.analyze()
        analy.get_all_prop(temp=temp_k, press=press_bar, save=True)
        result = analy.check_eq()

    equilibrium_payload = _load_equilibrium_payload(analysis_dir)
    equilibrium_status = _transport_confidence_from_equilibrium(equilibrium_payload, result)
    if not result:
        fail_summary = {
            "metadata": species_summary["metadata"],
            "analysis": {
                "analysis_profile": io_policy.analysis_profile,
                "performance_policy": io_policy.to_dict(),
            },
            "equilibration_ok": False,
            "density_warning_severity": equilibrium_status["density_warning_severity"],
            "transport_confidence": equilibrium_status["transport_confidence"],
            "equilibration": {
                "mode": selected_equilibration_mode,
                "has_polymer": bool(has_polymer),
                "additional_rounds_run": int(additional_rounds_run),
                "max_additional_rounds": int(max_additional_rounds),
                "additional_round_strategies": additional_round_strategies,
                "density_gate": equilibrium_status["density_gate"],
                "rg_gate": equilibrium_payload.get("rg_gate") if isinstance(equilibrium_payload, dict) else None,
                "relaxation_state": equilibrium_payload.get("relaxation_state") if isinstance(equilibrium_payload, dict) else None,
            },
            "status": "failed_equilibration_density_gate",
            "message": "Equilibration did not converge after the configured additional rounds; production/MSD was not run.",
        }
        _dump_json(analysis_dir / "benchmark_summary.json", fail_summary)
        utils.radon_print(
            "[ERROR] Equilibration density/Rg gate is still failing after additional rounds; "
            "production/MSD will not run. Set ALLOW_UNCONVERGED_PRODUCTION=1 only for diagnostics.",
            level=2,
        )
        if not allow_unconverged_production:
            raise SystemExit(2)
        utils.radon_print(
            "[WARN] ALLOW_UNCONVERGED_PRODUCTION=1: continuing for diagnostics only; "
            "diffusion/transport values may be severely overestimated.",
            level=2,
        )

    npt = eq.NPT(ac, work_dir=work_root)
    ac = npt.exec(
        temp=temp_k,
        press=press_bar,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        time=prod_ns,
        dt_ps=prod_dt_ps,
        constraints=prod_constraints,
        traj_ps=io_policy.traj_ps,
        energy_ps=io_policy.energy_ps,
        log_ps=io_policy.log_ps,
        trr_ps=io_policy.trr_ps,
        velocity_ps=io_policy.velocity_ps,
        performance_profile=io_policy.performance_profile,
        analysis_profile=io_policy.analysis_profile,
        max_trajectory_frames=io_policy.max_trajectory_frames,
        max_atom_frames=io_policy.max_atom_frames,
        start_gro=latest_equilibrated_gro,
    )

    analy = npt.analyze()
    basic = analy.get_all_prop(
        temp=temp_k,
        press=press_bar,
        save=True,
        include_polymer_metrics=bool(io_policy.include_polymer_metrics),
        analysis_profile=io_policy.analysis_profile,
    )
    msd = analy.msd(analysis_profile=io_policy.analysis_profile)
    msd_block_diagnostic = _msd_block_diffusion_diagnostic(
        analy,
        full_msd=msd,
        n_blocks=msd_blocks,
        min_block_ps=msd_block_min_ps,
    )
    _dump_json(analysis_dir / "msd_block_diffusion.json", msd_block_diagnostic)
    if salt_pairs > 0:
        rdf = analy.rdf(
            center_mol=li,
            analysis_profile=io_policy.analysis_profile,
            bin_nm=float(io_policy.rdf_bin_nm),
            r_max_nm=io_policy.rdf_rmax_nm,
            frame_stride=int(io_policy.rdf_frame_stride),
        )
        sigma = analy.sigma(msd=msd, temp_k=temp_k, eh_mode="gmx_current_only")
        coordination = {
            "EC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "ec"),
            "EMC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "emc"),
            "DEC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "dec"),
            "PF6_coordination_fluorine": _extract_rdf_site(rdf, "pf6:coordination_fluorine"),
            "PF6_fluorine_site": _extract_rdf_site(rdf, "pf6:fluorine_site"),
        }
        coordination_preference = _coordination_preference_summary(
            coordination,
            {"EC": count_ec, "EMC": count_emc, "DEC": count_dec},
        )
    else:
        sigma = {
            "sigma_ne_upper_bound_S_m": None,
            "sigma_eh_total_S_m": None,
            "haven_ratio": None,
            "eh": {"confidence": "skipped", "quality_note": "salt-free solvent mixture"},
        }
        coordination = {}
        coordination_preference = {
            "total_cn_shell": None,
            "notes": "Skipped for salt-free solvent mixture because no Li center group is present.",
        }

    diffusion_m2_s = {
        "EC": _extract_default_diffusivity(msd, "EC"),
        "EMC": _extract_default_diffusivity(msd, "EMC"),
        "DEC": _extract_default_diffusivity(msd, "DEC"),
        "Li": _extract_default_diffusivity(msd, "Li"),
        "PF6": _extract_default_diffusivity(msd, "PF6"),
    }
    summary = {
        "metadata": species_summary["metadata"],
        "equilibration_ok": equilibrium_status["equilibration_ok"],
        "density_warning_severity": equilibrium_status["density_warning_severity"],
        "transport_confidence": equilibrium_status["transport_confidence"],
        "equilibration": {
            "mode": selected_equilibration_mode,
            "has_polymer": bool(has_polymer),
            "additional_rounds_run": int(additional_rounds_run),
            "max_additional_rounds": int(max_additional_rounds),
            "additional_round_strategies": additional_round_strategies,
            "density_gate": equilibrium_status["density_gate"],
            "rg_gate": equilibrium_payload.get("rg_gate") if isinstance(equilibrium_payload, dict) else None,
            "relaxation_state": equilibrium_payload.get("relaxation_state") if isinstance(equilibrium_payload, dict) else None,
        },
        "basic_properties": basic.get("basic_properties", {}),
        "analysis": {
            "analysis_profile": io_policy.analysis_profile,
            "performance_policy": io_policy.to_dict(),
            "include_polymer_metrics": bool(io_policy.include_polymer_metrics),
            "rdf": {
                "bin_nm": float(io_policy.rdf_bin_nm),
                "r_max_nm": io_policy.rdf_rmax_nm,
                "frame_stride": int(io_policy.rdf_frame_stride),
            },
            "msd": {
                "selected_species": io_policy.msd_selected_species,
                "default_metric_only": bool(io_policy.msd_default_metric_only),
            },
        },
        "diffusion_m2_s": diffusion_m2_s,
        "solvent_diffusion_diagnostic": _solvent_diffusion_diagnostic(diffusion_m2_s),
        "msd_block_diffusion_diagnostic": msd_block_diagnostic,
        "conductivity": {
            "sigma_ne_upper_bound_S_m": sigma.get("sigma_ne_upper_bound_S_m"),
            "sigma_eh_total_S_m": sigma.get("sigma_eh_total_S_m"),
            "haven_ratio": sigma.get("haven_ratio"),
            "eh_confidence": ((sigma.get("eh") or {}) if isinstance(sigma.get("eh"), dict) else {}).get("confidence"),
            "eh_quality_note": ((sigma.get("eh") or {}) if isinstance(sigma.get("eh"), dict) else {}).get("quality_note"),
        },
        "coordination": coordination,
    }
    summary["coordination_preference"] = coordination_preference
    _dump_json(analysis_dir / "benchmark_summary.json", summary)
    print("[BENCHMARK] carbonate_lipf6_gaff2 completed")
    print(json.dumps(summary["diffusion_m2_s"], indent=2, ensure_ascii=False))
