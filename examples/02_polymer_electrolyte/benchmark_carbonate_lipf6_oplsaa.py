from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from yadonpy.core import poly, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2_mod, OPLSAA
from yadonpy.gmx.analysis.structured import build_msd_metric_catalog, compute_msd_series
from yadonpy.gmx.topology import parse_system_top
from yadonpy.runtime import set_run_options
from yadonpy.sim.analyzer import AnalyzeResult
from yadonpy.sim.benchmarking import _dump_json, summarize_rdkit_species_forcefield
from yadonpy.sim.preset import eq


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    return float(raw)


def _env_text(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return str(default)
    text = str(raw).strip()
    return text if text else str(default)


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


def _normalize_charge_mode(raw: str | None) -> str:
    mode = str(raw or "resp").strip().lower()
    if mode in {"native", "opls", "oplsaa"}:
        return "opls"
    if mode in {"resp2", "resp_2"}:
        return "resp2"
    return "resp"


def _json_prop(mol, key: str) -> dict[str, Any] | None:
    try:
        if mol.HasProp(key):
            value = json.loads(mol.GetProp(key))
            if isinstance(value, dict):
                return value
    except Exception:
        pass
    return None


def _extract_resp_route(mol, *, label: str) -> dict[str, Any]:
    route = {
        "label": str(label),
        "resp_profile": None,
        "qm_recipe": None,
        "constraint_mode": None,
        "equivalence_group_count": 0,
    }
    try:
        if mol.HasProp("_yadonpy_resp_profile"):
            route["resp_profile"] = str(mol.GetProp("_yadonpy_resp_profile"))
    except Exception:
        pass
    qm_recipe = _json_prop(mol, "_yadonpy_qm_recipe_json")
    if isinstance(qm_recipe, dict):
        route["qm_recipe"] = qm_recipe
        if route["resp_profile"] is None:
            route["resp_profile"] = qm_recipe.get("resp_profile")
    constraints = _json_prop(mol, "_yadonpy_resp_constraints_json")
    if isinstance(constraints, dict):
        route["constraint_mode"] = constraints.get("mode")
        route["equivalence_group_count"] = int(len(constraints.get("equivalence_groups") or []))
        if route["resp_profile"] is None:
            route["resp_profile"] = constraints.get("resp_profile")
    return route


def _equivalence_spread_diagnostic(mol, *, label: str) -> dict[str, Any]:
    constraints = _json_prop(mol, "_yadonpy_resp_constraints_json") or {}
    groups = list(constraints.get("equivalence_groups") or [])
    diagnostics = []
    for group in groups:
        idxs = sorted({int(i) for i in group})
        if len(idxs) <= 1:
            continue
        values = []
        for idx in idxs:
            atom = mol.GetAtomWithIdx(idx)
            if not atom.HasProp("AtomicCharge"):
                values = []
                break
            values.append(float(atom.GetDoubleProp("AtomicCharge")))
        diagnostics.append(
            {
                "atom_indices": idxs,
                "symbols": [str(mol.GetAtomWithIdx(idx).GetSymbol()) for idx in idxs],
                "atomic_charge_spread_e": float(max(values) - min(values)) if values else None,
            }
        )
    max_spread = max(
        (float(item["atomic_charge_spread_e"]) for item in diagnostics if item.get("atomic_charge_spread_e") is not None),
        default=0.0,
    )
    return {"label": str(label), "group_count": len(diagnostics), "max_spread_e": float(max_spread), "groups": diagnostics}


def _load_ready_opls_species(
    ff: OPLSAA,
    smiles: str,
    *,
    label: str,
    repo_db_dir: Path,
    charge_mode: str,
):
    last_exc: Exception | None = None
    db_charge = "RESP2" if charge_mode == "resp2" else "RESP"
    for db_dir, db_label in ((None, "default"), (repo_db_dir, "repo")):
        try:
            mol = ff.mol_rdkit(
                smiles,
                name=label,
                db_dir=db_dir,
                charge=db_charge,
                require_ready=True,
                prefer_db=True,
            )
            assign_charge = "opls" if charge_mode == "opls" else None
            mol = ff.ff_assign(mol, charge=assign_charge, report=False)
            if not mol:
                raise RuntimeError(f"Cannot assign OPLS-AA parameters for {label}.")
            if assign_charge == "opls":
                print(f"[MolDB] loaded {label} geometry from {db_label} db and switched to built-in OPLS-AA charges")
            else:
                print(f"[MolDB] loaded {label} with {db_charge} charges from {db_label} db")
            return mol
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"{label} is expected to be ready in MolDB for the OPLS-AA benchmark.") from last_exc


def _assign_builtin_opls_ion(ff: OPLSAA, smiles: str, *, label: str):
    mol = ff.mol(smiles, charge="opls", require_ready=False, prefer_db=False)
    mol = ff.ff_assign(mol, charge="opls", report=False)
    if not mol:
        raise RuntimeError(f"Cannot assign built-in OPLS-AA ion parameters for {label}.")
    print(f"[OPLS-AA] assigned built-in ion parameters for {label}")
    return mol


def _load_pf6_with_builtin_charges(*, ion_ff: OPLSAA, repo_db_dir: Path):
    gaff_ff = GAFF2_mod()
    last_exc: Exception | None = None
    opls_probe = ion_ff.mol(PF6_SMILES, charge="opls", require_ready=False, prefer_db=False)
    if not ion_ff.assign_ptypes(opls_probe, charge="opls"):
        raise RuntimeError("Cannot build the PF6 OPLS-AA atom-type probe from SMILES.")

    for db_dir, db_label in ((None, "default"), (repo_db_dir, "repo")):
        try:
            pf6 = gaff_ff.mol_rdkit(
                PF6_SMILES,
                name="PF6",
                db_dir=db_dir,
                charge="RESP",
                require_ready=True,
                prefer_db=True,
            )
            pf6 = gaff_ff.ff_assign(pf6, bonded="DRIH", report=False)
            if not pf6:
                raise RuntimeError("Cannot restore PF6 DRIH bonded topology from MolDB.")
            if pf6.GetNumAtoms() != opls_probe.GetNumAtoms():
                raise RuntimeError("PF6 probe atom count does not match MolDB topology.")

            for src_atom, dst_atom in zip(opls_probe.GetAtoms(), pf6.GetAtoms()):
                if src_atom.GetSymbol() != dst_atom.GetSymbol():
                    raise RuntimeError("PF6 probe atom ordering does not match MolDB topology.")
                dst_atom.SetProp("ff_btype", src_atom.GetProp("ff_btype"))
                dst_atom.SetProp("ff_type", src_atom.GetProp("ff_type"))
                dst_atom.SetDoubleProp("ff_sigma", src_atom.GetDoubleProp("ff_sigma"))
                dst_atom.SetDoubleProp("ff_epsilon", src_atom.GetDoubleProp("ff_epsilon"))
                dst_atom.SetDoubleProp("AtomicCharge", src_atom.GetDoubleProp("AtomicCharge"))
                if src_atom.HasProp("ff_desc"):
                    dst_atom.SetProp("ff_desc", src_atom.GetProp("ff_desc"))
            pf6.SetProp("ff_name", str(ion_ff.name))
            pf6.SetProp("ff_class", str(ion_ff.ff_class))
            pf6.SetProp("pair_style", str(ion_ff.pair_style))
            print(
                "[OPLS-AA] loaded PF6 bonded topology from "
                f"{db_label} db and replaced atom types / charges with built-in OPLS-AA values"
            )
            return pf6
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        "PF6 is expected to exist in MolDB with bonded='DRIH' for this OPLS-AA benchmark."
    ) from last_exc


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
    import numpy as np

    starts: list[float] = []
    ends: list[float] = []
    for label in labels:
        metric = _extract_default_msd_metric_record(msd, label)
        try:
            start_raw = metric.get("trajectory_time_start_ps")
            end_raw = metric.get("trajectory_time_end_ps")
            if start_raw is not None and end_raw is not None:
                start = float(start_raw)
                end = float(end_raw)
                if np.isfinite(start) and np.isfinite(end) and end > start:
                    starts.append(start)
                    ends.append(end)
                    continue
        except Exception:
            pass
    if not starts or not ends:
        return None, None
    return float(min(starts)), float(max(ends))


def _summarize_msd_block_diffusion(
    blocks: list[dict[str, Any]],
    *,
    expected_order: tuple[str, ...] = ("EMC", "DEC", "EC"),
) -> dict[str, Any]:
    import numpy as np

    valid_blocks = [block for block in blocks if isinstance(block.get("diffusion_m2_s"), dict)]
    if not valid_blocks:
        return {"status": "skipped", "reason": "no_valid_block_diffusion", "blocks": blocks}
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
        if values:
            arr = np.asarray(values, dtype=float)
            std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
            mean = float(np.mean(arr))
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
    elif all(frac >= 0.75 for frac in pairwise_fractions) and (match_fraction is None or match_fraction >= 0.75):
        ranking_confidence = "supports_expected"
    elif any(0.25 < frac < 0.75 for frac in pairwise_fractions):
        ranking_confidence = "ambiguous"
    else:
        ranking_confidence = "contradicts_expected"
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
        "blocks": blocks,
    }


def _msd_block_diffusion_diagnostic(
    analy: AnalyzeResult,
    *,
    full_msd: dict[str, Any],
    n_blocks: int,
    min_block_ps: float = 500.0,
    labels: tuple[str, ...] = ("EC", "EMC", "DEC"),
) -> dict[str, Any]:
    import numpy as np

    n_blocks = int(max(0, n_blocks))
    if n_blocks < 2:
        return {"status": "skipped", "reason": "MSD_BLOCKS<2", "n_blocks_requested": n_blocks}
    start_ps, end_ps = _default_msd_trajectory_bounds(full_msd, labels=labels)
    if start_ps is None or end_ps is None or end_ps <= start_ps:
        return {"status": "skipped", "reason": "trajectory_time_bounds_unavailable", "n_blocks_requested": n_blocks}
    duration_ps = float(end_ps - start_ps)
    block_count = min(n_blocks, int(np.floor(duration_ps / max(float(min_block_ps), 1.0e-12))))
    if block_count < 2:
        return {
            "status": "skipped",
            "reason": "trajectory_too_short_for_blocks",
            "duration_ps": duration_ps,
            "min_block_ps": float(min_block_ps),
        }
    topo = parse_system_top(Path(analy.top))
    system_dir = analy._system_dir()
    metric_catalog = build_msd_metric_catalog(topo, system_dir)
    xtc_path = analy._analysis_xtc_path()
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
            _moltype, entry = catalog_item
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

ff = OPLSAA()
ion_ff = OPLSAA()

charge_mode = _normalize_charge_mode(os.environ.get("YADONPY_OPLS_CHARGE_MODE"))

EC_SMILES = "O=C1OCCO1"
EMC_SMILES = "CCOC(=O)OC"
DEC_SMILES = "CCOC(=O)OCC"
LI_SMILES = "[Li+]"
PF6_SMILES = "F[P-](F)(F)(F)(F)F"

temp_k = _env_float("TEMP_K", 298.15)
press_bar = _env_float("PRESS_BAR", 1.0)
prod_ns = _env_float("PROD_NS", 5.0)
initial_density_g_cm3 = _env_float("INITIAL_DENSITY_G_CM3", 0.05)
max_additional_rounds = _env_int("MAX_ADDITIONAL_ROUNDS", 4)
prod_constraints = _normalize_constraints(os.environ.get("PROD_CONSTRAINTS"), default="h-bonds")
prod_dt_ps = _env_float("PROD_DT_PS", 0.002)
msd_blocks = _env_int("MSD_BLOCKS", 4)
msd_block_min_ps = _env_float("MSD_BLOCK_MIN_PS", 500.0)

mpi = _env_int("MPI", 1)
omp = _env_int("OMP", 16)
gpu = _env_int("GPU", 1)
gpu_id = _env_int("GPU_ID", 0)

count_ec = _env_int("COUNT_EC", 40)
count_emc = _env_int("COUNT_EMC", 50)
count_dec = _env_int("COUNT_DEC", 20)
salt_pairs = _env_int("SALT_PAIRS", 15)

li_charge_scale = _env_float("LI_CHARGE_SCALE", 0.8)
pf6_charge_scale = _env_float("PF6_CHARGE_SCALE", 0.8)

work_dir_name = _env_text("WORK_DIR_NAME", "benchmark_carbonate_lipf6_oplsaa_work")
work_root = Path(_env_text("WORK_DIR", str(BASE_DIR / work_dir_name))).resolve()


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    work_root = workdir(work_root, restart=restart_status)
    build_dir = work_root.child("00_build_cell")

    ec = _load_ready_opls_species(ff, EC_SMILES, label="EC", repo_db_dir=REPO_DB_DIR, charge_mode=charge_mode)
    emc = _load_ready_opls_species(ff, EMC_SMILES, label="EMC", repo_db_dir=REPO_DB_DIR, charge_mode=charge_mode)
    dec = _load_ready_opls_species(ff, DEC_SMILES, label="DEC", repo_db_dir=REPO_DB_DIR, charge_mode=charge_mode)
    li = _assign_builtin_opls_ion(ion_ff, LI_SMILES, label="Li")
    pf6 = _load_pf6_with_builtin_charges(ion_ff=ion_ff, repo_db_dir=REPO_DB_DIR)

    solvent_charge_method = "RESP2" if charge_mode == "resp2" else "RESP"
    _stamp_charge_route(ec, charge_method=solvent_charge_method, prefer_db=True, require_db=True, require_ready=True)
    _stamp_charge_route(emc, charge_method=solvent_charge_method, prefer_db=True, require_db=True, require_ready=True)
    _stamp_charge_route(dec, charge_method=solvent_charge_method, prefer_db=True, require_db=True, require_ready=True)
    _stamp_charge_route(li, charge_method="opls", prefer_db=False, require_db=False, require_ready=False)
    _stamp_charge_route(pf6, charge_method="RESP", prefer_db=True, require_db=True, require_ready=True)

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
            "benchmark_name": "carbonate_lipf6_oplsaa",
            "charge_mode": charge_mode,
            "solvent_charge_method": solvent_charge_method,
            "resolved_qm_recipes": solvent_routes,
            "pf6_charge_method": "RESP",
            "species": ["EC", "EMC", "DEC", "Li", "PF6"],
            "counts": {"EC": count_ec, "EMC": count_emc, "DEC": count_dec, "Li": salt_pairs, "PF6": salt_pairs},
            "charge_scale": {"EC": 1.0, "EMC": 1.0, "DEC": 1.0, "Li": li_charge_scale, "PF6": pf6_charge_scale},
            "expected_diffusion_trend": "EMC > DEC > EC (literature-guided target for mixed linear/cyclic carbonate electrolyte)",
            "neutral_charge_issues": neutral_charge_issues,
        },
        "species_pre_export": species_rows,
        "species_equivalence_spread": equivalence_spread,
    }
    analysis_dir = work_root / "06_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
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

    eqmd = eq.LiquidAnneal(ac, work_dir=work_root)
    if export_only:
        exported = eqmd.ensure_system_exported()
        print(f"[EXPORT-ONLY] Exported 02_system at {exported.system_top.parent}")
        raise SystemExit(0)

    ac = eqmd.exec(temp=temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
    latest_equilibrated_gro = eqmd.final_gro()
    analy = eqmd.analyze()
    analy.get_all_prop(temp=temp_k, press=press_bar, save=True)
    result = analy.check_eq()

    for _ in range(max_additional_rounds):
        if result:
            break
        eq_more = eq.Additional(ac, work_dir=work_root)
        ac = eq_more.exec(
            temp=temp_k,
            press=press_bar,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            constraints="none",
            start_gro=latest_equilibrated_gro,
            skip_rebuild=True,
            micro_relax=False,
        )
        latest_equilibrated_gro = eq_more.final_gro()
        analy = eq_more.analyze()
        analy.get_all_prop(temp=temp_k, press=press_bar, save=True)
        result = analy.check_eq()

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
    )

    analy = npt.analyze()
    basic = analy.get_all_prop(temp=temp_k, press=press_bar, save=True)
    rdf = analy.rdf(center_mol=li)
    msd = analy.msd()
    msd_block_diagnostic = _msd_block_diffusion_diagnostic(
        analy,
        full_msd=msd,
        n_blocks=msd_blocks,
        min_block_ps=msd_block_min_ps,
    )
    _dump_json(analysis_dir / "msd_block_diffusion.json", msd_block_diagnostic)
    sigma = analy.sigma(msd=msd, temp_k=temp_k, eh_mode="gmx_current_only")

    summary = {
        "metadata": species_summary["metadata"],
        "basic_properties": basic.get("basic_properties", {}),
        "equilibration_ok": bool(result),
        "transport_confidence": "high" if result else "low_density_not_converged",
        "diffusion_m2_s": {
            "EC": _extract_default_diffusivity(msd, "EC"),
            "EMC": _extract_default_diffusivity(msd, "EMC"),
            "DEC": _extract_default_diffusivity(msd, "DEC"),
            "Li": _extract_default_diffusivity(msd, "Li"),
            "PF6": _extract_default_diffusivity(msd, "PF6"),
        },
        "msd_block_diffusion_diagnostic": msd_block_diagnostic,
        "conductivity": {
            "sigma_ne_upper_bound_S_m": sigma.get("sigma_ne_upper_bound_S_m"),
            "sigma_eh_total_S_m": sigma.get("sigma_eh_total_S_m"),
            "haven_ratio": sigma.get("haven_ratio"),
            "eh_confidence": ((sigma.get("eh") or {}) if isinstance(sigma.get("eh"), dict) else {}).get("confidence"),
            "eh_quality_note": ((sigma.get("eh") or {}) if isinstance(sigma.get("eh"), dict) else {}).get("quality_note"),
        },
        "coordination": {
            "EC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "ec"),
            "EMC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "emc"),
            "DEC_carbonyl_oxygen": _extract_primary_oxygen_site(rdf, "dec"),
            "PF6_coordination_fluorine": _extract_rdf_site(rdf, "pf6:coordination_fluorine"),
            "PF6_fluorine_site": _extract_rdf_site(rdf, "pf6:fluorine_site"),
        },
    }
    summary["coordination_preference"] = _coordination_preference_summary(
        summary["coordination"],
        {"EC": count_ec, "EMC": count_emc, "DEC": count_dec},
    )
    _dump_json(analysis_dir / "benchmark_summary.json", summary)
    print("[BENCHMARK] carbonate_lipf6_oplsaa completed")
    print(json.dumps(summary["diffusion_m2_s"], indent=2, ensure_ascii=False))
