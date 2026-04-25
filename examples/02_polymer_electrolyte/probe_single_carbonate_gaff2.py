from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from yadonpy.core import utils
from yadonpy.core import workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2
from yadonpy.runtime import set_run_options
from yadonpy.sim import qm
from yadonpy.sim.benchmarking import summarize_rdkit_species_forcefield


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


def _env_text(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return str(default)
    text = str(raw).strip()
    return text if text else str(default)


def _normalize_geometry_source(raw: str | None) -> str:
    mode = str(raw or "qm").strip().lower()
    if mode in {"db", "moldb", "existing"}:
        return "moldb"
    if mode != "qm":
        raise ValueError(f"Unsupported GEOMETRY_SOURCE={raw!r}; expected qm/moldb.")
    return mode


def _load_benchmark_module():
    script_path = Path(__file__).with_name("benchmark_carbonate_lipf6_gaff2.py")
    spec = importlib.util.spec_from_file_location("benchmark_carbonate_lipf6_gaff2", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load benchmark helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    restart_status = _env_bool("RESTART_STATUS", False)
    set_run_options(restart=restart_status)

    bench = _load_benchmark_module()

    target = _env_text("TARGET", "DEC").strip().upper()
    if target not in {"EC", "EMC", "DEC"}:
        raise ValueError(f"Unsupported TARGET={target!r}; expected EC/EMC/DEC.")

    target_smiles = {
        "EC": bench.EC_SMILES,
        "EMC": bench.EMC_SMILES,
        "DEC": bench.DEC_SMILES,
    }[target]

    resp_profile = bench._normalize_resp_profile(os.environ.get("YADONPY_RESP_PROFILE"))
    charge_recipe = bench._charge_recipe_from_family(os.environ.get("YADONPY_CHARGE_DFT_FAMILY"))
    cache_to_repo_db = _env_bool("CACHE_TO_REPO_DB", True)
    geometry_source = _normalize_geometry_source(os.environ.get("GEOMETRY_SOURCE"))

    psi4_omp = _env_int("PSI4_OMP", 8)
    mpi = _env_int("MPI", 1)
    omp = _env_int("OMP", 1)
    memory_mb = _env_int("MEM_MB", 20000)

    work_dir_name = _env_text("WORK_DIR_NAME", f"probe_single_{target.lower()}_gaff2")
    work_root = workdir(Path(_env_text("WORK_DIR", str(Path(__file__).resolve().parent / work_dir_name))).resolve(), restart=restart_status)

    ff = GAFF2()
    if geometry_source == "moldb":
        mol = ff.mol_rdkit(
            target_smiles,
            name=target,
            prefer_db=True,
            require_ready=False,
        )
        log_name = f"{target.lower()}_{charge_recipe['family']}_{ff.name}_refit"
        qm.assign_charges(
            mol,
            charge="RESP",
            opt=False,
            work_dir=work_root,
            log_name=log_name,
            omp=psi4_omp,
            memory=memory_mb,
            charge_method=charge_recipe["charge_method"],
            charge_basis=charge_recipe["charge_basis"],
            charge_basis_gen=charge_recipe["charge_basis_gen"],
            resp_profile=resp_profile,
        )
        mol = ff.ff_assign(mol, charge=None, report=False)
        if not mol:
            raise RuntimeError(f"Cannot assign {ff.name} parameters for {target} after RESP refit.")
        if cache_to_repo_db:
            ff.store_to_db(
                mol,
                smiles_or_psmiles=target_smiles,
                name=target,
                db_dir=bench.REPO_DB_DIR,
                charge="RESP",
                basis_set=charge_recipe["charge_basis"],
                method=charge_recipe["charge_method"],
            )
            print(f"[MolDB] stored refit {target} RESP entry into repo db: {bench.REPO_DB_DIR}")
    else:
        mol = bench._build_qm_ready_gaff_species(
            ff,
            target_smiles,
            label=target,
            recipe=charge_recipe,
            resp_profile=resp_profile,
            work_root=work_root,
            psi4_omp=psi4_omp,
            mpi=mpi,
            omp=omp,
            memory_mb=memory_mb,
            repo_db_dir=bench.REPO_DB_DIR,
            cache_to_repo_db=cache_to_repo_db,
        )

    analysis_dir = work_root / "06_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "metadata": {
            "target": target,
            "resp_profile": resp_profile,
            "qm_charge_recipe": charge_recipe,
            "cache_to_repo_db": cache_to_repo_db,
            "geometry_source": geometry_source,
            "repo_db_dir": str(bench.REPO_DB_DIR),
        },
        "resp_route": bench._extract_resp_route(mol, label=target),
        "charge_sanity": bench._summarize_carbonate_charge_features(mol, label=target),
        "equivalence_spread": bench._equivalence_spread_diagnostic(mol, label=target),
        "forcefield_summary": summarize_rdkit_species_forcefield(mol, label=target, moltype_hint=target, charge_scale=1.0),
    }
    out = analysis_dir / "probe_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
