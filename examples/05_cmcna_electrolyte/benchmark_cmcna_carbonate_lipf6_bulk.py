"""CMC-Na / carbonate / LiPF6 bulk benchmark.

This script is intentionally close to the example02/example05 style: user
settings live near the top, MoldDB-ready species are reused, and no QM/RESP is
started from the benchmark script.  It builds a swollen CMC-Na bulk box and
reports transport with explicit MSD semantics:

- Li and Na: single-ion atomic MSD.
- PF6, EC, EMC, DEC: molecular COM MSD.
- CMC: whole-chain COM MSD is the polymer self-diffusion observable; residue
  and charged-group COM MSDs are local mobility diagnostics.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from rdkit.Chem import Descriptors

from yadonpy.core import naming, poly, utils, workdir
from yadonpy.core.chem_utils import correct_total_charge, symmetrize_equivalent_charge_props
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.core.polyelectrolyte import annotate_polyelectrolyte_metadata
from yadonpy.core.polymer_audit import audit_polymer_state, compare_exported_charge_groups, write_polymer_audit
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2, GAFF2_mod, MERZ, OPLSAA
from yadonpy.ff.oplsaa_reference import audit_oplsaa_assignment
from yadonpy.runtime import set_run_options
from yadonpy.sim.analyzer import AnalyzeResult
from yadonpy.sim.preset import eq


def _env_flag(name: str, default: bool = False) -> bool:
    token = str(os.environ.get(name, "")).strip().lower()
    if not token:
        return bool(default)
    return token in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    return int(raw) if raw else int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    return float(raw) if raw else float(default)


def _env_optional_float(name: str) -> float | None:
    raw = str(os.environ.get(name, "")).strip()
    return float(raw) if raw else None


def _env_text(name: str, default: str) -> str:
    raw = str(os.environ.get(name, "")).strip()
    return raw if raw else str(default)


def _normalize_forcefield(raw: str) -> str:
    token = str(raw or "gaff2").strip().lower().replace("-", "_")
    if token in {"gaff2", "gaff"}:
        return "gaff2"
    if token in {"gaff2_mod", "mod"}:
        return "gaff2_mod"
    if token in {"opls", "oplsaa", "opls_aa"}:
        return "oplsaa"
    raise ValueError("YADONPY_FORCEFIELD must be gaff2, gaff2_mod, or oplsaa")


def _formal_charge(mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def _net_charge(mol, prop: str = "AtomicCharge") -> float:
    total = 0.0
    for atom in mol.GetAtoms():
        if atom.HasProp(prop):
            total += float(atom.GetDoubleProp(prop))
    return float(total)


def _mol_weight(mol) -> float:
    return float(Descriptors.MolWt(mol))


def _set_zero_charge_props(mol) -> None:
    for atom in mol.GetAtoms():
        for prop in ("AtomicCharge", "RESP", "RESP2", "ESP"):
            atom.SetDoubleProp(prop, 0.0)


def _zero_charge_terminator(smiles: str):
    mol = utils.mol_from_smiles(smiles)
    _set_zero_charge_props(mol)
    naming.ensure_name(mol, name="TER", prefer_var=False)
    return mol


def _load_ready_gaff_species(
    ff,
    smiles: str,
    *,
    label: str,
    repo_db_dir: Path,
    bonded: str | None = None,
    polyelectrolyte_mode: bool = False,
):
    last_exc: Exception | None = None
    for db_dir, db_label in ((repo_db_dir, "repo"), (None, "default")):
        try:
            mol = ff.mol_rdkit(
                smiles,
                name=label,
                db_dir=db_dir,
                charge="RESP",
                require_ready=True,
                prefer_db=True,
                polyelectrolyte_mode=polyelectrolyte_mode,
                polyelectrolyte_detection="auto",
            )
            mol = ff.ff_assign(mol, bonded=bonded, polyelectrolyte_mode=polyelectrolyte_mode, report=False)
            if not mol:
                raise RuntimeError(f"Cannot assign GAFF parameters for {label}.")
            print(f"[MolDB] loaded {label} from {db_label} db with RESP charges")
            return mol
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"{label} must be RESP-ready in MolDB for this benchmark.") from last_exc


def _load_ready_opls_species(
    ff: OPLSAA,
    smiles: str,
    *,
    label: str,
    repo_db_dir: Path,
    bonded: str | None = None,
    polyelectrolyte_mode: bool = False,
):
    last_exc: Exception | None = None
    for db_dir, db_label in ((repo_db_dir, "repo"), (None, "default")):
        try:
            mol = ff.mol_rdkit(
                smiles,
                name=label,
                db_dir=db_dir,
                charge="RESP",
                require_ready=True,
                prefer_db=True,
                polyelectrolyte_mode=polyelectrolyte_mode,
                polyelectrolyte_detection="auto",
            )
            mol = ff.ff_assign(
                mol,
                charge=None,
                bonded=bonded,
                polyelectrolyte_mode=polyelectrolyte_mode,
                report=False,
            )
            if not mol:
                raise RuntimeError(f"Cannot assign OPLS-AA parameters for {label}.")
            print(f"[MolDB] loaded {label} from {db_label} db with RESP charges and OPLS-AA atom types")
            return mol
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"{label} must be RESP-ready in MolDB for this OPLS-AA benchmark.") from last_exc


def _assign_merz_ion(ff: MERZ, smiles: str, *, label: str):
    mol = ff.mol(smiles)
    mol = ff.ff_assign(mol, report=False)
    if not mol:
        raise RuntimeError(f"Cannot assign MERZ parameters for {label}.")
    naming.ensure_name(mol, name=label, prefer_var=False)
    return mol


def _assign_opls_ion(ff: OPLSAA, smiles: str, *, label: str):
    mol = ff.mol(smiles, charge="opls", require_ready=False, prefer_db=False)
    mol = ff.ff_assign(mol, charge="opls", report=False)
    if not mol:
        raise RuntimeError(f"Cannot assign built-in OPLS-AA parameters for {label}.")
    naming.ensure_name(mol, name=label, prefer_var=False)
    return mol


def _solvent_counts_for_swelling(
    *,
    solvent_mols: list[object],
    mass_ratio: list[float],
    dry_mass_amu: float,
    swelling_fraction: float,
) -> list[int]:
    target_mass = max(0.0, float(dry_mass_amu) * float(swelling_fraction))
    weights = [_mol_weight(mol) for mol in solvent_mols]
    ratio_sum = float(sum(float(x) for x in mass_ratio))
    counts: list[int] = []
    for mw, ratio in zip(weights, mass_ratio):
        target_i = target_mass * float(ratio) / ratio_sum
        counts.append(max(1, int(round(target_i / max(float(mw), 1.0e-12)))))
    return counts


def _salt_pairs_for_1m(*, solvent_mols: list[object], solvent_counts: list[int], density_g_cm3: float, molarity: float) -> int:
    solvent_mass_amu = sum(_mol_weight(mol) * int(count) for mol, count in zip(solvent_mols, solvent_counts))
    # molecules = c(mol/L) * volume(L) * Avogadro; with mass in amu this reduces
    # to c * mass_amu / (density_g_cm3 * 1000).
    return max(1, int(round(float(molarity) * float(solvent_mass_amu) / (float(density_g_cm3) * 1000.0))))


def _repair_polymer_net_charge(mol, *, target_q: float, audit_dir: Path, label: str) -> dict[str, Any]:
    """Keep polymer partial-charge net aligned with its formal charge.

    Random-walk linking correctly carries linker-neighbor charges, but tiny
    per-repeat residuals can accumulate over long CMC chains.  The system
    counterion count is based on formal charge, so we make that invariant
    explicit before packing/export.
    """

    before = {
        "AtomicCharge": _net_charge(mol, "AtomicCharge"),
        "RESP": _net_charge(mol, "RESP"),
        "RESP2": _net_charge(mol, "RESP2"),
        "ESP": _net_charge(mol, "ESP"),
    }
    correction = correct_total_charge(
        mol,
        target_q=float(target_q),
        props=("AtomicCharge", "RESP", "RESP2", "ESP"),
        tol=1.0e-8,
    )
    try:
        symmetrize_equivalent_charge_props(mol)
    except Exception:
        pass
    try:
        annotate_polyelectrolyte_metadata(mol, detection="auto", resp_profile="adaptive")
    except Exception:
        pass
    after = {
        "AtomicCharge": _net_charge(mol, "AtomicCharge"),
        "RESP": _net_charge(mol, "RESP"),
        "RESP2": _net_charge(mol, "RESP2"),
        "ESP": _net_charge(mol, "ESP"),
    }
    payload = {
        "label": str(label),
        "target_q": float(target_q),
        "before": before,
        "after": after,
        "correction": correction,
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"{label}_charge_repair.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _composition_for_atom_cap(
    *,
    cmc,
    na,
    solvent_mols: list[object],
    chain_count: int,
    q_poly: int,
    swelling_fraction: float,
    solvent_mass_ratio: list[float],
    solvent_density_g_cm3: float,
    salt_molarity: float,
    atom_cap: int,
) -> dict[str, Any]:
    n_chain = max(1, int(chain_count))
    while True:
        n_na = int(abs(int(q_poly)) * n_chain)
        dry_mass = _mol_weight(cmc) * n_chain + _mol_weight(na) * n_na
        solvent_counts = _solvent_counts_for_swelling(
            solvent_mols=solvent_mols,
            mass_ratio=solvent_mass_ratio,
            dry_mass_amu=dry_mass,
            swelling_fraction=swelling_fraction,
        )
        n_salt = _salt_pairs_for_1m(
            solvent_mols=solvent_mols,
            solvent_counts=solvent_counts,
            density_g_cm3=solvent_density_g_cm3,
            molarity=salt_molarity,
        )
        counts = [n_chain, *solvent_counts, n_salt, n_salt, n_na]
        atom_count = (
            cmc.GetNumAtoms() * counts[0]
            + sum(mol.GetNumAtoms() * count for mol, count in zip(solvent_mols, solvent_counts))
            + LI_NATOMS * n_salt
            + PF6_NATOMS * n_salt
            + na.GetNumAtoms() * n_na
        )
        if atom_count <= int(atom_cap) or n_chain <= 1:
            return {
                "counts": counts,
                "estimated_atom_count": int(atom_count),
                "chain_count": int(n_chain),
                "dry_mass_amu": float(dry_mass),
                "solvent_counts": solvent_counts,
                "salt_pairs": int(n_salt),
            }
        n_chain -= 1


def _metric_row(msd: dict[str, Any], moltype: str, metric_name: str, *, role: str) -> dict[str, Any]:
    species = dict(msd.get(moltype) or {})
    metric = dict((species.get("metrics") or {}).get(metric_name) or {})
    value = metric.get("D_m2_s")
    apparent = metric.get("apparent_D_m2_s")
    return {
        "role": role,
        "moltype": moltype,
        "metric": metric_name,
        "D_m2_s": value,
        "apparent_D_m2_s": apparent,
        "alpha_mean": metric.get("alpha_mean"),
        "alpha_std": metric.get("alpha_std"),
        "confidence": metric.get("confidence"),
        "status": metric.get("status"),
        "warning": metric.get("warning"),
        "fit_t_start_ps": metric.get("fit_t_start_ps"),
        "fit_t_end_ps": metric.get("fit_t_end_ps"),
        "fit_n_points": metric.get("fit_n_points"),
        "min_fit_points": metric.get("min_fit_points"),
        "min_fit_duration_ps": metric.get("min_fit_duration_ps"),
        "n_groups": metric.get("n_groups"),
    }


def _write_transport_table(analysis_dir: Path, msd: dict[str, Any], rdf_li: dict[str, Any] | None, rdf_na: dict[str, Any] | None) -> Path:
    # Keep whole-chain polymer self-diffusion separate from local segment or
    # charged-group mobility.  This mirrors the usual polymer-diffusion
    # convention and avoids reporting monomer motion as chain transport.
    rows = [
        _metric_row(msd, "Li", "ion_atomic_msd", role="ion_atomic_diffusion"),
        _metric_row(msd, "Na", "ion_atomic_msd", role="counterion_atomic_diffusion"),
        _metric_row(msd, "PF6", "molecule_com_msd", role="anion_com_diffusion"),
        _metric_row(msd, "EC", "molecule_com_msd", role="solvent_com_diffusion"),
        _metric_row(msd, "EMC", "molecule_com_msd", role="solvent_com_diffusion"),
        _metric_row(msd, "DEC", "molecule_com_msd", role="solvent_com_diffusion"),
        _metric_row(msd, "CMC", "chain_com_msd", role="polymer_chain_com_self_diffusion"),
        _metric_row(msd, "CMC", "residue_com_msd", role="polymer_segment_mobility_diagnostic"),
        _metric_row(msd, "CMC", "charged_group_com_msd", role="polymer_carboxylate_group_mobility"),
    ]
    out_csv = analysis_dir / "cmc_bulk_transport_table.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "transport_rows": rows,
        "rdf_li_first_shell": rdf_li or {},
        "rdf_na_first_shell": rdf_na or {},
        "notes": [
            "CMC polymer self-diffusion is reported from each independent chain COM (chain_com_msd).",
            "Residue and charged-group MSD rows are local mobility diagnostics, not whole-chain self-diffusion coefficients.",
            "Report D_m2_s only when alpha_mean is close to 1 and confidence/status are acceptable; otherwise prefer apparent_D_m2_s as a mobility index.",
        ],
    }
    (analysis_dir / "cmc_bulk_transport_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_csv


# ---------------- user inputs ----------------
restart_status = _env_flag("YADONPY_RESTART", default=True)
build_only = _env_flag("YADONPY_BUILD_ONLY", default=False)
export_only = _env_flag("YADONPY_EXPORT_ONLY", default=False)
analysis_only = _env_flag("YADONPY_ANALYSIS_ONLY", default=False)
smoke_mode = _env_flag("YADONPY_SMOKE", default=False)
skip_rdf = _env_flag("YADONPY_SKIP_RDF", default=False)
skip_sigma = _env_flag("YADONPY_SKIP_SIGMA", default=True)
skip_den_dis = _env_flag("YADONPY_SKIP_DEN_DIS", default=True)

forcefield_name = _normalize_forcefield(_env_text("YADONPY_FORCEFIELD", "gaff2"))
oplsaa_profile = _env_text("YADONPY_OPLSAA_PROFILE", "strict")
random_seed = _env_int("YADONPY_RANDOM_SEED", 20260507)
chain_len = _env_int("YADONPY_CMC_DP", 12 if smoke_mode else 50)
chain_count = _env_int("YADONPY_CMC_CHAINS", 2 if smoke_mode else 20)
atom_cap = _env_int("YADONPY_ATOM_CAP", 12000 if smoke_mode else 40000)
swelling_fraction = _env_float("YADONPY_SWELLING_FRACTION", 0.12)
salt_molarity = _env_float("YADONPY_SALT_MOLARITY", 1.0)
solvent_density_g_cm3 = _env_float("YADONPY_SOLVENT_DENSITY_G_CM3", 1.15)
charge_scale_value = _env_float("YADONPY_CHARGE_SCALE", 0.7)

temp = _env_float("YADONPY_TEMP_K", 318.15)
press = _env_float("YADONPY_PRESS_BAR", 1.0)
prod_ns = _env_float("YADONPY_PROD_NS", 100.0)
# OPLS-AA for CMC/electrolyte is still under active validation.  The default is
# intentionally conservative here; users can opt into 2 fs once their short
# preflight confirms the specific parameter set is stable.
prod_dt_default = 0.001 if forcefield_name == "oplsaa" else 0.002
prod_lincs_iter_default = 4 if forcefield_name == "oplsaa" else 2
prod_lincs_order_default = 12 if forcefield_name == "oplsaa" else 8
prod_dt_ps = _env_float("YADONPY_PROD_DT_PS", prod_dt_default)
prod_constraints = _env_text("YADONPY_PROD_CONSTRAINTS", "h-bonds")
prod_lincs_iter = _env_int("YADONPY_PROD_LINCS_ITER", prod_lincs_iter_default)
prod_lincs_order = _env_int("YADONPY_PROD_LINCS_ORDER", prod_lincs_order_default)
prod_ensemble = _env_text("YADONPY_PROD_ENSEMBLE", "npt").strip().lower()
prod_bridge_ps = _env_float("YADONPY_PROD_BRIDGE_PS", 100.0)
gpu_offload_default = "conservative" if forcefield_name == "oplsaa" else "full"
gpu_offload_mode = _env_text("YADONPY_GPU_OFFLOAD_MODE", gpu_offload_default)
eq_gpu_offload_mode = _env_text("YADONPY_EQ_GPU_OFFLOAD_MODE", gpu_offload_mode)
performance_profile = _env_text("PERFORMANCE_PROFILE", "auto")
analysis_profile = _env_text("ANALYSIS_PROFILE", "auto")

eq21_final_ns = _env_float("YADONPY_EQ21_FINAL_NS", 0.8)
eq21_npt_time_scale = _env_float("YADONPY_EQ21_NPT_TIME_SCALE", 2.0)
additional_ns = _env_float("YADONPY_ADDITIONAL_NS", 1.0)
additional_rounds = _env_int("YADONPY_ADDITIONAL_MAX_ROUNDS", 4)
allow_unconverged = _env_flag("ALLOW_UNCONVERGED_PRODUCTION", default=False)

mpi = _env_int("YADONPY_MPI", 1)
omp = _env_int("YADONPY_OMP", 14)
gpu = _env_int("YADONPY_GPU", 1)
gpu_id = _env_int("YADONPY_GPU_ID", 0)

msd_begin_ps = _env_optional_float("YADONPY_MSD_BEGIN_PS")
msd_end_ps = _env_optional_float("YADONPY_MSD_END_PS")
msd_drift = _env_text("YADONPY_MSD_DRIFT", "off")

BASE_DIR = Path(__file__).resolve().parent
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"
WORK_DIR_OVERRIDE = str(os.environ.get("YADONPY_WORK_DIR", "")).strip()
work_dir = (
    Path(WORK_DIR_OVERRIDE).expanduser()
    if WORK_DIR_OVERRIDE
    else BASE_DIR / f"work_dir_cmcna_carbonate_lipf6_{forcefield_name}"
)

glucose_0_smiles = "*OC1OC(CO)C(*)C(O)C1O"
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"
feed_ratio = [12, 26, 27, 35]
feed_prob = poly.ratio_to_prob(feed_ratio)
ter_smiles = "[H][*]"

EC_SMILES = "O=C1OCCO1"
EMC_SMILES = "CCOC(=O)OC"
DEC_SMILES = "CCOC(=O)OCC"
LI_SMILES = "[Li+]"
NA_SMILES = "[Na+]"
PF6_SMILES = "F[P-](F)(F)(F)(F)F"
LI_NATOMS = 1
PF6_NATOMS = 7


def main() -> int:
    doctor(print_report=True)
    ensure_initialized()
    set_run_options(restart=restart_status)
    np.random.seed(int(random_seed))

    wd = workdir(work_dir, restart=restart_status)
    analysis_dir = Path(wd) / "06_analysis"
    audit_dir = Path(wd) / "07_polymer_audit"
    cmc_rw_dir = wd.child("CMC_rw")
    cmc_term_dir = wd.child("CMC_term")
    ac_build_dir = wd.child("00_build_cell")

    if forcefield_name == "oplsaa":
        ff = OPLSAA(profile=oplsaa_profile)
        ion_ff = OPLSAA(profile=oplsaa_profile)
        load_species = lambda smiles, label, **kw: _load_ready_opls_species(  # noqa: E731
            ff, smiles, label=label, repo_db_dir=REPO_DB_DIR, **kw
        )
        Li = _assign_opls_ion(ion_ff, LI_SMILES, label="Li")
        Na = _assign_opls_ion(ion_ff, NA_SMILES, label="Na")
        PF6 = load_species(PF6_SMILES, "PF6", bonded="DRIH")
    else:
        ff = GAFF2_mod() if forcefield_name == "gaff2_mod" else GAFF2()
        ion_ff = MERZ()
        load_species = lambda smiles, label, **kw: _load_ready_gaff_species(  # noqa: E731
            ff, smiles, label=label, repo_db_dir=REPO_DB_DIR, **kw
        )
        Li = _assign_merz_ion(ion_ff, LI_SMILES, label="Li")
        Na = _assign_merz_ion(ion_ff, NA_SMILES, label="Na")
        PF6 = load_species(PF6_SMILES, "PF6", bonded="DRIH")

    glucose_0 = load_species(glucose_0_smiles, "glucose_0")
    glucose_2 = load_species(glucose_2_smiles, "glucose_2", polyelectrolyte_mode=True)
    glucose_3 = load_species(glucose_3_smiles, "glucose_3", polyelectrolyte_mode=True)
    glucose_6 = load_species(glucose_6_smiles, "glucose_6", polyelectrolyte_mode=True)

    ter = _zero_charge_terminator(ter_smiles)
    CMC = poly.random_copolymerize_rw(
        [glucose_0, glucose_2, glucose_3, glucose_6],
        chain_len,
        ratio=feed_prob,
        tacticity="atactic",
        name="CMC",
        work_dir=cmc_rw_dir,
    )
    write_polymer_audit(audit_polymer_state(CMC, label="cmc_random_walk", radius=2), audit_dir / "cmc_random_walk.json")
    CMC = poly.terminate_rw(CMC, ter, name="CMC", work_dir=cmc_term_dir)
    CMC = ff.ff_assign(CMC, charge=None, polyelectrolyte_mode=True, report=False)
    if not CMC:
        raise RuntimeError("Cannot assign force field parameters for CMC.")
    naming.ensure_name(CMC, name="CMC", prefer_var=False)
    q_poly = _formal_charge(CMC)
    charge_repair = _repair_polymer_net_charge(CMC, target_q=float(q_poly), audit_dir=audit_dir, label="cmc_final_assigned")
    write_polymer_audit(audit_polymer_state(CMC, label="cmc_final_assigned", radius=2), audit_dir / "cmc_final_assigned.json")
    if forcefield_name == "oplsaa":
        (audit_dir / "cmc_oplsaa_assignment_audit.json").write_text(
            json.dumps(audit_oplsaa_assignment(CMC, strict=True), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    EC = load_species(EC_SMILES, "EC")
    EMC = load_species(EMC_SMILES, "EMC")
    DEC = load_species(DEC_SMILES, "DEC")
    solvent_mols = [EC, EMC, DEC]

    composition = _composition_for_atom_cap(
        cmc=CMC,
        na=Na,
        solvent_mols=solvent_mols,
        chain_count=chain_count,
        q_poly=q_poly,
        swelling_fraction=swelling_fraction,
        solvent_mass_ratio=[3.0, 2.0, 5.0],
        solvent_density_g_cm3=solvent_density_g_cm3,
        salt_molarity=salt_molarity,
        atom_cap=atom_cap,
    )
    counts = [int(x) for x in composition["counts"]]
    species = [CMC, EC, EMC, DEC, Li, PF6, Na]
    charge_scale = [float(charge_scale_value)] * len(species)
    formulation = {
        "forcefield": forcefield_name,
        "oplsaa_profile": oplsaa_profile if forcefield_name == "oplsaa" else None,
        "random_seed": int(random_seed),
        "temperature_K": float(temp),
        "pressure_bar": float(press),
        "cmc_dp": int(chain_len),
        "cmc_chains_requested": int(chain_count),
        "cmc_chains_used": int(composition["chain_count"]),
        "cmc_formal_charge_per_chain": int(q_poly),
        "cmc_partial_charge_per_chain": float(charge_repair["after"].get("AtomicCharge", 0.0)),
        "cmc_charge_repair": charge_repair,
        "swelling_fraction": float(swelling_fraction),
        "salt_molarity_nominal": float(salt_molarity),
        "solvent_mass_ratio_EC_EMC_DEC": [3.0, 2.0, 5.0],
        "counts_CMC_EC_EMC_DEC_Li_PF6_Na": counts,
        "charge_scale": charge_scale,
        "production_dt_ps": float(prod_dt_ps),
        "production_constraints": str(prod_constraints),
        "production_lincs_iter": int(prod_lincs_iter),
        "production_lincs_order": int(prod_lincs_order),
        "production_gpu_offload_mode": str(gpu_offload_mode),
        "eq_gpu_offload_mode": str(eq_gpu_offload_mode),
        "production_stability_note": (
            "OPLS-AA defaults to 1 fs, LINCS 4/12, and conservative GPU offload for this CMC/electrolyte benchmark "
            "because remote diagnostics showed balanced/full GPU offload can trigger CUDA illegal-address failures "
            "for current refine-profile CMC assignments."
            if forcefield_name == "oplsaa"
            else "GAFF-family default production uses h-bonds with 2 fs."
        ),
        "estimated_atom_count": int(composition["estimated_atom_count"]),
        "dry_mass_amu": float(composition["dry_mass_amu"]),
        "solvent_counts": composition["solvent_counts"],
        "salt_pairs": int(composition["salt_pairs"]),
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "formulation.json").write_text(json.dumps(formulation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[FORMULATION] " + json.dumps(formulation, ensure_ascii=False))
    msd_mols = ["CMC"] + [mol for mol, count in zip(species[1:], counts[1:]) if int(count) > 0]

    if analysis_only:
        analy = AnalyzeResult.from_work_dir(wd)
        _ = analy.get_all_prop(temp=temp, press=press, save=True, analysis_profile=analysis_profile)
        rdf_li = None if skip_rdf else analy.rdf(
            center_mol=Li,
            site_filter=["carboxylate_oxygen", "hydroxyl_oxygen", "carbonyl_oxygen", "ether_oxygen", "coordination_fluorine", "fluorine_site"],
            analysis_profile=analysis_profile,
            r_max_nm=1.5,
            resume=True,
        )
        rdf_na = None if skip_rdf else analy.rdf(
            center_mol=Na,
            site_filter=["carboxylate_oxygen", "hydroxyl_oxygen", "carbonyl_oxygen", "ether_oxygen", "coordination_fluorine", "fluorine_site"],
            analysis_profile=analysis_profile,
            r_max_nm=1.5,
            resume=True,
        )
        msd = analy.msd(
            mols=msd_mols,
            geometry="3d",
            unwrap="on",
            drift=msd_drift,
            begin_ps=msd_begin_ps,
            end_ps=msd_end_ps,
            analysis_profile=analysis_profile,
            resume=True,
        )
        table = _write_transport_table(analysis_dir, msd, rdf_li, rdf_na)
        print(f"[ANALYSIS-ONLY] transport table: {table}")
        return 0

    active = [(mol, count, scale) for mol, count, scale in zip(species, counts, charge_scale) if int(count) > 0]
    ac = poly.amorphous_cell(
        [mol for mol, _count, _scale in active],
        [int(count) for _mol, count, _scale in active],
        charge_scale=[float(scale) for _mol, _count, scale in active],
        polyelectrolyte_mode=True,
        density=0.05,
        neutralize=False,
        work_dir=ac_build_dir,
    )
    if build_only:
        print(f"[BUILD-ONLY] built initial cell at {ac_build_dir}")
        return 0

    eqmd = eq.EQ21step(ac, work_dir=wd)
    if export_only:
        exported = eqmd.ensure_system_exported()
        write_polymer_audit(
            compare_exported_charge_groups(system_dir=exported.system_top.parent, moltype="CMC", mol=CMC),
            audit_dir / "cmc_export_charge_groups.json",
        )
        print(f"[EXPORT-ONLY] exported {exported.system_top.parent}")
        return 0

    ac = eqmd.exec(
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        time=eq21_final_ns,
        eq21_npt_time_scale=eq21_npt_time_scale,
        gpu_offload_mode=eq_gpu_offload_mode,
    )
    exported = eqmd.ensure_system_exported()
    write_polymer_audit(
        compare_exported_charge_groups(system_dir=exported.system_top.parent, moltype="CMC", mol=CMC),
        audit_dir / "cmc_export_charge_groups.json",
    )

    analy = eqmd.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    eq_ok = analy.check_eq()
    for round_idx in range(int(additional_rounds)):
        if eq_ok:
            break
        add = eq.Additional(ac, work_dir=wd)
        ac = add.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=additional_ns)
        analy = add.analyze()
        _ = analy.get_all_prop(temp=temp, press=press, save=True)
        eq_ok = analy.check_eq()
        print(f"[ADDITIONAL] round={round_idx + 1} equilibrium_ok={eq_ok}")
    if not eq_ok and not allow_unconverged:
        raise RuntimeError("Equilibration did not converge; set ALLOW_UNCONVERGED_PRODUCTION=1 for diagnostic production.")

    prod_cls = eq.NVT if prod_ensemble == "nvt" else eq.NPT
    prod = prod_cls(ac, work_dir=wd)
    prod_kwargs: dict[str, Any] = {
        "temp": temp,
        "mpi": mpi,
        "omp": omp,
        "gpu": gpu,
        "gpu_id": gpu_id,
        "time": prod_ns,
        "dt_ps": prod_dt_ps,
        "constraints": prod_constraints,
        "lincs_iter": prod_lincs_iter,
        "lincs_order": prod_lincs_order,
        "bridge_ps": prod_bridge_ps,
        "gpu_offload_mode": gpu_offload_mode,
        "performance_profile": performance_profile,
        "analysis_profile": analysis_profile,
    }
    if prod_ensemble != "nvt":
        prod_kwargs["press"] = press
    ac = prod.exec(**prod_kwargs)

    analy = prod.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True, analysis_profile=analysis_profile)
    rdf_li = None if skip_rdf else analy.rdf(
        center_mol=Li,
        site_filter=["carboxylate_oxygen", "hydroxyl_oxygen", "carbonyl_oxygen", "ether_oxygen", "coordination_fluorine", "fluorine_site"],
        analysis_profile=analysis_profile,
        r_max_nm=1.5,
        resume=True,
    )
    rdf_na = None if skip_rdf else analy.rdf(
        center_mol=Na,
        site_filter=["carboxylate_oxygen", "hydroxyl_oxygen", "carbonyl_oxygen", "ether_oxygen", "coordination_fluorine", "fluorine_site"],
        analysis_profile=analysis_profile,
        r_max_nm=1.5,
        resume=True,
    )
    msd = analy.msd(
        mols=msd_mols,
        geometry="3d",
        unwrap="on",
        drift=msd_drift,
        begin_ps=msd_begin_ps,
        end_ps=msd_end_ps,
        analysis_profile=analysis_profile,
        resume=True,
    )
    if not skip_sigma:
        _ = analy.sigma(temp_k=temp, msd=msd, drift=msd_drift)
    if not skip_den_dis:
        _ = analy.den_dis()
    table = _write_transport_table(analysis_dir, msd, rdf_li, rdf_na)
    print(f"[TRANSPORT] table: {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
