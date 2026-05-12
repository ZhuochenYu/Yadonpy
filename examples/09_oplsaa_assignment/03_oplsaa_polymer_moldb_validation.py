from __future__ import annotations

"""Example 09 / Step 3: MolDB-first OPLS-AA polymer validation.

This script is intentionally lightweight: it does not run QM or MD.  It checks
that polymer and electrolyte building blocks can be loaded from the repo MolDB,
assigned with OPLS-AA, audited for source-backed parameters, and exported for
GROMACS inspection.
"""

import json
from pathlib import Path

from yadonpy.core import naming, poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import OPLSAA
from yadonpy.ff.oplsaa_reference import audit_oplsaa_assignment
from yadonpy.io.gmx import write_gmx
from yadonpy.moldb import MolDB
from yadonpy.runtime import set_run_options


# ---------------- user inputs ----------------
restart_status = False
neutral_polymer_dp = 4
polyelectrolyte_dp = 4

PEO_smiles = "*CCO*"
CMC_neutral_smiles = "*OC1OC(CO)C(*)C(O)C1O"
CMC_carboxylate_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"
EC_smiles = "O=C1OCCO1"
EMC_smiles = "CCOC(=O)OC"
PF6_smiles = "F[P-](F)(F)(F)(F)F"
Na_smiles = "[Na+]"
ter_smiles = "[H][*]"

BASE_DIR = Path(__file__).resolve().parent
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"
work_dir = BASE_DIR / "work_dir"

set_run_options(restart=restart_status)


def _load_ready_resp(ff: OPLSAA, smiles: str, label: str, *, polyelectrolyte_mode: bool = False):
    """Load a RESP-ready species from the repo MolDB and attach a stable name."""

    mol = ff.mol_rdkit(
        smiles,
        db_dir=REPO_DB_DIR,
        charge="RESP",
        require_ready=True,
        prefer_db=True,
        polyelectrolyte_mode=polyelectrolyte_mode,
        polyelectrolyte_detection="auto",
    )
    naming.ensure_name(mol, name=label)
    return mol


def _assign_and_audit(
    ff: OPLSAA,
    mol,
    label: str,
    *,
    charge: str | None = None,
    polyelectrolyte_mode: bool = False,
    bonded_work_dir: Path | None = None,
):
    """Run OPLS-AA assignment and return a compact source/provenance audit."""

    assigned = ff.ff_assign(
        mol,
        charge=charge,
        report=False,
        polyelectrolyte_mode=polyelectrolyte_mode,
        bonded_work_dir=bonded_work_dir,
    )
    if not assigned:
        raise RuntimeError(f"OPLS-AA assignment returned False for {label}")
    naming.ensure_name(assigned, name=label)
    audit = audit_oplsaa_assignment(assigned, strict=True)
    return assigned, audit


def _summary_item(label: str, smiles: str, profile: str, audit: dict[str, object], *, note: str = "") -> dict[str, object]:
    """Normalize audit fields used by the JSON summary and terminal report."""

    return {
        "label": label,
        "smiles": smiles,
        "profile": profile,
        "note": note,
        "assignment_complete": bool(audit.get("assignment_complete")),
        "strict_source_clean": bool(audit.get("strict_source_clean")),
        "atom_count": int(audit.get("atom_count", 0)),
        "net_charge": float(audit.get("net_charge", 0.0)),
        "local_refine_count": len(audit.get("local_refines") or []),
        "missing_nonbonded_count": len(audit.get("missing_nonbonded") or []),
        "missing_bonded_count": len(audit.get("missing_bonded") or []),
        "external_bonded": audit.get("external_bonded"),
        "pf6": audit.get("pf6"),
    }


def _write_case_gmx(mol, root: Path, label: str) -> dict[str, str]:
    """Export one assigned molecule and return stable output paths."""

    out_dir = root / f"{label}_gmx"
    gro, itp, top = write_gmx(mol=mol, out_dir=out_dir, mol_name=label)
    return {"gro": str(gro), "itp": str(itp), "top": str(top)}


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    db = MolDB(REPO_DB_DIR)
    if not db.objects_dir.is_dir():
        raise RuntimeError(f"Repo MolDB is missing: {db.objects_dir}")

    example_wd = workdir(work_dir, restart=restart_status)
    job_wd = Path(example_wd.child("03_polymer_moldb_validation"))

    strict_ff = OPLSAA(profile="strict")
    refine_ff = OPLSAA(profile="refine")
    summary: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    exports: dict[str, dict[str, str]] = {}

    # Neutral polymer: this should be fully source-clean in strict mode.
    peo_monomer = _load_ready_resp(strict_ff, PEO_smiles, "PEO_monomer")
    peo_monomer, peo_monomer_audit = _assign_and_audit(strict_ff, peo_monomer, "PEO_monomer")
    summary.append(_summary_item("PEO_monomer", PEO_smiles, "strict", peo_monomer_audit))

    terminator = utils.mol_from_smiles(ter_smiles)
    peo_oligomer = poly.random_copolymerize_rw(
        [peo_monomer],
        neutral_polymer_dp,
        ratio=[1.0],
        tacticity="atactic",
        name="PEO_oligomer",
        work_dir=job_wd / "01_peo_rw",
    )
    peo_oligomer = poly.terminate_rw(
        peo_oligomer,
        terminator,
        name="PEO_oligomer",
        work_dir=job_wd / "02_peo_term",
    )
    peo_oligomer, peo_oligomer_audit = _assign_and_audit(strict_ff, peo_oligomer, "PEO_oligomer")
    summary.append(_summary_item("PEO_oligomer", PEO_smiles, "strict", peo_oligomer_audit))
    exports["PEO_oligomer"] = _write_case_gmx(peo_oligomer, job_wd, "PEO_oligomer")

    # Polyelectrolyte probe: strict should fail fast until all CMC terms are
    # promoted from refine-only analogs into source-backed OPLS-AA data.
    try:
        strict_cmc = _load_ready_resp(
            strict_ff,
            CMC_carboxylate_smiles,
            "CMC_carboxylate_strict_probe",
            polyelectrolyte_mode=True,
        )
        _assign_and_audit(
            strict_ff,
            strict_cmc,
            "CMC_carboxylate_strict_probe",
            polyelectrolyte_mode=True,
        )
    except Exception as exc:
        failures.append(
            {
                "label": "CMC_carboxylate_strict_probe",
                "profile": "strict",
                "expected": True,
                "reason": "CMC strict OPLS-AA still has source-backed bonded gaps.",
                "error": repr(exc),
            }
        )

    cmc_neutral = _load_ready_resp(
        refine_ff,
        CMC_neutral_smiles,
        "CMC_neutral_unit",
        polyelectrolyte_mode=True,
    )
    cmc_carboxylate = _load_ready_resp(
        refine_ff,
        CMC_carboxylate_smiles,
        "CMC_carboxylate_unit",
        polyelectrolyte_mode=True,
    )
    cmc_neutral, _ = _assign_and_audit(
        refine_ff,
        cmc_neutral,
        "CMC_neutral_unit",
        polyelectrolyte_mode=True,
    )
    cmc_carboxylate, _ = _assign_and_audit(
        refine_ff,
        cmc_carboxylate,
        "CMC_carboxylate_unit",
        polyelectrolyte_mode=True,
    )
    cmc_oligomer = poly.random_copolymerize_rw(
        [cmc_neutral, cmc_carboxylate],
        polyelectrolyte_dp,
        ratio=[1.0, 1.0],
        tacticity="atactic",
        name="CMC_Na_oligomer",
        work_dir=job_wd / "03_cmc_rw",
    )
    cmc_oligomer = poly.terminate_rw(
        cmc_oligomer,
        terminator,
        name="CMC_Na_oligomer",
        work_dir=job_wd / "04_cmc_term",
    )
    cmc_oligomer, cmc_oligomer_audit = _assign_and_audit(
        refine_ff,
        cmc_oligomer,
        "CMC_Na_oligomer",
        polyelectrolyte_mode=True,
    )
    summary.append(
        _summary_item(
            "CMC_Na_oligomer",
            f"{CMC_neutral_smiles} + {CMC_carboxylate_smiles}",
            "refine",
            cmc_oligomer_audit,
            note="assignment-complete but not strict-source-clean until CMC local refines are source-backed",
        )
    )
    exports["CMC_Na_oligomer"] = _write_case_gmx(cmc_oligomer, job_wd, "CMC_Na_oligomer")

    # Electrolyte support species used by polymer-electrolyte workflows.
    pf6 = _load_ready_resp(strict_ff, PF6_smiles, "PF6")
    pf6, pf6_audit = _assign_and_audit(strict_ff, pf6, "PF6", bonded_work_dir=job_wd / "05_pf6_drih")
    summary.append(_summary_item("PF6", PF6_smiles, "strict", pf6_audit, note="uses precomputed MolDB DRIH bonded patch"))
    exports["PF6"] = _write_case_gmx(pf6, job_wd, "PF6")

    na = strict_ff.mol(Na_smiles, charge="opls", require_ready=False, prefer_db=False)
    na, na_audit = _assign_and_audit(strict_ff, na, "Na", charge="opls")
    summary.append(_summary_item("Na", Na_smiles, "strict", na_audit, note="monatomic ion uses built-in OPLS-AA ion data"))
    exports["Na"] = _write_case_gmx(na, job_wd, "Na")

    for label, smiles in (("EC_strict_probe", EC_smiles),):
        try:
            mol = _load_ready_resp(strict_ff, smiles, label)
            _assign_and_audit(strict_ff, mol, label)
        except Exception as exc:
            failures.append(
                {
                    "label": label,
                    "profile": "strict",
                    "expected": True,
                    "reason": "carbonate strict OPLS-AA still has source-backed bonded gaps.",
                    "error": repr(exc),
                }
            )

    emc = _load_ready_resp(refine_ff, EMC_smiles, "EMC_refine_probe")
    emc, emc_audit = _assign_and_audit(refine_ff, emc, "EMC_refine_probe")
    summary.append(
        _summary_item(
            "EMC_refine_probe",
            EMC_smiles,
            "refine",
            emc_audit,
            note="assignment-complete but not strict-source-clean until carbonate local refines are source-backed",
        )
    )
    exports["EMC_refine_probe"] = _write_case_gmx(emc, job_wd, "EMC_refine_probe")

    strict_clean_required = [item for item in summary if item["profile"] == "strict"]
    strict_clean_ok = all(bool(item["strict_source_clean"]) for item in strict_clean_required)
    refine_only = [item for item in summary if item["profile"] == "refine" and not bool(item["strict_source_clean"])]
    out = {
        "repo_moldb": str(REPO_DB_DIR.resolve()),
        "work_root": str(job_wd.resolve()),
        "strict_clean_required_count": len(strict_clean_required),
        "strict_clean_ok": strict_clean_ok,
        "refine_only_count": len(refine_only),
        "expected_strict_gap_count": len([item for item in failures if item.get("expected")]),
        "cases": summary,
        "strict_gaps": failures,
        "exports": exports,
    }
    (job_wd / "oplsaa_polymer_validation_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nOPLS-AA polymer MolDB validation")
    print(f"Work root: {job_wd.resolve()}")
    for item in summary:
        print(
            f"[{item['profile']}] {item['label']:22s} "
            f"complete={item['assignment_complete']} "
            f"strict_source_clean={item['strict_source_clean']} "
            f"local_refines={item['local_refine_count']} "
            f"net={item['net_charge']:.6f}"
        )
    for item in failures:
        marker = "expected strict gap" if item.get("expected") else "unexpected failure"
        print(f"[{marker}] {item['label']}: {item['reason']}")
    print(f"Summary: {job_wd / 'oplsaa_polymer_validation_summary.json'}")

    unexpected_failures = [item for item in failures if not item.get("expected")]
    raise SystemExit(0 if strict_clean_ok and not unexpected_failures else 1)
