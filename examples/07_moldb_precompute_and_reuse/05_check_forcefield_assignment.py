from __future__ import annotations

"""Example 07 / Step 5: Spot-check force-field assignment on the precomputed catalog."""

import importlib.util
import json
import sys
from pathlib import Path
from dataclasses import dataclass

import yadonpy as yp
from yadonpy.core import workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.runtime import set_run_options


HERE = Path(__file__).resolve().parent
BUILD_SCRIPT = HERE / "01_build_moldb.py"
GROUP_ORDER = (
    "neutral_molecules",
    "drih_anions",
    "polyelectrolyte_monomers",
    "monatomic_ions",
)


@dataclass(frozen=True)
class DirectIonSpec:
    name: str
    smiles: str


DIRECT_ION_SPECS = (
    DirectIonSpec(name="Li", smiles="[Li+]"),
    DirectIonSpec(name="Na", smiles="[Na+]"),
)


def _load_build_module():
    spec = importlib.util.spec_from_file_location("example07_build_moldb", BUILD_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _formal_charge(mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def _catalog_report_group(spec, *, formal_charge: int) -> str:
    if bool(spec.polyelectrolyte_mode):
        return "polyelectrolyte_monomers"
    if str(spec.bonded or "").strip().upper() == "DRIH":
        return "drih_anions"
    if int(formal_charge) == 0:
        return "neutral_molecules"
    return "drih_anions"


def _empty_group_report() -> dict[str, dict[str, object]]:
    return {
        group: {
            "success_count": 0,
            "failure_count": 0,
            "success": [],
            "failures": [],
        }
        for group in GROUP_ORDER
    }


if __name__ == "__main__":
    restart_status = False
    set_run_options(restart=restart_status)

    doctor(print_report=True)
    ensure_initialized()

    ff = yp.get_ff("gaff2_mod")
    ion_ff = yp.get_ff("merz")

    build_mod = _load_build_module()
    species = build_mod._read_species_csv(build_mod.CATALOG_CSV)

    example_wd = workdir(HERE / "work_dir", restart=restart_status)
    job_wd = example_wd.child("05_check_forcefield_assignment")

    summary: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    grouped = _empty_group_report()

    for spec in species:
        try:
            mol = ff.mol(
                spec.smiles,
                charge=spec.charge,
                require_ready=True,
                prefer_db=True,
            )
            ok = bool(ff.ff_assign(mol, bonded=spec.bonded, report=False))

            if not ok:
                raise RuntimeError("ff_assign returned False")

            formal_charge = _formal_charge(mol)
            group = _catalog_report_group(spec, formal_charge=formal_charge)
            item = {
                "name": spec.name,
                "smiles": spec.smiles,
                "charge": spec.charge,
                "bonded": spec.bonded,
                "formal_charge": formal_charge,
                "atom_count": int(mol.GetNumAtoms()),
                "report_group": group,
            }
            summary.append(item)
            grouped[group]["success"].append(item)
            print(
                f"[OK] {spec.name:20s} group={group:24s} charge={spec.charge:5s} bonded={spec.bonded or '-'}"
            )
        except Exception as exc:
            group = (
                "polyelectrolyte_monomers"
                if spec.polyelectrolyte_mode
                else "drih_anions"
                if str(spec.bonded or "").strip().upper() == "DRIH"
                else "neutral_molecules"
            )
            item = {
                "name": spec.name,
                "smiles": spec.smiles,
                "charge": spec.charge,
                "bonded": spec.bonded,
                "report_group": group,
                "error": repr(exc),
            }
            failures.append(item)
            grouped[group]["failures"].append(item)
            print(
                f"[FAIL] {spec.name:20s} group={group:24s} charge={spec.charge:5s} bonded={spec.bonded or '-'} :: {exc}"
            )

    for ion_spec in DIRECT_ION_SPECS:
        try:
            mol = ion_ff.mol(ion_spec.smiles)
            ok = bool(ion_ff.ff_assign(mol, report=False))
            if not ok:
                raise RuntimeError("ff_assign returned False")

            item = {
                "name": ion_spec.name,
                "smiles": ion_spec.smiles,
                "charge": "MERZ",
                "bonded": None,
                "formal_charge": _formal_charge(mol),
                "atom_count": int(mol.GetNumAtoms()),
                "report_group": "monatomic_ions",
            }
            summary.append(item)
            grouped["monatomic_ions"]["success"].append(item)
            print(f"[OK] {ion_spec.name:20s} group=monatomic_ions          charge=MERZ  bonded=-")
        except Exception as exc:
            item = {
                "name": ion_spec.name,
                "smiles": ion_spec.smiles,
                "charge": "MERZ",
                "bonded": None,
                "report_group": "monatomic_ions",
                "error": repr(exc),
            }
            failures.append(item)
            grouped["monatomic_ions"]["failures"].append(item)
            print(f"[FAIL] {ion_spec.name:20s} group=monatomic_ions          charge=MERZ  bonded=- :: {exc}")

    for group_name in GROUP_ORDER:
        grouped[group_name]["success_count"] = len(grouped[group_name]["success"])
        grouped[group_name]["failure_count"] = len(grouped[group_name]["failures"])

        group_out = {
            "group": group_name,
            "success_count": grouped[group_name]["success_count"],
            "failure_count": grouped[group_name]["failure_count"],
            "success": grouped[group_name]["success"],
            "failures": grouped[group_name]["failures"],
        }
        (Path(job_wd) / f"{group_name}.json").write_text(
            json.dumps(group_out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    out = {
        "catalog_csv": str(build_mod.CATALOG_CSV.resolve()),
        "work_root": str(Path(job_wd).resolve()),
        "success_count": len(summary),
        "failure_count": len(failures),
        "groups": grouped,
        "success": summary,
        "failures": failures,
    }
    (Path(job_wd) / "forcefield_check_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nCatalog CSV   : {build_mod.CATALOG_CSV}")
    for group_name in GROUP_ORDER:
        print(
            f"{group_name:22s}: "
            f"{grouped[group_name]['success_count']} success / {grouped[group_name]['failure_count']} failure"
        )
    print(f"Success       : {len(summary)}")
    print(f"Failures      : {len(failures)}")
    raise SystemExit(0 if not failures else 1)
