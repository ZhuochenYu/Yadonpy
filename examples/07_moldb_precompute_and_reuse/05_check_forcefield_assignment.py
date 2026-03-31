from __future__ import annotations

"""Example 07 / Step 5: Spot-check force-field assignment on the precomputed catalog."""

import importlib.util
import json
import sys
from pathlib import Path

import yadonpy as yp
from yadonpy.core import workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.runtime import set_run_options


HERE = Path(__file__).resolve().parent
BUILD_SCRIPT = HERE / "01_build_moldb.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("example07_build_moldb", BUILD_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


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

    for spec in species:
        try:
            charge_route = build_mod._charge_route(spec)
            if charge_route == "ion_charge":
                mol = ion_ff.mol(spec.smiles)
                ok = bool(ion_ff.ff_assign(mol, report=False))
            else:
                mol = ff.mol(
                    spec.smiles,
                    charge=spec.charge,
                    require_ready=True,
                    prefer_db=True,
                )
                ok = bool(ff.ff_assign(mol, bonded=spec.bonded, report=False))

            if not ok:
                raise RuntimeError("ff_assign returned False")

            summary.append(
                {
                    "name": spec.name,
                    "smiles": spec.smiles,
                    "charge": spec.charge,
                    "charge_route": charge_route,
                    "bonded": spec.bonded,
                    "atom_count": int(mol.GetNumAtoms()),
                }
            )
            print(
                f"[OK] {spec.name:20s} charge={spec.charge:5s} "
                f"route={charge_route:12s} bonded={spec.bonded or '-'}"
            )
        except Exception as exc:
            failures.append(
                {
                    "name": spec.name,
                    "smiles": spec.smiles,
                    "charge": spec.charge,
                    "bonded": spec.bonded,
                    "error": repr(exc),
                }
            )
            print(
                f"[FAIL] {spec.name:20s} charge={spec.charge:5s} "
                f"bonded={spec.bonded or '-'} :: {exc}"
            )

    out = {
        "catalog_csv": str(build_mod.CATALOG_CSV.resolve()),
        "work_root": str(Path(job_wd).resolve()),
        "success_count": len(summary),
        "failure_count": len(failures),
        "success": summary,
        "failures": failures,
    }
    (Path(job_wd) / "forcefield_check_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nCatalog CSV   : {build_mod.CATALOG_CSV}")
    print(f"Success       : {len(summary)}")
    print(f"Failures      : {len(failures)}")
    raise SystemExit(0 if not failures else 1)
