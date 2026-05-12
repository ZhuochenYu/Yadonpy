from __future__ import annotations

import py_compile
from pathlib import Path

from yadonpy.ff import OPLSAA
from yadonpy.ff.oplsaa_reference import audit_oplsaa_assignment


ROOT = Path(__file__).resolve().parents[1]
REPO_DB_DIR = ROOT / "moldb"
SCRIPT = ROOT / "examples" / "09_oplsaa_assignment" / "03_oplsaa_polymer_moldb_validation.py"


def test_oplsaa_polymer_validation_example_is_import_safe_and_moldb_first():
    py_compile.compile(str(SCRIPT), doraise=True)
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":' in text
    assert "MolDB(REPO_DB_DIR)" in text
    assert "require_ready=True" in text
    assert "prefer_db=True" in text
    assert 'OPLSAA(profile="strict")' in text
    assert 'OPLSAA(profile="refine")' in text
    assert "audit_oplsaa_assignment" in text
    assert "poly.random_copolymerize_rw(" in text


def test_oplsaa_strict_peo_from_repo_moldb_is_source_clean():
    ff = OPLSAA(profile="strict")
    peo = ff.mol_rdkit(
        "*CCO*",
        db_dir=REPO_DB_DIR,
        charge="RESP",
        require_ready=True,
        prefer_db=True,
    )

    assigned = ff.ff_assign(peo, charge=None, report=False)

    assert assigned is not False
    audit = audit_oplsaa_assignment(assigned, strict=True)
    assert audit["assignment_complete"] is True
    assert audit["strict_source_clean"] is True
    assert audit["local_refines"] == []


def test_oplsaa_strict_pf6_from_repo_moldb_uses_external_drih_patch_as_complete_bonded_source():
    ff = OPLSAA(profile="strict")
    pf6 = ff.mol_rdkit(
        "F[P-](F)(F)(F)(F)F",
        db_dir=REPO_DB_DIR,
        charge="RESP",
        require_ready=True,
        prefer_db=True,
    )

    assigned = ff.ff_assign(pf6, charge=None, report=False)

    assert assigned is not False
    audit = audit_oplsaa_assignment(assigned, strict=True)
    assert audit["assignment_complete"] is True
    assert audit["strict_source_clean"] is True
    assert audit["missing_bonded"] == []
    assert audit["external_bonded"]["method"] == "DRIH"
    assert audit["external_bonded"]["covered_bond_count"] == 6
    assert audit["pf6"]["has_bonded_itp"] is True


def test_oplsaa_cmc_polyelectrolyte_requires_explicit_refine_until_source_terms_are_promoted():
    smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"
    strict_ff = OPLSAA(profile="strict")
    cmc_strict = strict_ff.mol_rdkit(
        smiles,
        db_dir=REPO_DB_DIR,
        charge="RESP",
        require_ready=True,
        prefer_db=True,
        polyelectrolyte_mode=True,
    )

    assert strict_ff.ff_assign(cmc_strict, charge=None, report=False, polyelectrolyte_mode=True) is False

    refine_ff = OPLSAA(profile="refine")
    cmc_refine = refine_ff.mol_rdkit(
        smiles,
        db_dir=REPO_DB_DIR,
        charge="RESP",
        require_ready=True,
        prefer_db=True,
        polyelectrolyte_mode=True,
    )
    assigned = refine_ff.ff_assign(cmc_refine, charge=None, report=False, polyelectrolyte_mode=True)

    assert assigned is not False
    audit = audit_oplsaa_assignment(assigned, strict=True)
    assert audit["assignment_complete"] is True
    assert audit["strict_source_clean"] is False
    assert audit["local_refines"]
