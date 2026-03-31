from __future__ import annotations

from yadonpy.moldb.store import MolDB, MolRecord


def test_moldb_check_sorts_rows_by_name(capsys, tmp_path):
    db = MolDB(db_dir=tmp_path)
    db.save_record(MolRecord(key="z_key", kind="smiles", canonical="O", name="zeta", ready=True))
    db.save_record(MolRecord(key="a_key", kind="smiles", canonical="CC", name="Alpha", ready=True))
    db.save_record(MolRecord(key="b_key", kind="smiles", canonical="N", name="beta", ready=True))

    db.check()
    out = capsys.readouterr().out

    assert out.index("Alpha") < out.index("beta") < out.index("zeta")
