from __future__ import annotations

import numpy as np

from yadonpy.core import poly, utils
import yadonpy.sim.qm as qm


def test_save_energy_json_serializes_numpy_array(tmp_path):
    path = tmp_path / "energy.json"
    energy = np.array([1.25, 2.5, 3.75], dtype=float)

    qm._save_energy_json(path, energy, log_name="conf_search")
    restored = qm._load_energy_json(path)

    assert path.exists()
    assert isinstance(restored, np.ndarray)
    assert np.allclose(restored, energy)


def test_rw_retry_budget_does_not_reduce_flexible_chain_defaults():
    mols = [utils.mol_from_smiles("*CCO*"), utils.mol_from_smiles("*COC*")]

    budget = poly._resolve_rw_retry_budget(
        mols,
        retry=60,
        rollback=3,
        rollback_shaking=False,
        retry_step=80,
        retry_opt_step=4,
    )

    assert budget["retry"] == 60
    assert budget["rollback"] == 3
    assert budget["retry_step"] == 80
    assert budget["retry_opt_step"] == 4
    assert budget["changed"] == {}


def test_rw_retry_budget_scales_up_for_long_flexible_chain():
    mols = [utils.mol_from_smiles("*CCO*"), utils.mol_from_smiles("*COC*")]

    budget = poly._resolve_rw_retry_budget(
        mols,
        retry=60,
        rollback=3,
        rollback_shaking=False,
        retry_step=80,
        retry_opt_step=4,
        total_steps=143,
    )

    assert budget["retry"] == 160
    assert budget["rollback"] == 4
    assert budget["retry_step"] == 240
    assert budget["retry_opt_step"] == 8
    assert budget["changed"] == {
        "retry": (60, 160),
        "rollback": (3, 4),
        "retry_step": (80, 240),
        "retry_opt_step": (4, 8),
    }
