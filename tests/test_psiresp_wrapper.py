from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rdkit import Chem

import yadonpy.sim.psiresp_wrapper as wrapper


class _FakeConstraintOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.charge_sum = []
        self.equiv = []

    def add_charge_sum_constraint_for_molecule(self, molecule, charge=0.0, indices=None):
        self.charge_sum.append((molecule, float(charge), list(indices or [])))

    def add_charge_equivalence_constraint_for_molecule(self, molecule, indices=None):
        self.equiv.append((molecule, list(indices or [])))


class _FakeOrientation:
    def __init__(self):
        self.grid = None
        self.esp = None
        self.qc_wavefunction = None

    def compute_grid(self, grid_options=None):
        self.grid = np.array([[0.0, 0.0, 0.0]])


class _FakePmol:
    def __init__(self, natoms: int):
        self.natoms = natoms
        self.stage_1_unrestrained_charges = None
        self.stage_2_restrained_charges = None


class _FakeMoleculeFactory:
    @staticmethod
    def from_rdkit(mol, **kwargs):
        return _FakePmol(mol.GetNumAtoms())


class _FakeJob:
    def __init__(self, *, molecules, charge_constraints, working_directory):
        self.molecules = molecules
        self.charge_constraints = charge_constraints
        self.working_directory = Path(working_directory)
        self.grid_options = SimpleNamespace(
            use_radii=None,
            vdw_scale_factors=None,
            vdw_point_density=None,
        )
        self._orientations = [_FakeOrientation()]
        self.generated = False
        self.computed = False

    def generate_orientations(self):
        self.generated = True

    def iter_orientations(self):
        return iter(self._orientations)

    def compute_charges(self, update_molecules=True):
        self.computed = True
        pmol = self.molecules[0]
        pmol.stage_1_unrestrained_charges = np.array([-0.2, 0.2])
        pmol.stage_2_restrained_charges = np.array([-0.3, 0.3])

    @property
    def charges(self):
        return np.array([-0.3, 0.3])


class _FakeResp2Job(_FakeJob):
    def __init__(self, *, molecules, charge_constraints, working_directory):
        super().__init__(molecules=molecules, charge_constraints=charge_constraints, working_directory=working_directory)
        self._label = str(working_directory)

    def compute_charges(self, update_molecules=True):
        self.computed = True
        pmol = self.molecules[0]
        if "solvated_water" in self._label:
            pmol.stage_1_unrestrained_charges = np.array([-0.25, 0.25])
            pmol.stage_2_restrained_charges = np.array([-0.40, 0.40])
        else:
            pmol.stage_1_unrestrained_charges = np.array([-0.20, 0.20])
            pmol.stage_2_restrained_charges = np.array([-0.30, 0.30])


def test_run_psiresp_fit_uses_precomputed_psi4_path(monkeypatch, tmp_path):
    fake_psiresp = SimpleNamespace(
        ChargeConstraintOptions=_FakeConstraintOptions,
        Molecule=_FakeMoleculeFactory,
        TwoStageRESP=_FakeJob,
        ESP=_FakeJob,
    )
    monkeypatch.setattr(wrapper, "psiresp", fake_psiresp)

    calls = []

    def fake_populate_orientation(
        orientation,
        *,
        grid_options,
        method,
        basis,
        run_dir=None,
        ncores=None,
        memory_mib=None,
        pcm_solvent=None,
    ):
        calls.append(
            {
                "method": method,
                "basis": basis,
                "run_dir": run_dir,
                "ncores": ncores,
                "memory_mib": memory_mib,
                "pcm_solvent": pcm_solvent,
                "grid_options": grid_options,
            }
        )
        orientation.grid = np.array([[0.0, 0.0, 0.0]])
        orientation.qc_wavefunction = object()
        orientation.esp = np.array([0.1])

    monkeypatch.setattr(wrapper, "_populate_orientation_with_precomputed_esp", fake_populate_orientation)

    mol = Chem.MolFromSmiles("CC")
    result = wrapper.run_psiresp_fit(
        mol,
        fit_kind="RESP",
        method="wb97m-d3bj",
        basis="def2-TZVPD",
        work_dir=tmp_path,
        name="ethane",
        ncores=36,
        memory_mib=20000,
    )

    assert np.allclose(result["resp"], [-0.3, 0.3])
    assert np.allclose(result["esp"], [-0.2, 0.2])
    assert len(calls) == 1
    assert calls[0]["method"] == "wb97m-d3bj"
    assert calls[0]["basis"] == "def2-TZVPD"
    assert calls[0]["run_dir"] == tmp_path / "psiresp"
    assert calls[0]["ncores"] == 36
    assert calls[0]["memory_mib"] == 20000
    assert calls[0]["pcm_solvent"] is None
    assert calls[0]["grid_options"].use_radii == "msk"
    assert calls[0]["grid_options"].vdw_scale_factors == [1.4, 1.6, 1.8, 2.0]
    assert calls[0]["grid_options"].vdw_point_density == 20.0


def test_ensure_psiresp_numpy_compat_restores_in1d(monkeypatch):
    monkeypatch.delattr(wrapper.np, "in1d", raising=False)

    wrapper._ensure_psiresp_numpy_compat()

    result = wrapper.np.in1d([0, 1, 2], [1, 3])  # type: ignore[attr-defined]
    assert np.array_equal(result, np.array([False, True, False]))


def test_run_psiresp_fit_resp2_mixes_solvated_and_gas_resp(monkeypatch, tmp_path):
    fake_psiresp = SimpleNamespace(
        ChargeConstraintOptions=_FakeConstraintOptions,
        Molecule=_FakeMoleculeFactory,
        TwoStageRESP=_FakeResp2Job,
        ESP=_FakeResp2Job,
    )
    monkeypatch.setattr(wrapper, "psiresp", fake_psiresp)

    calls = []

    def fake_populate_orientation(
        orientation,
        *,
        grid_options,
        method,
        basis,
        run_dir=None,
        ncores=None,
        memory_mib=None,
        pcm_solvent=None,
    ):
        calls.append(
            {
                "method": method,
                "basis": basis,
                "run_dir": Path(run_dir) if run_dir is not None else None,
                "pcm_solvent": pcm_solvent,
            }
        )
        orientation.grid = np.array([[0.0, 0.0, 0.0]])
        orientation.qc_wavefunction = object()
        orientation.esp = np.array([0.1])

    monkeypatch.setattr(wrapper, "_populate_orientation_with_precomputed_esp", fake_populate_orientation)

    mol = Chem.MolFromSmiles("CC")
    result = wrapper.run_psiresp_fit(
        mol,
        fit_kind="RESP2",
        method="wb97m-v",
        basis="def2-TZVP",
        work_dir=tmp_path,
        name="ethane",
    )

    assert np.allclose(result["resp_gas"], [-0.30, 0.30])
    assert np.allclose(result["resp_solvated"], [-0.40, 0.40])
    assert np.allclose(result["resp2"], [-0.36, 0.36])
    assert np.allclose(result["resp"], [-0.36, 0.36])
    assert len(calls) == 2
    assert {call["pcm_solvent"] for call in calls} == {None, "Water"}
    assert {call["run_dir"].name for call in calls if call["run_dir"] is not None} == {"vacuum", "solvated_water"}
