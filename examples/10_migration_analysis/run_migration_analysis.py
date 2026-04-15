#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from rdkit import Chem

import yadonpy as yp


# Example 10: standalone migration analysis for an existing YadonPy work directory.
#
# This example intentionally mirrors the Example 02 post-processing style:
#   analy = job.analyze()
#   rdf = analy.rdf(center_mol=...)
#   msd = analy.msd()
#   sigma = analy.sigma(msd=msd, temp_k=...)
#   migration = analy.migration(center_mol=...)
#
# The difference is that here we open an already-finished work directory via
# AnalyzeResult.from_work_dir(...).

WORK_DIR = Path(__file__).resolve().parent / "work_dir"
CENTER_SMILES = "[Li+]"
TEMPERATURE_K = 333.15
EXPERT_MODE = False


def main() -> None:
    yp.doctor(print_report=True)

    analy = yp.AnalyzeResult.from_work_dir(WORK_DIR)
    center_mol = Chem.MolFromSmiles(CENTER_SMILES)
    if center_mol is None:
        raise RuntimeError(f"Failed to parse center SMILES: {CENTER_SMILES!r}")

    rdf = analy.rdf(center_mol=center_mol)
    msd = analy.msd()
    sigma = analy.sigma(msd=msd, temp_k=TEMPERATURE_K)
    migration = analy.migration(center_mol=center_mol, expert_mode=EXPERT_MODE)

    migration_summary = migration.get("migration_summary") or {}
    event_counts = migration.get("event_counts") or {}
    residence_summary = migration.get("residence_summary") or {}
    role_markov = migration.get("markov_role_summary") or {}
    site_markov = migration.get("markov_site_summary") or {}

    print("\nMigration analysis complete.")
    print(f"Work dir: {WORK_DIR}")
    print(f"Center: {migration_summary.get('center_moltype')}")
    print(f"Frames: {migration_summary.get('n_frames')}")
    print(f"Selected lag: {migration_summary.get('selected_lag_ps')} ps")
    print(f"Markov confidence: {migration_summary.get('markov_confidence')}")
    print(f"RDF targets: {len([k for k in rdf.keys() if not str(k).startswith('_')])}")
    print(f"MSD species: {len([k for k in msd.keys() if not str(k).startswith('_')])}")
    print(f"Sigma_NE_upper: {sigma.get('sigma_ne_upper_bound_S_m')}")
    print(f"Sigma_EH_total: {sigma.get('sigma_eh_total_S_m')}")
    print(f"Role states: {role_markov.get('state_count')}")
    print(f"Site states: {site_markov.get('state_count')}")
    print("Observed event counts:")
    for key, value in sorted(event_counts.items()):
        print(f"  - {key}: {value}")
    print("Residence:")
    for role, rec in residence_summary.items():
        print(
            f"  - {role}: available={rec.get('available')} "
            f"continuous={rec.get('continuous_residence_time_ps')} ps "
            f"intermittent={rec.get('intermittent_residence_time_ps')} ps"
        )
    print(f"Outputs: {migration_summary.get('outputs')}")


if __name__ == "__main__":
    main()
