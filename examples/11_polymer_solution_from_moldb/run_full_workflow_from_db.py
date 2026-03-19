from __future__ import annotations

# Example 11: Polymer solution workflow using precomputed MolDB entries (geometry + charges).
# Prerequisite: run examples/10_moldb_batch_from_csv/run_batch_build_db.py first.

from pathlib import Path

from yadonpy.ff.gaff2 import GAFF2
from yadonpy.core import poly
from yadonpy.io.mol2 import write_mol2_from_rdkit
from yadonpy.sim.preset import eq

# Work directory
HERE = Path(__file__).resolve().parent
work_dir = HERE / "work_dir"
work_dir.mkdir(parents=True, exist_ok=True)

# Use the SAME project-local DB as Example 10:
db_dir = (HERE.parent / "10_moldb_batch_from_csv" / "moldb").resolve()

# --- load monomers/termination/solvent from DB (requires they exist) ---
smiles_A = r"*CCO*"
smiles_B = r"*COC*"
ter_smiles = r"[H][*]"
solvent_smiles_A = "CO"  # methanol
solvent_smiles_B = "O"   # water

monomer_A = GAFF2.mol(smiles_A, db_dir=db_dir, require_db=True)
monomer_B = GAFF2.mol(smiles_B, db_dir=db_dir, require_db=True)
ter1 = GAFF2.mol(ter_smiles, db_dir=db_dir, require_db=True)

solvent_A = GAFF2.mol(solvent_smiles_A, db_dir=db_dir, require_db=True)
solvent_B = GAFF2.mol(solvent_smiles_B, db_dir=db_dir, require_db=True)

# --- build a random copolymer by self-avoiding random walk ---
ratio = [0.5, 0.5]
reac_ratio = [1.0, 1.0]
dp = poly.calc_n_from_num_atoms([monomer_A, monomer_B], 1000, ratio=ratio, terminal1=ter1)

copoly = poly.random_copolymerize_rw(
    [monomer_A, monomer_B], dp, ratio=ratio, reac_ratio=reac_ratio, tacticity="atactic"
)
copoly = poly.terminate_rw(copoly, ter1)

# --- FF assignment is done on-demand (fast) ---
ff = GAFF2()
result = ff.ff_assign(copoly)
if not result:
    raise RuntimeError("Cannot assign FF to polymer.")
result = ff.ff_assign(solvent_A)
if not result:
    raise RuntimeError("Cannot assign FF to solvent_A.")
result = ff.ff_assign(solvent_B)
if not result:
    raise RuntimeError("Cannot assign FF to solvent_B.")

# Export molecules (optional)
write_mol2_from_rdkit(mol=copoly, out_dir=work_dir / "00_molecules", name="copoly")
write_mol2_from_rdkit(mol=solvent_A, out_dir=work_dir / "00_molecules", name="solvent_A")
write_mol2_from_rdkit(mol=solvent_B, out_dir=work_dir / "00_molecules", name="solvent_B")

# --- build amorphous cell ---
counts = [1, 500, 500]
ac = poly.amorphous_cell(
    [copoly, solvent_A, solvent_B],
    counts,
    charge_scale=1.0,
    density=0.05,
)

# --- equilibration (EQ21) ---
# NOTE: Use EQ21step (the unified staged equilibration) and do NOT rebuild the system.
eqmd = eq.EQ21step(ac, work_dir=work_dir)
ac = eqmd.exec(temp=298.15, press=1.0, mpi=1, omp=16, gpu=1, gpu_id=0, restart=True)

# --- analysis (includes polymer Rg gate) ---
analy = eqmd.analyze()
analy.check_eq()
prop_data = analy.get_all_prop(temp=298.15, press=1.0, save=True)
msd = analy.msd()

print("prop_data keys:", list(prop_data.keys()))
print("msd:", msd)
