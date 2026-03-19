# Example 01: Full workflow (SMILES to box to equilibrium to analysis)

Run:

```bash
python run_full_workflow.py
```

The script will:
- Build a random copolymer from two monomers (*...*).
- Resolve solvents/ions by SMILES using the built-in basic_top library (fallback: parameterize and optional RESP).
- Pack an amorphous cell.
- Run multi-stage GROMACS equilibration via `yadonpy.sim.preset.eq.EQ21step`.
- Run post-processing: thermo summary, RDF (by atomtypes), MSD, ionic conductivity (NE and EH), number density profiles.

Outputs are written under `work_dir/`.

## Charged SMILES note

If a molecule is provided as an explicitly charged (p)SMILES (e.g. `F[P-](F)(F)(F)(F)F`),
yadonpy will use **OpenBabel + UFF** to build and relax an initial 3D geometry, then run
**Psi4 geometry optimization + RESP** (i.e. OPT+RESP). This route avoids the neutral-molecule
conformer-search path.

Recommended install:

```bash
conda install -c conda-forge openbabel
pip install pybel
```
