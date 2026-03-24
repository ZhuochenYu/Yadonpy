# Example 12 - CMC-Na vs 1M LiPF6 electrolyte interface

This is the large charged-polymer interface example in the current release.

Current study shape:

- CMC degree of polymerization: `150`
- number of CMC chains: `6`
- electrolyte: `1 M` LiPF6
- solvent recipe: `EC:DEC:EMC = 3:2:5`

Workflow:

1. build and equilibrate the CMC-Na bulk first;
2. use the equilibrated CMC `XY` box as the authoritative interface footprint;
3. build and equilibrate an isotropic probe electrolyte bulk;
4. resize and rebuild the final electrolyte against the polymer-matched target box with extra initial `Z` slack;
5. let `recommend_polymer_diffusion_interface_recipe(...)` choose the route-B vacuum-buffered staged diffusion setup;
6. assemble the interface and run staged release before unrestricted exchange.

This example is intentionally larger and more conservative than Examples 10 and 11. It is meant as the reference workflow when interface robustness matters more than minimum system size.

PF6 is still treated as a MolDB-backed RESP + DRIH species. Run Example 01 first if the PF6 record is not already present.

## One-click entry points

Windows local validation on the current source tree:

```bat
run_eg12_local_cuda.bat
```

That wrapper activates the `yadonpy` conda environment, forces `PYTHONPATH=<repo>/src`, caps local CPU use to `12` threads, and defaults to `--profile smoke` so you can validate GROMACS plus YadonPy wiring before launching the full system.

Windows full run:

```bat
run_eg12_local_cuda.bat --profile full
```

Linux or remote GPU node:

```bash
./run_eg12_remote_cuda.sh
```

The remote wrapper also forces `PYTHONPATH=<repo>/src` so the example does not accidentally import an older site-packages build.

Useful direct flags:

- `--profile smoke` for a smaller, shorter debug system
- `--profile full` for the full `DP = 150`, `6`-chain study
- `--stop-after polymer_bulk|probe_bulk|electrolyte_bulk|interface_build` for staged debugging
- `--with-term-qm` only when the environment really has `psi4`; by default the `[H][*]` termination skips QM so local Windows runs are not blocked by missing Psi4
