# Umbrella Sampling PMF: Li+ From Electrolyte Into CMC-Na

This example reuses the Example 08-03 layer-stack chemistry:
basal graphite | CMC-Na | carbonate/LiPF6 electrolyte.  The enhanced-sampling
coordinate focuses only on the CMC/electrolyte interface.  YadonPy selects a Li+
whose initial solvent-oxygen coordination is closest to four, then umbrellas
the signed z offset between that Li+ and the CMCNA layer COM.

The umbrella bias is applied by the GROMACS pull code so `gmx wham` can consume
the standard `tpr/pullx/pullf` files.  PLUMED records coordination CVs only:
`cn_solvent`, `cn_target`, and `cn_anion`.

## Run

```bash
cd examples/12_enhanced_sampling/01_umbrella_sampling_pmf
python run_li_to_cmc_umbrella_pmf.py
```

Default production settings are 31 windows with 1 ns production per window.
For a short smoke test:

```bash
YADONPY_UMBRELLA_WINDOWS=5 \
YADONPY_UMBRELLA_STEERING_NS=0.01 \
YADONPY_UMBRELLA_WINDOW_EQ_NS=0.001 \
YADONPY_UMBRELLA_WINDOW_NS=0.01 \
python run_li_to_cmc_umbrella_pmf.py
```

## Outputs

- `02_solvated_li_selection/umbrella_sampling_manifest.json`
- `03_steering_pull/`
- `04_umbrella_windows/window_*/`
- `05_wham_pmf/pmf.xvg`
- `06_postprocess/pmf.csv`
- `06_postprocess/coordination_by_window.csv`
- `06_postprocess/pmf_coordination_overlay.svg`
- `06_postprocess/umbrella_pmf_summary.json`
- `06_postprocess/umbrella_pmf_timeseries.mp4` when `ffmpeg` is available

