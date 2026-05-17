# Example 12: Enhanced Sampling

This folder collects enhanced-sampling workflows that combine GROMACS and
PLUMED.  The first supported workflow is umbrella sampling for a solvated Li+
crossing from the electrolyte-rich side of a CMC/electrolyte interface into the
CMC-Na-rich region.

## Implemented

- `01_umbrella_sampling_pmf`: runnable GROMACS pull-code umbrella sampling,
  `gmx wham` PMF reconstruction, PLUMED coordination-CV recording, CSV/SVG
  summaries, and a slow MP4 overview when `ffmpeg` is available.

## Placeholders

- `02_metadynamics`: design placeholder only.
- `03_blue_moon`: design placeholder only.

The placeholder folders are intentionally explicit.  They reserve the example
layout for future work without implying that those workflows have been tested.

