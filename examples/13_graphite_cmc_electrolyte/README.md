# Example 13: Graphite Basal Plane + CMC + Electrolyte

This example builds a three-layer starting structure:

1. a GAFF-typed graphite basal substrate at the bottom of the box;
2. a compact CMC slab above the graphite;
3. a LiPF6 carbonate-electrolyte slab above the CMC.

The goal is to show the new `build_graphite(...)` workflow and the linear
stacking style for multi-block systems. The script focuses on construction and
GROMACS export. It does not run a long production relaxation by default.

Notes:

- The graphite builder uses an ideal AB-stacked graphite lattice
  (`C-C = 1.42 A`, interlayer distance `3.35 A`) instead of a proprietary
  Materials Studio source file.
- The electrolyte slab uses the same `EC:DEC:EMC = 3:2:5` and `1 M LiPF6`
  planning logic as the interface examples, but in a smaller export-oriented
  setup.
