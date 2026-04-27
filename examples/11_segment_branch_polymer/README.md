# Example 11: Segment-First Branched Polymer

This example demonstrates the segment-first polymer workflow:

- build reusable linear segments from prepared repeat units;
- preserve `[2*]` branch markers while consuming the main-chain `*` markers;
- build long block chains from prebuilt segments;
- attach branches either before polymerization (`mode="pre"`) or after main-chain growth (`mode="post"`);
- terminate, assign a force field, build a small amorphous cell, and optionally run MD.

Run the full workflow:

```bash
cd examples/11_segment_branch_polymer
python run_segment_branch_workflow.py
```

For a quick build-only check without MD:

```bash
RUN_MD=0 python run_segment_branch_workflow.py
```

Label convention:

- `*` or `[1*]`: main-chain head/tail connection points.
- `[2*]`, `[3*]`, ...: branch attachment sites preserved by `seg_gen`.
