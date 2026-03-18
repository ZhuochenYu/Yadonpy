# Example 05 — environment check + basic_top cache

This example shows how to:

1. Run a dependency/environment check (`doctor`).
2. Initialize YadonPy data directory.
3. Resolve molecules by **SMILES** via the built-in `basic_top` library.
   - If a SMILES is already present in the library, YadonPy will reuse the cached `.itp/.top/.gro`.
   - If not present, it will parameterize it (GAFF-family + optional RESP) and then cache the artifacts.

Run:

```bash
python run_doctor_and_cache.py
```
