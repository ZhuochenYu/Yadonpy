from pprint import pprint

from yadonpy.diagnostics import doctor
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.api import ensure_basic_top
from yadonpy.workflow import steps
from yadonpy.ff.gaff2_mod import GAFF2_mod
# Optional: pin data dir for reproducibility
# export YADONPY_DATA_DIR=/path/to/your/yadonpy_data

# --- global restart switch ---
restart_status = True  # set False to force re-run

# Prefer the GAFF2-family factory (default: GAFF2_mod) for robust ion handling.
_ff_gaff2 = GAFF2_mod()
_ff_gaff2_name = getattr(_ff_gaff2, "name", "gaff2_mod")

# Common battery molecules
SMILES_LIST = [
    ("EC", "O=C1OCCO1", _ff_gaff2_name),
    ("EMC", "CCOC(=O)OC", _ff_gaff2_name),
    ("Li+", "[Li+]", "merz"),
    # PF6- (high-symmetry anion) - use GAFF2-family (default variant is robust)
    ("PF6-", "F[P-](F)(F)(F)(F)F", _ff_gaff2_name),
]


def main():
    # 1) Check environment/dependencies
    doctor(print_report=True)

    # 2) Initialize data directory (copies built-in basic_top + library.json)
    layout = ensure_initialized()
    print("\n[DATA_DIR]")
    print(layout.root)

    # 3) Resolve / cache artifacts by SMILES (resumable)
    print("\n[RESOLVE basic_top artifacts]")
    steps.ensure_basic_tops(
        [smi for _name, smi, _ff in SMILES_LIST],
        ff_names=[ff for _name, _smi, ff in SMILES_LIST],
        restart=restart_status,
        work_dir=layout.root,
    )

    for name, smi, ff in SMILES_LIST:
        entry = ensure_basic_top(smi, ff_name=ff)
        print(f"\n== {name} ({ff}) ==")
        pprint(entry)


if __name__ == "__main__":
    main()
