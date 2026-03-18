"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class XvgData:
    """Parsed GROMACS XVG data."""

    df: pd.DataFrame
    xlabel: str = ""
    ylabel: str = ""
    legends: Optional[List[str]] = None


def read_xvg(path: Path) -> XvgData:
    """Read a GROMACS .xvg file into a DataFrame.

    Notes:
        - Ignores lines starting with '#' and '@'.
        - Tries to recover axis labels and legends.
        - Returned DataFrame columns: ['x', <series...>]

    Args:
        path: Path to the xvg file.

    Returns:
        XvgData
    """
    xlabel = ""
    ylabel = ""
    legends: List[str] = []
    data_rows: List[List[float]] = []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("@"):  # metadata
            # examples:
            # @    xaxis  label "Time (ps)"
            # @    yaxis  label "Density"
            # @    s0 legend "Density"
            if "xaxis" in line and "label" in line:
                q = line.split('"', 2)
                if len(q) >= 2:
                    xlabel = q[1]
            if "yaxis" in line and "label" in line:
                q = line.split('"', 2)
                if len(q) >= 2:
                    ylabel = q[1]
            if "legend" in line and "s" in line:
                q = line.split('"', 2)
                if len(q) >= 2:
                    legends.append(q[1])
            continue

        # numeric data line
        parts = line.split()
        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue
        if len(row) >= 2:
            data_rows.append(row)

    if not data_rows:
        raise ValueError(f"No numeric data found in XVG: {path}")

    ncols = len(data_rows[0])
    # Normalize ragged rows by trimming
    data_rows = [r[:ncols] for r in data_rows if len(r) >= ncols]

    cols = ["x"] + [f"y{i}" for i in range(1, ncols)]
    if legends and (len(legends) == ncols - 1):
        cols = ["x", *legends]

    df = pd.DataFrame(data_rows, columns=cols)
    return XvgData(df=df, xlabel=xlabel, ylabel=ylabel, legends=legends or None)
