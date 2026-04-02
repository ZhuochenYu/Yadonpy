from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
os.environ.setdefault("YADONPY_AUTO_INIT", "0")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
