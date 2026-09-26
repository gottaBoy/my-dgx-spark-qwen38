"""Offline test suite for qwen38-spark.

Runs anywhere, including a laptop with no GPU and no Docker: everything that
touches the box is behind an injected input, and the tests feed captured text.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put lib/ on the path so `python3 -m unittest discover` works from the repo
# root with no install step and no PYTHONPATH ritual. A suite that needs setup
# before it can be run is a suite that does not get run.
_LIB = Path(__file__).resolve().parents[1] / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
