"""Puts the repo root on the import path; the modules are flat scripts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
