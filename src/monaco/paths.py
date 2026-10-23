"""Standard locations for input data and generated outputs.

`data/` and `outputs/` live in the repo root. They may be plain directories or symlinks to
storage elsewhere; everything here resolves through them either way.
"""

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"


def output_dir(name: str, base: Path = OUTPUTS_DIR) -> Path:
    """Create and return a fresh run directory: `outputs/<name>/<yymmdd_hhmmss>/`."""
    path = base / name / datetime.now().strftime("%y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=False)
    return path
