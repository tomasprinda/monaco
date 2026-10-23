"""Template for an analysis entry point: read from data/, write to a timestamped run dir.

Usage: uv run scripts/example.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import monaco


def main() -> None:
    out = monaco.output_dir(Path(__file__).stem)  # outputs/example/<yymmdd_hhmmss>/

    # Illustrative numbers only; real inputs belong in data/.
    df = pd.DataFrame({"city": ["Monaco", "London", "Paris"], "eur_per_m2": [50000, 20000, 11000]})
    df.to_csv(out / "prices.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(df["city"], df["eur_per_m2"])
    ax.set_xlabel("EUR / m²")
    fig.savefig(out / "prices.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
