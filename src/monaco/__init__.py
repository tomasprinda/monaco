from monaco.paths import DATA_DIR, OUTPUTS_DIR, ROOT, output_dir

__all__ = ["DATA_DIR", "OUTPUTS_DIR", "ROOT", "main", "output_dir"]


def main() -> None:
    print(f"data:    {DATA_DIR}")
    print(f"outputs: {OUTPUTS_DIR}")
