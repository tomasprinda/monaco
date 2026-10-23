# CLAUDE.md

## Data and outputs

- The repo root must contain two directories: `data/` (input data) and `outputs/` (generated
  results). They are plain directories for now but may become symlinks later — always reach them
  via `monaco.DATA_DIR` / `monaco.OUTPUTS_DIR`, never assume they are real directories in the repo.
- Both are git-ignored except for a `.gitkeep`. Never commit data or outputs.
- Every run writes to its own directory: `outputs/<name>/<yymmdd_hhmmss>/`, where `<name>` is
  the entry-point script's name (e.g. `scripts/foo.py` → `outputs/foo/`) or another descriptive
  name, and the timestamp is the run's start time. Create it with `monaco.output_dir(name)`; never
  write files directly into `outputs/` or overwrite a previous run.
