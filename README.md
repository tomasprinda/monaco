# monaco

Data analysis of real estate: property prices, renting, and related topics such as demographics.

Named after Monaco, which has the most expensive real estate per square meter in the world.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The project pins Python 3.15 (`.python-version`) and is
configured to use only uv-managed interpreters, never the system Python.

```sh
uv sync                 # installs Python 3.15 if needed, creates .venv, installs deps
```

## Layout

```
src/monaco/    Python helpers shared across analyses (paths, loaders, ...)
scripts/       Analysis entry points
notebooks/     Jupyter notebooks for interactive exploration
data/          Input datasets (git-ignored)
outputs/       Generated results (git-ignored)
tests/         Tests
```

## Data and outputs

The repo root must contain two directories, `data/` and `outputs/`. For now they are plain
directories; later they may become symlinks to storage elsewhere, so code should only ever reach
them through these paths (`monaco.DATA_DIR`, `monaco.OUTPUTS_DIR`) and never assume they are
real directories inside the repo. Neither is tracked by git at all; after cloning, create them
(`mkdir data outputs`) or symlink them to where the data lives.

- `data/` holds input data: downloaded or hand-collected datasets that analyses read from.
- `outputs/` holds everything analyses produce, one directory per run:

  ```
  outputs/<name>/<yymmdd_hhmmss>/
  ```

  `<name>` is usually the entry-point script's name (e.g. `scripts/rent_yields.py` →
  `outputs/rent_yields/`), or another name that describes the analysis. The timestamp is the run's
  start time, e.g. `outputs/rent_yields/261023_142501/`. Runs never overwrite each other.
  Use `monaco.output_dir(name)` to create the run directory.

## Common commands

```sh
uv run scripts/example.py                   # sample run → outputs/example/<yymmdd_hhmmss>/
uv run monaco                               # print data/ and outputs/ locations
uv run jupyter lab                          # open notebooks
uv run pytest
uv run ruff check . && uv run ruff format .
uv add <package>                            # add a dependency
```
