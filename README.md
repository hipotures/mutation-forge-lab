# Mutation Forge Lab

Mutation Forge Lab is a separate research microproject for testing whether a
resource-bounded Python mutation policy can outperform fixed graph-mutation
baselines under matched budgets. It is a demonstrator, not a production HEG
component and not a claim about the Erdős–Gyárfás conjecture.

Stage 1 implements the deterministic experiment harness, immutable connected
cubic datasets, the two current HEG mutation baselines, typed graph/rewrite
interfaces, JSONL and Rich output, SQLite run metadata, and reproducible run
artifacts. It does **not** execute generated Python, call a model, evolve
programs, generate k-switches, or modify/integrate into HEG.

## Setup and checks

Python 3.12 or newer and the read-only sibling repository at `../heg` are
required.

```console
uv sync
uv run mforge doctor --heg-repo ../heg
uv run pytest
uv run ruff check .
uv run mypy
```

## Stage 1 workflow

```console
uv run mforge dataset build --config configs/stage1-smoke.toml
uv run mforge baseline run --config configs/stage1-smoke.toml
uv run mforge baseline run --config configs/stage1-smoke.toml --json
uv run mforge inspect RUN
uv run mforge inspect RUN --json
uv run mforge compare RUN_A RUN_B --json
```

`--json` baseline output is JSON Lines only. Durable artifacts are written
under `runs/<run-id>/`, including the resolved provenance, immutable dataset
manifest, event stream, SQLite archive, and canonical run summary.

The fixed `fixed_ils_tabu` controller accepts strict score improvements. Both
baseline policies receive the same initial graphs, graph and policy seeds,
evaluation count, witness cap, wall budget, validation, and scoring path. Only
the reviewed HEG proposal operator differs.

See [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) for milestones and
[docs/STAGE1_REPORT.md](docs/STAGE1_REPORT.md) for the validated first-pass
results. See [docs/PROFILING.md](docs/PROFILING.md) for the aggregate runtime
profile and an on/off overhead check.
