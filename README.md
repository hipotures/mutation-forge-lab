# Mutation Forge Lab

Mutation Forge Lab is a separate research microproject for testing whether a
resource-bounded Python mutation policy can outperform fixed graph-mutation
baselines under matched budgets. It is a demonstrator, not a production HEG
component and not a claim about the Erdős–Gyárfás conjecture.

Stage 1 is accepted and implements the deterministic experiment harness, immutable connected
cubic datasets, the two current HEG mutation baselines, typed graph/rewrite
interfaces, JSONL and Rich output, SQLite run metadata, and reproducible run
artifacts. The accepted frozen entry point is Mutation Forge
`3b9beba058f472d6f0cad5b6210f34c6dbf96731` with HEG
`fd97451b0f3d87400d1d955a2c6b1b18303344ff`.

Stage 2A adds a deterministic, resource-bounded runtime for one reviewed
`priority(ctx, proposal)` function over bounded immutable probe data. It does
not define the final scientific features, generate proposal pools, call a
model, evolve programs, execute a full proposer, or integrate a policy into
HEG. Stage 2B remains blocked until the Stage 2A report records
`GO_TO_STAGE_2B`.

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

## Stage 2A policy runtime

The checked-in rankers and fixed probes are execution-safety fixtures, not
evidence of graph-search quality:

```console
uv run mforge policy validate fixtures/rankers/weighted.py
uv run mforge policy validate fixtures/rankers/weighted.py --json
uv run mforge policy probe fixtures/rankers/weighted.py --json
uv run mforge policy evaluate fixtures/rankers/weighted.py \
  --config configs/stage2a-probe.toml
```

Evaluation writes the exact source, validation and identity records, effective
limits, fixed behavior signature, bounded worker telemetry, provenance, and
terminal status under `runs/stage2a-*`. The worker requires Linux and fails
closed if CPU, address-space, file-size, file-descriptor, or process-count
limits are unavailable. See
[docs/GENERATED_PYTHON_SECURITY.md](docs/GENERATED_PYTHON_SECURITY.md) for the
versioned contract and authority boundary.

See [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) for milestones and
[docs/STAGE1_REPORT.md](docs/STAGE1_REPORT.md) for the validated first-pass
results. The historical Stage 1 report is unchanged; current status is in
[docs/reports/STAGE2A_REPORT.md](docs/reports/STAGE2A_REPORT.md). See
[docs/PROFILING.md](docs/PROFILING.md) for the aggregate runtime profile and
an on/off overhead check.
