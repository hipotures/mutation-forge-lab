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

Stage 2A is accepted at Mutation Forge
`e2d11bb86b4fa5dbc7ebfb441923e0f02e9799a9` with the same HEG pin. It adds a
deterministic, resource-bounded runtime for one reviewed
`priority(ctx, proposal)` function over bounded immutable probe data. It does
not call a model, evolve programs, execute a full proposer, or integrate a
policy into HEG.

Stage 2B starts from that accepted entry point. It adds host-generated,
host-validated legal `k`-switch pools for `k = 2, 3, 4`, freezes the first
scientific context/proposal schemas, and compares reviewed deterministic
random and structural rankers over identical pools. Stage 2B still makes no
model or network call and is not a held-out or HEG-superiority claim.

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

## Stage 2B proposal pools and rankers

The checked-in preregistration freezes the graph, policy seeds, gate,
proposal/feature budgets, Stage 2A worker limits, and both repository pins
before the benchmark is run:

```console
uv run mforge proposals inspect \
  --config configs/stage2b-preregistered.toml --json
uv run mforge policy evaluate fixtures/rankers/stage2b_structural.py \
  --config configs/stage2b-preregistered.toml
uv run mforge policy compare RANDOM STRUCTURAL \
  --config configs/stage2b-preregistered.toml --json
```

Each step creates one immutable ordered pool. Both rankers receive that exact
pool through the Stage 2A worker, with deterministic proposal-ID tie-breaking.
The host alone owns graphs, rewrite validation, scoring, feature caches, and
experiment state. Only each ranker's selected plan is authoritatively scored;
there is no hidden best-of-pool scoring. The schemas are
[stage2b-context.schema.json](configs/schemas/stage2b-context.schema.json) and
[stage2b-proposal.schema.json](configs/schemas/stage2b-proposal.schema.json).

See [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) for milestones and
[docs/STAGE1_REPORT.md](docs/STAGE1_REPORT.md) for the validated first-pass
results. The historical Stage 1 report is unchanged; accepted Stage 2A status
is in [docs/reports/STAGE2A_REPORT.md](docs/reports/STAGE2A_REPORT.md), and
current Stage 2B evidence is in
[docs/reports/STAGE2B_REPORT.md](docs/reports/STAGE2B_REPORT.md). See
[docs/PROFILING.md](docs/PROFILING.md) for the aggregate runtime profile and
an on/off overhead check.
