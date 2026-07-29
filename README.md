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

Stage 2B is completed and validated. Issue #6 is closed as completed. The
implementation adds host-generated, host-validated legal `k`-switch pools for
`k = 2, 3, 4`, freezes the first scientific context/proposal schemas, and
compares reviewed deterministic random and structural rankers over identical
pools. The preregistered efficacy gate failed: the structural ranker did not
clear the required improvement over random, so the final decision is
**`NO_GO`**. The implementation and negative evidence remain retained.
Stage 3, model use, evolution, full proposer work, and HEG policy integration
remain blocked.

Stage 2C is completed as a diagnostic follow-up to that retained `NO_GO`. It
reproduced the immutable Stage 2B control, measured rank/tie/metric behavior,
and used an explicitly opt-in full-pool oracle for toy diagnostics only. The
primary diagnosis is `BENCHMARK_SATURATION`; the next-step decision is
`DESIGN_STAGE_2D_PREREGISTRATION`. That decision recommends only the design of
a separate, future, approved preregistration. It does not authorize Stage 2D
execution or Stage 3. Oracle scores are computed after policy selections are
fixed, are separately accounted, and cannot affect the historical gate or
normal Stage 2B commands.

Stage 2D is completed and validated. Its checked-in preregistration froze 512
paired real trajectories over toy orders 10 and 12 into exactly eight
immutable shards before any confirmatory result was observed. All eleven
preregistered gates passed, including exact primary/replay identity, and the
decision is **`GO_TO_STAGE_3`**. Stage 2B remains the historical `NO_GO`; the
Stage 2D result neither rewrites that evidence nor constitutes held-out
generalization or HEG superiority.

Stage 3 issue #9 is implemented, preregistered, and offline-validated on
`agent/stage3-issue-9`. The immutable generation freeze is commit
`a3bd09a0fcbc846c7b33b6c720eda96d136da87a`, tagged
`stage3-generation-frozen-v1`. The installed App Server advertised the exact
frozen `gpt-5.6-luna`/`high` profile, but the required private Codex home was
not authenticated and no supported isolated reference to the existing auth
store was available. The run failed closed before any model thread or turn:
the terminal decision is **`INCONCLUSIVE_INFRASTRUCTURE_FAILURE`** with zero
provider/model calls. Stage 4, evolution, full proposer work, held-out
evaluation, and HEG policy integration remain blocked. See
`docs/reports/STAGE3_REPORT.md`.

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

## Stage 2C diagnostics

The checked-in Stage 2C configuration freezes the exploratory orders, graph
seeds, policy seeds, horizons, unchanged Stage 2B pool/feature/worker budgets,
artifact bounds, and expected Stage 2B control identities before execution:

```console
uv run mforge diagnostics stage2c-control \
  --config configs/stage2b-preregistered.toml --json
uv run mforge diagnostics pool-oracle \
  --config configs/stage2c-diagnostic.toml --json
uv run mforge diagnostics stage2c-matrix \
  --config configs/stage2c-diagnostic.toml --json
```

The control command fails closed on any mismatch with the published Stage 2B
metrics or identities. `pool-oracle` and `stage2c-matrix` are the only commands
that enable full-pool toy scoring. They persist separately timed oracle
accounting, trajectory-parity proofs, feature statistics, bounded canonical
rank records, deterministic replay evidence, and terminal status under
`runs/stage2c-*`. Normal Stage 1/2A/2B commands remain selected-only and have no
oracle switch. Diagnostic execution does not use a network, model, or App
Server.

## Stage 2D preregistered trajectories

The coordinator plans and reduces; immutable shard workers only execute their
assigned episode IDs from the annotated `stage2d-preregistered-v1` tag:

```console
uv run mforge stage2d plan \
  --config configs/stage2d-preregistered.toml --json
uv run mforge stage2d run-shard \
  --config configs/stage2d-preregistered.toml \
  --shard shard-00 --output-dir RUN/shard-00 --json
uv run mforge stage2d reduce \
  --config configs/stage2d-preregistered.toml \
  --input-root RUN --output-dir RUN/reduction --workers 8 --json
uv run mforge stage2d verify-replay \
  --primary PRIMARY/reduction/summary.json \
  --replay REPLAY/reduction/summary.json \
  --output PRIMARY/replay-verification.json --json
```

Each policy owns its current graph and score. A strict accepted improvement
advances only that policy; after states diverge, each policy generates a
bounded pool independently from its own graph using the same
outcome-independent episode/step seed. Only the selected plan is scored.
Shard artifacts are bounded gzip JSONL with exactly-once assignment checks,
timing-stripped canonical hashes, CPU affinity and single-thread environment
evidence, and terminal status. The second complete run is replay evidence,
never an additional statistical sample. No Stage 2D command has a model,
network, App Server, oracle, evolution, or HEG-write path.

## Stage 3 frozen one-shot pipeline

The Stage 3 commands render the same canonical state in Rich and JSON modes:

```console
uv run mforge stage3 appserver-doctor \
  --config configs/stage3-generation.toml \
  --auth-json ~/.codex/auth.json \
  --json
uv run mforge stage3 freeze \
  --config configs/stage3-generation.toml --json
uv run mforge stage3 generate \
  --config configs/stage3-generation.toml \
  --auth-json ~/.codex/auth.json \
  --concurrency 8 \
  --json
uv run mforge stage3 validate RUN --json
uv run mforge stage3 evaluate \
  --config configs/stage3-generation.toml --run RUN --workers 8 --json
uv run mforge stage3 verify-replay PRIMARY REPLAY --json
```

`freeze`, `validate`, and replay are model-free. `generate` requires the
immutable preregistration tag and an explicitly authorized `auth.json`. The
adapter copies only that file into each private Codex home with bounded,
fail-closed permission checks; it never loads the user's Codex configuration,
skills, plugins, hooks, memories, or trust settings. Earlier protocol-failure
evidence remains retained, and each compatibility correction receives a new
immutable freeze tag instead of moving an existing tag.

See [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) for milestones and
[docs/STAGE1_REPORT.md](docs/STAGE1_REPORT.md) for the validated first-pass
results. The historical Stage 1 report is unchanged; accepted Stage 2A status
is in [docs/reports/STAGE2A_REPORT.md](docs/reports/STAGE2A_REPORT.md), and
completed Stage 2B implementation and negative evidence are retained in
[docs/reports/STAGE2B_REPORT.md](docs/reports/STAGE2B_REPORT.md). The Stage 2C
diagnosis and still-blocked next step are in
[docs/reports/STAGE2C_DIAGNOSTIC_REPORT.md](docs/reports/STAGE2C_DIAGNOSTIC_REPORT.md).
The frozen Stage 3 implementation and infrastructure result are in
[docs/reports/STAGE3_REPORT.md](docs/reports/STAGE3_REPORT.md).
See
[docs/PROFILING.md](docs/PROFILING.md) for the aggregate runtime profile and
an on/off overhead check.
