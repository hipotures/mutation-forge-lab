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
**`NO_GO`**. The implementation and negative evidence remain retained. At the
close of Stage 2B, Stage 3, model use, evolution, full proposer work, and HEG
policy integration remained blocked.

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

Stage 3 issue #9 is completed and validated on its dedicated branch. Its
frozen implementation used eight ordered one-shot Codex App Server slots at
`gpt-5.6-luna` with `high` reasoning, strict structured output, private
no-tool capsules, Stage 2A validation and 10,000-call smoke checks, immutable
development trajectories, deterministic replay, and a twelve-part gate. The
official campaign source was committed, pushed, annotated with
`stage3-generation-frozen-v11`, and recorded on issue #9 before generation.
Adapter troubleshooting used three explicitly
user-authorized connectivity turns: two completed and one ended with partial
usage. Their fixed connectivity payloads and outputs are retained as
diagnostic evidence, excluded from candidate generation and evaluation, and
were not used to change prompts, schemas, rankers, manifests, metrics, or
gates. The user reviewed that diagnostic boundary and authorized this fresh
freeze. The retained v6 official attempt was rejected before inference because
its structured-output `schema_version` property omitted the explicit JSON
Schema string type. v7 added that type but exposed a second pre-inference
transport rejection because `uniqueItems` is not supported by the App Server
structured-output subset. v8 restricts the transport schema to a tested
keyword allowlist while retaining all size/cardinality enforcement in the
application parser. It also fixes the schema-derived prompt renderer so
nullable and referenced fields are explicit and snapshots all eight rendered
slot prompts. A versioned glossary tied to the context/proposal schema hashes
defines the decision problem, every field, pool-constant versus
candidate-specific scope, vector alignment, aliases, and budget caveats. The
retained v8 official run passed structured-output validation and four slots
emitted partial model text, but no slot produced a final response or final
usage. The adapter had incorrectly reused the 64 KiB frame bound as an
aggregate stdout bound, while eight Codex processes could exhaust native
worker creation because the child `RLIMIT_NPROC=1024` was charged against all
tasks owned by the shared user rather than one capsule. The user authorized a
v9 infrastructure-only freeze that keeps the 64 KiB per-frame bound,
separately caps aggregate stdout at 2 MiB, raises that bounded user-wide limit
100-fold to 102400, and fixes Tokio, Rayon, and numerical-library workers to
one per capsule.
Prompts, schemas, rankers, manifests, metrics, gates, and evaluation behavior
are unchanged. The retained v9 run started all eight turns, completed four,
accepted one candidate, and exposed bounded 120-second turn, 256 KiB transport
log, and 64 KiB incoming-frame/512-event queue limits. The user authorized v10
with a 600-second turn limit, 100-fold final-response limit, and tenfold event,
transcript, stdout, and queue bounds. The v6–v9 tags and failure artifacts
remain unchanged.
The retained v10 run completed all eight model turns without an infrastructure
failure and produced four valid unique candidates. Four AST-invalid candidates
did not receive their permitted repair because the orchestrator recognized
only an incomplete hard-coded subset of validator error codes. v11 classifies
all errors originating from static AST validation as repairable while keeping
transport, usage, and runtime failures terminal.
The retained v11 campaign completed all eight initial turns and five permitted
repair turns. Seven candidates passed the old validator; one final response was
incorrectly rejected because its finite `range(min(...))` expression could not
be proven by the static loop-bound heuristic. The user-authorized v12
validation amendment removes all static loop-bound and termination inference,
permits `for` and `while`, and leaves slow or infinite programs to the existing
per-candidate worker CPU/wall limits. It does not make another model call.
Provider-free revalidation of the eight retained final responses produced eight
unique valid candidates and completed 80,000 persistent-worker smoke calls.
Raw prompts, responses, JSON-RPC transcripts, usage, and transport logs remain
unchanged; derived revalidation artifacts are stored separately.
The first frozen v12 evaluation completed both passes in memory but could not
persist the monolithic 128-record primary JSONL because duplicated per-step
traces exceeded the writer's 64 MiB pre-compression bound. No scientific
result was retained from that failed attempt. The user-authorized v13
evaluation-only correction stores each step trace exactly once, replaces
redundant full score objects with compact selected-plan score evidence, and
persists primary and replay as eight deterministic bounded shards of sixteen
episodes. It changes no candidate, baseline, episode, seed, metric, bootstrap,
threshold, or gate and makes no model call.
The primary and replay records have the identical timing-stripped SHA-256
`43dee7e356ccc3f11c3fff326a78d16c70b0524a5b046732f6aca289335ccd73`.
All twelve gates passed. `candidate-slot-04` improved pooled median normalized
best-so-far AUC by 15.097% relative to random with a paired 95% bootstrap
interval for the absolute delta of `[0.075000, 0.125000]`, while retaining
99.665% of the structural baseline's pooled median. The final decision is
**`GO_TO_STAGE_4`** and the complete evidence is retained in
`docs/reports/STAGE3_REPORT.md`.
Issue #10 completed the frozen Stage 4 archived search, but the terminal
decision is **`INCONCLUSIVE_INFRASTRUCTURE_FAILURE`** because eight accepted
generation-0 protocol-failure turns have no exact usage. The 40-record archive
contains 19 new unique valid offspring; no initial turn was replaced.
Validation was not eligible, Stage 5 was not started, and HEG was not
modified. See `docs/reports/STAGE4_REPORT.md`.

## Setup and checks

Python 3.12 or newer and the read-only sibling repository at `../heg` are
required.

```console
uv sync --locked
uv run mforge doctor --heg-repo ../heg
uv run pytest
uv run ruff check .
uv run mypy
```

## Experiment workspace workflow

New experiments use the native experiment engine through two stage-free public
commands. `mforge experiment run` invokes the native generation coordinator,
the local Codex App Server provider, sandbox validation/probing, HEG-backed
evaluation, selection, and durable checkpoints. `experiment.toml` is
authoritative: its model, search, evaluation, resource, and invocation-budget
fields are executed as written and immutable scientific fields are recorded in
the workspace lock.

The default file is `./experiment.toml`; `--config PATH` selects another file.
Every required prompt, schema, semantic glossary, and baseline ranker is a
version-controlled native asset. A new experiment does not need a historical
Stage 4 freeze, campaign directory, tag, or previous `runs/` output. HEG
remains the mathematical backend, with its current sibling-checkout commit and
dirty state recorded in the lock. Stage 4 commands and artifacts remain
historical/private regression material and are not part of this workflow.

```toml
schema_version = "mforge.experiment.v1"
exp_id = "test_bla_bla"
workspace = "./workspace"
kind = "ranker-search"
preset = "heg-ranker-evolution-v1"

[run]
wall_seconds = 3600

[model]
provider = "codex"
name = "gpt-5.6-luna"
effort = "high"
concurrency = 8
max_repairs = 1

[search]
population_size = 8
max_generations = 4
max_model_turns = 64
selection = "elite-diversity"

[evaluation]
orders = [10, 12]
graph_seeds = [401, 402, 403, 404]
policy_seeds = [
  4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008,
  4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016,
]
horizon = 32
proposal_pool_size = 12
baselines = ["random", "structural"]
replay = true

[resources]
workers = 8
thread_count = 1
```

Run and inspect the same experiment with:

```console
uv run mforge experiment run
uv run mforge experiment status
uv run mforge experiment run --config configs/large.toml --json
```

The first run creates `<workspace>/<exp_id>/` atomically, retaining the exact
configuration, immutable lock, SQLite orchestration state, append-only
checkpoints, numbered session logs, and expanded per-turn App Server evidence.
Later `run` invocations continue the same `exp_id` from its latest durable
checkpoint; `run.wall_seconds` is a per-invocation budget. All other scientific
configuration is locked, so changing it requires a new `exp_id`. A completed
experiment performs no additional provider or evaluation work. The generated
program contract remains the bounded `priority(ctx, proposal)` ranker, not a
full graph mutation operator.

## Legacy Stage 1 harness

The historical stage-specific commands are intentionally not registered by the
installed `mforge` entry point. Internal regression tests and archived evidence
tools may import the private `mutation_forge.cli.legacy_main` entry point; new
work must use `mforge experiment run/status` above.

The command blocks in the historical sections below preserve old argument
syntax for reproducibility; they are not accepted by the installed `mforge`
shell entry point.

```python
from mutation_forge.cli import legacy_main

raise SystemExit(
    legacy_main(["dataset", "build", "--config", "configs/stage1-smoke.toml"])
)
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

## Stage 3 preregistration and one-shot generation

The offline protocol audit and immutable freeze are separate from live
generation:

```console
uv run mforge stage3 appserver-doctor \
  --config configs/stage3-generation.toml \
  --auth-json ~/.codex/auth.json --json
uv run mforge stage3 freeze \
  --config configs/stage3-generation.toml --json
uv run mforge stage3 generate \
  --config configs/stage3-generation.toml \
  --auth-json ~/.codex/auth.json --concurrency 8 --json
uv run mforge stage3 validate RUN --json
uv run mforge stage3 evaluate \
  --config configs/stage3-generation.toml \
  --run RUN --workers 8 --json
```

`appserver-doctor` starts no inference turn. Production generation requires
the exact catalogued model/effort and creates exactly eight concurrent initial
one-turn capsules. Each private capsule has separate Codex and SQLite homes,
an empty work directory, disabled tools/skills/apps/plugins/web search, and a
secure copied credential that is never persisted. Durable per-slot evidence
uses TheML-compatible names for request, response, provider raw data, profile,
RPC, events, stdout, stderr, and the bounded rollout copy. The temporary
capsule is deleted only after those logs are flushed. Repair is limited to one
schema/AST-only turn per slot and receives no benchmark result.

The complete reviewed prompt artifacts are checked in under
`prompts/stage3-slots/`. Regenerate or verify them with:

```console
uv run python scripts/render_stage3_prompts.py
uv run python scripts/render_stage3_prompts.py --check
```

See [docs/MASTER_PLAN.md](docs/MASTER_PLAN.md) for milestones and
[docs/STAGE1_REPORT.md](docs/STAGE1_REPORT.md) for the validated first-pass
results. The historical Stage 1 report is unchanged; accepted Stage 2A status
is in [docs/reports/STAGE2A_REPORT.md](docs/reports/STAGE2A_REPORT.md), and
completed Stage 2B implementation and negative evidence are retained in
[docs/reports/STAGE2B_REPORT.md](docs/reports/STAGE2B_REPORT.md). The Stage 2C
diagnosis and still-blocked next step are in
[docs/reports/STAGE2C_DIAGNOSTIC_REPORT.md](docs/reports/STAGE2C_DIAGNOSTIC_REPORT.md).
The completed, infrastructure-inconclusive Stage 4 campaign is documented in
[docs/reports/STAGE4_REPORT.md](docs/reports/STAGE4_REPORT.md).
See
[docs/PROFILING.md](docs/PROFILING.md) for the aggregate runtime profile and
an on/off overhead check.
