# Mutation Forge Lab

Mutation Forge Lab searches for deterministic declarative programs that
construct bounded graph rewrites for HEG. It uses the local Codex App Server
to generate typed policy ASTs, validates and interprets them with strict
resource limits, evaluates them against HEG and fixed DSL baselines, and
evolves the best candidates across multiple generations.

The project is a standalone, auditable research application. It treats the
sibling HEG repository as a read-only mathematical backend and never modifies
HEG. Native v3 policies are data, not Python. They use a versioned
selector/action DSL to construct a private transactional overlay and emit one
bounded `RewritePlan` or `NoPlan`. The host still owns final legality,
authoritative scoring, acceptance, persistence, and exact verification.

## How it works

For every experiment, Mutation Forge Lab:

1. creates an immutable experiment workspace and records repository provenance;
2. freezes an epoch snapshot and asks the configured local Codex model for
   eight differentiated policy ASTs in bounded batches;
3. validates each AST independently and immediately streams valid programs to
   deterministic episode-level evaluator shards;
4. keeps provider calls and useful CPU evaluation concurrent with bounded
   queues and backpressure;
5. evaluates candidates, retained parents, and four fixed DSL baselines under
   matched manifests and score protocols;
6. freezes a development promotion shortlist, evaluates every shortlisted
   program on the sealed validation panel, and commits selection
   deterministically after the cohort reaches a terminal state;
7. retains raw and canonical programs, provenance, component score evidence,
   semantic checkpoints, observational telemetry, and verification artifacts.

The installed CLI intentionally exposes only the current native product:

```text
mforge doctor
mforge experiment run
mforge experiment status
mforge experiment stop --final
```

Historical stage-specific implementations remain in the repository as
regression and research evidence, but they are not public CLI commands.

## Requirements

- Linux;
- Python 3.12 or newer;
- [uv](https://docs.astral.sh/uv/);
- a functional sibling HEG checkout, normally at `../heg`;
- an installed and locally authenticated `codex` executable with App Server
  support.

Install the locked environment and check the local project/HEG setup:

```console
uv sync --locked
uv run mforge doctor --heg-repo ../heg
```

`doctor` does not authenticate Codex and does not start a model turn. Ensure
the local Codex installation is authenticated before running an experiment.
Credentials must not be placed in `experiment.toml`; credential-bearing
configuration fields are rejected.

## Quick start

Create `experiment.toml` in the repository root:

```toml
schema_version = "mforge.experiment.v3"
exp_id = "heg-native-v3-search-01"
workspace = "./workspace"
kind = "heg"
preset = "native"

[run]
wall_seconds = 3600
output = "rich"
profiling_enabled = false
deep_profiling_enabled = false
turn_timeout_base_seconds = 120
max_total_tokens_per_hour = 1_000_000

[model]
provider = "codex"
name = "gpt-5.6-luna"
effort = "xhigh"
concurrency = 8
max_repairs = 1
auth_json = "~/.codex/auth.json"

[search]
population_size = 8
max_generations = "unbounded"
max_model_turns = "unbounded"
selection = "persistent-elite-weighted-diversity"

[evaluation]
graph_mode = "unrestricted_min_degree_3"
order_schedule = "static"
orders = [10, 12]
graph_seeds = [401, 402, 403, 404]
validation_graph_seeds = [1401, 1402, 1403, 1404]
policy_seeds = [
  4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008,
  4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016,
]
validation_policy_seeds = [
  14001, 14002, 14003, 14004, 14005, 14006, 14007, 14008,
  14009, 14010, 14011, 14012, 14013, 14014, 14015, 14016,
]
horizon = 32
baselines = [
  "add-low-local-cycle-risk",
  "remove-low-bridge-risk",
  "random-valid",
  "degree-fanout",
]
replay = true

[resources]
workers = 8
thread_count = 8

[native_v3]
provider_batch_size = 4
candidate_queue_capacity = 40
evaluation_queue_capacity = 64
target_evaluation_backlog = 32
candidate_shard_size = 1
auxiliary_shard_size = 1
witness_cap = 64
```

Check the resolved target without contacting a model:

```console
uv run mforge experiment status --config experiment.toml
```

Start or resume the experiment:

```console
uv run mforge experiment run --config experiment.toml
```

This command starts Codex App Server model turns and consumes the configured
model account's tokens. For machine-readable progress and results:

```console
uv run mforge experiment run --config experiment.toml --json
uv run mforge experiment status --config experiment.toml --json
```

## Configuration

`experiment.toml` is authoritative. Unknown and legacy keys are rejected
instead of being silently ignored.

| Section | Purpose |
| --- | --- |
| top level | Experiment identity, workspace, native kind, and preset |
| `[run]` | Per-invocation wall budget, rolling token cap, output mode, profiling, and provider-call timeout |
| `[model]` | Codex model, reasoning effort, concurrency, repair limit, and explicitly authorized `auth.json` path |
| `[search]` | Population, global generation/turn limits, and parent selection |
| `[evaluation]` | Development/validation panels, HEG orders, seeds, horizon, baselines, and replay |
| `[resources]` | Resource reservation (`workers`) and native evaluation pool (`thread_count`) |
| `[native_v3]` | Provider batching, bounded queue targets, shard sizes, and witness cap |

Important behavior:

- Native v3 requires `model.auth_json`. The file is copied into each private
  App Server capsule; the user's Codex config, skills, plugins, and other home
  contents are not copied. Authentication is checked before scheduler work.

- The production open-ended HEG configuration uses `population_size = 8`,
  matching the eight differentiated mutation briefs.
- `max_generations` and `max_model_turns` are required. Each accepts a positive
  integer or the exact string `"unbounded"`.
- Model-turn accounting remains cumulative when the limit is unbounded.
- `order_schedule = "static"` requires an explicit `orders` array.
- `order_schedule = "adaptive"` instead requires `min_order`, `max_order`, and
  `orders_per_generation`. The first generation uses the lowest eligible
  orders. Each later generation expands the eligible prefix by
  `orders_per_generation` and evaluates evenly spaced orders from the minimum
  through the current frontier. The schedule stays fixed after reaching
  `max_order`; `cubic_first` skips odd orders.
- `selection = "persistent-elite-weighted-diversity"` is the locked Native v3
  selection profile. It retains up to four top programs that completed the
  sealed validation protocol and assigns at most two parent references to
  each new slot in deterministic round-robin order.
- `model.effort` may be changed between resumable sessions without creating a
  new experiment identity; the next provider turn uses the current value.
- `max_total_tokens_per_hour` is optional. It accepts a positive integer or
  `"unbounded"` and applies to the canonical `totalTokens` charged during the
  rolling previous 60 minutes.
- Reaching the hourly token cap ends the current session as resumable
  `IDLE` with `stop_reason=hourly_token_limit`; no new model turn starts until
  enough prior usage leaves the rolling window.
- `wall_seconds` applies to one invocation. Reaching it pauses the experiment
  in resumable `idle` with `stop_reason=session_wall_seconds`.
- Uncharged provider infrastructure failures are retried on the same
  idempotent request, including repair turns, with a bounded exponential
  backoff. They do not consume the cumulative model-turn budget.
- The effective model-turn timeout is
  `turn_timeout_base_seconds`. Provider concurrency does not silently multiply
  the timeout of each independent call.
- `max_repairs` is persisted with every slot and remains enforced after resume.
- `output` is either `rich` or `json`; `--json` selects JSON output for the
  current command.
- Model, search, evaluation, and resource settings are locked when the
  workspace is created. Any scientific change requires a fresh workspace.

## Lifecycle and resume

The first run atomically creates `<workspace>/<exp_id>/`. Later invocations
with the same configuration resume from the latest durable checkpoint.

| State | Meaning |
| --- | --- |
| `not_created` | Configuration is valid, but no workspace exists yet |
| `running` | A session currently owns the experiment |
| `idle` | A session wall boundary paused the run; running again resumes work |
| `paused` | Verification or an administrative policy requires explicit continuation |
| `interrupted` | The process was interrupted or its owner died; running again resumes |
| `exhausted` | An explicitly finite generation or model-turn range was consumed |
| `failed` | A non-recoverable contract, verification, or orchestration failure |
| `completed` | A counterexample was certified or the operator used `stop --final` |

`Ctrl-C` records an interrupted, resumable session. A subsequent
`experiment run` recovers completed turn artifacts and continues pending work
without replacing already accepted model turns. Running an already completed
experiment creates a zero-work `already_completed` session and does not call
the provider.

To keep one process running across repeated wall-budget sessions, use:

```console
uv run mforge experiment run --dashboard --until-complete
```

The loop only continues after a normal `session_wall_seconds` boundary. A
persistent provider infrastructure failure remains visible as a resumable
failure requiring inspection.

An ordinary `q` or `Ctrl-C` remains resumable. To make an explicit terminal
operator decision:

```console
uv run mforge experiment stop --final
```

Use the read-only status command at any time:

```console
uv run mforge experiment status
```

It reports progress, resumability, accepted/evaluated candidates, the current
winner and canonical program identity, model-turn and token totals, charged
failed turns, checkpoint identity, artifact locations, stop reason, and the
latest error.
It never calls the provider, scorer, oracle, or evaluator.

## Workspace and results

An experiment retains its full operational and scientific record:

```text
workspace/<exp_id>/
├── experiment.toml
├── experiment.lock.json.gz
├── state.sqlite3
├── checkpoints/
│   └── checkpoint-*.json.gz
└── artifacts/
    ├── experiment-manifest.json.gz
    ├── native-v3-state.sqlite3
    ├── provider-v3/
    │   └── epoch-*/epoch-*_provider-*/
    ├── counterexamples-v3/
    ├── sessions/
    │   └── session-NNNNNN/
    │       ├── summary.json.gz
    │       ├── events.jsonl.gz
    │       ├── stdout.log
    │       └── stderr.log
    └── reports/
```

Each provider call keeps the exact frozen request, raw response, independently
validated slot entries, canonical ASTs and hashes, diagnostics, transport
metadata, and token usage. Native v3 semantic state is separated from
observational timing and worker telemetry so scheduling order does not change
replay identity.

There is no separate public report command. `experiment status` is the
supported summary view; session summaries, canonical program records, score
evidence, provider artifacts, and semantic records provide the detailed
evidence.

`state.sqlite3` contains orchestration, idempotency, token accounting, and
compact dashboard projections. `native-v3-state.sqlite3` contains canonical
semantic evidence and separate observational telemetry through one bounded
writer. Runtime scheduling order is excluded from its semantic checkpoint
identity.

## Native v3 program boundary

The normative program, context, selector, action, and execution contracts live
under [`configs/native`](configs/native/). Policies cannot import code, call
host functions, inspect raw labels as strategy features, or bypass selector,
action, gross-work, and net-diff budgets. Selectors see only the current
private overlay; connectivity and minimum degree are final-emit invariants.

The provider returns `program_json_raw`. The host parses and validates it,
creates canonical JSON, and derives `program_hash`; lineage remains
host-owned. HEG owns authoritative graph validation, bounded score evidence,
and primary exact verification. An apparent zero is only a durable submission
to the bounded dual-verifier supervisor.

See [Native v3 semantics](configs/native/native-v3-semantics.md) and
[Codex App Server integration boundary](docs/APP_SERVER_INTEGRATION.md).

## Output and profiling

The default Rich interface displays generation/slot progress, active model
turns, evaluation throughput, CPU/profile summaries, candidates, token usage,
and recent activity. JSON mode emits structured events suitable for external
monitoring.

Profiling can be enabled in `[run]` or overridden for one invocation:

```console
uv run mforge experiment run --profile
uv run mforge experiment run --no-profile
```

Deep profiling is independently controlled by
`run.deep_profiling_enabled`. It adds measurement overhead and should be used
for diagnosis, not throughput comparisons.

The event fields and lifecycle are documented in
[Event schema](docs/EVENT_SCHEMA.md).

## Development

Run the local verification suite with:

```console
uv run pytest
uv run ruff check .
uv run mypy
uv run python scripts/render_stage3_prompts.py --check
```

Use `uv`, never `pip`, for dependency management. Keep the sibling HEG
repository read-only.

## Documentation

- [Native experiment workflow](docs/EXPERIMENT_WORKFLOW.md)
- [Event schema](docs/EVENT_SCHEMA.md)
- [Codex App Server integration](docs/APP_SERVER_INTEGRATION.md)
- [Generated Python security](docs/GENERATED_PYTHON_SECURITY.md)
- [Architecture](docs/ARCHITECTURE.md)

The milestone plan and retained Stage 1–4 reports are historical research
evidence. They are available under [`docs/`](docs/) and
[`docs/reports/`](docs/reports/) but are intentionally not part of the primary
product workflow.

## License

CC0-1.0.
