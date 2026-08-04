# Mutation Forge Lab

Mutation Forge Lab searches for deterministic Python policies that rank legal
graph mutations for HEG. It uses the local Codex App Server to generate policy
candidates, validates and behavior-probes them in a bounded sandbox, evaluates
them against HEG and fixed baselines, and evolves the best candidates across
multiple generations.

The project is a standalone, auditable research application. It treats the
sibling HEG repository as a read-only mathematical backend and never modifies
HEG. Generated policies implement only:

```python
def priority(ctx, proposal):
    ...
```

They rank host-generated legal proposals; they do not generate graph rewrites,
score hidden alternatives, or replace HEG's validation and verification.

## How it works

For every experiment, Mutation Forge Lab:

1. creates an immutable experiment workspace and records repository provenance;
2. asks the configured local Codex model for eight differentiated ranking
   policies;
3. validates the structured response, Python AST, declared fields, bounded
   runtime contract, and proposal-dependent behavior;
4. repairs eligible invalid responses within the configured repair limit;
5. evaluates accepted policies and the random/structural baselines under
   matched HEG budgets;
6. archives candidates, selects parents with the configured search strategy,
   and continues until a session boundary or terminal outcome;
7. retains prompts, raw responses, transport diagnostics, token usage,
   evaluations, checkpoints, and winner information.

The installed CLI intentionally exposes only the current native product:

```text
mforge doctor
mforge experiment run
mforge experiment status
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
schema_version = "mforge.experiment.v2"
exp_id = "heg-ranker-search-01"
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

[search]
population_size = 8
max_generations = "unbounded"
max_model_turns = "unbounded"
selection = "persistent-elite-weighted-diversity"

[evaluation]
graph_mode = "unrestricted_min_degree_3"
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
| `[run]` | Per-invocation wall budget, rolling token cap, output mode, profiling, and timeout base |
| `[model]` | Codex model, reasoning effort, model concurrency, and repair limit |
| `[search]` | Population, global generation/turn limits, and parent selection |
| `[evaluation]` | HEG orders, seeds, horizon, proposal pools, baselines, and replay |
| `[resources]` | Resource reservation (`workers`) and native evaluation pool (`thread_count`) |

Important behavior:

- The production open-ended HEG configuration uses `population_size = 8`,
  matching the eight differentiated mutation briefs.
- `max_generations` and `max_model_turns` are required. Each accepts a positive
  integer or the exact string `"unbounded"`.
- Model-turn accounting remains cumulative when the limit is unbounded.
- `selection = "elite-diversity"` keeps the previous generation's best and
  fills the population with distinct policies. The recommended
  `selection = "persistent-elite-weighted-diversity"` allows repeated parents;
  for a population of eight it assigns three slots to the all-time best, two
  to the current-generation best, two by objective-weighted allocation from
  the current top half, and one by AST diversity from that top half.
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
  `turn_timeout_base_seconds * (model.concurrency + 1)`.
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
winner and source path, model-turn and token totals, charged failed turns,
checkpoint identity, artifact locations, stop reason, and the latest error.
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
    ├── native-generation-checkpoint.json.gz
    ├── sessions/
    │   └── session-NNNNNN/
    │       ├── session.json.gz
    │       ├── summary.json.gz
    │       ├── events.jsonl
    │       ├── stdout.log
    │       └── stderr.log
    ├── generations/
    │   └── generation-NNNN/slot-NN/
    │       ├── initial/
    │       ├── repair-NN/
    │       └── retry-NN/
    ├── archive/
    ├── evaluations/
    │   ├── development/
    │   └── replay/
    └── reports/
```

Each model turn keeps separate evidence for:

- the exact model-facing request and system prompt;
- the raw and parsed response;
- JSON/schema and App Server transport diagnostics;
- AST validation and source-derived `used_fields`;
- behavior-probe output and worker telemetry;
- provider identifiers, event/RPC streams, and exact token usage.

There is no separate public report command. `experiment status` is the
supported summary view; `summary.json.gz`, archived source files, evaluation
records, and turn artifacts provide the detailed evidence.

## Generated-policy boundary

Native policies receive the public Stage 2B context/proposal contracts:

- [`stage2b-context.schema.json`](configs/schemas/stage2b-context.schema.json)
- [`stage2b-proposal.schema.json`](configs/schemas/stage2b-proposal.schema.json)

The host enforces the generated-policy response schema and a restricted Python
AST. Policies cannot use imports, attributes, method calls, comprehensions,
generator expressions, randomness, hidden scores, proposal IDs, schema
versions, or provenance-only ranking fields. A valid policy must use at least
one proposal-specific structural signal and must distinguish proposals in the
scientific behavior probe.

HEG owns graph construction, legal proposal generation, rewrite application,
scoring, and verification. Only the selected proposal is authoritatively
evaluated.

See [Generated Python security boundary](docs/GENERATED_PYTHON_SECURITY.md)
and [Codex App Server integration boundary](docs/APP_SERVER_INTEGRATION.md)
for the detailed isolation and authority contracts.

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
