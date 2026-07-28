# Mutation Forge Lab
## Staged implementation specification and Codex execution prompt

---

# 1. Codex execution instruction

Build a separate research microproject named **Mutation Forge Lab**. Its purpose is to determine whether constrained Python mutation policies generated and iteratively improved by an LLM can outperform fixed graph-mutation baselines derived from the sibling HEG project.

The sibling HEG repository is available locally, normally at `../heg`. Treat it as read-only. Inspect its current `AGENTS.md`, implementation, tests, scoring semantics, graph representation, mutation operators, and exact verification interfaces before writing adapters. Record the exact HEG commit used by every experiment.

A local skill describing communication with the Codex App Server is available in the environment. Discover and read that skill before implementing the LLM transport adapter. Do not guess or hard-code an undocumented app-server protocol. The first implementation milestone must not require live model access.

## Mandatory first-pass boundary

1. Create the complete architecture and milestone documentation described below.
2. Implement **Stage 1 only**.
3. Do not implement generated-code execution, LLM calls, evolutionary program search, or HEG production integration in the first pass.
4. Do not modify `../heg`.
5. Finish Stage 1 with tests, a deterministic smoke benchmark, a machine-readable run artifact, and a clean commit.
6. Report the commit SHA, test commands, benchmark commands, results, limitations, and exact Stage 2 prerequisites.

All repository files, code, comments, documentation, prompts, schemas, CLI text, and logs must be in English.

---

# 2. Research question

The project must answer one narrow question:

> Can an LLM-generated, resource-bounded Python policy for selecting or constructing legal graph rewrites achieve better held-out search performance than fixed hand-written mutation strategies under an equal evaluation budget?

The project is a demonstrator, not a production subsystem and not an attempt to claim a proof or disproof of the Erdős–Gyárfás conjecture.

## Primary hypotheses

- **H1 — Program-search utility:** generated Python ranking policies can outperform random selection from the same legal proposal pool.
- **H2 — Baseline utility:** generated policies can outperform the fixed HEG mutation baselines under matched graph, seed, scorer, and evaluation budgets.
- **H3 — Generalization:** improvements persist on unseen graph seeds, random vertex relabelings, and at least one unseen graph order.
- **H4 — Safe execution:** generated code can be evaluated with bounded CPU, wall time, memory, output size, and graph-observation cost without weakening graph or verifier invariants.
- **H5 — External memory:** a durable archive using source hashes, normalized AST hashes, behavior signatures, lineage, and measured results is sufficient; the complete search history must not be inserted into every LLM context.

---

# 3. Explicit non-goals

The microproject must not initially:

- modify the HEG production repository;
- replace HEG's exact verifier;
- claim that a heuristic zero is a counterexample;
- let generated code read or write arbitrary files;
- let generated code use network, subprocess, imports, reflection, or native extensions;
- let generated code alter the scorer, witness cap, verification rules, campaign state, or persistence database;
- let generated code create multiple graphs, merge graphs, add or remove vertices, or change the configured graph order;
- build a full web dashboard in the first stages;
- build distributed execution, multi-host scheduling, or a second full Research Director;
- store the entire program history in an LLM prompt;
- integrate a discovered policy into live HEG search before the final benchmark produces an explicit GO decision.

---

# 4. Design principle

Use **real Python syntax with narrow authority**.

The generated program may use ordinary local Python constructs such as:

- `if` / `elif` / `else`;
- bounded `for` loops;
- local variables;
- arithmetic and comparisons;
- indexing and iteration over bounded immutable inputs;
- selected safe built-ins;
- calls to a reviewed, budgeted mutation API in later stages.

However, the generated program must never own the graph, scorer, verifier, filesystem, process, database, or network. It may only return a proposed ranking value or a declarative `RewritePlan`. The host validates and executes all graph changes.

The core authority split is:

```text
Generated policy:
    proposes or ranks a rewrite

Host runtime:
    generates bounded observations
    validates syntax and resources
    validates every rewrite
    applies the rewrite
    scores the resulting graph
    persists evidence
    invokes exact verification when required
```

---

# 5. Why this is a separate repository

The demonstrator must isolate the scientific question from HEG's production concerns. It should be possible to discard or redesign the microproject without migrations or changes to HEG campaign state.

The separate repository should still reuse HEG through a narrow adapter so that the benchmark is aligned with the actual project:

- HEG graph representation where practical;
- connected cubic seed generation;
- current forbidden-cycle score semantics;
- current baseline mutation operators;
- exact candidate validation and verification entry points;
- graph6 serialization and stable hashes.

Do not copy large parts of HEG blindly. Prefer a read-only adapter. Where direct reuse is impractical, implement a local interface and add parity tests against the sibling repository.

---

# 6. Initial technical stack

- Python 3.12 or newer.
- `uv` for environment and lock management.
- `rich` for the human live console.
- standard-library `sqlite3` for the durable program and experiment archive.
- `pytest` for tests.
- `ruff` and `mypy` for static checks.
- TOML configuration.
- JSON Lines for machine-readable live output and durable event logs.
- No web framework in Stage 1.

Keep dependencies minimal. Generated programs must not inherit access to third-party packages.

---

# 7. Repository structure

Create this target structure, allowing small justified adjustments:

```text
mutation-forge-lab/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── configs/
│   ├── stage1-smoke.toml
│   ├── stage1-baseline.toml
│   └── schemas/
├── docs/
│   ├── MASTER_PLAN.md
│   ├── ARCHITECTURE.md
│   ├── MILESTONES.md
│   ├── SCORE_AND_FITNESS.md
│   ├── GENERATED_PYTHON_SECURITY.md
│   ├── EVENT_SCHEMA.md
│   ├── APP_SERVER_INTEGRATION.md
│   └── REPORTING.md
├── prompts/
│   ├── ranker_v1_system.md
│   ├── ranker_v1_request.md
│   └── ranker_v1_output_schema.json
├── src/mutation_forge/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── models.py
│   ├── events.py
│   ├── run_store.py
│   ├── backends/
│   │   ├── base.py
│   │   ├── toy.py
│   │   └── heg.py
│   ├── output/
│   │   ├── base.py
│   │   ├── rich_live.py
│   │   └── jsonl.py
│   ├── proposals/
│   │   ├── base.py
│   │   ├── two_switch.py
│   │   └── k_switch.py
│   ├── policies/
│   │   ├── base.py
│   │   └── baselines.py
│   ├── evaluation/
│   │   ├── episode.py
│   │   ├── fitness.py
│   │   └── benchmark.py
│   ├── sandbox/              # created in Stage 2
│   ├── llm/                  # created in Stage 3
│   ├── archive/              # expanded in Stage 4
│   └── evolution/            # created in Stage 4
└── tests/
    ├── unit/
    ├── integration/
    ├── parity/
    └── fixtures/
```

Stage 1 may create empty documented package boundaries for later stages, but must not include fake implementations that imply unfinished features work.

---

# 8. Core interfaces

Define stable typed interfaces before implementing the stages.

## 8.1 Graph backend

```python
class GraphBackend(Protocol):
    backend_id: str

    def generate_seed(self, *, order: int, seed: int) -> GraphState: ...
    def validate(self, graph: GraphState) -> GraphValidation: ...
    def score(self, graph: GraphState, *, witness_cap: int) -> GraphScore: ...
    def exact_verify(self, graph: GraphState) -> ExactVerification: ...
    def canonical_hash(self, graph: GraphState) -> str: ...
    def serialize_graph6(self, graph: GraphState) -> str: ...
    def apply_rewrite(
        self, graph: GraphState, rewrite: RewritePlan
    ) -> GraphState: ...
```

The HEG adapter should use the sibling project's authoritative implementation where possible.

## 8.2 Rewrite plan

```python
@dataclass(frozen=True, slots=True)
class RewritePlan:
    removed_edges: tuple[Edge, ...]
    added_edges: tuple[Edge, ...]
    operator_family: str
    metadata: Mapping[str, JsonValue]
```

A `RewritePlan` is declarative. Constructing one does not mutate a graph.

## 8.3 Host validation invariants

Every applied rewrite must satisfy:

- input and output contain exactly one graph;
- graph order is unchanged;
- removed edges exist;
- added edges do not already exist after removals;
- no loops;
- no duplicate edges;
- graph remains simple and undirected;
- graph remains connected;
- configured degree constraints remain satisfied;
- in connected cubic mode, every vertex remains degree 3;
- rewrite size is within the configured bound;
- proposal and result payload sizes are within limits.

The host must reject invalid plans before scoring them.

## 8.4 Graph score

Represent the HEG heuristic score explicitly:

```python
@dataclass(frozen=True, slots=True)
class GraphScore:
    valid: bool
    capped_cycle_counts: tuple[tuple[int, int], ...]
    total_capped_witnesses: int
    weighted_penalty: int
    complete: bool
    ordering_key: tuple[int, ...]
```

For the HEG backend, preserve current HEG ordering semantics rather than inventing a silently different score.

A heuristic score of zero is only a trigger for exact verification. It is never itself a proof.

---

# 9. Graph score and program fitness

## 9.1 Graph-level search signal

For each forbidden length `l` in `{4, 8, 16, ...} <= order`, count witnesses up to a configured `witness_cap`:

```text
c_l(G) = min(actual cycle count at length l, witness_cap)
```

The search objective is to reduce the capped witness vector toward zero. Short cycles may receive larger secondary weights, consistent with the HEG backend.

A single found cycle is enough to reject a candidate as a counterexample, but one-bit existence is too sparse to guide mutation search. Therefore the heuristic uses bounded counts.

## 9.2 Episode-level evaluation

Do not score a program from one mutation. Evaluate it across fixed episodes.

Each episode has:

- one immutable initial graph;
- one policy seed;
- one fixed search-controller configuration;
- one maximum graph-evaluation count;
- one maximum wall time;
- the same scorer and witness cap for all policies;
- the same proposal-generation budget for all ranking policies.

Record:

- initial graph score;
- best-so-far graph score;
- final graph score;
- best-so-far curve;
- normalized area under the best-so-far curve;
- time and evaluations to first improvement;
- exact-zero candidates submitted to verification;
- legal proposal rate;
- no-op rate;
- duplicate graph rate;
- policy call latency;
- timeout, crash, and resource-failure rates.

## 9.3 Program fitness key

Do not collapse all information prematurely. Persist the complete metric vector and use a deterministic lexicographic fitness key.

Recommended initial key to minimize:

```text
(
    -exact_verified_count,
    failure_episode_count,
    median_normalized_best_total_witnesses,
    median_normalized_best_weighted_penalty,
    median_normalized_best_so_far_auc,
    timeout_rate,
    illegal_or_noop_rate,
    median_policy_call_ms,
    normalized_ast_node_count,
)
```

Definitions and normalization must be frozen in `docs/SCORE_AND_FITNESS.md` before the first comparative experiment.

Timeouts and resource violations are not represented by an arbitrary numeric score that might accidentally look competitive. They increment `failure_episode_count` and receive the worst fitness ordering.

---

# 10. Fair comparison rules

All compared policies must use:

- the same initial graph manifest;
- identical graph seeds;
- identical policy seeds where applicable;
- the same maximum graph evaluations;
- the same scorer and witness cap;
- the same proposal-pool size for rankers;
- the same fixed acceptance/search controller;
- the same timeout and memory limits;
- no access to held-out test results during generation.

For ranking-policy experiments, proposal generation must not secretly compute the full post-mutation HEG score for every proposal. That would make the ranker trivial and distort compute accounting. Proposal features may use bounded witness samples and inexpensive structural features. The authoritative graph score is computed after selecting and applying the proposal.

An optional full-score best-of-K oracle may be reported only as a diagnostic upper bound, with its additional compute clearly accounted for. It is not a fair baseline.

---

# 11. Initial graph datasets

Persist immutable dataset manifests containing graph6, graph hash, order, seed, generator version, HEG commit, and split.

## Stage 1 smoke split

Use a very small deterministic connected-cubic set sufficient to validate infrastructure, for example:

- order 30;
- 4 graph seeds;
- 2 policy seeds per graph;
- 500 to 2,000 graph evaluations per episode.

The complete smoke run should finish in minutes, not hours.

## Pilot splits for later stages

Freeze separate manifests before program search:

- **train:** unseen connected-cubic graphs at order 30;
- **validation:** different seeds at order 30 plus random relabelings;
- **cross-order validation:** an even order above the `C32` threshold, initially order 34 or another measured practical value;
- **held-out test:** separate seeds and at least one order not present in training;
- **optional stress:** order 62 after the evaluator is fast and stable.

Never include held-out test graphs or their policy results in LLM prompts.

Random relabeling tests are mandatory because absolute vertex identifiers are not mathematical structure.

---

# 12. Baseline policies

Implement and label at least these baselines:

1. **HEG uniform two-edge switch** — use the current HEG implementation through the adapter.
2. **HEG forbidden-cycle-break switch** — use the current HEG implementation through the adapter.
3. **Random legal proposal ranker** — uniform selection from the same proposal pool used by generated rankers.
4. **Hand-written structural ranker** — a simple reviewed heuristic based on bounded witness-load and distance features.
5. **Random legal k-switch** — added in Stage 2 for `k` in a bounded configurable set.

Do not claim generated-code improvement unless it beats the strongest relevant baseline on held-out data.

---

# 13. Proposal-generation operations

The host, not generated code, initially constructs legal proposal candidates.

## Stage 1

- current HEG two-edge switches;
- current HEG forbidden-cycle-targeted two-edge switches.

## Stage 2 proposal pool

Add generalized degree-preserving `k`-switch proposals for configured `k`, initially `2 <= k <= 4`:

1. choose `k` pairwise vertex-disjoint existing edges;
2. remove those edges conceptually, producing `2k` endpoint stubs;
3. sample bounded perfect matchings of the stubs;
4. reject original pairings, loops, duplicate edges, existing edges, and disconnected results;
5. return only host-validated legal `RewritePlan` values.

Add reviewed edge-selection strategies:

- uniform random;
- anchored in a sampled forbidden cycle;
- high sampled witness-load;
- remote from an anchor edge;
- pairwise distant/disjoint;
- mixed exploit/explore selection.

Every proposal carries immutable, JSON-serializable features. Initial proposed features include:

```text
proposal_id
k
operator_family
selector_tags
anchor_forbidden_length
broken_sampled_witnesses_by_length
removed_edge_load_sum_by_length
removed_edge_load_max_by_length
minimum_distance_between_removed_edges
mean_distance_between_removed_edges
minimum_preexisting_distance_for_new_edges
mean_preexisting_distance_for_new_edges
local_triangle_risk
local_c4_risk
reconnection_span
```

All expensive feature queries require explicit budgets and caching.

---

# 14. Generated Python interface

## Stage 2 and Stage 3 interface: ranker only

The generated code is exactly one function:

```python
def priority(ctx, proposal):
    """Return a finite float. Larger values are preferred."""
    ...
```

The host generates a bounded set of legal proposals. The function ranks one proposal using immutable plain-data inputs. It cannot mutate the graph.

### Context fields

Freeze a versioned context schema. Initial fields should include:

```text
schema_version
order
forbidden_lengths
capped_cycle_counts
weighted_penalty
step
remaining_steps
stagnation
recent_best_improvement
recent_acceptance_rate
recent_duplicate_rate
```

### Proposal fields

Use the proposal feature schema from Section 13.

## Later interface: full mutation proposer

Only after the ranker demonstrator passes its gate, add:

```python
def propose_mutation(ctx, api, rng):
    """Return one declarative RewritePlan or a reviewed no-op."""
    ...
```

The reviewed API may expose bounded operations such as:

```text
sample_forbidden_cycle(length)
cycle_edges(cycle)
edge_witness_load(edge, length)
vertex_witness_load(vertex, length)
distance_between_edges(edge_a, edge_b)
select_remote_edges(...)
generate_k_switch_proposals(k, ...)
reconnection_candidates(removed_edges, limit)
make_plan(removed_edges, added_edges, metadata)
noop(reason)
```

Every call consumes an observation or proposal budget. The API must never expose the live scorer or verifier implementation.

---

# 15. Generated-code safety model

AST validation alone is not a sufficient execution boundary. Use defense in depth.

## 15.1 Static AST contract

Allow exactly one top-level function with the exact required name and parameters.

Initially allow:

- assignments to local variables;
- `if` statements;
- bounded `for` loops over provided collections or statically bounded `range`;
- arithmetic, comparisons, Boolean operations;
- indexing and slicing of bounded inputs;
- calls to an explicit allowlist of safe built-ins;
- return of one finite numeric value.

Initially reject:

- all imports;
- `while`;
- recursion;
- nested function or class definitions;
- decorators;
- generators and async code;
- `lambda` unless explicitly justified later;
- `try` / `except` in the first version;
- `with`;
- `yield`;
- global or nonlocal declarations;
- names or attributes beginning with `_`;
- reflection and dynamic code execution;
- file, process, environment, network, terminal, or database access;
- `eval`, `exec`, `compile`, `open`, `input`, `print`, `globals`, `locals`, `vars`, `dir`, `getattr`, `setattr`, `delattr`, `__import__`;
- unbounded source size or AST node count.

Recommended configurable defaults:

```text
max_source_bytes = 12 KiB
max_ast_nodes = 500
max_static_loop_bound = 256
```

## 15.2 Isolated process

Compile and execute candidate code in a dedicated spawned subprocess, never in the coordinator process.

Mandatory controls:

- sanitized environment;
- isolated temporary working directory;
- no inherited standard input;
- bounded request and response payloads;
- hard wall timeout controlled by parent;
- Linux `RLIMIT_CPU`;
- Linux `RLIMIT_AS`;
- `RLIMIT_FSIZE` set to zero or a minimal diagnostic allowance;
- small `RLIMIT_NOFILE`;
- bounded process count;
- captured and capped stdout/stderr even though printing is statically forbidden;
- kill and reap the entire worker on timeout, crash, or protocol violation.

Recommended initial ranker limits:

```text
worker_address_space = 128 MiB
per_call_wall_timeout = 25 ms
per_program_evaluation_wall_limit = 60 s for smoke
max_request_bytes = 64 KiB
max_response_bytes = 16 KiB
max_captured_output = 64 KiB
```

Make all limits configurable and record their values in run metadata.

A persistent worker per candidate may be used to avoid process startup on every rank call, but the parent must retain kill authority and enforce both per-call and total limits.

## 15.3 Host result validation

For a ranker:

- result must be a finite `int` or `float`;
- NaN and infinity are invalid;
- exceptions, timeout, protocol mismatch, and oversized output fail the candidate.

For a later proposer:

- parse only the `RewritePlan` schema;
- validate all graph invariants before application;
- never treat timeout, error, or unknown as a valid no-op or successful zero.

## 15.4 Threat-model statement

Document that the initial sandbox is a local research boundary, not a multi-tenant security product. Add an optional stronger Linux executor using a reviewed sandbox such as a container or namespace tool only after the process-based executor is complete and tested.

---

# 16. LLM integration architecture

Define a transport-independent interface:

```python
class ProgramGenerator(Protocol):
    def generate(
        self, request: ProgramGenerationRequest
    ) -> ProgramGenerationResult: ...
```

Implementations:

- `FixtureProgramGenerator` — deterministic, no model, mandatory for unit and integration tests;
- `FileProgramGenerator` — reads a previously persisted response;
- `CodexAppServerProgramGenerator` — Stage 3, implemented using the locally installed app-server skill.

The evaluator and archive must not depend directly on the app-server implementation.

Persist:

- exact request artifact;
- exact structured response artifact;
- hashes;
- model and effort metadata;
- usage metadata where available;
- latency;
- validation outcome;
- generated source and summaries.

Never persist credentials or private app-server home data.

---

# 17. LLM generation prompt

Create versioned prompt files. The following is the required semantic content for `ranker_v1`.

## 17.1 System prompt

```text
You are a program-synthesis component in a controlled graph-search experiment.

Your task is to produce exactly one Python function named `priority(ctx, proposal)`.
The function returns a finite numeric value; larger values are preferred.
It ranks one already-legal graph rewrite proposal. It does not receive authority to mutate a graph, call a scorer, access files, use the network, start processes, import modules, or inspect system state.

Use only the documented immutable context fields, proposal fields, ordinary local arithmetic and control flow, and the allowlisted built-ins. Do not use absolute vertex identifiers as semantic information. Do not attempt to bypass limits or infer hidden test data.

The objective is to select proposals that help a fixed search controller reduce capped forbidden-cycle witness counts for lengths 4, 8, 16, and further powers of two allowed by the graph order. Shorter forbidden cycles have higher secondary penalty, but total capped witness count remains primary according to the supplied evaluator contract.

Return one JSON object matching the provided schema. Do not return Markdown or explanatory text outside that JSON object.
```

## 17.2 Request template

```text
Generation mode: {{generation_mode}}
Candidate schema version: {{candidate_schema_version}}
Context schema: {{context_schema}}
Proposal schema: {{proposal_schema}}
Allowed built-ins: {{allowed_builtins}}
Static restrictions: {{static_restrictions}}
Resource limits: {{resource_limits}}
Fitness definition: {{fitness_definition}}

Parent programs and measured evidence:
{{selected_parent_programs}}

Archive summary:
{{bounded_archive_summary}}

Requested task:
{{task_instruction}}

Produce a candidate that is executable under the stated restrictions and that is meaningfully testable. Do not claim improvement; state a hypothesis that the evaluator will test.
```

Generation modes:

- `new_strategy`;
- `small_mutation`;
- `improve_family`;
- `crossover`;
- `simplify_without_regression`.

For `small_mutation`, instruct the model to preserve most of the parent and make one localized semantic change. The system still stores a complete new function, not a patch.

## 17.3 Structured output schema

```json
{
  "schema_version": "1.0",
  "function_source": "def priority(ctx, proposal):\n    ...",
  "strategy_summary": "Two or three concise sentences describing the strategy.",
  "change_summary": "A concise description of the semantic change from the parent, or 'new strategy'.",
  "hypothesis": "A falsifiable expectation about where this policy should help.",
  "expected_failure_modes": [
    "One concise expected failure mode"
  ]
}
```

The summaries are advisory metadata. Source, normalized AST, behavior, and measured evaluation remain authoritative.

---

# 18. Durable program memory

Use SQLite as the source of truth and files for immutable larger artifacts.

Minimum program record:

```text
program_id
schema_version
source_path
source_sha256
normalized_ast_sha256
ast_node_count
parent_program_ids
generation_mode
generation_index
island_id
strategy_summary
change_summary
hypothesis
expected_failure_modes
created_at
model_metadata
prompt_artifact_ref
response_artifact_ref
static_validation_status
sandbox_validation_status
archive_tier
```

Minimum evaluation record:

```text
evaluation_id
program_id
run_id
dataset_manifest_sha256
split
configuration_sha256
fitness_key
complete_metric_json
wall_seconds
cpu_seconds
peak_rss_bytes
policy_call_count
policy_timeout_count
invalid_result_count
legal_proposal_rate
duplicate_rate
created_at
```

## Duplicate detection hierarchy

1. exact source hash;
2. normalized AST hash after local-name and formatting normalization;
3. behavior signature on a fixed probe set;
4. optional semantic-code or summary similarity;
5. optional curator model only for ambiguous clustering.

No LLM curator may override source or behavior evidence.

## Archive tiers

- **hot:** current champions and diverse active parents;
- **warm:** specialists, prior champions, and useful alternate families;
- **cold:** all valid programs and their evidence, normally excluded from prompts.

Never delete a valid program merely because it is currently weak. Reduce its retrieval priority instead.

---

# 19. Behavior signatures

Evaluate each statically valid program on a fixed, versioned probe set and record a compact signature such as:

```text
selected proposal IDs
rank ordering over probe proposals
exceptions / timeout flags
finite-output flags
selected k distribution
operator-family distribution
resulting score deltas when applied
```

Use this to identify behaviorally redundant programs even when their ASTs differ.

Probe data must not include held-out benchmark graphs.

---

# 20. Live human output and machine-readable output

Choose **Rich + JSON Lines** as the initial interface. Do not build a live web UI in the first pass.

## 20.1 Default Rich mode

When stdout is an interactive terminal, show a continuously updating Rich display containing at least:

- run ID and stage;
- HEG commit and backend;
- elapsed time and remaining budget;
- dataset split and graph order;
- current baseline or program ID;
- completed episodes / total episodes;
- graph evaluations and evaluations per second;
- initial, current, and best score summaries;
- current champion and delta versus strongest baseline;
- legal proposal rate;
- timeout, crash, and invalid counts;
- policy-call median and p95 latency when applicable;
- LLM calls, latency, and token usage when applicable;
- archive size and deduplication counts in later stages;
- most recent event or failure.

Refresh at a bounded rate, initially 2 to 4 times per second, without materially affecting evaluation throughput.

## 20.2 `--json` mode

Every long-running command must support `--json` as an alias for `--output json`.

In JSON mode:

- stdout contains JSON Lines only;
- no ANSI escape sequences;
- no Rich tables, progress bars, or prose;
- one object per line;
- every event includes `schema_version`, `timestamp`, `run_id`, and `event_type`;
- fatal failures are also represented as a final JSON event;
- ordinary diagnostics go to durable artifacts; stderr is reserved for bootstrap failures that prevent JSON event initialization.

Required event types include:

```text
run_started
backend_ready
dataset_loaded
baseline_started
episode_started
episode_progress
episode_completed
program_generation_started
program_generated
static_validation_completed
sandbox_validation_completed
program_evaluation_started
program_evaluation_completed
champion_changed
checkpoint_written
run_completed
run_failed
```

## 20.3 Durable output

Every run writes:

```text
runs/<run_id>/
├── run_config.toml
├── run_manifest.json
├── events.jsonl
├── run_summary.json
├── environment.json
├── dataset_manifest.json
├── artifacts/
│   ├── programs/
│   ├── prompts/
│   ├── responses/
│   └── graphs/
└── mutation_forge.sqlite3 or a reference to the project archive
```

Rich and JSON modes must produce the same durable results and final summary hash for identical seeds and configuration.

## 20.4 Inspection commands

Provide machine-friendly inspection without rerunning experiments:

```text
mforge inspect RUN_DIR
mforge inspect RUN_DIR --json
mforge programs list --json
mforge compare RUN_A RUN_B --json
```

A static HTML report generated from persisted JSON may be added in Stage 6, but is not required for live monitoring.

---

# 21. Configuration contract

Use versioned TOML. A future full configuration should support:

```toml
schema_version = "1.0"

[run]
seed = 1234
wall_seconds = 300
output = "rich"
run_root = "./runs"

[heg]
repo = "../heg"

[dataset]
orders = [30]
graph_seeds = [101, 102, 103, 104]
policy_seeds = [1, 2]
split = "smoke"

[score]
witness_cap = 64

[search]
controller = "fixed_ils_tabu"
evaluations_per_episode = 1000
proposal_pool_size = 32

[proposals]
operator_families = ["heg_uniform_two_switch", "heg_forbidden_cycle_break"]
k_values = [2]

[sandbox]
max_source_bytes = 12288
max_ast_nodes = 500
worker_memory_bytes = 134217728
per_call_wall_ms = 25
program_wall_seconds = 60

[llm]
backend = "fixture"
prompt_version = "ranker_v1"

[archive]
database = "./mutation_forge.sqlite3"
```

Stage 1 may omit inactive sections from runtime use but must document their reserved meaning.

---

# 22. Milestones

Each stage is a complete, testable stopping point. Do not proceed automatically from one stage to the next.

## Stage 1 — Trusted harness, HEG adapter, baselines, and observability

### Goal

Establish a trustworthy experimental harness before any generated code or LLM integration.

### Implement

- project skeleton and documentation;
- typed configuration and models;
- event bus and durable JSONL event writer;
- Rich live output;
- `--json` output;
- SQLite run metadata store;
- toy backend for fast isolated tests;
- read-only HEG backend adapter;
- record HEG commit and environment metadata;
- immutable graph dataset manifests;
- current HEG uniform two-switch baseline;
- current HEG forbidden-cycle-break baseline;
- fixed episode runner and graph fitness collection;
- deterministic baseline comparison report;
- CLI commands:
  - `mforge doctor`;
  - `mforge dataset build`;
  - `mforge baseline run`;
  - `mforge inspect`;
  - `mforge compare`.

### Required tests

- configuration validation;
- event schema and JSON-only stdout;
- Rich and JSON output parity;
- deterministic rerun with identical seeds;
- graph manifest hashing;
- HEG import and commit detection;
- parity of graph validation and score payload against HEG;
- baseline operator result validity;
- no modification of sibling HEG;
- interrupted run leaves readable artifacts;
- wall-budget stop is bounded.

### Stage 1 acceptance criteria

1. `uv run mforge doctor --heg-repo ../heg` passes.
2. `uv run mforge baseline run --config configs/stage1-smoke.toml` displays a live Rich view and completes.
3. The same command with `--json` emits valid JSONL only.
4. Both modes produce the same final `run_summary.json` content hash for the same run seed and config, excluding timestamps and run IDs through a documented canonical comparison.
5. At least the two current HEG baselines run on the same graph manifest.
6. Every result graph passes the HEG structural validator.
7. Repeating the smoke run produces the same deterministic score trajectories under the same configuration.
8. `pytest`, `ruff`, and `mypy` pass.
9. The smoke benchmark finishes within a bounded, documented time on the local machine.
10. No files in `../heg` are modified.

### Stage 1 deliverable

A clean commit and `docs/reports/STAGE1_REPORT.md` containing:

- environment;
- HEG commit;
- commands;
- dataset manifest;
- baseline results;
- throughput;
- reproducibility evidence;
- limitations;
- Stage 2 readiness decision.

**Implement this stage now. Stop after it.**

---

## Stage 2 — Safe Python ranker runtime and generalized proposal pool

### Goal

Prove that a constrained Python ranker can be executed safely and can select among host-generated legal rewrites.

### Implement

- generalized legal `k`-switch proposal generator for `2 <= k <= 4`;
- proposal feature schema;
- exact ranker function template;
- AST allowlist validator;
- normalized AST hashing;
- isolated persistent policy worker;
- CPU, memory, wall-time, payload, and output limits;
- manual fixture rankers;
- random ranker and hand-written structural ranker;
- policy validation CLI;
- policy evaluation CLI;
- timeout and invalid-program penalties;
- behavior-signature probe set.

### Required adversarial fixtures

Include candidate sources attempting:

- import;
- file access;
- environment access;
- subprocess;
- network;
- dunder traversal;
- infinite loop;
- huge allocation;
- huge integer or output;
- recursion;
- NaN and infinity;
- exception;
- wrong function signature;
- multiple functions;
- hidden state or nondeterminism.

### Stage 2 acceptance criteria

1. All host-applied proposals preserve graph invariants.
2. Every forbidden AST fixture is rejected before execution.
3. Infinite or slow code is terminated within the configured bound.
4. Memory-abusive code cannot exceed the configured worker address-space limit without candidate failure.
5. A worker crash cannot terminate or corrupt the coordinator.
6. A valid manual ranker completes at least 10,000 bounded calls.
7. JSON and Rich modes remain equivalent.
8. Behavior signatures and AST hashes are stable across reruns.
9. The hand-written structural ranker beats random ranking on a toy benchmark designed for this test; this is an infrastructure assertion, not a HEG scientific claim.

### Exit artifact

`docs/reports/STAGE2_REPORT.md` and a GO/NO-GO decision for live LLM generation.

---

## Stage 3 — One-shot Codex App Server program generation

### Goal

Connect the controlled evaluator to the LLM without adding evolutionary memory yet.

### Implement

- `ProgramGenerator` protocol;
- deterministic fixture and replay providers;
- Codex App Server adapter using the local skill;
- versioned prompt and structured output schema;
- request/response artifact persistence;
- source extraction and validation;
- strategy and change summaries;
- one-shot batch generation;
- model latency and usage telemetry;
- CLI commands to generate, validate, evaluate, and replay candidates.

### Stage 3 acceptance criteria

1. All tests pass without model access using the fixture provider.
2. A recorded response can be replayed byte-for-byte without a model call.
3. One authorized live smoke request either produces a persisted candidate or fails closed with a precise contract error.
4. Generated source cannot bypass Stage 2 validation or sandboxing.
5. No credentials or private runtime data appear in run artifacts.
6. At least one valid generated candidate can be evaluated end-to-end; performance improvement is not required at this stage.

### Exit artifact

`docs/reports/STAGE3_REPORT.md`, including exact model request evidence and cost.

---

## Stage 4 — Evolutionary program search and external memory

### Goal

Implement the generator-of-generators loop around the safe ranker.

### Implement

- durable program archive;
- lineage;
- exact source and normalized AST deduplication;
- behavior signatures;
- hot/warm/cold tiers;
- multiple islands or niches;
- parent selection;
- generation modes:
  - new strategy;
  - small mutation;
  - family improvement;
  - crossover;
  - simplification;
- bounded context builder selecting only a few relevant programs;
- champion selection based on validation, not training alone;
- resumable evolution runs;
- archive inspection and lineage display in Rich and JSON modes;
- optional semantic curator only after AST and behavior checks.

### Stage 4 acceptance criteria

1. A fixture-provider evolution run of at least 25 generations is deterministic and resumable.
2. Exact and normalized-AST duplicates are detected.
3. Behaviorally redundant programs are clustered.
4. Cold programs remain recoverable but are absent from ordinary prompts.
5. Prompt size stays below a configured bound independent of total archive size.
6. A restarted run does not repeat already completed model requests or program evaluations.
7. A small live evolution run completes without corrupting the archive.
8. The system reports whether generated rankers improve validation fitness; improvement is measured, not assumed.

### Gate to Stage 5

Proceed only if at least one generated ranker demonstrates reproducible validation improvement over random ranking or reveals a clearly useful new strategy family.

---

## Stage 5 — Generated full mutation proposer

### Goal

Allow generated Python to construct a declarative rewrite using reviewed graph operations rather than merely ranking a host proposal pool.

### Implement

- versioned `MutationAPI` facade;
- strict per-call observation and proposal budgets;
- generated `propose_mutation(ctx, api, rng)` interface;
- safe random source supplied by host;
- bounded cycle sampling;
- bounded distance and witness-load queries;
- bounded `k`-switch construction;
- host validation of every returned plan;
- no-op reason schema;
- proposal behavior signatures;
- direct comparison with ranker-only policies.

### Stage 5 acceptance criteria

1. No generated program can apply a graph mutation directly.
2. No invalid rewrite reaches the scorer.
3. Observation and proposal budgets are enforced.
4. Timeouts and resource failures receive worst fitness and leave the run healthy.
5. Generated proposers are tested under random vertex relabeling.
6. The benchmark clearly separates policy runtime from graph-scoring runtime.
7. The full proposer either beats the ranker-only approach on validation or receives a documented NO-GO.

---

## Stage 6 — Pre-registered scientific benchmark and GO/NO-GO report

### Goal

Determine whether Mutation Forge merits integration work.

### Before running

Freeze and hash:

- train, validation, and test graph manifests;
- all program-search budgets;
- baseline set;
- score and fitness definitions;
- statistical analysis;
- runtime limits;
- GO/NO-GO thresholds.

### Required benchmark dimensions

- unseen graph seeds;
- random relabelings;
- at least one unseen graph order;
- multiple policy seeds;
- equal graph-evaluation budgets;
- wall-time and CPU-cost accounting;
- LLM cost accounting;
- ranker and full-proposer variants if both survived prior gates;
- ablation without summaries;
- ablation without islands;
- ablation without behavior deduplication;
- current HEG baselines.

### Provisional GO thresholds

Freeze these or explicitly revise them before seeing test results:

- at least 10% improvement in median normalized best-so-far AUC versus the strongest fixed baseline on held-out episodes;
- bootstrap confidence interval for the held-out improvement excludes zero;
- no material regression in the primary best-total-witness metric;
- generated-policy call overhead remains at or below 2x the fixed-policy selection overhead, excluding authoritative graph scoring;
- timeout rate <= 1%;
- host-applied graph validity = 100%;
- improvement remains under random vertex relabeling;
- no held-out leakage into prompts or parent selection.

These thresholds are research defaults, not mathematical claims.

### Exit artifact

`docs/reports/FINAL_BENCHMARK.md` with one explicit decision:

- `GO_FOR_HEG_SHADOW_INTEGRATION`, or
- `NO_GO`, or
- `INCONCLUSIVE_WITH_REQUIRED_NEXT_EXPERIMENT`.

---

## Stage 7 — Optional HEG shadow-integration package

### Goal

Prepare a reviewed integration artifact only after a Stage 6 GO.

### Implement

- versioned policy artifact bundle;
- source, AST hash, behavior signature, benchmark evidence, and limits;
- compatibility adapter matching HEG mutation interfaces;
- shadow mode that proposes mutations but does not affect HEG trajectories;
- parity and overhead measurement;
- fail-closed disable switch;
- separate GitHub issue and review before any production activation.

No automatic modification of HEG is permitted from this microproject.

---

# 23. Stage gates and stopping rules

- Stop after Stage 1 until the user reviews the baseline harness.
- Do not spend model tokens before Stage 2 safety and deterministic replay pass.
- Do not implement full mutation generation before ranker-only program search shows evidence of utility.
- Do not integrate into HEG before Stage 6 produces an explicit GO.
- A negative result is a valid project result and must be reported without hiding failed candidates.

---

# 24. Stage 1 CLI details

Implement these commands in the first pass:

```bash
uv run mforge doctor --heg-repo ../heg

uv run mforge dataset build \
  --config configs/stage1-smoke.toml

uv run mforge baseline run \
  --config configs/stage1-smoke.toml

uv run mforge baseline run \
  --config configs/stage1-smoke.toml \
  --json

uv run mforge inspect runs/<run-id>
uv run mforge inspect runs/<run-id> --json
uv run mforge compare runs/<run-a> runs/<run-b> --json
```

`doctor` must check:

- Python version;
- package installation;
- writable run directory;
- SQLite availability;
- Rich availability;
- sibling HEG location;
- HEG importability;
- HEG commit;
- expected graph, target, scorer, and mutation interfaces;
- no attempt to modify HEG;
- optional app-server skill discovery status, reported as informational only in Stage 1.

---

# 25. Stage 1 output example

The Rich view should resemble this information hierarchy, not necessarily this exact layout:

```text
Mutation Forge Lab — Stage 1 Baseline Run
Run: run-...          HEG commit: 35273bc...       Elapsed: 00:01:42
Dataset: smoke-...    Order: 30                    Episodes: 5 / 8

Baseline                         Best total   Best weighted   Eval/s   Status
HEG uniform two-switch           12           96              8,201    running
HEG forbidden-cycle-break        7            44              7,918    running

Current episode
Graph seed: 103    Policy seed: 2    Evaluations: 812 / 1000
Initial: total=64 weighted=512      Best: total=7 weighted=44

Last event: new best graph at evaluation 799
```

Equivalent JSONL events must contain the same facts.

---

# 26. Testing strategy

## Unit tests

- pure models and validation;
- config parsing;
- canonical JSON and hashes;
- event schema;
- output renderers;
- score and fitness normalization;
- proposal validation;
- later AST validation and sandbox protocol.

## Integration tests

- toy backend complete run;
- interrupted and resumed run artifacts;
- Rich/JSON canonical parity;
- SQLite persistence;
- later fixture LLM generation and replay.

## HEG parity tests

When `../heg` is available:

- seed graph generation;
- graph validation;
- graph6 round-trip;
- forbidden lengths;
- capped score payload;
- ordering key;
- baseline mutation output validity;
- exact verification status on controlled fixtures.

Tests must state whether they are mandatory or skipped when HEG is unavailable. The Stage 1 local acceptance run requires HEG and must not skip these tests.

## Performance tests

Keep separate from unit tests. Record:

- graph evaluations per second;
- output overhead Rich vs JSON vs no-live;
- SQLite/event overhead;
- policy-call overhead in later stages;
- scorer time separately from policy time.

---

# 27. Reproducibility and provenance

Every run manifest must include:

- Mutation Forge commit;
- HEG commit;
- dirty-tree status for both repositories;
- Python version;
- OS and architecture;
- dependency lock hash;
- complete resolved configuration;
- dataset manifest hash;
- score schema version;
- proposal schema version;
- policy schema version;
- prompt version when applicable;
- all seeds;
- resource limits;
- event schema version;
- start and end timestamps;
- terminal status.

A dirty HEG tree must be reported prominently. The project may allow it for exploratory runs but not for final benchmark evidence.

---

# 28. Failure semantics

Use explicit terminal states:

```text
completed
budget_exhausted
cancelled
configuration_invalid
backend_unavailable
static_validation_failed
sandbox_failed
policy_timeout
policy_memory_limit
policy_protocol_error
experiment_infrastructure_error
```

Never translate timeout, crash, incomplete cycle count, or verifier unknown into a successful zero.

A single bad generated program must fail only that program evaluation, not the complete archive or coordinator.

---

# 29. Documentation requirements

The first pass must create:

- `README.md`: purpose, quickstart, Stage 1 commands, current implemented stage;
- `docs/MASTER_PLAN.md`: this complete staged design;
- `docs/ARCHITECTURE.md`: modules, boundaries, data flow;
- `docs/MILESTONES.md`: stage status and gates;
- `docs/SCORE_AND_FITNESS.md`: exact graph and program score semantics;
- `docs/GENERATED_PYTHON_SECURITY.md`: planned Stage 2 threat model and limits;
- `docs/EVENT_SCHEMA.md`: Rich/JSON/durable event equivalence;
- `docs/APP_SERVER_INTEGRATION.md`: planned adapter boundary and instruction to follow the local skill;
- `docs/REPORTING.md`: run artifacts and report format;
- `docs/reports/STAGE1_REPORT.md`: generated after implementation and benchmark.

State clearly in README and milestone docs that only Stage 1 is implemented after the first pass.

---

# 30. Quality requirements

- typed public interfaces;
- deterministic seeds;
- no silent fallback to different scoring semantics;
- no broad exception swallowing;
- no hidden network or model calls;
- no model use during tests unless explicitly marked live and authorized;
- machine-readable errors;
- bounded files and logs;
- clean shutdown on Ctrl-C;
- atomic writes for final manifests and summaries;
- SQLite transactions for archive updates;
- clear separation between heuristic evidence and exact verification.

---

# 31. First-pass completion report

After implementing Stage 1, stop and return:

1. repository path;
2. final commit SHA;
3. HEG commit used;
4. files created;
5. exact setup and validation commands;
6. test counts and outcomes;
7. baseline smoke results;
8. Rich mode confirmation;
9. JSONL mode confirmation with example event types;
10. run artifact paths;
11. measured overhead and throughput;
12. known limitations;
13. explicit statement that Stages 2–7 were not implemented;
14. a short recommendation on whether Stage 2 should proceed.

Do not continue to Stage 2 without user approval.

---

# 32. Reference design notes

The project is inspired by the fixed-program-plus-evolved-function pattern used by FunSearch, but must not assume the public reference repository supplies an LLM backend, production sandbox, or distributed infrastructure. Those pieces are local project responsibilities.

The microproject differs from ordinary FunSearch examples because the evolved function does not directly construct the final mathematical object. It learns a policy for navigating a graph search space through legal, invariant-preserving rewrites.

The simplest defensible first experiment is therefore a generated **proposal ranker**, not unrestricted mutation code. This preserves normal Python control flow while keeping graph legality, scoring, and execution authority in the host.
