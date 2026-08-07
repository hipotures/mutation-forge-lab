# Native v3 ordinary-Python policy migration audit

Status: audit only; no migration is authorized or implemented.

Audit baseline: `native-v3-stepwise` at
`c95f4d9d33877ea312ae76725475566ea73172c5` (`Preserve Native v3 fallback
causes`).

Operator decision: the evolved artifact is ordinary Python source with the
public entry point `propose(ctx, graph, api, seed)`, not the current custom JSON
AST or either of its transport IRs. This decision supersedes the representation
part of the earlier Native v3 contract. It does not authorize Step 13, a
production migration, a Native v2 change, an `experiment.toml` change, an App
Server transport change, or a change to issue #47.

## 1. Scope and audit rules

This audit covers the production and test paths that implement or directly
protect:

- Codex App Server lifecycle, artifacts, persistence, resume, and
  `thread/fork`;
- provider prompts, output schemas, response parsing, and artifact projection;
- program validation, identity, behavior signatures, lineage, and
  `SearchMemoryV1`;
- graph selection, rewrite construction, serial evaluation, HEG score
  evidence, conservative interval fitness, status, telemetry, and exact
  counterexample verification; and
- the open GitHub dependency chain through Step 20.

The sibling `../heg` repository remains read-only. Native v2 remains the
default and its provider-turn artifact contract remains authoritative. The
existing worktree modification to `experiment.toml` was outside this audit and
was not touched.

Each file in Sections 7 and 8 has exactly one disposition:

- `KEEP_UNCHANGED`: representation-independent and directly reusable.
- `ADAPT`: the responsibility remains on the final path after replacing
  JSON-DSL-specific inputs, outputs, identities, or projections.
- `DONOR_ONLY`: contains useful behavior or tests to port, but the file must not
  remain on the final Python execution path.
- `SUPERSEDED`: defines or executes the custom DSL as the evolved artifact and
  must be disconnected by the Python path.
- `DELETE_LATER`: retained while both paths are needed for evidence and rollback
  comparison, then removable only after the complete Python end-to-end gates
  pass.

These are migration dispositions, not instructions to change files now.

## 2. Findings

### 2.1 Boundary that remains valid

The following contracts are independent of the evolved program representation:

1. `LocalCodexAppServerProvider`, `CodexAppServerAdapter`, authentication
   isolation, request correlation, retry accounting, terminal event handling,
   token usage, process cleanup, and provider-turn persistence.
2. Persistent thread IDs, opaque rollout paths, resume, exact inclusive
   `thread/fork` boundaries, and the rule that automatic compaction is not a
   correctness dependency.
3. The frozen Native v2 provider artifact tree and
   `make appserver-artifact-parity`. Python-policy semantic artifacts must still
   live outside the provider-turn directory.
4. `GraphState`, `RewritePlan`, backend rewrite validation, HEG provenance,
   component score evidence, exact-rational interval fitness, and conservative
   ordering.
5. One authoritative program invocation per serial step, `NoPlan` consuming the
   step, one authoritative candidate score, fail-closed infrastructure
   handling, and semantic traces whose identity excludes timing.
6. A heuristic zero being only a submission to exact verification. Only the
   exact verifier may produce `VERIFIED`.
7. Separation of contract validity, evaluation state, scientific outcome,
   infrastructure outcome, and operator activation.

### 2.2 Boundary that changes

The current model response is a nonrecursive slot-specific JSON object compiled
by the host into `mforge.native.program.v3`, then validated and executed by the
custom interpreter. Canonical identity, behavior summaries, prompts, response
schemas, provider projections, fixtures, Search Memory family extraction, and
serial invocation all assume that AST.

The Python migration must replace that chain with:

```text
provider JSON envelope
        |
        v
ordinary Python source
        |
        v
Python AST validation + canonical identity
        |
        v
isolated Python policy worker
        |
        v
RewritePlan | NoPlan
        |
        v
existing host validation, serial evaluator, scoring, and exact verification
```

The JSON response envelope remains a transport device. It is not the evolved
artifact. The evolved, selected, forked, persisted, replayed, and mutated object
is the exact Python source.

## 3. Proposed ordinary-Python policy contract

This section is normative for the proposed migration issue set, but it is not
implemented by this audit.

### 3.1 Exact model response envelope

Every root or mutation turn returns one JSON object and no Markdown fence:

```json
{
  "schema_version": "mforge.native.python_policy_response.v1",
  "source": "def propose(ctx, graph, api, seed) -> RewritePlan | NoPlan:\n    ..."
}
```

The App Server `outputSchema` is:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "source"],
  "properties": {
    "schema_version": {
      "type": "string",
      "const": "mforge.native.python_policy_response.v1"
    },
    "source": {
      "type": "string",
      "minLength": 1,
      "maxLength": 32768
    }
  }
}
```

There are no model-authored program IDs, behavior signatures, scores, lineage
IDs, host graph data, or validation claims. The host computes all of them.
`source` must be valid UTF-8 after JSON decoding, contain no NUL, use LF after
host newline normalization, and be at most 32 KiB.

### 3.2 Exact public entry point

The source must define exactly one public top-level entry point:

```python
def propose(ctx, graph, api, seed) -> RewritePlan | NoPlan:
    ...
```

The function is synchronous and undecorated. Its four positional parameters,
their order and names, and the return annotation are exact. It has no defaults,
positional-only marker, keyword-only arguments, variadic arguments, or type
parameters. The host injects the `RewritePlan` and `NoPlan` type names; imports
are forbidden. Up to 16 module-local helper functions may be defined in the
same module. Helper names must match
`helper_[A-Za-z][A-Za-z0-9_]{0,55}`. Helpers must be synchronous, undecorated,
nonrecursive, and reachable only from `propose`.

`ctx` is an immutable `PolicyContextV1` exposing only:

- `step_index: int`
- `horizon: int`
- `acceptance_profile_id: str`
- `stagnation_steps: int`
- `exploration_window_index: int`
- `accepted_rewrites: int`
- `accepted_non_improving_rewrites: int`
- `consecutive_non_improving_rewrites: int`
- `witness_cap: int`
- `invocation_ordinal: int`

`graph` is an immutable, label-opaque `GraphViewV1` exposing only:

- `order: int`
- `edge_count: int`
- `minimum_degree: int`
- `maximum_degree: int`

It does not expose an edge list, adjacency structure, scorer, verifier,
backend, graph labels, filesystem path, workspace, provider state, or lineage.
`seed` is the host-supplied non-Boolean 64-bit unsigned invocation seed.

### 3.3 Initial safe graph API

`api` is a per-invocation capability. It owns a private graph overlay and may
return only immutable, opaque references. Generated code cannot construct,
inspect, serialize, or access fields on reference objects; it may compare them
for equality, keep them in local containers, pass them back to `api`, or iterate
over bounded returned tuples.

The initial reference types are `VertexRef`, `EdgeRef`, `NonEdgeRef`, `PathRef`,
`MatchingRef`, `RelocationRef`, and `FanoutRef`. Every selector result is a
tuple capped at 64 references.

The complete initial selector surface is:

```python
api.vertices_degree_extreme(mode="max") -> tuple[VertexRef, ...]
api.vertices_degree_class(degree) -> tuple[VertexRef, ...]
api.vertices_witness_load_extreme(length, mode="max") -> tuple[VertexRef, ...]
api.edges_witness_load_extreme(length, mode="max") -> tuple[EdgeRef, ...]
api.vertices_articulation_risk(mode="max") -> tuple[VertexRef, ...]
api.edges_bridge_risk(mode="max") -> tuple[EdgeRef, ...]
api.edges_removable() -> tuple[EdgeRef, ...]
api.vertices_distance_band(source, minimum, maximum) -> tuple[VertexRef, ...]
api.non_edges_from_vertex(vertex) -> tuple[NonEdgeRef, ...]
api.non_edges_legal() -> tuple[NonEdgeRef, ...]
api.non_edges_local_cycle_risk(mode="max") -> tuple[NonEdgeRef, ...]
api.paths_length_two() -> tuple[PathRef, ...]
api.matching_k_switch_reconnections(k) -> tuple[MatchingRef, ...]
api.relocations_legal() -> tuple[RelocationRef, ...]
api.edge_fanouts_legal() -> tuple[FanoutRef, ...]
```

The only legal values for `mode` are `"min"` and `"max"`; `k` is one of
`2`, `3`, or `4`. Choice is host-deterministic:

```python
api.pick(items, seed, salt, feature="uniform") -> Ref | None
```

`salt` is an integer or a printable ASCII string of at most 128 bytes.
`feature` is `"uniform"`, `"degree"`, or `"inverse_degree"`; the two degree
features are legal only for vertex references. The host derives all draws from
the supplied seed, salt, invocation ordinal, and versioned random protocol. The
policy receives no RNG object and no ambient randomness.

The complete mutation/terminal surface is:

```python
api.add_edge(edge) -> None
api.remove_edge(edge) -> None
api.relocate_endpoint(relocation) -> None
api.k_switch(matching) -> None
api.edge_fanout(fanout) -> None
api.edge_fold(path) -> None
api.emit() -> RewritePlan
api.no_plan(reason="EXPLICIT") -> NoPlan
```

Legal `NoPlan` reasons are `EXPLICIT`, `NO_MATCH`, `ILLEGAL_FINAL_STATE`, and
`NO_EFFECT`. `api.emit()` computes the canonical edge delta from the private
overlay, enforces connectivity, minimum degree, net-edge bounds, and the
existing backend `apply_rewrite` check, and returns a host-minted
`RewritePlan`. The host accepts only a `RewritePlan` or `NoPlan` minted by the
current invocation. Direct construction, stale capabilities, foreign
references, and returning an overlay or graph object fail closed.

This surface deliberately ports the current selector/action semantics instead
of keeping the JSON DSL or exposing arbitrary graph access.

### 3.4 Allowed Python AST subset

Validation uses Python 3.12 syntax. Every node not explicitly allowed is
rejected. The initial allowlist is:

- module/function structure: `Module`, `FunctionDef`, `arguments`, `arg`,
  `Return`;
- statements: `Assign`, `AugAssign` to a local name, `If`, bounded `For`,
  `Break`, `Continue`, `Pass`, and `Expr` only when it is a permitted API call
  or the optional leading string docstring of a function;
- values: `Name`, bounded `Constant`, `Tuple`, `List`, `Dict`, `Subscript`,
  `Slice`, `IfExp`, and the narrowly permitted `Attribute` and `Call` forms
  below;
- logic: `BoolOp` with `And`/`Or`, `UnaryOp` with `Not`/`UAdd`/`USub`,
  `Compare` with `Eq`, `NotEq`, `Lt`, `LtE`, `Gt`, `GtE`, `In`, or `NotIn`;
- arithmetic: `BinOp` with `Add`, `Sub`, `Mult`, `FloorDiv`, or `Mod`; the
  `BitOr` node is permitted only in the exact public return annotation
  `RewritePlan | NoPlan`; and
- call plumbing: `keyword` with no `**` expansion.

Assignments may target only a local `Name`. Constants are `None`, Boolean,
signed 64-bit integers, or printable strings of at most 1,024 bytes. Literal
lists, tuples, and dictionaries contain at most 256 recursively bounded
values; dictionary keys are strings or integers.

All policy-defined identifiers are ASCII and at most 64 characters. Apart from
the fixed public names and the `helper_...` form, an identifier must match
`[A-Za-z][A-Za-z0-9_]{0,63}` and must not begin with an underscore.

Permitted attribute reads are exactly the documented `ctx` and `graph` scalar
fields. Permitted attribute calls are exactly the documented `api` methods.
All other attribute access is rejected. Calls may target those API methods, an
acyclic local helper, or these deterministic built-ins:
`abs`, `all`, `any`, `bool`, `enumerate`, `int`, `len`, `max`, `min`, `range`,
`reversed`, `sum`, and `tuple`.

A `for` iterator must be one of:

- `range(...)` whose start, stop, step, and maximum trip count are statically
  provable integers with at most 64 trips;
- a local name assigned directly from one selector call; or
- a literal tuple/list with at most 64 items.

`enumerate(...)` or `reversed(...)` may wrap one of those permitted iterators
without changing its proven bound.

The validated helper-call graph must be acyclic with maximum depth 8. A
host-inserted, semantics-preserving guard caps total loop-body entries at 4,096
and total helper invocations at 256 per `propose` call. Canonical identity is
computed from the validated source AST before guard insertion.

### 3.5 Forbidden Python AST subset

The following are explicitly forbidden, in addition to the fail-closed
allowlist rule:

- `Import`, `ImportFrom`, `Global`, `Nonlocal`, `ClassDef`, `AsyncFunctionDef`,
  `Await`, `Yield`, `YieldFrom`, `Lambda`, and decorators;
- `While`, comprehensions, generator expressions, `Match`, `Try`, `TryStar`,
  `Raise`, `Assert`, `With`, `AsyncWith`, `Delete`, and `NamedExpr`;
- `Attribute` except the exact `ctx`, `graph`, and `api` cases above;
- attribute/subscript destructuring or mutation, starred targets, and
  augmented assignment to anything except a local name;
- dynamic calls, method calls on policy-created objects, recursion, closures,
  default argument expressions, and unknown or shadowed names;
- `eval`, `exec`, `compile`, `open`, `input`, `print`, `__import__`,
  reflection, frame/code access, serialization, and every non-allowlisted
  builtin;
- names or attributes beginning with `_`, including all dunder access;
- floating-point and complex literals, `Pow`, true division, matrix,
  bitwise, or shift operators;
- exception handling, process/thread creation, signals, clocks, environment,
  filesystem, database, socket/network, subprocess, FFI, and package access;
  and
- any top-level statement other than the permitted function definitions.

Source validation is defense in depth. It is not the security boundary and is
never presented as proof that arbitrary Python is safe.

### 3.6 Process, time, memory, and API-call limits

One validated candidate is loaded into one dedicated worker process and reused
only for that candidate's serial development-panel evaluation. The coordinator
never imports or executes generated source. The worker starts in a new process
group with a sanitized environment, isolated empty working directory, no
inherited stdin, no network, no writable repository/workspace access, no
ambient package path, and no inherited file descriptors except its framed
protocol.

Initial hard limits are:

| Limit | Value |
| --- | ---: |
| Source bytes | 32 KiB |
| Python AST nodes | 2,000 |
| Top-level module-local helpers | 16 |
| Address space (`RLIMIT_AS`) | 128 MiB |
| Cumulative worker CPU (`RLIMIT_CPU`) | 60 s |
| Worker wall lifetime per candidate evaluation | 60 s |
| Wall time per `propose` call | 250 ms |
| File size (`RLIMIT_FSIZE`) | 64 KiB |
| Open descriptors (`RLIMIT_NOFILE`) | 16 |
| Processes (`RLIMIT_NPROC`) | 1 |
| Framed request | 256 KiB |
| Framed response | 32 KiB |
| Retained diagnostics | 64 KiB |
| Total API calls per `propose` | 256 |
| Selector calls per `propose` | 64 |
| Mutation action calls per `propose` | 64 |
| Selector result size | 64 references |
| Net added edges | 8 |
| Net removed edges | 8 |
| Deterministic random draws | 2,048 |
| Loop-body entries | 4,096 |
| Helper invocations / call depth | 256 / 8 |

Length-prefixed canonical JSON is the host/worker protocol; pickle is
forbidden. OS isolation must deny filesystem, network, subprocess, ptrace, and
new-process capabilities even if AST validation is bypassed. Unsupported
platforms or unavailable sandbox controls fail closed before evaluation.

### 3.7 Canonical Python program identity

The host retains exact normalized source and computes:

1. `source_sha256`: SHA-256 of LF-normalized UTF-8 source bytes.
2. A Python 3.12 AST with location fields, comments, type comments, and
   function docstrings excluded from semantic identity.
3. Deterministic alpha-normalization: `propose` parameters keep their required
   names; module-local helpers are renamed by source order; helper parameters and
   local bindings are renamed by first lexical binding in each function.
   Constants, statement order, control flow, API method names, and field names
   are not changed.
4. `canonical_ast_bytes`: UTF-8 bytes of
   `ast.dump(normalized_tree, annotate_fields=True, include_attributes=False)`.
5. `program_hash`: lowercase SHA-256 of:

```text
b"mforge-native-python-policy-v1\0"
+ b"python-3.12\0"
+ b"validator-v1\0"
+ canonical_ast_bytes
```

The exact source, source hash, program hash, Python version, validator version,
AST node count, validation diagnostics, and normalization protocol are
persisted. Equivalent formatting, comments, docstrings, and local-name changes
deduplicate; semantic changes do not.

Behavior identity is separate. A fixed, versioned, non-held-out probe manifest
runs the policy on bounded `(ctx, graph, seed)` cases and records for each case
the canonical plan delta or `NoPlan`/failure code plus the semantic API call
trace. The behavior signature is the lowercase SHA-256 of:

```text
b"mforge-native-python-behavior-v1\0" + canonical_json_bytes(probe_records)
```

Host duplicate rejection remains authoritative on either program hash or
behavior signature.

### 3.8 Sandbox and fail-closed behavior

Validation happens before worker creation. Invalid envelope, invalid UTF-8,
syntax error, forbidden AST, or canonical identity failure marks the candidate
`CONTRACT_INVALID`; it is not evaluated or repaired outside the configured
bounded provider repair policy. A policy-caused probe failure is recorded in
the behavior signature rather than reclassified as a contract error; a probe
infrastructure failure fails the run infrastructure.

Inside evaluation:

- policy exception, invalid API use, stale/foreign reference, loop/helper/API
  budget exhaustion, invalid return, per-call timeout, or candidate-caused
  worker crash becomes deterministic `PROGRAM_FAILURE` and worst candidate
  fitness for that episode;
- sandbox setup failure, host/worker protocol corruption, unavailable OS
  isolation, coordinator failure, or backend/scorer violation is
  infrastructure failure and cannot be converted into scientific fitness;
- illegal/no-effect final overlays become the existing explicit `NoPlan`
  outcomes and consume the serial step;
- any failed or timed-out worker is killed, reaped, and never reused; and
- stdout is unavailable, stderr is bounded/redacted, secrets are never passed,
  and no raw source or private graph data is written to provider-turn
  directories.

The host revalidates every returned plan through the existing rewrite host
before scoring. A plan cannot invoke the scorer or exact verifier.

### 3.9 Evaluator invocation

`evaluate_serial_program` keeps its graph order, graph/policy seed, horizon,
witness cap, scoring, interval proof, trace, and exact verification semantics.
Its representation-specific dependency changes from:

```python
invoke_program(validated_ast, graph, context, ...)
```

to the conceptual adapter:

```python
python_runner.invoke(
    validated_policy,
    ctx=policy_context,
    graph=graph_view,
    api=safe_graph_api,
    seed=invocation_seed,
) -> RewritePlan | NoPlan | ProgramFailure
```

The safe API owns the overlay and uses the existing backend rewrite boundary.
There remains exactly one invocation per step, `NoPlan` consumes the step, only
an emitted and host-validated plan is scored, and only one authoritative
candidate score is recorded. The trace replaces interpreter-node events with
bounded semantic API events while retaining graph identities, plan deltas,
score evidence, interval proofs, failure/no-plan reasons, verification records,
and timing-independent semantic hashing.

### 3.10 Parent source, exact fork, and feedback

The specification thread contains the complete Python contract, safe API,
limits, response envelope, and immutable experiment brief. Each accepted root
or child records the provider `threadId`, response `turnId`, exact source,
program hash, behavior signature, evaluation summary, and parent hashes in
host-only lineage state.

For a child:

1. select a parent from the immediately preceding evaluated generation;
2. call `thread/fork` at that parent's exact inclusive response `turnId`;
3. verify the returned fork is durable and contains the specification and
   parent turn but no later sibling;
4. send one mutation request containing the exact parent Python source again
   and compact evaluation feedback; and
5. require one complete replacement Python program in the response envelope.

Repeating the source makes the mutation request auditable and means compaction
is not a correctness dependency. The model receives no program hash, behavior
signature, provider/thread IDs, host paths, graph labels, raw score-worker
output, exact-verifier internals, or cryptographic Search Memory digests.

Compact feedback contains only bounded semantic data: generation, parent
fitness interval, component interval summaries, outcome counts, no-plan and
program-failure counts, accepted rewrite families, bounded failure reasons, and
one-to-three host summaries. It contains no held-out panel results.

Fresh roots fork from the exact specification anchor, not a parent, and receive
the bounded model projection of Search Memory with `active_parent = null`.

### 3.11 Root and child allocation

Let `P` be the immutable population size in the run manifest.

- Generation 0 has `P` fresh roots.
- Every later generation has
  `R = max(1, P // 4)` fresh-root slots and `P - R` child slots.
- Slot kinds and order are fixed in the immutable generation manifest before
  any provider turn: child slots first in deterministic selected-parent order,
  then root slots.
- Parent selection uses the existing conservative interval-fitness ordering,
  deterministic tie handling, and behavior diversity over the immediately
  preceding evaluated generation. Selection never uses held-out data.
- A parent may occupy multiple child slots only after every selectable parent
  has received one slot; repeated allocation is deterministic.
- If no valid evaluated parent exists, all `P` slots become fresh roots and the
  deviation is recorded before generation starts.
- Provider repairs do not create population slots. Invalid, duplicate, or
  failed candidates consume their planned slot and budget.

All candidates in a generation receive equal development graph/seed panels and
equal evaluation budgets. Root allocation is not opportunistically changed
after observing generation results.

### 3.12 Search Memory, status, telemetry, and stop behavior

`SearchMemoryV1` keeps its bounded host ownership, canonical ordering, maximum
64 identities/signatures, maximum 8 patterns per outcome, maximum 16 active
lineages/archive IDs, and 16 KiB canonical bound. Its schema and extractors
change to Python-policy program hashes, behavior signatures, safe API families,
and Python control-flow summaries. It still excludes complete program source
from its general model projection; only the selected child receives its exact
parent source.

Status retains all current lifecycle/scientific fields and adds:

- generation index and immutable generation-manifest hash;
- planned/attempted/contract-valid/unique/evaluated root and child counts;
- source bytes, AST nodes, program hash, behavior signature, and parent hashes;
- fork attempts/failures, provider repairs, sandbox starts/restarts/timeouts,
  program failures, API-budget failures, and invalid return counts;
- policy invocations, selector/action/random-draw counts, graph score attempts,
  exact-verifier submissions, and development-panel completion; and
- search stop reason: budget exhausted, operator stop, infrastructure failure,
  or verified counterexample.

Communication completion never implies scientific success. A heuristic zero
only queues the existing exact verification path. Search stops early only when
that path records a verified counterexample under the accepted verifier
contract.

## 4. Current provider, lineage, and scientific invariants

The migration issues must continuously enforce these existing gates:

- `make appserver-artifact-parity` remains byte-exact and representation
  independent.
- App Server uses the existing isolated lifecycle, durable thread/turn
  identities, retry boundary, usage finality, opaque rollout paths, and bounded
  diagnostics.
- `thread/fork` is inclusive at the specified `lastTurnId`; a child sees its
  parent but not later siblings. Fresh roots fork from the specification
  anchor.
- Search Memory is host authoritative, bounded, canonical, and advisory to the
  model. Automatic compaction remains best effort only.
- Development panel manifests, graph seeds, policy seeds, population slots,
  generation budgets, and evaluation order are immutable.
- HEG initial/expanded score attempts, safe partial-evidence rules,
  exact-rational intervals, and strict proof
  `candidate.upper < incumbent.lower` remain unchanged.
- Provider failure creates no scientific terminal result. Program failure is
  not infrastructure success. Exact verification alone establishes a
  counterexample.
- Native v2 remains default and its smoke, resume, artifact, and status tests
  pass at every issue gate.

## 5. Representation dependency cuts

The migration must break these dependency edges, not wrap them in compatibility
layers:

1. provider response -> `parse_strict_json` -> slot-specific/flat IR compiler
   -> `ValidatedProgram`;
2. `ValidatedProgram` -> `invoke_program` -> declarative graph runtime;
3. JSON canonical bytes -> `program_hash`;
4. AST token traversal -> behavior signature and Search Memory family/control
   flow;
5. AST provider/batch markdown projection -> provider artifacts;
6. AST fixture -> serial evaluator/provider-evaluation smoke;
7. AST communication-mode config -> preview routing and status; and
8. batch/slot-specific prompts and schemas -> parent/root generation.

The replacement edges are Python response parsing, AST validation/identity,
sandboxed `propose` invocation, semantic API traces, and Python source lineage.
There must be no JSON-AST-to-Python compiler and no Python-to-JSON-AST lowering.

## 6. Category summary

| Category | Meaning in this tree |
| --- | --- |
| `KEEP_UNCHANGED` | App Server lifecycle/isolation, generic artifacts/I/O, shared graph/rewrite models, HEG scoring/evidence, deterministic randomness, counterexample verification |
| `ADAPT` | provider projections, Python prompt/envelope, preview/search orchestration, Search Memory extractors, evaluator invocation, status/CLI, provider-to-evaluation smokes |
| `DONOR_ONLY` | current graph selectors/actions, persistent/fork/compaction experiments, DSL semantic tests whose assertions must be ported |
| `SUPERSEDED` | recursive JSON program contract, IR compiler, and custom interpreter as the evolved execution representation |
| `DELETE_LATER` | batch/IR-only schema experiments, fixtures, prompts, and tests retained until the Python end-to-end gate passes |

## 7. Production file audit

“Reusable” names exact current symbols or assets. “Break” identifies the
representation dependency to remove. No row authorizes a current edit.

| Exact path | Category | Current responsibility | Python-policy responsibility | Exact reusable symbols/assets | Break / risk / proposed migration step |
| --- | --- | --- | --- | --- | --- |
| `src/mutation_forge/native_v3/__init__.py` | `ADAPT` | Public Native v3 export surface. | Export Python policy contracts/runner and retained scoring/evaluation types. | Existing scoring, HEG, randomness, serial types. | Break exports of `ValidatedProgram`/interpreter/IR; risk eager DSL imports; update only after new modules pass isolated tests. |
| `src/mutation_forge/native_v3/canonical.py` | `ADAPT` | Strict JSON/canonical bytes and DSL program hash. | Keep canonical JSON for metadata; replace program identity with canonical Python AST identity. | `CanonicalJsonError`, `parse_strict_json`, `canonical_json_bytes`, `domain_hash`, `json_value`. | Break `program_hash` dependency on JSON AST; add versioned Python identity, then move callers atomically. |
| `src/mutation_forge/native_v3/contracts.py` | `SUPERSEDED` | Recursive DSL types, registries, validation, and static limits. | None on final execution path. | Registry facts may inform the initial safe API, but not remain imported. | Entire `ValueType`/`ProgramContract`/`ValidatedProgram`/`validate_program` path is DSL-specific; disconnect after Python validator/evaluator gate. |
| `src/mutation_forge/native_v3/graph_runtime.py` | `DONOR_ONLY` | Private overlay, typed refs, selectors, actions, and plan emission for the interpreter. | Port selector/action behavior behind the safe Python API and host-minted plans. | `VertexRef`, `EdgeRef`, `NonEdgeRef`, `PathRef`, `MatchingRef`, `RelocationRef`, `FanoutRef`, `GraphOverlay`, selector/action bodies, emit validation. | Break `ValueType`, string dispatch, interpreter path, and DSL metadata; risk semantic drift; prove donor-vs-new API parity before disconnecting. |
| `src/mutation_forge/native_v3/interpreter.py` | `SUPERSEDED` | Executes the custom AST with rollback, limits, `NoPlan`, failure taxonomy, and traces. | Replaced by isolated Python runner; port outcome semantics, not interpreter execution. | Outcome names and tests for `NoPlan`, illegal final state, program failure, and deterministic trace. | Break `ValidatedProgram` and AST-node dispatch; extract a representation-independent `NoPlan` contract before disconnection. |
| `src/mutation_forge/native_v3/single_program_contract.py` | `ADAPT` | Builds one-AST prompt, schema, request, and response validation. | Build exact Python envelope, source prompt, and Python validation diagnostics. | `SingleProgramRequest`, `SingleProgramResponse`, request-building/artifact boundary. | Replace selector/action AST projection and `validate_program`; risk schema keywords/provider parity; add Python golden responses first. |
| `src/mutation_forge/native_v3/single_program_ir.py` | `SUPERSEDED` | Builds slot-specific/flat schemas and compiles them to the DSL AST. | None. | Schema benchmark measurements are historical evidence only. | Remove compiler dependency; do not create a Python IR or compatibility lowering. |
| `src/mutation_forge/native_v3/cohort.py` | `ADAPT` | Sequential slots, manifests, response parsing, dedupe, repair, provenance, and reports. | Generation manifests, root/child slots, Python response validation, dedupe, evaluation, and reports. | `CohortEntry`, manifests, outcome/dedup/accounting/finalization patterns. | Break batch AST parsing and `ValidatedProgram`; risk unequal budget or changed repair counting; move to one source per provider turn. |
| `src/mutation_forge/native_v3/persistent_experiment.py` | `DONOR_ONLY` | Persistent-vs-fresh experiment, behavior signature, retries, and reports. | No final production path; donate lifecycle/usage assertions and experiment evidence. | `TurnObservation`, bootstrap/follow-up lifecycle, payload leak gate, resume/retry handling. | Break AST payload and `_behavior_signature`; final search belongs in preview/search orchestration, not this A/B experiment. |
| `src/mutation_forge/native_v3/preview.py` | `ADAPT` | Guarded persistent single-AST cohort, durable progress, workers, memory, artifacts, and replacement process. | Guarded Python-policy generations with exact parent forks, roots, evaluation, and durable state. | `run_persistent_single_ast_cohort` orchestration, atomic progress, replacement process, artifact completeness. | Break AST parsing/identity/IR/Search Memory projections; risk resume incompatibility; use a new workspace schema and reject old workspaces. |
| `src/mutation_forge/native_v3/lineage_experiment.py` | `DONOR_ONLY` | Standalone exact-fork/Search Memory experiment. | No final execution path; donate exact inclusive fork and artifact checks. | `_fork`, fork/program records, `run_lineage_experiment`, bounded memory construction. | Break AST fixtures/prompts/identity; port verified fork assertions into Python lineage tests. |
| `src/mutation_forge/native_v3/compaction_experiment.py` | `DONOR_ONLY` | Compaction retention experiment and manifest probes. | No correctness path; retain evidence that compaction is best effort only. | Compaction lifecycle, correlated acknowledgement, `compare_manifest`. | Break AST manifest fields; do not add production compaction dependency. |
| `src/mutation_forge/native_v3/search_memory.py` | `ADAPT` | Bounded `SearchMemoryV1`, AST family/control-flow summaries, lineage, and duplicate gate. | Bounded Python program/behavior memory and Python/API summaries. | `PatternSummary`, `LineageSummary`, `ActiveParentReference`, `SearchMemoryV1`, `reject_duplicate`, bounds/canonicalization. | Replace `program_families`/`program_control_flow` AST mapping and DSL schema IDs; risk leaking source/digests to model; preserve bounds. |
| `src/mutation_forge/native_v3/serial_evaluator.py` | `ADAPT` | One-program serial episodes, interpreter outcomes, scoring, trace, and verification. | Invoke isolated Python runner while preserving all host scientific semantics. | `SerialEpisodeConfig`, `GraphIdentity`, trace/result types, score/verification flow. | Break `ValidatedProgram`/`invoke_program`; risk confusing candidate and infrastructure failure; add adapter-level parity tests. |
| `src/mutation_forge/native_v3/scoring.py` | `KEEP_UNCHANGED` | Component evidence, rational intervals, AUC/fitness, conservative ordering/cache. | Same. | `IntegerInterval`, `RationalInterval`, `ScoreEvidence`, `EnergyScale`, `proved_strict_energy_improvement`, `candidate_fitness`, `conservative_fitness_key`, `ScoreEvidenceCache`. | No representation edge except caller identity payload; prove unchanged tests and hashes. |
| `src/mutation_forge/native_v3/heg_scoring.py` | `KEEP_UNCHANGED` | HEG evidence adapter, budgets, timeouts, provenance, merge. | Same. | `HegScoreEvidenceAdapter`, `merge_score_evidence`, `scorer_for_backend`, `backend_identity`. | Do not modify HEG or introduce fallback scorer. |
| `src/mutation_forge/native_v3/randomness.py` | `KEEP_UNCHANGED` | Versioned SplitMix64 seed derivation and bounded draws. | Same protocol behind `api.pick`. | `derive_seed64`, `splitmix64`, `draw64`, `uniform_below`, `weighted_index`. | Preserve frozen vectors; safe API must not add ambient RNG. |
| `src/mutation_forge/native_v3/experiment.py` | `ADAPT` | V3 config, workspace/status, preflight, routing, and defaults. | New opt-in Python preview/search route and Python status fields. | `V3Config`, `load_v3_config`, `experiment_protocol`, `v3_status`, provider/backend preflight. | Break AST communication modes/output hashes; risk Native v2 default; introduce a new explicit workspace/protocol version. |
| `src/mutation_forge/native_v3/provider_smoke.py` | `ADAPT` | One provider AST request, parse, failure class, artifacts. | One provider Python-source request and validation smoke. | `NativeV3ProviderSmokeError`, request/run structure, failure artifact handling. | Replace AST prompt/schema/parser; preserve provider lifecycle and no scientific result on failure. |
| `src/mutation_forge/native_v3/provider_evaluation.py` | `ADAPT` | Replay provider AST into serial HEG evaluation and provenance. | Replay exact Python source through validator/sandbox/evaluator. | `run_provider_evaluation_smoke`, provider/backend provenance reporting. | Replace `_load_validated_program`; risk replay executing in coordinator; require sandbox-only replay. |
| `src/mutation_forge/experiment/artifacts.py` | `ADAPT` | Generic turn store plus Native v3 batch/AST markdown projection. | Preserve turn tree; project Python envelope/source metadata outside provider-turn semantic path. | `TurnArtifactStore`, `copy_canonical_source`, generic diagnostics/usage quality. | Replace `NATIVE_V3_PROGRAM_BATCH_PROJECTION` and AST renderer without changing frozen provider file set. |
| `src/mutation_forge/experiment/provider.py` | `ADAPT` | Shared local App Server provider plus Native v3 decoded projection hook. | Same provider with Python-policy projection hook only. | `LocalCodexAppServerProvider` lifecycle/generation methods. | Break only DSL projection call; high risk to shared v2 transport, so enforce artifact parity before/after. |
| `src/mutation_forge/stage3/app_server.py` | `KEEP_UNCHANGED` | App Server protocol adapter, persistence/resume/fork, limits, usage, events. | Same. | `AppServerLimits`, `ModelProfile`, `TokenUsage`, `GenerationResult`, `CompactionResult`, `ForkResult`, `CodexAppServerAdapter`, `AppServerGenerationProvider`. | Callers change prompts/schema only; do not change lifecycle for migration. |
| `src/mutation_forge/stage3/isolation.py` | `KEEP_UNCHANGED` | Provider capsule and secure process parent. | Same provider isolation; Python policy sandbox is a separate capability-specific worker. | `IsolatedCapsule`, `secure_capsule_parent`, `IsolationError`. | Do not claim provider capsule alone safely executes generated Python. |
| `src/mutation_forge/experiment/generation.py` | `KEEP_UNCHANGED` | Generic generation request/result and usage/failure contracts. | Same. | `GenerationRequest`, `GenerationResult`. | Output schema content changes only at caller. |
| `src/mutation_forge/experiment/json_io.py` | `KEEP_UNCHANGED` | Bounded deterministic JSON/gzip I/O. | Same for metadata/envelopes. | `read_json`, `write_json`. | Python source remains a JSON string only at transport/artifact boundary. |
| `src/mutation_forge/models.py` | `KEEP_UNCHANGED` | Shared `GraphState`, `RewritePlan`, score and dataset types. | `RewritePlan` remains the host result contract. | `GraphState`, `RewritePlan`, `GraphScore`, `normalized_edge`. | Do not let generated code construct arbitrary plans; host mints and validates them. |
| `src/mutation_forge/counterexamples.py` | `KEEP_UNCHANGED` | Counterexample pipeline/inspector and verification results. | Same. | `CounterexamplePipeline`, `CounterexampleInspector`, verification result types. | Preserve exact-verifier-only `VERIFIED` authority. |
| `src/mutation_forge/backends/base.py` | `KEEP_UNCHANGED` | Backend/scoring/rewrite protocols. | Same. | `GraphBackend`, `ScoreProfileRecorder`, `ScoringBackendError`. | No policy-source dependency. |
| `src/mutation_forge/backends/heg.py` | `KEEP_UNCHANGED` | Current HEG backend. | Same. | `HegBackend`. | Sibling HEG stays read-only; no fallback or integration change. |
| `src/mutation_forge/artifacts.py` | `KEEP_UNCHANGED` | Run metadata, canonical hashes, git/environment provenance. | Same, including exact HEG commit/dirty state. | `canonical_json_hash`, `git_state`, `environment_record`, `RunArtifacts`. | Do not weaken provenance or copy SQLite unsafely. |
| `src/mutation_forge/cli.py` | `ADAPT` | Public v3 run/status routing and legacy protocols. | Expose only explicit Python preview/search route while v2 remains default. | Existing protocol dispatch and read-only status behavior. | Break DSL mode names; risk accidental default switch; require explicit operator gate. |
| `scripts/native_v3_compaction_experiment.py` | `DONOR_ONLY` | Runs AST compaction experiment. | Historical/diagnostic only. | Parser/report plumbing and compaction evidence. | AST fixtures are not a Python correctness path; no production compaction. |
| `scripts/native_v3_lineage_experiment.py` | `DONOR_ONLY` | Runs AST lineage/fork experiment. | Historical/diagnostic only. | CLI/fork report plumbing. | Port exact fork assertions to new tests, not this execution path. |
| `scripts/native_v3_persistent_experiment.py` | `DONOR_ONLY` | Runs persistent-vs-fresh AST A/B experiment. | Historical evidence only. | Cost/validity reporting and provider harness. | Do not keep an A/B experiment as the final search scheduler. |
| `scripts/native_v3_preview_ab_gate.py` | `ADAPT` | Verifies selected guarded AST preview evidence. | Verify guarded Python preview, sandbox, lineage, scientific, and v2 parity gates. | `REPORT_SCHEMA_VERSION`, selected-summary gate pattern. | Replace AST/brief-set identity; operator acceptance remains explicit. |
| `scripts/native_v3_provider_evaluation_smoke.py` | `ADAPT` | Runs provider-AST-to-real-HEG smoke. | Run provider-Python-to-sandbox-to-real-HEG smoke. | `_default_workspace`, HEG/serial setup. | Replace AST artifact loader; keep HEG read-only and opt-in. |
| `scripts/native_v3_provider_smoke.py` | `ADAPT` | Runs one AST provider smoke. | Run one Python response/validation smoke without evaluation. | `_default_workspace`, provider startup/error handling. | Replace prompt/schema/parser only. |
| `scripts/native_v3_transport_schema_experiment.py` | `DELETE_LATER` | Benchmarks slot-specific/flat JSON IR transport schemas. | None after Python envelope acceptance. | Historical metrics/report refresh only. | Retain until Python provider/artifact gates pass, then remove; do not extend with a third IR. |
| `scripts/appserver_artifact_parity.py` | `KEEP_UNCHANGED` | Frozen provider artifact parity checker. | Same mandatory gate. | Entire parity CLI. | Any change signals an out-of-scope transport regression. |
| `configs/native/native-v3-program.schema.json` | `SUPERSEDED` | Recursive JSON DSL program schema. | None. | Historical contract evidence. | Disconnect from prompts/provider/parser/evaluator; do not translate Python into it. |
| `configs/native/native-v3-cohort-envelope.schema.json` | `DELETE_LATER` | Multi-program AST batch envelope. | None; final turns return one Python source. | Historical batch evidence. | Retain through transition for rollback comparison only. |
| `configs/native/native-v3-provider-envelope.schema.json` | `ADAPT` | One-program AST provider envelope. | Exact two-field Python response envelope, preferably under a new schema filename/version. | Single-response transport placement. | Break AST payload and model-authored summaries; risk confusing envelope with evolved artifact. |
| `configs/native/native-v3-semantics.md` | `DONOR_ONLY` | DSL/interpreter selector and action semantics. | Source for safe API semantic parity tests, not runtime specification. | Selector/action relations and terminal semantics. | Rewrite as Python safe-API documentation only after donor parity; old semantics must leave final path. |
| `configs/native/generated-policy.schema.json` | `DELETE_LATER` | Legacy generated-policy schema used by older artifacts. | None for the new Native v3 path. | Historical artifact evidence only. | Confirm no Python route/import before eventual removal; do not add compatibility fallback. |
| `prompts/native-v3/system.md` | `ADAPT` | One declarative AST system prompt. | Python policy system contract and safe API/AST/security rules. | Mission and host/model authority separation. | Remove every “JSON AST” instruction and forbid model-authored host bookkeeping. |
| `prompts/native-v3/request.md` | `ADAPT` | Minimal AST smoke request. | Minimal valid Python policy smoke request. | Smoke request role. | Replace AST fixture expectation. |
| `prompts/native-v3/cohort-system.md` | `DELETE_LATER` | Eight-program AST batch system prompt. | None in one-source-per-turn search. | Historical batch comparison only. | Retain until guarded Python search passes; then remove with cohort envelope. |
| `prompts/native-v3/cohort-request.md` | `DELETE_LATER` | Eight-slot AST batch request. | None. | Historical slot manifest wording only. | Generation manifest moves host-side; no batch response compatibility. |
| `prompts/native-v3/cohort-repair.md` | `DELETE_LATER` | Repairs a wholly unusable AST batch. | None; repairs apply to one Python response turn. | Bounded-repair principle only. | New repair prompt belongs with Python single-program contract. |
| `prompts/native-v3/single-program-system.md` | `ADAPT` | Mathematical one-AST system prompt. | Mathematical ordinary-Python policy system prompt. | Conjecture context, label opacity, host authority. | Replace DSL/IR requirements with exact API and AST subset. |
| `prompts/native-v3/single-program-request.md` | `ADAPT` | Dynamic AST/IR checklist. | Root or child request, exact response envelope, parent source/feedback when applicable. | Brief/forbidden-length injection pattern. | Split root vs mutation payload; keep held-out data absent. |
| `tests/fixtures/native_v3_single_program_responses.json` | `DELETE_LATER` | Golden AST responses for contract/persistent/fork experiments. | New Python fixtures live under a new versioned fixture; this file is legacy evidence. | Response scenario coverage. | Retain until all adapted tests use Python fixtures and full gates pass. |

## 8. Test file audit

For mixed shared files, only the named tests are in migration scope; unrelated
Native v2/Stage 3 coverage remains untouched.

| Exact path | Category | Current responsibility | Python-policy responsibility | Exact reusable tests | Break / risk / proposed migration step |
| --- | --- | --- | --- | --- | --- |
| `tests/unit/test_native_v3_contracts.py` | `DONOR_ONLY` | Strict JSON, DSL schema/types/limits/hash, v2/v3 isolation. | Donate fail-closed diagnostics, canonicalization vectors, and import isolation to new Python contract tests. | `test_strict_json_rejects_ambiguous_numeric_and_object_syntax`; `test_canonical_json_has_normative_order_escaping_and_domain_hashing`; `test_invalid_envelope_nodes_fields_and_references_fail_closed`; `test_native_v2_and_v3_imports_are_isolated`. | DSL assertions cannot remain final acceptance; add separate Python validator/identity suite before disconnecting. |
| `tests/unit/test_native_v3_interpreter.py` | `DONOR_ONLY` | DSL execution, rollback, limits, replay, no-plan, label opacity. | Port semantic cases to safe API/sandbox/serial adapter tests. | `test_random_protocol_vectors_and_replay_are_frozen`; `test_failed_graph_branch_restores_overlay_and_outer_binding`; `test_noop_and_illegal_final_graph_are_no_plan_not_rewrites`; `test_graph_resource_error_is_not_catchable`. | Do not run custom interpreter on final path; prove selector/action/plan parity instead. |
| `tests/unit/test_native_v3_single_program_contract.py` | `ADAPT` | Prompt/schema/golden AST response validation. | Exact Python envelope, entry point, prompt, source limits, repair diagnostics. | `test_golden_single_program_responses_match_schema_and_validator` plus system/request separation, invalid-field, and deterministic-size cases. | Replace all AST goldens; risk permissive JSON Schema or Markdown source wrappers. |
| `tests/unit/test_native_v3_single_program_ir.py` | `DELETE_LATER` | Slot/flat IR schema/compiler tests. | None. | `test_slot_specific_schema_golden_and_compiler` as historical comparison only. | Remove only after Python envelope/provider gates; no new IR compiler tests. |
| `tests/unit/test_native_v3_cohort.py` | `ADAPT` | Slot ordering, partial validity, duplicate aliases, usage and manifests. | Root/child manifest ordering, invalid-slot consumption, Python identity dedupe, equal budgets. | `test_response_order_does_not_change_programs_or_lineage`; `test_partial_invalidity_keeps_valid_siblings`; `test_duplicate_program_retains_aliases_and_counts_once`. | Replace batch semantics with one-turn slots; preserve deterministic accounting. |
| `tests/unit/test_native_v3_persistent_experiment.py` | `DONOR_ONLY` | Persistent lifecycle, resume, retry, artifacts, terminal turns. | Port lifecycle assertions into Python search/provider tests. | `test_durable_thread_resumes_after_process_restart`; `test_resume_rejects_foreign_and_terminal_status_notifications`; `test_server_retry_is_internal_only_for_the_persistent_experiment`. | AST payload/behavior assertions leave final path; provider lifecycle remains unchanged. |
| `tests/unit/test_native_v3_lineage_experiment.py` | `ADAPT` | Exact forks, bounded Search Memory, duplicate gates. | Exact Python-parent fork/source feedback/root fork and Python identity/memory gates. | `test_thread_fork_is_inclusive_and_excludes_later_turns`; `test_search_memory_is_canonical_bounded_and_contains_no_ast`; `test_host_duplicate_gate_rejects_hash_and_behavior_signature`. | Replace “contains no AST” with “general projection contains no source”; selected child must receive exact parent source. |
| `tests/unit/test_native_v3_compaction_experiment.py` | `DONOR_ONLY` | Compaction acknowledgement/failure and manifest comparison. | Preserve only the assertion that compaction cannot be a correctness dependency. | `test_compaction_lifecycle_uses_exact_request_and_correlated_item`; `test_parent_probe_failure_preserves_completed_compaction_evidence`. | Do not require compaction for Python search correctness. |
| `tests/unit/test_native_v3_preview.py` | `ADAPT` | Pending/scientific separation, unique AST per turn, replacement process. | Unique Python program per turn, pending/evaluated outcomes, sandbox/provider process separation. | `test_preview_memory_separates_pending_and_scientific_outcomes`; `test_persistent_preview_publishes_one_unique_ast_per_turn`; `test_persistent_preview_uses_one_replacement_process_for_both_workers`. | Replace AST uniqueness; add root/child generation accounting and sandbox failures. |
| `tests/unit/test_native_v3_preview_ab_gate.py` | `ADAPT` | Selected preview/brief-set gate. | Python end-to-end acceptance gate and immutable panel/generation manifest. | `test_selected_summary_requires_the_integrated_contract`; `test_selected_summary_rejects_a_different_brief_set`. | Gate must require sandbox/evaluator/lineage/scientific/v2 parity, not communication validity alone. |
| `tests/unit/test_native_v3_provider_smoke.py` | `ADAPT` | AST parse, fake provider artifacts, auth failure, dedicated prompt/schema. | Python source parse/validate and unchanged provider artifacts/auth. | Auth failure and dedicated request/artifact cases; replace `test_recorded_response_parses_to_canonical_native_v3_ast`. | Preserve transport paths; change only semantic projection. |
| `tests/unit/test_native_v3_provider_evaluation.py` | `ADAPT` | Recorded AST replay/evaluation parity and provider-failure status. | Recorded Python source sandbox replay/evaluation parity. | `test_recorded_provider_response_replays_same_semantic_evaluation`; `test_provider_failure_creates_no_scientific_terminal_result`. | Replay must never import source in test coordinator. |
| `tests/unit/test_native_v3_heg_scoring.py` | `KEEP_UNCHANGED` | HEG budgets, timeout classes, fallback rejection. | Same. | All tests, especially `test_adapter_uses_locked_budgets_and_preserves_sound_bounds` and `test_cpp_worker_failure_never_enters_a_python_reference_fallback`. | Any change is out of scope. |
| `tests/unit/test_native_v3_scoring.py` | `KEEP_UNCHANGED` | Rational intervals, dominance, AUC/fitness, evidence hash. | Same. | `test_interval_utility_auc_and_fitness_are_hand_calculated_rationals`; `test_unproved_overlap_is_never_a_strict_improvement`; `test_identical_evidence_has_identical_semantic_hash`. | Candidate identity field changes must not change scientific arithmetic. |
| `tests/unit/test_native_v3_serial_evaluator.py` | `ADAPT` | Deterministic traces and interpreter outcome/scoring semantics. | Same semantics through Python runner. | `test_fixed_fixture_replays_identical_semantic_trace_and_hash`; `test_no_plan_consumes_horizon_without_scoring_nonexistent_graph`; `test_safe_timeout_partial_can_win_only_by_proved_interval_dominance`; `test_program_failure_is_worst_fitness_not_infrastructure`. | Replace AST fixture/invocation only; trace event schema must be versioned. |
| `tests/unit/test_native_v3_experiment.py` | `ADAPT` | V3 config, communication modes, v2-section rejection, route/status. | New explicit Python protocol/workspace and status fields. | `test_v3_config_is_explicit_and_bounded`; `test_v3_rejects_native_v2_sections`; `test_public_cli_routes_v3_run_and_status`. | Remove DSL modes without accepting old workspaces; keep v2 default. |
| `tests/integration/test_native_v3_route.py` | `ADAPT` | Preview/batch routing, preflight resumability, invalid/duplicate handling. | Python preview/search routing, resume, invalid/duplicate slot consumption. | `test_v3_routes_selected_preview_without_constructing_batch_provider`; current preflight/read-only status/mixed-config assertions. | Replace batch cases with root/child generation cases; never construct Python runner on preflight failure. |
| `tests/integration/test_native_v3_serial_heg.py` | `ADAPT` | One fixture AST serial episode on current HEG. | One fixture Python policy through sandbox on current HEG. | `test_one_serial_native_v3_episode_uses_current_heg_backend`. | HEG remains read-only; no in-process generated source. |
| `tests/integration/test_native_v3_provider_evaluation_heg.py` | `ADAPT` | Recorded provider AST through real HEG evaluation. | Recorded provider Python source through validation/sandbox/real HEG evaluation. | `test_recorded_provider_turn_completes_one_real_heg_evaluation`. | Preserve provider artifacts and exact provenance. |
| `tests/integration/test_native_experiment.py` | `KEEP_UNCHANGED` | Shared Native provider lifecycle, artifacts, retries, resume, isolation, config propagation. | Same shared regression gate. | `test_native_output_schema_uses_app_server_supported_keywords`; `test_public_experiment_imports_do_not_load_historical_stage_modules`; `test_native_repair_persists_separate_initial_and_repair_turns`; `test_interrupt_resume_reuses_durable_turn_and_evaluation`; `test_infrastructure_retry_limit_bounds_initial_attempts`. | Predominantly Native v2/shared; add Python-specific coverage elsewhere rather than changing v2 behavior. |
| `tests/unit/test_appserver_artifact_parity.py` | `KEEP_UNCHANGED` | Frozen provider artifact bytes/tree/schema cases. | Same mandatory gate. | `test_frozen_fixture_is_byte_identical_and_structurally_complete`; `test_parity_rejects_tree_byte_compression_and_schema_changes`; `test_contract_contains_all_required_provider_turn_cases`. | Must remain byte-exact. |
| `tests/unit/test_stage3_app_server.py` | `KEEP_UNCHANGED` | App Server lifecycle/auth/isolation/events/usage/limits/artifacts. | Same. | `test_thread_start_uses_and_verifies_configured_sandbox_mode`; `test_enabled_skills_are_disabled_before_thread_start`; `test_strict_argv_private_cwd_and_lifecycle_ids`; `test_json_transport_text_is_retained_without_markdown_wrapper`; `test_timeout_interrupts_and_failed_adapter_is_not_reused`; `test_logs_persist_incrementally_on_success_and_failure`; provider request/response bound tests. | Python migration changes caller payload only; transport regressions fail the gate. |
| `tests/unit/test_stage3_isolation.py` | `KEEP_UNCHANGED` | Provider capsule argv/config/auth/environment security. | Same provider boundary. | `test_strict_app_server_args_disable_agent_capabilities`; `test_capsule_copies_only_secure_explicit_auth_and_sanitizes_environment`; `test_capsule_rejects_insecure_or_symlinked_auth`. | A separate Python execution sandbox needs new tests; do not weaken this one. |
| `tests/unit/test_stage3_artifacts.py` | `KEEP_UNCHANGED` | Transport logging, bounds, redaction, canonical finish, path safety. | Same. | `test_generation_artifacts_are_bounded_redacted_and_canonically_finished`; `test_transport_logger_enforces_event_and_payload_bounds`; `test_artifact_and_transport_paths_cannot_escape_root`. | Python semantic artifacts stay outside the frozen provider tree. |
| `tests/integration/test_stage3_cli.py` | `KEEP_UNCHANGED` | Shared doctor/auth/worker/artifact/preflight and atomic-failure behavior. | Same subset as infrastructure regression. | `test_stage3_auth_json_is_explicit_for_doctor_and_generation`; `test_appserver_doctor_artifact_equals_returned_canonical_result`; output-schema preflight; atomic inconclusive/provenance failure cases. | Stage 3 policy semantics are historical; do not reuse them as Native v3 source authority. |
| `tests/unit/test_experiment.py` | `KEEP_UNCHANGED` | Shared workspace/provider-turn/retry/usage/artifact/status behavior. | Same shared gate. | `test_full_turn_artifacts_and_canonical_source`; `test_invalid_response_retains_raw_text_and_diagnostics_without_projection`; `test_native_transport_uses_per_turn_limit_and_retry_prefix`; `test_status_is_versioned_and_read_only`. | Most assertions are Native v2; Python changes belong in Native v3-specific tests. |
| `tests/unit/test_json_io.py` | `KEEP_UNCHANGED` | Deterministic JSON gzip I/O. | Same for envelopes and metadata. | `test_json_gzip_round_trip_is_deterministic`; `test_json_gzip_rejects_uncompressed_paths`. | No source execution implication. |

The following inspected Stage 3 files are intentionally outside the Native v3
migration test inventory: `tests/unit/test_stage3_contracts.py`,
`tests/unit/test_stage3_replay.py`, `tests/unit/test_stage3_prompts.py`,
`tests/unit/test_stage3_config.py`,
`tests/integration/test_stage3_evaluation.py`, and
`tests/unit/test_experiment_evaluation.py`. They test the historical Stage
2/3 generated-ranker/proposal representation, not the current Native v3
provider lifecycle or evaluator boundary. They remain historical regressions
and are not migration donors.

## 9. GitHub issue audit and dependencies

No issue is changed by this audit. “Historical-valid” means its accepted result
remains true; it does not authorize the Python migration.

| Issues | Disposition after operator decision | Required future action |
| --- | --- | --- |
| #5 | `VALID` precedent: accepted bounded Python ranker sandbox, identity, replay, behavior signatures. | Reuse evidence, but require a new proposer-specific API/security acceptance issue. |
| #6-#17 | `VALID_HISTORICAL`: Stage 2B through Stage 7 results, including NO_GO/inconclusive decisions, remain unchanged. | No amendments; do not treat ranker evolution as Native v3 authority or alter HEG. |
| #18-#22 | `VALID`: workspace, artifact readability, dashboard, repair/resume infrastructure. | Preserve as regression gates. |
| #23-#25 | `VALID_CONTROLLING`: Native v2 baseline, donor inventory, and App Server artifact parity. | Every migration issue must pass their Native v2/parity gates. |
| #26-#29 | `VALID_HISTORICAL`, representation `SUPERSEDED`: accepted JSON AST, interpreter, selectors/actions. | Do not rewrite closed history. Treat #29 selectors/actions as donors; disconnect #26/#28 from the final path. |
| #30-#34 | `VALID`: serial evaluator, provider-to-evaluation seam, opt-in routing, deterministic cohort accounting, score evidence and interval fitness. | Adapt representation-specific wording in new migration issues; do not amend closed issues. |
| #35 (Step 13) | `SUPERSEDED` as currently written because it requires matched DSL baselines/custom interpreter and is expressly not started. | Do not edit or start it now. After migration acceptance, operator decides whether to close/replace it; it must not gate migration implementation. |
| #36 | `AMEND_AFTER_MIGRATION`: development/validation panels and promotion shortlist remain valid. | Rebase dependency from #35 to the accepted Python search issue and replace AST/DSL identity with Python identity. |
| #37 | `AMEND_AFTER_MIGRATION`: canonical episode manifests/shards remain valid. | Include Python program/source/behavior identity and sandbox protocol version. |
| #38 | `AMEND_AFTER_MIGRATION`: bounded evaluator pool remains valid. | Specify separation between evaluator/scorer processes and per-candidate Python sandbox workers. |
| #39 | `AMEND_AFTER_MIGRATION`: provider/evaluator overlap remains valid. | Stream validated Python programs, preserve deterministic manifests/equal budgets, and never execute source in provider process. |
| #40 | `AMEND_AFTER_MIGRATION`: single-writer persistence/resume/replay remains valid. | Persist exact source, canonical Python identity, fork boundary, sandbox version, and timing-independent semantic replay. Use SQLite online backup for migration tests. |
| #41 | `VALID_WITH_DEPENDENCY_UPDATE`: dual exact verification remains controlling. | Change only its predecessor after #40 is amended; keep two-verifier/apparent-zero safety contract. |
| #42 | `AMEND_AFTER_MIGRATION`: dashboard and final activation gate remain valid. | Add Python contract/sandbox/root-child/lineage metrics and require DSL disconnection/cleanup gate; v2 default switch still needs explicit GO. |
| #43 | `VALID_HISTORICAL`, representation `SUPERSEDED`: complete one-program prompt evidence. | Replace its AST output contract in the new Python contract issue; do not amend closed history. |
| #44 | `VALID_HISTORICAL`: persistent one-program turns outperformed batch in the recorded experiment. | Reuse lifecycle evidence; “one AST” becomes “one Python program” only in new issues. |
| #45 | `VALID_CONTROLLING`: compaction is `BEST_EFFORT_ONLY`. | No amendment needed; Python search cannot depend on compaction. |
| #46 | `VALID_CONTROLLING`: exact forks and bounded host Search Memory were demonstrated. | Adapt identity/family extraction and parent payload in the new lineage issue. |
| #47 | `REQUIRES_OPERATOR_DISPOSITION`: implementation exists through the audit baseline, but the open issue still specifies selected one-AST/slot-specific behavior and blocks Step 13. | Do not modify it in this task. Before migration implementation, operator must accept an amendment or replacement that records this superseding Python contract while preserving the no-Step-13 gate. |
| #48 | `VALID_HISTORICAL`, representation `SUPERSEDED`: slot-specific JSON schema won its recorded comparison. | Keep benchmark evidence; its selected schema is not the Python evolved artifact. |

The controlling Native v3 dependency chain observed on GitHub is:

```text
#23 -> #24 -> #25 -> #26 -> #27 -> #28 -> #29 -> #30 -> #31
    -> #32 -> #33 -> #34

#34 -> #43 -> #44 -> #45 -> #46 -> #48 -> #47
                                      \-------> #47

#47 -> #35 -> #36 -> #37 -> #38 -> #39 -> #40 -> #41 -> #42
```

The operator decision breaks the proposed `#47 -> #35` continuation. It does
not erase accepted history. The new migration issue chain must be inserted
after operator disposition of this audit/#47 and before any amended #36-#42
work. Step 13 remains prohibited.

## 10. Smallest sequential migration issue set

Do not create these issues until the operator accepts this audit. Each issue
starts only after explicit operator acceptance of the previous issue. Each
issue keeps Native v2 unchanged and requires, at minimum, the frozen Native v2
smoke and `make appserver-artifact-parity`.

### M1. Freeze the Python policy contract, identity, and validator

Scope:

- add the exact response envelope, `propose` signature, Python AST allow/deny
  validator, canonical source/program identity, and machine-readable
  diagnostics;
- add Python fixtures and offline validator/identity/behavior-probe tests;
- define safe API protocols/types without executing generated code; and
- add a new opt-in protocol/workspace version while leaving current branch
  routing unchanged.

Independent acceptance:

- adversarial AST corpus fails closed;
- formatting/docstring/local-name equivalents have identical program hashes;
- semantic changes differ;
- behavior probes replay byte-identically;
- Native v2 and current v3 routes/artifacts are unchanged.

Operator gate: accept the contract and security surface before any generated
Python execution is added.

### M2. Implement the isolated policy worker and safe graph API

Scope:

- implement the OS-isolated worker, framed protocol, resource limits, runtime
  guards, and fail-closed taxonomy;
- port selector/action/overlay semantics from `graph_runtime.py` behind the
  exact safe API;
- return only host-minted `RewritePlan`/`NoPlan`; and
- keep the worker reachable only from offline fixture tests.

Independent acceptance:

- selector/action parity against donor cases;
- timeout, memory/process/file/network/import/reflection/protocol attacks fail
  only the candidate and reap the worker;
- unsupported sandbox fails before execution;
- deterministic seed/API traces replay exactly;
- no generated source is imported by coordinator/provider/test process.

Operator gate: accept sandbox evidence and residual risk before connecting the
worker to scientific evaluation.

### M3. Adapt serial evaluation, scoring evidence, and exact verification

Scope:

- replace interpreter invocation with the Python runner adapter for fixture
  policies only;
- version the semantic trace with API events;
- preserve `NoPlan`, program-vs-infrastructure failure, one-score,
  interval-fitness, and exact-verification semantics; and
- add offline/replay and one real HEG integration tests without changing HEG.

Independent acceptance:

- fixed Python fixtures reproduce deterministic graph/plan/score traces;
- no-plan and invalid final state consume the correct steps;
- safe partial evidence and strict interval dominance are unchanged;
- heuristic zero cannot become `VERIFIED` without exact verification;
- existing scoring/HEG tests pass unchanged.

Operator gate: accept scientific parity before any model-generated source is
evaluated.

### M4. Generate and evaluate one Python root through the unchanged provider

Scope:

- add Python system/request prompts and exact output schema;
- parse/validate/persist one Python root source;
- run one opt-in provider-to-sandbox-to-serial-HEG smoke;
- adapt semantic artifact projection outside the frozen provider-turn tree;
  and
- preserve bounded repair, auth/preflight, usage, retry, and resumability.

Independent acceptance:

- one recorded and one authorized live response pass envelope, validation,
  sandbox, evaluator, and provenance checks;
- malformed/invalid/provider-failed turns create no scientific result;
- provider lifecycle and artifact parity tests remain unchanged;
- Native v2 remains the default and byte-faithful.

Operator gate: accept the first real Python provider/evaluator seam before
adding generations.

### M5. Add multi-generation Python lineage, roots, selection, and Search Memory

Scope:

- implement immutable generation manifests and the exact `P`/`R` allocation;
- select parents from development fitness/diversity only;
- fork exact parent turns, resend exact parent source plus compact feedback,
  and fresh-fork roots from the specification anchor;
- adapt Search Memory, Python behavior signatures, dedupe, lineage, durable
  resume, and equal-budget evaluation; and
- stop on budget or exact verified counterexample.

Independent acceptance:

- generation 0 is all roots; later generations use exact deterministic slot
  allocation, including no-valid-parent behavior;
- child fork includes specification/parent and excludes later sibling;
- root fork has no parent and receives bounded source-free Search Memory;
- duplicates, invalid programs, repairs, and failures consume the correct slot
  and budget;
- crash/resume reproduces lineage, generation manifests, provider boundaries,
  evaluation order, and scientific result.

Operator gate: accept bounded evolutionary behavior before guarded product
integration.

### M6. Integrate the guarded Python preview and complete end-to-end gates

Scope:

- connect M1-M5 behind a new explicit Python preview flag;
- adapt CLI status/telemetry and the A/B acceptance report;
- run the immutable development-panel campaign and all failure/resume/parity
  gates;
- document the issue #35-#42 amendments without starting Step 13; and
- leave the JSON DSL current route available only for rollback evidence until
  operator acceptance.

Independent acceptance:

- complete provider -> Python -> sandbox -> serial panel -> selection/fork ->
  next generation -> exact verification path;
- equal graph/seed/evaluation budgets and immutable manifests;
- all status/scientific/infrastructure distinctions are correct;
- full Native v2 smoke, artifact parity, replay, crash/recovery, sandbox,
  scoring, HEG, and exact-verifier suites pass;
- no current DSL module is imported by the Python execution path.

Operator gate: explicit GO/NO_GO on the Python path. A GO authorizes cleanup,
not Step 13 or a default switch.

### M7. Disconnect and remove the superseded DSL path

Scope:

- remove final-path imports/routing for `SUPERSEDED` files;
- remove `DELETE_LATER` schemas/prompts/fixtures/experiments/tests;
- keep only explicitly documented historical reports/evidence;
- update open issue dependencies and terminology as accepted by the operator;
  and
- rerun the complete M6 gate after removal.

Independent acceptance:

- repository import/search proof finds no production dependency on the custom
  DSL, IR compiler, or interpreter;
- Python end-to-end and replay artifacts remain identical to pre-cleanup M6
  evidence;
- Native v2 remains unchanged and default;
- artifact parity and all scientific/sandbox/verification gates pass.

Operator gate: accept physical cleanup. Step 13 still does not begin unless the
operator separately authorizes a replacement issue.

## 11. Required operator decisions before implementation

Acceptance of this audit should explicitly decide:

1. whether #47 is amended or replaced before M1;
2. whether the exact 32 KiB/2,000-node source limits and 250 ms per-call limit
   are accepted for the first implementation;
3. whether the proposed label-opaque graph view and donor-derived safe API are
   sufficient, or must be narrower;
4. whether `P // 4` fresh roots per later generation is accepted;
5. whether the Python path receives a new experiment/workspace protocol ID and
   rejects current v3 workspaces, as recommended; and
6. whether #35 is later closed as superseded or rewritten only after M7.

Until those decisions and the sequential issue gates are accepted, the current
Native v3 and Native v2 behavior must remain unchanged.

STOP — waiting for operator acceptance
