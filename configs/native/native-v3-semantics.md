# Native v3 normative semantics

This document is normative for `mforge.native.program.v3`. The JSON schemas
define transport shape. The selector and action registries define the available
host operations. This document defines execution, failure, replay, and
scientific semantics. An implementation that disagrees with any of these four
locked assets must fail preflight.

## Program identity

The provider response is not canonical. The host retains four separate values:

- `program_json_raw`: the exact decoded string returned for one slot;
- `program_ast`: the parsed and statically validated object;
- `program_json_canonical`: `native_v3_canonical_json_v1` serialization;
- `program_hash`: the SHA-256 identity below.

Canonical JSON uses printable ASCII strings, ASCII/UTF-8 bytes, no BOM, no
whitespace, lexicographically sorted object keys, decimal integers, and the
JSON literals `true`, `false`, and `null`. It rejects duplicate keys,
floating-point numbers, non-finite numbers, and non-ASCII strings. No Unicode
normalization is performed because non-ASCII strings are outside the language.

The program hash is:

```text
SHA256(
  b"mforge-native-v3-program\0" +
  b"mforge.native.program.v3" +
  b"\0" +
  program_json_canonical_bytes
)
```

All integer literals are signed 64-bit values. Rational values are objects with
a signed 64-bit numerator and a positive uint32 denominator. They must already
be reduced. JSON floats are never accepted. Choose weights are positive uint32
values and their sum must fit in signed 64 bits.

## Invocation

One invocation has the conceptual signature:

```text
propose(ctx, graph_features, policy_seed, episode_id) -> RewritePlan | NoPlan
```

The AST cannot access files, imports, network services, host functions, archive
contents, validation-panel cases, raw graph storage, or raw vertex labels.
`ctx` and `graph_features` contain exactly the values in
`native-v3-context.schema.json`.

Scalar AST expressions may read only these context fields:

```text
step_index: Int
horizon: Int
acceptance_profile_id: String
stagnation_steps: Int
exploration_window_index: Int
accepted_rewrites: Int
accepted_non_improving_rewrites: Int
consecutive_non_improving_rewrites: Int
witness_cap: Int
```

An absent `exploration_window_index` is exposed to the AST as `-1`.

Scalar graph features available to expressions are:

```text
order: Int
edge_count: Int
minimum_degree: Int
maximum_degree: Int
```

The remaining context-schema fields belong to the host feature snapshot and
selector implementation. They are not untyped maps available to the AST.
Selectors are the only way to obtain `VertexRef`, `EdgeRef`, `NonEdgeRef`,
`Path2Ref`, or `MatchingRef`.

Policies are label-oblivious: no raw vertex identifier is an expression value,
comparison key, weight, or strategic feature. Replay is exact for the same
labeled graph and seed. Relabeling conformance compares authoritative canonical
forms of resulting graphs or distributions across seeds; pathwise selection of
one vertex from a symmetric class is not promised.

## Expressions

Literal booleans, integers, printable ASCII strings, normalized rationals, and
lexically bound references are pure.

`add`, `subtract`, `multiply`, `minimum`, and `maximum` accept numeric operands
and use exact integer or rational arithmetic. Comparisons return `Bool`.
`and`, `or`, and `not` return `Bool`. `exists` evaluates its operand and returns
false only for a catchable empty-selection branch failure or an empty selector
population.

`selector` invokes the exact registry entry against the current private
overlay. `pick` accepts a selector population:

- `require_singleton` fails with `NO_MATCH` unless its sample has one item;
- `seeded_uniform` uses the locked unbiased 64-bit choice kernel;
- `seeded_weighted` uses positive integer weights and rejection sampling, never
  modulo reduction.

The allowed weight features are `uniform`, `degree`, and `inverse_degree`,
subject to the selected reference type. Witness loads are available through
the dedicated costed witness-load selectors, so weighted picks cannot trigger
unbudgeted witness sampling through a cheaper selector.

## Nodes and control flow

`block` executes children in order. A terminal child ends the invocation.

`let` evaluates one expression and creates one lexical binding visible only in
its body. Branches receive copies of the lexical environment.

`if` evaluates exactly one branch.

`choose` selects exactly one branch from locked positive integer weights with
the versioned deterministic random kernel.

`repeat` executes its body exactly `count` times. A repeat body cannot be
terminal. Repeat indices participate in deterministic random invocation paths.

`try` records the entry overlay and lexical environment. Each branch starts
from those exact values. It catches only:

```text
NO_MATCH
LOCAL_PRECONDITION_FAILED
ILLEGAL_FINAL_STATE
NO_EFFECT
```

It does not refund selector calls, selector cost, gross actions, or random draw
ordinals already consumed. Program/type/resource failures are never catchable.

`apply` executes one registered primitive. `emit` terminates with the canonical
net rewrite. `no_plan` terminates with its declared reason. Static validation
requires every reachable path to terminate exactly once.

## Private overlay and invariants

Selectors and subsequent actions observe the current invocation-private
overlay. The following invariants hold after every primitive:

- the vertex set and graph order are fixed;
- every reference points to a vertex in that set;
- no loop or duplicate edge exists;
- bindings and transaction memory remain internally consistent;
- static and runtime resource budgets are not exceeded.

Connectivity and minimum degree are final `emit` invariants, not intermediate
overlay invariants. Private intermediate overlays may temporarily be
disconnected or have degree below three. They are never scored, verified,
persisted as graph states, or exposed to another policy invocation.

At `emit`, the host requires a simple connected graph of the same order with
minimum degree at least three. It computes sorted net edge differences from
the invocation input. Add-then-remove does not bypass gross-action limits.
Both gross primitive executions and net removed/added edges are bounded. An
empty net difference is `NO_EFFECT`.

## Action semantics

The action registry is authoritative for types and charged cost. Effects are:

- `add_edge(NonEdgeRef uv)`: add absent `uv`;
- `remove_edge(EdgeRef uv)`: remove present `uv`;
- `relocate_endpoint(EdgeRef uv, VertexRef keep, VertexRef new)`: remove `uv`
  and add `keep-new`, where `keep` is one endpoint;
- `k_switch(MatchingRef)`: atomically replace the registered disjoint source
  matching by its legal target matching for `k` in 2, 3, or 4;
- `edge_fanout(EdgeRef uv, VertexRef w)`: remove `uv`, add `uw` and `vw`;
- `edge_fold(Path2Ref u-w-v)`: remove `uw` and `wv`, add absent `uv`.

Failed local preconditions raise catchable `LOCAL_PRECONDITION_FAILED`.
Reference type corruption, unknown operators, or budget escape are
uncatchable `PROGRAM_FAILURE`.

## Selector execution and tie-sets

The selector registry is authoritative for signatures and uncached costs.
Every query observes the current overlay. An identical query on an identical
overlay is memoized within one invocation and costs one unit. An uncached query
costs its full registry value.

Witness-load selectors obtain their bounded HEG witness sample lazily for the
current overlay. The sample is shared by witness selectors on that unchanged
overlay and is recomputed only after the overlay changes. Before sampling, the
host applies a seeded unbiased vertex permutation and maps witness loads back
to the policy overlay. Thus raw vertex numbering does not choose which bounded
sample is exposed. It is a policy feature and never substitutes for
authoritative C++ score evidence.

Static analysis rejects a worst-case selector cost above 128 when it can prove
the bound. Runtime execution also enforces:

```text
max_selector_cost_units_per_propose = 128
```

Runtime excess is an uncatchable `PROGRAM_FAILURE`.

A selector tie-set is all candidates sharing the same label-oblivious selector
key; it is not claimed to be an automorphism orbit. If the population exceeds
64, versioned uniform reservoir sampling scans the entire population and
retains 64 candidates. It never takes a raw-label prefix. The selection record
stores population size, sample size, seed, path, and selected-reference hash.
The output distribution, rather than a particular pathwise reference, is
invariant to enumeration and relabeling.

## Deterministic randomness

`native_v3_splitmix64_v1` derives a big-endian 64-bit seed from a
domain-separated SHA-256 digest of length-prefixed protocol, program, episode,
policy-seed, step, AST-path, repeat-index, invocation-ordinal, and draw-ordinal
values. SplitMix64 and rejection sampling implement bounded and weighted
choice. Every attempted draw consumes a unique ordinal, including rejected
rejection-sampling draws.

The exploration controller is `native_v3_stagnation_8_window_4_v1`.
After eight consecutive non-improving steps, the next four policy steps use
temperatures:

```text
1/16, 1/32, 1/64, 1/128
```

No-plan and proposal-timeout steps consume a window position. A strict
improvement ends the window. The tabu key is the labeled graph content hash;
tabu blocks only non-improving exploration acceptance, never strict
improvement.

For exact energies:

```text
delta = (E_proposal - E_incumbent) / (E_max - E_min)
p = exp(-delta / temperature)
```

The Decimal/fixed-context acceptance protocol converts `p` to an integer
threshold in `[0, 2^64]`. Threshold `2^64` always accepts; otherwise a move
accepts exactly when `SplitMix64(seed) < threshold`. Native `float`,
`math.exp`, and `random.random` are not part of the protocol. Frozen test
vectors are normative.

## Score evidence and fitness

Native v3 valid-graph ordering is exactly:

```text
(total_capped_witnesses, weighted_penalty, edge_count)
```

Mixed-radix encoding preserves that lexicographic order. It does not reuse the
historical full `GraphScore.ordering_key`.

Each forbidden-length component records observed count, sound lower and upper
bounds, status, node budget, nodes visited, wall time, attempt kind, and
content-addressed backend identity. Only exact, cap-saturated,
budget-exhausted, and safe-partial timeout states contribute scientific
intervals. A proposal timeout without safe partial rejects that proposal and
keeps the incumbent. An initial timeout without partial yields `[0,1]` utility
for all `horizon + 1` curve points. Repeated infrastructure failure makes the
candidate evaluation infrastructure-inconclusive without fitness. Contract
violations fail the experiment closed.

For each graph order the energy protocol fixes `E_min` and `E_max` and defines:

```text
utility(E) = 1 - (E - E_min) / (E_max - E_min)
utility([E_low,E_high]) = [utility(E_high), utility(E_low)]
```

Episode AUC is the exact rational mean of the best-so-far utility intervals at
the initial state and every horizon step. Per-order episode means are averaged
with equal order weight. Development selection maximizes the conservative
lower bound, then exactness, narrower interval, upper bound, and program hash.
Midpoints are never used.

## Episode and epoch terminal states

Program/episode transitions are normative:

| Event | Consequence |
| --- | --- |
| empty selector or local precondition failure | catchable branch failure |
| illegal final graph | fallback branch or `NoPlan(ILLEGAL_FINAL_STATE)` |
| unknown node/type mismatch/resource escape | `PROGRAM_FAILURE` |
| proposal score timeout without partial | reject proposal, retain incumbent, continue |
| initial score timeout without partial | `INCONCLUSIVE_TIMEOUT` with full interval curve |
| first infrastructure failure | one residual reschedule on another worker |
| repeated infrastructure failure | no fitness; infrastructure-inconclusive candidate |
| contract violation or inconsistent evidence | global fail-closed |

An epoch has eight preallocated slots. Canonical duplicates retain all slot
aliases and lineage but count once:

```text
8 unique scientifically valid programs -> COMPLETE
4..7                                 -> DEGRADED
0..3                                 -> INCONCLUSIVE and safe stop
```

The parent/archive snapshot is frozen and versioned before provider work.
Adaptive development results are comparable only under identical development
manifest and protocol hashes. Retained parents and four fixed baselines are
evaluated once per current manifest/protocol and cached.

After development, the host persists a deterministic frozen promotion
shortlist of at most four canonical programs before validation starts. Every
shortlisted program completes the same sealed validation manifest unless it is
a canonical duplicate or fails a hard non-score eligibility rule. Development
bounds never prune a disjoint validation panel. `validated_global_best` means
best among programs that actually completed that identical validation
manifest and protocol.

## Streaming, persistence, and replay

Provider responses, validation, and episode shards use bounded queues.
Evaluation starts with the first valid AST. One AST fans out into deterministic
episode or microshard tasks over order, graph seed, policy seed, and panel.
Provider calls continue while CPU tasks run. Before the first AST, only missing
baseline and retained-parent microshards may use `W-1` workers. Candidate work
starts on the reserved worker immediately and takes dispatch priority at every
microshard boundary; auxiliary tasks are not force-killed.

The one persistence owner and one SQLite writer connection commit semantic
records idempotently through a bounded queue of `2 * evaluator_workers`. A
different payload for the same semantic key is a fail-closed conflict. A
durably committed terminal episode is never rerun. Computed but uncommitted
work may be repeated. Provider batches are durable recovery boundaries and
retain every slot result; a frozen repair resumes without repeating its
initial call.

Semantic checkpoints include manifests, canonical programs, results, evidence,
fitness, shortlist, selection, lineage, archive state, and terminal statuses.
They exclude timestamps, durations, worker identities, queue arrival order,
and observational telemetry. Provider and evaluator delay permutations must
produce identical semantic checkpoint hashes.

## Counterexample verification

Every legal apparent zero from an initial graph, generated program, or baseline
is durably persisted before the episode becomes terminal and before ordinary
acceptance filtering. Jobs are deduplicated by:

```text
(graph_hash, verification_protocol_id)
```

The locked supervisor profile is:

```text
concurrency = 1
unique_graph_queue_capacity = 16
primary_timeout = 600 seconds
primary_memory = 4 GiB
independent_timeout = 600 seconds
independent_memory = 4 GiB
```

The primary and independent verifiers run sequentially in isolated processes.
The independent verifier runs only after complete primary `VERIFIED`. A
certificate is written only after both return complete `VERIFIED`.
`INCONCLUSIVE` and `CONFLICT` never mean success. A full queue applies
backpressure to new scoring; apparent-zero events are never dropped.

## Structural reachability lemma

For fixed order `n`, let `G` and `H` be simple connected graphs with minimum
degree at least three. Repeated legal `add_edge` steps transform `G` into
`K_n`; every committed intermediate remains connected and cannot lower a
degree. Starting at `K_n`, remove edges in `K_n \ H`. Every intermediate
contains `H` as a spanning subgraph, so it remains connected and every degree
is at least its degree in `H`. Therefore add/remove primitives make this valid
fixed-order graph space structurally reachable. This is a mathematical lemma;
small exhaustive and randomized tests are supporting checks, not its proof.
