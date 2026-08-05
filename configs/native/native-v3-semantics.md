# Native v3 interpreter and graph semantics

Interpreter protocol: `native_v3_graph_interpreter_v1`

Graph runtime protocol: `native_v3_graph_runtime_v1`

The interpreter executes one statically validated Native v3 program against a
private copy-on-write graph overlay. It does not mutate the input `GraphState`.
It returns exactly one canonical v2 `RewritePlan`, one `NoPlan`, or one
`ProgramFailure`.

## Typed values

Runtime values are booleans, bounded exact integers, normalized rational
numbers, printable strings, and opaque typed graph references:

- `VertexRef` and `VertexSetRef`;
- `EdgeRef` and `EdgeSetRef`;
- `NonEdgeRef`;
- `PathRef`;
- `MatchingRef`.

The AST can bind and pass references but cannot inspect their raw vertex
labels. Selectors return complete semantic tie sets or deterministic,
uniformly sampled reservoirs of at most 64 items. Implementations enumerate
sets canonically before sampling; relabeling is assessed through canonical
graph classes or frozen seed distributions rather than raw reference values.

## Graph selectors

All selectors observe the current private overlay, including earlier actions
in the same successful branch:

- degree extreme and exact degree class;
- sampled vertex and edge witness-load extreme;
- articulation and bridge risk;
- bounded distance bands;
- all removable edges;
- all legal non-edges, non-edges from a selected vertex, and local cycle-risk
  extremes;
- length-two paths;
- legal seeded 2-, 3-, and 4-switch reconnections over vertex-disjoint source
  edges.

Witness-load providers receive only the current immutable overlay
`GraphState`. Their results are cached by canonical overlay edge tuple.

## Rewrite actions

`add_edge`, `remove_edge`, `relocate_endpoint`, `k_switch`, `edge_fanout`, and
`edge_fold` mutate only the private overlay. Every reference is checked at the
point of use, so stale or wrong-type references fail closed. Loops,
out-of-range vertices, duplicate edges, and existing-edge additions are never
allowed.

Temporary disconnection or degree below three is permitted inside the private
overlay. This is necessary for ordered multi-action rewrites. It is never
permitted in an emitted result.

## Control flow and transactions

- `let` creates an immutable lexical binding for its body.
- `block` evaluates children in order.
- `if` evaluates one branch.
- `repeat` executes its non-terminal body exactly `count` times.
- `choose` and seeded `pick` use deterministic integer-only randomness.
- `try` evaluates branches in order. A catchable failure restores the graph
  overlay and lexical bindings before the next branch.
- `emit` validates and canonicalizes the net rewrite.
- `no_plan` terminates without a rewrite.

Only `NO_MATCH`, `LOCAL_PRECONDITION_FAILED`, `ILLEGAL_FINAL_STATE`, and
`NO_EFFECT` are catchable. Invalid AST structure, runtime type errors,
resource-limit exhaustion, and interpreter faults are program failures.
Actions, selector charges, and random draws consumed by a failed branch are
not refunded.

## Emit boundary

`emit` computes sorted net removed and added edges relative to the invocation
input. It rejects:

- no-op output;
- order changes;
- loops, duplicates, invalid endpoints, or stale references;
- disconnected output;
- minimum degree below three;
- independent net-added or net-removed budget overflow.

The resulting v2 `RewritePlan` is then passed to the injected existing backend
`apply_rewrite` boundary. The backend-returned graph must exactly equal the
private overlay and retain the input order. Backend rejection becomes
`ILLEGAL_FINAL_STATE`; the scorer and backend implementation are not replaced.

Gross action count and net added/removed edges have independent dynamic limits.
Node/expression steps, repeat iterations, choices, bindings, selector calls,
selector cost, random draws, and numeric widths retain their separate Step 06
limits.

## Deterministic randomness

SHA-256 derives each SplitMix64 seed from length-prefixed typed components:

1. random protocol ID;
2. interpreter protocol ID;
3. canonical program hash;
4. episode fixture ID;
5. step index;
6. invocation ordinal;
7. dynamic AST path.

Repeat indices, selector reservoir positions, matching attempts, and shuffle
positions are encoded in the dynamic path. Uniform draws use rejection
sampling; weighted draws use positive exact integer weights with a signed
64-bit total bound.

## Deliberate Step 07 limits

This step provides graph construction and host validation only. It does not
integrate a scorer, provider-generated execution, evaluator, experiment
scheduler, persistence, or multiprocessing. HEG remains behind the existing
backend boundary.
