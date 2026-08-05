# Native v3 control-flow semantics

Protocol: `native_v3_synthetic_interpreter_v1`

This document defines the runtime semantics of a statically validated Native v3
program. The Step 06 interpreter is intentionally isolated: it runs only against
the `SyntheticFixture` protocol and does not read or mutate a HEG graph, an
experiment, a provider, or a v2 policy.

## Invocation boundary

An invocation receives a `ValidatedProgram`, a synthetic fixture, a complete
`ProgramContext`, exact integer graph-feature inputs, optional scalar overlay
values, and `InterpreterLimits`. The caller's inputs are never mutated. Actions
operate on a private overlay. A successful invocation returns exactly one
`emit` or `no_plan` outcome; a program failure returns no outcome.

Runtime values are booleans, bounded exact integers, normalized
`fractions.Fraction` rationals, printable strings, typed synthetic references,
and ordered typed selections. No floating-point arithmetic is performed.

## Expressions and control flow

- `let` evaluates once and adds an immutable lexical binding for its body.
- `block` evaluates children in order.
- `if` evaluates only its selected branch.
- `repeat` executes its non-terminal body exactly `count` times. The iteration
  index is part of the dynamic AST path.
- `choose` selects one branch using positive integer weights and the normative
  deterministic random protocol.
- `selector` calls the typed synthetic fixture boundary. Its declared static
  result type must match the runtime `Selection`.
- `pick` supports `require_singleton`, `seeded_uniform`, and
  `seeded_weighted`. An empty population is `NO_MATCH`; a non-singleton
  `require_singleton` is `LOCAL_PRECONDITION_FAILED`.
- `apply` evaluates and type-checks all arguments before the fixture action
  mutates the private overlay.
- `emit` asks the fixture to validate the overlay and then returns it.
- `no_plan` terminates with its declared reason.

Every evaluated node and expression consumes a dynamic step. Repeat iterations,
choices, bindings, selector calls, selector cost units, actions, random draws,
integer width, and rational denominator width have separate dynamic bounds.
Counters are monotonically consumed for the whole invocation.

## Transactional fallback

`try` evaluates branches in source order. Before each branch it snapshots the
private overlay and lexical bindings. A catchable branch failure restores both
snapshots completely and continues with the next branch.

The closed catchable class is:

- `NO_MATCH`
- `LOCAL_PRECONDITION_FAILED`
- `ILLEGAL_FINAL_STATE`
- `NO_EFFECT`

No other exception is catchable. In particular, invalid AST structure, runtime
type mismatches, dynamic budget exhaustion, and interpreter or fixture faults
terminate the program. Consumed budget counters and random draws are not
refunded when a branch rolls back.

If the final branch also fails catchably, the invocation produces `NoPlan`.
`LOCAL_PRECONDITION_FAILED` maps to the public `NO_MATCH` reason; the other
codes retain their names. An uncaught catchable failure also restores the
invocation's initial overlay before producing `NoPlan`.

## Deterministic randomness

Randomness uses `native_v3_splitmix64_v1`. SHA-256 derives a 64-bit seed from
length-prefixed typed components:

1. random protocol ID;
2. interpreter protocol ID;
3. canonical program hash;
4. synthetic episode fixture ID;
5. step index;
6. invocation ordinal;
7. dynamic AST path.

Uniform draws use rejection sampling over SplitMix64 output, so modulo bias is
not permitted. Weighted choices use positive exact integer weights whose sum is
at most `2**63 - 1`. The protocol and derivation vectors are frozen in
`tests/unit/test_native_v3_interpreter.py`.

## Deliberate Step 06 limits

The fixture boundary supplies only typed synthetic selector results, integer
weights, action effects, and final-overlay validation. Production graph
selectors, graph rewrites, HEG integration, provider calls, evaluation,
experiment scheduling, and artifact persistence are outside this step.
