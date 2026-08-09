# Native v3 Step 08 serial evaluator

Protocol: `native_v3_serial_evaluator_v1`

`evaluate_serial_program` executes one validated fixture program over one
deterministic graph trajectory. It is an internal Python API; Step 08 adds no
CLI route, provider call, cohort, panel, or concurrency.

## Scientific loop

The evaluator receives the existing `GraphBackend`, a `ValidatedProgram`, and
one `SerialEpisodeConfig` containing order, graph seed, policy seed, horizon,
witness cap, and episode ID.

Execution order is normative:

1. Generate and host-validate the seed graph.
2. Obtain its authoritative backend `GraphScore`.
3. Submit the initial graph to the existing counterexample inspection boundary
   only when the authoritative total is zero.
4. For every trajectory step, invoke the program once against the incumbent.
5. A `NoPlan`, including an illegal final overlay, consumes the step without
   applying or scoring a nonexistent candidate.
6. A rewrite outcome retains the candidate produced by the interpreter's
   existing backend `apply_rewrite` call. The evaluator does not apply it a
   second time.
7. Obtain exactly one authoritative candidate score with no alternate scorer
   or local fallback.
8. Submit a legal apparent-zero candidate to the same counterexample boundary.
9. Accept only when
   `candidate_score.ordering_key < incumbent_score.ordering_key`.

Equal and worse scores are rejected. A program or scoring failure terminates
the trajectory fail-closed. The caller owns the backend and must close it.

## Semantic trace

The result records:

- initial and terminal backend state/canonical identities;
- concrete selector populations, picks, lexical bindings, actions, fallback
  events, and terminal interpreter outcome;
- canonical rewrite edges;
- incumbent and candidate authoritative scores;
- strict-improvement acceptance;
- counterexample inspection decision and statuses.

The trace contains no timing values. Its canonical JSON projection is hashed
under `mforge-native-v3-serial-trace\0`, so identical inputs and backend
behavior produce identical records and hashes.

## Manual smoke

Run from the dedicated worktree:

```console
uv run pytest tests/integration/test_native_v3_serial_heg.py -q
```

Expected result: one test passes after a complete horizon-one trajectory using
the current sibling HEG backend. The test performs no provider call and writes
no production experiment workspace.
