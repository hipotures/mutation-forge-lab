# Native v3 ordinary-Python M3 fixture integration

## Scope

This report records the fixture-only M3 integration authorized by issue #52.
It does not activate the ordinary-Python experiment protocol, contact a model,
generate a program, select parents, create generations, or change the existing
JSON-DSL route.

The evaluated path is:

```text
checked-in Python fixture
  -> M1 source parser and fail-closed AST validator
  -> M2 isolated worker and safe graph API
  -> representation-independent serial evaluator
  -> current HEG C++ component-evidence scorer
  -> exact-rational interval utility and fitness
  -> existing exact-verification submission boundary
```

## Frozen inputs and protocols

- Fixture: `tests/fixtures/native_v3_python_m3/add_edge.py`
- Fixture manifest:
  `mforge.native.python_m3_fixture_manifest.v1`
- Python serial evaluator:
  `native_v3_python_serial_interval_evaluator_v1`
- Python semantic API trace:
  `mforge.native.python_semantic_trace.v1`
- Order: `30`
- Graph seed: `101`
- Policy seed: `17`
- Horizon: `1`
- Witness cap: `64`
- Forbidden lengths: the immutable tuple returned by the authoritative HEG
  backend for order 30
- Propose wall time: `1 s`
- Worker address-space limit: `256 MiB`
- Worker invocation-eligibility age boundary: `60 s`, with transparent
  between-invocation rotation

The remaining M2 capability and resource caps are unchanged.

## Scientific semantics

The ordinary-Python adapter and the historical JSON-DSL wrapper use one shared
serial scientific core. The JSON-DSL wrapper retains its existing protocol ID
and artifact shape.

For Python fixtures:

- exactly one isolated `propose` invocation occurs per serial step;
- `NoPlan` consumes the step and does not score a nonexistent candidate;
- only a host-minted rewrite whose candidate is accepted by the authoritative
  backend can reach scoring;
- candidate-invalid rewrites use the typed `InvalidRewriteError` boundary and
  become `NoPlan("ILLEGAL_FINAL_STATE")`;
- unexpected backend, scorer, sandbox, or protocol failures remain
  infrastructure failures and produce no scientific result;
- candidate exceptions, invalid returns, and propose timeouts become
  deterministic `PROGRAM_FAILURE` results with worst fitness `[0, 0]`;
- initial and candidate evidence use the episode's immutable forbidden-length
  tuple;
- overlapping non-point evidence is expanded under the existing locked HEG
  budgets, and acceptance still requires
  `candidate.upper < incumbent.lower`;
- an apparent heuristic zero is only submitted to the existing exact
  verification boundary and is never marked `VERIFIED` by the heuristic;
- timing, PID, RSS, and transport randomness are absent from scientific and
  behavior identities.

Canonical Python program identity remains name-preserving. Behavior identity is
computed separately from timing-free safe-API events and normalized outcomes;
program hashes embedded in rewrite metadata are deliberately excluded from the
behavior signature.

## Current HEG integration result

Command:

```bash
uv run pytest -q \
  tests/integration/test_native_v3_serial_heg.py::test_fixture_python_runs_through_sandbox_and_current_heg_scoring
```

Result: `1 passed in 0.13s`.

The sibling HEG repository was clean before and after the run at:

```text
27cbec9c2307b6ea5f936f858821d11d808b68f3
```

The fixture produced a host-minted rewrite, scientifically bounded initial,
candidate, and terminal component evidence, and a valid conservative fitness
interval. The test records:

```json
{
  "provider_turns": 0,
  "model_turns": 0,
  "app_server_calls": 0
}
```

No Python/reference scorer fallback exists on this path.

## Offline coverage

The checked-in corpus covers:

- explicit `NoPlan`;
- a legal add-edge rewrite;
- no-effect emission;
- legal or final-state-rejected removal;
- invalid return;
- propose timeout;
- M1-invalid import.

Focused tests additionally cover deterministic replay, separate program and
behavior identities, one authoritative candidate score, mismatched backend
candidate rejection, safe partial-evidence dominance, overlapping interval
rejection, exact-verifier submission, immutable forbidden lengths, sandbox
failure, backend failure, and the distinction between typed invalid rewrites
and untyped infrastructure exceptions.

## Deliberate exclusions

M3 remains fixture-only and inactive. It contains no App Server call, prompt,
provider turn, model-generated source, Search Memory, parent selection,
thread/fork, generation allocation, development-panel selection, preview
routing, or DSL disconnection. Native v2 remains the default. M4 / issue #53
has not started.

## Acceptance gates

- Ruff: passed.
- mypy: passed for 162 source files.
- Focused DSL/scoring/counterexample/M1/M2/M3/backend tests:
  `239 passed in 2.90s`.
- Current HEG fixture integration: `1 passed in 0.13s`.
- App Server artifact parity: four canonical cases, 131 files, seven parity
  tests passed.
- Native v2 real-provider smoke: passed with one completed turn, zero repairs,
  a final response, exact final usage, and complete provider artifacts.
- Full suite: 1,008 collected; 981 passed; 25 failed; two errors; one warning.
  The failure and error node-ID sets are exactly equal to the accepted
  pre-M3 baseline at
  `f14769a43db8fd127ec7b3fd9108a956dbd5a7f0`; M3 adds 11 passing tests and no
  new failure or error.
- `experiment.toml`: unchanged.
