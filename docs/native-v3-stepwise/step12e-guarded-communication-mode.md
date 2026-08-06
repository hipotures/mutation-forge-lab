# Native v3 Step 12E: guarded communication mode

Step 12E integrates the selected App Server communication mode into the
Native v3 preview. Native v2 is unchanged and the Step 11 four-program batch
remains the explicit rollback.

## Operator decision

The operator selected `persistent_single_ast` as the preview default in
[issue #47 comment 5208460774](https://github.com/hipotures/mutation-forge-lab/issues/47#issuecomment-5208460774).
The decision uses the accepted Step 12B through 12D evidence:

- the persistent arm reached its first valid AST in 57.438 seconds, compared
  with 71.367 seconds for fresh threads, and used 689.5 fewer total tokens per
  valid program;
- the same real four-brief batch reference produced zero valid programs;
- Step 12D proved exact inclusive fork boundaries, bounded host Search Memory,
  authoritative duplicate rejection, and a valid same-fork repair;
- Step 12C classified compaction as `BEST_EFFORT_ONLY`.

The decision is explicit and is not recomputed from runtime measurements.

## Configuration

The selected preview mode is:

```toml
[v3]
communication_mode = "persistent_single_ast"
```

It is also the default when the field is omitted. The rollback is:

```toml
[v3]
communication_mode = "multi_program_batch"
```

Changing the mode changes the immutable configuration identity, so an
existing workspace cannot be reinterpreted under another mode.

## Selected protocol

Each bounded epoch creates one durable specification thread and sends the
complete executable contract once. Exactly two durable worker threads are
forked at the inclusive specification `lastTurnId`. Slots alternate between
the workers; only one turn is active at a time.

Every program turn:

1. receives one Step 12A direct-program `outputSchema`;
2. receives the current bounded `SearchMemoryV1`, without a full program AST;
3. returns at most one AST;
4. is validated immediately by the host;
5. is rejected when its canonical hash or behavior signature is already in
   Search Memory;
6. receives at most one repair turn on the same worker thread;
7. publishes its semantic attempt and slot record before a later slot starts.

The specification anchor, worker thread IDs and rollout paths, exact fork
parent turn, every program turn ID, prompt and contract hashes, usage, host
validation, duplicate result, and semantic program record are persisted
outside transport artifacts. The same deterministic serial evaluator,
interval fitness, cohort threshold, and program selection used by the rollback
remain unchanged.

No automatic compaction is used. A future context rotation must create a fresh
fork from the specification anchor and inject bounded Search Memory.

## Resume and artifacts

The isolated capsule is host-owned until generation finishes. After every
completed slot, the next slot, both durable worker identities, usage, accepted
records, and Search Memory identity are written atomically. A replacement App
Server process resumes the selected worker with its persisted thread ID and
server-returned rollout path. Authentication and HEG preflight still occur
before scientific work; a preflight failure remains resumable.

The bootstrap, both fork RPCs, and every completed program turn have the exact
standard 16-file App Server artifact set. Semantic records are written below
`native-v3-output/epoch-0000/program-attempts` and `program-records`; no
semantic file is added to a provider-turn prefix and no rollout copy is added.

## Production-preview gate

The bounded real gate uses the same four Step 12A briefs in
`workspace/step12b_abc_medium_010`:

| Arm | Valid programs | First valid AST | Total tokens per valid program |
| --- | ---: | ---: | ---: |
| Persistent single-AST | 2/4 | 57.438 s | 11,606.5 |
| Multi-program batch rollback | 0/4 | unavailable | unavailable |

This gate selected the communication mechanism, not scientific quality. The
integrated preview retains Step 12D's two-worker fork and Search Memory
boundaries while using the unchanged production evaluator.

## Validation

```text
uv run ruff check src/mutation_forge/native_v3 src/mutation_forge/stage3/isolation.py tests/unit/test_native_v3_preview.py tests/unit/test_native_v3_experiment.py tests/integration/test_native_v3_route.py
uv run mypy src/mutation_forge/native_v3 src/mutation_forge/stage3/isolation.py
uv run pytest -q tests/unit/test_native_v3_preview.py tests/unit/test_native_v3_experiment.py tests/integration/test_native_v3_route.py
uv run pytest tests/unit/test_native_v3*.py tests/integration/test_native_v3_route.py -q
uv run pytest tests/unit/test_native_v2_smoke.py tests/integration/test_native_experiment.py tests/unit/test_native_resume.py tests/unit/test_native_selection.py tests/unit/test_native_progress.py -q
make appserver-artifact-parity
```

The focused tests cover explicit selected and rollback routing, the selected
default, authentication failure before provider construction, two exact
specification forks, eight direct unique AST turns, immediate semantic
publication, bounded Search Memory without ASTs, exact artifact parity, and
durable process replacement through persisted thread identities.

## Manual operator command

Use a fresh `exp_id`:

```console
uv run mforge experiment run --config experiment.toml --json
```

Set `communication_mode = "multi_program_batch"` with another fresh `exp_id`
to exercise the rollback.

STOP — waiting for operator acceptance. Step 13 has not started.
