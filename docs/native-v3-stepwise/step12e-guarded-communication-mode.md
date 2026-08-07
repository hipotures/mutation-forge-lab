# Native v3 Step 12E: guarded communication mode

Step 12E integrates the selected App Server communication mode into the
Native v3 preview. Native v2 is unchanged and the Step 11 four-program batch
remains the explicit rollback.

## Operator decision

The operator selected `persistent_single_ast` with the `slot_specific`
model-facing contract after accepting issue #48 at commit `c7dd740`. The final
decision is recorded in
[issue #47 comment 5210727373](https://github.com/hipotures/mutation-forge-lab/issues/47#issuecomment-5210727373).
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

The mode has one fixed model-facing contract, `slot_specific`. It is reported
in status and semantic artifacts, but is not a separate user option. The
unchanged v3 default and rollback is:

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

The selected preview publishes four program slots in four separate program
turns. Every program turn:

1. receives the accepted nonrecursive, brief-specific `slot_specific`
   `outputSchema`;
2. receives a semantic projection of the current bounded `SearchMemoryV1`,
   without a full program AST or any cryptographic identity;
3. returns one bounded model-facing representation;
4. is deterministically compiled into the existing AST and validated
   immediately by the host;
5. is evaluated with the exact `ProgramContract` used during compilation,
   including relation-safe typed relocation and fanout references;
6. is rejected when its canonical hash or behavior signature is already in
   Search Memory;
7. receives at most one repair turn on the same worker thread;
8. publishes its semantic attempt and slot record before a later slot starts.

Contract validation and scientific evaluation are separate states. A generated
root enters Search Memory as `contract_status = VALID` with
`scientific_outcome = NOT_EVALUATED`. After serial evaluation, the host moves
it to `successful_patterns` only for `ACCEPTED_IMPROVEMENT` or
`VERIFIED_COUNTEREXAMPLE`; every rejected, no-plan, illegal, failed, or
inconclusive result moves to `tested_patterns`. The model's hypothesis remains
labelled `model_hypothesis`, while measured evidence is recorded separately as
`observed_effect`.

Fallback summaries preserve both layers of causality. When a selected rewrite
fails final graph validation and the compiled fallback returns
`NoPlan(NO_MATCH)`, the scientific outcome is
`NO_PLAN_AFTER_ILLEGAL_FINAL_STATE`, with `primary_failure_code =
ILLEGAL_FINAL_STATE` and `terminal_fallback_reason = NO_MATCH`. The host does
not misreport this as an empty selector. Contract-valid archive IDs remain
host-owned and are not repeated in the model-facing projection.

All four preview slots are independent roots. Their model-facing
`active_parent` is therefore null, and their host lineage has no parent program
hash. An active parent is reserved for a real descendant fork.

Large relocation and fanout relation sets are bounded with one seeded cyclic
window before the ordinary seeded pick. This preserves a uniform marginal
candidate probability without charging one interpreter random draw per
enumerated relation. The existing rollback selectors retain their original
reservoir behavior.

The status, epoch manifest, communication state, attempt records, and program
records explicitly retain `provider_mode`, `output_contract`,
`output_schema_contract_sha256`, each brief-specific output-schema hash,
`search_memory_sha256`, `compaction_mode`,
rollback mode, specification and worker thread IDs, fork ancestry, turn IDs,
prompt hashes, usage, validation, duplicates, and publication timing. These
semantic projections remain outside transport artifacts. The same deterministic
serial evaluator, interval fitness, and program selection used by the rollback
remain unchanged. Cohort completeness is evaluated against the immutable
planned-slot manifest rather than a hard-coded production population size. The
selected preview passes its explicit program contract into the evaluator; the
rollback continues to use the default contract.

The host/model boundary is strict. Protocol, program, behavior-signature, and
schema hashes plus App Server request, thread, and turn IDs remain exclusively
in host-owned semantic metadata. Model-facing prompts contain short candidate
aliases, selector and action families, control-flow summaries, evaluation
outcomes, strengths, weaknesses, rejection reasons, and the exact parent AST
only when a mutation requires it. The bootstrap uses a readable acknowledgement;
the model never copies or interprets a digest. Every Native v3 App Server turn
fails closed before dispatch if its prompt, system prompt, or output schema
contains a 64-character hexadecimal digest or a transport-identifier field.

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

The final fresh `medium` smoke used the same four Step 12A briefs:

| Arm | Valid unique programs | First valid AST | Total tokens per valid program |
| --- | ---: | ---: | ---: |
| `persistent_single_ast + slot_specific` | 4/4 | 16.430 s | 6,157 |
| `multi_program_batch` rollback | 0/4 | unavailable | unavailable |

The selected arm used one bootstrap plus four program turns, exactly two forks
from the same specification anchor, zero provider retries or warnings, and
112/112 required App Server artifacts. All four evaluator records completed
without `INVALID_AST`, unknown selectors, or resource-budget failures. The
rollback A/B turn retained exact provider artifact parity but omitted all four
planned slots, so its valid-program rate was 0/4.

This gate validates the communication and model-facing contract integration,
not scientific quality. Cohort completeness is measured against the explicit
four-slot preview manifest, so four valid unique programs produce `COMPLETE`.
That does not imply scientific success: status separately reports
`scientific_result_kind`, `verified_counterexample`, selection tie metadata,
`program_evaluations`, and `graph_score_attempts`. A terminal preview with no
accepted scientific result reports `run_terminal = true`,
`terminal_reason = preview_cohort_complete`, and
`scientific_result_kind = NONE`.

The integrated preview retains Step 12D's two-worker fork and Search Memory
boundaries while using the unchanged production evaluator.

## Validation

```text
uv run ruff check src/mutation_forge/native_v3 src/mutation_forge/stage3/isolation.py tests/unit/test_native_v3_preview.py tests/unit/test_native_v3_experiment.py tests/integration/test_native_v3_route.py
uv run mypy src/mutation_forge/native_v3 src/mutation_forge/stage3/isolation.py
uv run pytest -q tests/unit/test_native_v3_interpreter.py tests/unit/test_native_v3_preview.py tests/unit/test_native_v3_experiment.py tests/integration/test_native_v3_route.py
uv run pytest tests/unit/test_native_v3*.py tests/integration/test_native_v3_route.py -q
uv run pytest tests/unit/test_native_v2_smoke.py tests/integration/test_native_experiment.py tests/unit/test_native_resume.py tests/unit/test_native_selection.py tests/unit/test_native_progress.py -q
make appserver-artifact-parity
```

The focused tests cover explicit selected and rollback routing, rollback by
default, authentication failure before provider construction, two exact
specification forks, four separately published unique compiled AST turns,
schema-identity fail-closed behavior, root-neutral parent context,
post-evaluation Search Memory classification, planned-manifest completeness,
usage finality, explicit score-attempt accounting, selection ties, bounded
Search Memory without ASTs, exact artifact parity, and durable process
replacement through persisted thread identities.

The final full suite collected 820 tests: 793 passed, 25 failed, and two had
setup errors. A targeted detached run at `c7dd740` reproduced the exact same
25 failed node IDs and two setup-error node IDs; the four tests added by Step
12E all pass. The unchanged failures are tied to the known sibling HEG
checkout mismatch: the suite expects
`fd97451b0f3d87400d1d955a2c6b1b18303344ff`, while the clean checkout is
`27cbec9c2307b6ea5f936f858821d11d808b68f3`. Ruff, mypy, diff checks, and
App Server artifact parity pass.

## Manual operator command

Use a fresh `exp_id`:

```toml
[v3]
communication_mode = "persistent_single_ast"
```

```console
uv run mforge experiment run --config experiment.toml --json
```

Set `communication_mode = "multi_program_batch"` with another fresh `exp_id`
to exercise the rollback.

STOP — waiting for operator acceptance. Step 13 has not started.
