# Native v3 Step 12C: context-compaction retention experiment

Step 12C adds a standalone test harness for measuring what survives an explicit
App Server context compaction. It does not enable compaction in production and
does not change Native v2.

## Protocol

The installed `codex-cli 0.146.0` accepts `thread/compact/start` with only a
`threadId`; it exposes no per-request compaction prompt or summary payload.
Consequently, the directive arm sends
`[CONTEXT COMPACTION RETENTION DIRECTIVE]` as a normal structured-
acknowledgement turn immediately before compaction. The control arm sends a
neutral checkpoint instead.

Each repetition uses one durable thread for:

1. one full-specification bootstrap and protocol-hash acknowledgement;
2. two fixture-AST population turns;
3. two host evaluation-summary turns;
4. one directive or control checkpoint;
5. one explicit compaction turn with a correlated `contextCompaction` item;
6. one structured compact-manifest probe;
7. one exact active-parent AST probe.

The host-held reference contains the protocol hash, specification invariants,
active generation, exact active-parent AST and canonical hash, three earlier
candidate IDs, one-sentence strategy summaries, outcomes, integer scores,
strengths, weaknesses, lineage relationships, rejected behavior signatures,
and the pending next action.

The manifest probe does not reveal reference values in its output schema. It
detects omitted and altered values as well as unknown candidate IDs, hashes,
scores, and relationships. The parent probe is accepted only when the returned
program passes the Native v3 contract and its canonical hash equals the host
reference.

Classification is:

- `RELIABLE_FOR_OPTIMIZATION` only when all three directive repetitions retain
  the exact manifest and exact active parent;
- `BEST_EFFORT_ONLY` when at least one directive repetition retains useful
  evidence but exact retention is not unanimous;
- `UNUSABLE` when the directive arm retains neither an exact parent nor useful
  summaries/signatures.

## Bounded live result

The accepted evidence is
`workspace/step12c_compaction_medium_002/compaction-report.json.gz`.
It used `gpt-5.6-luna` with `medium` reasoning effort for three directive and
three control repetitions.

Classification: **`BEST_EFFORT_ONLY`**

| Metric | Directive | Control |
| --- | ---: | ---: |
| Compaction success | 3/3 | 3/3 |
| Exact active-parent hash | 2/3 | 0/3 |
| Exact retained manifest | 0/3 | 0/3 |
| Compact summaries retained | 3/3 | 3/3 |
| Rejected signatures retained | 0/3 | 0/3 |
| Hallucinated manifest values | 0 | 1 |
| Mean compaction latency | 34.850 s | 19.281 s |

The directive repetitions had compaction latencies of 39.485, 25.442, and
39.624 seconds. Their active-parent probe matched exactly in repetitions 1 and
2. Repetition 0 returned an object that did not satisfy the direct
`program`/`design_summary`/`hypothesis` response contract.

All three control compactions and manifest probes completed. Their later
active-parent probes ended with upstream `systemError`, so the observed 0/3
parent result is not a clean scientific comparison against the directive arm.
The harness retained the completed compaction identities, usage, latencies,
and manifest comparisons despite those later failures.

No repetition retained the complete manifest. Compact candidate summaries,
outcomes, scores, strengths, weaknesses, and relationships survived in all six
repetitions, but `canonical_protocol_id`, `pending_next_action`, and rejected
behavior signatures were frequently omitted or altered. One control response
reported an incorrect active-parent hash and was classified as a hallucinated
hash.

## Token usage

Usage is the exact App Server `tokenUsage.last` data aggregated across the
three repetitions in each arm. Cached and cache-write input remain separate
from input; `totalTokens` is the provider value.

| Before compaction | Directive | Control |
| --- | ---: | ---: |
| Input | 16,265 | 16,063 |
| Cached input | 0 | 0 |
| Cache-write input | 0 | 0 |
| Output | 144 | 184 |
| Reasoning output | 48 | 91 |
| Total | 16,409 | 16,247 |

| After compaction probes | Directive | Control |
| --- | ---: | ---: |
| Input | 51,292 | 18,962 |
| Cached input | 12,800 | 0 |
| Cache-write input | 0 | 0 |
| Output | 3,113 | 2,407 |
| Reasoning output | 804 | 720 |
| Total | 54,405 | 21,369 |

Control after-compaction usage contains only completed manifest probes because
all three parent probes failed terminally before exact usage was available.

## Artifact and failure evidence

Every successful normal turn and successful compaction prefix has the same
16-file transport artifact set used by Step 12B. Each `06-compaction` wire and
event log contains one correlated `contextCompaction` item start/completion and
a completed compaction turn. Directive markers appear only in directive
checkpoint requests.

Failed parent probes preserve request, RPC, event, wire, stdout, stderr,
transcript, and bounded provider-failure evidence. A deterministic regression
forces a failure after successful compaction and manifest probing, then verifies
that the earlier compaction status, identities, metrics, and comparison remain
in the final report.

The first disposable run, `step12c_compaction_medium_001`, exposed that
late-failure reporting defect and remains unchanged as audit evidence. It is
not the accepted scientific report.

## Validation

```text
python3 /home/user/.codex/skills/codex-app-server/scripts/audit_installed_protocol.py
uv lock --check
uv run ruff check src/mutation_forge/stage3/app_server.py src/mutation_forge/native_v3/compaction_experiment.py scripts/native_v3_compaction_experiment.py tests/fixtures/fake_stage3_app_server.py tests/unit/test_native_v3_compaction_experiment.py
uv run mypy
uv run pytest -q tests/unit/test_native_v3_compaction_experiment.py
uv run pytest tests/unit/test_native_v3*.py tests/integration/test_native_v3_route.py -q
uv run pytest tests/unit/test_stage3_app_server.py tests/unit/test_stage4_app_server.py -q
uv run pytest tests/unit/test_native_v2_smoke.py tests/integration/test_native_experiment.py tests/unit/test_native_resume.py tests/unit/test_native_selection.py tests/unit/test_native_progress.py -q
make appserver-artifact-parity
```

The fake protocol suite covers successful lifecycle correlation, usage before
and after completion, timeout, failed compaction item, missing completion,
partial late-probe failure, hallucination detection, six-repetition reporting,
and exact artifact suffixes.

## Manual operator command

Run from the dedicated worktree with a new disposable workspace:

```bash
uv run python scripts/native_v3_compaction_experiment.py \
  --workspace workspace/step12c_operator_medium_001 \
  --auth-json /home/user/.codex/auth.json \
  --model gpt-5.6-luna \
  --effort medium \
  --turn-timeout 900
```

Expected result is exit status zero plus a JSON report on stdout. The workspace
contains `compaction-report.json.gz`, `compaction-report.md`, and six
arm/repetition artifact trees. A failed workspace is immutable evidence and
must not be reused.

## Limitations

- A normal pre-compaction retention directive is not a protocol-level
  guarantee.
- The sample contains only three repetitions per arm with one fixture shape.
- The control parent probes failed for infrastructure reasons, so the active-
  parent comparison is incomplete.
- No arm retained the exact complete manifest or rejected signatures.
- Compaction is not enabled in production; threshold logic, forks, scheduler
  integration, scoring, and selection remain out of scope.

STOP — waiting for operator acceptance.
