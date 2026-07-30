# Milestones and gates

## Stage 1: deterministic infrastructure

Exit requires doctor success, toy and HEG parity tests, both HEG baselines on
the same immutable manifest, valid result graphs, deterministic trajectories,
equivalent Rich and JSON summaries, bounded smoke runtime, `pytest`, `ruff`,
and `mypy`, and no changes to HEG.

**Status: accepted.** Frozen entry point: Mutation Forge
`3b9beba058f472d6f0cad5b6210f34c6dbf96731`; HEG
`fd97451b0f3d87400d1d955a2c6b1b18303344ff`.

## Stage 2A: safe deterministic Python ranker runtime

Entry requires accepted Stage 1. No model or network call is allowed. Stage 2A
implements only the exact `priority(ctx, proposal)` execution contract over
versioned, bounded probe schemas; static AST allowlisting; formatting- and
local-name-stable program identity; a spawned persistent Linux worker with
CPU/address-space/file-size/file-descriptor/process-count and parent wall
limits; fixed behavior probes; deterministic replay; reviewed ranker and
adversarial fixtures; machine-readable CLI commands; and durable artifacts.

Adversarial fixtures must cover imports, file/environment/subprocess/network
access, dunder and reflection, infinite loops, large allocation/output,
recursion, NaN/infinity, exceptions, wrong signatures, multiple functions,
hidden state, input mutation, and protocol corruption. Exit requires
pre-execution rejection where applicable, bounded termination and memory,
coordinator isolation, stable source/AST/behavior identities, 10,000 calls on
one valid persistent worker, replay, Rich/JSON canonical equivalence, unchanged
Stage 1 behavior, and a `STAGE2A_REPORT.md` decision of `GO_TO_STAGE_2B` or
`NO_GO`.

Stage 2A explicitly excludes generalized k-switches, final scientific
features, random/structural HEG comparison, proposal pools, a full proposer,
model/App Server use, evolution, held-out claims, and HEG integration.

**Status: accepted.** Issue #5 is closed as completed and
`docs/reports/STAGE2A_REPORT.md` records `GO_TO_STAGE_2B`. Frozen Stage 2B
entry point: Mutation Forge
`e2d11bb86b4fa5dbc7ebfb441923e0f02e9799a9`; HEG
`fd97451b0f3d87400d1d955a2c6b1b18303344ff`.

## Stage 2B: proposal and feature evidence

Stage 2B added host-generated legal
k-switch pools for `2 <= k <= 4`, frozen bounded immutable scientific schemas,
reviewed random/structural rankers through the Stage 2A worker, a
preregistered paired toy comparison, and a bounded order-30 HEG pilot.

**Status: completed and validated; `NO_GO`.** Issue #6 is closed as completed.
Safety, determinism, boundedness, fairness, artifact durability, and the HEG
pilot passed, but the preregistered efficacy gate failed: structural achieved
0.0% relative median normalized best-so-far AUC improvement against a required
10%, and the paired bootstrap interval included zero. The implementation and
negative evidence remain retained in `docs/reports/STAGE2B_REPORT.md`.

Stage 3, model use, evolution, full proposer work, and HEG policy integration
remain blocked. The failed gate must not be reinterpreted by changing its
threshold, dataset, benchmark result, or scientific interpretation.

## Stage 2C: diagnostic follow-up

Stage 2C reproduced the exact Stage 2B control, added bounded rank/metric and
feature diagnostics, isolated an opt-in full-pool oracle from normal search,
and executed the frozen non-confirmatory orders 8/10/12 discrimination matrix.

**Status: completed and validated; diagnostic only.** The primary diagnosis is
`BENCHMARK_SATURATION`: the order-8 benchmark and best-so-far metric collapsed
distinct policy behavior despite universal pool headroom and substantially
better immediate structural selections. The next-step decision is
`DESIGN_STAGE_2D_PREREGISTRATION`.

That decision does not approve or execute Stage 2D. Any Stage 2D benchmark
must be frozen and approved in a separate issue. Stage 2B remains `NO_GO`, and
Stage 3, model use, evolution, full proposer work, and HEG policy integration
remain blocked. Evidence is retained in
`docs/reports/STAGE2C_DIAGNOSTIC_REPORT.md`.

## Stage 2D: preregistered independent-trajectory confirmation

Stage 2D is approved as a separate two-phase confirmatory benchmark. Phase 1
freezes the runner, schemas, unchanged Stage 2B ranker identities, toy orders
10/12, graph seeds 201–208, policy seeds 1001–1032, horizon 32, 10,000-sample
hierarchical paired bootstrap, eleven-part gate, and exactly eight 64-episode
shards. The immutable annotated tag is `stage2d-preregistered-v1`. No
confirmatory episode may run before that tag and its commit are pushed and
recorded on issue #8.

Phase 2 runs all eight shards twice from clean detached preregistration
checkouts. Policies follow independent strict-improvement trajectories and
generate independent pools after their graphs diverge. The replay must match
all timing-stripped episode records, shard hashes, aggregate hash, statistics,
and gate result. The only terminal decisions are `GO_TO_STAGE_3`, `NO_GO`, and
`INCONCLUSIVE_INFRASTRUCTURE_FAILURE`.

**Status: completed and validated; `GO_TO_STAGE_3`.** All eleven
preregistered gates passed, including exact primary/replay identity. Stage 2B
remains the historical `NO_GO`. This non-held-out toy result is neither a
held-out generalization claim nor evidence of HEG superiority. Stage 3 has not
started and requires a separate issue and explicit user approval.

## Stages 3–7

**Stage 3 status: issue #9 Phase 1 preregistration in progress.** The offline
implementation freezes the exact `gpt-5.6-luna`/`high` profile, eight ordered
one-shot slots, one schema/AST-only repair maximum, private no-tool App Server
capsules, strict output and exact usage accounting, Stage 2A validation and
10,000-call smoke, the fixed non-held-out development manifest, independent
trajectories, deterministic replay, and twelve named gates. No live generation
for the official campaign is permitted before the Phase 1 commit and annotated
`stage3-generation-frozen-v11` tag are pushed and recorded on issue #9. Three
explicitly user-authorized adapter connectivity turns preceded this freeze;
two completed and one ended with partial usage. They are retained and excluded
as diagnostic evidence, did not generate candidates, and did not change the
scientific contract. The user reviewed that boundary and authorized the fresh
freeze.

The retained v6 official attempt reached all eight App Server turn boundaries
but was rejected before inference because the structured-output
`schema_version` property omitted an explicit JSON Schema string type. The
user authorized a repair. v7 adds only that required type and an offline
preflight regression; v6 and its failure artifacts remain immutable.

The retained v7 attempt then exposed a second pre-inference transport
rejection: App Server does not permit `uniqueItems` in its structured-output
schema subset. v8 removes unsupported transport-only length/cardinality
keywords, enforces a strict supported-keyword allowlist before freeze, and
retains the same limits in the application parser. It also expands nullable
and referenced scientific fields into readable prompt text and snapshots all
eight slot prompts. A versioned, schema-hash-bound semantic glossary covers
all context/proposal fields, pool scope, vector alignment, aliases,
directionality, and budget caveats without importing Stage 2C/2D empirical
findings. v6/v7 remain immutable.

The retained v8 official run passed the structured-output boundary and four
slots emitted partial model text, but it ended without final responses or
final usage. The adapter had incorrectly reused the 64 KiB single-frame bound
as an aggregate stdout bound, while eight Codex processes could exhaust native
worker creation because `RLIMIT_NPROC=1024` applied across all tasks owned by
the shared user, not one capsule. The user authorized a v9
infrastructure-only freeze: frames remain capped at 64 KiB, aggregate stdout
is separately capped at 2 MiB, the bounded user-wide process limit rises
100-fold to 102400, and each capsule fixes Tokio, Rayon, and
numerical-library workers to one. No prompt, schema, ranker, manifest, metric,
gate, or evaluation behavior changes.
All earlier tags and failure artifacts remain immutable.

The retained v9 run started all eight turns, completed four, and accepted one
candidate. Two turns reached the 120-second wall limit, one reached the 256
KiB transport-log limit, and one reached an incoming-message/queue bound. The
user authorized v10 with a 600-second turn limit, a 100-fold final-response
limit, and tenfold event, transcript, aggregate-stdout, and queue bounds.

The retained v10 run completed all eight model turns without timeout or
transport failure and produced four valid unique candidates. Four candidates
failed static AST validation, but the permitted repair turn was suppressed by
an incomplete hard-coded repair-code list. v11 tags every static-validator
error as AST-repairable while transport, usage, and runtime failures remain
terminal.

The retained v11 campaign completed all eight initial turns and five repair
turns. Seven candidates passed the former validator. The eighth final response
used a finite dynamic `range(min(...))` loop that the static-bound heuristic
could not prove. The user-authorized v12 validation amendment removes all
static loop-bound and termination inference, permits `for` and `while`, and
relies on the existing per-candidate CPU/wall limits for slow or infinite
programs. No additional model call is made. Provider-free revalidation of the
eight retained final responses produced eight unique valid candidates and
80,000 successful persistent-worker smoke calls. Raw generation prompts,
responses, usage, JSON-RPC, event, stderr, and transport logs remain unchanged;
the derived revalidation evidence is stored separately.

The first frozen v12 evaluation completed its primary and replay computations
in memory but failed before durable reduction. The runner serialized each
policy-step trace twice and attempted one monolithic 128-record JSONL; its
uncompressed payload exceeded the 64 MiB writer bound before gzip or a
temporary file was created. The failure retained no scientific metrics or
replay hashes. The user-authorized v13 evaluation-only correction stores each
trace exactly once, replaces redundant full score objects with compact
selected-plan score evidence, and writes primary/replay as eight deterministic
bounded shards of sixteen episodes. It does not change policies, prompts,
schemas visible to generated code, manifest, seeds, metrics, bootstrap,
thresholds, or gates and makes no model call.

The Stage 3 result can only be `GO_TO_STAGE_4`, `NO_GO`, or
`INCONCLUSIVE_INFRASTRUCTURE_FAILURE`. Stage 2B remains the historical
`NO_GO`; Stage 2D remains non-held-out toy evidence rather than HEG
superiority.

Stage 4 would add archived evolutionary search. Stage 5 would freeze and
evaluate held-out generalization. Stage 6 would perform independent
verification and red-team review. Stage 7 alone could recommend HEG policy
integration after a final scientific GO. None of Stages 4–7 has started.
