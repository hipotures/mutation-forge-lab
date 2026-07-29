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

**Status: preregistration implementation in progress; confirmatory result not
yet observed.** Stage 2B remains `NO_GO`. Stage 3, model use, evolution, full
proposer work, and HEG policy integration remain blocked.

## Stages 3–7

Stages 3–7 remain blocked pending a reviewed Stage 2D result. Stage 3 would add
schema-derived Codex App Server generation only after a separately approved
GO. Stage 4 would add archived evolutionary search. Stage 5 would freeze and
evaluate held-out generalization. Stage 6 would perform independent
verification and red-team review. Stage 7 alone could recommend HEG policy
integration after a final scientific GO.
