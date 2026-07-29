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

Stage 2B is active after the Stage 2A gate. It adds host-generated legal
k-switch pools for `2 <= k <= 4`, frozen bounded immutable scientific schemas,
reviewed random/structural rankers through the Stage 2A worker, a
preregistered paired toy comparison, and a bounded order-30 HEG pilot. It still
uses no model and does not integrate a policy into HEG. A full proposer remains
blocked until ranker evidence exists. The Stage 2B report must make an explicit
`GO_TO_STAGE_3` or `NO_GO` decision before any model use.

## Stages 3–7

Stage 3 adds schema-derived Codex App Server generation only after a safety GO.
Stage 4 adds archived evolutionary search. Stage 5 freezes and evaluates
held-out generalization. Stage 6 performs independent verification and
red-team review. Stage 7 alone may recommend HEG policy integration after the
final scientific GO.
