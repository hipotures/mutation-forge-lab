# Milestones and gates

## Stage 1: deterministic infrastructure

Exit requires doctor success, toy and HEG parity tests, both HEG baselines on
the same immutable manifest, valid result graphs, deterministic trajectories,
equivalent Rich and JSON summaries, bounded smoke runtime, `pytest`, `ruff`,
and `mypy`, and no changes to HEG.

## Stage 2: sandbox and local rankers

Entry requires an accepted Stage 1 report. No model tokens may be used.
Implement legal k-switch proposal pools for `2 <= k <= 4`, bounded immutable
features, a single ranker template, AST allowlisting and normalized hashes,
isolated workers with CPU/memory/wall/payload/output limits, fixture/random/
structural rankers, behavior probes, penalties, and validation/evaluation
commands.

Adversarial fixtures must cover imports, file/environment/subprocess/network
access, dunder and reflection, infinite loops, large allocation/output,
recursion, NaN/infinity, exceptions, wrong signatures, multiple functions,
and hidden state. Exit requires pre-execution rejection, bounded termination
and memory, coordinator isolation, stable hashes/signatures, 10,000 valid
calls, invariant preservation, Rich/JSON equivalence, and a structural ranker
beating random on a preregistered toy benchmark. `STAGE2_REPORT.md` must make a
GO/NO-GO decision for live model use.

## Stages 3–7

Stage 3 adds schema-derived Codex App Server generation only after a safety GO.
Stage 4 adds archived evolutionary search. Stage 5 freezes and evaluates
held-out generalization. Stage 6 performs independent verification and
red-team review. Stage 7 alone may recommend HEG production integration.
