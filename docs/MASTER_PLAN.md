# Master plan

## Research question

Can an LLM-generated, resource-bounded Python policy for selecting or
constructing legal graph rewrites improve held-out graph search compared with
fixed hand-written mutation strategies under an equal evaluation budget?

The evidence ladder intentionally separates infrastructure, safety, model
generation, program search, generalization, and production consideration.

1. **Stage 1 — deterministic harness and HEG baselines (accepted).** Typed interfaces,
   immutable datasets, two reviewed HEG baselines, event and run stores,
   reproducibility tests, and a bounded smoke benchmark. Frozen entry point:
   Mutation Forge `3b9beba058f472d6f0cad5b6210f34c6dbf96731` and HEG
   `fd97451b0f3d87400d1d955a2c6b1b18303344ff`.
2. **Stage 2A — deterministic ranker execution safety without a model.** A
   versioned probe-only input contract, exact ranker template, AST validation,
   normalized identity, isolated bounded worker, adversarial fixtures, fixed
   behavior signature, durable evidence, and deterministic replay.
3. **Stage 2B — proposal and scientific-feature evidence without a model.**
   Host-generated legal k-switch pools, final bounded immutable features,
   fixture/random/structural rankers, penalties, and preregistered toy
   comparisons. No full proposer is permitted before ranker evidence.
4. **Stage 3 — Codex App Server generation.** A schema-derived thin inference
   adapter, isolated authentication/configuration, structured output, saved
   evidence, and exact token usage.
5. **Stage 4 — evolutionary program search.** Archive, lineage, deduplication,
   selection, mutation prompts, checkpoints, and fixed compute accounting.
6. **Stage 5 — held-out generalization.** Unseen seeds, random relabelings,
   unseen order, preregistered comparisons, and uncertainty estimates.
7. **Stage 6 — independent verification and red-team review.** Safety and
   scientific-validity audit with reproducible artifacts.
8. **Stage 7 — HEG integration decision.** An explicit GO/NO-GO report before
   any production integration.

Stage 1 is accepted. Work is limited to Stage 2A until
`docs/reports/STAGE2A_REPORT.md` records `GO_TO_STAGE_2B` and issue #5 is
closed as completed. No model use is allowed before the later Stage 2B GO. No
full proposer may precede ranker evidence, and no HEG policy integration may
precede the final scientific GO.

## Scientific controls

- Immutable initial graph manifest and split.
- Fixed graph seeds, policy seeds, controller, evaluation and wall budgets.
- Same HEG scorer, witness cap, proposal budget, validation, and resource
  limits for compared policies.
- No held-out results in policy-generation context.
- Heuristic zero is only a candidate for exact verification.
- Complete provenance includes both repositories, dirty states, environment,
  lock hash, schemas, config, seeds, limits, timestamps, and terminal status.
