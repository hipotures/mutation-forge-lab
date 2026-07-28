# Master plan

## Research question

Can an LLM-generated, resource-bounded Python policy for selecting or
constructing legal graph rewrites improve held-out graph search compared with
fixed hand-written mutation strategies under an equal evaluation budget?

The evidence ladder intentionally separates infrastructure, safety, model
generation, program search, generalization, and production consideration.

1. **Stage 1 — deterministic harness and HEG baselines.** Typed interfaces,
   immutable datasets, two reviewed HEG baselines, event and run stores,
   reproducibility tests, and a bounded smoke benchmark.
2. **Stage 2 — generated-code safety without a model.** Legal k-switch pools,
   ranker template, AST validation, isolated bounded workers, adversarial
   fixtures, behavior signatures, and deterministic replay.
3. **Stage 3 — Codex App Server generation.** A schema-derived thin inference
   adapter, isolated authentication/configuration, structured output, saved
   evidence, and exact token usage.
4. **Stage 4 — evolutionary program search.** Archive, lineage, deduplication,
   selection, mutation prompts, checkpoints, and fixed compute accounting.
5. **Stage 5 — held-out generalization.** Unseen seeds, random relabelings,
   unseen order, preregistered comparisons, and uncertainty estimates.
6. **Stage 6 — independent verification and red-team review.** Safety and
   scientific-validity audit with reproducible artifacts.
7. **Stage 7 — HEG integration decision.** An explicit GO/NO-GO report before
   any production integration.

Stage 1 is the only implemented milestone. Work must stop at this boundary
until its report and Stage 2 prerequisites have been reviewed.

## Scientific controls

- Immutable initial graph manifest and split.
- Fixed graph seeds, policy seeds, controller, evaluation and wall budgets.
- Same HEG scorer, witness cap, proposal budget, validation, and resource
  limits for compared policies.
- No held-out results in policy-generation context.
- Heuristic zero is only a candidate for exact verification.
- Complete provenance includes both repositories, dirty states, environment,
  lock hash, schemas, config, seeds, limits, timestamps, and terminal status.
