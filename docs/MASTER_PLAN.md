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
2. **Stage 2A — deterministic ranker execution safety without a model
   (accepted).** A
   versioned probe-only input contract, exact ranker template, AST validation,
   normalized identity, isolated bounded worker, adversarial fixtures, fixed
   behavior signature, durable evidence, and deterministic replay.
   Accepted entry point: Mutation Forge
   `e2d11bb86b4fa5dbc7ebfb441923e0f02e9799a9` and HEG
   `fd97451b0f3d87400d1d955a2c6b1b18303344ff`.
3. **Stage 2B — proposal and scientific-feature evidence without a model
   (completed; `NO_GO`).** Host-generated legal k-switch pools, bounded
   immutable features, fixture/random/structural rankers, a preregistered toy
   comparison, and a bounded non-held-out HEG pilot. Issue #6 is closed as
   completed. The preregistered efficacy gate failed, and the implementation
   plus negative evidence remain retained.
4. **Stage 2C — diagnostic follow-up (completed; diagnostic only).** Exact
   Stage 2B control replay, isolated full-pool toy oracle, rank/metric/feature
   diagnostics, and a frozen exploratory discrimination matrix. Primary
   diagnosis: `BENCHMARK_SATURATION`. Decision:
   `DESIGN_STAGE_2D_PREREGISTRATION`.
5. **Stage 3 — Codex App Server generation (blocked).** A schema-derived thin
   inference adapter, isolated authentication/configuration, structured
   output, saved evidence, and exact token usage.
6. **Stage 4 — evolutionary program search.** Archive, lineage, deduplication,
   selection, mutation prompts, checkpoints, and fixed compute accounting.
7. **Stage 5 — held-out generalization.** Unseen seeds, random relabelings,
   unseen order, preregistered comparisons, and uncertainty estimates.
8. **Stage 6 — independent verification and red-team review.** Safety and
   scientific-validity audit with reproducible artifacts.
9. **Stage 7 — HEG integration decision.** An explicit GO/NO-GO report before
   any production integration.

Stages 1 and 2A are accepted. Stage 2B is completed and validated, issue #6 is
closed as completed, and `docs/reports/STAGE2B_REPORT.md` records **`NO_GO`**
after the preregistered efficacy gate failed. Stage 3, model use, evolution,
full proposer work, and HEG policy integration remain blocked. The Stage 2B
implementation and negative evidence are retained and must not be altered or
reinterpreted as positive evidence. Stage 2C is completed and records
`BENCHMARK_SATURATION` with `DESIGN_STAGE_2D_PREREGISTRATION`. This recommends
only a separately specified and approved future preregistration; it does not
execute Stage 2D, clear `NO_GO`, or unlock Stage 3.

## Scientific controls

- Immutable initial graph manifest and split.
- Fixed graph seeds, policy seeds, controller, evaluation and wall budgets.
- Same HEG scorer, witness cap, proposal budget, validation, and resource
  limits for compared policies.
- No held-out results in policy-generation context.
- Heuristic zero is only a candidate for exact verification.
- Complete provenance includes both repositories, dirty states, environment,
  lock hash, schemas, config, seeds, limits, timestamps, and terminal status.
