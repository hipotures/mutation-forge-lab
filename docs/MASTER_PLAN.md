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
5. **Stage 2D — preregistered independent trajectories (completed and
   validated; `GO_TO_STAGE_3`).** Exactly eight immutable shards containing
   512 paired toy episodes over orders 10/12, independent
   strict-improvement graph states, unchanged Stage 2B rankers, hierarchical
   paired bootstrap gates, and one complete deterministic replay. All eleven
   preregistered gates passed.
6. **Stage 3 — Codex App Server generation (issue #9 preregistration in
   progress).** A schema-derived thin inference adapter, eight isolated
   one-turn no-tool capsules, strict structured output, exact usage and
   provenance, Stage 2A validation, immutable development trajectories,
   deterministic replay, and a twelve-part gate. Live generation is blocked
   for the official campaign until the Phase 1 commit and annotated
   `stage3-generation-frozen-v11` tag are pushed and recorded. Three
   user-authorized connectivity turns used while debugging the adapter are
   retained as excluded diagnostic evidence and did not change the scientific
   contract. The retained v8 official run passed schema validation and emitted
   partial model text in four slots before aggregate stdout and native worker
   creation ended the run. The child `RLIMIT_NPROC=1024` was user-wide, not
   capsule-local. The user-authorized v9 infrastructure correction preserves
   the 64 KiB frame bound, adds a separate 2 MiB aggregate stdout cap, raises
   that bounded limit 100-fold to 102400, and fixes Tokio, Rayon, and numerical
   libraries to one worker per capsule. It does not alter prompts or
   scientific inputs. The retained v9 run started eight turns, completed four,
   and accepted one candidate before further bounded timeout/transport limits.
   The user-authorized v10 correction raises the turn wall limit to 600
   seconds, the final response bound 100-fold, and the event, transcript,
   stdout, and queue bounds tenfold.
   The retained v10 run completed all eight model turns and produced four
   valid unique candidates, but an incomplete hard-coded repair-code list
   suppressed the one permitted repair for four AST-invalid candidates. v11
   treats every error emitted by static AST validation as repairable while
   leaving infrastructure/runtime errors terminal.
7. **Stage 4 — evolutionary program search.** Archive, lineage, deduplication,
   selection, mutation prompts, checkpoints, and fixed compute accounting.
8. **Stage 5 — held-out generalization.** Unseen seeds, random relabelings,
   unseen order, preregistered comparisons, and uncertainty estimates.
9. **Stage 6 — independent verification and red-team review.** Safety and
   scientific-validity audit with reproducible artifacts.
10. **Stage 7 — HEG integration decision.** An explicit GO/NO-GO report before
   any production integration.

Stages 1 and 2A are accepted. Stage 2B is completed and validated, issue #6 is
closed as completed, and `docs/reports/STAGE2B_REPORT.md` records the
historical **`NO_GO`** after its preregistered efficacy gate failed. The Stage
2B implementation and negative evidence remain retained and must not be
altered or reinterpreted as positive evidence. Stage 2C is completed and
records `BENCHMARK_SATURATION` with `DESIGN_STAGE_2D_PREREGISTRATION`.

Issue #8 separately approved Stage 2D under a strict two-phase discipline.
The preregistration froze the runner, config, schemas, ordered 512-episode
manifest, eight shard assignments, unchanged ranker identities, primary and
secondary metrics, hierarchical bootstrap, thresholds, exclusions, resource
controls, and replay requirements before any confirmatory result. The
confirmatory phase did not modify those inputs or analyses. All eleven gates
passed and `docs/reports/STAGE2D_CONFIRMATORY_REPORT.md` records
`GO_TO_STAGE_3`. Stage 2D is non-held-out toy evidence, not a held-out
generalization claim or HEG superiority claim. Issue #9 separately authorizes
Stage 3 under a two-phase freeze. Phase 1 contains no official campaign
generation; Phase 2 is limited to the frozen eight initial slots and at most
one schema/AST-only repair per slot. Stage 4 remains blocked pending the
terminal Stage 3 decision and separate approval.

The Phase 1 statement refers to the official campaign. Before the fresh v6
freeze, three explicitly user-authorized adapter connectivity turns were
attempted, two completed and one ended with partial usage. Their fixed
connectivity outputs are retained, excluded from candidate generation and
evaluation, and were not used to change prompts, schemas, rankers, manifests,
metrics, thresholds, or gates. The user reviewed that diagnostic boundary and
authorized the new immutable freeze.

The retained v6 official attempt was rejected before inference at all eight
turn boundaries because `outputSchema.properties.schema_version` lacked an
explicit string type. After explicit user authorization, v7 adds only that
server-required type and a matching offline preflight regression. The v6 tag
and failed run remain unchanged.

The retained v7 attempt exposed another pre-inference App Server restriction:
`uniqueItems` is not accepted in the structured-output schema subset. v8
allows only the verified transport keywords and leaves all response
size/cardinality validation in the existing application parser. v8 also fixes
the schema-derived renderer's nullable/reference descriptions and snapshots
all eight rendered slot prompts. A versioned glossary bound to the scientific
schema hashes supplies the decision objective, every field's semantics,
pool-constant versus candidate-specific scope, vector alignment, aliases, and
bounded-feature caveats without empirical Stage 2C/2D leakage. The v6/v7 tags
and failed runs remain unchanged.

## Scientific controls

- Immutable initial graph manifest and split.
- Fixed graph seeds, policy seeds, controller, evaluation and wall budgets.
- Same HEG scorer, witness cap, proposal budget, validation, and resource
  limits for compared policies.
- No held-out results in policy-generation context.
- Heuristic zero is only a candidate for exact verification.
- Complete provenance includes both repositories, dirty states, environment,
  lock hash, schemas, config, seeds, limits, timestamps, and terminal status.
