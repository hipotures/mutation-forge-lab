# Native v3 Ordinary-Python M5 Search Report

## Decision

The bounded M5 acceptance campaign completed successfully:

```text
generation 0: 8 fresh roots
generation 1: 4 exact-parent children + 4 fresh roots
```

All 16 planned slots are terminal. The run required one provenance-checked
resume after one child turn exceeded the App Server event limit. That failed
slot remained consumed as `provider_failed`; the resume skipped it and
submitted only the six remaining pending slots.

M5 implementation and live acceptance gates pass. This report does not
authorize M6.

## Immutable acceptance provenance

The campaign ran from:

- Mutation Forge commit:
  `33d74b0608ba48c09743efba0af8fe0c16c61fe0`
- Mutation Forge tree:
  `912c6b9e64ceacb3bbc68f82cc49afa719232f43`
- Mutation Forge worktree: clean
- HEG commit:
  `27cbec9c2307b6ea5f936f858821d11d808b68f3`
- HEG tree:
  `85fb2a34a14fc0274137f91aef02cb8c33484d97`
- HEG worktree: clean
- experiment configuration SHA-256:
  `dbbb250fb44f4099ac7757d4b182909e706b88b432490373a90fff88fd73313a`
- provenance SHA-256:
  `ef44a92bde5554d0d0244f6908474f572a3717d7aa6108e6a4296d98b7a8f01d`
- model: `gpt-5.6-luna`
- reasoning effort: `medium`

The snapshot also freezes the M1 validator and identity versions, every M2
runtime and sandbox protocol plus all limits, the M3 evaluator and score
protocols, and these model-facing identities:

| Input | SHA-256 |
| --- | --- |
| system prompt | `46a8a8ae32252c4ab83eebf44e8d8cb849731b939b2c92887576b300f42e0c87` |
| request template | `a5e1b9d0f0a7dc27ea8a80c2966a46b72c15a4f2e6e4ee436d0e0523788be596` |
| specification prompt | `95a2f1b666c0df9f7519b0155000903691c7bcfed5865f62c609903dbb3317bc` |
| policy output schema | `42573a7691de841262f626221dd11ec8cf7214c5c7a75d04c6029f6d8f109a2b` |
| specification acknowledgement schema | `373e21e27b7e72d07f78b1944776e603532023d9bd3139393441063276033823` |

The provenance artifact was written before backend or App Server construction.
Dirty Mutation Forge worktrees fail before workspace creation. Resume requires
an exact snapshot match; there is no compatibility migration.

The earlier workspace
`native-v3-python-m5-20260808T120516Z` is retained only as
`ABORTED_UNVERIFIABLE_PROVENANCE`. It was not resumed, imported, or used as
acceptance evidence.

## Campaign and resume

- command:
  `uv run python scripts/native_v3_python_m5_live_search.py`
- resume command:
  `uv run python scripts/native_v3_python_m5_live_search.py --resume-workspace
  /tmp/mutation-forge-native-v3-python-m5/native-v3-python-m5-20260808T135516Z`
- workspace:
  `/tmp/mutation-forge-native-v3-python-m5/native-v3-python-m5-20260808T135516Z`
- final status: `completed`
- stop reason: `generation_budget`
- population size: 8
- generations: 2
- candidate slots: 16

The initial process completed generation 0, froze generation 1, evaluated
child `slot-00`, and committed child `slot-01` as `provider_failed` after:

```text
ProtocolError: event limit exceeded
```

This was not converted into a program or scientific failure. No scientific
result exists for that slot. The exact-provenance resume reused the generation
manifests and Search Memory, skipped all ten existing terminal candidates, and
submitted only generation-1 slots `02` through `07`.

The immutable hashes remained:

| Artifact | Generation 0 | Generation 1 |
| --- | --- | --- |
| manifest | `b18cda7e6a61aa85db8331fd1c843372969166b463e8a18c4c3e9733b1e90d10` | `1ef987dcab771c23d9f84e4b1ba54c8bc9bdb92199eb9ca68534835eed4dd463` |
| Search Memory | `8edf61d6394b2dfec740d47aeaec22f5021faa16419f9718b2643e4fc06a5796` | `7d64a5cbc069a4070eac6ad6d25c4567aefdfef51418674b533b8aa734062648` |

Generation 0 remained byte-for-byte terminal and was neither regenerated nor
reevaluated. Focused recovery tests separately prove that retained
`evaluated`, `contract_invalid`, `duplicate`, `provider_failed`, and `missing`
slots are skipped, including the exact `1 provider_failed + 7 pending` case.

## Population result

| Generation | Evaluated | Contract invalid | Provider failed | Total |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6 | 2 | 0 | 8 |
| 1 | 6 | 1 | 1 | 8 |
| Total | 12 | 3 | 1 | 16 |

There were no duplicates, missing slots, program failures, or exact-verifier
submissions. Every evaluated candidate received both immutable development
cases:

| Case | Order | Graph seed | Policy seed | Horizon | Witness cap | Forbidden lengths |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `order-30-seed-101` | 30 | 101 | 17 | 1 | 64 | 4, 8, 16 |
| `order-30-seed-103` | 30 | 103 | 19 | 1 | 64 | 4, 8, 16 |

The report contains 24 scientific evaluations, exactly two for each evaluated
candidate. All scientific evaluator external-activity counters are zero.
Candidate source executed only in the M2 worker; the JSON DSL runtime was not
used.

## Frozen selection and child lineage

The deterministic generation-1 manifest selected:

| Child | Parent | Child program status |
| --- | --- | --- |
| `g0001-slot-00` | `g0000-slot-02` | evaluated |
| `g0001-slot-01` | `g0000-slot-01` | provider failed |
| `g0001-slot-02` | `g0000-slot-04` | evaluated |
| `g0001-slot-03` | `g0000-slot-06` | evaluated |

The other four generation-1 slots are fresh roots with `active_parent = null`.
The manifest was frozen before provider submission and was unchanged by
resume.

For each evaluated child, the durable fork history is exactly:

```text
specification anchor
+ inclusive parent history ending at the exact parent response turn
+ one child response turn
```

It contains no sibling or later root turn. All three evaluated children have
source, program hash, and behavior signature different from their parents:

| Child | Program hash | Behavior signature |
| --- | --- | --- |
| `g0001-slot-00` | `7b9038f8038aa95033de199228722f8cd2e1c481491befab904539470b06d9ff` | `3b31355e1fbffc607042bc38c99e1bea568e18a6393199f9bdb6466158d6d94c` |
| `g0001-slot-02` | `0ce032bbb1f83270e8766eb4d2cb46929c2a10aed0ec847171c8e00f171048e3` | `a1feef4ce0349e75e702254e1351233829344d2b451db18a0b7047b8adf0140f` |
| `g0001-slot-03` | `51c8177401cb998823bee29940da41900fa6921148845c1b0c00753135bb57dc` | `eba108ecc4a3b226e9cd69b8753b23fe9652aff6875a76ad64abf6b15f27e7fb` |

Completion-order permutation tests produce the same parent selection.
Crash/resume tests reproduce manifests, Search Memory, lineage, provider order,
evaluation order, behavior profiles, usage, and the final report.

## Host-derived scientific behavior

The table reports actual sandbox API traces, not model claims:

| Candidate | Fitness | Accepted / rejected | NoPlan / illegal | Actual action family | C8 reduction |
| --- | --- | --- | --- | --- | ---: |
| `g0000-slot-01` | `154609609/270610316` | 0 / 2 | 0 / 0 | `add_edge: 2` | 0 |
| `g0000-slot-02` | `311334919/541220632` | 1 / 1 | 0 / 0 | `k_switch: 2` | 2 |
| `g0000-slot-03` | `154609609/270610316` | 0 / 2 | 0 / 0 | `add_edge: 2` | 0 |
| `g0000-slot-04` | `154609609/270610316` | 0 / 2 | 0 / 0 | `k_switch: 2` | 0 |
| `g0000-slot-06` | `154609609/270610316` | 0 / 2 | 0 / 0 | `add_edge: 2` | 0 |
| `g0000-slot-07` | `154609609/270610316` | 0 / 2 | 0 / 0 | `add_edge: 2` | 0 |
| `g0001-slot-00` | `154609609/270610316` | 0 / 0 | 2 / 2 | `relocate_endpoint: 2` | 0 |
| `g0001-slot-02` | `156019555/270610316` | 1 / 1 | 0 / 0 | `k_switch: 2` | 3 |
| `g0001-slot-03` | `158834755/270610316` | 2 / 0 | 0 / 0 | `k_switch: 2` | 12 |
| `g0001-slot-05` | `315547553/541220632` | 2 / 0 | 0 / 0 | `k_switch: 2` | 12 |
| `g0001-slot-06` | `78361873/135305158` | 2 / 0 | 0 / 0 | `k_switch: 2` | 5 |
| `g0001-slot-07` | `314145427/541220632` | 1 / 1 | 0 / 0 | `k_switch: 2` | 8 |

The strongest evaluated child, `g0001-slot-03`, improved its parent's exact
fitness from `154609609/270610316` to `158834755/270610316`. It used
`edges_witness_load_extreme`, `matching_k_switch_reconnections`, `pick`, and
`k_switch` twice each, accepted both rewrites, and proved a total C8 reduction
of 12 across the panel.

The full report retains selector frequencies, action frequencies, API-call
means, NoPlan reasons, witness deltas for lengths 4, 8, and 16, semantic trace
hashes, acceptance/rejection counts, and exact rational fitness for every
evaluated program.

The safe API still cannot causally constrain k-switch, fanout, or relocation
candidates to a witness-selected edge or vertex. M5 did not expand the API.
This remains an evidence-backed limitation for a later operator decision.

## Provider and exact-verifier accounting

- specification anchor turns: 1
- retained successful candidate/repair turns: 21
- bounded repair turns: 6
- one additional terminal failed child turn with no final usage
- successful-turn provider duration: 758,355 ms
- legal warning notifications: 16
- exact accounted input tokens: 107,680
- cached input tokens: 6,656
- output tokens: 39,274
- reasoning output tokens: 24,901
- successful-turn total tokens: 146,954

`usage_final_exact` is false for the cohort because the failed event-limit turn
had no final usage. Successful retained turns have exact final usage. The
failed turn produced no scientific result.

No candidate reached apparent heuristic zero, so the exact verifier received
zero submissions and produced zero records. Only the exact verifier retains
authority to mark a counterexample `VERIFIED`.

## Verification gates

- Ruff: passed.
- mypy: passed for 166 source files.
- focused provenance and resume tests: 37 passed.
- focused M1/M2/M3/M5 and current-HEG tests: 237 passed.
- App Server artifact parity: 131 files and 7 tests passed.
- Native v2 real-provider smoke: passed with one completed turn, no repair,
  exact final usage of 9,866 tokens, and all structural checks true.
- full suite: 1,070 collected, 1,043 passed, 25 failed, 2 errors, 1 warning.
- full-suite delta: 20 added passing tests and no new or resolved failure/error
  IDs compared with the accepted pre-follow-up M5 baseline.
- known failures/errors remain the exact HEG-pin, historical manifest, and
  dashboard set.
- `experiment.toml`: unchanged.
- active Native v2 and JSON-DSL routes: unchanged.
- preview: inactive.
- safe API: unchanged.
- evaluator concurrency and streaming: not added.

## Acceptance summary

All report acceptance checks are true:

- generation 0 contains exactly eight fresh roots;
- generation 1 contains exactly four children and four fresh roots;
- all planned slots are terminal and consumed;
- equal development panel and budget hold for every evaluated candidate;
- exact parent/child lineage is retained;
- evaluated child source and semantic identities differ from their parents;
- host-derived behavior profiles exist for every evaluated program;
- scientific evaluator model/provider activity is zero;
- selection and resume are deterministic;
- the DSL runtime was not used;
- M6 was not started.
