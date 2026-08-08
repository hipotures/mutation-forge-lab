# Native v3 Ordinary-Python M6 Preview Report

## Decision

The explicit ordinary-Python preview completed its bounded acceptance campaign:

```text
generation 0: 8 fresh roots
generation 1: 4 exact-parent children + 4 fresh roots
final:        16 planned, 16 terminal, 0 pending
```

The campaign required one provenance-checked restart after the cumulative App
Server event limit terminated one provider slot. The failed slot remained
consumed as `provider_failed`; the restart submitted only the three pending
slots and did not repeat any terminal provider turn or scientific evaluation.

This is preview evidence, not a change of default. Native v2 remains the
default, the JSON DSL remains available as rollback evidence, and M7 is not
implemented or authorized.

## Implementation lineage

The focused M6 series is:

- `b459eb821b5e2002ae44b44b0dd0e6266d75ea8b` — guarded Python preview route;
- `09dd9c08a7fc30efde3b4d15d30711b6770d071e` — retain the durable capsule
  across resumable failures;
- `cce5f38080784bab4e1434aae9f84d3c2663aa94` — initial anchor restoration;
- `2f9ec4320063ef1cc295b494cf44001aac7c6cab` — distinguish fresh and resumed
  anchor activation;
- `a2d11c933522881231e896c7d28f39a18f71f164` — retain original durable fork
  sources;
- `7cde3097bb8f8d44c1099bafbdcce0207fed4802` — remove the eager JSON-DSL
  interpreter import from the Python preview path;
- `840cbe61e7bc220eae64148d066d57f0f5bcd947` — preserve the direct no-DSL
  assertion against the legacy interpreter.

Final source tree:
`0973b172e68d67b56b5c6230be9645198a9e8608`.

## Explicit preview route and workspace

The new selector is:

```text
protocol = "native-v3-python-v1"
schema_version = "mforge.experiment.native_python_preview_config.v1"
```

It is available through the existing public commands:

```text
mforge experiment run --config <explicit-python-preview-config>
mforge experiment status --config <explicit-python-preview-config>
```

Protocol omission still selects Native v2. Existing Native v2, JSON-DSL v3,
obsolete Python, and mismatched Python workspaces fail before provider or
backend construction and are never reinterpreted.

The active preview call path is:

```text
Codex App Server
-> ordinary-Python source envelope
-> M1 validation and identity
-> M2 isolated policy worker
-> M3 serial evaluator
-> M5 generation, selection, Search Memory, and exact fork loop
-> current HEG evidence and interval fitness
-> exact-verification seam
```

An isolated import probe proves that importing the Python preview loads neither
`mutation_forge.native_v3.interpreter` nor a JSON-DSL IR compiler. Shared
execution records live in `native_v3.execution`; the legacy evaluator imports
its interpreter only inside `evaluate_serial_program`. Model-generated source
is compiled and executed only by the accepted M2 worker.

## Live campaign provenance

The successful campaign began from the clean implementation commit:

- Mutation Forge commit:
  `a2d11c933522881231e896c7d28f39a18f71f164`;
- Mutation Forge tree:
  `6155e8c0c850bdb716235cb12f286439b0a95f14`;
- Mutation Forge worktree: clean;
- HEG commit:
  `27cbec9c2307b6ea5f936f858821d11d808b68f3`;
- HEG tree:
  `85fb2a34a14fc0274137f91aef02cb8c33484d97`;
- HEG worktree: clean;
- experiment configuration SHA-256:
  `dbbb250fb44f4099ac7757d4b182909e706b88b432490373a90fff88fd73313a`;
- model: `gpt-5.6-luna`;
- reasoning effort: `medium`.

The post-campaign commits only decouple the legacy interpreter import and
strengthen its regression test. They do not change prompts, schema, model,
panel, budgets, validator, sandbox, evaluator, scoring, selection, Search
Memory, or generated policy behavior.

Campaign artifacts:

- configuration:
  `/tmp/mutation-forge-native-v3-python-m6-preview/configs/native-v3-python-m6-20260808T182017Z.toml`;
- workspace:
  `/tmp/mutation-forge-native-v3-python-m6-preview/workspaces/native-v3-python-m6-20260808T182017Z`.

These were temporary acceptance-run paths. Their hashes, counters, and
verification findings were distilled before a later power-loss reboot cleared
`/tmp`; the raw temporary trees are no longer locally accessible. Durable
evidence consists of this committed report, the synchronized implementation
history, and the issue record. The paths above identify the historical run and
must not be interpreted as currently downloadable artifacts.

## Recovery and no-repeat proof

The first process stopped after generation-1 root `slot-04` crossed the
cumulative 10,000-event App Server bound:

```text
ProtocolError: event limit exceeded
```

At that boundary:

- generation 0 was terminal;
- generation 1 was frozen as four children plus four roots;
- 13 candidates were terminal;
- slots `05`, `06`, and `07` were pending;
- the failed `slot-04` was terminal `provider_failed`;
- the capsule and provider state were retained.

One restart used the exact retained configuration and provenance. It submitted
only the three pending root slots. The pre-restart inventory contained 506
files and the post-restart inventory 613. No prior terminal candidate,
provider, evaluation, manifest, Search Memory, or provenance artifact changed.
Only the provider runtime state and public preview state advanced.

Final recovery state:

```text
resume attempts: 1
last boundary:   report_persisted
state:           terminal
resumable work:  none
pending slots:   0
```

During the run, `m5-stop.json.gz` was retained as historical recovery evidence.
Public status prioritized the final immutable `m5-report.json.gz` and reported
the run as terminal. Both files belonged to the temporary tree described
above.

## Population, lineage, and scientific accounting

| Generation | Roots | Children | Evaluated | Contract invalid | Provider failed | Terminal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 0 | 6 | 2 | 0 | 8 |
| 1 | 4 | 4 | 7 | 0 | 1 | 8 |
| Total | 12 | 4 | 13 | 2 | 1 | 16 |

There were no duplicate, missing, or evaluation-infrastructure terminal slots.
All four child programs changed source, program identity, and behavior
signature relative to their exact parents.

Every evaluated candidate received the same two-case development panel and
budget:

- panel hash:
  `2ab5a7eaf16db306b82c0127f325c61c13b67f550df1572dec507d173b61457d`;
- two order-30 cases;
- equal horizons, witness caps, forbidden lengths, and HEG budgets;
- no held-out evidence used.

Immutable generation artifacts:

| Artifact | Generation 0 | Generation 1 |
| --- | --- | --- |
| manifest | `b18cda7e6a61aa85db8331fd1c843372969166b463e8a18c4c3e9733b1e90d10` | `a1564574fe76411c2c602fc7720417a0c4ff8e14abf2955f48a47115c3176c23` |
| Search Memory | `8edf61d6394b2dfec740d47aeaec22f5021faa16419f9718b2643e4fc06a5796` | `c8ba2c1f6c293f98fe35d93d08693e3454cec0c3d7186c50978bcae25e78fefe` |

Scientific totals:

- 26 policy invocations;
- 26 complete panel evaluations;
- 44 graph-score attempts over 44 unique graphs;
- zero scientific evaluator provider activity;
- exact score protocol `native_v3_score_50k_200k_v1`;
- result kind `DEVELOPMENT_SEARCH_EVIDENCE`;
- `scientific_success = false`.

A completed provider cohort is therefore not reported as a scientific
counterexample. No apparent zero occurred:

- exact-verifier authority: `exact_verifier_only`;
- submissions: 0;
- records: 0;
- verified: false.

## Actual sandbox behavior

Host-derived API traces, rather than model descriptions, report:

- rewrite plans: 18;
- accepted/rejected rewrites are retained per candidate and case;
- aggregate actions:
  - `k_switch`: 18;
  - `remove_edge`: 4;
  - `relocate_endpoint`: 4;
  - `add_edge`: 2;
- aggregate selectors:
  - `pick`: 52;
  - `edges_witness_load_extreme`: 20;
  - `matching_k_switch_reconnections`: 18;
  - `vertices_witness_load_extreme`: 16;
  - `relocations_legal`: 4;
  - `non_edges_from_vertex`: 2.

Sandbox telemetry:

- starts: 26;
- rotations: 0;
- failures: 0;
- timeouts: 0;
- maximum observed RSS: 19,244 KiB.

All frozen M2 limits and transparent pre-invocation lifetime rotation semantics
remain unchanged. The safe API was not expanded. In particular, it still does
not provide witness-scoped k-switch, fanout, or relocation selectors.

## Provider accounting

- specification anchor turns: 1;
- candidate turns: 17;
- forks: 16;
- bounded contract repairs: 2;
- warning notifications: 16;
- thread resume attempts: 14;
- process restarts: 0;
- transport retries: 0;
- input tokens: 80,288;
- cached input tokens: 15,104;
- output tokens: 34,110;
- reasoning output tokens: 22,377;
- total tokens: 114,398.

The event-limit turn has no scientific result and remains visible as
`provider_failed`.

## Status and failure taxonomy

The bounded read-only status projection exposes:

- active protocol and preview mode;
- current generation and immutable manifest identities;
- planned, terminal, pending, valid, invalid, duplicate, provider-failed,
  evaluated, root, and child counts;
- provider turns, repairs, retries, warnings, and exact usage;
- public program identities and lineage without provider IDs or source;
- sandbox starts, rotations, failures, timeouts, and bounded resources;
- policy invocations and actual selector/action profiles;
- graph-score attempts, unique graphs, and interval fitness;
- resume attempts and last durable boundary;
- exact-verifier submissions, records, authority, and outcome;
- terminal reason and scientific result kind.

Unknown state fields fail closed. Source, prompts, raw responses, paths,
provider/session/thread identifiers, and unbounded exception text are not
projected.

Provider failure, contract invalidity, program failure, evaluation
infrastructure failure, NoPlan, scientific rejection, and exact verification
remain distinct. Terminal provider, invalid, duplicate, missing, and
evaluation-infrastructure slots consume their planned slots and are not
regenerated.

## Final verification

Final synchronized head:
`840cbe61e7bc220eae64148d066d57f0f5bcd947`.

- Ruff: passed.
- mypy: passed for 168 source files.
- focused M1-M6, App Server, lineage, DSL rollback, and current-HEG tests:
  309 passed.
- isolated Python preview import: no JSON-DSL interpreter or IR compiler
  loaded.
- App Server artifact parity:
  131 files, four cases, seven tests passed.
- final real Native v2 smoke:
  one provider turn, zero repair turns, exact final usage of 11,125 tokens,
  terminal with no run error, artifact and parity checks passed.
- full suite:
  1,084 collected, 1,057 passed, 25 failed, 2 errors, one warning in
  914.38 seconds.
- failure/error node-ID delta against accepted M5:
  no new IDs and no resolved IDs.
- `experiment.toml`: unchanged.
- Mutation Forge and HEG worktrees: clean.
- remote branch head: synchronized.

The remaining 25 failures and two errors are the unchanged accepted baseline,
including the known HEG pin mismatch between the frozen expected commit and the
clean current HEG commit recorded above.
