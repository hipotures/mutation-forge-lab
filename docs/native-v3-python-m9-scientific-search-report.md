# Native v3 ordinary-Python M9 scientific search report

## Outcome

M9 completed its bounded first sustained scientific search. It did not find a
verified counterexample.

The campaign consumed the frozen budget of 64 provider program turns and
terminated with `provider_turn_budget`. It retained 56 terminal planned slots
and left the eight generation-7 slots pending without submitting or
fabricating replacements. Of the terminal slots, 48 contained evaluated
ordinary-Python programs and eight were `contract_invalid`.

The best development result was `g0004-slot-04`, with the exact point fitness

```text
316968447 / 541220632
= 0.585654774151329840655446409515
```

This is a development score, not a counterexample certificate. The exact
verifier received zero submissions and produced zero records. Its authority
remained `exact_verifier_only`.

The small canonical result is
[`native-v3-python-m9-result-manifest.json`](native-v3-python-m9-result-manifest.json).
Large artifacts remain outside Git at
`/home/user/DEV/mutation-forge-lab-evidence/m9`.

## Implementation

M9 was built from merged M8 main commit
`60285f623fa0f0670518337d114d0e52866926f2` on
`native-v3-python-scientific-search`.

The implementation added:

- three edge-scoped safe selectors:
  `matching_k_switch_reconnections_for_edge`,
  `edge_fanouts_legal_for_edge`, and
  `relocations_legal_for_edge`;
- bounded evaluation concurrency with 12 independently owned evaluator
  backends and a canonical single writer;
- provider/evaluator overlap for already prepared programs;
- a versioned eight-generation, eight-hour, 64-program-turn profile;
- phase timing, throughput, queue, utilization, failure, lineage, fitness, and
  exact-verifier status;
- durable resume for prepared and terminal work;
- bounded Search Memory retention and deterministic offline terminalization.

Native v2 remained the default. The sustained Python route remained explicit
and opt-in. `experiment.toml` was not changed. No issue in #35–#42 was started,
and the removed JSON DSL was not restored.

## Profile before and after

The pre-M9 profile ran for 640.618 seconds. It evaluated four programs from
eight roots, consumed three invalid slots, and ended when the old App Server
event envelope failed the final slot. Retained provider work accounted for
about 61% of wall time. Evaluator work occupied approximately 0.05%.

The final replacement campaign recorded 3,045.387 seconds of active search
time and 4,623.211 seconds of wall time including offline finalization. It
evaluated 48 programs and completed 96 panel evaluations. Policy throughput
increased from approximately 0.0125/s to 0.0315/s, while score-attempt
throughput reached 0.0588/s.

The limiting resource remained program supply:

| Metric | Baseline | Sustained campaign |
| --- | ---: | ---: |
| Provider program turns | 8 candidate attempts | 64 candidate turns |
| Time to first valid source (mtime proxy) | 285.099 s | 165.862 s |
| Evaluated programs | 4 | 48 |
| Policy invocations | 8 | 96 |
| Policy invocations/s | 0.0125 | 0.0315 |
| Graph score attempts | 16 | 179 |
| Evaluator configured / peak active | 1 / 1 | 12 / 1 |
| Evaluator utilization | 0.051% | 0.028% |
| Provider-wait share | about 61% | 99.25% |
| Worker rotations | 0 | 0 |

The final phase totals were 3,022.503 seconds waiting for the provider,
10.317 seconds of evaluator work, 1.568 seconds of persistence, 0.394 seconds
inside Python sandboxes, 0.173 seconds in selectors, 0.022 seconds in actions,
and 0.018 seconds in HEG scoring. Evaluator queue wait was 0.007 seconds.
The time-to-first-valid values are durable artifact-persistence timestamp
proxies, not native monotonic timers. The sustained value measures the first
retained source for `g0000-slot-01`; its terminal evaluated record followed at
217.866 seconds.

Twelve evaluator workers therefore remain a safe bounded capacity, but the
campaign never had enough simultaneously valid programs to use more than one.
Increasing evaluator parallelism further would not improve this workload. The
next measured bottleneck is provider latency and valid-program supply.

## Campaign profile and lineage

The live profile was
[`configs/scientific/native-v3-python-m9-v1.toml`](../configs/scientific/native-v3-python-m9-v1.toml)
with:

- model `gpt-5.6-luna`, effort `medium`;
- one provider and 12 evaluators;
- 28,800 active wall seconds;
- 64 provider program turns, including repairs;
- generation 0 with eight roots;
- generations 1–7 with four exact children and four fresh roots;
- the frozen two-case order-30 development panel;
- no automatic replacement of terminal invalid or failed slots;
- stop on a verified counterexample.

Generation manifests preserved the required allocation through generation 7.
Twenty-four completed children had exact parent relationships and changed
source, program identity, and semantic behavior. Generation 7 persisted its
manifest but submitted none of its eight slots because the provider-turn budget
had already been exhausted.

The campaign produced 48 source files and 96 evaluation records:

| Outcome | Count |
| --- | ---: |
| Evaluated candidates | 48 |
| Contract-invalid candidates | 8 |
| Duplicate candidates | 0 |
| Provider-failed candidates | 0 |
| Complete evaluation episodes | 90 |
| Program-failure episodes | 6 |
| Pending generation-7 slots | 8 |

Terminal invalid slots consumed their planned positions. No replacement slot
was created.

## Scientific behavior

Across 96 policy invocations, generated programs produced 83 rewrite plans:
35 were accepted and 48 were scientifically rejected. Seven invocations
returned `NoPlan`, seven reached illegal final states, and six evaluation
episodes were classified as program failures.

Relation-aware selectors were invoked 79 times:

| Selector | Calls |
| --- | ---: |
| `matching_k_switch_reconnections_for_edge` | 66 |
| `relocations_legal_for_edge` | 7 |
| `edge_fanouts_legal_for_edge` | 6 |

The action mix was 72 k-switches, ten edge fanouts, seven endpoint
relocations, and one edge addition. The new edge-scoped selectors accounted
for approximately 22.8% of selector calls.

The best program was a fresh root, not an inherited child:

- candidate: `g0004-slot-04`;
- program hash:
  `485e744c0fbe5666a4224317aaf487424907fcd5af0553e6646b7fce5abcd142`;
- source SHA-256:
  `819bb5e003e5a040a1600c1536b76e4c4151a4a872dfe2d297fddbde271edf69`;
- durable source:
  `sources/485e744c0fbe5666a4224317aaf487424907fcd5af0553e6646b7fce5abcd142.py`;
- behavior: two accepted k-switches selected with
  `matching_k_switch_reconnections_for_edge`;
- episode status: two `COMPLETE` evaluations;
- exact-verifier submissions: zero.

Its later child `g0005-slot-00` changed source, program identity, and behavior
while preserving the same development fitness. This is lineage evidence, not
evidence of exact verification.

## Provider, sandbox, and status

The provider used one anchor turn, 64 candidate program turns, and eight repair
turns. Exact recorded usage was:

- input tokens: 454,411;
- cached input tokens: 34,816;
- output tokens: 102,028;
- reasoning output tokens: 61,228;
- total tokens: 556,439;
- provider warnings: 57;
- transport retries, process restarts, and thread resumes: zero.

The M2 sandbox started 96 workers. Six program executions failed inside the
sandbox, with zero sandbox timeouts and zero worker rotations. Peak RSS was
19,180 KiB. These failures remained program outcomes; evaluator
infrastructure recorded zero failed candidate evaluations.

Live JSON status and the existing dashboard exposed the generation, terminal
and pending counts, provider usage, evaluator capacity, queue wait, throughput,
NoPlan and illegal-final rates, selector/action families, best fitness and
lineage, worker health, exact-verifier activity, and the dominant bottleneck.

## Recovery and bounded terminalization

The first M9 campaign stopped after a provider lifecycle failure and two
subsequent provider-unavailable slots. A single fresh-process resume verified
that all 27 terminal slots were immutable and repeated no candidate,
evaluation, or provider turn. It also exposed a float-metadata resume defect,
which was fixed before the fresh replacement campaign.

The replacement campaign used the one authorized fresh replacement. It
submitted exactly 64 candidate program turns and no 65th call. At the
generation-7 boundary, retained Search Memory exceeded its 16 KiB projection
limit before the existing budget terminal path could write a report.

The final implementation bounds Search Memory deterministically and handles
provider-budget exhaustion as a terminal result. Its offline finalizer then
validated the retained protocol, provenance, anchor, manifests, candidates,
source, evaluation, and provider artifacts without starting a provider or
backend. It wrote only `m9-report.json.gz`, `m9-runtime.json.gz`, and
`python-preview-state.json.gz`.

The finalizer did not invent generation-7 Search Memory or candidate records.
It explicitly records generation 7 as the missing Search Memory generation
and its eight slots as pending.

Validation covered 2,167 immutable files. Their aggregate SHA-256 was
`cfc92dcadb3b14e4d612ae249a5427a550ce4dd5e6a592f11bec0bdf3d5b3b6d`
before and after finalization. The final durable report SHA-256 is
`12cb5fbe56fdca4c9472d5f31a82d2020027c7b2ae05c7684181a9babbb05af2`.

## Scientific conclusion

M9 demonstrated that ordinary Python can sustain a multi-generation search,
use witness-edge relations structurally, preserve exact lineage and budgets,
and retain auditable source and evaluation evidence.

It did not discover a verified counterexample. The best score is a bounded
development result only. The dominant remaining constraint is provider
latency and valid-program supply, not evaluator or HEG capacity.
