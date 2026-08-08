# Native v3 Ordinary-Python M5 Search Report

## Decision

M5 is **not ready for operator acceptance**. The one authorized live search
completed generation 0 but stopped before the first child provider turn with a
fail-closed App Server lifecycle error:

```text
IsolationError: fork requires an idle durable thread
```

No second live search was started. M6 remains unauthorized.

## Retained run

- command: `uv run python scripts/native_v3_python_m5_live_search.py`
- command exit: `1`
- workspace:
  `/tmp/mutation-forge-native-v3-python-m5/native-v3-python-m5-20260808T120516Z`
- terminal evidence: `m5-stop.json.gz`
- terminal status: `infrastructure_failure`
- resumable: `true`
- Codex CLI: `codex-cli 0.147.0`
- model: `gpt-5.6-luna`
- reasoning effort: `medium`
- HEG commit:
  `27cbec9c2307b6ea5f936f858821d11d808b68f3`, clean before and after
- active experiment route: unchanged
- `experiment.toml`: unchanged

The run used one specification anchor and the fixed two-case development panel:

| Case | Order | Graph seed | Policy seed | Horizon | Witness cap | Forbidden lengths |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `order-30-seed-101` | 30 | 101 | 17 | 1 | 64 | 4, 8, 16 |
| `order-30-seed-103` | 30 | 103 | 19 | 1 | 64 | 4, 8, 16 |

## Frozen allocation and lineage

Generation 0 completed its frozen manifest of eight fresh roots.

Generation 1 froze the required allocation before provider submission:

| Slot | Kind | Selected parent |
| --- | --- | --- |
| `slot-00` | child | `g0000-slot-00` |
| `slot-01` | child | `g0000-slot-03` |
| `slot-02` | child | `g0000-slot-05` |
| `slot-03` | child | `g0000-slot-01` |
| `slot-04` | root | none |
| `slot-05` | root | none |
| `slot-06` | root | none |
| `slot-07` | root | none |

The generation-1 Search Memory artifact is 1,165 compressed bytes, is
source-free in its model projection, and records the host-derived generation-0
behavior. The first child slot consumed its planned slot as
`provider_failed`; it has no scientific result. No child or generation-1 root
model turn occurred.

## Generation-0 scientific evidence

All valid programs received the same two cases and budgets. Invalid programs
were rejected before worker construction and received no scientific result.

| Candidate | Status | Repair | Fitness interval | Accepted / rejected | NoPlan / failure | Actual actions | C8 proved reduction |
| --- | --- | ---: | --- | --- | --- | --- | ---: |
| `g0000-slot-00` | evaluated | 0 | `78361091/135305158` | 1 / 1 | 0 / 0 | `k_switch: 2` | 6 |
| `g0000-slot-01` | evaluated | 0 | `309923409/541220632` | 1 / 1 | 0 / 0 | `k_switch: 2` | 1 |
| `g0000-slot-02` | contract invalid | 1 | none | none | none | none | none |
| `g0000-slot-03` | evaluated | 0 | `154609609/270610316` | 0 / 0 | 2 / 0 | `add_edge: 2`, `remove_edge: 2` | 0 |
| `g0000-slot-04` | contract invalid | 1 | none | none | none | none | none |
| `g0000-slot-05` | evaluated | 0 | `154609609/270610316` | 0 / 2 | 0 / 0 | `add_edge: 2` | 0 |
| `g0000-slot-06` | contract invalid | 1 | none | none | none | none | none |
| `g0000-slot-07` | evaluated | 0 | `154609609/270610316` | 0 / 0 | 2 / 0 | `add_edge: 2`, `remove_edge: 2` | 0 |

The two illegal-final-state programs account for four actual illegal final
states. No evaluated program had a program failure. There were no program-hash
or behavior-signature duplicates.

Actual selector behavior was also distinct:

- `slot-00`: `matching_k_switch_reconnections: 2`, `pick: 4`,
  `vertices_witness_load_extreme: 2`;
- `slot-01`: `matching_k_switch_reconnections: 2`, `pick: 2`;
- `slot-03`: `edges_witness_load_extreme: 2`,
  `non_edges_local_cycle_risk: 2`, `pick: 4`;
- `slot-05`: `non_edges_from_vertex: 2`,
  `vertices_witness_load_extreme: 2`, `pick: 4`;
- `slot-07`: `edges_witness_load_extreme: 2`,
  `non_edges_from_vertex: 2`, `vertices_witness_load_extreme: 2`, `pick: 6`.

Every scientific result reports zero App Server, model, and provider activity
inside the evaluator. Candidate source executed only in the accepted M2 worker.
No DSL interpreter was used. No heuristic-zero result was submitted to the
exact verifier.

## Provider evidence

The completed portion used:

- one specification-anchor turn;
- eight initial root turns;
- three bounded repair turns;
- 12 completed provider turns total;
- nine legal Code Mode warning notifications, recorded without relaxing
  terminal success conditions;
- no child provider turn;
- total provider duration: 453,636 ms;
- exact usage:
  - input: 48,083;
  - cached input: 8,448;
  - cache-write input: 0;
  - output: 23,715;
  - reasoning output: 16,469;
  - server total: 71,798;
  - final: true for every completed attempt;
  - partial: false for every completed attempt.

The accepted response schema remained the two-field
`schema_version + source` envelope. Generated source, source hash, canonical
AST hash, program hash, semantic traces, and scientific evidence are retained
outside provider-turn artifact directories.

## Root cause and correction

After all roots were evaluated, the provider selected
`g0000-slot-00` as the first parent. Switching the adapter from the last active
root back to that already-completed parent reset the host-side status to
`initialized`. `thread/fork` correctly refused to fork because it requires an
idle completed durable thread.

The correction is deliberately narrow:

1. activating a loaded parent may restore its exact immutable sequence of
   completed turn IDs;
2. a non-empty, unique sequence is required;
3. an already-known sequence must match byte-for-byte or activation fails
   closed;
4. only a restored completed sequence marks the selected thread `completed`;
5. the subsequent App Server `thread/fork` response must still prove the exact
   inclusive history;
6. fresh fork activation continues to use adapter-owned history;
7. the live-search command now has an explicit `--resume-workspace` mode that
   reuses frozen manifests instead of allocating a new run.

This does not allow a warning, interrupted turn, missing response, missing
usage, or failed turn to count as success. It does not change any M2 numerical
limit, safe API method, scientific evaluator rule, or Native v2 route.

## Verification before the live run

- Ruff: passed.
- mypy: passed for 165 source files.
- focused M1-M5/provider/fork tests: 284 passed.
- recorded current-HEG M5 integration: passed.
- Native v2 real-provider smoke: passed with one completed turn, no repair,
  exact final usage of 10,534 tokens.
- App Server artifact parity: passed, 131 files and 7 tests.
- final full suite: 1,050 collected, 1,023 passed, 25 failed, 2 errors,
  1 warning. The failure/error IDs exactly matched the accepted M4 baseline;
  all 21 M5 tests and both parent-switch regression tests passed.

The post-failure parent-switch correction has focused regression coverage for:

- switching away from and back to a completed parent;
- exact inclusive child fork after the switch;
- rejection of changed completed-turn history;
- retained provider idempotency;
- deterministic M5 scheduling, Search Memory, resume, and crash boundaries.

## Remaining gate

The corrected code has not been exercised by another live provider search.
The retained workspace is intentionally resumable, but resuming it requires a
new explicit operator authorization. Until that resumed acceptance run
completes generation 1 and all final gates pass:

```text
M5 implementation: PASS_PENDING_FINAL_GATES
M5 live acceptance: NO-GO
M6 authorization: NO
```

The existing safe API still cannot causally restrict k-switch, fanout, or
relocation candidates to a witness-selected edge or vertex. M5 does not expand
that API; this remains an evidence-backed limitation for a later independent
operator decision.
