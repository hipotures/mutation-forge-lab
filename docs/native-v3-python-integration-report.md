# Native v3 ordinary-Python integration report

## Scope

M8 reconciles the completed M1–M7 migration with current `main` and prepares a
merge-ready release candidate. It does not merge to `main`, change the Native
v2 default, modify `experiment.toml`, or start issues #35–#42.

## Provenance

- Current-main base: `daf7ab5c95a36c29842e6703705a381080edcfd5`
- Accepted M7 head: `728ea5f222c72184a9bca8ef69dfea90121e6fd4`
- Integration branch: `integration/native-v3-python-rc1`
- M7 archive branch: `archive/native-v3-python-m7-728ea5f`
- M7 annotated tag: `native-v3-python-m7-complete`
- Durable M8 evidence root:
  `/home/user/DEV/mutation-forge-lab-evidence/m8-rc1`

The current-main baseline used Python 3.14.4, uv 0.11.9, Codex CLI 0.147.0,
and a clean sibling HEG checkout at
`27cbec9c2307b6ea5f936f858821d11d808b68f3`.

## Preserved-history merge

The non-fast-forward merge commit is
`e024a7838cdfe76dc2d5172828c7958b4b374a9e`.

| Conflict | Resolution | Reason |
|---|---|---|
| `experiment.toml` | Kept current `main` byte-for-byte. | Native v2 must remain the unchanged default and the user's configuration must not be modified. |
| `backends/base.py` and related HEG behavior | Retained current-main Native v2 fallback behavior, added the migration's typed rewrite/scoring contracts, and made the Python HEG adapter fail closed if the authoritative worker returns no evidence. | Native v2 behavior and the no-fallback Python scientific boundary are both required. |
| `experiment/native.py` | Retained current-main Native v2 archive-context rendering and coordinator callback. The Python M5 Search Memory remains a separate, source-free and identity-free projection. | This preserves current-main Native v2 without leaking its host metadata into Python model prompts. |
| Stage 3 fake App Server output | Retained current-main crash-close behavior but restored blocking idle reads for persistent multi-turn/fork tests. | An idle durable App Server is not EOF; the timeout fixture incorrectly killed exact-parent fork sequences. |

No JSON-DSL production path was restored.

## Independent M7 verification

M8 independently verified M7 before accepting #56:

- the migration branch was clean at the expected head;
- Native v2 remained default and Python preview remained opt-in;
- removed DSL modules/assets were absent and not lazily imported;
- offline replay matched 14 programs across 28 cases;
- durable post-cleanup evidence recorded 16/16 terminal slots and a complete
  generation 1;
- resume repeated no immutable terminal work;
- Native v2 smoke and App Server parity passed;
- the full-suite failure/error set matched the reported M7 baseline;
- `experiment.toml` was unchanged.

## Integration hardening

M8 added fail-closed checks discovered during independent integration:

- a candidate score timeout without safe partial evidence now yields
  `INCONCLUSIVE_UNSAFE_TIMEOUT` and a full uncertainty interval, never
  scientific `COMPLETE`;
- the Python HEG adapter rejects a missing authoritative worker response
  without entering the Native v2 fallback;
- retained candidates are checked against frozen slot kind, parent assignment,
  parent identities, Search Memory identity, and prior matching duplicate;
- retained Search Memory is deterministically rebuilt from prior generations
  and compared byte-for-byte before reuse;
- the preview supports a controlled resumable stop at the next durable
  candidate boundary.

## Baseline and final verification

Current `main` collected 637 tests: 610 passed, 25 failed, and 2 errored.
The exact current-main node IDs are stored under
`m8-rc1/current-main/current-main-full-suite.json`. Historical M7 counts are
not used as the M8 baseline.

The final implementation head before this report and CI-only follow-up was
`5f7f259e2cc92898c324e7032713c1a60707bad0`. It collected 920 tests: 898
passed, 20 failed, and 2 errored. It introduced no new failure or error node
ID and resolved five current-main failures:

- two Stage 3 prompt consistency failures were resolved by aligning the
  renderer with the frozen 500-node sandbox contract;
- the graceful-stop dashboard test accounts for intentional table ellipsis;
- two Stage 4 App Server fork-isolation tests pass after the merge resolution.

Ruff, mypy, the 709-test self-contained CI set, the focused M1–M8 tests,
App Server artifact parity, the M7 and M8 durable replays, and a real Native
v2 smoke all passed. The CI workflow provisions its test-only dependencies
explicitly: a pinned sibling HEG checkout, `bubblewrap`, and an offline Codex
executable that cannot start App Server or make model calls.

Draft pull request [#60](https://github.com/hipotures/mutation-forge-lab/pull/60)
targets `main`. It remains a draft and is not authorized for merge or for a
default switch.

## Durable release-candidate campaign

The one authorized M8 campaign used experiment
`native-v3-python-m6-20260809T030126Z` under:

```text
/home/user/DEV/mutation-forge-lab-evidence/m8-rc1/live-rc
```

It completed the fixed population without replacement slots:

| Measure | Result |
|---|---:|
| Planned / terminal / pending | 16 / 16 / 0 |
| Generation 0 | 8 roots |
| Generation 1 | 4 children + 4 roots |
| Evaluated | 12 |
| Contract-invalid | 4 |
| Duplicate / provider-failed | 0 / 0 |
| Candidate program turns / repairs | 20 / 4 |
| Provider turns / warnings | 21 / 17 |
| Total tokens | 133,429 |
| Sandbox starts | 24 |
| Sandbox failures / timeouts / rotations | 0 / 0 / 0 |
| Exact-verifier submissions / records | 0 / 0 |

The host requested a controlled stop after all eight generation-0 roots and
the first two generation-1 slots were terminal. The stopped state was
`blocked`, `operator_stop`, and resumable, with exactly six pending slots. One
continuation submitted those six slots and reached `generation_budget`.
All 438 immutable pre-resume artifact hashes remained byte-identical.

Eight roots and four children were evaluated. All four evaluated children
differed from their parents in both source and behavior. The retained traces
recorded four `NoPlan` outcomes and no program failure. Nonzero action families
were `add_edge`, `edge_fanout`, `k_switch`, `relocate_endpoint`, and
`remove_edge`; selector profiles were derived from actual safe-API traces.
All evaluated candidates received the same two-case development panel and
budget.

The offline M8 replay reproduced all 12 evaluated programs and 24 cases with
matching semantic traces. Its internal report hash is:

```text
665fd9ee6515e6b7a28fbd1a8703b0f8f3939477fb0ebc3c6def5e76a2b3af56
```

The campaign and replay report `dsl_runtime_used=false`,
`native_v2_default=true`, and `safe_api_expanded=false`.

## Evidence index

- M7 independent verification: `m8-rc1/m7-verification`
- Current-main baseline: `m8-rc1/current-main`
- Integration offline gates: `m8-rc1/integration`
- M8 live campaign, sources, provider artifacts, snapshots, and replay:
  `m8-rc1/live-rc`
- Final static, smoke, full-suite, and CI evidence: `m8-rc1/final`

Large runtime trees remain outside Git. This repository commits only this
human-readable report, the operator guide, the roadmap, and the small example
configuration. The exact current-main failure/error delta and final command
results are recorded in the durable manifests, M8 issue
[#59](https://github.com/hipotures/mutation-forge-lab/issues/59), and draft
pull request.
