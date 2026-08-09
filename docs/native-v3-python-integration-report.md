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

## Baseline comparison

Current `main` collected 637 tests: 610 passed, 25 failed, and 2 errored.
The exact current-main node IDs are stored under
`m8-rc1/current-main/current-main-full-suite.json`. Historical M7 counts are
not used as the M8 baseline.

The final integration result, release-candidate campaign, replay hashes, and
draft pull request are appended after final verification.
