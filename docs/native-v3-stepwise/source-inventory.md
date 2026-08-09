# Native v3 preserved-source inventory

This report is the read-only donor map for Native v3 Step 02. It records the
preserved refs that later steps may inspect. It does not merge, rebase,
cherry-pick, import, or restore any donor code.

## Baseline and comparison rule

The operator-approved Native v2 base is:

```text
d847dc688e8a91ee7215b100852a2f0bb96f95ad
Remove fake App Server EOF race
```

Issue 24 still names `origin/main` at
`725d4d84566473e442097046f240c1af2b8e0132`. Step 01 superseded that revision
because it had reverted compact state persistence and caused multi-gigabyte
SQLite growth. This inventory therefore follows the accepted Step 01 base and
records the mismatch rather than changing branch ancestry.

Every comparison below is the net tree change from the corrected base to the
named ref. The corrected base is the exact merge base and an ancestor of all
five refs.

## Reproduce the ref check

Run this command from the dedicated worktree:

```bash
while read -r ref expected; do
  actual="$(git rev-parse --verify "${ref}^{commit}")" || exit 1
  test "$actual" = "$expected" || {
    echo "$ref: expected $expected, found $actual" >&2
    exit 1
  }
done <<'EOF'
native-v3-wip-71a9cc2 71a9cc271fafc27fa35df40591664535a5d45dd0
native-v3-before-v2-restore 2e61da4cbf81189daae15ea75994b9cbb1086327
before-final-v2-restore cf1a8a6517a7db1df06ec920aabe8dc1572cccd0
huj-nie-dziala 0343b69cdc2754aa2a7e4a5741aa6e69e9e80016
v2-255d55b-before-gzip-restore 420f1861d2f9fcd4e95fa90c1806b027df93d4b2
EOF
```

Expected result: exit status zero and no output.

## Preserved ref topology

| Ref | Full tip SHA | Full parent SHA | Tip subject and author time | Commits ahead | Role | Risk |
| --- | --- | --- | --- | ---: | --- | --- |
| `native-v3-wip-71a9cc2` | `71a9cc271fafc27fa35df40591664535a5d45dd0` | `de34a041f4ecd4c898a34b08f676a628fd0eed95` | `Add dashboard copy-all shortcut`, 2026-08-05 01:51:54 +01:00 | 4 | Preferred source for isolated Native v3 core logic | Its experiment integration, provider transport, event sinks, configuration, and dashboard are mixed with Native v2 behavior |
| `native-v3-before-v2-restore` | `2e61da4cbf81189daae15ea75994b9cbb1086327` | `f65d225375e006cf4489c0cff79e1468e25893d1` | `Restore original v2 experiment configuration`, 2026-08-05 03:30:46 +01:00 | 16 | Source for later scheduler, provider-batch, persistence, and integration forensics | Contains explicit Native v2 configuration, logging, transport, and telemetry restoration; never restore files wholesale |
| `before-final-v2-restore` | `cf1a8a6517a7db1df06ec920aabe8dc1572cccd0` | `420f1861d2f9fcd4e95fa90c1806b027df93d4b2` | `Restore compressed pre-v3 runtime`, 2026-08-05 03:45:56 +01:00 | 17 | Boundary snapshot after the compressed Native v2 restoration | No net Native v3 source remains; only `experiment.toml` differs from the corrected base |
| `huj-nie-dziala` | `0343b69cdc2754aa2a7e4a5741aa6e69e9e80016` | `8b9f3b4986e3a562b00ad8ece54fbf3012c97b79` | `Restore compatible Native v2 implementation`, 2026-08-05 03:39:12 +01:00 | 18 | Negative-test and regression forensics only | Destructive Native v2 restoration erased Native v3 files and changed broad runtime behavior |
| `v2-255d55b-before-gzip-restore` | `420f1861d2f9fcd4e95fa90c1806b027df93d4b2` | `f65d225375e006cf4489c0cff79e1468e25893d1` | `Revert "Add dashboard copy-all shortcut"`, 2026-08-05 03:42:37 +01:00 | 16 | Historical Native v2 and artifact-layout evidence only | Its tree still contains earlier Native v3 work, but the ref is not an implementation donor |

For every row, the ref is strictly ahead of the corrected base: the base is
zero commits ahead, the ref is the number of commits shown above, the merge
base is `d847dc688e8a91ee7215b100852a2f0bb96f95ad`, and the ref is not an ancestor
of the base.

## Net file inventory

### Shared Native v3 tree

`native-v3-wip-71a9cc2` has 69 changed paths against the corrected base:
22 modified, 8 deleted, and 39 added. The historical
`v2-255d55b-before-gzip-restore` ref retains the same core path set but is not
a donor.

Production paths:

```text
M src/mutation_forge/backends/heg.py
M src/mutation_forge/cli.py
M src/mutation_forge/events.py
M src/mutation_forge/experiment/__init__.py
M src/mutation_forge/experiment/config.py
M src/mutation_forge/experiment/layout.py
M src/mutation_forge/experiment/lock.py
D src/mutation_forge/experiment/native.py
A src/mutation_forge/experiment/native_v3.py
M src/mutation_forge/experiment/provider.py
D src/mutation_forge/experiment/rebuild.py
M src/mutation_forge/experiment/service.py
M src/mutation_forge/experiment/sessions.py
M src/mutation_forge/experiment/state.py
M src/mutation_forge/experiment/status.py
A src/mutation_forge/native_v3/__init__.py
A src/mutation_forge/native_v3/baselines.py
A src/mutation_forge/native_v3/calibration.py
A src/mutation_forge/native_v3/canonical.py
A src/mutation_forge/native_v3/contracts.py
A src/mutation_forge/native_v3/evaluation.py
A src/mutation_forge/native_v3/heg_scoring.py
A src/mutation_forge/native_v3/interpreter.py
A src/mutation_forge/native_v3/persistence.py
A src/mutation_forge/native_v3/provider.py
A src/mutation_forge/native_v3/randomness.py
A src/mutation_forge/native_v3/scheduler.py
A src/mutation_forge/native_v3/scoring.py
A src/mutation_forge/native_v3/selection.py
A src/mutation_forge/native_v3/telemetry.py
A src/mutation_forge/native_v3/verification.py
M src/mutation_forge/output/interactive_dashboard.py
```

Test paths:

```text
D tests/integration/test_native_experiment.py
A tests/integration/test_native_v3_pipeline.py
M tests/unit/test_experiment.py
M tests/unit/test_interactive_dashboard.py
M tests/unit/test_native_progress.py
D tests/unit/test_native_resume.py
D tests/unit/test_native_selection.py
A tests/unit/test_native_v3_baselines.py
A tests/unit/test_native_v3_calibration.py
A tests/unit/test_native_v3_config.py
A tests/unit/test_native_v3_contracts.py
A tests/unit/test_native_v3_evaluation.py
A tests/unit/test_native_v3_heg_scoring.py
A tests/unit/test_native_v3_interpreter.py
A tests/unit/test_native_v3_persistence.py
A tests/unit/test_native_v3_provider.py
A tests/unit/test_native_v3_scheduler.py
A tests/unit/test_native_v3_scoring.py
A tests/unit/test_native_v3_selection.py
A tests/unit/test_native_v3_telemetry.py
A tests/unit/test_native_v3_verification.py
D tests/unit/test_state_rebuild.py
```

Prompt, schema, semantic-asset, and documentation paths:

```text
D configs/native/baseline-rankers.json
D configs/native/generated-policy.schema.json
A configs/native/generated-program-batch.schema.json
A configs/native/native-v3-action-registry.json
A configs/native/native-v3-baseline-programs.json
A configs/native/native-v3-context.schema.json
A configs/native/native-v3-program.schema.json
A configs/native/native-v3-selector-registry.json
A configs/native/native-v3-semantics.md
M docs/EXPERIMENT_WORKFLOW.md
M prompts/native/repair.md
M prompts/native/request.md
M prompts/native/system.md
M README.md
M experiment.toml
```

No rename was detected.

### Ref-specific differences

- `native-v3-before-v2-restore` has 71 changed paths: the shared tree plus
  modified `src/mutation_forge/experiment/generation.py` and
  `src/mutation_forge/output/rich_live.py`.
- `v2-255d55b-before-gzip-restore` has the same 71-path shape, but its tip
  explicitly reverts the dashboard copy-all change in
  `src/mutation_forge/output/interactive_dashboard.py` and
  `tests/unit/test_interactive_dashboard.py`. It remains historical evidence
  only.
- `before-final-v2-restore` has no net Native v3 path. Its only net change is
  `experiment.toml`, including experiment identity and runtime-limit changes.
- `huj-nie-dziala` has no added Native v3 path. Its 70-path net diff contains
  62 modifications and 8 deletions across the existing Native v2 runtime,
  tests, prompts, documentation, CI, and configuration. The destructive
  deletions include `src/mutation_forge/experiment/control.py`,
  `src/mutation_forge/experiment/json_io.py`,
  `src/mutation_forge/experiment/rebuild.py`,
  `src/mutation_forge/output/display_ids.py`,
  `tests/fixtures/bin/codex`, `tests/unit/test_json_io.py`,
  `tests/unit/test_native_selection.py`, and
  `tests/unit/test_state_rebuild.py`.

## Selected donor fragments

The source coordinates below are forensic pointers, not permission to copy a
whole file. A later ticket must re-read the cited code and its tests against
the then-current Native v2 interfaces.

### Canonical identity and static contracts

- Donor: `native-v3-wip-71a9cc2` at
  `71a9cc271fafc27fa35df40591664535a5d45dd0`.
- Production:
  `src/mutation_forge/native_v3/canonical.py`:
  `parse_strict_json`, `canonical_json_bytes`, `domain_hash`, `program_hash`;
  `src/mutation_forge/native_v3/contracts.py`:
  `validate_program`, `validated_program_artifact`.
- Tests: `tests/unit/test_native_v3_contracts.py`, especially duplicate
  key/float rejection, canonical hashing, typed selector/action validation,
  and static repeat-cost coverage.
- Assets: `configs/native/native-v3-program.schema.json` and
  `configs/native/native-v3-context.schema.json`.
- Reason: strict JSON, canonical hashes, and typed AST validation have no
  experiment-service dependency.

### Deterministic DSL interpreter

- Donor: `native-v3-wip-71a9cc2` at
  `71a9cc271fafc27fa35df40591664535a5d45dd0`.
- Production: `src/mutation_forge/native_v3/interpreter.py`:
  `invoke_program`, `_selector`, `_apply_action`, `_execute_node`, `_Overlay`.
- Tests: `tests/unit/test_native_v3_interpreter.py`, especially overlay
  rollback, invalid-final-graph handling, uncatchable selector budget,
  seeded tie handling, replay, and exhaustive order-five reachability.
- Assets: `configs/native/native-v3-selector-registry.json`,
  `configs/native/native-v3-action-registry.json`, and
  `configs/native/native-v3-semantics.md`.
- Reason: the bounded overlay interpreter and registries are separable from
  experiment orchestration.

### Streaming scheduler and residual retry

- Donor: `native-v3-before-v2-restore` at
  `2e61da4cbf81189daae15ea75994b9cbb1086327`.
- Production: `src/mutation_forge/native_v3/scheduler.py`:
  `StreamingEpochScheduler`, `deterministic_interleave`,
  `build_episode_shards`, `split_residual_shard`, `SchedulerConfig`.
- Tests: `tests/unit/test_native_v3_scheduler.py`, especially provider and
  evaluator overlap, independent batch-entry publication, in-flight aliases,
  completion-order independence, residual splitting, one infrastructure
  retry, and recovered batch handling.
- Reason: scheduler behavior is expressed through injected provider and
  evaluator callables. It must still be ported fragment by fragment.

### Batched provider request and repair semantics

- Donor: `native-v3-before-v2-restore` at
  `2e61da4cbf81189daae15ea75994b9cbb1086327`.
- Production: `src/mutation_forge/native_v3/provider.py`:
  `build_provider_request`, `NativeV3Provider.call_streaming`,
  `NativeV3Provider._generate`, `parse_persisted_response`,
  `repair_persisted_batch`, `_parse`.
- Tests: `tests/unit/test_native_v3_provider.py`, especially bounded parent
  ASTs, frozen transport artifact paths, fail-closed parent sets, independent
  batch entries, retain-before-publish, single repair of a wholly invalid
  batch, and actual-request token ceilings.
- Assets: `configs/native/generated-program-batch.schema.json` and
  `prompts/native/{system,request,repair}.md`.
- Reason: frozen slots, per-entry validation, raw-response retention, and
  bounded repair are candidate semantics. The underlying Native v2 provider
  transport and artifact writer are not donors.

### Evaluation and HEG evidence

- Donor: `native-v3-wip-71a9cc2` at
  `71a9cc271fafc27fa35df40591664535a5d45dd0`.
- Production: `src/mutation_forge/native_v3/evaluation.py`:
  `evaluate_episode`, `evaluate_heg_shard`, `make_heg_shard_evaluator`;
  `src/mutation_forge/native_v3/heg_scoring.py`:
  `HegScoreEvidenceAdapter.score`, `merge_score_evidence`.
- Tests: `tests/unit/test_native_v3_evaluation.py` and
  `tests/unit/test_native_v3_heg_scoring.py`, especially no-plan curves,
  timeout intervals, component exactness, monotone evidence, worker
  fail-closed behavior, and witness feature extraction.
- Reason: these fragments are behind backend/scorer protocols. A later port
  must reuse the current HEG backend, scorer, validation, and exact pipeline;
  the sibling HEG repository remains read-only.

### Interval scoring and cross-panel selection

- Donor: `native-v3-wip-71a9cc2` at
  `71a9cc271fafc27fa35df40591664535a5d45dd0`.
- Production: `src/mutation_forge/native_v3/scoring.py`:
  `EnergyScale.build`, `ScoreEvidenceCache`, `aggregate_order_balanced`,
  `metropolis_accepts`; `src/mutation_forge/native_v3/selection.py`:
  `freeze_promotion_shortlist`, `validated_global_best`,
  `missing_current_manifest_evaluations`.
- Tests: `tests/unit/test_native_v3_scoring.py` and
  `tests/unit/test_native_v3_selection.py`, especially exact intervals,
  mixed-radix ordering, cache protocol identity, frozen Metropolis vectors,
  manifest locking, shortlist diversity, and locked validation.
- Reason: the exact arithmetic and selection rules are policy-level functions
  without transport ownership.

### Dual exact-verification supervisor

- Donor: `native-v3-wip-71a9cc2` at
  `71a9cc271fafc27fa35df40591664535a5d45dd0`.
- Production: `src/mutation_forge/native_v3/verification.py`:
  `graph_content_hash`, `verify_heg_primary`, `verify_independent_python`,
  `VerificationSupervisor`, `VerificationSupervisor.recover_pending`.
- Tests: `tests/unit/test_native_v3_verification.py`, especially locked
  limits, persist-before-verify, verifier ordering, deduplication, recovery,
  and independent cycle rejection.
- Reason: the durable candidate boundary and verifier ordering are
  encapsulated. A heuristic zero remains only a submission to exact
  verification.

### Semantic persistence

- Donor: `native-v3-before-v2-restore` at
  `2e61da4cbf81189daae15ea75994b9cbb1086327`.
- Production: `src/mutation_forge/native_v3/persistence.py`:
  `SemanticRecord`, `NativeV3Persistence`, `commit_semantic`,
  `semantic_checkpoint`, `read_connection`, `_writer_main`.
- Tests: `tests/unit/test_native_v3_persistence.py`, especially idempotent
  commits, fail-closed conflicts, semantic-cache-only SQLite, and read-only
  readers.
- Reason: the single-writer semantic store is separable, but later telemetry
  changes make the file a fragment donor only. Provider-turn artifacts must
  stay outside this semantic database and retain Native v2 relative paths,
  encodings, compression, schemas, RPC/event captures, identity, provenance,
  usage, and failure artifacts.

### Telemetry, calibration, baselines, and randomness

- Donor: `native-v3-wip-71a9cc2` at
  `71a9cc271fafc27fa35df40591664535a5d45dd0`.
- Production: `src/mutation_forge/native_v3/telemetry.py`:
  `summarize_scheduler_telemetry`;
  `src/mutation_forge/native_v3/calibration.py`:
  `calibrate_batch_size`, `select_batch_size`;
  `src/mutation_forge/native_v3/baselines.py`: `load_baseline_programs`;
  `src/mutation_forge/native_v3/randomness.py`:
  `derive_seed64`, `weighted_index`.
- Tests: `tests/unit/test_native_v3_telemetry.py`,
  `tests/unit/test_native_v3_calibration.py`,
  `tests/unit/test_native_v3_baselines.py`, and the SplitMix/unbiased-choice
  cases in `tests/unit/test_native_v3_contracts.py`.
- Asset: `configs/native/native-v3-baseline-programs.json`.
- Reason: these are isolated derived metrics, calibration gates, fixed DSL
  baselines, and versioned deterministic primitives.

### Configuration fragment

- Donor: `native-v3-before-v2-restore` at
  `2e61da4cbf81189daae15ea75994b9cbb1086327`.
- Production fragment: `NativeV3Config` and its bounded parser constraints in
  `src/mutation_forge/experiment/config.py`.
- Tests: `tests/unit/test_native_v3_config.py`, especially fixed scheduler
  parameters, disjoint development and validation seeds, rejection rather
  than migration of a Native v2 config, and absolute auth paths.
- Risk: the complete file changes the experiment schema, removes Native v2
  fields, and changes authentication configuration. It is not a whole-file
  donor.

## Unsafe wholesale sources

The following paths may contain useful evidence but must not be restored as
files:

- `src/mutation_forge/experiment/native_v3.py`: the large adapter couples
  scheduler, provider, persistence, and verification to experiment state,
  sessions, observer events, HEG paths, and provider-turn recording. Later
  tickets may inspect `_assets`, `preflight`, scheduler construction, and
  validation helpers, but must integrate against current Native v2 services.
- `src/mutation_forge/experiment/provider.py`: donor history changes auth
  isolation, lifecycle, retry prefixes, byte limits, timeouts, logging, and
  provider-call semantics. Native v2 transport and artifact writing are the
  authority.
- `src/mutation_forge/experiment/{config,generation,state,service,sessions,status}.py`,
  `src/mutation_forge/events.py`, `src/mutation_forge/cli.py`, and
  `experiment.toml`: each crosses the Native v2/v3 boundary or was touched by
  a restoration commit.
- `src/mutation_forge/output/interactive_dashboard.py` and
  `src/mutation_forge/output/rich_live.py`: donor event reducers introduce
  `provider_call_*` and verification-backpressure state that conflicts with
  Native v2 `provider_turn_*` event semantics.
- `tests/integration/test_native_v3_pipeline.py`: useful as a behavioral map,
  not as a test to restore wholesale; it couples all of the unsafe integration
  surfaces.

`huj-nie-dziala` is negative evidence only. In particular,
`src/mutation_forge/experiment/native.py` and its tests document durable turn
retention, charged-versus-uncharged retry, checkpoint/ledger accounting, and
artifact retry prefixes that Native v3 must not regress. No implementation
fragment is selected from that ref.

`v2-255d55b-before-gzip-restore` is historical Native v2 and artifact evidence
only. No `src/mutation_forge/native_v3` symbol from that ref is selected as a
donor.

## Constraints for later tickets

- Re-read each exact donor ref and cited test before porting a fragment.
- Port the smallest behavior that passes a focused test; do not merge,
  cherry-pick, or restore an entire donor file.
- Keep Native v2 as the default, runnable, and resumable path.
- Do not change Native v2 App Server transport, authentication, lifecycle,
  retry, token accounting, or provider-turn writing.
- Keep Native v3 semantic products outside provider-turn directories.
- Do not import or reference preserved branch names from production modules.
- Do not modify historical Stage 1 through Stage 7 runners.
- Keep `../heg` read-only and reuse the current HEG backend and exact
  verification pipeline.

## Step 02 result

This step changes documentation only. It introduces no runtime imports,
configuration changes, prompt changes, schemas, tests, or production behavior.
The Native v2 regression smoke and provider-turn artifact contract established
in Step 01 remain unchanged.

Known limitation: this inventory identifies candidate fragments from preserved
history. It does not certify those fragments against the future interfaces
that later steps will introduce.

STOP — waiting for operator acceptance
