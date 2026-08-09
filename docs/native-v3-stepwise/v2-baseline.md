# Native v2 baseline certification

This report certifies the Native v2 runtime before incremental Native v3 work
begins. The dedicated branch is `native-v3-stepwise`, and its worktree is:

```text
/home/user/DEV/mutation-forge-lab-native-v3-stepwise
```

## Baseline correction

Issue 23 originally named `725d4d84566473e442097046f240c1af2b8e0132`
as the base. During execution, that revision was found to have reverted compact
state persistence and to write complete evaluation JSON into SQLite. The
resulting live database grew to 6.33 GB.

The operator explicitly replaced that revision with:

```text
d847dc688e8a91ee7215b100852a2f0bb96f95ad
Remove fake App Server EOF race
```

This certification uses that corrected base. It does not merge or import any
Native v3 donor implementation.

## Repeat the bounded smoke

Run this single command from the dedicated worktree:

```console
uv run python scripts/native_v2_smoke.py
```

It creates a fresh disposable experiment under
`/tmp/mutation-forge-native-v2-smoke`, runs `mforge doctor`, completes one
bounded real-provider Native v2 experiment, runs the read-only status command,
and writes a JSON report beside the disposable workspace.

The smoke permits one generation, one candidate, at most two model turns
including one repair, one order-4 graph seed, one policy seed, and one
evaluation worker. It fails unless a model turn, accepted candidate, and
evaluation are recorded. It hashes every provider-turn artifact before and
after `experiment status` and fails if status changes any artifact.

The command consumes model tokens. Every invocation uses a new experiment ID.

## Certified baseline

- Corrected stable base:
  `d847dc688e8a91ee7215b100852a2f0bb96f95ad`.
- Clean smoke execution commit:
  `794620fe5a6b288f03bc475587779414e22d68df`.
- Donor refs inspected: `native-v3-wip-71a9cc2` (`71a9cc2`),
  `native-v3-before-v2-restore` (`2e61da4`),
  `before-final-v2-restore` (`cf1a8a6`), `huj-nie-dziala` (`0343b69`),
  and `v2-255d55b-before-gzip-restore` (`420f186`).
- Native v3 donor code used: none.
- Production runtime changes: none.
- Certification date: 2026-08-05.
- Mutation Forge environment: Python 3.12.13, uv 0.11.9, Codex CLI
  0.146.0, Linux 7.0.0-28-generic x86_64 with glibc 2.43.
- HEG: `27cbec9c2307b6ea5f936f858821d11d808b68f3`, clean.
- Native v2 regression gate: 48 passed.
- Repository checks: `uv lock --check`, Ruff, and mypy passed.
- Real-provider smoke: one model turn, one accepted candidate, and one
  evaluation.

The separate branch `fix/state-rebuild-historical-artifacts` is not part of
this baseline. It retains the offline state-rebuild repair required by the
operator's migrated live workspace.

## Commands and results

All certification commands ran from the dedicated worktree:

```console
uv run mforge doctor --heg-repo ../heg \
  --run-root /tmp/mutation-forge-native-v2-smoke/doctor-step01-d847
# PASS

uv run pytest tests/unit/test_native_v2_smoke.py \
  tests/integration/test_native_experiment.py \
  tests/unit/test_native_resume.py \
  tests/unit/test_native_selection.py \
  tests/unit/test_native_progress.py
# PASS: 48 tests

uv lock --check
uv run ruff check .
uv run mypy
# PASS

uv run python scripts/native_v2_smoke.py
# PASS
```

The successful smoke created:

- experiment ID:
  `native-v2-smoke-20260805T125439Z-5e56bbed`;
- config:
  `/tmp/mutation-forge-native-v2-smoke/configs/native-v2-smoke-20260805T125439Z-5e56bbed.toml`;
- workspace:
  `/tmp/mutation-forge-native-v2-smoke/native-v2-smoke-20260805T125439Z-5e56bbed`;
- JSON report:
  `/tmp/mutation-forge-native-v2-smoke/reports/native-v2-smoke-20260805T125439Z-5e56bbed.json`.

The report recorded exact usage of 11,929 total tokens: 8,218 input, 3,711
output, and 3,221 reasoning output. Cached input, cache-write input, and
charged failed turns were zero. Reasoning output is a subcategory; the
server-provided total remains authoritative.

## Provider-turn artifact evidence

The report contains a SHA-256 mapping for every provider-turn artifact. The
complete mapping was identical before and after the read-only status command.
The recorded tree, relative to `artifacts/generations/`, was:

```text
generation-0000/slot-00/initial/behavior.json.gz
generation-0000/slot-00/initial/canonical_response.json.gz
generation-0000/slot-00/initial/identity.json.gz
generation-0000/slot-00/initial/metadata-validation.json.gz
generation-0000/slot-00/initial/provenance.json.gz
generation-0000/slot-00/initial/slot-00.codex-profile.json.gz
generation-0000/slot-00/initial/slot-00.codex-rpc.jsonl
generation-0000/slot-00/initial/slot-00.events.jsonl
generation-0000/slot-00/initial/slot-00.output-schema.json.gz
generation-0000/slot-00/initial/slot-00.provider-raw.json.gz
generation-0000/slot-00/initial/slot-00.request.json.gz
generation-0000/slot-00/initial/slot-00.request.md
generation-0000/slot-00/initial/slot-00.response.json.gz
generation-0000/slot-00/initial/slot-00.response.md
generation-0000/slot-00/initial/slot-00.response.raw.txt
generation-0000/slot-00/initial/slot-00.stderr.txt
generation-0000/slot-00/initial/slot-00.stdout.jsonl
generation-0000/slot-00/initial/slot-00.system-prompt.md
generation-0000/slot-00/initial/slot-00.transcript.sha256
generation-0000/slot-00/initial/slot-00.transport-diagnostics.json.gz
generation-0000/slot-00/initial/slot-00.usage.json.gz
generation-0000/slot-00/initial/slot-00.wire.jsonl
generation-0000/slot-00/initial/source.py
generation-0000/slot-00/initial/turn-manifest.json.gz
generation-0000/slot-00/initial/validation.json.gz
generation-0000/slot-00/initial/worker_telemetry.json.gz
```

## Read-only status evidence

After the experiment process exited, the smoke ran:

```console
uv run mforge experiment status \
  --config /tmp/mutation-forge-native-v2-smoke/configs/native-v2-smoke-20260805T125439Z-5e56bbed.toml \
  --json
```

It returned the following relevant state:

```json
{
  "best_primary_metric": 0.0,
  "best_program_id": "g0000-slot-00",
  "configured_model_turns": 2,
  "effective_model_turns": 2,
  "evaluation_count": 1,
  "generation": 1,
  "last_error": null,
  "last_stop_reason": "generation_limit",
  "model_turns_used": 1,
  "provider_turns": 1,
  "remaining_model_turns": 1,
  "resumable": false,
  "schema_version": "mforge.experiment.status.v2",
  "state": "exhausted",
  "terminal": true,
  "total_tokens": 11929,
  "unique_candidate_count": 1
}
```

The zero heuristic metric was submitted only to ordinary evaluation. No
counterexample was verified.

## Original worktree isolation

Step 01 did not modify the original worktree. That worktree remains on
`fix/state-rebuild-historical-artifacts` with the operator-restored
`experiment.toml` as its only uncommitted file. Its Git blob remains
`31d84877070b3af8e3238509aead7bf2397d7c6c`.

## Known limitations

- This is a bounded connectivity and end-to-end regression smoke, not a
  scientific quality or performance benchmark.
- The response remains model-generated. A schema failure may consume the
  bounded repair turn and make the smoke fail closed.
- Step 01 records and hashes the Native v2 artifact tree but does not define
  the later Native v2/v3 parity contract.
- The retained JSON report and disposable workspace are local evidence, not
  committed repository artifacts.
- The corrected `d847dc6` base differs from the obsolete base originally
  written in issue 23; that correction must be stated when closing the issue.

## STOP handoff

Step 01 adds no Native v3 runtime, generated-code execution, evolutionary
search changes, sandbox changes, or transport changes. The next ticket must
begin from this clean, pushed branch and consume this report as the Native v2
regression baseline.

STOP — waiting for operator acceptance
