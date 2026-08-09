# Native v3 Python M1: Native v2 Smoke Triage

## Scope

This document records migration gate G0 only. It diagnoses the mandatory Native
v2 smoke failure observed while reviewing ordinary-Python M1 commit
`f79bc72344c4b40d792d39924aae77ebaff1e796`.

G0 made no production-code, test, configuration, authentication, provider,
workspace, routing, or `experiment.toml` changes. It did not modify M1, start
issue #51, or implement any M2 worker or sandbox behavior.

## Verdict

The three-way comparison is **case D / NO-GO**.

The exact smoke command cannot run on current `origin/main` because that tree
does not contain the Native v2 smoke harness or its parity resources. The audit
base and M1 revisions are directly comparable and fail identically under the
current installed Codex CLI, but that two-way result cannot establish the state
of current production Native v2 on `origin/main`.

The formal status remains:

```text
M1 implementation: PASS
M1 migration gate: BLOCKED_BY_PREEXISTING_OR_EXTERNAL_V2_SMOKE_FAILURE
M2 authorization: NO
```

Issue #50 must remain open. Issue #51 remains blocked.

## Compared revisions

The comparison used fetched `origin/main` and the two immutable migration
revisions:

| Revision | Commit | Git tree | Exact smoke result |
| --- | --- | --- | --- |
| current `origin/main` | `daf7ab5c95a36c29842e6703705a381080edcfd5` | `b7d0fc0e70901dbafee56045b333d312dfe3717e` | Exit 2 before provider startup: `scripts/native_v2_smoke.py` is absent |
| migration audit base | `594ac0a5baf753f153af4510857962b1bef93cba` | `f1a242edfb2b4e72ffd165f8d97ca379f0fef3df` | Exit 1 after four failed provider attempts |
| ordinary-Python M1 | `f79bc72344c4b40d792d39924aae77ebaff1e796` | `0428092c6178fb37065e93a1f514afeed96ae0e5` | Exit 1 after four failed provider attempts, identical to the audit base |

The exact command requested for every revision was:

```bash
uv run python scripts/native_v2_smoke.py
```

Current `origin/main` also lacks:

- `scripts/appserver_artifact_parity.py`; and
- `tests/fixtures/native_v2_appserver_artifact_contract.json`.

Consequently, copying only the smoke script outside the worktree would not
create an exact comparison. It would require staging a different harness and
parity fixture and changing project-relative resolution. G0 did not do that.
No provider or model turn was started for `origin/main`.

## Controlled environment

The two valid smoke runs used the same inherited environment, authentication
profile source, model, effort, external uv environment, installed Codex binary,
and sibling HEG checkout. They ran sequentially in clean detached worktrees
directly below `/home/user/DEV`, so the unchanged relative `../heg` contract
resolved identically.

| Control | Value |
| --- | --- |
| Codex command | `/usr/bin/codex` |
| Resolved installed file | `/usr/lib/node_modules/@openai/codex/bin/codex.js` |
| Installed-file SHA-256 | `134063e133f0b4244fa3b251acf973d4fe4b4aeeacbdc135211bf480f59f1477` |
| Codex version | `codex-cli 0.147.0` |
| Model | `gpt-5.6-luna` |
| Effort | `high` |
| Maximum model turns | `2` |
| Shared uv environment | `/tmp/mutation-forge-native-v2-g0-uv.G2A4WQ` |
| HEG checkout | `/home/user/DEV/heg` |
| HEG commit | `27cbec9c2307b6ea5f936f858821d11d808b68f3` |
| HEG state | clean |

The installed-protocol audit completed successfully:

```text
command: python /home/user/.codex/skills/codex-app-server/scripts/audit_installed_protocol.py
exit: 0
initialize: PASS
skills/list: PASS
thread/start: PASS
strict-config startup: PASS
```

Relevant global feature state reported by `codex features list` was:

| Feature | Stage | Effective global value |
| --- | --- | --- |
| `code_mode` | under development | `false` |
| `code_mode_buffered_exec` | under development | `false` |
| `code_mode_host` | stable | `true` |
| `code_mode_only` | under development | `false` |
| `use_linux_sandbox_bwrap` | removed | `true` |

The smoke's unchanged strict App Server argv nevertheless includes:

```text
--strict-config
--disable code_mode_host
--enable use_linux_sandbox_bwrap
```

Thus the warning is associated with the smoke's explicit per-process disabling
of `code_mode_host`, not with the global feature value being false. G0 did not
remove that disable, enable Code Mode, or otherwise change the configuration.

## Audit-base run

The clean detached worktree was:

```text
/home/user/DEV/mforge-native-v2-g0-594ac0
```

The retained provider artifact root is:

```text
/tmp/mutation-forge-native-v2-smoke/native-v2-smoke-20260807T201108Z-dc8c97ba
```

The doctor phase passed. The experiment then made four attempts: the initial
attempt followed by `retry-01`, `retry-02`, and `retry-03`. Every attempt
produced the same lifecycle:

1. RPC response `initialize` id 0 succeeded.
2. RPC response `skills/list` id 1 succeeded.
3. RPC responses `skills/config/write` ids 2 through 7 succeeded.
4. RPC response `skills/list` id 8 succeeded.
5. RPC response `thread/start` id 9 succeeded and returned a thread ID.
6. RPC response `turn/start` id 10 succeeded and returned an in-progress turn
   ID.
7. Notification `remoteControl/status/changed`.
8. Notification `thread/started`.
9. Notification `warning`.
10. Notification `thread/status/changed`.
11. Notification `turn/started` with status `inProgress`.
12. Notification `item/started` for the user message.
13. Notification `item/completed` for the user message.
14. Notification `thread/status/changed` with status `idle`.
15. RPC response `turn/interrupt` id 11 succeeded.
16. Notification `turn/completed` with status `interrupted`.

The exact warning was:

```json
{
  "method": "warning",
  "params": {
    "threadId": "<current provider thread UUID>",
    "message": "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`."
  }
}
```

There was no assistant item, final response, or token-usage event. The adapter
classified every attempt as:

```text
ProtocolError: unknown app-server notification: warning
```

All four turn manifests were terminal failures with
`request_accepted=false`, `content_received=false`, null provider thread/turn
IDs, `usage_final_exact=false`, and unknown usage quality. The four bounded
provider-raw artifacts were byte-identical and had SHA-256:

```text
2298b964a85480b26f725c79a74843d58511bf1152d8af34487808fff920327c
```

After the provider failures, the smoke's real-provider structural verification
raised:

```text
ParityError: real-provider turn does not match any frozen structural profile
```

## M1 run

The clean detached worktree was:

```text
/home/user/DEV/mforge-native-v2-g0-f79bc7
```

The retained provider artifact root is:

```text
/tmp/mutation-forge-native-v2-smoke/native-v2-smoke-20260807T201124Z-51dbdd6e
```

The doctor phase passed. This run had the same four attempts, exact warning
method and payload, RPC results, ordered event sequence, interruption status,
absence of assistant output and token usage, manifest classification,
provider-raw artifact hash, and final `ParityError` as the audit-base run.

Both detached worktrees remained clean after their runs. The relevant
`isolation.py` bytes were identical between the two revisions, with SHA-256:

```text
6fbcd916844944e08ca22455466f87ee55d5994cb927abca4c6de4fc4f09ccea
```

This proves that M1 did not change the failure.

## Comparison and classification

The observed matrix is:

| Observation | `origin/main` | `594ac0` | `f79bc7` |
| --- | --- | --- | --- |
| Exact smoke harness present | No | Yes | Yes |
| Doctor reached | No | Yes | Yes |
| `thread/start` succeeded | Not run | Yes | Yes |
| `turn/start` succeeded | Not run | Yes | Yes |
| Exact warning received | Not run | Yes | Yes |
| Assistant final response | Not run | No | No |
| Successful `turn/completed` | Not run | No; `interrupted` | No; `interrupted` |
| Token usage | Not run | Absent | Absent |
| Exit status | 2 | 1 | 1 |

This is not case A, B, or C because current `origin/main` cannot execute the
same gate. It is case D:

```text
D — origin/main is structurally non-comparable; the audit base and M1 fail
identically before a successful model turn.
```

The two migration revisions establish only:

```text
594ac0 — fails under Codex CLI 0.147.0
f79bc7 — fails identically under Codex CLI 0.147.0
```

They do not establish whether the current production Native v2 implementation
on `origin/main` would pass an equivalent, current smoke contract.

## Repair assessment

G0 does not prove that the warning is a benign capability notification.

Although `thread/start` and `turn/start` both return successfully, the adapter
raises on the warning, sends an interrupt, receives no assistant response or
token usage, and ends with `turn/completed.status=interrupted`. There is no
completed control turn showing what would happen if the warning were prevented
or narrowly accepted.

Therefore G0 proposes no warning allowlist and no configuration repair. In
particular, it would be unsafe at this point to:

- ignore all App Server warnings;
- allowlist this warning without a completed control turn;
- enable Code Mode merely to silence it;
- remove strict configuration; or
- change authentication, provider artifacts, or production routing.

The next operator decision must first define or restore a comparable,
current-`main` Native v2 smoke harness. Any later diagnostic that tests a
configuration change or exact warning allowlist requires its own authorization
and must preserve fail-closed handling for unknown, changed, or additional
warnings.

## Evidence retention

Retained valid-run evidence:

- `/tmp/mutation-forge-native-v2-smoke/native-v2-smoke-20260807T201108Z-dc8c97ba`
- `/tmp/mutation-forge-native-v2-smoke/native-v2-smoke-20260807T201124Z-51dbdd6e`
- `/home/user/DEV/mforge-native-v2-g0-594ac0`
- `/home/user/DEV/mforge-native-v2-g0-f79bc7`
- `/tmp/mutation-forge-native-v2-g0-uv.G2A4WQ`

An initial setup attempt placed worktrees under `/tmp`, where the unchanged
relative `../heg` lookup could not resolve. That attempt stopped in doctor
preflight and created no App Server/provider turn. It is excluded from the
comparison. The corrected valid worktrees above used the required repository
layout and clean sibling HEG checkout.

No evidence contains modified authentication or comparison-worktree content.
