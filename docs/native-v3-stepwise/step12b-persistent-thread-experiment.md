# Native v3 Step 12B: persistent App Server thread experiment

Step 12B adds an opt-in experimental harness for comparing fresh App Server
threads, one persistent thread, and the existing four-program batch protocol.
It does not switch the production provider or change Native v2.

## Protocol

The experiment uses `gpt-5.6-luna` with `medium` reasoning effort and the same
four add-edge, remove-edge, relocation, and fanout mechanisms.

- A starts four fresh processes and ephemeral threads. Infrastructure retries
  retain separate failure prefixes and use the production limit of three.
- B starts one process and one durable thread. It sends one structured
  readable acknowledgement followed by four direct Step 12A program turns.
  The host retains the protocol hash outside the conversation. App Server
  reconnects stay inside this persistent thread.
- C uses the unchanged production provider and `TurnArtifactStore` for exactly
  one four-program batch. It performs no repair, graph evaluation, HEG call, or
  scoring.

The production `CodexAppServerAdapter.generate()` path remains fail-closed on
server retry. Only the opt-in persistent method waits for an App Server
reconnect so thread identity is retained.

## Bounded live result

The accepted evidence is in
`workspace/step12b_abc_medium_010/abc-report.json.gz`. The final A/B report was
reconstructed offline from immutable provider artifacts after B's fourth
program turn ended in a terminal infrastructure failure. This operator
analysis made no provider request and did not alter any A/B artifact. C then
ran as the only additional model turn.

| Metric | A: fresh | B: persistent | C: batch |
| --- | ---: | ---: | ---: |
| Program turns | 4 | 4, plus bootstrap | 1 batch of 4 |
| Valid programs | 2/4 | 2/4 | 0/4 |
| Canonical duplicate rate | 0 | 0 | 0 |
| Behavior duplicate rate | 0 | 0 | 0 |
| Provider wall time | 87.076 s | 123.194 s | 88.128 s |
| Time to first valid AST | 71.367 s | 57.438 s | unavailable |
| Time to four valid unique ASTs | unavailable | unavailable | unavailable |
| Input tokens | 22,870 | 22,353 | 5,504 |
| Cached-input tokens | 0 | 16,640 | 0 |
| Cache-write tokens | 0 | 0 | 0 |
| Output tokens | 1,722 | 860 | 2,875 |
| Reasoning-output tokens | 995 | 191 | 933 |
| Total tokens | 24,592 | 23,213 | 8,379 |
| Total tokens per valid program | 12,296.0 | 11,606.5 | unavailable |

A used four distinct successful thread IDs. Two uncharged infrastructure
failures were retained below the `a-slot-00` and `a-slot-01` prefixes before
their successful retries.

B used one thread ID for the bootstrap and all four program turns, with five
distinct turn IDs. The first three program turns completed; two passed the
local program contract and one failed its non-ASCII hypothesis bound. The
fourth turn exhausted five internal reconnects and ended terminally. Its
failure did not invalidate the two earlier accepted programs.

Persistent mode improved time to first valid AST by 13.929 seconds and reduced
total tokens per valid program by 689.5 tokens in this sample. This is one
bounded four-brief observation, not evidence for selecting a production
default.

## Artifact and lifecycle evidence

Every successful A/B prefix has the same 16 transport artifacts. Failed fresh
attempts have the production 12-file failure set, including request envelope,
provider failure boundary, RPC, events, wire, stdout, stderr, and transcript
digest. Provider-turn filenames and mtimes were unchanged by the offline
analysis.

C has one `generation-0000/slot-00/initial` directory. Its production
`mforge.experiment.turn-manifest.v2` is complete, has exact usage, lists 22
artifacts with matching sizes and hashes, and has no repair or evaluator
directory. Experimental semantic reports remain outside provider-turn
directories.

The deterministic suite covers:

- four fresh turns and one bootstrap plus four repeated persistent turns;
- durable-thread resume after an App Server process restart;
- invalid and terminal program turns without corruption of earlier programs;
- production-style fresh-thread retry and persistent-only reconnect handling;
- exact successful and failed provider artifact suffix sets.

## Manual operator command

Run in the dedicated worktree with a new workspace:

```bash
uv run python scripts/native_v3_persistent_experiment.py \
  --workspace workspace/step12b_operator_medium_001 \
  --auth-json /home/user/.codex/auth.json \
  --model gpt-5.6-luna \
  --effort medium \
  --turn-timeout 900
```

Expected output is one JSON A/B/C report and exit status zero. The workspace
contains `ab-report.json.gz`, `abc-report.json.gz`, A/B provider-turn
artifacts, and one production C turn below `c-reference`. A failed workspace
must be preserved and never reused.

## Known limitations

- The real sample is one run and includes provider reconnect failures.
- Neither A nor B produced four valid unique programs.
- The C batch produced no valid program, so its time-to-valid and
  cost-per-valid metrics are unavailable.
- Persistent thread selection, forks, compaction, scheduler integration, and
  production routing remain out of scope.

STOP — waiting for operator acceptance.
