# Native v2 App Server artifact contract

This report freezes the provider-turn artifact boundary that Native v3 must
reuse without modification. It belongs to Native v3 Step 03 and changes no
provider transport, authentication, lifecycle, retry, token-accounting, or
artifact-writer production code.

## Baseline

The operator-approved Native v2 base remains:

```text
d847dc688e8a91ee7215b100852a2f0bb96f95ad
Remove fake App Server EOF race
```

Issue 25 still names the obsolete `725d4d8` base. Step 01 superseded that
revision because it had reverted compact state persistence. Step 02 was
accepted at `862ac7a606148f25daee731eb3c9dcd7c5bd2ef7`; the later
`364aabd24c5c75a1e3df73dd6fee845c18ce2745` progress-event fix is a separate
operator-directed commit and is part of the clean starting tip for this step.

## Authoritative command

Run the offline parity gate from the dedicated worktree:

```console
make appserver-artifact-parity
```

The command performs no inference and uses no provider credentials or model
tokens. It:

1. generates the complete deterministic fixture twice in independent
   temporary directories;
2. requires the two raw artifact trees to be byte-identical;
3. compares every raw file SHA-256 with the committed contract;
4. validates the exact file set, gzip metadata, UTF-8 and JSON/JSONL
   decodability, top-level JSON keys, schema versions, and raw-stream shapes;
5. runs the mutation tests that prove the gate fails for a removed, renamed,
   added, recompressed, or schema-key-changed artifact.

Expected result: the script prints a JSON success summary, the focused pytest
suite passes, and the command exits with status zero.

## Deterministic fixture cases

The fixture lives only in temporary test directories. Its frozen manifest is
`tests/fixtures/native_v2_appserver_artifact_contract.json`.

| Case | Frozen behavior |
| --- | --- |
| `initial-success` | A complete accepted initial turn with exact usage, valid generated policy, semantic projections, validation, identity, behavior, provenance, metadata validation, worker telemetry, and all transport streams |
| `repair-success` | A completed but validation-invalid initial turn followed by a complete valid `repair-01` turn |
| `retry-success` | A complete uncharged failed attempt archived as `turn-manifest.attempt-01.json.gz`, followed by a successful `slot-00.retry-01` transport attempt without overwriting the first attempt |
| `terminal-failure` | A complete charged provider failure with partial usage, no response projection, retained raw transport evidence, and a terminal failure manifest |

The fixture is a deterministic provider boundary, not an alternative provider
implementation. It materializes fixed redacted transport records and then
passes them through the current production `TurnArtifactStore` indexing and
verification methods. It never imports a preserved Native v3 branch.

## Frozen comparison

The committed contract has two layers:

- `raw_sha256` freezes every relative path and its exact stored bytes.
  Consequently, file removal, rename, addition, text or JSONL changes, gzip
  recompression, and decoded JSON changes all fail parity.
- `structural_profiles` freeze each turn's normalized relative file set,
  encoding or compression type, gzip `mtime`, JSON object keys or array item
  keys, and `schema_version` when present. The concrete slot prefix is
  represented as `{prefix}` so a real turn may use the ordinary slot name or
  a retry prefix without weakening the inner contract.

JSON gzip files must have gzip magic, deterministic `mtime=0`, valid UTF-8,
and valid JSON. JSONL files must contain valid JSON objects. Markdown, raw
response, stderr, system prompt, and Python source files must be UTF-8. A
transcript digest must be exactly one SHA-256 value.

For a successful Native v2 turn, the structural profile freezes the same
26-file tree recorded by Step 01, including:

- request, system prompt, output schema, raw and projected response;
- provider raw envelope, exact usage, Codex profile, transport diagnostics;
- RPC, event, stdout, stderr, wire, and transcript evidence;
- canonical response, source, validation, identity, behavior, provenance,
  metadata validation, and worker telemetry;
- `mforge.experiment.turn-manifest.v2`.

The manifest's own file list, sizes, and hashes are independently verified by
`TurnArtifactStore.verify_turn`.

## Real-provider structural smoke

Run:

```console
uv run python scripts/native_v2_smoke.py
```

This command consumes real provider tokens. In addition to the bounded Step 01
checks, it now validates every completed provider-turn manifest against one of
the frozen structural profiles and reports:

```json
{
  "checks": {
    "appserver_artifact_structure": true
  },
  "appserver_artifact_contract": {
    "turn_count": 1
  }
}
```

The turn count may exceed one when a bounded repair or retry is required. Each
turn must still match an explicit frozen profile; an unknown tree is a hard
failure.

Only values that are inherently run-specific are ignored by the real-provider
structural comparison:

- experiment and workspace paths;
- provider request, thread, turn, session, item, and RPC identifiers;
- event timestamps and emitted-time fields;
- model response text and generated source;
- token counts;
- PIDs, elapsed durations, nanosecond timings, and throughput counters;
- transcript and content hashes whose inputs contain those values.

Their field presence, container type, surrounding schema keys, artifact
encoding, and relative filename remain mandatory. The offline fixture has no
such relaxation: its complete stored bytes must match.

## Failure behavior

The gate fails closed when:

- a required path is missing or renamed;
- an unexpected path is added;
- any deterministic fixture byte changes;
- a gzip member is recompressed even if its decoded JSON is unchanged;
- a JSON schema key or schema version changes;
- gzip, UTF-8, JSON, JSONL, or transcript-digest decoding fails;
- a real turn manifest fails its internal file hash/size verification;
- a real turn does not match any frozen success, repair, retry, or failure
  structural profile.

The fixture and mutation tests use disposable pytest or system temporary
directories. They never use an experiment workspace and contain no
credentials. Secret-looking keys and private path values are absent.

## Native v3 handoff

Later Native v3 provider work must use the current Native v2 provider and
artifact writer. It may change only an outer directory prefix. Inside a
provider-turn directory, filenames, encodings, compression, schemas,
RPC/event/wire captures, identity, provenance, usage, failure evidence, and
manifest semantics must pass this gate unchanged.

Native v3 semantic products must remain outside provider-turn directories.
The batched-provider donor identified in Step 02 is not a donor for transport
or artifact persistence.

## Known limitations

- The deterministic corpus freezes the artifact boundary, not Codex model
  quality or generated-policy semantics.
- The real-provider check deliberately compares structure rather than dynamic
  bytes; the complete byte-for-byte authority is the offline fixture.
- This step does not add Native v3 parsing, batching, scheduling, or runtime
  behavior.

STOP — waiting for operator acceptance
