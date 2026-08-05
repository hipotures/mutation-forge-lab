# Native v3 Step 11 sequential cohort

The explicit `native-v3-preview` route now freezes one epoch with eight primary
slots (`slot-00` through `slot-07`). Provider generation is strictly
sequential: call 0 owns slots 00–03 and call 1 owns slots 04–07. Graph
evaluation starts only after both calls finish and remains single-process and
serial.

Before provider contact, `epoch-manifest.json.gz` records:

- every slot, its empty epoch-zero parent set, brief, and brief hash;
- the two provider-call partitions and exact rendered prompt hashes;
- model and reasoning effort;
- canonicalization, validation, interpreter, graph-runtime, evaluator, cohort,
  and provider-profile identities.

Hashes and host bookkeeping stay in the manifest. The model prompt receives
only the requested slots and briefs, the Native v3 AST schema, and the
executable selector/action/context registry.

## Validation, repair, and identity

Each returned batch entry is checked independently. A valid sibling survives
an invalid or omitted sibling. A partially valid batch is never repaired. If
all four entries are unusable, the host permits exactly one repair using the
same call, slots, and frozen snapshot.

Canonical program identity deduplicates equivalent ASTs. Every slot alias,
brief, parent set, raw provider turn, and lineage record is retained, but one
canonical program is evaluated only once. Aggregation and selection consume
canonical slot/program order, never provider response order.

The number of unique valid programs determines the epoch result:

- 8: `COMPLETE`;
- 4–7: `DEGRADED`;
- 0–3: `INCONCLUSIVE`, with a safe terminal stop.

## Provider and artifact boundary

The existing Native v2 Codex App Server transport, authentication, retry
boundary, token accounting, and turn artifact writer are unchanged. Each
four-program batch is carried inside the existing
`mforge.native.generated_policy.v1` envelope. Native v3 manifests, lineage,
validated programs, evaluations, and cohort reports are written outside the
provider turn.

The preview configuration schema and workspace marker are
`mforge.experiment.v3-preview.v2` and `native-v3-preview.v2`. Older preview
workspaces are rejected rather than migrated. Native v2 remains the default
when `protocol` is omitted.
