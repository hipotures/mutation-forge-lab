# Stage 4 report

## Decision

**`INCONCLUSIVE_INFRASTRUCTURE_FAILURE`**

Issue #10 completed the frozen Stage 4 archived evolutionary-search campaign,
but the result is not eligible for a scientific GO/NO-GO decision. The
terminal reason is `incomplete_turn_or_usage_accounting`: eight accepted
generation-0 turns ended in a retained schema-protocol failure before they
produced content or exact usage. Those turns are terminal and were never
replaced. Validation was therefore not eligible, no champion was advanced,
and Stage 5 was not started.

The issue remains open for review. This report does not authorize held-out
evaluation, HEG integration, or a merge to `main`.

## Scope and provenance

- Dedicated branch: `agent/stage4-issue-10`.
- Frozen starting point:
  `49b4a611a91f87bde0b7b7be2f97c5deec8f1e89`.
- Final recovery implementation commit:
  `261efa8cf892c715a29e64d326c577c114224c5a`.
- HEG remained read-only and clean at
  `fd97451b0f3d87400d1d955a2c6b1b18303344ff`.
- Campaign root:
  `runs/stage4-search/campaign-63eea8c2ddb9`.
- Frozen scientific configuration SHA-256:
  `63eea8c2ddb9318e84a161a909d18200cb9898b1e6f0e22ad4052ff71c37e179`.
- Search-freeze SHA-256:
  `5c9f6b2a1ae6d5b9d58fa6b86d93a58bbae975c3ef0b37143eb6004e3341d99f`.
- Generation model and effort: `gpt-5.6-luna`, `high`.
- Generation concurrency and deterministic evaluation workers: eight each,
  with eight physical cores reserved.

The retained tags are:

- `stage4-search-frozen-v1` at
  `795440a53361221341179c0a238646fca21b9790`;
- `stage4-search-amendment-v1` at
  `5d0ca241ad0d038438e85ef1c58b5f57b30af01`;
- `stage4-search-amendment-v2` at
  `c2cfaa1dc522c37d167b109a4c6679a60e31b211`;
- `stage4-search-amendment-v3` at
  `fdcc96e203c83ceea33d46b4c12e9411ed646163`;
- `stage4-search-amendment-v4` at
  `e73b5701bb5706a0a7fa29b85573a9021ef551cd`;
- `stage4-search-amendment-v5` at
  `afb359356fb75ab3472a613129b3797a30477d93`;
- `stage4-search-amendment-v6` at
  `499fcbc22c97988e3403de75a0c9d7d8c8c3b52b`;
- `stage4-search-amendment-v7` at
  `261efa8cf892c715a29e64d326c577c114224c5a`.

Every amendment was technical. The frozen candidates, seed manifest,
development manifest, briefs, prompts, model, effort, concurrency, repair
limit, evaluation budget, bootstrap, thresholds, and validation gate were not
changed.

## Authentication and protocol boundary

The official commands used the existing
`--auth-json ~/.codex/auth.json` option. The path was passed into a private
Codex capsule; credential contents were never read, printed, copied, hashed,
or retained by this work. The final authenticated preflight was `READY`.

The generation request identity remained pinned to the authenticated v3
doctor SHA-256
`99af1ae13933fdda570fdb4fdff78f83936323a986c9388c75b8bb73b9c1fa17`.
The raw frozen output schema remained unchanged. Amendment v4 introduced only
the App Server transport projection required by the installed protocol and
retained all failed protocol evidence.

The observed final resume command and exit status are retained in
`official-resume-v7-execution-evidence.json`, SHA-256
`6fca4462a78a5a09cfad203e4f43ea1735b120975aecb291ef39581790d04928`.
That record is explicit that the raw command stdout was not retained. The
terminal artifact hashes are authoritative, and the resume was not replayed
merely to capture a transcript.

## Campaign accounting

The terminal aggregate is:

| Measure | Result |
| --- | ---: |
| Initial turn identities | 32 |
| Initial turns recovered offline | 24 |
| Replacement initial turns | 0 |
| Repair turns | 6 |
| Accepted live turns | 38 |
| Frozen accepted-turn budget | 64 |
| New unique valid offspring | 19 |
| Archive records | 40 |
| Archive unique records, including seeds | 27 |
| Archive tombstones | 13 |
| Checkpoint sequence | 1–107, contiguous |

All six repair turns were accepted and contentful. They were restricted to
the six preverified schema/AST-repairable positions, with one repair attempt
per position. No accepted initial turn was resubmitted.

The 24 recovered generation 1–3 initial turns have exact aggregate usage:

| Counter | Recovered initials | Repairs | Combined accounted |
| --- | ---: | ---: | ---: |
| Input tokens | 178,773 | 27,069 | 205,842 |
| Cached input tokens | 14,336 | 7,424 | 21,760 |
| Cache-write input tokens | 0 | 0 | 0 |
| Output tokens | 79,079 | 16,353 | 95,432 |
| Reasoning output tokens | 61,078 | 11,911 | 72,989 |
| Total tokens | 257,852 | 43,422 | 301,274 |

These exact counters do not make whole-campaign usage exact: the eight
terminal generation-0 protocol failures still have no usage payload.

## Recovery and archive integrity

The initial unauthenticated attempt was caused by invoking the already
supported command without its `--auth-json` argument. Thirty-two proven
pre-turn, content-free, usage-free tombstones were retained before the same
frozen request identities were resumed with authentication.

The authenticated generation-0 turns were accepted, but the installed App
Server rejected the transport form of the frozen schema. Those eight turns
were retained as terminal protocol failures and were not retried.

The later generation 1–3 host errors were different: all 24 turns had
completed remotely in 28.619–102.787 seconds before the old 120-second host
deadline. Strict parsing of the retained RPC/events/profile artifacts
reconstructed one final response and one exact usage payload per turn. The
recovery performed zero initial provider calls for those turns.

One deterministic callback then failed because the process-local JSON-RPC
request number `10` had been projected as a globally unique archive request
identity. Amendment v7 retains that local number only as provider metadata
and uses the frozen generation idempotency key as the archive identity. The
eight partial records and seven source files were retained before the
uncommitted callback projection was reconciled. No completed model turn was
replayed.

The final archive reindexes exactly:

- archive SHA-256:
  `01d9e73e598d2cad952e507654688bb71e2715671c2f63a4e812b7708b3754c6`;
- 40 records, 32 globally unique non-seed request identities;
- no duplicate program, slot, request, or source identity;
- no missing parent, request, or source;
- no corrupt file or source-hash mismatch.

Generation 2–4 primary and replay evaluations each retain eight shards of
sixteen records. Their replay summaries are exact and record zero provider
calls.

## Evidence

The terminal `search-summary.json` SHA-256 is
`8931c72ce6f19fe07550932df1b3e1f2b12e89b173ff6a85551d901e13e37067`.
The final campaign evidence manifest contains 1,651 verified entries:

- evidence-manifest file SHA-256:
  `33fea2d59de22b8c88ef60638b243d3e1415d6ba8bb487f99b9660f7680b6e79`;
- internal manifest SHA-256:
  `8efabc3399e7c028dd5efe324809e53fe98ad3503f9695bc41b6f9397fffe832`.

Recovery evidence includes:

- pre-auth recovery internal SHA-256:
  `93dfb1062dc14657c6839aeaa8e378c42d2e8e4f62ebb83545be055ff7becfea`;
- completed-turn recovery internal SHA-256:
  `d36a5f9bc460cefa622acb8d3cf9642df1f1dd1116a07230d0afca3c7dcfff76`;
- partial-callback recovery internal SHA-256:
  `ec93b2965d0a44ae38ea027e34e7e4a13d64b8fd85428617b2ebaf6dbbad54d5`;
- v7 technical evidence SHA-256:
  `56ac66d5f5836f2dcaf3ac617ba48bcf18d3fd8c654039a7ec0bd88813138b08`.

Raw prompts, private responses, generated policy source, credentials, and
transcripts are not reproduced in this report. They remain only in their
bounded retained artifacts where applicable.

## Validation boundary

The terminal decision is infrastructure-inconclusive, not `NO_GO`,
`PENDING_VALIDATION`, or `GO`. The validation freeze was not created,
held-out data was not used, Stage 5 was not started, and HEG was not modified.

The safe next action is review of issue #10 and the retained evidence. Any
attempt to repair the missing generation-0 usage would require new scientific
authority because those accepted turns are terminal and cannot be replaced.

## Final engineering verification

The final tracked branch was checked with:

- full `pytest`: 380 passed in 15.09 seconds;
- Ruff: passed;
- strict mypy: passed for 79 source files;
- `git diff --check`: passed;
- terminal archive inspect/reindex: passed;
- evidence-manifest verification: passed;
- primary/replay shard assertions: passed;
- HEG pin and clean state: passed.
