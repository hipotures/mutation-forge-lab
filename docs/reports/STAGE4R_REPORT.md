# Stage 4R report

## Decision

**`NO_GO`**

The complete frozen Stage 4R comparison ran successfully. The champion passed
seven of eight scientific gates, but the 95% hierarchical paired-bootstrap
interval for pooled AUC improvement over the Stage 3 champion was
`[0.0, 0.03125]`. Its lower bound was not greater than zero.

This result does not authorize Stage 5. Issue #11 remains open for review.

## Canary

Canary attempt 1 stopped before an accepted turn. The no-inference model
catalog probe was writing into the retained Stage 4 campaign, whose 36.0 MB of
artifacts already exceeded the transport logger's 32 MiB default aggregate
limit. The smallest fix scoped doctor artifacts to the new Stage 4R run root;
the historical campaign remained unchanged.

Canary attempt 2 then passed all eight required checks with:

- concurrency `1`;
- model `gpt-5.6-luna`, reasoning effort `high`;
- the private `--auth-json ~/.codex/auth.json` capsule;
- one accepted, terminal completed turn;
- non-empty structured content and exact final server usage;
- successful parsing, Stage 2A validation, and bounded execution probe;
- a readable and exactly reindexed diagnostic archive.

The canary used no repair or automatic retry. Its candidate is diagnostic only
and was excluded from the scientific archive, selection, and validation.

## Recovery generation

The search freeze is `stage4r-search-frozen-v1` at commit `7259d92`, with
freeze SHA-256
`f0d99624410425e3107bd95a60a58f995bc7093f21074f74c29605ffbcd821bd`.

The single recovery generation produced:

| Measure | Result |
| --- | ---: |
| Ordered initial turns | 8 |
| Concurrency | 8 |
| Contract-only repairs | 2 |
| Candidate slots | 7 |
| Failed slots | 1 |
| Duplicate candidates | 0 |
| New unique valid offspring | 7 |
| Retained eligible offspring reused | 19 |

No replacement or quality-based calls were made. Model calls were closed after
the batch. Only the seven new candidates received search-training primary and
replay evaluation; retained Stage 4 metrics were reused. The new-candidate
primary and replay reductions matched exactly.

## Champion

The deterministic four-key search-training ordering selected:

- program: `program-d5ad1c8203e0d9f25f03aabd`;
- origin: Stage 4R recovery generation 5, slot 00;
- source SHA-256:
  `e444562c1b308e3b23cb732be5f769ea1923ac1809501cea8571318c4aff0a7b`;
- normalized AST SHA-256:
  `2243214df58c805e9a9343dc31ed082279e1c2ac31b21243bf889dbc9a19e165`;
- pooled search-training median AUC: `0.9479166666666666`;
- order-10 search-training median AUC: `0.95625`;
- median best-total-witness count: `0.0`.

The validation freeze is `stage4r-validation-frozen-v1` at commit `08f27cd`,
with freeze SHA-256
`1f5bce386600f65b1c5af159fc122d6ba901d9eeb3ec4267d7a3a8e68d5fd4e1`.
The final-validation manifest was verified before use at SHA-256
`87f5b6298e4c312feac2d9c4f6bafea63b70a3b29c0104a0aef33d4b91dcc91e`.

## Final validation

Exactly one primary and one replay pass evaluated the Stage 4R champion,
Stage 3 `candidate-slot-04`, random, and structural policies across the frozen
128-episode manifest with eight workers and eight shards.

| Metric | Result |
| --- | ---: |
| Stage 4R champion pooled median AUC | `0.953125` |
| Stage 3 champion pooled median AUC | `0.921875` |
| Relative pooled improvement | `3.3898305085%` |
| 95% hierarchical paired-bootstrap interval | `[0.0, 0.03125]` |
| Order-10 median paired delta | `0.02187500000000009` |
| Order-12 median paired delta | `0.03645833333333337` |
| Nonnegative graph seeds, order 10 | `4 / 4` |
| Nonnegative graph seeds, order 12 | `4 / 4` |
| Structural baseline retention | `1.023489932885906` |

Primary and replay matched exactly after timing removal. Graph validity was
100%, worker failures were zero, evaluation budgets were equal and
selected-plan-only, and oracle, model, and App Server calls were zero.

| Scientific gate | Result |
| --- | --- |
| Distinct Stage 4 offspring | Pass |
| At least 2% relative pooled improvement | Pass |
| Bootstrap lower bound greater than zero | **Fail** |
| Nonnegative order-10 and order-12 deltas | Pass |
| At least three nonnegative graph seeds per order | Pass |
| At least 99% structural retention | Pass |
| Exact primary/replay identity | Pass |
| Healthy provider-free evaluation | Pass |

The valid terminal decision is therefore **`NO_GO`**.

## Provenance and verification

- Required starting commit:
  `d90c83bd5a231af34ffab35b46a01073170f5de0`.
- Retained Stage 4 archive SHA-256:
  `01d9e73e598d2cad952e507654688bb71e2715671c2f63a4e812b7708b3754c6`.
- HEG remained read-only at
  `fd97451b0f3d87400d1d955a2c6b1b18303344ff`.
- Stage 5 was not started.

Final engineering verification is recorded in the terminal commit.
