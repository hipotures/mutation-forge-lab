# Stage 2D confirmatory report

## Decision

**`GO_TO_STAGE_3`**

All eleven preregistered gates passed, including exact primary/replay identity.
This decision closes only the Stage 2D evidence gate. Stage 3 was not started
and still requires separate user approval and integration of this branch.

Stage 2B remains the retained historical `NO_GO`. Stage 2D is a separately
preregistered comparison on independent real trajectories; it does not alter
the Stage 2B benchmark, threshold, result, report, or interpretation.

## Dependency and frozen provenance

- Stage 2C issue #7 was closed with state reason `COMPLETED`.
- Stage 2C commit
  `06ae4e7e5d958bb47ba0acba2e686afd7023b51b` was integrated as the exact
  `main` entry point before work began.
- Dedicated branch: `agent/stage2d-issue-8`.
- Immutable preregistration commit:
  `03a206561c4eeaace8f4c2c09646c54578166c12`.
- Annotated and pushed tag: `stage2d-preregistered-v1`.
- Preregistration issue comment: issue #8 comment `5121730469`.
- Config SHA-256:
  `c42f9eb5e438f88409f74a8e70767c774a33a6cfb7c05da8a5f64e68d123f4b4`.
- Manifest SHA-256:
  `9ebfb3e575065c3436aa04c4e6ef6f3d299f3583a8ff54a83fa9f0d003302e8f`.
- HEG remained read-only, clean, and pinned to
  `fd97451b0f3d87400d1d955a2c6b1b18303344ff`.

The tag, commit, config, schemas, ordered manifest, seeds, metrics, bootstrap,
thresholds, gates, rankers, exclusions, and analysis were pushed and recorded
before any confirmatory shard ran. No frozen input changed afterward.

Frozen ranker identities:

| Ranker | Source SHA-256 | Normalized AST SHA-256 |
|---|---|---|
| Random | `d4994fb96bdc3c23b8b24d9bca041f2822bc30329bcf8f9cdbd2e277e65b0612` | `f7f502b0319df5dc32ef0f8476024c4986dcb3422ef2e03b117a3d394bbfc7b7` |
| Structural | `68aba299d7735198d38a8d30e221ef99cdbb7d846c502aca41691c49ceef87be` | `5b017c2ba79953e31b224df91e060d4af27c3b212695a03e8650ec91e8b0ad81` |

## Preregistered design

The immutable manifest contains 512 paired episodes:

- toy connected-cubic orders 10 and 12;
- graph seeds 201–208 at each order;
- policy seeds 1001–1032;
- horizon 32;
- witness cap 64;
- pool size 12;
- legal `k` values 2, 3, and 4;
- unchanged Stage 2B selectors, weights, retry/matching bounds, feature
  budgets, scientific schemas, score semantics, and Stage 2A sandbox limits.

The canonical manifest partitions the episodes exactly once into eight shards
of 64. Both policies of an episode remain in the same shard.

Each policy owns its current immutable graph and current authoritative score.
At every step the host generates a bounded legal pool from that policy's
current graph, the unchanged Stage 2A worker ranks it, and only the selected
plan is applied, host-validated, and scored. A plan is accepted only when its
complete `GraphScore.ordering_key` strictly improves. An accepted mutation
advances only that policy's graph. The same outcome-independent seed is
derived from episode identity and step for both policies. A pool is shared
only while both policy graphs are identical; after divergence, pools are
generated independently from the respective current graphs.

The Stage 2B static-source repeated-pool behavior is not used. There is no
full-pool oracle or full-score-per-proposal path.

The normalized trajectory quality at each step is

```text
q = (initial_total_witnesses - best_total_witnesses) /
    max(1, initial_total_witnesses)
```

and episode AUC is the mean of the 32 best-so-far `q` values. The primary
effect is structural AUC minus random AUC on order 10. The hierarchical paired
bootstrap uses 10,000 samples, RNG seed 2026072902, graph-seed resampling
followed by paired policy-seed resampling, and a 95% percentile interval.
Order 12 is secondary; the pooled interval is stratified by order.

## Implementation and reduced validation before freeze

Stage 2D adds:

- strict versioned TOML and JSON schemas;
- a canonical immutable episode/shard manifest;
- a real independent-trajectory runner with selected-only scoring;
- bounded gzip JSONL shard artifacts and terminal failure records;
- exact assignment, cardinality, duplicate, corruption, and missing-shard
  rejection;
- deterministic completion-order-independent reduction;
- 1-worker/8-worker bootstrap parity;
- hierarchical bootstrap and the frozen eleven-part gate;
- CPU topology, affinity, thread, run/cache/tmp isolation, provenance, and
  replay verification;
- `mforge stage2d plan`, `run-shard`, `reduce`, and `verify-replay`.

Reduced tests covered strict improvement, state advancement, shared versus
independent pools, deterministic replay identity, selected-only accounting,
ranker identity, exact manifest partitioning, reducer failure artifacts,
1-vs-8 reducer parity, bootstrap/gate boundaries, all three terminal
decisions, CPU reservation/affinity, and interrupted shard artifacts.

Before the tag was created and pushed:

- `pytest`: 193 passed, zero skipped;
- Ruff: passed;
- strict mypy: passed for 51 source files;
- `mforge doctor`: passed;
- `git diff --check`: passed;
- retained Stage 2B rankers/config/report and the Stage 2C report: no diff;
- HEG: exact pin and clean;
- confirmatory results observed: false.

## Execution isolation and resources

`lscpu` exposed 16 physical/logical CPUs. The manifest assigned CPUs 0–7,
physical IDs `0:0` through `0:7`, and reserved eight physical cores.

Exactly eight native execution subagents were created, one per immutable
shard. Each used:

- a unique clean detached checkout of the annotated tag;
- a unique shard output root;
- unique `TMPDIR`, `UV_CACHE_DIR`, and `XDG_CACHE_HOME`;
- one distinct physical CPU affinity;
- `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
  `VECLIB_MAXIMUM_THREADS=1`, and `BLIS_NUM_THREADS=1`;
- offline, frozen, no-sync `uv` execution.

The environment exposes four agent slots including the coordinator, so shards
ran in parallel waves of 3, 3, and 2. This never exceeded three CPU-heavy
shards and left at least 13 of 16 physical cores unused. The same eight agents
then ran their assigned replay shards in the same wave structure. No shard
needed the one allowed infrastructure retry.

Per-shard wall time was approximately 32.0–32.8 seconds. Each shard made
24,576 bounded calls to each persistent ranker worker with zero worker
failures. Primary and replay each persisted about 9 MiB; each compressed
episode shard was at most 1,083,321 bytes (about 1.03 MiB), far below the
128 MiB shard bound.

## Primary confirmatory result

Primary artifacts:

- `runs/stage2d-primary/shard-00` through `shard-07`;
- `runs/stage2d-primary/reduction/summary.json`;
- `runs/stage2d-primary/reduction/metrics.json`;
- `runs/stage2d-primary/replay-verification.json`.

Canonical identities:

- aggregate SHA-256:
  `d6ec5fd052a50a8f52c0045f521320d9f3ae5a340972f9098ed03d3091d967f3`;
- timing-stripped episode-stream SHA-256:
  `f135e7d28d7d3d02815e913a42f5616181a29fbc7cdd84927bf0e18ad4b9b020`;
- reduction SHA-256:
  `80753ee42e610d5564c0b780a023d8d05799dff6b5101243b777a68be2748d16`.

### Primary order 10

| Metric | Random | Structural | Structural − random |
|---|---:|---:|---:|
| Median normalized best-so-far AUC | 0.793750 | 0.950000 | 0.143750 |
| Relative median improvement | — | — | 19.685% |
| Median best total witnesses | 0 | 0 | 0 |

The hierarchical paired 95% bootstrap interval for the median AUC delta is
`[0.103125, 0.175000]`, entirely above zero. All eight graph seeds had a
positive median paired delta:

| Graph seed | Median AUC delta |
|---:|---:|
| 201 | 0.209375 |
| 202 | 0.084375 |
| 203 | 0.171875 |
| 204 | 0.181250 |
| 205 | 0.134375 |
| 206 | 0.115625 |
| 207 | 0.171875 |
| 208 | 0.125000 |

### Secondary order 12

| Metric | Random | Structural | Structural − random |
|---|---:|---:|---:|
| Median normalized best-so-far AUC | 0.786458 | 0.890625 | 0.078125 |
| Relative median improvement | — | — | 13.245% |
| Median best total witnesses | 0 | 0 | 0 |

The order-12 95% bootstrap interval is
`[0.052083, 0.104167]`. The pooled order-stratified interval is
`[0.088021, 0.131250]`; both exclude zero.

### Accounting and health

- episodes: 512 exactly once;
- initial authoritative score calls: 512;
- selected-plan score calls: 32,768, exactly the frozen expectation;
- oracle score calls: 0;
- exact-verification calls: 0;
- invalid graphs: 0;
- policy failures, timeouts, crashes, or protocol failures: 0;
- model calls: 0;
- Codex App Server calls: 0;
- experiment-runtime network calls: 0;
- HEG writes or modifications: 0.

The full score/witness distributions, weighted penalties, ordering keys,
acceptance/rejection/duplicate counts, time and evaluations to first
improvement, raw and normalized curves, divergence steps, pool hashes,
selected IDs, timing, and worker telemetry remain in the bounded shard and
reduction artifacts.

## Deterministic replay

Replay artifacts are under `runs/stage2d-replay`. The replay was not pooled
with the primary sample and did not alter any statistical estimate.

Primary and replay matched exactly on:

- config hash;
- manifest hash;
- preregistration commit;
- 512 timing-stripped episode records;
- all eight shard hashes;
- timing-stripped episode-stream hash;
- aggregate hash;
- all metrics and bootstrap intervals;
- reduction hash;
- gate result.

The primary and replay reduction SHA-256 are both
`80753ee42e610d5564c0b780a023d8d05799dff6b5101243b777a68be2748d16`.
Replay verification SHA-256 is
`e2cba4a4c5e6b72ac33bd39596aac39ca45b362c14e76ed5af7ccf0e067c20ba`.
Raw compressed file hashes differ where captured timing differs, as expected;
timing is excluded only from the preregistered replay identity.

## Gate

| Preregistered criterion | Result |
|---|---|
| Order-10 relative median improvement at least 10% | PASS — 19.685% |
| Order-10 bootstrap lower bound above zero | PASS — 0.103125 |
| Pooled stratified bootstrap lower bound above zero | PASS — 0.088021 |
| Order-12 median delta nonnegative | PASS — 0.078125 |
| Structural median witness count no worse at each order | PASS — equal at 0 |
| At least 6/8 order-10 graph seeds nonnegative | PASS — 8/8 positive |
| Graph validity 100% | PASS — 512/512 episodes, zero invalid graphs |
| Policy failure rate zero | PASS |
| Selected-plan-only scoring and no oracle | PASS |
| Primary/replay canonical identity | PASS |
| All validation and provenance checks | PASS |

## Commands

Planning and preregistration validation:

```console
uv run mforge stage2d plan \
  --config configs/stage2d-preregistered.toml --json
uv run pytest
uv run ruff check .
uv run mypy
uv run mforge doctor --heg-repo ../heg
git diff --check
```

Each detached shard checkout ran its assigned command with the frozen
single-thread environment and `taskset --cpu-list N`:

```console
uv run --offline --frozen --no-sync mforge stage2d run-shard \
  --config configs/stage2d-preregistered.toml \
  --shard shard-NN --output-dir RUN/shard-NN --json
```

Coordinator-only reduction and replay verification:

```console
uv run mforge stage2d reduce \
  --config configs/stage2d-preregistered.toml \
  --input-root runs/stage2d-primary \
  --output-dir runs/stage2d-primary/reduction --workers 8 --json
uv run mforge stage2d reduce \
  --config configs/stage2d-preregistered.toml \
  --input-root runs/stage2d-replay \
  --output-dir runs/stage2d-replay/reduction --workers 8 --json
uv run mforge stage2d verify-replay \
  --primary runs/stage2d-primary/reduction/summary.json \
  --replay runs/stage2d-replay/reduction/summary.json \
  --output runs/stage2d-primary/replay-verification.json --json
```

## Final validation

After the report and all experiment artifacts existed:

- full `pytest`: 193 passed, zero skipped;
- Ruff: passed;
- strict mypy: passed for 51 source files;
- `mforge doctor`: passed;
- `git diff --check`: passed;
- primary: 512 unique exactly-once records in 8 completed shards of 64;
- replay: 512 unique exactly-once records in 8 completed shards of 64;
- primary and replay `summary.json` and `metrics.json`: byte-identical;
- all eight detached preregistration worktrees: exact tag commit and clean;
- frozen code, config, schemas, manifest, and rankers: byte-identical to the
  annotated tag;
- retained Stage 2B and Stage 2C evidence: unchanged;
- HEG: exact pinned commit and clean;
- report values and hashes: matched the canonical artifacts;
- no model, App Server, experiment-runtime network, Stage 3, or HEG-write
  activity.

## Final interpretation

The valid preregistered benchmark supports the unchanged structural ranker over
the unchanged random ranker on this non-held-out Stage 2D toy dataset under
the frozen independent-trajectory controller and equal selected-score budget.
It is infrastructure and pre-model policy evidence, not a held-out
generalization claim and not an HEG scientific superiority claim.

The required terminal decision is:

**`GO_TO_STAGE_3`**

Stage 3, model use, evolution, full proposer work, and HEG policy integration
were not started.
