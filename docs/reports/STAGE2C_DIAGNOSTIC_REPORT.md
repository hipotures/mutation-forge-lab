# Stage 2C diagnostic report

Date: 2026-07-29

Status: **implemented and validated; diagnostic only**

Primary diagnosis: **`BENCHMARK_SATURATION`**

Next-step decision: **`DESIGN_STAGE_2D_PREREGISTRATION`**

Neither the diagnosis nor the decision clears the Stage 2B gate. Stage 2B
remains `NO_GO`. Stage 3, model use, evolution, a full proposer, and HEG policy
integration remain blocked.

## Entry point and boundaries

Stage 2C started from:

- Mutation Forge `5b949f84dca77474d242665152300521fbe8dd31`;
- HEG `fd97451b0f3d87400d1d955a2c6b1b18303344ff`;
- the retained Stage 2B control in
  `runs/stage2b-20260729T144651.225406Z-23bdc38d7ce5`.

Implementation and final diagnostic execution used branch
`agent/stage2c-issue-7`. The final diagnostic implementation commit before
this report was `7d2124f7aa0cc8a2dc9e4d416c50712a817f4032`.

HEG remained pinned, read-only, and clean. No Mutation Forge runtime, ranker
worker, policy, or diagnostic experiment used a network. No experimental data
was downloaded. There were zero model calls and zero Codex App Server calls.
Stage 3, evolution, archive search, a generated proposer, and HEG integration
were not started.

The Stage 2B report, configuration, dataset, seeds, threshold, result, and
scientific interpretation were not modified. In particular, Stage 2C does not
replace or rerun the failed efficacy gate under a different threshold.

## Exact Stage 2B control

The documented command

```console
uv run mforge diagnostics stage2c-control \
  --config configs/stage2b-preregistered.toml --json
```

read the immutable durable Stage 2B result and reproduced all published
identities and metrics:

- Stage 2B config SHA-256
  `23bdc38d7ce5bef3694ff8ff49d0ce772ed495ec8afd2cfc464f6f58d2db6186`;
- Stage 2B behavior SHA-256
  `99ba29240a67a05455c26af54d5d0238f72fa5241465b6f2d2aefdb2c822f796`;
- random median normalized best-so-far AUC `0.5000`;
- structural median normalized best-so-far AUC `0.5000`;
- relative median improvement `0.0%`;
- paired bootstrap interval `[0.0, 0.03125]`;
- all 32 identical-pool proofs;
- 17 score calls per episode: one initial score plus two selected-plan scores
  for each of eight steps;
- zero historical full-pool oracle calls;
- deterministic HEG replay and Rich/JSON canonical parity.

The final control artifact is
`runs/stage2c-control-20260729T162549.384412Z-9a52333fa3dd`.
Its canonical result SHA-256 is
`eda371229c48ecaa35a2308cc50f40eb34caa4aab47fb8c50f91e397db3abcee`.

Any mismatch would have stopped Stage 2C before oracle or matrix execution.

## Diagnostic implementation and isolation

Stage 2C adds three separate commands:

```console
uv run mforge diagnostics stage2c-control \
  --config configs/stage2b-preregistered.toml --json
uv run mforge diagnostics pool-oracle \
  --config configs/stage2c-diagnostic.toml --json
uv run mforge diagnostics stage2c-matrix \
  --config configs/stage2c-diagnostic.toml --json
```

Normal Stage 1/2A/2B commands have no oracle option and retain selected-only
scoring. Oracle mode exists only in the diagnostic command path and defaults
off in the diagnostic cell API. For each pool it runs after both rankings and
selected-plan scores are fixed. It receives the already-generated immutable
pool and source graph, never exposes a score to a ranker, and cannot change
proposal order, policy inputs, RNG consumption, selections, controller state,
or the historical gate.

The implementation records, for every bounded pool:

- complete random, structural, and oracle orderings;
- priorities, selected IDs, same-selection status, ties, distinct priorities,
  rank correlation, and top-1/3/5 overlap;
- selected `k`, operator, selector, score delta, acceptance/rejection,
  duplicate status, and diagnostic stagnation;
- oracle headroom, improving count/fraction, best delta, hit rates, and regret;
- separately timed selected scoring and diagnostic oracle scoring.

Every episode records raw and normalized best-so-far curves, AUC, first policy
selection divergence, and a Stage 2B-compatible trajectory hash.

Numeric and categorical feature diagnostics are deterministic and bounded.
They include missing/constant/near-constant rates, capped distinct counts,
ranges, quantiles, distribution by `k` and selector, association with oracle
delta and improvement, association with structural priority, and ascending
versus descending univariate polarity. Samples and distinct values are capped
at 4,096 per feature.

Records are canonical JSON Lines in deterministic gzip shards. Bounds are:

- 65,536 bytes per record;
- 1,048,576 uncompressed bytes per shard;
- 50,000 records;
- 268,435,456 uncompressed record bytes;
- 4,194,304 bytes per ordinary JSON artifact.

Raw record hashes retain timing telemetry. Replay identities strip timing and
run paths. Every record is labelled with order, graph seed, policy seed,
horizon, and step where applicable.

## Original order-8 diagnostic

The final original-cell oracle artifact is
`runs/stage2c-pool-oracle-20260729T162552.582080Z-9a52333fa3dd`.
It contains 256 pools, 32 episodes, and 3,072 proposals.

All 32 Stage 2B trajectories reproduced exactly. There were:

- 512 ordinary selected-plan score calls;
- 3,072 separately accounted diagnostic oracle score calls;
- zero exact-verifier calls;
- zero invalid host-applied graphs;
- zero worker failures, timeouts, crashes, or protocol errors.

The control score was 4, so normalization changed in increments of `0.25`.
One unit of improvement at one of eight steps changed AUC by `0.03125`.
All episodes had an effective denominator, but all were classified as
effectively quantized. Observed candidate score levels were only
`[2, 3, 4, 6]`.

Random produced six normalized AUC levels:
`[0.25, 0.375, 0.40625, 0.4375, 0.46875, 0.5]`. Structural produced five:
`[0.375, 0.40625, 0.4375, 0.46875, 0.5]`. Every episode improved eventually.
Random first improved at step 1 in 68.75% of episodes and later in 31.25%;
structural first improved at step 1 in 75% and later in 25%. Neither policy
had floor or ceiling AUC episodes, but both medians landed on the same
upper observed plateau of `0.5`.

### Rank behavior

The policies did not behave alike:

| Diagnostic | Result |
| --- | ---: |
| same selection | 6.25% |
| policy disagreement | 93.75% |
| disagreement conditional on headroom | 93.75% |
| top-3 overlap | 23.83% |
| random tie frequency | 7.81% |
| structural tie frequency | 14.45% |
| median rank correlation | -0.0070 |

The mean number of distinct priorities in a 12-proposal pool was 10.77 for
random and 9.28 for structural. Thus ties and common selections cannot explain
the equal Stage 2B medians.

Selection divergence occurred immediately in 30 of the 32 control episodes
and at step 1 in the other two. Despite selection disagreement, selected score
totals were equal in 58.20% of the 256 pool steps, demonstrating coarse score
equivalence rather than common ranking.

### Pool headroom and selected regret

Every control pool contained an improving proposal. The mean best immediate
delta was 2.0 and 69.04% of all proposals improved on the source score.

| Oracle diagnostic | Random | Structural |
| --- | ---: | ---: |
| mean selected regret | 1.0195 | 0.2578 |
| oracle-best-tie hit rate | 60.16% | 86.33% |
| improvement after disagreement | 67.92% | 87.92% |

The structural policy therefore used meaningful immediate-move information
and materially outperformed random at selecting from the same pools, even
though the preregistered median AUC did not distinguish them.

Headroom also existed across operator strata. `k=2` and `k=3` strata had
headroom in every pool where present; `k=4` had headroom in 82.33%. The
sampled-forbidden-cycle selector was the weakest stratum, with 53.91%
headroom and 21.79% improving proposals, while several high-load, remote, and
distant strata had improving proposals in every observed pool.

### Feature evidence

The frozen features were not signal-free:

- `local_c4_risk` correlated `-0.8008` with immediate delta; ascending
  (lowest-risk first) achieved zero mean regret and a 100% oracle-best-tie hit
  rate;
- `local_triangle_risk` correlated `+0.7812` with immediate delta; descending
  achieved zero mean regret and a 100% hit rate;
- length-4 broken witnesses correlated `+0.5922` with delta, and descending
  ranking reduced mean regret to `0.4492`;
- length-7 broken witnesses correlated `-0.5134`; ascending achieved zero
  regret, opposite to the reviewed ranker's additive polarity.

There were secondary feature/ranker defects. Length-9 witness and load fields,
length-6 maximum load, and minimum pre-existing distance were constant in the
control. Removed-edge distance fields were at least 96% near-constant. The
reviewed policy gave these distances and triangle risk multipliers from
`1e-7` to `1e-10`, making them ineffective beside integer witness and C4
terms. It also used the wrong observed polarity for length-7 broken witnesses
and local triangle risk.

These are retained diagnostic findings, not a tuned replacement ranker. They
do not support `FEATURE_SIGNAL_INSUFFICIENT`, because strong signal existed,
and they do not make `STRUCTURAL_RANKER_INEFFECTIVE` primary, because the
unchanged structural policy already had substantially lower regret and higher
oracle hit/improvement rates than random.

## Frozen exploratory matrix

`configs/stage2c-diagnostic.toml` was committed before execution and froze:

- even orders 8, 10, and 12;
- graph seeds 101, 102, 103, and 104;
- policy seeds 1 through 32;
- horizons 8, 16, and 32;
- the exact Stage 2B pool, selector, `k`, feature, worker, scoring, and witness
  budgets.

There were no exclusions. The matrix was explicitly exploratory and did not
change the historical gate.

The final primary matrix artifact is
`runs/stage2c-matrix-20260729T162558.073507Z-9a52333fa3dd`.

Across all 36 cells:

- 1,152 episodes and 21,504 immutable pools completed;
- 258,048 proposals were oracle-scored after selection;
- ordinary selected-only accounting made 43,008 score calls;
- all graphs were valid;
- all workers remained usable with zero failures;
- all 36 representative replay proofs passed;
- all 32 original Stage 2B control trajectories matched.

Aggregate evidence by order and horizon follows. AUC counts are distinct
episode outcomes across four graph seeds and 32 policy seeds per cell group.

| Order | Steps | Random median AUC | Structural median AUC | Distinct R/S | Headroom | Disagreement | Mean regret R/S |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 0.4531 | 0.5000 | 14 / 5 | 100% | 94.73% | 1.356 / 0.176 |
| 8 | 16 | 0.4766 | 0.5000 | 14 / 5 | 100% | 94.38% | 1.327 / 0.169 |
| 8 | 32 | 0.4883 | 0.5000 | 14 / 5 | 100% | 94.12% | 1.333 / 0.173 |
| 10 | 8 | 0.5000 | 0.6000 | 28 / 26 | 100% | 93.07% | 1.767 / 0.798 |
| 10 | 16 | 0.5500 | 0.6688 | 42 / 39 | 100% | 93.41% | 1.771 / 0.822 |
| 10 | 32 | 0.5906 | 0.8063 | 66 / 54 | 100% | 93.43% | 1.781 / 0.818 |
| 12 | 8 | 0.5000 | 0.5833 | 34 / 19 | 100% | 92.97% | 2.182 / 1.049 |
| 12 | 16 | 0.5729 | 0.6302 | 44 / 29 | 100% | 92.58% | 2.244 / 1.040 |
| 12 | 32 | 0.6302 | 0.6536 | 65 / 44 | 100% | 92.48% | 2.264 / 1.048 |

Higher order increased initial scores, attainable levels, and distinct AUC
outcomes. The unchanged policies separated exploratorily at orders 10 and 12.
Longer horizons widened outcome support, but at order 8 they moved random
toward the same `0.5` plateau rather than creating durable discrimination.

Policy selections diverged at step 0 in 95.31% of order-8 episodes, 93.75% of
order-10 episodes, and 86.72% of order-12 episodes. Every episode diverged by
step 2. Policy differences therefore were not suppressed until late in the
horizon.

The matrix retained strong feature signal. Across cells the largest absolute
oracle-delta association was `local_c4_risk` (approximately `-0.757`), followed
by length-4 broken witnesses (`+0.532`) and local triangle risk (`+0.496`).
Structural priority was instead most associated with several high-length load
fields and `k`, confirming an imperfect combination without erasing the
observed immediate-selection advantage.

The sum of separately recorded proposal, feature, ranker, selected-scoring,
and oracle-scoring phase telemetry across order/horizon groups was 167.27
seconds. This is phase accounting, not end-to-end wall time.

## Replay, parity, and artifact bounds

The primary and full replay matrix artifacts had identical:

- matrix canonical SHA-256
  `f6320b75e36d1367b17bf802c1750315220f7cf2c6d4c8a30b9911cebd049662`;
- ordered 36 cell canonical hashes;
- timing-stripped canonical record SHA-256
  `5abf63bd77c284f419170c2e797554820830d05603da03158c13614f8cb5ad92`.

The replay artifact is
`runs/stage2c-matrix-20260729T163232.028323Z-9a52333fa3dd`.
Raw record hashes differed, as expected, because they retain timing telemetry.

The primary matrix persisted 22,656 records in 196 shards:

- 203,446,895 uncompressed bytes, below the 268,435,456-byte bound;
- 23,937,062 compressed bytes;
- largest shard 1,048,572 uncompressed bytes, below the 1,048,576-byte bound;
- zero missing labels, duplicate episode keys, duplicate pool keys, or curve
  length mismatches.

Every JSON result measured Rich/JSON parity by round-tripping the shared
canonical payload. The primary matrix parity SHA-256 was
`e2e071cb878e422bee467cb3ac60926b34bdce971ff96902ae07e18c9c59e397`.

An initial replay attempt correctly exposed two artifact-identity defects:
matrix identity included a run-specific cell path and record identity included
timing fields. The implementation was corrected to exclude paths/timing only
from replay identities and to label every record with order and horizon. No
policy, proposal, score, metric, seed, threshold, dataset, matrix cell, or
scientific result changed. The final full replay above passed.

## Tests and validation

Tests cover:

- saturation, quantization, no-improvement, immediate-improvement, ties, and
  degenerate denominators;
- tie-aware rank correlation, same-selection, and top-k overlap;
- oracle best/hit/regret calculations and separate accounting;
- feature statistics and univariate polarity;
- oracle isolation from policy inputs, RNG, selection, controller state, and
  trajectories;
- oracle disabled by default and unchanged Stage 2B selected-only accounting;
- deterministic reduced-cell and full-matrix replay;
- bounded sharded artifacts and interrupted terminal recovery;
- relabeling invariance of diagnostic features;
- the exact durable Stage 2B control;
- all existing Stage 1, 2A, and 2B regressions.

Final commands:

```console
uv run pytest
uv run ruff check .
uv run mypy
uv run mforge doctor --heg-repo ../heg
git diff --check
```

Final validation completed with **186 passed, zero failed, and zero skipped**
in 7.69 seconds. Ruff, strict mypy (46 source files), `mforge doctor`, and
`git diff --check` passed. Doctor and direct Git checks recorded HEG at
`fd97451b0f3d87400d1d955a2c6b1b18303344ff` with a clean worktree.

## Primary diagnosis

**`BENCHMARK_SATURATION`**

The original order-8 benchmark and metric could not discriminate the policies
despite measurable pool headroom:

1. every pool contained an improving alternative;
2. policies disagreed in 93.75% of control pools and had near-zero median rank
   correlation;
3. the unchanged structural policy had much lower selected regret, a higher
   oracle-best hit rate, and a higher improvement rate after disagreement;
4. frozen features contained strong immediate-delta signal;
5. the order-8 denominator was only 4, score levels were sparse, AUC moved in
   coarse increments, and best-so-far over repeated pools from the same static
   source state drove both medians to the same `0.5` plateau;
6. increasing the order produced many more AUC outcomes and exploratory policy
   separation, while increasing only the order-8 horizon intensified the
   plateau.

`NO_PROPOSAL_HEADROOM`, `FEATURE_SIGNAL_INSUFFICIENT`, and common
selection/ties are contradicted directly. `STRUCTURAL_RANKER_INEFFECTIVE` is a
secondary finding because polarity/scale defects exist, but it cannot explain
the primary failure when structural already selected much better immediate
moves than random. `CONTROLLER_HORIZON_SUPPRESSION` is not primary because
selection divergence appeared at step 0 or 1 and longer order-8 horizons
increased saturation.

## Next-step decision

**`DESIGN_STAGE_2D_PREREGISTRATION`**

The evidence justifies designing, but not executing, a separate confirmatory
Stage 2D issue. A proposed preregistration should:

- freeze a more discriminating non-held-out toy benchmark before execution,
  with orders 10 and/or 12 supported by the exploratory evidence;
- retain the current random and structural policy sources unchanged, avoiding
  post-diagnostic ranker tuning as confirmatory evidence;
- choose graph seeds, horizon, state-transition/controller semantics, primary
  metric, threshold, bootstrap procedure, exclusions, and compute bounds
  before results are observed;
- keep any full-pool oracle diagnostic-only and outside the confirmatory
  selection/controller path;
- preserve equal pools, selected-only ordinary scoring, HEG isolation, and all
  Stage 2A worker limits;
- require separate approval before any Stage 2D execution.

This report does not specify an approved Stage 2D experiment and does not
authorize one. It does not clear Stage 2B, authorize Stage 3, permit a model
call, or permit evolution, a full proposer, or HEG policy integration.
