# Stage 4R delta diagnostic

Issue #12 is a retained-evidence diagnostic.  It does not rerun Stage 4R,
generate a policy, call the App Server, evaluate a graph, alter a manifest, or
change the historical decision.  The command is:

```console
uv run mforge stage4r diagnose-deltas \
  --run runs/stage4r-search/issue-11 --json
```

The command reads only the eight primary and eight replay gzip shards under
`runs/stage4r-search/issue-11/evaluations/final-validation`.  It verifies both
summary hashes, every shard hash, the frozen validation manifest, and all
timing-stripped row identities before writing compact artifacts under
`<run>/diagnostics/`.  The complete paired table is in
`paired-deltas.csv` and `paired-deltas.jsonl`; its hashes are recorded in
`paired-deltas.sha256.json`.  The other artifacts are
`bootstrap-support.json`, `cluster-summary.json`, `power-study.json`, and
`diagnostic-summary.json`.  The generated set is below 16 MiB.

## Frozen inputs and pairing

The diagnostic used the final-validation manifest file SHA-256
`87f5b6298e4c312feac2d9c4f6bafea63b70a3b29c0104a0aef33d4b91dcc91e`, whose
content hash is `1d5f1b2bd4e7978337b9351fd050b0ea0069f4b30bed8cc830247724c42a777b`.
There are 128 episodes: orders 10 and 12, graph seeds 451–454, policy seeds
4501–4516, horizon 32.  The primary and replay passes each contain 128 rows;
all episode IDs, canonical episode hashes, metric inputs, and recursive
timing-stripped projections match exactly.  Both passes retain zero model,
App Server, and oracle calls.

The paired comparison is the frozen Stage 4R champion
(`program-d5ad1c8203e0d9f25f03aabd`, source SHA-256
`e444562c1b308e3b23cb732be5f769ea1923ac1809501cea8571318c4aff0a7b`) against
`stage3-candidate-slot-04` (source SHA-256
`a5f540459695bbf7d454eeccbb8e48158d6130df6a769b67d1447de18276dc01`).

The observed champion-minus-Stage-3 AUC delta has median `1/192`
(`0.005208333333333333`) and mean `0.031290690104166664`.  Episode sign mass
is 32 negative (25.000%), 30 zero (23.4375%), and 66 positive (51.5625%).
Order medians are 0 for order 10 and `1/64` for order 12.  The eight
order/graph cluster medians are non-negative: three are zero and five are
positive.  Within-cluster variation is materially larger than between-cluster
variation in both orders; the full values and first-difference step
distributions are retained in `cluster-summary.json`.

## AUC quantization

The exact rational support is computed from the persisted 32-point curves,
not from rounded display values.  The observed AUC support has lattice step
`1/960`; the normalized curve values include fractions such as `1/6`, `1/5`,
`1/3`, `1/2`, `2/3`, and `5/6`.  The paired-delta support and frequencies are
listed exactly in `bootstrap-support.json` and in the paired table.  The
`0.03125` value is `1/32`: over a 32-step AUC, one full normalized witness unit
of cumulative area separates the trajectories.  The retained exemplar is
`o12-g0452-p4503`; its curve differences occur at steps 5–7 and sum to exact
area 1 before division by the horizon.

## Frozen hierarchical bootstrap

The Stage 4 implementation is reproduced exactly: 10,000 draws with seed
`2026073004`, graph seeds sampled with replacement first, then policy seeds
sampled with replacement within each selected graph, preserving the paired
champion/Stage-3 delta.  Percentiles use the existing linear interpolation
rule.  The pooled result is:

| quantity | value |
| --- | ---: |
| observed median | `1/192` |
| 95% interval | `[0.0, 0.03125]` |
| lower percentile indices | 249 and 250, both 0 |
| upper percentile indices | 9749 and 9750, both `1/32` |
| bootstrap sign mass | 0 negative / 3,626 zero / 6,374 positive |

The order-10 interval is `[0.0, 0.0375]`; order 12 is `[0.0,
0.0546875]`.  The zero lower bound is caused by the median statistic and
graph-cluster resampling: observed negative episode deltas are diluted by
within-graph policy replacement and a large point mass at zero.  The positive
point estimate therefore does not establish a strictly positive lower bound.

Sensitivity reductions are diagnostic only.  Episode-mean resampling gives a
95% interval of approximately `[0.01561, 0.04731]`; episode-median resampling
gives `[0.0, 0.0229167]`; cluster-median resampling gives `[0.0,
0.0286458]`.  Paired sign randomization is reported with its two-sided
null-calibrated value, while zero-valued deltas remain a separate outcome.
These estimands answer different questions and do not replace the frozen
gate.

## Stage 2C comparison

Stage 2C diagnosed `BENCHMARK_SATURATION`: order-8 random and structural
medians were both `0.5` with bootstrap interval `[0.0, 0.03125]`, despite
93.75% policy disagreement and measurable pool headroom.  Stage 4R separates
the point medians, but retains the same lattice scale, exact ties, and a
zero-bounded cluster bootstrap.  The evidence is therefore consistent with
the earlier saturation warning rather than a confirmed generalization claim.

## Deterministic power study

`power-study.json` contains a deterministic retained-distribution simulation
for 4, 8, 12, 16, 24, and 32 graph seeds per order crossed with 8, 16, and 32
policy seeds per graph.  It uses 256 simulations per cell and seed
`2026080101`.  The observed-effect scenario resamples the retained deltas;
the conservative scenario halves them; the zero-effect calibration randomizes
their signs while preserving magnitudes.  A positive pooled median is the
predeclared directional detection rule, and each cell reports its Monte Carlo
standard error and interval.  This is a sensitivity study, not a new gate or
new graph outcome.

Independent graph seeds are the priority.  Four observed graph cells cannot
reveal unseen graph heterogeneity; increasing policy seeds mainly reduces
within-graph noise and cannot substitute for new graphs.

## Recommendation

**`REDESIGN_PRIMARY_METRIC_BEFORE_CONFIRMATION`**

Before any confirmatory claim, preregister a transition-aware paired area
estimand alongside (not instead of silently replacing) normalized AUC; freeze
the current champion and Stage-3 comparator, use orders 10, 12, and 16 with
at least 16 unseen graph seeds per order and 32 policy seeds per graph, and
retain horizon 32.  Use 10,000 graph-cluster bootstrap draws with the graph-
then-policy resampling order, a positive 95% lower bound and 2% relative
improvement threshold, exact timing-stripped primary/replay equality, equal
128-evaluation CPU budgets, artifacts below 16 MiB, and zero new model calls.

The historical Stage 4R result remains `NO_GO`; this diagnostic does not
authorize Stage 5 or any HEG integration.
