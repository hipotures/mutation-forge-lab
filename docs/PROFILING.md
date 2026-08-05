# Runtime profiling

Stage 1 can collect an internal aggregate timing profile without emitting an
event for every evaluation. Profiling is enabled by default and can be
controlled in the search configuration:

```toml
[search]
profiling_enabled = true
```

Set the value to `false` for a profiling-overhead control run. Older
configuration files without the key retain the enabled default.

HEG operator and scoring internals can be measured with a separate, opt-in
deep profile:

```toml
[search]
deep_profiling_enabled = true
```

Deep profiling is disabled by default and is independent of
`profiling_enabled`. For proposals, it enables HEG's
`MutationProfileAccumulator` and records witness search, witness-edge
materialization, switch attempts, partner-edge sampling, candidate
construction, connectivity validation, and graph-family validation. For
scoring, it records prepared-graph work, score-cache and cutoff counters,
request packing and pipe I/O, C++ cycle-count time and nodes by forbidden
length, score assembly, worker failures and restarts, and Python fallback.
These measurements do not change the search policy.

The aggregate results are stored as `deep_operator_profile` and
`deep_score_profile` in `run_summary.json` and terminal events. Per-episode
data uses the same names under `episodes[]`. Rich renders separate
`Deep operator profile` and `Deep score profile` panels. Inclusive operator
rows are broken into non-overlapping measured children; `other` is the untimed
remainder, including early-rejection and operator-loop overhead that the
current HEG interface does not expose separately.

Deep profiling adds timers inside witness searches, switch attempts, and the
score-worker Python protocol path. Use it for diagnosis, not for throughput
comparisons. Leave it disabled when measuring normal runtime performance.

Three scoring optimizations are enabled by default and can be disabled
independently for ablation runs:

```toml
[search]
score_cache_enabled = true
score_cutoff_enabled = true
prepared_graph_cache_enabled = true
prepared_proposal_handoff_enabled = true
score_longest_first_enabled = true
score_compact_dominated_enabled = true
score_prepared_request_cache_enabled = true
```

The episode-local score cache stores only complete `GraphScore` results; worker
failures and cutoff-dominated partial results are never cached. The inclusive
worker cutoff is used only when the current objective and strict-improvement
controller make rejection provably safe. Unsupported graph orders, score
shapes, invalid scores, and heuristic-zero incumbents automatically use full
scoring. The prepared-graph cache holds two immutable graph states so rewrite
validation, scoring, serialization, and proposal generation can reuse
`BitGraph` construction and validation.

The persistent C++ worker is restarted once after a request failure. A second
failure switches the backend to the bounded Python reference scorer for the
rest of the run. Deep profile counters make both transitions visible.

The prepared-proposal handoff is a bounded, backend-owned one-entry bridge
between HEG proposal generation and rewrite application. It reuses only the
exact `RewritePlan` object returned for the same source `GraphState`; replaced,
copied, modified, or externally constructed plans use the full reconstruction
and validation path. The handoff is consumed on application, replaced by every
new proposal, and cleared on graph deserialization, seed generation, or backend
close.

With `score_longest_first_enabled`, cutoff requests evaluate forbidden cycle
lengths from longest to shortest so a capped long-cycle count can prove
domination before shorter lengths are visited. Complete scores retain the
canonical increasing order, and worker responses are sorted before they cross
the protocol boundary. Disabling the flag preserves the former increasing
evaluation order for ablation and parity checks.

With `score_compact_dominated_enabled`, non-profiled cutoff requests ask the
worker to return a header-only dominated response. Full scores remain complete,
and deep profiling always requests detailed per-cycle records. Disable the
flag only for same-binary protocol ablation.

`score_prepared_request_cache_enabled` keeps one validated immutable worker
request plan for repeated requests with the same order, lengths, limits, and
protocol mode. Dynamic graph rows, request IDs, cutoffs, and flags are never
cached.

`HegBackend` normally keeps the current and most recent candidate HEG
`BitGraph`, plus one forbidden-witness context for the current immutable
`GraphState`. Repeated targeted proposals against that same state reuse
witness choices. An accepted graph replacement or a new episode supplies a
new logical state, causing one witness-cache miss and search before reuse
resumes. The context changes deterministic bookkeeping only; witness selection
remains at the same RNG call site.

Each episode measures these non-overlapping phases with
`time.perf_counter_ns()`:

- `scoring`: initial and candidate score calls;
- `proposal_generation`: baseline mutation proposal generation;
- `rewrite_application`: applying and validating a proposed rewrite;
- `duplicate_detection`: exact immutable-state set bookkeeping;
- `controller`: acceptance, best-state, and curve bookkeeping;
- `exact_verification`: exact checks for new heuristic-zero states;
- `progress_reporting`: aggregate progress callbacks and their event sinks;
- `finalization`: final graph serialization and canonical hashing.

For HEG-backed runs, `proposal_generation` expands into these child phases:

- `rng_setup`: operator selection and deterministic RNG construction;
- `graph_materialization`: conversion from `GraphState` to HEG `BitGraph`;
- `operator_search`: the selected HEG mutation operator;
- `proposal_packaging`: conversion of the operator delta to `RewritePlan`;
- `other`: proposal wrapper and timer bookkeeping not attributed above.

The episode profile is stored in `run_summary.json` and as
`episode_timing_profile` in the corresponding `episode_completed` event. That
event also contains the cumulative run `timing_profile`, allowing Rich output
to update a phase table after each episode. The final run summary and terminal
event contain the complete aggregate profile with phase seconds, measured
total, accounted time, unattributed time, and the dominant phase. Unattributed
time covers loop control, deadline checks, timer overhead, and other work
outside the named phases.

Rich renders the cumulative phase table during the run. A minimal internal
grid separates its columns and the phase, subtotal, and episode-total groups
without adding a second outer frame inside the panel. Its final table also
shows process `real/user/sys` time on a separate line below the grid, so the
long process-time value does not widen the phase columns. With profiling
disabled, process time remains in the main overview panel.

Child rows under `proposal_generation` form a tree. `Calls` counts the parent
and each explicitly instrumented child; synthetic `other` has no meaningful
call count. `graph_materialization` counts actual `GraphState` to `BitGraph`
conversions, so its count decreases when the current-graph cache is reused.
`Of parent` reports each child's share of proposal generation; `Of episode`
reports every row's share of the measured episode total. The same hierarchy is
available in JSON as `phase_children_seconds`, `phase_calls`, and
`phase_children_calls`.

Profiles are timing observations only. They are excluded from the canonical
summary hash. A valid overhead comparison uses identical configurations,
seeds, datasets, and evaluation budgets with only `profiling_enabled` changed,
then verifies equal canonical summary hashes. Run each mode more than once and
alternate their order before interpreting a small wall-time difference.

## Validation

A balanced `on, off, off, on` smoke comparison on 2026-07-29 completed 8,000
evaluations in every run and produced the same canonical summary hash four
times. After adding proposal child timers, mean benchmark real time was 3.318
seconds with profiling and 3.301 seconds without it, an observed overhead of
0.49%.

The two enabled runs attributed approximately 51% of measured episode time to
proposal generation, 26% to scoring, 12% to duplicate detection, 8% to rewrite
application, and 0.3% to progress reporting. Unattributed time was 0.3%.
These figures describe that machine and smoke workload; they are evidence that
the profiler is cheap, not a portable performance guarantee.

A current-graph witness-cache audit on 2026-07-29 recorded 5,000 targeted
operator calls, 5,000 cache lookups, 4,987 hits, 13 misses, and 13 witness
searches: a 99.74% hit rate with all cache accounting identities satisfied.
Paired cache-on/off targeted episodes produced identical logical results.
Across two policy seeds, mean targeted episode time fell from 2.344 seconds to
1.061 seconds. A 2,000-evaluation uniform control differed by 0.3%, within
single-pair timing noise.

A balanced `on, off, off, on` scoring-optimization ablation on 2026-07-29 used
the 16-episode, 80,000-evaluation Stage 1 workload. All four runs produced the
same canonical summary hash. Enabling the score cache, safe worker cutoff, and
prepared-graph cache together changed mean real time from 12.704 to 10.246
seconds, mean throughput from 6,302 to 7,808 evaluations/s, and attributed
scoring time from 6.857 to 4.454 seconds. On that machine and workload, this
was 19.3% less wall time, 23.9% higher throughput, and 35.0% less scoring time.
Proposal generation improved by 3.7% after repeated current-state proposals
were changed to use the identity fast path before consulting the prepared
graph LRU. These measurements describe the combined bundle; the three
switches exist so each component can be isolated in later runs.
