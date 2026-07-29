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

HEG operator internals can be measured with a separate, opt-in deep profile:

```toml
[search]
deep_profiling_enabled = true
```

Deep profiling is disabled by default and is independent of
`profiling_enabled`. It enables HEG's `MutationProfileAccumulator` for every
proposal and records witness search, witness-edge materialization, switch
attempts, partner-edge sampling, candidate construction, connectivity
validation, and graph-family validation. It does not enable the HEG mutation
witness cache or change the search policy.

The aggregate result is stored as `deep_operator_profile` in
`run_summary.json` and terminal events. Per-episode data is stored as
`episodes[].deep_operator_profile`. Rich renders it in a separate
`Deep operator profile` panel. Inclusive operator rows are broken into
non-overlapping measured children; `other` is the untimed remainder, including
early-rejection and operator-loop overhead that the current HEG interface does
not expose separately.

Deep profiling adds timers inside witness searches and switch attempts. Use it
for diagnosis, not for throughput comparisons. Leave it disabled when
measuring normal runtime performance.

Each episode measures these non-overlapping phases with
`time.perf_counter_ns()`:

- `scoring`: initial and candidate score calls;
- `proposal_generation`: baseline mutation proposal generation;
- `rewrite_application`: applying and validating a proposed rewrite;
- `duplicate_detection`: state hashing and duplicate-set bookkeeping;
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
call count. `Of parent` reports each child's share of proposal generation;
`Of episode` reports every row's share of the measured episode total. The same
hierarchy is available in JSON as `phase_children_seconds`, `phase_calls`, and
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
