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
shows process `real/user/sys` time below the profile totals. With profiling
disabled, process time remains in the main overview panel.

Profiles are timing observations only. They are excluded from the canonical
summary hash. A valid overhead comparison uses identical configurations,
seeds, datasets, and evaluation budgets with only `profiling_enabled` changed,
then verifies equal canonical summary hashes. Run each mode more than once and
alternate their order before interpreting a small wall-time difference.

## Validation

A balanced `on, off, off, on` smoke comparison on 2026-07-29 completed 8,000
evaluations in every run and produced the same canonical summary hash four
times. Mean benchmark real time was 3.324 seconds with profiling and 3.314
seconds without it, an observed overhead of 0.32%.

The two enabled runs attributed approximately 51% of measured episode time to
proposal generation, 26% to scoring, 12% to duplicate detection, 8% to rewrite
application, and 0.3% to progress reporting. Unattributed time was 0.3%.
These figures describe that machine and smoke workload; they are evidence that
the profiler is cheap, not a portable performance guarantee.
