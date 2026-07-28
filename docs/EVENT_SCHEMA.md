# Event schema

Every event is one UTF-8 JSON object on one line and contains:

- `schema_version`: currently `"1.0"`;
- `timestamp`: timezone-aware ISO 8601 UTC timestamp;
- `run_id`: identifier shared by all events in a run;
- `event_type`: a registered event name;
- event-specific fields.

Stage 1 emits `run_started`, `backend_ready`, `dataset_loaded`,
`baseline_started`, `episode_started`, `episode_progress`,
`episode_completed`, `checkpoint_written`, `run_completed`, and `run_failed`.
The registry also reserves the specified program-generation, static/sandbox
validation, evaluation, and champion events for later versioned milestones.

With `--json`, stdout contains JSON Lines only: no ANSI, progress bars, or
prose. Bootstrap failures that occur before a run identifier/artifact exists
may be reported on stderr. Once a run begins, a fatal error is persisted and
emitted as the final `run_failed` event.

Terminal events include `real_seconds`, `user_seconds`, and `system_seconds`.
CPU values combine the `mforge` process with reaped child processes, including
the persistent HEG score worker and external graph tools.

When runtime profiling is enabled, `episode_completed` contains the aggregate
episode `timing_profile`, while `run_completed` contains the aggregate run
profile. No per-evaluation profiling events are emitted.

Rich and JSON consume the same `Event` objects. Canonical run-summary equality,
not terminal formatting, defines output parity.
