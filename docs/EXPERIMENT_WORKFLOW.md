# Native v3 experiment workflow

Native v3 is the active experiment protocol. The installed public commands
are:

```console
uv run mforge experiment run [--config PATH] [--json]
uv run mforge experiment status [--config PATH] [--json]
uv run mforge experiment stop --final [--config PATH] [--json]
```

`experiment.toml` is authoritative. A fresh run atomically creates
`workspace/<exp_id>/`, records the project and read-only sibling HEG
provenance, and locks the Native v3 schemas, prompts, registries, scoring
protocol, and operator-selected profiles. Native v2 workspaces and generated
Python rankers are not accepted by this runtime.

## Frozen epoch and provider calls

At the start of every epoch the host freezes:

- the eight planned slot IDs;
- retained parent and archive identities;
- parent assignment and mutation brief for every slot;
- the development and sealed validation manifests;
- the complete protocol bundle.

Every provider call in that epoch uses this versioned snapshot. Calls return a
bounded batch of one, two, four, or eight independent `program_json_raw`
values. The default batch size is four. The host persists raw text separately
from the validated AST, host-canonical JSON, and program hash. Slot identity
and lineage are host-owned.

The input profile is a hard ceiling of four complete parent ASTs, two parent
references per slot, 128 KiB, and 32k tokens. Remaining archive content is a
deterministic bounded summary. The actually encoded request is checked before
dispatch; it is never padded, silently truncated, or rebound to different
parents.

## Streaming evaluation

Provider generation and graph evaluation form a bounded producer-consumer
pipeline:

```text
provider calls
  -> independently parsed and validated ASTs
  -> bounded candidate queue
  -> deterministic episode microshards
  -> bounded evaluator queue
  -> persistent CPU evaluator processes
```

The first valid AST starts immediately on the reserved evaluator. Candidate
microshards take priority as already-running auxiliary microshards reach their
next terminal boundary, allowing one program to fan out across all configured
workers. Provider calls continue while evaluation runs. A target evaluation
backlog pauses provider dispatch before an unbounded backlog forms and resumes
it before evaluators starve.

At epoch start, uncached retained-parent and fixed-baseline microshards may use
at most `W-1` evaluators. They are useful scientific work, not synthetic
occupancy. No new auxiliary shard is dispatched while generated-candidate
work waits.

## Program execution and scoring

Generated programs are declarative ASTs interpreted against a private graph
overlay. Selectors observe the current overlay and consume costs from the
versioned selector registry. The default per-`propose` limit is 128 cost
units; an uncached query consumes its registered cost and an identical cache
hit consumes one unit. Runtime budget escape is an uncatchable
`PROGRAM_FAILURE`.

Private intermediate overlays may temporarily violate connectivity or minimum
degree. They always preserve structural memory safety, and only a final
simple, connected, same-order graph with minimum degree at least three may be
emitted. The host converts the overlay to a canonical net edge difference,
validates it, scores only the final graph, and applies deterministic
acceptance.

Score results retain component evidence for every forbidden length. Bounded
search states produce sound intervals; infrastructure failure is not converted
to ordinary worst-case fitness, and a contract violation fails closed. The
mandatory persistent C++ scorer may restart once. A second failure raises an
infrastructure error; Native v3 never enters a Python reference scorer under
the same score protocol.

## Selection and replay

Development fitness is comparable only for identical development manifest and
protocol hashes. Retained parents are re-evaluated when the adaptive
development manifest changes. After development, the host durably freezes a
deterministic promotion shortlist of at most four programs and evaluates every
eligible distinct member on the sealed validation panel.

Epoch terminal semantics are:

```text
8 unique valid generated programs -> COMPLETE
4-7                              -> DEGRADED
0-3                              -> INCONCLUSIVE and safe stop
```

Provider and evaluator results may arrive in any order. Selection, archive,
lineage, and checkpoint state are committed in deterministic identity order
after the planned cohort reaches a terminal state. Timestamps, durations,
worker IDs, and queue arrival order are observational telemetry and do not
participate in semantic replay identity.

## Exact verification

Every legal apparent zero is durably written before ordinary continuation and
enters the dedicated verification supervisor. Jobs are deduplicated by
`(graph_hash, verification_protocol_id)`.

The locked default profile is:

```text
concurrency:       1
unique-job queue: 16
primary timeout:  600 seconds / 4 GiB
independent:      600 seconds / 4 GiB
```

The independent verifier runs only after the primary returns complete
`VERIFIED`. Both execute sequentially in isolated processes. A certificate is
created only when both return complete `VERIFIED`; incomplete or conflicting
outcomes remain durable non-success states. A full verification queue applies
backpressure to new scoring, and apparent-zero events are never dropped.

## Evidence and monitoring

Native v3 stores deterministic semantic records through one persistence owner
and separates them from observational telemetry. A durably committed terminal
episode is never rerun; computed but uncommitted work may be replayed
idempotently after a crash.

The dashboard and JSON events expose provider concurrency and latency,
programs and valid programs per call, queue depths, evaluator utilization,
provider starvation, evaluation backpressure, phase wall shares, first-AST
fan-out latency, graph-score and episode rates, rewrite acceptance, score-cache
hits, active C++ scorers, restarts, and forbidden fallback count. These fields
distinguish provider-bound, evaluator-bound, persistence-bound, and
verification-backpressured execution.
