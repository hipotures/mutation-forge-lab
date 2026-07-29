# Generated Python security boundary

Generated Python is not executed in Stage 1. Stage 2A executes only one
validated function:

```python
def priority(ctx, proposal):
    """Return a finite number. Larger values are preferred."""
    ...
```

`ctx` and `proposal` use `stage2a.probe.v1`. They are exact string-keyed
mappings containing probe IDs, counters, a proposal kind, and recursively
bounded plain-data feature mappings. Values may be `None`, Boolean, bounded
integer/finite float/string, lists or tuples, and string-keyed mappings.
Canonical protocol JSON turns tuples into arrays; the worker recursively
freezes arrays and mappings before the call. The schemas are execution probes,
not final scientific features.

Validation accepts exactly one undecorated top-level function with parameters
`ctx, proposal`, local-name assignments, conditionals, bounded `for` loops,
arithmetic, comparisons, Boolean logic, indexing/slicing, bounded literals,
and selected deterministic built-ins. It rejects every AST node not explicitly
allowed, all attribute access, private names, mutation targets other than local
names, unknown names, non-static loop sources, dynamic execution, imports, and
the other adversarial classes listed in `MILESTONES.md`. Defaults are 12 KiB
source, 500 AST nodes, and a 256-item static loop bound. Validation errors are
machine-readable and retain source locations.

The coordinator never executes policy code. A dedicated persistent subprocess
starts in a new process group, sanitized environment, isolated temporary
working directory, and receives no inherited stdin. Length-prefixed canonical
JSON is the only protocol; pickle is never used. Linux applies 128 MiB
`RLIMIT_AS`, a 60-second cumulative CPU ceiling, 64 KiB `RLIMIT_FSIZE`, 16
descriptors, and one process, while the parent enforces 25 ms per call and 60
seconds total. Requests are capped at 64 KiB, responses at 16 KiB, and captured
diagnostics at 64 KiB. Unsupported platforms fail closed. Timeout, crash,
protocol violation, runtime exception, invalid output, and shutdown all
terminate and reap the worker; a failed worker is never reused.

The only valid output is a finite non-Boolean `int` or `float`, additionally
bounded by integer bit length and response size. Policy code never receives a
graph, scorer, verifier, backend, filesystem, environment, process, database,
network, RNG, experiment state, or rewrite authority.

Identity records the exact source SHA-256, a normalized AST SHA-256 stable
across formatting and local-variable renaming, AST node count, validator
version, probe-schema version, and validation result. A fixed non-held-out
probe set records priorities, finite/failure flags, deterministic rank order
and selected IDs; its canonical hash is the versioned behavior signature.
Replay loads the persisted source and reproduces identity, outputs, and the
signature without a model or network call.

This is a narrow defense-in-depth contract, not a general Python sandbox and
not proof that arbitrary Python is safe. Stage 2B, model use, a full proposer,
and HEG integration remain separately gated.
