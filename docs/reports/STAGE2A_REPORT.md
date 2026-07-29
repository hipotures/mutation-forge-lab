# Stage 2A report

Date: 2026-07-29
Status: **accepted for the narrow Stage 2A execution-safety scope**

## Scope and frozen entry point

Stage 2A started from:

- Mutation Forge `3b9beba058f472d6f0cad5b6210f34c6dbf96731`
- HEG `fd97451b0f3d87400d1d955a2c6b1b18303344ff`

It implements deterministic validation, isolated execution, probing,
identity, replay, CLI, and durable artifacts for exactly:

```python
def priority(ctx, proposal):
    """Return a finite number. Larger values are preferred."""
    ...
```

The inputs are versioned, bounded Stage 2A execution probes. They are not the
final scientific feature schema. This stage adds no generalized k-switches,
proposal pool, full proposer, random/structural HEG comparison, model or App
Server call, evolution, held-out claim, or HEG integration. HEG was read only
throughout.

## Delivered contract

- `stage2a.probe.v1` defines exact typed `ctx` and `proposal` mappings and
  recursively bounded JSON-compatible plain data. Runtime canonicalization
  limits depth, mapping and sequence sizes, UTF-8 strings and keys, integers,
  finite floats, and encoded request size. The worker freezes mappings and
  sequences before calling the ranker.
- `stage2a.validator.v1` accepts one undecorated top-level
  `priority(ctx, proposal)` function, one final return, local-name assignments,
  conditionals, bounded `for`, arithmetic/comparison/Boolean expressions,
  indexing/slicing, bounded literals, and selected deterministic built-ins.
  Every other AST class, attribute access, private/unknown name, non-local
  assignment target, and unbounded loop source is rejected with structured
  source locations.
- Program identity persists exact source SHA-256, normalized AST SHA-256, AST
  node count, validator version, probe-schema version, and validation result.
  Tests show formatting and local-variable renaming preserve normalized
  identity while semantic operator changes do not.
- Only a spawned persistent subprocess compiles and calls a valid policy. The
  coordinator verifies the child's isolated cwd, sanitized environment,
  protocol-only stdin, separate process group, and all Linux resource limits
  during the initialization handshake.
- The protocol is length-prefixed canonical UTF-8 JSON. It uses no pickle.
  Invalid output, exception, timeout, crash, or protocol failure terminates
  and reaps the process group; the failed worker cannot be reused.
- `stage2a.behavior.v1` records finite priorities, deterministic rank order,
  proposal-ID tie-breaking, selected IDs, and
  exception/timeout/crash/protocol flags over a fixed non-held-out probe set.
- Replay reads the persisted source and effective limits. It reproduces exact
  source and normalized-AST hashes, outputs, rank order, selections, flags,
  and behavior-signature hash without a model or network call.

## Default limits

| Resource | Limit |
|---|---:|
| Source | 12 KiB |
| AST nodes | 500 |
| Static loop bound | 256 |
| Worker address space (`RLIMIT_AS`) | 128 MiB |
| Per-call parent wall limit | 25 ms |
| Per-program parent wall / smoke limit | 60 s |
| Worker CPU (`RLIMIT_CPU`) | 60 s soft / 61 s hard |
| Request frame | 64 KiB |
| Response frame | 16 KiB |
| Captured output / `RLIMIT_FSIZE` | 64 KiB |
| Open files (`RLIMIT_NOFILE`) | 16 |
| Process count (`RLIMIT_NPROC`) | 1 |

Unsupported platforms fail closed rather than dropping these limits.

## Commands validated

```console
uv run mforge policy validate fixtures/rankers/weighted.py --json
uv run mforge policy probe fixtures/rankers/weighted.py --json
uv run mforge policy evaluate fixtures/rankers/weighted.py \
  --config configs/stage2a-probe.toml --json
uv run pytest
uv run ruff check .
uv run mypy
uv run mforge doctor --heg-repo ../heg
git diff --check
git -C ../heg rev-parse HEAD
git -C ../heg status --short
```

All commands passed. The full suite reports **147 passed, zero skipped** in
2.46 seconds. It includes all existing Stage 1 unit, integration, parity, and
HEG tests plus Stage 2A contract, validator, adversarial, worker, CLI,
artifact, and replay tests. Ruff reports no findings, strict mypy reports no
issues in 38 source files, doctor reports every required check as passing, and
`git diff --check` is clean.

One valid persistent worker completed 10,000 bounded calls. Tests also force
runtime exceptions, invalid/non-finite/oversized outputs, memory/CPU abuse,
worker death, parent wall timeout, and protocol corruption, then confirm the
coordinator remains healthy. Shutdown reaps the worker and removes its
isolated temporary cwd. No policy worker remained after the suite.

Static tests reject imports; file, environment, process, network, and RNG
access; dunder/reflection and dynamic execution; `while`; recursion; nested or
multiple functions; decorators and other forbidden control forms; wrong
signatures; hidden state; input mutation; `print`; and every explicitly banned
dynamic/reflection built-in. Runtime tests cover allocation, large integer and
container output, Boolean, NaN/infinity, exceptions, timeout/crash, and
malformed protocol handling.

## Artifact evidence

Representative run:

`runs/stage2a-20260729T131812.596307Z-e7a2f4955565`

It completed with all failure flags false and contains:

- `artifacts/programs/policy.py` with the exact evaluated source;
- `validation.json` and `identity.json`;
- `limits.json`;
- `behavior_signature.json`;
- `worker_telemetry.json`;
- `provenance.json`;
- `result.json`;
- `terminal_status.json`;
- the copied Stage 2A evaluation config.

The CLI JSON and durable `result.json` are canonically identical. Rich and JSON
rendering receive the same canonical result object; tests compare both
renderings. Source, validation, behavior, request/response, and captured
diagnostics stay below their configured caps.

Worker telemetry records five successful probe calls, zero failures, the
isolated `/tmp/mforge-policy-*` cwd, only `HOME`, `LANG`, `LC_ALL`, and `PATH`
in the environment, protocol-pipe stdin, a separate process group, and the
exact five resource-limit pairs listed above.

Provenance records zero model calls and zero network calls. The frozen Mutation
Forge commit is verified as the development base. HEG pin verification passed,
and its checkout remained clean at
`fd97451b0f3d87400d1d955a2c6b1b18303344ff`.

## Failures found and resolved

Development validation initially found ordinary lint/type-narrowing issues and
a test whose 10 ms total limit expired during worker startup. These were fixed
without weakening runtime defaults.

A later evidence review found that replay used default limits instead of the
persisted effective limits. Replay now loads `limits.json`, and a non-default
limit is covered by the artifact/replay integration test. The same review led
to a verified worker-control handshake and direct assertions for resource
limits, sanitized authority, cleanup, full artifacts, and frozen-repository
gates. No known acceptance failure remains.

## Scientific interpretation and remaining gates

This evidence supports only the narrow claim that the Stage 2A probe ranker
runtime is deterministic, replayable, and bounded under the tested Linux
environment. The fixed probes are not a graph-search benchmark and do not
claim HEG superiority.

Stage 2B may design host-owned proposal pools and scientific features only
after issue #5 is closed as completed. Stage 2B must still use no model. No
full proposer may be implemented before ranker evidence, no model may be used
before the later Stage 2B GO, and no HEG policy integration may occur before
the final scientific GO.

## Gate decision

**GO_TO_STAGE_2B**

Stop here. Do not start Stage 2B or issue #6 automatically.
