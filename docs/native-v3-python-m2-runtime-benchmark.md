# Native v3 Ordinary-Python M2 Runtime Benchmark

## Status

This document records the M2 benchmark of the isolated ordinary-Python policy
worker. It does not accept or activate the measured limits. The operator must
freeze the runtime protocol before M3 begins.

The benchmark used Linux x86-64, CPython 3.12.13, and the M2 worker protocol
`mforge.native.python_policy_runtime.v1`.

## Method

The exact command was:

```text
uv run python scripts/native_v3_python_runtime_benchmark.py
```

It ran:

- 20 worker startups;
- 200 normal `NoPlan` calls;
- 200 calls distributed over seven selector/action fixtures;
- 20 isolated invalid-return failure paths.

The selector/action fixtures covered add-edge, relocation, fanout, 2-switch,
edge-fold, articulation/bridge/local-cycle risk selectors at graph order 128,
and witness-load selectors at graph order 64. The order-128 case intentionally
exercised the accepted graph-order cap.

The benchmark is an offline runtime measurement. It did not contact a model,
invoke HEG, score fitness, run exact verification, or integrate the scientific
evaluator.

## Results

| Measurement | p50 | p95 | maximum |
| --- | ---: | ---: | ---: |
| Worker startup | 25.761 ms | 27.075 ms | 27.200 ms |
| Normal call | 0.104 ms | 0.137 ms | 0.244 ms |
| Selector/action call | 1.402 ms | 6.354 ms | 11.790 ms |
| Invalid-return process lifetime | 27.738 ms | 28.716 ms | 28.944 ms |

The selector/action matrix was:

| Fixture | Calls | Graph order | p50 | p95 | maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Add edge | 29 | 30 | 3.599 ms | 4.094 ms | 4.436 ms |
| Relocate endpoint | 29 | 30 | 1.400 ms | 2.704 ms | 4.743 ms |
| Edge fanout | 29 | 30 | 1.192 ms | 1.341 ms | 1.686 ms |
| 2-switch | 29 | 30 | 1.559 ms | 1.644 ms | 1.926 ms |
| Edge fold | 28 | 30 | 0.675 ms | 0.848 ms | 0.979 ms |
| Risk selectors | 28 | 128 | 6.331 ms | 8.928 ms | 11.790 ms |
| Witness selectors | 28 | 64 | 0.817 ms | 1.152 ms | 1.165 ms |

Maximum observed worker RSS was 18,972 KiB. Maximum observed worker age was
0.220 seconds. The largest observed semantic trace contained four safe-API
calls.

Against the provisional limits, the measured headroom was:

| Resource | Provisional limit | Maximum observed | Measured headroom |
| --- | ---: | ---: | ---: |
| Propose wall time | 1.000 s | 0.011790 s | 0.988210 s |
| Address space | 256 MiB | 18,972 KiB RSS | 249,008,128 bytes |
| Worker lifetime | 60.000 s | 0.220 s | 59.780 s |
| Total API calls | 256 | 4 | 252 |
| Graph order | 128 | 128 | 0 by intentional boundary coverage |

RSS is not the same quantity as virtual address space. The address-space row
therefore demonstrates substantial observed headroom but is not a direct
measurement of the closest possible `RLIMIT_AS` value.

## Runtime and API Budget Recommendation

M2 recommends that the operator retain and freeze these primary limits for the
initial runtime protocol:

- propose wall time: 1 second;
- worker address space: 256 MiB;
- candidate-worker lifetime: 60 seconds.

The 1-second wall limit has more than 85 times the observed maximum-call
headroom while remaining a strict per-call boundary. The 256 MiB address-space
limit leaves room for CPython virtual-memory behavior that RSS alone does not
measure. The 60-second lifetime permits worker reuse while bounding accumulated
state and process lifetime. Before an invocation that cannot receive the full
1-second wall window, the host cleanly reaps the idle process, starts the same
validated source in a fresh sandbox, and repeats every sandbox attestation.
Host idle or scoring time therefore causes transparent rotation rather than a
program failure or a shortened `propose` deadline. Reducing either memory or
lifetime should require a separate cross-platform benchmark rather than
inference from this Linux run.

M2 also recommends freezing the following capability budgets:

- graph order: 128;
- total safe-API calls: 256;
- selector calls: 64;
- action calls: 64;
- selector result size and `pick` input size: 64 references;
- deterministic random draws: 2,048;
- net added edges: 8;
- net removed edges: 8;
- loop-body entries: 4,096;
- helper invocations: 256;
- helper call depth: 8;
- request frame: 256 KiB;
- response frame: 32 KiB;
- captured diagnostics: 64 KiB;
- file-size limit: 64 KiB;
- open descriptors: 16;
- process count: 1;
- CPU limit: 60 seconds.

Every limit is a hard maximum in M2: callers may lower it but cannot raise it
through `PolicyRuntimeLimitsV1`.

## Isolation and Fail-Closed Boundary

Generated source is validated by M1 before worker creation and is compiled and
executed only in the dedicated child. The coordinator never imports, compiles,
or executes candidate source.

The worker uses a private user, mount, PID, network, IPC, and UTS namespace; an
empty mount root; a read-only Python runtime and worker entry point; a private
working directory; a sanitized environment; resource limits; `no_new_privs`;
and seccomp denial of filesystem-open, network, process creation, ptrace,
ambient-randomness, and clock syscalls. Startup fails unless the child proves
the expected controls and exactly descriptors 0, 1, and 2.

The framed protocol uses canonical JSON and rejects duplicate keys, non-finite
numbers, non-canonical encodings, unexpected messages, oversized frames, stale
or foreign references, and unknown API methods. A program result is accepted
only when it resolves to an invocation-scoped result minted by the host.

Program-caused exceptions, budget exhaustion, timeout, invalid return, memory
exhaustion, and candidate-process crash are classified as
`PROGRAM_FAILURE`. Sandbox, protocol, or trusted-host failures are
`INFRASTRUCTURE_FAILURE` exceptions and must not become scientific fitness.
Only the explicit host `IllegalRewriteError` contract converts a
candidate-invalid final graph to `NoPlan("ILLEGAL_FINAL_STATE")`.
The 60-second process-lifetime boundary is enforced between invocations by
transparent rotation and is never reported as candidate fitness.

## Residual Limitations

- The sandbox implementation is Linux-only and requires bubblewrap,
  unprivileged namespaces, and libseccomp. Unsupported platforms fail closed.
- The seccomp profile is a denylist combined with an empty mount root and
  private namespaces, not a syscall allowlist.
- CPython and its read-only runtime remain part of the trusted computing base.
  Static AST validation is defense in depth and is not treated as the sandbox.
- vDSO-backed clock reads cannot be intercepted by seccomp alone. The accepted
  AST and minimal globals expose no clock or reflection capability, while
  direct clock syscalls are denied.
- A trusted host callback that never returns cannot be preempted by the current
  coordinator thread. M2 checks the wall deadline before and after every
  finite host API call; M3 must keep fixture callbacks bounded.
- The benchmark covers isolated fixture calls, not a development panel or
  scientific evaluator. M3 must measure serial fixture integration without
  changing these limits absent a new operator decision.

## Recommendation

M2 runtime implementation and benchmark evidence support operator acceptance
of the limits above. This is a recommendation only. M3 remains blocked until
the operator explicitly freezes the runtime protocol and accepts issue #51.
