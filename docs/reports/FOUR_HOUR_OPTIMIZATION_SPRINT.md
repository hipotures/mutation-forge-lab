# Four-hour optimization sprint

GitHub issue: [#2](https://github.com/hipotures/mutation-forge-lab/issues/2)

## Sprint record

- Started: 2026-07-29 05:16:38 BST
- Ended: 2026-07-29 06:38:27 BST
- Actual time: 1 h 22 min
- Time box end: 2026-07-29 09:16:38 BST
- Mutation Forge base: `2d80589d308c68b219208d8b3c95d382e48d3c99`
- Mutation Forge final implementation:
  `292274be3b63288b450ca3229f05c7072372b268`
- HEG base: `c691dbcec249afc602366cee092ad177303b4dbc`
- HEG final: `b228058463ac6c3d111ea983c346e2dcb07b5ee1`
- Score worker: `/home/user/DEV/heg/_build/sglab-score-worker`
- Score-worker SHA-256 before:
  `dcc040ddbe92c235a13134b7746707e59ec68d84c9be693a56672df9a0193df7`
- Score-worker SHA-256 after:
  `23668212f1a4421abaafa89c7802eda92fd5605dfb3979f322d1d9f6022228e1`
- Branch: `agent/four-hour-performance-sprint`
- Benchmark configuration:
  `runs/stage1-20260729-035113-94409551/run_config.toml`
- Machine: AMD Ryzen 9 7950X3D, 16 available logical CPUs, Linux
  7.0.0-28-generic x86-64
- Runtime: Python 3.12.13, uv 0.11.9
- Toolchain: GCC 15.2.0, CMake 4.2.3

Both repositories were clean, synchronized with `origin/main`, and contained
all issue-required starting revisions before the sprint branch was created.
No benchmark, test, or score-worker process was running.

## Experiment log

### Baseline

Smoke:

- artifact: `runs/stage1-20260729-041759-e698eaa3`
- evaluations: 8,000
- real/user/sys: 1.151 / 0.960 / 0.120 s
- throughput: 6,953 evaluations/s
- canonical summary hash:
  `8843261470ef4898c4a551b0db2e1aae54925af8fbf108a990eadd4b81abede6`

Fresh full baseline with deep profiling disabled:

- artifact: `runs/stage1-20260729-041805-99da18c6`
- evaluations: 80,000
- real/user/sys: 10.214 / 8.940 / 0.830 s
- throughput: 7,832 evaluations/s
- scoring: 4.426 s
- proposal generation: 2.495 s
- rewrite application: 2.836 s
- canonical summary hash:
  `d6b3645bea4cd9bcba471e1c729dc5211edb969e4dc7fc5d3fb86cecec74c678`

The final performance claims will use balanced repeated runs totaling at least
one minute per variant; this single full run only freezes the initial state.

### Prepared-proposal handoff

**GO.** Implemented a one-entry backend-owned handoff from
`propose_rewrite()` to `apply_rewrite()`. The retained lazy variant carries the
HEG `BitGraph` and constructs `GraphState` in `apply_rewrite()`. It therefore
avoids edge-set reconstruction, a second HEG graph materialization, and a
second validation without moving `GraphState` construction into proposal
generation.

The fast path requires the same source object, the exact backend-returned
`RewritePlan` object, and exact removed edges, added edges, operator family,
and evaluation metadata. Any mismatch consumes the handoff and uses the
original full reconstruction and validation path. Every proposal replaces the
one entry; seed generation, graph deserialization, application, and close
clear it.

Targeted tests cover hit, copied-plan fallback, stale-plan replacement,
equal-but-not-identical source fallback, graph deserialization invalidation,
close invalidation, and three-policy-seed episode parity.

Stable A/B evidence used six paired full runs in alternating order, plus two
additional optimized runs so each variant accumulated at least one minute:

| Metric | Handoff on | Handoff off | Change |
|---|---:|---:|---:|
| Runs in paired mean | 6 | 6 | — |
| Aggregate measured runtime | 68.333 s (8 runs) | 61.683 s (6 runs) | — |
| Mean real time | 8.515 s | 10.280 s | -17.2% |
| Mean throughput | 9,396/s | 7,782/s | +20.7% |
| Mean rewrite application | 1.035 s | 2.846 s | -63.6% |
| Mean proposal generation | 2.536 s | 2.496 s | +1.6% |
| Mean scoring | 4.380 s | 4.449 s | -1.6% |

All fourteen runs completed 80,000 evaluations with deep profiling disabled
and the same canonical summary hash:
`d6b3645bea4cd9bcba471e1c729dc5211edb969e4dc7fc5d3fb86cecec74c678`.
No worker failures occurred. The change clears both GO alternatives and keeps
proposal regression below the 2% ceiling.

### Score-worker round-trip overhead

#### A. Reusable Python request buffer

**NO-GO; reverted completely.** Tested one reusable bounded `bytearray`,
`Struct.pack_into()` directly into the final frame, and correct partial-write
handling. Targeted worker parity, bounded-buffer, and partial-write tests
passed.

The stable microbenchmark used 1,570,000 requests per variant:

| Metric | Before | Experiment | Change |
|---|---:|---:|---:|
| Wall time | 66.649 s | 66.280 s | -0.6% |
| Requests/s | 23,556 | 23,687 | +0.6% |
| Profiled worker round trip | 39.646 µs | 39.047 µs | -1.5% |
| Profiled request packing | 2.972 µs | 2.925 µs | -1.6% |

This did not approach the 10% microbenchmark gate, so no allocation-only
refactor or tests were retained.

#### B. Framed Python response read

**NO-GO; reverted completely.** Tested a bounded reusable response buffer,
one readiness/read path for header plus payload, correct fragmented reads,
retention of excess frame bytes, request-ID validation, and timeout behavior.
All targeted protocol and real-worker parity tests passed.

The same 1,570,000-request microbenchmark improved wall time by 4.9% and
profiled worker round trip by 4.8%, below the 10% micro gate. The alternate
full-workload gate used eight balanced runs per variant:

| Metric | Framed read | Control | Change |
|---|---:|---:|---:|
| Aggregate runtime | 67.201 s | 67.938 s | — |
| Mean throughput | 9,525/s | 9,421/s | +1.1% |
| Paired-median throughput | — | — | +1.4% |
| Mean scoring | 4.324 s | 4.384 s | -1.4% |

All sixteen full runs completed 80,000 evaluations with the canonical hash
`d6b3645bea4cd9bcba471e1c729dc5211edb969e4dc7fc5d3fb86cecec74c678`,
the same C++ binary, and no failures. The gain remained below the 3% workload
gate, so neither the implementation nor its added tests were retained.

#### C. Contiguous C++ request frame and container reuse

**NO-GO; reverted completely.** Tested one prefix read, one fixed-body/payload
read, in-memory little-endian parsing, and capacity reuse for request rows,
lengths, and result records. The rebuilt protocol-v2 worker passed all 19 HEG
score-worker tests. Experimental binary SHA-256:
`60447338d151cfc3ca0dea006374a81e9e2d759c256848b5e6c9e0d3a9ab0838`.

The 1,570,000-request microbenchmark improved request throughput by 5.4% but
reduced profiled round trip by only 2.0%, below the 10% gate. The alternate
eight-pairs-per-mode workload gate produced:

| Metric | C++ experiment | Control | Change |
|---|---:|---:|---:|
| Aggregate runtime | 66.822 s | 67.883 s | — |
| Mean throughput | 9,579/s | 9,429/s | +1.6% |
| Paired-median throughput | — | — | +2.1% |
| Mean scoring | 4.252 s | 4.384 s | -3.0% |

All sixteen runs retained the canonical hash and completed without failures,
but total throughput remained below the 3% gate. The source was reverted and
the original binary SHA-256 was restored:
`dcc040ddbe92c235a13134b7746707e59ec68d84c9be693a56672df9a0193df7`.

### Optional experiment

#### Cutoff-oriented internal cycle order

**GO; retained.** The final deep profile showed 1.730 s of C++ cycle work,
3.4 times `proposal_generation.other`. The worker now evaluates requested
cycle lengths from longest to shortest only when a cutoff is active. Partial
total and weighted penalties are monotone, so a domination proof is
independent of evaluation order. Complete scores are still evaluated in
canonical order, and all returned result records are sorted by length before
serialization.

The Python protocol accepts an ordered unique subset of requested lengths only
for a dominated response. An ordinary successful response must still contain
the exact complete requested sequence. This preserves protocol v2 and exposes
the timing/nodes for whichever lengths actually proved domination.

The representative 1,570,000-request microbenchmark used cutoff
`(64, 256, simplicity)` and accumulated at least one minute per variant:

| Metric | Longest first | Control | Change |
|---|---:|---:|---:|
| Requests/s | 28,299 | 23,784 | +19.0% |
| Profiled worker round trip | 33.347 µs | 40.146 µs | -16.9% |
| Result records/request | 1 | 3 | -66.7% |

The decisive full benchmark used policy seeds 1, 2, and 3: six balanced runs
per variant, 24 episodes and 120,000 evaluations per run.

| Metric | Longest first | Control | Change |
|---|---:|---:|---:|
| Aggregate runtime | 72.529 s | 77.322 s | — |
| Mean throughput | 9,933/s | 9,312/s | +6.7% |
| Paired-median throughput | — | — | +7.5% |
| Mean scoring | 5.816 s | 6.656 s | -12.6% |
| Mean real time | 12.088 s | 12.887 s | -6.2% |

All twelve runs shared canonical summary hash
`ac12ab84b03b71aea85510e66046e055c89dd745e3c2dc2fbf56f40bbe7498af`.
Their complete logical episode payloads excluding timing also shared SHA-256
`4ea81d5e25f4f19790eacd4a33ae7a87a86f94570bcb078204d3d906c36dd27f`.
The retained implementation is HEG `50db693`. Follow-up HEG commit `b228058`
adds a protocol-v2 ablation bit so ON/OFF can use the exact same commit and
binary. The final score-worker SHA-256 is
`23668212f1a4421abaafa89c7802eda92fd5605dfb3979f322d1d9f6022228e1`.

## Final validation

The final combined A/B used the same Mutation Forge code, HEG commit
`b228058`, and score-worker binary for both variants. Only
`prepared_proposal_handoff_enabled` and `score_longest_first_enabled` differed.
Eight runs per variant used the exact 16-episode / 80,000-evaluation workload
in `on, off, off, on` order repeated four times.

| Metric | Optimized | Control | Change |
|---|---:|---:|---:|
| Aggregate runtime | 62.887 s | 81.990 s | — |
| Mean real time | 7.861 s | 10.249 s | -23.3% |
| Mean throughput | 10,178/s | 7,806/s | +30.4% |
| Mean scoring | 3.785 s | 4.480 s | -15.5% |
| Mean proposal generation | 2.486 s | 2.477 s | +0.4% |
| Mean rewrite application | 1.029 s | 2.830 s | -63.6% |
| Mean user time | 6.580 s | 8.979 s | -26.7% |
| Mean system time | 0.840 s | 0.826 s | +1.7% |

All sixteen runs completed without failures or timeouts and shared the exact
canonical summary hash
`e0cc2e9edc409d84b3972da6af19b68dafb5bc68eea903944874678f96a16d94`.
Their complete episode payloads, trajectories, counts, and final/best graph
hashes were identical.

Final deep profile:

- artifact: `runs/stage1-20260729-053058-f97815cf`
- real/user/sys: 9.224 / 7.970 / 0.820 s
- throughput: 8,673 evaluations/s
- scoring: 4.590 s (50.9% of measured episode time)
- proposal generation: 2.971 s (33.0%)
- rewrite application: 1.055 s (11.7%)
- worker round trip: 3.400 s
- worker protocol overhead: 2.222 s
- cycle 16: 1.098 s / 79,960 calls
- cycle 8: 0.080 s / 54,221 calls
- cycle 4: 0.0004 s / 758 calls
- worker failures/restarts/fallbacks: 0 / 0 / 0

Mechanical validation:

- Mutation Forge: 58 tests passed, Ruff passed, mypy passed for 32 files,
  `git diff --check` passed;
- HEG: 344 tests passed, score-worker protocol tests passed, score-worker
  mypy passed for its source file, `git diff --check` passed;
- score worker reports protocol version 2 and the expected final binary SHA.

## Retained commits

- Mutation Forge `bf3e691`: backend-owned prepared-proposal handoff.
- Mutation Forge `292274b`: configurable same-binary cutoff-order control and
  three-seed episode-trajectory parity coverage.
- HEG `50db693`: cutoff-only longest-first cycle evaluation with canonical
  response ordering.
- HEG `b228058`: same-binary cutoff-order ablation bit for literal canonical
  parity testing.

## Changed files

Mutation Forge:

- `src/mutation_forge/backends/heg.py`
- `src/mutation_forge/config.py`
- `src/mutation_forge/evaluation/benchmark.py`
- `configs/stage1-baseline.toml`
- `configs/stage1-smoke.toml`
- `configs/schemas/stage1-config.schema.json`
- `tests/integration/test_benchmark.py`
- `tests/parity/test_heg_backend.py`
- `tests/unit/test_config.py`
- `docs/PROFILING.md`
- this report

HEG:

- `cpp/sglab_score_worker.cpp`
- `src/sglab/score_worker.py`
- `tests/test_score_worker.py`

## Rejected approaches

- Reusable Python request buffer and `pack_into`: micro gain too small.
- Framed Python response read: micro and full-workload gains below gates.
- Contiguous C++ request frame and container reuse: micro and full-workload
  gains below gates.

## Recommended next issue

Profile the remaining score-worker wait/read path with syscall-level evidence
before another protocol rewrite. The final worker spends 2.222 s in estimated
non-cycle protocol overhead, including 1.425 s in wait/read, but the three
low-risk allocation/framing experiments in this sprint failed their retention
gates. A follow-up should first separate C++ compute-to-flush latency from
Python readiness/read latency, then retain a mechanism only if it clears the
same stable micro and full-workload thresholds.
