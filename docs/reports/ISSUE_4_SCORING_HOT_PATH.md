# Issue #4: remaining HEG scoring hot path

GitHub issue:
[hipotures/mutation-forge-lab#4](https://github.com/hipotures/mutation-forge-lab/issues/4)

## Starting state

- Mutation Forge: `304e56ed5c53b6f008d9f8a8a54dc1dc0583f9d7`
- HEG: `b228058463ac6c3d111ea983c346e2dcb07b5ee1`
- score-worker protocol: 2
- score-worker SHA-256:
  `23668212f1a4421abaafa89c7802eda92fd5605dfb3979f322d1d9f6022228e1`
- both repositories were clean and synchronized with `origin/main`
- working branches: `agent/issue-4-scoring-hot-path`

Reference production-profile artifacts:

- non-deep:
  `runs/stage1-20260729-094032-dbc79392`
- deep:
  `runs/stage1-20260729-093503-4306424b`
- workload: order 30, graph seeds 101–104, policy seeds 1–2,
  16 episodes, 50,000 evaluations per episode, 800,000 evaluations total
- canonical summary hash:
  `d10dc72aa34ed478485d722e7ab992e024bb9ac00c0a50c13347269bc577de3c`

Reference non-deep result:

| Metric | Value |
|---|---:|
| Real | 76.327 s |
| Throughput | 10,481 evaluations/s |
| Scoring | 37.418 s |
| Proposal generation | 24.910 s |
| Rewrite application | 10.037 s |

## Retention gates

Cutoff order:

- at least 5% lower scoring wall; or
- at least 2% higher total throughput;
- balanced multi-seed A/B and exact canonical/episode parity.

Compact dominated response:

- at least 10% lower dominated-response protocol time in a stable
  microbenchmark; or
- at least 3% higher full-workload throughput;
- detailed deep-profile mode and full responses remain unchanged.

Prepared request plan and direct cutoff arithmetic:

- retain only a measured gain that clears the issue-specific targeted or
  total-throughput threshold.

## Experiment 1: six cutoff evaluation orders

Implementation under test:

- canonical request lengths remain increasing;
- one protocol-v2 binary accepts an explicit request-level permutation code;
- the legacy ascending/longest-first bit remains backward compatible;
- successful responses remain canonical and complete;
- detailed dominated responses contain the canonicalized subset actually
  evaluated.

Targeted validation before measurement:

- HEG score-worker tests: 22 passed;
- Mutation Forge config/parity/integration tests: 50 passed;
- all six permutations × three policy seeds preserved timing-free episode
  payloads exactly;
- HEG targeted mypy, Mutation Forge Ruff/mypy, and both diff checks passed.

The balanced worker microbenchmark used one protocol-v2 process and six
rotated/reversed 10-second blocks per order, for at least 60 seconds per
variant:

| Order | Requests/s | vs. `16-8-4` | Records/request |
|---|---:|---:|---:|
| `16-4-8` | 28,484 | +0.5% | 1.000 |
| `16-8-4` | 28,330 | — | 1.000 |
| `4-16-8` | 27,465 | -3.1% | 2.000 |
| `8-16-4` | 23,474 | -17.1% | 2.000 |
| `8-4-16` | 22,711 | -19.8% | 3.000 |
| `4-8-16` | 22,549 | -20.4% | 3.000 |

The production non-deep workload ran once per order with the exact reference
800,000-evaluation configuration:

| Order | Evaluations/s | vs. current | Scoring | Scoring change |
|---|---:|---:|---:|---:|
| `16-8-4` | 10,512 | — | 38.223 s | — |
| `16-4-8` | 10,410 | -1.0% | 38.931 s | +1.9% |
| `4-16-8` | 10,391 | -1.1% | 39.144 s | +2.4% |
| `8-16-4` | 9,787 | -6.9% | 43.470 s | +13.7% |
| `8-4-16` | 9,648 | -8.2% | 44.842 s | +17.3% |
| `4-8-16` | 9,619 | -8.5% | 44.961 s | +17.6% |

Non-deep artifacts:

- `16-8-4`: `runs/stage1-20260729-101840-82b98423`
- `8-16-4`: `runs/stage1-20260729-101956-a2499bc4`
- `4-8-16`: `runs/stage1-20260729-102118-ec7b3448`
- `16-4-8`: `runs/stage1-20260729-102242-79cb56cb`
- `8-4-16`: `runs/stage1-20260729-102359-af90795d`
- `4-16-8`: `runs/stage1-20260729-102522-1301f934`

The deep-profile run for every order produced the same 689 full and 799,327
dominated worker results, 498 score-cache hits, and no worker failures,
restarts, or fallbacks:

| Order | Worker round trip | Protocol overhead | Records/response |
|---|---:|---:|---:|
| `16-8-4` | 34.160 s | 22.365 s | 1.578 |
| `16-4-8` | 34.401 s | 22.665 s | 1.915 |
| `4-16-8` | 35.172 s | 23.058 s | 2.341 |
| `8-16-4` | 39.098 s | 22.893 s | 2.004 |
| `4-8-16` | 40.136 s | 23.165 s | 3.000 |
| `8-4-16` | 40.517 s | 23.468 s | 3.000 |

Per-length deep evidence is reported as `calls / seconds / nodes`:

| Order | Cycle 4 | Cycle 8 | Cycle 16 |
|---|---:|---:|---:|
| `16-8-4` | 3,381 / 0.001 / 158,751 | 458,937 / 0.519 / 121,343,144 | 799,518 / 11.275 / 2,238,947,866 |
| `16-4-8` | 458,937 / 0.341 / 80,395,349 | 272,488 / 0.242 / 56,452,330 | 799,518 / 11.154 / 2,238,947,866 |
| `4-16-8` | 799,518 / 0.842 / 192,152,776 | 272,488 / 0.242 / 56,452,330 | 799,518 / 11.030 / 2,224,565,154 |
| `8-16-4` | 3,381 / 0.001 / 158,751 | 799,518 / 5.923 / 1,310,226,550 | 799,518 / 10.281 / 2,075,922,522 |
| `4-8-16` | 799,518 / 0.839 / 192,152,776 | 799,518 / 5.896 / 1,310,226,550 | 799,518 / 10.237 / 2,061,314,057 |
| `8-4-16` | 799,518 / 0.813 / 192,152,776 | 799,518 / 5.965 / 1,310,226,550 | 799,518 / 10.270 / 2,061,314,057 |

Deep artifacts:

- `16-8-4`: `runs/stage1-20260729-103031-aeea218e`
- `8-16-4`: `runs/stage1-20260729-103202-f6dfbf00`
- `4-8-16`: `runs/stage1-20260729-103339-5ec32af9`
- `16-4-8`: `runs/stage1-20260729-103518-d9e3f510`
- `8-4-16`: `runs/stage1-20260729-103649-af5324fc`
- `4-16-8`: `runs/stage1-20260729-103828-a000a8ca`

Every micro and workload run used score-worker SHA-256
`01f143fed20a61579cbd1475dbe347ce9e5af3791bdcd7143bea813f00070591`.
All twelve workloads shared canonical summary hash
`d10dc72aa34ed478485d722e7ab992e024bb9ac00c0a50c13347269bc577de3c`
and timing-free episode-payload SHA-256
`fbc4624f7ebe008271fd5d961a5df087db26c78a181594fac5cb521b6b083fde`.

**NO-GO.** The current `16-8-4` order remained fastest. No alternative
reached either retention gate. The request-level permutation implementation
and its production configuration were removed before the next experiment.

## Experiment 2: compact dominated responses

The retained protocol-v2 request flag asks the C++ worker to return a
header-only response when a cutoff proves domination. It does not alter full
responses. Mutation Forge enables it only for non-deep requests; deep
profiling always retains detailed per-cycle records.

The balanced microbenchmark alternated six 10-second detailed/compact blocks,
giving at least 60 seconds per mode on one worker process:

| Metric | Detailed | Compact | Change |
|---|---:|---:|---:|
| Requests/s | 27,202 | 29,715 | +9.2% |
| Records/request | 1.000 | 0.000 | -100.0% |
| Response read | 25.300 µs | 23.179 µs | -8.4% |
| Response parsing | 1.785 µs | 0.918 µs | -48.5% |
| Read + parsing | 27.085 µs | 24.097 µs | -11.0% |
| Worker round trip | 32.813 µs | 29.739 µs | -9.4% |

This clears the 10% targeted dominated-response protocol gate.

The full-workload A/B used the sequence compact, detailed, detailed, compact.
Every run used policy seeds 1–3, 24 episodes, and 1,200,000 evaluations:

| Mode | Mean real | Mean evaluations/s | Mean scoring |
|---|---:|---:|---:|
| Detailed | 114.910 s | 10,480 | 57.554 s |
| Compact | 109.505 s | 10,998 | 52.838 s |
| Change | -4.7% | +4.9% | -8.2% |

Paired improvements were stable:

- compact run 1 versus detailed run 2: +5.0% throughput and
  -4.775 s scoring;
- compact run 4 versus detailed run 3: +4.9% throughput and
  -4.658 s scoring.

Artifacts:

- compact: `runs/stage1-20260729-105437-64e4ff6f`
- detailed: `runs/stage1-20260729-105626-982ad1bd`
- detailed: `runs/stage1-20260729-105822-1a07776e`
- compact: `runs/stage1-20260729-110016-1f59945f`

All four runs shared canonical summary hash
`27a0c05993b85761e8d26c72c326356e15d8796a7a334a69afbb059d82151591`
and timing-free episode-payload SHA-256
`914d1daa7b4976d7b4e796cec532cce60646bbc168389604f4bd0e915fbefeb0`.
Final/best graphs and all logical counters matched; worker failures, restarts,
fallbacks, timeouts, and crashes were zero. The retained experimental binary
SHA-256 is
`4a1f927f0f2c7e2d6c8111ef832b7960a382c2613d3eec6ce36f978c4d982e60`.

**GO.** Compact dominated responses cleared both the targeted microbenchmark
gate and the alternate full-workload throughput gate.

## Experiment 3: prepared request plan

The retained Python worker keeps one prepared immutable request plan keyed by
graph order, cycle lengths, limit, node budget, cutoff mode, inclusive mode,
compact mode, and legacy ordering mode. It caches only validation results,
word count, packed length bytes, payload size, and the fixed frame prefix.
Graph rows, request IDs, cutoff values, and flags remain dynamic. A mismatch
replaces the single entry.

The balanced 60-second-per-mode microbenchmark produced:

| Metric | Uncached | Cached | Change |
|---|---:|---:|---:|
| Requests/s | 29,300 | 30,795 | +5.1% |
| Request packing | 2.684 µs | 2.654 µs | -1.1% |
| Worker round trip | 29.517 µs | 29.490 µs | -0.1% |

Validation and plan lookup occur outside the existing request-packing timer,
so the targeted subphase does not capture most of the avoided Python work.
The full-workload gate was therefore required.

The balanced cached, uncached, uncached, cached A/B used three policy seeds,
24 episodes, and 1,200,000 evaluations per run:

| Mode | Mean real | Mean evaluations/s | Mean scoring |
|---|---:|---:|---:|
| Uncached | 112.085 s | 10,745 | 55.231 s |
| Cached | 108.270 s | 11,125 | 51.882 s |
| Change | -3.4% | +3.5% | -6.1% |

Paired throughput gains were +3.58% and +3.49%. Artifacts:

- cached: `runs/stage1-20260729-111645-953ce55a`
- uncached: `runs/stage1-20260729-111833-8ee12597`
- uncached: `runs/stage1-20260729-112025-5ae617ce`
- cached: `runs/stage1-20260729-112218-f5f4895d`

All four runs shared canonical summary hash
`ee85dab2eec4fcf08e554ef8cec26ea003769c98e100df9d8b9bab6ff69aee5e`
and timing-free episode-payload SHA-256
`914d1daa7b4976d7b4e796cec532cce60646bbc168389604f4bd0e915fbefeb0`.
No failures, timeouts, crashes, restarts, or fallbacks occurred.

**GO.** The one-entry request plan cleared the 2% full-workload throughput
gate with exact parity.

## Experiment 4: direct cutoff arithmetic

The candidate replaced the bounded `1..limit` scan with the exact count at
which partial total first equals the cutoff total, followed by at most one
additional count. A same-binary flag selected the original or candidate path.

Boundary tests covered inclusive and exclusive cutoffs, equal total,
weighted-penalty and simplicity boundaries, the witness cap, and full-result
cutoffs. The candidate also preserved three deterministic Mutation Forge
episode trajectories.

The balanced microbenchmark used eight alternating 10-second blocks per mode
(80 seconds per path) with compact dominated responses:

| Metric | Scanned | Direct | Direct change |
|---|---:|---:|---:|
| Requests | 2,506,774 | 2,497,802 | -0.36% |
| Requests/s | 31,335 | 31,222 | -0.36% |
| Worker round trip | 28.914 µs | 29.023 µs | +0.38% |

The experimental worker SHA-256 was
`3b77f446029c899f15b74f62abeae2bc3cbf5e528fc8212e97bf5a49bc043852`.
All response assertions passed.

**NO-GO.** The direct calculation was slightly slower, so the full-workload
A/B was not justified. The flag, implementation, configuration, and tests
were reverted completely.

## Final validation

Retained implementation commits:

- HEG `fab20312f5951235c977fc604be46e962b29e90`:
  compact dominated responses;
- Mutation Forge `670f1a9d49b347fb0415e5933b10808e3c9e20c6`:
  compact-response integration;
- HEG `fd97451b0f3d87400d1d955a2c6b1b18303344ff`:
  one-entry prepared request plan;
- Mutation Forge `9b3dee7`:
  prepared-plan configuration and integration.

The final worker was rebuilt from retained source at
`/home/user/DEV/heg/_build/sglab-score-worker`; SHA-256:
`4a1f927f0f2c7e2d6c8111ef832b7960a382c2613d3eec6ce36f978c4d982e60`.

The final 800,000-evaluation deep profile is:

`runs/stage1-20260729-114052-91ed871a`

| Metric | Final value |
|---|---:|
| Real | 88.894 s |
| Throughput | 8,999 evaluations/s |
| Scoring | 45.135 s |
| Proposal generation | 29.311 s |
| Rewrite application | 10.021 s |
| Duplicate detection | 1.942 s |
| Progress reporting | 0.624 s |
| Worker round trip | 33.991 s |
| Protocol overhead | 22.032 s |
| Cycle 16 | 11.432 s |
| Cycle 8 | 0.526 s |
| Cycle 4 | 0.001 s |

Deep profiling deliberately disables compact dominated responses so all
per-cycle records remain available. It reported 689 full scores, 799,327
dominated scores, 498 score-cache hits, and zero worker failures, restarts, or
fallbacks.

The run's canonical summary hash differs from the starting reference because
`dataset_manifest_hash` records the changed HEG revision. All 16 normalized
timing/profile-free episode payloads are byte-for-byte identical to the
reference deep run; no semantic episode field differs.

HEG final validation passed all 346 unit tests and `make check`. The test
runner emitted non-failing resource warnings at shutdown about three leaked
multiprocessing semaphores and unclosed file descriptors.

Mutation Forge final validation passed:

- `uv run pytest`: 67 passed;
- Ruff: all checks passed;
- mypy: no issues in 32 files;
- `git diff --check`;
- `mforge doctor`: every check passed against clean HEG commit
  `fd97451b0f3d87400d1d955a2c6b1b18303344ff`.

Final smoke artifact:

`runs/stage1-20260729-114548-478c9b35`

It completed 8 episodes and 8,000 evaluations at 9,104 evaluations/s with
score 83 → 64, 500 legal proposals, no invalid proposals, and no timeouts or
crashes.
