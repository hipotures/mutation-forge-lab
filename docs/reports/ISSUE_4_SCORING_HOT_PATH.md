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

Pending evidence from the first two experiments.

## Experiment 4: direct cutoff arithmetic

Pending evidence from the first two experiments.

## Final validation

Pending.
