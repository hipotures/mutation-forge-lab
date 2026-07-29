# Four-hour optimization sprint

GitHub issue: [#2](https://github.com/hipotures/mutation-forge-lab/issues/2)

## Sprint record

- Started: 2026-07-29 05:16:38 BST
- Time box ends: 2026-07-29 09:16:38 BST
- Mutation Forge base: `2d80589d308c68b219208d8b3c95d382e48d3c99`
- HEG base: `c691dbcec249afc602366cee092ad177303b4dbc`
- Score worker: `/home/user/DEV/heg/_build/sglab-score-worker`
- Score-worker SHA-256: `dcc040ddbe92c235a13134b7746707e59ec68d84c9be693a56672df9a0193df7`
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

Pending.

### Optional experiment

Pending.

## Final validation

Pending.

## Retained commits

Pending.

## Rejected approaches

Pending.

## Recommended next issue

Pending.
