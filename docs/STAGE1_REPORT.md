# Stage 1 report

Date: 2026-07-28
Status: **GO for Stage 1 completion; ready for Stage 2 review**

## Scope delivered

Stage 1 provides the typed experiment harness, deterministic immutable
connected-cubic dataset builder, toy backend, read-only HEG adapter, uniform
two-switch and forbidden-cycle-break HEG baselines, fixed episode controller,
fitness collection, Rich and JSONL event output, SQLite run metadata, complete
run artifacts, doctor/inspect/compare commands, tests, and architecture
documentation.

Generated-code execution, k-switch generation, sandbox workers, model calls,
Codex App Server transport, evolutionary program search, held-out evaluation,
and HEG production integration are not implemented.

## Environment and provenance

- Python: 3.12.13
- Mutation Forge development base commit: `8fa4d8947b4ae6ed2937ebe72079cc81244c5cfa`
  (the validated implementation is committed after this report)
- HEG commit: `b2a011f5c1060d2414d83c1804f5a16ae7d853c7`
- HEG worktree before and after all checks: clean
- Dataset manifest:
  `runs/datasets/037cdb74b7071a43df1b6aef107d879c02c64f0e060f00a562a20f5d42542ad9.json`
- Dataset: connected cubic, order 30, graph seeds 101–104, policy seeds 1–2,
  smoke split

The HEG checkout's untracked `_build/sglab-score-worker` predates the current
protocol-v2 source and exits on a v2 request. Mutation Forge did not rebuild or
modify HEG. The adapter detected the failure and used HEG's authoritative
bounded Python witness enumerator, followed by the same
`score_from_cycle_counts` implementation. Both runs record
`heg-python-bounded-reference` as their score implementation.

## Commands validated

```console
uv sync
uv run mforge doctor --heg-repo ../heg
uv run mforge dataset build --config configs/stage1-smoke.toml --json
uv run pytest
uv run ruff check .
uv run mypy
uv run mforge baseline run --config configs/stage1-smoke.toml
uv run mforge baseline run --config configs/stage1-smoke.toml --json
uv run mforge inspect runs/stage1-20260728-214254-903ef7d1
uv run mforge inspect runs/stage1-20260728-214335-38266f13 --json
uv run mforge compare \
  runs/stage1-20260728-214254-903ef7d1 \
  runs/stage1-20260728-214335-38266f13 --json
```

Validation result: 15 tests pass with zero skips, including mandatory HEG
parity and immutability checks. Ruff and strict mypy pass.

## Smoke benchmark

Both output modes completed 16 episodes and 8,000 graph evaluations: two
baselines × four graphs × two policy seeds × 500 evaluations. Every episode
completed its budget. There were no timeouts, score failures, invalid
proposals, or no-ops. Every final graph passed HEG validation and its canonical
hash matched the recorded value.

| Output | Run | Seconds | Evaluations/s | Summary hash |
|---|---|---:|---:|---|
| Rich | `stage1-20260728-214254-903ef7d1` | 38.196 | 209.444 | `9df68eb76ba678fd64684ba2dcb13b34e6c67482764f0029535522212837d770` |
| JSONL | `stage1-20260728-214335-38266f13` | 37.546 | 213.072 | `9df68eb76ba678fd64684ba2dcb13b34e6c67482764f0029535522212837d770` |

The identical canonical summary hash confirms deterministic trajectories and
Rich/JSON parity after excluding run IDs, timestamps, and timing measurements.
The JSON run emitted 215 valid JSON objects, no ANSI or prose, and ended with
`run_completed`.

Selected median normalized fitness metrics:

| Baseline | Best witnesses | Best weighted penalty | Best-so-far AUC | Failure/timeout |
|---|---:|---:|---:|---:|
| `heg_uniform_two_switch` | 0.8125000 | 0.6833726 | 0.8295645 | 0 / 0 |
| `heg_forbidden_cycle_break` | 0.7941176 | 0.6447124 | 0.8031287 | 0 / 0 |

These are infrastructure baselines, not a claim of statistical superiority.
Stage 1 has no generated policy and no held-out experiment.

## Artifact and recovery checks

Each run contains `run_config.toml`, `run_manifest.json`, `events.jsonl`,
`run_summary.json`, `environment.json`, `dataset_manifest.json`,
`archive.sqlite3`, and the four required artifact directories. SQLite
`PRAGMA integrity_check` returns `ok`, and the run row is completed. Tests also
force an immediate wall-time interruption and confirm a readable failed
summary, failed manifest, SQLite archive, and terminal `run_failed` event.

Representative event types are `run_started`, `backend_ready`,
`dataset_loaded`, `checkpoint_written`, `baseline_started`,
`episode_started`, `episode_progress`, `episode_completed`, and
`run_completed`. Every event includes `schema_version`, `timestamp`, `run_id`,
and `event_type`.

## Limitations

- The smoke set has four graphs at one order and is intentionally too small for
  scientific conclusions.
- The local HEG C++ score-worker binary is stale; reference Python scoring is
  semantically aligned but slower than a current worker build.
- Stage 1 uses fixed HEG mutation operators and a strict-improvement controller;
  it does not yet expose a proposal pool to a ranker.
- Canonical labeling depends on HEG's nauty integration; HEG documents a
  non-authoritative raw-graph6 fallback when nauty is absent.
- Exact verification is invoked only for heuristic-zero submissions; this
  smoke produced no verified counterexample claim.

## Exact Stage 2 prerequisites

Stage 2 may start only after user review of this report. It must use no model
tokens. It must add legal k-switch pools for `2 <= k <= 4`, bounded immutable
proposal features, the single ranker template, AST allowlisting and normalized
hashes, an isolated worker with CPU/memory/wall/payload/output limits,
fixture/random/structural rankers, behavior probes, penalties, and validation
and evaluation CLIs.

The adversarial suite must cover import, file, environment, subprocess,
network, dunder/reflection, infinite-loop, large allocation/output, recursion,
NaN/infinity, exception, wrong-signature, multiple-function, and hidden-state
cases. Exit requires rejection before execution where applicable, bounded
termination and memory, coordinator isolation, invariant preservation, 10,000
valid calls, Rich/JSON equivalence, stable source/AST/behavior identities, and
a structural ranker beating random on a frozen toy benchmark.
`STAGE2_REPORT.md` must make the explicit GO/NO-GO decision for live LLM use.

Recommendation: accept Stage 1 and review the Stage 2 safety design. Do not
begin Stage 3 model integration or any HEG production work.

## Post-Stage-1 hot-loop optimization

After rebuilding the HEG protocol-v2 C++ score worker, profiling a longer
80,000-evaluation run showed that per-candidate canonical hashing launched
nauty `labelg` approximately once per evaluation. Duplicate and exact-zero
submission sets now use HEG's label-sensitive stable graph6 hash. Canonical
hashes remain in immutable dataset entries and final episode results.

Using the identical stored configuration and C++ scorer:

| Implementation | Seconds | Evaluations/s |
|---|---:|---:|
| Per-evaluation canonical hash | 318.166 | 251.441 |
| Fast labeled state hash | 40.286 | 1985.822 |

This is a 7.90× end-to-end speedup. Across all 16 episodes, initial, best, and
final scores; every 5,000-point best-so-far curve; and final canonical graph
hashes matched exactly. The duplicate count decreased from 53,450 to 48,466
because it now measures labeled state identity rather than isomorphism. The
duplicate metric does not affect controller decisions.

## Aggregate runtime profiling

The harness now supports HEG-style aggregate phase timers controlled by
`search.profiling_enabled`. Per-episode and run-level summaries report scoring,
proposal generation, rewrite application, duplicate detection, controller,
exact verification, progress reporting, and finalization time, together with
accounted and unattributed totals. The profile emits no per-evaluation events
and is excluded from the canonical deterministic summary hash. Usage and the
required on/off comparison are documented in `docs/PROFILING.md`.

A balanced four-run smoke check after adding proposal child timers measured
3.318 seconds with profiling and 3.301 seconds without it, or 0.49% observed
overhead. All four runs produced the same canonical summary hash.
