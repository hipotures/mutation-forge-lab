# Architecture

Mutation Forge keeps policy authority narrow and host authority explicit.

```text
immutable dataset
      |
fixed episode controller
      |
host-validated proposal pool ---- policy/ranker (Stage 2A worker)
      |
declarative RewritePlan
      |
host invariant validation
      |
read-only graph backend -> scorer -> exact verifier for heuristic zero only
      |
events + SQLite + immutable artifacts
```

## Stage 1 components

- `config.py` parses the versioned TOML contract and rejects inactive or
  unsupported Stage 1 behavior.
- `models.py` defines immutable graph, score, validation, verification,
  rewrite, dataset, and episode records.
- `backends/base.py` defines the stable typed backend protocol.
- `backends/toy.py` supplies a deterministic isolated test backend.
- `backends/heg.py` imports the sibling HEG source tree without installing or
  modifying it. It composes HEG's persistent C++ score worker with
  `score_from_cycle_counts` and delegates exact Python verification. If the
  checkout contains a stale or incompatible untracked worker binary, the
  adapter falls back to HEG's bounded Python witness enumerator and records the
  selected score implementation rather than modifying HEG.
- `proposals/two_switch.py` exposes only the two reviewed HEG proposal
  families.
- `evaluation/episode.py` owns the fixed controller and all graph state.
- `evaluation/benchmark.py` owns budgets, events, provenance, summaries, and
  terminal status.
- `events.py`, `run_store.py`, and `artifacts.py` provide durable evidence.
- `output/` renders the same event stream as Rich or JSON Lines.

## Authority and invariants

`RewritePlan` is declarative. Before scoring, the host requires existing
removed edges; no loops, duplicates, or pre-existing additions; unchanged
order; a simple undirected connected result; degree three at every vertex; and
at most a two-edge switch in Stage 1. Policy code never owns a graph, scorer,
verifier, filesystem, database, process, or network.

The HEG backend is read-only and its exact commit and dirty state are recorded.
Its graph is converted through immutable edge tuples. Canonical hashing uses
HEG's canonical graph6 key for immutable datasets and final result identities;
serialization remains ordinary graph6. The per-evaluation duplicate and
exact-submission sets use HEG's label-sensitive graph6 `stable_hash`, avoiding
an external nauty `labelg` process in the hot loop. Stage 1 rewrites do not
relabel vertices, so labeled equality is sufficient for these transient sets.

## Later package boundaries

`sandbox` implements the Stage 2A validator, isolated worker, behavior
signature, replay, and artifacts. Stage 2B reuses it for frozen plain-data
scientific ranker payloads. `stage2b` generates bounded legal `k`-switch pools,
computes host-owned aggregate features, presents identical immutable pools to
paired rankers, and authoritatively scores selected plans only. It is not
connected to the Stage 1 episode controller or integrated into HEG. `llm`,
`archive`, and `evolution` remain inactive package boundaries; no model or
evolutionary path exists.
