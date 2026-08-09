# Native v3 Python M9 scientific search

M9 is the first search milestone after the ordinary-Python migration. It does
not change the Native v2 default. The sustained search remains an explicit
`native-v3-python-v1` preview selected by a separate configuration file.

## Scientific profile

The versioned profile is
`configs/scientific/native-v3-python-m9-v1.toml`. It freezes:

- the accepted two-case order-30 development panel;
- generation 0 as eight fresh roots;
- every later generation as four exact parent forks and four fresh roots;
- eight generations, for at most 64 planned slots;
- one serial provider and 12 bounded candidate evaluators;
- an eight-hour active wall-clock limit;
- at most 64 submitted provider program turns, including repairs;
- no replacement of invalid, duplicate, or provider-failed terminal slots;
- a resumable infrastructure stop after three consecutive provider-failed
  slots, so a persistent external outage cannot consume the whole campaign;
- clean-worktree provenance, durable resume, and stop-on-VERIFIED behavior.

If a completed generation has no valid evaluated parent, the accepted
fail-closed lineage rule records `all_root_fallback` and allocates eight fresh
roots in the next generation rather than inventing a child relationship. It
does not retry or replace terminal slots. A sustained-campaign acceptance run
must still demonstrate the ordinary four-child/four-root later-generation
allocation whenever eligible parents exist.

The profile resolves its workspace to
`/home/user/DEV/mutation-forge-lab-evidence/m9/sustained` in the standard
checkout layout. It is not `experiment.toml` and does not affect ordinary
`mforge experiment run`.

## Execution architecture

The provider remains serial because its durable thread activation, exact
parent forks, logger, and state file have one owner. A valid source is persisted
as a prepared candidate and immediately queued for evaluation. The provider
can generate the next slot while evaluation proceeds.

Each evaluator thread lazily owns a private HEG backend and scientific
evaluator. It evaluates the complete immutable panel for one candidate. No
backend or policy sandbox is shared between evaluator threads. The coordinator
is the only writer of evaluation and terminal candidate artifacts and commits
results in canonical slot order. This preserves deterministic duplicate
classification, selection, lineage, reports, and resume even when evaluations
finish out of order.

Generated source still executes only in the M2 bubblewrap/seccomp/rlimit
worker. Coordinator, provider, and evaluator-host processes validate and move
source bytes but do not execute them.

## Relation-aware safe graph API

The safe API adds exactly three selectors:

```python
api.matching_k_switch_reconnections_for_edge(edge, k)
api.edge_fanouts_legal_for_edge(edge)
api.relocations_legal_for_edge(edge)
```

The argument is an invocation-scoped opaque `EdgeRef`. Returned values are
bounded opaque references in deterministic order. A returned operation must
contain the supplied edge as its source edge. A reference that is foreign,
wrong-kind, or no longer present in the current overlay fails closed. No raw
labels, adjacency, or graph structure are exposed.

## Commands

Start or resume the sustained profile:

```text
uv run mforge experiment run \
  --config configs/scientific/native-v3-python-m9-v1.toml \
  --json
```

Run with the small read-only live dashboard in a TTY:

```text
uv run mforge experiment run \
  --config configs/scientific/native-v3-python-m9-v1.toml \
  --dashboard
```

Inspect the same metrics without starting provider or evaluator work:

```text
uv run mforge experiment status \
  --config configs/scientific/native-v3-python-m9-v1.toml \
  --json
```

Request a resumable stop at the next durable candidate boundary:

```text
uv run python scripts/native_v3_python_m6_live_preview.py \
  --request-stop \
  --config configs/scientific/native-v3-python-m9-v1.toml
```

Resume with the original run command and unchanged configuration. Terminal
candidate files are verified and skipped. Prepared, nonterminal candidates are
evaluated without repeating their provider turns.

## Live status

Status reports slot and failure counts, exact provider usage, active/idle/peak
evaluators, queue depth and queue wait, evaluator utilization, policy and score
throughput, accepted rewrites, NoPlan and illegal-final rates, selector/action
frequencies, worker failures and rotations, phase timings, the dominant
bottleneck, best fitness/identity/lineage, time since improvement, and exact
verifier activity.

The exact verifier is the only authority that can set
`scientific_success=true`. A heuristic zero remains a submission. It is never
reported as a counterexample by the search coordinator.

## Durable artifacts

The experiment root contains:

- `acceptance-provenance.json.gz`, `protocol.json.gz`, and `anchor.json.gz`;
- `m9-runtime.json.gz`, `m9-stop.json.gz`, and `m9-report.json.gz`;
- immutable generation manifests and Search Memory;
- per-slot provider artifacts, prepared-candidate state, terminal candidate
  records, evaluation traces, and exact generated `.py` source;
- sandbox telemetry, score evidence, fitness intervals, and counterexample
  verification records.

Large runtime trees remain outside Git. Human-readable reports and small
canonical manifests may be committed after the campaign.

## Baseline profile

The pre-M9 10.68-minute profile spent approximately 61% of wall time inside
retained provider turns and only about 0.05% in evaluator work. It evaluated
four programs, consumed three contract-invalid slots, and ended on the old
10,000-event App Server cap at the eighth root. M9 therefore prioritizes:

1. a bounded transport envelope sized for 64 program turns;
2. immediate evaluation of valid programs while provider work continues;
3. independent evaluator ownership and canonical single-writer commits;
4. direct edge-scoped operators that make witness-guided policies structural
   rather than merely suggestive.
