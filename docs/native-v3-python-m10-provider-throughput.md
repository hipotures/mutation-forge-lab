# Native v3 Python M10 provider throughput

M10 keeps Native v2 as the default and accelerates only the explicitly
selected ordinary-Python scientific-search protocol. The implementation still
requests exactly one complete Python policy per provider turn, validates it
with M1, executes it only inside M2, evaluates it with the accepted M3
semantics, and treats the exact verifier as the sole counterexample authority.

## Concurrent generation protocol

At the start of each generation the coordinator freezes one immutable
generation snapshot. It contains the eight canonical slot IDs and kinds,
exact parent assignments and parent response turns, source-free Search Memory
projection, model and effort, prompt and schema versions, development panel,
policy and evaluation seeds, and all provider and evaluator budgets.

One specification anchor is shared by up to four independent persistent
App Server workers. Fresh roots use worker forks of the anchor and have
`active_parent = null`. Children fork at the exact inclusive parent response
turn and receive the exact parent source plus compact host-derived feedback.
Each worker processes its own turns sequentially; different workers may run
concurrently. Completion order never changes the frozen snapshot or canonical
commit order.

Valid unique sources are persisted immediately and submitted to the bounded
12-worker evaluator pool. The validated-candidate queue target is 24 and its
hard capacity is 48. A generation commits only after all eight provider slots
and every corresponding evaluation are terminal. Selection and lineage use
canonical slot order.

Contract-invalid responses receive at most one same-thread repair, while the
campaign repair budget remains available. A repair receives concise M1
diagnostics and must return a complete source, not a patch. Exhausted repairs
leave their original population slots terminally `contract_invalid`; the host
never rewrites model source and never creates replacement slots.

## Complete-generation budgets

The sustained M10 profile is
`configs/scientific/native-v3-python-m10-v1.toml`:

- 12 complete generations and 96 primary slots;
- at most 24 repair turns and 120 total candidate turns;
- provider concurrency 4 and evaluator workers 12;
- a 10-hour wall limit;
- eight roots in generation zero, then four children and four roots;
- clean-worktree provenance, resume, no terminal replacements, and
  stop-on-VERIFIED.

The coordinator reserves all eight primary turns before starting a generation.
It never starts an incomplete generation. Normal budget and wall termination
therefore occur only between generations.

## Production CLI and observability

Run the sustained profile through the standard application entry point:

```bash
uv run mforge experiment run \
  --config configs/scientific/native-v3-python-m10-v1.toml \
  --dashboard
```

Run without a TTY and return the canonical JSON result:

```bash
uv run mforge experiment run \
  --config configs/scientific/native-v3-python-m10-v1.toml \
  --json
```

Inspect the same workspace without contacting the provider:

```bash
uv run mforge experiment status \
  --config configs/scientific/native-v3-python-m10-v1.toml \
  --json
```

Request a durable stop:

```bash
uv run mforge experiment stop \
  --config configs/scientific/native-v3-python-m10-v1.toml \
  --final
```

Resume with the original `experiment run` command and unchanged configuration.
Terminal slots are verified and skipped. Rich and JSON both consume
`python_preview_status`; neither owns scheduler counters or scientific logic.

The canonical status exposes slot taxonomy, provider turns and exact token
usage, configured/active/peak provider concurrency and its timeline,
evaluator activity and queue depth, valid programs per provider minute,
provider-wait share, policy and score rates, NoPlan and illegal-final-state
rates, selector/action frequencies, best fitness and lineage, worker
rotations/failures, exact-verifier activity, recovery, and the dominant
bottleneck.

## Frozen live profiles

Two smaller versioned profiles support the required live acceptance:

- `native-v3-python-m10-dashboard-acceptance-v4.toml` exercises Rich,
  durable stop/resume, no-repeat, and JSON parity;
- `native-v3-python-m10-benchmark-v2.toml` runs one eight-slot generation for
  the quick throughput benchmark.

They are explicit ordinary-Python configurations and do not alter
`experiment.toml` or Native v2 routing.
