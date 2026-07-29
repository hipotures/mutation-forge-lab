# Stage 2B report

Date: 2026-07-29

Status: **implemented and validated; scientific gate not cleared**

## Entry point and scope

Stage 2B started from:

- Mutation Forge `e2d11bb86b4fa5dbc7ebfb441923e0f02e9799a9`;
- HEG `fd97451b0f3d87400d1d955a2c6b1b18303344ff`.

Issue #5 was closed as completed with `GO_TO_STAGE_2B` before work began. Work
used branch `agent/stage2b-issue-6`. HEG remained at the frozen commit with a
clean worktree and was never modified.

No model, App Server, or research network call occurred. Stage 3, evolution,
a full proposer, held-out evaluation, and HEG policy integration were not
started.

## Implementation and safety boundary

The host now generates deterministic legal `k`-switch pools for `k = 2, 3, 4`.
It selects pairwise vertex-disjoint existing edges, enumerates at most 105
perfect matchings, and rejects the original pairing, partial reuse of removed
edges, loops, duplicate/pre-existing edges, duplicate plans, disconnected
graphs, and graph-family violations. The backend applies and validates every
retained declarative `RewritePlan` before it becomes ranker-visible.

The six reviewed bounded selectors are uniform random, sampled-forbidden-cycle
anchored, high sampled witness-load, remote from an anchor, pairwise
distant/disjoint, and mixed exploit/explore. Fixed seeds control every
selection, matching order, and tie-break.

Stage 2B freezes:

- `stage2b.context.v1`;
- `stage2b.proposal.v1`;
- `stage2b.features.v1`;
- `stage2b.pool.v1`;
- `stage2b.behavior.v1`;
- `stage2b.artifact.v1`;
- configuration contract `stage2b.1`.

The exact context and proposal fields are those in
`configs/schemas/stage2b-context.schema.json` and
`configs/schemas/stage2b-proposal.schema.json`. Runtime validation requires
exact keys, aligned feature vectors, finite bounded values, reviewed selector
names, matching `k`/operator family, and an opaque SHA-256 proposal ID. No
absolute vertex identifier, raw graph, backend, scorer, verifier, controller,
cache, RNG, or experiment state is exposed.

Both manual rankers execute only through the accepted Stage 2A worker. The
random ranker obtains a deterministic pseudo-random order from opaque proposal
hashes. The structural ranker uses only the frozen witness-load, broken
witness, distance, triangle/C4-risk, and reconnection fields. Priority ties
break by proposal ID.

## Bounds and accounting

The preregistered defaults are:

- pool size 12;
- `k_values = [2, 3, 4]`;
- 96 selection retries and 105 matchings per selected stub set;
- 32 sampled witnesses per forbidden length;
- 20,000 cycle-enumeration nodes;
- 256 distance queries with a bounded per-pool cache;
- 2,048 local-risk operations;
- eight toy steps, witness cap 64;
- Stage 2A limits: 12 KiB source, 500 AST nodes, 256 static loop bound,
  128 MiB address space, 25 ms per call, 60 seconds total, 64 KiB request,
  16 KiB response, and 64 KiB captured output.

Proposal legality, feature work, ranker latency, and authoritative scoring are
recorded separately. Attempted, rejected, deduplicated, retained, `k`, selector,
feature-use, budget-exhaustion, and selected-baseline labels are persisted.
Only the random and structural policies' selected plans are authoritatively
scored. There is no best-of-pool score oracle and no exact-verifier call.

Aggregate toy timing was 35,530,574 ns for proposal legality, 539,226,756 ns
for feature work, 1,075,455,703 ns for ranker calls, and 6,327,675 ns for
selected-plan scoring. The corresponding single HEG pilot-run totals were
1,012,420 ns, 13,596,274 ns, 10,343,930 ns, and 453,140 ns.

Across the final toy run, the host retained 3,072 proposals after 18,289
matching attempts and 1,568 deduplications. Rejections were 1,100 disjoint
selection failures, 5,558 reused original edges, 1,847 original pairings,
6,220 pre-existing edges, and 24 host-validation failures. All retained plans
were valid. Coverage was:

- `k=2`: 992; `k=3`: 1,511; `k=4`: 569;
- high-load: 289; mixed: 805; distant/disjoint: 226; remote: 240;
  witness-anchored: 748; uniform: 764.

Per-pool maxima were 3,648 cycle nodes, 30 distance queries, 156 local-risk
operations, 29 sampled witnesses, and 597 distance-cache hits. No configured
feature budget was exhausted.

## Fairness, signatures, and artifacts

At each toy and HEG pilot step, one immutable ordered pool was presented to
both rankers. All 32 toy seeds recorded `same_pool_proof=true`. Each seed made
one initial score call plus two selected-plan calls per step: 17 score calls,
not 97 calls for a 12-proposal pool. Random and structural workers each
completed 3,072 toy calls with zero failures.

The final durable run is:

`runs/stage2b-20260729T144651.225406Z-23bdc38d7ce5`

Important identities:

- config SHA-256:
  `23bdc38d7ce5bef3694ff8ff49d0ce772ed495ec8afd2cfc464f6f58d2db6186`;
- behavior signature:
  `99ba29240a67a05455c26af54d5d0238f72fa5241465b6f2d2aefdb2c822f796`;
- HEG pilot replay hash:
  `ca39f282cdcee0925abdf7d677c8dfb5b41d0e8049db60dc7f23ec0b726c0bb7`.

The reviewed random ranker identity is source
`d4994fb96bdc3c23b8b24d9bca041f2822bc30329bcf8f9cdbd2e277e65b0612`,
normalized AST
`f7f502b0319df5dc32ef0f8476024c4986dcb3422ef2e03b117a3d394bbfc7b7`,
134 nodes. The structural identity is source
`68aba299d7735198d38a8d30e221ef99cdbb7d846c502aca41691c49ceef87be`,
normalized AST
`5b017c2ba79953e31b224df91e060d4af27c3b212695a03e8650ec91e8b0ad81`,
108 nodes.

The largest artifact, `result.json`, is 1,775,300 bytes; every JSON artifact is
below the 2 MiB per-artifact limit. Source, validation identities, schema
hashes, config, budgets, behavior signature, worker telemetry, provenance, and
terminal status are durable. Rich and JSON render the same canonical result.

## Preregistered toy benchmark

The immutable config freezes toy order 8, graph seed 101, policy seeds 1–32,
eight steps, 512 deterministic paired-bootstrap resamples, a 95% interval, and
the 10% relative median normalized best-so-far AUC threshold.

Final result:

| Metric | Random | Structural |
| --- | ---: | ---: |
| median normalized best-so-far AUC | 0.5000 | 0.5000 |
| median best total witnesses | 2.0 | 2.0 |

Relative median AUC improvement was **0.0%**, below 10%. The paired bootstrap
interval was `[0.0, 0.03125]`, so it did not exclude zero. The remaining four
criteria passed: 32 paired seeds, no worse median best-total-witness metric,
zero invalid host-applied graphs, and zero structural-ranker
timeouts/crashes.

During validation, the first structural fixture was rejected before execution
because literal multipliers exceeded the Stage 2A AST bound. A subsequent
trial exposed an incorrect `local_c4_risk` calculation over the pre-removal
graph. That feature defect was corrected to count deduplicated new local
cycles on the conceptual rewritten graph without changing the preregistered
dataset, seeds, metric, or threshold. The final run above still failed the two
efficacy criteria. No threshold or dataset was changed after observing the
result.

This is an infrastructure comparison, not an HEG scientific claim.

## HEG order-30 pilot

The bounded order-30 pilot completed and replayed with the same canonical hash.
All graphs were valid, the HEG pin remained clean, and random/structural workers
each completed 24 calls with zero failures. Selected-only scoring accounting
matched, hidden best-of-K scoring was false, and exact-verifier calls were
zero. Rich/JSON canonical parity passed.

The existing HEG uniform and forbidden-cycle-break policies were recorded as
separate references. Their single pilot results had respectively 85 capped
witnesses/weighted penalty 440 and 82/408. These are non-held-out reference
observations, not a win claim and not inputs to the gate.

## Commands and tests

Primary commands:

```console
uv run mforge proposals inspect \
  --config configs/stage2b-preregistered.toml --json
uv run mforge policy evaluate fixtures/rankers/stage2b_structural.py \
  --config configs/stage2b-preregistered.toml
uv run mforge policy compare RANDOM STRUCTURAL \
  --config configs/stage2b-preregistered.toml --json
uv run pytest
uv run ruff check .
uv run mypy
uv run mforge doctor --heg-repo ../heg
git diff --check
```

Focused Stage 2B validation passed 25 tests covering exact perfect-matching
counts, legal `k=2,3,4` pools, invalid plans, disjoint selection,
determinism/deduplication, retry exhaustion, feature budgets, relabeling,
frozen contracts, same-pool proof, tie-breaking through the Stage 2A worker,
behavior signatures, Rich/JSON parity, the frozen toy gate, HEG replay and
accounting, and interrupted terminal artifacts.

Final repository validation completed with 172 passed, zero failed, and zero
skipped in 5.17 seconds. Ruff, strict mypy, `doctor`, and
`git diff --check` passed. The final proposal-inspection smoke retained 12
bounded plans after 35 attempts with `k` counts 5/5/2 for `k=2/3/4`; its JSON
payload was 12,656 bytes.

## Decision

Safety, boundedness, deterministic replay, fairness, artifact durability, and
the HEG infrastructure pilot passed. The preregistered toy efficacy gate did
not.

**NO_GO**

Do not start Stage 3, make a model call, implement evolution, or integrate a
policy into HEG. A future issue may preregister a more discriminating toy
dataset or investigate why this order-8 pool saturates both median AUCs, but it
must not reinterpret this failed gate.
