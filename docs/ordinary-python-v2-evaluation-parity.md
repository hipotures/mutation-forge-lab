# Ordinary-Python scientific evaluation parity

Issue #70 uses Native v2 commit
`ae3ca9ef931ffaa7ab922695ee95fc4461d4ec77` as its scientific reference.
The ordinary-Python evaluator keeps the `propose(ctx, graph, api, seed)`
contract and the safe graph API; it does not restore the ranker-only JSON
contract or the removed JSON DSL.

## Parity audit

| Concern | Status | Ordinary-Python result |
| --- | --- | --- |
| Graph mode | `PORTED` | `evaluation.graph_mode` configures the authoritative HEG backend and is included in every frozen case identity. |
| Order scheduling | `PORTED` | `adaptive` uses the Native v2 seed string `graph_seeds:generation`, samples the complete allowed domain, and sorts the result. |
| Minimum/maximum order | `PORTED` | `min_order` and `max_order` are validated in the Native v2 range `4..128`. |
| Orders per generation | `PORTED` | `orders_per_generation` controls the deterministic adaptive sample. |
| Graph seeds | `PORTED` | Every selected order is crossed with every configured graph seed. |
| Policy seeds | `PORTED` | Every order/graph-seed pair is crossed with every configured policy seed. |
| Horizon | `PORTED` | Every generated-policy and baseline episode uses the configured horizon. |
| Witness cap | `INTENTIONALLY_REPLACED` | The ordinary evaluator exposes the conservative score-evidence cap explicitly as `witness_cap`; it is frozen in every case and used by both candidates and baselines. |
| Random baseline | `PORTED` | Uses the Native v2 `heg_uniform_two_switch` host operator on the identical frozen cases, scored with the same conservative ordinary-Python fitness metric as candidates. |
| Structural baseline | `PORTED` | Uses the Native v2 `heg_forbidden_cycle_break` host operator on the identical frozen cases, scored with the same conservative ordinary-Python fitness metric as candidates. |
| Population size | `PORTED` | Eight primary slots remain frozen per generation. |
| Selection semantics | `INTENTIONALLY_REPLACED` | Deterministic lineage and measured behavior/fitness selection replace the v2 ranker-oriented persistent-elite selector. Exact parents and generation snapshots are durable, so the replacement remains reproducible. |
| Repair semantics | `INTENTIONALLY_REPLACED` | One schema/source-contract repair can follow a primary ordinary-Python turn; the campaign-wide repair budget remains separate and durable. Scientific evaluations are never repaired or replaced. |
| Evaluator resource controls | `PORTED` | `evaluator_workers` controls the evaluator thread pool, corresponding to the effective Native v2 `thread_count`. Native v2 `resources.workers` did not schedule evaluator processes and is not retained. |
| Provider concurrency | `PORTED` | `provider_concurrency` remains an independent frozen control; evaluator workload does not change the number of simultaneous model turns. |
| Provider timeout | `PORTED` | `python_preview.timeout_seconds` remains the per-turn timeout and is checked against the optional campaign wall budget. |
| Replay | `NOT_APPLICABLE` | The reference profile set `replay = false`. Ordinary-Python requires that value and instead verifies immutable manifests, snapshots, case artifacts, identities, and terminal candidates on resume. |
| Token budget | `PORTED` | The rolling one-hour total-token ceiling is checked before provider work and reported from durable authoritative usage. |
| Lifetime token budget | `NOT_APPLICABLE` | The Native v2 reference had no cumulative lifetime-token ceiling. Ordinary-Python retains primary/repair/total turn budgets and the rolling hourly token ceiling without inventing a different lifetime limit. |
| Dashboard projections | `PORTED` | JSON and Rich use the same canonical workload, baseline, candidate, worker, and timing projection. Unknown values render as `—`; measured zero remains `0`. |
| Resume semantics | `INTENTIONALLY_REPLACED` | Immutable generation manifests and per-case artifacts prevent repeated terminal provider work and verify the exact generation workload on resume. |
| Prompt/output contract | `INTENTIONALLY_REPLACED` | The only ordinary-Python contract is `propose(ctx, graph, api, seed)` in the two-field `mforge.native.python_policy_response.v1` envelope. `priority(ctx, proposal)` is rejected on this route. |
| Provider routing/provenance | `PORTED` | Protocol dispatch, prompt/schema hashes, repository identities, provider state, and generation snapshots are verified before provider work. |
| `proposal_pool_size = 12` | `INTENTIONALLY_REPLACED` | A full mutator does not rank a host-created proposal pool. Bounded safe-API selectors (at most 64 opaque references), `witness_cap`, policy runtime limits, and the finite horizon bound exploration without exposing proposal identities. |

## Native v2 parity profile

The sustained scientific configuration now expresses:

- graph mode `unrestricted_min_degree_3`;
- five adaptive orders per generation from `22..128`;
- graph seeds `401..404`;
- policy seeds `4001..4016`;
- 320 deterministic cases per candidate and generation;
- horizon 32;
- random and structural baselines on the same 320 cases.

Each generation persists its selected orders, complete case list, panel hash,
candidate case artifacts, and `generation-baselines.json.gz`. The baseline
summary contains exact rational fitness intervals. Rich derives its displayed
baseline values from that canonical summary; it does not recompute them.

The provider-free benchmark command is:

```bash
uv run python scripts/native_v3_python_evaluation_parity_benchmark.py \
  --heg-repo /home/user/DEV/heg --sample-cases 2
```

It measures two cases from each profile (including the full configured horizon
and both baselines) and reports a linear projection for the complete profile.
The output still records the complete workload as 2 cases for the removed
panel and 320 cases for the restored profile; the projection is deliberately
not presented as a full scientific evaluation.

The bounded run on 2026-08-09 measured 0.2565 s for the two-case profile and
2.5691 s for two parity-profile cases (horizon 32, both baselines), projecting
approximately 411.0619 s for all 320 parity cases. Provider turns, model turns,
and App Server calls were all zero. The complete-profile number is a workload
projection, not a campaign result.

## Prompt routing evidence

The diagnostic `exp_002` ordinary-Python requests use
`mforge.native.python_policy_response.v1` and sources defining
`propose(ctx, graph, api, seed)`. The observed `priority(ctx, proposal)` prompt
belongs to the separately routed Native v2 adapter. A new fail-closed guard
rejects that ranker signature if it appears in an ordinary-Python request.

## Deliberately remaining differences

The ordinary-Python candidate metric is the conservative interval fitness from
the serial exact-evidence seam, not the old ranker mean-AUC scalar. Both
restored baselines use that same candidate metric, so comparisons are
like-for-like. Native v2 remains a separate protocol; this work does not revive
its generated-ranker contract, proposal pool, or JSON DSL.
