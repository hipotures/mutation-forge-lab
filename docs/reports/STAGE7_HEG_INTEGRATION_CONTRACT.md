# Stage 7 HEG integration contract v1

Status at preregistration: frozen for authoritative execution. This document
describes an out-of-tree reference boundary only; it does not change the
`hipotures/heg` checkout.

## Identity and activation

The only reviewed catalog entry is `mutation_forge_stage4r_v1`. It resolves to
policy `program-d5ad1c8203e0d9f25f03aabd` and the exact source, normalized AST,
and preserved Stage 4R/Stage 5/Stage 6 behavior identities in
`configs/stage7-heg-frozen-policy-identity-v1.json`:

- source: `e444562c1b308e3b23cb732be5f769ea1923ac1809501cea8571318c4aff0a7b`;
- normalized AST: `2243214df58c805e9a9343dc31ed082279e1c2ac31b21243bf889dbc9a19e165`;
- behavior signature: `8c2bdaa213f11b253d3ffcae1653bd01536879bb5c254a1586ded9ae522a868e`.

Activation is disabled by default. A reviewed lane parameter must name the
catalog ID. Source text, arbitrary paths, Director input, downloads, and
silent activation on existing campaigns or resumes are rejected.

## Host/policy authority boundary

HEG remains responsible for graph ownership, deterministic legal proposal-pool
generation, exact bounded Stage 2B feature computation, worker invocation,
stable tie-breaking, applying one selected legal rewrite, selected-plan-only
authoritative scoring, acceptance, checkpointing, telemetry, and M4. The
policy receives only immutable JSON `stage2b.context.v1` and
`stage2b.proposal.v1` values. It cannot construct graphs, apply rewrites, call
the scorer/verifier/M4, select seeds, or observe lane state.

The bridge uses the accepted Stage 2A worker (`stage2a.worker.v1`) with its
validated AST, sanitized environment, protocol pipe, resource limits, and
reaped process lifecycle. A timeout, crash, protocol error, non-finite output,
input mutation, identity drift, or invalid proposal fails closed. There is no
random, uniform, structural, or forbidden-cycle fallback inside a trajectory.

## Exact feature and proposal mapping

The frozen context and proposal schemas are copied without transformation.
Vectors align on the frozen Stage 2B forbidden lengths `[4, 5, 6, 7, 8, 9]`
with witness sample cap 32, cycle-node budget 20,000, distance-query budget
256, and local-risk budget 2,048. The host must compute the Stage 2B lengths
and bounded sampled-witness, edge-load, distance, and local risk fields
exactly; HEG's existing power-of-two scorer lengths are not a substitute.
Legal `k=2`, `k=3`, and `k=4` proposals remain in one pool. A
future HEG implementation therefore needs an additive host-side generalized
k-switch generator and proposal-pool/ranker seam. The bridge does not silently
restrict to `k=2`.

Priorities are ordered descending, then by the lexicographic 64-hex
`proposal_id`. Only the selected proposal may be applied and scored. A policy
score is heuristic and never M4 evidence.

## Persisted identity and resume

The catalog ID, source/AST/behavior identities, validator/runtime versions,
feature and pool contract versions, tie rule, and failure policy bind lane and
checkpoint identity. Resume requires exact equality. A changed identity is a
reviewed new lane or trajectory-breaking fork; it is never silently continued.

## Observability and rollback

Only bounded micro-batch counters/histograms are persisted: policy calls,
invalid/timeout/crash/protocol counts, selection latency, selected `k` and
selector aggregates, ties, pool/feature time, scorer calls, and process
orphan count. Per-proposal histories and prompt/log payloads are forbidden.

The preregistered operational benchmark requires at least 100,000 worker calls,
p99 policy latency at or below 5 ms, zero failures/orphans/unauthorized calls,
and a faithful HEG end-to-end throughput projection. The projection is not
invented by the bridge when the pinned HEG lacks the required seam.

Disabling the feature restores the existing default for new lanes. Historical
campaigns, evidence, and checkpoints are not rewritten. Any production schema
change must be additive and backward-readable; rollback is an Online Backup
restore, not an in-place downgrade.

The machine-readable contract is
`configs/stage7-heg-integration-contract-v1.json` and its schema is
`configs/schemas/stage7-heg-integration-contract.schema.json`.
