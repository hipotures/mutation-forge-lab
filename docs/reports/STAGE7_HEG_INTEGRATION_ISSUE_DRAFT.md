# Draft issue for `hipotures/heg`: reviewed Stage 4R proposal-ranking lane

This is a draft only. Stage 7 does not create the issue, branch, pull request,
migration, or production rollout.

Stage 7 terminal decision: `NO_GO`. The draft is retained for a separately
reviewed future HEG issue; it was not created automatically.

## Base and reviewed identity

- base HEG commit: `fd97451b0f3d87400d1d955a2c6b1b18303344ff`;
- catalog ID: `mutation_forge_stage4r_v1`;
- policy ID: `program-d5ad1c8203e0d9f25f03aabd`;
- source SHA-256: `e444562c1b308e3b23cb732be5f769ea1923ac1809501cea8571318c4aff0a7b`;
- normalized AST SHA-256: `2243214df58c805e9a9343dc31ed082279e1c2ac31b21243bf889dbc9a19e165`;
- behavior signature SHA-256: `8c2bdaa213f11b253d3ffcae1653bd01536879bb5c254a1586ded9ae522a868e`.

## Required HEG change surface

Review `src/sglab/targets/erdos_gyarfas.py` and `targets/base.py` for a
bounded immutable proposal type and legal k-switch generation for k=2,3,4;
`research/catalog.py`, `research/validation.py`, and `research/lanes.py` for a
known-name reviewed action and host-owned pool/ranker seam; `research/store.py`,
`research/recovery.py`, and `db.py` for additive identity/checkpoint fields;
the scorer-worker/resource launcher for trusted digest, stdin, limits, and
process-group cleanup; and the verification broker/candidate lifecycle for
immutable M4 snapshot handoff. Update architecture references and tests in
the same review.

## Activation and API

Add a default-off lane/campaign parameter accepting only
`mutation_forge_stage4r_v1`. Reject source text, filesystem paths, environment
configuration, downloads, and unreviewed names. Existing lanes and resumes do
not activate it implicitly.

## Proposal pool and feature mapping

Generate a bounded pool of already legal proposals with the frozen
`stage2b.context.v1`, `stage2b.proposal.v1`, and `stage2b.pool.v1` semantics.
Compute forbidden lengths 4,5,6,7,8,9 and all aligned sampled-witness,
edge-load, distance, local-risk, selector, and k fields exactly. Preserve HEG
RNG ownership; policy calls must not consume lane RNG. Sort finite priorities
descending and resolve ties by stable proposal ID. Apply and authoritatively
score only the selected rewrite. M4 remains the only certification authority.

## Identity, failure, security, and persistence

Persist catalog/source/AST/behavior, validator/runtime, feature/pool, tie, and
failure versions in lane and checkpoint identity. Resume requires exact
equality. Fail closed on load/identity drift, timeout, crash, protocol error,
non-finite output, input mutation, invalid pool, or stale checkpoint; never
silently fall back. Run the policy only in the accepted bounded worker with no
filesystem, environment, subprocess, network, database, reflection, dynamic
code, or inherited stdin authority. Add fields additively and preserve
backward-readable historical evidence; rollback is an Online Backup restore.

## Telemetry and tests

Add bounded micro-batch counters for calls, invalid/failure classes, latency,
selected k/selectors, ties, pool/feature time, scorer calls, and process
orphans. Add unit, schema, migration/backup, replay, RNG/completion-order,
legality, selected-only scorer, process/resource, checkpoint/resume, Director
catalog, rollback, and M4-isolation tests. Add a deterministic fixture matrix
covering orders 14,18,20,22,24,28,30, all k/selectors, forbidden-cycle and
non-anchored pools, relabeling, duplicate/invalid/empty pools, and interrupted
worker/resume.

## Staged rollout and non-goals

First land the seam disabled, then run the frozen replay and operational gates,
then enable only in a reviewed canary lane. A separate explicit production
enablement decision is required after review. This issue does not authorize
automatic rollout, generation/mutation/repair/tuning, scorer or M4 changes,
policy replacement, arbitrary source loading, or merge.
