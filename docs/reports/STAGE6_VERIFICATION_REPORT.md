# Stage 6 independent verification report

Terminal decision: **`GO_TO_STAGE_7`**

This is the complete Stage 6 decision for issue #16. It does not start Stage 7,
merge a branch, or modify HEG. The issue remains open for review.

## Frozen boundary and provenance

The required entry commits were verified before the preregistration:

- Mutation Forge: `af8a3b5760fc2a8a9778aa575e63f573fd7eb828`.
- HEG: `fd97451b0f3d87400d1d955a2c6b1b18303344ff`, read-only and clean.
- Preregistration commit: `6eaf9a446668751706239e6c1d8d10a26e32fde2`.
- Annotated freeze tag: `stage6-verification-frozen-v1` (peeled commit
  `6eaf9a446668751706239e6c1d8d10a26e32fde2`).
- Freeze payload hash: `d252c8c0e037a9e515708eecf630cf7fbf02f6ff3f45a248238fba32824a7f98`.

The required preregistration issue comment was posted before official audit or
fresh execution and contained `Official Stage 6 verification results observed:
false`. The four policy identities were frozen in
`configs/stage6-verification-freeze-v1.json`; no policy source, manifest, seed,
metric, threshold, or gate was changed after the freeze. No model, Codex App
Server, provider, oracle, or runtime-network call was made.

Fresh Stage 6 inputs were the exact frozen matrix: orders 20/24/28; graph seeds
701--708; relabeling seeds 7101/7102; policy seeds 7001--7016; horizon 32;
768 identities; 12 shards of 64. The fresh manifest hash is
`862d5fb090a81c798f39caf48a7c29dd70ad931fa09204f2aa168bb2248cc982`.
Resources were eight workers at most, eight reserved physical cores, and one
thread per numerical library. Bootstrap was 10,000 paired hierarchical draws,
seed `2026080104`, with the deterministic linear percentile rule at 95%.

## Phase A — forensic Stage 5 audit

The source evidence was inventoried and analyzed only through the byte-identical
audit copy. The machine-readable report is
`runs/stage6-verification/issue-16/stage5-audit.json`; it records 19 passing
assertions and an inventory of 56 files. The source and audit-copy tree hash is
`e9aba85f717a867de86680c59481886d298322b48ff4519d616b422ee4c8871d`.

The preserved Stage 5 evidence is
`/home/user/DEV/mutation-forge-evidence/stage5-generalization/issue-15-final`.
Its existing sorted manifest was verified byte-for-byte and has the required
SHA-256 `e996563c145ac12bc7eae9bb284ae98d14a2990aaac9bce17e9992486780cce`.
The frozen Stage 5 scientific manifest is
`ded50562899fd3b5d6214757f2581a2aab6507444a216643ac11fba0bb748c9d`, and its
freeze payload hash is
`53f2df2d71b723dbdcd5983d24dcff25f977e2709a0089011882c4c56f860645`.
The audit passed Git ancestry and freeze provenance, the four policy source/AST/
behavior identities, all schemas and manifests, 24 primary plus 24 replay
shards, complete pairing, equal budgets, graph validity, Fisher--Yates relabel
proofs, selected-plan-only scoring, zero oracle/provider counters, and exact
timing-only replay projection. The original evidence was not modified.

## Phase B — independent Stage 5 recomputation

`runs/stage6-verification/issue-16/stage5-recomputation.json` parses raw gzip
shards directly and independently recomputes normalized curves, paired-area
effects, hierarchy, policy means, relative effects, structural retention, all
10,000 Stage 5 bootstrap draws (seed `2026080103`), intervals, sign counts, and
all seven retained metric gates. The primary/replay projection is exact for all
1,536 rows, the field-by-field comparison has `exact: true` and
`differences: []`, and the retained result SHA-256 is
`7b004772e2ac341c5af57fd10b850714b82b83405eaa7b6a2d8d61425675bb83`.

The independent package has an AST-enforced boundary: it imports no
`mutation_forge.stage5` implementation for reduction, metrics, bootstrap,
relabeling, timing stripping, gates, or terminal decisions.

## Phase C — red-team

The red-team corpus is recorded in
`runs/stage6-verification/issue-16/redteam/findings.json` and
`docs/reports/STAGE6_REDTEAM_REPORT.md`. All 30 cases passed: 25 deliberate
tamper cases were rejected and four timing/order/label-preserving metamorphic
changes were accepted. The rational-float drift metamorphic case was correctly
rejected as a medium-severity scientific substitution. There are no unresolved
critical, high, or scientifically material medium findings.

## Phase D — fresh independent replication

Official primary shard-00 was run first at concurrency one and was not rerun.
The remaining valid shards were resumed from checkpoints. Two concurrent
attempts exposed a recoverable ranker wall-timeout before an outcome row was
accepted; the failed shard artifacts remain retained, valid orphan-complete
shards were adopted, and the frozen run completed sequentially at one worker.
This changed no scientific input or completed outcome row and is not an
unresolved scientific finding.

The completed primary and exactly one replay each contain 768 rows, all 12
shards, equal budgets, valid graphs and relabel proofs, and zero persisted
worker/protocol failures. Timing-stripped replay verification is exact:

`64bcef7c6348a12df618c9c71d60071a590ec27753858112260c7df0effff2a8`

Fresh paired-area effects and deterministic 95% intervals are:

| comparison | theta | relative improvement | 95% interval |
| --- | ---: | ---: | ---: |
| C vs Stage 3 | 0.05395992218501984 | 0.06047967198054991 | [0.04772575499519469, 0.06010074918232267] |
| C vs random | 0.12260916573660714 | 0.14887882269265598 | [0.11552472311352927, 0.1297188459123884] |
| C vs structural | 0.15650198800223214 | 0.19818976811909636 | [0.14622203281947546, 0.16678297496977307] |

Policy means were C `0.9461592416914683`, Stage 3 `0.8921993195064484`,
random `0.8235500759548611`, and structural `0.7896572536892361`.
Structural retention was `1.1981897681190963`. Every bootstrap sign count was
10,000 positive, zero negative, and zero zero.

## Stage 6 gates

All 15 gates in `runs/stage6-verification/issue-16/stage6-terminal.json` passed:

1. PASS — Stage 5 audit and manifest verification.
2. PASS — independent Stage 5 recomputation exact.
3. PASS — no unresolved critical/high/material-medium red-team finding.
4. PASS — fresh manifest complete and disjoint.
5. PASS — primary/replay complete with equal budgets.
6. PASS — exact timing-stripped replay identity.
7. PASS — 100% graph validity.
8. PASS — zero persisted worker, crash, timeout, and protocol failures.
9. PASS — selected-plan-only scoring and zero oracle.
10. PASS — zero model/App Server/provider/runtime-network calls.
11. PASS — C vs Stage 3 relative improvement at least 2%.
12. PASS — C vs Stage 3 bootstrap lower bound positive.
13. PASS — C vs Stage 3 nonnegative order and six-stratum effects.
14. PASS — random lower bound and structural-retention thresholds.
15. PASS — artifact provenance, preservation, and repository verification.

## Evidence preservation and boundary

The run root is `runs/stage6-verification/issue-16`. The final preserved copy is
`/home/user/mutation-forge-evidence/stage6-verification/issue-16-final` with
122 files, sorted-manifest SHA-256
`ea2843e26527d036d989c6e3eaeff2d4028c81cba3b1a21f282cbd0a5ca1ac28`, and
byte-identical tree SHA-256
`cf099462f9e1a01aeb434a2e583dcffe5fa3ceb09a0bd81375190a0bf18f1e5c`.
Prior reducer checkpoints were retained separately; no Stage 5 or Stage 6
evidence was deleted.

Stage 7 was not started. HEG remains at
`fd97451b0f3d87400d1d955a2c6b1b18303344ff` and clean. No automatic merge was
performed.
