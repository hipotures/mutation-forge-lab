# Stage 6 red-team report

Red-team decision: **passed**. All 30 required fixtures behaved as specified;
no finding blocks the Stage 6 terminal decision `GO_TO_STAGE_7`.

Machine-readable findings: `runs/stage6-verification/issue-16/redteam/findings.json`.
The fixture corpus is also retained under
`runs/stage6-verification/issue-16/redteam/fixtures` and
`fixtures/stage6/tampering`.

## Severity summary

| severity | count | result |
| --- | ---: | --- |
| critical | 4 | all rejected; all tests passed |
| high | 21 | all rejected; all tests passed |
| medium | 1 | rational-float drift rejected; test passed |
| low | 0 | none |
| informational | 4 | permitted metamorphic changes accepted |

Severity describes the impact of the attempted corruption, not an unresolved
defect. Every listed fixture has `passed: true`.

## Post-result provenance amendment — resolved validation finding

An independent post-result review found that the immutable Stage 6 freeze
recorded `required_project_commit` as the older ancestor
`cc2f7b7254705d47fd4995a4b8a2bd45d545795c` instead of the issue-required Stage 5
final commit `af8a3b5760fc2a8a9778aa575e63f573fd7eb828`. This is classified as a
medium-severity provenance-validation finding: an ancestry-only check could have
accepted the stale ancestor and weakened entry validation. It is resolved by an
exact-entry guard and regression test; the original freeze payload and
`stage6-verification-frozen-v1` tag remain immutable.

The machine-readable amendment is
`configs/stage6-verification-provenance-amendment-v1.json` with canonical
SHA-256 `f7d4a80c1591f584562e47f95a9a53df8fb036e7100f53208ee5530b0cb3111a`.
Its Git-object proof shows merge-base
`af8a3b5760fc2a8a9778aa575e63f573fd7eb828` and exact ancestry from that commit
to the Stage 6 implementation, freeze, and final commits. It also records the
Stage 5 implementation/report tree as present before preregistration. The
actual run therefore used the required integrated base; no policy, manifest,
seed, metric, bootstrap, threshold, gate, outcome row, or evidence artifact
changed, and no experiment or scientific reduction was rerun. The existing
scientific terminal decision **`GO_TO_STAGE_7`** remains valid, and Stage 7 was
not started.

## Required corruption fixtures

The verifier rejected each of the following: missing episode; duplicate
episode; extra episode; swapped policy identity; altered policy source or AST
identity; changed graph, relabeling, or policy seed; non-bijective relabeling;
relabeling after policy observation; one-policy initial-graph substitution;
altered trajectory curve or selected score; hidden full-pool oracle accounting;
unequal proposal/score/horizon budget; incorrect order weighting; incorrect
graph/relabel/policy hierarchy; unpaired bootstrap resampling; changed
bootstrap seed or percentile rule; stripped scientific field; retained timing
field; completion-order-dependent reduction; shard truncation or decompression
corruption; manifest or schema substitution; moved preregistration tag; runtime
network/provider call; sandbox timeout/crash/protocol-input mutation; and
evidence path traversal or unsafe output overwrite.

The four critical cases were altered policy provenance, altered scientific
trajectory/selected-score evidence, runtime network/provider use, and unsafe
evidence output handling. All four were rejected. The 21 high cases were also
all rejected. The only medium case, `fraction_float_drift`, was rejected rather
than silently accepted as an exact rational value.

## Metamorphic fixtures

The verifier accepted only the transformations that preserve scientific
content: shard permutation, record permutation, harmless timing change, and an
equivalent label-preserving relabeling. It rejected `fraction_float_drift`,
which changes a scientific rational. This demonstrates that timing and
transport order are canonicalized while scientific fields remain protected.

## Unresolved findings

None. The transient worker-timeout artifacts from the first concurrent fresh
attempt are retained as recovery evidence and were resolved by adopting valid
checkpoints and completing the frozen matrix; they did not alter any completed
scientific row or input.
