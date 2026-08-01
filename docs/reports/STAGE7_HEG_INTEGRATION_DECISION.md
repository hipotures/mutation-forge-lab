# Stage 7 HEG integration decision

## Terminal decision

**`NO_GO`**

The authoritative Stage 7 execution completed successfully, but integration
is not approved. Two frozen gates failed because the pinned HEG does not yet
provide the required proposal-pool/ranker and faithful end-to-end operational
seams:

- `no_unresolved_high_or_material_medium`: **false**;
- `operational_thresholds`: **false** (a faithful HEG throughput projection was
  unavailable and was not invented).

All other frozen gates passed. This is a completed negative integration-
readiness decision, not an infrastructure failure and not HEG implementation.

## Frozen identity and provenance

- Mutation Forge entry: `a6f0da20fa5a3e1c8b58cbc77a0d613c54d9f051`;
- preregistration commit: `0e4baa988510fc61b5bea245b5112cba536a106e`;
- freeze tag: `stage7-heg-integration-decision-frozen-v1`;
- HEG: `fd97451b0f3d87400d1d955a2c6b1b18303344ff`, clean and read-only;
- reviewed catalog: `mutation_forge_stage4r_v1`;
- policy: `program-d5ad1c8203e0d9f25f03aabd`;
- source SHA-256: `e444562c1b308e3b23cb732be5f769ea1923ac1809501cea8571318c4aff0a7b`;
- normalized AST SHA-256: `2243214df58c805e9a9343dc31ed082279e1c2ac31b21243bf889dbc9a19e165`;
- behavior identity SHA-256: `8c2bdaa213f11b253d3ffcae1653bd01536879bb5c254a1586ded9ae522a868e`;
- contract SHA-256: `846c6e5a2f7d21350930824ea01d6bf4c37bcbdb0bb6f85f582a6b88eb051d4f`;
- capability matrix SHA-256: `f54d0725686d5463d6775f5037afa45d90ce00fad4b068097067aea369dfed87`;
- fixture manifest SHA-256: `0918b157f28aa27027bb4ab1c61e929308cd299ae0e50ab4c144e789818b85dc`;
- replay corpus SHA-256: `dafe31cbb832395e1ee6ae7b51f0fd51613ca175a7c391ba512640092110ab4f`;
- red-team findings SHA-256: `dfe0239a1f2a1150a9ff5b13e99c809feddeadaef92aaa12b9a9bebd6a3be069`.

The Stage 3–6 evidence chain was revalidated (`chain_ok=true`), including
Stage 5 manifest `e996563c145ac12bc7e7ae9bb284ae98d14a2990aaac9bce17e9992486780cce`
and Stage 6 manifest
`66064a1b9a7583da588d64cab2e3e4a79be6a5f77997be3df0e4fbbfd3677e87`.

## Authoritative execution

The run used only the frozen packaged source, the frozen fixture/replay/red-
team inputs, and the pinned HEG checkout. It made no model, Codex App Server,
provider, oracle, M4, or runtime-network calls.

- replay: 2,048/2,048 records, zero priority mismatches;
- HEG fixtures: 7 fixtures, 84 legal proposals, all selectors and `k=2,3,4`,
  zero RNG/completion-order drift, zero non-selected scorer calls;
- red-team: all 30 cases passed;
- policy benchmark: 100,000 calls, zero failures, zero process orphans, zero
  unauthorized calls; p50 0.157 ms, p95 0.201 ms, p99 0.250 ms; measured
  worker throughput 6,029.05 calls/s;
- memory/process verification: coordinator peak RSS 62,768 KiB against the
  frozen 131,072 KiB address-space limit; protocol worker process-group
  isolation and reaping were observed;
- benchmark replay hashes matched:
  `4f93bce3c854452d5aff38bb15c3d36c428cfc3dfe1361d22645225807db1b98`.

The p99, failure, security, and resource checks passed. The benchmark did not
claim a HEG throughput result because the pinned HEG lacks the required
policy-pool/ranker seam and its scorer fallback/process semantics are not
production-faithful.

## Frozen gate table

| Gate | Result |
| --- | --- |
| Stage 3–6 evidence chain | pass |
| Frozen champion identity | pass |
| HEG exact pin, clean/read-only | pass |
| Exact mapping or bounded additive plan | pass |
| No semantic mismatch or unknown | pass |
| Authority boundaries | pass |
| Default-off reviewed catalog ID | pass |
| Exact resume identity | pass |
| Fail-closed/no silent fallback | pass |
| 100% replay | pass |
| HEG fixture legality/parity | pass |
| RNG/completion-order stability | pass |
| All red-team cases | pass |
| No unresolved high/material-medium findings | **fail** |
| Security/resource boundaries | pass |
| Operational thresholds | **fail** |
| Default-disabled path | pass |
| Rollback/additive migration plan | pass |
| Complete future HEG issue draft | pass |
| Repository freeze/pin checks | pass |

The durable result is at
`/home/user/mutation-forge-evidence/stage7-heg-integration/issue-17-final`;
its evidence-manifest SHA-256 is
`e29c562d687d242a8194cd9e5f1079e376328b01dba2eba532dafc0dfbecb4d0`.

## Required follow-up boundary

The complete, self-contained HEG issue draft is retained in
`docs/reports/STAGE7_HEG_INTEGRATION_ISSUE_DRAFT.md` and is not created or
posted automatically. No HEG code, branch, pull request, migration, or
production integration was started. Issue #17 remains open for review.
