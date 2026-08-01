# Stage 7 HEG integration decision

## Preregistration

Frozen inputs:

- Mutation Forge entry commit `a6f0da20fa5a3e1c8b58cbc77a0d613c54d9f051`;
- HEG read-only commit `fd97451b0f3d87400d1d955a2c6b1b18303344ff`;
- policy `program-d5ad1c8203e0d9f25f03aabd` through reviewed catalog ID
  `mutation_forge_stage4r_v1`;
- replay records: 2,048;
- benchmark target: 100,000 bounded policy calls;
- benchmark specification: `configs/stage7-heg-operational-benchmark-v1.json`;
- policy p99 threshold: 5 ms; throughput thresholds: 10% median and 15% per
  stratum; zero failures/orphans/unauthorized calls;
- freeze artifact: `configs/stage7-heg-integration-freeze-v1.json`.

Authoritative results were not observed at preregistration. The required issue
comment is posted only after this commit and the annotated freeze tag are
remote.

## Scope and boundary

The reference bridge is in `src/mutation_forge/stage7_heg_bridge/`. It loads
only the packaged reviewed source, generates host-owned legal k=2/3/4 pools,
computes the frozen Stage 2B fields, invokes the bounded worker, applies stable
tie-breaking, and returns a selected ID plus bounded telemetry. It never
modifies HEG or invokes a scorer, oracle, verifier, M4, model, App Server, or
runtime network from the policy path.

## Terminal result

Pending authoritative execution. This section is replaced with the exact one of
`GO_TO_HEG_INTEGRATION_ISSUE`, `NO_GO`, or
`INCONCLUSIVE_INFRASTRUCTURE_FAILURE` and the gate table after the frozen tag
and preregistration comment exist.
