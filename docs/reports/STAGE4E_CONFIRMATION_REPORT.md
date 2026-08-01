# Stage 4E Confirmation Report

Decision: **`INCONCLUSIVE_INFRASTRUCTURE_FAILURE`**

The frozen Stage 4E primary and deterministic replay each completed all 1,536
paired episodes. No model, App Server, oracle, Stage 5, or HEG operation
occurred. The comparison cannot produce a valid scientific `GO_TO_STAGE_5` or
`NO_GO` result because the persisted primary/replay shard identities differ.

## Frozen execution

- Policies were the byte-identical Stage 4R champion and Stage 3 comparator.
- Manifest SHA-256: `d80164cc4e0f26e2a2999adb7b1f8ff4b40a194e6f2576962190bd7b7bd22a34`.
- Matrix: orders 10, 12, and 16; graph seeds 501–516 per order; policy seeds
  5001–5032; horizon 32; 24 shards × 64 episodes.
- Bootstrap was not reached because replay identity failed first.
- Historical Stage 4R decision remains `NO_GO`; the Stage 4D diagnosis is
  unchanged.

## Infrastructure failure

Both passes report 1,536 completed episodes, zero model/App Server/oracle/
network calls, identical canonical reduction hash
`c10e135df06963014be00a5cb262dce1260906b26ba5d84d8a9d79c282121282`, and
identical metrics-input hash
`92247c893a6e347925dec06cefec7c8d17b898b3ab7a23322e20c26e8f302bdd`.

The shard aggregate hashes differ:

- primary: `8787f4d6e7c6a98d6002469bb9c92c196cde75f379ccf34fbe52bd64fea1f0b4`;
- replay: `58b422585446e6f17b7ccdfe26170881e2b03350af8430f3a0cc5777b2f3d9d4`.

The first observed difference is `timing_ns` in episode
`o10-g0501-p5001`, retained in the compact shard row. This is timing-only, but
the preregistered acceptance criterion requires exact timing-stripped shard
identity. Therefore no valid confirmatory estimand or performance gate is
reported, and no recovery or rerun is permitted under this frozen issue.

Terminal artifact: `runs/stage4e-confirmation/terminal-gate.json`.
