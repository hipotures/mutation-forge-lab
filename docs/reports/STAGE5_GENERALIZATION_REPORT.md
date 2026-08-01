# Stage 5 held-out generalization report

Terminal decision: **GO_TO_STAGE_6**

Stage 5 used the frozen four-policy roster, held-out relabeled graph manifest, and provider-free Stage 4E evaluation contract. Stage 6 was not started.

- Freeze SHA-256: `53f2df2d71b723dbdcd5983d24dcff25f977e2709a0089011882c4c56f860645`
- Manifest SHA-256: `ded50562899fd3b5d6214757f2581a2aab6507444a216643ac11fba0bb748c9d`
- Policy provenance SHA-256: `0e4d50f35462a98edb124c30551b0eb9df9cbc0a7c73d61bcf8ef84cad1e7cf4`
- Metric implementation SHA-256: `66b7ea3a6aa3d3685419b1960eb12b63de8ab64a90198c76b8323037896f841a`
- Bootstrap SHA-256: `33cdb8b541088e2ba759b9a51a95d83930281c895bcec3b61ba2eeff24b837c8`
- Gate result SHA-256: `3511c26081359e225aa17ca2f75957e20d308c3fdef8b7c61cf5ce0d3733e300`
- Primary replay-comparison hash: `180c26d148627df26529c9a00bee5b4737a3ef8d5d0e64e136b53657decf82c0`
- Replay replay-comparison hash: `180c26d148627df26529c9a00bee5b4737a3ef8d5d0e64e136b53657decf82c0`
- Primary summary hash: `d96997869294be84f17aab4be79c626763e82b3a2b94dc3cd0a0dd0a91df3981`
- Replay summary hash: `02dee045ddf50f0aca2b9933c7a76f5283d877a5eb0b864fa88d08784313a85b`
- Primary/replay canonical reduction SHA-256: `5e09b0a1c3a5609ab383d1b06003ad6f3929f7e49d18e033dc815da61bdd4eea`
- Timing-stripped reduction SHA-256: `48e8b88c26bde28b20a29492d5d2a368932144e41cd612e059cdc826245170b1`
- Terminal result SHA-256: `5c3c52756b487a90f4e3e3de7c003123f53f50cd41f535500b9fe7d7dfc31f42`
- Preserved evidence: `/home/user/DEV/mutation-forge-evidence/stage5-generalization/issue-15-final`
- Evidence manifest SHA-256: `e996563c145ac12bc7e7ae9bb284ae98d14a2990aaac9bce17e9992486780cce`
- Provider/model/App Server/oracle/runtime-network calls: **0**
- HEG commit: `fd97451b0f3d87400d1d955a2c6b1b18303344ff` (read-only and clean)

## Principal effects

- C_vs_stage3: theta `25811991341991345599/491520000000000000000`, relative improvement `129059956709956727995/2212252958152958125318`, 95% interval `[{'fraction': '3146508465608466099029/65536000000000000000000', 'value': 0.04801190896009012}, {'fraction': '175212558621933643541/3072000000000000000000', 'value': 0.057035338093077356}]`.
- C_vs_random: theta `54416005291005293999/409600000000000000000`, relative improvement `326496031746031763994/2014816883116883089319`, 95% interval `[{'fraction': '24796068686868688184041/196608000000000000000000', 'value': 0.12611932722406355}, {'fraction': '2282894925444925586773/16384000000000000000000', 'value': 0.13933684847686315}]`.
- C_vs_structural: theta `293373376623376637741/2457600000000000000000`, relative improvement `293373376623376637741/2047939538239538215572`, 95% interval `[{'fraction': '7319678739778740211669/65536000000000000000000', 'value': 0.11168943389554963}, {'fraction': '12488067821067821571379/98304000000000000000000', 'value': 0.12703519511991193}]`.

## Secondary descriptive analyses

- Order effects (C−Stage 3): order 14 `0.0419747489`, order 18 `0.0560099284`, order 22 `0.0595592152`.
- Six order×relabel effects (C−Stage 3): 14/6101 `0.0425502232`, 14/6102 `0.0413992746`, 18/6101 `0.0542534722`, 18/6102 `0.0577663845`, 22/6101 `0.0605801669`, 22/6102 `0.0585382635`.
- Episode sign counts (negative/zero/positive): C−Stage 3 `248/81/1207`; C−random `34/3/1499`; C−structural `108/15/1413`. All graph, order, and six-stratum cluster effects are positive.
- Hierarchical policy mean AUC: C `0.9526826639`; Stage 3 `0.9001680331`; random `0.8198310885`; structural `0.8333087314`; structural retention `1.1432529482`.
- Median episode AUC: C `0.9553571429`; Stage 3 `0.90625`; random `0.8303571429`; structural `0.84375`. Median paired deltas: C−Stage 3 `0.0451388889`; C−random `0.1221590909`; C−structural `0.1071428571`.
- Best-witness min/median/max: C `0/0/1`; Stage 3 `0/0/2`; random `0/0/3`; structural `0/0/7`. Final normalized-curve min/median/max: C `0.8888888889/1/1`; Stage 3 `0.7142857143/1/1`; random `0.5714285714/1/1`; structural `0.3636363636/1/1`.
- Every policy had a first strict improvement in all 1,536 episodes; median evaluations to first improvement was 1 for each policy.
- Mean AUC by relabeling seed (6101/6102; absolute difference): C `0.9535987197/0.9517666081` (`0.0018321116`); Stage 3 `0.9011374323/0.8991986339` (`0.0019387983`); random `0.8222409643/0.8174212127` (`0.0048197516`); structural `0.8366480796/0.8299693832` (`0.0066786964`).
- Primary artifacts: 24 shards, 23,536,538 compressed bytes and 204,061,859 uncompressed bytes; all shards remained below the 32 MiB uncompressed limit. Runtime is intentionally timing-stripped from canonical artifacts.

## Gates

- PASS — 10_C_vs_stage3_bootstrap_lower_bound_positive
- PASS — 11_C_vs_stage3_nonnegative_each_order
- PASS — 12_C_vs_stage3_nonnegative_all_six_order_relabel_strata
- PASS — 13_C_vs_random_threshold_and_lower_bound
- PASS — 14_structural_retention_ge_99_percent
- PASS — 15_artifact_provenance_preservation_repository_verified
- PASS — 1_policy_provenance_exact
- PASS — 2_manifest_complete_and_disjoint
- PASS — 3_primary_and_replay_complete_equal_budgets
- PASS — 4_timing_stripped_replay_identity_exact
- PASS — 5_graph_validity_100_percent
- PASS — 6_zero_worker_failures_crashes_timeouts_protocol_violations
- PASS — 7_selected_plan_only_zero_oracle
- PASS — 8_zero_model_app_server_runtime_network_calls
- PASS — 9_C_vs_stage3_relative_improvement_ge_2_percent

The issue remains open for review; no automatic merge was performed.
