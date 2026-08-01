# Stage 4E Retained Recovery Report

Decision: **GO_TO_STAGE_5**

This report reduces the preserved Stage 4E primary and deterministic replay artifacts. It does not execute a graph, policy, model, App Server, oracle, or network provider.

## Preservation and frozen provenance

- Preservation metadata: `/home/user/DEV/mutation-forge-evidence/preservation-metadata.json`.
- Preserved run: `/home/user/DEV/mutation-forge-evidence/stage4e-confirmation-retained-recovery`.
- Source run: `/home/user/DEV/mutation-forge-lab/runs/stage4e-confirmation`.
- Preservation manifests byte-identical: `True`.
- Source file count/bytes: `54` / `707645`.
- Preserved-copy file count/bytes: `54` / `707645`.
- Frozen manifest SHA-256: `d80164cc4e0f26e2a2999adb7b1f8ff4b40a194e6f2576962190bd7b7bd22a34`.
- Canonical reduction SHA-256: `c10e135df06963014be00a5cb262dce1260906b26ba5d84d8a9d79c282121282`.
- Metrics-input SHA-256: `92247c893a6e347925dec06cefec7c8d17b898b3ab7a23322e20c26e8f302bdd`.
- Provider/model/App Server/oracle/runtime-network calls: `0`.

## Canonical replay identity

The explicit recursive projection removes only `timing_ns`; no arbitrary or scientific field is filtered.
- Primary rows: `1536`; replay rows: `1536`.
- Primary/replay shards: `24` / `24`.
- Timing-stripped row aggregate SHA-256: `e5b4f74a51ad53226510f582b675d7f8d35ef0280c3525df30c76571c571d3b2` (both passes).
- Timing-stripped shard aggregate SHA-256: `bb5b45ece4e1a6b8dafc80c3b47b64f4ce675a61d7f083f3ed476e65d57d6f34` (both passes).
- Timing-stripped rows exact: `True`.
- Timing-stripped shards exact: `True`.
- Non-timing differences: `0`.

## Frozen scientific result

- Paired-area theta: `73654166666666659967/2457600000000000000000` (0.02996995713975694).
- Comparator hierarchical mean AUC (mu_B): `746499722222222226613/819200000000000000000` (0.9112545437282986).
- Relative improvement: `73654166666666659967/2239499166666666679839` (0.03288867786287038).
- Bootstrap: `10000` draws, seed `2026080102`, interval `[9564155273437499053/384000000000000000000, 3452503333333333062179/98304000000000000000000]`.
- Bootstrap sign counts: `{'negative': 0, 'positive': 10000, 'zero': 0}`.
- Terminal gate: `{'all_pass': True, 'checks': {'all_primary_episodes_complete': True, 'all_replay_episodes_complete': True, 'bootstrap_lower_bound_at_least_threshold': True, 'bootstrap_lower_bound_positive': True, 'frozen_policy_identities_exact': True, 'graph_validity_100_percent': True, 'model_app_server_calls_zero': True, 'order_effects_nonnegative': True, 'primary_replay_exact': True, 'relative_improvement_at_least_threshold': True, 'selected_plan_only_and_oracle_zero': True, 'worker_failures_crashes_timeouts_protocol_zero': True}}`.

### Equal-weight order effects

- order 10: delta = `37/2048` (0.01806640625)
- order 12: delta = `24341666666666659967/819200000000000000000` (0.029713948567708325)
- order 16: delta = `2761/65536` (0.0421295166015625)

### Graph-cluster means

- order 10, graph 501: delta = `113/5120` (0.0220703125)
- order 10, graph 502: delta = `-41/5120` (-0.0080078125)
- order 10, graph 503: delta = `89/2560` (0.034765625)
- order 10, graph 504: delta = `-9/5120` (-0.0017578125)
- order 10, graph 505: delta = `5/256` (0.01953125)
- order 10, graph 506: delta = `29/1024` (0.0283203125)
- order 10, graph 507: delta = `119/5120` (0.0232421875)
- order 10, graph 508: delta = `0` (0.0)
- order 10, graph 509: delta = `17/1024` (0.0166015625)
- order 10, graph 510: delta = `-7/2560` (-0.002734375)
- order 10, graph 511: delta = `127/2560` (0.049609375)
- order 10, graph 512: delta = `81/2560` (0.031640625)
- order 10, graph 513: delta = `-1/1024` (-0.0009765625)
- order 10, graph 514: delta = `19/1280` (0.01484375)
- order 10, graph 515: delta = `31/1024` (0.0302734375)
- order 10, graph 516: delta = `81/2560` (0.031640625)
- order 12, graph 501: delta = `1683333333333332773/51200000000000000000` (0.03287760416666666)
- order 12, graph 502: delta = `933333333333332993/51200000000000000000` (0.01822916666666666)
- order 12, graph 503: delta = `66666666666666659/5120000000000000000` (0.013020833333333332)
- order 12, graph 504: delta = `448333333333333211/10240000000000000000` (0.04378255208333332)
- order 12, graph 505: delta = `39499999999999987/1024000000000000000` (0.038574218749999986)
- order 12, graph 506: delta = `96666666666666641/10240000000000000000` (0.009440104166666664)
- order 12, graph 507: delta = `205833333333333303/5120000000000000000` (0.04020182291666666)
- order 12, graph 508: delta = `389999999999999847/10240000000000000000` (0.038085937499999986)
- order 12, graph 509: delta = `1033333333333333191/25600000000000000000` (0.04036458333333333)
- order 12, graph 510: delta = `329999999999999921/10240000000000000000` (0.03222656249999999)
- order 12, graph 511: delta = `244166666666666603/5120000000000000000` (0.04768880208333332)
- order 12, graph 512: delta = `29687499999999991/1600000000000000000` (0.018554687499999993)
- order 12, graph 513: delta = `186666666666666609/10240000000000000000` (0.01822916666666666)
- order 12, graph 514: delta = `478333333333333163/10240000000000000000` (0.046712239583333315)
- order 12, graph 515: delta = `87916666666666647/2560000000000000000` (0.03434244791666666)
- order 12, graph 516: delta = `158333333333333207/51200000000000000000` (0.0030924479166666644)
- order 16, graph 501: delta = `481/8192` (0.0587158203125)
- order 16, graph 502: delta = `335/8192` (0.0408935546875)
- order 16, graph 503: delta = `21/512` (0.041015625)
- order 16, graph 504: delta = `255/8192` (0.0311279296875)
- order 16, graph 505: delta = `261/4096` (0.063720703125)
- order 16, graph 506: delta = `167/4096` (0.040771484375)
- order 16, graph 507: delta = `199/4096` (0.048583984375)
- order 16, graph 508: delta = `181/8192` (0.0220947265625)
- order 16, graph 509: delta = `445/8192` (0.0543212890625)
- order 16, graph 510: delta = `23/512` (0.044921875)
- order 16, graph 511: delta = `33/1024` (0.0322265625)
- order 16, graph 512: delta = `357/8192` (0.0435791015625)
- order 16, graph 513: delta = `221/4096` (0.053955078125)
- order 16, graph 514: delta = `125/4096` (0.030517578125)
- order 16, graph 515: delta = `359/8192` (0.0438232421875)
- order 16, graph 516: delta = `195/8192` (0.0238037109375)

Historical Stage 4E artifacts remain distinct; no Stage 5 work was started and HEG was not modified.
