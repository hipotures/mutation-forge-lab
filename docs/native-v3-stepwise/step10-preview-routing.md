# Native v3 Step 10 explicit preview routing

Native v2 remains the default experiment protocol. A configuration without a
`protocol` field follows the unchanged Native v2 parser, service, workspace,
resume, and artifact path.

The only Native v3 selector accepted in Step 10 is:

```toml
schema_version = "mforge.experiment.v3-preview.v1"
protocol = "native-v3-preview"
exp_id = "preview-smoke"
workspace = "/tmp/mforge-native-v3-preview"

[native_v3_preview]
model = "gpt-5.6-luna"
effort = "high"
timeout_seconds = 300
heg_repo = "../heg"
```

The preview configuration deliberately rejects the Native v2 `kind`, `preset`,
`run`, `model`, `search`, `evaluation`, and `resources` fields. Unknown fields
also fail before workspace creation, provider contact, or scorer construction.
Credentials remain local Codex state and are never accepted in TOML.

## Workspace and status

The selected `workspace/exp_id` must be fresh or already contain the exact
`native-v3-preview.v1` marker and unchanged configuration identity. A Native v2
workspace is rejected without mutation; it is never migrated or reinterpreted.

Both public commands route through the selector:

```console
uv run mforge experiment run --config preview.toml --json
uv run mforge experiment status --config preview.toml --json
```

Status is read-only and does not construct a provider, backend, scorer, oracle,
or evaluator. It reports `protocol`, `protocol_version`, state, resumability,
and the latest infrastructure and scientific stop reasons.

Step 10 is still the Step 09 smoke panel: one `slot-00`, one model turn, one
serial order-30 episode, graph seed 101, policy seed 17, horizon 1, and witness
cap 64. There is no cohort, parallelism, dashboard, until-complete loop,
profiling route, or default switch.
