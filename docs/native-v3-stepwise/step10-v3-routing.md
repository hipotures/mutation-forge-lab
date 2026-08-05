# Native v3 Step 10 explicit routing

Native v2 remains the default experiment protocol. A configuration without a
`protocol` field follows the unchanged Native v2 parser, service, workspace,
resume, and artifact path.

The Native v3 selector introduced in Step 10 is:

```toml
schema_version = "mforge.experiment.v3"
protocol = "v3"
exp_id = "v3-run"
workspace = "/tmp/mforge-v3"

[v3]
model = "gpt-5.6-luna"
effort = "high"
timeout_seconds = 300
heg_repo = "../heg"
```

The v3 configuration deliberately rejects the Native v2 `kind`, `preset`,
`run`, `model`, `search`, `evaluation`, and `resources` fields. Unknown fields
also fail before workspace creation, provider contact, or scorer construction.
Credentials remain local Codex state and are never accepted in TOML.

## Workspace and status

The selected `workspace/exp_id` must be fresh or already contain the exact v3
status marker and unchanged configuration identity. A Native v2
workspace is rejected without mutation; it is never migrated or reinterpreted.

Both public commands route through the selector:

```console
uv run mforge experiment run --config preview.toml --json
uv run mforge experiment status --config preview.toml --json
```

Status is read-only and does not construct a provider, backend, scorer, oracle,
or evaluator. It reports `protocol`, `protocol_version`, state, resumability,
and the latest infrastructure and scientific stop reasons.

Step 11 superseded the original one-slot behavior with the deterministic
cohort described in `step11-sequential-cohort.md`. The selector still does not
change the Native v2 default.
