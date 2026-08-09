# Native v3 Step 09 provider-to-evaluation smoke

`run_provider_evaluation_smoke` is the first real Native v3 path from one model
turn to one authoritative graph evaluation. It remains an internal Python API
and script. Native v2 remains the only public experiment route.

## Boundary

The smoke composes the accepted Step 05 and Step 08 boundaries:

1. `run_provider_smoke` performs exactly one call through the unchanged Native
   v2 `LocalCodexAppServerProvider`.
2. The provider turn is indexed and verified by the existing
   `TurnArtifactStore`; no v3 semantic file is written inside that directory.
3. The validated-program artifact outside the turn is revalidated against its
   raw source, canonical form, hash, and validator protocol.
4. A caller-supplied backend factory is opened only after provider success.
5. `evaluate_serial_program` runs one deterministic short trajectory with the
   current HEG backend and counterexample inspection pipeline.
6. The backend is closed by the orchestration boundary.

A provider or AST failure creates no evaluation artifact and no scientific
terminal result.

## Semantic output

`native-v3-output/evaluation-result.json.gz` is separate from the provider turn
and records:

- raw and canonical program forms and their identity;
- canonical JSON, validator, interpreter, graph runtime, and evaluator
  protocol identifiers;
- unchanged provider identity and provenance projections;
- graph backend, authoritative scorer, and exact HEG repository state;
- episode configuration, deterministic semantic trace, scores, acceptance,
  terminal graph identity, and trace hash.

## Manual smoke

From the dedicated worktree:

```console
uv run python scripts/native_v3_provider_evaluation_smoke.py
```

Expected result: a fresh `/tmp/mforge-native-v3-provider-evaluation-*`
workspace and a JSON report with `status: "completed"`, `model_turns: 1`,
`valid_ast: true`, and `graph_evaluations` of at least one. The command uses one
real model turn, leaves Native v2 routing unchanged, and never writes to a
production experiment workspace.
