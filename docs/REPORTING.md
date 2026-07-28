# Reporting and provenance

Every run directory contains:

```text
run_config.toml
run_manifest.json
events.jsonl
run_summary.json
environment.json
dataset_manifest.json
archive.sqlite3
artifacts/
  programs/
  prompts/
  responses/
  graphs/
```

The manifest records Mutation Forge and HEG commits and dirty states, resolved
configuration and hash, Python/OS/architecture, lock hash, schema versions,
seeds, resource limits, timestamps, and terminal status. SQLite stores run and
event metadata; larger immutable evidence remains in files.

Terminal states are explicit. Timeout, crash, incomplete execution, unknown
verification, or scorer failure must never be relabeled as success.

Run summaries report `real_seconds`, `user_seconds`, and `system_seconds`.
User and system CPU combine the `mforge` process with its reaped children and
exclude the outer `uv` launcher.

The Rich display auto-refreshes at four frames per second. Progress events
update its state without forcing an immediate redraw; terminal events refresh
immediately.

Milestone reports include exact commands, test counts and outcomes, dataset
manifest, baseline results, throughput, reproducibility evidence, artifact
paths, limitations, unimplemented stages, and a GO/NO-GO recommendation for
the next gate.
