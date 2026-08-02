# Native experiment workflow

Issue #18 is implemented by the native experiment path. The installed public
commands are:

```console
uv run mforge experiment run [--config PATH] [--json]
uv run mforge experiment status [--config PATH] [--json]
```

`experiment.toml` is authoritative. A fresh run creates an atomic
`workspace/<exp_id>/` containing the immutable configuration and lock, SQLite
state, checkpoints, native archive, evaluation artifacts, and complete Codex
App Server turn evidence. The native coordinator executes the configured
model, effort, concurrency, repair, generation, evaluation, worker, thread,
selection, and wall-time values; it never reads a historical campaign or
freeze.

Native prompts, response schema, and baseline descriptions live under
`prompts/native/` and `configs/native/`. The prompt and behavior probe use the
same context/proposal schemas and semantic glossary as the HEG evaluator:
`configs/schemas/stage2b-*.schema.json` and
`configs/stage3-field-semantics.v1.json`. HEG is the mathematical backend and
the lock records the current sibling checkout's commit and dirty state. No
historical `runs/` directory, Stage 4 freeze, tag, or campaign output is
required.

## Turn artifact semantics

Each native turn keeps the model-facing text separate from the host envelope:

- `slot-XX.request.md` is the exact final Markdown prompt sent to the model;
- `slot-XX.request.json` is the structured request envelope and metadata;
- `slot-XX.system-prompt.md` and `slot-XX.output-schema.json` retain the exact
  system instructions and schema supplied to the provider;
- `slot-XX.response.md` is a concise generated-policy projection with a fenced
  Python source block;
- `slot-XX.response.json` is the parsed policy object, while
  `slot-XX.response.raw.txt` is the byte-faithful assistant text;
- `slot-XX.response-diagnostics.json` contains only JSON/schema response
  diagnostics, and `slot-XX.transport-diagnostics.json` contains App Server
  lifecycle diagnostics;
- `validation.json`, `metadata-validation.json`, `behavior.json`, and
  `worker_telemetry.json` separately retain AST validation, source-derived
  `used_fields`, behavior-probe results, and worker evidence;
- `slot-XX.provider-raw.json` and the JSONL transport files retain provider
  lifecycle and wire evidence.

Malformed or schema-invalid responses retain `response.raw.txt` and
`response-diagnostics.json` but do not receive a misleading `response.md`
projection.

Stage 4 commands, adapters, and retained evidence remain private historical
regression material. They are not reachable from the public experiment run or
status path.
