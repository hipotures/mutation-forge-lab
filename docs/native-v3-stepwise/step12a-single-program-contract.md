# Native v3 Step 12A: direct single-program contract

Step 12A adds an experimental prompt and structured-output contract for one
typed graph-rewrite program. It does not switch the existing cohort provider
or any production execution path.

## Prompt boundary

The stable system prompt states the mathematical mission:

- search for a counterexample to the Erdős–Gyárfás conjecture;
- synthesize one reusable typed graph-rewrite AST;
- leave graph data, validation, scoring, acceptance, persistence, lineage, and
  exact verification to the host.

The dynamic request contains only the slot, one brief, active forbidden cycle
lengths, the executable selector/action registry, and a short pre-return
checklist. The recursive JSON Schema is supplied separately as
`output_schema`; it is not embedded in prompt prose.

## Direct response

The response has exactly three fields:

```json
{
  "program": {
    "schema_version": "mforge.native.program.v3",
    "entry": {}
  },
  "design_summary": "At most three sentences.",
  "hypothesis": "At most three sentences."
}
```

The program is a JSON object, not JSON encoded inside `source` or
`program_json_raw`.

The schema and response validator are generated from the same typed selector
and action registry. Exact selector/action identifiers, argument names,
argument types, enum literals, context fields, graph features, terminal
reasons, and active witness lengths are exposed to the model.

Relocation and fanout use relation-preserving values:

- `relocations_legal` produces `RelocationRef` values consumed by
  `relocate_endpoint`;
- `edge_fanouts_legal` produces `FanoutRef` values consumed by `edge_fanout`.

This prevents a model from combining unrelated edge and vertex bindings.

## Deterministic request sizes

For forbidden lengths `(4, 8, 16)`, compact UTF-8 request-contract sizes,
including the framing newline, are:

| Brief | Bytes |
| --- | ---: |
| add-edge | 19,720 |
| remove-edge | 19,733 |
| relocation | 19,722 |
| fanout | 19,719 |

## Boundaries and limitations

- The existing production batch prompt, batch schema, response projection,
  provider transport, App Server artifact writer, retry behavior, and usage
  accounting are unchanged.
- The Step 12 scorer, interval fitness, evaluator, scheduler, SQLite schema,
  dashboard, HEG backend, and Native v2 are unchanged.
- The new direct request is a builder and validation surface only. No provider
  route uses it in Step 12A.
- Relation-preserving relocation and fanout references are not executed by the
  current production interpreter because guarded integration belongs to a
  later accepted supplemental step.
- No live model call is required or performed.

## Manual verification

Run:

```bash
uv run pytest tests/unit/test_native_v3_single_program_contract.py -q
```

Expected result: all single-program prompt, schema, golden-response, malformed
response, relation, terminal-path, and request-size tests pass.

STOP — waiting for operator acceptance.
