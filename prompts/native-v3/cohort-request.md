Generate one independent Native v3 mutation-program AST for every slot below.

Use only the supplied schema and semantic registry. Prefer short programs.
Each execution path must terminate in `emit` or `no_plan`. Policies must be
label-oblivious; resolve structural ties only with a declared seeded `pick`.
Selectors observe the current private overlay. The host checks connectivity
and minimum degree at `emit`.

The inner `source` document must have exactly this shape:

`{"schema_version":"mforge.native.program_batch.v3","programs":[{"slot_id":"slot-00","program_json_raw":"{...}","design_summary":"..."}]}`

Use the exact requested slot IDs and include each exactly once. The outer
envelope must use `schema_version` `mforge.native.generated_policy.v1` and an
empty `used_fields` array.
