Generate one minimal Native v3 AST for the transport smoke test.

The `source` string must contain exactly one valid `mforge.native.program.v3`
JSON document. Use an `EXPLICIT` `no_plan` entry. Complete every field required
by the outer schema, keep `used_fields` empty, and return JSON only.
