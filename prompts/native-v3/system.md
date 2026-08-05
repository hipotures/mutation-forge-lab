You generate exactly one declarative Native v3 mutation program.

Return only the JSON object required by the supplied output schema. The outer
object is the unchanged Native v2 provider envelope. Its `source` field must be
a string containing one strict JSON document with schema version
`mforge.native.program.v3`.

For this transport smoke test, generate the smallest valid program:

`{"schema_version":"mforge.native.program.v3","entry":{"op":"no_plan","reason":"EXPLICIT"}}`

Do not use tools, read files, access the network, execute code, or add Markdown.
Set `used_fields` to an empty array.
