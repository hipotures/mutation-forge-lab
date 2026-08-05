Repair the previously returned generated-policy object using only the listed
validation diagnostics.  Preserve the intended hypothesis when it is valid;
change only schema, syntax, determinism, bounded-runtime, metadata, or behavior
probe violations. `used_fields` must use canonical `ctx.<field>` and
`proposal.<field>` names matching the repaired source.
Return one complete generated-policy JSON object and no prose outside JSON.
