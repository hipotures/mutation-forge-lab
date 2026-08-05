Generate the requested bounded batch of independent Native v3 policy programs.

Use only nodes, selectors, actions, context fields, and graph features declared
by the supplied versioned schemas and registries. Prefer short programs. Each
program must terminate through exactly one `emit` or `no_plan` path and must
remain within static and dynamic budgets. Selectors observe the current private
overlay; connectivity and minimum degree are checked only at final `emit`.

Policies must be label-oblivious. Resolve structural tie-sets only through the
declared seeded `pick` operations. Do not repeat or invent parent IDs: slot and
parent assignments are frozen and owned by the host.

Return every result as:

`{"slot_id":"...","program_json_raw":"...","design_summary":"..."}`

inside the required batch envelope.
