Generate one deterministic ranker for the supplied native experiment context.

The host owns proposal legality, authoritative scoring, and verification.
Select one legal proposal by implementing `priority(ctx, proposal) -> float` in
`source`. Larger finite priorities rank first. Use bounded arithmetic and
proposal-derived signals; context may modulate weights but may not become the
sole ranking signal. Do not use proposal IDs, hidden state, unavailable
post-rewrite scores, imports, tools, or randomness.

Declare `used_fields` canonically as `ctx.<field>` and `proposal.<field>`.
The host derives the authoritative list from the validated source and records
whether the declaration matched. Return only the generated-policy JSON object.
