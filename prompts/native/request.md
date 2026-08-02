Generate one deterministic ranker for the supplied native experiment context.

The host owns proposal legality, scoring, and verification.  Select one legal
proposal by implementing `priority(ctx, proposal) -> float` in `source`.
Use bounded arithmetic and proposal-derived signals; context may modulate
weights but may not become the sole ranking signal.  Do not use proposal IDs,
hidden state, unavailable post-rewrite scores, imports, tools, or randomness.
Return only the generated-policy JSON object.
