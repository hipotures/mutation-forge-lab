Synthesize exactly one program for the request and executable contract below.
Keep the program short and label-oblivious. Use a declared seeded `pick` for
structural ties.

Before returning, check:

1. Every selector and action identifier is exact.
2. Every enum literal is exact.
3. Every expression type and action relation is compatible.
4. Every reachable path ends in `emit` or `no_plan`.
5. Prefer a valid `no_plan` over a rewrite that routinely violates a relation.

Return a `design_summary` and a `hypothesis` of at most three sentences each.
