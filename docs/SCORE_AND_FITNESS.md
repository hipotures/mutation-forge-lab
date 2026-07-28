# Score and fitness

For forbidden lengths `4, 8, 16, ... <= order`, HEG reports
`c_l(G) = min(actual_count, witness_cap)`. `GraphScore` preserves validity,
the capped `(length, count)` vector, its total, HEG's shorter-cycle-weighted
penalty, completeness, and HEG's deterministic ordering key. A capped result
is incomplete. Worker failure is a failure, never an empty witness vector.

A heuristic total of zero triggers exact verification. Only the exact verifier
may return `VERIFIED`; the harness never labels heuristic zero as a verified
counterexample.

Each episode records initial, best, and final scores; best-so-far total curve;
normalized AUC; first improvement; exact-zero submissions; legal, invalid,
no-op, and duplicate rates; policy latency; score failures; wall status; and
the final graph.

The duplicate rate uses a label-sensitive SHA-256 graph6 state hash. Stage 1
rewrites preserve vertex labels, and duplicates do not gate controller
acceptance. Isomorphism-canonical hashes remain reserved for immutable dataset
and final result identities.

The normalized best-so-far AUC is:

```text
sum(best_total_after_each_evaluation)
------------------------------------------------
max(1, initial_total) * completed_evaluation_count
```

Aggregate fitness minimizes this frozen lexicographic vector:

1. negative exact-verified count;
2. failure episode count;
3. median best total divided by `max(1, initial total)`;
4. median best weighted penalty divided by `max(1, initial weighted penalty)`;
5. median normalized best-so-far AUC;
6. timeout rate;
7. invalid-or-no-op rate;
8. median policy call milliseconds per evaluation;
9. normalized AST node count (zero for fixed Stage 1 baselines).

Timeouts and scoring failures increment failure episodes and cannot be
reported as successful completion. Timing is reported but excluded from the
canonical deterministic summary hash. The optional runtime profile follows
the same rule; its phase, accounted, and unattributed times cannot affect
trajectory or fitness identity.
