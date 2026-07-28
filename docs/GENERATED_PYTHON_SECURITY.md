# Generated Python security boundary

Generated Python is not executed in Stage 1.

Stage 2 may accept one narrowly typed ranking function operating only on
bounded immutable observations. It must never receive a graph object, scorer,
verifier, filesystem, environment, process, database, network, imports,
reflection, native extension access, or mutable campaign state. It may return
a finite rank value or declarative rewrite selection; the host validates and
applies every rewrite.

Safety requires an AST allowlist before execution and an isolated worker with
independent CPU, wall, memory, recursion, payload, and output limits. The
coordinator must survive worker termination. Source hash, normalized AST hash,
behavior signature, validation result, resource use, and failure class must be
durable.

The Stage 2 adversarial and soak gates are enumerated in `MILESTONES.md`. A
passed safety gate limits authority; it does not prove arbitrary Python safe.
