# Native v3 ordinary-Python post-migration roadmap

This document reviews issues #35–#42 against the completed M1–M7 migration.
It is advisory only: M8 neither reopens, closes, edits, nor implements those
issues. Dependencies must be re-authorized by an operator before work starts.

| Issue | Classification | Recommendation |
|---|---|---|
| #35 | Must be replaced | The closed step assumes matched custom JSON-DSL programs and an interpreter. Its host-owned exploration and baseline questions remain scientifically interesting, but a new issue must define ordinary-Python policies, safe-API behavior, and M2 execution. Do not reopen the DSL design. |
| #36 | Still valid with dependency update | Keep separate immutable development and validation panels, promotion shortlist, and validated archive. Replace dependency #35 and all program identity references with the accepted Python protocol and M1 identities. M1–M7 provide only the development search, not promotion validation. |
| #37 | Still valid with dependency update | Canonical episode manifests and deterministic serial shards remain useful. Base them on Python source/program/behavior identities and M3 results. Generation manifests are already implemented, but episode sharding and reconstruction are not. |
| #38 | Still valid with dependency update | A bounded persistent evaluator pool remains future work. Workers must retain M2 isolation and frozen limits, and W=1 must be semantically identical to serial M3. M1–M7 deliberately did not add a pool. |
| #39 | Still valid with dependency update | Provider/evaluator overlap, backpressure, and bounded queues remain unimplemented. Stream only M1-valid Python programs into isolated evaluators; preserve immutable slot consumption and App Server artifacts. |
| #40 | Still valid with dependency update | Single-writer persistence and semantic checkpointing remain distinct from the file-based M5/M6 resume proof. Update protocols to Python identities and retain fail-closed provenance and manifest checks. |
| #41 | Still valid with dependency update | Supervision of every apparent zero remains representation-independent. Preserve the current exact-verifier-only `VERIFIED` authority; any dual-verifier expansion needs a separate operator scientific decision and Python provenance. |
| #42 | Still valid with dependency update | Final dashboard/throughput integration and a guarded activation gate remain future work. Reuse M6 status telemetry, but do not imply that preview evidence authorizes a default switch. Overlap/pool metrics depend on revised #38–#41. |

## Already delivered by M1–M7

- ordinary-Python policy contract and identity;
- fail-closed validation and isolated execution;
- authoritative serial scientific evaluation and exact-verification seam;
- real provider-generated Python;
- deterministic two-generation selection, lineage, forks, Search Memory, and
  resume;
- guarded explicit preview with Native v2 still default;
- removal of the custom JSON DSL, interpreter, compiler, prompts, schemas, and
  routing.

These foundations update the dependency graph, but they do not silently
complete validation promotion, sharding, pools, streaming, single-writer
persistence, dual-verifier supervision, or final default activation.

## Recommended dependency reset

Create a new ordinary-Python exploration/baseline issue to replace #35 only if
its scientific objective is still desired. Then re-triage #36–#42 in order,
replacing their dependency links with accepted Python milestones. Keep the
default-switch decision outside all implementation issues and require an
explicit operator GO after the final activation evidence.
