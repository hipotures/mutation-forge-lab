# Stage 7 HEG integration risk register

Pinned inputs: Mutation Forge `a6f0da20fa5a3e1c8b58cbc77a0d613c54d9f051` and
HEG `fd97451b0f3d87400d1d955a2c6b1b18303344ff` (clean, read-only).

| ID | Severity | Finding | Contract treatment | Decision status |
| --- | --- | --- | --- | --- |
| R-01 | high | Pinned HEG exposes only two-edge operators and no k=3/k=4 proposal pool/ranker. | Additive target/lane/pool seam; preserve all k values in the host contract. | blocker until implemented and reviewed |
| R-02 | high | HEG target scorer uses power-of-two cycle lengths, while the frozen Stage 2B vectors use 4,5,6,7,8,9. | Compute the frozen bounded vectors in the host; never map missing lengths to zero. | blocker until parity is implemented in HEG |
| R-03 | high | HEG transient live-frontier accounting can affect passive scheduling before a durable checkpoint; recovery does not bind the digest-derived checkpoint ID. | Persisted-only high-water and exact checkpoint-ID validation are required before policy activation. | blocker |
| R-04 | high | HEG external scorer launchers do not enforce a trusted executable digest/process-group cleanup; the Stage 1 adapter can fall back to Python scoring. | Use only the accepted worker, pin executable identity, kill/reap the process group, and fail closed. | blocker until HEG seam exists |
| R-05 | high | Direct adapter exact verification is not the M4 two-path certificate. | Pass immutable candidate snapshots to the HEG verification broker; policy never calls M4. | blocker until HEG integration tests pass |
| R-06 | medium | HEG v16→v17 migration has no in-place down migration. | Add fields additively and document Online Backup restore; no downgrade. | mitigated by contract, implementation required |
| R-07 | medium | HEG/Mutation Forge throughput telemetry has different denominators and resource assumptions. | Freeze one-thread benchmark and require a faithful HEG projection; do not invent a rate. | benchmark gate fails when projection unavailable |
| R-08 | low | Existing HEG tests emit teardown resource warnings. | Add explicit close/reap assertions to the future issue. | non-scientific maintenance |

No model, Codex App Server, provider, oracle, runtime-network, or HEG write
occurred. The authoritative decision must fail closed while any blocker above
remains unresolved.
