# HEG score-worker duty-cycle attribution

Date: 2026-08-11

This report is provider-free. It does not change the scientific workload,
fitness, ranking, cache, verification, or sandbox semantics.

The measured checkout started at Mutation Forge
`2ae025f00e2dc3f375561fe16fc8582a26b95b79` and HEG
`27cbec9c2307b6ea5f936f858821d11d808b68f3` with a clean HEG worktree.

## Result

The observed `top` values of roughly 55–58% are not a sustained measure of
useful C++ cycle-count work over an evaluator case. An exact replay of a
retained ordinary-Python candidate used 2.18% C++ process CPU and reported
2.38% C++ cycle-search wall time. A warm worker issued no requests and consumed
0.00 CPU seconds during a separate 10.000-second idle interval, so there is no
idle busy-spin.

The largest current evaluator cost is canonical graph identity computation.
The serial evaluator computes 98 canonical hashes in one horizon-32 case,
including repeated identities for unchanged incumbents. Native v2 did not
compute canonical graph identities in its per-step trajectory.

The live `top` observation is therefore a short-window scheduling sample of
bursty scorer requests, not the case-wide worker duty cycle. The current host
does substantial work between those bursts. Under CPU contention, `top` can
show each runnable scorer at a similar partial-core percentage without that
percentage representing its share of the full evaluator case.

## Exact retained-candidate replay

The replay used:

- retained source:
  `workspace/exp_test_004/sources/418eb3b35a2267bf58c54732409efaacfcef8e9bdc1652eef3dec68da43a5489.py`;
- order 79, graph seed 401, policy seed 4003;
- horizon 32 and witness cap 64;
- 49 score attempts, 44 unique scores, and 5 score-cache hits;
- zero provider, model, or App Server calls.

The case took 0.916330 seconds. The buckets are disjoint and close to 100%:

| Component | Current wall | Current % |
| --- | ---: | ---: |
| C++ cycle search | 0.021810 s | 2.38% |
| Score-worker protocol | 0.005712 s | 0.62% |
| Score-evidence host work | 0.024146 s | 2.64% |
| Score-cache hits | 0.001520 s | 0.17% |
| Policy execution CPU | 0.000000 s | 0.00% |
| Policy IPC, framing, and scheduling | 0.041050 s | 4.48% |
| Safe API host work | 0.086876 s | 9.48% |
| Rewrite application and validation | 0.012025 s | 1.31% |
| Canonical graph hashing | 0.545596 s | 59.54% |
| Policy-worker startup | 0.035039 s | 3.82% |
| Evaluator bookkeeping residual | 0.142555 s | 15.56% |
| **Total** | **0.916330 s** | **100.00%** |

The Linux process counter measured 0.020 seconds of C++ CPU, or 2.18% of case
wall. Its 10 ms clock resolution makes that consistent with the C++ worker's
reported 0.021810 seconds.

The residual contains serial-evaluator control flow, state hashes, result and
semantic-trace construction, source validation, and teardown. No exact
verification ran in this case. Counterexample artifact persistence was not
triggered.

Two synthetic controls produced the same ordering of costs:

| Case | Total | C++ search | Canonical hashing | Safe API |
| --- | ---: | ---: | ---: | ---: |
| order 37 | 1.150216 s | 0.52% | not separately sampled | 10.23% |
| order 108 | 1.433001 s | 0.57% | 69.17% | 7.08% |

The synthetic policies saturated the witness cap quickly, so they are controls
rather than substitutes for the retained candidate replay.

## Retained 12-worker run

`exp_test_004` completed 7,680 of 7,680 evaluations with 12 evaluator workers:

- evaluator busy time was 13,099.678 seconds, or 92.09% of
  `12 × 1,185.348` seconds of evaluator capacity;
- the final queue was empty after reaching a peak of 2,656 jobs;
- persistence used 255.262 seconds, 21.53% of campaign wall, but runs in the
  coordinator and overlaps evaluator work;
- provider wait and activity also overlap evaluator work and do not explain
  an individual scorer's duty cycle;
- no provider process restarts, thread-resume attempts, or transport retries
  occurred.

The evaluators were not starved for work. The remaining 7.91% of evaluator
capacity includes process startup, scheduling boundaries, and final drain.
Aggregate job queue-wait time is not a worker-idle metric because thousands of
queued jobs wait concurrently.

Across 64 retained candidate cases:

- mean policy sandbox wall was 0.215229 seconds per case;
- mean selector and action wall were 0.113441 and 0.018839 seconds;
- all 2,048 steps produced rewrites;
- policy workers had no failures or rotations;
- score attempts averaged 35.484 and unique scores averaged 35.094;
- 9,792 retained score components reported 0.530064 seconds total C++ search.

The 640 random and 640 structural baseline cases were somewhat more
score-intensive:

| Workload | Mean score attempts | C++ component time/case |
| --- | ---: | ---: |
| Candidate sample | 35.484 | 0.008282 s |
| Random baseline | 40.731 | 0.014603 s |
| Structural baseline | 41.583 | 0.015173 s |

Baseline runtime profiles are not retained, so a complete baseline wall-time
denominator cannot be reconstructed from artifacts. Their C++ component work
is still far too small to support interpreting 55–58% from `top` as a
case-wide useful-compute fraction.

## Native v2 comparison

Native v2 reference commit `255d55b` used
`experiment/evaluation.py:_trajectory`:

1. generate and validate a graph;
2. build a host `KSwitchPoolGenerator` proposal pool;
3. ask the persistent `SourceRanker` policy worker to rank it;
4. apply the selected rewrite;
5. call `backend.score`;
6. compare scalar ordering keys.

That loop did not compute canonical graph hashes for every incumbent,
candidate, and post-step state. Current ordinary-Python evaluation adds:

- isolated `propose(ctx, graph, api, seed)` execution;
- policy IPC and Safe API capability calls;
- rewrite minting and host validation;
- conservative interval score-evidence assembly and possible expanded retries;
- immutable semantic traces containing canonical identities;
- per-case sandbox startup and immutable artifact construction.

The measured sandbox-specific share in the retained replay is not the primary
cause: policy IPC/scheduling, Safe API, and startup total 17.78%, while
canonical hashing alone is 59.54%. Score protocol overhead is only 0.62%,
cache hits 0.17%, and rewrite validation 1.31%.

| Component | Current replay | Native v2 comparison |
| --- | ---: | --- |
| C++ useful cycle search | 2.38% | `top` was reported near 95%, but no comparable case-wall profile was retained |
| Canonical graph identities | 59.54% | absent from the per-step v2 trajectory |
| Ordinary-Python sandbox, API, and startup | 17.78% | replaced v2 proposal-pool ranking; timing is not directly comparable |
| Score-worker protocol | 0.62% | same persistent pipe architecture, without retained phase timing |
| Persistence | 21.53% of campaign wall, overlapping | v2 persistence architecture differed; not an evaluator-case bucket |

## Recommendation

The smallest semantics-preserving optimization to evaluate next is an
evaluator-local cache of `GraphIdentity` by immutable `GraphState`, or carrying
the already computed identity alongside the incumbent and candidate. It would
remove repeated canonical labeling without changing any recorded identity,
fitness, acceptance, ranking, graph workload, score cutoff, cache semantics,
or verification rule.

This optimization is deliberately not implemented here. It should first get a
trace-parity test proving byte-identical scientific results and identities.

## Reproduction

The gated profiler is:

```console
uv run python scripts/profile_native_v3_python_evaluator.py \
  --heg-repo ../heg \
  --source workspace/exp_test_004/sources/418eb3b35a2267bf58c54732409efaacfcef8e9bdc1652eef3dec68da43a5489.py \
  --order 79 \
  --graph-seed 401 \
  --policy-seed 4003 \
  --horizon 32 \
  --witness-cap 64
```

It enables timing only in the standalone profiler. No per-event profiling was
added to the production evaluator hot path.
