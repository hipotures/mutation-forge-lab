Generate one new deterministic ranker for this frozen slot.
Generation mode: new_strategy
Candidate output schema version: stage3.generated_policy.v1
Task brief: slot-02: Construct a piecewise arithmetic ranker whose principal signals are proposal-specific. Use bounded conditionals to modulate their weights from stagnation or the recent improvement, acceptance, and duplicate summaries maintained by native evaluation.
Preregistered focus: piecewise

SCIENTIFIC DECISION PROBLEM

Select legal graph rewrites likely to reduce forbidden-cycle witnesses as early as possible in a bounded search trajectory.

For one current graph, the host creates a bounded pool of already-legal k-switch proposals and calls priority(ctx, proposal) separately for every proposal.
Larger finite priorities are preferred; the host resolves equal numeric priorities deterministically by proposal_id.
Only the selected proposal is applied and authoritatively scored. The ranker never receives true post-rewrite scores for unselected proposals.

IMPORTANT WITHIN-POOL DISTINCTION

ctx describes the current graph and is identical for every proposal in the same pool. Context may modulate or normalize a ranking, but a context-only expression cannot distinguish candidates.
proposal contains candidate-specific bounded structural proxies and must provide the principal within-pool ranking signal.

CONTEXT FIELDS (POOL-CONSTANT)

- ctx.schema_version [string constant 'mforge.scientific_context.v2'; scope=pool_constant]:
  Fixed context contract literal mforge.scientific_context.v2.
  Interpretation: provenance_only.
- ctx.order [integer; minimum 4; scope=pool_constant]:
  Number of vertices in the current graph.
  Interpretation: neutral.
- ctx.forbidden_lengths [array (1..16 items; unique) of integer; minimum 1; scope=pool_constant]:
  Ordered configured cycle lengths used by the aligned context and proposal vectors.
  Interpretation: index_only.
- ctx.capped_cycle_counts [array (1..16 items) of integer; minimum 0; scope=pool_constant]:
  Current bounded witness counts aligned with forbidden_lengths; these are current-state counts, not proposal predictions.
  Interpretation: larger_is_generally_worse_but_capped.
- ctx.weighted_penalty [integer; minimum 0; scope=pool_constant]:
  Aggregate penalty of the current graph under the host scorer; identical for all proposals in the pool.
  Interpretation: larger_is_worse_for_current_state_only.
- ctx.step [integer; minimum 0; scope=pool_constant]:
  Current zero-based search-trajectory step.
  Interpretation: provenance_only.
- ctx.remaining_steps [integer; minimum 0; scope=pool_constant]:
  Number of search decisions remaining after the current step.
  Interpretation: provenance_only.
- ctx.stagnation [integer; minimum 0; scope=pool_constant]:
  Consecutive native evaluation steps without a strict accepted improvement; initialized to zero and reset after an accepted improvement.
  Interpretation: heuristic_history_only.
- ctx.recent_best_improvement [number; scope=pool_constant]:
  Largest normalized witness-count improvement over the previous eight native evaluation decisions; initialized from an empty history and updated after each decision.
  Interpretation: caller_defined_history.
- ctx.recent_acceptance_rate [number; range [0, 1]; scope=pool_constant]:
  Accepted-move fraction over the previous eight native evaluation decisions in [0,1]; initialized from an empty history and updated after each decision.
  Interpretation: heuristic_history_only.
- ctx.recent_duplicate_rate [number; range [0, 1]; scope=pool_constant]:
  Mean duplicate-proposal fraction over the previous eight native evaluation pools in [0,1]; initialized from an empty history and updated after each pool.
  Interpretation: heuristic_history_only.

PROPOSAL FIELDS (CANDIDATE-SPECIFIC OR PROVENANCE)

- proposal.schema_version [string constant 'mforge.scientific_proposal.v2'; scope=contract_constant]:
  Fixed proposal contract literal mforge.scientific_proposal.v2.
  Interpretation: provenance_only.
- proposal.proposal_id [string; pattern '^[0-9a-f]{64}$'; scope=candidate_specific]:
  Opaque deterministic SHA-256 identifier for the declarative rewrite plan.
  Interpretation: no_quality_signal.
- proposal.k [integer; allowed values: [2, 3, 4]; scope=candidate_specific]:
  Switch arity: the number of pairwise vertex-disjoint existing edges removed and new edges added; one of 2, 3, or 4.
  Interpretation: heuristic_no_guarantee.
- proposal.operator_family [string; allowed values: ['legal_2_switch', 'legal_3_switch', 'legal_4_switch']; scope=candidate_specific_alias]:
  Label legal_2_switch, legal_3_switch, or legal_4_switch, determined exactly by k.
  Interpretation: no_independent_signal.
- proposal.selector_tags [array (1..8 items) of string; allowed values: ['uniform_random', 'sampled_forbidden_cycle_anchored', 'high_sampled_witness_load', 'remote_from_anchor', 'pairwise_distant_disjoint', 'mixed_exploit_explore']; scope=candidate_specific_provenance]:
  Bounded labels describing which deterministic host selector generated the proposal; current generation emits one tag and no tag guarantees quality.
  Interpretation: heuristic_provenance_only.
- proposal.anchor_forbidden_length [null or integer; minimum 1; scope=candidate_specific_provenance]:
  Null unless sampled-forbidden-cycle anchoring was available; otherwise the first configured forbidden length with a nonempty sampled witness set. It does not guarantee that every selected edge breaks that anchor.
  Interpretation: heuristic_provenance_only.
- proposal.broken_sampled_witnesses_by_length [array (1..16 items) of integer; minimum 0; scope=candidate_specific]:
  At index i, the number of sampled source-graph cycles of length ctx.forbidden_lengths[i] touched by at least one removed edge. A cycle touched by multiple removed edges counts once.
  Interpretation: larger_may_be_better_sampled_proxy.
- proposal.removed_edge_load_sum_by_length [array (1..16 items) of integer; minimum 0; scope=candidate_specific]:
  At index i, the sum of sampled source-witness edge loads over all removed edges for ctx.forbidden_lengths[i]; multi-hit cycles can contribute more than once.
  Interpretation: larger_may_be_better_sampled_proxy.
- proposal.removed_edge_load_max_by_length [array (1..16 items) of integer; minimum 0; scope=candidate_specific]:
  At index i, the maximum sampled source-witness load among the removed edges for ctx.forbidden_lengths[i].
  Interpretation: larger_is_concentration_proxy_only.
- proposal.minimum_distance_between_removed_edges [integer; minimum 0; scope=candidate_specific]:
  Minimum pairwise edge distance between removed edges, computed by endpoint BFS distances in the original source graph.
  Interpretation: larger_means_more_separated_heuristically.
- proposal.mean_distance_between_removed_edges [number; minimum 0; scope=candidate_specific]:
  Arithmetic mean of pairwise removed-edge distances in the original source graph; for k=2 it equals the minimum because only one pair exists.
  Interpretation: larger_means_more_spread_heuristically.
- proposal.minimum_preexisting_distance_for_new_edges [integer; minimum 0; scope=candidate_specific]:
  Minimum original-source-graph BFS distance between endpoints that the proposal reconnects, measured before removing or adding edges.
  Interpretation: larger_means_more_remote_heuristically.
- proposal.mean_preexisting_distance_for_new_edges [number; minimum 0; scope=candidate_specific]:
  Mean original-source-graph BFS distance between endpoints that the proposal reconnects, measured before the rewrite.
  Interpretation: larger_means_more_remote_heuristically.
- proposal.local_triangle_risk [integer; minimum 0; scope=candidate_specific]:
  Bounded count of unique triangles around newly added edges after applying the proposal to a local cloned adjacency.
  Interpretation: larger_is_riskier_heuristically.
- proposal.local_c4_risk [integer; minimum 0; scope=candidate_specific]:
  Bounded count of unique 4-cycles around newly added edges after applying the proposal to a local cloned adjacency.
  Interpretation: larger_is_riskier_heuristically.
- proposal.reconnection_span [number; minimum 0; scope=candidate_specific_alias]:
  Exact alias of mean_preexisting_distance_for_new_edges in the current implementation.
  Interpretation: no_independent_signal.

VECTOR ALIGNMENT

- ctx.capped_cycle_counts[i] describes the current graph at cycle length ctx.forbidden_lengths[i].
- proposal.broken_sampled_witnesses_by_length[i], proposal.removed_edge_load_sum_by_length[i], and proposal.removed_edge_load_max_by_length[i] use the same ctx.forbidden_lengths[i] index.

ALIASES AND REDUNDANCIES

- proposal.k, proposal.operator_family: operator_family is exactly legal_{k}_switch and is not independent evidence.
- proposal.mean_preexisting_distance_for_new_edges, proposal.reconnection_span: the two fields are computed from the same arithmetic mean and are exact aliases.

BOUNDED-FEATURE CAVEATS

- Witness features use bounded sampled source-graph cycles and are not exhaustive.
- Distance-budget exhaustion may use graph order as a sentinel distance.
- Local-risk budget exhaustion may return partial or zero local triangle/C4 counts.
- Selector tags are bounded generator provenance, not ground-truth quality labels.

PROGRAM CONTRACT

Return source containing exactly one top-level function with this exact unannotated signature:

def priority(ctx, proposal):
    ...
    return finite_number

- The source must contain exactly one return statement, and it must be the final top-level statement in priority.
- Return a finite int or float; bool, NaN, infinity, complex values, and containers are rejected.
- Read ctx and proposal only by indexing or slicing. Do not mutate either input.
- Allowed local control flow: assignments, arithmetic, comparisons, Boolean logic, conditionals, for loops, and while loops.
- Allowed built-ins only: abs, all, any, len, max, min, range, round, sum.
- No imports, attributes or method calls, comprehensions, lambda, recursion, try, with, yield, async, decorators, annotations, classes, nested functions, reflection, dynamic execution, I/O, environment, process, network, database, RNG, or hidden state.
- Source <= 12288 bytes; AST <= 1000 nodes; per-call wall <= 0.025 seconds; total smoke wall <= 60.0 seconds.

OUTPUT FIELD REQUIREMENTS

- schema_version: exactly "stage3.generated_policy.v1".
- source: only the complete priority(ctx, proposal) function; no imports or other definitions.
- design_summary: begin with "Hypothesis:" and state why this ranking should select better mutations than an unstructured selection rule, in a falsifiable way.
- used_fields: list every accessed field exactly once as ctx.<field> or proposal.<field>.
- assumptions: explicitly list each assumed direction of effect that is not guaranteed by the field definition; use an empty array if none.

Return exactly one JSON object with keys schema_version, source, design_summary, used_fields, assumptions. Return no Markdown, commentary, benchmark claim, or extra key.
