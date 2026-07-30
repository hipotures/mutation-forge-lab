You write exactly one Python function priority(ctx, proposal) returning one finite number; larger is preferred. Inputs are immutable plain data with exact schemas stage2b.context.v1 and stage2b.proposal.v1. Use only the documented fields, local arithmetic/control flow and safe built-ins. No imports, attributes, I/O, network, processes, mutation, hidden state, reflection, dynamic execution, while loops, recursion, NaN or infinity. The function ranks an already-legal proposal to minimise capped forbidden-cycle witness counts; ties are resolved by proposal_id. Return only the JSON object required by the output schema.

Context schema (https://mutation-forge.invalid/schemas/stage2b-context.v1.json):
- schema_version: const='stage2b.context.v1' (required)
- order: integer, minimum=4 (required)
- forbidden_lengths: array, minItems=1, maxItems=16 (required)
- capped_cycle_counts: array, minItems=1, maxItems=16 (required)
- weighted_penalty: integer, minimum=0 (required)
- step: integer, minimum=0 (required)
- remaining_steps: integer, minimum=0 (required)
- stagnation: integer, minimum=0 (required)
- recent_best_improvement: number (required)
- recent_acceptance_rate: number, minimum=0, maximum=1 (required)
- recent_duplicate_rate: number, minimum=0, maximum=1 (required)
Alignment: forbidden_lengths and capped_cycle_counts have equal length

Proposal schema (https://mutation-forge.invalid/schemas/stage2b-proposal.v1.json):
- schema_version: const='stage2b.proposal.v1' (required)
- proposal_id: string (required)
- k: one of [2, 3, 4] (required)
- operator_family: one of ['legal_2_switch', 'legal_3_switch', 'legal_4_switch'] (required)
- selector_tags: array, minItems=1, maxItems=8 (required)
- anchor_forbidden_length:  (required)
- broken_sampled_witnesses_by_length: #/$defs/countVector (required)
- removed_edge_load_sum_by_length: #/$defs/countVector (required)
- removed_edge_load_max_by_length: #/$defs/countVector (required)
- minimum_distance_between_removed_edges: integer, minimum=0 (required)
- mean_distance_between_removed_edges: number, minimum=0 (required)
- minimum_preexisting_distance_for_new_edges: integer, minimum=0 (required)
- mean_preexisting_distance_for_new_edges: number, minimum=0 (required)
- local_triangle_risk: integer, minimum=0 (required)
- local_c4_risk: integer, minimum=0 (required)
- reconnection_span: number, minimum=0 (required)
Alignment: three count vectors align with context.forbidden_lengths

Safety limits: safe builtins: abs, all, any, len, max, min, range, round, sum; max source bytes=12288; max AST nodes=500; max static loop bound=256; per-call wall seconds=0.025; total wall seconds=60.0; allowed AST nodes: Add, And, Assign, AugAssign, BinOp, BoolOp, Call, Compare, Constant, Dict, Div, Eq, Expr, FloorDiv, For, FunctionDef, Gt, GtE, If, IfExp, In, Is, IsNot, List, Load, Lt, LtE, Mod, Module, Mult, Name, Not, NotEq, NotIn, Or, Pow, Return, Slice, Store, Sub, Subscript, Tuple, UAdd, USub, UnaryOp, arg, arguments, keyword
