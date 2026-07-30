You are proposing one safe ranker policy for Stage 4 search.

PROGRAM CONTRACT

Return exactly one JSON object matching stage4.generated_policy.v1. The source must define exactly priority(ctx, proposal) and no other top-level code.
Only the selected proposal is applied and authoritatively scored; never assume oracle access or inspect a full candidate pool.
safe builtins: abs, all, any, len, max, min, range, round, sum; max source bytes=12288; max AST nodes=500; per-call wall seconds=0.025; total wall seconds=60.0; allowed AST nodes: Add, And, Assign, AugAssign, BinOp, BoolOp, Break, Call, Compare, Constant, Continue, Dict, Div, Eq, Expr, FloorDiv, For, FunctionDef, Gt, GtE, If, IfExp, In, Is, IsNot, List, Load, Lt, LtE, Mod, Module, Mult, Name, Not, NotEq, NotIn, Or, Pow, Return, Slice, Store, Sub, Subscript, Tuple, UAdd, USub, UnaryOp, While, arg, arguments, keyword
Use only the supplied schema, semantic glossary, one parent policy, assigned brief, compact search-training feedback, and bounded archive context.
