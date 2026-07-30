def priority(ctx, proposal):
    lengths = ctx["forbidden_lengths"]
    risk = min(255, proposal["local_triangle_risk"] + proposal["local_c4_risk"])
    broken = 0
    loads = 0
    for i in range(min(16, len(lengths))):
        broken = min(3, broken + proposal["broken_sampled_witnesses_by_length"][i])
        loads = min(3, loads + proposal["removed_edge_load_sum_by_length"][i])
    score = -risk * 8 + broken + loads
    return score