def priority(ctx, proposal):
    broken = min(16, sum(proposal["broken_sampled_witnesses_by_length"]))
    triangle = min(16, proposal["local_triangle_risk"])
    c4 = min(16, proposal["local_c4_risk"])
    score = 8 * broken - triangle - c4
    return score