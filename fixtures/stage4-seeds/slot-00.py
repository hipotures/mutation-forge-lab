def priority(ctx, proposal):
    order = ctx["order"]
    broken = proposal["broken_sampled_witnesses_by_length"]
    loads = proposal["removed_edge_load_sum_by_length"]
    distance = proposal["mean_preexisting_distance_for_new_edges"]
    triangle_risk = proposal["local_triangle_risk"]
    c4_risk = proposal["local_c4_risk"]
    n = min(len(broken), len(loads))
    broken_signal = 0.0
    load_signal = 0.0
    for i in range(16):
        if i < n:
            broken_signal = broken_signal + broken[i]
            load_signal = load_signal + loads[i]
    broken_signal = broken_signal / n
    load_signal = load_signal / n
    distance_signal = distance / order
    risk_signal = (triangle_risk + c4_risk) / order
    return 4.0 * broken_signal + load_signal + distance_signal - 2.0 * risk_signal