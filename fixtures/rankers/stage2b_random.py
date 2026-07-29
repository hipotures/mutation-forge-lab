def priority(ctx, proposal):
    """Reviewed deterministic pseudo-random baseline over opaque proposal IDs."""
    key = proposal["proposal_id"]
    score = 0
    if key[0] >= "8":
        score += 128
    if key[0] >= "c":
        score += 64
    if key[1] >= "8":
        score += 32
    if key[1] >= "c":
        score += 16
    if key[2] >= "8":
        score += 8
    if key[2] >= "c":
        score += 4
    if key[3] >= "8":
        score += 2
    if key[3] >= "c":
        score += 1
    return score
