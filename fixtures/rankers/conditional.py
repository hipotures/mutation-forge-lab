def priority(ctx, proposal):
    """Reviewed Stage 2A conditional probe ranker."""
    score = proposal["features"]["weight"]
    if ctx["budget_remaining"] < 4:
        score = score - proposal["features"]["penalty"]
    return score
