def priority(ctx, proposal):
    return proposal.__class__.__mro__
