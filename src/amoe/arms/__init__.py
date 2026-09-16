"""amoe.arms — arm TYPES beyond the RelayPatchwork (each type is one module; nothing here is imported by amoe/__init__).

recurrent_wide — the Level-2 recurrent deduction arm (a causal GRU per site fed by the hub read, zero-init head, state
carried per row across cached decode steps) and its projection-only non-recurrent control of the same input and size.
"""
