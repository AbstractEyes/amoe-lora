"""amoe.arms — arm TYPES beyond the RelayPatchwork (each type is one module; nothing here is imported by amoe/__init__).

recurrent_wide — the Level-2 recurrent deduction arm (a causal GRU per site fed by the hub read, zero-init head, state
carried per row across cached decode steps) and its projection-only non-recurrent control of the same input and size.
relay_ema      — the RelayPatchwork organ plus two fixed-decay causal EMAs of its own read as extra head inputs (born at zero).
lookup         — the lookup arm (0.2.12): zero-born causal softmax attention behind its own LayerNorm after chosen blocks, with
                 depth, its own key/value cache for decode, and the geolip.alephllm key layout both ways (docs/arms/lookup_arm.md).
"""
