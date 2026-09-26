"""RelayPatchwork adapter + the hybrid-safe block wrapper.

The certified geometry (16 slots x D=4 through a 64-atom aleph address,
squared-ReLU patch head, sigmoid gate initialized at -3): ~261k params
at d=1024. Output head zero-initialized so a fresh anchor is inert
(modulo the documented LayerNorm-bias leak).

BlockWithAdapter carries an `enabled` switch: when False, the forward
returns the block output untouched — the single-anchor half of the
toggle law. It also carries a `recompute` switch (0.2.7): when True,
the adapter's intermediates are not kept for the backward pass but
recomputed from the block output when the backward reaches them
(torch.utils.checkpoint, non-reentrant). Same math, one extra adapter
forward per backward, a fraction of the activation memory: at d=1024
on 4096-token rows the kept intermediates run ~96 MB per block per
adapter per two-row micro-batch, which is what walls a 32-block trunk
with several stacked arms on a 32 GB card. A stack of wrappers on one
block recomputes as one chain (0.2.8), keeping one residual copy per
block for the whole stack.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .address import AlephAddress


@dataclass
class AdapterSpec:
    n_slots: int = 16
    K: int = 64
    D: int = 4
    tau: float = 0.1
    hidden: int = 178
    gate_init: float = -3.0
    zero_init_head: bool = True


class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class RelayPatchwork(nn.Module):
    def __init__(self, d: int, spec: AdapterSpec | None = None):
        super().__init__()
        s = spec or AdapterSpec()
        self.spec = s
        self.n_slots = s.n_slots
        self.proj = nn.Linear(d, s.n_slots * s.D, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.addr = AlephAddress(s.K, s.D, s.tau)
        self.consume = nn.Sequential(
            nn.Linear(s.n_slots * s.D, s.hidden), SquaredReLU(),
            nn.LayerNorm(s.hidden), nn.Linear(s.hidden, d))
        if s.zero_init_head:
            nn.init.zeros_(self.consume[-1].weight)
        self.gate = nn.Parameter(torch.tensor(float(s.gate_init)))

    def forward(self, x):
        B, n, _ = x.shape
        slots = self.proj(x).view(B, n, self.n_slots, self.spec.D)
        feats = self.addr.m_hat(slots).reshape(B, n, -1)
        return x + torch.sigmoid(self.gate) * self.consume(feats)


def _apply_chain(h, *adapters):
    for a in adapters:
        h = a(h)
    return h


# The compiled chain (0.2.9). The aleph address's compiled backward is
# non-finite under bf16 autocast unless inductor rounds its fused
# intermediates to bf16 the way eager does (measured 2026-09: of the
# adapter's stages compiled alone, only addr.m_hat fails; pure fp32 and
# emulated casts are clean), so the switch below is set when the chain
# is first compiled. It is process-global; set it False before the first
# compile to opt out (not recommended for bf16 training).
COMPILE_EMULATE_CASTS = True
COMPILE_MODE = "default"        # reduce-overhead (CUDA graphs) fails inside a checkpointed chain
_COMPILED_CHAIN = None


def compiled_chain():
    global _COMPILED_CHAIN
    if _COMPILED_CHAIN is None:
        if COMPILE_EMULATE_CASTS:
            import torch._inductor.config as ic
            ic.emulate_precision_casts = True
        _COMPILED_CHAIN = torch.compile(_apply_chain, dynamic=False,
                                        mode=COMPILE_MODE)
    return _COMPILED_CHAIN


class BlockWithAdapter(nn.Module):
    """Wraps one decoder block; hybrid-safe (tuple or tensor output).

    `recompute=True` (0.2.7) routes the adapter call through a
    non-reentrant checkpoint whenever autograd is recording: the slot
    projection, the address features and the consume MLP's hidden
    activations are dropped after the forward and rebuilt from the block
    output during the backward. The head is deterministic and
    dropout-free, so the RNG state is not snapshotted (no device sync per
    call) and the recomputed values are the values the forward produced.
    Inference paths (no_grad, prefill, step) never checkpoint.

    Stacked wrappers (0.2.8; an arm attached over an arm nests a wrapper
    around a wrapper) recompute as ONE chain: the outermost wrapper with
    recompute on runs the core block once and applies every enabled
    adapter of the stack, innermost first, inside a single checkpoint, so
    one residual copy per block is kept for the whole stack instead of
    one per adapter. Each wrapper's `enabled` switch is read where it
    sits, so masks behave exactly as in the kept path.

    `compile_chain=True` (0.2.9) runs that same chain through
    torch.compile (default mode, one graph per mask pattern, shared by
    every block since the adapters enter as arguments), with or without
    the recompute; see COMPILE_EMULATE_CASTS above for the precision
    switch it needs. The compiled and the eager chain agree to bf16
    rounding, not bit for bit: a training run that switches it on owes a
    gradient census on its own card first.
    """

    def __init__(self, block: nn.Module, adapter: RelayPatchwork,
                 recompute: bool = False, compile_chain: bool = False):
        super().__init__()
        self.block = block
        self.adapter = adapter
        self.enabled = True
        self.recompute = bool(recompute)
        self.compile_chain = bool(compile_chain)

    def stack(self):
        """(wrappers innermost first, the core block) of the stack this
        wrapper tops."""
        wraps, core = [], self
        while isinstance(core, BlockWithAdapter):
            wraps.append(core)
            core = core.block
        wraps.reverse()
        return wraps, core

    def forward(self, *args, **kwargs):
        if self.compile_chain or (self.recompute and torch.is_grad_enabled()):
            wraps, core = self.stack()
            out = core(*args, **kwargs)
            live = [w.adapter for w in wraps if w.enabled]
            if not live:
                return out
            h = out[0] if isinstance(out, tuple) else out
            fn = compiled_chain() if self.compile_chain else _apply_chain
            if self.recompute and torch.is_grad_enabled():
                y = checkpoint(fn, h, *live, use_reentrant=False,
                               preserve_rng_state=False)
            else:
                y = fn(h, *live)
            return (y,) + out[1:] if isinstance(out, tuple) else y
        out = self.block(*args, **kwargs)
        if not self.enabled:
            return out
        if isinstance(out, tuple):
            return (self.adapter(out[0]),) + out[1:]
        return self.adapter(out)

    # Incremental-decode passthroughs (trunks with a cached decode path,
    # e.g. AlephLM prefill/step). The patch head is position-wise, so
    # applying it to the single new position is exact.
    def prefill(self, *args, **kwargs):
        out, cache = self.block.prefill(*args, **kwargs)
        return (self.adapter(out) if self.enabled else out), cache

    def step(self, *args, **kwargs):
        out = self.block.step(*args, **kwargs)
        if not self.enabled:
            return out
        if isinstance(out, tuple):
            return (self.adapter(out[0]),) + out[1:]
        return self.adapter(out)
