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

import contextlib
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


def _autocast_uncached(device_type: str):
    """The active autocast region re-entered with its weight-cast cache OFF
    (0.2.10), for the checkpointed chain. Autocast caches the low-precision
    copies of parameters it casts; when an earlier pass inside the same
    autocast region (a no_grad pass with a member masked, as the abstention
    term runs) already cast a member's weights, the recorded forward of the
    checkpointed chain holds no cast ops for them, while the recompute,
    which re-enters autocast fresh, records the casts, and the checkpoint's
    operator-list check fails ('different metadata'; torch 2.8). With the
    cache off inside the chain both passes record the same ops. The cost is
    one bf16 cast of each live adapter's weights per block call."""
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:  # older signature: cuda only
        enabled = device_type == "cuda" and torch.is_autocast_enabled()
    if not enabled:
        return contextlib.nullcontext()
    try:
        dtype = torch.get_autocast_dtype(device_type)
    except (TypeError, AttributeError):
        dtype = torch.get_autocast_gpu_dtype() if device_type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype, cache_enabled=False)


# The compiled chain (0.2.9). The aleph address's compiled backward is
# non-finite under bf16 autocast unless inductor rounds its fused
# intermediates to bf16 the way eager does (measured 2026-09: of the
# adapter's stages compiled alone, only addr.m_hat fails; pure fp32 and
# emulated casts are clean), so the switch below is set when the chain
# is first compiled. It is process-global; set it False before the first
# compile to opt out (not recommended for bf16 training).
COMPILE_EMULATE_CASTS = True
COMPILE_MODE = "default"        # reduce-overhead (CUDA graphs) fails inside a checkpointed chain
# The recompile budget and the shape gate (0.2.11). The chain is compiled
# static (dynamic=False), so every (mask pattern x input shape) pair is one
# cache entry of `_apply_chain`, and when torch's per-code recompile limit
# (8 by default) is exceeded dynamo does not merely run that call eagerly:
# it marks the code object SKIP and the chain runs eager for the rest of the
# process, with a single warning. An evaluation that feeds the armed model
# short inputs of many lengths (probe items, chat documents) spends that
# budget in seconds. Two guards: the dynamo limits are raised to the floors
# below when the chain is first compiled, and only the first
# COMPILE_MAX_SHAPES distinct (batch, length) shapes are routed through the
# compiled chain; any other shape runs the eager chain (exact; the compile's
# gain is in the training shape, which is the first shape a training run
# presents). The budget then holds the mask patterns of the training shapes
# only (m + 1 per shape at m arms).
COMPILE_RECOMPILE_LIMIT = 64
COMPILE_ACCUMULATED_LIMIT = 4096
COMPILE_MAX_SHAPES = 2
_COMPILED_CHAIN = None
_COMPILED_SHAPES: set = set()


def _raise_dynamo_limits():
    import torch._dynamo.config as dc
    for names, floor in ((("recompile_limit", "cache_size_limit"),
                          COMPILE_RECOMPILE_LIMIT),
                         (("accumulated_recompile_limit",
                           "accumulated_cache_size_limit"),
                          COMPILE_ACCUMULATED_LIMIT)):
        for name in names:            # 2.8 carries both spellings; older torch one
            if hasattr(dc, name):
                setattr(dc, name, max(int(getattr(dc, name)), int(floor)))


def compiled_chain():
    global _COMPILED_CHAIN
    if _COMPILED_CHAIN is None:
        if COMPILE_EMULATE_CASTS:
            import torch._inductor.config as ic
            ic.emulate_precision_casts = True
        _raise_dynamo_limits()
        _COMPILED_CHAIN = torch.compile(_apply_chain, dynamic=False,
                                        mode=COMPILE_MODE)
    return _COMPILED_CHAIN


def chain_for(h):
    """The chain to run on an input of h's shape (0.2.11): the compiled
    chain for the first COMPILE_MAX_SHAPES distinct (batch, length) shapes
    seen, the eager chain for every other shape."""
    key = (int(h.shape[0]), int(h.shape[1]))
    if key in _COMPILED_SHAPES:
        return compiled_chain()
    if len(_COMPILED_SHAPES) < COMPILE_MAX_SHAPES:
        _COMPILED_SHAPES.add(key)
        return compiled_chain()
    return _apply_chain


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
            fn = chain_for(h) if self.compile_chain else _apply_chain
            if self.recompute and torch.is_grad_enabled():
                with _autocast_uncached(h.device.type):
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
