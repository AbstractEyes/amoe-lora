"""RelayEMA — the RelayPatchwork organ (core/adapter.py) plus ONE mechanism: two fixed-decay causal EMAs of the arm's own
aleph read, fed to the head through columns that are ZERO at birth — the minimal memory extension within the organ's own grammar:
    slots_t = proj(x_t)                (B, T, n_slots x D)   — the organ's own projection, orthogonal init
    f_t     = addr.m_hat(slots_t)      (B, T, nD)            — the organ's own reconstructive aleph read (no selector)
    F1_t = (1 - r1) F1_{t-1} + r1 f_t ;  F2_t likewise       — fixed decays r1 = 1/16, r2 = 1/64 (chunked closed form in
                                                                training, exact one-step update in cached decode)
    y_t  = x_t + sigmoid(gate) * consume(cat(f_t, F1_t, F2_t))
consume = Linear(3 nD -> hidden) SquaredReLU LayerNorm Linear(hidden -> d): the organ's own head, first layer widened; the added
2 nD input columns are zeroed at birth, so at birth the forward EQUALS RelayPatchwork with the shared weights, and the write head
is zero-born as always. No softmax / argmax / top-k anywhere. Trained-arm record and birth-parity certification ship with the
published mini-beatrix-2s arms: huggingface.co/AbstractPhil/mini-beatrix-2s, arms/btx_e003."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from amoe.binding.resolver import resolve
from amoe.core.adapter import AdapterSpec, RelayPatchwork
from amoe.runtime.attach import _probe
from .recurrent_wide import RecurrentHandle

RHO1, RHO2 = 1.0 / 16.0, 1.0 / 64.0
EMA_CHUNK = 256


@dataclass
class RelayEMASpec:
    base: AdapterSpec = None                  # the organ's spec (WIDE-1024: n32 / K64 / D4 / h256)
    rho1: float = RHO1
    rho2: float = RHO2


def ema_chunked(f: torch.Tensor, rho: float, s0: torch.Tensor, chunk: int = EMA_CHUNK):
    """Causal EMA over dim 1, closed form per chunk (renormalized so q^-t never overflows; autograd-safe).
    F_t = q^t (s0 + rho * sum_{s<=t} q^-s f_s), q = 1 - rho; state carried between chunks. -> (F (B,T,C), F_last (B,C))."""
    B, T, C = f.shape; q = 1.0 - rho; outs = []; s = s0
    for a in range(0, T, chunk):
        fb = f[:, a:a + chunk]; t = fb.shape[1]
        idx = torch.arange(1, t + 1, device=f.device, dtype=f.dtype)
        acc = torch.cumsum(fb * torch.pow(q, -idx).view(1, t, 1), dim=1)
        Fb = torch.pow(q, idx).view(1, t, 1) * (s.unsqueeze(1) + rho * acc)
        s = Fb[:, -1]; outs.append(Fb)
    return torch.cat(outs, dim=1), s


class RelayEMA(nn.Module):
    def __init__(self, d: int, spec: RelayEMASpec | None = None):
        super().__init__()
        sp = spec or RelayEMASpec(); base = sp.base or AdapterSpec()
        self.spec = sp; self.base = base; self.nD = base.n_slots * base.D
        organ = RelayPatchwork(d, base)                       # the certified organ builds itself...
        self.proj, self.addr, self.gate = organ.proj, organ.addr, organ.gate
        self.consume = nn.Sequential(nn.Linear(3 * self.nD, base.hidden), organ.consume[1], organ.consume[2], organ.consume[3])
        with torch.no_grad():                                  # ...and the widened first layer starts AS the organ's
            self.consume[0].weight[:, :self.nD] = organ.consume[0].weight
            self.consume[0].weight[:, self.nD:] = 0.0
            self.consume[0].bias.copy_(organ.consume[0].bias)

    @property
    def recurrent(self) -> bool:
        return True

    def feats(self, x):
        B, n, _ = x.shape
        slots = self.proj(x).view(B, n, self.base.n_slots, self.base.D)
        return self.addr.m_hat(slots).reshape(B, n, -1)

    def run(self, x, state=None):
        """-> (y = x + write, (F1_last, F2_last)). state = None starts both EMAs at zero (fresh context)."""
        f = self.feats(x); B = f.shape[0]
        s1 = state[0] if state is not None else f.new_zeros(B, self.nD)
        s2 = state[1] if state is not None else f.new_zeros(B, self.nD)
        F1, s1 = ema_chunked(f, self.spec.rho1, s1); F2, s2 = ema_chunked(f, self.spec.rho2, s2)
        y = x + torch.sigmoid(self.gate) * self.consume(torch.cat([f, F1, F2], dim=-1))
        return y, (s1, s2)

    def forward(self, x, state=None):
        return self.run(x, state)[0]


class BlockWithEMA(nn.Module):
    """BlockWithAdapter's shape (core/adapter.py:62-92) + the EMA state carried across prefill/step; .arm for RecurrentHandle."""

    def __init__(self, block: nn.Module, arm: RelayEMA):
        super().__init__()
        self.block = block; self.arm = arm; self.enabled = True; self._state = None

    def reset_state(self):
        self._state = None

    def unpatch(self):
        """RecurrentHandle.detach calls w.unpatch(); BlockWithEMA installs no hooks — a no-op."""

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if not self.enabled:
            return out
        h = out[0] if isinstance(out, tuple) else out
        y, _ = self.arm.run(h, None)                          # a full forward is a fresh context (training rows are whole rows)
        return (y,) + out[1:] if isinstance(out, tuple) else y

    def prefill(self, *args, **kwargs):
        out, cache = self.block.prefill(*args, **kwargs)
        if not self.enabled:
            self._state = None
            return out, cache
        y, st = self.arm.run(out, None)
        self._state = st
        return y, cache

    def step(self, *args, **kwargs):
        out = self.block.step(*args, **kwargs)
        if not self.enabled:
            return out
        h = out[0] if isinstance(out, tuple) else out
        y, st = self.arm.run(h, self._state)
        self._state = st
        return (y,) + out[1:] if isinstance(out, tuple) else y


def attach_relay_ema(model, name: str, spec: RelayEMASpec | None = None, *, seed: int | None = None,
                     binding: str = "alephlm", zero_bias: bool = True) -> RecurrentHandle:
    """attach_recurrent's flow (arms/recurrent_wide.py:261-289) with RelayEMA + BlockWithEMA; bias-zeroed head like attach_wide."""
    b = resolve(model, binding)
    layers = list(b.layers(model)); d = b.hidden_size(model)
    fingerprint = _probe(model, b)
    if seed is not None:
        torch.manual_seed(int(seed))
    spec = spec or RelayEMASpec()
    blocks, wraps = [], []
    for layer in layers:
        arm = RelayEMA(d, spec).to(next(layer.parameters()).device)
        if zero_bias:
            with torch.no_grad():
                arm.consume[-1].bias.zero_()
        w = BlockWithEMA(layer, arm)
        blocks.append(w); wraps.append(w)
    b.set_layers(model, blocks)
    return RecurrentHandle(model, b, layers, [name], wraps, fingerprint)
