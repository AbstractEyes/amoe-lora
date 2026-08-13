"""RelayPatchwork adapter + the hybrid-safe block wrapper.

The certified geometry (16 slots x D=4 through a 64-atom aleph address,
squared-ReLU patch head, sigmoid gate initialized at -3): ~261k params
at d=1024. Output head zero-initialized so a fresh anchor is inert
(modulo the documented LayerNorm-bias leak).

BlockWithAdapter carries an `enabled` switch: when False, the forward
returns the block output untouched — the single-anchor half of the
toggle law.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class BlockWithAdapter(nn.Module):
    """Wraps one decoder block; hybrid-safe (tuple or tensor output)."""

    def __init__(self, block: nn.Module, adapter: RelayPatchwork):
        super().__init__()
        self.block = block
        self.adapter = adapter
        self.enabled = True

    def forward(self, *args, **kwargs):
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
