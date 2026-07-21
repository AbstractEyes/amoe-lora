"""AlephCondAdapter — conditioning-stream address injection (research-grade
export; verbatim port of pod2/aleph_diffusion_core.py).

STATUS (read before using): exp002 certified this channel as a Law-2 honest
negative IN CONTEXT — appended to full text conditioning the address is
marginally inert (real-vs-deranged −0.0009). But addr-only steering is real
(+0.0287 after 2k steps, gates GREW) — the channel is redundant-in-context,
not dead. The open redesign target is COMPLEMENTARITY: inject where text is
not (reduced-text regimes, sigma-gated, non-text streams). Shipped as a
component for that line of work, not as a quickstart verb.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AlephCondAdapter(nn.Module):
    """Frozen [n_addr, addr_dim] address -> n_addr conditioning positions.
    Everything zero-init => tokens are EXACT zeros at init and when
    disabled (the position-presence offset is handled by the bed: SD15
    masked-append/length-toggle, Anima write-into-padding)."""

    def __init__(self, cond_dim: int, addr_dim: int = 128, n_addr: int = 32):
        super().__init__()
        self.cond_dim, self.addr_dim, self.n_addr = cond_dim, addr_dim, n_addr
        self.addr_proj = nn.Linear(addr_dim, cond_dim)
        nn.init.zeros_(self.addr_proj.weight)
        nn.init.zeros_(self.addr_proj.bias)
        self.pos_table = nn.Parameter(torch.zeros(n_addr, cond_dim))
        self.gate = nn.Parameter(torch.tensor(-3.0))
        self.enabled = True

    def assert_zero_init(self):
        assert self.addr_proj.weight.abs().max().item() == 0.0
        assert self.addr_proj.bias.abs().max().item() == 0.0
        assert self.pos_table.abs().max().item() == 0.0

    def tokens(self, addr: torch.Tensor) -> torch.Tensor:
        """addr [B, n_addr, addr_dim] -> [B, n_addr, cond_dim]."""
        assert addr.shape[-2:] == (self.n_addr, self.addr_dim), addr.shape
        if not self.enabled:
            return addr.new_zeros(*addr.shape[:-1], self.cond_dim)
        return torch.sigmoid(self.gate) * (self.addr_proj(addr) + self.pos_table)

    @classmethod
    def from_state_dict(cls, sd: dict) -> "AlephCondAdapter":
        cond_dim, addr_dim = sd["addr_proj.weight"].shape
        n_addr = sd["pos_table"].shape[0]
        m = cls(cond_dim, addr_dim=addr_dim, n_addr=n_addr)
        m.load_state_dict(sd, strict=True)
        return m
