"""AlephAddress — closed-form soft read over 2K oriented half-axes.

m_hat(x) = sum_k sinh(u_k) A_k / sum_k cosh(u_k), computed stably via
max-|u| factor-out. The codebook trains only through whatever consumes
the address (no commit/EMA/VQ). The `home` buffer snapshots the init
for drift measurement and is REQUIRED in checkpoints.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlephAddress(nn.Module):
    def __init__(self, K: int = 64, D: int = 4, tau: float = 0.1):
        super().__init__()
        self.K, self.D, self.tau = K, D, tau
        self.codebook = nn.Parameter(
            F.normalize(torch.randn(K, D), dim=-1))
        self.register_buffer("home", self.codebook.detach().clone())

    def m_hat(self, x: torch.Tensor) -> torch.Tensor:
        A = F.normalize(self.codebook, dim=-1)
        u = (F.normalize(x, dim=-1) @ A.transpose(-1, -2)) / self.tau
        m = u.abs().amax(dim=-1, keepdim=True)
        ep, en = torch.exp(u - m), torch.exp(-u - m)
        return ((ep - en) @ A) / (ep + en).sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def drift(self) -> float:
        """Mean angular drift (radians) of the codebook from `home`."""
        A = F.normalize(self.codebook, dim=-1)
        H = F.normalize(self.home, dim=-1)
        cos = (A * H).sum(-1).clamp(-1.0, 1.0)
        return float(torch.arccos(cos).mean())
