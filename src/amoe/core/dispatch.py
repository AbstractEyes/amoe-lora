"""AnchorDispatch — dense signed sinh/cosh dispatch over frozen anchors.

THE DAMPING LAW (do not "fix" this): z sums cosh over ALL anchors —
including disabled ones — and masking an anchor removes its delta
WITHOUT renormalizing. That all-anchor damped denominator is the
mechanism the campaigns measured; renormalizing on mask is a different
(uncertified) system. No top-k, no load-balancing, no auxiliary losses
(laws.py, law 2).

Telemetry: `rec` (when a list) accumulates mean |w/z| per anchor per
forward — the blend-escape gauge; `_last_shadow` holds the argmax
anchor per token — the usage gauge.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_rows(k: int, d: int) -> torch.Tensor:
    m = torch.empty(k, d)
    nn.init.orthogonal_(m)
    return F.normalize(m, dim=-1)


class AnchorDispatch(nn.Module):
    def __init__(self, anchors: nn.ModuleList, d: int,
                 emb: int = 64, tau: float = 0.1):
        super().__init__()
        self.anchors = anchors
        self.tau = tau
        A = len(anchors)
        self.register_buffer("key_proj", orthogonal_rows(emb, d))
        self.dispatch = nn.Parameter(orthogonal_rows(A, emb))
        self.enabled = [True] * A
        self.rec = None
        self._last_shadow = None

    def forward(self, x):
        emb = F.normalize(x @ self.key_proj.T, dim=-1)
        C = F.normalize(self.dispatch, dim=-1)
        u = (emb @ C.T) / self.tau
        m = u.abs().amax(dim=-1, keepdim=True)
        ep, en = torch.exp(u - m), torch.exp(-u - m)
        w = ep - en
        z = (ep + en).sum(dim=-1, keepdim=True)   # ALL anchors (law)
        out = x
        for k, a in enumerate(self.anchors):
            if not self.enabled[k]:
                continue
            out = out + (w[..., k:k + 1] / z) * (a(x) - x)
        if self.rec is not None:
            self.rec.append((w / z).detach().abs().mean(dim=(0, 1)).cpu())
        self._last_shadow = u.detach().abs().argmax(-1)
        return out


class BlockWithDispatch(nn.Module):
    def __init__(self, block: nn.Module, disp: AnchorDispatch):
        super().__init__()
        self.block = block
        self.disp = disp

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.disp(out[0]),) + out[1:]
        return self.disp(out)


def set_mask(dispatches, enabled) -> None:
    """Apply one enabled-list (or dict handled by the caller) to every
    per-block dispatch. Masking never renormalizes (law)."""
    for dp in dispatches:
        dp.enabled = list(enabled)
