"""MultibandDelta — the certified band-assignment mechanism (verbatim port
of pod2/dexp008_multiband.py; exp008: band lesions surgical 3/3 at both
seeds, own-band damage 50–200× cross-band; exp012's role-aligned gauge:
multiband beats the matched monolith ~10% on HIGH-band foreground
structure, 2-seed).

Per site: N_BANDS rank-r zero-init deltas (W+b, own gates −3.0), combined
by smooth COSINE CROSSFADE WINDOWS on s01 — STRUCTURAL/positional gating
on the sigma axis, dense and differentiable, no selection event (law 2 of
amoe.diffusion.laws). Per-band gradients arise from the windows: a sample
at s01=0.9 trains (almost) only the HIGH expert.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..laws import BAND_EDGES, N_BANDS, XFADE


def band_weights(s01: torch.Tensor) -> torch.Tensor:
    """Smooth structural band windows: (B, N_BANDS), sum to 1 everywhere.
    Cosine crossfade of half-width XFADE around each edge — positional
    gating on the sigma axis, never comparative."""
    def ramp(x):                     # 0 below -XFADE, 1 above +XFADE, smooth
        t = ((x / XFADE).clamp(-1, 1) + 1) / 2
        return 0.5 - 0.5 * torch.cos(t * math.pi)
    e1, e2 = BAND_EDGES
    up1, up2 = ramp(s01 - e1), ramp(s01 - e2)
    low = 1 - up1
    mid = up1 * (1 - up2)
    high = up1 * up2
    return torch.stack([low, mid, high], dim=-1)


def band_of(s01: float) -> int:
    e1, e2 = BAND_EDGES
    return 0 if s01 <= e1 else (1 if s01 <= e2 else 2)


class MultibandDelta(nn.Module):
    """Three band experts per site, window-combined; per-expert enable
    flags (band lesions) + global enable (toggle law, code-path skip).
    `needs_bands` lets host hooks (e.g. diffusion-pipe's Block.forward)
    dispatch uniformly between relay and multiband modules."""

    needs_bands = True

    def __init__(self, d: int, r: int = 16):
        super().__init__()
        self.d, self.r = d, r
        self.down = nn.ModuleList(nn.Linear(d, r, bias=False)
                                  for _ in range(N_BANDS))
        self.up = nn.ModuleList(nn.Linear(r, d) for _ in range(N_BANDS))
        for dn, up in zip(self.down, self.up):
            nn.init.orthogonal_(dn.weight)
            nn.init.zeros_(up.weight)
            nn.init.zeros_(up.bias)
        self.gates = nn.Parameter(torch.full((N_BANDS,), -3.0))
        self.enabled = True
        self.band_enabled = [True] * N_BANDS

    def assert_zero_init(self):
        for up in self.up:
            assert up.weight.abs().max().item() == 0.0
            assert up.bias.abs().max().item() == 0.0

    def forward(self, x, w_bands):                  # w_bands: (B, N_BANDS)
        if not self.enabled:
            return x
        if w_bands is None:
            raise RuntimeError(
                "MultibandDelta needs band windows — set them per step "
                "(handle.set_band_windows / StepGatedSampler) or per "
                "training batch from the sampled sigmas")
        g = torch.sigmoid(self.gates)
        delta = 0
        w = w_bands.view(w_bands.shape[0],
                         *([1] * (x.ndim - 2)), N_BANDS)
        for b in range(N_BANDS):
            if not self.band_enabled[b]:
                continue
            delta = delta + g[b] * w[..., b:b + 1] * self.up[b](
                self.down[b](x))
        return x + delta if not isinstance(delta, int) else x

    @classmethod
    def from_state_dict(cls, sd: dict) -> "MultibandDelta":
        d = sd["down.0.weight"].shape[1]
        r = sd["down.0.weight"].shape[0]
        m = cls(d, r=r)
        m.load_state_dict(sd, strict=True)
        return m


class BandBlockWrap(nn.Module):
    """Wrap one trunk block with a window-consuming module. `w_bands` is
    set externally (per training batch, or per sampling step by the
    controller). Tuple outputs pass through."""

    def __init__(self, block: nn.Module, mod: nn.Module):
        super().__init__()
        self.block = block
        self.mod = mod
        self.w_bands = None

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        h = self.mod(h, self.w_bands)
        return (h,) + out[1:] if isinstance(out, tuple) else h
