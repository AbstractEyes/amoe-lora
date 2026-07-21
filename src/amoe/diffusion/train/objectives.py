"""Objectives — the certified loss pieces, ported verbatim.

eps  : stock epsilon-MSE on the shipped schedule (exp006 recipe).
flow : rectified-flow v-MSE with the SHIFT warp; x0 recovery is EXACT AND
       LINEAR at all sigma (x0 = x_t − σ·v) — the mechanism behind the
       conditioning law (exp013).
blob : foreground-LP-x0 coupling, weighted by the HIGH-band window
       (exp012/exp013; pays ~125–200× more on flow than eps).
roles: HP/LP frequency-reweighted band pressure (exp009 — measured
       directional-but-negligible; shipped for completeness, honest).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ── frequency filters (dexp009) ─────────────────────────────────────────

def hp(x):
    """High-pass: x − avgpool3(x) (finer-detail component)."""
    return x - F.avg_pool2d(x, 3, stride=1, padding=1)


def lp(x):
    """Low-pass: avgpool7 (coarse structure)."""
    return F.avg_pool2d(x, 7, stride=1, padding=3)


# ── eps path (dexp006) ──────────────────────────────────────────────────

def make_schedule(base_schedule_id: str, device):
    """Stock training schedule (alphas_cumprod) from the shipped
    scheduler config."""
    from diffusers import DDPMScheduler
    sch = DDPMScheduler.from_pretrained(base_schedule_id,
                                        subfolder="scheduler")
    assert sch.config.prediction_type == "epsilon", \
        sch.config.prediction_type
    return sch.alphas_cumprod.to(device)


def add_noise(lat, noise, t, acp):
    a = acp[t].sqrt()[:, None, None, None]
    s = (1 - acp[t]).sqrt()[:, None, None, None]
    return a * lat + s * noise


# ── flow path (dexp013) ─────────────────────────────────────────────────

def warp_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    return (shift * u) / (1 + (shift - 1) * u)


def flow_pieces(lat, s, noise):
    """x_t and the v target; x0 = x_t − σ·v holds exactly."""
    s4 = s[:, None, None, None]
    return noise * s4 + lat * (1 - s4), noise - lat   # x_t, v


# ── blob coupling (dexp012/013) ─────────────────────────────────────────

def blob_lp_err(x0_hat, x0, blob):
    """Foreground-masked LP-x0 error, per-sample (B,). blob: (B, H, W)
    binary mask on the latent grid."""
    d2 = (lp(x0_hat) - lp(x0)) ** 2
    m = blob[:, None]
    denom = m.sum(dim=(1, 2, 3)).clamp_min(1.0) * d2.shape[1]
    return (d2 * m).sum(dim=(1, 2, 3)) / denom


# ── role pressure (dexp009) ─────────────────────────────────────────────

def role_losses(pred, target, lam: float = 0.5):
    """Per-sample (B,) losses for each band role: LOW +HP, MID std,
    HIGH +LP."""
    base = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    low = base + lam * ((hp(pred) - hp(target)) ** 2).mean(dim=(1, 2, 3))
    high = base + lam * ((lp(pred) - lp(target)) ** 2).mean(dim=(1, 2, 3))
    return low, base, high
