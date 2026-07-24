"""antipode-conv smoke suite — the antipode must be LOAD-BEARING.

This bed exists because three prior beds discarded or mutilated the antipodal
address. The load-bearing tests here are (a) the read is ODD (so +A_k and -A_k
content push opposite ways — the antipode is used, not summed away), (b) the
`mag` control is sign-BLIND (so `soft - mag` isolates the sign), (c) the read
is a SUBSTANTIAL additive contribution (not the ~0.7% mean-perturbation the
filter-steering bed collapsed to), and (d) the consumer is sign-preserving.

    pytest tests/test_antipode_conv_smoke.py
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aleph_mnist.model.antipode_conv import (              # noqa: E402
    AntipodeConv2d, AntipodeConvTrunk, AntipodeRead, ConvTokenConfig,
    ConvTokenTrunk, SignedSquare, signed_antipode_read)


def _read(mode, d=32, k_addr=64, n_slots=16):
    return AntipodeRead(d=d, k_addr=k_addr, n_slots=n_slots, mode=mode)


# ─────────── (a) the antipode is load-bearing: the soft read is ODD ────────
def test_soft_read_is_odd():
    """m_hat(-s) == -m_hat(s): a token and its contrast-inverse produce
    opposite reads. This is the antipode being USED, not collapsed."""
    r = _read("soft")
    t = torch.randn(4, 20, 32)
    slots = r.proj(t).view(4, 20, 16, 4)
    assert torch.allclose(r._read(slots), -r._read(-slots), atol=1e-6)


def test_soft_block_contribution_is_odd_in_token():
    """The STRONGEST antipode guarantee: the whole read contribution is odd in
    the token — proj (bias-free) -> odd read -> strictly-odd consume — so a
    token and its antipode push the stream in EXACTLY opposite directions. Not
    merely preserved: the antipode IS the contribution. `mag` breaks this."""
    r = _read("soft").eval()
    t = torch.randn(3, 16, 32)
    add_pos, add_neg = r(t) - t, r(-t) - (-t)
    assert torch.allclose(add_pos, -add_neg, atol=1e-5)
    m = _read("mag").eval()
    m_pos, m_neg = m(t) - t, m(-t) - (-t)
    assert not torch.allclose(m_pos, -m_neg, atol=1e-3)   # mag discards the sign


# ─────────── (b) the mag control is sign-BLIND (isolates the sign) ─────────
def test_mag_read_is_sign_blind():
    r = _read("mag")
    t = torch.randn(4, 20, 32)
    slots = r.proj(t).view(4, 20, 16, 4)
    assert torch.allclose(r._read(slots), r._read(-slots), atol=1e-6), \
        "mag must be even in the slot sign — that is what makes soft-mag the "\
        "sign-isolating control"


def test_soft_and_mag_reads_are_scale_matched():
    """soft vs mag must differ ONLY in direction (the sign), not magnitude —
    else soft>mag would confound the sign with scale. Both reads are unit-norm
    per slot (mag's |u| otherwise shrinks its norm ~2.3x)."""
    r, rm = _read("soft"), _read("mag")
    with torch.no_grad():
        for p, q in zip(rm.parameters(), r.parameters()):
            if p.shape == q.shape:
                p.copy_(q)
    s = r.proj(torch.randn(16, 20, 32)).view(16, 20, 16, 4)
    assert torch.allclose(r._read(s).norm(dim=-1),
                          torch.ones(16, 20, 16), atol=1e-5)
    assert torch.allclose(rm._read(s).norm(dim=-1),
                          torch.ones(16, 20, 16), atol=1e-5)


# ─────────── (c) the read is a SUBSTANTIAL additive contribution ───────────
def test_read_amplitude_is_substantial():
    """||gate*consume(read)|| / ||t|| must be well above the ~0.007 the
    filter-steering bed collapsed to — the read genuinely moves the stream."""
    r = _read("soft").eval()
    t = torch.randn(4, 40, 32)
    assert r.read_amplitude(t) > 0.02
    assert _read("off").read_amplitude(t) == 0.0


def test_soft_mag_none_all_differ():
    r = _read("soft")
    t = torch.randn(4, 20, 32)
    outs = {}
    for mode in ("soft", "mag", "none"):
        m = _read(mode)
        with torch.no_grad():                     # share params
            for p, q in zip(m.parameters(), r.parameters()):
                if p.shape == q.shape:
                    p.copy_(q)
        outs[mode] = m(t)
    assert (outs["soft"] - outs["mag"]).abs().max() > 1e-3    # sign matters
    assert (outs["soft"] - outs["none"]).abs().max() > 1e-3   # read matters
    assert (outs["mag"] - outs["none"]).abs().max() > 1e-3


# ─────────── (d) the consumer preserves the sign (SignedSquare is odd) ─────
def test_signed_square_is_odd():
    f = SignedSquare()
    x = torch.randn(50)
    assert torch.allclose(f(-x), -f(x), atol=1e-6)
    assert torch.allclose(f(x), x * x.abs(), atol=1e-6)


# ─────────── param-match + none-zero-grad ──────────────────────────────────
def test_arms_param_matched_and_none_unread():
    n = {m: sum(p.numel() for p in _read(m).parameters())
         for m in ("soft", "mag", "none")}
    assert n["soft"] == n["mag"] == n["none"], n
    none = _read("none")
    none(torch.randn(4, 12, 32)).pow(2).mean().backward()
    assert none.addr.codebook.grad is None or \
        float(none.addr.codebook.grad.norm()) == 0.0


# ─────────── trunk shapes (classify + generative) ──────────────────────────
@pytest.mark.parametrize("objective,shape", [
    ("classify", (3, 10)), ("generate", (3, 16, 28, 28))])
def test_conv_token_trunk_shapes(objective, shape):
    cfg = ConvTokenConfig(channels=1, height=28, width=28, mode="soft", d=16,
                          conv_layers=2, read_layers=2, n_slots=8,
                          objective=objective, n_bins=16)
    t = ConvTokenTrunk(cfg)
    out = t(torch.randn(3, 784))
    assert out.logits.shape == shape
    # the read report exposes the amplitude gauge for a live (soft) read
    assert t.read_report(torch.randn(3, 784))["read_amp_mean"] > 0.0


def test_off_arm_has_no_read():
    cfg = ConvTokenConfig(channels=1, height=28, width=28, mode="off", d=16,
                          conv_layers=2, read_layers=2)
    t = ConvTokenTrunk(cfg)
    assert t.read_report(torch.randn(2, 784))["read_amp_mean"] == 0.0


# ═══════════ AntipodeConv2d: the ENTIRE conv IS the antipode read ══════════
def test_antipode_conv_read_is_odd():
    c = AntipodeConv2d(4, 8, kernel=3, mode="soft").eval()
    x = torch.randn(2, 4, 10, 10)
    s = c.query(x)
    B, _, H, W = s.shape
    sl = s.view(B, c.n_slots, c.d_addr, H, W).permute(0, 1, 3, 4, 2)
    r = signed_antipode_read(sl, c.addr.codebook, c.tau, "soft")
    rn = signed_antipode_read(-sl, c.addr.codebook, c.tau, "soft")
    assert torch.allclose(r, -rn, atol=1e-6)


def test_off_is_linear_soft_is_not():
    """off = out(query(x)) is a pure LINEAR conv (the plain-conv control);
    soft is NONLINEAR because the antipode read is the sole nonlinearity."""
    x1, x2 = torch.randn(2, 4, 9, 9), torch.randn(2, 4, 9, 9)
    off = AntipodeConv2d(4, 8, mode="off").eval()
    assert torch.allclose(off(2 * x1 + 3 * x2), 2 * off(x1) + 3 * off(x2),
                          atol=1e-5)
    soft = AntipodeConv2d(4, 8, mode="soft").eval()
    assert not torch.allclose(soft(2 * x1 + 3 * x2),
                              2 * soft(x1) + 3 * soft(x2), atol=1e-3)


def test_antipode_is_the_only_nonlinearity():
    """No ReLU/GELU/SiLU/SquaredReLU anywhere in the trunk — the antipode read
    is the ENTIRE conv's sole nonlinear operation. This is the whole claim."""
    import torch.nn as nn

    from aleph_mnist.model.trunk import SquaredReLU
    t = AntipodeConvTrunk(ConvTokenConfig(channels=1, height=28, width=28,
                          mode="soft", d=16, conv_layers=2, n_slots=8))
    banned = (nn.ReLU, nn.GELU, nn.SiLU, nn.ELU, nn.Tanh, SquaredReLU)
    assert not [m for m in t.modules() if isinstance(m, banned)]


def test_antipode_conv_arms_param_matched():
    n = {m: sum(p.numel() for p in AntipodeConv2d(4, 8, mode=m).parameters())
         for m in ("soft", "mag", "none")}
    assert n["soft"] == n["mag"] == n["none"], n


@pytest.mark.parametrize("objective,shape", [
    ("classify", (3, 10)), ("generate", (3, 16, 28, 28))])
def test_antipode_conv_trunk_shapes(objective, shape):
    cfg = ConvTokenConfig(channels=1, height=28, width=28, mode="soft", d=16,
                          conv_layers=2, n_slots=8, objective=objective,
                          n_bins=16)
    t = AntipodeConvTrunk(cfg)
    out = t(torch.randn(3, 784))
    assert out.logits.shape == shape
    assert t.read_report(torch.randn(3, 784))["read_amp_mean"] > 0.0
