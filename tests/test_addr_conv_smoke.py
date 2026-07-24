"""addressed-conv smoke suite — SHAPES, PARSE, and the REDUCTION control.

Same discipline as test_aleph_smoke.py: tiny fabricated tensors on CPU, no
full training, no downloads. The load-bearing test here is (b) — the
addressed conv must reduce EXACTLY (fp32 tol) to a plain conv under a uniform
address, because that reduction is what makes `soft − none` attributable to
the aleph read and nothing else. If it fails, the whole head-to-head is
meaningless.

    pytest tests/test_addr_conv_smoke.py
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aleph_mnist.model.addressed_conv import (             # noqa: E402
    AddressedConv2d, AddrConvSpec, ConvConfig, ConvGenHead, ConvTrunk)


def _spec(c_in=3, c_out=8, k_bank=16, k_addr=16):
    return AddrConvSpec(c_in=c_in, c_out=c_out, kernel=3, k_bank=k_bank,
                        k_addr=k_addr, n_slots=4)


# ─────────────────────────── shapes ───────────────────────────────────────
@pytest.mark.parametrize("mode", ["soft", "sign", "none", "learned", "off"])
def test_addressed_conv_forward_shape(mode):
    m = AddressedConv2d(_spec(), mode=mode)
    y = m(torch.randn(4, 3, 12, 12))
    assert y.shape == (4, 8, 12, 12) and torch.isfinite(y).all()


def test_conv_trunk_classify_shape():
    cfg = ConvConfig(channels=1, height=28, width=28, n_classes=10,
                     conv_channels=16, conv_layers=2, k_bank=8, k_addr=8)
    t = ConvTrunk(cfg)
    x = torch.randn(4, 28 * 28)
    out = t(x, labels=torch.arange(4) % 10)
    assert out.logits.shape == (4, 10) and out.loss.ndim == 0


# ─────────────── (b) THE reduction control — none == plain conv ────────────
@pytest.mark.parametrize("k_addr", [16, 64])
def test_none_reduces_to_mean_filter_conv(k_addr):
    """AddressedConv2d(none) == nn.Conv2d with the mean of the filter bank.
    fp32 summation order makes this allclose, not literal bit-exact — stated
    so no one 'fixes' it into a false failure. This REPLACES the amoe toggle
    bit-exact test as the bed's load-bearing control."""
    torch.manual_seed(0)
    m = AddressedConv2d(_spec(k_addr=k_addr), mode="none")
    x = torch.randn(4, 3, 12, 12)
    w_eff, b_eff = m.weight.mean(0), m.bias.mean(0)
    ref = F.conv2d(x, w_eff, b_eff, padding=1)
    assert torch.allclose(m(x), ref, atol=1e-5), \
        f"none must reduce to the mean-filter conv (k_addr={k_addr})"


def test_off_reduces_to_mean_filter_conv():
    torch.manual_seed(0)
    m = AddressedConv2d(_spec(), mode="off")
    x = torch.randn(4, 3, 12, 12)
    ref = F.conv2d(x, m.weight.mean(0), m.bias.mean(0), padding=1)
    assert torch.allclose(m(x), ref, atol=1e-5)


# ─────────────── (d) param-match: soft == none == sign ─────────────────────
def test_arms_are_param_matched():
    """soft/sign/none must be byte-identical in parameter count — the codebook
    is PRESENT but unread for none ('no champion if not param-matched')."""
    n = {m: sum(p.numel() for p in AddressedConv2d(_spec(), mode=m).parameters())
         for m in ("soft", "sign", "none")}
    assert n["soft"] == n["none"] == n["sign"], n


# ─────────────── (e) the address is actually used / none is inert ──────────
def test_soft_differs_from_none_but_none_is_uniform():
    torch.manual_seed(0)
    spec = _spec()
    soft = AddressedConv2d(spec, mode="soft")
    none = AddressedConv2d(spec, mode="none")
    with torch.no_grad():                     # share the bank
        none.weight.copy_(soft.weight)
        none.bias.copy_(soft.bias)
    x = torch.randn(4, 3, 12, 12)
    assert (soft(x) - none(x)).abs().max() > 1e-3, "address must change output"
    a_none = none._address(x)                  # the uniform map
    assert a_none.var().item() < 1e-12, "none address must be spatially flat"


def test_none_codebook_gets_zero_gradient():
    """none never reads the codebook, so it must receive exactly zero grad —
    the conv analogue of test_control_arm_gets_no_codebook_gradient."""
    m = AddressedConv2d(_spec(), mode="none")
    m(torch.randn(4, 3, 12, 12)).pow(2).mean().backward()
    assert m.addr.codebook.grad is None or float(m.addr.codebook.grad.norm()) == 0.0
    assert m.slot_proj.weight.grad is None


# ─────────────── (c) translation equivariance (approx, interior) ───────────
def test_translation_equivariance_interior():
    """A same-pad stride-1 addressed conv is pointwise-equivariant. Put a small
    pattern in the centre (far from every edge) and shift it by s with zeros;
    the conv response must shift by exactly s. Comparing well inside the image
    keeps both boundaries out of the 3x3 receptive field, so equivariance is
    exact there (the address is content-based but computed per-position)."""
    torch.manual_seed(0)
    m = AddressedConv2d(_spec(c_in=1, c_out=4), mode="soft").eval()
    s = 2
    x = torch.zeros(1, 1, 24, 24)
    pat = torch.randn(1, 1, 8, 8)
    x[:, :, 8:16, 8:16] = pat
    xs = torch.zeros(1, 1, 24, 24)
    xs[:, :, 8 + s:16 + s, 8 + s:16 + s] = pat          # same pattern, shifted
    y, ys = m(x), m(xs)
    # compare an interior window that neither the edge nor the shift reaches
    a, b = 6, 18
    assert torch.allclose(y[:, :, a:b, a:b],
                          ys[:, :, a + s:b + s, a + s:b + s], atol=1e-5)


# ─────────────── (g) config rejects + generative head shape ────────────────
def test_addr_conv_spec_rejects_bad_config():
    with pytest.raises(ValueError, match="k_bank"):
        AddrConvSpec(c_in=3, c_out=8, k_bank=0)
    with pytest.raises(ValueError, match="multiple"):
        AddrConvSpec(c_in=3, c_out=8, k_bank=16, k_addr=17)
    with pytest.raises(ValueError, match="odd"):
        AddrConvSpec(c_in=3, c_out=8, kernel=4)


def test_gen_head_and_generative_trunk_shape():
    head = ConvGenHead(8, n_bins=16)
    assert head(torch.randn(2, 8, 10, 10)).shape == (2, 16, 10, 10)
    cfg = ConvConfig(channels=1, height=16, width=16, objective="generate",
                     n_bins=16, conv_channels=8, conv_layers=2, k_bank=8,
                     k_addr=8)
    t = ConvTrunk(cfg)
    out = t(torch.randn(3, 16 * 16))
    assert out.logits.shape == (3, 16, 16, 16)     # (B, n_bins, H, W)
