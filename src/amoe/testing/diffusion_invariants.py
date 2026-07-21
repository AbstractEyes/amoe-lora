"""CPU-testable diffusion invariants on tiny stubs (no downloads).

The contracts, each tracing to a paid-for receipt:
  toggle law (code-path skip, bit-exact)        — every r2 experiment
  P-INIT (zero-init W+B exactly inert)          — exp006 bias-leak law
  detach bit-exact via binding probe            — the text line's exp011 bar
  band windows sum-to-1, smooth, positional     — exp008
  flow x0 recovery exact and linear             — exp013 (the conditioning law)
  dtype law (declared, refuses meta)            — R0b catch, fork f4f1c46
  checkpoint round trips (.pt/.safetensors/legacy shapes) — burndown record
  blob-on-eps refusal                           — exp012, 2-seed
"""
from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn


# ── tiny stubs ──────────────────────────────────────────────────────────

class _TinySite(nn.Module):
    """Marker block with a width witness; tensor output."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return x + torch.tanh(self.lin(x))


class _TinyTupleSite(_TinySite):
    def forward(self, x):
        return (x + torch.tanh(self.lin(x)), None)


class TinyUNetStub(nn.Module):
    """Scattered heterogeneous-width sites (down 8/8, mid 16, up 8) —
    the SD15 shape in miniature, one tuple-output site for the
    hybrid-safety path."""

    def __init__(self, tuple_site=False):
        super().__init__()
        torch.manual_seed(11)
        cls_last = _TinyTupleSite if tuple_site else _TinySite
        self.down = nn.ModuleList([_TinySite(8), _TinySite(8)])
        self.mid = _TinySite(16)
        self.up = nn.ModuleList([cls_last(8)])
        self.enc = nn.Linear(8, 16)
        self.dec = nn.Linear(16, 8)

    def forward(self, x):                       # x: [B, T, 8]
        for blk in self.down:
            out = blk(x)
            x = out[0] if isinstance(out, tuple) else out
        h = self.enc(x)
        out = self.mid(h)
        h = out[0] if isinstance(out, tuple) else out
        x = self.dec(h)
        out = self.up[0](x)
        return out[0] if isinstance(out, tuple) else out


class TinyBinding:
    name = "tiny_diffusion"
    expected_sites = 4

    def sites(self, model):
        from ..binding.diffusion import DiffusionBinding  # noqa: F401
        out = [(n, m, m.d) for n, m in model.named_modules()
               if isinstance(m, _TinySite)]
        assert len(out) == self.expected_sites, len(out)
        return out

    def replace(self, model, name, new):
        from ..binding.diffusion import walk_replace
        walk_replace(model, name, new)

    def declared_dtype(self, model):
        from ..binding.diffusion import _declared_dtype
        return _declared_dtype(model)

    def probe(self, model):
        g = torch.Generator().manual_seed(1400)
        dt = self.declared_dtype(model)
        x = torch.randn(2, 5, 8, generator=g).to(dt)
        with torch.no_grad():
            return model(x).detach().float().cpu()


# ── checkpoint builders ─────────────────────────────────────────────────

def _relay_ckpt(model, *, trained=True, name="t"):
    from ..diffusion.core.relay import RelayPatch2D
    from ..io.checkpoint import DiffusionAnchorCheckpoint
    b = TinyBinding()
    torch.manual_seed(23)
    adapters = {}
    widths = []
    for i, (_, _, d) in enumerate(b.sites(model)):
        r = RelayPatch2D(d, n_slots=4, K=8, hidden=12)
        if trained:
            nn.init.normal_(r.consume[-1].weight, std=0.05)
            nn.init.normal_(r.consume[-1].bias, std=0.05)
        for k, v in r.state_dict().items():
            adapters[f"{i}.{k}"] = v.clone()
        widths.append(d)
    return DiffusionAnchorCheckpoint(adapters, {
        "name": name, "adapter": {"kind": "relay"},
        "substrate": {"family": "tiny_diffusion", "n_sites": len(widths),
                      "site_names": [s[0] for s in b.sites(model)],
                      "widths": widths}})


def _mb_ckpt(model, *, trained=True):
    from ..diffusion.core.multiband import MultibandDelta
    from ..io.checkpoint import DiffusionAnchorCheckpoint
    b = TinyBinding()
    torch.manual_seed(29)
    adapters = {}
    for i, (_, _, d) in enumerate(b.sites(model)):
        m = MultibandDelta(d, r=3)
        if trained:
            for up in m.up:
                nn.init.normal_(up.weight, std=0.05)
        for k, v in m.state_dict().items():
            adapters[f"{i}.{k}"] = v.clone()
    return DiffusionAnchorCheckpoint(adapters, {
        "name": "mb", "adapter": {"kind": "multiband3"},
        "substrate": {"family": "tiny_diffusion", "n_sites": 4}})


# ── invariants ──────────────────────────────────────────────────────────

def assert_band_windows():
    from ..diffusion.core.multiband import band_of, band_weights
    s = torch.linspace(0, 1, 501)
    w = band_weights(s)
    assert (w.sum(-1) - 1).abs().max() < 1e-6, "windows do not sum to 1"
    assert (w >= 0).all() and (w <= 1).all()
    for v, b in ((0.1, 0), (0.5, 1), (0.9, 2)):
        assert band_of(v) == b
        assert int(band_weights(torch.tensor([v])).argmax()) == b
    # max slope of the cosine ramp is pi/(4*XFADE) ~= 13.1 per unit s01;
    # on a 1/500 grid that is ~0.0262 per step — assert the theory bound
    d = (w[1:] - w[:-1]).abs().max()
    assert d < 0.03, f"windows not smooth (max step {d} > theory ~0.0262)"


def assert_band_coordinate():
    """The eps band coordinate is s01 = t/1000 — the normalized DISCRETE
    timestep. Every certified bed trained on it (dexp008/011/012) and the
    proven controller gates on it (dexp010). The tempting substitute,
    1 - alphas_cumprod[t], puts t=500 in a different band entirely, which
    silently decouples training from inference. This test pins the
    convention and documents the divergence it guards against."""
    from ..diffusion.core.multiband import band_of

    # the REAL SD1.5 schedule (scaled_linear betas), not a stand-in: this
    # is the schedule the shipped eps stacks were trained against
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000,
                           dtype=torch.float64) ** 2
    acp = torch.cumprod(1.0 - betas, dim=0)

    disagreements = [t for t in range(1000)
                     if band_of(t / 1000.0) != band_of(float(1 - acp[t]))]
    assert len(disagreements) > 300, (
        f"only {len(disagreements)} timesteps disagree — the guard is not "
        "sharp enough to catch a regression; revisit it")
    # measured on this schedule: 316/1000 timesteps train a DIFFERENT band
    # under the proxy. t=300 -> trained LOW (0.300) vs proxy MID (0.410);
    # t=700 -> trained MID (0.700) vs proxy HIGH (0.918).
    assert band_of(0.300) == 0 and band_of(float(1 - acp[300])) == 1
    assert band_of(0.700) == 1 and band_of(float(1 - acp[700])) == 2

    # the trainer must use the trained coordinate
    import inspect
    from ..diffusion.train import trainer as _tr
    src = inspect.getsource(_tr.train)
    assert "t.float() / 1000.0" in src, \
        "trainer eps band coordinate is not t/1000 (band coordinate law)"
    assert "1 - acp[t]" not in src, \
        "trainer still uses the alphas_cumprod proxy for band windows"

    # and the sampler must agree with the trainer
    src_s = inspect.getsource(
        __import__("amoe.diffusion.runtime.sampler", fromlist=["x"]))
    assert "float(t) / 1000.0" in src_s, \
        "StepGatedSampler eps gate diverged from the training coordinate"


def assert_kind_classification():
    """Dispatch banks carry BOTH key_proj and down.0.weight; the bank test
    must win, or exp007/014/015 stacks load as multiband3 and blow up at
    module construction. Flat single-module cond adapters (exp002) must
    classify too."""
    from ..io.checkpoint import _sd_kind
    bank = {"key_proj.weight": None, "down.0.weight": None,
            "gates": None, "codebook": None}
    assert _sd_kind(bank) == "bank", "bank misclassified (ordering bug)"
    assert _sd_kind({"down.0.weight": None, "gates": None}) == "multiband3"
    assert _sd_kind({"down.weight": None, "gate": None}) == "mono"
    assert _sd_kind({"pos_table": None, "addr_proj.weight": None}) == "cond"
    assert _sd_kind({"addr.codebook": None, "proj.weight": None}) == "relay"
    assert _sd_kind({"nonsense": None}) == "unknown"


def assert_flow_x0_recovery():
    from ..diffusion.train.objectives import flow_pieces
    g = torch.Generator().manual_seed(5)
    lat = torch.randn(6, 4, 8, 8, generator=g)
    s = torch.rand(6, generator=g)
    noise = torch.randn(6, 4, 8, 8, generator=g)
    x_t, v = flow_pieces(lat, s, noise)
    x0 = x_t - s[:, None, None, None] * v
    assert (x0 - lat).abs().max() < 1e-6, "flow x0 recovery not linear/exact"


def assert_toggle_law_diffusion(tuple_site=False):
    from ..diffusion.runtime.attach import attach
    m = TinyUNetStub(tuple_site=tuple_site)
    b = TinyBinding()
    g = torch.Generator().manual_seed(31)
    x = torch.randn(2, 5, 8, generator=g)
    with torch.no_grad():
        base = m(x).clone()

    # relay: P-INIT (fresh zero-init anchors are EXACTLY inert)
    h = attach(m, _relay_ckpt(m, trained=False), binding=b)
    with torch.no_grad():
        assert torch.equal(m(x), base), "P-INIT broken (zero-init leak)"
    h.detach(verify=True)

    # relay: trained anchor moves the output; all_off is bit-exact
    h = attach(m, _relay_ckpt(m, trained=True), binding=b)
    with torch.no_grad():
        on = m(x)
    assert not torch.equal(on, base), "trained relay produced no delta"
    with h.all_off():
        with torch.no_grad():
            assert torch.equal(m(x), base), "toggle law violated (relay)"
    h.detach(verify=True)
    with torch.no_grad():
        assert torch.equal(m(x), base), "model altered after detach"

    # multiband: windows drive it; all_off + full lesion are bit-exact
    h = attach(m, _mb_ckpt(m, trained=True), binding=b)
    h.set_band_windows(torch.tensor([0.9, 0.1]))
    with torch.no_grad():
        on = m(x)
    assert not torch.equal(on, base), "trained multiband produced no delta"
    with h.all_off():
        with torch.no_grad():
            assert torch.equal(m(x), base), "toggle law violated (multiband)"
    with h.lesion_band(0, 1, 2):
        with torch.no_grad():
            assert torch.equal(m(x), base), "full band lesion not bit-exact"
    h.detach(verify=True)


def assert_lesion_guard():
    from ..diffusion.runtime.attach import attach
    m = TinyUNetStub()
    h = attach(m, _relay_ckpt(m), binding=TinyBinding())
    try:
        with h.lesion_band(0):
            pass
        raise AssertionError("lesion_band on a relay anchor must TypeError")
    except TypeError:
        pass
    h.detach(verify=True)


def assert_dtype_law():
    from ..binding.diffusion import _declared_dtype
    from ..diffusion.runtime.attach import attach
    m = TinyUNetStub().to(torch.bfloat16)
    b = TinyBinding()
    g = torch.Generator().manual_seed(31)
    x = torch.randn(2, 5, 8, generator=g).to(torch.bfloat16)
    with torch.no_grad():
        base = m(x).clone()
    ck = _relay_ckpt(m, trained=True)
    h = attach(m, ck, binding=b)                 # dtype=None -> declared
    for mod in h.modules():
        for p in mod.parameters():
            assert p.dtype == torch.bfloat16, "dtype law: adapter != trunk"
    with h.all_off():
        with torch.no_grad():
            assert torch.equal(m(x), base), \
                "toggle law must be dtype-independent (code-path skip)"
    h.detach(verify=True)

    class _Meta(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.empty(1, device="meta"))
    try:
        _declared_dtype(_Meta())
        raise AssertionError("declared_dtype must refuse meta params")
    except RuntimeError:
        pass


def assert_checkpoint_roundtrip():
    from ..io.checkpoint import load_diffusion_anchor
    from ..io.safetensors_io import (load_anchor_safetensors,
                                     save_anchor_safetensors)
    m = TinyUNetStub()
    ck = _relay_ckpt(m, trained=True)
    tmp = tempfile.mkdtemp()

    # .pt round trip
    p = os.path.join(tmp, "a.pt")
    ck.save(p)
    back = load_diffusion_anchor(p)
    assert back.kind == "relay" and set(back.adapters) == set(ck.adapters)
    for k in ck.adapters:
        assert torch.equal(back.adapters[k], ck.adapters[k])

    # safetensors round trip, canonical + comfy layouts, hash verified
    for layout in ("amoe", "comfy"):
        sp = os.path.join(tmp, f"a_{layout}.safetensors")
        save_anchor_safetensors(ck, sp, key_layout=layout)
        b2 = load_anchor_safetensors(sp)
        assert set(b2.adapters) == set(ck.adapters), layout
        for k in ck.adapters:
            assert torch.equal(b2.adapters[k], ck.adapters[k]), (layout, k)

    # legacy shapes: relays list / mods list / fork dict (home rebuilt)
    per = [ck.per_site(i) for i in range(4)]
    lp = os.path.join(tmp, "legacy_relays.pt")
    torch.save({"relays": per}, lp)
    assert load_diffusion_anchor(lp).kind == "relay"

    mb = _mb_ckpt(m)
    mp = os.path.join(tmp, "legacy_mods.pt")
    torch.save({"mods": [mb.per_site(i) for i in range(4)]}, mp)
    assert load_diffusion_anchor(mp).kind == "multiband3"

    fork = {str(i): {k: v for k, v in per[i].items() if k != "addr.home"}
            for i in range(4)}
    fp = os.path.join(tmp, "fork.pt")
    torch.save({"relays": fork}, fp)
    fk = load_diffusion_anchor(fp)
    assert fk.meta.get("home_reconstructed") is True, \
        "fork import must STAMP the reconstructed home"
    assert "0.addr.home" in fk.adapters


def assert_blob_on_eps_refusal():
    from ..diffusion.train.config import DiffusionTrainConfig
    from ..diffusion.train.trainer import train
    cfg = DiffusionTrainConfig(objective="eps", blob=True)
    try:
        train(None, {}, cfg)
        raise AssertionError("blob-on-eps must refuse (conditioning law)")
    except ValueError as e:
        assert "conditioning law" in str(e)


def assert_align_negative():
    from ..diffusion.train.aligner import align
    try:
        align()
        raise AssertionError("align must raise with the record")
    except NotImplementedError as e:
        assert "falsified" in str(e)


def run_all() -> None:
    assert_band_windows()
    assert_band_coordinate()
    assert_kind_classification()
    assert_flow_x0_recovery()
    assert_toggle_law_diffusion(tuple_site=False)
    assert_toggle_law_diffusion(tuple_site=True)
    assert_lesion_guard()
    assert_dtype_law()
    assert_checkpoint_roundtrip()
    assert_blob_on_eps_refusal()
    assert_align_negative()
    print("amoe.diffusion invariants: band windows, BAND COORDINATE "
          "(t/1000), kind classification, exact flow x0, toggle law "
          "(relay+multiband, tensor+tuple), P-INIT, bit-exact detach, "
          "dtype law, checkpoint round trips (.pt/.safetensors/legacy/"
          "fork), blob-on-eps refusal, align negative — ALL GREEN")


if __name__ == "__main__":
    run_all()
