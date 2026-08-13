"""AlephLM substrate binding: auto-resolve, attach/detach roundtrip,
cached-decode passthrough. Uses a stub trunk mirroring the geolip.alephllm
AlephLM contract (blocks ModuleList, dataclass cfg with d_model, tensor-
returning blocks with prefill/step) — no external dependency, CPU-only."""
from dataclasses import dataclass

import torch
import torch.nn as nn

import amoe
from amoe.binding.resolver import resolve
from amoe.core.adapter import AdapterSpec, BlockWithAdapter, RelayPatchwork


def _fresh_ck(name, d, n_sites, spec):
    ads = {}
    for i in range(n_sites):
        a = RelayPatchwork(d, spec)
        for k, v in a.state_dict().items():
            ads[f"{i}.{k}"] = v.clone()
    return amoe.AnchorCheckpoint(ads, {"name": name})


@dataclass
class _Cfg:
    d_model: int = 32
    context: int = 64


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        return x + self.lin(x)

    def prefill(self, x):
        return x + self.lin(x), {"t": x.shape[1]}

    def step(self, x_t, cache):
        cache["t"] += 1
        return x_t + self.lin(x_t)


class _StubAlephLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = _Cfg()
        self.emb = nn.Embedding(256, self.cfg.d_model)
        self.blocks = nn.ModuleList(_Block(self.cfg.d_model)
                                    for _ in range(3))
        self.head = nn.Linear(self.cfg.d_model, 256)

    def forward(self, idx=None, input_ids=None, **kw):
        x = self.emb(idx if idx is not None else input_ids)
        for b in self.blocks:
            x = b(x)
        logits = self.head(x)
        return logits, None


def test_resolve_autodetects_alephlm():
    b = resolve(_StubAlephLM())
    assert b.name == "alephlm"
    m = _StubAlephLM()
    assert b.hidden_size(m) == 32
    assert len(b.layers(m)) == 3


def test_attach_detach_roundtrip_bit_exact():
    torch.manual_seed(0)
    m = _StubAlephLM()
    ids = torch.randint(0, 256, (1, 8))
    before, _ = m(ids)
    spec = AdapterSpec(n_slots=4, K=8, D=4, hidden=16)
    ck = _fresh_ck("t", d=32, n_sites=3, spec=spec)
    h = amoe.attach(m, ck, spec=spec)
    assert all(isinstance(b, BlockWithAdapter) for b in m.blocks)
    m(ids)  # runs wrapped
    amoe.detach(h, verify=True)   # bit-exact restore asserted inside
    after, _ = m(ids)
    assert torch.equal(before, after)


def test_cached_decode_passthrough():
    torch.manual_seed(1)
    m = _StubAlephLM()
    spec = AdapterSpec(n_slots=4, K=8, D=4, hidden=16)
    ck = _fresh_ck("t", d=32, n_sites=3, spec=spec)
    amoe.attach(m, ck, spec=spec)
    x = torch.randn(2, 5, 32)
    outs, caches = zip(*(b.prefill(x) for b in m.blocks))
    assert all(o.shape == x.shape for o in outs)
    xt = torch.randn(2, 1, 32)
    for b, c in zip(m.blocks, caches):
        y = b.step(xt, c)
        assert y.shape == xt.shape and c["t"] == 6
    # disabled adapter: step passthrough is exactly the raw block
    for b, c in zip(m.blocks, caches):
        b.enabled = False
        raw = b.block.step(xt, dict(c))
        assert torch.equal(b.step(xt, dict(c)), raw)
