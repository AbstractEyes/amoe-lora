"""Adapter recompute-in-backward (0.2.7): the same forward values and the
same gradients as the kept-intermediates path, bit for bit on CPU; inert
under no_grad and under a disabled wrapper; the handle switches every
wrapper; on a CUDA card the peak activation memory of a stacked pair of
arms drops. Uses the AlephLM-shaped stub trunk of test_alephlm_binding."""
import copy
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

import amoe
from amoe.core.adapter import AdapterSpec, BlockWithAdapter, RelayPatchwork


def _fresh_ck(name, d, n_sites, spec, seed=0):
    torch.manual_seed(seed)
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
    def __init__(self, d=32, n_blocks=3):
        super().__init__()
        self.cfg = _Cfg(d_model=d)
        self.emb = nn.Embedding(256, d)
        self.blocks = nn.ModuleList(_Block(d) for _ in range(n_blocks))
        self.head = nn.Linear(d, 256)

    def forward(self, idx=None, input_ids=None, **kw):
        x = self.emb(idx if idx is not None else input_ids)
        for b in self.blocks:
            x = b(x)
        return self.head(x), None


# a LIVE head (zero_init_head=False) so every adapter parameter receives a
# non-zero gradient and the comparison has teeth
SPEC = AdapterSpec(n_slots=4, K=8, D=4, hidden=16, zero_init_head=False)


def _pair(seed=0):
    torch.manual_seed(seed)
    m_eager = _StubAlephLM()
    m_rec = copy.deepcopy(m_eager)
    ck = _fresh_ck("t", d=32, n_sites=3, spec=SPEC, seed=seed + 1)
    h_eager = amoe.attach(m_eager, ck, spec=SPEC)
    h_rec = amoe.attach(m_rec, ck, spec=SPEC, recompute=True)
    return m_eager, m_rec, h_eager, h_rec


def _grads(m):
    return {n: (None if p.grad is None else p.grad.clone())
            for n, p in m.named_parameters()}


def test_recompute_matches_eager_forward_and_grads_bit_for_bit():
    m_eager, m_rec, _, _ = _pair()
    assert not any(b.recompute for b in m_eager.blocks)
    assert all(b.recompute for b in m_rec.blocks)
    ids = torch.randint(0, 256, (2, 16))
    le, _ = m_eager(ids)
    lr, _ = m_rec(ids)
    assert torch.equal(le, lr)
    le.float().pow(2).mean().backward()
    lr.float().pow(2).mean().backward()
    ge, gr = _grads(m_eager), _grads(m_rec)
    assert ge.keys() == gr.keys()
    n_live = 0
    for k in ge:
        if ge[k] is None:
            assert gr[k] is None, k
            continue
        assert torch.equal(ge[k], gr[k]), k
        n_live += int(bool(ge[k].abs().sum() > 0))
    # every adapter tensor (proj, address atoms, consume MLP, gate) carried a
    # non-zero gradient in both models
    adapter_keys = [k for k in ge if ".adapter." in k]
    assert adapter_keys and all(ge[k] is not None and ge[k].abs().sum() > 0
                                for k in adapter_keys)
    assert n_live >= len(adapter_keys)


def test_recompute_is_inert_under_no_grad_and_when_disabled():
    m_eager, m_rec, h_eager, h_rec = _pair(seed=3)
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        le, _ = m_eager(ids)
        lr, _ = m_rec(ids)
    assert torch.equal(le, lr)
    # the enabled switch comes first: a disabled wrapper returns the block
    # output whether or not recompute is on
    h_eager.set_mask({"t": False})
    h_rec.set_mask({"t": False})
    le, _ = m_eager(ids)
    lr, _ = m_rec(ids)
    assert torch.equal(le, lr)
    # cached decode never checkpoints (no autograd on that path anyway)
    x = torch.randn(1, 4, 32)
    out_e, _ = m_eager.blocks[0].prefill(x)
    out_r, _ = m_rec.blocks[0].prefill(x)
    assert torch.equal(out_e, out_r)


def test_handle_recompute_switches_every_wrapper():
    torch.manual_seed(0)
    m = _StubAlephLM()
    ck = _fresh_ck("t", d=32, n_sites=3, spec=SPEC)
    h = amoe.attach(m, ck, spec=SPEC)
    assert not any(b.recompute for b in m.blocks)
    assert h.recompute(True) == 3
    assert all(b.recompute for b in m.blocks)
    assert h.recompute(False) == 3
    assert not any(b.recompute for b in m.blocks)
    # the switch is orthogonal to the masks
    h.recompute(True)
    with h.all_off():
        assert not any(b.enabled for b in m.blocks)
        assert all(b.recompute for b in m.blocks)
    assert all(b.enabled for b in m.blocks)


def test_stacked_wrappers_each_carry_the_switch():
    torch.manual_seed(0)
    m = _StubAlephLM()
    h1 = amoe.attach(m, _fresh_ck("a1", 32, 3, SPEC, seed=1), spec=SPEC, recompute=True)
    h2 = amoe.attach(m, _fresh_ck("a2", 32, 3, SPEC, seed=2), spec=SPEC)
    outer = list(m.blocks)
    assert all(isinstance(b, BlockWithAdapter) and isinstance(b.block, BlockWithAdapter)
               for b in outer)
    assert all(b.block.recompute and not b.recompute for b in outer)
    assert h2.recompute(True) == 3
    assert all(b.recompute for b in outer)
    assert h1.recompute(False) == 3
    assert all(not b.block.recompute for b in outer)
    wraps, core = outer[0].stack()
    assert [w.adapter for w in wraps] == [outer[0].block.adapter, outer[0].adapter]
    assert isinstance(core, _Block)


def _stacked_pair(seed, recompute):
    torch.manual_seed(seed)
    m = _StubAlephLM()
    h1 = amoe.attach(m, _fresh_ck("a1", 32, 3, SPEC, seed=seed + 1), spec=SPEC, recompute=recompute)
    h2 = amoe.attach(m, _fresh_ck("a2", 32, 3, SPEC, seed=seed + 2), spec=SPEC, recompute=recompute)
    return m, h1, h2


@pytest.mark.parametrize("mask", [("a1", "a2"), ("a2",), ("a1",), ()])
def test_stacked_chain_matches_eager_under_every_mask(mask):
    """Two arms stacked (a2 over a1), the kept path vs the one-chain
    recompute, under each mask: same forward, same gradients, bit for bit."""
    m_e, e1, e2 = _stacked_pair(11, False)
    m_r, r1, r2 = _stacked_pair(11, True)
    for h, hh in ((e1, r1), (e2, r2)):
        name = h.names[0]
        h.set_mask({name: name in mask})
        hh.set_mask({name: name in mask})
    ids = torch.randint(0, 256, (2, 12))
    le, _ = m_e(ids)
    lr, _ = m_r(ids)
    assert torch.equal(le, lr)
    le.float().pow(2).mean().backward()
    lr.float().pow(2).mean().backward()
    ge, gr = _grads(m_e), _grads(m_r)
    assert ge.keys() == gr.keys()
    for k in ge:
        if ge[k] is None:
            assert gr[k] is None, k
        else:
            assert torch.equal(ge[k], gr[k]), k
    # a masked member receives no gradient in either path; a live one does
    for k in ge:
        if ".adapter." not in k:
            continue
        inner = ".block.adapter." in k          # a1 sits inside a2's wrapper
        live = ("a1" in mask) if inner else ("a2" in mask)
        got = ge[k] is not None and bool(ge[k].abs().sum() > 0)
        assert got == live, (k, live)


def test_stacked_chain_calls_every_live_adapter_once_per_forward():
    """Counted with pre-forward hooks: the recompute may stop early inside
    the last adapter once the tensor the backward asked for is rebuilt, so
    a post-forward hook would under-count it."""
    m, h1, h2 = _stacked_pair(5, True)
    calls = {"a1": 0, "a2": 0}
    hooks = [b.block.adapter.register_forward_pre_hook(lambda *_: calls.__setitem__("a1", calls["a1"] + 1))
             for b in m.blocks]
    hooks += [b.adapter.register_forward_pre_hook(lambda *_: calls.__setitem__("a2", calls["a2"] + 1))
              for b in m.blocks]
    ids = torch.randint(0, 256, (1, 8))
    logits, _ = m(ids)
    assert calls == {"a1": 3, "a2": 3}
    logits.float().pow(2).mean().backward()     # the recompute calls each once more
    assert calls == {"a1": 6, "a2": 6}
    h1.set_mask({"a1": False})
    calls.update(a1=0, a2=0)
    logits, _ = m(ids)
    logits.float().pow(2).mean().backward()
    assert calls == {"a1": 0, "a2": 6}
    for h in hooks:
        h.remove()


def test_compile_switch_is_carried_and_orthogonal():
    torch.manual_seed(0)
    m = _StubAlephLM()
    ck = _fresh_ck("t", d=32, n_sites=3, spec=SPEC)
    h = amoe.attach(m, ck, spec=SPEC)
    assert not any(b.compile_chain for b in m.blocks)
    assert h.compile_chain(True) == 3
    assert all(b.compile_chain and not b.recompute for b in m.blocks)
    assert h.compile_chain(False) == 3
    assert not any(b.compile_chain for b in m.blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")
def test_compiled_chain_matches_eager_and_stays_finite_on_cuda():
    """A stacked pair through the compiled chain (default mode, emulated
    casts) vs the eager chain under bf16 autocast: finite gradients on every
    tensor, outputs and gradients within bf16 rounding, and the same
    behaviour under a mask. Skips when the card has no working inductor
    backend (no Triton)."""
    from amoe.core import adapter as A
    dev = "cuda"
    spec = AdapterSpec(n_slots=16, K=16, D=8, hidden=256, zero_init_head=False)
    d, n_blocks, B, T = 256, 3, 2, 1024

    def build(compile_chain):
        torch.manual_seed(0)
        m = _StubAlephLM(d=d, n_blocks=n_blocks).to(dev)
        hs = [amoe.attach(m, _fresh_ck(f"a{i}", d, n_blocks, spec, seed=i), spec=spec,
                          compile_chain=compile_chain) for i in (1, 2)]
        return m, hs

    def run(m, mask_off=None):
        torch.manual_seed(3)
        ids = torch.randint(0, 256, (B, T), device=dev)
        for p in m.parameters():
            p.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = m(ids)
        logits.float().pow(2).mean().backward()
        return logits.detach().float(), _grads(m)

    m_e, _ = build(False)
    try:
        m_c, hs = build(True)
        out_c, g_c = run(m_c)
    except Exception as e:  # noqa: BLE001 - no inductor backend on this machine
        pytest.skip(f"torch.compile unavailable here: {type(e).__name__}: {str(e)[:120]}")
    out_e, g_e = run(m_e)
    assert torch.isfinite(out_c).all()
    assert torch.allclose(out_e, out_c, rtol=5e-2, atol=5e-2)
    for k in g_e:
        if g_e[k] is None:
            assert g_c[k] is None, k
            continue
        assert torch.isfinite(g_c[k]).all(), k
        scale = g_e[k].abs().max().clamp_min(1e-12)
        assert float((g_e[k] - g_c[k]).abs().max() / scale) < 5e-2, k
    # a masked member: no gradient reaches it through the compiled chain
    hs[0].set_mask({"a1": False})
    _, g_m = run(m_c)
    inner = [k for k in g_m if ".block.adapter." in k]
    assert inner and all(g_m[k] is None or float(g_m[k].abs().sum()) == 0.0 for k in inner)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")
def test_recompute_cuts_peak_memory_of_stacked_arms_on_cuda():
    """A 4-block trunk at d=512 on 2048-token rows with two stacked arms
    (the LAWFUL_16x8 geometry): forward + backward under bf16 autocast,
    the kept path vs the recompute path. The recompute path peaks lower
    and returns the same gradients."""
    dev = "cuda"
    spec = AdapterSpec(n_slots=16, K=16, D=8, hidden=256, zero_init_head=False)
    d, n_blocks, B, T = 512, 4, 2, 2048

    def build(recompute):
        torch.manual_seed(0)
        m = _StubAlephLM(d=d, n_blocks=n_blocks).to(dev)
        for i in (1, 2):
            amoe.attach(m, _fresh_ck(f"a{i}", d, n_blocks, spec, seed=i), spec=spec,
                        recompute=recompute)
        return m

    def run(m):
        torch.manual_seed(7)
        ids = torch.randint(0, 256, (B, T), device=dev)
        for p in m.parameters():
            p.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = m(ids)
        logits.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return peak, _grads(m)

    m_e = build(False)
    peak_e, g_e = run(m_e)
    del m_e
    torch.cuda.empty_cache()
    m_r = build(True)
    peak_r, g_r = run(m_r)
    assert g_e.keys() == g_r.keys()
    for k in g_e:
        if g_e[k] is None:
            assert g_r[k] is None
            continue
        assert torch.allclose(g_e[k], g_r[k], rtol=1e-4, atol=1e-6), k
    assert peak_r < 0.85 * peak_e, (peak_e / 2**20, peak_r / 2**20)
