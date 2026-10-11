"""The lookup arm (amoe.arms.lookup) and arm-key ownership (amoe.train.ownership), on a stub decoder that mirrors the
AlephLM contract (blocks ModuleList, dataclass cfg with d_model, blocks with prefill / step). CPU, fp32, no downloads.

The last test is the PARITY read against the real geolip.alephllm trunk and a trained Beatrix lookup arm: it runs only
when AMOE_LOOKUP_PARITY_CKPT names a local checkpoint of a lookup-arm craft (the hub's step_*.safetensors, trunk and arm
keys together) and geolip.alephllm imports; otherwise it is skipped. Run as a script to execute every test in order.
"""
from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn

from amoe.arms.lookup import (LookupArm, LookupSpec, attach_from_alephllm, attach_lookup, attach_saved_lookup,
                              from_alephllm_state, load_lookup, spec_from_alephllm_state, to_alephllm_state)
from amoe.train.ownership import arm_optimizer, assert_trunk_identical, freeze_except, trunk_fingerprint

torch.manual_seed(0)
D, H, V, L = 32, 4, 64, 4


@dataclass
class _Cfg:
    d_model: int = D
    context: int = 64
    name: str = "stub"


class _Block(nn.Module):
    """A position-wise block (so its prefill / step are trivially exact); the arm supplies the cross-position part."""

    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x, **kw):
        return x + torch.tanh(self.lin(x))

    def prefill(self, x):
        return self.forward(x), {"t": x.shape[1]}

    def step(self, x_t, cache):
        cache["t"] += 1
        return self.forward(x_t)


class _StubLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = _Cfg()
        self.emb = nn.Embedding(V, D)
        self.blocks = nn.ModuleList(_Block(D) for _ in range(L))
        self.head = nn.Linear(D, V)

    def forward(self, idx=None, input_ids=None, **kw):
        x = self.emb(idx if idx is not None else input_ids)
        for b in self.blocks:
            x = b(x)
        return self.head(x), None

    @torch.no_grad()
    def prefill(self, idx):
        x = self.emb(idx)
        caches = []
        for b in self.blocks:
            x, c = b.prefill(x)
            caches.append(c)
        return self.head(x[:, -1:]), {"layers": caches}

    @torch.no_grad()
    def step(self, next_id, cache):
        x = self.emb(next_id).unsqueeze(1)
        for b, c in zip(self.blocks, cache["layers"]):
            x = b.step(x, c)
        return self.head(x)


def _lm(seed=0):
    torch.manual_seed(seed)
    return _StubLM().eval()


def _ids(n=24, b=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (b, n), generator=g)


def _logits(m, ids):
    with torch.no_grad():
        return m(ids)[0].clone()


def _wake(handle, scale=0.5, seed=3):
    """Give the zero-born output projections a random value, so the arm writes."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for arm in handle.arms():
            for layer in arm.layers:
                layer.o.weight.copy_(torch.randn(layer.o.weight.shape, generator=g) * scale / D ** 0.5)


SPEC = LookupSpec(heads=H, depth=1)
SPEC2 = LookupSpec(heads=H, depth=2)


def test_birth_identity_bit_exact():
    for spec in (SPEC, SPEC2, LookupSpec(heads=H, depth=1, qk_norm="")):
        m = _lm()
        ids = _ids()
        bare = _logits(m, ids)
        h = attach_lookup(m, [0, 2], spec, seed=7)
        assert torch.equal(_logits(m, ids), bare)
        assert h.sites == [0, 2] and len(h.arms()) == 2 and h.arms()[0].depth == spec.depth
        h.detach(verify=True)


def test_toggle_and_detach_bit_exact():
    m = _lm()
    ids = _ids()
    bare = _logits(m, ids)
    h = attach_lookup(m, [1, 3], SPEC2, seed=7)
    _wake(h)
    armed = _logits(m, ids)
    assert not torch.equal(armed, bare), "a woken arm must write"
    with h.all_off():
        assert torch.equal(_logits(m, ids), bare)
    assert torch.equal(_logits(m, ids), armed)
    h.detach(verify=True)
    assert torch.equal(_logits(m, ids), bare)


def test_arm_is_causal_and_depends_on_the_past():
    m = _lm()
    h = attach_lookup(m, [0], SPEC, seed=7)
    _wake(h, scale=1.0)
    ids = _ids(n=16)
    a = _logits(m, ids)
    other = ids.clone()
    other[:, 12:] = (other[:, 12:] + 1) % V
    b = _logits(m, other)
    assert torch.equal(a[:, :12], b[:, :12]), "a later byte must not change earlier logits (causal)"
    other2 = ids.clone()
    other2[:, 0] = (other2[:, 0] + 1) % V
    c = _logits(m, other2)
    assert not torch.equal(a[:, 12:], c[:, 12:]), "the arm must carry the past forward"


def test_prefill_and_steps_match_the_forward():
    m = _lm()
    h = attach_lookup(m, [0, 2], SPEC2, seed=7)
    _wake(h, scale=1.0)
    ids = _ids(n=20, b=2)
    full = _logits(m, ids)
    t0 = 9
    lg, cache = m.prefill(ids[:, :t0])
    outs = [lg]
    for t in range(t0, ids.shape[1]):
        outs.append(m.step(ids[:, t], cache))
    inc = torch.cat(outs, dim=1)
    ref = full[:, t0 - 1:]
    err = (inc - ref).abs().max().item()
    assert torch.allclose(inc, ref, atol=1e-5, rtol=1e-5), f"decode drifts from the forward: max |delta| {err:.2e}"
    h.reset_state()
    try:
        m.step(ids[:, 0], cache)
        raise AssertionError("step after reset_state must refuse")
    except RuntimeError:
        pass
    h.detach(verify=True)


def test_depth_grows_from_a_shallower_state_bit_exact():
    m = _lm()
    ids = _ids()
    h1 = attach_lookup(m, [1, 2], SPEC, seed=7)
    _wake(h1)
    shallow = _logits(m, ids)
    state = h1.state_dict()
    h1.detach(verify=True)
    h2 = attach_lookup(m, [1, 2], SPEC2, state=state)
    assert all(arm.depth == 2 for arm in h2.arms())
    assert torch.equal(_logits(m, ids), shallow), "a deeper arm born from a shallower state is that arm"
    for arm in h2.arms():
        assert float(arm.layers[1].o.weight.detach().abs().sum()) == 0.0
    try:
        attach_lookup(_lm(), [1, 2], SPEC, state=h2.state_dict())
        raise AssertionError("a deeper state must not load into a shallower arm")
    except KeyError:
        pass
    h2.detach(verify=True)


def test_ownership_only_the_arm_moves():
    m = _lm()
    ids = _ids()
    h = attach_lookup(m, [0, 3], SPEC2, seed=7)
    frozen, kept = freeze_except(m, h.parameters())
    assert kept == len(h.parameters()) and frozen == sum(1 for _ in m.parameters()) - kept
    before = trunk_fingerprint(m, exclude=h.parameters())
    opt = arm_optimizer(h.parameters(), lr=1e-2)
    assert type(opt).__name__ == "Adam" and opt.param_groups[0]["weight_decay"] == 0.0
    o_before = [l.o.weight.clone() for a in h.arms() for l in a.layers]
    for _ in range(2):
        opt.zero_grad()
        logits, _ = m(ids)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
    assert_trunk_identical(before, trunk_fingerprint(m, exclude=h.parameters()))
    moved = [not torch.equal(w, l.o.weight) for w, (a, l) in zip(o_before, [(a, l) for a in h.arms() for l in a.layers])]
    assert all(moved), "every layer's output projection must move under the task gradient"
    with torch.no_grad():
        m.blocks[1].block.lin.weight.add_(1e-3) if hasattr(m.blocks[1], "block") else m.blocks[1].lin.weight.add_(1e-3)
    try:
        assert_trunk_identical(before, trunk_fingerprint(m, exclude=h.parameters()))
        raise AssertionError("a changed trunk tensor must be caught")
    except AssertionError as e:
        assert "differ" in str(e)
    h.detach(verify=False)


def test_save_load_roundtrip(tmp_path=None):
    import tempfile
    d = tmp_path or tempfile.mkdtemp()
    m = _lm()
    ids = _ids()
    h = attach_lookup(m, [0, 2], SPEC2, name="probe", seed=7)
    _wake(h)
    armed = _logits(m, ids)
    p = h.save(os.path.join(str(d), "lookup.pt"), meta={"base_model_id": "stub"})
    adapters, meta, spec, sites = load_lookup(p)
    assert sites == [0, 2] and spec == SPEC2 and meta["arm_type"] == "LookupArm" and meta["name"] == "probe"
    assert meta["params"] == h.n_params()
    h.detach(verify=True)
    m2 = _lm()
    h2 = attach_saved_lookup(m2, p)
    assert torch.equal(_logits(m2, ids), armed)
    h2.detach(verify=True)
    m3 = _lm()
    h3 = attach_saved_lookup(m3, p, depth=3)
    assert torch.equal(_logits(m3, ids), armed) and all(a.depth == 3 for a in h3.arms())
    h3.detach(verify=True)


def test_alephllm_key_map_round_trip():
    m = _lm()
    h = attach_lookup(m, [2, 6 % L], SPEC2, seed=7)
    _wake(h)
    ids = _ids()
    armed = _logits(m, ids)
    sd = h.alephllm_state()
    keys = sorted(sd)
    assert any(k.startswith("lookup_arms.2.qkv") for k in keys)
    assert any(k.startswith("lookup_arms.2_l2.") for k in keys) and any(k.startswith("lookup_norms.2_l2.") for k in keys)
    assert not any("_l1" in k for k in keys)
    adapters, sites, depth = from_alephllm_state(dict(sd, **{"blocks.0.x": torch.zeros(1)}))
    assert sites == sorted({2, 6 % L}) and depth == 2
    assert to_alephllm_state(adapters) == {k: v for k, v in sd.items()}
    spec = spec_from_alephllm_state(sd)
    assert spec.heads == H and spec.depth == 2 and spec.qk_norm == "rms"
    h.detach(verify=True)
    m2 = _lm()
    h2 = attach_from_alephllm(m2, sd)
    assert torch.equal(_logits(m2, ids), armed)
    h2.detach(verify=True)
    plain = {k: v for k, v in sd.items() if not k.endswith("_gain")}
    try:
        spec_from_alephllm_state(plain)
        raise AssertionError("no gains and no heads must refuse")
    except ValueError:
        pass
    assert spec_from_alephllm_state(plain, heads=H).qk_norm == ""


def test_strict_shapes_and_sites():
    m = _lm()
    try:
        attach_lookup(m, [L], SPEC)
        raise AssertionError("a site past the stack must refuse")
    except ValueError:
        pass
    try:
        LookupArm(D, LookupSpec(heads=5))
        raise AssertionError("d not a multiple of heads must refuse")
    except ValueError:
        pass
    try:
        LookupSpec(heads=H, qk_norm="layer")
        raise AssertionError("an unknown guard must refuse")
    except ValueError:
        pass


PARITY = os.environ.get("AMOE_LOOKUP_PARITY_CKPT", "")


def test_parity_with_the_alephllm_trunk():
    """The trained Beatrix arm through this package == the trunk's own arm path, bit for bit, on real bytes."""
    if not PARITY or not os.path.exists(PARITY):
        print("parity: skipped (AMOE_LOOKUP_PARITY_CKPT unset or absent)")
        return
    try:
        from dataclasses import replace
        from safetensors.torch import load_file
        from geolip.alephllm.model.alephlm import AlephLM
        from geolip.alephllm.presets import make_v3_lookup_arm_preset
    except Exception as ex:   # pragma: no cover
        print(f"parity: skipped ({type(ex).__name__}: {ex})")
        return
    sd = load_file(PARITY)
    adapters, sites, depth = from_alephllm_state(sd)
    cfg_arm = make_v3_lookup_arm_preset(tuple(sites), 1, depth=depth).model
    ref = AlephLM(cfg_arm).eval()
    res = ref.load_state_dict(sd, strict=True)
    cfg_bare = replace(cfg_arm, lookup_arm_sites=())
    bare = AlephLM(cfg_bare).eval()
    res = bare.load_state_dict(sd, strict=False)
    assert not res.missing_keys, res.missing_keys[:4]
    assert all(k.startswith(("lookup_arms.", "lookup_norms.")) for k in res.unexpected_keys), res.unexpected_keys[:4]
    for p in list(ref.parameters()) + list(bare.parameters()):
        p.requires_grad_(False)
    ids = torch.randint(0, 256, (2, 192), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        want = ref(ids).logits
        before = bare(ids).logits
        assert not torch.equal(want, before), "the arm must write on these rows"
        h = attach_from_alephllm(bare, sd, attn_fp32=bool(getattr(cfg_arm, "attn_fp32", True)))
        assert h.sites == sites and h.spec.heads == int(cfg_arm.n_heads) and h.spec.depth == depth
        got = bare(ids).logits
        err = (got - want).abs().max().item()
        assert torch.equal(got, want), f"parity broken: max |delta| {err:.3e}"
        with h.all_off():
            assert torch.equal(bare(ids).logits, before)
        lg, cache = bare.prefill(ids[:, :160])
        outs = [lg]
        for t in range(160, ids.shape[1]):
            outs.append(bare.decode_step(ids[:, t], cache))      # AlephLM's cached step
        inc = torch.cat(outs, dim=1)
        derr = (inc - want[:, 159:]).abs().max().item()
        assert torch.allclose(inc, want[:, 159:], atol=1e-4, rtol=1e-4), f"decode parity: max |delta| {derr:.3e}"
        print(f"parity: OK on {ids.shape[1]} bytes x {ids.shape[0]} rows; forward bit-exact; decode max |delta| {derr:.2e}; "
              f"sites {sites} depth {depth} params {h.n_params():,}")
    h.detach(verify=True)


if __name__ == "__main__":
    names = [n for n in list(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("ok", n)
    print(f"{len(names)} passed")
