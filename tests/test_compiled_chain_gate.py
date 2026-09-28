"""The compiled adapter chain's recompile budget and shape gate (0.2.11).

The chain is compiled static, so every (mask pattern x input shape) is one
dynamo cache entry, and past torch's per-code recompile limit dynamo marks
the chain SKIP for the rest of the process (eager from then on, one warning).
An evaluation that feeds the armed model short inputs of many lengths spent
that budget in seconds. The test stacks three arms over a toy block, runs the
training shape armed and with each member masked (grad and no_grad), then
forty inputs of distinct lengths, then the training shape again, with dynamo
configured to raise on a limit hit: the budget is never hit, the training
shape keeps the compiled path, the probe shapes take the eager path, and every
output equals the plain eager stack.
"""
import pytest
import torch
import torch.nn as nn

from amoe.core import adapter as A
from amoe.core.adapter import AdapterSpec, BlockWithAdapter, RelayPatchwork

pytest.importorskip("torch._dynamo")


class _Core(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l = nn.Linear(d, d)

    def forward(self, x):
        return torch.tanh(self.l(x))


def _build(d, dev):
    torch.manual_seed(0)
    core = _Core(d)
    spec = AdapterSpec(n_slots=4, K=4, D=8, hidden=16, zero_init_head=False)
    ads = [RelayPatchwork(d, spec) for _ in range(3)]
    for a in ads:
        with torch.no_grad():
            a.gate.fill_(0.0)          # heads live: the output depends on every member
    w, wraps = core, []
    for a in ads:                      # an arm over an arm over an arm
        w = BlockWithAdapter(w, a, recompute=True, compile_chain=True)
        wraps.append(w)
    return core, ads, wraps, w.to(dev)


def test_compiled_chain_survives_many_shapes():
    import torch._dynamo.config as dc
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    old = (dc.fail_on_recompile_limit_hit, A.COMPILE_MAX_SHAPES)
    dc.fail_on_recompile_limit_hit = True   # a limit hit is an exception, not a silent skip
    A._COMPILED_SHAPES.clear()
    try:
        d = 32
        core, ads, wraps, model = _build(d, dev)
        ac = dict(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda"))

        def eager(x, enabled):
            with torch.no_grad(), torch.autocast(**ac):
                h = core(x)
                for a, e in zip(ads, enabled):
                    if e:
                        h = a(h)
            return h.float()

        def run(x, enabled, grad):
            for wr, e in zip(wraps, enabled):
                wr.enabled = e
            if grad:
                with torch.enable_grad(), torch.autocast(**ac):
                    y = model(x)
                    y.float().sum().backward()
            else:
                with torch.no_grad(), torch.autocast(**ac):
                    y = model(x)
            return y.detach().float()

        def close(a, b):
            return torch.allclose(a, b, atol=3e-2, rtol=3e-2)

        x = torch.randn(2, 64, d, device=dev)
        pats = [(True, True, True), (False, True, True), (True, False, True), (True, True, False)]
        for p in pats:
            assert close(run(x, p, True), eager(x, p)), ("grad", p)
            assert close(run(x, p, False), eager(x, p)), ("no_grad", p)
        assert dc.recompile_limit >= A.COMPILE_RECOMPILE_LIMIT

        calls = {"n": 0}
        orig = A._COMPILED_CHAIN

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        A._COMPILED_CHAIN = counting
        try:
            for L in range(50, 90):   # forty probe-like inputs of distinct lengths
                xp = torch.randn(1, L, d, device=dev)
                assert close(run(xp, (True, True, True), False), eager(xp, (True, True, True))), L
            assert calls["n"] <= 1    # at most the one shape that took the second slot
            assert len(A._COMPILED_SHAPES) == A.COMPILE_MAX_SHAPES
            calls["n"] = 0
            assert close(run(x, (True, True, True), False), eager(x, (True, True, True)))
            assert close(run(x, (False, True, True), True), eager(x, (False, True, True)))
            # 1 no_grad forward + 1 grad forward whose recompute re-runs the chain in the backward
            assert calls["n"] == 3
        finally:
            A._COMPILED_CHAIN = orig
    finally:
        dc.fail_on_recompile_limit_hit, A.COMPILE_MAX_SHAPES = old
        A._COMPILED_SHAPES.clear()
