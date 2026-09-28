"""The compiled chain's recompile budget over a whole arm program (0.2.11).

Arms attach progressively 3 -> 8 as a staged curriculum does, two blocks share
the chain, every stage runs the armed forward and each member masked (grad with
recompute, and no_grad), the training shape plus one evaluation shape, and a
burst of forty prompt lengths at every stage boundary. dynamo is set to raise
on a limit hit. The cache-entry count of `_apply_chain` must end well under the
library's budget, so no later stage can lose the compile.
"""
import pytest
import torch
import torch.nn as nn

from amoe.core import adapter as A
from amoe.core.adapter import AdapterSpec, BlockWithAdapter, RelayPatchwork, _apply_chain

pytest.importorskip("torch._dynamo")


class _Core(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l = nn.Linear(d, d)

    def forward(self, x):
        return torch.tanh(self.l(x))


def _entries():
    try:
        return len(torch._C._dynamo.eval_frame._debug_get_cache_entry_list(_apply_chain.__code__))
    except Exception:
        return -1


def test_budget_holds_over_eight_progressive_arms():
    import torch._dynamo.config as dc
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    old = (dc.fail_on_recompile_limit_hit, A.COMPILE_MAX_SHAPES)
    dc.fail_on_recompile_limit_hit = True
    A._COMPILED_SHAPES.clear()
    try:
        d = 32
        ac = dict(device_type=dev, dtype=torch.bfloat16, enabled=(dev == "cuda"))
        spec = AdapterSpec(n_slots=4, K=4, D=8, hidden=16, zero_init_head=False)
        torch.manual_seed(0)
        blocks = [{"core": _Core(d), "arms": [], "wraps": [], "top": None} for _ in range(2)]

        def attach(seed):
            for b in blocks:
                torch.manual_seed(seed + len(b["arms"]))
                a = RelayPatchwork(d, spec)
                with torch.no_grad():
                    a.gate.fill_(0.0)
                b["arms"].append(a)
                top = b["top"] if b["top"] is not None else b["core"]
                w = BlockWithAdapter(top, a, recompute=True, compile_chain=True).to(dev)
                b["wraps"].append(w)
                b["top"] = w

        def run(b, x, enabled, grad):
            for wr, e in zip(b["wraps"], enabled):
                wr.enabled = e
            if grad:
                with torch.enable_grad(), torch.autocast(**ac):
                    y = b["top"](x)
                    y.float().sum().backward()
            else:
                with torch.no_grad(), torch.autocast(**ac):
                    y = b["top"](x)
            return y.detach().float()

        def eager(b, x, enabled):
            with torch.no_grad(), torch.autocast(**ac):
                h = b["core"](x)
                for a, e in zip(b["arms"], enabled):
                    if e:
                        h = a(h)
            return h.float()

        def close(a, c):
            return torch.allclose(a, c, atol=3e-2, rtol=3e-2)

        x_train = torch.randn(2, 64, d, device=dev)
        x_eval = torch.randn(1, 128, d, device=dev)
        attach(100); attach(200)
        for stage in range(3, 9):
            attach(100 * stage)
            m = stage
            for b in blocks:
                pats = [tuple(True for _ in range(m))] + [tuple(i != k for i in range(m)) for k in range(m)]
                for p in pats:
                    assert close(run(b, x_train, p, True), eager(b, x_train, p)), (stage, p, "grad")
                    assert close(run(b, x_train, p, False), eager(b, x_train, p)), (stage, p, "no_grad")
                assert close(run(b, x_eval, pats[0], False), eager(b, x_eval, pats[0]))
                for L in range(50, 90):
                    xp = torch.randn(1, L, d, device=dev)
                    assert close(run(b, xp, pats[0], False), eager(b, xp, pats[0]))
        n = _entries()
        assert len(A._COMPILED_SHAPES) == 2
        assert n < 0 or n <= A.COMPILE_RECOMPILE_LIMIT // 2, n
    finally:
        dc.fail_on_recompile_limit_hit, A.COMPILE_MAX_SHAPES = old
        A._COMPILED_SHAPES.clear()
