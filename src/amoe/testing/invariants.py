"""CPU-testable invariants on a tiny synthetic trunk (no downloads).

assert_toggle_law: with every anchor disabled, wrapped-model logits ==
base logits BIT-EXACT. assert_detach_bitexact: detach(verify=True)
round-trips. These are the package's contract with the research line's
toggle law (exp011: max|Δlogit| = 0.0).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _TinyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)          # tensor output (tuple path tested too)


class _TupleBlock(_TinyBlock):
    def forward(self, x):
        return (self.lin(x), None)


class _TinyConfig:
    model_type = "tiny"
    hidden_size = 32
    _name_or_path = "amoe/tiny-test"


class _Inner(nn.Module):
    def __init__(self, d, L, tuple_blocks):
        super().__init__()
        cls = _TupleBlock if tuple_blocks else _TinyBlock
        self.layers = nn.ModuleList([cls(d) for _ in range(L)])


class TinyTrunk(nn.Module):
    """model.model.layers shape -> resolves via generic_causal."""

    def __init__(self, d=32, L=4, vocab=17, tuple_blocks=False):
        super().__init__()
        self.config = _TinyConfig()
        self.model = _Inner(d, L, tuple_blocks)
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids):
        h = self.emb(input_ids)
        for layer in self.model.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        class _O:
            pass
        o = _O()
        o.logits = self.head(h)
        return o


def _fresh_anchor_ckpt(d, L, name="test"):
    from ..core.adapter import RelayPatchwork
    from ..io.checkpoint import AnchorCheckpoint
    torch.manual_seed(7)
    state = {}
    for i in range(L):
        a = RelayPatchwork(d)
        for k, v in a.state_dict().items():
            state[f"{i}.{k}"] = v.clone()
    return AnchorCheckpoint(state, {"name": name,
                                    "base_model_id": "amoe/tiny-test"})


def assert_toggle_law(tuple_blocks=False) -> None:
    from ..runtime.attach import attach
    torch.manual_seed(3)
    m = TinyTrunk(tuple_blocks=tuple_blocks)
    ids = torch.arange(6).unsqueeze(0) % 5
    with torch.no_grad():
        base = m(ids).logits.clone()
    h = attach(m, _fresh_anchor_ckpt(32, 4))
    with h.all_off():
        with torch.no_grad():
            off = m(ids).logits
        assert torch.equal(off, base), "toggle law violated (single)"
    # dispatch path
    m2 = TinyTrunk(tuple_blocks=tuple_blocks)
    with torch.no_grad():
        base2 = m2(ids).logits.clone()
    h2 = attach(m2, [_fresh_anchor_ckpt(32, 4, "a"),
                     _fresh_anchor_ckpt(32, 4, "b")], dispatch="init")
    with h2.all_off():
        with torch.no_grad():
            off2 = m2(ids).logits
        assert torch.equal(off2, base2), "toggle law violated (dispatch)"


def assert_detach_bitexact() -> None:
    from ..runtime.attach import attach
    torch.manual_seed(5)
    m = TinyTrunk()
    h = attach(m, _fresh_anchor_ckpt(32, 4))
    h.detach(verify=True)           # raises on any non-bit-exact detach


def run_all() -> None:
    assert_toggle_law(tuple_blocks=False)
    assert_toggle_law(tuple_blocks=True)
    assert_detach_bitexact()
    print("amoe invariants: toggle law (tensor+tuple, single+dispatch) "
          "and bit-exact detach — ALL GREEN")


if __name__ == "__main__":
    run_all()
