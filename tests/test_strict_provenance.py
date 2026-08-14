"""strict= must be a REAL guard on every substrate, including ones with
no HF config. Adds to amoe's own suite."""
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

import amoe
from amoe.core.adapter import AdapterSpec, RelayPatchwork


@dataclass
class _Cfg:
    d_model: int = 32
    context: int = 64
    name: str = "mini-beatrix-1"


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x, **kw):
        return x + self.lin(x)


class _StubAlephLM(nn.Module):
    def __init__(self, name="mini-beatrix-1"):
        super().__init__()
        self.cfg = _Cfg(name=name)
        self.emb = nn.Embedding(256, 32)
        self.blocks = nn.ModuleList(_Block(32) for _ in range(2))
        self.head = nn.Linear(32, 256)

    def forward(self, idx=None, input_ids=None, **kw):
        x = self.emb(idx if idx is not None else input_ids)
        for b in self.blocks:
            x = b(x)
        return self.head(x), None


def _ck(base_model_id, spec):
    ads = {}
    for i in range(2):
        a = RelayPatchwork(32, spec)
        for k, v in a.state_dict().items():
            ads[f"{i}.{k}"] = v.clone()
    return amoe.AnchorCheckpoint(ads, {"name": "t",
                                       "base_model_id": base_model_id})


SPEC = AdapterSpec(n_slots=4, K=8, D=4, hidden=16)


def test_strict_rejects_foreign_substrate_without_hf_config():
    m = _StubAlephLM("mini-beatrix-1")
    with pytest.raises(ValueError, match="was trained on"):
        amoe.attach(m, _ck("alephllm/mini-beatrix-2@step7", SPEC),
                    spec=SPEC, strict=True)


def test_strict_accepts_coarser_binding_identity():
    """Binding knows the craft; the anchor knows craft@step. Prefix
    agreement is a match — the loader verifies the step."""
    m = _StubAlephLM("mini-beatrix-1")
    h = amoe.attach(m, _ck("alephllm/mini-beatrix-1@step51882", SPEC),
                    spec=SPEC, strict=True)
    assert h is not None
    amoe.detach(h, verify=True)


def test_strict_refuses_when_identity_is_unknowable():
    class _Nameless(_StubAlephLM):
        def __init__(self):
            super().__init__()
            self.cfg = type("C", (), {"d_model": 32, "context": 64})()
    with pytest.raises(ValueError, match="CANNOT be verified"):
        amoe.attach(_Nameless(), _ck("alephllm/x@step1", SPEC),
                    spec=SPEC, strict=True)


def test_strict_false_still_permits_knowing_override():
    m = _StubAlephLM("mini-beatrix-1")
    h = amoe.attach(m, _ck("alephllm/mini-beatrix-2@step7", SPEC),
                    spec=SPEC, strict=False)
    assert h is not None
