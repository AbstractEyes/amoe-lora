"""Smoke tests for amoe.train.quiet — the abstention term and the alternating-chunk step."""
import contextlib

import torch
import torch.nn as nn

from amoe.train.quiet import QuietSpec, kl_abstain, quiet_step

V, D = 32, 16


class _ToyLM(nn.Module):
    """Embedding -> (optional arm write) -> head; returns raw logits."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(V, D)
        self.head = nn.Linear(D, V)
        self.arm = nn.Linear(D, D, bias=False)
        nn.init.zeros_(self.arm.weight)
        self.arm_on = True

    def forward(self, x):
        h = self.emb(x)
        if self.arm_on:
            h = h + self.arm(h)
        return self.head(h)


class _Handle:
    def __init__(self, m):
        self.m = m

    @contextlib.contextmanager
    def all_off(self):
        prev = self.m.arm_on
        self.m.arm_on = False
        try:
            yield self
        finally:
            self.m.arm_on = prev


def test_zero_write_zero_kl():
    m = _ToyLM()
    assert float(kl_abstain(m, _Handle(m), [1, 2, 3, 4])) == 0.0


def test_live_write_positive_kl_and_grad_reaches_arm():
    m = _ToyLM()
    nn.init.normal_(m.arm.weight, std=0.5)
    loss = kl_abstain(m, _Handle(m), [1, 2, 3, 4])
    assert float(loss) > 0.0
    loss.backward()
    assert m.arm.weight.grad is not None and float(m.arm.weight.grad.abs().sum()) > 0
    # the bare side carried no gradient: only arm/head/emb of the ARMED pass contribute
    assert torch.isfinite(m.arm.weight.grad).all()


def test_quiet_step_alternation_and_descent():
    m = _ToyLM()
    nn.init.normal_(m.arm.weight, std=0.5)
    h = _Handle(m)
    opt = torch.optim.Adam(m.arm.parameters(), lr=1e-2, weight_decay=0.0)
    calls = {"task": 0, "abstain": 0}

    def task_loss():
        calls["task"] += 1
        lg = m(torch.tensor([[1, 2, 3]]))
        return nn.functional.cross_entropy(lg[0, :-1], torch.tensor([2, 3]))

    def abstain_loss():
        calls["abstain"] += 1
        return kl_abstain(m, h, [4, 5, 6, 7])

    first = None
    for _ in range(20):
        t, a = quiet_step(opt, task_loss, abstain_loss, batch_size=2, accum=4,
                          spec=QuietSpec(lam=1.0))
        first = first if first is not None else a
    # accum 4 -> 2 task + 2 abstention chunks x batch 2 per step
    assert calls["task"] == calls["abstain"] == 2 * 2 * 20
    assert a < first  # the abstention term descends on the arm's off-domain write
