"""The quiet (abstention) term — selectivity trained INTO an arm.

An arm trained only on its own rows writes wherever its gradient reaches;
stacked always-on, such arms compound or erase each other. The quiet term
adds, on OFF-DOMAIN rows, the KL divergence between the model with this
arm masked (the bare side, no gradient) and the model with the arm live:

    L = L_task(own rows) + lambda * KL( p_bare(x) || p_armed(x) )

summed over the vocabulary at every position of the off-domain row and
averaged over positions. The arm keeps its on-domain skill and stops
moving anything else; quiet arms then compose ALWAYS-ON with no mixer,
and every member of a stack must carry the term — one member without it
erases the stack while the quiet members keep their solo reads intact.

Validated at two seeds on the published mini-beatrix-2s arms
(huggingface.co/AbstractPhil/mini-beatrix-2s, arms/btx_e003 + reports):
off-domain drift falls 7-25x while each arm's own read holds, and a pair
of quiet arms reads where the same pair without the term reads zero.

Protocol constants of record: lambda 1.0 for template and capability
arms (harder task pools may need a larger dose — calibrate against an
off-domain bits-per-byte bar); a 50/50 alternation of task and
abstention chunks inside each optimizer step; pure Adam (amoe.laws).
Known limit: the term is measured at the logits, so a mid-stack residual
write the trunk absorbs before the output is invisible to it —
representation-level arms need a residual-level form of the penalty.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _logits(model, x):
    out = model(x)
    return out.logits if hasattr(out, "logits") else \
        (out[0] if isinstance(out, (tuple, list)) else out)


def kl_abstain(model, handle, ids, device=None):
    """KL(bare || armed) over every position of one off-domain row.

    The bare side is the same model with THIS arm masked
    (``handle.all_off()``) under no_grad; only the armed side carries
    gradient, so the arm is pushed toward the masked model's own
    distribution on rows that are not its job. ``ids`` is a token/byte
    id sequence (list or 1-D tensor); returns a scalar loss.
    """
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long)
    if device is None:
        device = next(model.parameters()).device
    x = ids.view(1, -1).to(device)
    with torch.no_grad():
        with handle.all_off():
            lb = _logits(model, x)[0].float().log_softmax(-1)
    la = _logits(model, x)[0].float().log_softmax(-1)
    return (lb.exp() * (lb - la)).sum(-1).mean()


@dataclass
class QuietSpec:
    """The protocol constants of record (see the module docstring)."""
    lam: float = 1.0          # abstention weight; template/capability default
    task_fraction: float = 0.5  # 50/50 task / abstention chunk alternation


def quiet_step(optimizer, task_loss, abstain_loss, *, batch_size, accum,
               spec: QuietSpec | None = None):
    """One optimizer step of the certified alternating-chunk protocol.

    ``task_loss()`` / ``abstain_loss()`` each return one row's scalar loss
    (the caller samples its own rows; ``abstain_loss`` will usually wrap
    :func:`kl_abstain`). Accumulation chunks alternate task / abstention
    (even chunks task), each chunk ``batch_size`` rows, gradients scaled
    by 1/accum. Returns (mean task chunk loss, mean abstention chunk
    loss) as floats for logging.
    """
    spec = spec or QuietSpec()
    optimizer.zero_grad(set_to_none=True)
    n_t = n_a = 0
    tot_t = tot_a = 0.0
    for a in range(accum):
        is_task = a % 2 == 0
        losses = [(task_loss() if is_task else spec.lam * abstain_loss())
                  for _ in range(batch_size)]
        lb = torch.stack(losses).mean()
        (lb / accum).backward()
        if is_task:
            tot_t += float(lb); n_t += 1
        else:
            tot_a += float(lb); n_a += 1
    optimizer.step()
    return (tot_t / max(n_t, 1), tot_a / max(n_a, 1))
