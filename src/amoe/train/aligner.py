"""align() — keys-only dispatch training over frozen anchors
(reference implementation of the research line's exp014 recipe, with
the exp007 starvation safeguards wired in).

REFERENCE-GRADE: runs on the reference substrate; single-GPU-verified.
The trainable set is ONLY the per-block dispatch key matrices (A x emb
per block). train_new_anchor=True enables a trainable generalist and
emits the exp010 warning (generalists derailed formats at small scale).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from .. import laws
from ..binding.resolver import resolve
from ..io.checkpoint import DispatchCheckpoint
from ..runtime.attach import attach
from .config import AlignConfig
from .data import pad_batch, render_rows


class GeneralistDerailWarning(UserWarning):
    pass


def align(model, anchors, dataset, config: AlignConfig | None = None, *,
          tokenizer=None, binding=None, device="cuda") -> DispatchCheckpoint:
    cfg = config or AlignConfig()
    laws.pin_precision()
    if cfg.train_new_anchor:
        warnings.warn(
            "train_new_anchor=True: the research line found trainable "
            "generalists derail structured formats at small scale "
            "(exp010: format derailment; exp020: the passenger role is "
            "an attractor). Proceeding as configured.",
            GeneralistDerailWarning, stacklevel=2)
    h = attach(model, anchors, dispatch="init", binding=binding)
    dispatches = [blk.disp for blk in h._blocks]
    trainable = [dp.dispatch for dp in dispatches]

    streams = {k: (render_rows(tokenizer, v, 1024)
                   if tokenizer is not None else list(v))
               for k, v in dataset.items()}
    names = list(streams)
    weights = {n: 1.0 for n in names}
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    opt = laws.make_optimizer(trainable, cfg.lr)
    pad_id = tokenizer.pad_token_id if tokenizer is not None else 0
    strikes, alarms = 0, []
    model.train()
    for step in range(1, cfg.steps + 1):
        batch = []
        for _ in range(cfg.batch_size):
            w = torch.tensor([weights[n] for n in names])
            nm = names[int(torch.multinomial(w, 1, generator=g))]
            rows = streams[nm]
            batch.append(rows[int(torch.randint(0, len(rows), (1,),
                                                generator=g))])
        x, y, am = pad_batch(batch, pad_id, device)
        out = model(input_ids=x, attention_mask=am, labels=y)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        if step % cfg.check_every == 0:
            us = h.usage()
            if us:
                p = torch.tensor([max(v, 1e-9) for v in us.values()])
                ent = float(torch.exp(-(p * p.log()).sum()))
                starved = min(us, key=us.get)
                if ent < cfg.usage_ppl_floor or \
                        min(us.values()) < cfg.usage_min:
                    strikes += 1
                    weights[starved] = weights.get(starved, 1.0) * 2.0
                    alarms.append({"step": step, "usage": us,
                                   "starved": starved,
                                   "strike": strikes})
                    if strikes >= cfg.max_strikes:
                        break
                else:
                    strikes = 0
    model.eval()
    per_block = [dp.state_dict() for dp in dispatches]
    meta = {"anchors": h.names, "seed": cfg.seed,
            "recipe": {"optimizer": "adam", "lr": cfg.lr,
                       "steps": cfg.steps},
            "safeguard_log": alarms}
    h.detach(verify=False)   # training perturbs nothing but keys; the
    #                          wrapped layers are restored regardless
    return DispatchCheckpoint(per_block, meta)
