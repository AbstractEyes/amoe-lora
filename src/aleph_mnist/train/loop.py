"""The training loop — phase-0 pretrain and one dial cell.

THE QUESTION. exp012 measured that the address-bottleneck prior PAYS when
trunk and head co-train (7/7 across seeds and budgets; it held at d=384).
exp013 Track A measured that it COSTS on a frozen substrate (-0.09 CE, 2
seeds; a plain MLP wins there). Both poles are canon. The BOUNDARY between
them is what this bed dials:

    trainable trunk blocks  0 (frozen) -> 1 -> 2 -> 4 (full co-training)
  x address mode           soft | sign | none (control) | off (baseline)
  x seeds

PHASE 0 IS NOT OPTIONAL. "Frozen" has to mean frozen-PRETRAINED, or the
dial's left pole is a random-feature bed and nothing transfers to exp013's
finding. Every cell starts from ONE shared trunk checkpoint trained without
adapters; that checkpoint is also the baseline curve. `pretrain_steps=0`
gives the other reading — snap the adapter onto a fresh trunk and let both
move from step 0.

HOUSE RIDERS OBSERVED: pure Adam wd=0 (`amoe.laws.make_optimizer` is the
only optimizer constructor called), fp32 with TF32 off
(`laws.pin_precision`), no global average pooling, peak VRAM + s/step
printed at an early step so an overrun fails loud instead of silently
spilling to shared memory.
"""
from __future__ import annotations

import time
from dataclasses import asdict

import torch

from amoe import laws

from ..config import RunConfig
from ..data import Bed
from ..diagnostics import probes
from ..model import anchor_state, build_heads, build_model


def _democracy(trunk_params, head_params) -> dict:
    groups = {}
    if trunk_params:
        groups["trunk"] = trunk_params
    if head_params:
        groups["adapter"] = head_params
    from ..diagnostics.vitals import grad_norm_spread
    return grad_norm_spread(groups)


# --------------------------------------------------------------- phase 0
def pretrain(cfg: RunConfig, bed: Bed) -> dict:
    """Train the bare trunk. Shared by every cell at this seed so the dial
    compares arms, not initializations."""
    laws.pin_precision()
    trunk = build_model(bed, cfg)          # lands on the bed's device
    if cfg.pretrain_steps == 0:
        return {k: v.cpu().clone() for k, v in trunk.state_dict().items()}
    trunk.set_trainable_blocks(cfg.n_blocks)
    opt = laws.make_optimizer(trunk.parameters(), cfg.lr_trunk * 10)
    trunk.train()
    for step, (xb, yb) in enumerate(
            bed.batches(cfg.batch, cfg.pretrain_steps, seed=cfg.seed), 1):
        loss = trunk(xb, labels=yb).loss
        lv = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0 or step == 1:
            print(f"[pretrain s{cfg.seed}] step {step} loss={lv:.4f}",
                  flush=True)
    ev = probes.evaluate(trunk, bed.xte, bed.yte)
    print(f"[pretrain s{cfg.seed}] base ce={ev['ce']:.4f} "
          f"acc={ev['acc']:.4f}", flush=True)
    return {k: v.cpu().clone() for k, v in trunk.state_dict().items()}


# --------------------------------------------------------------- one cell
def run(cfg: RunConfig, bed: Bed, base_state: dict | None = None) -> dict:
    laws.pin_precision()
    torch.manual_seed(cfg.seed)
    trunk = build_model(bed, cfg)          # lands on the bed's device
    dev = bed.xtr.device                   # one source of truth for placement
    if base_state is not None:
        trunk.load_state_dict({k: v.to(dev) for k, v in base_state.items()})
    base_eval = probes.evaluate(trunk, bed.xte, bed.yte)

    # ORDER MATTERS: dial first (it freezes everything, then unfreezes the
    # tail), heads second (they arrive trainable and must stay that way).
    census = trunk.set_trainable_blocks(cfg.trainable_blocks)
    heads, sites, wrappers = ([], [], [])
    if cfg.mode != "off":
        heads, sites, wrappers = build_heads(
            trunk, mode=cfg.mode, codebook_init=cfg.codebook_init,
            seed=cfg.seed)

    head_params = [p for h in heads for p in h.parameters() if p.requires_grad]
    trunk_params = trunk.trunk_parameters()
    if not head_params and not trunk_params:
        raise ValueError(
            f"cell {cfg.cell} has nothing to train (mode='off' with the trunk "
            "frozen) — that cell IS the base checkpoint")
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": cfg.lr_head})
    if trunk_params:
        groups.append({"params": trunk_params, "lr": cfg.lr_trunk})
    opt = laws.make_optimizer(groups, cfg.lr_head)

    row = {"config": asdict(cfg), "cell": cfg.cell, "device": str(dev),
           "name_key": bed.name_key, "bed": bed.name,
           "params": {**census,
                      "adapter": sum(p.numel() for h in heads
                                     for p in h.parameters()),
                      "adapter_trainable": sum(p.numel()
                                               for p in head_params)},
           "base_eval": base_eval, "traj": []}

    # the zero-init leak, measured before a single step
    if heads:
        row["inertness_at_attach"] = probes.inertness(
            trunk, wrappers, bed.xte[:512])

    cuda = dev.type == "cuda" and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    spread = {}
    trunk.train()
    for step, (xb, yb) in enumerate(
            bed.batches(cfg.batch, cfg.steps, seed=cfg.seed + 7), 1):
        loss = trunk(xb, labels=yb).loss
        lv = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step % cfg.probe_every == 0 or step == 1:
            # BEFORE opt.step(): the grads are live exactly here
            spread = _democracy(trunk_params, head_params)
        opt.step()
        if step == 1 and cuda:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            row["peak_gib"] = round(peak, 3)
            row["s_per_step"] = round(time.time() - t0, 4)
            print(f"[{cfg.cell}] step1 peak={peak:.2f}GiB "
                  f"s/step={row['s_per_step']:.3f} d={cfg.d} "
                  f"batch={cfg.batch}", flush=True)
        if step % cfg.probe_every == 0 or step == 1:
            ev = probes.evaluate(trunk, bed.xte, bed.yte)
            snap = {"step": step, "train_loss": lv, **ev, "grad": spread}
            if heads:
                v = probes.vitals_report(trunk, heads)
                snap["gate_mean"] = v["gate"]["mean"]
                snap["gate_in_band"] = v["gate"]["in_band"]
                snap["drift_mean"] = v["drift_mean"]
                snap["delta_ratio"] = probes.delta_ratio(
                    trunk, heads, bed.xte[:512])
            row["traj"].append(snap)
            trunk.train()
        if step % cfg.log_every == 0 or step == 1:
            print(f"[{cfg.cell}] step {step} loss={lv:.4f}", flush=True)
    row["seconds"] = round(time.time() - t0, 1)

    # ------------------------------------------------------ final probes
    trunk.eval()
    row["final"] = probes.evaluate(trunk, bed.xte, bed.yte)
    row["delta_vs_base_ce"] = base_eval["ce"] - row["final"]["ce"]
    row["delta_vs_base_acc"] = row["final"]["acc"] - base_eval["acc"]
    if heads:
        xd = bed.xte[:1024]
        row["vitals"] = probes.vitals_report(trunk, heads, xd)
        row["escape"] = probes.escape_report(trunk, heads, xd, bed.neutral)
        row["sign_codes"] = probes.sign_code_report(trunk, heads, xd,
                                                    bed.yte[:1024])
        row["toggle"] = probes.toggle_report(trunk, wrappers, bed.xte, bed.yte)
        row["inertness_at_end"] = probes.inertness(trunk, wrappers, xd)
        row["_artifact"] = {"state": anchor_state(heads, sites),
                            "sites": sites}
    return row
