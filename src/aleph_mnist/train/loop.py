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


# how much of the training set the train-side eval samples for the gap.
# a FIXED slice, so the train number is stable across steps (the batch loss
# is too noisy to read a generalization gap off of).
_TRAIN_EVAL_ROWS = 2048


def _snapshot(step: int, train_loss: float, trunk, bed, heads,
              spread: dict) -> dict:
    """One trajectory row: train vs val (the generalization gap), plus the
    adapter vitals that are cheap to read every probe. The heavy end-of-run
    instruments (escape, sign codes, toggle damage, aliveness) stay at the
    end — they are quadratic in the token count.

    `train_ce`/`train_acc` are on a fixed training slice, so `val_ce -
    train_ce` is a real gap and not batch noise. This bed was built to catch
    exactly the failure where train loss collapses while val CE climbs."""
    va = probes.evaluate(trunk, bed.xte, bed.yte)
    tr = probes.evaluate(trunk, bed.xtr[:_TRAIN_EVAL_ROWS],
                         bed.ytr[:_TRAIN_EVAL_ROWS])
    snap = {"step": step, "train_loss": train_loss,
            "train_ce": tr["ce"], "train_acc": tr["acc"],
            "ce": va["ce"], "acc": va["acc"], "n": va["n"],
            "gap_ce": va["ce"] - tr["ce"],          # >0 => overfitting
            "gap_acc": tr["acc"] - va["acc"],
            "grad": spread}
    if heads:
        v = probes.vitals_report(trunk, heads)
        snap["gate_mean"] = v["gate"]["mean"]
        snap["gate_in_band"] = v["gate"]["in_band"]
        snap["gate_std"] = v["gate"]["std"]
        snap["drift_mean"] = v["drift_mean"]
        dr = probes.delta_ratio(trunk, heads, bed.xte[:512])
        snap["delta_ratio"] = dr
        snap["delta_ratio_mean"] = sum(dr) / len(dr)
    return snap


def _fmt_progress(snap: dict, base_eval: dict, cfg: RunConfig) -> str:
    """A dense, ASCII-only progress line — the thing the console was missing.
    train/val/gap first (the ask), then improvement over the frozen base,
    grad democracy, and the adapter vitals when an adapter is present."""
    parts = [
        f"[{cfg.cell}] {snap['step']:>4}/{cfg.steps}",
        f"train ce {snap['train_ce']:.3f} acc {snap['train_acc'] * 100:.1f}",
        f"val ce {snap['ce']:.3f} acc {snap['acc'] * 100:.1f}",
        f"gap ce {snap['gap_ce']:+.3f} acc {snap['gap_acc'] * 100:+.1f}",
        f"vs-base {base_eval['ce'] - snap['ce']:+.3f}",
    ]
    g = snap.get("grad") or {}
    if g.get("norms"):
        gd = " ".join(f"{k[0]} {v:.1e}" for k, v in g["norms"].items())
        parts.append(f"grad {gd} spread {g.get('spread_orders', 0.0):.1f}")
    if "gate_mean" in snap:
        band = "*" if snap.get("gate_in_band") else " "
        parts.append(
            f"gate {snap['gate_mean']:.4f}{band} drift "
            f"{snap['drift_mean']:.3f} amp {snap.get('delta_ratio_mean', 0):.2f}")
    return " | ".join(parts)


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
            va = probes.evaluate(trunk, bed.xte, bed.yte)
            tr = probes.evaluate(trunk, bed.xtr[:_TRAIN_EVAL_ROWS],
                                 bed.ytr[:_TRAIN_EVAL_ROWS])
            print(f"[pretrain s{cfg.seed}] {step:>4}/{cfg.pretrain_steps} "
                  f"train ce {tr['ce']:.3f} acc {tr['acc'] * 100:.1f} | "
                  f"val ce {va['ce']:.3f} acc {va['acc'] * 100:.1f} | "
                  f"gap ce {va['ce'] - tr['ce']:+.3f}", flush=True)
            trunk.train()
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
        probe_due = step % cfg.probe_every == 0 or step == 1
        log_due = step % cfg.log_every == 0 or step == 1
        if probe_due or log_due:
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
        if probe_due or log_due:
            # one snapshot serves both the persisted trajectory and the
            # console line, so the eval never runs twice for one step.
            snap = _snapshot(step, lv, trunk, bed, heads, spread)
            if probe_due:
                row["traj"].append(snap)
            if log_due:
                print(_fmt_progress(snap, base_eval, cfg), flush=True)
            trunk.train()
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
    _print_final(row, cfg)
    return row


def _print_final(row: dict, cfg: RunConfig) -> None:
    """The end-of-run instruments the console never showed: toggle damage
    (the co-training tax on detachability), blend-escape ratios, code
    diversity, axis aliveness. These are the numbers the ledger keeps and a
    reader has to grep for — print them once, at the end of the cell."""
    f = row["final"]
    line = [f"[{cfg.cell}] DONE {row['seconds']}s",
            f"val ce {f['ce']:.4f} acc {f['acc'] * 100:.2f}",
            f"vs-base ce {row['delta_vs_base_ce']:+.4f} "
            f"acc {row['delta_vs_base_acc'] * 100:+.2f}"]
    if "toggle" in row:
        t = row["toggle"]
        line.append(f"toggle-damage ce {t['damage_ce']:+.4f} "
                    f"acc {t['damage_acc'] * 100:+.2f}")
    if "vitals" in row:
        v = row["vitals"]
        line.append(f"gate {v['gate']['mean']:.4f}"
                    f"{'*' if v['gate']['in_band'] else ' '}")
        line.append(f"drift {v['drift_mean']:.4f}/{v['binding_target']}")
        if v.get("aliveness"):
            a = v["aliveness"][0]
            line.append(f"alive {a['axes_alive']}/{a['axes_total']} "
                        f"ppl {a['usage_ppl']:.1f}")
    if "escape" in row:
        e = row["escape"]
        worst = min(e["ratio"].values()) if e["ratio"] else float("nan")
        tag = f" ESCAPED:{','.join(e['escaped'])}" if e["escaped"] else ""
        line.append(f"escape-ratio min {worst:.2f}{tag}")
    if "sign_codes" in row:
        s = row["sign_codes"]
        mi = max((b["max"] for b in s["slot_mi_nats"]), default=0.0)
        line.append(f"codes org {s['organism_unique']} "
                    f"slot-MI {mi:.3f}/{s['label_entropy_nats']:.3f}nats")
    if "inertness_at_end" in row:
        line.append(f"inertness {row['inertness_at_end']['max_abs_dlogit']:.2e}")
    print("  " + " | ".join(line), flush=True)
