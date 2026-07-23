"""The co-training dial.

THE QUESTION. exp012 measured that the address-bottleneck prior PAYS when
trunk and head co-train (7/7 across seeds and budgets; it held at d=384).
exp013 Track A measured that it COSTS on a frozen substrate (-0.09 CE, 2
seeds; a plain MLP wins there). Both poles are canon. The BOUNDARY
between them has never been measured — the plan that would have measured
it (exp014 phase 1, rank 1 in the standing queue) was displaced by the
genetic-distillation campaign and never ran.

This bed turns that dial on a substrate small enough to sweep in an
evening instead of a GPU week:

    trainable trunk blocks  0 (frozen) -> 1 -> 2 -> 4 (full co-training)
  x address mode           soft | sign | none (control) | off (baseline)
  x seeds

PHASE 0 IS NOT OPTIONAL. "Frozen" has to mean frozen-PRETRAINED, or the
dial's left pole is a random-feature bed and nothing transfers to
exp013's finding. Every cell starts from ONE shared trunk checkpoint
trained without adapters; that checkpoint is also the baseline curve.
`pretrain_steps=0` gives the other reading of the question — snap the
adapter onto a fresh trunk and let both move from step 0.

HOUSE RIDERS OBSERVED: pure Adam wd=0 (`amoe.laws.make_optimizer` is the
only constructor called), fp32 with TF32 off (`laws.pin_precision`), no
argparse at import, no `__file__`, no global average pooling.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import torch

from amoe import laws
from amoe.io.checkpoint import AnchorCheckpoint

from . import probes
from .data import Bed, build_bed
from .heads import anchor_state, build_heads
from .trunk import build_trunk

RESULTS = os.path.join("experiments", "results")


@dataclass
class RunConfig:
    mode: str = "soft"              # soft | sign | none | frozen | off
    trainable_blocks: int = 4       # THE DIAL: 0 = frozen substrate
    seed: int = 0
    # substrate
    d: int = 64
    n_blocks: int = 4
    tokens: int = 1
    # data
    dataset: str = "mnist"
    train_n: int = 4096
    batch: int = 128
    synthetic: bool = False
    # schedule
    pretrain_steps: int = 1500      # 0 = both move from step 0
    steps: int = 1500
    lr_head: float = 1e-3
    lr_trunk: float = 1e-4
    codebook_init: str = "random"   # random | fibonacci
    # bookkeeping
    probe_every: int = 250
    log_every: int = 250
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tag: str = ""

    @property
    def cell(self) -> str:
        return (f"{self.mode}/dial{self.trainable_blocks}/s{self.seed}"
                + (f"/{self.tag}" if self.tag else ""))


# --------------------------------------------------------------- phase 0
def pretrain(cfg: RunConfig, bed: Bed) -> dict:
    """Train the bare trunk. Shared by every cell at this seed so the
    dial compares arms, not initializations."""
    laws.pin_precision()
    trunk = build_trunk(cfg.d, cfg.n_blocks, cfg.tokens,
                        seed=cfg.seed).to(cfg.device)
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
            print(f"[pretrain s{cfg.seed}] step {step} "
                  f"loss={lv:.4f}", flush=True)
    ev = probes.evaluate(trunk, bed.xte, bed.yte)
    print(f"[pretrain s{cfg.seed}] base ce={ev['ce']:.4f} "
          f"acc={ev['acc']:.4f}", flush=True)
    return {k: v.cpu().clone() for k, v in trunk.state_dict().items()}


# --------------------------------------------------------------- one cell
def run(cfg: RunConfig, bed: Bed, base_state: dict | None = None) -> dict:
    laws.pin_precision()
    torch.manual_seed(cfg.seed)
    trunk = build_trunk(cfg.d, cfg.n_blocks, cfg.tokens,
                        seed=cfg.seed).to(cfg.device)
    if base_state is not None:
        trunk.load_state_dict({k: v.to(cfg.device)
                               for k, v in base_state.items()})
    base_eval = probes.evaluate(trunk, bed.xte, bed.yte)

    # ORDER MATTERS: dial first (it freezes everything, then unfreezes the
    # tail), heads second (they arrive trainable and must stay that way).
    census = trunk.set_trainable_blocks(cfg.trainable_blocks)
    heads, sites, wrappers = ([], [], [])
    if cfg.mode != "off":
        heads, sites, wrappers = build_heads(
            trunk, mode=cfg.mode, codebook_init=cfg.codebook_init,
            seed=cfg.seed)

    head_params = [p for h in heads for p in h.parameters()
                   if p.requires_grad]
    trunk_params = trunk.trunk_parameters()
    if not head_params and not trunk_params:
        raise ValueError(
            f"cell {cfg.cell} has nothing to train (mode=off with the "
            "trunk frozen) — that cell is the base checkpoint itself")
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": cfg.lr_head})
    if trunk_params:
        groups.append({"params": trunk_params, "lr": cfg.lr_trunk})
    opt = laws.make_optimizer(groups, cfg.lr_head)

    row = {"config": asdict(cfg), "cell": cfg.cell,
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

    t0 = time.time()
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
        if step % cfg.probe_every == 0 or step == 1:
            ev = probes.evaluate(trunk, bed.xte, bed.yte)
            snap = {"step": step, "train_loss": lv, **ev,
                    "grad": spread}
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
        row["sign_codes"] = probes.sign_code_report(
            trunk, heads, xd, bed.yte[:1024])
        row["toggle"] = probes.toggle_report(trunk, wrappers,
                                             bed.xte, bed.yte)
        row["inertness_at_end"] = probes.inertness(trunk, wrappers, xd)
        row["_artifact"] = {"state": anchor_state(heads, sites),
                            "sites": sites}
    return row


def _democracy(trunk_params, head_params) -> dict:
    groups = {}
    if trunk_params:
        groups["trunk"] = trunk_params
    if head_params:
        groups["adapter"] = head_params
    from .vitals import grad_norm_spread
    return grad_norm_spread(groups)


# ----------------------------------------------------------------- sweep
def save_anchor(row: dict, path: str) -> str:
    """Ship the trained heads as a real amoe.anchor v1 checkpoint."""
    art = row.get("_artifact")
    if art is None:
        raise ValueError("this row has no adapter to save")
    cfg = row["config"]
    meta = {"name": f"mnist-{cfg['mode']}-dial{cfg['trainable_blocks']}",
            "base_model_id": "tiny-mnist-4block",
            "d": cfg["d"], "n_layers": cfg["n_blocks"],
            "sites": art["sites"], "seed": cfg["seed"], "precision": "fp32",
            "address_mode": cfg["mode"],      # NOT stock if != soft
            "codebook_init": cfg["codebook_init"],
            "recipe": {"optimizer": "adam", "lr": cfg["lr_head"],
                       "weight_decay": 0.0, "steps": cfg["steps"],
                       "batch": cfg["batch"],
                       "trainable_trunk_blocks": cfg["trainable_blocks"]},
            "experiment": "aleph_mnist co-training dial"}
    return AnchorCheckpoint(art["state"], meta).save(path)


def append_ledger(row: dict, path: str | None = None) -> str:
    path = path or os.path.join(RESULTS, "ledger.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = {k: v for k, v in row.items() if k != "_artifact"}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean) + "\n")
    return path


DIAL = (0, 1, 2, 4)
ARMS = ("soft", "sign", "none")


def sweep(seeds=(0, 1), dial=DIAL, arms=ARMS, base: RunConfig | None = None,
          bed: Bed | None = None, ledger: str | None = None,
          include_scratch: bool = True,
          bases: dict | None = None) -> list[dict]:
    """The default dial. 3 arms x 4 dial positions x 2 seeds = 24 cells,
    plus the trunk-only baseline at each dial position and the
    from-scratch co-training row that answers the question in its
    literal form."""
    base = base or RunConfig()
    bed = bed or build_bed(base.dataset, base.train_n, seed=base.seed,
                           synthetic=base.synthetic).to(base.device)
    rows, bases = [], dict(bases or {})
    for seed in seeds:
        if seed not in bases:       # reuse a base the caller already paid for
            bases[seed] = pretrain(
                RunConfig(**{**asdict(base), "seed": seed}), bed)
        for n in dial:
            cells = [RunConfig(**{**asdict(base), "mode": m, "seed": seed,
                                  "trainable_blocks": n})
                     for m in arms]
            if n > 0:      # trunk-only baseline; at dial 0 it is the base
                cells.append(RunConfig(**{**asdict(base), "mode": "off",
                                          "seed": seed,
                                          "trainable_blocks": n}))
            for cfg in cells:
                row = run(cfg, bed, bases[seed])
                append_ledger(row, ledger)
                rows.append(row)
                print(f"  -> {cfg.cell}: ce={row['final']['ce']:.4f} "
                      f"acc={row['final']['acc']:.4f} "
                      f"dCE={row['delta_vs_base_ce']:+.4f}", flush=True)
        if include_scratch:
            for m in arms + ("off",):
                cfg = RunConfig(**{**asdict(base), "mode": m, "seed": seed,
                                   "trainable_blocks": base.n_blocks,
                                   "pretrain_steps": 0, "tag": "scratch"})
                row = run(cfg, bed, None)
                append_ledger(row, ledger)
                rows.append(row)
                print(f"  -> {cfg.cell}: ce={row['final']['ce']:.4f} "
                      f"acc={row['final']['acc']:.4f}", flush=True)
    return rows


def smoke(device: str = "cpu") -> list[dict]:
    """Shapes/parse only — 20 steps on synthetic data, never a result."""
    cfg = RunConfig(steps=20, pretrain_steps=20, probe_every=10,
                    log_every=10, train_n=512, batch=32, synthetic=True,
                    device=device)
    bed = build_bed(train_n=512, synthetic=True).to(device)
    return sweep(seeds=(0,), dial=(0, 4), arms=("soft", "none"),
                 base=cfg, bed=bed,
                 ledger=os.path.join(RESULTS, "smoke.jsonl"),
                 include_scratch=False)


def main(argv=None) -> None:
    """Colab-cell-safe entry: argparse only behind an explicit call, and
    parse_known_args so ipykernel's injected -f never raises SystemExit."""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--pretrain-steps", type=int, default=1500)
    p.add_argument("--train-n", type=int, default=4096)
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--dial", type=int, nargs="+", default=list(DIAL))
    p.add_argument("--codebook-init", default="random")
    p.add_argument("--smoke", action="store_true")
    args, _ = p.parse_known_args(argv)
    if args.smoke:
        smoke()
        return
    base = RunConfig(steps=args.steps, pretrain_steps=args.pretrain_steps,
                     train_n=args.train_n, dataset=args.dataset,
                     codebook_init=args.codebook_init)
    sweep(seeds=tuple(args.seeds), dial=tuple(args.dial),
          arms=tuple(args.arms), base=base)


if __name__ == "__main__":
    main()
