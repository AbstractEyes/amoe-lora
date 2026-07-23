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
from dataclasses import asdict, dataclass, replace

import torch

from amoe import laws
from amoe.io.checkpoint import AnchorCheckpoint

from . import probes
from .data import DATASET_PIXELS, Bed, build_bed
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
    train_n: int | None = 4096      # None = the full training set
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
    trunk = build_trunk(cfg.d, cfg.n_blocks, cfg.tokens, seed=cfg.seed,
                        pixels=bed.pixels,
                        n_classes=bed.n_classes).to(cfg.device)
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
    trunk = build_trunk(cfg.d, cfg.n_blocks, cfg.tokens, seed=cfg.seed,
                        pixels=bed.pixels,
                        n_classes=bed.n_classes).to(cfg.device)
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

    cuda = cfg.device.startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
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
        if step == 1 and cuda:
            # MANIFEST rider: always print peak_mem + s/step at an early
            # step so an overrun fails LOUD instead of silently spilling.
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3
            row["peak_gib"] = round(peak, 3)
            row["s_per_step"] = round(time.time() - t0, 4)
            print(f"[{cfg.cell}] step1 peak={peak:.2f}GiB "
                  f"s/step={row['s_per_step']:.3f} d={cfg.d} "
                  f"batch={cfg.batch}", flush=True)
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
    meta = {"name": f"{cfg['dataset']}-{cfg['mode']}-d{cfg['d']}"
                    f"-dial{cfg['trainable_blocks']}",
            "base_model_id": f"tiny-{cfg['dataset']}-4block",
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


# -------------------------------------------------------- the big sweep
DIMS_CLIMB = (64, 128, 256, 512, 1024)
DATASETS = ("mnist", "fashion", "cifar10")


def grid(datasets=DATASETS, dims=DIMS_CLIMB, seeds=(0, 1, 2),
         base: RunConfig | None = None, root: str = "./data",
         ledger: str | None = None, include_scratch: bool = False
         ) -> list[dict]:
    """THE SUBSTRATE CLIMB. exp012's co-training win lived at d=384; the
    seed-0 MNIST-at-d=64 tie says that substrate is below the complexity
    where the address bottleneck is load-bearing. This walks d up the
    ladder across three task difficulties, re-pretraining a fresh trunk at
    every width, to find where (if anywhere) soft separates from the
    passthrough control — L-AR8's real crossing.

    One bed per dataset (shared across widths and seeds; full-set beds are
    width-independent), a fresh phase-0 base per (dataset, d, seed)."""
    base = base or RunConfig()
    ledger = ledger or os.path.join(RESULTS, "grid.jsonl")
    if base.device.startswith("cuda") and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[grid] card: {p.name} ({p.total_memory / 1024**3:.0f} GiB)",
              flush=True)
    print(f"[grid] datasets={list(datasets)} dims={list(dims)} "
          f"seeds={list(seeds)} steps={base.steps} batch={base.batch} "
          f"train_n={base.train_n} -> ledger {ledger}", flush=True)
    rows = []
    for dataset in datasets:
        try:
            bed = build_bed(dataset, base.train_n, seed=base.seed,
                            root=root).to(base.device)
        except Exception as e:
            # A slow/failed mirror (CIFAR's default host is slow) must not
            # sink an overnight run — the earlier datasets are already in
            # the ledger. Skip and move on.
            print(f"[grid] SKIP {dataset} — could not build bed ({e}). "
                  "cifar10 loads from HF by default (needs `datasets`); set "
                  "AMOE_CIFAR_HF_REPO/SOURCE/URL or rerun with "
                  f"--datasets {dataset} to fill it in.", flush=True)
            continue
        assert bed.pixels == DATASET_PIXELS[dataset], "pixel-count mismatch"
        print(f"[grid] {bed.name}: train {tuple(bed.xtr.shape)} "
              f"pixels={bed.pixels} neutral={list(bed.neutral)}", flush=True)
        for d in dims:
            print(f"[grid] ===== {dataset} d={d} =====", flush=True)
            rows += sweep(seeds=seeds, base=replace(base, dataset=dataset,
                                                    d=d),
                          bed=bed, ledger=ledger,
                          include_scratch=include_scratch)
    print(f"[grid] done: {len(rows)} cells -> {ledger}", flush=True)
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
    parse_known_args so ipykernel's injected -f never raises SystemExit.

    Two modes:
      (default) one dial sweep on --dataset at RunConfig's d.
      --big     the substrate climb: grid over --datasets x --dims x
                --seeds, full training sets, the workstation workout.
    """
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--pretrain-steps", type=int, default=None)
    p.add_argument("--train-n", type=int, default=None,
                   help="rows per class-balanced subset; <=0 or omit in "
                        "--big = FULL set")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--dial", type=int, nargs="+", default=list(DIAL))
    p.add_argument("--codebook-init", default="random")
    p.add_argument("--big", action="store_true",
                   help="run the substrate-climb grid (workstation workout)")
    p.add_argument("--dims", type=int, nargs="+", default=list(DIMS_CLIMB))
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--scratch", action="store_true",
                   help="also run the from-scratch co-training rows")
    p.add_argument("--smoke", action="store_true")
    args, _ = p.parse_known_args(argv)
    if args.smoke:
        smoke()
        return

    big = args.big
    steps = args.steps if args.steps is not None else (2000 if big else 1500)
    pre = args.pretrain_steps if args.pretrain_steps is not None else steps
    batch = args.batch if args.batch is not None else (1024 if big else 128)
    if args.train_n is None:
        train_n = None if big else 4096         # big defaults to the FULL set
    else:
        train_n = None if args.train_n <= 0 else args.train_n
    seeds = tuple(args.seeds) if args.seeds else ((0, 1, 2) if big else (0, 1))

    base = RunConfig(steps=steps, pretrain_steps=pre, train_n=train_n,
                     batch=batch, dataset=args.dataset,
                     codebook_init=args.codebook_init)
    if big:
        grid(datasets=tuple(args.datasets), dims=tuple(args.dims),
             seeds=seeds, base=base, include_scratch=args.scratch)
    else:
        sweep(seeds=seeds, dial=tuple(args.dial), arms=tuple(args.arms),
              base=base, include_scratch=args.scratch)


if __name__ == "__main__":
    main()
