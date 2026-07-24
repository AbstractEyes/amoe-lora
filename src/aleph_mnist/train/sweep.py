"""Campaign orchestration: the dial sweep, the substrate climb, the smoke.

`sweep` runs one dial (arms x dial positions x seeds) on a single bed.
`grid` walks that dial up a width ladder across datasets — the substrate
climb. `smoke` is shapes/parse only and is never a result.
"""
from __future__ import annotations

from dataclasses import asdict, replace

import torch

from ..config import RunConfig, resolve_device
from ..data import Bed, build_bed, get_spec
from .ledger import append_ledger, ledger_path
from .loop import pretrain, run

DIAL = (0, 1, 2, 4)
ARMS = ("soft", "sign", "none")
DIMS_CLIMB = (64, 128, 256, 512, 1024)
DATASETS = ("mnist", "fashion", "cifar10")


def sweep(seeds=(0, 1), dial=DIAL, arms=ARMS, base: RunConfig | None = None,
          bed: Bed | None = None, ledger: str | None = None,
          include_scratch: bool = True,
          bases: dict | None = None) -> list[dict]:
    """The dial: arms x dial positions x seeds, plus the trunk-only
    baseline at each position and (optionally) the from-scratch rows that
    answer the co-training question in its literal form."""
    base = base or RunConfig()
    dev = resolve_device(base.device)
    if bed is None:
        bed = build_bed(base.dataset, base.train_n, seed=base.seed,
                        root=base.root, synthetic=base.synthetic).to(dev)
    rows, bases = [], dict(bases or {})
    for seed in seeds:
        if seed not in bases:      # reuse a base the caller already paid for
            bases[seed] = pretrain(replace(base, seed=seed), bed)
        for n in dial:
            cells = [replace(base, mode=m, seed=seed, trainable_blocks=n)
                     for m in arms]
            if n > 0:     # trunk-only baseline; at dial 0 it IS the base
                cells.append(replace(base, mode="off", seed=seed,
                                     trainable_blocks=n))
            for cfg in cells:
                row = run(cfg, bed, bases[seed])
                append_ledger(row, ledger)
                rows.append(row)
                print(f"  -> {cfg.cell}: ce={row['final']['ce']:.4f} "
                      f"acc={row['final']['acc']:.4f} "
                      f"dCE={row['delta_vs_base_ce']:+.4f}", flush=True)
        if include_scratch:
            rows += scratch(seeds=(seed,), arms=arms, base=base, bed=bed,
                            ledger=ledger)
    return rows


def scratch(seeds=(0,), arms=ARMS, base: RunConfig | None = None,
            bed: Bed | None = None, ledger: str | None = None,
            steps: int | None = None) -> list[dict]:
    """The from-scratch co-training rows ONLY — no phase-0 pretrain, trunk
    and head move together from step 0. This is the co-training question in
    its most literal form, and it is the one to re-run on its own when the
    pretrained dial is already banked.

    THE BUDGET MATTERS. A pretrained dial cell's trunk saw
    `pretrain_steps + steps` of training (phase-0 solo, then co-trained); a
    scratch cell that runs only `base.steps` from random init is undertrained
    by exactly `pretrain_steps` and loses the comparison on compute, not on
    the address. So the fair default is `pretrain_steps + steps` — pass
    `steps=` to override.
    """
    base = base or RunConfig()
    dev = resolve_device(base.device)
    if bed is None:
        bed = build_bed(base.dataset, base.train_n, seed=base.seed,
                        root=base.root, synthetic=base.synthetic).to(dev)
    steps = steps if steps is not None else base.pretrain_steps + base.steps
    ledger = ledger or ledger_path()
    rows = []
    for seed in seeds:
        for m in tuple(arms) + ("off",):
            cfg = replace(base, mode=m, seed=seed,
                          trainable_blocks=base.n_blocks, pretrain_steps=0,
                          steps=steps, tag="scratch")
            row = run(cfg, bed, None)
            append_ledger(row, ledger)
            rows.append(row)
            print(f"  -> {cfg.cell}: ce={row['final']['ce']:.4f} "
                  f"acc={row['final']['acc']:.4f} "
                  f"(scratch, {steps} steps)", flush=True)
    return rows


def grid(datasets=DATASETS, dims=DIMS_CLIMB, seeds=(0, 1, 2),
         base: RunConfig | None = None, root: str | None = None,
         ledger: str | None = None, include_scratch: bool = False
         ) -> list[dict]:
    """THE SUBSTRATE CLIMB. exp012's co-training win lived at d=384; a tie
    at d=64 says that substrate is below the complexity where the address
    bottleneck is load-bearing. This walks d up the ladder across task
    difficulties, re-pretraining a fresh trunk at every width, to find
    where (if anywhere) soft separates from the passthrough control.

    One bed per dataset (full-set beds are width-independent), a fresh
    phase-0 base per (dataset, d, seed)."""
    base = base or RunConfig()
    dev = resolve_device(base.device)
    root = root or base.root
    ledger = ledger or ledger_path("grid.jsonl")
    if dev.startswith("cuda") and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"[grid] card: {p.name} ({p.total_memory / 1024**3:.0f} GiB)",
              flush=True)
    print(f"[grid] datasets={list(datasets)} dims={list(dims)} "
          f"seeds={list(seeds)} steps={base.steps} batch={base.batch} "
          f"train_n={base.train_n} mode={base.input_mode} -> {ledger}",
          flush=True)
    rows = []
    for dataset in datasets:
        try:
            bed = build_bed(dataset, base.train_n, seed=base.seed,
                            root=root).to(dev)
        except Exception as e:
            # A slow/failed mirror must not sink an overnight run — the
            # earlier datasets are already in the ledger. Skip and move on.
            print(f"[grid] SKIP {dataset} — could not build bed ({e}). "
                  "cifar10 loads from HF by default (needs `datasets`); set "
                  "AMOE_CIFAR_HF_REPO/SOURCE/URL, or install "
                  "'amoe-lora[experiment-hf]', or rerun with "
                  f"--datasets {dataset} to fill it in.", flush=True)
            continue
        assert bed.pixels == get_spec(dataset).pixels, "pixel-count mismatch"
        print(f"[grid] {bed.name}: train {tuple(bed.xtr.shape)} "
              f"pixels={bed.pixels} neutral={list(bed.neutral)}", flush=True)
        for d in dims:
            print(f"[grid] ===== {dataset} d={d} =====", flush=True)
            rows += sweep(seeds=seeds,
                          base=replace(base, dataset=dataset, d=d),
                          bed=bed, ledger=ledger,
                          include_scratch=include_scratch)
    print(f"[grid] done: {len(rows)} cells -> {ledger}", flush=True)
    return rows


def smoke(device: str | None = None, ledger: str | None = None) -> list[dict]:
    """Shapes/parse only — a handful of steps on tiny synthetic data, never
    a result. Exercises the arm ladder, both trigram forms, and the
    routed-attention PATCH mode. Uses CUDA when present (falls back to CPU on
    a GPU-less CI runner).

    Sizes are deliberately minuscule. The trigram stem turns each image into
    a T-token sequence, so a realistic bed (T=784-1024) is ~1000x the linear
    bed's compute. Here T is 64 (spatial) / 16 (channel) / 4+CLS (patch):
    enough to prove every code path, cheap enough to be a smoke.
    """
    device = resolve_device(device)
    base = RunConfig(d=16, steps=4, pretrain_steps=4, probe_every=2,
                     log_every=999, train_n=128, batch=16, synthetic=True,
                     device=device)
    ledger = ledger or ledger_path("smoke.jsonl")
    bed = build_bed(train_n=128, synthetic=True, pixels=64).to(device)
    rows = sweep(seeds=(0,), dial=(0, 4), arms=("soft", "none"), base=base,
                 bed=bed, ledger=ledger, include_scratch=False)
    # both trigram forms: spatial (gray, T=pixels) and channel (RGB, T=H*W)
    for pixels, channels in ((64, 1), (48, 3)):
        tb = build_bed(train_n=128, synthetic=True, pixels=pixels,
                       channels=channels).to(device)
        row = run(replace(base, input_mode="trigram", mode="soft",
                          trainable_blocks=4, tag=f"trigram{channels}c"),
                  tb, None)
        append_ledger(row, ledger)
        rows.append(row)
    # patch mode: 8x8 image, patch 4 -> 2x2 grid + CLS, aleph-routed mixing
    pb = build_bed(train_n=128, synthetic=True, pixels=64, channels=1).to(device)
    row = run(replace(base, input_mode="patch", patch_size=4, num_heads=4,
                      mode="soft", trainable_blocks=4, tag="patch"), pb, None)
    append_ledger(row, ledger)
    rows.append(row)
    # addressed-conv bed: soft (aleph read) + none (uniform = plain conv)
    from .conv_bed import run_conv
    cb = build_bed(train_n=128, synthetic=True, pixels=64, channels=1).to(device)
    for m in ("soft", "none"):
        row = run_conv(replace(base, input_mode="addr_conv", mode=m, k_bank=8,
                               k_addr=8, conv_channels=8, conv_layers=2,
                               tag="conv"), cb, None)
        append_ledger(row, ledger)
        rows.append(row)
    return rows
