"""`aleph-mnist` — the console script (also `python -m aleph_mnist`).

Every flag defaults to the matching `RunConfig` field, so the CLI and the
API expose exactly the same knobs and cannot drift.

Notebook-safe: argparse lives inside `main()` (never at import), and
`parse_known_args` swallows the `-f /path/kernel.json` that ipykernel
injects, so calling `main([])` from a cell never raises SystemExit.

    aleph-mnist --smoke                      # shapes/parse, seconds
    aleph-mnist --dataset mnist --seeds 0 1  # one dial sweep (linear)
    aleph-mnist --patch --dataset mnist      # the aleph-routed ViT (real bed)
    aleph-mnist --big --patch                # the substrate climb
    aleph-mnist --patch --publish            # ...and push evidence to HF
"""
from __future__ import annotations

from .api import publish, run_climb, run_conv, run_scratch, run_sweep, smoke
from .config import RunConfig
from .train import ARMS, CONV_ARMS, DATASETS, DEFAULT_REPO, DIAL, DIMS_CLIMB


def build_parser():
    import argparse
    d = RunConfig()          # defaults come from the one Config
    p = argparse.ArgumentParser(
        prog="aleph-mnist", add_help=False,
        description="The aleph co-training dial / substrate climb.")
    p.add_argument("-h", "--help", action="help")
    p.add_argument("--smoke", action="store_true",
                   help="shapes/parse only on synthetic data (never a result)")
    p.add_argument("--big", action="store_true",
                   help="run the substrate-climb grid instead of one dial")
    # substrate
    p.add_argument("--d", type=int, default=d.d)
    p.add_argument("--n-blocks", type=int, default=d.n_blocks)
    p.add_argument("--patch", action="store_true",
                   help="patch mode: the aleph-routed ViT (the real bed)")
    p.add_argument("--patch-size", type=int, default=d.patch_size)
    p.add_argument("--num-heads", type=int, default=d.num_heads)
    p.add_argument("--conv", action="store_true",
                   help="conv bed: match/defeat a plain conv2d")
    p.add_argument("--conv-tokens", action="store_true",
                   help="conv bed variant: conv stem + per-token SIGNED antipode "
                        "read (the load-bearing one); default is filter-steering")
    p.add_argument("--k-bank", type=int, default=d.k_bank)
    p.add_argument("--conv-channels", type=int, default=d.conv_channels)
    p.add_argument("--conv-layers", type=int, default=d.conv_layers)
    p.add_argument("--generate", action="store_true",
                   help="conv bed: masked-pixel reconstruction (bpb), not classify")
    p.add_argument("--trigram", action="store_true",
                   help="byte_emb x3 input (discovery #16: channel = n-gram "
                        "order) instead of the single-linear unigram stem")
    # data
    p.add_argument("--dataset", default=d.dataset)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--train-n", type=int, default=None,
                   help="class-balanced rows; <=0 means the FULL set")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--root", default=d.root, help="dataset cache root")
    # schedule
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--pretrain-steps", type=int, default=None)
    p.add_argument("--codebook-init", default=d.codebook_init,
                   choices=("random", "fibonacci"))
    # campaign shape
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--dial", type=int, nargs="+", default=list(DIAL))
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--dims", type=int, nargs="+", default=list(DIMS_CLIMB))
    p.add_argument("--scratch", action="store_true",
                   help="also run the from-scratch co-training rows")
    p.add_argument("--scratch-only", action="store_true",
                   help="run ONLY the from-scratch rows (fair budget), no dial")
    p.add_argument("--scratch-steps", type=int, default=None,
                   help="steps per scratch cell (default pretrain_steps+steps)")
    # bookkeeping
    p.add_argument("--device", default=d.device or None)
    p.add_argument("--out", default=None,
                   help="ledger path (default $ALEPH_RESULTS or ./results)")
    # publish
    p.add_argument("--publish", action="store_true",
                   help="after the run, push the evidence to a HuggingFace repo")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"HF repo id for --publish (default {DEFAULT_REPO})")
    p.add_argument("--public", action="store_true",
                   help="make the published repo public (default private)")
    return p


def main(argv=None) -> None:
    args, _ = build_parser().parse_known_args(argv)
    if args.smoke:
        smoke(device=args.device, out=args.out)   # None -> CUDA when present
        return

    if args.conv or args.conv_tokens:  # the conv head-to-head
        steps = args.steps if args.steps is not None else 1500
        train_n = (None if (args.train_n is not None and args.train_n <= 0)
                   else args.train_n)
        mode = "conv_tokens" if args.conv_tokens else "addr_conv"
        conv_cfg = RunConfig(
            dataset=args.dataset, train_n=train_n,
            batch=args.batch or 128, root=args.root, steps=steps,
            probe_every=250, log_every=250,
            input_mode=mode, objective="generate" if args.generate
            else "classify", k_bank=args.k_bank,
            conv_channels=args.conv_channels, conv_layers=args.conv_layers,
            device=args.device or "")
        arms = tuple(args.arms) if args.arms != list(ARMS) else None
        seeds = tuple(args.seeds) if args.seeds else (0,)
        rows = run_conv(conv_cfg, seeds=seeds, arms=arms, out=args.out)
        if args.publish:
            import os
            publish(args.repo, results_dir=os.path.dirname(args.out)
                    if args.out else None, private=not args.public)
        return

    big = args.big
    trigram = args.trigram
    patch = args.patch
    input_mode = "patch" if patch else ("trigram" if trigram else "linear")
    steps = args.steps if args.steps is not None else (2000 if big else 1500)
    pre = args.pretrain_steps if args.pretrain_steps is not None else steps
    if args.batch is not None:
        batch = args.batch
    elif trigram:            # trigram makes each image a T~1000 sequence
        batch = 256
    else:
        batch = 1024 if big else 128
    if args.train_n is None:
        train_n = None if big else RunConfig().train_n   # big -> FULL set
    else:
        train_n = None if args.train_n <= 0 else args.train_n
    seeds = tuple(args.seeds) if args.seeds else ((0, 1) if big else (0, 1))

    cfg = RunConfig(d=args.d, n_blocks=args.n_blocks, dataset=args.dataset,
                    train_n=train_n, batch=batch, root=args.root,
                    steps=steps, pretrain_steps=pre,
                    codebook_init=args.codebook_init,
                    input_mode=input_mode, patch_size=args.patch_size,
                    num_heads=args.num_heads, device=args.device or "")
    if args.scratch_only:
        run_scratch(cfg, seeds=seeds, arms=tuple(args.arms), out=args.out,
                    steps=args.scratch_steps)
    elif big:
        run_climb(cfg, datasets=tuple(args.datasets), dims=tuple(args.dims),
                  seeds=seeds, out=args.out, include_scratch=args.scratch)
    else:
        run_sweep(cfg, seeds=seeds, dial=tuple(args.dial),
                  arms=tuple(args.arms), out=args.out,
                  include_scratch=args.scratch)

    if args.publish:
        import os
        results_dir = os.path.dirname(args.out) if args.out else None
        publish(args.repo, results_dir=results_dir, private=not args.public)


if __name__ == "__main__":
    main()
