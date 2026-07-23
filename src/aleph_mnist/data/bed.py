"""The `Bed` — a dataset plus the neutral sets the blend-escape gauge needs.

ROWS ARE THE INSTRUMENT'S CALIBRATION. The seed-0 pass ran on 4096 rows and
the adapter overfit them (train loss -> ~8e-4 while val CE rose), which
smeared the soft-vs-control signal into overfitting noise. `train_n=None`
(the default) uses the FULL training set.

NEUTRAL SETS (the single-anchor analogue of the P5d blend-escape gauge):
  permuted  test set under ONE fixed pixel permutation — identical
            first-order statistics, zero class structure
  noise     gaussian matched to the normalized test mean/std
  cross     a REAL different domain at the SAME input dim (mnist<->fashion)

HARD-ZERO NOTE (house law: every float must carry signal). Standard
normalization maps the grayscale background to ~-0.42 and CIFAR is dense,
so no input float is a hard zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .loaders import balanced_subset, load
from .registry import DatasetSpec, get_spec


@dataclass
class Bed:
    xtr: torch.Tensor          # (N, P) normalized, flattened, channel-major
    ytr: torch.Tensor          # (N,)
    xte: torch.Tensor
    yte: torch.Tensor
    neutral: dict[str, torch.Tensor]
    name: str                  # human label, e.g. "cifar10-n50000"
    channels: int = 1          # 3 for RGB — the trigram order for this bed
    name_key: str | None = None   # registry key, or None for synthetic beds

    @property
    def pixels(self) -> int:
        return int(self.xtr.shape[1])

    @property
    def n_classes(self) -> int:
        return int(self.ytr.max()) + 1

    @property
    def spec(self) -> DatasetSpec | None:
        """The registry entry this bed came from, or None if synthetic."""
        return None if self.name_key is None else get_spec(self.name_key)

    def to(self, device) -> "Bed":
        return Bed(self.xtr.to(device), self.ytr.to(device),
                   self.xte.to(device), self.yte.to(device),
                   {k: v.to(device) for k, v in self.neutral.items()},
                   self.name, self.channels, self.name_key)

    def batches(self, batch: int, steps: int, seed: int = 0):
        """Deterministic with-replacement sampling — the same discipline as
        amoe.train's loop, and no DataLoader workers to make a notebook cell
        nondeterministic."""
        g = torch.Generator().manual_seed(seed)
        n = self.xtr.shape[0]
        for _ in range(steps):
            ix = torch.randint(0, n, (batch,), generator=g).to(self.xtr.device)
            yield self.xtr[ix], self.ytr[ix]


def _neutral_sets(xte: torch.Tensor, seed: int, spec: DatasetSpec | None,
                  root: str) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed + 1)
    n_neu = min(2048, xte.shape[0])
    perm = torch.randperm(xte.shape[1], generator=g)          # ONE fixed perm
    neutral = {
        "permuted": xte[:n_neu][:, perm].clone(),
        "noise": xte[:n_neu].mean() + xte[:n_neu].std()
        * torch.randn(n_neu, xte.shape[1], generator=g),
    }
    cross = spec.cross if spec else None
    if cross:
        try:
            cx = load(cross, root)[2][:n_neu]                 # cross-domain xte
            if cx.shape[1] == xte.shape[1]:
                neutral[cross] = cx
        except Exception as e:                     # offline is not fatal
            print(f"[data] cross-domain neutral {cross!r} unavailable ({e}); "
                  "continuing with permuted + noise", flush=True)
    return neutral


def build_bed(dataset: str = "mnist", train_n: int | None = None,
              seed: int = 0, root: str = "./data", synthetic: bool = False,
              pixels: int = 784, channels: int = 1) -> Bed:
    """Build a bed.

    `train_n=None` (default) uses the full training set. `synthetic=True`
    fabricates shapes for smoke tests ONLY — the bed's name says so and its
    `name_key` is None, so the model builder skips dataset parity checks.
    """
    if synthetic:
        if pixels % channels:
            raise ValueError(f"synthetic pixels={pixels} must be divisible "
                             f"by channels={channels}")
        g = torch.Generator().manual_seed(seed)
        proto = torch.randn(10, pixels, generator=g)

        def make(n):
            yy = torch.randint(0, 10, (n,), generator=g)
            return proto[yy] + 0.7 * torch.randn(n, pixels, generator=g), yy

        xtr, ytr = make(train_n or 512)
        xte, yte = make(512)
        return Bed(xtr, ytr, xte, yte,
                   _neutral_sets(xte, seed, None, root),
                   f"SYNTHETIC-p{pixels}c{channels}-smoke-only",
                   channels, None)

    spec = get_spec(dataset)                        # friendly ValueError
    xtr, ytr, xte, yte = load(dataset, root)
    xtr, ytr = balanced_subset(xtr, ytr, train_n, spec.classes, seed)
    return Bed(xtr, ytr, xte, yte,
               _neutral_sets(xte, seed, spec, root),
               f"{dataset}-n{xtr.shape[0]}", spec.channels, dataset)
