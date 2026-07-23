"""Data — a deliberately starved MNIST bed plus the neutral sets the
blend-escape gauge needs.

STARVATION IS THE INSTRUMENT. Full MNIST on a 4-block MLP saturates near
98% and every arm difference disappears into seed noise; the Jun 19
ruling on this exact hazard was "recon-on-BERT is the wrong probe — too
easy, bypassed". Default here is 4096 class-balanced train rows at d=64,
which leaves real headroom for an adapter to be load-bearing. Report CE,
not only accuracy: CE is the sensitive channel (the campaign's val_ce
deltas ran ~0.09, far below accuracy's resolution).

NEUTRAL SETS (the single-anchor analogue of the P5d blend-escape gauge):
  permuted  MNIST test under ONE fixed pixel permutation — identical
            first-order pixel statistics, zero digit structure. The
            sharpest control: if the adapter fires as hard here as on
            real digits, it is in the blend regime.
  noise     gaussian matched to the normalized train mean/std
  fashion   FashionMNIST test — a real, different domain (optional)

HARD-ZERO NOTE: MNIST is ~80% exact-zero pixels, and the house law is
that every float in a patch must carry signal — hard-zero padding
starves. The standard (x-0.1307)/0.3081 normalization maps background to
-0.4245, so no input float is a hard zero. That is not cosmetic here; it
is the law being satisfied.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
FASHION_MEAN, FASHION_STD = 0.2860, 0.3530


@dataclass
class Bed:
    xtr: torch.Tensor          # (N, 784) normalized
    ytr: torch.Tensor          # (N,)
    xte: torch.Tensor
    yte: torch.Tensor
    neutral: dict[str, torch.Tensor]
    name: str

    def to(self, device) -> "Bed":
        return Bed(self.xtr.to(device), self.ytr.to(device),
                   self.xte.to(device), self.yte.to(device),
                   {k: v.to(device) for k, v in self.neutral.items()},
                   self.name)

    def batches(self, batch: int, steps: int, seed: int = 0):
        """Deterministic with-replacement sampling — same discipline as
        amoe.train's loop, and no DataLoader workers to make a Colab
        cell nondeterministic."""
        g = torch.Generator().manual_seed(seed)
        n = self.xtr.shape[0]
        for _ in range(steps):
            ix = torch.randint(0, n, (batch,), generator=g)
            yield self.xtr[ix.to(self.xtr.device)], \
                self.ytr[ix.to(self.ytr.device)]


def _balanced_subset(x, y, n_total: int, n_classes: int, seed: int):
    if n_total is None or n_total >= x.shape[0]:
        return x, y
    g = torch.Generator().manual_seed(seed)
    per = n_total // n_classes
    keep = []
    for c in range(n_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        keep.append(idx[torch.randperm(idx.numel(), generator=g)[:per]])
    keep = torch.cat(keep)
    return x[keep], y[keep]


def _torchvision(name: str, root: str):
    try:
        from torchvision import datasets
    except ImportError as e:                       # pragma: no cover
        raise ImportError(
            "this bed needs torchvision for MNIST: "
            "pip install torchvision") from e
    cls = {"mnist": datasets.MNIST,
           "fashion": datasets.FashionMNIST}[name]
    tr = cls(root=root, train=True, download=True)
    te = cls(root=root, train=False, download=True)
    return tr, te


def _flat(ds, mean: float, std: float):
    x = ds.data.float().div_(255.0).sub_(mean).div_(std)
    return x.reshape(x.shape[0], -1), ds.targets.long()


def build_bed(dataset: str = "mnist", train_n: int | None = 4096,
              seed: int = 0, root: str = "./data",
              with_fashion: bool = True, synthetic: bool = False) -> Bed:
    """Build the bed. `synthetic=True` fabricates shapes for smoke tests
    ONLY — it is never a result and the Bed name says so."""
    if synthetic:
        g = torch.Generator().manual_seed(seed)
        proto = torch.randn(10, 784, generator=g)
        def make(n):
            y = torch.randint(0, 10, (n,), generator=g)
            return proto[y] + 0.7 * torch.randn(n, 784, generator=g), y
        xtr, ytr = make(train_n or 512)
        xte, yte = make(512)
        name = "SYNTHETIC-smoke-only"
    else:
        mean, std = ((MNIST_MEAN, MNIST_STD) if dataset == "mnist"
                     else (FASHION_MEAN, FASHION_STD))
        tr, te = _torchvision(dataset, root)
        xtr, ytr = _flat(tr, mean, std)
        xte, yte = _flat(te, mean, std)
        xtr, ytr = _balanced_subset(xtr, ytr, train_n, 10, seed)
        name = f"{dataset}-n{xtr.shape[0]}"

    g = torch.Generator().manual_seed(seed + 1)
    n_neu = min(1024, xte.shape[0])
    perm = torch.randperm(xte.shape[1], generator=g)      # ONE fixed perm
    neutral = {
        "permuted": xte[:n_neu][:, perm].clone(),
        "noise": xte[:n_neu].mean() + xte[:n_neu].std()
        * torch.randn(n_neu, xte.shape[1], generator=g),
    }
    if with_fashion and not synthetic and dataset != "fashion":
        try:
            _, fte = _torchvision("fashion", root)
            fx, _ = _flat(fte, FASHION_MEAN, FASHION_STD)
            neutral["fashion"] = fx[:n_neu]
        except Exception as e:                     # offline is not fatal
            print(f"[data] fashion neutral set unavailable ({e}); "
                  "continuing with permuted + noise")
    return Bed(xtr, ytr, xte, yte, neutral, name)
