"""Data — MNIST / FashionMNIST / CIFAR-10 beds plus the neutral sets the
blend-escape gauge needs.

ROWS ARE THE INSTRUMENT'S CALIBRATION. The seed-0 pass ran on 4096 rows
and the 110,940-param adapter overfit it (train loss -> ~8e-4, val CE
rising) — which smeared the soft-vs-control signal into the overfitting
noise. `train_n=None` uses the FULL training set; that is the default for
the substrate-climb sweep, and the fix for that smear.

SUBSTRATE COMPLEXITY IS THE REAL DIAL. exp012's co-training win lived at
d=384; the seed-0 MNIST-at-d=64 tie says that substrate is below the
complexity where the address bottleneck is load-bearing. So this file
carries the harder tasks the climb needs: FashionMNIST (a linear-MLP
plateau ~88-90%, real headroom) and CIFAR-10 (raw-pixel MLPs sit ~50-60%
— genuinely hard, the far end of the climb).

NEUTRAL SETS (the single-anchor analogue of the P5d blend-escape gauge):
  permuted  test set under ONE fixed pixel permutation — identical
            first-order statistics, zero class structure
  noise     gaussian matched to the normalized test mean/std
  cross     a REAL different domain at the SAME input dim (mnist<->fashion)

HARD-ZERO NOTE (house law: every float in a patch must carry signal).
MNIST/Fashion background maps to -0.42 under standard normalization, and
CIFAR is dense — so no input float is a hard zero. Verified at build.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
FASHION_MEAN, FASHION_STD = 0.2860, 0.3530
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR_HF_REPO = "uoft-cs/cifar10"   # canonical, parquet on HF's CDN (fast)

DATASET_PIXELS = {"mnist": 784, "fashion": 784, "cifar10": 3072}
DATASET_CLASSES = {"mnist": 10, "fashion": 10, "cifar10": 10}
DATASET_CHANNELS = {"mnist": 1, "fashion": 1, "cifar10": 3}  # trigram order
_CROSS = {"mnist": "fashion", "fashion": "mnist"}   # same-dim real domain


@dataclass
class Bed:
    xtr: torch.Tensor          # (N, P) normalized, flattened
    ytr: torch.Tensor          # (N,)
    xte: torch.Tensor
    yte: torch.Tensor
    neutral: dict[str, torch.Tensor]
    name: str
    channels: int = 1          # 3 for RGB — the trigram order for this bed

    @property
    def pixels(self) -> int:
        return int(self.xtr.shape[1])

    @property
    def n_classes(self) -> int:
        return int(self.ytr.max()) + 1

    def to(self, device) -> "Bed":
        return Bed(self.xtr.to(device), self.ytr.to(device),
                   self.xte.to(device), self.yte.to(device),
                   {k: v.to(device) for k, v in self.neutral.items()},
                   self.name, self.channels)

    def batches(self, batch: int, steps: int, seed: int = 0):
        """Deterministic with-replacement sampling — same discipline as
        amoe.train's loop, no DataLoader workers to make a cell
        nondeterministic."""
        g = torch.Generator().manual_seed(seed)
        n = self.xtr.shape[0]
        for _ in range(steps):
            ix = torch.randint(0, n, (batch,), generator=g)
            ix = ix.to(self.xtr.device)
            yield self.xtr[ix], self.ytr[ix]


# ---------------------------------------------------------------- loaders
def _cifar_norm(arr) -> torch.Tensor:
    """(N,32,32,3) uint8 ndarray -> (N,3072) per-channel-normalized, flat,
    channel-major. Shared by the HF and torchvision CIFAR paths so both
    produce byte-identical tensors."""
    m = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    s = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    x = torch.from_numpy(arr).float().div(255.0)      # (N,32,32,3)
    x = (x.permute(0, 3, 1, 2) - m) / s               # (N,3,32,32)
    return x.reshape(arr.shape[0], -1)


def _load_cifar10(root: str):
    """CIFAR-10, HF-first. torchvision's default mirror (cs.toronto.edu) is
    slow; this pulls the parquet from a Hugging Face dataset repo over HF's
    CDN instead. Precedence:
      AMOE_CIFAR_URL set  -> torchvision with that mirror (skips HF)
      AMOE_CIFAR_SOURCE   -> 'hf' (default) or 'torchvision'
      AMOE_CIFAR_HF_REPO  -> override the HF repo (default uoft-cs/cifar10;
                             point it at your own mirror namespace if you
                             prefer to own the copy)
    HF failure (offline / no `datasets`) falls back to torchvision, so a
    run never hard-stops on the data source."""
    import numpy as np

    force_tv = bool(os.environ.get("AMOE_CIFAR_URL"))
    source = ("torchvision" if force_tv
              else os.environ.get("AMOE_CIFAR_SOURCE", "hf"))

    if source == "hf":
        try:
            from datasets import load_dataset
            repo = os.environ.get("AMOE_CIFAR_HF_REPO", CIFAR_HF_REPO)
            dd = load_dataset(repo, cache_dir=root)   # img (PIL), label (int)

            def flat(split):
                arr = np.stack([np.asarray(im) for im in split["img"]])
                return _cifar_norm(arr), torch.tensor(split["label"]).long()
            xtr, ytr = flat(dd["train"])
            xte, yte = flat(dd["test"])
            print(f"[data] cifar10 <- HF {repo} "
                  f"(train {xtr.shape[0]}, test {xte.shape[0]})", flush=True)
            return xtr, ytr, xte, yte
        except Exception as e:
            print(f"[data] HF cifar10 load failed ({e}); falling back to "
                  "the torchvision mirror", flush=True)

    from torchvision import datasets
    cls = datasets.CIFAR10
    url = os.environ.get("AMOE_CIFAR_URL")
    if url:                                           # a faster host of the tar
        cls.url = url
    tr = cls(root=root, train=True, download=True)
    te = cls(root=root, train=False, download=True)
    return (_cifar_norm(tr.data), torch.tensor(tr.targets).long(),
            _cifar_norm(te.data), torch.tensor(te.targets).long())


def _load(dataset: str, root: str):
    """Return (xtr, ytr, xte, yte): normalized, flattened, on CPU."""
    if dataset == "cifar10":
        return _load_cifar10(root)
    try:
        from torchvision import datasets
    except ImportError as e:                       # pragma: no cover
        raise ImportError("this bed needs torchvision: "
                          "pip install torchvision") from e
    if dataset in ("mnist", "fashion"):
        cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        mean, std = ((MNIST_MEAN, MNIST_STD) if dataset == "mnist"
                     else (FASHION_MEAN, FASHION_STD))

        def flat(ds):
            x = ds.data.float().div(255.0).sub(mean).div(std)
            return x.reshape(x.shape[0], -1), ds.targets.long()
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    tr = cls(root=root, train=True, download=True)
    te = cls(root=root, train=False, download=True)
    return (*flat(tr), *flat(te))


def _balanced_subset(x, y, n_total, n_classes, seed):
    if n_total is None or n_total >= x.shape[0]:
        return x, y                                # FULL set — the default
    g = torch.Generator().manual_seed(seed)
    per = n_total // n_classes
    keep = []
    for c in range(n_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        keep.append(idx[torch.randperm(idx.numel(), generator=g)[:per]])
    keep = torch.cat(keep)
    return x[keep], y[keep]


def build_bed(dataset: str = "mnist", train_n: int | None = None,
              seed: int = 0, root: str = "./data", synthetic: bool = False,
              pixels: int = 784) -> Bed:
    """Build a bed. `train_n=None` (default) uses the full training set —
    the fix for the seed-0 overfitting. `synthetic=True` fabricates shapes
    for smoke tests ONLY and the Bed name says so."""
    if synthetic:
        g = torch.Generator().manual_seed(seed)
        proto = torch.randn(10, pixels, generator=g)

        def make(n):
            yy = torch.randint(0, 10, (n,), generator=g)
            return proto[yy] + 0.7 * torch.randn(n, pixels, generator=g), yy
        xtr, ytr = make(train_n or 512)
        xte, yte = make(512)
        name = f"SYNTHETIC-p{pixels}-smoke-only"
    else:
        n_cls = DATASET_CLASSES[dataset]
        xtr, ytr, xte, yte = _load(dataset, root)
        xtr, ytr = _balanced_subset(xtr, ytr, train_n, n_cls, seed)
        name = f"{dataset}-n{xtr.shape[0]}"

    g = torch.Generator().manual_seed(seed + 1)
    n_neu = min(2048, xte.shape[0])
    perm = torch.randperm(xte.shape[1], generator=g)      # ONE fixed perm
    neutral = {
        "permuted": xte[:n_neu][:, perm].clone(),
        "noise": xte[:n_neu].mean() + xte[:n_neu].std()
        * torch.randn(n_neu, xte.shape[1], generator=g),
    }
    cross = _CROSS.get(dataset)
    if cross and not synthetic:
        try:
            cx = _load(cross, root)[2][:n_neu]            # cross test xte
            if cx.shape[1] == xte.shape[1]:
                neutral[cross] = cx
        except Exception as e:                            # offline is not fatal
            print(f"[data] cross-domain neutral '{cross}' unavailable "
                  f"({e}); continuing with permuted + noise")
    channels = 1 if synthetic else DATASET_CHANNELS[dataset]
    return Bed(xtr, ytr, xte, yte, neutral, name, channels)
