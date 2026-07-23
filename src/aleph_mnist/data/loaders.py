"""Raw dataset loading — the only place that touches torchvision / HF.

Every loader returns the same contract: `(xtr, ytr, xte, yte)` normalized,
flattened **channel-major**, on CPU. Normalization constants come from the
dataset's `DatasetSpec` (see `registry.py`), never from module globals, so
a spec and a loader cannot drift apart.

Heavy imports (`torchvision`, `datasets`, `numpy`) are deliberately LAZY —
inside the functions — so `import aleph_mnist` needs only torch + amoe.
That keeps the base install importable and the console script loadable
without the `[experiment]` extra.
"""
from __future__ import annotations

import os

import torch

from .registry import DatasetSpec, get_spec


def _norm_rgb(arr, spec: DatasetSpec) -> torch.Tensor:
    """(N,H,W,3) uint8 ndarray -> (N, 3*H*W) per-channel-normalized, flat,
    CHANNEL-MAJOR. Shared by the HF and torchvision RGB paths so both
    produce byte-identical tensors."""
    m = torch.tensor(spec.mean).view(1, spec.channels, 1, 1)
    s = torch.tensor(spec.std).view(1, spec.channels, 1, 1)
    x = torch.from_numpy(arr).float().div(255.0)          # (N,H,W,C)
    x = (x.permute(0, 3, 1, 2) - m) / s                   # (N,C,H,W)
    return x.reshape(arr.shape[0], -1)                    # channel-major


def _norm_gray(data, spec: DatasetSpec) -> torch.Tensor:
    """(N,H,W) uint8 tensor -> (N, H*W) normalized, flat."""
    x = data.float().div(255.0).sub(spec.mean[0]).div(spec.std[0])
    return x.reshape(x.shape[0], -1)


def _load_rgb_hf(root: str, spec: DatasetSpec):
    """RGB dataset from a Hugging Face parquet mirror (the fast path).

    torchvision's CIFAR mirror (cs.toronto.edu) is slow; HF's CDN pulls the
    full 50k/10k in seconds. Env precedence:
      AMOE_CIFAR_URL     -> force torchvision with that tar mirror (skips HF)
      AMOE_CIFAR_SOURCE  -> 'hf' (default) | 'torchvision'
      AMOE_CIFAR_HF_REPO -> override the repo (default from the spec); point
                            it at your own namespace to own the copy
    """
    import numpy as np
    from datasets import load_dataset

    repo = os.environ.get("AMOE_CIFAR_HF_REPO") or spec.hf_repo
    dd = load_dataset(repo, cache_dir=root)      # columns: img (PIL), label

    def flat(split):
        arr = np.stack([np.asarray(im) for im in split["img"]])
        return _norm_rgb(arr, spec), torch.tensor(split["label"]).long()

    xtr, ytr = flat(dd["train"])
    xte, yte = flat(dd["test"])
    print(f"[data] {spec.name} <- HF {repo} "
          f"(train {xtr.shape[0]}, test {xte.shape[0]})", flush=True)
    return xtr, ytr, xte, yte


def _load_rgb_torchvision(root: str, spec: DatasetSpec):
    """RGB dataset via torchvision (the fallback; slow default mirror)."""
    from torchvision import datasets
    cls = datasets.CIFAR10
    url = os.environ.get("AMOE_CIFAR_URL")
    if url:                                      # a faster host of the tar
        cls.url = url
    tr = cls(root=root, train=True, download=True)
    te = cls(root=root, train=False, download=True)
    return (_norm_rgb(tr.data, spec), torch.tensor(tr.targets).long(),
            _norm_rgb(te.data, spec), torch.tensor(te.targets).long())


def _load_gray(root: str, spec: DatasetSpec):
    """MNIST / FashionMNIST via torchvision."""
    try:
        from torchvision import datasets
    except ImportError as e:                     # pragma: no cover
        raise ImportError(
            "this bed needs torchvision for the grayscale sets: "
            "pip install 'amoe-lora[experiment]'") from e
    cls = datasets.MNIST if spec.name == "mnist" else datasets.FashionMNIST
    tr = cls(root=root, train=True, download=True)
    te = cls(root=root, train=False, download=True)
    return (_norm_gray(tr.data, spec), tr.targets.long(),
            _norm_gray(te.data, spec), te.targets.long())


def load(dataset: str, root: str):
    """Return (xtr, ytr, xte, yte) for a registered dataset.

    RGB sets prefer their HF mirror and fall back to torchvision, so a run
    never hard-stops on the data source."""
    spec = get_spec(dataset)
    if spec.channels == 1:
        return _load_gray(root, spec)

    force_tv = bool(os.environ.get("AMOE_CIFAR_URL"))
    source = ("torchvision" if force_tv
              else os.environ.get("AMOE_CIFAR_SOURCE", "hf"))
    if source == "hf" and spec.hf_repo:
        try:
            return _load_rgb_hf(root, spec)
        except Exception as e:
            print(f"[data] HF {spec.name} load failed ({e}); falling back "
                  "to the torchvision mirror", flush=True)
    return _load_rgb_torchvision(root, spec)


def balanced_subset(x, y, n_total, n_classes, seed):
    """Class-balanced subsample. `n_total=None` keeps the FULL set (the
    default — the fix for the seed-0 overfitting smear)."""
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
