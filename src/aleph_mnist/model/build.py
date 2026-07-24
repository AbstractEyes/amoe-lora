"""`build_model(bed, cfg)` — the ONE supported trunk constructor.

Every shape the trunk needs (pixels, channels, classes, token count, the
trigram window, the strict-attach identity) is DERIVED from the bed and its
`DatasetSpec`, then VALIDATED. A disagreement raises a clear error here, at
build time, instead of surfacing as a reshape explosion mid-forward or — far
worse — as a silently wrong model that trains and reports numbers.

What each check exists to stop (all were real, latent failure modes):

  pixel/channel parity   a caller building a 784-pixel MNIST trunk and then
                         feeding it a 3072-pixel CIFAR bed
  class sizing           `n_classes` taken from `ytr.max()+1` under-sizes the
                         readout whenever a subsample happens to drop a class
  trigram channels       the stem assumes RGB(3) or gray(1); anything else
                         built a stem that could not represent the input
  token divisibility     linear mode reshapes `pixels -> (tokens, patch)`
  readout width          linear `tokens>1` builds `Linear(d*T, C)`, which at
                         large d*T allocates a multi-GB activation that
                         WDDM-spills and crawls
  attach identity        `save_anchor` writes `base_model_id="tiny-{dataset}
                         -4block"` while `TinyConfig._name_or_path` was frozen
                         to `"tiny-mnist-4block"`, so a fashion/cifar anchor
                         failed `amoe.attach(strict=True)` against its OWN
                         trunk (see src/amoe/runtime/attach.py strict block).
                         The name is now derived per dataset.
"""
from __future__ import annotations

import torch

from ..data.bed import Bed
from .stem import VALID_CHANNELS
from .trunk import TinyConfig, TinyTrunk

READOUT_MAX = 1 << 20      # flattened readout width we refuse to allocate


def trunk_identity(name_key: str | None) -> str:
    """The strict-attach identity string.

    `build_model` stamps this on the trunk's `_name_or_path` and
    `save_anchor` writes the SAME string into the checkpoint's
    `base_model_id`. It lives in one function precisely so the two cannot
    disagree — when they did, `amoe.attach(strict=True)` rejected an anchor
    against the very trunk that produced it.
    """
    return f"tiny-{name_key}-4block" if name_key else "tiny-synthetic-4block"


def _trigram_tokens(pixels: int, channels: int) -> int:
    """T for the trigram stem: RGB -> one token per pixel (H*W); grayscale
    -> one token per raster position (pixels)."""
    return pixels // channels if channels == 3 else pixels


def build_model(bed: Bed, cfg, *, seed: int | None = None) -> TinyTrunk:
    """Derive-then-validate. `cfg` is a `RunConfig` (or anything carrying
    d / n_blocks / tokens / input_mode / seed)."""
    pixels, channels = bed.pixels, bed.channels
    n_classes = bed.n_classes
    spec = bed.spec                     # None for synthetic beds

    d = getattr(cfg, "d", 64)
    tokens = getattr(cfg, "tokens", 1)
    n_blocks = getattr(cfg, "n_blocks", 4)
    input_mode = getattr(cfg, "input_mode", "linear")
    readout_dim = getattr(cfg, "readout_dim", 16)
    n_bins = getattr(cfg, "n_bins", 256)

    # 1. dataset parity — only when the bed came from the registry
    if spec is not None:
        if pixels != spec.pixels:
            raise ValueError(
                f"bed {bed.name!r} has {pixels} pixels but the {spec.name} "
                f"spec is {spec.pixels} ({spec.channels}x{spec.height}x"
                f"{spec.width}) — loader and registry disagree")
        if channels != spec.channels:
            raise ValueError(
                f"bed {bed.name!r} has channels={channels} but the "
                f"{spec.name} spec is {spec.channels}")
        if n_classes > spec.classes:
            raise ValueError(
                f"bed {bed.name!r} carries label {n_classes - 1}, outside "
                f"{spec.name}'s {spec.classes} classes")
        if n_classes < spec.classes:
            raise ValueError(
                f"bed {bed.name!r} spans only {n_classes}/{spec.classes} "
                f"classes — a class is absent from ytr, which would silently "
                f"under-size the readout (raise train_n or reseed the subset)")
        n_classes = spec.classes        # size the readout from the SPEC

    # 2. input-mode parity
    if input_mode == "trigram":
        if channels not in VALID_CHANNELS:
            raise ValueError(
                f"trigram stem needs channels in {VALID_CHANNELS}, got "
                f"{channels} (3 -> RGB channel trigram, 1 -> spatial trigram)")
        if pixels % channels:
            raise ValueError(f"pixels={pixels} must be divisible by "
                             f"channels={channels}")
    elif input_mode == "linear":
        if tokens < 1 or pixels % tokens:
            raise ValueError(f"linear mode needs tokens>=1 dividing "
                             f"pixels={pixels}, got tokens={tokens}")
    else:
        raise ValueError(f"unknown input_mode {input_mode!r}; "
                         "expected 'linear' or 'trigram'")

    # 3. readout-width guard (the multi-GB flatten that WDDM-spills)
    if input_mode == "trigram":
        T = _trigram_tokens(pixels, channels)
        flat = min(readout_dim, d) * T
    else:
        T = tokens
        flat = d * T
    if flat > READOUT_MAX:
        raise ValueError(
            f"readout would flatten to width {flat} (d={d}, T={T}, "
            f"mode={input_mode}) — over the {READOUT_MAX} guard. Lower d or "
            "tokens, or use trigram mode (its readout_dim bottleneck keeps "
            "the flatten small at any d)")

    # 4. identity + per-dataset trigram window
    name_or_path = trunk_identity(bed.name_key)
    lo, hi = (spec.trigram_lo, spec.trigram_hi) if spec else (-3.0, 3.0)

    torch.manual_seed(getattr(cfg, "seed", 0) if seed is None else seed)
    trunk = TinyTrunk(TinyConfig(
        hidden_size=d, n_blocks=n_blocks, tokens=tokens, pixels=pixels,
        channels=channels, n_classes=n_classes, input_mode=input_mode,
        n_bins=n_bins, readout_dim=readout_dim, _name_or_path=name_or_path,
        trigram_lo=lo, trigram_hi=hi))
    # THE MODEL FOLLOWS THE DATA. The bed is the thing this trunk must
    # consume, so placing the trunk anywhere else is always a bug. Deriving
    # the device here (instead of trusting a caller's `.to(...)`) removes
    # the whole "index is on cuda:0, other tensors on cpu" class — which is
    # exactly what an explicit `.to(cfg.device or 'cpu')` produced once
    # `cfg.device` became lazily resolved (i.e. "" by default).
    return trunk.to(bed.xtr.device)
