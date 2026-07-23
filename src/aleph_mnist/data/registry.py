"""The dataset registry — ONE entry per dataset.

Before this file the same dataset was described by four parallel dicts
(`DATASET_PIXELS`, `DATASET_CLASSES`, `DATASET_CHANNELS`, `_CROSS`) plus
loose normalization constants plus a branch in `_load`. Adding a dataset
meant editing five places and any one of them silently disagreeing with
the others produced a shape mismatch deep inside a forward pass.

Now a dataset IS a `DatasetSpec`. The model builder derives every shape
from it (`model/build.py`), so a disagreement is impossible rather than
merely unlikely.

CHANNELS IS THE TRIGRAM ORDER (discovery #16: channel count = n-gram
order). RGB is a natural byte-trigram; grayscale has none, so the trigram
stem forms a spatial one instead. That is why `channels` lives here and
not in the model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the loaders and the model builder need to agree on."""
    name: str
    height: int
    width: int
    channels: int                 # 1 gray, 3 rgb == trigram order
    classes: int
    mean: tuple[float, ...]       # per channel
    std: tuple[float, ...]        # per channel
    cross: str | None = None      # same-dim real domain used as a neutral set
    hf_repo: str | None = None    # fast HF parquet mirror, when one exists
    trigram_lo: float = -3.0      # byte-quantization window, PER DATASET
    trigram_hi: float = 3.0

    @property
    def pixels(self) -> int:
        return self.channels * self.height * self.width

    @property
    def hw(self) -> int:
        return self.height * self.width

    def __post_init__(self) -> None:
        if len(self.mean) != self.channels or len(self.std) != self.channels:
            raise ValueError(
                f"{self.name}: mean/std must have one entry per channel "
                f"(channels={self.channels}, mean={len(self.mean)}, "
                f"std={len(self.std)})")


DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        "mnist", 28, 28, 1, 10, (0.1307,), (0.3081,), cross="fashion"),
    "fashion": DatasetSpec(
        "fashion", 28, 28, 1, 10, (0.2860,), (0.3530,), cross="mnist"),
    "cifar10": DatasetSpec(
        "cifar10", 32, 32, 3, 10,
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616),
        hf_repo="uoft-cs/cifar10"),   # parquet on HF's CDN — the fast path
}


def get_spec(dataset: str) -> DatasetSpec:
    """Look up a spec, failing with a friendly message instead of a bare
    KeyError from deep inside a loader."""
    try:
        return DATASETS[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset {dataset!r}; known datasets: "
            f"{sorted(DATASETS)}") from None


# ── back-compat shims: the old flat dicts, derived from the one source ──
DATASET_PIXELS = {k: s.pixels for k, s in DATASETS.items()}
DATASET_CLASSES = {k: s.classes for k, s in DATASETS.items()}
DATASET_CHANNELS = {k: s.channels for k, s in DATASETS.items()}
