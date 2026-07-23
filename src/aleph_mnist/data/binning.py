"""Byte-quantization for the trigram stem, and the channel-major layout
contract — both in ONE place.

The trigram stem embeds discrete byte levels (`byte_emb x3`, the canonical
AlephLM input). Two things used to be hardcoded inside the stem and could
drift from the data without anything noticing:

  * the quantization window was a fixed [-3, 3] for every dataset;
  * the RGB path did `x.view(B, 3, hw)`, which is only correct if the flat
    vector is CHANNEL-MAJOR — an assumption the loaders had to honour
    silently.

Both now live here: the window comes from the dataset's `DatasetSpec`, and
the layout is asserted rather than assumed.
"""
from __future__ import annotations

import torch


def bin_to_bytes(x: torch.Tensor, lo: float, hi: float,
                 n_bins: int) -> torch.Tensor:
    """Map normalized floats to discrete byte levels in [0, n_bins-1].

    The embedding only needs same-value -> same-index and monotonicity in
    intensity, so a linear window is enough; values outside [lo, hi] clamp.
    """
    if hi <= lo:
        raise ValueError(f"binning window must have hi > lo, got {lo}..{hi}")
    idx = ((x - lo) / (hi - lo) * n_bins).long()
    return idx.clamp(0, n_bins - 1)


def assert_channel_major(x: torch.Tensor, channels: int, hw: int) -> None:
    """The RGB trigram reads `x.view(B, channels, hw)`, which is only the
    (R,G,B) triple per pixel if the flat vector is channel-major. Check it
    instead of trusting it."""
    if x.dim() != 2:
        raise ValueError(f"expected a flat (B, P) batch, got shape "
                         f"{tuple(x.shape)}")
    if x.shape[1] != channels * hw:
        raise ValueError(
            f"expected {channels * hw} flat features (channel-major "
            f"{channels}x{hw}), got {x.shape[1]} — the loader and the "
            "dataset spec disagree about layout")
