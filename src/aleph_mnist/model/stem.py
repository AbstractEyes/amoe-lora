"""TrigramStem — `byte_emb x3`, the canonical aleph input.

Discovery #16: **CHANNEL COUNT = N-GRAM ORDER.** A single linear projection
of a normalized pixel hands the address a *unigram* — one value per
position, nothing three-way to bind — so the addressed read cannot pull
ahead of a passthrough. That, not "the aleph is inert on vision," is why
the linear-stem climb tied `soft == none` everywhere. The trigram lineage
(AlephLM `byte_emb x3`; L-AR5: -10% bpb, and it *differentially* benefits
the ADDRESSED head) restores the structure.

Two forms, per the research:
  channel  RGB pixel is a natural byte-trigram (R,G,B) — embed each channel
           with its own table and sum. "byte-trigram-as-RGB engaged first
           try" (tri_band_omega_arc). T = H*W tokens.
  spatial  grayscale has no channel trigram, so form the sequence one:
           emb0(px_t) + emb1(px_{t-1}) + emb2(px_{t-2}), past-only — the
           exact AlephLM form with pixels as the bytes. T = pixels.

The quantization window is passed in from the dataset's `DatasetSpec` (it
used to be a hardcoded [-3, 3] for every dataset) and the channel-major
layout is asserted, not assumed — both via `data.binning`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.binning import assert_channel_major, bin_to_bytes

TRIGRAM_ORDER = 3          # discovery #16 — the n in "n-gram"
VALID_CHANNELS = (1, 3)    # 3 -> channel trigram (RGB); 1 -> spatial trigram


class TrigramStem(nn.Module):
    def __init__(self, d: int, channels: int, pixels: int,
                 n_bins: int = 256, lo: float = -3.0, hi: float = 3.0):
        super().__init__()
        if channels not in VALID_CHANNELS:
            raise ValueError(
                f"trigram stem needs channels in {VALID_CHANNELS}, got "
                f"{channels} (3 -> RGB channel trigram, 1 -> spatial trigram)")
        if pixels % channels:
            raise ValueError(f"pixels={pixels} must be divisible by "
                             f"channels={channels}")
        self.d, self.channels, self.n_bins = d, channels, n_bins
        self.lo, self.hi = lo, hi
        self.kind = "channel" if channels == 3 else "spatial"
        self.hw = pixels // channels
        self.n_tokens = self.hw if self.kind == "channel" else pixels
        # +1 row = the past-only PAD index (spatial form), so pre-sequence
        # positions carry a learned vector rather than a hard zero.
        self.embs = nn.ModuleList([nn.Embedding(n_bins + 1, d)
                                   for _ in range(TRIGRAM_ORDER)])
        for e in self.embs:
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.kind == "channel":                       # RGB byte-trigram
            assert_channel_major(x, self.channels, self.hw)
            v = x.view(B, self.channels, self.hw)        # (B,3,HW)
            idx = bin_to_bytes(v, self.lo, self.hi, self.n_bins)
            return sum(self.embs[c](idx[:, c]) for c in range(self.channels))
        idx = bin_to_bytes(x, self.lo, self.hi, self.n_bins)   # (B,T) raster
        T = idx.shape[1]
        out = self.embs[0](idx)                          # emb0(px_t)
        for k in (1, 2):                                 # + emb_k(px_{t-k})
            shifted = torch.full((B, T), self.n_bins, dtype=torch.long,
                                 device=x.device)
            shifted[:, k:] = idx[:, :T - k]
            out = out + self.embs[k](shifted)
        return out                                       # (B,T,d)
