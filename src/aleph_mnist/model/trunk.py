"""TinyTrunk — a 4-block linear classifier, shaped so amoe can attach to it
unmodified.

Three things make this a valid amoe substrate rather than a toy:

1. RESIDUAL STREAM. Each block is pre-norm `x + W2 sq(W1 LN(x))` over a
   (B, T, d) stream — the same object a decoder block hands the adapter.
   "Linear" here means linear-algebraic: no attention, no convolution.
2. A CONFIG SHIM. `binding.PathBinding` reads `model.config.hidden_size`
   and `attach(strict=...)` reads `_name_or_path`, so the trunk carries an
   HF-shaped config object.
3. AN `input_ids` PROBE PATH. `runtime.attach._probe` fingerprints a model
   by calling `model(input_ids=...)` with an int tensor — it is written for
   causal LMs. Rather than fork attach(), the trunk accepts `input_ids` and
   maps it DETERMINISTICALLY to a float input, which makes the bit-exact
   detach guarantee testable here for real.

NO GLOBAL AVERAGE POOLING anywhere (house law: GAP collapsed a geometric
encoder 70% -> 29%, replicated). The readout flattens.

Construct trunks with `model.build.build_model(bed, cfg)` — it derives every
shape from the dataset and validates parity. `build_trunk` below is the
low-level constructor it delegates to; calling it directly is how shape
mismatches used to sneak in.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stem import TrigramStem


@dataclass
class TinyConfig:
    """HF-shaped enough for amoe's resolver, binding and strict check."""
    hidden_size: int = 64
    model_type: str = "tiny_mnist"
    _name_or_path: str = "tiny-mnist-4block"   # per-dataset; set by build_model
    n_blocks: int = 4
    tokens: int = 1
    n_classes: int = 10
    pixels: int = 784
    channels: int = 1               # 3 for RGB == the trigram order
    input_mode: str = "linear"      # "linear" | "trigram"
    n_bins: int = 256               # byte quantization for the trigram embed
    readout_dim: int = 16           # per-token bottleneck before the flatten
    trigram_lo: float = -3.0        # quantization window, PER DATASET
    trigram_hi: float = 3.0


@dataclass
class TrunkOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class LinearBlock(nn.Module):
    """Pre-norm residual MLP block — the miniature of a decoder block with
    the attention removed. SquaredReLU matches the adapter's own
    nonlinearity so the trunk and the patch head speak one dialect."""

    def __init__(self, d: int, mult: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.act = SquaredReLU()
        self.fc2 = nn.Linear(d * mult, d)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class TinyTrunk(nn.Module):
    """stem -> blocks (the attach sites) -> norm -> readout."""

    def __init__(self, config: TinyConfig | None = None):
        super().__init__()
        cfg = config or TinyConfig()
        self.config = cfg
        d = cfg.hidden_size
        self.trigram = cfg.input_mode == "trigram"
        if self.trigram:
            # byte_emb x3 stem: T is set by the trigram form, not cfg.tokens.
            # The blocks are per-token MLPs (no cross-token mixing), so spatial
            # aggregation lives entirely in the flatten readout — the aleph
            # read still operates per pixel-trigram, exactly as it does per
            # token in the AlephLM.
            self.stem = TrigramStem(d, cfg.channels, cfg.pixels, cfg.n_bins,
                                    cfg.trigram_lo, cfg.trigram_hi)
            T = self.stem.n_tokens
        else:
            T = cfg.tokens
            if cfg.pixels % T:
                raise ValueError(f"tokens={T} must divide pixels={cfg.pixels}")
            self.patch = cfg.pixels // T
            self.stem = nn.Linear(self.patch, d)
        self.n_tokens = T
        self.blocks = nn.ModuleList([LinearBlock(d)
                                     for _ in range(cfg.n_blocks)])
        self.norm = nn.LayerNorm(d)
        # Readout aggregates across tokens by FLATTEN (never mean-pool — house
        # law). At trigram T (784-1024) a d*T readout explodes: d=1024 -> a
        # ~1M-wide flatten, a ~2GB activation that WDDM-spills and crawls. So
        # trigram gets a per-token bottleneck d->r BEFORE the flatten — r*T
        # stays small and positional info survives (not GAP). Linear mode
        # (T=1, d*T=d) needs none and keeps its exact prior readout.
        if self.trigram:
            self.readout_dim = min(cfg.readout_dim, d)
            self.readout_proj = nn.Linear(d, self.readout_dim)
            self.readout = nn.Linear(self.readout_dim * T, cfg.n_classes)
        else:
            self.readout_proj = None
            self.readout = nn.Linear(d * T, cfg.n_classes)

    # -- the probe shim ------------------------------------------------
    def _from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Deterministic int -> float image map, so amoe's LM-shaped
        fingerprint probe works on a vision trunk. Pure function of the ids:
        same ids, same activations, bit for bit."""
        base = torch.arange(self.config.pixels, device=ids.device,
                            dtype=torch.float32)
        seed = ids.float().sum(dim=-1, keepdim=True)
        return torch.sin(0.01 * base.unsqueeze(0) + seed)

    def forward(self, x: torch.Tensor | None = None, *,
                input_ids: torch.Tensor | None = None,
                labels: torch.Tensor | None = None,
                **_) -> TrunkOutput:
        if x is None:
            if input_ids is None:
                raise ValueError("TinyTrunk needs x or input_ids")
            x = self._from_ids(input_ids)
        if self.trigram:
            h = self.stem(x)                       # (B, T, d) byte_emb x3
        else:
            x = x.reshape(x.shape[0], self.config.tokens, self.patch)
            h = self.stem(x)
        for blk in self.blocks:
            out = blk(h)
            h = out[0] if isinstance(out, tuple) else out
        h = self.norm(h)
        if self.readout_proj is not None:
            h = self.readout_proj(h)           # (B,T,d) -> (B,T,r) bottleneck
        logits = self.readout(h.reshape(h.shape[0], -1))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return TrunkOutput(logits=logits, loss=loss)

    # -- the dial ------------------------------------------------------
    def set_trainable_blocks(self, n: int) -> dict:
        """THE DIAL. Unfreeze the LAST `n` blocks (0 = fully frozen
        substrate, n_blocks = full co-training). stem/norm/readout follow the
        trunk: they are trunk, not adapter.

        Returns the parameter census, which goes in the ledger — an arm
        comparison is meaningless without it."""
        total = len(self.blocks)
        if not 0 <= n <= total:
            raise ValueError(f"trainable blocks must be in [0, {total}]")
        for p in self.parameters():
            p.requires_grad_(False)
        live = list(self.blocks[total - n:]) if n else []
        for m in live:
            for p in m.parameters():
                p.requires_grad_(True)
        if n:                       # a moving trunk owns its readout path
            mods = [self.stem, self.norm, self.readout]
            if self.readout_proj is not None:
                mods.append(self.readout_proj)
            for m in mods:
                for p in m.parameters():
                    p.requires_grad_(True)
        return {"trainable_blocks": n,
                "trunk_trainable": sum(p.numel() for p in self.parameters()
                                       if p.requires_grad),
                "trunk_total": sum(p.numel() for p in self.parameters())}

    def trunk_parameters(self) -> list[nn.Parameter]:
        """Trainable trunk params, EXCLUDING anything a wrapper added."""
        from amoe.core.adapter import BlockWithAdapter
        skip = set()
        for m in self.modules():
            if isinstance(m, BlockWithAdapter):
                skip |= {id(p) for p in m.adapter.parameters()}
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in skip]


def build_trunk(d: int = 64, n_blocks: int = 4, tokens: int = 1,
                seed: int = 0, pixels: int = 784, n_classes: int = 10,
                channels: int = 1, input_mode: str = "linear",
                name_or_path: str = "tiny-mnist-4block",
                trigram_lo: float = -3.0,
                trigram_hi: float = 3.0) -> TinyTrunk:
    """Low-level constructor. Prefer `build_model(bed, cfg)`, which derives
    and validates these fields from the dataset instead of trusting the
    caller to pass them consistently."""
    torch.manual_seed(seed)
    return TinyTrunk(TinyConfig(
        hidden_size=d, n_blocks=n_blocks, tokens=tokens, pixels=pixels,
        n_classes=n_classes, channels=channels, input_mode=input_mode,
        _name_or_path=name_or_path,
        trigram_lo=trigram_lo, trigram_hi=trigram_hi))
