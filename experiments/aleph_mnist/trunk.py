"""TinyTrunk — a 4-block linear MNIST classifier, shaped so amoe can
attach to it unmodified.

Three things make this a valid amoe substrate rather than a toy:

1. RESIDUAL STREAM. Each block is pre-norm `x + W2 sq(W1 LN(x))` over a
   (B, T, d) stream — the same object a decoder block hands the adapter.
   "Linear" here means linear-algebraic: no attention, no convolution.
2. A CONFIG SHIM. `binding.PathBinding` reads `model.config.hidden_size`
   and `attach(strict=...)` reads `_name_or_path`, so the trunk carries
   an HF-shaped config object.
3. AN `input_ids` PROBE PATH. `runtime.attach._probe` fingerprints a
   model by calling `model(input_ids=...)` with an int tensor — it is
   written for causal LMs. Rather than fork attach(), the trunk accepts
   `input_ids` and maps it DETERMINISTICALLY to a float input, which
   makes the bit-exact detach guarantee testable here for real. The
   LM-shaped probe is a portability wart in amoe 0.2.2, noted in the
   experiments README.

NO GLOBAL AVERAGE POOLING anywhere (house law: GAP collapsed a geometric
encoder 70% -> 29%, replicated). The T>1 readout flattens.

CAPACITY IS DELIBERATELY STARVED (d=64, 4096 train rows by default).
Full MNIST on a 4-block MLP saturates ~98% and compresses every arm
difference into seed noise; a starved bed is the measurement instrument.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    """HF-shaped enough for amoe's resolver, binding and strict check."""
    hidden_size: int = 64
    model_type: str = "tiny_mnist"
    _name_or_path: str = "tiny-mnist-4block"
    n_blocks: int = 4
    tokens: int = 1
    n_classes: int = 10
    pixels: int = 784


@dataclass
class TrunkOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class LinearBlock(nn.Module):
    """Pre-norm residual MLP block — the miniature of a decoder block
    with the attention removed. SquaredReLU matches the adapter's own
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
        d, T = cfg.hidden_size, cfg.tokens
        if cfg.pixels % T:
            raise ValueError(f"tokens={T} must divide pixels={cfg.pixels}")
        self.patch = cfg.pixels // T
        self.stem = nn.Linear(self.patch, d)
        self.blocks = nn.ModuleList([LinearBlock(d)
                                     for _ in range(cfg.n_blocks)])
        self.norm = nn.LayerNorm(d)
        # flatten, never mean-pool (house law)
        self.readout = nn.Linear(d * T, cfg.n_classes)

    # -- the probe shim ------------------------------------------------
    def _from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Deterministic int -> float image map, so amoe's LM-shaped
        fingerprint probe works on a vision trunk. Pure function of the
        ids: same ids, same activations, bit for bit."""
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
        x = x.reshape(x.shape[0], self.config.tokens, self.patch)
        h = self.stem(x)
        for blk in self.blocks:
            out = blk(h)
            h = out[0] if isinstance(out, tuple) else out
        h = self.norm(h).reshape(h.shape[0], -1)
        logits = self.readout(h)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return TrunkOutput(logits=logits, loss=loss)

    # -- the dial ------------------------------------------------------
    def set_trainable_blocks(self, n: int) -> dict:
        """THE DIAL. Unfreeze the LAST `n` blocks (0 = fully frozen
        substrate, n_blocks = full co-training). stem/norm/readout follow
        the trunk: they are trunk, not adapter.

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
            for m in (self.stem, self.norm, self.readout):
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
                seed: int = 0, pixels: int = 784,
                n_classes: int = 10) -> TinyTrunk:
    torch.manual_seed(seed)
    return TinyTrunk(TinyConfig(hidden_size=d, n_blocks=n_blocks,
                                tokens=tokens, pixels=pixels,
                                n_classes=n_classes))
