"""Train/Align configs — certified campaign defaults, plain dataclasses.

Deliberately absent fields (see amoe.laws): optimizer choice, weight
decay, top-k, load-balancing coefficients. They are not options.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.adapter import AdapterSpec


@dataclass
class TrainConfig:
    name: str = "anchor"
    steps: int = 1200
    batch_size: int = 8
    lr: float = 1e-3
    seed: int = 0
    sites: str | tuple[int, ...] = "all"
    max_len: int = 1024
    precision: str = "fp32"      # "bf16" = documented fallback
    grad_checkpointing: bool = True
    log_every: int = 50
    adapter: AdapterSpec = field(default_factory=AdapterSpec)


@dataclass
class AlignConfig:
    steps: int = 600
    batch_size: int = 8
    lr: float = 1e-3
    seed: int = 0
    emb: int = 64
    tau: float = 0.1
    train_new_anchor: bool = False   # exp010 warning if True
    check_every: int = 500
    usage_ppl_floor: float = 1.5
    usage_min: float = 0.02
    max_strikes: int = 3
