"""DiffusionTrainConfig — the certified recipes as the only knobs.

Deliberately ABSENT (laws, not options): optimizer choice (pure Adam,
wd=0), top-k/load-balancing (no comparative selectors), band edges/xfade
(law constants, exp008).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..laws import BLOB_LAMBDA, FLOW_SHIFT


@dataclass
class RelaySpec:
    n_slots: int = 16
    K: int = 64
    tau: float = 0.1
    hidden: int = 178


@dataclass
class DiffusionTrainConfig:
    name: str = "anchor"
    objective: str = "eps"            # "eps" | "flow"
    shift: float = FLOW_SHIFT         # flow sigma warp (exp013)
    adapter: str = "relay"            # "relay" | "multiband3"
    relay: RelaySpec = field(default_factory=RelaySpec)
    rank: int = 16                    # multiband3 per-band rank
    band_roles: bool = False          # exp009 HP/LP role pressure (measured
                                      # inert at gauge scale — kept honest)
    blob: bool = False                # foreground-LP-x0 supervision
    blob_lambda: float = BLOB_LAMBDA  # λ≈1: the 3-point dose-curve operating
                                      # point (exp013); 0.5/2.0 are the
                                      # mapped under/over-couple arms
    force_blob_on_eps: bool = False   # conditioning law: eps refuses blob
                                      # unless forced (exp012, 2-seed inert)
    steps: int = 3000
    batch_size: int = 16
    lr: float = 1e-3
    cfg_dropout: float = 0.1
    seed: int = 0
    log_every: int = 50
    base_schedule_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
