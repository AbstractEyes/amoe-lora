"""`RunConfig` — the single configuration object.

Both faces of the package read this one dataclass: the API takes it
directly, and the CLI's flags default to its field defaults, so the two can
never drift apart.

DEVICE IS RESOLVED LAZILY. It used to be a dataclass field default
(`"cuda" if torch.cuda.is_available() else "cpu"`), which ran a CUDA probe
at *import* time and baked the answer into the class. Now the field is ""
and `resolve_device()` is called at the point of use.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def resolve_device(pref: str | None = None) -> str:
    """Explicit preference wins; otherwise pick CUDA if it is actually
    available. Called at use time, never at import time."""
    if pref:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RunConfig:
    mode: str = "soft"              # soft | sign | none | frozen | off
    trainable_blocks: int = 4       # THE DIAL: 0 = frozen substrate
    seed: int = 0
    # substrate
    d: int = 64
    n_blocks: int = 4
    tokens: int = 1
    readout_dim: int = 16           # per-token bottleneck (trigram readout)
    n_bins: int = 256               # byte levels for the trigram embed
    # data
    dataset: str = "mnist"
    train_n: int | None = 4096      # None = the full training set
    batch: int = 128
    synthetic: bool = False
    root: str = "./data"            # dataset cache/download root
    # schedule
    pretrain_steps: int = 1500      # 0 = trunk and head both move from step 0
    steps: int = 1500
    lr_head: float = 1e-3
    lr_trunk: float = 1e-4
    codebook_init: str = "random"   # random | fibonacci
    input_mode: str = "linear"      # linear | trigram (byte_emb x3)
    # bookkeeping
    probe_every: int = 250
    log_every: int = 250
    device: str = ""                # "" -> resolve_device() at use time
    tag: str = ""

    @property
    def cell(self) -> str:
        return (f"{self.mode}/dial{self.trainable_blocks}/s{self.seed}"
                + (f"/{self.tag}" if self.tag else ""))
