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
    input_mode: str = "linear"      # linear|trigram|patch|addr_conv|
    #                                 conv_tokens|antipode_conv
    # patch mode: a 2D-patch ViT whose token mixer is the aleph router (not
    # softmax). This is the substrate the architecture requires — the
    # linear/trigram trunks do no cross-token mixing, so they are additive
    # models and the adapter reads a stream that was never mixed.
    patch_size: int = 4             # P: a token is one C x P x P image region
    num_heads: int = 4             # routed-attention heads; must divide d
    # addressed-conv mode (input_mode="addr_conv"): a CNN whose per-position
    # conv kernel is composed from a filter bank by the aleph read. The `mode`
    # field above is the ADDRESS mode (soft|sign|none|learned|off), not an amoe
    # arm. This bed has no adapter — the arm ladder lives inside AddressedConv2d.
    k_bank: int = 16                # conv filter branches (the cost)
    k_addr: int = 16               # codebook atoms on S^3 (multiple of k_bank)
    tau: float = 0.1                # address read temperature (sharpness)
    kernel_size: int = 3            # conv kernel (odd, for 'same' padding)
    conv_channels: int = 32         # base width; block i has conv_channels*2^i
    conv_layers: int = 2            # addressed-conv blocks (also conv-stem depth)
    read_layers: int = 2            # conv_tokens: per-token antipode read blocks
    n_slots: int = 4                # address slots read per position
    objective: str = "classify"     # classify | generate (masked-pixel recon)
    n_bins: int = 16               # generative: byte levels per pixel
    # bookkeeping
    probe_every: int = 250
    log_every: int = 250
    device: str = ""                # "" -> resolve_device() at use time
    tag: str = ""

    @property
    def cell(self) -> str:
        return (f"{self.mode}/dial{self.trainable_blocks}/s{self.seed}"
                + (f"/{self.tag}" if self.tag else ""))
