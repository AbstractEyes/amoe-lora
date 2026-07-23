"""The arm ladder — one class, four address modes, IDENTICAL parameter
tensors.

This is the whole reason the bed can make a claim. Every arm carries the
same `proj`, the same 64x4 codebook, the same consume stack, the same
gate scalar; the only difference is how the 16 slot rows are READ. So a
delta between arms is attributable to the read and to nothing else — not
to width, not to depth, not to parameter count.

    soft    M_hat = sum_k sinh(u_k) A_k / sum_k cosh(u_k)   the aleph
    sign    hard oriented code, straight-through            the sign code
    none    M_hat = M (slots pass through, sphere-normed)   THE CONTROL
    frozen  soft read, codebook is not trainable            basin test

`none` is the canonical GATE CONTROL: the codebook parameter is still
present (so counts match exactly) but nothing reads it, so it receives no
gradient. It doubles as the param-matched MLP arm — proj -> linear ->
squared-ReLU -> LayerNorm -> linear is an MLP adapter.

WHY THE CONTROL IS MANDATORY, in Phil's own words from the Jun 19
codebook-pressure probe: a four-arm BERT sweep found gate-without-
codebook (.9980) TIED soft-fibonacci (.9979) — "pentachoron/fibonacci
addressing is NOT load-bearing for BERT recon", ruling "no champion if
not actually using alephs". Any MNIST result reported without this arm
is uninterpretable.

ON THE ARGMAX IN `sign`: this is HARD mode from the hosted checkpoints,
not the forbidden failure class. The failure class is COMPARATIVE
selection over a roster of alternatives (argmax anchors, softmax routing,
STE one-hots, VQ). Hard mode argmaxes over the model's OWN oriented
half-axes to read the committed sign code — "read the signs, not the
probabilities". Known counter-evidence: on BERT reconstruction, hard
destroyed fidelity (argmax tiles the continuum); on classification heads
sign beat soft in 12/12 frozen-substrate cells. Classification is the
regime where it has won.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from amoe.core.adapter import AdapterSpec, BlockWithAdapter, RelayPatchwork

MODES = ("soft", "sign", "none", "frozen")

PHI = math.sqrt(2.0)
PSI = 1.533751168755204288118041


def super_fibonacci_s3(n: int, device=None) -> torch.Tensor:
    """Super-Fibonacci spiral on S^3 (Alexa, CVPR'22) — the init that
    starts INSIDE the RP^3 attractor basin. D=4 only, which is exactly
    the adapter's address dimension."""
    i = torch.arange(n, dtype=torch.float64, device=device)
    s = (i + 0.5) / n
    r, R = torch.sqrt(s), torch.sqrt(1.0 - s)
    alpha = 2.0 * math.pi * i / PHI
    beta = 2.0 * math.pi * i / PSI
    q = torch.stack([r * torch.sin(alpha), r * torch.cos(alpha),
                     R * torch.sin(beta), R * torch.cos(beta)], dim=-1)
    return F.normalize(q.float(), dim=-1)


class PatchHead(RelayPatchwork):
    """RelayPatchwork with a switchable address read.

    State dict is byte-compatible with stock RelayPatchwork, so a `soft`
    artifact from this bed is a REAL amoe anchor that `amoe.attach` loads
    unmodified. The other modes share the layout but not the semantics —
    the checkpoint meta records `address_mode` and the notebook refuses
    to pass a non-soft anchor off as a stock one.
    """

    def __init__(self, d: int, spec: AdapterSpec | None = None, *,
                 mode: str = "soft", codebook_init: str = "random"):
        super().__init__(d, spec)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.codebook_init = codebook_init
        if codebook_init == "fibonacci":
            with torch.no_grad():
                self.addr.codebook.copy_(
                    super_fibonacci_s3(self.addr.K).to(self.addr.codebook))
        elif codebook_init != "random":
            raise ValueError("codebook_init must be 'random' or 'fibonacci'")
        # home is the drift gauge: re-snapshot AFTER any re-init, or every
        # drift number is measured against the wrong origin.
        with torch.no_grad():
            self.addr.home.copy_(
                F.normalize(self.addr.codebook.detach(), dim=-1))
        if mode == "frozen":
            self.addr.codebook.requires_grad_(False)

    # -- the four reads ------------------------------------------------
    def read(self, slots: torch.Tensor) -> torch.Tensor:
        if self.mode in ("soft", "frozen"):
            return self.addr.m_hat(slots)
        if self.mode == "none":
            return F.normalize(slots, dim=-1)          # M_hat = M
        A = F.normalize(self.addr.codebook, dim=-1)
        cos = F.normalize(slots, dim=-1) @ A.transpose(-1, -2)
        idx = cos.abs().argmax(dim=-1, keepdim=True)
        sgn = torch.sign(cos.gather(-1, idx))
        hard = sgn * A[idx.squeeze(-1)]
        soft = self.addr.m_hat(slots)
        return hard + (soft - soft.detach())           # straight-through

    def forward(self, x):
        B, n, _ = x.shape
        slots = self.proj(x).view(B, n, self.n_slots, self.spec.D)
        feats = self.read(slots).reshape(B, n, -1)
        return x + torch.sigmoid(self.gate) * self.consume(feats)

    # -- readouts ------------------------------------------------------
    @torch.no_grad()
    def slots_of(self, x: torch.Tensor) -> torch.Tensor:
        B, n, _ = x.shape
        return self.proj(x).view(B, n, self.n_slots, self.spec.D)

    @torch.no_grad()
    def sign_code(self, x: torch.Tensor) -> torch.Tensor:
        """Per-slot committed axis id in [0, 2K): the oriented half-axis
        the slot points hardest at. The discrete identity of a sample as
        this block sees it, regardless of mode."""
        A = F.normalize(self.addr.codebook, dim=-1)
        cos = F.normalize(self.slots_of(x), dim=-1) @ A.transpose(-1, -2)
        idx = cos.abs().argmax(dim=-1)
        neg = (cos.gather(-1, idx.unsqueeze(-1)).squeeze(-1) < 0).long()
        return idx * 2 + neg                       # (B, n, n_slots)


def spec_for(d: int) -> AdapterSpec:
    """Certified geometry, unchanged: 16 slots x D=4 over a 64-atom
    address, hidden 178, gate init -3."""
    return AdapterSpec()


def build_heads(trunk, mode: str = "soft", codebook_init: str = "random",
                sites: list[int] | None = None, seed: int = 0):
    """Wrap the trunk's blocks in BlockWithAdapter and return
    (heads, sites, wrappers). Wrapping mutates trunk.blocks in place."""
    torch.manual_seed(seed + 9973)      # heads seeded apart from the trunk
    d = trunk.config.hidden_size
    sites = list(range(len(trunk.blocks))) if sites is None else list(sites)
    dev = next(trunk.parameters()).device
    heads, blocks = [], list(trunk.blocks)
    for i in sites:
        h = PatchHead(d, spec_for(d), mode=mode,
                      codebook_init=codebook_init).to(dev)
        heads.append(h)
        blocks[i] = BlockWithAdapter(blocks[i], h)
    trunk.blocks = nn.ModuleList(blocks)
    wrappers = [b for b in blocks if isinstance(b, BlockWithAdapter)]
    return heads, sites, wrappers


def set_adapters(wrappers, enabled: bool) -> None:
    """The single-anchor half of the toggle law."""
    for w in wrappers:
        w.enabled = enabled


def anchor_state(heads, sites) -> dict:
    """Flatten to the amoe.anchor '{site}.{param}' layout. addr.home
    rides along because the drift gauge is required by the format."""
    state = {}
    for si, h in zip(sites, heads):
        for k, v in h.state_dict().items():
            state[f"{si}.{k}"] = v.detach().cpu()
    return state
