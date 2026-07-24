"""AlephRoutedAttention — the aleph address AS the token mixer.

This is the piece the linear/trigram trunks were missing. Those trunks do
NO cross-token mixing: every block is a per-token MLP, so with T tokens the
model is provably a generalized additive model over single tokens and all
spatial integration is dumped on the final readout. An aleph adapter bolted
onto that reads a stream that was never mixed — the opposite of how
RelayPatchwork rides a real host (Qwen-VL / GPT-2 decoder layers, whose
self-attention has already mixed the stream).

The house law forbids the obvious fix: geometry ERODES through standard
(softmax) attention, so the answer is "a dedicated pathway, not injection".
This module IS that pathway. The mixer is not softmax — it is the signed
projective aleph address used as a linear-attention feature map:

    score(i, j) = p+_q(i) · p+_k(j)  +  p-_q(i) · p-_k(j)

where p+/p- are the exact soft address over the 2K oriented half-axes of a
K-atom S^(D-1) codebook (the SAME sinh/cosh antipodal read as
AlephAddress.m_hat, here computed as a normalized softmax so the kernel is
non-negative and the linear-attention denominator is strictly positive).
Because the score factors through two K-wide memories it runs as pure GEMM
in O(S * K), no S x S matrix — the `_hub_full` recurrence.

So the trunk mixes, the mixing is geometric (non-eroding), and the aleph
geometry the RelayPatchwork adapter reads has been mixed by MORE aleph
geometry rather than by foreign attention. Cross-token mixing now exists;
the model is no longer additive.

PROVENANCE: the hub kernel and the address/projection conventions are lifted
from acd_attention.py::ACDRoutedAttention (AbstractPhil + Mirel, MIT,
https://huggingface.co/AbstractPhil/geolip-aleph-differentiation), focused
here to the single-stage, non-causal case a patch grid needs (every patch
sees every patch; there is no causal order over an image). The exact
multi-stage / streaming / causal machinery is deliberately not ported — this
bed does not need it, and a faithful subset is easier to audit than a
transplanted superset.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoutedAttnConfig:
    """Geometry matched to the RelayPatchwork adapter: K=64 atoms on S^3,
    D_addr=4, tau=0.1 — so the trunk's mixer and the adapter's read speak the
    same address dialect. `num_heads` must divide `dim`."""
    dim: int = 64
    num_heads: int = 4
    K: int = 64
    D_addr: int = 4
    tau: float = 0.1
    causal: bool = False          # a patch grid has no causal order
    qkv_bias: bool = False
    out_bias: bool = True
    eps: float = 1e-6
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.dim % self.num_heads:
            raise ValueError(
                f"dim={self.dim} must be divisible by num_heads="
                f"{self.num_heads}")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


class AlephRoutedAttention(nn.Module):
    """Signed-projective aleph routing as linear attention. Input and output
    are both (B, S, dim); no positional requirement, no pre-norm (the module
    sphere-projects its own address rows), fp32."""

    def __init__(self, cfg: RoutedAttnConfig):
        super().__init__()
        self.cfg = cfg
        self.H, self.hd, self.Da, self.tau = (
            cfg.num_heads, cfg.head_dim, cfg.D_addr, cfg.tau)
        addr_out = self.H * self.Da
        self.q_addr = nn.Linear(cfg.dim, addr_out, bias=cfg.qkv_bias)
        self.k_addr = nn.Linear(cfg.dim, addr_out, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim, bias=cfg.out_bias)
        # orthogonal init on the address projections — the load-bearing
        # convention from the stock router (and from RelayPatchwork.proj).
        g = torch.Generator().manual_seed(cfg.seed)
        for lin in (self.q_addr, self.k_addr):
            nn.init.orthogonal_(lin.weight)
        # the routing codebook: K atoms on S^(D_addr-1), co-trained trunk
        # state (this is trunk, not adapter — the adapter carries its own).
        book = F.normalize(torch.randn(cfg.K, self.Da, generator=g), dim=-1)
        self.codebook = nn.Parameter(book)

    # -- address pathway ------------------------------------------------
    def _rows(self, t: torch.Tensor, B: int, S: int) -> torch.Tensor:
        """(B, S, H*Da) -> sphere rows (B, H, S, Da)."""
        t = t.view(B, S, self.H, self.Da).transpose(1, 2)
        return F.normalize(t, dim=-1)

    def _address(self, xh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact soft read over 2K oriented half-axes [+A; -A], antipodally
        factored and max-subtracted for stability (Z >= 1). Returns
        (p_plus, p_minus), each (B, H, S, K), summing to 1 jointly."""
        A = F.normalize(self.codebook, dim=-1)              # (K, Da)
        u = (xh @ A.t()) * (1.0 / self.tau)                 # (B, H, S, K)
        mm = u.abs().amax(dim=-1, keepdim=True)
        ep, en = torch.exp(u - mm), torch.exp(-u - mm)
        Z = (ep + en).sum(dim=-1, keepdim=True)
        return ep / Z, en / Z

    # -- hub kernel (non-causal; verbatim math from acd_attention) -------
    def _hub_full(self, pq_p, pq_m, pk_p, pk_m, v) -> torch.Tensor:
        """out_i = num_i / den_i, where score(i,j) factors through two
        K-wide memories. p*: (B,H,S,K), v: (B,H,S,hd) -> (B,H,S,hd)."""
        Mp = torch.einsum('bhsk,bhsd->bhkd', pk_p, v)
        Mm = torch.einsum('bhsk,bhsd->bhkd', pk_m, v)
        zp = pk_p.sum(dim=2)
        zm = pk_m.sum(dim=2)
        num = (torch.einsum('bhsk,bhkd->bhsd', pq_p, Mp)
               + torch.einsum('bhsk,bhkd->bhsd', pq_m, Mm))
        den = (torch.einsum('bhsk,bhk->bhs', pq_p, zp)
               + torch.einsum('bhsk,bhk->bhs', pq_m, zm))
        return num / den.unsqueeze(-1).clamp_min(self.cfg.eps)

    def _hub_causal(self, pq_p, pq_m, pk_p, pk_m, v) -> torch.Tensor:
        """Lower-triangular masked version (kept for parity with the stock
        router; unused on a patch grid). Materializes the S x S kernel, so
        it is O(S^2) — fine for the short sequences this bed uses."""
        score = (torch.einsum('bhik,bhjk->bhij', pq_p, pk_p)
                 + torch.einsum('bhik,bhjk->bhij', pq_m, pk_m))
        S = score.shape[-1]
        tri = torch.tril(torch.ones(S, S, device=v.device, dtype=v.dtype))
        score = score * tri
        den = score.sum(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
        return (score / den) @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qh = self._rows(self.q_addr(x), B, S)
        kh = self._rows(self.k_addr(x), B, S)
        v = self.v_proj(x).view(B, S, self.H, self.hd).transpose(1, 2)
        pq_p, pq_m = self._address(qh)
        pk_p, pk_m = self._address(kh)
        if self.cfg.causal:
            out = self._hub_causal(pq_p, pq_m, pk_p, pk_m, v)
        else:
            out = self._hub_full(pq_p, pq_m, pk_p, pk_m, v)
        out = out.transpose(1, 2).reshape(B, S, self.cfg.dim)
        return self.out_proj(out)
