"""AddressedConv2d — a convolution whose per-position kernel is composed from
a learnable filter bank by the ALEPH ADDRESS, and a small ConvTrunk built
from it.

WHY. The aleph line diverged from convolution: the patch-ViT bed lost to a
plain conv on small vision data because it threw away conv's locality /
weight-sharing / edge prior. This brings the geometry back INSIDE the conv.
A bank of `k_bank` filters is convolved with the input; at every position the
aleph read produces a convex weight over the branches, so the effective
kernel is `W[p] = Sum_k a_k[p] F_k` — a content-steered, still-local,
still-weight-shared convolution.

THE LOAD-BEARING PROPERTY. Convolution is linear in the kernel, so for ANY
position-independent address `a`,
    y = Sum_k a_k (F_k * x) = (Sum_k a_k F_k) * x + Sum_k a_k b_k
is a single ordinary Conv2d. The `none` arm hard-sets `a_k = 1/k_bank`, so it
IS a plain conv with the mean filter. AddressedConv2d is therefore a STRICT
GENERALIZATION of nn.Conv2d, and any `soft - none` delta is attributable to
the address alone — the "no champion if not using alephs" control, at the
conv level. `test_addr_conv_smoke.py` asserts the reduction to fp32 tol.

FOUR TRAPS THAT SILENTLY VOID THE REDUCTION (guarded here, do not add):
  1. a per-branch nonlinearity before the weighted sum (activation must come
     AFTER, in the enclosing block);
  2. BatchNorm / any per-branch norm in the bank (also house-law-banned on the
     geometric path);
  3. computing `none` by reading the address and hoping it is uniform (fp32
     non-uniformity leaks) — `none` HARD-SETS the constant, it does not read;
  4. a content-dependent k_addr->k_bank bridge — the bridge is a fixed,
     address-independent group-sum buffer.

The address itself reuses the amoe closed form: `AlephAddress` (codebook on
S^3, D=4) and the same antipodal sinh/cosh read used by the router
(routed_attention._address / address.m_hat).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from amoe.core.address import AlephAddress

from .heads import super_fibonacci_s3
from .trunk import SquaredReLU, TrunkOutput

ADDR_MODES = ("soft", "sign", "none", "learned", "off")


@dataclass
class AddrConvSpec:
    """Geometry + shape of one addressed conv layer. `k_addr` (codebook atoms
    on S^3) and `k_bank` (conv filter branches = the cost) are DISTINCT; the
    first cut ties them (k_addr == k_bank) so one atom indexes one filter with
    no bridge. Decoupling to a richer k_addr uses a fixed group-sum bridge."""
    c_in: int
    c_out: int
    kernel: int = 3
    k_bank: int = 16
    k_addr: int = 16          # == k_bank ties atoms 1:1 to filters
    d_addr: int = 4           # S^3
    n_slots: int = 4          # address slots read per position, then averaged
    tau: float = 0.1

    def __post_init__(self) -> None:
        if self.k_bank < 1:
            raise ValueError(f"k_bank must be >= 1, got {self.k_bank}")
        if self.k_addr % self.k_bank:
            raise ValueError(
                f"k_addr ({self.k_addr}) must be a multiple of k_bank "
                f"({self.k_bank}) for the fixed group-sum bridge")
        if self.kernel % 2 == 0:
            raise ValueError(f"kernel must be odd for 'same' padding, got "
                             f"{self.kernel}")


class AddressedConv2d(nn.Module):
    """One addressed convolution. Input/output are (B, C_in/C_out, H, W);
    stride is fixed at 1 with 'same' padding (spatial reduction belongs in an
    explicit pool, so a constant address always factors cleanly through the
    reduction). The activation is NOT applied here — the enclosing block adds
    it AFTER the address-weighted sum (trap #1)."""

    def __init__(self, spec: AddrConvSpec, mode: str = "soft",
                 codebook_init: str = "random"):
        super().__init__()
        if mode not in ADDR_MODES:
            raise ValueError(f"mode must be one of {ADDR_MODES}, got {mode!r}")
        s = spec
        self.spec, self.mode = s, mode
        self.pad = s.kernel // 2

        # the filter bank: k_bank ordinary conv kernels, applied as ONE grouped
        # cuDNN call (reshape to (k_bank*C_out, C_in, k, k)).
        self.weight = nn.Parameter(torch.empty(
            s.k_bank, s.c_out, s.c_in, s.kernel, s.kernel))
        self.bias = nn.Parameter(torch.zeros(s.k_bank, s.c_out))
        for k in range(s.k_bank):                 # per-filter Kaiming, as Conv2d
            nn.init.kaiming_uniform_(self.weight[k], a=5 ** 0.5)

        # The address pathway is built for soft/sign/none ALIKE, so the three
        # arms are BYTE-IDENTICAL in parameter count ("no champion if not
        # param-matched"). `none` simply never reads it (see _address), so its
        # codebook and slot_proj receive exactly zero gradient — the same
        # "present but unread" control the amoe `none` arm uses.
        if mode in ("soft", "sign", "none"):
            self.slot_proj = nn.Conv2d(s.c_in, s.n_slots * s.d_addr, 1,
                                       bias=False)
            nn.init.orthogonal_(self.slot_proj.weight.view(
                s.n_slots * s.d_addr, -1))
            self.addr = AlephAddress(s.k_addr, s.d_addr, s.tau)
            if codebook_init == "fibonacci":
                with torch.no_grad():
                    self.addr.codebook.copy_(super_fibonacci_s3(s.k_addr).to(
                        self.addr.codebook))
                    self.addr.home.copy_(F.normalize(
                        self.addr.codebook.detach(), dim=-1))
            # fixed, address-INDEPENDENT bridge k_addr -> k_bank (trap #4)
            groups = s.k_addr // s.k_bank
            bridge = torch.zeros(s.k_bank, s.k_addr)
            for j in range(s.k_bank):
                bridge[j, j * groups:(j + 1) * groups] = 1.0
            self.register_buffer("bridge", bridge)
        elif mode == "learned":                   # non-aleph dynamic control
            self.slot_proj = nn.Conv2d(s.c_in, s.n_slots * s.d_addr, 1,
                                       bias=False)
            self.gate = nn.Linear(s.n_slots * s.d_addr, s.k_bank)
        # mode == "off": plain conv, no address parameters at all.

    # -- the address maps (B, C_in, H, W) -> (B, k_bank, H, W), rows sum to 1
    def _usage(self, x: torch.Tensor) -> torch.Tensor:
        """Per-position, per-atom antipodal soft read, bridged to k_bank.
        `g = (e^u + e^-u)/Z` is the total mass on each atom (both half-axes) —
        the same closed form as address.m_hat, kept as the distribution."""
        B, _, H, W = x.shape
        s = self.spec
        slots = self.slot_proj(x).view(B, s.n_slots, s.d_addr, H, W)
        slots = F.normalize(slots, dim=2)
        A = F.normalize(self.addr.codebook, dim=-1)          # (k_addr, D)
        # (B, n_slots, k_addr, H, W)
        u = torch.einsum("bsdhw,kd->bskhw", slots, A) / s.tau
        m = u.abs().amax(dim=2, keepdim=True)
        g = (torch.exp(u - m) + torch.exp(-u - m))
        g = g / g.sum(dim=2, keepdim=True)                   # atom usage, sums 1
        g = g.mean(dim=1)                                    # avg slots -> (B,k_addr,H,W)
        return torch.einsum("jk,bkhw->bjhw", self.bridge, g)  # -> (B,k_bank,H,W)

    def _address(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        kb = self.spec.k_bank
        if self.mode == "none":
            # HARD-SET uniform (trap #3): never read the codebook.
            return x.new_full((B, kb, H, W), 1.0 / kb)
        if self.mode == "learned":
            slots = self.slot_proj(x).flatten(2).transpose(1, 2)   # (B,HW,ns*D)
            a = self.gate(slots).softmax(-1)                       # (B,HW,k_bank)
            return a.transpose(1, 2).view(B, kb, H, W)
        g = self._usage(x)                                          # soft usage
        if self.mode == "sign":                                    # hard + STE
            idx = g.argmax(dim=1, keepdim=True)
            hard = torch.zeros_like(g).scatter_(1, idx, 1.0)
            return hard + (g - g.detach())
        return g                                                   # soft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spec
        w = self.weight.reshape(s.k_bank * s.c_out, s.c_in, s.kernel, s.kernel)
        Y = F.conv2d(x, w, self.bias.reshape(-1), stride=1, padding=self.pad)
        if self.mode == "off":                     # plain conv: mean of the bank
            return Y.view(x.shape[0], s.k_bank, s.c_out, *Y.shape[-2:]).mean(1)
        Y = Y.view(x.shape[0], s.k_bank, s.c_out, *Y.shape[-2:])
        a = self._address(x)                       # (B, k_bank, H, W)
        return torch.einsum("bkhw,bkchw->bchw", a, Y)

    @torch.no_grad()
    def address_usage(self, x: torch.Tensor) -> dict:
        """Read-only Law-2 gauge: how NON-uniform is the address? mean KL(a ||
        uniform) ~= 0 means the read is inert (soft behaves like none); plus
        per-branch usage perplexity and alive-count. Returns {} for modes with
        no read."""
        if self.mode in ("none", "off"):
            return {}
        a = self._address(x).clamp_min(1e-12)
        kb = self.spec.k_bank
        kl = (a * (a * kb).log()).sum(1).mean()             # mean_p KL(a||unif)
        usage = a.mean(dim=(0, 2, 3))
        usage = usage / usage.sum()
        ppl = float(torch.exp(-(usage * usage.log()).sum()))
        alive = int((usage > 1e-3 / kb).sum())
        return {"kl_to_uniform": float(kl), "usage_ppl": ppl,
                "branches_alive": alive, "branches": kb}


def _groups(c: int) -> int:
    """Largest of {8,4,2,1} dividing c — GroupNorm groups (never BatchNorm)."""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """AddressedConv2d -> GroupNorm -> SquaredReLU -> optional MaxPool.

    The activation is here, AFTER the addressed conv's per-position weighted
    sum (trap #1: never per-branch). GroupNorm, not BatchNorm — per-sample, no
    batch coupling, and it sits on the content path after the address sum, so
    it does not touch the soft-vs-none attribution."""

    def __init__(self, spec: AddrConvSpec, mode: str, codebook_init: str,
                 pool: bool):
        super().__init__()
        self.conv = AddressedConv2d(spec, mode, codebook_init)
        self.norm = nn.GroupNorm(_groups(spec.c_out), spec.c_out)
        self.act = SquaredReLU()
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.norm(self.conv(x))))


class ConvGenHead(nn.Module):
    """1x1 conv -> (B, n_bins, H, W): a per-pixel categorical distribution.
    Phase-2 generative head — the addressed-conv output parameterizes the
    pixel distribution, which is what makes the address load-bearing (Law 2)."""

    def __init__(self, c_in: int, n_bins: int = 16):
        super().__init__()
        self.n_bins = n_bins
        self.proj = nn.Conv2d(c_in, n_bins, 1)

    def forward(self, x):
        return self.proj(x)


class ConvConfig:
    """Plain-attribute config for a ConvTrunk (built by build_conv_trunk /
    the conv runner; not an HF-shaped shim — this bed has no adapter)."""

    def __init__(self, *, channels=1, height=28, width=28, n_classes=10,
                 addr_mode="soft", k_bank=16, k_addr=16, kernel=3,
                 conv_channels=32, conv_layers=2, n_slots=4, tau=0.1,
                 codebook_init="random", objective="classify", n_bins=16,
                 seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]


class ConvTrunk(nn.Module):
    """A small addressed-conv CNN. Drop-in for TinyTrunk at the harness
    interface: forward(x | input_ids, labels) -> TrunkOutput(logits, loss),
    so probes.evaluate / the ledger / verdict / publish all work unchanged.

    Every arm (soft/sign/none/learned/off) is the SAME architecture with the
    block's address `mode` switched — identical stem/pools/norm/readout — so a
    delta is attributable to the address (the arm-ladder discipline, at the
    conv level). NO GAP (flatten readout, house law); NO BatchNorm."""

    def __init__(self, cfg: ConvConfig):
        super().__init__()
        self.cfg = cfg
        self.C, self.H, self.W = cfg.channels, cfg.height, cfg.width
        self.pixels = cfg.channels * cfg.height * cfg.width
        self.addr_mode = cfg.addr_mode
        self.objective = cfg.objective
        torch.manual_seed(cfg.seed)

        classify = cfg.objective == "classify"
        chs = [cfg.channels] + [cfg.conv_channels * (2 ** i)
                                for i in range(cfg.conv_layers)]
        self.blocks = nn.ModuleList([
            ConvBlock(
                AddrConvSpec(chs[i], chs[i + 1], kernel=cfg.kernel,
                             k_bank=cfg.k_bank, k_addr=cfg.k_addr,
                             n_slots=cfg.n_slots, tau=cfg.tau),
                cfg.addr_mode, cfg.codebook_init, pool=classify)
            for i in range(cfg.conv_layers)])

        if classify:                          # flatten (NO GAP) -> logits
            hf = cfg.height // (2 ** cfg.conv_layers)
            wf = cfg.width // (2 ** cfg.conv_layers)
            self.norm = nn.GroupNorm(_groups(chs[-1]), chs[-1])
            self.readout = nn.Linear(chs[-1] * hf * wf, cfg.n_classes)
            self.gen_head = None
        else:                                 # per-pixel categorical (full res)
            self.norm = nn.GroupNorm(_groups(chs[-1]), chs[-1])
            self.readout = None
            self.gen_head = ConvGenHead(chs[-1], cfg.n_bins)

    # -- probe shim: deterministic int -> flat float image (harness uniformity)
    def _from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        base = torch.arange(self.pixels, device=ids.device,
                            dtype=torch.float32)
        seed = ids.float().sum(dim=-1, keepdim=True)
        return torch.sin(0.01 * base.unsqueeze(0) + seed)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = x.view(x.shape[0], self.C, self.H, self.W)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

    def forward(self, x: torch.Tensor | None = None, *,
                input_ids: torch.Tensor | None = None,
                labels: torch.Tensor | None = None, **_) -> TrunkOutput:
        if x is None:
            if input_ids is None:
                raise ValueError("ConvTrunk needs x or input_ids")
            x = self._from_ids(input_ids)
        h = self.features(x)
        if self.objective == "classify":
            logits = self.readout(h.reshape(h.shape[0], -1))
            loss = None if labels is None else F.cross_entropy(logits, labels)
            return TrunkOutput(logits=logits, loss=loss)
        logits = self.gen_head(h)                      # (B, n_bins, H, W)
        loss = None
        if labels is not None:                         # labels = target bytes
            loss = F.cross_entropy(logits, labels)
        return TrunkOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def address_report(self, x: torch.Tensor) -> dict:
        """Block-0 address-usage gauge (KL-to-uniform etc.). Block 0 is the one
        whose address reads the raw pixels; empty for off/none — the Law-2
        readout only has meaning when a read happens."""
        return self.blocks[0].conv.address_usage(
            x.view(x.shape[0], self.C, self.H, self.W))

    def param_census(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return {"addr_mode": self.addr_mode, "params_total": total,
                "params_trainable": trainable}


def build_conv_trunk(bed, cfg) -> ConvTrunk:
    """Construct a ConvTrunk from a bed + RunConfig-like cfg, placing it on the
    bed's device (the model follows the data)."""
    spec = bed.spec
    if spec is not None:
        channels, height, width = spec.channels, spec.height, spec.width
        n_classes = spec.classes
    else:                                    # synthetic square bed
        channels = bed.channels
        side = int(round((bed.pixels / channels) ** 0.5))
        if side * side * channels != bed.pixels:
            raise ValueError(
                f"addr_conv needs a 2D shape; bed {bed.name!r} has "
                f"pixels={bed.pixels}, channels={channels} (not C*side^2)")
        height = width = side
        n_classes = bed.n_classes
    conv_cfg = ConvConfig(
        channels=channels, height=height, width=width, n_classes=n_classes,
        addr_mode=getattr(cfg, "mode", "soft"),
        k_bank=getattr(cfg, "k_bank", 16), k_addr=getattr(cfg, "k_addr", 16),
        kernel=getattr(cfg, "kernel_size", 3),
        conv_channels=getattr(cfg, "conv_channels", 32),
        conv_layers=getattr(cfg, "conv_layers", 2),
        n_slots=getattr(cfg, "n_slots", 4),
        tau=getattr(cfg, "tau", 0.1),
        codebook_init=getattr(cfg, "codebook_init", "random"),
        objective=getattr(cfg, "objective", "classify"),
        n_bins=getattr(cfg, "n_bins", 16),
        seed=getattr(cfg, "seed", 0))
    return ConvTrunk(conv_cfg).to(bed.xtr.device)
