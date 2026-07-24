"""Antipode-conv — a conv stem produces per-position tokens, and the SIGNED
aleph read is ADDED to each token. Built specifically so the antipodal
structure is load-bearing, after three prior beds discarded or mutilated it:

  patch-ViT      : the address read weak unigram tokens (linear stem).
  AddressedConv2d: the address only RE-WEIGHTED a filter-bank mean — a convex,
                   hull-bounded perturbation, and it collapsed the sign
                   (used 2cosh(u), even in u) AND averaged the slots. The mean
                   overrode it; soft == none.

What makes the antipode meaningful HERE — three commitments:

  1. SIGNED READ. `m_hat(s) = Sum_k sinh(u_k) A_k / Sum_k cosh(u_k)` is ODD:
     `m_hat(-s) = -m_hat(s)`. So a token and its contrast-inverse push the
     stream in OPPOSITE directions — the +A_k / -A_k half-axes are distinct,
     not summed away. The `mag` arm deliberately reads `|u|` (sign-collapsed)
     so `soft - mag` ISOLATES whether the sign carries task signal.

  2. ADDITIVE, not a convex mean. The read is `t + gate * consume(read)` — an
     unbounded signed vector ADDED to the token, not a `Sum a_k = 1`
     re-weighting of a filter mean. There is no mean for it to collapse into.

  3. SIGN-PRESERVING consume. `SignedSquare(x) = x*|x|` is ODD, so the sign of
     m_hat survives the consumer MLP to the readout (SquaredReLU would rectify
     the negative half-axis to zero — mutilating the antipode again).

The tokens come from a REAL convolution (rich k-neighbourhood features), so
the address finally reads an n-gram, not the 1x1 unigram that went constant on
grayscale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from amoe.core.address import AlephAddress

from .heads import super_fibonacci_s3
from .trunk import SquaredReLU, TrunkOutput

READ_MODES = ("soft", "mag", "none", "off")


def signed_antipode_read(slots: torch.Tensor, codebook: torch.Tensor,
                         tau: float, mode: str) -> torch.Tensor:
    """The one antipode read, shared by every antipode module. `slots` is
    (..., D); returns (..., D), UNIT-NORM. soft = signed m_hat (ODD:
    read(-s) = -read(s)); mag = the same read on |u| (EVEN — sign discarded);
    none = passthrough. Sphere-normalized so soft and mag are scale-matched
    (their only difference is direction = the sign)."""
    if mode == "none":
        return F.normalize(slots, dim=-1)                  # codebook unread
    A = F.normalize(codebook, dim=-1)
    u = (F.normalize(slots, dim=-1) @ A.transpose(-1, -2)) / tau
    if mode == "mag":
        u = u.abs()                                        # sign-collapsed
    m = u.abs().amax(dim=-1, keepdim=True)
    ep, en = torch.exp(u - m), torch.exp(-u - m)
    read = ((ep - en) @ A) / (ep + en).sum(dim=-1, keepdim=True)
    return F.normalize(read, dim=-1)                       # scale-matched


def _groups(c: int) -> int:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class SignedSquare(nn.Module):
    """Odd square: x*|x|. Preserves the sign of the aleph read through the
    consumer (unlike SquaredReLU, which rectifies the negative half-axis to
    zero and would mutilate the antipode)."""

    def forward(self, x):
        return x * x.abs()


class ConvStem(nn.Module):
    """A real convolution producing per-position feature tokens (B, T, d).
    Each token summarizes a growing k-neighbourhood — the n-gram the address
    needs. `downsample` halves spatially per layer (classifier); off keeps
    full resolution (generative per-pixel head)."""

    def __init__(self, channels: int, height: int, width: int, d: int,
                 n_layers: int = 2, kernel: int = 3, downsample: bool = True):
        super().__init__()
        self.C, self.H, self.W, self.d = channels, height, width, d
        stride = 2 if downsample else 1
        pad = kernel // 2
        layers, c_in, h, w = [], channels, height, width
        for _ in range(n_layers):
            layers += [nn.Conv2d(c_in, d, kernel, stride=stride, padding=pad),
                       nn.GroupNorm(_groups(d), d), SquaredReLU()]
            c_in = d
            if downsample:
                h, w = h // 2, w // 2
        self.net = nn.Sequential(*layers)
        self.hf, self.wf = h, w
        self.n_tokens = h * w

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        img = x_flat.view(x_flat.shape[0], self.C, self.H, self.W)
        h = self.net(img)                                  # (B, d, hf, wf)
        return h.permute(0, 2, 3, 1).reshape(h.shape[0], -1, self.d)  # (B,T,d)


class AntipodeRead(nn.Module):
    """Per-token signed aleph read, ADDED to the stream:
        t -> t + sigmoid(gate) * consume(read(proj(t)))
    `read` is the SIGNED m_hat (soft) — odd, antipode-preserving. Arms:
        soft  signed m_hat (the antipode used)
        mag   read on |u| (sign-collapsed) — the control that isolates the sign
        none  passthrough normalize(slots) — codebook present but UNREAD
        off   identity (no address at all)
    soft/mag/none are byte-identical in parameters (codebook present for all)."""

    def __init__(self, d: int, *, k_addr: int = 64, d_addr: int = 4,
                 n_slots: int = 16, tau: float = 0.1, hidden: int = 178,
                 gate_init: float = -1.5, mode: str = "soft",
                 codebook_init: str = "fibonacci"):
        super().__init__()
        if mode not in READ_MODES:
            raise ValueError(f"mode must be one of {READ_MODES}, got {mode!r}")
        self.mode, self.n_slots, self.d_addr, self.tau = (
            mode, n_slots, d_addr, tau)
        if mode != "off":
            self.proj = nn.Linear(d, n_slots * d_addr, bias=False)
            nn.init.orthogonal_(self.proj.weight)
            self.addr = AlephAddress(k_addr, d_addr, tau)
            if codebook_init == "fibonacci":
                with torch.no_grad():
                    self.addr.codebook.copy_(
                        super_fibonacci_s3(k_addr).to(self.addr.codebook))
                    self.addr.home.copy_(
                        F.normalize(self.addr.codebook.detach(), dim=-1))
            # STRICTLY-ODD consumer, so consume(-r) == -consume(r): every
            # piece is odd — Linear(bias=False), SignedSquare (x*|x|),
            # affine-free LayerNorm ((x-mean)/std, mean odd + std even), and a
            # final bias-free Linear. A bias anywhere would add a symmetric
            # offset that dilutes the sign; there is none. So the address's
            # contribution to the stream is ODD IN THE TOKEN — a token and its
            # antipode push the stream in exactly opposite directions. The
            # antipode is not merely preserved, it is the whole contribution.
            self.consume = nn.Sequential(
                nn.Linear(n_slots * d_addr, hidden, bias=False), SignedSquare(),
                nn.LayerNorm(hidden, elementwise_affine=False),
                nn.Linear(hidden, d, bias=False))
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def _read(self, slots: torch.Tensor) -> torch.Tensor:
        return signed_antipode_read(slots, self.addr.codebook, self.tau,
                                    self.mode)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return t
        B, T, _ = t.shape
        slots = self.proj(t).view(B, T, self.n_slots, self.d_addr)
        feats = self._read(slots).reshape(B, T, -1)
        return t + torch.sigmoid(self.gate) * self.consume(feats)

    @torch.no_grad()
    def read_amplitude(self, t: torch.Tensor) -> float:
        """||gate * consume(read)|| / ||t|| — how much the address actually
        contributes to the stream. The 'is it meaningful?' gauge (the old
        filter-steering bed sat near 0.7%). {} -> 0 for off."""
        if self.mode == "off":
            return 0.0
        B, T, _ = t.shape
        slots = self.proj(t).view(B, T, self.n_slots, self.d_addr)
        add = torch.sigmoid(self.gate) * self.consume(
            self._read(slots).reshape(B, T, -1))
        return float(add.norm() / t.norm().clamp_min(1e-9))


class ConvTokenConfig:
    def __init__(self, *, channels=1, height=28, width=28, n_classes=10,
                 mode="soft", d=64, k_addr=64, n_slots=16, tau=0.1,
                 conv_layers=2, kernel=3, read_layers=2, gate_init=-1.5,
                 codebook_init="fibonacci", objective="classify", n_bins=16,
                 seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]


class ConvTokenTrunk(nn.Module):
    """Conv stem -> per-token signed antipode reads -> readout. Drop-in for the
    harness (forward -> TrunkOutput). NO GAP (flatten / per-pixel head)."""

    def __init__(self, cfg: ConvTokenConfig):
        super().__init__()
        self.cfg = cfg
        self.C, self.H, self.W = cfg.channels, cfg.height, cfg.width
        self.pixels = cfg.channels * cfg.height * cfg.width
        self.objective = cfg.objective
        self.addr_mode = cfg.mode
        torch.manual_seed(cfg.seed)
        classify = cfg.objective == "classify"
        self.stem = ConvStem(cfg.channels, cfg.height, cfg.width, cfg.d,
                             n_layers=cfg.conv_layers, kernel=cfg.kernel,
                             downsample=classify)
        self.reads = nn.ModuleList([
            AntipodeRead(cfg.d, k_addr=cfg.k_addr, n_slots=cfg.n_slots,
                         tau=cfg.tau, gate_init=cfg.gate_init, mode=cfg.mode,
                         codebook_init=cfg.codebook_init)
            for _ in range(cfg.read_layers)])
        self.norm = nn.GroupNorm(1, cfg.d)     # over channels of a (B,T,d) view
        if classify:
            self.readout = nn.Linear(cfg.d * self.stem.n_tokens, cfg.n_classes)
            self.gen_head = None
        else:
            self.readout = None
            self.gen_head = nn.Conv2d(cfg.d, cfg.n_bins, 1)

    def _from_ids(self, ids):
        base = torch.arange(self.pixels, device=ids.device,
                            dtype=torch.float32)
        return torch.sin(0.01 * base.unsqueeze(0)
                         + ids.float().sum(-1, keepdim=True))

    def features(self, x):
        h = self.stem(x)                       # (B, T, d)
        for r in self.reads:
            h = r(h)
        return F.layer_norm(h, (h.shape[-1],))  # token-wise norm, no batch stat

    def forward(self, x=None, *, input_ids=None, labels=None, **_):
        if x is None:
            if input_ids is None:
                raise ValueError("ConvTokenTrunk needs x or input_ids")
            x = self._from_ids(input_ids)
        h = self.features(x)                   # (B, T, d)
        if self.objective == "classify":
            logits = self.readout(h.reshape(h.shape[0], -1))
            loss = None if labels is None else F.cross_entropy(logits, labels)
            return TrunkOutput(logits=logits, loss=loss)
        # generative: (B,T,d) -> (B,d,H,W) -> per-pixel categorical
        img = h.transpose(1, 2).reshape(h.shape[0], self.cfg.d,
                                        self.stem.hf, self.stem.wf)
        logits = self.gen_head(img)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return TrunkOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def read_report(self, x) -> dict:
        """The 'is the address meaningful?' gauge: the per-block read amplitude
        (contribution to the stream) and the antipode-oddness residual."""
        h = self.stem(x)
        amps = [r.read_amplitude(h) for r in self.reads]
        return {"read_amp_mean": sum(amps) / len(amps) if amps else 0.0,
                "read_amps": [round(a, 4) for a in amps]}

    def param_census(self) -> dict:
        return {"addr_mode": self.addr_mode,
                "params_total": sum(p.numel() for p in self.parameters())}


def build_conv_token_trunk(bed, cfg) -> ConvTokenTrunk:
    spec = bed.spec
    if spec is not None:
        channels, height, width, n_classes = (
            spec.channels, spec.height, spec.width, spec.classes)
    else:
        channels = bed.channels
        side = int(round((bed.pixels / channels) ** 0.5))
        if side * side * channels != bed.pixels:
            raise ValueError(
                f"conv_tokens needs a 2D shape; bed {bed.name!r} has "
                f"pixels={bed.pixels}, channels={channels}")
        height = width = side
        n_classes = bed.n_classes
    conv_cfg = ConvTokenConfig(
        channels=channels, height=height, width=width, n_classes=n_classes,
        mode=getattr(cfg, "mode", "soft"), d=getattr(cfg, "conv_channels", 64),
        k_addr=getattr(cfg, "k_addr", 64), n_slots=getattr(cfg, "n_slots", 16),
        tau=getattr(cfg, "tau", 0.1),
        conv_layers=getattr(cfg, "conv_layers", 2),
        read_layers=getattr(cfg, "read_layers", 2),
        codebook_init=getattr(cfg, "codebook_init", "fibonacci"),
        objective=getattr(cfg, "objective", "classify"),
        n_bins=getattr(cfg, "n_bins", 16), seed=getattr(cfg, "seed", 0))
    return ConvTokenTrunk(conv_cfg).to(bed.xtr.device)


# ══════════════════════════════════════════════════════════════════════════
# AntipodeConv2d — the ENTIRE convolution is the antipode read.
# ══════════════════════════════════════════════════════════════════════════
class AntipodeConv2d(nn.Module):
    """A convolution whose filter IS the signed antipode read. A standard conv
    is `unfold -> linear -> fold`; this is `unfold -> antipode-read -> fold`.

    At each position the k x k neighbourhood is projected to `n_slots` D-sphere
    slots (the `query` conv — this is the conv's locality + weight-sharing, the
    only thing kept from convolution), read by the SIGNED m_hat (the sole
    operation and the sole nonlinearity — no ReLU), and projected to the output
    channels (the 1x1 `out` conv). There is NO plain-conv filter doing separate
    work: every output value is a signed antipode read of a local neighbourhood.

    Arms: soft (signed antipode) / mag (|u|, sign-collapsed) / none (passthrough,
    codebook unread) / off (skip the read: `out(query(x))` = a factored LINEAR
    conv — the plain-conv control the antipode must beat)."""

    def __init__(self, c_in: int, c_out: int, *, kernel: int = 3, stride: int = 1,
                 k_addr: int = 64, d_addr: int = 4, n_slots: int = 16,
                 tau: float = 0.1, mode: str = "soft",
                 codebook_init: str = "fibonacci"):
        super().__init__()
        if mode not in READ_MODES:
            raise ValueError(f"mode must be one of {READ_MODES}, got {mode!r}")
        self.mode, self.n_slots, self.d_addr, self.tau = (
            mode, n_slots, d_addr, tau)
        # the neighbourhood -> slots map: the conv's locality + weight sharing
        self.query = nn.Conv2d(c_in, n_slots * d_addr, kernel, stride=stride,
                               padding=kernel // 2, bias=False)
        nn.init.orthogonal_(self.query.weight.view(n_slots * d_addr, -1))
        self.out = nn.Conv2d(n_slots * d_addr, c_out, 1, bias=False)
        if mode not in ("off",):
            self.addr = AlephAddress(k_addr, d_addr, tau)
            if codebook_init == "fibonacci":
                with torch.no_grad():
                    self.addr.codebook.copy_(
                        super_fibonacci_s3(k_addr).to(self.addr.codebook))
                    self.addr.home.copy_(
                        F.normalize(self.addr.codebook.detach(), dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.query(x)                                  # (B, n_slots*D, H, W)
        B, _, H, W = s.shape
        if self.mode == "off":                             # factored linear conv
            return self.out(s)
        slots = s.view(B, self.n_slots, self.d_addr, H, W)
        # D to the last axis for the read, then back
        slots = slots.permute(0, 1, 3, 4, 2)               # (B,n_slots,H,W,D)
        read = signed_antipode_read(slots, self.addr.codebook, self.tau,
                                    self.mode)
        read = read.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        return self.out(read)


class AntipodeConvBlock(nn.Module):
    """Residual antipode conv: `x + out(m_hat(query(GN x)))`, downsample via the
    query stride. The affine-free GroupNorm is standardization only — the
    antipode read is the block's sole nonlinear OPERATION (no ReLU anywhere)."""

    def __init__(self, c_in: int, c_out: int, *, stride: int, mode: str,
                 **kw):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(c_in), c_in, affine=False)
        self.conv = AntipodeConv2d(c_in, c_out, stride=stride, mode=mode, **kw)
        self.proj = (nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)
                     if (c_in != c_out or stride != 1) else nn.Identity())

    def forward(self, x):
        return self.proj(x) + self.conv(self.norm(x))


class AntipodeConvTrunk(nn.Module):
    """A CNN whose EVERY convolution is an antipode read — no plain-conv filter,
    no ReLU. The antipode is the whole computation. Drop-in for the harness."""

    def __init__(self, cfg: ConvTokenConfig):
        super().__init__()
        self.cfg = cfg
        self.C, self.H, self.W = cfg.channels, cfg.height, cfg.width
        self.pixels = cfg.channels * cfg.height * cfg.width
        self.objective, self.addr_mode = cfg.objective, cfg.mode
        torch.manual_seed(cfg.seed)
        classify = cfg.objective == "classify"
        chs = [cfg.channels] + [cfg.d] * cfg.conv_layers
        self.blocks = nn.ModuleList([
            AntipodeConvBlock(chs[i], chs[i + 1],
                              stride=2 if classify else 1, mode=cfg.mode,
                              k_addr=cfg.k_addr, n_slots=cfg.n_slots,
                              tau=cfg.tau, codebook_init=cfg.codebook_init)
            for i in range(cfg.conv_layers)])
        self.norm = nn.GroupNorm(_groups(cfg.d), cfg.d, affine=False)
        if classify:
            hf = cfg.height // (2 ** cfg.conv_layers)
            wf = cfg.width // (2 ** cfg.conv_layers)
            self.readout = nn.Linear(cfg.d * hf * wf, cfg.n_classes)
            self.gen_head = None
            self.hf, self.wf = hf, wf
        else:
            self.readout = None
            self.gen_head = nn.Conv2d(cfg.d, cfg.n_bins, 1)
            self.hf, self.wf = cfg.height, cfg.width

    def _from_ids(self, ids):
        base = torch.arange(self.pixels, device=ids.device,
                            dtype=torch.float32)
        return torch.sin(0.01 * base.unsqueeze(0)
                         + ids.float().sum(-1, keepdim=True))

    def forward(self, x=None, *, input_ids=None, labels=None, **_):
        if x is None:
            if input_ids is None:
                raise ValueError("AntipodeConvTrunk needs x or input_ids")
            x = self._from_ids(input_ids)
        h = x.view(x.shape[0], self.C, self.H, self.W)
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        if self.objective == "classify":
            logits = self.readout(h.reshape(h.shape[0], -1))
        else:
            logits = self.gen_head(h)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return TrunkOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def read_report(self, x) -> dict:
        """Antipode contribution amplitude at block 0: how much the READ shapes
        the output vs the bare query->out LINEAR path (`m_hat` removed),
        ||conv(xn) - out(query(xn))|| / ||conv(xn)||. 0 for off."""
        if self.addr_mode == "off":
            return {"read_amp_mean": 0.0}
        b = self.blocks[0]
        xn = b.norm(x.view(x.shape[0], self.C, self.H, self.W))
        y = b.conv(xn)                                     # antipode read path
        lin = b.conv.out(b.conv.query(xn))                 # the read removed
        return {"read_amp_mean": float(
            (y - lin).norm() / y.norm().clamp_min(1e-9))}

    def param_census(self) -> dict:
        return {"addr_mode": self.addr_mode,
                "params_total": sum(p.numel() for p in self.parameters())}


def build_antipode_conv_trunk(bed, cfg) -> AntipodeConvTrunk:
    spec = bed.spec
    if spec is not None:
        channels, height, width, n_classes = (
            spec.channels, spec.height, spec.width, spec.classes)
    else:
        channels = bed.channels
        side = int(round((bed.pixels / channels) ** 0.5))
        if side * side * channels != bed.pixels:
            raise ValueError(f"antipode_conv needs a 2D shape; bed "
                             f"{bed.name!r} pixels={bed.pixels}")
        height = width = side
        n_classes = bed.n_classes
    conv_cfg = ConvTokenConfig(
        channels=channels, height=height, width=width, n_classes=n_classes,
        mode=getattr(cfg, "mode", "soft"), d=getattr(cfg, "conv_channels", 64),
        k_addr=getattr(cfg, "k_addr", 64), n_slots=getattr(cfg, "n_slots", 16),
        tau=getattr(cfg, "tau", 0.1),
        conv_layers=getattr(cfg, "conv_layers", 2),
        codebook_init=getattr(cfg, "codebook_init", "fibonacci"),
        objective=getattr(cfg, "objective", "classify"),
        n_bins=getattr(cfg, "n_bins", 16), seed=getattr(cfg, "seed", 0))
    return AntipodeConvTrunk(conv_cfg).to(bed.xtr.device)
