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
        """(B,T,n_slots,D) -> (B,T,n_slots,D), UNIT-NORM per slot. soft = signed
        m_hat direction (odd); mag = the same read on |u| (a token and its
        antipode give the SAME read — sign discarded); none = passthrough.

        The read is sphere-normalized so soft and mag are SCALE-MATCHED: mag's
        |u| collapses every atom into the +hemisphere, which shrinks its raw
        norm ~2.3x, so without this `soft - mag` would confound the SIGN with
        magnitude. Normalized, the arms differ only in DIRECTION — which is the
        sign. Normalization is odd, so soft stays odd and mag stays even."""
        if self.mode == "none":
            return F.normalize(slots, dim=-1)              # codebook unread
        A = F.normalize(self.addr.codebook, dim=-1)
        u = (F.normalize(slots, dim=-1) @ A.transpose(-1, -2)) / self.tau
        if self.mode == "mag":
            u = u.abs()                                    # sign-collapsed
        m = u.abs().amax(dim=-1, keepdim=True)
        ep, en = torch.exp(u - m), torch.exp(-u - m)
        read = ((ep - en) @ A) / (ep + en).sum(dim=-1, keepdim=True)
        return F.normalize(read, dim=-1)                   # scale-matched

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
