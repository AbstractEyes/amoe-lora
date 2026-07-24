"""TinyTrunk — a 4-block classifier, shaped so amoe can attach to it
unmodified. Three input modes, and the mode is the whole story of what the
model IS:

  linear   `nn.Linear(pixels, d)` — tokens=1, the WHOLE image is one token.
           A plain 4-block MLP classifier.
  trigram  `byte_emb x3` per pixel — T = pixels (or H*W for RGB). Restores
           the n-gram structure the address needs (discovery #16), but the
           blocks are per-token MLPs, so it does NO cross-token mixing.
  patch    a 2D-patch ViT whose token mixer is the ALEPH ROUTER
           (AlephRoutedBlock), not softmax. A token is one C x P x P region;
           the router mixes tokens. THE ARCHITECTURE THE BED REQUIRES.

Why patch mode exists: linear and trigram are both provably GENERALIZED
ADDITIVE models. Their blocks and the RelayPatchwork adapter are per-token,
so with T tokens no two pixels outside a single token ever interact
nonlinearly — all spatial integration is dumped on the readout, and an
adapter bolted on reads a stream that was never mixed. That is the opposite
of how RelayPatchwork rides a real host (Qwen-VL / GPT-2 decoder layers,
whose self-attention has already mixed the stream). Patch mode gives the
adapter that host — but the mixer is the geometric aleph router, because the
house law says geometry ERODES through standard (softmax) attention. So
"no attention" in the older linear/trigram sense means no softmax mixing;
patch mode adds mixing that is itself aleph geometry, non-eroding.

What makes any of these a valid amoe substrate rather than a toy:

1. RESIDUAL STREAM. Every block updates a (B, T, d) pre-norm residual stream
   — the same object a decoder block hands the adapter. No convolution in
   any mode (the patch stem is reshape->Linear, not Conv2d).
2. A CONFIG SHIM. `binding.PathBinding` reads `model.config.hidden_size`
   and `attach(strict=...)` reads `_name_or_path`, so the trunk carries an
   HF-shaped config object.
3. AN `input_ids` PROBE PATH. `runtime.attach._probe` fingerprints a model
   by calling `model(input_ids=...)` with an int tensor — it is written for
   causal LMs. Rather than fork attach(), the trunk accepts `input_ids` and
   maps it DETERMINISTICALLY to a float input, which makes the bit-exact
   detach guarantee testable here for real.

NO GLOBAL AVERAGE POOLING anywhere (house law: GAP collapsed a geometric
encoder 70% -> 29%, replicated). Linear/trigram readouts FLATTEN; patch mode
reads the CLS token — a learned aggregator, neither GAP nor a giant flatten,
and the mixing already lives in the router.

Construct trunks with `model.build.build_model(bed, cfg)` — it derives every
shape from the dataset and validates parity. `build_trunk` below is the
low-level constructor it delegates to; calling it directly is how shape
mismatches used to sneak in.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routed_attention import AlephRoutedAttention, RoutedAttnConfig
from .stem import TrigramStem


@dataclass
class TinyConfig:
    """HF-shaped enough for amoe's resolver, binding and strict check."""
    hidden_size: int = 64
    model_type: str = "tiny_mnist"
    _name_or_path: str = "tiny-mnist-4block"   # per-dataset; set by build_model
    n_blocks: int = 4
    tokens: int = 1
    n_classes: int = 10
    pixels: int = 784
    channels: int = 1               # 3 for RGB == the trigram order
    input_mode: str = "linear"      # "linear" | "trigram" | "patch"
    n_bins: int = 256               # byte quantization for the trigram embed
    readout_dim: int = 16           # per-token bottleneck before the flatten
    trigram_lo: float = -3.0        # quantization window, PER DATASET
    trigram_hi: float = 3.0
    # patch mode (input_mode="patch"): a 2D-patch ViT whose MIXER is the
    # aleph-routed attention, not softmax. height/width let the flat input be
    # un-raveled back to (C, H, W) before patchifying.
    height: int = 28
    width: int = 28
    patch_size: int = 4
    num_heads: int = 4


@dataclass
class TrunkOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class LinearBlock(nn.Module):
    """Pre-norm residual MLP block — the miniature of a decoder block with
    the attention removed. SquaredReLU matches the adapter's own
    nonlinearity so the trunk and the patch head speak one dialect."""

    def __init__(self, d: int, mult: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.act = SquaredReLU()
        self.fc2 = nn.Linear(d * mult, d)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class PatchStem2D(nn.Module):
    """The ViT patch embed, done as reshape -> Linear (NOT Conv2d — the bed's
    house law is "no convolution", and a non-overlapping patch conv is a
    linear map anyway). A flat channel-major input vector is un-raveled to
    (B, C, H, W), cut into non-overlapping P x P patches in raster order, and
    each C*P*P patch is projected to d. This is the thing the single-pixel
    and whole-image stems were not: a token that is a genuine 2D region, so
    the aleph router has real spatial structure to bind."""

    def __init__(self, channels: int, height: int, width: int,
                 patch_size: int, d: int):
        super().__init__()
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"patch_size={patch_size} must divide H={height} and "
                f"W={width}")
        self.C, self.H, self.W, self.P = channels, height, width, patch_size
        self.nh, self.nw = height // patch_size, width // patch_size
        self.n_tokens = self.nh * self.nw
        self.proj = nn.Linear(channels * patch_size * patch_size, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.view(B, self.C, self.nh, self.P, self.nw, self.P)
        # -> (B, nh, nw, C, P, P) -> (B, nh*nw, C*P*P), raster patch order
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, self.n_tokens, -1)
        return self.proj(x)


class AlephRoutedBlock(nn.Module):
    """A real pre-norm transformer block whose token mixer is the aleph
    router instead of softmax attention: `x + attn(LN x)` then
    `x + mlp(LN x)`. The MLP is the same SquaredReLU dialect as LinearBlock
    so the trunk and the RelayPatchwork adapter still speak one language.

    This is the amoe attach SITE: `build_heads` wraps it in
    `BlockWithAdapter`, exactly as the adapter wraps a Qwen/GPT-2 decoder
    layer — except here the host mixing is geometric, so the adapter's
    geometry rides a non-eroding stream."""

    def __init__(self, d: int, num_heads: int, k: int = 64, d_addr: int = 4,
                 tau: float = 0.1, mult: int = 2, seed: int = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = AlephRoutedAttention(RoutedAttnConfig(
            dim=d, num_heads=num_heads, K=k, D_addr=d_addr, tau=tau,
            causal=False, seed=seed))
        self.norm2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.act = SquaredReLU()
        self.fc2 = nn.Linear(d * mult, d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class TinyTrunk(nn.Module):
    """stem -> blocks (the attach sites) -> norm -> readout."""

    def __init__(self, config: TinyConfig | None = None):
        super().__init__()
        cfg = config or TinyConfig()
        self.config = cfg
        d = cfg.hidden_size
        self.mode = cfg.input_mode
        self.trigram = cfg.input_mode == "trigram"
        self.patched = cfg.input_mode == "patch"
        # CLS / pos are patch-only; declared for every mode so the dial and
        # the forward can reference them unconditionally.
        self.cls_token = None
        self.pos_embed = None
        if self.patched:
            # 2D-patch ViT whose mixer is the aleph router. A token is a
            # genuine C x P x P region, and AlephRoutedBlock mixes tokens —
            # so the model is no longer additive and the adapter reads a
            # (geometrically) mixed stream.
            self.stem = PatchStem2D(cfg.channels, cfg.height, cfg.width,
                                    cfg.patch_size, d)
            N = self.stem.n_tokens
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, d))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            T = N + 1                              # + CLS; the adapter sees T
            self.blocks = nn.ModuleList([
                AlephRoutedBlock(d, cfg.num_heads, k=64, d_addr=4,
                                 tau=0.1, seed=17 * i)
                for i in range(cfg.n_blocks)])
        elif self.trigram:
            # byte_emb x3 stem: T is set by the trigram form, not cfg.tokens.
            # The blocks are per-token MLPs (no cross-token mixing), so spatial
            # aggregation lives entirely in the flatten readout — the aleph
            # read still operates per pixel-trigram, exactly as it does per
            # token in the AlephLM.
            self.stem = TrigramStem(d, cfg.channels, cfg.pixels, cfg.n_bins,
                                    cfg.trigram_lo, cfg.trigram_hi)
            T = self.stem.n_tokens
            self.blocks = nn.ModuleList([LinearBlock(d)
                                         for _ in range(cfg.n_blocks)])
        else:
            T = cfg.tokens
            if cfg.pixels % T:
                raise ValueError(f"tokens={T} must divide pixels={cfg.pixels}")
            self.patch = cfg.pixels // T
            self.stem = nn.Linear(self.patch, d)
            self.blocks = nn.ModuleList([LinearBlock(d)
                                         for _ in range(cfg.n_blocks)])
        self.n_tokens = T
        self.norm = nn.LayerNorm(d)
        # Readout. PATCH mode reads the CLS token only (Linear(d, C)) — not
        # GAP (house law: GAP collapsed a geometric encoder 70->29), not the
        # multi-GB flatten either; the mixing already lives in the routed
        # attention, so CLS is a real learned aggregator. TRIGRAM flattens a
        # per-token bottleneck d->r (r*T stays small; positional info
        # survives). LINEAR (T=1) flattens d*T=d directly.
        if self.patched:
            self.readout_dim = None
            self.readout_proj = None
            self.readout = nn.Linear(d, cfg.n_classes)
        elif self.trigram:
            self.readout_dim = min(cfg.readout_dim, d)
            self.readout_proj = nn.Linear(d, self.readout_dim)
            self.readout = nn.Linear(self.readout_dim * T, cfg.n_classes)
        else:
            self.readout_dim = None
            self.readout_proj = None
            self.readout = nn.Linear(d * T, cfg.n_classes)

    # -- the probe shim ------------------------------------------------
    def _from_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Deterministic int -> float image map, so amoe's LM-shaped
        fingerprint probe works on a vision trunk. Pure function of the ids:
        same ids, same activations, bit for bit."""
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
        if self.patched:
            h = self.stem(x)                       # (B, N, d) patch tokens
            cls = self.cls_token.expand(h.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1) + self.pos_embed   # (B, N+1, d)
        elif self.trigram:
            h = self.stem(x)                       # (B, T, d) byte_emb x3
        else:
            x = x.reshape(x.shape[0], self.config.tokens, self.patch)
            h = self.stem(x)
        for blk in self.blocks:
            out = blk(h)
            h = out[0] if isinstance(out, tuple) else out
        h = self.norm(h)
        if self.patched:
            logits = self.readout(h[:, 0])         # CLS token, no flatten/GAP
        else:
            if self.readout_proj is not None:
                h = self.readout_proj(h)       # (B,T,d) -> (B,T,r) bottleneck
            logits = self.readout(h.reshape(h.shape[0], -1))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return TrunkOutput(logits=logits, loss=loss)

    # -- the dial ------------------------------------------------------
    def set_trainable_blocks(self, n: int) -> dict:
        """THE DIAL. Unfreeze the LAST `n` blocks (0 = fully frozen
        substrate, n_blocks = full co-training). stem/norm/readout follow the
        trunk: they are trunk, not adapter.

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
            mods = [self.stem, self.norm, self.readout]
            if self.readout_proj is not None:
                mods.append(self.readout_proj)
            for m in mods:
                for p in m.parameters():
                    p.requires_grad_(True)
            for p in (self.cls_token, self.pos_embed):   # patch-mode trunk
                if p is not None:
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
                seed: int = 0, pixels: int = 784, n_classes: int = 10,
                channels: int = 1, input_mode: str = "linear",
                name_or_path: str = "tiny-mnist-4block",
                trigram_lo: float = -3.0,
                trigram_hi: float = 3.0) -> TinyTrunk:
    """Low-level constructor. Prefer `build_model(bed, cfg)`, which derives
    and validates these fields from the dataset instead of trusting the
    caller to pass them consistently."""
    torch.manual_seed(seed)
    return TinyTrunk(TinyConfig(
        hidden_size=d, n_blocks=n_blocks, tokens=tokens, pixels=pixels,
        n_classes=n_classes, channels=channels, input_mode=input_mode,
        _name_or_path=name_or_path,
        trigram_lo=trigram_lo, trigram_hi=trigram_hi))
