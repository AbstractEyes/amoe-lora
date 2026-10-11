"""The lookup arm (0.2.12): a zero-born causal softmax attention organ attached after chosen blocks of a trunk.

WHAT IT IS. After a chosen block the stream receives one more causal attention module behind its own LayerNorm:

    x = block_i(x)
    x = x + attn_i(norm_i(x))                      # one layer; `depth` layers run in series, each behind its own norm

The attention is the ordinary kind (fused q/k/v projection, scaled dot product with a causal mask, an output projection)
with two guards: a per-head RMS normalization of q and k with learned per-head per-channel gains (born at one), and the
whole attention block in fp32 when the trunk runs under autocast on a card. The output projection is born at ZERO, so a
fresh arm is exactly inert: the trunk's logits at step 0 are its own, bit for bit, and the arm's parameters are a key
subset that ships, loads and detaches on its own. Born null by weight, never by a gate (a gate at zero has no gradient
through it; a zero weight does).

WHY. A trunk whose memory is address-based or recurrent fetches a value planted behind text only within a few hundred
bytes of its cue; softmax attention fetches it to the end of the context. The lookup arm gives such a trunk content-
addressed lookup without a retrain: the trunk stays frozen, pure Adam owns the arm's keys (amoe.train.ownership), and
the arm trains on recall rows mixed into ordinary text (the public AlephLLM recipe: docs/arms/lookup_arm.md).

DEPTH. `LookupSpec.depth` k puts k layers at every site in series, each behind its own LayerNorm, all zero-born. A
state saved at a shallower depth loads into a deeper arm (`LookupArm.load_grown`): the layers the state lacks stay
zero-born, so the deeper arm IS the shallower arm at birth and a continuation adds structure without a restart.

DECODE. The wrapper's prefill runs the full causal pass and keeps the arm's own key/value cache; step appends the new
position's keys and values and attends over the cache. One decode stream per handle at a time; prefill resets the
cache, reset_state() clears it.

THE ALEPHLLM KEYS. geolip.alephllm's AlephLM carries this arm natively under `lookup_arms.<site>[_l<k>].*` and
`lookup_norms.<site>[_l<k>].*` (the trained Beatrix arms on the Hugging Face hub ship in that layout).
`from_alephllm_state` / `to_alephllm_state` map that layout to this module's and back, so a hub arm attaches to the
trunk through this package (bit-identical to the trunk's own path; tests/test_lookup_arm.py asserts it) and an arm
trained through this package ships into the trunk's own keys.

LAWS. Pure Adam on the arm keys only (laws 1); no selectors (the attention is a read, not a router; laws 2); zero-born
output (laws 3); the toggle law (all arms off = the bare model bit-exact) and detach(verify=True) hold as for every
handle (laws 4).
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..binding.resolver import resolve
from ..runtime.attach import AttachHandle, _probe

FORMAT = "amoe.lookup"
VERSION = 1


@dataclass
class LookupSpec:
    heads: int = 16
    depth: int = 1
    qk_norm: str = "rms"        # "" (plain q and k) or "rms" (per-head RMS norm with learned gains)
    attn_fp32: bool = True      # under autocast on a card the whole attention block runs in fp32
    norm_eps: float = 1e-5      # the LayerNorm in front of every layer
    rms_eps: float = 1e-6       # the q/k RMS normalization

    def __post_init__(self):
        if self.qk_norm not in ("", "rms"):
            raise ValueError(f"qk_norm: '' or 'rms', got {self.qk_norm!r}")
        if int(self.depth) < 1:
            raise ValueError(f"depth >= 1, got {self.depth}")
        if int(self.heads) < 1:
            raise ValueError(f"heads >= 1, got {self.heads}")


def _rms_normalize(t: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS-normalize the last dim; the statistic in fp32, the dtype kept."""
    inv = torch.rsqrt(t.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return t * inv.to(t.dtype)


class LookupAttention(nn.Module):
    """One attention layer: fused q/k/v (no bias), the q/k guard, causal scaled dot product, the output projection
    born at zero (no bias). prefill / step carry a key/value cache (the guarded k)."""

    def __init__(self, d: int, spec: LookupSpec):
        super().__init__()
        if d % int(spec.heads):
            raise ValueError(f"d={d} is not a multiple of heads={spec.heads}")
        self.spec = spec
        self.h = int(spec.heads)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.orthogonal_(self.qkv.weight)
        nn.init.zeros_(self.o.weight)                   # born null by weight
        if spec.qk_norm:
            hd = d // self.h
            self.q_gain = nn.Parameter(torch.ones(self.h, 1, hd))
            self.k_gain = nn.Parameter(torch.ones(self.h, 1, hd))

    def extra_repr(self) -> str:
        return f"heads={self.h}, qk_norm={self.spec.qk_norm!r}, attn_fp32={self.spec.attn_fp32}"

    def _split(self, x, n):
        B, _, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return tuple(t.view(B, n, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))

    def _guard(self, q, k):
        if not self.spec.qk_norm:
            return q, k
        eps = float(self.spec.rms_eps)
        return (_rms_normalize(q, eps) * self.q_gain.to(q.dtype),
                _rms_normalize(k, eps) * self.k_gain.to(k.dtype))

    def _fp32_active(self, x) -> bool:
        return bool(self.spec.attn_fp32) and x.is_cuda and torch.is_autocast_enabled()

    def _run(self, x, n):
        B, d = x.shape[0], x.shape[-1]
        q, k, v = self._split(x, n)
        q, k = self._guard(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, n, d))

    def forward(self, x):
        n = x.shape[1]
        if self._fp32_active(x):
            with torch.autocast("cuda", enabled=False):
                return self._run(x.float(), n)
        return self._run(x, n)

    def prefill(self, x):
        """The full causal pass that also returns the decode cache."""
        B, n, d = x.shape
        q, k, v = self._split(x, n)
        q, k = self._guard(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, n, d)), {"k": k, "v": v}

    def step(self, x_t, cache):
        """One new position attending over everything cached."""
        B, _, d = x_t.shape
        q, k, v = self._split(x_t, 1)
        q, k = self._guard(q, k)
        cache["k"] = torch.cat([cache["k"], k], dim=2)
        cache["v"] = torch.cat([cache["v"], v], dim=2)
        y = F.scaled_dot_product_attention(q, cache["k"], cache["v"])
        return self.o(y.transpose(1, 2).reshape(B, 1, d))


class LookupArm(nn.Module):
    """`depth` attention layers in series, each behind its own LayerNorm: x <- x + layer_k(norm_k(x))."""

    def __init__(self, d: int, spec: LookupSpec | None = None):
        super().__init__()
        s = spec or LookupSpec()
        self.spec = s
        self.d = int(d)
        self.norms = nn.ModuleList(nn.LayerNorm(d, eps=float(s.norm_eps)) for _ in range(int(s.depth)))
        self.layers = nn.ModuleList(LookupAttention(d, s) for _ in range(int(s.depth)))

    @property
    def depth(self) -> int:
        return len(self.layers)

    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        return x

    def prefill(self, x):
        caches = []
        for norm, layer in zip(self.norms, self.layers):
            y, c = layer.prefill(norm(x))
            x = x + y
            caches.append(c)
        return x, caches

    def step(self, x_t, caches):
        for norm, layer, c in zip(self.norms, self.layers, caches):
            x_t = x_t + layer.step(norm(x_t), c)
        return x_t

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_grown(self, state: dict) -> int:
        """Load a state of the same or a SHALLOWER depth. The layers the state lacks keep their zero-born output, so
        the arm equals the shallower arm at birth. Returns the depth the state carried; raises on unexpected keys or
        on a state missing keys of a layer it does carry."""
        res = self.load_state_dict(state, strict=False)
        if res.unexpected_keys:
            raise KeyError(f"unexpected keys for a LookupArm of depth {self.depth}: {res.unexpected_keys[:4]}")
        carried = {int(k.split(".")[1]) for k in state if k.startswith(("layers.", "norms."))}
        j = (max(carried) + 1) if carried else 0
        bad = [k for k in res.missing_keys if int(k.split(".")[1]) < j]
        if bad:
            raise KeyError(f"the state carries layers 0..{j - 1} but lacks {bad[:4]}")
        return j


class BlockWithLookup(nn.Module):
    """Wraps one decoder block; the arm reads the block's output and writes to the stream (tensor or tuple outputs).
    `enabled` False returns the block output untouched (the toggle law). prefill / step keep the arm's own cache."""

    def __init__(self, block: nn.Module, arm: LookupArm):
        super().__init__()
        self.block = block
        self.arm = arm
        self.enabled = True
        self._caches = None

    def reset_state(self):
        self._caches = None

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if not self.enabled:
            return out
        if isinstance(out, tuple):
            return (self.arm(out[0]),) + out[1:]
        return self.arm(out)

    def prefill(self, *args, **kwargs):
        out, cache = self.block.prefill(*args, **kwargs)
        if not self.enabled:
            self._caches = None
            return out, cache
        y, self._caches = self.arm.prefill(out)
        return y, cache

    def step(self, *args, **kwargs):
        out = self.block.step(*args, **kwargs)
        if not self.enabled:
            return out
        if self._caches is None:
            raise RuntimeError("LookupArm step before prefill: the arm keeps its own key/value cache; run prefill first")
        x = out[0] if isinstance(out, tuple) else out
        y = self.arm.step(x, self._caches)
        return (y,) + out[1:] if isinstance(out, tuple) else y


class LookupHandle(AttachHandle):
    """AttachHandle (mask / only / all_off / detach verify) over BlockWithLookup wrappers, plus the arm verbs."""

    def __init__(self, model, binding, original_layers, names, wraps, fingerprint, sites):
        super().__init__(model, binding, original_layers, names, wraps, fingerprint)
        self.sites = [int(s) for s in sites]

    @property
    def spec(self) -> LookupSpec:
        return self._blocks[0].arm.spec

    def arms(self) -> list[LookupArm]:
        return [w.arm for w in self._blocks]

    def wraps(self) -> list[BlockWithLookup]:
        return list(self._blocks)

    def parameters(self):
        return [p for w in self._blocks for p in w.arm.parameters()]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset_state(self):
        for w in self._blocks:
            w.reset_state()

    def state_dict(self) -> dict:
        """'{site}.{param}' -> tensor (cpu clones)."""
        return {f"{s}.{k}": v.detach().cpu().clone()
                for s, w in zip(self.sites, self._blocks) for k, v in w.arm.state_dict().items()}

    def alephllm_state(self) -> dict:
        """The arm in geolip.alephllm's own key layout (lookup_arms.* / lookup_norms.*), ready for the trunk's loader."""
        return to_alephllm_state(self.state_dict())

    def save(self, path: str, meta: dict | None = None) -> str:
        meta = dict(meta or {})
        meta.setdefault("created", _dt.datetime.now(_dt.timezone.utc).isoformat())
        meta.setdefault("name", self.names[0])
        meta["spec"] = asdict(self.spec)
        meta["sites"] = list(self.sites)
        meta["arm_type"] = "LookupArm"
        meta["params"] = self.n_params()
        torch.save({"format": FORMAT, "version": VERSION, "meta": meta, "adapters": self.state_dict()}, path)
        return path


def _per_site_state(adapters: dict, site: int) -> dict:
    pre = f"{site}."
    return {k[len(pre):]: v for k, v in adapters.items() if k.startswith(pre)}


def attach_lookup(model, sites, spec: LookupSpec | None = None, *, name: str = "lookup", binding=None,
                  state: dict | None = None, seed: int | None = None) -> LookupHandle:
    """Wrap the bound decoder layers at `sites` with a LookupArm each: fresh (torch seed `seed` if given) or loaded
    from `state` ('{site}.{param}' keys; a shallower state grows into the spec's depth). The arm follows its block's
    device. The pre-attach probe fingerprint is taken first so detach(verify=True) can assert bit-exactness."""
    b = resolve(model, binding)
    layers = list(b.layers(model))
    d = b.hidden_size(model)
    sites = sorted({int(s) for s in sites})
    if not sites or not all(0 <= s < len(layers) for s in sites):
        raise ValueError(f"sites must name layers in 0..{len(layers) - 1}, got {sites}")
    fingerprint = _probe(model, b)
    if seed is not None:
        torch.manual_seed(int(seed))
    spec = spec or LookupSpec()
    blocks, wraps = [], []
    for i, layer in enumerate(layers):
        if i not in sites:
            blocks.append(layer)
            continue
        arm = LookupArm(d, spec)
        if state is not None:
            st = _per_site_state(state, i)
            if not st:
                raise KeyError(f"the state carries no arm for site {i}")
            arm.load_grown(st)
        arm.to(next(layer.parameters()).device)
        w = BlockWithLookup(layer, arm)
        blocks.append(w)
        wraps.append(w)
    b.set_layers(model, blocks)
    return LookupHandle(model, b, layers, [name], wraps, fingerprint, sites)


def load_lookup(path: str):
    """-> (adapters '{site}.{param}', meta, LookupSpec, sites)."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("format") != FORMAT:
        raise ValueError(f"not an {FORMAT} checkpoint: {path}")
    meta = blob.get("meta", {})
    return blob["adapters"], meta, LookupSpec(**meta["spec"]), [int(s) for s in meta["sites"]]


def attach_saved_lookup(model, path: str, *, name: str | None = None, binding=None,
                        depth: int | None = None) -> LookupHandle:
    """Attach a saved arm; `depth` deeper than the saved one grows it (the new layers zero-born)."""
    adapters, meta, spec, sites = load_lookup(path)
    if depth is not None and int(depth) != spec.depth:
        if int(depth) < spec.depth:
            raise ValueError(f"a saved depth {spec.depth} arm cannot shrink to {depth}")
        spec = replace(spec, depth=int(depth))
    return attach_lookup(model, sites, spec, name=name or meta.get("name", "lookup"), binding=binding, state=adapters)


# ------------------------------------------------------------------ the AlephLLM key layout
_ALEPH_RE = re.compile(r"^lookup_(arms|norms)\.(\d+)(?:_l(\d+))?\.(.+)$")


def from_alephllm_state(sd: dict):
    """geolip.alephllm's keys -> this module's: lookup_arms.<s>.<p> -> '<s>.layers.0.<p>', lookup_arms.<s>_l<k>.<p> ->
    '<s>.layers.<k-1>.<p>', lookup_norms likewise under 'norms'. Other keys are ignored. -> (adapters, sites, depth)."""
    out, sites, depth = {}, set(), 1
    for k, v in sd.items():
        m = _ALEPH_RE.match(k)
        if not m:
            continue
        kind, site, lk, rest = m.groups()
        j = int(lk) - 1 if lk else 0
        out[f"{site}.{'layers' if kind == 'arms' else 'norms'}.{j}.{rest}"] = v
        sites.add(int(site))
        depth = max(depth, j + 1)
    if not out:
        raise KeyError("no lookup arm keys (lookup_arms.* / lookup_norms.*) in the state")
    return out, sorted(sites), depth


def to_alephllm_state(adapters: dict) -> dict:
    """The reverse map: '<s>.layers.<j>.<p>' -> lookup_arms.<s>[_l<j+1>].<p>, '<s>.norms.<j>.<p>' -> lookup_norms.<s>[_l<j+1>].<p>."""
    out = {}
    for k, v in adapters.items():
        site, kind, j, rest = k.split(".", 3)
        j = int(j)
        suffix = "" if j == 0 else f"_l{j + 1}"
        out[f"lookup_{'arms' if kind == 'layers' else 'norms'}.{site}{suffix}.{rest}"] = v
    return out


def spec_from_alephllm_state(sd: dict, *, heads: int | None = None, attn_fp32: bool = True) -> LookupSpec:
    """The LookupSpec a geolip.alephllm state implies: depth from the keys, QK norm from the gains' presence, heads
    from the gains' shape (or `heads` when the state has no gains)."""
    adapters, sites, depth = from_alephllm_state(sd)
    gains = [v for k, v in adapters.items() if k.endswith(".q_gain")]
    if gains:
        return LookupSpec(heads=int(gains[0].shape[0]), depth=depth, qk_norm="rms", attn_fp32=attn_fp32)
    if heads is None:
        raise ValueError("the state has no q/k gains, so `heads` must be given")
    return LookupSpec(heads=int(heads), depth=depth, qk_norm="", attn_fp32=attn_fp32)


def attach_from_alephllm(model, sd: dict, *, name: str = "lookup", binding=None, heads: int | None = None,
                         depth: int | None = None, attn_fp32: bool = True) -> LookupHandle:
    """Attach a geolip.alephllm lookup arm (a state dict holding lookup_arms.* / lookup_norms.*, e.g. the hub's arm-only
    file) to a trunk through the binding; `depth` deeper than the state's grows the arm."""
    adapters, sites, d0 = from_alephllm_state(sd)
    spec = spec_from_alephllm_state(sd, heads=heads, attn_fp32=attn_fp32)
    if depth is not None and int(depth) != spec.depth:
        if int(depth) < spec.depth:
            raise ValueError(f"the state carries depth {spec.depth}; it cannot shrink to {depth}")
        spec = replace(spec, depth=int(depth))
    return attach_lookup(model, sites, spec, name=name, binding=binding, state=adapters)
