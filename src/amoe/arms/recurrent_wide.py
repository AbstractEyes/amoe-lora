"""RecurrentWide — a recurrent deduction arm: a small causal recurrent cell per site, fed by the hub read,
writing back through a zero-born head, with its state carried across cached decode steps. Certification record
(attach bit-inert, cached == full forward, recurrence live) ships with the published mini-beatrix-2s arms:
huggingface.co/AbstractPhil/mini-beatrix-2s, reports/arms.

Per site (one wrapped decoder block):
    a_t     = the HUB READ = the block's attention-sublayer output (CausalSplatHUB / CausalSDPA), the tensor the block's
              first residual adds: Block.forward -> x + attn(n1(x)); Block.prefill -> attn.prefill(n1(x))[0]; Block.step -> attn.step(...)
    r_t     = in_proj(a_t)                          d -> p
    h_t     = GRU(r_t, h_{t-1})                     p -> H   (recurrent=True; h_0 = 0 per row; the state is CARRIED per row
                                                              across cached decode steps — prefill writes it, every step updates it,
                                                              a fresh prefill starts it at 0 again: no resets inside a row)
    out_t   = block_out_t + head(h_t)               H -> d   (head weight AND bias zero at birth -> bit-inert: x + 0.0 == x)
Control (recurrent=False): the same in_proj and head around a POSITION-WISE projection cell (Linear p->C, tanh, Linear C->H)
with C sized to the GRU cell's parameter count, so recurrence is the only difference between the arm and its control.

Laws kept: no gate (born weight-zero, never gate-zero), no selector of any kind, no aleph address (this is not a patchwork);
the toggle law holds through `enabled` (disabled = the block output untouched, bit-exact); detach restores the retained layers
and verifies bit-exactness through amoe.runtime.attach.AttachHandle. The hub read is captured WITHOUT touching the block's
own code: a forward hook on the attention sublayer for the parallel path, and instance-level wrappers of its prefill/step
bound methods for the cached path (removed at detach; the class methods are never modified).

Parameter count per site at d = 1024, p = 128:  in_proj (d p + p) + GRU (3H(p + H) + 6H) + head (d H + d)
    = 3H^2 + (3p + 6 + d) H + (d p + p + d) = 3H^2 + 1414 H + 132,224
    H = 128 -> 362,368 (x20 = 7,247,360; 15.3% under WIDE-1024's 8,560,660)
    H = 160 -> 435,264 (x20 = 8,705,280; +1.69%)   <- the default (the plan's 'hidden ~160'); exact parity would be H = 157 (+0.03%)
Control cell at p = 128, H = 160: C = 481 -> 139,169 params vs the GRU cell's 139,200; site 435,233 (x20 = 8,704,660).

Checkpoint format 'amoe.recurrent_wide' v1: {"format", "version", "meta" (incl. "spec"), "adapters": {"{block}.{param}": tensor}}
(the anchor key layout; NOT loadable by amoe.io.checkpoint.load_anchor, which requires addr.home per block).
"""
from __future__ import annotations

import datetime as _dt
import types
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from amoe.binding.resolver import resolve
from amoe.runtime.attach import AttachHandle, _probe

FORMAT = "amoe.recurrent_wide"
VERSION = 1


@dataclass
class RecurrentSpec:
    proj: int = 128                 # p: the projection width fed to the cell
    hidden: int = 160               # H: the cell width (re-derived for parity with WIDE-1024, see the module docstring)
    recurrent: bool = True          # False -> the projection-only control (position-wise cell of the same count)
    control_width: int | None = None   # C for the control cell; None -> sized to the GRU cell's count
    zero_init_head: bool = True     # weight AND bias (born weight-zero, never gate-zero)


# ----------------------------------------------------------------------------------------------- counts (pure functions)
def gru_cell_params(p: int, H: int) -> int:
    """nn.GRU(p, H): weight_ih (3H, p) + weight_hh (3H, H) + bias_ih (3H) + bias_hh (3H)."""
    return 3 * H * (p + H) + 6 * H


def control_width_for(p: int, H: int, target: int | None = None) -> int:
    """C such that Linear(p, C) + tanh + Linear(C, H) = (p + 1) C + C H + H matches the GRU cell's count (or `target`)."""
    t = gru_cell_params(p, H) if target is None else target
    return max(1, int(round((t - H) / (p + 1 + H))))


def site_params(d: int, spec: RecurrentSpec) -> int:
    p, H = spec.proj, spec.hidden
    cell = gru_cell_params(p, H) if spec.recurrent else ((p + 1) * (spec.control_width or control_width_for(p, H)) + (spec.control_width or control_width_for(p, H)) * H + H)
    return (d * p + p) + cell + (d * H + d)


def derive_hidden(d: int, p: int, target_per_site: int) -> dict:
    """Solve 3H^2 + (3p + 6 + d) H + (d p + p + d) = target for H (the recurrent form); returns the real root, the two
    integer neighbours with their counts and relative errors, and the nearest integer."""
    import math
    a, b, c = 3.0, float(3 * p + 6 + d), float(d * p + p + d - target_per_site)
    root = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    lo, hi = int(math.floor(root)), int(math.ceil(root))
    def cnt(H):
        return site_params(d, RecurrentSpec(proj=p, hidden=H, recurrent=True))
    return {"root": round(root, 3), "target_per_site": target_per_site,
            "neighbours": {str(H): {"site": cnt(H), "rel": round(cnt(H) / target_per_site - 1, 5)} for H in (lo, hi)},
            "nearest": lo if abs(cnt(lo) - target_per_site) <= abs(cnt(hi) - target_per_site) else hi,
            "formula": "3H^2 + (3p + 6 + d)H + (dp + p + d) = target"}


# ----------------------------------------------------------------------------------------------- the arm
class RecurrentWide(nn.Module):
    def __init__(self, d: int, spec: RecurrentSpec | None = None):
        super().__init__()
        s = spec or RecurrentSpec()
        self.spec = s
        self.d, self.p, self.H = d, s.proj, s.hidden
        self.in_proj = nn.Linear(d, s.proj)
        if s.recurrent:
            self.cell = nn.GRU(s.proj, s.hidden, batch_first=True)
        else:
            C = s.control_width or control_width_for(s.proj, s.hidden)
            self.cell = nn.Sequential(nn.Linear(s.proj, C), nn.Tanh(), nn.Linear(C, s.hidden))
        self.head = nn.Linear(s.hidden, d)
        if s.zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @property
    def recurrent(self) -> bool:
        return bool(self.spec.recurrent)

    def run(self, a: torch.Tensor, h0: torch.Tensor | None = None):
        """a: (B, n, d) the hub read; h0: (B, H) carried state or None (= zeros). -> (delta (B, n, d), h_n (B, H) | None)."""
        r = self.in_proj(a)
        if self.recurrent:
            if h0 is not None and (h0.shape[0] != a.shape[0] or h0.device != a.device):
                h0 = None                                   # a fresh context (bucket size changed): start at 0
            out, hn = self.cell(r, None if h0 is None else h0.unsqueeze(0).contiguous())
            return self.head(out), hn.squeeze(0)
        return self.head(self.cell(r)), None

    def forward(self, a, h0=None):
        return self.run(a, h0)[0]


class BlockWithRecurrent(nn.Module):
    """Wraps one decoder block; captures the block's attention-sublayer output (the hub read) on all three paths and adds
    the arm's write to the block output. `enabled` False = the block output untouched (toggle law)."""

    def __init__(self, block: nn.Module, arm: RecurrentWide, reader: str = "attn"):
        super().__init__()
        self.block = block
        self.arm = arm
        self.enabled = True
        self.reader = reader
        self._state = None          # (B, H) after prefill / step (recurrent arms); None = fresh
        self._stash = {}
        self._hook = None
        self._patch()

    # -- the hub-read capture ------------------------------------------------------------------
    def _patch(self):
        sub = getattr(self.block, self.reader)
        stash = self._stash

        def hook(mod, inp, out):
            stash["a"] = out[0] if isinstance(out, tuple) else out
        self._hook = sub.register_forward_hook(hook)
        for meth in ("prefill", "step"):
            cls_fn = getattr(type(sub), meth, None)
            if cls_fn is None:
                continue

            def wrapped(self_, *args, _fn=cls_fn, _m=meth, **kw):
                out = _fn(self_, *args, **kw)
                stash["a"] = out[0] if _m == "prefill" else (out[0] if isinstance(out, tuple) else out)
                return out
            setattr(sub, meth, types.MethodType(wrapped, sub))

    def unpatch(self):
        if self._hook is not None:
            self._hook.remove(); self._hook = None
        sub = getattr(self.block, self.reader)
        for meth in ("prefill", "step"):
            if meth in sub.__dict__:
                del sub.__dict__[meth]          # the class method shows through again
        self._stash.clear()

    def _take(self):
        a = self._stash.pop("a", None)
        if a is None:
            raise RuntimeError("RecurrentWide: the hub read was not produced on this call (the attention sublayer did not run)")
        return a

    def reset_state(self):
        self._state = None

    # -- the three paths ----------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if not self.enabled:
            self._stash.clear()
            return out
        a = self._take()
        x = out[0] if isinstance(out, tuple) else out
        delta, _ = self.arm.run(a)                          # a full forward is a fresh context; never touches the carried state
        y = x + delta
        return (y,) + out[1:] if isinstance(out, tuple) else y

    def prefill(self, *args, **kwargs):
        out, cache = self.block.prefill(*args, **kwargs)
        if not self.enabled:
            self._stash.clear(); self._state = None
            return out, cache
        a = self._take()
        delta, hn = self.arm.run(a)
        self._state = hn
        return out + delta, cache

    def step(self, *args, **kwargs):
        out = self.block.step(*args, **kwargs)
        if not self.enabled:
            self._stash.clear()
            return out
        a = self._take()
        x = out[0] if isinstance(out, tuple) else out
        delta, hn = self.arm.run(a, self._state)
        self._state = hn
        y = x + delta
        return (y,) + out[1:] if isinstance(out, tuple) else y


class RecurrentHandle(AttachHandle):
    """AttachHandle (mask / only / all_off / detach verify) over BlockWithRecurrent wrappers, plus the arm verbs."""

    def detach(self, *, verify: bool = True) -> nn.Module:
        for w in self._blocks:
            w.unpatch()
        return super().detach(verify=verify)

    def arms(self) -> list[RecurrentWide]:
        return [w.arm for w in self._blocks]

    def wraps(self) -> list[BlockWithRecurrent]:
        return list(self._blocks)

    def parameters(self):
        return [p for w in self._blocks for p in w.arm.parameters()]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset_state(self):
        for w in self._blocks:
            w.reset_state()

    def state_dict(self) -> dict:
        return {f"{i}.{k}": v.detach().cpu().clone() for i, w in enumerate(self._blocks) for k, v in w.arm.state_dict().items()}

    def save(self, path: str, meta: dict | None = None) -> str:
        meta = dict(meta or {})
        meta.setdefault("created", _dt.datetime.now(_dt.timezone.utc).isoformat())
        meta["spec"] = asdict(self._blocks[0].arm.spec)
        meta["arm_type"] = "RecurrentWide" if self._blocks[0].arm.recurrent else "RecurrentWide-control (projection-only)"
        meta["params"] = self.n_params()
        torch.save({"format": FORMAT, "version": VERSION, "meta": meta, "adapters": self.state_dict()}, path)
        return path


def _per_block_state(adapters: dict, i: int) -> dict:
    pre = f"{i}."
    return {k[len(pre):]: v for k, v in adapters.items() if k.startswith(pre)}


def load_recurrent(path: str):
    """-> (adapters dict keyed '{block}.{param}', meta dict, RecurrentSpec)."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("format") != FORMAT:
        raise ValueError(f"not an {FORMAT} checkpoint: {path}")
    return blob["adapters"], blob.get("meta", {}), RecurrentSpec(**blob["meta"]["spec"])


def attach_recurrent(model, name: str, spec: RecurrentSpec | None = None, *, seed: int | None = None, binding="alephlm",
                     state: dict | None = None, reader: str = "attn", sites=None) -> RecurrentHandle:
    """Wrap every bound decoder layer (or the given `sites`) with a fresh RecurrentWide (torch seed `seed` if given) or the
    saved `state`; adapters follow their block's device; returns a RecurrentHandle. The pre-attach probe fingerprint is
    taken first so detach(verify=True) can assert bit-exactness."""
    b = resolve(model, binding)
    layers = list(b.layers(model))
    d = b.hidden_size(model)
    fingerprint = _probe(model, b)
    if seed is not None:
        torch.manual_seed(int(seed))
    spec = spec or RecurrentSpec()
    blocks, wraps = [], []
    for i, layer in enumerate(layers):
        if sites is not None and i not in sites:
            blocks.append(layer); continue
        arm = RecurrentWide(d, spec)
        if state is not None:
            arm.load_state_dict(_per_block_state(state, i))
        arm.to(next(layer.parameters()).device)
        if arm.recurrent:
            arm.cell.flatten_parameters()
        w = BlockWithRecurrent(layer, arm, reader)
        blocks.append(w); wraps.append(w)
    b.set_layers(model, blocks)
    return RecurrentHandle(model, b, layers, [name], wraps, fingerprint)


def attach_saved_recurrent(model, path: str, name: str | None = None, *, binding="alephlm", reader: str = "attn") -> RecurrentHandle:
    adapters, meta, spec = load_recurrent(path)
    return attach_recurrent(model, name or meta.get("name", "recurrent"), spec, binding=binding, state=adapters, reader=reader)
