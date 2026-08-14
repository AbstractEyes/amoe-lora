"""attach / AttachHandle / detach — the runtime verbs.

attach(model, anchor) wraps every bound decoder layer; adapters follow
their block's device (device_map="auto" compatible). detach(verify=True)
restores the retained layers and asserts BIT-EXACT equality with a
pre-attach probe (fp32; dtype-conditional under bf16).
"""
from __future__ import annotations

import contextlib
from typing import Sequence

import torch
import torch.nn as nn

from ..binding.resolver import resolve
from ..core.adapter import AdapterSpec, BlockWithAdapter, RelayPatchwork
from ..core.dispatch import AnchorDispatch, BlockWithDispatch
from ..io.checkpoint import (AnchorCheckpoint, DispatchCheckpoint,
                             load_anchor, load_dispatch)


def _per_block_state(adapters: dict, i: int) -> dict:
    pre = f"{i}."
    return {k[len(pre):]: v for k, v in adapters.items()
            if k.startswith(pre)}


def _probe(model, binding) -> torch.Tensor:
    """Deterministic logits fingerprint on a tiny fixed input."""
    dev = next(model.parameters()).device
    ids = torch.arange(8, dtype=torch.long, device=dev).unsqueeze(0) % 7 + 1
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits.detach().float().cpu()


class AttachHandle:
    def __init__(self, model, binding, original_layers, names,
                 per_block_modules, fingerprint):
        self.model = model
        self.binding = binding
        self._original = original_layers
        self.names = list(names)
        self._blocks = per_block_modules   # BlockWithAdapter|BlockWithDispatch list
        self._fingerprint = fingerprint

    # -- masking -------------------------------------------------------
    def set_mask(self, mask: dict[str, bool]) -> None:
        enabled = [mask.get(n, True) for n in self.names]
        for b in self._blocks:
            if isinstance(b, BlockWithDispatch):
                b.disp.enabled = list(enabled)
            else:
                b.enabled = enabled[0]

    def enable(self, *names):
        self.set_mask({n: (n in names or not names) for n in self.names})

    def disable(self, *names):
        self.set_mask({n: n not in names for n in self.names})

    @contextlib.contextmanager
    def only(self, *names):
        prev = self._snapshot()
        self.set_mask({n: n in names for n in self.names})
        try:
            yield self
        finally:
            self._restore(prev)

    @contextlib.contextmanager
    def all_off(self):
        prev = self._snapshot()
        self.set_mask({n: False for n in self.names})
        try:
            yield self
        finally:
            self._restore(prev)

    def _snapshot(self):
        out = []
        for b in self._blocks:
            out.append(list(b.disp.enabled)
                       if isinstance(b, BlockWithDispatch) else b.enabled)
        return out

    def _restore(self, snap):
        for b, s in zip(self._blocks, snap):
            if isinstance(b, BlockWithDispatch):
                b.disp.enabled = list(s)
            else:
                b.enabled = s

    # -- telemetry -----------------------------------------------------
    def telemetry(self, on: bool) -> None:
        for b in self._blocks:
            if isinstance(b, BlockWithDispatch):
                b.disp.rec = [] if on else None

    def usage(self) -> dict[str, float]:
        import torch as _t
        ids = [b.disp._last_shadow.flatten() for b in self._blocks
               if isinstance(b, BlockWithDispatch)
               and b.disp._last_shadow is not None]
        if not ids:
            return {}
        counts = _t.bincount(_t.cat(ids), minlength=len(self.names)).float()
        p = counts / counts.sum()
        return {n: round(float(v), 4) for n, v in zip(self.names, p)}

    def amplitude(self) -> dict[str, float]:
        recs = [b.disp.rec for b in self._blocks
                if isinstance(b, BlockWithDispatch) and b.disp.rec]
        if not recs:
            return {}
        amp = torch.stack([torch.stack(r).mean(0) for r in recs]).mean(0)
        return {n: round(float(a), 5) for n, a in zip(self.names, amp)}

    # -- detach --------------------------------------------------------
    def detach(self, *, verify: bool = True) -> nn.Module:
        self.binding.set_layers(self.model, list(self._original))
        if verify:
            post = _probe(self.model, self.binding)
            if not torch.equal(post, self._fingerprint):
                raise RuntimeError(
                    "detach verification FAILED: post-detach logits are "
                    "not bit-exact with the pre-attach fingerprint")
        return self.model


def attach(model, anchors, dispatch=None, *, binding=None,
           spec: AdapterSpec | None = None,
           strict: bool = True) -> AttachHandle:
    b = resolve(model, binding)
    layers = list(b.layers(model))
    d = b.hidden_size(model)
    fingerprint = _probe(model, b)

    def _as_ckpt(a):
        if isinstance(a, AnchorCheckpoint):
            return a
        return load_anchor(a)

    single = not isinstance(anchors, (list, tuple))
    anchor_list = [anchors] if single else list(anchors)
    ckpts = [_as_ckpt(a) for a in anchor_list]
    names = [c.meta.get("name", f"anchor{i}")
             for i, c in enumerate(ckpts)]

    if strict:
        # Identity comes from the BINDING, not from an assumed HF config:
        # substrates without a .config (e.g. AlephLM) used to make this
        # check silently inert, which is worse than no check at all.
        live = (b.identity(model) if hasattr(b, "identity")
                else getattr(getattr(model, "config", None),
                             "_name_or_path", None))
        for c in ckpts:
            bid = c.meta.get("base_model_id")
            if not bid:
                continue
            if live is None:
                raise ValueError(
                    f"anchor '{c.meta.get('name')}' declares base_model_id "
                    f"'{bid}' but this substrate exposes no identity, so "
                    "provenance CANNOT be verified here — verify it in your "
                    "loader, or pass strict=False to proceed knowingly")
            bid, live_s = str(bid), str(live)
            # a binding identity may be coarser than the anchor's (a craft
            # name vs craft@step): prefix agreement either way is a match,
            # anything else is a genuine substrate mismatch
            if not (bid.startswith(live_s) or live_s.startswith(bid)):
                raise ValueError(
                    f"anchor '{c.meta.get('name')}' was trained on "
                    f"{bid}, live model is {live_s} (pass strict=False "
                    "to override)")

    blocks = []
    if single and dispatch is None:
        ck = ckpts[0]
        for i, layer in enumerate(layers):
            st = _per_block_state(ck.adapters, i)
            if not st:
                blocks.append(layer)
                continue
            a = RelayPatchwork(d, spec)
            a.load_state_dict(st)
            a.to(next(layer.parameters()).device)   # device-following
            wrapped = BlockWithAdapter(layer, a)
            blocks.append(wrapped)
    else:
        if dispatch is None:
            raise ValueError("multiple anchors require a dispatch "
                             "(align one, or pass dispatch='init')")
        disp_ck = (None if dispatch == "init"
                   else dispatch if isinstance(dispatch, DispatchCheckpoint)
                   else load_dispatch(dispatch))
        for i, layer in enumerate(layers):
            stack = nn.ModuleList()
            for ck in ckpts:
                a = RelayPatchwork(d, spec)
                a.load_state_dict(_per_block_state(ck.adapters, i))
                for p in a.parameters():
                    p.requires_grad_(False)
                stack.append(a)
            dev = next(layer.parameters()).device
            dp = AnchorDispatch(stack.to(dev), d).to(dev)
            if disp_ck is not None:
                dp.load_state_dict(disp_ck.dispatch[i], strict=False)
            blocks.append(BlockWithDispatch(layer, dp))

    b.set_layers(model, blocks)
    wrapped_blocks = [x for x in blocks
                      if isinstance(x, (BlockWithAdapter,
                                        BlockWithDispatch))]
    return AttachHandle(model, b, layers, names, wrapped_blocks,
                        fingerprint)


def detach(handle: AttachHandle, *, verify: bool = True) -> nn.Module:
    return handle.detach(verify=verify)
