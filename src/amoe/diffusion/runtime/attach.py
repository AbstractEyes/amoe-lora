"""attach / DiffusionAttachHandle / detach — the diffusion runtime verbs.

attach(model, anchor) wraps every bound site (RelayPatch2D or
MultibandDelta by anchor kind) at the DECLARED trunk dtype (dtype law).
detach(verify=True) restores the retained modules and asserts BIT-EXACT
equality with a pre-attach probe where the binding provides one
(UNet bindings do; the DiT binding defers to the diffusion-pipe gates —
documented F4 — and degrades to structural restore with a warning).
"""
from __future__ import annotations

import contextlib
import warnings

import torch
import torch.nn as nn

from ...binding.diffusion import resolve_diffusion
from ...io.checkpoint import DiffusionAnchorCheckpoint, load_diffusion_anchor
from ..core.multiband import BandBlockWrap, MultibandDelta, band_weights
from ..core.relay import BlockWithRelay, RelayPatch2D
from ..laws import N_BANDS


class DiffusionAttachHandle:
    def __init__(self, model, binding, originals, names, wraps, fingerprint):
        self.model = model
        self.binding = binding
        self._originals = originals          # [(qualified_name, module)]
        self.names = list(names)
        self._wraps = wraps                  # BlockWithRelay | BandBlockWrap
        self._fingerprint = fingerprint

    @property
    def kind(self) -> str:
        return ("multiband3"
                if isinstance(self._wraps[0].mod, MultibandDelta) else "relay")

    def modules(self):
        return [w.mod for w in self._wraps]

    # -- masking / toggling (never renormalizes anything) ---------------
    def set_mask(self, mask: dict[str, bool]) -> None:
        for n, w in zip(self.names, self._wraps):
            w.mod.enabled = mask.get(n, True)

    @contextlib.contextmanager
    def only(self, *names):
        prev = [w.mod.enabled for w in self._wraps]
        self.set_mask({n: n in names for n in self.names})
        try:
            yield self
        finally:
            for w, e in zip(self._wraps, prev):
                w.mod.enabled = e

    @contextlib.contextmanager
    def all_off(self):
        prev = [w.mod.enabled for w in self._wraps]
        self.set_mask({n: False for n in self.names})
        try:
            yield self
        finally:
            for w, e in zip(self._wraps, prev):
                w.mod.enabled = e

    @contextlib.contextmanager
    def lesion_band(self, *bands: int):
        """Disable band expert(s) at every site (the exp008/exp010 lesion
        instrument). Multiband anchors only."""
        if self.kind != "multiband3":
            raise TypeError("lesion_band is a multiband instrument; this "
                            f"anchor is kind={self.kind}")
        bad = [b for b in bands if not 0 <= b < N_BANDS]
        if bad:
            raise ValueError(f"bands out of range: {bad}")
        prev = [list(w.mod.band_enabled) for w in self._wraps]
        for w in self._wraps:
            for b in bands:
                w.mod.band_enabled[b] = False
        try:
            yield self
        finally:
            for w, p in zip(self._wraps, prev):
                w.mod.band_enabled = p

    # -- band windows (the step gate) ------------------------------------
    def set_band_windows(self, s01: torch.Tensor) -> None:
        """s01: (B,) noise levels in [0,1] -> cosine crossfade windows on
        every wrap. No-op for relay anchors (windows unused)."""
        w = band_weights(s01)
        for wr in self._wraps:
            wr.w_bands = w

    # -- telemetry --------------------------------------------------------
    def telemetry(self, on: bool) -> None:
        for w in self._wraps:
            if hasattr(w.mod, "telemetry"):
                w.mod.telemetry = on

    def amplitude(self) -> dict[str, float]:
        out = {}
        for n, w in zip(self.names, self._wraps):
            r = getattr(w.mod, "last_delta_ratio", None)
            if r is not None:
                out[n] = round(float(r), 6)
        return out

    def gates(self) -> dict:
        vals = []
        for w in self._wraps:
            m = w.mod
            if hasattr(m, "gate"):
                vals.append(torch.sigmoid(m.gate).item())
            elif hasattr(m, "gates"):
                vals.extend(torch.sigmoid(m.gates).tolist())
        return {"gate_mean": round(sum(vals) / max(len(vals), 1), 5),
                "gate_min": round(min(vals), 5) if vals else None,
                "gate_max": round(max(vals), 5) if vals else None}

    # -- detach -----------------------------------------------------------
    def detach(self, *, verify: bool = True) -> nn.Module:
        for name, mod in self._originals:
            self.binding.replace(self.model, name, mod)
        if verify:
            if self._fingerprint is None:
                warnings.warn(
                    "detach: no pre-attach fingerprint for this binding "
                    "(probe deferred — F4); structural restore only. The "
                    "diffusion-pipe path carries its own parity gates.")
            else:
                post = self.binding.probe(self.model)
                if not torch.equal(post, self._fingerprint):
                    raise RuntimeError(
                        "detach verification FAILED: post-detach output is "
                        "not bit-exact with the pre-attach fingerprint")
        return self.model


def attach(model, anchor, *, binding=None, dtype: "torch.dtype | None" = None,
           strict: bool = True) -> DiffusionAttachHandle:
    b = resolve_diffusion(model, binding)
    sites = b.sites(model)

    ck = (anchor if isinstance(anchor, DiffusionAnchorCheckpoint)
          else load_diffusion_anchor(anchor))
    kind = ck.kind
    if kind not in ("relay", "multiband3"):
        raise ValueError(
            f"anchor kind '{kind}' is not attachable — 'mono'/'bank' are "
            "research-record controls (monolith falsifier / falsified "
            "dispatch bank), loadable for analysis via "
            "load_diffusion_anchor only. See "
            "amoe.diffusion.laws.ROUTING_NEGATIVES.")
    if ck.n_sites != len(sites):
        raise ValueError(
            f"anchor carries {ck.n_sites} sites, binding '{b.name}' "
            f"enumerates {len(sites)} — wrong substrate?")
    if strict:
        fam = ck.meta.get("substrate", {}).get("family")
        if fam and fam not in b.name:
            raise ValueError(
                f"anchor substrate family '{fam}' vs binding '{b.name}' "
                "(pass strict=False to override)")

    dt = dtype if dtype is not None else b.declared_dtype(model)   # dtype law
    fingerprint = b.probe(model)

    originals, wraps, names = [], [], []
    for i, (name, block, d) in enumerate(sites):
        sd = ck.per_site(i)
        if kind == "relay":
            mod = RelayPatch2D.from_state_dict(sd)
            assert mod.d == d, (f"site {i} ({name}): anchor width {mod.d} "
                                f"!= trunk width {d}")
            wrap = BlockWithRelay(block, mod)
        else:
            mod = MultibandDelta.from_state_dict(sd)
            assert mod.d == d, (f"site {i} ({name}): anchor width {mod.d} "
                                f"!= trunk width {d}")
            wrap = BandBlockWrap(block, mod)
        p0 = next(block.parameters(), None)
        dev = p0.device if p0 is not None else torch.device("cpu")
        mod.to(device=dev, dtype=dt)
        b.replace(model, name, wrap)
        originals.append((name, block))
        wraps.append(wrap)
        names.append(name)
    return DiffusionAttachHandle(model, b, originals, names, wraps,
                                 fingerprint)


def detach(handle: DiffusionAttachHandle, *,
           verify: bool = True) -> nn.Module:
    return handle.detach(verify=verify)
