"""Diffusion model binding: where do the adapters attach on a denoiser?

Same substrate-shim discipline as resolver.py — everything
architecture-specific lives here and only here. Diffusion sites are
SCATTERED (heterogeneous widths across down/mid/up), so the binding
speaks (qualified_name, module, width) triples and replaces modules by
attribute walk, not by swapping one ModuleList.

DTYPE LAW: `declared_dtype` is the trunk dtype AFTER materialization.
It refuses meta-device models — first-param sniffing during init can
read a still-meta fp32 tensor (the R0b catch, fork f4f1c46). Attach
after weights are real, or pass dtype= explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch
import torch.nn as nn

from .resolver import REGISTRY


class DiffusionBinding(Protocol):
    name: str
    expected_sites: int | None

    def sites(self, model) -> list[tuple[str, nn.Module, int]]: ...
    def replace(self, model, name: str, new: nn.Module) -> None: ...
    def declared_dtype(self, model) -> torch.dtype: ...
    def probe(self, model) -> "torch.Tensor | None": ...


def walk_replace(root: nn.Module, name: str, new: nn.Module) -> None:
    """setattr/index walk to a qualified module name (the certified
    attach pattern from the r2 beds)."""
    parent = root
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def _declared_dtype(model) -> torch.dtype:
    metas = [n for n, p in model.named_parameters() if p.is_meta]
    if metas:
        raise RuntimeError(
            "dtype law: model has meta-device parameters "
            f"(e.g. {metas[0]}) — a dtype read here can lie (the R0b "
            "catch). Materialize the weights first, or pass dtype= "
            "explicitly at attach.")
    dt = getattr(model, "dtype", None)
    if isinstance(dt, torch.dtype):
        return dt
    return next(model.parameters()).dtype


@dataclass
class UNetBinding:
    """diffusers UNet2DConditionModel — every BasicTransformerBlock.
    SD15 certifies at 16 sites (asserted). SDXL's count is PINNED AT THE
    FIRST REAL RUN (deferred-by-design F1): expected_sites=None means
    'record, do not assert a guess'."""
    name: str = "sd15_unet"
    expected_sites: "int | None" = 16

    def sites(self, model):
        from diffusers.models.attention import BasicTransformerBlock
        out = []
        for name, mod in model.named_modules():
            if isinstance(mod, BasicTransformerBlock):
                out.append((name, mod, mod.norm1.normalized_shape[0]))
        if self.expected_sites is not None:
            assert len(out) == self.expected_sites, (
                f"{self.name}: enumerated {len(out)} BasicTransformerBlocks, "
                f"expected {self.expected_sites}")
        assert out, f"{self.name}: no BasicTransformerBlocks found"
        return out

    def replace(self, model, name, new):
        walk_replace(model, name, new)

    def declared_dtype(self, model):
        return _declared_dtype(model)

    def probe(self, model):
        """Deterministic denoiser fingerprint on a tiny fixed input,
        built from the LIVE config (never from memory — F4)."""
        cfg = model.config
        n_down = len(cfg.block_out_channels)
        size = 2 ** (n_down - 1) * 4               # divisible through the UNet
        g = torch.Generator().manual_seed(1400)
        dev = next(model.parameters()).device
        dt = self.declared_dtype(model)
        x = torch.randn(1, cfg.in_channels, size, size, generator=g).to(dev, dt)
        ehs = torch.randn(1, 8, cfg.cross_attention_dim, generator=g).to(dev, dt)
        kwargs = {}
        if getattr(cfg, "addition_embed_type", None) == "text_time":
            # SDXL-style added conditioning, shapes from the live config
            ta = torch.randn(1, cfg.projection_class_embeddings_input_dim
                             - 6 * cfg.addition_time_embed_dim,
                             generator=g).to(dev, dt)
            tids = torch.tensor([[size * 8, size * 8, 0, 0, size * 8, size * 8]],
                                device=dev)
            kwargs["added_cond_kwargs"] = {"text_embeds": ta, "time_ids": tids}
        with torch.no_grad():
            out = model(x, 17, ehs, return_dict=False, **kwargs)[0]
        return out.detach().float().cpu()


@dataclass
class DiTBinding:
    """Cosmos-Predict2/Anima-style DiT: a `blocks` ModuleList of uniform
    width. The production trainer for this family is the diffusion-pipe
    fork (which carries its own P-INIT/P-TOGGLE gates); the in-package
    probe is None — detach verification degrades to structural restore
    with a warning (documented F4 deferral)."""
    name: str = "cosmos_dit"
    expected_sites: "int | None" = None

    def sites(self, model):
        blocks = getattr(model, "blocks", None)
        assert isinstance(blocks, nn.ModuleList) and len(blocks) > 0, (
            "cosmos_dit binding expects a `blocks` ModuleList")
        d = None
        for attr in ("model_channels", "hidden_size", "dim"):
            v = getattr(getattr(model, "config", model), attr,
                        getattr(model, attr, None))
            if isinstance(v, int):
                d = v
                break
        if d is None:
            lin = next(m for m in blocks[0].modules()
                       if isinstance(m, nn.Linear))
            d = lin.in_features
        return [(f"blocks.{i}", b, d) for i, b in enumerate(blocks)]

    def replace(self, model, name, new):
        walk_replace(model, name, new)

    def declared_dtype(self, model):
        return _declared_dtype(model)

    def probe(self, model):
        return None


DIFF_REGISTRY: dict[str, Callable[[], "DiffusionBinding"]] = {}


def register_diffusion(key: str):
    def deco(fn):
        DIFF_REGISTRY[key] = fn
        REGISTRY[key] = fn          # discoverable from the one registry
        return fn
    return deco


@register_diffusion("sd15_unet")
def _sd15():
    return UNetBinding(name="sd15_unet", expected_sites=16)


@register_diffusion("sdxl_unet")
def _sdxl():
    return UNetBinding(name="sdxl_unet", expected_sites=None)   # F1


@register_diffusion("cosmos_dit")
def _cosmos():
    return DiTBinding()


def resolve_diffusion(model, binding=None) -> DiffusionBinding:
    if binding is not None:
        if isinstance(binding, str):
            if binding in DIFF_REGISTRY:
                return DIFF_REGISTRY[binding]()
            raise ValueError(f"unknown diffusion binding '{binding}'; "
                             f"known: {sorted(DIFF_REGISTRY)}")
        return binding
    cls = type(model).__name__
    if cls == "UNet2DConditionModel":
        cad = int(getattr(model.config, "cross_attention_dim", 0) or 0)
        return DIFF_REGISTRY["sdxl_unet" if cad >= 2048 else "sd15_unet"]()
    if isinstance(getattr(model, "blocks", None), nn.ModuleList):
        return DIFF_REGISTRY["cosmos_dit"]()
    cands = [f"{n} (len {len(m)})" for n, m in model.named_modules()
             if isinstance(m, nn.ModuleList) and len(m) >= 4]
    raise ValueError(
        "amoe.diffusion could not resolve an attach site — pass "
        f"binding=<DiffusionBinding|key>. ModuleList candidates: {cands}")
