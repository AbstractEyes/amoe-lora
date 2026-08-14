"""Model binding: where do the adapters attach?

Everything architecture-specific lives in binding/ and only here (the
substrate-shim discipline from the research line). Resolution order:
explicit binding object -> dotted path string -> registry by
model_type/class -> error listing candidate ModuleLists (never guess).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import torch.nn as nn


class ModelBinding(Protocol):
    name: str

    def layers(self, model) -> Sequence[nn.Module]: ...
    def set_layers(self, model, new: list[nn.Module]) -> None: ...
    def hidden_size(self, model) -> int: ...


@dataclass
class PathBinding:
    """Generic binding around a dotted attribute path to a ModuleList."""
    path: str
    name: str = "path"

    def _parent(self, model):
        obj = model
        parts = self.path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return obj, parts[-1]

    def layers(self, model):
        parent, leaf = self._parent(model)
        return getattr(parent, leaf)

    def set_layers(self, model, new):
        parent, leaf = self._parent(model)
        setattr(parent, leaf, nn.ModuleList(new))

    def hidden_size(self, model) -> int:
        cfg = model.config
        for attr in ("text_config", None):
            c = getattr(cfg, attr, cfg) if attr else cfg
            if hasattr(c, "hidden_size"):
                return int(c.hidden_size)
        raise AttributeError("no hidden_size on model.config")

    def identity(self, model) -> str | None:
        return getattr(getattr(model, "config", None), "_name_or_path", None)


REGISTRY: dict[str, Callable[[], ModelBinding]] = {}


def register(key: str):
    def deco(fn):
        REGISTRY[key] = fn
        return fn
    return deco


@register("qwen3_5_vl")
def _qwen35() -> ModelBinding:
    # dual-tower VLM: LLM tower at model.model.language_model.layers
    return PathBinding("model.language_model.layers", name="qwen3_5_vl")


@register("generic_causal")
def _generic() -> ModelBinding:
    # Llama/Qwen3/Mistral-style: model.model.layers
    return PathBinding("model.layers", name="generic_causal")


@dataclass
class AlephLMBinding:
    """geolip.alephllm AlephLM: blocks at model.blocks, width on the
    dataclass config (model.cfg.d_model) — no HF config object."""
    name: str = "alephlm"

    def layers(self, model):
        return model.blocks

    def set_layers(self, model, new):
        model.blocks = nn.ModuleList(new)

    def hidden_size(self, model) -> int:
        return int(model.cfg.d_model)

    def identity(self, model) -> str | None:
        n = getattr(getattr(model, "cfg", None), "name", None)
        return f"alephllm/{n}" if n else None


@register("alephlm")
def _alephlm() -> ModelBinding:
    return AlephLMBinding()


def _candidates(model) -> list[str]:
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
            out.append(f"{name} (len {len(mod)})")
    return out


def resolve(model, binding: "ModelBinding | str | None" = None) -> ModelBinding:
    if binding is None:
        # AlephLM ducks first: dataclass cfg, no .config object at all
        if hasattr(model, "blocks") and \
                hasattr(getattr(model, "cfg", None), "d_model"):
            return REGISTRY["alephlm"]()
        mt = getattr(getattr(model, "config", None), "model_type", "") or ""
        if "qwen3_5" in mt and hasattr(model, "model") and \
                hasattr(model.model, "language_model"):
            return REGISTRY["qwen3_5_vl"]()
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return REGISTRY["generic_causal"]()
        raise ValueError(
            "amoe could not resolve an attach site. Pass "
            "binding=<dotted path to the decoder ModuleList>. "
            f"Candidates found: {_candidates(model)}")
    if isinstance(binding, str):
        if binding in REGISTRY:
            return REGISTRY[binding]()
        # dotted path relative to the model object
        return PathBinding(binding)
    return binding
