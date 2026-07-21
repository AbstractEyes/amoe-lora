"""safetensors serialization for amoe anchors (new in 0.2 — the diffusion
line needs ComfyUI-consumable artifacts; a documented 0.1 non-goal ends).

Two key layouts:
  "amoe"  (canonical, portable): blocks.{site_index}.{param_path}
  "comfy" (export): diffusion_model.{site_name}.aleph_relay.{param_path}
          — the cosmos_predict2 save_adapter precedent. Requires
          meta.substrate.site_names. The ComfyUI node consumes the
          canonical layout and asserts the width signature (F2).

safetensors metadata is str:str only, so the full meta rides as one JSON
blob ("amoe_meta") plus greppable duplicates. Content-hash v2 is defined
over sorted(key)+shape+dtype+little-endian bytes — format-independent
(the .pt v1 hash depends on torch.save serialization).
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:   # pragma: no cover
    from .checkpoint import DiffusionAnchorCheckpoint


def _require_safetensors():
    try:
        import safetensors.torch as st  # noqa: F401
        return st
    except ImportError as e:            # pragma: no cover
        raise ImportError(
            "safetensors is required for this path: "
            "pip install amoe-lora[diffusion] (or pip install safetensors)"
        ) from e


def content_hash_v2(tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        # byte view works for every dtype incl. bf16 (no numpy dtype needed)
        h.update(t.flatten().view(torch.uint8).numpy().tobytes())
    return "sha256v2:" + h.hexdigest()


def save_anchor_safetensors(ck: "DiffusionAnchorCheckpoint", path: str,
                            *, key_layout: str = "amoe") -> str:
    st = _require_safetensors()
    if key_layout == "amoe":
        tensors = {f"blocks.{k}": v.detach().cpu().contiguous()
                   for k, v in ck.adapters.items()}
    elif key_layout == "comfy":
        names = ck.meta.get("substrate", {}).get("site_names")
        if not names:
            raise ValueError(
                "comfy layout needs meta.substrate.site_names (the trunk "
                "module paths); save the canonical 'amoe' layout instead")
        tensors = {}
        for k, v in ck.adapters.items():
            i, param = k.split(".", 1)
            tensors[f"diffusion_model.{names[int(i)]}.aleph_relay.{param}"] = \
                v.detach().cpu().contiguous()
    else:
        raise ValueError(f"unknown key_layout '{key_layout}'")
    chash = content_hash_v2({k: v for k, v in ck.adapters.items()})
    meta = dict(ck.meta)
    meta["content_hash_v2"] = chash
    md = {"format": "amoe.diffusion.anchor", "version": "1",
          "key_layout": key_layout, "amoe_meta": json.dumps(meta),
          "adapter_kind": ck.kind,
          "substrate_family": str(meta.get("substrate", {}).get("family", "")),
          "content_hash_v2": chash}
    st.save_file(tensors, path, metadata=md)
    return chash


def load_anchor_safetensors(path: str) -> "DiffusionAnchorCheckpoint":
    st = _require_safetensors()
    from .checkpoint import DiffusionAnchorCheckpoint
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    if md.get("format") != "amoe.diffusion.anchor":
        raise ValueError(f"not an amoe.diffusion.anchor safetensors: {path}")
    meta = json.loads(md.get("amoe_meta", "{}"))
    layout = md.get("key_layout", "amoe")
    if layout == "amoe":
        adapters = {k[len("blocks."):]: v for k, v in tensors.items()}
    else:                               # comfy layout round-trip
        names = meta.get("substrate", {}).get("site_names", [])
        idx = {n: i for i, n in enumerate(names)}
        adapters = {}
        for k, v in tensors.items():
            body = k[len("diffusion_model."):]
            site, param = body.split(".aleph_relay.")
            adapters[f"{idx[site]}.{param}"] = v
    ck = DiffusionAnchorCheckpoint(adapters, meta)
    want = md.get("content_hash_v2")
    if want:
        got = content_hash_v2(adapters)
        if got != want:
            raise ValueError(f"content hash mismatch in {path}: "
                             f"{got} != {want}")
    return ck
