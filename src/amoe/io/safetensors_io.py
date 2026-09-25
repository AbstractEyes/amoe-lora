"""safetensors serialization for amoe anchors (new in 0.2 — the diffusion
line needs ComfyUI-consumable artifacts; a documented 0.1 non-goal ends).
0.2.6 adds block anchors (amoe.anchor): AnchorCheckpoint.save writes this
format for a path ending in .safetensors, and load_anchor reads either
format by content (is_safetensors), never by file name.

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
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:   # pragma: no cover
    from .checkpoint import AnchorCheckpoint, DiffusionAnchorCheckpoint


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


# ── block anchors (amoe.anchor, 0.2.6) ──────────────────────────────────

def is_safetensors(src) -> bool:
    """True when `src` (a path or a seekable binary handle) holds a
    safetensors file: an 8-byte little-endian header length followed by a
    JSON object that fits inside the file. Decided by content, never by
    the file name; a handle's read position is restored."""
    try:
        if hasattr(src, "read"):
            pos = src.tell()
            try:
                head = src.read(9)
                src.seek(0, os.SEEK_END)
                size = src.tell() - pos
            finally:
                src.seek(pos)
        else:
            size = os.path.getsize(src)
            with open(src, "rb") as fh:
                head = fh.read(9)
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    if len(head) < 9 or head[8:9] != b"{":
        return False
    n = int.from_bytes(head[:8], "little")
    return 0 < n <= size - 8


def save_block_anchor_safetensors(ck: "AnchorCheckpoint", path: str) -> str:
    """Write a block anchor (amoe.anchor) as safetensors: tensors under the
    canonical layout blocks.{block_index}.{param_path}, the meta as one
    JSON blob ("amoe_meta") plus greppable duplicates. The meta must be
    JSON-serializable. Returns the format-independent content hash."""
    st = _require_safetensors()
    from .checkpoint import ANCHOR_FORMAT, VERSION
    chash = content_hash_v2(ck.adapters)
    meta = dict(ck.meta)
    meta["content_hash_v2"] = chash
    try:
        blob = json.dumps(meta)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "anchor meta must be JSON-serializable to be stored in "
            f"safetensors metadata ({e}); simplify the meta or save to a "
            ".pt path") from e
    tensors = {f"blocks.{k}": v.detach().to("cpu", copy=True).contiguous()
               for k, v in ck.adapters.items()}
    md = {"format": ANCHOR_FORMAT, "version": str(VERSION),
          "key_layout": "amoe", "amoe_meta": blob, "content_hash_v2": chash}
    for key in ("name", "base_model_id", "content_hash"):
        if isinstance(meta.get(key), str):
            md[key] = meta[key]
    st.save_file(tensors, str(path), metadata=md)
    return chash


def load_block_anchor_safetensors(src) -> "AnchorCheckpoint":
    """Read a block anchor (amoe.anchor) safetensors file from a path or a
    binary handle; content_hash_v2 is verified when present."""
    st = _require_safetensors()
    from .checkpoint import ANCHOR_FORMAT, AnchorCheckpoint
    where = getattr(src, "name", src)
    if hasattr(src, "read"):
        data = src.read()
        n = int.from_bytes(data[:8], "little")
        md = json.loads(data[8:8 + n].decode("utf-8")).get("__metadata__") or {}
        tensors = st.load(data)
    else:
        from safetensors import safe_open
        with safe_open(str(src), framework="pt", device="cpu") as f:
            md = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in f.keys()}
    if md.get("format") != ANCHOR_FORMAT:
        raise ValueError(f"not an {ANCHOR_FORMAT} safetensors file: {where} "
                         f"(format={md.get('format')!r})")
    if md.get("key_layout", "amoe") != "amoe" or \
            not all(k.startswith("blocks.") for k in tensors):
        raise ValueError(f"unsupported key layout in {where}: expected "
                         "blocks.{block_index}.{param_path}")
    adapters = {k[len("blocks."):]: v for k, v in tensors.items()}
    want = md.get("content_hash_v2")
    if want:
        got = content_hash_v2(adapters)
        if got != want:
            raise ValueError(f"content hash mismatch in {where}: "
                             f"{got} != {want}")
    return AnchorCheckpoint(adapters, json.loads(md.get("amoe_meta", "{}")))
