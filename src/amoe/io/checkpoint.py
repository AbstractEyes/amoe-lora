"""Versioned checkpoints: amoe.anchor v1 and amoe.dispatch v1.

weights_only=True-safe (tensors + primitives). Anchor state dicts MUST
carry `addr.home` per block (law 6 of the research line). import_legacy
converts the campaign's shipped formats.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io as _io
from dataclasses import dataclass, field
from typing import Any

import torch

ANCHOR_FORMAT = "amoe.anchor"
DISPATCH_FORMAT = "amoe.dispatch"
VERSION = 1


def _content_hash(state: dict) -> str:
    buf = _io.BytesIO()
    torch.save({k: state[k] for k in sorted(state)}, buf)
    return "sha256:" + hashlib.sha256(buf.getvalue()).hexdigest()


@dataclass
class AnchorCheckpoint:
    adapters: dict[str, torch.Tensor]
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> str:
        meta = dict(self.meta)
        meta.setdefault("created", _dt.datetime.now(
            _dt.timezone.utc).isoformat())
        meta["content_hash"] = _content_hash(self.adapters)
        torch.save({"format": ANCHOR_FORMAT, "version": VERSION,
                    "meta": meta, "adapters": self.adapters}, path)
        return meta["content_hash"]


@dataclass
class DispatchCheckpoint:
    dispatch: list[dict[str, torch.Tensor]]
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        meta = dict(self.meta)
        meta.setdefault("created", _dt.datetime.now(
            _dt.timezone.utc).isoformat())
        torch.save({"format": DISPATCH_FORMAT, "version": VERSION,
                    "meta": meta, "dispatch": self.dispatch}, path)


def _require_home(adapters: dict) -> None:
    blocks = {k.split(".", 1)[0] for k in adapters}
    for b in blocks:
        if f"{b}.addr.home" not in adapters:
            raise ValueError(
                f"anchor state for block {b} lacks addr.home — the "
                "drift-gauge buffer is required (see amoe.laws)")


def load_anchor(path: str) -> AnchorCheckpoint:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("format") == ANCHOR_FORMAT:
        _require_home(blob["adapters"])
        return AnchorCheckpoint(blob["adapters"], blob.get("meta", {}))
    # legacy campaign format: flat {"adapters": {...}, ...extras}
    if "adapters" in blob:
        _require_home(blob["adapters"])
        meta = {k: v for k, v in blob.items()
                if k != "adapters" and isinstance(v, (str, int, float,
                                                      list, tuple))}
        meta["imported_from"] = "legacy_flat"
        return AnchorCheckpoint(blob["adapters"], meta)
    raise ValueError(f"unrecognized anchor checkpoint at {path}")


def load_dispatch(path: str) -> DispatchCheckpoint:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("format") == DISPATCH_FORMAT:
        return DispatchCheckpoint(blob["dispatch"], blob.get("meta", {}))
    if "dispatch" in blob:                       # legacy exp007/exp008 shape
        return DispatchCheckpoint(
            list(blob["dispatch"]), {"imported_from": "legacy_dispatch"})
    if "keys" in blob:
        raise ValueError(
            "legacy keys-only dispatch checkpoint: this format saved "
            "only the key matrix and relied on torch.manual_seed to "
            "reproduce key_proj — reconstruct via "
            "import_legacy_keys(path, seed=..., d=..., emb=...) which "
            "re-derives key_proj deterministically, and verify against "
            "a known output before trusting it across torch versions")
    raise ValueError(f"unrecognized dispatch checkpoint at {path}")


# ── diffusion anchors (amoe 0.2) ─────────────────────────────────────────

DIFF_ANCHOR_FORMAT = "amoe.diffusion.anchor"


@dataclass
class DiffusionAnchorCheckpoint:
    """One diffusion adapter stack: adapters keyed "{site_index}.{param}".
    meta.adapter.kind is 'relay' | 'multiband3' (attachable) or a
    load-only kind ('mono', 'bank', 'cond'). Relay kind requires per-site
    addr.home (drift gauge); the others have no address, so the
    requirement is kind-aware.

    The meta block is the SELF-DOCUMENTING half of the format — see
    SCHEMA.md. Everything beyond `adapter`/`substrate` is optional, so
    raw campaign stacks load unchanged while production artifacts can
    carry their own usage card (display_name / usage / evidence /
    recommended_strength / caveats / license)."""
    adapters: dict[str, torch.Tensor]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return self.meta.get("adapter", {}).get("kind", "unknown")

    @property
    def n_sites(self) -> int:
        return len({k.split(".", 1)[0] for k in self.adapters})

    @property
    def widths(self) -> list[int]:
        """Per-site hidden width, read from the tensors themselves — the
        cross-framework ALIGNMENT SIGNATURE. Site order is the training
        order (diffusers named_modules: down -> up -> mid), which is NOT
        a denoiser's execution order; any host that enumerates blocks
        differently must match on this signature, never positionally."""
        out = []
        for i in range(self.n_sites):
            sd = self.per_site(i)
            for key, dim in (("proj.weight", 1), ("down.0.weight", 1),
                             ("down.weight", 1), ("addr_proj.weight", 0)):
                if key in sd:
                    out.append(int(sd[key].shape[dim]))
                    break
            else:
                raise ValueError(f"site {i}: cannot infer width from "
                                 f"{sorted(sd)[:6]}")
        return out

    def card(self) -> dict:
        """The human-facing subset of meta (schema v1.1), with safe
        defaults — what a UI should render about this adapter."""
        m = self.meta
        return {
            "display_name": m.get("display_name", m.get("name", "unnamed")),
            "kind": self.kind,
            "n_sites": self.n_sites,
            "trunk": m.get("substrate", {}).get("base_model_id", "unknown"),
            "objective": m.get("objective", {}).get("kind", "unknown"),
            "usage": m.get("usage", ""),
            "evidence": m.get("evidence", ""),
            "recommended_strength": m.get("recommended_strength", 1.0),
            "band_roles": m.get("band_roles", []),
            "caveats": m.get("caveats", []),
            "license": m.get("license", ""),
            "nc": bool(m.get("nc", False)),
        }

    def per_site(self, i: int) -> dict[str, torch.Tensor]:
        pre = f"{i}."
        return {k[len(pre):]: v for k, v in self.adapters.items()
                if k.startswith(pre)}

    def save(self, path: str) -> str:
        if str(path).endswith(".safetensors"):
            from .safetensors_io import save_anchor_safetensors
            return save_anchor_safetensors(self, path)
        meta = dict(self.meta)
        meta.setdefault("created", _dt.datetime.now(
            _dt.timezone.utc).isoformat())
        meta["content_hash"] = _content_hash(self.adapters)
        torch.save({"format": DIFF_ANCHOR_FORMAT, "version": VERSION,
                    "meta": meta, "adapters": self.adapters}, path)
        return meta["content_hash"]


def _sd_kind(sd: dict) -> str:
    """Classify one site's state dict.

    ORDER IS LOAD-BEARING: a dispatch bank carries BOTH `key_proj.weight`
    and `down.0.weight`, so the bank test MUST come before the multiband
    test — otherwise every exp007/exp014/exp015 bank silently loads as a
    multiband3 stack and fails later at module construction.
    """
    if any(k.endswith("addr.codebook") for k in sd):
        return "relay"
    if any(k.startswith("key_proj") for k in sd) or "keys" in sd:
        return "bank"                       # dispatch bank — must precede
    if "pos_table" in sd:
        return "cond"                       # AlephCondAdapter (exp002)
    if "down.0.weight" in sd:
        return "multiband3"
    if "down.weight" in sd:
        return "mono"
    return "unknown"


def _flatten_stack(stack, *, kind: str, imported_from: str,
                   reconstruct_home: bool = False) -> "DiffusionAnchorCheckpoint":
    adapters, recon = {}, False
    for i, sd in enumerate(stack):
        sd = dict(sd)
        if kind == "relay" and "addr.home" not in sd:
            if not reconstruct_home:
                raise ValueError(
                    f"site {i} lacks addr.home (drift gauge); pass a source "
                    "that carries it or use the fork import path")
            import torch.nn.functional as _F
            sd["addr.home"] = _F.normalize(
                sd["addr.codebook"].detach().float(), dim=-1)
            recon = True
        for k, v in sd.items():
            adapters[f"{i}.{k}"] = v
    meta: dict[str, Any] = {"adapter": {"kind": kind},
                            "imported_from": imported_from}
    if recon:
        meta["home_reconstructed"] = True     # drift gauge void — never silent
    return DiffusionAnchorCheckpoint(adapters, meta)


def load_diffusion_anchor(path: str, *, substrate: "dict | None" = None
                          ) -> DiffusionAnchorCheckpoint:
    """Load a diffusion anchor: the versioned format, a .safetensors
    companion, or any of the campaign's legacy stack shapes
    ({'relays': [...]}, {'mods': [...]}, {'banks'/'monos': [...]},
    diffusion-pipe {'relays': {idx: sd}})."""
    if str(path).endswith(".safetensors"):
        from .safetensors_io import load_anchor_safetensors
        ck = load_anchor_safetensors(path)
    else:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob.get("format") == DIFF_ANCHOR_FORMAT:
            ck = DiffusionAnchorCheckpoint(blob["adapters"],
                                           blob.get("meta", {}))
        elif "relays" in blob:
            r = blob["relays"]
            if isinstance(r, dict):        # diffusion-pipe fork save
                stack = [r[k] for k in sorted(r, key=int)]
                ck = _flatten_stack(stack, kind="relay",
                                    imported_from="diffusion_pipe",
                                    reconstruct_home=True)
            else:                          # exp001/exp006 relay stacks
                ck = _flatten_stack(list(r), kind="relay",
                                    imported_from="legacy_relays")
        elif "mods" in blob:               # exp008/009/011/012/013 stacks
            stack = list(blob["mods"])
            ck = _flatten_stack(stack, kind=_sd_kind(stack[0]),
                                imported_from="legacy_mods")
        elif "banks" in blob or "monos" in blob:   # exp007 (load-only)
            key = "banks" if "banks" in blob else "monos"
            ck = _flatten_stack(list(blob[key]), kind=_sd_kind(blob[key][0]),
                                imported_from=f"legacy_{key}")
        elif _sd_kind(blob) != "unknown":
            # a FLAT single-module state dict (exp002 AlephCondAdapter):
            # one "site", stored under index 0 so per_site(0) round-trips
            ck = _flatten_stack([blob], kind=_sd_kind(blob),
                                imported_from="legacy_flat_module")
        else:
            raise ValueError(f"unrecognized diffusion anchor at {path}")
    if ck.kind == "bank":
        ck.meta.setdefault("routing_negative", (
            "exp007/014/015: comparative dispatch on diffusion is a 2-seed "
            "falsified line — load for analysis, not deployment"))
    if ck.kind == "cond":
        ck.meta.setdefault("law2_negative", (
            "exp002: the frozen address beside full text conditioning is "
            "redundant-in-context (real vs deranged -0.0009); the address "
            "ALONE steers (+0.0287). Redesign target = complementarity"))
    if ck.kind == "relay":
        blocks = {k.split(".", 1)[0] for k in ck.adapters}
        for b in blocks:
            if f"{b}.addr.home" not in ck.adapters:
                raise ValueError(f"relay anchor site {b} lacks addr.home")
    if substrate:
        ck.meta.setdefault("substrate", dict(substrate))
    return ck


def import_legacy_keys(path: str, *, seed: int, d: int,
                       emb: int = 64) -> DispatchCheckpoint:
    """Reconstruct a full dispatch state from a keys-only legacy file
    (the v35e14_keys*.pt shape). key_proj is re-derived from the same
    torch.manual_seed sequence the original build used — verify
    numerically before trusting across torch versions."""
    from ..core.dispatch import orthogonal_rows
    blob = torch.load(path, map_location="cpu", weights_only=True)
    keys = blob["keys"]
    torch.manual_seed(1400 + seed)
    out = []
    for k in keys:
        kp = orthogonal_rows(emb, d)
        out.append({"key_proj": kp, "dispatch": k.clone()})
    return DispatchCheckpoint(
        out, {"imported_from": "legacy_keys", "seed": seed,
              "reconstruction": "key_proj re-derived from manual_seed; "
                                "verify before production use"})
