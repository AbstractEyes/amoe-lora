"""The addressed-conv bed runner — the head-to-head against a plain conv.

Self-contained on purpose (the plan's Option A): it does NOT go through
`loop.run`/`build_model`/`TinyTrunk`, because those derive a (B,T,d) residual
stream and the amoe adapter/toggle path, and a conv is (B,C,H,W) with no
adapter. Instead it reuses every DECOUPLED, stable piece — the pure-Adam
optimizer law, the fixed-train-slice train/val snapshot, the ledger, the
verdict table, and publish — and adds only what a conv bed needs: the address
modes as arms, a batch-time CIFAR aug, grad-clip max(loss,1), and the Law-2
address-usage readout.

The arms are ADDRESS MODES of `AddressedConv2d`, not amoe anchors:
    soft   the aleph read (per-position filter composition)
    sign   hard one-hot branch (straight-through)
    none   uniform address == a plain conv with the mean filter (THE control,
           param-matched: codebook present but unread, zero gradient)
    learned  a non-aleph learned-softmax dynamic conv (is it the aleph, or any
             dynamic conv?)
    off    a lean plain conv (no address params)

`none`/`off` produce identical outputs; `none` is the param-matched control,
`off` the lean baseline. The load-bearing control is the reduction itself
(`AddressedConv2d(none)` == a fixed conv), asserted in the smoke tests.
"""
from __future__ import annotations

import time
from dataclasses import asdict, replace

import torch
import torch.nn.functional as F

from amoe import laws

from ..config import RunConfig
from ..data import Bed, build_bed
from ..config import resolve_device
from ..diagnostics import probes
from ..model.addressed_conv import build_conv_trunk
from ..model.antipode_conv import build_conv_token_trunk
from .ledger import append_ledger, ledger_path
from .loop import _fmt_progress, _snapshot

CONV_ARMS = ("soft", "sign", "none", "learned", "off")     # addr_conv (filter)
CONV_TOKEN_ARMS = ("soft", "mag", "none", "off")           # conv_tokens (read)


def _build_trunk(bed, cfg):
    """Dispatch on input_mode: conv_tokens = conv stem + per-token signed
    antipode read; addr_conv = the filter-steering primitive (the cautionary
    mean-collapse control)."""
    if getattr(cfg, "input_mode", "") == "conv_tokens":
        return build_conv_token_trunk(bed, cfg)
    return build_conv_trunk(bed, cfg)


def _report(trunk, x) -> dict:
    """Unified 'is the address meaningful?' gauge. conv_tokens reports the read
    amplitude (contribution to the stream); addr_conv reports address usage."""
    if hasattr(trunk, "read_report"):
        return trunk.read_report(x)
    return trunk.address_report(x)


# ------------------------------------------------------ CIFAR batch aug
def _augment(x: torch.Tensor, channels: int, h: int, w: int,
             pad: int = 4) -> torch.Tensor:
    """RandomCrop(pad)+HFlip in tensor space (the Bed is static flat tensors,
    so aug happens per batch, not in a DataLoader). Applied to a whole batch;
    every arm sees identically-seeded aug so the comparison stays fair."""
    B = x.shape[0]
    img = x.view(B, channels, h, w)
    img = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    top = torch.randint(0, 2 * pad + 1, (1,)).item()
    left = torch.randint(0, 2 * pad + 1, (1,)).item()
    img = img[:, :, top:top + h, left:left + w]
    if torch.rand(1).item() < 0.5:
        img = img.flip(-1)
    return img.reshape(B, -1)


# ------------------------------------------------------ generative helpers
@torch.no_grad()
def _quantize(x: torch.Tensor, n_bins: int, lo: float = -3.0,
              hi: float = 3.0) -> torch.Tensor:
    """Normalized pixels -> byte-bin indices in [0, n_bins). Targets for the
    masked-pixel reconstruction objective."""
    t = ((x - lo) / (hi - lo) * n_bins).floor().clamp(0, n_bins - 1)
    return t.long()


@torch.no_grad()
def _eval_bpb(trunk, x, channels, h, w, n_bins, mask_frac=0.5,
              batch=512) -> dict:
    """Masked-pixel NLL in bits/pixel over the masked positions."""
    trunk.eval()
    tot, n = 0.0, 0
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch]
        tgt = _quantize(xb, n_bins).view(-1, channels, h, w)[:, 0]   # (B,h,w)
        g = torch.rand(xb.shape[0], 1, h, w, device=xb.device) < mask_frac
        masked = xb.view(-1, channels, h, w).clone()
        masked = masked * (~g)                          # zero the masked pixels
        logits = trunk(masked.reshape(xb.shape[0], -1)).logits  # (B,bins,h,w)
        ce = F.cross_entropy(logits, tgt, reduction="none")     # (B,h,w)
        m = g[:, 0]
        tot += float((ce * m).sum())
        n += int(m.sum())
    return {"bpb": tot / max(n, 1) / 0.6931471805599453, "ce": tot / max(n, 1),
            "acc": 0.0, "n": n}


# --------------------------------------------------------------- one arm
def run_conv(cfg: RunConfig, bed: Bed, base_state=None) -> dict:
    """Train one addressed-conv arm and return a ledger row. `cfg.mode` is the
    ADDRESS mode. Full co-training from init (no phase-0 for conv)."""
    laws.pin_precision()
    torch.manual_seed(cfg.seed)
    trunk = _build_trunk(bed, cfg)              # follows the bed's device
    dev = bed.xtr.device
    generative = getattr(cfg, "objective", "classify") == "generate"
    spec = bed.spec
    ch, H, W = trunk.C, trunk.H, trunk.W
    aug = (spec is not None and (spec.name or "").startswith("cifar"))

    def _eval(t):
        if generative:
            return _eval_bpb(t, bed.xte, ch, H, W, cfg.n_bins)
        return probes.evaluate(t, bed.xte, bed.yte)

    base_eval = _eval(trunk)
    opt = laws.make_optimizer(trunk.parameters(), cfg.lr_head)
    row = {"config": asdict(cfg), "cell": cfg.cell, "device": str(dev),
           "name_key": bed.name_key, "bed": bed.name,
           "params": trunk.param_census(), "base_eval": base_eval, "traj": []}

    t0 = time.time()
    trunk.train()
    for step, (xb, yb) in enumerate(
            bed.batches(cfg.batch, cfg.steps, seed=cfg.seed + 7), 1):
        if aug:
            xb = _augment(xb, ch, H, W)
        if generative:
            tgt = _quantize(xb, cfg.n_bins).view(-1, ch, H, W)[:, 0]
            g = torch.rand(xb.shape[0], 1, H, W, device=dev) < 0.5
            masked = (xb.view(-1, ch, H, W) * (~g)).reshape(xb.shape[0], -1)
            loss = F.cross_entropy(trunk(masked).logits, tgt)
        else:
            loss = trunk(xb, labels=yb).loss
        lv = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trunk.parameters(), max(lv, 1.0))
        opt.step()
        if step % cfg.probe_every == 0 or step == 1:
            if generative:
                va = _eval(trunk)
                snap = {"step": step, "train_loss": lv, "ce": va["ce"],
                        "acc": 0.0, "bpb": va["bpb"], "train_ce": lv,
                        "train_acc": 0.0, "gap_ce": va["ce"] - lv,
                        "gap_acc": 0.0, "grad": {}}
            else:
                snap = _snapshot(step, lv, trunk, bed, None, {})
            snap["addr"] = _report(trunk, bed.xte[:512])
            row["traj"].append(snap)
            trunk.train()
        if step % cfg.log_every == 0 or step == 1:
            snap = row["traj"][-1] if row["traj"] else _snapshot(
                step, lv, trunk, bed, None, {})
            line = _fmt_progress(snap, base_eval, cfg)
            u = snap.get("addr") or {}
            if "read_amp_mean" in u:            # conv_tokens: additive read
                line += f" | read-amp {u['read_amp_mean']:.3f}"
            elif "kl_to_uniform" in u:          # addr_conv: filter usage
                line += f" | addr KL {u['kl_to_uniform']:.3f} ppl {u['usage_ppl']:.1f}"
            print(line, flush=True)
    row["seconds"] = round(time.time() - t0, 1)

    trunk.eval()
    row["final"] = _eval(trunk)
    key = "bpb" if generative else "ce"
    row["delta_vs_base_ce"] = base_eval[key] - row["final"][key]
    row["delta_vs_base_acc"] = row["final"]["acc"] - base_eval["acc"]
    row["addr_final"] = _report(trunk, bed.xte[:1024])
    return row


# --------------------------------------------------------------- the sweep
def sweep_conv(cfg: RunConfig | None = None, seeds=(0,), arms=None,
               bed: Bed | None = None, ledger: str | None = None) -> list[dict]:
    """The head-to-head: run every arm on one bed. For conv_tokens: `soft`
    (signed antipode) vs `mag` (sign-collapsed) isolates whether the ANTIPODE
    pays; vs `none`/`off` whether the read pays at all. For addr_conv: `soft`
    vs `none` (uniform=plain conv) with `learned`/`off` controls."""
    cfg = cfg or RunConfig(input_mode="addr_conv")
    if arms is None:
        arms = (CONV_TOKEN_ARMS if cfg.input_mode == "conv_tokens"
                else CONV_ARMS)
    dev = resolve_device(cfg.device)
    if bed is None:
        bed = build_bed(cfg.dataset, cfg.train_n, seed=cfg.seed, root=cfg.root,
                        synthetic=cfg.synthetic).to(dev)
    ledger = ledger or ledger_path("conv.jsonl")
    rows = []
    for seed in seeds:
        for m in arms:
            row = run_conv(replace(cfg, mode=m, seed=seed), bed, None)
            append_ledger(row, ledger)
            rows.append(row)
            f = row["final"]
            head = "bpb" if getattr(cfg, "objective", "") == "generate" else "acc"
            print(f"  -> {row['cell']}: {head}="
                  f"{f.get(head, f.get('acc')):.4f} "
                  f"dCE={row['delta_vs_base_ce']:+.4f}", flush=True)
    return rows
