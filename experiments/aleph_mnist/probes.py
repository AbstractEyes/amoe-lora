"""Probes — what we actually look at when the trunk and the adapter move
together.

Five questions, five instruments:

1. IS A FRESH ANCHOR INERT?  `inertness` measures max|delta logit| with
   adapters on vs off at step 0. Law 3 says zero-init output heads make a
   fresh anchor inert; the campaign already documents that the LayerNorm
   bias path leaks (~+0.5 ppl on the reference substrate even at
   zero-init). This bed can put an exact number on the leak, because a
   784-in/10-out trunk has no vocabulary to hide it in.

2. HOW HARD IS IT FIRING, AND ON WHAT?  `escape_report` is the
   single-anchor analogue of the P5d blend-escape gauge. The shipped
   `amoe.diagnostics.diagnose` reads amplitude off the dispatch, which
   only exists with multiple anchors; with one anchor the honest
   equivalent is the residual delta ratio ||A(h)-h|| / ||h|| measured on
   domain vs neutral input. Ratio <= 1.5 is blend-regime escape, the same
   threshold (`amoe.laws.BLEND_ESCAPE_RATIO`).

3. DID IT DIFFERENTIATE?  `sign_code_report` counts unique committed sign
   codes and their mutual information with the digit label. This is the
   campaign law 3 gauge: differentiation must be STRUCTURAL — gradient-
   learned alphabets collapsed 1,594 -> 116 unique paths. If co-training
   collapses the codes here, it is the same disease at 1/1000 the cost.

4. WHO IS LEARNING?  `grad_norm_spread` (vitals) on {trunk, adapter}.

5. CAN YOU STILL TAKE IT OFF?  `toggle_report`. The toggle law only
   promises the MECHANISM is bit-exact; it promises nothing about what
   the trunk has come to depend on. Under co-training the trunk can grow
   into the adapter, and then toggling off is bit-exact AND catastrophic.
   That gap — mechanism intact, behaviour destroyed — is the sharpest
   thing this bed can show.
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F

from amoe import laws

from . import vitals
from .heads import set_adapters

MOD = (1 << 31) - 1          # keeps folded codes inside int64


# ------------------------------------------------------------ capturing
@contextlib.contextmanager
def capture(heads):
    """Collect (input, output) of each patch head for one forward."""
    store: dict[int, tuple] = {}
    hooks = [h.register_forward_hook(
        lambda m, i, o, k=k: store.__setitem__(k, (i[0].detach(),
                                                   o.detach())))
        for k, h in enumerate(heads)]
    try:
        yield store
    finally:
        for hk in hooks:
            hk.remove()


# ------------------------------------------------------------ accuracy
@torch.no_grad()
def evaluate(trunk, x, y, batch: int = 1024) -> dict:
    trunk.eval()
    tot_ce, correct, n = 0.0, 0, 0
    for i in range(0, x.shape[0], batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        logits = trunk(xb).logits
        tot_ce += float(F.cross_entropy(logits, yb, reduction="sum"))
        correct += int((logits.argmax(-1) == yb).sum())
        n += yb.numel()
    return {"ce": tot_ce / n, "acc": correct / n, "n": n}


# ----------------------------------------------------------- inertness
@torch.no_grad()
def inertness(trunk, wrappers, x) -> dict:
    """max|delta logit| between adapters-on and adapters-off. At step 0
    this is the zero-init leak; run it any time to size the adapter's
    total influence on the decision surface."""
    trunk.eval()
    set_adapters(wrappers, False)
    off = trunk(x).logits.clone()
    set_adapters(wrappers, True)
    on = trunk(x).logits
    d = (on - off).abs()
    return {"max_abs_dlogit": float(d.max()),
            "mean_abs_dlogit": float(d.mean()),
            "argmax_flip_frac": float(
                (on.argmax(-1) != off.argmax(-1)).float().mean())}


# ------------------------------------------------------ delta amplitude
@torch.no_grad()
def delta_ratio(trunk, heads, x) -> list[float]:
    """Per-block ||A(h) - h|| / ||h||: how much of the residual stream
    the adapter is rewriting. The single-anchor amplitude gauge."""
    trunk.eval()
    with capture(heads) as store:
        trunk(x)
    out = []
    for k in range(len(heads)):
        inp, o = store[k]
        num = (o - inp).flatten(0, -2).norm(dim=-1)
        den = inp.flatten(0, -2).norm(dim=-1).clamp_min(1e-9)
        out.append(float((num / den).mean()))
    return out


@torch.no_grad()
def escape_report(trunk, heads, x_domain, neutral: dict) -> dict:
    """Domain vs neutral amplitude. ratio <= BLEND_ESCAPE_RATIO (1.5)
    means the adapter fires as hard on structureless input as on real
    digits — blend-regime escape, the failure the campaign's diagnostics
    exist to catch."""
    dom = delta_ratio(trunk, heads, x_domain)
    dom_mean = sum(dom) / len(dom)
    rep = {"domain_per_block": [round(v, 5) for v in dom],
           "domain_mean": dom_mean, "neutral": {}, "ratio": {},
           "escaped": []}
    for name, xn in neutral.items():
        neu = delta_ratio(trunk, heads, xn)
        neu_mean = sum(neu) / len(neu)
        r = dom_mean / max(neu_mean, 1e-9)
        rep["neutral"][name] = neu_mean
        rep["ratio"][name] = round(r, 3)
        if r <= laws.BLEND_ESCAPE_RATIO:
            rep["escaped"].append(name)
    return rep


# ----------------------------------------------------------- sign codes
def _fold(codes: torch.Tensor, radix: int) -> torch.Tensor:
    """Fold a (..., S) integer code vector into one id per row, modular so
    int64 never overflows. Collisions are negligible at eval batch sizes
    against a 2^31 space."""
    out = torch.zeros(codes.shape[:-1], dtype=torch.int64,
                      device=codes.device)
    for s in range(codes.shape[-1]):
        out = (out * radix + codes[..., s].to(torch.int64)) % MOD
    return out


def _mutual_information(a: torch.Tensor, b: torch.Tensor) -> float:
    """MI(a; b) in nats with the Miller-Madow bias correction. Raw MI on
    a 128-symbol code and a few thousand samples is upward-biased enough
    to invent structure that is not there."""
    a = a.reshape(-1).cpu()
    b = b.reshape(-1).cpu()
    ua, ia = torch.unique(a, return_inverse=True)
    ub, ib = torch.unique(b, return_inverse=True)
    n = a.numel()
    joint = torch.zeros(ua.numel(), ub.numel())
    joint.index_put_((ia, ib), torch.ones(n), accumulate=True)
    p = joint / n
    pa, pb = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
    nz = p > 0
    mi = float((p[nz] * (p[nz].log() - (pa @ pb)[nz].log())).sum())
    bias = (int(nz.sum()) - ua.numel() - ub.numel() + 1) / (2.0 * n)
    return max(0.0, mi - bias)


@torch.no_grad()
def sign_code_report(trunk, heads, x, y) -> dict:
    """Unique committed codes per block, the composed whole-organism path
    count, and the label information the most selective slot carries.
    H(digit) = ln 10 = 2.303 nats is the ceiling."""
    trunk.eval()
    with capture(heads) as store:
        trunk(x)
    K2 = 2 * heads[0].addr.K
    per_block, block_ids, slot_mi = [], [], []
    for k, h in enumerate(heads):
        inp, _ = store[k]
        codes = h.sign_code(inp)                     # (B, T, S)
        codes = codes.reshape(codes.shape[0], -1)    # (B, T*S)
        fid = _fold(codes, K2)
        block_ids.append(fid)
        per_block.append(vitals.path_diversity(fid))
        mis = [_mutual_information(codes[:, s], y)
               for s in range(codes.shape[1])]
        slot_mi.append({"max": max(mis), "mean": sum(mis) / len(mis)})
    whole = _fold(torch.stack(block_ids, dim=-1), MOD % (1 << 20))
    return {"per_block_unique": [d["unique_raw"] for d in per_block],
            "organism_unique": vitals.path_diversity(whole)["unique_raw"],
            "n_samples": int(x.shape[0]),
            "label_entropy_nats": math.log(10.0),
            "slot_mi_nats": slot_mi}


# --------------------------------------------------------------- toggle
@torch.no_grad()
def toggle_report(trunk, wrappers, x, y) -> dict:
    """Accuracy and CE with the adapters on vs off.

    `damage` is the co-training tax on detachability: how much the trunk
    has come to DEPEND on a module the API says you can remove. On a
    frozen trunk it must be <= 0 (off == the original model). Large
    positive damage means the artifact is no longer an adapter — it is
    half the model."""
    set_adapters(wrappers, True)
    on = evaluate(trunk, x, y)
    set_adapters(wrappers, False)
    off = evaluate(trunk, x, y)
    set_adapters(wrappers, True)
    return {"on": on, "off": off,
            "damage_acc": on["acc"] - off["acc"],
            "damage_ce": off["ce"] - on["ce"]}


# --------------------------------------------------------------- vitals
@torch.no_grad()
def vitals_report(trunk, heads, x=None) -> dict:
    """drift vs BINDING, pentachoron CV (logged, not gated — D=4 is the
    volatile regime by design), gate mean vs the 0.012-0.03 band, and
    axis aliveness if an input batch is supplied."""
    gates = torch.stack([torch.sigmoid(h.gate.detach()) for h in heads])
    rep = {"gate": vitals.gate_stats(gates),
           "gate_per_block": [round(float(g), 5) for g in gates],
           "drift": [], "cv": []}
    for h in heads:
        d = vitals.anchor_drift(h.addr.codebook, h.addr.home)
        rep["drift"].append({k: round(v, 5) for k, v in d.items()})
        rep["cv"].append(round(vitals.pentachoron_cv(h.addr.codebook), 4))
    rep["drift_mean"] = sum(d["mean"] for d in rep["drift"]) / len(heads)
    rep["binding_target"] = vitals.BINDING
    if x is not None:
        trunk.eval()
        with capture(heads) as store:
            trunk(x)
        alive = []
        for k, h in enumerate(heads):
            inp, _ = store[k]
            w = vitals.oriented_weights(h.slots_of(inp), h.addr.codebook,
                                        h.spec.tau)
            alive.append(vitals.axis_aliveness(w))
        rep["aliveness"] = alive
    return rep
