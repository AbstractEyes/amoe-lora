"""vitals — the campaign's shared readouts, vendored for this bed.

Every function here is a READOUT: no gradients, no losses, no targets.
CV is logged, never optimized. The constants are the campaign's, and
they are the reason this bed is worth running at all — a 4-block MNIST
trunk can be measured against numbers that were paid for on GPT-2,
Qwen and SD15 substrates.

  BINDING 0.29154 rad   binding/separation constant — judge an address
                        by drift toward it, NEVER by recon cosine
  GATE_BAND 0.012-0.03  live invariant candidate across 6 architectures
                        and 2 optimizers; this bed is a 7th data point
  path hash             Knuth multiplicative, HIGH bits. The low-16
                        variant produced a retracted ~1,500-path ceiling
                        and must never be reintroduced.

D=4 CAVEAT (aleph_core): a D=4 codebook extrapolates pentachoron
CV ~0.9, far above the 0.13-0.30 band. That is the VOLATILE regime, on
purpose — do not read an out-of-band CV here as a failure. It is logged
as a trajectory (does co-training move it?), not as a gate.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BINDING = 0.29154            # radians
CV_BAND = (0.13, 0.30)       # CM CV band (D=32-112; NOT applicable at D=4)
GATE_BAND = (0.012, 0.03)    # live invariant candidate
KNUTH32 = 2654435761


# ----------------------------------------------------------------- drift
@torch.no_grad()
def anchor_drift(current: torch.Tensor, init: torch.Tensor,
                 tol: float = 0.05) -> dict:
    """Geodesic drift (radians) of each row of `current` from `init`, both
    row-normalized, plus the fraction of rows sitting within +/-tol of
    BINDING. Reference bands: champion books inside an LM trunk drift
    0.28-0.32, random books 0.32-0.44."""
    a = F.normalize(current.float(), dim=-1)
    b = F.normalize(init.float(), dim=-1)
    cos = (a * b).sum(-1).clamp(-1.0, 1.0)
    drift = torch.arccos(cos)
    frac = ((drift - BINDING).abs() <= tol).float().mean()
    return {"mean": drift.mean().item(), "std": drift.std().item(),
            "max": drift.max().item(), "binding_fraction": frac.item()}


# -------------------------------------------------------------------- cv
@torch.no_grad()
def _pentachoron_volumes(pts: torch.Tensor) -> torch.Tensor:
    """Batched Cayley-Menger 4-simplex volumes. pts (B,5,D) -> (B,).
    ONE float64 det over all samples; vol^2 = -det(CM)/9216 for n=4.
    fp64 is not optional: fp32 dets lose up to ~4% on near-degenerate
    pentachora."""
    B = pts.shape[0]
    d2 = torch.cdist(pts.double(), pts.double()).pow(2)
    cm = torch.ones(B, 6, 6, dtype=torch.float64, device=pts.device)
    cm[:, 0, 0] = 0.0
    cm[:, 1:, 1:] = d2
    det = torch.linalg.det(cm)
    return (-det / 9216.0).clamp_min(0.0).sqrt().float()


@torch.no_grad()
def pentachoron_cv(rows: torch.Tensor, n_samples: int = 200,
                   seed: int = 0) -> float:
    """CV (std/mean) of CM 4-volumes over random 5-row subsets."""
    x = F.normalize(rows.float(), dim=-1).cpu()
    n = x.shape[0]
    if n < 5:
        raise ValueError(f"pentachoron_cv needs >=5 rows, got {n}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.stack([torch.randperm(n, generator=g)[:5]
                       for _ in range(n_samples)])
    v = _pentachoron_volumes(x[idx])
    return (v.std() / v.mean().clamp_min(1e-12)).item()


# ------------------------------------------------------------- aliveness
@torch.no_grad()
def axis_aliveness(oriented: torch.Tensor, alive_thresh: float = 1e-3) -> dict:
    """`oriented`: (..., 2K) nonnegative oriented-address rows summing to 1
    on the last dim. Returns axes alive, usage perplexity (the hppl
    analogue; healthy hosted reference 125-126 of 128) and a collapse
    flag. Materializing the explicit 2K softmax is FOR MEASUREMENT ONLY —
    the training path never builds it (that was the memory wall)."""
    w = oriented.reshape(-1, oriented.shape[-1]).float()
    usage = w.mean(0)
    usage = usage / usage.sum().clamp_min(1e-12)
    alive = int((usage > alive_thresh * (1.0 / usage.numel())).sum())
    ent = -(usage.clamp_min(1e-12) * usage.clamp_min(1e-12).log()).sum()
    ppl = float(ent.exp())
    return {"axes_total": int(usage.numel()), "axes_alive": alive,
            "usage_ppl": ppl, "collapsed": bool(ppl < 0.05 * usage.numel())}


@torch.no_grad()
def oriented_weights(slots: torch.Tensor, codebook: torch.Tensor,
                     tau: float = 0.1) -> torch.Tensor:
    """Explicit softmax over the 2K oriented half-axes [+A; -A], for the
    aliveness readout only. Closed form m_hat never builds this."""
    A = F.normalize(codebook.float(), dim=-1)
    u = (F.normalize(slots.float(), dim=-1) @ A.T) / tau
    return torch.softmax(torch.cat([u, -u], dim=-1), dim=-1)


# ------------------------------------------------------------------ gate
@torch.no_grad()
def gate_stats(gates: torch.Tensor) -> dict:
    """Post-sigmoid gate values vs the 0.012-0.03 candidate band.
    Read-only — the band is never a target."""
    g = gates.float().flatten()
    m = g.mean().item()
    return {"mean": m, "std": g.std().item() if g.numel() > 1 else 0.0,
            "min": g.min().item(), "max": g.max().item(),
            "in_band": bool(GATE_BAND[0] <= m <= GATE_BAND[1])}


# ----------------------------------------------------------------- paths
@torch.no_grad()
def path_diversity(ids: torch.Tensor) -> dict:
    """Unique-path counting with the FIXED high-bits hash:
    ((ids * 2654435761) % 2^32) >> 16. Knuth needs the HIGH bits."""
    x = ids.reshape(-1).to(torch.int64)
    hashed = ((x * KNUTH32) % (1 << 32)) >> 16
    return {"n": int(x.numel()),
            "unique_raw": int(torch.unique(x).numel()),
            "unique_hashed": int(torch.unique(hashed).numel())}


@torch.no_grad()
def compose_path_ids(stage_indices: list[torch.Tensor], radix: int) -> torch.Tensor:
    """Compose per-stage discrete indices into one positional base-`radix`
    path id — construction, not hashing."""
    out = torch.zeros_like(stage_indices[0], dtype=torch.int64)
    for s in stage_indices:
        out = out * radix + s.to(torch.int64)
    return out


# ------------------------------------------------------- grad democracy
@torch.no_grad()
def grad_norm_spread(groups: dict[str, list[torch.nn.Parameter]]) -> dict:
    """Gradient-democracy monitor. Reference: unequalized heterogeneous
    towers spread ~20 orders of magnitude (a member died at 2.25e-21).
    On this bed the two groups are `trunk` and `adapter` — the co-training
    question in one number."""
    norms = {}
    for name, params in groups.items():
        gs = [p.grad for p in params if p.grad is not None]
        norms[name] = float(torch.sqrt(
            sum((g.float() ** 2).sum() for g in gs)).item()) if gs else 0.0
    vals = [v for v in norms.values() if v > 0]
    spread = (math.log10(max(vals)) - math.log10(min(vals))) \
        if len(vals) >= 2 else 0.0
    return {"norms": norms, "spread_orders": spread,
            "dead": [k for k, v in norms.items() if v == 0.0]}


# ----------------------------------------------------------------- smoke
def _smoke() -> None:
    """Shapes/parse only — never a training run."""
    g = torch.Generator().manual_seed(0)
    K, D = 64, 4
    init = F.normalize(torch.randn(K, D, generator=g), dim=-1)
    cur = F.normalize(init + 0.29 * torch.randn(K, D, generator=g), dim=-1)
    print("drift:", anchor_drift(cur, init))
    print("cv(D=4, volatile by design):", round(pentachoron_cv(cur), 4))
    print("aliveness:", axis_aliveness(
        oriented_weights(torch.randn(8, 16, D, generator=g), cur)))
    print("gates:", gate_stats(torch.sigmoid(torch.full((4,), -3.0))))
    ids = compose_path_ids(
        [torch.randint(0, 16, (4096,), generator=g) for _ in range(4)], 16)
    print("paths:", path_diversity(ids))
    lin = torch.nn.Linear(8, 8)
    lin(torch.randn(4, 8)).sum().backward()
    print("democracy:", grad_norm_spread({"a": list(lin.parameters())}))
    print("OK - vitals smoke passed")


if __name__ == "__main__":
    _smoke()
