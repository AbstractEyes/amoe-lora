"""amoe.diffusion.laws — the diffusion-line laws, encoded with their receipts.

Every rule was paid for on the r2 rented card; the receipts live in the
research record (https://huggingface.co/AbstractPhil/geolip-aleph-diffusion).

1. DTYPE LAW (supersedes amoe's fp32 default ON DIFFUSION TRUNKS) — adapters
   attach at the DECLARED trunk dtype, never a sniffed one: fp32 adapters on
   a bf16 trunk spin fp32 noise into the environment, and first-param
   sniffing can read a still-meta fp32 tensor (the R0b integration-gate
   catch, diffusion-pipe fork f4f1c46). Judged gauges stay fp32.
2. STRUCTURAL BANDING, NOT COMPARATIVE ROUTING — the certified partition of
   the sigma axis is cosine crossfade windows (positional, dense,
   differentiable; no selection event). Comparative routing on diffusion was
   falsified 2-seed: state+sigma dispatch keys found nothing to route on
   (exp007), and address-as-key died against the repeated-key null in both
   raw-flattened and M-hat-bottlenecked forms (exp014/exp015).
3. CONDITIONING LAW — structural/blob supervision pays where x0 recovery is
   LINEAR (flow: x0 = x_t − σ·v; ~125–200× the eps effect, 2-seed) and is
   inert on eps, which divides by a vanishing sqrt(alpha_bar) exactly in
   the supervised band (exp012 vs exp013). blob-on-eps therefore refuses
   unless forced. Operating point λ≈1 (3-point dose curve, exp013).
4. ZERO-INIT W **AND** B — the exp006 bias-leak law: output weight and bias
   both zero so a fresh adapter is EXACTLY inert (P-INIT bit-exact).
5. TOGGLE LAW — disabled adapters return x via CODE-PATH SKIP (`return x`,
   not a zero-mul), bit-exact at every dtype. Held post-train in every
   experiment of the line.
6. PURE ADAM, NEVER ADAMW — inherited program-wide (amoe.laws law 1).
"""
from __future__ import annotations

from ..laws import make_optimizer, pin_precision  # noqa: F401  (the constructors)

# The certified band constants (exp008; edges/xfade are LAW, not config):
N_BANDS = 3
BAND_EDGES = (0.35, 0.75)   # LOW [0,.35] | MID (.35,.75] | HIGH (.75,1]
XFADE = 0.06                # cosine crossfade half-width
BAND_ROLES = ("fidelity/detail (LOW noise)",
              "continuity/semantics (MID)",
              "diversity/structure (HIGH noise)")

BLOB_LAMBDA = 1.0           # the dose-curve operating point (exp013, 3 points)
FLOW_SHIFT = 2.5            # the trained sigma warp of the flow substrate

# The negatives record (why align()/dispatch is not a diffusion verb in 0.2):
ROUTING_NEGATIVES = (
    "exp007: A=4 bank with dense signed state+sigma dispatch ties/loses to a "
    "matched monolith under uniform pressure; sigma-band usage flat.",
    "exp014: text/address dispatch keys within 0.07% of the state-key control; "
    "raw-flattened [32,128] address KILLS routing (disease geometry).",
    "exp015: M-hat slot-bottlenecked address key — routing excess 2.5e-06 over "
    "the REPEATED-key null, match advantage ~0, both seeds. The frozen "
    "address-as-key form is CLOSED; the open form is TE+AMoE joint training.",
)
