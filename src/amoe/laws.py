"""amoe.laws — the house laws, encoded as the only constructors.

Every rule here was paid for with GPU hours in the geolip-aleph
campaigns; see the research record for the receipts
(https://huggingface.co/AbstractPhil/geolip-aleph-qwen-3.5-0.8b-instruct).

1. PURE ADAM, NEVER ADAMW — weight decay on these adapters destroys
   the zero-init inertness contract. `make_optimizer` is the only
   optimizer constructor in the package.
2. NO SELECTORS — the dispatch is dense signed sinh/cosh with an
   all-anchor damped denominator. No top-k, no load-balancing loss, no
   auxiliary independence loss. Masking never renormalizes.
3. ZERO-INIT OUTPUT HEADS — a fresh anchor must be (near-)inert at
   attach. (Note: the LayerNorm bias path leaks ~+0.5 ppl on the
   reference substrate even at zero-init; measured, documented.)
4. TOGGLE LAW — all anchors disabled must equal the base model
   BIT-EXACT (fp32). `testing.invariants` asserts it.
5. FP32 DEFAULT, TF32 OFF — bit-exact guarantees are dtype-conditional
   under bf16 (documented fallback).
"""
from __future__ import annotations

import torch


def make_optimizer(params, lr: float = 1e-3) -> torch.optim.Adam:
    """The only optimizer: pure Adam, weight_decay=0.0 (law 1)."""
    return torch.optim.Adam(params, lr=lr, weight_decay=0.0)


def pin_precision() -> None:
    """fp32 discipline: TF32 off on matmul and cudnn (law 5)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


PPL_TAX_FLAG = 2.0          # ppl@512 delta beyond which an anchor is flagged
BLEND_ESCAPE_RATIO = 1.5    # on-domain/neutral amplitude at or below = escape
DAMPED_TARGET_RATIO = 3.0   # healthy specialize-regime damping target
