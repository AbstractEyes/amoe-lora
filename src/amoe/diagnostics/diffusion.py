"""Diffusion diagnostics — the campaign's honesty instruments as API.

lesion_report   : the exp008/exp010 band-lesion battery shape — run YOUR
                  gauge under each single-band lesion and read whether
                  specialization is surgical (own-band damage >> cross).
foreground_gauge: the exp012 role-aligned payer instrument — HIGH-band
                  foreground-LP-x0 error (the gauge that showed multiband
                  beating the matched monolith ~10% when every
                  aggregate-eps comparison was blind to it).
gate_stats      : gate health (sigmoid means; the substrate-dependent
                  dynamics finding — core grew, lune shrank).
diagnose        : one-call summary with the loud flags.
"""
from __future__ import annotations

import torch

from ..diffusion.laws import N_BANDS
from ..diffusion.train.objectives import blob_lp_err


def lesion_report(handle, eval_fn) -> dict:
    """eval_fn() -> float, evaluated under all_on and each single-band
    lesion. Multiband anchors only. Returns {'all_on': x,
    'lesion_band0': ..., 'surgical': bool} where surgical means every
    lesion moved the gauge (the exp008 signature reads per-band gauges;
    with a single gauge this reports the monotone lesion profile the
    exp010 battery certified)."""
    out = {"all_on": float(eval_fn())}
    for b in range(N_BANDS):
        with handle.lesion_band(b):
            out[f"lesion_band{b}"] = float(eval_fn())
    deltas = [out[f"lesion_band{b}"] - out["all_on"] for b in range(N_BANDS)]
    out["lesion_deltas"] = [round(d, 6) for d in deltas]
    return out


def foreground_gauge(x0_hat: torch.Tensor, x0: torch.Tensor,
                     blob: torch.Tensor) -> float:
    """Mean foreground-LP-x0 error (lower is better). Judge in fp32 (law)."""
    return float(blob_lp_err(x0_hat.float(), x0.float(),
                             blob.float()).mean())


@torch.no_grad()
def gate_stats(handle) -> dict:
    return handle.gates()


@torch.no_grad()
def diagnose(handle) -> dict:
    """One-call health readout: gates, amplitude telemetry (if armed),
    and the standing scope note on gate bands."""
    rep = {"kind": handle.kind, "n_sites": len(handle.names),
           **gate_stats(handle)}
    amp = handle.amplitude()
    if amp:
        vals = list(amp.values())
        rep["delta_ratio_mean"] = round(sum(vals) / len(vals), 6)
    rep["note"] = ("gate dynamics are SUBSTRATE-DEPENDENT on diffusion "
                   "(exp006: core grew 0.047->0.070; exp001: lune shrank) "
                   "— read direction against your own frozen baseline, "
                   "not against the LM band")
    return rep
