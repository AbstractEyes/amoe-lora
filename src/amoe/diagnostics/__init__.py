"""Diagnostics — the campaign's honesty instruments as first-class API.

diagnose(handle, ...) is the one-call health check:
- blend-escape (P5d gauge): per-anchor mean |w/z| amplitude on neutral
  vs on-domain text; ratio <= 1.5 warns (the anchor fires as hard on
  unrelated prose as on its own domain — expect a perplexity tax and
  cross-task trampling; the research line measured up to 4.7x tax and
  chain-of-thought destruction from escaped anchors).
- usage shares (argmax-shadow proxy).
"""
from __future__ import annotations

import warnings

import torch

from .. import laws

NEUTRAL_DEFAULT = [
    "The museum's new wing opened after years of renovation, drawing "
    "visitors from across the region to its glass-roofed atrium.",
    "She packed the last of the boxes and stood in the empty kitchen, "
    "listening to the rain against the window.",
    "The committee postponed its decision until the spring session, "
    "citing the need for further public consultation.",
    "Migrating birds follow coastlines and river valleys, resting in "
    "wetlands that have shrunk decade by decade.",
    "Volunteers repainted the community hall over the weekend and "
    "replaced the broken flooring near the stage.",
    "Early photographs of the valley show orchards where the highway "
    "now runs, and a station long since demolished.",
]


class BlendEscapeWarning(UserWarning):
    pass


@torch.no_grad()
def _amplitude_pass(handle, tok, texts, device):
    handle.telemetry(True)
    for t in texts:
        ids = tok(t, return_tensors="pt",
                  add_special_tokens=False).input_ids.to(device)
        handle.model(input_ids=ids)
    amp = handle.amplitude()
    handle.telemetry(False)
    return amp


def diagnose(handle, tokenizer, domain_texts=None, neutral_texts=None,
             device="cuda") -> dict:
    neutral = neutral_texts or NEUTRAL_DEFAULT
    amp_n = _amplitude_pass(handle, tokenizer, neutral, device)
    report = {"amplitude_neutral": amp_n}
    if domain_texts:
        amp_d = _amplitude_pass(handle, tokenizer, domain_texts, device)
        ratios = {k: round(amp_d[k] / max(amp_n.get(k, 0.0), 1e-9), 2)
                  for k in amp_d}
        report["amplitude_domain"] = amp_d
        report["on_over_neutral_ratio"] = ratios
        escaped = [k for k, r in ratios.items()
                   if r <= laws.BLEND_ESCAPE_RATIO]
        report["blend_escape"] = escaped
        for k in escaped:
            warnings.warn(
                f"anchor '{k}' fires at near on-domain amplitude on "
                f"neutral text (ratio {ratios[k]} <= "
                f"{laws.BLEND_ESCAPE_RATIO}): blend-regime escape — "
                "expect perplexity tax and cross-task interference",
                BlendEscapeWarning, stacklevel=2)
    report["usage"] = handle.usage()
    return report
