"""align() — a grounded negative on diffusion, kept for API symmetry.

The text line's align() trains dispatch keys over frozen anchors. On
diffusion that entire family was falsified at 2 seeds — see
amoe.diffusion.laws.ROUTING_NEGATIVES (exp007: state+sigma keys find
nothing to route on; exp014/exp015: address-as-key dead against the
repeated-key null in both raw and M-hat-bottlenecked forms).

The certified replacement for comparative routing on the sigma axis is
STRUCTURAL banding (adapter='multiband3' in train(); exp008 lesions
surgical 3/3 both seeds). The open research form — TE+AMoE joint
training, where the text encoder adapter learns WITH the dispatch — is
queued science (history/open_questions.md), not shippable API.
"""
from __future__ import annotations

from ..laws import ROUTING_NEGATIVES


def align(*args, **kwargs):
    raise NotImplementedError(
        "amoe.diffusion.align: comparative dispatch on diffusion is a "
        "2-seed falsified line — there is nothing honest to align here. "
        "Use train(adapter='multiband3') for the certified structural "
        "banding, or follow the open TE+AMoE joint-training question. "
        "The record:\n  - " + "\n  - ".join(ROUTING_NEGATIVES))
