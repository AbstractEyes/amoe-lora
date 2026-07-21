"""RelayPatch2D — the certified per-block residual relay for diffusion
token streams (verbatim port of the r2 bed, pod2/aleph_diffusion_core.py;
certified relay-favorable 3-for-3 substrates: Lune flow / SD15-core eps /
Anima DiT flow — exp001/exp006/exp004).

Differences from the text-side RelayPatchwork (amoe.core.adapter), each a
paid-for law:
  * trailing-dim generalization: forward works on any [..., d] stream;
  * zero-init output weight AND bias (exp006 bias-leak law) — a fresh
    relay is EXACTLY inert, P-INIT bit-exact;
  * `enabled=False` returns x via CODE-PATH SKIP (`return x`, `is x`) —
    the toggle law is bit-exact at every dtype;
  * dims-from-state-dict reconstruction (never trust constructor defaults).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...core.address import AlephAddress
from ...core.adapter import SquaredReLU


class RelayPatch2D(nn.Module):
    """forward(x[..., d]) = x + sigmoid(gate) * consume(m_hat(proj(x) slots))."""

    def __init__(self, d: int, n_slots: int = 16, K: int = 64,
                 tau: float = 0.1, hidden: int = 178):
        super().__init__()
        self.d, self.n_slots, self.hidden = d, n_slots, hidden
        self.proj = nn.Linear(d, n_slots * 4, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        self.addr = AlephAddress(K, 4, tau)
        self.consume = nn.Sequential(
            nn.Linear(n_slots * 4, hidden), SquaredReLU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, d))
        nn.init.zeros_(self.consume[-1].weight)   # zero-init out: weight
        nn.init.zeros_(self.consume[-1].bias)     # AND bias (exp006 law)
        self.gate = nn.Parameter(torch.tensor(-3.0))
        self.enabled = True                       # plain attr, not a buffer
        self.telemetry = False                    # read-only ratio stash
        self.last_delta_ratio = None

    def assert_zero_init(self):
        w, b = self.consume[-1].weight, self.consume[-1].bias
        assert w.abs().max().item() == 0.0, "out weight not zero-init"
        assert b.abs().max().item() == 0.0, "out BIAS not zero-init (leak law)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        lead = x.shape[:-1]
        slots = self.proj(x).view(*lead, self.n_slots, 4)
        feats = self.addr.m_hat(slots).reshape(*lead, self.n_slots * 4)
        delta = torch.sigmoid(self.gate) * self.consume(feats)
        if self.telemetry:
            with torch.no_grad():
                self.last_delta_ratio = (
                    delta.norm() / x.norm().clamp_min(1e-12)).item()
        return x + delta

    @classmethod
    def from_state_dict(cls, sd: dict) -> "RelayPatch2D":
        """Rebuild dims FROM the state dict, then strict-load. Pre-amoe
        fork saves lack the addr.home drift-gauge buffer — reconstruct
        it from the codebook, LOUDLY (the gauge starts here)."""
        sd = dict(sd)
        d = sd["proj.weight"].shape[1]
        n_slots = sd["proj.weight"].shape[0] // 4
        hidden = sd["consume.0.weight"].shape[0]
        K = sd["addr.codebook"].shape[0]
        if "addr.home" not in sd:
            import torch.nn.functional as F
            sd["addr.home"] = F.normalize(
                sd["addr.codebook"].detach().float(), dim=-1)
            print("amoe.diffusion: addr.home reconstructed from codebook "
                  "(pre-amoe save; drift gauge starts here)")
        m = cls(d, n_slots=n_slots, K=K, hidden=hidden)
        m.load_state_dict(sd, strict=True)
        return m


class BlockWithRelay(nn.Module):
    """Wrap one trunk block; the relay reads its hidden-states output.
    Tuple outputs pass through (hybrid-safe, as amoe.core BlockWithAdapter).
    Carries a `w_bands` slot for wrap-uniformity with multiband sites —
    the relay ignores it."""

    def __init__(self, block: nn.Module, relay: nn.Module):
        super().__init__()
        self.block = block
        self.mod = relay
        self.w_bands = None            # unused by relays; uniform wrap API

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if isinstance(out, tuple):
            return (self.mod(out[0]),) + out[1:]
        return self.mod(out)
