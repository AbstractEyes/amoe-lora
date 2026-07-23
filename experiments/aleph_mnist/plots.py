"""Figures. Every panel answers one preregistered question; nothing here
is decoration.

  fig_dial          Delta(arm - control) CE vs trainable blocks. THE
                    RESULT: where (if anywhere) the curve crosses zero is
                    the boundary the campaign never measured.
  fig_gates         gate mean trajectories with the 0.012-0.03 candidate
                    band shaded. A 7th architecture for a live invariant.
  fig_drift         codebook drift vs the 0.29154 binding constant.
  fig_toggle        accuracy with adapters on vs off — the detachability
                    tax that co-training buys.
  fig_escape        domain/neutral amplitude vs the 1.5 escape threshold.
  fig_democracy     trunk vs adapter gradient norms over training.

matplotlib only; no seaborn, no style sheets, one chart per figure.
"""
from __future__ import annotations

import json
from collections import defaultdict

import matplotlib.pyplot as plt

from .vitals import BINDING, GATE_BAND

ESCAPE = 1.5


def load_ledger(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _key(row) -> tuple:
    c = row["config"]
    return c["mode"], c["trainable_blocks"], c["seed"], c.get("tag", "")


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fig_dial(rows, control: str = "none", metric: str = "ce", ax=None):
    """The headline. Negative delta = the arm beats the control."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    by = defaultdict(dict)
    for r in rows:
        m, n, s, tag = _key(r)
        if tag:
            continue
        by[(n, s)][m] = r["final"][metric]
    modes = sorted({m for v in by.values() for m in v} - {control})
    dials = sorted({n for n, _ in by})
    for m in modes:
        ys = [_mean([v[m] - v[control] for (n_, _), v in by.items()
                     if n_ == n and m in v and control in v])
              for n in dials]
        ax.plot(dials, ys, marker="o", label=f"{m} - {control}")
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.set_xlabel("trainable trunk blocks  (0 = frozen substrate)")
    ax.set_ylabel(f"delta {metric}   (negative = arm wins)")
    ax.set_title("The co-training dial")
    ax.set_xticks(dials)
    ax.legend()
    return ax


def fig_gates(rows, ax=None):
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.axhspan(*GATE_BAND, color="tab:green", alpha=0.15,
               label=f"candidate band {GATE_BAND[0]}-{GATE_BAND[1]}")
    for r in rows:
        m, n, s, tag = _key(r)
        traj = [(t["step"], t.get("gate_mean")) for t in r["traj"]
                if t.get("gate_mean") is not None]
        if not traj:
            continue
        ax.plot([t for t, _ in traj], [g for _, g in traj],
                alpha=0.8, label=f"{m} dial{n}" if s == 0 else None)
    ax.set_xlabel("step")
    ax.set_ylabel("mean sigmoid(gate)")
    ax.set_title("Gate trajectories vs the live invariant band")
    ax.legend(fontsize=7)
    return ax


def fig_drift(rows, ax=None):
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.axhline(BINDING, color="tab:red", ls="--",
               label=f"binding constant {BINDING}")
    for r in rows:
        m, n, s, tag = _key(r)
        traj = [(t["step"], t.get("drift_mean")) for t in r["traj"]
                if t.get("drift_mean") is not None]
        if not traj:
            continue
        ax.plot([t for t, _ in traj], [d for _, d in traj],
                alpha=0.8, label=f"{m} dial{n}" if s == 0 else None)
    ax.set_xlabel("step")
    ax.set_ylabel("mean codebook drift (rad)")
    ax.set_title("Does the book move? (basin-set-at-init says barely)")
    ax.legend(fontsize=7)
    return ax


def fig_toggle(rows, ax=None):
    """Detachability tax: how much accuracy leaves when you switch the
    adapter off, by dial position."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    by = defaultdict(list)
    for r in rows:
        m, n, s, tag = _key(r)
        if "toggle" not in r or tag:
            continue
        by[m].append((n, r["toggle"]["damage_acc"]))
    for m, pts in sorted(by.items()):
        d = defaultdict(list)
        for n, v in pts:
            d[n].append(v)
        xs = sorted(d)
        ax.plot(xs, [_mean(d[x]) for x in xs], marker="s", label=m)
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.set_xlabel("trainable trunk blocks")
    ax.set_ylabel("acc(on) - acc(off)")
    ax.set_title("What the trunk came to depend on")
    ax.legend()
    return ax


def fig_escape(rows, neutral: str = "permuted", ax=None):
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    by = defaultdict(list)
    for r in rows:
        m, n, s, tag = _key(r)
        rep = r.get("escape")
        if not rep or neutral not in rep["ratio"] or tag:
            continue
        by[m].append((n, rep["ratio"][neutral]))
    for m, pts in sorted(by.items()):
        d = defaultdict(list)
        for n, v in pts:
            d[n].append(v)
        xs = sorted(d)
        ax.plot(xs, [_mean(d[x]) for x in xs], marker="^", label=m)
    ax.axhline(ESCAPE, color="tab:red", ls="--",
               label=f"escape threshold {ESCAPE}")
    ax.set_xlabel("trainable trunk blocks")
    ax.set_ylabel(f"amplitude ratio  domain / {neutral}")
    ax.set_title("Specialize regime or blend regime")
    ax.legend()
    return ax


def fig_democracy(rows, ax=None):
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    for r in rows:
        m, n, s, tag = _key(r)
        if s != 0 or n == 0:
            continue
        steps = [t["step"] for t in r["traj"]]
        for grp, style in (("trunk", "-"), ("adapter", "--")):
            ys = [t["grad"]["norms"].get(grp) for t in r["traj"]]
            if not any(ys):
                continue
            ax.plot(steps, ys, style, alpha=0.8, label=f"{m} d{n} {grp}")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("grad norm")
    ax.set_title("Who is actually learning")
    ax.legend(fontsize=6, ncol=2)
    return ax


def figure_set(rows, path: str | None = None):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig_dial(rows, ax=axes[0][0])
    fig_toggle(rows, ax=axes[0][1])
    fig_escape(rows, ax=axes[0][2])
    fig_gates(rows, ax=axes[1][0])
    fig_drift(rows, ax=axes[1][1])
    fig_democracy(rows, ax=axes[1][2])
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def verdict_table(rows) -> str:
    """One line per cell, ledger order — the thing to paste into a
    session digest."""
    head = (f"{'cell':28} {'ce':>7} {'acc':>7} {'dCE':>8} {'gate':>7} "
            f"{'drift':>7} {'tog_acc':>8} {'esc':>6} {'codes':>7}")
    out = [head, "-" * len(head)]
    for r in rows:
        v = r.get("vitals", {})
        esc = r.get("escape", {}).get("ratio", {}).get("permuted")
        sc = r.get("sign_codes", {}).get("organism_unique")
        out.append(
            f"{r['cell']:28} {r['final']['ce']:7.4f} "
            f"{r['final']['acc']:7.4f} {r['delta_vs_base_ce']:+8.4f} "
            f"{v.get('gate', {}).get('mean', float('nan')):7.4f} "
            f"{v.get('drift_mean', float('nan')):7.4f} "
            f"{r.get('toggle', {}).get('damage_acc', float('nan')):8.4f} "
            f"{esc if esc is not None else float('nan'):6.2f} "
            f"{sc if sc is not None else -1:7d}")
    return "\n".join(out)
