"""Where results land — and nowhere else.

The old default was `os.path.join("experiments", "results")`, relative to a
cwd that only exists inside a source checkout. After a `pip install` there
is no `experiments/` directory, so an installed CLI silently created
`./experiments/results/` wherever it happened to be run.

`results_root()` resolves in this order:
    ALEPH_RESULTS -> AMOE_RESULTS -> "results"
so an installed run writes to `./results/`, and a source checkout that
wants the historical location just exports
`ALEPH_RESULTS=experiments/results`.
"""
from __future__ import annotations

import json
import os

from amoe.io.checkpoint import AnchorCheckpoint


def results_root() -> str:
    return (os.environ.get("ALEPH_RESULTS")
            or os.environ.get("AMOE_RESULTS")
            or "results")


def ledger_path(name: str = "ledger.jsonl", path: str | None = None) -> str:
    return path or os.path.join(results_root(), name)


def append_ledger(row: dict, path: str | None = None,
                  name: str = "ledger.jsonl") -> str:
    """Append one cell to the JSONL ledger. The ledger IS the evidence —
    figures are derived from it, never hand-transcribed."""
    path = ledger_path(name, path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    clean = {k: v for k, v in row.items() if k != "_artifact"}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean) + "\n")
    return path


def save_anchor(row: dict, path: str) -> str:
    """Ship the trained heads as a real amoe.anchor v1 checkpoint.

    Writes to `path` and returns the checkpoint's CONTENT HASH (not the
    path) — the same contract as `amoe.io.checkpoint.AnchorCheckpoint.save`.

    `base_model_id` must match the live trunk's `_name_or_path` or
    `amoe.attach(strict=True)` refuses. `build_model` derives that name per
    dataset, and this writes the same string — so the round-trip holds for
    fashion and cifar, not just mnist."""
    art = row.get("_artifact")
    if art is None:
        raise ValueError("this row has no adapter to save (mode='off'?)")
    cfg = row["config"]
    key = row.get("name_key") or cfg["dataset"]
    meta = {"name": f"{cfg['dataset']}-{cfg['mode']}-d{cfg['d']}"
                    f"-dial{cfg['trainable_blocks']}",
            "base_model_id": f"tiny-{key}-4block",
            "d": cfg["d"], "n_layers": cfg["n_blocks"],
            "sites": art["sites"], "seed": cfg["seed"], "precision": "fp32",
            "address_mode": cfg["mode"],      # NOT a stock anchor if != soft
            "codebook_init": cfg["codebook_init"],
            "input_mode": cfg.get("input_mode", "linear"),
            "recipe": {"optimizer": "adam", "lr": cfg["lr_head"],
                       "weight_decay": 0.0, "steps": cfg["steps"],
                       "batch": cfg["batch"],
                       "trainable_trunk_blocks": cfg["trainable_blocks"]},
            "experiment": "aleph_mnist co-training dial"}
    return AnchorCheckpoint(art["state"], meta).save(path)
