"""Training-data guards — the campaign's honesty instruments.

check_dataset: the question-space / overlap guard (research line
exp013b): an answer-diversity check alone does NOT protect against
question-space exhaustion — an expert trained on more draws than the
space holds distinct questions memorizes the space and posts fake
generalization.
"""
from __future__ import annotations

import warnings
from typing import Callable, Sequence


class MemorizationRiskWarning(UserWarning):
    pass


def check_dataset(train_rows: Sequence[dict],
                  eval_rows: Sequence[dict] | None = None,
                  key: Callable[[dict], str] | None = None) -> dict:
    key = key or (lambda r: str(r.get("messages", r.get("ids"))))
    train_keys = [key(r) for r in train_rows]
    space = len(set(train_keys))
    report = {"train_draws": len(train_keys),
              "distinct_questions": space,
              "space_over_draws": round(space / max(len(train_keys), 1), 3)}
    if space < len(train_keys):
        warnings.warn(
            f"question space ({space}) is smaller than the training "
            f"draw ({len(train_keys)}): the adapter can memorize the "
            "space and any in-distribution gain may be fake "
            "generalization. Widen the space or shrink the draw "
            "(research line: sequences 480 < 800 -> memorized).",
            MemorizationRiskWarning, stacklevel=2)
    if eval_rows is not None:
        ek = {key(r) for r in eval_rows}
        overlap = sum(1 for k in train_keys if k in ek) / max(len(ek), 1)
        report["train_eval_overlap"] = round(overlap, 4)
        if overlap > 0.05:
            warnings.warn(
                f"train/eval question overlap {overlap:.2%} — eval "
                "gains are contaminated above ~5%",
                MemorizationRiskWarning, stacklevel=2)
    return report
