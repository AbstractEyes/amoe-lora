"""Row rendering — the exact-prefix collator law.

Rows are either pre-tokenized {"ids": [...], "n_prefix": int} or
messages-form {"messages": [...], "target": str}. Messages-form is
rendered by the TWO-PASS chat-template method and the exact-prefix
assertion is a hard error, never a fudge (the campaign's collator law:
a silent prefix drift poisons the label mask).
"""
from __future__ import annotations

from typing import Sequence

import torch


def render_rows(tok, rows: Sequence[dict], max_len: int) -> list[dict]:
    out = []
    n_drift = 0
    for r in rows:
        if "ids" in r:
            if len(r["ids"]) <= max_len:
                out.append({"ids": list(r["ids"]),
                            "n_prefix": int(r["n_prefix"])})
            continue
        msgs = r["messages"]
        pre_t = tok.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        full_t = tok.apply_chat_template(
            msgs + [{"role": "assistant", "content": r["target"]}],
            tokenize=False, add_generation_prompt=False)
        pre = tok(pre_t, add_special_tokens=False).input_ids
        full = tok(full_t, add_special_tokens=False).input_ids
        if full[:len(pre)] != pre:
            n_drift += 1
            continue
        if len(full) <= max_len:
            out.append({"ids": full, "n_prefix": len(pre)})
    if n_drift:
        raise ValueError(
            f"exact-prefix drift on {n_drift} rows: the chat template "
            "re-tokenizes the prefix differently with the assistant "
            "turn appended — fix the template usage, do not mask over "
            "it (collator law)")
    return out


def pad_batch(rows: Sequence[dict], pad_id: int, device):
    """Right-pad; labels -100 over prefix AND pads."""
    W = max(len(r["ids"]) for r in rows)
    ii, ll, am = [], [], []
    for r in rows:
        ids, n = r["ids"], len(r["ids"])
        lab = [-100] * r["n_prefix"] + ids[r["n_prefix"]:]
        ii.append(ids + [pad_id] * (W - n))
        ll.append(lab + [-100] * (W - n))
        am.append([1] * n + [0] * (W - n))
    t = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    return t(ii), t(ll), t(am)
