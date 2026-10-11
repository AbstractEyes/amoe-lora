"""Arm-key ownership (0.2.12): the arm's parameters are the only ones that train, and the trunk stays bit-identical.

The rule every arm in the research line trained under: name the arm's parameters (a handle's `parameters()`, or key
prefixes), freeze everything else, give the named parameters to pure Adam (laws.make_optimizer: no weight decay), and
assert after training that no trunk tensor changed by a single bit. `trunk_fingerprint` hashes every parameter and
buffer outside the arm; `assert_trunk_identical` compares two fingerprints and names the first tensors that differ.

    handle = attach_lookup(model, sites=[2, 6])
    frozen, kept = freeze_except(model, handle.parameters())
    opt = arm_optimizer(handle.parameters(), lr=3e-4)
    before = trunk_fingerprint(model, exclude=handle.parameters())
    ...  # train
    assert_trunk_identical(before, trunk_fingerprint(model, exclude=handle.parameters()))
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import torch
import torch.nn as nn

from ..laws import make_optimizer


def _ids(params: Iterable[nn.Parameter]) -> set:
    return {id(p) for p in params}


def named_by_prefix(model: nn.Module, prefixes: Iterable[str]) -> list[nn.Parameter]:
    """The parameters whose names start with any of `prefixes` (the trunk-native way to name an arm's keys)."""
    pre = tuple(prefixes)
    return [p for n, p in model.named_parameters() if n.startswith(pre)]


def freeze_except(model: nn.Module, keep: Iterable[nn.Parameter] | Iterable[str]) -> tuple[int, int]:
    """requires_grad False on every parameter not in `keep` (parameters, or name prefixes), True on those in it.
    Returns (frozen, kept) counts."""
    keep = list(keep)
    if keep and isinstance(keep[0], str):
        keep = named_by_prefix(model, keep)
    kept_ids = _ids(keep)
    frozen = kept = 0
    for p in model.parameters():
        on = id(p) in kept_ids
        p.requires_grad_(on)
        kept += on
        frozen += not on
    return frozen, kept


def arm_optimizer(params: Iterable[nn.Parameter], lr: float = 1e-3) -> torch.optim.Adam:
    """Pure Adam over the arm's parameters (law 1); the only optimizer an arm trains under."""
    params = list(params)
    if not params:
        raise ValueError("arm_optimizer: no parameters")
    return make_optimizer(params, lr=lr)


def _digest(t: torch.Tensor) -> str:
    t = t.detach()
    if t.is_sparse:
        t = t.to_dense()
    b = t.cpu().contiguous().flatten()
    if b.numel() == 0:
        return "empty"
    raw = b.view(torch.uint8) if b.element_size() > 1 else b
    return hashlib.blake2b(raw.numpy().tobytes(), digest_size=16).hexdigest()


def trunk_fingerprint(model: nn.Module, exclude: Iterable[nn.Parameter] = ()) -> dict:
    """name -> (shape, dtype, blake2b of the raw bytes) for every parameter and buffer not in `exclude`."""
    ex = _ids(exclude)
    out = {}
    for n, p in model.named_parameters():
        if id(p) not in ex:
            out[n] = (tuple(p.shape), str(p.dtype), _digest(p))
    for n, b in model.named_buffers():
        if b is not None and id(b) not in ex:
            out["buffer:" + n] = (tuple(b.shape), str(b.dtype), _digest(b))
    return out


def assert_trunk_identical(before: dict, after: dict, *, show: int = 4) -> None:
    """Raise unless the two fingerprints agree tensor for tensor (names, shapes, dtypes, bytes)."""
    missing = [k for k in before if k not in after]
    added = [k for k in after if k not in before]
    changed = [k for k in before if k in after and before[k] != after[k]]
    if missing or added or changed:
        raise AssertionError(
            "the trunk changed: "
            f"{len(changed)} tensors differ {changed[:show]}; {len(missing)} missing {missing[:show]}; "
            f"{len(added)} added {added[:show]}")
