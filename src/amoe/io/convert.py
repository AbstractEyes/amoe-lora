"""amoe-convert — .pt anchor stacks -> .safetensors companions, verified.

Converts the versioned format AND every legacy campaign shape
(load_diffusion_anchor handles the zoo). Each conversion round-trips the
result and asserts BITWISE tensor equality before reporting success.

Colab-safe usage (no argparse required):
    from amoe.io.convert import convert_one, convert_dir
    convert_one("mb3_s0.pt", substrate={"family": "sd15"}, objective="eps")
    convert_dir("ckpts/", substrate={"family": "sd15"})

CLI: amoe-convert <path.pt|dir> [--substrate sd15] [--objective eps]
                  [--layout amoe|comfy] [--out OUT]

Block anchors (amoe.anchor): convert_block_anchor(path) rewrites one file
as safetensors, in place when it already carries the .safetensors name (a
torch file saved under that extension by amoe < 0.2.6):
    from amoe.io.convert import convert_block_anchor
    convert_block_anchor("arms/s1_step1000.safetensors")
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from .checkpoint import load_diffusion_anchor
from .safetensors_io import load_anchor_safetensors, save_anchor_safetensors


def convert_one(path, *, substrate: "dict | None" = None,
                objective: "str | None" = None, layout: str = "amoe",
                out: "str | None" = None) -> str:
    path = Path(path)
    ck = load_diffusion_anchor(str(path), substrate=substrate)
    if objective:
        ck.meta.setdefault("objective", {"kind": objective})
    dst = Path(out) if out else path.with_suffix(".safetensors")
    save_anchor_safetensors(ck, str(dst), key_layout=layout)
    back = load_anchor_safetensors(str(dst))
    assert set(back.adapters) == set(ck.adapters), "key set changed"
    for k in ck.adapters:
        a, b = ck.adapters[k], back.adapters[k]
        assert a.dtype == b.dtype and torch.equal(a, b), \
            f"bitwise mismatch at {k}"
    print(f"converted {path.name} -> {dst.name} "
          f"({len(ck.adapters)} tensors, kind={ck.kind}, verified bitwise)")
    return str(dst)


def convert_block_anchor(path, *, out: "str | None" = None) -> str:
    """Rewrite one block anchor (amoe.anchor) as safetensors. The meta
    travels verbatim; the file is written beside the target first, read
    back with BITWISE tensor and meta checks, then moved into place, so a
    failed conversion never replaces the original."""
    import json
    import os

    from .checkpoint import load_anchor
    from .safetensors_io import is_safetensors, save_block_anchor_safetensors
    path = Path(path)
    dst = Path(out) if out else path.with_suffix(".safetensors")
    if dst == path and is_safetensors(str(path)):
        print(f"{path.name}: already safetensors")
        return str(path)
    ck = load_anchor(str(path))
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        save_block_anchor_safetensors(ck, str(tmp))
        back = load_anchor(str(tmp))
        # explicit checks (not asserts): this path replaces files in place
        if not is_safetensors(str(tmp)):
            raise RuntimeError("not written as safetensors")
        if set(back.adapters) != set(ck.adapters):
            raise RuntimeError("key set changed")
        for k in ck.adapters:
            a, b = ck.adapters[k], back.adapters[k]
            if a.dtype != b.dtype or not torch.equal(a, b):
                raise RuntimeError(f"bitwise mismatch at {k}")
        meta = {k: v for k, v in back.meta.items() if k != "content_hash_v2"}
        if meta != json.loads(json.dumps(ck.meta)):
            raise RuntimeError("meta changed")
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"converted {path.name} -> {dst.name} "
          f"({len(ck.adapters)} tensors, verified bitwise)")
    return str(dst)


def convert_dir(root, **kw) -> list[str]:
    out = []
    for p in sorted(Path(root).rglob("*.pt")):
        try:
            out.append(convert_one(p, **kw))
        except Exception as e:          # noqa: BLE001 — report and continue
            print(f"SKIP {p.name}: {e}")
    return out


def main(argv=None):
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print(__doc__)
        return 1
    kw = {}
    target = args.pop(0)
    while args:
        flag = args.pop(0)
        if flag == "--substrate":
            kw["substrate"] = {"family": args.pop(0)}
        elif flag == "--objective":
            kw["objective"] = args.pop(0)
        elif flag == "--layout":
            kw["layout"] = args.pop(0)
        elif flag == "--out":
            kw["out"] = args.pop(0)
        else:
            print(f"unknown flag {flag}")
            return 1
    p = Path(target)
    if p.is_dir():
        kw.pop("out", None)
        convert_dir(p, **kw)
    else:
        convert_one(p, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
