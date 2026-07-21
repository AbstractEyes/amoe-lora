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
