"""build_cache — images + captions -> the trainer's cache dict.

The trainer is cache-fed (the pod-bed discipline): latents and text
features are encoded ONCE, then training never touches the VAE/TE again.
The campaign's caches load directly too — any dict with lat/ehs
(+ optional blob, val_lat/val_ehs) works.

Blob targets: the campaign rasterized entity polygons from the fused
dataset (research record, exp011). The generic path here accepts a
directory of binary masks (PNG, same stem as each image), resized to the
latent grid. Empty masks are counted and reported LOUDLY (the exp011
schema lesson: an empty blob target is a silent zero).
"""
from __future__ import annotations

from pathlib import Path

import torch


@torch.no_grad()
def build_cache(image_dir, pipe, *, captions=None, n_val: int = 256,
                size: int = 512, blob_masks=None, device: str = "cuda",
                batch: int = 8) -> dict:
    """image_dir: folder of images. pipe: a diffusers pipeline carrying
    vae/text_encoder/tokenizer. captions: None | txt file (one caption per
    image, sorted order) | dict stem->caption. blob_masks: None | folder
    of binary PNG masks (same stems)."""
    from PIL import Image
    import numpy as np

    paths = sorted(p for p in Path(image_dir).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    assert paths, f"no images under {image_dir}"
    if isinstance(captions, (str, Path)):
        lines = Path(captions).read_text(encoding="utf-8").splitlines()
        assert len(lines) >= len(paths), \
            f"{len(lines)} captions for {len(paths)} images"
        cap_of = {p.stem: lines[i] for i, p in enumerate(paths)}
    elif isinstance(captions, dict):
        cap_of = captions
    else:
        cap_of = {p.stem: "" for p in paths}

    vae, te, tok = pipe.vae, pipe.text_encoder, pipe.tokenizer
    sf = vae.config.scaling_factor
    lat_sz = size // (2 ** (len(vae.config.block_out_channels) - 1))

    lats, ehss, blobs, empty = [], [], [], 0
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = []
        for p in chunk:
            im = Image.open(p).convert("RGB").resize((size, size))
            imgs.append(torch.from_numpy(
                np.asarray(im)).float().permute(2, 0, 1) / 127.5 - 1)
        x = torch.stack(imgs).to(device)
        lats.append((vae.encode(x).latent_dist.sample(
            generator=torch.Generator(device=device).manual_seed(1400 + i))
            * sf).cpu())
        t = tok([cap_of.get(p.stem, "") for p in chunk],
                padding="max_length", truncation=True,
                max_length=tok.model_max_length, return_tensors="pt")
        ehss.append(te(t.input_ids.to(device))[0].cpu())
        if blob_masks is not None:
            for p in chunk:
                mp = Path(blob_masks) / f"{p.stem}.png"
                if mp.exists():
                    m = Image.open(mp).convert("L").resize(
                        (lat_sz, lat_sz), Image.NEAREST)
                    mm = (torch.from_numpy(np.asarray(m)) > 127).float()
                else:
                    mm = torch.zeros(lat_sz, lat_sz)
                if mm.sum() == 0:
                    empty += 1
                blobs.append(mm)

    lat = torch.cat(lats)
    ehs = torch.cat(ehss)
    cache = {"lat": lat[:-n_val] if n_val else lat,
             "ehs": ehs[:-n_val] if n_val else ehs}
    if n_val:
        cache["val_lat"] = lat[-n_val:]
        cache["val_ehs"] = ehs[-n_val:]
    if blob_masks is not None:
        bl = torch.stack(blobs)
        cache["blob"] = bl[:-n_val] if n_val else bl
        if n_val:
            cache["val_blob"] = bl[-n_val:]
        cache["blob_empty_count"] = empty
        if empty:
            print(f"WARNING: {empty}/{len(paths)} blob masks are EMPTY — "
                  "empty targets train nothing and read as silent zeros "
                  "(the exp011 schema lesson)", flush=True)
    print(f"cache: {cache['lat'].shape[0]} train rows"
          + (f" + {n_val} val" if n_val else ""), flush=True)
    return cache
