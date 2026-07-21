# `amoe.diffusion.anchor` — the adapter file format

One file = one adapter stack for one trunk. The file carries its own
documentation: a loader can tell you what the adapter is, what it was
trained on, what it measurably does, how strongly to run it, and what it
does *not* do — without any external database.

Two containers, same logical content:

| container | when | notes |
|---|---|---|
| `.safetensors` | **preferred**, and what ComfyUI loads | meta rides in the safetensors metadata block (str→str) |
| `.pt` | the campaign's original saves | meta rides as a plain dict under `"meta"` |

## Tensor layout

Flat, one namespace, site index first:

```
blocks.{site_index}.{param_path}        # in the safetensors key space
{site_index}.{param_path}               # in memory / in the .pt payload
```

`site_index` is `0..n_sites-1` in **training enumeration order**.

> ### The ordering law (read this before writing a loader)
>
> Site order is diffusers `named_modules()` order, which registers
> `down_blocks` → `up_blocks` → `mid_block`. **The mid block is LAST**, not
> in the middle. For SD1.5 the width signature is
>
> ```
> [320,320,640,640,1280,1280, 1280,1280,1280, 640,640,640, 320,320,320, 1280]
> ```
>
> A denoiser's *execution* order is different (`input → middle → output`,
> giving `…1280,1280,1280,1280…` with mid at index 6). Zipping the two
> positionally misplaces 7 of 16 sites **and still runs**, producing quietly
> wrong images. Map by site identity, then verify the width signature —
> the two orders differ at indices 9 and 15, so the signature catches it.
> `DiffusionAnchorCheckpoint.widths` reads the signature off the tensors.

## Metadata

### Required (provenance — every file has these)

| field | type | meaning |
|---|---|---|
| `format` | str | `"amoe.diffusion.anchor"` |
| `version` | int/str | `1` |
| `adapter.kind` | str | `relay` · `multiband3` · `mono` · `bank` · `cond` |
| `substrate.family` | str | `sd15_unet` · `sdxl_unet` · `cosmos_dit` |
| `substrate.n_sites` | int | must equal the enumerated site count at attach |

Only `relay` and `multiband3` are **attachable**. `mono` and `bank` are
matched controls and falsified-routing evidence; `cond` is the Law-2
negative. Loaders may read them; `attach()` refuses them by design and
says why.

### Optional (provenance, written when known)

`adapter.*` spec (`n_slots`/`K`/`tau`/`hidden` for relay, `rank` for
multiband3) · `substrate.base_model_id` · `substrate.site_names` ·
`substrate.widths` · `objective.kind` (`eps`|`flow`|`v`) ·
`objective.shift` · `blob.lambda` · `dtype` · `seed` · `recipe.*` ·
`created` · `content_hash_v2` · `imported_from` · `home_reconstructed`

### Optional (the usage card — schema v1.1)

What a UI renders. All optional, so raw campaign stacks stay valid.

| field | type | meaning |
|---|---|---|
| `display_name` | str | human name, e.g. `"SD1.5 Relay — Grounding v1 (seed 0)"` |
| `usage` | str | what it does and when to reach for it |
| `evidence` | str | the measured verdict **with numbers and seed status** |
| `recommended_strength` | float | sane default for a strength slider |
| `band_roles` | list[str] | multiband only: role per band, LOW→HIGH |
| `caveats` | list[str] | what it costs, where it fails, what is unverified |
| `license` | str | SPDX-ish string |
| `nc` | bool | `true` = non-commercial (derived from NC weights) |

`DiffusionAnchorCheckpoint.card()` returns exactly this block with safe
defaults filled in.

### Honesty rule for `evidence`

`evidence` states what was measured, at how many seeds, against what
control — never a marketing claim. If a result is single-seed, it says so.
If a matched control beat it on the aggregate metric, that goes in
`evidence` or `caveats`, not omitted. Controls and falsified artifacts
carry `evidence` describing what they *refuted*; that is their value.

## Worked example

```json
{
  "format": "amoe.diffusion.anchor",
  "version": 1,
  "display_name": "SD1.5 Multiband — Coarse-to-Fine v1 (seed 0)",
  "adapter": {"kind": "multiband3", "rank": 16},
  "substrate": {"family": "sd15_unet", "n_sites": 16,
                "base_model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5"},
  "objective": {"kind": "eps"},
  "band_roles": ["fidelity/detail (LOW noise)",
                 "continuity/semantics (MID)",
                 "diversity/structure (HIGH noise)"],
  "usage": "Three sigma-band experts gated per sampling step. Lesion a band to see what it carries.",
  "evidence": "exp008, 2 seeds: band lesions surgical 3/3 — own-band damage 50-200x cross-band. A matched rank-48 monolith still edges it on aggregate eps-MSE under uniform pressure.",
  "recommended_strength": 1.0,
  "caveats": ["Trained on the stock SD1.5 eps trunk; other trunks are untested.",
              "The aggregate win belongs to the monolith control — the value here is the band structure, not the loss number."],
  "license": "MIT", "nc": false,
  "dtype": "float32", "seed": 0
}
```

## Band gating (multiband3 only)

Band windows are cosine crossfades on `s01`, the **normalized discrete
timestep**:

```
s01 = t / 1000            # eps trunks — t from the model's own sampling
s01 = sigma               # flow trunks — the SHIFT-warped sigma
edges = (0.35, 0.75)      # LAW constants, not tunables
xfade = 0.06
```

> `s01` is **not** a noise-level proxy. On the real SD1.5 schedule,
> `1 - alphas_cumprod[t]` puts 316 of 1000 timesteps in a different band
> than the one the expert was trained on. In ComfyUI, get it from
> `model_sampling.timestep(sigma) / 1000`.

## Loader checklist

1. Read meta; reject if `format` is absent or unknown.
2. Enumerate the host's sites; sort into **training order**.
3. Assert `n_sites` matches, then assert the width signature matches
   `checkpoint.widths` — refuse loudly on mismatch.
4. Cast adapters to the trunk's declared dtype (the dtype law).
5. `kind` in `{relay, multiband3}` to attach; otherwise explain and stop.
6. Render `card()` so the user sees provenance, evidence, and caveats.
