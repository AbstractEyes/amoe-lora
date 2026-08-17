---
license: apache-2.0
tags:
- adapters
- mixture-of-experts
- aleph
- diffusion
---

# amoe-lora — aleph mixture-of-experts adapters

Train, **attach**, align, **detach** small aleph-addressed adapters on
a frozen trunk — with the honesty diagnostics that found this
architecture's real failure modes built in as first-class API.

Productized from the geolip-aleph research lines
([LM research record](https://huggingface.co/AbstractPhil/geolip-aleph-qwen-3.5-0.8b-instruct) ·
[diffusion research record](https://huggingface.co/AbstractPhil/geolip-aleph-diffusion) ·
[working caption adapter](https://huggingface.co/AbstractPhil/qwen3.5-0.8b-relay-caption)).
"LoRA-style" refers to the attach/detach usage pattern, not the math:
these are 16-slot patch heads reading the residual stream through a
closed-form sinh/cosh address over a 64-atom S³ codebook (~261k
params/block at d=1024).

**Status: 0.2 — working-state framework.** The runtime verbs (attach /
toggle / detach) and checkpoint I/O are implemented and invariant-
tested (CPU CI, no downloads), now on BOTH substrate families: language
trunks (`amoe`) and diffusion denoisers (`amoe.diffusion` — SD1.5-class
UNets, SDXL, Cosmos-Predict2/Anima DiT). 0.2 also ends the safetensors
non-goal: anchors save/load `.safetensors` with a ComfyUI-ready key
layout. The train recipes are reference-grade transplants of the
certified campaign recipes — runnable, single-GPU-verified, not yet
optimized or deep-tested. See *Maturity* below.

## Install

```
pip install git+https://github.com/AbstractEyes/amoe-lora            # core
pip install "amoe-lora[diffusion] @ git+https://github.com/AbstractEyes/amoe-lora"
```

or from a checkout: `pip install -e .`, extras `.[hf]` (language trunks)
and `.[diffusion]` (diffusers + safetensors). torch is a dependency, but
install the CUDA build that matches your machine first — a blind
`pip install torch` can replace a working one.

**Code lives on [GitHub](https://github.com/AbstractEyes/amoe-lora)**
(canonical, CI-guarded); the Hugging Face repo mirrors it as the card.
The adapter file format is documented in [SCHEMA.md](SCHEMA.md); trained
adapters live in
[aleph-diffusion-adapters](https://huggingface.co/AbstractPhil/aleph-diffusion-adapters)
and load into ComfyUI via
[comfyui-geolip-amoe-lora](https://github.com/AbstractEyes/comfyui-geolip-amoe-lora).

## The five verbs

```python
import amoe

# TRAIN an anchor (reference recipe: frozen trunk, pure Adam, fp32)
ck = amoe.train(model, rows, amoe.TrainConfig(name="caption"),
                tokenizer=tok)
ck.save("caption.anchor.pt")

# ATTACH (single, always-on) — adapters follow each block's device,
# so device_map="auto" sharded loading works unchanged
h = amoe.attach(model, "caption.anchor.pt")

# ALIGN a dispatch over frozen anchors (keys-only training,
# starvation safeguards on by default)
disp = amoe.align(model, ["arith.anchor.pt", "algebra.anchor.pt"],
                  streams, tokenizer=tok)
disp.save("math.dispatch.pt")

# TOGGLE at runtime (masking never renormalizes — the damping law)
h = amoe.attach(model, ["arith.anchor.pt", "algebra.anchor.pt"],
                dispatch="math.dispatch.pt")
with h.only("algebra"):
    out = model.generate(**enc)
print(h.usage())

# DETACH — verified BIT-EXACT against a pre-attach fingerprint,
# or it raises
base = h.detach()
```

## The diagnostics are the point

The research line's biggest finding is that these adapters have two
regimes, and **the dispatch only contains one of them**: specialize-
regime anchors damp cleanly off-domain; blend-regime anchors escape
damping, fire at full amplitude on unrelated text, tax perplexity, and
measurably trample chain-of-thought. You cannot see the difference
from training loss. You can see it in one call:

```python
from amoe.diagnostics import diagnose
report = diagnose(h, tok, domain_texts=my_domain_prompts)
# warns per anchor when on-domain/neutral amplitude ratio <= 1.5
```

`amoe.train` also runs a question-space guard on your dataset (an
adapter trained on more draws than the space holds distinct questions
memorizes the space and posts fake generalization — measured in the
research line, twice).

## Diffusion (new in 0.2)

The same verbs on a frozen denoiser — the productized r2 campaign
([geolip-aleph-diffusion](https://huggingface.co/AbstractPhil/geolip-aleph-diffusion):
relay certified relay-favorable 3-for-3 substrates, the multiband
mechanism with surgical band lesions, the step-gated controller, and
the **conditioning law**).

```python
# pip install amoe-lora[diffusion]
import torch, amoe.diffusion as ad
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float32).to("cuda")

h = ad.attach(pipe.unet, hf_hub_download(
    "AbstractPhil/aleph-diffusion-adapters", "sd15/mb3_s0.safetensors"))
img = ad.sample(pipe, h, "a lighthouse at dusk", seed=7)   # step-gated

with h.lesion_band(2):                     # generate without the HIGH band
    img_no_high = ad.sample(pipe, h, "a lighthouse at dusk", seed=7)
base = h.detach()                          # bit-exact or raises
```

Training on a folder of images is five lines:

```python
cache = ad.data.build_cache("photos/", pipe, captions="photos/captions.txt")
cfg = ad.TrainConfig(name="mystyle", objective="flow",
                     adapter="multiband3", blob=False)
ck = ad.train(pipe.unet, cache, cfg)
ck.save("mystyle.anchor.safetensors")
```

Three certified deviations from the LM defaults, each encoded as law
with its receipt (`amoe.diffusion.laws`):

- **Dtype law**: adapters attach at the DECLARED trunk dtype (bf16
  relays on a bf16 DiT), never a sniffed one — first-param sniffing can
  read a still-meta fp32 tensor. Judged gauges stay fp32.
- **`align()` is a grounded negative**: comparative routing on
  diffusion was falsified at 2 seeds (state+sigma keys route nothing;
  address-as-key dies against the repeated-key null in both raw and
  M-hat forms). `ad.align()` raises with the record. The certified
  alternative is STRUCTURAL banding — cosine crossfade windows on the
  sigma axis (`adapter="multiband3"`), whose band lesions are surgical
  (own-band damage 50–200× cross-band, 2 seeds).
- **Conditioning law**: blob/structural supervision pays where x0
  recovery is linear (flow: `x0 = x_t − σ·v`; ~125–200× the eps
  effect) and is inert on eps — `blob=True` with `objective="eps"`
  refuses unless forced.

### safetensors + ComfyUI

`ck.save("x.safetensors")` writes the canonical flat layout
(`blocks.{site}.{param}`) with the full meta as safetensors metadata;
`amoe-convert path.pt --substrate sd15` converts every legacy campaign
shape with bitwise verification. A `comfyui-amoe` custom-node package
(loader / attach / band-toggle / detach, with step gating installed as
a pre-forward hook so ANY stock KSampler becomes step-gated) is the
next-turn deliverable; the node consumes the canonical layout and
asserts the anchor's site-count + width signature before patching.

## House laws (encoded, not optional)

Pure Adam only (`laws.make_optimizer`); dense signed dispatch with an
all-anchor damped denominator — no top-k, no load-balancing, masking
never renormalizes; zero-init output heads; fp32/TF32-off default;
checkpoints carry the codebook `home` buffer. Each law's receipt is in
the research record.

## Multi-GPU

- **Inference**: works today with `device_map="auto"` — adapters
  follow their block's device.
- **Training**: the trainer is DDP-aware (rank-sharded sampling,
  `find_unused_parameters=False`; launch with `torchrun`). Correct by
  construction; multi-GPU smoke is **deferred** (single-GPU-verified).
- **Diffusion — the declared split (the standard paradigm)**: native
  `amoe.diffusion.train` = single-GPU + DDP at framework level;
  the PRODUCTION multi-GPU path is the
  [diffusion-pipe fork](https://github.com/AbstractEyes/diffusion-pipe)
  (DeepSpeed pipeline engine, aleph relays baked in, proven end-to-end
  on the Anima 2B DiT — r2 exp004). Use the fork when you have real
  scale; use the native trainer for the certified single-card recipes.
- **FSDP / tensor parallel**: not implemented; see
  `docs/distributed.md` for why and the sketch.

## Maturity

| area | state |
|---|---|
| core math (address/adapter/dispatch) | ported verbatim from certified code; parity by construction |
| attach / toggle / detach | implemented, invariant-tested (toggle law + bit-exact detach, CPU CI) |
| checkpoint I/O + legacy import | implemented; round-trip verified against the shipped campaign checkpoints |
| train / align | reference-grade (the certified campaign recipes, transplanted); single-GPU |
| DDP | implemented, untested on >1 GPU |
| diffusion core (relay/multiband/windows) | ported verbatim from the certified r2 beds; invariant-tested |
| diffusion attach / lesions / detach | implemented, invariant-tested incl. dtype law (CPU CI) |
| diffusion train (eps/flow/blob/roles) | reference-grade transplant; single-GPU-verified recipes |
| StepGatedSampler | eps/DDIM = the exp010-proven path; flow/Euler = same warp as the beds |
| safetensors I/O + amoe-convert | implemented in 0.2 (bitwise-verified round trips); ComfyUI node = next turn |
| SDXL binding | enumeration implemented; site count PINNED AT FIRST REAL RUN (never guessed) |
| FSDP/TP, token-level gating, streaming data | documented non-goals |

## Loading the campaign's shipped artifacts

```python
from amoe.io.checkpoint import load_anchor, import_legacy_keys
ck = load_anchor(hf_hub_download(
    "AbstractPhil/geolip-aleph-qwen-3.5-0.8b-instruct",
    "exp013_experts/ckpt/v35e13_algebra_steps_s0.pt"))   # legacy OK
```

## Lineage

Architecture and every default in this package trace to the
[geolip-aleph-qwen-3.5-0.8b-instruct](https://huggingface.co/AbstractPhil/geolip-aleph-qwen-3.5-0.8b-instruct)
campaign (19 experiment packages, raw ledgers, adversarial audits, one
published self-retraction plus a corrected overclaim), its
[Qwen2.5 predecessor](https://huggingface.co/AbstractPhil/geolip-aleph-qwen),
and — for the 0.2 diffusion subsystem — the
[geolip-aleph-diffusion](https://huggingface.co/AbstractPhil/geolip-aleph-diffusion)
campaign (16 experiment packages, 2-seed program, the conditioning law).


**Technical companion:** [TECHNICAL.md](TECHNICAL.md) — the arm system, its laws, and the Beatrix arm-program results.
