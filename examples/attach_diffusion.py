# attach_diffusion.py — the diffusion quickstart, runnable as-is on a
# CUDA machine with `pip install amoe-lora[diffusion]` (Colab: paste the
# cell; no argparse, no __file__).
import torch

import amoe.diffusion as ad

from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float32).to("cuda")

# The certified multiband stack (exp008 s0; 2-seed surgical band lesions).
anchor = hf_hub_download("AbstractPhil/aleph-diffusion-adapters",
                         "sd15/mb3_s0.safetensors")
h = ad.attach(pipe.unet, anchor)
print(h.gates())                       # gate health at load

img = ad.sample(pipe, h, "a lighthouse at dusk, film photo", seed=7)
img.save("lighthouse.png")

# Band lesion: generate WITHOUT the HIGH-noise/structure band — the
# exp010 image-space instrument (lesioning HIGH cut grounding most and
# was 14x LP-dominated: it carries coarse structure).
with h.lesion_band(2):
    ad.sample(pipe, h, "a lighthouse at dusk, film photo",
              seed=7).save("lighthouse_no_high.png")

base = h.detach()                      # verified bit-exact, or it raises
print("detached clean")
