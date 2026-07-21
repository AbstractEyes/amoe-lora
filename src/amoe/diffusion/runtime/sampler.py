"""StepGatedSampler — the controller (exp010): per-step band gating at
inference. Each denoising step sets the crossfade windows from the
CURRENT noise level, activating band experts coarse-to-fine across the
trajectory exactly as trained. Image space sees what eps-MSE could not:
on the certified mb3 stack the controller moved grounding +0.089 over
frozen, with monotone band-lesion signatures (exp010, image-space).

Two objectives:
  "eps"  — DDIM on the stock SD15-style schedule (the exp010-proven path);
  "flow" — Euler on the SHIFT-warped sigma grid (the trained warp of the
           flow substrate, exp013).

Works for relay anchors too — the windows are simply unused.
Uncond convention follows the certified beds: ZEROED conditioning (the
CFG-dropout convention the adapters were trained with), not an
empty-prompt embedding.
"""
from __future__ import annotations

import torch

from ..laws import FLOW_SHIFT
from .attach import DiffusionAttachHandle


def _default_ddim():
    from diffusers import DDIMScheduler
    # The stock SD15 schedule, constructed explicitly (no download):
    return DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085,
                         beta_end=0.012, beta_schedule="scaled_linear",
                         clip_sample=False, set_alpha_to_one=False,
                         steps_offset=1, prediction_type="epsilon")


class StepGatedSampler:
    def __init__(self, model, handle: DiffusionAttachHandle, *,
                 scheduler=None, objective: str = "eps",
                 shift: float = FLOW_SHIFT, device=None,
                 latent_channels: int = 4):
        assert objective in ("eps", "flow"), objective
        self.model, self.handle = model, handle
        self.objective, self.shift = objective, shift
        self.latent_channels = latent_channels
        self.device = device or next(model.parameters()).device
        self.sched = None
        if objective == "eps":
            self.sched = scheduler or _default_ddim()

    @torch.no_grad()
    def sample_latents(self, ehs_cond: torch.Tensor,
                       ehs_uncond: "torch.Tensor | None" = None, *,
                       seed: int, steps: int = 30, guidance: float = 7.5,
                       latent_size: tuple = (64, 64)) -> torch.Tensor:
        B = ehs_cond.shape[0]
        dev = self.device
        dt = ehs_cond.dtype
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(B, self.latent_channels, *latent_size,
                        generator=g, device=dev, dtype=dt)
        unc = (torch.zeros_like(ehs_cond) if ehs_uncond is None
               else ehs_uncond)
        ehs_in = torch.cat([ehs_cond, unc], dim=0)

        if self.objective == "eps":
            self.sched.set_timesteps(steps, device=dev)
            x = x * self.sched.init_noise_sigma
            for t in self.sched.timesteps:
                s01 = torch.full((2 * B,), float(t) / 1000.0, device=dev)
                self.handle.set_band_windows(s01)         # the step gate
                xin = self.sched.scale_model_input(torch.cat([x, x]), t)
                pred = self.model(xin, t, ehs_in, return_dict=False)[0]
                p_c, p_u = pred.chunk(2)
                pred = p_u + guidance * (p_c - p_u)
                x = self.sched.step(pred, t, x).prev_sample
            return x

        # flow: Euler from s=1 -> 0 on the SHIFT-warped grid (exp013 warp)
        u = torch.linspace(1.0, 0.0, steps + 1, device=dev)
        s_grid = (self.shift * u) / (1 + (self.shift - 1) * u)
        for i in range(steps):
            s, s_next = s_grid[i], s_grid[i + 1]
            s01 = torch.full((2 * B,), float(s), device=dev)
            self.handle.set_band_windows(s01)             # the step gate
            v = self.model(torch.cat([x, x]), s01[:1].expand(2 * B) * 1000,
                           ehs_in, return_dict=False)[0]
            v_c, v_u = v.chunk(2)
            v = v_u + guidance * (v_c - v_u)
            x = x + (s_next - s) * v                      # dx/ds = v
        return x


@torch.no_grad()
def sample(pipe, handle: DiffusionAttachHandle, prompt: str, *,
           negative_prompt: "str | None" = None, seed: int = 0,
           steps: int = 30, guidance: float = 7.5, objective: str = "eps"):
    """Ease-of-use path for a diffusers StableDiffusionPipeline whose UNet
    carries an attached anchor: encode -> step-gated sample -> decode."""
    dev = pipe.device if pipe.device.type != "meta" else "cuda"
    tok = pipe.tokenizer(prompt, padding="max_length", truncation=True,
                         max_length=pipe.tokenizer.model_max_length,
                         return_tensors="pt")
    ehs = pipe.text_encoder(tok.input_ids.to(dev))[0]
    unc = None
    if negative_prompt is not None:
        ntok = pipe.tokenizer(negative_prompt, padding="max_length",
                              truncation=True,
                              max_length=pipe.tokenizer.model_max_length,
                              return_tensors="pt")
        unc = pipe.text_encoder(ntok.input_ids.to(dev))[0]
    sz = pipe.unet.config.sample_size
    sampler = StepGatedSampler(pipe.unet, handle, scheduler=None,
                               objective=objective, device=dev)
    lat = sampler.sample_latents(ehs, unc, seed=seed, steps=steps,
                                 guidance=guidance, latent_size=(sz, sz))
    img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    from PIL import Image
    import numpy as np
    arr = (img[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(np.asarray(arr))
