"""train() — the certified diffusion recipes as a verb.

Reference-grade transplant of the campaign recipes: exp006 (eps on the
stock schedule) and exp013 (flow with the SHIFT warp, optional blob
coupling at the λ≈1 operating point), on a frozen trunk with fresh
zero-init adapters, pure Adam wd=0.

DDP posture mirrors amoe.train: rank-sharded batch sampling, rank-0
logging, correct-by-construction — multi-GPU smoke deferred (F6). The
production multi-GPU path for this family is the diffusion-pipe fork
(DeepSpeed; proven end-to-end on Anima, exp004). No FSDP.

CONDITIONING LAW (encoded): blob supervision on objective='eps' REFUSES —
it was measured inert-to-negative at 2 seeds exactly where the eps
parameterization divides by a vanishing sqrt(alpha_bar) (exp012). Pass
force_blob_on_eps=True to run the control anyway (a ConditioningLawWarning
fires).
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from ...binding.diffusion import resolve_diffusion
from ...io.checkpoint import DiffusionAnchorCheckpoint
from ..core.multiband import BandBlockWrap, MultibandDelta, band_weights
from ..core.relay import BlockWithRelay, RelayPatch2D
from ..laws import make_optimizer, pin_precision
from .config import DiffusionTrainConfig
from .objectives import (add_noise, blob_lp_err, flow_pieces, make_schedule,
                         role_losses, warp_sigma)


class ConditioningLawWarning(UserWarning):
    pass


def _dist():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:                                       # noqa: BLE001
        pass
    return 0, 1


def train(model, cache: dict, config: "DiffusionTrainConfig | None" = None,
          *, binding=None, device: str = "cuda"
          ) -> DiffusionAnchorCheckpoint:
    cfg = config or DiffusionTrainConfig()
    assert cfg.objective in ("eps", "flow"), cfg.objective
    assert cfg.adapter in ("relay", "multiband3"), cfg.adapter
    if cfg.blob and cfg.objective == "eps":
        if not cfg.force_blob_on_eps:
            raise ValueError(
                "conditioning law: blob supervision on the eps objective "
                "was measured inert at 2 seeds (exp012) — the eps x0 "
                "recovery divides by a vanishing sqrt(alpha_bar) exactly "
                "in the supervised band. Use objective='flow' (or v-pred "
                "when available), or set force_blob_on_eps=True to run "
                "the control.")
        warnings.warn("blob-on-eps forced: this reproduces the exp012 "
                      "CONTROL arm, not a paying configuration",
                      ConditioningLawWarning)
    if cfg.blob and "blob" not in cache:
        raise ValueError("cfg.blob=True but the cache has no 'blob' masks "
                         "(see amoe.diffusion.data.build_cache)")

    rank, world = _dist()
    pin_precision()                       # judged gauges stay fp32 (law)
    b = resolve_diffusion(model, binding)
    dt = b.declared_dtype(model)          # dtype law
    model.requires_grad_(False)
    model.eval()
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    sites = b.sites(model)
    originals, wraps, mods = [], [], nn.ModuleList()
    for name, block, d in sites:
        if cfg.adapter == "relay":
            s = cfg.relay
            m = RelayPatch2D(d, n_slots=s.n_slots, K=s.K, tau=s.tau,
                             hidden=s.hidden)
            m.assert_zero_init()
            wrap = BlockWithRelay(block, m)
        else:
            m = MultibandDelta(d, r=cfg.rank)
            m.assert_zero_init()
            wrap = BandBlockWrap(block, m)
        p0 = next(block.parameters(), None)
        m.to(device=p0.device if p0 is not None else device, dtype=dt)
        b.replace(model, name, wrap)
        originals.append((name, block))
        wraps.append(wrap)
        mods.append(m)

    ddp_model = model
    if world > 1:
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False)

    opt = make_optimizer([p for p in mods.parameters()], lr=cfg.lr)
    acp = (make_schedule(cfg.base_schedule_id, device)
           if cfg.objective == "eps" else None)
    lat_all, ehs_all = cache["lat"], cache["ehs"]
    blob_all = cache.get("blob")
    n = lat_all.shape[0]
    g = torch.Generator().manual_seed(cfg.seed * 1000 + rank)
    gd = torch.Generator(device=device).manual_seed(cfg.seed * 1000 + rank)

    def set_w(s01):
        w = band_weights(s01)
        for wr in wraps:
            wr.w_bands = w
        return w

    for step in range(cfg.steps):
        idx = torch.randint(rank, n, (cfg.batch_size,), generator=g)
        idx = idx - (idx % world) + rank if world > 1 else idx
        idx = idx.clamp(0, n - 1)
        lat = lat_all[idx].to(device, torch.float32)
        ehs = ehs_all[idx].to(device, torch.float32)
        bsz = lat.shape[0]
        drop = torch.rand(bsz, generator=gd, device=device) < cfg.cfg_dropout
        ehs = ehs.clone()
        ehs[drop] = 0
        noise = torch.randn(lat.shape, generator=gd, device=device)

        if cfg.objective == "eps":
            t = torch.randint(0, 1000, (bsz,), generator=gd, device=device)
            # BAND COORDINATE LAW: s01 = t/1000 — the normalized DISCRETE
            # timestep, exactly what every certified bed trained on
            # (dexp008/011/012) and what the proven controller gates on at
            # inference (dexp010, StepGatedSampler). Do NOT substitute a
            # noise-level proxy such as 1 - alphas_cumprod[t]: measured on
            # the real SD1.5 scaled_linear schedule, 316 of 1000 timesteps
            # land in a DIFFERENT band under that proxy (t=300 trains LOW
            # but the proxy says MID; t=700 trains MID, proxy says HIGH),
            # so roughly a third of training would teach the wrong expert
            # and inference would gate on an axis the stack never learned.
            # Pinned by testing.assert_band_coordinate.
            s01 = t.float() / 1000.0
            w = set_w(s01) if cfg.adapter == "multiband3" else None
            x_t = add_noise(lat, noise, t, acp)
            pred = ddp_model(x_t.to(dt), t, ehs.to(dt),
                             return_dict=False)[0].float()
            target = noise
            x0_hat = None
        else:
            u = torch.rand(bsz, generator=gd, device=device)
            s = warp_sigma(u, cfg.shift)
            w = set_w(s) if cfg.adapter == "multiband3" else None
            x_t, v = flow_pieces(lat, s, noise)
            pred = ddp_model(x_t.to(dt), s * 1000, ehs.to(dt),
                             return_dict=False)[0].float()
            target = v
            x0_hat = x_t - s[:, None, None, None] * pred   # EXACT linear

        if cfg.band_roles and cfg.adapter == "multiband3":
            low, base, high = role_losses(pred, target)
            loss_vec = w[:, 0] * low + w[:, 1] * base + w[:, 2] * high
        else:
            loss_vec = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        if cfg.blob and x0_hat is not None:
            blob = blob_all[idx].to(device, torch.float32)
            wb = w[:, 2] if w is not None else torch.ones_like(loss_vec)
            loss_vec = loss_vec + cfg.blob_lambda * wb * blob_lp_err(
                x0_hat, lat, blob)
        loss = loss_vec.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if rank == 0 and (step % cfg.log_every == 0 or
                          step == cfg.steps - 1):
            print(f"[amoe.diffusion.train {cfg.name}] step {step} "
                  f"loss {loss.item():.5f}", flush=True)

    adapters = {}
    for i, m in enumerate(mods):
        for k, v in m.state_dict().items():
            adapters[f"{i}.{k}"] = v.detach().cpu()
    meta = {
        "name": cfg.name,
        "substrate": {"family": b.name, "n_sites": len(sites),
                      "site_names": [s[0] for s in sites],
                      "widths": [s[2] for s in sites]},
        "adapter": ({"kind": "relay", **vars(cfg.relay)}
                    if cfg.adapter == "relay"
                    else {"kind": "multiband3", "rank": cfg.rank}),
        "objective": ({"kind": "eps"} if cfg.objective == "eps"
                      else {"kind": "flow", "shift": cfg.shift}),
        "blob": ({"lambda": cfg.blob_lambda} if cfg.blob else None),
        "dtype": str(dt).replace("torch.", ""),
        "seed": cfg.seed,
        "recipe": {"optimizer": "adam", "lr": cfg.lr, "weight_decay": 0.0,
                   "steps": cfg.steps, "batch": cfg.batch_size,
                   "cfg_dropout": cfg.cfg_dropout,
                   "band_roles": cfg.band_roles},
    }
    # restore the unwrapped trunk before returning (amoe.train contract)
    for name, block in originals:
        b.replace(model, name, block)
    return DiffusionAnchorCheckpoint(adapters, meta)
