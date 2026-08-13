"""train() — the certified expert recipe (reference implementation).

REFERENCE-GRADE: this transplants the research line's exp013 recipe
(frozen trunk, RelayPatchwork at every bound layer, pure Adam 1e-3,
fp32/TF32-off, gradient checkpointing use_reentrant=False, batch 8,
1200 steps). It runs on the reference substrate; treat as the starting
point, not an optimized trainer. DDP-aware: under torch.distributed it
rank-shards sampling and wraps in DDP (find_unused_parameters=False —
all trainable params fire every step in this recipe); multi-GPU smoke
is deferred (single-GPU-verified only).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import laws
from ..binding.resolver import resolve
from ..core.adapter import BlockWithAdapter, RelayPatchwork
from ..io.checkpoint import AnchorCheckpoint
from .config import TrainConfig
from .data import pad_batch, render_rows
from .guards import check_dataset


def _sites(cfg: TrainConfig, n_layers: int) -> list[int]:
    return list(range(n_layers)) if cfg.sites == "all" else list(cfg.sites)


def train(model, dataset, config: TrainConfig | None = None, *,
          tokenizer=None, binding=None, device="cuda",
          eval_rows=None) -> AnchorCheckpoint:
    cfg = config or TrainConfig()
    laws.pin_precision()
    b = resolve(model, binding)
    layers = list(b.layers(model))
    d = b.hidden_size(model)
    guard_report = check_dataset(dataset, eval_rows)

    rows = render_rows(tokenizer, dataset, cfg.max_len) \
        if tokenizer is not None else list(dataset)

    torch.manual_seed(cfg.seed)
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = nn.ModuleList()
    sites = _sites(cfg, len(layers))
    new_layers = list(layers)
    for i in sites:
        a = RelayPatchwork(d, cfg.adapter).to(
            next(layers[i].parameters()).device)
        adapters.append(a)
        new_layers[i] = BlockWithAdapter(layers[i], a)
    b.set_layers(model, new_layers)

    ddp = torch.distributed.is_available() and \
        torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if ddp else 0
    run_model = model
    if ddp:
        run_model = nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=False)

    g = torch.Generator().manual_seed(cfg.seed + rank)
    opt = laws.make_optimizer(adapters.parameters(), cfg.lr)
    if cfg.grad_checkpointing and hasattr(model,
                                          "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    pad_id = tokenizer.pad_token_id if tokenizer is not None else 0
    for step in range(1, cfg.steps + 1):
        ix = torch.randint(0, len(rows), (cfg.batch_size,), generator=g)
        x, y, am = pad_batch([rows[i] for i in ix], pad_id, device)
        out = run_model(input_ids=x, attention_mask=am, labels=y)
        loss = out.loss if hasattr(out, "loss") else F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.shape[-1]),
            y[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if rank == 0 and (step % cfg.log_every == 0 or step == 1):
            print(f"[amoe.train {cfg.name}] step {step} "
                  f"loss={float(loss):.4f}", flush=True)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()

    state = {}
    for si, a in zip(sites, adapters):
        for k, v in a.state_dict().items():
            state[f"{si}.{k}"] = v.detach().cpu()
    # restore the unwrapped model
    b.set_layers(model, layers)
    meta = {"name": cfg.name,
            "base_model_id": getattr(getattr(model, "config", None),
                                     "_name_or_path", ""),
            "d": d, "n_layers": len(layers), "sites": sites,
            "seed": cfg.seed, "precision": cfg.precision,
            "recipe": {"optimizer": "adam", "lr": cfg.lr,
                       "weight_decay": 0.0, "steps": cfg.steps,
                       "batch": cfg.batch_size},
            "guards": guard_report}
    return AnchorCheckpoint(state, meta)
