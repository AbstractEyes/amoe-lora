# Distributed story (0.2)

## The diffusion split (the standard paradigm, 0.2)

Diffusion training has TWO sanctioned paths, by design:

1. **Native `amoe.diffusion.train`** — single-GPU + the same DDP
   posture as the LM trainer below (rank-sharded cache sampling,
   `find_unused_parameters=False`, rank-0 saves). This is the
   framework-level path for the certified single-card recipes;
   correct-by-construction, multi-GPU smoke deferred.
2. **The diffusion-pipe fork**
   (https://github.com/AbstractEyes/diffusion-pipe) — the PRODUCTION
   multi-GPU trainer: DeepSpeed pipeline engine with the aleph relays
   baked into the model integrations (attach at declared dtype,
   freeze-by-lr-0 trunk, plain-Adam branch, adapter-only saves in the
   amoe anchor format). Proven end-to-end on the Anima 2B DiT
   (r2 exp004). Multiband w_bands plumbing is working at
   `pipeline_stages=1`; stages>1 is documented-deferred.

No FSDP on either path, same reasons as below.


## Implemented

- **Device-following attach**: each adapter/dispatch is placed on its
  wrapped block's device at attach time. This is the entire
  `device_map="auto"` compatibility story for inference — accelerate
  shards the trunk, the adapters follow, nothing else changes.
- **DDP-aware train/align**: when `torch.distributed` is initialized
  (launch with `torchrun`), the trainer rank-shards row sampling
  (per-rank generator offset), wraps in DDP with
  `find_unused_parameters=False` (all trainable params fire every step
  in the default recipes), and logs/saves on rank 0 only. Gradient
  sync is trivial at these sizes (6.3M train / ~8k align params).
  **Verified on one GPU only** — the DDP branch is correct by
  construction but multi-GPU smoke is deferred.
- Gradient checkpointing uses `use_reentrant=False` exclusively (the
  DDP-compatible mode).

## Documented non-goals (0.1)

- **FSDP**: the layer-replacement wrap conflicts with auto-wrap
  policies; `home`/`key_proj` buffers need sharded-state-dict care;
  and sharding megabyte-scale adapters buys nothing. Only relevant
  when the *trunk* must shard. Sketch for later: attach after FSDP
  wrapping at block granularity with a custom wrap policy that treats
  BlockWithAdapter as a unit.
- **Tensor/pipeline parallel, DeepSpeed**: the dispatch reads the
  full residual per block; TP would require the address/adapter to be
  sharded consistently with the trunk's TP plan — out of scope.
- Edge case encoded: `AlignConfig.train_new_anchor=True` plus runtime
  masks during training creates unused parameters; the aligner flips
  `find_unused_parameters=True` in that configuration.
