# Changelog

The `amoe` package's version is `amoe.__version__` (this line of work; `pyproject.toml` carries the distribution's
older number until the planned renumber).

## 0.2.12 — the lookup arm and arm-key ownership
- `amoe.arms.lookup`: a zero-born causal softmax-attention arm after chosen blocks of any bound decoder (`LookupSpec`,
  `LookupAttention`, `LookupArm`, `BlockWithLookup`, `LookupHandle`, `attach_lookup`, `attach_saved_lookup`,
  `load_lookup`); depth (layers in series, a shallower state grows into a deeper arm); prefill / step with the arm's own
  key/value cache; the geolip.alephllm key layout both ways (`from_alephllm_state`, `to_alephllm_state`,
  `spec_from_alephllm_state`, `attach_from_alephllm`).
- `amoe.train.ownership`: `freeze_except`, `arm_optimizer` (pure Adam), `named_by_prefix`, `trunk_fingerprint`,
  `assert_trunk_identical`.
- `tests/test_lookup_arm.py`: birth identity, toggle and detach bit-exact, causality, prefill + steps against the
  forward, depth growth, ownership (the trunk bit-identical after optimizer steps), save / load, the key map, and a
  parity read against a real AlephLM trunk with a trained arm when `AMOE_LOOKUP_PARITY_CKPT` names its checkpoint.
- `docs/arms/`: the arm families and the lookup arm's writeup.

## 0.2.11 — the compiled chain's recompile budget and shape gate
## 0.2.10 — the checkpointed chain with the autocast cast cache off
## 0.2.9 — the compiled adapter chain
## 0.2.8 — stacked wrappers recompute as one chain
## 0.2.7 — adapter recompute in the backward
## 0.2.6 — block anchors saved as real `.safetensors`
## 0.2.5 — the abstention term (`amoe.train.quiet`); `amoe.arms.recurrent_wide`, `amoe.arms.relay_ema`
## 0.2.4 — `strict=` checks identity through the binding
## 0.2.3 — the AlephLM binding and the cached-decode passthroughs
## 0.2.2 — `needs_bands` on diffusion adapters
## 0.2 — the diffusion subsystem (`amoe.diffusion`) and safetensors I/O
