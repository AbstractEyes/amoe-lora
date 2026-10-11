# The lookup arm

`amoe.arms.lookup` (0.2.12). A removable causal softmax-attention organ that gives a trunk content-addressed lookup
without a retrain. Built for the Beatrix byte models, whose attention is address-based rather than softmax; it attaches
to any decoder the binding resolver can address.

## What it is

After each chosen block the stream receives one more causal attention module behind its own LayerNorm:

```
x = block_i(x)
x = x + attn_i(norm_i(x))        # for every site i; `depth` such layers run in series, each behind its own norm
```

`attn_i` is ordinary attention: a fused q/k/v projection without bias, scaled dot product with a causal mask, an output
projection without bias. Two guards are on by default: a per-head RMS normalization of q and k with learned per-head,
per-channel gains (born at one), and, under autocast on a GPU, the whole attention block computed in fp32.

The output projection is **born at zero**. A fresh arm is therefore exactly inert: the trunk's logits at step 0 are its
own, bit for bit, and `handle.all_off()` or `handle.detach(verify=True)` give the bare model back bit for bit at any
later time. The arm is born null by *weight*, never by a gate: a gate at zero passes no gradient through the module,
a zero weight does, so the arm can start learning on its first step.

The arm's parameters are a key subset of the model. They ship, load and detach on their own; the trunk never changes.

## Why

A trunk whose memory is address-based or recurrent fetches a value planted behind text only within a few hundred bytes
of its cue. Softmax attention fetches it to the end of the context. On the Beatrix byte model the bare trunk returned a
planted five-byte value from 2,048 bytes back on about one trial in twenty; with a lookup arm trained on recall rows it
returned it on more than nine in ten, with the natural-text loss unchanged to the third decimal. The trunk was frozen
throughout; only the arm's keys trained.

## Attaching

```python
from amoe.arms.lookup import LookupSpec, attach_lookup, attach_saved_lookup, attach_from_alephllm

spec = LookupSpec(heads=16, depth=1)                 # qk_norm="rms", attn_fp32=True by default
handle = attach_lookup(model, sites=[2, 6, 12, 20], spec=spec, seed=0)     # fresh, zero-born
handle = attach_saved_lookup(model, "lookup.pt")                           # a saved arm
handle = attach_from_alephllm(model, state_dict)                           # a geolip.alephllm arm (hub layout)
```

`model` is any decoder the binding resolver knows: an AlephLM (`model.blocks`), a Llama / Qwen / Mistral-style HF
model (`model.model.layers`), or any module list named by a dotted path (`binding="transformer.h"`). The arm follows
its block's device.

The handle is an ordinary `AttachHandle`: `all_off()`, `only()`, `enable()`, `disable()`, `detach(verify=True)`, plus
`arms()`, `parameters()`, `n_params()`, `state_dict()`, `save(path, meta)`, `reset_state()`, `alephllm_state()`.

## Decode

The wrapper's `prefill` runs the full causal pass and keeps the arm's own key/value cache; `step` appends the new
position's keys and values and attends over the cache. The trunk's own decode cache is untouched. One decode stream per
handle at a time: `prefill` resets the arm's cache, `reset_state()` clears it, and a `step` with no cache refuses.
`tests/test_lookup_arm.py` asserts prefill + steps against the full forward.

## Depth

`LookupSpec(depth=k)` puts k layers at every site, in series, each behind its own LayerNorm, all zero-born. A state saved
at a shallower depth loads into a deeper arm: the layers the state lacks stay zero-born, so the deeper arm *is* the
shallower arm at birth and a continuation adds structure without a restart (`attach_saved_lookup(model, path, depth=2)`;
`LookupArm.load_grown`). A deeper state never loads into a shallower arm.

## Training it: the recipe in outline

The trunk frozen; pure Adam on the arm's keys only (`amoe.train.ownership`: `freeze_except`, `arm_optimizer`,
`trunk_fingerprint` / `assert_trunk_identical` prove the trunk did not move by a bit); a curriculum that mixes recall
rows into ordinary text. The recall rows plant a value behind a span of text and ask for it at a cue; the distance from
the plant to the question must reach the context the arm is meant to cover, and the rows must be laid out so that the
training stream's packing never cuts an episode (documents that fill the window exactly). The rate matters: a high rate
peaks early and drifts; read the arm at every checkpoint and keep the best one. The full recipe, the data renderer and
the Colab notebook live in the public AlephLLM repository (`docs/lookup_arm.md` there); the trained Beatrix arms ship
on the Hugging Face hub in the AlephLM key layout, which `attach_from_alephllm` reads.

```python
from amoe.train.ownership import freeze_except, arm_optimizer, trunk_fingerprint, assert_trunk_identical

handle = attach_lookup(model, sites=[2, 6, 12, 20], spec=LookupSpec(heads=16))
freeze_except(model, handle.parameters())
opt = arm_optimizer(handle.parameters(), lr=3e-4)
before = trunk_fingerprint(model, exclude=handle.parameters())
...                                                   # the training loop: task loss on the mixed stream
assert_trunk_identical(before, trunk_fingerprint(model, exclude=handle.parameters()))
handle.save("lookup.pt", meta={"base_model_id": "alephllm/<craft>@step<N>"})
```

## Keys and formats

| layout | keys | where |
|---|---|---|
| this module | `{site}.layers.{k}.qkv.weight`, `{site}.layers.{k}.o.weight`, `{site}.layers.{k}.q_gain`, `{site}.layers.{k}.k_gain`, `{site}.norms.{k}.weight`, `{site}.norms.{k}.bias` | `handle.state_dict()`, `handle.save()` (`format: amoe.lookup`, loadable with `weights_only=True`) |
| geolip.alephllm | `lookup_arms.{site}.*` and `lookup_norms.{site}.*` for the first layer; `lookup_arms.{site}_l{k+1}.*` for the deeper ones | `from_alephllm_state`, `to_alephllm_state`, `handle.alephllm_state()` |

The parity test (`AMOE_LOOKUP_PARITY_CKPT=<a lookup-arm craft's checkpoint> pytest tests/test_lookup_arm.py`) attaches
a trained Beatrix arm to the bare trunk through this module and asserts the logits equal the trunk's own arm path bit
for bit.

## Laws

Pure Adam, no weight decay, on the arm keys only (law 1). The attention is a read, not a router: no top-k, no selector
(law 2). The output projection is born at zero (law 3). All arms off equals the bare model bit for bit, and detach is
verified against a pre-attach fingerprint (law 4).

## Background

The components are known: softmax attention's role in associative recall and the limits of attention-free sequence
mixers on recall tasks (Arora et al., "Zoology", 2023; Arora et al., "Simple linear attention language models balance the
recall-throughput tradeoff", 2024), induction heads as the circuit behind in-context copying (Olsson et al., 2022), and
synthetic retrieval curricula that transfer to natural text (Hsieh et al., "From Artificial Needles to Real Haystacks",
2024). The combination here is the package's contribution: a removable zero-born attention organ on a frozen
attention-free trunk, trained alone on a recall curriculum laid out for the packed stream, with depth as a growth path.
