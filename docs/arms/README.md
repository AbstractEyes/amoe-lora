# The arm families

An **arm** is a removable module added after a trunk block. Every family in this package obeys the same contract: born
inert by *weight* (the output projection at zero, never a gate at zero), its parameters a key subset that ships and
detaches alone, pure Adam with no weight decay on those keys only, no selectors anywhere, all arms off equal to the bare
model bit for bit, and detach verified against a pre-attach fingerprint. The families differ in what they read and what
they compute.

| family | module | reads | computes | state across decode steps | since |
|---|---|---|---|---|---|
| relay | `amoe.core.adapter.RelayPatchwork` | the block output | a signed address read over a small codebook, a two-layer patch head, a sigmoid gate | none (position-wise) | 0.1 |
| EMA relay | `amoe.arms.relay_ema.RelayEMA` | the block output | the relay plus two fixed-decay causal averages of its own read as extra head inputs (born at zero) | the two averages | 0.2.5 |
| recurrent | `amoe.arms.recurrent_wide.RecurrentWide` | the block's attention output | a GRU per site on a projection of the read, a head to the stream | the GRU state | 0.2.5 |
| lookup | `amoe.arms.lookup.LookupArm` | the block output | zero-born causal softmax attention behind its own LayerNorm; `depth` layers in series | the arm's own key/value cache | 0.2.12 |

How to attach each: `amoe.attach(model, anchor)` for relay anchors (always-on; a list of anchors with a dispatch);
`attach_relay_ema`, `attach_recurrent`, `attach_lookup` for the others. Every attach returns a handle with
`all_off()`, `only()`, `detach(verify=True)`.

## What each family is for

- **relay**: the general adapter of the research line; capability, template and distillation arms, frame alloys from
  other models' states, surface arms that teach a byte model to read a tokenizer's spelling, and the stage arms of a
  curriculum, all trained on this module with different recipes.
- **EMA relay**: a test of whether a short memory of the arm's own read buys anything on a closure task. It did not,
  measurably; the module stays as the certified negative.
- **recurrent**: a deduction arm with per-site recurrent state, the control for the question of whether recurrence
  helps where the trunk's own memory is address-based.
- **lookup**: content-addressed lookup for a trunk without softmax attention; see [lookup_arm.md](lookup_arm.md).

## The recipes (training procedures)

| recipe | what it does | in the package |
|---|---|---|
| task rows | the language-model loss on the arm's domain, the trunk frozen | `amoe.train` (reference-grade) |
| the abstention term | KL(bare || armed) on off-domain rows, alternated with task chunks: the arm learns to stay quiet where it has nothing to say | `amoe.train.quiet` |
| arm-key ownership | freeze everything but the arm; pure Adam on its keys; prove the trunk did not move by a bit | `amoe.train.ownership` (0.2.12) |
| the recall curriculum | recall rows mixed into text, laid out so the packed stream never cuts an episode; the lookup arm's recipe | documented in [lookup_arm.md](lookup_arm.md); the renderer lives in the AlephLLM repository |
| staged co-training as a group; the surface recipe; the frame import | planned for a later release | — |

## Laws, in one place

1. Pure Adam, never weight decay, on the arm's parameters only.
2. No selectors: no top-k, no routing loss; a dispatch over anchors is dense and signed, and masking never renormalizes.
3. Zero-born output: a fresh arm is inert.
4. The toggle law and verified detach: all arms off, and the model after detach, equal the bare model bit for bit.
5. fp32 by default; bit-exact guarantees are dtype-conditional under bf16.
