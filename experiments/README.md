# experiments/ — beds that ask one question and can answer it

Reference-grade research beds built on the shipped `amoe` API. Each one
declares its question, its controls and its preregistered forks BEFORE it
runs, so the result is usable whichever way it lands.

| bed | question | cost |
|---|---|---|
| [`aleph_mnist`](aleph_mnist/) | Where is the boundary between "adapter on a frozen trunk" and "trunk + adapter trained together"? | one evening, one GPU (runs on CPU) |

---

## aleph_mnist — the co-training dial

### The question

Snap an aleph adapter onto a miniature 4-block linear MNIST classifier
and train both at once. What happens?

The campaign record already answers the two extremes, and they disagree
with each other:

- **Co-trained (exp012, differentiation line):** the address-bottleneck
  prior **pays** — the multi-slot M̂ head beat an unrestricted head 7/7
  across seeds and budgets, and the advantage held at d=384.
- **Frozen substrate (exp013 Track A):** it **costs** — a param-matched
  MLP won, aleph tax ≈ +0.09 CE, 2 seeds.

Both are canon. **Nothing between them has ever been measured.** The plan
that would have measured it was ranked first in the standing queue and
was displaced by the genetic-distillation campaign before it ran.

So "train them both simultaneously" is not one run. It is a **dial**, and
the interesting number is where the curve crosses zero:

```
trainable trunk blocks:   0        1        2        4
                       frozen   ----------------->  full co-training
                       (exp013 pole)              (exp012 pole)
```

If Δ(aleph − control) crosses zero somewhere on that dial, the crossing
point is the boundary law the two poles have been waiting for. If it
never crosses, exp012's result is scale- or architecture-scoped and that
is worth knowing too.

### The bed

A 4-block pre-norm residual MLP over a (B, T, d) stream — a decoder block
with the attention removed, which is exactly the object an amoe adapter
expects to receive. `d=64`, T=1, ~4096 class-balanced MNIST rows.

**The starvation is the instrument.** Full MNIST on a 4-block MLP
saturates near 98% and every arm difference vanishes into seed noise. The
campaign has been burned by exactly this before — a four-arm codebook
sweep on BERT reconstruction found the addressing was not load-bearing at
all, because the task was too easy and the model routed around it. The
ruling that came out of it governs this bed: *no champion if it is not
actually using the alephs.* Hence the control arm below, and hence a
capacity-starved trunk with real headroom left in it.

### The arms — one class, four reads, identical parameters

Every arm carries the same `proj`, the same 64×4 codebook, the same
consume stack, the same gate scalar: **110,940 trainable parameters,
byte-identical layout.** Only the read of the 16 slot rows differs.

| arm | read | role |
|---|---|---|
| `soft` | M̂ = Σₖ sinh(uₖ)Aₖ / Σₖ cosh(uₖ) | the aleph, stock `RelayPatchwork` |
| `sign` | hard oriented code, straight-through | the sign code — won 12/12 frozen-substrate cells |
| `none` | M̂ = M (slots pass through) | **the control** |
| `off`  | no adapter | the baseline curve |
| `frozen` | soft read, codebook not trainable | basin-set-at-init test |

`none` is the canonical gate control: the codebook parameter is still
present so counts match exactly, but nothing reads it, so it receives no
gradient — **verified: its measured drift is exactly 0.000000**. It
doubles as the param-matched MLP arm, since proj → linear → squared-ReLU
→ LayerNorm → linear *is* an MLP adapter.

Reporting an MNIST result without this arm would be uninterpretable. That
is not a stylistic preference; it is the standing ruling from the
codebook-pressure probe.

**On the argmax inside `sign`:** this is HARD mode from the hosted
checkpoints, not the forbidden failure class. The failure class is
*comparative* selection over a roster of alternatives (argmax anchors,
softmax routing, STE one-hots, VQ). Hard mode argmaxes over the model's
own oriented half-axes to read a committed sign code. Known counter-
evidence, recorded here on purpose: on BERT reconstruction hard mode
destroyed fidelity (argmax tiles the continuum); on classification heads
sign ≥ soft in 12/12 frozen-substrate cells. Classification is the regime
where it has won.

### Phase 0 is not optional

"Frozen" has to mean **frozen-pretrained**, or the dial's left pole is a
random-feature bed and nothing transfers to exp013's finding. Every cell
starts from one shared trunk checkpoint trained without adapters; that
checkpoint is also the baseline. `pretrain_steps=0` gives the other
reading of the question — snap the adapter onto a fresh trunk and let
both move from step 0, which the sweep runs as its `scratch` rows.

### Preregistered forks

Write the verdict from whichever of these fires. All are canon-grade.

1. **Δ(soft − none) crosses zero on the dial.** The boundary law lands:
   *the bottleneck prior pays iff the substrate can co-adapt, and the
   boundary sits at N trainable blocks.* The unification of exp012 and
   exp013.
2. **No crossing, aleph trails everywhere.** exp012's result is scale- or
   architecture-scoped; MNIST-at-d=64 is below the substrate size where
   the prior pays. Recorded as a scope limit.
3. **No crossing, aleph leads everywhere.** The advantage is placement-
   specific (adapter position) rather than co-training-specific — which
   would contradict the frozen-substrate pole and demand a rerun there.
4. **Sign/soft ordering flips along the dial** — sign ≥ soft at the frozen
   end, soft ≥ sign at the full end. This is the mechanism prediction the
   two campaigns jointly imply, and it comes free on the same dial: soft
   reads pay only where the substrate can move toward the book.
5. **Toggle damage grows with the dial.** Expected, and the sharpest
   practical finding if it is large — see below.

### What gets measured

| probe | reads | reference to judge against |
|---|---|---|
| `inertness` | max\|Δlogit\| adapters on vs off at step 0 | law 3 says ~0; the LayerNorm/bias path is documented to leak |
| `delta_ratio` | ‖A(h)−h‖ / ‖h‖ per block | how much of the residual stream is being rewritten |
| `escape_report` | domain vs neutral amplitude | ratio ≤ 1.5 = blend-regime escape (`amoe.laws`) |
| `sign_code_report` | unique committed codes + MI with the digit label | differentiation must be structural; gradient-learned alphabets have collapsed 1,594 → 116 paths |
| `vitals_report` | codebook drift, gate mean, axis aliveness, CM CV | drift → **0.29154 rad**; gate band **0.012–0.03** |
| `grad_norm_spread` | trunk vs adapter gradient norms | who is actually learning |
| `toggle_report` | accuracy on vs off | **the detachability tax** |

Two of these are worth more than the accuracy numbers:

**Gate band.** The 0.012–0.03 gate-mean band is a live invariant
candidate observed across 6 architectures and 2 optimizers. This bed is a
cheap 7th data point on a task geometry unlike any of them, and there is
a standing open question asking whether the band is a delta-window prior
or a property of task geometry. Note the adapter *starts* at
sigmoid(−3) = 0.0474, above the band — so "does training pull it in?" is
a real question, not a formality.

**Toggle damage.** The toggle law promises the *mechanism* is bit-exact.
It promises nothing about what the trunk has come to depend on. Under
co-training the trunk can grow into the adapter, and then switching it
off is bit-exact **and catastrophic**. Mechanism intact, behaviour
destroyed — that gap is the sharpest thing this bed can show, and it
bears directly on whether co-trained aleph artifacts can be shipped as
detachable adapters at all.

### Input structure: the trigram rule (why the linear climb tied)

The single-linear stem feeds the aleph a **unigram** — one value per position,
nothing three-way to bind — and the address's advantage (L-AR5: byte_emb×3
*differentially* benefits the addressed head) cannot appear on a unigram. That,
not "the aleph is inert on vision," is why the linear-stem climb tied
`soft == none` across every dataset and width. The campaign rule is discovery
#16: **channel count = n-gram order**, and *byte-trigram-as-RGB engaged first
try*.

`input_mode="trigram"` (CLI `--trigram`, notebook `TRIGRAM=True`) swaps in the
byte_emb×3 stem — the canonical AlephLM input — in two forms:

- **channel** (RGB, e.g. cifar): each pixel is a natural byte-trigram (R,G,B);
  embed each channel with its own table and sum. `T = H·W` tokens.
- **spatial** (grayscale mnist/fashion): no channel trigram, so form the
  sequence one — `emb0(px_t) + emb1(px_{t-1}) + emb2(px_{t-2})`, past-only —
  the exact AlephLM form with pixels as the bytes. `T = pixels`.

Pixels are quantized to byte levels; the per-token blocks stay per-token (no
cross-token mixing), so the flatten readout does the spatial aggregation and the
aleph reads each pixel-trigram exactly as it reads each token in the LM. **This
is the real test of whether the address manifests on pixels**

**Cost.** Trigram turns each image into a `T=784–1024` sequence — ≈1000× the
linear bed's single token — so it runs at smaller `d` and batch (the notebook and
`--trigram` pick these automatically). A `d·T` flatten readout would explode at
large `d` (d=1024 → a ~1M-wide flatten, a ~2 GB activation that WDDM-spills and
crawls), so a per-token bottleneck `d→readout_dim` (default 16) precedes the
flatten — `readout_dim·T` stays small and memory is flat across widths. — and it pairs
with law 2 (the address must parameterize the output): if trigram *input* alone
still ties, the objective is the remaining lever (a next-pixel-byte / trigram-
reconstruction head, where the address parameterizes the prediction).

### Result so far (seed 0, d=64, MNIST) — the linear-stem climb (unigram input)

The QUICK pass landed on **the tie**: `Δ(soft − none)` never crosses zero
(−0.00016 / −0.00055 / −0.00026 / +0.00008 CE across the dial; sign
unstable; the control wins accuracy at 3 of 4 positions). exp012's
co-training win and exp013's −0.09 frozen tax **both fail to appear** —
d=64 MNIST is below the substrate complexity where the address bottleneck
is load-bearing, and the escape gauge confirms it (ratio < 1 on every
pretrained cell: the adapter fires *harder* on scrambled input than on
real digits). Two findings that are **not** noise: co-training dissolves
the adapter's load-bearingness (toggle `damage_ce` goes negative once the
trunk moves — turning the head off *improves* CE), and from scratch the
adapter is a net liability the trunk grows to depend on (+6.4% toggle
damage, yet a plain trunk beats every adapter arm). The gate sits
0.05–0.06, *above* the 0.012–0.03 band, and rises. Single seed — the tie
needs seeds, and the real signal needs a bigger substrate.

### Running it

Two Colab notebooks:
[`aleph_mnist_climb.ipynb`](notebooks/aleph_mnist_climb.ipynb) is the full,
maintained one (dial + the section-9 substrate climb);
[`aleph_mnist_cotrain.ipynb`](notebooks/aleph_mnist_cotrain.ipynb) is the
original, kept as the **executed seed-0 record** (its cells carry the run
outputs this result is read from).

Locally, from a checkout with `pip install -e .` — one dial sweep:

```bash
python -m aleph_mnist.runner --seeds 0 1 --steps 1500 --pretrain-steps 1500
```

Shapes/parse smoke (synthetic data, no download, seconds):

```bash
python -m aleph_mnist.runner --smoke
```

### Scaling up — the substrate climb (`--big`)

The seed-0 tie says: climb the substrate on **full** training sets until
the bottleneck becomes load-bearing. `--big` runs the grid — `d ∈
{64,128,256,512,1024}` × `{mnist, fashion, cifar10}` × seeds 0–2, full
sets, a fresh phase-0 trunk at every width. exp012's win lived at d=384,
so the ladder brackets it.

```bash
python -m aleph_mnist.runner --big                       # the whole climb
python -m aleph_mnist.runner --big --datasets mnist fashion   # skip cifar
python -m aleph_mnist.runner --big --dims 64 256 1024 --batch 2048 --seeds 0 1
```

Every cell prints peak VRAM + s/step at its first step (the WDDM rider —
an overrun fails loud, never silently spills). These trunks are tiny
(≈0.1 GiB at d=64), so the card is **compute-bound, not memory-bound**:
the workout is sustained utilization across the ~450-cell climb on full
data. To actually stress VRAM, push `--batch 8192` and/or `--dims 2048`.

**CIFAR-10 from Hugging Face (default).** torchvision's default host
(cs.toronto.edu) is slow, so CIFAR loads from the
[`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10)
parquet over HF's CDN instead — ~37 s for the full 50k/10k here vs a
2-minute mirror timeout. Knobs (env):

- `AMOE_CIFAR_HF_REPO` — point at a different HF repo (e.g. your own
  mirror namespace) instead of `uoft-cs/cifar10`.
- `AMOE_CIFAR_SOURCE=torchvision` — force the old tar path.
- `AMOE_CIFAR_URL` — a faster host of `cifar-10-python.tar.gz` for the
  torchvision path (implies `torchvision`).

HF needs the `datasets` library (present on Colab; `pip install datasets`
otherwise); if it's missing or offline, the loader falls back to
torchvision automatically. The grid still runs cifar **last** and skips it
only if *both* sources fail, so mnist+fashion always complete.

Every cell appends a JSON row to a ledger (`results/ledger.jsonl` for a
sweep, `results/grid.jsonl` for `--big`); `plots.figure_set(rows)` and
`plots.verdict_table(rows)` read it back. The ledger is the shippable
evidence — figures are derived from it, never hand-transcribed.

### House riders observed

Pure Adam, `weight_decay=0` (`amoe.laws.make_optimizer` is the only
optimizer constructor called anywhere in the bed). fp32 with TF32 off.
No global average pooling — the T>1 readout flattens. No bare argparse at
import and no `__file__` reliance, so every module is paste-safe into a
Colab cell. MNIST's ~80% exact-zero pixels go through the standard
(x−0.1307)/0.3081 normalization, which maps background to −0.4245: no
input float is a hard zero, which is the "every float must carry signal"
law being satisfied rather than assumed.

### One finding already, before any training

`amoe.runtime.attach._probe` fingerprints a model by calling
`model(input_ids=...)` with an integer tensor — it is written for causal
LMs. Rather than fork `attach()`, `TinyTrunk` accepts `input_ids` and maps
it deterministically to a float input, which makes the guarantee testable
here for real. It holds:

```
toggle all_off  ce 0.0394255370  ==  pre-attach ce 0.0394255370   True
handle.detach(verify=True)                          bit-exact PASS
```

So amoe 0.2.2's runtime verbs work unmodified on a vision trunk with a
generic `binding='blocks'` — **gated only on the LM-shaped probe**. If a
future version takes a `probe_fn` (or sniffs the forward signature), the
substrate-shim discipline extends to any architecture with a residual
stream, with no other change. Worth a line in the roadmap.

### Limits, stated up front

- MNIST at d=64 is a small substrate. Fork 2 above exists because a null
  result here does not overturn a d=384 result — it bounds it.
- The dial has 4 positions because the trunk has 4 blocks. The crossing
  point, if any, is located to ±1 block; a finer dial needs a deeper
  trunk.
- `sign` and `none` artifacts share the stock checkpoint layout but not
  its semantics. The meta records `address_mode`; loading one with plain
  `amoe.attach` would silently read it as `soft`. Only `soft` rows are
  stock anchors.
- The CM CV readout is logged, never gated: a D=4 codebook sits near
  CV ≈ 0.99 (measured here: 0.9925), far above the 0.13–0.30 band, which
  is the documented volatile regime for D=4 and not a fault.
