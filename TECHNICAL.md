# amoe-lora — Technical Companion (Beatrix era)

Companion to *Raising Beatrix: A Byte-Level Model's Measured Childhood*
(AbstractPhil with Claude Fable 5 & Claude Opus 5, August 2026). This
document carries the arm-system detail: the adapter architecture, the
laws its failures purchased, and the arm-program results measured on
mini-beatrix-1. The anchors themselves ship in the
[training repository](https://huggingface.co/AbstractPhil/alephllm-mini-beatrix-training)
(`arms/`, `arms/night/`, `arms/day1/`, `arms/bdist/` — refuted arms
included, per the weights-completeness law).

## 1. The arm system

An **arm** is an anchored mixture-of-experts adapter (`RelayPatchwork`)
attached at every block of a frozen trunk. `AdapterSpec(n_slots, K, D,
tau, hidden, gate_init, zero_init_head)` — the Beatrix default is
16 slots / K=64 / D=4 / hidden=178 (≈3.2M parameters at 16 sites); the
"wide" spec used in capacity sweeps is 32 slots / hidden=256. Experts
are selected by the same signed-address rule as the trunk's own organs;
`zero_init_head` makes every arm **born null**.

Five verbs: `train`, `attach`, `align`, `toggle`, `detach`.

- **The toggle law**: arm-off must equal the untouched base to
  bit-exactness — asserted (`torch.equal` / max|Δ| == 0.0) against a
  fresh-loaded reference, never a flag flip. Every shipped arm passes.
- **Binding identity (0.2.4)**: provenance is resolved from the model
  *binding* (`alephllm/<craft>@step<N>`), not from `model.config`
  attributes the model class may not have. History: the original
  strict= check was **structurally inert** for AlephLM — a foreign-core
  arm attached silently and produced plausible wrong output,
  reproduced and indistinguishable by inspection. Declared-but-
  unverifiable identity now raises. Consumers additionally verify the
  anchor's own `base_model_id` metadata at load.
- **Label-shift semantics**: the trainer feeds HF-convention labels;
  the AlephLM side shifts internally. The original mismatch trained an
  arm to copy the current byte (loss ~10.7 nats); corrected training
  starts ~0.72 nats.
- **Memorization guard**: `check_dataset` compares question space to
  the training draw and warns when an adapter can memorize the space —
  the seed of the program's capacity-data law (below). Guard reports
  are captured into ledgers, never scrolled past.
- **Evaluation law — attach before gauging**: `train()` detaches on
  return (bit-exact restore contract). No gauge is valid until the
  saved anchor is re-attached and the wrapper sweep verified non-empty.
  (Purchased by a same-day retraction: two stop-arm gauges measured the
  bare core twice; the tell was arm-on ≡ arm-off to three decimals.)

## 2. The laws the arm program measured (2026-08-16 → 08-17)

1. **Capacity-data law** (program lead): arm capacity and corpus must
   match — small same-template corpora make "the memorization become
   the arm." 1,500 same-frame rows poisoned arms (chat repetition 12–16,
   target families *degraded*); 16k–24k varied-frame rows produced real
   capabilities with clean chat behavior (repetition 2–4).
2. **Template competence is scale-invariant**: narrow formats buy
   in-format perfection and near-chance transfer, in cores and in arms
   alike. Hence the three-tier evaluation: exam-family (fully
   disjoint) / seen-frame / **held-frame** — verdicts come from
   held-frame only.
3. **Mint-vs-steer**: an arm can sharpen an in-distribution convention
   (newline-pair turn ending: 0/6 → 6/6, replies 200 → 109 bytes,
   capabilities held, stable at T=1.0) but cannot lift an
   out-of-alphabet symbol against a lifetime prior (NUL: P unmoved at
   4e-8 after identical training). Control symbols enter at core
   pretraining or not at all.
4. **Composition**: pairs compose (stop+identity clean; stop+d5 lifts
   strict chaining 0.70 → 0.90 — each arm repairing the other's failure
   mode); same-direction stacks compress (three behavioral arms →
   13-byte replies). Deep stacks need orthogonality or gating.
5. **Weights-completeness**: every anchor ships, refuted included; the
   ledger census equals the anchor census.

## 3. Wave results on the graduate (step 88,508)

| arm | spec | corpus | held-frame | note |
|---|---|---|---|---|
| stopnl (turn-end \n\n) | 16-slot | 7.5k chat rows +stop | 6/6 stop-rate | first working post-grad arm; robust at T=1.0 |
| NUL stop (control) | 16-slot | same rows, NUL target | no stop; P(NUL)≈4e-8 | the mint-vs-steer refutation (valid instrument) |
| d5 rule-chaining | 16-slot | 24k, 3 frames | 0.70 strict (alien predicates) | failure mode = non-termination |
| d5 wide | 32-slot | same | **1.00 strict** | capacity was binding (gate G3) |
| stop + d5 collective | — | — | 0.90 strict | mutual failure-mode repair |
| poly generalist | 32-slot | 30k mixed (+stop) | **1.00 strict** | integrates capability + termination in one anchor |
| sub3 (four variants) | 16/32-slot | 24k bare / wide / mixed / show-work | 0.00–0.02 | the quadruple wall; distillation target |
| defs | 16-slot | 16k opengloss | 0.15 | knowledge-shaped tasks don't fit 3.2M |
| identity | 16-slot | 64 persona rows | — | persona lives in arms by law; memorization-by-design, flagged |

Gauges per the evaluation laws: generative, order-aware, surface-
disjoint; MC-likelihood exams are blind to format-reshaping arms
(measured: exam 0.00 vs generative 0.70–1.00 on the same capability).

## 4. The distillation lane (bdist-e001, pre-registered)

Target: the sub3 wall. Teacher Qwen2.5-3B-Instruct (accepted at 0.96
ceiling; the 1.5B variant **rejected** at 0.86 — a 14%-wrong teacher
would poison the bank). Teacher token distributions are pushed onto the
shared 256-byte simplex (alignment gate: 1.0000 gold-byte argmax,
0 sum violations across fixtures) — a **pinned frame by construction**,
so the frame-ambiguity law of the previous installment needs no
Procrustes step here. Arms: byte-KL (mimicry) vs chunk-likelihood
matching at bytelex co-boundaries (comparative), 2 seeds, against the
four SFT-refuted controls — same rows, same positions, same capacity,
hard labels vs distributions. Bars are numeric and public before
results; either outcome is a finding.
