# Competition Design & Strategy

## What the host told us (verbatim points)

From the competition host (Ivan Relić), after the full-data release:

- The leaderboard was **reset**; full train + public test released. Sample-phase submissions
  invalidated.
- **The goal is generalization to NEW document domains, not seen-type accuracy.**
- **The private test contains TWO document types unseen in both train and public test**,
  deliberately to "reduce the value of solutions that mainly memorize document templates,
  layouts, or source-specific artifacts."
- The private test **emphasizes non-synthetic, captured examples** — robustness to
  "print-and-capture effects, lighting and imaging variation, and other real-world conditions
  that may suppress fragile digital artifacts."
- Recommended: validation that **tests generalization across document types, countries,
  layouts, capture conditions, and digital-vs-captured** — explicitly *more informative than
  splits that only mirror the training distribution*.

## Submission mechanics (confirmed by host)

- Submit predictions for **all 142,818 ids**. For private ids (no images yet), provide **dummy
  outputs** — they are **NOT counted for the public leaderboard**.
- ⇒ **Public LB is scored only on the 7,821 `public_test` ids.** The fill value for private ids
  is **inert** for the public score (we use a fixed constant 0.5; see `scripts/12`).
- When the private images are released, replace dummies with real predictions and resubmit.

## The two stacked distribution shifts (our risk model)

1. **Unseen document types** — train/public have 5 types; private adds 2 new ones.
   → LOTO is the proxy.
2. **Digital → captured** — train is 99.97% digital; private emphasizes captured/print-scan.
   → We have almost **no captured training data**, so *no holdout can directly measure this*.
   This is likely the **bigger** gap and the harder problem.

## Strategy implications

- **Do NOT optimize the public LB.** It is in-distribution (same 5 types) and leakage-inflated
  (~19% near-dup twins). It will flatter any model that memorizes the seen types.
- **Optimize LOTO** (unseen-type proxy) as the headline offline metric.
- **Engineer for the digital→captured shift even though we can't directly validate it:**
  - Strong **capture-simulating augmentation**: JPEG recompression, blur, downscale-upscale,
    brightness/contrast/gamma, perspective warp, mild noise / print-scan / moiré.
  - Prefer signals that survive capture over fragile digital forensic artifacts; avoid
    overfitting to template/source/layout cues (which won't transfer to new types).
  - Consider: heavier/higher-res backbones, multi-crop TTA, k-fold ensembles, and possibly
    self-supervised / forensic features. (The A100 is under-used by b2 — room to scale.)
- **Anchor, don't chase:** one public submission per meaningful change is enough as a sanity
  check; rank candidate models by LOTO + group-CV locally.

## On the A100

- b2 saturates the GPU in *time* (~100% util) but uses ~14 GB of 80 GB. Capacity headroom is
  the real lever (bigger model / higher res / TTA), not kernel micro-opt.
- torchao fp8 does **not** apply (Hopper-only). bf16 + channels_last + compile + TF32 is the
  right A100 setup here; cheap extras: `AdamW(fused=True)`, `compile(mode="max-autotune")`.
