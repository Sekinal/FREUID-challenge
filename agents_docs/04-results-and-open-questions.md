# Results, Runs & Open Questions

## Synthesis: where the lever is (read this first)

Everything we've measured triangulates on one conclusion: **the fraud signature is
*type-entangled*, so there is no single-image silver bullet — the lever is data, not a
cleverer feature.** Evidence:

| Approach | In-dist | Unseen-type (LOTO/x-country) | Lesson |
|---|---|---|---|
| EfficientNet-b2 (digital) | FREUID ~0.0001 | public 0.291 | in-dist ≫ overstates reality |
| ConvNeXt **no aux** (x-country) | — | FREUID 0.98 | no external data ⇒ no OOD |
| EffNetV2-M + **30k IDNet aux** | — | FREUID **0.356** | **data is the 3× lever** |
| 66 scalar forensic features | AUC 0.995 | AUC **0.452** (below chance) | features memorise type |
| FFT cross-type fingerprint corr | — | **+0.06** | no universal generation signature |
| Normalised spectral *shape* | — | AUC **0.63–0.65** | transfers better; fusion candidate |
| **NPR** (CVPR 2024, up-sampling artifacts) | — | AUC **0.49–0.53** (chance) | wrong paradigm (see below) |
| **CLIP frozen + linear probe** (UnivFD) | — | AUC **0.686** | **best single-feature OOD**; semantic, fusion candidate |

So, in priority order, the levers that actually move the OOD number:
1. **More + more *diverse* external fraud data** — the only thing that attacks type-entanglement
   head-on (IDNet gave the one 3× win). Highest expected payoff.
2. **Fuse the normalised-spectral stream into the CNN** — orthogonal to RGB, transfers across
   types (0.45→0.65). Secondary, cheap, stackable on a *single* model.
3. **Scale the single EffNetV2** (resolution / epochs / TTA) on the leakage-safe LOTO harness.

(No ensembling until one model is genuinely strong — user directive.)

## Baseline (EfficientNet-b2, leakage-safe stratified-group splits)

3 epochs, batch 512, bf16 + channels_last + torch.compile, AuDET checkpoint selection, A100.

| Metric | val | test |
|---|---|---|
| FREUID | 0.0001 | 0.0001 |
| AuDET | 0.0000 | 0.0000 |
| APCER@1%BPCER | 0.0002 | 0.0002 |

- **In-distribution → near-perfect.** Expected: all 5 types seen in training; this is the
  "memorize the seen types" regime the host warned about. Do **not** read this as solved.
- Checkpoint: `runs/baseline/best.pt` (git-ignored). Results: `artifacts/baseline_results.json`.
- Note: checkpoints from a compiled model carry a `_orig_mod.` key prefix — `baseline.load_model`
  strips it.

## Leaderboard

| Submission | Model | publicScore (lower=better) |
|---|---|---|
| 53623661 | earlier type-holdout b2 (epoch2) | 0.333 |
| 53627140 | leakage-safe b2 (3 epoch) | 0.291 |
| 53954498 | EffNetV2-M **3-country** + domain aug + focal + IDNet aux | 0.30353 |
| 53954497 | EffNetV2-M **all-5** + domain aug + focal + IDNet aux | **0.20744** |

**Standing (public LB, 109 teams):** best = **0.20744 → rank 59/109**, up from ~86 with the
0.291 baseline. The 30k-IDNet-aux EffNetV2-M moved us **+27 places**. Note the all-5 model
(0.207) beats the 3-country one (0.304) on public *because public_test is all seen
types/countries* — that edge will not fully transfer to the private set's 2 unseen types, so
the cross-country-robust model remains the more honest generalizer. Top of LB is still
~0.0006 (rank 1); the lever for closing it = more external fraud data + heavier capture aug
on a single strong model. **Directive: NO ensembling until we have a genuinely strong single
model** — ensembling weak bases only hides the real problem. (We got +27 from one backbone,
30k aux, 3 epochs.)

- **The headline calibration:** our leakage-safe b2 scores **local in-distribution FREUID
  ≈ 0.0001** but **public LB 0.291**. The public test is the *same 5 types*, held-out images —
  yet performance collapses ~0.29, and that's *with* ~19% leaked train-twins helping it.
- **Most likely cause: the digital→captured shift.** Our local val is held-out-but-digital
  (easy); the public/private tests lean captured/print-and-capture (hard). This proves the
  in-distribution CV (0.0001) massively overstates reality.
- The earlier type-holdout model scored 0.333; our all-types model beats it (0.291) only
  because public_test types are all seen in training — that gain will NOT transfer to the
  private set's 2 unseen types.
- ⇒ Expect the private set (unseen types + *more* captured) to be **harder than 0.291**.
  Optimize capture-robustness + LOTO, not the public LB.
- Public LB is scored only on the 7,821 public_test ids (private dummies ignored).

## Cross-country OOD sweep (`aux-modeling` branch — second A100 box)

A parallel line of work ran on a second A100 box (`154.54.100.217`). It forked from
`7d8aeb4` (before the leakage-safe-splits overhaul) and is preserved faithfully as the
**`aux-modeling`** branch on GitHub. Two things make it valuable, and one caveat matters:

- **Different (and arguably better) validation philosophy.** Instead of `main`'s
  leakage-safe *stratified-group* split (in-distribution), it uses a **cross-country
  holdout**: train on 3 countries, test on the held-out one (Mozambique). That is a real
  **OOD proxy** — much closer to what the private set rewards (unseen types + captured)
  than an in-distribution split. Caveat: it does **not** carry `main`'s near-dup leakage
  grouping, so within-country twins can still leak; treat the absolute numbers as
  optimistic-but-directionally-honest.
- **External data (the big lever).** It mixes **IDNet-2025** (per-country fraud/genuine
  ID docs, CC-BY-4.0) into *train only* via `freuid/aux_data.py`, with a `domain`
  augmentation pipeline (compression/noise/perspective/blur/lighting) for the
  digital→captured shift.

Sweep (lower cross-country **TEST** FREUID = better OOD generalization):

| run | model | aug | loss | n_train | val (in-dist) | **TEST (x-country)** | AuDET | APCER@1% |
|---|---|---|---|---|---|---|---|---|
| e5 | **EffNetV2-M** | domain | focal | 72,625 | 0.0039 | **0.3556** | 0.1635 | 0.476 |
| e1 | ConvNeXt-base | domain | bce | 72,625 | 0.0015 | 0.4026 | 0.2564 | 0.501 |
| e4 | DINOv2 ViT-B | domain | focal | 72,625 | 0.8189 | 0.8764 | 0.3778 | 0.931 |
| e2 | ConvNeXt-base | domain | focal | 72,625 | 0.8148 | 0.9179 | 0.3875 | 0.956 |
| e3 | ConvNeXt-base | domain | focal | 87,625 | 0.9996 | 0.9952 | 0.5738 | 0.998 |
| e0 | ConvNeXt-base (**no aux**) | domain | focal | 42,625 | 0.8199 | 0.9809 | 0.4209 | 0.990 |

Findings:
- **IDNet aux is the dominant lever.** Same arch/aug/loss, only difference is aux:
  e0 (no aux) **0.9809** → with 30k IDNet aux it drops by an order of magnitude. Without
  external fraud examples the model does not generalize across countries at all.
- **Best so far: EffNetV2-M + domain aug + focal + 30k IDNet aux → 0.3556** cross-country.
- **Focal loss interacts with architecture.** Focal *helped* EffNetV2-M (0.3556, best) but
  *wrecked* ConvNeXt (e2 0.9179) vs the same ConvNeXt with plain BCE (e1 0.4026). Don't
  treat focal as a free win — tune per backbone.
- **More aux ≠ better.** Adding the *scanned* IDNet split (e3, 45k) made it dramatically
  worse (0.9952). The scanned/print-capture aux distribution likely fights the objective;
  needs investigation before reusing.
- **In-dist val is again worthless as a guide.** e5 val 0.0039 vs cross-country test 0.3556;
  e1 val 0.0015 vs 0.4026 — the *same* lesson as `main`'s 0.0001-vs-0.291-public gap. Only
  the OOD/cross-country number means anything.
- **Pseudo-labeling failed** (phase2): the confidence threshold kept **0** pseudo-labels
  from public_test → no signal. Student holdout 0.1823 is in-distribution, not OOD.
- These cross-country models were **never submitted to Kaggle** (CSVs written, "NOT
  uploaded"), so they have no public LB number. `main`'s only real public score remains
  **0.29127** (the leakage-safe b2). The aux models *might* do better on the public LB and
  the private set — untested.

## Hand-crafted forensic feature bank (statistical noise analysis)

Hypothesis tested: a generator-agnostic *statistical/noise* prior (ELA, noise residuals,
DCT/Benford, radial power spectrum, blockiness, compression metadata — 66 features in
`freuid/forensics.py`) might generalise to **unseen document types** better than backbones
that overfit seen types. Extracted on all 69,352 images; evaluated with `scripts/15` (GBM)
and `scripts/16` (per-feature within-type robustness).

| Setting | AUC | FREUID |
|---|---|---|
| In-distribution 5-fold CV (HistGB on 66 features) | **0.995** | 0.026 |
| **Leave-one-type-out** (train 4 types → predict 5th) | **0.452** | 0.948 |
| → OOD gap | — | **+0.92** |

**Verdict: the hypothesis is *not* supported.** The features carry very strong fraud signal
*in-distribution* (AUC 0.995) but **collapse to below chance on an unseen type** (mean LOTO
AUC 0.452; MAURITIUS/ID — the only ID among 4 DLs — inverts to 0.19). The GBM learns
*absolute*, type-specific thresholds (each issuer/capture pipeline has its own baseline
feature levels) that invert on a new type. This mirrors the deep models' OOD collapse and is
*worse* OOD than EffNetV2-M cross-country — so raw forensics → GBM is **not** the shortcut to
the unseen-type private set.

What survives (per-feature within-type AUC, direction-agnostic, `scripts/16`):
- **ELA (error-level analysis) is the standout family**: `ela_std` mean 0.74 (up to **0.995**
  within some types), `ela_frac_high` mean 0.72, `ela_mean` 0.69. Compression/recompression
  signatures are the strongest hand-crafted fraud cue here.
- But **no single feature has MIN within-type AUC ≥ 0.58** — every feature is strong in *some*
  types and near-useless in others (`ela_std` ranges 0.56→0.995 across the 5 types). The
  signal is real but **type-entangled and uneven**, never universal.

Takeaways:
- Confirms *why* OOD is hard: fraud signatures are heavily type-specific — which is also why
  adding *diverse external data* (IDNet) was the dominant lever, not architecture.
- Forensics are **not** a standalone OOD classifier. The only remaining credible use is
  **fusion**: feed ELA / noise-residual maps as extra input channels to EffNetV2 so the
  backbone supplies the type context the raw feature lacks. Expectations tempered — ELA is
  strong on *digital-recompression* fraud but likely weak on the captured/unseen types the
  private set emphasises (exactly the types where ELA's within-type AUC was lowest).
- Artifacts: `artifacts/forensic_eda.json`, `artifacts/forensic_feature_robustness.json`.

## Fourier / spectral fingerprint analysis (GPU)

Follow-up to the forensic-feature collapse: do *frequency-domain* patterns carry a
**type-agnostic** fraud signature? `scripts/17_spectral_fingerprint.py` (PyTorch, A100;
all 69,352 spectra in 63 s) computes the noise-residual 2-D FFT log-power per image,
**per-image normalised** (so it captures spectral *shape*, not type-specific energy level),
then aggregates mean fraud/genuine spectra per type.

**Decisive test — is the (fraud − genuine) spectral difference consistent across types?**
Cross-type correlation of the per-type difference maps = **+0.062** (≈ 0). → **No universal
generation fingerprint.** If GenAI/recompression left a consistent periodic signature the
per-type differences would align; they don't. Fraud is type-specific even in frequency space.

**But normalised spectral *shape* generalises to unseen types better than scalar forensics:**

| Representation (leave-one-type-out) | OOD AUC | FREUID |
|---|---|---|
| 66 scalar forensic features (`scripts/15`) | 0.452 (below chance) | 0.948 |
| Spectral radial profile, 64-bin (normalised) | **0.630** | 0.893 |
| Downsampled 2-D spectrum, 32×32 (normalised) | **0.646** | 0.843 |

Takeaways:
- Per-image normalisation is the trick: removing the type-specific energy *level* lifts
  unseen-type AUC from 0.45 → 0.63-0.65. Still weak in absolute terms (not a standalone lever),
  but it's the **first representation meaningfully above chance on unseen types**.
- It's **orthogonal** to the CNN's RGB view → a credible **fusion stream** (normalised
  residual-spectrum / radial profile as an extra input to EffNetV2). More promising than raw
  ELA because it actually transfers across types.
- Consolidates the meta-finding: **no single-image statistic is a silver bullet — fraud is
  type-entangled.** Dominant lever stays diverse data (IDNet); normalised-spectral fusion is a
  secondary, genuinely-transferable signal to stack on the CNN.
- Artifacts: `artifacts/spectral_fingerprint.json` (+ `spectral_means.npz`, git-ignored).

## Recent-literature techniques tried (and why most don't fit)

Surveyed 2024-2025 generalizable-forgery-detection papers and tested the cheapest, most apt:

- **NPR — "Rethinking Up-Sampling Operations", CVPR 2024** (`scripts/18`, GPU). NPR =
  `x − nearest_upsample(x[::2,::2])` on native pixels (no resize) — the generator up-sampling
  fingerprint, SOTA across 28 unseen GAN/diffusion models. On FREUID it is **chance** (LOTO AUC
  0.49–0.53 across magnitude-stats, NPR-spectrum, and combined; n=3,000 balanced, proper LOTO).
  **Why it fails here:** NPR detects *whole-image* generative synthesis, but FREUID fraud is
  *localized* document manipulation + print-and-capture. There is no global up-sampling fingerprint,
  and any capture-resampling artifact is shared by fraud and genuine. The off-the-shelf
  "AI-generated-image detector" paradigm is a category mismatch for document fraud.
- **CLIP frozen-feature probe — UnivFD (Ojha et al.)** (`scripts/19`, GPU). Frozen
  ViT-B/16 OpenAI-CLIP features (768-d) + a probe, leave-one-type-out, n=24,000.
  **Best single-feature OOD result so far:** linear probe **mean AUC 0.686** (per type
  0.54 / 0.65 / **0.85** / **0.81** / 0.59), GBM 0.622. Two findings that match the paper and
  matter: (1) **linear > GBM** — frozen CLIP + a *simple* probe generalises best; complex
  classifiers overfit seen types (so keep it linear, keep CLIP frozen). (2) **CLIP handles the
  unseen ID type** (MAURITIUS/ID AUC 0.81) where scalar forensics *inverted* (0.19) — its
  semantic features transfer to a structurally-different document, exactly the private-set
  challenge. Still not "strong" absolutely (FREUID 0.82), but the best, and **orthogonal to the
  CNN/low-level features → the top fusion candidate.** Follow-up: ViT-L/14 (UnivFD's backbone,
  usually stronger; the 1.7 GB download died on this box's flaky link) may push it higher.
- **Still on the list:** forgery *localization* (SAFIRE / SAM, 2025; DocForge-Bench, 2026) —
  predict the tampered *region*; apt since manipulations are local, but needs region labels we
  mostly lack (IDNet may provide some).
- Sources: NPR arXiv:2312.10461 · UnivFD arXiv:2508.01603 · C2P-CLIP arXiv:2408.09647 ·
  DocForge-Bench arXiv:2603.01433 · ID-card PAD review arXiv:2511.06056.

## Leave-one-type-out (LOTO)

- Smoke (random scorer) wired and working: freuid_mean ≈ 0.98 (chance), per-type + summary.
- **TODO / in progress:** real LOTO = 5 retrains (train on 4 types, score the held type) with
  the capture-augmented model → the honest unseen-type number. Script: `scripts/13`.

## Open questions / next steps

1. **Unify the two branches (the big one).** `main` has the better *infrastructure*
   (leakage-safe stratified-group splits, `iter_type_holdout`/LOTO support, vectorized
   metric, tests). `aux-modeling` has the better *experiments* (IDNet aux mix-in, multi-arch
   sweep, focal, the real cross-country OOD numbers). The win is to run `aux-modeling`'s
   aux-data + multi-arch modeling **through `main`'s leakage-safe LOTO harness** — i.e. port
   `aux_data.py` + the `domain`-aug + focal + model-arch flexibility onto `main`'s core, then
   get honest, leakage-safe, unseen-type numbers. Needs the box (data + training) to validate;
   `baseline.py`/`data.py` are clean superset merges, only the split *semantics* must be
   reconciled (stratified-group + a held-out type/country, not one or the other).
2. **IDNet aux is confirmed the dominant lever** (e0 0.98 → e5 0.36). Get more/better external
   fraud data; investigate why the *scanned* IDNet split hurt (e3).
3. **Real LOTO numbers** with the aux-augmented model — the OOD headline metric.
4. Scale a **single** model on the A100 (EffNetV2-M is the current best backbone; try
   higher res / longer training / TTA / better aug) once leakage-safe LOTO is the yardstick.
   **No ensembling until one model is genuinely strong** (user directive) — ensembling weak
   bases masks the real generalization gap rather than closing it.
5. **Submit the aux models** — the cross-country models (`submissions/sub_e5_*`, `sub_f3_*`)
   were never uploaded; calibrate them against the public LB.
6. When private images drop: re-predict and resubmit the full 142,818.

## Reproduce

```bash
.venv/bin/python scripts/05_duplicates_leakage.py      # dedup + leakage (artifacts/duplicates.json)
.venv/bin/python scripts/09_build_splits.py            # leakage-safe splits
.venv/bin/python scripts/11_train_baseline.py --epochs 3 --batch-size 512
.venv/bin/python scripts/12_predict_baseline.py --checkpoint runs/baseline/best.pt
.venv/bin/python scripts/13_leave_one_type_out.py      # OOD: 5 retrains
.venv/bin/python tests/test_metrics.py && .venv/bin/python tests/test_dedup.py && .venv/bin/python tests/test_splits.py
```
