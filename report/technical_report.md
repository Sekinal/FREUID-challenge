# A Dual-Model Portfolio for Identity-Document Fraud Detection: Resolution as the Universal Lever

**The FREUID Challenge 2026 (IJCAI-ECAI) — Technical Report**
Team: EliasTSJ + Sekinal · July 2026 · Public LB: **0.02514**

---

## 1. Introduction

The FREUID Challenge asks for a fraud score per identity-document image across
five African countries, with a public leaderboard computed on ~5% of the test
set and a final private evaluation announced to contain (a) two document
types / countries absent from training, and (b) an emphasis on real
print-and-capture examples that suppress fragile digital artifacts.

This design creates a fundamental tension: the strategies that maximize the
public score (exploiting fragile pixel-level traces of digital manipulation)
are exactly the ones announced to be penalized on the private set. Our
solution embraces the tension instead of resolving it: we submit a
**two-model portfolio** — a *public specialist* and a *capture-robust
generalist* — trained identically except for two flags, and we validate every
robustness decision with **Leave-One-Country-Out (LOCO)** protocols rather
than the public leaderboard.

Our central empirical finding is that **input resolution dominates every
other design choice**, improving in-distribution and out-of-distribution
performance simultaneously and by an order of magnitude: public LB
0.123 → 0.025 and unseen-country FREUID 0.40 → 0.01–0.03 when moving from
768 px to 1024 px (near-native). The forensic traces of document manipulation
live at the pixel scale, and they transfer across countries — document
templates do not need to.

## 2. Data

### 2.1 Competition data (used by the final models)

The final models train **exclusively on the official FREUID training set**:
69,352 images, five types (EGYPT/DL, GUINEA/DL, BENIN/DL, MOZAMBIQUE/DL,
MAURITIUS/ID), fraud rate 42.8%, native resolutions up to 1585×1000 (median
width 1000). Only 20 training images are non-digital (captured); we reserve
them as a "canary" probe (§4.3). No pseudo-labels, no external data, no
public-test images are used in training the final submissions.

### 2.2 External sources (explored, measured, and rejected)

We evaluated four external augmentation strategies. Each was rejected on
measurement — a result we consider a key finding (§5.4):

| External source | License | Result |
|---|---|---|
| IDNet-2025 digital (~88k EU documents) | CC-BY-4.0 | No LOCO gain; >30k rows add noise on the public LB |
| IDNet-2025 scanned (print-captured) | CC-BY-4.0 | **Destroys** real-capture detection (canary AUC 0.50) |
| Synthetic GenAI fraud, 7.5k images we generated (InsightFace `inswapper` face-swap, SDXL-inpainting face/text edits, classic splices over FREUID genuines) | model-specific | Teaches "unknown country ⇒ fraud": held-out-country FREUID degrades to 1.0 (§5.4) |
| FantasyID (bona-fide capture set) | research | Neutral; dropped with the rest |
| Public-test pseudo-labels (transductive self-training) | — | Helps only while the teacher is stronger than the student; harmful at our final level (§5.5) |

## 3. Method

### 3.1 Architecture

Both final models are a `timm` **EfficientNetV2-M** (`tf_efficientnetv2_m.in21k_ft_in1k`)
with a small MLP head (LayerNorm → Dropout → Linear(1280→512) → GELU →
Dropout → Linear(512→1)), trained with BCE at **1024×1024** input resolution,
batch 12, AdamW, lr 1e-4, on a single NVIDIA L40S (48 GB).

### 3.2 The two training regimes

| | 🔵 Public specialist | 🔴 Capture-robust generalist |
|---|---|---|
| Capture-simulating augmentation | **off** | **heavy** |
| Epochs | 3 | **1** (early stop) |
| Seeds | 1 (seed 4) | 3, probability-averaged |
| Horizontal-flip TTA | yes | yes |

*Capture-simulating augmentation* (the single most important robustness
component) re-degrades every training image randomly: JPEG re-encoding at
quality 22–90, downscale–upscale (0.3–1.0×), Gaussian blur, moiré synthesis,
perspective warp, and photometric jitter. The public specialist disables it
to preserve the fragile digital artifacts the public test rewards; the robust
model relies on it to learn manipulation evidence that survives
print-and-capture.

*Early stopping* is the second robustness lever: with capture-aug on,
held-out-country FREUID degrades monotonically with training (0.025 at epoch
0 → 0.097 at epoch 2 on Mozambique-LOCO) as the network drifts from general
manipulation evidence toward memorizable digital artifacts. Without
capture-aug at 1024 px the effect reverses (0.151 → 0.033), indicating the
two regimes converge to different solution families.

### 3.3 What we tried that did not survive (mid-competition lineage)

Before the resolution finding, our best public model was a ConvNeXtV2-L
pipeline: domain-adaptive masked-autoencoder pretraining on FREUID images
(public 0.172 → 0.156), then 512 px fine-tuning (0.153). Approaches that
failed outright: frozen foundation-model probes (DINOv2/CLIP, 0.24–0.37),
soft-AUC ranking loss (0.37), REINFORCE fine-tuning (0.175), SRM
noise-residual stream (1.0), soft pseudo-labels (0.35), DANN
domain-adversarial training (LOCO 0.92 vs 0.40 baseline), and every
cross-architecture ensemble (consistent dilution). The final EffNetV2-M @1024
recipe supersedes the entire lineage on both axes.

## 4. Validation protocols (the compass)

The public LB is 5% of the test data and rewards transductive overfitting, so
all robustness decisions used local protocols:

### 4.1 Leave-One-Country-Out (LOCO)

Hold out one full country/type from training; evaluate the competition metric
on it. We ran LOCO on three countries (Mozambique, Benin, Mauritius) to avoid
tuning to a single held-out choice. Historical baselines: 0.40 (Mozambique)
and 0.39 (Benin) — the private-test scenario was genuinely hard for every
recipe before the resolution finding.

### 4.2 Seed-noise calibration

Identical 1-epoch runs differ by up to ±0.05 LOCO FREUID; we treated
sub-0.05 differences as inconclusive and re-ran candidates, and we used
probability-averaged seed ensembles (which beat the best individual seed:
0.131 vs 0.136) for the final robust model.

### 4.3 Capture canary

The 20 captured training images (held out of LOCO training runs that exclude
Mauritius, their majority country) probe print-and-capture robustness:
capture-aug models reach AUC 0.917 with clean separation; no-capture-aug
models degrade to 0.726 and over-flag captured genuines — the measured cost
of public-LB specialization, and the reason the portfolio has two models.

## 5. Results

### 5.1 Final models

| Model | Public LB | LOCO Mozambique | LOCO Benin | LOCO Mauritius | Canary AUC |
|---|---|---|---|---|---|
| 🔵 Public specialist @1024 | **0.02514** | 0.033 | 0.0101 | 0.0069 | 0.726 |
| 🔴 Robust ensemble @1024 | 0.11925 | **0.025** (ep 0) | — | 0.0156 | **0.917** |

### 5.2 The resolution ladder

Public LB (same recipe, resolution only): 384 → ~0.19 · 512 → 0.153 ·
768 → 0.123 · 896 → 0.034 · **1024 → 0.025**. Unseen-country (LOCO
Mozambique): 384 → 0.40 · 768 → 0.15–0.22 · **1024 → 0.03**. Native images
are up to 1585×1000: at 768 px, training destroys the resampling seams,
JPEG-grid mismatches, and interpolation traces that identify manipulation; at
1024 px (≈ native height) they reach the network intact. Returns were
*increasing*, not diminishing — evidence the decisive signal lives at the
finest scale. Verified as legitimate out-of-sample performance: no
public-test data in training, bimodal score distribution, inter-resolution
agreement r = 0.91, and LOCO transfer (this section).

### 5.3 Progression (public LB)

0.333 baseline → 0.291 leakage-safe splits → 0.207 aug+focal → 0.187 (+30k
IDNet) → 0.178 (+TTA) → 0.172 (pseudo-labels) → 0.156 (domain-MAE) → 0.153
(512 px) → 0.123 (EffNetV2-M 768 px) → 0.034 (896 px) → **0.025 (1024 px)**.

### 5.4 External data poisoning (negative result we consider a contribution)

Synthetic GenAI fraud generated from FREUID genuines collapses LOCO to 1.0:
the synthetic edits include held-out-country genuines re-labeled as fraud, so
the network learns "unfamiliar country ⇒ fraud" — the worst possible prior
for a private set of unseen countries. The mechanism is general: *synthetic
attack data built from in-domain genuines silently leaks country identity
into the fraud label.* Similarly, IDNet-scanned (print-captured European
documents) reduces the capture canary to chance (AUC 0.50): off-domain
capture statistics displace, rather than teach, the target capture
distribution.

### 5.5 Pseudo-labels and self-training limits

Public-test pseudo-labels helped while the teacher outperformed the student
(0.187 → 0.172 era). At the final level they consistently hurt: a 0.153
teacher drags a 0.123-recipe student to 0.175; even self-training with the
best available teacher (0.123) lands at 0.137. With 99.6% of pseudo-labels
above the confidence threshold, pseudo-labeling is effectively distillation
of the teacher's errors onto the test distribution.

## 6. Inference

Single pass over the input directory (`infer.py`): each decoded batch is
scored by every checkpoint and TTA view before the next batch loads, so
decode cost is amortized. bf16 autocast, channels-last, batch 32.

**Runtime** (measured, NVIDIA L40S 48 GB, decoding native-resolution JPEGs):
the 3-seed + hflip-TTA robust ensemble (6 forwards/image at 1024 px) runs at
**≈19 img/s** → a 135k-image hidden test set completes in **≈2 h**, well
inside the 6-hour single-A100 budget (the A100-80GB is ≥ the L40S on bf16
throughput). The single-model public specialist is ~3× faster (≈40 min).

## 7. Reproducibility

- **Code**: public repository (MIT license), frozen commit; layout and exact
  training commands in `README.md`. Total training compute for the final
  models: ~3 GPU-hours (1× L40S).
- **Weights**: 4 checkpoints (~216 MB each) attached as a GitHub Release
  (`final-models`).
- **Docker**: `docker/Dockerfile` implements the sandbox contract — reads
  `/data` (read-only, flat images), writes `/submissions/submission.csv`,
  runs with `--network none` (weights and code embedded; no downloads).
  Default entrypoint produces the robust-ensemble submission;
  `FREUID_MODEL=public` switches to the public specialist.
- **Hardware for verification**: any single NVIDIA GPU ≥24 GB (A100
  recommended); CPU fallback works but is slow.
- **Determinism note**: training used unfixed seeds where noted; the released
  checkpoints are the exact artifacts behind the final submissions, so
  verification is inference-only and deterministic up to bf16 nondeterminism
  (rank correlation ≥0.999 across reruns).

## 8. Acknowledgements

Built with PyTorch, torchvision, and timm (Apache-2.0). We thank the
organizers (Microblink Fraud Lab) for a competition design that rewards
genuine robustness — and for the forum guidance that shaped the portfolio
strategy.
