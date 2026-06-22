# FREUID 2026 — Agent Findings & Working Notes

Knowledge base for anyone (human or agent) picking up this repo. Written from the
full-dataset work on the GPU box (`/root/freuid`, A100-80GB).

## TL;DR

- **The competition is won on out-of-distribution generalization, not in-distribution accuracy.**
  The host confirmed the private test contains **2 document types unseen in train/public**,
  plus an emphasis on **captured / print-and-capture** images. In-distribution scores
  (and the public leaderboard) are misleading.
- Our training data is **99.97% digital** (`is_digital=True` for all but 20 of 69,352).
  The private set emphasizes *captured* images → a large **digital→captured domain shift**
  stacked on top of the unseen-type shift.
- A vanilla EfficientNet-b2 gets **FREUID ≈ 0.0001 in-distribution** (near-perfect) but only
  **0.291 on the public LB** (same 5 types, held-out, even *with* ~19% leaked twins) — a ~0.29
  gap that is the whole game (the digital→captured shift, confirmed by a real submission).
- Validation is now **leakage-safe** (group-stratified by near-duplicate component) and the
  FREUID metric is **vectorized (~1130× faster, bit-identical)**.
- A second-box experiment line (branch **`aux-modeling`**) found the **dominant lever for OOD:
  mixing in external fraud data (IDNet-2025)** — a cross-country no-aux model scores 0.98,
  with 30k IDNet aux it drops to **0.3556**. Best backbone so far: EffNetV2-M + domain aug +
  focal. The two branches should be unified (see [`04`](04-results-and-open-questions.md)).
- **Hand-crafted forensic features (statistical noise analysis) tested and *do not* beat the
  vision models on the OOD axis** (`freuid/forensics.py`, 66 features): in-distribution AUC
  **0.995** but leave-one-type-out AUC **0.452** (below chance) — fraud signatures are
  type-specific, so raw forensics → GBM collapse on unseen types. ELA is the strongest family
  (within-type AUC up to 0.995) but no feature is universal. Only credible use is **fusion**
  into the CNN, not a standalone model. Details in [`04`](04-results-and-open-questions.md).

## Key numbers

| Thing | Value |
|---|---|
| Train images | 69,352 (5 types, ~0.42 fraud) |
| Public test | 7,821 (= public LB set) |
| Private test | 134,997 (hidden; **2 unseen types**, captured emphasis) |
| `is_digital=True` | 69,332 / 69,352 (99.97%) |
| Near-dup pairs (256-bit pHash, thr 10) | 10,240 / 1,124 components |
| Train↔public_test near-dup leaks | ~1,505 (~19% of public test) |
| Split groups | 64,135 (all partitions fraud-rate 0.423) |
| Baseline in-dist val/test FREUID | ~0.0001 |
| Baseline **public LB** (same types, held-out) | **0.291** (0.29127; was 0.333) |
| → in-dist vs public gap | ~0.29 (the digital→captured shift, made real) |
| Best **cross-country OOD** FREUID (`aux-modeling`) | **0.3556** (EffNetV2-M + domain aug + focal + 30k IDNet aux) |
| No-aux cross-country FREUID (ablation) | 0.9809 → **IDNet aux is the dominant lever** |
| **Best public LB** (EffNetV2-M all-5 + aug + focal + IDNet aux) | **0.20744 → rank 59/109** (was ~86 at 0.291) |
| Forensic features (66) in-dist vs LOTO AUC | 0.995 vs **0.452** → not OOD-robust standalone (ELA strongest) |

## Documents

- [`01-dataset-and-leakage.md`](01-dataset-and-leakage.md) — data shape, near-dup detection, leakage.
- [`02-validation.md`](02-validation.md) — leakage-safe splits, LOTO, metric vectorization.
- [`03-competition-and-strategy.md`](03-competition-and-strategy.md) — host's OOD design, public/private, what to build.
- [`04-results-and-open-questions.md`](04-results-and-open-questions.md) — runs, scores, TODO.

## Branches

- **`main`** — leakage-safe stratified-group splits, LOTO/`iter_type_holdout` support,
  vectorized metric, tests, these docs. The solid *infrastructure*. Public LB 0.291.
- **`aux-modeling`** — second-box line of work, forked from `7d8aeb4`. IDNet-2025 aux
  mix-in, multi-arch sweep, focal loss, **cross-country OOD eval** (real 0.3556). The
  better *experiments*, on older (non-leakage-safe) splits. See
  [`04`](04-results-and-open-questions.md). Next step is to unify the two.

## Environment

- Boxes (both A100-80GB, run with `.venv/bin/python`, no `uv`):
  - `ssh root@154.54.100.217 -p 40299` — **current** working box (`aux-modeling`), full data
    + IDNet aux at `/root/freuid`.
  - `ssh root@216.81.248.172 -p 40299` — original box (`main` work). May be retired.
- Repo: `github.com/Sekinal/FREUID-challenge` (push from local with the user's creds).
