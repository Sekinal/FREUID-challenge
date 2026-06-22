# Results, Runs & Open Questions

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
| 53627140 | leakage-safe b2 (3 epoch) | **0.291** |

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
4. Scale on the A100 (EffNetV2-M is the current best backbone; try b4 / higher res / TTA /
   ensemble) once leakage-safe LOTO is the yardstick.
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
