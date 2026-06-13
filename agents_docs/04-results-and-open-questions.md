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

## Leave-one-type-out (LOTO)

- Smoke (random scorer) wired and working: freuid_mean ≈ 0.98 (chance), per-type + summary.
- **TODO / in progress:** real LOTO = 5 retrains (train on 4 types, score the held type) with
  the capture-augmented model → the honest unseen-type number. Script: `scripts/13`.

## Open questions / next steps

1. **Capture-robustness augmentation** (done? see `freuid/data.py`) — the main lever for the
   digital→captured private shift we cannot directly validate.
2. **Real LOTO numbers** with the augmented model — the OOD headline metric.
3. Scale the model to use the A100 (b4 / higher res / TTA / ensemble) once LOTO is the yardstick.
4. When private images drop: re-predict and resubmit the full 142,818.
5. Consider sourcing/synthesizing **captured-style** examples (the 20 `is_digital=False` rows
   are too few) to validate the capture shift directly.

## Reproduce

```bash
.venv/bin/python scripts/05_duplicates_leakage.py      # dedup + leakage (artifacts/duplicates.json)
.venv/bin/python scripts/09_build_splits.py            # leakage-safe splits
.venv/bin/python scripts/11_train_baseline.py --epochs 3 --batch-size 512
.venv/bin/python scripts/12_predict_baseline.py --checkpoint runs/baseline/best.pt
.venv/bin/python scripts/13_leave_one_type_out.py      # OOD: 5 retrains
.venv/bin/python tests/test_metrics.py && .venv/bin/python tests/test_dedup.py && .venv/bin/python tests/test_splits.py
```
