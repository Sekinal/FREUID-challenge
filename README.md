# FREUID Challenge 2026 — EDA + Modeling

End-to-end pipeline for the
[FREUID Challenge 2026 (IJCAI-ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai) —
**next-generation identity-document fraud detection** across physical manipulations,
GenAI-driven multimodal edits, and print-and-capture forgeries.

The repo covers **EDA → leakage-safe validation → a trainable baseline → Kaggle
submission**. It is schema-introspecting and **scales automatically** from the tiny
public sample to the full release (the GPU box has the full data).

## The task

- **Goal:** predict `P(fraud)` per image `id`. `sample_submission.csv` is `id,label`
  with `label` a probability in `[0, 1]`.
- **Metric:** FREUID — the harmonic combination of `AuDET` (area under the DET curve)
  and `APCER@1%BPCER`; **lower is better**. Implemented + tested in `freuid/metrics.py`.
- **Labels** (`train_labels.csv`): `id, image_path, label (0=genuine/1=fraud),
  is_digital (bool), type ("<COUNTRY>/<DOC_TYPE>")`.

## Dataset scale (full release, on the GPU box)

| Split | Images | Notes |
|---|---|---|
| `train/train/` | 69,352 (~15 GB) | labeled via `train_labels.csv` |
| `public_test/` | 7,821 (~1.7 GB) | scored for submission |
| `train_sample/` | 13 | the original public demo sample |

**Composition:** 5 document types — EGYPT/DL, GUINEA/DL, BENIN/DL, MOZAMBIQUE/DL,
MAURITIUS/ID (each ~13–16k images); fraud rate ≈ 0.42 (EGYPT/DL ≈ 0.50);
`is_digital` is True for all but 20 images. `data/` and `embeddings/` are git-ignored.

## Leakage-safe validation (the core of this repo)

Naive validation is badly misleading here: there are only **5 document types**, so
holding out whole types gives a handful of high-variance configurations, and the
training set contains **near-duplicate twins** (a genuine document and its tampered
copy, shared blank templates) that leak across folds. The pipeline addresses both:

- **Duplicate detection at scale** (`freuid/dedup.py`, `scripts/05`): 256-bit pHash
  (a 64-bit hash is too coarse — distinct same-template docs collide even at distance 0),
  with LSH banding (pigeonhole-complete; *fails closed* if a bucket is skipped) and
  union-find components. On the full train set: **10,240 near-dup pairs / 1,124
  components**, ~35% with conflicting labels (genuine↔tampered twins). It also flags
  **~1,505 train↔public_test near-duplicate pairs** — direct test-time leakage to
  exploit/account for.
- **Group-stratified splits** (`freuid/splits.py`, `scripts/09`): every image is grouped
  by its duplicate component, then `StratifiedGroupKFold` assigns **whole groups** to
  train/val/test and CV folds, balancing the `type × label` distribution. Result:
  **64,135 groups**, all three partitions at fraud rate **0.423**, CV fold fraud-rate
  **std = 0.0001**, and a hard `assert_no_leakage` (no group or id crosses a boundary).
- **Cross-type stress test** (leave-one-type-out) as a *secondary*, pessimistic signal
  for the type-shift scenario — group-aware so cross-type twins never leak.
- **Uncertainty + breakdowns** (`freuid/validation.py`, `freuid/metrics.bootstrap_metric`):
  stratified bootstrap confidence intervals, per-type metrics, and group CV summaries —
  because a single point estimate is not trustworthy on this data.

## Baseline model

`freuid/baseline.py` (+ `scripts/11`, `scripts/12`): pretrained timm `efficientnet_b2`
+ binary head, `pos_weight` for class imbalance. Best checkpoint is chosen by **AuDET**
(smooth) rather than FREUID (which folds in a single near-threshold operating point and
can swing to 1.0 between epochs).

**Tuned to saturate the A100:** bf16 autocast (no GradScaler; fixes silent fp16
underflow), `channels_last`, `torch.compile`, TF32 + cuDNN autotune, wide dataloader —
~100% GPU utilization during training.

## Layout

```
freuid/            # package
  config.py        # paths/schema; auto-selects full vs sample data
  io.py            # schema-agnostic loaders + image discovery
  data.py          # PyTorch DocumentDataset + transforms
  metrics.py       # FREUID bundle + bootstrap CIs
  dedup.py         # scalable exact/near-dup detection (256-bit pHash + LSH)
  splits.py        # leakage-safe StratifiedGroupKFold + LOTO + assertions
  validation.py    # holdout/CV/LOTO scoring, CIs, per-type, leakage checks
  baseline.py      # A100-tuned timm baseline (train/predict)
  viz.py
scripts/           # numbered, runnable via `uv run` (or .venv/bin/python on the box)
  00..08           # EDA: download → inventory → image stats → report
  05_duplicates_leakage.py   # full-scale dup + train/test leakage detection
  09_build_splits.py         # leakage-safe splits (--strategy stratified_group|type_holdout)
  10_validate_smoke.py       # validation wiring smoke (no model)
  11_train_baseline.py       # train baseline on the A100
  12_predict_baseline.py     # public_test -> submission CSV
tests/             # metrics, dedup (LSH completeness), splits (leakage-safety)
artifacts/         # small JSON/parquet stats (committed); splits/ git-ignored
data/ embeddings/ runs/ submissions/   # git-ignored (live on the box)
```

## Reproduce

Prereqs: [`uv`](https://docs.astral.sh/uv/) (or the box `.venv`) and
[`typst`](https://typst.app/) for the PDF. Kaggle auth via `~/.kaggle/access_token`.

```bash
uv sync
bash scripts/run_eda.sh                          # EDA 00..08
uv run python scripts/05_duplicates_leakage.py   # full-scale dedup (artifacts/duplicates.json)
uv run python scripts/09_build_splits.py         # leakage-safe splits
uv run python scripts/10_validate_smoke.py       # validate the validator
uv run python scripts/11_train_baseline.py --epochs 3 --batch-size 512   # A100
uv run python scripts/12_predict_baseline.py --checkpoint runs/baseline/best.pt
uv run python tests/test_metrics.py && uv run python tests/test_dedup.py && uv run python tests/test_splits.py
```

On the box (no `uv`): use `.venv/bin/python` in place of `uv run python`.

## GPU server (team box)

SSH: `ssh root@216.81.248.172 -p 40299` — repo and full dataset under `/root/freuid`.
Setup & coordination: **[`docs/SERVER.md`](docs/SERVER.md)**. The committed
`artifacts/`/`figures/`/`report/` EDA snapshots describe the **public sample**; re-run
`01`–`08` on the box to regenerate at full scale.

## Notes

- `config.py` auto-picks `train_labels.csv` + `train/train/` when present, else the sample.
- Kaggle token lives only in `~/.kaggle/` on the box; `LOCAL_CREDENTIALS.txt` is git-ignored.
