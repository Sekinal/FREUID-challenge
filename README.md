# FREUID Challenge 2026 — EDA + Modeling

End-to-end pipeline for the
[FREUID Challenge 2026 (IJCAI-ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai) —
**next-generation identity-document fraud detection** across physical manipulations,
GenAI-driven multimodal edits, and print-and-capture forgeries.

The repo covers **EDA → group-aware validation → a trainable baseline → Kaggle
submission**. It is schema-introspecting and **scales automatically** from the tiny
public sample to the full release (this box has the full data).

## The task

- **Goal:** predict `P(fraud)` per image `id`. `sample_submission.csv` is `id,label`
  with `label` a probability in `[0, 1]`.
- **Metric:** FREUID — the harmonic combination of `AuDET` (area under the DET curve)
  and `APCER@1%BPCER`; **lower is better**. Implemented in `freuid/metrics.py`.
- **Labels** (`train_labels.csv`): `id, image_path, label (0=genuine/1=fraud),
  is_digital (bool), type ("<COUNTRY>/<DOC_TYPE>")`, e.g. `MAURITIUS/ID`, `GUINEA/DL`.

## Dataset scale (full release, on the GPU box)

| Split | Images | Notes |
|---|---|---|
| `train/train/` | 69,352 (~15 GB) | labeled via `train_labels.csv` |
| `public_test/` | 7,821 (~1.7 GB) | scored for submission |
| `train_sample/` | 13 | the original public demo sample |

`data/` and `embeddings/` are git-ignored and live only on the box.

## Key EDA findings (public sample)

| Finding | Detail |
|---|---|
| Class balance | 3 fraud / 10 genuine (23% fraud) |
| Strata | 5 countries (BENIN, EGYPT, GUINEA, MAURITIUS, MOZAMBIQUE); doc types DL / ID; `is_digital` split |
| Integrity | 13/13 JPEG, RGB, 0 corrupt; ~840–1585 px wide, aspect ≈ 1.58 |
| **Near-duplicates** | **5 near-dup pairs (pHash ≤ 8), 3 with conflicting labels** → matched genuine↔tampered pairs / shared templates. **Group them into the same CV fold or folds leak.** |
| Embedding structure | CLIP ViT-B/32 features separate by `doc_type`/`country` but **not** by `label` → fraud cues are subtle, likely need forensic/high-res features |

> The committed `artifacts/`, `figures/`, and `report/report.pdf` describe the **public
> sample**; re-run scripts `01`–`08` on the box to regenerate them at full scale.

Full write-up: **[`report/report.pdf`](report/report.pdf)**.

## Modeling & validation

- **Group-aware splits** (`freuid/splits.py`, `scripts/09_build_splits.py`): hold out
  whole document `type` groups (never random rows); union-find keeps near-duplicate ids
  in the same partition; greedy stratified assignment by size + fraud rate; `GroupKFold`
  on `type_component`. Outputs under `artifacts/splits/` (git-ignored — large &
  regenerable).
- **Metrics** (`freuid/metrics.py`): DET curve, AuDET, APCER@1%BPCER, EER, FREUID.
  Covered by `tests/test_metrics.py`.
- **Baseline** (`freuid/baseline.py`, `scripts/11_train_baseline.py`): pretrained timm
  `efficientnet_b2` + binary head, AMP, `pos_weight` for class imbalance. Best checkpoint
  is selected by **AuDET** (smooth) rather than FREUID (which folds in a single
  near-threshold operating point and can swing to 1.0 between epochs). TF32 + cuDNN
  autotuning are enabled for the A100.
- **Predict** (`scripts/12_predict_baseline.py`): scores `public_test` → submission CSV.

**Current baseline (efficientnet_b2, 3 epochs, full data):** val FREUID ≈ 0.017,
**test FREUID ≈ 0.41**. The large val↔test gap is expected: there are only ~5 distinct
`type` components, so holding out whole types makes the held-out metric high-variance.
Next steps live in `docs/`.

## Layout

```
freuid/            # package
  config.py        # paths/schema; auto-selects full vs sample data
  io.py            # schema-agnostic loaders + image discovery
  data.py          # PyTorch DocumentDataset + transforms
  metrics.py       # FREUID metric bundle
  splits.py        # group-aware train/val/test + CV folds
  validation.py    # validation pipeline (holdout + group CV)
  baseline.py      # timm baseline train / predict
  viz.py           # plot style
scripts/           # numbered, each runnable via `uv run`
  00_download.py … 08_build_report.py   # EDA: download → inventory → report
  09_build_splits.py        # group-aware splits + CV manifests
  10_validate_smoke.py      # wiring smoke test (constant scorer)
  11_train_baseline.py      # train the EfficientNet baseline
  12_predict_baseline.py    # public_test -> submission CSV
tests/             # pytest-style; also runnable as plain scripts
artifacts/         # small JSON/parquet stats (committed); splits/ ignored
figures/  report/  # PNGs + Typst report (public-sample)
data/  embeddings/ runs/  submissions/   # git-ignored (live on the box)
```

## Reproduce

Prereqs: [`uv`](https://docs.astral.sh/uv/) (or the existing `.venv` on the box), and
[`typst`](https://typst.app/) for the PDF. Kaggle auth via `~/.kaggle/access_token`
or `KAGGLE_API_TOKEN`.

```bash
uv sync                              # install deps (torch is CUDA cu12x)
bash scripts/run_eda.sh              # EDA: 00 -> 08
uv run python scripts/09_build_splits.py
uv run python scripts/11_train_baseline.py        # full train (A100)
uv run python scripts/11_train_baseline.py --max-train 2000   # quick smoke
uv run python scripts/12_predict_baseline.py --checkpoint runs/baseline/best.pt
uv run python tests/test_metrics.py  # or: uv run python -m pytest tests/
```

On the box (no `uv`): swap `uv run python` for `.venv/bin/python`.

## GPU server (team box)

SSH: `ssh root@216.81.248.172 -p 40299` — repo and full dataset under
`/root/freuid`. Full setup & coordination: **[`docs/SERVER.md`](docs/SERVER.md)**.

## Notes

- Everything is schema-introspecting: `config.py` auto-picks `train_labels.csv` +
  `train/train/` when present, else the sample. The same scripts run at either scale.
- The Kaggle token lives only in `~/.kaggle/` on the box; `LOCAL_CREDENTIALS.txt` is
  git-ignored and never committed.
