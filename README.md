# FREUID Challenge 2026 — Starter / EDA

Exploratory data analysis scaffold for the
[FREUID Challenge 2026 (IJCAI-ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai) —
**next-generation identity-document fraud detection** across physical manipulations,
GenAI-driven multimodal edits, and print-and-capture forgeries.

This repo currently covers **data analysis & exploration only** (no modelling yet).
It is built to run on a GPU box and to **scale automatically** from the tiny public
sample to the full release.

## The task (from the public sample)

- **Goal:** predict `P(fraud)` per image `id`. `sample_submission.csv` is `id,label`
  with `label` a probability in `[0, 1]` → a ranking/AUC-style metric.
- **Labels** (`train_sample_labels.csv`): `id, image_path, label (0=genuine/1=fraud),
  is_digital (bool), type ("<COUNTRY>/<DOC_TYPE>")`, e.g. `MAURITIUS/ID`, `GUINEA/DL`.
- The public release ships only a **13-image `train_sample/`**; the test set is hidden.

## Key EDA findings (public sample)

| Finding | Detail |
|---|---|
| Class balance | 3 fraud / 10 genuine (23% fraud) |
| Strata | 5 countries (BENIN, EGYPT, GUINEA, MAURITIUS, MOZAMBIQUE); doc types DL (10) / ID (3); `is_digital` 7 / 6 |
| Integrity | 13/13 JPEG, RGB, 0 corrupt; ~840–1585 px wide, aspect ≈ 1.58 |
| **Near-duplicates** | **5 near-dup pairs (pHash ≤ 8), 3 with conflicting labels** → matched genuine↔tampered pairs / shared templates. **Group them into the same CV fold or folds leak.** |
| Embedding structure | CLIP ViT-B/32 features separate by `doc_type`/`country` but **not** by `label` (LOO-kNN ≈ 0.77, label silhouette ≈ 0) → fraud cues are subtle, likely need forensic/high-res features |

Full write-up with figures: **[`report/report.pdf`](report/report.pdf)**.

## Layout

```
freuid/            # package: config (paths/schema), io (loaders), viz (plot style)
scripts/           # numbered, each runnable via `uv run`
  00_download.py           # Kaggle download + extract -> data/
  01_inventory.py          # files, sizes, CSV schemas, submission format
  02_labels.py             # class balance + fraud rate across strata
  03_image_stats.py        # dims/aspect/mode/size + corruption check
  04_sample_grids.py       # per-label / per-doctype montages
  05_duplicates_leakage.py # pHash/dHash near-dups + train/test leakage scaffold
  06_embeddings_gpu.py     # GPU image embeddings (OpenCLIP)
  07_clustering_umap.py    # UMAP + KMeans/HDBSCAN, separability probes
  08_build_report.py       # assemble + compile the Typst report
artifacts/         # small JSON/parquet stats (committed)
figures/           # PNGs embedded by the report (committed)
report/            # report.typ + report.pdf
data/, embeddings/ # git-ignored (live on the box)
```

## Reproduce

Prereqs: [`uv`](https://docs.astral.sh/uv/), and [`typst`](https://typst.app/)
on `PATH` (for the PDF). Kaggle auth via the standalone token in
`~/.kaggle/access_token` or the `KAGGLE_API_TOKEN` env var.

```bash
uv sync                      # install dependencies (torch is CUDA cu12x)
bash scripts/run_eda.sh      # 00 -> 08, end to end
# or step by step:
uv run python scripts/00_download.py
uv run python scripts/01_inventory.py
# ...
uv run python scripts/08_build_report.py   # -> report/report.pdf
```

GPU is auto-detected (`06` uses CUDA when available; verified on an RTX PRO 6000
Blackwell). Override the encoder with `FREUID_EMB_MODEL` / `FREUID_EMB_PRETRAINED`.

## GPU server (team box)

SSH: `ssh root@216.81.248.172 -p 40299` — repo and full dataset live under
`/root/freuid`. Full setup: **[`docs/SERVER.md`](docs/SERVER.md)**.

## Notes

- Everything is schema-introspecting: when the full dataset lands, drop it under
  `data/` (or re-run `00`) and the same scripts produce the same report at scale.
- Modelling is intentionally out of scope for this pass.
