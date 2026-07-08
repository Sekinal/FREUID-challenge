# Auxiliary public datasets (Hugging Face & others)

For **FREUID Challenge 2026** pre-training. Competition rules allow public datasets
if they are genuinely public, license-compatible with open-source prizes, and cited
in the technical report.

**Primary training data:** Kaggle FREUID (`scripts/00_download.py`).

**Download heavy aux data on the GPU server only** (`/root/freuid/data/aux/`).

## Tier A — Document fraud / PAD (use first)

| Dataset | URL | Labels | License | Notes |
|---------|-----|--------|---------|-------|
| IDNet-2025 | [cactuslab/IDNet-2025](https://huggingface.co/datasets/cactuslab/IDNet-2025) | `positive`→0, `fraud5_*`/`fraud6_*`→1 | CC-BY-4.0 | ~125 GB HF; DHS-related |
| IDSpace | [cactuslab/IDSpace](https://huggingface.co/datasets/cactuslab/IDSpace) | same layout + mobile | CC-BY-SA-4.0 | Print/mobile capture |
| IDNet full | [Zenodo](https://zenodo.org/records/13854897) | 4 fraud types | See paper | ~598k images |
| SIDTD | [CVC](https://tc11.cvc.uab.es/datasets/SIDTD_1/) | 0/1 forged | CC-BY-SA 2.5 | Not on HF; best PAD match |

```bash
hf download cactuslab/IDNet-2025 ALB.tar.gz --repo-type dataset \
  --local-dir data/aux/idnet2025
```

## Tier B — Synthetic IDs (layout / OCR; weak fraud signal)

| Dataset | License | Use |
|---------|---------|-----|
| [Voxel51/synthetic_us_passports_easy](https://huggingface.co/datasets/Voxel51/synthetic_us_passports_easy) | Apache-2.0 | US passport VLM/OCR |
| [arnaudstiegler/synthetic_us_passports_hard](https://huggingface.co/datasets/arnaudstiegler/synthetic_us_passports_hard) | Apache-2.0 | Harder variants |
| [ud-biometrics/synthetic-usa-driver-license](https://huggingface.co/datasets/ud-biometrics/synthetic-usa-driver-license) | CC-BY-NC-ND-4.0 | DL layout |
| [thekfp/synthetic-cameroon-national-id-card-orc-dataset](https://huggingface.co/datasets/thekfp/synthetic-cameroon-national-id-card-orc-dataset) | — | Africa ID OCR |
| [Noaman/midv500](https://huggingface.co/datasets/Noaman/midv500) | MIT | MIDV captures |

**NC licenses:** verify team policy before use if competing for prizes.

## Tier C — Do not use for FREUID detector

- Tabular fraud (`electricsheepafrica/africa-identity-fraud-national-id`, etc.)
- Face PAD only (`UniqueData/presentation-attack-detection-2d-dataset`)
- Financial fraud tabular datasets
- [Akash076/synthetic_cards](https://huggingface.co/datasets/Akash076/synthetic_cards) — card says **not** for fraud detection

## Tier D — Avoid (project HANDOFF)

- DocTamper and generic natural-image tampering benchmarks

## Label mapping (IDNet / IDSpace)

```text
positive/                    → label 0 (genuine)
fraud5_inpaint_and_rewrite/  → label 1 (fraud)
fraud6_crop_and_replace/     → label 1 (fraud)
```

## Workflow

1. Pre-train on Tier A (optional Tier B).
2. Fine-tune on FREUID `data/` only.
3. Validate with FREUID metric (AuDET + APCER@1% BPCER), not AUC alone.

See also: [`HANDOFF.txt`](../../HANDOFF.txt) section 13 (Spanish, full search log).
