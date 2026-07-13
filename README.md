# The FREUID Challenge 2026 — Solution (Team EliasTSJ + Sekinal)

Fraud detection on identity documents for [The FREUID Challenge 2026 (IJCAI-ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai).

## Results

| Model | Public LB | Cross-country (LOCO) | Capture canary (AUC) |
|---|---|---|---|
| **Public specialist** (`bm1024`) | **0.02514** | 0.033 Moz / 0.010 Benin / 0.007 Maur | 0.726 |
| **Robust / private** (`slot2v3` 3-seed ensemble) | 0.11925 | **0.025** (Mozambique, epoch 0) | **0.917** |

Both final submissions share the same backbone (EffNetV2-M) and training data
(official FREUID train set only — no external data). They differ in exactly two
training flags, which specialize them for the two announced test regimes:

| | Public specialist | Robust model |
|---|---|---|
| Resolution | 1024 | 1024 |
| Capture-simulating augmentation | **off** (preserve fragile digital artifacts) | **heavy** (JPEG recompress, blur, moiré, perspective) |
| Epochs | 3 | **1** (early stopping — more epochs memorize fragile artifacts) |
| Seeds | 1 | 3, probability-averaged |

### Key findings (see the technical report for details)

1. **Resolution is the universal lever.** Fraud edits leave pixel-scale traces
   (resampling seams, JPEG-grid mismatches). Downscaling to 384–768 destroys
   them; at 1024 (near-native) both in-distribution AND cross-country scores
   improve by an order of magnitude (public 0.123→0.025; unseen-country
   0.40→0.01–0.03). The traces generalize across countries — templates do not.
2. **Early stopping controls the robustness trade-off.** With capture-aug,
   epoch 0 generalizes best (0.025 vs 0.097 at epoch 2 on a held-out country).
3. **External data poisoned everything it touched** (measured, then removed):
   IDNet-scanned killed real-capture detection (canary AUC 0.50); GenAI
   synthetic fraud made from FREUID genuines taught "unknown country = fraud"
   (LOCO 1.0); pseudo-labels dragged the student toward a weaker teacher.
4. **Leave-One-Country-Out (LOCO) validation** — not the public LB — was the
   compass for every robustness decision, mirroring the announced private-test
   design (unseen countries + print-and-capture).

## Repository layout

```
freuid/          core package: data pipeline, transforms, model, metrics
scripts/         numbered experiment scripts (EDA → training → submission)
infer.py         standalone inference entrypoint (used by the Docker image)
docker/          Dockerfile + build instructions (no-network sandbox contract)
docs/            notes and experiment logs
```

## Reproducing the final models

Hardware used: 1× NVIDIA L40S 48 GB, ~3 h total training. Data: official
competition data only, extracted to `data/extracted/`.

```bash
# 1) Public specialist (single seed, 3 epochs, no capture-aug, 1024 px)
python3 scripts/24_train_fusion.py --train-all --no-fusion --loss bce \
    --img-size 1024 --no-capture-aug --batch-size 12 --workers 10 --epochs 3 --seed 4

# 2) Robust model (3 seeds, 1 epoch, heavy capture-aug, 1024 px)
for S in a b c; do
python3 scripts/30_train2.py --train-all \
    --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 1024 --batch-size 12 \
    --workers 10 --loss bce --aug-strength heavy --epochs 1 --save-name slot2v3_1024_$S
done
```

## Inference

```bash
python3 infer.py --data-dir /path/to/images --out submission.csv \
    --checkpoints weights/slot2v3_1024_a.pt,weights/slot2v3_1024_b.pt,weights/slot2v3_1024_c.pt \
    --batch-size 32 --workers 8
```

Reads a flat directory of images (`.jpeg/.jpg/.png/.webp/.bmp/.tif/.tiff`),
writes `id,label` CSV (id = filename without extension, label = fraud score).
Checkpoints are probability-averaged with horizontal-flip TTA.

Final model weights are published as a GitHub Release (see Releases tab).

## Docker (no-network sandbox)

See `docker/README.md`. The image embeds the weights and code; it reads
`/data/` (read-only) and writes `/submissions/submission.csv`, with
`--network none`, per the competition reproducibility contract.

## External resources credited

- [timm](https://github.com/huggingface/pytorch-image-models) — ConvNeXtV2 / EfficientNetV2 backbones (Apache-2.0)
- [PyTorch / torchvision](https://pytorch.org)
- Explored but **not used** in the final models: IDNet-2025 (CC-BY-4.0),
  FantasyID, InsightFace `inswapper`, SDXL-inpainting (all documented in the report)

## Code-freeze compliance

Timeline anchors (all organizer-verifiable):

- **Final model weights frozen 2026-07-08 17:12 UTC** — the `final-models`
  GitHub Release asset timestamps.
- **Selected final submissions produced 2026-07-07 / 2026-07-08** (Kaggle
  submission timestamps) — five days before the private test release.
- **Last change to model / training / inference code: commit `19630e2`,
  pushed 2026-07-12 03:07 UTC** — before the private test release
  (2026-07-13 07:02 UTC).

Commits after the private test release only add documentation, result
artifacts, and archival copies of pre-freeze experiment launcher scripts
(`scripts/experiments/`, runs corroborated by Kaggle submission timestamps).
Verify with one command — no solution code changed after `19630e2`:

```bash
git diff 19630e2 HEAD --stat -- freuid/ infer.py docker/Dockerfile \
  docker/entrypoint.sh 'scripts/*.py' 'scripts/*.sh'   # additions only, 0 modifications
```

## License

MIT — see [LICENSE](LICENSE).
