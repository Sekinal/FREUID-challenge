# Docker reproducibility image

Implements the FREUID 2026 no-network sandbox contract: reads a flat directory
of images mounted at `/data` (read-only), writes `/submissions/submission.csv`
(`id,label`; id = filename without extension; label = fraud score, higher =
more fraudulent). No network access is required at any point.

## Build

1. Download the released weights (GitHub Release `final-models`) into
   `docker/weights/`:
   - `slot2v3_1024_a.pt`, `slot2v3_1024_b.pt`, `slot2v3_1024_c.pt` (robust ensemble)
   - `fusion_nofusion_all_noaug_s4.pt` (public specialist)
2. From the repo root:

```bash
docker build -t freuid2026-eliastsj -f docker/Dockerfile .
```

## Run (sandbox)

```bash
docker run --rm --network none --gpus all --shm-size 2g \
  -v /path/to/test_images:/data:ro \
  -v /path/to/output:/submissions \
  freuid2026-eliastsj
```

Default produces the **robust ensemble** submission (primary). To produce the
public-specialist submission instead:

```bash
docker run --rm --network none --gpus all --shm-size 2g \
  -e FREUID_MODEL=public \
  -v /path/to/test_images:/data:ro -v /path/to/output:/submissions \
  freuid2026-eliastsj
```

## Runtime

Measured throughput (see technical report): the default 3-model + hflip-TTA
ensemble at 1024 px processes the full hidden test set well within the 6-hour
single-A100 budget (details and exact numbers in the report's Reproducibility
section). `FREUID_BATCH` / `FREUID_WORKERS` can be tuned if needed.

## Final submissions mapping (per host guidance, forum thread 723991)

| Kaggle final pick | Command | Submitted-file sha256 |
|---|---|---|
| `FINAL_public_bm1024.csv` (public specialist, submitted 2026-07-13 18:50:38 UTC) | `docker run --rm --network none --gpus all --shm-size 2g -e FREUID_MODEL=public -v <test>:/data:ro -v <out>:/submissions freuid2026-eliastsj` | `5f83120d681bcfa555b3c8604aa23fe77ef4c776c4968ef94966a3207536ddb6` |
| `FINAL_robust_slot2v3.csv` (robust ensemble, submitted 2026-07-13 18:50:40 UTC) | same, with `-e FREUID_MODEL=robust` (default) | `05d694281f17f8cfc9383534540cafbfbbe802d40bbe645c80727a9544abd28a` |

Weights are frozen as of the `final-models` release; the flag is inference
orchestration only (allowed under the code-freeze rules).

## Reproducibility tolerance

The sha256 above identifies the exact file we submitted to Kaggle. A
re-run of the container reproduces it **numerically, not bitwise**: inference
uses bf16 autocast with cuDNN autotuning (`cudnn.benchmark`), which is
non-deterministic across runs and GPU models. Measured on back-to-back
container runs (same GPU, same inputs): per-image score drift
max |Δ| ≈ 1.5e-4, mean ≈ 8e-7; with different DataLoader settings or
hardware, max |Δ| ≈ 1e-3. Score distributions are strongly bimodal, so
this drift is orders of magnitude below the decision scale and does not
measurably change the FREUID/AuDET metric.

If the container is launched without `--shm-size 2g`, the entrypoint
detects the small default `/dev/shm` and falls back to `FREUID_WORKERS=0`
(slower decode, same outputs) rather than crashing.
