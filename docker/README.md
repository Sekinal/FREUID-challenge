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

| Kaggle final pick | Command | Output sha256 |
|---|---|---|
| `sub_bm1024` (public specialist) | `docker run --rm --network none --gpus all --shm-size 2g -e FREUID_MODEL=public -v <test>:/data:ro -v <out>:/submissions freuid2026-eliastsj` | *(added after private-test inference, July 13)* |
| `sub_slot2v3_1024_ens3` (robust ensemble) | same, with `-e FREUID_MODEL=robust` (default) | *(added after private-test inference, July 13)* |

Weights are frozen as of the `final-models` release; the flag is inference
orchestration only (allowed under the code-freeze rules).
