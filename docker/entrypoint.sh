#!/bin/bash
set -e
if [ "$FREUID_MODEL" = "public" ]; then
  CKPTS="/app/weights/fusion_nofusion_all_noaug_s4.pt"
else
  CKPTS="/app/weights/slot2v3_1024_a.pt,/app/weights/slot2v3_1024_b.pt,/app/weights/slot2v3_1024_c.pt"
fi
echo "[entrypoint] FREUID_MODEL=$FREUID_MODEL"
exec python3 /app/infer.py \
  --data-dir /data \
  --out /submissions/submission.csv \
  --checkpoints "$CKPTS" \
  --batch-size "${FREUID_BATCH}" \
  --workers "${FREUID_WORKERS}"
