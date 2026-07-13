#!/bin/bash
set -e
if [ "$FREUID_MODEL" = "public" ]; then
  CKPTS="/app/weights/fusion_nofusion_all_noaug_s4.pt"
else
  CKPTS="/app/weights/slot2v3_1024_a.pt,/app/weights/slot2v3_1024_b.pt,/app/weights/slot2v3_1024_c.pt"
fi
echo "[entrypoint] FREUID_MODEL=$FREUID_MODEL"
# DataLoader workers pass tensors through /dev/shm; Docker's default is 64MB.
# If the container was launched without --shm-size, fall back to in-process
# decoding (workers=0) instead of crashing mid-run.
SHM_KB=$(df -k /dev/shm 2>/dev/null | awk 'NR==2 {print $2}')
if [ -n "$SHM_KB" ] && [ "$SHM_KB" -lt 1048576 ] && [ "${FREUID_WORKERS}" != "0" ]; then
  echo "[entrypoint] /dev/shm is ${SHM_KB}KB (<1GB): forcing FREUID_WORKERS=0 (slower decode, same output)"
  FREUID_WORKERS=0
fi
exec python3 /app/infer.py \
  --data-dir /data \
  --out /submissions/submission.csv \
  --checkpoints "$CKPTS" \
  --batch-size "${FREUID_BATCH}" \
  --workers "${FREUID_WORKERS}"
