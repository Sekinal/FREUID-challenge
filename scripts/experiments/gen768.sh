#!/bin/bash
cd /root/freuid
export KAGGLE_API_TOKEN=$(cat .kaggle/access_token 2>/dev/null)
echo "[$(date)] generando CSV 768 (single-scale, batch 8)..."
python3 scripts/31_polish_submit.py --checkpoints runs/fusion_nofusion_all_res768.pt \
    --scales 1.0 --batch-size 8 --out submissions/sub_res768.csv > gen768_inner.log 2>&1
if [ -f submissions/sub_res768.csv ]; then
  echo "[$(date)] CSV OK, subiendo..."
  kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai -f submissions/sub_res768.csv -m "resolution 768 (warm-start res512) single-scale TTA" 2>&1 | grep -E "Successfully|error" | tail -1
  sleep 80
  kaggle competitions submissions -c the-freuid-challenge-2026-ijcai-ecai 2>/dev/null | head -3 | sed -E "s/  +/ | /g" | cut -c1-100
else
  echo "[$(date)] CSV FALLO:"; tail -4 gen768_inner.log
fi
echo "[$(date)] GEN768 DONE"
