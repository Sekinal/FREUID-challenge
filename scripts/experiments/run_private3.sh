#!/bin/bash
# Ablacion del colapso rc2: aislar heavy-aug vs extra_clean sobre cnxV2-L+MAE @384.
# Referencia conocida: default aug, sin extras = LOTO-Moz 0.4023 (loto_baseline).
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "===== A1: default aug + extra_clean (aisla EXTRAS) ====="
python3 -u scripts/30_train2.py --loto-type "MOZAMBIQUE/DL" \
  --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from runs/mae_cnxv2L.pt \
  --img-size 384 --batch-size 32 --workers 10 \
  --loss bce --aug-strength default --extra-csv artifacts/extra_clean.csv \
  --epochs 3 --save-name ab_extras_mozLOTO

echo "===== A2: heavy aug, SIN extras (aisla HEAVY) ====="
python3 -u scripts/30_train2.py --loto-type "MOZAMBIQUE/DL" \
  --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from runs/mae_cnxv2L.pt \
  --img-size 384 --batch-size 32 --workers 10 \
  --loss bce --aug-strength heavy \
  --epochs 3 --save-name ab_heavy_mozLOTO

echo ""
echo "########## RESUMEN ABLACION (ref: 0.4023) ##########"
grep -hE "PROXY|LOTO-test" /root/private3.log | tail -10 || true
echo "PRIVATE3_DONE_MARKER"
