#!/bin/bash
# Final slot 2: confirmar LOTO 0.152 (variacion natural de semilla) + train-all 1 epoca + CSV.
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "===== A: CONFIRMACION LOTO MOZAMBIQUE effM@768 heavy 1ep (run 2) ====="
python3 -u scripts/30_train2.py --loto-type "MOZAMBIQUE/DL" \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy \
  --epochs 1 --save-name es_mozLOTO_768_ep1_v2

echo "===== B: TRAIN-ALL definitivo effM@768 heavy 1ep (SLOT 2) ====="
python3 -u scripts/30_train2.py --train-all \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy \
  --epochs 1 --save-name slot2_es768_ep1

echo "===== C: PREDICT (TTA single-scale, NO submit) ====="
python3 -u scripts/31_polish_submit.py --checkpoints runs/slot2_es768_ep1.pt \
  --scales 1.0 --batch-size 8 --out submissions/sub_slot2_es768.csv

echo ""
echo "########## RESUMEN ##########"
grep -hE "PROXY|LOTO-test" /root/private5.log | tail -6 || true
echo "PRIVATE5_DONE_MARKER"
