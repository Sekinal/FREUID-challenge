#!/bin/bash
# Slot 2 FINAL: 2 semillas mas del train-all effM@768 heavy 1ep + CSV ensemble prob-avg de las 3.
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "===== SEMILLA B: train-all effM@768 heavy 1ep ====="
python3 -u scripts/30_train2.py --train-all \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy --epochs 1 --save-name slot2_es768_ep1_b

echo "===== SEMILLA C: train-all effM@768 heavy 1ep ====="
python3 -u scripts/30_train2.py --train-all \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy --epochs 1 --save-name slot2_es768_ep1_c

echo "===== CSV ENSEMBLE prob-avg 3 semillas (NO submit) ====="
python3 -u scripts/31_polish_submit.py \
  --checkpoints runs/slot2_es768_ep1.pt,runs/slot2_es768_ep1_b.pt,runs/slot2_es768_ep1_c.pt \
  --scales 1.0 --batch-size 8 --out submissions/sub_slot2_es768_ens3.csv

echo "PRIVATE6_DONE_MARKER"
