#!/bin/bash
# PUBLICO revancha: benchmax exacto (no-capture-aug, effM@768, BCE, s1) + pseudo res512.
# Unica variable nueva vs bm_s1_solo (0.12267): los pseudo-labels.
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "===== TRAIN: benchmax s1 + pseudo_res512 ====="
python3 -u scripts/24_train_fusion.py --train-all --no-fusion --loss bce \
  --img-size 768 --no-capture-aug --batch-size 24 --workers 10 --epochs 3 --seed 1 \
  --extra-csv artifacts/pseudo_res512.csv

echo "===== PREDICT + SUBMIT ====="
python3 -u scripts/25_predict_fusion.py \
  --checkpoint runs/fusion_nofusion_all_noaug_s1_pl.pt --tta --out submissions/sub_bmpl_s1.csv
kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai \
  -f submissions/sub_bmpl_s1.csv -m "benchmax s1 recipe + fresh pseudo(res512 teacher): effM@768 no-capture-aug BCE 3ep TTA"

echo "PUBLIC2_DONE_MARKER"
