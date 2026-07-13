#!/bin/bash
# Ultimo escalon de resolucion publico: benchmax @1024 (sin cache, originales directos).
set -e
cd /root/freuid
unset FREUID_IMG_CACHE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "===== TRAIN benchmax @1024 seed 4 ====="
python3 -u scripts/24_train_fusion.py --train-all --no-fusion --loss bce \
  --img-size 1024 --no-capture-aug --batch-size 12 --workers 10 --epochs 3 --seed 4
echo "===== PREDICT + SUBMIT ====="
python3 -u scripts/25_predict_fusion.py --checkpoint runs/fusion_nofusion_all_noaug_s4.pt \
  --tta --batch-size 8 --workers 6 --out submissions/sub_bm1024.csv
kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai \
  -f submissions/sub_bm1024.csv -m "resolution ceiling test: effM@1024 no-capture-aug BCE 3ep seed4 TTA (native width ~1585)"
echo "BM1024_DONE_MARKER"
