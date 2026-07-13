#!/bin/bash
# PUBLICO ronda 3: self-training con el MEJOR profesor (bm_s1 0.12267 ensena, alumno re-entrena).
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "===== 1) Pseudo frescos con profesor bm_s1 (0.12267) ====="
python3 -u scripts/27_make_pseudolabels.py --checkpoint runs/fusion_nofusion_all_noaug_s1.pt \
  --lo 0.05 --hi 0.95 --batch-size 16 --out artifacts/pseudo_bm.csv

echo "===== 2) TRAIN: benchmax s2 + pseudo_bm ====="
python3 -u scripts/24_train_fusion.py --train-all --no-fusion --loss bce \
  --img-size 768 --no-capture-aug --batch-size 24 --workers 10 --epochs 3 --seed 2 \
  --extra-csv artifacts/pseudo_bm.csv

echo "===== 3) PREDICT + SUBMIT ====="
python3 -u scripts/25_predict_fusion.py \
  --checkpoint runs/fusion_nofusion_all_noaug_s2_pl.pt --tta --out submissions/sub_bmpl2.csv
kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai \
  -f submissions/sub_bmpl2.csv -m "self-training round2: teacher bm_s1(0.12267) -> student benchmax s2 + pseudo, effM@768 no-capture-aug"

echo "PUBLIC3_DONE_MARKER"
