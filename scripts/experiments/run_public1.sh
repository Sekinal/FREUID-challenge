#!/bin/bash
# PUBLICO "fusion de linajes": effM@768 + domain-MAE (mae2_effM) + pseudo frescos del res512.
# Meta: bajar del 0.15336 (sub_res512). 1 slot Kaggle al final.
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "===== 1) Pseudo-labels frescos (profesor = res512 0.15336) ====="
python3 -u scripts/27_make_pseudolabels.py --checkpoint runs/fusion_nofusion_all_res512.pt \
  --lo 0.05 --hi 0.95 --batch-size 32 --out artifacts/pseudo_res512.csv

echo "===== 2) TRAIN: effM@768 warm-start mae2_effM + pseudo, 3 ep ====="
python3 -u scripts/30_train2.py --train-all \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --init-from runs/mae2_effM.pt \
  --img-size 768 --batch-size 24 --workers 10 --loss bce --aug-strength default \
  --extra-csv artifacts/pseudo_res512.csv \
  --epochs 3 --save-name pub_effM768_maePL

echo "===== 3) PREDICT + SUBMIT (1 slot) ====="
python3 -u scripts/31_polish_submit.py --checkpoints runs/pub_effM768_maePL.pt \
  --scales 1.0 --batch-size 8 --out submissions/sub_pub_effM768_maePL.csv \
  --submit -m "effM@768 + domain-MAE warmstart + fresh pseudo(res512 teacher) 3ep TTA"

echo "PUBLIC1_DONE_MARKER"
