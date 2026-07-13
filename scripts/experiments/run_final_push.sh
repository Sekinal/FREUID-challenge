#!/bin/bash
# EMPUJON FINAL: +2 semillas slot2 (ens5) + benchmax@896 publico. 2 submits al final.
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
M=tf_efficientnetv2_m.in21k_ft_in1k

echo "===== SEMILLA D train-all @768 heavy 1ep ====="
python3 -u scripts/30_train2.py --train-all --backbone $M --img-size 768 --batch-size 24 \
  --workers 10 --loss bce --aug-strength heavy --epochs 1 --save-name slot2_es768_ep1_d
echo "===== SEMILLA E train-all @768 heavy 1ep ====="
python3 -u scripts/30_train2.py --train-all --backbone $M --img-size 768 --batch-size 24 \
  --workers 10 --loss bce --aug-strength heavy --epochs 1 --save-name slot2_es768_ep1_e

echo "===== CSV ENSEMBLE 5 semillas + SUBMIT (sanity) ====="
python3 -u scripts/31_polish_submit.py \
  --checkpoints runs/slot2_es768_ep1.pt,runs/slot2_es768_ep1_b.pt,runs/slot2_es768_ep1_c.pt,runs/slot2_es768_ep1_d.pt,runs/slot2_es768_ep1_e.pt \
  --scales 1.0 --batch-size 8 --out submissions/sub_slot2_es768_ens5.csv
kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai \
  -f submissions/sub_slot2_es768_ens5.csv -m "PRIVATE-slot v2: 5-seed ensemble effM@768 1ep capture-aug prob-avg"

echo "===== PUBLICO: benchmax @896 (no-capture-aug) ====="
python3 -u scripts/24_train_fusion.py --train-all --no-fusion --loss bce \
  --img-size 896 --no-capture-aug --batch-size 16 --workers 10 --epochs 3 --seed 3
python3 -u scripts/25_predict_fusion.py \
  --checkpoint runs/fusion_nofusion_all_noaug_s3.pt --tta --out submissions/sub_bm896.csv
kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai \
  -f submissions/sub_bm896.csv -m "benchmax capacity push: effM@896 no-capture-aug BCE 3ep seed3 TTA"

echo "FINAL_PUSH_DONE_MARKER"
