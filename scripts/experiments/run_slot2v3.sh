#!/bin/bash
# SLOT 2 v3 (hibrido 1024): canary-check + 3 semillas train-all 1ep + ensemble CSV + sanity submit.
set -e
cd /root/freuid
unset FREUID_IMG_CACHE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=tf_efficientnetv2_m.in21k_ft_in1k

echo "===== 1) LOTO MAURITIUS hibrido @1024 heavy 1ep (para canary) ====="
python3 -u scripts/30_train2.py --loto-type "MAURITIUS/ID" \
  --backbone $M --img-size 1024 --batch-size 12 --workers 10 \
  --loss bce --aug-strength heavy --epochs 1 --save-name hib_maurLOTO_1024_ep1

echo "===== 2) CANARY del hibrido ====="
python3 scripts/eval_canary.py --checkpoint runs/hib_maurLOTO_1024_ep1.pt --batch-size 4 || true

echo "===== 3) SEMILLAS A/B/C train-all hibrido @1024 heavy 1ep ====="
for S in a b c; do
  echo "--- semilla $S ($(date -u +%H:%M)) ---"
  python3 -u scripts/30_train2.py --train-all \
    --backbone $M --img-size 1024 --batch-size 12 --workers 10 \
    --loss bce --aug-strength heavy --epochs 1 --save-name slot2v3_1024_$S
done

echo "===== 4) CSV ENSEMBLE + SUBMIT sanity ====="
python3 -u scripts/31_polish_submit.py \
  --checkpoints runs/slot2v3_1024_a.pt,runs/slot2v3_1024_b.pt,runs/slot2v3_1024_c.pt \
  --scales 1.0 --batch-size 4 --out submissions/sub_slot2v3_1024_ens3.csv
kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai \
  -f submissions/sub_slot2v3_1024_ens3.csv -m "PRIVATE-slot v3: 3-seed ensemble effM@1024 1ep capture-aug prob-avg (LOCO Moz 0.025 ep0)"

echo ""
echo "########## RESUMEN ##########"
grep -hE "PROXY|CANARY|separation|AUC|TRIPWIRE" /root/slot2v3.log | tail -10 || true
echo "SLOT2V3_DONE_MARKER"
