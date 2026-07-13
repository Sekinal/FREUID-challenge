#!/bin/bash
# Confirmacion early-stop: effM @768 heavy SIN extras (los extras eran el veneno).
# Referencias: con extras ep0=0.2914/ep1=0.48/ep2=0.60; baseline cnx+MAE 0.4023.
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "===== A: LOTO MOZAMBIQUE effM@768 heavy SIN extras, 3 ep (curva) ====="
python3 -u scripts/30_train2.py --loto-type "MOZAMBIQUE/DL" \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy \
  --epochs 3 --save-name es_mozLOTO_768

echo "===== B: LOTO MAURITIUS idem, 1 EPOCA (checkpoint=ep0) ====="
python3 -u scripts/30_train2.py --loto-type "MAURITIUS/ID" \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy \
  --epochs 1 --save-name es_maurLOTO_768_ep1

echo "===== CANARY sobre el modelo de 1 epoca ====="
python3 scripts/eval_canary.py --checkpoint runs/es_maurLOTO_768_ep1.pt || true

echo ""
echo "########## RESUMEN ##########"
grep -hE "PROXY|LOTO-test" /root/private4.log | tail -8 || true
echo "PRIVATE4_DONE_MARKER"
