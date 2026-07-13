#!/bin/bash
# DIA 1 empujon final: barrido de receta OOD en Mozambique-LOCO, 1 epoca, una variable por run.
# Anclas previas (misma receta lr1e-4 heavy @768 M): 0.1520 y 0.2252 (ruido semilla ±0.05).
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
M=tf_efficientnetv2_m.in21k_ft_in1k
L=tf_efficientnetv2_l.in21k_ft_in1k
run() { name=$1; shift; echo "===== SWEEP: $name ($(date -u +%H:%M)) ====="; \
  python3 -u scripts/30_train2.py --loto-type "MOZAMBIQUE/DL" --loss bce --epochs 1 \
    --workers 10 --save-name "sw_$name" "$@"; }

run lr3e5   --backbone $M --img-size 768 --batch-size 24 --lr 3e-5  --aug-strength heavy
run lr3e4   --backbone $M --img-size 768 --batch-size 24 --lr 3e-4  --aug-strength heavy
run res640  --backbone $M --img-size 640 --batch-size 32 --lr 1e-4  --aug-strength heavy
run res896  --backbone $M --img-size 896 --batch-size 16 --lr 1e-4  --aug-strength heavy
run effL768 --backbone $L --img-size 768 --batch-size 12 --lr 1e-4  --aug-strength heavy
run defaug  --backbone $M --img-size 768 --batch-size 24 --lr 1e-4  --aug-strength default

echo ""
echo "########## RESUMEN SWEEP (anclas: 0.152/0.225) ##########"
grep -h "PROXY" /root/sweep1.log || true
echo "SWEEP1_DONE_MARKER"
