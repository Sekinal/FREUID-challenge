#!/bin/bash
# DIA 2a: composicion de ensemble (inferencia pura, gratis). Referencia: duo semillas identicas 0.1307.
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== (a) 3x mismo config: 768 anchors + lr3e5 ====="
python3 -u scripts/eval_loto_ens.py --loto-type MOZAMBIQUE/DL \
  --checkpoints runs/es_mozLOTO_768.pt,runs/es_mozLOTO_768_ep1_v2.pt,runs/sw_lr3e5.pt

echo "===== (b) MULTI-RES: 640 + 768 + 896 ====="
python3 -u scripts/eval_loto_ens.py --loto-type MOZAMBIQUE/DL \
  --checkpoints runs/sw_res640.pt,runs/es_mozLOTO_768.pt,runs/sw_res896.pt

echo "===== (c) top-3 individuales: 640 + lr3e5 + 768-ancla ====="
python3 -u scripts/eval_loto_ens.py --loto-type MOZAMBIQUE/DL \
  --checkpoints runs/sw_res640.pt,runs/sw_lr3e5.pt,runs/es_mozLOTO_768.pt

echo "===== (d) los 6 decentes ====="
python3 -u scripts/eval_loto_ens.py --loto-type MOZAMBIQUE/DL \
  --checkpoints runs/sw_res640.pt,runs/sw_lr3e5.pt,runs/es_mozLOTO_768.pt,runs/es_mozLOTO_768_ep1_v2.pt,runs/sw_res896.pt,runs/sw_defaug.pt

echo "ENSCOMP_DONE_MARKER"
