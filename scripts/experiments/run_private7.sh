#!/bin/bash
# #4 Validacion 3er pais (Benin) + #3 diversidad de backbone en el ensemble (medida en LOTO).
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "[$(date -u +%H:%M:%S)] esperando a que termine private6..."
while tmux has-session -t private6 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] GPU libre"

echo "===== #4: LOTO BENIN/DL effM@768 heavy 1ep (3er pais) ====="
python3 -u scripts/30_train2.py --loto-type "BENIN/DL" \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy --epochs 1 --save-name es_benLOTO_768_ep1

echo "===== #3: LOTO MOZAMBIQUE cnxV2-L+MAE @384 heavy 1ep (diversidad) ====="
python3 -u scripts/30_train2.py --loto-type "MOZAMBIQUE/DL" \
  --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from runs/mae_cnxv2L.pt \
  --img-size 384 --batch-size 32 --workers 10 \
  --loss bce --aug-strength heavy --epochs 1 --save-name cnx_mozLOTO_384_ep1

echo "===== MEDICION: ensemble 2xeffM vs 2xeffM+cnx en Mozambique ====="
echo "--- solo cnx + 2 effM:"
python3 -u scripts/eval_loto_ens.py \
  --checkpoints runs/es_mozLOTO_768.pt,runs/es_mozLOTO_768_ep1_v2.pt,runs/cnx_mozLOTO_384_ep1.pt \
  --loto-type MOZAMBIQUE/DL

echo "PRIVATE7_DONE_MARKER"
