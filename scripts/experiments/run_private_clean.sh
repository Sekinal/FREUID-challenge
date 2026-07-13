#!/bin/bash
# Cadena "privado limpio": receta benchmax (EffNetV2-M @768) pero SIN pseudo-labels,
# CON heavy aug (robustez captura) y extras limpios (synth GenAI + FantasyID).
# 3 fases: LOTO Mozambique (numero OOD real) -> LOTO Mauritius (+canary) -> train-all (candidato slot 2).
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "[$(date -u +%H:%M:%S)] esperando a que termine slot2 (pid 177904)..."
while kill -0 177904 2>/dev/null; do sleep 60; done
echo "[$(date -u +%H:%M:%S)] GPU libre, arrancamos"

echo "===== FASE A: LOTO MOZAMBIQUE/DL @768 heavy, extras limpios ====="
python3 -u scripts/30_train2.py --loto-type "MOZAMBIQUE/DL" \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy --extra-csv artifacts/extra_clean.csv \
  --epochs 3 --save-name rc_mozLOTO_768

echo "===== FASE B: LOTO MAURITIUS/ID @768 heavy, extras limpios ====="
python3 -u scripts/30_train2.py --loto-type "MAURITIUS/ID" \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy --extra-csv artifacts/extra_clean.csv \
  --epochs 3 --save-name rc_maurLOTO_768

echo "===== CANARY sobre Mauritius-LOTO (valido: no vio esas imgs) ====="
python3 scripts/eval_canary.py --checkpoint runs/rc_maurLOTO_768.pt || true

echo "===== FASE C: TRAIN-ALL @768 heavy, extras limpios (candidato slot 2) ====="
python3 -u scripts/30_train2.py --train-all \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 768 --batch-size 24 --workers 10 \
  --loss bce --aug-strength heavy --extra-csv artifacts/extra_clean.csv \
  --epochs 3 --save-name rc_all_768

echo "===== PREDICT (TTA single-scale, NO submit) ====="
python3 -u scripts/31_polish_submit.py --checkpoints runs/rc_all_768.pt \
  --scales 1.0 --batch-size 8 --out submissions/sub_rc_all768.csv

echo ""
echo "########## RESUMEN ##########"
grep -h "CROSS-COUNTRY-FREUID" /root/private_clean.log | tail -5 || true
echo "PRIVATE_CLEAN_DONE_MARKER"
