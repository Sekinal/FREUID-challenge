#!/bin/bash
# Privado v2: base ganadora OOD (ConvNeXtV2-L + MAE @384) + heavy aug + extras limpios.
# Solo LOTOs (validacion); el train-all se decide al ver estos numeros (incl. mejor epoca).
set -e
cd /root/freuid
export FREUID_IMG_CACHE=/root/freuid/data/cache896 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

echo "===== FASE A: LOTO MOZAMBIQUE/DL cnxV2-L+MAE @384 heavy + extras limpios ====="
python3 -u scripts/30_train2.py --loto-type "MOZAMBIQUE/DL" \
  --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from runs/mae_cnxv2L.pt \
  --img-size 384 --batch-size 32 --workers 10 \
  --loss bce --aug-strength heavy --extra-csv artifacts/extra_clean.csv \
  --epochs 3 --save-name rc2_mozLOTO_cnx384

echo "===== FASE B: LOTO MAURITIUS/ID idem ====="
python3 -u scripts/30_train2.py --loto-type "MAURITIUS/ID" \
  --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from runs/mae_cnxv2L.pt \
  --img-size 384 --batch-size 32 --workers 10 \
  --loss bce --aug-strength heavy --extra-csv artifacts/extra_clean.csv \
  --epochs 3 --save-name rc2_maurLOTO_cnx384

echo "===== CANARY sobre Mauritius-LOTO ====="
python3 scripts/eval_canary.py --checkpoint runs/rc2_maurLOTO_cnx384.pt || true

echo ""
echo "########## RESUMEN ##########"
grep -hE "PROXY|LOTO-test" /root/private2.log | tail -10 || true
echo "PRIVATE2_DONE_MARKER"
