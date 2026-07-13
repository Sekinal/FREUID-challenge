#!/bin/bash
# TRIO 1024: validar el hallazgo OOD del campeon en Benin, capturas (canary) e hibrido.
set -e
cd /root/freuid
unset FREUID_IMG_CACHE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== A: LOTO BENIN @1024 no-capture-aug 3ep ====="
python3 -u scripts/24_train_fusion.py --loto-type "BENIN/DL" --no-fusion --loss bce \
  --img-size 1024 --no-capture-aug --batch-size 12 --workers 10 --epochs 3 --seed 4

echo "===== B: LOTO MAURITIUS @1024 no-capture-aug 3ep ====="
python3 -u scripts/24_train_fusion.py --loto-type "MAURITIUS/ID" --no-fusion --loss bce \
  --img-size 1024 --no-capture-aug --batch-size 12 --workers 10 --epochs 3 --seed 4

echo "===== B2: CANARY del modelo Mauritius-LOTO @1024 ====="
python3 scripts/eval_canary.py --checkpoint runs/fusion_nofusion_loto_MAURITIUS-ID_noaug_s4.pt --batch-size 4 || true

echo "===== C: HIBRIDO LOTO MOZ @1024 heavy (capture-aug ON) 3ep ====="
python3 -u scripts/30_train2.py --loto-type "MOZAMBIQUE/DL" \
  --backbone tf_efficientnetv2_m.in21k_ft_in1k --img-size 1024 --batch-size 12 --workers 10 \
  --loss bce --aug-strength heavy --epochs 3 --save-name hib_mozLOTO_1024

echo ""
echo "########## RESUMEN TRIO ##########"
grep -hE "test:|PROXY|CANARY|separation|AUC|TRIPWIRE" /root/trio1024.log | tail -20 || true
echo "TRIO1024_DONE_MARKER"
