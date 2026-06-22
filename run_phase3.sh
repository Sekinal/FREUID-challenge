#!/bin/bash
cd /root/freuid
PY=.venv/bin/python
exec > >(tee -a phase3.log) 2>&1
echo "=================================================================="
echo "[$(date)] PHASE3: submission del mejor (e5) + final all-5 EffNetV2-M"
mkdir -p logs submissions
# 1) submission del mejor modelo 3-country (e5) con score local honesto
echo "[$(date)] >>> submission e5 (EffNetV2-M 3-country)"
$PY scripts/22_ensemble_submission.py --checkpoints runs/e5_effnetv2m_aug_focal/best.pt \
  --out submissions/sub_e5_effnetv2_3country.csv > logs/sub_e5.log 2>&1 \
  && { echo "[$(date)] sub_e5 OK"; grep -E "LOCAL TEST|wrote" logs/sub_e5.log; } \
  || { echo "[$(date)] sub_e5 FAILED"; tail -8 logs/sub_e5.log; }
# 2) final sobre los 5 paises con la receta GANADORA (EffNetV2-M + aug + focal)
echo "[$(date)] >>> f3_effnetv2_all (train-all)"
$PY scripts/13_train_with_aux.py --model tf_efficientnetv2_m.in21k_ft_in1k --img-size 384 \
  --aug domain --loss focal --aux-dirs data/aux/idnet2025 --max-aux 40000 \
  --epochs 4 --batch-size 96 --lr 1e-4 --train-all --run-dir runs/f3_effnetv2_all \
  > logs/f3_effnetv2_all.log 2>&1 \
  && { echo "[$(date)] f3 OK"; tr '\r' '\n' < logs/f3_effnetv2_all.log | grep "val  FREUID" | tail -1; } \
  || { echo "[$(date)] f3 FAILED"; tail -8 logs/f3_effnetv2_all.log; }
# 3) submission del final all-5
echo "[$(date)] >>> submission f3 (all-5)"
$PY scripts/22_ensemble_submission.py --checkpoints runs/f3_effnetv2_all/best.pt --no-score \
  --out submissions/sub_f3_effnetv2_all.csv > logs/sub_f3.log 2>&1 \
  && echo "[$(date)] sub_f3 OK" || echo "[$(date)] sub_f3 FAILED"
echo "[$(date)] PHASE3 COMPLETO"; ls -la submissions/sub_e5*.csv submissions/sub_f3*.csv 2>/dev/null
echo "=================================================================="
