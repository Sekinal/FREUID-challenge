#!/bin/bash
cd /root/freuid
echo "[$(date)] INICIO entrenamiento ConvNeXt-Base + IDNet"
.venv/bin/python scripts/13_train_with_aux.py \
  --model convnext_base.fb_in22k_ft_in1k \
  --epochs 3 --batch-size 128 --img-size 384 --lr 5e-5 \
  --num-workers 16 --max-aux 30000 \
  --aux-dir data/aux/idnet2025 \
  --run-dir runs/convnext_aux
echo "[$(date)] FIN entrenamiento (exit $?)"
