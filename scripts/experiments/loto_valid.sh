#!/bin/bash
cd /root/freuid
echo "[$(date)] LOTO valido robusto (warm-start loto_baseline, capturados+heavy)"
python3 scripts/30_train2.py --loto-type "MOZAMBIQUE/DL" --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --init-from runs/loto_baseline.pt --img-size 384 --batch-size 32 --loss bce \
    --aux --aux-roots data/aux/idnet2025_scanned --max-aux 30000 --aug-strength heavy --epochs 2 \
    --save-name loto_robust_valid > logs/loto_robust_valid.log 2>&1
echo "[$(date)] DONE"
grep "CROSS-COUNTRY-FREUID" logs/loto_robust_valid.log | tail -1
