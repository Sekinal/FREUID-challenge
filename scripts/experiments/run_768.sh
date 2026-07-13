#!/bin/bash
# Push resolution further: 768px, warm-start from res512 (0.15336).
cd /root/freuid
PY=python3
exec > >(tee -a r768.log) 2>&1
echo "[$(date)] 768 START"
$PY scripts/30_train2.py --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --init-from runs/fusion_nofusion_all_res512.pt --img-size 768 --batch-size 8 --loss bce --lr 3e-5 \
    --aux --max-aux 30000 --extra-csv artifacts/pseudo_ens.csv --epochs 2 \
    --save-name fusion_nofusion_all_res768 > logs/r768_train.log 2>&1 \
  && { echo "[$(date)] 768 OK"; grep "val FREUID" logs/r768_train.log | tail -1; } \
  || { echo "[$(date)] 768 FAILED"; tail -6 logs/r768_train.log; }
R=runs/fusion_nofusion_all_res768.pt
[ -f "$R" ] && $PY scripts/31_polish_submit.py --checkpoints "$R" --out submissions/sub_res768.csv > logs/r768_c1.log 2>&1 && echo "sub_res768 OK"
[ -f "$R" ] && $PY scripts/31_polish_submit.py --checkpoints "$R",runs/fusion_nofusion_all_res512.pt --rank-avg --out submissions/sub_res768_512.csv > logs/r768_c2.log 2>&1 && echo "sub_res768_512 OK"
echo "[$(date)] 768 DONE"; ls -la submissions/sub_res768*.csv 2>/dev/null
