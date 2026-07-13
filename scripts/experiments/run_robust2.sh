#!/bin/bash
# Robust v2 (FIX): heavy aug needs a TRAINED head (warm-start), not the raw MAE backbone.
cd /root/freuid
PY=python3
exec > >(tee -a robust2.log) 2>&1
mkdir -p logs submissions
echo "[$(date)] ROBUST2 START"
MAE=runs/mae_cnxv2L.pt; RES512=runs/fusion_nofusion_all_res512.pt
MAEFT=runs/fusion_nofusion_all_maeFT.pt; LOTO="MOZAMBIQUE/DL"; SCAN="data/aux/idnet2025_scanned"
run(){ local n=$1; shift; echo "[$(date)] >>> $n"
  $PY scripts/30_train2.py "$@" --save-name "$n" > "logs/r2_$n.log" 2>&1 \
    && { echo "[$(date)] $n OK"; grep -E "PROXY|val FREUID" "logs/r2_$n.log"|tail -2; } \
    || { echo "[$(date)] $n FAILED"; tail -6 "logs/r2_$n.log"; }; }

# LOCO: does CAPTURED data help OOD? (default aug so fresh head converges; clean test)
run loto_scanned --loto-type "$LOTO" --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from "$MAE" \
    --img-size 384 --batch-size 32 --loss bce --aug-strength default \
    --aux --aux-roots "$SCAN" --max-aux 30000 --epochs 3

# FINAL robust models: warm-start from TRAINED models (head survives heavy aug) + captured + heavy aug, NO pseudo
run all_robust384 --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from "$MAEFT" \
    --img-size 384 --batch-size 32 --loss bce --lr 3e-5 --aug-strength heavy \
    --aux --aux-roots "$SCAN" --max-aux 30000 --epochs 2
run all_robust512 --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from "$RES512" \
    --img-size 512 --batch-size 16 --loss bce --lr 3e-5 --aug-strength heavy \
    --aux --aux-roots "$SCAN" --max-aux 30000 --epochs 2

for m in all_robust384 all_robust512; do
  [ -f runs/$m.pt ] && $PY scripts/31_polish_submit.py --checkpoints runs/$m.pt --scales 1.0 --batch-size 8 \
      --out submissions/sub_$m.csv > logs/r2_csv_$m.log 2>&1 && echo "  CSV $m OK"
done
echo "[$(date)] ROBUST2 DONE"
echo "LOCO baseline(viejo)=0.4023 | loto_scanned(capturados):"; grep -h CROSS-COUNTRY logs/r2_loto_scanned.log
ls -la submissions/sub_all_robust*.csv 2>/dev/null
