#!/bin/bash
# "FOR THE PRIVATE TEST" strategy (host: OOD to unseen countries + captured robustness,
# NO memorizing digital artifacts). Clean models = NO pseudo-labels. Validate with LOCO.
cd /root/freuid
PY=python3
exec > >(tee -a robust.log) 2>&1
mkdir -p logs submissions
echo "=================================================================="
echo "[$(date)] ROBUST START"
MAE=runs/mae_cnxv2L.pt; LOTO="MOZAMBIQUE/DL"; SCAN="data/aux/idnet2025_scanned"
run(){ local n=$1; shift; echo "[$(date)] >>> $n"
  $PY scripts/30_train2.py "$@" --save-name "$n" > "logs/rb_$n.log" 2>&1 \
    && { echo "[$(date)] $n OK"; grep -E "PROXY|val FREUID" "logs/rb_$n.log"|tail -2; } \
    || { echo "[$(date)] $n FAILED"; tail -6 "logs/rb_$n.log"; }; }

# PHASE 1: LOCO (hold out MOZAMBIQUE) -> does robustness improve OOD generalization?
run loto_robust --loto-type "$LOTO" --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from "$MAE" \
    --img-size 384 --batch-size 32 --loss bce --aug-strength heavy \
    --aux --aux-roots "$SCAN" --max-aux 30000 --epochs 3
run loto_baseline --loto-type "$LOTO" --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from "$MAE" \
    --img-size 384 --batch-size 32 --loss bce --aug-strength default --max-aux 0 --epochs 3

# PHASE 2: clean robust model on ALL 5 (no pseudo) @384, then adapt to @512
run fusion_nofusion_all_robust384 --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k --init-from "$MAE" \
    --img-size 384 --batch-size 32 --loss bce --aug-strength heavy \
    --aux --aux-roots "$SCAN" --max-aux 30000 --epochs 3
[ -f runs/fusion_nofusion_all_robust384.pt ] && \
run fusion_nofusion_all_robust512 --train-all --backbone convnextv2_large.fcmae_ft_in22k_in1k \
    --init-from runs/fusion_nofusion_all_robust384.pt \
    --img-size 512 --batch-size 16 --loss bce --lr 3e-5 --aug-strength heavy \
    --aux --aux-roots "$SCAN" --max-aux 30000 --epochs 2

# CSVs (single-scale, small batch -> no OOM)
for m in fusion_nofusion_all_robust384 fusion_nofusion_all_robust512; do
  [ -f runs/$m.pt ] && $PY scripts/31_polish_submit.py --checkpoints runs/$m.pt --scales 1.0 --batch-size 8 \
      --out submissions/sub_${m#fusion_nofusion_all_}.csv > logs/rb_csv_$m.log 2>&1 && echo "  CSV $m OK"
done
echo "=================================================================="
echo "[$(date)] ROBUST DONE."
echo "LOCO (cross-country FREUID, MENOR=mejor generalizacion OOD):"
grep -h "CROSS-COUNTRY-FREUID" logs/rb_loto_*.log | tail -4
ls -la submissions/sub_robust*.csv 2>/dev/null
echo "=================================================================="
